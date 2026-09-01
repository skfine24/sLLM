"""Device-resident GPU decode engine (standard decoder family: Llama/Qwen2/
Mistral-class GQA dense models): weights + KV live on the GPU, activations
stay on-device for the whole step, and the loop synchronises exactly ONCE per
step.

This removes the per-op H2D/D2H + cudaMalloc/free + sync of the transfer-era
drivers (kernels/standard_decode.py), which kept every weight on the host and
shipped it to the GPU at each call.

General-purpose design (architecture/dtype agnostic serving):

* dtype: `fp32` or `bf16` model dtype (weights, embedding, KV, GEMM operands
  and GEMM outputs feeding the next GEMM). The residual stream, softmax and
  logits stay fp32 — the split mainstream engines use. bf16 tables are
  RNE-rounded from the decoded checkpoint floats (bit-exact for BF16
  checkpoints).
* fusion: each layer runs in 11 device ops: add_rms(qkv-in) -> gemm(qkv) ->
  rope+bias -> kv_write x2 -> attention -> gemm(o) -> add_rms(mlp-in) ->
  gemm(gate|up) -> silu*mul -> gemm(down). QKV and gate/up weights are
  concatenated at upload time; biases fold into rope/kv_write; residual adds
  fold into the neighbouring RMSNorm.
* partial RoPE: rotary width comes from the recipe (`rotary_dim()`), so
  full- and partial-rotary checkpoints both work.

The host `IncrementalCache` stays authoritative: the rows the GPU appended
are copied back each step (one small D2H) and concatenated into it, so the
numpy oracle/fallback keeps working transparently if the GPU path is later
disabled mid-generation. KV capacity doubles on demand (relayout kernel);
weights are uploaded once per model by `DeviceWeightTable`.
"""

from __future__ import annotations

import numpy as np

from kernels import _sllm_cuda as ck

_DTYPES = {"fp32": ck.T_F32, "bf16": ck.T_BF16}


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


class DeviceWeightTable:
    """All standard-path weights uploaded to the device ONCE (read-only).

    Per layer, q/k/v projections are stored as ONE concatenated matrix
    (`<p>.qkv_w`) and gate/up as one (`<p>.gu_w`), so each block needs a
    single GEMM. Norm weights and biases stay fp32; projection weights,
    o/down and the embedding/head honour `dtype` (fp32|bf16).

    Raises RuntimeError when no GPU/.so is present or the free device memory
    cannot hold the table (+ slack), so callers can fall back transparently.
    """

    def __init__(self, weights: dict, recipe, dtype: str = "fp32",
                 slack_bytes: int = 64 << 20):
        if ck.device_count() < 1:
            raise RuntimeError("no CUDA device")
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported device dtype {dtype!r}")
        self.dtype = dtype
        self.t = _DTYPES[dtype]
        self.elem = 2 if dtype == "bf16" else 4
        prefix = recipe.text_prefix
        self._bufs: dict[str, ck.DeviceBuffer] = {}

        # -- plan FIRST ------------------------------------------------------
        # The free-memory guard must run BEFORE any cudaMalloc: an
        # upload-then-check order can already have OOMed / hung the node.
        plan: list[tuple[str, list, bool]] = []  # (key, source arrays, fp32?)

        def add(key: str, *arrs, is32: bool = False) -> None:
            plan.append((key, list(arrs), is32))

        tied = recipe.tie_word_embeddings
        add("__embed__", weights[f"{prefix}.embed_tokens.weight"])
        if not tied:
            add("__head__", weights["lm_head.weight"])
        add("__final_norm__", weights[f"{prefix}.norm.weight"], is32=True)
        for i in range(recipe.num_layers):
            p = f"{prefix}.layers.{i}"
            add(f"{p}.qkv_w",
                weights[f"{p}.self_attn.q_proj.weight"],
                weights[f"{p}.self_attn.k_proj.weight"],
                weights[f"{p}.self_attn.v_proj.weight"])
            for leaf in ("q_proj", "k_proj", "v_proj"):
                b = weights.get(f"{p}.self_attn.{leaf}.bias")
                if b is not None:
                    add(f"{p}.self_attn.{leaf}.bias", b, is32=True)
            add(f"{p}.o_proj", weights[f"{p}.self_attn.o_proj.weight"])
            add(f"{p}.gu_w",
                weights[f"{p}.mlp.gate_proj.weight"],
                weights[f"{p}.mlp.up_proj.weight"])
            add(f"{p}.down_proj", weights[f"{p}.mlp.down_proj.weight"])
            add(f"{p}.in_ln", weights[f"{p}.input_layernorm.weight"], is32=True)
            add(f"{p}.post_ln", weights[f"{p}.post_attention_layernorm.weight"],
                is32=True)

        sizes = sum(sum(a.size for a in arrs) * (4 if is32 else self.elem)
                    for _k, arrs, is32 in plan)
        free = ck.mem_free_bytes()
        if free < 0:
            # GB10 over-subscription can hang the node until power-off: an
            # unknown free-memory figure is treated as unsafe, not as "skip
            # the check".
            raise RuntimeError(
                "cannot query free device memory (cudaMemGetInfo failed); "
                "refusing the device-resident upload (GB10 guard)")
        if free < sizes + slack_bytes:
            raise RuntimeError(
                f"insufficient free device memory for resident weights: "
                f"{free} B free < {sizes} B weights + {slack_bytes} B slack")

        # -- upload ----------------------------------------------------------
        for key, arrs, is32 in plan:
            a = _f32(arrs[0] if len(arrs) == 1
                     else np.concatenate([_f32(x) for x in arrs], axis=0))
            if is32 or self.elem == 4:
                self._bufs[key] = ck.to_device(a)
            else:
                buf = ck.alloc_n(a.nbytes // 2)
                buf.upload_raw(ck.to_bf16(a))
                self._bufs[key] = buf

        self.embed = self._bufs["__embed__"]
        self.head = self.embed if tied else self._bufs["__head__"]
        self.final_norm = self._bufs["__final_norm__"]
        self.total_bytes = sizes

    def get(self, key: str):
        """Device buffer for a table key, or None when absent (no bias)."""
        return self._bufs.get(key)

    def free(self) -> None:
        for buf in self._bufs.values():
            buf.free()
        self._bufs.clear()
        self.embed = self.head = self.final_norm = None


class DeviceDecodeState:
    """Per-sequence device-resident decode state over a shared weight table.

    Owns the on-device KV (capacity-doubling, table dtype), the per-step
    activation scratch (allocated once) and the end-of-step host-mirror
    staging. The paired numpy `IncrementalCache` (host K/V + n_ctx) is kept
    authoritative: every successful step copies the appended rows back into
    it, so a failed/disabled GPU step never corrupts the numpy fallback.
    """

    def __init__(self, table: DeviceWeightTable, cache, recipe):
        if cache.kernel != "standard_gqa":
            raise ValueError("device-resident decode supports standard_gqa only")
        self.table = table
        self.cache = cache
        self.recipe = recipe
        self.t = table.t
        self.elem = table.elem
        prefix = recipe.text_prefix
        L = self.L = recipe.num_layers
        nh, kvh, hd = cache.num_heads, cache.kv_heads, cache.head_dim
        self.nh, self.kvh, self.hd = nh, kvh, hd
        self.NQ, self.NK = nh * hd, kvh * hd
        self.H = table.final_norm.nbytes // 4
        self.V = table.embed.nbytes // self.elem // self.H
        self.rot = recipe.rotary_dim()
        if self.rot <= 0 or self.rot % 2 or self.rot > hd:
            raise ValueError(f"invalid rotary_dim {self.rot} for head_dim {hd}")
        self.eps, self.theta = cache.eps, cache.theta
        self.scale = float(hd) ** -0.5

        # Geometry sanity against the concatenated tables.
        qkv_rows = table.get(f"{prefix}.layers.0.qkv_w").nbytes // self.elem // self.H
        if qkv_rows != self.NQ + 2 * self.NK:
            raise ValueError(
                f"qkv width {qkv_rows} != heads*head_dim {self.NQ + 2 * self.NK}")
        self.I = (table.get(f"{prefix}.layers.0.gu_w").nbytes
                  // self.elem // 2 // self.H)

        S = cache.n_ctx
        self.cap = max(int(S), 1)
        self._k: dict[int, ck.DeviceBuffer] = {}
        self._v: dict[int, ck.DeviceBuffer] = {}
        for i in range(L):
            for store, arr in ((self._k, cache.k[i]), (self._v, cache.v[i])):
                a = _f32(arr)
                if self.elem == 4:
                    store[i] = ck.to_device(a)
                else:
                    buf = ck.alloc_n(a.nbytes // 2)
                    buf.upload_raw(ck.to_bf16(a))
                    store[i] = buf

        self._scores = ck.alloc(self.nh * self.cap)
        a32, at = ck.alloc, lambda n: ck.alloc_n(n * self.elem)
        # fp32 residual stream + fp32 GEMM outputs
        self._hx = a32(self.H)
        self._qkv = a32(self.NQ + 2 * self.NK)
        self._gu = a32(2 * self.I)
        self._o_out, self._d_out = a32(self.H), a32(self.H)
        # dtype-qualified GEMM operands (norm/attention/silu outputs)
        self._hn = at(self.H)
        self._attn = at(self.NQ)
        self._down_in = at(self.I)
        # slices into the fused-GEMM outputs (non-owning views)
        f4 = 4  # fp32 element
        self._qv = ck.DeviceView(self._qkv, 0, self.NQ * f4)
        self._kv_ = ck.DeviceView(self._qkv, self.NQ * f4, self.NK * f4)
        self._vv = ck.DeviceView(self._qkv, (self.NQ + self.NK) * f4, self.NK * f4)
        self._gv = ck.DeviceView(self._gu, 0, self.I * f4)
        self._uv = ck.DeviceView(self._gu, self.I * f4, self.I * f4)
        self._cs = a32(2 * self.rot)
        self._logits = a32(self.V)
        self._stage = ck.alloc(2 * L * self.NK)

    # -- KV capacity ----------------------------------------------------------

    def _grow(self, cap_new: int) -> None:
        nb = self.kvh * cap_new * self.hd * self.elem
        blk_words = self.hd * self.elem // 4
        if self.hd * self.elem % 4:
            raise ValueError("head row not word-aligned")
        # exception-safe: allocate ALL new buffers first (a failure leaves the
        # old KV + self.cap untouched -> no mixed-stride state), relayout,
        # and only then free the old buffers.
        fresh: list[ck.DeviceBuffer] = []
        try:
            for _store in (self._k, self._v):
                for _i in _store:
                    fresh.append(ck.alloc_n(nb))
        except BaseException:
            for buf in fresh:
                try:
                    buf.free()
                except Exception:
                    pass
            raise
        new_k: dict[int, ck.DeviceBuffer] = {}
        new_v: dict[int, ck.DeviceBuffer] = {}
        it = iter(fresh)
        try:
            for store, new_store in ((self._k, new_k), (self._v, new_v)):
                for i, buf in store.items():
                    new = next(it)
                    ck.kv_relayout_w(new, buf, self.kvh, self.cap, cap_new,
                                     blk_words)
                    new_store[i] = new
        except BaseException:
            for buf in fresh:
                try:
                    buf.free()
                except Exception:
                    pass
            raise
        for store in (self._k, self._v):
            for buf in store.values():
                buf.free()
        self._k, self._v = new_k, new_v
        self.cap = cap_new
        self._scores.free()
        self._scores = ck.alloc(self.nh * cap_new)

    # -- one decode step --------------------------------------------------------

    def step(self, last_id: int) -> np.ndarray:
        from ref import incremental as inc  # local import to avoid a cycle

        cache, table, p0 = self.cache, self.table, self.recipe.text_prefix
        t = self.t
        last_id = int(last_id)
        if not 0 <= last_id < self.V:
            raise ValueError(f"token id {last_id} outside vocab [0, {self.V})")
        pos = cache.n_ctx
        max_pos = int(self.recipe.max_position_embeddings)
        if pos >= max_pos:
            raise RuntimeError(
                f"decode position {pos} reached max_position_embeddings "
                f"({max_pos})")
        if pos >= self.cap:
            self._grow(min(max(2 * self.cap, pos + 1), max_pos))

        cos, sin = inc._pos_cos_sin(self.theta, self.rot, pos)
        self._cs.upload(np.concatenate([cos, sin]))

        ck.gather_row_t(table.embed, last_id, self._hx, self.H, t)
        hx, pend = self._hx, None  # pend: MLP delta folded into next add_rms
        for i in range(self.L):
            p = f"{p0}.layers.{i}"
            # attention block: residual stream hx += (previous MLP delta)
            ck.add_rms(hx, pend, hx, self._hn, table.get(f"{p}.in_ln"),
                       1, self.H, self.eps, t)
            ck.gemm_ex(self._hn, table.get(f"{p}.qkv_w"), self._qkv,
                       1, self.NQ + 2 * self.NK, self.H, t)
            ck.rope_bias(self._qv, self._kv_,
                         table.get(f"{p}.self_attn.q_proj.bias"),
                         table.get(f"{p}.self_attn.k_proj.bias"),
                         self._cs, self.nh, self.kvh, self.hd, self.rot)
            # K slice already carries its bias (added in-place by rope_bias);
            # write it verbatim (bias=None) to avoid double-adding.
            ck.kv_write_t(self._k[i], self.cap, pos, self._kv_, None,
                          self.kvh, self.hd, self._stage, i * self.NK, t)
            ck.kv_write_t(self._v[i], self.cap, pos, self._vv,
                          table.get(f"{p}.self_attn.v_proj.bias"),
                          self.kvh, self.hd, self._stage,
                          (self.L + i) * self.NK, t)
            ck.attention_decode_t(self._qv, self._k[i], self._v[i],
                                  self._scores, self._attn,
                                  self.nh, self.kvh, pos + 1, self.cap,
                                  self.hd, self.scale, t)
            ck.gemm_ex(self._attn, table.get(f"{p}.o_proj"), self._o_out,
                       1, self.H, self.NQ, t)
            # mlp block: hx += o_out (residual), normalized for gate/up
            ck.add_rms(hx, self._o_out, hx, self._hn,
                       table.get(f"{p}.post_ln"), 1, self.H, self.eps, t)
            ck.gemm_ex(self._hn, table.get(f"{p}.gu_w"), self._gu,
                       1, 2 * self.I, self.H, t)
            ck.silu_mul(self._gv, self._uv, self._down_in, t)
            ck.gemm_ex(self._down_in, table.get(f"{p}.down_proj"),
                       self._d_out, 1, self.H, self.I, t)
            pend = self._d_out

        ck.add_rms(hx, pend, hx, self._hn, table.final_norm, 1, self.H,
                   self.eps, t)
        ck.gemm_ex(self._hn, table.head, self._logits, 1, self.V, self.H, t)

        # One sync + two small transfers; then the host cache mirror.
        ck.sync()
        logits = self._logits.copy_host()
        stage = self._stage.copy_host()
        for i in range(self.L):
            k_row = stage[i * self.NK:(i + 1) * self.NK].reshape(self.kvh, 1, self.hd)
            v_row = stage[(self.L + i) * self.NK:(self.L + i + 1) * self.NK] \
                .reshape(self.kvh, 1, self.hd)
            cache.k[i] = np.concatenate([cache.k[i], k_row], axis=1)
            cache.v[i] = np.concatenate([cache.v[i], v_row], axis=1)
        cache.n_ctx += 1
        return logits

    def free(self) -> None:
        for store in (self._k, self._v):
            for buf in store.values():
                buf.free()
            store.clear()
        for name in ("_scores", "_hx", "_qkv", "_gu", "_o_out",
                     "_d_out", "_hn", "_attn", "_down_in", "_cs", "_logits",
                     "_stage"):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.free()
