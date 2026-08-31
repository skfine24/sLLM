"""Generation loop: token ids -> [chat template] -> forward -> sample -> stop."""

from __future__ import annotations

import os

import numpy as np

from ref import pipeline as _pipeline
from ref import incremental as _incremental
from runtime import sampler as _sampler

QWEN4_EXP_ARCH = "qwen4_exp"


class ReferenceModel:
    """NumPy reference model backing the dev serving stub (and cluster fallback).

    `logits(ids)` runs the whole text forward (prefill path) on the provided
    weight dict. No incremental KV cache: reference-only. Real kernels will
    replace this behind the same interface in P1/P2.

    Supported kernels additionally expose `prefill`/`decode_step` so the
    generation loop keeps its KV/recurrent state across steps (incremental
    decode) instead of recomputing the full context every step; `logits()`
    stays as the recompute oracle.
    """

    def __init__(self, recipe, weights: dict, use_gpu: bool | None = None,
                 gpu_dtype: str | None = None, q4_cfg=None):
        self.recipe = recipe
        self.weights = weights
        # qwen4_exp pipeline config (ref/qwen4_exp_pipeline.Qwen4ExpCfg). When
        # the arch is qwen4_exp and no cfg is given, it is derived from the
        # recipe lazily (Qwen4ExpCfg.from_recipe).
        self.q4_cfg = q4_cfg
        # GPU decode toggle (per step): enables the fused kernel decode drivers
        # for standard/hybrid when the built .so + a CUDA device are present.
        # Default from env SLLM_USE_GPU=1; any runtime failure falls back to
        # the numpy incremental path transparently.
        self.use_gpu = bool(use_gpu if use_gpu is not None
                            else os.environ.get("SLLM_USE_GPU", "0") == "1")
        # Device-resident decode dtype (weights/KV): fp32 | bf16
        # (env SLLM_GPU_DTYPE; fp32 default until wider validation).
        self.gpu_dtype = (gpu_dtype or os.environ.get("SLLM_GPU_DTYPE", "fp32")).lower()
        self._gpu_ok = None
        # Device-resident decode (standard path): weights uploaded once, KV
        # lives on the GPU, one sync per step. Built lazily; any failure
        # disables it once and the slower transfer/numpy paths take over.
        self._dev_table = None
        self._resident_off = False

    @property
    def _is_q4(self) -> bool:
        return self.recipe.arch == QWEN4_EXP_ARCH

    def _q4cfg(self):
        if self.q4_cfg is None:
            from ref.qwen4_exp_pipeline import Qwen4ExpCfg
            self.q4_cfg = Qwen4ExpCfg.from_recipe(self.recipe)
        return self.q4_cfg

    def _gpu_available(self) -> bool:
        if self._gpu_ok is None:
            try:
                from kernels import _sllm_cuda as ck
                self._gpu_ok = ck.device_count() >= 1
            except Exception:
                self._gpu_ok = False
        return self._gpu_ok

    def _resident_step(self, cache, last_id: int):
        """Device-resident decode step; raises on any failure (caller falls
        back). The host `cache` is only mutated at the END of a successful
        step, so a failed step leaves it exactly as the numpy path expects."""
        import warnings
        from kernels.device_decode import DeviceDecodeState, DeviceWeightTable

        if self.recipe.full_attention.kernel != "standard_gqa":
            raise RuntimeError("resident decode is standard_gqa-only")
        if os.environ.get("SLLM_GPU_RESIDENT", "1") != "1":
            raise RuntimeError("disabled via SLLM_GPU_RESIDENT=0")
        if self._dev_table is None:
            self._dev_table = DeviceWeightTable(self.weights, self.recipe,
                                                dtype=self.gpu_dtype)
        state = getattr(cache, "_resident", None)
        if state is None or state.table is not self._dev_table:
            if state is not None:
                state.free()
            state = DeviceDecodeState(self._dev_table, cache, self.recipe)
            cache._resident = state
        try:
            return state.step(last_id)
        except Exception:
            self._resident_off = True
            warnings.warn("device-resident decode failed; using the transfer/"
                          "numpy path for the rest of this model's runtime",
                          RuntimeWarning)
            raise

    def logits(self, ids) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        if self._is_q4:
            from ref import qwen4_exp_pipeline as _q4
            # qwen4_exp prefill returns unbatched logits (S, V); the executor
            # contract is (1, S, V).
            _, lg = _q4.prefill(ids, self.weights, self._q4cfg())
            return lg[None, :, :]
        kernel = self.recipe.full_attention.kernel
        if kernel == "standard_gqa":
            from ref.standard import standard_model_forward
            return standard_model_forward(ids, self.weights, self.recipe)
        return _pipeline.model_forward(ids, self.weights, self.recipe)

    # -- incremental decode (loop-carried runtime memory) -------------------

    @property
    def supports_incremental(self) -> bool:
        """True when this recipe can run the KV-cached decode path."""
        if self._is_q4:
            # qwen4_exp carries its own incremental state (Qwen4ExpState:
            # GDN + conv window, KV, indexer compressed-key caches).
            return True
        if self.recipe.full_attention.kernel not in _incremental.SUPPORTED_KERNELS:
            return False
        return all(bt in ("full_attention", "linear_attention")
                   for bt in self.recipe.layer_types)

    def max_context(self) -> int:
        return self.recipe.max_position_embeddings

    def prefill(self, ids):
        """Run one full forward, capture per-layer K/V + recurrent state."""
        ids = np.asarray(ids, dtype=np.int64)
        if self._is_q4:
            from ref import qwen4_exp_pipeline as _q4
            state, lg = _q4.prefill(ids, self.weights, self._q4cfg())
            return state, lg[None, :, :]
        return _incremental.prefill(ids, self.weights, self.recipe)

    def decode_step(self, cache, last_id: int) -> np.ndarray:
        """Logits (V,) for the token after `last_id`, using only cached state.

        With `use_gpu=True` and a working kernel build this runs on the GPU
        (kernel drivers); any failure transparently falls back to the numpy
        incremental path.
        """
        last_id = int(last_id)
        if self._is_q4:
            # qwen4_exp GPU kernels are milestone Q4-GPU; numpy pipeline only.
            from ref import qwen4_exp_pipeline as _q4
            return _q4.decode_step(cache, self.weights, self._q4cfg(), last_id)
        if self.use_gpu and self._gpu_available():
            if (self.recipe.full_attention.kernel == "standard_gqa"
                    and not self._resident_off):
                try:
                    return self._resident_step(cache, last_id)
                except Exception:  # noqa: BLE001 - host cache untouched, degrade
                    pass
            try:
                if self.recipe.full_attention.kernel == "standard_gqa":
                    from kernels.standard_decode import gpu_standard_decode_step
                    return gpu_standard_decode_step(cache, self.weights, self.recipe, last_id)
                from kernels.hybrid_decode import gpu_hybrid_decode_step
                return gpu_hybrid_decode_step(cache, self.weights, self.recipe, last_id)
            except Exception:  # pragma: no cover - degrade to numpy, never fail
                pass
        return _incremental.decode_step(cache, self.weights, self.recipe, last_id)


def generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_new: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: tuple[int, ...] = (),
    seed: int | None = None,
    log_progress: bool = False,
    repetition_penalty: float | None = None,
) -> list[int]:
    """Autoregressively generate up to max_new tokens after prompt_ids.

    Returns the FULL id sequence (prompt + generated). Greedy (temperature=0)
    is deterministic regardless of rng. `repetition_penalty > 1` discounts
    tokens already present in the running context (HF-style).
    """
    rng = np.random.default_rng(seed)
    ids = list(int(i) for i in prompt_ids)

    # Incremental path: keep the loop-carried memory (KV / recurrent state)
    # continuous across decode steps. Otherwise fall back to recompute-every-step.
    if getattr(model, "supports_incremental", False):
        max_context = model.max_context()
        if len(ids) >= max_context:
            raise ValueError(
                f"prompt length {len(ids)} exceeds max_position_embeddings "
                f"({max_context}); truncate the prompt or lower max_new"
            )
        cache, plogits = model.prefill(ids)
        logits = plogits[0, -1]  # prefill's last position predicts the NEXT token
        for step in range(max_new):
            logits = _maybe_penalize(logits, ids, repetition_penalty)
            if temperature is not None and temperature <= 0:
                next_id = _sampler.greedy(logits)
            else:
                next_id = _sampler.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng)
            if next_id in stop_ids:
                break
            ids.append(next_id)
            if len(ids) >= max_context:
                break
            logits = model.decode_step(cache, next_id)
            if log_progress:
                print(f"[gen {step}] id={next_id}")
        return ids

    for step in range(max_new):
        logits = model.logits(ids)[0, -1, :]
        logits = _maybe_penalize(logits, ids, repetition_penalty)
        if temperature is not None and temperature <= 0:
            next_id = _sampler.greedy(logits)
        else:
            next_id = _sampler.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng)
        if next_id in stop_ids:
            break
        ids.append(next_id)
        if log_progress:
            print(f"[gen {step}] id={next_id}")
    return ids


def _maybe_penalize(logits, ids, repetition_penalty) -> np.ndarray:
    if repetition_penalty is None or repetition_penalty == 1.0:
        return logits
    return _sampler.apply_repetition_penalty(logits, ids, repetition_penalty)


class InferenceEngine:
    """Minimal serving facade: prompt/chat text in -> assistant text out."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def complete(
        self,
        prompt: str,
        max_new: int = 16,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        stop_ids = tuple(i for i in (self.tokenizer.eos_id(),) if i is not None)
        ids = generate(
            self.model, self.tokenizer, prompt_ids, max_new,
            temperature=temperature, top_k=top_k, top_p=top_p,
            stop_ids=stop_ids, seed=seed, repetition_penalty=repetition_penalty,
        )
        return self.tokenizer.decode(ids[len(prompt_ids):])

    def chat(
        self,
        messages,
        add_generation_prompt: bool = True,
        max_new: int = 16,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        return self.complete(text, max_new=max_new, temperature=temperature,
                             top_k=top_k, top_p=top_p, seed=seed,
                             repetition_penalty=repetition_penalty)


class BatchedInferenceEngine:
    """Continuous-batching facade: the runtime scheduler drives real generation.

    Requests are admitted as blocks/state allow (continuous batching), prefilled
    in chunks, then decoded. Decode steps run the model forward for each running
    sequence.

    With an incremental-capable model, each sequence keeps its own loop-carried
    runtime memory (`IncrementalCache`): `prefill` builds the cache from the
    full prompt once, then every decode step is `decode_step(cache, last_id)`
    (O(context) last-row attention) instead of a full recompute. `chunk_size`
    bookkeeping still advances, and cache building happens on the first prefill
    action. Non-incremental models fall back to the recompute-every-step path.
    """

    def __init__(
        self,
        model,
        tokenizer,
        kv_capacity: int = 1000,
        state_capacity: int = 64,
        block_size: int = 16,
        chunk_size: int = 32,
        max_concurrency: int = 8,
    ):
        from runtime.scheduler import Scheduler

        self.model = model
        self.tokenizer = tokenizer
        self.sched = Scheduler(
            kv_capacity, state_capacity,
            block_size=block_size, chunk_size=chunk_size, max_concurrency=max_concurrency,
        )
        self._seqs = {}
        self._order = []
        self._next_seq = 0
        self._stop = tuple(i for i in (tokenizer.eos_id(),) if i is not None)
        self._inc = bool(getattr(model, "supports_incremental", False))

    # -- request lifecycle ---------------------------------------------------

    def submit(self, prompt: str, max_new: int, temperature: float = 0.0,
               top_k: int | None = None, top_p: float | None = None,
               seed: int | None = None, repetition_penalty: float | None = None) -> int:
        seq_id = self._next_seq
        self._next_seq += 1
        prompt_ids = self.tokenizer.encode(prompt)
        self._seqs[seq_id] = {
            "prompt": prompt_ids, "gen": [], "max_new": int(max_new),
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "rng": np.random.default_rng(seed), "text": None,
            "rep": repetition_penalty,
            "cache": None, "last_id": None, "prefill_L": None,
        }
        self._order.append(seq_id)
        self.sched.add(seq_id, prompt_len=len(prompt_ids), max_new=int(max_new))
        return seq_id

    def result_text(self, seq_id: int) -> str:
        info = self._seqs[seq_id]
        if info["text"] is None:
            info["text"] = self.tokenizer.decode(info["gen"])
        return info["text"]

    def results_in_order(self) -> list[str]:
        return [self.result_text(i) for i in self._order]

    @property
    def queue_length(self) -> int:
        return self.sched.waiting_count + len(self.sched.running)

    @property
    def done_count(self) -> int:
        return len(self.sched.done)

    # -- execution -----------------------------------------------------------

    def _sample_token(self, info: dict, logits) -> int:
        if info["rep"] is not None and info["rep"] != 1.0:
            logits = _sampler.apply_repetition_penalty(
                logits, info["prompt"] + info["gen"], info["rep"])
        if info["temperature"] is not None and info["temperature"] <= 0:
            return _sampler.greedy(logits)
        return _sampler.sample(
            logits, temperature=info["temperature"], top_k=info["top_k"],
            top_p=info["top_p"], rng=info["rng"],
        )

    def step(self) -> dict:
        """Run one scheduler step: schedule actions, execute decodes, advance."""
        sched = self.sched.step()
        for a in sched.actions:
            info = self._seqs[a.seq_id]
            if a.phase == "decode":
                if self._inc:
                    if info["cache"] is None:
                        info["cache"], pl = self.model.prefill(info["prompt"])
                        info["prefill_L"] = pl[0, -1]
                    if info["prefill_L"] is not None:
                        logits = info["prefill_L"]
                        info["prefill_L"] = None
                    else:
                        logits = self.model.decode_step(info["cache"], info["last_id"])
                else:
                    full = info["prompt"] + info["gen"]
                    logits = self.model.logits(full)[0, -1, :]
                tok = self._sample_token(info, logits)
                eos = tok in self._stop
                if not eos:
                    info["gen"].append(tok)
                    info["last_id"] = tok
                self.sched.advance(a, eos=eos)
            else:
                if self._inc and info["cache"] is None:
                    info["cache"], pl = self.model.prefill(info["prompt"])
                    info["prefill_L"] = pl[0, -1]
                self.sched.advance(a)
        return {"actions": sched.actions, "queue_length": self.queue_length,
                "done": self.done_count}

    def run_all(self, max_steps: int | None = None) -> list[str]:
        """Run steps until every submitted request is finished."""
        max_steps = max_steps or (int(1e6))
        for _ in range(max_steps):
            res = self.step()
            if not res["actions"]:
                break
        return self.results_in_order()


def generate_batch(model, tokenizer, prompts: list[str], max_new: int,
                   **submit_kwargs) -> list[str]:
    """Convenience: continuously serve a list of prompts, return texts in order."""
    eng = BatchedInferenceEngine(model, tokenizer)
    for p in prompts:
        eng.submit(p, max_new=max_new, **submit_kwargs)
    return eng.run_all()
