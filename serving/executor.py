"""Generation loop: token ids -> [chat template] -> forward -> sample -> stop."""

from __future__ import annotations

import json
import os
import time

import numpy as np

from ref import pipeline as _pipeline
from ref import incremental as _incremental
from runtime import sampler as _sampler
from serving import diag

QWEN4_EXP_ARCH = "qwen4_exp"
DEEPSEEK_V4_ARCH = "deepseek_v4"


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
                 gpu_dtype: str | None = None, q4_cfg=None,
                 dsv4_cfg=None, vision_cfg=None):
        self.recipe = recipe
        self.weights = weights
        # qwen4_exp pipeline config (ref/qwen4_exp_pipeline.Qwen4ExpCfg). When
        # the arch is qwen4_exp and no cfg is given, it is derived from the
        # recipe lazily (Qwen4ExpCfg.from_recipe).
        self.q4_cfg = q4_cfg
        # DeepSeek-V4 oracle config (ref.deepseek_v4.DeepseekV4Cfg); like
        # q4_cfg, it falls back to DeepseekV4Cfg.from_recipe when not given.
        self.dsv4_cfg = dsv4_cfg
        self.vision_cfg = vision_cfg
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
        self._gpu_warned = False

    @property
    def _is_q4(self) -> bool:
        return self.recipe.arch == QWEN4_EXP_ARCH

    @property
    def _is_dsv4(self) -> bool:
        return self.recipe.arch == DEEPSEEK_V4_ARCH

    def _dsv4model(self):
        # built lazily (mirrors _q4cfg); the DeepSeek-V4 oracle owns its
        # prefill/decode state (state list + position), no shared buffers.
        m = getattr(self, "_dsv4_model", None)
        if m is None:
            from ref.deepseek_v4 import DeepseekV4Cfg, DeepseekV4Model
            cfg = self.dsv4_cfg or DeepseekV4Cfg.from_recipe(self.recipe)
            kwargs = {"vision_cfg": self.vision_cfg} if self.vision_cfg else {}
            self._dsv4_model = DeepseekV4Model(self.weights, cfg, self.recipe,
                                               **kwargs)
        return self._dsv4_model

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
            import traceback
            self._resident_off = True
            # the underlying error MUST be visible: a silent permanent
            # fallback hides real bugs (missing .so, CUDA OOM, code faults)
            warnings.warn("device-resident decode failed; using the transfer/"
                          "numpy path for the rest of this model's runtime\n"
                          + traceback.format_exc(limit=4), RuntimeWarning,
                          stacklevel=2)
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
        if self._is_dsv4:
            _, lg = self._dsv4model().prefill(ids.reshape(-1))
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
        if self._is_dsv4:
            return True
        if self.recipe.full_attention.kernel not in _incremental.SUPPORTED_KERNELS:
            return False
        return all(bt in ("full_attention", "linear_attention")
                   for bt in self.recipe.layer_types)

    def max_context(self) -> int:
        return self.recipe.max_position_embeddings

    def prefill(self, ids, images=None):
        """Run one full forward, capture per-layer K/V + recurrent state."""
        ids = np.asarray(ids, dtype=np.int64)
        if self._is_q4:
            from ref import qwen4_exp_pipeline as _q4
            state, lg = _q4.prefill(ids, self.weights, self._q4cfg())
            return state, lg[None, :, :]
        if self._is_dsv4:
            state, lg = self._dsv4model().prefill(ids.reshape(-1), images)
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
        if self._is_dsv4:
            return self._dsv4model().decode_step(cache, last_id)
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
                if not self._gpu_warned:
                    import traceback
                    import warnings
                    self._gpu_warned = True
                    warnings.warn(
                        "GPU decode step failed; falling back to the numpy "
                        "path (for this step)\n" + traceback.format_exc(limit=4),
                        RuntimeWarning, stacklevel=2)
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
    return_info: bool = False,
    images=None,
):
    """Autoregressively generate up to max_new tokens after prompt_ids.

    Returns the FULL id sequence (prompt + generated), or — with
    `return_info=True` — a tuple (ids, {"finish_reason": stop|length,
    "output_len": int}). Greedy (temperature=0) is deterministic regardless
    of rng. `repetition_penalty > 1` discounts tokens already present in the
    running context (HF-style). `images` are the expanded multimodal
    ImageInput blocks consumed by the DeepSeek-V4 prefill (None = text-only).
    """
    rng = np.random.default_rng(seed)
    ids = list(int(i) for i in prompt_ids)
    n_prompt = len(ids)
    reason = "length"
    _t0 = time.perf_counter()

    # Incremental path: keep the loop-carried memory (KV / recurrent state)
    # continuous across decode steps. Otherwise fall back to recompute-every-step.
    if getattr(model, "supports_incremental", False):
        max_context = model.max_context()
        if len(ids) >= max_context:
            raise ValueError(
                f"prompt length {len(ids)} exceeds max_position_embeddings "
                f"({max_context}); truncate the prompt or lower max_new"
            )
        if images is not None and not getattr(model, "_is_dsv4", False):
            raise ValueError("images are only supported by the deepseek_v4 arch")
        cache, plogits = model.prefill(ids) if images is None \
            else model.prefill(ids, images)
        logits = plogits[0, -1]  # prefill's last position predicts the NEXT token
        for step in range(max_new):
            logits = _maybe_penalize(logits, ids, repetition_penalty)
            if temperature is not None and temperature <= 0:
                next_id = _sampler.greedy(logits)
            else:
                next_id = _sampler.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng)
            if next_id in stop_ids:
                reason = "stop"
                break
            ids.append(next_id)
            if len(ids) >= max_context:
                break
            logits = model.decode_step(cache, next_id)
            diag.debug("gen",
                       f"step={step} id={next_id} "
                       f"top1={int(_sampler.greedy(logits))}")
    else:
        max_context = getattr(model, "max_context", None)
        if callable(max_context):
            max_context = max_context()
        for step in range(max_new):
            logits = model.logits(ids)[0, -1, :]
            logits = _maybe_penalize(logits, ids, repetition_penalty)
            if temperature is not None and temperature <= 0:
                next_id = _sampler.greedy(logits)
            else:
                next_id = _sampler.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng)
            if next_id in stop_ids:
                reason = "stop"
                break
            ids.append(next_id)
            # the recompute path must not silently run past the trained
            # position range either (the incremental path clamps above)
            if max_context is not None and len(ids) >= max_context:
                break
            diag.debug("gen",
                       f"step={step} id={next_id} "
                       f"top1={int(_sampler.greedy(logits))}")
    _emit_gen_stats(_t0, n_prompt, ids, reason)
    if return_info:
        return ids, {"finish_reason": reason, "output_len": len(ids) - n_prompt}
    return ids


def _emit_gen_stats(t0, n_prompt: int, ids, reason: str) -> None:
    """INFO line for one completed generation (vLLM-style request stats)."""
    wall = time.perf_counter() - t0
    gen = len(ids) - n_prompt
    diag.info("gen", f"prompt={n_prompt} out={gen} wall={wall:.3f}s "
                     f"({diag.tps(gen, wall)}) finish={reason}")


def _maybe_penalize(logits, ids, repetition_penalty) -> np.ndarray:
    if repetition_penalty is None or repetition_penalty == 1.0:
        return logits
    return _sampler.apply_repetition_penalty(logits, ids, repetition_penalty)


def generate_stream(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_new: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: tuple[int, ...] = (),
    seed: int | None = None,
    repetition_penalty: float | None = None,
    images=None,
):
    """SSE-friendly generator: yields (delta_text, finish_reason|None) for each
    emitted token, mirroring `generate` exactly (same incremental/recompute
    branches, same stop/length semantics). The final `finish_reason` is non-None
    only on the last yield; callers accumulate deltas."""
    rng = np.random.default_rng(seed)
    ids = list(int(i) for i in prompt_ids)
    n_prompt = len(ids)
    reason = "length"
    _t0 = time.perf_counter()
    last_full = ""

    def _delta() -> str:
        nonlocal last_full
        gen = ids[n_prompt:]
        full = tokenizer.decode(gen)
        d = full[len(last_full):]
        last_full = full
        return d

    if getattr(model, "supports_incremental", False):
        max_context = model.max_context()
        if len(ids) >= max_context:
            raise ValueError(
                f"prompt length {len(ids)} exceeds max_position_embeddings "
                f"({max_context}); truncate the prompt or lower max_new")
        if images is not None and not getattr(model, "_is_dsv4", False):
            raise ValueError("images are only supported by the deepseek_v4 arch")
        cache, plogits = model.prefill(ids) if images is None \
            else model.prefill(ids, images)
        logits = plogits[0, -1]
        for _step in range(max_new):
            logits = _maybe_penalize(logits, ids, repetition_penalty)
            if temperature is not None and temperature <= 0:
                next_id = _sampler.greedy(logits)
            else:
                next_id = _sampler.sample(logits, temperature=temperature,
                                          top_k=top_k, top_p=top_p, rng=rng)
            if next_id in stop_ids:
                reason = "stop"
                break
            ids.append(next_id)
            yield _delta(), None
            if len(ids) >= max_context:
                break
            logits = model.decode_step(cache, next_id)
    else:
        max_context = getattr(model, "max_context", None)
        if callable(max_context):
            max_context = max_context()
        for _step in range(max_new):
            logits = model.logits(ids)[0, -1, :]
            logits = _maybe_penalize(logits, ids, repetition_penalty)
            if temperature is not None and temperature <= 0:
                next_id = _sampler.greedy(logits)
            else:
                next_id = _sampler.sample(logits, temperature=temperature,
                                          top_k=top_k, top_p=top_p, rng=rng)
            if next_id in stop_ids:
                reason = "stop"
                break
            ids.append(next_id)
            yield _delta(), None
            if max_context is not None and len(ids) >= max_context:
                break
    _emit_gen_stats(_t0, n_prompt, ids, reason)
    yield "", reason


class InferenceEngine:
    """Minimal serving facade: prompt/chat text in -> assistant text out."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def describe(self) -> list[str]:
        """Uniform engine summary lines (arch registry in service.engine_cards)."""
        from serving import engine_cards
        weights = getattr(self.model, "weights", {})
        return engine_cards.describe(self.model.recipe, weights, self.model)

    def show_banner(self, tag: str = "sllm") -> None:
        """vLLM-style INFO startup banner for this engine."""
        rec = self.model.recipe
        title = f"{rec.name or rec.model_id} (arch={rec.arch})"
        diag.banner(tag, title, self.describe())

    def complete_detail(
        self,
        prompt: str,
        max_new: int = 16,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> dict:
        """Same as complete() but returns the real token ledger + finish
        reason ("stop" | "length") instead of only the text."""
        prompt_ids = self.tokenizer.encode(prompt)
        stop_ids = tuple(i for i in (self.tokenizer.eos_id(),) if i is not None)
        ids, info = generate(
            self.model, self.tokenizer, prompt_ids, max_new,
            temperature=temperature, top_k=top_k, top_p=top_p,
            stop_ids=stop_ids, seed=seed, repetition_penalty=repetition_penalty,
            return_info=True,
        )
        gen = ids[len(prompt_ids):]
        return {
            "text": self.tokenizer.decode(gen),
            "finish_reason": info["finish_reason"],
            "prompt_len": len(prompt_ids),
            "completion_len": info["output_len"],
        }

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
        return self.complete_detail(
            prompt, max_new=max_new, temperature=temperature, top_k=top_k,
            top_p=top_p, seed=seed,
            repetition_penalty=repetition_penalty)["text"]

    def stream_complete(
        self,
        prompt: str,
        max_new: int = 16,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ):
        """Yield (delta_text, finish_reason|None) events for /v1/completions."""
        prompt_ids = self.tokenizer.encode(prompt)
        stop_ids = tuple(i for i in (self.tokenizer.eos_id(),) if i is not None)
        yield from generate_stream(
            self.model, self.tokenizer, prompt_ids, max_new,
            temperature=temperature, top_k=top_k, top_p=top_p,
            stop_ids=stop_ids, seed=seed,
            repetition_penalty=repetition_penalty)

    def chat_detail(
        self,
        messages,
        add_generation_prompt: bool = True,
        max_new: int = 16,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> dict:
        if self._messages_have_images(messages) and self._model_has_vision():
            return self.vl_chat_detail(
                messages, max_new=max_new, temperature=temperature,
                top_k=top_k, top_p=top_p, seed=seed,
                repetition_penalty=repetition_penalty)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt)
        d = self.complete_detail(prompt_text, max_new=max_new,
                                 temperature=temperature, top_k=top_k,
                                 top_p=top_p, seed=seed,
                                 repetition_penalty=repetition_penalty)
        d["prompt_text"] = prompt_text
        return d

    def stream_chat(
        self,
        messages,
        add_generation_prompt: bool = True,
        max_new: int = 16,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ):
        """Yield (delta_text, finish_reason|None) events for /v1/chat/
        completions (vision prompts stream their text-only tail as well)."""
        if self._messages_have_images(messages) and self._model_has_vision():
            (ids, inputs), _ = self._vl_prompt(messages)
            stop_ids = tuple(i for i in (self.tokenizer.eos_id(),)
                             if i is not None)
            yield from generate_stream(
                self.model, self.tokenizer, ids, max_new,
                temperature=temperature, top_k=top_k, top_p=top_p,
                stop_ids=stop_ids, seed=seed, images=inputs,
                repetition_penalty=repetition_penalty)
            return
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt)
        yield from self.stream_complete(
            prompt_text, max_new=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, seed=seed,
            repetition_penalty=repetition_penalty)

    # -- multimodal (vision) chat ------------------------------------------

    def _model_has_vision(self) -> bool:
        if not getattr(self.model, "_is_dsv4", False):
            return False
        try:
            return self.model._dsv4model().vision is not None
        except Exception:  # noqa: BLE001 - a broken oracle must not 500 the API
            return False

    @staticmethod
    def _messages_have_images(messages) -> bool:
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in (
                            "image", "image_url"):
                        return True
        return False

    def _vl_prompt(self, messages):
        """Encode OpenAI content-block messages -> (prompt_ids, image inputs)
        for a DeepSeek-V4 vision model."""
        from . import encoding_dsv4 as enc
        from . import image_processor as ip

        model = self.model
        prompt_text, media = enc.encode_messages(
            messages, "chat", return_multi_modal_data=True)
        images_records = media.get("images") or []
        if not images_records:
            raise ValueError("vl_chat<detail|stream> needs an image block")
        pl_id = getattr(self.tokenizer, "image_token_id", None)
        if pl_id is None:
            raise ValueError("tokenizer does not expose an image placeholder id")
        prompt_ids = self.tokenizer.encode(prompt_text)
        args = self._vision_args(ip)
        vocab = model.recipe.vocab_size
        ids, inputs = ip.expand_image_placeholders(
            prompt_ids, images_records, args=args, vocab_size=vocab,
            image_placeholder_id=pl_id())
        return (ids, inputs), len(images_records)

    def vl_chat_detail(
        self,
        messages,
        max_new: int = 16,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> dict:
        """OpenAI content-block chat -> DeepSeek-V4 image blocks -> generate."""
        (ids, inputs), n_images = self._vl_prompt(messages)
        stop_ids = tuple(i for i in (self.tokenizer.eos_id(),) if i is not None)
        out_ids, info = generate(
            self.model, self.tokenizer, ids, max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, stop_ids=stop_ids, seed=seed,
            repetition_penalty=repetition_penalty, return_info=True,
            images=inputs)
        gen = out_ids[len(ids):]
        return {
            "text": self.tokenizer.decode(gen),
            "finish_reason": info["finish_reason"],
            "prompt_len": len(ids),
            "completion_len": info["output_len"],
            "n_images": n_images,
        }

    def _vision_args(self, ip):
        vs = getattr(self.model.recipe, "vision", None)
        if vs is not None and vs.enabled:
            return ip.VisionArgs(
                patch_size=vs.ds_patch_size or 14,
                downsample_ratio=vs.downsample_ratio or 3,
                max_n_token=vs.max_n_token or 384,
                min_pixels=vs.min_pixels or 147456,
                max_wh_ratio=vs.max_wh_ratio if vs.max_wh_ratio is not None
                else 8.0)
        return ip.VisionArgs()

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
        return self.chat_detail(
            messages, add_generation_prompt=add_generation_prompt,
            max_new=max_new, temperature=temperature, top_k=top_k,
            top_p=top_p, seed=seed,
            repetition_penalty=repetition_penalty)["text"]


        d = self.chat_detail(
            messages, add_generation_prompt=add_generation_prompt,
            max_new=max_new, temperature=temperature, top_k=top_k, top_p=top_p,
            seed=seed,
            repetition_penalty=repetition_penalty)
        return d["text"]

    def _dsv4_text_chat_detail(self, messages, max_new=16, temperature=0.0,
                               top_k=None, top_p=None, seed=None,
                               repetition_penalty=None):
        """DeepSeek-V4 TEXT (no-image) chat turn through enc.encode_messages
        (the DSML prompt builder the VL path shares), so tool/thinking tags
        are present in the prompt even without images."""
        from . import encoding_dsv4 as enc
        prompt_text, _media = enc.encode_messages(messages, "chat")
        prompt_ids = self.tokenizer.encode(prompt_text)
        stop_ids = tuple(i for i in (self.tokenizer.eos_id(),)
                         if i is not None)
        out_ids, info = generate(
            self.model, self.tokenizer, prompt_ids, max_new,
            temperature=temperature, top_k=top_k, top_p=top_p,
            stop_ids=stop_ids, seed=seed,
            repetition_penalty=repetition_penalty, return_info=True)
        gen = out_ids[len(prompt_ids):]
        return {
            "text": self.tokenizer.decode(gen),
            "finish_reason": info["finish_reason"],
            "prompt_len": len(prompt_ids),
            "completion_len": info["output_len"],
            "prompt_text": prompt_text,
        }

    def chat_tools(
        self,
        messages,
        tools=None,
        max_new: int = 256,
        max_turns: int = 4,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> dict:
        """DeepSeek-V4 DSML tool-call loop.

        Runs assistant turns; when a turn contains DSML tool calls, executes
        them against the registered `tools` mapping (name -> callable or
        {"callable": fn, "requires": [arg,...]}) and feeds the results back as
        tool_result blocks before re-generating. Returns the final assistant
        message dict (OpenAI-format: content/reasoning_content/tool_calls) plus
        the multipart `messages` ledger.

        The turn text is parsed with encoding_dsv4.parse_message_from_...
        only for the deepseek_v4 architecture; any other architecture returns
        the plain chat text (no tool protocol).
        """
        from . import encoding_dsv4 as enc

        is_dsv4 = bool(getattr(self.model, "_is_dsv4", False))
        tools = dict(tools or {})
        msgs = list(messages)
        final = None
        turns = 0
        while turns < max_turns:
            turns += 1
            if is_dsv4:
                d = self._dsv4_text_chat_detail(
                    msgs, max_new=max_new, temperature=temperature,
                    top_k=top_k, top_p=top_p, seed=seed,
                    repetition_penalty=repetition_penalty)
            else:
                d = self.chat_detail(
                    msgs, add_generation_prompt=True, max_new=max_new,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    seed=seed, repetition_penalty=repetition_penalty)
            text = d["text"]
            if not is_dsv4 or not tools:
                final = {"role": "assistant", "content": text,
                         "reasoning_content": "", "tool_calls": []}
                break
            try:
                parsed = enc.parse_message_from_completion_text(
                    text, thinking_mode="chat")
            except Exception:  # noqa: BLE001 - malformed turn = final answer
                final = {"role": "assistant", "content": text,
                         "reasoning_content": "", "tool_calls": []}
                break
            final = parsed
            tcs = parsed.get("tool_calls") or []
            if not tcs:
                break
            msgs.append(parsed)
            results = []
            for tc in tcs:
                fn_name = (tc.get("function") or {}).get("name") or tc.get("name")
                args = (tc.get("function") or {}).get("arguments")
                spec = tools.get(fn_name)
                if spec is None:
                    results.append({"tool_call_id": tc.get("id"),
                                    "content": f"unknown tool: {fn_name}"})
                    continue
                callable_fn = (spec.get("callable") if isinstance(spec, dict)
                               else spec)
                if not callable(callable_fn):
                    results.append({"tool_call_id": tc.get("id"),
                                    "content": "tool not callable"})
                    continue
                try:
                    if isinstance(args, str):
                        kwargs = json.loads(args) if args.strip() else {}
                    else:
                        kwargs = dict(args or {})
                    out = callable_fn(**kwargs)
                    content = str(out) if out is not None else "ok"
                except Exception as exc:  # noqa: BLE001 - tool errors are data
                    content = f"error: {exc}"
                results.append({"tool_call_id": tc.get("id"),
                                "content": content})
            msgs.extend(results)  # role "tool" messages merged by encode path
            msgs = enc.merge_tool_messages(msgs)
        return {"assistant": final, "messages": msgs, "turns": turns}


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
        # pure full-attention models (qwen2/llama) need no recurrent-state
        # slot; asking for one would needlessly cap concurrency
        rec = getattr(model, "recipe", None)
        self._use_state = not (rec is not None and getattr(rec, "layer_types", None)
                               and not any(bt == "linear_attention"
                                           for bt in rec.layer_types))

    # -- request lifecycle ---------------------------------------------------

    def submit(self, prompt: str, max_new: int, temperature: float = 0.0,
               top_k: int | None = None, top_p: float | None = None,
               seed: int | None = None, repetition_penalty: float | None = None) -> int:
        seq_id = self._next_seq
        self._next_seq += 1
        prompt_ids = self.tokenizer.encode(prompt)
        max_ctx = getattr(self.model, "max_context", None)
        if callable(max_ctx):
            max_ctx = max_ctx()
        if max_ctx and len(prompt_ids) + int(max_new) > max_ctx:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) + max_new ({max_new}) exceeds "
                f"max_position_embeddings ({max_ctx})")
        self._seqs[seq_id] = {
            "prompt": prompt_ids, "gen": [], "max_new": int(max_new),
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "rng": np.random.default_rng(seed), "text": None,
            "rep": repetition_penalty,
            "cache": None, "last_id": None, "prefill_L": None,
        }
        self._order.append(seq_id)
        self.sched.add(seq_id, prompt_len=len(prompt_ids), max_new=int(max_new),
                       use_state=self._use_state)
        return seq_id

    def abort(self, seq_id: int) -> bool:
        """Cancel a submitted request (client disconnect)."""
        if self.sched.abort(seq_id):
            info = self._seqs.get(seq_id)
            if info is not None and info["text"] is None:
                info["text"] = self.tokenizer.decode(info["gen"])
            return True
        return False

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
        """Run steps until every submitted request is finished.

        Raises RuntimeError if the scheduler stalls (requests remain but no
        action is ever scheduled — e.g. a prompt that can never be admitted)
        instead of silently returning partial/empty results."""
        if max_steps is None:
            max_steps = int(1e6)
        for _ in range(max_steps):
            res = self.step()
            if not res["actions"]:
                if res["queue_length"]:
                    raise RuntimeError(
                        f"scheduler stalled: {res['queue_length']} request(s) "
                        "admitted=no action scheduled (prompt too large for "
                        "the KV/state budget?)")
                break
        else:
            if self.queue_length:
                raise RuntimeError(
                    f"run_all exhausted max_steps={max_steps} with "
                    f"{self.queue_length} request(s) unfinished")
        return self.results_in_order()


def generate_batch(model, tokenizer, prompts: list[str], max_new: int,
                   **submit_kwargs) -> list[str]:
    """Convenience: continuously serve a list of prompts, return texts in order."""
    eng = BatchedInferenceEngine(model, tokenizer)
    for p in prompts:
        eng.submit(p, max_new=max_new, **submit_kwargs)
    return eng.run_all()
