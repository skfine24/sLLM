"""D+V cluster-gated real-checkpoint vision test (SKIPPED on the dev box).

Runs only where the DeepSeek-V4-Flash-0731 checkpoint's index + shards and a
CUDA device exist, gated by environment variables so the local suite stays
green without them:

    SLLM_REAL_CKPT_DIR   real checkpoint dir (index json + shards)
    SLLM_REAL_ALLOW      set to "1" to actually run

Loads a MEMORY-SANE slice from the real bytes -- top-level weights, the first
two layers (ratios exactly as the real layout), and the 259-tensor BF16 vision
tower (all on shard 1) -- via the index + per-tensor Range reads, folds the
E8M0/FP4 scales with `dequant_tensors`, and drives the numpy oracle with
`vision_bf16=True` through a real (tiny) image prefill + decode. This is the
first byte-level validation of the VL path against the actual checkpoint; the
full 43-layer decode is bench/deepseek_subset_parity.py + the cluster golden.
"""

import json
import os
import unittest

import numpy as np

_REAL = os.environ.get("SLLM_REAL_CKPT_DIR")
_ALLOW = os.environ.get("SLLM_REAL_ALLOW") == "1"


def _find_index(model_dir: str) -> str:
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith(".safetensors.index.json"):
            return os.path.join(model_dir, fn)
    raise SystemExit("[real] no safetensors index json in the checkpoint dir")


@unittest.skipUnless(_REAL and _ALLOW,
                     "real 168GiB checkpoint not staged (SLLM_REAL_CKPT_DIR "
                     "+ SLLM_REAL_ALLOW=1)")
class TestRealCheckpointVision(unittest.TestCase):
    def test_real_vision_prefill_decode(self):
        import ref.deepseek_v4 as ds
        from loaders import safetensors_reader as sr
        from loaders.weights import dequant_tensors
        from serving.image_processor import VisionArgs, expand_image_placeholders
        from serving.encoding_dsv4 import IMAGE_PLACEHOLDER

        model_dir = _REAL
        idx_path = _find_index(model_dir)
        with open(idx_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = {}
        for k, s in weight_map.items():
            shards.setdefault(s, []).append(k)
        shard_path = {s: os.path.join(model_dir, s) for s in shards}

        # cfg: real geometry, first two layers
        cfg_raw = json.load(open(os.path.join(model_dir, "config.json"),
                                 encoding="utf-8"))
        cfg = ds.DeepseekV4Cfg()
        cfg.vocab_size = int(cfg_raw["vocab_size"])
        cfg.dim = int(cfg_raw["hidden_size"])
        cfg.n_layers = 2
        cfg.n_heads = int(cfg_raw["num_attention_heads"])
        cfg.head_dim = int(cfg_raw.get("qk_nope_head_dim", 512)) + \
            int(cfg_raw.get("qk_rope_head_dim", 64))
        cfg.rope_head_dim = int(cfg_raw.get("qk_rope_head_dim", 64))
        cfg.window_size = int(cfg_raw.get("sliding_window", 128))
        ratios = cfg_raw.get("compress_ratios") or [0, 0, 4, 128]
        cfg.compress_ratios = tuple(ratios[:2])  # real first-two-layer layout
        cfg.q_lora_rank = int(cfg_raw.get("q_lora_rank", 1024))
        cfg.o_lora_rank = int(cfg_raw.get("o_lora_rank", 1024))
        cfg.o_groups = int(cfg_raw.get("o_groups", 8))
        cfg.n_routed_experts = int(cfg_raw["n_routed_experts"])
        cfg.n_activated_experts = int(cfg_raw["num_experts_per_tok"])
        cfg.moe_inter_dim = int(cfg_raw.get("moe_intermediate_size", 2048))
        cfg.vision = True

        # tensor slice: top-level + 2 layers + vision tower
        want = {"embed.weight", "head.weight", "norm.weight", "hc_head_fn",
                "hc_head_scale", "hc_head_base", "image_start", "image_pad",
                "image_end", "image_newline"}
        want.update(k for k in weight_map if k.startswith("vision."))
        for i in (0, 1):
            want.update(k for k in weight_map if k.startswith(f"layers.{i}."))
        # layer-index remap 0..1 is identity here (we use real indices 0,1)

        loaded = {}
        for s, names in shards.items():
            selected = [n for n in names if n in want]
            if not selected:
                continue
            loaded.update(sr.load_tensors(shard_path[s], names=selected))
        dequant_tensors(loaded, scale_suffix=".scale")
        self.assertIn("embed.weight", loaded)
        self.assertTrue(any(k.startswith("vision.") for k in loaded),
                        "vision tower not present on shard 1")

        model = ds.DeepseekV4Model(
            loaded, cfg, vision_cfg=ds.VisionCfg(dim=cfg.dim), vision_bf16=True)
        self.assertIsNotNone(model.vision)

        # a 32x24 png -> expand -> prefill (+ one decode step)
        png = _tiny_png(32, 24)
        args = VisionArgs(patch_size=14, downsample_ratio=3, max_n_token=384,
                          min_pixels=147456, max_wh_ratio=8.0)
        codes, inputs = expand_image_placeholders(
            [2, 3, cfg.vocab_size + 2, 5], [{"data": png}], args=args,
            vocab_size=cfg.vocab_size, image_placeholder_id=cfg.vocab_size)
        try:
            state, lg = model.prefill(codes, images=inputs)
        except ValueError as exc:
            self.fail(f"vision prefill against real bytes failed: {exc}")
        self.assertTrue(np.isfinite(lg).all())
        lr = model.decode_step(state, int(codes[-1]))
        self.assertTrue(np.isfinite(lr).all())
        self.assertEqual(lr.shape, (cfg.vocab_size,))


def _tiny_png(w: int, h: int) -> str:
    """A tiny RGBA PNG as a base64 data URI (transparent gradient)."""
    import base64
    import struct
    import zlib

    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes((x * 255 // max(w - 1, 1),
                          y * 255 // max(h - 1, 1), 128, 255))
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(bytes(raw)))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


if __name__ == "__main__":
    unittest.main()
