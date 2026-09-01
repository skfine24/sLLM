"""DeepSeek-V4 vision (L3): image processor + ViT/aligner oracle + multimodal
model prefill."""

import io
import unittest

import numpy as np
from PIL import Image

from ref.deepseek_v4 import DeepseekV4Model
from ref.vision_deepseek import DeepseekVisionEncoder, VisionCfg
from serving import image_processor as ip
from serving.dev_model import tiny_deepseek_v4_cfg, tiny_deepseek_v4_weights


def _tiny_rgb_png(w=96, h=48):
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _vision_weights(conf: VisionCfg, llm_dim: int, rng=None):
    rng = rng or np.random.default_rng(3)
    d, nh, inter, p = conf.dim, conf.n_heads, conf.inter_dim, conf.patch_size
    w = {}
    w["vision.patch_embed.proj.weight"] = \
        (rng.standard_normal((d, 3 * p * p)) * 0.05).astype(np.float32)
    w["vision.patch_embed.proj.bias"] = rng.standard_normal(d).astype(np.float32)
    for i in range(conf.n_layers):
        b = f"vision.blocks.{i}."
        w[b + "norm1.weight"] = rng.standard_normal(d).astype(np.float32)
        w[b + "attn.wqkv.weight"] = (rng.standard_normal((3 * d, d)) * 0.05
                                     ).astype(np.float32)
        w[b + "attn.wqkv.bias"] = rng.standard_normal(3 * d).astype(np.float32)
        w[b + "attn.wo.weight"] = (rng.standard_normal((d, d)) * 0.05
                                   ).astype(np.float32)
        w[b + "attn.wo.bias"] = rng.standard_normal(d).astype(np.float32)
        w[b + "norm2.weight"] = rng.standard_normal(d).astype(np.float32)
        w[b + "mlp.w1.weight"] = (rng.standard_normal((2 * inter, d)) * 0.05
                                  ).astype(np.float32)
        w[b + "mlp.w2.weight"] = (rng.standard_normal((d, inter)) * 0.05
                                  ).astype(np.float32)
    w["vision.norm.weight"] = rng.standard_normal(d).astype(np.float32)
    r = conf.downsample_ratio
    w["aligner.w1.weight"] = (rng.standard_normal((llm_dim, d * r * r)) * 0.05
                              ).astype(np.float32)
    w["aligner.w1.bias"] = rng.standard_normal(llm_dim).astype(np.float32)
    w["aligner.w2.weight"] = (rng.standard_normal((llm_dim, llm_dim)) * 0.05
                              ).astype(np.float32)
    w["aligner.w2.bias"] = rng.standard_normal(llm_dim).astype(np.float32)
    for n in ("image_start", "image_pad", "image_end", "image_newline"):
        w[n] = rng.standard_normal(llm_dim).astype(np.float32) * 0.05
    return w


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.args = ip.VisionArgs(patch_size=14, downsample_ratio=3,
                                  max_n_token=384, min_pixels=0,
                                  max_wh_ratio=8.0)

    def test_load_image_patches(self):
        patches, n_h, n_w, nlh, nlw = ip.load_image(
            {"data": _tiny_rgb_png(100, 60)}, self.args)
        self.assertEqual(patches.shape[1:], (3, 14, 14))
        self.assertTrue(np.isfinite(patches).all())
        self.assertEqual(n_h, patches.shape[0] // n_w)

    def test_build_image_block_consistent(self):
        n_h, n_w = 2, 3
        types, perm = ip.build_image_block(n_h, n_w, 0)
        # every aligned token index is a real patch (perm) and the type list
        # covers START + grid + (START/END padding) + END
        self.assertEqual(perm.dtype, np.int64)
        self.assertGreater(len(perm), 0)
        self.assertEqual(types[-1], ip.IMAGE_END)
        self.assertIn(ip.IMAGE_START, types.tolist())
        self.assertEqual(int(np.count_nonzero(types == ip.IMAGE)), n_h * n_w)

    def test_expand_placeholders(self):
        prompt = [5, 6, ip.IMAGE_START + 64, 7]
        vocab = 64
        imgs = [{"data": _tiny_rgb_png(60, 60)}]
        codes, inputs = ip.expand_image_placeholders(
            prompt, imgs, vocab_size=vocab, image_placeholder_id=64)
        self.assertIsNotNone(inputs)
        # the first two text tokens stay put; the block is inserted after them
        self.assertEqual(codes[:2], [5, 6])
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].start, 2)

    def test_placeholder_mismatch_raises(self):
        # two placeholders but only one image -> generator underflow
        with self.assertRaises(StopIteration):
            img = _tiny_rgb_png(10, 8)
            ip.expand_image_placeholders([64, 64], [{"data": img}],
                                         vocab_size=64,
                                         image_placeholder_id=64)


class TestVisionEncoder(unittest.TestCase):
    def test_vit_and_align_shapes(self):
        conf = VisionCfg(dim=32, n_layers=1, n_heads=4, inter_dim=16,
                         patch_size=2, downsample_ratio=2)
        w = _vision_weights(conf, llm_dim=32)
        enc = DeepseekVisionEncoder(w, conf)
        n_h, n_w = 2, 3  # 6 patches
        patches = (np.random.default_rng(1).random((n_h * n_w, 3, 2, 2))
                   .astype(np.float32))
        x = enc.vit(patches, n_h, n_w)
        self.assertEqual(x.shape, (n_h * n_w, conf.dim))
        self.assertTrue(np.isfinite(x).all())
        aligned = enc.align(x, n_h, n_w)
        # 2x3 patch grid, downsample 2, padded to even -> (1 rows, 2 cols)
        self.assertEqual(aligned.shape[0], 2)
        self.assertEqual(aligned.shape[1], 32)
        self.assertTrue(np.isfinite(aligned).all())

    def test_bf16_floor_stays_close(self):
        """bf16 storage simulation must stay within the documented float floor
        of the fp32 oracle (the target a BF16 GEMM engine must meet)."""
        conf = VisionCfg(dim=32, n_layers=2, n_heads=4, inter_dim=16,
                         patch_size=2, downsample_ratio=2)
        w = _vision_weights(conf, llm_dim=32)
        n_h, n_w = 2, 4
        patches = (np.random.default_rng(2).random((n_h * n_w, 3, 2, 2))
                   .astype(np.float32))
        a = DeepseekVisionEncoder(w, conf).vit(patches, n_h, n_w)
        b = DeepseekVisionEncoder(w, conf, bf16=True).vit(patches, n_h, n_w)
        rel = float(np.abs(b - a).std() / (a.std() + 1e-12))
        self.assertLess(rel, 0.1)
        al_a = DeepseekVisionEncoder(w, conf).align(a, n_h, n_w)
        al_b = DeepseekVisionEncoder(w, conf, bf16=True).align(b, n_h, n_w)
        self.assertTrue(np.isfinite(al_b).all())
        self.assertLess(float(np.abs(al_b - al_a).std() / (al_a.std() + 1e-12)),
                        0.2)


class TestMultimodalModel(unittest.TestCase):
    def test_multimodal_prefill(self):
        cfg = tiny_deepseek_v4_cfg()
        w = tiny_deepseek_v4_weights(cfg)
        vconf = VisionCfg(dim=32, n_layers=1, n_heads=4, inter_dim=16,
                          patch_size=2, downsample_ratio=2)
        w.update(_vision_weights(vconf, llm_dim=cfg.dim))
        cfg.vision = True
        m = DeepseekV4Model(w, cfg, vision_cfg=vconf)
        self.assertIsNotNone(m.vision)
        args = ip.VisionArgs(patch_size=2, downsample_ratio=2,
                             max_n_token=384, min_pixels=0, max_wh_ratio=None)
        codes, inputs = ip.expand_image_placeholders(
            [3, 5, 64, 7], [{"data": _tiny_rgb_png(10, 8)}], args=args,
            vocab_size=64, image_placeholder_id=64)
        state, lg = m.prefill(codes, images=inputs)
        self.assertEqual(lg.shape[1], cfg.vocab_size)
        self.assertTrue(np.isfinite(lg).all())
        # decode continues (text prompt after the image block)
        lr = m.decode_step(state, 9)
        self.assertEqual(lr.shape, (cfg.vocab_size,))
        self.assertTrue(np.isfinite(lr).all())


class TestEncodingAdoption(unittest.TestCase):
    """serving.encoding_dsv4 is the adopted reference encoder (OpenAI
    message blocks -> prompt text + image records)."""

    def test_process_image_messages(self):
        from serving import encoding_dsv4 as enc
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ]},
        ]
        processed, images = enc.process_image_messages(messages)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["url"], "https://x/y.png")
        self.assertEqual(processed[0]["content_blocks"][-1]["type"], "text")
        self.assertIn(enc.IMAGE_PLACEHOLDER, processed[0]["content"])

    def test_encode_messages_multimodal(self):
        import base64
        from serving import encoding_dsv4 as enc
        b64 = base64.b64encode(_tiny_rgb_png(10, 8)).decode()
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ]
        prompt, media = enc.encode_messages(messages, "chat",
                                            return_multi_modal_data=True)
        self.assertIn(enc.USER_SP_TOKEN, prompt)
        self.assertIn(enc.IMAGE_PLACEHOLDER, prompt)
        self.assertEqual(len(media["images"]), 1)


def _data_uri_png(w=10, h=8) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(_tiny_rgb_png(w, h)).decode()


class TestVLServerWire(unittest.TestCase):
    def test_vl_engine_chat_via_content_blocks(self):
        from serving.dev_model import build_dev_deepseek_v4_vl_engine
        eng = build_dev_deepseek_v4_vl_engine()
        d = eng.chat_detail([
            {"role": "user", "content": [
                {"type": "text", "text": "what is in the image?"},
                {"type": "image_url", "image_url": {"url": _data_uri_png()}},
            ]},
        ], max_new=5, temperature=0.0)
        self.assertIn("n_images", d)
        self.assertEqual(d["n_images"], 1)
        self.assertIsInstance(d["text"], str)
        self.assertIn("finish_reason", d)

    def test_vl_engine_text_only_routing(self):
        from serving.dev_model import build_dev_deepseek_v4_vl_engine
        eng = build_dev_deepseek_v4_vl_engine()
        d = eng.chat_detail([{"role": "user", "content": "hi"}],
                            max_new=4, temperature=0.0)
        # no image blocks -> text path (apply_chat_template on the VL tokenizer)
        self.assertIn("prompt_text", d)
        self.assertIsInstance(d["text"], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
