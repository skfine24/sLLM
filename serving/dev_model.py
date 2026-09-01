"""Dev-machine tiny model + engine, used by the serving stub (CLI/HTTP) and by
tests. Same checkpoint-shaped weight dictionary and recipe as the real
Qwen3.8-27B-FP8, but with tiny dimensions so full wiring runs anywhere.
"""

from __future__ import annotations


import numpy as np

from recipes.schema import Recipe

HIDDEN = 24
# Match the real checkpoint's vocab so the REAL Qwen tokenizer ids fit the
# embedding row-space (embed_tokens/lm_head ~24M float32 rows each).
VOCAB = 248320
EPS = 1e-6
LAYER_TYPES = ["linear_attention", "full_attention"]

# Q27B_TOKENIZER_DIR env > config.env (shared-asset section) > dev fallback.
from env_config import get_path as _env_path  # noqa: E402

DEFAULT_TOKENIZER_DIR = _env_path(
    "Q27B_TOKENIZER_DIR",
    r"C:\Users\skfin\AppData\Local\Temp\opencode\qwen27b_tok",
)


def tiny_recipe() -> Recipe:
    return Recipe.from_dict({
        "model_id": "tiny/qwen3_5", "arch": "qwen3_5", "dtype": "bfloat16",
        "text": {
            "hidden_size": HIDDEN,
            "num_layers": len(LAYER_TYPES),
            "layer_types": LAYER_TYPES,
            "vocab_size": VOCAB,
            "max_position_embeddings": 512,
            "rms_norm_eps": EPS,
            "linear_attention": {
                "num_key_heads": 2, "key_head_dim": 8,
                "num_value_heads": 4, "value_head_dim": 8,
                "conv_kernel_size": 4, "state_dtype": "float32", "qk_l2norm": True,
            },
            "full_attention": {
                "num_heads": 3, "num_kv_heads": 1, "head_dim": 8,
                "output_gate": True,
                "rope": {"type": "mrope", "theta": 1e4, "mrope_section": [2, 2, 2],
                         "mrope_interleaved": True, "partial_rotary_factor": 0.25},
            },
            "mlp": {"type": "dense", "intermediate_size": 16, "hidden_act": "silu"},
        },
        "mtp": {"enabled": True, "layers": 1, "attention_type": "full", "has_fc": True},
        "vision": {"enabled": False},
        "tp": {"size": 2},
    })


def tiny_weights(rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(42)
    d = {}
    d["model.language_model.embed_tokens.weight"] = rng.standard_normal((VOCAB, HIDDEN), dtype=np.float32) * 0.05
    d["model.language_model.norm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d["lm_head.weight"] = rng.standard_normal((VOCAB, HIDDEN), dtype=np.float32) * 0.05

    i = 0
    p = f"model.language_model.layers.{i}.linear_attn"
    d["model.language_model.layers.0.input_layernorm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d["model.language_model.layers.0.post_attention_layernorm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d[f"{p}.in_proj_qkv.weight"] = np.float32(rng.standard_normal((64, HIDDEN)) * 0.1)
    d[f"{p}.conv1d.weight"] = np.float32(rng.standard_normal((64, 1, 4)) * 0.5)
    d[f"{p}.in_proj_z.weight"] = np.float32(rng.standard_normal((32, HIDDEN)) * 0.1)
    d[f"{p}.in_proj_b.weight"] = np.float32(rng.standard_normal((4, HIDDEN)) * 0.1)
    d[f"{p}.in_proj_a.weight"] = np.float32(rng.standard_normal((4, HIDDEN)) * 0.1)
    d[f"{p}.A_log"] = np.log(np.abs(np.float32(rng.standard_normal(4))) + 0.5)
    d[f"{p}.dt_bias"] = np.float32(rng.standard_normal(4)) * 0.1
    d[f"{p}.norm.weight"] = np.ones(8, dtype=np.float32)
    d[f"{p}.out_proj.weight"] = np.float32(rng.standard_normal((HIDDEN, 32)) * 0.1)

    i = 1
    p = f"model.language_model.layers.{i}.self_attn"
    d["model.language_model.layers.1.input_layernorm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d["model.language_model.layers.1.post_attention_layernorm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d[f"{p}.q_proj.weight"] = np.float32(rng.standard_normal((48, HIDDEN)) * 0.1)
    d[f"{p}.k_proj.weight"] = np.float32(rng.standard_normal((8, HIDDEN)) * 0.1)
    d[f"{p}.v_proj.weight"] = np.float32(rng.standard_normal((8, HIDDEN)) * 0.1)
    d[f"{p}.o_proj.weight"] = np.float32(rng.standard_normal((HIDDEN, 24)) * 0.1)
    d[f"{p}.q_norm.weight"] = np.zeros(8, dtype=np.float32)
    d[f"{p}.k_norm.weight"] = np.zeros(8, dtype=np.float32)

    for i in range(2):
        p = f"model.language_model.layers.{i}.mlp"
        d[f"{p}.gate_proj.weight"] = np.float32(rng.standard_normal((16, HIDDEN)) * 0.1)
        d[f"{p}.up_proj.weight"] = np.float32(rng.standard_normal((16, HIDDEN)) * 0.1)
        d[f"{p}.down_proj.weight"] = np.float32(rng.standard_normal((HIDDEN, 16)) * 0.1)

    # -- MTP (single full-attention layer + fc), shapes mirror the checkpoint
    mp = "mtp"
    d[f"{mp}.pre_fc_norm_embedding.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d[f"{mp}.pre_fc_norm_hidden.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d[f"{mp}.fc.weight"] = np.float32(rng.standard_normal((HIDDEN, 2 * HIDDEN)) * 0.1)
    d[f"{mp}.norm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    p = f"{mp}.layers.0"
    d[f"{p}.input_layernorm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d[f"{p}.post_attention_layernorm.weight"] = np.zeros(HIDDEN, dtype=np.float32)
    d[f"{p}.self_attn.q_proj.weight"] = np.float32(rng.standard_normal((48, HIDDEN)) * 0.1)
    d[f"{p}.self_attn.k_proj.weight"] = np.float32(rng.standard_normal((8, HIDDEN)) * 0.1)
    d[f"{p}.self_attn.v_proj.weight"] = np.float32(rng.standard_normal((8, HIDDEN)) * 0.1)
    d[f"{p}.self_attn.o_proj.weight"] = np.float32(rng.standard_normal((HIDDEN, 24)) * 0.1)
    d[f"{p}.self_attn.q_norm.weight"] = np.zeros(8, dtype=np.float32)
    d[f"{p}.self_attn.k_norm.weight"] = np.zeros(8, dtype=np.float32)
    d[f"{p}.mlp.gate_proj.weight"] = np.float32(rng.standard_normal((16, HIDDEN)) * 0.1)
    d[f"{p}.mlp.up_proj.weight"] = np.float32(rng.standard_normal((16, HIDDEN)) * 0.1)
    d[f"{p}.mlp.down_proj.weight"] = np.float32(rng.standard_normal((HIDDEN, 16)) * 0.1)
    return d


def build_dev_engine(tokenizer_dir: str = DEFAULT_TOKENIZER_DIR):
    """Tokenizer (real Qwen3.8-27B-FP8) + tiny numpy model, ready to serve."""
    from .executor import InferenceEngine, ReferenceModel
    from .tokenizer import Tokenizer

    tok = Tokenizer(tokenizer_dir)
    model = ReferenceModel(tiny_recipe(), tiny_weights())
    return InferenceEngine(model, tok)


# ---------------------------------------------------------------------------
# tiny STANDARD (Llama/Qwen2-style) model for tests
# ---------------------------------------------------------------------------

def tiny_standard_recipe() -> Recipe:
    return Recipe.from_dict({
        "model_id": "tiny/qwen2", "arch": "qwen2", "dtype": "bfloat16",
        "text": {
            "prefix": "model",
            "tie_word_embeddings": True,
            "hidden_size": 16,
            "num_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "full_attention": {
                "kernel": "standard_gqa",
                "num_heads": 4, "num_kv_heads": 2, "head_dim": 4,
                "output_gate": False,
                "rope": {"type": "default", "theta": 1e4,
                         "partial_rotary_factor": 1.0},
            },
            "mlp": {"type": "dense", "intermediate_size": 32, "hidden_act": "silu"},
            "vocab_size": 64,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
        },
        "mtp": {"enabled": False},
        "vision": {"enabled": False},
        "tp": {"size": 2},
    })


def tiny_standard_weights(rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(3)
    H = 16
    d = {}
    d["model.embed_tokens.weight"] = rng.standard_normal((64, H), dtype=np.float32) * 0.1
    for i in range(2):
        p = f"model.layers.{i}"
        d[f"{p}.input_layernorm.weight"] = np.ones(H, dtype=np.float32)
        d[f"{p}.post_attention_layernorm.weight"] = np.ones(H, dtype=np.float32)
        a = f"{p}.self_attn"
        d[f"{a}.q_proj.weight"] = rng.standard_normal((16, H), dtype=np.float32) * 0.1
        d[f"{a}.k_proj.weight"] = rng.standard_normal((8, H), dtype=np.float32) * 0.1
        d[f"{a}.v_proj.weight"] = rng.standard_normal((8, H), dtype=np.float32) * 0.1
        # Qwen2-style qkv biases at a scale large enough (>>1e-2) that any
        # bias-indexing bug in a fused kernel (e.g. reusing head-0 bias for all
        # heads) produces logits errors far beyond test tolerance, yet small
        # enough to keep the tiny 64-vocab model from saturating attention.
        d[f"{a}.q_proj.bias"] = rng.standard_normal(16, dtype=np.float32) * 0.3
        d[f"{a}.k_proj.bias"] = rng.standard_normal(8, dtype=np.float32) * 0.3
        d[f"{a}.v_proj.bias"] = rng.standard_normal(8, dtype=np.float32) * 0.3
        d[f"{a}.o_proj.weight"] = rng.standard_normal((H, 16), dtype=np.float32) * 0.1
        m = f"{p}.mlp"
        d[f"{m}.gate_proj.weight"] = rng.standard_normal((32, H), dtype=np.float32) * 0.1
        d[f"{m}.up_proj.weight"] = rng.standard_normal((32, H), dtype=np.float32) * 0.1
        d[f"{m}.down_proj.weight"] = rng.standard_normal((H, 32), dtype=np.float32) * 0.1
    d["model.norm.weight"] = np.ones(H, dtype=np.float32)
    # lm_head omitted: tied to embed_tokens
    return d


def build_dev_standard_engine():
    """Tiny standard (qwen2-family) ReferenceModel, for tests and the
    `tiny/` standard-arch engine path (wrap in InferenceEngine with a
    tokenizer)."""
    from .executor import InferenceEngine, ReferenceModel

    model = ReferenceModel(tiny_standard_recipe(), tiny_standard_weights())
    return model


# ---------------------------------------------------------------------------
# tiny QWEN4_EXP (HC + GDN + QSA + MoE) model for tests (milestone Q2)
# ---------------------------------------------------------------------------

def tiny_qwen4_exp_recipe() -> Recipe:
    """Recipe twin of `tiny_qwen4_exp_cfg()` (milestone Q2 knobs at tiny size);
    `Qwen4ExpCfg.from_recipe` must reproduce the cfg from this document."""
    return Recipe.from_dict({
        "model_id": "tiny/qwen4_exp", "arch": "qwen4_exp", "dtype": "bfloat16",
        "status": "skeleton",
        "text": {
            "hidden_size": 16,
            "num_layers": 3,
            "layer_types": ["linear_attention", "linear_attention",
                            "qsa_attention"],
            "vocab_size": 32,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "linear_attention": {
                "kernel": "gated_delta_rule",
                "num_key_heads": 2, "key_head_dim": 4,
                "num_value_heads": 2, "value_head_dim": 4,
                "conv_kernel_size": 3, "state_dtype": "float32", "qk_l2norm": True,
            },
            "full_attention": {
                "kernel": "qsa_indexed",
                "num_heads": 2, "num_kv_heads": 1, "head_dim": 8,
                "output_gate": False,
                "rope": {"type": "mrope", "theta": 1e4,
                         "partial_rotary_factor": 0.5},
            },
            "mlp": {"type": "moe", "num_experts": 4, "num_experts_per_tok": 2,
                    "shared_experts": 1, "intermediate_size": 8,
                    "shared_expert_intermediate_size": 8, "hidden_act": "silu"},
        },
        "spec": {
            "qsa": {"indexer_n_heads": 2, "indexer_head_dim": 8,
                    "indexer_budget": 8, "indexer_compress_ratio": 2},
            "hc": {"hc_count": 2, "hc_lowrank": 4},
        },
        "mtp": {"enabled": False},
        "vision": {"enabled": False},
        "tp": {"size": 2},
    })


def tiny_qwen4_exp_cfg():
    """Qwen4ExpCfg with real-knob shapes shrunk; QSA budget covers tiny S."""
    from ref.qwen4_exp_pipeline import Qwen4ExpCfg

    return Qwen4ExpCfg(
        hidden=16, hc_count=2, hc_lowrank=4,
        layer_types=("linear_attention", "linear_attention", "full_attention"),
        rms_norm_eps=1e-6, rope_theta=1e4,
        lin_k_heads=2, lin_k_dim=4, lin_v_heads=2, lin_v_dim=4,
        lin_conv=3, lin_l2norm=True,
        attn_heads=2, attn_kv_heads=1, attn_head_dim=8, rotary_factor=0.5,
        idx_heads=2, idx_dim=8, idx_budget=8, idx_ratio=2,
        n_experts=4, top_k=2, moe_inter=8, shared_inter=8,
    )


def tiny_qwen4_exp_ple_cfg():
    """tiny cfg + one PLE layer (layer 2, 1-indexed) and its knobs.

    ngram_size=2, heads_per_ngram=2 => ngram_heads=2 => head_dim_per_ngram = 8//2
    = 4; primes after 15: {17,19} => padded ngram vocab 64 (divisible 64)."""
    cfg = tiny_qwen4_exp_cfg()
    cfg.ple_layer_ids = (2,)
    cfg.ple_embed_dim = 8
    cfg.ngram_size = 2
    cfg.heads_per_ngram = 2
    cfg.ngram_vocab_size_base = 16
    cfg.ple_conv_kernel_size = 3
    cfg.ple_make_divisible = 64
    cfg.seed = 0
    cfg.eos_token_id = 0
    cfg.vocab_size = 32
    return cfg


def _q4_emit_ple(d, rng, p, cfg, ple_index):
    """Emit the PLE layer weights under prefix `p` (`layers.{i}.ple.*`)."""
    C = cfg.hc_count * cfg.hidden
    ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    hdpn = cfg.ple_embed_dim // ngram_heads
    from ref.qwen4_exp_ple import NGramLayout
    layout = NGramLayout(cfg.vocab_size, cfg.ngram_size, cfg.heads_per_ngram,
                         cfg.ngram_vocab_size_base, ple_index,
                         cfg.ple_make_divisible, cfg.seed)
    q = f"{p}.ple"
    d[f"{q}.ple_embedding.ngram_embedding.weight"] = \
        rng.standard_normal((layout.padded_vocab_size, hdpn), np.float32) * 0.1
    d[f"{q}.key_proj.weight"] = \
        rng.standard_normal((C, cfg.ple_embed_dim), np.float32) * 0.1
    d[f"{q}.value_proj.weight"] = \
        rng.standard_normal((cfg.hidden, cfg.ple_embed_dim), np.float32) * 0.1
    d[f"{q}.conv1d.weight"] = \
        rng.standard_normal((C, 1, cfg.ple_conv_kernel_size), np.float32) * 0.1
    for n in ("norm_key", "norm_query", "norm_conv"):
        d[f"{q}.{n}.weight"] = \
            rng.standard_normal(C, dtype=np.float32) * 0.05



def tiny_qwen4_exp_weights(cfg, rng: np.random.Generator | None = None) -> dict:
    """Checkpoint-named weights (verified names, docs 09 sec.1) at tiny size."""
    rng = rng or np.random.default_rng(11)
    H, hc, lr = cfg.hidden, cfg.hc_count, cfg.hc_lowrank
    V = 32
    key_dim = cfg.lin_k_heads * cfg.lin_k_dim
    val_dim = cfg.lin_v_heads * cfg.lin_v_dim
    nh, kvh, hd, rot = (cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim,
                        cfg.rotary_dim)
    ih, idim = cfg.idx_heads, cfg.idx_dim

    d = {}
    d["model.language_model.embed_tokens.weight"] = \
        rng.standard_normal((V, H), dtype=np.float32) * 0.1
    d["lm_head.weight"] = rng.standard_normal((V, H), dtype=np.float32) * 0.1
    mp = "model.language_model.hyper_connection_mixer"
    d[f"{mp}.hc_norm.weight"] = rng.standard_normal(hc * H, dtype=np.float32) * 0.05
    d[f"{mp}.input_mix_weight_down.weight"] = \
        rng.standard_normal((lr, hc * H), dtype=np.float32) * 0.05
    d[f"{mp}.input_mix_weight_up.weight"] = \
        rng.standard_normal((hc * H, lr), dtype=np.float32) * 0.05

    for i, bt in enumerate(cfg.layer_types):
        _q4_emit_layer(d, rng, f"model.language_model.layers.{i}", cfg, bt)
        if i + 1 in cfg.ple_layer_ids:
            _q4_emit_ple(d, rng, f"model.language_model.layers.{i}", cfg,
                         cfg.ple_layer_ids.index(i + 1))
    return d


def tiny_qwen4_exp_mtp_weights(cfg, main: dict,
                               rng: np.random.Generator | None = None) -> dict:
    """MTP module weights over a main-model dict (checkpoint-style names:
    mtp.pre_fc_norm_*, mtp.fc_*, mtp.hyper_connection_mixer.*, mtp.layers.0.*
    = one full-attention QSA+MoE layer). Embed and lm_head are shared with
    the main model, as in oracle/upstream/sglang/qwen4_exp_mtp.py."""
    rng = rng or np.random.default_rng(17)
    d = dict(main)
    H, hc, lr = cfg.hidden, cfg.hc_count, cfg.hc_lowrank
    d["mtp.pre_fc_norm_embedding.weight"] = \
        rng.standard_normal(H, dtype=np.float32) * 0.05
    d["mtp.pre_fc_norm_hidden.weight"] = \
        rng.standard_normal(hc * H, dtype=np.float32) * 0.05
    d["mtp.fc_embedding.weight"] = \
        rng.standard_normal((H, H), dtype=np.float32) * 0.1
    d["mtp.fc_hidden.weight"] = \
        rng.standard_normal((H, H), dtype=np.float32) * 0.1
    _q4_emit_layer(d, rng, "mtp.layers.0", cfg, "full_attention")
    mp = "mtp.hyper_connection_mixer"
    d[f"{mp}.hc_norm.weight"] = rng.standard_normal(hc * H, dtype=np.float32) * 0.05
    d[f"{mp}.input_mix_weight_down.weight"] = \
        rng.standard_normal((lr, hc * H), dtype=np.float32) * 0.05
    d[f"{mp}.input_mix_weight_up.weight"] = \
        rng.standard_normal((hc * H, lr), dtype=np.float32) * 0.05
    return d


def _q4_emit_layer(d: dict, rng: np.random.Generator, p: str, cfg, bt: str):
    """Emit one decoder layer (HC pair + GDN/QSA + MoE) under prefix `p`."""
    H, hc, lr = cfg.hidden, cfg.hc_count, cfg.hc_lowrank
    key_dim = cfg.lin_k_heads * cfg.lin_k_dim
    val_dim = cfg.lin_v_heads * cfg.lin_v_dim
    nh, kvh, hd = cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
    ih, idim = cfg.idx_heads, cfg.idx_dim
    for hcname in ("attn_hyper_connection", "mlp_hyper_connection"):
        q = f"{p}.{hcname}"
        d[f"{q}.hc_norm.weight"] = \
            rng.standard_normal(hc * H, dtype=np.float32) * 0.05
        d[f"{q}.input_mix_weight_down.weight"] = \
            rng.standard_normal((lr, hc * H), dtype=np.float32) * 0.05
        d[f"{q}.input_mix_weight_up.weight"] = \
            rng.standard_normal((hc * H, lr), dtype=np.float32) * 0.05
        d[f"{q}.block_inject_weight.weight"] = \
            rng.standard_normal((hc, hc * H), dtype=np.float32) * 0.05
    if bt == "linear_attention":
        a = f"{p}.linear_attn"
        d[f"{a}.in_proj_qkv.weight"] = \
            rng.standard_normal((2 * key_dim + val_dim, H), dtype=np.float32) * 0.1
        d[f"{a}.conv1d.weight"] = \
            rng.standard_normal((2 * key_dim + val_dim, 1, cfg.lin_conv),
                                dtype=np.float32) * 0.5
        d[f"{a}.in_proj_z.weight"] = \
            rng.standard_normal((val_dim, H), dtype=np.float32) * 0.1
        d[f"{a}.in_proj_b.weight"] = \
            rng.standard_normal((cfg.lin_v_heads, H), dtype=np.float32) * 0.1
        d[f"{a}.in_proj_a.weight"] = \
            rng.standard_normal((cfg.lin_v_heads, H), dtype=np.float32) * 0.1
        d[f"{a}.A_log"] = np.log(np.abs(
            rng.standard_normal(cfg.lin_v_heads).astype(np.float32)) + 0.5)
        d[f"{a}.dt_bias"] = rng.standard_normal(cfg.lin_v_heads, dtype=np.float32) * 0.1
        d[f"{a}.norm.weight"] = np.ones(cfg.lin_v_dim, dtype=np.float32)
        d[f"{a}.out_proj.weight"] = \
            rng.standard_normal((H, val_dim), dtype=np.float32) * 0.1
    else:
        a = f"{p}.self_attn"
        d[f"{a}.q_proj.weight"] = \
            rng.standard_normal((nh * hd * 2, H), dtype=np.float32) * 0.1
        d[f"{a}.k_proj.weight"] = \
            rng.standard_normal((kvh * hd, H), dtype=np.float32) * 0.1
        d[f"{a}.v_proj.weight"] = \
            rng.standard_normal((kvh * hd, H), dtype=np.float32) * 0.1
        d[f"{a}.o_proj.weight"] = \
            rng.standard_normal((H, nh * hd), dtype=np.float32) * 0.1
        d[f"{a}.q_norm.weight"] = rng.standard_normal(hd, dtype=np.float32) * 0.05
        d[f"{a}.k_norm.weight"] = rng.standard_normal(hd, dtype=np.float32) * 0.05
        d[f"{a}.indexer.index_qk_proj.weight"] = \
            rng.standard_normal(((ih + 1) * idim, H), dtype=np.float32) * 0.1
        d[f"{a}.indexer.q_layernorm.weight"] = \
            rng.standard_normal(idim, dtype=np.float32) * 0.05
        d[f"{a}.indexer.k_layernorm.weight"] = \
            rng.standard_normal(idim, dtype=np.float32) * 0.05
    m = f"{p}.mlp"
    d[f"{m}.gate.weight"] = rng.standard_normal((cfg.n_experts, H), dtype=np.float32) * 0.1
    for e in range(cfg.n_experts):
        d[f"{m}.experts.{e}.gate_proj.weight"] = \
            rng.standard_normal((cfg.moe_inter, H), dtype=np.float32) * 0.1
        d[f"{m}.experts.{e}.up_proj.weight"] = \
            rng.standard_normal((cfg.moe_inter, H), dtype=np.float32) * 0.1
        d[f"{m}.experts.{e}.down_proj.weight"] = \
            rng.standard_normal((H, cfg.moe_inter), dtype=np.float32) * 0.1
    d[f"{m}.shared_expert.gate_proj.weight"] = \
        rng.standard_normal((cfg.shared_inter, H), dtype=np.float32) * 0.1
    d[f"{m}.shared_expert.up_proj.weight"] = \
        rng.standard_normal((cfg.shared_inter, H), dtype=np.float32) * 0.1
    d[f"{m}.shared_expert.down_proj.weight"] = \
        rng.standard_normal((H, cfg.shared_inter), dtype=np.float32) * 0.1
    d[f"{m}.shared_expert_gate.weight"] = \
        rng.standard_normal((1, H), dtype=np.float32) * 0.1


class TinyCharTokenizer:
    """Deterministic char-level stand-in for the tiny fixtures (V=32).

    The real Qwen tokenizer ids exceed the tiny embedding rows, so engine-level
    qwen4_exp dev runs use this stub: encode/decode are deterministic inverses
    on lowercase letters, `eos_id()` is None (generation runs to max_new).
    """

    def __init__(self, vocab: int = 32):
        self.vocab = vocab

    def eos_id(self):
        return None

    def encode(self, text: str) -> list[int]:
        return [1 + ((ord(c) - 97) % 26) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(97 + (int(i) - 1) % 26) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt: bool = True) -> str:
        return "".join(str(m.get("content", "")) for m in messages)


def build_dev_qwen4_exp_engine():
    """Tiny qwen4_exp model + stub tokenizer, ready to serve (dev/tests).

    The cfg is derived from the recipe (no explicit q4_cfg) so the CLI path
    exercises `Qwen4ExpCfg.from_recipe` end to end.
    """
    from .executor import InferenceEngine, ReferenceModel

    cfg = tiny_qwen4_exp_cfg()
    model = ReferenceModel(tiny_qwen4_exp_recipe(),
                           tiny_qwen4_exp_weights(cfg))
    return InferenceEngine(model, TinyCharTokenizer())


# ---------------------------------------------------------------------------
# DeepSeek-V4 tiny dev model (D+V track, L2)
# ---------------------------------------------------------------------------


def tiny_deepseek_v4_cfg():
    """DeepseekV4Cfg at tiny shapes covering both compress styles: layer 0 =
    ratio 4 (Compressor+Indexer overlap), layer 1 = pure sliding-window MLA."""
    from ref.deepseek_v4 import DeepseekV4Cfg

    c = DeepseekV4Cfg()
    c.vocab_size = 64
    c.dim = 32
    c.n_layers = 2
    c.n_heads = 4
    c.head_dim = 16
    c.rope_head_dim = 8
    c.window_size = 16
    c.compress_ratios = (4, 0)
    c.q_lora_rank = 16
    c.o_lora_rank = 16
    c.o_groups = 2
    c.n_routed_experts = 4
    c.n_activated_experts = 2
    c.n_shared_experts = 1
    c.moe_inter_dim = 16
    c.swiglu_limit = 5.0
    c.n_hash_layers = 0
    c.index_n_heads = 2
    c.index_head_dim = 16
    c.index_topk = 4
    c.hc_mult = 2
    c.hc_sinkhorn_iters = 4
    c.norm_eps = 1e-6
    c.max_seq_len = 1024
    c.dspark_target_layer_ids = (1,)
    c.n_mtp_layers = 2
    c.dspark_block_size = 3
    c.dspark_markov_rank = 4
    c.dspark_noise_token_id = 0
    c.original_seq_len = 0
    return c


def _tiny_r(*s):
    return (np.random.default_rng(7).standard_normal(s) * 0.05).astype(
        np.float32)


def tiny_deepseek_v4_weights(cfg, rng: np.random.Generator | None = None):
    """DeepSeek-V4 flat checkpoint-named weights at tiny size (fp32, no
    quant companions -- the oracle consumes dequantized fp32)."""
    rng = rng or np.random.default_rng(7)
    H, V = cfg.dim, cfg.vocab_size
    w = {}

    def r(*s, scale=0.05):
        return (rng.standard_normal(s) * scale).astype(np.float32)

    def flat(d, pre=""):
        for k, v in d.items():
            kk = pre + k
            if isinstance(v, dict):
                flat(v, kk + ".")
            else:
                w[kk] = v

    w["embed.weight"] = r(V, H)
    w["head.weight"] = r(V, H)
    w["norm.weight"] = np.ones(H, np.float32)
    w["hc_head_fn"] = r(cfg.hc_mult, cfg.hc_mult * H)
    w["hc_head_scale"] = np.ones(1, np.float32)
    w["hc_head_base"] = r(cfg.hc_mult)

    def attn(ratio):
        a = {}
        a["wq_a.weight"] = r(cfg.q_lora_rank, H, scale=0.1)
        a["q_norm.weight"] = r(cfg.q_lora_rank)
        a["wq_b.weight"] = r(cfg.n_heads * cfg.head_dim, cfg.q_lora_rank,
                             scale=0.1)
        a["wkv.weight"] = r(cfg.head_dim, H, scale=0.1)
        a["kv_norm.weight"] = r(cfg.head_dim)
        a["wo_a.weight"] = r(cfg.o_groups * cfg.o_lora_rank,
                             cfg.n_heads * cfg.head_dim // cfg.o_groups)
        a["wo_b.weight"] = r(H, cfg.o_groups * cfg.o_lora_rank, scale=0.1)
        a["attn_sink"] = r(cfg.n_heads)
        if ratio:
            a["compressor"] = {
                "wkv.weight": r(cfg.head_dim * (2 if ratio == 4 else 1), H),
                "wgate.weight": r(cfg.head_dim * (2 if ratio == 4 else 1), H),
                "ape": r(ratio, cfg.head_dim * (2 if ratio == 4 else 1)),
                "norm.weight": r(cfg.head_dim),
            }
            if ratio == 4:
                a["indexer"] = {
                    "wq_b.weight": r(cfg.index_n_heads * cfg.index_head_dim,
                                     cfg.q_lora_rank),
                    "weights_proj.weight": r(cfg.index_n_heads, H),
                    "compressor": {
                        "wkv.weight": r(cfg.index_head_dim * 2, H),
                        "wgate.weight": r(cfg.index_head_dim * 2, H),
                        "ape": r(ratio, cfg.index_head_dim * 2),
                        "norm.weight": r(cfg.index_head_dim),
                    },
                }
        return a

    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        ratio = cfg.compress_ratios[i]
        d = {p + "attn_norm.weight": r(H), p + "ffn_norm.weight": r(H)}
        for prefix in ("hc_attn", "hc_ffn"):
            d[p + prefix + "_fn"] = r((2 + cfg.hc_mult) * cfg.hc_mult,
                                      cfg.hc_mult * H)
            d[p + prefix + "_scale"] = r(3)
            d[p + prefix + "_base"] = r((2 + cfg.hc_mult) * cfg.hc_mult)
        d[p + "ffn.gate.weight"] = r(cfg.n_routed_experts, H)
        d[p + "ffn.gate.bias"] = r(cfg.n_routed_experts)
        for tag, outd, ind in (("w1", cfg.moe_inter_dim, H),
                               ("w3", cfg.moe_inter_dim, H),
                               ("w2", H, cfg.moe_inter_dim)):
            d[p + f"ffn.shared_experts.{tag}.weight"] = r(outd, ind)
        for e in range(cfg.n_routed_experts):
            for tag, outd, ind in (("w1", cfg.moe_inter_dim, H),
                                   ("w3", cfg.moe_inter_dim, H),
                                   ("w2", H, cfg.moe_inter_dim)):
                d[p + f"ffn.experts.{e}.{tag}.weight"] = r(outd, ind)
        flat(attn(ratio), p + "attn.")
        flat(d)
    return w


def tiny_deepseek_v4_spec_weights(cfg, main_w,
                                  rng: np.random.Generator | None = None):
    """Add the DSPark (`mtp.*`) weights over a tiny main-model dict. Embed and
    head are shared with the main model (reference Transformer wire-up)."""
    rng = rng or np.random.default_rng(19)
    H, V, hc = cfg.dim, cfg.vocab_size, cfg.hc_mult
    d = dict(main_w)
    T = max(len(cfg.dspark_target_layer_ids), 1)
    mrank = cfg.dspark_markov_rank

    def r(*s, scale=0.05):
        return (rng.standard_normal(s) * scale).astype(np.float32)

    def stage(sid):
        p = f"mtp.{sid}."
        out = {}
        a = {}
        a["wq_a.weight"] = r(cfg.q_lora_rank, H, scale=0.1)
        a["q_norm.weight"] = r(cfg.q_lora_rank)
        a["wq_b.weight"] = r(cfg.n_heads * cfg.head_dim, cfg.q_lora_rank,
                             scale=0.1)
        a["wkv.weight"] = r(cfg.head_dim, H, scale=0.1)
        a["kv_norm.weight"] = r(cfg.head_dim)
        a["wo_a.weight"] = r(cfg.o_groups * cfg.o_lora_rank,
                             cfg.n_heads * cfg.head_dim // cfg.o_groups)
        a["wo_b.weight"] = r(H, cfg.o_groups * cfg.o_lora_rank, scale=0.1)
        a["attn_sink"] = r(cfg.n_heads)
        for k, v in a.items():
            out[p + "attn." + k] = v
        out[p + "attn_norm.weight"] = r(H)
        out[p + "ffn_norm.weight"] = r(H)
        for prefix in ("hc_attn", "hc_ffn"):
            out[p + prefix + "_fn"] = r((2 + hc) * hc, hc * H)
            out[p + prefix + "_scale"] = r(3)
            out[p + prefix + "_base"] = r((2 + hc) * hc)
        out[p + "ffn.gate.weight"] = r(cfg.n_routed_experts, H)
        out[p + "ffn.gate.bias"] = r(cfg.n_routed_experts)
        for tag, outd, ind in (("w1", cfg.moe_inter_dim, H),
                               ("w3", cfg.moe_inter_dim, H),
                               ("w2", H, cfg.moe_inter_dim)):
            out[p + f"ffn.shared_experts.{tag}.weight"] = r(outd, ind)
        for e in range(cfg.n_routed_experts):
            for tag, outd, ind in (("w1", cfg.moe_inter_dim, H),
                                   ("w3", cfg.moe_inter_dim, H),
                                   ("w2", H, cfg.moe_inter_dim)):
                out[p + f"ffn.experts.{e}.{tag}.weight"] = r(outd, ind)
        return out

    d.update(stage(0))
    d.update(stage(1))
    if cfg.n_mtp_layers >= 2:
        d["mtp.0.main_proj.weight"] = r(H, T * H, scale=0.1)
        d["mtp.0.main_norm.weight"] = r(H)
        d["mtp.1.norm.weight"] = r(H)
        d["mtp.1.markov_head.markov_w1.weight"] = r(V, mrank, scale=0.1)
        d["mtp.1.markov_head.markov_w2.weight"] = r(V, mrank, scale=0.1)
        d["mtp.1.confidence_head.proj.weight"] = r(1, H + mrank, scale=0.1)
        d["mtp.1.hc_head_fn"] = r(hc, hc * H)
        d["mtp.1.hc_head_base"] = r(hc)
        d["mtp.1.hc_head_scale"] = np.ones(1, np.float32)
    return d


def tiny_deepseek_v4_recipe(vision: bool = False):
    from recipes.schema import Recipe

    return Recipe.from_dict({
        "model_id": "tiny/deepseek-v4-vision-exp-dev",
        "arch": "deepseek_v4",
        "dtype": "bfloat16",
        "status": "skeleton",
        "text": {
            "vocab_size": 64, "hidden_size": 32, "num_layers": 2,
            "num_heads": 4, "num_experts_per_tok": 2,
            "max_position_embeddings": 1024,
        },
        "spec": {"compress_ratios": [4, 0]},
        "mtp": {"enabled": False},
        "vision": {
            "enabled": vision,
            "dim": 32, "n_layers": 1, "heads": 4, "inter_dim": 16,
            "ds_patch_size": 2, "downsample_ratio": 2,
            "max_n_token": 384, "min_pixels": 0, "max_wh_ratio": 8.0,
        } if vision else {"enabled": False},
        "tp": {"size": 1},
    })


def tiny_deepseek_v4_vision_weights(llm_dim: int, rng=None):
    """Tiny SigLIP tower + aligner weights (sizes matched to the text cfg)."""
    from ref.vision_deepseek import VisionCfg
    conf = VisionCfg(dim=32, n_layers=1, n_heads=4, inter_dim=16,
                     patch_size=2, downsample_ratio=2)
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


class TinyVLTokenizer(TinyCharTokenizer):
    """Char stand-in that also maps the DeepSeek-V4 image placeholder literal
    to a reserved id inside the tiny vocab (63), mirroring the real tokenizer's
    image placeholder token id."""

    PLACEHOLDER = "<｜deepseek_image｜>"
    PLACEHOLDER_ID = 63

    def encode(self, text: str) -> list[int]:
        out = []
        for part in text.split(self.PLACEHOLDER):
            if out:  # a placeholder just finished
                out.append(self.PLACEHOLDER_ID)
            out.extend(TinyCharTokenizer.encode(self, part))
        if text.endswith(self.PLACEHOLDER):
            out.append(self.PLACEHOLDER_ID)
        return out

    def decode(self, ids) -> str:
        return TinyCharTokenizer.decode(self, ids)

    def image_token_id(self) -> int:
        return self.PLACEHOLDER_ID


def build_dev_deepseek_v4_engine():
    """Tiny deepseek_v4 oracle + stub tokenizer, ready to serve (dev/tests).
    The cfg is passed explicitly (the tiny geometry is not derivable from a
    skeleton-size recipe spec)."""
    from .executor import InferenceEngine, ReferenceModel

    cfg = tiny_deepseek_v4_cfg()

    model = ReferenceModel(tiny_deepseek_v4_recipe(),
                           tiny_deepseek_v4_weights(cfg), dsv4_cfg=cfg)
    return InferenceEngine(model, TinyCharTokenizer())


def build_dev_deepseek_v4_vl_engine():
    """Tiny deepseek_v4 + tiny SigLIP tower + VL tokenizer, ready to serve
    multimodal chat (dev/tests)."""
    from .executor import InferenceEngine, ReferenceModel
    from ref.vision_deepseek import VisionCfg

    cfg = tiny_deepseek_v4_cfg()
    cfg.vision = True
    w = tiny_deepseek_v4_weights(cfg)
    w.update(tiny_deepseek_v4_vision_weights(cfg.dim))
    vconf = VisionCfg(dim=32, n_layers=1, n_heads=4, inter_dim=16,
                      patch_size=2, downsample_ratio=2)
    model = ReferenceModel(tiny_deepseek_v4_recipe(vision=True), w,
                           dsv4_cfg=cfg, vision_cfg=vconf)
    return InferenceEngine(model, TinyVLTokenizer())


