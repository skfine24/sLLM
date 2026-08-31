"""Recipe schema v1.

A recipe maps a model checkpoint to the engine: architecture knobs, layer-type
schedule, kernel choices, FP8 layout, tokenizer, and TP sharding plan. The
engine executes a recipe; it does not hard-code any model.

Parsing is torch-free (pure stdlib + PyYAML), so recipes can be validated on
the dev machine and inside the cluster container alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


class RecipeError(ValueError):
    """Raised when a recipe document is invalid."""


# Executable block types (engine runs these). Additional structural markers
# (qsa_attention / mla_attention) are permitted for skeleton recipes whose
# kernels are not implemented yet.
KNOWN_LAYER_TYPES = ("linear_attention", "full_attention", "qsa_attention", "mla_attention")


def _require(d: dict, key: str, typename: str) -> Any:
    if key not in d:
        raise RecipeError(f"missing required key: {key!r}")
    v = d[key]
    if typename == "int" and not isinstance(v, int):
        raise RecipeError(f"key {key!r} must be int, got {type(v).__name__}")
    if typename == "str" and not isinstance(v, str):
        raise RecipeError(f"key {key!r} must be str, got {type(v).__name__}")
    if typename == "bool" and not isinstance(v, bool):
        raise RecipeError(f"key {key!r} must be bool, got {type(v).__name__}")
    return v


@dataclass
class QuantSpec:
    method: str = "fp8"
    fmt: str = "e4m3"
    activation: str = "dynamic"
    weight_block_size: tuple[int, int] = (128, 128)
    scale_tensor_suffix: str = "weight_scale_inv"
    modules_not_quantize: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict) -> "QuantSpec":
        spec = cls()
        if not isinstance(d, dict):
            raise RecipeError("quant must be a mapping")
        spec.method = d.get("method", spec.method)
        spec.fmt = d.get("fmt", spec.fmt)
        spec.activation = d.get("activation", spec.activation)
        wbs = d.get("weight_block_size", list(spec.weight_block_size))
        if not (isinstance(wbs, list) and len(wbs) == 2 and all(isinstance(x, int) for x in wbs)):
            raise RecipeError("quant.weight_block_size must be [ow, ic] of ints")
        spec.weight_block_size = (wbs[0], wbs[1])
        spec.scale_tensor_suffix = d.get("scale_tensor_suffix", spec.scale_tensor_suffix)
        spec.modules_not_quantize = tuple(d.get("modules_not_quantize", []))
        return spec


@dataclass
class RopeSpec:
    type: str = "mrope"
    theta: float = 1e7
    mrope_section: tuple[int, ...] = (11, 11, 10)
    mrope_interleaved: bool = True
    partial_rotary_factor: float = 0.25

    @classmethod
    def from_dict(cls, d: dict) -> "RopeSpec":
        spec = cls()
        spec.type = d.get("type", spec.type)
        spec.theta = d.get("theta", spec.theta)
        spec.mrope_section = tuple(d.get("mrope_section", list(spec.mrope_section)))
        spec.mrope_interleaved = d.get("mrope_interleaved", spec.mrope_interleaved)
        spec.partial_rotary_factor = d.get("partial_rotary_factor", spec.partial_rotary_factor)
        return spec


@dataclass
class LinearAttentionSpec:
    kernel: str = "gated_delta_rule"
    num_key_heads: int = 16
    key_head_dim: int = 128
    num_value_heads: int = 48
    value_head_dim: int = 128
    conv_kernel_size: int = 4
    state_dtype: str = "float32"
    qk_l2norm: bool = True
    q_scaling: str = "head_k_dim^-0.5"

    @classmethod
    def from_dict(cls, d: dict) -> "LinearAttentionSpec":
        s = cls()
        for k in ("kernel", "state_dtype", "q_scaling"):
            setattr(s, k, d.get(k, getattr(s, k)))
        for k in ("num_key_heads", "key_head_dim", "num_value_heads", "value_head_dim", "conv_kernel_size"):
            setattr(s, k, _require(d, k, "int") if k in d else getattr(s, k))
        s.qk_l2norm = d.get("qk_l2norm", s.qk_l2norm)
        return s


@dataclass
class FullAttentionSpec:
    kernel: str = "paged_flash"
    num_heads: int = 24
    num_kv_heads: int = 4
    head_dim: int | None = None       # None -> derived from hidden_size // num_heads
    rope: RopeSpec = field(default_factory=RopeSpec)
    output_gate: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "FullAttentionSpec":
        s = cls()
        s.kernel = d.get("kernel", s.kernel)
        if "num_heads" in d:
            s.num_heads = _require(d, "num_heads", "int")
        if "num_kv_heads" in d:
            s.num_kv_heads = _require(d, "num_kv_heads", "int")
        if "head_dim" in d:
            s.head_dim = _require(d, "head_dim", "int")
        s.output_gate = d.get("output_gate", s.output_gate)
        if "rope" in d:
            s.rope = RopeSpec.from_dict(d["rope"])
        return s

    def effective_head_dim(self, hidden_size: int) -> int:
        return self.head_dim if self.head_dim is not None else hidden_size // self.num_heads


@dataclass
class MLPSpec:
    type: str = "dense"  # "dense" (gate/up/down) | "moe" (routed + shared)
    intermediate_size: int = 17408
    hidden_act: str = "silu"
    num_experts: int = 0
    num_experts_per_tok: int = 0
    shared_experts: int = 0
    shared_expert_intermediate_size: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "MLPSpec":
        s = cls()
        s.type = d.get("type", s.type)
        if s.type not in ("dense", "moe"):
            raise RecipeError(f"unsupported mlp.type {s.type!r}")
        if "intermediate_size" in d:
            s.intermediate_size = _require(d, "intermediate_size", "int")
        s.hidden_act = d.get("hidden_act", s.hidden_act)
        for k in ("num_experts", "num_experts_per_tok", "shared_experts",
                  "shared_expert_intermediate_size"):
            if k in d:
                setattr(s, k, _require(d, k, "int"))
        return s


@dataclass
class MTPSpec:
    enabled: bool = True
    layers: int = 1
    attention_type: str = "full"  # audited: mtp layer 0 uses self_attn (full)
    has_fc: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "MTPSpec":
        s = cls()
        s.enabled = d.get("enabled", s.enabled)
        s.layers = d.get("layers", s.layers)
        s.attention_type = d.get("attention_type", s.attention_type)
        s.has_fc = d.get("has_fc", s.has_fc)
        return s


@dataclass
class VisionSpec:
    enabled: bool = False
    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    intermediate_size: int = 4304
    num_position_embeddings: int = 2304
    out_hidden_size: int = 5120

    @classmethod
    def from_dict(cls, d: dict) -> "VisionSpec":
        s = cls()
        if not isinstance(d, dict):
            return s
        s.enabled = d.get("enabled", s.enabled)
        for k in ("depth", "hidden_size", "num_heads", "patch_size", "temporal_patch_size",
                  "spatial_merge_size", "intermediate_size", "num_position_embeddings", "out_hidden_size"):
            if k in d:
                setattr(s, k, _require(d, k, "int"))
        return s


@dataclass
class MemorySpec:
    """Runtime memory placement options (selection applies on GPU-backed runs;
    the CPU/numpy path always uses host RAM, so `kv_placement` is a no-op there).

    - kv_placement "device": conventional all-on-GPU (weights + compute + KV).
      Safe bound enforced by the planned KV budget (admission rejects overflow);
      on GB10, over-subscribing device memory can hang the node until power
      off (see docs/design/03).
    - kv_placement "host": KV / recurrent state live in host RAM; the decode
      gathers only the touched block(s) per step. OOM degrades to a recoverable
      process-level failure / swap instead of a system hang.
    """

    kv_placement: str = "device"          # "device" | "host"
    kv_host_bytes: int | None = None      # "host" mode budget cap (None -> auto)
    kv_device_bytes: int | None = None    # "device" mode budget cap (None -> auto)
    kv_utilization: float = 0.9           # fraction of budget used for blocks

    KV_PLACEMENTS = ("device", "host")

    @classmethod
    def from_dict(cls, d: dict | None) -> "MemorySpec":
        s = cls()
        if not isinstance(d, dict):
            return s
        p = d.get("kv_placement", s.kv_placement)
        if p not in cls.KV_PLACEMENTS:
            raise RecipeError(
                f"memory.kv_placement must be one of {cls.KV_PLACEMENTS}, got {p!r}")
        s.kv_placement = p
        for k in ("kv_host_bytes", "kv_device_bytes"):
            if k in d:
                setattr(s, k, _require(d, k, "int"))
        if "kv_utilization" in d:
            u = d["kv_utilization"]
            if not isinstance(u, (int, float)) or not (0 < u <= 1):
                raise RecipeError("memory.kv_utilization must be in (0, 1]")
            s.kv_utilization = float(u)
        return s


@dataclass
class TPSpec:
    size: int = 1
    shard_axes: dict = field(default_factory=lambda: {
        "column_parallel": ["qkv", "gate_up", "o_defer"],
        "row_parallel": ["o_proj", "down_proj"],
    })

    @classmethod
    def from_dict(cls, d: dict) -> "TPSpec":
        s = cls()
        if "size" in d:
            s.size = _require(d, "size", "int")
        if "shard_axes" in d:
            s.shard_axes = d["shard_axes"]
        return s


@dataclass
class Recipe:
    model_id: str
    arch: str
    dtype: str
    quant: QuantSpec
    hidden_size: int
    num_layers: int
    layer_types: list[str]
    linear_attention: LinearAttentionSpec
    full_attention: FullAttentionSpec
    mlp: MLPSpec
    mtp: MTPSpec
    vision: VisionSpec
    tp: TPSpec
    memory: MemorySpec
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    full_attention_interval: int | None = None
    text_prefix: str = "model.language_model"
    tie_word_embeddings: bool = False
    status: str = "ready"               # "ready" | "skeleton"
    meta: dict = field(default_factory=dict)  # passthrough of unknown/extra keys
    paths: dict = field(default_factory=dict)  # deploy paths (e.g. local_dir)
    # ---- launch section (recipe = model + how to launch it) ----------------
    recipe_version: str = "1"
    name: str = ""
    description: str = ""
    container: str | None = None        # image override (default sllm-node)
    defaults: dict = field(default_factory=dict)   # launcher defaults (CLI wins)
    launch_env: dict = field(default_factory=dict)  # env injected into container
    command: str | None = None          # command template ({key} from defaults)

    _KNOWN_TOP = frozenset({
        "model_id", "arch", "dtype", "quant", "text", "mtp", "vision", "tp",
        "memory", "status", "paths",
        "recipe_version", "name", "description", "container", "defaults",
        "env", "command",
    })

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        model_id = _require(d, "model_id", "str")
        arch = _require(d, "arch", "str")
        dtype = d.get("dtype", "bfloat16")
        quant = QuantSpec.from_dict(d.get("quant", {}))
        text = d.get("text", {})
        if not isinstance(text, dict):
            raise RecipeError("text must be a mapping")
        hidden_size = text.get("hidden_size", 5120)
        num_layers = text.get("num_layers", 64)
        layer_types = list(text.get("layer_types", []))
        if not layer_types:
            full_interval = text.get("full_attention_interval")
            if full_interval:
                layer_types = [
                    "linear_attention" if (i + 1) % full_interval else "full_attention"
                    for i in range(num_layers)
                ]
        if layer_types and len(layer_types) != num_layers:
            raise RecipeError(f"layer_types length {len(layer_types)} != num_layers {num_layers}")
        known = set(KNOWN_LAYER_TYPES)
        for lt in layer_types:
            if lt not in known:
                raise RecipeError(f"unknown layer_type {lt!r}")

        defaults = d.get("defaults", {})
        if not isinstance(defaults, dict):
            raise RecipeError("defaults must be a mapping")
        launch_env = d.get("env", {})
        if not isinstance(launch_env, dict):
            raise RecipeError("env must be a mapping")
        tp_spec = TPSpec.from_dict(d.get("tp", {}))
        if "tensor_parallel" in defaults and \
                int(defaults["tensor_parallel"]) != tp_spec.size:
            raise RecipeError(
                f"defaults.tensor_parallel="
                f"{defaults['tensor_parallel']} != tp.size={tp_spec.size}")
        if "nodes" in defaults and int(defaults["nodes"]) < tp_spec.size:
            raise RecipeError(
                f"defaults.nodes={defaults['nodes']} < tp.size={tp_spec.size}")

        return cls(
            model_id=model_id,
            arch=arch,
            dtype=dtype,
            quant=quant,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_types=layer_types,
            linear_attention=LinearAttentionSpec.from_dict(text.get("linear_attention", {})),
            full_attention=FullAttentionSpec.from_dict(text.get("full_attention", {})),
            mlp=MLPSpec.from_dict(text.get("mlp", {})),
            mtp=MTPSpec.from_dict(d.get("mtp", {})),
            vision=VisionSpec.from_dict(d.get("vision", {})),
            tp=tp_spec,
            memory=MemorySpec.from_dict(d.get("memory", {})),
            vocab_size=text.get("vocab_size", 248320),
            max_position_embeddings=text.get("max_position_embeddings", 262144),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            full_attention_interval=text.get("full_attention_interval"),
            text_prefix=text.get("prefix", "model.language_model"),
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            status=d.get("status", "ready"),
            meta={k: v for k, v in d.items() if k not in cls._KNOWN_TOP},
            paths=d.get("paths", {}),
            recipe_version=str(d.get("recipe_version", "1")),
            name=d.get("name", "") or model_id,
            description=d.get("description", ""),
            container=d.get("container"),
            defaults=defaults,
            launch_env={k: str(v) for k, v in launch_env.items()},
            command=d.get("command"),
        )

    def render_command(self, **extra) -> str:
        """Fill the command template from defaults + extra vars (e.g. recipe
        path, resolved port). Unresolved placeholders -> RecipeError."""
        if not self.command:
            raise RecipeError("recipe has no command template")
        vals = {**self.defaults, **extra}
        try:
            return self.command.format(**vals).strip()
        except KeyError as exc:
            raise RecipeError(f"command template missing {exc}") from exc

    @property
    def local_dir(self) -> str | None:
        """Deploy-side local checkpoint directory (may use '~' for $HOME)."""
        return self.paths.get("local_dir")

    @classmethod
    def from_yaml(cls, text: str) -> "Recipe":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RecipeError("PyYAML is required to load a recipe") from exc
        doc = yaml.safe_load(text)
        if not isinstance(doc, dict):
            raise RecipeError("recipe must be a YAML mapping")
        return cls.from_dict(doc)

    # -- derived helpers ---------------------------------------------------

    def linear_attn_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    def full_attn_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    def rotary_dim(self) -> int:
        head_dim = self.full_attention.effective_head_dim(self.hidden_size)
        return int(math.floor(head_dim * self.full_attention.rope.partial_rotary_factor))
