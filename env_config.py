"""`config.env` reader: SLLM COMMON environment info (cluster nodes,
toolchain, shared assets). Model identity/geometry/weights-location belong to
`recipes/*.yaml` — do not add model keys here.

Precedence: real OS environment variable (non-empty) > config.env value >
caller default. Point at another file with SLLM_ENV_FILE.
"""

from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


def env_file() -> str:
    return os.environ.get("SLLM_ENV_FILE") or os.path.join(_ROOT, "config.env")


def parse(path: str | None = None) -> dict[str, str]:
    """Minimal KEY=VALUE parser (# comments, optional `export `, empty value
    = unset, later lines win). Missing file -> {}."""
    out: dict[str, str] = {}
    p = path or env_file()
    if not os.path.isfile(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k:
                out[k] = v
    return out


def get(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    if v:
        return v
    v = parse().get(key)
    return v if v else default


def get_path(key: str, default: str | None = None) -> str | None:
    v = get(key, default)
    return os.path.expanduser(v) if v else v


def get_int(key: str, default: int) -> int:
    v = get(key)
    return int(v) if v else default
