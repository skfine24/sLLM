"""sLLM version + git build stamp (shown in the startup banner / --version).

`VERSION` is the release semver. `version_string()` appends the short git
revision when the repo is available, so debug output pinpoints the exact
checkout (mirrors how vLLM prints its version banner).
"""

from __future__ import annotations

VERSION = "0.1.0"


def _git_rev() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - not a git checkout / no git
        return ""


def version_string() -> str:
    rev = _git_rev()
    return f"{VERSION} ({rev})" if rev else VERSION
