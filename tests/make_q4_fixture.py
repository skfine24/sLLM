"""Build a tiny qwen4_exp safetensors fixture (plumbing for bench runs).

Usage: python tests/make_q4_fixture.py DIR
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _synth import write_q4_dev_fixture  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    write_q4_dev_fixture(sys.argv[1])
    print(f"fixture written -> {sys.argv[1]}")
