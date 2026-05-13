#!/usr/bin/env python3
"""
Phase 1 — Non-Associative Epistemic Connectomics
abide_fetch.py — thin wrapper around scripts/research/abide_preprocess.py.

Invokes the preprocessor to populate artifacts/research/abide/frames.bin.
Logs subject counts. Exits non-zero with a readable diagnostic on network or
auth failure so the pilot step skips cleanly.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREPROCESS = os.path.join(ROOT, "scripts", "research", "abide_preprocess.py")
FRAMES_BIN = os.path.join(ROOT, "artifacts", "research", "abide", "frames.bin")


def log(msg: str) -> None:
    print(f"[abide_fetch] {msg}", flush=True)


def header_counts(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        n_asd, n_td = struct.unpack("<qq", f.read(16))
    return n_asd, n_td


def main() -> int:
    if not os.path.exists(PREPROCESS):
        log(f"ERROR: preprocessor not found at {PREPROCESS}")
        return 2

    log(f"invoking {PREPROCESS}")
    try:
        result = subprocess.run(
            [sys.executable, PREPROCESS],
            cwd=ROOT,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log("ERROR: preprocessor timed out after 30 min")
        return 3
    except Exception as e:
        log(f"ERROR: failed to invoke preprocessor: {e}")
        return 4

    if result.returncode != 0:
        log(f"ERROR: preprocessor exited with status {result.returncode}")
        log("  likely network or data-availability issue — phase 1 pilot will skip")
        return result.returncode

    if not os.path.exists(FRAMES_BIN):
        log(f"ERROR: preprocessor returned 0 but {FRAMES_BIN} is missing")
        return 5

    try:
        n_asd, n_td = header_counts(FRAMES_BIN)
    except Exception as e:
        log(f"ERROR: cannot read frames.bin header: {e}")
        return 6

    size = os.path.getsize(FRAMES_BIN)
    expected = 16 + (n_asd + n_td) * 7 * 200 * 8
    log(f"frames.bin populated: n_asd={n_asd}, n_td={n_td}, size={size} bytes")
    if size != expected:
        log(f"  WARNING: size mismatch (expected {expected})")
    else:
        log("  size matches header ✓")

    if n_asd < 5 or n_td < 5:
        log(f"ERROR: need ≥5 subjects per group for pilot; got {n_asd}/{n_td}")
        return 7

    log("ready for pilot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
