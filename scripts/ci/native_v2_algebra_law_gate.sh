#!/usr/bin/env bash
# Gate: algebra law enforcement -- reassoc_strategy decisions for the Cayley-Dickson tower.
# Runs the lean modular compiler algebra-law self-test and asserts T1201 passes.
# T1201 verifies that ir_algebra_info_for_tag returns the correct reassoc_strategy for each level:
#   Quaternion: strategy 0 (free), Octonion: strategy 2 (fano_selective), Sedenion: strategy 1 (blocked).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-bin/souc}"

echo "[algebra-law-gate] running lean compiler algebra-law self-test for T1201..."
OUTPUT=$("$SOUC" run self-hosted/compiler/lean.sio -- --algebra-law-self-test 2>&1 || true)

if echo "$OUTPUT" | grep -q "T1201 OK"; then
    echo "[algebra-law-gate] PASS: T1201 algebra law enforcement -- reassoc_strategy for CD tower"
else
    echo "[algebra-law-gate] FAIL: T1201 not found in self-test output"
    echo "$OUTPUT" | grep -E "T1201|FAIL" || true
    exit 1
fi
