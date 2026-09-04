#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-bin/souc}"
# Byte-level imported native tests currently hit the same Madaros runtime
# segfault as the pre-existing test_e8.sio. Keep semantic execution explicit;
# the heavy remote compiler check below the lane remains on the default engine.
SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
export SOUNIO_SOUC_ENGINE
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

"$SOUC" run self-hosted/native/test_pireus_xor.sio >"$OUT" 2>&1
grep -q '^instructions=146 bytes=1004 vpermpd=48 sign_loads=32 vxorpd=34 fma=32$' "$OUT"
grep -q '^tests_passed=4/4$' "$OUT"
grep -q '^PIREUS_XEON_AVX512_LOWERING_PASS$' "$OUT"
env -u SOUNIO_SOUC_ENGINE bash scripts/ci/pireus_xeon_avx512_xor_plan_gate.sh
printf 'PIREUS_XEON_AVX512_LOWERING_GATE_PASS\n'
