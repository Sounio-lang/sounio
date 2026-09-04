#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-bin/souc}"
SOURCE="tools/pireus/xeon_avx512_xor_plan.sio"
FROZEN="tools/pireus/xeon_avx512_xor_plan.values.v1"
EXPECTED_SOURCE_HASH="d90943bff2dbee3862772e43867eb09f1f27692296ded253f366ebc5ca51f9ad"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
fail() { printf 'PIREUS_XEON_AVX512_XOR_PLAN_GATE_FAIL: %s\n' "$*" >&2; exit 1; }

actual_hash="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
[ "$actual_hash" = "$EXPECTED_SOURCE_HASH" ] || fail "Sounio Xeon plan changed without a new freeze"
"$SOUC" run "$SOURCE" >"$TMP_DIR/run.log" 2>&1 || fail "Sounio Xeon plan execution failed"
sed -n '/^schema=pireus-xeon-avx512-xor-plan-v1$/,$p' "$TMP_DIR/run.log" >"$TMP_DIR/values"
cmp -s "$FROZEN" "$TMP_DIR/values" || fail "Xeon plan freeze diverges from Sounio execution"

# Remove the bit-3 half swap. Eight shifts must then disagree with authority.
sed 's/let source_half = out_half \^ swap/let source_half = out_half/' "$SOURCE" >"$TMP_DIR/forged.sio"
if "$SOUC" run "$TMP_DIR/forged.sio" >"$TMP_DIR/forged.log" 2>&1; then
    fail "forged no-half-swap plan was accepted"
fi
grep -q '^selection_failures=[1-9]' "$TMP_DIR/forged.log" || fail "negative control did not expose selection failures"
grep -q '^vpermpd_total=32$' "$FROZEN" || fail "two-ZMM permute count is not frozen"
grep -q '^negative_mask_bits=120$' "$FROZEN" || fail "Cayley-Dickson sign population is not preserved"
grep -q '^PIREUS_XEON_AVX512_XOR_PLAN_PASS$' "$FROZEN" || fail "Sounio authority did not pass"
printf 'PIREUS_XEON_AVX512_XOR_PLAN_GATE_PASS\n'
