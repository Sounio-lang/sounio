#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-bin/souc}"
SOURCE="tools/pireus/cross_arch_operator_synthesizer.sio"
FROZEN="tools/pireus/cross_arch_candidates.values.v1"
EXPECTED_SOURCE_HASH="48d69e53a6f115ba79d78b2544ce0d0c7f36d3ef8fbad18fea14fe9160d53757"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
fail() { printf 'PIREUS_CROSS_ARCH_SYNTHESIZER_GATE_FAIL: %s\n' "$*" >&2; exit 1; }
actual_hash="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
[ "$actual_hash" = "$EXPECTED_SOURCE_HASH" ] || fail "Sounio synthesizer changed without a new freeze"
"$SOUC" run "$SOURCE" >"$TMP_DIR/run.log" 2>&1 || fail "Sounio synthesizer execution failed"
sed -n '/^schema=pireus-cross-arch-candidates-v1$/,$p' "$TMP_DIR/run.log" >"$TMP_DIR/values"
cmp -s "$FROZEN" "$TMP_DIR/values" || fail "candidate freeze diverges from Sounio execution"
sed 's/native_f64: false/native_f64: true/' "$SOURCE" >"$TMP_DIR/forged.sio"
if "$SOUC" run "$TMP_DIR/forged.sio" >"$TMP_DIR/forged.log" 2>&1; then
    fail "forged Metal native-f64 fact was accepted"
fi
grep -q '^decision=apple-silicon-metal/EXACT DENY ' "$FROZEN" || fail "Metal exact refusal is missing"
grep -q 'candidate=xeon-avx512/two-zmm-indexed-permute equivalence=EXACT' "$FROZEN" || fail "Xeon AVX-512 candidate is missing"
grep -q 'candidate=dgx-sm121/warp-shuffle equivalence=EXACT' "$FROZEN" || fail "DGX sm_121 candidate is missing"
printf 'PIREUS_CROSS_ARCH_SYNTHESIZER_GATE_PASS\n'
