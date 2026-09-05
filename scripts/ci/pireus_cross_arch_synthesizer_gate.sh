#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-bin/souc}"
SOURCE="tools/pireus/cross_arch_operator_synthesizer.sio"
FROZEN="tools/pireus/cross_arch_candidates.values.v1"
EXPECTED_SOURCE_HASH="d41bd7e395c56cb9f1460e4871a51364be65302718c6ff9e924faaa221ab4943"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
fail() { printf 'PIREUS_CROSS_ARCH_SYNTHESIZER_GATE_FAIL: %s\n' "$*" >&2; exit 1; }
actual_hash="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
[ "$actual_hash" = "$EXPECTED_SOURCE_HASH" ] || fail "Sounio synthesizer changed without a new freeze"
"$SOUC" run "$SOURCE" >"$TMP_DIR/run.log" 2>&1 || fail "Sounio synthesizer execution failed"
sed -n '/^schema=pireus-cross-arch-candidates-v2$/,$p' "$TMP_DIR/run.log" >"$TMP_DIR/values"
cmp -s "$FROZEN" "$TMP_DIR/values" || fail "candidate freeze diverges from Sounio execution"

if rg -q 'emit_dgx|emit_xeon|emit_apple|candidate=dgx|candidate=xeon|candidate=apple' "$SOURCE"; then
    fail "synthesizer contains target-coded candidate branches"
fi
grep -q '^ontology_order_invariant=1$' "$FROZEN" || fail "ontology order invariant is unproven"
grep -q '^selected=dgx-sm121/xor-shuffle+' "$FROZEN" || fail "DGX selection was not derived"
grep -q '^selected=xeon-avx512/indexed-select+' "$FROZEN" || fail "Xeon selection was not derived"
grep -q '^decision=apple-silicon-metal/EXACT DENY reason=no-native-f64$' "$FROZEN" || fail "Apple exact refusal is missing"

sed 's/P_INDEXED_SELECT | P_SIGN_XOR, 8, true/P_SIGN_XOR, 8, true/' "$SOURCE" >"$TMP_DIR/no-xeon-indexed.sio"
if "$SOUC" run "$TMP_DIR/no-xeon-indexed.sio" >"$TMP_DIR/no-xeon-indexed.log" 2>&1; then
    fail "removing Xeon indexed-select did not invalidate the frozen cardinality"
fi
sed 's/, 32, false)/, 32, true)/' "$SOURCE" >"$TMP_DIR/forged-apple-f64.sio"
if "$SOUC" run "$TMP_DIR/forged-apple-f64.sio" >"$TMP_DIR/forged-apple-f64.log" 2>&1; then
    fail "forged Apple native-f64 fact was accepted without a new freeze"
fi
sed '0,/EQ_APPROXIMATE, PROOF_APPROXIMATE_CONSTRUCTION/s//EQ_EXACT, PROOF_APPROXIMATE_CONSTRUCTION/' "$SOURCE" >"$TMP_DIR/forged-exact.sio"
if "$SOUC" run "$TMP_DIR/forged-exact.sio" >"$TMP_DIR/forged-exact.log" 2>&1; then
    fail "approximate recipe was promoted without an algebraic exact proof class"
fi
printf 'PIREUS_CROSS_ARCH_SYNTHESIZER_GATE_PASS\n'
