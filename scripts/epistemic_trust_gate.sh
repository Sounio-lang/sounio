#!/usr/bin/env bash
# Epistemic trust gate — the "safe to use today under native import" boundary for
# stdlib/epistemic on the default Madaros engine.
#
# Section A (TRUSTWORTHY) gates the build: these primitives import natively AND
# return first-principles-correct numbers; if any breaks, the gate fails.
# Section C is residual bookkeeping for promoted witnesses that must stay green.
# See docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md.
#
# Wave10 (2026-07-21): Section B (gum k95 "CONFIRMED CORRUPT") retired. D1/#983
# + #1252 closed the f64-param cast; the old witness was mis-designed (Type-B
# dominant → k95=1.960 correct). Finite-dof k95/U95 now gate in Section A.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

runproof () {
  local name="$1" driver="$2" sentinel="$3"
  echo "== $name =="
  if $SOUC compile "$driver" -o "$OUT/a.elf" >/dev/null 2>"$OUT/e"; then
    if "$OUT/a.elf" | grep -q "$sentinel"; then echo "PASS: $sentinel"; else echo "FAIL: assertions"; fail=1; fi
  else echo "FAIL: compile (a TRUSTWORTHY primitive regressed!)"; tail -2 "$OUT/e"; fail=1; fi
}

echo "### A. TRUSTWORTHY under native import (gating) ###"
runproof "gum: value + u_c + k95/U95" tests/epistemic_trust/gum_trust.sio         GUM_TRUST_OK
runproof "correlation: covariance"   tests/epistemic_trust/correlation_trust.sio CORRELATION_TRUST_OK
runproof "knightian: p-box"          tests/epistemic_trust/knightian_trust.sio   KNIGHTIAN_TRUST_OK
runproof "knowledge: free Epistemic" tests/epistemic_trust/knowledge_trust.sio   KNOWLEDGE_TRUST_OK
# Method form + free/method parity (Wave9 residual closeout — was Root-2 blocked).
runproof "knowledge: method Epistemic" tests/epistemic_trust/knowledge_method_parity.sio KNOWLEDGE_METHOD_PARITY_OK
runproof "knowledge: method witness" tests/epistemic_trust/witness_import_knowledge_method.sio KNOWLEDGE_METHOD_OK
# CPC N=4 exact-spread leaf (2026-07-20): self-contained OsOct path; no algebra:: use.
runproof "order_spread4: CPC N=4"    tests/epistemic_trust/order_spread_trust.sio ORDER_SPREAD_TRUST_OK
# Structural nonassoc variance leaf (2026-07-20): self-contained PnOct path; no algebra:: use.
runproof "product_nonassoc: fano/nonfano" tests/epistemic_trust/product_nonassoc_trust.sio PRODUCT_NONASSOC_TRUST_OK
# Full propagate delta-method + value-style LCG MC (2026-07-20): exp/product + MC kernels.
runproof "propagate: exp/product/MC" tests/epistemic_trust/propagate_trust.sio PROPAGATE_TRUST_OK

# C1 (2026-08-06): imported Epistemic Var preserve under particle amp graph.
# Dual-engine parity (lean_single ≡ Madaros scaled i64). Fail-closed.
echo "== imported ep-var preserve (Madaros ≡ lean_single) =="
if bash scripts/ci/madaros_imported_ep_var_preserve_gate.sh; then
  echo "PASS: MADAROS_IMPORTED_EP_VAR_PRESERVE_GATE_OK"
else
  echo "FAIL: imported ep-var preserve gate"
  fail=1
fi

# Finite-dof coverage factor (promoted from retired Section B trip-wire).
# Expect k95*1000 = 2776 = t95(4) on Type-A-dominant budget (NOT 1960).
echo "== gum k95 coverage factor (gating: expect 2776 = t95(4)) =="
if $SOUC compile tests/epistemic_trust/witness_gum_k95.sio -o "$OUT/k.elf" >/dev/null 2>"$OUT/ke"; then
  k95=$("$OUT/k.elf" | tr -d '[:space:]')
  if [ "$k95" = "2776" ]; then
    echo "PASS: k95i=2776 (Student-t finite-dof coverage)"
  else
    echo "FAIL: k95i=$k95 (want 2776; 1960 would reintroduce D1 bitcast collapse)"
    fail=1
  fi
else
  echo "FAIL: witness_gum_k95 failed to compile"
  tail -2 "$OUT/ke" || true
  fail=1
fi

echo "### C. promoted witnesses (must stay green) ###"
# order_spread_exact + product_nonassoc + knowledge free/method + gum k95 → Section A.
# Do not reintroduce algebra:: exclusive-ref import chains without current-source evidence.

# Legacy free-function witness must still succeed
echo "== witness_import_knowledge free API =="
if $SOUC compile tests/epistemic_trust/witness_import_knowledge.sio -o "$OUT/w.elf" >/dev/null 2>&1; then
  echo "OK free-function knowledge import COMPILES (Section A)"
else echo "FAIL unexpected: free knowledge import broke"; fail=1; fi

echo
[ $fail -eq 0 ] && echo "EPISTEMIC_TRUST_GATE_OK"
exit $fail
