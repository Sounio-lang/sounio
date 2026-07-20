#!/usr/bin/env bash
# Epistemic trust gate — the "safe to use today under native import" boundary for
# stdlib/epistemic on the default Madaros engine.
#
# Section A (TRUSTWORTHY) gates the build: these primitives import natively AND
# return first-principles-correct numbers; if any breaks, the gate fails.
# Sections B/C are trip-wires: they document the currently-broken boundary and
# only PRINT when a known bug is fixed (so the trust map gets updated), without
# failing the gate. See docs/audit/EPISTEMIC_TRUST_MAP_2026-07-14.md.
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
runproof "gum: value + u_c"          tests/epistemic_trust/gum_trust.sio         GUM_TRUST_OK
runproof "correlation: covariance"   tests/epistemic_trust/correlation_trust.sio CORRELATION_TRUST_OK
runproof "knightian: p-box"          tests/epistemic_trust/knightian_trust.sio   KNIGHTIAN_TRUST_OK
runproof "knowledge: free Epistemic" tests/epistemic_trust/knowledge_trust.sio   KNOWLEDGE_TRUST_OK
# CPC N=4 exact-spread leaf (2026-07-20): self-contained OsOct path; no algebra:: use.
runproof "order_spread4: CPC N=4"    tests/epistemic_trust/order_spread_trust.sio ORDER_SPREAD_TRUST_OK
# Structural nonassoc variance leaf (2026-07-20): self-contained PnOct path; no algebra:: use.
runproof "product_nonassoc: fano/nonfano" tests/epistemic_trust/product_nonassoc_trust.sio PRODUCT_NONASSOC_TRUST_OK
# Full propagate delta-method + value-style LCG MC (2026-07-20): exp/product + MC kernels.
runproof "propagate: exp/product/MC" tests/epistemic_trust/propagate_trust.sio PROPAGATE_TRUST_OK

echo "### B. KNOWN-CORRUPTED trip-wire (informational) ###"
echo "== gum k95 coverage factor (should be 2776 = t95(4); bug gives 1960) =="
if $SOUC compile tests/epistemic_trust/witness_gum_k95.sio -o "$OUT/k.elf" >/dev/null 2>&1; then
  k95=$("$OUT/k.elf" | tr -d '[:space:]')
  if [ "$k95" = "1960" ]; then echo "CONFIRMED CORRUPT (k95=1960, f64->i64 cast bug persists)";
  else echo "!! k95=$k95 — coverage factor may be FIXED; update trust map + re-enable U95"; fi
else echo "witness_gum_k95 failed to compile (unexpected)"; fi

echo "### C. KNOWN-UNIMPORTABLE trip-wire (informational) ###"
# Free-function knowledge import is now Section A. Residual: method-call form (Root 2).
echo "== witness_import_knowledge_method (expected: native compile FAILS) =="
if $SOUC compile tests/epistemic_trust/witness_import_knowledge_method.sio -o "$OUT/w.elf" >/dev/null 2>&1; then
  echo "!! method-call knowledge now COMPILES — Root 2 may be FIXED; update trust map"
else echo "CONFIRMED unimportable (method-call form, as documented)"; fi

# order_spread_exact + product_nonassoc promoted to Section A.
# Residual multi-module: algebra::associator_field / algebra::octonion exclusive-ref
# import chain still SEGV at runtime under Madaros — do not reintroduce algebra:: uses.

# Legacy free-function witness should now succeed (informational flip)
echo "== witness_import_knowledge free API (expected: now COMPILES) =="
if $SOUC compile tests/epistemic_trust/witness_import_knowledge.sio -o "$OUT/w.elf" >/dev/null 2>&1; then
  echo "OK free-function knowledge import COMPILES (promoted to Section A)"
else echo "FAIL unexpected: free knowledge import broke"; fail=1; fi

echo
[ $fail -eq 0 ] && echo "EPISTEMIC_TRUST_GATE_OK"
exit $fail
