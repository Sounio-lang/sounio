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

echo "### B. KNOWN-CORRUPTED trip-wire (informational) ###"
echo "== gum k95 coverage factor (should be 2776 = t95(4); bug gives 1960) =="
if $SOUC compile tests/epistemic_trust/witness_gum_k95.sio -o "$OUT/k.elf" >/dev/null 2>&1; then
  k95=$("$OUT/k.elf" | tr -d '[:space:]')
  if [ "$k95" = "1960" ]; then echo "CONFIRMED CORRUPT (k95=1960, f64->i64 cast bug persists)";
  else echo "!! k95=$k95 — coverage factor may be FIXED; update trust map + re-enable U95"; fi
else echo "witness_gum_k95 failed to compile (unexpected)"; fi

echo "### C. KNOWN-UNIMPORTABLE trip-wire (informational) ###"
for w in witness_import_knowledge witness_import_order_spread; do
  echo "== $w (expected: native compile FAILS) =="
  if $SOUC compile "tests/epistemic_trust/$w.sio" -o "$OUT/w.elf" >/dev/null 2>&1; then
    echo "!! $w now COMPILES — module became native-importable; update trust map"
  else echo "CONFIRMED unimportable (native compile fails, as documented)"; fi
done

echo
[ $fail -eq 0 ] && echo "EPISTEMIC_TRUST_GATE_OK"
exit $fail
