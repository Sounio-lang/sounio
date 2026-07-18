#!/usr/bin/env bash
# Verified octonion / O-SSM research probes — keep them live (math-bearing .sio rots
# if nothing runs it). Each probe is self-validating; this gate asserts the scientific
# invariants in its stdout, not just that it compiled. Octonion/f64 -> lean_single engine.
#   oct_truth        — valid octonion table: alternative error 0, norm-mult error 0
#   oct_algebra      — associator antisymmetry 0, alternative 0, flexible 0 (Fano/Moufang)
#   ossm_separation  — representational separation O-SSM(oct) - H-SSM(assoc) = 500 permil
#   ossm_recover     — associator-recovery positive control: octonion reaches 987 permil
#   rk4_correlated   — CorrelatedValue sd matches Monte-Carlo truth (3.279153) vs independent
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
run() { # file, then one or more required stdout substrings
  local f="$1"; shift
  echo "== $f =="
  if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile "$f" -o "$OUT/x.elf" >/dev/null 2>&1; then
    chmod +x "$OUT/x.elf"
    local out; out="$("$OUT/x.elf" 2>&1)"
    local s
    for s in "$@"; do
      printf '%s' "$out" | grep -qF "$s" || { echo "FAIL invariant [$f]: «$s»"; fail=1; }
    done
  else echo "FAIL compile $f"; fail=1; fi
}
run scripts/research/oct_truth.sio \
  "alternative (x,x,y) ||.||^2 (*1e6): 0" \
  "norm-mult |xy|^2-|x|^2|y|^2 (*1e6): 0"
run scripts/research/oct_algebra.sio \
  "antisymmetry violations = 0" \
  "alternative violations = 0" \
  "flexible ||.||^2 (*1e6) = 0"
run scripts/research/ossm_separation.sio \
  "permil): 500"
run scripts/research/ossm_recover.sio \
  "cc     987"
run examples/epistemic/rk4_correlated_uncertainty.sio \
  "sd = 3.279153"
[ $fail -eq 0 ] && echo "OCTONION_PROBES_GATE_OK"
exit $fail
