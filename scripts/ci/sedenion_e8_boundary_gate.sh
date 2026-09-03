#!/usr/bin/env bash
# Cross-toolchain replication gate for the sedenion zero-divisor e8-BOUNDARY (Frente B).
#
# WHY: souc v0.80.0 has a documented false-green mode (silent stubs / miscompiles). A bare `PASS` is
# not proof of execution. This gate closes that gap for the e8-boundary claim: the 28 SPECIFIC
# excluded primitives emitted by souc must be set-identical to those computed by an INDEPENDENT
# toolchain (Python). No stub reproduces 28 specific, correct triples by accident.
#
# Claim under test (tests/run-pass/sedenion_e8_boundary.sio): of the 112 mixed-half signed two-support
# primitives e_lo (+/-) e_hi (lo in 1..7, hi in 8..15), exactly 84 participate in a sedenion
# zero-divisor pair and 28 in none, and the 28 dead ones are EXACTLY {hi==8} U {lo XOR hi==8} — the
# doubling unit e8 (O->S) and its xor-grade-8 diagonal. This is the exact algebraic boundary the
# 168-census's generation filter ("hi in 9..15, lo^hi != 8") silently assumes.
#
# Producers:
#   (1) souc   -> tests/run-pass/sedenion_e8_boundary.sio      emits `EXCL <code>` (code=lo*10000+hi*100+neg)
#   (2) python -> scripts/research/sedenion_e8_boundary_oracle.py (transcribes ir_cd_sigma)
# Asserter: /usr/bin/diff (not souc). Exit 0 + CROSS-VERIFIED iff the two 28-sets are identical AND
# both report PARTICIPATE 84 / EXCLUDED 28 / TOUCH_E8 14 / DIAGONAL 14 / INVARIANT HOLDS.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() {  # run a .sio, stdout only; support both `souc run <src>` and `souc <src> <out>` forms
  if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
    "$SOUC" "$1" "$WORK/e8.elf" >/dev/null 2>&1; chmod +x "$WORK/e8.elf"; "$WORK/e8.elf" 2>/dev/null; fi
}

echo "[e8-boundary] running souc + oracle ..."
run_souc tests/run-pass/sedenion_e8_boundary.sio > "$WORK/souc.txt"
python3 scripts/research/sedenion_e8_boundary_oracle.py > "$WORK/py.txt"

grep '^EXCL ' "$WORK/souc.txt" | awk '{print $2}' | sort -n > "$WORK/souc_excl.txt"
grep '^EXCL ' "$WORK/py.txt"   | awk '{print $2}' | sort -n > "$WORK/py_excl.txt"
SN=$(wc -l < "$WORK/souc_excl.txt"); PN=$(wc -l < "$WORK/py_excl.txt")

field() { grep -m1 "^$1 " "$2" | awk '{print $2}'; }
fail=0
for key in PARTICIPATE EXCLUDED TOUCH_E8 DIAGONAL INVARIANT; do
  s=$(field "$key" "$WORK/souc.txt"); p=$(field "$key" "$WORK/py.txt")
  if [ "$s" != "$p" ]; then echo "MISMATCH $key: souc=$s oracle=$p"; fail=1; fi
done
[ "$(field PARTICIPATE "$WORK/souc.txt")" = "84" ] || { echo "souc PARTICIPATE != 84"; fail=1; }
[ "$(field EXCLUDED "$WORK/souc.txt")" = "28" ] || { echo "souc EXCLUDED != 28"; fail=1; }
[ "$(field INVARIANT "$WORK/souc.txt")" = "HOLDS" ] || { echo "souc INVARIANT != HOLDS"; fail=1; }

if [ "$SN" -ne 28 ] || [ "$PN" -ne 28 ] || ! diff -q "$WORK/souc_excl.txt" "$WORK/py_excl.txt" >/dev/null; then
  echo "MISMATCH excluded set: souc=$SN oracle=$PN"
  diff "$WORK/souc_excl.txt" "$WORK/py_excl.txt" | head
  fail=1
fi

if [ "$fail" -ne 0 ]; then echo "e8-boundary gate: FAIL"; exit 1; fi
echo "CROSS-VERIFIED: 28/28 excluded primitives ELEMENT-WISE IDENTICAL (souc == Python oracle)"
echo "  84 participate / 28 excluded = {hi==8}(14) U {lo^hi==8}(14) — the e8 doubling seam."
echo "e8-boundary gate: PASS"
