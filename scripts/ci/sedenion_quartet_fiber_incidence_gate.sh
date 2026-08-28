#!/usr/bin/env bash
# Cross-toolchain gate for the quartet<->fiber incidence (Frente B): the 42 quartets form 2*K_7 on the
# 7 fibers. souc executes the incidence; the Python oracle recomputes it (+ emits the 21 fiber-pair
# records); /usr/bin/diff on the summary. A bare PASS is not proof.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/i.elf" >/dev/null 2>&1; chmod +x "$WORK/i.elf"; "$WORK/i.elf" 2>/dev/null; fi }
echo "[quartet-fiber-incidence] running souc + oracle ..."
run_souc tests/run-pass/sedenion_quartet_fiber_incidence.sio > "$WORK/souc.txt"
python3 scripts/research/sedenion_quartet_fiber_incidence_oracle.py > "$WORK/py.txt"
field() { grep -m1 "^$1 " "$2" | awk '{print $2}'; }
fail=0
for key in PAIRS FIBERPAIRS BAD_FIBERS BAD_PAIRCT BAD_DEG; do
  s=$(field "$key" "$WORK/souc.txt"); p=$(field "$key" "$WORK/py.txt")
  [ "$s" = "$p" ] || { echo "MISMATCH $key: souc=$s oracle=$p"; fail=1; }
done
[ "$(field PAIRS "$WORK/souc.txt")" = "168" ]      || { echo "souc PAIRS != 168"; fail=1; }
[ "$(field FIBERPAIRS "$WORK/souc.txt")" = "21" ]  || { echo "souc FIBERPAIRS != 21"; fail=1; }
[ "$(grep -c '^INCIDENCE OK' "$WORK/souc.txt")" = "1" ] || { echo "souc verdict != INCIDENCE OK"; fail=1; }
[ "$(grep -c '^FP ' "$WORK/py.txt")" = "21" ]      || { echo "oracle FP count != 21"; fail=1; }
[ "$(field INCIDENCE "$WORK/py.txt")" = "OK" ]     || { echo "oracle verdict != OK"; fail=1; }
[ "$fail" -eq 0 ] || { echo "quartet-fiber-incidence gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: souc == oracle — the 42 quartets form 2*K_7 on the 7 fibers"
echo "  (all 21 fiber-pairs, 2 quartets each; every fiber incidence-degree 12; each quartet spans 2 fibers)."
echo "quartet-fiber-incidence gate: PASS"
