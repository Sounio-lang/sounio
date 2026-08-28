#!/usr/bin/env bash
# Cross-toolchain gate for the isomorphism type of the sedenion ZD fibers (Frente B, brick 3).
#
# Claim (tests/run-pass/sedenion_zd_fiber_identity.sio): each of the 7 fibers has common-neighbor
# profile (4:6, 2:24, 0:36) — the signature of K_{6,6} minus three disjoint 4-cycles (= K_{6,6}-3K_{2,2}),
# given brick 2's 4-regular-bipartite-6+6 structure. souc executes the (BFS-free) profile; the Python
# oracle additionally confirms the rigorous "complement = three 4-cycles" isomorphism by traversal.
#
# souc false-greens; a bare PASS is not proof. Asserter: /usr/bin/diff on the 7 specific fiber records.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() {
  if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
    "$SOUC" "$1" "$WORK/f.elf" >/dev/null 2>&1; chmod +x "$WORK/f.elf"; "$WORK/f.elf" 2>/dev/null; fi
}

echo "[fiber-identity] running souc + oracle ..."
run_souc tests/run-pass/sedenion_zd_fiber_identity.sio > "$WORK/souc.txt"
python3 scripts/research/sedenion_zd_fiber_identity_oracle.py > "$WORK/py.txt"

grep '^FIBERID ' "$WORK/souc.txt" | awk '{print $2}' | sort -n > "$WORK/souc_f.txt"
grep '^FIBERID ' "$WORK/py.txt"   | awk '{print $2}' | sort -n > "$WORK/py_f.txt"
field() { grep -m1 "^$1 " "$2" | awk '{print $2}'; }
fail=0
for key in VERTICES FIBER_ID; do
  s=$(field "$key" "$WORK/souc.txt"); p=$(field "$key" "$WORK/py.txt")
  [ "$s" = "$p" ] || { echo "MISMATCH $key: souc=$s oracle=$p"; fail=1; }
done
[ "$(field VERTICES "$WORK/souc.txt")" = "84" ]  || { echo "souc VERTICES != 84"; fail=1; }
[ "$(field FIBER_ID "$WORK/souc.txt")" = "OK" ]  || { echo "souc FIBER_ID != OK"; fail=1; }
[ "$(field COMPLEMENT_C4 "$WORK/py.txt")" = "7" ] || { echo "oracle COMPLEMENT_C4 != 7"; fail=1; }

if [ "$(wc -l < "$WORK/souc_f.txt")" -ne 7 ] || ! diff -q "$WORK/souc_f.txt" "$WORK/py_f.txt" >/dev/null; then
  echo "MISMATCH fiber-identity records:"; diff "$WORK/souc_f.txt" "$WORK/py_f.txt" | head; fail=1
fi

[ "$fail" -eq 0 ] || { echo "fiber-identity gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: 7/7 fiber common-neighbor profiles (6,24,36) IDENTICAL (souc == Python oracle)"
echo "  each fiber = K_{6,6} minus three disjoint 4-cycles; oracle confirms complement = 3xC4 for all 7."
echo "fiber-identity gate: PASS"
