#!/usr/bin/env bash
# Differential test: sporos_fast (C++23) against the Lean reference interpreter.
#   bootstrap/model/sporos_diff.sh [N]
# Builds both, then runs the example programs on random inputs and N random
# programs (sporos_gen), and compares exit status, standard output and the
# error line byte for byte. Exit 0 when every run agrees.
set -euo pipefail
N="${1:-500}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
( cd "$ROOT/formal" && lake build sporos sporos-examples >/dev/null )
REF="$ROOT/formal/.lake/build/bin/sporos"
g++ -std=c++23 -O2 -o "$W/fast" "$ROOT/bootstrap/model/sporos_fast.cpp"
g++ -std=c++23 -O2 -o "$W/gen" "$ROOT/bootstrap/model/sporos_gen.cpp"
"$ROOT/formal/.lake/build/bin/sporos-examples" "$W/ex" >/dev/null
fail=0; runs=0
one() {  # prog input
  local rc1=0 rc2=0
  "$REF" "$1" < "$2" > "$W/o1" 2> "$W/e1" || rc1=$?
  "$W/fast" "$1" < "$2" > "$W/o2" 2> "$W/e2" || rc2=$?
  sed -i 's/^sporos_fast: [^:]*: /sporos: X: /; s/^sporos: [^:]*: \(not\|refused\)/sporos: X: \1/' "$W/e1" "$W/e2"
  runs=$((runs + 1))
  if [ "$rc1" != "$rc2" ] || ! cmp -s "$W/o1" "$W/o2" || ! cmp -s "$W/e1" "$W/e2"; then
    fail=$((fail + 1)); echo "DIFF: $1 (rc $rc1 vs $rc2)"; cp "$1" "$W/../sporos_diff_fail_$runs.spb" 2>/dev/null || true
  fi
}
for p in "$W"/ex/*.spb; do
  for k in 0 1 2 3; do head -c $((k * 37)) /dev/urandom > "$W/in"; one "$p" "$W/in"; done
done
for s in $(seq 1 "$N"); do
  "$W/gen" "$s" > "$W/p.spb"; head -c $((s % 9)) /dev/urandom > "$W/in"; one "$W/p.spb" "$W/in"
done
echo "sporos_diff: runs=$runs disagreements=$fail"
[ "$fail" = 0 ]
