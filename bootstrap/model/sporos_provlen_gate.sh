#!/bin/sh
# Reproducible differential and negative-control gate for P4-1 (`SporosProvLen`):
# compares the Lean reference interpreter (`formal/.lake/build/bin/sporos`) against
# `bootstrap/model/sporos_fast.cpp` across compiled examples and `sporos_gen` seeds,
# then verifies two built-in provenance-tracking mutations fail cleanly.
set -eu

N="${1:-500}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

pick_cxx() {
  cat <<'EOF' > "$W/probe.cpp"
#include <cstdint>
int main() { return 0; }
EOF
  if [ -n "${CXX:-}" ]; then
    for std in -std=c++23 -std=c++2b; do
      if $CXX $std -O2 -Wall -Wextra -o "$W/probe" "$W/probe.cpp" >/dev/null 2>&1; then
        printf '%s %s' "$CXX" "$std"
        return 0
      fi
    done
  fi
  for cand in \
    "g++ -std=c++23" \
    "c++ -std=c++23" \
    "/Library/Developer/CommandLineTools/usr/bin/clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++2b" \
    "clang++ -std=c++2b"; do
    if $cand -O2 -Wall -Wextra -o "$W/probe" "$W/probe.cpp" >/dev/null 2>&1; then
      printf '%s' "$cand"
      return 0
    fi
  done
  echo "sporos_provlen_gate: no working C++23 compiler found" >&2
  return 1
}

pick_lake() {
  if [ -n "${LAKE:-}" ] && [ -x "${LAKE}" ]; then
    printf '%s' "$LAKE"
    return 0
  fi
  if command -v lake >/dev/null 2>&1; then
    command -v lake
    return 0
  fi
  if [ -x "$HOME/.elan/bin/lake" ]; then
    printf '%s' "$HOME/.elan/bin/lake"
    return 0
  fi
  echo "sporos_provlen_gate: lake not found" >&2
  return 1
}

CXX_CMD="$(pick_cxx)"
LAKE_BIN="$(pick_lake)"

( cd "$ROOT/formal" && "$LAKE_BIN" build sporos sporos-examples >/dev/null )

REF="$ROOT/formal/.lake/build/bin/sporos"
EXGEN="$ROOT/formal/.lake/build/bin/sporos-examples"
FAST_SRC="$ROOT/bootstrap/model/sporos_fast.cpp"
GEN_SRC="$ROOT/bootstrap/model/sporos_gen.cpp"

$CXX_CMD -O2 -Wall -Wextra -o "$W/fast" "$FAST_SRC"
$CXX_CMD -O2 -Wall -Wextra -o "$W/gen" "$GEN_SRC"

mkdir -p "$W/ex"
"$EXGEN" "$W/ex" >/dev/null

mk_det_in() {
  len="$1"
  out="$2"
  seed="$3"
  if [ "$len" -le 0 ]; then
    : > "$out"
  else
    LC_ALL=C awk -v n="$len" -v s="$seed" 'BEGIN {
      for (i = 0; i < n; i++) printf "%c", ((i * 73 + s * 19) % 255) + 1
    }' > "$out"
  fi
}

norm_err() {
  sed 's/^sporos_fast: [^:]*: /sporos: X: /; s/^sporos: [^:]*: not/sporos: X: not/; s/^sporos: [^:]*: refused/sporos: X: refused/' "$1" > "$2"
}

cmp_pair() {
  prog="$1"
  inp="$2"
  bin="$3"
  rc1=0; rc2=0
  "$REF" "$prog" < "$inp" > "$W/o1" 2> "$W/e1_raw" || rc1=$?
  "$bin" "$prog" < "$inp" > "$W/o2" 2> "$W/e2_raw" || rc2=$?
  norm_err "$W/e1_raw" "$W/e1"
  norm_err "$W/e2_raw" "$W/e2"
  [ "$rc1" = "$rc2" ] && cmp -s "$W/o1" "$W/o2" && cmp -s "$W/e1" "$W/e2"
}

ex_runs=0
gen_runs=0
runs=0
disagreements=0

for p in "$W"/ex/*.spb; do
  k=0
  while [ "$k" -lt 4 ]; do
    mk_det_in $((k * 37)) "$W/in" "$k"
    runs=$((runs + 1))
    ex_runs=$((ex_runs + 1))
    if ! cmp_pair "$p" "$W/in" "$W/fast"; then
      disagreements=$((disagreements + 1))
      echo "DIFF: $p (input len $((k * 37)))" >&2
    fi
    k=$((k + 1))
  done
done

s=1
while [ "$s" -le "$N" ]; do
  "$W/gen" "$s" > "$W/p.spb"
  mk_det_in $((s % 9)) "$W/in" "$s"
  runs=$((runs + 1))
  gen_runs=$((gen_runs + 1))
  if ! cmp_pair "$W/p.spb" "$W/in" "$W/fast"; then
    disagreements=$((disagreements + 1))
    echo "DIFF: sporos_gen seed=$s" >&2
  fi
  s=$((s + 1))
done

# Negative control M1: drop +1 instruction provenance tag in prim execution
sed 's/const u64 q = qa + m.pc + 1;/const u64 q = qa + m.pc;/' "$FAST_SRC" > "$W/fast_m1.cpp"
if cmp -s "$FAST_SRC" "$W/fast_m1.cpp"; then
  echo "sporos_provlen_gate: M1 mutation was a no-op" >&2
  exit 1
fi
$CXX_CMD -O2 -Wall -Wextra -o "$W/fast_m1" "$W/fast_m1.cpp"

# Negative control M2: drop jz/jnz condition provenance tag on m.pc
sed 's/m\.pc = m\.pc + c\.p + 1;/m.pc = m.pc + 1;/' "$FAST_SRC" > "$W/fast_m2.cpp"
if cmp -s "$FAST_SRC" "$W/fast_m2.cpp"; then
  echo "sporos_provlen_gate: M2 mutation was a no-op" >&2
  exit 1
fi
$CXX_CMD -O2 -Wall -Wextra -o "$W/fast_m2" "$W/fast_m2.cpp"

neg_runs=50
m1_caught=0
m2_caught=0
s=1
while [ "$s" -le "$neg_runs" ]; do
  "$W/gen" "$s" > "$W/p.spb"
  mk_det_in $((s % 9)) "$W/in" "$s"
  if ! cmp_pair "$W/p.spb" "$W/in" "$W/fast_m1"; then
    m1_caught=$((m1_caught + 1))
  fi
  if ! cmp_pair "$W/p.spb" "$W/in" "$W/fast_m2"; then
    m2_caught=$((m2_caught + 1))
  fi
  s=$((s + 1))
done

echo "sporos_provlen_gate: ex_runs=$ex_runs gen_runs=$gen_runs runs=$runs disagreements=$disagreements neg_runs=$neg_runs m1_caught=$m1_caught m2_caught=$m2_caught"
[ "$disagreements" = 0 ] && [ "$m1_caught" -gt 0 ] && [ "$m2_caught" -gt 0 ]
