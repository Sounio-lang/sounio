#!/bin/bash
# A/B harness for unit-propagation work in the self-hosted compiler.
#
#   usage: bash scripts/dev/unit_propagation_ab.sh <label> [gen3.elf]
#
# Why this is versioned. The unit-propagation series was verified entirely by a
# harness that lived in /tmp. Everything it taught had to be re-learned by being
# burned again, and three of its runs produced results that were simply wrong:
#
#   * two runs overlapped and interleaved their output into one file, and the
#     mixed result read as six regressions that did not exist -- nearly reverting
#     a correct fix on that evidence. Hence the lock, and per-process temp paths.
#
#   * two runs launched under tmux resolved souc from a different checkout,
#     because the tmux server environment is anchored elsewhere, and reported a
#     failure set belonging to another repository. Hence SOUC_BIN is pinned here
#     and the suite log is checked to confirm which binary it actually used.
#
#   * the unit gates were reported as failing when they were not, because they
#     were invoked without SOUC_BIN and resolved something else. Hence the
#     explicit export below. Three of the four do fail; one does not.
#
# The suite's timeout count swings with machine load (3, 46, 72 observed on the
# same tree), so only `run exited 1` is comparable between runs. That is the one
# number this harness reports.
set -u

L="${1:?usage: unit_propagation_ab.sh <label> [gen3.elf]}"
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"

LOCK=/tmp/unit_propagation_ab.lock
mkdir "$LOCK" 2>/dev/null || { echo "ABORT: another run holds the lock. Two runs share paths and will race."; exit 9; }
trap 'rmdir "$LOCK" 2>/dev/null; rm -rf "${T:-}"' EXIT
T=$(mktemp -d)

COMPILER=${2:-$R/gen3.elf}
[ -x "$COMPILER" ] || { echo "ABORT: no compiler at $COMPILER"; exit 2; }

export SOUC_BIN="$R/bin/souc"
export SOUNIO_STDLIB_PATH="$R/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single

OUT=$R/unit_ab_$L.txt
: > "$OUT"
{
  echo "== compiler =="
  md5sum "$COMPILER"
} >> "$OUT"

# The suite runs through bin/souc, which resolves the lean_single engine binary.
cp "$COMPILER" "$R/bin/souc-lean-single-x86_64"
chmod +x "$R/bin/souc-lean-single-x86_64"

echo "== suite ==" >> "$OUT"
bash scripts/run_sio_test_suite.sh > "$T/suite.log" 2>&1

USED=$(grep -m1 "Using souc" "$T/suite.log" | sed 's/.*: //')
if [ -n "$USED" ] && [ "$USED" != "$SOUC_BIN" ]; then
  echo "  ABORT: the suite used $USED, not $SOUC_BIN" >> "$OUT"
  echo DONE >> "$OUT"; cat "$OUT"; exit 8
fi
echo "  binary confirmed: ${USED:-<not reported>}" >> "$OUT"

grep "  FAIL" "$T/suite.log" | grep "run exited 1" \
  | sed 's/.*FAIL  //; s/ (.*//' | sort -u > "$R/unit_ab_${L}_exit1.txt"
echo "  genuine failures (run exited 1): $(wc -l < "$R/unit_ab_${L}_exit1.txt")" >> "$OUT"

echo "== unit gates ==" >> "$OUT"
for g in unit_types_phase1_gate unit_types_derived_gate \
         unit_types_clinical_current_source_gate knowledge_context_unit_gate; do
  if [ -f "scripts/ci/$g.sh" ]; then
    if timeout 900 bash "scripts/ci/$g.sh" > "$T/$g.log" 2>&1; then
      echo "  PASS $g" >> "$OUT"
    else
      echo "  FAIL $g (exit $?)" >> "$OUT"
      grep -m1 "^FAIL" "$T/$g.log" | sed 's/^/        /' >> "$OUT"
    fi
  else
    echo "  ABSENT $g" >> "$OUT"
  fi
done

echo "== tests/frontend/unit_* (outside the suite globs) ==" >> "$OUT"
for f in tests/frontend/unit_*.sio; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  if ! "$COMPILER" "$f" "$T/fe" > "$T/fe.log" 2>&1; then
    echo "  COMPILE-FAIL $b" >> "$OUT"; continue
  fi
  chmod +x "$T/fe"
  got=$(timeout 30 "$T/fe" 2>&1); rc=$?
  want=$(grep -m1 '^//@ expect-stdout:' "$f" | sed 's|^//@ expect-stdout: *||')
  if [ $rc -ne 0 ]; then echo "  RUN-FAIL $b (exit $rc)" >> "$OUT"
  elif [ -n "$want" ] && ! printf '%s' "$got" | grep -qF "$want"; then
    echo "  WRONG-OUTPUT $b" >> "$OUT"
  else echo "  ok $b" >> "$OUT"; fi
done

echo DONE >> "$OUT"
cat "$OUT"
echo
echo "Compare two runs with: diff unit_ab_<a>_exit1.txt unit_ab_<b>_exit1.txt"
echo "Only that set is comparable; timeout counts move with machine load."
