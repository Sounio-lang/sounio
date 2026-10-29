#!/usr/bin/env bash
# Four total tag-dispatch functions; no producer or modern compiler rebuild.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo SLURM_REQUIRED >&2;exit 2; }
[[ $# == 6 ]] || { echo 'usage: total_tags_gate.sh ROOT BEFORE_ROOT FRESH_OUT RAW04_CONTROL OLD_SEED FINAL_SEED' >&2;exit 2; }
ROOT=$(realpath -e "$1")
BEFORE=$(realpath -e "$2")
OUT=$(realpath -m "$3")
RAW=$(realpath -e "$4")
OLD=$(realpath -e "$5")
FINAL=$(realpath -e "$6")
NODE=${NODE_BIN:-node}
CXX=${CXX:-g++}
test ! -e "$OUT"
mkdir -p "$OUT/input" "$OUT/source" "$OUT/results"
for family in dependent epistemic;do
 cp "$BEFORE/self-hosted/check/$family.sio" "$OUT/input/$family.before.sio"
 cp "$ROOT/self-hosted/check/$family.sio" "$OUT/input/$family.after.sio"
done
cp "$ROOT/tests/fixtures/wave4_total_tags/"* "$OUT/input/"
"$NODE" "$OUT/input/extract.mjs" "$OUT"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$OUT/input/reference.cpp" -o "$OUT/results/reference"
find "$OUT/input" "$OUT/source" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/INPUTS.sha256"
sha256sum "$OLD" "$FINAL" "$RAW" "$(command -v "$CXX")" "$(realpath -e "$0")" "$OUT/results/reference" > "$OUT/TOOLS.sha256"
printf 'seed\tfamily\tvariant\tphase\tstatus\n' > "$OUT/results/PHASES.tsv"
ulimit -s 524288
ulimit -c 0
finish(){
 result=$?;trap - EXIT;set +e
 sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/results/inputs.verify" || result=1
 sha256sum -c "$OUT/TOOLS.sha256" > "$OUT/results/tools.verify" || result=1
 printf '%s\n' "$result" > "$OUT/results/STATUS"
 exit "$result"
}
trap finish EXIT
for family in dependent epistemic;do
 timeout --kill-after=30s 600 "$RAW" check "$OUT/source/after/$family.sio" > "$OUT/results/after-$family.old-raw-check.stdout" 2> "$OUT/results/after-$family.old-raw-check.stderr"
 grep -Fxq 'check: OK' "$OUT/results/after-$family.old-raw-check.stdout"
 for seed in old final;do
  compiler=$OLD
  if test "$seed" = final;then compiler=$FINAL;fi
  for variant in before after;do
   name=$seed-$variant-$family
   rc=0
   timeout --kill-after=30s 600 "$compiler" "$OUT/source/$variant/$family.sio" "$OUT/results/$name.elf" > "$OUT/results/$name.compile.stdout" 2> "$OUT/results/$name.compile.stderr" || rc=$?
   printf '%s\t%s\t%s\tcompile\t%s\n' "$seed" "$family" "$variant" "$rc" >> "$OUT/results/PHASES.tsv"
   errors=0
   if grep -Ei 'error\[E|(^|[[:space:]])error:|parse error|undefined identifier|field initializer mismatch|would have dropped|cannot compile an assignment|non-Unit function has a reachable path' "$OUT/results/$name.compile.stdout" "$OUT/results/$name.compile.stderr" > "$OUT/results/$name.EXPLICIT_ERRORS";then errors=1;fi
   if test "$seed" = final && test "$variant" = before;then
    test "$errors" = 1
    expected=1
    if test "$family" = epistemic;then expected=3;fi
    test "$(grep -c 'non-Unit function has a reachable path without a return value' "$OUT/results/$name.EXPLICIT_ERRORS")" = "$expected"
    test "$(wc -l < "$OUT/results/$name.EXPLICIT_ERRORS")" = "$expected"
    printf 'explicit source diagnostic; any emitted artifact is unqualified and not executed\n' > "$OUT/results/$name.NOT_EXECUTED"
    continue
   fi
   test "$rc" = 0
   test "$errors" = 0
   chmod u+x "$OUT/results/$name.elf"
   native=0
   timeout --kill-after=10s 60 "$OUT/results/$name.elf" > "$OUT/results/$name.native.stdout" 2> "$OUT/results/$name.native.stderr" || native=$?
   printf '%s\t%s\t%s\tnative\t%s\n' "$seed" "$family" "$variant" "$native" >> "$OUT/results/PHASES.tsv"
   if test "$variant" = after;then test "$native" = 0;fi
   oracle=71
   if test "$native" = 0;then
    oracle=0
    "$OUT/results/reference" "$family" "$variant" "$OUT/results/$name.native.stdout" > "$OUT/results/$name.oracle.stdout" 2> "$OUT/results/$name.oracle.stderr" || oracle=$?
   fi
   printf '%s\t%s\t%s\toracle\t%s\n' "$seed" "$family" "$variant" "$oracle" >> "$OUT/results/PHASES.tsv"
   if test "$variant" = after;then test "$oracle" = 0;fi
  done
 done
done
for mutant in wrong_value duplicate truncate;do
 "$OUT/results/reference" mutate "$mutant" "$OUT/results/final-after-epistemic.native.stdout" "$OUT/results/$mutant.txt"
 rc=0
 "$OUT/results/reference" epistemic after "$OUT/results/$mutant.txt" > "$OUT/results/$mutant.oracle.stdout" 2> "$OUT/results/$mutant.oracle.stderr" || rc=$?
 expected=1
 if test "$mutant" = duplicate;then expected=69;fi
 if test "$mutant" = truncate;then expected=70;fi
 test "$rc" = "$expected"
 printf 'oracle\tepistemic\t%s\trefuse\t%s\n' "$mutant" "$rc" >> "$OUT/results/PHASES.tsv"
done
find "$OUT/results" -type f -name '*.elf' -print0 | sort -z | xargs -0 sha256sum > "$OUT/results/ELFS.sha256"
printf 'TOTAL_TAGS_ACTUAL_SOURCE_COMPONENT_AND_CPP23_OK\n'
