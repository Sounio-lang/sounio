#!/bin/bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && $# == 2 && "$2" == /* && ! -e "$2" ]] || exit 64
: "${BEFORE_RAW_BIN:?modern raw compiler before constant-scope fix required}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
seed=$(realpath "$1")
B=$2
mkdir -p "$B"
ulimit -s 524288
ulimit -c 0
cd "$ROOT"
sha256sum "$seed" "$BEFORE_RAW_BIN" >"$B/COMPILERS.sha256"
find tests/fixtures/wave4_constant_scope -type f -print0|sort -z|xargs -0 sha256sum >"$B/INPUTS.sha256"
export SOUNIO_LOOP_REGIONS=0
bash tests/fixtures/wave4_borrow_capacity/gate_checker_component.sh "$seed" "$B/regression" >"$B/regression.stdout" 2>&1
if grep -Eq 'field initializer type does not match struct field' "$B/regression/build.stdout" "$B/regression/build.stderr";then exit 1;fi
while IFS=$'\t' read -r name before after;do
 rc=0
 timeout 120 "$BEFORE_RAW_BIN" --check "tests/fixtures/wave4_constant_scope/$name.sio" >"$B/$name.before" 2>&1 || rc=$?
 [[ "$rc" == "$before" ]]
 printf '%s\tbefore\t%s\n' "$name" "$rc" >>"$B/RESULTS.tsv"
 rc=0
 timeout 120 "$B/regression/checker.elf" "tests/fixtures/wave4_constant_scope/$name.sio" >"$B/$name.after" 2>&1 || rc=$?
 [[ "$rc" == "$after" ]]
 printf '%s\tafter\t%s\n' "$name" "$rc" >>"$B/RESULTS.tsv"
 grep -Fxq "ACTUAL_CHECKER_VERDICT $after" "$B/$name.after"
 if [[ "$after" == 1 ]];then grep -Fq 'error[E056]' "$B/$name.after";fi
done < tests/fixtures/wave4_constant_scope/CASES.tsv
sha256sum -c "$B/COMPILERS.sha256" >"$B/COMPILERS.verify"
sha256sum -c "$B/INPUTS.sha256" >"$B/INPUTS.verify"
echo 0 >"$B/STATUS"
