#!/usr/bin/env bash
# Checker-source component: single-file parsing/specialization/checking, no BFS/native-emission claim.
set -euo pipefail
[[ $# == 2 && -n "${SLURM_JOB_ID:-}" ]] || { echo "usage (inside Slurm): gate_checker_component.sh SEED_COMPILER FRESH_RESULTS" >&2;exit 2; }
repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
seed=$(realpath "$1");out=$(realpath -m "$2")
[[ -x "$seed" && ! -e "$out" ]]
mkdir -p "$out"
cd "$repo"
ulimit -s 524288;ulimit -c 0
export SOUNIO_STDLIB_PATH="$repo/self-hosted" SOUNIO_LOOP_REGIONS=0
find self-hosted stdlib tests/fixtures/wave4_borrow_capacity tests/fixtures/wave3_generic_enum tests/fixtures/wave4_tuple_pattern -type f -print0 |sort -z|xargs -0 sha256sum > "$out/SOURCES.sha256"
sha256sum "$seed" tests/reference/wave3_returned_loan_matrix_reference.cpp > "$out/INPUTS.sha256"
timeout 1800 /usr/bin/time -v -o "$out/build.resource" "$seed" tests/fixtures/wave4_borrow_capacity/checker_driver.sio "$out/checker.elf" > "$out/build.stdout" 2> "$out/build.stderr"
[[ -s "$out/checker.elf" ]]
if grep -E '^error\[E|^error:' "$out/build.stdout" "$out/build.stderr" > "$out/DIAGNOSTICS";then exit 1;fi
chmod +x "$out/checker.elf"
"${CXX:-g++}" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic tests/reference/wave3_returned_loan_matrix_reference.cpp -o "$out/reference"
"$out/reference" --manifest > "$out/EXPECTED.tsv"
printf 'case\texit_status\n' > "$out/VERDICTS.tsv"
while IFS=$'\t' read -r name expected code marker;do
 if [[ "$name" == tuple_pattern_* ]];then source="tests/fixtures/wave4_tuple_pattern/${name#tuple_pattern_}.sio"
 else source="tests/fixtures/wave3_generic_enum/$name.sio";fi
 rc=0;timeout 180 "$out/checker.elf" "$source" > "$out/$name.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$name" "$rc" >> "$out/VERDICTS.tsv"
done < "$out/EXPECTED.tsv"
"$out/reference" --kernel --verify "$out" > "$out/reference.stdout"
"$out/reference" --kernel --self-test "$out" > "$out/reference-self-test.stdout"
for n in 0000 0001 0126 0127 0128 0129 0160 0512;do
 rc=0;timeout 180 "$out/checker.elf" "tests/fixtures/wave4_borrow_capacity/bindings_$n.sio" > "$out/bindings_$n.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$n" "$rc" >> "$out/BINDINGS.tsv"
 [[ "$rc" == 0 ]]
 grep -Fxq 'ACTUAL_CHECKER_VERDICT 0' "$out/bindings_$n.stdout"
done
for n in 0 160;do
 rc=0;timeout 180 "$out/checker.elf" "tests/fixtures/wave4_borrow_capacity/illegal_loan_$n.sio" > "$out/illegal_$n.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$n" "$rc" >> "$out/ILLEGAL.tsv"
 [[ "$rc" == 1 ]];grep -Fq 'error[E038]' "$out/illegal_$n.stdout"
done
for branch in balanced asymmetry;do
 rc=0;timeout 180 "$out/checker.elf" "tests/fixtures/wave4_borrow_capacity/branch_24_$branch.sio" > "$out/branch_$branch.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$branch" "$rc" >> "$out/BRANCHES.tsv"
 if [[ "$branch" == balanced ]];then [[ "$rc" == 0 ]];else [[ "$rc" == 1 ]];grep -Fq 'error[E040]' "$out/branch_$branch.stdout";fi
done
sha256sum "$out/checker.elf" "$out/reference" > "$out/EXECUTABLES.sha256"
sha256sum -c "$out/SOURCES.sha256" > "$out/SOURCES.final.verify"
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.final.verify"
echo BORROW_ACTUAL_CHECKER_COMPONENT_OK
