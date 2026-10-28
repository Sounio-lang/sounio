#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]];then
 echo "usage: $0 RAW_MADAROS RESULTS_DIRECTORY" >&2
 exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" && "${GITHUB_ACTIONS:-}" != true ]];then
 echo "returned-loan gate requires Slurm or the official CI runner" >&2
 exit 2
fi
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
raw=$(realpath -- "$1")
results=$(realpath -m -- "$2")
[[ -x "$raw" ]]
file "$raw" | grep -q ELF
mkdir -p "$results"
[[ ! -e "$results/VERDICTS.tsv" && ! -e "$results/EXPECTED.tsv" ]]
cd "$repo"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -pedantic tests/reference/wave3_returned_loan_matrix_reference.cpp -o "$results/reference"
"$results/reference" --manifest > "$results/EXPECTED.tsv"
sha256sum "$raw" "$results/reference" > "$results/ARTIFACT_INPUTS.sha256"
{
 printf '%s\0' self-hosted/check/borrow.sio self-hosted/check/check.sio tests/reference/wave3_returned_loan_matrix_reference.cpp scripts/ci/wave3_returned_loans_gate.sh
 while IFS=$'\t' read -r name expected code marker;do
  printf '%s\0' "tests/fixtures/wave3_generic_enum/$name.sio"
 done < "$results/EXPECTED.tsv"
} | xargs -0 sha256sum > "$results/SOURCE_INPUTS.sha256"
export SOUNIO_STDLIB_PATH="$repo/stdlib"
"$raw" --version > "$results/COMPILER_VERSION.txt" 2>&1
printf 'case\texit_status\n' > "$results/VERDICTS.tsv"
printf 'case\tphase\texit_status\n' > "$results/PHASES.tsv"
failed=0
while IFS=$'\t' read -r name expected code marker;do
 source="tests/fixtures/wave3_generic_enum/$name.sio"
 rc=0;timeout 90 "$raw" check "$source" > "$results/$name.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$name" "$rc" >> "$results/VERDICTS.tsv"
 printf '%s\tcheck\t%s\n' "$name" "$rc" >> "$results/PHASES.tsv"
 if [[ "$rc" != "$expected" ]];then failed=1;fi
 rc=0;timeout 180 "$raw" compile "$source" -o "$results/$name.elf" > "$results/$name-compile.stdout" 2>&1 ||rc=$?
 printf '%s\tcompile\t%s\n' "$name" "$rc" >> "$results/PHASES.tsv"
 if [[ "$expected" == 1 ]];then
  if [[ "$rc" != 1 || -e "$results/$name.elf" ]];then failed=1;fi
  if ! grep -Fq "error[$code]" "$results/$name-compile.stdout";then failed=1;fi
 else
  if [[ "$rc" != 0 || ! -s "$results/$name.elf" ]];then failed=1;continue;fi
  chmod +x "$results/$name.elf"
  rc=0;timeout 30 "$results/$name.elf" > "$results/$name-native.stdout" 2>&1 ||rc=$?
  printf '%s\tnative\t%s\n' "$name" "$rc" >> "$results/PHASES.tsv"
  if [[ "$rc" != 0 ]] || ! grep -Fxq "$marker" "$results/$name-native.stdout";then failed=1;fi
 fi
done < "$results/EXPECTED.tsv"
"$results/reference" --raw --verify "$results" > "$results/reference.stdout" 2>&1 ||failed=1
"$results/reference" --raw --self-test "$results" > "$results/wrong-output.stdout" 2>&1 ||failed=1
sha256sum -c "$results/SOURCE_INPUTS.sha256" > "$results/SOURCE_INPUTS.final.verify"
sha256sum -c "$results/ARTIFACT_INPUTS.sha256" > "$results/ARTIFACT_INPUTS.final.verify"
printf '%s\n' "$failed" > "$results/GATE_STATUS"
if [[ "$failed" != 0 ]];then
 echo "RETURNED_LOAN_NATIVE_GATE_REFUSED"
 exit 1
fi
echo "RETURNED_LOAN_NATIVE_GATE_OK"
