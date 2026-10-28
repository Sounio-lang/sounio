#!/usr/bin/env bash
# Require an actual rebuilt checker/compiler; do not apply to the bootstrap seed.
set -euo pipefail
[[ $# == 2 && -n "${SLURM_JOB_ID:-}" ]] || { echo "usage (inside Slurm): gate_capacity.sh RAW_COMPILER FRESH_RESULTS" >&2;exit 2; }
repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
raw=$(realpath "$1")
out=$(realpath -m "$2")
[[ -x "$raw" && ! -e "$out" ]]
mkdir -p "$out"
cd "$repo"
ulimit -s 524288;ulimit -c 0
export SOUNIO_STDLIB_PATH="$repo/stdlib"
sha256sum "$raw" > "$out/COMPILER.sha256"
find tests/fixtures/wave4_borrow_capacity -type f -print0 |sort -z|xargs -0 sha256sum > "$out/FIXTURES.sha256"
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS="$mode"
 for n in 0000 0001 0126 0127 0128 0129 0160 0512;do
  rc=0
  timeout 180 "$raw" --check "tests/fixtures/wave4_borrow_capacity/bindings_$n.sio" > "$out/bindings_$n-$mode.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\t%s\n' "$mode" "$n" "$rc" >> "$out/BINDINGS.tsv"
  [[ "$rc" == 0 ]]
 done
 for n in 0 160;do
  rc=0
  timeout 180 "$raw" --check "tests/fixtures/wave4_borrow_capacity/illegal_loan_$n.sio" > "$out/illegal_$n-$mode.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\t%s\n' "$mode" "$n" "$rc" >> "$out/ILLEGAL.tsv"
  [[ "$rc" == 1 ]]
  grep -Fq 'error[E038]' "$out/illegal_$n-$mode.stdout"
 done
 for branch in balanced asymmetry;do
  rc=0
  timeout 180 "$raw" --check "tests/fixtures/wave4_borrow_capacity/branch_24_$branch.sio" > "$out/branch_$branch-$mode.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\t%s\n' "$mode" "$branch" "$rc" >> "$out/BRANCHES.tsv"
  if [[ "$branch" == balanced ]];then
   [[ "$rc" == 0 ]]
   timeout 300 "$raw" "tests/fixtures/wave4_borrow_capacity/branch_24_$branch.sio" -o "$out/branch$mode.elf" > "$out/branch$mode.compile.stdout" 2>&1
   [[ -s "$out/branch$mode.elf" ]]
   chmod +x "$out/branch$mode.elf"
   timeout 30 "$out/branch$mode.elf" > "$out/branch$mode.native.stdout" 2>&1
   grep -Fxq DEEP_BRANCH_OK "$out/branch$mode.native.stdout"
  else
   [[ "$rc" == 1 ]]
   grep -Fq 'error[E040]' "$out/branch_$branch-$mode.stdout"
  fi
 done
done
sha256sum -c "$out/FIXTURES.sha256" > "$out/FIXTURES.final.verify"
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.final.verify"
echo BORROW_CAPACITY_AND_BRANCHES_OK
