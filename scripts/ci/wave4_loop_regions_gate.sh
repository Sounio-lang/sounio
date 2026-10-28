#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
ulimit -s 524288
ulimit -c 0
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ $# == 1 && "$1" == /* && ! -e "$1" ]] || exit 64
: "${AFTER_RAW_BIN:?rebuilt raw compiler required}"
[[ "$AFTER_RAW_BIN" == /* && -x "$AFTER_RAW_BIN" ]] || exit 64
B=$1
mkdir -p "$B/source"
for file in "$ROOT/tests/fixtures/wave4_loop_regions"/*.sio.in;do
 name=$(basename "$file" .sio.in)
 cp "$file" "$B/source/$name.sio"
done
cd "$B/source"
sha256sum "$AFTER_RAW_BIN" >"$B/COMPILER.sha256"
sha256sum ./*.sio >"$B/INPUTS.sha256"
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS=$mode
 names="while continue_break for"
 if [[ "$mode" == 1 ]];then names="$names escape_nested closure_labels closure_counter inner_carry";fi
 for name in $names;do
  prefix="$B/$mode-$name"
  timeout 300 "$AFTER_RAW_BIN" --check "$name.sio" >"$prefix.check" 2>&1
  timeout 300 "$AFTER_RAW_BIN" compile "$name.sio" -o "$prefix.elf" >"$prefix.compile" 2>&1
  chmod +x "$prefix.elf"
  sha256sum "$prefix.elf" >>"$B/ELFS.sha256"
  set +e
  timeout --kill-after=2 120 "$prefix.elf" >"$prefix.native" 2>&1
  result=$?
  set -e
  printf '%s\t%s\t%s\n' "$mode" "$name" "$result" >>"$B/RESULTS.tsv"
  expected=0
  if [[ "$mode" == 0 ]];then expected=181;fi
  [[ "$result" == "$expected" ]]
  if [[ "$mode" == 1 ]];then
   marker=LOOP_REGION_WHILE_OK
   case "$name" in continue_break)marker=LOOP_REGION_CONTROL_OK;;for)marker=LOOP_REGION_FOR_OK;;escape_nested)marker=LOOP_REGION_ESCAPE_NESTED_OK;;closure_labels)marker=LOOP_REGION_CLOSURE_LABELS_OK;;closure_counter)marker=LOOP_REGION_CLOSURE_COUNTER_OK;;inner_carry)marker=LOOP_REGION_INNER_CARRY_OK;;esac
   grep -qx "$marker" "$prefix.native"
  fi
 done
done
sha256sum -c "$B/INPUTS.sha256" >"$B/INPUTS.verify"
sha256sum -c "$B/COMPILER.sha256" >"$B/COMPILER.verify"
sha256sum -c "$B/ELFS.sha256" >"$B/ELFS.verify"
printf '0\n' >"$B/STATUS"
