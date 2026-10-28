#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && -n "${SLURM_JOB_ID:-}" ]] || { echo "usage inside Slurm: gate_existing.sh CHECKER BASELINE_RESULTS FRESH_RESULTS" >&2;exit 2; }
fixtures=$(cd -- "$(dirname -- "$0")" && pwd)
checker=$(realpath "$1");before=$(realpath "$2");out=$(realpath -m "$3")
[[ -x "$checker" && -s "$before/FOCUSED.tsv" && ! -e "$out" ]]
mkdir -p "$out"
ulimit -s 524288;ulimit -c 0
sha256sum "$checker" "$fixtures/EXPECTED.tsv" "$fixtures/reference.cpp" "$fixtures/"*.sio > "$out/INPUTS.sha256"
"${CXX:-g++}" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$fixtures/reference.cpp" -o "$out/reference"
while IFS=$'\t' read -r name expected code;do
 [[ -n "$name" && -s "$fixtures/$name.sio" ]]
 rc=0
 SOUNIO_NS_TRACE=1 timeout 180 "$checker" "$fixtures/$name.sio" > "$out/$name.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$name" "$rc" >> "$out/FOCUSED.tsv"
done < "$fixtures/EXPECTED.tsv"
"$out/reference" --verify "$before" "$out" > "$out/reference.stdout"
"$out/reference" --self-test "$before" "$out" > "$out/reference-selftest.stdout"
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.final.verify"
echo LOGICAL_EXISTING_CHECKER_GATE_OK
