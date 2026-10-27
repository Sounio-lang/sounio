#!/usr/bin/env bash
# Independent C++23 concordance and original DP regression; Linux worker or CI.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 64
out=$(realpath -m "$1")
[[ ! -e "$out" ]] || exit 64
case "$out/" in "$root/stdlib/"*|"$root/tests/"*|"$root/scripts/"*) echo "output overlaps source" >&2;exit 64;;esac
if [[ -n "${SLURM_JOB_ID:-}" ]]; then authority="slurm:$SLURM_JOB_ID"
elif [[ "${GITHUB_ACTIONS:-}" == true && -n "${GITHUB_RUN_ID:-}" && -n "${GITHUB_JOB:-}" ]];then authority="github:$GITHUB_RUN_ID:$GITHUB_JOB"
else echo "scheduled worker or official CI required" >&2;exit 64;fi
raw=$(realpath -e "${MADAROS_RAW_BIN:?qualified raw compiler required}")
[[ "$raw" == /* && -x "$raw" ]] || exit 64
node=${NODE:-node}
cxx=${CXX:-g++}
mkdir -p "$out"
printf '%s\n' "$authority" > "$out/EXECUTION_AUTHORITY"
export SOUNIO_SOUC_BIN="$raw" SOUC_NATIVE_BIN="$raw" SOUNIO_SOUC_RAW_MODE=modular
export SOUNIO_STDLIB_PATH="$root/stdlib"
ulimit -s 524288
ulimit -c 0
cd "$root"
printf 'phase\tstatus\n' > "$out/RESULTS.tsv"
find stdlib -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum tests/stdlib/epistemic/test_dp_value_iteration.sio tests/stdlib/epistemic/test_legacy_constructor_bridge.sio tests/reference/stdlib_epistemic_bridge_reference.cpp scripts/stdlib/wave3_epistemic_gate.sh scripts/stdlib/bounded_command.mjs scripts/ci/souc-native-wrapper.sh >> "$out/INPUTS.sha256"
sha256sum "$raw" > "$out/COMPILER.sha256"
: > "$out/EXECUTABLES.sha256"
finish(){
 local r=$? bad=0
 trap - EXIT
 for name in INPUTS COMPILER EXECUTABLES; do
  sha256sum -c "$out/$name.sha256" > "$out/$name.final.verify" 2>&1 || bad=1
 done
 [[ "$bad" == 0 ]] || r=1
 printf '%s\n' "$r" > "$out/STATUS"
 if [[ "$r" == 0 ]];then echo WAVE3_EPISTEMIC_GATE_OK;fi
 exit "$r"
}
trap finish EXIT
phase(){
 local name=$1 seconds=$2 r=0;shift 2
 "$node" "$root/scripts/stdlib/bounded_command.mjs" "$seconds" 16777216 -- "$@" > "$out/$name.log" 2>&1 || r=$?
 printf '%s\t%s\n' "$name" "$r" >> "$out/RESULTS.tsv"
 return "$r"
}
phase cpp-build 120 "$cxx" -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror tests/reference/stdlib_epistemic_bridge_reference.cpp -o "$out/reference"
sha256sum "$out/reference" >> "$out/EXECUTABLES.sha256"
for n in test_dp_value_iteration test_legacy_constructor_bridge;do
 src="$root/tests/stdlib/epistemic/$n.sio"
 phase "$n-check" 300 bash scripts/ci/souc-native-wrapper.sh check "$src"
 phase "$n-compile" 300 bash scripts/ci/souc-native-wrapper.sh compile "$src" -o "$out/$n.elf"
 chmod +x "$out/$n.elf"
 sha256sum "$out/$n.elf" >> "$out/EXECUTABLES.sha256"
 phase "$n-native" 120 "$out/$n.elf"
done
grep -Fx DP_VALUE_ITERATION_PASS "$out/test_dp_value_iteration-native.log"
phase cpp-concordance 30 "$out/reference" "$out/test_legacy_constructor_bridge-native.log"
phase mutate 30 "$out/reference" "$out/test_legacy_constructor_bridge-native.log" --mutate "$out/mutant.log"
r=0;phase reject-mutant 30 "$out/reference" "$out/mutant.log" || r=$?
[[ "$r" == 1 ]] && grep -Fq 'numerical mismatch' "$out/reject-mutant.log"
for n in missing-prob io-callback empty-samples;do "$out/reference" "--$n" > "$out/$n.sio";done
for n in missing-prob io-callback;do
 r=0;phase "$n-check" 300 bash scripts/ci/souc-native-wrapper.sh check "$out/$n.sio" || r=$?
 [[ "$r" == 1 ]]
done
grep -F E035 "$out/missing-prob-check.log"
grep -F E009 "$out/io-callback-check.log"
phase empty-samples-compile 300 bash scripts/ci/souc-native-wrapper.sh compile "$out/empty-samples.sio" -o "$out/empty-samples.elf"
chmod +x "$out/empty-samples.elf"
sha256sum "$out/empty-samples.elf" >> "$out/EXECUTABLES.sha256"
r=0;phase empty-samples-native 30 "$out/empty-samples.elf" || r=$?
[[ "$r" != 0 && "$r" != 122 && "$r" != 124 && "$r" != 127 ]]
! grep -q BAD_EMPTY_SAMPLE "$out/empty-samples-native.log"
find "$out" -maxdepth 1 -type f \( -name '*.sio' -o -name '*.log' \) -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$out/ARTIFACTS.sha256"
sha256sum -c "$out/ARTIFACTS.sha256" > "$out/ARTIFACTS.final.verify"
