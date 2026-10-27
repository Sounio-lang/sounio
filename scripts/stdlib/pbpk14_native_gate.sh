#!/usr/bin/env bash
# Original ODE regressions and independent C++23 matrix-exponential concordance.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 64
out=$(realpath -m "$1")
[[ ! -e "$out" ]] || exit 64
case "$out/" in "$root/stdlib/"*|"$root/tests/"*|"$root/scripts/"*) exit 64;;esac
if [[ -n "${SLURM_JOB_ID:-}" ]];then authority="slurm:$SLURM_JOB_ID"
elif [[ "${GITHUB_ACTIONS:-}" == true && "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ && "${GITHUB_JOB:-}" =~ ^[A-Za-z0-9_-]+$ ]];then authority="github:$GITHUB_RUN_ID:$GITHUB_JOB"
else echo "scheduled worker or official CI required" >&2;exit 64;fi
raw=$(realpath -e "${MADAROS_RAW_BIN:?qualified raw compiler required}")
[[ -x "$raw" ]] || exit 64
node=${NODE:-node}
cxx=${CXX:-g++}
export SOUC_NATIVE_BIN="$raw" SOUNIO_SOUC_BIN="$raw" SOUNIO_SOUC_RAW_MODE=modular
export SOUNIO_STDLIB_PATH="$root/stdlib"
ulimit -s 524288
ulimit -c 0
mkdir -p "$out"
cd "$root"
printf '%s\n' "$authority" > "$out/EXECUTION_AUTHORITY"
catalog(){
 find stdlib -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
 sha256sum tests/stdlib/ode/test_pbpk14_rk4_e2e.sio tests/stdlib/ode/test_tsit5_e2e.sio tests/stdlib/ode/test_pbpk14_checked.sio tests/stdlib/ode/test_solver_option_factories.sio tests/reference/stdlib_pbpk14_reference.cpp scripts/stdlib/pbpk14_native_gate.sh scripts/stdlib/bounded_command.mjs scripts/ci/souc-native-wrapper.sh
}
catalog > "$out/INPUTS.sha256"
sha256sum "$raw" "$(command -v "$node")" "$(command -v "$cxx")" > "$out/TOOLS.sha256"
: > "$out/EXECUTABLES.sha256"
: > "$out/DERIVED.sha256"
printf 'test\tphase\tstatus\n' > "$out/RESULTS.tsv"
finish(){
 local r=$? bad=0
 trap - EXIT
 catalog > "$out/INPUTS.final.sha256" || bad=1
 cmp "$out/INPUTS.sha256" "$out/INPUTS.final.sha256" > "$out/INPUTS.catalog.verify" 2>&1 || bad=1
 for name in INPUTS TOOLS EXECUTABLES DERIVED;do
  sha256sum -c "$out/$name.sha256" > "$out/$name.final.verify" 2>&1 || bad=1
 done
 [[ "$bad" == 0 ]] || r=1
 find "$out" -type f \( -name '*.log' -o -name '*.tsv' -o -name '*.sio' \) -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$out/ARTIFACTS.sha256" || r=1
 sha256sum -c "$out/ARTIFACTS.sha256" > "$out/ARTIFACTS.final.verify" 2>&1 || r=1
 echo "$r" > "$out/STATUS"
 if [[ "$r" == 0 ]];then echo PBPK14_NATIVE_GATE_OK;fi
 exit "$r"
}
trap finish EXIT
phase(){
 local name=$1 kind=$2 seconds=$3 r=0;shift 3
 "$node" scripts/stdlib/bounded_command.mjs "$seconds" 16777216 -- "$@" > "$out/$name.$kind.log" 2>&1 || r=$?
 printf '%s\t%s\t%s\n' "$name" "$kind" "$r" >> "$out/RESULTS.tsv"
 return "$r"
}
phase reference compile 180 "$cxx" -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror tests/reference/stdlib_pbpk14_reference.cpp -o "$out/reference"
sha256sum "$out/reference" >> "$out/EXECUTABLES.sha256"
"$cxx" --version > "$out/CXX_VERSION"
ldd "$out/reference" > "$out/BOUND_LIBRARIES"
! grep -q 'not found' "$out/BOUND_LIBRARIES"
awk '/=> \// {print $3} /^[[:space:]]*\// {print $1}' "$out/BOUND_LIBRARIES" | LC_ALL=C sort -u | xargs -r sha256sum >> "$out/TOOLS.sha256"
phase reference selftest 180 "$out/reference" --selftest
"$out/reference" --fixture > "$out/generated-pbpk.sio"
"$out/reference" --solver-fixture > "$out/generated-solver.sio"
cmp "$out/generated-pbpk.sio" tests/stdlib/ode/test_pbpk14_checked.sio
cmp "$out/generated-solver.sio" tests/stdlib/ode/test_solver_option_factories.sio
sha256sum "$out/generated-pbpk.sio" "$out/generated-solver.sio" >> "$out/DERIVED.sha256"
for name in test_pbpk14_rk4_e2e test_tsit5_e2e test_pbpk14_checked test_solver_option_factories;do
 source="$root/tests/stdlib/ode/$name.sio"
 phase "$name" check 1200 bash scripts/ci/souc-native-wrapper.sh check "$source"
 phase "$name" compile 1200 bash scripts/ci/souc-native-wrapper.sh compile "$source" -o "$out/$name.elf"
 chmod +x "$out/$name.elf"
 sha256sum "$out/$name.elf" >> "$out/EXECUTABLES.sha256"
 phase "$name" native 30 "$out/$name.elf"
done
phase reference concordance 180 "$out/reference" --check "$out/test_pbpk14_checked.native.log"
phase reference solver-concordance 30 "$out/reference" --check-solver "$out/test_solver_option_factories.native.log"
phase oracle mutate 30 "$out/reference" --wrong-output "$out/test_pbpk14_checked.native.log" "$out/wrong-output.log"
sha256sum "$out/wrong-output.log" >> "$out/DERIVED.sha256"
r=0;phase oracle reject-wrong-value 180 "$out/reference" --check "$out/wrong-output.log" || r=$?
[[ "$r" == 1 ]] && grep -Fq 'numerical mismatch' "$out/oracle.reject-wrong-value.log"
