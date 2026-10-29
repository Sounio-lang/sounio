#!/usr/bin/env bash
# Run inside a Slurm allocation. Never replaces a repository compiler/seed.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || $# != 5 ]]; then
  echo "usage (Slurm only): $0 BASELINE_SOURCE CANDIDATE_SOURCE SEED SELF_HOSTED_SDK NEW_OUTPUT_DIR" >&2
  exit 2
fi
baseline=$1
candidate=$2
seed=$3
sdk=$4
out=$5
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${CXX:?set CXX to the qualified C++23 compiler}"
[[ ! -e "$out" ]] || { echo "output directory already exists" >&2; exit 2; }
mkdir -p "$out/input"
out=$(cd "$out" && pwd)
cp "$baseline" "$out/baseline.sio"
cp "$candidate" "$out/candidate.sio"
cp "$repo"/tests/fixtures/wave4_legacy_flow/input/*.sio "$out/input/"
cp "$repo"/tests/fixtures/wave4_sig139_census/packet/returning-arm/input/*.sio "$out/input/"
cp "$repo/tests/fixtures/wave4_legacy_flow/reference.cpp" "$out/reference.cpp"
cp "$repo/tests/fixtures/wave4_legacy_flow/CASES.json" "$out/CASES.json"
cd "$out"
find input -type f -print0 | sort -z | xargs -0 sha256sum > INPUTS.before.sha256
sha256sum baseline.sio candidate.sio reference.cpp CASES.json "$seed" >> INPUTS.before.sha256
export SOUNIO_STDLIB_PATH="$sdk"
ulimit -s 524288
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror reference.cpp -o reference > reference.build.stdout 2>reference.build.stderr
for kind in baseline candidate; do
 set +e
 timeout 600 "$seed" "$kind.sio" "$kind-producer.elf" > "$kind.build.stdout" 2>"$kind.build.stderr"
 status=$?
 set -e
 echo "$status" > "$kind.BUILD_STATUS"
 [[ "$status" == 0 ]] || exit 1
 chmod +x "$kind-producer.elf"
done
run_matrix() {
 local name=$1 producer=$2
 mkdir "$name"
 for f in input/*.sio; do
  local stem
  stem=$(basename "$f" .sio)
  set +e
  timeout 30 "$producer" "$f" "$name/$stem.elf" > "$name/$stem.compile.stdout" 2>"$name/$stem.compile.stderr"
  local cr=$?
  echo "$cr" > "$name/$stem.compile.status"
  local nr=999
  if [[ "$cr" == 0 ]]; then
   chmod +x "$name/$stem.elf"
   timeout 5 "$name/$stem.elf" >"$name/$stem.native.stdout" 2>"$name/$stem.native.stderr"
   nr=$?
  fi
  echo "$nr" > "$name/$stem.native.status"
  set -e
 done
 set +e
 ./reference "$name" >"$name/oracle.stdout" 2>"$name/oracle.stderr"
 local rc=$?
 echo "$rc" >"$name/ORACLE_STATUS"
 set -e
}
run_matrix before "$out/baseline-producer.elf"
run_matrix after "$out/candidate-producer.elf"
set +e
timeout 600 "$out/candidate-producer.elf" candidate.sio generation2-producer.elf >selfcompile.stdout 2>selfcompile.stderr
sc=$?
set -e
echo "$sc" > SELFCOMPILE_STATUS
if [[ "$sc" == 0 ]]; then
 chmod +x generation2-producer.elf
 run_matrix generation2 "$out/generation2-producer.elf"
fi
# Independent oracle corruption controls. Hardlinks retain exact ELFs; each
# mutated output is unlinked before replacement, preserving original receipts.
for control in wrong_value duplicate_output missing_output; do
 cp -al after "control-$control"
 target="control-$control/flow_component.native.stdout"
 rm "$target"
 if [[ "$control" == wrong_value ]]; then printf 'FLOW 0\n' > "$target"; fi
 if [[ "$control" == duplicate_output ]]; then printf 'FLOW 11\nFLOW 11\n' > "$target"; fi
 set +e
 ./reference "control-$control" > "$control.oracle.stdout" 2>"$control.oracle.stderr"
 rc=$?
 set -e
 echo "$rc" > "$control.ORACLE_STATUS"
 if [[ "$control" == missing_output ]]; then [[ "$rc" == 3 ]]; else [[ "$rc" == 4 ]]; fi
done
sha256sum -c INPUTS.before.sha256 > INPUTS.final.verify
find . -type f -name '*.elf' -print0 | sort -z | xargs -0 sha256sum > ELFS.sha256
[[ "$(cat after/ORACLE_STATUS)" == 0 && "$sc" == 0 && "$(cat generation2/ORACLE_STATUS)" == 0 ]]
