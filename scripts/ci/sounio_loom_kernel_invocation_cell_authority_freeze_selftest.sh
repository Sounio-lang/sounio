#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-kernel-invocation-cell-authority-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-kernel-invocation-cell-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/kernel-invocation-cell-one"
RUNTIME_TWO="$TEST_ROOT/kernel-invocation-cell-two"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-invocation-cell-authority-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "record field $key occurs $count times in $path"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

stream_hash() {
  local sum
  sum="$(sha256sum)"
  printf '%s' "${sum%% *}"
}

hash_u32_csv() {
  local hash="$1" values='' offset chunk value
  for offset in 0 8 16 24 32 40 48 56; do
    chunk="${hash:$offset:8}"
    value="$((16#$chunk))"
    if [[ -n "$values" ]]; then values="$values,$value"; else values="$value"; fi
  done
  printf '%s' "$values"
}

[[ -f "$MANIFEST" ]] || fail 'freeze manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'freeze evidence is missing'
[[ "$(field schema)" == loom-kernel-invocation-cell-authority-freeze-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'manifest is not frozen'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field action)" == 9029 ]] || fail 'unexpected Sounio action'
[[ "$(field parent_actions)" == 9028,9025,9023 ]] || fail 'unexpected parent actions'
for boundary in material_invocation material_coverage same_uid_peer_isolation parity_open claim_ready exec_attached commit_attached ci_attached; do
  [[ "$(field "$boundary")" == false ]] || fail "$boundary was promoted during freeze"
done

garden_commit="$(field garden_commit)"
executable_commit="$(field sounio_executable_commit)"
git -C "$ROOT_DIR" cat-file -e "${garden_commit}^{commit}" || fail 'Garden commit is absent'
git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" || fail 'Sounio executable commit is absent'
[[ "$(git -C "$ROOT_DIR" rev-parse "${executable_commit}^")" == "$garden_commit" ]] ||
  fail 'Sounio executable commit does not immediately follow Garden'

garden_path="$(field garden_path)"
source_path="$(field source_path)"
entrypoint_path="$(field entrypoint_path)"
[[ "$(file_hash "$ROOT_DIR/$garden_path")" == "$(field garden_sha256)" ]] || fail 'working Garden drifted'
[[ "$(file_hash "$ROOT_DIR/$source_path")" == "$(field source_sha256)" ]] || fail 'working source drifted'
[[ "$(file_hash "$ROOT_DIR/$entrypoint_path")" == "$(field entrypoint_sha256)" ]] || fail 'working entrypoint drifted'
[[ "$(git -C "$ROOT_DIR" show "$garden_commit:$garden_path" | stream_hash)" == "$(field garden_sha256)" ]] || fail 'Garden commit hash differs'
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$source_path" | stream_hash)" == "$(field source_sha256)" ]] || fail 'frozen source differs'
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$entrypoint_path" | stream_hash)" == "$(field entrypoint_sha256)" ]] || fail 'frozen entrypoint differs'
[[ "$(hash_u32_csv "$(field source_sha256)")" == "$(field source_sha256_u32)" ]] || fail 'source limbs differ'

semantics_hash="$(sed -n '1,$p' "$ROOT_DIR/$source_path" "$ROOT_DIR/$entrypoint_path" | stream_hash)"
[[ "$semantics_hash" == "$(field semantics_sha256)" ]] || fail 'semantic bundle hash differs'
[[ "$(hash_u32_csv "$semantics_hash")" == "$(field semantics_sha256_u32)" ]] || fail 'semantics limbs differ'

for action in 9028 9025 9023; do
  manifest="$ROOT_DIR/$(field "parent_${action}_manifest_path")"
  [[ "$(file_hash "$manifest")" == "$(field "parent_${action}_manifest_sha256")" ]] || fail "parent action $action manifest drifted"
  [[ "$(record_field "$manifest" action)" == "$action" ]] || fail "wrong parent action $action"
  [[ "$(record_field "$manifest" stage)" == SEMANTICS_FROZEN ]] || fail "parent action $action is not frozen"
  [[ "$(record_field "$manifest" semantics_sha256)" == "$(field "parent_${action}_semantics_sha256")" ]] || fail "parent action $action semantics differ"
done

wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
build_script_path="$(field build_script_path)"
gate_script_path="$(field command_script_path)"
freeze_script_path="$(field freeze_selftest_path)"
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$executable_commit" "$wrapper_path" "$compiler_path" | tar -x -C "$TOOLCHAIN_ROOT"
[[ "$(file_hash "$TOOLCHAIN_ROOT/$wrapper_path")" == "$(field toolchain_wrapper_sha256)" ]] || fail 'frozen compiler wrapper drifted'
[[ "$(file_hash "$TOOLCHAIN_ROOT/$compiler_path")" == "$(field toolchain_compiler_sha256)" ]] || fail 'frozen compiler binary drifted'
[[ "$(file_hash "$ROOT_DIR/$build_script_path")" == "$(field build_script_sha256)" ]] || fail 'build script drifted'
[[ "$(file_hash "$ROOT_DIR/$gate_script_path")" == "$(field command_script_sha256)" ]] || fail 'gate script drifted'
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$build_script_path" | stream_hash)" == "$(field build_script_sha256)" ]] || fail 'frozen build script differs'
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$gate_script_path" | stream_hash)" == "$(field command_script_sha256)" ]] || fail 'frozen gate script differs'
[[ "$(file_hash "$ROOT_DIR/$freeze_script_path")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze selftest drifted'

toolchain_hash="$({
  printf '%s\n' "engine=$(field toolchain_engine)"
  printf '%s\n' "wrapper=$wrapper_path"
  printf '%s\n' "wrapper_sha256=$(field toolchain_wrapper_sha256)"
  printf '%s\n' "compiler=$compiler_path"
  printf '%s\n' "compiler_sha256=$(field toolchain_compiler_sha256)"
} | stream_hash)"
[[ "$toolchain_hash" == "$(field toolchain_record_sha256)" ]] || fail 'toolchain record hash differs'
[[ "$(hash_u32_csv "$toolchain_hash")" == "$(field toolchain_record_sha256_u32)" ]] || fail 'toolchain limbs differ'

hardware_hash="$({
  printf '%s\n' "kernel=$(field hardware_kernel)"
  printf '%s\n' "architecture=$(field hardware_architecture)"
  printf '%s\n' "logical_cpus=$(field hardware_logical_cpus)"
  printf '%s\n' "cpu_model=$(field hardware_cpu_model)"
  printf '%s\n' "outer_uid=$(field hardware_outer_uid)"
  printf '%s\n' "outer_gid=$(field hardware_outer_gid)"
  printf '%s\n' "pid1=$(field hardware_pid1)"
  printf '%s\n' "host_service_boundary=$(field hardware_host_service_boundary)"
  printf '%s\n' "parent_9028_material=$(field hardware_parent_9028_material)"
  printf '%s\n' "parent_9025_material=$(field hardware_parent_9025_material)"
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail 'hardware record hash differs'
[[ "$(hash_u32_csv "$hardware_hash")" == "$(field hardware_record_sha256_u32)" ]] || fail 'hardware limbs differ'

SOUNIO_LOOM_KERNEL_INVOCATION_CELL_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$RUNTIME_ONE" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_KERNEL_INVOCATION_CELL_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$RUNTIME_TWO" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio rebuilds differ'
[[ "$(file_hash "$RUNTIME_ONE")" == "$(field executable_sha256)" ]] || fail 'rebuilt executable hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_kernel_invocation_cell_authority_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$gate_script_path")"
[[ "$result" == "$(field result)" ]] || fail 'gate result differs from frozen Sounio result'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'gate result hash differs'

parent_9028='1991017987 113822720 1367310835 4264184359 1117900107 2622180275 1259621157 4224578159'
parent_9025='3253784467 4165106381 4153681002 298013982 643434942 312724736 195896759 132696721'
parent_9023='2365323 2301161672 762924345 38070334 1558458629 1166539901 3590963442 1546541903'
one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
bindings="$parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $one"
capsule='1 1 5 6 7 1 1 0 0 1 1 1'
membrane='1 8 9 10 11 1 1 1 1'
scope='1 1 1 1 1 1'
coverage='1 100 1 50 1 1 1 1'
open_lifecycle='1 1 1 12 13 1 0 0 0'
close_lifecycle='1 1 1 12 13 1 0 1 0'
abort_lifecycle='1 1 1 12 13 1 0 0 1'
open_outcome='0 0 0 0 0 0 0 0 0 0'
close_outcome='14 1 1 1 1 1 0 0 0 0'
abort_outcome='14 0 1 1 1 1 1 1 1 1'
authority='1 1 1 1 1 1'
evidence='1 1 10 10'
prepare_frame="9029 3 1 1 1 1 1 1 1 $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
admit_frame="9029 3 1 2 2 1 1 1 1 $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
close_frame="9029 3 1 3 3 1 1 1 1 $capsule $membrane $scope $coverage $close_lifecycle $close_outcome $authority $evidence $bindings"
abort_frame="9029 3 1 4 4 1 1 1 1 $capsule $membrane $scope $coverage $abort_lifecycle $abort_outcome $authority $evidence $bindings"
current_frame="9029 3 1 1 1 0 0 1 0 $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
python_frame="9029 3 1 1 1 1 1 1 1 $capsule $membrane $scope $coverage $open_lifecycle $open_outcome 0 0 1 1 0 1 $evidence $bindings"
unbound_frame="9029 3 1 1 1 1 1 1 1 $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $zero"

for decision_name in prepare admit close abort current python unbound; do
  frame_var="${decision_name}_frame"
  actual="$(printf '%s\n' "${!frame_var}" | "$RUNTIME_ONE" || true)"
  [[ "$actual" == "$(field "${decision_name}_decision")" ]] || fail "$decision_name decision differs"
  [[ "$(printf '%s\n' "$actual" | stream_hash)" == "$(field "${decision_name}_decision_sha256")" ]] || fail "$decision_name decision hash differs"
  printf -v "${decision_name}_decision_actual" '%s' "$actual"
done

fixture_hash="$({
  printf '%s\n' "$prepare_decision_actual"
  printf '%s\n' "$admit_decision_actual"
  printf '%s\n' "$close_decision_actual"
  printf '%s\n' "$abort_decision_actual"
  printf '%s\n' "$current_decision_actual"
  printf '%s\n' "$python_decision_actual"
  printf '%s\n' "$unbound_decision_actual"
} | stream_hash)"
[[ "$fixture_hash" == "$(field fixture_bundle_sha256)" ]] || fail 'decision fixture bundle hash differs'

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the manifest hash'

printf '%s\n' \
  "sounio-loom-kernel-invocation-cell-authority-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN parent_actions=9028,9025,9023 semantics_sha256=$semantics_hash manifest_sha256=$manifest_hash fixture_sha256=$fixture_hash rebuilds=2 cases=19 causal_sabotage=ALLOWx10 current_material=DENY481 python_oracle=DENY488 material_invocation=false material_coverage=false same_uid_peer_isolation=false parity_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false"
