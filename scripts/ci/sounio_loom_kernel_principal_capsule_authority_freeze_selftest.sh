#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/kernel_principal_capsule_authority.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-kernel-principal-capsule-authority-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-kernel-principal-capsule-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/kernel-principal-capsule-one"
RUNTIME_TWO="$TEST_ROOT/kernel-principal-capsule-two"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-principal-capsule-authority-freeze-selftest: FAIL: %s\n' "$*" >&2
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
[[ "$(field schema)" == loom-kernel-principal-capsule-authority-freeze-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'manifest is not frozen'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field action)" == 9028 ]] || fail 'unexpected Sounio action'
[[ "$(field parent_action)" == 9027 ]] || fail 'unexpected parent action'
[[ "$(field grandparent_action)" == 9026 ]] || fail 'unexpected grandparent action'
[[ "$(field material_capsule)" == false ]] || fail 'freeze promoted a material capsule'
[[ "$(field parity_open)" == false ]] || fail 'freeze opened parity'
[[ "$(field claim_ready)" == false ]] || fail 'freeze promoted a claim'
for surface in exec_attached commit_attached ci_attached; do
  [[ "$(field "$surface")" == false ]] || fail "$surface was promoted during freeze"
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

parent_9026_manifest="$ROOT_DIR/$(field parent_9026_manifest_path)"
parent_9027_manifest="$ROOT_DIR/$(field parent_9027_manifest_path)"
[[ "$(file_hash "$parent_9026_manifest")" == "$(field parent_9026_manifest_sha256)" ]] || fail 'parent action 9026 manifest drifted'
[[ "$(file_hash "$parent_9027_manifest")" == "$(field parent_9027_manifest_sha256)" ]] || fail 'parent action 9027 manifest drifted'
[[ "$(record_field "$parent_9026_manifest" action)" == 9026 ]] || fail 'wrong grandparent action'
[[ "$(record_field "$parent_9027_manifest" action)" == 9027 ]] || fail 'wrong parent action'
[[ "$(record_field "$parent_9026_manifest" stage)" == SEMANTICS_FROZEN ]] || fail 'action 9026 is not frozen'
[[ "$(record_field "$parent_9027_manifest" stage)" == SEMANTICS_FROZEN ]] || fail 'action 9027 is not frozen'
[[ "$(record_field "$parent_9026_manifest" semantics_sha256)" == "$(field parent_9026_semantics_sha256)" ]] || fail 'parent 9026 semantics differ'
[[ "$(record_field "$parent_9027_manifest" semantics_sha256)" == "$(field parent_9027_semantics_sha256)" ]] || fail 'parent 9027 semantics differ'

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
  printf '%s\n' "parent_9026_material=$(field hardware_parent_9026_material)"
  printf '%s\n' "parent_9027_material=$(field hardware_parent_9027_material)"
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail 'hardware record hash differs'
[[ "$(hash_u32_csv "$hardware_hash")" == "$(field hardware_record_sha256_u32)" ]] || fail 'hardware limbs differ'

SOUNIO_LOOM_KERNEL_PRINCIPAL_CAPSULE_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_KERNEL_PRINCIPAL_CAPSULE_OUTPUT="$RUNTIME_ONE" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_CAPSULE_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_KERNEL_PRINCIPAL_CAPSULE_OUTPUT="$RUNTIME_TWO" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio rebuilds differ'
[[ "$(file_hash "$RUNTIME_ONE")" == "$(field executable_sha256)" ]] || fail 'rebuilt executable hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_kernel_principal_capsule_authority_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$gate_script_path")"
[[ "$result" == "$(field result)" ]] || fail 'gate result differs from frozen Sounio result'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'gate result hash differs'

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
bindings="$one $one $one $one $one $one $one $one $one $one $one $one $one"
mint_order='1 1 1 2 3 1 1'
recovery_order='2 1 1 5 5 0 0'
identity='100 200 300 4 5 6 1 1 1'
isolation='1 1 1 1 1 1 1'
privilege='1 1 1 1 1 1 1 1'
custody='1 1 0 0 0 0 1'
mint_grant='1 1 1 1 1 1 1 1 1'
recovery_grant='1 0 0 0 0 0 0 0 1'
mint_recovery='0 3 3 5 5 0 0 0 0 0'
recovery='1 3 4 5 6 1 1 1 1 1'
evidence='9 9 1'
valid_mint_frame="9028 3 1 $mint_order $identity $isolation $privilege $custody $mint_grant $mint_recovery $evidence $bindings"
valid_recovery_frame="9028 3 1 $recovery_order $identity $isolation $privilege $custody $recovery_grant $recovery $evidence $bindings"
current_frame="9028 3 1 1 0 0 2 3 1 1 $identity $isolation $privilege $custody $mint_grant $mint_recovery $evidence $bindings"
bare_pid_frame="9028 3 1 $mint_order $identity $isolation $privilege 1 1 0 0 1 0 1 $mint_grant $mint_recovery $evidence $bindings"
unbound_frame="9028 3 1 $mint_order $identity $isolation $privilege $custody $mint_grant $mint_recovery $evidence $one $one $one $one $one $one $one $one $one $one $one $one $zero"

valid_mint_decision="$(printf '%s\n' "$valid_mint_frame" | "$RUNTIME_ONE")"
valid_recovery_decision="$(printf '%s\n' "$valid_recovery_frame" | "$RUNTIME_ONE")"
current_material_decision="$(printf '%s\n' "$current_frame" | "$RUNTIME_ONE" || true)"
bare_pid_decision="$(printf '%s\n' "$bare_pid_frame" | "$RUNTIME_ONE" || true)"
unbound_decision="$(printf '%s\n' "$unbound_frame" | "$RUNTIME_ONE" || true)"
for decision_name in valid_mint valid_recovery current_material bare_pid unbound; do
  decision_var="${decision_name}_decision"
  actual="${!decision_var}"
  [[ "$actual" == "$(field "${decision_name}_decision")" ]] || fail "$decision_name decision differs"
  [[ "$(printf '%s\n' "$actual" | stream_hash)" == "$(field "${decision_name}_decision_sha256")" ]] || fail "$decision_name decision hash differs"
done

fixture_hash="$({
  printf '%s\n' "$valid_mint_decision"
  printf '%s\n' "$valid_recovery_decision"
  printf '%s\n' "$current_material_decision"
  printf '%s\n' "$bare_pid_decision"
  printf '%s\n' "$unbound_decision"
} | stream_hash)"
[[ "$fixture_hash" == "$(field fixture_bundle_sha256)" ]] || fail 'decision fixture bundle hash differs'

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the manifest hash'

printf '%s\n' \
  "sounio-loom-kernel-principal-capsule-authority-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN parent_action=9027 grandparent_action=9026 semantics_sha256=$semantics_hash manifest_sha256=$manifest_hash fixture_sha256=$fixture_hash rebuilds=2 cases=19 causal_sabotage=ALLOWx9 current_material=DENY472 bare_pid=DENY476 material_capsule=false parity_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false"
