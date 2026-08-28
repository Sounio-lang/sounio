#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/kernel_principal_authority.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-kernel-principal-authority-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-kernel-principal-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/kernel-principal-one"
RUNTIME_TWO="$TEST_ROOT/kernel-principal-two"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-principal-authority-freeze-selftest: FAIL: %s\n' "$*" >&2
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
[[ "$(field schema)" == loom-kernel-principal-authority-freeze-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'manifest is not frozen'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field action)" == 9026 ]] || fail 'unexpected Sounio action'
[[ "$(field parent_action)" == 9025 ]] || fail 'unexpected parent action'
[[ "$(field material_isolation)" == false ]] || fail 'freeze promoted material isolation'
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

parent_manifest="$ROOT_DIR/$(field parent_9025_manifest_path)"
[[ "$(file_hash "$parent_manifest")" == "$(field parent_9025_manifest_sha256)" ]] || fail 'parent action 9025 manifest drifted'
[[ "$(record_field "$parent_manifest" stage)" == SEMANTICS_FROZEN ]] || fail 'action 9025 is not frozen'
[[ "$(record_field "$parent_manifest" action)" == 9025 ]] || fail 'wrong parent action'
[[ "$(record_field "$parent_manifest" semantics_sha256)" == "$(field parent_9025_semantics_sha256)" ]] || fail 'parent semantics binding differs'

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
  printf '%s\n' "subuid_range=$(field hardware_subuid_range)"
  printf '%s\n' "subgid_range=$(field hardware_subgid_range)"
  printf '%s\n' "newuidmap=$(field hardware_newuidmap)"
  printf '%s\n' "newgidmap=$(field hardware_newgidmap)"
  printf '%s\n' "cgroup=$(field hardware_cgroup)"
  printf '%s\n' "ptrace_scope=$(field hardware_ptrace_scope)"
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail 'hardware record hash differs'
[[ "$(hash_u32_csv "$hardware_hash")" == "$(field hardware_record_sha256_u32)" ]] || fail 'hardware limbs differ'

SOUNIO_LOOM_KERNEL_PRINCIPAL_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_KERNEL_PRINCIPAL_OUTPUT="$RUNTIME_ONE" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_KERNEL_PRINCIPAL_OUTPUT="$RUNTIME_TWO" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio rebuilds differ'
[[ "$(file_hash "$RUNTIME_ONE")" == "$(field executable_sha256)" ]] || fail 'rebuilt executable hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_kernel_principal_authority_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$gate_script_path")"
[[ "$result" == "$(field result)" ]] || fail 'gate result differs from frozen Sounio result'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'gate result hash differs'

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
all_bindings="$one $one $one $one $one $one $one $one $one $one $one $one"
substrate='1 1 1 1 1 1'
mapping='1 1 1 1 1'
identities='1000 100000 100001 1000 100000 100001'
peer='1 1 1 1 1 1 1 1'
isolation='1 1 1 1 1 1'
grant='1 1 1 1 1'
valid_frame="9026 3 1 $substrate $mapping $identities $peer $isolation $grant 5 5 1 $all_bindings"
current_frame="9026 3 1 $substrate 0 0 0 0 1 1000 1000 1000 1000 1000 1000 $peer 0 0 0 0 0 0 $grant 0 5 1 $one $zero $zero $zero $one $zero $one $one $one $one $one $one"
equal_frame="9026 3 1 $substrate $mapping 1000 1000 100001 1000 100000 100001 $peer $isolation $grant 5 5 1 $all_bindings"
injectable_frame="9026 3 1 $substrate $mapping $identities $peer 1 1 1 0 1 1 $grant 5 5 1 $all_bindings"
unbound_frame="9026 3 1 $substrate $mapping $identities $peer $isolation $grant 5 5 1 $one $one $one $one $one $zero $one $one $one $one $one $one"

valid_decision="$(printf '%s\n' "$valid_frame" | "$RUNTIME_ONE")"
current_material_decision="$(printf '%s\n' "$current_frame" | "$RUNTIME_ONE" || true)"
equal_principal_decision="$(printf '%s\n' "$equal_frame" | "$RUNTIME_ONE" || true)"
injectable_decision="$(printf '%s\n' "$injectable_frame" | "$RUNTIME_ONE" || true)"
unbound_decision="$(printf '%s\n' "$unbound_frame" | "$RUNTIME_ONE" || true)"
for decision_name in valid current_material equal_principal injectable unbound; do
  decision_var="${decision_name}_decision"
  actual="${!decision_var}"
  [[ "$actual" == "$(field "${decision_name}_decision")" ]] || fail "$decision_name decision differs"
  [[ "$(printf '%s\n' "$actual" | stream_hash)" == "$(field "${decision_name}_decision_sha256")" ]] || fail "$decision_name decision hash differs"
done

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the manifest hash'

printf '%s\n' \
  "sounio-loom-kernel-principal-authority-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN parent_action=9025 semantics_sha256=$semantics_hash manifest_sha256=$manifest_hash rebuilds=2 cases=17 causal_sabotage=ALLOWx5 current_material=DENY455 material_isolation=false parity_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false"
