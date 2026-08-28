#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-kernel-exec-grant-cell-authority-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-kernel-exec-grant-cell-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/kernel-exec-grant-cell-one"
RUNTIME_TWO="$TEST_ROOT/kernel-exec-grant-cell-two"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-exec-grant-cell-authority-freeze-selftest: FAIL: %s\n' "$*" >&2
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
[[ "$(field schema)" == loom-kernel-exec-grant-cell-authority-freeze-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'manifest is not frozen'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field action)" == 9030 ]] || fail 'unexpected Sounio action'
[[ "$(field parent_actions)" == 9029,9021,9022 ]] || fail 'unexpected parent actions'
[[ "$(field material_grant)" == false ]] || fail 'freeze promoted a material grant'
[[ "$(field same_uid_peer_isolation)" == false ]] || fail 'freeze promoted same-UID isolation'
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
build_script_path="$(field build_script_path)"
gate_script_path="$(field command_script_path)"
freeze_script_path="$(field freeze_selftest_path)"

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
fixtures_hash="$(sed -n '/^parent_9029=/,/^malformed_flag=/p' "$ROOT_DIR/$gate_script_path" | stream_hash)"
[[ "$fixtures_hash" == "$(field fixture_bundle_sha256)" ]] || fail 'fixture bundle drifted'

parent_9029="$ROOT_DIR/$(field parent_9029_manifest_path)"
parent_9021="$ROOT_DIR/$(field parent_9021_manifest_path)"
parent_9022="$ROOT_DIR/$(field parent_9022_manifest_path)"
[[ "$(file_hash "$parent_9029")" == "$(field parent_9029_manifest_sha256)" ]] || fail 'parent action 9029 manifest drifted'
[[ "$(file_hash "$parent_9021")" == "$(field parent_9021_manifest_sha256)" ]] || fail 'parent action 9021 manifest drifted'
[[ "$(file_hash "$parent_9022")" == "$(field parent_9022_manifest_sha256)" ]] || fail 'parent action 9022 manifest drifted'
for parent in "$parent_9029" "$parent_9021" "$parent_9022"; do
  [[ "$(record_field "$parent" stage)" == SEMANTICS_FROZEN ]] || fail "$parent is not frozen"
  [[ "$(record_field "$parent" producing_language)" == Sounio ]] || fail "$parent is not Sounio-produced"
  [[ "$(record_field "$parent" language_role)" == SEMANTIC_AUTHORITY ]] || fail "$parent role drifted"
done
[[ "$(record_field "$parent_9029" action)" == 9029 ]] || fail 'wrong action 9029 parent'
[[ "$(record_field "$parent_9022" action)" == 9022 ]] || fail 'wrong action 9022 parent'
[[ "$(record_field "$parent_9029" semantics_sha256)" == "$(field parent_9029_semantics_sha256)" ]] || fail 'parent 9029 semantics differ'
[[ "$(record_field "$parent_9021" semantics_sha256)" == "$(field parent_9021_semantics_sha256)" ]] || fail 'parent 9021 semantics differ'
[[ "$(record_field "$parent_9022" semantics_sha256)" == "$(field parent_9022_semantics_sha256)" ]] || fail 'parent 9022 semantics differ'

wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
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

cpu_model="$(awk -F: '/model name/{gsub(/^[ \t]+/, "", $2); print toupper($2); exit}' /proc/cpuinfo)"
pid1="$(tr '\0' ' ' </proc/1/cmdline | awk '{print $1}')"
hardware_hash="$({
  printf '%s\n' "kernel=$(uname -r)"
  printf '%s\n' "architecture=$(uname -m)"
  printf '%s\n' "logical_cpus=$(getconf _NPROCESSORS_ONLN)"
  printf '%s\n' "cpu_model=$cpu_model"
  printf '%s\n' "outer_uid=$(id -u)"
  printf '%s\n' "outer_gid=$(id -g)"
  printf '%s\n' "pid1=$pid1"
  printf '%s\n' 'host_service_boundary=absent'
  printf '%s\n' 'parent_9029_material=DENY481'
  printf '%s\n' 'parent_9021_material=SEMANTICS_FROZEN'
  printf '%s\n' 'parent_9022_material=SEMANTICS_FROZEN'
  printf '%s\n' 'same_uid_peer_isolation=false'
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail 'hardware record hash differs'
[[ "$(hash_u32_csv "$hardware_hash")" == "$(field hardware_record_sha256_u32)" ]] || fail 'hardware limbs differ'

SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$RUNTIME_ONE" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$RUNTIME_TWO" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio rebuilds differ'
[[ "$(file_hash "$RUNTIME_ONE")" == "$(field executable_sha256)" ]] || fail 'rebuilt executable hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_kernel_exec_grant_cell_authority_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$gate_script_path")"
[[ "$result" == "$(field result)" ]] || fail 'gate result differs from frozen Sounio result'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'gate result hash differs'

parent_9029_limbs='1636926980 3205986131 3323207532 3505413428 706242987 2411760920 1929815169 3727939342'
parent_9021_limbs='3497534264 556131944 3943529214 1565657389 3821375173 3204015455 2733765994 2625951936'
parent_9022_limbs='4125506095 3601417934 2711931735 20635855 2708941890 3284947684 758124027 2068177262'
one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
bindings="$parent_9029_limbs $parent_9021_limbs $parent_9022_limbs $one $one $one $one $one $one"
issue_transition='1 0 1 2 3 4 5 6 7 100 50'
consume_transition='2 1 3 2 3 4 5 6 7 100 50'
close_transition='3 3 4 2 3 4 5 6 7 100 50'
revoke_transition='4 1 5 2 3 4 5 6 7 100 50'
issue_parents='1 1 0 0 0 1 0 0'
consume_parents='1 1 1 0 0 1 1 0'
close_parents='1 1 1 1 0 1 1 1'
revoke_parents='1 1 0 0 1 1 0 0'
identity='1 1 1 1 1 1 1 1'
peer='1 1 1 1 1 1 1 1 1'
shape='1 1 1 1 1 1 1 1'
consumption='1 1 1 1 1 1 1'
revocation='1 1 1 1 1 1 1'
live_extinction='0 0 0 0 1'
terminal_extinction='1 1 1 1 1'
live_outcome='0 0 0 0 0 0 0 0'
close_outcome='1 1 1 1 1 1 0 0'
revoke_outcome='0 0 0 0 0 1 1 1'
authority='1 1 1 1 1 1 1'
evidence='1 1 11 11'
issue_frame="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
consume_frame="9030 3 1 $consume_transition $consume_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
close_frame="9030 3 1 $close_transition $close_parents $identity $peer $shape $consumption $revocation $terminal_extinction $close_outcome $authority $evidence $bindings"
revoke_frame="9030 3 1 $revoke_transition $revoke_parents $identity $peer $shape $consumption $revocation $terminal_extinction $revoke_outcome $authority $evidence $bindings"
current_frame="9030 3 1 $issue_transition 1 0 0 0 0 0 0 0 $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
extinction_frame="9030 3 1 $close_transition $close_parents $identity $peer $shape $consumption $revocation 0 1 1 1 1 $close_outcome $authority $evidence $bindings"
python_frame="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome 0 0 0 1 1 1 1 $evidence $bindings"
unbound_frame="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $parent_9029_limbs $parent_9021_limbs $parent_9022_limbs $one $one $one $one $one $zero"

for pair in issue:"$issue_frame" consume:"$consume_frame" close:"$close_frame" revoke:"$revoke_frame" current:"$current_frame" extinction:"$extinction_frame" python:"$python_frame" unbound:"$unbound_frame"; do
  name="${pair%%:*}"
  frame="${pair#*:}"
  decision="$(printf '%s\n' "$frame" | "$RUNTIME_ONE" || true)"
  [[ "$decision" == "$(field "${name}_decision")" ]] || fail "$name decision differs"
  [[ "$(printf '%s\n' "$decision" | stream_hash)" == "$(field "${name}_decision_sha256")" ]] || fail "$name decision hash differs"
done

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the manifest hash'

printf '%s\n' \
  "sounio-loom-kernel-exec-grant-cell-authority-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN action=9030 parents=9029+9021+9022 semantics_sha256=$semantics_hash manifest_sha256=$manifest_hash rebuilds=2 cases=21 causal_sabotage=ALLOWx11 current_material=DENY491 extinction=AFFIRMATIVE_TRIPLE python_oracle=DENY499 python_executed=false material_grant=false same_uid_peer_isolation=false parity_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false"
