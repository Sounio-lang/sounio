#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
MANIFEST=tools/loom/kernel_peer_activation_capsule_authority.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-activation-capsule-authority-v1-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-activation-capsule-authority-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}
file_hash() { sha256sum "$1" | cut -d ' ' -f 1; }
stream_hash() { sha256sum | cut -d ' ' -f 1; }
record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in $path"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}
field() { record_field "$MANIFEST" "$1"; }
evidence_field() { record_field "$EVIDENCE" "$1"; }
hash_u32_csv() {
  local hash="$1" values='' offset chunk value
  for offset in 0 8 16 24 32 40 48 56; do
    chunk="${hash:$offset:8}"
    value="$((16#$chunk))"
    if [[ -n "$values" ]]; then values="$values,$value"; else values="$value"; fi
  done
  printf '%s' "$values"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" && -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] ||
  fail 'freeze inputs are absent or linked'
! grep -Fq '__' "$MANIFEST" || fail 'manifest contains an unresolved marker'
! grep -Fq '__' "$EVIDENCE" || fail 'evidence contains an unresolved marker'

[[ "$(field schema)" == loom-kernel-peer-activation-capsule-authority-freeze-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'action 9031 semantics are not frozen'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role drifted'
[[ "$(field action)" == 9031 && "$(field parent_actions)" == 9025,9030 ]] || fail 'action lineage drifted'
[[ "$(field write_shape_validation)" == pre_mutation ]] || fail 'write validation moved after mutation'
[[ "$(field absence_model)" == registry_absent+kernel_extinct+replay_refused ]] || fail 'absence triplet drifted'
[[ "$(field capsule_is_bearer)" == false ]] || fail 'capsule became a bearer capability'
[[ "$(field one_shot)" == true ]] || fail 'capsule is no longer affine'
for boundary in operational_realization capsule_material production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed; do
  [[ "$(field "$boundary")" == false ]] || fail "$boundary was promoted during semantic freeze"
done

GARDEN_COMMIT="$(field garden_commit)"
SOURCE_COMMIT="$(field sounio_executable_commit)"
git cat-file -e "${GARDEN_COMMIT}^{commit}" || fail 'Garden commit is absent'
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'Sounio executable commit is absent'
[[ "$(git rev-parse "${SOURCE_COMMIT}^")" == "$GARDEN_COMMIT" ]] ||
  fail 'Sounio executable did not immediately follow preregistration'

for pair in garden_path:garden_sha256 source_path:source_sha256 entrypoint_path:entrypoint_sha256 build_script_path:build_script_sha256 command_script_path:command_script_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  path="$(field "$path_key")"; expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  commit="$SOURCE_COMMIT"
  [[ "$path_key" == garden_path ]] && commit="$GARDEN_COMMIT"
  [[ "$(git show "$commit:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from its frozen commit"
done
[[ "$(file_hash "$(field freeze_selftest_path)")" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze selftest drifted'
[[ "$(hash_u32_csv "$(field source_sha256)")" == "$(field source_sha256_u32)" ]] ||
  fail 'source digest limbs drifted'

semantics_hash="$(sed -n '1,$p' "$(field source_path)" "$(field entrypoint_path)" | stream_hash)"
[[ "$semantics_hash" == "$(field semantics_sha256)" ]] || fail 'semantic bundle drifted'
[[ "$(hash_u32_csv "$semantics_hash")" == "$(field semantics_sha256_u32)" ]] ||
  fail 'semantic digest limbs drifted'

PARENT_9025="$(field parent_9025_manifest_path)"
PARENT_9030="$(field parent_9030_manifest_path)"
[[ "$(file_hash "$PARENT_9025")" == "$(field parent_9025_manifest_sha256)" ]] || fail 'action 9025 parent drifted'
[[ "$(file_hash "$PARENT_9030")" == "$(field parent_9030_manifest_sha256)" ]] || fail 'action 9030 parent drifted'
[[ "$(record_field "$PARENT_9025" producing_language)" == Sounio && "$(record_field "$PARENT_9025" language_role)" == SEMANTIC_AUTHORITY ]] || fail 'action 9025 parent authority drifted'
[[ "$(record_field "$PARENT_9025" action)" == 9025 && "$(record_field "$PARENT_9025" stage)" == SOUNIO_MATERIAL_JUDGMENT_FROZEN_V13 ]] || fail 'wrong action 9025 parent'
for fact in action_9025_allow same_uid_peer_isolation material_execution; do
  [[ "$(record_field "$PARENT_9025" "$fact")" == true ]] || fail "action 9025 parent omitted $fact"
done
[[ "$(record_field "$PARENT_9030" producing_language)" == Sounio && "$(record_field "$PARENT_9030" language_role)" == SEMANTIC_AUTHORITY ]] || fail 'action 9030 parent authority drifted'
[[ "$(record_field "$PARENT_9030" action)" == 9030 && "$(record_field "$PARENT_9030" stage)" == SEMANTICS_FROZEN ]] || fail 'wrong action 9030 parent'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-activation-capsule-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
toolchain_root="$work/toolchain"
runtime_a="$work/runtime-a"
runtime_b="$work/runtime-b"
mkdir -p "$toolchain_root"
wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
git archive "$SOURCE_COMMIT" "$wrapper_path" "$compiler_path" | tar -x -C "$toolchain_root"
[[ "$(file_hash "$toolchain_root/$wrapper_path")" == "$(field toolchain_wrapper_sha256)" ]] || fail 'frozen wrapper drifted'
[[ "$(file_hash "$toolchain_root/$compiler_path")" == "$(field toolchain_compiler_sha256)" ]] || fail 'frozen compiler drifted'
toolchain_hash="$({
  printf '%s\n' "engine=$(field toolchain_engine)"
  printf '%s\n' "wrapper=$wrapper_path"
  printf '%s\n' "wrapper_sha256=$(field toolchain_wrapper_sha256)"
  printf '%s\n' "compiler=$compiler_path"
  printf '%s\n' "compiler_sha256=$(field toolchain_compiler_sha256)"
} | stream_hash)"
[[ "$toolchain_hash" == "$(field toolchain_record_sha256)" ]] || fail 'toolchain record drifted'
[[ "$(hash_u32_csv "$toolchain_hash")" == "$(field toolchain_record_sha256_u32)" ]] || fail 'toolchain limbs drifted'

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
  printf '%s\n' 'parent_9025_material=ALLOW'
  printf '%s\n' 'parent_9030_material=SEMANTICS_FROZEN'
  printf '%s\n' 'same_uid_peer_isolation=true'
  printf '%s\n' 'production_activation=false'
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail 'hardware record drifted'
[[ "$(hash_u32_csv "$hardware_hash")" == "$(field hardware_record_sha256_u32)" ]] || fail 'hardware limbs drifted'

SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_SOUC="$toolchain_root/$wrapper_path" SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT="$runtime_a" bash "$(field build_script_path)" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_SOUC="$toolchain_root/$wrapper_path" SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT="$runtime_b" bash "$(field build_script_path)" >/dev/null
cmp "$runtime_a" "$runtime_b" || fail 'two frozen-toolchain rebuilds differ'
[[ "$(file_hash "$runtime_a")" == "$(field executable_sha256)" ]] || fail 'rebuilt executable drifted'
selftest="$(printf '0\n' | "$runtime_a")"
[[ "$(printf '%s\n' "$selftest" | stream_hash)" == "$(field sounio_selftest_sha256)" ]] || fail 'Sounio selftest drifted'
fixtures="$(printf '1\n' | "$runtime_a")"
[[ "$(printf '%s\n' "$fixtures" | stream_hash)" == "$(field fixture_bundle_sha256)" ]] || fail 'Sounio fixture bundle drifted'
[[ "$(printf '%s\n' "$fixtures" | grep -c '^CASE ')" == "$(field cases)" ]] || fail 'fixture count drifted'

for label in seal consume extinguish poison current_material kernel_drift principal_drift postwrite bearer silent_absence python_oracle unbound_result incomplete_sabotage wrong_stage wrong_parent malformed_flag; do
  line="$(printf '%s\n' "$fixtures" | sed -n "/^CASE label=${label} /p")"
  [[ -n "$line" ]] || fail "$label fixture is absent"
  frame="${line#* FRAME }"
  decision="$(printf '%s\n' "$frame" | "$runtime_a" || true)"
  [[ "$decision" == "$(field "${label}_decision")" ]] || fail "$label decision drifted"
  [[ "$(printf '%s\n' "$decision" | stream_hash)" == "$(field "${label}_decision_sha256")" ]] || fail "$label decision hash drifted"
done

command="$(field command)"
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'gate command hash drifted'
result="$(bash "$(field command_script_path)")"
[[ "$result" == "$(field result)" ]] || fail 'source-fresh gate result drifted'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'gate result hash drifted'

[[ "$(evidence_field schema)" == loom-kernel-peer-activation-capsule-authority-evidence-v1 ]] || fail 'evidence schema drifted'
[[ "$(evidence_field stage)" == SEMANTICS_FROZEN ]] || fail 'evidence stage drifted'
[[ "$(evidence_field manifest_sha256)" == "$(file_hash "$MANIFEST")" ]] || fail 'evidence does not bind the manifest'
for key in producing_language language_role action parent_actions garden_commit sounio_executable_commit source_sha256 entrypoint_sha256 semantics_sha256 fixture_bundle_sha256 parent_9025_manifest_sha256 parent_9030_manifest_sha256 executable_sha256 command result result_sha256 write_shape_validation absence_model capsule_is_bearer one_shot operational_realization capsule_material production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed; do
  [[ "$(evidence_field "$key")" == "$(field "$key")" ]] || fail "evidence $key drifted"
done

printf 'sounio-loom-kernel-peer-activation-capsule-authority-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN action=9031 parents=9025+9030 semantics_sha256=%s manifest_sha256=%s rebuilds=2 cases=16 positive=ALLOWx4 causal_sabotage=ALLOWx9 current_material=DENY502 absence_model=AFFIRMATIVE_TRIPLE python_oracle=DENY508 python_executed=false rust_executed=false operational_realization=false capsule_material=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=OCAML_OPERATIONAL_REALIZATION\n' \
  "$semantics_hash" "$(file_hash "$MANIFEST")"
