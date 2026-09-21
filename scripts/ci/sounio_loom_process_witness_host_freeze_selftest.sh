#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/process_witness_host.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-host-v1-20260828.txt"

fail() {
  printf 'sounio-loom-process-witness-host-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in ${path#$ROOT_DIR/}"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

field() { record_field "$MANIFEST" "$1"; }
evidence_field() { record_field "$EVIDENCE" "$1"; }

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

expect_field() {
  local key="$1" expected="$2" actual
  actual="$(field "$key")"
  [[ "$actual" == "$expected" ]] || fail "$key drifted: expected=$expected actual=$actual"
}

expect_evidence() {
  local key="$1" expected="$2" actual
  actual="$(evidence_field "$key")"
  [[ "$actual" == "$expected" ]] || fail "evidence $key drifted: expected=$expected actual=$actual"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'host ProcessWitness manifest is missing or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'host ProcessWitness evidence is missing or linked'

expect_field schema loom-process-witness-host-runtime-v1
expect_field stage MATERIAL_EXECUTION_CORE_FROZEN
expect_field producing_language C++20+Linux+systemd
expect_field language_role MATERIAL_PARITY
expect_field transitory true
expect_field semantic_authority Sounio
expect_field controller_language OCaml
expect_field controller_role EFFECT_PARITY
expect_field action 9030
for fact in process_witness_core affirmative_extinction principal_distinct_uid same_pid start_tick credential_unchanged cgroup_unchanged namespace_unchanged environment_empty descendants_empty state_extinct generation_extinct authority_extinct pidfd_extinct cgroup_unpopulated unit_inactive material_grant; do
  expect_field "$fact" true
done
expect_field pidfd_at_ready live
expect_field executable_transition cell-to-Sounio
expect_field causal_sabotage PASS
expect_field complete_effects false
expect_field material_execution false
expect_field production_activation false
expect_field production_current_unchanged true
expect_field production_broker_unchanged true
expect_field rollback identity-operation
for boundary in launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done

SOURCE_COMMIT="$(field source_commit)"
FREEZE_COMMIT="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'source commit is absent'
git -C "$ROOT_DIR" cat-file -e "${FREEZE_COMMIT}^{commit}" || fail 'freeze-gate commit is absent'

for pair in \
  garden_path:garden_sha256 \
  semantic_manifest_path:semantic_manifest_sha256 \
  parent_host_grant_manifest_path:parent_host_grant_manifest_sha256 \
  controller_manifest_path:controller_manifest_sha256 \
  fixture_manifest_path:fixture_manifest_sha256 \
  resident_manifest_path:resident_manifest_sha256 \
  payload_manifest_path:payload_manifest_sha256 \
  cell_source_path:cell_source_sha256 \
  cell_build_script_path:cell_build_script_sha256 \
  cell_selftest_path:cell_selftest_sha256 \
  process_witness_lab_path:process_witness_lab_sha256 \
  quorum_module_path:quorum_module_sha256 \
  broker_source_path:broker_source_sha256 \
  broker_build_script_path:broker_build_script_sha256 \
  host_gate_path:host_gate_sha256 \
  capsule_builder_path:capsule_builder_sha256 \
  promoter_path:promoter_sha256 \
  transport_path:transport_sha256 \
  local_quorum_selftest_path:local_quorum_selftest_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the source commit"
done

FREEZE_PATH="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$FREEZE_PATH")" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate drifted'
[[ "$(git -C "$ROOT_DIR" show "$FREEZE_COMMIT:$FREEZE_PATH" | stream_hash)" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate differs from its commit'

expect_evidence schema loom-process-witness-host-evidence-v1
expect_evidence stage MATERIAL_EXECUTION_CORE_MEASURED
expect_evidence semantic_authority Sounio
expect_evidence semantic_action 9030
expect_evidence source_commit "$SOURCE_COMMIT"
expect_evidence garden_sha256 "$(field garden_sha256)"
expect_evidence payload_manifest_sha256 "$(field payload_manifest_sha256)"
expect_evidence payload_sha256 "$(field payload_sha256)"
expect_evidence cell_sha256 "$(field cell_binary_sha256)"
expect_evidence capsule_sha256 "$(field capsule_sha256)"
expect_evidence release_id "$(field release_id)"
expect_evidence release_manifest_sha256 "$(field release_manifest_sha256)"
expect_evidence broker_output_sha256 "$(field broker_output_sha256)"
expect_evidence process_witness_output_sha256 "$(field process_witness_output_sha256)"
expect_evidence host_output_sha256 "$(field host_output_sha256)"
expect_evidence raw_receipt_sha256 "$(field raw_receipt_sha256)"
expect_evidence hardware_host "$(field hardware_host)"
expect_evidence hardware_kernel "$(field hardware_kernel)"
expect_evidence hardware_architecture "$(field hardware_architecture)"
expect_evidence systemd_version "$(field systemd_version)"
expect_evidence systemd_run_sha256 "$(field systemd_run_sha256)"
expect_evidence systemctl_sha256 "$(field systemctl_sha256)"
expect_evidence transport "$(field transport)"
expect_evidence command "$(field command)"
expect_evidence attempt_1 'FAIL_CLOSED resident-process-identity-unavailable'
expect_evidence attempt_2 'FAIL_CLOSED phase=positive resident-process-identity-unavailable'
expect_evidence attempt_3 'FAIL_CLOSED phase=positive-replay resident-process-identity-unavailable'
expect_evidence attempt_4 PASS
expect_evidence treatment closed
expect_evidence positive done
expect_evidence causal_bypass done
expect_evidence causal_sabotage PASS
expect_evidence wrong_generation closed
expect_evidence payload_substitution closed
expect_evidence forged_ready closed
expect_evidence wrong_close Sounio_refusal
expect_evidence controller_death closed
expect_evidence broker_death_after_ready closed
expect_evidence replay closed
expect_evidence controller_terminal_extinct true
expect_evidence pidfd_at_ready live
expect_evidence executable_transition cell-to-Sounio
expect_evidence extinction_omission closed
expect_evidence process_witness_core true
expect_evidence affirmative_extinction true
expect_evidence complete_effects false
expect_evidence material_grant true
expect_evidence material_execution false
expect_evidence production_current_unchanged true
expect_evidence production_broker_unchanged true
expect_evidence rollback identity-operation
for fact in same_pid start_tick credential_unchanged cgroup_unchanged namespace_unchanged environment_empty descendants_empty state_extinct generation_extinct authority_extinct pidfd_extinct cgroup_unpopulated unit_inactive; do
  expect_evidence "$fact" true
done
for boundary in launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_evidence "$boundary" false
done

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'host evidence hash drifted'
[[ "$(printf '%s\n' "$(field command)" | stream_hash)" == "$(field command_sha256)" ]] || fail 'host command hash drifted'
[[ "$(file_hash "$(field cxx_path)")" == "$(field cxx_sha256)" ]] || fail 'C++ compiler drifted'
[[ "$("$(field cxx_path)" --version | sed -n '1p')" == "$(field cxx_version)" ]] || fail 'C++ compiler version drifted'

transport_receipt="$(evidence_field transport_receipt)"
host_receipt="$(evidence_field host_gate_receipt)"
grant_receipt="$(evidence_field grant_receipt)"
witness_receipt="$(evidence_field process_witness_receipt)"
install_receipt="$(evidence_field install_receipt)"
[[ "$(printf '%s\n%s\n%s\n%s\n%s\n' "$transport_receipt" "$host_receipt" "$grant_receipt" "$witness_receipt" "$install_receipt" | stream_hash)" == "$(field raw_receipt_sha256)" ]] ||
  fail 'ordered raw receipt hash drifted'
[[ "$transport_receipt" == 'LOOM_HOST_EXEC_QUORUM_TRANSPORT PASS '* ]] || fail 'transport receipt malformed'
[[ "$host_receipt" == 'sounio-loom-host-exec-quorum-host-gate: HOST_MEASUREMENT_PASS '* ]] || fail 'host receipt malformed'
[[ "$grant_receipt" == 'LOOM_HOST_EXEC_QUORUM_HOST_GATE PASS '* ]] || fail 'grant receipt malformed'
[[ "$witness_receipt" == 'LOOM_HOST_PROCESS_WITNESS_GATE PASS '* ]] || fail 'ProcessWitness receipt malformed'
[[ "$install_receipt" == 'LOOM_HOST_EXEC_QUORUM_EXPERIMENT_INSTALL PASS '* ]] || fail 'install receipt malformed'
for token in \
  dynamic_user=true principal_distinct_uid=true treatment=closed positive=done \
  causal_bypass=done causal_sabotage=PASS wrong_generation=closed \
  payload_substitution=closed forged_ready=closed wrong_close=Sounio_refusal \
  controller_death=closed broker_death_after_ready=closed replay=closed \
  controller_terminal_extinct=true same_pid=true start_tick=true \
  pidfd_at_ready=live executable_transition=cell-to-Sounio \
  credential_unchanged=true cgroup_unchanged=true namespace_unchanged=true \
  environment_empty=true descendants_empty=true state_extinct=true \
  generation_extinct=true authority_extinct=true pidfd_extinct=true \
  cgroup_unpopulated=true unit_inactive=true extinction_omission=closed \
  complete_effects=false process_witness_core=true material_grant=true \
  material_execution=false launch_open=false parity_open=false claim_ready=false; do
  [[ " $witness_receipt " == *" $token "* ]] || fail "ProcessWitness receipt omitted $token"
done
for token in production_current_unchanged=true production_broker_unchanged=true rollback=identity-operation; do
  [[ " $install_receipt " == *" $token "* ]] || fail "install receipt omitted $token"
done

payload_result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_process_witness_handshake_payload_freeze_selftest.sh")"
[[ "$payload_result" == sounio-loom-process-witness-handshake-payload-freeze-selftest:\ PASS* ]] ||
  fail 'frozen Sounio payload gate failed'
cell_result="$(bash "$ROOT_DIR/$(field cell_selftest_path)")"
[[ "$cell_result" == sounio-loom-process-witness-principal-cell-selftest:\ PASS* ]] ||
  fail 'local ProcessWitness cell gate failed'
[[ "$cell_result" == *'principal_distinct_uid=false'* && "$cell_result" == *'material_execution=false'* ]] ||
  fail 'local cell was laundered as host execution'
local_result="$(bash "$ROOT_DIR/$(field local_quorum_selftest_path)")"
[[ "$local_result" == sounio-loom-host-exec-quorum-selftest:\ PASS* ]] ||
  fail 'local causal quorum gate failed'

printf 'sounio-loom-process-witness-host-freeze-selftest: PASS semantic_authority=Sounio action=9030 stage=MATERIAL_EXECUTION_CORE_FROZEN manifest_sha256=%s evidence_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve dynamic_user=true principal_distinct_uid=true treatment=closed positive=done causal_bypass=done causal_sabotage=PASS same_pid=true start_tick=true pidfd_at_ready=live executable_transition=cell-to-Sounio credential_unchanged=true cgroup_unchanged=true namespace_unchanged=true environment_empty=true descendants_empty=true affirmative_extinction=true state_extinct=true generation_extinct=true authority_extinct=true pidfd_extinct=true cgroup_unpopulated=true unit_inactive=true complete_effects=false production_activation=false production_current_unchanged=true production_broker_unchanged=true rollback=identity-operation material_grant=true material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
