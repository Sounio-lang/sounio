#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/host_exec_quorum_host.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-host-exec-quorum-host-v1-20260828.txt"

fail() {
  printf 'sounio-loom-host-exec-quorum-host-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in ${path#$ROOT_DIR/}"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

field() {
  record_field "$MANIFEST" "$1"
}

evidence_field() {
  record_field "$EVIDENCE" "$1"
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

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'host material-grant manifest is missing or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'host material-grant evidence is missing or linked'

expect_field schema loom-host-exec-quorum-host-runtime-v1
expect_field stage MATERIAL_GRANT_FROZEN
expect_field producing_language C++20+Linux+systemd
expect_field language_role MATERIAL_PARITY
expect_field transitory true
expect_field semantic_authority Sounio
expect_field controller_language OCaml
expect_field controller_role EFFECT_PARITY
expect_field action 9030
expect_field principal_distinct_uid true
expect_field non_bearer_exec_quorum true
expect_field descriptor_barrier_causal true
expect_field linear_grant_consumption true
expect_field causal_sabotage PASS
expect_field material_grant true
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
git -C "$ROOT_DIR" cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'host source commit is absent'
git -C "$ROOT_DIR" cat-file -e "${FREEZE_COMMIT}^{commit}" || fail 'host freeze-gate commit is absent'

for pair in \
  derived_garden_path:derived_garden_sha256 \
  parent_local_manifest_path:parent_local_manifest_sha256 \
  semantic_manifest_path:semantic_manifest_sha256 \
  broker_source_path:broker_source_sha256 \
  quorum_module_path:quorum_module_sha256 \
  principal_cell_source_path:principal_cell_source_sha256 \
  principal_cell_build_script_path:principal_cell_build_script_sha256 \
  principal_cell_selftest_path:principal_cell_selftest_sha256 \
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
    fail "$path differs from the host source commit"
done

FREEZE_PATH="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$FREEZE_PATH")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze gate drifted'
[[ "$(git -C "$ROOT_DIR" show "$FREEZE_COMMIT:$FREEZE_PATH" | stream_hash)" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate differs from its commit'

SEMANTIC_MANIFEST="$ROOT_DIR/$(field semantic_manifest_path)"
PARENT_MANIFEST="$ROOT_DIR/$(field parent_local_manifest_path)"
expect_evidence schema loom-host-exec-quorum-material-grant-evidence-v1
expect_evidence stage MATERIAL_GRANT_MEASURED
expect_evidence semantic_authority Sounio
expect_evidence semantic_action 9030
expect_evidence semantic_source_path "$(record_field "$SEMANTIC_MANIFEST" source_path)"
expect_evidence semantic_source_sha256 "$(record_field "$SEMANTIC_MANIFEST" source_sha256)"
expect_evidence semantic_freeze_manifest_sha256 "$(field semantic_manifest_sha256)"
expect_evidence semantic_semantics_sha256 "$(record_field "$SEMANTIC_MANIFEST" semantics_sha256)"
expect_evidence toolchain_cxx_path "$(field cxx_path)"
expect_evidence toolchain_cxx_sha256 "$(field cxx_sha256)"
expect_evidence toolchain_cxx_version "$(field cxx_version)"
[[ "$(record_field "$SEMANTIC_MANIFEST" stage)" == SEMANTICS_FROZEN ]] || fail 'Sounio semantics are not frozen'
[[ "$(record_field "$SEMANTIC_MANIFEST" producing_language)" == Sounio ]] || fail 'Sounio did not produce the semantic root'
[[ "$(record_field "$SEMANTIC_MANIFEST" language_role)" == SEMANTIC_AUTHORITY ]] || fail 'semantic root role drifted'
[[ "$(record_field "$PARENT_MANIFEST" stage)" == MATERIAL_PARITY_FROZEN ]] || fail 'local causal quorum is not frozen'
[[ "$(record_field "$PARENT_MANIFEST" material_grant)" == false ]] || fail 'local pod manifest was laundered as host grant'

expect_evidence derived_garden_sha256 "$(field derived_garden_sha256)"
expect_evidence parent_local_quorum_manifest_sha256 "$(field parent_local_manifest_sha256)"
expect_evidence source_commit "$SOURCE_COMMIT"
expect_evidence capsule_sha256 "$(field capsule_sha256)"
expect_evidence release_id "$(field release_id)"
expect_evidence release_manifest_sha256 "$(field release_manifest_sha256)"
expect_evidence host "$(field hardware_host)"
expect_evidence kernel "$(field hardware_kernel)"
expect_evidence architecture "$(field hardware_architecture)"
expect_evidence systemd_version "$(field systemd_version)"
expect_evidence systemd_run_sha256 "$(field systemd_run_sha256)"
expect_evidence systemctl_sha256 "$(field systemctl_sha256)"
expect_evidence transport "$(field transport)"
expect_evidence command "$(field command)"
expect_evidence command_sha256 "$(field command_sha256)"
expect_evidence transport_output_sha256 "$(field transport_output_sha256)"
expect_evidence result HOST_MEASUREMENT_PASS
expect_evidence first_attempt FAIL_CLOSED
expect_evidence first_attempt_material_grant false
expect_evidence treatment closed
expect_evidence positive_host open
expect_evidence principal_distinct_uid true
expect_evidence non_bearer_exec_quorum true
expect_evidence descriptor_barrier_causal true
expect_evidence linear_grant_consumption true
expect_evidence positive_open_sentinels 1
expect_evidence exact_write_sabotage open
expect_evidence sabotage_open_sentinels 1
expect_evidence total_open_sentinels 2
expect_evidence second_release refused
expect_evidence replay closed
expect_evidence controller_death closed
expect_evidence resident_death closed
expect_evidence wrong_generation closed
expect_evidence python closed
expect_evidence textual_receipt closed
expect_evidence same_uid closed
expect_evidence causal_sabotage PASS
expect_evidence material_grant true
expect_evidence material_execution false
expect_evidence production_current_unchanged true
expect_evidence production_broker_unchanged true
expect_evidence rollback identity-operation
for boundary in launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_evidence "$boundary" false
done

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'host evidence hash drifted'
[[ "$(printf '%s\n' "$(field command)" | stream_hash)" == "$(field command_sha256)" ]] || fail 'host command hash drifted'
[[ "$(file_hash "$(field cxx_path)")" == "$(field cxx_sha256)" ]] || fail 'C++ compiler drifted'
[[ "$("$(field cxx_path)" --version | sed -n '1p')" == "$(field cxx_version)" ]] || fail 'C++ compiler version drifted'

transport_receipt="$(evidence_field transport_receipt)"
host_receipt="$(evidence_field host_gate_receipt)"
broker_receipt="$(evidence_field broker_receipt)"
install_receipt="$(evidence_field install_receipt)"
[[ "$transport_receipt" == 'LOOM_HOST_EXEC_QUORUM_TRANSPORT PASS '* ]] || fail 'transport receipt is malformed'
[[ "$host_receipt" == 'sounio-loom-host-exec-quorum-host-gate: HOST_MEASUREMENT_PASS '* ]] || fail 'host receipt is malformed'
[[ "$broker_receipt" == 'LOOM_HOST_EXEC_QUORUM_HOST_GATE PASS '* ]] || fail 'broker receipt is malformed'
[[ "$install_receipt" == 'LOOM_HOST_EXEC_QUORUM_EXPERIMENT_INSTALL PASS '* ]] || fail 'install receipt is malformed'
for token in \
  dynamic_user=true principal_distinct_uid=true non_bearer_exec_quorum=true \
  descriptor_barrier_causal=true linear_grant_consumption=true \
  positive_open_sentinels=1 sabotage_open_sentinels=1 total_open_sentinels=2 \
  material_grant=true material_execution=false launch_open=false parity_open=false claim_ready=false; do
  [[ " $host_receipt " == *" $token "* ]] || fail "host receipt omitted $token"
done
for token in \
  "release_manifest_sha256=$(field release_manifest_sha256)" \
  "broker_output_sha256=$(field broker_output_sha256)" \
  "systemd_run_sha256=$(field systemd_run_sha256)" \
  "systemctl_sha256=$(field systemctl_sha256)"; do
  [[ " $host_receipt " == *" $token "* ]] || fail "host receipt omitted $token"
done
for token in \
  treatment=closed positive_host=open exact_write_sabotage=open second_release=refused \
  replay=closed controller_death=closed resident_death=closed wrong_generation=closed \
  python=closed textual_receipt=closed same_uid=closed causal_sabotage=PASS; do
  [[ " $broker_receipt " == *" $token "* ]] || fail "broker receipt omitted $token"
done
for token in production_current_unchanged=true production_broker_unchanged=true rollback=identity-operation; do
  [[ " $install_receipt " == *" $token "* ]] || fail "install receipt omitted $token"
done
for token in \
  "archive_sha256=$(field capsule_sha256)" \
  "promoter_sha256=$(field promoter_sha256)" \
  "host_output_sha256=$(field host_output_sha256)"; do
  [[ " $transport_receipt " == *" $token "* ]] || fail "transport receipt omitted $token"
done

principal_result="$(bash "$ROOT_DIR/$(field principal_cell_selftest_path)")"
[[ "$principal_result" == sounio-loom-host-exec-quorum-principal-cell-selftest:\ PASS* ]] ||
  fail 'local principal-cell gate failed'
local_result="$(bash "$ROOT_DIR/$(field local_quorum_selftest_path)")"
[[ "$local_result" == sounio-loom-host-exec-quorum-selftest:\ PASS* ]] || fail 'local causal quorum gate failed'
[[ "$local_result" == *'principal_distinct_uid=false'* && "$local_result" == *'material_grant=false material_execution=false'* ]] ||
  fail 'local pod baseline was promoted during host freeze'

printf 'sounio-loom-host-exec-quorum-host-freeze-selftest: PASS semantic_authority=Sounio action=9030 stage=MATERIAL_GRANT_FROZEN manifest_sha256=%s evidence_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve dynamic_user=true principal_distinct_uid=true non_bearer_exec_quorum=true descriptor_barrier_causal=true linear_grant_consumption=true treatment=closed positive_host=open positive_open_sentinels=1 exact_write_sabotage=open sabotage_open_sentinels=1 total_open_sentinels=2 second_release=refused replay=closed controller_death=closed resident_death=closed wrong_generation=closed python=closed textual_receipt=closed same_uid=closed causal_sabotage=PASS production_activation=false production_current_unchanged=true production_broker_unchanged=true rollback=identity-operation material_grant=true material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
