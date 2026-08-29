#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/product_dynamic_user_lane_cell_host_canary.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-product-dynamic-user-lane-cell-host-canary-v1-20260829.txt"

fail() {
  printf 'sounio-loom-product-dynamic-user-lane-cell-host-canary-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  [[ "$actual" == "$expected" ]] ||
    fail "$key drifted: expected=$expected actual=$actual"
}

expect_evidence() {
  local key="$1" expected="$2" actual
  actual="$(evidence_field "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "evidence $key drifted: expected=$expected actual=$actual"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is missing or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'host evidence is missing or linked'

expect_field schema loom-product-dynamic-user-lane-cell-host-canary-runtime-v1
expect_field stage MATERIAL_CANARY_FROZEN
expect_field semantic_authority Sounio
expect_field semantic_action 9031
expect_field operational_language OCaml
expect_field operational_role OPERATIONAL_ATTACHMENT
expect_field material_language C++20+Linux+systemd
expect_field material_role MATERIAL_PARITY
expect_field material_transitory true
expect_field product_lane_cell_canary true
expect_field distinct_uid_product_broker_canary true
expect_field causal_sabotage PASS
expect_field command_executed false
expect_field material_execution false
expect_field production_activation false
for boundary in fleet_lane_cell_attached exec_cell_attached launch_open recycle_open \
  exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done

SOURCE_COMMIT="$(field source_commit)"
FREEZE_COMMIT="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${SOURCE_COMMIT}^{commit}" ||
  fail 'measured source commit is absent'
git -C "$ROOT_DIR" cat-file -e "${FREEZE_COMMIT}^{commit}" ||
  fail 'freeze-gate commit is absent'

for pair in \
  semantic_manifest_path:semantic_manifest_sha256 \
  product_ingress_manifest_path:product_ingress_manifest_sha256 \
  product_contract_path:product_contract_sha256 \
  ingress_source_path:ingress_source_sha256 \
  host_canary_source_path:host_canary_source_sha256 \
  broker_source_path:broker_source_sha256 \
  capsule_builder_path:capsule_builder_sha256 \
  host_gate_path:host_gate_sha256 \
  promoter_path:promoter_sha256 \
  transport_path:transport_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the measured source commit"
done

FREEZE_PATH="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$FREEZE_PATH")" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate drifted'
[[ "$(git -C "$ROOT_DIR" show "$FREEZE_COMMIT:$FREEZE_PATH" | stream_hash)" == \
   "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate differs from its commit'

SEMANTIC_MANIFEST="$ROOT_DIR/$(field semantic_manifest_path)"
PRODUCT_MANIFEST="$ROOT_DIR/$(field product_ingress_manifest_path)"
[[ "$(record_field "$SEMANTIC_MANIFEST" stage)" == SEMANTICS_FROZEN ]] ||
  fail 'Sounio semantics are not frozen'
[[ "$(record_field "$SEMANTIC_MANIFEST" producing_language)" == Sounio ]] ||
  fail 'Sounio did not produce the semantic root'
[[ "$(record_field "$SEMANTIC_MANIFEST" language_role)" == SEMANTIC_AUTHORITY ]] ||
  fail 'Sounio semantic role drifted'
[[ "$(record_field "$PRODUCT_MANIFEST" semantic_authority)" == Sounio && \
   "$(record_field "$PRODUCT_MANIFEST" operational_language)" == OCaml && \
   "$(record_field "$PRODUCT_MANIFEST" operational_role)" == OPERATIONAL_ATTACHMENT ]] ||
  fail 'product ingress language authority drifted'

expect_evidence schema loom-product-dynamic-user-lane-cell-host-canary-evidence-v1
expect_evidence stage MATERIAL_CANARY_MEASURED
expect_evidence semantic_authority Sounio
expect_evidence semantic_action 9031
expect_evidence sounio_source_path "$(record_field "$SEMANTIC_MANIFEST" source_path)"
expect_evidence sounio_source_sha256 "$(record_field "$SEMANTIC_MANIFEST" source_sha256)"
expect_evidence sounio_semantics_sha256 "$(record_field "$SEMANTIC_MANIFEST" semantics_sha256)"
expect_evidence sounio_producing_language Sounio
expect_evidence sounio_language_role SEMANTIC_AUTHORITY
expect_evidence sounio_toolchain_engine "$(record_field "$SEMANTIC_MANIFEST" toolchain_engine)"
expect_evidence sounio_toolchain_record_sha256 "$(record_field "$SEMANTIC_MANIFEST" toolchain_record_sha256)"
expect_evidence sounio_hardware_record_sha256 "$(record_field "$SEMANTIC_MANIFEST" hardware_record_sha256)"
expect_evidence sounio_command "$(record_field "$SEMANTIC_MANIFEST" command)"
expect_evidence sounio_command_sha256 "$(record_field "$SEMANTIC_MANIFEST" command_sha256)"
expect_evidence sounio_result "$(record_field "$SEMANTIC_MANIFEST" result)"
expect_evidence sounio_result_sha256 "$(record_field "$SEMANTIC_MANIFEST" result_sha256)"
expect_evidence semantic_manifest_sha256 "$(field semantic_manifest_sha256)"
expect_evidence source_commit "$SOURCE_COMMIT"
expect_evidence capsule_sha256 "$(field capsule_sha256)"
expect_evidence release_id "$(field release_id)"
expect_evidence release_manifest_sha256 "$(field release_manifest_sha256)"
expect_evidence host "$(field hardware_host)"
expect_evidence kernel "$(field hardware_kernel)"
expect_evidence architecture "$(field hardware_architecture)"
expect_evidence systemd_version "$(field systemd_version)"
expect_evidence transport "$(field transport)"
expect_evidence transport_command "$(field transport_command)"
expect_evidence transport_command_sha256 "$(field transport_command_sha256)"
expect_evidence result HOST_MEASUREMENT_PASS
expect_evidence treatment Sounio-DENY+hook-continues
expect_evidence sounio_allow_sabotage hook-refused
expect_evidence binding_sabotage refused
expect_evidence guardian_death refused
expect_evidence same_uid refused
expect_evidence missing_descriptor refused
expect_evidence python_oracle refused-before-execution
expect_evidence rust_oracle refused-before-execution
expect_evidence causal_sabotage PASS
expect_evidence command_executed false
expect_evidence product_lane_cell_canary true
expect_evidence distinct_uid_product_broker_canary true
expect_evidence material_execution false
expect_evidence production_activation false
for boundary in fleet_lane_cell_attached exec_cell_attached launch_open recycle_open \
  exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_evidence "$boundary" false
done

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'host evidence hash drifted'
[[ "$(printf '%s\n' "$(field transport_command)" | stream_hash)" == \
   "$(field transport_command_sha256)" ]] ||
  fail 'transport command hash drifted'

transport_receipt="$(evidence_field transport_receipt)"
host_receipt="$(evidence_field host_gate_receipt)"
broker_receipt="$(evidence_field broker_receipt)"
process_receipt="$(evidence_field process_witness_receipt)"
product_receipt="$(evidence_field product_receipt)"
install_receipt="$(evidence_field install_receipt)"
raw_receipt_bundle_sha256="$(evidence_field raw_receipt_bundle_sha256)"
[[ "$transport_receipt" == 'LOOM_HOST_EXEC_QUORUM_TRANSPORT PASS '* ]] ||
  fail 'transport receipt is malformed'
[[ "$host_receipt" == 'sounio-loom-host-exec-quorum-host-gate: HOST_MEASUREMENT_PASS '* ]] ||
  fail 'host receipt is malformed'
[[ "$broker_receipt" == 'LOOM_HOST_EXEC_QUORUM_HOST_GATE PASS '* ]] ||
  fail 'broker receipt is malformed'
[[ "$process_receipt" == 'LOOM_HOST_PROCESS_WITNESS_GATE PASS '* ]] ||
  fail 'ProcessWitness receipt is malformed'
[[ "$product_receipt" == 'LOOM_PRODUCT_EXEC_INGRESS_DYNAMIC_USER_HOST_GATE PASS '* ]] ||
  fail 'product receipt is malformed'
[[ "$install_receipt" == 'LOOM_HOST_EXEC_QUORUM_EXPERIMENT_INSTALL PASS '* ]] ||
  fail 'install receipt is malformed'
[[ "$(printf '%s\n' "$transport_receipt" "$host_receipt" "$broker_receipt" \
       "$process_receipt" "$product_receipt" "$install_receipt" | stream_hash)" == \
   "$raw_receipt_bundle_sha256" ]] ||
  fail 'raw host receipt bundle hash drifted'

for token in \
  semantic_authority=Sounio dynamic_user=true principal_distinct_uid=true \
  product_lane_cell_canary=true distinct_uid_product_broker_canary=true \
  material_execution=false production_activation=false launch_open=false \
  parity_open=false claim_ready=false; do
  [[ " $host_receipt " == *" $token "* ]] || fail "host receipt omitted $token"
done
for token in \
  treatment=Sounio-DENY+hook-continues sounio_allow_sabotage=hook-refused \
  binding_sabotage=refused guardian_death=refused same_uid=refused \
  missing_descriptor=refused python_oracle=refused-before-execution \
  rust_oracle=refused-before-execution causal_sabotage=PASS \
  command_executed=false material_execution=false production_activation=false; do
  [[ " $product_receipt " == *" $token "* ]] || fail "product receipt omitted $token"
done
for token in \
  "archive_sha256=$(field capsule_sha256)" \
  "host_output_sha256=$(field host_output_sha256)"; do
  [[ " $transport_receipt " == *" $token "* ]] || fail "transport receipt omitted $token"
done
for token in \
  "release_id=$(field release_id)" \
  "release_manifest_sha256=$(field release_manifest_sha256)" \
  production_current_unchanged=true production_broker_unchanged=true; do
  [[ " $install_receipt " == *" $token "* ]] || fail "install receipt omitted $token"
done

printf 'sounio-loom-product-dynamic-user-lane-cell-host-canary-freeze-selftest: PASS semantic_authority=Sounio action=9031 stage=MATERIAL_CANARY_FROZEN manifest_sha256=%s evidence_sha256=%s treatment=Sounio-DENY+hook-continues sounio_allow_sabotage=hook-refused binding_sabotage=refused guardian_death=refused same_uid=refused missing_descriptor=refused python_oracle=refused-before-execution rust_oracle=refused-before-execution causal_sabotage=PASS product_lane_cell_canary=true distinct_uid_product_broker_canary=true command_executed=false fleet_lane_cell_attached=false exec_cell_attached=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
