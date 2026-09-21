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

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in the manifest"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in ${path#$ROOT_DIR/}"
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

expect() {
  local key="$1" expected="$2" actual
  actual="$(field "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "$key drifted: expected=$expected actual=$actual"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is missing or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'host evidence is missing or linked'

expect schema loom-product-dynamic-user-lane-cell-host-canary-runtime-v1
expect stage MATERIAL_CANARY_FROZEN
expect semantic_authority Sounio
expect semantic_action 9031
expect sounio_producing_language Sounio
expect sounio_language_role SEMANTIC_AUTHORITY
expect operational_language OCaml
expect operational_role OPERATIONAL_ATTACHMENT
expect material_language C++20+Linux+systemd
expect material_role MATERIAL_PARITY
expect material_transitory true
expect result HOST_MEASUREMENT_PASS
expect treatment Sounio-DENY+hook-continues
expect sounio_allow_sabotage hook-refused
expect binding_sabotage refused
expect guardian_death refused
expect same_uid refused
expect missing_descriptor refused
expect python_oracle refused-before-execution
expect rust_oracle refused-before-execution
expect causal_sabotage PASS
expect product_lane_cell_canary true
expect distinct_uid_product_broker_canary true
expect command_executed false
expect material_execution false
expect production_activation false
for boundary in fleet_lane_cell_attached exec_cell_attached launch_open recycle_open \
  exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect "$boundary" false
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
for pair in \
  source_path:sounio_source_path \
  source_sha256:sounio_source_sha256 \
  semantics_sha256:sounio_semantics_sha256 \
  producing_language:sounio_producing_language \
  language_role:sounio_language_role \
  toolchain_engine:sounio_toolchain_engine \
  toolchain_record_sha256:sounio_toolchain_record_sha256 \
  hardware_record_sha256:sounio_hardware_record_sha256 \
  command:sounio_command \
  command_sha256:sounio_command_sha256 \
  result:sounio_result \
  result_sha256:sounio_result_sha256; do
  semantic_key="${pair%%:*}"
  manifest_key="${pair#*:}"
  expect "$manifest_key" "$(record_field "$SEMANTIC_MANIFEST" "$semantic_key")"
done
[[ "$(record_field "$PRODUCT_MANIFEST" semantic_authority)" == Sounio && \
   "$(record_field "$PRODUCT_MANIFEST" operational_language)" == OCaml && \
   "$(record_field "$PRODUCT_MANIFEST" operational_role)" == OPERATIONAL_ATTACHMENT ]] ||
  fail 'product ingress language authority drifted'

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'raw host evidence hash drifted'
[[ "$(printf '%s\n' "$(field transport_command)" | stream_hash)" == \
   "$(field transport_command_sha256)" ]] ||
  fail 'transport command hash drifted'

mapfile -t receipts < "$EVIDENCE"
[[ ${#receipts[@]} -eq 6 ]] || fail "raw host evidence has ${#receipts[@]} lines"
transport_receipt="${receipts[0]}"
host_receipt="${receipts[1]}"
broker_receipt="${receipts[2]}"
process_receipt="${receipts[3]}"
product_receipt="${receipts[4]}"
install_receipt="${receipts[5]}"
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
