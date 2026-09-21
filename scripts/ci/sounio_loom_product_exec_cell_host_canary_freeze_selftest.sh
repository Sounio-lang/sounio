#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/product_exec_cell_host_canary.runtime.v1"

fail() {
  printf 'sounio-loom-product-exec-cell-host-canary-freeze-selftest: FAIL: %s\n' "$*" >&2
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

receipt_value() {
  local receipt="$1" key="$2" token found=''
  for token in $receipt; do
    if [[ "$token" == "$key="* ]]; then
      [[ -z "$found" ]] || fail "receipt duplicated $key"
      found="${token#*=}"
    fi
  done
  [[ -n "$found" ]] || fail "receipt omitted $key"
  printf '%s' "$found"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is missing or linked'
EVIDENCE="$ROOT_DIR/$(field evidence_path)"
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'host evidence is missing or linked'

expect schema loom-product-exec-cell-host-canary-runtime-v1
expect stage MATERIAL_EXEC_CELL_CANARY_FROZEN
expect semantic_authority Sounio
expect semantic_action 9030
expect lane_semantic_action 9031
expect fixture_producing_language Sounio
expect controller_language OCaml
expect controller_role EFFECT_PARITY
expect material_language C++20+Linux+systemd
expect material_role MATERIAL_PARITY
expect material_transitory true
expect result HOST_MEASUREMENT_PASS
expect product_exec_cell_canary true
expect simultaneous_distinct_dynamic_users true
expect same_pid_exec_transition true
expect outcome DONE
expect extinction_complete true
expect controller_extinct true
expect command_mismatch DENY492
expect causal_sabotage PASS
expect sabotage_exec_cell_created false
expect sabotage_payload_executed false
expect python_executed false
expect rust_executed false
expect exec_cell_attached true
expect material_grant true
expect material_execution true
expect test_only true
expect production_activation false
for boundary in launch_open recycle_open exec_attached commit_attached ci_attached \
  parity_open claim_ready; do
  expect "$boundary" false
done

SOURCE_COMMIT="$(field source_commit)"
FREEZE_COMMIT="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${SOURCE_COMMIT}^{commit}" ||
  fail 'measured source commit is absent'
git -C "$ROOT_DIR" cat-file -e "${FREEZE_COMMIT}^{commit}" ||
  fail 'freeze-gate commit is absent'

for pair in \
  fixture_manifest_path:fixture_manifest_sha256 \
  lane_manifest_path:lane_manifest_sha256 \
  process_witness_manifest_path:process_witness_manifest_sha256 \
  controller_manifest_path:controller_manifest_sha256 \
  broker_source_path:broker_source_sha256 \
  quorum_source_path:quorum_source_sha256 \
  process_witness_source_path:process_witness_source_sha256 \
  ingress_source_path:ingress_source_sha256 \
  composition_source_path:composition_source_sha256 \
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

FIXTURE_MANIFEST="$ROOT_DIR/$(field fixture_manifest_path)"
LANE_MANIFEST="$ROOT_DIR/$(field lane_manifest_path)"
PROCESS_MANIFEST="$ROOT_DIR/$(field process_witness_manifest_path)"
CONTROLLER_MANIFEST="$ROOT_DIR/$(field controller_manifest_path)"
[[ "$(record_field "$FIXTURE_MANIFEST" stage)" == SEMANTICS_FROZEN && \
   "$(record_field "$FIXTURE_MANIFEST" producing_language)" == Sounio && \
   "$(record_field "$FIXTURE_MANIFEST" semantic_authority)" == Sounio && \
   "$(record_field "$FIXTURE_MANIFEST" action)" == 9030 && \
   "$(record_field "$FIXTURE_MANIFEST" command_sha256)" == "$(field command_sha256)" && \
   "$(record_field "$FIXTURE_MANIFEST" intent_sha256)" == "$(field intent_sha256)" && \
   "$(record_field "$FIXTURE_MANIFEST" payload_sha256)" == "$(field payload_sha256)" && \
   "$(record_field "$FIXTURE_MANIFEST" command_mismatch_result)" == DENY492 ]] ||
  fail 'frozen Sounio ExecCell fixture authority drifted'
[[ "$(record_field "$LANE_MANIFEST" semantic_authority)" == Sounio && \
   "$(record_field "$LANE_MANIFEST" semantic_action)" == 9031 && \
   "$(record_field "$LANE_MANIFEST" stage)" == MATERIAL_CANARY_FROZEN ]] ||
  fail 'frozen product LaneCell authority drifted'
[[ "$(record_field "$PROCESS_MANIFEST" producing_language)" == Sounio && \
   "$(record_field "$PROCESS_MANIFEST" semantic_authority)" == Sounio && \
   "$(record_field "$PROCESS_MANIFEST" executable_sha256)" == "$(field payload_sha256)" ]] ||
  fail 'frozen Sounio ProcessWitness payload drifted'
[[ "$(record_field "$CONTROLLER_MANIFEST" producing_language)" == OCaml && \
   "$(record_field "$CONTROLLER_MANIFEST" language_role)" == EFFECT_PARITY && \
   "$(record_field "$CONTROLLER_MANIFEST" semantic_authority)" == Sounio && \
   "$(record_field "$CONTROLLER_MANIFEST" action)" == 9030 ]] ||
  fail 'frozen OCaml effect-parity controller drifted'

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'raw host evidence hash drifted'
[[ "$(printf '%s\n' "$(field transport_command)" | stream_hash)" == \
   "$(field transport_command_sha256)" ]] ||
  fail 'transport command hash drifted'

mapfile -t receipts < "$EVIDENCE"
[[ ${#receipts[@]} -eq 7 ]] || fail "raw host evidence has ${#receipts[@]} lines"
transport_receipt="${receipts[0]}"
host_receipt="${receipts[1]}"
broker_receipt="${receipts[2]}"
process_receipt="${receipts[3]}"
lane_receipt="${receipts[4]}"
exec_cell_receipt="${receipts[5]}"
install_receipt="${receipts[6]}"
[[ "$transport_receipt" == 'LOOM_HOST_EXEC_QUORUM_TRANSPORT PASS '* && \
   "$host_receipt" == 'sounio-loom-host-exec-quorum-host-gate: HOST_MEASUREMENT_PASS '* && \
   "$broker_receipt" == 'LOOM_HOST_EXEC_QUORUM_HOST_GATE PASS '* && \
   "$process_receipt" == 'LOOM_HOST_PROCESS_WITNESS_GATE PASS '* && \
   "$lane_receipt" == 'LOOM_PRODUCT_EXEC_INGRESS_DYNAMIC_USER_HOST_GATE PASS '* && \
   "$exec_cell_receipt" == 'loom-product-exec-cell-host: PASS '* && \
   "$install_receipt" == 'LOOM_HOST_EXEC_QUORUM_EXPERIMENT_INSTALL PASS '* ]] ||
  fail 'raw host receipt ordering or prefixes diverged'

for token in \
  semantic_authority=Sounio product_exec_cell_canary=true \
  exec_cell_attached=true material_grant=true material_execution=true \
  test_only=true production_activation=false launch_open=false \
  parity_open=false claim_ready=false; do
  [[ " $host_receipt " == *" $token "* ]] || fail "host receipt omitted $token"
done
for token in \
  semantic_authority=Sounio action=9030 lane_action=9031 \
  "intent_sha256=$(field intent_sha256)" \
  "command_sha256=$(field command_sha256)" \
  "product_command_sha256=$(field command_sha256)" \
  simultaneous_distinct_dynamic_users=true same_pid_exec_transition=true \
  outcome=DONE extinction_complete=true controller_extinct=true \
  command_mismatch=DENY492 causal_sabotage=PASS \
  sabotage_exec_cell_created=false sabotage_payload_executed=false \
  python_executed=false rust_executed=false exec_cell_attached=true \
  material_execution=true test_only=true production_activation=false; do
  [[ " $exec_cell_receipt " == *" $token "* ]] ||
    fail "product ExecCell receipt omitted $token"
done
LANE_UID="$(receipt_value "$exec_cell_receipt" lane_uid)"
LANE_GID="$(receipt_value "$exec_cell_receipt" lane_gid)"
EXEC_UID="$(receipt_value "$exec_cell_receipt" exec_uid)"
EXEC_GID="$(receipt_value "$exec_cell_receipt" exec_gid)"
[[ "$LANE_UID" =~ ^[0-9]+$ && "$LANE_GID" =~ ^[0-9]+$ && \
   "$EXEC_UID" =~ ^[0-9]+$ && "$EXEC_GID" =~ ^[0-9]+$ && \
   "$LANE_UID" != 0 && "$LANE_GID" != 0 && "$EXEC_UID" != 0 && \
   "$EXEC_GID" != 0 && "$LANE_UID" != "$EXEC_UID" && \
   "$LANE_GID" != "$EXEC_GID" ]] ||
  fail 'simultaneous DynamicUser identities were not materially distinct'
expect lane_uid "$LANE_UID"
expect lane_gid "$LANE_GID"
expect exec_uid "$EXEC_UID"
expect exec_gid "$EXEC_GID"

for token in \
  "archive_sha256=$(field capsule_sha256)" \
  "host_output_sha256=$(field host_output_sha256)"; do
  [[ " $transport_receipt " == *" $token "* ]] ||
    fail "transport receipt omitted $token"
done
for token in \
  "release_id=$(field release_id)" \
  "release_manifest_sha256=$(field release_manifest_sha256)" \
  production_current_unchanged=true production_broker_unchanged=true \
  exec_cell_attached=true material_execution=true test_only=true \
  production_activation=false; do
  [[ " $install_receipt " == *" $token "* ]] ||
    fail "install receipt omitted $token"
done

printf 'sounio-loom-product-exec-cell-host-canary-freeze-selftest: PASS semantic_authority=Sounio action=9030 lane_action=9031 stage=MATERIAL_EXEC_CELL_CANARY_FROZEN manifest_sha256=%s evidence_sha256=%s lane_uid=%s exec_uid=%s simultaneous_distinct_dynamic_users=true same_pid_exec_transition=true outcome=DONE extinction_complete=true controller_extinct=true command_mismatch=DENY492 causal_sabotage=PASS sabotage_exec_cell_created=false sabotage_payload_executed=false python_executed=false rust_executed=false product_exec_cell_canary=true exec_cell_attached=true material_grant=true material_execution=true test_only=true production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)" "$LANE_UID" "$EXEC_UID"
