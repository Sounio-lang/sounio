#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/host_durable_lane_supervisor.runtime.v1"

fail() {
  printf 'sounio-loom-host-durable-lane-supervisor-host-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  sha256sum "$1" | cut -d ' ' -f 1
}

stream_hash() {
  sha256sum | cut -d ' ' -f 1
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

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'runtime manifest is missing or linked'
EVIDENCE="$ROOT_DIR/$(field evidence_path)"
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'host evidence is missing or linked'

expect schema loom-host-durable-lane-supervisor-runtime-v1
expect stage MATERIAL_SAME_PHYSICAL_REATTACH_FROZEN
expect semantic_authority Sounio
expect semantic_action 9032
expect semantic_language_role SEMANTIC_AUTHORITY
expect operational_language OCaml
expect operational_role EFFECT_PARITY
expect material_platform Linux+systemd
expect material_role MATERIAL_OBSERVATION
expect shell_oracle_authority false
expect transport_pod_deleted true
expect transport_replaced true
expect same_physical_reattach true
expect kernel_recovered true
expect output_prefix_preserved true
expect semantic_journal_verified true
expect guardian_journal_verified true
expect sounio_decision SAME_PHYSICAL_REATTACH
expect sabotage_decision DENY526
expect causal_sabotage PASS
expect sabotage_process_created false
expect full_extinction true
expect host_systemd_custody true
expect tmux_used false
expect python_executed false
expect rust_executed false
expect production_activation false
expect parity_open false
expect claim_ready false

SOURCE_COMMIT="$(field source_commit)"
FREEZE_GATE_COMMIT="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${SOURCE_COMMIT}^{commit}" ||
  fail 'measured source commit is absent'
git -C "$ROOT_DIR" cat-file -e "${FREEZE_GATE_COMMIT}^{commit}" ||
  fail 'freeze-gate commit is absent'

for pair in \
  semantic_freeze_path:semantic_freeze_sha256 \
  sounio_source_path:sounio_source_sha256 \
  sounio_entrypoint_path:sounio_entrypoint_sha256 \
  action_builder_path:action_builder_sha256 \
  ocaml_source_path:ocaml_source_sha256 \
  capsule_builder_path:capsule_builder_sha256 \
  host_gate_path:host_gate_sha256 \
  transport_path:transport_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the measured source commit"
done

FREEZE_GATE_PATH="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$FREEZE_GATE_PATH")" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate drifted'
[[ "$(git -C "$ROOT_DIR" show "$FREEZE_GATE_COMMIT:$FREEZE_GATE_PATH" | stream_hash)" == \
   "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate differs from its commit'

SEMANTIC_FREEZE="$ROOT_DIR/$(field semantic_freeze_path)"
[[ "$(record_field "$SEMANTIC_FREEZE" stage)" == SEMANTICS_FROZEN && \
   "$(record_field "$SEMANTIC_FREEZE" producing_language)" == Sounio && \
   "$(record_field "$SEMANTIC_FREEZE" language_role)" == SEMANTIC_AUTHORITY && \
   "$(record_field "$SEMANTIC_FREEZE" action)" == 9032 && \
   "$(record_field "$SEMANTIC_FREEZE" executable_sha256)" == \
     "$(field action_runtime_sha256)" && \
   "$(record_field "$SEMANTIC_FREEZE" same_physical_decision)" == \
     'SOUNIO_HOST_DURABLE_LANE SAME_PHYSICAL_REATTACH semantic_authority=Sounio action=9032' && \
   "$(record_field "$SEMANTIC_FREEZE" guardian_start_mismatch_decision)" == \
     'SOUNIO_HOST_DURABLE_LANE DENY526 semantic_authority=Sounio action=9032' ]] ||
  fail 'frozen Sounio action-9032 authority drifted'

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'raw host evidence hash drifted'
mapfile -t receipts < "$EVIDENCE"
[[ ${#receipts[@]} -eq 5 ]] || fail "raw host evidence has ${#receipts[@]} lines"
phase_a_receipt="${receipts[0]}"
host_receipt="${receipts[1]}"
transport_receipt="${receipts[2]}"
extinction_receipt="${receipts[3]}"
hardware_receipt="${receipts[4]}"
[[ "$phase_a_receipt" == \
    'sounio-loom-host-durable-lane-supervisor-host-selftest: PHASE_A_PASS '* && \
   "$host_receipt" == \
    'sounio-loom-host-durable-lane-supervisor-host-selftest: HOST_MEASUREMENT_PASS '* && \
   "$transport_receipt" == 'LOOM_HOST_DURABLE_LANE_TRANSPORT PASS '* && \
   "$extinction_receipt" == 'LOOM_HOST_DURABLE_LANE_EXTINCTION PASS '* && \
   "$hardware_receipt" == 'LOOM_HOST_DURABLE_LANE_HARDWARE '* ]] ||
  fail 'raw host receipt ordering or prefixes diverged'

for token in \
  semantic_authority=Sounio action=9032 operational_language=OCaml \
  operational_role=EFFECT_PARITY material_platform=Linux+systemd \
  transport_replaced=true predecessor_transport_extinct=true \
  transport_pod_deleted=true state_root_equal=true guardian_pid_equal=true \
  guardian_start_equal=true guardian_instance_equal=true harness_pid_equal=true \
  harness_start_equal=true command_equal=true boot_id_equal=true \
  output_prefix_preserved=true semantic_journal_verified=true \
  guardian_journal_verified=true kernel_recovered=true \
  same_physical_reattach=true sounio_decision=SAME_PHYSICAL_REATTACH \
  sabotage_decision=DENY526 sabotage_process_created=false causal_sabotage=PASS \
  full_extinction=true host_systemd_custody=true tmux_used=false \
  python_executed=false rust_executed=false production_activation=false \
  parity_open=false claim_ready=false; do
  [[ " $host_receipt " == *" $token "* ]] || fail "host receipt omitted $token"
done
for token in \
  pod_a_deleted=true distinct_transport=true same_physical_reattach=true \
  kernel_recovered=true causal_sabotage=PASS full_extinction=true tmux_used=false \
  python_executed=false rust_executed=false production_activation=false \
  parity_open=false claim_ready=false; do
  [[ " $transport_receipt " == *" $token "* ]] ||
    fail "transport receipt omitted $token"
done
for token in root_absent=true start_unit_inactive=true recover_unit_inactive=true \
  guardian_pid_extinct=true harness_pid_extinct=true \
  recovered_kernel_pid_extinct=true; do
  [[ " $extinction_receipt " == *" $token "* ]] ||
    fail "extinction receipt omitted $token"
done

[[ "$(receipt_value "$phase_a_receipt" run_id)" == "$(field run_id)" ]] ||
  fail 'run identity differs between phase-A receipt and manifest'
for mapping in \
  transport_a_uid:pod_a_uid \
  transport_b_uid:pod_b_uid \
  guardian_pid:guardian_pid \
  guardian_start_tick:guardian_start_tick \
  harness_pid:harness_pid \
  harness_start_tick:harness_start_tick \
  kernel_pid_before:kernel_pid_before \
  kernel_pid_after:kernel_pid_after; do
  receipt_key="${mapping%%:*}"
  manifest_key="${mapping#*:}"
  [[ "$(receipt_value "$host_receipt" "$receipt_key")" == "$(field "$manifest_key")" ]] ||
    fail "$receipt_key differs between host receipt and manifest"
done
[[ "$(receipt_value "$transport_receipt" pod_a_uid)" == "$(field pod_a_uid)" && \
   "$(receipt_value "$transport_receipt" pod_b_uid)" == "$(field pod_b_uid)" && \
   "$(field pod_a_uid)" != "$(field pod_b_uid)" ]] ||
  fail 'transport Pod identities are not distinct and consistently bound'
[[ "$(field kernel_pid_before)" != "$(field kernel_pid_after)" ]] ||
  fail 'kernel recovery reused the predecessor PID'
[[ "$(receipt_value "$transport_receipt" archive_sha256)" == \
     "$(field capsule_sha256)" && \
   "$(receipt_value "$transport_receipt" action_9032_runtime_sha256)" == \
     "$(field action_runtime_sha256)" && \
   "$(receipt_value "$transport_receipt" host_gate_sha256)" == \
     "$(field host_gate_sha256)" ]] ||
  fail 'transport receipt artifact bindings drifted'
[[ "$(printf '%s\n' "$host_receipt" | stream_hash)" == \
     "$(field host_output_sha256)" ]] ||
  fail 'host output digest drifted'

for mapping in \
  host:hardware_host kernel:hardware_kernel architecture:hardware_architecture \
  logical_cpus:hardware_logical_cpus systemd_version:systemd_version \
  systemd_run_sha256:systemd_run_sha256 systemctl_sha256:systemctl_sha256 \
  boot_id:hardware_boot_id; do
  receipt_key="${mapping%%:*}"
  manifest_key="${mapping#*:}"
  [[ "$(receipt_value "$hardware_receipt" "$receipt_key")" == "$(field "$manifest_key")" ]] ||
    fail "$receipt_key differs between hardware receipt and manifest"
done

[[ "$(printf '%s\n' "$(field verification_command)" | stream_hash)" == \
   "$(field verification_command_sha256)" ]] ||
  fail 'verification command hash drifted'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-durable-freeze.XXXXXX")"
cleanup() {
  chmod -R u+rwX "$WORK" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT
SOURCE_ROOT="$WORK/source"
git clone --local --no-hardlinks --quiet "$ROOT_DIR" "$SOURCE_ROOT"
git -C "$SOURCE_ROOT" checkout --quiet --detach "$SOURCE_COMMIT"
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=normal)" ]] ||
  fail 'isolated measured source is dirty'
REBUILT_ACTION="$WORK/action-9032"
SOUNIO_LOOM_HOST_DURABLE_LANE_OUTPUT="$REBUILT_ACTION" \
  bash "$SOURCE_ROOT/$(field action_builder_path)" >/dev/null
[[ "$(file_hash "$REBUILT_ACTION")" == "$(field action_runtime_sha256)" ]] ||
  fail 'source-fresh Sounio action-9032 runtime drifted'
positive_frame='9032 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1'
[[ "$(printf '%s\n' "$positive_frame" | "$REBUILT_ACTION")" == \
   'SOUNIO_HOST_DURABLE_LANE SAME_PHYSICAL_REATTACH semantic_authority=Sounio action=9032' ]] ||
  fail 'rebuilt Sounio authority refused the frozen positive frame'
sabotage_frame='9032 3 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1'
set +e
sabotage_output="$(printf '%s\n' "$sabotage_frame" | "$REBUILT_ACTION" 2>&1)"
sabotage_status=$?
set -e
[[ $sabotage_status -eq 42 && "$sabotage_output" == \
   'SOUNIO_HOST_DURABLE_LANE DENY526 semantic_authority=Sounio action=9032' ]] ||
  fail 'rebuilt Sounio authority lost the causal sabotage'

REBUILT_CAPSULE="$WORK/capsule.tar"
bash "$SOURCE_ROOT/$(field capsule_builder_path)" --output "$REBUILT_CAPSULE" >/dev/null
[[ "$(file_hash "$REBUILT_CAPSULE")" == "$(field capsule_sha256)" ]] ||
  fail 'measured host capsule is not reproducible from its source commit'

printf 'sounio-loom-host-durable-lane-supervisor-host-freeze-selftest: PASS semantic_authority=Sounio action=9032 stage=MATERIAL_SAME_PHYSICAL_REATTACH_FROZEN manifest_sha256=%s evidence_sha256=%s capsule_sha256=%s pod_a_deleted=true transport_replaced=true same_guardian=true same_harness=true same_instance=true kernel_recovered=true sounio_decision=SAME_PHYSICAL_REATTACH sabotage_decision=DENY526 causal_sabotage=PASS full_extinction=true host_systemd_custody=true tmux_used=false python_executed=false rust_executed=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)" "$(field capsule_sha256)"
