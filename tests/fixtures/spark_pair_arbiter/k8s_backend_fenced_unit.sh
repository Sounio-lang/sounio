#!/usr/bin/env bash

set -euo pipefail
umask 077

[[ $# -eq 2 ]] || {
  printf 'usage: %s BACKEND WORK_DIR\n' "$0" >&2
  exit 64
}

BACKEND="$1"
WORK_DIR="$2"
mkdir -p "$WORK_DIR"
export SOUNIO_SPARK_PAIR_BACKEND_LIBRARY_MODE=1

# shellcheck source=/dev/null
source "$BACKEND"

POLICY="$WORK_DIR/policy"
FREEZE="$WORK_DIR/freeze"
LEASE_STATE="$WORK_DIR/lease.json"
PERSISTED="$WORK_DIR/persisted.json"
receipt="$WORK_DIR/decision.receipt"
: > "$POLICY"
printf 'frozen semantics\n' > "$FREEZE"
printf 'Sounio decision\n' > "$receipt"

source_sha="$(printf 'a%.0s' {1..64})"
freeze_sha="$(sha256_file "$FREEZE")"
decision_sha="$(sha256_file "$receipt")"
transaction="$(printf 'c%.0s' {1..64})"
lease_uid=22222222-2222-4222-8222-222222222222
receipt0="$(printf 'e%.0s' {1..64})"
receipt1="$(printf 'f%.0s' {1..64})"
barrier_sha="$(printf 'b%.0s' {1..64})"
barrier_binary_sha="$(printf '9%.0s' {1..64})"

policy_value() {
  case "$2" in
    authority_sha256) printf '%s\n' "$source_sha" ;;
    device_barrier_source_sha256) printf '%s\n' "$barrier_sha" ;;
    host_boot_0_annotation) printf 'unit/boot0\n' ;;
    host_boot_1_annotation) printf 'unit/boot1\n' ;;
    host_receipt_0_annotation) printf 'unit/receipt0\n' ;;
    host_receipt_1_annotation) printf 'unit/receipt1\n' ;;
    host_fence_epoch_annotation) printf 'unit/epoch\n' ;;
    host_fence_owner_annotation) printf 'unit/owner\n' ;;
    host_transaction_annotation) printf 'unit/transaction\n' ;;
    host_pair_digest_annotation) printf 'unit/pair\n' ;;
    host_decision_receipt_annotation) printf 'unit/decision\n' ;;
    host_lease_uid_annotation) printf 'unit/lease-uid\n' ;;
    host_lease_resource_version_annotation) printf 'unit/lease-rv\n' ;;
    host_prepare_0_annotation) printf 'unit/prepare0\n' ;;
    host_prepare_1_annotation) printf 'unit/prepare1\n' ;;
    host_intent_base_rv_annotation) printf 'unit/base-rv\n' ;;
    host_watchdog_fresh_seconds) printf '8\n' ;;
    host_runtime_restart_required) printf 'false\n' ;;
    minimum_free_memory_mb) printf '1\n' ;;
    *) printf 'unit-value\n' ;;
  esac
}

node0() { printf 'spark-3c59\n'; }
node1() { printf 'spark-8e54\n'; }
lease_json() { cat "$LEASE_STATE"; }
require_lease_context() { cat "$LEASE_STATE"; }
replace_lease() { cat > "$PERSISTED"; }
host_fence_pair_exact() { return 0; }
host_fence_report() {
  if [[ "$1" == spark-3c59 ]]; then printf '%s\n' "$UNIT_REPORT0"; else printf '%s\n' "$UNIT_REPORT1"; fi
}
admission_fail_closed() { return 1; }
slurm_free_memory_ready() { return 1; }

UNIT_REPORT0="PIREUS_HOST_FACTS node=spark-3c59 boot_id=11111111-1111-4111-8111-111111111111 grant_mode=FENCED grant_epoch=7 grant_owner=holder grant_valid=0 source_sha256=$source_sha freeze_sha256=$freeze_sha device_barrier=1 device_barrier_source_sha256=$barrier_sha device_barrier_binary_sha256=$barrier_binary_sha transaction_id=$transaction lease_uid=$lease_uid lease_resource_version=101 decision_receipt_sha256=$decision_sha pair_digest=none receipt_sha256=$receipt0"
UNIT_REPORT1="PIREUS_HOST_FACTS node=spark-8e54 boot_id=33333333-3333-4333-8333-333333333333 grant_mode=FENCED grant_epoch=7 grant_owner=holder grant_valid=0 source_sha256=$source_sha freeze_sha256=$freeze_sha device_barrier=1 device_barrier_source_sha256=$barrier_sha device_barrier_binary_sha256=$barrier_binary_sha transaction_id=$transaction lease_uid=$lease_uid lease_resource_version=101 decision_receipt_sha256=$decision_sha pair_digest=none receipt_sha256=$receipt1"

run_case() {
  local scenario="$1" mask
  if [[ "$scenario" == stale-intent ]]; then
    jq -n --arg uid "$lease_uid" '{metadata:{uid:$uid,resourceVersion:"101",annotations:{
      "unit/prepare0":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "unit/prepare1":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "unit/base-rv":"99"}},spec:{holderIdentity:"holder"}}' > "$LEASE_STATE"
  else
    jq -n --arg uid "$lease_uid" \
      '{metadata:{uid:$uid,resourceVersion:"101",annotations:{}},spec:{holderIdentity:"holder"}}' > "$LEASE_STATE"
  fi
  : > "$PERSISTED"
  bind_host_commit_receipts holder 7 UNINITIALIZED FENCED "$receipt" \
    "$transaction" none "$lease_uid" 101 "$UNIT_REPORT0" "$UNIT_REPORT1"
  jq -e '
    .metadata.annotations["unit/pair"] == "none" and
    (.metadata.annotations | has("unit/prepare0") | not) and
    (.metadata.annotations | has("unit/prepare1") | not) and
    (.metadata.annotations | has("unit/base-rv") | not)
  ' "$PERSISTED" >/dev/null
  mask="$(host_mask_from_facts "$(cat "$PERSISTED")" '' holder 7)"
  (( (mask & 4) == 4 )) || {
    printf 'backend-fenced-unit: %s did not satisfy the fenced epoch relation: mask=%s\n' \
      "$scenario" "$mask" >&2
    exit 1
  }
  (( (mask & 65536) == 65536 )) || {
    printf 'backend-fenced-unit: %s did not satisfy the device barrier relation: mask=%s\n' \
      "$scenario" "$mask" >&2
    exit 1
  }
}

run_case first-bootstrap
run_case stale-intent

printf 'K8S_BACKEND_FENCED_UNIT_PASS first_bootstrap=PASS stale_intent=CLEARED epoch_relation=PASS\n'
