#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
STACK="$ROOT_DIR/tools/loom/kernel_exec_grant_cell.stack.v1"
STACK_SHA256=1d7b8a3b1dfba1d1f9e60b5392cdf7e57a8d085cd872659feea5e333e43759b1

fail() {
  printf 'sounio-loom-kernel-exec-grant-cell-stack-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

value() {
  local key="$1" line name result=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$result" ]] || fail "duplicate stack field: $key"
      result="${line#*=}"
    fi
  done < "$STACK"
  [[ -n "$result" ]] || fail "stack omitted field: $key"
  printf '%s' "$result"
}

expect() {
  local key="$1" expected="$2" actual
  actual="$(value "$key")"
  [[ "$actual" == "$expected" ]] || fail "$key drifted: expected=$expected actual=$actual"
}

[[ -f "$STACK" && ! -L "$STACK" ]] || fail 'stack receipt is absent or linked'
[[ "$(sha256sum "$STACK" | cut -d ' ' -f 1)" == "$STACK_SHA256" ]] || fail 'stack receipt hash drifted'
expect schema loom-kernel-exec-grant-cell-stack-v1
expect stage MATERIAL_PREREQUISITE_MEASURED
expect authority_order GARDEN,SOUNIO_EXECUTABLE,SEMANTICS_FROZEN,MATERIAL_MEASUREMENT
expect semantic_producer Sounio
expect semantic_role SEMANTIC_AUTHORITY
expect semantic_action 9030
expect semantic_current_decision DENY491
expect resident_producer Sounio
expect resident_role SEMANTIC_TRANSPORT
expect resident_exact_output_parity 12/12
expect operational_producer OCaml
expect operational_role OPERATIONAL_KERNEL
expect operational_expected_results_encoded false
expect material_producer C++20+Linux+systemd
expect material_role MATERIAL_PARITY
expect material_transitory true
expect kernel_distinct_principal_candidate true
expect copied_pidfd_signal_distinct_uid EPERM
expect copied_pidfd_getfd_distinct_uid EPERM
expect shared_principal_signal_control ALLOWED
expect shared_principal_copied_pidfd_signal_control ALLOWED
expect causal_rule kernel-distinct-principal
expect causal_sabotage PASS
expect host_broker_socket_activation verified
expect host_broker_root_peer verified
expect host_broker_launch closed
expect host_broker_recycle closed
expect host_broker_action_9030 absent
expect remaining_barrier not-created
expect remaining_grant_extinction not-proven
expect remaining_product_attachment not-started
expect material_grant false
expect grant_extinction false
expect same_uid_peer_isolation false
expect exec_attached false
expect commit_attached false
expect ci_attached false
expect launch_open false
expect parity_open false
expect claim_ready false

ASSEMBLY_COMMIT="$(value assembly_commit)"
git -C "$ROOT_DIR" cat-file -e "$ASSEMBLY_COMMIT^{commit}" 2>/dev/null || fail 'assembly commit is absent'
for pair in \
  'semantic_manifest_path semantic_manifest_sha256' \
  'resident_manifest_path resident_manifest_sha256' \
  'operational_manifest_path operational_manifest_sha256' \
  'material_contract_path material_contract_sha256' \
  'sabotage_contract_path sabotage_contract_sha256' \
  'material_evidence_path material_evidence_sha256'; do
  read -r path_key digest_key <<< "$pair"
  path="$(value "$path_key")"
  [[ "$path" =~ ^[A-Za-z0-9._/-]+$ && -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" ]] ||
    fail "stack path is absent, linked, or unsafe: $path"
  [[ "$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)" == "$(value "$digest_key")" ]] ||
    fail "stack component hash drifted: $path"
done

host_broker_receipt="$(value host_broker_receipt)"
[[ "$host_broker_receipt" == 'sounio-loom-kernel-principal-broker-host-gate: HOST_ACTIVATION_PASS '* ]] ||
  fail 'host broker receipt is malformed'
[[ "$(printf '%s\n' "$host_broker_receipt" | sha256sum | cut -d ' ' -f 1)" == "$(value host_broker_receipt_sha256)" ]] ||
  fail 'host broker receipt digest differs'
for control in socket_activation=verified root_peer=verified admission=DENY424 \
  launch=closed recycle=closed material_broker=false material_invocation=false \
  same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false; do
  [[ " $host_broker_receipt " == *" $control "* ]] || fail "host broker receipt omitted $control"
done

bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_exec_grant_cell_authority_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_resident_transport_v4_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_exec_grant_cell_ocaml_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_host_principal_cell_freeze_selftest.sh" >/dev/null

printf 'sounio-loom-kernel-exec-grant-cell-stack-selftest: PASS stage=MATERIAL_PREREQUISITE_MEASURED semantic_authority=Sounio action=9030 stack_sha256=%s semantic_manifest_sha256=%s resident_manifest_sha256=%s operational_manifest_sha256=%s material_evidence_sha256=%s semantic_current=DENY491 resident_exact_output_parity=12/12 ocaml_expected_results=false kernel_distinct_principal_candidate=true copied_pidfd_signal=EPERM copied_pidfd_getfd=EPERM shared_principal_signal=ALLOWED shared_principal_copied_pidfd_signal=ALLOWED causal_rule=kernel-distinct-principal causal_sabotage=PASS host_broker_action_9030=absent host_broker_launch=closed host_broker_recycle=closed material_grant=false grant_extinction=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false launch_open=false parity_open=false claim_ready=false\n' \
  "$STACK_SHA256" "$(value semantic_manifest_sha256)" "$(value resident_manifest_sha256)" \
  "$(value operational_manifest_sha256)" "$(value material_evidence_sha256)"
