#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-host-principal-cell-v1-20260828.txt"

fail() {
  printf 'sounio-loom-host-principal-cell-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

value() {
  local key="$1" line name result=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$result" ]] || fail "duplicate evidence field: $key"
      result="${line#*=}"
    fi
  done < "$EVIDENCE"
  [[ -n "$result" ]] || fail "evidence omitted field: $key"
  printf '%s' "$result"
}

expect() {
  local key="$1" expected="$2" actual
  actual="$(value "$key")"
  [[ "$actual" == "$expected" ]] || fail "$key drifted: expected=$expected actual=$actual"
}

sha_at_commit() {
  local commit="$1" path="$2"
  git -C "$ROOT_DIR" show "$commit:$path" | sha256sum | cut -d ' ' -f 1
}

[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'material evidence is absent or linked'
expect schema loom-host-exec-grant-principal-cell-material-evidence-v1
expect stage MATERIAL_MEASURED
expect semantic_authority Sounio
expect semantic_action 9030
expect semantic_manifest_sha256 8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051
expect semantic_source_sha256 f32cb56118c842753ce3da1acdc08c321e37243ffecbe93e960bb24b552ae90f
expect semantic_semantics_sha256 7df064d4f074e2b278a5a22ac90cfd833436562aea154c25a2407e25faf5a61c
expect semantic_executable_sha256 cb123c2db31037ee2388a12fe4ed2055d430600b948a8dc317640155f9611e78
expect material_producer C++20
expect material_role MATERIAL_PARITY
expect material_transitory true
expect transport_role MECHANICAL_TRANSPORT
expect transport kubectl+hostPID+nsenter
expect local_rebuilds 2
expect local_builds_identical true
expect semantic_results_encoded_in_cpp false
expect python_runtime_dependency false
expect rust_runtime_dependency false
expect host t560-proxmox
expect kernel 7.0.2-5-pve
expect architecture x86_64
expect systemd_version 257
expect logical_cpus 64
expect cpu_model 'INTEL(R) XEON(R) GOLD 6526Y'
expect simultaneous_uid_distinct true
expect simultaneous_gid_distinct true
expect cgroup_distinct true
expect pidfd_live true
expect start_tick_stable true
expect signal_cross_uid EPERM
expect proc_mem_cross_uid EACCES
expect ptrace_cross_uid EPERM
expect process_vm_readv_cross_uid EPERM
expect proc_fd_cross_uid EACCES
expect copied_pidfd_signal EPERM
expect copied_pidfd_getfd EPERM
expect reciprocal_attacks refused
expect dynamic_user true
expect no_new_privileges true
expect protect_system strict
expect protect_proc invisible
expect private_network true
expect capabilities zero
expect sabotage_cgroup_distinct true
expect sabotage_signal_cross_cell ALLOWED
expect sabotage_copied_pidfd_signal ALLOWED
expect sabotage_reciprocal ALLOWED
expect causal_rule kernel-distinct-principal
expect causal_sabotage PASS
expect process_cleanup observed
expect result HOST_MEASUREMENT_PASS
expect kernel_distinct_principal_candidate true
expect same_uid_peer_isolation false
expect material_grant false
expect grant_extinction false
expect exec_attached false
expect commit_attached false
expect ci_attached false
expect launch_open false
expect parity_open false
expect claim_ready false

CONTRACT_PATH="$(value preregistered_contract_path)"
CONTRACT_COMMIT="$(value preregistered_contract_commit)"
SABOTAGE_CONTRACT_PATH="$(value sabotage_contract_path)"
SABOTAGE_CONTRACT_COMMIT="$(value sabotage_contract_commit)"
SOURCE_PATH="$(value material_source_path)"
SOURCE_COMMIT="$(value material_source_commit)"
[[ "$CONTRACT_PATH" == tools/loom/HOST_EXEC_GRANT_PRINCIPAL_CELL_V1.md ]] || fail 'contract path drifted'
[[ "$SABOTAGE_CONTRACT_PATH" == tools/loom/HOST_EXEC_GRANT_PRINCIPAL_CELL_SABOTAGE_V1.md ]] || fail 'sabotage contract path drifted'
[[ "$SOURCE_PATH" == tools/loom/src/loom_host_principal_cell.cpp ]] || fail 'material source path drifted'
git -C "$ROOT_DIR" cat-file -e "$CONTRACT_COMMIT^{commit}" 2>/dev/null || fail 'contract commit is absent'
git -C "$ROOT_DIR" cat-file -e "$SABOTAGE_CONTRACT_COMMIT^{commit}" 2>/dev/null || fail 'sabotage contract commit is absent'
git -C "$ROOT_DIR" cat-file -e "$SOURCE_COMMIT^{commit}" 2>/dev/null || fail 'material source commit is absent'
[[ "$(sha_at_commit "$CONTRACT_COMMIT" "$CONTRACT_PATH")" == "$(value preregistered_contract_sha256)" ]] ||
  fail 'preregistered contract bytes differ from evidence'
[[ "$(sha_at_commit "$SABOTAGE_CONTRACT_COMMIT" "$SABOTAGE_CONTRACT_PATH")" == "$(value sabotage_contract_sha256)" ]] ||
  fail 'preregistered sabotage contract bytes differ from evidence'
[[ "$(sha_at_commit "$SOURCE_COMMIT" "$SOURCE_PATH")" == "$(value material_source_sha256)" ]] ||
  fail 'material source commit bytes differ from evidence'

for pair in \
  'tools/loom/src/loom_host_principal_cell.cpp material_source_sha256' \
  'scripts/dev/build_loom_host_principal_cell.sh build_script_sha256' \
  'scripts/ci/sounio_loom_host_principal_cell_selftest.sh local_gate_sha256' \
  'scripts/ci/sounio_loom_host_principal_cell_host_gate.sh host_gate_sha256' \
  'scripts/dev/run_loom_host_principal_cell_probe.sh transport_sha256'; do
  read -r path key <<< "$pair"
  [[ "$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)" == "$(value "$key")" ]] ||
    fail "current file hash drifted: $path"
done

bash "$ROOT_DIR/scripts/ci/sounio_loom_host_principal_cell_selftest.sh" >/dev/null
transport_receipt="$(value transport_receipt)"
host_receipt="$(value host_receipt)"
[[ "$transport_receipt" == 'LOOM_HOST_PRINCIPAL_CELL_TRANSPORT PASS '* ]] || fail 'transport receipt is malformed'
[[ "$host_receipt" == 'sounio-loom-host-principal-cell-host-gate: HOST_MEASUREMENT_PASS '* ]] || fail 'host receipt is malformed'
for receipt in "$transport_receipt" "$host_receipt"; do
  [[ " $receipt " == *' kernel_distinct_principal_candidate=true '* ]] || fail 'receipt lost candidate boundary result'
  [[ " $receipt " == *' material_grant=false '* ]] || fail 'receipt promoted a material grant'
  [[ " $receipt " == *' grant_extinction=false '* ]] || fail 'receipt promoted grant extinction'
  [[ " $receipt " == *' same_uid_peer_isolation=false '* ]] || fail 'receipt promoted same-UID isolation'
  [[ " $receipt " == *' exec_attached=false '* ]] || fail 'receipt attached execution'
  [[ " $receipt " == *' launch_open=false'* ]] || fail 'receipt opened launch'
done
for control in signal_cross_uid=EPERM ptrace_cross_uid=EPERM \
  process_vm_readv_cross_uid=EPERM copied_pidfd_signal=EPERM \
  copied_pidfd_getfd=EPERM reciprocal_attacks=refused \
  sabotage_signal_cross_cell=ALLOWED sabotage_copied_pidfd_signal=ALLOWED \
  sabotage_reciprocal=ALLOWED causal_rule=kernel-distinct-principal \
  causal_sabotage=PASS; do
  [[ " $host_receipt " == *" $control "* ]] || fail "host receipt omitted hostile control: $control"
done

printf 'sounio-loom-host-principal-cell-freeze-selftest: PASS semantic_authority=Sounio action=9030 evidence_sha256=%s material_source_sha256=%s material_binary_sha256=%s host_gate_sha256=%s transport_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve dynamic_user=true uid_distinct=true gid_distinct=true cgroup_distinct=true pidfd_live=true start_tick_stable=true cross_signal=EPERM ptrace=EPERM process_vm_readv=EPERM copied_pidfd_signal=EPERM copied_pidfd_getfd=EPERM reciprocal_attacks=refused same_principal_signal=ALLOWED same_principal_copied_pidfd_signal=ALLOWED causal_rule=kernel-distinct-principal causal_sabotage=PASS kernel_distinct_principal_candidate=true same_uid_peer_isolation=false material_grant=false grant_extinction=false exec_attached=false commit_attached=false ci_attached=false launch_open=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$EVIDENCE" | cut -d ' ' -f 1)" "$(value material_source_sha256)" \
  "$(value material_binary_sha256)" "$(value host_gate_sha256)" "$(value transport_sha256)"
