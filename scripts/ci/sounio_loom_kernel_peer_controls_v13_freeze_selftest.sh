#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
MANIFEST=tools/loom/kernel_peer_controls_v13.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-controls-v13-host-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-controls-v13-freeze-selftest: FAIL: %s\n' "$*" >&2
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
expect_field() {
  local actual
  actual="$(field "$1")"
  [[ "$actual" == "$2" ]] || fail "$1 drifted: expected=$2 actual=$actual"
}
expect_evidence() {
  local actual
  actual="$(evidence_field "$1")"
  [[ "$actual" == "$2" ]] || fail "evidence $1 drifted: expected=$2 actual=$actual"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" && -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] ||
  fail 'V13 material freeze inputs are absent or linked'
! grep -Fq '__' "$MANIFEST" || fail 'manifest contains an unresolved marker'
! grep -Fq '__' "$EVIDENCE" || fail 'evidence contains an unresolved marker'

expect_field schema loom-kernel-peer-controls-v13-freeze-v1
expect_field stage MATERIAL_CONTROL_MATRIX_FROZEN_V13
expect_field semantic_authority Sounio
expect_field action 9025
expect_field sounio_source_sha256 3545f75dca264b4378ab4cf633a686ffcde5152cb02ac18b74ab00192baed7f0
expect_field semantic_manifest_sha256 b3557d850ce0dc13c900f8dbb10c33f824ac25e908cb4a48dd2ef913267194c2
expect_field material_source_commit 84ecdf9b553935db867e936f807fe643cea1c26f
expect_field producing_language C+BPF+C++20
expect_field language_role MATERIAL_BOOTSTRAP
expect_field transitory true
expect_field host_stream_sha256 7f38f15eed2cc21d2effa6ebd5d86a8535b276ca3893f568b98ba70e79007844
expect_field host_output_sha256 7c67e0e774246be21a68cd9a8e21407f28cae0b904f428442cab7652b3e53e74
expect_field pair_set_sha256 c3dc1cffeb28e9f213809c643390b5ffa623d1426b6f722ca90dacd90a9d7e9c
expect_field control_set_sha256 9309a2cbb5571fc1cc63e252b58dab93ea06852760ea53faca5a7e449bdd7298
expect_field sabotage_set_sha256 9fcda176e2edaec93f5d305695ec2fe2860b2b74caf288dfc9d9a084720a8fee
expect_field hardware_host t560-proxmox
expect_field hardware_arch x86_64
expect_field kernel 7.0.2-5-pve
expect_field hypervisor KVM
expect_field qemu_version 11.0.0
expect_field active_lsm lockdown,capability,bpf,ima,evm
expect_field observations 50
expect_field decisive_pairs 10
expect_field controls 30
expect_field refused 25
expect_field completed 15
expect_field unavailable 10
expect_field crossed 0
expect_field treatment_refused 10
expect_field mediator_removed_completed 10
expect_field distinct_refused 10
expect_field caller_seccomp_unavailable 10
expect_field dumpable_completed 5
expect_field dumpable_refused 5
expect_field sabotage_twins 5
for enabled in guest_distinct_from_host guest_pid_1 qemu_extinct bpf_lsm_active btf_core same_kuid_pair_observed attacker_syscalls_open receiver_mediator_active all_epoch_objects_extinct distinct_processes distinct_pidfds distinct_start_ticks distinct_cgroups same_user_namespace v12_hypothesis_falsified controls_executed material_peer_matrix; do
  expect_field "$enabled" true
done
expect_field guest_disk none
expect_field guest_network none
expect_field competing_ptrace_lsms absent
expect_field principal_uid 61234
expect_field principal_gid 61234
expect_field principal_capability CAP_SYS_NICE_ONLY
expect_field same_uid_peer_isolation false
expect_field action_9025_decision DENY451
for boundary in material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed; do
  expect_field "$boundary" false
done
expect_field next_stage SOUNIO_JUDGMENT_V13

SOURCE_COMMIT="$(field material_source_commit)"
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'V13 material source commit is absent'
for pair in contract_path:contract_sha256 init_source_path:init_source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256 host_gate_path:host_gate_sha256 host_probe_path:host_probe_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  path="$(field "$path_key")"; expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the V13 material source commit"
done
[[ "$(file_hash "$(field semantic_manifest_path)")" == "$(field semantic_manifest_sha256)" ]] ||
  fail 'V13 semantic manifest drifted'
[[ "$(file_hash "$(field sounio_source_path)")" == "$(field sounio_source_sha256)" ]] ||
  fail 'V13 Sounio semantic source drifted'
[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'V13 material evidence drifted'

expect_evidence schema loom-kernel-peer-controls-v13-evidence-v1
expect_evidence stage MATERIAL_CONTROL_MATRIX_V13
expect_evidence material_source_commit "$SOURCE_COMMIT"
for key in sounio_source_sha256 semantic_manifest_sha256 producing_language language_role transitory contract_sha256 init_source_sha256 build_script_sha256 selftest_sha256 host_gate_sha256 host_probe_sha256 local_output_sha256 host_stream_sha256 host_output_sha256 pair_set_sha256 control_set_sha256 sabotage_set_sha256 base_initramfs_sha256 final_initramfs_sha256 bpf_object_sha256 loader_source_sha256 loader_sha256 packer_sha256 kernel_sha256 local_bpf_toolchain local_cxx_toolchain host_cxx_toolchain hardware_host hardware_arch kernel hypervisor qemu_version transport namespace node pod guest_distinct_from_host guest_pid_1 guest_disk guest_network qemu_extinct active_lsm bpf_lsm_active btf_core competing_ptrace_lsms observations decisive_pairs controls refused completed unavailable crossed treatment_refused mediator_removed_completed distinct_refused caller_seccomp_unavailable dumpable_completed dumpable_refused sabotage_twins same_kuid_pair_observed attacker_syscalls_open receiver_mediator_active all_epoch_objects_extinct principal_uid principal_gid principal_capability distinct_processes distinct_pidfds distinct_start_ticks distinct_cgroups same_user_namespace v12_hypothesis_falsified controls_executed material_peer_matrix same_uid_peer_isolation action_9025_decision material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed next_stage; do
  expect_evidence "$key" "$(field "$key")"
done

HOST_BOOT="$(evidence_field host_boot_id)"
GUEST_BOOT="$(evidence_field guest_boot_id)"
[[ "$HOST_BOOT" != "$GUEST_BOOT" ]] || fail 'guest and host boot identities alias'
[[ "$(grep -c '^operation_[0-9]\+=' "$EVIDENCE")" == 10 ]] || fail 'evidence lacks ten operation rows'
syscalls=(kill_SIGTERM tgkill_SIGTERM rt_sigqueueinfo pidfd_send_signal ptrace_ATTACH process_vm_readv open_read_proc_pid_mem pidfd_getfd prlimit64 process_madvise)
dumpable=(EFFECT_COMPLETED EFFECT_COMPLETED EFFECT_COMPLETED EFFECT_COMPLETED REFUSED_BEFORE_EFFECT REFUSED_BEFORE_EFFECT REFUSED_BEFORE_EFFECT REFUSED_BEFORE_EFFECT EFFECT_COMPLETED REFUSED_BEFORE_EFFECT)
for index in $(seq 1 10); do
  row="$(evidence_field "operation_$index")"
  for fact in "syscall=${syscalls[$((index - 1))]}" treatment=REFUSED_BEFORE_EFFECT mediator_removed=EFFECT_COMPLETED distinct_kuid=REFUSED_BEFORE_EFFECT caller_seccomp=EXPERIMENT_UNAVAILABLE "dumpable_only=${dumpable[$((index - 1))]}"; do
    [[ " $row " == *" $fact "* ]] || fail "operation_$index omitted $fact"
  done
done

[[ "$(grep -c '^sabotage_[1-5]=' "$EVIDENCE")" == 5 ]] || fail 'evidence lacks five sabotage twins'
sabotage_facts=(
  'source=TREATMENT target=MEDIATOR_REMOVED delta=REMOVE_MEDIATOR operations=10 crossed=10 epoch_mode=SAME_PROCESS expected=ALL_COMPLETED observed=ALL_COMPLETED'
  'source=MEDIATOR_REMOVED target=TREATMENT delta=INSTALL_MEDIATOR operations=10 crossed=10 epoch_mode=SAME_PROCESS expected=ALL_REFUSED observed=ALL_REFUSED'
  'source=DISTINCT_KUID_CONTROL target=MEDIATOR_REMOVED delta=COLLAPSE_TO_SAME_KUID operations=10 crossed=10 epoch_mode=FRESH_REQUIRED expected=CREDENTIAL_REFUSAL_DISAPPEARS observed=CREDENTIAL_REFUSAL_DISAPPEARS'
  'source=CALLER_SECCOMP_CONTROL target=MEDIATOR_REMOVED delta=OPEN_CALLER_FILTER operations=10 crossed=10 epoch_mode=FRESH_REQUIRED expected=UNAVAILABILITY_DISAPPEARS observed=UNAVAILABILITY_DISAPPEARS'
  'source=DUMPABLE_ONLY_CONTROL target=MEDIATOR_REMOVED delta=SET_DUMPABLE_ONE operations=5 crossed=5 unaffected=5 epoch_mode=FRESH_REQUIRED expected=FIVE_PARTIAL_REFUSALS_COMPLETE observed=FIVE_PARTIAL_REFUSALS_COMPLETE'
)
for index in $(seq 1 5); do
  row="$(evidence_field "sabotage_$index")"
  [[ " $row " == *" ${sabotage_facts[$((index - 1))]} "* ]] || fail "sabotage_$index causal rule drifted"
  [[ "$(grep -oE '[0-9a-f]{64}' <<<"$row" | wc -l)" == 1 ]] || fail "sabotage_$index lacks a unique receipt hash"
done

[[ "$(printf '%s' "$(evidence_field local_selftest_command)" | stream_hash)" == "$(evidence_field local_selftest_command_sha256)" ]] || fail 'local command hash drifted'
[[ "$(printf '%s' "$(evidence_field local_selftest_result)" | stream_hash)" == "$(evidence_field local_selftest_result_sha256)" ]] || fail 'local result hash drifted'
[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] || fail 'host command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] || fail 'host result hash drifted'
for fact in observations=50 decisive_pairs=10 controls=30 refused=25 completed=15 unavailable=10 crossed=0 sabotage_twins=5 controls_executed=true material_peer_matrix=true same_uid_peer_isolation=false action_9025_decision=DENY451 next_stage=SOUNIO_JUDGMENT_V13; do
  [[ " $(evidence_field result) " == *" $fact "* ]] || fail "transport result omitted $fact"
done

tampered="$(mktemp)"
trap 'rm -f "$tampered"' EXIT
sed 's/^crossed=0$/crossed=1/' "$EVIDENCE" >"$tampered"
[[ "$(file_hash "$tampered")" != "$(field evidence_sha256)" ]] || fail 'tamper control did not break the evidence hash'

local_result="$(bash "$(field selftest_path)" 2>/dev/null)"
[[ "$local_result" == "$(evidence_field local_selftest_result)" ]] || fail 'source-fresh V13 material build drifted'

printf 'sounio-loom-kernel-peer-controls-v13-freeze-selftest: PASS semantic_authority=Sounio action=9025 manifest_sha256=%s evidence_sha256=%s source_commit=%s observations=50 decisive_pairs=10 controls=30 refused=25 completed=15 unavailable=10 crossed=0 sabotage_twins=5 v12_hypothesis_falsified=true controls_executed=true material_peer_matrix=true same_uid_peer_isolation=false action_9025=DENY451 python_executed=false rust_executed=false next_stage=SOUNIO_JUDGMENT_V13\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)" "$SOURCE_COMMIT"
