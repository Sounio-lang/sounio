#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
MANIFEST=tools/loom/kernel_peer_matrix_v12.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-matrix-v12-host-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-matrix-v12-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  fail 'peer-matrix freeze inputs are absent or linked'
! grep -Fq '__' "$MANIFEST" || fail 'manifest contains an unresolved marker'
! grep -Fq '__' "$EVIDENCE" || fail 'evidence contains an unresolved marker'

expect_field schema loom-kernel-peer-matrix-v12-freeze-v1
expect_field stage BPF_LSM_PEER_MATRIX_FROZEN_V12
expect_field semantic_authority Sounio
expect_field action 9025
expect_field semantic_manifest_sha256 daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30
expect_field load_manifest_sha256 6109ff02d1078ca3f0f21dcb98c189db12f59950899e375cac77c4ec7d4bfe75
expect_field material_source_commit fa8e3fab70ed77f17e24f48ff4f4007fc5bcc286
expect_field producing_language C+BPF+C++20
expect_field language_role MATERIAL_BOOTSTRAP
expect_field transitory true
expect_field contract_sha256 1aa04a29f6e443944709428fa1283055e4ed5741ba20344012d0bfe790faa5e2
expect_field init_source_sha256 54a447bd18a7d0319edda89fb01c593e5e28448c3994c4c0002c7b74795b4ab2
expect_field build_script_sha256 74be15ed9a00701c7d09c18682857aeb6de0cc8a8739b10257f28ef3a4ace71f
expect_field selftest_sha256 7571635f67efd38f7d800a9903cd69f20719f5af7c926c5dbf49b95044eb0a67
expect_field host_gate_sha256 ef7847dc5fdd2afc6413d9f570c79c16d226591821f6c6193410ffd66f84f3be
expect_field host_probe_sha256 57a215607a03166eb7524a74fbdb30f4320bc07d17264ee45482d47ad1f4c057
expect_field evidence_sha256 e11905937cb75cad46c3704ffb399ec1f2092e980e98dcbad333e72afc49f996
expect_field init_sha256 e3f164d58a05453151378170b7121365043a8c1f094ad5b22923f5970bf91234
expect_field loader_object_sha256 300b52202c2903d5fd81fc1219d8ea6fcfa8993108881c61ec08b04d9b372463
expect_field bpf_object_sha256 633849ca3dae7c8898c78e8e0049af4d689e04a27a468829fdc257ee427675d5
expect_field packer_sha256 09ff4699c9232d439e65e8e725c7bdcc949c59438602569650a3c9048b8f6629
expect_field base_initramfs_sha256 3b7620d0ec4b3072599429e4a9e4c8b64b3ff4f729284e51b6bd5abb59bd21e1
expect_field loader_source_sha256 7f3db89ef62d8ebc24b241a2121e83d81a5e18bd51dd4012d84a9b7a03212abc
expect_field loader_sha256 fbcc27e96e1cc10091ad5c60d81d5ddf70a39097031f4b6589422738935bb92b
expect_field final_initramfs_sha256 81bdc6bc6515df7f6446200eb1c5b66f4222c015564267c24848c9a94fd1b1e2
expect_field kernel_sha256 842932f8f994b201309efc386a5f1049377388aa8689b17e987e5c681e58e1ef
expect_field host_output_sha256 997d69ec447c8b9e73064359a694db91ff591fba7a646eb6ad7bdfaeaa0679f5
expect_field pair_set_sha256 fcca34baf88e23622ed0a511438709f28080885ff70cae2976cabd7f4ef8a257
expect_field pair_lines_sha256 588a6eedc90ac6fc42d055bc0dc8521723c57b685fea048e10f918ded6c8fda4
expect_field hardware_host t560-proxmox
expect_field kernel 7.0.2-5-pve
expect_field hypervisor KVM
expect_field qemu_version 11.0.0
expect_field active_lsm lockdown,capability,bpf,ima,evm
expect_field competing_ptrace_lsms absent
expect_field principal_uid 61234
expect_field principal_gid 61234
expect_field principal_capability CAP_SYS_NICE_ONLY
expect_field attacker_seccomp 0
expect_field operations 10
expect_field decisive_pairs 10
expect_field treatment_refused 10
expect_field mediator_removed_completed 10
expect_field mediator_quiescence_ms 250
expect_field only_delta mediator_presence+policy_hash
expect_field treatment_delta_sha256 15a3ce5e153f541ce66846434785b8080bd7fd5a7a5eb228d5059d8492e7c68f
expect_field sabotage_delta_sha256 06151e29a06c3569f958cacea5265da1efcbaa0710dc8bd99491b536a45bb0e3

for enabled in bpf_lsm_active btf_core guest_distinct guest_pid_1 qemu_extinct decisive_peer_matrix native_material_matrix_bytes_created same_kuid_pair_observed all_four_kernel_uid_slots_equal same_four_uids attacker_syscalls_open distinct_cgroups same_process_epoch receiver_mediator_active mediator_links_extinct mediator_programs_extinct all_epoch_objects_extinct; do
  expect_field "$enabled" true
done
expect_field guest_disk none
expect_field guest_network none
expect_field controls_executed false
expect_field material_peer_matrix false
expect_field same_uid_peer_isolation false
expect_field action_9025_decision DENY451
for boundary in material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed; do
  expect_field "$boundary" false
done
expect_field next_stage BPF_LSM_PEER_CONTROLS

SOURCE_COMMIT="$(field material_source_commit)"
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'peer-matrix source commit is absent'
for pair in contract_path:contract_sha256 init_source_path:init_source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256 host_gate_path:host_gate_sha256 host_probe_path:host_probe_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  path="$(field "$path_key")"; expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the peer-matrix source commit"
done
[[ "$(file_hash "$(field semantic_manifest_path)")" == "$(field semantic_manifest_sha256)" ]] ||
  fail 'semantic manifest drifted'
[[ "$(file_hash "$(field load_manifest_path)")" == "$(field load_manifest_sha256)" ]] ||
  fail 'BPF-load manifest drifted'
[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'peer-matrix evidence drifted'

expect_evidence schema loom-kernel-peer-matrix-v12-evidence-v1
expect_evidence stage BPF_LSM_PEER_MATRIX_V12
expect_evidence material_source_commit "$SOURCE_COMMIT"
for key in semantic_manifest_sha256 load_manifest_sha256 contract_sha256 init_source_sha256 build_script_sha256 selftest_sha256 host_gate_sha256 host_probe_sha256 init_sha256 loader_object_sha256 bpf_object_sha256 packer_sha256 base_initramfs_sha256 loader_source_sha256 loader_sha256 final_initramfs_sha256 kernel_sha256 host_output_sha256 pair_set_sha256 pair_lines_sha256 hardware_host kernel hypervisor qemu_version active_lsm competing_ptrace_lsms guest_disk guest_network operations decisive_pairs treatment_refused mediator_removed_completed same_kuid_pair_observed all_four_kernel_uid_slots_equal same_four_uids principal_uid principal_gid principal_capability attacker_seccomp attacker_syscalls_open distinct_cgroups same_process_epoch only_delta receiver_mediator_active mediator_links_extinct mediator_programs_extinct mediator_quiescence_ms all_epoch_objects_extinct treatment_delta_sha256 sabotage_delta_sha256 decisive_peer_matrix native_material_matrix_bytes_created controls_executed material_peer_matrix same_uid_peer_isolation action_9025_decision material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed next_stage; do
  expect_evidence "$key" "$(field "$key")"
done

HOST_BOOT="$(evidence_field host_boot_id)"
GUEST_BOOT="$(evidence_field guest_boot_id)"
[[ "$HOST_BOOT" != "$GUEST_BOOT" ]] || fail 'guest and host boot identities alias'
[[ "$(grep -c '^pair_[0-9]\+=' "$EVIDENCE")" == 10 ]] || fail 'evidence does not contain exactly ten causal pairs'

syscalls=(kill_SIGTERM tgkill_SIGTERM rt_sigqueueinfo pidfd_send_signal ptrace_ATTACH process_vm_readv open_read_proc_pid_mem pidfd_getfd prlimit64 process_madvise)
completions=(TARGET_TERMINATED TARGET_THREAD_TERMINATED SIGNAL_PAYLOAD_OBSERVED TARGET_TERMINATED PTRACE_ATTACH_DETACH CANARY_BYTES_READ PROC_MEM_CANARY_READ TARGET_FD_DUPLICATED LIMIT_CHANGED_RESTORED MADVISE_COMPLETED_4096_BYTES)
for index in $(seq 1 10); do
  receipt="$(evidence_field "pair_$index")"
  for fact in "operation=$index" "syscall=${syscalls[$((index - 1))]}" treatment=REFUSED_BEFORE_EFFECT sabotage=EFFECT_COMPLETED "completion=${completions[$((index - 1))]}" treatment_delta_sha256=15a3ce5e153f541ce66846434785b8080bd7fd5a7a5eb228d5059d8492e7c68f sabotage_delta_sha256=06151e29a06c3569f958cacea5265da1efcbaa0710dc8bd99491b536a45bb0e3 same_four_uids=true attacker_seccomp=0 distinct_cgroups=true same_process_epoch=true only_delta=mediator_presence+policy_hash mediator_links_extinct=true mediator_programs_extinct=true mediator_quiescence_ms=250 competing_ptrace_lsms=absent guest_root_traversable=true principal_capability=CAP_SYS_NICE_ONLY; do
    [[ " $receipt " == *" $fact "* ]] || fail "pair_$index omitted $fact"
  done
  [[ " $receipt " == *' treatment_errno=EACCES '* || " $receipt " == *' treatment_errno=EPERM '* ]] ||
    fail "pair_$index has no admissible refusal errno"
  [[ "$(grep -oE '[0-9a-f]{64}' <<<"$receipt" | wc -l)" == 8 ]] ||
    fail "pair_$index does not carry eight independent hashes"
done

pair_lines_hash="$(grep '^pair_[0-9]\+=' "$EVIDENCE" | sed 's/^pair_[0-9]\+=//' | stream_hash)"
[[ "$pair_lines_hash" == "$(field pair_lines_sha256)" ]] || fail 'raw causal pair set drifted'
[[ "$(printf '%s' "$(evidence_field local_selftest_command)" | stream_hash)" == "$(evidence_field local_selftest_command_sha256)" ]] ||
  fail 'local selftest command hash drifted'
[[ "$(printf '%s' "$(evidence_field local_selftest_result)" | stream_hash)" == "$(evidence_field local_selftest_result_sha256)" ]] ||
  fail 'local selftest result hash drifted'
[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'host command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'host result hash drifted'
for fact in host_output_sha256=997d69ec447c8b9e73064359a694db91ff591fba7a646eb6ad7bdfaeaa0679f5 operations=10 decisive_pairs=10 treatment_refused=10 mediator_removed_completed=10 same_kuid_pair_observed=true all_four_kernel_uid_slots_equal=true attacker_syscalls_open=true receiver_mediator_active=true only_delta_mediator=true competing_ptrace_lsms=absent principal_capability=CAP_SYS_NICE_ONLY all_epoch_objects_extinct=true controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
  [[ " $(evidence_field result) " == *" $fact "* ]] || fail "transport result omitted $fact"
done

local_result="$(bash "$(field selftest_path)" 2>/dev/null)"
[[ "$local_result" == "$(evidence_field local_selftest_result)" ]] ||
  fail 'source-fresh peer-matrix build drifted'

printf 'sounio-loom-kernel-peer-matrix-v12-freeze-selftest: PASS semantic_authority=Sounio action=9025 material_producer=C+BPF+C++20 material_role=MATERIAL_BOOTSTRAP manifest_sha256=%s evidence_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve hypervisor=KVM operations=10 decisive_pairs=10 treatment_refused=10 mediator_removed_completed=10 same_four_uids=true principal_capability=CAP_SYS_NICE_ONLY attacker_seccomp=0 same_process_epoch=true only_delta=mediator_presence+policy_hash mediator_links_extinct=true mediator_programs_extinct=true all_epoch_objects_extinct=true decisive_peer_matrix=true native_material_matrix_bytes_created=true controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=BPF_LSM_PEER_CONTROLS\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
