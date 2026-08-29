#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
MANIFEST=tools/loom/kernel_peer_microhost_v12.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-microhost-v12-host-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-microhost-v12-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  fail 'microhost freeze inputs are absent or linked'
expect_field schema loom-kernel-peer-microhost-v12-freeze-v1
expect_field stage BPF_LSM_MICROHOST_FROZEN_V12
expect_field semantic_authority Sounio
expect_field action 9025
expect_field semantic_manifest_sha256 daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30
expect_field backend_manifest_sha256 bb695f07b9d752f025be0f101fde27fb42d635cc13ab3a9b34f2241ccab3b8c5
expect_field material_source_commit 51dbbbf0bc417763834868f25f0d2b5ce0755aca
expect_field producing_language C++20
expect_field language_role MATERIAL_BOOTSTRAP
expect_field transitory true
expect_field contract_sha256 b7aca6211a6d86e69ac71092d4c57617cb77ae940b18ee8c30066bf20c37089a
expect_field source_sha256 747f44c6ce7956362b7b0969598623c2901eec3ee9966c6a70d5e482d046b01f
expect_field packer_source_sha256 1dbd9e719a0582f66e11a460e4cd678d2fa5750bff255e3be364b023a5108826
expect_field build_script_sha256 bbcebd2125c27ab58e2f24aa185197b7cca3aaae9472f26ce59a5fe751f6009a
expect_field selftest_sha256 f121ccb0515c615a9c98e1c956981c722edcd5b8f05c9c8cc03f1a8ebeafdc7f
expect_field host_gate_sha256 36116f71467b51808bbcf691ec5c7c45e2e06e50b90d9ad537c6259ed3c7feb1
expect_field host_probe_sha256 806e6776443b318687c0ff61e00530c5ec9b8118aec044acce8a5f4102865641
expect_field init_sha256 1aad25d58aa68e2306523b58babd21e9eecbe7ef656885156da94338026fb83e
expect_field packer_sha256 09ff4699c9232d439e65e8e725c7bdcc949c59438602569650a3c9048b8f6629
expect_field initramfs_sha256 cc76bbb81aad1b0fdaf7e6122f23aff25c24c488dacd9a85d2d34dc5e9ec8cf5
expect_field kernel_sha256 842932f8f994b201309efc386a5f1049377388aa8689b17e987e5c681e58e1ef
expect_field evidence_sha256 00107207e56e784b0cd1fcd43ff63d819a6cba64a21e6422f2a12f898d2dffc0
expect_field hardware_host t560-proxmox
expect_field kernel 7.0.2-5-pve
expect_field hypervisor KVM
expect_field qemu_version 11.0.0
expect_field guest_distinct true
expect_field guest_pid_1 true
expect_field active_lsm lockdown,capability,yama,apparmor,bpf,ima,evm
for enabled in bpf_lsm_active securityfs bpffs btf qemu_extinct archive_reproducible init_static packer_static bpf_lsm_microhost native_microhost_bytes_created; do
  expect_field "$enabled" true
done
expect_field guest_disk none
expect_field guest_network none
expect_field python_executed false
expect_field rust_executed false
expect_field bpf_program_loaded false
expect_field backend_candidate_complete false
expect_field native_material_matrix_bytes_created false
for boundary in material_peer_matrix same_uid_peer_isolation material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done
expect_field action_9025_decision DENY451
expect_field next_stage BPF_LSM_PROGRAM_LOAD

SOURCE_COMMIT="$(field material_source_commit)"
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'microhost source commit is absent'
for pair in contract_path:contract_sha256 source_path:source_sha256 packer_source_path:packer_source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256 host_gate_path:host_gate_sha256 host_probe_path:host_probe_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  path="$(field "$path_key")"; expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the microhost source commit"
done
if git cat-file -e "$SOURCE_COMMIT:tools/loom/bpf/loom_kernel_peer_v12.bpf.c" 2>/dev/null; then
  fail 'BPF mediator bytes existed before microhost freeze'
fi
[[ "$(file_hash "$(field semantic_manifest_path)")" == "$(field semantic_manifest_sha256)" ]] ||
  fail 'semantic manifest drifted'
[[ "$(file_hash "$(field backend_manifest_path)")" == "$(field backend_manifest_sha256)" ]] ||
  fail 'negative backend manifest drifted'
[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'microhost evidence drifted'

expect_evidence schema loom-kernel-peer-microhost-v12-evidence-v1
expect_evidence stage BPF_LSM_MICROHOST_V12
expect_evidence material_source_commit "$SOURCE_COMMIT"
expect_evidence initramfs_sha256 "$(field initramfs_sha256)"
expect_evidence kernel_sha256 "$(field kernel_sha256)"
expect_evidence hardware_host t560-proxmox
expect_evidence guest_distinct true
expect_evidence guest_pid_1 true
expect_evidence active_lsm lockdown,capability,yama,apparmor,bpf,ima,evm
expect_evidence bpf_lsm_active true
expect_evidence guest_disk none
expect_evidence guest_network none
expect_evidence qemu_extinct true
expect_evidence bpf_lsm_microhost true
expect_evidence bpf_program_loaded false
expect_evidence material_peer_matrix false
expect_evidence same_uid_peer_isolation false
expect_evidence action_9025_decision DENY451
expect_evidence next_stage BPF_LSM_PROGRAM_LOAD
[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'host command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'host result hash drifted'
[[ "$(printf '%s' "$(evidence_field local_selftest_result)" | stream_hash)" == "$(evidence_field local_selftest_result_sha256)" ]] ||
  fail 'local result hash drifted'
[[ "$(evidence_field result)" == *"host_output_sha256=$(evidence_field host_output_sha256)"* ]] ||
  fail 'transport receipt does not bind host output'

local_result="$(bash "$(field selftest_path)" 2>/dev/null)"
[[ "$local_result" == "$(evidence_field local_selftest_result)" ]] ||
  fail 'source-fresh microhost build drifted'
[[ "$local_result" == *'archive_reproducible=true init_static=true packer_static=true guest_disk=none guest_network=none'* ]] ||
  fail 'microhost material boundary drifted'

printf 'sounio-loom-kernel-peer-microhost-v12-freeze-selftest: PASS semantic_authority=Sounio action=9025 material_producer=C++20 material_role=MATERIAL_BOOTSTRAP manifest_sha256=%s evidence_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve hypervisor=KVM guest_distinct=true guest_pid_1=true active_lsm=lockdown,capability,yama,apparmor,bpf,ima,evm bpf_lsm_active=true securityfs=true bpffs=true btf=true init_static=true packer_static=true archive_reproducible=true guest_disk=none guest_network=none qemu_extinct=true bpf_lsm_microhost=true bpf_program_loaded=false backend_candidate_complete=false native_material_matrix_bytes_created=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=BPF_LSM_PROGRAM_LOAD\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
