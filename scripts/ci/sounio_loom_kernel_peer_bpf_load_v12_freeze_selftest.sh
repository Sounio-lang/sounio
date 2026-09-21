#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
MANIFEST=tools/loom/kernel_peer_bpf_load_v12.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-bpf-load-v12-host-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-bpf-load-v12-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  fail 'BPF-load freeze inputs are absent or linked'
! grep -Fq '__' "$MANIFEST" || fail 'manifest contains an unresolved marker'
! grep -Fq '__' "$EVIDENCE" || fail 'evidence contains an unresolved marker'
expect_field schema loom-kernel-peer-bpf-load-v12-freeze-v1
expect_field stage BPF_LSM_MEDIATOR_LOAD_FROZEN_V12
expect_field semantic_authority Sounio
expect_field action 9025
expect_field semantic_manifest_sha256 daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30
expect_field microhost_manifest_sha256 0c0f49287a838b9fd1836d4771036ca4ba5bba62a69bd9993c5056ef65186328
expect_field material_source_commit e70813e44c0a3d9e66c557d77ab9b9c177cb9348
expect_field producing_language C+BPF+C++20
expect_field language_role MATERIAL_BOOTSTRAP
expect_field transitory true
expect_field contract_sha256 2f7ab7aaf24a3d2a60513bd14e28764193a6658c6e0072d39acd49ab67ddd36b
expect_field bpf_source_sha256 d7d1e9d482fd8b80aa47f9ac171f33478e9960f53c54b50643e8ef004b433b08
expect_field loader_source_sha256 7f3db89ef62d8ebc24b241a2121e83d81a5e18bd51dd4012d84a9b7a03212abc
expect_field init_source_sha256 a9b193f84a3b2e2219f6d1302456db3ed26b04ed0b65d061473560bca07bb4da
expect_field build_script_sha256 3a119859d9a655d5371111ab93f343cdbb7d576509c3c6fd6b265d240eaaa996
expect_field selftest_sha256 150d8253a2199d486ee49893004a27d8261606655c876e39ffa38afd7ccda04b
expect_field host_gate_sha256 a12d96742f135b642c05e7aa930cd84d2a673f9b9b501d64cf3dad34326b90dd
expect_field host_probe_sha256 5c33e2fc273c7556827b15801d143e6067387d3fcb2ae84f55913ae74279b6fd
expect_field init_sha256 40ab7a76bb8b9673f48435d09cefc8e8cf77b515299c02c550c60eb0d57eae06
expect_field loader_object_sha256 300b52202c2903d5fd81fc1219d8ea6fcfa8993108881c61ec08b04d9b372463
expect_field bpf_object_sha256 633849ca3dae7c8898c78e8e0049af4d689e04a27a468829fdc257ee427675d5
expect_field packer_sha256 09ff4699c9232d439e65e8e725c7bdcc949c59438602569650a3c9048b8f6629
expect_field base_initramfs_sha256 e93f6cef0a7c949725ccc21adfde92cf863b593bdf6658632322b75e024b3f74
expect_field loader_sha256 fbcc27e96e1cc10091ad5c60d81d5ddf70a39097031f4b6589422738935bb92b
expect_field final_initramfs_sha256 ea48a4910a65184186e18988a19f036da03efcf6f03ff52cca7aff4969a7b27e
expect_field kernel_sha256 842932f8f994b201309efc386a5f1049377388aa8689b17e987e5c681e58e1ef
expect_field libbpf_sha256 0c49fc88c249ba8f584103d5db55a84bd84735d8529e64122a3433c50b21a5b7
expect_field evidence_sha256 e78eda277931ab7608d1041d25241af0b899552265313f0e486e2bbcb4b45dd8
expect_field hardware_host t560-proxmox
expect_field kernel 7.0.2-5-pve
expect_field hypervisor KVM
expect_field qemu_version 11.0.0
expect_field guest_distinct true
expect_field guest_pid_1 true
expect_field active_lsm lockdown,capability,yama,apparmor,bpf,ima,evm
for enabled in bpf_lsm_active btf_core loader_exited loader_link_fds_closed pin_survival link_extinction loader_reproducibility_twin cross_run_reproducibility qemu_extinct bpf_lsm_microhost bpf_program_loaded native_microhost_bytes_created; do
  expect_field "$enabled" true
done
expect_field programs_loaded 3
expect_field links_pinned 3
expect_field pins_unlinked 3
expect_field independent_guest_boots 2
expect_field guest_disk none
expect_field guest_network none
expect_field python_executed false
expect_field rust_executed false
expect_field backend_candidate_complete false
expect_field native_material_matrix_bytes_created false
for boundary in material_peer_matrix same_uid_peer_isolation material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done
expect_field action_9025_decision DENY451
expect_field next_stage BPF_LSM_PEER_MATRIX

SOURCE_COMMIT="$(field material_source_commit)"
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'BPF-load source commit is absent'
for pair in contract_path:contract_sha256 bpf_source_path:bpf_source_sha256 loader_source_path:loader_source_sha256 init_source_path:init_source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256 host_gate_path:host_gate_sha256 host_probe_path:host_probe_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  path="$(field "$path_key")"; expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the BPF-load source commit"
done
[[ "$(file_hash "$(field semantic_manifest_path)")" == "$(field semantic_manifest_sha256)" ]] ||
  fail 'semantic manifest drifted'
[[ "$(file_hash "$(field microhost_manifest_path)")" == "$(field microhost_manifest_sha256)" ]] ||
  fail 'microhost manifest drifted'
[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'BPF-load evidence drifted'

expect_evidence schema loom-kernel-peer-bpf-load-v12-evidence-v1
expect_evidence stage BPF_LSM_MEDIATOR_LOAD_V12
expect_evidence material_source_commit "$SOURCE_COMMIT"
for key in init_sha256 loader_object_sha256 bpf_object_sha256 packer_sha256 base_initramfs_sha256 loader_sha256 final_initramfs_sha256 kernel_sha256 libbpf_sha256 programs_loaded links_pinned loader_exited loader_link_fds_closed pin_survival pins_unlinked link_extinction loader_reproducibility_twin cross_run_reproducibility independent_guest_boots guest_disk guest_network qemu_extinct bpf_lsm_microhost bpf_program_loaded backend_candidate_complete native_material_matrix_bytes_created material_peer_matrix same_uid_peer_isolation action_9025_decision next_stage; do
  expect_evidence "$key" "$(field "$key")"
done
expect_evidence hook_1 lsm/task_kill
expect_evidence hook_2 lsm/ptrace_access_check
expect_evidence hook_3 lsm/task_prlimit
expect_evidence link_identity_query BPF_OBJ_GET+BPF_OBJ_GET_INFO_BY_FD
expect_evidence link_extinction_query BPF_LINK_GET_FD_BY_ID
expect_evidence attempt_1 REFUSED_PRE_BOOT_MISSING_LIBELF_DEVELOPMENT_SYMLINK
expect_evidence attempt_2 REFUSED_IN_GUEST_MISSING_LIBSTDCXX
expect_evidence python_executed false
expect_evidence rust_executed false

BOOT_A="$(evidence_field guest_boot_id_a)"
BOOT_B="$(evidence_field guest_boot_id_b)"
HOST_BOOT="$(evidence_field host_boot_id)"
[[ "$BOOT_A" != "$BOOT_B" && "$BOOT_A" != "$HOST_BOOT" && "$BOOT_B" != "$HOST_BOOT" ]] ||
  fail 'independent boot identities alias'
for boot in raw_boot_a raw_boot_b; do
  receipt="$(evidence_field "$boot")"
  for fact in programs_loaded=3 links_pinned=3 loader_exited=true loader_link_fds_closed=true pin_survival=true pins_unlinked=3 link_extinction=true guest_disk=none guest_network=none material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
    [[ " $receipt " == *" $fact "* ]] || fail "$boot omitted $fact"
  done
done
[[ "$(printf '%s' "$(evidence_field local_selftest_command)" | stream_hash)" == "$(evidence_field local_selftest_command_sha256)" ]] ||
  fail 'local selftest command hash drifted'
[[ "$(printf '%s' "$(evidence_field local_selftest_result)" | stream_hash)" == "$(evidence_field local_selftest_result_sha256)" ]] ||
  fail 'local selftest result hash drifted'
[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'host command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'host result hash drifted'
[[ "$(evidence_field result)" == *"host_output_sha256=edd14ecc3d6ad1372d0715ce2797e4b476846bef2a0c0bde2c4f810ff35437f0"* ]] ||
  fail 'transport result omitted the measured host output hash'

local_result="$(bash "$(field selftest_path)" 2>/dev/null)"
[[ "$local_result" == "$(evidence_field local_selftest_result)" ]] ||
  fail 'source-fresh BPF-load build drifted'

printf 'sounio-loom-kernel-peer-bpf-load-v12-freeze-selftest: PASS semantic_authority=Sounio action=9025 material_producer=C+BPF+C++20 material_role=MATERIAL_BOOTSTRAP manifest_sha256=%s evidence_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve hypervisor=KVM programs_loaded=3 links_pinned=3 loader_exited=true loader_link_fds_closed=true pin_survival=true pins_unlinked=3 link_extinction=true loader_reproducibility_twin=true cross_run_reproducibility=true independent_guest_boots=2 guest_disk=none guest_network=none qemu_extinct=true bpf_program_loaded=true backend_candidate_complete=false native_material_matrix_bytes_created=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=BPF_LSM_PEER_MATRIX\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
