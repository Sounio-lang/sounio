#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
MANIFEST=tools/loom/kernel_peer_dumpable_prlimit_falsification_v12.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-dumpable-prlimit-falsification-v12-host-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-dumpable-prlimit-falsification-v12-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  fail 'falsification freeze inputs are absent or linked'
! grep -Fq '__' "$MANIFEST" || fail 'manifest contains an unresolved marker'
! grep -Fq '__' "$EVIDENCE" || fail 'evidence contains an unresolved marker'
expect_field schema loom-kernel-peer-dumpable-prlimit-falsification-v12-freeze-v1
expect_field stage V12_MATERIAL_HYPOTHESIS_FALSIFIED
expect_field semantic_authority Sounio
expect_field action 9025
expect_field semantic_manifest_sha256 daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30
expect_field matrix_manifest_sha256 1692782657cbe6fe7a548b6f11d4d542d24fe05569686d536a4c69af0775cd75
expect_field material_source_commit 856dd1deed7743515b6eaaf0ac35793ee091f63c
expect_field producing_language C++20
expect_field language_role MATERIAL_BOOTSTRAP
expect_field transitory true
expect_field contract_sha256 cfac5d59dfc965ca2891704f30199876be3579a773ebac99d46fd2089811e89e
expect_field source_sha256 d6d907fa3d1fad1b224a9fe1280091ea6fd5c0d209d14884b731505d47bd83b8
expect_field base_source_sha256 54a447bd18a7d0319edda89fb01c593e5e28448c3994c4c0002c7b74795b4ab2
expect_field build_script_sha256 bbd9d8d1734904b1b767726dc1372bf294dc16675ad3d06b41be17263f77faf1
expect_field selftest_sha256 51842b973577eb5c91bd5e60ed9ba968aec6bf77fede816ee4ad071df275f8b3
expect_field host_gate_sha256 c61baae4fb437cb014b04277606eb0cef6292aacc4526d92170cf22d883b9037
expect_field host_probe_sha256 b0c2eaef2942d27aff73a07fdd0d4575628d65e4ccb92554199c93edec62a79d
expect_field evidence_sha256 feca1802d54b2fdd746e7e14252b7e2452316f666442408b3b59bdd8a3e455a6
expect_field base_initramfs_sha256 72fe7194bcd065a7648c467a171ddb7274ed04ecbec548c075e133b4c976b7bf
expect_field init_sha256 f94baf38c7bea37f52344814bcba113bea0738440602ce24db4dec04b067f84b
expect_field packer_sha256 09ff4699c9232d439e65e8e725c7bdcc949c59438602569650a3c9048b8f6629
expect_field kernel_sha256 842932f8f994b201309efc386a5f1049377388aa8689b17e987e5c681e58e1ef
expect_field host_output_sha256 45265b44e051ee68373e14bcc8e18c5f0466b0bf8deb92d68fd12d4614ea0bdf
expect_field hardware_host t560-proxmox
expect_field kernel 7.0.2-5-pve
expect_field hypervisor KVM
expect_field qemu_version 11.0.0
expect_field active_lsm lockdown,capability,bpf,ima,evm
expect_field operation 9
expect_field syscall prlimit64
expect_field vertex DUMPABLE_ONLY_CONTROL
expect_field frozen_expected REFUSED_BEFORE_EFFECT
expect_field material_observed EFFECT_COMPLETED
expect_field completion LIMIT_CHANGED_RESTORED
expect_field v12_hypothesis_falsified true
expect_field counterexamples 1
for enabled in same_four_uids same_user_namespace distinct_processes distinct_pidfds distinct_start_ticks distinct_cgroups target_limit_restored typed_witness all_epoch_objects_extinct guest_root_traversable qemu_extinct; do
  expect_field "$enabled" true
done
expect_field target_dumpable 0
expect_field attacker_seccomp 0
expect_field mediator absent
expect_field principal_capability CAP_SYS_NICE_ONLY
expect_field guest_disk none
expect_field guest_network none
for boundary in controls_executed material_peer_matrix same_uid_peer_isolation material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed; do
  expect_field "$boundary" false
done
expect_field action_9025_decision DENY451
expect_field next_stage SOUNIO_V13_GARDEN

SOURCE_COMMIT="$(field material_source_commit)"
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'falsification source commit is absent'
for pair in contract_path:contract_sha256 source_path:source_sha256 base_source_path:base_source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256 host_gate_path:host_gate_sha256 host_probe_path:host_probe_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  path="$(field "$path_key")"; expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the falsification source commit"
done
[[ "$(file_hash "$(field semantic_manifest_path)")" == "$(field semantic_manifest_sha256)" ]] ||
  fail 'Sounio V12 semantic manifest drifted'
[[ "$(file_hash "$(field matrix_manifest_path)")" == "$(field matrix_manifest_sha256)" ]] ||
  fail 'V12 peer-matrix manifest drifted'
[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'falsification evidence drifted'

expect_evidence schema loom-kernel-peer-dumpable-prlimit-falsification-v12-evidence-v1
expect_evidence stage V12_MATERIAL_HYPOTHESIS_FALSIFIED
expect_evidence material_source_commit "$SOURCE_COMMIT"
for key in semantic_manifest_sha256 matrix_manifest_sha256 contract_sha256 source_sha256 base_source_sha256 build_script_sha256 selftest_sha256 host_gate_sha256 host_probe_sha256 base_initramfs_sha256 init_sha256 packer_sha256 kernel_sha256 host_output_sha256 hardware_host kernel hypervisor qemu_version active_lsm operation syscall vertex frozen_expected material_observed completion v12_hypothesis_falsified counterexamples same_four_uids same_user_namespace distinct_processes distinct_pidfds distinct_start_ticks distinct_cgroups target_dumpable attacker_seccomp mediator principal_capability target_limit_restored typed_witness all_epoch_objects_extinct guest_root_traversable guest_disk guest_network qemu_extinct controls_executed material_peer_matrix same_uid_peer_isolation action_9025_decision material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed next_stage; do
  expect_evidence "$key" "$(field "$key")"
done
[[ "$(evidence_field host_boot_id)" != "$(evidence_field guest_boot_id)" ]] ||
  fail 'guest and host boot identities alias'

counterexample="$(evidence_field counterexample)"
for fact in 'COUNTEREXAMPLE vertex=DUMPABLE_ONLY_CONTROL' operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT material_observed=EFFECT_COMPLETED completion=LIMIT_CHANGED_RESTORED errno=NONE same_four_uids=true same_user_namespace=true distinct_processes=true distinct_pidfds=true distinct_start_ticks=true distinct_cgroups=true target_dumpable=0 attacker_seccomp=0 mediator=absent principal_capability=CAP_SYS_NICE_ONLY target_limit_restored=true all_epoch_objects_extinct=true python_executed=false rust_executed=false; do
  [[ " $counterexample " == *" $fact "* ]] || fail "counterexample omitted $fact"
done
[[ "$(grep -oE '[0-9a-f]{64}' <<<"$counterexample" | wc -l)" == 5 ]] ||
  fail 'counterexample does not carry five independent hashes'
[[ "$(printf '%s' "$(evidence_field local_selftest_command)" | stream_hash)" == "$(evidence_field local_selftest_command_sha256)" ]] ||
  fail 'local selftest command hash drifted'
[[ "$(printf '%s' "$(evidence_field local_selftest_result)" | stream_hash)" == "$(evidence_field local_selftest_result_sha256)" ]] ||
  fail 'local selftest result hash drifted'
[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'host command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'host result hash drifted'
for fact in host_output_sha256=45265b44e051ee68373e14bcc8e18c5f0466b0bf8deb92d68fd12d4614ea0bdf operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT material_observed=EFFECT_COMPLETED v12_hypothesis_falsified=true same_four_uids=true target_dumpable=0 attacker_seccomp=0 mediator=absent principal_capability=CAP_SYS_NICE_ONLY typed_witness=true all_epoch_objects_extinct=true controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false next_stage=SOUNIO_V13_GARDEN; do
  [[ " $(evidence_field result) " == *" $fact "* ]] || fail "transport result omitted $fact"
done

local_result="$(bash "$(field selftest_path)" 2>/dev/null)"
[[ "$local_result" == "$(evidence_field local_selftest_result)" ]] ||
  fail 'source-fresh falsification build drifted'

printf 'sounio-loom-kernel-peer-dumpable-prlimit-falsification-v12-freeze-selftest: PASS semantic_authority=Sounio action=9025 manifest_sha256=%s evidence_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve hypervisor=KVM operation=9 syscall=prlimit64 vertex=DUMPABLE_ONLY_CONTROL frozen_expected=REFUSED_BEFORE_EFFECT material_observed=EFFECT_COMPLETED completion=LIMIT_CHANGED_RESTORED v12_hypothesis_falsified=true counterexamples=1 same_four_uids=true target_dumpable=0 attacker_seccomp=0 mediator=absent principal_capability=CAP_SYS_NICE_ONLY typed_witness=true all_epoch_objects_extinct=true controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=SOUNIO_V13_GARDEN\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
