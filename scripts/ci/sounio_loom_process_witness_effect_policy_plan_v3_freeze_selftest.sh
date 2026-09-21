#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v3.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v3-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v3-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen path is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen path hash drifted: $path"
}

require_line() {
  local path="$1" value="$2"
  grep -Fxq "$value" "$path" || fail "required line is absent: $value"
}

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V3.md \
  9c49364e27d36b2057e9ba33606c6dcbeb83c89b86b501645de9ea7fbf8e8185
expect_hash tools/loom/process_witness_effect_policy_plan_v3_main.sio \
  fe53eadd81580e8d382e7d5edc8b665621f684087e7a2102a98285a18ea0c7a7
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v3.sh \
  0b345b75cad68e142baba4e3341f742b71106806a792dbc615fe36ab437a4909
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v3_selftest.sh \
  ee7423872c0bf2fbaaa8487e9aee022b850ccc33efd45d40124539d413d03633
expect_hash tools/loom/process_witness_effect_policy_plan_v2.freeze.v1 \
  d66b13252479252d5922ee0091e51a5bdb6a5eca9a592bb21f5db9dde344fee9
expect_hash tools/loom/evidence/loom-process-witness-effect-policy-host-attempt-v1-20260828.txt \
  e702ceb3e2149d2d83cd054b147f9130e97fdb2d082b4ee839e24d4fcfdd24bb
expect_hash "$EVIDENCE" \
  515c3166e9e3943d7132b039590a359a95126be51582090017da4d286fa787e2

require_line "$MANIFEST" 'schema=loom-process-witness-effect-policy-plan-v3-freeze-v1'
require_line "$MANIFEST" 'stage=SEMANTICS_FROZEN'
require_line "$MANIFEST" 'producing_language=Sounio'
require_line "$MANIFEST" 'language_role=SEMANTIC_POLICY_PLAN'
require_line "$MANIFEST" 'semantic_authority=Sounio'
require_line "$MANIFEST" 'action=9025'
require_line "$MANIFEST" 'garden_commit=8d6ca448fb6f78afb0168d4a78c940768a4eebd8'
require_line "$MANIFEST" 'sounio_executable_commit=68ef3f5852ac216d67d41573ea4169c3043415ca'
require_line "$MANIFEST" 'executable_sha256=d36ed365f5fc95b453f49dd4184676a2d06bb082f4fb2e48a2f9a7873ce3281f'
require_line "$MANIFEST" 'bundle_sha256=e365f0b1e0028bd0cddd129e1110126dd82b0c33ca268d427c39fe870b0efe34'
require_line "$MANIFEST" 'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE'
require_line "$MANIFEST" 'root_read_only=true'
require_line "$MANIFEST" 'root_owned=true'
require_line "$MANIFEST" 'root_single_link_files=true'
require_line "$MANIFEST" 'dynamic_linker_visible=false'
require_line "$MANIFEST" 'host_root_visible=false'
require_line "$MANIFEST" 'pathname_syscalls_after_filter=0'
require_line "$MANIFEST" 'allowed_syscall_count=4'
require_line "$MANIFEST" 'allowed_syscalls=0,1,60,322'
require_line "$MANIFEST" 'read_constraint=fd0'
require_line "$MANIFEST" 'write_constraint=fd1_or_fd2'
require_line "$MANIFEST" 'execveat_constraint=fd3_and_AT_EMPTY_PATH'
require_line "$MANIFEST" 'architecture=AUDIT_ARCH_X86_64'
require_line "$MANIFEST" 'architecture_mismatch=KILL_PROCESS'
require_line "$MANIFEST" 'default_action=ERRNO_EP1'
require_line "$MANIFEST" 'landlock_required=false'
require_line "$MANIFEST" 'landlock_fallback=false'
require_line "$MANIFEST" 'family_10_probe=personality_change'
require_line "$MANIFEST" 'v2_materializable=false'
require_line "$MANIFEST" 'v3_required_for_native=true'
require_line "$MANIFEST" 'expected_results_encoded_in_shell=false'
require_line "$MANIFEST" 'python_executable_invoked=false'
require_line "$MANIFEST" 'rust_executable_invoked=false'
require_line "$MANIFEST" 'material_coverage=false'
require_line "$MANIFEST" 'complete_effects=false'
require_line "$MANIFEST" 'material_execution=false'
require_line "$MANIFEST" 'launch_open=false'
require_line "$MANIFEST" 'recycle_open=false'
require_line "$MANIFEST" 'exec_attached=false'
require_line "$MANIFEST" 'commit_attached=false'
require_line "$MANIFEST" 'ci_attached=false'
require_line "$MANIFEST" 'parity_open=false'
require_line "$MANIFEST" 'claim_ready=false'
require_line "$MANIFEST" 'evidence_sha256=515c3166e9e3943d7132b039590a359a95126be51582090017da4d286fa787e2'

require_line "$EVIDENCE" 'hardware_arch=x86_64'
require_line "$EVIDENCE" 'hardware_kernel=Linux_7.0.2-5-pve'
require_line "$EVIDENCE" 'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE'
require_line "$EVIDENCE" 'root_read_only=true'
require_line "$EVIDENCE" 'host_root_visible=false'
require_line "$EVIDENCE" 'landlock_required=false'
require_line "$EVIDENCE" 'family_10_probe=personality_change'
require_line "$EVIDENCE" 'v2_native_consumption=refused'
require_line "$EVIDENCE" 'native_executable_invoked=false'
require_line "$EVIDENCE" 'expected_results_source=Sounio'
require_line "$EVIDENCE" 'expected_results_encoded_in_shell=false'
require_line "$EVIDENCE" 'python_executable_invoked=false'
require_line "$EVIDENCE" 'rust_executable_invoked=false'
require_line "$EVIDENCE" 'complete_effects=false'
require_line "$EVIDENCE" 'material_execution=false'

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v3_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v3-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V3 gate failed'
[[ "$result" == *'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change landlock_required=false v2_native=refused native_executed=false deterministic=true shell_expected_results=false'* ]] ||
  fail 'Sounio V3 object boundary, surface, or decision classes drifted'
[[ "$result" == *'material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V3 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v3-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE executable_sha256=d36ed365f5fc95b453f49dd4184676a2d06bb082f4fb2e48a2f9a7873ce3281f bundle_sha256=e365f0b1e0028bd0cddd129e1110126dd82b0c33ca268d427c39fe870b0efe34 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 family10=personality_change v2_native=refused material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
