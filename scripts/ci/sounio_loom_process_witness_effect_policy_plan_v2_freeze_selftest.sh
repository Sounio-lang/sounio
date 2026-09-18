#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v2.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v2-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v2-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V2.md \
  08edcae4f07091b999f6c56e6a95cf7bca6d0bfa54ed6b9e13f2e49ea14a90ad
expect_hash tools/loom/process_witness_effect_policy_plan_v2_main.sio \
  d0285b0c2f724c68622deb5bda7406e57c19fb0deaf0288e9047cf56d6c1a395
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v2.sh \
  af6755b19fc59ecc94d2512345fb6a7150beedd11ee6480b60d4148d9623808c
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v2_selftest.sh \
  bade496fb6a6c2617c9fb28e9049f1df69f49d359b1b7753abe413ad0f08a21c
expect_hash tools/loom/process_witness_effect_policy_plan.freeze.v1 \
  14ee27eee71f04d1aa5462426379b37bb9c775215e94e17a864dbea308e43f21
expect_hash tools/loom/effect_closure_authority.freeze.v1 \
  c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
expect_hash tools/loom/process_witness_host.runtime.v1 \
  eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00
expect_hash "$EVIDENCE" \
  daa5496a949524712845ba99aa581a3dfd242f39cb7d843ed16891963a6a1cd8

require_line "$MANIFEST" 'schema=loom-process-witness-effect-policy-plan-v2-freeze-v1'
require_line "$MANIFEST" 'stage=SEMANTICS_FROZEN'
require_line "$MANIFEST" 'producing_language=Sounio'
require_line "$MANIFEST" 'language_role=SEMANTIC_POLICY_PLAN'
require_line "$MANIFEST" 'semantic_authority=Sounio'
require_line "$MANIFEST" 'action=9025'
require_line "$MANIFEST" 'garden_commit=13ccfc7896b401983a4125e4df3db5e3a243a993'
require_line "$MANIFEST" 'sounio_executable_commit=13aa57729fb4aac233db1381844dc89952216dbb'
require_line "$MANIFEST" 'executable_sha256=eb9d105c7f9069596a67f836449314806e2a8b296d85f26da35f884b215b7e25'
require_line "$MANIFEST" 'bundle_sha256=5d9f3528e8dd5238c388f5bfd00606eeb13ddfa927ab48bca296fc69b9e2d236'
require_line "$MANIFEST" 'allowed_syscall_count=4'
require_line "$MANIFEST" 'allowed_syscalls=0,1,60,322'
require_line "$MANIFEST" 'read_constraint=fd0'
require_line "$MANIFEST" 'write_constraint=fd1_or_fd2'
require_line "$MANIFEST" 'execveat_constraint=fd3_and_AT_EMPTY_PATH'
require_line "$MANIFEST" 'architecture=AUDIT_ARCH_X86_64'
require_line "$MANIFEST" 'architecture_mismatch=KILL_PROCESS'
require_line "$MANIFEST" 'default_action=ERRNO_EP1'
require_line "$MANIFEST" 'landlock_required=true'
require_line "$MANIFEST" 'landlock_fallback=false'
require_line "$MANIFEST" 'v1_sufficient_for_native=false'
require_line "$MANIFEST" 'v2_required_for_native=true'
require_line "$MANIFEST" 'expected_results_encoded_in_shell=false'
require_line "$MANIFEST" 'python_executable_invoked=false'
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
require_line "$MANIFEST" 'evidence_sha256=daa5496a949524712845ba99aa581a3dfd242f39cb7d843ed16891963a6a1cd8'

require_line "$EVIDENCE" 'allowed_syscalls=0,1,60,322'
require_line "$EVIDENCE" 'allowlist_kind=positive'
require_line "$EVIDENCE" 'blacklist_fallback=false'
require_line "$EVIDENCE" 'landlock_required=true'
require_line "$EVIDENCE" 'v1_native_consumption=refused'
require_line "$EVIDENCE" 'native_executable_invoked=false'
require_line "$EVIDENCE" 'expected_results_source=Sounio'
require_line "$EVIDENCE" 'expected_results_encoded_in_shell=false'
require_line "$EVIDENCE" 'python_executable_invoked=false'
require_line "$EVIDENCE" 'complete_effects=false'
require_line "$EVIDENCE" 'material_execution=false'

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v2_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v2-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V2 gate failed'
[[ "$result" == *'allowed_syscalls=4 syscall_surface=0+1+60+322 authority_cases=14 complete=ALLOW current=DENY447 missing_known=DENY447x11 missing_unknown=DENY452 v1_native=refused native_executed=false python_control=refused python_executed=false deterministic=true shell_expected_results=false'* ]] ||
  fail 'Sounio V2 surface or decision classes drifted'
[[ "$result" == *'material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V2 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v2-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s executable_sha256=eb9d105c7f9069596a67f836449314806e2a8b296d85f26da35f884b215b7e25 bundle_sha256=5d9f3528e8dd5238c388f5bfd00606eeb13ddfa927ab48bca296fc69b9e2d236 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 v1_native=refused material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
