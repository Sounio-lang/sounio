#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v1-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_CELL_V1.md \
  21630b55ce12d7823e7d66408a2ef7af53d833a4b83182b130a76e83c6395cb3
expect_hash tools/loom/process_witness_effect_policy_plan_main.sio \
  abe96bae04fe8ad371e332aed188cf47fb65f1f333713d2ea6b8d455810d9790
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan.sh \
  08ce7778fccd93e4b469593ec86cc9d0d910682cc885a3bd9a3f5a007716d041
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_selftest.sh \
  1d346f7c716aac70e775abbab02744b532906649596fd2f699afe22ff5ffc00a
expect_hash tools/loom/effect_closure_authority.freeze.v1 \
  c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
expect_hash tools/loom/process_witness_host.runtime.v1 \
  eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00
expect_hash "$EVIDENCE" \
  c93c6dabf144555e324504fc0dce41ada94df5096f5afa444846bb02c3c00199

require_line "$MANIFEST" 'schema=loom-process-witness-effect-policy-plan-freeze-v1'
require_line "$MANIFEST" 'stage=SEMANTICS_FROZEN'
require_line "$MANIFEST" 'producing_language=Sounio'
require_line "$MANIFEST" 'language_role=SEMANTIC_POLICY_PLAN'
require_line "$MANIFEST" 'semantic_authority=Sounio'
require_line "$MANIFEST" 'action=9025'
require_line "$MANIFEST" 'garden_commit=e2fe391d6ccfc5ffc2813b4fc9d6345ba54afd8a'
require_line "$MANIFEST" 'sounio_executable_commit=29887d05a3f3a2b26c326328fc7aeed7744dbdac'
require_line "$MANIFEST" 'executable_sha256=fe000ce51ef59322061d26da33d36aa57801487b903c9e1c6617021d02b8cf13'
require_line "$MANIFEST" 'bundle_sha256=5560d1780df5bc83beb7b966c8533ad703d2b562aff0f8c6e3b0eed09d56e6a2'
require_line "$MANIFEST" 'family_count=12'
require_line "$MANIFEST" 'treatment_count=12'
require_line "$MANIFEST" 'sabotage_count=12'
require_line "$MANIFEST" 'authority_case_count=14'
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
require_line "$MANIFEST" 'evidence_sha256=c93c6dabf144555e324504fc0dce41ada94df5096f5afa444846bb02c3c00199'

require_line "$EVIDENCE" 'expected_results_source=Sounio'
require_line "$EVIDENCE" 'expected_results_encoded_in_shell=false'
require_line "$EVIDENCE" 'python_control=refused'
require_line "$EVIDENCE" 'python_executable_invoked=false'
require_line "$EVIDENCE" 'deterministic_rebuild=true'
require_line "$EVIDENCE" 'complete_effects=false'
require_line "$EVIDENCE" 'material_execution=false'
require_line "$EVIDENCE" 'launch_open=false'
require_line "$EVIDENCE" 'recycle_open=false'
require_line "$EVIDENCE" 'exec_attached=false'
require_line "$EVIDENCE" 'commit_attached=false'
require_line "$EVIDENCE" 'ci_attached=false'
require_line "$EVIDENCE" 'parity_open=false'
require_line "$EVIDENCE" 'claim_ready=false'

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio policy-plan gate failed'
[[ "$result" == *'families=12 treatments=12 sabotages=12 authority_cases=14 complete=ALLOW current=DENY447 missing_known=DENY447x11 missing_unknown=DENY452 python_control=refused python_executed=false deterministic=true shell_expected_results=false'* ]] ||
  fail 'Sounio policy-plan decision classes drifted'
[[ "$result" == *'complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio policy-plan gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s executable_sha256=fe000ce51ef59322061d26da33d36aa57801487b903c9e1c6617021d02b8cf13 bundle_sha256=5560d1780df5bc83beb7b966c8533ad703d2b562aff0f8c6e3b0eed09d56e6a2 families=12 treatments=12 sabotages=12 complete=ALLOW current=DENY447 material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
