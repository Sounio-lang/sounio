#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v11.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v11-20260829.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v11-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V11.md \
  c065f6f30c721711ffa9cff74b3961043a9ca5ce1ab4938b440d984afab524cb
expect_hash tools/loom/process_witness_effect_policy_plan_v11_main.sio \
  42fe8c08510f00159f0ad63cb0aac620c35776d661a41691d290ffd44ae402e2
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v11.sh \
  6e33fa37c5fbbb1acd3570d4c8316ab37fff86ca6e528f2a8ffb731a05749513
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v11_selftest.sh \
  aff0683aca32e325cb10c72e0005457aa8eaabfe71b0509a89dabf0033a92517
expect_hash tools/loom/process_witness_effect_policy_plan_v10.freeze.v1 \
  9e7f42fd4bd18fd2b5f996b279a67f46a50546a20ef6949e4dc069c16b3d0dda
expect_hash tools/loom/evidence/loom-process-witness-effect-root-v10-host-attempt-v1-20260828.txt \
  96bea5a8306d61ed4528b5b29f92493c98fe6e95c1c6c8ee28930b0f5c2b0ca5
expect_hash "$EVIDENCE" \
  5191a0d46b4ef4086967ff90cc482efa673c65ff8764626245132e720cc2bc5d

for line in \
  'schema=loom-process-witness-effect-policy-plan-v11-freeze-v1' \
  'stage=SEMANTICS_FROZEN' \
  'producing_language=Sounio' \
  'language_role=SEMANTIC_POLICY_PLAN' \
  'semantic_authority=Sounio' \
  'action=9025' \
  'garden_commit=bdacaebe62d7d1a22c3462264a83cb4739bf3489' \
  'sounio_executable_commit=23b3756e1b40771e46f2e0a0ceebd1d5f7a8412b' \
  'source_sha256=42fe8c08510f00159f0ad63cb0aac620c35776d661a41691d290ffd44ae402e2' \
  'executable_sha256=cf820428b1927128b82dfdd057c1cacf364f8b25f87cc8e52c1a22185ffb9db5' \
  'bundle_sha256=876dce5e9445a5c29236689699719e53ebf79930afae75f8ad5ff21544664394' \
  'family_count=12' \
  'probe_count=13' \
  'mechanism_dimension_count=18' \
  'vertex_count=40' \
  'mincut_count=13' \
  'proc_treatment=CAPSULE_EMPTY_BIND' \
  'legacy_proc_absence=false' \
  'observation_types=REFUSED_BEFORE_EFFECT,CROSSED_NAMED_RULE,EFFECT_COMPLETED,EXPERIMENT_UNAVAILABLE' \
  'crossed_named_rule_counts_as_completion=false' \
  'experiment_unavailable_counts_as_coverage=false' \
  'vertex_hash_binding=invariant_sha256+delta_sha256+witness_sha256' \
  'refusal_monotone_under_mechanism_superset=true' \
  'completion_monotone_under_mechanism_subset=true' \
  'nonmonotone_decision=DENY457 nonmonotone-material-effect' \
  'family_1_repeat_exact_exec_mincuts=10' \
  'family_1_first_wrong_flags_exec_mincuts=01' \
  'family_3_mincuts=10,01' \
  'family_7_mincuts=10,01' \
  'family_8_mincuts=10,01' \
  'family_10_mincuts=10,01' \
  'family_11_mincuts=10,01' \
  'all_full_treatment_vertices=REFUSED_BEFORE_EFFECT' \
  'all_open_vertices=EFFECT_COMPLETED' \
  'all_effect_completions_have_typed_witness=true' \
  'all_vertices_triple_hash_slots_required=true' \
  'expected_results_source=Sounio' \
  'expected_results_encoded_in_shell=false' \
  'python_executable_invoked=false' \
  'rust_executable_invoked=false' \
  'native_v11_bytes_created=false' \
  'semantics_frozen=true' \
  'parity_open=false' \
  'material_hypercube=false' \
  'material_coverage=false' \
  'complete_effects=false' \
  'material_execution=false' \
  'action_9025_judged=false' \
  'launch_open=false' \
  'recycle_open=false' \
  'exec_attached=false' \
  'commit_attached=false' \
  'ci_attached=false' \
  'claim_ready=false' \
  'evidence_sha256=5191a0d46b4ef4086967ff90cc482efa673c65ff8764626245132e720cc2bc5d'; do
  require_line "$MANIFEST" "$line"
done

for line in \
  'schema=loom-process-witness-effect-policy-plan-v11-evidence-v1' \
  'stage=SOUNIO_EXECUTABLE' \
  'sounio_executable_commit=23b3756e1b40771e46f2e0a0ceebd1d5f7a8412b' \
  'v10_root_treatment=true' \
  'v10_bootstrap_sabotage=true' \
  'v10_bootstrap_negative_controls=7' \
  'v10_typed_proc_sabotages=4' \
  'v10_material_coverage=false' \
  'bundle_line_count=113' \
  'family_count=12' \
  'probe_count=13' \
  'mechanism_dimension_count=18' \
  'vertex_count=40' \
  'mincut_count=13' \
  'proc_treatment=CAPSULE_EMPTY_BIND' \
  'legacy_proc_absence=false' \
  'crossed_named_rule_counts_as_completion=false' \
  'experiment_unavailable_counts_as_coverage=false' \
  'vertex_hash_binding=invariant_sha256+delta_sha256+witness_sha256' \
  'all_full_treatment_vertices=REFUSED_BEFORE_EFFECT' \
  'all_open_vertices=EFFECT_COMPLETED' \
  'all_effect_completions_have_typed_witness=true' \
  'all_vertices_triple_hash_slots_required=true' \
  'expected_results_source=Sounio' \
  'native_v11_bytes_created=false' \
  'semantics_frozen=false' \
  'material_hypercube=false' \
  'material_coverage=false' \
  'complete_effects=false' \
  'material_execution=false'; do
  require_line "$EVIDENCE" "$line"
done

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v11_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v11-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V11 gate failed'
[[ "$result" == *'families=12 probes=13 mechanism_dimensions=18 vertices=40 mincuts=13 triple_hash_binding=true typed_observations=true crossed_is_not_completion=true unavailable_is_not_coverage=true full_treatments=REFUSED open_vertices=EFFECT_COMPLETED'* ]] ||
  fail 'Sounio V11 causal topology drifted'
[[ "$result" == *'proc_treatment=CAPSULE_EMPTY_BIND legacy_proc_absence=false'* ]] ||
  fail 'Sounio V11 proc correction drifted'
[[ "$result" == *'semantics_frozen=false parity_open=false material_hypercube=false material_coverage=false complete_effects=false material_execution=false action_9025_judged=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false' ]] ||
  fail 'Sounio V11 executable promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v11-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s families=12 probes=13 mechanism_dimensions=18 vertices=40 mincuts=13 triple_hash_binding=true proc_treatment=CAPSULE_EMPTY_BIND legacy_proc_absence=false semantics_frozen=true native_v11_bytes_created=false parity_open=false material_hypercube=false material_coverage=false complete_effects=false material_execution=false action_9025_judged=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
