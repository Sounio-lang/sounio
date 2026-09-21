#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v4.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v4-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v4-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V4.md \
  a0339f8cfc3e070db19e86fe29cd301a3e09f0bc5883e85e8dab7eaf21a87744
expect_hash tools/loom/process_witness_effect_policy_plan_v4_main.sio \
  d0e8db991e56952ed950bebf03b20823867004fe75ae3e7cd9710f1d35df0222
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v4.sh \
  e484581a772973196f505f1047b0cd6d8033b47cd858e40a1ea9137d20c12fe8
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v4_selftest.sh \
  fb6f5c933be678912658fb45633dc158ef594ed42b16f60163f63bca464e639e
expect_hash tools/loom/process_witness_effect_policy_plan_v3.freeze.v1 \
  40407323594e37d44b9002d1cdd390677416048221ace446693919f8415ca480
expect_hash tools/loom/evidence/loom-process-witness-effect-root-v3-host-attempt-v1-20260828.txt \
  baeb296039daf112f66d22e7ad7f57e2a605702a964b444dc8a8a1c6325c37e5
expect_hash "$EVIDENCE" \
  59ea1c0febd2ac2b472b69483bcd8e8ffcebd0be9f16a20946c1cb9048453075

for line in \
  'schema=loom-process-witness-effect-policy-plan-v4-freeze-v1' \
  'stage=SEMANTICS_FROZEN' \
  'producing_language=Sounio' \
  'language_role=SEMANTIC_POLICY_PLAN' \
  'semantic_authority=Sounio' \
  'action=9025' \
  'garden_commit=6baa8508e9643c175e346a2038d00d0d0f246a14' \
  'sounio_executable_commit=0bd4915460954d558c687c403ffd3850cda1e9a8' \
  'executable_sha256=4ba84e085797d3bb7ee95c066fba51dd1efd2e77225a9816ac2450af2a5e0a4d' \
  'bundle_sha256=3bce80f8d74098470566b3ce3c0b872992ac0cf1d42ce9c64df4bc06ae57901f' \
  'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE' \
  'systemd_mount_path=/run/systemd/incoming' \
  'systemd_mount_source=/run/systemd/propagate/EXACT_UNIT' \
  'systemd_mount_principal_writable=false' \
  'systemd_mount_ready_contents=empty' \
  'systemd_version=257' \
  'bootstrap_case_count=2' \
  'bootstrap_treatment_code=0' \
  'bootstrap_missing_code=226' \
  'allowed_syscalls=0,1,60,322' \
  'family_10_probe=personality_change' \
  'v3_materializable=false' \
  'v4_required_for_native=true' \
  'expected_results_encoded_in_shell=false' \
  'python_executable_invoked=false' \
  'rust_executable_invoked=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'material_coverage=false' \
  'complete_effects=false' \
  'material_execution=false' \
  'launch_open=false' \
  'recycle_open=false' \
  'exec_attached=false' \
  'commit_attached=false' \
  'ci_attached=false' \
  'parity_open=false' \
  'claim_ready=false' \
  'evidence_sha256=59ea1c0febd2ac2b472b69483bcd8e8ffcebd0be9f16a20946c1cb9048453075'; do
  require_line "$MANIFEST" "$line"
done

for line in \
  'hardware_arch=x86_64' \
  'hardware_kernel=Linux_7.0.2-5-pve' \
  'systemd_mount_path=/run/systemd/incoming' \
  'bootstrap_treatment_decision=SOUNIO_ROOT_BOOTSTRAP_ALLOW code=0 reason=systemd-incoming-present stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=namespace-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'v3_native_consumption=refused' \
  'native_executable_invoked=false' \
  'expected_results_source=Sounio' \
  'expected_results_encoded_in_shell=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'complete_effects=false' \
  'material_execution=false'; do
  require_line "$EVIDENCE" "$line"
done

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v4_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v4-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V4 gate failed'
[[ "$result" == *'families=12 treatments=12 sabotages=12 bootstrap_cases=2 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change'* ]] ||
  fail 'Sounio V4 bootstrap, surface, or decision classes drifted'
[[ "$result" == *'root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V4 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v4-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s systemd_mount=/run/systemd/incoming bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
