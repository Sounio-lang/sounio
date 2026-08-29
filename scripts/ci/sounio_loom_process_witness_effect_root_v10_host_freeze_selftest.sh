#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v10-host-attempt-v1-20260828.txt"

fail() {
  printf 'sounio-loom-process-witness-effect-root-v10-host-freeze-selftest: FAIL reason=%s\n' "$*" >&2
  exit 1
}

field() {
  local key="$1" value
  value="$(grep -E "^${key}=" "$EVIDENCE" || true)"
  [[ -n "$value" && "$value" != *$'\n'* ]] || fail "evidence field is absent or repeated: $key"
  printf '%s' "${value#*=}"
}

require_value() {
  local key="$1" expected="$2" observed
  observed="$(field "$key")"
  [[ "$observed" == "$expected" ]] ||
    fail "evidence field drifted: $key observed=$observed expected=$expected"
}

require_hash() {
  local path="$1" key="$2" expected observed
  [[ -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" ]] ||
    fail "hashed input is absent or linked: $path"
  expected="$(field "$key")"
  observed="$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)"
  [[ "$expected" == "$observed" ]] ||
    fail "hashed input drifted: $path observed=$observed expected=$expected"
}

[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'V10 host evidence is absent or linked'
require_value schema loom-process-witness-effect-root-v10-host-attempt-v1
require_value stage MATERIAL_TREATMENT_AND_CAUSAL_CONTROLS_PASSED
require_value semantic_authority Sounio
require_value language_role MATERIAL_PARITY
require_value action 9025
require_value v10_policy_manifest_sha256 9e7f42fd4bd18fd2b5f996b279a67f46a50546a20ef6949e4dc069c16b3d0dda
require_value namespace_mutator_role MATERIAL_PARITY
require_value namespace_mutator_semantic_decision false
require_value namespace_mutator_static true
require_value namespace_mutator_set_id false
require_value typed_structural_mounts /proc:CAPSULE_EMPTY_BIND
require_value proc_mount_count 1
require_value proc_mount_source_identity device+inode
require_value proc_mount_filesystem CAPSULE_ROOT_FILESYSTEM
require_value proc_mount_root_owned true
require_value proc_mount_contents empty
require_value proc_mount_vfs_read_only true
require_value proc_mount_principal_writable false
require_value procfs_visible false
require_value forbidden_mounts_observed none
require_value bootstrap_missing_incoming DENY226_NAMESPACE
require_value bootstrap_missing_sys DENY226_NAMESPACE
require_value bootstrap_missing_var_tmp DENY226_NAMESPACE
require_value bootstrap_live_procfs DENY453
require_value bootstrap_wrong_proc_source DENY454
require_value bootstrap_writable_proc_bind DENY455
require_value bootstrap_nonempty_proc_bind DENY456
require_value typed_proc_sabotages 4
require_value bootstrap_negative_controls 7
require_value expected_results_source Sounio
require_value expected_results_encoded_in_shell false
require_value python_executable_invoked false
require_value rust_executable_invoked false
require_value root_treatment true
require_value bootstrap_sabotage true
require_value material_sabotages 0
for closed_flag in material_coverage complete_effects material_execution \
                   production_activation launch_open recycle_open exec_attached \
                   commit_attached ci_attached parity_open claim_ready; do
  require_value "$closed_flag" false
done

require_hash tools/loom/process_witness_effect_policy_plan_v10.freeze.v1 v10_policy_manifest_sha256
require_hash tools/loom/src/loom_process_witness_effect_policy_v3.cpp native_shared_source_sha256
require_hash tools/loom/src/loom_process_witness_effect_policy_v10.cpp native_v10_source_sha256
require_hash scripts/dev/build_loom_process_witness_effect_policy_v10.sh native_builder_sha256
require_hash scripts/ci/sounio_loom_process_witness_effect_policy_v10_selftest.sh native_selftest_sha256
require_hash scripts/dev/build_loom_process_witness_effect_root_v10.sh root_builder_sha256
require_hash scripts/ci/sounio_loom_process_witness_effect_root_v10_selftest.sh root_selftest_sha256
require_hash tools/loom/src/loom_mount_namespace_mutator.cpp namespace_mutator_source_sha256
require_hash scripts/dev/build_loom_mount_namespace_mutator.sh namespace_mutator_builder_sha256
require_hash scripts/ci/sounio_loom_mount_namespace_mutator_selftest.sh namespace_mutator_selftest_sha256
require_hash scripts/ci/sounio_loom_process_witness_effect_root_v10_host_gate.sh host_gate_sha256
require_hash scripts/dev/run_loom_process_witness_effect_root_v10_host_probe.sh host_probe_sha256

materializer_commit="$(field materializer_commit)"
[[ "$materializer_commit" =~ ^[0-9a-f]{40}$ ]] || fail 'materializer commit is malformed'
git -C "$ROOT_DIR" cat-file -e "$materializer_commit^{commit}" 2>/dev/null ||
  fail 'materializer commit is absent'

"$ROOT_DIR/scripts/ci/sounio_loom_process_witness_effect_policy_plan_v10_freeze_selftest.sh" >/dev/null
"$ROOT_DIR/scripts/ci/sounio_loom_mount_namespace_mutator_selftest.sh" >/dev/null
"$ROOT_DIR/scripts/ci/sounio_loom_process_witness_effect_policy_v10_selftest.sh" >/dev/null
"$ROOT_DIR/scripts/ci/sounio_loom_process_witness_effect_root_v10_selftest.sh" >/dev/null

printf 'sounio-loom-process-witness-effect-root-v10-host-freeze-selftest: PASS semantic_authority=Sounio action=9025 materializer_commit=%s policy_manifest_sha256=%s evidence_sha256=%s cell_sha256=%s tree_sha256=%s namespace_mutator_sha256=%s root_treatment=true bootstrap_sabotage=true bootstrap_negative_controls=7 typed_proc_sabotages=4 material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false parity_open=false claim_ready=false\n' \
  "$materializer_commit" "$(field v10_policy_manifest_sha256)" \
  "$(sha256sum "$EVIDENCE" | cut -d ' ' -f 1)" "$(field cell_sha256)" \
  "$(field tree_sha256)" "$(field namespace_mutator_binary_sha256)"
