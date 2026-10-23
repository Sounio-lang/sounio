#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusOperatorOrbitAdmissionReconstruction.lean'
AUDIT_REL='formal/lean4/SounioPireusOperatorOrbitAdmissionReconstructionAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
PARENT_RECEIPT_REL='tools/pireus/operator_orbit_class_reconstruction.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_operator_orbit_class_reconstruction_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/operator_orbit_class_reconstruction_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/operator_orbit_admission_reconstruction.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/operator_orbit_admission_reconstruction_v13.formal-parity.txt'

PARENT_COMMIT='7813940232094db36e1660c4fc3f3f00de72a5b1'
SOURCE_COMMIT='64d45df932b10264f73b71c64ead6997d985716e'
SOURCE_SHA256='1338b78be40e26666e562430ff9273e39c486bf098bae02d520021f67deab4af'
AUDIT_SHA256='15830f9bf2b51aeafb158369153ba73e07a6b5b3f0fd83e8d91c6d142aa3a34b'
LAKEFILE_SHA256='9fb4b0e7c311ceb4c6bc2ad07f6ccb42c992c47cd7d1520a135b7f1a5b190546'
OFFLOAD_LOG_SHA256='c9d4a2937497f9df3e20be9d200beed2e5c1527ab1ae04c8213cf4966f1f680b'
PARENT_RECEIPT_SHA256='df1b3d746353d02a3b769e80e1e92c334d0eb1efda76ce17408c07abf45aa97d'
PARENT_GATE_SHA256='711616993efdea67f47dea8362e8760d2a5d69b304ee11d74ea25620fda78c57'
PARENT_EVIDENCE_SHA256='3910e1a24d9b502018d757433c8b647184c00ce2f949e023fdf00b54df1752fd'
RECEIPT_SHA256='ab67493583ec87b45bc374a15a83363e746cd5ef8bb8aed72f87ca4bb32177e3'
EVIDENCE_SHA256='ba29ea873b10fca3c4a8d067dd2c0e2337489ee06ff84b0090be6b97d9ab8059'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='da7cf2273a3c4d120c899bc6b46cb3e531bac3ab77289e49d5ea2468a702dde2'
HARDWARE_SHA256='40d1dfd6cb2c1d3caf22220112051b6399125d0ccd97a24442dc7c3ef21e84f4'
BUILD_COMMAND_SHA256='fa183c18603e4e1ee06af56584ae346b59aed720c93d1c4db1a04335a57485a7'
AUDIT_COMMAND_SHA256='1eb04fcf883df7d08fb0150be6f326e71fdaf024be49952ffb049c4557ed9cfd'
BUILD_FRAME_SHA256='daece9e585a1c210c06037fbb0af943f85133bc54dfd6f236a427704b3468c38'
AUDIT_FRAME_SHA256='bb49736d996f750612f2b4e2ffc2d04d00edcf195f1ce031b97deb645469e034'
BUILD_OUTPUT_SHA256='830994b13229838e67d34d2dde777ab2a39b70e4302b255f11bc1d00e99ffb57'
OLEAN_SHA256='1d07f7dfcf13fb9740c206daea1808e5ff1ac5c142e23a545cf60300f8f1faf8'
AUDIT_OUTPUT_SHA256='3424cf3d21e8bf48b15a1079022fae910c96970d982c6e455befa1e402e5e3b5'
ZERO='0 0 0 0 0 0 0 0'
EXPECTED_THEOREMS=(
  concrete_admission_reconstruction_matches_declared_frozen_summary
  coefficient_toggle_trace_has_one_collapse_and_32_admissions
  every_admission_has_exact_noncollision_separators
  bounded_admission_census_does_not_prove_global_novelty
)

fail() {
  printf 'pireus orbit admission reconstruction formal parity receipt: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

require_hash() {
  local path="$1" expected="$2"
  [[ -f "${path}" ]] || fail "missing file: ${path}"
  [[ "$(sha_file "${path}")" == "${expected}" ]] || fail "hash drift: ${path}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" || fail "missing exact line in ${path}: ${expected}"
}

require_committed_hash() {
  local commit="$1" path="$2" expected="$3"
  [[ "$(git -C "${ROOT}" show "${commit}:${path}" | sha256sum | cut -d' ' -f1)" == "${expected}" ]] ||
    fail "committed hash drift: ${commit}:${path}"
}

sha_limbs() {
  local hex="$1" out='' i part
  [[ "${#hex}" -eq 64 && "${hex}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256: ${hex}"
  for ((i=0; i<8; i++)); do
    part="${hex:$((i*8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

parity_frame() {
  local source_sha="$1" command_sha="$2"
  printf '9020 4 4 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${source_sha}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_sha}")" "${ZERO}" "${ZERO}"
}

check_guardian_receipt() {
  local label="$1" frame="$2" expected_sha="$3" decision
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq 82 ]] || fail "Guardian frame words: ${label}"
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
    fail "Guardian decision drift: ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s process_launched=false recorded_k8s_process_exit=0\n' \
    "${label}" "${decision}"
}

"${ROOT}/scripts/ci/pireus_operator_orbit_canonicalization.sh" >/dev/null

require_hash "${ROOT}/${SOURCE_REL}" "${SOURCE_SHA256}"
require_hash "${ROOT}/${AUDIT_REL}" "${AUDIT_SHA256}"
require_hash "${ROOT}/${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_hash "${ROOT}/${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_COMMIT}" "${SOURCE_COMMIT}" ||
  fail 'parent class reconstruction does not precede admission reconstruction'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD ||
  fail 'admission reconstruction source commit is not in current history'
require_committed_hash "${PARENT_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${OFFLOAD_LOG_REL}" "${OFFLOAD_LOG_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "admission_reconstruction_source_commit=${SOURCE_COMMIT}"
require_line "${RECEIPT}" 'execution_route=KUBERNETES'
require_line "${RECEIPT}" 'kubernetes_node=r770-proxmox'
require_line "${RECEIPT}" 'hardware_model=Intel(R)_Xeon(R)_6730P'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" 'build_preexec_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${RECEIPT}" 'build_exit_code=0'
require_line "${RECEIPT}" "build_raw_output_sha256=${BUILD_OUTPUT_SHA256}"
require_line "${RECEIPT}" "build_olean_sha256=${OLEAN_SHA256}"
require_line "${RECEIPT}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_preexec_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${RECEIPT}" 'axiom_audit_exit_code=0'
require_line "${RECEIPT}" "axiom_audit_raw_output_sha256=${AUDIT_OUTPUT_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_theorem_reports=4'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=4'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=4'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'mutation_requests=3600'
require_line "${RECEIPT}" 'mutation_requests_derivation=16_PARENT_EPOCHS_TIMES_225_COEFFICIENTS'
require_line "${RECEIPT}" 'mutation_request_grid_parent_epochs=16'
require_line "${RECEIPT}" 'mutation_request_grid_coefficients_per_epoch=225'
require_line "${RECEIPT}" 'mutation_attempts=33'
require_line "${RECEIPT}" 'attempted_parent_epochs=0_ONLY'
require_line "${RECEIPT}" 'attempted_coefficients=0_THROUGH_32_INCLUSIVE'
require_line "${RECEIPT}" 'single_coefficient_toggle_checks=7425'
require_line "${RECEIPT}" 'single_coefficient_toggle_checks_derivation=33_EXECUTED_ATTEMPTS_TIMES_225_COEFFICIENT_COMPARISONS'
require_line "${RECEIPT}" 'single_coefficient_toggle_failures=0'
require_line "${RECEIPT}" 'anf_phase_transform_failures=0'
require_line "${RECEIPT}" 'equivalent_collapses=1'
require_line "${RECEIPT}" 'collapse_attempt_index=15'
require_line "${RECEIPT}" 'collapse_class_ids=31'
require_line "${RECEIPT}" 'admitted_classes=32'
require_line "${RECEIPT}" 'baseline_canonical_tables=30'
require_line "${RECEIPT}" 'final_canonical_tables=62'
require_line "${RECEIPT}" 'separator_certificates=1456'
require_line "${RECEIPT}" 'separator_failures=0'
require_line "${RECEIPT}" 'total_canonicalizations=161'
require_line "${RECEIPT}" 'total_canonicalizations_derivation=128_BASELINE_PLUS_33_MUTATION_ATTEMPTS'
require_line "${RECEIPT}" 'action_views_per_canonicalization=40320'
require_line "${RECEIPT}" 'total_action_views=6491520'
require_line "${RECEIPT}" 'total_action_views_derivation=161_CANONICALIZATIONS_TIMES_20160_GL4_F2_MATRICES_TIMES_2_INPUT_SWAP_VALUES'
require_line "${RECEIPT}" 'action_views_term_scope=ENUMERATED_MATRIX_SWAP_APPLICATORS_BEFORE_TREE_GAUGE_NORMALIZATION_NOT_EFFECTIVE_GROUP_ORDER'
require_line "${RECEIPT}" 'declared_action_group_scope=GL4_F2_X_INPUT_SWAP_WITH_TREE_GAUGE_NORMALIZATION_NOT_S8_PERMUTATION_ACTION'
require_line "${RECEIPT}" 'concrete_32_admission_reconstruction_complete=true'
require_line "${RECEIPT}" 'concrete_32_admission_reconstruction_scope=FINITE_FROZEN_SOUNIO_TRACE_REPLAY_NOT_ABSTRACT_ORBIT_CLASSIFICATION'
require_line "${RECEIPT}" 'bounded_declared_action_relative_noncollision_census_complete=true'
require_line "${RECEIPT}" 'canonical_representative_equality_iff_same_declared_orbit_proved=false'
require_line "${RECEIPT}" 'abstract_orbit_classes_proved=false'
require_line "${RECEIPT}" 'formal_parity_complete=false'
require_line "${RECEIPT}" 'global_novelty=false'
require_line "${RECEIPT}" 'claim_ready=false'
require_line "${RECEIPT}" 'python_processes_launched=0'
require_line "${RECEIPT}" 'rust_processes_launched=0'
require_line "${RECEIPT}" 'slurm_processes_launched=0'
require_line "${RECEIPT}" 'spark_route_policy=KUBERNETES_ONLY'
require_line "${RECEIPT}" 'spark_nodes_used=false'
require_line "${RECEIPT}" 'dgx_nodes_used=false'
require_line "${RECEIPT}" 'u250_declared_card_count=2'
require_line "${RECEIPT}" 'u250_installed_card_count=1'
require_line "${RECEIPT}" 'u250_pending_installation_card_count=1'
require_line "${RECEIPT}" 'u250_pending_reason=SECOND_CARD_NOT_YET_INSTALLED'
require_line "${RECEIPT}" 'u250_enumeration_failure_count=0'
require_line "${RECEIPT}" 'llm_role=REVIEW_ONLY'
require_line "${RECEIPT}" 'llm_confirmed_result=false'
require_line "${RECEIPT}" 'llm_second_opinion_available=false'
require_line "${RECEIPT}" 'llm_second_opinion_failure=ZAI_PROVIDER_ERROR_1313'

require_line "${EVIDENCE}" 'collapse_attempt=15'
require_line "${EVIDENCE}" 'collapse_class_id=31'
require_line "${EVIDENCE}" 'separator_arithmetic=32*30+32*31/2=1456'
require_line "${EVIDENCE}" 'concrete_32_admission_reconstruction_complete=true'
require_line "${EVIDENCE}" 'concrete_32_admission_reconstruction_scope=FINITE_FROZEN_SOUNIO_TRACE_REPLAY_NOT_ABSTRACT_ORBIT_CLASSIFICATION'
require_line "${EVIDENCE}" 'u250_pending_reason=SECOND_CARD_NOT_YET_INSTALLED'
require_line "${EVIDENCE}" 'u250_enumeration_failure_count=0'

[[ "$(grep -c '^theorem ' "${ROOT}/${SOURCE_REL}")" -eq 4 ]] || fail 'theorem count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 4 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
require_line "${ROOT}/${SOURCE_REL}" '    admissionReconstructionSummary = frozenAdmissionReconstructionSummary := by'
require_line "${ROOT}/${SOURCE_REL}" '      admissionReconstructionSummary.collapseAttempts = [15] ∧'
require_line "${ROOT}/${SOURCE_REL}" '      !admissionReconstructionSummary.globalNoveltyProved &&'
if grep -Eq '\bsorry\b|sorryAx' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry or sorryAx found in admission reconstruction proof surface'
fi

(( 15 * 15 == 225 )) || fail 'interior arithmetic drift'
(( 33 * 225 == 7425 )) || fail 'mutation-check arithmetic drift'
(( 30 + 32 == 62 )) || fail 'class-growth arithmetic drift'
(( 32 * 30 + 32 * 31 / 2 == 1456 )) || fail 'separator arithmetic drift'
(( 128 + 33 == 161 )) || fail 'canonicalization arithmetic drift'
(( 161 * 40320 == 6491520 )) || fail 'action-view arithmetic drift'
(( 2 == 1 + 1 )) || fail 'U250 inventory arithmetic drift'

check_guardian_receipt K8S_BUILD_RECEIPT \
  "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
check_guardian_receipt K8S_AXIOM_AUDIT_RECEIPT \
  "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"

printf '%s\n' \
  'PIREUS_OPERATOR_ORBIT_ADMISSION_RECONSTRUCTION_FORMAL_PARITY_RECEIPT_PASS=true status=PARTIAL_PASS verification=KUBERNETES_EXECUTION_RECEIPT_REPLAY fresh_lean_execution=false language=Lean4 role=FORMAL_PARITY mutation_requests=3600 request_grid_parent_epochs=16 request_grid_coefficients_per_epoch=225 mutation_attempts=33 attempted_parent_epochs=0_ONLY attempted_coefficients=0_THROUGH_32 collapse_attempt=15 collapse_class=31 collapses=1 admissions=32 baseline_canonical_tables=30 final_canonical_tables=62 separator_certificates=1456 separator_failures=0 mutation_checks=7425 mutation_failures=0 transform_failures=0 total_canonicalizations=161 action_views=6491520 action_views_derivation=161_TIMES_20160_TIMES_2 action_views_kind=ENUMERATED_MATRIX_SWAP_APPLICATORS action_group=GL4_F2_X_INPUT_SWAP_NOT_S8 concrete_frozen_trace_reconstruction=true concrete_reconstruction_scope=FINITE_FROZEN_SOUNIO_TRACE_NOT_ABSTRACT_ORBIT_CLASSIFICATION abstract_orbit_classes_proved=false canonical_iff_orbit=false global_novelty=false formal_parity_complete=false semantic_authority=Sounio expected_results_supplied_by_lean=false spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 claim_ready=false'
