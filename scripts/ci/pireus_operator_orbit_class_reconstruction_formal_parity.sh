#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusOperatorOrbitClassReconstruction.lean'
AUDIT_REL='formal/lean4/SounioPireusOperatorOrbitClassReconstructionAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
PARENT_RECEIPT_REL='tools/pireus/operator_orbit_archive_reconstruction.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_operator_orbit_archive_reconstruction_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/operator_orbit_archive_reconstruction_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/operator_orbit_class_reconstruction.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/operator_orbit_class_reconstruction_v13.formal-parity.txt'

PARENT_COMMIT='e07d1cbc464ade663de5a84be5982b61a6de03ee'
SOURCE_COMMIT='a7cc707e6d6ade58a305f51128eeb9bfbfd9f373'
SOURCE_SHA256='32a85dd0044597ca890dacb75a5d80766e5294441298f77fde2df00c4550b0ce'
AUDIT_SHA256='8f2fafbc7aeb2171d8f3320ed55df7f48eb336c9269bcb7e311ad181c652719b'
LAKEFILE_SHA256='80d9f9d166d362a5a19e472263111f54f44fe063eee903e6b7dfa5e0ba31bb88'
PARENT_RECEIPT_SHA256='98f736dcfc947e5669d8c2a1329afc3f4d12fa16ae07d0631152fe5e115e962b'
PARENT_GATE_SHA256='129f9db7e44e1f9c92e70a217642baa546f0b88bddff1c8ea6c366fd84585919'
PARENT_EVIDENCE_SHA256='570f61d7515e7a068f21a38eee1ac7235499fff37eba5ca12a415a8f47303dfb'
RECEIPT_SHA256='df1b3d746353d02a3b769e80e1e92c334d0eb1efda76ce17408c07abf45aa97d'
EVIDENCE_SHA256='3910e1a24d9b502018d757433c8b647184c00ce2f949e023fdf00b54df1752fd'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='da7cf2273a3c4d120c899bc6b46cb3e531bac3ab77289e49d5ea2468a702dde2'
HARDWARE_SHA256='40d1dfd6cb2c1d3caf22220112051b6399125d0ccd97a24442dc7c3ef21e84f4'
BUILD_COMMAND_SHA256='ad25bbbe12830745a1993815cdda878bab2a50be6955ae558321e97c6de906b4'
AUDIT_COMMAND_SHA256='77957831e8c7b8b99148c8ff513406b6d05f2179b6906cedb0521e54ebab2c13'
BUILD_FRAME_SHA256='30fde3699781abad0c2f979140fb7ae29a036945df5100a8009c5a932da43bb4'
AUDIT_FRAME_SHA256='21cba12160ae89f53b0429928e19a89a20477367fb8cd7c85ca82c6abb8ba411'
BUILD_OUTPUT_SHA256='70f5237ec1987b245cbc7776f14c380599b946679617f4a8835039e105b427d9'
OLEAN_SHA256='4b29dbe435f7cf4b9784b53d878d7b3653196f1090acbdf2d1fb733491d385be'
AUDIT_OUTPUT_SHA256='0c4a613471c5c8eaa2f6ca1ebbafef3349e19e2146dbfd93b3a1cc5c43166600'
ZERO='0 0 0 0 0 0 0 0'
EXPECTED_THEOREMS=(
  concrete_class_reconstruction_matches_declared_frozen_summary
  reconstructed_128_image_archive_has_exactly_30_canonical_tables
  every_reconstructed_image_maps_to_one_of_30_distinct_classes
  class_census_does_not_yet_prove_canonical_iff_orbit
)

fail() {
  printf 'pireus orbit class reconstruction formal parity receipt: FAIL: %s\n' "$*" >&2
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
  fail 'parent archive reconstruction does not precede class reconstruction'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD ||
  fail 'class reconstruction source commit is not in current history'
require_committed_hash "${PARENT_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "class_reconstruction_source_commit=${SOURCE_COMMIT}"
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
require_line "${RECEIPT}" 'matrix_codes_scanned=65536'
require_line "${RECEIPT}" 'invertible_matrices=20160'
require_line "${RECEIPT}" 'action_views_per_image=40320'
require_line "${RECEIPT}" 'action_views_term_scope=ENUMERATED_MATRIX_SWAP_APPLICATORS_BEFORE_TREE_GAUGE_NORMALIZATION_NOT_EFFECTIVE_GROUP_ORDER'
require_line "${RECEIPT}" 'archive_images=128'
require_line "${RECEIPT}" 'canonicalizations=128'
require_line "${RECEIPT}" 'canonical_class_count=30'
require_line "${RECEIPT}" 'lean_field_canonical_classes_distinct=true'
require_line "${RECEIPT}" 'canonical_table_distinctness_scope=PAIRWISE_DISTINCT_ARRAY_VALUES_IN_DEDUPLICATED_30_TABLE_OUTPUT'
require_line "${RECEIPT}" 'lean_field_class_membership_complete=true'
require_line "${RECEIPT}" 'canonical_table_membership_scope=EVERY_OF_128_COMPUTED_CANONICAL_TABLES_OCCURS_IN_DEDUPLICATED_30_TABLE_OUTPUT'
require_line "${RECEIPT}" 'lean_field_concrete_30_class_census_complete=true'
require_line "${RECEIPT}" 'finite_128_image_to_30_canonical_table_census_complete=true'
require_line "${RECEIPT}" 'class_term_scope=DEDUPLICATED_CANONICAL_TABLES_UNDER_EXECUTED_NORMALIZER_NOT_PROVED_ABSTRACT_ORBIT_EQUIVALENCE_CLASSES'
require_line "${RECEIPT}" 'declared_action_group_scope=GL4_F2_X_INPUT_SWAP_WITH_TREE_GAUGE_NORMALIZATION_NOT_S8_PERMUTATION_ACTION'
require_line "${RECEIPT}" 'action_view_40320_s8_inference=false'
require_line "${RECEIPT}" 'source_optimization_claim_scope=REPRESENTATION_ONLY_NO_ASYMPTOTIC_OR_CORRECTNESS_CLAIM'
require_line "${RECEIPT}" 'canonical_representative_equality_iff_same_declared_orbit_proved=false'
require_line "${RECEIPT}" 'concrete_32_admission_reconstruction_proved=false'
require_line "${RECEIPT}" 'formal_parity_complete=false'
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
require_line "${RECEIPT}" 'llm_role=REVIEW_ONLY'
require_line "${RECEIPT}" 'llm_confirmed_result=false'
require_line "${RECEIPT}" 'llm_second_opinion_available=false'
require_line "${RECEIPT}" 'llm_second_opinion_failure=ZAI_PROVIDER_ERROR_1313'

[[ "$(grep -c '^theorem ' "${ROOT}/${SOURCE_REL}")" -eq 4 ]] || fail 'theorem count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 4 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
require_line "${ROOT}/${SOURCE_REL}" '    classReconstructionSummary = frozenClassReconstructionSummary ∧'
require_line "${ROOT}/${SOURCE_REL}" '      canonicalArchive.length = 128 ∧ canonicalClasses.length = 30 := by'
if grep -Eq '\bsorry\b|sorryAx' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry or sorryAx found in class reconstruction proof surface'
fi

(( 2 ** 16 == 65536 )) || fail 'matrix-code arithmetic drift'
(( 20160 * 2 == 40320 )) || fail 'action-view arithmetic drift'
(( 30 < 128 )) || fail 'class quotient arithmetic drift'
(( 2 == 1 + 1 )) || fail 'U250 inventory arithmetic drift'

check_guardian_receipt K8S_BUILD_RECEIPT \
  "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
check_guardian_receipt K8S_AXIOM_AUDIT_RECEIPT \
  "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"

printf '%s\n' \
  'PIREUS_OPERATOR_ORBIT_CLASS_RECONSTRUCTION_FORMAL_PARITY_RECEIPT_PASS=true status=PARTIAL_PASS verification=KUBERNETES_EXECUTION_RECEIPT_REPLAY fresh_lean_execution=false language=Lean4 role=FORMAL_PARITY matrix_codes=65536 invertible_matrices=20160 action_views_per_image=40320 action_views_kind=ENUMERATED_MATRIX_SWAP_APPLICATORS action_group=GL4_F2_X_INPUT_SWAP_NOT_S8 archive_images=128 canonical_tables=30 canonical_table_membership=128_IN_DEDUPLICATED_30_OUTPUT abstract_orbit_classes_proved=false theorem_reports=4 propext_mentions=4 native_decide_mentions=4 classical_choice_mentions=0 quot_sound_mentions=0 sorryax_mentions=0 finite_128_image_to_30_canonical_table_census_complete=true canonical_iff_orbit=false concrete_32_admission_reconstruction=false formal_parity_complete=false semantic_authority=Sounio expected_results_supplied_by_lean=false spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false u250_declared=2 u250_installed=1 u250_pending_installation=1 claim_ready=false'
