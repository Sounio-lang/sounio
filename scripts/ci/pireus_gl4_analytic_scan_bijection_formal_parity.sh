#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

EMBEDDING_REL='formal/lean4/SounioPireusGL4AnalyticScanEmbedding.lean'
ENCODER_REL='formal/lean4/SounioPireusGL4AnalyticBasisEncoder.lean'
BIJECTION_REL='formal/lean4/SounioPireusGL4AnalyticScanBijection.lean'
ACTION_REL='formal/lean4/SounioPireusGL4AnalyticActionCensus.lean'
EMBEDDING_AUDIT_REL='formal/lean4/SounioPireusGL4AnalyticScanEmbeddingAxiomAudit.lean'
ENCODER_AUDIT_REL='formal/lean4/SounioPireusGL4AnalyticBasisEncoderAxiomAudit.lean'
AGGREGATE_AUDIT_REL='formal/lean4/SounioPireusGL4AnalyticScanBijectionAxiomAudit.lean'
ACTION_AUDIT_REL='formal/lean4/SounioPireusGL4AnalyticActionCensusAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
BASE_GATE_REL='scripts/ci/pireus_operator_orbit_canonicalization.sh'
BASE_FREEZE_REL='tools/pireus/operator_orbit_canonicalization.freeze.v13'
PARENT_RECEIPT_REL='tools/pireus/gl4_analytic_census.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_gl4_analytic_census_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/gl4_analytic_census_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/gl4_analytic_scan_bijection.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/gl4_analytic_scan_bijection_v13.formal-parity.txt'

PARENT_GATE_COMMIT='df2d1b3e8aa12d53a9d5e8edbbc86dc0efd7bb4e'
SOURCE_IMPLEMENTATION_COMMIT='ed2be897f45ba8a09392be019a1c2a5f71bbdd49'
AUDIT_COMPLETION_COMMIT='c7e30abe46caa510d900a8765d1f793f3d28bba0'
ARTIFACT_COMMIT='6da6971ae91e56ccdb5fc8a6a2d27125b89f4c9a'
SOURCE_BUNDLE_SHA256='cc840b176d55da5d61174267874f3a5d9b215c08e5e5b3473b710adbeb4ea22e'
AUDIT_BUNDLE_SHA256='3ab9fabe97e898bb09231e606a2fe59933483ed3c5e3d5817a74b3a6e23162a1'
EMBEDDING_SHA256='14a94f98ecbaef7ca9ab81908204fcb9daa472c54189451afa0e282cb1cefcf1'
ENCODER_SHA256='8c11b018505e91f468a882b361d8e7b1cba3775becaea836a37c0ff6a9e9ba98'
BIJECTION_SHA256='819af488425094489ae01501a4382fafcb29a137a2118ce72e74762fba58cf9a'
ACTION_SHA256='475cd3707ac86b2d4aac43a3aa89201ed06f36d7544dcc2b74cf47739fa2cf75'
EMBEDDING_AUDIT_SHA256='3f98ba10985e9d0db0c5515cda4294cc10757cd3372bdc1498ce9096fe509594'
ENCODER_AUDIT_SHA256='a23d8d8a3d3e4acc219ddb99254277cf7d0a83f6d4c60848904b681b07a3daf4'
AGGREGATE_AUDIT_SHA256='4d9b76a51f0a8180455f88df38ea8b3dc58362d3215c21df5f36168e82823f10'
ACTION_AUDIT_SHA256='015b9d2df4ca1e093748c6fc1d179486fb9c69e323e4ed2dbdc0be185f3d6a3c'
LAKEFILE_SHA256='7c48f8fe0b007a863f878921297328011effd07c4b7351f47f203f2eddeb4e7e'
SOURCE_OFFLOAD_LOG_SHA256='40543d22371016d3b05984b14a1aa626176933f8afa8ecbad64ebd4e26658684'
AUDIT_OFFLOAD_LOG_SHA256='daa30aee5dd33b0587927d44b6663e36ed2e997d62991c51f8130243c38da90c'
ARTIFACT_OFFLOAD_LOG_SHA256='6173ebfc11efbd27d3b4b55d2993e30c354cecc6c8a733c93b5d6aabbd6c3748'
GATE_OFFLOAD_LOG_SHA256='2b9162d74077e3a86102c6d9cabccb5ff16e5899e5c41f8f629b57ccfda159b4'
PARENT_RECEIPT_SHA256='5643e9dac169f08017be363d8b59f880205a2dbbf65d4cb4521b9d1a39f73430'
PARENT_GATE_SHA256='666254cce258d005fa9b51399fcd3b622c6659ccf056578f2a3f37adc9a4dae3'
PARENT_EVIDENCE_SHA256='ccd2f0f7c7cca061774db0667039a89742ee66d0bac8d4f5691b98c2b9a7a203'
RECEIPT_SHA256='36c23734c1a05ddea1716de5d0013cb795aeaf20ebb811848a7d6699211e6b3e'
EVIDENCE_SHA256='5363ce3100b5429c7a8d7c9dd9dba14a836f2af10d1f1e4f5823e521ea250218'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
BASE_GATE_SHA256='6a18d7061bd408a3050d468d65c53231d0010865543346352e7ae91a0ff11f0e'
BASE_FREEZE_SHA256='11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84'
BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusGL4AnalyticActionCensus'
EMBEDDING_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4AnalyticScanEmbedding.lean'
ENCODER_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4AnalyticBasisEncoder.lean'
BIJECTION_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4AnalyticScanBijection.lean'
ACTION_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4AnalyticActionCensus.lean'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4AnalyticScanBijectionAxiomAudit.lean'
BUILD_COMMAND_SHA256='9302d4f859aa58c292b61346903798e330672d3361730303533961ec988af43a'
EMBEDDING_COMMAND_SHA256='f37888ba9ad920b9a387f03bb418144dfeaebc0e448fe3beffbb9718daa705ab'
ENCODER_COMMAND_SHA256='8b9806171c5c0a45460fc78dd4e495f9eeda92ca556741dc43d235846b79f208'
BIJECTION_COMMAND_SHA256='4a183f78ea53b0ea7cda4648b9515e7bb1e020b17907d082dfb60833baac482a'
ACTION_COMMAND_SHA256='4f8315df1b7b16f571c2db398b9a42940d0e762873126f83e1d74e78c26c3b3d'
AUDIT_COMMAND_SHA256='14592f429604ec074f4d7e2c2cd690decef05653ef10f932678f7a48c26754d5'
BUILD_FRAME_SHA256='c904b6a047ff5ac2e4a30233bfc5d60efeda46228bde29ac24e161b9230fd420'
EMBEDDING_FRAME_SHA256='90d90302311a422d4bb4556f4066211b269f35d670c2c7d217ded4f460ca483b'
ENCODER_FRAME_SHA256='1c8ae727aee0e196ce5e5ac478370672b6b97d4d842678dd2fbcbf455d0586a3'
BIJECTION_FRAME_SHA256='862577a9d6105ac6cd1115bb5e605ff8092405071cadbd0fd5a82a5386ffe6dd'
ACTION_FRAME_SHA256='731e84474699463bf395836e4f1b0f390e9689255d2c3f9f05e18c48b9392541'
AUDIT_FRAME_SHA256='a6fea5af1ca30786701c5217fb46e00ee715957bf98f39cf8b0d023165ac259b'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
PYTHON_COMMAND_SHA256='92ad024a5b29367eeccdf93e8171f280baaa3c62bf46cc190ddd22ae8ad8cfc5'
ZERO='0 0 0 0 0 0 0 0'

EXPECTED_THEOREMS=(
  mem_eraseMany_of_mem_of_not_mem mem_choicesOutside_of_not_mem_span map_spanZero
  map_spanOne map_spanTwo map_spanThree image_not_mem_map_of_injective
  standard_first_outside standard_second_outside standard_third_outside
  standard_fourth_outside basis_of_scan_entry_first_mem
  basis_of_scan_entry_second_mem basis_of_scan_entry_third_mem
  basis_of_scan_entry_fourth_mem basis_of_scan_entry_mem_analytic_ordered_bases
  encode_row_bits parity4_and_lane1 parity4_and_lane2 parity4_and_lane4
  parity4_and_lane8 matrix_parity_lane1 matrix_parity_lane2 matrix_parity_lane4
  matrix_parity_lane8 encode_f2_word4_eq_iff
  matrix_parities_equal_of_lane_image_equal matrix_row_eq_of_basis_parities
  matrix_rows_determine_bounded_code basis_of_scan_entry_injective
  encoded_row_lt_sixteen encoded_row_components matrix_code_of_basis_lt_matrix_codes
  matrix_row_code_of_basis_zero matrix_row_code_of_basis_one
  matrix_row_code_of_basis_two matrix_row_code_of_basis_three
  matrix_parity_code_of_basis_lane1 matrix_parity_code_of_basis_lane2
  matrix_parity_code_of_basis_lane4 matrix_parity_code_of_basis_lane8
  matrix_lane_map_code_of_basis_lane1 matrix_lane_map_code_of_basis_lane2
  matrix_lane_map_code_of_basis_lane4 matrix_lane_map_code_of_basis_lane8
  span_extend_closed spanThree_closed spanFour_nodup_of_outside map_spanFour
  standard_span_four_is_lane_universe analytic_ordered_basis_membership_facts
  analytic_basis_span_four_nodup encoded_basis_lane_universe_image_nodup
  injective_of_lane_universe_map_nodup matrix_lane_map_code_of_analytic_basis_injective
  eraseDups_eq_self_of_nodup_nat matrix_images_code_of_analytic_basis_nodup
  matrix_code_of_analytic_basis_invertible basis_of_scan_entry_of_analytic_basis
  scan_entry_of_basis_of_scan_entry analytic_scan_to_basis_injective
  analytic_basis_to_scan_injective eraseMany_nodup choicesOutside_nodup
  nodup_flatMap_of_tagged_fibers fourth_completions_nodup
  fourth_completions_tagged third_completions_nodup third_completions_tagged
  second_completions_nodup second_completions_tagged analytic_ordered_bases_nodup
  nodup_of_map_nodup frozen_scan_entries_nodup analytic_basis_entries_nodup
  frozen_scan_length_eq_analytic_ordered_basis_length
  frozen_scan_census_is_20160_analytically
  analytic_concrete_action_list_length_is_40320 action_view_fiber_nodup
  analytic_action_views_nodup action_of_view_injective
  analytic_concrete_action_list_nodup
  any_enumeration_containing_declared_actions_has_at_least_40320
  scan_basis_extraction_is_partial_not_target03
)

EXPECTED_DEFINITIONS=(
  basisOfScanEntry rowBit basisFirst basisSecond basisThird basisFourth encodedRow
  encodedRowLane matrixCodeOfBasis boundedMatrixCodeOfBasis spanFour AnalyticBasisEntry
  matrixWitnessOfAnalyticBasis scanEntryOfAnalyticBasis AnalyticScanBijection
  analyticScanEquiv frozenScanEntries analyticBasisEntries analyticActionViews
  actionOfView analyticConcreteActionList GL4AnalyticScanBijectionBoundary
  gl4AnalyticScanBijectionBoundary
)

SOURCE_FILES=(
  "${ROOT}/${EMBEDDING_REL}" "${ROOT}/${ENCODER_REL}"
  "${ROOT}/${BIJECTION_REL}" "${ROOT}/${ACTION_REL}"
)

fail() {
  printf 'pireus GL4 analytic scan bijection formal parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }
count_occurrences() { grep -c -- "$1" <<<"$2" || true; }

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

python_oracle_frame() {
  printf '9020 4 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_BUNDLE_SHA256}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${PYTHON_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" decision
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq 82 ]] || fail "Guardian frame words: ${label}"
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
    fail "Guardian decision drift: ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
}

require_hash "${ROOT}/${BASE_GATE_REL}" "${BASE_GATE_SHA256}"
require_hash "${ROOT}/${BASE_FREEZE_REL}" "${BASE_FREEZE_SHA256}"
base_semantics_sha256="$(
  sed -n '/^semantics_material_begin$/,/^semantics_material_end$/p' "${ROOT}/${BASE_FREEZE_REL}" |
    sed '1d;$d' |
    sha256sum |
    cut -d' ' -f1
)"
[[ "${base_semantics_sha256}" == "${SOUNIO_SEMANTICS_SHA256}" ]] ||
  fail 'live Sounio base semantics digest drift'
base_output="$("${ROOT}/${BASE_GATE_REL}")"
grep -Fq 'PIREUS_OPERATOR_ORBIT_CANONICALIZATION_GATE_PASS=true' <<<"${base_output}" ||
  fail 'Sounio semantic-authority gate did not pass'
grep -Fq ' stage=SEMANTICS_FROZEN ' <<<"${base_output}" || fail 'Sounio base stage drift'
grep -Fq ' language=Sounio role=SEMANTIC_AUTHORITY ' <<<"${base_output}" || fail 'semantic authority drift'
grep -Fq ' python_dispatch=E110 python_process_launched=false ' <<<"${base_output}" ||
  fail 'Sounio base Python refusal drift'
grep -Fq ' spark_route=KUBERNETES_ONLY ' <<<"${base_output}" || fail 'Spark route drift'
grep -Fq ' u250_declared=2 u250_installed=1 u250_pending_installation=1 ' <<<"${base_output}" ||
  fail 'U250 inventory drift'

require_hash "${ROOT}/${EMBEDDING_REL}" "${EMBEDDING_SHA256}"
require_hash "${ROOT}/${ENCODER_REL}" "${ENCODER_SHA256}"
require_hash "${ROOT}/${BIJECTION_REL}" "${BIJECTION_SHA256}"
require_hash "${ROOT}/${ACTION_REL}" "${ACTION_SHA256}"
require_hash "${ROOT}/${EMBEDDING_AUDIT_REL}" "${EMBEDDING_AUDIT_SHA256}"
require_hash "${ROOT}/${ENCODER_AUDIT_REL}" "${ENCODER_AUDIT_SHA256}"
require_hash "${ROOT}/${AGGREGATE_AUDIT_REL}" "${AGGREGATE_AUDIT_SHA256}"
require_hash "${ROOT}/${ACTION_AUDIT_REL}" "${ACTION_AUDIT_SHA256}"
require_hash "${ROOT}/${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_hash "${ROOT}/${OFFLOAD_LOG_REL}" "${GATE_OFFLOAD_LOG_SHA256}"
require_hash "${ROOT}/${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"

source_manifest="$(printf '%s\n' \
  "${EMBEDDING_REL}=${EMBEDDING_SHA256}" \
  "${ENCODER_REL}=${ENCODER_SHA256}" \
  "${BIJECTION_REL}=${BIJECTION_SHA256}" \
  "${ACTION_REL}=${ACTION_SHA256}")"
[[ "$(sha_text "${source_manifest}")" == "${SOURCE_BUNDLE_SHA256}" ]] || fail 'source bundle digest drift'
audit_manifest="$(printf '%s\n' \
  "${EMBEDDING_AUDIT_REL}=${EMBEDDING_AUDIT_SHA256}" \
  "${ENCODER_AUDIT_REL}=${ENCODER_AUDIT_SHA256}" \
  "${AGGREGATE_AUDIT_REL}=${AGGREGATE_AUDIT_SHA256}" \
  "${ACTION_AUDIT_REL}=${ACTION_AUDIT_SHA256}")"
[[ "$(sha_text "${audit_manifest}")" == "${AUDIT_BUNDLE_SHA256}" ]] || fail 'audit bundle digest drift'

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_GATE_COMMIT}" "${SOURCE_IMPLEMENTATION_COMMIT}" ||
  fail 'parent analytic census gate does not precede scan/action implementation'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_IMPLEMENTATION_COMMIT}" "${AUDIT_COMPLETION_COMMIT}" ||
  fail 'scan/action implementation does not precede audit completion'
git -C "${ROOT}" merge-base --is-ancestor "${AUDIT_COMPLETION_COMMIT}" "${ARTIFACT_COMMIT}" ||
  fail 'audit completion does not precede receipt/evidence seal'
git -C "${ROOT}" merge-base --is-ancestor "${ARTIFACT_COMMIT}" HEAD ||
  fail 'receipt/evidence seal is not in current history'
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${EMBEDDING_REL}" "${EMBEDDING_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${ENCODER_REL}" "${ENCODER_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${BIJECTION_REL}" "${BIJECTION_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${ACTION_REL}" "${ACTION_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${EMBEDDING_AUDIT_REL}" "${EMBEDDING_AUDIT_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${ENCODER_AUDIT_REL}" "${ENCODER_AUDIT_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${ACTION_AUDIT_REL}" "${ACTION_AUDIT_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${SOURCE_IMPLEMENTATION_COMMIT}" "${OFFLOAD_LOG_REL}" "${SOURCE_OFFLOAD_LOG_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${AGGREGATE_AUDIT_REL}" "${AGGREGATE_AUDIT_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${OFFLOAD_LOG_REL}" "${AUDIT_OFFLOAD_LOG_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${OFFLOAD_LOG_REL}" "${ARTIFACT_OFFLOAD_LOG_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'analytic_ordered_basis_count_20160_proved=true'
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'analytic_basis_to_frozen_scan_bijection_proved=false'
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "source_implementation_commit=${SOURCE_IMPLEMENTATION_COMMIT}"
require_line "${RECEIPT}" "audit_completion_commit=${AUDIT_COMPLETION_COMMIT}"
require_line "${RECEIPT}" "source_bundle_sha256=${SOURCE_BUNDLE_SHA256}"
require_line "${RECEIPT}" "audit_bundle_sha256=${AUDIT_BUNDLE_SHA256}"
require_line "${RECEIPT}" 'execution_route=LOCAL_XEON_WORKSPACE_CONTROL'
require_line "${RECEIPT}" 'execution_cpu=INTEL(R)_XEON(R)_GOLD_6526Y'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" "scan_embedding_typecheck_preexec_frame_sha256=${EMBEDDING_FRAME_SHA256}"
require_line "${RECEIPT}" "basis_encoder_typecheck_preexec_frame_sha256=${ENCODER_FRAME_SHA256}"
require_line "${RECEIPT}" "scan_bijection_typecheck_preexec_frame_sha256=${BIJECTION_FRAME_SHA256}"
require_line "${RECEIPT}" "action_census_typecheck_preexec_frame_sha256=${ACTION_FRAME_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_total_reports=107'
require_line "${RECEIPT}" 'axiom_audit_public_theorem_coverage=84_OF_84'
require_line "${RECEIPT}" 'axiom_audit_public_definition_coverage=23_OF_23'
require_line "${RECEIPT}" 'axiom_audit_no_axiom_reports=11'
require_line "${RECEIPT}" 'axiom_audit_axiom_bearing_reports=96'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=96'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=40'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=79'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'scan_embedding_proved=true'
require_line "${RECEIPT}" 'analytic_basis_encoder_proved=true'
require_line "${RECEIPT}" 'analytic_scan_bijection_proved=true'
require_line "${RECEIPT}" 'analytic_ordered_bases_nodup_proved=true'
require_line "${RECEIPT}" 'frozen_scan_count_20160_proved_analytically=true'
require_line "${RECEIPT}" 'outer_action_list_count_40320_proved=true'
require_line "${RECEIPT}" 'concrete_action_list_distinctness_proved=true'
require_line "${RECEIPT}" 'declared_action_family_container_minimum_40320_proved=true'
require_line "${RECEIPT}" 'native_matrix_scan_evaluated_for_count=false'
require_line "${RECEIPT}" 'concrete_canonical_equality_iff_full_declared_orbit_proved=false'
require_line "${RECEIPT}" 'formal_target_03_closed=false'
require_line "${RECEIPT}" 'formal_parity_complete=false'
require_line "${RECEIPT}" 'python_oracle_dispatch=E110'
require_line "${RECEIPT}" 'python_processes_launched=0'
require_line "${RECEIPT}" 'guardian_override_current_gate_failed_before_lean=true'
require_line "${RECEIPT}" 'spark_route_policy=KUBERNETES_ONLY'
require_line "${RECEIPT}" 'slurm_processes_launched=0'
require_line "${RECEIPT}" 'u250_declared_card_count=2'
require_line "${RECEIPT}" 'u250_installed_card_count=1'
require_line "${RECEIPT}" 'u250_pending_installation_card_count=1'
require_line "${RECEIPT}" 'u250_enumeration_failure_count=0'
require_line "${RECEIPT}" 'llm_role=REVIEW_ONLY'
require_line "${RECEIPT}" 'llm_confirmed_result=false'
require_line "${RECEIPT}" 'claim_ready=false'

require_line "${EVIDENCE}" 'scan_embedding_proved=true'
require_line "${EVIDENCE}" 'analytic_basis_encoder_proved=true'
require_line "${EVIDENCE}" 'analytic_scan_bijection_proved=true'
require_line "${EVIDENCE}" 'analytic_ordered_bases_nodup_proved=true'
require_line "${EVIDENCE}" 'frozen_scan_count_20160_proved_analytically=true'
require_line "${EVIDENCE}" 'outer_action_list_count_40320_proved=true'
require_line "${EVIDENCE}" 'concrete_action_list_distinctness_proved=true'
require_line "${EVIDENCE}" 'declared_action_family_container_minimum_40320_proved=true'
require_line "${EVIDENCE}" 'native_matrix_scan_evaluated_for_count=false'
require_line "${EVIDENCE}" 'concrete_canonical_equality_iff_full_declared_orbit_proved=false'
require_line "${EVIDENCE}" 'formal_target_03_closed=false'
require_line "${EVIDENCE}" 'formal_parity_complete=false'
require_line "${EVIDENCE}" 'python_oracle=PREEXEC_REFUSED_E110'
require_line "${EVIDENCE}" 'spark_route=KUBERNETES_ONLY'
require_line "${EVIDENCE}" 'u250_declared=2'
require_line "${EVIDENCE}" 'u250_installed=1'
require_line "${EVIDENCE}" 'u250_pending_installation=1'
require_line "${EVIDENCE}" 'u250_enumeration_failures=0'
require_line "${EVIDENCE}" 'claim_ready=false'

[[ "$(hostname)" == 'sounio-workspace-control-0' ]] || fail 'execution node drift'
[[ "$(uname -m)" == 'x86_64' ]] || fail 'execution architecture drift'
[[ "$(nproc)" -eq 64 ]] || fail 'logical CPU count drift'
[[ "$(lscpu | sed -n 's/^Model name:[[:space:]]*//p')" == 'INTEL(R) XEON(R) GOLD 6526Y' ]] ||
  fail 'execution CPU model drift'
[[ "$(lean --version | head -1)" == 'Lean (version 4.33.1, x86_64-unknown-linux-gnu, commit 819816b2e0a3bf405af45ae5c7af2491d8f5bee6, Release)' ]] ||
  fail 'Lean version drift'
[[ "$(lake --version)" == 'Lake version 5.0.0-src+819816b (Lean version 4.33.1)' ]] ||
  fail 'Lake version drift'
[[ "$(sha_text "${BUILD_COMMAND}")" == "${BUILD_COMMAND_SHA256}" ]] || fail 'build command drift'
[[ "$(sha_text "${EMBEDDING_COMMAND}")" == "${EMBEDDING_COMMAND_SHA256}" ]] || fail 'embedding command drift'
[[ "$(sha_text "${ENCODER_COMMAND}")" == "${ENCODER_COMMAND_SHA256}" ]] || fail 'encoder command drift'
[[ "$(sha_text "${BIJECTION_COMMAND}")" == "${BIJECTION_COMMAND_SHA256}" ]] || fail 'bijection command drift'
[[ "$(sha_text "${ACTION_COMMAND}")" == "${ACTION_COMMAND_SHA256}" ]] || fail 'action command drift'
[[ "$(sha_text "${AUDIT_COMMAND}")" == "${AUDIT_COMMAND_SHA256}" ]] || fail 'audit command drift'

[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${EMBEDDING_REL}")" -eq 30 ]] ||
  fail 'embedding theorem count drift'
[[ "$(grep -Ec '^(def|abbrev|structure) ' "${ROOT}/${EMBEDDING_REL}")" -eq 2 ]] ||
  fail 'embedding definition count drift'
[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${ENCODER_REL}")" -eq 30 ]] ||
  fail 'encoder theorem count drift'
[[ "$(grep -Ec '^(def|abbrev|structure) ' "${ROOT}/${ENCODER_REL}")" -eq 12 ]] ||
  fail 'encoder definition count drift'
[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${BIJECTION_REL}")" -eq 17 ]] ||
  fail 'bijection theorem count drift'
[[ "$(grep -Ec '^(def|abbrev|structure) ' "${ROOT}/${BIJECTION_REL}")" -eq 4 ]] ||
  fail 'bijection definition count drift'
[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${ACTION_REL}")" -eq 7 ]] ||
  fail 'action theorem count drift'
[[ "$(grep -Ec '^(def|abbrev|structure) ' "${ROOT}/${ACTION_REL}")" -eq 5 ]] ||
  fail 'action definition count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${EMBEDDING_AUDIT_REL}")" -eq 32 ]] || fail 'embedding audit count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${ENCODER_AUDIT_REL}")" -eq 42 ]] || fail 'encoder audit count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AGGREGATE_AUDIT_REL}")" -eq 107 ]] || fail 'aggregate audit count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${ACTION_AUDIT_REL}")" -eq 12 ]] || fail 'action audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^(@\[[^]]+\][[:space:]]+)?theorem ${theorem_name}([[:space:]]|:|$)" "${SOURCE_FILES[@]}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AGGREGATE_AUDIT_REL}" "#print axioms ${theorem_name}"
done
for definition_name in "${EXPECTED_DEFINITIONS[@]}"; do
  grep -Eq "^(def|abbrev|structure) ${definition_name}([[:space:]]|:|$)" "${SOURCE_FILES[@]}" ||
    fail "missing Lean definition declaration: ${definition_name}"
  require_line "${ROOT}/${AGGREGATE_AUDIT_REL}" "#print axioms ${definition_name}"
done
for source_file in "${SOURCE_FILES[@]}"; do
  [[ "$(grep -c 'native_decide' "${source_file}" || true)" -eq 0 ]] || fail "source native_decide drift: ${source_file}"
  [[ "$(grep -Ec '\bsorry\b|sorryAx' "${source_file}" || true)" -eq 0 ]] || fail "source sorry drift: ${source_file}"
done
require_line "${ROOT}/${ACTION_REL}" '  { parentAnalyticCensusProved := true'
require_line "${ROOT}/${ACTION_REL}" '  , scanEmbeddingProved := true'
require_line "${ROOT}/${ACTION_REL}" '  , analyticBasisEncoderProved := true'
require_line "${ROOT}/${ACTION_REL}" '  , analyticScanBijectionProved := true'
require_line "${ROOT}/${ACTION_REL}" '  , analyticOrderedBasesNodupProved := true'
require_line "${ROOT}/${ACTION_REL}" '  , frozenScanCount20160Proved := true'
require_line "${ROOT}/${ACTION_REL}" '  , outerActionListCount40320Proved := true'
require_line "${ROOT}/${ACTION_REL}" '  , concreteActionListDistinctnessProved := true'
require_line "${ROOT}/${ACTION_REL}" '  , outer40320MinimumProved := true'
require_line "${ROOT}/${ACTION_REL}" '  , nativeMatrixScanEvaluatedForCount := false'
require_line "${ROOT}/${ACTION_REL}" '  , concreteCanonicalEqualityIffFullDeclaredOrbitProved := false'
require_line "${ROOT}/${ACTION_REL}" '  , formalTarget03Closed := false'
require_line "${ROOT}/${ACTION_REL}" '  , formalParityClosed := false'
require_line "${ROOT}/${ACTION_REL}" '  , claimReady := false }'
require_line "${ROOT}/${BIJECTION_REL}" '    frozenScanEntries.length = 20160 := by'
require_line "${ROOT}/${ACTION_REL}" '    analyticConcreteActionList.length = 40320 := by'
require_line "${ROOT}/${ACTION_REL}" '    40320 ≤ candidate.length := by'

python_frame="$(python_oracle_frame)"
[[ "$(wc -w <<<"${python_frame}" | tr -d ' ')" -eq 82 ]] || fail 'Python frame word count drift'
set +e
python_decision="$(printf '%s\n' "${python_frame}" | "${GUARDIAN}")"
python_rc=$?
set -e
[[ "${python_rc}" -eq 110 ]] || fail "Python oracle exit drift: ${python_rc}"
[[ "${python_decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN' ]] ||
  fail "Python oracle decision drift: ${python_decision}"
python_processes_launched=0

negative_dir="$(mktemp -d /tmp/pireus-gl4-analytic-scan-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
set +e
(
  GUARDIAN=/bin/false
  authorize LOCAL_XEON_BUILD \
    "$(parity_frame "${SOURCE_BUNDLE_SHA256}" "${BUILD_COMMAND_SHA256}")" \
    "${BUILD_FRAME_SHA256}"
  printf 'LEAN_STARTED\n' >"${negative_dir}/lean-started.txt"
  cd "${ROOT}/formal/lean4"
  lake build SounioPireusGL4AnalyticActionCensus
) >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
[[ ! -e "${negative_dir}/lean-started.txt" ]] || fail 'Guardian override reached Lean execution'

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_BUNDLE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusGL4AnalyticActionCensus 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}: ${build_output}"
if grep -Eq 'SounioPireusGL4Analytic(ScanEmbedding|BasisEncoder|ScanBijection|ActionCensus)\.lean:' <<<"${build_output}"; then
  fail 'local GL4 analytic scan/action source warning drift'
fi
build_warning_count="$(count_occurrences '^warning:' "${build_output}")"
(( build_warning_count == 0 || build_warning_count == 5 )) || fail "dependency warning replay drift: ${build_warning_count}"

authorize LOCAL_XEON_SCAN_EMBEDDING_TYPECHECK "$(parity_frame "${EMBEDDING_SHA256}" "${EMBEDDING_COMMAND_SHA256}")" "${EMBEDDING_FRAME_SHA256}"
embedding_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4AnalyticScanEmbedding.lean 2>&1)"
[[ "$(count_occurrences '^warning:' "${embedding_output}")" -eq 0 ]] || fail 'embedding typecheck warning drift'

authorize LOCAL_XEON_BASIS_ENCODER_TYPECHECK "$(parity_frame "${ENCODER_SHA256}" "${ENCODER_COMMAND_SHA256}")" "${ENCODER_FRAME_SHA256}"
encoder_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4AnalyticBasisEncoder.lean 2>&1)"
[[ "$(count_occurrences '^warning:' "${encoder_output}")" -eq 0 ]] || fail 'encoder typecheck warning drift'

authorize LOCAL_XEON_SCAN_BIJECTION_TYPECHECK "$(parity_frame "${BIJECTION_SHA256}" "${BIJECTION_COMMAND_SHA256}")" "${BIJECTION_FRAME_SHA256}"
bijection_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4AnalyticScanBijection.lean 2>&1)"
[[ "$(count_occurrences '^warning:' "${bijection_output}")" -eq 0 ]] || fail 'bijection typecheck warning drift'

authorize LOCAL_XEON_ACTION_CENSUS_TYPECHECK "$(parity_frame "${ACTION_SHA256}" "${ACTION_COMMAND_SHA256}")" "${ACTION_FRAME_SHA256}"
action_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4AnalyticActionCensus.lean 2>&1)"
[[ "$(count_occurrences '^warning:' "${action_output}")" -eq 0 ]] || fail 'action census typecheck warning drift'

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AGGREGATE_AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
audit_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4AnalyticScanBijectionAxiomAudit.lean 2>&1)"
[[ "$(count_occurrences "^'SounioPireusGL4Analytic" "${audit_output}")" -eq 107 ]] || fail 'axiom report count drift'
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 96 ]] || fail 'axiom-bearing report count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 11 ]] || fail 'axiom-free report count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 96 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 40 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 79 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'

[[ "${python_processes_launched}" -eq 0 ]] || fail 'Python process launch counter drift'
printf 'PIREUS_GL4_ANALYTIC_SCAN_BIJECTION_GATE_PASS_PARTIAL=true stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY semantic_authority=Sounio source_commit=%s audit_commit=%s scan_embedding=true analytic_basis_encoder=true analytic_scan_bijection=true analytic_bases_nodup=true frozen_scan_count_20160=true native_matrix_scan_evaluated_for_count=false declared_action_count_40320=true declared_action_distinctness=true declared_family_container_minimum_40320=true full_declared_orbit_canonical_equality=false target03=false formal_parity_complete=false axiom_reports=107 axiom_free=11 propext=96 classical_choice=40 quot_sound=79 native_decide=0 sorryax=0 python_oracle=E110 python_process_launched=false guardian_current_gate_false_exit=1 spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false slurm_processes_launched=0 u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 llm_role=REVIEW_ONLY llm_confirmed_result=false zai=ERROR_1313 claim_ready=false\n' "${SOURCE_IMPLEMENTATION_COMMIT}" "${AUDIT_COMPLETION_COMMIT}"
