#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusGL4AnalyticCensus.lean'
AUDIT_REL='formal/lean4/SounioPireusGL4AnalyticCensusAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
BASE_GATE_REL='scripts/ci/pireus_operator_orbit_canonicalization.sh'
BASE_FREEZE_REL='tools/pireus/operator_orbit_canonicalization.freeze.v13'
PARENT_RECEIPT_REL='tools/pireus/gl4_action_enumeration.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_gl4_action_enumeration_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/gl4_action_enumeration_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/gl4_analytic_census.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/gl4_analytic_census_v13.formal-parity.txt'

PARENT_GATE_COMMIT='5ef39be8f0785e5f9118c40e3f0a171fb9f57afd'
SOURCE_AUDIT_COMMIT='0cd8126bfdbb56e7edfa42a396bc1573c5229026'
ARTIFACT_COMMIT='e29241896bcb3d027922986b692c954fb4ca3859'
SOURCE_SHA256='fe8a8ff47e70003f52ab466a42142db87956ab3f0b26397c4b376fda0e09aeaa'
AUDIT_SHA256='18851b43185c9a701b0216d172ca28c5585c83f5c0513ed8f9c4d8b5eb0cd5bf'
LAKEFILE_SHA256='e4d0c3e7e40be74611942aa505bdaefec6bfd46d90c63d8a19f75fbef2b33ba2'
SOURCE_AUDIT_OFFLOAD_LOG_SHA256='6e47e277fb6dee32eaec94f915989f2b05d8b27937c37f6c15cbf9a90d03d2a8'
GATE_OFFLOAD_LOG_SHA256='685da231cda1ee452438438cb96dd3ea0b6d6d7801be76d837d072c8e1883c93'
PARENT_RECEIPT_SHA256='cba2f498dadf7de2973b16ac509e4e3cfcdba495d0883b7999c2dbe731443073'
PARENT_GATE_SHA256='20fcdc11adf4190c51d61a6bc7f3ab2c5134164df35048cffada9ba329e958ca'
PARENT_EVIDENCE_SHA256='7039830468a5fdd214de401ae8753b1ef01473d66b36c3e07d7d51a1105f7393'
RECEIPT_SHA256='5643e9dac169f08017be363d8b59f880205a2dbbf65d4cb4521b9d1a39f73430'
EVIDENCE_SHA256='ccd2f0f7c7cca061774db0667039a89742ee66d0bac8d4f5691b98c2b9a7a203'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
BASE_GATE_SHA256='6a18d7061bd408a3050d468d65c53231d0010865543346352e7ae91a0ff11f0e'
BASE_FREEZE_SHA256='11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84'
BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusGL4AnalyticCensus'
TYPECHECK_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4AnalyticCensus.lean'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4AnalyticCensusAxiomAudit.lean'
BUILD_COMMAND_SHA256='cb3bb2037e1d52559814bd767db4ce422383c6ccedef00bd2a5c3dcfc3d1e271'
TYPECHECK_COMMAND_SHA256='91d320d39645e7eb34fcb9323cab29389021ac59db176d7e0f8a4f6970abbbaf'
AUDIT_COMMAND_SHA256='ef1dd2690bd680ff41be94b6be72ed478403c36f296822bb332aa315982eaedf'
BUILD_FRAME_SHA256='6e51548b4f60788fe21854109744490fe36f66dc56dec767fac5b3a10fc9378c'
TYPECHECK_FRAME_SHA256='d78c62f58352b9997f287c33584a5dbac8be48846b02ae4ae230eb74dc693810'
AUDIT_FRAME_SHA256='58677bbdd58bbe2aeaf29547105cee6028f3bb0f4e590c96c5f7a2b001faff51'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
PYTHON_COMMAND_SHA256='92ad024a5b29367eeccdf93e8171f280baaa3c62bf46cc190ddd22ae8ad8cfc5'
ZERO='0 0 0 0 0 0 0 0'

EXPECTED_THEOREMS=(
  eraseMany_nil eraseMany_cons eraseMany_subset nodup_not_mem_erase_self
  mem_eraseMany_not_mem_forbidden eraseMany_length_eq_sub lane_xor_zero
  lane_zero_xor lane_xor_self lane_xor_assoc lane_xor_right_injective
  lane_xor_swap_middle lane_xor_cancel_coset nodup_map_of_injective
  spanZero_nodup spanOne_closed nodup_append_xor_coset
  spanOne_nodup_of_not_zero spanTwo_nodup_of_outside spanTwo_closed
  spanThree_nodup_of_outside choicesOutside_length
  mem_choicesOutside_not_mem_span first_choices_length
  spanOne_nodup_of_first_choice second_choices_length
  spanTwo_nodup_of_second_choice third_choices_length
  spanThree_nodup_of_third_choice fourth_choices_length
  length_flatMap_of_constant fourth_completions_length
  third_completions_length second_completions_length
  analytic_ordered_basis_census_is_20160
  analytic_census_does_not_close_frozen_scan_or_target03
)

EXPECTED_DEFINITIONS=(
  eraseMany spanZero spanOne spanTwo spanThree laneUniverse choicesOutside
  firstChoices secondChoices thirdChoices fourthChoices OrderedBasis4
  fourthCompletions thirdCompletions secondCompletions analyticOrderedBases
  GL4AnalyticCensusBoundary gl4AnalyticCensusBoundary
)

fail() {
  printf 'pireus GL4 analytic census formal parity: FAIL: %s\n' "$*" >&2
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
    "$(sha_limbs "${SOURCE_SHA256}")" \
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

require_hash "${ROOT}/${SOURCE_REL}" "${SOURCE_SHA256}"
require_hash "${ROOT}/${AUDIT_REL}" "${AUDIT_SHA256}"
require_hash "${ROOT}/${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_hash "${ROOT}/${OFFLOAD_LOG_REL}" "${GATE_OFFLOAD_LOG_SHA256}"
require_hash "${ROOT}/${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_GATE_COMMIT}" "${SOURCE_AUDIT_COMMIT}" ||
  fail 'parent action-family gate does not precede analytic census source'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_AUDIT_COMMIT}" "${ARTIFACT_COMMIT}" ||
  fail 'analytic census source does not precede receipt/evidence seal'
git -C "${ROOT}" merge-base --is-ancestor "${ARTIFACT_COMMIT}" HEAD ||
  fail 'analytic census receipt/evidence seal is not in current history'
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${SOURCE_AUDIT_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${SOURCE_AUDIT_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${SOURCE_AUDIT_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${SOURCE_AUDIT_COMMIT}" "${OFFLOAD_LOG_REL}" "${SOURCE_AUDIT_OFFLOAD_LOG_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${OFFLOAD_LOG_REL}" "${GATE_OFFLOAD_LOG_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'concrete_linear_swap_action_family_instantiated=true'
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'typed_witness_count_20160_proved=false'
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "source_audit_commit=${SOURCE_AUDIT_COMMIT}"
require_line "${RECEIPT}" 'execution_route=LOCAL_XEON_WORKSPACE_CONTROL'
require_line "${RECEIPT}" 'execution_cpu=INTEL(R)_XEON(R)_GOLD_6526Y'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" "direct_typecheck_command_sha256=${TYPECHECK_COMMAND_SHA256}"
require_line "${RECEIPT}" "direct_typecheck_preexec_frame_sha256=${TYPECHECK_FRAME_SHA256}"
require_line "${RECEIPT}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_total_reports=54'
require_line "${RECEIPT}" 'axiom_audit_public_theorem_coverage=36_OF_36'
require_line "${RECEIPT}" 'axiom_audit_definition_coverage=18_OF_18'
require_line "${RECEIPT}" 'axiom_audit_no_axiom_reports=9'
require_line "${RECEIPT}" 'axiom_audit_axiom_bearing_reports=45'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=45'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=16'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=29'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'analytic_ordered_basis_list_instantiated=true'
require_line "${RECEIPT}" 'first_fiber_count_15_proved=true'
require_line "${RECEIPT}" 'second_fiber_count_14_proved=true'
require_line "${RECEIPT}" 'third_fiber_count_12_proved=true'
require_line "${RECEIPT}" 'fourth_fiber_count_8_proved=true'
require_line "${RECEIPT}" 'analytic_ordered_basis_count_20160_proved=true'
require_line "${RECEIPT}" 'native_matrix_scan_consumed=false'
require_line "${RECEIPT}" 'ordered_frames_identified_with_gl4_group=false'
require_line "${RECEIPT}" 'span_lists_equal_linear_combination_images_proved=false'
require_line "${RECEIPT}" 'analytic_basis_to_frozen_scan_bijection_proved=false'
require_line "${RECEIPT}" 'frozen_scan_count_20160_proved_analytically=false'
require_line "${RECEIPT}" 'outer_40320_action_count_proved=false'
require_line "${RECEIPT}" 'action_list_distinctness_proved=false'
require_line "${RECEIPT}" 'outer_40320_view_minimum_proved=false'
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

require_line "${EVIDENCE}" 'analytic_ordered_basis_list_instantiated=true'
require_line "${EVIDENCE}" 'analytic_ordered_basis_count_20160_proved=true'
require_line "${EVIDENCE}" 'native_matrix_scan_consumed=false'
require_line "${EVIDENCE}" 'ordered_frames_identified_with_gl4_group=false'
require_line "${EVIDENCE}" 'analytic_basis_to_frozen_scan_bijection_proved=false'
require_line "${EVIDENCE}" 'frozen_scan_count_20160_proved_analytically=false'
require_line "${EVIDENCE}" 'outer_40320_action_count_proved=false'
require_line "${EVIDENCE}" 'action_list_distinctness_proved=false'
require_line "${EVIDENCE}" 'outer_40320_view_minimum_proved=false'
require_line "${EVIDENCE}" 'formal_target_03_closed=false'
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
[[ "$(sha_text "${TYPECHECK_COMMAND}")" == "${TYPECHECK_COMMAND_SHA256}" ]] || fail 'typecheck command drift'
[[ "$(sha_text "${AUDIT_COMMAND}")" == "${AUDIT_COMMAND_SHA256}" ]] || fail 'audit command drift'

[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${SOURCE_REL}")" -eq 36 ]] ||
  fail 'public theorem count drift'
[[ "$(grep -Ec '^(def|abbrev|structure) ' "${ROOT}/${SOURCE_REL}")" -eq 18 ]] ||
  fail 'public definition count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 54 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^(@\[[^]]+\][[:space:]]+)?theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
for definition_name in "${EXPECTED_DEFINITIONS[@]}"; do
  grep -Eq "^(def|abbrev|structure) ${definition_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean definition declaration: ${definition_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${definition_name}"
done
[[ "$(grep -c 'native_decide' "${ROOT}/${SOURCE_REL}" || true)" -eq 0 ]] || fail 'source native_decide drift'
[[ "$(grep -Ec '\bsorry\b|sorryAx' "${ROOT}/${SOURCE_REL}" || true)" -eq 0 ]] || fail 'source sorry drift'
require_line "${ROOT}/${SOURCE_REL}" '  , analyticOrderedBasisListInstantiated := true'
require_line "${ROOT}/${SOURCE_REL}" '  , firstFiberCount15Proved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , secondFiberCount14Proved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , thirdFiberCount12Proved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , fourthFiberCount8Proved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , analyticOrderedBasisCount20160Proved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , nativeMatrixScanConsumed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , orderedFramesIdentifiedWithGl4Group := false'
require_line "${ROOT}/${SOURCE_REL}" '  , spanListsEqualLinearCombinationImagesProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , analyticBasisToFrozenScanBijectionProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , frozenScanCount20160ProvedAnalytically := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ActionCountProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , actionListDistinctnessProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ViewMinimumProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalTarget03Closed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalParityClosed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , claimReady := false }'
require_line "${ROOT}/${SOURCE_REL}" 'theorem first_choices_length : firstChoices.length = 15 := by'
require_line "${ROOT}/${SOURCE_REL}" '    (secondChoices first).length = 14 := by'
require_line "${ROOT}/${SOURCE_REL}" '    (thirdChoices first second).length = 12 := by'
require_line "${ROOT}/${SOURCE_REL}" '    (fourthChoices first second third).length = 8 := by'
require_line "${ROOT}/${SOURCE_REL}" '    analyticOrderedBases.length = 20160 := by'

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

negative_dir="$(mktemp -d /tmp/pireus-gl4-analytic-census-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
set +e
(
  GUARDIAN=/bin/false
  authorize LOCAL_XEON_BUILD \
    "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" \
    "${BUILD_FRAME_SHA256}"
  printf 'LEAN_STARTED\n' >"${negative_dir}/lean-started.txt"
  cd "${ROOT}/formal/lean4"
  lake build SounioPireusGL4AnalyticCensus
) >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
[[ ! -e "${negative_dir}/lean-started.txt" ]] || fail 'Guardian override reached Lean execution'

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusGL4AnalyticCensus 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}: ${build_output}"
if grep -Fq 'SounioPireusGL4AnalyticCensus.lean:' <<<"${build_output}"; then
  fail 'local GL4 analytic census source warning drift'
fi
build_warning_count="$(count_occurrences '^warning:' "${build_output}")"
(( build_warning_count == 0 || build_warning_count == 5 )) ||
  fail "dependency warning replay drift: ${build_warning_count}"

authorize LOCAL_XEON_DIRECT_TYPECHECK "$(parity_frame "${SOURCE_SHA256}" "${TYPECHECK_COMMAND_SHA256}")" "${TYPECHECK_FRAME_SHA256}"
set +e
typecheck_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4AnalyticCensus.lean 2>&1)"
typecheck_rc=$?
set -e
[[ "${typecheck_rc}" -eq 0 ]] || fail "Lean direct typecheck exit drift: ${typecheck_rc}: ${typecheck_output}"
[[ "$(count_occurrences '^warning:' "${typecheck_output}")" -eq 0 ]] || fail 'Lean direct typecheck warning drift'

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
set +e
audit_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4AnalyticCensusAxiomAudit.lean 2>&1)"
audit_rc=$?
set -e
[[ "${audit_rc}" -eq 0 ]] || fail "Lean axiom audit exit drift: ${audit_rc}: ${audit_output}"
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 45 ]] || fail 'axiom-bearing report count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 9 ]] || fail 'axiom-free report count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 45 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 16 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 29 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'

[[ "${python_processes_launched}" -eq 0 ]] || fail 'Python process launch counter drift'
printf 'PIREUS_GL4_ANALYTIC_CENSUS_GATE_PASS_PARTIAL=true stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY semantic_authority=Sounio source_audit_commit=%s ordered_basis_list=true fiber_counts=15:14:12:8 analytic_ordered_basis_count_20160=true native_matrix_scan_consumed=false frames_identified_with_gl4_group=false span_image_equality=false analytic_to_frozen_scan_bijection=false frozen_scan_count_20160=false outer_40320_count=false action_distinctness=false outer_minimum=false target03=false formal_parity_complete=false axiom_reports=54 axiom_free=9 propext=45 classical_choice=16 quot_sound=29 native_decide=0 sorryax=0 python_oracle=E110 python_process_launched=false guardian_current_gate_false_exit=1 spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false slurm_processes_launched=0 u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 llm_role=REVIEW_ONLY llm_confirmed_result=false zai=ERROR_1313 claim_ready=false\n' "${SOURCE_AUDIT_COMMIT}"
