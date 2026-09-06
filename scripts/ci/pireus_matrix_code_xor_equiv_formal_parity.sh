#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusMatrixCodeXorEquiv.lean'
AUDIT_REL='formal/lean4/SounioPireusMatrixCodeXorEquivAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
PARENT_RECEIPT_REL='tools/pireus/basis_fixed_gauge_rebase.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_basis_fixed_gauge_rebase_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/basis_fixed_gauge_rebase_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/matrix_code_xor_equiv.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/matrix_code_xor_equiv_v13.formal-parity.txt'

PARENT_COMMIT='3a9935b500507c1dbf23cb047ec70c0d00094934'
IMPLEMENTATION_COMMIT='0a8748e0cc510c373fa82f34100ae642796d7d12'
AUDIT_COMPLETION_COMMIT='f844891a223fbadae4f32bdda94463f807d4b226'
SOURCE_SHA256='3e698dc11c7fcef1807e0bb45703ac2a92d363495c62f554278f54f47df72b57'
AUDIT_SHA256='3a02c051f3ccb66ee93778eb8ed0ae4f587d9af3e9ed135d353c9339a076b766'
LAKEFILE_SHA256='799fcab9e26bb09d1efdc5c3d578d1b9ef245b344c483f33ab5f3886c28184d1'
IMPLEMENTATION_OFFLOAD_LOG_SHA256='42a1ef205a4d200985fd6a40434b07f9406accd98def157dfbd0e77961ba1c45'
AUDIT_COMPLETION_OFFLOAD_LOG_SHA256='5e4e5ab9d25aa2d50ec92b6fcfda1d0902f85a52dc63eb0a90001ab66475b807'
GATE_OFFLOAD_LOG_SHA256='a489c53c47a0a13d4c6ad19bdd60ec7c82d18297b00cb2dc599771ae6d62e7b4'
PARENT_RECEIPT_SHA256='79fcee60e9c773476c0df92b19332d77c2c1223ad1e36d03ef12d86620c08753'
PARENT_GATE_SHA256='822183bf2a9e280e752028a4c818b6f0cd26bdf432ec3f9179145d45d37ca27d'
PARENT_EVIDENCE_SHA256='64cff676c37031ba817dfd9476abbcbc38fb8bba633a73a8a216f23b4a362903'
RECEIPT_SHA256='26a3fc26ded44a7f3e9e7c5d1f3448523fc4a2f261efb19b6f754c9c16821d7a'
EVIDENCE_SHA256='cfa5a7fc367036103b33ec0f59d3687f92676bb10ecbcc4a621d3c5a3b1ceb91'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusMatrixCodeXorEquiv'
TYPECHECK_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusMatrixCodeXorEquiv.lean'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusMatrixCodeXorEquivAxiomAudit.lean'
BUILD_COMMAND_SHA256='fb9c10defab68a98ce3ee8e3b8f65c7c6faac96d831435378a8fa67e23eb79b9'
TYPECHECK_COMMAND_SHA256='23438a6cc792c9695923fc4925d46c39faba0ad64a5342228dfae657cd3d81f7'
AUDIT_COMMAND_SHA256='ea148389e549368a0307a90bd6c4ddde625e3f11753392d436e0d929ce9f602d'
BUILD_FRAME_SHA256='c4e740b7ccf1f0a5918cec2cdbaafed0a504f457ec71f81411b8f4c74de46a67'
TYPECHECK_FRAME_SHA256='ba2e3936c7617c8daacf01123fbd1adf9f9bedff70ba09ebb3ccd14f6848301e'
AUDIT_FRAME_SHA256='0dac69b2e431e365d5346e262ae6dda3335e60b655172afd3c3d3a359ac561e2'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
PYTHON_COMMAND_SHA256='92ad024a5b29367eeccdf93e8171f280baaa3c62bf46cc190ddd22ae8ad8cfc5'
ZERO='0 0 0 0 0 0 0 0'

EXPECTED_THEOREMS=(
  encode_f2_word4_xor
  parity4_lt_two
  parity4_and_xor_on_lanes
  matrix_parity_xor
  matrix_apply_eq_encode_f2_word4
  matrix_apply_lt_sixteen
  matrix_apply_xor
  matrix_lane_map_zero
  matrix_lane_map_xor
  eraseDups_length_le_nat
  nodup_of_eraseDups_length_eq_nat
  matrix_images_nodup_of_invertible
  matrix_images_subset_range
  matrix_images_get
  every_lane_mem_matrix_images
  matrix_lane_right_inverse
  matrix_lane_map_injective
  firstPreimage16_left_inverse
  matrix_lane_left_inverse
  matrix_lane_inverse_zero
  matrix_lane_inverse_xor
  matrix_code_action_transports_basis_fixed_gauge
  matrix_code_bridge_progress_does_not_close_v13_target03
)

EXPECTED_DEFINITIONS=(
  natMemDecidable
  matrixCodeXorEquiv
)

fail() {
  printf 'pireus matrix-code XOR equivalence formal parity: FAIL: %s\n' "$*" >&2
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

base_output="$("${ROOT}/scripts/ci/pireus_operator_orbit_canonicalization.sh")"
grep -Fq 'PIREUS_OPERATOR_ORBIT_CANONICALIZATION_GATE_PASS=true' <<<"${base_output}" ||
  fail 'Sounio semantic authority gate did not pass'
grep -Fq ' stage=SEMANTICS_FROZEN ' <<<"${base_output}" || fail 'Sounio base stage drift'
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

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_COMMIT}" "${AUDIT_COMPLETION_COMMIT}" ||
  fail 'parent basis-fixed gate does not precede matrix-code bridge source'
git -C "${ROOT}" merge-base --is-ancestor "${IMPLEMENTATION_COMMIT}" "${AUDIT_COMPLETION_COMMIT}" ||
  fail 'matrix-code implementation does not precede completed axiom audit'
git -C "${ROOT}" merge-base --is-ancestor "${AUDIT_COMPLETION_COMMIT}" HEAD ||
  fail 'matrix-code bridge source commit is not in current history'
require_committed_hash "${PARENT_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${IMPLEMENTATION_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${IMPLEMENTATION_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${IMPLEMENTATION_COMMIT}" "${OFFLOAD_LOG_REL}" "${IMPLEMENTATION_OFFLOAD_LOG_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${OFFLOAD_LOG_REL}" "${AUDIT_COMPLETION_OFFLOAD_LOG_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'basis_fixed_gauge_rebase_after_linear_map_proved=true'
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'concrete_matrix_code_to_xor_equiv_bridge_proved=false'
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "matrix_code_bridge_source_commit=${IMPLEMENTATION_COMMIT}"
require_line "${RECEIPT}" "matrix_code_bridge_audit_completion_commit=${AUDIT_COMPLETION_COMMIT}"
require_line "${RECEIPT}" "implementation_offload_log_sha256=${IMPLEMENTATION_OFFLOAD_LOG_SHA256}"
require_line "${RECEIPT}" "audit_completion_offload_log_sha256=${AUDIT_COMPLETION_OFFLOAD_LOG_SHA256}"
require_line "${RECEIPT}" 'source_stage_exit_code=0'
require_line "${RECEIPT}" 'source_commit_exit_code=0'
require_line "${RECEIPT}" 'execution_route=LOCAL_XEON_WORKSPACE_CONTROL'
require_line "${RECEIPT}" 'execution_cpu=INTEL(R)_XEON(R)_GOLD_6526Y'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" 'build_exit_code=0'
require_line "${RECEIPT}" 'source_validation_build_dependency_warning_count=5'
require_line "${RECEIPT}" 'build_local_source_warning_count=0'
require_line "${RECEIPT}" 'gate_replay_build_warning_count_policy=CACHE_DEPENDENT_ZERO_OR_FIVE_BASELINE_ONLY'
require_line "${RECEIPT}" "direct_typecheck_command_sha256=${TYPECHECK_COMMAND_SHA256}"
require_line "${RECEIPT}" "direct_typecheck_preexec_frame_sha256=${TYPECHECK_FRAME_SHA256}"
require_line "${RECEIPT}" 'direct_typecheck_exit_code=0'
require_line "${RECEIPT}" 'direct_typecheck_warning_count=0'
require_line "${RECEIPT}" 'lean_kernel_typechecked_matrix_code_xor_equiv_theorems=true'
require_line "${RECEIPT}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_exit_code=0'
require_line "${RECEIPT}" 'axiom_audit_total_reports=25'
require_line "${RECEIPT}" 'axiom_audit_public_theorem_coverage=23_OF_23'
require_line "${RECEIPT}" 'axiom_audit_definition_reports=2'
require_line "${RECEIPT}" 'axiom_audit_complete_public_surface=true'
require_line "${RECEIPT}" 'axiom_audit_no_axiom_reports=2'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=23'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=6'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_provenance=LIST_NODUP_LENGTH_LE_OF_SUBSET_PIGEONHOLE_AND_RIGHT_INVERSE_DEPENDENTS'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=18'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'parent_basis_fixed_rebase_proved=true'
require_line "${RECEIPT}" 'matrix_parity_decomposition_proved=true'
require_line "${RECEIPT}" 'matrix_application_bounded_proved=true'
require_line "${RECEIPT}" 'matrix_application_preserves_xor_proved=true'
require_line "${RECEIPT}" 'invertible_image_list_nodup_proved=true'
require_line "${RECEIPT}" 'every_lane_has_preimage_proved=true'
require_line "${RECEIPT}" 'explicit_first_preimage_16_computable=true'
require_line "${RECEIPT}" 'matrix_lane_left_inverse_proved=true'
require_line "${RECEIPT}" 'matrix_lane_right_inverse_proved=true'
require_line "${RECEIPT}" 'matrix_lane_inverse_zero_proved=true'
require_line "${RECEIPT}" 'matrix_lane_inverse_xor_proved=true'
require_line "${RECEIPT}" 'matrix_code_to_xor_equiv_bridge_proved=true'
require_line "${RECEIPT}" 'per_witness_linear_swap_action_instantiated=true'
require_line "${RECEIPT}" 'matrix_code_action_transports_basis_fixed_gauge_proved=true'
require_line "${RECEIPT}" 'concrete_20160_witness_list_instantiated=false'
require_line "${RECEIPT}" 'outer_40320_view_list_instantiated=false'
require_line "${RECEIPT}" 'outer_40320_view_minimum_proved=false'
require_line "${RECEIPT}" 'outer_40320_count_interpretation=DECLARED_GL4_CODES_TIMES_INPUT_SWAP_NOT_S8_ACTION_IDENTIFICATION'
require_line "${RECEIPT}" 'outer_view_group_identification_proved=false'
require_line "${RECEIPT}" 'concrete_canonical_equality_iff_full_declared_orbit_proved=false'
require_line "${RECEIPT}" 'formal_target_03_closed=false'
require_line "${RECEIPT}" 'formal_parity_complete=false'
require_line "${RECEIPT}" 'python_oracle_dispatch=E110'
require_line "${RECEIPT}" 'python_processes_launched=0'
require_line "${RECEIPT}" 'rust_processes_launched=0'
require_line "${RECEIPT}" 'guardian_override_current_gate_negative_exit_code=1'
require_line "${RECEIPT}" 'guardian_override_current_gate_failed_before_lean=true'
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
require_line "${RECEIPT}" 'llm_second_opinion_failure=ZAI_PROVIDER_ERROR_1313'
require_line "${RECEIPT}" 'claim_ready=false'

require_line "${EVIDENCE}" 'matrix_parity_decomposition_proved=true'
require_line "${EVIDENCE}" 'matrix_application_preserves_xor_proved=true'
require_line "${EVIDENCE}" 'invertible_image_list_nodup_proved=true'
require_line "${EVIDENCE}" 'every_lane_has_preimage_proved=true'
require_line "${EVIDENCE}" 'explicit_first_preimage_16_computable=true'
require_line "${EVIDENCE}" 'matrix_lane_left_inverse_proved=true'
require_line "${EVIDENCE}" 'matrix_lane_right_inverse_proved=true'
require_line "${EVIDENCE}" 'matrix_lane_inverse_xor_proved=true'
require_line "${EVIDENCE}" 'matrix_code_to_xor_equiv_bridge_proved=true'
require_line "${EVIDENCE}" 'per_witness_linear_swap_action_instantiated=true'
require_line "${EVIDENCE}" 'concrete_20160_witness_list_instantiated=false'
require_line "${EVIDENCE}" 'outer_40320_view_list_instantiated=false'
require_line "${EVIDENCE}" 'outer_40320_view_minimum_proved=false'
require_line "${EVIDENCE}" 'formal_target_03_closed=false'
require_line "${EVIDENCE}" 'python_oracle=PREEXEC_REFUSED_E110'
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

[[ "$(grep -c '^theorem ' "${ROOT}/${SOURCE_REL}")" -eq 23 ]] || fail 'public theorem count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 25 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
for definition_name in "${EXPECTED_DEFINITIONS[@]}"; do
  grep -Eq "^def ${definition_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean definition declaration: ${definition_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${definition_name}"
done
require_line "${ROOT}/${SOURCE_REL}" 'def encodeF2Word4 (b0 b1 b2 b3 : F2Bit) : Nat :='
require_line "${ROOT}/${SOURCE_REL}" 'def matrixLaneMap (code : Nat) (lane : Lane) : Lane :='
require_line "${ROOT}/${SOURCE_REL}" 'def firstPreimage16 (map : Lane -> Lane) (target : Lane) : Lane :='
require_line "${ROOT}/${SOURCE_REL}" 'def matrixCodeXorEquiv (matrix : InvertibleMatrixCode) : XorLaneEquiv :='
require_line "${ROOT}/${SOURCE_REL}" '  , matrixCodeToXorEquivBridgeProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , perWitnessLinearSwapActionInstantiated := true'
require_line "${ROOT}/${SOURCE_REL}" '  , concrete20160WitnessListInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ViewListInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ViewMinimumProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteCanonicalEqualityIffFullDeclaredOrbitProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalTarget03Closed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalParityClosed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , claimReady := false }'
if grep -Eq '\bsorry\b|sorryAx|native_decide' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry, sorryAx, or native_decide found in matrix-code bridge surface'
fi
(( 16 == 2 ** 4 )) || fail 'lane cardinality arithmetic drift'
(( 256 == 16 * 16 )) || fail 'sign-table cell arithmetic drift'
(( 65536 == 2 ** 16 )) || fail 'matrix-code cardinality arithmetic drift'
(( 20160 == 15 * 14 * 12 * 8 )) || fail 'GL4 cardinality arithmetic drift'
(( 40320 == 20160 * 2 )) || fail 'outer view arithmetic drift'
(( 2 == 1 + 1 )) || fail 'U250 inventory arithmetic drift'

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

negative_dir="$(mktemp -d /tmp/pireus-matrix-code-xor-equiv-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
set +e
(
  GUARDIAN=/bin/false
  authorize LOCAL_XEON_BUILD \
    "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" \
    "${BUILD_FRAME_SHA256}"
  printf 'LEAN_STARTED\n' >"${negative_dir}/lean-started.txt"
  cd "${ROOT}/formal/lean4"
  lake build SounioPireusMatrixCodeXorEquiv
) >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
[[ ! -e "${negative_dir}/lean-started.txt" ]] || fail 'Guardian override reached Lean execution'

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusMatrixCodeXorEquiv 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}: ${build_output}"
if grep -Fq 'SounioPireusMatrixCodeXorEquiv.lean:' <<<"${build_output}"; then
  fail 'local matrix-code source warning drift'
fi
build_warning_count="$(count_occurrences '^warning:' "${build_output}")"
(( build_warning_count == 0 || build_warning_count == 5 )) ||
  fail "dependency warning replay drift: ${build_warning_count}"

authorize LOCAL_XEON_DIRECT_TYPECHECK "$(parity_frame "${SOURCE_SHA256}" "${TYPECHECK_COMMAND_SHA256}")" "${TYPECHECK_FRAME_SHA256}"
set +e
typecheck_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusMatrixCodeXorEquiv.lean 2>&1)"
typecheck_rc=$?
set -e
[[ "${typecheck_rc}" -eq 0 ]] || fail "Lean direct typecheck exit drift: ${typecheck_rc}: ${typecheck_output}"
[[ "$(count_occurrences '^warning:' "${typecheck_output}")" -eq 0 ]] || fail 'Lean direct typecheck warning drift'

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
set +e
audit_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusMatrixCodeXorEquivAxiomAudit.lean 2>&1)"
audit_rc=$?
set -e
[[ "${audit_rc}" -eq 0 ]] || fail "Lean axiom audit exit drift: ${audit_rc}: ${audit_output}"
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 23 ]] || fail 'axiom-bearing report count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 2 ]] || fail 'axiom-free report count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 23 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 6 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 18 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'

[[ "${python_processes_launched}" -eq 0 ]] || fail 'Python process launch counter drift'
printf 'PIREUS_MATRIX_CODE_XOR_EQUIV_GATE_PASS_PARTIAL=true stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY semantic_authority=Sounio source_commit=%s matrix_parity=true bounded=true xor=true nodup=true lane_surjective=true inverse16=true left_inverse=true right_inverse=true inverse_xor=true matrix_code_bridge=true per_witness_action=true basis_fixed_transport=true concrete_20160=false outer_40320_list=false outer_40320_minimum=false target03=false formal_parity_complete=false classical_choice=6 native_decide=0 sorryax=0 python_oracle=E110 python_process_launched=false guardian_current_gate_false_exit=1 spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false slurm_processes_launched=0 u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 llm_role=REVIEW_ONLY llm_confirmed_result=false zai=ERROR_1313 claim_ready=false\n' "${AUDIT_COMPLETION_COMMIT}"
