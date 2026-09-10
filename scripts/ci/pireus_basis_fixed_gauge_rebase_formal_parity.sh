#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusBasisFixedGaugeRebase.lean'
AUDIT_REL='formal/lean4/SounioPireusBasisFixedGaugeRebaseAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
PARENT_RECEIPT_REL='tools/pireus/linear_swap_gauge_descent.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_linear_swap_gauge_descent_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/linear_swap_gauge_descent_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/basis_fixed_gauge_rebase.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/basis_fixed_gauge_rebase_v13.formal-parity.txt'

PARENT_COMMIT='67722ca91b637efe438065cba2506f2d58acef61'
SOURCE_COMMIT='72a0f2476bc3052696ce048c4152e71268a8925f'
SOURCE_SHA256='df1e04896f1dd347f14441d7018bba632091b8870137df407a005bc040585630'
AUDIT_SHA256='8bd183df6a4c7fd3ca27dc46c5cd6a5b4de73bdfa97a122a37c62dfe7a525cd2'
LAKEFILE_SHA256='2c8f8bcde0fbd9b05d9546a4a723e9da04752a58203342c24786d76ec4448b56'
OFFLOAD_LOG_SHA256='92ee71991ed8c16208374280d533b1282eeab4af678f72ac61938a9884429cf4'
PARENT_RECEIPT_SHA256='362b8d0831c4e16a8a678dab93c8c40a5a475de005435a41eb2a9bffd2b1dd52'
PARENT_GATE_SHA256='b62bb55ae9752fc77e15d74269f2f3306ebc6eb2cf6d6e86db00756169ca368f'
PARENT_EVIDENCE_SHA256='cd997425607133beab568f3d428fe5785c191077004b6b8dec864ca023aa187f'
RECEIPT_SHA256='79fcee60e9c773476c0df92b19332d77c2c1223ad1e36d03ef12d86620c08753'
EVIDENCE_SHA256='64cff676c37031ba817dfd9476abbcbc38fb8bba633a73a8a216f23b4a362903'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusBasisFixedGaugeRebase'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusBasisFixedGaugeRebaseAxiomAudit.lean'
BUILD_COMMAND_SHA256='5cfcd0b0662caeb65ce373da7a0b8b219bfdd3c926f79c1f5006016a484cda83'
AUDIT_COMMAND_SHA256='f5577d983f5eee9d1fcfc1b197fadef115d7508257584c913c01186114ab3fc9'
BUILD_FRAME_SHA256='b9427c89b41cd498a834d3497e5e12807db9fa2da2b6c9dbd08e6e96ca4d7524'
AUDIT_FRAME_SHA256='1861c4b201bc6a4e208f30b48b6e3355bbc630354346133be45ba45fa71c2a5f'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
PYTHON_COMMAND_SHA256='92ad024a5b29367eeccdf93e8171f280baaa3c62bf46cc190ddd22ae8ad8cfc5'
ZERO='0 0 0 0 0 0 0 0'

EXPECTED_THEOREMS=(
  basis_character_xor
  basis_character_coboundary_zero
  rebased_potential_is_basis_fixed
  rebased_potential_preserves_coboundary
  basis_fixed_word_testBit
  basis_fixed_word_roundtrip
  pulled_gauge_potential_origin_zero
  rebased_gauge_word_potential
  pulled_gauge_coboundary_reencoded
  raw_action_transports_basis_fixed_gauge
  basis_fixed_rebase_progress_does_not_close_v13_target03
)

fail() {
  printf 'pireus basis-fixed gauge rebase formal parity: FAIL: %s\n' "$*" >&2
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
require_hash "${ROOT}/${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_COMMIT}" "${SOURCE_COMMIT}" ||
  fail 'parent linear/swap gate does not precede basis-fixed rebase source'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD ||
  fail 'basis-fixed rebase source commit is not in current history'
require_committed_hash "${PARENT_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${OFFLOAD_LOG_REL}" "${OFFLOAD_LOG_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'basis_fixed_gauge_rebase_after_linear_map_proved=false'
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'concrete_matrix_code_to_xor_equiv_bridge_proved=false'
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "basis_fixed_rebase_source_commit=${SOURCE_COMMIT}"
require_line "${RECEIPT}" 'source_stage_exit_code=0'
require_line "${RECEIPT}" 'source_commit_exit_code=0'
require_line "${RECEIPT}" 'execution_route=LOCAL_XEON_WORKSPACE_CONTROL'
require_line "${RECEIPT}" 'execution_cpu=INTEL(R)_XEON(R)_GOLD_6526Y'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" 'build_exit_code=0'
require_line "${RECEIPT}" 'build_warning_count=0'
require_line "${RECEIPT}" 'lean_kernel_typechecked_basis_fixed_gauge_rebase_theorems=true'
require_line "${RECEIPT}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_exit_code=0'
require_line "${RECEIPT}" 'axiom_audit_theorem_reports=11'
require_line "${RECEIPT}" 'axiom_audit_no_axiom_reports=1'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=10'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=8'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'parent_xor_linear_descent_proved=true'
require_line "${RECEIPT}" 'basis_character_linearity_proved=true'
require_line "${RECEIPT}" 'basis_character_coboundary_zero_proved=true'
require_line "${RECEIPT}" 'rebased_potential_basis_fixed_proved=true'
require_line "${RECEIPT}" 'rebased_potential_preserves_coboundary_proved=true'
require_line "${RECEIPT}" 'basis_fixed_word_encoder_explicit=true'
require_line "${RECEIPT}" 'basis_fixed_word_encoder_free_lanes=3,5,6,7,9,10,11,12,13,14,15'
require_line "${RECEIPT}" 'basis_fixed_word_encoder_roundtrip_proved=true'
require_line "${RECEIPT}" 'pulled_gauge_potential_origin_zero_proved=true'
require_line "${RECEIPT}" 'pulled_gauge_coboundary_reencoded_proved=true'
require_line "${RECEIPT}" 'raw_action_transports_basis_fixed_gauge_proved=true'
require_line "${RECEIPT}" 'basis_fixed_gauge_rebase_after_linear_map_proved=true'
require_line "${RECEIPT}" 'concrete_matrix_code_to_xor_equiv_bridge_proved=false'
require_line "${RECEIPT}" 'concrete_gl4_action_instantiated=false'
require_line "${RECEIPT}" 'gl4_code_count=20160'
require_line "${RECEIPT}" 'outer_40320_view_minimum_proved=false'
require_line "${RECEIPT}" 'outer_40320_count_interpretation=DECLARED_GL4_CODES_TIMES_INPUT_SWAP_NOT_S8_ACTION_IDENTIFICATION'
require_line "${RECEIPT}" 'outer_view_group_identification_proved=false'
require_line "${RECEIPT}" 'concrete_canonical_equality_iff_full_declared_orbit_proved=false'
require_line "${RECEIPT}" 'formal_target_03_closed=false'
require_line "${RECEIPT}" 'formal_parity_complete=false'
require_line "${RECEIPT}" 'python_oracle_dispatch=E110'
require_line "${RECEIPT}" 'python_processes_launched=0'
require_line "${RECEIPT}" 'rust_processes_launched=0'
require_line "${RECEIPT}" 'guardian_override_negative_exit_code=1'
require_line "${RECEIPT}" 'guardian_override_failed_before_lean=true'
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

require_line "${EVIDENCE}" 'basis_character_linearity_proved=true'
require_line "${EVIDENCE}" 'basis_character_coboundary_zero_proved=true'
require_line "${EVIDENCE}" 'rebased_potential_preserves_coboundary_proved=true'
require_line "${EVIDENCE}" 'basis_fixed_word_encoder_roundtrip_proved=true'
require_line "${EVIDENCE}" 'pulled_gauge_coboundary_reencoded_proved=true'
require_line "${EVIDENCE}" 'basis_fixed_gauge_rebase_after_linear_map_proved=true'
require_line "${EVIDENCE}" 'concrete_matrix_code_to_xor_equiv_bridge_proved=false'
require_line "${EVIDENCE}" 'concrete_gl4_action_instantiated=false'
require_line "${EVIDENCE}" 'outer_40320_view_minimum_proved=false'
require_line "${EVIDENCE}" 'outer_40320_count_interpretation=DECLARED_GL4_CODES_TIMES_INPUT_SWAP_NOT_S8_ACTION_IDENTIFICATION'
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
[[ "$(sha_text "${AUDIT_COMMAND}")" == "${AUDIT_COMMAND_SHA256}" ]] || fail 'audit command drift'

[[ "$(grep -c '^theorem ' "${ROOT}/${SOURCE_REL}")" -eq 13 ]] || fail 'public theorem count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 11 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
require_line "${ROOT}/${SOURCE_REL}" 'def basisCharacter (potential : Potential) : Potential :='
require_line "${ROOT}/${SOURCE_REL}" 'def rebasedPotential (potential : Potential) : Potential :='
require_line "${ROOT}/${SOURCE_REL}" 'def basisFixedBits (potential : Potential) : BitVec 11 :='
require_line "${ROOT}/${SOURCE_REL}" 'def rebasedGaugeWord (linear : XorLaneEquiv) (word : GaugeWord) : GaugeWord :='
require_line "${ROOT}/${SOURCE_REL}" '  , basisCharacterLinearityProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , basisCharacterCoboundaryZeroProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , rebasedPotentialBasisFixedProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , rebasedPotentialPreservesCoboundaryProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , basisFixedWordEncoderRoundtripProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , basisFixedGaugeRebaseAfterLinearMapProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteMatrixCodeToXorEquivBridgeProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteGL4ActionInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ViewMinimumProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalTarget03Closed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , claimReady := false }'
if grep -Eq '\bsorry\b|sorryAx|native_decide' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry, sorryAx, or native_decide found in basis-fixed rebase surface'
fi
(( 16 == 2 ** 4 )) || fail 'lane cardinality arithmetic drift'
(( 256 == 16 * 16 )) || fail 'sign-table cell arithmetic drift'
(( 2048 == 2 ** 11 )) || fail 'gauge cardinality arithmetic drift'
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

negative_dir="$(mktemp -d /tmp/pireus-basis-fixed-rebase-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
set +e
SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME=/bin/false \
  "${ROOT}/scripts/ci/pireus_operator_orbit_canonicalization.sh" \
  >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
if grep -Eq 'Built SounioPireusBasisFixedGaugeRebase|SounioPireusBasisFixedGaugeRebase.*depends on axioms' \
    "${negative_dir}/guardian-false.txt"; then
  fail 'Guardian override reached Lean execution'
fi

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusBasisFixedGaugeRebase 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}: ${build_output}"
[[ "$(count_occurrences '^warning:' "${build_output}")" -eq 0 ]] || fail 'Lean build warning drift'

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
set +e
audit_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusBasisFixedGaugeRebaseAxiomAudit.lean 2>&1)"
audit_rc=$?
set -e
[[ "${audit_rc}" -eq 0 ]] || fail "Lean axiom audit exit drift: ${audit_rc}: ${audit_output}"
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 10 ]] || fail 'axiom-bearing report count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 1 ]] || fail 'axiom-free report count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 10 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 0 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 8 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'

[[ "${python_processes_launched}" -eq 0 ]] || fail 'Python process launch counter drift'
printf 'PIREUS_BASIS_FIXED_GAUGE_REBASE_FORMAL_PARITY_PASS=true stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY semantic_authority=Sounio source_commit=%s character_linearity=true character_coboundary_zero=true basis_fixed_rebase=true encoder_11bit_roundtrip=true pulled_coboundary_reencoded=true linear_rebase=true matrix_code_bridge=false concrete_gl4=false outer_40320_minimum=false target03=false formal_parity_complete=false python_oracle=E110 python_process_launched=false guardian_false_exit=1 spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false slurm_processes_launched=0 u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 llm_role=REVIEW_ONLY llm_confirmed_result=false zai=ERROR_1313 claim_ready=false\n' "${SOURCE_COMMIT}"
