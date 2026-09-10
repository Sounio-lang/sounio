#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusGaugeSectionCanonicalization.lean'
AUDIT_REL='formal/lean4/SounioPireusGaugeSectionCanonicalizationAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
PARENT_RECEIPT_REL='tools/pireus/gauge_coboundary_faithfulness.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_gauge_coboundary_faithfulness_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/gauge_coboundary_faithfulness_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/gauge_section_canonicalization.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/gauge_section_canonicalization_v13.formal-parity.txt'

PARENT_COMMIT='469eb0b9eb9346132ab7a59524092cf184353d82'
SOURCE_COMMIT='0d0fca42be43072b54b142e0646715f81ad0b139'
SOURCE_SHA256='a44d28a07ac257553655c89382bb11b73879fc45c2b1130a80a5a8557029eed3'
AUDIT_SHA256='4a9073e5968d456fbc4af77b12efb6c16bbe253251a17a21d5694a5fbff55941'
LAKEFILE_SHA256='1c1aba76cae2077bd7ce1bae7232fce5dc4e81269f219c22f1bbdf79b47ecedc'
OFFLOAD_LOG_SHA256='04bdfb0565286083d91d7ce68c03db122e297cce2a64240038487086d05906d6'
PARENT_RECEIPT_SHA256='6dffdeeb74628f676c58b8618d462ac469f4767da468da48a7c9de1e555ec4ab'
PARENT_GATE_SHA256='1f35e1022ac15725b952e059649f3d776788309b8f28902e226cbe5341a7fb45'
PARENT_EVIDENCE_SHA256='d0ef0abfa88c0482c183f09db3848246577fe1d5f1250459e233c1e0569b791f'
RECEIPT_SHA256='7bb68f8640f93d8ffe946db1350b82665881638a6d6c4f40948aa04c78cfefc2'
EVIDENCE_SHA256='3643f76bdec2798631b8dfccad3f4aaff9089cbca9dbb5fb33b2ed29cfb1cfe8'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusGaugeSectionCanonicalization'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGaugeSectionCanonicalizationAxiomAudit.lean'
BUILD_COMMAND_SHA256='167c7b082adf1652df429f2f6ab99d9eb63cacbafe0f3ec287584484e3723d7a'
AUDIT_COMMAND_SHA256='98749b166210afce7b0c5cbdc1322d8fe1b9b7f0509f936c794c3203753fd274'
BUILD_FRAME_SHA256='6b952a0533a954dd15262ae4cc1ddd6296f1df04d8f1345aeac3775849557dca'
AUDIT_FRAME_SHA256='17795721ba7e6ca69d9644116e1fa335125222bb4ccec63c5e33a4a32ac37759'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
PYTHON_COMMAND_SHA256='92ad024a5b29367eeccdf93e8171f280baaa3c62bf46cc190ddd22ae8ad8cfc5'
ZERO='0 0 0 0 0 0 0 0'

EXPECTED_THEOREMS=(
  section_bit_xor
  section_bit_gauge_action
  direct_section_word_testBit
  direct_section_word_gauge_action
  normalize_gauge_has_zero_section_bits
  normalize_gauge_idempotent
  normalize_gauge_invariant
  same_gauge_orbit_refl
  same_gauge_orbit_symm
  same_gauge_orbit_trans
  normalize_gauge_equal_of_same_orbit
  same_orbit_of_normalize_gauge_equal
  normalize_gauge_equal_iff_same_orbit
  direct_section_is_unique_orbit_representative
  gauge_section_progress_does_not_close_v13_target03
)

fail() {
  printf 'pireus gauge section canonicalization formal parity: FAIL: %s\n' "$*" >&2
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
  fail 'parent gauge faithfulness gate does not precede gauge section source'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD ||
  fail 'gauge section source commit is not in current history'
require_committed_hash "${PARENT_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${OFFLOAD_LOG_REL}" "${OFFLOAD_LOG_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'gauge_action_free_on_every_sign_table_proved=true'
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'tree_section_decoder_matches_frozen_sounio_recurrence=true'
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'tree_section_lexicographic_minimality_proved=false'
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" 'semantic_authority_function=poc_direct_section'
require_line "${RECEIPT}" "gauge_section_source_commit=${SOURCE_COMMIT}"
require_line "${RECEIPT}" 'source_stage_frame_sha256=557c89d93034bda48fdab71f12c17df698192093d47e762a266d5503bedb5f4f'
require_line "${RECEIPT}" 'source_commit_frame_sha256=ba384fdb2a0922ec2681b6b28099a1630aa8199828292e0c6e08c054025631b5'
require_line "${RECEIPT}" 'execution_route=LOCAL_XEON_WORKSPACE_CONTROL'
require_line "${RECEIPT}" 'execution_cpu=INTEL(R)_XEON(R)_GOLD_6526Y'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" 'build_exit_code=0'
require_line "${RECEIPT}" 'build_warning_count=0'
require_line "${RECEIPT}" 'lean_kernel_typechecked_gauge_section_theorems=true'
require_line "${RECEIPT}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_exit_code=0'
require_line "${RECEIPT}" 'axiom_audit_theorem_reports=15'
require_line "${RECEIPT}" 'axiom_audit_no_axiom_reports=1'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=14'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=13'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'gauge_bits=11'
require_line "${RECEIPT}" 'gauge_word_count=2048'
require_line "${RECEIPT}" 'direct_section_bit_count=11'
require_line "${RECEIPT}" 'direct_section_word_constructed=true'
require_line "${RECEIPT}" 'direct_section_word_uses_parent_verified_tree_recurrence=true'
require_line "${RECEIPT}" 'direct_section_word_equivariance_proved=true'
require_line "${RECEIPT}" 'normalization_produces_zero_section_representative=true'
require_line "${RECEIPT}" 'normalization_idempotent_proved=true'
require_line "${RECEIPT}" 'normalization_invariant_on_gauge_orbit_proved=true'
require_line "${RECEIPT}" 'same_gauge_orbit_reflexive_proved=true'
require_line "${RECEIPT}" 'same_gauge_orbit_symmetric_proved=true'
require_line "${RECEIPT}" 'same_gauge_orbit_transitive_proved=true'
require_line "${RECEIPT}" 'normalization_equality_iff_same_gauge_orbit_proved=true'
require_line "${RECEIPT}" 'at_most_one_zero_section_representative_per_gauge_orbit_proved=true'
require_line "${RECEIPT}" 'basis_fixed_gauge_quotient_canonicalization_proved=true'
require_line "${RECEIPT}" 'gauge_section_scope=BASIS_FIXED_11_BIT_GAUGE_ACTION_ONLY'
require_line "${RECEIPT}" 'tree_section_lexicographic_minimality_proved=false'
require_line "${RECEIPT}" 'concrete_input_swap_action_instantiated=false'
require_line "${RECEIPT}" 'concrete_gl4_action_instantiated=false'
require_line "${RECEIPT}" 'outer_40320_view_minimum_proved=false'
require_line "${RECEIPT}" 'outer_gl4_swap_orbit_canonicalization_proved=false'
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

require_line "${EVIDENCE}" 'direct_section_word_constructed=true'
require_line "${EVIDENCE}" 'direct_section_word_equivariance_proved=true'
require_line "${EVIDENCE}" 'normalization_produces_zero_section_representative=true'
require_line "${EVIDENCE}" 'normalization_idempotent_proved=true'
require_line "${EVIDENCE}" 'normalization_invariant_on_gauge_orbit_proved=true'
require_line "${EVIDENCE}" 'same_gauge_orbit_reflexive_proved=true'
require_line "${EVIDENCE}" 'same_gauge_orbit_symmetric_proved=true'
require_line "${EVIDENCE}" 'same_gauge_orbit_transitive_proved=true'
require_line "${EVIDENCE}" 'normalization_equality_iff_same_gauge_orbit_proved=true'
require_line "${EVIDENCE}" 'at_most_one_zero_section_representative_per_gauge_orbit_proved=true'
require_line "${EVIDENCE}" 'basis_fixed_gauge_quotient_canonicalization_proved=true'
require_line "${EVIDENCE}" 'tree_section_lexicographic_minimality_proved=false'
require_line "${EVIDENCE}" 'outer_40320_view_minimum_proved=false'
require_line "${EVIDENCE}" 'formal_target_03_closed=false'
require_line "${EVIDENCE}" 'python_oracle=PREEXEC_REFUSED_E110'
require_line "${EVIDENCE}" 'u250_declared=2'
require_line "${EVIDENCE}" 'u250_installed=1'
require_line "${EVIDENCE}" 'u250_pending_installation=1'
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

[[ "$(grep -c '^theorem ' "${ROOT}/${SOURCE_REL}")" -eq 17 ]] || fail 'public theorem count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 15 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
require_line "${ROOT}/${SOURCE_REL}" 'def directSectionBits (table : SignTable) : BitVec 11 :='
require_line "${ROOT}/${SOURCE_REL}" 'def directSectionWord (table : SignTable) : GaugeWord :='
require_line "${ROOT}/${SOURCE_REL}" 'def normalizeGauge (table : SignTable) : SignTable :='
require_line "${ROOT}/${SOURCE_REL}" 'def SameGaugeOrbit (left right : SignTable) : Prop :='
require_line "${ROOT}/${SOURCE_REL}" 'theorem direct_section_word_zero_of_section'
require_line "${ROOT}/${SOURCE_REL}" 'theorem normalize_gauge_fixed_of_section'
require_line "${ROOT}/${SOURCE_REL}" '  , sameGaugeOrbitEquivalenceProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , normalizationEqualityIffGaugeOrbitProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , uniqueDirectSectionRepresentativeProved := true'
require_line "${ROOT}/${SOURCE_REL}" '  , treeSectionLexicographicMinimumProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteInputSwapActionInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteGL4ActionInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ViewMinimumProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalTarget03Closed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , claimReady := false }'
if grep -Eq '\bsorry\b|sorryAx|native_decide' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry, sorryAx, or native_decide found in gauge section surface'
fi
(( 2048 == 2 ** 11 )) || fail 'gauge cardinality arithmetic drift'
(( 256 == 16 * 16 )) || fail 'sign-table cell arithmetic drift'
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

negative_dir="$(mktemp -d /tmp/pireus-gauge-section-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
set +e
SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME=/bin/false \
  "${ROOT}/scripts/ci/pireus_operator_orbit_canonicalization.sh" \
  >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
if grep -Eq 'Built SounioPireusGaugeSectionCanonicalization|SounioPireusGaugeSectionCanonicalization.*depends on axioms' \
    "${negative_dir}/guardian-false.txt"; then
  fail 'Guardian override reached Lean execution'
fi

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusGaugeSectionCanonicalization 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}: ${build_output}"
[[ "$(count_occurrences '^warning:' "${build_output}")" -eq 0 ]] || fail 'Lean build warning drift'

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
set +e
audit_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGaugeSectionCanonicalizationAxiomAudit.lean 2>&1)"
audit_rc=$?
set -e
[[ "${audit_rc}" -eq 0 ]] || fail "Lean axiom audit exit drift: ${audit_rc}: ${audit_output}"
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 14 ]] || fail 'axiom-bearing report count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 1 ]] || fail 'axiom-free report count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 14 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 0 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 13 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'

[[ "${python_processes_launched}" -eq 0 ]] || fail 'Python process launch counter drift'
printf 'PIREUS_GAUGE_SECTION_CANONICALIZATION_FORMAL_PARITY_PASS=true stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY semantic_authority=Sounio source_commit=%s gauge_words=2048 direct_section=true orbit_equivalence=true gauge_quotient_canonicalization=true tree_minimum=false input_swap=false gl4=false outer_40320_minimum=false target03=false formal_parity_complete=false python_oracle=E110 python_process_launched=false guardian_false_exit=1 spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false slurm_processes_launched=0 u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 llm_role=REVIEW_ONLY llm_confirmed_result=false zai=ERROR_1313 claim_ready=false\n' "${SOURCE_COMMIT}"
