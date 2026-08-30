#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusGL4ActionEnumeration.lean'
AUDIT_REL='formal/lean4/SounioPireusGL4ActionEnumerationAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
PARENT_RECEIPT_REL='tools/pireus/matrix_code_xor_equiv.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_matrix_code_xor_equiv_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/matrix_code_xor_equiv_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/gl4_action_enumeration.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/gl4_action_enumeration_v13.formal-parity.txt'

PARENT_GATE_COMMIT='97a89c7220043ba85c5480a068a7514150069c49'
IMPLEMENTATION_COMMIT='ea85c892d7d98be5c133e8ba2b96681358baea47'
AUDIT_COMPLETION_COMMIT='0d4e0d4ea71d2cd4460202f073bbacd139dc3ff5'
ARTIFACT_COMMIT='0c6e2b248120d0f6a5bb178143ed21eade238ad4'
SOURCE_SHA256='3550ed03d792fe5f0810a3d0de4164d869d9194b63b48da455e7f5cc844aaba0'
AUDIT_SHA256='cffb8260ad1645f8ff0631a3dc037c6bfe2bd896462a51c7d39e33f574cc0210'
LAKEFILE_SHA256='f21acd88b61280866e8dac97f12fa736d6c296e4ea50f889149e7947e6a57c2e'
IMPLEMENTATION_OFFLOAD_LOG_SHA256='ebc42592ee2b1e43c40b9ac0342290ed1c457263aaa8e152b5204806175f2186'
AUDIT_COMPLETION_OFFLOAD_LOG_SHA256='ebc42592ee2b1e43c40b9ac0342290ed1c457263aaa8e152b5204806175f2186'
GATE_OFFLOAD_LOG_SHA256='29a23bd01706abecdbd147e296be75ce5c9779760467220ece1d046d65b5ae70'
PARENT_RECEIPT_SHA256='26a3fc26ded44a7f3e9e7c5d1f3448523fc4a2f261efb19b6f754c9c16821d7a'
PARENT_GATE_SHA256='b2b658458930434fdb3ae7fb2a4ea2809c03715aa65521e3cdecbd7b98c7fc9f'
PARENT_EVIDENCE_SHA256='cfa5a7fc367036103b33ec0f59d3687f92676bb10ecbcc4a621d3c5a3b1ceb91'
RECEIPT_SHA256='cba2f498dadf7de2973b16ac509e4e3cfcdba495d0883b7999c2dbe731443073'
EVIDENCE_SHA256='7039830468a5fdd214de401ae8753b1ef01473d66b36c3e07d7d51a1105f7393'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusGL4ActionEnumeration'
TYPECHECK_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4ActionEnumeration.lean'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGL4ActionEnumerationAxiomAudit.lean'
BUILD_COMMAND_SHA256='b17283eb67bb107d776dd8a1d8b6ee1c928829d0afea24ed62b03f7c35a09402'
TYPECHECK_COMMAND_SHA256='7334ff85820ac55ad1d04bc5c4d6b601176d9b3ddde807c875d384d28510fc9a'
AUDIT_COMMAND_SHA256='c76b782b1a2a5106410235fae56c4ed868a3727acdfbcde2b678b666cf5eba86'
BUILD_FRAME_SHA256='abd957d66dba4fa67311a1bd48210cfd50d0b39ca3b5488994077bda85e8b203'
TYPECHECK_FRAME_SHA256='cc0af6332f9bdf6160fdb974d392fb0f5c276e8343121e777a8e21e2d304b080'
AUDIT_FRAME_SHA256='c0f7c9b1e150281d981d717805575d0d0834daf1cb66d94b4c4c97d27ce30d96'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
PYTHON_COMMAND_SHA256='92ad024a5b29367eeccdf93e8171f280baaa3c62bf46cc190ddd22ae8ad8cfc5'
ZERO='0 0 0 0 0 0 0 0'

EXPECTED_THEOREMS=(
  scan_membership_facts
  matrix_witness_of_scan_entry_code
  typed_gl4_witness_originates_in_frozen_scan
  every_admitted_matrix_code_has_typed_entry
  each_scan_entry_has_both_concrete_actions
  the_two_views_have_the_same_matrix_witness
  gl4_action_enumeration_does_not_close_v13_target03
)

EXPECTED_DEFINITIONS=(
  matrixWitnessOfScanEntry
  viewWitness
  concreteLinearSwapActionAt
  unswappedViewOf
  swappedViewOf
  gl4ActionEnumerationBoundary
)

fail() {
  printf 'pireus GL4 action enumeration formal parity: FAIL: %s\n' "$*" >&2
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

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_GATE_COMMIT}" "${IMPLEMENTATION_COMMIT}" ||
  fail 'parent matrix-code gate does not precede GL4 action implementation'
git -C "${ROOT}" merge-base --is-ancestor "${IMPLEMENTATION_COMMIT}" "${AUDIT_COMPLETION_COMMIT}" ||
  fail 'GL4 action implementation does not precede completed axiom audit'
git -C "${ROOT}" merge-base --is-ancestor "${AUDIT_COMPLETION_COMMIT}" HEAD ||
  fail 'GL4 action audit completion is not in current history'
git -C "${ROOT}" merge-base --is-ancestor "${AUDIT_COMPLETION_COMMIT}" "${ARTIFACT_COMMIT}" ||
  fail 'GL4 action audit completion does not precede receipt/evidence seal'
git -C "${ROOT}" merge-base --is-ancestor "${ARTIFACT_COMMIT}" HEAD ||
  fail 'GL4 action receipt/evidence seal is not in current history'
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${IMPLEMENTATION_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${IMPLEMENTATION_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${IMPLEMENTATION_COMMIT}" "${OFFLOAD_LOG_REL}" "${IMPLEMENTATION_OFFLOAD_LOG_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${AUDIT_COMPLETION_COMMIT}" "${OFFLOAD_LOG_REL}" "${AUDIT_COMPLETION_OFFLOAD_LOG_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${OFFLOAD_LOG_REL}" "${GATE_OFFLOAD_LOG_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'matrix_code_to_xor_equiv_bridge_proved=true'
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'concrete_20160_witness_list_instantiated=false'
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "implementation_commit=${IMPLEMENTATION_COMMIT}"
require_line "${RECEIPT}" "audit_completion_commit=${AUDIT_COMPLETION_COMMIT}"
require_line "${RECEIPT}" 'execution_route=LOCAL_XEON_WORKSPACE_CONTROL'
require_line "${RECEIPT}" 'execution_cpu=INTEL(R)_XEON(R)_GOLD_6526Y'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" "direct_typecheck_command_sha256=${TYPECHECK_COMMAND_SHA256}"
require_line "${RECEIPT}" "direct_typecheck_preexec_frame_sha256=${TYPECHECK_FRAME_SHA256}"
require_line "${RECEIPT}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_total_reports=13'
require_line "${RECEIPT}" 'axiom_audit_public_theorem_coverage=7_OF_7'
require_line "${RECEIPT}" 'axiom_audit_definition_coverage=6_OF_6'
require_line "${RECEIPT}" 'axiom_audit_no_axiom_reports=2'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=11'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=2'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=9'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'typed_witness_subtype_instantiated=true'
require_line "${RECEIPT}" 'every_scan_entry_has_typed_witness=true'
require_line "${RECEIPT}" 'every_predicate_witness_has_entry=true'
require_line "${RECEIPT}" 'two_views_per_entry_instantiated=true'
require_line "${RECEIPT}" 'concrete_linear_swap_action_family_instantiated=true'
require_line "${RECEIPT}" 'typed_witness_list_instantiated=false'
require_line "${RECEIPT}" 'typed_witness_count_20160_proved=false'
require_line "${RECEIPT}" 'imported_native_census_consumed=false'
require_line "${RECEIPT}" 'outer_40320_view_list_instantiated=false'
require_line "${RECEIPT}" 'concrete_linear_swap_action_list_instantiated=false'
require_line "${RECEIPT}" 'outer_40320_view_count_proved=false'
require_line "${RECEIPT}" 'action_list_distinctness_proved=false'
require_line "${RECEIPT}" 'outer_40320_view_minimum_proved=false'
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

require_line "${EVIDENCE}" 'typed_witness_subtype_instantiated=true'
require_line "${EVIDENCE}" 'concrete_linear_swap_action_family_instantiated=true'
require_line "${EVIDENCE}" 'typed_witness_count_20160_proved=false'
require_line "${EVIDENCE}" 'outer_40320_view_count_proved=false'
require_line "${EVIDENCE}" 'concrete_linear_swap_action_list_instantiated=false'
require_line "${EVIDENCE}" 'concrete_canonical_equality_iff_full_declared_orbit_proved=false'
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

[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${SOURCE_REL}")" -eq 7 ]] ||
  fail 'public theorem count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 13 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^(@\[[^]]+\][[:space:]]+)?theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
for definition_name in "${EXPECTED_DEFINITIONS[@]}"; do
  grep -Eq "^def ${definition_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean definition declaration: ${definition_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${definition_name}"
done
require_line "${ROOT}/${SOURCE_REL}" 'abbrev GL4ScanEntry := {code // code ∈ invertibleMatrixCodes}'
require_line "${ROOT}/${SOURCE_REL}" '  , typedWitnessSubtypeInstantiated := true'
require_line "${ROOT}/${SOURCE_REL}" '  , everyScanEntryHasTypedWitness := true'
require_line "${ROOT}/${SOURCE_REL}" '  , everyPredicateWitnessHasEntry := true'
require_line "${ROOT}/${SOURCE_REL}" '  , twoViewsPerEntryInstantiated := true'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteLinearSwapActionFamilyInstantiated := true'
require_line "${ROOT}/${SOURCE_REL}" '  , typedWitnessListInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , typedWitnessCount20160Proved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , importedNativeCensusConsumed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ViewListInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteLinearSwapActionListInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ViewCountProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , actionListDistinctnessProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , outer40320ViewMinimumProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteCanonicalEqualityIffFullDeclaredOrbitProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalTarget03Closed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalParityClosed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , claimReady := false }'

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

negative_dir="$(mktemp -d /tmp/pireus-gl4-action-enumeration-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
set +e
(
  GUARDIAN=/bin/false
  authorize LOCAL_XEON_BUILD \
    "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" \
    "${BUILD_FRAME_SHA256}"
  printf 'LEAN_STARTED\n' >"${negative_dir}/lean-started.txt"
  cd "${ROOT}/formal/lean4"
  lake build SounioPireusGL4ActionEnumeration
) >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
[[ ! -e "${negative_dir}/lean-started.txt" ]] || fail 'Guardian override reached Lean execution'

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusGL4ActionEnumeration 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}: ${build_output}"
if grep -Fq 'SounioPireusGL4ActionEnumeration.lean:' <<<"${build_output}"; then
  fail 'local GL4 action enumeration source warning drift'
fi
build_warning_count="$(count_occurrences '^warning:' "${build_output}")"
(( build_warning_count == 0 || build_warning_count == 5 )) ||
  fail "dependency warning replay drift: ${build_warning_count}"

authorize LOCAL_XEON_DIRECT_TYPECHECK "$(parity_frame "${SOURCE_SHA256}" "${TYPECHECK_COMMAND_SHA256}")" "${TYPECHECK_FRAME_SHA256}"
set +e
typecheck_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4ActionEnumeration.lean 2>&1)"
typecheck_rc=$?
set -e
[[ "${typecheck_rc}" -eq 0 ]] || fail "Lean direct typecheck exit drift: ${typecheck_rc}: ${typecheck_output}"
[[ "$(count_occurrences '^warning:' "${typecheck_output}")" -eq 0 ]] || fail 'Lean direct typecheck warning drift'

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
set +e
audit_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGL4ActionEnumerationAxiomAudit.lean 2>&1)"
audit_rc=$?
set -e
[[ "${audit_rc}" -eq 0 ]] || fail "Lean axiom audit exit drift: ${audit_rc}: ${audit_output}"
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 11 ]] || fail 'axiom-bearing report count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 2 ]] || fail 'axiom-free report count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 11 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 2 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 9 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'

[[ "${python_processes_launched}" -eq 0 ]] || fail 'Python process launch counter drift'
printf 'PIREUS_GL4_ACTION_ENUMERATION_GATE_PASS_PARTIAL=true stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY semantic_authority=Sounio implementation_commit=%s audit_completion_commit=%s typed_subtype=true every_scan_entry=true every_predicate_witness=true two_views_per_entry=true concrete_action_family=true typed_list=false count_20160=false imported_native_census=false outer_40320_list=false concrete_action_list=false outer_40320_count=false action_distinctness=false outer_minimum=false canonical_iff_full_orbit=false target03=false formal_parity_complete=false classical_choice=2 native_decide=0 sorryax=0 python_oracle=E110 python_process_launched=false guardian_current_gate_false_exit=1 spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false slurm_processes_launched=0 u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 llm_role=REVIEW_ONLY llm_confirmed_result=false zai=ERROR_1313 claim_ready=false\n' "${IMPLEMENTATION_COMMIT}" "${AUDIT_COMPLETION_COMMIT}"
