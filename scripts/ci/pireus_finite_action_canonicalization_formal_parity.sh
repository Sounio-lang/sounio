#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusFiniteActionCanonicalization.lean'
AUDIT_REL='formal/lean4/SounioPireusFiniteActionCanonicalizationAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
PARENT_RECEIPT_REL='tools/pireus/operator_orbit_admission_reconstruction.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_operator_orbit_admission_reconstruction_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/operator_orbit_admission_reconstruction_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/finite_action_canonicalization.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/finite_action_canonicalization_v13.formal-parity.txt'

PARENT_COMMIT='cc31e1c2866f6fa69a96e6266217d6fa42d98f90'
SOURCE_COMMIT='7295c79fde2c36d6bd612d2570514ba9fab1442e'
SOURCE_SHA256='cd56a16c585637e75f36a235cff4d6fd3b04a04be36449ad3db17b38ee916aba'
AUDIT_SHA256='b35a14759a33c4f43efe521aa2d6b6306e7c7976c17fe2d9b40c22e9a4270dce'
LAKEFILE_SHA256='2c78c99482ce6c64bdc233d63f07748dd67f0a56fc893e6b73da38f2e05dc74e'
OFFLOAD_LOG_SHA256='19e91cd77ed5549d1364eaa6172d13a230a4d46c441515b515b99c7903efc6da'
PARENT_RECEIPT_SHA256='ab67493583ec87b45bc374a15a83363e746cd5ef8bb8aed72f87ca4bb32177e3'
PARENT_GATE_SHA256='8fc601aead7035631ff3b7c10ee4e08c931db85ed6e2c688014f7e5df2eb9563'
PARENT_EVIDENCE_SHA256='ba29ea873b10fca3c4a8d067dd2c0e2337489ee06ff84b0090be6b97d9ab8059'
RECEIPT_SHA256='b66fdc7e093306f99de74f4680a4c3eb14056bf658c22e982461071a2c0e3873'
EVIDENCE_SHA256='b436a7483541bd50b1acbb2f44a36261e107d45daefa687c53232bb3052c9e41'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusFiniteActionCanonicalization'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusFiniteActionCanonicalizationAxiomAudit.lean'
BUILD_COMMAND_SHA256='c378b441f38522fdca6c5bde4c32dbc73439a9f16505c7d596ec89bc6b7eead9'
AUDIT_COMMAND_SHA256='b3ecac327b7b639c61b23148ff3a3c6c2ee54d34645d01dc551819737f083e69'
BUILD_FRAME_SHA256='a3d9a22f52cf11734ebb05c521faafd8447c5aee1c5fb9915dc20b61fe83efca'
AUDIT_FRAME_SHA256='03d29302dbb8498e4f09045654daf8da74766c21803772d6168999eefe0564f0'
ZERO='0 0 0 0 0 0 0 0'
EXPECTED_THEOREMS=(
  sameOrbit_refl
  sameOrbit_symm
  sameOrbit_trans
  canonicalOption_eq_iff_sameOrbit
  generic_theorem_does_not_close_v13_without_concrete_instantiation
)

fail() {
  printf 'pireus finite-action canonicalization formal parity: FAIL: %s\n' "$*" >&2
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

authorize() {
  local label="$1" frame="$2" expected_sha="$3" decision
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq 82 ]] || fail "Guardian frame words: ${label}"
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
    fail "Guardian decision drift: ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
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
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_COMMIT}" "${SOURCE_COMMIT}" ||
  fail 'parent admission reconstruction does not precede finite-action theorem'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD ||
  fail 'finite-action source commit is not in current history'
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
require_line "${RECEIPT}" "finite_action_source_commit=${SOURCE_COMMIT}"
require_line "${RECEIPT}" 'execution_route=LOCAL_XEON_WORKSPACE_CONTROL'
require_line "${RECEIPT}" 'execution_cpu=INTEL(R)_XEON(R)_GOLD_6526Y'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" 'build_exit_code=0'
require_line "${RECEIPT}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_exit_code=0'
require_line "${RECEIPT}" 'axiom_audit_theorem_reports=5'
require_line "${RECEIPT}" 'axiom_audit_no_axiom_reports=4'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=1'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=1'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=1'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'canonical_minimum_equality_implies_same_orbit_proved=true'
require_line "${RECEIPT}" 'same_orbit_implies_canonical_minimum_equality_proved=true'
require_line "${RECEIPT}" 'generic_canonical_minimum_equality_iff_same_orbit_proved=true'
require_line "${RECEIPT}" 'concrete_gl4_action_laws_instantiated=false'
require_line "${RECEIPT}" 'concrete_gauge_coboundary_action_instantiated=false'
require_line "${RECEIPT}" 'concrete_executed_normalizer_equals_abstract_minimum_proved=false'
require_line "${RECEIPT}" 'concrete_canonical_equality_iff_declared_orbit_proved=false'
require_line "${RECEIPT}" 'formal_target_03_closed=false'
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
require_line "${RECEIPT}" 'u250_pending_reason=SECOND_CARD_NOT_YET_INSTALLED'
require_line "${RECEIPT}" 'u250_enumeration_failure_count=0'
require_line "${RECEIPT}" 'llm_role=REVIEW_ONLY'
require_line "${RECEIPT}" 'llm_confirmed_result=false'
require_line "${RECEIPT}" 'llm_second_opinion_failure=ZAI_PROVIDER_ERROR_1313'
require_line "${EVIDENCE}" 'same_orbit_reflexive_proved=true'
require_line "${EVIDENCE}" 'same_orbit_symmetric_proved=true'
require_line "${EVIDENCE}" 'same_orbit_transitive_proved=true'
require_line "${EVIDENCE}" 'related_orbits_have_identical_membership_proved=true'
require_line "${EVIDENCE}" 'generic_canonical_minimum_equality_iff_same_orbit_proved=true'
require_line "${EVIDENCE}" 'concrete_v13_instantiation_complete=false'
require_line "${EVIDENCE}" 'formal_target_03_closed=false'

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

[[ "$(grep -c '^theorem ' "${ROOT}/${SOURCE_REL}")" -eq 11 ]] || fail 'theorem count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 5 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
require_line "${ROOT}/${SOURCE_REL}" '    system.canonicalOption left = system.canonicalOption right ↔'
require_line "${ROOT}/${SOURCE_REL}" '      system.sameOrbit left right := by'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteCanonicalEqualityIffDeclaredOrbitProved := false'
if grep -Eq '\bsorry\b|sorryAx' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry or sorryAx found in finite-action theorem surface'
fi
(( 2 == 1 + 1 )) || fail 'U250 inventory arithmetic drift'

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusFiniteActionCanonicalization 2>&1)"
build_exit=$?
set -e
[[ "${build_exit}" -eq 0 ]] || fail "fresh Lean build failed: ${build_output}"

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
set +e
audit_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusFiniteActionCanonicalizationAxiomAudit.lean 2>&1)"
audit_exit=$?
set -e
[[ "${audit_exit}" -eq 0 ]] || fail "fresh axiom audit failed: ${audit_output}"
[[ "$(grep -c 'does not depend on any axioms' <<<"${audit_output}")" -eq 4 ]] || fail 'no-axiom report drift'
[[ "$(grep -c 'depends on axioms:' <<<"${audit_output}")" -eq 1 ]] || fail 'axiom-bearing report drift'
[[ "$(grep -c 'propext' <<<"${audit_output}")" -eq 1 ]] || fail 'propext report drift'
[[ "$(grep -c 'Classical.choice' <<<"${audit_output}")" -eq 1 ]] || fail 'Classical.choice report drift'
[[ "$(grep -c 'Quot.sound' <<<"${audit_output}")" -eq 1 ]] || fail 'Quot.sound report drift'
[[ "$(grep -c 'native_decide' <<<"${audit_output}")" -eq 0 ]] || fail 'native_decide report drift'
[[ "$(grep -c 'sorryAx' <<<"${audit_output}")" -eq 0 ]] || fail 'sorryAx report drift'

printf '%s\n' \
  'PIREUS_FINITE_ACTION_CANONICALIZATION_FORMAL_PARITY_PASS=true status=PARTIAL_PASS verification=FRESH_LOCAL_XEON_EXECUTION language=Lean4 role=FORMAL_PARITY theorem_reports=5 no_axiom_reports=4 propext_mentions=1 classical_choice_mentions=1 quot_sound_mentions=1 native_decide_mentions=0 sorryax_mentions=0 same_orbit_equivalence=true generic_canonical_minimum_iff_same_orbit=true generic_scope=EXPLICIT_FINITE_ACTION_SYSTEM_WITH_DECLARED_LAWS concrete_gl4_instantiated=false concrete_gauge_instantiated=false executed_normalizer_bridge=false concrete_v13_canonical_iff_orbit=false formal_target_03_closed=false formal_parity_complete=false semantic_authority=Sounio expected_results_supplied_by_lean=false spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 claim_ready=false'
