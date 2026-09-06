#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusGaugeCoboundaryAction.lean'
AUDIT_REL='formal/lean4/SounioPireusGaugeCoboundaryActionAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
PARENT_RECEIPT_REL='tools/pireus/finite_action_canonicalization.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_finite_action_canonicalization_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/finite_action_canonicalization_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/gauge_coboundary_action.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/gauge_coboundary_action_v13.formal-parity.txt'

PARENT_COMMIT='04cf18cc2a8a074d4511c49d8be035d4bb9b9bf2'
SOURCE_COMMIT='07309cacbb6a57b6cd5120f29adf44bb0791d399'
SOURCE_SHA256='bf25ef66f9b5fab5f1c08e7aa16fb3875bc8ee990a7615d20ea87d4357db5901'
AUDIT_SHA256='1c5840354677c9bc1c63a94a44157409ca51337617fc5405271c4864a6c7f6a9'
LAKEFILE_SHA256='d20b5436676a961f54809d573153a68dbf43cc472649462cbfad1675d9f0700c'
OFFLOAD_LOG_SHA256='3852b7b67fbc2dba8c93dc74f8ff64ab8bf54376b522ee23e9f2da3856b2f89e'
PARENT_RECEIPT_SHA256='b66fdc7e093306f99de74f4680a4c3eb14056bf658c22e982461071a2c0e3873'
PARENT_GATE_SHA256='1957e3e04b5d2a2e6a569d51aebacc2fae63ca891058d95952c63cc0fd04de58'
PARENT_EVIDENCE_SHA256='b436a7483541bd50b1acbb2f44a36261e107d45daefa687c53232bb3052c9e41'
RECEIPT_SHA256='fa1adc34ad7517c0ccc788b378a3f4bc7ff764cfb7661be9a03687e25207b643'
EVIDENCE_SHA256='39462e6a416d3284ceac70141d2ba659538c7adbe177126f7b9098bd1ab1547b'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusGaugeCoboundaryAction'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusGaugeCoboundaryActionAxiomAudit.lean'
BUILD_COMMAND_SHA256='326dc6e54a9cae7329bceb38b9b8982b398c01c0e9a8a873492c5c01438febe8'
AUDIT_COMMAND_SHA256='b81ec15ac66f801bc96bdab7705e4b0bf9d08cf4c725317574b5b768722dc700'
BUILD_FRAME_SHA256='74860509fcf4478375bcb7077a61f22c7aa15d115583a0af40db81ff1c724b02'
AUDIT_FRAME_SHA256='fb1b6230fa815658dfe26cbdf23fa2bfd969b75729b28510eef595de6264cb0b'
ZERO='0 0 0 0 0 0 0 0'
EXPECTED_THEOREMS=(
  gauge_value_xor
  gauge_coboundary_xor
  gauge_action_identity
  gauge_action_compose
  gauge_action_inverse
  gauge_word_enumeration_has_exactly_2048_actions
  gauge_action_system_satisfies_concrete_finite_action_laws
  gauge_action_progress_does_not_close_v13_target03
)

fail() {
  printf 'pireus gauge coboundary action formal parity: FAIL: %s\n' "$*" >&2
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
  fail 'parent finite-action theorem does not precede gauge action'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD ||
  fail 'gauge action source commit is not in current history'
require_committed_hash "${PARENT_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${OFFLOAD_LOG_REL}" "${OFFLOAD_LOG_SHA256}"

RECEIPT="${ROOT}/${RECEIPT_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'generic_canonical_minimum_equality_iff_same_orbit_proved=true'
require_line "${ROOT}/${PARENT_RECEIPT_REL}" 'concrete_gauge_coboundary_action_instantiated=false'
require_line "${RECEIPT}" 'status=PARTIAL_PASS'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=Lean4'
require_line "${RECEIPT}" 'language_role=FORMAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "gauge_action_source_commit=${SOURCE_COMMIT}"
require_line "${RECEIPT}" 'execution_route=LOCAL_XEON_WORKSPACE_CONTROL'
require_line "${RECEIPT}" 'execution_cpu=INTEL(R)_XEON(R)_GOLD_6526Y'
require_line "${RECEIPT}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${RECEIPT}" "build_preexec_frame_sha256=${BUILD_FRAME_SHA256}"
require_line "${RECEIPT}" 'build_exit_code=0'
require_line "${RECEIPT}" 'lean_kernel_typechecked_gauge_theorems=true'
require_line "${RECEIPT}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${RECEIPT}" "axiom_audit_preexec_frame_sha256=${AUDIT_FRAME_SHA256}"
require_line "${RECEIPT}" 'axiom_audit_exit_code=0'
require_line "${RECEIPT}" 'axiom_audit_theorem_reports=8'
require_line "${RECEIPT}" 'axiom_audit_no_axiom_reports=1'
require_line "${RECEIPT}" 'axiom_audit_propext_mentions=7'
require_line "${RECEIPT}" 'axiom_audit_classical_choice_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_quot_sound_mentions=7'
require_line "${RECEIPT}" 'axiom_audit_native_decide_mentions=0'
require_line "${RECEIPT}" 'axiom_audit_sorryax_mentions=0'
require_line "${RECEIPT}" 'gauge_bits=11'
require_line "${RECEIPT}" 'gauge_word_count=2048'
require_line "${RECEIPT}" 'gauge_action_carrier_word_count=2048'
require_line "${RECEIPT}" 'gauge_vector_count=11'
require_line "${RECEIPT}" 'gauge_cell_domain=FIN_16_X_FIN_16'
require_line "${RECEIPT}" 'gauge_sign_table_cell_count=256'
require_line "${RECEIPT}" 'gauge_action_law_proof_surface=HASH_PINNED_LEAN_THEOREM_STATEMENTS_FRESHLY_BUILT_AND_AXIOM_AUDITED'
require_line "${RECEIPT}" 'gauge_vectors_exact_nonbasis_complement_proved=true'
require_line "${RECEIPT}" 'gauge_value_zero_proved=true'
require_line "${RECEIPT}" 'gauge_value_xor_proved=true'
require_line "${RECEIPT}" 'gauge_coboundary_zero_proved=true'
require_line "${RECEIPT}" 'gauge_coboundary_xor_proved=true'
require_line "${RECEIPT}" 'gauge_action_identity_proved=true'
require_line "${RECEIPT}" 'gauge_action_compose_proved=true'
require_line "${RECEIPT}" 'gauge_action_inverse_proved=true'
require_line "${RECEIPT}" 'gauge_action_membership_closure_instantiated=true'
require_line "${RECEIPT}" 'concrete_gauge_coboundary_action_instantiated=true'
require_line "${RECEIPT}" 'gauge_action_laws_require_faithfulness=false'
require_line "${RECEIPT}" 'gauge_action_function_faithfulness_proved=false'
require_line "${RECEIPT}" 'distinct_gauge_coboundary_functions_proved=false'
require_line "${RECEIPT}" 'gauge_table_lawful_minimum_instantiated=false'
require_line "${RECEIPT}" 'tree_section_equals_gauge_orbit_minimum_proved=false'
require_line "${RECEIPT}" 'concrete_input_swap_action_instantiated=false'
require_line "${RECEIPT}" 'concrete_gl4_action_laws_instantiated=false'
require_line "${RECEIPT}" 'concrete_v13_finite_action_instantiation_complete=false'
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
require_line "${EVIDENCE}" 'gauge_word_count=2048'
require_line "${EVIDENCE}" 'gauge_action_carrier_word_count=2048'
require_line "${EVIDENCE}" 'cell_domain=Fin_16_x_Fin_16'
require_line "${EVIDENCE}" 'gauge_action_identity_proved=true'
require_line "${EVIDENCE}" 'gauge_action_compose_proved=true'
require_line "${EVIDENCE}" 'gauge_action_inverse_proved=true'
require_line "${EVIDENCE}" 'concrete_gauge_coboundary_action_instantiated=true'
require_line "${EVIDENCE}" 'lean_kernel_typechecked_gauge_theorems=true'
require_line "${EVIDENCE}" 'gauge_action_laws_require_faithfulness=false'
require_line "${EVIDENCE}" 'gauge_action_function_faithfulness_proved=false'
require_line "${EVIDENCE}" 'distinct_gauge_coboundary_functions_proved=false'
require_line "${EVIDENCE}" 'u250_declared=2'
require_line "${EVIDENCE}" 'u250_installed=1'
require_line "${EVIDENCE}" 'u250_pending_installation=1'

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
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 8 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
require_line "${ROOT}/${SOURCE_REL}" 'abbrev Cell := Fin 16 × Fin 16'
require_line "${ROOT}/${SOURCE_REL}" '  { actions := List.finRange (2 ^ gaugeBits)'
require_line "${ROOT}/${SOURCE_REL}" 'theorem gauge_action_identity (table : SignTable) :'
require_line "${ROOT}/${SOURCE_REL}" '    gaugeAct zeroGauge table = table := by'
require_line "${ROOT}/${SOURCE_REL}" 'theorem gauge_action_compose'
require_line "${ROOT}/${SOURCE_REL}" '    gaugeAct (composeGauge outer inner) table ='
require_line "${ROOT}/${SOURCE_REL}" '      gaugeAct outer (gaugeAct inner table) := by'
require_line "${ROOT}/${SOURCE_REL}" 'theorem gauge_action_inverse (word : GaugeWord) (table : SignTable) :'
require_line "${ROOT}/${SOURCE_REL}" '    gaugeAct (inverseGauge word) (gaugeAct word table) = table := by'
require_line "${ROOT}/${SOURCE_REL}" 'theorem gauge_word_enumeration_has_exactly_2048_actions :'
require_line "${ROOT}/${SOURCE_REL}" '    gaugeActionSystem.actions.length = 2048 := by'
require_line "${ROOT}/${SOURCE_REL}" '  , gaugeCoboundaryActionInstantiated := true'
require_line "${ROOT}/${SOURCE_REL}" '  , gaugeTableLawfulMinimumInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , treeSectionEqualsGaugeOrbitMinimumProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteGL4ActionInstantiated := false'
require_line "${ROOT}/${SOURCE_REL}" '  , concreteExecutedNormalizerEqualsAbstractMinimumProved := false'
require_line "${ROOT}/${SOURCE_REL}" '  , formalTarget03Closed := false'
require_line "${ROOT}/${SOURCE_REL}" '  , claimReady := false }'
if grep -Eq '\bsorry\b|sorryAx|native_decide' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry, sorryAx, or native_decide found in gauge action surface'
fi
(( 2048 == 2 ** 11 )) || fail 'gauge cardinality arithmetic drift'
(( 256 == 16 * 16 )) || fail 'sign-table cell arithmetic drift'
(( 2 == 1 + 1 )) || fail 'U250 inventory arithmetic drift'

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusGaugeCoboundaryAction 2>&1)"
build_exit=$?
set -e
[[ "${build_exit}" -eq 0 ]] || fail "fresh Lean build failed: ${build_output}"

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
set +e
audit_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusGaugeCoboundaryActionAxiomAudit.lean 2>&1)"
audit_exit=$?
set -e
[[ "${audit_exit}" -eq 0 ]] || fail "fresh axiom audit failed: ${audit_output}"
[[ "$(count_occurrences 'SounioPireusGaugeCoboundaryAction.' "${audit_output}")" -eq 8 ]] || fail 'theorem report drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 1 ]] || fail 'no-axiom report drift'
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 7 ]] || fail 'axiom-bearing report drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 7 ]] || fail 'propext report drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 0 ]] || fail 'Classical.choice report drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 7 ]] || fail 'Quot.sound report drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide report drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx report drift'

printf '%s\n' \
  'PIREUS_GAUGE_COBOUNDARY_ACTION_FORMAL_PARITY_PASS=true status=PARTIAL_PASS verification=FRESH_LOCAL_XEON_EXECUTION language=Lean4 role=FORMAL_PARITY theorem_reports=8 no_axiom_reports=1 propext_mentions=7 classical_choice_mentions=0 quot_sound_mentions=7 native_decide_mentions=0 sorryax_mentions=0 gauge_carrier_words=2048 cell_domain=FIN_16_X_FIN_16 gauge_action_laws=true concrete_gauge_instantiated=true gauge_action_faithfulness=false distinct_gauge_coboundary_functions=false tree_section_minimum=false concrete_input_swap=false concrete_gl4=false executed_normalizer_bridge=false concrete_v13_canonical_iff_orbit=false formal_target_03_closed=false formal_parity_complete=false semantic_authority=Sounio expected_results_supplied_by_lean=false spark_route=KUBERNETES_ONLY spark_nodes_used=false dgx_nodes_used=false u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_pending_reason=SECOND_CARD_NOT_YET_INSTALLED u250_enumeration_failures=0 claim_ready=false'
