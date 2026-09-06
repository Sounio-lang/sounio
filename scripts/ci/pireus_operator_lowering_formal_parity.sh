#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

SOUNIO_REL='stdlib/hardware/pireus/operator_lowering_forge.sio'
FREEZE_REL='tools/pireus/operator_lowering_forge.freeze.v6'
PARITY_OPEN_REL='tools/pireus/operator_lowering_forge.parity-open.v6'
LEAN_REL='formal/lean4/SounioPireusOperatorLoweringForge.lean'
AXIOM_AUDIT_REL='formal/lean4/SounioPireusOperatorLoweringForgeAxiomAudit.lean'
LAKE_REL='formal/lean4/lakefile.lean'
EVIDENCE_REL='tools/pireus/evidence/operator_lowering_forge_v6.lean.txt'
RECEIPT_REL='tools/pireus/operator_lowering_forge.formal-parity.v6'
OLEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorLoweringForge.olean'
ILEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorLoweringForge.ilean'
PARENT_GATE_REL='scripts/ci/pireus_operator_lowering_forge.sh'

PARITY_OPEN_COMMIT='c15e34c51cd40c696ff971dc26b59631abe0263d'
FORMAL_PARITY_COMMIT='f3f4f34c161c7eed2c3445b7d52634829ef186ae'
SOUNIO_SHA256='178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0'
SEMANTICS_SHA256='bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
FREEZE_SHA256='973d620f30337378b760aa185ddbe9897bdd82ce18ee9e212756f519d1ed7181'
PARITY_OPEN_SHA256='4dbd89c5a18a2771bda46674b4ad93849e9f0ea160c7c9f42ce511307c7a6eba'
LEAN_SHA256='f52d2e630c22bc51a7cfdc350670f716a57bf2416c680999efb2bbde308a2a86'
AXIOM_AUDIT_SHA256='11d14e62ee33151f3121cf15395c07635e8b17f0e790e54ef612ec1cfc9b584c'
LAKE_SHA256='019126978985264fa2aa11ffb85c74ecdb7a4e70c471a73f5cd6a7d37c7c7b38'
EVIDENCE_SHA256='cc1e98d43b244e3e7fb237955e6e8fe9a46e76bba48d23222120a3ebbe871af2'
RECEIPT_SHA256='31f229664a627134898d476d3e5374cd7458401420f49316e129ea951386d169'
OLEAN_SHA256='ebe944c4f2e11c3a896784c6507d8fb489dbaa9b04d05d7186d2e78014ae93ac'
ILEAN_SHA256='518326903d22a34b311548f90549eae30ef6f776f0f602b091629b599d41e139'
PARENT_GATE_SHA256='94a8f841c022e7ebf793a6500b901f16f553ab590d203a873a6e27af3802b958'
TOOLCHAIN_SHA256='03526d62a2416b90e4cad0c369f73bbed9f38f700e0182f220f511235548bf63'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='b0b192699732f0e3ef17ce82ce3feb767ecaf92590d2e4b43327db1dff3cae37'
AXIOM_AUDIT_COMMAND_SHA256='f6411cc99db5b0c6cbcbd6d08291ccb0b2c9595f540a866f1c77e053c75d7387'
PREEXEC_FRAME_SHA256='1f3b36b69a8dd769c93604d86a3aa866670762e53dc53252b45eac95bf7e4955'
SEAL_FRAME_SHA256='301e2478c9378242421efca35bb8b0514a11883320276dea800db1bb93d63dc6'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator lowering formal parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

sha_limbs() {
  local hex="$1" out='' i part
  [[ "${#hex}" -eq 64 && "${hex}" =~ ^[0-9a-f]{64}$ ]] ||
    fail "invalid sha256 digest: ${hex}"
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

require_hash() {
  local path="$1" expected="$2"
  [[ -f "${path}" ]] || fail "missing file: ${path}"
  [[ "$(sha_file "${path}")" == "${expected}" ]] || fail "hash drift: ${path}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" ||
    fail "missing exact line in ${path}: ${expected}"
}

parity_frame() {
  local action="$1" semantic_write="$2" parent_hash="$3" result_hash="$4"
  local stage=3 receipt_valid=0 result_limbs="${ZERO}"
  if [[ "${action}" == 8 ]]; then
    stage=4
    receipt_valid=1
    result_limbs="$(sha_limbs "${result_hash}")"
  fi
  printf '9020 %s %s 2 2 1 %s 0 %s 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${semantic_write}" "${receipt_valid}" \
    "$(sha_limbs "${LEAN_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "${result_limbs}" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s\n' \
    "${label}" "${expected_sha}" "${decision}"
}

require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${LEAN_REL}" "${LEAN_SHA256}"
require_hash "${ROOT}/${AXIOM_AUDIT_REL}" "${AXIOM_AUDIT_SHA256}"
require_hash "${ROOT}/${LAKE_REL}" "${LAKE_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

git -C "${ROOT}" merge-base --is-ancestor \
  "${PARITY_OPEN_COMMIT}" "${FORMAL_PARITY_COMMIT}" ||
  fail 'formal parity predates PARITY_OPEN'
git -C "${ROOT}" merge-base --is-ancestor "${FORMAL_PARITY_COMMIT}" HEAD ||
  fail 'formal parity commit missing from current history'
[[ "$(git -C "${ROOT}" show "${FORMAL_PARITY_COMMIT}:${LEAN_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${LEAN_SHA256}" ]] || fail 'committed Lean source drift'
[[ "$(git -C "${ROOT}" show "${FORMAL_PARITY_COMMIT}:${AXIOM_AUDIT_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${AXIOM_AUDIT_SHA256}" ]] || fail 'committed Lean axiom audit drift'
[[ "$(git -C "${ROOT}" show "${FORMAL_PARITY_COMMIT}:${RECEIPT_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${RECEIPT_SHA256}" ]] || fail 'committed formal receipt drift'

if grep -Eq '\b(sorry|axiom|native_decide)\b' "${ROOT}/${LEAN_REL}"; then
  fail 'Lean parity contains sorry, axiom, or native_decide'
fi
for theorem in \
  formal_parity_summary_matches_frozen_sounio \
  candidate_index_roundtrip_and_grammar_cardinality \
  program_serialization_quotient_exact \
  machine_envelope_partition_exact \
  parent_representative_lineage_bound \
  residual_seed_partition_exact \
  obligation_ledger_and_no_admission; do
  grep -Fq "theorem ${theorem}" "${ROOT}/${LEAN_REL}" ||
    fail "missing theorem: ${theorem}"
done

require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'lean_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${RECEIPT_REL}" 'status=FORMAL_PARITY_COMPLETE'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_language=Lean_4'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_role=FORMAL_PARITY'
require_line "${ROOT}/${RECEIPT_REL}" "sounio_source_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "result_sha256=${OLEAN_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" 'formal_obligations=6'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_obligations_discharged=6'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'proof_reduction=KERNEL_DECIDE'
require_line "${ROOT}/${RECEIPT_REL}" 'axiom_audit_theorems=7'
require_line "${ROOT}/${RECEIPT_REL}" 'axiom_closure=EMPTY'
require_line "${ROOT}/${RECEIPT_REL}" 'effect_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'material_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'semantic_write=false'
require_line "${ROOT}/${RECEIPT_REL}" 'expected_result_write=false'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'

for expected in \
  'discharged_obligations=6' \
  'operator_classes=14' \
  'parent_child_universe=48' \
  'target_envelopes=4' \
  'target_population_each=280' \
  'candidate_cells=1120' \
  'program_classes=560' \
  'lowering_seeds=560' \
  'primitive_seeds=420' \
  'fabric_seeds=140' \
  'operator_seeds=0' \
  'unresolved_target_obligations=10080' \
  'admitted_lowerings=0' \
  'selected_candidate=-1' \
  'claim_ready=false'; do
  require_line "${ROOT}/${EVIDENCE_REL}" "${expected}"
done

lean_version="$(lean --version | sed -n '1p')"
lake_version="$(cd "${ROOT}/formal/lean4" && lake --version)"
toolchain_record="lean=${lean_version} lake=${lake_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'Lean toolchain drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware drift'
command_record='cd formal/lean4 && lake build SounioPireusOperatorLoweringForge'
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'command drift'
axiom_audit_command_record='cd formal/lean4 && lake env lean SounioPireusOperatorLoweringForgeAxiomAudit.lean'
[[ "$(sha_text "${axiom_audit_command_record}")" == "${AXIOM_AUDIT_COMMAND_SHA256}" ]] ||
  fail 'axiom audit command drift'

set +e
invalid_hash_output="$(sha_limbs 'not-a-sha256' 2>&1)"
invalid_hash_rc=$?
set -e
[[ "${invalid_hash_rc}" -eq 1 ]] || fail 'malformed SHA-256 text did not fail closed'
[[ "${invalid_hash_output}" == 'pireus operator lowering formal parity: FAIL: invalid sha256 digest: not-a-sha256' ]] ||
  fail 'malformed SHA-256 refusal drift'
printf 'GUARDIAN_DISPATCH label=MALFORMED_SHA256 process_launched=false\n'

authorize PREEXEC \
  "$(parity_frame 4 0 "${SEMANTICS_SHA256}" zero)" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

wrong_parent="0d69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1"
wrong_parent_frame="$(parity_frame 4 0 "${wrong_parent}" zero)"
set +e
wrong_parent_decision="$(printf '%s\n' "${wrong_parent_frame}" | "${GUARDIAN}")"
wrong_parent_rc=$?
set -e
[[ "${wrong_parent_rc}" -eq 117 ]] || fail 'parent laundering did not fail closed'
[[ "${wrong_parent_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN' ]] ||
  fail 'parent laundering decision drift'
printf 'GUARDIAN_DISPATCH label=PARENT_LAUNDERING process_launched=false\n'

semantic_write_frame="$(parity_frame 4 1 "${SEMANTICS_SHA256}" zero)"
set +e
semantic_write_decision="$(printf '%s\n' "${semantic_write_frame}" | "${GUARDIAN}")"
semantic_write_rc=$?
set -e
[[ "${semantic_write_rc}" -eq 113 ]] || fail 'Lean semantic write did not fail closed'
[[ "${semantic_write_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN' ]] ||
  fail 'Lean semantic write decision drift'
printf 'GUARDIAN_DISPATCH label=LEAN_SEMANTIC_WRITE process_launched=false\n'

parent_gate_output="$("${ROOT}/${PARENT_GATE_REL}")"
printf '%s\n' "${parent_gate_output}" | grep -Fqx -- \
  'pireus operator lowering forge: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN freeze_scope=BOUNDED_ATLAS_AND_RESIDUAL_TAXONOMY_NOT_LOWERING_SUCCESS language=Sounio operator_classes=14 candidates=1120 program_classes=560 target_envelopes=4 residuals=1120 unresolved=10080 admitted_lowerings=0 selected_candidate=-1 formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'Sounio authority gate terminal marker drift'

(cd "${ROOT}/formal/lean4" && lake build SounioPireusOperatorLoweringForge)
require_hash "${ROOT}/${OLEAN_REL}" "${OLEAN_SHA256}"
require_hash "${ROOT}/${ILEAN_REL}" "${ILEAN_SHA256}"

axiom_audit_output="$(
  cd "${ROOT}/formal/lean4" &&
    lake env lean SounioPireusOperatorLoweringForgeAxiomAudit.lean
)"
[[ "$(printf '%s\n' "${axiom_audit_output}" | wc -l)" -eq 7 ]] ||
  fail 'Lean axiom audit theorem count drift'
[[ "$(printf '%s\n' "${axiom_audit_output}" | grep -Fc ' does not depend on any axioms')" -eq 7 ]] ||
  fail 'Lean axiom closure is not empty'
printf 'LEAN_AXIOM_AUDIT theorems=7 axiom_closure=EMPTY\n'

authorize SEAL \
  "$(parity_frame 8 0 "${SEMANTICS_SHA256}" "${OLEAN_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

printf '%s\n' \
  'pireus operator lowering formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY obligations=6/6 candidates=1120 program_classes=560 target_envelopes=4 admitted_lowerings=0 formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED selected_candidate=-1 claim_ready=false python_process_launched=false rust_process_launched=false'
