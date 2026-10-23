#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
KOKA="${PIREUS_KOKA_BIN:-/workspace/.home/openvscode-server/.local/pireus-toolchains/koka-v3.2.3/bin/koka}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/koka-v6'
RESULT="${BUILD_ROOT}/pireus-operator-lowering-effect-parity"

SOUNIO_REL='stdlib/hardware/pireus/operator_lowering_forge.sio'
FREEZE_REL='tools/pireus/operator_lowering_forge.freeze.v6'
PARITY_OPEN_REL='tools/pireus/operator_lowering_forge.parity-open.v6'
FORMAL_RECEIPT_REL='tools/pireus/operator_lowering_forge.formal-parity.v6'
KOKA_REL='formal/koka/pireus_operator_lowering_effect_parity.kk'
EVIDENCE_REL='tools/pireus/evidence/operator_lowering_forge_v6.koka.txt'
RECEIPT_REL='tools/pireus/operator_lowering_forge.effect-parity.v6'
PARENT_GATE_REL='scripts/ci/pireus_operator_lowering_formal_parity.sh'

FORMAL_GATE_COMMIT='03631c1e4974abc09eb0ba1e3bd27955cfb0c427'
EFFECT_PARITY_COMMIT='91f6eeb8bacc8abd8dc674cd8292372aa07dd36f'
SOUNIO_SHA256='178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0'
SEMANTICS_SHA256='bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
FREEZE_SHA256='973d620f30337378b760aa185ddbe9897bdd82ce18ee9e212756f519d1ed7181'
PARITY_OPEN_SHA256='4dbd89c5a18a2771bda46674b4ad93849e9f0ea160c7c9f42ce511307c7a6eba'
FORMAL_RECEIPT_SHA256='31f229664a627134898d476d3e5374cd7458401420f49316e129ea951386d169'
KOKA_SHA256='5093cf513e5a442aec65db6e3a3acd00606db7a0529bb37781de88d43eec464e'
EVIDENCE_SHA256='e6de339d194743032d0ba3afee9d9e7decf99c333e8266e783794cb9fbf5bbe3'
RECEIPT_SHA256='9deba7c7f66d9e75e82dbfce7b0ed65e94713f602d0cce6a8190218c5b32629f'
PARENT_GATE_SHA256='d1002b6637a3006ace80f2efc97ec6866bc202e837ab6739f671a7200fb94f33'
KOKA_BINARY_SHA256='5268748ed5082f3693ddf9fa40e560020aa16b6be6bd52b86c97ce5435b24cba'
TOOLCHAIN_SHA256='273f70c80ed71dcfbe1ee077607ec435d8791e59032cc13e30e479fd25995332'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='02b7e081299a2dd97d22120a440d745ce853f41ba31bc4d58a667ea275260022'
RESULT_SHA256='9b8bbc68735584ebc23a11400329a24653ee20c21fcc427a224d4835656206ba'
PREEXEC_FRAME_SHA256='d7d80ca420b13f835013c68f48173ae9aba3fdeb13ee5e305316f703d8085f8e'
SEAL_FRAME_SHA256='b19762cff29c0c4b86874ba84644b30f2c73ae664ad36a6775539a3083d8d0a8'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator lowering effect parity: FAIL: %s\n' "$*" >&2
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
  [[ "$(sha_file "${path}")" == "${expected}" ]] ||
    fail "hash drift: ${path}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" ||
    fail "missing exact line in ${path}: ${expected}"
}

parity_frame() {
  local stage="$1" action="$2" policy="$3" semantic_write="$4"
  local expected_write="$5" parity_valid="$6" review_promoted="$7"
  local parent_hash="$8" result_hash="$9" result_limbs="${ZERO}"
  if [[ "${result_hash}" != zero ]]; then
    result_limbs="$(sha_limbs "${result_hash}")"
  fi
  printf '9020 %s %s 3 3 %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${policy}" "${semantic_write}" \
    "${expected_write}" "${parity_valid}" "${review_promoted}" \
    "$(sha_limbs "${KOKA_SHA256}")" \
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

deny_without_dispatch() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4"
  local decision rc
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
}

require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${FORMAL_RECEIPT_REL}" "${FORMAL_RECEIPT_SHA256}"
require_hash "${ROOT}/${KOKA_REL}" "${KOKA_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
[[ -x "${KOKA}" ]] || fail 'Koka 3.2.3 executable unavailable'
[[ "$(sha_file "${KOKA}")" == "${KOKA_BINARY_SHA256}" ]] ||
  fail 'Koka executable hash drift'

git -C "${ROOT}" merge-base --is-ancestor \
  "${FORMAL_GATE_COMMIT}" "${EFFECT_PARITY_COMMIT}" ||
  fail 'effect parity predates the formal parity gate'
git -C "${ROOT}" merge-base --is-ancestor "${EFFECT_PARITY_COMMIT}" HEAD ||
  fail 'effect parity commit missing from current history'
[[ "$(git -C "${ROOT}" show "${EFFECT_PARITY_COMMIT}:${KOKA_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${KOKA_SHA256}" ]] || fail 'committed Koka source drift'
[[ "$(git -C "${ROOT}" show "${EFFECT_PARITY_COMMIT}:${EVIDENCE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${EVIDENCE_SHA256}" ]] || fail 'committed effect evidence drift'
[[ "$(git -C "${ROOT}" show "${EFFECT_PARITY_COMMIT}:${RECEIPT_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${RECEIPT_SHA256}" ]] || fail 'committed effect receipt drift'
[[ "$(git -C "${ROOT}" show "${FORMAL_GATE_COMMIT}:${PARENT_GATE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${PARENT_GATE_SHA256}" ]] || fail 'committed formal gate drift'

require_line "${ROOT}/${FREEZE_REL}" "module_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'koka_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'axiom_closure=EMPTY'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'effect_parity_complete=false'

for declaration in \
  'effect forge-authority' \
  'type target-envelope' \
  'type seed-kind' \
  'type fold-order' \
  'type obligation-kind' \
  'type ledger-snapshot' \
  'fun frozen-authority' \
  'fun run-parity'; do
  grep -Fq -- "${declaration}" "${ROOT}/${KOKA_REL}" ||
    fail "missing Koka declaration: ${declaration}"
done
[[ "$(grep -c '^  require(' "${ROOT}/${KOKA_REL}")" -eq 38 ]] ||
  fail 'Koka executable check census drift'

koka_version="$("${KOKA}" --version --console=raw | sed -n '1p')"
gcc_version="$(gcc --version | sed -n '1p')"
toolchain_record="koka=${koka_version} koka_binary_sha256=${KOKA_BINARY_SHA256} cc=${gcc_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'Koka toolchain drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware drift'
command_record='/workspace/.home/openvscode-server/.local/pireus-toolchains/koka-v3.2.3/bin/koka -O2 --builddir=/workspace/.home/openvscode-server/.cache/pireus/koka-v6/build -o /workspace/.home/openvscode-server/.cache/pireus/koka-v6/pireus-operator-lowering-effect-parity formal/koka/pireus_operator_lowering_effect_parity.kk && chmod 0755 /workspace/.home/openvscode-server/.cache/pireus/koka-v6/pireus-operator-lowering-effect-parity && /workspace/.home/openvscode-server/.cache/pireus/koka-v6/pireus-operator-lowering-effect-parity'
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'command drift'

set +e
invalid_hash_output="$(sha_limbs 'not-a-sha256' 2>&1)"
invalid_hash_rc=$?
set -e
[[ "${invalid_hash_rc}" -eq 1 ]] ||
  fail 'malformed SHA-256 text did not fail closed'
[[ "${invalid_hash_output}" == 'pireus operator lowering effect parity: FAIL: invalid sha256 digest: not-a-sha256' ]] ||
  fail 'malformed SHA-256 refusal drift'
printf 'GUARDIAN_DISPATCH label=MALFORMED_SHA256 process_launched=false\n'

wrong_parent='0d69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
deny_without_dispatch PARENT_LAUNDERING \
  "$(parity_frame 3 4 1 0 0 0 0 "${wrong_parent}" zero)" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN'
deny_without_dispatch KOKA_SEMANTIC_WRITE \
  "$(parity_frame 3 4 1 1 0 0 0 "${SEMANTICS_SHA256}" zero)" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch KOKA_EXPECTED_RESULT_WRITE \
  "$(parity_frame 3 4 1 0 1 0 0 "${SEMANTICS_SHA256}" zero)" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch REVIEW_PROMOTION \
  "$(parity_frame 3 4 1 0 0 0 1 "${SEMANTICS_SHA256}" zero)" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_MISSING \
  "$(parity_frame 3 4 0 0 0 0 0 "${SEMANTICS_SHA256}" zero)" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_TIMEOUT \
  "$(parity_frame 3 4 2 0 0 0 0 "${SEMANTICS_SHA256}" zero)" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_ERROR \
  "$(parity_frame 3 4 3 0 0 0 0 "${SEMANTICS_SHA256}" zero)" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'

parent_gate_output="$("${ROOT}/${PARENT_GATE_REL}")"
printf '%s\n' "${parent_gate_output}" | grep -Fqx -- \
  'pireus operator lowering formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY obligations=6/6 candidates=1120 program_classes=560 target_envelopes=4 admitted_lowerings=0 formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED selected_candidate=-1 claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'Lean formal parity gate terminal marker drift'

authorize PREEXEC \
  "$(parity_frame 3 4 1 0 0 0 0 "${SEMANTICS_SHA256}" zero)" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

mkdir -p "${BUILD_ROOT}/build"
"${KOKA}" -O2 \
  --builddir="${BUILD_ROOT}/build" \
  -o "${RESULT}" \
  "${ROOT}/${KOKA_REL}"
chmod 0755 "${RESULT}"
require_hash "${RESULT}" "${RESULT_SHA256}"

program_output="$("${RESULT}")"
expected_output="$(printf '%s\n' \
  'schema=pireus-operator-lowering-effect-parity-evidence-v6' \
  'producing_language=Koka' \
  'producing_role=EFFECT_PARITY' \
  "sounio_source_sha256=${SOUNIO_SHA256}" \
  "sounio_semantics_sha256=${SEMANTICS_SHA256}" \
  "formal_parity_receipt_sha256=${FORMAL_RECEIPT_SHA256}" \
  'authority_stage_transition_effects=true' \
  'read_only_atlas_effect=true' \
  'frozen_obligation_ledger_effect=true' \
  'seed_kind_nonconversion_effect=true' \
  'ordered_fold_noncollapse_effect=true' \
  'canonical_target_nonexecution_effect=true' \
  'claim_boundary_refusal_effect=true' \
  'checked_obligations=7' \
  'formal_proofs=0' \
  'checks=38' \
  'adversarial_checks=30' \
  'candidate_cells=1120' \
  'effect_memory_unresolved=1120' \
  'admitted_lowerings=0' \
  'semantic_write=false' \
  'expected_result_write=false' \
  'target_processes_launched=0' \
  'selected_candidate=-1' \
  'claim_ready=false')"
[[ "${program_output}" == "${expected_output}" ]] ||
  fail 'Koka effect parity output drift'

authorize SEAL \
  "$(parity_frame 4 8 1 0 0 1 0 "${SEMANTICS_SHA256}" "${RESULT_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

require_line "${ROOT}/${RECEIPT_REL}" 'status=EFFECT_PARITY_COMPLETE'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_language=Koka'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_role=EFFECT_PARITY'
require_line "${ROOT}/${RECEIPT_REL}" "sounio_source_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "formal_parity_receipt_sha256=${FORMAL_RECEIPT_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "result_sha256=${RESULT_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "command_sha256=${COMMAND_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "preexec_frame_sha256=${PREEXEC_FRAME_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "seal_frame_sha256=${SEAL_FRAME_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" 'unsealed_probe_count=2'
require_line "${ROOT}/${RECEIPT_REL}" 'unsealed_probes_authoritative=false'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'effect_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'effect_memory_unresolved=1120'
require_line "${ROOT}/${RECEIPT_REL}" 'effect_memory_discharged_by_koka=0'
require_line "${ROOT}/${RECEIPT_REL}" 'material_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'promotable_to_semantic_authority=false'
require_line "${ROOT}/${RECEIPT_REL}" 'semantic_write=false'
require_line "${ROOT}/${RECEIPT_REL}" 'expected_result_write=false'
require_line "${ROOT}/${RECEIPT_REL}" 'target_processes_launched=0'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'

for obligation in \
  authority_stage_transition_effects \
  read_only_atlas_effect \
  frozen_obligation_ledger_effect \
  seed_kind_nonconversion_effect \
  ordered_fold_noncollapse_effect \
  canonical_target_nonexecution_effect \
  claim_boundary_refusal_effect; do
  grep -Eq "^obligation_[0-9][0-9]=${obligation} status=CHECKED$" \
    "${ROOT}/${EVIDENCE_REL}" || fail "unchecked Koka obligation: ${obligation}"
done
require_line "${ROOT}/${EVIDENCE_REL}" 'precanonical_attempts_used_as_evidence=false'
require_line "${ROOT}/${EVIDENCE_REL}" 'canonical_attempt_01_process_launched=false'
require_line "${ROOT}/${EVIDENCE_REL}" 'successful_attempt=CANONICAL_02'

printf '%s\n' \
  'pireus operator lowering effect parity: STAGE_REACHED_NOT_A_CLAIM gate_mode=CONTENT_ADDRESSED_REPLAY stage=PARITY_OPEN language=Koka role=EFFECT_PARITY effect_scope=AUTHORITY_AND_EFFECT_TOPOLOGY_ONLY effect_checks=7/7 checks=38 adversarial=30 candidates=1120 effect_memory_unresolved=1120 effect_memory_discharged_by_koka=0 admitted_lowerings=0 formal=COMPLETE effect=COMPLETE material=OPEN_NOT_EXECUTED selected_candidate=-1 claim_ready=false python_process_launched=false rust_process_launched=false'
