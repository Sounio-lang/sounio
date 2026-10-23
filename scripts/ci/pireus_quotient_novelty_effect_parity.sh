#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
KOKA="${PIREUS_KOKA_BIN:-/workspace/.home/openvscode-server/.local/pireus-toolchains/koka-v3.2.3/bin/koka}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/koka-v5'
RESULT="${BUILD_ROOT}/pireus-quotient-novelty-effect-parity"

SOUNIO_REL='stdlib/hardware/pireus/quotient_novelty_forge.sio'
FREEZE_REL='tools/pireus/quotient_novelty_forge.freeze.v5'
PARITY_OPEN_REL='tools/pireus/quotient_novelty_forge.parity-open.v5'
FORMAL_RECEIPT_REL='tools/pireus/quotient_novelty_forge.formal-parity.v5'
KOKA_REL='formal/koka/pireus_quotient_novelty_effect_parity.kk'
EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.koka.txt'
RECEIPT_REL='tools/pireus/quotient_novelty_forge.effect-parity.v5'
PARENT_GATE_REL='scripts/ci/pireus_quotient_novelty_formal_parity.sh'

SOUNIO_SHA256='791d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667'
SEMANTICS_SHA256='9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21'
FREEZE_SHA256='640a271bbe1966a3993e72be8fe019b1152530372cfb3ab91ede92011c0fc8c7'
PARITY_OPEN_SHA256='108ac3dd8df394e01a5a3293aab8d9fe312d522245ed2ee02e8bc5db37fa2943'
FORMAL_RECEIPT_SHA256='cff661497206523f273613e07fd8455ba7f036c62853b3b978c1bc29aa527593'
KOKA_SHA256='66e24766d52d7582cb398b315010fec21fa73be93cb9395cd7a042bf32e4ab46'
EVIDENCE_SHA256='097e5dad4ecbb3b7507c5b4c1ccdc54ac390c7a2028619c662dd0b892d106979'
RECEIPT_SHA256='09dce0400a1fdac8876bdd136c011bc06db6e2651b1da6d39b3b414ba7e7330e'
PARENT_GATE_SHA256='c47f1b93af89e6b6db2b13a199f3c878d7331e4e03f3e8e3a1a0a4dce5d3c487'
KOKA_BINARY_SHA256='5268748ed5082f3693ddf9fa40e560020aa16b6be6bd52b86c97ce5435b24cba'
TOOLCHAIN_SHA256='273f70c80ed71dcfbe1ee077607ec435d8791e59032cc13e30e479fd25995332'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='3d470bb38c06afe73bb36cf16827350fe8c6f271b2b564aaaf3487f66ea07fcf'
RESULT_SHA256='0c934db8759726e815e9c19fca213f5157219c9951734a2a77c952a3df1577ef'
PREEXEC_FRAME_SHA256='1e895e2ce4d5312a62aa8b940ddd62c8c7449a8780e4be0c32a6871d1b4ecfe2'
SEAL_FRAME_SHA256='10eb748db63d8400108236ae238a3bb8cff1a3bc42607c27ba71234ddbaf81fe'

fail() {
  printf 'pireus quotient novelty effect parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

sha_limbs() {
  local hex="$1" out='' i part
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
  grep -Fqx -- "${expected}" "${path}" || fail "missing exact line in ${path}: ${expected}"
}

parity_frame() {
  local stage="$1" action="$2" policy="$3" semantic_write="$4"
  local expected_write="$5" parity_valid="$6" review_promoted="$7"
  local parent_hash="$8" result_hash="$9"
  printf '9020 %s %s 3 3 %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${policy}" "${semantic_write}" \
    "${expected_write}" "${parity_valid}" "${review_promoted}" \
    "$(sha_limbs "${SOUNIO_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$([[ "${result_hash}" == zero ]] && printf '0 0 0 0 0 0 0 0' || sha_limbs "${result_hash}")" \
    '0 0 0 0 0 0 0 0'
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
}

deny_without_dispatch() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4"
  local decision rc
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "Guardian decision drift for ${label}: ${decision}"
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
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
[[ -x "${KOKA}" ]] || fail 'Koka 3.2.3 executable unavailable'
[[ "$(sha_file "${KOKA}")" == "${KOKA_BINARY_SHA256}" ]] || fail 'Koka executable hash drift'

require_line "${ROOT}/${FREEZE_REL}" "frozen_source_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'koka_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'effect_parity_complete=false'

for declaration in \
  'effect authority-request' \
  'type relative-decision' \
  'type material-decision' \
  'type scientific-decision' \
  'type historical-decision' \
  'fun frozen-authority' \
  'fun run-parity'; do
  grep -Fq -- "${declaration}" "${ROOT}/${KOKA_REL}" || fail "missing Koka declaration: ${declaration}"
done

koka_version="$("${KOKA}" --version --console=raw | sed -n '1p')"
gcc_version="$(gcc --version | sed -n '1p')"
toolchain_record="koka=${koka_version} koka_binary_sha256=${KOKA_BINARY_SHA256} cc=${gcc_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] || fail 'Koka toolchain drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] || fail 'hardware drift'
command_record='koka -O2 --builddir=/workspace/.home/openvscode-server/.cache/pireus/koka-v5/build -o /workspace/.home/openvscode-server/.cache/pireus/koka-v5/pireus-quotient-novelty-effect-parity formal/koka/pireus_quotient_novelty_effect_parity.kk && chmod 0755 /workspace/.home/openvscode-server/.cache/pireus/koka-v5/pireus-quotient-novelty-effect-parity && /workspace/.home/openvscode-server/.cache/pireus/koka-v5/pireus-quotient-novelty-effect-parity'
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] || fail 'command drift'

wrong_parent='0dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21'
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
  'pireus quotient novelty formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY obligations=4/4 admitted_actions=12 q0_classes=48 q1_classes=48 q2_classes=14 formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED selected_child=-1 claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'Lean formal parity gate terminal marker drift'

authorize PREEXEC \
  "$(parity_frame 3 4 1 0 0 0 0 "${SEMANTICS_SHA256}" zero)" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

mkdir -p "${BUILD_ROOT}/build"
PATH="$(dirname "${KOKA}"):${PATH}" koka -O2 \
  --builddir="${BUILD_ROOT}/build" \
  -o "${RESULT}" \
  "${ROOT}/${KOKA_REL}"
chmod 0755 "${RESULT}"
require_hash "${RESULT}" "${RESULT_SHA256}"

program_output="$("${RESULT}")"
expected_output="$(printf '%s\n' \
  'schema=pireus-quotient-novelty-effect-parity-evidence-v5' \
  'producing_language=Koka' \
  'producing_role=EFFECT_PARITY' \
  "sounio_source_sha256=${SOUNIO_SHA256}" \
  "sounio_semantics_sha256=${SEMANTICS_SHA256}" \
  'authority_stage_transition_effects=true' \
  'fail_closed_request_refusals=true' \
  'no_result_injection_or_review_promotion=true' \
  'typed_novelty_scope_separation_checked=true' \
  'checked_obligations=4' \
  'formal_proofs=0' \
  'checks=16' \
  'adversarial_checks=15' \
  'semantic_write=false' \
  'expected_result_write=false' \
  'selected_child=-1' \
  'claim_ready=false')"
[[ "${program_output}" == "${expected_output}" ]] || fail 'Koka effect parity output drift'

authorize SEAL \
  "$(parity_frame 4 8 1 0 0 1 0 "${SEMANTICS_SHA256}" "${RESULT_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

require_line "${ROOT}/${RECEIPT_REL}" 'status=EFFECT_PARITY_COMPLETE'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_language=Koka'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_role=EFFECT_PARITY'
require_line "${ROOT}/${RECEIPT_REL}" "sounio_source_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "result_sha256=${RESULT_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'effect_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'material_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'promotable_to_semantic_authority=false'
require_line "${ROOT}/${RECEIPT_REL}" 'semantic_write=false'
require_line "${ROOT}/${RECEIPT_REL}" 'expected_result_write=false'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'

for obligation in \
  authority_stage_transition_effects \
  fail_closed_request_refusals \
  no_result_injection_or_review_promotion \
  typed_novelty_scope_preservation; do
  grep -Eq "^obligation_[0-9][0-9]=${obligation} status=CHECKED$" \
    "${ROOT}/${EVIDENCE_REL}" || fail "unchecked Koka obligation: ${obligation}"
done

printf '%s\n' \
  'pireus quotient novelty effect parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Koka role=EFFECT_PARITY effect_checks=4/4 checks=16 adversarial=15 formal=COMPLETE effect=COMPLETE material=OPEN_NOT_EXECUTED selected_child=-1 claim_ready=false python_process_launched=false rust_process_launched=false'
