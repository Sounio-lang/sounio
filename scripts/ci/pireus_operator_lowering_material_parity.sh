#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/material-v6/xeon'
BINARY="${BUILD_ROOT}/operator-lowering-material-parity"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pireus-operator-lowering-material-v6.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

SOUNIO_REL='stdlib/hardware/pireus/operator_lowering_forge.sio'
FREEZE_REL='tools/pireus/operator_lowering_forge.freeze.v6'
PARITY_OPEN_REL='tools/pireus/operator_lowering_forge.parity-open.v6'
FORMAL_RECEIPT_REL='tools/pireus/operator_lowering_forge.formal-parity.v6'
EFFECT_RECEIPT_REL='tools/pireus/operator_lowering_forge.effect-parity.v6'
CPP_REL='tools/pireus/operator_lowering_material_parity.cpp'
EVIDENCE_REL='tools/pireus/evidence/operator_lowering_forge_v6.material.xeon.txt'
RECEIPT_REL='tools/pireus/operator_lowering_forge.material-parity.v6'
PARENT_EFFECT_GATE_REL='scripts/ci/pireus_operator_lowering_effect_parity.sh'
PARENT_MATERIAL_GATE_REL='scripts/ci/pireus_quotient_novelty_material_parity.sh'
PARENT_MATERIAL_RECEIPT_REL='tools/pireus/quotient_novelty_forge.material-parity.v5'
PARENT_XEON_REL='tools/pireus/evidence/quotient_novelty_forge_v5.material.xeon.txt'
PARENT_APPLE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.material.apple.txt'
PARENT_DGX_REL='tools/pireus/evidence/quotient_novelty_forge_v5.material.dgx.txt'
PARENT_U250_REL='tools/pireus/evidence/quotient_novelty_forge_v5.material.u250.txt'

PARENT_MATERIAL_COMMIT='ec4027845a48c4f2083d24889168bf4aaa4e2c24'
EFFECT_GATE_COMMIT='2bfc0a227fb4bb7d4fca31002668ba22631a5b5d'
SUPERSEDED_MATERIAL_COMMIT='18e7dbc38dcd610f56ffb27ff01580c4972af05f'
MATERIAL_PARITY_COMMIT='f394e6cd18008fe40dcebe98735a4d9e58fb297c'
SOUNIO_SHA256='178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0'
SEMANTICS_SHA256='bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
FREEZE_SHA256='973d620f30337378b760aa185ddbe9897bdd82ce18ee9e212756f519d1ed7181'
PARITY_OPEN_SHA256='4dbd89c5a18a2771bda46674b4ad93849e9f0ea160c7c9f42ce511307c7a6eba'
FORMAL_RECEIPT_SHA256='31f229664a627134898d476d3e5374cd7458401420f49316e129ea951386d169'
EFFECT_RECEIPT_SHA256='9deba7c7f66d9e75e82dbfce7b0ed65e94713f602d0cce6a8190218c5b32629f'
CPP_SHA256='5129010594908f260fab17adb9fa057eec4aca2827eddd898c526916b6d23607'
EVIDENCE_SHA256='3fd890e59a11d30944d1b3613c98d97af49cef8803f311d421243883edad13c2'
RECEIPT_SHA256='867a1cb7098371494a6aa80df4d0a42dbf62b3cc7aa6dffc38fe87c7fcb9e1be'
SUPERSEDED_RECEIPT_SHA256='c1320a03aa0cec0cf495376bf470f351bee264cbbf48391cac80bb9458550b24'
PARENT_EFFECT_GATE_SHA256='8d3bfc9e333ba53459805c3fd48703b6f22b05cb86696535f233848e12323ae9'
PARENT_MATERIAL_GATE_SHA256='52cfcf4956b9fbdd3c8fbf40575cc4710942aa3d44a8a36d7456b79f9d0be804'
PARENT_MATERIAL_RECEIPT_SHA256='c9a09126ff8f0de58d4054a201f5bcfcf39d998d4087a02ea949ee578b4623b5'
PARENT_XEON_SHA256='35207450acd83578c2584a316def2b5db4090c620b009529a78526df2721af90'
PARENT_APPLE_SHA256='6f832c8bdac679bf010b3e1dc133222d27a49b8173535170ae0652abba0f17ab'
PARENT_DGX_SHA256='c431deff96ca855062c61ba0b2922b368e3bf5da30c6709f838e2a4848ec0bf9'
PARENT_U250_SHA256='4e7f26d70e65ec7a449e48be7e3a7dbfb4886ea575cfa995e787dd5abfba5b3f'
TOOLCHAIN_SHA256='f72fdfe92dea5e229c3ca852f031171ce2007e15f71fb2632edb24b2850db450'
HARDWARE_SHA256='6c0cad13fd376aea694c4a7a73e603194713a938d6198c8ebddf16f3a1a75689'
COMMAND_SHA256='b24ee433eec3b8c8a024d8ce552074061b413f25525dcbbe719f76d46c014a01'
BINARY_SHA256='4fe7c3050f97c8f9e410df425adff1a7379cf9a3aa4bff55b4e00f142c7797f4'
PREEXEC_FRAME_SHA256='67a3e202451864bb4be534adc2626b4b2230c8f48d32f942a7239666ada4d9ec'
SEAL_FRAME_SHA256='4c1cee0266b7a24a9b5f1918e28dca6f7204aa0bec4d3858092f7c623aa70325'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='1aec3312ab003a9f9c250a9f0cb701fe8da392a4710c9998a100d04a68a7678d'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='cff252b7b71a6ab0b0e5518a7589040ad223ef8f607e5eae6f25e1bf474cf2cf'
ZERO='0 0 0 0 0 0 0 0'

STAGE_SOUNIO_EXECUTABLE=2
STAGE_SEMANTICS_FROZEN=3
STAGE_PARITY_OPEN=4
ACTION_PARITY_EXEC=4
ACTION_CLAIM_PROMOTION=7
ACTION_SEAL=8
LANG_CPP=4
LANG_PYTHON=7
LANG_RUST=8
ROLE_MATERIAL_PARITY=4
ROLE_REVIEW_ONLY=7
POLICY_MISSING=0
POLICY_READY=1
POLICY_TIMEOUT=2
POLICY_ERROR=3
GUARDIAN_FRAME_SCHEMA=9020
GUARDIAN_FRAME_WORDS=82

fail() {
  printf 'pireus operator lowering material parity: FAIL: %s\n' "$*" >&2
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

require_committed_hash() {
  local commit="$1" path="$2" expected="$3"
  [[ "$(git -C "${ROOT}" show "${commit}:${path}" | sha256sum | cut -d' ' -f1)" == \
    "${expected}" ]] || fail "committed hash drift: ${commit}:${path}"
}

material_identity_receipt_admissible() {
  local path="$1"
  grep -Fqx 'schema=pireus-operator-lowering-forge.material-identity.v6.1' "${path}" &&
    grep -Fqx 'status=MATERIAL_IDENTITY_ACCOUNTING_RECORDED_WITH_OPEN_EXECUTION_DEBT' "${path}" &&
    grep -Fqx "supersedes_receipt_sha256=${SUPERSEDED_RECEIPT_SHA256}" "${path}" &&
    grep -Fqx 'material_parity_complete=false' "${path}" &&
    grep -Fqx 'material_parity_obligations_discharged=0' "${path}" &&
    grep -Fqx 'material_parity_obligations_unresolved=1120' "${path}" &&
    ! grep -Fqx 'material_parity_complete=true' "${path}"
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8"
  local review_promoted="$9" parent_hash="${10}" toolchain_hash="${11}"
  local hardware_hash="${12}" command_hash="${13}" result_hash="${14}"
  local result_limbs="${ZERO}"
  if [[ "${result_hash}" != zero ]]; then
    result_limbs="$(sha_limbs "${result_hash}")"
  fi
  printf '%s %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${GUARDIAN_FRAME_SCHEMA}" \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" \
    "${review_promoted}" "$(sha_limbs "${CPP_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${toolchain_hash}")" "$(sha_limbs "${hardware_hash}")" \
    "$(sha_limbs "${command_hash}")" "${result_limbs}" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4" mode="$5"
  local decision rc
  [[ "${frame%% *}" == "${GUARDIAN_FRAME_SCHEMA}" ]] ||
    fail "Guardian frame schema drift for ${label}"
  [[ "$(wc -w <<< "${frame}")" -eq "${GUARDIAN_FRAME_WORDS}" ]] ||
    fail "Guardian frame field-count drift for ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s mode=%s frame_sha256=%s decision=%s\n' \
    "${label}" "${mode}" "$(sha_text "${frame}")" "${decision}"
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "Guardian frame drift: ${label}"
  check_guardian "${label}" "${frame}" "${expected_rc}" "${expected}" AUTHORITY
}

deny_without_dispatch() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4"
  check_guardian "${label}" "${frame}" "${expected_rc}" "${expected}" DENY_TEST
  printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
}

require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${FORMAL_RECEIPT_REL}" "${FORMAL_RECEIPT_SHA256}"
require_hash "${ROOT}/${EFFECT_RECEIPT_REL}" "${EFFECT_RECEIPT_SHA256}"
require_hash "${ROOT}/${CPP_REL}" "${CPP_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_EFFECT_GATE_REL}" "${PARENT_EFFECT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_MATERIAL_GATE_REL}" "${PARENT_MATERIAL_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_MATERIAL_RECEIPT_REL}" "${PARENT_MATERIAL_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_XEON_REL}" "${PARENT_XEON_SHA256}"
require_hash "${ROOT}/${PARENT_APPLE_REL}" "${PARENT_APPLE_SHA256}"
require_hash "${ROOT}/${PARENT_DGX_REL}" "${PARENT_DGX_SHA256}"
require_hash "${ROOT}/${PARENT_U250_REL}" "${PARENT_U250_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
command -v g++ >/dev/null 2>&1 || fail 'g++ unavailable for Xeon replay'

git -C "${ROOT}" merge-base --is-ancestor \
  "${EFFECT_GATE_COMMIT}" "${MATERIAL_PARITY_COMMIT}" ||
  fail 'material parity predates the v6 effect gate'
git -C "${ROOT}" merge-base --is-ancestor \
  "${PARENT_MATERIAL_COMMIT}" "${MATERIAL_PARITY_COMMIT}" ||
  fail 'material parity predates the parent target receipts'
git -C "${ROOT}" merge-base --is-ancestor "${MATERIAL_PARITY_COMMIT}" HEAD ||
  fail 'material parity commit missing from current history'
[[ "$(git -C "${ROOT}" show "${MATERIAL_PARITY_COMMIT}:${CPP_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${CPP_SHA256}" ]] || fail 'committed C++ source drift'
[[ "$(git -C "${ROOT}" show "${MATERIAL_PARITY_COMMIT}:${EVIDENCE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${EVIDENCE_SHA256}" ]] || fail 'committed material evidence drift'
[[ "$(git -C "${ROOT}" show "${MATERIAL_PARITY_COMMIT}:${RECEIPT_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${RECEIPT_SHA256}" ]] || fail 'committed material receipt drift'
[[ "$(git -C "${ROOT}" show "${EFFECT_GATE_COMMIT}:${PARENT_EFFECT_GATE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${PARENT_EFFECT_GATE_SHA256}" ]] || fail 'committed effect gate drift'
[[ "$(git -C "${ROOT}" show "${PARENT_MATERIAL_COMMIT}:${PARENT_MATERIAL_GATE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${PARENT_MATERIAL_GATE_SHA256}" ]] || fail 'committed parent material gate drift'
require_committed_hash "${MATERIAL_PARITY_COMMIT}" "${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_committed_hash "${MATERIAL_PARITY_COMMIT}" "${FREEZE_REL}" "${FREEZE_SHA256}"
require_committed_hash "${MATERIAL_PARITY_COMMIT}" "${PARITY_OPEN_REL}" \
  "${PARITY_OPEN_SHA256}"
require_committed_hash "${MATERIAL_PARITY_COMMIT}" "${FORMAL_RECEIPT_REL}" \
  "${FORMAL_RECEIPT_SHA256}"
require_committed_hash "${MATERIAL_PARITY_COMMIT}" "${EFFECT_RECEIPT_REL}" \
  "${EFFECT_RECEIPT_SHA256}"

material_identity_receipt_admissible "${ROOT}/${RECEIPT_REL}" ||
  fail 'current material identity receipt is inadmissible'
git -C "${ROOT}" show \
  "${SUPERSEDED_MATERIAL_COMMIT}:${RECEIPT_REL}" > "${TMP_ROOT}/superseded-receipt.v6"
require_hash "${TMP_ROOT}/superseded-receipt.v6" "${SUPERSEDED_RECEIPT_SHA256}"
if material_identity_receipt_admissible "${TMP_ROOT}/superseded-receipt.v6"; then
  fail 'superseded false-completion receipt was admitted'
fi
printf '%s\n' \
  'SUPERSESSION_CHECK old_material_parity_complete=true current_material_parity_complete=false old_receipt_admitted=false'

require_line "${ROOT}/${FREEZE_REL}" "module_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'cpp_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${EFFECT_RECEIPT_REL}" 'effect_parity_complete=true'
require_line "${ROOT}/${EFFECT_RECEIPT_REL}" 'material_parity_complete=false'
require_line "${ROOT}/${PARENT_MATERIAL_RECEIPT_REL}" \
  'status=MATERIAL_PARITY_COMPLETE_WITH_UNRESOLVED_TARGETS'
require_line "${ROOT}/${PARENT_MATERIAL_RECEIPT_REL}" 'observed_physical_endpoints=4'
require_line "${ROOT}/${PARENT_MATERIAL_RECEIPT_REL}" 'unresolved_physical_endpoints=2'

toolchain_record="compiler=$(g++ --version | sed -n '1p') source_sha256=${CPP_SHA256} standard=c++17 optimization=-O2"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'live C++ toolchain drift'
hardware_record="hostname=$(hostname) kernel=$(uname -s) release=$(uname -r) architecture=$(uname -m) cpu_model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1) online_cpus=$(getconf _NPROCESSORS_ONLN)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'live Xeon hardware drift'
command_record='g++ -std=c++17 -O2 tools/pireus/operator_lowering_material_parity.cpp -o /workspace/.home/openvscode-server/.cache/pireus/material-v6/xeon/operator-lowering-material-parity && /workspace/.home/openvscode-server/.cache/pireus/material-v6/xeon/operator-lowering-material-parity --target=xeon'
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'live material command drift'

set +e
invalid_hash_output="$(sha_limbs 'not-a-sha256' 2>&1)"
invalid_hash_rc=$?
set -e
[[ "${invalid_hash_rc}" -eq 1 ]] ||
  fail 'malformed SHA-256 text did not fail closed'
[[ "${invalid_hash_output}" == 'pireus operator lowering material parity: FAIL: invalid sha256 digest: not-a-sha256' ]] ||
  fail 'malformed SHA-256 refusal drift'
printf 'GUARDIAN_DISPATCH label=MALFORMED_SHA256 process_launched=false\n'

wrong_parent='0d69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
deny_without_dispatch PARENT_LAUNDERING \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" 0 0 0 0 "${wrong_parent}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN'
deny_without_dispatch CPP_SEMANTIC_WRITE \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" 1 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch CPP_EXPECTED_RESULT_WRITE \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" 0 1 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch REVIEW_PROMOTION \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" 0 0 0 1 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_MISSING \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_MISSING}" 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_TIMEOUT \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_TIMEOUT}" 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_ERROR \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_ERROR}" 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
deny_without_dispatch PREFREEZE \
  "$(authority_frame "${STAGE_SOUNIO_EXECUTABLE}" "${ACTION_PARITY_EXEC}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
deny_without_dispatch CLAIM_PROMOTION \
  "$(authority_frame "${STAGE_PARITY_OPEN}" "${ACTION_CLAIM_PROMOTION}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 123 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=123 reason=action-forbidden-for-role next_stage=PARITY_OPEN'
deny_without_dispatch PYTHON_ORACLE \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_PYTHON}" "${ROLE_REVIEW_ONLY}" "${POLICY_READY}" 0 0 0 0 "${SEMANTICS_SHA256}" "${PYTHON_TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${PYTHON_COMMAND_SHA256}" zero)" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
deny_without_dispatch RUST_ORACLE \
  "$(authority_frame "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" "${LANG_RUST}" "${ROLE_REVIEW_ONLY}" "${POLICY_READY}" 0 0 0 0 "${SEMANTICS_SHA256}" "${RUST_TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${RUST_COMMAND_SHA256}" zero)" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
deny_without_dispatch SEAL_WITHOUT_RESULT \
  "$(authority_frame "${STAGE_PARITY_OPEN}" "${ACTION_SEAL}" "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)" 118 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=118 reason=receipt-incomplete next_stage=PARITY_OPEN'

effect_gate_output="$("${ROOT}/${PARENT_EFFECT_GATE_REL}")"
printf '%s\n' "${effect_gate_output}" | grep -Fqx -- \
  'pireus operator lowering effect parity: STAGE_REACHED_NOT_A_CLAIM gate_mode=CONTENT_ADDRESSED_REPLAY stage=PARITY_OPEN language=Koka role=EFFECT_PARITY effect_scope=AUTHORITY_AND_EFFECT_TOPOLOGY_ONLY effect_checks=7/7 checks=38 adversarial=30 candidates=1120 effect_memory_unresolved=1120 effect_memory_discharged_by_koka=0 admitted_lowerings=0 formal=COMPLETE effect=COMPLETE material=OPEN_NOT_EXECUTED selected_candidate=-1 claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'v6 Koka effect gate terminal marker drift'
require_hash "${ROOT}/${EFFECT_RECEIPT_REL}" "${EFFECT_RECEIPT_SHA256}"

printf '%s\n' \
  'PARENT_MATERIAL_REFERENCE mode=SEALED_IDENTITY_ARTIFACTS gate_process_launched=false historical_dependencies_reinterpreted=false v6_obligations_discharged=0'

preexec_frame="$(authority_frame \
  "${STAGE_SEMANTICS_FROZEN}" "${ACTION_PARITY_EXEC}" \
  "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" \
  0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" \
  "${HARDWARE_SHA256}" "${COMMAND_SHA256}" zero)"
authorize PREEXEC \
  "${preexec_frame}" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

mkdir -p "${BUILD_ROOT}"
(
  cd "${ROOT}"
  g++ -std=c++17 -O2 "${CPP_REL}" -o "${BINARY}"
)
require_hash "${BINARY}" "${BINARY_SHA256}"
"${BINARY}" --target=xeon > "${TMP_ROOT}/xeon.txt"
require_hash "${BINARY}" "${BINARY_SHA256}"
require_hash "${TMP_ROOT}/xeon.txt" "${EVIDENCE_SHA256}"
cmp "${TMP_ROOT}/xeon.txt" "${ROOT}/${EVIDENCE_REL}" >/dev/null ||
  fail 'Xeon material replay evidence drift'

seal_frame="$(authority_frame \
  "${STAGE_PARITY_OPEN}" "${ACTION_SEAL}" \
  "${LANG_CPP}" "${ROLE_MATERIAL_PARITY}" "${POLICY_READY}" \
  0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" \
  "${HARDWARE_SHA256}" "${COMMAND_SHA256}" "${EVIDENCE_SHA256}")"
authorize SEAL \
  "${seal_frame}" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

require_line "${ROOT}/${RECEIPT_REL}" \
  'status=MATERIAL_IDENTITY_ACCOUNTING_RECORDED_WITH_OPEN_EXECUTION_DEBT'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_language=C++'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_role=MATERIAL_PARITY'
require_line "${ROOT}/${RECEIPT_REL}" \
  'material_identity_accounting_scope=TARGET_IDENTITY_AND_ENDPOINT_ACCOUNTING_ONLY'
require_line "${ROOT}/${RECEIPT_REL}" \
  'material_parity_scope=NO_GENERATED_LOWERING_EXECUTED'
require_line "${ROOT}/${RECEIPT_REL}" \
  "supersedes_receipt_sha256=${SUPERSEDED_RECEIPT_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "cpp_source_sha256=${CPP_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "xeon_evidence_sha256=${EVIDENCE_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "toolchain_sha256=${TOOLCHAIN_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "hardware_sha256=${HARDWARE_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "command_sha256=${COMMAND_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "binary_sha256=${BINARY_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "preexec_frame=${preexec_frame}"
require_line "${ROOT}/${RECEIPT_REL}" "preexec_frame_sha256=${PREEXEC_FRAME_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "seal_frame=${seal_frame}"
require_line "${ROOT}/${RECEIPT_REL}" "seal_frame_sha256=${SEAL_FRAME_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" 'checked_material_accounting_guardrails=6'
require_line "${ROOT}/${RECEIPT_REL}" 'checked_observer_predicates=4'
require_line "${ROOT}/${RECEIPT_REL}" 'canonical_target_classes_declared=4'
require_line "${ROOT}/${RECEIPT_REL}" 'parent_target_identity_classes_bound=4'
require_line "${ROOT}/${RECEIPT_REL}" 'declared_physical_endpoints=6'
require_line "${ROOT}/${RECEIPT_REL}" 'parent_observed_physical_endpoints=4'
require_line "${ROOT}/${RECEIPT_REL}" 'unresolved_physical_endpoints=2'
require_line "${ROOT}/${RECEIPT_REL}" 'material_execution_unresolved=1120'
require_line "${ROOT}/${RECEIPT_REL}" \
  'v6_lowering_obligations_discharged_by_parent_receipts=0'
require_line "${ROOT}/${RECEIPT_REL}" 'admitted_lowerings=0'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_receipt_bound=true'
require_line "${ROOT}/${RECEIPT_REL}" 'effect_parity_receipt_bound=true'
require_line "${ROOT}/${RECEIPT_REL}" 'identity_accounting_coverage=1/1'
require_line "${ROOT}/${RECEIPT_REL}" 'material_parity_obligations_discharged=0'
require_line "${ROOT}/${RECEIPT_REL}" 'material_parity_obligations_unresolved=1120'
require_line "${ROOT}/${RECEIPT_REL}" 'material_identity_accounting_recorded=true'
require_line "${ROOT}/${RECEIPT_REL}" 'material_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" \
  'material_parity_incomplete_reason=NO_GENERATED_LOWERING_EXECUTED'
require_line "${ROOT}/${RECEIPT_REL}" 'material_target_coverage_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'material_lowering_coverage_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'generated_lowering_processes_launched=0'
require_line "${ROOT}/${RECEIPT_REL}" 'remote_target_processes_launched=0'
require_line "${ROOT}/${RECEIPT_REL}" 'target_lowerings=0'
require_line "${ROOT}/${RECEIPT_REL}" 'target_cost_records=0'
require_line "${ROOT}/${RECEIPT_REL}" 'target_performance_records=0'
require_line "${ROOT}/${RECEIPT_REL}" 'promotable_to_semantic_authority=false'
require_line "${ROOT}/${RECEIPT_REL}" 'selected_candidate=-1'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'

for obligation in \
  current_xeon_identity_observed \
  parent_target_receipt_lineage_bound \
  target_identity_not_lowering_evidence \
  canonical_endpoint_holes_retained \
  no_material_receipt_semantic_promotion \
  no_generated_lowering_cost_or_performance_execution; do
  grep -Eq "^obligation_[0-9][0-9]=${obligation} status=CHECKED$" \
    "${ROOT}/${RECEIPT_REL}" || fail "unchecked material obligation: ${obligation}"
done

for observer_check in \
  receipt_hash_shapes \
  candidate_program_quotient \
  target_population_partition \
  xeon_identity; do
  grep -Eq "^observer_check_[0-9][0-9]=${observer_check} status=CHECKED$" \
    "${ROOT}/${EVIDENCE_REL}" || fail "unchecked observer predicate: ${observer_check}"
done
require_line "${ROOT}/${EVIDENCE_REL}" 'checked_observer_predicates=4'
require_line "${ROOT}/${EVIDENCE_REL}" 'material_parity_complete=false'

printf '%s\n' \
  'pireus operator lowering material parity: STAGE_REACHED_NOT_A_CLAIM gate_mode=CONTENT_ADDRESSED_REPLAY stage=PARITY_OPEN language=C++ role=MATERIAL_PARITY identity_accounting=RECORDED accounting_guardrails=6/6 observer_checks=4/4 target_classes=4/4 endpoints=4/6 unresolved_endpoints=2 candidates=1120 material_execution_unresolved=1120 material_parity_obligations_discharged=0 material_parity_obligations_unresolved=1120 admitted_lowerings=0 formal_receipt=BOUND effect_receipt=BOUND material=INCOMPLETE target_coverage=PARTIAL lowering_coverage=NONE selected_candidate=-1 claim_ready=false python_process_launched=false rust_process_launched=false'
