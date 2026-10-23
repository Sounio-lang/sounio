#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${ROOT}"

GIT_COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/material-v5/xeon'
BINARY="${BUILD_ROOT}/quotient-novelty-material-parity"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pireus-material-parity-v5.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

SOUNIO_REL='stdlib/hardware/pireus/quotient_novelty_forge.sio'
FREEZE_REL='tools/pireus/quotient_novelty_forge.freeze.v5'
PARITY_OPEN_REL='tools/pireus/quotient_novelty_forge.parity-open.v5'
FORMAL_RECEIPT_REL='tools/pireus/quotient_novelty_forge.formal-parity.v5'
EFFECT_RECEIPT_REL='tools/pireus/quotient_novelty_forge.effect-parity.v5'
CPP_REL='tools/pireus/quotient_novelty_material_parity.cpp'
XEON_EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.material.xeon.txt'
APPLE_EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.material.apple.txt'
DGX_EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.material.dgx.txt'
U250_EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.material.u250.txt'
RECEIPT_REL='tools/pireus/quotient_novelty_forge.material-parity.v5'
PARENT_GATE_REL='scripts/ci/pireus_quotient_novelty_effect_parity.sh'

SOUNIO_SHA256='791d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667'
SEMANTICS_SHA256='9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21'
FREEZE_SHA256='640a271bbe1966a3993e72be8fe019b1152530372cfb3ab91ede92011c0fc8c7'
PARITY_OPEN_SHA256='108ac3dd8df394e01a5a3293aab8d9fe312d522245ed2ee02e8bc5db37fa2943'
FORMAL_RECEIPT_SHA256='cff661497206523f273613e07fd8455ba7f036c62853b3b978c1bc29aa527593'
EFFECT_RECEIPT_SHA256='09dce0400a1fdac8876bdd136c011bc06db6e2651b1da6d39b3b414ba7e7330e'
CPP_SHA256='05b7772ad83d8607f5354dff738ceba35e0429d386d21406734f685c8229b413'
XEON_EVIDENCE_SHA256='35207450acd83578c2584a316def2b5db4090c620b009529a78526df2721af90'
APPLE_EVIDENCE_SHA256='6f832c8bdac679bf010b3e1dc133222d27a49b8173535170ae0652abba0f17ab'
DGX_EVIDENCE_SHA256='c431deff96ca855062c61ba0b2922b368e3bf5da30c6709f838e2a4848ec0bf9'
U250_EVIDENCE_SHA256='4e7f26d70e65ec7a449e48be7e3a7dbfb4886ea575cfa995e787dd5abfba5b3f'
RECEIPT_SHA256='c9a09126ff8f0de58d4054a201f5bcfcf39d998d4087a02ea949ee578b4623b5'
PARENT_GATE_SHA256='a44a39049ca00bb60d3d95fcfd1ba3955e0c9054cb050a340d96c48a1b88250b'

XEON_TOOLCHAIN_SHA256='73e8bd5e9a37d1c6ff1ac2bab00d7469ea4baa6825a39a7205676de24254f5c4'
XEON_HARDWARE_SHA256='6c0cad13fd376aea694c4a7a73e603194713a938d6198c8ebddf16f3a1a75689'
XEON_COMMAND_SHA256='98248cfa272fcdec37025820525f527d8d58a0de0074781a4d53bd9035ff1b80'
XEON_BINARY_SHA256='2f7dafa3be1df927065cd0b7b38cfa48802439f671503554e732e87cf2634d0c'
XEON_PREEXEC_FRAME_SHA256='8c7875a751c94077dcbaa65e94c098c701664e2278dc2c232a60a53c0a84d5cd'
XEON_SEAL_FRAME_SHA256='54ab8b174a03d63c097c67316707bf2b0dc1ecb7084a7909ebf15b4a45ee8bbc'

APPLE_TOOLCHAIN_SHA256='ed553b2ba93385df7f58f25f1ba8df8b3515e0be54e161cf98091e8210dcc681'
APPLE_HARDWARE_SHA256='ee7d9dfd166eac0ec2df224c2715a1fe75f89e47b0eea7e6707f25714e014f7e'
APPLE_COMMAND_SHA256='a664bcdb5e7fb485c8a7cf4cd7b68d40025c73111ddd34447a785fbc92b7573a'
APPLE_PREEXEC_FRAME_SHA256='59301f37391cb5feb0814681b39164456ce3c40b12428e4782a0ac7a0cb5cb6d'
APPLE_SEAL_FRAME_SHA256='d0c6ce01f3c4240979409bf243a493bc6ad8fe186e30cf328891d1f4e017fd72'

DGX_TOOLCHAIN_SHA256='73e8bd5e9a37d1c6ff1ac2bab00d7469ea4baa6825a39a7205676de24254f5c4'
DGX_HARDWARE_SHA256='fbee99f0a5413deba59115de3e68110df918282a01676de6805ba5926150f687'
DGX_COMMAND_SHA256='0d925c109d74d3adb88c1533dd41a049d59e262b8b69bdfa95832d91e9b6c39a'
DGX_PREEXEC_FRAME_SHA256='15a9f4b3a94fa3e6092c93a078c65f442a90097f331ec66589dfa167ff18893e'
DGX_SEAL_FRAME_SHA256='b1445d646857073f7ed29ae67f86a677ee7c40245fc07ea7bfae506be3bb73bf'

U250_TOOLCHAIN_SHA256='75b85b926fc0f62ceab909eb53ca02da9a81bf26f11dd532cd74ef2040b9131a'
U250_HARDWARE_SHA256='ce1287dd5cc698636454ae790c5d6af86e7cbacaeb26cfd13297976c73d5c6a4'
U250_COMMAND_SHA256='6d9dcd353dc350d9975816ab1a2427a77e007b4440d4c355bc0a7fe8640b2159'
U250_PREEXEC_FRAME_SHA256='0ba6fc38a203ff9435cc0e0017c9302c2630b501790e5aee17bc019e4a7a1c72'
U250_SEAL_FRAME_SHA256='c59962aa31808c10c76cdfb75de7c249ef3849805ed453c41dd277013d305b31'

PARENT_LAUNDERING_FRAME_SHA256='68f886f897bfdc81164de2b90400ab30bad216e2eb88aec7a3bd5c0a2f2a77b4'
SEMANTIC_WRITE_FRAME_SHA256='6f23c4273a905557bb13dfa2e7f945d12a259177bc2fb77be41038c8e290a0ef'
EXPECTED_WRITE_FRAME_SHA256='fcbd406ebff6fd651f711335da9da795e86305c1c4312c13229a3bdbc139a5e6'
REVIEW_PROMOTION_FRAME_SHA256='a2594511b09307b9ba28414c448b7c4eabdf06158a189177fbbfe1b214c7521b'
POLICY_MISSING_FRAME_SHA256='05c25505b80a634cac1a9a6636fc2766242f0ab0fcd45ab788971fd4dc6f765a'
POLICY_TIMEOUT_FRAME_SHA256='62a875454719e116b75507020ba1c9613e14846be5502d87922042b4993529ef'
POLICY_ERROR_FRAME_SHA256='5b75852fd0c368bf24707435c4c7954f4305bbeb54092c2d37a222b3b56124a5'
PREFREEZE_FRAME_SHA256='d1eaa3fd61e7178b5143b884c2e6cce10f0a69db3e6a36acdf7882dedf2d2ef1'
CLAIM_PROMOTION_FRAME_SHA256='a064822d2e454d425aeb3d6bdb2d16722ee5d3e9b91572a5e85bc622615d3a38'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='1aec3312ab003a9f9c250a9f0cb701fe8da392a4710c9998a100d04a68a7678d'
PYTHON_FRAME_SHA256='27328a2bc30d2cb78376706391b2eb866b0d5415d2ebb450fdfec8980476c911'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='cff252b7b71a6ab0b0e5518a7589040ad223ef8f607e5eae6f25e1bf474cf2cf'
RUST_FRAME_SHA256='2686c96caa3374ae75f752af3c573441976e2314839d84a7ce8b551905f8f5c4'

fail() {
  printf 'pireus quotient novelty material parity: FAIL: %s\n' "$*" >&2
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

receipt_value() {
  local key="$1" count value
  count="$(grep -c "^${key}=" "${ROOT}/${RECEIPT_REL}" || true)"
  [[ "${count}" -eq 1 ]] || fail "receipt key ${key} has count ${count}"
  value="$(sed -n "s/^${key}=//p" "${ROOT}/${RECEIPT_REL}")"
  printf '%s' "${value}"
}

require_record_hash() {
  local key="$1" expected="$2" value
  value="$(receipt_value "${key}")"
  [[ "$(sha_text "${value}")" == "${expected}" ]] || fail "record hash drift: ${key}"
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8" review_promoted="$9"
  local parent_hash="${10}" toolchain_hash="${11}" hardware_hash="${12}"
  local command_hash="${13}" result_hash="${14}"
  local zero='0 0 0 0 0 0 0 0'
  printf '9020 %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" "${review_promoted}" \
    "$(sha_limbs "${SOUNIO_SHA256}")" "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${parent_hash}")" "$(sha_limbs "${toolchain_hash}")" \
    "$(sha_limbs "${hardware_hash}")" "$(sha_limbs "${command_hash}")" \
    "$([[ "${result_hash}" == zero ]] && printf '%s' "${zero}" || sha_limbs "${result_hash}")" \
    "${zero}"
}

check_guardian() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5" mode="$6"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s mode=%s decision=%s\n' "${label}" "${mode}" "${decision}"
}

deny_without_dispatch() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  check_guardian "${label}" "${frame}" "${expected_sha}" "${expected_rc}" "${expected}" DENY_TEST
  printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
}

common_evidence() {
  local path="$1"
  require_line "${path}" 'schema=pireus-quotient-novelty-material-parity-v5'
  require_line "${path}" 'producing_language=C++'
  require_line "${path}" 'producing_role=MATERIAL_PARITY'
  require_line "${path}" 'authority_language=Sounio'
  require_line "${path}" "sounio_source_sha256=${SOUNIO_SHA256}"
  require_line "${path}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
  require_line "${path}" "formal_parity_receipt_sha256=${FORMAL_RECEIPT_SHA256}"
  require_line "${path}" "effect_parity_receipt_sha256=${EFFECT_RECEIPT_SHA256}"
  require_line "${path}" 'target_identity_observed=true'
  require_line "${path}" 'hash_bound_replay_only_after_sounio_freeze=true'
  require_line "${path}" 'canonical_target_receipt_semantics_fixed=true'
  require_line "${path}" 'lowering_cost_and_performance_remain_separate=true'
  require_line "${path}" 'lowering_cost_present=false'
  require_line "${path}" 'performance_present=false'
  require_line "${path}" 'cross_target_ranking_present=false'
  require_line "${path}" 'semantic_write=false'
  require_line "${path}" 'expected_result_write=false'
  require_line "${path}" 'no_material_receipt_promoted_to_semantic_authority=true'
  require_line "${path}" 'material_receipt_promotable_to_semantic_authority=false'
  require_line "${path}" 'selected_child=-1'
  require_line "${path}" 'claim_ready=false'
}

require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${FORMAL_RECEIPT_REL}" "${FORMAL_RECEIPT_SHA256}"
require_hash "${ROOT}/${EFFECT_RECEIPT_REL}" "${EFFECT_RECEIPT_SHA256}"
require_hash "${ROOT}/${CPP_REL}" "${CPP_SHA256}"
require_hash "${ROOT}/${XEON_EVIDENCE_REL}" "${XEON_EVIDENCE_SHA256}"
require_hash "${ROOT}/${APPLE_EVIDENCE_REL}" "${APPLE_EVIDENCE_SHA256}"
require_hash "${ROOT}/${DGX_EVIDENCE_REL}" "${DGX_EVIDENCE_SHA256}"
require_hash "${ROOT}/${U250_EVIDENCE_REL}" "${U250_EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
command -v g++ >/dev/null 2>&1 || fail 'g++ unavailable for Xeon replay'

require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" "frozen_source_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" 'expected_unresolved_target_obligations=1920'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'cpp_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${EFFECT_RECEIPT_REL}" 'effect_parity_complete=true'

require_record_hash xeon_toolchain_record "${XEON_TOOLCHAIN_SHA256}"
require_record_hash xeon_hardware_record "${XEON_HARDWARE_SHA256}"
require_record_hash xeon_command "${XEON_COMMAND_SHA256}"
require_record_hash apple_toolchain_record "${APPLE_TOOLCHAIN_SHA256}"
require_record_hash apple_hardware_record "${APPLE_HARDWARE_SHA256}"
require_record_hash apple_command "${APPLE_COMMAND_SHA256}"
require_record_hash dgx_24_toolchain_record "${DGX_TOOLCHAIN_SHA256}"
require_record_hash dgx_24_hardware_record "${DGX_HARDWARE_SHA256}"
require_record_hash dgx_24_command "${DGX_COMMAND_SHA256}"
require_record_hash u250_toolchain_record "${U250_TOOLCHAIN_SHA256}"
require_record_hash u250_hardware_record "${U250_HARDWARE_SHA256}"
require_record_hash u250_command "${U250_COMMAND_SHA256}"

live_toolchain_record="compiler=$(g++ --version | sed -n '1p') source_sha256=${CPP_SHA256} standard=c++17 optimization=-O2 dynamic_loading=-ldl"
[[ "$(sha_text "${live_toolchain_record}")" == "${XEON_TOOLCHAIN_SHA256}" ]] || fail 'live Xeon toolchain drift'
live_hardware_record="hostname=$(hostname) kernel=$(uname -s) release=$(uname -r) architecture=$(uname -m) cpu_model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1) online_cpus=$(getconf _NPROCESSORS_ONLN)"
[[ "$(sha_text "${live_hardware_record}")" == "${XEON_HARDWARE_SHA256}" ]] || fail 'live Xeon hardware drift'
live_command_record='g++ -std=c++17 -O2 tools/pireus/quotient_novelty_material_parity.cpp -ldl -o /workspace/.home/openvscode-server/.cache/pireus/material-v5/xeon/quotient-novelty-material-parity && /workspace/.home/openvscode-server/.cache/pireus/material-v5/xeon/quotient-novelty-material-parity --target=xeon'
[[ "$(sha_text "${live_command_record}")" == "${XEON_COMMAND_SHA256}" ]] || fail 'live Xeon command drift'

for evidence in \
  "${ROOT}/${XEON_EVIDENCE_REL}" \
  "${ROOT}/${APPLE_EVIDENCE_REL}" \
  "${ROOT}/${DGX_EVIDENCE_REL}" \
  "${ROOT}/${U250_EVIDENCE_REL}"; do
  common_evidence "${evidence}"
done

require_line "${ROOT}/${XEON_EVIDENCE_REL}" 'target_name=DARWIN_XEON'
require_line "${ROOT}/${XEON_EVIDENCE_REL}" 'architecture=x86_64'
require_line "${ROOT}/${APPLE_EVIDENCE_REL}" 'target_name=APPLE_SILICON'
require_line "${ROOT}/${APPLE_EVIDENCE_REL}" 'cpu_model=Apple M5 Max'
require_line "${ROOT}/${APPLE_EVIDENCE_REL}" 'machine_model=Mac17,7'
require_line "${ROOT}/${DGX_EVIDENCE_REL}" 'target_name=DGX_SPARK'
require_line "${ROOT}/${DGX_EVIDENCE_REL}" 'cuda_device_name=NVIDIA GB10'
require_line "${ROOT}/${DGX_EVIDENCE_REL}" 'cuda_compute_capability=12.1'
require_line "${ROOT}/${U250_EVIDENCE_REL}" 'target_name=DUAL_AMD_ALVEO_U250'
require_line "${ROOT}/${U250_EVIDENCE_REL}" 'u250_paired_card_count=1'
require_line "${ROOT}/${U250_EVIDENCE_REL}" 'u250_paired_slots=0000:d8:00'
require_line "${ROOT}/${U250_EVIDENCE_REL}" 'declared_u250_card_count=2'
require_line "${ROOT}/${U250_EVIDENCE_REL}" 'observed_u250_card_count=1'
require_line "${ROOT}/${U250_EVIDENCE_REL}" 'unresolved_u250_card_count=1'

parent_output="$("${ROOT}/${PARENT_GATE_REL}")"
printf '%s\n' "${parent_output}" | grep -Fqx -- \
  'pireus quotient novelty effect parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Koka role=EFFECT_PARITY effect_checks=4/4 checks=16 adversarial=15 formal=COMPLETE effect=COMPLETE material=OPEN_NOT_EXECUTED selected_child=-1 claim_ready=false python_process_launched=false rust_process_launched=false' || \
  fail 'Koka effect parity gate terminal marker drift'

wrong_parent='0dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21'
deny_without_dispatch PARENT_LAUNDERING \
  "$(authority_frame 3 4 4 4 1 0 0 0 0 "${wrong_parent}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${PARENT_LAUNDERING_FRAME_SHA256}" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN'
deny_without_dispatch CPP_SEMANTIC_WRITE \
  "$(authority_frame 3 4 4 4 1 1 0 0 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${SEMANTIC_WRITE_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch CPP_EXPECTED_RESULT_WRITE \
  "$(authority_frame 3 4 4 4 1 0 1 0 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${EXPECTED_WRITE_FRAME_SHA256}" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch REVIEW_PROMOTION \
  "$(authority_frame 3 4 4 4 1 0 0 0 1 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${REVIEW_PROMOTION_FRAME_SHA256}" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_MISSING \
  "$(authority_frame 3 4 4 4 0 0 0 0 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_TIMEOUT \
  "$(authority_frame 3 4 4 4 2 0 0 0 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_ERROR \
  "$(authority_frame 3 4 4 4 3 0 0 0 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${POLICY_ERROR_FRAME_SHA256}" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
deny_without_dispatch PREFREEZE \
  "$(authority_frame 2 4 4 4 1 0 0 0 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${PREFREEZE_FRAME_SHA256}" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
deny_without_dispatch CLAIM_PROMOTION \
  "$(authority_frame 4 7 4 4 1 0 0 1 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" "${XEON_EVIDENCE_SHA256}")" \
  "${CLAIM_PROMOTION_FRAME_SHA256}" 123 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=123 reason=action-forbidden-for-role next_stage=PARITY_OPEN'
deny_without_dispatch PYTHON_ORACLE \
  "$(authority_frame 3 4 7 7 1 0 0 0 0 "${SEMANTICS_SHA256}" "${PYTHON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${PYTHON_COMMAND_SHA256}" zero)" \
  "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
deny_without_dispatch RUST_ORACLE \
  "$(authority_frame 3 4 8 7 1 0 0 0 0 "${SEMANTICS_SHA256}" "${RUST_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${RUST_COMMAND_SHA256}" zero)" \
  "${RUST_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'

for target in apple dgx u250; do
  case "${target}" in
    apple)
      toolchain="${APPLE_TOOLCHAIN_SHA256}"; hardware="${APPLE_HARDWARE_SHA256}"
      command="${APPLE_COMMAND_SHA256}"; evidence="${APPLE_EVIDENCE_SHA256}"
      preexec="${APPLE_PREEXEC_FRAME_SHA256}"; seal="${APPLE_SEAL_FRAME_SHA256}"
      ;;
    dgx)
      toolchain="${DGX_TOOLCHAIN_SHA256}"; hardware="${DGX_HARDWARE_SHA256}"
      command="${DGX_COMMAND_SHA256}"; evidence="${DGX_EVIDENCE_SHA256}"
      preexec="${DGX_PREEXEC_FRAME_SHA256}"; seal="${DGX_SEAL_FRAME_SHA256}"
      ;;
    u250)
      toolchain="${U250_TOOLCHAIN_SHA256}"; hardware="${U250_HARDWARE_SHA256}"
      command="${U250_COMMAND_SHA256}"; evidence="${U250_EVIDENCE_SHA256}"
      preexec="${U250_PREEXEC_FRAME_SHA256}"; seal="${U250_SEAL_FRAME_SHA256}"
      ;;
  esac
  check_guardian "${target^^}_PREEXEC_RECEIPT" \
    "$(authority_frame 3 4 4 4 1 0 0 0 0 "${SEMANTICS_SHA256}" "${toolchain}" "${hardware}" "${command}" zero)" \
    "${preexec}" 0 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' RECEIPT_REPLAY
  printf 'GUARDIAN_DISPATCH label=%s_PREEXEC_RECEIPT process_launched=false receipt_replay=true\n' "${target^^}"
  check_guardian "${target^^}_SEAL_RECEIPT" \
    "$(authority_frame 4 8 4 4 1 0 0 1 0 "${SEMANTICS_SHA256}" "${toolchain}" "${hardware}" "${command}" "${evidence}")" \
    "${seal}" 0 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' RECEIPT_REPLAY
done

check_guardian XEON_PREEXEC \
  "$(authority_frame 3 4 4 4 1 0 0 0 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" zero)" \
  "${XEON_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' PREEXEC

mkdir -p "${BUILD_ROOT}"
g++ -std=c++17 -O2 "${CPP_REL}" -ldl -o "${BINARY}"
require_hash "${BINARY}" "${XEON_BINARY_SHA256}"
"${BINARY}" --target=xeon > "${TMP_ROOT}/xeon.txt"
require_hash "${TMP_ROOT}/xeon.txt" "${XEON_EVIDENCE_SHA256}"
cmp "${TMP_ROOT}/xeon.txt" "${ROOT}/${XEON_EVIDENCE_REL}" >/dev/null || fail 'Xeon replay evidence drift'

check_guardian XEON_SEAL \
  "$(authority_frame 4 8 4 4 1 0 0 1 0 "${SEMANTICS_SHA256}" "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}" "${XEON_EVIDENCE_SHA256}")" \
  "${XEON_SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' SEAL

require_line "${ROOT}/${RECEIPT_REL}" 'status=MATERIAL_PARITY_COMPLETE_WITH_UNRESOLVED_TARGETS'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_language=C++'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_role=MATERIAL_PARITY'
require_line "${ROOT}/${RECEIPT_REL}" 'receipt_authority=NON_SEMANTIC'
require_line "${ROOT}/${RECEIPT_REL}" 'checked_obligations=4'
require_line "${ROOT}/${RECEIPT_REL}" 'canonical_target_classes_declared=4'
require_line "${ROOT}/${RECEIPT_REL}" 'canonical_target_classes_observed=4'
require_line "${ROOT}/${RECEIPT_REL}" 'declared_physical_endpoints=6'
require_line "${ROOT}/${RECEIPT_REL}" 'observed_physical_endpoints=4'
require_line "${ROOT}/${RECEIPT_REL}" 'unresolved_physical_endpoints=2'
require_line "${ROOT}/${RECEIPT_REL}" 'unresolved_endpoint_01=DGX_SPARK_192.168.3.48'
require_line "${ROOT}/${RECEIPT_REL}" 'unresolved_endpoint_02=AMD_ALVEO_U250_SLOT_1'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'effect_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'material_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'material_target_coverage_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'target_lowerings=0'
require_line "${ROOT}/${RECEIPT_REL}" 'target_cost_records=0'
require_line "${ROOT}/${RECEIPT_REL}" 'target_performance_records=0'
require_line "${ROOT}/${RECEIPT_REL}" 'unresolved_target_obligations=1920'
require_line "${ROOT}/${RECEIPT_REL}" 'unresolved_target_obligations_source=sounio_frozen_result_contract'
require_line "${ROOT}/${RECEIPT_REL}" 'promotable_to_semantic_authority=false'
require_line "${ROOT}/${RECEIPT_REL}" 'selected_child=-1'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'
require_line "${ROOT}/${RECEIPT_REL}" 'python_forbidden_process_launched=false'
require_line "${ROOT}/${RECEIPT_REL}" 'rust_forbidden_process_launched=false'

printf '%s\n' \
  'pireus quotient novelty material parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=C++ role=MATERIAL_PARITY obligations=4/4 target_classes=4/4 endpoints=4/6 unresolved=2 formal=COMPLETE effect=COMPLETE material=COMPLETE coverage=PARTIAL target_lowerings=0 target_costs=0 target_performance=0 selected_child=-1 claim_ready=false python_process_launched=false rust_process_launched=false'
