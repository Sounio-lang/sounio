#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
KOKA="${PIREUS_KOKA_BIN:-/workspace/.home/openvscode-server/.local/pireus-toolchains/koka-v3.2.3/bin/koka}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/koka-v7'
RESULT="${BUILD_ROOT}/pireus-operator-novelty-feedback-effect-parity"

SOUNIO_REL='stdlib/hardware/pireus/operator_novelty_feedback.sio'
FREEZE_REL='tools/pireus/operator_novelty_feedback.freeze.v7'
PARITY_OPEN_REL='tools/pireus/operator_novelty_feedback.parity-open.v7'
FORMAL_RECEIPT_REL='tools/pireus/operator_novelty_feedback.formal-parity.v7'
KOKA_REL='formal/koka/pireus_operator_novelty_feedback_effect_parity.kk'
EVIDENCE_REL='tools/pireus/evidence/operator_novelty_feedback_v7.koka.txt'
RECEIPT_REL='tools/pireus/operator_novelty_feedback.effect-parity.v7'
PARENT_GATE_REL='scripts/ci/pireus_operator_novelty_feedback_formal_parity.sh'
GUARDIAN_POLICY_REL='stdlib/coordination/loom_language_authority.sio'

FORMAL_GATE_COMMIT='79271b834ad3a5d84bcce54c37d78117940c8a0f'
EFFECT_PARITY_COMMIT='29912c0ed3c8ec504d6caaff85c8dc582f8e1447'
SOUNIO_SHA256='b73cc3fb6a905193a68a65eb6afd5d27da80395a0c38ae3772f9df56e8c8deaf'
SEMANTICS_SHA256='a1be292392727cf515baf6d95a376d6060d56f9b807fc58d8998fbe23bdc7726'
FREEZE_SHA256='7293594eb7a881d1f89d9593b1cc19e3e611f99a491a4cd1146afe0a68cd623a'
PARITY_OPEN_SHA256='1ae8fe022071d12193624477f531595a789ffd05f97489e4ccd05d93cf78f7ef'
FORMAL_RECEIPT_SHA256='305b75a4bb40f2568ff743780f5866565ac5da8e9d46c2fee9952f6990248b47'
KOKA_SHA256='b9b131d50494f0bc5570033f5d19c4fc1bd3c715acdb0fc993be2cf1085edf13'
EVIDENCE_SHA256='85873bccb1553cde028b7f45d9f15055f8bca8eb20c240e6ad9600ebd1b27eed'
RECEIPT_SHA256='9f4235dd7b178540cf74b169aab2dafbfbb44e199c1f7523de83dd9c967e2e0b'
PARENT_GATE_SHA256='1a15bc9981ff9e43cdebb4ffbcd29fa5f14750d083f750dde6e753835854f3c2'
KOKA_BINARY_SHA256='5268748ed5082f3693ddf9fa40e560020aa16b6be6bd52b86c97ce5435b24cba'
TOOLCHAIN_SHA256='273f70c80ed71dcfbe1ee077607ec435d8791e59032cc13e30e479fd25995332'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='29c6687dd85adad88b89b786ce68c88bbb00f04eb094f244d40ba5794f743d72'
RESULT_SHA256='7a95604d563ed4868247be7c2f03f831fe8b40c3a1cd4380e1116254394a3d97'
PREEXEC_FRAME_SHA256='c21e87e61ccd93bad30a4daacf663c38a02ae1a50762658659977ab8bbf2ea28'
SEAL_FRAME_SHA256='bff8143ea358426de0ab886e8e73b30cdebfe0d44742af79af6725e0d7408a3a'
WRONG_PARENT_FRAME_SHA256='f6e6cdfe91094c0eeb6e86219035427fdc5826fb6e8fd2686510065fbfd721af'
SEMANTIC_WRITE_FRAME_SHA256='3fdc9d826a66bba5dd1d3e6d64272855f091dc55932ea6f20c2012d1ea5ed0b1'
EXPECTED_WRITE_FRAME_SHA256='38e6f55de5b59aaba1d0d97dcc2d015ab77eb83d7cb30ac6ed055caaba6595d5'
REVIEW_PROMOTION_FRAME_SHA256='62eff968e14a5e875e15595d26bbd3082b939cf1b2dbb95eacc0a718e55cb98c'
POLICY_MISSING_FRAME_SHA256='bc41917af4a160ea1667522b2d328e3aacb4d22147a51c9fd5cb5f2afbfd7cfc'
POLICY_TIMEOUT_FRAME_SHA256='14481adc101b99d25ade94ff6a4275bfed3cd17b8050e64f5ea6d41f5c3e1577'
POLICY_ERROR_FRAME_SHA256='5295b527e33298ff1eaedf42412a3d2d1335b586f040aaca38e0332a63df663c'
PYTHON_FRAME_SHA256='94b8608415ccf3ce9d5274703c725cfe15c4e24cc9fbe03ddbbbfc6fa5d43f75'
RUST_FRAME_SHA256='fb777ba2671a2ac628c4558ba326137af18a1b7a8066e610c025cb14555afca1'
CLAIM_PROMOTION_FRAME_SHA256='0d1e3334ee7fdc2ffaeb1f74c8d780a6760b810a592e006e3c13afca8d0b72d9'
BAD_SEAL_FRAME_SHA256='a936b28b36d72be959e8c40e08ac9b6ce415303b78768ee01559fab1b8abf870'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
WRONG_PARENT='01be292392727cf515baf6d95a376d6060d56f9b807fc58d8998fbe23bdc7726'
ZERO='0 0 0 0 0 0 0 0'
GUARDIAN_FRAME_SCHEMA=9020
GUARDIAN_FRAME_WORDS=82

fail() {
  printf 'pireus operator novelty effect parity: FAIL: %s\n' "$*" >&2
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

unique_keys() {
  local path="$1"
  [[ -z "$(cut -d= -f1 "${path}" | sort | uniq -d)" ]]
}

source_admitted() {
  local path="$1"
  [[ "$(sha_file "${path}")" == "${KOKA_SHA256}" ]] &&
    grep -Fq 'effect novelty-effect-parity' "${path}" &&
    grep -Fq 'fun frozen-handler' "${path}" &&
    grep -Fq 'fun run-parity' "${path}" &&
    [[ "$(grep -c '^  require(' "${path}")" -eq 50 ]] &&
    ! grep -Fq 'effect novelty-authority' "${path}" &&
    ! grep -Fq 'system_wide_enforcement=true' "${path}"
}

evidence_admitted() {
  local path="$1"
  unique_keys "${path}" &&
    grep -Fqx 'status=EFFECT_PARITY_COMPLETE' "${path}" &&
    grep -Fqx 'successful_attempt=CANONICAL_03' "${path}" &&
    grep -Fqx 'discarded_attempt_01_authoritative=false' "${path}" &&
    grep -Fqx 'discarded_attempt_02_authoritative=false' "${path}" &&
    grep -Fqx 'effect_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY' "${path}" &&
    grep -Fqx 'system_wide_enforcement=false' "${path}" &&
    grep -Fqx 'local_fixture_equality_only=true' "${path}" &&
    grep -Fqx 'local_handler_checks=8' "${path}" &&
    grep -Fqx 'checks=50' "${path}" &&
    grep -Fqx 'adversarial_checks=47' "${path}" &&
    grep -Fqx 'program_output_lines=43' "${path}" &&
    grep -Fqx 'claim_ready=false' "${path}"
}

receipt_admitted() {
  local path="$1" key
  unique_keys "${path}" || return 1
  for key in \
    status stage producing_language producing_role effect_parity_scope \
    system_wide_enforcement local_fixture_equality_only \
    formal_parity_complete effect_parity_complete material_parity_complete \
    semantic_write expected_result_write operator_seed_generated_by_koka \
    broad_novelty historical_novelty priority_claim claim_ready; do
    [[ "$(grep -c "^${key}=" "${path}")" -eq 1 ]] || return 1
  done
  grep -Fqx 'status=EFFECT_PARITY_COMPLETE' "${path}" &&
    grep -Fqx 'stage=PARITY_OPEN' "${path}" &&
    grep -Fqx 'producing_language=Koka' "${path}" &&
    grep -Fqx 'producing_role=EFFECT_PARITY' "${path}" &&
    grep -Fqx 'effect_parity_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY' "${path}" &&
    grep -Fqx 'system_wide_enforcement=false' "${path}" &&
    grep -Fqx 'local_fixture_equality_only=true' "${path}" &&
    grep -Fqx 'formal_parity_complete=true' "${path}" &&
    grep -Fqx 'effect_parity_complete=true' "${path}" &&
    grep -Fqx 'material_parity_complete=false' "${path}" &&
    grep -Fqx 'semantic_write=false' "${path}" &&
    grep -Fqx 'expected_result_write=false' "${path}" &&
    grep -Fqx 'operator_seed_generated_by_koka=false' "${path}" &&
    grep -Fqx 'broad_novelty=false' "${path}" &&
    grep -Fqx 'historical_novelty=false' "${path}" &&
    grep -Fqx 'priority_claim=false' "${path}" &&
    grep -Fqx 'claim_ready=false' "${path}"
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8"
  local review_promoted="$9" parent_hash="${10}" toolchain_hash="${11}"
  local command_hash="${12}" result_hash="${13}" result_limbs="${ZERO}"
  if [[ "${result_hash}" != zero ]]; then
    result_limbs="$(sha_limbs "${result_hash}")"
  fi
  printf '%s %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${GUARDIAN_FRAME_SCHEMA}" \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" \
    "${review_promoted}" "$(sha_limbs "${KOKA_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${toolchain_hash}")" "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${result_limbs}" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4"
  local expected="$5" mode="$6" decision rc
  [[ "${frame%% *}" == "${GUARDIAN_FRAME_SCHEMA}" ]] ||
    fail "Guardian frame schema drift for ${label}"
  [[ "$(wc -w <<< "${frame}")" -eq "${GUARDIAN_FRAME_WORDS}" ]] ||
    fail "Guardian frame field-count drift for ${label}"
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "Guardian frame hash drift for ${label}"
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
  if [[ "${mode}" == deny ]]; then
    printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
  fi
}

for pair in \
  "${SOUNIO_REL}:${SOUNIO_SHA256}" \
  "${FREEZE_REL}:${FREEZE_SHA256}" \
  "${PARITY_OPEN_REL}:${PARITY_OPEN_SHA256}" \
  "${FORMAL_RECEIPT_REL}:${FORMAL_RECEIPT_SHA256}" \
  "${KOKA_REL}:${KOKA_SHA256}" \
  "${EVIDENCE_REL}:${EVIDENCE_SHA256}" \
  "${RECEIPT_REL}:${RECEIPT_SHA256}" \
  "${PARENT_GATE_REL}:${PARENT_GATE_SHA256}" \
  "${GUARDIAN_POLICY_REL}:${GUARDIAN_POLICY_SHA256}"
do
  require_hash "${ROOT}/${pair%%:*}" "${pair##*:}"
done
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
[[ -x "${KOKA}" ]] || fail 'Koka 3.2.3 executable unavailable'
require_hash "${KOKA}" "${KOKA_BINARY_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor \
  "${FORMAL_GATE_COMMIT}" "${EFFECT_PARITY_COMMIT}" ||
  fail 'effect parity predates the formal parity gate'
git -C "${ROOT}" merge-base --is-ancestor "${EFFECT_PARITY_COMMIT}" HEAD ||
  fail 'effect parity commit missing from current history'
require_committed_hash "${FORMAL_GATE_COMMIT}" "${PARENT_GATE_REL}" \
  "${PARENT_GATE_SHA256}"
require_committed_hash "${EFFECT_PARITY_COMMIT}" "${KOKA_REL}" "${KOKA_SHA256}"
require_committed_hash "${EFFECT_PARITY_COMMIT}" "${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_committed_hash "${EFFECT_PARITY_COMMIT}" "${RECEIPT_REL}" "${RECEIPT_SHA256}"

source_admitted "${ROOT}/${KOKA_REL}" || fail 'Koka source admission failed'
evidence_admitted "${ROOT}/${EVIDENCE_REL}" || fail 'effect evidence admission failed'
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'effect receipt admission failed'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'koka_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'axiom_closure=BOUNDED_LEAN_FOUNDATION'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'effect_parity_complete=false'

koka_version="$("${KOKA}" --version --console=raw | sed -n '1p')"
gcc_version="$(gcc --version | sed -n '1p')"
toolchain_record="koka=${koka_version} koka_binary_sha256=${KOKA_BINARY_SHA256} cc=${gcc_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'Koka toolchain drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware drift'
command_record='/workspace/.home/openvscode-server/.local/pireus-toolchains/koka-v3.2.3/bin/koka -O2 --builddir=/workspace/.home/openvscode-server/.cache/pireus/koka-v7/build -o /workspace/.home/openvscode-server/.cache/pireus/koka-v7/pireus-operator-novelty-feedback-effect-parity formal/koka/pireus_operator_novelty_feedback_effect_parity.kk && chmod 0755 /workspace/.home/openvscode-server/.cache/pireus/koka-v7/pireus-operator-novelty-feedback-effect-parity && /workspace/.home/openvscode-server/.cache/pireus/koka-v7/pireus-operator-novelty-feedback-effect-parity'
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'command drift'

set +e
invalid_hash_output="$(sha_limbs 'not-a-sha256' 2>&1)"
invalid_hash_rc=$?
set -e
[[ "${invalid_hash_rc}" -eq 1 ]] || fail 'malformed SHA-256 did not fail closed'
[[ "${invalid_hash_output}" == \
  'pireus operator novelty effect parity: FAIL: invalid sha256 digest: not-a-sha256' ]] ||
  fail 'malformed SHA-256 refusal drift'
printf 'GUARDIAN_DISPATCH label=MALFORMED_SHA256 process_launched=false\n'

check_guardian WRONG_PARENT \
  "$(authority_frame 3 4 3 3 1 0 0 0 0 "${WRONG_PARENT}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${WRONG_PARENT_FRAME_SHA256}" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN' deny
check_guardian KOKA_SEMANTIC_WRITE \
  "$(authority_frame 3 4 3 3 1 1 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${SEMANTIC_WRITE_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN' deny
check_guardian KOKA_EXPECTED_RESULT_WRITE \
  "$(authority_frame 3 4 3 3 1 0 1 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${EXPECTED_WRITE_FRAME_SHA256}" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=SEMANTICS_FROZEN' deny
check_guardian REVIEW_PROMOTION \
  "$(authority_frame 3 4 3 3 1 0 0 0 1 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${REVIEW_PROMOTION_FRAME_SHA256}" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN' deny
check_guardian POLICY_MISSING \
  "$(authority_frame 3 4 3 3 0 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN' deny
check_guardian POLICY_TIMEOUT \
  "$(authority_frame 3 4 3 3 2 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN' deny
check_guardian POLICY_ERROR \
  "$(authority_frame 3 4 3 3 3 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${POLICY_ERROR_FRAME_SHA256}" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN' deny
check_guardian PYTHON_ORACLE \
  "$(authority_frame 3 4 7 7 1 0 0 0 0 "${SEMANTICS_SHA256}" "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}" zero)" \
  "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' deny
check_guardian RUST_ORACLE \
  "$(authority_frame 3 4 8 7 1 0 0 0 0 "${SEMANTICS_SHA256}" "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}" zero)" \
  "${RUST_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' deny
check_guardian CLAIM_PROMOTION \
  "$(authority_frame 4 7 3 3 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${RESULT_SHA256}")" \
  "${CLAIM_PROMOTION_FRAME_SHA256}" 123 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=123 reason=action-forbidden-for-role next_stage=PARITY_OPEN' deny
check_guardian BAD_SEAL \
  "$(authority_frame 4 8 3 3 1 0 1 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${RESULT_SHA256}")" \
  "${BAD_SEAL_FRAME_SHA256}" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=PARITY_OPEN' deny

tmp_dir="$(mktemp -d /tmp/pireus-onf-effect-gate.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT
sed 's/^claim_ready=false$/claim_ready=true/' "${ROOT}/${RECEIPT_REL}" \
  >"${tmp_dir}/claim-promoted.v7"
receipt_admitted "${tmp_dir}/claim-promoted.v7" &&
  fail 'claim promotion sabotage passed receipt admission'
sed 's/^system_wide_enforcement=false$/system_wide_enforcement=true/' \
  "${ROOT}/${RECEIPT_REL}" >"${tmp_dir}/system-wide.v7"
receipt_admitted "${tmp_dir}/system-wide.v7" &&
  fail 'system-wide enforcement sabotage passed receipt admission'
sed 's/^discarded_attempt_01_authoritative=false$/discarded_attempt_01_authoritative=true/' \
  "${ROOT}/${EVIDENCE_REL}" >"${tmp_dir}/discarded-promoted.v7"
evidence_admitted "${tmp_dir}/discarded-promoted.v7" &&
  fail 'discarded typecheck attempt was promoted'
sed 's/effect novelty-effect-parity/effect novelty-authority/' \
  "${ROOT}/${KOKA_REL}" >"${tmp_dir}/authority-widened.kk"
source_admitted "${tmp_dir}/authority-widened.kk" &&
  fail 'authority-widened Koka source was admitted'
printf 'SABOTAGE claim_promotion=REFUSED system_wide=REFUSED discarded_attempt=REFUSED source_authority=REFUSED\n'

parent_gate_output="$("${ROOT}/${PARENT_GATE_REL}")"
printf '%s\n' "${parent_gate_output}" | grep -Fqx -- \
  'pireus operator novelty formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY certificate_nodes=39/39 atlas_pairs=168 zero_residual_hits=0 outcome=OPERATOR_SEED positive_bridge_branch=NOT_RECONSTRUCTED axiom_closure=BOUNDED_LEAN_FOUNDATION formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED broad_novelty=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'Lean formal parity gate terminal marker drift'

check_guardian PREEXEC \
  "$(authority_frame 3 4 3 3 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' allow

require_hash "${ROOT}/${KOKA_REL}" "${KOKA_SHA256}"
source_admitted "${ROOT}/${KOKA_REL}" || fail 'Koka source drifted before build'
mkdir -p "${BUILD_ROOT}/build"
(cd "${ROOT}" && "${KOKA}" -O2 \
  --builddir="${BUILD_ROOT}/build" \
  -o "${RESULT}" \
  "${KOKA_REL}")
chmod 0755 "${RESULT}"
require_hash "${RESULT}" "${RESULT_SHA256}"
require_hash "${ROOT}/${KOKA_REL}" "${KOKA_SHA256}"
source_admitted "${ROOT}/${KOKA_REL}" || fail 'Koka source drifted during build'

program_output="$("${RESULT}")"
expected_output="$(printf '%s\n' \
  'schema=pireus-operator-novelty-feedback-local-handler-receipt-v7' \
  'producing_language=Koka' \
  'producing_role=EFFECT_PARITY' \
  "sounio_source_sha256=${SOUNIO_SHA256}" \
  "sounio_semantics_sha256=${SEMANTICS_SHA256}" \
  "formal_parity_receipt_sha256=${FORMAL_RECEIPT_SHA256}" \
  'effect_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY' \
  'system_wide_enforcement=false' \
  'local_fixture_equality_only=true' \
  'not_semantic_authority=true' \
  'not_formal_authority=true' \
  'not_material_authority=true' \
  'not_novelty_confirmation=true' \
  'transition_request_shape_effect=true' \
  'exact_snapshot_read_effect=true' \
  'exact_seed_record_read_effect=true' \
  'handler_positive_bridge_operation_denied=true' \
  'handler_atlas_mutation_denied=true' \
  'handler_lowering_dispatch_denied=true' \
  'handler_target_dispatch_denied=true' \
  'handler_claim_operations_denied=true' \
  'local_handler_checks=8' \
  'formal_proofs=0' \
  'checks=50' \
  'adversarial_checks=47' \
  'frozen_input_atlas_classes=14' \
  'frozen_input_atlas_actions=12' \
  'frozen_input_atlas_pairs=168' \
  'frozen_input_zero_residual_hits=0' \
  'frozen_input_outcome=OPERATOR_SEED' \
  'frozen_input_best_class=8' \
  'frozen_input_best_representative=13' \
  'frozen_input_best_action_code=68674' \
  'frozen_input_best_residual_nonzero=96' \
  'positive_bridge_branch_reconstructed=false' \
  'semantic_write=false' \
  'expected_result_write=false' \
  'lowering_processes_launched=0' \
  'target_processes_launched=0' \
  'broad_novelty=false' \
  'historical_novelty=false' \
  'priority_claim=false' \
  'claim_ready=false')"
[[ "${program_output}" == "${expected_output}" ]] ||
  fail 'Koka local-handler output drift'

check_guardian SEAL \
  "$(authority_frame 4 8 3 3 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${RESULT_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' allow

require_hash "${ROOT}/${KOKA_REL}" "${KOKA_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
evidence_admitted "${ROOT}/${EVIDENCE_REL}" ||
  fail 'effect evidence drifted before terminal marker'
receipt_admitted "${ROOT}/${RECEIPT_REL}" ||
  fail 'effect receipt drifted before terminal marker'

printf '%s\n' \
  'pireus operator novelty effect parity: STAGE_REACHED_NOT_A_CLAIM gate_mode=CONTENT_ADDRESSED_REPLAY stage=PARITY_OPEN language=Koka role=EFFECT_PARITY effect_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY local_handler_checks=8/8 checks=50 adversarial=47 formal=COMPLETE effect=COMPLETE material=OPEN_NOT_EXECUTED broad_novelty=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false'
