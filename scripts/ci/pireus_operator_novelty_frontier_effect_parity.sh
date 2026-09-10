#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
KOKA="${PIREUS_KOKA_BIN:-/workspace/.home/openvscode-server/.local/pireus-toolchains/koka-v3.2.3/bin/koka}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/koka-v11-canonical'
RESULT="${BUILD_ROOT}/pireus-operator-novelty-frontier-effect-parity"
OUTPUT='/tmp/pireus-operator-novelty-frontier-effect-parity-v11.canonical.out'

SOUNIO_REL='stdlib/hardware/pireus/operator_novelty_frontier.sio'
FREEZE_REL='tools/pireus/operator_novelty_frontier.freeze.v11'
PARITY_OPEN_REL='tools/pireus/operator_novelty_frontier.parity-open.v11'
FORMAL_RECEIPT_REL='tools/pireus/operator_novelty_frontier.formal-parity.v11'
FORMAL_GATE_REL='scripts/ci/pireus_operator_novelty_frontier_formal_parity.sh'
KOKA_REL='formal/koka/pireus_operator_novelty_frontier_effect_parity.kk'
EVIDENCE_REL='tools/pireus/evidence/operator_novelty_frontier_v11.koka.txt'
RECEIPT_REL='tools/pireus/operator_novelty_frontier.effect-parity.v11'

FORMAL_PARITY_COMMIT='02dd802f0bc4c0af816029b08869acc08b233879'
SOUNIO_SHA256='9289cd504385e2f1f4eed095d82a963cf2e5e67124bf8d267d1bc6ccda7ac36b'
FREEZE_SHA256='b57decc8ff929640345e47edc931bdfa6cd06c738d3ff9591d3a460593dae242'
PARITY_OPEN_SHA256='f7cde0ed063d136bbef43cf9e820d734341f87717bb26e130a3643bc62fb31de'
FORMAL_RECEIPT_SHA256='b56b1f331879c2a8bbb70dc0adfc5ac61e21e922834c391ce4d815397a589d21'
FORMAL_GATE_SHA256='09f4e776c44875af757314a64d44dba3c2245cc76f6e8a7eb198d4d4d8e023e3'
SEMANTICS_SHA256='f1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc'
KOKA_SHA256='1c9b7cc0c87156278bd6b0079102b3cf7aabc630bab19c321531973d4899354f'
EVIDENCE_SHA256='9950d542685d7746626486ef7186cdf13ecb669a69dd9feeea736e2d93a97f75'
RECEIPT_SHA256='b18f91987a5b169bebb1a02d3b200f4ecae513c28f83f16dabaf3a96f2524d71'
KOKA_BINARY_SHA256='5268748ed5082f3693ddf9fa40e560020aa16b6be6bd52b86c97ce5435b24cba'
TOOLCHAIN_SHA256='273f70c80ed71dcfbe1ee077607ec435d8791e59032cc13e30e479fd25995332'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='ddd2baebadd294d4c9eb4f464db5af3275602f3a75bd81a6c10cfcf60d34f1df'
RESULT_SHA256='12c3fa91669c8311706614508f2431302f2f167d28ce5638076e0e62a1450093'
OUTPUT_SHA256='79d0fff657d8ff067823e8dc7a85f6857121b9ae97b40b0b5c3e70078482931f'
PREEXEC_FRAME_SHA256='1be409aec6ad61654ac2f8c073b2990e1f7cc39facd3f5882b6846ce2c35138c'
SEAL_FRAME_SHA256='e42c3f9620011752b48bce0e47887dd5403d89d9cd6ba12b6b904bfc6af039b4'
WRITE_FRAME_SHA256='48d4667e755b460ba17c2fbef909d079056d81355147996b86da47aae7261dad'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82

fail() {
  printf 'pireus operator novelty frontier effect parity: FAIL: %s\n' "$*" >&2
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
  rg -Fqx -- "${expected}" "${path}" || fail "missing exact line in ${path}: ${expected}"
}

unique_keys() {
  local path="$1"
  [[ -z "$(cut -d= -f1 "${path}" | sort | uniq -d)" ]]
}

source_admitted() {
  local path="$1"
  [[ "$(sha_file "${path}")" == "${KOKA_SHA256}" ]] &&
    rg -Fq 'effect operator-novelty-frontier-effect-parity' "${path}" &&
    rg -Fq 'fun frozen-handler' "${path}" &&
    rg -Fq 'fun run-parity' "${path}" &&
    [[ "$(rg -c '^  require\(' "${path}")" -eq 78 ]] &&
    [[ "$(rg -c '^  println\(' "${path}")" -eq 47 ]] &&
    rg -Fq 'effect_parity_status_scope=LOCAL_HANDLER_FIXTURE_EQUALITY_ONLY' "${path}" &&
    rg -Fq 'formal_proofs=0' "${path}" &&
    rg -Fq 'algebraic_laws_proved_by_koka=false' "${path}" &&
    rg -Fq 'candidate_selected_by_koka=false' "${path}" &&
    ! rg -Fq 'effect operator-novelty-frontier-authority' "${path}"
}

receipt_admitted() {
  local path="$1" key
  unique_keys "${path}" || return 1
  for key in status stage producing_language producing_role receipt_authority \
    effect_parity_scope effect_parity_status_scope system_wide_enforcement \
    local_fixture_equality_only parent_formal_parity_receipt_hash_matched \
    formal_parity_produced_by_koka effect_parity_complete material_parity_complete \
    semantic_write expected_result_write candidate_selected_by_koka \
    material_novelty historical_novelty priority_claim claim_ready; do
    [[ "$(rg -c "^${key}=" "${path}")" -eq 1 ]] || return 1
  done
  rg -Fqx 'status=EFFECT_PARITY_LOCAL_HANDLER_PASSED' "${path}" &&
    rg -Fqx 'stage=PARITY_OPEN' "${path}" &&
    rg -Fqx 'producing_language=Koka' "${path}" &&
    rg -Fqx 'producing_role=EFFECT_PARITY' "${path}" &&
    rg -Fqx 'receipt_authority=NON_SEMANTIC' "${path}" &&
    rg -Fqx 'discarded_attempts_authoritative=false' "${path}" &&
    rg -Fqx 'discarded_attempts_used_as_evidence=false' "${path}" &&
    rg -Fqx 'discarded_attempts_promoted=false' "${path}" &&
    rg -Fqx 'effect_parity_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY' "${path}" &&
    rg -Fqx 'effect_parity_status_scope=LOCAL_HANDLER_FIXTURE_EQUALITY_ONLY' "${path}" &&
    rg -Fqx 'system_wide_enforcement=false' "${path}" &&
    rg -Fqx 'local_fixture_equality_only=true' "${path}" &&
    rg -Fqx 'parent_formal_parity_receipt_hash_matched=true' "${path}" &&
    rg -Fqx 'formal_parity_produced_by_koka=false' "${path}" &&
    rg -Fqx 'effect_parity_complete=true' "${path}" &&
    rg -Fqx 'material_parity_complete=false' "${path}" &&
    rg -Fqx 'semantic_write=false' "${path}" &&
    rg -Fqx 'expected_result_write=false' "${path}" &&
    rg -Fqx 'candidate_selected_by_koka=false' "${path}" &&
    rg -Fqx 'material_novelty=false' "${path}" &&
    rg -Fqx 'historical_novelty=false' "${path}" &&
    rg -Fqx 'priority_claim=false' "${path}" &&
    rg -Fqx 'claim_ready=false' "${path}" &&
    rg -Fqx 'llm_confirmed_result=false' "${path}"
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8"
  local review_promoted="$9" parent_hash="${10}" toolchain_hash="${11}"
  local command_hash="${12}" result_hash="${13}" result_limbs="${ZERO}"
  if [[ "${result_hash}" != zero ]]; then
    result_limbs="$(sha_limbs "${result_hash}")"
  fi
  printf '9020 %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" \
    "${review_promoted}" "$(sha_limbs "${KOKA_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${toolchain_hash}")" "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${result_limbs}" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_hash="$3" expected_rc="$4" expected="$5"
  local mode="$6" decision rc
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq "${FRAME_WORDS}" ]] ||
    fail "${label}: frame word count drift"
  [[ "$(sha_text "${frame}")" == "${expected_hash}" ]] || fail "${label}: frame hash drift"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "${label}: rc=${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s rc=%s %s\n' \
    "${label}" "${expected_hash}" "${rc}" "${decision}"
  if [[ "${mode}" == deny ]]; then
    printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
  fi
}

deny() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4"
  check_guardian "${label}" "${frame}" "$(sha_text "${frame}")" "${expected_rc}" "${expected}" deny
}

cd "${ROOT}"
require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${FORMAL_RECEIPT_REL}" "${FORMAL_RECEIPT_SHA256}"
require_hash "${ROOT}/${FORMAL_GATE_REL}" "${FORMAL_GATE_SHA256}"
require_hash "${ROOT}/${KOKA_REL}" "${KOKA_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" "${GUARDIAN_POLICY_SHA256}"
require_hash "${KOKA}" "${KOKA_BINARY_SHA256}"
[[ -x "${GUARDIAN}" && -x "${KOKA}" ]] || fail 'required native tool unavailable'

git -C "${ROOT}" merge-base --is-ancestor "${FORMAL_PARITY_COMMIT}" HEAD ||
  fail 'formal parity commit is not an ancestor of HEAD'
[[ "$(git -C "${ROOT}" show "${FORMAL_PARITY_COMMIT}:${FORMAL_RECEIPT_REL}" | sha256sum | cut -d' ' -f1)" == "${FORMAL_RECEIPT_SHA256}" ]] ||
  fail 'committed formal receipt drift'
[[ "$(git -C "${ROOT}" show "${FORMAL_PARITY_COMMIT}:${FORMAL_GATE_REL}" | sha256sum | cut -d' ' -f1)" == "${FORMAL_GATE_SHA256}" ]] ||
  fail 'committed formal gate drift'

koka_version="$("${KOKA}" --version --console=raw | sed -n '1p')"
gcc_version="$(gcc --version | sed -n '1p')"
toolchain_record="koka=${koka_version} koka_binary_sha256=${KOKA_BINARY_SHA256} cc=${gcc_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'Koka toolchain record drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'effect parity hardware record drift'

source_admitted "${ROOT}/${KOKA_REL}" || fail 'Koka source admission failed'
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'Koka receipt admission failed'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'status=FORMAL_PARITY_COMPLETE'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'effect_parity_complete=false'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'claim_ready=false'

wrong_parent='e1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc'
deny WRONG_PARENT \
  "$(authority_frame 3 4 3 3 1 0 0 0 0 "${wrong_parent}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN'
deny KOKA_SEMANTIC_WRITE \
  "$(authority_frame 3 4 3 3 1 1 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
deny KOKA_EXPECTED_RESULT_WRITE \
  "$(authority_frame 3 4 3 3 1 0 1 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=SEMANTICS_FROZEN'
deny REVIEW_PROMOTION \
  "$(authority_frame 3 4 3 3 1 0 0 0 1 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
deny POLICY_MISSING \
  "$(authority_frame 3 4 3 3 0 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
deny POLICY_TIMEOUT \
  "$(authority_frame 3 4 3 3 2 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
deny POLICY_ERROR \
  "$(authority_frame 3 4 3 3 3 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
deny PYTHON_ORACLE \
  "$(authority_frame 3 4 7 7 1 0 0 0 0 "${SEMANTICS_SHA256}" "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}" zero)" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
deny RUST_ORACLE \
  "$(authority_frame 3 4 8 7 1 0 0 0 0 "${SEMANTICS_SHA256}" "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}" zero)" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
deny CLAIM_PROMOTION \
  "$(authority_frame 4 7 3 3 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${RESULT_SHA256}")" 123 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=123 reason=action-forbidden-for-role next_stage=PARITY_OPEN'

tmp_dir="$(mktemp -d /tmp/pireus-novelty-effect-gate.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT
sed 's/^claim_ready=false$/claim_ready=true/' "${ROOT}/${RECEIPT_REL}" >"${tmp_dir}/claim.v11"
receipt_admitted "${tmp_dir}/claim.v11" && fail 'claim promotion passed receipt admission'
sed 's/^system_wide_enforcement=false$/system_wide_enforcement=true/' "${ROOT}/${RECEIPT_REL}" >"${tmp_dir}/system.v11"
receipt_admitted "${tmp_dir}/system.v11" && fail 'system-wide overclaim passed receipt admission'
sed 's/effect operator-novelty-frontier-effect-parity/effect operator-novelty-frontier-authority/' \
  "${ROOT}/${KOKA_REL}" >"${tmp_dir}/authority.kk"
source_admitted "${tmp_dir}/authority.kk" && fail 'authority-widened source passed admission'
printf 'SABOTAGE claim_promotion=REFUSED system_wide=REFUSED source_authority=REFUSED\n'

check_guardian PREEXEC \
  "$(authority_frame 3 4 3 3 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' allow
mkdir -p "${BUILD_ROOT}/build"
"${KOKA}" -O2 --builddir="${BUILD_ROOT}/build" -o "${RESULT}" "${KOKA_REL}"
chmod 0755 "${RESULT}"
"${RESULT}" >"${OUTPUT}"
require_hash "${RESULT}" "${RESULT_SHA256}"
require_hash "${OUTPUT}" "${OUTPUT_SHA256}"
[[ "$(wc -l <"${OUTPUT}" | tr -d ' ')" -eq 47 ]] || fail 'output line count drift'
require_line "${OUTPUT}" 'checks=78'
require_line "${OUTPUT}" 'positive_checks=6'
require_line "${OUTPUT}" 'adversarial_checks=72'
require_line "${OUTPUT}" 'effect_parity_status_scope=LOCAL_HANDLER_FIXTURE_EQUALITY_ONLY'
require_line "${OUTPUT}" 'formal_proofs=0'
require_line "${OUTPUT}" 'claim_ready=false'

check_guardian SEAL \
  "$(authority_frame 4 8 3 3 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${RESULT_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' allow
check_guardian RECEIPT_WRITE \
  "$(authority_frame 4 9 3 3 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${RESULT_SHA256}")" \
  "${WRITE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' allow

require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'receipt drifted during gate'

printf '%s\n' \
  'pireus operator novelty frontier effect parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Koka role=EFFECT_PARITY scope=LOCAL_HANDLER_FIXTURE_EQUALITY_ONLY checks=78 positive=6 adversarial=72 formal=COMPLETE effect=COMPLETE material=OPEN_NOT_EXECUTED n3=false n4=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false'
