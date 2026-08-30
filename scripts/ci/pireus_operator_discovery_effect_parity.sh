#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
KOKA="${PIREUS_KOKA_BIN:-/workspace/.home/openvscode-server/.local/pireus-toolchains/koka-v3.2.3/bin/koka}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/koka-v10-canonical'
RESULT="${BUILD_ROOT}/pireus-operator-discovery-effect-parity"
OUTPUT='/tmp/pireus-operator-discovery-effect-parity-v10.canonical.out'

SOUNIO_REL='stdlib/hardware/pireus/operator_discovery_engine.sio'
FREEZE_REL='tools/pireus/operator_discovery_engine.freeze.v10'
PARITY_OPEN_REL='tools/pireus/operator_discovery_engine.parity-open.v10'
FORMAL_RECEIPT_REL='tools/pireus/operator_discovery_engine.formal-parity.v10'
FORMAL_GATE_REL='scripts/ci/pireus_operator_discovery_formal_parity.sh'
KOKA_REL='formal/koka/pireus_operator_discovery_effect_parity.kk'
EVIDENCE_REL='tools/pireus/evidence/operator_discovery_engine_v10.koka.txt'
RECEIPT_REL='tools/pireus/operator_discovery_engine.effect-parity.v10'

FORMAL_PARITY_COMMIT='ab4e0e434b42b9deeb04a46dfb2a66cb5b1988b5'
SOUNIO_SHA256='919b6104cbce1c5f8643f5df88b9071305d3fee854f785ac63a883bc45f16117'
FREEZE_SHA256='9a83c9a4b920d41ee91bd7681f4e95ac11480d762185ec9ff003692d3c01d247'
PARITY_OPEN_SHA256='5f109404d2a2e8e56e6cff486f871e0961f843edd2e48e2feb5f5717d1d8d39d'
FORMAL_RECEIPT_SHA256='dddc85352de064baeee09da91917ecc3790ac5fd362ba29b4dc204d86addaa30'
FORMAL_GATE_SHA256='3ecf2818873ceb3684ab4d0d0c4b333b2c577a01ba5f88b2193afadd7ea64e12'
SEMANTICS_SHA256='2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5'
KOKA_SHA256='9ebfd12a0228f80631cdc5702175f24f8e1e8bb30d9818074f161d905a29ed9b'
EVIDENCE_SHA256='38d87503b4c684bf941b98db2bc86443afefb6dd2b81448d2bbf5de6912e1c45'
RECEIPT_SHA256='eb8778c8ab7bf1627ef915ef6412bbc3de1e81e0807df7459858a03ecfe4d537'
KOKA_BINARY_SHA256='5268748ed5082f3693ddf9fa40e560020aa16b6be6bd52b86c97ce5435b24cba'
TOOLCHAIN_SHA256='273f70c80ed71dcfbe1ee077607ec435d8791e59032cc13e30e479fd25995332'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='2f802823f5ae06c100f2b0681d5a15c5063c326ed240c76e0d2dfaf09a67c6ce'
RESULT_SHA256='efaae407fdbd26b9fd169806b6359e236f240638f648e39f1258b829e2985bd4'
OUTPUT_SHA256='4fc5d409a47eb42e0772af5e66117c62b403d105a81401988ffcff9e8157e08c'
PREEXEC_FRAME_SHA256='393174951efa62a86a9e3ac4734f90b53f8a4d092ce8167b16b24c9f75cef927'
SEAL_FRAME_SHA256='1d2c226f3344d642dd8b8e8845b8a6cf66433b91f3da3bc127c93e128f054f0c'
WRITE_FRAME_SHA256='ce9674b1ed16af8c3b1dc685d444ded0ce6c085b7a15accd74da473ae3c73e55'
WRONG_PARENT_FRAME_SHA256='e2d87cbbc5f53320c7722b0700c80c8c06f6eeef63769767dfd9b9fdaa9a33bf'
SEMANTIC_WRITE_FRAME_SHA256='3e508d53cd9e178cb15b9d17160c7630fa5921f9abaedc2b5b5c00f7e83d8dbc'
EXPECTED_WRITE_FRAME_SHA256='cf28ee6bdb25bb790c84b6c8db018d3dcd1b636fbd10db481dd37b86676478a3'
REVIEW_PROMOTION_FRAME_SHA256='2f49f7525e40aa1f804401b0999b829632669629cc675f501e510f175a1a6cf6'
POLICY_MISSING_FRAME_SHA256='084424fd5de971267776360233081be9f81b73ce1021eda5293716e8cfbf7666'
POLICY_TIMEOUT_FRAME_SHA256='06f1f3435e2af6d3415d92f61934bd45b4fa430a97b5027d274388cf3617236c'
POLICY_ERROR_FRAME_SHA256='8c7e7e6ebe47a1708d8bda4257435048b2f5903dcc71715a8ad76250756e20f7'
PYTHON_FRAME_SHA256='b2f5afc20bceca14619729a131fdf249c5ceb086e1c25b2d82c251de7bd75895'
RUST_FRAME_SHA256='fe1b366274340fdd6d420b20ab04467be8973223e960baa4038bec0a331eae79'
CLAIM_PROMOTION_FRAME_SHA256='f147eb940d05664c1b61136695d17507cff20c0f87c32de6524bcf3713c0ff0e'
BAD_SEAL_FRAME_SHA256='71c444317d0ab90c01c07785b0185583b18102f4f61ea88422854ec8c1abe474'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
WRONG_PARENT='1640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82

fail() {
  printf 'pireus operator discovery effect parity: FAIL: %s\n' "$*" >&2
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
    grep -Fq 'effect operator-discovery-effect-parity' "${path}" &&
    grep -Fq 'fun frozen-handler' "${path}" &&
    grep -Fq 'fun run-parity' "${path}" &&
    [[ "$(grep -c '^  require(' "${path}")" -eq 71 ]] &&
    [[ "$(grep -c '^  println(' "${path}")" -eq 53 ]] &&
    grep -Fq 'algebraic_laws_proved_by_koka=false' "${path}" &&
    grep -Fq 'separator_family_proved_by_koka=false' "${path}" &&
    grep -Fq 'action_census_derived_by_koka=false' "${path}" &&
    ! grep -Fq 'effect operator-discovery-authority' "${path}"
}

evidence_admitted() {
  local path="$1"
  unique_keys "${path}" &&
    grep -Fqx 'status=EFFECT_PARITY_LOCAL_HANDLER_PASSED' "${path}" &&
    grep -Fqx 'discarded_attempt_01_status=DISCARDED_NONCOMPLIANT' "${path}" &&
    grep -Fqx 'discarded_attempt_01_diagnostic_replay_authorized=false' "${path}" &&
    grep -Fqx 'discarded_attempt_01_diagnostic_replay_processes_launched=1' "${path}" &&
    grep -Fqx 'discarded_attempt_01_authoritative=false' "${path}" &&
    grep -Fqx 'discarded_attempt_01_used_as_evidence=false' "${path}" &&
    grep -Fqx 'discarded_attempt_01_promoted=false' "${path}" &&
    grep -Fqx 'canonical_attempt=CANONICAL_02_CLEAN_ROOT' "${path}" &&
    grep -Fqx 'canonical_attempt_all_processes_preexec_authorized=true' "${path}" &&
    grep -Fqx 'checks=71' "${path}" &&
    grep -Fqx 'positive_checks=4' "${path}" &&
    grep -Fqx 'adversarial_checks=67' "${path}" &&
    grep -Fqx 'effect_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY' "${path}" &&
    grep -Fqx 'effect_parity_status_scope=LOCAL_HANDLER_FIXTURE_EQUALITY_ONLY' "${path}" &&
    grep -Fqx 'system_wide_enforcement=false' "${path}" &&
    grep -Fqx 'local_fixture_equality_only=true' "${path}" &&
    grep -Fqx 'parent_formal_parity_receipt_admitted=true' "${path}" &&
    grep -Fqx 'formal_parity_produced_by_koka=false' "${path}" &&
    grep -Fqx 'llm_confirmed_result=false' "${path}" &&
    grep -Fqx 'claim_ready=false' "${path}" &&
    ! grep -q '^formal_parity_complete=' "${path}" &&
    ! grep -q '^diagnostic_replay_correction=' "${path}"
}

receipt_admitted() {
  local path="$1" key
  unique_keys "${path}" || return 1
  for key in status stage producing_language producing_role receipt_authority \
    effect_parity_scope effect_parity_status_scope system_wide_enforcement \
    local_fixture_equality_only parent_formal_parity_receipt_admitted \
    formal_parity_produced_by_koka effect_parity_complete \
    material_parity_complete semantic_write expected_result_write \
    candidate_selected_by_koka material_novelty historical_novelty \
    priority_claim claim_ready; do
    [[ "$(grep -c "^${key}=" "${path}")" -eq 1 ]] || return 1
  done
  grep -Fqx 'status=EFFECT_PARITY_LOCAL_HANDLER_PASSED' "${path}" &&
    grep -Fqx 'stage=PARITY_OPEN' "${path}" &&
    grep -Fqx 'producing_language=Koka' "${path}" &&
    grep -Fqx 'producing_role=EFFECT_PARITY' "${path}" &&
    grep -Fqx 'receipt_authority=NON_SEMANTIC' "${path}" &&
    grep -Fqx 'discarded_attempt_01_status=DISCARDED_NONCOMPLIANT' "${path}" &&
    grep -Fqx 'discarded_attempt_01_authoritative=false' "${path}" &&
    grep -Fqx 'discarded_attempt_01_used_as_evidence=false' "${path}" &&
    grep -Fqx 'discarded_attempt_01_promoted=false' "${path}" &&
    grep -Fqx 'canonical_attempt_all_processes_preexec_authorized=true' "${path}" &&
    grep -Fqx 'effect_parity_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY' "${path}" &&
    grep -Fqx 'effect_parity_status_scope=LOCAL_HANDLER_FIXTURE_EQUALITY_ONLY' "${path}" &&
    grep -Fqx 'system_wide_enforcement=false' "${path}" &&
    grep -Fqx 'local_fixture_equality_only=true' "${path}" &&
    grep -Fqx 'parent_formal_parity_receipt_admitted=true' "${path}" &&
    grep -Fqx 'formal_parity_produced_by_koka=false' "${path}" &&
    grep -Fqx 'effect_parity_complete=true' "${path}" &&
    grep -Fqx 'material_parity_complete=false' "${path}" &&
    grep -Fqx 'semantic_write=false' "${path}" &&
    grep -Fqx 'expected_result_write=false' "${path}" &&
    grep -Fqx 'candidate_selected_by_koka=false' "${path}" &&
    grep -Fqx 'material_novelty=false' "${path}" &&
    grep -Fqx 'historical_novelty=false' "${path}" &&
    grep -Fqx 'priority_claim=false' "${path}" &&
    grep -Fqx 'claim_ready=false' "${path}" &&
    grep -Fqx 'llm_confirmed_result=false' "${path}" &&
    ! grep -q '^formal_parity_complete=' "${path}"
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
  [[ "$(sha_text "${frame}")" == "${expected_hash}" ]] ||
    fail "${label}: frame hash drift"
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
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"
require_hash "${KOKA}" "${KOKA_BINARY_SHA256}"
[[ -x "${GUARDIAN}" && -x "${KOKA}" ]] || fail 'required native tool unavailable'

git -C "${ROOT}" merge-base --is-ancestor "${FORMAL_PARITY_COMMIT}" HEAD ||
  fail 'formal parity commit is not an ancestor of HEAD'
require_committed_hash "${FORMAL_PARITY_COMMIT}" "${FORMAL_RECEIPT_REL}" \
  "${FORMAL_RECEIPT_SHA256}"
require_committed_hash "${FORMAL_PARITY_COMMIT}" "${FORMAL_GATE_REL}" \
  "${FORMAL_GATE_SHA256}"

koka_version="$("${KOKA}" --version --console=raw | sed -n '1p')"
gcc_version="$(gcc --version | sed -n '1p')"
toolchain_record="koka=${koka_version} koka_binary_sha256=${KOKA_BINARY_SHA256} cc=${gcc_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'Koka toolchain record drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'effect parity hardware record drift'

source_admitted "${ROOT}/${KOKA_REL}" || fail 'Koka source admission failed'
evidence_admitted "${ROOT}/${EVIDENCE_REL}" || fail 'Koka evidence admission failed'
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'Koka receipt admission failed'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'status=FORMAL_PARITY_COMPLETE'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'effect_parity_complete=false'
require_line "${ROOT}/${FORMAL_RECEIPT_REL}" 'claim_ready=false'

set +e
invalid_hash_output="$(sha_limbs not-a-sha256 2>&1)"
invalid_hash_rc=$?
set -e
[[ "${invalid_hash_rc}" -eq 1 ]] || fail "malformed SHA-256 rc=${invalid_hash_rc}"
[[ "${invalid_hash_output}" == \
  'pireus operator discovery effect parity: FAIL: invalid sha256 digest: not-a-sha256' ]] ||
  fail 'malformed SHA-256 did not fail closed'
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
  "$(authority_frame 4 8 3 3 1 0 1 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${RESULT_SHA256}")" \
  "${BAD_SEAL_FRAME_SHA256}" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=PARITY_OPEN' deny

tmp_dir="$(mktemp -d /tmp/pireus-operator-discovery-effect-gate.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT
sed 's/^claim_ready=false$/claim_ready=true/' "${ROOT}/${RECEIPT_REL}" \
  >"${tmp_dir}/claim-promoted.v10"
receipt_admitted "${tmp_dir}/claim-promoted.v10" &&
  fail 'claim promotion sabotage passed receipt admission'
sed 's/^system_wide_enforcement=false$/system_wide_enforcement=true/' \
  "${ROOT}/${RECEIPT_REL}" >"${tmp_dir}/system-wide.v10"
receipt_admitted "${tmp_dir}/system-wide.v10" &&
  fail 'system-wide enforcement sabotage passed receipt admission'
sed 's/^discarded_attempt_01_authoritative=false$/discarded_attempt_01_authoritative=true/' \
  "${ROOT}/${EVIDENCE_REL}" >"${tmp_dir}/discarded-promoted.v10"
evidence_admitted "${tmp_dir}/discarded-promoted.v10" &&
  fail 'discarded noncompliant attempt was promoted'
sed 's/effect operator-discovery-effect-parity/effect operator-discovery-authority/' \
  "${ROOT}/${KOKA_REL}" >"${tmp_dir}/authority-widened.kk"
source_admitted "${tmp_dir}/authority-widened.kk" &&
  fail 'authority-widened Koka source passed admission'
printf 'SABOTAGE claim_promotion=REFUSED system_wide=REFUSED discarded_attempt=REFUSED source_authority=REFUSED\n'

formal_output="$("${ROOT}/${FORMAL_GATE_REL}")"
printf '%s\n' "${formal_output}" | grep -Fqx -- \
  'pireus operator discovery formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY obligations=6/6 atlas_classes=3 actions=2 action_cells=49152 separators=272:0:0:257:272:0 collision_control=EXACT incomplete_control=EXACT law_spectrum=112:824 axiom_closure=EXPLICIT_NATIVE_DECIDE_TRUST_BOUNDARY formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED n3=false n4=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'formal parity terminal marker drift'

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
[[ "$(wc -l <"${OUTPUT}" | tr -d ' ')" -eq 53 ]] || fail 'output line count drift'
require_line "${OUTPUT}" 'checks=71'
require_line "${OUTPUT}" 'positive_checks=4'
require_line "${OUTPUT}" 'adversarial_checks=67'
require_line "${OUTPUT}" 'not_semantic_authority=true'
require_line "${OUTPUT}" 'not_formal_authority=true'
require_line "${OUTPUT}" 'not_material_authority=true'
require_line "${OUTPUT}" 'not_novelty_confirmation=true'
require_line "${OUTPUT}" 'algebraic_laws_proved_by_koka=false'
require_line "${OUTPUT}" 'separator_family_proved_by_koka=false'
require_line "${OUTPUT}" 'action_census_derived_by_koka=false'
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
evidence_admitted "${ROOT}/${EVIDENCE_REL}" || fail 'evidence drifted during gate'
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'receipt drifted during gate'

printf '%s\n' \
  'pireus operator discovery effect parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Koka role=EFFECT_PARITY scope=LOCAL_HANDLER_FIXTURE_EQUALITY checks=71 positive=4 adversarial=67 canonical_attempt=CANONICAL_02_CLEAN_ROOT discarded_noncompliant_attempts=1 atlas_mutation=false group_action_mutation=false candidate_selection=false budget_extension=false material_dispatch=false formal_proofs=0 algebraic_laws_proved=false separator_family_proved=false action_census_derived=false effect=LOCAL_HANDLER_PASSED material=OPEN_NOT_EXECUTED n3=false n4=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false'
