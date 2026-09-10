#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='docs/internal/garden/seeds/2026-08-28-pireus-u250-dual-card-admission.md'
MODULE_REL='stdlib/hardware/pireus/u250_dual_card_admission.sio'
EXAMPLE_REL='examples/pireus_u250_dual_card_admission.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_u250_dual_card_admission.sio'
FREEZE_REL='tools/pireus/u250_dual_card_admission.freeze.v0'
EVIDENCE_REL='tools/pireus/evidence/u250_dual_card_admission_v0.txt'

GARDEN="${ROOT}/${GARDEN_REL}"
MODULE="${ROOT}/${MODULE_REL}"
EXAMPLE="${ROOT}/${EXAMPLE_REL}"
TEST="${ROOT}/${TEST_REL}"
FREEZE="${ROOT}/${FREEZE_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
COMPILER="${ROOT}/bin/souc-lean-single-x86_64"

GARDEN_COMMIT='e1df9118448ddb9afb7b7cda62207a5cf653916e'
EXECUTABLE_COMMIT='8cff423c79a2140deaf287a330a452c7b36c38c9'
GARDEN_SHA256='5562809be3ef99a5a432be810c8e4b7f3f0f6bd2a27c141f542ba3c2d20fb8b8'
MODULE_SHA256='bf952aa999dad0e74871a0fc78dd6fe67479840a8f334de1c639ceaabd37eafb'
EXAMPLE_SHA256='e553a31425bc2b48337d02c7191592e9267b5eedf1cfaaa08468f2aefe9b35f9'
TEST_SHA256='de2fd3119f2eaaedd80afd71486d1d2b69575b56d0a9f48763bf7c62a6432eea'
FREEZE_SHA256='db90647e5ce23029699c2c75232ac8e84ccd9818ec597083f6ce56739843f64a'
EVIDENCE_SHA256='894cf462155139ebf437ad65b90d3c1bd648d5bb5b120e7f201304439f5dab2f'
SOURCE_MANIFEST_SHA256='4b385ceb229d60c730f08dc0418ab2fe3ebdb18d9b8f7c3ca5b74588b77dc8f7'
SEMANTICS_SHA256='9f0fe0bd01baadec0c60b370bf9dd616a6d2063f1f22b7cdf131f2bc9b6f5586'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
HARDWARE_SHA256='46262c5d0fc8df5734998677c2ad063c686fa2d1120fb8dc18dc5b382c7c4805'
COMMAND_SHA256='f6e89924f2fc0803b91410f657463ec76bf035bad76b267448c72a6813309edb'
RESULT_SHA256='5b6abc6bfab9511c2366005d97f891f11239702c847c2aedc6a55da70ee5a45e'
EXAMPLE_OUTPUT_SHA256='731dd30137347a0a3fa4a8d3dfe77a273b2db78915ed1aeb64f0cadda026893b'
TEST_OUTPUT_SHA256='f5e0346a79f73f4b5d3f6daea0f92521da3d142f445859abdddeb079211ff84d'
PREEXEC_FRAME_SHA256='69e1febd0d4098566d40409434f333b9fb74fe2411b56ebe83e0dd6ad5658e8c'
FREEZE_FRAME_SHA256='5f6b70a065d500a276909a86846a1500cfbd9237db9ff6884e4bafeae597eb4a'
PYTHON_FRAME_SHA256='18bb8167e16878884119a8a57c3192c0cd4944bdd7ec05c200fe1470b1a9f79a'
LLM_FRAME_SHA256='1ae6818f4e69e0c899d8b5235fde9eed507fbc1fc65d0a123f173f2b9303ca5b'
CPP_FRAME_SHA256='75bb838e26c57168afde29bbc8913579cca55079ce714d8fbb9b1f9e87f4aaba'
POLICY_MISSING_FRAME_SHA256='7623214aedb16d37b030b25ea8d7be38d60d581074c68336ed92da4bfa7590dd'
POLICY_TIMEOUT_FRAME_SHA256='81d822862f45e797645ec7cf15aef3bcdfca066995b873189b3da6dcc26244c2'
PYTHON_TOOLCHAIN_SHA256='5c8cfd947420cd48743adb75469089a210d7782421a4e9e46bfc4c40021fb7cf'
PYTHON_COMMAND_SHA256='414924cba8feb2885a5ccc9758c978c05db4e5ee65f36d09f3493b17b70646b8'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus U250 dual-card admission: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

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

sha_limbs() {
  local hex="$1" out='' i part
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

preexec_frame() {
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" "${ZERO}" "${ZERO}" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  local policy="${1:-1}"
  printf '9020 2 3 1 1 %s 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${policy}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${GARDEN_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

python_frame() {
  printf '9020 3 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${PYTHON_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

llm_authority_frame() {
  printf '9020 3 5 6 1 1 1 1 0 1 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

cpp_authority_frame() {
  printf '9020 3 4 4 1 1 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

authorize() {
  local frame="$1" expected_sha="$2" expected_rc="$3" expected="$4"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "Guardian frame drift: ${expected_sha}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift: expected ${expected_rc}, got ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift: ${decision}"
}

[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
require_hash "${GARDEN}" "${GARDEN_SHA256}"
require_hash "${MODULE}" "${MODULE_SHA256}"
require_hash "${EXAMPLE}" "${EXAMPLE_SHA256}"
require_hash "${TEST}" "${TEST_SHA256}"
require_hash "${FREEZE}" "${FREEZE_SHA256}"
require_hash "${EVIDENCE}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${COMPILER}" "${COMPILER_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" HEAD ||
  fail 'Sounio executable commit is not an ancestor of HEAD'

actual_manifest="$(
  cd "${ROOT}"
  sha256sum "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}" |
    sha256sum | cut -d' ' -f1
)"
[[ "${actual_manifest}" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'
actual_semantics="$(cat "${MODULE}" "${EXAMPLE}" "${TEST}" |
  sha256sum | cut -d' ' -f1)"
[[ "${actual_semantics}" == "${SEMANTICS_SHA256}" ]] ||
  fail 'semantics bundle drift'

toolchain_record="$(printf '%s\n' \
  'engine=lean_single' \
  'wrapper_path=bin/souc' \
  "wrapper_sha256=${WRAPPER_SHA256}" \
  'compiler_path=bin/souc-lean-single-x86_64' \
  "compiler_sha256=${COMPILER_SHA256}")"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'toolchain record drift'

hardware_record="$(printf '%s\n' \
  'hostname=sounio-workspace-control-0' \
  'os=Linux 7.0.2-5-pve' \
  'architecture=x86_64' \
  'cpu_model=INTEL(R) XEON(R) GOLD 6526Y' \
  'sockets=2' \
  'cores_per_socket=16' \
  'threads_per_core=2' \
  'logical_cpus=64')"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware record drift'

command_record="$(printf '%s\n' \
  'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_u250_dual_card_admission.sio' \
  'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_u250_dual_card_admission.sio')"
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'command record drift'

result_record="$(printf '%s\n' \
  'stage=SOUNIO_EXECUTABLE' 'status=711091' 'target_family=711001' \
  'declared_card_count=2' 'material_slot_count=2' \
  'discovered_card_count=0' 'admitted_card_count=0' \
  'missing_card_count=2' 'inventory_complete=false' \
  'material_parity_ready=false' 'cost_present=false' \
  'speedup_present=false' 'kernel_correctness_present=false' \
  'parity_open=false' 'claim_ready=false' 'failures=0' \
  'negative_cases=14' 'python_oracle=PREEXEC_REFUSED' \
  'python_process_launched=false')"
[[ "$(sha_text "${result_record}")" == "${RESULT_SHA256}" ]] ||
  fail 'result record drift'

require_line "${FREEZE}" 'stage=SEMANTICS_FROZEN'
require_line "${FREEZE}" 'producing_language=Sounio'
require_line "${FREEZE}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${FREEZE}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${FREEZE}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${FREEZE}" 'expected_status_name=INVENTORY_OPEN'
require_line "${FREEZE}" 'expected_declared_card_count=2'
require_line "${FREEZE}" 'expected_discovered_card_count=0'
require_line "${FREEZE}" 'expected_admitted_card_count=0'
require_line "${FREEZE}" 'live_u250_facts_promoted=false'
require_line "${FREEZE}" 'parity_open=false'
require_line "${FREEZE}" 'claim_ready=false'

authorize "$(preexec_frame)" "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-u250-admission-v0.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}"
) >"${work}/example.txt"
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}"
) >"${work}/test.txt"

require_hash "${work}/example.txt" "${EXAMPLE_OUTPUT_SHA256}"
require_hash "${work}/test.txt" "${TEST_OUTPUT_SHA256}"
require_line "${work}/example.txt" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.u250-dual-card-admission.v0 stage=SOUNIO_EXECUTABLE'
require_line "${work}/example.txt" ' status=711091'
require_line "${work}/example.txt" ' declared=2'
require_line "${work}/example.txt" ' discovered=0'
require_line "${work}/example.txt" ' admitted=0'
require_line "${work}/example.txt" ' parity_open=0'
require_line "${work}/example.txt" ' claim_ready=0'
require_line "${work}/test.txt" \
  'PIREUS_U250_DUAL_CARD_ADMISSION_TEST_PASS initial=INVENTORY_OPEN declared=2 discovered=0 admitted=0 prefreeze=REFUSED unsealed=CONTAINED duplicate=REFUSED missing_pf=CONTAINED wrong_pci=CONTAINED python=PREEXEC_REFUSED cpp_authority=REFUSED llm_authority=REFUSED llm_review=REVIEW_ONLY complete=INVENTORY_ONLY claim_ready=0'

authorize "$(freeze_frame 1)" "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize "$(python_frame)" "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
authorize "$(llm_authority_frame)" "${LLM_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
authorize "$(cpp_authority_frame)" "${CPP_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
authorize "$(freeze_frame 0)" "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize "$(freeze_frame 2)" "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'

printf 'PIREUS_U250_DUAL_CARD_ADMISSION_GATE_PASS=true stage=SEMANTICS_FROZEN target=AMD_ALVEO_U250 declared=2 discovered=0 admitted=0 negatives=14 python_oracle=E110 python_process_launched=false llm_authority=E113 cpp_authority=E113 policy_missing=E101 policy_timeout=E102 material_probe=false parity_open=false claim_ready=false\n'
