#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

MODULE_REL='stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_material_ingestion.sio'
EXAMPLE_REL='examples/pireus_apple_cpu_dependency_latency_interface_material_ingestion.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_interface_material_ingestion.sio'
TRANSPORT_REL='tests/fixtures/pireus_apple_cpu_material_transport_failure_v1.txt'
UNSEALED_REL='tests/fixtures/pireus_apple_cpu_material_summary_unsealed_v1.txt'
CHILD_VERDICT_REL='tests/fixtures/pireus_apple_cpu_material_summary_child_verdict_v1.txt'
CHILD_CLASSIFICATION_REL='tests/fixtures/pireus_apple_cpu_material_summary_child_classification_v1.txt'
MALFORMED_REL='tests/fixtures/pireus_apple_cpu_material_summary_malformed_v1.txt'
TAMPERED_RECEIPT_REL='tests/fixtures/pireus_apple_cpu_material_receipt_tampered_v1.txt'
MATERIAL_RECEIPT_REL='docs/research/evidence/pireus_apple_cpu_dependency_latency_interface_material_parity_20260828.txt'
MATERIAL_SUMMARY_REL='docs/research/evidence/pireus_apple_cpu_dependency_latency_interface_material_parity_summary_20260828.txt'
MATERIAL_SAMPLES_REL='docs/research/evidence/pireus_apple_cpu_dependency_latency_interface_material_parity_samples_20260828.tsv'

MODULE="${ROOT}/${MODULE_REL}"
EXAMPLE="${ROOT}/${EXAMPLE_REL}"
TEST="${ROOT}/${TEST_REL}"
TRANSPORT="${ROOT}/${TRANSPORT_REL}"
UNSEALED="${ROOT}/${UNSEALED_REL}"
CHILD_VERDICT="${ROOT}/${CHILD_VERDICT_REL}"
CHILD_CLASSIFICATION="${ROOT}/${CHILD_CLASSIFICATION_REL}"
MALFORMED="${ROOT}/${MALFORMED_REL}"
TAMPERED_RECEIPT="${ROOT}/${TAMPERED_RECEIPT_REL}"
MATERIAL_RECEIPT="${ROOT}/${MATERIAL_RECEIPT_REL}"
MATERIAL_SUMMARY="${ROOT}/${MATERIAL_SUMMARY_REL}"
MATERIAL_SAMPLES="${ROOT}/${MATERIAL_SAMPLES_REL}"
FREEZE_V0="${ROOT}/tools/pireus/apple_cpu_dependency_latency_interface_material_ingestion.freeze.v0"
FREEZE_V1="${ROOT}/tools/pireus/apple_cpu_dependency_latency_interface_material_ingestion.freeze.v1"
GATE_EVIDENCE="${ROOT}/tools/pireus/evidence/apple_cpu_dependency_latency_interface_material_ingestion_v1.txt"
COMPILER="${ROOT}/bin/souc-lean-single-x86_64"

PARENT_AUTHORITY_COMMIT='ba85ed0689484f747e392783de4f912001153360'
MATERIAL_PARITY_COMMIT='5249712530e1d7d302e5d919f3c440c692cb8185'
CLASSIFICATION_COMMIT='732bc9694b2c3b30f391e295aca792c189b1b6f7'
MODULE_SHA256='3366c4502b793fc68ec96e8a9b9813c0563ebfc7a42be6fcfacb41436d51217f'
EXAMPLE_SHA256='29b2bae7a5258171d5c84bca7ce3d993f9004b9e9ab2aeea6793aeb03a03cf3b'
TEST_SHA256='358773bdf418ab94354b516f465ec3ac155630d931a93ac2048fc421ac2790f7'
TRANSPORT_SHA256='2a130fdaec28c0c1cb163be6687aa9e8b306a0e3537bcdc22d98cc778b040714'
UNSEALED_SHA256='e312524e18510575075629bc15c3bbb8fc14762672f844206d155a47d0e18263'
CHILD_VERDICT_SHA256='935fabd4900b2a09d4870740b90b07512ddeff03dc28b1f1ed26836c5d3305ea'
CHILD_CLASSIFICATION_SHA256='eca1fa09fe7a4a3a4ce4e919fa6ff858a8ceb4500c14b19bf1a093a53c5295b1'
MALFORMED_SHA256='6e2eace286487cc7948175eab7a3d6f164cd01e1f8efe0bbad3e9caeac9755a7'
TAMPERED_RECEIPT_SHA256='a1977c8fe563204a5ffb46bf12e4c131a87100afeb96c30082492a5cf450c719'
MATERIAL_RECEIPT_SHA256='038176404c65e19c4c3424c0081f7b8c660638ad788ee300a32c67a84c705109'
MATERIAL_SUMMARY_SHA256='42701c319b81b3372098b53a8bb100e29c40a85f24be611eb9896d1675dc0913'
MATERIAL_SAMPLES_SHA256='8fe07f2b44174af64ba8014b151a86c8a37589e35b7753a5b037cb75e5cae582'
FREEZE_V0_SHA256='4195c09118723770b85a4d7ae0a6a47c85a77b9909f62de22e86b874dd2d79c3'
FREEZE_V1_SHA256='5bd9b51f78a4b8585f72ebea3d03fd4ca88f5eac60d25e9e076df029bb26d8b5'
GATE_EVIDENCE_SHA256='3843310060622c26c6df43f19927d5aa76916392f7207955e7835973f8948c90'
SOURCE_MANIFEST_SHA256='f419c84b217f57db99f8042bc0f212d68bd78aa6935090e5e039bc87a1cd2bde'
SEMANTICS_SHA256='e0d46aa4cb7eda0593f651decdcead3eccd9147e58eb24b35e05a4c1fa6eb5b5'
PARENT_SEMANTICS_SHA256='29079bee84a2e480096462d762729bba46ebc9e52e593f21f27cf26e9c82f435'
TOOLCHAIN_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_RECORD_SHA256='98e3b172481889255395f717a02d9bf74276552d19c8df9a6ef350f23472c1ae'
HARDWARE_SHA256='79b36bf67aad36018f00e3e4360be992940c5fe2acb126103830cc2f6534b6db'
COMMAND_SHA256='89a36bf69e43804b8f7257004e6d97d7484d11c0030634e8b41c555d0685a9c5'
RESULT_SHA256='710ac0f5e8e43d446281335bf72ed62f0e9d8ac1809727348bb4aa0542fd483d'
EXAMPLE_OUTPUT_SHA256='31821ae89ac8364ac83a97047aecc77b6a09410ddcbac811f2554a1d1029688b'
TEST_OUTPUT_SHA256='01b2914ce3ef09de5bf999d6ce1f261f5d615c7cc4c73166ee307287a9ca9ff5'
RECEIPT_SEAL_FRAME_SHA256='f3fc2e838ae22f7a40e41ac84587efe59163d05256769eef4d74f51b37ff0f43'
PYTHON_FRAME_SHA256='523614e01a6e3553835239d2951d37a0be89187345f2bcbe4fff2aa45e7e1f1f'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus Apple CPU material ingestion: FAIL: %s\n' "$*" >&2
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
  local file="$1" expected="$2"
  [[ -f "${file}" ]] || fail "missing file: ${file}"
  [[ "$(sha_file "${file}")" == "${expected}" ]] || fail "hash drift: ${file}"
}

require_line() {
  local file="$1" expected="$2"
  grep -Fqx -- "${expected}" "${file}" ||
    fail "missing exact line in ${file}: ${expected}"
}

[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
require_hash "${MODULE}" "${MODULE_SHA256}"
require_hash "${EXAMPLE}" "${EXAMPLE_SHA256}"
require_hash "${TEST}" "${TEST_SHA256}"
require_hash "${TRANSPORT}" "${TRANSPORT_SHA256}"
require_hash "${UNSEALED}" "${UNSEALED_SHA256}"
require_hash "${CHILD_VERDICT}" "${CHILD_VERDICT_SHA256}"
require_hash "${CHILD_CLASSIFICATION}" "${CHILD_CLASSIFICATION_SHA256}"
require_hash "${MALFORMED}" "${MALFORMED_SHA256}"
require_hash "${TAMPERED_RECEIPT}" "${TAMPERED_RECEIPT_SHA256}"
require_hash "${MATERIAL_RECEIPT}" "${MATERIAL_RECEIPT_SHA256}"
require_hash "${MATERIAL_SUMMARY}" "${MATERIAL_SUMMARY_SHA256}"
require_hash "${MATERIAL_SAMPLES}" "${MATERIAL_SAMPLES_SHA256}"
require_hash "${FREEZE_V0}" "${FREEZE_V0_SHA256}"
require_hash "${FREEZE_V1}" "${FREEZE_V1_SHA256}"
require_hash "${GATE_EVIDENCE}" "${GATE_EVIDENCE_SHA256}"
require_hash "${COMPILER}" "${TOOLCHAIN_SHA256}"

for commit in "${PARENT_AUTHORITY_COMMIT}" "${MATERIAL_PARITY_COMMIT}" \
  "${CLASSIFICATION_COMMIT}"; do
  git -C "${ROOT}" merge-base --is-ancestor "${commit}" HEAD ||
    fail "required commit is not an ancestor: ${commit}"
done

for entry in \
  "${MODULE_REL}:${MODULE_SHA256}" \
  "${EXAMPLE_REL}:${EXAMPLE_SHA256}" \
  "${TEST_REL}:${TEST_SHA256}" \
  "${TAMPERED_RECEIPT_REL}:${TAMPERED_RECEIPT_SHA256}"; do
  path="${entry%%:*}"
  expected="${entry#*:}"
  actual="$(git -C "${ROOT}" show "${CLASSIFICATION_COMMIT}:${path}" |
    sha256sum | cut -d' ' -f1)"
  [[ "${actual}" == "${expected}" ]] || fail "classification commit drift: ${path}"
done

relative_files=(
  "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}" "${TRANSPORT_REL}"
  "${UNSEALED_REL}" "${CHILD_VERDICT_REL}" "${CHILD_CLASSIFICATION_REL}"
  "${MALFORMED_REL}" "${TAMPERED_RECEIPT_REL}"
)
absolute_files=(
  "${MODULE}" "${EXAMPLE}" "${TEST}" "${TRANSPORT}" "${UNSEALED}"
  "${CHILD_VERDICT}" "${CHILD_CLASSIFICATION}" "${MALFORMED}"
  "${TAMPERED_RECEIPT}"
)
actual_source_manifest="$(
  cd "${ROOT}"
  sha256sum "${relative_files[@]}" | sha256sum | cut -d' ' -f1
)"
[[ "${actual_source_manifest}" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'
actual_semantics="$(cat "${absolute_files[@]}" | sha256sum | cut -d' ' -f1)"
[[ "${actual_semantics}" == "${SEMANTICS_SHA256}" ]] || fail 'semantics bundle drift'

require_line "${FREEZE_V0}" 'stage=SEMANTICS_FROZEN'
require_line "${FREEZE_V0}" "semantics_sha256=${PARENT_SEMANTICS_SHA256}"
require_line "${FREEZE_V1}" \
  'schema=pireus-apple-cpu-interface-material-ingestion-freeze-v1'
require_line "${FREEZE_V1}" 'stage=PARITY_OPEN'
require_line "${FREEZE_V1}" 'producing_language=Sounio'
require_line "${FREEZE_V1}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${FREEZE_V1}" \
  "classification_executable_commit=${CLASSIFICATION_COMMIT}"
require_line "${FREEZE_V1}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${FREEZE_V1}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${FREEZE_V1}" \
  "parent_ingestion_semantics_sha256=${PARENT_SEMANTICS_SHA256}"
require_line "${FREEZE_V1}" "material_receipt_sha256=${MATERIAL_RECEIPT_SHA256}"
require_line "${FREEZE_V1}" "material_summary_sha256=${MATERIAL_SUMMARY_SHA256}"
require_line "${FREEZE_V1}" "material_samples_sha256=${MATERIAL_SAMPLES_SHA256}"
require_line "${FREEZE_V1}" 'samples_identity_rule=EXACT_SHA256'
require_line "${FREEZE_V1}" 'expected_authority_verdict=709513'
require_line "${FREEZE_V1}" 'expected_candidate_count=6'
require_line "${FREEZE_V1}" 'expected_complete_terminal_count=5'
require_line "${FREEZE_V1}" 'expected_feasible_count=0'
require_line "${FREEZE_V1}" 'expected_refusal_count=2'
require_line "${FREEZE_V1}" 'expected_cycle_ineligible_count=3'
require_line "${FREEZE_V1}" 'expected_manifest_closed=false'
require_line "${FREEZE_V1}" 'expected_cost_present=false'
require_line "${FREEZE_V1}" 'expected_claim_ready=false'
require_line "${FREEZE_V1}" "gate_evidence_sha256=${GATE_EVIDENCE_SHA256}"
require_line "${FREEZE_V1}" \
  'receipt_seal_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${FREEZE_V1}" \
  'python_oracle_decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN'
require_line "${FREEZE_V1}" 'python_process_launched=false'
require_line "${FREEZE_V1}" 'parity_open=true'
require_line "${FREEZE_V1}" 'claim_ready=false'

toolchain_record="engine=lean_single;wrapper=bin/souc;compiler=bin/souc-lean-single-x86_64;compiler_sha256=${TOOLCHAIN_SHA256}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_RECORD_SHA256}" ]] ||
  fail 'toolchain record drift'
hardware_record='hostname=sounio-workspace-control-0;os=Linux;architecture=x86_64;cpu=INTEL(R) XEON(R) GOLD 6526Y;cpus=64'
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware record drift'
command_0="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${EXAMPLE_REL} ${MATERIAL_RECEIPT_REL} ${MATERIAL_SUMMARY_REL} ${MATERIAL_SAMPLES_REL}"
command_1="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${TEST_REL} ${MATERIAL_RECEIPT_REL} ${MATERIAL_SUMMARY_REL} ${MATERIAL_SAMPLES_REL} ${TRANSPORT_REL} ${UNSEALED_REL} ${CHILD_VERDICT_REL} ${CHILD_CLASSIFICATION_REL} ${MALFORMED_REL} ${TAMPERED_RECEIPT_REL}"
command_record="$(printf '%s\n' "${command_0}" "${command_1}")"
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] || fail 'command record drift'
result_record="$(printf '%s\n' \
  "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}" \
  "semantics_sha256=${SEMANTICS_SHA256}" \
  "parent_v0_semantics_sha256=${PARENT_SEMANTICS_SHA256}" \
  "material_receipt_sha256=${MATERIAL_RECEIPT_SHA256}" \
  "material_summary_sha256=${MATERIAL_SUMMARY_SHA256}" \
  "material_samples_sha256=${MATERIAL_SAMPLES_SHA256}" \
  'example_exit=0' "example_output_sha256=${EXAMPLE_OUTPUT_SHA256}" \
  'test_exit=0' "test_output_sha256=${TEST_OUTPUT_SHA256}" \
  'status=709903' 'authority_verdict=709513' 'candidate_count=6' \
  'complete_terminal_count=5' 'feasible_count=0' 'refusal_count=2' \
  'cycle_ineligible_count=3' 'manifest_closed=false' \
  'parity_receipt_valid=true' 'material_observation_ready=true' \
  'classification_allowed=true' \
  'semantic_verdict_emitted_by_child=false' 'cost_present=false' \
  'claim_ready=false' 'samples_identity_rule=EXACT_SHA256' \
  'python_oracle=REFUSED' 'python_process_launched=false')"
[[ "$(sha_text "${result_record}")" == "${RESULT_SHA256}" ]] || fail 'result record drift'

receipt_seal_frame="9020 4 8 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 $(sha_limbs "${SOURCE_MANIFEST_SHA256}") $(sha_limbs "${SEMANTICS_SHA256}") $(sha_limbs "${PARENT_SEMANTICS_SHA256}") $(sha_limbs "${TOOLCHAIN_SHA256}") $(sha_limbs "${HARDWARE_SHA256}") $(sha_limbs "${COMMAND_SHA256}") $(sha_limbs "${RESULT_SHA256}") ${ZERO}"
[[ "$(sha_text "${receipt_seal_frame}")" == "${RECEIPT_SEAL_FRAME_SHA256}" ]] ||
  fail 'receipt seal frame drift'
receipt_seal_decision="$(printf '%s\n' "${receipt_seal_frame}" | "${GUARDIAN}")"
[[ "${receipt_seal_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
  fail "receipt seal decision drift: ${receipt_seal_decision}"

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-apple-material-ingestion-v1.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
example_output="${work}/example.txt"
test_output="${work}/test.txt"
default_output="${work}/default.txt"
set +e
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" \
    "${MATERIAL_RECEIPT_REL}" "${MATERIAL_SUMMARY_REL}" "${MATERIAL_SAMPLES_REL}"
) >"${example_output}"
example_rc=$?
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}" \
    "${MATERIAL_RECEIPT_REL}" "${MATERIAL_SUMMARY_REL}" \
    "${MATERIAL_SAMPLES_REL}" "${TRANSPORT_REL}" "${UNSEALED_REL}" \
    "${CHILD_VERDICT_REL}" "${CHILD_CLASSIFICATION_REL}" \
    "${MALFORMED_REL}" "${TAMPERED_RECEIPT_REL}"
) >"${test_output}"
test_rc=$?
(
  cd "${ROOT}"
  ./bin/souc run "${EXAMPLE_REL}" "${MATERIAL_RECEIPT_REL}" \
    "${MATERIAL_SUMMARY_REL}" "${MATERIAL_SAMPLES_REL}"
) >"${default_output}" 2>&1
default_rc=$?
set -e
[[ "${example_rc}" -eq 0 ]] || fail "example exited ${example_rc}"
[[ "${test_rc}" -eq 0 ]] || fail "test exited ${test_rc}"
[[ "${default_rc}" -ne 0 ]] || fail 'default engine unexpectedly succeeded'
require_hash "${example_output}" "${EXAMPLE_OUTPUT_SHA256}"
require_hash "${test_output}" "${TEST_OUTPUT_SHA256}"
require_line "${example_output}" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.apple-cpu-interface-material-ingestion.v1 stage=PARITY_OPEN'
require_line "${example_output}" 'PIREUS_APPLE_MATERIAL_INGEST valid=1'
require_line "${example_output}" ' status=709903'
require_line "${example_output}" ' material_facts=1'
require_line "${example_output}" ' parity_receipt_valid=1'
require_line "${example_output}" ' observation_ready=1'
require_line "${example_output}" ' classification_requested=1'
require_line "${example_output}" ' classification_allowed=1'
require_line "${example_output}" ' child_verdict=0'
require_line "${example_output}" ' cost_present=0'
require_line "${example_output}" ' claim_ready=0'
require_line "${example_output}" ' authority_verdict=709513'
require_line "${example_output}" ' candidates=6'
require_line "${example_output}" ' terminal=5'
require_line "${example_output}" ' feasible=0'
require_line "${example_output}" ' refusals=2'
require_line "${example_output}" ' ineligible=3'
require_line "${example_output}" ' manifest_closed=0'
require_line "${test_output}" \
  'PIREUS_APPLE_MATERIAL_INGESTION_TEST_PASS sealed=1 verdict=INDETERMINATE candidates=6 terminal=5 feasible=0 refusals=2 ineligible=3 manifest_closed=0 transport=1 unsealed=1 child_verdict=REFUSED child_classification=REFUSED malformed=REFUSED tampered=REFUSED cost_present=0 claim_ready=0'
grep -Fq 'visibility preflight failed' "${default_output}" ||
  fail 'default engine failure classification drift'

python_toolchain_sha="$(printf '%s\n' \
  'schema=pireus-negative-oracle-toolchain.v1' \
  'resolved_executable=/usr/bin/python3' \
  'language=Python' | sha256sum | cut -d' ' -f1)"
python_command_sha="$(printf '%s\n' \
  'schema=pireus-negative-oracle-command.v1' \
  'action=PARITY_EXECUTE' \
  'resolved_executable=/usr/bin/python3' \
  'oracle=true' \
  'expected_result_write=true' \
  'execution_requested=true' \
  'process_launched=false' | sha256sum | cut -d' ' -f1)"
python_frame="9020 4 4 7 7 1 0 1 1 0 0 0 0 0 0 0 0 0 $(sha_limbs "${SOURCE_MANIFEST_SHA256}") $(sha_limbs "${SEMANTICS_SHA256}") $(sha_limbs "${SEMANTICS_SHA256}") $(sha_limbs "${python_toolchain_sha}") $(sha_limbs "${HARDWARE_SHA256}") $(sha_limbs "${python_command_sha}") ${ZERO} ${ZERO}"
[[ "$(sha_text "${python_frame}")" == "${PYTHON_FRAME_SHA256}" ]] || fail 'Python frame drift'
set +e
python_decision="$(printf '%s\n' "${python_frame}" | "${GUARDIAN}" 2>&1)"
python_rc=$?
set -e
[[ "${python_rc}" -eq 110 ]] || fail 'Python oracle was not denied with 110'
[[ "${python_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN' ]] ||
  fail "Python denial drift: ${python_decision}"

printf '%s\n' \
  "PIREUS_APPLE_MATERIAL_INGESTION_GATE_PASS=true schema=pireus.apple-cpu-interface-material-ingestion.v1 producer=Sounio role=SEMANTIC_AUTHORITY stage=PARITY_OPEN source_sha256=${MODULE_SHA256} source_manifest_sha256=${SOURCE_MANIFEST_SHA256} semantics_sha256=${SEMANTICS_SHA256} material_receipt=SEALED material_observation=CLASSIFIED verdict=INDETERMINATE candidates=6 terminal=5 feasible=0 refusals=2 ineligible=3 manifest_closed=false samples_identity=EXACT_SHA256 tampered=REFUSED python_oracle=REFUSED python_process_launched=false default_engine=BLOCKED_VISIBILITY_PREFLIGHT bootstrap_engine=PASS parity_open=true claim_ready=false"
