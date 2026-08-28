#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

MODULE_REL='stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_material_ingestion.sio'
EXAMPLE_REL='examples/pireus_apple_cpu_dependency_latency_interface_material_ingestion.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_interface_material_ingestion.sio'
FIXTURE_UNSEALED_REL='tests/fixtures/pireus_apple_cpu_material_summary_unsealed_v1.txt'
FIXTURE_VERDICT_REL='tests/fixtures/pireus_apple_cpu_material_summary_child_verdict_v1.txt'
FIXTURE_CLASSIFICATION_REL='tests/fixtures/pireus_apple_cpu_material_summary_child_classification_v1.txt'
FIXTURE_MALFORMED_REL='tests/fixtures/pireus_apple_cpu_material_summary_malformed_v1.txt'
TRANSPORT_REL='tests/fixtures/pireus_apple_cpu_material_transport_failure_v1.txt'

MODULE="${ROOT}/${MODULE_REL}"
EXAMPLE="${ROOT}/${EXAMPLE_REL}"
TEST="${ROOT}/${TEST_REL}"
FIXTURE_UNSEALED="${ROOT}/${FIXTURE_UNSEALED_REL}"
FIXTURE_VERDICT="${ROOT}/${FIXTURE_VERDICT_REL}"
FIXTURE_CLASSIFICATION="${ROOT}/${FIXTURE_CLASSIFICATION_REL}"
FIXTURE_MALFORMED="${ROOT}/${FIXTURE_MALFORMED_REL}"
TRANSPORT="${ROOT}/${TRANSPORT_REL}"
MANIFEST="${ROOT}/tools/pireus/apple_cpu_dependency_latency_interface_material_ingestion.freeze.v0"
COMPILER="${ROOT}/bin/souc-lean-single-x86_64"

FIRST_EXECUTABLE_COMMIT='ed4fa16558b1815796da1bf00ccb80decb2439bf'
PARENT_AUTHORITY_COMMIT='ba85ed0689484f747e392783de4f912001153360'
MODULE_SHA256='935beba05badb174bfcaf3353c2c97a7ce6bd029f75840db37880e2cf28312aa'
EXAMPLE_SHA256='a9e6e8f6d18b44789ed4613273a55b66e425de6cf9d8ec81127af9b16b95769c'
TEST_SHA256='441da30cd562c343de6dcc1ff3175eceac521637fdb5997d7f93fa66f00a9c1a'
FIXTURE_UNSEALED_SHA256='e312524e18510575075629bc15c3bbb8fc14762672f844206d155a47d0e18263'
FIXTURE_VERDICT_SHA256='935fabd4900b2a09d4870740b90b07512ddeff03dc28b1f1ed26836c5d3305ea'
FIXTURE_CLASSIFICATION_SHA256='eca1fa09fe7a4a3a4ce4e919fa6ff858a8ceb4500c14b19bf1a093a53c5295b1'
FIXTURE_MALFORMED_SHA256='6e2eace286487cc7948175eab7a3d6f164cd01e1f8efe0bbad3e9caeac9755a7'
SOURCE_MANIFEST_SHA256='89dee58c66276e5e320218e6a727a8b99533574626f4e53e5a7071a2a5eace73'
SEMANTICS_SHA256='29079bee84a2e480096462d762729bba46ebc9e52e593f21f27cf26e9c82f435'
PARENT_SEMANTICS_SHA256='6819916ac4240923a149dd95ee9dcbeaba8d3826b7452dd819e177ff62ce8c7f'
TRANSPORT_SHA256='2a130fdaec28c0c1cb163be6687aa9e8b306a0e3537bcdc22d98cc778b040714'
TOOLCHAIN_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_RECORD_SHA256='98e3b172481889255395f717a02d9bf74276552d19c8df9a6ef350f23472c1ae'
HARDWARE_SHA256='79b36bf67aad36018f00e3e4360be992940c5fe2acb126103830cc2f6534b6db'
COMMAND_SHA256='ed20d81e9948134c2b3f759b3c341f98842ea5eb00c17aed09124037c45e00eb'
RESULT_SHA256='d17a486b4f5b67bdc907b346dcab2863420b009d99a211e411fe4ff578ccbcca'
EXAMPLE_OUTPUT_SHA256='87de751ef4662f4ba251a5bbce9a25f7975d10dd555dc14117f4894d9034ea8c'
TEST_OUTPUT_SHA256='0986571ba542e680b056a0f0bc14203ac45e021e9f8eb449d6af1ded06a184f4'
EXECUTION_FRAME_SHA256='a1654dab122ab7361ca810964d0ada1f0a65deaecc0ca9676777029847c57c5a'
FREEZE_FRAME_SHA256='05ffd094866552961027b8cd7d87a0d234aeae3d3c61f879536fe0dadcca267d'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus Apple CPU material ingestion: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() {
  sha256sum "$1" | cut -d' ' -f1
}

sha_text() {
  printf '%s\n' "$1" | sha256sum | cut -d' ' -f1
}

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
  [[ "$(sha_file "${file}")" == "${expected}" ]] ||
    fail "hash drift: ${file}"
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
require_hash "${FIXTURE_UNSEALED}" "${FIXTURE_UNSEALED_SHA256}"
require_hash "${FIXTURE_VERDICT}" "${FIXTURE_VERDICT_SHA256}"
require_hash "${FIXTURE_CLASSIFICATION}" "${FIXTURE_CLASSIFICATION_SHA256}"
require_hash "${FIXTURE_MALFORMED}" "${FIXTURE_MALFORMED_SHA256}"
require_hash "${TRANSPORT}" "${TRANSPORT_SHA256}"
require_hash "${COMPILER}" "${TOOLCHAIN_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor \
  "${FIRST_EXECUTABLE_COMMIT}" HEAD || fail 'first executable is not an ancestor'
git -C "${ROOT}" merge-base --is-ancestor \
  "${PARENT_AUTHORITY_COMMIT}" HEAD || fail 'parent authority is not an ancestor'

first_source_sha="$(
  git -C "${ROOT}" show \
    "${FIRST_EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1
)"
[[ "${first_source_sha}" == \
  'deb52286f33443de9543b5818a6e932593f4fa649229716143c0e1b330998ba8' ]] ||
  fail 'first executable source drift'
if git -C "${ROOT}" show "${FIRST_EXECUTABLE_COMMIT}:${MODULE_REL}" |
    grep -Fq 'matches_frozen_semantics'; then
  fail 'first executable already contained frozen matcher'
fi
grep -Fq 'pireus_apple_material_ingestion_transport_matches_frozen_semantics' \
  "${MODULE}" || fail 'current source lacks frozen matcher'

relative_files=(
  "${MODULE_REL}"
  "${EXAMPLE_REL}"
  "${TEST_REL}"
  "${TRANSPORT_REL}"
  "${FIXTURE_UNSEALED_REL}"
  "${FIXTURE_VERDICT_REL}"
  "${FIXTURE_CLASSIFICATION_REL}"
  "${FIXTURE_MALFORMED_REL}"
)
absolute_files=(
  "${MODULE}"
  "${EXAMPLE}"
  "${TEST}"
  "${TRANSPORT}"
  "${FIXTURE_UNSEALED}"
  "${FIXTURE_VERDICT}"
  "${FIXTURE_CLASSIFICATION}"
  "${FIXTURE_MALFORMED}"
)
actual_source_manifest="$(
  cd "${ROOT}"
  sha256sum "${relative_files[@]}" | sha256sum | cut -d' ' -f1
)"
[[ "${actual_source_manifest}" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'
actual_semantics="$(cat "${absolute_files[@]}" | sha256sum | cut -d' ' -f1)"
[[ "${actual_semantics}" == "${SEMANTICS_SHA256}" ]] ||
  fail 'semantics bundle drift'

require_line "${MANIFEST}" \
  'schema=pireus-apple-cpu-interface-material-ingestion-freeze-v0'
require_line "${MANIFEST}" 'stage=SEMANTICS_FROZEN'
require_line "${MANIFEST}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${MANIFEST}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${MANIFEST}" \
  "transport_fixture_sha256=${TRANSPORT_SHA256}"
require_line "${MANIFEST}" \
  'default_engine_result=BLOCKED_VISIBILITY_PREFLIGHT'
require_line "${MANIFEST}" \
  'freeze_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
require_line "${MANIFEST}" 'parity_open=false'
require_line "${MANIFEST}" 'claim_ready=false'

toolchain_record="engine=lean_single;wrapper=bin/souc;compiler=bin/souc-lean-single-x86_64;compiler_sha256=${TOOLCHAIN_SHA256}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_RECORD_SHA256}" ]] ||
  fail 'toolchain record drift'
hardware_record='hostname=sounio-workspace-control-0;os=Linux;architecture=x86_64;cpu=INTEL(R) XEON(R) GOLD 6526Y;cpus=64'
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware record drift'
command_0="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${EXAMPLE_REL} ${TRANSPORT_REL}"
command_1="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${TEST_REL} ${TRANSPORT_REL} ${FIXTURE_UNSEALED_REL} ${FIXTURE_VERDICT_REL} ${FIXTURE_CLASSIFICATION_REL} ${FIXTURE_MALFORMED_REL}"
command_record="$(printf '%s\n' "${command_0}" "${command_1}")"
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'command record drift'
result_record="$(printf '%s\n' \
  "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}" \
  "semantics_sha256=${SEMANTICS_SHA256}" \
  'example_exit=0' \
  "example_output_sha256=${EXAMPLE_OUTPUT_SHA256}" \
  'test_exit=0' \
  "test_output_sha256=${TEST_OUTPUT_SHA256}" \
  'transport_status=709901' \
  'unsealed_status=709902' \
  'authority_verdict=709510' \
  'parity_receipt_valid=false' \
  'material_observation_ready=false' \
  'classification_allowed=false' \
  'cost_present=false' \
  'claim_ready=false')"
[[ "$(sha_text "${result_record}")" == "${RESULT_SHA256}" ]] ||
  fail 'result record drift'

execution_frame="9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 $(sha_limbs "${SOURCE_MANIFEST_SHA256}") ${ZERO} ${ZERO} $(sha_limbs "${TOOLCHAIN_SHA256}") $(sha_limbs "${HARDWARE_SHA256}") $(sha_limbs "${COMMAND_SHA256}") ${ZERO} ${ZERO}"
[[ "$(sha_text "${execution_frame}")" == "${EXECUTION_FRAME_SHA256}" ]] ||
  fail 'execution frame drift'
execution_decision="$(printf '%s\n' "${execution_frame}" | "${GUARDIAN}")"
[[ "${execution_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' ]] ||
  fail "execution decision drift: ${execution_decision}"

freeze_frame="9020 2 3 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $(sha_limbs "${SOURCE_MANIFEST_SHA256}") $(sha_limbs "${SEMANTICS_SHA256}") $(sha_limbs "${PARENT_SEMANTICS_SHA256}") $(sha_limbs "${TOOLCHAIN_SHA256}") $(sha_limbs "${HARDWARE_SHA256}") $(sha_limbs "${COMMAND_SHA256}") $(sha_limbs "${RESULT_SHA256}") ${ZERO}"
[[ "$(sha_text "${freeze_frame}")" == "${FREEZE_FRAME_SHA256}" ]] ||
  fail 'freeze frame drift'
freeze_decision="$(printf '%s\n' "${freeze_frame}" | "${GUARDIAN}")"
[[ "${freeze_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' ]] ||
  fail "freeze decision drift: ${freeze_decision}"

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-apple-material-ingestion.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
example_output="${work}/example.txt"
test_output="${work}/test.txt"
set +e
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
    "${EXAMPLE_REL}" "${TRANSPORT_REL}"
) >"${example_output}"
example_rc=$?
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
    "${TEST_REL}" "${TRANSPORT_REL}" "${FIXTURE_UNSEALED_REL}" \
    "${FIXTURE_VERDICT_REL}" "${FIXTURE_CLASSIFICATION_REL}" \
    "${FIXTURE_MALFORMED_REL}"
) >"${test_output}"
test_rc=$?
set -e
[[ "${example_rc}" -eq 0 ]] || fail "example exited ${example_rc}"
[[ "${test_rc}" -eq 0 ]] || fail "test exited ${test_rc}"
require_hash "${example_output}" "${EXAMPLE_OUTPUT_SHA256}"
require_hash "${test_output}" "${TEST_OUTPUT_SHA256}"
require_line "${example_output}" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.apple-cpu-interface-material-ingestion.v0 stage=PARITY_OPEN'
require_line "${example_output}" 'PIREUS_APPLE_MATERIAL_INGEST valid=1'
require_line "${example_output}" ' status=709901'
require_line "${example_output}" ' transport_failure=1'
require_line "${example_output}" ' material_facts=0'
require_line "${example_output}" ' parity_receipt_valid=0'
require_line "${example_output}" ' observation_ready=0'
require_line "${example_output}" ' classification_requested=0'
require_line "${example_output}" ' classification_allowed=0'
require_line "${example_output}" ' child_verdict=0'
require_line "${example_output}" ' cost_present=0'
require_line "${example_output}" ' claim_ready=0'
require_line "${example_output}" ' authority_verdict=709510'
require_line "${test_output}" \
  'PIREUS_APPLE_MATERIAL_INGESTION_TEST transport=1'
require_line "${test_output}" ' unsealed=1'
require_line "${test_output}" ' child_verdict_refused=1'
require_line "${test_output}" ' child_classification_refused=1'
require_line "${test_output}" ' malformed_refused=1'
require_line "${test_output}" \
  'PIREUS_APPLE_MATERIAL_INGESTION_TEST_PASS transport=1 unsealed=1 child_verdict=REFUSED child_classification=REFUSED malformed=REFUSED classification_allowed=0'

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
python_frame="9020 3 4 7 7 1 0 1 0 0 0 0 0 0 0 0 0 0 $(sha_limbs "${SOURCE_MANIFEST_SHA256}") $(sha_limbs "${SEMANTICS_SHA256}") $(sha_limbs "${SEMANTICS_SHA256}") $(sha_limbs "${python_toolchain_sha}") $(sha_limbs "${HARDWARE_SHA256}") $(sha_limbs "${python_command_sha}") ${ZERO} ${ZERO}"
set +e
python_decision="$(printf '%s\n' "${python_frame}" | "${GUARDIAN}" 2>&1)"
python_rc=$?
set -e
[[ "${python_rc}" -eq 110 ]] || fail 'Python oracle was not denied with 110'
[[ "${python_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' ]] ||
  fail "Python denial drift: ${python_decision}"

printf '%s\n' \
  "PIREUS_APPLE_MATERIAL_INGESTION_GATE_PASS=true schema=pireus.apple-cpu-interface-material-ingestion.v0 producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN source_sha256=${MODULE_SHA256} source_manifest_sha256=${SOURCE_MANIFEST_SHA256} semantics_sha256=${SEMANTICS_SHA256} transport=ACCEPTED_OPERATIONAL_ONLY verdict=UNASSESSED parity_receipt_valid=false observation_ready=false classification_allowed=false python_oracle=REFUSED python_process_launched=false default_engine=BLOCKED_VISIBILITY_PREFLIGHT bootstrap_engine=PASS parity_open=false claim_ready=false"
