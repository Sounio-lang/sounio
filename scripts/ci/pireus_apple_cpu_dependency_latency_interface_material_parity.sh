#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
NAMESPACE="${PIREUS_T560_NAMESPACE:-beagle}"
POD="${PIREUS_T560_POD:-node-ephemeral-governance-kp96t}"
KEY="${PIREUS_T560_KEY:-/host/home/devsounio/.ssh/id_ed25519}"
PINNED_KNOWN_HOSTS="${PIREUS_T560_KNOWN_HOSTS:-/host/home/devsounio/.ssh/known_hosts}"
APPLE_USER="${PIREUS_APPLE_USER:-demetriosagourakis}"
APPLE_ADDRESS="${PIREUS_APPLE_ADDRESS:-100.91.184.41}"
APPLE_TAILNET_IDENTITY="${PIREUS_APPLE_TAILNET_IDENTITY:-sounio-language-macbook}"
APPLE_LOGIN_LOCATOR="${APPLE_USER}@${APPLE_TAILNET_IDENTITY}"
APPLE_TARGET="${APPLE_LOGIN_LOCATOR}"
APPLE_HOST_KEY_ALIAS="${PIREUS_APPLE_HOST_KEY_ALIAS:-${APPLE_ADDRESS}}"
APPLE_HOST="${APPLE_ADDRESS}"
REMOTE_DIR='/tmp/pireus-apple-cpu-interface-material-parity-20260828'
STAGE_DIR='/tmp/pireus-apple-cpu-interface-stage-20260828'
RETURN_DIR='/tmp/pireus-apple-cpu-interface-return-20260828'
OUTPUT_DIR="${PIREUS_OUTPUT_DIR:-/tmp/pireus-apple-cpu-interface-material-parity-20260828}"
FAILURE_RECORD="${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity.failure.txt"
MATERIAL_RECEIPT="${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity.receipt.txt"
ZERO='0 0 0 0 0 0 0 0'

CPP="${ROOT}/tools/pireus/apple_cpu_dependency_latency_interface_material_parity.cpp"
SOUNIO_SOURCE="${ROOT}/stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio"
SEMANTICS="${ROOT}/docs/research/pireus_apple_cpu_dependency_latency_interface_feasibility_semantics.md"

AUTHORITY_COMMIT='ba85ed0689484f747e392783de4f912001153360'
CPP_SHA256='b0ac066a1b2bb085296d05b25eac0e8c25d38c8c662d911b806af32c7d6e075f'
SOUNIO_SOURCE_SHA256='d8c7e6f9410c36f6858fb2379efa010a5adbaa32c615d89edc3e764a0606a6be'
SEMANTICS_SHA256='6819916ac4240923a149dd95ee9dcbeaba8d3826b7452dd819e177ff62ce8c7f'

PHASE='preflight'
TAILNET_STATUS='ABSENT'
TRANSPORT_FAILURE_REASON='ABSENT'
PREEXEC_DECISION='ABSENT'
PREEXEC_FRAME='ABSENT'
COMMAND_SHA256='ABSENT'
HARDWARE_SHA256='ABSENT'
TOOLCHAIN_SHA256='ABSENT'
REMOTE_WRITE_STARTED=false
CPP_EXECUTION_REQUESTED=false
CPP_EXECUTION_CONFIRMED=false

failure_record() {
  local rc=$?
  local transport_failure=false
  local failure_class='LOCAL_OR_POLICY'
  trap - ERR
  case "${PHASE}" in
    tailnet_status|transport_prepare|hardware_identity|toolchain_identity|copy_source|return_copy)
      transport_failure=true
      failure_class='TRANSPORT'
      ;;
    compile_execute)
      failure_class='MATERIAL_EXECUTION'
      ;;
    validate_result)
      failure_class='MATERIAL_RESULT_VALIDATION'
      ;;
    receipt_seal)
      failure_class='RECEIPT_SEAL'
      ;;
  esac
  mkdir -p "${OUTPUT_DIR}"
  printf '%s\n' \
    'schema=pireus-apple-cpu-interface-material-failure.v1' \
    "phase=${PHASE}" \
    "failure_class=${failure_class}" \
    "exit_code=${rc}" \
    "transport_failure=${transport_failure}" \
    "transport_failure_reason=${TRANSPORT_FAILURE_REASON}" \
    "login_locator=${APPLE_LOGIN_LOCATOR}" \
    "transport_address=${APPLE_ADDRESS}" \
    "tailnet_status=${TAILNET_STATUS//$'\n'/ }" \
    "sounio_source_sha256=${SOUNIO_SOURCE_SHA256}" \
    "sounio_semantics_sha256=${SEMANTICS_SHA256}" \
    "authority_commit=${AUTHORITY_COMMIT}" \
    "cpp_sha256=${CPP_SHA256}" \
    "toolchain_sha256=${TOOLCHAIN_SHA256}" \
    "hardware_sha256=${HARDWARE_SHA256}" \
    "command_sha256=${COMMAND_SHA256}" \
    "preexec_frame_sha256=$(if [[ "${PREEXEC_FRAME}" == ABSENT ]]; then printf ABSENT; else sha_text "${PREEXEC_FRAME}"$'\n'; fi)" \
    "preexec_decision=${PREEXEC_DECISION}" \
    "remote_write_started=${REMOTE_WRITE_STARTED}" \
    "cpp_execution_requested=${CPP_EXECUTION_REQUESTED}" \
    "cpp_execution_confirmed=${CPP_EXECUTION_CONFIRMED}" \
    'producer_language=NONE' \
    'producer_role=TRANSPORT_ONLY' \
    'parity_receipt_valid=false' \
    'classification_requested=false' \
    'semantic_verdict_emitted=false' \
    'cost_present=false' \
    'claim_ready=false' > "${FAILURE_RECORD}"
  cat "${FAILURE_RECORD}"
  exit "${rc}"
}

trap failure_record ERR

fail() {
  printf 'pireus Apple CPU interface material parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() {
  sha256sum "$1" | cut -d' ' -f1
}

sha_text() {
  printf '%s' "$1" | sha256sum | cut -d' ' -f1
}

sha_limbs() {
  local hex="$1"
  local out='' i part
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

hardware_record_expected() {
  printf '%s\n' \
    'schema=pireus-apple-cpu-interface-hardware.v1' \
    'hostname=Sounio-Language-MacBook' \
    'os=Darwin' \
    'os_release=27.0.0' \
    'architecture=arm64' \
    'model=Mac17,7' \
    'cpu=Apple M5 Max' \
    'target=J714c'
}

toolchain_record_expected() {
  printf '%s\n' \
    'schema=pireus-apple-cpu-interface-toolchain.v1' \
    'compiler=Apple clang version 21.0.0 (clang-2100.3.27.1)' \
    'compiler_target=arm64-apple-darwin27.0.0' \
    'xcode=27.0' \
    'xcode_build=27A5228h'
}

command_record() {
  printf '%s\n' \
    'schema=pireus-apple-cpu-interface-material-command.v1' \
    'action=PARITY_EXECUTE' \
    'transport=kubectl/nsenter/ssh-canonical-locator-pinned-tailnet-address-strict-host-key' \
    'host_key_source=pinned-t560-known-hosts' \
    "host_key_alias=${APPLE_HOST_KEY_ALIAS}" \
    'reachability=authorized-tailscale-status-before-ssh' \
    "login_locator=${APPLE_LOGIN_LOCATOR}" \
    "tailnet_identity=${APPLE_TAILNET_IDENTITY}" \
    "transport_address=${APPLE_ADDRESS}" \
    "remote_dir=${REMOTE_DIR}" \
    "cpp_sha256=${CPP_SHA256}" \
    "sounio_source_sha256=${SOUNIO_SOURCE_SHA256}" \
    "sounio_semantics_sha256=${SEMANTICS_SHA256}" \
    "authority_commit=${AUTHORITY_COMMIT}" \
    'identity=hostname; uname -s; uname -r; uname -m; sysctl -n hw.model; sysctl -n machdep.cpu.brand_string; sysctl -n hw.targettype' \
    'toolchain=xcrun clang++ --version | first-line; xcrun clang++ -dumpmachine; xcodebuild -version' \
    'compile=xcrun clang++ -std=c++20 -O3 -fno-fast-math -fno-associative-math -ffp-contract=off -Wall -Wextra -Werror -arch arm64 apple_cpu_dependency_latency_interface_material_parity.cpp -o apple_cpu_dependency_latency_interface_material_parity' \
    'execute=./apple_cpu_dependency_latency_interface_material_parity apple_cpu_dependency_latency_interface_material_parity.samples.tsv > apple_cpu_dependency_latency_interface_material_parity.summary.txt' \
    'return=source,binary,summary,raw-samples,host-key-fingerprint'
}

authority_frame() {
  local stage="$1" action="$2" source="$3" semantics="$4"
  local toolchain="$5" hardware="$6" command="$7" result="$8"
  local receipt_valid="$9"
  printf '9020 %s %s 4 4 1 0 0 %s 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${receipt_valid}" \
    "$(sha_limbs "${source}")" "$(sha_limbs "${semantics}")" \
    "$(sha_limbs "${semantics}")" "$(sha_limbs "${toolchain}")" \
    "$(sha_limbs "${hardware}")" "$(sha_limbs "${command}")" \
    "${result}" "${ZERO}"
}

guardian_allow() {
  local frame="$1" expected_stage="$2"
  local decision
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == "SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=${expected_stage}" ]] || \
    fail "Loom refused action: ${decision}"
  printf '%s' "${decision}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" || \
    fail "missing exact line in ${path}: ${expected}"
}

value_of() {
  local path="$1" key="$2"
  local value count
  count="$(grep -c "^${key}=" "${path}" || true)"
  [[ "${count}" == 1 ]] || fail "expected one ${key} entry in ${path}"
  value="$(sed -n "s/^${key}=//p" "${path}")"
  printf '%s' "${value}"
}

prepare_transport() {
  kubectl -n "${NAMESPACE}" exec "${POD}" -- test -r "${KEY}"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- test -r "${PINNED_KNOWN_HOSTS}"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- rm -rf "${STAGE_DIR}" "${RETURN_DIR}"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- mkdir -p "${STAGE_DIR}" "${RETURN_DIR}"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- sh -c \
    "ssh-keygen -F '${APPLE_HOST}' -f '${PINNED_KNOWN_HOSTS}' | sed '/^#/d' > '${STAGE_DIR}/known_hosts'"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- test -s "${STAGE_DIR}/known_hosts"
}

tailnet_status() {
  kubectl -n "${NAMESPACE}" exec "${POD}" -- sh -c \
    "nsenter -t 1 -n -m /usr/bin/tailscale status | grep -F ' ${APPLE_TAILNET_IDENTITY} '"
}

ssh_target() {
  local command="$1"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- \
    nsenter -t 1 -n /usr/bin/ssh -i "${KEY}" \
    -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=yes \
    -o HostName="${APPLE_ADDRESS}" \
    -o HostKeyAlias="${APPLE_HOST_KEY_ALIAS}" \
    -o UserKnownHostsFile="${STAGE_DIR}/known_hosts" \
    "${APPLE_TARGET}" "${command}"
}

copy_source_to_target() {
  kubectl -n "${NAMESPACE}" cp "${CPP}" \
    "${POD}:${STAGE_DIR}/$(basename "${CPP}")"
  ssh_target "rm -rf '${REMOTE_DIR}' && mkdir -p '${REMOTE_DIR}'"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- \
    nsenter -t 1 -n /usr/bin/scp -i "${KEY}" \
    -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=yes \
    -o HostName="${APPLE_ADDRESS}" \
    -o HostKeyAlias="${APPLE_HOST_KEY_ALIAS}" \
    -o UserKnownHostsFile="${STAGE_DIR}/known_hosts" \
    "${STAGE_DIR}/$(basename "${CPP}")" \
    "${APPLE_TARGET}:${REMOTE_DIR}/"
}

copy_result_from_target() {
  local name="$1"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- \
    nsenter -t 1 -n /usr/bin/scp -i "${KEY}" \
    -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=yes \
    -o HostName="${APPLE_ADDRESS}" \
    -o HostKeyAlias="${APPLE_HOST_KEY_ALIAS}" \
    -o UserKnownHostsFile="${STAGE_DIR}/known_hosts" \
    "${APPLE_TARGET}:${REMOTE_DIR}/${name}" "${RETURN_DIR}/${name}"
  kubectl -n "${NAMESPACE}" cp \
    "${POD}:${RETURN_DIR}/${name}" "${OUTPUT_DIR}/${name}"
}

[[ -x "${GUARDIAN}" ]] || fail "Loom Guardian unavailable: ${GUARDIAN}"
[[ "$(sha_file "${CPP}")" == "${CPP_SHA256}" ]] || fail 'C++ source hash drift'
[[ "$(sha_file "${SOUNIO_SOURCE}")" == "${SOUNIO_SOURCE_SHA256}" ]] || \
  fail 'Sounio source hash drift'
[[ "$(sha_file "${SEMANTICS}")" == "${SEMANTICS_SHA256}" ]] || \
  fail 'frozen semantics hash drift'
git -C "${ROOT}" merge-base --is-ancestor "${AUTHORITY_COMMIT}" HEAD || \
  fail 'frozen authority commit is not an ancestor of HEAD'

HARDWARE_RECORD_EXPECTED="$(hardware_record_expected)"
TOOLCHAIN_RECORD_EXPECTED="$(toolchain_record_expected)"
HARDWARE_SHA256="$(sha_text "${HARDWARE_RECORD_EXPECTED}"$'\n')"
TOOLCHAIN_SHA256="$(sha_text "${TOOLCHAIN_RECORD_EXPECTED}"$'\n')"
COMMAND_SHA256="$(command_record | sha256sum | cut -d' ' -f1)"

PREEXEC_FRAME="$(authority_frame 3 4 "${SOUNIO_SOURCE_SHA256}" \
  "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" \
  "${COMMAND_SHA256}" "${ZERO}" 0)"
PREEXEC_DECISION="$(guardian_allow "${PREEXEC_FRAME}" PARITY_OPEN)"

# No target lookup, identity query, copy, compile, or execution occurs above.
rm -f "${FAILURE_RECORD}" "${MATERIAL_RECEIPT}"
PHASE='tailnet_status'
TRANSPORT_FAILURE_REASON='peer-status-missing'
TAILNET_STATUS="$(tailnet_status)"
if [[ "${TAILNET_STATUS}" == *offline* ]]; then
  TRANSPORT_FAILURE_REASON='peer-offline'
  false
fi
TRANSPORT_FAILURE_REASON='ABSENT'
PHASE='transport_prepare'
prepare_transport
HOST_KEY_FINGERPRINT="$(kubectl -n "${NAMESPACE}" exec "${POD}" -- \
  ssh-keygen -lf "${STAGE_DIR}/known_hosts" -E sha256)"
[[ -n "${HOST_KEY_FINGERPRINT}" ]] || fail 'empty host-key fingerprint'

PHASE='hardware_identity'
HARDWARE_RECORD_ACTUAL="$(ssh_target \
  "printf 'schema=pireus-apple-cpu-interface-hardware.v1\\n'; printf 'hostname='; hostname; printf 'os='; uname -s; printf 'os_release='; uname -r; printf 'architecture='; uname -m; printf 'model='; sysctl -n hw.model; printf 'cpu='; sysctl -n machdep.cpu.brand_string; printf 'target='; sysctl -n hw.targettype")"
[[ "${HARDWARE_RECORD_ACTUAL}" == "${HARDWARE_RECORD_EXPECTED}" ]] || \
  fail "Apple hardware identity drift: ${HARDWARE_RECORD_ACTUAL}"

PHASE='toolchain_identity'
TOOLCHAIN_RECORD_ACTUAL="$(ssh_target \
  "printf 'schema=pireus-apple-cpu-interface-toolchain.v1\\n'; printf 'compiler='; xcrun clang++ --version | sed -n '1p'; printf 'compiler_target='; xcrun clang++ -dumpmachine; printf 'xcode='; xcodebuild -version | sed -n '1s/^Xcode //p'; printf 'xcode_build='; xcodebuild -version | sed -n '2s/^Build version //p'")"
[[ "${TOOLCHAIN_RECORD_ACTUAL}" == "${TOOLCHAIN_RECORD_EXPECTED}" ]] || \
  fail "Apple toolchain identity drift: ${TOOLCHAIN_RECORD_ACTUAL}"

PHASE='copy_source'
REMOTE_WRITE_STARTED=true
copy_source_to_target
BUILD_COMMAND="cd '${REMOTE_DIR}' && xcrun clang++ -std=c++20 -O3 -fno-fast-math -fno-associative-math -ffp-contract=off -Wall -Wextra -Werror -arch arm64 apple_cpu_dependency_latency_interface_material_parity.cpp -o apple_cpu_dependency_latency_interface_material_parity && ./apple_cpu_dependency_latency_interface_material_parity apple_cpu_dependency_latency_interface_material_parity.samples.tsv > apple_cpu_dependency_latency_interface_material_parity.summary.txt && shasum -a 256 apple_cpu_dependency_latency_interface_material_parity.cpp apple_cpu_dependency_latency_interface_material_parity apple_cpu_dependency_latency_interface_material_parity.summary.txt apple_cpu_dependency_latency_interface_material_parity.samples.tsv"
PHASE='compile_execute'
CPP_EXECUTION_REQUESTED=true
REMOTE_HASHES="$(ssh_target "${BUILD_COMMAND}")"
CPP_EXECUTION_CONFIRMED=true

mkdir -p "${OUTPUT_DIR}"
rm -f \
  "${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity.cpp" \
  "${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity" \
  "${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity.summary.txt" \
  "${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity.samples.tsv"
PHASE='return_copy'
copy_result_from_target apple_cpu_dependency_latency_interface_material_parity.cpp
copy_result_from_target apple_cpu_dependency_latency_interface_material_parity
copy_result_from_target apple_cpu_dependency_latency_interface_material_parity.summary.txt
copy_result_from_target apple_cpu_dependency_latency_interface_material_parity.samples.tsv

LOCAL_CPP="${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity.cpp"
LOCAL_BINARY="${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity"
LOCAL_SUMMARY="${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity.summary.txt"
LOCAL_SAMPLES="${OUTPUT_DIR}/apple_cpu_dependency_latency_interface_material_parity.samples.tsv"

PHASE='validate_result'
[[ "$(sha_file "${LOCAL_CPP}")" == "${CPP_SHA256}" ]] || \
  fail 'returned C++ source hash drift'
require_line "${LOCAL_SUMMARY}" 'PIREUS_APPLE_CPU_INTERFACE_MATERIAL_PARITY_V1'
require_line "${LOCAL_SUMMARY}" 'producer_language=C++'
require_line "${LOCAL_SUMMARY}" 'producer_role=MATERIAL_PARITY'
require_line "${LOCAL_SUMMARY}" "sounio_source_sha256=${SOUNIO_SOURCE_SHA256}"
require_line "${LOCAL_SUMMARY}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
require_line "${LOCAL_SUMMARY}" "authority_commit=${AUTHORITY_COMMIT}"
require_line "${LOCAL_SUMMARY}" 'semantic_write=false'
require_line "${LOCAL_SUMMARY}" 'expected_result_write=false'
require_line "${LOCAL_SUMMARY}" 'classification_requested=false'
require_line "${LOCAL_SUMMARY}" 'semantic_verdict_emitted=false'
require_line "${LOCAL_SUMMARY}" 'cost_present=false'
require_line "${LOCAL_SUMMARY}" 'measurand_validated=false'
require_line "${LOCAL_SUMMARY}" 'requested_warmups=128'
require_line "${LOCAL_SUMMARY}" 'requested_samples=1001'
require_line "${LOCAL_SUMMARY}" 'candidate_count=6'
require_line "${LOCAL_SUMMARY}" 'hostname=Sounio-Language-MacBook'
require_line "${LOCAL_SUMMARY}" 'os=Darwin'
require_line "${LOCAL_SUMMARY}" 'os_release=27.0.0'
require_line "${LOCAL_SUMMARY}" 'architecture=arm64'
require_line "${LOCAL_SUMMARY}" 'model=Mac17,7'
require_line "${LOCAL_SUMMARY}" 'cpu=Apple M5 Max'
require_line "${LOCAL_SUMMARY}" 'target=J714c'
require_line "${LOCAL_SUMMARY}" 'candidate_0_family=CORE_CYCLE_COUNTER'
require_line "${LOCAL_SUMMARY}" 'candidate_1_family=PROCESS_PMU_CYCLE_EVENT'
require_line "${LOCAL_SUMMARY}" 'candidate_2_family=SYSTEM_TRACE_CYCLE_EVENT'
require_line "${LOCAL_SUMMARY}" 'candidate_3_family=ARCHITECTURAL_TIMER_TICK'
require_line "${LOCAL_SUMMARY}" 'candidate_4_family=OS_MONOTONIC_TIME'
require_line "${LOCAL_SUMMARY}" 'candidate_5_family=FREQUENCY_DERIVED_ESTIMATE'
require_line "${LOCAL_SUMMARY}" 'candidate_5_native_cycle_claim=false'
require_line "${LOCAL_SUMMARY}" 'probe_completed=true'

[[ "$(wc -l < "${LOCAL_SAMPLES}" | tr -d ' ')" == 1002 ]] || \
  fail 'raw sample row count drift'
[[ "$(grep -c '^verdict=' "${LOCAL_SUMMARY}" || true)" == 0 ]] || \
  fail 'C++ emitted a semantic verdict'

BINARY_SHA256="$(sha_file "${LOCAL_BINARY}")"
SUMMARY_SHA256="$(sha_file "${LOCAL_SUMMARY}")"
SAMPLES_SHA256="$(sha_file "${LOCAL_SAMPLES}")"
HOST_KEY_SHA256="$(sha_text "${HOST_KEY_FINGERPRINT}"$'\n')"
RESULT_RECORD="$(printf '%s\n' \
  'schema=pireus-apple-cpu-interface-material-result.v1' \
  "cpp_sha256=${CPP_SHA256}" \
  "binary_sha256=${BINARY_SHA256}" \
  "summary_sha256=${SUMMARY_SHA256}" \
  "samples_sha256=${SAMPLES_SHA256}" \
  "host_key_sha256=${HOST_KEY_SHA256}" \
  "hardware_sha256=${HARDWARE_SHA256}" \
  "toolchain_sha256=${TOOLCHAIN_SHA256}" \
  "command_sha256=${COMMAND_SHA256}" \
  'classification_requested=false' \
  'cost_present=false' \
  'claim_ready=false')"
RESULT_SHA256="$(sha_text "${RESULT_RECORD}"$'\n')"

SEAL_FRAME="$(authority_frame 4 8 "${SOUNIO_SOURCE_SHA256}" \
  "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}" \
  "${COMMAND_SHA256}" "$(sha_limbs "${RESULT_SHA256}")" 1)"
PHASE='receipt_seal'
SEAL_DECISION="$(guardian_allow "${SEAL_FRAME}" PARITY_OPEN)"
PHASE='complete'

PREEXEC_FRAME_SHA256="$(sha_text "${PREEXEC_FRAME}"$'\n')"
SEAL_FRAME_SHA256="$(sha_text "${SEAL_FRAME}"$'\n')"
RUNNER_SHA256="$(sha_file "${BASH_SOURCE[0]}")"
REMOTE_HASHES_SHA256="$(sha_text "${REMOTE_HASHES}"$'\n')"
printf '%s\n' \
  'schema=pireus-apple-cpu-interface-material-receipt.v1' \
  'stage=PARITY_OPEN' \
  'producer_language=C++' \
  'producer_role=MATERIAL_PARITY' \
  "login_locator=${APPLE_LOGIN_LOCATOR}" \
  "tailnet_identity=${APPLE_TAILNET_IDENTITY}" \
  "transport_address=${APPLE_ADDRESS}" \
  'material_hostname=Sounio-Language-MacBook' \
  'material_os=Darwin' \
  'material_os_release=27.0.0' \
  'material_architecture=arm64' \
  'material_model=Mac17,7' \
  'material_cpu=Apple M5 Max' \
  'material_target=J714c' \
  'compiler=Apple clang version 21.0.0 (clang-2100.3.27.1)' \
  'compiler_target=arm64-apple-darwin27.0.0' \
  'xcode=27.0' \
  'xcode_build=27A5228h' \
  "sounio_source_sha256=${SOUNIO_SOURCE_SHA256}" \
  "sounio_semantics_sha256=${SEMANTICS_SHA256}" \
  "authority_commit=${AUTHORITY_COMMIT}" \
  "cpp_sha256=${CPP_SHA256}" \
  "binary_sha256=${BINARY_SHA256}" \
  "summary_sha256=${SUMMARY_SHA256}" \
  "samples_sha256=${SAMPLES_SHA256}" \
  "host_key_sha256=${HOST_KEY_SHA256}" \
  "hardware_sha256=${HARDWARE_SHA256}" \
  "toolchain_sha256=${TOOLCHAIN_SHA256}" \
  "runner_sha256=${RUNNER_SHA256}" \
  "command_sha256=${COMMAND_SHA256}" \
  "remote_hashes_sha256=${REMOTE_HASHES_SHA256}" \
  "preexec_frame_sha256=${PREEXEC_FRAME_SHA256}" \
  "preexec_decision=${PREEXEC_DECISION}" \
  "result_sha256=${RESULT_SHA256}" \
  "seal_frame_sha256=${SEAL_FRAME_SHA256}" \
  "seal_decision=${SEAL_DECISION}" \
  'remote_write_started=true' \
  'cpp_execution_requested=true' \
  'cpp_execution_confirmed=true' \
  'material_facts_present=true' \
  'measurand_validated=false' \
  'parity_receipt_valid=true' \
  'material_observation_ready=false' \
  'classification_requested=false' \
  'semantic_verdict_emitted=false' \
  'cost_present=false' \
  'parity_open=true' \
  'claim_ready=false' > "${MATERIAL_RECEIPT}"
MATERIAL_RECEIPT_SHA256="$(sha_file "${MATERIAL_RECEIPT}")"

printf '%s\n' "${HARDWARE_RECORD_ACTUAL}"
printf '%s\n' "${TOOLCHAIN_RECORD_ACTUAL}"
printf 'host_key_fingerprint=%s\n' "${HOST_KEY_FINGERPRINT}"
printf '%s\n' "${RESULT_RECORD}"
printf '%s\n' \
  "remote_hashes_begin" "${REMOTE_HASHES}" "remote_hashes_end" \
  "preexec_frame_sha256=${PREEXEC_FRAME_SHA256}" \
  "preexec_decision=${PREEXEC_DECISION}" \
  "seal_frame_sha256=${SEAL_FRAME_SHA256}" \
  "seal_decision=${SEAL_DECISION}" \
  "result_sha256=${RESULT_SHA256}" \
  "material_receipt_sha256=${MATERIAL_RECEIPT_SHA256}" \
  "output_dir=${OUTPUT_DIR}" \
  'PARITY_OPEN=true' \
  'CLAIM_READY=false' \
  'PIREUS_APPLE_CPU_INTERFACE_MATERIAL_PARITY_PASS=true'
