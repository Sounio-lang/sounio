#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pireus-operator-discovery-material-v10.XXXXXX")"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/material-v10/xeon'
trap 'rm -rf "${TMP_ROOT}"' EXIT

SOUNIO_REL='stdlib/hardware/pireus/operator_discovery_engine.sio'
FREEZE_REL='tools/pireus/operator_discovery_engine.freeze.v10'
OPEN_REL='tools/pireus/operator_discovery_engine.parity-open.v10'
FORMAL_REL='tools/pireus/operator_discovery_engine.formal-parity.v10'
EFFECT_REL='tools/pireus/operator_discovery_engine.effect-parity.v10'
CPP_REL='tools/pireus/operator_discovery_material_parity.cpp'
XEON_REL='tools/pireus/evidence/operator_discovery_engine_v10.material.xeon.txt'
APPLE_REL='tools/pireus/evidence/operator_discovery_engine_v10.material.apple.txt'
DGX24_REL='tools/pireus/evidence/operator_discovery_engine_v10.material.dgx24.txt'
DGX48_REL='tools/pireus/evidence/operator_discovery_engine_v10.material.dgx48.txt'
U250_REL='tools/pireus/evidence/operator_discovery_engine_v10.material.u250.txt'
RECEIPT_REL='tools/pireus/operator_discovery_engine.material-parity.v10'
EFFECT_GATE_REL='scripts/ci/pireus_operator_discovery_effect_parity.sh'

SOUNIO_SHA256='919b6104cbce1c5f8643f5df88b9071305d3fee854f785ac63a883bc45f16117'
SEMANTICS_SHA256='2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5'
FREEZE_SHA256='9a83c9a4b920d41ee91bd7681f4e95ac11480d762185ec9ff003692d3c01d247'
OPEN_SHA256='5f109404d2a2e8e56e6cff486f871e0961f843edd2e48e2feb5f5717d1d8d39d'
FORMAL_SHA256='dddc85352de064baeee09da91917ecc3790ac5fd362ba29b4dc204d86addaa30'
EFFECT_SHA256='eb8778c8ab7bf1627ef915ef6412bbc3de1e81e0807df7459858a03ecfe4d537'
CPP_SHA256='cf9243e23c4b5dad72de07f71f12696352f14d46c3e6ffa91e34c3c7c3d624a5'
XEON_SHA256='052d79db2ee896ea410ef3ed803602ef82c93e48ec93283cc91876d92633181a'
APPLE_SHA256='2eab8bbc3ebd77646bce54de60c999e10e42e310835ced022dd907d30a263a13'
DGX24_SHA256='d06e518f754b753d7af870e86882e72bab325f4152cdf661ca6161f6b8c17ea5'
DGX48_SHA256='ba168c7c669df498432e08b4483c0f9c8bcbde51dc44ea7a3fe1d95c7434e3d2'
U250_SHA256='730546483fac35a9ddcb9b9ff20e8c631362252a60fe0ddb9ac8811ec8a05977'
RECEIPT_SHA256='56d0ed053a67ad1f1d3065411b48638571b2c77d2f64361090e1d9d6e21e78ab'
EFFECT_GATE_SHA256='9f7f8f2a832c1569de1f165adcd28399f9a17286689f1f4917d8560e23896666'
XEON_TOOLCHAIN_SHA256='3fc6e85d1e3ce2f84227517b167118ad521d1c9e297fa6afdf7abab4f0f613f9'
XEON_HARDWARE_SHA256='481ceb32ded26a254050ffeb1a9102812df485727a39eb1174c0ac534a643355'
XEON_COMMAND_SHA256='5f95862943c202057b7d12ad85e7f41ee8bceba9592c7a94ab341c3486a4b891'
XEON_PREEXEC_FRAME_SHA256='75b5834ce0cd19da50cba1b4e64192c985f4ee8d4c263ba9f60ac727eeb0dc8e'
XEON_BINARY_SHA256='e9d48d0dafb0788c80f77360fdabc8ece0d8a528530d778a6cc13bbda9c4260d'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator discovery material parity: FAIL: %s\n' "$*" >&2
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

authority_frame() {
  local language="$1" role="$2" toolchain="$3" hardware="$4" command="$5"
  printf '9020 3 4 %s %s 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' "${language}" "${role}" "$(sha_limbs "${CPP_SHA256}")" "$(sha_limbs "${SEMANTICS_SHA256}")" "$(sha_limbs "${SEMANTICS_SHA256}")" "$(sha_limbs "${toolchain}")" "$(sha_limbs "${hardware}")" "$(sha_limbs "${command}")" "${ZERO}" "${ZERO}"
}

guardian_decision() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4" rc decision
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian rc drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s\n' "${label}" "$(sha_text "${frame}")" "${decision}"
}

validate_observed_evidence() {
  local path="$1" target="$2" target_name="$3"
  require_line "${path}" 'schema=pireus-operator-discovery-material-parity-v10'
  require_line "${path}" 'producing_language=C++'
  require_line "${path}" 'producing_role=MATERIAL_PARITY'
  require_line "${path}" 'authority_language=Sounio'
  require_line "${path}" "sounio_source_sha256=${SOUNIO_SHA256}"
  require_line "${path}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
  require_line "${path}" "formal_parity_receipt_sha256=${FORMAL_SHA256}"
  require_line "${path}" "effect_parity_receipt_sha256=${EFFECT_SHA256}"
  require_line "${path}" "target=${target}"
  require_line "${path}" "target_name=${target_name}"
  require_line "${path}" 'grammar_candidates_declared=7200'
  require_line "${path}" 'grammar_candidates_evaluated_by_cpp=1'
  require_line "${path}" 'grammar_enumeration_performed_by_cpp=false'
  require_line "${path}" 'search_budget_declared=64'
  require_line "${path}" 'search_comparisons_consumed=6'
  require_line "${path}" 'seed_weight=96'
  require_line "${path}" 'parent_associator_failures=768'
  require_line "${path}" 'parent_commutator_failures=112'
  require_line "${path}" 'group_action_checks=49152'
  require_line "${path}" 'group_failures=0'
  require_line "${path}" 'candidate_id=0'
  require_line "${path}" 'mutation_tensor_index=272'
  require_line "${path}" 'mutation_delta=1'
  require_line "${path}" 'candidate_outcome=N2_RELATIVE_NOVELTY'
  require_line "${path}" 'separator_witnesses=272:0:0:257:272:0'
  require_line "${path}" 'collision_control_exact=true'
  require_line "${path}" 'incomplete_control_exact=true'
  require_line "${path}" 'commutator_failures=112'
  require_line "${path}" 'associator_failures=824'
  require_line "${path}" 'material_reconstruction_match=true'
  require_line "${path}" 'target_identity_observed=true'
  require_line "${path}" 'material_scope=HOST_CXX_FROZEN_VALUE_RECONSTRUCTION_PLUS_TARGET_IDENTITY'
  require_line "${path}" 'candidate_replayed_by_cpp=true'
  require_line "${path}" 'sounio_executable_replayed_by_cpp=false'
  require_line "${path}" 'cross_language_equivalence_proved_by_cpp=false'
  require_line "${path}" 'formal_effect_receipt_hashes_verified_by_cpp=false'
  require_line "${path}" 'fpga_operator_kernel_execution=false'
  require_line "${path}" 'semantic_write=false'
  require_line "${path}" 'expected_result_write=false'
  require_line "${path}" 'candidate_selected_by_cpp=false'
  require_line "${path}" 'n3_novelty=false'
  require_line "${path}" 'n4_novelty=false'
  require_line "${path}" 'algorithmic_novelty=false'
  require_line "${path}" 'material_novelty=false'
  require_line "${path}" 'historical_novelty=false'
  require_line "${path}" 'priority_claim=false'
  require_line "${path}" 'claim_ready=false'
  require_line "${path}" 'result=PASS'
}

receipt_admissible() {
  local path="$1"
  grep -Fqx 'schema=pireus-operator-discovery-engine.material-parity.v10' "${path}" &&
  grep -Fqx 'status=MATERIAL_FROZEN_VALUE_RECONSTRUCTION_WITH_OPEN_NATIVE_AND_ENDPOINT_DEBT' "${path}" &&
    grep -Fqx 'authority_language=Sounio' "${path}" &&
    grep -Fqx 'producing_language=C++' "${path}" &&
    grep -Fqx 'observed_platform_classes_with_frozen_value_reconstruction=4' "${path}" &&
    grep -Fqx 'observed_physical_endpoints=4' "${path}" &&
    grep -Fqx 'unresolved_physical_endpoints=2' "${path}" &&
    grep -Fqx 'canonical_endpoint_coverage=4/6' "${path}" &&
    grep -Fqx 'host_cxx_frozen_value_reconstruction_match_on_observed_endpoints=true' "${path}" &&
    grep -Fqx 'grammar_candidates_evaluated_by_cpp=1' "${path}" &&
    grep -Fqx 'grammar_enumeration_performed_by_cpp=false' "${path}" &&
    grep -Fqx 'candidate_replayed_by_cpp=true' "${path}" &&
    grep -Fqx 'sounio_executable_replayed_by_material_processes=false' "${path}" &&
    grep -Fqx 'cross_language_equivalence_proved_by_cpp=false' "${path}" &&
    grep -Fqx 'remote_material_evidence_replayed_live_by_gate=false' "${path}" &&
    grep -Fqx 'coherent_tree_rewrite_prevention=false' "${path}" &&
    grep -Fqx 'native_target_lowering_parity_complete=false' "${path}" &&
    grep -Fqx 'native_target_execution_processes=0' "${path}" &&
    grep -Fqx 'u250_fpga_operator_kernel_execution=false' "${path}" &&
    grep -Fqx 'material_target_coverage_complete=false' "${path}" &&
    grep -Fqx 'material_parity_complete=false' "${path}" &&
    grep -Fqx 'n3_novelty=false' "${path}" &&
    grep -Fqx 'n4_novelty=false' "${path}" &&
    grep -Fqx 'algorithmic_novelty=false' "${path}" &&
    grep -Fqx 'material_novelty=false' "${path}" &&
    grep -Fqx 'historical_novelty=false' "${path}" &&
    grep -Fqx 'priority_claim=false' "${path}" &&
    grep -Fqx 'claim_ready=false' "${path}" &&
    ! grep -Fqx 'observed_physical_endpoints=6' "${path}" &&
    ! grep -Fqx 'grammar_enumeration_performed_by_cpp=true' "${path}" &&
    ! grep -Fqx 'sounio_executable_replayed_by_material_processes=true' "${path}" &&
    ! grep -Fqx 'cross_language_equivalence_proved_by_cpp=true' "${path}" &&
    ! grep -Fqx 'remote_material_evidence_replayed_live_by_gate=true' "${path}" &&
    ! grep -Fqx 'coherent_tree_rewrite_prevention=true' "${path}" &&
    ! grep -Fqx 'u250_fpga_operator_kernel_execution=true' "${path}" &&
    ! grep -Fqx 'material_parity_complete=true' "${path}" &&
    ! grep -Fqx 'historical_novelty=true' "${path}" &&
    ! grep -Fqx 'priority_claim=true' "${path}" &&
    ! grep -Fqx 'claim_ready=true' "${path}"
}

require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${OPEN_REL}" "${OPEN_SHA256}"
require_hash "${ROOT}/${FORMAL_REL}" "${FORMAL_SHA256}"
require_hash "${ROOT}/${EFFECT_REL}" "${EFFECT_SHA256}"
require_hash "${ROOT}/${CPP_REL}" "${CPP_SHA256}"
require_hash "${ROOT}/${XEON_REL}" "${XEON_SHA256}"
require_hash "${ROOT}/${APPLE_REL}" "${APPLE_SHA256}"
require_hash "${ROOT}/${DGX24_REL}" "${DGX24_SHA256}"
require_hash "${ROOT}/${DGX48_REL}" "${DGX48_SHA256}"
require_hash "${ROOT}/${U250_REL}" "${U250_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${EFFECT_GATE_REL}" "${EFFECT_GATE_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Sounio Guardian unavailable'
command -v g++ >/dev/null 2>&1 || fail 'g++ unavailable for Xeon replay'

require_line "${ROOT}/${FREEZE_REL}" "module_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${FORMAL_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${EFFECT_REL}" 'effect_parity_complete=true'
require_line "${ROOT}/${EFFECT_REL}" 'material_parity_complete=false'

validate_observed_evidence "${ROOT}/${XEON_REL}" xeon XEON
validate_observed_evidence "${ROOT}/${APPLE_REL}" apple APPLE_SILICON
validate_observed_evidence "${ROOT}/${DGX24_REL}" dgx24 DGX_GB10_24
validate_observed_evidence "${ROOT}/${U250_REL}" u250 AMD_ALVEO_U250
require_line "${ROOT}/${DGX48_REL}" 'schema=pireus-operator-discovery-material-endpoint-status-v10'
require_line "${ROOT}/${DGX48_REL}" 'preflight_result=NO_ROUTE_TO_HOST'
require_line "${ROOT}/${DGX48_REL}" 'target_identity_observed=false'
require_line "${ROOT}/${DGX48_REL}" 'cpp_material_process_launched=false'
require_line "${ROOT}/${DGX48_REL}" 'claim_ready=false'
require_line "${ROOT}/${U250_REL}" 'u250_declared_card_count=2'
require_line "${ROOT}/${U250_REL}" 'u250_observed_card_count=1'
require_line "${ROOT}/${U250_REL}" 'u250_unresolved_card_count=1'

receipt_admissible "${ROOT}/${RECEIPT_REL}" ||
  fail 'material receipt is inadmissible'

for mutation in endpoint_6_of_6 grammar_enumerated remote_replayed fpga_executed parity_complete historical_novelty claim_ready; do
  cp "${ROOT}/${RECEIPT_REL}" "${TMP_ROOT}/${mutation}.receipt"
done
sed -i 's/^observed_physical_endpoints=4$/observed_physical_endpoints=6/' "${TMP_ROOT}/endpoint_6_of_6.receipt"
sed -i 's/^grammar_enumeration_performed_by_cpp=false$/grammar_enumeration_performed_by_cpp=true/' "${TMP_ROOT}/grammar_enumerated.receipt"
sed -i 's/^remote_material_evidence_replayed_live_by_gate=false$/remote_material_evidence_replayed_live_by_gate=true/' "${TMP_ROOT}/remote_replayed.receipt"
sed -i 's/^u250_fpga_operator_kernel_execution=false$/u250_fpga_operator_kernel_execution=true/' "${TMP_ROOT}/fpga_executed.receipt"
sed -i 's/^material_parity_complete=false$/material_parity_complete=true/' "${TMP_ROOT}/parity_complete.receipt"
sed -i 's/^historical_novelty=false$/historical_novelty=true/' "${TMP_ROOT}/historical_novelty.receipt"
sed -i 's/^claim_ready=false$/claim_ready=true/' "${TMP_ROOT}/claim_ready.receipt"
for mutation in endpoint_6_of_6 grammar_enumerated remote_replayed fpga_executed parity_complete historical_novelty claim_ready; do
  if receipt_admissible "${TMP_ROOT}/${mutation}.receipt"; then
    fail "forged receipt admitted: ${mutation}"
  fi
  printf 'NEGATIVE_RECEIPT mutation=%s admitted=false\n' "${mutation}"
done

effect_output="$("${ROOT}/${EFFECT_GATE_REL}")"
grep -Fq 'effect=LOCAL_HANDLER_PASSED material=OPEN_NOT_EXECUTED' <<< "${effect_output}" ||
  fail 'parent effect gate terminal marker drift'

toolchain_record="compiler=$(g++ --version | sed -n '1p') source_sha256=${CPP_SHA256} standard=c++20 optimization=-O2 warnings=-Wall,-Wextra,-Werror linker=-ldl"
hardware_record="hostname=$(hostname) kernel=$(uname -s) release=$(uname -r) architecture=$(uname -m) cpu_model=INTEL(R)_XEON(R)_GOLD_6526Y online_cpus=$(getconf _NPROCESSORS_ONLN)"
command_record='g++ c++20 O2 Wall Wextra Werror source -ldl -o /tmp/pireus-v10.xeon; run --target=xeon'
[[ "$(sha_text "${toolchain_record}")" == "${XEON_TOOLCHAIN_SHA256}" ]] ||
  fail 'live Xeon toolchain drift'
[[ "$(sha_text "${hardware_record}")" == "${XEON_HARDWARE_SHA256}" ]] ||
  fail 'live Xeon hardware drift'
[[ "$(sha_text "${command_record}")" == "${XEON_COMMAND_SHA256}" ]] ||
  fail 'Xeon command contract drift'

xeon_frame="$(authority_frame 4 4 "${XEON_TOOLCHAIN_SHA256}" "${XEON_HARDWARE_SHA256}" "${XEON_COMMAND_SHA256}")"
[[ "$(sha_text "${xeon_frame}")" == "${XEON_PREEXEC_FRAME_SHA256}" ]] ||
  fail 'Xeon preexec frame drift'
guardian_decision XEON_REPLAY "${xeon_frame}" 0 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

mkdir -p "${BUILD_ROOT}"
(
  cd "${ROOT}"
  g++ -std=c++20 -O2 -Wall -Wextra -Werror "${CPP_REL}" -ldl -o "${BUILD_ROOT}/operator-discovery-material-parity"
)
require_hash "${BUILD_ROOT}/operator-discovery-material-parity" "${XEON_BINARY_SHA256}"
"${BUILD_ROOT}/operator-discovery-material-parity" --target=xeon > "${TMP_ROOT}/xeon.txt"
require_hash "${TMP_ROOT}/xeon.txt" "${XEON_SHA256}"
cmp "${TMP_ROOT}/xeon.txt" "${ROOT}/${XEON_REL}" >/dev/null ||
  fail 'Xeon content-addressed replay drift'

python_toolchain="$(sha_text 'resolved_interpreter=python3 forbidden_oracle=true')"
python_command="$(sha_text 'python3 pireus_operator_discovery_oracle.py')"
python_frame="$(authority_frame 7 7 "${python_toolchain}" "${XEON_HARDWARE_SHA256}" "${python_command}")"
guardian_decision PYTHON_ORACLE "${python_frame}" 110 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=PYTHON_ORACLE process_launched=false\n'

printf '%s\n' 'pireus operator discovery material parity: STAGE_REACHED_NOT_A_CLAIM gate_mode=CONTENT_ADDRESSED_LOCAL_REPLAY_AND_SEALED_REMOTE_OBSERVATIONS stage=PARITY_OPEN language=C++ role=MATERIAL_PARITY frozen_value_reconstruction=MATCH_ON_OBSERVED_ENDPOINTS platform_classes=4/4 endpoints=4/6 unresolved_endpoints=2 local_live_replays=1 remote_live_replays=0 sealed_remote_observations=3 native_target_executions=0 xeon_native=false apple_native=false dgx_gpu_kernel=false u250_fpga_kernel=false grammar_candidates_evaluated=1/7200 cross_language_equivalence_proved=false formal=COMPLETE effect=LOCAL_HANDLER_PASSED material=INCOMPLETE n2=FROZEN_INTERNAL n3=false n4=false historical=false priority=false claim_ready=false python_process_launched=false rust_process_launched=false'
