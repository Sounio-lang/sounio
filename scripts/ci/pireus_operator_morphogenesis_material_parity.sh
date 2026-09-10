#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/material-v12/gate'
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pireus-pom-material-v12.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

SOUNIO_REL='stdlib/hardware/pireus/operator_morphogenesis.sio'
FREEZE_REL='tools/pireus/operator_morphogenesis.freeze.v12'
OPEN_REL='tools/pireus/operator_morphogenesis.parity-open.v12'
FORMAL_REL='tools/pireus/operator_morphogenesis.formal-parity.v12'
EFFECT_REL='tools/pireus/operator_morphogenesis.effect-parity.v12'
TRANSCRIPT_REL='tools/pireus/evidence/operator_morphogenesis_v12.first.txt'
CPP_REL='tools/pireus/operator_morphogenesis_material_parity.cpp'
LAUNCHER_REL='scripts/dev/pireus_operator_morphogenesis_material_k8s.sh'
XEON_REL='tools/pireus/evidence/operator_morphogenesis_v12.material.xeon.txt'
APPLE_REL='tools/pireus/evidence/operator_morphogenesis_v12.material.apple.txt'
DGX24_REL='tools/pireus/evidence/operator_morphogenesis_v12.material.dgx24.txt'
DGX48_REL='tools/pireus/evidence/operator_morphogenesis_v12.material.dgx48.txt'
U250_REL='tools/pireus/evidence/operator_morphogenesis_v12.material.u250.txt'
RECEIPT_REL='tools/pireus/operator_morphogenesis.material-parity.v12'

EFFECT_COMMIT='a8ccf7d768f1fd91d10304c5f0794e5fe118188e'
SOUNIO_SHA256='0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c'
SEMANTICS_SHA256='999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4'
FREEZE_SHA256='14277a28f21a044bd55bd670b5b7447789c2f4e2780251c861ee4880ef739de7'
OPEN_SHA256='b3cc6a9e67471c61eab5d42d103b21a3874ade3e6dd9a340dd8856dea4bd2909'
FORMAL_SHA256='0eb932b96838383a800f3889a331d16a10886621f29cda9c19e4e1ef74e0077c'
EFFECT_SHA256='714a662a230f986b934ccf709d782883633deef8e96184057a2782af47e70a5e'
TRANSCRIPT_SHA256='148dc490e1f6aaaf672e85fd06411755b7521930f3de5998f4c98b32af25f816'
CPP_SHA256='a7b9baf26e6b7c0ddd0ac85456455ccf08c34f9f9721e226a59175f725bf9ec6'
LAUNCHER_SHA256='1b10ec4e0973a36137ef22b60e089e9c7ab50537e942432d6c0ba090c6530b25'
XEON_SHA256='34e812e516215a44a6342d57e9fb2d909ff5d5785731d5f47476a2946a72ba95'
APPLE_SHA256='34d055f8bdb40983ecc60059d07d4caa8ab1057a8c64352512fa2e8e14178d47'
DGX24_SHA256='9383adea7a31026aa045445be8887ce93b951ea7045c8c3f8e19ad2c53d867e4'
DGX48_SHA256='12b7debcb2d90db29acb7a3fe846cf9bbf25e0a3d29ebfb0da13990bd2411b69'
U250_SHA256='b956e3ea71a908bc24d3c2bb73dd93dae7a2899a145e742230363220b784ae64'
RECEIPT_SHA256='ad6b547f5ebf471eb8ff989d4b50fbe4c5a21b15097faaf943e42e55207db3f7'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
COMPILER_SHA256='1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769'
TOOLCHAIN_SHA256='39f78f0f707e9309d10f43bab232a5315fbf226e3b5f44eaeb76e6a6e72b36f4'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='1ad3003a0d272ae3b1404591da3b67a213ec7569814b36b68c02cba3580eb259'
BINARY_SHA256='6a216e336224f394b456ac956a173d577228d096dfdc507d011e68eabe027b14'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82

REPLAY_BINARY="${BUILD_ROOT}/operator_morphogenesis_material_parity"
REPLAY_OUTPUT="${BUILD_ROOT}/xeon.txt"
REPLAY_COMMAND='mkdir -p /workspace/.home/openvscode-server/.cache/pireus/material-v12/gate && g++ -std=c++20 -O2 -Wall -Wextra -Werror tools/pireus/operator_morphogenesis_material_parity.cpp -ldl -o /workspace/.home/openvscode-server/.cache/pireus/material-v12/gate/operator_morphogenesis_material_parity && /workspace/.home/openvscode-server/.cache/pireus/material-v12/gate/operator_morphogenesis_material_parity --target=xeon --transcript=tools/pireus/evidence/operator_morphogenesis_v12.first.txt > /workspace/.home/openvscode-server/.cache/pireus/material-v12/gate/xeon.txt'

fail() {
  printf 'pireus operator morphogenesis material parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

sha_limbs() {
  local hex="$1" out='' i part
  [[ "${#hex}" -eq 64 && "${hex}" =~ ^[0-9a-f]{64}$ ]] ||
    fail "invalid sha256: ${hex}"
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
  rg -Fqx -- "$2" "$1" || fail "missing exact line in $1: $2"
}

unique_keys() {
  [[ -z "$(cut -d= -f1 "$1" | rg -v '^$' | sort | uniq -d)" ]]
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8"
  local review_promoted="$9" parent_hash="${10}" toolchain_hash="${11}"
  local command_hash="${12}" result_hash="${13}" result_limbs="${ZERO}"
  [[ "${result_hash}" == zero ]] || result_limbs="$(sha_limbs "${result_hash}")"
  printf '9020 %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" \
    "${review_promoted}" "$(sha_limbs "${CPP_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${toolchain_hash}")" "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${result_limbs}" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4" decision rc
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq "${FRAME_WORDS}" ]] ||
    fail "${label}: guardian frame width drift"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "${label}: guardian rc=${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s rc=%s %s\n' \
    "${label}" "$(sha_text "${frame}")" "${rc}" "${decision}"
  if [[ "${expected_rc}" -ne 0 ]]; then
    printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
  fi
}

validate_pass_evidence() {
  local path="$1" target="$2" target_name="$3"
  unique_keys "${path}" || fail "duplicate evidence key: ${path}"
  require_line "${path}" 'schema=pireus-operator-morphogenesis-material-parity-v12'
  require_line "${path}" 'producing_language=C++'
  require_line "${path}" 'producing_role=MATERIAL_PARITY'
  require_line "${path}" 'authority_language=Sounio'
  require_line "${path}" "sounio_source_sha256=${SOUNIO_SHA256}"
  require_line "${path}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
  require_line "${path}" "formal_parity_receipt_sha256=${FORMAL_SHA256}"
  require_line "${path}" "effect_parity_receipt_sha256=${EFFECT_SHA256}"
  require_line "${path}" "sounio_transcript_sha256=${TRANSCRIPT_SHA256}"
  require_line "${path}" 'sounio_transcript_sha256_match=true'
  require_line "${path}" "target=${target}"
  require_line "${path}" "target_name=${target_name}"
  require_line "${path}" 'slurm_route_used=false'
  require_line "${path}" 'transcript_parse_failures=0'
  require_line "${path}" 'transcript_records_complete=true'
  require_line "${path}" 'epoch_records_consumed=16'
  require_line "${path}" 'genome_records_consumed=3600'
  require_line "${path}" 'certificate_rows_consumed_from_sounio=3552'
  require_line "${path}" 'certificate_rows_reconstructed_by_cpp=false'
  require_line "${path}" 'certificate_inequality_failures=0'
  require_line "${path}" 'anf_reconstruction_checks=3600'
  require_line "${path}" 'anf_reconstruction_failures=0'
  require_line "${path}" 'cd_sign_reconstruction_checks=4096'
  require_line "${path}" 'microprogram_entries_reconstructed=4096'
  require_line "${path}" 'microprogram_field_checks=20480'
  require_line "${path}" 'microprogram_failures=0'
  require_line "${path}" 'diagnostic_checks=64'
  require_line "${path}" 'diagnostic_failures=0'
  require_line "${path}" 'probe_checks=256'
  require_line "${path}" 'probe_failures=0'
  require_line "${path}" 'frozen_counters_consistent=true'
  require_line "${path}" 'sounio_boundaries_preserved=true'
  require_line "${path}" 'material_reconstruction_match=true'
  require_line "${path}" 'target_identity_observed=true'
  require_line "${path}" 'analytic_proof_by_cpp=false'
  require_line "${path}" 'native_gpu_operator_kernel_execution=false'
  require_line "${path}" 'fpga_operator_kernel_execution=false'
  require_line "${path}" 'semantic_write=false'
  require_line "${path}" 'expected_result_write=false'
  require_line "${path}" 'candidate_selected=false'
  require_line "${path}" 'material_novelty=false'
  require_line "${path}" 'historical_novelty=false'
  require_line "${path}" 'claim_ready=false'
  require_line "${path}" 'result=PASS'
}

validate_apple_blocker() {
  local path="$1"
  unique_keys "${path}" || fail 'duplicate Apple evidence key'
  require_line "${path}" 'status=BLOCKED_AUTHENTICATION'
  require_line "${path}" 'authority_language=Sounio'
  require_line "${path}" 'target=apple'
  require_line "${path}" 'target_name=APPLE_SILICON'
  require_line "${path}" 'target_locator=tailnet:sounio-language-macbook.tail21cbc4.ts.net'
  require_line "${path}" 'producing_language=NONE'
  require_line "${path}" 'producing_role=NONE'
  require_line "${path}" 'scheduler_route=KUBERNETES_TAILSCALE_EGRESS_SSH'
  require_line "${path}" 'tailnet_peer_online=true'
  require_line "${path}" 'tailscale_egress_cleanup_complete=true'
  require_line "${path}" 'tailscale_egress_resources_remaining=0'
  require_line "${path}" 'openssh_server_reached=true'
  require_line "${path}" 'laboratory_key_authorized=false'
  require_line "${path}" 'tailscale_ssh_acl_authorized=false'
  require_line "${path}" 'target_identity_observed=false'
  require_line "${path}" 'cxx_process_launched=false'
  require_line "${path}" 'material_reconstruction_match=false'
  require_line "${path}" 'apple_material_parity_complete=false'
  require_line "${path}" 'canonical_target_substituted=false'
  require_line "${path}" 'claim_ready=false'
  require_line "${path}" 'result=BLOCKED_AUTHENTICATION'
}

receipt_admitted() {
  local path="$1"
  unique_keys "${path}" &&
    rg -Fqx 'status=MATERIAL_PARITY_PARTIAL_CANONICAL_COVERAGE' "${path}" &&
    rg -Fqx 'stage=PARITY_OPEN' "${path}" &&
    rg -Fqx 'authority_language=Sounio' "${path}" &&
    rg -Fqx 'producing_language=C++' "${path}" &&
    rg -Fqx 'producing_role=MATERIAL_PARITY' "${path}" &&
    rg -Fqx 'receipt_authority=NON_SEMANTIC' "${path}" &&
    rg -Fqx 'canonical_target_classes_materially_observed=3' "${path}" &&
    rg -Fqx 'canonical_target_class_coverage=3/4' "${path}" &&
    rg -Fqx 'canonical_target_class_coverage_complete=false' "${path}" &&
    rg -Fqx 'canonical_physical_endpoints_declared=6' "${path}" &&
    rg -Fqx 'canonical_physical_endpoints_installed=5' "${path}" &&
    rg -Fqx 'canonical_physical_endpoints_materially_observed=4' "${path}" &&
    rg -Fqx 'canonical_physical_endpoint_coverage=4/6' "${path}" &&
    rg -Fqx 'canonical_physical_endpoint_coverage_complete=false' "${path}" &&
    rg -Fqx 'currently_installed_endpoint_coverage=4/5' "${path}" &&
    rg -Fqx 'apple_result=BLOCKED_AUTHENTICATION' "${path}" &&
    rg -Fqx 'apple_authentication_authorized=false' "${path}" &&
    rg -Fqx 'apple_cxx_process_launched=false' "${path}" &&
    rg -Fqx 'u250_declared_card_count=2' "${path}" &&
    rg -Fqx 'u250_installed_card_count=1' "${path}" &&
    rg -Fqx 'u250_pending_installation_count=1' "${path}" &&
    rg -Fqx 'u250_second_card_state=PENDING_INSTALLATION' "${path}" &&
    rg -Fqx 'u250_second_card_enumeration_failure=false' "${path}" &&
    rg -Fqx 'u250_pending_installation_not_enumeration_failure=true' "${path}" &&
    rg -Fqx 'material_reconstruction_parity_complete_on_executed_targets=true' "${path}" &&
    rg -Fqx 'spark_scheduler_route=KUBERNETES' "${path}" &&
    rg -Fqx 'spark_slurm_processes_launched=0' "${path}" &&
    rg -Fqx 'u250_scheduler_route=KUBERNETES' "${path}" &&
    rg -Fqx 'u250_execution_holder_is_slurm_worker_container=true' "${path}" &&
    rg -Fqx 'u250_slurm_scheduler_invoked=false' "${path}" &&
    rg -Fqx 'u250_slurm_worker_container_does_not_imply_scheduler_dispatch=true' "${path}" &&
    rg -Fqx 'slurm_route_used=false' "${path}" &&
    rg -Fqx 'analytic_proof_by_cpp=false' "${path}" &&
    rg -Fqx 'native_gpu_operator_kernel_execution=false' "${path}" &&
    rg -Fqx 'fpga_operator_kernel_execution=false' "${path}" &&
    rg -Fqx 'material_target_coverage_complete=false' "${path}" &&
    rg -Fqx 'material_parity_complete=false' "${path}" &&
    rg -Fqx 'semantic_write=false' "${path}" &&
    rg -Fqx 'expected_result_write=false' "${path}" &&
    rg -Fqx 'candidate_selected=false' "${path}" &&
    rg -Fqx 'historical_novelty=false' "${path}" &&
    rg -Fqx 'priority_claim=false' "${path}" &&
    rg -Fqx 'claim_ready=false' "${path}" &&
    rg -Fqx 'llm_review_role=REVIEW_ONLY' "${path}" &&
    rg -Fqx 'llm_confirmed_result=false' "${path}" &&
    rg -Fqx 'result=PASS_PARTIAL_MATERIAL_PARITY_WITH_EXPLICIT_CANONICAL_GAPS' "${path}"
}

cd "${ROOT}"
require_hash "${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${OPEN_REL}" "${OPEN_SHA256}"
require_hash "${FORMAL_REL}" "${FORMAL_SHA256}"
require_hash "${EFFECT_REL}" "${EFFECT_SHA256}"
require_hash "${TRANSCRIPT_REL}" "${TRANSCRIPT_SHA256}"
require_hash "${CPP_REL}" "${CPP_SHA256}"
require_hash "${LAUNCHER_REL}" "${LAUNCHER_SHA256}"
require_hash "${XEON_REL}" "${XEON_SHA256}"
require_hash "${APPLE_REL}" "${APPLE_SHA256}"
require_hash "${DGX24_REL}" "${DGX24_SHA256}"
require_hash "${DGX48_REL}" "${DGX48_SHA256}"
require_hash "${U250_REL}" "${U250_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash 'stdlib/coordination/loom_language_authority.sio' "${GUARDIAN_POLICY_SHA256}"
require_hash '/usr/bin/g++' "${COMPILER_SHA256}"
[[ -x "${GUARDIAN}" && -x /usr/bin/g++ && -x "${LAUNCHER_REL}" ]] ||
  fail 'required native executable unavailable'

git merge-base --is-ancestor "${EFFECT_COMMIT}" HEAD ||
  fail 'effect parity commit is not an ancestor of HEAD'
[[ "$(git show "${EFFECT_COMMIT}:${EFFECT_REL}" | sha256sum | cut -d' ' -f1)" == "${EFFECT_SHA256}" ]] ||
  fail 'committed effect receipt drift'

toolchain_record="g++=$(g++ --version | sed -n '1p') binary_sha256=${COMPILER_SHA256}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'local C++ toolchain record drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'local Xeon hardware record drift'
[[ "$(sha_text "${REPLAY_COMMAND}")" == "${COMMAND_SHA256}" ]] ||
  fail 'local replay command drift'

require_line "${FREEZE_REL}" "matcher_module_sha256=${SOUNIO_SHA256}"
require_line "${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${FORMAL_REL}" 'formal_parity_complete=true'
require_line "${EFFECT_REL}" 'effect_parity_complete=true'
require_line "${EFFECT_REL}" 'material_parity_complete=false'
rg -Fq 'KUBERNETES' "${LAUNCHER_REL}" || fail 'Kubernetes route absent from launcher'
rg -Fq 'sounio.dev/u250' "${LAUNCHER_REL}" || fail 'U250 resource absent from launcher'
! rg -q '(^|[^[:alnum:]_])(sbatch|srun)([^[:alnum:]_]|$)' "${LAUNCHER_REL}" ||
  fail 'Slurm dispatch command present in Kubernetes launcher'

validate_pass_evidence "${XEON_REL}" xeon XEON
validate_pass_evidence "${DGX24_REL}" dgx24 DGX_GB10_24
validate_pass_evidence "${DGX48_REL}" dgx48 DGX_GB10_48
validate_pass_evidence "${U250_REL}" u250 AMD_ALVEO_U250_DECLARED_DUAL_CARD
validate_apple_blocker "${APPLE_REL}"

require_line "${XEON_REL}" 'scheduler_route=LOCAL'
require_line "${XEON_REL}" 'architecture=x86_64'
require_line "${DGX24_REL}" 'scheduler_route=KUBERNETES'
require_line "${DGX24_REL}" 'kubernetes_node_identity=spark-3c59'
require_line "${DGX24_REL}" 'cuda_device_name=NVIDIA GB10'
require_line "${DGX24_REL}" 'cuda_compute_capability=12.1'
require_line "${DGX24_REL}" 'material_k8s_resource=nvidia.com/gpu=1'
require_line "${DGX24_REL}" 'material_slurm_process_launched=false'
require_line "${DGX48_REL}" 'scheduler_route=KUBERNETES'
require_line "${DGX48_REL}" 'kubernetes_node_identity=spark-8e54'
require_line "${DGX48_REL}" 'cuda_device_name=NVIDIA GB10'
require_line "${DGX48_REL}" 'cuda_compute_capability=12.1'
require_line "${DGX48_REL}" 'material_k8s_resource=nvidia.com/gpu=1'
require_line "${DGX48_REL}" 'material_slurm_process_launched=false'
require_line "${U250_REL}" 'scheduler_route=KUBERNETES'
require_line "${U250_REL}" 'kubernetes_node_identity=dl380-proxmox'
require_line "${U250_REL}" 'u250_management_bdfs=0000:d8:00.0'
require_line "${U250_REL}" 'u250_user_bdfs=0000:d8:00.1'
require_line "${U250_REL}" 'u250_declared_card_count=2'
require_line "${U250_REL}" 'u250_installed_card_count=1'
require_line "${U250_REL}" 'u250_pending_installation_count=1'
require_line "${U250_REL}" 'u250_second_card_state=PENDING_INSTALLATION'
require_line "${U250_REL}" 'u250_declared_dual_card_coverage_complete=false'
require_line "${U250_REL}" 'material_k8s_dispatch_mode=EXEC_IN_ACTIVE_RESOURCE_HOLDER'
require_line "${U250_REL}" 'material_slurm_scheduler_invoked=false'
require_line "${U250_REL}" 'material_slurm_process_launched=false'

receipt_admitted "${RECEIPT_REL}" || fail 'material receipt admission failed'

for mutation in apple_auth u250_install u250_pending slurm parity claim python; do
  cp "${RECEIPT_REL}" "${TMP_ROOT}/${mutation}.receipt"
done
sed -i 's/^apple_authentication_authorized=false$/apple_authentication_authorized=true/' "${TMP_ROOT}/apple_auth.receipt"
sed -i 's/^u250_installed_card_count=1$/u250_installed_card_count=2/' "${TMP_ROOT}/u250_install.receipt"
sed -i 's/^u250_pending_installation_count=1$/u250_pending_installation_count=0/' "${TMP_ROOT}/u250_pending.receipt"
sed -i 's/^slurm_route_used=false$/slurm_route_used=true/' "${TMP_ROOT}/slurm.receipt"
sed -i 's/^material_parity_complete=false$/material_parity_complete=true/' "${TMP_ROOT}/parity.receipt"
sed -i 's/^claim_ready=false$/claim_ready=true/' "${TMP_ROOT}/claim.receipt"
sed -i 's/^producing_language=C++$/producing_language=Python/' "${TMP_ROOT}/python.receipt"
for mutation in apple_auth u250_install u250_pending slurm parity claim python; do
  receipt_admitted "${TMP_ROOT}/${mutation}.receipt" &&
    fail "receipt sabotage passed: ${mutation}"
done
printf 'SABOTAGE apple_auth=REFUSED u250_2_installed=REFUSED u250_0_pending=REFUSED slurm=REFUSED material_complete=REFUSED claim=REFUSED python_authority=REFUSED\n'

wrong_parent='199c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4'
check_guardian WRONG_PARENT \
  "$(authority_frame 4 4 4 4 1 0 0 1 0 "${wrong_parent}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=PARITY_OPEN'
check_guardian SEMANTIC_WRITE \
  "$(authority_frame 4 4 4 4 1 1 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=PARITY_OPEN'
check_guardian EXPECTED_RESULT_WRITE \
  "$(authority_frame 4 4 4 4 1 0 1 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=PARITY_OPEN'
check_guardian REVIEW_PROMOTION \
  "$(authority_frame 4 4 4 4 1 0 0 1 1 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=PARITY_OPEN'
check_guardian POLICY_MISSING \
  "$(authority_frame 4 4 4 4 0 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=PARITY_OPEN'
check_guardian POLICY_TIMEOUT \
  "$(authority_frame 4 4 4 4 2 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=PARITY_OPEN'
check_guardian POLICY_ERROR \
  "$(authority_frame 4 4 4 4 3 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=PARITY_OPEN'
check_guardian PYTHON_ORACLE \
  "$(authority_frame 4 4 7 7 1 0 0 1 0 "${SEMANTICS_SHA256}" "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}" zero)" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN'
check_guardian RUST_ORACLE \
  "$(authority_frame 4 4 8 7 1 0 0 1 0 "${SEMANTICS_SHA256}" "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}" zero)" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN'
check_guardian CLAIM_PROMOTION \
  "$(authority_frame 4 7 4 4 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${XEON_SHA256}")" 123 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=123 reason=action-forbidden-for-role next_stage=PARITY_OPEN'

check_guardian PREEXEC \
  "$(authority_frame 4 4 4 4 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
bash -c "${REPLAY_COMMAND}"
require_hash "${REPLAY_BINARY}" "${BINARY_SHA256}"
require_hash "${REPLAY_OUTPUT}" "${XEON_SHA256}"
cmp -s "${REPLAY_OUTPUT}" "${XEON_REL}" || fail 'local replay differs from Xeon evidence'

cp "${TRANSCRIPT_REL}" "${TMP_ROOT}/tampered-transcript.txt"
printf '\nTAMPERED_AFTER_FROZEN_TRANSCRIPT\n' >> "${TMP_ROOT}/tampered-transcript.txt"
tamper_command="${REPLAY_BINARY} --target=xeon --transcript=${TMP_ROOT}/tampered-transcript.txt > ${TMP_ROOT}/tampered-output.txt"
tamper_command_sha="$(sha_text "${tamper_command}")"
check_guardian TAMPERED_TRANSCRIPT_PREEXEC \
  "$(authority_frame 4 4 4 4 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${tamper_command_sha}" zero)" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
set +e
bash -c "${tamper_command}"
tamper_rc=$?
set -e
[[ "${tamper_rc}" -eq 2 ]] || fail "tampered transcript rc=${tamper_rc}"
require_line "${TMP_ROOT}/tampered-output.txt" 'sounio_transcript_sha256_match=false'
require_line "${TMP_ROOT}/tampered-output.txt" 'material_reconstruction_match=false'
require_line "${TMP_ROOT}/tampered-output.txt" 'result=FAIL'
printf 'NEGATIVE transcript_hash_tamper=REFUSED process_preexec_authorized=true semantic_result_promoted=false\n'

check_guardian SEAL \
  "$(authority_frame 4 8 4 4 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${XEON_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
check_guardian RECEIPT_WRITE \
  "$(authority_frame 4 9 4 4 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${XEON_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

require_hash "${RECEIPT_REL}" "${RECEIPT_SHA256}"
receipt_admitted "${RECEIPT_REL}" || fail 'material receipt drifted during gate'
printf '%s\n' \
  'pireus operator morphogenesis material parity: PASS_PARTIAL stage=PARITY_OPEN authority=Sounio language=C++ role=MATERIAL_PARITY classes=3/4 endpoints=4/6 installed=4/5 xeon=PASS dgx24=PASS_K8S dgx48=PASS_K8S u250_card0=PASS_K8S u250_card1=PENDING_INSTALLATION apple=BLOCKED_AUTHENTICATION slurm_route=false material_parity_complete=false claim_ready=false python_process_launched=false rust_process_launched=false'
