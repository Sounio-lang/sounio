#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAMESPACE="slurm-pilot"
BUILD_DIR="/tmp/pireus-darwin-xeon-fleet-20260827"
LOCAL_BINARY="${BUILD_DIR}/pireus_xor_lowering_material_parity"
REMOTE_BINARY="/tmp/pireus_xor_lowering_material_parity"
SOURCE="${ROOT}/tools/pireus/xor_lowering_material_parity.cpp"
MATERIAL_RUNNER="${ROOT}/scripts/ci/pireus_xor_lowering_darwin_xeon_material_parity.sh"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

SOURCE_SHA256="c5d1ab99da8d7567387772f1b98baf4a162618b82378876853a57ff0362b6cf8"
SEMANTICS_SHA256="9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970"
TOOLCHAIN_SHA256="1d1e239e199ce5e7416e3d5c66892121ee7bfd1436d1cb2f5f77a486aff85b72"
RUNNER_SHA256="d6fde54a113edc76291ab6f6b94168e943e13ce0577d4d785a4ba09a34b89d3b"
BINARY_SHA256="c88cd9ba43e106c1721ab99ea501c1c797935ed77e46f64aedab333f963e399f"
RESULT_SHA256="fe851cccb1487d3977c491426cd89e1445e3c234fbce8c5444972a441b8876e4"
DECISION_SHA256="d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e"

NODES=(
  5860-proxmox
  dl380-proxmox
  r740-proxmox
  r770-proxmox
  t560-proxmox
)

PODS=(
  slurm-pilot-worker-gpuorangefs-tszjc
  slurm-pilot-worker-dl380-wbvrt
  slurm-pilot-worker-gpuorangefs-multi-fql7q
  slurm-pilot-worker-gpuorangefs-wtvvg
  slurm-pilot-worker-cpuops-6ddgg
)

HARDWARE_SHA256=(
  c332ba1d6d856853066c832d7af9d03a78f893e2b7948f3d30eb882c6a673d96
  804744bbf0d1952b4bd53b6659eae1dbc73e768ff67f2d5ee8dba45608b40f8f
  64da2959236efb728c8357aa6248701330123083843d44cf005261d2cc432e7e
  aae90fbfd19196b1b4973f7c5a3929e1d4c01d6aef9a7f4d2d2fd36e15cac418
  88092a01d22b4a1b5eaca14564096c7302cc64284b510841b0a5c573975b17fa
)

COMMAND_SHA256=(
  ea92e61c18a23203052f142cab42b57a9491b59203d06b4cdd3a6333fc8fc854
  0ccd04cf58fb160746d45da289b9fd0a218f85b75005adb40f559b2f6e161e33
  3f1df160287ad4c31319d6440029ff87f29a18999bd26dbbee4380e9989592ce
  eae6d4ee5f2e8980edfd844b33cc2147582fa3424c09b06a2b0c283fc3ff361d
  47005409a054e664284d6efa01feaf5409007b6b2f27db7fa28bdcd6ee2700ad
)

fail() {
  printf 'pireus Darwin Xeon fleet parity: %s\n' "$*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | sed 's/[[:space:]].*$//'
}

sha256_text() {
  printf '%s' "$1" | sha256sum | sed 's/[[:space:]].*$//'
}

lscpu_field() {
  local raw="$1"
  local label="$2"
  printf '%s\n' "${raw}" | sed -n "s/^${label}:[[:space:]]*//p"
}

hardware_record() {
  local pod="$1"
  local node="$2"
  local raw model_name family model stepping cpus sockets numa flags avx512f avx512dq

  raw="$(kubectl -n "${NAMESPACE}" exec "${pod}" -- lscpu)"
  model_name="$(lscpu_field "${raw}" 'Model name')"
  family="$(lscpu_field "${raw}" 'CPU family')"
  model="$(lscpu_field "${raw}" 'Model')"
  stepping="$(lscpu_field "${raw}" 'Stepping')"
  cpus="$(lscpu_field "${raw}" 'CPU(s)')"
  sockets="$(lscpu_field "${raw}" 'Socket(s)')"
  numa="$(lscpu_field "${raw}" 'NUMA node(s)')"
  flags="$(lscpu_field "${raw}" 'Flags')"

  avx512f=false
  avx512dq=false
  [[ " ${flags} " == *' avx512f '* ]] && avx512f=true
  [[ " ${flags} " == *' avx512dq '* ]] && avx512dq=true

  printf 'schema=pireus-darwin-xeon-hardware.v1\ncluster=Darwin\nnamespace=%s\npod=%s\nnode=%s\narchitecture=x86_64\nvendor_id=GenuineIntel\ncpu_model=%s\ncpu_family=%s\nmodel=%s\nstepping=%s\nlogical_cpus=%s\nsockets=%s\nnuma_nodes=%s\navx512f=%s\navx512dq=%s\n' \
    "${NAMESPACE}" "${pod}" "${node}" "${model_name}" "${family}" "${model}" \
    "${stepping}" "${cpus}" "${sockets}" "${numa}" "${avx512f}" "${avx512dq}"
}

command_record() {
  local pod="$1"
  local node="$2"

  printf 'schema=pireus-darwin-xeon-fleet-command.v1\naction=PARITY_EXECUTE\ntransport=kubectl\nnamespace=%s\npod=%s\nnode=%s\nlocal_binary=%s\nbinary_sha256=%s\nremote_binary=%s\ncopy=kubectl -n slurm-pilot cp LOCAL_BINARY POD:/tmp/pireus_xor_lowering_material_parity\nchmod=kubectl -n slurm-pilot exec POD -- chmod 0755 /tmp/pireus_xor_lowering_material_parity\nexecute=kubectl -n slurm-pilot exec POD -- /tmp/pireus_xor_lowering_material_parity\ncleanup=kubectl -n slurm-pilot exec POD -- rm -f /tmp/pireus_xor_lowering_material_parity\n' \
    "${NAMESPACE}" "${pod}" "${node}" "${LOCAL_BINARY}" "${BINARY_SHA256}" \
    "${REMOTE_BINARY}"
}

sha_limbs() {
  local hex="$1"
  local i part
  local limbs=()

  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    limbs+=("$((16#${part}))")
  done
  printf '%s' "${limbs[*]}"
}

authority_frame() {
  local hardware_sha256="$1"
  local command_sha256="$2"
  local zero='0 0 0 0 0 0 0 0'

  printf '9020 4 4 4 4 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${hardware_sha256}")" \
    "$(sha_limbs "${command_sha256}")" \
    "${zero}" "${zero}"
}

[[ -x "${GUARDIAN}" ]] || fail "Sounio Loom Guardian is unavailable: ${GUARDIAN}"
[[ "$(sha256_file "${SOURCE}")" == "${SOURCE_SHA256}" ]] || fail 'C++ source hash drift'
[[ "$(sha256_file "${MATERIAL_RUNNER}")" == "${RUNNER_SHA256}" ]] || fail 'material runner hash drift'

mkdir -p "${BUILD_DIR}"
PIREUS_MATERIAL_BUILD_DIR="${BUILD_DIR}" "${MATERIAL_RUNNER}" > "${BUILD_DIR}/local-material-result.txt"
[[ "$(sha256_file "${LOCAL_BINARY}")" == "${BINARY_SHA256}" ]] || fail 'binary hash drift'

printf 'PIREUS_XOR_DARWIN_XEON_FLEET_PARITY_V1\n'
printf 'producer_language=C++\n'
printf 'producer_role=MATERIAL_PARITY\n'
printf 'semantic_authority_language=Sounio\n'
printf 'source_sha256=%s\n' "${SOURCE_SHA256}"
printf 'source_semantics_sha256=%s\n' "${SEMANTICS_SHA256}"
printf 'binary_sha256=%s\n' "${BINARY_SHA256}"
printf 'nodes_expected=%s\n' "${#NODES[@]}"

for ((i = 0; i < ${#NODES[@]}; i++)); do
  node="${NODES[i]}"
  pod="${PODS[i]}"
  expected_hardware_sha256="${HARDWARE_SHA256[i]}"
  expected_command_sha256="${COMMAND_SHA256[i]}"
  hardware="$(hardware_record "${pod}" "${node}")"
  hardware_sha256="$(sha256_text "${hardware}")"
  command="$(command_record "${pod}" "${node}")"
  command_sha256="$(sha256_text "${command}")"

  [[ "${hardware_sha256}" == "${expected_hardware_sha256}" ]] || \
    fail "${node}: hardware record drift: ${hardware_sha256}"
  [[ "${command_sha256}" == "${expected_command_sha256}" ]] || \
    fail "${node}: command record drift: ${command_sha256}"

  frame="$(authority_frame "${hardware_sha256}" "${command_sha256}")"
  frame_sha256="$(sha256_text "${frame}"$'\n')"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  decision_sha256="$(sha256_text "${decision}"$'\n')"
  [[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] || \
    fail "${node}: Loom refused parity execution: ${decision}"
  [[ "${decision_sha256}" == "${DECISION_SHA256}" ]] || \
    fail "${node}: Loom decision hash drift: ${decision_sha256}"

  kubectl -n "${NAMESPACE}" cp "${LOCAL_BINARY}" "${pod}:${REMOTE_BINARY}"
  kubectl -n "${NAMESPACE}" exec "${pod}" -- chmod 0755 "${REMOTE_BINARY}"
  result_file="${BUILD_DIR}/${node}.result.txt"
  kubectl -n "${NAMESPACE}" exec "${pod}" -- "${REMOTE_BINARY}" > "${result_file}"
  kubectl -n "${NAMESPACE}" exec "${pod}" -- rm -f "${REMOTE_BINARY}"

  result_sha256="$(sha256_file "${result_file}")"
  [[ "${result_sha256}" == "${RESULT_SHA256}" ]] || \
    fail "${node}: result hash drift: ${result_sha256}"
  rg -q '^result=PASS$' "${result_file}" || fail "${node}: result is not PASS"

  printf 'node[%s].name=%s\n' "${i}" "${node}"
  printf 'node[%s].pod=%s\n' "${i}" "${pod}"
  printf 'node[%s].hardware_sha256=%s\n' "${i}" "${hardware_sha256}"
  printf 'node[%s].command_sha256=%s\n' "${i}" "${command_sha256}"
  printf 'node[%s].loom_frame_sha256=%s\n' "${i}" "${frame_sha256}"
  printf 'node[%s].loom_decision_sha256=%s\n' "${i}" "${decision_sha256}"
  printf 'node[%s].result_sha256=%s\n' "${i}" "${result_sha256}"
  printf 'node[%s].result=PASS\n' "${i}"
done

printf 'nodes_observed=%s\n' "${#NODES[@]}"
printf 'matching_nodes=%s\n' "${#NODES[@]}"
printf 'apple_silicon_observed=false\n'
printf 'dgx_observed=false\n'
printf 'generic_cost_claim=false\n'
printf 'claim_ready=false\n'
printf 'result=PASS\n'
