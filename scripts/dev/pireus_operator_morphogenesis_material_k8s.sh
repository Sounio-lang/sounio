#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
NAMESPACE="${PIREUS_MATERIAL_NAMESPACE:-default}"
IMAGE="${PIREUS_MATERIAL_K8S_IMAGE:-ubuntu:24.04}"

CPP_REL='tools/pireus/operator_morphogenesis_material_parity.cpp'
TRANSCRIPT_REL='tools/pireus/evidence/operator_morphogenesis_v12.first.txt'
SOUNIO_SHA256='0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c'
SEMANTICS_SHA256='999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4'
TRANSCRIPT_SHA256='148dc490e1f6aaaf672e85fd06411755b7521930f3de5998f4c98b32af25f816'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82

fail() {
  printf 'pireus operator morphogenesis material k8s: FAIL: %s\n' "$*" >&2
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

authority_frame() {
  local source_sha="$1" toolchain_sha="$2" hardware_sha="$3" command_sha="$4"
  printf '9020 4 4 4 4 1 0 0 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${source_sha}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${toolchain_sha}")" \
    "$(sha_limbs "${hardware_sha}")" \
    "$(sha_limbs "${command_sha}")" "${ZERO}" "${ZERO}"
}

usage() {
  printf 'usage: %s dgx24|dgx48|u250\n' "${0##*/}" >&2
  exit 64
}

[[ "$#" -eq 1 ]] || usage
target="$1"
case "${target}" in
  dgx24)
    node='spark-3c59'
    resource='nvidia.com/gpu'
    ;;
  dgx48)
    node='spark-8e54'
    resource='nvidia.com/gpu'
    ;;
  u250)
    node='dl380-proxmox'
    resource='sounio.dev/u250'
    ;;
  *) usage ;;
esac

cd "${ROOT}"
[[ -x "${GUARDIAN}" ]] || fail "guardian unavailable: ${GUARDIAN}"
command -v jq >/dev/null 2>&1 || fail 'jq is required to encode the Kubernetes PodSpec'
[[ "$(sha_file "${GUARDIAN}")" == "${GUARDIAN_SHA256}" ]] ||
  fail 'guardian hash drift'
[[ -f "${CPP_REL}" && -f "${TRANSCRIPT_REL}" ]] || fail 'material inputs missing'
[[ "$(sha_file "${TRANSCRIPT_REL}")" == "${TRANSCRIPT_SHA256}" ]] ||
  fail 'frozen Sounio transcript hash drift'

cpp_sha="$(sha_file "${CPP_REL}")"
node_record="$(kubectl get node "${node}" -o jsonpath='{.metadata.name}|{.status.nodeInfo.architecture}|{.status.nodeInfo.kernelVersion}|{.status.capacity}')"
[[ -n "${node_record}" ]] || fail "Kubernetes node unavailable: ${node}"
hardware_sha="$(sha_text "${node_record}")"
toolchain_record="route=KUBERNETES node=${node} host_toolchain=/usr/bin/g++ image=${IMAGE} resource=${resource}=1"
toolchain_sha="$(sha_text "${toolchain_record}")"

run_id="pireus-pom-v12-${target}-$(date -u +%Y%m%d%H%M%S)-$$"
pod="${run_id}"
config="${run_id}"
remote_root="/tmp/${run_id}"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/pireus-pom-k8s.XXXXXX")"

holder_line=''
if [[ "${target}" == u250 ]]; then
  holder_line="$(kubectl get pods -A -o json | jq -r \
    --arg node "${node}" --arg resource "${resource}" '
      .items[] |
      select(.spec.nodeName == $node and .status.phase == "Running") as $pod |
      $pod.spec.containers[] |
      select((((.resources.limits // {})[$resource]) // "0") != "0") |
      [$pod.metadata.namespace, $pod.metadata.name, .name] | @tsv
    ' | sed -n '1p')"
fi

if [[ -n "${holder_line}" ]]; then
  IFS=$'\t' read -r holder_namespace holder_pod holder_container <<<"${holder_line}"
  [[ -n "${holder_namespace}" && -n "${holder_pod}" && -n "${holder_container}" ]] ||
    fail 'invalid Kubernetes U250 resource-holder identity'
  holder_ref="${holder_namespace}/${holder_pod}:${holder_container}"
  local_compiler='/usr/bin/g++'
  [[ -x "${local_compiler}" ]] || fail 'local same-ISA C++ compiler unavailable'
  local_compiler_sha="$(sha_file "${local_compiler}")"
  holder_toolchain_record="route=KUBERNETES compiler_origin=LOCAL_XEON_SAME_ISA compiler=${local_compiler} compiler_sha256=${local_compiler_sha} execution_holder=${holder_ref}"
  holder_toolchain_sha="$(sha_text "${holder_toolchain_record}")"
  local_binary="${tmp_root}/material-parity"
  remote_command="set -eu; work=${remote_root}; chmod 0755 \"\${work}/material-parity\"; set +e; PIREUS_K8S_NODE_NAME=${node} \"\${work}/material-parity\" --target=${target} --transcript=\"\${work}/transcript.txt\" > \"\${work}/result.txt\"; result_rc=\$?; set -e; cat \"\${work}/result.txt\"; printf 'material_k8s_node=${node}\\nmaterial_k8s_resource=${resource}=1\\nmaterial_k8s_dispatch_mode=EXEC_IN_ACTIVE_RESOURCE_HOLDER\\nmaterial_k8s_resource_holder=${holder_ref}\\nmaterial_compiler_origin=LOCAL_XEON_SAME_ISA\\nmaterial_slurm_scheduler_invoked=false\\n'; exit \"\${result_rc}\""
  holder_command_record="${local_compiler} -std=c++20 -O2 -Wall -Wextra -Werror ${CPP_REL} -ldl -o <sealed-local-binary>|kubectl exec/cp via Kubernetes API|holder=${holder_ref}|node=${node}|resource=${resource}=1|source_sha256=${cpp_sha}|transcript_sha256=${TRANSCRIPT_SHA256}|remote_command=${remote_command}|cleanup=${remote_root}"
  holder_command_sha="$(sha_text "${holder_command_record}")"
  holder_frame="$(authority_frame "${cpp_sha}" "${holder_toolchain_sha}" "${hardware_sha}" "${holder_command_sha}")"
  [[ "$(wc -w <<<"${holder_frame}" | tr -d ' ')" -eq "${FRAME_WORDS}" ]] ||
    fail 'U250 holder guardian frame width drift'
  holder_frame_sha="$(sha_text "${holder_frame}")"
  holder_decision="$(printf '%s\n' "${holder_frame}" | "${GUARDIAN}")"
  [[ "${holder_decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
    fail "guardian denied U250 holder dispatch: ${holder_decision}"

  holder_cleanup() {
    kubectl -n "${holder_namespace}" exec "${holder_pod}" -c "${holder_container}" -- \
      rm -rf "${remote_root}" >/dev/null 2>&1 || true
    rm -rf "${tmp_root}"
  }
  trap holder_cleanup EXIT

  "${local_compiler}" -std=c++20 -O2 -Wall -Wextra -Werror "${CPP_REL}" \
    -ldl -o "${local_binary}"
  execution_binary_sha="$(sha_file "${local_binary}")"
  kubectl -n "${holder_namespace}" exec "${holder_pod}" -c "${holder_container}" -- \
    mkdir -p "${remote_root}"
  kubectl -n "${holder_namespace}" cp "${local_binary}" \
    "${holder_pod}:${remote_root}/material-parity" -c "${holder_container}"
  kubectl -n "${holder_namespace}" cp "${TRANSCRIPT_REL}" \
    "${holder_pod}:${remote_root}/transcript.txt" -c "${holder_container}"
  set +e
  kubectl -n "${holder_namespace}" exec "${holder_pod}" -c "${holder_container}" -- \
    /bin/sh -c "${remote_command}"
  holder_rc=$?
  set -e
  printf 'material_preexec_frame_sha256=%s\n' "${holder_frame_sha}"
  printf 'material_preexec_command_sha256=%s\n' "${holder_command_sha}"
  printf 'material_preexec_toolchain_record_sha256=%s\n' "${holder_toolchain_sha}"
  printf 'material_compiler_sha256=%s\n' "${local_compiler_sha}"
  printf 'material_compiler_version=%s\n' "$("${local_compiler}" --version | sed -n '1p')"
  printf 'material_execution_binary_sha256=%s\n' "${execution_binary_sha}"
  printf 'material_hardware_record_sha256=%s\n' "${hardware_sha}"
  printf 'material_guardian_decision=%s\n' "${holder_decision}"
  printf 'material_kubernetes_process_launched=true\n'
  printf 'material_slurm_process_launched=false\n'
  [[ "${holder_rc}" -eq 0 ]] || fail "U250 material process failed in ${holder_ref}"
  exit 0
fi

cleanup() {
  kubectl -n "${NAMESPACE}" delete pod "${pod}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  kubectl -n "${NAMESPACE}" delete configmap "${config}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  rm -rf "${tmp_root}"
}
trap cleanup EXIT

gzip -n -c "${TRANSCRIPT_REL}" > "${tmp_root}/transcript.txt.gz"
cp "${CPP_REL}" "${tmp_root}/source.cpp"

container_command="set -eu; work=/host${remote_root}; mkdir -p \"\${work}\"; trap 'rm -rf \"\${work}\"' EXIT; cp /payload/source.cpp \"\${work}/source.cpp\"; gzip -dc /payload/transcript.txt.gz > \"\${work}/transcript.txt\"; chroot /host /usr/bin/g++ -std=c++20 -O2 -Wall -Wextra -Werror ${remote_root}/source.cpp -ldl -o ${remote_root}/material-parity; set +e; LD_LIBRARY_PATH=/host/usr/lib/aarch64-linux-gnu:/host/lib/aarch64-linux-gnu:/host/usr/lib/x86_64-linux-gnu:/host/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64 \"\${work}/material-parity\" --target=${target} --transcript=\"\${work}/transcript.txt\" > \"\${work}/result.txt\"; result_rc=\$?; set -e; cat \"\${work}/result.txt\"; printf 'material_k8s_node=${node}\\nmaterial_k8s_resource=${resource}=1\\nmaterial_k8s_image=${IMAGE}\\n'; printf 'material_toolchain_sha256='; chroot /host /usr/bin/sha256sum /usr/bin/g++ | cut -d' ' -f1; printf 'material_toolchain_version='; chroot /host /usr/bin/g++ --version | sed -n '1p'; exit \"\${result_rc}\""
dispatch_command="kubectl -n ${NAMESPACE} run ${pod} --image=${IMAGE} --restart=Never --node-selector=none --overrides=<hash-bound-pod-spec> --command -- /bin/sh -c <hash-bound-container-command>"
command_record="${dispatch_command}|node=${node}|resource=${resource}=1|source_sha256=${cpp_sha}|transcript_sha256=${TRANSCRIPT_SHA256}|container_command=${container_command}"
command_sha="$(sha_text "${command_record}")"

frame="$(authority_frame "${cpp_sha}" "${toolchain_sha}" "${hardware_sha}" "${command_sha}")"
[[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq "${FRAME_WORDS}" ]] ||
  fail 'guardian frame width drift'
frame_sha="$(sha_text "${frame}")"
decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
[[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
  fail "guardian denied Kubernetes dispatch: ${decision}"

kubectl -n "${NAMESPACE}" create configmap "${config}" \
  --from-file=source.cpp="${tmp_root}/source.cpp" \
  --from-file=transcript.txt.gz="${tmp_root}/transcript.txt.gz" \
  --dry-run=client -o yaml | kubectl -n "${NAMESPACE}" apply -f - >/dev/null

overrides="$(jq -cn \
  --arg name "${pod}" \
  --arg node "${node}" \
  --arg image "${IMAGE}" \
  --arg resource "${resource}" \
  --arg config "${config}" \
  --arg command "${container_command}" \
  '{
    apiVersion: "v1",
    spec: {
      nodeName: $node,
      restartPolicy: "Never",
      containers: [{
        name: $name,
        image: $image,
        imagePullPolicy: "IfNotPresent",
        command: ["/bin/sh", "-c"],
        args: [$command],
        env: [{
          name: "PIREUS_K8S_NODE_NAME",
          valueFrom: {fieldRef: {fieldPath: "spec.nodeName"}}
        }],
        securityContext: {privileged: true},
        resources: {
          limits: {($resource): "1"},
          requests: {($resource): "1"}
        },
        volumeMounts: [
          {name: "host-root", mountPath: "/host"},
          {name: "payload", mountPath: "/payload", readOnly: true}
        ]
      }],
      volumes: [
        {name: "host-root", hostPath: {path: "/", type: "Directory"}},
        {name: "payload", configMap: {name: $config}}
      ]
    }
  }')"

kubectl -n "${NAMESPACE}" run "${pod}" \
  --image="${IMAGE}" --restart=Never --overrides="${overrides}" >/dev/null

deadline=$((SECONDS + 600))
phase=''
while ((SECONDS < deadline)); do
  phase="$(kubectl -n "${NAMESPACE}" get "pod/${pod}" -o jsonpath='{.status.phase}')"
  [[ "${phase}" == Succeeded || "${phase}" == Failed ]] && break
  sleep 2
done
if [[ "${phase}" != Succeeded ]]; then
  kubectl -n "${NAMESPACE}" get pod "${pod}" -o wide >&2 || true
  kubectl -n "${NAMESPACE}" describe pod "${pod}" >&2 || true
  kubectl -n "${NAMESPACE}" logs "${pod}" >&2 || true
  fail "Kubernetes material process did not succeed on ${node}"
fi

kubectl -n "${NAMESPACE}" logs "${pod}"
printf 'material_preexec_frame_sha256=%s\n' "${frame_sha}"
printf 'material_preexec_command_sha256=%s\n' "${command_sha}"
printf 'material_preexec_toolchain_record_sha256=%s\n' "${toolchain_sha}"
printf 'material_hardware_record_sha256=%s\n' "${hardware_sha}"
printf 'material_guardian_decision=%s\n' "${decision}"
printf 'material_kubernetes_process_launched=true\n'
printf 'material_slurm_process_launched=false\n'
