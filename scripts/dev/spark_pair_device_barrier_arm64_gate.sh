#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
SOURCE="$ROOT_DIR/tools/cluster/pireus_spark_device_barrier.cpp"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1"
NAMESPACE=beagle
LABEL='app.kubernetes.io/name=pireus-device-barrier-cgroup-canary'
NODES=(spark-3c59 spark-8e54)
PODS=(pireus-device-barrier-cgroup-canary-3c59 pireus-device-barrier-cgroup-canary-8e54)
MODE="${1:-}"
CREATED=0

fail() {
  printf 'spark-pair-device-barrier-arm64-gate: FAIL: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is missing: $1"
}

freeze_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key { print substr($0, length(key) + 2) }' "$FREEZE"
}

cleanup_best_effort() {
  [[ $CREATED -eq 1 ]] || return 0
  kubectl -n "$NAMESPACE" delete pod "${PODS[@]}" \
    --ignore-not-found --wait=true >/dev/null 2>&1 || true
  kubectl -n "$NAMESPACE" delete configmap "$CONFIGMAP" \
    --ignore-not-found --wait=true >/dev/null 2>&1 || true
}

trap cleanup_best_effort EXIT

render_configmap() {
  kubectl -n "$NAMESPACE" create configmap "$CONFIGMAP" \
    --from-file="pireus_spark_device_barrier.cpp=$SOURCE" \
    --dry-run=client -o json |
    jq --arg sha "$SOURCE_SHA" '
      .metadata.labels = {
        "app.kubernetes.io/name": "pireus-device-barrier-cgroup-canary"
      }
      | .metadata.annotations = {
          "pireus.sounio.dev/source-sha256": $sha
        }
      | .immutable = true
    '
}

render_pod() {
  local node="$1" short
  short="${node#spark-}"
  sed -e "s/__NODE__/$node/g" -e "s/__SHORT__/$short/g" \
      -e "s/__CONFIGMAP__/$CONFIGMAP/g" \
      -e "s|__IMAGE__|$FROZEN_CANARY_IMAGE|g" <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: pireus-device-barrier-cgroup-canary-__SHORT__
  namespace: beagle
  labels:
    app.kubernetes.io/name: pireus-device-barrier-cgroup-canary
spec:
  restartPolicy: Never
  automountServiceAccountToken: false
  nodeName: __NODE__
  tolerations:
    - key: sounio.dev/arch
      operator: Equal
      value: arm64
      effect: NoSchedule
  containers:
    - name: canary
      image: __IMAGE__
      imagePullPolicy: IfNotPresent
      securityContext:
        privileged: true
        readOnlyRootFilesystem: true
      command: [/bin/sh, -ceu]
      args:
        - |
          work="$(mktemp -d "/host/tmp/pireus-barrier-canary-${NODE_NAME}.XXXXXX")"
          host_work="${work#/host}"
          cleanup() {
            rm -f "$work/source.cpp" "$work/barrier" "$work/failure.log" \
              "$work"/pireus-device-canary-*
            rmdir "$work"
          }
          trap cleanup EXIT
          cp /source/pireus_spark_device_barrier.cpp "$work/source.cpp"
          chroot /host /usr/bin/c++ -std=c++20 -O2 -Wall -Wextra -Werror "$host_work/source.cpp" -o "$host_work/barrier"
          printf 'NODE=%s ARCH=%s KERNEL=%s\n' "$NODE_NAME" "$(uname -m)" "$(uname -r)"
          chroot /host /usr/bin/c++ --version | sed -n '1,2p'
          chroot /host /usr/bin/sha256sum "$host_work/source.cpp" "$host_work/barrier"
          chroot /host "$host_work/barrier" selftest
          chroot /host "$host_work/barrier" verify-devices /dev 195,226,247,498,501
          if chroot /host "$host_work/barrier" canary-self-fail \
              /sys/fs/cgroup "$host_work" 195,226,247,498,501 \
              >"$work/failure.log" 2>&1; then
            printf 'injected failure unexpectedly succeeded\n' >&2
            exit 1
          fi
          cat "$work/failure.log"
          grep -Fq 'PIREUS_DEVICE_BARRIER_CANARY_FAILURE_CLEANUP_PASS' \
            "$work/failure.log"
          chroot /host "$host_work/barrier" canary-self /sys/fs/cgroup "$host_work" 195,226,247,498,501
      env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
      volumeMounts:
        - name: host-root
          mountPath: /host
          readOnly: true
        - name: host-tmp
          mountPath: /host/tmp
        - name: source
          mountPath: /source
          readOnly: true
  volumes:
    - name: host-root
      hostPath:
        path: /
        type: Directory
    - name: host-tmp
      hostPath:
        path: /tmp
        type: Directory
    - name: source
      configMap:
        name: __CONFIGMAP__
EOF
}

validate_cluster_preflight() {
  local node node_json pod_count queue
  for node in "${NODES[@]}"; do
    node_json="$(kubectl get node "$node" -o json)"
    jq -e '
      ([.status.conditions[] | select(.type == "Ready")][0].status == "True") and
      (.status.allocatable["nvidia.com/gpu"] == "1") and
      (.metadata.labels["kubernetes.io/arch"] == "arm64")
    ' <<<"$node_json" >/dev/null || fail "$node is not one Ready ARM64 GPU node"

    pod_count="$(kubectl -n slurm-pilot get pods \
      -l app.kubernetes.io/name=slurmd -o json |
      jq --arg node "$node" '[.items[] | select(
        .spec.nodeName == $node and
        .status.phase == "Running" and
        .status.containerStatuses[0].ready == true and
        all(.spec.containers[];
          ((.resources.requests["nvidia.com/gpu"] // "0") == "0")) and
        all(.spec.initContainers[]?;
          ((.resources.requests["nvidia.com/gpu"] // "0") == "0")) and
        all(.spec.ephemeralContainers[]?;
          ((.resources.requests["nvidia.com/gpu"] // "0") == "0"))
      )] | length')"
    [[ "$pod_count" == 1 ]] || fail "$node does not have one ready GPU-free slurmd Pod"
  done

  queue="$(kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- \
    squeue -a -h -o '%N|%T')"
  if grep -Eq 'gpuorangefs-multi-spark-(3c59|8e54)' <<<"$queue"; then
    fail 'a Slurm job is using one of the Spark nodes'
  fi
  if kubectl -n "$NAMESPACE" get lease pireus-spark-pair \
      >/dev/null 2>&1; then
    fail 'the production Spark Pair Lease already exists'
  fi
  if kubectl -n "$NAMESPACE" get daemonset pireus-spark-host-fence \
      >/dev/null 2>&1; then
    fail 'the production host-fence DaemonSet already exists'
  fi
}

validate_manifests() {
  local node
  render_configmap | kubectl apply --dry-run=server -f - >/dev/null
  for node in "${NODES[@]}"; do
    render_pod "$node" | kubectl apply --dry-run=server -f - >/dev/null
  done
}

wait_for_pod() {
  local pod="$1" phase='' attempt
  for attempt in $(seq 1 120); do
    phase="$(kubectl -n "$NAMESPACE" get "pod/$pod" \
      -o jsonpath='{.status.phase}')"
    [[ "$phase" == Succeeded || "$phase" == Failed ]] && break
    sleep 1
  done
  if [[ "$phase" != Succeeded ]]; then
    kubectl -n "$NAMESPACE" get "pod/$pod" -o wide >&2 || true
    kubectl -n "$NAMESPACE" logs "$pod" >&2 || true
    kubectl -n "$NAMESPACE" get events \
      --field-selector "involvedObject.name=$pod" --sort-by=.lastTimestamp |
      tail -n 12 >&2 || true
    fail "$pod did not complete successfully"
  fi
}

validate_log() {
  local pod="$1" log="$2" observed_cgroup
  grep -Fq "$SOURCE_SHA  /tmp/pireus-barrier-canary-" <<<"$log" ||
    fail "$pod did not compile the frozen source bytes"
  grep -Fq 'PIREUS_DEVICE_BARRIER_SELFTEST_PASS majors=195,226,247,498,501 default=ALLOW matched=DENY duplicates=REFUSE root_target=REFUSE' \
    <<<"$log" || fail "$pod failed the instruction selftest"
  grep -Fq 'PIREUS_DEVICE_BARRIER_INVENTORY_PASS root=/dev majors=195,226,247,498,501' \
    <<<"$log" || fail "$pod failed the device inventory gate"
  grep -Eq 'PIREUS_DEVICE_BARRIER_CANARY_PASS .* baseline_programs=[0-9]+ access=MKNOD_DENIED detach=BASELINE_RESTORED' \
    <<<"$log" || fail "$pod failed the kernel attach/deny/detach canary"
  grep -Fq " tag=$FROZEN_BPF_TAG majors=195,226,247,498,501 " <<<"$log" ||
    fail "$pod did not prove the frozen BPF tag"
  grep -Eq 'PIREUS_DEVICE_BARRIER_CANARY_FAILURE_CLEANUP_PASS cgroup=/sys/fs/cgroup/[^ ]+ lifetime=FD_SCOPED baseline=RESTORED' \
    <<<"$log" || fail "$pod failed the injected-failure cleanup proof"
  observed_cgroup="$(awk '/^PIREUS_DEVICE_BARRIER_CANARY_PASS / {
    for (i = 1; i <= NF; ++i) if ($i ~ /^cgroup=/) { sub(/^cgroup=/, "", $i); print $i }
  }' <<<"$log")"
  [[ "$observed_cgroup" == /sys/fs/cgroup/* &&
     "$observed_cgroup" != *'..'* && "$observed_cgroup" != *'//'* ]] ||
    fail "$pod did not prove a strict child-cgroup target: $observed_cgroup"
}

[[ "$MODE" == --check || "$MODE" == --apply ]] ||
  fail 'usage: spark_pair_device_barrier_arm64_gate.sh --check|--apply'
for command in awk cut grep jq kubectl sed seq sha256sum sleep tail; do
  require_command "$command"
done
[[ -f "$SOURCE" && -f "$FREEZE" ]] || fail 'source or freeze is missing'

SOURCE_SHA="$(sha256sum "$SOURCE" | cut -d ' ' -f 1)"
FROZEN_SHA="$(freeze_value device_barrier_source_sha256)"
[[ "$SOURCE_SHA" == "$FROZEN_SHA" ]] ||
  fail "device barrier source is not frozen: source=$SOURCE_SHA freeze=$FROZEN_SHA"
GATE_SHA="$(sha256sum "${BASH_SOURCE[0]}" | cut -d ' ' -f 1)"
FROZEN_GATE_SHA="$(freeze_value device_barrier_arm64_gate_sha256)"
[[ "$GATE_SHA" == "$FROZEN_GATE_SHA" ]] ||
  fail "ARM64 gate source is not frozen: source=$GATE_SHA freeze=$FROZEN_GATE_SHA"
FROZEN_BINARY_SHA="$(freeze_value device_barrier_arm64_binary_sha256)"
FROZEN_BPF_TAG="$(freeze_value device_barrier_arm64_bpf_tag)"
FROZEN_CANARY_IMAGE="$(freeze_value device_barrier_arm64_canary_image)"
[[ "$FROZEN_BINARY_SHA" =~ ^[0-9a-f]{64}$ &&
   "$FROZEN_BPF_TAG" =~ ^[0-9a-f]{16}$ &&
   "$FROZEN_CANARY_IMAGE" =~ ^docker\.io/library/ubuntu@sha256:[0-9a-f]{64}$ ]] ||
  fail 'frozen ARM64 evidence identities are malformed'
CONFIGMAP="pireus-device-barrier-cgroup-canary-${SOURCE_SHA:0:12}"

validate_cluster_preflight
validate_manifests
if [[ "$MODE" == --check ]]; then
  printf 'SPARK_PAIR_DEVICE_BARRIER_ARM64_CHECK_PASS nodes=2 source_sha=%s mutation=none\n' \
    "$SOURCE_SHA"
  exit 0
fi

existing="$(kubectl -n "$NAMESPACE" get pod,configmap -l "$LABEL" -o name)"
[[ -z "$existing" ]] || fail "a prior canary object exists: $existing"
CREATED=1
render_configmap | kubectl apply -f - >/dev/null
for node in "${NODES[@]}"; do
  render_pod "$node" | kubectl apply -f - >/dev/null
done
for pod in "${PODS[@]}"; do
  wait_for_pod "$pod"
done

binary_sha=''
container_image_id=''
for pod in "${PODS[@]}"; do
  log="$(kubectl -n "$NAMESPACE" logs "$pod")"
  printf '%s\n' "--- $pod ---" "$log"
  validate_log "$pod" "$log"
  observed="$(awk '/\/barrier$/ { print $1 }' <<<"$log")"
  [[ "$observed" =~ ^[0-9a-f]{64}$ ]] || fail "$pod omitted the binary hash"
  [[ "$observed" == "$FROZEN_BINARY_SHA" ]] ||
    fail "$pod produced a binary outside the frozen ARM64 identity"
  observed_image_id="$(kubectl -n "$NAMESPACE" get "pod/$pod" \
    -o jsonpath='{.status.containerStatuses[0].imageID}')"
  [[ "$observed_image_id" =~ ^docker\.io/library/ubuntu@sha256:[0-9a-f]{64}$ ]] ||
    fail "$pod reported an unexpected container image identity: $observed_image_id"
  [[ "$observed_image_id" == "$FROZEN_CANARY_IMAGE" ]] ||
    fail "$pod ran an image outside the frozen ARM64 identity"
  if [[ -z "$binary_sha" ]]; then
    binary_sha="$observed"
    container_image_id="$observed_image_id"
  else
    [[ "$observed" == "$binary_sha" ]] ||
      fail 'the two ARM64 nodes produced different helper binaries'
    [[ "$observed_image_id" == "$container_image_id" ]] ||
      fail 'the two ARM64 nodes ran different canary container images'
  fi
done

kubectl -n "$NAMESPACE" delete pod "${PODS[@]}" --wait=true >/dev/null
kubectl -n "$NAMESPACE" delete configmap "$CONFIGMAP" --wait=true >/dev/null
CREATED=0
remaining="$(kubectl -n "$NAMESPACE" get pod,configmap -l "$LABEL" -o name)"
[[ -z "$remaining" ]] || fail "canary cleanup left objects behind: $remaining"

printf 'SPARK_PAIR_DEVICE_BARRIER_ARM64_GATE_PASS nodes=2 gate_sha=%s source_sha=%s binary_sha=%s image_id=%s bpf_tag=%s target=STRICT_CHILD lifetime=FD_SCOPED access=MKNOD_DENIED detach=BASELINE_RESTORED injected_failure_cleanup=PASS cleanup=PASS\n' \
  "$GATE_SHA" "$SOURCE_SHA" "$binary_sha" "$container_image_id" "$FROZEN_BPF_TAG"
