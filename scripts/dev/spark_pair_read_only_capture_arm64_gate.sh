#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOURCE="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.cpp"
MATERIAL_SELFTEST="$ROOT_DIR/scripts/ci/spark_pair_read_only_capture_material_selftest.sh"
PROFILE_BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_read_only_capture_profile.sh"
PROFILE_FREEZE="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture_profile.freeze.v1"
RESTORE_FREEZE="$ROOT_DIR/tools/cluster/spark_pair_restore_capsule.freeze.v1"
NODE0_RESTORABLE="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node0-restorable.v1"
NODE0_OBSERVATION="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node0-observation.v1"
NODE0_MANIFEST="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node0-manifest.v1"
NODE1_RESTORABLE="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node1-restorable.v1"
NODE1_OBSERVATION="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node1-observation.v1"
NODE1_MANIFEST="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.node1-manifest.v1"
PAIR_RECEIPT="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.pair-receipt.v1"
MATERIAL_RECEIPT="$ROOT_DIR/tools/cluster/spark_pair_read_only_capture.material-parity.v1"
MODE="${1:-}"
NODES=(spark-3c59 spark-8e54)
MULTUS_NAMESPACE=kube-system
MULTUS_DAEMONSET=kube-multus-ds
MULTUS_CONTAINER=kube-multus
MULTUS_IMAGE=ghcr.io/k8snetworkplumbingwg/multus-cni:v4.2.4-thick
EXPECTED_MULTUS_IMAGE_ID=ghcr.io/k8snetworkplumbingwg/multus-cni@sha256:3c20900b5381fac7f9cbbdfac8370ea10a2f6ed7fbecc678384a9db57047abb1
SLURM_NAMESPACE=slurm-pilot
SLURM_LOGIN=slurm-pilot-login-slinky
EXPECTED_PROFILE_FREEZE_SHA=3edfa1e7394b8e82ce8d5e4c81e0450b88dc5b72e1eb71c6acf33f6e2c705223
EXPECTED_RESTORE_FREEZE_SHA=d1d67253355be3deab0b3faf05fb345497b1c98dfc15f1194b787830e632fb50
EXPECTED_COLLECTOR_SOURCE_SHA=385d2756c9ce607834ade6dc22d325090fa04841cfdbb1287278ab19ba34e479
ZERO_SHA=0000000000000000000000000000000000000000000000000000000000000000
WORK=''
declare -A PODS POD_UIDS POD_IMAGE_IDS DS_UIDS REMOTE_DIRS REMOTE_BINS BINARY_SHAS

fail() {
  printf 'spark-pair-read-only-capture-arm64-gate: FAIL: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is missing: $1"
}

digest() {
  sha256sum "$1" | cut -d' ' -f1
}

now_ns() {
  date -u +%s%N
}

receipt_value() {
  local file="$1" key="$2"
  sed -n "s/^${key}=//p" "$file"
}

cleanup_remote_best_effort() {
  local node pod remote
  for node in "${NODES[@]}"; do
    pod="${PODS[$node]:-}"
    remote="${REMOTE_DIRS[$node]:-}"
    [[ -n "$pod" && "$remote" == /dev/shm/pireus-ro-capture.* ]] || continue
    timeout --kill-after=5s 20s kubectl -n "$MULTUS_NAMESPACE" exec \
      -c "$MULTUS_CONTAINER" "$pod" -- chroot /hostroot /bin/bash -ceu '
        remote="$1"
        case "$remote" in /dev/shm/pireus-ro-capture.*) ;; *) exit 64 ;; esac
        rm -f "$remote/collector.cpp" "$remote/collector"
        rmdir "$remote"
      ' _ "$remote" >/dev/null 2>&1 || true
  done
}

cleanup_all() {
  cleanup_remote_best_effort
  [[ -z "$WORK" ]] || rm -rf "$WORK"
}

trap cleanup_all EXIT

validate_local() {
  local profile_sha restore_sha source_sha
  for command in bash c++ cmp cut date install jq kubectl rg sed sha256sum timeout; do
    require_command "$command"
  done
  [[ -f "$SOURCE" && -x "$MATERIAL_SELFTEST" && -x "$PROFILE_BUILD" ]] || \
    fail 'collector source or local material gates are missing'
  profile_sha="$(digest "$PROFILE_FREEZE")"
  restore_sha="$(digest "$RESTORE_FREEZE")"
  source_sha="$(digest "$SOURCE")"
  [[ "$profile_sha" == "$EXPECTED_PROFILE_FREEZE_SHA" ]] || \
    fail "capture profile freeze drifted: $profile_sha"
  [[ "$restore_sha" == "$EXPECTED_RESTORE_FREEZE_SHA" ]] || \
    fail "frame 9027 freeze drifted: $restore_sha"
  [[ "$source_sha" == "$EXPECTED_COLLECTOR_SOURCE_SHA" ]] || \
    fail "material collector source drifted before privileged transport: $source_sha"
  bash "$MATERIAL_SELFTEST" --local-only >/dev/null
}

discover_multus_pod() {
  local node="$1" json pod
  json="$(kubectl -n "$MULTUS_NAMESPACE" get pods -l app=multus -o json)"
  pod="$(jq -er --arg node "$node" --arg ds "$MULTUS_DAEMONSET" \
    --arg container "$MULTUS_CONTAINER" --arg image "$MULTUS_IMAGE" \
    --arg image_id "$EXPECTED_MULTUS_IMAGE_ID" '
      [.items[] | select(
        .spec.nodeName == $node and
        .status.phase == "Running" and
        .spec.hostPID == true and
        (.metadata.ownerReferences | length) == 1 and
        .metadata.ownerReferences[0].apiVersion == "apps/v1" and
        .metadata.ownerReferences[0].kind == "DaemonSet" and
        .metadata.ownerReferences[0].name == $ds and
        .metadata.ownerReferences[0].controller == true and
        any(.spec.containers[];
          .name == $container and .image == $image and
          .securityContext.privileged == true and
          any(.volumeMounts[]?; .mountPath == "/hostroot" and .name == "hostroot")) and
        any(.spec.volumes[]?; .name == "hostroot" and .hostPath.path == "/") and
        any(.status.containerStatuses[]?;
          .name == $container and .ready == true and .imageID == $image_id)
      )] | if length == 1 then .[0].metadata.name else error("multus cardinality") end
    ' <<<"$json")" || fail "$node does not have one exact trusted Multus host bridge"
  PODS[$node]="$pod"
  POD_UIDS[$node]="$(jq -er --arg pod "$pod" \
    '.items[] | select(.metadata.name == $pod) | .metadata.uid' <<<"$json")"
  POD_IMAGE_IDS[$node]="$(jq -er --arg pod "$pod" --arg container "$MULTUS_CONTAINER" \
    '.items[] | select(.metadata.name == $pod) |
     .status.containerStatuses[] | select(.name == $container) | .imageID' <<<"$json")"
  DS_UIDS[$node]="$(jq -er --arg pod "$pod" \
    '.items[] | select(.metadata.name == $pod) | .metadata.ownerReferences[0].uid' \
    <<<"$json")"
  current_ds_uid="$(kubectl -n "$MULTUS_NAMESPACE" get daemonset "$MULTUS_DAEMONSET" \
    -o jsonpath='{.metadata.uid}')"
  [[ "${DS_UIDS[$node]}" == "$current_ds_uid" ]] || \
    fail "$node Multus Pod owner UID is stale"
}

validate_cluster_read_only() {
  local node ready arch allocatable
  for node in "${NODES[@]}"; do
    read -r ready arch allocatable < <(kubectl get node "$node" -o json | jq -er '
      [([.status.conditions[] | select(.type == "Ready")][0].status),
       .status.nodeInfo.architecture,
       (.status.allocatable["nvidia.com/gpu"] // "0")] | @tsv')
    [[ "$ready" == True && "$arch" == arm64 && "$allocatable" == 1 ]] || \
      fail "$node is not one Ready ARM64 GB10 node"
    discover_multus_pod "$node"
  done
  kubectl -n "$SLURM_NAMESPACE" get deployment "$SLURM_LOGIN" >/dev/null
  kubectl -n "$SLURM_NAMESPACE" get nodeset slurm-pilot-worker-spark >/dev/null
}

host_chroot() {
  local node="$1"
  shift
  timeout --kill-after=5s 30s kubectl -n "$MULTUS_NAMESPACE" exec \
    -i -c "$MULTUS_CONTAINER" "${PODS[$node]}" -- chroot /hostroot "$@"
}

host_namespace() {
  local node="$1"
  shift
  host_chroot "$node" /usr/bin/nsenter -t 1 -m -u -i -n -p -- "$@"
}

record_result() {
  local node="$1" label="$2" output="$3" error="$4" rc="$5" started="$6" finished="$7"
  printf 'query=%s transport=%s rc=%s started_unix_ns=%s finished_unix_ns=%s stdout_sha256=%s stderr_sha256=%s\n' \
    "$label" "${8}" "$rc" "$started" "$finished" \
    "$(material_sha "$node" "$output")" "$(material_sha "$node" "$error")" >> \
    "$WORK/$node.transcript"
  [[ "$rc" -eq 0 ]] || fail "$node query $label failed with rc=$rc"
}

material_sha() {
  local node="$1" file="$2" result
  result="$(host_chroot "$node" "${REMOTE_BINS[$node]}" --sha256 < "$file")"
  [[ "$result" =~ ^[0-9a-f]{64}$ ]] || \
    fail "$node native material SHA-256 returned a malformed digest"
  printf '%s\n' "$result"
}

domain_digest() {
  local node="$1" domain="$2" file="$3" result
  result="$(host_chroot "$node" "${REMOTE_BINS[$node]}" --hash-domain "$domain" \
    < "$file")"
  [[ "$result" =~ ^[0-9a-f]{64}$ ]] || \
    fail "$node native domain digest failed for $domain"
  printf '%s\n' "$result"
}

capture_host_script() {
  local node="$1" label="$2" output="$3" script="$4"
  shift 4
  local error="$output.stderr" started finished rc
  started="$(now_ns)"
  set +e
  host_namespace "$node" /bin/bash -ceu "$script" _ "$@" >"$output" 2>"$error"
  rc=$?
  set -e
  finished="$(now_ns)"
  record_result "$node" "$label" "$output" "$error" "$rc" "$started" "$finished" \
    KUBERNETES_PODS_EXEC_READ_ONLY_OBSERVATION
}

capture_local_script() {
  local node="$1" label="$2" output="$3" script="$4"
  shift 4
  local error="$output.stderr" started finished rc
  started="$(now_ns)"
  set +e
  bash -o pipefail -ceu "$script" _ "$@" >"$output" 2>"$error"
  rc=$?
  set -e
  finished="$(now_ns)"
  record_result "$node" "$label" "$output" "$error" "$rc" "$started" "$finished" \
    KUBERNETES_API_GET_OR_SLURM_QUERY
}

system_units_for() {
  case "$1" in
    spark-3c59)
      printf '%s\n' docker.service kubelet.service nvidia-persistenced.service \
        ollama.service vxlan-cluster.service ;;
    spark-8e54)
      printf '%s\n' docker.service kubelet.service nvidia-persistenced.service \
        tei-shim.service vxlan-cluster.service ;;
    *) return 64 ;;
  esac
}

capture_systemd_system() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.systemd-system.$suffix"
  local units=()
  mapfile -t units < <(system_units_for "$node")
  capture_host_script "$node" "systemd_system_$suffix" "$output" '
    export LC_ALL=C SYSTEMD_COLORS=0 SYSTEMD_PAGER=cat
    for unit in "$@"; do
      printf "unit=%s\n" "$unit"
      systemctl show "$unit" --no-pager \
        -p Id -p LoadState -p UnitFileState -p FragmentPath -p DropInPaths \
        -p Restart -p RestartSec -p ExecStart
      systemctl cat "$unit" --no-pager
    done
  ' "${units[@]}"
}

capture_systemd_user() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.systemd-user.$suffix"
  capture_host_script "$node" "systemd_user_$suffix" "$output" '
    export LC_ALL=C
    root=/home/demetrios/.config/systemd/user
    printf "root=%s\n" "$root"
    if [[ ! -d "$root" ]]; then printf "status=NOT_PRESENT\n"; exit 0; fi
    while IFS= read -r -d "" file; do
      stat -c "path=%n mode=%a uid=%u gid=%g size=%s mtime_ns=%Y" "$file"
      sha256sum "$file"
    done < <(find "$root" -xdev -type f -print0 | sort -z)
  '
}

capture_docker_recreate() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.docker-recreate.$suffix"
  capture_host_script "$node" "docker_recreate_$suffix" "$output" '
    export LC_ALL=C
    while IFS= read -r id; do
      [[ -n "$id" ]] || continue
      docker inspect --format \
"id={{.Id}} name={{.Name}} image={{json .Config.Image}} cmd={{json .Config.Cmd}} entrypoint={{json .Config.Entrypoint}} restart={{json .HostConfig.RestartPolicy}} ports={{json .HostConfig.PortBindings}} binds={{json .HostConfig.Binds}} mounts={{json .Mounts}} devices={{json .HostConfig.Devices}} requests={{json .HostConfig.DeviceRequests}} network={{json .HostConfig.NetworkMode}} privileged={{json .HostConfig.Privileged}} readonly={{json .HostConfig.ReadonlyRootfs}}" \
        "$id"
    done < <(docker ps -aq --no-trunc | sort)
  '
}

capture_nodeset() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.nodeset.$suffix"
  capture_local_script "$node" "nodeset_spec_$suffix" "$output" '
    kubectl -n slurm-pilot get nodeset slurm-pilot-worker-spark -o json |
      jq -S "del(.metadata.managedFields, .metadata.resourceVersion, .status)"
  '
}

capture_device_plugin() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.device-plugin.$suffix"
  capture_local_script "$node" "device_plugin_spec_$suffix" "$output" '
    node="$1"; name="nvidia-device-plugin-${node#spark-}"
    kubectl -n kube-system get daemonset "$name" -o json |
      jq -S "del(.metadata.managedFields, .metadata.resourceVersion, .status)"
  ' "$node"
}

capture_taints() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.taints.$suffix"
  capture_local_script "$node" "taints_$suffix" "$output" '
    kubectl get node "$1" -o json | jq -S ".spec.taints // []"
  ' "$node"
}

capture_labels() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.labels.$suffix"
  capture_local_script "$node" "labels_$suffix" "$output" '
    kubectl get node "$1" -o json | jq -S ".metadata.labels"
  ' "$node"
}

capture_protected_metadata() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.protected-metadata.$suffix"
  capture_host_script "$node" "protected_paths_metadata_$suffix" "$output" '
    export LC_ALL=C
    for path in "$@"; do
      printf "path=%s\n" "$path"
      if [[ -e "$path" ]]; then
        stat -c "status=PRESENT type=%F mode=%a uid=%u gid=%g size=%s mtime_epoch=%Y" "$path"
        findmnt -n -T "$path" -o TARGET,SOURCE,FSTYPE,OPTIONS
      else
        printf "status=NOT_PRESENT\n"
      fi
    done
  ' /home/demetrios/beagle-memory-pg-pdb/data /opt/sounio-ckpt /opt/sounio-py
}

capture_managed_surfaces() {
  local node="$1" suffix="$2"
  capture_systemd_system "$node" "$suffix"
  capture_systemd_user "$node" "$suffix"
  capture_docker_recreate "$node" "$suffix"
  capture_nodeset "$node" "$suffix"
  capture_device_plugin "$node" "$suffix"
  capture_taints "$node" "$suffix"
  capture_labels "$node" "$suffix"
}

managed_bundle() {
  local node="$1" suffix="$2" output="$3" file_domain digest_domain
  : > "$output"
  while IFS='|' read -r file_domain digest_domain; do
    printf '%s_sha256=%s\n' "$file_domain" \
      "$(domain_digest "$node" "$digest_domain" \
        "$WORK/$node.$file_domain.$suffix")" >> "$output"
  done <<'EOF'
systemd-system|restorable.systemd_system
systemd-user|restorable.systemd_user
docker-recreate|restorable.docker_recreate
nodeset|restorable.nodeset_spec
device-plugin|restorable.device_plugin_spec
taints|restorable.taints
labels|restorable.labels
EOF
}

capture_boot_identity() {
  local node="$1" suffix="$2"
  local output="$WORK/$node.boot.$suffix"
  capture_host_script "$node" "boot_identity_$suffix" "$output" '
    printf "boot_id="; cat /proc/sys/kernel/random/boot_id
    printf "architecture="; uname -m
    printf "kernel="; uname -r
    printf "machine="; uname -n
    printf "cmdline_sha256="; sha256sum /proc/cmdline | cut -d" " -f1
  '
}

capture_runtime_observations() {
  local node="$1"
  local units=()
  mapfile -t units < <(system_units_for "$node")
  capture_host_script "$node" systemd_runtime "$WORK/$node.systemd-runtime.raw" '
    export LC_ALL=C SYSTEMD_COLORS=0 SYSTEMD_PAGER=cat
    for unit in "$@"; do
      systemctl show "$unit" --no-pager -p Id -p ActiveState -p SubState \
        -p MainPID -p NRestarts -p ExecMainStartTimestampMonotonic
    done
  ' "${units[@]}"
  capture_host_script "$node" docker_runtime "$WORK/$node.docker-runtime.raw" '
    export LC_ALL=C
    docker ps --no-trunc --format \
"id={{.ID}} name={{.Names}} image={{.Image}} status={{.Status}} ports={{.Ports}}"
    while IFS= read -r id; do
      [[ -n "$id" ]] || continue
      docker inspect --format \
"id={{.Id}} running={{.State.Running}} pid={{.State.Pid}} started={{.State.StartedAt}} status={{.State.Status}}" "$id"
    done < <(docker ps -q --no-trunc | sort)
  '
  capture_local_script "$node" k8s_identity "$WORK/$node.k8s-identity.raw" '
    node="$1"
    kubectl get node "$node" -o json | jq -S "{
      name:.metadata.name, uid:.metadata.uid, labels:.metadata.labels,
      taints:(.spec.taints // []), unschedulable:(.spec.unschedulable // false),
      ready:([.status.conditions[] | select(.type == \"Ready\")][0].status),
      architecture:.status.nodeInfo.architecture,
      allocatable_gpu:(.status.allocatable[\"nvidia.com/gpu\"] // \"0\") }"
    kubectl get pods -A --field-selector "spec.nodeName=$node" -o json | jq -S "[
      .items[] | {namespace:.metadata.namespace,name:.metadata.name,uid:.metadata.uid,
        owner:(.metadata.ownerReferences[0] // null),phase:.status.phase,
        containers:[.spec.containers[] | {name,image,resources}]}
    ] | sort_by(.namespace,.name)"
  ' "$node"
  capture_local_script "$node" slurm_runtime "$WORK/$node.slurm-runtime.raw" '
    node="$1"; slurm_node="gpuorangefs-multi-$node"
    kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- \
      scontrol show node "$slurm_node" -o
    kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- \
      squeue -a -h --nodes="$slurm_node" -o "%i|%N|%T|%j"
    kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- \
      sinfo -N -h -n "$slurm_node" -o "%N|%T|%G|%C"
  ' "$node"
  capture_host_script "$node" gpu_runtime "$WORK/$node.gpu-runtime.raw" '
    nvidia-smi --query-gpu=uuid,name,driver_version,pstate,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits
  '
  capture_host_script "$node" bpf_runtime "$WORK/$node.bpf-runtime.raw" '
    bpftool -j prog show
    bpftool -j cgroup tree /sys/fs/cgroup
  '
  capture_protected_metadata "$node" current
  capture_host_script "$node" toolchain_hardware_commands \
    "$WORK/$node.toolchain.raw" '
    source="$1"; binary="$2"; expected_source="$3"; gate_sha="$4"
    [[ "$(sha256sum "$source" | cut -d" " -f1)" == "$expected_source" ]]
    printf "collector_source_sha256=%s\n" "$expected_source"
    printf "collector_binary_sha256="; sha256sum "$binary" | cut -d" " -f1
    printf "gate_source_sha256=%s\n" "$gate_sha"
    printf "architecture="; uname -m
    printf "kernel="; uname -r
    c++ --version | sed -n "1,2p"
    ldd --version | sed -n "1p"
    lscpu
    printf "command_profile=PIREUS_READ_ONLY_CAPTURE_FIXED_ENUM_V1\n"
  ' "${REMOTE_DIRS[$node]}/collector.cpp" "${REMOTE_BINS[$node]}" \
    "$SOURCE_SHA" "$GATE_SHA"
}

materialize_node() {
  local node="$1" remote source_remote binary_remote output fixture
  remote="$(host_chroot "$node" /usr/bin/mktemp -d \
    /dev/shm/pireus-ro-capture.XXXXXXXX)"
  [[ "$remote" == /dev/shm/pireus-ro-capture.* ]] || \
    fail "$node returned an unsafe ephemeral materialization path"
  REMOTE_DIRS[$node]="$remote"
  source_remote="$remote/collector.cpp"
  binary_remote="$remote/collector"
  REMOTE_BINS[$node]="$binary_remote"
  host_chroot "$node" /bin/bash -ceu '
    umask 077
    target="$1"
    case "$target" in /dev/shm/pireus-ro-capture.*/collector.cpp) ;; *) exit 64 ;; esac
    cat > "$target"
  ' _ "$source_remote" < "$SOURCE"
  output="$WORK/$node.materialization.raw"
  host_chroot "$node" /bin/bash -ceu '
    source="$1"; binary="$2"; expected="$3"
    [[ "$(uname -m)" == aarch64 ]]
    [[ "$(sha256sum "$source" | cut -d" " -f1)" == "$expected" ]]
    c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
      -fno-exceptions -fno-rtti -fstack-protector-strong -D_FORTIFY_SOURCE=3 \
      "$source" -o "$binary"
    sha256sum "$source" "$binary"
    "$binary" --selftest
  ' _ "$source_remote" "$binary_remote" "$SOURCE_SHA" > "$output"
  rg -F "$SOURCE_SHA  $source_remote" "$output" >/dev/null || \
    fail "$node did not compile the exact collector source"
  rg -F 'PIREUS_SPARK_PAIR_READ_ONLY_CAPTURE_CPP_SELFTEST_PASS role=MATERIAL_OBSERVER_NON_AUTHORITY' \
    "$output" >/dev/null || fail "$node ARM64 collector selftest failed"
  BINARY_SHAS[$node]="$(sed -n "s#  $binary_remote\$##p" "$output")"
  [[ "${BINARY_SHAS[$node]}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "$node ARM64 binary digest is malformed"
  for fixture in node0 node1 node0-restorable node1-restorable \
      node0-observation node1-observation domain-contract pair; do
    host_chroot "$node" "$binary_remote" "--fixture-$fixture" > \
      "$WORK/$node.fixture-$fixture"
    cmp -s "$WORK/sounio.fixture-$fixture" "$WORK/$node.fixture-$fixture" || \
      fail "$node ARM64 $fixture bytes differ from frozen Sounio bytes"
  done
}

produce_node_receipts() {
  local node="$1" prefix="$2" started="$3" finished="$4"
  local restorable_hashes observation_hashes binary
  binary="${REMOTE_BINS[$node]}"
  restorable_hashes=(
    "$(domain_digest "$node" restorable.systemd_system "$WORK/$node.systemd-system.pre")"
    "$(domain_digest "$node" restorable.systemd_user "$WORK/$node.systemd-user.pre")"
    "$(domain_digest "$node" restorable.docker_recreate "$WORK/$node.docker-recreate.pre")"
    "$(domain_digest "$node" restorable.nodeset_spec "$WORK/$node.nodeset.pre")"
    "$(domain_digest "$node" restorable.device_plugin_spec "$WORK/$node.device-plugin.pre")"
    "$(domain_digest "$node" restorable.taints "$WORK/$node.taints.pre")"
    "$(domain_digest "$node" restorable.labels "$WORK/$node.labels.pre")"
    "$(domain_digest "$node" restorable.protected_paths_metadata "$WORK/$node.protected-metadata.pre")"
  )
  observation_hashes=(
    "$(domain_digest "$node" observation.boot_identity "$WORK/$node.boot.pre")"
    "$(domain_digest "$node" observation.systemd_runtime "$WORK/$node.systemd-runtime.raw")"
    "$(domain_digest "$node" observation.docker_runtime "$WORK/$node.docker-runtime.raw")"
    "$(domain_digest "$node" observation.k8s_identity "$WORK/$node.k8s-identity.raw")"
    "$(domain_digest "$node" observation.slurm_runtime "$WORK/$node.slurm-runtime.raw")"
    "$(domain_digest "$node" observation.gpu_runtime "$WORK/$node.gpu-runtime.raw")"
    "$(domain_digest "$node" observation.bpf_runtime "$WORK/$node.bpf-runtime.raw")"
    "$(domain_digest "$node" observation.protected_paths_current "$WORK/$node.protected-metadata.current")"
    "$(domain_digest "$node" observation.toolchain_hardware_commands "$WORK/$node.toolchain.raw")"
    "$(domain_digest "$node" observation.capture_transcript "$WORK/$node.transcript")"
    "$(domain_digest "$node" observation.managed_state_sentinel "$WORK/$node.managed.pre")"
    "$(domain_digest "$node" observation.managed_state_sentinel "$WORK/$node.managed.post")"
  )
  host_chroot "$node" "$binary" --restorable "$node" \
    "${restorable_hashes[@]}" > "$WORK/$prefix-restorable.v1"
  host_chroot "$node" "$binary" --observation "$node" \
    "${observation_hashes[@]}" "$started" "$finished" true true true > \
    "$WORK/$prefix-observation.v1"
  host_chroot "$node" "$binary" --node "$node" \
    "${restorable_hashes[@]}" "${observation_hashes[@]}" \
    "$started" "$finished" true true true > "$WORK/$prefix-manifest.v1"
}

capture_node() {
  local node="$1" prefix="$2" started finished domain
  : > "$WORK/$node.transcript"
  started="$(now_ns)"
  capture_boot_identity "$node" pre
  capture_managed_surfaces "$node" pre
  capture_protected_metadata "$node" pre
  managed_bundle "$node" pre "$WORK/$node.managed.pre"
  capture_runtime_observations "$node"
  capture_managed_surfaces "$node" post
  managed_bundle "$node" post "$WORK/$node.managed.post"
  capture_boot_identity "$node" post
  finished="$(now_ns)"

  cmp -s "$WORK/$node.boot.pre" "$WORK/$node.boot.post" || \
    fail "$node rebooted or changed boot identity during capture"
  cmp -s "$WORK/$node.managed.pre" "$WORK/$node.managed.post" || \
    fail "$node managed scheduler or host configuration changed during capture"
  for domain in systemd-system systemd-user docker-recreate nodeset \
      device-plugin taints labels; do
    cmp -s "$WORK/$node.$domain.pre" "$WORK/$node.$domain.post" || \
      fail "$node managed domain $domain changed during capture"
  done
  produce_node_receipts "$node" "$prefix" "$started" "$finished"
}

cleanup_remote_strict() {
  local node remote
  for node in "${NODES[@]}"; do
    remote="${REMOTE_DIRS[$node]}"
    host_chroot "$node" /bin/bash -ceu '
      remote="$1"
      case "$remote" in /dev/shm/pireus-ro-capture.*) ;; *) exit 64 ;; esac
      rm -f "$remote/collector.cpp" "$remote/collector"
      rmdir "$remote"
      [[ ! -e "$remote" ]]
    ' _ "$remote"
    REMOTE_DIRS[$node]=''
    REMOTE_BINS[$node]=''
  done
}

publish_receipts() {
  install -m 0644 "$WORK/node0-restorable.v1" "$NODE0_RESTORABLE"
  install -m 0644 "$WORK/node0-observation.v1" "$NODE0_OBSERVATION"
  install -m 0644 "$WORK/node0-manifest.v1" "$NODE0_MANIFEST"
  install -m 0644 "$WORK/node1-restorable.v1" "$NODE1_RESTORABLE"
  install -m 0644 "$WORK/node1-observation.v1" "$NODE1_OBSERVATION"
  install -m 0644 "$WORK/node1-manifest.v1" "$NODE1_MANIFEST"
  install -m 0644 "$WORK/pair-receipt.v1" "$PAIR_RECEIPT"
  install -m 0644 "$WORK/material-parity.v1" "$MATERIAL_RECEIPT"
}

[[ "$MODE" == --check || "$MODE" == --capture ]] || \
  fail 'usage: spark_pair_read_only_capture_arm64_gate.sh --check|--capture'
[[ -z "${PIREUS_CAPTURE_ORACLE:-}" ]] || \
  fail "external oracle injection is forbidden: $PIREUS_CAPTURE_ORACLE"
validate_local
validate_cluster_read_only
SOURCE_SHA="$(digest "$SOURCE")"
GATE_SHA="$(digest "${BASH_SOURCE[0]}")"

if [[ "$MODE" == --check ]]; then
  printf 'SPARK_PAIR_READ_ONLY_CAPTURE_ARM64_PREFLIGHT_PASS nodes=2 capture_executed=false authority=Sounio material=C++20 preflight_cluster_mutation=NONE preflight_host_configuration_mutation=NONE\n'
  exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-read-only-capture-arm64.XXXXXX")"
profile_adapter="$WORK/profile"
SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_OUTPUT="$profile_adapter" \
  bash "$PROFILE_BUILD" >/dev/null
for fixture in node0 node1 node0-restorable node1-restorable \
    node0-observation node1-observation domain-contract pair; do
  "$profile_adapter" "--fixture-$fixture" > "$WORK/sounio.fixture-$fixture"
done

for node in "${NODES[@]}"; do materialize_node "$node"; done
[[ "${BINARY_SHAS[spark-3c59]}" == "${BINARY_SHAS[spark-8e54]}" ]] || \
  fail 'the two native ARM64 materializations are not byte-identical'

capture_node spark-3c59 node0
capture_node spark-8e54 node1

for prefix in node0 node1; do
  [[ "$(receipt_value "$WORK/$prefix-observation.v1" scheduler_mutation)" == NONE &&
     "$(receipt_value "$WORK/$prefix-observation.v1" host_configuration_mutation)" == NONE &&
     "$(receipt_value "$WORK/$prefix-manifest.v1" scheduler_mutation)" == NONE &&
     "$(receipt_value "$WORK/$prefix-manifest.v1" host_configuration_mutation)" == NONE ]] || \
    fail "$prefix material receipt did not prove the exact NONE mutation state"
  [[ "$(receipt_value "$WORK/$prefix-observation.v1" managed_state_pre_sha256)" == \
     "$(receipt_value "$WORK/$prefix-observation.v1" managed_state_post_sha256)" ]] || \
    fail "$prefix material receipt did not bind equal managed-state sentinels"
  for receipt in restorable observation manifest; do
    [[ "$(receipt_value "$WORK/$prefix-$receipt.v1" restorable)" == false &&
       "$(receipt_value "$WORK/$prefix-$receipt.v1" snapshot_binding_receipt)" == NOT_ISSUED &&
       "$(receipt_value "$WORK/$prefix-$receipt.v1" state_transition)" == false ]] || \
      fail "$prefix $receipt receipt promoted forbidden restore authority"
  done
done

node0_manifest_sha="$(material_sha spark-3c59 "$WORK/node0-manifest.v1")"
node0_restorable_sha="$(material_sha spark-3c59 "$WORK/node0-restorable.v1")"
node0_observation_sha="$(material_sha spark-3c59 "$WORK/node0-observation.v1")"
node1_manifest_sha="$(material_sha spark-8e54 "$WORK/node1-manifest.v1")"
node1_restorable_sha="$(material_sha spark-8e54 "$WORK/node1-restorable.v1")"
node1_observation_sha="$(material_sha spark-8e54 "$WORK/node1-observation.v1")"
host_chroot spark-3c59 "${REMOTE_BINS[spark-3c59]}" --pair \
  "$node0_manifest_sha" "$node0_restorable_sha" "$node0_observation_sha" \
  "$node1_manifest_sha" "$node1_restorable_sha" "$node1_observation_sha" > \
  "$WORK/pair-receipt.v1"
pair_sha="$(material_sha spark-3c59 "$WORK/pair-receipt.v1")"

decision="$($profile_adapter "$pair_sha" "$node0_manifest_sha" "$node1_manifest_sha" \
  "$ZERO_SHA" 0 131071 127 0 0)"
[[ "$decision" == SOUNIO_SPARK_PAIR_READ_ONLY_CAPTURE_PROFILE_PASS* ]] || \
  fail "live pair did not satisfy the strict observation profile: $decision"
[[ "$decision" == *'frame_9027_invoked=true restore_allowed=false reason=PREINSTALL_PROVENANCE code=315 '* ]] || \
  fail "live pair did not terminate at exact frame 9027 DENY315: $decision"
case "$decision" in
  *'restorable=true'*|*'snapshot_binding_receipt=ISSUED'*|*'state_transition=true'*)
    fail 'live pair was promoted into forbidden restore authority' ;;
esac

cleanup_remote_strict
captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$WORK/material-parity.v1" <<EOF
schema=sounio-spark-pair-read-only-capture-material-parity-v1
status=ARM64_PAIR_CAPTURED_FRAME_9027_DENY315
captured_at_utc=$captured_at
supersedes_rejected_pair_sha256=c0f6235dc7b93aca8d674ba66c28d66ea34d00b0bcd5b36904cef8b8891120a9
superseded_rejection_reason=DOMAIN_SEPARATED_SENTINEL_CONTRADICTION
semantic_authority=Sounio
semantic_authority_role=SEMANTIC_AUTHORITY
profile_freeze_sha256=$EXPECTED_PROFILE_FREEZE_SHA
capsule_frame=9027
capsule_freeze_sha256=$EXPECTED_RESTORE_FREEZE_SHA
material_language=C++20
material_role=MATERIAL_OBSERVER_NON_AUTHORITY
collector_source_sha256=$SOURCE_SHA
capture_gate_source_sha256=$GATE_SHA
arm64_node0=spark-3c59
arm64_node0_multus_pod=${PODS[spark-3c59]}
arm64_node0_multus_pod_uid=${POD_UIDS[spark-3c59]}
arm64_node0_multus_image_id=${POD_IMAGE_IDS[spark-3c59]}
arm64_node0_binary_sha256=${BINARY_SHAS[spark-3c59]}
arm64_node1=spark-8e54
arm64_node1_multus_pod=${PODS[spark-8e54]}
arm64_node1_multus_pod_uid=${POD_UIDS[spark-8e54]}
arm64_node1_multus_image_id=${POD_IMAGE_IDS[spark-8e54]}
arm64_node1_binary_sha256=${BINARY_SHAS[spark-8e54]}
arm64_binary_parity=true
fixture_byte_parity=8
native_domain_hashing=true
domain_frame_schema=sounio-spark-read-only-domain-frame-v1
node0_manifest_sha256=$node0_manifest_sha
node0_restorable_sha256=$node0_restorable_sha
node0_observation_sha256=$node0_observation_sha
node1_manifest_sha256=$node1_manifest_sha
node1_restorable_sha256=$node1_restorable_sha
node1_observation_sha256=$node1_observation_sha
pair_manifest_sha256=$pair_sha
historical_preinstall_receipt=NOT_PRESENT
historical_preinstall_receipt_sha256=$ZERO_SHA
protected_content_receipt=NOT_OBSERVED
producer_effect=READ_ONLY_OBSERVATION
transport_effect=KUBERNETES_PODS_EXEC_PROCESS_AND_AUDIT_EVENTS
ephemeral_materialization_mutation=DEV_SHM_ONLY_REMOVED_AND_VERIFIED
scheduler_mutation=NONE
host_configuration_mutation=NONE
frame_9027_invoked=true
restore_allowed=false
restore_reason=PREINSTALL_PROVENANCE
restore_code=315
decider_effect=NONE
material_dispatch=false
restorable=false
snapshot_binding_receipt=NOT_ISSUED
state_transition=false
offline_replay=NOT_OPEN
claim_ready=false
python_authority=false
rust_authority=false
external_llm_authority=false
sounio_decision=$decision
EOF

publish_receipts
printf 'SPARK_PAIR_READ_ONLY_CAPTURE_ARM64_PASS nodes=2 pair_sha=%s frame_9027=DENY315 restorable=false scheduler_mutation=NONE host_configuration_mutation=NONE\n' \
  "$pair_sha"
