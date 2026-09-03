#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_SPARK_PAIR_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_SPARK_PAIR_ENGINE:-madaros}"
MODULE="$ROOT_DIR/stdlib/coordination/spark_pair_arbiter.sio"
VECTORS="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/spark_pair_arbiter_vectors.sio"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_arbiter.sh"
ARBITER="$ROOT_DIR/scripts/dev/spark_pair_arbiter.sh"
MOCK_BACKEND="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/mock_backend.sh"
MATERIAL_BACKEND="$ROOT_DIR/scripts/dev/spark_pair_arbiter_k8s_backend.sh"
POLICY="$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1"
HOST_FENCE="$ROOT_DIR/tools/cluster/spark_pair_host_fence.yaml"
DEVICE_BARRIER="$ROOT_DIR/tools/cluster/pireus_spark_device_barrier.cpp"
DEVICE_BARRIER_ARM64_GATE="$ROOT_DIR/scripts/dev/spark_pair_device_barrier_arm64_gate.sh"
ADMISSION="$ROOT_DIR/tools/cluster/spark_pair_arbiter_admission.yaml"
HOST_FENCE_UNIT="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/host_fence_unit.sh"
K8S_BACKEND_TRANSACTION_UNIT="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/k8s_backend_transaction_unit.sh"
K8S_BACKEND_FENCED_UNIT="$ROOT_DIR/tests/fixtures/spark_pair_arbiter/k8s_backend_fenced_unit.sh"
TEST_FREEZE=''

fail() {
  printf 'spark-pair-arbiter-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-spark-pair-selftest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/spark_pair_arbiter_selftest.sio"
executable="$work/spark_pair_arbiter_selftest"
ADAPTER="$work/sounio-spark-pair-arbiter"
TEST_ARBITER="$work/spark-pair-arbiter-fixture"
MOCK_DIR="$work/mock"
RECEIPTS="$work/receipts"
HOST_FENCE_SCRIPT="$work/host-fence.sh"
RESERVATION_PROBE_SCRIPT="$work/reservation-probe.sh"
DEVICE_BARRIER_EXECUTABLE="$work/pireus-spark-device-barrier"

awk '
  /^  host-fence\.sh: \|$/ { in_script=1; next }
  in_script && /^---$/ { exit }
  in_script { sub(/^    /, ""); print }
' "$HOST_FENCE" > "$HOST_FENCE_SCRIPT"
awk '
  /^  reservation-probe\.sh: \|$/ { in_script=1; next }
  in_script && /^---$/ { exit }
  in_script { sub(/^    /, ""); print }
' "$ADMISSION" > "$RESERVATION_PROBE_SCRIPT"
[[ "$(sed -n '1p' "$HOST_FENCE_SCRIPT")" == '#!/usr/bin/env bash' ]] || \
  fail 'host fence ConfigMap script key or extraction boundary drifted'
bash -n "$HOST_FENCE_SCRIPT" || fail 'host fence ConfigMap script is not valid Bash'
bash -n "$DEVICE_BARRIER_ARM64_GATE" || fail 'ARM64 device-barrier gate is not valid Bash'
grep -Fq 'canary-self /sys/fs/cgroup' "$DEVICE_BARRIER_ARM64_GATE" || \
  fail 'ARM64 gate does not exercise a live child cgroup'
grep -Fq 'access=MKNOD_DENIED detach=BASELINE_RESTORED' "$DEVICE_BARRIER_ARM64_GATE" || \
  fail 'ARM64 gate does not require deny and exact-detach evidence'
if grep -Eiq '\<(python[0-9.]*|rustc|cargo)\>' "$DEVICE_BARRIER_ARM64_GATE"; then
  fail 'ARM64 device-barrier gate invokes a prohibited oracle'
fi
c++ -std=c++20 -O2 -Wall -Wextra -Werror "$DEVICE_BARRIER" \
  -o "$DEVICE_BARRIER_EXECUTABLE"
[[ "$($DEVICE_BARRIER_EXECUTABLE selftest)" == \
    'PIREUS_DEVICE_BARRIER_SELFTEST_PASS majors=498,501 default=ALLOW matched=DENY duplicates=REFUSE root_target=REFUSE' ]] || \
  fail 'device barrier executable selftest failed'
[[ "$(sed -n '1p' "$RESERVATION_PROBE_SCRIPT")" == '#!/usr/bin/env bash' ]] || \
  fail 'reservation probe ConfigMap script key or extraction boundary drifted'
bash -n "$RESERVATION_PROBE_SCRIPT" || fail 'reservation probe ConfigMap script is not valid Bash'
host_fence_sha="$(sha256sum "$HOST_FENCE_SCRIPT" | cut -d ' ' -f 1)"
device_barrier_sha="$(sha256sum "$DEVICE_BARRIER" | cut -d ' ' -f 1)"
reservation_probe_sha="$(sha256sum "$RESERVATION_PROBE_SCRIPT" | cut -d ' ' -f 1)"
[[ "$(awk -F= '$1 == "host_fence_configmap" { print $2 }' "$POLICY")" == \
    "pireus-spark-host-fence-${host_fence_sha:0:12}" ]] || \
  fail 'host fence ConfigMap is not content addressed by its script'
[[ "$(awk -F= '$1 == "host_device_barrier_configmap" { print $2 }' "$POLICY")" == \
    "pireus-spark-device-barrier-${device_barrier_sha:0:12}" ]] || \
  fail 'device barrier ConfigMap is not content addressed by its C++ source'
[[ "$(awk -F= '$1 == "host_device_inventory_names" { print $2 }' "$POLICY")" == \
    'nvidia,drm,dma_heap,nvidia-uvm,nvidia-caps' ]] || fail 'device inventory name profile drifted'
[[ "$(awk -F= '$1 == "host_device_barrier_names" { print $2 }' "$POLICY")" == \
    'nvidia-uvm,nvidia-caps' ]] || fail 'compute-only device deny name profile drifted'
! grep -Eq '^host_device_(inventory|barrier)_majors=' "$POLICY" || \
  fail 'policy still freezes host-specific NVIDIA major numbers'
grep -Fq 'readonly DEVICE_INVENTORY_NAMES=nvidia,drm,dma_heap,nvidia-uvm,nvidia-caps' "$HOST_FENCE_SCRIPT" || \
  fail 'host fence does not preserve the exact device inventory name profile'
grep -Fq 'readonly DEVICE_BARRIER_NAMES=nvidia-uvm,nvidia-caps' "$HOST_FENCE_SCRIPT" || \
  fail 'host fence does not use the compute-only deny name profile'
grep -Fq 'device_majors_for_names "$DEVICE_INVENTORY_NAMES"' "$HOST_FENCE_SCRIPT" || \
  fail 'host fence does not resolve the inventory profile on each host'
grep -Fq 'device_majors_for_names "$DEVICE_BARRIER_NAMES"' "$HOST_FENCE_SCRIPT" || \
  fail 'host fence does not resolve the deny profile on each host'
[[ "$(awk -F= '$1 == "reservation_probe_configmap" { print $2 }' "$POLICY")" == \
    "pireus-spark-pair-reservation-probe-${reservation_probe_sha:0:12}" ]] || \
  fail 'reservation probe ConfigMap is not content addressed by its script'
grep -Fq 'CRI_RUNTIME_ENDPOINT=unix:///var/run/containerd/containerd.sock' "$HOST_FENCE_SCRIPT" || \
  fail 'host fence does not pin the kubelet CRI endpoint'
[[ "$(grep -Fc -- '--runtime-endpoint "$CRI_RUNTIME_ENDPOINT"' "$HOST_FENCE_SCRIPT")" == 1 ]] || \
  fail 'host fence CRI calls are not routed through the pinned endpoint'
grep -Fq 'printf '\''1\n'\'' > "$group/cgroup.kill"' "$HOST_FENCE_SCRIPT" || \
  fail 'host fence lacks cgroup-v2 atomic workload termination'
grep -Fq 'write_grant_record "$PREPARE_FILE" "PREPARED_$mode"' "$HOST_FENCE_SCRIPT" || \
  fail 'host fence lacks a non-authorizing prepare record'
grep -Fq 'Type=notify' "$HOST_FENCE_SCRIPT" || fail 'host fence systemd unit is not notify/watchdog bound'
grep -Fq '/usr/bin/systemd-notify --pid="$$" WATCHDOG=1' "$HOST_FENCE_SCRIPT" || \
  fail 'host fence heartbeat is not attributed to the main watchdog PID'
grep -Fq 'command: [/bin/bash, /fence/host-fence.sh, daemonset-agent]' "$HOST_FENCE" || \
  fail 'host fence DaemonSet is not a non-mutating exec bridge'
grep -Fq 'pireus.sounio.dev/spark-pair-host-fence-bootstrap: "unbound"' "$HOST_FENCE" || \
  fail 'host fence DaemonSet no longer starts with an inert selector'
grep -Fq 'bind_host_fence_daemonset_uid' "$MATERIAL_BACKEND" || \
  fail 'material backend does not bind the DaemonSet UID before activation'
grep -Fq '.spec.template.spec.runtimeClassName = "nvidia"' "$MATERIAL_BACKEND" || \
  fail 'material backend does not bind Spark Slurmd to the NVIDIA runtime'
grep -Fq "object.spec.runtimeClassName == 'nvidia'" "$ADMISSION" || \
  fail 'admission does not require the NVIDIA runtime for Spark Slurmd'
grep -Fq 'request.userInfo.username == '\''system:serviceaccount:kube-system:daemon-set-controller'\''' \
  "$ROOT_DIR/tools/cluster/spark_pair_arbiter_admission.yaml" || \
  fail 'host infrastructure admission is not bound to the DaemonSet controller identity'
host_unit_result="$(bash "$HOST_FENCE_UNIT" "$HOST_FENCE_SCRIPT" "$work/host-unit")"
[[ "$host_unit_result" == 'HOST_FENCE_UNIT_PASS pair_digest=DENY swapped_receipts=DENY intent_rv=PASS cgroup_mapping=DENY pid_exit_race=PASS live_unknown_pid=DENY notready_kill=PASS systemd_graph=PASS reboot_baseline=PASS failed_cycle_heartbeat=DENY' ]] || \
  fail "host fence executable unit failed: $host_unit_result"
transaction_unit_result="$(bash "$K8S_BACKEND_TRANSACTION_UNIT" "$MATERIAL_BACKEND" "$work/k8s-backend-unit")"
[[ "$transaction_unit_result" == 'K8S_BACKEND_TRANSACTION_UNIT_PASS kill_after_commit_1=REFENCED kill_after_commit_2=REFENCED cas_conflict=REFENCED persisted_grants=PROVEN' ]] || \
  fail "Kubernetes backend transaction unit failed: $transaction_unit_result"
fenced_unit_result="$(bash "$K8S_BACKEND_FENCED_UNIT" "$MATERIAL_BACKEND" "$work/k8s-backend-fenced-unit")"
[[ "$fenced_unit_result" == 'K8S_BACKEND_FENCED_UNIT_PASS first_bootstrap=PASS stale_intent=CLEARED epoch_relation=PASS' ]] || \
  fail "Kubernetes backend fenced unit failed: $fenced_unit_result"

sed -n '1,$p' "$MODULE" "$VECTORS" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$executable"
result="$($executable)"
[[ "$result" == 'SOUNIO_SPARK_PAIR_SELFTEST_PASS vectors=88 authority=Sounio' ]] || \
  fail "Sounio vectors failed: $result"
[[ "$(grep -Fc "printf 'PIREUS_NVML_CLEAN node=%s epoch=%s uuid=%s product=%s driver=%s memory_observation=%s utilization_pct=%s" "$MATERIAL_BACKEND")" == 1 &&
   "$(grep -Fc "printf 'PIREUS_NVML_CLEAN node=%s epoch=%s uuid=%s product=%s driver=%s memory_observation=%s utilization_pct=%s" "$ROOT_DIR/tools/cluster/spark_pair_arbiter_admission.yaml")" == 1 ]] || \
  fail 'immutable initial and fresh NVML probes do not share the frozen evidence frame'
[[ "$(grep -Fc "memory=UNAVAILABLE_UNIFIED" "$MATERIAL_BACKEND")" == 1 &&
   "$(grep -Fc "memory=UNAVAILABLE_UNIFIED" "$ROOT_DIR/tools/cluster/spark_pair_arbiter_admission.yaml")" == 1 ]] || \
  fail 'immutable initial and fresh NVML probes do not normalize unified memory identically'
[[ "$(grep -Fc "pgrep -f '[n]vidia-cuda-mps'" "$MATERIAL_BACKEND")" == 1 &&
   "$(grep -Fc "pgrep -f '[n]vidia-cuda-mps'" "$ROOT_DIR/tools/cluster/spark_pair_arbiter_admission.yaml")" == 1 ]] || \
  fail 'immutable initial and fresh MPS probes are not protected against self-match'
grep -Fq "lease_is_live \"\$lease\" || fail 'Lease expired before material keepalive'" "$MATERIAL_BACKEND" || \
  fail 'material keepalive can revive an expired Lease'
grep -Fq "lease_is_live \"\$lease\" || fail 'Lease expired before recording NVML receipts'" "$MATERIAL_BACKEND" || \
  fail 'NVML receipt recording can revive an expired Lease'
[[ "$(grep -Fc 'verify_lease_freeze_binding "$lease"' "$MATERIAL_BACKEND")" -ge 10 ]] || \
  fail 'a Lease mutation path is not bound to the active semantics freeze'
[[ "$(grep -Fc 'verify_bootstrap_journal_binding' "$MATERIAL_BACKEND")" -ge 3 ]] || \
  fail 'bootstrap recovery does not bind the journal to the active semantics freeze'

SOUNIO_SPARK_PAIR_ENGINE="$ENGINE" SOUNIO_SPARK_PAIR_OUTPUT="$ADAPTER" "$BUILD" >/dev/null
install -m 0755 "$ARBITER" "$TEST_ARBITER"
"$ARBITER" verify >/dev/null
DRIFT_FREEZE="$work/drift.freeze.v1"
sed 's/^authority_sha256=.*/authority_sha256=0000000000000000000000000000000000000000000000000000000000000000/' \
  "$FREEZE" > "$DRIFT_FREEZE"
set +e
drift_output="$(SOUNIO_SPARK_PAIR_TEST_MODE=fixture-v1 SOUNIO_SOURCE_ROOT="$ROOT_DIR" \
  SOUNIO_SPARK_PAIR_FREEZE="$DRIFT_FREEZE" "$TEST_ARBITER" verify 2>&1)"
drift_status=$?
set -e
[[ $drift_status -eq 42 && "$drift_output" == *'frozen file drifted: authority_source'* ]] || \
  fail "semantic hash drift did not fail closed: status=$drift_status output=$drift_output"
set +e
override_output="$(SOUNIO_SPARK_PAIR_BACKEND=/bin/true "$ARBITER" verify 2>&1)"
override_status=$?
set -e
[[ $override_status -eq 42 && "$override_output" == *'runtime path overrides are forbidden'* ]] || \
  fail "production root override did not fail closed: status=$override_status output=$override_output"
set +e
timeout_override_output="$(SOUNIO_SPARK_PAIR_COMMAND_TIMEOUT=0 "$ARBITER" verify 2>&1)"
timeout_override_status=$?
set -e
[[ $timeout_override_status -eq 42 && "$timeout_override_output" == *'runtime path overrides are forbidden'* ]] || \
  fail "production timeout override did not fail closed: status=$timeout_override_status output=$timeout_override_output"
set +e
fixture_output="$(SOUNIO_SPARK_PAIR_TEST_MODE=fixture-v1 SOUNIO_SOURCE_ROOT="$ROOT_DIR" "$ARBITER" verify 2>&1)"
fixture_status=$?
set -e
[[ $fixture_status -eq 42 && "$fixture_output" == *'fixture-v1 is forbidden in the canonical controller'* ]] || \
  fail "canonical fixture mode did not fail closed: status=$fixture_status output=$fixture_output"
set +e
malformed="$($ADAPTER 9024 14 1 1 1 1017 255 1009 65535 2>&1)"
malformed_status=$?
set -e
[[ $malformed_status -eq 64 ]] || fail "malformed frame exited $malformed_status, expected 64"
[[ "$malformed" == *'reason=MALFORMED_FRAME code=104'* ]] || \
  fail "malformed frame did not preserve Sounio reason: $malformed"

export SOUNIO_SPARK_PAIR_BACKEND="$MOCK_BACKEND"
export SOUNIO_SPARK_PAIR_POLICY="$POLICY"
export SOUNIO_SPARK_PAIR_TEST_MODE=fixture-v1
export SOUNIO_SOURCE_ROOT="$ROOT_DIR"
TEST_FREEZE="$work/mock.freeze.v1"
adapter_hash="$(sha256sum "$ADAPTER" | cut -d ' ' -f 1)"
mock_hash="$(sha256sum "$MOCK_BACKEND" | cut -d ' ' -f 1)"
sed \
  -e "s|^native_executable_sha256=.*|native_executable_sha256=$adapter_hash|" \
  -e 's|^material_backend_source=.*|material_backend_source=tests/fixtures/spark_pair_arbiter/mock_backend.sh|' \
  -e "s|^material_backend_sha256=.*|material_backend_sha256=$mock_hash|" \
  "$FREEZE" > "$TEST_FREEZE"
export SOUNIO_SPARK_PAIR_FREEZE="$TEST_FREEZE"
export SOUNIO_SPARK_PAIR_AUTHORITY="$ADAPTER"
export SOUNIO_SPARK_PAIR_RECEIPT_DIR="$RECEIPTS"
export SOUNIO_SPARK_PAIR_MOCK_DIR="$MOCK_DIR"
ARBITER="$TEST_ARBITER"

reset_mock() {
  rm -rf "$MOCK_DIR"
  mkdir -p "$MOCK_DIR"
  "$MOCK_BACKEND" --policy "$POLICY" --freeze "$FREEZE" fixture-slurm-owned
}

reset_bootstrap() {
  rm -rf "$MOCK_DIR"
  mkdir -p "$MOCK_DIR"
  "$MOCK_BACKEND" --policy "$POLICY" --freeze "$FREEZE" fixture-uninitialized
}

reset_empty() {
  rm -rf "$MOCK_DIR"
  mkdir -p "$MOCK_DIR"
}

expect_refusal() {
  local name="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 42 ]] || fail "$name exited $status, expected 42: $output"
}

expect_refusal_reason() {
  local name="$1" reason="$2"
  shift 2
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 42 ]] || fail "$name exited $status, expected 42: $output"
  [[ "$output" == *"reason=$reason"* ]] || fail "$name did not preserve $reason: $output"
}

host_action_receipt() {
  local action="$1" state="$2" state_number="$3" host_mask="$4" output receipt
  output="$("$ADAPTER" 9025 "$action" "$state_number" 1 1 1017 255 1009 "$host_mask")"
  [[ "$output" == SOUNIO_SPARK_PAIR_ALLOW* ]] || fail "host action $action was not admitted: $output"
  receipt="$work/host-action-$action.receipt"
  {
    printf 'decision_producer_language=Sounio\n'
    printf 'epoch=1\n'
    printf 'action_code=%s\n' "$action"
    printf 'from_state=%s\n' "$state"
    printf 'expected_to_state=%s\n' "$state"
  } > "$receipt"
  sha256sum "$receipt" | cut -d ' ' -f 1 > "$receipt.sha256"
  printf '%s\n' "$receipt"
}

reset_mock
SOUNIO_SPARK_PAIR_HOLDER=holder-positive "$ARBITER" hold 1 >/dev/null
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'positive hold did not return the pair to Slurm'

reset_empty
SOUNIO_SPARK_PAIR_HOLDER=bootstrap-old "$ARBITER" bootstrap-init >/dev/null
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'happy bootstrap did not establish Slurm ownership'
[[ "$(sed -n '1p' "$MOCK_DIR/nodeset_generation")" == 2 ]] || fail 'happy bootstrap did not refresh NodeSet generation'
[[ "$(sed -n '1,4p' "$MOCK_DIR/effects")" == \
   $'install-fence\ninstall-host-fence\ndrain-slurm\nfence-host-pair' ]] || \
  fail 'bootstrap did not arm admission and host fences before draining and activating the host fence'

reset_bootstrap
receipt="$(host_action_receipt 29 UNINITIALIZED 0 0)"
"$MOCK_BACKEND" --policy "$POLICY" --freeze "$TEST_FREEZE" install-host-fence \
  --holder bootstrap-old --epoch 1 --receipt "$receipt"
receipt="$(host_action_receipt 30 UNINITIALIZED 0 131071)"
"$MOCK_BACKEND" --policy "$POLICY" --freeze "$TEST_FREEZE" fence-host-pair \
  --holder bootstrap-old --epoch 1 --receipt "$receipt"
receipt="$(host_action_receipt 31 UNINITIALIZED 0 131071)"
"$MOCK_BACKEND" --policy "$POLICY" --freeze "$TEST_FREEZE" grant-host-slurm \
  --holder bootstrap-old --epoch 1 --receipt "$receipt"
printf 'SLURM_QUIESCENT\n' > "$MOCK_DIR/state"
receipt="$(host_action_receipt 32 SLURM_QUIESCENT 3 131071)"
"$MOCK_BACKEND" --policy "$POLICY" --freeze "$TEST_FREEZE" grant-host-k8s \
  --holder bootstrap-old --epoch 1 --receipt "$receipt"
[[ "$(sed -n '1,4p' "$MOCK_DIR/effects")" == \
   $'install-host-fence\nfence-host-pair\ngrant-host-slurm\ngrant-host-k8s' ]] || \
  fail 'host action receipts were not bound to their material effects'
[[ "$(sed -n '1p' "$MOCK_DIR/host_grant")" == K8S ]] || fail 'host K8s grant effect was not recorded'

reset_empty
expect_refusal action28-post-lease-crash env \
  SOUNIO_SPARK_PAIR_MOCK_FAIL_AFTER_LEASE=1 \
  SOUNIO_SPARK_PAIR_HOLDER=bootstrap-crash "$ARBITER" bootstrap-init
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == UNINITIALIZED ]] || \
  fail 'action28 post-Lease crash did not preserve the UNINITIALIZED anchor'
[[ "$(sed -n '1p' "$MOCK_DIR/journal")" == 0 ]] || \
  fail 'action28 crash fixture unexpectedly created a journal'
SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 \
  SOUNIO_SPARK_PAIR_HOLDER=bootstrap-crash-recovery \
  "$ARBITER" bootstrap-recover >/dev/null
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || \
  fail 'action28 post-Lease crash was not recovered to Slurm'
[[ "$(sed -n '1p' "$MOCK_DIR/journal")" == 1 ]] || \
  fail 'bootstrap takeover did not reconstruct the missing journal'

reset_bootstrap
expect_refusal live-foreign-bootstrap-takeover env \
  SOUNIO_SPARK_PAIR_HOLDER=bootstrap-foreign "$ARBITER" bootstrap-recover

reset_bootstrap
expect_refusal bootstrap-journal-freeze-drift env \
  SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 \
  SOUNIO_SPARK_PAIR_MOCK_JOURNAL_BOUND=0 \
  SOUNIO_SPARK_PAIR_HOLDER=bootstrap-journal-drift "$ARBITER" bootstrap-recover

for bootstrap_failure in drain-slurm install-fence install-host-fence fence-host-pair \
  grant-host-slurm install-gpu-bound-slurmd resume-slurm; do
  reset_bootstrap
  expect_refusal "bootstrap-$bootstrap_failure" env \
    SOUNIO_SPARK_PAIR_MOCK_FAIL="$bootstrap_failure" \
    SOUNIO_SPARK_PAIR_HOLDER=bootstrap-old "$ARBITER" bootstrap
  [[ "$(sed -n '1p' "$MOCK_DIR/state")" == UNINITIALIZED ]] || \
    fail "$bootstrap_failure did not remain fenced in UNINITIALIZED"
  SOUNIO_SPARK_PAIR_MOCK_FAIL="$bootstrap_failure" \
    SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 \
    SOUNIO_SPARK_PAIR_HOLDER="bootstrap-recovery-$bootstrap_failure" \
    "$ARBITER" bootstrap-recover >/dev/null
  [[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || \
    fail "$bootstrap_failure recovery did not prove Slurm ownership"
  [[ "$(sed -n '1p' "$MOCK_DIR/nodeset_generation")" == 2 ]] || \
    fail "$bootstrap_failure recovery did not refresh NodeSet generation"
done

reset_mock
heartbeat_seconds="$(awk -F= '$1 == "heartbeat_seconds" { print $2 }' "$POLICY")"
[[ "$heartbeat_seconds" =~ ^[1-9][0-9]*$ ]] || fail 'heartbeat policy is not a positive integer'
timing_window_seconds=$((heartbeat_seconds * 3))
SOUNIO_SPARK_PAIR_HOLDER=holder-first "$ARBITER" hold "$timing_window_seconds" >"$work/first-holder.log" 2>&1 &
first_pid=$!
for ((attempt = 1; attempt <= timing_window_seconds; attempt++)); do
  [[ "$(sed -n '1p' "$MOCK_DIR/state")" == K8S_OWNED ]] && break
  sleep 1
done
if [[ "$(sed -n '1p' "$MOCK_DIR/state")" != K8S_OWNED ]]; then
  first_state="$(sed -n '1p' "$MOCK_DIR/state")"
  if kill -0 "$first_pid" >/dev/null 2>&1; then
    first_status=RUNNING
    kill "$first_pid" >/dev/null 2>&1 || true
    wait "$first_pid" >/dev/null 2>&1 || true
  else
    set +e
    wait "$first_pid"
    first_status=$?
    set -e
  fi
  fail "first holder did not reach K8S_OWNED: state=$first_state status=$first_status log=$(sed -n '1,120p' "$work/first-holder.log")"
fi
expect_refusal concurrent-holder env SOUNIO_SPARK_PAIR_HOLDER=holder-second "$ARBITER" hold 1
wait "$first_pid" || fail "first holder failed: $(sed -n '1,120p' "$work/first-holder.log")"
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'concurrent-holder test did not restore Slurm'

reset_mock
expect_refusal forbidden-python-authority env \
  SOUNIO_SPARK_PAIR_AUTHORITY="$(command -v python3)" SOUNIO_SPARK_PAIR_HOLDER=python-oracle "$ARBITER" status

reset_mock
expect_refusal direct-backend-without-receipt "$MOCK_BACKEND" \
  --policy "$POLICY" --freeze "$TEST_FREEZE" drain-slurm \
  --holder unauthorized --epoch 1

reset_mock
expect_refusal stale-epoch env SOUNIO_SPARK_PAIR_MOCK_OBSERVED_EPOCH=999 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-stale "$ARBITER" hold 1

reset_mock
expect_refusal dead-lease env SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-dead "$ARBITER" hold 1

reset_mock
expect_refusal persisted-freeze-drift env SOUNIO_SPARK_PAIR_MOCK_FREEZE_BOUND=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-freeze-drift "$ARBITER" hold 1

reset_mock
expect_refusal_reason host-legacy-consumers HOST_LEGACY_INVENTORY env \
  SOUNIO_SPARK_PAIR_MOCK_HOST_LEGACY_INVENTORY_EXACT=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-host-legacy "$ARBITER" hold 1

reset_mock
expect_refusal_reason host-restart-armed HOST_RESTARTS_ARMED env \
  SOUNIO_SPARK_PAIR_MOCK_HOST_RESTARTS_BLOCKED=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-host-restart "$ARBITER" hold 1

reset_mock
expect_refusal_reason host-boot-mismatch HOST_BOOT_PAIR env \
  SOUNIO_SPARK_PAIR_MOCK_HOST_BOOT_PAIR_BOUND=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-host-boot "$ARBITER" hold 1

reset_mock
expect_refusal_reason host-memory-low HOST_MEMORY_FLOOR env \
  SOUNIO_SPARK_PAIR_MOCK_HOST_MEMORY_FLOOR_MET=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-host-memory "$ARBITER" hold 1

reset_mock
printf '1\n' > "$MOCK_DIR/nvml_clean"
expect_refusal_reason nvml-clean-host-dirty HOST_GPU_CGROUP env \
  SOUNIO_SPARK_PAIR_MOCK_HOST_CGROUPS_EMPTY=0 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-host-dirty "$ARBITER" hold 1

reset_mock
expect_refusal drain-failure env SOUNIO_SPARK_PAIR_MOCK_FAIL=drain-slurm \
  SOUNIO_SPARK_PAIR_HOLDER=holder-drain "$ARBITER" hold 1
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'drain rollback did not restore Slurm'

reset_mock
expect_refusal partial-reservation env SOUNIO_SPARK_PAIR_MOCK_PARTIAL_RESERVATION=1 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-partial "$ARBITER" hold 1
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'partial reservation rollback did not restore Slurm'

reset_mock
expect_refusal heartbeat-loss env SOUNIO_SPARK_PAIR_MOCK_FAIL=lease-renew \
  SOUNIO_SPARK_PAIR_HOLDER=holder-heartbeat "$ARBITER" hold "$timing_window_seconds"
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'heartbeat rollback did not restore Slurm'

reset_mock
expect_refusal sticky-workload env SOUNIO_SPARK_PAIR_MOCK_STICKY_WORKLOAD=1 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-sticky "$ARBITER" hold 1
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == RECOVERY_REQUIRED ]] || fail 'sticky workload was not fenced in recovery'

reset_mock
expect_refusal observation-timeout env SOUNIO_SPARK_PAIR_COMMAND_TIMEOUT=1 \
  SOUNIO_SPARK_PAIR_MOCK_SLEEP_COMMAND=facts SOUNIO_SPARK_PAIR_MOCK_SLEEP_SECONDS=3 \
  SOUNIO_SPARK_PAIR_HOLDER=holder-timeout "$ARBITER" hold 1

reset_mock
SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE=0 SOUNIO_SPARK_PAIR_HOLDER=holder-recovery \
  "$ARBITER" recover >/dev/null
[[ "$(sed -n '1p' "$MOCK_DIR/state")" == SLURM_OWNED ]] || fail 'manual recovery did not prove Slurm ownership'

compgen -G "$RECEIPTS/material-*.receipt" >/dev/null || fail 'material result receipts were not emitted'
grep -h -q '^result=PASS$' "$RECEIPTS"/material-*.receipt || fail 'material PASS receipt is missing'
grep -h -q '^result=FAIL$' "$RECEIPTS"/material-*.receipt || fail 'material FAIL receipt is missing'
awk -F= '$1 == "decision_receipt_sha256" && $2 ~ /^[0-9a-f]+$/ && length($2) == 64 { found=1 } END { exit found ? 0 : 1 }' \
  "$RECEIPTS"/material-*.receipt || fail 'material receipt is not linked to a Sounio decision'

printf '%s\n' "$result"
printf 'SPARK_PAIR_ADAPTER_NEGATIVE_PASS reason=MALFORMED_FRAME status=64\n'
printf 'SPARK_PAIR_MATERIAL_SELFTEST_PASS positive=12 negative=30 freeze_drift=DENY persisted_freeze=DENY journal_freeze=DENY root_override=DENY timeout_override=DENY canonical_fixture=DENY python_oracle=DENY direct_backend=DENY concurrency=DENY bootstrap_recovery=PASS action28_crash_recovery=PASS bootstrap_fence_first=PASS host_actions=PASS host_dirty=DENY recovery_drain_before_fence=PASS material_keepalive_expiry=DENY material_receipts=PASS nvml_formats=PASS host_fence_unit=PASS transaction_kill=REFENCED transaction_cas=REFENCED device_barrier=PASS arm64_child_gate=FROZEN\n'
