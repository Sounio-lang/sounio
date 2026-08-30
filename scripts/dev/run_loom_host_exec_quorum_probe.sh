#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_host_exec_quorum_capsule.sh"
PROMOTER="$ROOT_DIR/scripts/dev/promote_loom_host_exec_quorum_capsule.sh"

fail() {
  printf 'run-loom-host-exec-quorum-probe: REFUSE reason=%s material_grant=false material_execution=false launch_open=false\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s [--capsule ABSOLUTE_PATH] [--expected-sha256 HEX] [--namespace NAME] [--node NAME] [--selector LABELS] [--receipt-output PATH]\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

CAPSULE=''
EXPECTED_SHA256=''
NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'
RECEIPT_OUTPUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --capsule)
      [[ $# -ge 2 ]] || usage
      CAPSULE="$2"
      shift 2
      ;;
    --expected-sha256)
      [[ $# -ge 2 ]] || usage
      EXPECTED_SHA256="$2"
      shift 2
      ;;
    --namespace)
      [[ $# -ge 2 ]] || usage
      NAMESPACE="$2"
      shift 2
      ;;
    --node)
      [[ $# -ge 2 ]] || usage
      NODE="$2"
      shift 2
      ;;
    --selector)
      [[ $# -ge 2 ]] || usage
      SELECTOR="$2"
      shift 2
      ;;
    --receipt-output)
      [[ $# -ge 2 ]] || usage
      RECEIPT_OUTPUT="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$NAMESPACE" =~ ^[a-z0-9.-]+$ && "$NODE" =~ ^[A-Za-z0-9._-]+$ ]] || fail 'namespace or node is unsafe'
[[ "$SELECTOR" =~ ^[A-Za-z0-9._,/=-]+$ ]] || fail 'pod selector is unsafe'
for input in "$BUILDER" "$PROMOTER"; do
  [[ -f "$input" && ! -L "$input" && -x "$input" ]] || fail "required probe input is absent, linked, or non-executable: $input"
done
for tool in kubectl sha256sum mktemp timeout install; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-exec-quorum-probe.XXXXXX")"
POD=''
REMOTE_ARCHIVE=''
REMOTE_PROMOTER=''
cleanup() {
  if [[ -n "$POD" && -n "$REMOTE_ARCHIVE" && -n "$REMOTE_PROMOTER" ]]; then
    kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
      "rm -f '/proc/1/root$REMOTE_ARCHIVE' '/proc/1/root$REMOTE_PROMOTER'" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

if [[ -z "$CAPSULE" ]]; then
  CAPSULE="$WORK/loom-host-exec-quorum.tar"
  build_output="$($BUILDER --output "$CAPSULE")"
  [[ "$build_output" == 'LOOM_HOST_EXEC_QUORUM_CAPSULE_BUILD PASS '* ]] || fail "capsule build failed: $build_output"
else
  [[ "$CAPSULE" == /* && -f "$CAPSULE" && ! -L "$CAPSULE" ]] || fail 'provided capsule is absent, linked, or non-absolute'
fi
ACTUAL_SHA256="$(sha256_file "$CAPSULE")"
[[ -z "$EXPECTED_SHA256" ]] && EXPECTED_SHA256="$ACTUAL_SHA256"
[[ "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ && "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]] || fail 'capsule hash differs from expected transport hash'
verify_output="$($PROMOTER --archive "$CAPSULE" --expected-sha256 "$EXPECTED_SHA256" --mode verify)"
[[ "$verify_output" == 'LOOM_HOST_EXEC_QUORUM_CAPSULE_VERIFY PASS '* ]] || fail "local capsule verification failed: $verify_output"

mapfile -t candidate_pods < <(
  kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" \
    --field-selector "spec.nodeName=$NODE,status.phase=Running" -o name
)
[[ ${#candidate_pods[@]} -eq 1 ]] || fail "expected one host transport pod on $NODE; found ${#candidate_pods[@]}"
POD="${candidate_pods[0]#pod/}"
[[ "$POD" =~ ^[a-z0-9.-]+$ ]] || fail 'selected transport pod name is unsafe'
pod_boundary="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.hostPID}|{.spec.containers[0].securityContext.privileged}|{.status.phase}')"
[[ "$pod_boundary" == 'true|true|Running' ]] || fail "transport pod lacks privileged hostPID boundary: $pod_boundary"
kubectl -n "$NAMESPACE" exec "$POD" -- sh -lc \
  'command -v nsenter >/dev/null && test "$(id -u)" = 0 && test "$(nsenter -t 1 -m -u -i -n -p -- tr -d "\n" </proc/1/comm)" = systemd' ||
  fail 'transport pod cannot enter the host systemd namespace'

PROMOTER_SHA256="$(sha256_file "$PROMOTER")"
REMOTE_ARCHIVE="/var/tmp/loom-host-exec-quorum-${EXPECTED_SHA256:0:24}.tar"
REMOTE_PROMOTER="/var/tmp/loom-host-exec-quorum-promoter-${PROMOTER_SHA256:0:24}.sh"
kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
  "umask 077; cat > '/proc/1/root$REMOTE_ARCHIVE'" < "$CAPSULE"
kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
  "umask 077; cat > '/proc/1/root$REMOTE_PROMOTER'; chmod 0500 '/proc/1/root$REMOTE_PROMOTER'" < "$PROMOTER"
remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "sha256sum '/proc/1/root$REMOTE_ARCHIVE' '/proc/1/root$REMOTE_PROMOTER' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$EXPECTED_SHA256" && "${remote_hash_lines[1]:-}" == "$PROMOTER_SHA256" ]] || fail 'host transport hash drifted before namespace entry'

set +e
host_output="$(timeout --signal=TERM --kill-after=10s 720s \
  kubectl -n "$NAMESPACE" exec "$POD" -- nsenter -t 1 -m -u -i -n -p -- \
  "$REMOTE_PROMOTER" --archive "$REMOTE_ARCHIVE" \
  --expected-sha256 "$EXPECTED_SHA256" --mode host-gate 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] || fail "host experiment failed or timed out status=$host_status output=$host_output"
[[ "$host_output" == 'sounio-loom-host-exec-quorum-host-gate: HOST_MEASUREMENT_PASS '* && \
   "$host_output" == *'LOOM_HOST_PROCESS_WITNESS_GATE PASS '* && \
   "$host_output" == *'LOOM_PRODUCT_EXEC_INGRESS_DYNAMIC_USER_HOST_GATE PASS '* && \
   "$host_output" == *'loom-product-exec-cell-host: PASS '* && \
   "$host_output" == *' provider_hook_switched=true provider_lifecycle_attached=true '* && \
   "$host_output" == *'LOOM_HOST_EXEC_QUORUM_EXPERIMENT_INSTALL PASS '* ]] || fail 'host experiment receipts diverged'

transport_receipt="LOOM_HOST_EXEC_QUORUM_TRANSPORT PASS namespace=$NAMESPACE node=$NODE pod=$POD archive_sha256=$EXPECTED_SHA256 promoter_sha256=$PROMOTER_SHA256 transport=kubectl+hostPID+nsenter production_activation=false process_witness_core=true affirmative_extinction=true complete_effects=false product_lane_cell_canary=true distinct_uid_product_broker_canary=true fleet_lane_cell_attached=false product_exec_cell_canary=true provider_hook_switched=true provider_lifecycle_attached=true provider_fixture_language=OCaml exec_cell_attached=true material_grant=true material_execution=true test_only=true launch_open=false parity_open=false claim_ready=false host_output_sha256=$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)"
if [[ -n "$RECEIPT_OUTPUT" ]]; then
  receipt_stage="$(mktemp "$(dirname "$RECEIPT_OUTPUT")/.loom-hostq-receipt.XXXXXX")"
  printf '%s\n%s\n' "$transport_receipt" "$host_output" > "$receipt_stage"
  install -m 0644 "$receipt_stage" "$RECEIPT_OUTPUT"
  rm -f "$receipt_stage"
fi
printf '%s\n%s\n' "$host_output" "$transport_receipt"
