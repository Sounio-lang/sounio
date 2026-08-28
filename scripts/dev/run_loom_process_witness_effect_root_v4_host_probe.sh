#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
ROOT_BUILDER="$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_root_v4.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_process_witness_effect_root_v4_host_gate.sh"
NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'

fail() {
  printf 'run-loom-process-witness-effect-root-v4-host-probe: REFUSE reason=%s root_treatment=false bootstrap_sabotage=false material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s [--namespace NAME] [--node NAME] [--selector LABELS]\n' "$0" >&2
  exit 64
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "builder receipt omitted field: $key"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    *) usage ;;
  esac
done

[[ "$NAMESPACE" =~ ^[a-z0-9.-]+$ && "$NODE" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail 'namespace or node is unsafe'
[[ "$SELECTOR" =~ ^[A-Za-z0-9._,/=-]+$ ]] || fail 'pod selector is unsafe'
[[ -x "$ROOT_BUILDER" && -f "$ROOT_BUILDER" && ! -L "$ROOT_BUILDER" ]] ||
  fail 'V4 root builder is unavailable'
[[ -x "$HOST_GATE" && -f "$HOST_GATE" && ! -L "$HOST_GATE" ]] ||
  fail 'V4 host root gate is unavailable'
for tool in kubectl sha256sum mktemp timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-effect-root-v4-host.XXXXXX")"
CAPSULE="$WORK/capsule"
build_receipt="$(SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_ROOT_V4_OUTPUT="$CAPSULE" \
  "$ROOT_BUILDER")"
[[ "$build_receipt" == BUILT_LOOM_PROCESS_WITNESS_EFFECT_ROOT_V4\ * ]] ||
  fail 'V4 root builder receipt diverged'
CELL_SHA256="$(field "$build_receipt" cell_sha256)"
TREE_SHA256="$(field "$build_receipt" tree_sha256)"
[[ "$CELL_SHA256" =~ ^[0-9a-f]{64}$ && "$TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'V4 root builder hashes are malformed'

mapfile -t candidate_pods < <(
  kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" \
    --field-selector "spec.nodeName=$NODE,status.phase=Running" -o name
)
[[ ${#candidate_pods[@]} -eq 1 ]] ||
  fail "expected one host transport pod on $NODE; found ${#candidate_pods[@]}"
POD="${candidate_pods[0]#pod/}"
[[ "$POD" =~ ^[a-z0-9.-]+$ ]] || fail 'selected pod name is unsafe'
pod_boundary="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.hostPID}|{.spec.containers[0].securityContext.privileged}|{.status.phase}')"
[[ "$pod_boundary" == 'true|true|Running' ]] ||
  fail "transport pod lacks privileged hostPID boundary: $pod_boundary"

nonce="$$-$(date +%s%N)"
REMOTE_ROOT="/var/tmp/loom-effect-root-v4-${TREE_SHA256:0:16}-$nonce"
REMOTE_GATE="/var/tmp/loom-effect-root-v4-gate-${CELL_SHA256:0:16}-$nonce"
cleanup() {
  if [[ -n "${POD:-}" ]]; then
    kubectl -n "$NAMESPACE" exec "$POD" -- \
      nsenter -t 1 -m -u -i -n -p -- umount "$REMOTE_ROOT" \
      >/dev/null 2>&1 || true
    kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
      "chmod -R u+w '/proc/1/root$REMOTE_ROOT' 2>/dev/null || true; rm -rf '/proc/1/root$REMOTE_ROOT'; rm -f '/proc/1/root$REMOTE_GATE'" \
      >/dev/null 2>&1 || true
  fi
  chmod -R u+w "$WORK" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "umask 077; install -d -m 0755 '/proc/1/root$REMOTE_ROOT' '/proc/1/root$REMOTE_ROOT/loom' '/proc/1/root$REMOTE_ROOT/dev' '/proc/1/root$REMOTE_ROOT/proc' '/proc/1/root$REMOTE_ROOT/tmp' '/proc/1/root$REMOTE_ROOT/run' '/proc/1/root$REMOTE_ROOT/run/systemd' '/proc/1/root$REMOTE_ROOT/run/systemd/incoming'"
transfer_file() {
  local source="$1" destination="$2" mode="$3"
  kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
    "umask 077; cat > '/proc/1/root$destination'; chmod '$mode' '/proc/1/root$destination'; chown 0:0 '/proc/1/root$destination'" \
    < "$source"
}
transfer_file "$CAPSULE/loom/effect-cell" "$REMOTE_ROOT/loom/effect-cell" 0555
transfer_file "$CAPSULE/loom/payload" "$REMOTE_ROOT/loom/payload" 0555
transfer_file "$CAPSULE/loom/payload.freeze.v1" \
  "$REMOTE_ROOT/loom/payload.freeze.v1" 0444
transfer_file "$CAPSULE/loom/effect-policy-v4.freeze.v1" \
  "$REMOTE_ROOT/loom/effect-policy-v4.freeze.v1" 0444
transfer_file "$HOST_GATE" "$REMOTE_GATE" 0500
kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "mknod '/proc/1/root$REMOTE_ROOT/dev/null' c 1 3; chown 0:0 '/proc/1/root$REMOTE_ROOT/dev/null'; chmod 0666 '/proc/1/root$REMOTE_ROOT/dev/null'; chown -R 0:0 '/proc/1/root$REMOTE_ROOT'; chmod 0555 '/proc/1/root$REMOTE_ROOT' '/proc/1/root$REMOTE_ROOT/loom' '/proc/1/root$REMOTE_ROOT/dev' '/proc/1/root$REMOTE_ROOT/proc' '/proc/1/root$REMOTE_ROOT/tmp' '/proc/1/root$REMOTE_ROOT/run' '/proc/1/root$REMOTE_ROOT/run/systemd' '/proc/1/root$REMOTE_ROOT/run/systemd/incoming'"

remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "sha256sum '/proc/1/root$REMOTE_ROOT/loom/effect-cell' '/proc/1/root$REMOTE_ROOT/loom/payload' '/proc/1/root$REMOTE_ROOT/loom/payload.freeze.v1' '/proc/1/root$REMOTE_ROOT/loom/effect-policy-v4.freeze.v1' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$CELL_SHA256" &&
   "${remote_hash_lines[1]:-}" == 7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d &&
   "${remote_hash_lines[2]:-}" == 624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da &&
   "${remote_hash_lines[3]:-}" == 60cff91db90e9214e62a6fa5b45521249e31649c63dce297683ca477fcd3d627 ]] ||
  fail 'host root transport hash drifted'

set +e
host_output="$(timeout --signal=TERM --kill-after=5s 120s \
  kubectl -n "$NAMESPACE" exec "$POD" -- \
  nsenter -t 1 -m -u -i -n -p -- \
  /bin/bash "$REMOTE_GATE" --root "$REMOTE_ROOT" \
    --cell-sha256 "$CELL_SHA256" --tree-sha256 "$TREE_SHA256" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] ||
  fail "host immutable-root gate failed status=$host_status output=$host_output"
[[ "$host_output" == sounio-loom-process-witness-effect-root-v4-host-gate:\ HOST_MEASUREMENT_PASS* ]] ||
  fail 'host immutable-root receipt shape diverged'
[[ "$host_output" == *'root_owned=true root_read_only=true root_exact=true dynamic_user=true'* &&
   "$host_output" == *'proc_treatment=absent tmp_read_only=true fd_inventory=0+1+2 capabilities=zero no_new_privileges=true seccomp=true process_extinction=observed'* ]] ||
  fail 'host immutable-root measurement omitted its kernel facts'
[[ "$host_output" == *'root_treatment=true bootstrap_sabotage=true bootstrap_missing_status=226/NAMESPACE material_sabotages=0 material_coverage=false complete_effects=false material_execution=false'* ]] ||
  fail 'host immutable-root probe promoted beyond evidence'

printf '%s\n' "$host_output"
printf 'LOOM_PROCESS_WITNESS_EFFECT_ROOT_V4_HOST_TRANSPORT PASS namespace=%s node=%s pod=%s transport=kubectl+hostPID+nsenter tree_sha256=%s cell_sha256=%s host_output_sha256=%s root_treatment=true bootstrap_sabotage=true bootstrap_missing_status=226/NAMESPACE material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false parity_open=false claim_ready=false\n' \
  "$NAMESPACE" "$NODE" "$POD" "$TREE_SHA256" "$CELL_SHA256" \
  "$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)"
