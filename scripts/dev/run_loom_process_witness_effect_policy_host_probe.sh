#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_policy.sh"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v2.freeze.v1"
NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'

fail() {
  printf 'run-loom-process-witness-effect-policy-host-probe: REFUSE reason=%s material_coverage=false complete_effects=false material_execution=false launch_open=false\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s [--namespace NAME] [--node NAME] [--selector LABELS]\n' "$0" >&2
  exit 64
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
[[ -x "$BUILDER" && -f "$BUILDER" && ! -L "$BUILDER" ]] ||
  fail 'native policy builder is unavailable'
[[ -f "$POLICY_MANIFEST" && ! -L "$POLICY_MANIFEST" ]] ||
  fail 'frozen Sounio V2 policy manifest is unavailable'
for tool in kubectl sha256sum mktemp timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-effect-policy-host.XXXXXX")"
BINARY="$WORK/loom-process-witness-effect-policy"
SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_OUTPUT="$BINARY" \
  "$BUILDER" >/dev/null
BINARY_SHA256="$(sha256sum "$BINARY" | cut -d ' ' -f 1)"
MANIFEST_SHA256="$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)"
[[ "$MANIFEST_SHA256" == d66b13252479252d5922ee0091e51a5bdb6a5eca9a592bb21f5db9dde344fee9 ]] ||
  fail 'Sounio V2 policy manifest drifted before transport'

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

REMOTE_BINARY="/var/tmp/loom-effect-policy-${BINARY_SHA256:0:24}"
REMOTE_MANIFEST="/var/tmp/loom-effect-policy-v2-${MANIFEST_SHA256:0:24}.freeze"
cleanup() {
  if [[ -n "${POD:-}" ]]; then
    kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
      "rm -f '/proc/1/root$REMOTE_BINARY' '/proc/1/root$REMOTE_MANIFEST'" \
      >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
  "umask 077; cat > '/proc/1/root$REMOTE_BINARY'; chmod 0500 '/proc/1/root$REMOTE_BINARY'" \
  < "$BINARY"
kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
  "umask 077; cat > '/proc/1/root$REMOTE_MANIFEST'; chmod 0400 '/proc/1/root$REMOTE_MANIFEST'" \
  < "$POLICY_MANIFEST"

remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "sha256sum '/proc/1/root$REMOTE_BINARY' '/proc/1/root$REMOTE_MANIFEST' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$BINARY_SHA256" &&
   "${remote_hash_lines[1]:-}" == "$MANIFEST_SHA256" ]] ||
  fail 'host transport hash drifted'

set +e
host_output="$(timeout --signal=TERM --kill-after=5s 90s \
  kubectl -n "$NAMESPACE" exec "$POD" -- \
  nsenter -t 1 -m -u -i -n -p -- \
  "$REMOTE_BINARY" --selftest --policy-manifest "$REMOTE_MANIFEST" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] ||
  fail "host kernel probe failed status=$host_status output=$host_output"
[[ "$host_output" == LOOM_PROCESS_WITNESS_EFFECT_POLICY_SELFTEST\ PASS* ]] ||
  fail 'host policy receipt shape diverged'
[[ "$host_output" == *'landlock_local=available'* &&
   "$host_output" == *'seccomp_treatments=12 local_landlock_treatments=12 structural_sabotages=12 material_sabotages=0'* ]] ||
  fail "host kernel did not realize the frozen Landlock/seccomp treatment: $host_output"
[[ "$host_output" == *'host_gate_required=true material_coverage=false complete_effects=false material_execution=false'* ]] ||
  fail 'host policy probe promoted beyond evidence'

printf '%s\n' "$host_output"
printf 'LOOM_PROCESS_WITNESS_EFFECT_POLICY_HOST_TRANSPORT PASS namespace=%s node=%s pod=%s binary_sha256=%s policy_manifest_sha256=%s transport=kubectl+hostPID+nsenter host_output_sha256=%s material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false parity_open=false claim_ready=false\n' \
  "$NAMESPACE" "$NODE" "$POD" "$BINARY_SHA256" "$MANIFEST_SHA256" \
  "$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)"
