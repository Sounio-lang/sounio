#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_host_promotion_capsule.sh"
PROMOTER="$ROOT_DIR/scripts/dev/promote_loom_host_capsule.sh"

fail() {
  printf 'run-loom-host-promotion: REFUSE reason=%s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --mode preflight|promote [--capsule ABSOLUTE_PATH] [--expected-sha256 HEX] [--namespace NAME] [--node NAME] [--selector LABELS] [--receipt-output PATH]\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

MODE=''
CAPSULE=''
EXPECTED_SHA256=''
NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'
RECEIPT_OUTPUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || usage
      MODE="$2"
      shift 2
      ;;
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

[[ "$MODE" == preflight || "$MODE" == promote ]] || usage
[[ "$NAMESPACE" =~ ^[a-z0-9.-]+$ && "$NODE" =~ ^[A-Za-z0-9._-]+$ ]] || fail 'namespace or node name is unsafe'
[[ "$SELECTOR" =~ ^[A-Za-z0-9._,/=-]+$ ]] || fail 'pod selector is unsafe'
[[ -x "$BUILDER" && -x "$PROMOTER" ]] || fail 'capsule builder or promoter is unavailable'
for tool in kubectl sha256sum mktemp install; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is missing: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-promotion-run.XXXXXX")"
REMOTE_ARCHIVE=''
REMOTE_PROMOTER=''
POD=''
cleanup() {
  if [[ -n "$POD" && -n "$REMOTE_ARCHIVE" && -n "$REMOTE_PROMOTER" ]]; then
    kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
      "rm -f '/proc/1/root$REMOTE_ARCHIVE' '/proc/1/root$REMOTE_PROMOTER'" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

if [[ -z "$CAPSULE" ]]; then
  CAPSULE="$WORK/loom-host-promotion.tar"
  build_output="$($BUILDER --output "$CAPSULE")"
  [[ "$build_output" == 'LOOM_HOST_PROMOTION_CAPSULE_BUILD PASS '* ]] || fail "capsule build failed: $build_output"
else
  [[ "$CAPSULE" == /* && -f "$CAPSULE" && ! -L "$CAPSULE" ]] || fail 'provided capsule is absent, linked, or non-absolute'
fi

ACTUAL_SHA256="$(sha256_file "$CAPSULE")"
if [[ -z "$EXPECTED_SHA256" ]]; then
  EXPECTED_SHA256="$ACTUAL_SHA256"
fi
[[ "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail 'expected capsule hash is not canonical SHA-256'
[[ "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]] || fail 'local capsule differs from the expected transport hash'
PROMOTER_SHA256="$(sha256_file "$PROMOTER")"

mapfile -t candidate_pods < <(
  kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" \
    --field-selector "spec.nodeName=$NODE,status.phase=Running" -o name
)
[[ ${#candidate_pods[@]} -eq 1 ]] ||
  fail "expected exactly one running host transport pod on $NODE; found ${#candidate_pods[@]}"
POD="${candidate_pods[0]#pod/}"
[[ -n "$POD" && "$POD" =~ ^[a-z0-9.-]+$ ]] || fail 'selected transport pod name is unsafe'

pod_boundary="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{.spec.hostPID}|{.spec.containers[0].securityContext.privileged}|{.status.phase}')"
[[ "$pod_boundary" == 'true|true|Running' ]] || fail "transport pod lacks the privileged hostPID boundary: $pod_boundary"
kubectl -n "$NAMESPACE" exec "$POD" -- sh -lc \
  'command -v nsenter >/dev/null && command -v sha256sum >/dev/null && test "$(id -u)" = 0 && test "$(nsenter -t 1 -m -u -i -n -p -- tr -d "\n" </proc/1/comm)" = systemd' ||
  fail 'transport pod cannot reach the host systemd namespace as root'

REMOTE_ARCHIVE="/var/tmp/sounio-loom-host-promotion-${EXPECTED_SHA256:0:24}.tar"
REMOTE_PROMOTER="/var/tmp/sounio-loom-host-promoter-${PROMOTER_SHA256:0:24}.sh"

kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
  "umask 077; cat > '/proc/1/root$REMOTE_ARCHIVE'" < "$CAPSULE"
kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
  "umask 077; cat > '/proc/1/root$REMOTE_PROMOTER'; chmod 0500 '/proc/1/root$REMOTE_PROMOTER'" < "$PROMOTER"

remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "sha256sum '/proc/1/root$REMOTE_ARCHIVE' '/proc/1/root$REMOTE_PROMOTER' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$EXPECTED_SHA256" ]] || fail 'host transport archive hash drifted before namespace entry'
[[ "${remote_hash_lines[1]:-}" == "$PROMOTER_SHA256" ]] || fail 'host transport promoter hash drifted before namespace entry'

host_output="$(kubectl -n "$NAMESPACE" exec "$POD" -- nsenter -t 1 -m -u -i -n -p -- \
  "$REMOTE_PROMOTER" --archive "$REMOTE_ARCHIVE" --expected-sha256 "$EXPECTED_SHA256" --mode "$MODE")"
if [[ "$MODE" == preflight ]]; then
  [[ "$host_output" == 'LOOM_HOST_PROMOTION_PREFLIGHT PASS '* ]] || fail "host preflight did not pass: $host_output"
else
  [[ "$host_output" == *'LOOM_HOST_PROMOTION PASS '* ]] || fail "host promotion did not pass: $host_output"
fi

receipt="LOOM_HOST_PROMOTION_TRANSPORT PASS mode=$MODE namespace=$NAMESPACE node=$NODE pod=$POD archive_sha256=$EXPECTED_SHA256 promoter_sha256=$PROMOTER_SHA256 transport=kubectl+hostPID+nsenter host_output_sha256=$(printf '%s' "$host_output" | sha256sum | cut -d ' ' -f 1)"
if [[ -n "$RECEIPT_OUTPUT" ]]; then
  receipt_stage="$(mktemp "$(dirname "$RECEIPT_OUTPUT")/.loom-host-receipt.XXXXXX")"
  printf '%s\n%s\n' "$receipt" "$host_output" > "$receipt_stage"
  install -m 0644 "$receipt_stage" "$RECEIPT_OUTPUT"
  rm -f "$receipt_stage"
fi
printf '%s\n%s\n' "$host_output" "$receipt"
