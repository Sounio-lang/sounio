#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
ROOT_BUILDER="$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_hypercube_root_v11.sh"
PLAN_BUILDER="$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v11.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_process_witness_effect_hypercube_v11_host_gate.sh"
NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'

fail() {
  printf 'run-loom-process-witness-effect-hypercube-v11-host-probe: REFUSE reason=%s material_hypercube=false material_coverage=false complete_effects=false material_execution=false action_9025_judged=false claim_ready=false\n' "$*" >&2
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
    --namespace) [[ $# -ge 2 ]] || usage; NAMESPACE="$2"; shift 2 ;;
    --node) [[ $# -ge 2 ]] || usage; NODE="$2"; shift 2 ;;
    --selector) [[ $# -ge 2 ]] || usage; SELECTOR="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$NAMESPACE" =~ ^[a-z0-9.-]+$ && "$NODE" =~ ^[A-Za-z0-9._-]+$ &&
   "$SELECTOR" =~ ^[A-Za-z0-9._,/=-]+$ ]] || fail 'transport selector is unsafe'
for path in "$ROOT_BUILDER" "$PLAN_BUILDER" "$HOST_GATE"; do
  [[ -x "$path" && -f "$path" && ! -L "$path" ]] ||
    fail "required V11 component is unavailable: $path"
done
for tool in kubectl sha256sum mktemp timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-effect-hypercube-v11-host.XXXXXX")"
CAPSULE="$WORK/capsule"
PLAN="$WORK/sounio-plan"
BUNDLE="$WORK/sounio-bundle"
build_receipt="$(SOUNIO_LOOM_EFFECT_HYPERCUBE_ROOT_V11_OUTPUT="$CAPSULE" \
  "$ROOT_BUILDER")"
[[ "$build_receipt" == BUILT_LOOM_PROCESS_WITNESS_EFFECT_HYPERCUBE_ROOT_V11\ * ]] ||
  fail 'V11 root builder receipt diverged'
CELL_SHA256="$(field "$build_receipt" cell_sha256)"
TREE_SHA256="$(field "$build_receipt" tree_sha256)"
[[ "$CELL_SHA256" =~ ^[0-9a-f]{64}$ && "$TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'V11 root builder hashes are malformed'
SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V11_OUTPUT="$PLAN" "$PLAN_BUILDER" >/dev/null
"$PLAN" > "$BUNDLE"
BUNDLE_SHA256="$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
[[ "$BUNDLE_SHA256" == 876dce5e9445a5c29236689699719e53ebf79930afae75f8ad5ff21544664394 &&
   "$(grep -c '^VERTEX ' "$BUNDLE" || true)" == 40 ]] ||
  fail 'source-fresh Sounio V11 expected bundle drifted'

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
REMOTE_ROOT="/var/tmp/loom-effect-hypercube-v11-${TREE_SHA256:0:16}-$nonce"
REMOTE_GATE="/var/tmp/loom-effect-hypercube-v11-gate-${CELL_SHA256:0:16}-$nonce"
REMOTE_BUNDLE="/var/tmp/loom-effect-hypercube-v11-bundle-${BUNDLE_SHA256:0:16}-$nonce"
cleanup() {
  if [[ -n "${POD:-}" ]]; then
    kubectl -n "$NAMESPACE" exec "$POD" -- \
      nsenter -t 1 -m -u -i -n -p -- umount "$REMOTE_ROOT" \
      >/dev/null 2>&1 || true
    kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
      "chmod -R u+w '/proc/1/root$REMOTE_ROOT' 2>/dev/null || true; rm -rf '/proc/1/root$REMOTE_ROOT'; rm -f '/proc/1/root$REMOTE_GATE' '/proc/1/root$REMOTE_BUNDLE'" \
      >/dev/null 2>&1 || true
  fi
  chmod -R u+w "$WORK" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "umask 077; install -d -m 0755 '/proc/1/root$REMOTE_ROOT' '/proc/1/root$REMOTE_ROOT/loom' '/proc/1/root$REMOTE_ROOT/dev' '/proc/1/root$REMOTE_ROOT/proc' '/proc/1/root$REMOTE_ROOT/tmp' '/proc/1/root$REMOTE_ROOT/run' '/proc/1/root$REMOTE_ROOT/run/systemd' '/proc/1/root$REMOTE_ROOT/run/systemd/incoming' '/proc/1/root$REMOTE_ROOT/sys' '/proc/1/root$REMOTE_ROOT/var' '/proc/1/root$REMOTE_ROOT/var/tmp'"
transfer_file() {
  local source="$1" destination="$2" mode="$3"
  kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
    "umask 077; cat > '/proc/1/root$destination'; chmod '$mode' '/proc/1/root$destination'; chown 0:0 '/proc/1/root$destination'" \
    < "$source"
}
transfer_file "$CAPSULE/loom/effect-cell" "$REMOTE_ROOT/loom/effect-cell" 0555
transfer_file "$CAPSULE/loom/effect-policy-v11.freeze.v1" \
  "$REMOTE_ROOT/loom/effect-policy-v11.freeze.v1" 0444
transfer_file "$HOST_GATE" "$REMOTE_GATE" 0500
transfer_file "$BUNDLE" "$REMOTE_BUNDLE" 0400
kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "mknod '/proc/1/root$REMOTE_ROOT/dev/null' c 1 3; chown 0:0 '/proc/1/root$REMOTE_ROOT/dev/null'; chmod 0666 '/proc/1/root$REMOTE_ROOT/dev/null'; chown -R 0:0 '/proc/1/root$REMOTE_ROOT'; chmod 0555 '/proc/1/root$REMOTE_ROOT' '/proc/1/root$REMOTE_ROOT/loom' '/proc/1/root$REMOTE_ROOT/dev' '/proc/1/root$REMOTE_ROOT/proc' '/proc/1/root$REMOTE_ROOT/tmp' '/proc/1/root$REMOTE_ROOT/run' '/proc/1/root$REMOTE_ROOT/run/systemd' '/proc/1/root$REMOTE_ROOT/run/systemd/incoming' '/proc/1/root$REMOTE_ROOT/sys' '/proc/1/root$REMOTE_ROOT/var' '/proc/1/root$REMOTE_ROOT/var/tmp'"

remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "sha256sum '/proc/1/root$REMOTE_ROOT/loom/effect-cell' '/proc/1/root$REMOTE_ROOT/loom/effect-policy-v11.freeze.v1' '/proc/1/root$REMOTE_GATE' '/proc/1/root$REMOTE_BUNDLE' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$CELL_SHA256" &&
   "${remote_hash_lines[1]:-}" == adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c &&
   "${remote_hash_lines[2]:-}" == "$(sha256sum "$HOST_GATE" | cut -d ' ' -f 1)" &&
   "${remote_hash_lines[3]:-}" == "$BUNDLE_SHA256" ]] ||
  fail 'host transport hash drifted'

set +e
host_output="$(timeout --signal=TERM --kill-after=10s 300s \
  kubectl -n "$NAMESPACE" exec "$POD" -- \
  nsenter -t 1 -m -u -i -n -p -- \
  /bin/bash "$REMOTE_GATE" --root "$REMOTE_ROOT" \
    --cell-sha256 "$CELL_SHA256" --tree-sha256 "$TREE_SHA256" \
    --bundle "$REMOTE_BUNDLE" --bundle-sha256 "$BUNDLE_SHA256" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] ||
  fail "host hypercube gate failed status=$host_status output=$host_output"
summary="$(printf '%s\n' "$host_output" | grep '^sounio-loom-process-witness-effect-hypercube-v11-host-gate: HOST_MEASUREMENT_PASS ' || true)"
[[ -n "$summary" && "$(printf '%s\n' "$summary" | wc -l)" == 1 ]] ||
  fail 'host hypercube summary is absent or duplicated'
[[ "$(printf '%s\n' "$host_output" | grep -c '^LOOM_EFFECT_VERTEX_V11 OBSERVED ' || true)" == 40 ]] ||
  fail 'host hypercube receipt count diverged'
[[ "$summary" == *'families=12 probes=13 mechanism_dimensions=18 vertices=40 refusals=25 completions=15 extinctions=15 mincuts_expected=13 crossed_named_rule=0 experiment_unavailable=0 invariant_stable=true delta_distinct=true triple_hash_binding=true'* &&
   "$summary" == *'vfs_read_only_toggled=true private_network_toggled=true unix_endpoint_absence_toggled=true lock_personality_toggled=true proc_treatment_toggled=CAPSULE_EMPTY_BIND+LIVE_PROCFS endpoint_extinction=true process_extinction=true scratch_extinction=true material_hypercube=true material_coverage=false'* ]] ||
  fail 'host hypercube measurement promoted or omitted its causal facts'

printf '%s\n' "$host_output"
printf 'LOOM_PROCESS_WITNESS_EFFECT_HYPERCUBE_V11_HOST_TRANSPORT PASS namespace=%s node=%s pod=%s transport=kubectl+hostPID+nsenter policy_manifest_sha256=adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c expected_bundle_sha256=%s tree_sha256=%s cell_sha256=%s host_output_sha256=%s families=12 probes=13 mechanism_dimensions=18 vertices=40 refusals=25 completions=15 extinctions=15 mincuts_expected=13 crossed_named_rule=0 experiment_unavailable=0 invariant_stable=true delta_distinct=true triple_hash_binding=true material_hypercube=true material_coverage=false complete_effects=false material_execution=false action_9025_judged=false claim_ready=false\n' \
  "$NAMESPACE" "$NODE" "$POD" "$BUNDLE_SHA256" "$TREE_SHA256" \
  "$CELL_SHA256" "$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)"
