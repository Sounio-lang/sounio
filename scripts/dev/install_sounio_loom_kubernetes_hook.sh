#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${SOUNIO_LOOM_K8S_NAMESPACE:-beagle}"
STATEFULSET="${SOUNIO_LOOM_K8S_STATEFULSET:-sounio-workspace-control}"
CONTAINER="${SOUNIO_LOOM_K8S_CONTAINER:-workspace-ssh}"
POD="${SOUNIO_LOOM_K8S_POD:-${STATEFULSET}-0}"
MARKER='loom-2026.08.24.2-user-bound'
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/dev/install_sounio_loom_kubernetes_hook.sh [options]

Options:
  --namespace NAME    Kubernetes namespace (default: beagle)
  --statefulset NAME  StatefulSet to patch (default: sounio-workspace-control)
  --container NAME    Container receiving the hook (default: workspace-ssh)
  --pod NAME          Existing Pod whose identity must not change
  --dry-run           Print the strategic merge patch without applying it
  -h, --help          Show this help
USAGE
}

while (($#)); do
  case "$1" in
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --statefulset) STATEFULSET="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for command_name in kubectl jq; do
  command -v "$command_name" >/dev/null || {
    printf 'error: required command is unavailable: %s\n' "$command_name" >&2
    exit 1
  }
done

statefulset_json="$(kubectl -n "$NAMESPACE" get statefulset "$STATEFULSET" -o json)"
strategy="$(jq -r '.spec.updateStrategy.type // "RollingUpdate"' <<<"$statefulset_json")"
if [[ "$strategy" != 'OnDelete' ]]; then
  printf 'error: refusing Kubernetes hook install: update strategy is %s, not OnDelete\n' \
    "$strategy" >&2
  exit 1
fi

container_count="$(
  jq --arg container "$CONTAINER" \
    '[.spec.template.spec.containers[] | select(.name == $container)] | length' \
    <<<"$statefulset_json"
)"
if [[ "$container_count" != 1 ]]; then
  printf 'error: expected exactly one container named %s, found %s\n' \
    "$CONTAINER" "$container_count" >&2
  exit 1
fi

pod_uid_before="$(
  kubectl -n "$NAMESPACE" get pod "$POD" \
    -o jsonpath='{.metadata.uid}' 2>/dev/null || true
)"

worker="$(cat <<'WORKER'
set -u
umask 077
log=/workspace/.beagle/sounio-loom-reconcile.log
runtime=/workspace/sounio/.git/sounio-coord-runtime/current/bin/sounio-loom-runtime
mkdir -p /workspace/.beagle
attempt=1
while [ "$attempt" -le 60 ]; do
  printf '%s attempt=%s actor_uid=%s event=reconcile-start\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" "$(id -u)" >> "$log"
  if [ -x "$runtime" ] && "$runtime" fleet-reconcile --cwd /workspace/sounio --apply >> "$log" 2>&1; then
    printf '%s attempt=%s actor_uid=%s event=reconcile-pass\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" "$(id -u)" >> "$log"
    exit 0
  fi
  attempt=$((attempt + 1))
  sleep 2
done
printf '%s actor_uid=%s event=reconcile-exhausted attempts=60\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(id -u)" >> "$log"
exit 0
WORKER
)"

command_json="$(
  jq -cn --arg worker "$worker" '[
    "/usr/sbin/runuser", "-u", "openvscode-server", "--",
    "/usr/bin/env",
    "HOME=/workspace/.home/openvscode-server",
    "USER=openvscode-server",
    "LOGNAME=openvscode-server",
    "LANG=C.UTF-8",
    "LC_ALL=C.UTF-8",
    "PATH=/workspace/.home/openvscode-server/.empryo/bin:/workspace/.home/openvscode-server/.local/node-v20.20.2-linux-x64/bin:/workspace/.home/openvscode-server/.local/bin:/usr/local/bin:/usr/bin:/bin",
    "/bin/sh", "-lc", $worker
  ]'
)"
patch_json="$(
  jq -cn --arg container "$CONTAINER" --arg marker "$MARKER" \
    --argjson command "$command_json" '{
      spec: {
        template: {
          metadata: {annotations: {"sounio.dev/loom-post-pod-reconcile": $marker}},
          spec: {containers: [{
            name: $container,
            lifecycle: {postStart: {exec: {command: $command}}}
          }]}
        }
      }
    }'
)"

if ((DRY_RUN)); then
  jq . <<<"$patch_json"
  printf 'LOOM_KUBERNETES_HOOK mode=dry-run strategy=OnDelete namespace=%s statefulset=%s container=%s\n' \
    "$NAMESPACE" "$STATEFULSET" "$CONTAINER"
  exit 0
fi

kubectl -n "$NAMESPACE" patch statefulset "$STATEFULSET" \
  --type=strategic --patch "$patch_json" >/dev/null

updated_json="$(kubectl -n "$NAMESPACE" get statefulset "$STATEFULSET" -o json)"
jq -e --arg container "$CONTAINER" --arg marker "$MARKER" \
  --argjson command "$command_json" '
    .spec.updateStrategy.type == "OnDelete" and
    .spec.template.metadata.annotations["sounio.dev/loom-post-pod-reconcile"] == $marker and
    ([.spec.template.spec.containers[] |
      select(.name == $container and .lifecycle.postStart.exec.command == $command)] |
      length == 1)
  ' <<<"$updated_json" >/dev/null || {
    printf 'error: Kubernetes accepted a template that does not match the requested Loom hook\n' >&2
    exit 1
  }

pod_uid_after="$(
  kubectl -n "$NAMESPACE" get pod "$POD" \
    -o jsonpath='{.metadata.uid}' 2>/dev/null || true
)"
if [[ -n "$pod_uid_before" && "$pod_uid_after" != "$pod_uid_before" ]]; then
  printf 'error: Pod identity changed during an OnDelete hook install: before=%s after=%s\n' \
    "$pod_uid_before" "${pod_uid_after:-absent}" >&2
  exit 1
fi

update_revision="$(jq -r '.status.updateRevision // "pending"' <<<"$updated_json")"
printf 'LOOM_KUBERNETES_HOOK mode=applied strategy=OnDelete namespace=%s statefulset=%s container=%s update_revision=%s pod_uid=%s\n' \
  "$NAMESPACE" "$STATEFULSET" "$CONTAINER" "$update_revision" \
  "${pod_uid_after:-absent}"
