#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALLER="$ROOT_DIR/scripts/dev/install_sounio_loom_kubernetes_hook.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-k8s-hook.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-kubernetes-hook-selftest: FAIL: %s\n' "$1" >&2
  exit 1
}

cat > "$TEST_ROOT/kubectl" <<'FAKE_KUBECTL'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$SOUNIO_FAKE_KUBECTL_LOG"
if [[ "$*" == *'get statefulset'* ]]; then
  printf '{"spec":{"updateStrategy":{"type":"%s"},"template":{"spec":{"containers":[{"name":"workspace-ssh"}]}}}}\n' \
    "${SOUNIO_FAKE_KUBECTL_STRATEGY:-OnDelete}"
elif [[ "$*" == *'get pod'* ]]; then
  printf 'stable-pod-uid'
else
  printf 'error: fake kubectl received an unexpected command: %s\n' "$*" >&2
  exit 2
fi
FAKE_KUBECTL
chmod +x "$TEST_ROOT/kubectl"

export SOUNIO_FAKE_KUBECTL_LOG="$TEST_ROOT/kubectl.log"
PATH="$TEST_ROOT:$PATH" "$INSTALLER" --dry-run > "$TEST_ROOT/dry-run.out"
grep -q 'LOOM_KUBERNETES_HOOK mode=dry-run strategy=OnDelete' \
  "$TEST_ROOT/dry-run.out" || fail 'dry-run receipt is absent'
grep -q 'runuser' "$TEST_ROOT/dry-run.out" || \
  fail 'generated hook does not bind the lane owner identity'
grep -q 'fleet-reconcile' "$TEST_ROOT/dry-run.out" || \
  fail 'generated hook does not invoke fleet reconciliation'
if grep -q 'patch statefulset' "$SOUNIO_FAKE_KUBECTL_LOG"; then
  fail 'dry-run mutated the StatefulSet'
fi

if SOUNIO_FAKE_KUBECTL_STRATEGY=RollingUpdate PATH="$TEST_ROOT:$PATH" \
  "$INSTALLER" --dry-run > "$TEST_ROOT/rolling.out" 2>&1; then
  fail 'installer accepted a rolling update strategy'
fi
grep -q 'not OnDelete' "$TEST_ROOT/rolling.out" || \
  fail 'rolling update strategy was refused for the wrong reason'

echo 'sounio-loom-kubernetes-hook-selftest: PASS dry_run=no-mutation rolling_update=refused identity=user-bound'
