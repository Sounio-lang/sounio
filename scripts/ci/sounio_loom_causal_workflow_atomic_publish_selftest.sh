#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BROKER="$ROOT_DIR/tools/loom/_build/default/src/loom-kernel-principal-broker"

fail() {
  printf 'sounio-loom-causal-workflow-atomic-publish-selftest: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for tool in c++ mktemp sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || fail "required tool absent: $tool"
done

bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-causal-atomic-publish.XXXXXX")"
cleanup() {
  chmod -R u+rwX "$WORK" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

printf 'prior-authority-must-survive\n' > "$WORK/sentinel.record"
chmod 0400 "$WORK/sentinel.record"
BEFORE="$(sha256sum "$WORK/sentinel.record" | cut -d ' ' -f 1)"

OUTPUT="$($BROKER --selftest-causal-atomic-publish \
  --causal-material-store "$WORK")"

AFTER="$(sha256sum "$WORK/sentinel.record" | cut -d ' ' -f 1)"
[[ "$BEFORE" == "$AFTER" ]] || fail 'pre-existing authority changed'
[[ "$OUTPUT" == 'LOOM_CAUSAL_ATOMIC_PUBLISH PASS '* ]] ||
  fail "broker output malformed: $OUTPUT"
[[ "$OUTPUT" == *'existing_publish=REFUSED'* &&
   "$OUTPUT" == *'prior_authority_preserved=true'* &&
   "$OUTPUT" == *'fresh_publish=PASS'* ]] ||
  fail "required assertions absent: $OUTPUT"

printf 'sounio-loom-causal-workflow-atomic-publish-selftest: PASS %s\n' \
  "${OUTPUT#LOOM_CAUSAL_ATOMIC_PUBLISH PASS }"
