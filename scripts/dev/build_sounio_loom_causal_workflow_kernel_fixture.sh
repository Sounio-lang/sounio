#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_CAUSAL_WORKFLOW_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_CAUSAL_WORKFLOW_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_CAUSAL_WORKFLOW_MODULE:-$ROOT_DIR/stdlib/coordination/loom_causal_workflow_kernel_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_CAUSAL_WORKFLOW_MAIN:-$ROOT_DIR/tools/loom/causal_workflow_kernel_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_CAUSAL_WORKFLOW_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-causal-workflow-kernel}"

fail() {
  printf 'build-sounio-loom-causal-workflow-kernel: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'Sounio authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'Sounio entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-workflow.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/causal_workflow.sio"
compiled="$work/sounio-loom-causal-workflow-kernel"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_CAUSAL_WORKFLOW_SELFTEST PASS cases=12' ]] ||
  fail "Sounio selftest diverged: $probe"
printf 'BUILT_CAUSAL_WORKFLOW path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9037 engine=%s cases=12\n' \
  "$OUTPUT" "$ENGINE"
