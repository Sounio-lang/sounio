#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_CAUSAL_MID_EXEC_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_CAUSAL_MID_EXEC_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_CAUSAL_MID_EXEC_MODULE:-$ROOT_DIR/stdlib/coordination/loom_causal_workflow_mid_exec_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_CAUSAL_MID_EXEC_MAIN:-$ROOT_DIR/tools/loom/causal_workflow_mid_exec_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_CAUSAL_MID_EXEC_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-causal-workflow-mid-exec}"

fail() {
  printf 'build-sounio-loom-causal-workflow-mid-exec: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'Sounio subordinate authority is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'Sounio subordinate entrypoint is absent or linked'
work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-mid-exec.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/mid_exec.sio"
compiled="$work/sounio-loom-causal-workflow-mid-exec"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"
set +e
probe="$(printf '0\n' | "$OUTPUT")"
probe_code=$?
set -e
[[ $probe_code -eq 0 ]] || fail "Sounio selftest exited $probe_code: $probe"
[[ "$probe" == 'SOUNIO_CAUSAL_WORKFLOW_MID_EXEC_SELFTEST PASS cases=11 action=9037 subordinate_contract=mid-exec-v1' ]] ||
  fail "Sounio selftest diverged: $probe"
printf 'BUILT_CAUSAL_WORKFLOW_MID_EXEC path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9037 subordinate_contract=mid-exec-v1 engine=%s cases=11\n' \
  "$OUTPUT" "$ENGINE"
