#!/usr/bin/env bash
# Native Arrow IPC gate. Default compiler is Madaros via ./bin/souc.
# The Sounio test round-trips file and stream IPC, including a pyarrow 26
# fixture embedded as hex. This script then checks the bytes on disk and,
# when pyarrow imports, compares the logical columns.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export SOUNIO_STDLIB_PATH="$PWD/stdlib"
SOUC="${SOUC:-./bin/souc}"

FILE_OUT="/tmp/sounio_arrow_ipc.arrow"
STREAM_OUT="/tmp/sounio_arrow_ipc.arrows"
MULTI_FILE="/tmp/sounio_arrow_ipc_multi.arrow"
MULTI_STREAM="/tmp/sounio_arrow_ipc_multi.arrows"
rm -f "$FILE_OUT" "$STREAM_OUT" "$MULTI_FILE" "$MULTI_STREAM"

echo "== arrow ipc (madaros) =="
"$SOUC" run tests/stdlib/arrow/test_arrow_ipc.sio

test -f "$FILE_OUT"
test -f "$STREAM_OUT"
test -f "$MULTI_FILE"
test -f "$MULTI_STREAM"

# Prefer an already-importable pyarrow. This VM keeps a newer numpy+pyarrow
# under /tmp/pyarrow-pkgs because the system numpy is too old for pyarrow 26.
# Set the path before Python starts so numpy is not imported from the system.
if ! python3 -c 'import pyarrow' >/dev/null 2>&1; then
  if [ -d /tmp/pyarrow-pkgs ]; then
    export PYTHONPATH="/tmp/pyarrow-pkgs${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi

python3 scripts/arrow_ipc_check.py "$FILE_OUT" "$STREAM_OUT" "$MULTI_FILE" "$MULTI_STREAM"
echo "ARROW_IPC_GATE_OK"
