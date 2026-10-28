#!/usr/bin/env bash
# Native Arrow IPC gate. Default compiler is Madaros via ./bin/souc.
# Stdlib Deep Science runs this script with MADAROS_RAW_BIN set to the
# compiler built from the pull request. The full suite skips the fixtures
# on lean_single because they carry //@ requires: madaros.
# The Sounio test round-trips file and stream IPC, including a pyarrow 26
# fixture embedded as hex. This script then checks the bytes on disk and,
# when pyarrow imports, compares the logical columns.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export SOUNIO_STDLIB_PATH="$PWD/stdlib"
SOUC="${SOUC:-./bin/souc}"

# Pyarrow comes from the caller's environment. Set ARROW_PYARROW_PATH only
# when that directory is one the caller trusts; a shared temp tree is not.
# The required-import preflight and the later checkers must see the same path.
if [[ -n "${ARROW_PYARROW_PATH:-}" ]]; then
  export PYTHONPATH="${ARROW_PYARROW_PATH}${PYTHONPATH:+:$PYTHONPATH}"
fi

prove_pyarrow_absence() {
  local rc=0
  local out
  set +e
  out=$(env -u PYTHONPATH -u PYTHONHOME ARROW_REQUIRE_PYARROW=1 python3 -S -c '
import importlib.util
import os
from pathlib import Path
os.environ["ARROW_REQUIRE_PYARROW"] = "1"
spec = importlib.util.spec_from_file_location("arrow_ipc_check", "scripts/arrow_ipc_check.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.try_pyarrow(Path("unused-file"), Path("unused-stream"), None)
' 2>&1)
  rc=$?
  set -e
  if [[ "$rc" == 0 || "$out" == *PYARROW_SKIP* || "$out" != *"pyarrow required"* ]]; then
    echo "FAIL pyarrow absence control rc=$rc"
    printf '%s\n' "$out"
    exit 1
  fi
  echo "pyarrow absence control: required import failed closed"
}

if [[ "${ARROW_REQUIRE_PYARROW:-0}" == 1 ]]; then
  ver=$(python3 -c 'import pyarrow, pyarrow.ipc, pyarrow.flight; print(pyarrow.__version__)')
  if [[ "$ver" != "26.0.0" ]]; then
    echo "FAIL required pyarrow version ${ver}"
    exit 1
  fi
  echo "Required pyarrow: ${ver}"
  prove_pyarrow_absence
fi

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

python3 scripts/arrow_ipc_check.py "$FILE_OUT" "$STREAM_OUT" "$MULTI_FILE" "$MULTI_STREAM"

echo "== arrow spec schema / batch / dict / compression =="
"$SOUC" run tests/stdlib/arrow/test_arrow_codec.sio
"$SOUC" run tests/stdlib/arrow/test_arrow_spec.sio
python3 scripts/arrow_spec_check.py

echo "== arrow canonical extensions =="
"$SOUC" run tests/stdlib/arrow/test_arrow_ext.sio
python3 scripts/arrow_ext_check.py

echo "== arrow flight do_put / do_exchange =="
"$SOUC" run tests/stdlib/arrow/test_arrow_flight.sio
python3 scripts/arrow_flight_check.py \
  /tmp/sounio_flight_do_put.bin \
  /tmp/sounio_flight_put_result.bin \
  /tmp/sounio_flight_exchange_client.bin \
  /tmp/sounio_flight_exchange_server.bin
echo "ARROW_IPC_GATE_OK"
