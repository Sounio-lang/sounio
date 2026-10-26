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

if [[ "${ARROW_REQUIRE_PYARROW:-0}" == 1 ]]; then
  python3 -c 'import pyarrow, pyarrow.ipc, pyarrow.flight; print("Required pyarrow:", pyarrow.__version__)'
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

# Pyarrow comes from the caller's environment. Set ARROW_PYARROW_PATH only
# when that directory is one the caller trusts; a shared temp tree is not.
if [[ -n "${ARROW_PYARROW_PATH:-}" ]]; then
  export PYTHONPATH="${ARROW_PYARROW_PATH}${PYTHONPATH:+:$PYTHONPATH}"
fi

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
