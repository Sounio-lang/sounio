#!/usr/bin/env bash
# Gate: D2 residual #933 — write_file / str_from_bytes accept both handle-by-value
# and &buf under native-v2 (after lower auto-unwrap of OpRef slot addresses).
#
# Requires a current-source Madaros that carries the lower.sio change:
#   SOUNIO_BUILD_LOCK=/tmp/sounio-d2buf-build.lock \
#     bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
#   SOUNIO_TEST_SOUC_BIN=artifacts/self-hosted/madaros \
#     bash scripts/native_d2_ref_buf_builtin_gate.sh
#
# Also re-runs the #1247 handle-by-value write_file gate and the #1258 packed
# string index gate so this PR cannot regress either sibling.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
GOLDEN="tests/data_io_gated/write_file_csv_golden.csv"
fail=0

echo "=== A. str_from_bytes dual shape (handle + &buf) ==="
PROBE_SFB="tests/data_io_gated/str_from_bytes_dual.sio"
if ! "$SOUC" compile "$PROBE_SFB" -o "$OUT/sfb.elf" >/dev/null 2>"$OUT/sfb.cerr"; then
  echo "FAIL compile str_from_bytes_dual.sio"
  tail -20 "$OUT/sfb.cerr" || true
  fail=1
else
  chmod +x "$OUT/sfb.elf"
  set +e
  got_sfb="$(timeout 20 "$OUT/sfb.elf" 2>/dev/null)"
  rc_sfb=$?
  set -e
  if [[ $rc_sfb -ne 0 || "$got_sfb" != "hi|hi!" ]]; then
    echo "FAIL str_from_bytes dual: exit=$rc_sfb stdout='${got_sfb:-<empty>}' want 'hi|hi!'"
    fail=1
  else
    echo "PASS str_from_bytes(buf,2) + str_from_bytes(&buf,3) -> hi|hi!"
  fi
fi

echo "=== B. write_file handle-by-value (must not regress #1247) ==="
if ! SOUNIO_TEST_SOUC_BIN="$SOUC" bash scripts/native_write_file_handle_abi_gate.sh; then
  echo "FAIL write_file handle-by-value gate (#1247 regression)"
  fail=1
else
  echo "PASS write_file handle-by-value recheck"
fi

echo "=== C. write_file(path, &buf, n) residual shape ==="
PROBE_REF="tests/data_io_gated/write_file_ref_buf.sio"
if ! "$SOUC" compile "$PROBE_REF" -o "$OUT/ref.elf" >/dev/null 2>"$OUT/ref.cerr"; then
  echo "FAIL compile write_file_ref_buf.sio"
  tail -20 "$OUT/ref.cerr" || true
  fail=1
else
  chmod +x "$OUT/ref.elf"
  set +e
  got_ref="$( cd "$OUT" && timeout 20 ./ref.elf 2>/dev/null )"
  rc_ref=$?
  set -e
  out_file="$OUT/write_file_ref_buf.out"
  if [[ $rc_ref -ne 0 || "$got_ref" != "WRITE_REF_OK" ]]; then
    echo "FAIL write_file &buf stdout: exit=$rc_ref got '${got_ref:-<crash/empty>}' want 'WRITE_REF_OK'"
    fail=1
  elif [[ ! -f "$out_file" ]]; then
    echo "FAIL missing output file $out_file"
    fail=1
  elif ! cmp -s "$out_file" "$GOLDEN"; then
    echo "FAIL write_file &buf byte-exact: output != golden"
    od -An -tx1c "$GOLDEN" || true
    od -An -tx1c "$out_file" || true
    fail=1
  else
    echo "PASS write_file(path, &buf, 12) -> byte-exact golden CSV"
  fi
fi

echo "=== D. packed-string s[i] must not regress (#1258) ==="
if [[ -x scripts/native_string_index_packed_gate.sh ]]; then
  if ! SOUNIO_TEST_SOUC_BIN="$SOUC" bash scripts/native_string_index_packed_gate.sh; then
    echo "FAIL packed-string index gate (#1258 regression)"
    fail=1
  else
    echo "PASS packed-string index recheck"
  fi
else
  echo "SKIP packed-string gate (script missing)"
fi

if [[ "$fail" = 0 ]]; then
  echo "NATIVE_D2_REF_BUF_BUILTIN_GATE_OK"
else
  echo "GATE FAILED"
  exit 1
fi
