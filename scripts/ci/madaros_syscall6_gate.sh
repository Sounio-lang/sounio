#!/usr/bin/env bash
# scripts/ci/madaros_syscall6_gate.sh
#
# Gate: the syscall6(nr, a1..a6) raw Linux syscall builtin works end-to-end
# under default Madaros (checker arm call_expr_is_syscall6 in
# self-hosted/check/check.sio, raw-arg lowering lower_expr_args_raw_ref in
# self-hosted/ir/lower.sio, emit_builtin_syscall6_into id 27 in
# self-hosted/native/codegen_x86_linux.sio).
#
# Coverage:
#   1. getpid(39) returns a plausible pid (scalar-only args + return value).
#   2. open/write/close + read_file round-trip (tests/run-pass/test_syscall_ffi.sio).
#   3. BSS global buffer as pointer arg in BOTH forms (&!GB borrow and bare
#      GB): write emits buffer contents, read lands stdin bytes in the buffer
#      (tests/run-pass/syscall6_global_buffer.sio) — the reference-handle
#      regression that broke lsp/server.sio's I/O loop.
#
# Requires a current-source Madaros (artifacts/self-hosted/madaros or
# MADAROS_RAW_BIN). Checked-in bin/madaros-linux-x86_64 may predate syscall6.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true

SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_syscall6_gate =="

check_run() {
  local name="$1" src="$2" stdin_text="$3" marker="$4"
  if ! "$SOUC" compile "$src" -o "$OUT/$name.elf" >"$OUT/$name.build.log" 2>&1; then
    echo "FAIL $name: compile" >&2
    tail -5 "$OUT/$name.build.log" >&2
    fail=1
    return
  fi
  local out
  out="$(printf '%s' "$stdin_text" | "$OUT/$name.elf" 2>&1)" || {
    echo "FAIL $name: run rc=$?" >&2
    echo "$out" >&2
    fail=1
    return
  }
  if ! grep -q "$marker" <<<"$out"; then
    echo "FAIL $name: missing marker $marker" >&2
    echo "$out" >&2
    fail=1
    return
  fi
  echo "ok $name"
}

# 1. getpid witness (generated; not part of the run-pass suite).
cat >"$OUT/getpid.sio" <<'SIO'
fn main() -> i32 with IO {
    let pid = syscall6(39, 0, 0, 0, 0, 0, 0)
    if pid <= 1 {
        print("FAIL pid\n")
        return 1
    }
    print("GETPID_OK\n")
    0
}
SIO
check_run getpid "$OUT/getpid.sio" "" "GETPID_OK"

# 2. open/write/close FFI round-trip.
check_run ffi tests/run-pass/test_syscall_ffi.sio "" "syscall ffi ok"

# 3. Global buffer pointer args (borrow + bare forms, read + write).
check_run global_buf tests/run-pass/syscall6_global_buffer.sio "STDIN" "SYSCALL6_GLOBAL_BUFFER_OK"

# Receipt
mkdir -p "$ROOT/artifacts/compiler"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat > "$ROOT/artifacts/compiler/madaros_syscall6_receipt.v1.json" <<JSON
{
  "schema": "madaros_syscall6_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "commit": "$COMMIT",
  "claims": [
    "syscall6_scalar_args_getpid",
    "syscall6_open_write_close_ffi",
    "syscall6_bss_global_pointer_borrow_and_bare"
  ],
  "claims_not_made": [
    "aarch64 emission",
    "syscall6 for local (frame) array borrows"
  ]
}
JSON

if [[ $fail -ne 0 ]]; then
  echo "MADAROS_SYSCALL6_GATE_FAIL" >&2
  exit 1
fi
echo "MADAROS_SYSCALL6_GATE_OK"
