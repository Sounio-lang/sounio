#!/usr/bin/env bash
# Method calls resolve through a reference receiver.
#
# `impl S { fn get(self: &S) }` declares the borrow on self, so a caller holding
# `s: &S` writing `s.get()` is the ordinary shape for code that does not own its
# argument. It used to report E019 -- "method calls are not supported for this
# type" -- about a struct that has an impl block. On the committed compiler that
# single defect accounted for 47 of the 84 stdlib files reporting E019, all of
# them downstream of one module.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_method_ref_receiver_gate =="

# tests/run-pass/method_on_deref_field.sio reaches the same defect through
# (*o).field.method() and was red on the committed binary; keep both.
run_fixture() {
  local src="$1" sentinel="$2"
  local elf="$OUT/$(basename "$src" .sio).elf"
  local log="$OUT/$(basename "$src" .sio).log"
  if ! "$SOUC" compile "$src" -o "$elf" >"$log" 2>&1; then
    echo "FAIL: compile $src"
    tail -40 "$log" || true
    exit 1
  fi
  chmod +x "$elf"
  if ! "$elf" >"$log" 2>&1; then
    echo "FAIL: run $src"
    cat "$log" || true
    exit 1
  fi
  grep -q "$sentinel" "$log" || {
    echo "FAIL: $src missing sentinel $sentinel"
    cat "$log" || true
    exit 1
  }
}

run_fixture tests/run-pass/method_call_ref_receiver.sio METHOD_REF_RECEIVER_OK
run_fixture tests/run-pass/method_on_deref_field.sio    via_ref=42

echo "MADAROS_METHOD_REF_RECEIVER_GATE_OK"
