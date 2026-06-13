#!/usr/bin/env bash
# Prove that the promoted Madaros compiler preserves real single-file source
# body semantics when lowering to native ELF.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_LAUNCHER:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-$ROOT_DIR/bin/madaros-linux-x86_64}"
TMP_DIR="$(mktemp -d /tmp/sounio-madaros-source-semantics.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[madaros-source-semantics] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros launcher is not executable: $MADAROS"
[[ -x "$RAW_MADAROS" ]] || fail "Madaros raw binary is not executable: $RAW_MADAROS"

write_case() {
  local name="$1"
  local expected="$2"
  local src="$TMP_DIR/$name.sio"
  cat >"$src"
  printf '%s\n' "$expected" >"$TMP_DIR/$name.expected"
}

write_case literal_body 42 <<'SRC'
fn main() -> i64 {
    42
}
SRC

write_case let_call_if 42 <<'SRC'
fn ten() -> i64 { 10 }
fn main() -> i64 {
    let x: i64 = ten() + 5
    let y = x * 3
    if y == 45 { return y - 3 } else { return 1 }
}
SRC

write_case while_var 42 <<'SRC'
fn main() -> i64 {
    var x: i64 = 0
    var acc: i64 = 0
    while x < 6 {
        acc = acc + 7
        x = x + 1
    }
    return acc
}
SRC

write_case bool_cmp 42 <<'SRC'
fn main() -> i64 {
    let a = 6 * 7
    if a == 42 && 2 < 3 { return 42 } else { return 0 }
}
SRC

write_case local_param_call 42 <<'SRC'
fn add(a: i64, b: i64) -> i64 {
    return a + b
}

fn main() -> i64 {
    return add(20, 22)
}
SRC

write_case local_param_const_call 42 <<'SRC'
fn forty_two(a: i64, b: i64) -> i64 {
    return 42
}

fn main() -> i64 {
    return forty_two(20, 22)
}
SRC

run_case() {
  local name="$1"
  local src="$TMP_DIR/$name.sio"
  local bin="$TMP_DIR/$name.elf"
  local log="$TMP_DIR/$name.compile.log"
  local expected
  expected="$(cat "$TMP_DIR/$name.expected")"

  echo "[madaros-source-semantics] running $name"
  "$MADAROS" compile "$src" -o "$bin" >"$log" 2>&1 || {
    cat "$log" >&2
    fail "$name compile failed"
  }
  [[ -s "$bin" ]] || {
    cat "$log" >&2
    fail "$name did not produce an ELF"
  }
  chmod +x "$bin"
  set +e
  "$bin"
  local rc=$?
  set -e
  [[ "$rc" -eq "$expected" ]] || {
    cat "$log" >&2
    fail "$name expected exit $expected, got $rc"
  }
}

run_case literal_body
run_case let_call_if
run_case while_var
run_case bool_cmp
run_case local_param_call
run_case local_param_const_call

echo "[madaros-source-semantics] running raw_compile_alias"
RAW_BIN="$TMP_DIR/raw_compile_alias.elf"
RAW_LOG="$TMP_DIR/raw_compile_alias.compile.log"
"$RAW_MADAROS" compile "$TMP_DIR/literal_body.sio" -o "$RAW_BIN" >"$RAW_LOG" 2>&1 || {
  cat "$RAW_LOG" >&2
  fail "raw compile alias failed"
}
[[ -s "$RAW_BIN" ]] || {
  cat "$RAW_LOG" >&2
  fail "raw compile alias did not produce an ELF"
}
chmod +x "$RAW_BIN"
set +e
"$RAW_BIN"
RAW_RC=$?
set -e
[[ "$RAW_RC" -eq 42 ]] || {
  cat "$RAW_LOG" >&2
  fail "raw compile alias expected exit 42, got $RAW_RC"
}

echo "MADAROS_SOURCE_SEMANTICS_PASS"
