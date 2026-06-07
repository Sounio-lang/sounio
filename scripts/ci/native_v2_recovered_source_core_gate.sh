#!/usr/bin/env bash
# Honest replacement coverage for retired native_compile_driver.sio gates.
#
# Source fixtures are compiled with the modular compiler's --native-v2-compile
# path, then the emitted ELF is executed and checked. No native_compile_driver.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s 2>/dev/null || echo unknown):$(uname -m 2>/dev/null || echo unknown)" in
  Linux:x86_64|Linux:amd64) ;;
  *)
    echo "[native-v2-recovered-source-core] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

ulimit -s 1048576 2>/dev/null || true

WORK="$(mktemp -d /tmp/sounio-native-v2-recovered-core.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

BOOTSTRAP_SOUC="${SOUNIO_BOOTSTRAP_SOUC:-./bin/souc}"
MODULAR_SOUC="${SOUNIO_MODULAR_SOUC:-}"

if [[ -z "$MODULAR_SOUC" ]]; then
  echo "[native-v2-recovered-source-core] building modular compiler from self-hosted/compiler/main.sio"
  MODULAR_SOUC="$WORK/mc.elf"
  if ! "$BOOTSTRAP_SOUC" self-hosted/compiler/main.sio "$MODULAR_SOUC" >"$WORK/build.log" 2>&1; then
    echo "[native-v2-recovered-source-core] FAIL: could not build modular compiler" >&2
    tail -20 "$WORK/build.log" >&2 || true
    exit 2
  fi
  chmod +x "$MODULAR_SOUC"
fi

if [[ ! -x "$MODULAR_SOUC" ]]; then
  echo "[native-v2-recovered-source-core] FAIL: modular compiler is not executable: $MODULAR_SOUC" >&2
  exit 2
fi

FAILED=0

expect_stdout() {
  local expected_file="$1"
  local actual_file="$2"
  local label="$3"
  if ! cmp -s "$expected_file" "$actual_file"; then
    echo "[native-v2-recovered-source-core] FAIL[$label]: stdout mismatch" >&2
    diff -u "$expected_file" "$actual_file" >&2 || true
    FAILED=1
  fi
}

compile_run_case() {
  local label="$1"
  local src="$2"
  local expected_exit="$3"
  local expected_stdout_literal="$4"
  local elf="$WORK/$label.elf"
  local compile_log="$WORK/$label.compile.log"
  local stdout_log="$WORK/$label.stdout"
  local stderr_log="$WORK/$label.stderr"
  local expected_stdout="$WORK/$label.expected.stdout"
  rm -f "$elf"

  "$MODULAR_SOUC" --native-v2-compile "$src" -o "$elf" >"$compile_log" 2>&1
  local compile_rc=$?
  if [[ "$compile_rc" -ne 0 || ! -f "$elf" ]]; then
    echo "[native-v2-recovered-source-core] FAIL[$label]: compile failed rc=$compile_rc src=$src" >&2
    tail -40 "$compile_log" >&2 || true
    FAILED=1
    return
  fi

  local magic
  magic="$(head -c4 "$elf" | od -An -tx1 | tr -d ' ')"
  if [[ "$magic" != "7f454c46" ]]; then
    echo "[native-v2-recovered-source-core] FAIL[$label]: emitted file is not ELF (magic=$magic)" >&2
    FAILED=1
    return
  fi

  chmod +x "$elf"
  "$elf" >"$stdout_log" 2>"$stderr_log"
  local got=$?
  if [[ "$got" -ne "$expected_exit" ]]; then
    echo "[native-v2-recovered-source-core] FAIL[$label]: exit=$got expected=$expected_exit" >&2
    cat "$stdout_log" >&2 || true
    cat "$stderr_log" >&2 || true
    FAILED=1
    return
  fi

  if [[ "$expected_stdout_literal" != "-" ]]; then
    printf "%b" "$expected_stdout_literal" >"$expected_stdout"
    expect_stdout "$expected_stdout" "$stdout_log" "$label"
  elif [[ -s "$stdout_log" ]]; then
    echo "[native-v2-recovered-source-core] FAIL[$label]: expected empty stdout" >&2
    cat "$stdout_log" >&2 || true
    FAILED=1
    return
  fi

  echo "[native-v2-recovered-source-core] PASS[$label]: $src exit=$got"
}

echo "[native-v2-recovered-source-core] source -> native-v2 ELF recovered gate"

compile_run_case serious_track_hello examples/native/hello.sio 0 'Hello from self-hosted Sounio!\n42\n'
compile_run_case array_min tests/native-v2/recovered/array_min_exit45.sio 45 -
compile_run_case logical_min tests/native-v2/recovered/logical_min_exit7.sio 7 -
compile_run_case enum_match_min tests/native-v2/recovered/enum_match_min_exit2.sio 2 -
compile_run_case nested_field examples/native/nested_field.sio 0 '9\n'
compile_run_case struct_basic examples/native/struct_basic.sio 0 '7\n'
compile_run_case struct_mutation examples/native/struct_mutation.sio 0 '14\n'
compile_run_case struct_param examples/native/struct_param.sio 0 '12\n'
compile_run_case struct_return examples/native/struct_return.sio 0 '8\n'

if [[ "$FAILED" -ne 0 ]]; then
  echo "[native-v2-recovered-source-core] FAIL: one or more recovered source gates failed" >&2
  exit 1
fi

echo "[native-v2-recovered-source-core] PASS: recovered serious_track/array/logical/enum/nested/struct source coverage"
