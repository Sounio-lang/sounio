#!/usr/bin/env bash
# Exercise the imported struct-layout boundary through the native Madaros path.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECT="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECT:-baseline}"
KEEP_WORK="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_KEEP:-0}"
WORK="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_DIR:-$(mktemp -d /tmp/sounio-madaros-struct-layout-capacity.XXXXXX)}"
GENERATOR="$ROOT_DIR/scripts/research/generate_madaros_struct_layout_capacity_fixture.py"

fail() {
  echo "[madaros-struct-layout-capacity] FAIL: $*" >&2
  exit 1
}

elf_magic() {
  od -An -tx1 -N4 "$1" 2>/dev/null | tr -d '[:space:]'
}

assert_executable_elf() {
  local path="$1"
  local label="$2"

  [[ -f "$path" && -s "$path" ]] || fail "$label is missing or empty: $path"
  [[ -x "$path" ]] || fail "$label is not executable: $path"
  [[ "$(elf_magic "$path")" == 7f454c46 ]] || fail "$label is not an ELF: $path"
}

case "$EXPECT" in
  baseline|resolved) ;;
  *) fail "invalid SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECT=$EXPECT" ;;
esac

[[ -n "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN must name an explicit current-source Madaros ELF"
assert_executable_elf "$RAW_MADAROS" 'Madaros input'
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ -f "$GENERATOR" ]] || fail "fixture generator is missing: $GENERATOR"

mkdir -p "$WORK"
if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

run_case() {
  local custom_layouts="$1"
  local case_dir="$WORK/custom-$custom_layouts"
  local main_source="$case_dir/layout_capacity_main.sio"
  local output="$case_dir/layout_capacity.elf"
  local compile_log="$case_dir/compile.log"
  local runtime_log="$case_dir/runtime.log"
  local marker

  mkdir -p "$case_dir"
  python3 "$GENERATOR" --custom-layouts "$custom_layouts" --out-dir "$case_dir"
  marker="$(<"$case_dir/expected_marker.txt")"

  rm -f "$output"
  set +e
  (
    cd "$case_dir"
    exec env \
      -u MADAROS_RAW_BIN \
      -u SOUNIO_MADAROS_BIN \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$RAW_MADAROS" --native-v2-compile "$main_source" "$output"
  ) >"$compile_log" 2>&1
  CASE_COMPILE_RC=$?
  set -e

  CASE_RUNTIME_RC=127
  if [[ "$CASE_COMPILE_RC" -eq 0 ]]; then
    if [[ -e "$output" ]]; then
      chmod +x "$output"
      assert_executable_elf "$output" "${custom_layouts}-custom-layout witness ELF"
      set +e
      (cd "$case_dir" && "$output") >"$runtime_log" 2>&1
      CASE_RUNTIME_RC=$?
      set -e
    else
      printf 'compiler exited zero without an output ELF\n' >"$runtime_log"
    fi
  else
    : >"$runtime_log"
  fi

  CASE_MARKER=0
  grep -Fxq "$marker" "$runtime_log" && CASE_MARKER=1
  CASE_COMPILE_LOG="$compile_log"
  CASE_RUNTIME_LOG="$runtime_log"
}

expect_runtime_witness() {
  local label="$1"
  [[ "$CASE_COMPILE_RC" -eq 0 ]] || {
    cat "$CASE_COMPILE_LOG" >&2
    fail "$label did not compile (rc=$CASE_COMPILE_RC)"
  }
  [[ "$CASE_RUNTIME_RC" -eq 0 ]] || {
    cat "$CASE_RUNTIME_LOG" >&2
    fail "$label ELF exited rc=$CASE_RUNTIME_RC"
  }
  [[ "$CASE_MARKER" -eq 1 ]] || {
    cat "$CASE_RUNTIME_LOG" >&2
    fail "$label returned zero without its exact field-access witness"
  }
}

run_case 255
expect_runtime_witness "255-custom-layout boundary"

run_case 256
case "$EXPECT" in
  baseline)
    if [[ "$CASE_COMPILE_RC" -eq 0 && "$CASE_RUNTIME_RC" -eq 0 && "$CASE_MARKER" -eq 1 ]]; then
      cat "$CASE_RUNTIME_LOG" >&2
      fail "256-custom-layout boundary already executes; rerun with SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECT=resolved"
    fi
    echo "[madaros-struct-layout-capacity] BASELINE: 255 custom + Knowledge executes; 256 custom exposes the unrepaired catalog boundary"
    echo "[madaros-struct-layout-capacity] baseline_compile_log=$CASE_COMPILE_LOG"
    echo "[madaros-struct-layout-capacity] baseline_runtime_log=$CASE_RUNTIME_LOG"
    ;;
  resolved)
    expect_runtime_witness "256-custom-layout boundary"
    echo "[madaros-struct-layout-capacity] PASS: imported field access crosses 257 catalog layouts via the explicit Madaros ELF"
    ;;
esac
