#!/usr/bin/env bash
# Exercise the imported struct-layout boundary through the native Madaros path.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_BIN:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECT="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECT:-baseline}"
KEEP_WORK="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_KEEP:-0}"
WORK="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_DIR:-$(mktemp -d /tmp/sounio-madaros-struct-layout-capacity.XXXXXX)}"
GENERATOR="$ROOT_DIR/scripts/research/generate_madaros_struct_layout_capacity_fixture.py"

fail() {
  echo "[madaros-struct-layout-capacity] FAIL: $*" >&2
  exit 1
}

case "$EXPECT" in
  baseline|resolved) ;;
  *) fail "invalid SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECT=$EXPECT" ;;
esac

[[ -x "$MADAROS" ]] || fail "Madaros wrapper is missing or not executable: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN must name an explicit current-source Madaros ELF"
[[ -x "$RAW_MADAROS" ]] || fail "explicit current-source Madaros is missing or not executable: $RAW_MADAROS"
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
  local marker

  mkdir -p "$case_dir"
  python3 "$GENERATOR" --custom-layouts "$custom_layouts" --out-dir "$case_dir"
  marker="$(<"$case_dir/expected_marker.txt")"

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" run "$main_source" >"$case_dir/run.log" 2>&1
  CASE_RC=$?
  set -e
  CASE_MARKER=0
  grep -Fxq "$marker" "$case_dir/run.log" && CASE_MARKER=1
  CASE_LOG="$case_dir/run.log"
}

expect_runtime_witness() {
  local label="$1"
  [[ "$CASE_RC" -eq 0 ]] || {
    cat "$CASE_LOG" >&2
    fail "$label exited rc=$CASE_RC"
  }
  [[ "$CASE_MARKER" -eq 1 ]] || {
    cat "$CASE_LOG" >&2
    fail "$label returned zero without its exact field-access witness"
  }
}

expect_known_baseline_boundary() {
  [[ "$CASE_RC" -eq 1 ]] || {
    cat "$CASE_LOG" >&2
    fail "256-custom-layout baseline must exit rc=1, got rc=$CASE_RC"
  }
  [[ "$CASE_MARKER" -eq 0 ]] || {
    cat "$CASE_LOG" >&2
    fail "256-custom-layout baseline unexpectedly printed its runtime witness"
  }
  grep -Fxq 'run_check_mode: verdict=0' "$CASE_LOG" || {
    cat "$CASE_LOG" >&2
    fail "256-custom-layout baseline did not reach a clean imported check"
  }
  grep -Fxq 'IR lowering failed during merge: ir_summary_failed' "$CASE_LOG" || {
    cat "$CASE_LOG" >&2
    fail "256-custom-layout baseline changed before the known IR-summary boundary"
  }
}

run_case 255
expect_runtime_witness "255-custom-layout boundary"

run_case 256
case "$EXPECT" in
  baseline)
    expect_known_baseline_boundary
    echo "[madaros-struct-layout-capacity] BASELINE: 255 custom + Knowledge executes; 256 custom exposes the unrepaired catalog boundary"
    echo "[madaros-struct-layout-capacity] baseline_log=$CASE_LOG"
    ;;
  resolved)
    expect_runtime_witness "256-custom-layout boundary"
    echo "[madaros-struct-layout-capacity] PASS: imported field access crosses 257 catalog layouts via the explicit Madaros ELF"
    ;;
esac
