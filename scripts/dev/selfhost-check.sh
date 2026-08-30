#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/release/souc}"
FILE="${FILE:-examples/minimal.sio}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-check}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"
GOLDEN_FILE="${GOLDEN_FILE:-tests/golden/minimal.hlir.golden.txt}"
TIMEOUT_SECS="${TIMEOUT_SECS:-180}"
BUILD_SOUC="${BUILD_SOUC:-1}"
CLEAN_SYSROOT_HOME="${CLEAN_SYSROOT_HOME:-1}"
STRICT_MODE="${SOUNIO_SELFHOST_STRICT:-0}"

if [ -z "${TARGET_TRIPLE:-}" ]; then
  TARGET_TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
else
  TARGET_TRIPLE="${TARGET_TRIPLE}"
fi

if [ -z "$TARGET_TRIPLE" ]; then
  echo "FAIL [preflight] unable to detect rust host target triple" >&2
  exit 1
fi

if [ -z "${SOUNIO_SYSROOT_HOME:-}" ]; then
  export SOUNIO_SYSROOT_HOME="/tmp/sounio-sysroots-stage0"
fi

if [ -z "${SOUNIO_STDLIB_PATH:-}" ] && [ -d "$ROOT_DIR/stdlib" ]; then
  export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
fi

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0
STRICT_VIOLATION=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "PASS [$1] $2"
}

warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  echo "WARN [$1] $2"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  echo "FAIL [$1] $2"
}

run_with_timeout() {
  local seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
  else
    "$@"
  fi
}

run_capture() {
  local stdout_file="$1"
  local stderr_file="$2"
  shift 2
  run_with_timeout "$TIMEOUT_SECS" "$@" >"$stdout_file" 2>"$stderr_file"
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

if [ "$BUILD_SOUC" = "1" ]; then
  if run_with_timeout 900 cargo build -p souc --release >"$LOG_DIR/build.stdout" 2>"$LOG_DIR/build.stderr"; then
    pass "build" "cargo build -p souc --release"
  else
    fail "build" "cargo build failed (see $LOG_DIR/build.stderr)"
  fi
fi

if [ ! -x "$SOUC_BIN" ]; then
  fail "preflight" "compiler binary not found at $SOUC_BIN"
  echo
  echo "Summary: PASS=$PASS_COUNT WARN=$WARN_COUNT FAIL=$FAIL_COUNT"
  exit 1
fi

if [ ! -f "$FILE" ]; then
  fail "preflight" "input file not found: $FILE"
  echo
  echo "Summary: PASS=$PASS_COUNT WARN=$WARN_COUNT FAIL=$FAIL_COUNT"
  exit 1
fi

if [ "$CLEAN_SYSROOT_HOME" = "1" ]; then
  rm -rf "$SOUNIO_SYSROOT_HOME"
fi
mkdir -p "$SOUNIO_SYSROOT_HOME"

if run_capture "$LOG_DIR/sysroot.install.stdout" "$LOG_DIR/sysroot.install.stderr" \
  "$SOUC_BIN" sysroot install "$TARGET_TRIPLE" -v; then
  pass "stage0-preflight" "sysroot install completed for $TARGET_TRIPLE"
else
  fail "stage0-preflight" "sysroot install failed for $TARGET_TRIPLE"
fi

if run_capture "$LOG_DIR/sysroot.list.stdout" "$LOG_DIR/sysroot.list.stderr" \
  "$SOUC_BIN" sysroot list -v; then
  :
else
  fail "stage0-preflight" "sysroot list command failed"
fi

if rg -q "$TARGET_TRIPLE" "$LOG_DIR/sysroot.list.stdout"; then
  pass "stage0-preflight" "sysroot list includes $TARGET_TRIPLE"
else
  fail "stage0-preflight" "sysroot list is missing $TARGET_TRIPLE"
fi

if run_capture "$LOG_DIR/sysroot.show.stdout" "$LOG_DIR/sysroot.show.stderr" \
  "$SOUC_BIN" sysroot show "$TARGET_TRIPLE" -v; then
  :
else
  fail "stage0-preflight" "sysroot show command failed for $TARGET_TRIPLE"
fi

if rg -q "$TARGET_TRIPLE|source_kind|sysroot_path|stdlib_path" "$LOG_DIR/sysroot.show.stdout"; then
  pass "stage0-preflight" "sysroot show returns metadata"
else
  fail "stage0-preflight" "sysroot show output did not include expected metadata"
fi

if run_capture "$LOG_DIR/sysroot.stdlib.stdout" "$LOG_DIR/sysroot.stdlib.stderr" \
  "$SOUC_BIN" sysroot stdlib-paths -v; then
  :
else
  fail "stage0-preflight" "sysroot stdlib-paths command failed"
fi

if [ -n "${SOUNIO_STDLIB_PATH:-}" ] && [ -d "$SOUNIO_STDLIB_PATH" ]; then
  pass "stage0-preflight" "stdlib path exists at SOUNIO_STDLIB_PATH=$SOUNIO_STDLIB_PATH"
else
  fail "stage0-preflight" "SOUNIO_STDLIB_PATH is not set to a valid directory"
fi

for emit in ast hir hlir; do
  out_file="$ARTIFACT_DIR/minimal.$emit.txt"
  err_file="$LOG_DIR/compile.$emit.stderr"
  if run_with_timeout "$TIMEOUT_SECS" "$SOUC_BIN" compile --emit "$emit" "$FILE" -v >"$out_file" 2>"$err_file"; then
    if [ -s "$out_file" ]; then
      pass "stage0-ir" "compile --emit $emit produced $out_file"
    else
      fail "stage0-ir" "compile --emit $emit produced empty output"
    fi
  else
    fail "stage0-ir" "compile --emit $emit failed"
  fi
done

rust_stdout="$ARTIFACT_DIR/run.rust.stdout"
rust_stderr="$LOG_DIR/run.rust.stderr"
self_stdout="$ARTIFACT_DIR/run.self.stdout"
self_stderr="$LOG_DIR/run.self.stderr"

if run_capture "$rust_stdout" "$rust_stderr" "$SOUC_BIN" run "$FILE" -v; then
  pass "stage1-smoke" "run without self-host flag succeeded"
else
  fail "stage1-smoke" "run without self-host flag failed"
fi

if run_capture "$self_stdout" "$self_stderr" "$SOUC_BIN" run "$FILE" --use-sounio-compiler -v; then
  pass "stage1-smoke" "run with --use-sounio-compiler succeeded"
else
  fail "stage1-smoke" "run with --use-sounio-compiler failed"
fi

if cmp -s "$rust_stdout" "$self_stdout"; then
  pass "bootstrap-artifacts" "rust and self-host run stdout match"
else
  fail "bootstrap-artifacts" "rust and self-host run stdout differ"
fi

if rg -q "using Rust compiler" "$self_stderr"; then
  warn "stage1-smoke" "self-host path fell back to Rust compiler"
  if [ "$STRICT_MODE" = "1" ]; then
    fail "strict-mode" "fallback is forbidden when SOUNIO_SELFHOST_STRICT=1"
    STRICT_VIOLATION=1
  fi
else
  pass "stage1-smoke" "self-host path active (no fallback marker detected)"
fi

current_hlir="$ARTIFACT_DIR/minimal.hlir.now.txt"
if run_with_timeout "$TIMEOUT_SECS" "$SOUC_BIN" compile --emit hlir "$FILE" -v >"$current_hlir" 2>"$LOG_DIR/compile.hlir.now.stderr"; then
  pass "golden-check" "current HLIR generated"
else
  fail "golden-check" "failed to generate current HLIR for golden comparison"
fi

if [ -f "$GOLDEN_FILE" ]; then
  if diff -u "$GOLDEN_FILE" "$current_hlir" >"$LOG_DIR/golden.diff"; then
    pass "golden-check" "HLIR matches golden file ($GOLDEN_FILE)"
  else
    fail "golden-check" "HLIR drift detected (see $LOG_DIR/golden.diff)"
  fi
else
  fail "golden-check" "missing golden file: $GOLDEN_FILE (run dev/golden-update.sh)"
fi

echo
echo "Summary: PASS=$PASS_COUNT WARN=$WARN_COUNT FAIL=$FAIL_COUNT"
echo "Artifacts: $WORK_DIR"

if [ "$STRICT_VIOLATION" -eq 1 ]; then
  exit 2
fi

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0
