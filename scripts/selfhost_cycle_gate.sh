#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-cycle-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"

STAGE1_PATH="${STAGE1_PATH:-$ARTIFACT_DIR/selfhost.stage1.sobc}"
STAGE2_PATH="${STAGE2_PATH:-$ARTIFACT_DIR/selfhost.stage2.sobc}"
CACHE_PATH="${CACHE_PATH:-self-hosted/.sounio_bytecode.sobc}"

BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-900}"
COMPILE_TIMEOUT_SECS="${COMPILE_TIMEOUT_SECS:-900}"
SKIP_BUILD="${SOUNIO_SELFHOST_CYCLE_SKIP_BUILD:-0}"
SEED_ENFORCE="${SOUNIO_SELFHOST_CYCLE_SEED_ENFORCE:-0}"
SEED_PATH="${SOUNIO_SELFHOST_CYCLE_SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
SEED_SHA256_PATH="${SOUNIO_SELFHOST_CYCLE_SEED_SHA256_PATH:-${SEED_PATH}.sha256}"
SEED_SIG_PATH="${SOUNIO_SELFHOST_CYCLE_SEED_SIG_PATH:-${SEED_PATH}.sig}"
CYCLE_FORCE_DYNAMIC="${SOUNIO_SELFHOST_CYCLE_FORCE_DYNAMIC:-1}"
NO_RUST_MARKER_ENFORCE="${SOUNIO_SELFHOST_CYCLE_NO_RUST_MARKER_ENFORCE:-1}"
NO_RUST_MARKER_TIMEOUT_SECS="${SOUNIO_SELFHOST_CYCLE_NO_RUST_MARKER_TIMEOUT_SECS:-30}"
NO_RUST_HARNESS="${SOUNIO_SELFHOST_CYCLE_NO_RUST_HARNESS:-1}"
BOOTSTRAP_MANIFEST_PATH="${BOOTSTRAP_MANIFEST_PATH:-${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}}"

if [ "$CYCLE_FORCE_DYNAMIC" = "1" ]; then
  # Cycle reproducibility should remain testable even when a seed is present.
  # Force the bootstrap seed loader down the dynamic path for this gate.
  SEED_PATH="${SEED_PATH}.missing"
  SEED_SHA256_PATH="${SEED_SHA256_PATH}.missing"
  SEED_SIG_PATH="${SEED_SIG_PATH}.missing"
fi

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$seconds" "$@" <<'PY'
import subprocess
import sys

seconds = int(sys.argv[1])
command = sys.argv[2:]
try:
    completed = subprocess.run(command, timeout=seconds)
    sys.exit(completed.returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
PY
    return $?
  fi

  "$@"
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

BUILD_LOG="$LOG_DIR/build.log"
STAGE1_LOG="$LOG_DIR/stage1.log"
STAGE2_LOG="$LOG_DIR/stage2.log"
NO_RUST_MARKER_LOG="$LOG_DIR/no-rust-markers.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
rm -f "$BUILD_LOG" "$STAGE1_LOG" "$STAGE2_LOG" "$NO_RUST_MARKER_LOG"
rm -f "$SUMMARY_FILE" "$STAGE1_PATH" "$STAGE2_PATH"

echo "SELFHOST_CYCLE_GATE_START"
echo "stage1_path=$STAGE1_PATH"
echo "stage2_path=$STAGE2_PATH"
echo "cache_path=$CACHE_PATH"
echo "seed_enforce=$SEED_ENFORCE"
echo "seed_path=$SEED_PATH"
echo "cycle_force_dynamic=$CYCLE_FORCE_DYNAMIC"
echo "no_rust_marker_enforce=$NO_RUST_MARKER_ENFORCE"
echo "no_rust_harness=$NO_RUST_HARNESS"
echo "bootstrap_manifest=$BOOTSTRAP_MANIFEST_PATH"

if [ "$SKIP_BUILD" = "1" ]; then
  : >"$BUILD_LOG"
else
  run_with_timeout "$BUILD_TIMEOUT_SECS" cargo build -p souc >"$BUILD_LOG" 2>&1
fi

if [ ! -x "$SOUC_BIN" ]; then
  echo "error: missing compiler binary at $SOUC_BIN" >&2
  exit 1
fi

capture_stage_artifact() {
  local stage_label="$1"
  local stage_log="$2"
  local stage_path="$3"
  local marker_cmd="grep -E"

  if [ -f "$CACHE_PATH" ]; then
    cp "$CACHE_PATH" "$stage_path"
    return 0
  fi

  if [ "$SEED_ENFORCE" = "1" ] && [ "$CYCLE_FORCE_DYNAMIC" != "1" ]; then
    if [ ! -s "$stage_log" ]; then
      echo "error: missing seed-root stage log payload: $stage_log" >&2
      exit 1
    fi

    if command -v rg >/dev/null 2>&1; then
      marker_cmd="rg -n"
    fi

    if ! $marker_cmd "SELFHOST=run schema=v1 event=selfhost_input_check status=resolved" "$stage_log" >/dev/null 2>&1; then
      echo "error: seed-root stage log missing selfhost_input_check marker: $stage_log" >&2
      exit 1
    fi

    if ! $marker_cmd "SELFHOST=seed schema=v1 event=bootstrap_seed status=ok" "$stage_log" >/dev/null 2>&1; then
      echo "error: seed-root stage log missing bootstrap_seed marker: $stage_log" >&2
      exit 1
    fi

    # Seed-root mode can execute directly from the trusted seed artifact without
    # producing a dynamic directory cache; use deterministic stage logs as the
    # reproducibility artifact in this mode.
    cp "$stage_log" "$stage_path"
    echo "SELFHOST_CYCLE_GATE_INFO stage=$stage_label artifact_source=log"
    return 0
  fi

  echo "error: missing ${stage_label} cache artifact at $CACHE_PATH" >&2
  exit 1
}

# Stage1: compile self-hosted suite and persist deterministic directory cache bytes.
rm -f "$CACHE_PATH"
run_with_timeout "$COMPILE_TIMEOUT_SECS" env \
  SOUNIO_BOOTSTRAP_SEED_ENFORCE="$SEED_ENFORCE" \
  SOUNIO_BOOTSTRAP_SEED_PATH="$SEED_PATH" \
  SOUNIO_BOOTSTRAP_SEED_SHA256_PATH="$SEED_SHA256_PATH" \
  SOUNIO_BOOTSTRAP_SEED_SIG_PATH="$SEED_SIG_PATH" \
  SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST="$BOOTSTRAP_MANIFEST_PATH" \
  SOUNIO_SELFHOST_STRICT_MODULE_GATING="1" \
  SOUNIO_SELFHOST_WRITE_DIR_CACHE="1" \
  SOUNIO_SELFHOST_NO_RUST_HARNESS="$NO_RUST_HARNESS" \
  SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT="0" \
  SOUNIO_SELFHOST_PIPELINE="driver" \
  "$SOUC_BIN" run self-hosted/ -- parse-all shard 0 1 balanced >"$STAGE1_LOG" 2>&1

capture_stage_artifact "stage1" "$STAGE1_LOG" "$STAGE1_PATH"

# Stage2: repeat the self-hosted build and compare resulting cache bytes.
rm -f "$CACHE_PATH"
run_with_timeout "$COMPILE_TIMEOUT_SECS" env \
  SOUNIO_BOOTSTRAP_SEED_ENFORCE="$SEED_ENFORCE" \
  SOUNIO_BOOTSTRAP_SEED_PATH="$SEED_PATH" \
  SOUNIO_BOOTSTRAP_SEED_SHA256_PATH="$SEED_SHA256_PATH" \
  SOUNIO_BOOTSTRAP_SEED_SIG_PATH="$SEED_SIG_PATH" \
  SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST="$BOOTSTRAP_MANIFEST_PATH" \
  SOUNIO_SELFHOST_STRICT_MODULE_GATING="1" \
  SOUNIO_SELFHOST_WRITE_DIR_CACHE="1" \
  SOUNIO_SELFHOST_NO_RUST_HARNESS="$NO_RUST_HARNESS" \
  SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT="0" \
  SOUNIO_SELFHOST_PIPELINE="driver" \
  "$SOUC_BIN" run self-hosted/ -- parse-all shard 0 1 balanced >"$STAGE2_LOG" 2>&1

capture_stage_artifact "stage2" "$STAGE2_LOG" "$STAGE2_PATH"

if ! cmp -s "$STAGE1_PATH" "$STAGE2_PATH"; then
  echo "error: self-host cycle mismatch: $STAGE1_PATH != $STAGE2_PATH" >&2
  exit 1
fi

if [ "$NO_RUST_MARKER_ENFORCE" = "1" ]; then
  if ! run_with_timeout "$NO_RUST_MARKER_TIMEOUT_SECS" \
    bash scripts/assert_no_rust_markers.sh \
    "$STAGE1_LOG" \
    "$STAGE2_LOG" >"$NO_RUST_MARKER_LOG" 2>&1; then
    echo "error: rust marker leakage detected (see $NO_RUST_MARKER_LOG)" >&2
    cat "$NO_RUST_MARKER_LOG" >&2
    exit 1
  fi
fi

SHA_STAGE1="$(sha256sum "$STAGE1_PATH" | awk '{print $1}')"
SHA_STAGE2="$(sha256sum "$STAGE2_PATH" | awk '{print $1}')"

{
  echo "stage1_sha256=$SHA_STAGE1"
  echo "stage2_sha256=$SHA_STAGE2"
  echo "stage_bytes=$(wc -c <"$STAGE1_PATH")"
  echo "cache_path=$CACHE_PATH"
} >"$SUMMARY_FILE"

echo "SELFHOST_CYCLE_GATE_DONE sha=$SHA_STAGE1 bytes=$(wc -c <"$STAGE1_PATH")"
