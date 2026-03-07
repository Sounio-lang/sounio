#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

SOUC_BIN="${SOUC_BIN:-./souc}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-bootstrap-kernel-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"
CYCLE_WORK_DIR="${CYCLE_WORK_DIR:-$WORK_DIR/cycle}"

BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-900}"
CYCLE_TIMEOUT_SECS="${CYCLE_TIMEOUT_SECS:-1200}"
BENCH_TIMEOUT_SECS="${BENCH_TIMEOUT_SECS:-60}"
KNOWLEDGE_TIMEOUT_SECS="${KNOWLEDGE_TIMEOUT_SECS:-300}"
BOOTSTRAP_MANIFEST_PATH="${BOOTSTRAP_MANIFEST_PATH:-${SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST:-bootstrap/selfhost-kernel.manifest}}"
SKIP_BUILD="${SKIP_BUILD:-0}"
BENCH_COMPILE_TARGET="${BENCH_COMPILE_TARGET:-examples/minimal.sio}"
BENCH_CHECK_TARGET="${BENCH_CHECK_TARGET:-examples/minimal.sio}"
BENCH_OUTPUT_PATH="${BENCH_OUTPUT_PATH:-$WORK_DIR/kernel-bench.bin}"
BENCH_SEED_ENFORCE="${BENCH_SEED_ENFORCE:-1}"
BENCH_SEED_PATH="${BENCH_SEED_PATH:-bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin}"
BENCH_SEED_SHA256_PATH="${BENCH_SEED_SHA256_PATH:-${BENCH_SEED_PATH}.sha256}"
BENCH_SEED_SIG_PATH="${BENCH_SEED_SIG_PATH:-${BENCH_SEED_PATH}.sig}"
BOOTSTRAP_MANIFEST_METADATA_PATH="${BOOTSTRAP_MANIFEST_METADATA_PATH:-bootstrap/artifacts/manifest.v2.json}"
BENCH_SEED_TRUSTED_KEY="${BENCH_SEED_TRUSTED_KEY:-}"
BENCH_NO_ITEM_COUNT="${BENCH_NO_ITEM_COUNT:-1}"
BENCH_FAST_SKIP_BLOCKS="${BENCH_FAST_SKIP_BLOCKS:-0}"
NO_RUST_MARKER_ENFORCE="${NO_RUST_MARKER_ENFORCE:-1}"
LAST_STEP_RC=0

resolve_bootstrap_seed_trusted_key() {
  if [[ -n "$BENCH_SEED_TRUSTED_KEY" ]]; then
    echo "$BENCH_SEED_TRUSTED_KEY"
    return 0
  fi

  if [[ -f "$BOOTSTRAP_MANIFEST_METADATA_PATH" ]] && command -v python3 >/dev/null 2>&1; then
    local manifest_key
    manifest_key="$(
      python3 - "$BOOTSTRAP_MANIFEST_METADATA_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
print(payload.get("key_id", ""))
PY
    )"
    if [[ -n "$manifest_key" ]]; then
      echo "$manifest_key"
      return 0
    fi
  fi

  echo "sounio-dev"
}

normalize_bool_01() {
  local raw="$1"
  local name="$2"
  local normalized

  normalized="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$normalized" in
    1|true|yes|on)
      echo "1"
      ;;
    0|false|no|off)
      echo "0"
      ;;
    *)
      echo "error: invalid $name=$raw (expected 0/1/true/false/yes/no/on/off)" >&2
      return 1
      ;;
  esac
}

SKIP_BUILD="$(normalize_bool_01 "$SKIP_BUILD" "SKIP_BUILD")"
BENCH_SEED_ENFORCE="$(normalize_bool_01 "$BENCH_SEED_ENFORCE" "BENCH_SEED_ENFORCE")"
BENCH_NO_ITEM_COUNT="$(normalize_bool_01 "$BENCH_NO_ITEM_COUNT" "BENCH_NO_ITEM_COUNT")"
BENCH_FAST_SKIP_BLOCKS="$(normalize_bool_01 "$BENCH_FAST_SKIP_BLOCKS" "BENCH_FAST_SKIP_BLOCKS")"
NO_RUST_MARKER_ENFORCE="$(normalize_bool_01 "$NO_RUST_MARKER_ENFORCE" "NO_RUST_MARKER_ENFORCE")"
BENCH_SEED_TRUSTED_KEY="$(resolve_bootstrap_seed_trusted_key)"

PASS_COUNT=0
FAIL_COUNT=0
declare -a STEP_RESULTS=()

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "PASS [$1] $2"
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

now_ns() {
  local candidate
  candidate="$(date +%s%N 2>/dev/null || true)"
  if [[ "$candidate" =~ ^[0-9]+$ ]]; then
    echo "$candidate"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import time
print(time.time_ns())
PY
    return 0
  fi

  date +%s
}

format_seconds_from_ms() {
  local millis="$1"
  awk -v ms="$millis" 'BEGIN { printf "%.3f", ms / 1000 }'
}

run_timed_step() {
  local step="$1"
  local timeout_secs="$2"
  local stdout_file="$3"
  local stderr_file="$4"
  shift 4

  local start_ns
  local end_ns
  local elapsed_ms
  local elapsed_s
  local rc

  start_ns="$(now_ns)"
  set +e
  run_with_timeout "$timeout_secs" "$@" >"$stdout_file" 2>"$stderr_file"
  rc=$?
  set -e
  end_ns="$(now_ns)"

  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
  elapsed_s="$(format_seconds_from_ms "$elapsed_ms")"
  STEP_RESULTS+=("$step,$rc,$elapsed_ms,$stdout_file,$stderr_file")
  LAST_STEP_RC="$rc"

  if [ "$rc" -eq 0 ]; then
    pass "$step" "elapsed=${elapsed_s}s timeout=${timeout_secs}s"
  # GNU timeout with --preserve-status propagates child TERM as 143 on timeout.
  elif [ "$rc" -eq 124 ] || [ "$rc" -eq 143 ] || [ "$rc" -eq 137 ]; then
    fail "$step" "timed out after ${timeout_secs}s (elapsed=${elapsed_s}s)"
  else
    fail "$step" "exit=$rc elapsed=${elapsed_s}s"
  fi
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"
TOTAL_START_NS="$(now_ns)"

declare -a BASE_ENV=(
  "SOUNIO_SELFHOST_STRICT_MODULE_GATING=1"
  "SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST=$BOOTSTRAP_MANIFEST_PATH"
  "SOUNIO_BOOTSTRAP_SEED_ENFORCE=$BENCH_SEED_ENFORCE"
  "SOUNIO_BOOTSTRAP_SEED_PATH=$BENCH_SEED_PATH"
  "SOUNIO_BOOTSTRAP_SEED_SHA256_PATH=$BENCH_SEED_SHA256_PATH"
  "SOUNIO_BOOTSTRAP_SEED_SIG_PATH=$BENCH_SEED_SIG_PATH"
  "SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY=$BENCH_SEED_TRUSTED_KEY"
)

echo "BOOTSTRAP_KERNEL_GATE_START"
echo "work_dir=$WORK_DIR"
echo "souc_bin=$SOUC_BIN"
echo "bench_timeout_secs=$BENCH_TIMEOUT_SECS"
echo "knowledge_timeout_secs=$KNOWLEDGE_TIMEOUT_SECS"
echo "cycle_timeout_secs=$CYCLE_TIMEOUT_SECS"
echo "bootstrap_manifest=$BOOTSTRAP_MANIFEST_PATH"
echo "skip_build=$SKIP_BUILD"
echo "bench_compile_target=$BENCH_COMPILE_TARGET"
echo "bench_check_target=$BENCH_CHECK_TARGET"
echo "bench_seed_enforce=$BENCH_SEED_ENFORCE"
echo "bench_seed_path=$BENCH_SEED_PATH"
echo "bench_seed_trusted_key=$BENCH_SEED_TRUSTED_KEY"
echo "bench_no_item_count=$BENCH_NO_ITEM_COUNT"
echo "bench_fast_skip_blocks=$BENCH_FAST_SKIP_BLOCKS"
echo "no_rust_marker_enforce=$NO_RUST_MARKER_ENFORCE"

if [ "$SKIP_BUILD" = "1" ]; then
  run_timed_step \
    "build-souc" \
    "$BUILD_TIMEOUT_SECS" \
    "$LOG_DIR/build.stdout.log" \
    "$LOG_DIR/build.stderr.log" \
    test -x "$SOUC_BIN"
else
  run_timed_step \
    "build-souc" \
    "$BUILD_TIMEOUT_SECS" \
    "$LOG_DIR/build.stdout.log" \
    "$LOG_DIR/build.stderr.log" \
    test -x "$SOUC_BIN"
fi

if [ "$LAST_STEP_RC" -ne 0 ]; then
  echo "BOOTSTRAP_KERNEL_GATE_PREREQ_SKIP reason=build-souc failed"
else

  declare -a CYCLE_ENV=(
    "${BASE_ENV[@]}"
    "WORK_DIR=$CYCLE_WORK_DIR"
    "SOUC_BIN=$SOUC_BIN"
    "SOUNIO_SELFHOST_CYCLE_SKIP_BUILD=1"
  )

  run_timed_step \
    "selfhost-cycle" \
    "$CYCLE_TIMEOUT_SECS" \
    "$LOG_DIR/selfhost-cycle.stdout.log" \
    "$LOG_DIR/selfhost-cycle.stderr.log" \
    env "${CYCLE_ENV[@]}" bash scripts/selfhost/selfhost_cycle_gate.sh

  run_timed_step \
    "compile-60s" \
    "$BENCH_TIMEOUT_SECS" \
    "$LOG_DIR/compile-60s.stdout.log" \
    "$LOG_DIR/compile-60s.stderr.log" \
    "$SOUC_BIN" compile "$BENCH_COMPILE_TARGET" -o "$BENCH_OUTPUT_PATH"

  run_timed_step \
    "check-60s" \
    "$BENCH_TIMEOUT_SECS" \
    "$LOG_DIR/check-60s.stdout.log" \
    "$LOG_DIR/check-60s.stderr.log" \
    "$SOUC_BIN" check "$BENCH_CHECK_TARGET"

  if [ "$NO_RUST_MARKER_ENFORCE" = "1" ]; then
    run_timed_step \
      "no-rust-markers" \
      30 \
      "$LOG_DIR/no-rust-markers.stdout.log" \
      "$LOG_DIR/no-rust-markers.stderr.log" \
      bash scripts/assert_no_rust_markers.sh \
      "$LOG_DIR/selfhost-cycle.stdout.log" \
      "$LOG_DIR/selfhost-cycle.stderr.log" \
      "$LOG_DIR/compile-60s.stdout.log" \
      "$LOG_DIR/compile-60s.stderr.log" \
      "$LOG_DIR/check-60s.stdout.log" \
      "$LOG_DIR/check-60s.stderr.log"
  fi

  run_timed_step \
    "knowledge-bootstrap" \
    "$KNOWLEDGE_TIMEOUT_SECS" \
    "$LOG_DIR/knowledge-bootstrap.stdout.log" \
    "$LOG_DIR/knowledge-bootstrap.stderr.log" \
    bash scripts/bootstrap/run_knowledge_bootstrap_tests.sh
fi

TOTAL_END_NS="$(now_ns)"
TOTAL_ELAPSED_MS=$(( (TOTAL_END_NS - TOTAL_START_NS) / 1000000 ))
TOTAL_ELAPSED_S="$(format_seconds_from_ms "$TOTAL_ELAPSED_MS")"

{
  echo "BOOTSTRAP_KERNEL_GATE_SUMMARY"
  for result in "${STEP_RESULTS[@]}"; do
    IFS=',' read -r step rc elapsed_ms stdout_file stderr_file <<<"$result"
    echo "$step rc=$rc elapsed_ms=$elapsed_ms stdout=$stdout_file stderr=$stderr_file"
  done
  echo "pass_count=$PASS_COUNT"
  echo "fail_count=$FAIL_COUNT"
  echo "total_elapsed_ms=$TOTAL_ELAPSED_MS"
} >"$SUMMARY_FILE"

echo "BOOTSTRAP_KERNEL_GATE_SUMMARY summary_file=$SUMMARY_FILE total_elapsed=${TOTAL_ELAPSED_S}s pass_count=$PASS_COUNT fail_count=$FAIL_COUNT"
for result in "${STEP_RESULTS[@]}"; do
  IFS=',' read -r step rc elapsed_ms _ _ <<<"$result"
  echo "BOOTSTRAP_KERNEL_GATE_STEP step=$step rc=$rc elapsed_ms=$elapsed_ms"
done

if [ "$FAIL_COUNT" -eq 0 ]; then
  echo "BOOTSTRAP_KERNEL_GATE_RESULT PASS total_elapsed=${TOTAL_ELAPSED_S}s"
  exit 0
fi

echo "BOOTSTRAP_KERNEL_GATE_RESULT FAIL total_elapsed=${TOTAL_ELAPSED_S}s"
exit 1
