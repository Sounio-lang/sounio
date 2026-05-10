#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

SOUC_BIN="${SOUC_BIN:-./souc}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-zero-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"
DRIVER_HARNESS_CACHE_DIR="${DRIVER_HARNESS_CACHE_DIR:-$WORK_DIR/driver_harness_cache}"
PREWARM_TARGET="${PREWARM_TARGET:-examples/minimal.sio}"

FULL_TIMEOUT_SECS="${FULL_TIMEOUT_SECS:-900}"
PARSE_TIMEOUT_SECS="${PARSE_TIMEOUT_SECS:-900}"
BUILD_TIMEOUT_SECS="${BUILD_TIMEOUT_SECS:-900}"

PARSE_SHARD_COUNT="${PARSE_SHARD_COUNT:-56}"
PARSE_SHARD_SELECTOR="${PARSE_SHARD_SELECTOR:-all}"
PARSE_SHARD_MODE="${PARSE_SHARD_MODE:-balanced}"

FULL_TARGET="${FULL_TARGET:-self-hosted/}"

STRICT_MODE="${SOUNIO_SELFHOST_STRICT:-1}"
INDEPENDENCE_CONTRACT_PATH="${INDEPENDENCE_CONTRACT_PATH:-benchmarks/independence/contract.v1.json}"
DECISION_TRAIL_REQUIRED="${SOUNIO_OPT_DECISION_TRAIL_REQUIRED:-1}"
DECISION_TRAIL_PATH="${SOUNIO_OPT_DECISION_TRAIL_PATH:-$LOG_DIR/decision_trail.jsonl}"

PASS_COUNT=0
FAIL_COUNT=0

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

  # Pure-perl timeout fallback (replaces python3 subprocess.run heredoc).
  # See selfhost_driver_output_parity_gate.sh for rationale.
  if command -v perl >/dev/null 2>&1; then
    perl -e 'alarm $ARGV[0]; shift @ARGV; exec @ARGV; exit 127' "$seconds" "$@"
    local rc=$?
    [[ $rc -eq 142 ]] && rc=124
    return $rc
  fi

  "$@"
}

run_with_timeout_pty_log() {
  local seconds="$1"
  local log_file="$2"
  shift 2

  # `souc run self-hosted/` can run for minutes and writes a lot of output.
  # When stdout/stderr are redirected to a file, Rust's buffered IO can make logs
  # appear empty until the process exits. Run it under a PTY when available so
  # progress is visible and the gate can diagnose failures deterministically.
  if command -v script >/dev/null 2>&1; then
    if command -v timeout >/dev/null 2>&1; then
      timeout --preserve-status "${seconds}s" script -f -q -e -c "$*" "$log_file"
      return $?
    fi
    script -f -q -e -c "$*" "$log_file"
    return $?
  fi

  run_with_timeout "$seconds" "$@" >"$log_file" 2>&1
  return $?
}

count_matches() {
  local pattern="$1"
  local file="$2"
  local count="0"
  local status=0

  if [ ! -f "$file" ]; then
    echo "0"
    return 0
  fi

  if command -v rg >/dev/null 2>&1; then
    set +e
    count="$(rg -c -- "$pattern" "$file" 2>/dev/null)"
    status=$?
    set -e
  else
    set +e
    count="$(grep -E -c -- "$pattern" "$file" 2>/dev/null)"
    status=$?
    set -e
  fi

  if [ "$status" -ne 0 ] && [ "$status" -ne 1 ]; then
    count="0"
  fi

  if [[ ! "$count" =~ ^[0-9]+$ ]]; then
    count="0"
  fi

  echo "$count"
}

first_match() {
  local pattern="$1"
  local file="$2"
  local line=""

  if [ ! -f "$file" ]; then
    echo ""
    return 0
  fi

  if command -v rg >/dev/null 2>&1; then
    line="$(rg -- "$pattern" "$file" -m 1 2>/dev/null || true)"
  else
    line="$(grep -E -- "$pattern" "$file" 2>/dev/null | head -n 1 || true)"
  fi

  echo "$line"
}

extract_kv_value() {
  local line="$1"
  local key="$2"
  local value
  value="$(printf '%s\n' "$line" | sed -n "s/.*${key}=\\([^ ]*\\).*/\\1/p")"
  if [ -z "$value" ]; then
    echo ""
  else
    echo "$value"
  fi
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"
mkdir -p "$DRIVER_HARNESS_CACHE_DIR"

export SOUNIO_OPT_DECISION_TRAIL_REQUIRED="$DECISION_TRAIL_REQUIRED"
export SOUNIO_OPT_DECISION_TRAIL_PATH="$DECISION_TRAIL_PATH"

BUILD_LOG="$LOG_DIR/build.log"
PREWARM_LOG="$LOG_DIR/prewarm.log"
FULL_LOG="$LOG_DIR/full_selfhost.log"
REPORT_LOG="$LOG_DIR/parse_all_report.log"
SHARDS_LOG="$LOG_DIR/parse_all_shards.log"
SUMMARY_FILE="$ARTIFACT_DIR/summary.txt"

echo "SELFHOST_ZERO_GATE_START"
echo "work_dir=$WORK_DIR"
echo "full_timeout_secs=$FULL_TIMEOUT_SECS"
echo "parse_timeout_secs=$PARSE_TIMEOUT_SECS"
echo "parse_shard_count=$PARSE_SHARD_COUNT"
echo "parse_shard_selector=$PARSE_SHARD_SELECTOR"
echo "parse_shard_mode=$PARSE_SHARD_MODE"
echo "strict_mode=$STRICT_MODE"
echo "independence_contract=$INDEPENDENCE_CONTRACT_PATH"
echo "decision_trail_required=$SOUNIO_OPT_DECISION_TRAIL_REQUIRED"
echo "decision_trail_path=$SOUNIO_OPT_DECISION_TRAIL_PATH"
echo "driver_harness_cache_dir=$DRIVER_HARNESS_CACHE_DIR"
echo "prewarm_target=$PREWARM_TARGET"

if [ -f "$INDEPENDENCE_CONTRACT_PATH" ]; then
  # Validate schema via pure-Sounio kaxi-validate-evidence (replaces python json.loads).
  # ROOT_DIR in this gate is scripts/, not repo root — use ../bin/kretikos to reach it.
  "$ROOT_DIR/../bin/kretikos" kaxi-validate-evidence "$INDEPENDENCE_CONTRACT_PATH" \
    --expect "schema=sounio.independence.contract.v1"
fi

if [[ "$SKIP_BUILD" = "1" ]]; then
  if [[ -x "$SOUC_BIN" ]]; then
    pass "build" "skipped (SKIP_BUILD=1, binary at $SOUC_BIN)"
  else
    fail "build" "SKIP_BUILD=1 but binary not found at $SOUC_BIN"
  fi
else
{
  echo "SELFHOST_ZERO_GATE_INFO build_step=disabled mode=repo-hard-no-rust"
  echo "hint=provide SOUC_BIN prebuilt compiler"
} >"$BUILD_LOG"
pass "build" "cargo build disabled (repo-hard no-rust)"
fi

if [ ! -x "$SOUC_BIN" ]; then
  fail "preflight" "missing compiler binary at $SOUC_BIN"
fi

if [ "$FAIL_COUNT" -eq 0 ]; then
  set +e
  run_with_timeout "$FULL_TIMEOUT_SECS" env \
    SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
    "$SOUC_BIN" run "$PREWARM_TARGET" --use-sounio-compiler --check-only >"$PREWARM_LOG" 2>&1
  prewarm_code=$?
  set -e
  prewarm_harness_unavailable=0
  if [ -f "$PREWARM_LOG" ] && rg -n "SELFHOST_DRIVER_HARNESS_UNAVAILABLE" "$PREWARM_LOG" >/dev/null 2>&1; then
    prewarm_harness_unavailable=1
  fi

  if [ "$prewarm_code" -eq 0 ]; then
    pass "prewarm" "compiled and executed $PREWARM_TARGET to populate driver harness cache"
  elif [ "$prewarm_harness_unavailable" -eq 1 ]; then
    echo "SELFHOST_ZERO_GATE_INFO prewarm_status=blocked reason=SELFHOST_DRIVER_HARNESS_UNAVAILABLE cache_dir=$DRIVER_HARNESS_CACHE_DIR"
    fail "prewarm" "driver harness unavailable during prewarm (exit=$prewarm_code target=$PREWARM_TARGET)"
  else
    fail "prewarm" "failed to prewarm harness cache (exit=$prewarm_code target=$PREWARM_TARGET)"
  fi

  if ls "$DRIVER_HARNESS_CACHE_DIR"/driver_harness_*.sobc >/dev/null 2>&1; then
    pass "prewarm-cache" "found driver_harness_*.sobc in $DRIVER_HARNESS_CACHE_DIR"
  elif [ "$prewarm_harness_unavailable" -eq 1 ]; then
    echo "SELFHOST_ZERO_GATE_INFO prewarm_cache_status=blocked reason=SELFHOST_DRIVER_HARNESS_UNAVAILABLE cache_dir=$DRIVER_HARNESS_CACHE_DIR"
    fail "prewarm-cache" "no driver_harness_*.sobc found due to SELFHOST_DRIVER_HARNESS_UNAVAILABLE in $DRIVER_HARNESS_CACHE_DIR"
  else
    fail "prewarm-cache" "no driver_harness_*.sobc found in $DRIVER_HARNESS_CACHE_DIR"
  fi
fi

if [ "$FAIL_COUNT" -eq 0 ]; then
  set +e
  run_with_timeout_pty_log "$FULL_TIMEOUT_SECS" "$FULL_LOG" env \
    SOUNIO_SELFHOST_STRICT="$STRICT_MODE" \
    SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
    "$SOUC_BIN" run "$FULL_TARGET"
  full_code=$?
  set -e

  parse_all_header_count="$(count_matches '=== Parse All ===' "$FULL_LOG")"
  parse_all_summary_count="$(count_matches 'Parse All:' "$FULL_LOG")"
  suites_all_count="$(count_matches 'Suites: all passed' "$FULL_LOG")"
  suite_fail_count="$(count_matches 'SUITE FAIL:|Suites: suite fail' "$FULL_LOG")"
  driver_fallback_count="$(count_matches 'SELFHOST=driver-first schema=v1 event=driver_orchestration .* status=fallback' "$FULL_LOG")"
  driver_strict_error_count="$(count_matches 'SELFHOST=driver-first schema=v1 event=driver_orchestration .* status=strict_error' "$FULL_LOG")"
  oracle_count="$(count_matches 'SELFHOST=oracle backend=rust' "$FULL_LOG")"

  if [ "$full_code" -eq 0 ] &&
     [ "$parse_all_header_count" -gt 0 ] &&
     [ "$parse_all_summary_count" -gt 0 ] &&
     [ "$suites_all_count" -gt 0 ] &&
     [ "$suite_fail_count" -eq 0 ] &&
     [ "$driver_fallback_count" -eq 0 ] &&
     [ "$driver_strict_error_count" -eq 0 ] &&
     [ "$oracle_count" -eq 0 ]; then
    pass "full-selfhost" "strict no-fallback gate passed"
  else
    fail "full-selfhost" "failed (exit=$full_code parse_all_header=$parse_all_header_count parse_all_summary=$parse_all_summary_count suites_all=$suites_all_count suite_fail=$suite_fail_count driver_fallback=$driver_fallback_count driver_strict_error=$driver_strict_error_count rust_oracle=$oracle_count)"
  fi
fi

if [ "$FAIL_COUNT" -eq 0 ]; then
  set +e
  run_with_timeout "$PARSE_TIMEOUT_SECS" env \
    SOUNIO_SELFHOST_STRICT="$STRICT_MODE" \
    SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
    "$SOUC_BIN" run "$FULL_TARGET" -- parse-all report "$PARSE_SHARD_COUNT" "$PARSE_SHARD_MODE" >"$REPORT_LOG" 2>&1
  report_code=$?
  set -e

  report_summary_line="$(first_match '^PARSE_ALL_REPORT_SUMMARY ' "$REPORT_LOG")"
  report_summary_count="$(count_matches '^PARSE_ALL_REPORT_SUMMARY ' "$REPORT_LOG")"
  report_shard_count="$(count_matches '^PARSE_ALL_REPORT_SHARD ' "$REPORT_LOG")"
  report_mode="$(extract_kv_value "$report_summary_line" "shard_mode")"

  if [ "$report_code" -eq 0 ] &&
     [ "$report_summary_count" -gt 0 ] &&
     [ "$report_shard_count" -eq "$PARSE_SHARD_COUNT" ] &&
     [ "$report_mode" = "$PARSE_SHARD_MODE" ]; then
    pass "parse-all-report" "report emitted summary + shard lines"
  else
    fail "parse-all-report" "failed (exit=$report_code summary_count=$report_summary_count shard_lines=$report_shard_count expected_shards=$PARSE_SHARD_COUNT report_mode=$report_mode expected_mode=$PARSE_SHARD_MODE)"
  fi
fi

if [ "$FAIL_COUNT" -eq 0 ]; then
  set +e
  run_with_timeout "$PARSE_TIMEOUT_SECS" env \
    SOUNIO_SELFHOST_STRICT="$STRICT_MODE" \
    SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR="$DRIVER_HARNESS_CACHE_DIR" \
    "$SOUC_BIN" run "$FULL_TARGET" -- parse-all shards "$PARSE_SHARD_COUNT" "$PARSE_SHARD_SELECTOR" "$PARSE_SHARD_MODE" >"$SHARDS_LOG" 2>&1
  shards_code=$?
  set -e

  multi_summary_line="$(first_match '^PARSE_ALL_MULTI_SUMMARY ' "$SHARDS_LOG")"
  summary_count="$(count_matches '^PARSE_ALL_SUMMARY ' "$SHARDS_LOG")"
  fallback_count_shards="$(count_matches 'SELFHOST=driver-first schema=v1 event=driver_orchestration .* status=fallback' "$SHARDS_LOG")"
  strict_error_count_shards="$(count_matches 'SELFHOST=driver-first schema=v1 event=driver_orchestration .* status=strict_error' "$SHARDS_LOG")"
  oracle_count_shards="$(count_matches 'SELFHOST=oracle backend=rust' "$SHARDS_LOG")"

  executed="$(extract_kv_value "$multi_summary_line" "executed")"
  passed="$(extract_kv_value "$multi_summary_line" "passed")"
  failed_shards="$(extract_kv_value "$multi_summary_line" "failed")"
  invalid_shards="$(extract_kv_value "$multi_summary_line" "invalid")"
  summary_mode="$(extract_kv_value "$multi_summary_line" "shard_mode")"

  if [ -z "$executed" ]; then executed="-1"; fi
  if [ -z "$passed" ]; then passed="-1"; fi
  if [ -z "$failed_shards" ]; then failed_shards="-1"; fi
  if [ -z "$invalid_shards" ]; then invalid_shards="-1"; fi
  if [ -z "$summary_mode" ]; then summary_mode=""; fi

  if [ "$shards_code" -eq 0 ] &&
     [ "$summary_count" -gt 0 ] &&
     [ "$executed" -gt 0 ] &&
     [ "$executed" -eq "$passed" ] &&
     [ "$failed_shards" -eq 0 ] &&
     [ "$invalid_shards" -eq 0 ] &&
     [ "$summary_mode" = "$PARSE_SHARD_MODE" ] &&
     [ "$fallback_count_shards" -eq 0 ] &&
     [ "$strict_error_count_shards" -eq 0 ] &&
     [ "$oracle_count_shards" -eq 0 ]; then
    pass "parse-all-shards" "full-corpus shard selector completed"
  else
    fail "parse-all-shards" "failed (exit=$shards_code summary_count=$summary_count executed=$executed passed=$passed failed=$failed_shards invalid=$invalid_shards shard_mode=$summary_mode expected_mode=$PARSE_SHARD_MODE driver_fallback=$fallback_count_shards driver_strict_error=$strict_error_count_shards rust_oracle=$oracle_count_shards)"
  fi
fi

{
  echo "summary_pass=$PASS_COUNT"
  echo "summary_fail=$FAIL_COUNT"
  echo "full_timeout_secs=$FULL_TIMEOUT_SECS"
  echo "parse_timeout_secs=$PARSE_TIMEOUT_SECS"
  echo "parse_shard_count=$PARSE_SHARD_COUNT"
  echo "parse_shard_selector=$PARSE_SHARD_SELECTOR"
  echo "parse_shard_mode=$PARSE_SHARD_MODE"
  echo "full_log=$FULL_LOG"
  echo "prewarm_log=$PREWARM_LOG"
  echo "report_log=$REPORT_LOG"
  echo "shards_log=$SHARDS_LOG"
} >"$SUMMARY_FILE"

echo "SELFHOST_ZERO_GATE_SUMMARY pass=$PASS_COUNT fail=$FAIL_COUNT artifacts=$WORK_DIR"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0
