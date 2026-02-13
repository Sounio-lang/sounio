#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"
MATRIX_FILE="${MATRIX_FILE:-scripts/poseidon_compat_matrix.txt}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-poseidon-gate}"
LOG_DIR="$WORK_DIR/logs"
ARTIFACT_DIR="$WORK_DIR/artifacts"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"
BUILD_SOUC="${BUILD_SOUC:-1}"
SELFHOST_MODE="${SOUNIO_SELFHOST_MODE:-vm}"
SKIP_FULL_SELFHOST="${POSEIDON_SKIP_FULL_SELFHOST:-0}"
RUN_ORACLE_PARITY="${POSEIDON_ORACLE_PARITY:-0}"
BASELINE_COMPARE="${POSEIDON_BASELINE_COMPARE:-1}"
BASELINE_COMMIT="${POSEIDON_BASELINE_COMMIT:-01ecf01}"
BASELINE_WORKTREE="$WORK_DIR/baseline-wt"
BASELINE_SOUC="$BASELINE_WORKTREE/target/debug/souc"
SELFHOST_NO_RUST_FALLBACK="${SOUNIO_SELFHOST_NO_RUST_FALLBACK:-1}"
FULL_SELFHOST_TARGET="${FULL_SELFHOST_TARGET:-self-hosted/}"

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
  else
    "$@"
  fi
}

flag_enabled() {
  local raw="${1:-}"
  raw="${raw,,}"
  case "$raw" in
    1|true|yes|on)
      return 0
      ;;
  esac
  return 1
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

marker_value() {
  local marker_file="$1"
  local key="$2"
  local value

  if [ ! -f "$marker_file" ]; then
    echo "0"
    return 0
  fi

  value="$(awk -F= -v key="$key" '$1 == key { print $2; found=1; exit } END { if (!found) print "0" }' "$marker_file")"
  if [[ ! "$value" =~ ^-?[0-9]+$ ]]; then
    value="0"
  fi
  echo "$value"
}

run_full_selfhost() {
  local souc_bin="$1"
  local log_file="$2"

  run_with_timeout "$TIMEOUT_SECS" env \
    SOUNIO_SELFHOST_MODE="$SELFHOST_MODE" \
    SOUNIO_SELFHOST_NO_RUST_FALLBACK="$SELFHOST_NO_RUST_FALLBACK" \
    "$souc_bin" run "$FULL_SELFHOST_TARGET" >"$log_file" 2>&1
}

run_capture() {
  local out_file="$1"
  local err_file="$2"
  local code_file="$3"
  local env_prefix="$4"
  local cmd="$5"

  set +e
  if [ -n "$env_prefix" ]; then
    bash -lc "$env_prefix \"$SOUC_BIN\" $cmd" >"$out_file" 2>"$err_file"
  else
    bash -lc "\"$SOUC_BIN\" $cmd" >"$out_file" 2>"$err_file"
  fi
  local code=$?
  set -e

  echo "$code" >"$code_file"
}

compare_case() {
  local case_id="$1"
  local compare_spec="$2"
  local base_prefix="$3"
  local cand_prefix="$4"

  IFS=',' read -r -a checks <<<"$compare_spec"
  for check in "${checks[@]}"; do
    case "$check" in
      exit)
        if ! cmp -s "$base_prefix.exit" "$cand_prefix.exit"; then
          fail "$case_id" "exit code mismatch"
          return 1
        fi
        ;;
      stdout)
        if ! cmp -s "$base_prefix.stdout" "$cand_prefix.stdout"; then
          fail "$case_id" "stdout mismatch"
          return 1
        fi
        ;;
      stderr)
        if ! cmp -s "$base_prefix.stderr" "$cand_prefix.stderr"; then
          fail "$case_id" "stderr mismatch"
          return 1
        fi
        ;;
      *)
        fail "$case_id" "unknown compare token '$check'"
        return 1
        ;;
    esac
  done

  pass "$case_id" "baseline and candidate match ($compare_spec)"
}

extract_full_selfhost_markers() {
  local log_file="$1"
  local marker_file="$2"
  local exit_code="$3"

  local parse_all_header_count
  local parse_all_summary_count
  local parse_all_count
  local all_passed_count
  local suite_fail_detail_count
  local suite_fail_summary_count
  local suite_fail_count
  local rust_fallback_count
  local rust_oracle_count

  parse_all_header_count="$(count_matches '=== Parse All ===' "$log_file")"
  parse_all_summary_count="$(count_matches 'Parse All:' "$log_file")"
  parse_all_count=$((parse_all_header_count + parse_all_summary_count))
  all_passed_count="$(count_matches 'Suites: all passed' "$log_file")"
  suite_fail_detail_count="$(count_matches 'SUITE FAIL:' "$log_file")"
  suite_fail_summary_count="$(count_matches 'Suites: suite fail' "$log_file")"
  suite_fail_count=$((suite_fail_detail_count + suite_fail_summary_count))
  rust_fallback_count="$(count_matches 'SELFHOST=driver-first schema=v1 event=driver_orchestration .* status=fallback' "$log_file")"
  rust_oracle_count="$(count_matches 'SELFHOST=oracle backend=rust' "$log_file")"

  {
    echo "exit=$exit_code"
    echo "parse_all=$parse_all_count"
    echo "parse_all_header=$parse_all_header_count"
    echo "parse_all_summary=$parse_all_summary_count"
    echo "all_passed=$all_passed_count"
    echo "suite_fail=$suite_fail_count"
    echo "suite_fail_detail=$suite_fail_detail_count"
    echo "suite_fail_summary=$suite_fail_summary_count"
    echo "rust_fallback=$rust_fallback_count"
    echo "rust_oracle=$rust_oracle_count"
  } >"$marker_file"
}

is_full_selfhost_success() {
  local marker_file="$1"
  local exit_code
  local parse_all_header_count
  local parse_all_summary_count
  local parse_all_count
  local all_passed_count
  local suite_fail_count
  local rust_fallback_count
  local rust_oracle_count

  exit_code="$(marker_value "$marker_file" "exit")"
  parse_all_count="$(marker_value "$marker_file" "parse_all")"
  parse_all_header_count="$(marker_value "$marker_file" "parse_all_header")"
  parse_all_summary_count="$(marker_value "$marker_file" "parse_all_summary")"
  all_passed_count="$(marker_value "$marker_file" "all_passed")"
  suite_fail_count="$(marker_value "$marker_file" "suite_fail")"
  rust_fallback_count="$(marker_value "$marker_file" "rust_fallback")"
  rust_oracle_count="$(marker_value "$marker_file" "rust_oracle")"

  if [ "$exit_code" -ne 0 ]; then
    return 1
  fi

  # Keep marker checks strict to avoid false positives from partial output.
  if [ "$parse_all_count" -le 0 ] || [ "$parse_all_header_count" -le 0 ] || [ "$parse_all_summary_count" -le 0 ]; then
    return 1
  fi

  if [ "$all_passed_count" -le 0 ] || [ "$suite_fail_count" -ne 0 ]; then
    return 1
  fi

  if flag_enabled "$SELFHOST_NO_RUST_FALLBACK"; then
    if [ "$rust_fallback_count" -ne 0 ] || [ "$rust_oracle_count" -ne 0 ]; then
      return 1
    fi
  fi

  return 0
}

classify_full_selfhost_failure() {
  local current_marker_file="$1"
  local baseline_log="$LOG_DIR/full_selfhost.baseline.log"
  local baseline_marker_file="$LOG_DIR/full_selfhost.baseline.markers"

  if [ "$BASELINE_COMPARE" != "1" ]; then
    echo "classification=skipped reason=POSEIDON_BASELINE_COMPARE=$BASELINE_COMPARE"
    return 0
  fi

  if ! git cat-file -e "${BASELINE_COMMIT}^{commit}" 2>/dev/null; then
    echo "classification=unknown reason=missing_commit commit=$BASELINE_COMMIT"
    return 0
  fi

  if [ -d "$BASELINE_WORKTREE" ]; then
    git worktree remove --force "$BASELINE_WORKTREE" >/dev/null 2>&1 || rm -rf "$BASELINE_WORKTREE"
  fi

  if ! git worktree add --detach "$BASELINE_WORKTREE" "$BASELINE_COMMIT" >"$LOG_DIR/baseline_worktree.stdout" 2>"$LOG_DIR/baseline_worktree.stderr"; then
    echo "classification=unknown reason=worktree_add_failed commit=$BASELINE_COMMIT"
    return 0
  fi

  if ! run_with_timeout 900 bash -lc "cd \"$BASELINE_WORKTREE\" && cargo build -p souc" >"$LOG_DIR/baseline_build.stdout" 2>"$LOG_DIR/baseline_build.stderr"; then
    echo "classification=unknown reason=baseline_build_failed commit=$BASELINE_COMMIT"
    git worktree remove --force "$BASELINE_WORKTREE" >/dev/null 2>&1 || true
    return 0
  fi

  set +e
  run_full_selfhost "$BASELINE_SOUC" "$baseline_log"
  local baseline_exit=$?
  set -e
  extract_full_selfhost_markers "$baseline_log" "$baseline_marker_file" "$baseline_exit"

  local current_success=0
  local baseline_success=0
  if is_full_selfhost_success "$current_marker_file"; then
    current_success=1
  fi
  if is_full_selfhost_success "$baseline_marker_file"; then
    baseline_success=1
  fi

  git worktree remove --force "$BASELINE_WORKTREE" >/dev/null 2>&1 || true

  if [ "$current_success" = "1" ]; then
    echo "classification=none reason=current_passed"
    return 0
  fi

  if [ "$baseline_success" = "1" ]; then
    echo "classification=branch_regression baseline_commit=$BASELINE_COMMIT"
    return 0
  fi

  local current_exit current_parse current_parse_header current_parse_summary current_all current_fail current_rust_fallback current_rust_oracle
  local baseline_exit_v baseline_parse baseline_parse_header baseline_parse_summary baseline_all baseline_fail baseline_rust_fallback baseline_rust_oracle
  current_exit="$(marker_value "$current_marker_file" "exit")"
  current_parse="$(marker_value "$current_marker_file" "parse_all")"
  current_parse_header="$(marker_value "$current_marker_file" "parse_all_header")"
  current_parse_summary="$(marker_value "$current_marker_file" "parse_all_summary")"
  current_all="$(marker_value "$current_marker_file" "all_passed")"
  current_fail="$(marker_value "$current_marker_file" "suite_fail")"
  current_rust_fallback="$(marker_value "$current_marker_file" "rust_fallback")"
  current_rust_oracle="$(marker_value "$current_marker_file" "rust_oracle")"
  baseline_exit_v="$(marker_value "$baseline_marker_file" "exit")"
  baseline_parse="$(marker_value "$baseline_marker_file" "parse_all")"
  baseline_parse_header="$(marker_value "$baseline_marker_file" "parse_all_header")"
  baseline_parse_summary="$(marker_value "$baseline_marker_file" "parse_all_summary")"
  baseline_all="$(marker_value "$baseline_marker_file" "all_passed")"
  baseline_fail="$(marker_value "$baseline_marker_file" "suite_fail")"
  baseline_rust_fallback="$(marker_value "$baseline_marker_file" "rust_fallback")"
  baseline_rust_oracle="$(marker_value "$baseline_marker_file" "rust_oracle")"

  echo "classification=pre_existing baseline_commit=$BASELINE_COMMIT current(exit=$current_exit parse_all=$current_parse parse_all_header=$current_parse_header parse_all_summary=$current_parse_summary all_passed=$current_all suite_fail=$current_fail rust_fallback=$current_rust_fallback rust_oracle=$current_rust_oracle) baseline(exit=$baseline_exit_v parse_all=$baseline_parse parse_all_header=$baseline_parse_header parse_all_summary=$baseline_parse_summary all_passed=$baseline_all suite_fail=$baseline_fail rust_fallback=$baseline_rust_fallback rust_oracle=$baseline_rust_oracle)"
}

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

if [ "$BUILD_SOUC" = "1" ]; then
  if run_with_timeout 900 cargo build -p souc >"$LOG_DIR/build.stdout" 2>"$LOG_DIR/build.stderr"; then
    pass "build" "cargo build -p souc"
  else
    fail "build" "cargo build failed (see $LOG_DIR/build.stderr)"
  fi
fi

if [ ! -x "$SOUC_BIN" ]; then
  fail "preflight" "compiler binary not found at $SOUC_BIN"
  echo
  echo "Summary: PASS=$PASS_COUNT FAIL=$FAIL_COUNT"
  exit 1
fi

if [ ! -f "$MATRIX_FILE" ]; then
  fail "preflight" "matrix file missing: $MATRIX_FILE"
  echo
  echo "Summary: PASS=$PASS_COUNT FAIL=$FAIL_COUNT"
  exit 1
fi

while IFS='|' read -r case_id mode command compare_spec; do
  if [[ -z "${case_id:-}" ]] || [[ "$case_id" =~ ^# ]]; then
    continue
  fi

  base_prefix="$ARTIFACT_DIR/${case_id}.baseline"
  cand_prefix="$ARTIFACT_DIR/${case_id}.candidate"

  case "$mode" in
    run_selfhost|run_default_vs_compat)
      common_selfhost_env="SOUNIO_SELFHOST_MODE=$SELFHOST_MODE SOUNIO_SELFHOST_NO_RUST_FALLBACK=$SELFHOST_NO_RUST_FALLBACK"
      run_capture \
        "$base_prefix.stdout" \
        "$base_prefix.stderr" \
        "$base_prefix.exit" \
        "$common_selfhost_env" \
        "$command"

      run_capture \
        "$cand_prefix.stdout" \
        "$cand_prefix.stderr" \
        "$cand_prefix.exit" \
        "$common_selfhost_env" \
        "$command --use-sounio-compiler"
      ;;
    direct)
      run_capture \
        "$base_prefix.stdout" \
        "$base_prefix.stderr" \
        "$base_prefix.exit" \
        "" \
        "$command"

      run_capture \
        "$cand_prefix.stdout" \
        "$cand_prefix.stderr" \
        "$cand_prefix.exit" \
        "" \
        "$command"
      ;;
    *)
      fail "$case_id" "unknown mode '$mode'"
      continue
      ;;
  esac

  compare_case "$case_id" "$compare_spec" "$base_prefix" "$cand_prefix" || true
done <"$MATRIX_FILE"

if [ "$RUN_ORACLE_PARITY" = "1" ]; then
  while IFS='|' read -r case_id mode command compare_spec; do
    if [[ -z "${case_id:-}" ]] || [[ "$case_id" =~ ^# ]]; then
      continue
    fi

    if [ "$mode" != "run_selfhost" ] && [ "$mode" != "run_default_vs_compat" ]; then
      continue
    fi

    base_prefix="$ARTIFACT_DIR/oracle_${case_id}.baseline"
    cand_prefix="$ARTIFACT_DIR/oracle_${case_id}.candidate"

    run_capture \
      "$base_prefix.stdout" \
      "$base_prefix.stderr" \
      "$base_prefix.exit" \
      "SOUNIO_SELFHOST_MODE=$SELFHOST_MODE SOUNIO_SELFHOST_NO_RUST_FALLBACK=$SELFHOST_NO_RUST_FALLBACK" \
      "$command"

    run_capture \
      "$cand_prefix.stdout" \
      "$cand_prefix.stderr" \
      "$cand_prefix.exit" \
      "SOUNIO_SELFHOST_MODE=$SELFHOST_MODE SOUNIO_SELFHOST_NO_RUST_FALLBACK=0 SOUNIO_SELFHOST_PIPELINE=rust" \
      "$command --use-sounio-compiler"

    compare_case "oracle-$case_id" "exit,stdout" "$base_prefix" "$cand_prefix" || true
  done <"$MATRIX_FILE"
fi

if [ "$SKIP_FULL_SELFHOST" != "1" ]; then
  FULL_LOG="$LOG_DIR/full_selfhost.log"
  FULL_MARKERS="$LOG_DIR/full_selfhost.markers"
  set +e
  run_full_selfhost "$SOUC_BIN" "$FULL_LOG"
  full_code=$?
  set -e
  extract_full_selfhost_markers "$FULL_LOG" "$FULL_MARKERS" "$full_code"

  parse_all_count="$(marker_value "$FULL_MARKERS" "parse_all")"
  parse_all_header_count="$(marker_value "$FULL_MARKERS" "parse_all_header")"
  parse_all_summary_count="$(marker_value "$FULL_MARKERS" "parse_all_summary")"
  all_passed_count="$(marker_value "$FULL_MARKERS" "all_passed")"
  suite_fail_count="$(marker_value "$FULL_MARKERS" "suite_fail")"
  rust_fallback_count="$(marker_value "$FULL_MARKERS" "rust_fallback")"
  rust_oracle_count="$(marker_value "$FULL_MARKERS" "rust_oracle")"

  if is_full_selfhost_success "$FULL_MARKERS"; then
    pass "full-selfhost" "self-hosted suite completed with required markers (no_rust_fallback=$SELFHOST_NO_RUST_FALLBACK)"
  else
    fail "full-selfhost" "gate failed (exit=$full_code parse_all=$parse_all_count parse_all_header=$parse_all_header_count parse_all_summary=$parse_all_summary_count all_passed=$all_passed_count suite_fail=$suite_fail_count rust_fallback=$rust_fallback_count rust_oracle=$rust_oracle_count no_rust_fallback=$SELFHOST_NO_RUST_FALLBACK)"
    classification="$(classify_full_selfhost_failure "$FULL_MARKERS")"
    if [ -n "$classification" ]; then
      echo "INFO [full-selfhost] $classification"
    fi
  fi
else
  pass "full-selfhost" "skipped by POSEIDON_SKIP_FULL_SELFHOST=1"
fi

echo
echo "Summary: PASS=$PASS_COUNT FAIL=$FAIL_COUNT"
echo "Artifacts: $WORK_DIR"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0
