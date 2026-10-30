#!/usr/bin/env bash
# Stdlib E2E test runner
#
# Runs all .sio tests in tests/stdlib/**/ using the souc binary.
# Each test imports from the real stdlib to validate E2E correctness.
#
# Annotations:
#   //@ run-pass                expect souc check and runtime execution
#   //@ check-only              only run souc check, not souc run
#   //@ expect-stdout: X        stdout must contain X (run-pass only, optional)
#   //@ ignore [reason]         skip this test
#   //@ ignore-owner: TEAM      optional owner metadata for ignored tests
#   //@ ignore-unblock: TEXT    optional unblock condition for ignored tests
#   //@ contract-adjustment: X  tracks rewritten active-contract tests
#   //@ timeout: SECONDS        run timeout for this test (default
#                               STDLIB_E2E_RUN_TIMEOUT_SEC, 30)
#   //@ requires: slow          skipped unless SOUNIO_SLOW_TESTS_AVAILABLE is set;
#                               the nightly slow lane in ci.yml runs these
#
# The last two match scripts/dev/run_sio_test_suite_v2.sh. This runner used to
# ignore them, so 19 of the 24 tests it reported as run-timeout on 2026-10-25
# had been killed at 30 s although they declare 60 s to 3600 s.
#
# Usage:
#   bash scripts/stdlib/run_stdlib_e2e.sh [--filter PATTERN] [--verbose] [--json-out PATH]
#   bash scripts/stdlib/run_stdlib_e2e.sh PATTERN --verbose

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

# Stdlib path is mandatory for E2E tests.
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

FILTER=""
VERBOSE=0
JSON_OUT="${STDLIB_E2E_JSON_OUT:-}"
RUN_TIMEOUT_SEC="${STDLIB_E2E_RUN_TIMEOUT_SEC:-30}"
RUN_ALL_RUN_PASS="${STDLIB_E2E_RUN_ALL_RUN_PASS:-1}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/stdlib/run_stdlib_e2e.sh [--filter PATTERN] [--verbose] [--json-out PATH]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --filter)
      if [[ $# -lt 2 ]]; then
        echo "error: --filter requires a value" >&2
        exit 2
      fi
      FILTER="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    --json-out)
      if [[ $# -lt 2 ]]; then
        echo "error: --json-out requires a path" >&2
        exit 2
      fi
      JSON_OUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -* )
      echo "error: unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -z "$FILTER" ]]; then
        FILTER="$1"
        shift
      else
        echo "error: unexpected argument: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

PASS=0
FAIL=0
SKIP=0
ERRORS=""

RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT

b64_encode() {
  printf '%s' "$1" | base64 | tr -d '\n'
}

record_result() {
  local status="$1"
  local relpath="$2"
  local mode="$3"
  local reason="$4"
  local owner="$5"
  local unblock="$6"
  local exit_code="$7"
  local detail="$8"
  local detail_b64
  detail_b64="$(b64_encode "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$status" "$relpath" "$mode" "$reason" "$owner" "$unblock" "$exit_code" "$detail_b64" >> "$RESULTS_FILE"
}

run_test() {
  local file="$1"
  local relpath="${file#$ROOT_DIR/}"

  if [[ -n "$FILTER" && "$relpath" != *"$FILTER"* ]]; then
    return
  fi

  local is_run_pass=false
  local is_check_only=false
  local is_ignored=false
  local ignore_reason=""
  local ignore_owner=""
  local ignore_unblock=""
  local expect_stdout=()
  local test_timeout="$RUN_TIMEOUT_SEC"
  local requires_slow=false

  while IFS= read -r line; do
    if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//([[:space:]]|$) && ! "$line" =~ ^[[:space:]]*$ ]]; then
      break
    fi
    # Only a line that starts with `//@ ` is an annotation; the =~ checks
    # below match by substring, so prose such as "NOT `//@ check-only`" in a
    # header comment would otherwise count as one.
    [[ "$line" =~ ^[[:space:]]*//@\  ]] || continue

    if [[ "$line" =~ "//@ run-pass" ]]; then
      is_run_pass=true
    fi
    if [[ "$line" =~ "//@ check-only" ]]; then
      is_check_only=true
    fi
    if [[ "$line" =~ ^[[:space:]]*//@[[:space:]]*ignore-owner:[[:space:]]*(.*)$ ]]; then
      ignore_owner="${BASH_REMATCH[1]}"
    fi
    if [[ "$line" =~ ^[[:space:]]*//@[[:space:]]*ignore-unblock:[[:space:]]*(.*)$ ]]; then
      ignore_unblock="${BASH_REMATCH[1]}"
    fi
    if [[ "$line" =~ ^[[:space:]]*//@[[:space:]]*ignore([[:space:]].*)?$ ]]; then
      is_ignored=true
      ignore_reason="$(echo "${BASH_REMATCH[1]:-}" | sed 's/^[[:space:]]*//')"
      if [[ -z "$ignore_reason" ]]; then
        ignore_reason="ignored"
      fi
    fi
    if [[ "$line" =~ "//@ expect-stdout: "(.*) ]]; then
      expect_stdout+=("${BASH_REMATCH[1]}")
    fi
    if [[ "$line" =~ ^[[:space:]]*//@[[:space:]]*timeout:[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
      test_timeout="${BASH_REMATCH[1]}"
    fi
    if [[ "$line" =~ ^[[:space:]]*//@[[:space:]]*requires:[[:space:]]*slow[[:space:]]*$ ]]; then
      requires_slow=true
    fi
  done < "$file"

  if $requires_slow && [[ -z "${SOUNIO_SLOW_TESTS_AVAILABLE:-}" ]]; then
    SKIP=$((SKIP + 1))
    record_result "skip" "$relpath" "requires" "requires:slow" "" "set SOUNIO_SLOW_TESTS_AVAILABLE=1 (nightly slow lane)" "0" ""
    if [[ $VERBOSE -eq 1 ]]; then
      echo "  SKIP  $relpath (requires: slow)"
    fi
    return
  fi

  if $is_ignored; then
    SKIP=$((SKIP + 1))
    record_result "skip" "$relpath" "ignore" "$ignore_reason" "$ignore_owner" "$ignore_unblock" "0" ""
    if [[ $VERBOSE -eq 1 ]]; then
      echo "  SKIP  $relpath"
    fi
    return
  fi

  if ! $is_run_pass && ! $is_check_only; then
    SKIP=$((SKIP + 1))
    record_result "skip" "$relpath" "no-annotation" "no-annotation" "" "" "0" ""
    return
  fi

  local output
  local exit_code=0
  output=$("$SOUC_BIN" check "$file" 2>&1) || exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    local excerpt
    excerpt="$(head -6 <<< "$output")"
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}\n  FAIL  $relpath (check exited $exit_code)"
    record_result "fail" "$relpath" "check" "check-failed" "" "" "$exit_code" "$excerpt"
    if [[ $VERBOSE -eq 1 ]]; then
      echo "  FAIL  $relpath (check exited $exit_code)"
      echo "        $(echo "$excerpt" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
    fi
    return
  fi

  if $is_check_only; then
    PASS=$((PASS + 1))
    record_result "pass" "$relpath" "check-only" "" "" "" "0" ""
    if [[ $VERBOSE -eq 1 ]]; then
      echo "  PASS  $relpath (check-only)"
    fi
    return
  fi

  if [[ ${#expect_stdout[@]} -eq 0 && "$RUN_ALL_RUN_PASS" != "1" ]]; then
    PASS=$((PASS + 1))
    record_result "pass" "$relpath" "run-pass-check-only" "no-expect-stdout" "" "" "0" ""
    if [[ $VERBOSE -eq 1 ]]; then
      echo "  PASS  $relpath (run-pass check-only fallback)"
    fi
    return
  fi

  exit_code=0
  if command -v timeout >/dev/null 2>&1; then
    output=$(timeout "${test_timeout}s" "$SOUC_BIN" run "$file" 2>&1) || exit_code=$?
  else
    output=$("$SOUC_BIN" run "$file" 2>&1) || exit_code=$?
  fi
  if [[ $exit_code -ne 0 ]]; then
    local excerpt
    excerpt="$(head -6 <<< "$output")"
    FAIL=$((FAIL + 1))
    local reason="run-failed"
    if [[ $exit_code -eq 124 ]]; then
      reason="run-timeout"
      excerpt="timeout after ${test_timeout}s"
    fi
    ERRORS="${ERRORS}\n  FAIL  $relpath (run exited $exit_code)"
    record_result "fail" "$relpath" "run" "$reason" "" "" "$exit_code" "$excerpt"
    if [[ $VERBOSE -eq 1 ]]; then
      echo "  FAIL  $relpath (run exited $exit_code)"
    fi
    return
  fi

  local pattern
  for pattern in "${expect_stdout[@]}"; do
    if ! grep -qF -- "$pattern" <<< "$output"; then
      FAIL=$((FAIL + 1))
      ERRORS="${ERRORS}\n  FAIL  $relpath (missing stdout: $pattern)"
      record_result "fail" "$relpath" "stdout" "missing-stdout" "" "" "0" "missing: $pattern"
      if [[ $VERBOSE -eq 1 ]]; then
        echo "  FAIL  $relpath (missing stdout: $pattern)"
      fi
      return
    fi
  done

  PASS=$((PASS + 1))
  record_result "pass" "$relpath" "run-pass" "" "" "" "0" ""
  if [[ $VERBOSE -eq 1 ]]; then
    echo "  PASS  $relpath"
  fi
}

emit_json() {
  local out_path="$1"
  if [[ -z "$out_path" ]]; then
    return
  fi
  mkdir -p "$(dirname "$out_path")"

  node "$ROOT_DIR/scripts/stdlib/emit_stdlib_e2e_result.mjs" \
    "$RESULTS_FILE" "$out_path" "$PASS" "$FAIL" "$SKIP" \
    "$ROOT_DIR" "$SOUC_BIN" "$SOUNIO_STDLIB_PATH" "$FILTER"
}

echo "=== Stdlib E2E Test Suite ==="
echo "Using souc: $SOUC_BIN"
echo "Stdlib path: $SOUNIO_STDLIB_PATH"
echo ""

shopt -s globstar nullglob
for f in "$ROOT_DIR"/tests/stdlib/**/*.sio; do
  run_test "$f"
done

echo ""
echo "=== Results ==="
echo "  Pass: $PASS"
echo "  Fail: $FAIL"
echo "  Skip: $SKIP"
echo "  Total: $((PASS + FAIL + SKIP))"

emit_json "$JSON_OUT"

if [[ $FAIL -gt 0 ]]; then
  echo ""
  echo "=== Failures ==="
  echo -e "$ERRORS"
  exit 1
fi

echo ""
echo "All stdlib E2E tests passed."
