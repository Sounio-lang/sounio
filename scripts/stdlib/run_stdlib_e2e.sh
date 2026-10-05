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

  while IFS= read -r line; do
    if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//\  && ! "$line" =~ ^[[:space:]]*$ ]]; then
      break
    fi

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
  done < "$file"

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
    excerpt="$(echo "$output" | head -6)"
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
    output=$(timeout "${RUN_TIMEOUT_SEC}s" "$SOUC_BIN" run "$file" 2>&1) || exit_code=$?
  else
    output=$("$SOUC_BIN" run "$file" 2>&1) || exit_code=$?
  fi
  if [[ $exit_code -ne 0 ]]; then
    local excerpt
    excerpt="$(echo "$output" | head -6)"
    FAIL=$((FAIL + 1))
    local reason="run-failed"
    if [[ $exit_code -eq 124 ]]; then
      reason="run-timeout"
      excerpt="timeout after ${RUN_TIMEOUT_SEC}s"
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
    if ! echo "$output" | grep -qF "$pattern"; then
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

  python3 - "$RESULTS_FILE" "$out_path" "$PASS" "$FAIL" "$SKIP" "$ROOT_DIR" "$SOUC_BIN" "$SOUNIO_STDLIB_PATH" "$FILTER" <<'PY'
import base64
import datetime
import json
import pathlib
import sys

results_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
pass_count = int(sys.argv[3])
fail_count = int(sys.argv[4])
skip_count = int(sys.argv[5])
root_dir = sys.argv[6]
souc_bin = sys.argv[7]
stdlib_path = sys.argv[8]
filter_value = sys.argv[9]

def decode_field(raw: str) -> str:
    if not raw:
        return ""
    return base64.b64decode(raw.encode("ascii")).decode("utf-8", errors="replace")

results = []
ignored = []
for line in results_path.read_text(encoding="utf-8").splitlines():
    if not line:
        continue
    parts = line.split("\t")
    if len(parts) != 8:
        continue
    status, test_path, mode, reason, owner, unblock, exit_code, detail_b64 = parts
    detail = decode_field(detail_b64)
    row = {
        "status": status,
        "test_path": test_path,
        "mode": mode,
        "reason": reason,
        "owner": owner,
        "unblock_condition": unblock,
        "exit_code": int(exit_code),
        "detail_excerpt": detail,
    }
    results.append(row)
    if mode == "ignore":
        ignored.append(
            {
                "test_path": test_path,
                "reason": reason,
                "owner": owner,
                "unblock_condition": unblock,
            }
        )

obj = {
    "schema": "sounio.stdlib.e2e.result.v1",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "command": "bash scripts/stdlib/run_stdlib_e2e.sh",
    "context": {
        "root_dir": root_dir,
        "souc_bin": souc_bin,
        "stdlib_path": stdlib_path,
        "filter": filter_value,
    },
    "totals": {
        "pass": pass_count,
        "fail": fail_count,
        "skip": skip_count,
        "total": pass_count + fail_count + skip_count,
    },
    "results": results,
    "failures": [r for r in results if r["status"] == "fail"],
    "ignored": ignored,
}
out_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
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
