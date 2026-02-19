#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <log-file> [<log-file> ...]" >&2
  exit 2
fi

for log_file in "$@"; do
  if [ ! -f "$log_file" ]; then
    echo "error: missing log file: $log_file" >&2
    exit 2
  fi
  if [ ! -r "$log_file" ]; then
    echo "error: unreadable log file: $log_file" >&2
    exit 2
  fi
done

count_pattern_in_file() {
  local pattern="$1"
  local file="$2"
  local rc=0
  local count=0

  if command -v rg >/dev/null 2>&1; then
    set +e
    count="$(rg --no-messages --count "$pattern" "$file")"
    rc=$?
    set -e
  else
    set +e
    count="$(grep -E -c "$pattern" "$file")"
    rc=$?
    set -e
  fi

  case "$rc" in
    0) ;;
    1) count=0 ;;
    *)
      echo "error: failed to scan log file: $file" >&2
      exit 3
      ;;
  esac

  echo "$count"
}

count_pattern_matches() {
  local pattern="$1"
  shift
  local total=0
  local count=0
  local file=""

  for file in "$@"; do
    count="$(count_pattern_in_file "$pattern" "$file")"
    total=$((total + count))
  done

  echo "$total"
}

driver_fallback_count="$(count_pattern_matches 'SELFHOST=driver-first schema=v1 event=driver_orchestration\b.*\bstatus=fallback\b' "$@")"
oracle_count="$(count_pattern_matches 'SELFHOST=oracle backend=rust' "$@")"
rust_fallback_count="$(count_pattern_matches 'SELFHOST=fallback backend=rust' "$@")"

echo "NO_RUST_MARKERS driver_fallback=$driver_fallback_count rust_oracle=$oracle_count rust_fallback=$rust_fallback_count"

if [ "$driver_fallback_count" -eq 0 ] && [ "$oracle_count" -eq 0 ] && [ "$rust_fallback_count" -eq 0 ]; then
  echo "NO_RUST_MARKERS_RESULT PASS"
  exit 0
fi

echo "NO_RUST_MARKERS_RESULT FAIL" >&2
exit 1
