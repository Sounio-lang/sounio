#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/bin/souc"
FAIL_LOG="/tmp/sounio_examples_failures.txt"

if [[ ! -x "$BIN" ]]; then
  echo "error: missing wrapper at $BIN" >&2
  exit 1
fi

: > "$FAIL_LOG"
PASS=0
FAIL=0

while IFS= read -r file; do
  if "$BIN" check "$file" > /tmp/sounio_check_one.log 2>&1; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
    {
      echo "==== FAIL: $file ===="
      cat /tmp/sounio_check_one.log
      echo
    } >> "$FAIL_LOG"
  fi
done < <(find "$ROOT_DIR/examples" -type f -name '*.sio' -print | sort)

echo "PASS=$PASS FAIL=$FAIL"
if [[ $FAIL -ne 0 ]]; then
  echo "Failure log: $FAIL_LOG"
  exit 1
fi
