#!/usr/bin/env bash
# scripts/gates/g5b_repl_eval.sh — Gate for G5b: souc-native REPL.
#
# Discriminator: REPL accepts a function definition, then an expression
# that calls it, and prints the expected integer result.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"

OUT="$(printf '%s\n' \
    'fn dbl(x: i64) -> i64 { x + x }' \
    'dbl(21)' \
    ':quit' \
  | timeout 60 "$SOUC" repl 2>&1)"

if grep -qE '(^|[[:space:]>])42([[:space:]]|$)' <<<"$OUT"; then
  echo "G5b REPL eval gate: PASS"
  exit 0
else
  echo "G5b REPL eval gate: FAIL" >&2
  echo "----- transcript -----" >&2
  echo "$OUT" >&2
  exit 1
fi
