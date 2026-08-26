#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOUC="${SOUNIO_LOOM_PORTFOLIO_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PORTFOLIO_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/loom_portfolio_attention.sio"
WITNESS="$ROOT_DIR/tests/compiler/loom_portfolio_attention_typestate_check.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-portfolio-typestate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-portfolio-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

compose_source() {
  local source="$1" output="$2"
  {
    sed -n '1,$p' "$MODULE"
    awk '
      /^use coordination::loom_portfolio_attention::\{/ { skip=1; next }
      skip && /^\}/ { skip=0; next }
      !skip { print }
    ' "$source"
  } > "$output"
}

expect_rejection() {
  local label="$1" source="$2" code="$3" rc=0
  local log="$WORK/$label.log"
  set +e
  SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || { sed -n '1,240p' "$log" >&2; fail "$label rc=$rc"; }
  rg -q "error\[$code" "$log" || {
    sed -n '1,240p' "$log" >&2
    fail "$label omitted $code"
  }
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"
combined="$WORK/loom_portfolio_typestate.sio"
compose_source "$WITNESS" "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" >"$WORK/runtime.log" 2>&1 || {
  sed -n '1,260p' "$WORK/runtime.log" >&2
  fail 'linear portfolio witness did not run'
}
rg -Fxq \
  'loom-portfolio-typestate: PASS budgets=4 linear=1 portfolio=compiled completion=typed comparator=bounded' \
  "$WORK/runtime.log" || {
  sed -n '1,260p' "$WORK/runtime.log" >&2
  fail 'linear portfolio witness omitted its exact receipt'
}

expect_rejection private-host-receipt \
  "$ROOT_DIR/tests/compile-fail/loom_portfolio_host_receipt_private.sio" E176

reuse="$WORK/reuse.sio"
compose_source "$ROOT_DIR/tests/compile-fail/loom_portfolio_budget_reuse.sio" \
  "$reuse"
expect_rejection budget-reuse "$reuse" E039

printf 'loom-portfolio-typestate: PASS privacy=E176 linear=E039 budgets=4 policies=3\n'
