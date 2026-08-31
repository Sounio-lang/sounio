#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUNIO_LOOM_OBLIGATION_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_OBLIGATION_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/loom_obligation.sio"
WITNESS="$ROOT_DIR/tests/compiler/loom_obligation_typestate_check.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-obligation-typestate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-obligation-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

compose_source() {
  local source="$1" output="$2"
  {
    cat "$MODULE"
    awk '
      /^use coordination::loom_obligation::\{/ { skip=1; next }
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
  [[ "$rc" -eq 1 ]] || { cat "$log" >&2; fail "$label rc=$rc"; }
  rg -q "error\[$code" "$log" || { cat "$log" >&2; fail "$label omitted $code"; }
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"
combined="$WORK/loom_obligation_typestate.sio"
compose_source "$WITNESS" "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" >"$WORK/runtime.log" 2>&1 || {
  cat "$WORK/runtime.log" >&2
  fail 'linear typestate witness did not run'
}
rg -Fxq 'loom-obligation-typestate: PASS host_seal=private linear=1 recovery=1 exclusive_claim=1 evidence_bound=1 zero_evidence=refused' "$WORK/runtime.log" || {
  cat "$WORK/runtime.log" >&2
  fail 'linear typestate witness omitted its exact receipt'
}

expect_rejection private-host-receipt \
  "$ROOT_DIR/tests/compile-fail/loom_obligation_host_receipt_private.sio" E176

wrong_state="$WORK/wrong_state.sio"
compose_source "$ROOT_DIR/tests/compile-fail/loom_obligation_consumed_as_claim.sio" "$wrong_state"
expect_rejection consumed-as-claim "$wrong_state" E009

reuse="$WORK/reuse.sio"
compose_source "$ROOT_DIR/tests/compile-fail/loom_obligation_claim_reuse.sio" "$reuse"
expect_rejection claim-reuse "$reuse" E039

printf 'loom-obligation-typestate: PASS privacy=E176 role=E009 linear=E039\n'
