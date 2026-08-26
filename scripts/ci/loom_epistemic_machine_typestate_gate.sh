#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOUC="${SOUNIO_LOOM_EPISTEMIC_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EPISTEMIC_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/loom_epistemic_machine.sio"
WITNESS="$ROOT_DIR/tests/compiler/loom_epistemic_machine_typestate_check.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-epistemic-typestate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-epistemic-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

compose_source() {
  local source="$1" output="$2"
  {
    sed -n '1,$p' "$MODULE"
    awk '
      /^use coordination::loom_epistemic_machine::\{/ { skip=1; next }
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
combined="$WORK/loom_epistemic_typestate.sio"
compose_source "$WITNESS" "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" >"$WORK/runtime.log" 2>&1 || {
  sed -n '1,260p' "$WORK/runtime.log" >&2
  fail 'linear epistemic witness did not run'
}
rg -Fxq \
  'loom-epistemic-typestate: PASS host_seal=private knowledge_axes=5 claim=1 challenge=1 linear_capability=1 fork=1' \
  "$WORK/runtime.log" || {
  sed -n '1,260p' "$WORK/runtime.log" >&2
  fail 'linear epistemic witness omitted its exact receipt'
}

expect_rejection private-host-receipt \
  "$ROOT_DIR/tests/compile-fail/loom_epistemic_host_receipt_private.sio" E176

wrong_state="$WORK/wrong_state.sio"
compose_source "$ROOT_DIR/tests/compile-fail/loom_epistemic_claim_as_challenged.sio" \
  "$wrong_state"
expect_rejection open-claim-as-challenged "$wrong_state" E009

reuse="$WORK/reuse.sio"
compose_source "$ROOT_DIR/tests/compile-fail/loom_epistemic_capability_reuse.sio" \
  "$reuse"
expect_rejection capability-reuse "$reuse" E039

printf 'loom-epistemic-typestate: PASS privacy=E176 role=E009 linear=E039\n'
