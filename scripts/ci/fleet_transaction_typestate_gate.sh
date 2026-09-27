#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUNIO_FLEET_TYPESTATE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_FLEET_TYPESTATE_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/fleet_transaction.sio"
WITNESS="$ROOT_DIR/tests/compiler/fleet_transaction_typestate_check.sio"
PRIVACY="$ROOT_DIR/tests/compiler/fleet_transaction_privacy"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-fleet-typestate.XXXXXX")"

cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'fleet-transaction-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_rejection() {
  local label="$1" source="$2" code="$3" rc=0
  local log="$WORK/$label.log"
  set +e
  SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" == 1 ]] || {
    cat "$log" >&2
    fail "$label must reject with rc=1, got rc=$rc"
  }
  local count
  count="$(rg -c "error\[$code" "$log")"
  if [[ "$code" == E039 || "$code" == E175 ]]; then
    [[ "$count" -ge 1 ]] || {
      cat "$log" >&2
      fail "$label must emit at least one $code"
    }
    return
  fi
  [[ "$count" == 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one $code"
  }
}

expect_composed_rejection() {
  local label="$1" source="$2" code="$3"
  local program="$WORK/$label.sio"
  compose_source "$source" "$program"
  expect_rejection "$label" "$program" "$code"
}

compose_source() {
  local source="$1" output="$2"
  {
    cat "$MODULE"
    awk '
      /^use coordination::fleet_transaction::\{/ { skip=1; next }
      skip && /^\}/ { skip=0; next }
      !skip { print }
    ' "$source"
  } > "$output"
}

expect_runtime_refusal() {
  local label="$1" source="$2" rc=0
  local program="$WORK/$label.sio" log="$WORK/$label-runtime.log"
  compose_source "$source" "$program"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" check "$program" \
    >"$WORK/$label-check.log" 2>&1 || {
    cat "$WORK/$label-check.log" >&2
    fail "$label sabotage control did not typecheck"
  }
  set +e
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$program" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" != 0 ]] || {
    cat "$log" >&2
    fail "$label must refuse at runtime"
  }
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"

compose_source "$WITNESS" "$WORK/fleet_transaction_kernel.sio"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$WORK/fleet_transaction_kernel.sio" \
  >"$WORK/runtime.log" 2>&1 || {
  cat "$WORK/runtime.log" >&2
  fail 'single-module linear typestate witness did not run'
}
rg -Fxq \
  'fleet-transaction-typestate: PASS linear=1 argv=1 checkpoint=1 handoff=1' \
  "$WORK/runtime.log" || {
    cat "$WORK/runtime.log" >&2
    fail 'linear typestate witness omitted its exact receipt'
  }

expect_rejection private-constructor \
  "$PRIVACY/fleet_transaction_private_struct_main.sio" E176
expect_rejection unsealed-host-admission \
  "$PRIVACY/fleet_transaction_unsealed_admission_main.sio" E175
expect_composed_rejection wrong-state \
  "$PRIVACY/fleet_transaction_wrong_state_main.sio" E009
expect_composed_rejection linear-reuse \
  "$PRIVACY/fleet_transaction_linear_reuse_main.sio" E039

expect_runtime_refusal argv-binding \
  "$PRIVACY/fleet_transaction_argv_mismatch.sio" \
  'argv attestation does not match consumed start authority'
expect_runtime_refusal evidence-binding \
  "$PRIVACY/fleet_transaction_evidence_mismatch.sio" \
  'evidence verification does not match checkpoint draft'
expect_runtime_refusal recipient-binding \
  "$PRIVACY/fleet_transaction_recipient_mismatch.sio" \
  'anchor or recipient verification does not match prepared handoff'
expect_runtime_refusal anchor-prefix-binding \
  "$PRIVACY/fleet_transaction_anchor_prefix_mismatch.sio" \
  'anchor or recipient verification does not match prepared handoff'

echo "fleet-transaction-typestate: PASS positive_engine=$ENGINE negative_engine=madaros host_bridge=sealed-E175 runtime=1 private=E176 wrong-state=E009 linear-reuse=E039 argv-binding=refused evidence-binding=refused recipient-binding=refused anchor-prefix-binding=refused"
