#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUNIO_CAUSAL_TYPESTATE_SOUC:-$ROOT_DIR/bin/souc}"
STDLIB="$ROOT_DIR/stdlib"
WITNESS="$ROOT_DIR/tests/compiler/causal_receipt_typestate_check.sio"
PRIVACY="$ROOT_DIR/tests/compiler/causal_receipt_privacy"
MODULE="$STDLIB/coordination/causal_receipt.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-causal-typestate.XXXXXX")"

cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  echo "causal-receipt-typestate: FAIL: $*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"

expect_rejection() {
  local label="$1" source="$2" code="$3" rc=0
  local log="$WORK/$label.log"
  set +e
  SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must reject with rc=1, got rc=$rc"
  }
  [[ "$(rg -c "error\[$code" "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one $code"
  }
}

compose_witness() {
  local module="$1" output="$2"
  {
    cat "$module"
    awk '
      /^use coordination::causal_receipt::\{/ { skip=1; next }
      skip && /^\}/ { skip=0; next }
      !skip { print }
    ' "$WITNESS"
  } > "$output"
}

modular_check_log="$WORK/modular-check.log"
SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$WITNESS" >"$modular_check_log" 2>&1 || {
  cat "$modular_check_log" >&2
  fail "typed causal module did not pass imported check"
}

combined="$WORK/causal_receipt_typestate_kernel.sio"
compose_witness "$MODULE" "$combined"
normal_log="$WORK/normal.log"
"$SOUC" run "$combined" >"$normal_log" 2>&1 || {
  cat "$normal_log" >&2
  fail "single-module typed causal kernel failed"
}
rg -Fxq \
  'causal-typestate: PASS supported=1 control-refusal=1 falsified=1 inconclusive=1 distinct=1 prereg-before-subject=1' \
  "$normal_log" || {
    cat "$normal_log" >&2
    fail "typed causal witness omitted its exact receipt"
  }

expect_rejection private-supported-constructor \
  "$PRIVACY/causal_receipt_private_struct_main.sio" E176
expect_rejection wrong-state-promotion \
  "$PRIVACY/causal_receipt_wrong_state_main.sio" E009

mkdir -p "$WORK/stdlib/coordination"
cp "$MODULE" "$WORK/stdlib/coordination/causal_receipt.sio"
mutation_count="$(rg -c '^    if observed\.control_pass == false \{ return None \}$' \
  "$WORK/stdlib/coordination/causal_receipt.sio")"
[[ "$mutation_count" -eq 1 ]] || \
  fail "expected one support control guard before targeted mutation, got $mutation_count"
awk '
  BEGIN { changed=0 }
  !changed && $0 == "    if observed.control_pass == false { return None }" {
    print "    if false { return None }"
    changed=1
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$WORK/stdlib/coordination/causal_receipt.sio" > "$WORK/mutated.sio" || \
  fail "could not apply the targeted support-control mutation"
mv "$WORK/mutated.sio" "$WORK/stdlib/coordination/causal_receipt.sio"

sabotage_log="$WORK/sabotage.log"
compose_witness "$WORK/stdlib/coordination/causal_receipt.sio" "$WORK/sabotage-kernel.sio"
set +e
"$SOUC" run "$WORK/sabotage-kernel.sio" >"$sabotage_log" 2>&1
sabotage_rc=$?
set -e
if [[ "$sabotage_rc" -eq 0 ]]; then
  cat "$sabotage_log" >&2
  fail "disabling the support control guard did not expose the negative witness"
fi

echo "causal-receipt-typestate: PASS modular-check=1 single-module-runtime=1 privacy=E176 wrong-state=E009 sabotage-control=1"

"$ROOT_DIR/scripts/ci/fleet_transaction_typestate_gate.sh"
