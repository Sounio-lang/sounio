#!/usr/bin/env bash
# Gate: bare `let var =` binding + `var` as expression under Madaros.
# Also re-checks stdlib/epistemic/knightian.sio parses cleanly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"

echo "=== A. let var binding run-pass ==="
out="$("$SOUC" run tests/run-pass/let_var_binding_name.sio 2>/tmp/let_var_gate.err | grep -E 'LET_VAR_BINDING_OK' || true)"
if [[ "$out" != *LET_VAR_BINDING_OK* ]]; then
  echo "LET_VAR_BINDING_GATE_FAIL: run-pass" >&2
  cat /tmp/let_var_gate.err >&2 || true
  exit 1
fi
echo "PASS let var binding run"

echo "=== B. knightian.sio check (was parse-fail on let var) ==="
set +e
"$SOUC" check stdlib/epistemic/knightian.sio >/tmp/knightian_check.out 2>/tmp/knightian_check.err
rc=$?
set -e
if grep -q 'parse error' /tmp/knightian_check.out /tmp/knightian_check.err 2>/dev/null; then
  echo "LET_VAR_BINDING_GATE_FAIL: knightian still has parse errors" >&2
  rg 'parse error' /tmp/knightian_check.out /tmp/knightian_check.err >&2 || true
  exit 1
fi
if ! grep -q 'check: OK\|Type check complete\|verdict=0' /tmp/knightian_check.out /tmp/knightian_check.err 2>/dev/null; then
  # Madaros library check may still advisory-warn; reject only hard parse errors (above)
  # Accept if no parse error and exit 0-ish or AST closure complete
  if rg -q 'AST closure incomplete' /tmp/knightian_check.out /tmp/knightian_check.err 2>/dev/null; then
    echo "LET_VAR_BINDING_GATE_FAIL: knightian AST incomplete" >&2
    tail -30 /tmp/knightian_check.out >&2
    exit 1
  fi
fi
echo "PASS knightian.sio no parse error (rc=$rc)"

echo "=== C. knightian_trust import+run ==="
set +e
trust_out="$("$SOUC" run tests/epistemic_trust/knightian_trust.sio 2>/tmp/knightian_trust.err)"
trc=$?
set -e
if [[ $trc -ne 0 ]] || ! echo "$trust_out" | grep -q 'KNIGHTIAN_TRUST_OK'; then
  echo "LET_VAR_BINDING_GATE_FAIL: knightian_trust" >&2
  echo "$trust_out" >&2
  cat /tmp/knightian_trust.err >&2 || true
  exit 1
fi
echo "PASS knightian_trust"

echo "LET_VAR_BINDING_GATE_PASS"
