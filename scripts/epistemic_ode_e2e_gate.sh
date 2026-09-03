#!/usr/bin/env bash
# scripts/epistemic_ode_e2e_gate.sh
# Epistemic multi-compartment ODE E2E under lean_single (self-contained driver).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/stdlib/ode/test_epistemic_ode_e2e.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/epistemic_ode_e2e.elf"
LOG="$OUT/run.log"
RECEIPT_DIR="$ROOT/artifacts/ode"
RECEIPT="$RECEIPT_DIR/epistemic_ode_e2e_receipt.v1.json"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
fail=0

echo "== epistemic_ode_e2e_gate: engine=$SOUNIO_SOUC_ENGINE =="
if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"; tail -40 "$OUT/compile.log" || true; fail=1
else
  chmod +x "$ELF"
  if ! "$ELF" >"$LOG" 2>&1; then
    echo "FAIL: run"; cat "$LOG" || true; fail=1
  elif ! grep -q "EPISTEMIC_ODE_E2E_OK" "$LOG"; then
    echo "FAIL: missing sentinel"; cat "$LOG" || true; fail=1
  else
    grep '^EPISTEMIC_ODE_E2E' "$LOG" || true
  fi
fi

parse() {
  local key="$1"
  local line
  line="$(grep -E "$2" "$LOG" 2>/dev/null | head -1 || true)"
  if [[ -z "$line" ]]; then echo "null"; return; fi
  echo "$line" | tr ' ' '\n' | grep -E "^${key}=" | head -1 | cut -d= -f2-
}

C1="$(parse c1_c 'EPISTEMIC_ODE_E2E c1_c=')"
U1="$(parse c1_u 'EPISTEMIC_ODE_E2E c1_c=')"
FCL="$(parse frac_cl 'EPISTEMIC_ODE_E2E c1_c=')"
FV="$(parse frac_v 'EPISTEMIC_ODE_E2E c1_c=')"
C2="$(parse c2_c1 'EPISTEMIC_ODE_E2E c2_c1=')"
UC2="$(parse c2_u_c1 'EPISTEMIC_ODE_E2E c2_c1=')"

ORACLE_STATUS="skipped"
ORACLE_DETAIL="{}"
if [[ $fail -eq 0 ]] && [[ -f "$ROOT/scripts/epistemic_ode_e2e_oracle.py" ]]; then
  echo "== optional closed-form oracle =="
  set +e
  ORACLE_OUT="$(python3 "$ROOT/scripts/epistemic_ode_e2e_oracle.py" \
    --c1 "$C1" --u1 "$U1" --frac-cl "$FCL" --frac-v "$FV" 2>&1)"
  ORC=$?
  set -e
  echo "$ORACLE_OUT"
  if [[ $ORC -eq 0 ]]; then
    ORACLE_STATUS="pass"
    ORACLE_DETAIL="$(echo "$ORACLE_OUT" | tail -1)"
  elif [[ $ORC -eq 2 ]]; then
    ORACLE_STATUS="skipped"
    ORACLE_DETAIL="$(echo "$ORACLE_OUT" | tail -1)"
  else
    ORACLE_STATUS="fail"
    ORACLE_DETAIL="$(echo "$ORACLE_OUT" | tail -1)"
    fail=1
  fi
fi

mkdir -p "$RECEIPT_DIR"
STATUS="fail"
[[ $fail -eq 0 ]] && STATUS="pass"

cat >"$RECEIPT" <<EOF
{
  "schema": "epistemic_ode_e2e_receipt.v1",
  "status": "$STATUS",
  "engine": "$SOUNIO_SOUC_ENGINE",
  "commit": "$COMMIT",
  "source": "$SRC",
  "sounio": {
    "one_cmt_c": $C1,
    "one_cmt_u_c": $U1,
    "budget_frac_cl": $FCL,
    "budget_frac_v": $FV,
    "two_cmt_c1": $C2,
    "two_cmt_u_c1": $UC2
  },
  "claims": [
    "linear_pk_trajectory_with_standard_uncertainty",
    "iso_style_param_budget_cl_vs_v_on_terminal_c",
    "two_cmt_linear_rk4_with_state_u_propagation",
    "multi_state_harmonic_smoke"
  ],
  "claims_not_made": [
    "full_diffeq_jl",
    "sciml_events_dae",
    "madaros_multimodule",
    "bedside_dosing",
    "nonmem_foce_parity",
    "numpy_sklearn"
  ],
  "oracle": { "status": "$ORACLE_STATUS", "detail": $ORACLE_DETAIL }
}
EOF
echo "receipt: $RECEIPT"

if [[ $fail -eq 0 ]]; then
  echo "EPISTEMIC_ODE_E2E_GATE_OK"
  exit 0
fi
exit 1
