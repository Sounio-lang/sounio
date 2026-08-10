#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_witness_derivative_budget_worker.py"
VERIFY="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_witness_derivative_budget_verify.py"
MUTATIONS="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_witness_derivative_budget_mutations.py"
WITNESS="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_prerecond_witness_event_v1/witness_event.json"
DEFAULT_OUT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_prerecond_witness_derivative_budget_v1"
OUT_DIR="${CS6_OUTPUT_DIR:-$DEFAULT_OUT}"

[[ -f "$WITNESS" ]] || { echo "frozen witness receipt is unavailable" >&2; exit 2; }
mkdir -p "$OUT_DIR"
result="$OUT_DIR/derivative_budget.json"
stderr="$OUT_DIR/derivative_budget.stderr.txt"
verified="$OUT_DIR/derivative_budget.verified.txt"
mutations="$OUT_DIR/derivative_budget.mutations.txt"
tmp_result="$result.tmp.$$"
tmp_stderr="$stderr.tmp.$$"
tmp_verified="$verified.tmp.$$"
tmp_mutations="$mutations.tmp.$$"
trap 'rm -f "$tmp_result" "$tmp_stderr" "$tmp_verified" "$tmp_mutations"' EXIT

PYTHONDONTWRITEBYTECODE=1 python3 -B "$WORKER" > "$tmp_result" 2> "$tmp_stderr"
PYTHONDONTWRITEBYTECODE=1 python3 -B "$VERIFY" "$tmp_result" \
  --worker "$WORKER" \
  --witness-receipt "$WITNESS" > "$tmp_verified"

mv "$tmp_result" "$result"
mv "$tmp_stderr" "$stderr"
mv "$tmp_verified" "$verified"

PYTHONDONTWRITEBYTECODE=1 python3 -B "$MUTATIONS" "$result" \
  --verifier "$VERIFY" \
  --worker "$WORKER" \
  --witness-receipt "$WITNESS" > "$tmp_mutations"
mv "$tmp_mutations" "$mutations"

trap - EXIT
echo "CS6_WITNESS_DERIVATIVE_BUDGET_COMPLETE=true"
