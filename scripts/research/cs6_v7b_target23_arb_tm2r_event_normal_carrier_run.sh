#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_normal_carrier_worker.py"
VERIFY="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_normal_carrier_verify.py"
MUTATIONS="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_normal_carrier_mutations.py"
WITNESS_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_witness_event_worker.py"
BUDGET_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_witness_derivative_budget_worker.py"
BASE="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py"
ADAPTIVE="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker.py"
WITNESS_RECEIPT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_prerecond_witness_event_v1/witness_event.json"
BUDGET_RECEIPT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_prerecond_witness_derivative_budget_v1/derivative_budget.json"
DEFAULT_OUT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_normal_carrier_v1"
OUT_DIR="${CS6_OUTPUT_DIR:-$DEFAULT_OUT}"
DEPS="${CS6_PYTHONPATH:-/tmp/sounio-cs6-arb-full-leaf-deps}"
EXECUTION_MODE="${CS6_EXECUTION_MODE:-PREFLIGHT}"
CARRIER_MODE="${CS6_CARRIER_MODE:-EVENT_NORMAL_TRIPLETON}"

[[ -d "$DEPS/flint" ]] || { echo "python-flint dependency directory is unavailable: $DEPS" >&2; exit 2; }
[[ -f "$WITNESS_RECEIPT" ]] || { echo "frozen witness receipt is unavailable" >&2; exit 2; }
[[ -f "$BUDGET_RECEIPT" ]] || { echo "frozen derivative budget is unavailable" >&2; exit 2; }

case "$EXECUTION_MODE" in
  PREFLIGHT) stem="preflight" ;;
  TRANSPORT)
    case "$CARRIER_MODE" in
      EVENT_NORMAL_DOUBLETON) stem="transport_doubleton" ;;
      EVENT_NORMAL_TRIPLETON) stem="transport_tripleton" ;;
      *) echo "invalid CS6_CARRIER_MODE=$CARRIER_MODE" >&2; exit 2 ;;
    esac
    ;;
  *) echo "invalid CS6_EXECUTION_MODE=$EXECUTION_MODE" >&2; exit 2 ;;
esac

mkdir -p "$OUT_DIR"
result="$OUT_DIR/$stem.json"
stderr="$OUT_DIR/$stem.stderr.txt"
verified="$OUT_DIR/$stem.verified.txt"
mutations="$OUT_DIR/$stem.mutations.txt"
tmp_result="$result.tmp.$$"
tmp_stderr="$stderr.tmp.$$"
tmp_verified="$verified.tmp.$$"
tmp_mutations="$mutations.tmp.$$"
trap 'rm -f "$tmp_result" "$tmp_stderr" "$tmp_verified" "$tmp_mutations"' EXIT

set +e
PYTHONPATH="$DEPS:$ROOT/scripts/research" PYTHONDONTWRITEBYTECODE=1 \
  CS6_EXECUTION_MODE="$EXECUTION_MODE" CS6_CARRIER_MODE="$CARRIER_MODE" \
  python3 -B "$WORKER" > "$tmp_result" 2> "$tmp_stderr"
worker_rc=$?
set -e
if [[ $worker_rc -ne 0 ]]; then
  mv "$tmp_stderr" "$OUT_DIR/$stem.incomplete.stderr.txt"
  if [[ -s "$tmp_result" ]]; then mv "$tmp_result" "$OUT_DIR/$stem.incomplete.json"; fi
  echo "event-normal carrier worker failed with rc=$worker_rc" >&2
  exit "$worker_rc"
fi

verify_args=(
  "$tmp_result"
  --worker "$WORKER"
  --witness-worker "$WITNESS_WORKER"
  --budget-worker "$BUDGET_WORKER"
  --base "$BASE"
  --adaptive "$ADAPTIVE"
  --witness-receipt "$WITNESS_RECEIPT"
  --budget-receipt "$BUDGET_RECEIPT"
)
PYTHONDONTWRITEBYTECODE=1 python3 -B "$VERIFY" "${verify_args[@]}" > "$tmp_verified"
mv "$tmp_result" "$result"
mv "$tmp_stderr" "$stderr"
mv "$tmp_verified" "$verified"

if [[ "$EXECUTION_MODE" == "PREFLIGHT" ]]; then
  PYTHONDONTWRITEBYTECODE=1 python3 -B "$MUTATIONS" "$result" \
    --verifier "$VERIFY" \
    --worker "$WORKER" \
    --witness-worker "$WITNESS_WORKER" \
    --budget-worker "$BUDGET_WORKER" \
    --base "$BASE" \
    --adaptive "$ADAPTIVE" \
    --witness-receipt "$WITNESS_RECEIPT" \
    --budget-receipt "$BUDGET_RECEIPT" > "$tmp_mutations"
  mv "$tmp_mutations" "$mutations"
fi

trap - EXIT
echo "CS6_EVENT_NORMAL_CARRIER_COMPLETE=true mode=$EXECUTION_MODE carrier=$CARRIER_MODE"
