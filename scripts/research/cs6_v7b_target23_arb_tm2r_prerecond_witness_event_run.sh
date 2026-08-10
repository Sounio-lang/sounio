#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_witness_event_worker.py"
VERIFY="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_witness_event_verify.py"
MUTATIONS="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_witness_event_mutations.py"
PRIOR_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_transport_worker.py"
PRERECOND_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_prerecond_worker.py"
CENTERED_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_centered_worker.py"
COMPOSABILITY="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_composability_carrier_worker.py"
TRANSPORT="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_hset_covering_carrier_worker.py"
CHAIN="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_chain_second_return_worker.py"
ADAPTIVE="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker.py"
EVENT="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_second_return_worker.py"
BASE="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py"
PRERECOND_RECEIPT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_prerecond_v1/event_prerecond.json"
TRANSPORT_RECEIPT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_prerecond_transport_v1/prerecond_transport.json"
DEFAULT_OUT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_prerecond_witness_event_v1"
OUT_DIR="${CS6_OUTPUT_DIR:-$DEFAULT_OUT}"
DEPS="${CS6_PYTHONPATH:-/tmp/sounio-cs6-arb-full-leaf-deps}"

[[ -d "$DEPS/flint" ]] || { echo "python-flint dependency directory is unavailable: $DEPS" >&2; exit 2; }
[[ -f "$PRERECOND_RECEIPT" ]] || { echo "preconditioned event receipt is unavailable" >&2; exit 2; }
[[ -f "$TRANSPORT_RECEIPT" ]] || { echo "frozen transport refusal receipt is unavailable" >&2; exit 2; }

mkdir -p "$OUT_DIR"
result="$OUT_DIR/witness_event.json"
stderr="$OUT_DIR/witness_event.stderr.txt"
verified="$OUT_DIR/witness_event.verified.txt"
mutations="$OUT_DIR/witness_event.mutations.txt"
tmp_result="$result.tmp.$$"
tmp_stderr="$stderr.tmp.$$"
tmp_verified="$verified.tmp.$$"
tmp_mutations="$mutations.tmp.$$"
trap 'rm -f "$tmp_result" "$tmp_stderr" "$tmp_verified" "$tmp_mutations"' EXIT

set +e
PYTHONPATH="$DEPS" PYTHONDONTWRITEBYTECODE=1 \
  python3 -B "$WORKER" > "$tmp_result" 2> "$tmp_stderr"
worker_rc=$?
set -e
if [[ $worker_rc -ne 0 ]]; then
  mv "$tmp_stderr" "$OUT_DIR/witness_event.incomplete.stderr.txt"
  if [[ -s "$tmp_result" ]]; then
    mv "$tmp_result" "$OUT_DIR/witness_event.incomplete.json"
  fi
  echo "witness-local event worker failed with rc=$worker_rc" >&2
  exit "$worker_rc"
fi

verify_args=(
  "$tmp_result"
  --worker "$WORKER"
  --prior-worker "$PRIOR_WORKER"
  --prerecond-worker "$PRERECOND_WORKER"
  --centered-worker "$CENTERED_WORKER"
  --composability "$COMPOSABILITY"
  --transport "$TRANSPORT"
  --chain "$CHAIN"
  --adaptive "$ADAPTIVE"
  --event "$EVENT"
  --base "$BASE"
  --prerecond-receipt "$PRERECOND_RECEIPT"
  --transport-receipt "$TRANSPORT_RECEIPT"
)
PYTHONDONTWRITEBYTECODE=1 python3 -B "$VERIFY" "${verify_args[@]}" > "$tmp_verified"

mv "$tmp_result" "$result"
mv "$tmp_stderr" "$stderr"
mv "$tmp_verified" "$verified"

PYTHONDONTWRITEBYTECODE=1 python3 -B "$MUTATIONS" "$result" \
  --verifier "$VERIFY" \
  --worker "$WORKER" \
  --prior-worker "$PRIOR_WORKER" \
  --prerecond-worker "$PRERECOND_WORKER" \
  --centered-worker "$CENTERED_WORKER" \
  --composability "$COMPOSABILITY" \
  --transport "$TRANSPORT" \
  --chain "$CHAIN" \
  --adaptive "$ADAPTIVE" \
  --event "$EVENT" \
  --base "$BASE" \
  --prerecond-receipt "$PRERECOND_RECEIPT" \
  --transport-receipt "$TRANSPORT_RECEIPT" > "$tmp_mutations"
mv "$tmp_mutations" "$mutations"

trap - EXIT
echo "CS6_PRERECOND_WITNESS_EVENT_COMPLETE=true"
