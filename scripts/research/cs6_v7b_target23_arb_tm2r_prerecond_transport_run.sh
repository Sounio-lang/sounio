#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_transport_worker.py"
VERIFY="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_transport_verify.py"
MUTATIONS="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_prerecond_transport_mutations.py"
DEFAULT_OUT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_prerecond_transport_v1"
OUT_DIR="${CS6_OUTPUT_DIR:-$DEFAULT_OUT}"
DEPS="${CS6_PYTHONPATH:-/tmp/sounio-cs6-arb-full-leaf-deps}"
PRERECOND_RECEIPT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_prerecond_v1/event_prerecond.json"
PRERECOND_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_prerecond_worker.py"
CENTERED_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_centered_worker.py"
COMPOSABILITY="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_composability_carrier_worker.py"
TRANSPORT="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_hset_covering_carrier_worker.py"
CHAIN="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_chain_second_return_worker.py"
ADAPTIVE="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker.py"
EVENT="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_second_return_worker.py"
BASE="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py"

[[ -d "$DEPS/flint" ]] || {
  echo "python-flint dependency directory is unavailable: $DEPS" >&2
  exit 2
}
[[ -f "$PRERECOND_RECEIPT" ]] || {
  echo "frozen pre-QR receipt is unavailable: $PRERECOND_RECEIPT" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
result="$OUT_DIR/prerecond_transport.json"
stderr="$OUT_DIR/prerecond_transport.stderr.txt"
verified="$OUT_DIR/prerecond_transport.verified.txt"
mutations="$OUT_DIR/prerecond_transport.mutations.txt"
tmp_result="$result.tmp.$$"
tmp_stderr="$stderr.tmp.$$"
tmp_verified="$verified.tmp.$$"
tmp_mutations="$mutations.tmp.$$"
trap 'rm -f "$tmp_result" "$tmp_stderr" "$tmp_verified" "$tmp_mutations"' EXIT

set +e
PYTHONPATH="$DEPS" PYTHONDONTWRITEBYTECODE=1 \
  python3 -B "$WORKER" > "$tmp_result" 2> "$tmp_stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  mv "$tmp_stderr" "$OUT_DIR/prerecond_transport.incomplete.stderr.txt"
  echo "pre-QR transport worker failed with rc=$rc" >&2
  exit "$rc"
fi

verify_args=(
  "$tmp_result"
  --worker "$WORKER"
  --prerecond-worker "$PRERECOND_WORKER"
  --centered-worker "$CENTERED_WORKER"
  --composability "$COMPOSABILITY"
  --transport "$TRANSPORT"
  --chain "$CHAIN"
  --adaptive "$ADAPTIVE"
  --event "$EVENT"
  --base "$BASE"
  --prerecond-receipt "$PRERECOND_RECEIPT"
)
set +e
PYTHONDONTWRITEBYTECODE=1 python3 -B "$VERIFY" "${verify_args[@]}" \
  > "$tmp_verified" 2> "$OUT_DIR/prerecond_transport.verify.stderr.txt"
verify_rc=$?
set -e

mv "$tmp_result" "$result"
mv "$tmp_stderr" "$stderr"
if [[ $verify_rc -ne 0 ]]; then
  echo "pre-QR transport verify failed with rc=$verify_rc" >&2
  exit "$verify_rc"
fi
mv "$tmp_verified" "$verified"

PYTHONDONTWRITEBYTECODE=1 python3 -B "$MUTATIONS" "$result" \
  --verifier "$VERIFY" \
  --worker "$WORKER" \
  --prerecond-worker "$PRERECOND_WORKER" \
  --centered-worker "$CENTERED_WORKER" \
  --composability "$COMPOSABILITY" \
  --transport "$TRANSPORT" \
  --chain "$CHAIN" \
  --adaptive "$ADAPTIVE" \
  --event "$EVENT" \
  --base "$BASE" \
  --prerecond-receipt "$PRERECOND_RECEIPT" \
  > "$tmp_mutations"
mv "$tmp_mutations" "$mutations"

trap - EXIT
echo "CS6_PRERECOND_TRANSPORT_COMPLETE=true"
