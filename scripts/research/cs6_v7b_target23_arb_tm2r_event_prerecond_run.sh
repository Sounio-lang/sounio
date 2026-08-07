#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_prerecond_worker.py"
VERIFY="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_prerecond_verify.py"
DEFAULT_OUT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_prerecond_v1"
OUT_DIR="${CS6_OUTPUT_DIR:-$DEFAULT_OUT}"
DEPS="${CS6_PYTHONPATH:-/tmp/sounio-cs6-arb-full-leaf-deps}"
EVENT_LOCAL="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_local_v1/event_local_diagnostic.json"
EVENT_CENTERED="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_centered_v1/event_centered.json"
RHO3_LOCUS="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_rho3_locus_v1/rho3_locus.json"
CENTERED_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_centered_worker.py"
PRIOR_WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_local_diagnostic_worker.py"

[[ -d "$DEPS/flint" ]] || {
  echo "python-flint dependency directory is unavailable: $DEPS" >&2
  exit 2
}
for path in "$EVENT_LOCAL" "$EVENT_CENTERED" "$RHO3_LOCUS"; do
  [[ -f "$path" ]] || {
    echo "frozen prior receipt is unavailable: $path" >&2
    exit 2
  }
done

mkdir -p "$OUT_DIR"
result="$OUT_DIR/event_prerecond.json"
stderr="$OUT_DIR/event_prerecond.stderr.txt"
verified="$OUT_DIR/event_prerecond.verified.txt"
tmp_result="$result.tmp.$$"
tmp_stderr="$stderr.tmp.$$"
tmp_verified="$verified.tmp.$$"
trap 'rm -f "$tmp_result" "$tmp_stderr" "$tmp_verified"' EXIT

set +e
PYTHONPATH="$DEPS" PYTHONDONTWRITEBYTECODE=1 \
  python3 -B "$WORKER" > "$tmp_result" 2> "$tmp_stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  mv "$tmp_stderr" "$OUT_DIR/event_prerecond.incomplete.stderr.txt"
  echo "prerecond worker failed with rc=$rc" >&2
  exit "$rc"
fi

set +e
PYTHONPATH="$DEPS" PYTHONDONTWRITEBYTECODE=1 python3 -B "$VERIFY" \
  "$tmp_result" \
  --worker "$WORKER" \
  --centered-worker "$CENTERED_WORKER" \
  --prior-worker "$PRIOR_WORKER" \
  --carrier "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_composability_carrier_worker.py" \
  --chain "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_chain_second_return_worker.py" \
  --adaptive "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker.py" \
  --event "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_second_return_worker.py" \
  --base "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py" \
  --event-local-receipt "$EVENT_LOCAL" \
  --event-centered-receipt "$EVENT_CENTERED" \
  --rho3-locus-receipt "$RHO3_LOCUS" \
  > "$tmp_verified" 2> "$OUT_DIR/event_prerecond.verify.stderr.txt"
verify_rc=$?
set -e

mv "$tmp_result" "$result"
mv "$tmp_stderr" "$stderr"
if [[ $verify_rc -ne 0 ]]; then
  echo "prerecond verify failed with rc=$verify_rc" >&2
  exit "$verify_rc"
fi

mv "$tmp_verified" "$verified"
trap - EXIT
echo "CS6_EVENT_PRERECOND_COMPLETE=true"
