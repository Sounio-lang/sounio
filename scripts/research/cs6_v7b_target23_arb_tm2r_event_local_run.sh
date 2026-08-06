#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_local_diagnostic_worker.py"
VERIFY="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_local_verify.py"
DEFAULT_OUT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_event_local_v1"
OUT_DIR="${CS6_OUTPUT_DIR:-$DEFAULT_OUT}"
DEPS="${CS6_PYTHONPATH:-/tmp/sounio-cs6-arb-full-leaf-deps}"

[[ -d "$DEPS/flint" ]] || {
  echo "python-flint dependency directory is unavailable: $DEPS" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
result="$OUT_DIR/event_local_diagnostic.json"
stderr="$OUT_DIR/event_local_diagnostic.stderr.txt"
verified="$OUT_DIR/event_local_diagnostic.verified.txt"
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
  mv "$tmp_stderr" "$OUT_DIR/event_local_diagnostic.incomplete.stderr.txt"
  echo "event-local diagnostic worker failed with rc=$rc" >&2
  exit "$rc"
fi

PYTHONPATH="$DEPS" PYTHONDONTWRITEBYTECODE=1 python3 -B "$VERIFY" \
  "$tmp_result" \
  --worker "$WORKER" \
  --carrier "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_composability_carrier_worker.py" \
  --chain "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_event_chain_second_return_worker.py" \
  --adaptive "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker.py" \
  --event "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_second_return_worker.py" \
  --base "$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py" \
  > "$tmp_verified"

mv "$tmp_result" "$result"
mv "$tmp_stderr" "$stderr"
mv "$tmp_verified" "$verified"
trap - EXIT
echo "CS6_EVENT_LOCAL_DIAGNOSTIC_COMPLETE=true"
