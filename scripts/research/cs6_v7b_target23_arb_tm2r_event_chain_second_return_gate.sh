#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/research"
RECEIPTS="$SCRIPT_DIR/receipts/cs6_v7b_target23_arb_tm2r_event_chain_second_return_v1"
DEPS="${CS6_PYTHONPATH:-/tmp/sounio-cs6-arb-full-leaf-deps}"
WORKER="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_event_chain_second_return_worker.py"

if [[ "${CS6_REGENERATE:-0}" == "1" ]]; then
  for tile in XLEL XLEH XHEL XHEH; do
    CS6_SOURCE_TILE="$tile" PYTHONPATH="$DEPS" python3 "$WORKER" \
      > "$RECEIPTS/$tile.stdout.txt" \
      2> "$RECEIPTS/$tile.stderr.txt"
  done
fi

python3 "$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_event_chain_second_return_verify.py" \
  --receipts "$RECEIPTS"
python3 "$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_event_chain_second_return_mutations.py"

echo "CS6_V7B_TARGET23_ARB_TM2R_EVENT_CHAIN_SECOND_RETURN_GATE=true"
