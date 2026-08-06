#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECEIPTS="$SCRIPT_DIR/receipts/cs6_v7b_target23_arb_tm2r_composability_v1"
ANALYZER="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_composability_analyze.py"
VERIFIER="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_composability_verify.py"
MUTATIONS="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_composability_mutations.py"
TEMP_AGGREGATE="$(mktemp /tmp/cs6-composability-aggregate.XXXXXX)"
trap 'rm -f "$TEMP_AGGREGATE"' EXIT

cd "$REPO_ROOT"
python3 "$ANALYZER" \
  "$RECEIPTS/support_XLEL.json" \
  "$RECEIPTS/support_XLEH.json" \
  "$RECEIPTS/support_XHEL.json" \
  "$RECEIPTS/support_XHEH.json" \
  > "$TEMP_AGGREGATE"
cmp "$TEMP_AGGREGATE" "$RECEIPTS/aggregate.txt"
python3 "$VERIFIER" --receipts "$RECEIPTS"
python3 "$MUTATIONS"

echo "DETERMINISTIC_REANALYSIS=true"
echo "FULL_SUPPORT_CERTIFICATE=true"
echo "LOCAL_HSET_COVERING_RELATION_B_TO_C_CERTIFICATE=true"
echo "RECURRENT_COVERING_GRAPH_CERTIFICATE=false"
echo "CHAOS_PROVED=false"
echo "GATE_PASS=true"
