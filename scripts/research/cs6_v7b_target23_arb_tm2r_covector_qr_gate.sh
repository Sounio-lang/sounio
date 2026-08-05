#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECEIPTS="$SCRIPT_DIR/receipts/cs6_v7b_target23_arb_tm2r_covector_qr_v1"
ANALYZER="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_covector_qr_analyze.py"
VERIFIER="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_covector_qr_verify.py"
MUTATIONS="$SCRIPT_DIR/cs6_v7b_target23_arb_tm2r_covector_qr_mutations.py"
TEMP_AGGREGATE="$(mktemp /tmp/cs6-covector-qr-aggregate.XXXXXX)"
trap 'rm -f "$TEMP_AGGREGATE"' EXIT

cd "$REPO_ROOT"
python3 "$ANALYZER" > "$TEMP_AGGREGATE"
cmp "$TEMP_AGGREGATE" "$RECEIPTS/aggregate.txt"
python3 "$VERIFIER" --receipts "$RECEIPTS"
python3 "$MUTATIONS"

echo "DETERMINISTIC_REANALYSIS=true"
echo "QR_FALSIFIER_CERTIFICATE=true"
echo "C2_ANCHORED_LOCAL_HSET_COVERING_CERTIFICATE=true"
echo "RECURRENT_COVERING_GRAPH_CERTIFICATE=false"
echo "CHAOS_PROVED=false"
echo "GATE_PASS=true"
