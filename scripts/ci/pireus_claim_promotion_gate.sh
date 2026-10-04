#!/usr/bin/env bash
# pireus_claim_promotion_gate.sh — PIREUS Batch 2 claim-promotion gate.
#
# Integrity-only composition of scripts/research/pireus_claim_promotion_contract.py:
# the four sealed PARITY_OPEN receipts are hash-checked for drift, the four
# consuming claims are parsed against the ledger schema, falsifiers must pin
# the receipt digests, the P5 evidence ceiling is enforced, and overclaim
# strings are refused. This script performs NO hardware access and NO external
# orchestration calls of any kind; it only reads files already in the tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONTRACT="scripts/research/pireus_claim_promotion_contract.py"

fail() {
  printf 'PIREUS_CLAIM_PROMOTION_GATE_FAIL reason=%s\n' "$1" >&2
  exit 1
}

[[ -f "$CONTRACT" ]] || fail contract_missing

PYTHON=""
for cand in "$ROOT/.venv/bin/python3" python3; do
  if [[ -x "$cand" ]] || command -v "$cand" >/dev/null 2>&1; then
    PYTHON="$cand"
    break
  fi
done
[[ -n "$PYTHON" ]] || fail python3_missing

"$PYTHON" "$CONTRACT" || fail contract_not_green

echo 'PIREUS_CLAIM_PROMOTION_GATE_OK claims=4 receipts=4 reexecution=none'
