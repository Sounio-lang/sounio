#!/usr/bin/env bash
# scripts/ci/witness_based_compilation_paper_gate.sh
#
# CI gate for the Witness-Based Compilation paper
# (docs/papers/witness_based_compilation_2026-07-28.md): the draft bound to the
# rung evidence it cites.
#
#   W1_TOKENS_BOUND    every verdict token cited matches its spec
#   W2_WITNESS_PINNED  the witness fingerprints quoted match the manifest claim
#   W3_FIGURES_PINNED  the load-bearing measured figures are present
#   W4_HONESTY_MARKERS the measured/derived distinction and limits survive
#   W5_PEER_REVIEW_FIXES the 2026-07-28 peer-review fixes survive
#
# Exit 0 = PASS (prints WITNESS_BASED_COMPILATION_PAPER_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/witness_based_compilation_paper_contract.py"
PAPER="docs/papers/witness_based_compilation_2026-07-28.md"

fail() {
    echo "WITNESS_BASED_COMPILATION_PAPER_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$PAPER" ]] || fail "paper draft missing: $PAPER"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in W1_TOKENS_BOUND W2_WITNESS_PINNED W3_FIGURES_PINNED W4_HONESTY_MARKERS W5_PEER_REVIEW_FIXES; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

echo "WITNESS_BASED_COMPILATION_PAPER_GATE_OK"
