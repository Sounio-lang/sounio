#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r5_gate.sh
#
# CI gate for rung R5 of the self-falsifying compilation line: the OOPSLA 2027
# paper skeleton (docs/papers/oopsla2027/outline.md), bound to its evidence.
#
#   W1_TOKENS_BOUND     every verdict token cited in the paper matches its spec
#   W2_ALL_RUNGS_CITED  every rung of the line is represented
#   W3_HONESTY_MARKERS  the unverified related-work section is still marked
#
# The paper is where claims get restated far from the harness that measured
# them. This gate is what stops the restatement from drifting.
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R5_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r5_contract.py"
PAPER="docs/papers/oopsla2027/outline.md"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R5_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$PAPER" ]] || fail "paper skeleton missing: $PAPER"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in W1_TOKENS_BOUND W2_ALL_RUNGS_CITED W3_HONESTY_MARKERS; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# Drift guard on the paper's own Status token.
PAPER_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$PAPER" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$PAPER_TOKEN" ]] || fail "paper declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R5_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$PAPER_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: paper says '${PAPER_TOKEN}', contract emits '${CONTRACT_TOKEN}'"

echo "SELF_FALSIFYING_COMPILATION_LINE_R5_GATE_OK"
