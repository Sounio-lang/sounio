#!/usr/bin/env bash
# scripts/ci/oopsla2027_paper_gate.sh
#
# CI gate for the OOPSLA 2027 full draft (docs/papers/oopsla2027/paper.md):
# the draft bound to the rung evidence it cites.
#
#   P1_TOKENS_BOUND     every verdict token cited in the draft matches its spec
#   P2_ALL_RUNGS_CITED  every rung R0..R15 on disk is cited
#   P3_FIGURES_PINNED   the load-bearing figures are present
#   P4_HONESTY_MARKERS  the four narrowings and scope concessions survive prose
#
# Companion to scripts/ci/self_falsifying_compilation_line_r5_gate.sh, which
# binds the SKELETON (outline.md). This gate binds the DRAFT.
#
# Exit 0 = PASS (prints OOPSLA2027_PAPER_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/oopsla2027_paper_contract.py"
PAPER="docs/papers/oopsla2027/paper.md"

fail() {
    echo "OOPSLA2027_PAPER_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$PAPER" ]] || fail "paper draft missing: $PAPER"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in P1_TOKENS_BOUND P2_ALL_RUNGS_CITED P3_FIGURES_PINNED P4_HONESTY_MARKERS; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

echo "OOPSLA2027_PAPER_GATE_OK"
