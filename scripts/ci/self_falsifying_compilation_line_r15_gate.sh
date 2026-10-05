#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r15_gate.sh
#
# CI gate for rung R15 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r15_2026-07-28.md): a verdict
# token is blind to whatever preserves the truth of its proposition.
#
#   C1_CONSTRUCTION_VALIDATED   fresh re-implementation vs published counts
#   C2_COUNT_PRESERVING_FAMILY  the family, with controls
#   C3_WITNESS_WOULD_CATCH_IT   the repair, verified
#
# n=5 and n=6 are derived live (~1 min); n=7/8 are read from record.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R15_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r15_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r15_2026-07-28.md"
DATA="scripts/research/r15"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R15_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"
for f in recorded.json fiber_reconstruction.py cardinality_probe.py cardinality_control.py; do
    [[ -f "$DATA/$f" ]] || fail "recorded evidence missing: $DATA/$f"
done

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in C1_CONSTRUCTION_VALIDATED C2_COUNT_PRESERVING_FAMILY C3_WITNESS_WOULD_CATCH_IT; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# The control column is what makes C2 a finding rather than a robustness note.
# If a future edit made every flip preserve the count, C2 would still say PASS
# on the survivor line while meaning nothing.
grep -q "control ->  5 changes" <<<"$OUT" \
    || fail "the n=5 control no longer changes the count — C2 is vacuous"
grep -q "control ->  7 changes" <<<"$OUT" \
    || fail "the n=6 control no longer changes the count — C2 is vacuous"

# C3 discriminates only if the sets genuinely differ at equal cardinality.
grep -q "sets DIFFER" <<<"$OUT" || fail "witness-binding no longer discriminates"

# Verdict drift, header AND prose.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R15_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r t; do
    [[ -z "$t" ]] && continue
    [[ "$t" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${t}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R15_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

# The scope limit is the single most deletable sentence here, and deleting it
# turns a statement about a CHECK into a refutation of a live CLAIM.
grep -q "Not a refutation of the completeness claim" "$SPEC" \
    || fail "the scope limit was deleted — the spec now reads as refuting the n<=8 claim"
grep -q "not the truth of the \*\*claim\*\*" "$SPEC" \
    || fail "the check-vs-claim distinction was deleted"
grep -q "Not a cospectral counterexample" "$SPEC" \
    || fail "the disclosure that the pre-registered hypothesis lost was deleted"
grep -q "is measured across n = 5–8, not derived" "$SPEC" \
    || fail "the concession that the regularity is unexplained was deleted"

echo "SELF_FALSIFYING_COMPILATION_LINE_R15_GATE_OK"
