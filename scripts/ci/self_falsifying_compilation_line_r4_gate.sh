#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r4_gate.sh
#
# CI gate for rung R4 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r4_2026-07-26.md): the
# retrospective over the correction history, run under the predicate R0 §5 fixed
# before the study.
#
#   R4_POPULATION       both populations measured
#   R4_CLASSIFICATION   every (commit, spec) pair bucketed
#
# The scan walks the whole git history; allow a few minutes.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R4_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r4_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r4_2026-07-26.md"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R4_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in R4_POPULATION R4_CLASSIFICATION; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# The study must remain gradeable: if NOTHING is classifiable the retrospective
# reports nothing, and that should be loud rather than a green tick.
CLASSIFIABLE="$(grep -m1 -oE 'classifiable pairs +: [0-9]+' <<<"$OUT" | grep -oE '[0-9]+$')"
[[ -n "$CLASSIFIABLE" && "$CLASSIFIABLE" -gt 0 ]] \
    || fail "no classifiable pairs — the retrospective graded nothing"

# Floor tied to the KNOWN population, not to zero. P1 (the objective, token-change
# population) is exactly these two commits; both must still land in a graded
# bucket. Without this, NO_PRIOR_CLAIM grows monotonically as specs accumulate
# and the classifiable set can decay towards nothing while the gate stays green.
for known in daa0635d0 ec579a24c; do
    grep -qE "^ +(CAUGHT_[ABC]|SILENT): ${known}" <<<"$OUT" \
        || fail "known correction ${known} no longer lands in a graded bucket — \
the retrospective has lost the cases it was built to grade"
done

# Drift guard on this rung's own spec — header and prose alike.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R4_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R4_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

echo "SELF_FALSIFYING_COMPILATION_LINE_R4_GATE_OK"
