#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r12_gate.sh
#
# CI gate for rung R12 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r12_2026-07-27.md): the
# N-version search, which terminated the branch at Phase 0.
#
#   C1_PRIOR_ART_PINNED      external figures the narrowing rests on
#   C2_R6_MEASURE_IS_POORER  checked against R6's source, not a reading of it
#   C3_STOP_WAS_HONOURED     the pre-registered stop is falsifiable
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R12_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r12_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r12_2026-07-27.md"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R12_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in C1_PRIOR_ART_PINNED C2_R6_MEASURE_IS_POORER C3_STOP_WAS_HONOURED; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# Drift guard on this rung's own spec — header AND prose. R11 shipped with a
# stale headline and a green gate because its guard only checked the Status
# line; that is the same sub-token failure this line documents. Check every
# occurrence.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R12_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R12_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

# The headline must stay consistent with the finding. R11's title said "three
# hazards" for hours after the count reached five, gate green throughout,
# because nothing checked the prose against the result. This rung's whole claim
# is that the branch STOPPED — so guard that word, not just the token.
grep -qi "stopped at Phase 0" "$SPEC" \
    || fail "spec no longer states the Phase-0 stop — §0's claim has drifted"

# The concession is load-bearing and deletable. C6's contribution survives only
# as one-sided: reliable when it reports SHARED evidence, unreliable when it
# reports INDEPENDENT evidence. Losing that sentence silently re-widens the
# paper to the refuted claim. Same guard shape as W3_HONESTY_MARKERS.
grep -q "one-sided test" "$SPEC" \
    || fail "the one-sided-test concession was deleted — spec re-widens to the refuted claim"

echo "SELF_FALSIFYING_COMPILATION_LINE_R12_GATE_OK"
