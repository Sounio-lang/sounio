#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r10_gate.sh
#
# CI gate for rung R10 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r10_2026-07-26.md): evidential
# independence as a statically checkable property.
#
#   I1_IMPORT_CLOSURE       the obvious notion — vacuous on this corpus
#   I2_DERIVATION_DISJOINT  the notion that discriminates (designed pairs)
#   I3_CORPUS_SWEEP         every pair of research contracts
#
# The sweep takes a couple of minutes.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R10_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r10_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r10_2026-07-26.md"


fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R10_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"


OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in L1_PROBEABLE L2_LATENT_CORROBORATION L3_CORROBORATION_DEPTH; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# The search must actually exercise something. A probe that accepts nothing
# reports zero corroborations and looks like a clean negative.
PROBE="$(grep -m1 -oE 'probeable functions +: [0-9]+' <<<"$OUT" | grep -oE '[0-9]+$')"
[[ -n "$PROBE" && "$PROBE" -ge 10 ]] \
    || fail "only ${PROBE:-0} probeable functions — the search covered nothing"

# It must still rediscover the corroboration R8 found by hand. Losing that is
# how this rung silently becomes a no-op that still prints a verdict.
grep -qE 'REAL corroborations +: [1-9]' <<<"$OUT" \
    || fail "the known cds/cd_sigma corroboration is no longer rediscovered — \
the discovery procedure has stopped working"

# The inflated pair count must stay labelled. Deleting the label is how 130
# copies get read as 130 findings.
grep -q "copies, NOT corroborations" <<<"$OUT" \
    || fail "the cross-derivation pair count lost its 'copies, not corroborations' label"

# Drift guard on this rung's own spec — header and prose alike.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R10_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R10_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

echo "SELF_FALSIFYING_COMPILATION_LINE_R10_GATE_OK"
