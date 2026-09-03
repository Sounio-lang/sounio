#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r9_gate.sh
#
# CI gate for rung R9 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r9_2026-07-26.md): evidential
# independence as a statically checkable property.
#
#   I1_IMPORT_CLOSURE       the obvious notion — vacuous on this corpus
#   I2_DERIVATION_DISJOINT  the notion that discriminates (designed pairs)
#   I3_CORPUS_SWEEP         every pair of research contracts
#
# The sweep takes a couple of minutes.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R9_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r9_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r9_2026-07-26.md"


fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R9_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"


OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in K9_AUDIT; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# The audit must actually corroborate something. A run where every kernel came
# back NO_ADJUDICATOR would report zero divergences and look like a pass.
CORR="$(grep -m1 -oE 'corroborated +: [0-9]+' <<<"$OUT" | grep -oE '[0-9]+$')"
[[ -n "$CORR" && "$CORR" -ge 5 ]] \
    || fail "only ${CORR:-0} kernels corroborated — the audit adjudicated almost nothing"

# The predictive kernels are the ones that can be WRONG. If they stop being
# checked, the audit has lost its point even while staying green.
for k in expected_labels missing_diagonal; do
    grep -qE "^ +${k} +\[PREDICTIVE\] CORROBORATED" <<<"$OUT" \
        || fail "${k} is no longer corroborated as PREDICTIVE — the kernels that \
can actually be false are unchecked"
done

# Drift guard on this rung's own spec — header and prose alike.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R9_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R9_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

echo "SELF_FALSIFYING_COMPILATION_LINE_R9_GATE_OK"
