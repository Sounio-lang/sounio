#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r11_gate.sh
#
# CI gate for rung R11 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r11_2026-07-26.md): evidential
# independence as a statically checkable property.
#
#   I1_IMPORT_CLOSURE       the obvious notion — vacuous on this corpus
#   I2_DERIVATION_DISJOINT  the notion that discriminates (designed pairs)
#   I3_CORPUS_SWEEP         every pair of research contracts
#
# The sweep takes a couple of minutes.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R11_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r11_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r11_2026-07-26.md"


fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R11_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"


OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in P1_WIDER_PROBE P2_DEPTH; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# The widening must actually widen. If it reaches no more behaviour classes than
# R10 did, the rung is R10 again with a longer runtime.
CLASSES="$(grep -m1 -oE 'distinct behaviour classes : [0-9]+' <<<"$OUT" | grep -oE '[0-9]+$')"
[[ -n "$CLASSES" && "$CLASSES" -ge 2 ]] \
    || fail "only ${CLASSES:-0} behaviour classes — the probe did not widen past R10"

PROBE="$(grep -m1 -oE 'probeable functions +: [0-9]+' <<<"$OUT" | grep -oE '[0-9]+$')"
[[ -n "$PROBE" && "$PROBE" -ge 31 ]] \
    || fail "only ${PROBE:-0} probeable functions — fewer than R10 reached"

# The known corroboration must still be found, and must still be attributed to
# the CORPUS rather than to this line's own oracle. Losing the split is how a
# self-introduced check gets counted as evidence the project already had.
grep -qE 'pre-existing corroborations: [1-9]' <<<"$OUT" \
    || fail "the known cds/cd_sigma corroboration is no longer found as PRE-EXISTING"
grep -q "line-introduced ones" <<<"$OUT" \
    || fail "the self-reference split is gone — introduced evidence would be \
counted as the corpus's own"

# Drift guard on this rung's own spec — header and prose alike.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R11_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R11_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

echo "SELF_FALSIFYING_COMPILATION_LINE_R11_GATE_OK"
