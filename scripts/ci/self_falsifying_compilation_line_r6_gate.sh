#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r6_gate.sh
#
# CI gate for rung R6 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r6_2026-07-26.md): evidential
# independence as a statically checkable property.
#
#   I1_IMPORT_CLOSURE       the obvious notion — vacuous on this corpus
#   I2_DERIVATION_DISJOINT  the notion that discriminates (designed pairs)
#   I3_CORPUS_SWEEP         every pair of research contracts
#
# The sweep takes a couple of minutes.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R6_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r6_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r6_2026-07-26.md"
FIXTURE="scripts/ci/fixtures/independence_copypaste_corroborator.py"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R6_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"
[[ -f "$FIXTURE" ]] || fail "negative fixture missing: $FIXTURE"

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in I1_IMPORT_CLOSURE I2_DERIVATION_DISJOINT I3_CORPUS_SWEEP; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# The guard must DISCRIMINATE, not merely compute. Both designed cases must land
# on their expected side; a guard that passes everything (or nothing) is
# worthless and would otherwise show up as a green tick.
grep -q "expected independent) \[OK\]" <<<"$OUT" \
    || fail "the independent pair was not classified independent — guard over-flags"
grep -q "expected shared) \[OK\]" <<<"$OUT" \
    || fail "the copy-paste corroborator was not rejected — guard is VACUOUS"

# Drift guard on this rung's own spec — header and prose alike.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R6_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R6_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

echo "SELF_FALSIFYING_COMPILATION_LINE_R6_GATE_OK"
