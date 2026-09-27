#!/usr/bin/env bash
# CI gate for rung R19 of the self-falsifying compilation line.
#   Y1_LOCALITY_DERIVED  Y2_EXCEPTIONAL_PREDICTED  Y3_MARKING_DOES_NOT_REFINE
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r19_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r19_2026-07-28.md"
fail() { echo "SELF_FALSIFYING_COMPILATION_LINE_R19_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"
[[ -f "$CONTRACT" ]] || fail "contract missing"
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in Y1_LOCALITY_DERIVED Y2_EXCEPTIONAL_PREDICTED Y3_MARKING_DOES_NOT_REFINE; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
# Y2 is the derivation earning its keep: the fiber must be PREDICTED, not found.
grep -q "derivation predicts" <<<"$OUT" || fail "Y2 no longer predicts the fiber"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R19_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: spec '${SPEC_TOKEN}' vs contract '${CT}'"
while read -r t; do
    [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R19_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
# The concessions. A rung that derives half a result is one sentence away from
# reading as if it derived all of it.
grep -q "Not a proof of R16's inference" "$SPEC" || fail "the not-a-proof concession was deleted"
# R21 later proved the half R19 left measured. The guard now checks that R19's
# own limit stays on the record AND that the pointer to the proof is there --
# superseded, not erased, the same rule R17 got.
grep -q "was true of R19 and stays on the record" "$SPEC" \
    || fail "R19's own limit was erased instead of superseded"
grep -q "Closed in R21" "$SPEC" \
    || fail "the pointer to R21's proof was deleted"
grep -q "Not a proof for all n" "$SPEC" || fail "the n-range limit was deleted"
grep -q "Two explanations tested, both refuted" "$SPEC" \
    || fail "the refuted-hypotheses disclosure was deleted"
echo "SELF_FALSIFYING_COMPILATION_LINE_R19_GATE_OK"
