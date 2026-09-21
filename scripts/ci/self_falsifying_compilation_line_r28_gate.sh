#!/usr/bin/env bash
# CI gate for rung R28: the confidence gate separates almost nothing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r28_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r28_2026-08-01.md"
RECEIPT="scripts/research/r28/conf_census.json"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R28_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"
[[ -f "$CONTRACT" ]] || fail "contract missing"
[[ -f "$RECEIPT" ]] || fail "census receipt missing -- the corpus figures have no source"
# The subject. Without it a green gate would certify the absence of the thing measured.
[[ -x bin/souc-lean-single-x86_64 ]] \
    || fail "bin/souc-lean-single-x86_64 absent: the live clauses cannot run"
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in B1_SUPPORT_IS_GRADED B2_MASS_IS_BINARY B3_GATE_SEPARATES_ALMOST_NOTHING \
         B4_SHARED_REDIRECT_INVENTS_VALUES B5_LIVE_CENSUS_AGREES; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
# B1 and B2 are a pair and must stay one. Either alone is a misleading headline:
# "graded" without the mass, or "binary" without the support.
grep -q 'Graded in principle; two-valued in practice' <<<"$OUT" \
    || fail "the graded/binary pair is no longer stated together"
# B4 is the instrument check. If it stops finding torn values the control has
# broken, and B1's support can no longer be trusted.
grep -qE 'values above 1000, which the scalar cannot take: [1-9]' <<<"$OUT" \
    || fail "the shared-redirect control no longer demonstrates fabrication"
grep -qE 'values above 1000: 0' <<<"$OUT" \
    || fail "the correctly-written census is itself producing impossible values"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R28_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: '${SPEC_TOKEN}' vs '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R28_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
# A rung about calibration is one sentence from claiming it calibrated something.
grep -q "Not a calibration" "$SPEC" \
    || fail "the disclaimer that nothing here was calibrated was deleted"
grep -q "Not a claim that the confidence is wrong" "$SPEC" \
    || fail "the disclaimer separating 'degenerate' from 'incorrect' was deleted"
grep -q "Not a compiler change" "$SPEC" || fail "the spec now implies the compiler was changed"
echo "SELF_FALSIFYING_COMPILATION_LINE_R28_GATE_OK"
