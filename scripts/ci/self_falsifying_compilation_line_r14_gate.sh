#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r14_gate.sh
#
# CI gate for rung R14 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r14_2026-07-27.md): what the
# corpus computes vs what it checks. The vacuity hypothesis was refuted.
#
#   C1_CONTROL_INERT          instrument before corpus
#   C2_LOAD_BEARING_MEASURED  three outcomes, not two
#   C3_VACUITY_REFUTED        every all-survive level explained by invariance
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R14_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r14_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r14_2026-07-27.md"
DATA="scripts/research/r14"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R14_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"
for f in call_trace.json loadbearing.json trace.py loadbearing.py; do
    [[ -f "$DATA/$f" ]] || fail "recorded evidence missing: $DATA/$f"
done

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in C1_CONTROL_INERT C2_LOAD_BEARING_MEASURED C3_VACUITY_REFUTED; do
    grep -q "^${clause}" <<<"$OUT" || fail "${clause} missing from output"
done
grep -qE "^C1_CONTROL_INERT PASS  ([0-9]+)/\1 inert" <<<"$OUT" \
    || fail "not every usable contract is inert to the null wrapper"
grep -q "^C2_LOAD_BEARING_MEASURED PASS" <<<"$OUT" || fail "C2 did not PASS"
grep -q "every all-survive level is explained by single-flip invariance: YES" <<<"$OUT" \
    || fail "an all-survive level is no longer explained -- C3's claim has changed"

# Verdict drift, header AND prose.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R14_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r t; do
    [[ -z "$t" ]] && continue
    [[ "$t" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${t}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R14_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

# The concessions. Each keeps a real result from being read as more than it is,
# and each is one sentence away from deletion.
grep -q "It is refuted" "$SPEC" \
    || fail "the spec no longer states that its own hypothesis lost"
# grep is line-oriented and this phrase wraps, so match the part that sits on
# one line. The first version used \s* across the wrap and failed at baseline --
# a guard that cannot pass is as useless as one that cannot fail.
grep -q "the measure reports \*\*fragility, not" "$SPEC" \
    || fail "the ALL-CRASH qualification was deleted"
grep -q "Nothing further is claimed" "$SPEC" \
    || fail "the L8_64_192 restraint was deleted -- the anomaly now reads as a finding"
grep -q "denominator attached" "$SPEC" \
    || fail "the sampling concession was deleted"
grep -q "strengthened, not weakened" "$SPEC" \
    || fail "the R13 correction disclosure was deleted"

echo "SELF_FALSIFYING_COMPILATION_LINE_R14_GATE_OK"
