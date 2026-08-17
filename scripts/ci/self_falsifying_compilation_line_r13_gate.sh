#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r13_gate.sh
#
# CI gate for rung R13 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r13_2026-07-27.md):
# co-sensitivity vs structural distance, measured on this corpus.
#
#   C1_CONTROL_INERT                   the instrument, checked before the corpus
#   C2_BATTERY_DISCRIMINATES           >= 8 informative mutants (pre-registered)
#   C3_IDENTICAL_FATE_BELOW_THRESHOLD  the finding
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R13_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r13_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r13_2026-07-27.md"
DATA="scripts/research/r13"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R13_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"
for f in battery_results.json structural_similarity.json manifest.json battery.py probe.py; do
    [[ -f "$DATA/$f" ]] || fail "recorded evidence missing: $DATA/$f"
done

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in C1_CONTROL_INERT C2_BATTERY_DISCRIMINATES C3_IDENTICAL_FATE_BELOW_THRESHOLD; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# C1 is not one number among many: if the harness contaminates the measurement
# every figure below it is void. The first battery failed exactly here and
# reported the contamination as a corpus finding (spec 5.1).
#
# Matched as N/N, not as a literal count. The first version pinned "28/28" and
# went red the moment R14 recovered two contracts the battery had lost to a
# timeout -- a guard that fails on its own subject growing is a guard that has
# to be edited to stay green, which is how counts go stale.
grep -qE "^C1_CONTROL_INERT PASS  ([0-9]+)/\1 inert" <<<"$OUT" \
    || fail "not every usable contract is inert to the null wrapper"

# Verdict drift, header AND prose -- R11 shipped a stale headline with a green
# gate because its guard checked only the Status line.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R13_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r t; do
    [[ -z "$t" ]] && continue
    [[ "$t" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${t}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R13_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

# The concessions are load-bearing and are the easiest sentences to lose. Each
# one keeps a real finding from being read as more than it is.
grep -q "Not a demonstration of shared misinterpretation" "$SPEC" \
    || fail "the co-sensitivity/misinterpretation distinction was deleted"
grep -q "carries little" "$SPEC" \
    || fail "the concession that Pearson r is near-degenerate was deleted"
grep -q "cannot certify one" "$SPEC" \
    || fail "the negative-test-only concession was deleted -- spec now implies the measure can certify a corroborator"
grep -qi "instrument had\s*changed the question\|It was the harness" "$SPEC" \
    || fail "5.1's instrument-contamination disclosure was deleted"

echo "SELF_FALSIFYING_COMPILATION_LINE_R13_GATE_OK"
