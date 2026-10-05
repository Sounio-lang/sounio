#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r16_gate.sh
#
# CI gate for rung R16 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r16_2026-07-28.md): the
# invariance group is partition-preserving, not merely count-preserving.
#
#   C1_FLIP_IS_MINIMAL_BY_CONSTRUCTION  arithmetic, n = 5..12
#   C2_TWO_EDGES_PER_FIBER              the perturbation is minimal and uniform
#   C3_PARTITION_PRESERVED_LABELS_NOT   same blocks, new spectra
#
# n = 5 and n = 6 are derived live (~1 min); n = 7 is read from record.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R16_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r16_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r16_2026-07-28.md"
DATA="scripts/research/r16"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R16_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"
for f in recorded.json partition_probe.py locality_probe.py; do
    [[ -f "$DATA/$f" ]] || fail "recorded evidence missing: $DATA/$f"
done

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in C1_FLIP_IS_MINIMAL_BY_CONSTRUCTION C2_TWO_EDGES_PER_FIBER \
              C3_PARTITION_PRESERVED_LABELS_NOT; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# C3 is the whole finding and it needs BOTH halves. Same blocks with the same
# spectra would be a no-op; different blocks would be an ordinary perturbation.
grep -q "blocks IDENTICAL" <<<"$OUT" || fail "the partition is no longer preserved"
grep -q "spectra DIFFER" <<<"$OUT" \
    || fail "the spectra no longer differ — the flip has become a no-op and C3 is vacuous"

# Verdict drift, header AND prose.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R16_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r t; do
    [[ -z "$t" ]] && continue
    [[ "$t" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${t}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R16_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

# The concessions. This rung explains a mechanism and the temptation is to call
# it a proof; the sentence saying it is not is one deletion away.
grep -q "^- \*\*Not a proof" "$SPEC" \
    || fail "the not-a-proof concession was deleted"
# The emphasis wraps both words -- "**not established**", not "not **established**".
# The first version of this guard got that wrong and failed at baseline. Same
# class of slip as R14's line-wrapped grep: a guard that cannot pass is as
# useless as one that cannot fail, and only a negative test at baseline shows it.
grep -q "\*\*not established\*\*" "$SPEC" \
    || fail "the concession that the key inference is unproved was deleted"
grep -q "Not a refutation of anything" "$SPEC" \
    || fail "the scope limit was deleted — the spec now reads as refuting a live claim"
grep -q "Not a complete description of the group" "$SPEC" \
    || fail "the one-element-per-level limit was deleted"

echo "SELF_FALSIFYING_COMPILATION_LINE_R16_GATE_OK"
