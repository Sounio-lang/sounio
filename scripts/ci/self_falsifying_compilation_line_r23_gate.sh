#!/usr/bin/env bash
# CI gate for rung R23 (inverted 2026-08-16): the guard that replaced the
# ownership label misnamed as validation. The original rung demonstrated that
# the docs-registry checker REJECTED a truthful validated_by; #1752 closed
# that defect by making the provenance pair preserve-per-document and
# shape-checked. This gate now guards the FIXED property in both directions:
# the truth is accepted, blanked and forged values are still rejected.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r23_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r23_2026-07-30.md"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R23_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"; [[ -f "$CONTRACT" ]] || fail "contract missing"
[[ -f scripts/docs/governance_registry.mjs ]] || fail "governance_registry.mjs absent -- nothing to measure"
[[ -f scripts/docs/check_docs_registry.mjs ]] || fail "check_docs_registry.mjs absent -- nothing to measure"
command -v node >/dev/null 2>&1 || fail "node absent -- the enforcement arm cannot be exercised"
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in V1_GENERATOR_PRESERVES_VALIDATOR V2_CORPUS_VALIDATORS_WELL_FORMED \
         V3_STRUCTURE_STAYS_REGISTRY_BOUND V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
grep -q 'V4 farm synced to consistency' <<<"$OUT" \
    || fail "the farm is no longer synced before measurement"
grep -qE 'V4 hermetic: [0-9]+ working-tree files unchanged' <<<"$OUT" \
    || fail "hermeticity is no longer asserted after the farm sync"

# An instrument with one arm measures nothing. All five arms must remain.
grep -qE 'negative control -- farm unmodified: checker rc=0' <<<"$OUT" \
    || fail "the negative control no longer reproduces the green result"
grep -qE 'truthful-validator control -- non-owner name: checker rc=0 \(accepted\)' <<<"$OUT" \
    || fail "a truthful validator is no longer accepted -- the fixed property regressed"
grep -qE 'preserve control -- survives the sync: validated_by=.+' <<<"$OUT" \
    || fail "the preserve control (truthful validator surviving a re-sync) is no longer exercised"
grep -qE 'empty-validator control -- blanked field: checker rc=[1-9]' <<<"$OUT" \
    || fail "the empty-validator control no longer exercises rejection"
grep -q 'expected a non-empty validator' <<<"$OUT" \
    || fail "the empty-validator rejection is no longer attributed to the field under study"
grep -qE 'structural control -- forged topic_id: checker rc=[1-9]' <<<"$OUT" \
    || fail "the structural control no longer bites -- the guard would pass vacuously"
grep -q 'metadata mismatch for topic_id' <<<"$OUT" \
    || fail "the structural rejection is no longer attributed to a registry-bound field"
# The preserve rule is the finding now. If it is gone, this rung is stale.
grep -qE 'governance_registry\.mjs:[0-9]+ +-> export function preservedProvenance' <<<"$OUT" \
    || fail "the preserve rule is no longer reported as the rung's subject"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R23_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: '${SPEC_TOKEN}' vs '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R23_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
grep -q "Not a claim that A6 did or did not validate" "$SPEC" \
    || fail "the disclaimer separating preserved names from audited ones was deleted"
grep -q "Sibling of R22" "$SPEC" \
    || fail "the disclosure that this is the sibling field of last_validated was deleted"
# The closure is load-bearing history: the original finding stays recorded, and
# the spec must say the rung was closed by inversion, by whom, and when.
grep -q "Closed by inversion" "$SPEC" \
    || fail "the spec no longer records the closure by inversion"
grep -q "#1752" "$SPEC" \
    || fail "the spec no longer names the change that closed the defect"
echo "SELF_FALSIFYING_COMPILATION_LINE_R23_GATE_OK"
