#!/usr/bin/env bash
# CI gate for rung R22: the gate that certifies a literal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r22_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r22_2026-07-29.md"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R22_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"; [[ -f "$CONTRACT" ]] || fail "contract missing"
# The subject of the rung is the governance generator and its checker. Without
# them there is nothing to measure, and a green gate would certify their absence.
[[ -f scripts/docs/governance_registry.mjs ]] || fail "governance_registry.mjs absent -- nothing to measure"
[[ -f scripts/docs/check_docs_registry.mjs ]] || fail "check_docs_registry.mjs absent -- nothing to measure"
command -v node >/dev/null 2>&1 || fail "node absent -- the enforcement arm cannot be exercised"
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in V1_VALUE_IS_A_LITERAL V2_ONE_DATE_FOR_EVERY_DOC \
         V3_DATE_PRECEDES_THE_REPO V4_GATE_REJECTS_THE_TRUE_DATE; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
# The separation from registry staleness is the point of V4's farm sync: without
# it this rung goes red whenever anyone adds a document, which is the docs-registry
# gate's job and not this one's. If the sync is removed, say so here.
grep -q 'V4 farm synced to consistency' <<<"$OUT" \
    || fail "the farm is no longer synced before measurement -- V4 has gone back to inheriting registry staleness"
# A sync WRITES. If a real copy ever degrades back into a hardlink it writes
# through to the working tree, so hermeticity is asserted, not assumed.
grep -qE 'V4 hermetic: [0-9]+ working-tree files unchanged' <<<"$OUT" \
    || fail "hermeticity is no longer asserted after the farm sync"

# An instrument with one arm measures nothing. Both controls must remain.
grep -qE 'negative control -- farm unmodified: checker rc=0' <<<"$OUT" \
    || fail "the negative control no longer reproduces the green result"
grep -qE 'positive control -- truthful date: checker rc=[1-9]' <<<"$OUT" \
    || fail "the positive control no longer exercises rejection"
grep -q 'metadata mismatch for last_validated' <<<"$OUT" \
    || fail "the rejection is no longer attributed to the field under study"
# The two literal sites are the finding. If they are gone, this rung is stale.
grep -qE 'governance_registry\.mjs:[0-9]+ +-> ' <<<"$OUT" \
    || fail "the literal sites are no longer reported"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R22_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: '${SPEC_TOKEN}' vs '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R22_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
# A rung that finds a defect is one sentence from overclaiming what it found.
grep -q "Not a claim that the documents are unvalidated" "$SPEC" \
    || fail "the disclaimer separating 'uninformative field' from 'unvalidated docs' was deleted"
grep -q "Not the sweep's fault" "$SPEC" \
    || fail "the disclosure that the sweep is not the culprit was deleted"
grep -q "Not fixed" "$SPEC" || fail "the spec now implies the defect is fixed"
echo "SELF_FALSIFYING_COMPILATION_LINE_R22_GATE_OK"
