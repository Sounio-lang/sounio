#!/usr/bin/env bash
# CI gate for rung R25: research authority is path-default historical.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r25_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r25_2026-07-31.md"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R25_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"; [[ -f "$CONTRACT" ]] || fail "contract missing"
[[ -f scripts/docs/governance_registry.mjs ]] || fail "governance_registry.mjs absent -- nothing to measure"
[[ -f scripts/docs/check_docs_registry.mjs ]] || fail "check_docs_registry.mjs absent -- nothing to measure"
command -v node >/dev/null 2>&1 || fail "node absent -- the enforcement arm cannot be exercised"
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in V1_WHITELIST_IS_THREE V2_DEFAULT_IS_HISTORICAL \
         V3_CORPUS_IS_LINEAGE_DEFAULT V4_GATE_REJECTS_CURRENT; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
grep -q 'V4 farm synced to consistency' <<<"$OUT" \
    || fail "the farm is no longer synced before measurement"
grep -qE 'V4 hermetic: [0-9]+ working-tree files unchanged' <<<"$OUT" \
    || fail "hermeticity is no longer asserted after the farm sync"
grep -qE 'negative control -- farm unmodified: checker rc=0' <<<"$OUT" \
    || fail "the negative control no longer reproduces the green result"
grep -qE 'positive control -- claim current: checker rc=[1-9]' <<<"$OUT" \
    || fail "the positive control no longer exercises rejection"
grep -q 'metadata mismatch for authority' <<<"$OUT" \
    || fail "the rejection is no longer attributed to the field under study"
grep -q 'ACTIVE_RESEARCH_DOCS' <<<"$OUT" \
    || fail "the whitelist is no longer reported"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R25_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: '${SPEC_TOKEN}' vs '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R25_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
grep -q "Not a claim that historical is always wrong" "$SPEC" \
    || fail "the disclaimer separating path default from true lineage was deleted"
grep -q "Not fixed" "$SPEC" || fail "the spec now implies the defect is fixed"
grep -q "Third field" "$SPEC" \
    || fail "the disclosure that this is the third docs:meta field of the family was deleted"
echo "SELF_FALSIFYING_COMPILATION_LINE_R25_GATE_OK"
