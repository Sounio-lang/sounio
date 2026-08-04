#!/usr/bin/env bash
# CI gate for rung R24: provenance bound where honest, hollow elsewhere.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r24_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r24_2026-07-31.md"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R24_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"; [[ -f "$CONTRACT" ]] || fail "contract missing"
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in U1_EVERY_CLAIM_CLASSIFIED U2_BOUND_IFF_BINDABLE U3_THE_BINDABLE_ONE_IS_BOUND; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
# U2 is the whole point: provenance declared exactly on the bindable claims,
# both directions. A hollow binding or a missing honest one must red it.
grep -q "^U2_BOUND_IFF_BINDABLE PASS" <<<"$OUT" \
    || fail "provenance is no longer bound exactly where it is honest"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R24_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: '${SPEC_TOKEN}' vs '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R24_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
grep -q "would be authoring the disease the line studies" "$SPEC" \
    || fail "the reason the hollow binding is refused was deleted"
grep -q "Not a refusal to bind more later" "$SPEC" \
    || fail "the both-directions concession was deleted"
echo "SELF_FALSIFYING_COMPILATION_LINE_R24_GATE_OK"
