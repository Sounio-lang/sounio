#!/usr/bin/env bash
# CI gate for rung R26: the reconstructed oracle; the orbit verifier runs in-tree.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r26_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r26_2026-07-31.md"
ORACLE="scripts/research/cd_tower_automorphism_oracle.py"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R26_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"; [[ -f "$CONTRACT" ]] || fail "contract missing"
[[ -f "$ORACLE" ]] || fail "the reconstructed oracle is absent again -- the dependency has re-dangled"
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in W1_ORACLE_VALIDATES W2_ORBIT_THEOREM_RUNS W3_DANGLING_DEP_CLOSED; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
grep -q "first in-tree run of the orbit verifier" <<<"$OUT" \
    || fail "W2 no longer records the orbit verifier actually running"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R26_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: '${SPEC_TOKEN}' vs '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R26_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
grep -q "Not a new proof of the orbit theorem" "$SPEC" || fail "the not-a-new-proof concession was deleted"
grep -q "Not recovery" "$SPEC" || fail "the reconstruction-not-recovery concession was deleted"
grep -q "only \*\*21\*\* of the 168" "$SPEC" || fail "the 168-vs-21 validation subtlety was deleted"
echo "SELF_FALSIFYING_COMPILATION_LINE_R26_GATE_OK"
