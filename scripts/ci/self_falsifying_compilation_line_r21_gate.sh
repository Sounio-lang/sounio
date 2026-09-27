#!/usr/bin/env bash
# CI gate for rung R21: the equivariance lemma, proved.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r21_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r21_2026-07-28.md"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R21_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"; [[ -f "$CONTRACT" ]] || fail "contract missing"
# the proof needs the artifact R20 restored; without it the rung is not reproducible
[[ -f scripts/research/cd_tower_collapse_isomorphism.py ]] \
    || fail "cd_tower_collapse_isomorphism.py absent again -- the proof cannot be checked"
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in V1_H_IS_A_SEAM_BIT V2_TAU_FIXES_H V3_BLOCKS_ARE_ORBITS_PLUS_COLLAPSE; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
grep -q "lsb never n-2" <<<"$OUT" || fail "the parity side-condition is no longer checked"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R21_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: '${SPEC_TOKEN}' vs '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R21_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
# A rung that proves something is one sentence from claiming it proved more.
grep -q "Not a proof from first principles" "$SPEC" || fail "the prior-results dependency was deleted"
grep -q "Not ∀n" "$SPEC" || fail "the n-range limit was deleted"
grep -q "independently of the orbit contract" "$SPEC" || fail "the disclosure that V3 avoids the unrunnable contract was deleted"
echo "SELF_FALSIFYING_COMPILATION_LINE_R21_GATE_OK"
