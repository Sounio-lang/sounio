#!/usr/bin/env bash
# CI gate for rung R27: declared alive, never checked.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"; cd "$REPO_ROOT"
CONTRACT="scripts/research/self_falsifying_compilation_line_r27_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r27_2026-08-01.md"
fail(){ echo "SELF_FALSIFYING_COMPILATION_LINE_R27_GATE_FAIL: $*" >&2; exit 1; }
[[ -f "$SPEC" ]] || fail "spec missing"; [[ -f "$CONTRACT" ]] || fail "contract missing"
# The subjects of the rung. Without them a green gate would certify their absence.
for f in examples/epistemic/rupture_claims_verified.sio \
         self-hosted/compiler/claim_executor.sio \
         self-hosted/compiler/lean_single.sio; do
    [[ -f "$f" ]] || fail "subject absent: $f"
done
OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"
for c in A1_ALIVE_IS_UNCHECKED A2_PROMISE_SCOPE_IS_NARROWER_THAN_THE_MECHANISM \
         A3_BINDINGS_ARE_RARE A4_ANCHORING_CHANGES_THE_CENSUS; do
    grep -q "^${c} PASS" <<<"$OUT" || fail "${c} did not PASS"
done
# A1 is the finding. If the executor ever starts reading the verdict for real,
# this rung is stale and must be re-measured rather than left asserting.
grep -qE 'occurrences of `Alive` in the executor \(code, not comments\): 0' <<<"$OUT" \
    || fail "the executor now mentions Alive -- re-measure A1, do not edit the token"
# A4 is the instrument check. An unanchored census would inflate the headline.
grep -q 'comment-stripped, field-anchored' <<<"$OUT" \
    || fail "the census is no longer anchored; its numbers cannot be trusted"
# A2 must keep naming the uncovered lane, not merely count it.
grep -q 'lean_single emits and never verifies' <<<"$OUT" \
    || fail "the uncovered ELF lane is no longer named"
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token"
CT="$(grep -m1 '^SELF_FALSIFYING_R27_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CT" ]] || fail "verdict drift: '${SPEC_TOKEN}' vs '${CT}'"
while read -r t; do [[ -z "$t" ]] && continue
    [[ "$t" == "$CT" ]] || fail "verdict drift in prose: '${t}'"
done < <(grep -oE 'SELF_FALSIFYING_R27_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')
# A rung that finds a defect is one sentence from overclaiming what it found.
grep -q "Not a claim that any claim is false" "$SPEC" \
    || fail "the disclaimer separating 'unchecked' from 'untrue' was deleted"
grep -q "Not a refutability test" "$SPEC" \
    || fail "the concession that no gate was perturbed was deleted"
grep -q "Not a compiler change" "$SPEC" || fail "the spec now implies the compiler was changed"
echo "SELF_FALSIFYING_COMPILATION_LINE_R27_GATE_OK"
