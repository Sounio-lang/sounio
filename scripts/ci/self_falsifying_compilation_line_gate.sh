#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_gate.sh
#
# CI gate for the self-falsifying compilation research line
# (docs/research/self_falsifying_compilation_line_2026-07-26.md).
#
# The line's opening rung is an audit: it measures how much of the repository's
# empirical surface is bound to source-level claims, and whether a claim gate
# would have fired at the historical corrections. This gate runs that audit and
# checks the verdict token.
#
# Clauses (executed by the contract):
#   S1_SUBSTRATE_SURFACE  --verify-claims machinery present in compiler source
#   S2_CORPUS_GAP         native claims vs CI gates / research contracts
#   S3_BINDING_GAP        how many CI gates are named by any claim
#   S4_RETROSPECTIVE      would a claim gate have fired at known corrections?
#
# Set SFCL_RUN_COMPILER_GATE=1 to additionally run the underlying mechanism
# gate (scripts/ci/self_falsifying_compiler_gate.sh), which needs a claim-aware
# Madaros and is CPU-heavy; it is not run by default.
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_2026-07-26.md"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

OUT="$(python3 "$CONTRACT" 2>&1)" || {
    echo "$OUT"
    fail "contract exited non-zero"
}
echo "$OUT"

for clause in S1_SUBSTRATE_SURFACE S2_CORPUS_GAP S3_BINDING_GAP S4_RETROSPECTIVE; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

grep -q "^SELF_FALSIFYING_LINE_VERDICT " <<<"$OUT" \
    || fail "contract emitted no verdict token"

# The spec's declared verdict token must be the one the contract emits. This is
# the very drift guard the line proposes, applied to the line's own spec.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"

CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_LINE_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"

# Resolution fix: the Status line is not the only place the verdict is stated.
# A restatement in the prose drifted silently once (spec §1) while this gate
# stayed green, because it compared only the header. Every in-prose
# `SELF_FALSIFYING_LINE_VERDICT <TOKEN>` must agree too.
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_LINE_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

if [[ "${SFCL_RUN_COMPILER_GATE:-0}" == "1" ]]; then
    echo "--- running underlying mechanism gate ---"
    SFC_SKIP_BUILD=1 bash scripts/ci/self_falsifying_compiler_gate.sh \
        || fail "underlying self_falsifying_compiler_gate.sh failed"
fi

echo "SELF_FALSIFYING_COMPILATION_LINE_GATE_OK"
