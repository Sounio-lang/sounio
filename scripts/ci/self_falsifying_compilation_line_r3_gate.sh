#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r3_gate.sh
#
# CI gate for rung R3 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r3_2026-07-26.md): whether an
# executable falsifier can be expressed INDEPENDENTLY of a claim's own harness,
# and would have refuted the parent commit's proposition.
#
#   E1_EXPRESSIBILITY    how many of the three audited corrections admit one
#   E2_WOULD_HAVE_FIRED  whether the expressible ones refute their parent
#
# Pure Python + numpy; no compiler needed.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R3_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r3_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r3_2026-07-26.md"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R3_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in E1_EXPRESSIBILITY E2_WOULD_HAVE_FIRED; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# The one falsifier that is expressible must actually refute its parent, or the
# rung's headline is unsupported.
grep -q "REFUTES the parent" <<<"$OUT" \
    || fail "no falsifier refuted its parent proposition"

# Drift guard on this rung's own spec — header and prose alike.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R3_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R3_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

echo "SELF_FALSIFYING_COMPILATION_LINE_R3_GATE_OK"
