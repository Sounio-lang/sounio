#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r8_gate.sh
#
# CI gate for rung R8 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r8_2026-07-26.md): evidential
# independence as a statically checkable property.
#
#   I1_IMPORT_CLOSURE       the obvious notion — vacuous on this corpus
#   I2_DERIVATION_DISJOINT  the notion that discriminates (designed pairs)
#   I3_CORPUS_SWEEP         every pair of research contracts
#
# The sweep takes a couple of minutes.
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R8_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r8_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r8_2026-07-26.md"


fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R8_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"


OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in K1_CLUSTER_MAP K2_TRUSTED_BASE K3_KERNELS_AGREE; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# The audit must compare more than one derivation, over the full range. A run
# with one derivation trivially reports zero disagreements and looks like a pass.
DERIVS="$(grep -m1 -oE 'sign derivations +: [0-9]+' <<<"$OUT" | grep -oE '[0-9]+$')"
PRODUCTS="$(grep -m1 -oE 'over [0-9]+ products' <<<"$OUT" | grep -oE '[0-9]+')"
[[ -n "$DERIVS" && "$DERIVS" -ge 3 ]] \
    || fail "only ${DERIVS:-0} derivations compared — corroboration needs at least three"
[[ -n "$PRODUCTS" && "$PRODUCTS" -ge 5000 ]] \
    || fail "only ${PRODUCTS:-0} products compared — levels 3-6 not covered"

# Wrappers must actually be separated from kernels, or step 3 of the method did
# nothing and the "trusted base" is just the cluster list renamed.
grep -qE 'K2_TRUSTED_BASE [0-9]+ irreducible kernels, [1-9][0-9]* wrappers' <<<"$OUT" \
    || fail "no wrappers were distinguished from kernels — the collapse step is inert"

# Drift guard on this rung's own spec — header and prose alike.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R8_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R8_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

echo "SELF_FALSIFYING_COMPILATION_LINE_R8_GATE_OK"
