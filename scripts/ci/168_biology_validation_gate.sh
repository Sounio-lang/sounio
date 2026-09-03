#!/usr/bin/env bash
# CI gate for the biological claims of the 168 biology preprint
# (docs/papers/main/168-biology-preprint.typ).
#
# Re-verifies, in one shot, every quantitative biological claim of the
# preprint via the deterministic contract
# (scripts/research/168_biology_validation_contract.py):
#   (1) the 343 = 133 + 42 + 168 ordered-triple partition and the octonion
#       associator census (168 nonzero, norm in {0, 2});
#   (2) the CYP450 locus audit (NCBI cytogenetic bands, CYP2C cluster on
#       the (2,3,5) Fano line) and the gauge analysis of the bijection
#       (6 equivalence classes under PSL(2,7); only the CYP2C line is
#       gauge-invariant; 2 classes after adding the big-three constraint);
#   (3) the genetic-code PG(5,2) claims: Hamming/hydrophobicity table
#       (both stop conventions, strict monotonicity, r = 0.199 / 0.218),
#       permutation-test p < 0.05, encoding non-optimality audit,
#       mutation robustness (26.3% synonymous, 55.9% class-preserving),
#       and the Fano-line class-coherence null result.
#
# Acceptance: rc=0 iff the contract reaches C_GREEN.
# Prints BIO168_VALIDATION_GATE_OK on success.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/bio168-gate.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

fail() {
    echo "BIO168_VALIDATION_GATE_FAIL: $1" >&2
    exit 1
}

CONTRACT="${REPO_ROOT}/scripts/research/168_biology_validation_contract.py"
[[ -f "${CONTRACT}" ]] || fail "missing ${CONTRACT}"

echo "[1/1] 168 biology validation contract (C1..C10) ..."
"${PYTHON}" "${CONTRACT}" > "${WORK}/contract.txt" || {
    cat "${WORK}/contract.txt" >&2
    fail "contract exited non-zero"
}
grep -q 'BIO168_VALIDATION_VERDICT C_GREEN' "${WORK}/contract.txt" \
    || fail "contract did not reach C_GREEN"

# Spot-check the headline values so a silently weakened contract cannot
# pass the gate.
grep -q 'trivial=133 fano=42 nonassoc=168' "${WORK}/contract.txt" \
    || fail "343 partition not reproduced"
grep -q 'nonzero basis associators=168' "${WORK}/contract.txt" \
    || fail "octonion associator census not reproduced"
grep -q 'r=0.1990' "${WORK}/contract.txt" \
    || fail "Hamming/hydrophobicity correlation not reproduced"
grep -q '651 Fano lines' "${WORK}/contract.txt" \
    || fail "PG(5,2) line census not reproduced"

echo "BIO168_VALIDATION_GATE_OK"
