#!/usr/bin/env bash
# CI gate for the headline claims of the 168-theorem paper
# (docs/papers/main/168-theorem-preprint.md).
#
# Re-verifies, in one shot:
#   (1) the sedenion zero-divisor counting chain 84 -> 336 -> 168 and the
#       168 non-Fano count, via the Lean-transcribed Python oracle
#       (scripts/research/verify_zd168_oracle.py);
#   (2) the ZD-graph invariant theory (Section 6): pair criterion,
#       crown-join recursion, degree law, generator isolation,
#       alpha = b+4, omega = chi = 2^(b-3), census identity, levels 4..9
#       (scripts/research/cd_zd_graph_invariants_contract.py, T1..T8);
#   (3) the nullity histogram law (Section 7): fiber type tau, label
#       count law, recursion, aggregate generation law, L8 falsification,
#       census identity (scripts/research/cd_tower_nullity_histogram_law_contract.py,
#       C1..C7).
#
# Acceptance: rc=0 iff the oracle emits the exact counts AND both
# contracts reach C_GREEN. Prints PAPER_168_THEOREM_GATE_OK on success.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/paper168-gate.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

fail() {
    echo "PAPER_168_THEOREM_GATE_FAIL: $1" >&2
    exit 1
}

# --- Leg 1: census oracle (Lean-transcribed) -------------------------------
ORACLE="${REPO_ROOT}/scripts/research/verify_zd168_oracle.py"
[[ -f "${ORACLE}" ]] || fail "missing ${ORACLE}"
echo "[1/3] zero-divisor census oracle (84/336/168, non-Fano 168) ..."
"${PYTHON}" "${ORACLE}" > "${WORK}/oracle.txt"
grep -qx 'COUNT validPrims 84'  "${WORK}/oracle.txt" || fail "oracle: validPrims != 84"
grep -qx 'COUNT ordered 336'    "${WORK}/oracle.txt" || fail "oracle: ordered ZD pairs != 336"
grep -qx 'COUNT unordered 168'  "${WORK}/oracle.txt" || fail "oracle: projective classes != 168"
grep -qx 'COUNT nonfano 168'    "${WORK}/oracle.txt" || fail "oracle: non-Fano triples != 168"
grep -qx 'COUNT arrows 84'      "${WORK}/oracle.txt" || fail "oracle: dagger arrows != 84"

# --- Leg 2: ZD-graph invariants (paper Section 6) --------------------------
GRAPH="${REPO_ROOT}/scripts/research/cd_zd_graph_invariants_contract.py"
[[ -f "${GRAPH}" ]] || fail "missing ${GRAPH}"
echo "[2/3] ZD-graph invariants contract (levels 4..9) ..."
"${PYTHON}" "${GRAPH}" > "${WORK}/graph.txt" || fail "graph contract exited non-zero"
grep -q 'CD_ZD_GRAPH_INVARIANTS_VERDICT C_GREEN' "${WORK}/graph.txt" \
    || fail "graph contract did not reach C_GREEN"
grep -q 'Z(9) = 249084' "${WORK}/graph.txt" \
    || fail "graph contract: Z(9) = 249084 not reproduced"

# --- Leg 3: nullity histogram law (paper Section 7) ------------------------
HIST="${REPO_ROOT}/scripts/research/cd_tower_nullity_histogram_law_contract.py"
[[ -f "${HIST}" ]] || fail "missing ${HIST}"
echo "[3/3] nullity histogram law contract (levels 4..8) ..."
"${PYTHON}" "${HIST}" > "${WORK}/hist.txt" || fail "histogram contract exited non-zero"
grep -q 'CD_HISTOGRAM_LAW_VERDICT C_GREEN' "${WORK}/hist.txt" \
    || fail "histogram contract did not reach C_GREEN"

echo "PAPER_168_THEOREM_GATE_OK"
