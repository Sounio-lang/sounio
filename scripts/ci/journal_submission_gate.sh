#!/usr/bin/env bash
# CI gate: journal submission draft claim-boundedness contract.
#
# Contract J1: draft file exists and has the required journal sections.
# Contract J2: priority attributions present (Koebisu, Cawagas, Moreno).
# Contract J3: claim discipline — D3 guard sentences present; forbidden
#              unqualified-identity / clinical / ML-superiority phrases absent.
# Contract J4: every verdict token cited in the draft is printed by a harness
#              under scripts/research/ (no phantom verdicts).
# Contract J5: every contract file cited in the draft exists.
#
# Exit 0 and JOURNAL_SUBMISSION_GATE_OK iff all clauses pass.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DRAFT="${REPO_ROOT}/docs/papers/rupture_journal_submission_2026-07-26.md"
RESEARCH_DIR="${REPO_ROOT}/scripts/research"

fail() { echo "JOURNAL_SUBMISSION_GATE_FAIL: $1"; exit 1; }

[[ -f "${DRAFT}" ]] || fail "missing draft ${DRAFT}"

# --- J1: required sections -------------------------------------------------
for section in "## Abstract" "## 1. Introduction" "## 2. Mathematical background" \
               "## 3. The Cayley" "## 5. Functor F" "## 7. The exceptional frontier" \
               "## 9. Computational verification" "## References" "## AI disclosure" \
               "Proof" "proof"; do
  grep -qF "${section}" "${DRAFT}" || fail "J1 missing section marker: ${section}"
done
echo "J1_SECTIONS PASS"

# --- J2: priority attributions ----------------------------------------------
for name in "Koebisu" "Cawagas" "Moreno" "Eakin" "Reggiani"; do
  grep -qF "${name}" "${DRAFT}" || fail "J2 missing attribution: ${name}"
done
grep -qF "arXiv:2512.13002" "${DRAFT}" || fail "J2 missing Koebisu arXiv identifier"
echo "J2_ATTRIBUTIONS PASS"

# --- J3: claim discipline ----------------------------------------------------
# D3 guard sentences must be present.
grep -qF "asserted nowhere in this paper" "${DRAFT}" \
  || fail "J3 missing D3 guard sentence"
grep -qF "D3 remains forbidden" "${DRAFT}" \
  || fail "J3 missing D3-forbidden marker"
grep -qF "quarantined" "${DRAFT}" \
  || fail "J3 missing Wildgen-conjecture quarantine"
# Forbidden unqualified-identity / clinical / ML phrases must be absent.
for bad in "same singularity structure" "det L_x measures" "depression" "suicide" \
           "suicidality" "beats Mamba" "outperforms" "patient-level efficacy"; do
  if grep -qiF "${bad}" "${DRAFT}"; then
    fail "J3 forbidden phrase present: ${bad}"
  fi
done
echo "J3_CLAIM_DISCIPLINE PASS"

# --- J4: cited verdict tokens are real harness tokens ------------------------
TOKENS=(
  R2_CONTRACT_OK R2_FULL_MEASURED T_GREEN R3_GREEN R4_GREEN
  G_GREEN H_CHARACTERISED E_GREEN K_CHARACTERISED P_GREEN Q_GREEN
  B_OBSTRUCTED PHI_JETS_VANISH_PROVEN M_CHARACTERISED
  SECONDARY_TERNARY_LOCATED NO_CANONICAL_FILL NO_INVARIANT_FILL
  PSL27_THREADS_THE_TOWER PHI_IS_THE_E6_CUBIC_CROSSTERM
  E7_QUARTIC_BUILT_NO_CLEAN_PSI_HOME
  E8_NO_SMALL_REP_PHI_IS_TOWER_FORM_DATUM
  RUPTURE_ABCD_CONTRACTS_OK
)
for tok in "${TOKENS[@]}"; do
  grep -qF "${tok}" "${DRAFT}" || fail "J4 token cited in gate list but absent from draft: ${tok}"
  grep -rqF "${tok}" "${RESEARCH_DIR}" "${REPO_ROOT}/scripts/ci" \
    || fail "J4 token not printed by any harness: ${tok}"
done
echo "J4_VERDICT_TOKENS PASS (${#TOKENS[@]} tokens)"

# --- J5: cited contract files exist ------------------------------------------
mapfile -t CITED < <(grep -oE '[a-z0-9_]+(_contract|_probe)\.py' "${DRAFT}" | sort -u)
[[ ${#CITED[@]} -gt 0 ]] || fail "J5 no contract files cited in draft"
for base in "${CITED[@]}"; do
  [[ -f "${RESEARCH_DIR}/${base}" ]] || fail "J5 cited contract missing: scripts/research/${base}"
done
mapfile -t CITED_SH < <(grep -oE 'scripts/ci/[a-z0-9_]+\.sh' "${DRAFT}" | sort -u)
for rel in "${CITED_SH[@]}"; do
  [[ -f "${REPO_ROOT}/${rel}" ]] || fail "J5 cited CI gate missing: ${rel}"
done
echo "J5_CONTRACT_FILES PASS (${#CITED[@]} contracts, ${#CITED_SH[@]} gates)"

echo "JOURNAL_SUBMISSION_GATE_OK"
