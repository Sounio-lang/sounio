#!/usr/bin/env bash
# Gate for docs/research/rupture-abcd-claims_2026-07-24.md
#
# R2-partial       — fiber + Frente A measure + random control
# R2_FULL_MEASURED — exact det/rank anchors + MC tubular measurements (not a proof)
# R3_GREEN         — Φ_fp jet + path classes (single-line; D3 forbidden)
# R4_GREEN         — multi-line field + Φ_fp Path C/D from cross-line jet (D3 forbidden)
#
# Usage:
#   bash scripts/ci/rupture_abcd_contracts_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== rupture A+B+C+D (+R2 full measured + R4 field) contracts =="

echo "-- R2 fiber+measure (partial) --"
python3 scripts/research/rupture_r2_fiber_measure_contract.py | tee /tmp/rupture_r2_out.txt
grep -q 'R2_CONTRACT_OK' /tmp/rupture_r2_out.txt
grep -q 'R2_PARTIAL PASS' /tmp/rupture_r2_out.txt

echo "-- R2 full tubular (measured, not proved) --"
python3 scripts/research/rupture_r2_full_tubular_probe.py | tee /tmp/rupture_r2_full_out.txt
grep -q 'R2_FULL_PROBE_OK' /tmp/rupture_r2_full_out.txt
grep -q 'R2_FULL_VERDICT R2_FULL_MEASURED' /tmp/rupture_r2_full_out.txt
grep -q 'EXACT_ANCHORS' /tmp/rupture_r2_full_out.txt

echo "-- R3 Fano-restriction probe (Φ_fp) --"
python3 scripts/research/rupture_r3_fano_restriction_probe.py | tee /tmp/rupture_r3_out.txt
grep -q 'R3_CONTRACT_PROBE_OK' /tmp/rupture_r3_out.txt
grep -q 'DIVERGENCE' /tmp/rupture_r3_out.txt
grep -q 'JET_LEMMA PASS' /tmp/rupture_r3_out.txt
grep -q 'CLAUSE_III_PLUS contrariety_vs_contradiction_paths -> PASS' /tmp/rupture_r3_out.txt
grep -q 'R3_VERDICT R3_GREEN' /tmp/rupture_r3_out.txt
# R3_GREEN is operational (path classes), not D3 identity
if grep -q 'D3_identity_still_forbidden' /tmp/rupture_r3_out.txt; then
  :
else
  echo "R3_GREEN must keep D3_identity_still_forbidden marker" >&2
  exit 1
fi

echo "-- R4 multi-line Fano field + multi-line Phi --"
python3 scripts/research/rupture_r4_fano_field_contract.py | tee /tmp/rupture_r4_out.txt
grep -q 'R4_CONTRACT_OK' /tmp/rupture_r4_out.txt
grep -q 'R4_VERDICT R4_GREEN' /tmp/rupture_r4_out.txt
grep -q 'F5_SYSTEM_RESIDUAL nonzero_cross -> PASS' /tmp/rupture_r4_out.txt
grep -q 'F7_MULTI_LINE_PHI_PATHS -> PASS' /tmp/rupture_r4_out.txt
if grep -q 'D3_identity_still_forbidden' /tmp/rupture_r4_out.txt; then
  :
else
  echo "R4_GREEN must keep D3_identity_still_forbidden marker" >&2
  exit 1
fi

# Claim docs present
test -f docs/research/rupture-abcd-claims_2026-07-24.md
test -f docs/research/rupture-programme-synthesis_2026-07-25.md
test -f docs/research/rupture-r2-full-tubular_2026-07-25.md
test -f docs/research/rupture-r3-fano-phi_2026-07-25.md
test -f docs/research/rupture-r4-fano-field_2026-07-25.md

echo "RUPTURE_ABCD_CONTRACTS_OK"
