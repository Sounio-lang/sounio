#!/usr/bin/env bash
# Gate for docs/research/rupture-abcd-claims_2026-07-24.md
#
# R2-partial must PASS (fiber + Frente A measure + random control).
# R3 probe must RUN with sound divergence reconfirm; R3_GREEN is not required
# (hypothesis remains open until B-contract (i)–(iii) are met).
#
# Usage:
#   bash scripts/ci/rupture_abcd_contracts_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== rupture A+B+C+D contracts =="

echo "-- R2 fiber+measure --"
python3 scripts/research/rupture_r2_fiber_measure_contract.py | tee /tmp/rupture_r2_out.txt
grep -q 'R2_CONTRACT_OK' /tmp/rupture_r2_out.txt
grep -q 'R2_PARTIAL PASS' /tmp/rupture_r2_out.txt

echo "-- R3 Fano-restriction probe (Φ_fp) --"
python3 scripts/research/rupture_r3_fano_restriction_probe.py | tee /tmp/rupture_r3_out.txt
grep -q 'R3_CONTRACT_PROBE_OK' /tmp/rupture_r3_out.txt
grep -q 'DIVERGENCE' /tmp/rupture_r3_out.txt
grep -q 'JET_LEMMA PASS' /tmp/rupture_r3_out.txt
grep -q 'R3_VERDICT R3_PARTIAL' /tmp/rupture_r3_out.txt
# Must not falsely claim D3 / R3_GREEN
if grep -q 'R3_VERDICT R3_GREEN' /tmp/rupture_r3_out.txt; then
  echo "unexpected R3_GREEN without full B-contract" >&2
  exit 1
fi

# Claim docs present
test -f docs/research/rupture-abcd-claims_2026-07-24.md
test -f docs/research/rupture-r3-fano-phi_2026-07-25.md

echo "RUPTURE_ABCD_CONTRACTS_OK"
