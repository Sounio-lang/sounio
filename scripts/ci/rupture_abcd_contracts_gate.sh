#!/usr/bin/env bash
# Gate for docs/research/rupture-abcd-claims_2026-07-24.md
#
# R2-partial       — fiber + Frente A measure + random control
# R2_FULL_MEASURED — exact det/rank anchors + MC tubular measurements (not a proof)
# ORD2             — ord 2″ subspace alignment (gap alone is FP)
# R3_GREEN         — Φ_fp jet + path classes (single-line; D3 forbidden)
# R4_GREEN         — multi-line field + Φ_fp Path C/D from cross-line jet (D3 forbidden)
#
# Usage:
#   bash scripts/ci/rupture_abcd_contracts_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Prefer project venv when present (torch/numpy for trained LSTM probe)
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "== rupture A+B+C+D (+R2 full + ORD2 + R3/R4) contracts =="
echo "python=$PYTHON"

echo "-- R2 fiber+measure (partial) --"
$PYTHON scripts/research/rupture_r2_fiber_measure_contract.py | tee /tmp/rupture_r2_out.txt
grep -q 'R2_CONTRACT_OK' /tmp/rupture_r2_out.txt
grep -q 'R2_PARTIAL PASS' /tmp/rupture_r2_out.txt

echo "-- R2 full tubular (measured, not proved) --"
$PYTHON scripts/research/rupture_r2_full_tubular_probe.py | tee /tmp/rupture_r2_full_out.txt
grep -q 'R2_FULL_PROBE_OK' /tmp/rupture_r2_full_out.txt
grep -q 'R2_FULL_VERDICT R2_FULL_MEASURED' /tmp/rupture_r2_full_out.txt
grep -q 'EXACT_ANCHORS' /tmp/rupture_r2_full_out.txt

echo "-- Ord 2″ subspace alignment (composed annihilation instrument) --"
$PYTHON scripts/research/rupture_ord2_alignment_contract.py | tee /tmp/rupture_ord2_out.txt
grep -q 'ORD2_CONTRACT_OK' /tmp/rupture_ord2_out.txt
grep -q 'ORD2_VERDICT ORD2_INSTRUMENT_OK' /tmp/rupture_ord2_out.txt
grep -q 'ALIGN_SEPARATION aligned_vs_rotating -> PASS' /tmp/rupture_ord2_out.txt
grep -q 'GAP_ALONE_INVALID_AS_DISCRIMINANT -> PASS' /tmp/rupture_ord2_out.txt

echo "-- Ord 2″ trained LSTM multi-path (non-sedenion target) --"
$PYTHON scripts/research/rupture_ord2_trained_lstm_probe.py | tee /tmp/rupture_ord2_trained_out.txt
grep -q 'ORD2_TRAINED_CONTRACT_OK' /tmp/rupture_ord2_trained_out.txt
# Accept NO_SIGNATURE (expected) or SKIP (no torch); reject BROKEN/FAIL
if grep -q 'ORD2_TRAINED_VERDICT ORD2_TRAINED_BROKEN' /tmp/rupture_ord2_trained_out.txt; then
  echo "trained LSTM probe broken" >&2
  exit 1
fi
if grep -q 'ORD2_TRAINED_VERDICT ORD2_TRAINED_SUBSPACE_DEATH' /tmp/rupture_ord2_trained_out.txt; then
  echo "unexpected SUBSPACE_DEATH — requires human review before green gate" >&2
  exit 1
fi
grep -qE 'ORD2_TRAINED_VERDICT ORD2_TRAINED_(NO_SIGNATURE|SKIP)' /tmp/rupture_ord2_trained_out.txt

echo "-- Ord 2″ trained diagonal S4 multi-path (structured SSM family) --"
$PYTHON scripts/research/rupture_ord2_trained_s4_probe.py | tee /tmp/rupture_ord2_s4_out.txt
grep -q 'ORD2_S4_CONTRACT_OK' /tmp/rupture_ord2_s4_out.txt
if grep -q 'ORD2_S4_VERDICT ORD2_S4_BROKEN' /tmp/rupture_ord2_s4_out.txt; then
  echo "trained S4 probe broken" >&2
  exit 1
fi
if grep -q 'ORD2_S4_VERDICT ORD2_S4_SUBSPACE_DEATH' /tmp/rupture_ord2_s4_out.txt; then
  echo "unexpected S4 SUBSPACE_DEATH — requires human review before green gate" >&2
  exit 1
fi
grep -qE 'ORD2_S4_VERDICT ORD2_S4_(NO_SIGNATURE|SKIP)' /tmp/rupture_ord2_s4_out.txt

echo "-- Ord 2″ protocol §5 performance–alignment link --"
$PYTHON scripts/research/rupture_ord2_perf_link_probe.py | tee /tmp/rupture_ord2_perf_out.txt
grep -q 'ORD2_PERF_CONTRACT_OK' /tmp/rupture_ord2_perf_out.txt
if grep -q 'ORD2_PERF_VERDICT ORD2_PERF_BROKEN' /tmp/rupture_ord2_perf_out.txt; then
  echo "perf-link probe broken" >&2
  exit 1
fi
if grep -q 'ORD2_PERF_VERDICT ORD2_PERF_LINK_PRESENT' /tmp/rupture_ord2_perf_out.txt; then
  echo "unexpected PERF_LINK_PRESENT — requires human review before green gate" >&2
  exit 1
fi
grep -qE 'ORD2_PERF_VERDICT ORD2_PERF_(NO_LINK|SKIP)' /tmp/rupture_ord2_perf_out.txt

echo "-- R3 Fano-restriction probe (Φ_fp) --"
$PYTHON scripts/research/rupture_r3_fano_restriction_probe.py | tee /tmp/rupture_r3_out.txt
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
$PYTHON scripts/research/rupture_r4_fano_field_contract.py | tee /tmp/rupture_r4_out.txt
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
test -f docs/research/rupture-ord2-alignment_2026-07-25.md
test -f docs/research/rupture-ord2-trained-lstm_2026-07-25.md
test -f docs/research/rupture-ord2-trained-s4_2026-07-25.md
test -f docs/research/rupture-ord2-perf-link_2026-07-25.md
test -f docs/research/rupture-r2-full-tubular_2026-07-25.md
test -f docs/research/rupture-r3-fano-phi_2026-07-25.md
test -f docs/research/rupture-r4-fano-field_2026-07-25.md

echo "RUPTURE_ABCD_CONTRACTS_OK"
