#!/usr/bin/env bash
# SCO Step III — E(τ) install gate.
# 1) Source markers present in lower.sio (theory-carrying strategy elevation).
# 2) SCO corpus still green (V1–V3 exact sentinels).
#
# Does NOT claim the currently shipping bin/souc ELF was rebuilt from this
# source. Rebuild separately with make build-madaros under the build lock.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOWER="self-hosted/ir/lower.sio"
fail=0

echo "### SCO E(τ) source install ###"
if [[ ! -f "$LOWER" ]]; then
  echo "FAIL: missing $LOWER"
  exit 1
fi

for marker in SCO_ETAU_INSTALL_V1 SCO_ETAU_ELEVATE_SCIENTIFIC_EFFECTS SCO_ETAU_ABLATION_ENV; do
  if grep -q "$marker" "$LOWER"; then
    echo "PASS: marker $marker"
  else
    echo "FAIL: marker $marker missing in $LOWER"
    fail=1
  fi
done

# Structural install: scientific effects must be named in the elevation helper.
for eff in NonAssoc Epistemic Observe; do
  if grep -q "\"$eff\"" "$LOWER" && grep -q "lowerer_sco_scientific_effects_ref" "$LOWER"; then
    echo "PASS: effect $eff referenced under SCO helper"
  else
    echo "FAIL: effect $eff / helper not wired"
    fail=1
  fi
done

# Strategy target must be precision-preserving for the elevation return.
if grep -A20 "lowerer_sco_scientific_effects_ref(effects)" "$LOWER" | grep -q "IR_STRATEGY_PRECISION_PRESERVING"; then
  echo "PASS: elevation returns IR_STRATEGY_PRECISION_PRESERVING"
else
  echo "FAIL: elevation does not return PRECISION_PRESERVING nearby"
  fail=1
fi

echo "### SCO corpus (Step II) ###"
if bash scripts/ci/sco_corpus_gate.sh; then
  echo "PASS: corpus"
else
  echo "FAIL: corpus"
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  echo "SCO_ETAU_INSTALL_GATE_FAIL"
  exit 1
fi

echo "SCO_ETAU_INSTALL_GATE_OK"
echo "NOTE: for runtime discharge under a rebuilt Madaros, run scripts/ci/sco_step4_runtime_gate.sh"
exit 0
