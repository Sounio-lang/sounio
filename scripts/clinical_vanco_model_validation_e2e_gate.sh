#!/usr/bin/env bash
# scripts/clinical_vanco_model_validation_e2e_gate.sh
#
# Primary clinical MODEL validation drug = vancomycin (not rapamycin).
# Combines:
#   1) 2-comp IV endpoint validation vs Matzke/Roberts/ASHP bands
#   2) existing vancomycin_icu_pbpk.sio module ALL PASS
#   3) AUC/MIC TDM decision selftest (optional sibling gate)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== clinical_vanco_model_validation_e2e_gate: engine=$SOUNIO_SOUC_ENGINE =="
echo "== drug switch: vancomycin (open literature endpoints); rapamycin Ferron C(t) deferred =="

# Module selftest
MOD="stdlib/darwin_pbpk/validation/vancomycin_icu_pbpk.sio"
echo "== module $MOD =="
if ! "$SOUC" compile "$MOD" -o "$OUT/mod.elf" >"$OUT/modc.log" 2>&1; then
  echo "FAIL: module compile"; tail -20 "$OUT/modc.log" || true; fail=1
else
  chmod +x "$OUT/mod.elf"
  if ! "$OUT/mod.elf" >"$OUT/mod.log" 2>&1 || ! grep -q "ALL PASS" "$OUT/mod.log"; then
    echo "FAIL: module selftest"; cat "$OUT/mod.log" || true; fail=1
  else
    echo "OK module ALL PASS"
  fi
fi

# E2E driver
SRC="tests/stdlib/clinical/test_vanco_model_validation_e2e.sio"
echo "== e2e $SRC =="
if ! "$SOUC" compile "$SRC" -o "$OUT/e2e.elf" >"$OUT/e2ec.log" 2>&1; then
  echo "FAIL: e2e compile"; tail -30 "$OUT/e2ec.log" || true; fail=1
else
  chmod +x "$OUT/e2e.elf"
  if ! "$OUT/e2e.elf" >"$OUT/e2e.log" 2>&1; then
    echo "FAIL: e2e run"; cat "$OUT/e2e.log" || true; fail=1
  elif ! grep -q "CLINICAL_VANCO_MODEL_VALIDATION_E2E_OK" "$OUT/e2e.log"; then
    echo "FAIL: missing sentinel"; cat "$OUT/e2e.log" || true; fail=1
  else
    grep '^VANCO_MODEL_VAL' "$OUT/e2e.log" || true
  fi
fi

# Receipt
mkdir -p "$ROOT/artifacts/clinical"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/clinical/vanco_model_validation_receipt.v1.json" <<EOF
{
  "schema": "clinical_vanco_model_validation_e2e_receipt.v1",
  "status": "$STATUS",
  "engine": "$SOUNIO_SOUC_ENGINE",
  "commit": "$COMMIT",
  "primary_drug": "vancomycin",
  "literature": ["Matzke 1984", "Roberts 2011", "ASHP/SIDP/IDSA 2020"],
  "deferred_drug": "rapamycin",
  "deferred_reason": "Ferron 1997 C(t) series not open-access/digitizable via web+MCP; scaffold citation 61:696-708 is wrong paper (correct popPK is 61:416-428 PMID 9129559)",
  "claims": [
    "two_comp_iv_endpoint_validation_vs_open_literature_bands",
    "auc24_efficacy_safety_window_ashp2020",
    "renal_impairment_auc_increase",
    "module_selftest_all_pass"
  ],
  "claims_not_made": [
    "ferron_1997_ct_digitization",
    "pbpk28_rapamycin_gmfe_pass",
    "nonmem_foce",
    "bedside_dosing",
    "madaros_multimodule"
  ]
}
EOF

if [[ $fail -eq 0 ]]; then
  echo "CLINICAL_VANCO_MODEL_VALIDATION_E2E_GATE_OK"
  exit 0
fi
exit 1
