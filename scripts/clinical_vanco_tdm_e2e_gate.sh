#!/usr/bin/env bash
# scripts/clinical_vanco_tdm_e2e_gate.sh
# Vancomycin AUC/MIC TDM decision E2E under lean_single.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/stdlib/clinical/test_vanco_auc_tdm_e2e.sio"
MOD="stdlib/darwin_pbpk/pd/vancomycin_auc_gum.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/vanco_tdm.elf"
LOG="$OUT/run.log"
MODLOG="$OUT/mod.log"
RECEIPT_DIR="$ROOT/artifacts/clinical"
RECEIPT="$RECEIPT_DIR/vanco_tdm_e2e_receipt.v1.json"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
fail=0

echo "== clinical_vanco_tdm_e2e_gate: engine=$SOUNIO_SOUC_ENGINE =="

# 1) Existing module selftest still green
echo "== module selftest $MOD =="
if ! "$SOUC" compile "$MOD" -o "$OUT/mod.elf" >"$OUT/mod_compile.log" 2>&1; then
  echo "FAIL: module compile"; tail -20 "$OUT/mod_compile.log" || true; fail=1
else
  chmod +x "$OUT/mod.elf"
  if ! "$OUT/mod.elf" >"$MODLOG" 2>&1; then
    echo "FAIL: module run"; cat "$MODLOG" || true; fail=1
  elif ! grep -q "ALL PASS" "$MODLOG"; then
    echo "FAIL: module missing ALL PASS"; cat "$MODLOG" || true; fail=1
  fi
fi

# 2) E2E driver
echo "== e2e driver $SRC =="
if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: e2e compile"; tail -30 "$OUT/compile.log" || true; fail=1
else
  chmod +x "$ELF"
  if ! "$ELF" >"$LOG" 2>&1; then
    echo "FAIL: e2e run"; cat "$LOG" || true; fail=1
  elif ! grep -q "CLINICAL_VANCO_TDM_E2E_OK" "$LOG"; then
    echo "FAIL: missing sentinel"; cat "$LOG" || true; fail=1
  else
    grep '^VANCO_TDM_E2E' "$LOG" || true
  fi
fi

# 3) Compile-fail witnesses still present (structural refuse path)
for f in \
  tests/compile-fail/med/vancomycin_low_conf_refusal.sio \
  tests/compile-fail/med/vancomycin_weak_evidence_refusal.sio
do
  if [[ ! -f "$f" ]]; then
    echo "FAIL: missing compile-fail witness $f"
    fail=1
  else
    echo "OK witness present: $f"
  fi
done

mkdir -p "$RECEIPT_DIR"
STATUS="fail"
[[ $fail -eq 0 ]] && STATUS="pass"
STD_RATIO="$(grep 'std_ratio=' "$LOG" 2>/dev/null | head -1 | tr ' ' '\n' | grep '^std_ratio=' | cut -d= -f2- || echo null)"
REN_DEC="$(grep 'ren_dec=' "$LOG" 2>/dev/null | head -1 | tr ' ' '\n' | grep '^ren_dec=' | cut -d= -f2- || echo null)"

cat >"$RECEIPT" <<EOF
{
  "schema": "clinical_vanco_tdm_e2e_receipt.v1",
  "status": "$STATUS",
  "engine": "$SOUNIO_SOUC_ENGINE",
  "commit": "$COMMIT",
  "source": "$SRC",
  "module_selftest": "$MOD",
  "sounio": {
    "std_ratio": $STD_RATIO,
    "ren_dec": $REN_DEC
  },
  "claims": [
    "gum_auc_mic_seven_source_budget",
    "knightian_pbox_recommend_adjust_refuse",
    "renal_impairment_refuse_same_mg_per_kg",
    "compile_fail_witnesses_present"
  ],
  "claims_not_made": [
    "bedside_dosing_product",
    "nonmem_foce_parity",
    "madaros_multimodule",
    "mimic_real_tdm_calibration",
    "numpy_sklearn"
  ]
}
EOF
echo "receipt: $RECEIPT"

if [[ $fail -eq 0 ]]; then
  echo "CLINICAL_VANCO_TDM_E2E_GATE_OK"
  exit 0
fi
exit 1
