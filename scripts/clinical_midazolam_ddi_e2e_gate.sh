#!/usr/bin/env bash
# scripts/clinical_midazolam_ddi_e2e_gate.sh
# Midazolam CYP3A DDI E2E under lean_single.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== clinical_midazolam_ddi_e2e_gate: engine=$SOUNIO_SOUC_ENGINE =="

MOD="stdlib/darwin_pbpk/validation/midazolam_ddi.sio"
echo "== module $MOD =="
if ! "$SOUC" compile "$MOD" -o "$OUT/mod.elf" >"$OUT/modc.log" 2>&1; then
  echo "FAIL: module compile"; tail -25 "$OUT/modc.log" || true; fail=1
else
  chmod +x "$OUT/mod.elf"
  if ! "$OUT/mod.elf" >"$OUT/mod.log" 2>&1 || ! grep -q "ALL 6 TESTS PASSED" "$OUT/mod.log"; then
    echo "FAIL: module selftest"; cat "$OUT/mod.log" || true; fail=1
  else
    echo "OK module ALL 6 TESTS PASSED"
  fi
fi

SRC="tests/stdlib/clinical/test_midazolam_ddi_e2e.sio"
echo "== e2e $SRC =="
if ! "$SOUC" compile "$SRC" -o "$OUT/e2e.elf" >"$OUT/e2ec.log" 2>&1; then
  echo "FAIL: e2e compile"; tail -30 "$OUT/e2ec.log" || true; fail=1
else
  chmod +x "$OUT/e2e.elf"
  if ! "$OUT/e2e.elf" >"$OUT/e2e.log" 2>&1; then
    echo "FAIL: e2e run"; cat "$OUT/e2e.log" || true; fail=1
  elif ! grep -q "CLINICAL_MIDAZOLAM_DDI_E2E_OK" "$OUT/e2e.log"; then
    echo "FAIL: missing sentinel"; cat "$OUT/e2e.log" || true; fail=1
  else
    grep '^MDZ_DDI_E2E' "$OUT/e2e.log" || true
    # hard literature bands on printed tokens
    if ! grep -E 'aucr_keto=1[2-7]\.' "$OUT/e2e.log"; then
      # allow 14.x etc via broader check
      aucr=$(grep 'aucr_keto=' "$OUT/e2e.log" | head -1 | tr ' ' '\n' | grep '^aucr_keto=' | cut -d= -f2)
      python3 - <<PY
aucr=float("$aucr")
import sys
sys.exit(0 if 12.0 < aucr < 18.0 else 1)
PY
      if [[ $? -ne 0 ]]; then echo "FAIL: aucr band"; fail=1; fi
    fi
  fi
fi

mkdir -p "$ROOT/artifacts/clinical"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
F=$(grep 'f_oral=' "$OUT/e2e.log" 2>/dev/null | head -1 | tr ' ' '\n' | grep '^f_oral=' | cut -d= -f2- || echo null)
AUCR=$(grep 'aucr_keto=' "$OUT/e2e.log" 2>/dev/null | head -1 | tr ' ' '\n' | grep '^aucr_keto=' | cut -d= -f2- || echo null)

cat >"$ROOT/artifacts/clinical/midazolam_ddi_e2e_receipt.v1.json" <<EOF
{
  "schema": "clinical_midazolam_ddi_e2e_receipt.v1",
  "status": "$STATUS",
  "engine": "$SOUNIO_SOUC_ENGINE",
  "commit": "$COMMIT",
  "source": "$SRC",
  "module_selftest": "$MOD",
  "sounio": { "f_oral": $F, "aucr_keto": $AUCR },
  "literature": [
    "Heizmann 1984 Br J Anaesth (oral F)",
    "Olkkola 1994 Clin Pharmacol Ther (keto AUCR ~15x)",
    "Thummel 1996 Clin Pharmacol Ther (gut first-pass)",
    "Shih 2013 / Kuehl 2001 (CYP3A5)",
    "Wang 2011 (CYP3A4*22)"
  ],
  "claims": [
    "oral_f_in_heizmann_band",
    "ketoconazole_aucr_olkkola_band",
    "aucr_monotonic_in_i_over_ki",
    "gut_wall_required_for_full_aucr",
    "cyp3a5_expresser_faster_clh",
    "cyp3a4_22_slower_clh"
  ],
  "claims_not_made": [
    "pbpk28_ct_digitization",
    "bedside_ddi_product",
    "madaros_multimodule",
    "nonmem_foce",
    "numpy_sklearn"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/clinical/midazolam_ddi_e2e_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "CLINICAL_MIDAZOLAM_DDI_E2E_GATE_OK"
  exit 0
fi
exit 1
