#!/usr/bin/env bash
# scripts/epistemic_prescription_chain_e2e_gate.sh
#
# LEVEL-3 ousadia gate:
#   1) Default Madaros multi-module import of epistemic::gum with CORRECT k95
#      (D1 GUM-site fix: arithmetic-sourced f64→i64 in dof_to_i64)
#   2) Vanco GUM → AUC/MIC → Knightian ADMIT/ADJUST/REFUSE
#   3) Compile-fail confidence witnesses present
#
# Does NOT pin lean_single — runs under default ./bin/souc (Madaros).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
# Explicitly clear lean_single pin — this gate claims default Madaros.
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== epistemic_prescription_chain_e2e_gate =="
echo "== engine: default Madaros (no lean_single pin) =="
"$SOUC" --version 2>&1 | head -2 || true

# --- D1 k95 smoke under multi-module import ---
echo "== D1 k95 import smoke =="
cat >"$OUT/k95.sio" <<'EOF'
use epistemic::gum::{gum_type_a, gum_type_b, gum_combine2, gum_k95, gum_dof}
fn main() -> i32 with IO, Mut, Div, Panic {
    let a = gum_type_a(2.0, 5)
    let z = gum_type_b(0.0)
    let r = gum_combine2(100.0, a, z)
    let k = gum_k95(r)
    let d = gum_dof(r)
    print("K95_SMOKE k95="); print(k); print(" dof="); print(d); print("\n")
    if k < 2.5 { print("STILL_CORRUPT\n"); return 1 }
    if d < 3.5 || d > 4.5 { return 2 }
    print("K95_SMOKE_OK\n")
    return 0
}
EOF
if ! "$SOUC" compile "$OUT/k95.sio" -o "$OUT/k95.elf" >"$OUT/k95c.log" 2>&1; then
  echo "FAIL: k95 smoke compile"; tail -20 "$OUT/k95c.log" || true; fail=1
else
  chmod +x "$OUT/k95.elf"
  if ! "$OUT/k95.elf" >"$OUT/k95.log" 2>&1 || ! grep -q "K95_SMOKE_OK" "$OUT/k95.log"; then
    echo "FAIL: k95 smoke run"; cat "$OUT/k95.log" || true; fail=1
  else
    grep 'K95_SMOKE' "$OUT/k95.log" || true
  fi
fi

# --- Prescription chain ---
SRC="tests/stdlib/clinical/test_prescription_chain_e2e.sio"
echo "== prescription chain $SRC =="
if ! "$SOUC" compile "$SRC" -o "$OUT/rx.elf" >"$OUT/rxc.log" 2>&1; then
  echo "FAIL: chain compile"; tail -30 "$OUT/rxc.log" || true; fail=1
else
  chmod +x "$OUT/rx.elf"
  if ! "$OUT/rx.elf" >"$OUT/rx.log" 2>&1; then
    echo "FAIL: chain run"; cat "$OUT/rx.log" || true; fail=1
  elif ! grep -q "EPISTEMIC_PRESCRIPTION_CHAIN_E2E_OK" "$OUT/rx.log"; then
    echo "FAIL: missing sentinel"; cat "$OUT/rx.log" || true; fail=1
  else
    grep '^RX_CHAIN' "$OUT/rx.log" || true
  fi
fi

# --- Compile-fail witnesses present ---
for f in \
  tests/compile-fail/med/vancomycin_low_conf_refusal.sio \
  tests/compile-fail/med/vancomycin_weak_evidence_refusal.sio
do
  if [[ ! -f "$f" ]]; then echo "FAIL: missing $f"; fail=1
  else echo "OK witness: $f"; fi
done

# --- Knowledge import still blocked (honest boundary) ---
echo "== Knowledge import boundary (expect fail under Madaros) =="
cat >"$OUT/know.sio" <<'EOF'
use epistemic::knowledge::{measure}
fn main() -> i32 with IO { print("UNEXPECTED_KNOWLEDGE_OK\n"); return 0 }
EOF
if "$SOUC" compile "$OUT/know.sio" -o "$OUT/know.elf" >"$OUT/knowc.log" 2>&1; then
  echo "NOTE: Knowledge import compiled — update trust map if intentional"
  # not a hard fail if Knowledge got fixed
else
  echo "OK Knowledge import still blocked (D3 boundary documented)"
fi

mkdir -p "$ROOT/artifacts/clinical"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
K95=$(grep 'k95=' "$OUT/rx.log" 2>/dev/null | head -1 | tr ' ' '\n' | grep '^k95=' | cut -d= -f2- || echo null)
DEC=$(grep 'decide=' "$OUT/rx.log" 2>/dev/null | head -1 | tr ' ' '\n' | grep '^decide=' | cut -d= -f2- || echo null)
# JSON-safe: numeric k95, string decide
if [[ -z "$K95" || "$K95" == "null" ]]; then K95_JSON=null; else K95_JSON="$K95"; fi
if [[ -z "$DEC" || "$DEC" == "null" ]]; then DEC_JSON=null; else DEC_JSON="\"$DEC\""; fi

cat >"$ROOT/artifacts/clinical/prescription_chain_e2e_receipt.v1.json" <<EOF
{
  "schema": "epistemic_prescription_chain_e2e_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "lean_single_pin": false,
  "commit": "$COMMIT",
  "sounio": { "k95": $K95_JSON, "decide_std": $DEC_JSON },
  "d1_gum_fix": "dof_to_i64: arithmetic-source + round-half-up (Madaros #983 workaround; D5 avoided; FP ν_eff→table)",
  "claims": [
    "madaros_multimodule_gum_import_correct_k95",
    "gum_to_auc_mic_to_knightian_decision",
    "renal_refuse_same_mg_per_kg",
    "compile_fail_confidence_witnesses_present"
  ],
  "claims_not_made": [
    "knowledge_t_import_under_madaros",
    "bedside_dosing_product",
    "nonmem_foce",
    "full_d1_param_scalar_kind_fix_without_d5",
    "numpy_sklearn"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/clinical/prescription_chain_e2e_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "EPISTEMIC_PRESCRIPTION_CHAIN_E2E_GATE_OK"
  exit 0
fi
exit 1
