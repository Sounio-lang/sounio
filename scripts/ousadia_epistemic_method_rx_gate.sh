#!/usr/bin/env bash
# scripts/ousadia_epistemic_method_rx_gate.sh
#
# OUSADIA gate: Epistemic methods under default Madaros drive ADMIT/ADJUST/REFUSE.
# No lean_single pin. Dual gum+knowledge import residual documented.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== ousadia_epistemic_method_rx_gate =="
echo "== engine: default Madaros (no lean_single) =="
"$SOUC" --version 2>&1 | head -2 || true

# 1) Flagship chain
SRC="tests/stdlib/clinical/test_epistemic_method_rx_chain_e2e.sio"
echo "== method RX chain $SRC =="
if ! "$SOUC" compile "$SRC" -o "$OUT/rx.elf" >"$OUT/rxc.log" 2>&1; then
  echo "FAIL: compile"; tail -30 "$OUT/rxc.log" || true; fail=1
else
  chmod +x "$OUT/rx.elf"
  if ! "$OUT/rx.elf" >"$OUT/rx.log" 2>&1 || ! grep -q "OUSADIA_EPISTEMIC_METHOD_RX_CHAIN_OK" "$OUT/rx.log"; then
    echo "FAIL: run"; cat "$OUT/rx.log" || true; fail=1
  else
    grep '^OUSADIA_RX' "$OUT/rx.log" || true
  fi
fi

# 2) GUM k95 still correct alone (D1 site)
echo "== GUM k95 multi-module smoke =="
cat >"$OUT/k95.sio" <<'EOF'
use epistemic::gum::{gum_type_a, gum_type_b, gum_combine2, gum_k95, gum_dof}
fn main() -> i32 with IO, Mut, Div, Panic {
    let a = gum_type_a(2.0, 5)
    let z = gum_type_b(0.0)
    let r = gum_combine2(100.0, a, z)
    let k = gum_k95(r)
    let d = gum_dof(r)
    print("K95 k95="); print(k); print(" dof="); print(d); print("\n")
    if k < 2.5 { return 1 }
    if d < 3.5 || d > 4.5 { return 2 }
    print("K95_OK\n")
    return 0
}
EOF
if ! "$SOUC" compile "$OUT/k95.sio" -o "$OUT/k95.elf" >"$OUT/k95c.log" 2>&1; then
  echo "FAIL: k95 compile"; tail -15 "$OUT/k95c.log" || true; fail=1
else
  chmod +x "$OUT/k95.elf"
  if ! "$OUT/k95.elf" >"$OUT/k95.log" 2>&1 || ! grep -q "K95_OK" "$OUT/k95.log"; then
    echo "FAIL: k95 run"; cat "$OUT/k95.log" || true; fail=1
  else
    grep 'K95' "$OUT/k95.log" || true
  fi
fi

# 3) Compile-fail confidence witnesses
for f in \
  tests/compile-fail/med/vancomycin_low_conf_refusal.sio \
  tests/compile-fail/med/vancomycin_weak_evidence_refusal.sio
do
  if [[ ! -f "$f" ]]; then echo "FAIL: missing $f"; fail=1
  else echo "OK witness: $f"; fi
done

# 4) Dual gum+knowledge import (required green — dual_import landing; do not regress)
echo "== dual gum+knowledge import (required) =="
if ! "$SOUC" compile tests/run-pass/madaros_dual_gum_knowledge.sio -o "$OUT/dual.elf" >"$OUT/dualc.log" 2>&1; then
  echo "FAIL: dual gum+knowledge compile"; tail -20 "$OUT/dualc.log" || true; fail=1
else
  chmod +x "$OUT/dual.elf"
  if ! "$OUT/dual.elf" >"$OUT/dual.log" 2>&1 || ! grep -q "DUAL_GUM_KNOWLEDGE_OK" "$OUT/dual.log"; then
    echo "FAIL: dual gum+knowledge run"; cat "$OUT/dual.log" || true; fail=1
  else
    echo "PASS: DUAL_GUM_KNOWLEDGE_OK"
  fi
fi

mkdir -p "$ROOT/artifacts/clinical"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/clinical/ousadia_epistemic_method_rx_receipt.v1.json" <<EOF
{
  "schema": "ousadia_epistemic_method_rx_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "lean_single_pin": false,
  "commit": "$COMMIT",
  "claims": [
    "epistemic_methods_under_madaros_multimodule",
    "measured_val_std_add_is_credible_drive_decision",
    "type_a_u95_t4_band_adjust",
    "renal_refuse_same_mg_per_kg",
    "compile_fail_confidence_witnesses_present",
    "dual_gum_and_knowledge_import_under_madaros"
  ],
  "claims_not_made": [
    "bedside_dosing_product",
    "nonmem_foce",
    "language_knowledge_t_generic",
    "numpy_sklearn"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/clinical/ousadia_epistemic_method_rx_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "OUSADIA_EPISTEMIC_METHOD_RX_GATE_OK"
  exit 0
fi
exit 1
