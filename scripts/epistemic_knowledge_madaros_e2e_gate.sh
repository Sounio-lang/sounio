#!/usr/bin/env bash
# scripts/epistemic_knowledge_madaros_e2e_gate.sh
#
# D3 + Wave9 residual closeout: free-function AND method-form Epistemic API
# under default Madaros multi-module. Does NOT pin lean_single.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== epistemic_knowledge_madaros_e2e_gate =="
echo "== engine: default Madaros (no lean_single pin) =="
"$SOUC" --version 2>&1 | head -2 || true

# --- Module selftest under Madaros ---
echo "== knowledge.sio selftest (Madaros) =="
if ! "$SOUC" compile stdlib/epistemic/knowledge.sio -o "$OUT/self.elf" >"$OUT/selfc.log" 2>&1; then
  echo "FAIL: knowledge selftest compile"; tail -20 "$OUT/selfc.log" || true; fail=1
else
  chmod +x "$OUT/self.elf"
  if ! "$OUT/self.elf" >"$OUT/self.log" 2>&1 || ! grep -q "ALL PASS" "$OUT/self.log"; then
    echo "FAIL: knowledge selftest run"; cat "$OUT/self.log" || true; fail=1
  else
    grep -E 'PASS|ALL' "$OUT/self.log" || true
  fi
fi

# --- Import E2E ---
SRC="tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio"
echo "== import E2E $SRC =="
if ! "$SOUC" compile "$SRC" -o "$OUT/e2e.elf" >"$OUT/e2ec.log" 2>&1; then
  echo "FAIL: import E2E compile"; tail -30 "$OUT/e2ec.log" || true; fail=1
else
  chmod +x "$OUT/e2e.elf"
  if ! "$OUT/e2e.elf" >"$OUT/e2e.log" 2>&1 || ! grep -q "KNOWLEDGE_MADAROS_IMPORT_E2E_OK" "$OUT/e2e.log"; then
    echo "FAIL: import E2E run"; cat "$OUT/e2e.log" || true; fail=1
  else
    grep -E 'KNOW_|KNOWLEDGE_' "$OUT/e2e.log" || true
  fi
fi

# --- Trust harness ---
echo "== knowledge_trust.sio =="
if ! "$SOUC" compile tests/epistemic_trust/knowledge_trust.sio -o "$OUT/tr.elf" >"$OUT/trc.log" 2>&1; then
  echo "FAIL: knowledge_trust compile"; tail -20 "$OUT/trc.log" || true; fail=1
else
  chmod +x "$OUT/tr.elf"
  if ! "$OUT/tr.elf" >"$OUT/tr.log" 2>&1 || ! grep -q "KNOWLEDGE_TRUST_OK" "$OUT/tr.log"; then
    echo "FAIL: knowledge_trust run"; cat "$OUT/tr.log" || true; fail=1
  else
    echo "PASS: KNOWLEDGE_TRUST_OK"
  fi
fi

# --- Method-call form (Wave9 residual closeout: required green) ---
echo "== method-call form (required) =="
if ! "$SOUC" compile tests/epistemic_trust/witness_import_knowledge_method.sio -o "$OUT/meth.elf" >"$OUT/methc.log" 2>&1; then
  echo "FAIL: method-call form compile"; tail -20 "$OUT/methc.log" || true; fail=1
else
  chmod +x "$OUT/meth.elf"
  if ! "$OUT/meth.elf" >"$OUT/meth.log" 2>&1 || ! grep -q "KNOWLEDGE_METHOD_OK" "$OUT/meth.log"; then
    echo "FAIL: method-call form run"; cat "$OUT/meth.log" || true; fail=1
  else
    echo "PASS: KNOWLEDGE_METHOD_OK"
  fi
fi

# Free vs method parity
echo "== free vs method parity =="
if ! "$SOUC" compile tests/epistemic_trust/knowledge_method_parity.sio -o "$OUT/par.elf" >"$OUT/parc.log" 2>&1; then
  echo "FAIL: method parity compile"; tail -20 "$OUT/parc.log" || true; fail=1
else
  chmod +x "$OUT/par.elf"
  if ! "$OUT/par.elf" >"$OUT/par.log" 2>&1 || ! grep -q "KNOWLEDGE_METHOD_PARITY_OK" "$OUT/par.log"; then
    echo "FAIL: method parity run"; cat "$OUT/par.log" || true; fail=1
  else
    echo "PASS: KNOWLEDGE_METHOD_PARITY_OK"
  fi
fi

mkdir -p "$ROOT/artifacts/epistemic"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass

cat >"$ROOT/artifacts/epistemic/knowledge_madaros_e2e_receipt.v1.json" <<EOF
{
  "schema": "epistemic_knowledge_madaros_e2e_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "lean_single_pin": false,
  "commit": "$COMMIT",
  "d3_scope": "free_and_method_epistemic_api",
  "claims": [
    "madaros_multimodule_import_epistemic_knowledge_free_api",
    "madaros_multimodule_import_epistemic_knowledge_method_api",
    "ep_measured_add_mul_merge_gate_numeric",
    "free_vs_method_numeric_parity",
    "knowledge_sio_selftest_all_pass_under_madaros"
  ],
  "claims_not_made": [
    "language_knowledge_t_generic_import",
    "full_root2_census_closed",
    "numpy_sklearn"
  ],
  "closed_elsewhere": [
    "gum_k95_f64_i64_cast_fixed (Wave10: scripts/epistemic_trust_gate.sh Section A; #1252+#983)"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/epistemic/knowledge_madaros_e2e_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "EPISTEMIC_KNOWLEDGE_MADAROS_E2E_GATE_OK"
  exit 0
fi
exit 1
