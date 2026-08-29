#!/usr/bin/env bash
# scripts/madaros_knowledge_method_residual_gate.sh
#
# Wave9 residual closeout: Epistemic method form under default Madaros multi-module
# must compile, run, and match free-function ep_* numerics.
#
# FIXED_ALREADY on origin/main (Root 2 multi-module methods + dual import landings).
# This gate hard-requires the former residual; it must not silently degrade to free-only.
#
# Does not pin lean_single. Does not rebuild Madaros (uses bin/souc default engine).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_knowledge_method_residual_gate =="
echo "== engine: default Madaros (no lean_single pin) =="
"$SOUC" --version 2>&1 | head -2 || true

run_ok() {
  local name="$1" src="$2" sentinel="$3"
  echo "== $name =="
  if ! "$SOUC" compile "$src" -o "$OUT/t.elf" >"$OUT/c.log" 2>&1; then
    echo "FAIL: compile $src"
    tail -20 "$OUT/c.log" || true
    fail=1
    return
  fi
  chmod +x "$OUT/t.elf"
  if ! "$OUT/t.elf" >"$OUT/r.log" 2>&1 || ! grep -q "$sentinel" "$OUT/r.log"; then
    echo "FAIL: run $src"
    cat "$OUT/r.log" || true
    fail=1
    return
  fi
  grep -E 'OK|METHOD|KNOW|ROOT2|DUAL' "$OUT/r.log" || true
  echo "PASS: $sentinel"
}

# Primary residual: free vs method numeric parity (measured/val/add/mul/std/variance)
run_ok "free vs method parity" \
  tests/epistemic_trust/knowledge_method_parity.sio \
  KNOWLEDGE_METHOD_PARITY_OK

# Method-form smoke (associated + instance under multi-mod import)
run_ok "method form multi-mod" \
  tests/run-pass/madaros_knowledge_method_form.sio \
  KNOWLEDGE_METHOD_FORM_OK

# Legacy Root-2 trip-wire now required green
run_ok "legacy witness_import_knowledge_method" \
  tests/epistemic_trust/witness_import_knowledge_method.sio \
  KNOWLEDGE_METHOD_OK

# Free-function control must stay green (method promotion must not regress free path)
run_ok "free-function knowledge control" \
  tests/epistemic_trust/knowledge_trust.sio \
  KNOWLEDGE_TRUST_OK

# Sibling Root-2 multi-module method must not regress
if [[ -f tests/run-pass/madaros_root2_multimodule_method.sio ]]; then
  run_ok "root2 multimodule method regression" \
    tests/run-pass/madaros_root2_multimodule_method.sio \
    ROOT2_MULTIMODULE_METHOD_OK
fi

# Inline method chain on imported Epistemic (Root-2 chain residual closeout)
if [[ -f tests/run-pass/madaros_root2_multimodule_method_chain.sio ]]; then
  run_ok "root2 multimodule method chain regression" \
    tests/run-pass/madaros_root2_multimodule_method_chain.sio \
    ROOT2_MULTIMODULE_METHOD_CHAIN_OK
fi

mkdir -p "$ROOT/artifacts/compiler"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/compiler/madaros_knowledge_method_residual_receipt.v1.json" <<EOF
{
  "schema": "madaros_knowledge_method_residual_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "lean_single_pin": false,
  "commit": "$COMMIT",
  "verdict": "FIXED_ALREADY",
  "claims": [
    "epistemic_method_form_multimodule_import",
    "epistemic_measured_val_add_mul_std_method_path",
    "epistemic_inline_method_chain_multimodule",
    "free_vs_method_numeric_parity",
    "legacy_method_witness_now_required_green"
  ],
  "claims_not_made": [
    "language_knowledge_t_generic_import",
    "full_root2_census_closed",
    "enum_ctor_path",
    "arbitrary_depth_method_chain_census"
  ],
  "closed_elsewhere": [
    "gum_k95_f64_i64_cast_fixed (Wave10: scripts/epistemic_trust_gate.sh Section A; #1252+#983)",
    "root2_inline_method_chain (lower_method_recv_type MethodCall/Call; scripts/madaros_root2_method_gate.sh)"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/compiler/madaros_knowledge_method_residual_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "MADAROS_KNOWLEDGE_METHOD_RESIDUAL_GATE_OK"
  exit 0
fi
exit 1
