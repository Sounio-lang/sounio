#!/usr/bin/env bash
# scripts/madaros_root2_multimodule_method_gate.sh
#
# Multi-module instance methods under default Madaros (Root 2 completion).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_root2_multimodule_method_gate =="
"$SOUC" --version 2>&1 | head -2 || true

run_ok() {
  local name="$1" src="$2" sentinel="$3"
  echo "== $name =="
  if ! "$SOUC" compile "$src" -o "$OUT/t.elf" >"$OUT/c.log" 2>&1; then
    echo "FAIL: compile $src"; tail -15 "$OUT/c.log" || true; fail=1; return
  fi
  chmod +x "$OUT/t.elf"
  if ! "$OUT/t.elf" >"$OUT/r.log" 2>&1 || ! grep -q "$sentinel" "$OUT/r.log"; then
    echo "FAIL: run $src"; cat "$OUT/r.log" || true; fail=1; return
  fi
  grep -E 'OK|ROOT2' "$OUT/r.log" || true
  echo "PASS: $sentinel"
}

run_ok "multi-module Epistemic methods" \
  tests/run-pass/madaros_root2_multimodule_method.sio \
  ROOT2_MULTIMODULE_METHOD_OK

run_ok "same-module method+associated (regression)" \
  tests/run-pass/madaros_root2_method_associated.sio \
  ROOT2_METHOD_ASSOCIATED_OK

# Inline method chain on imported Epistemic (science path; was residual SEGV)
run_ok "multi-module Epistemic method chain" \
  tests/run-pass/madaros_root2_multimodule_method_chain.sio \
  ROOT2_MULTIMODULE_METHOD_CHAIN_OK

run_ok "same-module method chain (regression)" \
  tests/run-pass/madaros_root2_method_chain.sio \
  ROOT2_METHOD_CHAIN_OK

mkdir -p "$ROOT/artifacts/compiler"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/compiler/madaros_root2_multimodule_method_receipt.v1.json" <<EOF
{
  "schema": "madaros_root2_multimodule_method_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "commit": "$COMMIT",
  "claims": [
    "multimodule_instance_method_call",
    "multimodule_method_f64_print",
    "multimodule_inline_method_chain",
    "same_module_inline_method_chain",
    "epistemic_measured_val_add_std_import"
  ],
  "claims_not_made": [
    "enum_ctor_path",
    "full_root2_census_closed",
    "arbitrary_depth_method_chain_census"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/compiler/madaros_root2_multimodule_method_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "MADAROS_ROOT2_MULTIMODULE_METHOD_GATE_OK"
  exit 0
fi
exit 1
