#!/usr/bin/env bash
# scripts/madaros_root2_method_gate.sh
#
# Madaros Root 2 gate: associated Type::method + same-module &self methods.
# Multi-module instance method calls remain residual (document, do not fail gate).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== madaros_root2_method_gate =="
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
  echo "PASS: $sentinel"
}

run_ok "same-module method+associated" \
  tests/run-pass/madaros_root2_method_associated.sio \
  ROOT2_METHOD_ASSOCIATED_OK

run_ok "multi-module associated import" \
  tests/run-pass/madaros_root2_associated_import.sio \
  ROOT2_ASSOCIATED_IMPORT_OK

# Residual: multi-module instance method — expected compile fail under Madaros
echo "== residual multi-module instance method (expect fail) =="
cat >"$OUT/mm.sio" <<'EOF'
use epistemic::knowledge::{Epistemic, ep_measured}
fn main() -> i32 with IO, Mut, Div, Panic {
    let e = ep_measured(10.0, 0.5)
    print(e.val())
    print("\n")
    print("UNEXPECTED_MM_METHOD_OK\n")
    return 0
}
EOF
if "$SOUC" compile "$OUT/mm.sio" -o "$OUT/mm.elf" >"$OUT/mmc.log" 2>&1; then
  echo "NOTE: multi-module instance method COMPILED — residual may be FIXED; update audit"
else
  echo "OK multi-module instance method still blocked (residual documented)"
fi

mkdir -p "$ROOT/artifacts/compiler"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/compiler/madaros_root2_method_receipt.v1.json" <<EOF
{
  "schema": "madaros_root2_method_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "commit": "$COMMIT",
  "claims": [
    "same_module_self_ref_method_call",
    "same_module_associated_type_method",
    "same_module_method_on_method_return",
    "multimodule_associated_type_method_import"
  ],
  "claims_not_made": [
    "multimodule_instance_method_call",
    "full_root2_null_deref_closed",
    "enum_ctor_path"
  ]
}
EOF
echo "receipt: $ROOT/artifacts/compiler/madaros_root2_method_receipt.v1.json"

if [[ $fail -eq 0 ]]; then
  echo "MADAROS_ROOT2_METHOD_GATE_OK"
  exit 0
fi
exit 1
