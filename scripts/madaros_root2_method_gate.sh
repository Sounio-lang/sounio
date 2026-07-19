#!/usr/bin/env bash
# scripts/madaros_root2_method_gate.sh
#
# Madaros Root 2 gate: associated Type::method + &self methods across module boundaries.
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
  rm -f "$OUT/t.elf"
  if ! "$SOUC" compile "$src" -o "$OUT/t.elf" >"$OUT/c.log" 2>&1; then
    echo "FAIL: compile $src"; tail -15 "$OUT/c.log" || true; fail=1; return
  fi
  if [[ ! -s "$OUT/t.elf" ]]; then
    echo "FAIL: compile $src produced no fresh ELF"; fail=1; return
  fi
  chmod +x "$OUT/t.elf"
  if ! "$OUT/t.elf" >"$OUT/r.log" 2>&1 || ! grep -Fxq "$sentinel" "$OUT/r.log"; then
    echo "FAIL: run $src"; cat "$OUT/r.log" || true; fail=1; return
  fi
  echo "PASS: $sentinel"
}

is_fatal_log() {
  grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$1"
}

run_ambiguous_impl_authority_rejected() {
  local src="tests/compiler/module_graph_impl_authority_ambiguity/main.sio"
  local elf="$OUT/ambiguous-impl-authority.elf"
  local log="$OUT/ambiguous-impl-authority.log"
  local rc=0

  echo "== multi-module impl-method authority ambiguity (expect reject) =="
  rm -f "$elf"
  set +e
  "$SOUC" compile "$src" -o "$elf" >"$log" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 1 ]]; then
    echo "FAIL: ambiguous impl-method authority must reject with rc=1, got rc=$rc"
    cat "$log" || true
    fail=1
    return
  fi
  if is_fatal_log "$log"; then
    echo "FAIL: ambiguous impl-method authority produced a fatal compiler log"
    cat "$log" || true
    fail=1
    return
  fi
  if [[ -e "$elf" ]]; then
    echo "FAIL: ambiguous impl-method authority emitted an output artifact"
    ls -l "$elf" || true
    fail=1
    return
  fi
  if [[ $(grep -Fc 'MODULE_FRONTEND_PROVENANCE_FAILURE kind=target_identity' "$log" || true) -ne 1 ]] ||
     [[ $(grep -Fc 'instr_name=Authority_Collision_witness' "$log" || true) -ne 1 ]] ||
     [[ $(grep -Fxc 'IR merge failed: unresolved or ambiguous function provenance' "$log" || true) -ne 1 ]]; then
    echo "FAIL: ambiguous impl-method authority did not emit the exact provenance refusal"
    cat "$log" || true
    fail=1
    return
  fi
  if grep -Fq 'Compilation successful!' "$log" || grep -Fq 'Written to ' "$log"; then
    echo "FAIL: ambiguous impl-method authority reported output success"
    cat "$log" || true
    fail=1
    return
  fi
  echo "PASS: imported impl-method ambiguity rejected without output"
}

run_ok "same-module method+associated" \
  tests/run-pass/madaros_root2_method_associated.sio \
  ROOT2_METHOD_ASSOCIATED_OK

run_ok "multi-module associated import" \
  tests/run-pass/madaros_root2_associated_import.sio \
  ROOT2_ASSOCIATED_IMPORT_OK

run_ok "multi-module instance method" \
  tests/run-pass/madaros_root2_instance_import.sio \
  ROOT2_INSTANCE_IMPORT_OK

run_ambiguous_impl_authority_rejected

mkdir -p "$ROOT/artifacts/compiler"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
COMPILER_AUTHORITY=resolver_default
COMPILER_PATH="$(realpath "$SOUC")"
if [[ -n "${MADAROS_RAW_BIN:-}" ]]; then
  COMPILER_AUTHORITY=explicit_madaros_raw_bin
  COMPILER_PATH="$(realpath "$MADAROS_RAW_BIN")"
fi
COMPILER_SHA256="$(sha256sum "$COMPILER_PATH" | awk '{print $1}')"
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/compiler/madaros_root2_method_receipt.v1.json" <<EOF
{
  "schema": "madaros_root2_method_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "commit": "$COMMIT",
  "scope": "source_to_elf",
  "compiler_authority": "$COMPILER_AUTHORITY",
  "compiler_path": "$COMPILER_PATH",
  "compiler_sha256": "$COMPILER_SHA256",
  "claims": [
    "same_module_self_ref_method_call",
    "same_module_associated_type_method",
    "same_module_method_on_method_return",
    "multimodule_associated_type_method_import",
    "multimodule_instance_method_call",
    "multimodule_impl_method_authority_ambiguity_fail_closed"
  ],
  "claims_not_made": [
    "injective_impl_method_symbol_mangling",
    "soir_v4_roundtrip_semantics",
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
