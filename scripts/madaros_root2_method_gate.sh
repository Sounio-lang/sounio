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

SOURCE_COMMIT="$(git rev-parse HEAD)"
SOURCE_TREE="$(git rev-parse HEAD^{tree})"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "FAIL: Root2 source-to-ELF receipt requires a clean source tree" >&2
  git status --short >&2
  exit 2
fi

COMPILER_AUTHORITY=resolved_souc_raw_elf
COMPILER_PATH=""
if [[ -n "${MADAROS_RAW_BIN:-}" ]]; then
  COMPILER_AUTHORITY=explicit_madaros_raw_bin
  COMPILER_PATH="$(realpath "$MADAROS_RAW_BIN")"
elif [[ -x "$SOUC" && "$(head -c4 "$SOUC" 2>/dev/null)" == $'\x7fELF' ]]; then
  COMPILER_AUTHORITY=explicit_souc_elf
  COMPILER_PATH="$(realpath "$SOUC")"
else
  COMPILER_PATH="$("$SOUC" info 2>/dev/null | awk -F: '/^raw_elf:/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }')"
fi
if [[ -z "$COMPILER_PATH" || ! -x "$COMPILER_PATH" || "$(head -c4 "$COMPILER_PATH" 2>/dev/null)" != $'\x7fELF' ]]; then
  echo "FAIL: unable to bind Root2 receipt to the executed compiler ELF" >&2
  exit 2
fi
COMPILER_PATH="$(realpath "$COMPILER_PATH")"
COMPILER_SHA256="$(sha256sum "$COMPILER_PATH" | awk '{print $1}')"

ROOT2_FIXTURE_MAINS=(
  tests/run-pass/madaros_root2_method_associated.sio
  tests/run-pass/madaros_root2_associated_import.sio
  tests/run-pass/madaros_root2_instance_import.sio
  tests/run-pass/madaros_root2_multimodule_method.sio
  tests/compiler/module_graph_impl_authority_reexport/main.sio
  tests/compiler/module_graph_impl_authority_ambiguity/main.sio
)
sha256sum "${ROOT2_FIXTURE_MAINS[@]}" >"$OUT/fixture-main.sha256"
FIXTURE_MANIFEST_SHA256="$(sha256sum "$OUT/fixture-main.sha256" | awk '{print $1}')"
: >"$OUT/emitted-elf.sha256"

echo "== madaros_root2_method_gate =="
"$SOUC" --version 2>&1 | head -2 || true

run_ok() {
  local key="$1" name="$2" src="$3" sentinel="$4"
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
  printf '%s %s\n' "$key" "$(sha256sum "$OUT/t.elf" | awk '{print $1}')" >>"$OUT/emitted-elf.sha256"
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
  sha256sum "$log" >"$OUT/ambiguity-negative-log.sha256"
  echo "PASS: imported impl-method ambiguity rejected without output"
}

run_ok same_module "same-module method+associated" \
  tests/run-pass/madaros_root2_method_associated.sio \
  ROOT2_METHOD_ASSOCIATED_OK

run_ok associated_import "multi-module associated import" \
  tests/run-pass/madaros_root2_associated_import.sio \
  ROOT2_ASSOCIATED_IMPORT_OK

run_ok instance_import "multi-module instance method" \
  tests/run-pass/madaros_root2_instance_import.sio \
  ROOT2_INSTANCE_IMPORT_OK

run_ok multimodule_f64 "multi-module imported method with f64 result" \
  tests/run-pass/madaros_root2_multimodule_method.sio \
  ROOT2_MULTIMODULE_METHOD_OK

run_ok facade_reexport "facade reexport preserves impl-method authority" \
  tests/compiler/module_graph_impl_authority_reexport/main.sio \
  MODULEGRAPH_IMPL_AUTHORITY_REEXPORT_OK

run_ambiguous_impl_authority_rejected

mkdir -p "$ROOT/artifacts/compiler"
EMITTED_ELF_MANIFEST_SHA256="$(sha256sum "$OUT/emitted-elf.sha256" | awk '{print $1}')"
SAME_MODULE_ELF_SHA256="$(awk '$1 == "same_module" { print $2 }' "$OUT/emitted-elf.sha256")"
ASSOCIATED_IMPORT_ELF_SHA256="$(awk '$1 == "associated_import" { print $2 }' "$OUT/emitted-elf.sha256")"
INSTANCE_IMPORT_ELF_SHA256="$(awk '$1 == "instance_import" { print $2 }' "$OUT/emitted-elf.sha256")"
MULTIMODULE_F64_ELF_SHA256="$(awk '$1 == "multimodule_f64" { print $2 }' "$OUT/emitted-elf.sha256")"
FACADE_REEXPORT_ELF_SHA256="$(awk '$1 == "facade_reexport" { print $2 }' "$OUT/emitted-elf.sha256")"
AMBIGUITY_LOG_SHA256=unavailable
if [[ -s "$OUT/ambiguity-negative-log.sha256" ]]; then
  AMBIGUITY_LOG_SHA256="$(awk '{print $1}' "$OUT/ambiguity-negative-log.sha256")"
fi
STATUS=fail
[[ $fail -eq 0 ]] && STATUS=pass
cat >"$ROOT/artifacts/compiler/madaros_root2_method_receipt.v1.json" <<EOF
{
  "schema": "madaros_root2_method_receipt.v1",
  "status": "$STATUS",
  "engine": "madaros_default",
  "commit": "$SOURCE_COMMIT",
  "source_tree": "$SOURCE_TREE",
  "source_clean": true,
  "scope": "source_to_elf",
  "compiler_authority": "$COMPILER_AUTHORITY",
  "compiler_path": "$COMPILER_PATH",
  "compiler_sha256": "$COMPILER_SHA256",
  "fixture_mains": [
    "tests/run-pass/madaros_root2_method_associated.sio",
    "tests/run-pass/madaros_root2_associated_import.sio",
    "tests/run-pass/madaros_root2_instance_import.sio",
    "tests/run-pass/madaros_root2_multimodule_method.sio",
    "tests/compiler/module_graph_impl_authority_reexport/main.sio",
    "tests/compiler/module_graph_impl_authority_ambiguity/main.sio"
  ],
  "fixture_main_manifest_sha256": "$FIXTURE_MANIFEST_SHA256",
  "emitted_elf_manifest_sha256": "$EMITTED_ELF_MANIFEST_SHA256",
  "emitted_elf_sha256": {
    "same_module": "$SAME_MODULE_ELF_SHA256",
    "associated_import": "$ASSOCIATED_IMPORT_ELF_SHA256",
    "instance_import": "$INSTANCE_IMPORT_ELF_SHA256",
    "multimodule_f64": "$MULTIMODULE_F64_ELF_SHA256",
    "facade_reexport": "$FACADE_REEXPORT_ELF_SHA256"
  },
  "ambiguity_negative_log_sha256": "$AMBIGUITY_LOG_SHA256",
  "claims": [
    "same_module_self_ref_method_call",
    "same_module_associated_type_method",
    "same_module_method_on_method_return",
    "multimodule_associated_type_method_import",
    "multimodule_instance_method_call",
    "multimodule_imported_method_f64_result",
    "multimodule_imported_epistemic_method_chain",
    "multimodule_impl_method_authority_through_facade_reexport",
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
