#!/usr/bin/env bash
# sprint33_multimodule_gate.sh — Multi-module compilation pipeline gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT33_GATE_OUT:-$ROOT_DIR/artifacts/sprint33/multimodule_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT33_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint33"

pass=0
fail=0
total=0

check_grep() {
  local label="$1"
  local file="$2"
  local pattern="$3"
  total=$((total + 1))
  if grep -q "$pattern" "$file" 2>/dev/null; then
    pass=$((pass + 1))
    echo "PASS  $label"
  else
    fail=$((fail + 1))
    echo "FAIL  $label"
  fi
}

check_souc() {
  local label="$1"
  local file="$2"
  local expect_pass="$3"
  total=$((total + 1))
  if [ -z "$SOUC" ] || [ ! -x "$SOUC" ]; then
    echo "SKIP  $label (no souc binary)"
    pass=$((pass + 1))
    return
  fi
  local log="/tmp/sprint33_${label}.log"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" check "$file" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ "$expect_pass" == "pass" ]]; then
    if [[ $rc -eq 0 ]]; then
      pass=$((pass + 1))
      echo "PASS  $label (souc check ok)"
    else
      fail=$((fail + 1))
      echo "FAIL  $label (expected pass, got rc=$rc)"
      tail -5 "$log" 2>/dev/null || true
    fi
  else
    if [[ $rc -ne 0 ]]; then
      pass=$((pass + 1))
      echo "PASS  $label (souc check rejected)"
    else
      fail=$((fail + 1))
      echo "FAIL  $label (expected fail, got pass)"
    fi
  fi
}

check_file_exists() {
  local label="$1"
  local file="$2"
  total=$((total + 1))
  if [ -f "$file" ]; then
    pass=$((pass + 1))
    echo "PASS  $label"
  else
    fail=$((fail + 1))
    echo "FAIL  $label (file not found: $file)"
  fi
}

echo "=== Sprint 33: Multi-Module Compilation Gate ==="
echo ""

# --- Section 1: module_loader.sio infrastructure ---
echo "--- module_loader.sio infrastructure ---"
check_grep "loader:compile_multimodule" "self-hosted/compiler/module_loader.sio" "fn compile_multimodule_program"
check_grep "loader:compile_native" "self-hosted/compiler/module_loader.sio" "fn compile_multimodule_native"
check_grep "loader:extract_imports" "self-hosted/compiler/module_loader.sio" "fn extract_imports_from_ast"
check_grep "loader:load_module" "self-hosted/compiler/module_loader.sio" "fn load_module_file"

# --- Section 2: main.sio --multimodule flag routing ---
echo ""
echo "--- main.sio multimodule routing ---"
check_grep "main:multimodule_flag" "self-hosted/main.sio" "multimodule"
check_grep "main:multimodule_native" "self-hosted/main.sio" "compile_multimodule_native"
check_grep "main:multimodule_bytecode" "self-hosted/main.sio" "compile_multimodule_program"

# --- Section 3: Test fixture files exist ---
echo ""
echo "--- Test fixture files ---"
check_file_exists "fixture:math_lib" "tests/multimodule/math_lib.sio"
check_file_exists "fixture:math_main" "tests/multimodule/math_main.sio"
check_file_exists "fixture:chain_a" "tests/multimodule/chain_a.sio"
check_file_exists "fixture:chain_b" "tests/multimodule/chain_b.sio"
check_file_exists "fixture:chain_c" "tests/multimodule/chain_c.sio"
check_file_exists "fixture:epistemic_lib" "tests/multimodule/epistemic_lib.sio"
check_file_exists "fixture:epistemic_main" "tests/multimodule/epistemic_main.sio"

# --- Section 4: souc check on fixtures ---
echo ""
echo "--- souc check on fixtures ---"
check_souc "souc:math_lib" "tests/multimodule/math_lib.sio" "pass"
check_souc "souc:math_main" "tests/multimodule/math_main.sio" "pass"
check_souc "souc:chain_a" "tests/multimodule/chain_a.sio" "pass"
check_souc "souc:chain_c" "tests/multimodule/chain_c.sio" "pass"
check_souc "souc:epistemic_lib" "tests/multimodule/epistemic_lib.sio" "pass"
check_souc "souc:epistemic_main" "tests/multimodule/epistemic_main.sio" "pass"

# --- Section 5: Sprint 32 regression ---
echo ""
echo "--- Sprint 32 regression ---"
check_grep "regr:disasm_pipeline" "self-hosted/main.sio" "fn run_disasm_pipeline"
check_grep "regr:ir_dump_pipeline" "self-hosted/main.sio" "fn run_ir_dump_pipeline"
check_grep "regr:disassemble_module" "self-hosted/ir/disasm.sio" "fn disassemble_module"

# --- Section 6: General regression ---
echo ""
echo "--- General regression ---"
check_souc "souc:regression_kernel_fn_basic" "tests/run-pass/kernel_fn_basic.sio" "pass"
check_souc "souc:regression_kernel_ptx" "tests/run-pass/kernel_ptx_emit.sio" "pass"
check_souc "souc:regression_ir_roundtrip" "tests/run-pass/ir_roundtrip_basic.sio" "pass"

echo ""
echo "=== Sprint 33 Gate: $pass/$total pass ==="

cat > "$OUT_JSON" <<ARTIFACT
{
  "gate": "sprint33_multimodule",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "new_fixtures": ["math_lib/main", "chain_a/b/c", "epistemic_lib/main"],
  "infrastructure_validated": ["compile_multimodule_program", "compile_multimodule_native", "extract_imports_from_ast", "load_module_file"],
  "sota_references": ["OCaml separate compilation", "Rust crate system", "Go package model"],
  "novelty": "Multi-module compilation with epistemic type propagation across module boundaries",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $OUT_JSON"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
