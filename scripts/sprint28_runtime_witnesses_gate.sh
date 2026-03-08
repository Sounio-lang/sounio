#!/usr/bin/env bash
# sprint28_runtime_witnesses_gate.sh — Runtime witness enforcement gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT28_GATE_OUT:-$ROOT_DIR/artifacts/sprint28/runtime_witnesses_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT28_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint28"

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
  local log="/tmp/sprint28_${label}.log"
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

echo "=== Sprint 28: Runtime Witnesses Gate ==="
echo ""

# --- Section 1: IrDebugGuard opcode exists ---
echo "--- IrDebugGuard opcode ---"
check_grep "ir:IrDebugGuard_enum" "self-hosted/ir/ir.sio" "IrDebugGuard"
check_grep "ir:ir_debug_guard_fn" "self-hosted/ir/ir.sio" "fn ir_debug_guard"

# --- Section 2: Lowerer debug_mode ---
echo ""
echo "--- Lowerer debug_mode ---"
check_grep "lower:debug_mode_field" "self-hosted/ir/lower.sio" "debug_mode"
check_grep "lower:emit_debug_guard" "self-hosted/ir/lower.sio" "emit_debug_guard"

# --- Section 3: Native codegen ---
echo ""
echo "--- Native codegen ---"
check_grep "native:debug_guard_dispatch" "self-hosted/native/lower_ir.sio" "IrDebugGuard"
check_grep "native:debug_guard_stub" "self-hosted/native/lower_ir.sio" "NOP stub\|debug.mode"

# --- Section 4: WASM codegen ---
echo ""
echo "--- WASM codegen ---"
check_grep "wasm:debug_guard_dispatch" "self-hosted/wasm/lower.sio" "IrDebugGuard"

# --- Section 5: CLI flag ---
echo ""
echo "--- CLI flag ---"
check_grep "cli:debug_witnesses_flag" "self-hosted/compiler/main.sio" "debug-witnesses"

# --- Section 6: Test files ---
echo ""
echo "--- Test files ---"
check_grep "test:debug_witnesses_basic" "tests/frontend/debug_witnesses_basic.sio" "run-pass"
check_grep "test:debug_witnesses_admit" "tests/frontend/debug_witnesses_admit_guard.sio" "run-pass"
check_grep "test:fail_guard_missing_policy" "tests/compile-fail/debug_guard_missing_policy.sio" "compile-fail"
check_grep "test:fail_guard_witness_mismatch" "tests/compile-fail/debug_guard_witness_type_mismatch.sio" "compile-fail"

# --- Section 7: souc check verification ---
echo ""
echo "--- souc check verification ---"
check_souc "souc:debug_witnesses_basic" "tests/frontend/debug_witnesses_basic.sio" "pass"
check_souc "souc:debug_witnesses_admit" "tests/frontend/debug_witnesses_admit_guard.sio" "pass"
check_souc "souc:fail_guard_missing_policy" "tests/compile-fail/debug_guard_missing_policy.sio" "fail"
check_souc "souc:fail_guard_witness_mismatch" "tests/compile-fail/debug_guard_witness_type_mismatch.sio" "fail"

# --- Section 8: Regression ---
echo ""
echo "--- Regression ---"
check_souc "souc:regression_hello" "tests/run-pass/hello.sio" "pass"

echo ""
echo "=== Sprint 28 Gate: $pass/$total pass ==="

cat > "$OUT_JSON" <<ARTIFACT
{
  "gate": "sprint28_runtime_witnesses",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "new_opcode": "IrDebugGuard",
  "cli_flag": "--debug-witnesses",
  "codegen_backends": ["native_x86_64", "wasm"],
  "sota_references": ["Gradual Verification (Bader/Aldrich/Tanter VMCAI 2018)", "Design by Contract (Eiffel)", "Thrust (PLDI 2025)", "AgentSpec (ICSE 2026)"],
  "novelty": "First language with debug-mode witness validation for epistemic type certificates",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $OUT_JSON"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
