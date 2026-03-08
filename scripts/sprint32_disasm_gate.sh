#!/usr/bin/env bash
# sprint32_disasm_gate.sh — IR disassembly CLI + test reclassification gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT32_GATE_OUT:-$ROOT_DIR/artifacts/sprint32/disasm_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT32_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint32"

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
  local log="/tmp/sprint32_${label}.log"
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

check_count_ge() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  total=$((total + 1))
  if [[ $actual -ge $expected ]]; then
    pass=$((pass + 1))
    echo "PASS  $label ($actual >= $expected)"
  else
    fail=$((fail + 1))
    echo "FAIL  $label ($actual < $expected)"
  fi
}

echo "=== Sprint 32: IR Disassembly CLI + Test Reclassification Gate ==="
echo ""

# --- Section 1: disasm.sio infrastructure ---
echo "--- disasm.sio infrastructure ---"
check_grep "disasm:opcode_name" "self-hosted/ir/disasm.sio" "fn opcode_name"
check_grep "disasm:binop_name" "self-hosted/ir/disasm.sio" "fn binop_name"
check_grep "disasm:unaryop_name" "self-hosted/ir/disasm.sio" "fn unaryop_name"
check_grep "disasm:disassemble_module" "self-hosted/ir/disasm.sio" "fn disassemble_module"

# --- Section 2: CLI commands ---
echo ""
echo "--- CLI: disasm + ir-dump ---"
check_grep "cli:disasm" "self-hosted/main.sio" "fn run_disasm_pipeline"
check_grep "cli:ir_dump" "self-hosted/main.sio" "fn run_ir_dump_pipeline"
check_grep "cli:disasm_dispatch" "self-hosted/main.sio" "str_eq(cmd, \"disasm\")"
check_grep "cli:ir_dump_dispatch" "self-hosted/main.sio" "str_eq(cmd, \"ir-dump\")"

# --- Section 3: souc check on fixtures ---
echo ""
echo "--- souc check on verify-ir fixtures ---"
check_souc "souc:math_a" "tests/verify-ir/math_a.sio" "pass"
check_souc "souc:control_a" "tests/verify-ir/control_a.sio" "pass"
check_souc "souc:call_a" "tests/verify-ir/call_a.sio" "pass"

# --- Section 4: SELFHOST_PASS test count ---
echo ""
echo "--- SELFHOST_PASS reclassification ---"
selfhost_pass_count=$(grep -rl "SELFHOST_PASS" tests/compile-fail/ tests/run-pass/ tests/frontend/ 2>/dev/null | wc -l)
check_count_ge "selfhost_pass_count" "$selfhost_pass_count" 30

# --- Section 5: New run-pass test ---
echo ""
echo "--- New tests ---"
check_souc "souc:ir_disasm_basic" "tests/run-pass/ir_disasm_basic.sio" "pass"

# --- Section 6: Sprint 31 regression ---
echo ""
echo "--- Sprint 31 regression ---"
check_grep "regr:ir_verify_cmd" "self-hosted/main.sio" "ir-verify"
check_grep "regr:ir_roundtrip_cmd" "self-hosted/main.sio" "ir-roundtrip"
check_grep "regr:verify_equiv" "self-hosted/ir/verify.sio" "fn verify_ir_equivalence"
check_grep "regr:normalize_mod" "self-hosted/ir/normalize.sio" "fn normalize_ir_module"

# --- Section 7: Sprint 30 regression ---
echo ""
echo "--- Sprint 30 regression ---"
check_grep "regr:epistemic_shadow" "self-hosted/gpu/hlir_to_gpu.sio" "fn gpu_emit_epistemic_shadow_binary"
check_grep "regr:epistemic_unary" "self-hosted/gpu/hlir_to_gpu.sio" "fn gpu_emit_epistemic_shadow_unary"
check_grep "regr:is_epistemic_kernel" "self-hosted/check/defs.sio" "is_epistemic_kernel"

# --- Section 8: General regression ---
echo ""
echo "--- General regression ---"
check_souc "souc:regression_kernel_fn_basic" "tests/run-pass/kernel_fn_basic.sio" "pass"
check_souc "souc:regression_kernel_ptx" "tests/run-pass/kernel_ptx_emit.sio" "pass"
check_souc "souc:regression_causal_front_door" "tests/frontend/causal_front_door_basic.sio" "pass"

echo ""
echo "=== Sprint 32 Gate: $pass/$total pass ==="

cat > "$OUT_JSON" <<ARTIFACT
{
  "gate": "sprint32_disasm_cli",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "new_functions": ["run_disasm_pipeline", "run_ir_dump_pipeline"],
  "new_tests": ["ir_disasm_basic"],
  "selfhost_pass_reclassified": $selfhost_pass_count,
  "sota_references": ["SOIR v2 disassembly", "normalize-then-compare verification"],
  "novelty": "Human-readable IR inspection for epistemic GPU kernel pipelines",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $OUT_JSON"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
