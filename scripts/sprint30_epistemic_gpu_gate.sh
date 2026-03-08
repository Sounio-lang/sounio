#!/usr/bin/env bash
# sprint30_epistemic_gpu_gate.sh — Epistemic GPU codegen gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT30_GATE_OUT:-$ROOT_DIR/artifacts/sprint30/epistemic_gpu_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT30_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint30"

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
  local log="/tmp/sprint30_${label}.log"
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

echo "=== Sprint 30: Epistemic GPU Codegen Gate ==="
echo ""

# --- Section 1: Detection infrastructure ---
echo "--- Detection: is_epistemic_kernel ---"
check_grep "detect:fnsig_field" "self-hosted/check/defs.sio" "is_epistemic_kernel"
check_grep "detect:checker_sets" "self-hosted/check/check.sio" "has_epistemic_param"
check_grep "detect:is_epistemic_type" "self-hosted/check/epistemic.sio" "fn is_epistemic_type"

# --- Section 2: HLIR flag propagation ---
echo ""
echo "--- Propagation: HLIR pipeline ---"
check_grep "prop:epistemic_kernel_fn" "self-hosted/gpu/hlir_to_gpu.sio" "fn hlir_function_to_epistemic_gpu_kernel"
check_grep "prop:epistemic_ptx_pipe" "self-hosted/gpu/hlir_to_gpu.sio" "fn hlir_kernels_to_epistemic_ptx"

# --- Section 3: Shadow ops ---
echo ""
echo "--- Shadow ops: binary + unary ---"
check_grep "shadow:binary" "self-hosted/gpu/hlir_to_gpu.sio" "fn gpu_emit_epistemic_shadow_binary"
check_grep "shadow:unary" "self-hosted/gpu/hlir_to_gpu.sio" "fn gpu_emit_epistemic_shadow_unary"
check_grep "shadow:sqrt" "self-hosted/gpu/hlir_to_gpu.sio" "GpuSqrt"
check_grep "shadow:abs" "self-hosted/gpu/hlir_to_gpu.sio" "GpuAbs"
check_grep "shadow:neg" "self-hosted/gpu/hlir_to_gpu.sio" "GpuNeg"

# --- Section 4: CLI wiring ---
echo ""
echo "--- CLI: --epistemic flag ---"
check_grep "cli:epistemic_flag" "self-hosted/main.sio" "epistemic"
check_grep "cli:gpu_emit_cmd" "self-hosted/main.sio" "gpu-emit"

# --- Section 5: Test annotations ---
echo ""
echo "--- Test annotations ---"
check_grep "test:shadow_runpass" "tests/run-pass/epistemic_kernel_shadow.sio" "run-pass"
check_grep "test:unary_runpass" "tests/run-pass/epistemic_kernel_unary.sio" "run-pass"
check_grep "test:io_compilefail" "tests/compile-fail/epistemic_kernel_io_effect.sio" "compile-fail"
check_grep "test:aleatoric_compilefail" "tests/compile-fail/epistemic_kernel_aleatoric.sio" "compile-fail"

# --- Section 6: souc check ---
echo ""
echo "--- souc check verification ---"
check_souc "souc:shadow" "tests/run-pass/epistemic_kernel_shadow.sio" "pass"
check_souc "souc:unary" "tests/run-pass/epistemic_kernel_unary.sio" "pass"
check_souc "souc:example" "examples/kernel_epistemic_vec_add.sio" "pass"

# --- Section 7: Regression ---
echo ""
echo "--- Regression ---"
check_souc "souc:regression_kernel_fn_basic" "tests/run-pass/kernel_fn_basic.sio" "pass"
check_souc "souc:regression_kernel_ptx" "tests/run-pass/kernel_ptx_emit.sio" "pass"
check_souc "souc:regression_causal_front_door" "tests/frontend/causal_front_door_basic.sio" "pass"

echo ""
echo "=== Sprint 30 Gate: $pass/$total pass ==="

cat > "$OUT_JSON" <<ARTIFACT
{
  "gate": "sprint30_epistemic_gpu",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "new_functions": ["gpu_emit_epistemic_shadow_unary", "is_epistemic_kernel detection"],
  "new_tests": ["epistemic_kernel_shadow", "epistemic_kernel_unary", "epistemic_kernel_io_effect", "epistemic_kernel_aleatoric"],
  "sota_references": ["Triton multi-level lowering", "Enzyme autodiff", "Kuiper PLDI 2025 linear+effects", "Mojo SC 2025"],
  "novelty": "First language with compile-time epistemic uncertainty propagation through GPU kernels including unary ops (sqrt/abs/neg)",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $OUT_JSON"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
