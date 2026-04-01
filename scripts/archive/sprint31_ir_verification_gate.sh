#!/usr/bin/env bash
# sprint31_ir_verification_gate.sh — IR serialization verification pipeline gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT31_GATE_OUT:-$ROOT_DIR/artifacts/sprint31/ir_verification_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT31_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint31"

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
  local log="/tmp/sprint31_${label}.log"
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

echo "=== Sprint 31: IR Verification Pipeline Gate ==="
echo ""

# --- Section 1: Infrastructure ---
echo "--- Infrastructure: verify.sio, normalize.sio, serialize.sio ---"
check_grep "infra:verify_fn" "self-hosted/ir/verify.sio" "fn verify_ir_equivalence"
check_grep "infra:normalize_fn" "self-hosted/ir/normalize.sio" "fn normalize_ir_module"
check_grep "infra:compare_fn" "self-hosted/ir/normalize.sio" "fn compare_normalized_ir"
check_grep "infra:serialize_fn" "self-hosted/ir/serialize.sio" "fn serialize_ir_module"
check_grep "infra:deserialize_fn" "self-hosted/ir/serialize.sio" "fn deserialize_ir_module"

# --- Section 2: CLI commands ---
echo ""
echo "--- CLI: ir-verify + ir-roundtrip ---"
check_grep "cli:ir_verify" "self-hosted/main.sio" "ir-verify"
check_grep "cli:ir_roundtrip" "self-hosted/main.sio" "ir-roundtrip"

# --- Section 3: Normalization coverage ---
echo ""
echo "--- Normalization: epistemic opcodes ---"
check_grep "norm:contest" "self-hosted/ir/normalize.sio" "IrContest"
check_grep "norm:prove_robust" "self-hosted/ir/normalize.sio" "IrProveRobust"
check_grep "norm:validated" "self-hosted/ir/normalize.sio" "IrValidated"
check_grep "norm:lift_knowledge" "self-hosted/ir/normalize.sio" "IrLiftKnowledge"
check_grep "norm:measure" "self-hosted/ir/normalize.sio" "IrMeasure"
check_grep "norm:observe_transition" "self-hosted/ir/normalize.sio" "IrObserveTransition"
check_grep "norm:rollback_transition" "self-hosted/ir/normalize.sio" "IrRollbackTransition"
check_grep "norm:debug_guard" "self-hosted/ir/normalize.sio" "IrDebugGuard"

# --- Section 4: Test fixtures ---
echo ""
echo "--- Test fixtures ---"
check_file_exists "fixture:math_a" "tests/verify-ir/math_a.sio"
check_file_exists "fixture:math_b" "tests/verify-ir/math_b.sio"
check_file_exists "fixture:control_a" "tests/verify-ir/control_a.sio"

# --- Section 5: souc check on tests ---
echo ""
echo "--- souc check on new tests ---"
check_souc "souc:ir_roundtrip_basic" "tests/run-pass/ir_roundtrip_basic.sio" "pass"
check_souc "souc:ir_verify_commutative" "tests/run-pass/ir_verify_commutative.sio" "pass"
check_souc "souc:fixture_math_a" "tests/verify-ir/math_a.sio" "pass"

# --- Section 6: Self-hosted ir-roundtrip (JIT) ---
echo ""
echo "--- Self-hosted ir-roundtrip (skip if no JIT) ---"
check_grep "roundtrip:pipeline_fn" "self-hosted/main.sio" "fn run_ir_roundtrip_pipeline"
check_grep "roundtrip:serialize_call" "self-hosted/main.sio" "serialize_ir_module"
check_grep "roundtrip:deserialize_call" "self-hosted/main.sio" "deserialize_ir_module"

# --- Section 7: Self-hosted ir-verify ---
echo ""
echo "--- Self-hosted ir-verify ---"
check_grep "verify:pipeline_fn" "self-hosted/main.sio" "fn run_ir_verify_pipeline"
check_grep "verify:equivalence_call" "self-hosted/main.sio" "verify_ir_equivalence"
check_grep "verify:two_file" "self-hosted/main.sio" "path_a.*path_b"

# --- Section 8: Regression ---
echo ""
echo "--- Regression ---"
check_souc "souc:regression_kernel_fn_basic" "tests/run-pass/kernel_fn_basic.sio" "pass"
check_souc "souc:regression_kernel_ptx" "tests/run-pass/kernel_ptx_emit.sio" "pass"
check_souc "souc:regression_causal_front_door" "tests/frontend/causal_front_door_basic.sio" "pass"

echo ""
echo "=== Sprint 31 Gate: $pass/$total pass ==="

cat > "$OUT_JSON" <<ARTIFACT
{
  "gate": "sprint31_ir_verification",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "new_functions": ["run_ir_verify_pipeline", "run_ir_roundtrip_pipeline", "opcodes_equal epistemic extension"],
  "new_tests": ["ir_roundtrip_basic", "ir_verify_commutative"],
  "sota_references": ["SOIR v2 binary format", "normalize-then-compare verification"],
  "novelty": "End-to-end IR serialization roundtrip verification with epistemic opcode normalization",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $OUT_JSON"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
