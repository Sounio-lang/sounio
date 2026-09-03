#!/usr/bin/env bash
# tests/gpu/gate_ptx_codegen.sh — GPU PTX codegen gate
#
# Validates the GPU kernel check pipeline and PTX codegen dispatch.
# Does NOT require GPU hardware — tests type-checking and code generation only.
#
# Usage: bash tests/gpu/gate_ptx_codegen.sh
# Exit:  0 = all non-SKIP tests pass, non-zero = failure count

set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"

PASS=0
FAIL=0
SKIP=0
TOTAL=0

pass() {
    TOTAL=$((TOTAL + 1))
    PASS=$((PASS + 1))
    echo "PASS  $1"
}

fail() {
    TOTAL=$((TOTAL + 1))
    FAIL=$((FAIL + 1))
    echo "FAIL  $1: $2"
}

skip() {
    TOTAL=$((TOTAL + 1))
    SKIP=$((SKIP + 1))
    echo "SKIP  $1: $2"
}

# ── check_souc: run `$SOUC check <file>`, expect exit 0 ───────────────────────

check_souc() {
    local label="$1"
    local file="$2"
    local log="/tmp/gate_ptx_${label}.log"
    local _ec=0
    timeout 30 "$SOUC" check "$file" >"$log" 2>&1 || _ec=$?
    if [ $_ec -eq 124 ]; then
        fail "$label" "timeout (30s)"
    elif [ $_ec -eq 0 ]; then
        pass "$label"
    else
        fail "$label" "exit $_ec — $(tail -3 "$log" | tr '\n' ' ')"
    fi
}

# ── check_souc_or_skip: skip when file absent, check otherwise ────────────────

check_souc_or_skip() {
    local label="$1"
    local file="$2"
    if [ ! -f "$file" ]; then
        skip "$label" "$file not found"
        return
    fi
    check_souc "$label" "$file"
}

run_souc_tail() {
    local file="$1"
    local log="/tmp/gate_ptx_run_$(basename "$file").log"
    local _ec=0
    timeout 30 "$SOUC" run "$file" >"$log" 2>&1 || _ec=$?
    if [ $_ec -eq 124 ]; then
        echo "timeout (30s)"
    else
        tail -1 "$log"
    fi
}

echo "=== GPU PTX Codegen Gate ==="
echo "SOUC:               $SOUC"
echo "SOUNIO_STDLIB_PATH: $SOUNIO_STDLIB_PATH"
echo ""

# ── Precondition ───────────────────────────────────────────────────────────────

if [ ! -x "$SOUC" ]; then
    echo "FATAL: souc binary not executable: $SOUC"
    exit 1
fi

# ── Section 1: souc check — required files ────────────────────────────────────

echo "--- Section 1: souc check (required) ---"

# T1
check_souc \
    "T1_kernel_vec_add" \
    "examples/kernel_vec_add.sio"

# T2
check_souc \
    "T2_kernel_source_level" \
    "examples/kernel_source_level.sio"

# T3
check_souc \
    "T3_gpu_kernel_basic" \
    "tests/run-pass/gpu_kernel_basic.sio"

# T4
check_souc \
    "T4_stdlib_test_gpu" \
    "tests/stdlib/gpu/test_gpu.sio"

# T5
check_souc \
    "T5_kernel_epistemic_wmma_matmul" \
    "examples/kernel_epistemic_wmma_matmul.sio"

# ── Section 2: GPU compile pipeline (self-hosted compiler) ────────────────────

echo ""
echo "--- Section 2: GPU compile pipeline ---"

# T6: GPU pipeline wiring — structural check (JIT runner OOM prevents full execution
#     of compiler/main.sio; validate via grep instead)
{
    TOTAL=$((TOTAL + 1))
    # Verify run_gpu_compile_pipeline is wired in compiler/main.sio with PTX+Metal+SPIR-V
    if grep -q "run_gpu_compile_pipeline" self-hosted/compiler/main.sio && \
       grep -q 'use hlir::lower::\*' self-hosted/compiler/main.sio && \
       grep -q 'use gpu::hlir_to_gpu::\*' self-hosted/compiler/main.sio && \
       grep -q 'hlir_kernels_to_ptx' self-hosted/compiler/main.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T6_gpu_pipeline_wired"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T6_gpu_pipeline_wired: run_gpu_compile_pipeline or imports missing from compiler/main.sio"
    fi
}

# T8: SPIR-V dispatch wired — structural check
{
    TOTAL=$((TOTAL + 1))
    # Verify hlir_kernels_to_spirv is imported and dispatched in compiler/main.sio
    if grep -q 'hlir_kernels_to_spirv_artifact' self-hosted/compiler/main.sio && \
       grep -q 'use gpu::epistemic_spirv::\*' self-hosted/compiler/main.sio && \
       grep -q 'hlir_kernels_to_spirv' self-hosted/gpu/hlir_to_gpu.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T8_spirv_dispatch_wired"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T8_spirv_dispatch_wired: hlir_kernels_to_spirv missing from main.sio or hlir_to_gpu.sio"
    fi
}

# T9: --gpu-target all dispatch wired — structural check
{
    TOTAL=$((TOTAL + 1))
    if grep -q 'str_eq(gpu_target, "all")' self-hosted/compiler/main.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T9_gpu_target_all_wired"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T9_gpu_target_all_wired: --gpu-target all dispatch not found in main.sio"
    fi
}

# T9b: DGX Spark Blackwell PTX selection and shuffle lowering
{
    TOTAL=$((TOTAL + 1))
    if grep -q 'str_eq(gpu_target, "dgx-sm121")' self-hosted/compiler/main.sio && \
       grep -q 'hlir_kernels_to_dgx_sm121_ptx' self-hosted/gpu/hlir_to_gpu.sio && \
       grep -q '\.target sm_121' self-hosted/gpu/ptx.sio && \
       grep -q 'shfl\.sync\.bfly\.b32' self-hosted/gpu/lower_to_ptx.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T9b_dgx_sm121_ptx_wired"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T9b_dgx_sm121_ptx_wired"
    fi
}

# T9c: execute the Sounio-native DGX header + butterfly emitter witness
{
    result="$(run_souc_tail self-hosted/gpu/pireus_dgx_sm121_codegen.sio)"
    if [ "$result" = "PIREUS_DGX_SM121_CODEGEN_PASS" ]; then
        pass "T9c_dgx_sm121_codegen_witness"
    else
        fail "T9c_dgx_sm121_codegen_witness" "$result"
    fi
}

# T9d/T9e: explicit Pireus sedenion XOR-convolution ABI and 16-lane f64 lowering
check_souc \
    "T9d_pireus_xor_convolution_source" \
    "tests/gpu/pireus_sed_xor_convolution_f64.sio"

{
    result="$(run_souc_tail self-hosted/gpu/pireus_xor_convolution_f64_codegen.sio)"
    if [ "$result" = "PIREUS_XOR_CONV_F64_CODEGEN_PASS" ]; then
        pass "T9e_pireus_xor_convolution_codegen"
    else
        fail "T9e_pireus_xor_convolution_codegen" "$result"
    fi
}

# T10: Epistemic SPIR-V wired — structural check
{
    TOTAL=$((TOTAL + 1))
    if grep -q 'hlir_kernels_to_epistemic_spirv' self-hosted/gpu/hlir_to_gpu.sio && \
       [ -f self-hosted/gpu/epistemic_spirv.sio ] && \
       grep -q 'gpu_lower_to_epistemic_spirv' self-hosted/gpu/epistemic_spirv.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T10_epistemic_spirv_wired"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T10_epistemic_spirv_wired: epistemic SPIR-V module or wiring missing"
    fi
}

# T11: Metal launch orchestration wired — structural check
{
    TOTAL=$((TOTAL + 1))
    if grep -q 'gpu_launch_msl_vecadd' self-hosted/gpu/runtime/launch.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T11_metal_launch_wired"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T11_metal_launch_wired: gpu_launch_msl_vecadd missing from runtime/launch.sio"
    fi
}

# T12: Host glue codegen wired — structural check
{
    TOTAL=$((TOTAL + 1))
    if grep -q 'gpu_emit_host_glue' self-hosted/compiler/main.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T12_host_glue_codegen"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T12_host_glue_codegen: gpu_emit_host_glue missing from compiler/main.sio"
    fi
}

# T13: Jacobian-Guided GUM (epistemic_autodiff) — structural + self-test
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/epistemic_autodiff.sio ] && \
       grep -q 'jad_gum_combine' self-hosted/gpu/epistemic_autodiff.sio && \
       grep -q 'jad_propagate_mul' self-hosted/gpu/epistemic_autodiff.sio; then
        # Run self-tests
        RESULT=$(run_souc_tail self-hosted/gpu/epistemic_autodiff.sio)
        if echo "$RESULT" | grep -q "12/12 tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T13_jacobian_gum (12/12 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T13_jacobian_gum: self-tests did not all pass: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T13_jacobian_gum: epistemic_autodiff.sio missing or incomplete"
    fi
}

# T14: Built-in kernel AD — kernel_autodiff structural + self-test
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/kernel_autodiff.sio ] && \
       grep -q 'kad_generate_backward' self-hosted/gpu/kernel_autodiff.sio && \
       grep -q 'kad_tape_backward' self-hosted/gpu/kernel_autodiff.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/kernel_autodiff.sio)
        if echo "$RESULT" | grep -q "10/10 tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T14_kernel_autodiff (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T14_kernel_autodiff: self-tests did not all pass: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T14_kernel_autodiff: kernel_autodiff.sio missing or incomplete"
    fi
}

# T15: CUDA Tile IR backend — structural + self-test
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/cuda_tile.sio ] && \
       grep -q 'gpu_lower_to_cuda_tile' self-hosted/gpu/cuda_tile.sio && \
       grep -q 'gpu_lower_to_cuda_tile' self-hosted/compiler/main.sio && \
       grep -q 'hlir_kernels_to_cuda_tile' self-hosted/gpu/hlir_to_gpu.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/cuda_tile.sio)
        if echo "$RESULT" | grep -q "ALL 10 PASS"; then
            PASS=$((PASS + 1))
            echo "PASS  T15_cuda_tile_ir (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T15_cuda_tile_ir: self-tests did not all pass: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T15_cuda_tile_ir: cuda_tile.sio or pipeline wiring missing"
    fi
}

# ── Section 2b: Boundary-Breaking GPU Novelties (Sprints 248-250) ─────────────────

echo ""
echo "--- Section 2b: Boundary-Breaking Novelties (T16-T18) ---"

# T16: Epistemic Kernel Fusion (Novelty 4) — provenance+uncertainty-guided DAG fusion
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/opt/epistemic_fusion.sio ] && \
       grep -q 'epfuse_build_graph' self-hosted/gpu/opt/epistemic_fusion.sio && \
       grep -q 'epfuse_score_edge' self-hosted/gpu/opt/epistemic_fusion.sio && \
       grep -q 'hlir_kernels_to_epistemic_fused_ptx' self-hosted/gpu/hlir_to_gpu.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/opt/epistemic_fusion.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T16_epistemic_fusion (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T16_epistemic_fusion: self-tests did not all pass: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T16_epistemic_fusion: epistemic_fusion.sio or pipeline wiring missing"
    fi
}

# T17: Speculative Epistemic Execution (Novelty 5) — uncertainty-threshold skip guards
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/speculative.sio ] && \
       grep -q 'spec_classify_kernels' self-hosted/gpu/speculative.sio && \
       grep -q 'spec_emit_branch_guard' self-hosted/gpu/speculative.sio && \
       grep -q 'hlir_kernels_to_speculative_ptx' self-hosted/gpu/hlir_to_gpu.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/speculative.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T17_speculative_execution (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T17_speculative_execution: self-tests did not all pass: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T17_speculative_execution: speculative.sio or pipeline wiring missing"
    fi
}

# T18: Provenance-Aware DAG Scheduler (Novelty 6) — information-flow parallelism
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/dag_scheduler.sio ] && \
       grep -q 'dag_build' self-hosted/gpu/dag_scheduler.sio && \
       grep -q 'dag_topo_sort' self-hosted/gpu/dag_scheduler.sio && \
       grep -q 'dag_assign_streams' self-hosted/gpu/dag_scheduler.sio && \
       grep -q 'hlir_kernels_to_dag_scheduled_ptx' self-hosted/gpu/hlir_to_gpu.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/dag_scheduler.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T18_dag_scheduler (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T18_dag_scheduler: self-tests did not all pass: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T18_dag_scheduler: dag_scheduler.sio or pipeline wiring missing"
    fi
}

echo ""
echo "--- Section 2c: Deep Novelties (T19-T21) ---"

# T19: Warp-Vote Epistemic Fast-Path (Novelty 7) — vote.sync.ballot dual-path PTX
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/opt/warp_vote_fastpath.sio ] && \
       grep -q 'wv_emit_ballot_check' self-hosted/gpu/opt/warp_vote_fastpath.sio && \
       grep -q 'wv_emit_epsilon_check' self-hosted/gpu/opt/warp_vote_fastpath.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/opt/warp_vote_fastpath.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T19_warp_vote_fastpath (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T19_warp_vote_fastpath: self-tests did not all pass: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T19_warp_vote_fastpath: warp_vote_fastpath.sio missing"
    fi
}

# T20: Entropy-Gated Kernel Dispatch (Novelty 8) — H(epsilon) variant selection
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/opt/entropy_dispatch.sio ] && \
       grep -q 'ed_shannon_entropy' self-hosted/gpu/opt/entropy_dispatch.sio && \
       grep -q 'ed_decide_variant' self-hosted/gpu/opt/entropy_dispatch.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/opt/entropy_dispatch.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T20_entropy_dispatch (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T20_entropy_dispatch: self-tests did not all pass: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T20_entropy_dispatch: entropy_dispatch.sio missing"
    fi
}

# T21: Production PTX metadata parser (structural check)
{
    TOTAL=$((TOTAL + 1))
    if grep -q 'ptx_meta_dag_stream_count' self-hosted/gpu/hlir_to_gpu.sio && \
       grep -q 'ptx_meta_stream_assignments' self-hosted/gpu/hlir_to_gpu.sio && \
       grep -q 'ptx_meta_dependencies' self-hosted/gpu/hlir_to_gpu.sio && \
       grep -q 'ptx_meta_spec_plan' self-hosted/gpu/hlir_to_gpu.sio && \
       grep -q 'ptx_meta_spec_guards' self-hosted/gpu/hlir_to_gpu.sio; then
        PASS=$((PASS + 1))
        echo "PASS  T21_ptx_metadata_parser (5/5 parser functions present)"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T21_ptx_metadata_parser: production metadata parser functions missing"
    fi
}

# T22: Epistemic GPU Showcase — practical utility demonstration
{
    TOTAL=$((TOTAL + 1))
    if [ -f examples/gpu_epistemic_showcase.sio ]; then
        RESULT=$(run_souc_tail examples/gpu_epistemic_showcase.sio)
        if echo "$RESULT" | grep -q "10/10 tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T22_epistemic_showcase (10/10 tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T22_epistemic_showcase: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T22_epistemic_showcase: showcase file missing"
    fi
}

echo ""
echo "--- Section 2d: Limitation Resolution (T23-T25) ---"

# T23: Second-Order GUM Shadow Lanes (Limitation 1 — Hessian correction)
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/opt/second_order_gum.sio ] && \
       grep -q 'so_compute_hessian_correction' self-hosted/gpu/opt/second_order_gum.sio && \
       grep -q 'so_compute_first_order' self-hosted/gpu/opt/second_order_gum.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/opt/second_order_gum.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T23_second_order_gum (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T23_second_order_gum: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T23_second_order_gum: second_order_gum.sio missing"
    fi
}

# T24: Covariance Matrix GPU Shadow Lanes (Limitation 2 — correlation support)
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/opt/covariance_shadow.sio ] && \
       grep -q 'cs_propagate_nd' self-hosted/gpu/opt/covariance_shadow.sio && \
       grep -q 'cs_tri_index' self-hosted/gpu/opt/covariance_shadow.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/opt/covariance_shadow.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T24_covariance_shadow (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T24_covariance_shadow: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T24_covariance_shadow: covariance_shadow.sio missing"
    fi
}

# T25: Warp Divergence Cost Model (Limitation 3 — quantified penalty)
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/opt/divergence_cost.sio ] && \
       grep -q 'dc_compute_penalty' self-hosted/gpu/opt/divergence_cost.sio && \
       grep -q 'dc_estimate_speedup' self-hosted/gpu/opt/divergence_cost.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/opt/divergence_cost.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T25_divergence_cost (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T25_divergence_cost: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T25_divergence_cost: divergence_cost.sio missing"
    fi
}

# T26: Tiled Covariance (CUTLASS-style blocking, N=128)
{
    TOTAL=$((TOTAL + 1))
    if [ -f self-hosted/gpu/opt/tiled_covariance.sio ] && \
       grep -q 'tc_propagate_tile' self-hosted/gpu/opt/tiled_covariance.sio && \
       grep -q 'tc_build_plan' self-hosted/gpu/opt/tiled_covariance.sio; then
        RESULT=$(run_souc_tail self-hosted/gpu/opt/tiled_covariance.sio)
        if echo "$RESULT" | grep -q "10/10 self-tests passed"; then
            PASS=$((PASS + 1))
            echo "PASS  T26_tiled_covariance (10/10 self-tests)"
        else
            FAIL=$((FAIL + 1))
            echo "FAIL  T26_tiled_covariance: $RESULT"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  T26_tiled_covariance: tiled_covariance.sio missing"
    fi
}

# ── Section 3: souc check — optional files ────────────────────────────────────

echo ""
echo "--- Section 3: souc check (optional) ---"

# T7: skip if absent
check_souc_or_skip \
    "T7_kernel_matmul" \
    "examples/kernel_matmul.sio"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Results: PASS=$PASS FAIL=$FAIL SKIP=$SKIP TOTAL=$TOTAL ==="

if [ "$FAIL" -gt 0 ]; then
    exit "$FAIL"
fi
exit 0
