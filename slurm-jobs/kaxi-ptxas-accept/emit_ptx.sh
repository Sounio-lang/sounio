#!/usr/bin/env bash
# Emit every K-AXI→PTX (pattern × mode) combo locally into <out>/ for staging.
# Mirrors the pattern/mode matrix of scripts/ci/kaxi_ptx_golden_gate.sh.
# Requires bin/kretikos (and a souc seed it can resolve, or SOUNIO_KRETIKOS_COMPILER).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="${1:-/tmp/kaxi_ptx_stage/ptx}"
mkdir -p "$OUT_DIR"

PATTERNS=(
    exit_only
    vec_add vec_sub vec_mul vec_div vec_min vec_max vec_abs vec_neg vec_sqrt vec_recip vec_lg2 vec_ex2 vec_sin vec_cos vec_rsqrt vec_conf_sqrt vec_sqrt_gate_var vec_sqrt_gate_var_mb fma
    source_vec_add_f64 source_vec_sub_f64 source_vec_mul_f64 source_vec_div_f64 source_fma_f64
    source_vec_add_f32 source_vec_sub_f32 source_vec_mul_f32 source_vec_div_f32 source_fma_f32
    source_epistemic_dual_output_f32 epistemic_elementwise_f32 epistemic_dual_output_f32
    mb_vec_double_f32
    pbpk_euler pbpk_4step pbpk_8step pbpk_2comp pbpk_2comp_4step
    atomic_sum_f32
    pbpk_2comp_mb pbpk_2comp_4step_mb
    x4_tree x4_chain
    pbpk_rapamycin_1comp_epistemic
    pbpk_2comp_epistemic pbpk_2comp_epistemic_4step pbpk_traj_4step
    conf_gate
    octonion_assoc octonion_mul sedenion_mul
    pbpk_2comp_gum_4step
)
MODES=(
    "default:"
    "epistemic:--epistemic"
    "f32:--f32"
    "f32_2c:--f32-2c"
    "f32e:--f32-epistemic"
    "f32_gum:--f32-gum"
)

n=0; err=0
for mode_entry in "${MODES[@]}"; do
    mode="${mode_entry%%:*}"; flag="${mode_entry#*:}"
    for pattern in "${PATTERNS[@]}"; do
        out="$OUT_DIR/${mode}__${pattern}.ptx"
        if [[ -n "$flag" ]]; then
            ./bin/kretikos kaxi-emit-ptx "$pattern" -o "$out" --no-ptxas "$flag" >/dev/null 2>&1
        else
            ./bin/kretikos kaxi-emit-ptx "$pattern" -o "$out" --no-ptxas >/dev/null 2>&1
        fi
        if [[ $? -eq 0 && -s "$out" ]]; then n=$((n+1)); else err=$((err+1)); echo "emit FAIL: $mode/$pattern" >&2; fi
    done
done
echo "emitted=$n emit_errs=$err -> $OUT_DIR"
[[ $err -eq 0 ]] || exit 1
