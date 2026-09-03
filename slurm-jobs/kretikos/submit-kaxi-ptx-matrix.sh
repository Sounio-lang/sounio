#!/usr/bin/env bash
# Submit a single Slurm job that exercises ALL K-AXI → PTX patterns × modes
# on a real GPU and verifies output against the expected GUM-propagated
# reference values.
#
# Cases (12 total):
#   source_vec_add_f32                   basic         mem=2*init
#   source_vec_add_f32                   epistemic     mem,var both verified
#   source_vec_sub_f32                   basic
#   source_vec_sub_f32                   epistemic
#   source_vec_mul_f32                   basic         mem=init²
#   source_vec_mul_f32                   epistemic     var=2·init² (independent path)
#   source_vec_div_f32                   basic
#   source_vec_div_f32                   epistemic
#   source_fma_f32                       basic         mem=init·(init+1)
#   source_fma_f32                       epistemic     var=2·init²+1
#   source_epistemic_dual_output_f32     basic         mem=8·init²
#   source_epistemic_dual_output_f32     epistemic     mem=8·init², var=64·init² (X² delta-method)
#
# Reports "kaxi_matrix passed=N/TOTAL failed=..." via Slurm Comment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS="${NS:-slurm-pilot}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
JOB_MEM="${JOB_MEM:-4G}"
JOB_TIME="${JOB_TIME:-00:10:00}"
WAIT_FOR_RESULT="${WAIT_FOR_RESULT:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-600}"

if [[ ! -x "${ROOT_DIR}/bin/kretikos" ]]; then echo "missing kretikos" >&2; exit 1; fi
if ! command -v cc >/dev/null 2>&1; then echo "local cc required" >&2; exit 1; fi

RUN_ID="${RUN_ID:-kaxi-matrix-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_DIR="/tmp/${RUN_ID}.stage"
LOCAL_TARBALL="/tmp/${RUN_ID}.tgz"
LOCAL_SBATCH="/tmp/${RUN_ID}.sbatch"
cleanup() { rm -rf "${STAGE_DIR}" "${LOCAL_TARBALL}" "${LOCAL_SBATCH}"; }
trap cleanup EXIT

mkdir -p "${STAGE_DIR}"

# ----- declare cases as parallel arrays ---------------------------------
# (name|mode|blocks|threads|mem-words|init-mem|init-var|expected-mem|expected-var)
# init-mem of "_seq_" means seed mem with [1..mem-words] sequence (test multi-block).
# mode: basic | epistemic | f32 | f64
#   f32 → uses --f32 lowering, --type f32 runner
#   f64 → uses typed K-AXI metadata with the basic lowering, --type f64 runner
CASES=(
  "source_vec_add_f64|f64|1|8|8|1.25,2.5,3.75,4.5,5.25,6.5,7.75,8.125||2.5,5,7.5,9,10.5,13,15.5,16.25|"
  "source_vec_sub_f64|f64|1|8|8|1.25,2.5,3.75,4.5,5.25,6.5,7.75,8.125||0,0,0,0,0,0,0,0|"
  "source_vec_mul_f64|f64|1|8|8|1.25,2.5,3.75,4.5,5.25,6.5,7.75,8.125||1.5625,6.25,14.0625,20.25,27.5625,42.25,60.0625,66.015625|"
  "source_vec_div_f64|f64|1|8|8|1.25,2.5,3.75,4.5,5.25,6.5,7.75,8.125||1,1,1,1,1,1,1,1|"
  "source_fma_f64|f64|1|8|8|1.25,2.5,3.75,4.5,5.25,6.5,7.75,8.125||2.8125,8.75,17.8125,24.75,32.8125,48.75,67.8125,74.140625|"
  "source_vec_add_f32|typed_f32|1|8|8|1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5||3,5,7,9,11,13,15,17|"
  "source_vec_mul_f32|typed_f32|1|8|8|1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5||2.25,6.25,12.25,20.25,30.25,42.25,56.25,72.25|"
  "source_vec_add_i32|typed_i32|1|8|8|1,2,3,4,5,6,7,8||2,4,6,8,10,12,14,16|"
  "source_vec_mul_i32|typed_i32|1|8|8|1,2,3,4,5,6,7,8||1,4,9,16,25,36,49,64|"
  "vec_add|typed_f32|1|8|8|1,2,3,4,5,6,7,8||2,4,6,8,10,12,14,16|"
  "vec_add|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|2,4,6,8,10,12,14,16|2,2,2,2,2,2,2,2"
  "vec_sub|typed_f32|1|8|8|1,2,3,4,5,6,7,8||0,0,0,0,0,0,0,0|"
  "vec_sub|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|0,0,0,0,0,0,0,0|2,2,2,2,2,2,2,2"
  "vec_mul|typed_f32|1|8|8|1,2,3,4,5,6,7,8||1,4,9,16,25,36,49,64|"
  "vec_mul|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|1,4,9,16,25,36,49,64|2,8,18,32,50,72,98,128"
  "vec_div|typed_f32|1|8|8|1,2,3,4,5,6,7,8||1,1,1,1,1,1,1,1|"
  "vec_div|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|1,1,1,1,1,1,1,1|2,2,2,2,2,2,2,2"
  "fma|typed_f32|1|8|8|1,2,3,4,5,6,7,8||2,6,12,20,30,42,56,72|"
  "fma|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|2,6,12,20,30,42,56,72|3,9,19,33,51,73,99,129"
  "edo|basic|1|8|8|1,2,3,4,5,6,7,8||8,32,72,128,200,288,392,512|"
  "edo|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|8,32,72,128,200,288,392,512|64,256,576,1024,1600,2304,3136,4096"
  "vec_add|f32|1|8|8|1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5||3,5,7,9,11,13,15,17|"
  "vec_mul|f32|1|8|8|1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5||2.25,6.25,12.25,20.25,30.25,42.25,56.25,72.25|"
  "fma|f32|1|8|8|2,3,4,5,6,7,8,9||6,12,20,30,42,56,72,90|"
  "pbpk|basic|1|8|8|1,2,3,4,5,6,7,8||9,8,7,6,5,4,3,2|"
  "pbpk|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|9,8,7,6,5,4,3,2|5,5,5,5,5,5,5,5"
  "pbpk|f32e|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|9,8,7,6,5,4,3,2|5,5,5,5,5,5,5,5"
  "pbpk4|f32|1|8|8|1,2,3,4,5,6,7,8||4.75,4.8125,4.875,4.9375,5,5.0625,5.125,5.1875|"
  "pbpk4|f32e|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|4.75,4.8125,4.875,4.9375,5,5.0625,5.125,5.1875|2.44141,2.44141,2.44141,2.44141,2.44141,2.44141,2.44141,2.44141"
  "pbpk8|f32|1|8|8|1,2,3,4,5,6,7,8||4.98438,4.98828,4.99219,4.99609,5,5.00391,5.00781,5.01172|"
  "pbpk2c|f32_2c|1|8|8|1,2,3,4,5,6,7,8|0,0,0,0,0,0,0,0|1.25,2,2.75,3.5,4.25,5,5.75,6.5|0.25,0.5,0.75,1,1.25,1.5,1.75,2"
  "pbpk2c4|f32_2c|1|8|8|1,2,3,4,5,6,7,8|0,0,0,0,0,0,0,0|1.85498,2.29004,2.7251,3.16016,3.59521,4.03027,4.46533,4.90039|1.14502,1.70996,2.2749,2.83984,3.40479,3.96973,4.53467,5.09961"
  "atomic|f32|1|8|8|0,0,0,0,0,0,0,0||28,0,0,0,0,0,0,0|"
  "atomic|f32|1|1024|1|0||523776|"
  "x4_tree|f32e|1|8|8|2,3,4,5,6,7,8,9|1,1,1,1,1,1,1,1|16,81,256,625,1296,2401,4096,6561|1024,11664,65536,250000,746496,1.88238e+06,4.1943e+06,8.50306e+06"
  "x4_chain|f32e|1|8|8|2,3,4,5,6,7,8,9|1,1,1,1,1,1,1,1|16,81,256,625,1296,2401,4096,6561|384,4374,24576,93750,279936,705894,1.57286e+06,3.18865e+06"
  "pbpk2c_e|f32e|1|8|16|1,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0|1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0|1.25,2,2.75,3.5,4.25,5,5.75,6.5,0.25,0.5,0.75,1,1.25,1.5,1.75,2|1.0625,1.0625,1.0625,1.0625,1.0625,1.0625,1.0625,1.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625"
  "pbpk2c_e4|f32e|1|8|16|1,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0|1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0|1.85498,2.29004,2.7251,3.16016,3.59521,4.03027,4.46533,4.90039,1.14502,1.70996,2.2749,2.83984,3.40479,3.96973,4.53467,5.09961|1.28085,1.28085,1.28085,1.28085,1.28085,1.28085,1.28085,1.28085,0.280853,0.280853,0.280853,0.280853,0.280853,0.280853,0.280853,0.280853"
  "pbpk_traj|f32|1|8|40|1,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0||1,2,3,4,5,6,7,8,3,3.5,4,4.5,5,5.5,6,6.5,4,4.25,4.5,4.75,5,5.25,5.5,5.75,4.5,4.625,4.75,4.875,5,5.125,5.25,5.375,4.75,4.8125,4.875,4.9375,5,5.0625,5.125,5.1875|"
  "conf_gate|f32|1|8|16|1,1.2,1.4,1.5,1.6,1.8,2,5,0,0,0,0,0,0,0,0||1,1.2,1.4,1.5,1.6,1.8,2,5,1,1,1,1,0,0,0,0|"
  "octonion|f32|1|2|6|1,1,1,-1,0,0||1,1,1,-1,1,-1|"
  "octonion_mul|f32|1|2|48|1,2,3,4,5,6,7,8,9,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0||1,2,3,4,5,6,7,8,9,1,0,0,0,0,0,0,7,19,31,33,51,49,55,79,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0|"
  "octonion_associator|f32|1|2|64|0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0||0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2|"
  "mb_double|basic|4|8|32|_seq_||2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64|"
  "mb_double|basic|8|8|64|_seq_||2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128|"
)

# Map short names → kretikos pattern names
declare -A PATTERN_FOR
PATTERN_FOR[vec_add]=source_vec_add_f32
PATTERN_FOR[source_vec_add_f64]=source_vec_add_f64
PATTERN_FOR[source_vec_sub_f64]=source_vec_sub_f64
PATTERN_FOR[source_vec_mul_f64]=source_vec_mul_f64
PATTERN_FOR[source_vec_div_f64]=source_vec_div_f64
PATTERN_FOR[source_fma_f64]=source_fma_f64
PATTERN_FOR[vec_sub]=source_vec_sub_f32
PATTERN_FOR[vec_mul]=source_vec_mul_f32
PATTERN_FOR[vec_div]=source_vec_div_f32
PATTERN_FOR[fma]=source_fma_f32
PATTERN_FOR[source_vec_add_f32]=source_vec_add_f32
PATTERN_FOR[source_vec_mul_f32]=source_vec_mul_f32
PATTERN_FOR[source_vec_add_i32]=source_vec_add_i32
PATTERN_FOR[source_vec_mul_i32]=source_vec_mul_i32
PATTERN_FOR[edo]=source_epistemic_dual_output_f32
PATTERN_FOR[mb_double]=mb_vec_double_f32
PATTERN_FOR[pbpk]=pbpk_euler
PATTERN_FOR[pbpk4]=pbpk_4step
PATTERN_FOR[pbpk8]=pbpk_8step
PATTERN_FOR[pbpk2c]=pbpk_2comp
PATTERN_FOR[pbpk2c4]=pbpk_2comp_4step
PATTERN_FOR[atomic]=atomic_sum_f32
PATTERN_FOR[x4_tree]=x4_tree
PATTERN_FOR[x4_chain]=x4_chain
PATTERN_FOR[pbpk2c_e]=pbpk_2comp_epistemic
PATTERN_FOR[pbpk2c_e4]=pbpk_2comp_epistemic_4step
PATTERN_FOR[pbpk_traj]=pbpk_traj_4step
PATTERN_FOR[conf_gate]=conf_gate
PATTERN_FOR[octonion]=octonion_assoc
PATTERN_FOR[octonion_mul]=octonion_mul
# octonion_associator: thread 0 verifies Fano triple [e1,e2,e3]=0; thread 1
# verifies non-Fano triple [e1,e2,e4]=2·e7 (CPU-derived from oct_mul).
PATTERN_FOR[octonion_associator]=octonion_associator
PATTERN_FOR[vec_min]=vec_min
PATTERN_FOR[vec_max]=vec_max
PATTERN_FOR[vec_abs]=vec_abs
PATTERN_FOR[vec_neg]=vec_neg
PATTERN_FOR[vec_sqrt]=vec_sqrt
PATTERN_FOR[vec_recip]=vec_recip
PATTERN_FOR[vec_lg2]=vec_lg2
PATTERN_FOR[vec_ex2]=vec_ex2

# Phase P: log₂ / 2^x — PTX-native. Bit-exact value lane on power-of-2
# inputs (lg2(2^k)=k; ex2(k)=2^k both exact f32). init_var=0 keeps
# variance lane at 0 even though it scales by ln(2) — bit-exact without
# modelling the irrational scaling.
CASES+=(
  "vec_lg2|f32|1|8|8|1,2,4,8,16,32,64,128||0,1,2,3,4,5,6,7|"
  "vec_ex2|f32|1|8|8|0,1,2,3,4,5,6,7||1,2,4,8,16,32,64,128|"
)

# Phase O: 1/x with variance σ²/x⁴. Power-of-2 inputs → exact f32
# values for both reciprocal and x⁴ → bit-exact comparison.
# init_var = 1; expected var_out = 1 / x⁴ → exact f32.
CASES+=(
  "vec_recip|f32|1|8|8|1,2,4,8,16,32,64,128||1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125|"
  "vec_recip|f32e|1|8|8|1,2,4,8,16,32,64,128|1,1,1,1,1,1,1,1|1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125|1,0.0625,0.00390625,0.000244141,1.52588e-05,9.53674e-07,5.96046e-08,3.72529e-09"
)

# Phase N: sqrt with non-trivial variance lane. σ²(√x) = σ²/(4x).
# All inputs chosen as exact f32 perfect squares (powers of 2 squared)
# so sqrt.approx.f32 produces bit-exact results vs the analytic reference.
# In f32e mode init_var=4 → expected_var = 4/(4·x) = 1/x, with each x
# also a power of 2 → exact f32 representation throughout.
CASES+=(
  "vec_sqrt|f32|1|8|8|1,4,9,16,25,36,49,64||1,2,3,4,5,6,7,8|"
  "vec_sqrt|f32e|1|8|8|1,4,16,64,256,1024,4096,16384|4,4,4,4,4,4,4,4|1,2,4,8,16,32,64,128|1,0.25,0.0625,0.015625,0.00390625,0.000976562,0.000244141,6.10352e-05"
)

# Phase M: unary opcodes abs / neg. Exercise actual sign-flip work
# (negative inputs for abs, positive for neg). Variance lane is COPIED
# from src by kaxi_lower_unary, so var_out == var_in.
CASES+=(
  "vec_abs|basic|1|8|8|-1,-2,-3,-4,-5,-6,-7,-8||1,2,3,4,5,6,7,8|"
  "vec_abs|epistemic|1|8|8|-1,-2,-3,-4,-5,-6,-7,-8|1,1,1,1,1,1,1,1|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1"
  "vec_abs|f32|1|8|8|-1.5,-2.5,-3.5,-4.5,-5.5,-6.5,-7.5,-8.5||1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5|"
  "vec_neg|basic|1|8|8|1,2,3,4,5,6,7,8||-1,-2,-3,-4,-5,-6,-7,-8|"
  "vec_neg|epistemic|1|8|8|1,2,3,4,5,6,7,8|2,2,2,2,2,2,2,2|-1,-2,-3,-4,-5,-6,-7,-8|2,2,2,2,2,2,2,2"
  "vec_neg|f32|1|8|8|1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5||-1.5,-2.5,-3.5,-4.5,-5.5,-6.5,-7.5,-8.5|"
)

# Phase F: validate Phase E's new vec_min / vec_max opcodes on real L4.
# Pattern computes y[tid] = op(x[tid], x[tid]) → output = input (identity);
# epistemic variance lane uses additive propagation → var_out = 2·var_in.
CASES+=(
  "vec_max|basic|1|8|8|1,2,3,4,5,6,7,8||1,2,3,4,5,6,7,8|"
  "vec_max|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|1,2,3,4,5,6,7,8|2,2,2,2,2,2,2,2"
  "vec_max|f32|1|8|8|1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5||1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5|"
  "vec_min|basic|1|8|8|1,2,3,4,5,6,7,8||1,2,3,4,5,6,7,8|"
  "vec_min|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|1,2,3,4,5,6,7,8|2,2,2,2,2,2,2,2"
  "vec_min|f32|1|8|8|1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5||1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5|"
)

# Sedenion zero-divisor: (e₁+e₁₀)·(e₅+e₁₄) = 0 on thread 0,
# (e₀)·(e₀) = e₀ on thread 1. Witnesses non-trivial ZD in 𝕊.
SEDZD_INIT=$(LC_ALL=C awk 'BEGIN{
  split("1 10 21 30 48 64", a, " "); for(k in a) ones[a[k]]=1;
  for(i=0;i<96;i++) printf "%s%d", (i?",":""), (i in ones ? 1 : 0);
}')
SEDZD_EXP=$(LC_ALL=C awk 'BEGIN{
  split("1 10 21 30 48 64 80", a, " "); for(k in a) ones[a[k]]=1;
  for(i=0;i<96;i++) printf "%s%d", (i?",":""), (i in ones ? 1 : 0);
}')
CASES+=("sedenion_mul|f32|1|2|96|${SEDZD_INIT}||${SEDZD_EXP}|")

# Phase I: re-segmented variance register file (value 0..127, variance
# 128..255, scratch %rd256/%rd257, base %rd258/%rd259) unblocks the
# big-kernel variance modes that Phase H surfaced as broken.
SEDZD_VAR_ZERO=$(LC_ALL=C awk 'BEGIN{ for(i=0;i<96;i++) printf "%s0", (i?",":"") }')
CASES+=("sedenion_mul|epistemic|1|2|96|${SEDZD_INIT}|${SEDZD_VAR_ZERO}|${SEDZD_EXP}|${SEDZD_VAR_ZERO}")
CASES+=("sedenion_mul|f32e|1|2|96|${SEDZD_INIT}|${SEDZD_VAR_ZERO}|${SEDZD_EXP}|${SEDZD_VAR_ZERO}")

# Phase J: octonion_mul in variance modes — gives the octonion-connectomics
# research line a GUM-propagating GPU primitive. Inputs are 8-vector basis
# elements; outputs are the 8-component product. With init_var=0, var stays
# 0 through every mul (zero · anything = zero). Same mem expected as f32.
OCT_INIT="1,2,3,4,5,6,7,8,9,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0"
OCT_EXP="1,2,3,4,5,6,7,8,9,1,0,0,0,0,0,0,7,19,31,33,51,49,55,79,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0"
OCT_VAR_ZERO=$(LC_ALL=C awk 'BEGIN{ for(i=0;i<48;i++) printf "%s0", (i?",":"") }')
CASES+=("octonion_mul|epistemic|1|2|48|${OCT_INIT}|${OCT_VAR_ZERO}|${OCT_EXP}|${OCT_VAR_ZERO}")
CASES+=("octonion_mul|f32e|1|2|48|${OCT_INIT}|${OCT_VAR_ZERO}|${OCT_EXP}|${OCT_VAR_ZERO}")
PATTERN_FOR[sedenion_mul]=sedenion_mul

# Sedenion associator: 𝕊 [a,b,c]=(a·b)·c−a·(b·c). Two threads:
# T0=[e1,e2,e3] (octonion-embedded Fano)→0; T1=[e1,e2,e10] (CD-mixed)→+2·e9.
# CPU oracle path verified via stdlib::algebra::sedenion::sed_associator.
CASES+=("sedenion_associator|f32|1|2|192|0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0||0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0|")
PATTERN_FOR[sedenion_associator]=sedenion_associator

# sedenion_sqr: per-thread z² in 𝕊 with X² GUM variance rule.
# Layout: 32 words/thread (16 z input + 16 z² output).
# T0: z=e₀ (identity) → z²=e₀. T1: z=e₁ → z²=−e₀.
# These two threads exercise component positions 0 and 1 of the input array,
# covering the diagonal variance rule (z[i]·z[i] same-reg X² path).
SEDSQR_INIT="1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
SEDSQR_EXP="1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
CASES+=("sedenion_sqr|f32|1|2|64|${SEDSQR_INIT}||${SEDSQR_EXP}|")
PATTERN_FOR[sedenion_sqr]=sedenion_sqr

# sedenion_sqr_mb: same fixture with blocks=2, threads_per_block=1 so each
# thread is on a different block — exercises gid = bid*ntid+tid addressing.
# bit-exact vs sedenion_sqr (same input/output arrays, different block layout).
CASES+=("sedenion_sqr_mb|f32|2|1|64|${SEDSQR_INIT}||${SEDSQR_EXP}|")
PATTERN_FOR[sedenion_sqr_mb]=sedenion_sqr_mb

# Connectome N=8 demo: 56 triples (i<j<k from 0..7), each thread runs
# the existing octonion_associator kernel on a basis-vector triple. Tests the
# Fano partition: 21 e_0-triples (identity) + 7 Fano basis triples + 28 non-Fano
# triples (assoc=±2·e_l). Drives Phase L kernel at multi-thread connectomics scale.
CASES+=("connectome_n8_demo|f32|1|56|1792|1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0||1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,-2,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-2,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,-2,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,-2,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-2,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,-2,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,-2,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,2,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,-2,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,-2,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,-2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,-2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,-2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,-2,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,-2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0,0,0|")
PATTERN_FOR[connectome_n8_demo]=octonion_associator

# C.0+1+2+3 closure: 2-thread multi-block test. Validates the full new
# assoc_norm_sum codepath in a fixture small enough to hand-check:
#   - get_nctaid (returns 2)
#   - gid = bid·ntid + tid (here ntid=1 so gid=bid)
#   - runtime atomic-dest = nctaid·ntid·24 = 48 (was hardcoded imm=1344)
#   - per-thread store at mem[48 + 1 + gid]
# gid 0 = [e₁,e₂,e₃] (Fano, ‖[·]‖²=0); gid 1 = [e₁,e₂,e₄] (non-Fano, ‖·‖²=4).
# Expected: inputs preserved + mem[48]=4 (atomic), mem[49]=0, mem[50]=4.
CASES+=("assoc_norm_sum_mb2|f32|2|1|51|0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0||0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,4,0,4|")
PATTERN_FOR[assoc_norm_sum_mb2]=octonion_assoc_norm_sum

# Connectome N=8 norm-sum: same 56 triples but kernel computes ‖[·]‖² inline
# and atomic-adds to mem[1344]. Expected sum = 28 non-Fano triples × ‖±2·e_l‖² = 112.
# TODO(C.0+1+2+3 follow-up): the abbdb512 kernel rewrite added a per-thread
# slab at mem[total_threads*24 + 1 + gid], so mem-words=1345 now OOB-writes.
# Restore once the 56-triple enumeration order is extracted from the emitter
# and the expected per-thread slab (56 floats: 0 or 4 in Fano-derived order)
# is appended. assoc_norm_sum_mb2 above covers the new codepaths in the
# meantime.
: <<'OOB_TODO'
CASES+=("connectome_n8_norm_sum|f32|1|56|1345|1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0||1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,112|")
PATTERN_FOR[connectome_n8_norm_sum]=octonion_assoc_norm_sum
OOB_TODO




# Programmatically generate large multi-block cases (256, 1024 elements).
# Each writes mem[i] = 2*(i+1).
for N in 256 1024; do
  # Pick blocks/threads to multiply to N (max 1024 threads/block on sm89)
  if [[ "$N" -le 256 ]]; then BLOCKS=8; THREADS=$((N / 8)); else BLOCKS=32; THREADS=$((N / 32)); fi
  EMEM=$(seq 1 "$N" | awk '{ printf "%s%d", (NR>1?",":""), $1*2 }')
  CASES+=("mb_double_${N}|basic|${BLOCKS}|${THREADS}|${N}|_seq_||${EMEM}|")
  PATTERN_FOR[mb_double_${N}]=mb_vec_double_f32
done

# 1024-element PBPK 4-step f32 convergence:
#   per-thread C₀ = i+1, after 4 steps: C₄ = 0.0625*(i+1) + 4.6875
#   formatted with awk %g (matches runner's %.6g output)
EMEM_PBPK1024=$(seq 1 1024 | awk '{ printf "%s%g", (NR>1?",":""), $1*0.0625 + 4.6875 }')
CASES+=("pbpk4_1024|f32|1|1024|1024|_seq_||${EMEM_PBPK1024}|")
PATTERN_FOR[pbpk4_1024]=pbpk_4step

# Multi-block 2-compartment 1-step at 256 threads (8 blocks × 32):
#   C₁_new[i] = 0.75·(i+1) + 0.5 ; C₂_new[i] = 0.25·(i+1) ; C₂_0 = 0
EMEM_2CMB=$(seq 1 256 | awk '{ printf "%s%g", (NR>1?",":""), 0.75*$1 + 0.5 }')
EVAR_2CMB=$(seq 1 256 | awk '{ printf "%s%g", (NR>1?",":""), 0.25*$1 }')
INIT_2CMB_C2=$(seq 1 256 | awk '{ printf "%s0", (NR>1?",":"") }')
CASES+=("pbpk2c_mb|f32_2c|8|32|256|_seq_|${INIT_2CMB_C2}|${EMEM_2CMB}|${EVAR_2CMB}")
PATTERN_FOR[pbpk2c_mb]=pbpk_2comp_mb

# Multi-block 4-step coupled 2-compartment at 1024 threads (32×32):
# Expected computed via per-thread f32 simulation of the EXACT op sequence
# the kernel executes (so f32 rounding matches bit-exactly).
EMEM_2C4MB=$(cat "${ROOT_DIR}/tests/golden/kretikos_pbpk_expected/emem_2c4mb.txt")
EVAR_2C4MB=$(cat "${ROOT_DIR}/tests/golden/kretikos_pbpk_expected/evar_2c4mb.txt")
INIT_2C4MB_C2=$(seq 1 1024 | awk '{ printf "%s0", (NR>1?",":"") }')
CASES+=("pbpk2c4_mb|f32_2c|32|32|1024|_seq_|${INIT_2C4MB_C2}|${EMEM_2C4MB}|${EVAR_2C4MB}")
PATTERN_FOR[pbpk2c4_mb]=pbpk_2comp_4step_mb

# Clinical-scale dissertation chapter: 1024-patient f32-epistemic 4-step
# coupled-ODE with full GUM. Per-thread C₁_0 = tid+1, C₂_0 = 0,
# σ²(C₁)=1, σ²(C₂)=0. Single block of 1024 threads; mem layout is
# [C₁(1024) | C₂(1024)] = 2048 slots. ntid=1024 used for C₂ offset.
EMEM_2CE4_1024=$(cat "${ROOT_DIR}/tests/golden/kretikos_pbpk_expected/emem_2ce4_1024.txt")
EVAR_2CE4_1024=$(cat "${ROOT_DIR}/tests/golden/kretikos_pbpk_expected/evar_2ce4_1024.txt")
# Init mem: [tid+1 for tid 0..1023, then zeros for C2]
INIT_C1_C2=$(seq 1 1024 | awk '{ printf "%s%d", (NR>1?",":""), $1 }')$(seq 1 1024 | awk '{ printf ",0" }')
# Init var: [1 × 1024 for vC1, 0 × 1024 for vC2]
INIT_VC1_VC2=$(seq 1 1024 | awk '{ printf "%s1", (NR>1?",":"") }')$(seq 1 1024 | awk '{ printf ",0" }')
CASES+=("pbpk2c_e4_1024|f32e|1|1024|2048|${INIT_C1_C2}|${INIT_VC1_VC2}|${EMEM_2CE4_1024}|${EVAR_2CE4_1024}")
PATTERN_FOR[pbpk2c_e4_1024]=pbpk_2comp_epistemic_4step

# ----- build PTX for every case (locally, both basic + epistemic) -------
echo "[0/3] generating ${#CASES[@]} PTX kernels"
INDEX=0
> "${STAGE_DIR}/cases.tsv"
for c in "${CASES[@]}"; do
  IFS='|' read -r name mode blocks threads mw imem ivar emem evar <<<"$c"
  pat="${PATTERN_FOR[$name]}"
  [[ -z "$pat" ]] && { echo "unknown name: $name" >&2; exit 1; }
  ptx="${STAGE_DIR}/case_${INDEX}.ptx"
  EMIT_FLAGS=()
  [[ "$mode" == "epistemic" ]] && EMIT_FLAGS+=(--epistemic)
  [[ "$mode" == "f32" ]] && EMIT_FLAGS+=(--f32)
  [[ "$mode" == "f32e" ]] && EMIT_FLAGS+=(--f32-epistemic)
  [[ "$mode" == "f32_2c" ]] && EMIT_FLAGS+=(--f32-2c)
  "${ROOT_DIR}/bin/kretikos" kaxi-emit-ptx "$pat" "${EMIT_FLAGS[@]}" -o "$ptx" >/dev/null
  echo "${INDEX}|${name}|${mode}|${blocks}|${threads}|${mw}|${imem}|${ivar}|${emem}|${evar}" >> "${STAGE_DIR}/cases.tsv"
  INDEX=$((INDEX+1))
done

# ----- build runner -----------------------------------------------------
echo "[1/3] building runner"
cc -O2 "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c" -ldl -o "${STAGE_DIR}/runner"

# ----- worker driver script --------------------------------------------
cat > "${STAGE_DIR}/run_cases.sh" <<'SHELL_EOF'
#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
chmod +x runner
PASSED=0
TOTAL=0
FAILED_LIST=""
while IFS='|' read -r idx name mode blocks threads mw imem ivar emem evar; do
  TOTAL=$((TOTAL+1))
  ARGS=("case_${idx}.ptx" --blocks "$blocks" --threads "$threads" --mem-words "$mw")
  [[ "$mode" == "epistemic" ]] && ARGS+=(--epistemic)
  [[ "$mode" == "f32" ]] && ARGS+=(--type f32)
  [[ "$mode" == "f64" ]] && ARGS+=(--type f64)
  [[ "$mode" == "f32e" ]] && ARGS+=(--epistemic --type f32)
  [[ "$mode" == "f32_2c" ]] && ARGS+=(--epistemic --type f32)
  [[ "$mode" == "typed_f32" ]] && ARGS+=(--type f32)
  [[ "$mode" == "typed_i32" ]] && ARGS+=(--type i32)
  if [[ "$imem" == "_seq_" ]]; then
    ARGS+=(--init-seq)
  elif [[ -n "$imem" ]]; then
    ARGS+=(--init-mem "$imem")
  fi
  [[ -n "$ivar" ]] && ARGS+=(--init-var "$ivar")
  out="$(./runner "${ARGS[@]}" 2>&1)"
  rc=$?
  status="$(echo "$out" | grep '^sounio_kaxi_runtime' | tail -1 | awk '{print $2}' | sed 's/status=//')"
  mem_got="$(echo "$out" | grep '^MEM:' | head -1 | sed 's/MEM://' | xargs | tr ' ' ',')"
  var_got="$(echo "$out" | grep '^VAR:' | head -1 | sed 's/VAR://' | xargs | tr ' ' ',')"
  ok=1
  [[ "$status" != "pass" ]] && ok=0
  [[ "$mem_got" != "$emem" ]] && ok=0
  if [[ "$mode" == "epistemic" && -n "$evar" ]]; then
    [[ "$var_got" != "$evar" ]] && ok=0
  fi
  if [[ $ok -eq 1 ]]; then
    echo "PASS $idx $name $mode"
    PASSED=$((PASSED+1))
  else
    echo "FAIL $idx $name $mode status=$status"
    echo "  mem got=$mem_got"
    echo "  mem exp=$emem"
    if [[ "$mode" == "epistemic" ]]; then
      echo "  var got=$var_got"
      echo "  var exp=$evar"
    fi
    FAILED_LIST="${FAILED_LIST}${idx}_${name}_${mode},"
  fi
done < cases.tsv
echo "kaxi_matrix passed=${PASSED}/${TOTAL} failed=${FAILED_LIST%,}"
SHELL_EOF
chmod +x "${STAGE_DIR}/run_cases.sh"

# RUN_LOCAL=1: skip Slurm entirely, run on the local GPU directly.
if [[ "${RUN_LOCAL:-0}" == "1" ]]; then
  echo "[2/3] local GPU run (RUN_LOCAL=1)"
  cd "${STAGE_DIR}"
  export PATH="/usr/local/cuda/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
  ./run_cases.sh 2>&1
  cd "${ROOT_DIR}"
  exit $?
fi

echo "[2/3] packaging"
tar -C "${STAGE_DIR}" -czf "${LOCAL_TARBALL}" .
PAYLOAD_B64="$(base64 -w 0 "${LOCAL_TARBALL}" 2>/dev/null || base64 "${LOCAL_TARBALL}" | tr -d '\n')"

echo "[3/3] resolving login pod + submitting"
if [[ -n "${LOGIN_POD_NAME}" ]]; then
  LOGIN_POD="${LOGIN_POD_NAME}"
else
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi
[[ -z "${LOGIN_POD}" ]] && { echo "no live login pod" >&2; exit 1; }

cat > "${LOCAL_SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH -J ${RUN_ID}
#SBATCH -p ${SBATCH_PARTITION}
#SBATCH -A ${SBATCH_ACCOUNT}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 2
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH -w ${SBATCH_NODELIST}
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail
LOCAL_ROOT="/tmp/${RUN_ID}-\${SLURM_JOB_ID:-manual}"
LOG="\${LOCAL_ROOT}/run.log"
mark() { [[ -n "\${SLURM_JOB_ID:-}" ]] && scontrol update "JobId=\${SLURM_JOB_ID}" "Comment=\$1" >/dev/null 2>&1 || true; }
fail() {
  set +e; local rc=\$1 line=\$2
  local tail=""
  [[ -f "\${LOG}" ]] && tail="\$(tail -10 "\${LOG}" | tr '\n' ';' | tr -cd '[:alnum:]_./:=,; -' | cut -c1-700)"
  mark "kaxi_matrix=fail rc=\${rc} line=\${line} log=\${tail}"
  exit "\${rc}"
}
trap 'fail "\$?" "\$LINENO"' ERR
mkdir -p "\${LOCAL_ROOT}"
mark "kaxi_matrix=running phase=decode"
cat > "\${LOCAL_ROOT}/payload.tgz.b64" <<'PAYLOAD_EOF'
${PAYLOAD_B64}
PAYLOAD_EOF
base64 -d "\${LOCAL_ROOT}/payload.tgz.b64" > "\${LOCAL_ROOT}/payload.tgz"
tar -xzf "\${LOCAL_ROOT}/payload.tgz" -C "\${LOCAL_ROOT}"
cd "\${LOCAL_ROOT}"
export PATH="/usr/local/cuda/bin:/usr/bin:/bin:/usr/sbin:/sbin:\${PATH:-}"
mark "kaxi_matrix=running phase=run"
./run_cases.sh > "\${LOG}" 2>&1
summary="\$(grep '^kaxi_matrix' "\${LOG}" | tail -1 || echo "kaxi_matrix=no_summary")"
short="\$(echo "\${summary}" | tr -cd '[:alnum:]_./:=,; -' | cut -c1-700)"
mark "\${short}"
EOF

"${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch" >/dev/null
JOB_ID="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch --parsable /tmp/${RUN_ID}.sbatch" | tr -d '\r\n' | awk '{print $NF}')"
[[ -z "${JOB_ID}" || ! "${JOB_ID}" =~ ^[0-9]+$ ]] && { echo "submit failed: ${JOB_ID}" >&2; exit 1; }
echo "submitted job: ${JOB_ID}"

[[ "${WAIT_FOR_RESULT}" != "1" ]] && exit 0

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while [[ $(date +%s) -lt $deadline ]]; do
  state="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
    "scontrol show job ${JOB_ID} --oneliner 2>/dev/null | tr ' ' '\n' | grep -E '^JobState=' | head -1" \
    | tr -d '\r\n' || true)"
  echo "  job ${JOB_ID}: ${state}"
  case "${state}" in
    JobState=COMPLETED|JobState=FAILED|JobState=CANCELLED|JobState=TIMEOUT|JobState=NODE_FAIL|JobState=BOOT_FAIL) break ;;
  esac
  sleep 5
done

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "scontrol show job ${JOB_ID}" \
  | grep -E "JobState|ExitCode|Comment"
