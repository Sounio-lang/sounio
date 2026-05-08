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
  "vec_add|basic|1|8|8|1,2,3,4,5,6,7,8||2,4,6,8,10,12,14,16|"
  "vec_add|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|2,4,6,8,10,12,14,16|2,2,2,2,2,2,2,2"
  "vec_sub|basic|1|8|8|1,2,3,4,5,6,7,8||0,0,0,0,0,0,0,0|"
  "vec_sub|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|0,0,0,0,0,0,0,0|2,2,2,2,2,2,2,2"
  "vec_mul|basic|1|8|8|1,2,3,4,5,6,7,8||1,4,9,16,25,36,49,64|"
  "vec_mul|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|1,4,9,16,25,36,49,64|2,8,18,32,50,72,98,128"
  "vec_div|basic|1|8|8|1,2,3,4,5,6,7,8||1,1,1,1,1,1,1,1|"
  "vec_div|epistemic|1|8|8|1,2,3,4,5,6,7,8|1,1,1,1,1,1,1,1|1,1,1,1,1,1,1,1|2,2,2,2,2,2,2,2"
  "fma|basic|1|8|8|1,2,3,4,5,6,7,8||2,6,12,20,30,42,56,72|"
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
PATTERN_FOR[vec_min]=vec_min
PATTERN_FOR[vec_max]=vec_max

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
SEDZD_INIT=$(python3 -c "
m=[0.0]*96
m[1]=1; m[10]=1; m[16+5]=1; m[16+14]=1
m[48+0]=1; m[48+16+0]=1
print(','.join(f'{int(x) if x==int(x) else x:g}' for x in m))")
SEDZD_EXP=$(python3 -c "
m=[0.0]*96
m[1]=1; m[10]=1; m[16+5]=1; m[16+14]=1
m[48+0]=1; m[48+16+0]=1; m[48+32+0]=1
print(','.join(f'{int(x) if x==int(x) else x:g}' for x in m))")
CASES+=("sedenion_mul|f32|1|2|96|${SEDZD_INIT}||${SEDZD_EXP}|")

# Phase I: re-segmented variance register file (value 0..127, variance
# 128..255, scratch %rd256/%rd257, base %rd258/%rd259) unblocks the
# big-kernel variance modes that Phase H surfaced as broken.
SEDZD_VAR_ZERO=$(python3 -c "print(','.join(['0']*96))")
CASES+=("sedenion_mul|epistemic|1|2|96|${SEDZD_INIT}|${SEDZD_VAR_ZERO}|${SEDZD_EXP}|${SEDZD_VAR_ZERO}")
CASES+=("sedenion_mul|f32e|1|2|96|${SEDZD_INIT}|${SEDZD_VAR_ZERO}|${SEDZD_EXP}|${SEDZD_VAR_ZERO}")

# Phase J: octonion_mul in variance modes — gives the octonion-connectomics
# research line a GUM-propagating GPU primitive. Inputs are 8-vector basis
# elements; outputs are the 8-component product. With init_var=0, var stays
# 0 through every mul (zero · anything = zero). Same mem expected as f32.
OCT_INIT="1,2,3,4,5,6,7,8,9,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0"
OCT_EXP="1,2,3,4,5,6,7,8,9,1,0,0,0,0,0,0,7,19,31,33,51,49,55,79,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0"
OCT_VAR_ZERO=$(python3 -c "print(','.join(['0']*48))")
CASES+=("octonion_mul|epistemic|1|2|48|${OCT_INIT}|${OCT_VAR_ZERO}|${OCT_EXP}|${OCT_VAR_ZERO}")
CASES+=("octonion_mul|f32e|1|2|48|${OCT_INIT}|${OCT_VAR_ZERO}|${OCT_EXP}|${OCT_VAR_ZERO}")
PATTERN_FOR[sedenion_mul]=sedenion_mul

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
EMEM_2C4MB=$(python3 -c "
import struct
def step(c1, c2, k_in=1.0, k12=0.5, k21=0.25, dt=0.5):
    def f32(x): return struct.unpack('f', struct.pack('f', x))[0]
    kc1=f32(k12*c1); kc2=f32(k21*c2); di=f32(k_in-kc1); d1=f32(di+kc2)
    dd=f32(dt*d1); c1n=f32(c1+dd); df=f32(kc1-kc2); ddf=f32(dt*df); c2n=f32(c2+ddf)
    return c1n, c2n
out=[]
for i in range(1024):
    c1,c2 = float(i+1), 0.0
    for _ in range(4): c1,c2 = step(c1,c2)
    out.append(f'{c1:g}')
print(','.join(out))")
EVAR_2C4MB=$(python3 -c "
import struct
def step(c1, c2, k_in=1.0, k12=0.5, k21=0.25, dt=0.5):
    def f32(x): return struct.unpack('f', struct.pack('f', x))[0]
    kc1=f32(k12*c1); kc2=f32(k21*c2); di=f32(k_in-kc1); d1=f32(di+kc2)
    dd=f32(dt*d1); c1n=f32(c1+dd); df=f32(kc1-kc2); ddf=f32(dt*df); c2n=f32(c2+ddf)
    return c1n, c2n
out=[]
for i in range(1024):
    c1,c2 = float(i+1), 0.0
    for _ in range(4): c1,c2 = step(c1,c2)
    out.append(f'{c2:g}')
print(','.join(out))")
INIT_2C4MB_C2=$(seq 1 1024 | awk '{ printf "%s0", (NR>1?",":"") }')
CASES+=("pbpk2c4_mb|f32_2c|32|32|1024|_seq_|${INIT_2C4MB_C2}|${EMEM_2C4MB}|${EVAR_2C4MB}")
PATTERN_FOR[pbpk2c4_mb]=pbpk_2comp_4step_mb

# Clinical-scale dissertation chapter: 1024-patient f32-epistemic 4-step
# coupled-ODE with full GUM. Per-thread C₁_0 = tid+1, C₂_0 = 0,
# σ²(C₁)=1, σ²(C₂)=0. Single block of 1024 threads; mem layout is
# [C₁(1024) | C₂(1024)] = 2048 slots. ntid=1024 used for C₂ offset.
EMEM_2CE4_1024=$(python3 -c "
import struct
def f32(x): return struct.unpack('f', struct.pack('f', x))[0]
def step(c1, vc1, c2, vc2):
    k_in,k12,k21,dt = 1.0,0.5,0.25,0.5
    kc1=f32(k12*c1); vkc1=f32(f32(k12*k12)*vc1)
    kc2=f32(k21*c2); vkc2=f32(f32(k21*k21)*vc2)
    di=f32(k_in-kc1); vdi=vkc1
    d1=f32(di+kc2); vd1=f32(vdi+vkc2)
    dd=f32(dt*d1); vdd=f32(f32(dt*dt)*vd1)
    c1n=f32(c1+dd); vc1n=f32(vc1+vdd)
    df=f32(kc1-kc2); vdf=f32(vkc1+vkc2)
    ddf=f32(dt*df); vddf=f32(f32(dt*dt)*vdf)
    c2n=f32(c2+ddf); vc2n=f32(vc2+vddf)
    return c1n, vc1n, c2n, vc2n
c1s,c2s = [], []
for tid in range(1024):
    c1,c2 = float(tid+1), 0.0; vc1,vc2 = 1.0, 0.0
    for _ in range(4):
        c1,vc1,c2,vc2 = step(c1,vc1,c2,vc2)
    c1s.append(f'{c1:g}'); c2s.append(f'{c2:g}')
print(','.join(c1s + c2s))")
EVAR_2CE4_1024=$(python3 -c "
import struct
def f32(x): return struct.unpack('f', struct.pack('f', x))[0]
def step(c1, vc1, c2, vc2):
    k_in,k12,k21,dt = 1.0,0.5,0.25,0.5
    kc1=f32(k12*c1); vkc1=f32(f32(k12*k12)*vc1)
    kc2=f32(k21*c2); vkc2=f32(f32(k21*k21)*vc2)
    di=f32(k_in-kc1); vdi=vkc1
    d1=f32(di+kc2); vd1=f32(vdi+vkc2)
    dd=f32(dt*d1); vdd=f32(f32(dt*dt)*vd1)
    c1n=f32(c1+dd); vc1n=f32(vc1+vdd)
    df=f32(kc1-kc2); vdf=f32(vkc1+vkc2)
    ddf=f32(dt*df); vddf=f32(f32(dt*dt)*vdf)
    c2n=f32(c2+ddf); vc2n=f32(vc2+vddf)
    return c1n, vc1n, c2n, vc2n
v1s,v2s = [], []
for tid in range(1024):
    c1,c2 = float(tid+1), 0.0; vc1,vc2 = 1.0, 0.0
    for _ in range(4):
        c1,vc1,c2,vc2 = step(c1,vc1,c2,vc2)
    v1s.append(f'{vc1:g}'); v2s.append(f'{vc2:g}')
print(','.join(v1s + v2s))")
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
