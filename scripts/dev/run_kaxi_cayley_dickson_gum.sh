#!/usr/bin/env bash
# scripts/dev/run_kaxi_cayley_dickson_gum.sh
#
# Local-A5000 GUM regression for the four Cayley-Dickson kernels.
# Verifies that K-AXI → PTX → A5000 emits AND propagates GUM variance
# end-to-end for octonion_mul, octonion_associator, sedenion_mul, and
# sedenion_associator in --f32-epistemic mode.
#
# This complements the slurm matrix gate at
#   slurm-jobs/kretikos/submit-kaxi-ptx-matrix.sh
# but runs entirely locally via the kretikos PTX runner — no slurm needed.
#
# Inputs per case: each component is a basis vector (e₁, e₂, e₄) with
# σ²=1 on every input component. Expected outputs are the canonical
# Cayley-Dickson product / associator values:
#
#   octonion_mul        : c = e₁·e₂ = e₃,         σ²(c_k) = 2   ∀k
#   octonion_associator : [e₁,e₂,e₄] = ±2·e₇,     σ² = 6        ∀k
#   sedenion_mul        : c = e₁·e₂ = e₃ (16-c),  σ²(c_k) = 2   ∀k
#   sedenion_associator : [e₁,e₂,e₄] = ±e₇ (16-c),σ² = 3        ∀k
#
# Single observed-value canon per kernel — caller can extend.
#
# Requirements: A5000 (or any cuda-cc6.0+ GPU) reachable via libcuda.so;
# the runner at /tmp/runner (built once via:
#   cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o /tmp/runner
# ) auto-rebuilt if missing.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_KAXI_GUM_WORK:-$(mktemp -d /tmp/cayley_gum.XXXXXX)}"
PASS=0
FAIL=0

if [ ! -x "$RUNNER" ]; then
    echo "[setup] building runner -> $RUNNER"
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

# Args: <label> <pattern> <mem-words> <init-mem CSV> <init-var CSV> <expected-mem CSV> <expected-var CSV>
check_case() {
    local label="$1" pattern="$2" mw="$3" init_mem="$4" init_var="$5" exp_mem="$6" exp_var="$7"
    local safe_label="${label//\//_}"
    local ptx="$WORK/${pattern}.ptx"
    local out="$WORK/${safe_label}.out"

    if [ ! -f "$ptx" ]; then
        ./bin/kretikos kaxi-emit-ptx "$pattern" --f32-epistemic -o "$ptx" --no-ptxas \
            >"$WORK/${pattern}.emit.log" 2>&1 || {
            echo "[FAIL] $label: kaxi-emit-ptx failed"
            FAIL=$((FAIL + 1))
            return
        }
    fi

    "$RUNNER" "$ptx" --kernel kaxi_kernel --mode epistemic --type f32 \
        --blocks 1 --threads 1 --mem-words "$mw" \
        --init-mem "$init_mem" --init-var "$init_var" --print-count "$mw" \
        > "$out" 2>&1 || {
        echo "[FAIL] $label: runner failure"
        cat "$out"
        FAIL=$((FAIL + 1))
        return
    }

    # Parse MEM/VAR lines and compare against expected.
    local actual_mem actual_var
    actual_mem=$(grep '^MEM:' "$out" | sed 's/^MEM: //; s/ /,/g')
    actual_var=$(grep '^VAR:' "$out" | sed 's/^VAR: //; s/ /,/g')
    if [ "$actual_mem" = "$exp_mem" ] && [ "$actual_var" = "$exp_var" ]; then
        echo "[PASS] $label"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $label"
        echo "  expected MEM: $exp_mem"
        echo "  actual   MEM: $actual_mem"
        echo "  expected VAR: $exp_var"
        echo "  actual   VAR: $actual_var"
        FAIL=$((FAIL + 1))
    fi
}

# --- octonion_mul: a=e₁, b=e₂, σ²=1 each → c=e₃, σ²(c_k)=2 ------------
A_O="0,1,0,0,0,0,0,0"
B_O="0,0,1,0,0,0,0,0"
C0_O="0,0,0,0,0,0,0,0"
ONES_O="1,1,1,1,1,1,1,1"
ZEROS_O="0,0,0,0,0,0,0,0"
TWOS_O="2,2,2,2,2,2,2,2"
SIXES_O="6,6,6,6,6,6,6,6"

check_case "octonion_mul/e1*e2" octonion_mul 24 \
    "$A_O,$B_O,$C0_O" \
    "$ONES_O,$ONES_O,$ZEROS_O" \
    "$A_O,$B_O,0,0,0,1,0,0,0,0" \
    "$ONES_O,$ONES_O,$TWOS_O"

# --- octonion_associator: [e₁,e₂,e₄] = ±2·e₇, σ²=6 ---------------------
A2_O="$A_O"
B2_O="$B_O"
C2_O="0,0,0,0,1,0,0,0"
OUT0_O="$C0_O"

check_case "octonion_associator/[e1,e2,e4]" octonion_associator 32 \
    "$A2_O,$B2_O,$C2_O,$OUT0_O" \
    "$ONES_O,$ONES_O,$ONES_O,$ZEROS_O" \
    "$A2_O,$B2_O,$C2_O,0,0,0,0,0,0,0,2" \
    "$ONES_O,$ONES_O,$ONES_O,$SIXES_O"

# --- sedenion_mul: a=e₁ (16-c), b=e₂ (16-c) → c=e₃ at 16-c, σ²=2 -------
A_S="0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
B_S="0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0"
C0_S="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
ONES_S="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
ZEROS_S="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
TWOS_S="2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
THREES_S="3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"
EXP_E3_16="0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0"
EXP_E7_16="0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0"

check_case "sedenion_mul/e1*e2" sedenion_mul 48 \
    "$A_S,$B_S,$C0_S" \
    "$ONES_S,$ONES_S,$ZEROS_S" \
    "$A_S,$B_S,$EXP_E3_16" \
    "$ONES_S,$ONES_S,$TWOS_S"

# --- sedenion_associator: [e₁,e₂,e₄] = ±e₇ (16-c), σ²=3 ----------------
A2_S="$A_S"
B2_S="$B_S"
C2_S="0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0"
OUT0_S="$C0_S"

check_case "sedenion_associator/[e1,e2,e4]" sedenion_associator 64 \
    "$A2_S,$B2_S,$C2_S,$OUT0_S" \
    "$ONES_S,$ONES_S,$ONES_S,$ZEROS_S" \
    "$A2_S,$B2_S,$C2_S,$EXP_E7_16" \
    "$ONES_S,$ONES_S,$ONES_S,$THREES_S"

echo "==="
echo "kaxi_cayley_dickson_gum: PASS=$PASS FAIL=$FAIL  (work=$WORK)"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
