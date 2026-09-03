#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_hessian.sh
#
# C1 v0.1 — first epistemic Sedenion compute on A5000.
#
# Composes the verified sedenion_mul K-AXI kernel (variance lanes wired
# end-to-end, see scripts/dev/run_kaxi_cayley_dickson_gum.sh) with a
# host-side scalar reduction to produce a quadratic form
#
#   κ(z, v) = ‖ z · v ‖²  ∈ ℝ
#
# carrying GUM-propagated variance σ²(κ) from σ²(z) and σ²(v).
#
# This is *not* the full sedenion-Mandelbrot Hessian from
# project_sedenion_hessian.md (that needs grid sweep + 30× Q→S
# amplification probe + symmetry classification). It IS the building
# block: first time the K-AXI/PTX/A5000 stack runs a sedenion-algebra
# product end-to-end with full GUM variance propagation, reduced to
# a scalar epistemic observable on owned silicon.
#
# Math
# ----
# 1. GPU computes u = z · v  (16-component sedenion product)
#    via the per-thread sedenion_mul kernel, with paired variance lane:
#    σ²(u_i) = Σ_t (a_t² · σ²(b_t) + b_t² · σ²(a_t))   for the t-th
#    contributing pair in the i-th output's Cayley-Dickson sum.
#
# 2. Host computes:
#       κ      = Σ_i u_i²
#       σ²(κ)  = Σ_i (2 · u_i)² · σ²(u_i)
#    (independent-component approximation; tightens with diagonal
#    structure but is exact when only one u_i is non-zero, as in
#    the canonical e₁·e₂ probe.)
#
# Canonical case
# --------------
#   z = e₁ = (0, 1, 0, ..., 0),  σ²(z_i) = 1 ∀i
#   v = e₂ = (0, 0, 1, 0, ..., 0),  σ²(v_i) = 1 ∀i
#   u = z · v = e₃ = (0, 0, 0, 1, 0, ..., 0)
#   σ²(u_i) = 2 ∀i  (verified bit-exact in run_kaxi_cayley_dickson_gum.sh)
#   κ = Σ u_i² = 1
#   σ²(κ) = (2 · u₃)² · σ²(u₃) = 4 · 2 = 8
#
# So canonical output: κ = 1, σ²(κ) = 8, i.e. κ = 1 ± √8 ≈ 1 ± 2.828.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_WORK:-$(mktemp -d /tmp/c1.XXXXXX)}"

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

PTX="$WORK/sedenion_mul.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_mul --f32-epistemic -o "$PTX" --no-ptxas \
    >"$WORK/emit.log" 2>&1

# Inputs: z = e₁, v = e₂, all σ²=1 per component, output region σ²=0.
Z="0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
V="0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0"
OUT0="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
ONES="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
ZEROS="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

INIT_MEM="$Z,$V,$OUT0"
INIT_VAR="$ONES,$ONES,$ZEROS"

OUT="$WORK/run.out"
"$RUNNER" "$PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
    --blocks 1 --threads 1 --mem-words 48 \
    --init-mem "$INIT_MEM" --init-var "$INIT_VAR" --print-count 48 \
    > "$OUT" 2>&1

# Parse u (mem[32..47]) and σ²(u) (var[32..47]) — last 16 of 48 each.
read_slice() {
    local stream="$1" file="$2" start="$3" count="$4"
    grep "^${stream}:" "$file" | sed "s/^${stream}: //" \
        | awk -v s="$start" -v c="$count" '{ for (i = s + 1; i <= s + c; i++) printf "%s%s", $i, (i < s + c ? " " : "\n") }'
}
U=$(read_slice MEM "$OUT" 32 16)
SIG2_U=$(read_slice VAR "$OUT" 32 16)

# Host-side reduction (use awk for f32-clean arithmetic).
awk -v u="$U" -v var="$SIG2_U" '
BEGIN {
    n = split(u, U_a, " ")
    split(var, V_a, " ")
    kappa = 0
    sig2 = 0
    for (i = 1; i <= n; i++) {
        ui = U_a[i] + 0
        vi = V_a[i] + 0
        kappa += ui * ui
        sig2  += 4 * ui * ui * vi
    }
    printf "u                = %s\n", u
    printf "sigma2(u)        = %s\n", var
    printf "kappa = sum u_i² = %g\n", kappa
    printf "sigma2(kappa)    = %g\n", sig2
    printf "stddev(kappa)    = %g\n", sqrt(sig2)
}'

# Canonical-case sanity check
EXPECT_U="0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0"
EXPECT_S="2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
if [ "$U" = "$EXPECT_U" ] && [ "$SIG2_U" = "$EXPECT_S" ]; then
    echo "[PASS] canonical e₁·e₂ — u and σ²(u) match expected"
    echo "       κ=1 ± √8 ≈ 1 ± 2.828 (GUM through sedenion product on A5000)"
else
    echo "[FAIL] canonical case mismatch"
    echo "  expected u: $EXPECT_U"
    echo "  actual u:   $U"
    echo "  expected σ²(u): $EXPECT_S"
    echo "  actual σ²(u):   $SIG2_U"
    exit 1
fi

echo "===  C1 v0.1: first epistemic sedenion compute on A5000 — PASS  ==="
echo "    work dir: $WORK"
