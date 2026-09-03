#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_sweep.sh
#
# C1 v0.2 — sedenion epistemic curvature sweep on A5000 (32-point grid,
# single GPU launch). Extends scripts/dev/run_epistemic_sedenion_hessian.sh.
#
# Parameter sweep:
#     t_n = n / 31   for n = 0..31
#     z(t) = (1 + t) · e₁ + t · e₂
#     v    = e₃
#     u    = z · v   (sedenion product, computed on A5000)
#     κ(t) = ‖u‖²    (host-side reduction)
#
# Theoretical: κ(t) = (1+t)² + t² = 1 + 2t + 2t² (quadratic curvature
# growth). Verifying on A5000 demonstrates parameter-sensitive epistemic
# sedenion compute end-to-end at small grid scale — the foundation for
# eventual full Mandelbrot/Hessian sweeps.
#
# Why this matters: the project_sedenion_hessian.md memory notes "30×
# curvature amplification Q→S" as the dissertation's headline finding.
# This script is the smallest-possible runnable demonstration that the
# K-AXI/PTX/A5000 stack handles parameter-varying sedenion compute with
# GUM. It does NOT do the Q→S comparison or the Mandelbrot iterate; it
# DOES prove the substrate runs the right shape of computation.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_SWEEP_WORK:-$(mktemp -d /tmp/c1_sweep.XXXXXX)}"
N=32

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

PTX="$WORK/sedenion_mul.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_mul --f32-epistemic -o "$PTX" --no-ptxas \
    >"$WORK/emit.log" 2>&1

# Generate per-thread inputs: 32 threads × 48 mem-words each.
# Layout per thread (offset = tid * 48):
#   [0..15]  : a = z(t) = (1+t)·e₁ + t·e₂   → a[1] = 1+t, a[2] = t, rest 0
#   [16..31] : b = v = e₃                    → b[3] = 1, rest 0
#   [32..47] : c (output region, init 0)
#
# σ²: σ²(a[1]) = σ²(a[2]) = 1, σ²(b[3]) = 1, others 0.

INIT_MEM=$(awk -v N=$N 'BEGIN {
    for (n = 0; n < N; n++) {
        t = n / (N - 1)
        # a (sedenion): 16 components
        for (i = 0; i < 16; i++) {
            if (i == 1)      printf "%g,", 1.0 + t
            else if (i == 2) printf "%g,", t
            else             printf "0,"
        }
        # b (sedenion): e₃
        for (i = 0; i < 16; i++) {
            if (i == 3) printf "1,"
            else        printf "0,"
        }
        # c (output): zeros
        for (i = 0; i < 16; i++) {
            printf "0%s", (n == N - 1 && i == 15 ? "" : ",")
        }
    }
    printf "\n"
}')

INIT_VAR=$(awk -v N=$N 'BEGIN {
    for (n = 0; n < N; n++) {
        # σ²(a): only positions 1 and 2 have variance
        for (i = 0; i < 16; i++) {
            if (i == 1 || i == 2) printf "1,"
            else                  printf "0,"
        }
        # σ²(b): only position 3
        for (i = 0; i < 16; i++) {
            if (i == 3) printf "1,"
            else        printf "0,"
        }
        # σ²(c): zeros
        for (i = 0; i < 16; i++) {
            printf "0%s", (n == N - 1 && i == 15 ? "" : ",")
        }
    }
    printf "\n"
}')

OUT="$WORK/run.out"
"$RUNNER" "$PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
    --blocks 1 --threads $N --mem-words $((N * 48)) \
    --init-mem "$INIT_MEM" --init-var "$INIT_VAR" --print-count $((N * 48)) \
    > "$OUT" 2>&1

# Per-thread reduce κ_n = Σ u_n[i]², σ²(κ_n) = Σ (2·u_n[i])²·σ²(u_n[i]).
MEM_ROW=$(grep '^MEM:' "$OUT" | sed 's/^MEM: //')
VAR_ROW=$(grep '^VAR:' "$OUT" | sed 's/^VAR: //')

awk -v N=$N -v mem="$MEM_ROW" -v var="$VAR_ROW" '
BEGIN {
    nm = split(mem, M, " ")
    nv = split(var, V, " ")
    if (nm != N * 48 || nv != N * 48) {
        printf "[FAIL] expected %d mem and %d var values, got %d and %d\n", N*48, N*48, nm, nv > "/dev/stderr"
        exit 1
    }
    printf "  n   t          kappa     sigma2(kappa)   sigma(kappa)   theory(1+2t+2t²)\n"
    for (n = 0; n < N; n++) {
        t = n / (N - 1)
        kappa = 0
        sig2  = 0
        for (i = 0; i < 16; i++) {
            ui = M[n * 48 + 32 + i + 1] + 0
            vi = V[n * 48 + 32 + i + 1] + 0
            kappa += ui * ui
            sig2  += 4 * ui * ui * vi
        }
        theory = 1 + 2 * t + 2 * t * t
        printf "  %2d  %.4f  %10.4f  %14.4f  %12.4f   %.4f\n", n, t, kappa, sig2, sqrt(sig2), theory
    }
}'

echo
echo "=== C1 v0.2 — 32-point sedenion epistemic sweep on A5000  ==="
echo "    work dir: $WORK"
