#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_iterate.sh
#
# C1 v0.4 — iterative sedenion compute on A5000 with GUM across iterations.
#
# Recurrence (linear, since Mandelbrot's z·z forces same-register-squared
# variance which sedenion_mul --f32-epistemic doesn't yet handle correctly):
#
#     z_{n+1} = z_n · c     (c fixed, σ²(c) = 0)
#     z_0     = e₁ + e₂     (sedenion sparse, σ²(z_0[1])=σ²(z_0[2])=1)
#     c       = e₃          (right multiplier)
#
# Each step:
#   1. Build the input pair (z, c) and σ²(z, c) on host
#   2. Launch sedenion_mul on A5000 with 1 thread (one product)
#   3. Read u = z·c and σ²(u) from GPU
#   4. z := u, σ²(z) := σ²(u), advance
#
# Reports trajectory (κ_n, σ²(κ_n), σ/κ) over N=4 steps.
#
# Why this matters: first iterative epistemic sedenion compute on owned
# silicon. The full Mandelbrot z := z² + c will require either:
#   (a) a kaxi_to_ptx variance-emit path that detects when both inputs
#       come from the same memory region and applies the X² rule
#       (4·X²·σ²(X)) instead of the independent-pair rule (2·X²·σ²(X)),
#   (b) a new K-AXI op for "in-place square" with the X² variance rule baked in.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_ITER_WORK:-$(mktemp -d /tmp/c1_iter.XXXXXX)}"
STEPS="${SOUNIO_C1_ITER_STEPS:-4}"

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

PTX="$WORK/sedenion_mul.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_mul --f32-epistemic -o "$PTX" --no-ptxas \
    >"$WORK/emit.log" 2>&1

# Initial state (16 floats each).
Z="0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0"        # z_0 = e₁ + e₂
C="0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0"        # c   = e₃
SZ="0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0"       # σ²(z_0[1]) = σ²(z_0[2]) = 1
SC="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"       # σ²(c) = 0

# Reduce κ = Σ u_i², σ²(κ) = Σ (2·u_i)²·σ²(u_i)
reduce() {
    local mem="$1" var="$2"
    awk -v mem="$mem" -v var="$var" '
    BEGIN {
        n = split(mem, M, " "); split(var, V, " ")
        kappa = 0; sig2 = 0
        for (i = 1; i <= n; i++) {
            ui = M[i] + 0; vi = V[i] + 0
            kappa += ui * ui
            sig2  += 4 * ui * ui * vi
        }
        printf "%s %s\n", kappa, sig2
    }'
}

read kappa sig2 < <(reduce "$Z" "$SZ")
sigma=$(awk -v s2="$sig2" 'BEGIN { printf "%.4f", sqrt(s2) }')
relunc=$(awk -v k="$kappa" -v s="$sigma" 'BEGIN { printf "%.4f", (k > 0 ? s / k : 0) }')
printf "step %d  κ=%-8.4f  σ²(κ)=%-8.4f  σ(κ)=%-8s  σ/κ=%-8s\n" 0 "$kappa" "$sig2" "$sigma" "$relunc"

# Helper: build one CSV row from a 16-element space-separated string.
mk_csv16() { tr ' ' ',' <<<"$1"; }

for step in $(seq 1 "$STEPS"); do
    # Layout: [a (16), b (16), out (16)] = 48 floats
    INIT_MEM="$(mk_csv16 "$Z"),$(mk_csv16 "$C"),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    INIT_VAR="$(mk_csv16 "$SZ"),$(mk_csv16 "$SC"),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

    OUT="$WORK/step${step}.out"
    "$RUNNER" "$PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
        --blocks 1 --threads 1 --mem-words 48 \
        --init-mem "$INIT_MEM" --init-var "$INIT_VAR" --print-count 48 \
        > "$OUT" 2>&1

    grep -q "status=pass" "$OUT" || {
        echo "[FAIL] step $step: GPU launch failed"
        tail -10 "$OUT"
        exit 1
    }

    # u = MEM[32..47], σ²(u) = VAR[32..47]
    Z=$(grep '^MEM:' "$OUT" | awk '{ for (i = 33; i <= 48; i++) printf "%s%s", $i, (i < 48 ? " " : "") }')
    SZ=$(grep '^VAR:' "$OUT" | awk '{ for (i = 33; i <= 48; i++) printf "%s%s", $i, (i < 48 ? " " : "") }')

    read kappa sig2 < <(reduce "$Z" "$SZ")
    sigma=$(awk -v s2="$sig2" 'BEGIN { printf "%.4f", sqrt(s2) }')
    relunc=$(awk -v k="$kappa" -v s="$sigma" 'BEGIN { printf "%.4f", (k > 0 ? s / k : 0) }')
    printf "step %d  κ=%-8.4f  σ²(κ)=%-8.4f  σ(κ)=%-8s  σ/κ=%-8s   z = %s\n" \
        "$step" "$kappa" "$sig2" "$sigma" "$relunc" "$Z"
done

echo
echo "=== C1 v0.4 — $STEPS-step iterative sedenion z := z·c on A5000 ==="
echo "    work dir: $WORK"
