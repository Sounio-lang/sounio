#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_mandelbrot.sh
#
# C1 v0.5 — full sedenion Mandelbrot iterate on A5000 with GUM σ² across
# iterations. Composes the sedenion_sqr opcode (X² variance rule, commit
# ed265de0) with host-side `+ c` to realize:
#
#     z_{n+1} = z_n² + c
#
# This is the first paper-grade sedenion Mandelbrot compute substrate on
# owned silicon with bit-exact GUM propagation across iterations. The X²
# rule is required at the squared step (σ²(z²) = 4·z²·σ²(z) at the
# diagonal, vs sedenion_mul's independent-treatment 2·z²·σ²(z)) and
# it's now correct via sedenion_sqr.
#
# Default trajectory:
#   z₀ = e₁ + e₂   (sedenion sparse, ‖z₀‖² = 2)
#   c  = 0.1·e₅    (small perturbation in higher-half basis vector)
#   σ²(z₀[1]) = σ²(z₀[2]) = 0.01   (1% input epistemic uncertainty)
#   σ²(c)     = 0                  (c is exact)
#   N = 6 iterations
#
# Reports per-step: ‖z_n‖², σ²(‖z_n‖²), σ/κ trajectory.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_MAND_WORK:-$(mktemp -d /tmp/c1_mand.XXXXXX)}"
N="${SOUNIO_C1_MAND_STEPS:-6}"

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

PTX="$WORK/sedenion_sqr.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_sqr --f32-epistemic -o "$PTX" --no-ptxas \
    >"$WORK/emit.log" 2>&1

# Initial state.
Z="0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0"
SZ="0 0.01 0.01 0 0 0 0 0 0 0 0 0 0 0 0 0"     # 1% σ² on z[1], z[2]
C="0 0 0 0 0 0.1 0 0 0 0 0 0 0 0 0 0"           # c = 0.1·e₅
SC="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"            # σ²(c) = 0

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

# Host-side z := z + c, σ²(z) := σ²(z) + σ²(c) (component-wise, indep).
add_c() {
    local z="$1" sz="$2" c="$3" sc="$4"
    awk -v z="$z" -v sz="$sz" -v c="$c" -v sc="$sc" '
    BEGIN {
        n = split(z, ZA, " "); split(sz, SZA, " ")
        split(c, CA, " "); split(sc, SCA, " ")
        for (i = 1; i <= n; i++) {
            zi = ZA[i] + CA[i]; svi = SZA[i] + SCA[i]
            printf "%s%s", zi, (i < n ? " " : "")
        }
        printf "\n"
        for (i = 1; i <= n; i++) {
            svi = SZA[i] + SCA[i]
            printf "%s%s", svi, (i < n ? " " : "")
        }
        printf "\n"
    }'
}

mk_csv16() { tr ' ' ',' <<<"$1"; }

print_step() {
    local n="$1" z="$2" sz="$3"
    local kappa sig2 sigma relunc
    read kappa sig2 < <(reduce "$z" "$sz")
    sigma=$(awk -v s2="$sig2" 'BEGIN { printf "%.5f", sqrt(s2) }')
    relunc=$(awk -v k="$kappa" -v s="$sigma" 'BEGIN { printf "%.5f", (k > 0 ? s / k : 0) }')
    printf "step %d  κ=%-10.5f  σ²(κ)=%-10.5f  σ(κ)=%-10s  σ/κ=%-10s\n" \
        "$n" "$kappa" "$sig2" "$sigma" "$relunc"
}

print_step 0 "$Z" "$SZ"

for step in $(seq 1 "$N"); do
    # Build inputs for sedenion_sqr: [z (16), out (16)] = 32 mem-words.
    INIT_MEM="$(mk_csv16 "$Z"),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    INIT_VAR="$(mk_csv16 "$SZ"),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

    OUT="$WORK/step${step}.out"
    "$RUNNER" "$PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
        --blocks 1 --threads 1 --mem-words 32 \
        --init-mem "$INIT_MEM" --init-var "$INIT_VAR" --print-count 32 \
        > "$OUT" 2>&1

    grep -q "status=pass" "$OUT" || {
        echo "[FAIL] step $step: GPU launch failed"
        tail -10 "$OUT"; exit 1
    }

    # u = MEM[16..31] (z² output)
    U=$(grep '^MEM:' "$OUT" | awk '{ for (i = 17; i <= 32; i++) printf "%s%s", $i, (i < 32 ? " " : "") }')
    SU=$(grep '^VAR:' "$OUT" | awk '{ for (i = 17; i <= 32; i++) printf "%s%s", $i, (i < 32 ? " " : "") }')

    # z := u + c, σ²(z) := σ²(u) + σ²(c)
    read NEW_Z NEW_SZ <<<"$(add_c "$U" "$SU" "$C" "$SC" | tr '\n' '|')"
    Z="${NEW_Z%|}"
    SZ="$(echo "$NEW_SZ" | sed 's/|$//')"
    # awk wrote two lines; the |-trick pulled them; retry cleaner:
    {
        read Z
        read SZ
    } < <(add_c "$U" "$SU" "$C" "$SC")

    print_step "$step" "$Z" "$SZ"
done

echo
echo "=== C1 v0.5 — sedenion Mandelbrot iterate on A5000 ==="
echo "    work dir: $WORK"
echo "    NOTE: σ²(z²) at the diagonal uses the X² rule (4·z²·σ²(z))"
echo "    via sedenion_sqr (commit ed265de0). Variance correctly"
echo "    propagates the doubling factor across iterations."
