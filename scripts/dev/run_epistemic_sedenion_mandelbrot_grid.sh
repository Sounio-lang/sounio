#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_mandelbrot_grid.sh
#
# C1 v0.6 — parallel sedenion Mandelbrot grid sweep on A5000.
#
# Extends v0.5 from one orbit to N parallel orbits in a single sedenion_sqr
# launch per iteration. Each thread carries its own (z_n, σ²(z_n), c_n)
# tuple and iterates z := z² + c independently. The host orchestrates the
# `+ c` step per thread between launches.
#
# Default grid (N=64 threads):
#   z₀ = e₁ + e₂   (uniform across all threads, σ²(z₀[1]) = σ²(z₀[2]) = 0.01)
#   c_n = a_n · e₅, with a_n = 0.05 · (n − 31.5) / 31.5  for n = 0..63
#         → c_n ranges from −0.05·e₅ at n=0 to +0.05·e₅ at n=63
#
# After ITERS steps, reports per-thread (κ, σ(κ), σ/κ, escape?) so the
# user can see how varying c affects orbit divergence at uniform z₀ +
# uniform initial σ². "Escape" flagged when κ > 1e6.
#
# Why this matters: first parallel sedenion Mandelbrot grid on owned
# silicon with full GUM tracked across iterations. Phase precursor to
# the eventual N=10⁵+ paper-grade sweep (which needs multi-block
# sedenion_sqr — currently single-block-capped).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_GRID_WORK:-$(mktemp -d /tmp/c1_grid.XXXXXX)}"
N="${SOUNIO_C1_GRID_THREADS:-64}"
ITERS="${SOUNIO_C1_GRID_ITERS:-4}"

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

PTX="$WORK/sedenion_sqr.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_sqr --f32-epistemic -o "$PTX" --no-ptxas \
    >"$WORK/emit.log" 2>&1

# Generate initial Z, σ²(Z), per-thread C arrays as awk-driven CSVs.
# Layout per thread: 16 (z) + 16 (out) = 32 mem-words per thread.

INIT_MEM_CSV=$(awk -v N=$N 'BEGIN {
    for (n = 0; n < N; n++) {
        # z_0 uniform = e₁ + e₂
        for (i = 0; i < 16; i++) {
            v = (i == 1 || i == 2) ? 1 : 0
            printf "%s,", v
        }
        # output region (init zeros)
        for (i = 0; i < 16; i++) printf "0%s", (n == N - 1 && i == 15 ? "" : ",")
    }
}')

INIT_VAR_CSV=$(awk -v N=$N 'BEGIN {
    for (n = 0; n < N; n++) {
        # σ²(z_0) = 0.01 on positions 1, 2; 0 elsewhere
        for (i = 0; i < 16; i++) {
            v = (i == 1 || i == 2) ? 0.01 : 0
            printf "%s,", v
        }
        for (i = 0; i < 16; i++) printf "0%s", (n == N - 1 && i == 15 ? "" : ",")
    }
}')

# Per-thread c_n vector (only component 5 varies across threads).
C_LIST=$(awk -v N=$N 'BEGIN {
    for (n = 0; n < N; n++) {
        a = 0.05 * (n - (N - 1) / 2.0) / ((N - 1) / 2.0)
        # Print c_n as 16 space-sep floats, semicolon-separated across threads
        for (i = 0; i < 16; i++) {
            v = (i == 5) ? a : 0
            printf "%s%s", v, (i < 15 ? " " : "")
        }
        printf "%s", (n < N - 1 ? ";" : "")
    }
    printf "\n"
}')

# Initial Z and SZ as space-separated 16-tuples per thread; semicolons across
Z_LIST=$(awk -v N=$N 'BEGIN {
    for (n = 0; n < N; n++) {
        for (i = 0; i < 16; i++) {
            v = (i == 1 || i == 2) ? 1 : 0
            printf "%s%s", v, (i < 15 ? " " : "")
        }
        printf "%s", (n < N - 1 ? ";" : "")
    }
}')

SZ_LIST=$(awk -v N=$N 'BEGIN {
    for (n = 0; n < N; n++) {
        for (i = 0; i < 16; i++) {
            v = (i == 1 || i == 2) ? 0.01 : 0
            printf "%s%s", v, (i < 15 ? " " : "")
        }
        printf "%s", (n < N - 1 ? ";" : "")
    }
}')

# Build init-mem CSV from Z_LIST + zeros, init-var from SZ_LIST + zeros
build_csv() {
    local zlist="$1"
    local include_out_zeros="$2"
    awk -v zlist="$zlist" -v zeros="$include_out_zeros" -v N=$N '
    BEGIN {
        n_th = split(zlist, T, ";")
        for (k = 1; k <= n_th; k++) {
            n_v = split(T[k], V, " ")
            for (i = 1; i <= n_v; i++) printf "%s,", V[i]
            if (zeros == "1") {
                for (i = 0; i < 16; i++) printf "0%s", (k == n_th && i == 15 ? "" : ",")
            }
        }
    }'
}

# Run iterations
for step in $(seq 1 "$ITERS"); do
    INIT_MEM=$(build_csv "$Z_LIST" 1)
    INIT_VAR=$(build_csv "$SZ_LIST" 1)

    OUT="$WORK/step${step}.out"
    "$RUNNER" "$PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
        --blocks 1 --threads $N --mem-words $((N * 32)) \
        --init-mem "$INIT_MEM" --init-var "$INIT_VAR" --print-count $((N * 32)) \
        > "$OUT" 2>&1

    grep -q "status=pass" "$OUT" || {
        echo "[FAIL] step $step: GPU launch failed"
        tail -10 "$OUT"; exit 1
    }

    # Parse per-thread output u_n and σ²(u_n), then add c_n to get next z, σ²(z).
    MEM=$(grep '^MEM:' "$OUT" | sed 's/^MEM: //')
    VAR=$(grep '^VAR:' "$OUT" | sed 's/^VAR: //')

    NEXT=$(awk -v N=$N -v mem="$MEM" -v var="$VAR" -v clist="$C_LIST" '
    BEGIN {
        nm = split(mem, M, " "); split(var, V, " ")
        n_th = split(clist, C, ";")
        # Build new Z_LIST and SZ_LIST as semicolon-separated 16-tuples.
        z_out = ""; sz_out = ""
        for (n = 0; n < N; n++) {
            n_v = split(C[n + 1], CN, " ")
            for (i = 0; i < 16; i++) {
                # output u_n[i] is at MEM[n*32 + 16 + i + 1]
                u = M[n * 32 + 16 + i + 1] + 0
                sv = V[n * 32 + 16 + i + 1] + 0
                cn = CN[i + 1] + 0
                # z_{n+1}[i] = u_n[i] + c_n[i],  σ²(z_{n+1}[i]) = σ²(u_n[i])
                z_out = z_out (u + cn) (i < 15 ? " " : "")
                sz_out = sz_out sv (i < 15 ? " " : "")
            }
            z_out = z_out (n < N - 1 ? ";" : "")
            sz_out = sz_out (n < N - 1 ? ";" : "")
        }
        # Print z_out then sz_out separated by newline
        print z_out
        print sz_out
    }')
    {
        IFS= read -r Z_LIST
        IFS= read -r SZ_LIST
    } <<<"$NEXT"
done

# Final: per-thread κ = ||z||², σ²(κ), σ/κ, escape flag.
echo "  thread  c_5            κ                σ²(κ)            σ/κ          escape"
echo "  ------  -----------    --------------   --------------   ----------   ------"
awk -v N=$N -v zlist="$Z_LIST" -v szlist="$SZ_LIST" -v clist="$C_LIST" '
BEGIN {
    n_th = split(zlist, ZT, ";")
    split(szlist, SZT, ";")
    split(clist, CT, ";")
    n_escape = 0
    for (n = 0; n < N; n++) {
        nz = split(ZT[n + 1], Z, " ")
        split(SZT[n + 1], SZ, " ")
        split(CT[n + 1], C, " ")
        kappa = 0; sig2 = 0
        for (i = 0; i < 16; i++) {
            kappa += Z[i + 1] * Z[i + 1]
            sig2  += 4 * Z[i + 1] * Z[i + 1] * SZ[i + 1]
        }
        sigma = sqrt(sig2)
        rel = (kappa > 0) ? sigma / kappa : 0
        c5 = C[6] + 0
        escape = (kappa > 1e6) ? "YES" : "no"
        if (escape == "YES") n_escape++
        if (n % 8 == 0 || n == N - 1) {
            printf "  %-6d  %+11.5f    %14.4g   %14.4g   %10.4f   %s\n", n, c5, kappa, sig2, rel, escape
        }
    }
    printf "\n  Escape count: %d / %d threads (after %s iterations)\n", n_escape, N, "'"$ITERS"'"
}'

echo
echo "=== C1 v0.6 — parallel sedenion Mandelbrot grid on A5000 ==="
echo "    work dir: $WORK"
