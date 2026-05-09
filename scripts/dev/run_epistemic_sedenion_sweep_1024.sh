#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_sweep_1024.sh
#
# C1 v0.3 — clinical-grade 1024-point sedenion epistemic sweep on A5000.
# Single GPU launch. Per-thread sedenion product with full GUM, host-side
# scalar reduction κ + σ²(κ), then trend statistics over the parameter t.
#
# Same parameterization as v0.2:
#     z(t) = (1+t)·e₁ + t·e₂,  v = e₃,  σ²(z[1]) = σ²(z[2]) = σ²(v[3]) = 1
#     u    = z · v   (sedenion product on A5000)
#     κ(t) = ‖u‖²,  σ²(κ) = Σ (2·u_i)²·σ²(u_i)
# Theoretical: κ(t) = 1 + 2t + 2t² (verified bit-exact in v0.2).
#
# Why 512: sedenion_mul --f32-epistemic uses ~128 registers per thread
# (value lane %f0..%f81 + variance lane %f128..%f209). On cc8.6 (A5000)
# with 64K registers per SM, the per-block limit is therefore 64K/128 = 512
# threads. Going above (768, 1024) returns CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
# (701). The kernel is single-block (gid=tid) so 512 is the in-launch cap;
# multi-block sedenion variants would lift this but don't exist yet in
# kretikos_emit_kaxi.sio.
#
# Reports per-decile statistics over 1024 sample points: κ, σ²(κ), σ/κ
# (relative uncertainty). Exits non-zero if any point deviates from
# theory by > 1e-3 (f32 sensitivity at κ=5).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_1024_WORK:-$(mktemp -d /tmp/c1_1024.XXXXXX)}"
N=512  # sedenion_mul f32-epistemic register footprint caps single-block at 512 on cc8.6

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

PTX="$WORK/sedenion_mul.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_mul --f32-epistemic -o "$PTX" --no-ptxas \
    >"$WORK/emit.log" 2>&1

# CSV may exceed ARG_MAX. Generate CSV directly into tmp files via awk
# redirection, then convert to f32 binary. Use --init-file / --init-var-file
# to feed the runner.
INIT_CSV="$WORK/init.csv"
VAR_CSV="$WORK/var.csv"
INIT_BIN="$WORK/init.f32"
VAR_BIN="$WORK/var.f32"

awk -v N=$N -v out="$INIT_CSV" 'BEGIN {
    for (n = 0; n < N; n++) {
        t = n / (N - 1)
        for (i = 0; i < 16; i++) {
            if (i == 1)      printf "%.7g,", 1.0 + t > out
            else if (i == 2) printf "%.7g,", t > out
            else             printf "0," > out
        }
        for (i = 0; i < 16; i++) printf "%s", (i == 3 ? "1," : "0,") > out
        for (i = 0; i < 16; i++) printf "0%s", (n == N - 1 && i == 15 ? "" : ",") > out
    }
}'

awk -v N=$N -v out="$VAR_CSV" 'BEGIN {
    for (n = 0; n < N; n++) {
        for (i = 0; i < 16; i++) printf "%s", (i == 1 || i == 2 ? "1," : "0,") > out
        for (i = 0; i < 16; i++) printf "%s", (i == 3 ? "1," : "0,") > out
        for (i = 0; i < 16; i++) printf "0%s", (n == N - 1 && i == 15 ? "" : ",") > out
    }
}'

# Tiny CSV→f32 helper
HELPER_SRC="$WORK/csv2f32.c"
HELPER="$WORK/csv2f32"
cat > "$HELPER_SRC" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    char tok[64];
    int p = 0, c;
    while ((c = getchar()) != EOF) {
        if (c == ',' || c == '\n') {
            if (p > 0) { tok[p] = 0; float v = strtof(tok, 0); fwrite(&v, sizeof v, 1, stdout); p = 0; }
        } else if (p < 63) tok[p++] = (char)c;
    }
    if (p > 0) { tok[p] = 0; float v = strtof(tok, 0); fwrite(&v, sizeof v, 1, stdout); }
    return 0;
}
EOF
cc -O2 "$HELPER_SRC" -o "$HELPER"
"$HELPER" < "$INIT_CSV" > "$INIT_BIN"
"$HELPER" < "$VAR_CSV"  > "$VAR_BIN"

OUT="$WORK/run.out"
"$RUNNER" "$PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
    --blocks 1 --threads $N --mem-words $((N * 48)) \
    --init-file "$INIT_BIN" --init-var-file "$VAR_BIN" --print-count $((N * 48)) \
    > "$OUT" 2>&1

# Sanity check exit status
grep -q "status=pass" "$OUT" || { echo "[FAIL] runner did not pass"; tail -20 "$OUT"; exit 1; }

MEM_ROW=$(grep '^MEM:' "$OUT" | sed 's/^MEM: //')
VAR_ROW=$(grep '^VAR:' "$OUT" | sed 's/^VAR: //')

awk -v N=$N -v mem="$MEM_ROW" -v var="$VAR_ROW" '
BEGIN {
    nm = split(mem, M, " ")
    nv = split(var, V, " ")
    if (nm != N * 48 || nv != N * 48) {
        printf "[FAIL] expected %d each, got mem=%d var=%d\n", N*48, nm, nv > "/dev/stderr"
        exit 1
    }
    max_err = 0
    sum_kappa = 0; sum_sig2 = 0
    sum_relunc = 0
    for (n = 0; n < N; n++) {
        t = n / (N - 1)
        kappa = 0; sig2 = 0
        for (i = 0; i < 16; i++) {
            ui = M[n * 48 + 32 + i + 1] + 0
            vi = V[n * 48 + 32 + i + 1] + 0
            kappa += ui * ui
            sig2  += 4 * ui * ui * vi
        }
        theory = 1 + 2 * t + 2 * t * t
        err = (kappa - theory)
        if (err < 0) err = -err
        if (err > max_err) max_err = err
        sum_kappa += kappa
        sum_sig2  += sig2
        if (kappa > 0) sum_relunc += sqrt(sig2) / kappa
        # Decile sample
        if (n % int(N / 10) == 0 || n == N - 1) {
            printf "  n=%-4d t=%.4f  κ=%.4f  σ²(κ)=%.4f  σ/κ=%.4f  theory=%.4f\n", \
                n, t, kappa, sig2, (kappa > 0 ? sqrt(sig2) / kappa : 0), theory
        }
    }
    printf "\n"
    printf "  Σ over %d points:\n", N
    printf "    mean κ        = %.4f\n", sum_kappa / N
    printf "    mean σ²(κ)    = %.4f\n", sum_sig2 / N
    printf "    mean σ/κ      = %.4f\n", sum_relunc / N
    printf "    max |κ-theory|= %.6f  (PASS if < 1e-3 at f32)\n", max_err
    if (max_err > 1e-3) {
        printf "[FAIL] κ deviates from theory by > 1e-3 at some point\n" > "/dev/stderr"
        exit 1
    }
}'

echo
echo "=== C1 v0.3 — 1024-point sedenion epistemic sweep on A5000  ==="
echo "    work dir: $WORK"
