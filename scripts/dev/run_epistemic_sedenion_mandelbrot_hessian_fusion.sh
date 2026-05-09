#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_mandelbrot_hessian_fusion.sh
#
# Phase D — sedenion Mandelbrot × Hessian fusion on K-AXI / A5000.
#
# Extends the C1 v0.6 Mandelbrot grid sweep with Hessian curvature probes
# at gate iterations k ∈ {1, ITERS/2, ITERS}.  For each gate step and each
# Mandelbrot thread, runs 4 launches of sedenion_sqr_mb (256 threads each,
# one per (i,j) pair) and reduces the full 16×16 Hessian of f(z)=||z²||².
#
# Output: artifacts/mandelbrot_hessian_fusion.csv
# Columns: thread, step, c5, kappa, sigma_kappa, trace, sigma_trace, anisotropy, escape
#
# Finding under test: does anisotropy spike at the escape moment?
# Also: does ZD-aligned c (e₅ direction) produce a different escape-curvature
# signature, and does the κ(c)=κ(−c) symmetry hold in curvature?
#
# Default: N=16 Mandelbrot threads, ITERS=8, h=0.001.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_FUSION_WORK:-$(mktemp -d /tmp/c1_fusion.XXXXXX)}"
N="${SOUNIO_C1_FUSION_THREADS:-16}"
ITERS="${SOUNIO_C1_FUSION_ITERS:-8}"
HESSIAN_H="${SOUNIO_HESSIAN_DELTA:-0.001}"
CSV_FILE="${SOUNIO_C1_FUSION_CSV:-$(pwd)/artifacts/mandelbrot_hessian_fusion.csv}"

mkdir -p "$(dirname "$CSV_FILE")"

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

# Emit PTX for Mandelbrot iterations (single-block sedenion_sqr)
PTX="$WORK/sedenion_sqr.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_sqr --f32-epistemic -o "$PTX" --no-ptxas \
    >"$WORK/emit.log" 2>&1

# Emit PTX for Hessian probes (multi-block sedenion_sqr_mb)
MB_PTX="$WORK/sedenion_sqr_mb.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_sqr_mb --f32-epistemic -o "$MB_PTX" \
    >>"$WORK/emit.log" 2>&1

echo "mandelbrot_hessian_fusion: PTX emitted  N=$N  ITERS=$ITERS  h=$HESSIAN_H"
echo "mandelbrot_hessian_fusion: CSV -> $CSV_FILE"

# Compute gate steps where we probe Hessian.
GATE_STEPS=""
GATE_STEPS="${GATE_STEPS} 1"
HALF=$((ITERS / 2))
[ "$HALF" -gt 1 ] && GATE_STEPS="${GATE_STEPS} $HALF"
[ "$ITERS" -gt 1 ] && GATE_STEPS="${GATE_STEPS} $ITERS"

# ---- Initial state (same as v0.6 default grid) ----------------------------
C_LIST=$(awk -v N=$N 'BEGIN {
    for (n = 0; n < N; n++) {
        a = 0.05 * (n - (N - 1) / 2.0) / ((N - 1) / 2.0)
        for (i = 0; i < 16; i++) {
            v = (i == 5) ? a : 0
            printf "%s%s", v, (i < 15 ? " " : "")
        }
        printf "%s", (n < N - 1 ? ";" : "")
    }
    printf "\n"
}')

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

# Write CSV header.
echo "thread,step,c5,kappa,sigma_kappa,trace,sigma_trace,anisotropy,escape" > "$CSV_FILE"

# ---- Hessian probe function -------------------------------------------------
# For one Mandelbrot thread n with z given as "f0 f1 ... f15", run 4 stencil
# launches (256 threads each) and return: trace sigma_trace anisotropy
#
# Called as: run_hessian_one_thread <step> <z_space_sep> <c5> <kappa> <sigma_kappa>
run_hessian_one_thread() {
    local step="$1"
    local z_space="$2"   # "f0 f1 ... f15"
    local c5="$3"
    local kappa="$4"
    local sigma_kappa="$5"
    local thread_n="$6"
    local escape="$7"

    local hwork="$WORK/hessian_s${step}_t${thread_n}"
    mkdir -p "$hwork"

    local Z_CSV
    Z_CSV=$(echo "$z_space" | tr ' ' ',')

    # Build one 256-thread stencil memory block.
    build_stencil_one() {
        local si="$1"; local sj="$2"
        awk -v Z="$Z_CSV" -v H="$HESSIAN_H" -v si="$si" -v sj="$sj" 'BEGIN {
            n = split(Z, z, ",")
            for (k = 0; k < n; k++) zv[k] = z[k+1]+0
            out = ""
            for (pair = 0; pair < 256; pair++) {
                ii = int(pair / 16); jj = pair % 16
                for (k = 0; k < 16; k++) {
                    v = zv[k]
                    if (k == ii) v += si * H
                    if (k == jj) v += sj * H
                    sep = (out == "") ? "" : ","
                    out = out sep v
                }
                for (k = 0; k < 16; k++) { out = out ",0" }
            }
            print out
        }'
    }

    local VARINIT
    VARINIT=$(python3 -c "print(','.join(['0']*(256*32)))")

    run_stencil_one() {
        local mem="$1"; local outf="$2"
        "$RUNNER" "$MB_PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
            --blocks 256 --threads 1 --mem-words $((256*32)) \
            --init-mem "$mem" --init-var "$VARINIT" --print-count $((256*32)) \
            >"$outf" 2>&1
        grep -q "status=pass" "$outf" || {
            echo "mandelbrot_hessian_fusion: FAIL hessian s${step}_t${thread_n} $(basename $outf)" >&2
            exit 1
        }
    }

    run_stencil_one "$(build_stencil_one  1  1)" "$hwork/pp.txt"
    run_stencil_one "$(build_stencil_one  1 -1)" "$hwork/pm.txt"
    run_stencil_one "$(build_stencil_one -1  1)" "$hwork/mp.txt"
    run_stencil_one "$(build_stencil_one -1 -1)" "$hwork/mm.txt"

    python3 - "$hwork/pp.txt" "$hwork/pm.txt" "$hwork/mp.txt" "$hwork/mm.txt" \
        "$HESSIAN_H" "$step" "$thread_n" "$c5" "$kappa" "$sigma_kappa" "$escape" \
        "$CSV_FILE" <<'PYEOF'
import sys, math

def parse_mem(path):
    with open(path) as f:
        for line in f:
            if line.startswith("MEM: "):
                return [float(x) for x in line[5:].split()]

def norm_sq_thread(mem, tid):
    base = tid * 32 + 16
    return sum(mem[base + k]**2 for k in range(16))

fpp=parse_mem(sys.argv[1]); fpm=parse_mem(sys.argv[2])
fmp=parse_mem(sys.argv[3]); fmm=parse_mem(sys.argv[4])
h=float(sys.argv[5]); step=int(sys.argv[6]); thread_n=int(sys.argv[7])
c5=float(sys.argv[8]); kappa=float(sys.argv[9]); sigma_kappa=float(sys.argv[10])
escape_str=sys.argv[11]; csv_file=sys.argv[12]
denom=4*h*h

H_mat=[[0.0]*16 for _ in range(16)]
for pair in range(256):
    i=pair//16; j=pair%16
    npp=norm_sq_thread(fpp,pair); npm=norm_sq_thread(fpm,pair)
    nmp=norm_sq_thread(fmp,pair); nmm=norm_sq_thread(fmm,pair)
    H_mat[i][j]=(npp-npm-nmp+nmm)/denom

trace=sum(H_mat[i][i] for i in range(16))
diag=[H_mat[i][i] for i in range(16)]
abs_diag=[abs(v) for v in diag if abs(v)>1e-12]
aniso=max(abs_diag)/min(abs_diag) if len(abs_diag)>=2 else 0.0

# sigma_trace: propagate input variance through the FD stencil.
# Each H[i][i] is a linear combination of 4 norm_sq values;
# variance of norm_sq = (2*z^T H z)^2 sigma^2 (first-order GUM).
# Here we use sigma_kappa as a proxy for sigma_trace (exact propagation
# requires per-thread input variance at each stencil point, deferred).
sigma_trace = 0.0  # placeholder; exact GUM propagation is deferred to Phase B2

with open(csv_file, "a") as f:
    f.write(f"{thread_n},{step},{c5:.6f},{kappa:.6f},{sigma_kappa:.6f},"
            f"{trace:.6f},{sigma_trace:.6f},{aniso:.6f},{escape_str}\n")
PYEOF
}

# ---- Utility: compute per-thread κ and escape from current Z_LIST -----------
kappa_and_escape() {
    awk -v N=$N -v zlist="$Z_LIST" -v szlist="$SZ_LIST" 'BEGIN {
        n_th = split(zlist, ZT, ";")
        split(szlist, SZT, ";")
        for (n = 0; n < N; n++) {
            split(ZT[n+1], Z, " "); split(SZT[n+1], SZ, " ")
            kappa=0; sig2=0
            for (i=1; i<=16; i++) { kappa+=Z[i]*Z[i]; sig2+=4*Z[i]*Z[i]*SZ[i] }
            escape=(kappa>1e6)?"YES":"no"
            printf "%d %.6f %.6f %s\n", n, kappa, sqrt(sig2), escape
        }
    }'
}

# ---- Build CSV / awk helper for iteration (reused from v0.6) ---------------
build_csv() {
    local zlist="$1"; local include_out_zeros="$2"
    awk -v zlist="$zlist" -v zeros="$include_out_zeros" -v N=$N '
    BEGIN {
        n_th = split(zlist, T, ";")
        for (k=1; k<=n_th; k++) {
            n_v = split(T[k], V, " ")
            for (i=1; i<=n_v; i++) printf "%s,", V[i]
            if (zeros == "1") {
                for (i=0; i<16; i++) printf "0%s", (k==n_th && i==15 ? "" : ",")
            }
        }
    }'
}

# ---- Main loop -------------------------------------------------------------
for step in $(seq 1 "$ITERS"); do
    INIT_MEM=$(build_csv "$Z_LIST" 1)
    INIT_VAR=$(build_csv "$SZ_LIST" 1)

    OUT="$WORK/step${step}.out"
    "$RUNNER" "$PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
        --blocks 1 --threads $N --mem-words $((N * 32)) \
        --init-mem "$INIT_MEM" --init-var "$INIT_VAR" --print-count $((N * 32)) \
        > "$OUT" 2>&1

    grep -q "status=pass" "$OUT" || {
        echo "mandelbrot_hessian_fusion: [FAIL] Mandelbrot step $step"
        tail -10 "$OUT"; exit 1
    }

    MEM=$(grep '^MEM:' "$OUT" | sed 's/^MEM: //')
    VAR=$(grep '^VAR:' "$OUT" | sed 's/^VAR: //')

    NEXT=$(awk -v N=$N -v mem="$MEM" -v var="$VAR" -v clist="$C_LIST" '
    BEGIN {
        split(mem, M, " "); split(var, V, " ")
        n_th = split(clist, C, ";")
        z_out=""; sz_out=""
        for (n=0; n<N; n++) {
            n_v = split(C[n+1], CN, " ")
            for (i=0; i<16; i++) {
                u = M[n*32+16+i+1]+0; sv = V[n*32+16+i+1]+0; cn = CN[i+1]+0
                z_out = z_out (u+cn) (i<15?" ":"")
                sz_out = sz_out sv (i<15?" ":"")
            }
            z_out  = z_out  (n<N-1?";":"")
            sz_out = sz_out (n<N-1?";":"")
        }
        print z_out; print sz_out
    }')
    { IFS= read -r Z_LIST; IFS= read -r SZ_LIST; } <<<"$NEXT"

    # ---- Hessian probe gate ------------------------------------------------
    is_gate=0
    for g in $GATE_STEPS; do
        [ "$step" -eq "$g" ] && is_gate=1
    done

    if [ "$is_gate" -eq 1 ]; then
        echo "mandelbrot_hessian_fusion: Hessian gate at step=$step ..."
        # Compute per-thread κ and escape for this step
        readarray -t KAPPA_ROWS < <(kappa_and_escape)
        # Extract per-thread z tuples from Z_LIST
        IFS=';' read -ra Z_THREADS <<< "$Z_LIST"
        for n in $(seq 0 $((N - 1))); do
            z_space="${Z_THREADS[$n]}"
            row="${KAPPA_ROWS[$n]}"
            kappa=$(echo "$row" | awk '{print $2}')
            sigma_kappa=$(echo "$row" | awk '{print $3}')
            escape=$(echo "$row" | awk '{print $4}')
            c5=$(awk -v clist="$C_LIST" -v n=$n 'BEGIN {
                split(clist, T, ";"); split(T[n+1], V, " "); print V[6]+0
            }')
            run_hessian_one_thread "$step" "$z_space" "$c5" "$kappa" "$sigma_kappa" "$n" "$escape"
        done
        echo "mandelbrot_hessian_fusion: Hessian gate step=$step done (N=$N threads)"
    fi
done

echo
echo "=== Mandelbrot × Hessian Fusion CSV ==="
wc -l "$CSV_FILE"
echo "--- head ---"
head -8 "$CSV_FILE"
echo "--- curvature-blowup hypothesis (anisotropy at escape vs non-escape) ---"
python3 - "$CSV_FILE" <<'PYEOF'
import sys, csv, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
escape_aniso   = [float(r['anisotropy']) for r in rows if r['escape']=='YES' and float(r['anisotropy'])>0]
nonescape_aniso= [float(r['anisotropy']) for r in rows if r['escape']=='no'  and float(r['anisotropy'])>0]
def stats(vals, label):
    if not vals: print(f"  {label}: no data"); return
    print(f"  {label}: n={len(vals)}  mean={statistics.mean(vals):.4f}  "
          f"median={statistics.median(vals):.4f}  "
          f"max={max(vals):.4f}  min={min(vals):.4f}")
stats(escape_aniso,    "escape  anisotropy")
stats(nonescape_aniso, "non-esc anisotropy")
# Symmetry check: κ(c) = κ(−c)?  Group by |c5|, compare escape flags.
by_abs_c5 = {}
for r in rows:
    key = f"{abs(float(r['c5'])):.5f}"
    by_abs_c5.setdefault(key, []).append(r)
asym = [(k, vs) for k, vs in by_abs_c5.items() if len(vs) >= 2]
print(f"\n  c5 symmetry pairs checked: {len(asym)}")
for k, vs in sorted(asym)[:4]:
    escapes = [v['escape'] for v in vs]
    kappas  = [float(v['kappa']) for v in vs]
    print(f"    |c5|={k}  kappas={[f'{k:.2g}' for k in kappas]}  escape={escapes}")
PYEOF

echo
echo "=== C1 Phase D done ==="
echo "    work dir:  $WORK"
echo "    CSV:       $CSV_FILE"
