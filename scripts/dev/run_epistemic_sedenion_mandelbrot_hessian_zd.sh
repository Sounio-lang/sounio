#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_mandelbrot_hessian_zd.sh
#
# Phase D — ZD-aligned c experiment.
#
# Identical to run_epistemic_sedenion_mandelbrot_hessian_fusion.sh except
# c_n varies along e₁₀ (a zero-divisor direction: e₁·e₁₀ = 0 in sedenions)
# instead of e₅ (a non-ZD direction). Compares:
#
#   e₅ run:  c_n = a_n · e₅   (no ZD coupling to z₀ = e₁+e₂)
#   e₁₀ run: c_n = a_n · e₁₀  (ZD coupling: e₁·e₁₀ = 0)
#
# Question: does ZD-aligned c produce a different escape-curvature signature?
# Does the anisotropy spike timing or magnitude shift when c lies in the ZD locus?
#
# Output: artifacts/mandelbrot_hessian_fusion_zd.csv
# Same column schema as Phase D CSV.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_ZD_WORK:-$(mktemp -d /tmp/c1_zd.XXXXXX)}"
N="${SOUNIO_C1_ZD_THREADS:-16}"
ITERS="${SOUNIO_C1_ZD_ITERS:-8}"
HESSIAN_H="${SOUNIO_HESSIAN_DELTA:-0.001}"
CSV_FILE="${SOUNIO_C1_ZD_CSV:-$(pwd)/artifacts/mandelbrot_hessian_fusion_zd.csv}"
# Which component of c to vary (10 = e₁₀, ZD direction; 5 = e₅, non-ZD)
C_COMP="${SOUNIO_C1_ZD_COMP:-10}"

mkdir -p "$(dirname "$CSV_FILE")"

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

PTX="$WORK/sedenion_sqr.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_sqr --f32-epistemic -o "$PTX" --no-ptxas \
    >"$WORK/emit.log" 2>&1

MB_PTX="$WORK/sedenion_sqr_mb.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_sqr_mb --f32-epistemic -o "$MB_PTX" \
    >>"$WORK/emit.log" 2>&1

echo "mandelbrot_hessian_zd: PTX emitted  N=$N  ITERS=$ITERS  c_comp=e${C_COMP}  h=$HESSIAN_H"
echo "mandelbrot_hessian_zd: CSV -> $CSV_FILE"

GATE_STEPS=""
GATE_STEPS="${GATE_STEPS} 1"
HALF=$((ITERS / 2))
[ "$HALF" -gt 1 ] && GATE_STEPS="${GATE_STEPS} $HALF"
[ "$ITERS" -gt 1 ] && GATE_STEPS="${GATE_STEPS} $ITERS"

# c_n = a_n · e_{C_COMP}
C_LIST=$(awk -v N=$N -v comp=$C_COMP 'BEGIN {
    for (n = 0; n < N; n++) {
        a = 0.05 * (n - (N - 1) / 2.0) / ((N - 1) / 2.0)
        for (i = 0; i < 16; i++) {
            v = (i == comp) ? a : 0
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

echo "thread,step,c${C_COMP},kappa,sigma_kappa,trace,sigma_trace,anisotropy,escape" > "$CSV_FILE"

kappa_and_escape() {
    awk -v N=$N -v zlist="$Z_LIST" -v szlist="$SZ_LIST" 'BEGIN {
        n_th = split(zlist, ZT, ";"); split(szlist, SZT, ";")
        for (n = 0; n < N; n++) {
            split(ZT[n+1], Z, " "); split(SZT[n+1], SZ, " ")
            kappa=0; sig2=0
            for (i=1; i<=16; i++) { kappa+=Z[i]*Z[i]; sig2+=4*Z[i]*Z[i]*SZ[i] }
            printf "%d %.6f %.6f %s\n", n, kappa, sqrt(sig2), (kappa>1e6)?"YES":"no"
        }
    }'
}

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

run_hessian_one_thread() {
    local step="$1" z_space="$2" c_val="$3" kappa="$4" sigma_kappa="$5" thread_n="$6" escape="$7"
    local hwork="$WORK/hessian_s${step}_t${thread_n}"
    mkdir -p "$hwork"
    local Z_CSV; Z_CSV=$(echo "$z_space" | tr ' ' ',')
    build_stencil_one() {
        local si="$1" sj="$2"
        awk -v Z="$Z_CSV" -v H="$HESSIAN_H" -v si="$si" -v sj="$sj" 'BEGIN {
            n = split(Z, z, ",")
            for (k=0; k<n; k++) zv[k]=z[k+1]+0
            out=""
            for (pair=0; pair<256; pair++) {
                ii=int(pair/16); jj=pair%16
                for (k=0; k<16; k++) {
                    v=zv[k]; if(k==ii) v+=si*H; if(k==jj) v+=sj*H
                    sep=(out=="")?"":","; out=out sep v
                }
                for (k=0; k<16; k++) { out=out",0" }
            }
            print out
        }'
    }
    local VARINIT; VARINIT=$(python3 -c "print(','.join(['0']*(256*32)))")
    run_stencil_one() {
        local mem="$1" outf="$2"
        "$RUNNER" "$MB_PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
            --blocks 256 --threads 1 --mem-words $((256*32)) \
            --init-mem "$mem" --init-var "$VARINIT" --print-count $((256*32)) \
            >"$outf" 2>&1
        grep -q "status=pass" "$outf" || { echo "[FAIL] hessian s${step}_t${thread_n} $(basename $outf)" >&2; exit 1; }
    }
    run_stencil_one "$(build_stencil_one  1  1)" "$hwork/pp.txt"
    run_stencil_one "$(build_stencil_one  1 -1)" "$hwork/pm.txt"
    run_stencil_one "$(build_stencil_one -1  1)" "$hwork/mp.txt"
    run_stencil_one "$(build_stencil_one -1 -1)" "$hwork/mm.txt"
    python3 - "$hwork/pp.txt" "$hwork/pm.txt" "$hwork/mp.txt" "$hwork/mm.txt" \
        "$HESSIAN_H" "$step" "$thread_n" "$c_val" "$kappa" "$sigma_kappa" "$escape" \
        "$CSV_FILE" "$C_COMP" <<'PYEOF'
import sys, math
def parse_mem(path):
    with open(path) as f:
        for line in f:
            if line.startswith("MEM: "): return [float(x) for x in line[5:].split()]
def norm_sq_thread(mem, tid):
    base = tid * 32 + 16; return sum(mem[base+k]**2 for k in range(16))
fpp=parse_mem(sys.argv[1]); fpm=parse_mem(sys.argv[2])
fmp=parse_mem(sys.argv[3]); fmm=parse_mem(sys.argv[4])
h=float(sys.argv[5]); step=int(sys.argv[6]); thread_n=int(sys.argv[7])
c_val=float(sys.argv[8]); kappa=float(sys.argv[9]); sigma_kappa=float(sys.argv[10])
escape_str=sys.argv[11]; csv_file=sys.argv[12]; c_comp=int(sys.argv[13])
denom=4*h*h
H_mat=[[0.0]*16 for _ in range(16)]
for pair in range(256):
    i=pair//16; j=pair%16
    npp=norm_sq_thread(fpp,pair); npm=norm_sq_thread(fpm,pair)
    nmp=norm_sq_thread(fmp,pair); nmm=norm_sq_thread(fmm,pair)
    H_mat[i][j]=(npp-npm-nmp+nmm)/denom
trace=sum(H_mat[i][i] for i in range(16))
abs_diag=[abs(H_mat[i][i]) for i in range(16) if abs(H_mat[i][i])>1e-12]
aniso=max(abs_diag)/min(abs_diag) if len(abs_diag)>=2 else 0.0
with open(csv_file,"a") as f:
    f.write(f"{thread_n},{step},{c_val:.6f},{kappa:.6f},{sigma_kappa:.6f},"
            f"{trace:.6f},0.000000,{aniso:.6f},{escape_str}\n")
PYEOF
}

for step in $(seq 1 "$ITERS"); do
    INIT_MEM=$(build_csv "$Z_LIST" 1)
    INIT_VAR=$(build_csv "$SZ_LIST" 1)
    OUT="$WORK/step${step}.out"
    "$RUNNER" "$PTX" --kernel kaxi_kernel --mode epistemic --type f32 \
        --blocks 1 --threads $N --mem-words $((N * 32)) \
        --init-mem "$INIT_MEM" --init-var "$INIT_VAR" --print-count $((N * 32)) \
        > "$OUT" 2>&1
    grep -q "status=pass" "$OUT" || { echo "[FAIL] Mandelbrot step $step"; tail -5 "$OUT"; exit 1; }
    MEM=$(grep '^MEM:' "$OUT" | sed 's/^MEM: //'); VAR=$(grep '^VAR:' "$OUT" | sed 's/^VAR: //')
    NEXT=$(awk -v N=$N -v mem="$MEM" -v var="$VAR" -v clist="$C_LIST" '
    BEGIN {
        split(mem,M," "); split(var,V," "); n_th=split(clist,C,";")
        z_out=""; sz_out=""
        for (n=0;n<N;n++) {
            n_v=split(C[n+1],CN," ")
            for (i=0;i<16;i++) {
                u=M[n*32+16+i+1]+0; sv=V[n*32+16+i+1]+0; cn=CN[i+1]+0
                z_out=z_out (u+cn) (i<15?" ":""); sz_out=sz_out sv (i<15?" ":"")
            }
            z_out=z_out (n<N-1?";":""); sz_out=sz_out (n<N-1?";":"")
        }
        print z_out; print sz_out
    }')
    { IFS= read -r Z_LIST; IFS= read -r SZ_LIST; } <<<"$NEXT"

    is_gate=0
    for g in $GATE_STEPS; do [ "$step" -eq "$g" ] && is_gate=1; done
    if [ "$is_gate" -eq 1 ]; then
        echo "mandelbrot_hessian_zd: Hessian gate at step=$step ..."
        readarray -t KAPPA_ROWS < <(kappa_and_escape)
        IFS=';' read -ra Z_THREADS <<< "$Z_LIST"
        for n in $(seq 0 $((N - 1))); do
            z_space="${Z_THREADS[$n]}"
            row="${KAPPA_ROWS[$n]}"
            kappa=$(echo "$row" | awk '{print $2}')
            sigma_kappa=$(echo "$row" | awk '{print $3}')
            escape=$(echo "$row" | awk '{print $4}')
            c_val=$(awk -v clist="$C_LIST" -v n=$n -v comp=$C_COMP 'BEGIN {
                split(clist,T,";"); split(T[n+1],V," "); print V[comp+1]+0
            }')
            run_hessian_one_thread "$step" "$z_space" "$c_val" "$kappa" "$sigma_kappa" "$n" "$escape"
        done
        echo "mandelbrot_hessian_zd: gate step=$step done"
    fi
done

echo
echo "=== ZD experiment (c along e${C_COMP}) vs Phase D (c along e5) ==="
python3 - "$CSV_FILE" "$C_COMP" <<'PYEOF'
import sys, csv, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
comp = sys.argv[2]
by_step = {}
for r in rows:
    if r['trace'] != 'nan' and float(r['anisotropy']) > 0:
        s = int(r['step'])
        by_step.setdefault(s, []).append(float(r['anisotropy']))
print(f"c_comp = e{comp}")
for s, vals in sorted(by_step.items()):
    print(f"  step {s}: n={len(vals)}  mean_aniso={statistics.mean(vals):.4f}  "
          f"max={max(vals):.4f}  min={min(vals):.4f}")
# Symmetry check
by_abs_c = {}
for r in rows:
    key = f"{abs(float(r[f'c{comp}'])):.5f}"
    by_abs_c.setdefault(key, []).append(r)
asym = [(k, vs) for k, vs in by_abs_c.items() if len(vs) >= 2]
print(f"\nc{comp} symmetry pairs: {len(asym)}")
for k, vs in sorted(asym)[:3]:
    aniso = [float(v['anisotropy']) for v in vs if v['trace'] != 'nan']
    print(f"  |c{comp}|={k}  aniso={[f'{a:.2f}' for a in aniso]}")
PYEOF

echo
echo "=== ZD done ==="
echo "    CSV: $CSV_FILE"
echo "    work: $WORK"
