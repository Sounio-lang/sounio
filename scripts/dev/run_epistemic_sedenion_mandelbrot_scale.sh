#!/usr/bin/env bash
# scripts/dev/run_epistemic_sedenion_mandelbrot_scale.sh
#
# Paper-grade scale sweep: N=8192 Mandelbrot threads on A5000.
#
# Uses --init-file / --init-var-file (binary f32) to bypass ARG_MAX limits
# that prevent large N via --init-mem CSV strings.
#
# Mandelbrot kernel: sedenion_sqr_mb (multi-block, 32 blocks × 256 threads)
# Hessian probes: sedenion_sqr_mb per-thread at gate steps (subsample HESSIAN_N threads)
#
# Output:
#   artifacts/mandelbrot_scale_N${N}.csv  — per-thread escape statistics
#   artifacts/mandelbrot_hessian_scale.csv — Hessian on HESSIAN_N sampled threads
#
# Headline finding: escape-count distribution and anisotropy spike at N=8192.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNNER="${SOUNIO_KAXI_RUNNER:-/tmp/runner}"
WORK="${SOUNIO_C1_SCALE_WORK:-$(mktemp -d /tmp/c1_scale.XXXXXX)}"
N="${SOUNIO_C1_SCALE_N:-8192}"
ITERS="${SOUNIO_C1_SCALE_ITERS:-8}"
HESSIAN_N="${SOUNIO_C1_SCALE_HESSIAN_N:-32}"   # threads to probe with Hessian
HESSIAN_H="${SOUNIO_HESSIAN_DELTA:-0.001}"
THREADS_PER_BLOCK=256
BLOCKS=$((N / THREADS_PER_BLOCK))

mkdir -p artifacts

if [ ! -x "$RUNNER" ]; then
    cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -o "$RUNNER"
fi

MB_PTX="$WORK/sedenion_sqr_mb.ptx"
./bin/kretikos kaxi-emit-ptx sedenion_sqr_mb --f32-epistemic -o "$MB_PTX" \
    >"$WORK/emit.log" 2>&1

echo "mandelbrot_scale: PTX emitted  N=$N  ITERS=$ITERS  HESSIAN_N=$HESSIAN_N"
echo "mandelbrot_scale: $BLOCKS blocks × $THREADS_PER_BLOCK threads"
echo "mandelbrot_scale: work=$WORK"

MEM_WORDS=$((N * 32))
MEM_BYTES=$((MEM_WORDS * 4))   # f32

GATE_STEPS=""; HALF=$((ITERS / 2))
GATE_STEPS="$GATE_STEPS 1"
[ "$HALF" -gt 1 ] && GATE_STEPS="$GATE_STEPS $HALF"
[ "$ITERS" -gt 1 ] && GATE_STEPS="$GATE_STEPS $ITERS"

# --- Python helper: write binary init files and run kernel -------------------
# generate initial state using Python (handles N=8192 without shell limits)
python3 - "$WORK" "$N" "$THREADS_PER_BLOCK" "$BLOCKS" "$ITERS" \
    "$GATE_STEPS" "$HESSIAN_N" "$HESSIAN_H" "$RUNNER" "$MB_PTX" \
    "$(pwd)/artifacts/mandelbrot_scale_N${N}.csv" \
    "$(pwd)/artifacts/mandelbrot_hessian_scale.csv" <<'PYEOF'
import sys, os, struct, subprocess, math, csv

WORK       = sys.argv[1]
N          = int(sys.argv[2])
TPB        = int(sys.argv[3])   # threads per block
BLOCKS     = int(sys.argv[4])
ITERS      = int(sys.argv[5])
gate_steps = set(int(x) for x in sys.argv[6].split())
HESSIAN_N  = int(sys.argv[7])
H          = float(sys.argv[8])
RUNNER     = sys.argv[9]
MB_PTX     = sys.argv[10]
CSV_SCALE  = sys.argv[11]
CSV_HESS   = sys.argv[12]

MEM_WORDS = N * 32

def f32pack(vals): return struct.pack(f'{len(vals)}f', *vals)
def f32unpack(data): return list(struct.unpack(f'{len(data)//4}f', data))

def write_bin(path, vals):
    with open(path, 'wb') as f: f.write(f32pack(vals))

def run_kernel(mem_file, var_file, out_mem_file, blocks, tpb, mem_words):
    r = subprocess.run([
        RUNNER, MB_PTX, '--kernel', 'kaxi_kernel', '--mode', 'epistemic', '--type', 'f32',
        '--blocks', str(blocks), '--threads', str(tpb),
        '--mem-words', str(mem_words),
        '--init-file', mem_file, '--init-var-file', var_file,
        '--print-count', str(mem_words)
    ], capture_output=True, text=True)
    status = next((l for l in r.stdout.splitlines() if 'status=' in l), 'no status')
    if 'status=pass' not in status:
        print(f"KERNEL FAIL: {status}", file=sys.stderr)
        print(r.stderr[:500], file=sys.stderr)
        sys.exit(1)
    mem_line = next((l[5:] for l in r.stdout.splitlines() if l.startswith('MEM: ')), '')
    var_line = next((l[5:] for l in r.stdout.splitlines() if l.startswith('VAR: ')), '')
    out_vals = [float(x) for x in mem_line.split()]
    out_vars = [float(x) for x in var_line.split()] if var_line else [0.0] * mem_words
    if out_mem_file:
        write_bin(out_mem_file, out_vals)
    return out_vals, out_vars, status

# Initial state: z_n = e₁+e₂, σ²=0.01 on positions 1,2
# c_n = a_n · e₅ with a_n in [-0.05, +0.05]
z_init  = [0.0] * MEM_WORDS
sz_init = [0.0] * MEM_WORDS
for n in range(N):
    base = n * 32
    z_init[base + 1] = 1.0;  z_init[base + 2] = 1.0   # z = e₁+e₂
    sz_init[base + 1] = 0.01; sz_init[base + 2] = 0.01  # σ²

c_vals = [0.05 * (n - (N-1)/2.0) / ((N-1)/2.0) for n in range(N)]  # a_n

Z  = list(z_init)
SZ = list(sz_init)

print(f"mandelbrot_scale: initial state generated, starting {ITERS} iterations...")

# CSV headers
with open(CSV_SCALE, 'w') as f:
    f.write("thread,step,c5,kappa,sigma_kappa,escape\n")
with open(CSV_HESS, 'w') as f:
    f.write("thread,step,c5,kappa,sigma_kappa,trace,anisotropy,escape\n")

def compute_kappa(z_flat, n):
    base = n*32           # input slot holds z_{n+1} after update
    return sum(z_flat[base+i]**2 for i in range(16))

def compute_sigma_kappa(z_flat, sz_flat, n):
    base = n*32           # input slot
    return math.sqrt(sum(4 * z_flat[base+i]**2 * sz_flat[base+i] for i in range(16)))

# Hessian probe per thread (256 pairs, 4 launches)
def run_hessian_probe(thread_n, z_flat, kappa, sigma_kappa, c5, step, escape):
    z_n = z_flat[thread_n*32 : thread_n*32 + 16]  # input z for this thread
    hwork = os.path.join(WORK, f'hess_s{step}_t{thread_n}')
    os.makedirs(hwork, exist_ok=True)

    def stencil_mem(si, sj):
        vals = []
        for pair in range(256):
            ii = pair // 16; jj = pair % 16
            z_p = list(z_n)
            z_p[ii] += si * H;  z_p[jj] += sj * H
            vals.extend(z_p + [0.0]*16)
        return vals

    zeros256 = [0.0] * (256*32)
    stencils = [(1,1,'pp'), (1,-1,'pm'), (-1,1,'mp'), (-1,-1,'mm')]
    results = {}
    for si, sj, name in stencils:
        mem_f = os.path.join(hwork, f'{name}_mem.bin')
        var_f = os.path.join(hwork, f'{name}_var.bin')
        write_bin(mem_f, stencil_mem(si, sj))
        write_bin(var_f, zeros256)
        out_vals, _, _ = run_kernel(mem_f, var_f, None, 256, 1, 256*32)
        results[name] = out_vals

    denom = 4 * H * H
    def norm_sq(mem, tid):
        base = tid*32+16; return sum(mem[base+k]**2 for k in range(16))

    H_mat = [[0.0]*16 for _ in range(16)]
    for pair in range(256):
        i=pair//16; j=pair%16
        npp=norm_sq(results['pp'],pair); npm=norm_sq(results['pm'],pair)
        nmp=norm_sq(results['mp'],pair); nmm=norm_sq(results['mm'],pair)
        H_mat[i][j]=(npp-npm-nmp+nmm)/denom

    trace = sum(H_mat[i][i] for i in range(16))
    abs_diag = [abs(H_mat[i][i]) for i in range(16) if abs(H_mat[i][i]) > 1e-12]
    aniso = max(abs_diag)/min(abs_diag) if len(abs_diag) >= 2 else 0.0

    with open(CSV_HESS, 'a') as f:
        f.write(f"{thread_n},{step},{c5:.6f},{kappa:.6f},{sigma_kappa:.6f},"
                f"{trace:.6f},{aniso:.6f},{escape}\n")

for step in range(1, ITERS+1):
    # Build input file from current Z (input slot only: first 16 words per thread)
    # The kernel reads words [base..base+15] as input, writes [base+16..base+31]
    # We need to pack: [z input][zeros for output] for each thread
    mem_in = []
    var_in = []
    for n in range(N):
        mem_in.extend(Z[n*32:n*32+16] + [0.0]*16)
        var_in.extend(SZ[n*32:n*32+16] + [0.0]*16)

    m_file = os.path.join(WORK, f's{step}_mem.bin')
    v_file = os.path.join(WORK, f's{step}_var.bin')
    write_bin(m_file, mem_in)
    write_bin(v_file, var_in)

    out_vals, out_vars, status = run_kernel(m_file, v_file, None, BLOCKS, TPB, MEM_WORDS)

    # Update Z: z_{n+1}[i] = sedenion_sqr(z_n)[i] + c_n[i]
    # SZ: σ²(z_{n+1}[i]) = σ²(sedenion_sqr(z_n)[i])  (c_n is deterministic)
    new_Z  = [0.0] * MEM_WORDS
    new_SZ = [0.0] * MEM_WORDS
    for n in range(N):
        for i in range(16):
            c5_comp = c_vals[n] if i == 5 else 0.0
            new_Z[n*32 + i]  = out_vals[n*32 + 16 + i] + c5_comp
            new_SZ[n*32 + i] = out_vars[n*32 + 16 + i]
    Z  = new_Z
    SZ = new_SZ

    print(f"  step {step}: {status}")

    # Escape statistics
    if step in gate_steps or step == ITERS:
        n_escape = 0
        for n in range(N):
            kappa = compute_kappa(Z, n)
            sk    = compute_sigma_kappa(Z, SZ, n)
            esc   = 'YES' if kappa > 1e6 else 'no'
            if esc == 'YES': n_escape += 1
            with open(CSV_SCALE, 'a') as f:
                f.write(f"{n},{step},{c_vals[n]:.6f},{kappa:.6f},{sk:.6f},{esc}\n")

        print(f"  step {step}: escape={n_escape}/{N}  ({100*n_escape/N:.1f}%)")

        # Hessian on HESSIAN_N evenly sampled threads
        if HESSIAN_N > 0:
            sample_threads = [int(n * (N-1) / (HESSIAN_N-1)) for n in range(HESSIAN_N)] if HESSIAN_N > 1 else [0]
            print(f"  step {step}: Hessian probing {HESSIAN_N} sampled threads...")
            for t in sample_threads:
                kappa = compute_kappa(Z, t)
                sk    = compute_sigma_kappa(Z, SZ, t)
                esc   = 'YES' if kappa > 1e6 else 'no'
                run_hessian_probe(t, Z, kappa, sk, c_vals[t], step, esc)

print()
print(f"mandelbrot_scale: DONE  N={N}  ITERS={ITERS}")
print(f"  scale CSV: {CSV_SCALE}")
print(f"  hessian CSV: {CSV_HESS}")

# Summary
import statistics as stats
all_rows = list(csv.DictReader(open(CSV_SCALE)))
last_step = max(int(r['step']) for r in all_rows)
last_rows = [r for r in all_rows if int(r['step']) == last_step]
n_esc = sum(1 for r in last_rows if r['escape'] == 'YES')
print(f"\nFinal step {last_step}: {n_esc}/{N} escaped ({100*n_esc/N:.1f}%)")

hess_rows = [r for r in csv.DictReader(open(CSV_HESS)) if r['anisotropy'] != 'nan' and float(r['anisotropy']) > 0]
by_step = {}
for r in hess_rows:
    by_step.setdefault(int(r['step']), []).append(float(r['anisotropy']))
print("Hessian anisotropy by gate step:")
for s, vals in sorted(by_step.items()):
    print(f"  step {s}: n={len(vals)} mean={stats.mean(vals):.2f} max={max(vals):.2f}")
PYEOF

echo "mandelbrot_scale: all done"
echo "  scale CSV:   artifacts/mandelbrot_scale_N${N}.csv"
echo "  hessian CSV: artifacts/mandelbrot_hessian_scale.csv"
