#!/usr/bin/env bash
# Reproducible bigframe-vs-pandas benchmark. Compiles the 4 Sounio bench programs under lean_single,
# times them (min of 4 process launches, build baseline subtracted, divided by K), runs pandas_bench.py,
# and prints the comparison. Honest: Sounio uses scalar bounds-checked loops (no SIMD) -- raw-reduction
# speed trails pandas' vectorized C kernels until roadmap C3 (vectorization/GPU) lands; groupby is at parity.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
D="$(mktemp -d)"; trap 'rm -rf "$D"' EXIT
for f in b_build b_sum b_filtercount b_groupby; do
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile scripts/bench/sio/$f.sio -o "$D/$f" >/dev/null 2>&1
  chmod +x "$D/$f"
done
python3 - "$D" <<'PY'
import subprocess, time, sys
D=sys.argv[1]
def tmin(p,runs=4):
    b=1e9
    for _ in range(runs):
        t=time.perf_counter(); subprocess.run([f"{D}/{p}"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); b=min(b,(time.perf_counter()-t)*1000)
    return b
B=tmin("b_build"); ss=(tmin("b_sum")-B)/100; sf=(tmin("b_filtercount")-B)/100; sg=(tmin("b_groupby")-B)/20
import numpy as np, pandas as pd
n=1_000_000; df=pd.DataFrame({'a':np.arange(n,dtype='float64'),'k':(np.arange(n)%10).astype('float64'),'v':np.ones(n)})
def pb(fn,K):
    fn(); t=time.perf_counter()
    for _ in range(K): fn()
    return (time.perf_counter()-t)/K*1000
ps=pb(lambda:df['a'].sum(),200); pf=pb(lambda:int((df['a']>499999.5).sum()),200); pg=pb(lambda:df.groupby('k')['v'].sum(),50)
print(f"| operation | 1M rows | Sounio ms | pandas {pd.__version__} ms | Sounio/pandas |")
print("|---|---|---|---|---|")
print(f"| col_sum | | {ss:.2f} | {ps:.2f} | {ss/ps:.1f}x |")
print(f"| filter_count | | {sf:.2f} | {pf:.2f} | {sf/pf:.1f}x |")
print(f"| groupby_sum (10 keys) | | {sg:.2f} | {pg:.2f} | {sg/pg:.2f}x |")
print(f"| frame build (1M x 3) | | {B:.0f} | (once) | - |")
PY
