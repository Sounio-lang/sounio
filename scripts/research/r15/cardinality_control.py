#!/usr/bin/env python3
"""Control for the cardinality explanation.

If EVERY level-8 flip preserved the count of distinct spectra, the count would
simply be robust and sigma(64,192) would not be special. The explanation only
holds if flips that DO kill the contract change the count.

Runs three flips known to kill (from R14's battery and probes) plus the
surviving one, and reports the count each produces.
"""
import concurrent.futures as cf, sys
sys.path.insert(0,"/tmp/claude-1000/-workspace-sounio/1d762349-7a51-4c09-8c6a-44223c57352d/scratchpad")
from resolve import fiber_adj, spec

CASES = [("none (baseline)", None),
         ("SURVIVOR  (64,192)", (64, 192)),
         ("killer    (64,160)", (64, 160)),
         ("killer    (32,160)", (32, 160)),
         ("killer     (1,129)", (1, 129))]

def one(a):
    Llo, flip = a
    return spec(fiber_adj(8, Llo, flip)[0])

if __name__ == "__main__":
    rng = list(range(1, 128))
    with cf.ProcessPoolExecutor(max_workers=11) as ex:
        for label, flip in CASES:
            s = set(ex.map(one, [(l, flip) for l in rng], chunksize=2))
            print(f"  {label:<22} distinct spectra = {len(s):3}", flush=True)
    print("\nIf the killers move off 24 and the survivor stays at 24, the")
    print("contract's insensitivity is exactly the cardinality being preserved.")
