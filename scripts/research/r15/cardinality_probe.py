#!/usr/bin/env python3
"""Does the flip preserve the CARDINALITY the contract actually checks?

126/128 fiber graphs change and so do their spectra, yet the verdict holds. The
contract's claim is #distinct spectra = 3*2^(n-5) = #iso classes. If the flip
permutes the geometries while preserving that COUNT, the check passes by
construction -- a cardinality is blind to any transformation that permutes the
classes it counts.
"""
import concurrent.futures as cf, sys
sys.path.insert(0,"/tmp/claude-1000/-workspace-sounio/1d762349-7a51-4c09-8c6a-44223c57352d/scratchpad")
from resolve import fiber_adj, spec, FLIP

def one(a):
    Llo, flip = a
    return spec(fiber_adj(8, Llo, flip)[0])

if __name__ == "__main__":
    rng = range(1, 128)                       # the contract's own fiber range
    with cf.ProcessPoolExecutor(max_workers=11) as ex:
        base = list(ex.map(one, [(l, None) for l in rng], chunksize=2))
        flip = list(ex.map(one, [(l, FLIP) for l in rng], chunksize=2))
    B, F = set(base), set(flip)
    print(f"n=8, {len(list(rng))} fibers")
    print(f"  distinct spectra, unperturbed : {len(B)}   (claim: 3*2^3 = 24)")
    print(f"  distinct spectra, sigma(64,192) flipped : {len(F)}")
    print(f"  the two SETS are {'IDENTICAL' if B==F else 'DIFFERENT'}")
    print(f"  per-fiber assignment changed on {sum(1 for x,y in zip(base,flip) if x!=y)}/{len(base)} fibers")
    print()
    if len(B)==len(F):
        print("CONFIRMED: the flip preserves the CARDINALITY the check tests,")
        print("  while changing essentially every geometry that cardinality counts.")
    else:
        print("NOT the explanation: the count changes too, so the verdict's")
        print("  insensitivity lies somewhere else again.")
