#!/usr/bin/env python3
"""Why does sigma(H/2, H+H/2) preserve the count?

Hypothesis, from the shape of the number: the count is 3*2^(n-5) -- three strata
times a doubling. If the flip preserves the PARTITION of fibers into
spectrum-classes while replacing the spectra that label them, the count survives
and the values change. Exactly what R15 measured.

Test: compare the set partitions, not the spectra. Same blocks -> the flip acts
WITHIN classes, preserving the classification. Different blocks -> the mechanism
is something else and the hypothesis is dead.
"""
import sys
sys.path.insert(0,"/workspace/sounio/scripts/research/r15")
sys.path.insert(0,"/workspace/sounio")
import importlib.util
spec=importlib.util.spec_from_file_location(
    "r15","scripts/research/self_falsifying_compilation_line_r15_contract.py")
r15=importlib.util.module_from_spec(spec); spec.loader.exec_module(r15)

def partition(n, flip):
    by={}
    for L in range(1, 1<<(n-1)):
        by.setdefault(r15.fiber_spectrum(n,L,flip,n),[]).append(L)
    # canonical set partition: frozenset of frozensets of fiber labels
    return frozenset(frozenset(v) for v in by.values()), by

for n in (5,6,7):
    H=1<<(n-1); surv=(H//2, H+H//2)
    P0,b0 = partition(n, None)
    P1,b1 = partition(n, surv)
    sizes0=sorted(len(x) for x in P0); sizes1=sorted(len(x) for x in P1)
    print(f"n={n}  blocks {len(P0)} -> {len(P1)}")
    print(f"   PARTITION {'IDENTICAL' if P0==P1 else 'DIFFERENT'}"
          f"   block sizes {'match' if sizes0==sizes1 else 'differ'}: {sizes0} vs {sizes1}")
    print(f"   spectra sets {'identical' if set(b0)==set(b1) else 'DIFFER'}")
