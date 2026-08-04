#!/usr/bin/env python3
"""How local is the flip's effect, and why is the partition preserved?

Structural note derived by hand, to be checked: the flipped pair is
(h, H+h) with h = H/2, and h ^ (H+h) = H, so that pair's own vertex lives in
fiber L = H i.e. Llo = 0 -- which is OUTSIDE the contract's fiber range
(1..H-1). So the flip never touches a vertex's internal product; it can only
alter adjacency BETWEEN the vertex whose lo = h and the vertex whose hi = H+h.
That predicts a small, fixed number of changed edges per fiber.
"""
import sys, numpy as np, collections
sys.path.insert(0,"/workspace/sounio")
import importlib.util
sp=importlib.util.spec_from_file_location(
    "r15","scripts/research/self_falsifying_compilation_line_r15_contract.py")
r15=importlib.util.module_from_spec(sp); sp.loader.exec_module(r15)

def adj(n, Llo, flip):
    H,N=1<<(n-1),1<<n; L=Llo|H
    V=[{lo:1,hi:(-1 if neg else 1)} for lo in range(1,H) for hi in range(H,N)
       for neg in (0,1) if (lo^hi)==L]
    m=len(V); A=np.zeros((m,m),dtype=np.int8)
    for i in range(m):
        for j in range(i+1,m):
            if not r15.mul(V[i],V[j],n,flip,n) and not r15.mul(V[j],V[i],n,flip,n):
                A[i,j]=A[j,i]=1
    return A,V

for n in (5,6):
    H=1<<(n-1); h=H//2; surv=(h,H+h)
    print(f"n={n}  flipped pair {surv};  ({h} ^ {H+h}) = {h^(H+h)} = H -> its own fiber is "
          f"Llo=0, {'EXCLUDED' if True else ''} from range(1,{H})")
    diffs=collections.Counter(); deg_same=0
    for L in range(1,H):
        A0,V=adj(n,L,None); A1,_=adj(n,L,surv)
        nd=int((A0!=A1).sum())//2
        diffs[nd]+=1
        if sorted(A0.sum(1).tolist())==sorted(A1.sum(1).tolist()): deg_same+=1
    print(f"   edges changed per fiber: {dict(diffs)}")
    print(f"   fibers whose DEGREE SEQUENCE is unchanged: {deg_same}/{H-1}")
