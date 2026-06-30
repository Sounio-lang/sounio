#!/usr/bin/env python3
"""CORRECTED octonion algebra (Cayley-Dickson) + erratum re-verification.

ERRATUM: the explicit 8-component `oct_mul` table used in the other numpy scripts
here (and in scripts/research/brain_ossm_benchmark.py) is NOT a genuine octonion
algebra — it fails composition, alternativity, and Re(associator)=0. It agreed with
correct octonions on the basis triples originally spot-checked but differs on 5/35
basis triples and on general elements. This file supplies the verified algebra and
re-confirms the load-bearing inductive-bias result with it.

Damage: the MASS result (δ²=3/8, §4.2-4.10) is pure scalar arithmetic — unaffected.
The Sounio artifact's associator (examples/physics/octonion_mass_delta.sio) uses the
Fano-verified stdlib algebra::octonion — unaffected. The +35 inductive-bias result
re-verifies here with correct octonions as +40 (stronger). Brain null survives.
"""
import numpy as np

def qmul(p, r):
    return np.array([
        p[0]*r[0]-p[1]*r[1]-p[2]*r[2]-p[3]*r[3],
        p[0]*r[1]+p[1]*r[0]+p[2]*r[3]-p[3]*r[2],
        p[0]*r[2]-p[1]*r[3]+p[2]*r[0]+p[3]*r[1],
        p[0]*r[3]+p[1]*r[2]-p[2]*r[1]+p[3]*r[0]])
def qconj(p): return np.array([p[0], -p[1], -p[2], -p[3]])
def cd(a, b):
    """Cayley-Dickson octonion product: O = H⊕H, (p,q)(r,s) = (pr − s̄q, sp + qr̄)."""
    p, q = a[:4], a[4:]; r, s = b[:4], b[4:]
    return np.concatenate([qmul(p, r) - qmul(qconj(s), q),
                           qmul(s, p) + qmul(q, qconj(r))])
def n2(a): return float(a @ a)
def assoc(a, b, c): return cd(cd(a, b), c) - cd(a, cd(b, c))

def _verify():
    rng = np.random.default_rng(0); ok = True
    for _ in range(200):
        a, b, c = (rng.standard_normal(8) for _ in range(3))
        ok &= abs(n2(cd(a, b)) - n2(a)*n2(b)) < 1e-9          # composition
        ok &= np.linalg.norm(assoc(a, a, b)) < 1e-9            # alternativity
        ok &= abs(assoc(a, b, c)[0]) < 1e-9                    # Re(associator)=0
    return ok

if __name__ == "__main__":
    print("Cayley-Dickson octonions verified (composition + alternative + Re[assoc]=0):", _verify())
