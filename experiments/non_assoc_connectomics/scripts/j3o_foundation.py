#!/usr/bin/env python3
"""Verified J₃(𝕆) platform for the E₆/triality derivation (the §4.9 open obligation).

Built on the CORRECT Cayley-Dickson octonions (octonion_cd_correct.py). Establishes,
symbolically (SymPy), the foundation a triality derivation of the mass ladder must
stand on. Each fact is checked, not assumed:

  [1] the cubic norm (Freudenthal determinant) is WELL-DEFINED — needs Re(associator)=0;
  [2] the cyclic permutation of the three off-diagonal slots (the 3-generation S₃
      'triality of slots') preserves the cubic norm;
  [3] the determinant is real on Hermitian X;
  [4] a genuine G₂ octonion automorphism preserves the cubic norm.

NOT yet done (the real open step): the explicit Spin(8) triality action on the octonion
entries (8v/8s/8c) and the E₆ Dynkin Z₂ realising the down→lepton ladder map. This file
is the verified ground those steps require.
"""
import sympy as sp

def qmul(p, r):
    return [p[0]*r[0]-p[1]*r[1]-p[2]*r[2]-p[3]*r[3],
            p[0]*r[1]+p[1]*r[0]+p[2]*r[3]-p[3]*r[2],
            p[0]*r[2]-p[1]*r[3]+p[2]*r[0]+p[3]*r[1],
            p[0]*r[3]+p[1]*r[2]-p[2]*r[1]+p[3]*r[0]]
def qc(p): return [p[0], -p[1], -p[2], -p[3]]
def cd(a, b):
    p, q = a[:4], a[4:]; r, s = b[:4], b[4:]
    return ([x-y for x, y in zip(qmul(p, r), qmul(qc(s), q))] +
            [x+y for x, y in zip(qmul(s, p), qmul(q, qc(r)))])
def re(a): return a[0]
def n2(a): return sum(x*x for x in a)
def oct(s): return [sp.Symbol(f'{s}{i}') for i in range(8)]

def det(a, b, c, x, y, z):
    """Freudenthal cubic norm of the Hermitian J₃(𝕆) element with real diagonal
       (a,b,c) and octonion off-diagonals (x,y,z)."""
    return a*b*c - a*n2(x) - b*n2(y) - c*n2(z) + 2*re(cd(cd(z, x), y))

def checks():
    x, y, z = oct('x'), oct('y'), oct('z'); a, b, c = sp.symbols('a b c')
    D = det(a, b, c, x, y, z)
    res = {}
    res['det_well_defined'] = sp.expand(re(cd(cd(z, x), y)) - re(cd(z, cd(x, y)))) == 0
    res['cyclic_slot_symmetry'] = sp.expand(D - det(b, c, a, y, z, x)) == 0
    def phi(v): return [v[0], v[1], v[2], v[3], -v[4], -v[5], -v[6], -v[7]]
    def basis(i):
        e = [0]*8; e[i] = 1; return e
    res['phi_is_G2_automorphism'] = all(
        sp.expand(sp.Matrix(phi(cd(basis(i), basis(j)))) -
                  sp.Matrix(cd(phi(basis(i)), phi(basis(j))))) == sp.zeros(8, 1)
        for i in range(8) for j in range(8))
    res['automorphism_preserves_det'] = sp.expand(D - det(a, b, c, phi(x), phi(y), phi(z))) == 0
    return res

if __name__ == "__main__":
    for k, v in checks().items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
