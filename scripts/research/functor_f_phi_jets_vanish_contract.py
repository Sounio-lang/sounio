#!/usr/bin/env python3
"""
Functor F — the coupling-jet G2 3-form phi(a1,a2,a3) VANISHES: structural theorem.

Turns the B4 measurement of functor_f_petitot_bridge (phi(a1,a2,a3)=0 over 840 configs,
"structural proof open") into a proved theorem over the octonion core.

VERDICT TYPE NAMED IN ADVANCE: this is a POSITIVE structural theorem (green) whose
*content* is the negative for the Petitot butterfly stratum. It PROVES phi(jets)=0; it
does NOT prove "no canonical x^4 factor exists" (that needs the parent's domain-bounded
rank=3, which is left intact). D3 respected: this is a combinatorial fact about the Fano
plane PG(2,2), not an "X === ZD/associator/Petitot" identity.

Setup. Base Fano line (i,j,k) with i^j=k (=> i^j^k=0). Off-line units u1,u2,u3 are three
distinct members of the 4-point complement. The three coupling jets are
    a1 = [e_u1, e_j,  e_k ]      a2 = [e_i,  e_u2, e_k ]      a3 = [e_i,  e_j,  e_u3].

Structure of the proof (each clause below is the executable witness):
  For octonion basis units, omul(e_a,e_b) = cds(a,b) e_{a^b}, so
      [e_a,e_b,e_c] = ( cds(a,b)cds(a^b,c) - cds(b,c)cds(a,b^c) ) e_{a^b^c},
  a SINGLE axis a^b^c whose coefficient is 0 (associative triple / Fano line) or +-2.
  Hence, using i^j^k=0:
      axis(a1) = u1^j^k = u1^i        axis(a2) = i^u2^k = u2^j        axis(a3) = i^j^u3 = u3^k
  each of magnitude 2 (each substituted triple is NOT a Fano line, so non-associative).
  phi3(e_p,e_q,e_r) = cds(p,q)[p^q==r] is nonzero only when p^q^r=0. The axis-triple XOR is
      (u1^i)^(u2^j)^(u3^k) = (u1^u2^u3)^(i^j^k) = u1^u2^u3.
  LEMMA (PG(2,2) incidence): two distinct lines meet in exactly one point, so any line has
  exactly 2 points off L; a line inside the 4-point complement would have 0 points on L ->
  contradiction. Hence no 3 off-line units form a Fano line, i.e. u1^u2^u3 != 0 ALWAYS.
  Therefore axis(a1)^axis(a2)^axis(a3) != 0, phi3(axes)=0, and by trilinearity (a_m is a
  scalar-multiple of a single basis axis) phi(a1,a2,a3)=0 for ALL real jet scalars d.
  The sign prefactor and pairwise-axis-distinctness are NOT load-bearing: the XOR
  obstruction kills phi on its own (measured: axes coincide in 112/168 configs, yet phi=0
  in every one).

Self-contained; embeds an independent axiom-audit of the inherited octonion core first.
"""
import numpy as np
import itertools

np.seterr(all='ignore')
EXACT = 1e-9


def cds(a, b, bits=3):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah, bh = a >= h, b >= h
        al, bl = a & (h - 1), b & (h - 1)
        if not ah and not bh:
            a, b = al, bl
        elif not ah and bh:
            a, b = bl, al
        elif ah and not bh:
            a, b, s = ((al, 0, s) if bl == 0 else (al, bl, -s))
        else:
            a, b, s = ((0, al, -s) if bl == 0 else (bl, al, s))
        bits -= 1
    return s


def omul(A, B):
    C = np.zeros(8)
    for i in range(8):
        for j in range(8):
            C[i ^ j] += cds(i, j) * A[i] * B[j]
    return C


def e(i):
    v = np.zeros(8); v[i] = 1.0; return v


def assoc(u, v, w):
    return omul(omul(u, v), w) - omul(u, omul(v, w))


def phi3(a, b, c):
    return float(np.dot(omul(a, b), c))


FANO = [(i, j, i ^ j) for i in range(1, 8) for j in range(i + 1, 8) if (i ^ j) > j]


def audit_core():
    ident = all(np.allclose(omul(e(0), e(j)), e(j)) for j in range(8))
    sq = all(np.allclose(omul(e(i), e(i)), -e(0)) for i in range(1, 8))
    anti = all(np.allclose(omul(e(i), e(j)), -omul(e(j), e(i)))
               for i in range(1, 8) for j in range(1, 8) if i != j)
    alt = all(np.allclose(omul(omul(e(i), e(i)), e(j)), omul(e(i), omul(e(i), e(j))))
              for i in range(8) for j in range(8))
    ok = ident and sq and anti and alt
    print(f"C0_CORE_AUDIT identity={ident} sq=-1={sq} anticomm={anti} alternative={alt} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def single_axis(v, tol=EXACT):
    """Return (index, coeff) if v is supported on one basis axis, else (None, None)."""
    nz = np.nonzero(np.abs(v) > tol)[0]
    if len(nz) == 1:
        return int(nz[0]), float(v[nz[0]])
    if len(nz) == 0:
        return 0, 0.0
    return None, None


def configs():
    """All (line, u1,u2,u3): 7 lines x 4P3 = 168 distinct off-line assignments."""
    for (i, j, k) in FANO:
        offs = [u for u in range(1, 8) if u not in (i, j, k)]
        for (u1, u2, u3) in itertools.permutations(offs, 3):
            yield (i, j, k), u1, u2, u3


def main():
    print("=" * 74)
    print("FUNCTOR F — phi(a1,a2,a3) VANISHES on the three coupling jets (structural)")
    print("=" * 74)
    core = audit_core()

    cfgs = list(configs())
    ncfg = len(cfgs)

    # C1 — axis(a_m) EXHIBITED: each jet is a single basis axis of magnitude 2,
    #      with axis(a1)=u1^i, axis(a2)=u2^j, axis(a3)=u3^k.
    c1 = True
    for (i, j, k), u1, u2, u3 in cfgs:
        a1 = assoc(e(u1), e(j), e(k))
        a2 = assoc(e(i), e(u2), e(k))
        a3 = assoc(e(i), e(j), e(u3))
        for a, want in ((a1, u1 ^ i), (a2, u2 ^ j), (a3, u3 ^ k)):
            ax, co = single_axis(a)
            if ax != want or abs(abs(co) - 2.0) > EXACT:
                c1 = False
                print(f"  C1 MISMATCH line={(i,j,k)} u=({u1},{u2},{u3}) axis={ax} want={want} coeff={co}")
    print(f"C1_AXIS_EXHIBITED assoc(a_m)=+-2*e_axis, axis(a1)=u1^i axis(a2)=u2^j axis(a3)=u3^k "
          f"over {ncfg} configs {'PASS' if c1 else 'FAIL'}")

    # C2 — no-line-in-complement LEMMA (PG(2,2) incidence), exhaustive: for each of the 7
    #      lines, none of the C(4,3)=4 off-line triples XORs to 0  (28 checks).
    n_checks = 0
    n_lines_in_complement = 0
    for (i, j, k) in FANO:
        offs = [u for u in range(1, 8) if u not in (i, j, k)]
        for tri in itertools.combinations(offs, 3):
            n_checks += 1
            if tri[0] ^ tri[1] ^ tri[2] == 0:
                n_lines_in_complement += 1
    c2 = (n_checks == 28 and n_lines_in_complement == 0)
    print(f"C2_NO_LINE_IN_COMPLEMENT 7 lines x C(4,3)=4 -> {n_checks} triples, "
          f"{n_lines_in_complement} are Fano lines {'PASS' if c2 else 'FAIL'}")

    # C3 — axis-triple XOR == u1^u2^u3, and != 0 over all configs. Also REPORT (not assume)
    #      pairwise-distinctness of the three axes: it is NOT load-bearing.
    xor_ok = True
    xor_zero = 0
    axes_nonzero = True
    distinct_fail = 0
    for (i, j, k), u1, u2, u3 in cfgs:
        ax = (u1 ^ i, u2 ^ j, u3 ^ k)
        xr = ax[0] ^ ax[1] ^ ax[2]
        if xr != (u1 ^ u2 ^ u3):
            xor_ok = False
        if xr == 0:
            xor_zero += 1
        if 0 in ax:
            axes_nonzero = False
        if len(set(ax)) < 3:
            distinct_fail += 1
    c3 = xor_ok and xor_zero == 0 and axes_nonzero
    print(f"C3_XOR_OBSTRUCTION axis1^axis2^axis3 == u1^u2^u3 (={xor_ok}), "
          f"==0 in {xor_zero}/{ncfg}, all-axes-nonzero={axes_nonzero} {'PASS' if c3 else 'FAIL'}")
    print(f"C3_NOTE axes pairwise-distinct FAILS in {distinct_fail}/{ncfg} configs "
          f"=> distinctness is NOT the reason; the XOR!=0 obstruction alone carries it")

    # C4 — axis-triple is NEVER a Fano line (direct membership test against FANO).
    n_axis_line = 0
    for (i, j, k), u1, u2, u3 in cfgs:
        ax = tuple(sorted((u1 ^ i, u2 ^ j, u3 ^ k)))
        # a Fano line = 3 distinct nonzero indices XORing to 0
        if len(set(ax)) == 3 and 0 not in ax and (ax[0] ^ ax[1] ^ ax[2]) == 0:
            n_axis_line += 1
    c4 = (n_axis_line == 0)
    print(f"C4_AXES_NEVER_FANO_LINE axis-triple is a Fano line in {n_axis_line}/{ncfg} configs "
          f"{'PASS' if c4 else 'FAIL'}")

    # C5 — phi(a1,a2,a3) == 0 over all configs, for d=1 AND random real d (trilinearity).
    rng = np.random.default_rng(0)
    worst = 0.0
    n_scaled = 0
    for (i, j, k), u1, u2, u3 in cfgs:
        base = [assoc(e(u1), e(j), e(k)), assoc(e(i), e(u2), e(k)), assoc(e(i), e(j), e(u3))]
        ds = [np.array([1.0, 1.0, 1.0])] + [rng.random(3) * 1.5 for _ in range(5)]
        for d in ds:
            a1, a2, a3 = d[0] * base[0], d[1] * base[1], d[2] * base[2]
            worst = max(worst, abs(phi3(a1, a2, a3)))
            n_scaled += 1
    c5 = worst < EXACT
    print(f"C5_PHI_JETS_VANISH phi(a1,a2,a3)=0 over {n_scaled} config*scalings "
          f"(worst={worst:.1e}) {'PASS' if c5 else 'FAIL'}")

    print("=" * 74)
    if core and c1 and c2 and c3 and c4 and c5:
        print("FUNCTOR_F_PHI_JETS_VERDICT PHI_JETS_VANISH_PROVEN")
        print("FUNCTOR_F_PHI_JETS_NOTE axis(a_m)=+-2*e_{u_m^line_m}; axis-triple XOR = u1^u2^u3; "
              "PG(2,2) => no line in the 4-pt off-line complement => u1^u2^u3!=0 => phi3(axes)=0; "
              "trilinear in d => phi(jets)=0 for all real d; octonion-core-scoped; "
              "strengthens B_OBSTRUCTED (measured->theorem), does NOT prove x^4 non-existence; "
              "D3_respected (Fano-plane combinatorics, not an identity claim)")
        return 0
    print("FUNCTOR_F_PHI_JETS_VERDICT INCONCLUSIVE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
