#!/usr/bin/env python3
"""Oracle: Aut(S)=G2xS3 executed (Frente B, vector 4/3 capstone²). Verifies eps/psi automorphisms, the
S3 braid relation, and that the color-Weyl phi commutes with both family generators (direct product).
Over Q(sqrt3). Ref: Gresnigt arXiv:2306.13098 §4.3 eq (47)-(55). See gresnigt_g2s3.md."""
from fractions import Fraction as F

radd = lambda x, y: (x[0] + y[0], x[1] + y[1])
rmul = lambda x, y: (x[0] * y[0] + 3 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])
RZ = (F(0), F(0)); RO = (F(1), F(0)); RN = (F(-1), F(0))


def cd_sigma(a, b, bits=4):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def smul(u, v):
    w = [RZ] * 16
    for a in range(16):
        if u[a] == RZ:
            continue
        for b in range(16):
            if v[b] == RZ:
                continue
            k = a ^ b
            s = cd_sigma(a, b); t = rmul(u[a], v[b])
            w[k] = radd(w[k], (s * t[0], s * t[1]))
    return w


def unit(i):
    e = [RZ] * 16; e[i] = RO; return e


def psi_u(i):
    e = [RZ] * 16
    if i in (0, 8):
        e[i] = RO
    elif 1 <= i <= 7:
        e[i] = (F(-1, 2), F(0)); e[i + 8] = (F(0), F(-1, 2))
    else:
        e[i - 8] = (F(0), F(1, 2)); e[i] = (F(-1, 2), F(0))
    return e


def eps_u(i):
    e = [RZ] * 16; e[i] = RO if i <= 7 else RN; return e


g = [0, 2, 3, 1, 4, 6, 7, 5, 8, 10, 11, 9, 12, 14, 15, 13]


def phi_u(i):
    e = [RZ] * 16; e[g[i]] = RO; return e


def applyvec(fu, u):
    out = [RZ] * 16
    for i in range(16):
        if u[i] == RZ:
            continue
        fi = fu(i)
        for k in range(16):
            out[k] = radd(out[k], rmul(u[i], fi[k]))
    return out


def is_auto(fu):
    return all(applyvec(fu, smul(unit(j), unit(k))) == smul(fu(j), fu(k)) for j in range(16) for k in range(16))


def comp(f, h):
    return lambda i: applyvec(f, h(i))


def eqmap(f, h):
    return all(f(i) == h(i) for i in range(16))


def main():
    eps_auto = 1 if is_auto(eps_u) else 0
    s3_rel = 1 if eqmap(comp(eps_u, psi_u), comp(comp(psi_u, psi_u), eps_u)) else 0
    comm_pp = 1 if eqmap(comp(phi_u, psi_u), comp(psi_u, phi_u)) else 0
    comm_pe = 1 if eqmap(comp(phi_u, eps_u), comp(eps_u, phi_u)) else 0
    psi_nonmono = 1 if any(psi_u(j)[k][1] != 0 for j in range(16) for k in range(16)) else 0
    def mono(fu):
        for j in range(16):
            nz = [(k, fu(j)[k]) for k in range(16) if fu(j)[k] != RZ]
            if len(nz) != 1 or nz[0][1][1] != 0 or nz[0][1][0] not in (F(1), F(-1)):
                return False
        return True
    eps_mono = 1 if mono(eps_u) else 0
    def mixes(fu):
        return any((i <= 7 and any(fu(i)[k] != RZ for k in range(8, 16)))
                   or (i >= 9 and any(fu(i)[k] != RZ for k in range(1, 8))) for i in range(16))
    only_psi = 1 if (mixes(psi_u) and not mixes(eps_u) and not mixes(phi_u)) else 0
    print(f"EPS_AUTO {eps_auto}")
    print(f"S3_REL {s3_rel}")
    print(f"COMM_PHI_PSI {comm_pp}")
    print(f"COMM_PHI_EPS {comm_pe}")
    print(f"PSI_NONMONO {psi_nonmono}")
    print(f"EPS_MONO {eps_mono}")
    print(f"ONLY_PSI_MIXES {only_psi}")
    ok = eps_auto and s3_rel and comm_pp and comm_pe and psi_nonmono and eps_mono and only_psi
    print(f"G2S3 {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
