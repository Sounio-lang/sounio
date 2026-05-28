#!/usr/bin/env python3
"""Machine-checked structure of the sedenion Φ-map (correction to "A Dual Pathway to 168").

This module establishes, by exact integer arithmetic, that the *unordered* primitive
zero-divisor map Φ̄ from the 168 ordered non-collinear Fano triples is NOT injective
(Theorem 2 of the preprint is false as stated), and gives the full structure of the
defect — including the closed-form GL(3,2) sign cocycle that obstructs an equivariant
bijection.

It reuses the Cayley–Dickson sign `cd_sigma` and basis multiplication `basis_mul` from
the companion oracle `generate_sedenion_zero_divisor_geometry.py` (no reimplementation).

Verified results (all N/N, zero violations):
  - 168 admissible ordered triples (s = p⊕q⊕r ≠ 0, {p,q,r,s} distinct; p⊕q=r⊕s is vacuous).
  - Eq.12 δ_b = −σ(p,r)σ(s,q) is the UNIQUE δ_b making a·b = 0  (ground truth vs basis_mul).
  - I4: ι(p,q,r) = (r,s,p) is a fixed-point-free involution closing on the 168 (84 orbits).
  - I1: δ_b ∘ ι = δ_b.
  - I3: σ(p,r)σ(s,q)σ(s,p)σ(q,r) = −1 (Sign Cancellation Lemma).
  - I2: Φ̄(T)=Φ̄(ιT) ⇔ δ_b(T)=+1; every unordered collision is an ι-orbit.
  - |Im Φ̄| = 126, with 42 collisions (= the 84 δ_b=+1 triples / 42 ι-orbits). 126 = 84 + 42.
  - Incompatibility: GL(3,2) acts transitively on the 168 ⇒ NO equivariant non-collision
    gauge exists; a non-equivariant gauge gives a genuine 168↔168 bijection.
  - Cocycle: δ_b(gT) = ω(g,T)·δ_b(T) with ω(g,T) = ∏_{x∈{p,q,r,s}} ε_g(x); ω trivial exactly
    on the Frobenius subgroup F_21 = 7:3 (the 21 sign-free octonion automorphisms).
  - Fano structure: 84 ι-orbits ↔ 21 Fano line-pairs (flags) × 4 orientations, split 2/2 by δ_b.

Run:  python3 scripts/research/generate_sedenion_phi_map_injectivity.py [--out FILE]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORACLE = os.path.join(ROOT, "scripts", "research", "generate_sedenion_zero_divisor_geometry.py")
DEFAULT_OUT = os.path.join(ROOT, "artifacts", "research", "sedenion_phi_map_injectivity.v1.json")

# ---- reuse the compiler-faithful sign + multiplication from the companion oracle ----
_spec = importlib.util.spec_from_file_location("zdgeom", ORACLE)
_orc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orc)


def sigma(a: int, b: int) -> int:
    """σ(a,b) for sedenion basis indices (bits=4); equals octonion σ on a,b ∈ {1..7}."""
    return _orc.cd_sigma(a, b, 4)


def vmul(A: dict, B: dict) -> dict:
    """Sparse product of two basis-coefficient vectors via the oracle's basis_mul."""
    out: dict = {}
    for i, ci in A.items():
        for j, cj in B.items():
            s, basis = _orc.basis_mul(i, j, 4)
            out[basis] = out.get(basis, 0) + ci * cj * s
    return {k: v for k, v in out.items() if v}


# ---- the Φ-map construction (preprint definitions) ----
def s_of(t):
    p, q, r = t
    return p ^ q ^ r


def admissible_triples():
    """The 168 ordered non-collinear triples: distinct p,q,r in {1..7}, s≠0, {p,q,r,s} distinct."""
    out = []
    for p in range(1, 8):
        for q in range(1, 8):
            for r in range(1, 8):
                if len({p, q, r}) < 3:
                    continue
                s = p ^ q ^ r
                if s == 0 or len({p, q, r, s}) < 4:
                    continue
                out.append((p, q, r))
    return out


def delta_b(t):
    """Eq.12: δ_b = −δ_a·σ(p,r)·σ(s,q), with δ_a = +1."""
    p, q, r = t
    return -sigma(p, r) * sigma(s_of(t), q)


def iota(t):
    """Order-swap involution: the triple whose Φ is the reverse of Φ(T)."""
    p, q, r = t
    return (r, s_of(t), p)


def phi(t, alpha=1):
    """Φ_α(T) = (a, b) as basis-coefficient vectors; a's high-index sign is δ_a=α,
    b's is δ_b' = α·δ_b(T) (= −δ_a σσ). α=1 recovers the preprint's Φ."""
    p, q, r = t
    s = s_of(t)
    a = {p: 1, q + 8: alpha}
    b = {r: 1, s + 8: alpha * delta_b(t)}
    return a, b


def vec_key(v: dict):
    return tuple(sorted(v.items()))


def phi_unordered(t, alpha_map=None):
    a, b = phi(t, 1 if alpha_map is None else alpha_map[t])
    return frozenset({vec_key(a), vec_key(b)})


# ---- GL(3,2): invertible 3x3 over F2, acting on points {1..7} = F2^3\{0} ----
def matvec(M, v):
    r = 0
    for k in range(3):
        r |= (bin(M[k] & v).count("1") & 1) << k
    return r


def gl32():
    return [rows for rows in itertools.product(range(8), repeat=3)
            if len({matvec(rows, v) for v in range(1, 8)}) == 7]


def perm(g):
    return {i: matvec(g, i) for i in range(1, 8)}


def solve_eps(g):
    """Sign-lift ε_g of g as a G2 automorphism: solve η(i)+η(j)+η(i⊕j)=b_ij over F2,
    b_ij = [σ(gi,gj) ≠ σ(i,j)]. Returns ε:{1..7}→{±1} (one solution; free vars 0), or None."""
    pi = perm(g)
    rows = []
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = i ^ j
            b = 0 if sigma(pi[i], pi[j]) == sigma(i, j) else 1
            row = [0] * 7
            for idx in (i, j, k):
                row[idx - 1] ^= 1
            rows.append(row + [b])
    rr = 0
    for c in range(7):
        pr = next((r for r in range(rr, len(rows)) if rows[r][c] == 1), None)
        if pr is None:
            continue
        rows[rr], rows[pr] = rows[pr], rows[rr]
        for r in range(len(rows)):
            if r != rr and rows[r][c] == 1:
                rows[r] = [a ^ b for a, b in zip(rows[r], rows[rr])]
        rr += 1
    for r in rows:
        if all(c == 0 for c in r[:7]) and r[7] == 1:
            return None
    eta = [0] * 7
    for r in rows:
        cols = [c for c in range(7) if r[c] == 1]
        if cols:
            eta[cols[0]] = r[7]
    return [-1 if eta[i] else 1 for i in range(7)]


def compose(g, h):
    cols = [matvec(g, matvec(h, 1 << k)) for k in range(3)]
    M = [0, 0, 0]
    for c in range(3):
        for r in range(3):
            if (cols[c] >> r) & 1:
                M[r] |= (1 << c)
    return tuple(M)


def check(name, confirmed, total, rec):
    ok = confirmed == total
    rec[name] = {"confirmed": confirmed, "total": total, "pass": ok}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:42s} {confirmed}/{total}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    T = admissible_triples()
    rec = {}
    allok = True

    print("== Φ-map injectivity structure (machine-checked) ==")
    allok &= check("admissible triples == 168", len(T), 168, rec)
    allok &= check("parity p⊕q=r⊕s vacuous (auto)",
                   sum(1 for (p, q, r) in T if (p ^ q) == (r ^ (p ^ q ^ r))), len(T), rec)

    # Eq.12 is the unique zero-divisor sign
    uniq_zd = 0
    for t in T:
        ds = [d for d in (1, -1) if not vmul({t[0]: 1, t[1] + 8: 1}, {t[2]: 1, s_of(t) + 8: d})]
        if ds == [delta_b(t)]:
            uniq_zd += 1
    allok &= check("Eq.12 δ_b is the unique a·b=0 sign", uniq_zd, len(T), rec)

    # I4 involution
    Tset = set(T)
    allok &= check("I4 ι involution closes (ι²=id, ι(T)∈dom)",
                   sum(1 for t in T if iota(iota(t)) == t and iota(t) in Tset), len(T), rec)
    allok &= check("I4 ι fixed-point-free", sum(1 for t in T if iota(t) != t), len(T), rec)

    # I1, I3
    allok &= check("I1 δ_b∘ι = δ_b", sum(1 for t in T if delta_b(t) == delta_b(iota(t))), len(T), rec)
    allok &= check("I3 Sign Cancellation Lemma == −1",
                   sum(1 for (p, q, r) in T
                       if sigma(p, r) * sigma(p ^ q ^ r, q) * sigma(p ^ q ^ r, p) * sigma(q, r) == -1),
                   len(T), rec)

    # ordered injectivity
    ordimg = [(vec_key(a), vec_key(b)) for a, b in (phi(t) for t in T)]
    allok &= check("Φ ordered injective (168 distinct)", len(set(ordimg)), len(T), rec)

    # I2 collision criterion + image size
    cU = Counter(phi_unordered(t) for t in T)
    allok &= check("I2 collision ⇔ δ_b=+1",
                   sum(1 for t in T if (cU[phi_unordered(t)] > 1) == (delta_b(t) == 1)), len(T), rec)
    allok &= check("every collision is an ι-orbit",
                   sum(1 for t in T if cU[phi_unordered(t)] <= 1 or phi_unordered(t) == phi_unordered(iota(t))),
                   len(T), rec)
    img = len(cU)
    collisions = sum(1 for v in cU.values() if v > 1)
    db_dist = Counter(delta_b(t) for t in T)
    rec["image_size"] = img
    rec["collisions"] = collisions
    rec["delta_b_distribution"] = {str(k): v for k, v in db_dist.items()}
    print(f"  |Im Φ̄| = {img}  (target 126)   collisions = {collisions}   δ_b: {dict(db_dist)}")
    allok &= check("|Im Φ̄| == 126", img, 126, rec)
    allok &= check("collisions == 42", collisions, 42, rec)
    allok &= check("126 == 84 + 42", img, db_dist[-1] + collisions, rec)

    # all Φ images are genuine zero divisors
    allok &= check("all Φ(T) are zero divisors (a·b=0)",
                   sum(1 for t in T if not vmul(*phi(t))), len(T), rec)

    # ---- incompatibility: GL(3,2) transitive ⇒ no equivariant gauge ----
    GL = gl32()
    rec["GL32_order"] = len(GL)
    # transitivity on triples
    def actT(g, t):
        pi = perm(g)
        return (pi[t[0]], pi[t[1]], pi[t[2]])
    orbit0 = {actT(g, T[0]) for g in GL} & Tset
    allok &= check("GL(3,2) order 168", len(GL), 168, rec)
    allok &= check("GL(3,2) transitive on the 168", len(orbit0), 168, rec)
    # equivariant gauge would be constant (transitive) ⇒ needs δ_b≡−1 (false) ⇒ obstructed
    rec["equivariant_gauge_exists"] = (db_dist[1] == 0)
    print(f"  [{'PASS' if not rec['equivariant_gauge_exists'] else 'FAIL'}] "
          f"equivariant non-collision gauge OBSTRUCTED (δ_b=+1 count={db_dist[1]}>0, action transitive)")
    allok &= not rec["equivariant_gauge_exists"]
    # non-equivariant gauge: α(rep)=+1, α(ιrep)=−δ_b(rep) ⇒ 168 distinct unordered images
    alpha = {}
    done = set()
    for t in T:
        if t in done:
            continue
        it = iota(t)
        alpha[t] = 1
        alpha[it] = -delta_b(t)
        done |= {t, it}
    gauged = len({phi_unordered(t, alpha) for t in T})
    allok &= check("non-equivariant gauge ⇒ 168↔168 bijection", gauged, 168, rec)

    # ---- closed-form cocycle ω(g,T) = ∏_{Q} ε_g ----
    eps = {}
    for g in GL:
        e = solve_eps(g)
        if e is not None:
            eps[g] = e
    allok &= check("all g lift to G2 with signs ε_g", len(eps), len(GL), rec)
    sigma_free = [g for g in GL
                  if all(sigma(perm(g)[i], perm(g)[j]) == sigma(i, j)
                         for i in range(1, 8) for j in range(1, 8) if i != j)]
    sf = set(sigma_free)
    rec["sigma_free_subgroup_order"] = len(sigma_free)
    allok &= check("σ-free set has order 21 (Frobenius 7:3)", len(sigma_free), 21, rec)
    allok &= check("σ-free set is a subgroup",
                   sum(1 for g in sigma_free for h in sigma_free if compose(g, h) in sf),
                   len(sigma_free) ** 2, rec)
    # ω(g,T) = ∏_Q ε_g(x) == δ_b(gT)/δ_b(T)
    coc_ok = 0
    coc_tot = 0
    for g, e in eps.items():
        for t in T:
            p, q, r = t
            s = s_of(t)
            omega = e[p - 1] * e[q - 1] * e[r - 1] * e[s - 1]
            ratio = delta_b(actT(g, t)) // delta_b(t)
            coc_tot += 1
            coc_ok += (omega == ratio)
    allok &= check("ω(g,T)=∏_Q ε_g == δ_b(gT)/δ_b(T)", coc_ok, coc_tot, rec)
    # ω genuinely T-dependent (not a group character): only the 21 σ-free g give constant ω
    const_g = sum(1 for g in eps
                  if len({eps[g][t[0] - 1] * eps[g][t[1] - 1] * eps[g][t[2] - 1] * eps[g][s_of(t) - 1]
                          for t in T}) == 1)
    rec["g_with_constant_omega"] = const_g
    allok &= check("ω constant in T only on the 21 σ-free g", const_g, 21, rec)

    # ---- Fano structure: 84 ι-orbits = 21 line-pairs × 4 orientations, split 2/2 by δ_b ----
    byLP = defaultdict(list)
    seen = set()
    for t in T:
        if t in seen:
            continue
        it = iota(t)
        seen |= {t, it}
        p, q, r = t
        s = s_of(t)
        lp = frozenset({frozenset({p, q}), frozenset({r, s})})
        byLP[lp].append(delta_b(t))
    allok &= check("distinct Fano line-pairs == 21", len(byLP), 21, rec)
    allok &= check("every line-pair splits 2/2 by δ_b",
                   sum(1 for v in byLP.values() if len(v) == 4 and v.count(1) == 2 and v.count(-1) == 2),
                   len(byLP), rec)
    # orbit ↦ {(p,q),(r,s)} is a bijection on the 84 orbits
    seen2 = set()
    labels = []
    for t in T:
        if t in seen2:
            continue
        it = iota(t)
        seen2 |= {t, it}
        labels.append(frozenset({(t[0], t[1]), (t[2], s_of(t))}))
    allok &= check("orbit ↦ {(p,q),(r,s)} injective on 84", len(set(labels)), 84, rec)

    rec["all_pass"] = bool(allok)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    print(f"\n{'ALL CHECKS PASS' if allok else 'SOME CHECKS FAILED'} — artifact: {args.out}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
