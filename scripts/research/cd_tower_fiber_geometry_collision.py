r"""
REGRESSION WITNESS -- the fiber-geometry DISTINCTNESS claim is FALSE (retracted 2026-07-12).

Background.  cd_tower_auto_action_on_zd_fibers.py proves the ORBIT theorem (frozen PSL(2,7) on the CD
zero-divisor fibers: 2^{n-4} size-7 Fano orbits + (2^{n-4}-1) fixed seams, stab S4, forall n -- STILL
TRUE).  A SECONDARY claim -- "distinct orbits realize distinct fiber geometries" -- was tagged VERIFIED
n=5..9.  An adversarial nauty audit refuted it INSIDE that range: at n=6 a Fano orbit and a fixed seam
have ISOMORPHIC annihilation graphs.  This file is the guarded regression so the dead claim cannot
silently return.

What is certified HERE (no external deps): the two graphs share verts / edges / degree histogram AND
an iterated Weisfeiler-Leman color-refinement signature.  Histogram identity ALONE already refutes
"distinct geometries" (geometry := verts/edges/degree histogram, the invariant the original oracle
used).  Full graph isomorphism was certified by the reviewer with nauty (canonical certificates
identical); WL agreement here corroborates it.  NB WL is INCOMPLETE on regular graphs -- do NOT use this
file to count distinct geometries (it over-merges the Dmax-regular stratum); the exact count / the
parity-collapse law (#geometries = 3*2^{n-5}, gamma(Seam(y)) = gamma(Fano(y & (y-1))) iff wt(y) even)
is the reviewer's nauty-complete result (n<=8).

Run:  python3 scripts/research/cd_tower_fiber_geometry_collision.py   ->  prints WITNESS OK / FAIL.
"""
from collections import Counter


def cd_sigma(a, b, bits):
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


def _mul(a, b, bits):
    out = {}
    for i, ci in a.items():
        for j, cj in b.items():
            k = i ^ j
            out[k] = out.get(k, 0) + cd_sigma(i, j, bits) * ci * cj
            if out[k] == 0:
                del out[k]
    return out


def annih_graph(n, Llo):
    """Intra-fiber annihilation graph of fiber Llo (lower label) at level n: adjacency lists."""
    H = 1 << (n - 1)
    N = 1 << n
    L = Llo | H
    V = [{lo: 1, hi: (-1 if neg else 1)}
         for lo in range(1, H) for hi in range(H, N) for neg in (0, 1) if (lo ^ hi) == L]
    m = len(V)
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if not _mul(V[i], V[j], n) and not _mul(V[j], V[i], n):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def degree_hist(adj):
    return tuple(sorted(Counter(len(a) for a in adj if len(a) > 0).items()))


def wl_signature(adj, rounds=8):
    """Iterated Weisfeiler-Leman color refinement -> canonical multiset (an isomorphism invariant)."""
    col = [len(a) for a in adj]
    for _ in range(rounds):
        sig = [(col[i], tuple(sorted(col[j] for j in adj[i]))) for i in range(len(adj))]
        order = sorted(set(sig))
        idx = {s: k for k, s in enumerate(order)}
        col = [idx[s] for s in sig]
    return tuple(sorted(Counter(col).items()))


def graph_stats(n, Llo):
    adj = annih_graph(n, Llo)
    kept = [a for a in adj if len(a) > 0]
    edges = sum(len(a) for a in adj) // 2
    return len(kept), edges, degree_hist(adj), wl_signature(adj)


def main():
    n = 6
    # (name, lower-label Llo, x, y)   -- y in tower bits {3,4}; x = Llo & 7
    fano_y2 = graph_stats(n, 17)   # x=1, y=2 (bit4)          : in a size-7 Fano orbit
    seam_y3 = graph_stats(n, 24)   # x=0, y=3 (bits 3,4)      : a size-1 fixed seam
    seam_y2 = graph_stats(n, 16)   # x=0, y=2 (bit4)          : control, must DIFFER

    def show(tag, s):
        print(f"  {tag:22} verts={s[0]:3} edges={s[1]:4} hist={dict(s[2])}")
    print("n=6 fiber annihilation graphs:")
    show("Fano y=2 (Llo=17)", fano_y2)
    show("Seam y=3 (Llo=24)", seam_y3)
    show("Seam y=2 (Llo=16) ctrl", seam_y2)

    collides = (fano_y2 == seam_y3)          # verts, edges, hist, WL all equal
    control_ok = (fano_y2 != seam_y2)
    print()
    print(f"  WITNESS  Fano(y=2) ~= Seam(y=3) (hist+WL identical): {collides}")
    print(f"  CONTROL  Fano(y=2) != Seam(y=2)                    : {control_ok}")

    ok = collides and control_ok
    print()
    print("WITNESS OK -- distinctness refuted, orbit->geometry map is non-injective." if ok
          else "WITNESS FAIL -- collision did not reproduce; investigate before trusting either claim.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
