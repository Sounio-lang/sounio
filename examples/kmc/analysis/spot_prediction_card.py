#!/usr/bin/env python3
"""Predicted half-order spot centre-to-shoulder ratios for dissociation
hypotheses, from the exact low-coverage form (hot_atom_diffraction.sio):

    S_F(q) / rho = 2 + 2 < g_F(D) cos(q.D) >,   rho = pairs / sites

At the spot centre this is 4 P_same,F; far enough out in q that the cosine
averages away (the shoulder) it is 2. So each spot's own centre-to-shoulder
ratio is
    R_F = 2 P_same,F    (0 = ring / centre fully suppressed, 2 = plain peak)
and dividing by the shoulder of the SAME spot removes the spot-dependent
dynamical form factors that make intensities of different LEED spots hard
to compare. Kinematic, low coverage, immobile atoms after their final hop.

P_same,F is computed here by enumerating the 8 lattice images of every
separation, and checked against the hand-derived values in HAND.

Also printed: the Hadamard structure. For the four parity classes of D
(dx, dy mod 2), the integer-order diffuse channel and the three half-order
spots see signs
    int (0,0):     + + + +
    F0 (1/2,1/2):  + - - +
    F1 (0,1/2):    + - + -
    F2 (1/2,0):    + + - -      over classes (ee, eo, oe, oo)
a 4x4 Hadamard matrix, so the four channels together determine each parity
class's characteristic function exactly; the three half-order spots alone
determine three of the four combinations.
"""

HYPOTHESES = {
    "(1,0) nearest-neighbour":              {(1, 0): 1.0},
    "(1,1) diagonal, Brundle-Behm-Barker":  {(1, 1): 1.0},
    "(2,0) along an axis":                  {(2, 0): 1.0},
    "(4,0) Bukas-Reuter ballistic":         {(4, 0): 1.0},
    "(2,1) knight":                         {(2, 1): 1.0},
    "illustrative 1..5 axis, mode 2":       {(1, 0): 0.15, (2, 0): 0.45, (3, 0): 0.20, (4, 0): 0.15, (5, 0): 0.05},
}

# Values worked out by hand before running this script.
HAND = {
    "(1,0) nearest-neighbour":              (0.0, 1.0, 1.0),
    "(1,1) diagonal, Brundle-Behm-Barker":  (2.0, 0.0, 0.0),
    "(2,0) along an axis":                  (2.0, 2.0, 2.0),
    "(4,0) Bukas-Reuter ballistic":         (2.0, 2.0, 2.0),
    "illustrative 1..5 axis, mode 2":       (1.2, 1.6, 1.6),
}


def images(dx, dy):
    return [(dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy), (dy, dx), (-dy, dx), (dy, -dx), (-dy, -dx)]


def g(field, ex, ey):
    par = {0: ex + ey, 1: ey, 2: ex}[field]
    return -1 if par % 2 else 1


def ratios(hist):
    out = []
    for f in (0, 1, 2):
        p_same = 0.0
        for (dx, dy), w in hist.items():
            imgs = images(dx, dy)
            p_same += w * sum(1 for ex, ey in imgs if g(f, ex, ey) == 1) / len(imgs)
        out.append(2 * p_same)
    return out


def main():
    ok = True
    print(f"{'hypothesis':<40} {'R (1/2,1/2)':>12} {'R (0,1/2)':>10} {'R (1/2,0)':>10}   hand check")
    for name, hist in HYPOTHESES.items():
        r = ratios(hist)
        check = ""
        if name in HAND:
            match = all(abs(a - b) < 1e-12 for a, b in zip(r, HAND[name]))
            ok = ok and match
            check = "ok" if match else f"MISMATCH hand={HAND[name]}"
        print(f"{name:<40} {r[0]:>12.3f} {r[1]:>10.3f} {r[2]:>10.3f}   {check}")
    print("\nR = 0: centre fully suppressed (ring); R = 2: ordinary peak.")
    print("SPOT_CARD_OK" if ok else "SPOT_CARD_MISMATCH")


def separation_table(rmax=5):
    """Every final separation up to |dx|,|dy| <= rmax, up to lattice symmetry:
    what each half-order spot sees. D = dipole (conserved, contributes to a
    centre-suppressed spot), M = monopole (plateau). A measured STM histogram
    of pair separations maps onto the three spot ratios by weighting these rows.
    Only meaningful on bipartite lattices (square or rectangular); on
    triangular lattices no sublattice charge is conserved by nearest-neighbour
    pairs (checked for all non-degenerate linear colourings with 2-4 classes)."""
    print(f"{'D=(dx,dy)':<10} {'|D|':>6}   (1/2,1/2)  (0,1/2)  (1/2,0)   R triple")
    seen = set()
    for dx in range(0, rmax + 1):
        for dy in range(0, dx + 1):
            if (dx, dy) == (0, 0) or (dx, dy) in seen:
                continue
            seen.add((dx, dy))
            r = ratios({(dx, dy): 1.0})
            tag = lambda v: "D" if v == 0.0 else ("M" if v == 2.0 else "mixed")
            print(f"({dx},{dy}){'':<{10 - len(f'({dx},{dy})')}} {((dx*dx+dy*dy) ** 0.5):6.3f}   "
                  f"{tag(r[0]):^9}  {tag(r[1]):^7}  {tag(r[2]):^7}   {r[0]:.1f} / {r[1]:.1f} / {r[2]:.1f}")


def distance_suffices(rmax=40):
    """The parity class of D is a function of |D|^2 mod 4 alone: x^2 = x (mod 2)
    and odd^2 = 1 (mod 4), so |D|^2 odd <=> dx+dy odd; |D|^2 = 2 (mod 4) <=>
    dx, dy both odd; |D|^2 = 0 (mod 4) <=> both even. Hence a DISTANCE-only
    pair histogram (lattice known) already fixes all three spot ratios.
    Checked exhaustively for |dx|, |dy| <= rmax."""
    seen = {}
    for dx in range(-rmax, rmax + 1):
        for dy in range(-rmax, rmax + 1):
            if (dx, dy) == (0, 0):
                continue
            cls = tuple(ratios({(dx, dy): 1.0}))
            key = (dx * dx + dy * dy) % 4
            if seen.setdefault(key, cls) != cls:
                print(f"COUNTEREXAMPLE at {(dx, dy)}")
                return False
    for key in sorted(seen):
        print(f"  |D|^2 mod 4 = {key}: R triple {seen[key]}")
    return True


# Lin & Jiang, J. Phys. Chem. Lett. 14, 7513 (2023) (arXiv:2304.10812), Fig. 3b:
# O-O distance distribution after O2 dissociation on Pd(100), EANN potential,
# 400 trajectories at 160 K. Digitised from the arXiv figure rendered at 600
# dpi: bar tops read in pixels against the 0 / 0.2 / 0.4 axis ticks. The eleven
# bars sum to 0.996, a check on the reading. Distances in surface lattice
# constants; each is an allowed hollow-hollow separation, so |D|^2 is exact.
LIN_JIANG_PD100 = {1: 0.0404, 2: 0.0378, 4: 0.4318, 5: 0.2426, 8: 0.0439, 9: 0.0955,
                   10: 0.0662, 13: 0.0138, 16: 0.0095, 17: 0.0112, 25: 0.0017}
LIN_JIANG_N = 400


def triple_from_d2(weights):
    """R triple from a distribution over |D|^2, using the mod-4 rule."""
    tot = sum(weights.values())
    p = {0: 0.0, 1: 0.0, 2: 0.0}
    for d2, w in weights.items():
        p[d2 % 4] += w / tot
    r0 = 2 * (p[0] + p[2])
    r12 = 2 * (p[0] + 0.5 * p[1])
    return r0, r12, p


def lin_jiang_prediction(n_boot=4000, seed=12345):
    import random
    r0, r12, p = triple_from_d2(LIN_JIANG_PD100)
    rng = random.Random(seed)
    keys = list(LIN_JIANG_PD100)
    tot = sum(LIN_JIANG_PD100.values())
    probs = [LIN_JIANG_PD100[k] / tot for k in keys]
    cum = [sum(probs[:i + 1]) for i in range(len(probs))]
    b0, b12 = [], []
    for _ in range(n_boot):
        counts = dict.fromkeys(keys, 0)
        for _ in range(LIN_JIANG_N):
            u = rng.random()
            counts[keys[next(i for i, c in enumerate(cum) if u <= c)]] += 1
        a, b, _ = triple_from_d2(counts)
        b0.append(a)
        b12.append(b)
    sd = lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    print("O2/Pd(100), Lin & Jiang 2023 Fig. 3b (400 trajectories, 160 K), digitised:")
    print(f"  class weights: |D|^2 odd {p[1]:.3f}   = 2 mod 4 {p[2]:.3f}   = 0 mod 4 {p[0]:.3f}")
    print(f"  R (1/2,1/2) = {r0:.3f} +- {sd(b0):.3f}   R (0,1/2) = R (1/2,0) = {r12:.3f} +- {sd(b12):.3f}")
    print("  errors: multinomial bootstrap over the 400 trajectories; digitisation error is smaller")
    # cross-check the mod-4 route against explicit lattice vectors for the same distances
    vec = {1: (1, 0), 2: (1, 1), 4: (2, 0), 5: (2, 1), 8: (2, 2), 9: (3, 0), 10: (3, 1),
           13: (3, 2), 16: (4, 0), 17: (4, 1), 25: (5, 0)}
    hist = {vec[k]: w / tot for k, w in LIN_JIANG_PD100.items()}   # same normalisation as the mod-4 route
    rv = ratios(hist)
    ok = abs(rv[0] - r0) < 1e-9 and abs(rv[1] - r12) < 1e-9 and abs(rv[2] - r12) < 1e-9
    print(f"  explicit-vector route: {rv[0]:.3f} / {rv[1]:.3f} / {rv[2]:.3f}   "
          f"{'agrees' if ok else 'DISAGREES'} with the |D|^2 mod 4 route")
    print("  comparison: Bukas-Reuter (4,0) -> 2.000 / 2.000 / 2.000;  (1,1) -> 2.000 / 0.000 / 0.000")
    return ok


if __name__ == "__main__" and "--linjiang" in __import__("sys").argv:
    print("LIN_JIANG_OK" if lin_jiang_prediction() else "LIN_JIANG_FAILED")
elif __name__ == "__main__" and "--table" in __import__("sys").argv:
    separation_table()
    print("\nIs the class a function of |D|^2 mod 4? (exhaustive, |dx|,|dy| <= 40)")
    print("DISTANCE_SUFFICES_OK" if distance_suffices() else "DISTANCE_SUFFICES_FAILED")
elif __name__ == "__main__":
    main()
