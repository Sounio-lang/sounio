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


if __name__ == "__main__":
    main()
