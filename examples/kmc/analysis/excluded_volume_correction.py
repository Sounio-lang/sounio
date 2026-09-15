#!/usr/bin/env python3
"""First-order-in-coverage correction to the exact half-order spot reading
of hot_atom_diffraction.sio, from excluded volume between dissociated pairs.

S_F(q) = (1/N) < |sum_p A_p|^2 >,  A_p = sigma(r_p) e^(iq r_p) (1 + g_F(D_p) e^(iq D_p)).
Self terms give the exact reading PRED. Cross terms between pairs p != p'
at relative position R (first atom to first atom) carry
    g_F(R) e^(-iq R) (1 + g_F(D) e^(iq D)) (1 + g_F(D') e^(-iq D')),
weighted by the pair correlation minus one. For random sequential landing
with an exclusion rule, to first order in the pair density rho = pairs/N
that weight is -1 on the excluded set E(D, D') and 0 elsewhere, so
    S_F / rho = PRED / rho - rho * sum_{D, D'} P(D) P(D') sum_{R in E} Re[...]
and the centre-shell ratio SPOT/PRED = 1 + (cross / rho) / (PRED / rho).

Exclusion rule (matches hot_run's landing test): a new pair is refused if
either of its atoms lands on, or on a nearest neighbour of, either atom of
an existing pair (the new pair's own partner exempt). Transient hops are not
modelled here, so only landing-histogram scenarios without hops are covered.

P(D) in PRED is the REALISED distribution, already reweighted by
acceptance; this script uses the landing histogram, which differs at O(rho),
a second-order effect in the ratio.

Modes and shells match sk_field: q = 2 pi (mx, my)/L, |mx|,|my| <= 4, half
plane, shells by mx^2 + my^2.
"""
import cmath
import math
import sys

L = 256


def images(dx, dy):
    return [(dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy), (dy, dx), (-dy, dx), (dy, -dx), (-dy, -dx)]


def g(field, x, y):
    par = {0: x + y, 1: y, 2: x}[field]
    return -1 if par % 2 else 1


def shell_modes(shell):
    out = []
    for mx in range(0, 5):
        for my in range(-4, 5):
            if (mx > 0 or (mx == 0 and my > 0)) and mx * mx + my * my == shell:
                out.append((mx, my))
    return out


def near(p, q):
    """same site or nearest neighbour"""
    d = abs(p[0] - q[0]) + abs(p[1] - q[1])
    return d <= 1


def excluded(D, Dp):
    """Relative positions R of the new pair's first atom that are refused."""
    reach = max(abs(D[0]), abs(D[1]), abs(Dp[0]), abs(Dp[1])) + 2
    a, b = (0, 0), D
    out = []
    for rx in range(-2 * reach, 2 * reach + 1):
        for ry in range(-2 * reach, 2 * reach + 1):
            ap, bp = (rx, ry), (rx + Dp[0], ry + Dp[1])
            if near(ap, a) or near(ap, b) or near(bp, a) or near(bp, b):
                out.append((rx, ry))
    return out


def centre_terms(hist, field, shell):
    """Return (pred_per_rho, cross_per_rho_squared) averaged over the shell's modes."""
    modes = shell_modes(shell)
    dists = []
    for (dx, dy), w in hist.items():
        for im in images(dx, dy):
            dists.append((im, w / 8.0))
    ex_cache = {}
    pred, cross = 0.0, 0.0
    for mx, my in modes:
        qx, qy = 2 * math.pi * mx / L, 2 * math.pi * my / L
        p_mode = sum(w * (2 + 2 * g(field, *D) * math.cos(qx * D[0] + qy * D[1])) for D, w in dists)
        c_mode = 0.0
        for D, w in dists:
            fD = 1 + g(field, *D) * cmath.exp(1j * (qx * D[0] + qy * D[1]))
            for Dp, wp in dists:
                fDp = 1 + g(field, *Dp) * cmath.exp(-1j * (qx * Dp[0] + qy * Dp[1]))
                key = (D, Dp)
                if key not in ex_cache:
                    ex_cache[key] = excluded(D, Dp)
                s = sum(g(field, rx, ry) * cmath.exp(-1j * (qx * rx + qy * ry)) for rx, ry in ex_cache[key])
                c_mode += w * wp * (s * fD * fDp).real
        pred += p_mode
        cross -= c_mode
    return pred / len(modes), cross / len(modes)


def ratio(hist, field, shell, theta):
    rho = theta / 2.0
    p, c = centre_terms(hist, field, shell)
    return 1 + rho * c / p, p, c


def main():
    hist = {(1, 0): 0.15, (2, 0): 0.45, (3, 0): 0.20, (4, 0): 0.15, (5, 0): 0.05}
    print("POST HOC (these simulated ratios were seen before this script existed):")
    print("illustrative histogram with NN exclusion, centre shell n^2=1")
    sim = {0: {0.05: (1.09, 0.10), 0.10: (1.19, 0.15), 0.15: (1.52, 0.18), 0.20: (2.04, 0.22)},
           2: {0.05: (1.01, 0.11), 0.10: (0.88, 0.09), 0.15: (0.84, 0.08), 0.20: (0.65, 0.08)}}
    for f in (0, 2):
        _, p, c = ratio(hist, f, 1, 0.1)
        print(f"  F{f}: PRED/rho = {p:.4f}, first-order cross coefficient = {c:+.3f}  "
              f"(ratio = 1 {'+' if c >= 0 else '-'} {abs(c / p) / 2:.3f} * theta)")
        for th, (r_sim, e_sim) in sim[f].items():
            r_th = 1 + (th / 2) * c / p
            print(f"     theta={th:.2f}: first order {r_th:.3f}   simulated {r_sim:.2f} +- {e_sim:.2f}   "
                  f"({(r_sim - r_th) / e_sim:+.1f} SE)")


if __name__ == "__main__" and "--blind" not in sys.argv:
    main()


BLIND = {
    "B0 (3,0)": {(3, 0): 1.0},
    "B1 (2,1)": {(2, 1): 1.0},
    "B2 (2,2)": {(2, 2): 1.0},
    "B3 mix (1,1).5 (2,0).3 (3,1).2": {(1, 1): 0.5, (2, 0): 0.3, (3, 1): 0.2},
}


def blind_predictions():
    """Predictions for scenarios that had NOT been simulated when this ran.

    Rule BJ1 (fixed here): for each blind scenario and field whose centre
    PRED/rho exceeds 0.5 (a real monopole plateau; dipole-only centres are
    O(q^2) over O(q^2) and ill-conditioned), the first-order ratio
    SPOT/PRED of shell n^2=1 at coverage 0.05 and 0.10 agrees with
    simulation within 3 SE (jackknife over seeds), multiplicity-aware.
    Rule BJ2 (sharpness, reported): whether each prediction differs from the
    no-correction null (ratio 1) by more than 3 simulated SE, i.e. whether
    the comparison could have told the theory apart from no theory.
    """
    print("BLIND PREDICTIONS -- first-order excluded-volume ratio SPOT/PRED, shell 1, NN exclusion, no hops")
    for name, hist in BLIND.items():
        for f in (0, 1, 2):
            p, c = centre_terms(hist, f, 1)
            use = p > 0.5
            cells = "  ".join(f"theta={th:.2f}: {1 + (th / 2) * c / p:.4f}" for th in (0.02, 0.05, 0.10, 0.15, 0.20))
            print(f"PREDICT {name} | F{f} | PRED/rho={p:.4f} cross={c:+.4f} | in_BJ1={use} | {cells}")


if __name__ == "__main__" and "--blind" in sys.argv:
    blind_predictions()
