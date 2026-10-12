#!/usr/bin/env python3
"""Two-body master equation for one dissociated O pair on Pd(100).

ROLE (ADR-008 / ADR-009): external_corroboration_only / research_harness.
This is SciPy float integration. It is not the M9 claim clock, supplies no
expected value that can pass or fail an M9 rule, and cannot be promoted to
verified_foreign_reference. It exists to find bugs and size the problem
(boxes, time scales, where the two dissociation pictures separate) before
the Sounio claim clock for these numbers is written.

At infinite dilution the pair's relative coordinate D = r_B - r_A is itself a
Markov chain: an atom hop changes D by one lattice vector, and rates depend
only on D because the energy is the pair energy V(D). A hop of B by +e and a
hop of A by -e both take D to D + e with the same energy change, so

    W(D -> D+e) = 2 nu exp(-(E0 + (V(D+e) - V(D)) / 2) / kT),   D + e != 0.

With pi(D) ~ exp(-V(D)/kT), pi(D) W(D->D') = pi(D') W(D'->D), so the generator
is symmetrisable: A = Pi^(1/2) Q Pi^(-1/2) has every off-diagonal entry equal
to 2 nu exp(-E0/kT). The stiffness (rates from 1e-17 to 1e9 per second) sits
entirely on the diagonal. In float64 the similarity transform spans ~26
decades and a symmetric eigendecomposition returned R = 2.0004 for a frozen
pair, so the chain is integrated instead with an implicit Radau method on the
sparse generator (rtol 1e-11).

Checks. Dense expm is not a usable reference here: at |Q| t ~ 1e12 its scaling
and squaring lost 1e-4 of the probability mass (60 K, (4,0), 1000 s), more than
the physics it was meant to check. Radau is checked instead against two closed
forms in frozen regimes (analytic_checks), against itself at rtol 1e-13, and
against a box six sites larger. The reflecting box is sized from the free walk
of the relative coordinate, sigma = sqrt(4 k_free t) per component, so the
edge carries no probability; temperatures that would need a box beyond BOX_CAP
are reported as not computed rather than solved in a box they fill.

Diffraction reading: R_F(t) = 2 P_same,F(t), with P_same = probability that
the two atoms carry the same sign of field F:
    F0 (1/2,1/2): dx + dy even;   F1 (0,1/2): dy even;   F2 (1/2,0): dx even.
The dynamics are symmetric under the square's point group and the initial
histogram is spread over the 8 images, so R_F1 = R_F2 throughout.

Usage:
  pair_master_equation.py                 predictions table (pre-registration)
  pair_master_equation.py --window        R(1000 s) against T, and sensitivity
                                          to E0 and to V(2,0)
"""
import math
import sys

import numpy as np
import scipy.integrate
import scipy.sparse

KB = 0.08617333262  # meV / K
NU = 1.0e12
E0 = 230.0
V_PD100 = {(1, 0): 360.0, (1, 1): 128.0, (2, 0): -48.0}

LJ = {(1, 0): 0.0404, (1, 1): 0.0378, (2, 0): 0.4318, (2, 1): 0.2426, (2, 2): 0.0439, (3, 0): 0.0955,
      (3, 1): 0.0662, (3, 2): 0.0138, (4, 0): 0.0095, (4, 1): 0.0112, (5, 0): 0.0017}
BR = {(4, 0): 1.0}
TIMES = (0.0, 1.0, 10.0, 100.0, 1000.0)
# rates below this are dropped: over 1e4 s they move less than 1e-26 of probability
RATE_FLOOR = 1e-30
TEMPS = (40.0, 60.0, 70.0, 80.0)
BOX_CAP = 100  # (2 * 100 + 1)^2 = 40401 states
TOL_ABS = 1e-6  # box, rtol, mass, negativity
TOL_REL = 1e-3  # Radau drift against a closed form, relative


def images(dx, dy):
    return {(dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy), (dy, dx), (-dy, dx), (dy, -dx), (-dy, -dx)}


def pair_energy(v, dx, dy):
    key = tuple(sorted((abs(dx), abs(dy)), reverse=True))
    return v.get(key, 0.0)


def box_for(temp, t_max, e0=E0):
    """Half-width holding the initial histogram (|D| <= 5) plus eight free-walk
    standard deviations of each component of D after t_max."""
    sigma = math.sqrt(4.0 * NU * math.exp(-e0 / (KB * temp)) * t_max)
    return max(12, math.ceil(5 + 8 * sigma))


class Chain:
    def __init__(self, box, temp, e0=E0, v=V_PD100):
        self.box = box
        self.states = [(x, y) for x in range(-box, box + 1) for y in range(-box, box + 1) if (x, y) != (0, 0)]
        self.index = {s: i for i, s in enumerate(self.states)}
        n = len(self.states)
        kt = KB * temp
        energy = [pair_energy(v, x, y) for x, y in self.states]
        rows, cols, vals = [], [], []
        out = np.zeros(n)
        self.dropped = 0.0
        for i, (x, y) in enumerate(self.states):
            for ex, ey in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = self.index.get((x + ex, y + ey))
                if j is None:
                    continue  # reflecting box edge, or the forbidden D = 0
                k = 2.0 * NU * math.exp(-(e0 + (energy[j] - energy[i]) / 2.0) / kt)
                if k < RATE_FLOOR:
                    self.dropped = max(self.dropped, k)
                    continue
                rows.append(i)
                cols.append(j)
                vals.append(k)
                out[i] += k
        rows += list(range(n))
        cols += list(range(n))
        vals += list(-out)
        self.q = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        self.qt = self.q.T.tocsr()
        self.parity = {
            0: np.array([1.0 if (x + y) % 2 == 0 else 0.0 for x, y in self.states]),
            1: np.array([1.0 if y % 2 == 0 else 0.0 for x, y in self.states]),
            2: np.array([1.0 if x % 2 == 0 else 0.0 for x, y in self.states]),
        }

    def initial(self, hist):
        p = np.zeros(len(self.states))
        tot = sum(hist.values())
        for (dx, dy), w in hist.items():
            ims = images(dx, dy)
            for im in ims:
                p[self.index[im]] += w / tot / len(ims)
        return p

    def evolve(self, p0, times, rtol=1e-11):
        """p(t) for each t in times (increasing, first may be 0), dp/dt = Q^T p,
        by the implicit Radau IIA integrator with the exact sparse Jacobian."""
        ts = [t for t in times if t > 0]
        out = {0.0: p0.copy()} if 0.0 in times else {}
        if ts:
            sol = scipy.integrate.solve_ivp(lambda t, p: self.qt @ p, (0.0, ts[-1]), p0, method="Radau",
                                            t_eval=ts, jac=self.qt, rtol=rtol, atol=1e-14)
            assert sol.success, sol.message
            for k, t in enumerate(ts):
                out[t] = sol.y[:, k]
        return out

    def ratios(self, p):
        return {f: 2.0 * float(p @ self.parity[f]) for f in (0, 1, 2)}

    def mass_near(self, p, radius):
        return float(sum(pi for pi, (x, y) in zip(p, self.states) if max(abs(x), abs(y)) >= radius))


def class_weights(chain, p):
    out = {}
    for pi, (x, y) in zip(p, chain.states):
        key = tuple(sorted((abs(x), abs(y)), reverse=True))
        out[key] = out.get(key, 0.0) + pi
    return out


def analytic_checks():
    """Relative misfit of Radau against two closed forms; both drifts are
    below 1e-3 in absolute terms, so an absolute tolerance would not see them."""
    worst = 0.0
    # 40 K, Lin & Jiang: once (1,0) has emptied (1e-9 s), the only motion within
    # 1000 s is (1,1) -> (2,1), two relative channels at 2 k(dE = -128) each.
    # (1,1) carries F0 = 2, F1 = 0; (2,1) carries F0 = 0, F1 = 1.
    ch = Chain(6, 40.0)
    p = ch.evolve(ch.initial(LJ), (1.0, 1000.0))
    w11 = LJ[(1, 1)] / sum(LJ.values())
    rate = 4.0 * NU * math.exp(-(E0 - 64.0) / (KB * 40.0))
    moved = w11 * (math.exp(-rate * 1.0) - math.exp(-rate * 1000.0))
    r1, r2 = ch.ratios(p[1.0]), ch.ratios(p[1000.0])
    for f, per in ((0, -2.0), (1, 1.0)):
        exact = per * moved
        mis = abs((r2[f] - r1[f]) / exact - 1.0)
        print(f"# check 40 K LJ drift F{f}: Radau {r2[f] - r1[f]:+.4e}  closed form {exact:+.4e}  misfit {mis:.1e}")
        worst = max(worst, mis)
    # 60 K, (4,0): out of interaction range every hop flips F0 and y-hops flip F1;
    # the relative coordinate hops at 8 k_free, so R_F0 = 1 + exp(-16 k t), R_F1 = 1 + exp(-8 k t).
    ch = Chain(12, 60.0)
    t = 100.0
    r = ch.ratios(ch.evolve(ch.initial(BR), (t,))[t])
    k = NU * math.exp(-E0 / (KB * 60.0))
    for f, exact in ((0, 1.0 + math.exp(-16.0 * k * t)), (1, 1.0 + math.exp(-8.0 * k * t))):
        mis = abs((2.0 - r[f]) / (2.0 - exact) - 1.0)
        print(f"# check 60 K (4,0) at {t:g} s F{f}: Radau 2 - R = {2.0 - r[f]:.4e}  closed form {2.0 - exact:.4e}  misfit {mis:.1e}")
        worst = max(worst, mis)
    return worst


def rtol_check():
    """Largest change in R when rtol drops from 1e-11 to 1e-13, over the fastest-moving cases."""
    worst = 0.0
    for temp in (70.0, 80.0):
        ch = Chain(box_for(temp, TIMES[-1]), temp)
        for hist in (LJ, BR):
            p0 = ch.initial(hist)
            a, b = ch.evolve(p0, TIMES), ch.evolve(p0, TIMES, rtol=1e-13)
            for t in TIMES:
                ra, rb = ch.ratios(a[t]), ch.ratios(b[t])
                worst = max(worst, max(abs(ra[f] - rb[f]) for f in ra))
    print(f"# check rtol 1e-11 against 1e-13 at 70 and 80 K: {worst:.1e}")
    return worst


def predictions():
    print("# EXACT TWO-BODY PREDICTIONS, infinite dilution, pair-only O/Pd(100) energies")
    print(f"# E0 = {E0} meV, nu = {NU:g} /s, V = {V_PD100}, reflecting box sized by box_for(T, {TIMES[-1]:g} s)")
    print("# columns: T/K  picture  t/s  R_F0  R_F1(=R_F2)  P(2,0)  P(1,0)  P(1,1)  P(2,1)  P(3,0)")
    worst = 0.0
    for temp in TEMPS:
        box = box_for(temp, TIMES[-1])
        ch = Chain(box, temp)
        big = Chain(box + 6, temp)
        for name, hist in (("LJ", LJ), ("BR40", BR)):
            ps = ch.evolve(ch.initial(hist), TIMES)
            pb = big.evolve(big.initial(hist), TIMES)
            for t in TIMES:
                p = ps[t]
                r = ch.ratios(p)
                rb = big.ratios(pb[t])
                dev_box = max(abs(r[f] - rb[f]) for f in r)
                mass = abs(float(p.sum()) - 1.0)
                neg = float(-min(0.0, p.min()))
                worst = max(worst, dev_box, mass, neg)
                cw = class_weights(ch, p)
                print(f"{temp:5.1f} {name:<5} {t:7.1f}  {r[0]:.4f}  {r[1]:.4f}  {cw.get((2, 0), 0):.4f}  "
                      f"{cw.get((1, 0), 0):.4f}  {cw.get((1, 1), 0):.4f}  {cw.get((2, 1), 0):.4f}  {cw.get((3, 0), 0):.4f}"
                      f"   # box {box} box+6 {dev_box:.0e} mass {mass:.0e} neg {neg:.0e}"
                      f" edge {ch.mass_near(p, box - 2):.0e}")
                assert abs(r[1] - r[2]) < 1e-8
        print(f"# T = {temp}: largest dropped rate {ch.dropped:.1e} /s")
    worst = max(worst, rtol_check())
    rel = analytic_checks()
    print(f"# largest of: box vs box+6, rtol, mass loss, negativity = {worst:.1e} (tolerance {TOL_ABS:g})")
    print(f"# largest relative misfit against closed forms = {rel:.1e} (tolerance {TOL_REL:g})")
    ok = worst < TOL_ABS and rel < TOL_REL
    print("PAIR_ME_OK" if ok else "PAIR_ME_FAILED")
    return ok


def window():
    print("# R(1/2,1/2) and R(1/2,0) after 1000 s against T; the pictures' difference")
    worst = 0.0
    for label, e0, v in (("baseline", E0, V_PD100),
                         ("E0 = 200", 200.0, V_PD100),
                         ("E0 = 300", 300.0, V_PD100),
                         ("V(2,0) = 0", E0, {(1, 0): 360.0, (1, 1): 128.0}),
                         ("V(2,0) = -96", E0, {(1, 0): 360.0, (1, 1): 128.0, (2, 0): -96.0})):
        print(f"## {label}")
        last_sep = None
        for temp in range(30, 101, 5):
            box = box_for(float(temp), 1000.0, e0)
            if box > BOX_CAP:
                print(f"  T={temp:3d}  not computed: the free walk needs box {box} > {BOX_CAP}")
                continue
            ch = Chain(box, float(temp), e0=e0, v=v)
            big = Chain(box + 6, float(temp), e0=e0, v=v)
            r = {}
            for name, hist in (("LJ", LJ), ("BR", BR)):
                a = ch.ratios(ch.evolve(ch.initial(hist), (1000.0,))[1000.0])
                b = big.ratios(big.evolve(big.initial(hist), (1000.0,))[1000.0])
                worst = max(worst, max(abs(a[f] - b[f]) for f in a))
                r[name] = a
            rl, rb = r["LJ"], r["BR"]
            sep = rb[0] - rl[0]
            if sep > 0.5:
                last_sep = temp
            print(f"  T={temp:3d}  LJ {rl[0]:.3f} {rl[1]:.3f}   (4,0) {rb[0]:.3f} {rb[1]:.3f}   "
                  f"(4,0) - LJ in (1/2,1/2): {sep:+.3f}   # box {box}")
        print(f"  highest T on this grid with (4,0) - LJ > 0.5 in (1/2,1/2): {last_sep} K")
    print(f"# largest box vs box+6 deviation in the scan = {worst:.1e} (tolerance {TOL_ABS:g})")
    ok = worst < TOL_ABS
    print("WINDOW_OK" if ok else "WINDOW_FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if (window() if "--window" in sys.argv else predictions()) else 1)
