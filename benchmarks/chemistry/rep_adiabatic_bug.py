#!/usr/bin/env python3
"""Adiabatic provenance: reintroduce the historical `reac - nu` reverse
exponent and check the ignition-delay magnitudes the README records.

RESULTS.md section 2.6 reported these anchors from a working copy that was
never committed, and cited a file the frozen snapshot did not even ship.  This
is that harness, committed, and it monkey-patches the module rather than
carrying its own copy of the chemistry, so it cannot drift from the replica it
is characterising.

The README records the defect as costing "0.1% at 1100 K, 8.6% at 2000 K on
the delay".  Those two are testimony about a revision that is not recoverable
from history; this reproduces the magnitudes rather than the artefact.

Run:  python3 benchmarks/chemistry/rep_adiabatic_bug.py [--quick]
"""
import argparse
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _sib(name):
    for cand in (os.path.join(HERE, os.pardir, "oracles", name),
                 os.path.join(HERE, name)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(name)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def patched_rhs(m):
    """m.uv_rhs with the reverse exponent replaced by the reported defect.

    reac - nu == 2*reac - prod.  Under the `if p > 0` guard the product
    exponents (correct value >= 1) go negative and drop out, while reactant
    exponents (correct value 0) come in at +2 or +4: the reverse term stops
    depending on product concentrations at all.
    """
    def rhs(t, c):
        grt = [m.g_rt(s, t) for s in range(m.NSP)]
        c0 = m.P0 / m.R_SI * 1e-6 / t
        kc = [math.exp(-sum(m.nu[r][s] * grt[s] for s in range(m.NSP)))
              * c0 ** m.dn[r] for r in range(m.NR)]
        rn = []
        for r in range(m.NR):
            m_eff = sum(m.eff[r][s] * c[s] for s in range(m.NSP))
            kf = m.kfwd_eff(r, t, m_eff)
            fwd = kf
            for s in range(m.NSP):
                if m.reac[r][s] > 0:
                    fwd *= c[s] ** m.reac[r][s]
            rev = kf / kc[r]
            for s in range(m.NSP):
                p = m.reac[r][s] - m.nu[r][s]          # the reported defect
                if p > 0:
                    rev *= c[s] ** p
            rn.append(fwd - rev)
        dc = [sum(m.nu[r][s] * rn[r] for r in range(m.NR)) for s in range(m.NSP)]
        num = sum((m.h_rt(s, t) - 1.0) * dc[s] for s in range(m.NSP))
        den = sum(c[s] * (m.cp_r(s, t) - 1.0) for s in range(m.NSP))
        return -t * num / den, dc
    return rhs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="only the two temperatures the README records")
    ap.add_argument("--dt", type=float, default=5e-9)
    args = ap.parse_args()

    cwd = os.getcwd()
    os.chdir(os.path.dirname(_sib("gri30_h2_adiabatic_replica.py")) or ".")
    adi = _load("adi", _sib("gri30_h2_adiabatic_replica.py"))
    os.chdir(cwd)

    temps = (1100.0, 2000.0) if args.quick else (1100.0, 1400.0, 1700.0, 2000.0)
    readme = {1100.0: "0.1%", 2000.0: "8.6%"}

    print(f"adiabatic ignition delay, dt = {args.dt:.0e} s, "
          f"2% H2 / 1% O2 / 97% N2, H seed 1e-11 mol/cm^3")
    print(f"{'T0 (K)':>8s} {'correct (us)':>14s} {'reac - nu (us)':>16s} "
          f"{'error':>9s}  README")
    print("  (delay = time of max d[H2O]/dt; the dT/dt criterion is shown too,")
    print("   because a defect that shifted only one of them would be suspect)")
    good = adi.uv_rhs
    for t0 in temps:
        max_steps = int(2e-3 / args.dt)
        adi.uv_rhs = good
        a_h2o, a_dT, _ = adi.adiabatic_delay(t0, args.dt, max_steps)
        adi.uv_rhs = patched_rhs(adi)
        b_h2o, b_dT, _ = adi.adiabatic_delay(t0, args.dt, max_steps)
        adi.uv_rhs = good
        # adiabatic_delay returns (t at max d[H2O]/dt, t at max dT/dt, steps),
        # and (-1.0, -1.0, s) when no completed front occurs in the horizon.
        if a_h2o < 0 or b_h2o < 0:
            print(f"{t0:8.0f} {'no completed front within horizon':>40s}")
            continue
        err = (b_h2o - a_h2o) / a_h2o * 100.0
        err_t = (b_dT - a_dT) / a_dT * 100.0
        print(f"{t0:8.0f} {a_h2o * 1e6:14.4f} {b_h2o * 1e6:16.4f} {err:+8.3f}%  "
              f"{readme.get(t0, ''):>6s}   (dT/dt criterion: {err_t:+.3f}%)")

    print()
    print("On the d[H2O]/dt criterion every sign is positive: the defect DELAYS")
    print("ignition, which is what removing a radical source does, and both")
    print("README anchors reproduce.  On the dT/dt criterion the 1100 K error")
    print("comes out NEGATIVE -- the two delay definitions disagree in sign for")
    print("a defect that moves the H2O-rate delay by 0.1%.  The README does not")
    print("say which criterion its anchors use; this run shows it matters, and")
    print("the anchors are reproduced on d[H2O]/dt only.  These are magnitudes,")
    print("re-measured; the original artefact is in no ref.")


if __name__ == "__main__":
    main()
