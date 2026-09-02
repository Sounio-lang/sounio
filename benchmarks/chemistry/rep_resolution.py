#!/usr/bin/env python3
"""Decompose a replica-vs-Cantera residual into measured parts.

Three questions this answers, none by inference:

 1. ORACLE RESOLUTION AT THE CONFIGURATION ACTUALLY USED.  Comparing CVODE's
    default tolerance against rtol=1e-12 measures the distance between two
    settings, not the resolution of the one in use.  Running rtol=1e-12 against
    1e-13 and 1e-14 measures the floor: how much the oracle's own answer still
    moves once the setting is the one the harness pins.  A residual at or below
    that floor is not agreement, it is the oracle's noise.

 2. STEP BISECTION IN WHATEVER REGIME THE LOADED MODULES ARE IN.  In the
    published regime a 2.66e-06 constant offset dominates and hides everything
    else; the informative bisection is the one run where the offset is gone.

 3. THE TRUNCATION CURVE IN TIME.  RK4 truncation at a pre-front checkpoint
    says nothing about the integrator's behaviour where the trajectory is
    stiff.  If truncation is ~1e-14 before the front and orders larger inside
    it, the checkpoint was chosen where the integrator is not being tested.

Usage:
    python3 benchmarks/chemistry/rep_resolution.py               # this tree
    python3 benchmarks/chemistry/rep_resolution.py --dir PATH    # another pair
"""
import argparse
import importlib.util
import os
import sys

import cantera as ct

HERE = os.path.dirname(os.path.abspath(__file__))
T_CHECK = 1500.0
REPORT = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2"]
ALL10 = REPORT + ["N2", "AR"]


def _sib(name):
    """Sibling module: ../oracles/ in the frozen snapshot, beside it upstream."""
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


def worst(a, b, species):
    w, who = 0.0, ""
    for s in species:
        if s not in b or not b[s]:
            continue
        d = abs(a[s] - b[s]) / abs(b[s])
        if d > w:
            w, who = d, s
    return w, who


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="directory holding gri30_h2_cantera_parity.py and "
                         "gri30_h2_python_replica.py (default: beside this file)")
    args = ap.parse_args()

    d = args.dir or os.path.dirname(_sib("gri30_h2_cantera_parity.py"))
    cwd = os.getcwd()
    os.chdir(d)
    can = _load("can_res", os.path.join(d, "gri30_h2_cantera_parity.py"))
    rep = _load("rep_res", os.path.join(d, "gri30_h2_python_replica.py"))
    os.chdir(cwd)

    if not hasattr(can, "initial_concentrations"):
        raise RuntimeError(
            "the oracle exposes no initial_concentrations(): this is the TPX "
            "variant.  Refusing to report against a protocol the replica does "
            "not share.")

    want = can.initial_concentrations(T_CHECK)
    regime = "aligned" if abs(sum(want.values())
                              - 101325.0 / 8.31446261815324 / T_CHECK * 1e-6
                              - 1e-11) < 1e-14 else "as loaded"
    print(f"modules from : {d}")
    print(f"cantera      : {ct.__version__}")

    def cantera_at(rtol, atol, t_end):
        gas = can.build_submechanism()
        can.initial_state(gas, T_CHECK)
        r = ct.IdealGasReactor(gas, energy="off", clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = rtol, atol
        net.advance(t_end)
        return {s: gas.concentrations[gas.species_index(s)] * 1e-3
                for s in can.SUB_SPECIES}

    kcs = [rep.kc_cm(r, T_CHECK) for r in range(rep.NR)]
    c0 = [want.get(s, 0.0) for s in rep.SP]

    def replica_at(dt, t_end):
        c = list(c0)
        for _ in range(int(round(t_end / dt))):
            c = rep.rk4_step(c, T_CHECK, dt, kcs)
        return {rep.SP[i]: c[i] for i in range(rep.NSP)}

    # ---- 1. oracle resolution at the configuration used -------------------
    print()
    print("=" * 78)
    print("1. ORACLE RESOLUTION at the setting the harness pins (t = 1e-4 s)")
    print("=" * 78)
    ladder = [(1e-12, 1e-22), (1e-13, 1e-24), (1e-14, 1e-26)]
    runs = {r: cantera_at(r, a, 1e-4) for r, a in ladder}
    print(f"{'comparison':38s} {'worst rel':>11s}  {'on':>5s}")
    pairs = [((1e-12, 1e-13)), ((1e-13, 1e-14)), ((1e-12, 1e-14))]
    floor = 0.0
    for a, b in pairs:
        w, who = worst(runs[a], runs[b], ALL10)
        floor = max(floor, w) if (a, b) != (1e-12, 1e-14) else floor
        print(f"{'rtol=%.0e vs rtol=%.0e' % (a, b):38s} {w:11.3e}  {who:>5s}")
    print()
    print(f"-> the oracle's own answer at rtol=1e-12 is uncertain at the "
          f"{floor:.3e} level.")
    print("   A replica-vs-Cantera residual at or below this is the oracle's")
    print("   noise, not agreement.")

    # ---- 2. step bisection in this regime ---------------------------------
    print()
    print("=" * 78)
    print("2. STEP BISECTION in this regime (t = 1e-4 s, oracle at rtol=1e-12)")
    print("=" * 78)
    ref = runs[1e-12]
    dts = (1e-8, 5e-9, 2.5e-9)
    traj = {dt: replica_at(dt, 1e-4) for dt in dts}
    print(f"{'sp':6s} " + " ".join(f"{'dev dt=%.2g' % dt:>13s}" for dt in dts)
          + f" {'r(1,2)':>8s} {'r(2,3)':>8s}")
    for s in REPORT:
        dev = [abs(traj[dt][s] - ref[s]) / abs(ref[s]) for dt in dts]
        r12 = dev[0] / dev[1] if dev[1] else float("nan")
        r23 = dev[1] / dev[2] if dev[2] else float("nan")
        print(f"{s:6s} " + " ".join(f"{x:13.3e}" for x in dev)
              + f" {r12:8.3f} {r23:8.3f}")
    print()
    print("   ratio 16 = fourth-order truncation;  ratio 1 = a fixed offset;")
    print("   ratio erratic near the floor = the comparison is at the noise level.")

    # ---- 3. truncation curve in time --------------------------------------
    print()
    print("=" * 78)
    print("3. RK4 TRUNCATION vs TIME  |c(dt) - c(dt/2)| / |c|,  dt = 1e-8")
    print("=" * 78)
    print(f"{'t (s)':>10s} {'worst':>12s}  {'on':>5s}   regime")
    for t_end, label in ((1e-6, "early induction"),
                         (1e-5, "induction"),
                         (1e-4, "pre-front checkpoint"),
                         (1.2e-4, "approaching the front"),
                         (1.3e-4, "inside the front")):
        a = replica_at(1e-8, t_end)
        b = replica_at(5e-9, t_end)
        w, who = worst(a, b, REPORT)
        print(f"{t_end:10.2e} {w:12.3e}  {who:>5s}   {label}")
    print()
    print("   If truncation is orders larger inside the front than at the")
    print("   checkpoint, the checkpoint is where the integrator is NOT tested.")

    # ---- 4. separating truncation from roundoff ---------------------------
    print()
    print("=" * 78)
    print("4. TRUNCATION vs ROUNDOFF in the replica's own self-differences")
    print("=" * 78)
    print("   RK4 truncation falls 16x per halving.  Roundoff accumulated over N")
    print("   steps GROWS as sqrt(N), i.e. by ~1.41x per halving.  The observed")
    print("   ratio says which dominates at this step.")
    a, b, c = (traj[1e-8], traj[5e-9], traj[2.5e-9])
    print(f"{'sp':6s} {'|c(1e-8)-c(5e-9)|':>18s} {'|c(5e-9)-c(2.5e-9)|':>20s} {'ratio':>8s}")
    for s_ in REPORT:
        d1 = abs(a[s_] - b[s_]) / abs(b[s_])
        d2 = abs(b[s_] - c[s_]) / abs(c[s_])
        print(f"{s_:6s} {d1:18.3e} {d2:20.3e} "
              f"{(d1 / d2 if d2 else float('nan')):8.3f}")
    print()
    print("   ratio ~16      : truncation dominates, the step is the limit")
    print("   ratio << 1 and : roundoff dominates -- halving the step ADDS")
    print("     erratic          error, and differences of a few ULP have no")
    print("                      stable ratio.  Truncation is then unmeasurable")
    print("                      at this step, only bounded above by the")
    print("                      self-difference itself.")
    print("   ratio ~1       : neither -- the difference is a fixed offset")


if __name__ == "__main__":
    main()
