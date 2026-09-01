#!/usr/bin/env python3
"""Instrument-resolution probe: how much of a replica-vs-Cantera deviation is
the ORACLE's own tolerance error, and how much is the replica's truncation?

Written because RESULTS.md section 6.3 instance (4) reported three numbers with
no committed producer, and on re-measurement the headline figure did not
reproduce (published 2.515e-08; measured 3.251e-09).  A claim about instrument
resolution that cannot itself be re-measured is the very failure it describes.

Reports, at the isothermal pre-front checkpoint:
  * Cantera against itself at default vs pinned tolerances -- the oracle's own
    resolution floor;
  * the replica's RK4 against Cantera at both settings;
  * the replica's RK4 self-convergence under step halving, which is the only
    quantity of the three that is actually RK4 truncation.

Run:  python3 benchmarks/chemistry/rep_tolerance.py
"""
import importlib.util
import os
import sys

import cantera as ct

HERE = os.path.dirname(os.path.abspath(__file__))


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


T_CHECK = 1500.0
T_END = 1e-4
REPORT = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2"]


def worst(a, b):
    w, who = 0.0, ""
    for s in REPORT:
        if not b[s]:
            continue
        d = abs(a[s] - b[s]) / abs(b[s])
        if d > w:
            w, who = d, s
    return w, who


def main():
    cwd = os.getcwd()
    os.chdir(os.path.dirname(_sib("gri30_h2_cantera_parity.py")) or ".")
    can = _load("can_tol", _sib("gri30_h2_cantera_parity.py"))
    rep = _load("rep_tol", _sib("gri30_h2_python_replica.py"))
    os.chdir(cwd)

    print(f"cantera {ct.__version__}")

    def cantera_at(rtol=None, atol=None):
        gas = can.build_submechanism()
        can.initial_state(gas, T_CHECK)
        r = ct.IdealGasReactor(gas, energy="off", clone=False)
        net = ct.ReactorNet([r])
        seen = (net.rtol, net.atol)
        if rtol is not None:
            net.rtol, net.atol = rtol, atol
        net.advance(T_END)
        return ({s: gas.concentrations[gas.species_index(s)] * 1e-3
                 for s in can.SUB_SPECIES}, seen)

    def replica_at(dt):
        mtot = can.initial_concentrations(T_CHECK) \
            if hasattr(can, "initial_concentrations") else None
        kcs = [rep.kc_cm(r, T_CHECK) for r in range(rep.NR)]
        c = [0.0] * rep.NSP
        if mtot is not None:
            for i, s in enumerate(rep.SP):
                c[i] = mtot.get(s, 0.0)
        else:  # the oracle exposes no intended-state helper: fall back to its protocol
            raise RuntimeError("oracle has no initial_concentrations(); cannot "
                               "guarantee the two sides share a protocol")
        n = int(round(T_END / dt))
        for _ in range(n):
            c = rep.rk4_step(c, T_CHECK, dt, kcs)
        return {rep.SP[i]: c[i] for i in range(rep.NSP)}, n

    ct_tight, _ = cantera_at(1e-12, 1e-22)
    ct_def, defaults = cantera_at()
    print(f"ReactorNet defaults: rtol={defaults[0]:.0e} atol={defaults[1]:.0e}; "
          f"oracle pins rtol=1e-12 atol=1e-22")

    r8, n8 = replica_at(1e-8)
    r5, n5 = replica_at(5e-9)

    w_oracle, s_oracle = worst(ct_def, ct_tight)
    w_def, s_def = worst(r8, ct_def)
    w_tight, s_tight = worst(r8, ct_tight)
    w_self, s_self = worst(r8, r5)

    print()
    print(f"{'comparison':52s} {'worst rel':>11s}  {'on':>5s}  what it measures")
    print("-" * 108)
    print(f"{'CVODE default vs CVODE rtol=1e-12':52s} {w_oracle:11.3e}  {s_oracle:>5s}  "
          f"the ORACLE's own resolution")
    print(f"{'RK4 dt=1e-8 vs CVODE default':52s} {w_def:11.3e}  {s_def:>5s}  "
          f"replica vs a loosely-set oracle")
    print(f"{'RK4 dt=1e-8 vs CVODE rtol=1e-12':52s} {w_tight:11.3e}  {s_tight:>5s}  "
          f"replica vs a tightly-set oracle")
    print(f"{'RK4 dt=1e-8 vs RK4 dt=5e-9':52s} {w_self:11.3e}  {s_self:>5s}  "
          f"RK4 TRUNCATION -- the only one that is")
    print()
    print("Reading: any replica-vs-Cantera figure below the first row is below the")
    print("oracle's own resolution and cannot be attributed to the replica.  RK4")
    print("truncation at this step is the last row, and if it is orders below the")
    print("second and third rows, the deviation they show is NOT truncation.")

    print()
    print("Truncation test -- halving dt.  RK4 is fourth order, so a deviation MADE")
    print("of truncation falls by 2^4 = 16.  A deviation that is a fixed offset does")
    print("not move at all.")
    print(f"{'sp':6s} {'|dev| dt=1e-8':>14s} {'|dev| dt=5e-9':>14s} {'ratio':>8s}")
    for s in REPORT:
        d1 = abs(r8[s] - ct_tight[s]) / abs(ct_tight[s])
        d2 = abs(r5[s] - ct_tight[s]) / abs(ct_tight[s])
        print(f"{s:6s} {d1:14.3e} {d2:14.3e} {(d1 / d2 if d2 else float('nan')):8.3f}")
    print(f"({n8} and {n5} RK4 steps respectively)")


if __name__ == "__main__":
    main()
