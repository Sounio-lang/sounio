#!/usr/bin/env python3
"""Step-stagnation test: does halving the step push per-step increments below
half an ULP, so that species FREEZE and the error grows with lost steps?

Motivated by RESULTS.md section 7.7: halving dt from 5e-9 to 2.5e-9 made the
replica's self-difference WORSE by 12x to 140x for five of eight species.  No
known mechanism gives that: random-walk roundoff gives sqrt(2) = 1.41x,
systematic roundoff gives 2x, fourth-order truncation gives 1/16.  The
hypothesis tested here is stagnation: at the checkpoint, if dt*|dc/dt| for a
species is below ulp(c)/2, the RK4 update c + dt*(...) rounds back to c and
the step is lost entirely; between 0.5 and a few ULP it is heavily quantised.

For each species, at dt = 1e-8, 5e-9 and 2.5e-9, this reports
    increment / half-ULP  =  dt * |dc/dt| / (ulp(c) / 2)
evaluated at that dt's own checkpoint state.  A value below 1 is a frozen
species; below ~10 is a quantised one.

Run:  python3 benchmarks/chemistry/rep_stagnation.py
"""
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T_CHECK, T_END = 1500.0, 1e-4
REPORT = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2"]


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


def main():
    cwd = os.getcwd()
    os.chdir(os.path.dirname(_sib("gri30_h2_python_replica.py")) or ".")
    rep = _load("rep_stag", _sib("gri30_h2_python_replica.py"))
    can = _load("can_stag", _sib("gri30_h2_cantera_parity.py"))
    os.chdir(cwd)
    if not hasattr(can, "initial_concentrations"):
        raise RuntimeError("oracle has no initial_concentrations(): TPX variant; refusing")

    want = can.initial_concentrations(T_CHECK)
    kcs = [rep.kc_cm(r, T_CHECK) for r in range(rep.NR)]
    c0 = [want.get(s, 0.0) for s in rep.SP]
    idx = {s: i for i, s in enumerate(rep.SP)}

    print(f"checkpoint T={T_CHECK:.0f} K, t={T_END:.0e} s;  ratio = dt*|dc/dt| / (ulp(c)/2)")
    print(f"{'sp':6s}" + "".join(f"{'dt=%.2g' % dt:>16s}" for dt in (1e-8, 5e-9, 2.5e-9))
          + f"   {'|dc/dt|/c (1/s)':>16s}")
    rows = {}
    for dt in (1e-8, 5e-9, 2.5e-9):
        c = list(c0)
        for _ in range(int(round(T_END / dt))):
            c = rep.rk4_step(c, T_CHECK, dt, kcs)
        dc = rep.dc_dt(T_CHECK, c, kcs)
        for s in REPORT:
            i = idx[s]
            half_ulp = math.ulp(c[i]) / 2.0
            rows.setdefault(s, {})[dt] = (dt * abs(dc[i]) / half_ulp, abs(dc[i]) / abs(c[i]))
    frozen = []
    for s in REPORT:
        r = rows[s]
        line = f"{s:6s}" + "".join(f"{r[dt][0]:16.3e}" for dt in (1e-8, 5e-9, 2.5e-9))
        line += f"   {r[1e-8][1]:16.3e}"
        flag = ""
        if r[2.5e-9][0] < 1.0:
            flag = "  <-- FROZEN at 2.5e-9"; frozen.append(s)
        elif r[2.5e-9][0] < 10.0:
            flag = "  <-- quantised at 2.5e-9"
        print(line + flag)
    print()
    print("ratio < 1  : the update rounds back to c -- the step is lost (stagnation)")
    print("ratio < 10 : the update is a handful of ULP -- quantised, error grows per step")
    print("ratio >> 10: the increment is well resolved; roundoff is the ordinary kind")
    print()
    if frozen:
        print(f"STAGNATION CONFIRMED for {', '.join(frozen)} at dt = 2.5e-9.")
        print("Halving the step past this point does not reduce error; it ADDS lost steps.")
        print("The self-difference floor at dt=1e-8 is then a CEILING on what refinement can")
        print("give, not a floor on the method's accuracy.")
    else:
        print("Stagnation NOT confirmed: every increment exceeds half an ULP at 2.5e-9.")
        print("The 12x-140x growth of the self-difference under halving is UNEXPLAINED.")


if __name__ == "__main__":
    main()
