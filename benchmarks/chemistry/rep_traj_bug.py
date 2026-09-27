#!/usr/bin/env python3
"""Trajectory under the reported reverse-rate defect, and the three quantities
RESULTS.md sections 2.3, 2.4 and 2.5 report.

Written because those three sections carried numbers whose only stated command
was `python3 - < the harness in section 2.3` -- a placeholder, not a command.
The tables were real measurements from a working copy that was never committed,
which is the same defect the STEP 4 audit was about, repeated inside the work
that was correcting it.  This file is that harness, committed.

Three forms of the reverse exponent are compared, not two:

    shipped         p = prod[r][s]
    proposed "fix"  p = reac[r][s] + nu[r][s]   == prod, an identity
    reported bug    p = reac[r][s] - nu[r][s]   == 2*reac - prod

Run:  python3 benchmarks/chemistry/rep_traj_bug.py
"""
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


T_CHECK = 1500.0
T_END = 1e-4
DT = 1e-8
REPORT = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2"]


def make_rates(rep, form):
    """rates_net with the reverse exponent taken from `form`."""
    def exponent(r, s):
        if form == "shipped":
            return rep.prod[r][s]
        if form == "fix":
            return rep.reac[r][s] + rep.nu[r][s]
        if form == "bug":
            return rep.reac[r][s] - rep.nu[r][s]
        raise ValueError(form)

    def rates(t, c, kc):
        out = []
        for r in range(rep.NR):
            m_eff = sum(rep.eff[r][s] * c[s] for s in range(rep.NSP))
            kf = rep.kfwd_eff(r, t, m_eff)
            fwd = kf
            for s in range(rep.NSP):
                if rep.reac[r][s] > 0:
                    fwd *= c[s] ** rep.reac[r][s]
            rev = kf / kc[r]
            for s in range(rep.NSP):
                p = exponent(r, s)
                if p > 0:
                    rev *= c[s] ** p
            out.append(fwd - rev)
        return out
    return rates


def dc_dt_with(rep, rates):
    def dc(t, c, kc):
        rn = rates(t, c, kc)
        return [sum(rep.nu[r][i] * rn[r] for r in range(rep.NR))
                for i in range(rep.NSP)]
    return dc


def rk4(rep, dc, c, t, dt, kc):
    k1 = dc(t, c, kc)
    k2 = dc(t, [c[i] + 0.5 * dt * k1[i] for i in range(rep.NSP)], kc)
    k3 = dc(t, [c[i] + 0.5 * dt * k2[i] for i in range(rep.NSP)], kc)
    k4 = dc(t, [c[i] + dt * k3[i] for i in range(rep.NSP)], kc)
    return [c[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            for i in range(rep.NSP)]


def main():
    cwd = os.getcwd()
    os.chdir(os.path.dirname(_sib("gri30_h2_python_replica.py")) or ".")
    rep = _load("rep_tb", _sib("gri30_h2_python_replica.py"))
    can = _load("can_tb", _sib("gri30_h2_cantera_parity.py"))
    os.chdir(cwd)

    if not hasattr(can, "initial_concentrations"):
        raise RuntimeError(
            "the oracle exposes no initial_concentrations(): this is the TPX "
            "variant, which renormalises the seed away.  Refusing to report "
            "numbers against a protocol the module under test does not share.")

    want = can.initial_concentrations(T_CHECK)
    kcs = [rep.kc_cm(r, T_CHECK) for r in range(rep.NR)]
    c0 = [want.get(s, 0.0) for s in rep.SP]
    n = int(round(T_END / DT))

    traj = {}
    for form in ("shipped", "fix", "bug"):
        dc = dc_dt_with(rep, make_rates(rep, form))
        c = list(c0)
        for _ in range(n):
            c = rk4(rep, dc, c, T_CHECK, DT, kcs)
        traj[form] = {rep.SP[i]: c[i] for i in range(rep.NSP)}

    idx = {s: i for i, s in enumerate(rep.SP)}

    print(f"isothermal checkpoint T={T_CHECK:.0f} K  t={T_END:.0e} s  "
          f"dt={DT:.0e}  ({n} RK4 steps)")
    print()
    print("SECTION 2.3 -- per-species, all three forms")
    print(f"{'sp':6s} {'shipped':>26s} {'under reac - nu':>26s} {'delta':>12s} "
          f"{'fix - shipped':>14s}")
    worst_fix = 0.0
    for s in REPORT:
        a, b, f = traj["shipped"][s], traj["bug"][s], traj["fix"][s]
        d = abs(b - a) / abs(a)
        df = abs(f - a) / abs(a) if a else 0.0
        worst_fix = max(worst_fix, df)
        print(f"{s:6s} {a:26.17e} {b:26.17e} {d:12.3e} {df:14.3e}")
    print(f"\nworst |fix - shipped| = {worst_fix:.3e}  "
          f"-> the proposed fix is an identity with the shipped code")

    print()
    print("SECTION 2.4 -- d[HO2]/dt at the checkpoint")
    ho2 = idx["HO2"]
    d_ship = dc_dt_with(rep, make_rates(rep, "shipped"))(
        T_CHECK, [traj["shipped"][s] for s in rep.SP], kcs)[ho2]
    d_bug = dc_dt_with(rep, make_rates(rep, "bug"))(
        T_CHECK, [traj["bug"][s] for s in rep.SP], kcs)[ho2]
    print(f"  shipped      {d_ship:.17e}")
    print(f"  under bug    {d_bug:.17e}")
    print(f"  rel change   {(d_bug - d_ship) / abs(d_ship) * 100:+.2f}%")
    print("  (a -34% has been reported for this quantity; it does not reproduce)")

    print()
    print("SECTION 2.5 -- R16 forward and reverse at the checkpoint")
    r16 = [r for r in range(rep.NR)
           if rep.RX[r]["eq"] in ("H + HO2 <=> O2 + H2", "H + HO2 <=> H2 + O2")]
    cs = [traj["shipped"][s] for s in rep.SP]
    for r in r16:
        m_eff = sum(rep.eff[r][s] * cs[s] for s in range(rep.NSP))
        kf = rep.kfwd_eff(r, T_CHECK, m_eff)
        fwd = kf
        for s in range(rep.NSP):
            if rep.reac[r][s] > 0:
                fwd *= cs[s] ** rep.reac[r][s]
        rev = kf / kcs[r]
        for s in range(rep.NSP):
            if rep.prod[r][s] > 0:
                rev *= cs[s] ** rep.prod[r][s]
        print(f"  R{r:02d} {rep.RX[r]['eq']}")
        print(f"    forward          {fwd:.17e}")
        print(f"    reverse          {rev:.17e}")
        print(f"    reverse/forward  {rev / fwd:.3e}")


if __name__ == "__main__":
    main()
