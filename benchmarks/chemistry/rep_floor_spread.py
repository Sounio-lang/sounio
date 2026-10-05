#!/usr/bin/env python3
"""The oracle's resolution is a distribution, not a constant: measure it.

RESULTS.md section 7.7 partitioned the aligned residual using ONE value of
the oracle's floor (CVODE rtol=1e-12 vs 1e-13 at one initial state).  Section
6.3 (4) then found the oracle's tolerance spread changes by 7.7x between two
regimes whose initial densities differ by 5.7e-06.  So the single floor is a
sample from a distribution, and the partition inherits that fragility.

This measures the floor over N initial states with the total density scaled
by (1 + delta), delta uniformly spaced in [-1e-6, +1e-6], fresh gas object per
run, both tolerance settings per state, worst relative difference over all
ten species.  It reports the interval, and the partition of the residual as a
range rather than a number.

Run:  python3 benchmarks/chemistry/rep_floor_spread.py [--n 10] [--eps 1e-6]
"""
import argparse
import importlib.util
import os
import sys

import cantera as ct

HERE = os.path.dirname(os.path.abspath(__file__))
T_CHECK, T_END = 1500.0, 1e-4


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--residual", type=float, default=2.074e-11,
                    help="aligned replica-vs-oracle residual to partition (section 7.5b)")
    args = ap.parse_args()

    cwd = os.getcwd()
    os.chdir(os.path.dirname(_sib("gri30_h2_cantera_parity.py")) or ".")
    can = _load("can_fs", _sib("gri30_h2_cantera_parity.py"))
    os.chdir(cwd)
    if not hasattr(can, "initial_concentrations"):
        raise RuntimeError("oracle has no initial_concentrations(): TPX variant; refusing")

    base = can.initial_concentrations(T_CHECK)

    def run(scale, rtol, atol):
        gas = can.build_submechanism()
        # the oracle's own TDY protocol, with the intended density scaled
        conc = [base.get(s, 0.0) * scale * 1e3 for s in gas.species_names]
        tot = sum(conc)
        gas.set_unnormalized_mole_fractions([x / tot for x in conc])
        rho = sum(conc[i] * gas.molecular_weights[i] for i in range(gas.n_species))
        gas.TD = T_CHECK, rho
        r = ct.IdealGasReactor(gas, energy="off", clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = rtol, atol
        net.advance(T_END)
        return {s: gas.concentrations[gas.species_index(s)] * 1e-3
                for s in can.SUB_SPECIES}

    print(f"oracle floor = worst |c(rtol=1e-12) - c(rtol=1e-13)| / |c| over 10 species,")
    print(f"at {args.n} initial states, density scaled by 1+delta, delta in [-{args.eps:g}, +{args.eps:g}]")
    print(f"{'delta':>12s} {'floor':>12s}  {'on':>5s}")
    floors = []
    for k in range(args.n):
        delta = -args.eps + 2 * args.eps * k / (args.n - 1)
        a = run(1 + delta, 1e-12, 1e-22)
        b = run(1 + delta, 1e-13, 1e-24)
        w, who = 0.0, ""
        for s in can.SUB_SPECIES:
            if not b[s]:
                continue
            d = abs(a[s] - b[s]) / abs(b[s])
            if d > w:
                w, who = d, s
        floors.append(w)
        print(f"{delta:12.2e} {w:12.3e}  {who:>5s}")
    lo, hi = min(floors), max(floors)
    print()
    print(f"floor interval : [{lo:.3e}, {hi:.3e}]   ratio hi/lo = {hi / lo:.2f}")
    res = args.residual
    print(f"residual       : {res:.3e}  (section 7.5b, worst species)")
    print(f"oracle explains: between {lo / res * 100:.0f}% and {hi / res * 100:.0f}% of the residual")
    if hi >= res:
        print("NOTE: at the top of the interval the oracle's floor EXCEEDS the residual --")
        print("      the partition is not even defined there; the residual is inside the noise.")
    print()
    print("An adaptive integrator's resolution is not a smooth function of the initial")
    print("state: the step sequence re-selects under perturbation. It therefore has no")
    print("value independent of the regime, and must be measured in regime, every time.")


if __name__ == "__main__":
    main()
