#!/usr/bin/env python3
"""Cantera 3.2 reference for the FULL GRI-Mech 3.0 mechanism (53 species,
325 reactions) -- the oracle for Sounio's stdlib/chemistry/gri30_full.sio
and for benchmarks/chemistry/gri30_full_python_replica.py.

Mirrors benchmarks/chemistry/gri30_h2_cantera_parity.py, but over the whole
mechanism rather than the H/O sub-mechanism, so that the full-mechanism
results have a reproduction path of their own.

Protocol (identical to the replica and to the Sounio module):

  2% H2 / 1% O2 / 97% N2 at T = 1500 K, total concentration exactly
  1/(82.057*T) mol/cm3, plus an ADDITIVE H-atom seed of 1e-11 mol/cm3.
  Isothermal, constant volume.  Checkpoints t = 4e-6 s and t = 2e-5 s.

  The reference fixed-step integrator uses RK4 with dt = 2e-9 (the full
  mechanism carries NNH <=> N2 + H at k ~ 3.3e8 /s at 1500 K, so the H/O
  module's dt = 1e-8 is outside the RK4 stability limit).  Cantera itself
  integrates adaptively with CVODE; dt is recorded for provenance only.

INITIALISATION -- read this before changing it.  The additive seed makes the
true initial pressure 101325.576758 Pa, not 101325 Pa.  Setting the state
through `TPX` renormalises the mole fractions and pins P = 101325 Pa
exactly, which shifts EVERY initial concentration by -5.692129e-06 relative
and is then amplified by chain branching to ~4e-05 in the radicals by the
pre-front checkpoint.  This script therefore sets unnormalised mole
fractions and then fixes the density (`TD`), which reproduces the intended
absolute concentrations to 0.0e+00 relative deviation.  See
benchmarks/chemistry/RESULTS.md, section "STEP 1".

Run:  python3 benchmarks/chemistry/gri30_full_cantera_parity.py
"""
import argparse
import json
import time

import cantera as ct

P0 = 101325.0
T_CHECK = 1500.0
DT_REPLICA = 2e-9          # provenance only; Cantera integrates adaptively
CHECKPOINTS = (4e-6, 2e-5)
SEED_C = 1e-11             # mol/cm^3, absolute (additive)
# the eight H/O species the parity tables are reported over
REPORT = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2"]


def initial_concentrations(gas, T):
    """Intended absolute initial concentrations, mol/cm^3 (Sounio protocol)."""
    mtot = 1.0 / (82.057 * T)
    c = {s: 0.0 for s in gas.species_names}
    c["H2"], c["O2"], c["N2"] = mtot * 0.02, mtot * 0.01, mtot * 0.97
    c["H"] = SEED_C
    return c


def set_state(gas, T):
    """Set T and the exact absolute concentrations WITHOUT renormalising."""
    c = initial_concentrations(gas, T)
    conc = [c[s] * 1e3 for s in gas.species_names]      # mol/cm^3 -> kmol/m^3
    total = sum(conc)
    gas.set_unnormalized_mole_fractions([x / total for x in conc])
    rho = sum(conc[i] * gas.molecular_weights[i] for i in range(gas.n_species))
    gas.TD = T, rho
    return gas


def verify_initial_state(gas, T):
    """Assert the initialisation reproduces the intended concentrations exactly."""
    want = initial_concentrations(gas, T)
    set_state(gas, T)
    worst = 0.0
    for s, w in want.items():
        if w == 0.0:
            continue
        got = gas.concentrations[gas.species_index(s)] * 1e-3
        worst = max(worst, abs(got - w) / w)
    return worst, gas.P


def integrate(gas, T, t_end):
    """Isothermal constant-volume integration; returns conc in mol/cm^3."""
    set_state(gas, T)
    r = ct.IdealGasReactor(gas, energy="off", clone=False)
    net = ct.ReactorNet([r])
    net.rtol = 1e-12
    net.atol = 1e-22
    net.advance(t_end)
    return gas.concentrations * 1e-3                    # kmol/m^3 -> mol/cm^3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    gas = ct.Solution("gri30.yaml")
    print(f"cantera {ct.__version__}")
    print(f"FULLMECH species={gas.n_species} reactions={gas.n_reactions}")
    assert gas.n_species == 53 and gas.n_reactions == 325, "not GRI-Mech 3.0"

    worst, p_init = verify_initial_state(gas, T_CHECK)
    print(f"INIT worst relative deviation from intended concentrations = {worst:.6e}")
    print(f"INIT pressure = {p_init:.6f} Pa  (renormalising TPX would give exactly {P0:.0f})")

    out = {"cantera_version": ct.__version__,
           "n_species": int(gas.n_species), "n_reactions": int(gas.n_reactions),
           "T_K": T_CHECK, "dt_replica_s": DT_REPLICA,
           "init_worst_rel_dev": worst, "init_pressure_Pa": p_init,
           "checkpoints": {}}

    for t_end in CHECKPOINTS:
        t0 = time.perf_counter()
        conc = integrate(gas, T_CHECK, t_end)
        wall = time.perf_counter() - t0
        print(f"\nCHECK T={T_CHECK:.0f}K t={t_end:.0e}s wall={wall:.3f}s "
              f"(RK4 reference dt={DT_REPLICA:.0e}, n={int(round(t_end/DT_REPLICA))})")
        rec = {}
        for name in REPORT:
            v = float(conc[gas.species_index(name)])
            rec[name] = v
            print(f"  {name:5s} {v:.17e}")
        # NNH is the delta-n = +1 species whose equilibrium pins its population
        for name in ("N2", "NNH"):
            v = float(conc[gas.species_index(name)])
            rec[name] = v
            print(f"  {name:5s} {v:.17e}")
        out["checkpoints"][f"{t_end:.0e}"] = {"wall_s": wall, "conc_mol_cm3": rec}

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
