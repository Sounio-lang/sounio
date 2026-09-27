#!/usr/bin/env python3
"""Cantera 3.2 ADIABATIC (constant-U,V) reference for the Sounio gri30_h2
shock-tube benchmark — the UV counterpart of gri30_h2_cantera_parity.py.

Same H/O sub-mechanism (10 species, 29 reactions) built from Cantera's own
gri30.yaml, same mixture/seed protocol as the isothermal work:

  2% H2 / 1% O2 / 97% N2, 1 atm initial, H-atom seed 1e-11 mol/cm^3,

now in an IdealGasConstVolumeReactor with energy="on" (sensible energy
conservation through the ignition front), CVODE adaptive stepping.

Delay definition is identical to the Sounio module and the Python replica:
mid-step time of max d[H2O]/dt (primary) and max dT/dt (secondary), from
finite differences of the sampled trajectory. Sampling at the Sounio RK4
step (dt = 1e-8 s) so the finite-difference delay is directly comparable.

Outputs (machine-checkable):
  - NASA-7 cp/R and h/(RT) spot values at 900/1500 K (Sounio test pins)
  - adiabatic delays for T0 = 1000..2000 K (1000/T Arrhenius table)

Run:  PYTHONPATH=/tmp/pylibs python3 benchmarks/chemistry/gri30_h2_adiabatic_cantera.py
"""
import re
import time
import cantera as ct

SUB_SPECIES = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2", "AR"]
SUB_SET = set(SUB_SPECIES)
P0 = 101325.0
R_SI = 8.31446261815324  # J/(mol*K)
DT = 1e-8
SEED_C = 1e-11  # mol/cm^3, absolute (matches the Sounio test protocol)


def equation_species(eq):
    eq = eq.split("#", 1)[0]
    eq = eq.replace("(+M)", " ").replace("<=>", " + ").replace("=>", " + ")
    toks = set()
    for tok in eq.split("+"):
        tok = re.sub(r"^\s*\d+(\.\d+)?\s*", "", tok).strip()
        if tok and tok != "M":
            toks.add(tok)
    return toks


def build_submechanism():
    full = ct.Solution("gri30.yaml")
    keep_species = [s for s in full.species() if s.name in SUB_SET]
    keep_rxns = [r for r in full.reactions()
                 if equation_species(r.equation) <= SUB_SET]
    return ct.Solution(thermo="ideal-gas", kinetics="gas",
                       species=keep_species, reactions=keep_rxns)


def initial_state(gas, T):
    mtot = P0 / (R_SI * T) * 1e-6  # mol/cm^3 at 1 atm
    X = {s: 0.0 for s in SUB_SPECIES}
    X["H2"], X["O2"], X["N2"] = 0.02, 0.01, 0.97
    X["H"] = SEED_C / mtot  # absolute seed -> mole fraction
    gas.TPX = T, P0, X
    return gas


def adiabatic_delay(gas, T0, max_time):
    """UV reactor delay: (t_h2o, t_dT) mid-step maxima; (-1, -1) if no front."""
    initial_state(gas, T0)
    # IdealGasReactor is Cantera's constant-V ideal-gas reactor; energy="on"
    # conserves sensible internal energy through the ignition front.
    r = ct.IdealGasReactor(gas, energy="on")
    net = ct.ReactorNet([r])
    ih2o = gas.species_index("H2O")
    t, prev, prev_t, prev_c = 0.0, 0.0, None, None
    best_h2o, best_h2o_t = 0.0, None
    best_dt, best_dt_t = 0.0, None
    while t < max_time:
        net.advance(t + DT)
        t = net.time
        c = gas.concentrations[ih2o]
        tt = gas.T
        if prev_c is not None:
            rate_h2o = (c - prev_c) / (t - prev)
            rate_t = (tt - prev_t) / (t - prev)
            if rate_h2o > best_h2o:
                best_h2o, best_h2o_t = rate_h2o, 0.5 * (t + prev)
            if rate_t > best_dt:
                best_dt, best_dt_t = rate_t, 0.5 * (t + prev)
            if (best_h2o_t is not None and rate_h2o < 0.05 * best_h2o
                    and t > 2 * best_h2o_t):
                return best_h2o_t, best_dt_t
        prev, prev_t, prev_c = t, tt, c
    return -1.0, -1.0


def main():
    gas = build_submechanism()
    print(f"SUBMECH species={gas.n_species} reactions={gas.n_reactions}")
    assert gas.n_species == 10 and gas.n_reactions == 29, "sub-mechanism mismatch"

    print("NASA-7 spot values (Cantera):")
    R = ct.gas_constant  # J/(kmol K)
    for name, t in (("H2", 1500.0), ("O2", 1500.0), ("H2O", 1500.0), ("N2", 1500.0),
                    ("H", 1500.0), ("H2", 900.0), ("H2O", 900.0)):
        gas.TP = t, P0
        k = gas.species_index(name)
        cps = gas.partial_molar_cp[k] / R
        hs = gas.partial_molar_enthalpies[k] / (R * t)
        print(f"  cp_r({name:4s},{t:6.0f}K) = {cps:.6f}   h_rt({name},{t:.0f}K) = {hs:.6f}")

    print("ADIABATIC DELAYS (2%H2/1%O2/97%N2, 1 atm, seed H=1e-11, UV reactor)")
    print("  T0(K)  1000/T    t_h2o(us)  t_dT(us)")
    for t0 in range(1000, 2001, 100):
        # uniform 3 ms horizon: the early-exit certificate needs t > 2*peak,
        # so a shorter horizon can false-negative slow-but-real ignitions
        max_time = 3e-3
        th, tt = adiabatic_delay(gas, float(t0), max_time)
        if th > 0:
            print(f"  {t0:5d}  {1000.0/t0:7.4f}  {th*1e6:9.2f}  {tt*1e6:9.2f}")
        else:
            print(f"  {t0:5d}  {1000.0/t0:7.4f}  no-ign<{max_time*1e6:.0f}us  no-ign")


if __name__ == "__main__":
    t_start = time.perf_counter()
    main()
    print(f"wall={time.perf_counter()-t_start:.1f}s")
