#!/usr/bin/env python3
"""Cantera 3.2 reference for the Sounio gri30_h2 demo/benchmark.

Builds the H/O sub-mechanism of GRI-Mech 3.0 (10 species, 29 reactions —
every reaction whose equation mentions only species in
{H2,H,O,O2,OH,H2O,HO2,H2O2,N2,AR}, with M/(+M) treated as bath) directly from
Cantera's own gri30.yaml, then runs the exact same isothermal,
constant-density protocol as stdlib/chemistry/gri30_h2.sio:

  2% H2 / 1% O2 / 97% N2, 1 atm, T = 1500 K,
  H-atom seed 1e-11 mol/cm^3 (absolute), RK4-checkpoint t = 1e-4 s.

Note: this cross-check is what caught a missing reaction in the first version
of the Sounio module (2 O + M <=> O2 + M, GRI-Mech Reaction 1) — the module
now carries all 29.

Outputs (machine-checkable):
  - species/reaction counts of the filtered sub-mechanism (must be 10/29)
  - concentrations at t = 1e-4 s in mol/cm^3 (Sounio parity checkpoint)
  - isothermal ignition delays (max d[H2O]/dt) for T = 1400..1800 K
  - wall time for the t = 1e-4 integration (Sounio vs Cantera timing)

Run:  PYTHONPATH=/tmp/pylibs python3 benchmarks/chemistry/gri30_h2_cantera_parity.py
"""
import re
import time
import cantera as ct

SUB_SPECIES = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2", "AR"]
SUB_SET = set(SUB_SPECIES)
P0 = 101325.0
DT = 1e-8
T_END = 1e-4
T_CHECK = 1500.0
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


def initial_concentrations(T):
    """Intended absolute initial concentrations, mol/cm^3 (Sounio protocol)."""
    mtot = 1.0 / (82.057 * T)
    c = {s: 0.0 for s in SUB_SPECIES}
    c["H2"], c["O2"], c["N2"] = mtot * 0.02, mtot * 0.01, mtot * 0.97
    c["H"] = SEED_C  # additive H-atom seed, NOT taken out of the bath
    return c


def initial_state(gas, T):
    """Set T and the exact absolute concentrations WITHOUT renormalising.

    The additive seed makes the true initial pressure 101325.576758 Pa, not
    101325 Pa.  The previous implementation used `gas.TPX = T, P0, X`, which
    renormalises the mole fractions and pins P = 101325 Pa exactly; that
    shifts EVERY initial concentration by -5.692129e-06 relative, which chain
    branching then amplifies to ~3.9e-05 in the radicals by the t = 1e-4 s
    pre-front checkpoint -- about 15x the deviation of the protocol the
    README documents.  Setting unnormalised mole fractions and then fixing
    the density reproduces the intended concentrations to 0.0e+00 relative.
    Superseded 2026-09-01; see benchmarks/chemistry/RESULTS.md section 1.
    """
    c = initial_concentrations(T)
    conc = [c[s] * 1e3 for s in gas.species_names]  # mol/cm^3 -> kmol/m^3
    total = sum(conc)
    gas.set_unnormalized_mole_fractions([x / total for x in conc])
    rho = sum(conc[i] * gas.molecular_weights[i] for i in range(gas.n_species))
    gas.TD = T, rho
    return gas


def initial_state_deviation(gas, T):
    """Worst relative deviation of the realised initial state from intent."""
    want = initial_concentrations(T)
    initial_state(gas, T)
    worst = 0.0
    for s, w in want.items():
        if w == 0.0:
            continue
        got = gas.concentrations[gas.species_index(s)] * 1e-3
        worst = max(worst, abs(got - w) / w)
    return worst, gas.P


def integrate(gas, T, t_end, dt=DT):
    """Isothermal constant-density integration; returns conc in mol/cm^3."""
    initial_state(gas, T)
    r = ct.IdealGasReactor(gas, energy="off", clone=False)
    net = ct.ReactorNet([r])
    net.rtol = 1e-12
    net.atol = 1e-22
    net.advance(t_end)
    return gas.concentrations * 1e-3  # kmol/m^3 -> mol/cm^3


def ignition_delay(gas, T, t_max=5e-3, dt=DT):
    """Delay = time of max d[H2O]/dt (isothermal, seeded)."""
    initial_state(gas, T)
    r = ct.IdealGasReactor(gas, energy="off", clone=False)
    net = ct.ReactorNet([r])
    net.rtol = 1e-12
    net.atol = 1e-22
    ih2o = gas.species_index("H2O")
    t, prev, prev_c = 0.0, 0.0, None
    best_rate, best_t, rate = 0.0, None, 0.0
    while t < t_max:
        t = net.advance(t + dt)
        c = gas.concentrations[ih2o]
        if prev_c is not None:
            rate = (c - prev_c) / (t - prev)
            if rate > best_rate:
                best_rate, best_t = rate, 0.5 * (t + prev)
        prev, prev_c = t, c
        if best_t is not None and rate < 0.05 * best_rate and t > 2 * best_t:
            break
    return best_t


def main():
    gas = build_submechanism()
    print(f"SUBMECH species={gas.n_species} reactions={gas.n_reactions}")
    assert gas.n_species == 10 and gas.n_reactions == 29, "sub-mechanism mismatch"

    worst, p_init = initial_state_deviation(gas, T_CHECK)
    print(f"INIT worst relative deviation from intended concentrations = {worst:.6e}")
    print(f"INIT pressure = {p_init:.6f} Pa (renormalising TPX would pin {P0:.0f})")
    # A few ULP, not exact zero.  With the unaligned 1/(82.057*T) constant the
    # TDY round trip happens to land exactly on the intended concentrations;
    # with the aligned P0/(R_SI*T)*1e-6 it lands within one ULP (1.629030e-16),
    # because the intended total then *is* the ideal-gas concentration at P0 and
    # the trip through molar masses re-rounds.  `== 0.0` therefore crashes the
    # moment claude/align-molar-volume-constant lands.  The bound below still
    # catches the defect this guard exists for by ten orders of magnitude: TPX
    # renormalisation gives 5.692129e-06.
    assert worst < 1e-14, f"initial state was renormalised (worst={worst:.6e})"

    t0 = time.perf_counter()
    conc = integrate(gas, T_CHECK, T_END)
    wall = time.perf_counter() - t0
    print(f"CHECK T=1500K t=1e-4s wall={wall:.3f}s")
    for name in SUB_SPECIES:
        print(f"  {name:5s} {conc[gas.species_index(name)]:.17e}")

    print("IGNITION DELAYS (isothermal, seed H=1e-11 mol/cm3, 2%H2/1%O2/97%N2, 1 atm)")
    for T in (1400.0, 1500.0, 1600.0, 1700.0, 1800.0):
        t_ign = ignition_delay(gas, T)
        print(f"  T={T:5.0f}K  t_ign={t_ign*1e6:9.2f} us")


if __name__ == "__main__":
    main()
