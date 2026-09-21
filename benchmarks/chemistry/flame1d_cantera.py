#!/usr/bin/env python3
"""Cantera 3.2 reference for the Sounio 1-D H2/air laminar flame-speed
benchmark (benchmarks/chemistry/flame1d_replica.py, stdlib/chemistry/flame1d.sio).

TRANSPORT DECISION (the match target): the Sounio solver and the replica use
a UNITY LEWIS NUMBER closure — one shared diffusivity D(T) = lambda/(rho cp)
for heat and all species. The Cantera reference that is matched is therefore
the SAME H/O sub-mechanism (10 species, 29 reactions, from Cantera's own
gri30.yaml) run as a free flame with transport_model="unity-Lewis-number".
The mixture-averaged value (printed as MIXAVG) is CONTEXT ONLY: preferential
H2 diffusion roughly doubles H2 flame speed, so mixture-averaged numbers
(~2.1 m/s at phi=1) must never be the match target.

Outputs (machine-checkable):
  - UNITY-LEWIS free flame at phi=1: S_L, T_ad, grid size, flame-zone widths
  - D(T) = lambda/(rho cp) samples along that solution (the replica/Sounio
    power-law D0 (T/300)^n is fit to these)
  - HP equilibrium of the unburned mixture (initial condition of the solvers)
  - MIXAVG free flame at phi=1 (context column)
  - phi sweep 0.6..1.4 unity-Lewis (context for the single-phi Sounio run)

Cantera 3.2 quirk worked around here: a sub-mechanism Solution built via the
Python constructor (or via write_yaml unmodified) silently loses its
transport model, so we round-trip through YAML and insert
"transport: unity-Lewis-number" into the phases block.

Run:  PYTHONPATH=/tmp/pylibs python3 benchmarks/chemistry/flame1d_cantera.py
"""
import os
import re
import time

import cantera as ct

SUB_SPECIES = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2", "AR"]
SUB_SET = set(SUB_SPECIES)
P0 = ct.one_atm
T_U = 300.0
X_U = "H2:2, O2:1, N2:3.76"
LEWIS_YAML = "/tmp/gri30_h2_sub_lewis.yaml"


def equation_species(eq):
    eq = eq.split("#", 1)[0]
    eq = eq.replace("(+M)", " ").replace("<=>", " + ").replace("=>", " + ")
    toks = set()
    for tok in eq.split("+"):
        tok = re.sub(r"^\s*\d+(\.\d+)?\s*", "", tok).strip()
        if tok and tok != "M":
            toks.add(tok)
    return toks


def build_lewis_yaml(path=LEWIS_YAML):
    """Write the sub-mechanism YAML with transport + Troe falloff re-inserted.

    Cantera 3.2 write_yaml drops the single Troe falloff reaction
    (2 OH (+M) <=> H2O2 (+M)) from the 29-reaction subset, so it is appended
    by hand below, with rate constants converted from gri30.yaml units
    (cm, mol, cal/mol) to the writer's SI units (m, kmol, J/kmol):
    low-P A 2.3e18 cm^6/mol^2/s -> 2.3e12 m^6/kmol^2/s,
    high-P A 7.4e13 cm^3/mol/s -> 7.4e10 m^3/kmol/s, Ea x 4184."""
    full = ct.Solution("gri30.yaml")
    keep_species = [s for s in full.species() if s.name in SUB_SET]
    keep_rxns = [r for r in full.reactions()
                 if equation_species(r.equation) <= SUB_SET]
    sub = ct.Solution(thermo="ideal-gas", kinetics="gas",
                      species=keep_species, reactions=keep_rxns)
    sub.write_yaml(path)
    with open(path) as f:
        txt = f.read()
    # insert the transport model into the phases block after "kinetics: bulk"
    assert "transport:" not in txt.split("species:")[0], "already has transport?"
    txt = txt.replace("    kinetics: bulk\n",
                      "    kinetics: bulk\n"
                      "    transport: unity-Lewis-number\n", 1)
    if "2 OH (+M)" not in txt:
        txt += (
            "  - equation: 2 OH (+M) <=> H2O2 (+M)\n"
            "    type: falloff\n"
            "    low-P-rate-constant: {A: 2.3e+12, b: -0.9, Ea: -7.1128e+06}\n"
            "    high-P-rate-constant: {A: 7.4e+10, b: -0.37, Ea: 0.0}\n"
            "    Troe: {A: 0.7346, T3: 94.0, T1: 1756.0, T2: 5182.0}\n"
            "    efficiencies: {H2: 2.0, H2O: 6.0, CH4: 2.0, CO: 1.5, "
            "CO2: 2.0, C2H6: 3.0, AR: 0.7}\n")
    with open(path, "w") as f:
        f.write(txt)
    # rate-constant spot check: the hand-converted falloff block must
    # reproduce the Python-constructor mechanism's rates exactly
    sub.TPX = 1500.0, P0, X_U
    chk = ct.Solution(path)
    chk.TPX = 1500.0, P0, X_U
    kf_sub = sub.forward_rate_constants
    kf_chk = chk.forward_rate_constants
    assert len(kf_chk) == len(kf_sub) == 29, (len(kf_chk), len(kf_sub))
    rel = max(abs(a - b) / max(abs(a), 1e-300)
              for a, b in zip(sorted(kf_sub), sorted(kf_chk)))
    assert rel < 1e-9, f"falloff reinsertion mismatch: {rel:.2e}"
    return path


def lewis_gas():
    if not os.path.exists(LEWIS_YAML):
        build_lewis_yaml()
    gas = ct.Solution(LEWIS_YAML)
    assert gas.transport_model == "unity-Lewis-number", gas.transport_model
    assert gas.n_reactions == 29, f"stale YAML ({gas.n_reactions} rxns) — " \
        f"delete {LEWIS_YAML} and rerun to rebuild with the Troe falloff"
    return gas


def free_flame(gas, phi, width=0.03, loglevel=0):
    gas.TP = T_U, P0
    gas.set_equivalence_ratio(phi, "H2", "O2:1, N2:3.76")
    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.10)
    f.solve(loglevel=loglevel, auto=True)
    return f


def main():
    t_start = time.perf_counter()
    gas = lewis_gas()
    print(f"SUBMECH species={gas.n_species} reactions={gas.n_reactions} "
          f"transport={gas.transport_model}")
    assert gas.n_species == 10 and gas.n_reactions == 29

    # HP equilibrium: adiabatic flame temperature + equilibrium products
    # (initial condition of the Sounio/replica slab)
    gas.TPX = T_U, P0, X_U
    rho_u = gas.density
    Y_u = list(gas.Y)
    gas.equilibrate("HP")
    print(f"HP-EQUIL T_ad={gas.T:.2f} K rho_eq={gas.density:.5f} kg/m3")
    print("Y_eq=" + " ".join(f"{y:.6e}" for y in gas.Y))
    print(f"Y_U(H2,O2,N2)=({Y_u[0]:.5f},{Y_u[3]:.5f},{Y_u[8]:.5f}) "
          f"rho_u={rho_u:.4f} kg/m3")

    # unity-Lewis free flame at phi=1 (the match target)
    f = free_flame(lewis_gas(), 1.0)
    print(f"UNITY-LEWIS phi=1.0: S_L={f.velocity[0]:.4f} m/s "
          f"T_ad={f.T[-1]:.1f} K grid={f.flame.n_points} pts")

    # D(T) = lambda/(rho cp) samples along the solution
    gas2 = lewis_gas()
    print("D(T) = lambda/(rho cp) samples:")
    for tv in (300.0, 500.0, 800.0, 1200.0, 1600.0, 2000.0, 2300.0):
        # sample the flame state nearest to tv
        import numpy as np
        j = int(np.argmin(np.abs(f.T - tv)))
        gas2.TPY = f.T[j], P0, f.Y[:, j]
        d = gas2.thermal_conductivity / (gas2.density * gas2.cp_mass)
        print(f"  ({tv:.0f}, {d:.3e})")

    # mixture-averaged context value
    gm = ct.Solution(LEWIS_YAML)
    gm.transport_model = "mixture-averaged"
    fm = free_flame(gm, 1.0)
    print(f"MIXAVG phi=1.0 (CONTEXT ONLY): S_L={fm.velocity[0]:.4f} m/s "
          f"T_ad={fm.T[-1]:.1f} K")

    # dump the converged unity-Lewis phi=1 profile on a uniform 20 um grid,
    # x = 0 at the T = 1500 K point, covering [-1 mm, +5 mm]. This is the
    # initial condition of the replica and the Sounio solver: starting at the
    # converged profile skips the ~1 ms slab-ignition transient (which would
    # blow the lean_single runtime budget) and turns the benchmark into a
    # pure propagation-speed (eigenvalue) measurement.
    import json
    import numpy as np
    gasp = lewis_gas()
    fp = free_flame(gasp, 1.0)
    j0 = int(np.argmin(np.abs(fp.T - 1500.0)))
    x_rel = fp.grid - fp.grid[j0]
    xs_out = np.arange(-1e-3, 5e-3 + 1e-9, 2e-5)
    # Cantera's FreeFlame runs fresh (x=0, inflow) -> burned (right); the
    # solvers want the mirror image: burned plateau on the LEFT, fresh gas on
    # the right, so the tracked front moves right into unburned mixture.
    T_out = np.interp(-xs_out, x_rel, fp.T)
    prof = {"dx": 2e-5, "x0": -1e-3, "n": len(xs_out),
            "T": [round(float(v), 4) for v in T_out],
            "Y": []}
    for xv in xs_out:
        row = []
        for i in range(10):
            row.append(float(np.interp(-xv, x_rel, fp.Y[i, :])))
        prof["Y"].append([float(f"{v:.8e}") for v in row])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "flame1d_profile_phi1.json")
    with open(out, "w") as fh:
        json.dump(prof, fh)
    print(f"PROFILE dumped: {out} n={prof['n']} dx=20um "
          f"T[0]={prof['T'][0]:.1f} T[-1]={prof['T'][-1]:.1f}")

    # phi sweep (unity-Lewis), context for the single-phi Sounio run
    print("phi sweep (unity-Lewis):")
    for phi in (0.6, 0.8, 1.0, 1.2, 1.4):
        fp2 = free_flame(lewis_gas(), phi)
        print(f"  phi={phi:.1f}  S_L={fp2.velocity[0]:.4f} m/s  "
              f"T_ad={fp2.T[-1]:.1f} K")

    print(f"wall={time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
