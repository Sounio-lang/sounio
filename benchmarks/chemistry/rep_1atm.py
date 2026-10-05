#!/usr/bin/env python3
"""RECONSTRUCTION (2026-09-01) -- this is NOT the original artefact.

The preprint cites `benchmarks/chemistry/rep_1atm.py`. That file was never
committed to this repository: no blob under that name exists in any tree
reachable from any ref, there are no stashes, no other worktrees and no
dangling objects (see RESULTS.md section 4). What follows was written from
the protocol described in the preprint, not recovered.

WHAT IT PROBES -- and why the answer is a counterfactual.

The preprint reports a standard-state defect: the replicas allegedly set
P0 = 1.0e5 Pa (1 bar) while the GRI-Mech 3.0 NASA-7 coefficients assume
1 atm. Measured at this commit, THAT DEFECT DOES NOT EXIST:

  * Cantera reports reference_pressure = 101325.0 Pa for all 53 species of
    gri30.yaml (it declares no `reference-pressure` key, so the Cantera
    default applies and it is 1 atm, not 1 bar);
  * both replicas already set P0 = 101325.0, with the comment
    "CHEMKIN/GRI standard state: 1 atm";
  * the replica's Kc agrees with Cantera's to <= 1.843e-11 on all 29 H/O
    reactions, and that residual is fully explained by the gas constant
    (8.314462618 vs Cantera's 8.31446261815324, a 1.84e-11 difference)
    appearing once per net mole -- it shows up on exactly the 12 reactions
    with dn != 0 and on none of the 17 with dn = 0.

This script therefore does two things: (1) it VERIFIES the shipped P0 against
the oracle, so the absence of the defect is checkable rather than asserted;
and (2) it quantifies the COUNTERFACTUAL -- what a P0 = 1e5 Pa error would do
to Kc, per reaction -- so the preprint's argument can be evaluated on its
merits. No trajectory is rerun under the counterfactual, because no shipped
artefact carries the defect.

Run:  python3 benchmarks/chemistry/rep_1atm.py
"""
import importlib.util
import os
import sys

import cantera as ct

HERE = os.path.dirname(os.path.abspath(__file__))
P0_ATM = 101325.0
P0_BAR = 1.0e5
T = 1500.0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    rep = _load("rep", os.path.join(HERE, "gri30_h2_python_replica.py"))
    can = _load("can", os.path.join(HERE, "gri30_h2_cantera_parity.py"))

    print(f"cantera {ct.__version__}")

    # (0) what does the oracle actually assume?
    full = ct.Solution("gri30.yaml")
    refs = sorted({s.thermo.reference_pressure for s in full.species()})
    print(f"\ngri30.yaml NASA-7 reference pressures, as Cantera reads them: {refs}")
    assert refs == [P0_ATM], "gri30.yaml reference pressure is not 1 atm"
    print(f"  -> 1 atm = {P0_ATM} Pa for all {full.n_species} species (NOT 1 bar)")
    print(f"shipped replica P0 = {rep.P0}  -> "
          f"{'CORRECT' if rep.P0 == P0_ATM else 'DEFECTIVE'}")

    # (1) verify Kc against the oracle, reaction by reaction
    gas = can.build_submechanism()
    gas.TPX = T, P0_ATM, {"N2": 0.97, "H2": 0.02, "O2": 0.01}
    print(f"\nKc verification at T = {T:.0f} K (mol/cm^3 basis), replica vs Cantera:")
    print(f"{'#':>3} {'reaction':32s} {'dn':>3} {'rel dev':>10s} {'P0=1e5 would give':>19s}")
    worst_zero = worst_nonzero = 0.0
    n_nonzero = 0
    for r in range(rep.NR):
        dn = rep.dn[r]
        kc_rep = rep.kc_cm(r, T)
        kc_ct = gas.equilibrium_constants[r] * (1e-3) ** dn
        dev = abs(kc_rep - kc_ct) / abs(kc_ct)
        cf = (P0_BAR / P0_ATM) ** dn - 1.0
        if dn == 0:
            worst_zero = max(worst_zero, dev)
        else:
            worst_nonzero = max(worst_nonzero, dev)
            n_nonzero += 1
        print(f"{r:3d} {rep.RX[r]['eq'][:32]:32s} {dn:3.0f} {dev:10.3e} "
              f"{cf * 100:+18.4f}%")

    print(f"\nworst |Kc_replica/Kc_cantera - 1|:")
    print(f"  dn = 0 reactions ({rep.NR - n_nonzero:2d}): {worst_zero:.3e}  (machine precision)")
    print(f"  dn != 0 reactions ({n_nonzero:2d}): {worst_nonzero:.3e}  "
          f"(one factor of c0 = P0/(R T); R differs by 1.84e-11)")

    # (2) the counterfactual
    ratio = P0_BAR / P0_ATM
    print(f"\nCOUNTERFACTUAL: if P0 were {P0_BAR:.0f} Pa, Kc would scale by "
          f"(1e5/101325)^dn:")
    for dn in (-1, 0, 1):
        print(f"  dn = {dn:+d}  factor {ratio ** dn:.9f}  "
              f"error {abs(ratio ** dn - 1) * 100:.4f}%")
    print(f"\n{n_nonzero} of {rep.NR} H/O reactions have dn != 0, so such a defect "
          f"would be PRESENT in {100.0 * n_nonzero / rep.NR:.0f}% of the")
    print("submechanism -- not rare. What makes it UNOBSERVABLE in a "
          "far-from-equilibrium")
    print("trajectory is the forward/reverse flux ratio: Kc enters only the reverse "
          "term,")
    print("which at 1500 K in the induction period is 6-15 orders of magnitude below "
          "the")
    print("forward term. It becomes observable only where an equilibrium pins a "
          "population")
    print("(NNH, dn = +1, in the full mechanism).")


if __name__ == "__main__":
    main()
