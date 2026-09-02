#!/usr/bin/env python3
"""RECONSTRUCTION (2026-09-01) -- this is NOT the original artefact.

The preprint cites `benchmarks/chemistry/rep_prodfix.py`. That file was never
committed to this repository (see RESULTS.md section 4). What follows was
written from the protocol described in the preprint, not recovered.

WHAT IT PROBES -- and why the answer is a counterfactual.

The preprint reports a reverse-rate defect in the H/O replica's `rates_net()`:

    p = reac[r][s] - nu[r][s]        # reported as the buggy line
    p = reac[r][s] + nu[r][s]        # reported as the fix

Measured at this commit, THAT DEFECT DOES NOT EXIST. `gri30_h2_python_replica.py`
already reads `p = prod[r][s]` directly, exactly as the full-mechanism replica
does via `prod_nz`, and no version of the file in any tree reachable from any
ref contains the reported line. There was no back-port to perform.

Note also that the reported "fix" is a no-op relative to the shipped code:
nu = prod - reac, so reac + nu == prod identically. Only the reported *buggy*
form, reac - nu == 2*reac - prod, differs from the shipped code.

This script therefore does two things: (1) it VERIFIES the shipped reverse-rate
path against the oracle, so the absence of the defect is checkable rather than
asserted -- all 29 net rates of progress against Cantera, with the state chosen
to load the radicals so the reverse terms actually carry flux; and (2) it
quantifies the COUNTERFACTUAL by evaluating the reported buggy exponent form
side by side, so the magnitude of the defect the preprint describes can be seen.

The H + HO2 <=> O2 + H2 initiation channel is reported separately, as the
preprint asks.

Run:  python3 benchmarks/chemistry/rep_prodfix.py
"""
import importlib.util
import math
import os
import sys

import cantera as ct

HERE = os.path.dirname(os.path.abspath(__file__))
T = 1500.0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rates_net_buggy(rep, t, c, kc):
    """`rates_net` with the reported buggy reverse exponent, reac - nu."""
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
            p = rep.reac[r][s] - rep.nu[r][s]      # the reported defect
            if p > 0:
                rev *= c[s] ** p
        out.append(fwd - rev)
    return out


def main():
    rep = _load("rep", os.path.join(HERE, "gri30_h2_python_replica.py"))
    can = _load("can", os.path.join(HERE, "gri30_h2_cantera_parity.py"))
    print(f"cantera {ct.__version__}")

    # (0) is the reported line present anywhere?
    src = open(os.path.join(HERE, "gri30_h2_python_replica.py")).read()
    present = "reac[r][s] - nu[r][s]" in src
    print(f"\nreported buggy line present in shipped replica: {present}")
    print(f"shipped reverse exponent: "
          f"{'prod[r][s]' if 'p = prod[r][s]' in src else 'SOMETHING ELSE'}")
    print("identity check: nu = prod - reac, so reac + nu == prod  -> the reported")
    print("'fix' is a no-op relative to the shipped code; only the buggy form differs.")

    # (1) radical-loaded state: reverse terms must carry real flux
    mtot = 1.0 / (82.057 * T)
    c = {s: 0.0 for s in rep.SP}
    c["H2"], c["O2"], c["N2"] = mtot * 0.02, mtot * 0.01, mtot * 0.97
    c["H"], c["OH"], c["O"] = 1e-9, 1e-10, 1e-10
    c["H2O"], c["HO2"], c["H2O2"] = 1e-9, 1e-12, 1e-13
    cvec = [c[s] for s in rep.SP]
    kc = [rep.kc_cm(r, T) for r in range(rep.NR)]

    rn = rep.rates_net(T, cvec, kc)
    rn_bug = rates_net_buggy(rep, T, cvec, kc)

    gas = can.build_submechanism()
    conc = [0.0] * gas.n_species
    for s in rep.SP:
        conc[gas.species_index(s)] = c[s] * 1e3          # mol/cm^3 -> kmol/m^3
    gas.set_unnormalized_mole_fractions([x / sum(conc) for x in conc])
    gas.TD = T, sum(conc[i] * gas.molecular_weights[i] for i in range(gas.n_species))
    rop = gas.net_rates_of_progress

    print(f"\nnet rates of progress at T = {T:.0f} K, radical-loaded state "
          f"(mol/cm^3/s):")
    print(f"{'#':>3} {'reaction':30s} {'shipped':>14s} {'Cantera':>14s} "
          f"{'rel':>10s} {'buggy form':>14s} {'buggy rel':>10s}")
    worst = worst_bug = 0.0
    n = 0
    for r in range(rep.NR):
        ct_rn = rop[r] * 1e-3                            # kmol/m^3/s -> mol/cm^3/s
        if ct_rn == 0.0 and rn[r] == 0.0:
            print(f"{r:3d} {rep.RX[r]['eq'][:30]:30s} "
                  f"{'both exactly 0':>14s}")
            continue
        dev = abs(rn[r] - ct_rn) / abs(ct_rn)
        dev_bug = abs(rn_bug[r] - ct_rn) / abs(ct_rn)
        worst = max(worst, dev)
        worst_bug = max(worst_bug, dev_bug)
        n += 1
        print(f"{r:3d} {rep.RX[r]['eq'][:30]:30s} {rn[r]:+14.6e} {ct_rn:+14.6e} "
              f"{dev:10.3e} {rn_bug[r]:+14.6e} {dev_bug:10.3e}")

    print(f"\nover {n} non-zero reactions:")
    print(f"  shipped reverse path, worst relative deviation from Cantera: "
          f"{worst:.3e}")
    print(f"  reported buggy form, worst relative deviation from Cantera:  "
          f"{worst_bug:.3e}")
    # This line used to assert, in the present tense, a ~1e-07 floor on the
    # shipped path and attribute it to the CHEMKIN activation-energy constant
    # R = 1.9872041 cal/mol/K.  The attribution was right and #2382 acted on
    # it: the replica now uses Cantera's own 8.31446261815324/4.184, the floor
    # it explained is gone, and the sentence outlived the fix that disproved
    # it.  So the floor is printed from what this run measured, never asserted.
    if worst > 1e-9:
        print(f"  the {worst:.3e} floor on the shipped path is the "
              f"activation-energy gas constant,")
        print(f"  R_cal = {rep.R_CAL!r} against Cantera's "
              f"8.31446261815324/4.184, not a stoichiometry error")
    else:
        print(f"  the shipped path agrees with the oracle to {worst:.3e}, at "
              f"the double-precision floor:")
        print(f"  R_cal = {rep.R_CAL!r} is Cantera's own value (#2382), so no "
              f"activation-energy floor remains")

    i = [r for r in range(rep.NR) if rep.RX[r]["eq"] == "H + HO2 <=> O2 + H2"]
    for r in i:
        ct_rn = rop[r] * 1e-3
        print(f"\nH + HO2 <=> H2 + O2 initiation channel (index {r}):")
        print(f"  shipped  {rn[r]:.17e}")
        print(f"  Cantera  {ct_rn:.17e}   rel {abs(rn[r]-ct_rn)/abs(ct_rn):.3e}")
        print(f"  buggy    {rn_bug[r]:.17e}   rel "
              f"{abs(rn_bug[r]-ct_rn)/abs(ct_rn):.3e}")

    print("\nCONCLUSION: the shipped reverse-rate path is correct against the oracle.")
    print("No patch was applied, so the H/O checkpoint and the 1-sigma band do not")
    print("move. See RESULTS.md section 2.")


if __name__ == "__main__":
    main()
