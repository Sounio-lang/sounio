#!/usr/bin/env python3
"""Cantera 3.2 reference for the full GRI30 forward-sensitivity UQ smoke.

The calculation matches ``simulate_gri30_full_epistemic`` at its short
checkpoint: 53 species, 325 reaction-rate parameters, 1% initial H2/O2
uncertainty, T=1500 K, constant density, energy off, and t=4e-7 s.

Each reaction sensitivity is a central difference in ln(m_r), using reaction
multipliers exp(+-delta). Initial-condition columns use the same construction
for ln(c_H2,0) and ln(c_O2,0), then are scaled by 1%.

Run with Cantera 3.2.0, for example:

    PYTHONPATH=/tmp/sounio-cantera-py python3 \
      benchmarks/chemistry/gri30_full_cantera_uq_reference.py --jobs 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

import cantera as ct
import numpy as np


TEMPERATURE = 1500.0
END_TIME = 4.0e-7
SEED_H = 1.0e-11  # mol / cm^3
# mol/cm^3 at 1 atm is P0/(R*T)*1e-6.  This file used to carry the 82.057
# shorthand, whose truncated molar gas volume ran 4.461e-06 high; it matched
# the Sounio checkpoint only because Sounio carried the same shorthand.
#
# The form matters, not just the value: 1/(Rcgs*T) and P0/(R*T)*1e-6 differ by
# 1 ULP, and this stack is compared at 1-30 ULP.  Every site uses the SAME
# expression the Sounio modules use, so the arithmetic is bit-identical.
P0 = 101325.0
R_SI = 8.31446261815324  # J/(mol*K)
RTOL = 1.0e-12
ATOL = 1.0e-22

# Exact mirror of g30f_urel: default 0.30 plus these overrides. The Sounio
# regression pins the resulting sigma vector, so a table change must update
# this referee and its checked-in reference values together.
NAMED_UREL = {
    0: 0.25,   # 2 O + M <=> O2 + M
    1: 0.25,   # H + O + M <=> OH + M
    2: 0.15,   # H2 + O <=> H + OH
    32: 0.25,  # H + O2 + M <=> HO2 + M
    37: 0.10,  # H + O2 <=> O + OH
    42: 0.30,  # H + OH + M <=> H2O + M
    83: 0.10,  # H2 + OH <=> H + H2O
    85: 0.20,  # 2 OH <=> H2O + O
}


def initial_concentrations(gas: ct.Solution) -> np.ndarray:
    """Exact Sounio initial concentrations in mol/cm^3."""
    total = P0 / (R_SI * TEMPERATURE) * 1e-6
    c = np.zeros(gas.n_species)
    c[gas.species_index("H2")] = 0.02 * total
    c[gas.species_index("O2")] = 0.01 * total
    c[gas.species_index("N2")] = 0.97 * total
    c[gas.species_index("H")] = SEED_H
    return c


def set_concentrations(gas: ct.Solution, c_mol_cm3: np.ndarray) -> None:
    """Set T, density, and composition without renormalizing concentrations."""
    c_kmol_m3 = c_mol_cm3 * 1000.0
    rho = float(np.dot(c_kmol_m3, gas.molecular_weights))
    mass_fractions = c_kmol_m3 * gas.molecular_weights / rho
    gas.TDY = TEMPERATURE, rho, mass_fractions


def integrate(
    gas: ct.Solution,
    initial_scale_species: int | None = None,
    initial_scale: float = 1.0,
) -> np.ndarray:
    c0 = initial_concentrations(gas)
    if initial_scale_species is not None:
        c0[initial_scale_species] *= initial_scale
    set_concentrations(gas, c0)
    reactor = ct.IdealGasReactor(gas, energy="off", clone=False)
    network = ct.ReactorNet([reactor])
    network.rtol = RTOL
    network.atol = ATOL
    network.advance(END_TIME)
    return reactor.phase.concentrations * 1.0e-3


def reaction_sensitivity(task: tuple[int, float]) -> tuple[int, np.ndarray]:
    reaction, delta = task
    gas = ct.Solution("gri30.yaml")
    gas.set_multiplier(math.exp(delta), reaction)
    plus = integrate(gas)
    gas.set_multiplier(math.exp(-delta), reaction)
    minus = integrate(gas)
    return reaction, (plus - minus) / (2.0 * delta)


def initial_sensitivity(task: tuple[str, float]) -> tuple[str, np.ndarray]:
    species, delta = task
    gas = ct.Solution("gri30.yaml")
    index = gas.species_index(species)
    plus = integrate(gas, index, math.exp(delta))
    minus = integrate(gas, index, math.exp(-delta))
    return species, (plus - minus) / (2.0 * delta)


def uncertainty_table(n_reactions: int) -> np.ndarray:
    urel = np.full(n_reactions, 0.30)
    for reaction, uncertainty in NAMED_UREL.items():
        urel[reaction] = uncertainty
    return urel


def compute_reference(delta: float, jobs: int) -> dict[str, object]:
    gas = ct.Solution("gri30.yaml")
    assert ct.__version__ == "3.2.0", ct.__version__
    assert gas.n_species == 53, gas.n_species
    assert gas.n_reactions == 325, gas.n_reactions

    nominal = integrate(gas)
    sensitivities = np.zeros((gas.n_species, gas.n_reactions))
    tasks = [(reaction, delta) for reaction in range(gas.n_reactions)]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for reaction, column in executor.map(reaction_sensitivity, tasks):
            sensitivities[:, reaction] = column

        initial_columns = dict(executor.map(
            initial_sensitivity,
            [("H2", delta), ("O2", delta)],
        ))

    urel = uncertainty_table(gas.n_reactions)
    variance = np.sum(np.square(sensitivities * urel), axis=1)
    variance += np.square(0.01 * initial_columns["H2"])
    variance += np.square(0.01 * initial_columns["O2"])
    sigma = np.sqrt(variance)

    return {
        "cantera_version": ct.__version__,
        "mechanism": "gri30.yaml",
        "temperature_K": TEMPERATURE,
        "time_s": END_TIME,
        "energy": "off",
        "constant_density": True,
        "rtol": RTOL,
        "atol": ATOL,
        "central_log_multiplier_delta": delta,
        "species": gas.species_names,
        "nominal_mol_cm3": nominal.tolist(),
        "sigma_mol_cm3": sigma.tolist(),
        "reaction_urel": urel.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=1.0e-3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.delta <= 0.0:
        parser.error("--delta must be positive")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    started = time.perf_counter()
    result = compute_reference(args.delta, args.jobs)
    elapsed = time.perf_counter() - started
    species = result["species"]
    nominal = result["nominal_mol_cm3"]
    sigma = result["sigma_mol_cm3"]

    print(
        f"Cantera {result['cantera_version']} {result['mechanism']} "
        f"species={len(species)} reactions=325 wall={elapsed:.3f}s"
    )
    print(
        f"T={TEMPERATURE:.0f}K t={END_TIME:.1e}s energy=off "
        f"rtol={RTOL:.0e} atol={ATOL:.0e} delta={args.delta:.0e}"
    )
    for name in ("H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "NNH"):
        index = species.index(name)
        print(f"{name:5s} y={nominal[index]:.15e} u={sigma[index]:.15e}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"json={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
