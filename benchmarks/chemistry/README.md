# Chemistry benchmarks: Sounio vs Cantera (GRI-Mech 3.0 H/O)

Cross-validation harness for `stdlib/chemistry/gri30_h2.sio` — the complete
GRI-Mech 3.0 H/O sub-mechanism (10 species, 29 reactions) with NASA-7 detailed
balance and native 1-sigma GUM uncertainty bands.

## Files

- `extract_gri30_h2.py` — extracts the H/O sub-mechanism (default) or the full
  53-species/325-reaction mechanism (`--full`) from Cantera's `gri30.yaml`
  into JSON. Usage: `GRI30_YAML=/path/to/gri30.yaml python3 extract_gri30_h2.py [--full]`
- `gri30_h2_mechanism.json` — the extracted H/O sub-mechanism (input to both
  the Sounio generator and the Python replica)
- `gri30_full_mechanism.json` — the full GRI-Mech 3.0 (53 species, 325 reactions)
- `gri30_h2_python_replica.py` — independent pure-Python recomputation of every
  number the Sounio run-pass tests assert (rates, Kc, trajectory checkpoint,
  epistemic band). No dependencies.
- `gri30_h2_cantera_parity.py` — Cantera 3.2 reference: builds the same
  sub-mechanism from Cantera's own `gri30.yaml`, runs the identical isothermal
  protocol, prints the parity table and ignition delays.
  Run: `PYTHONPATH=/tmp/pylibs python3 gri30_h2_cantera_parity.py`
  (install: `pip install --target=/tmp/pylibs cantera`)

## Protocol (all three implementations)

2% H2 / 1% O2 / 97% N2, 1 atm, isothermal, H-atom seed 1e-11 mol/cm3
(chain initiation — GRI-Mech has no thermal initiation at these temperatures).
Epistemic runs add 1% standard uncertainty on initial H2/O2 and per-reaction
1-sigma relative rate uncertainties (representative magnitudes from
Baulch 2005 / Konnov 2008 / Hong 2011 — not a refit).

## Headline results (2026-07-27)

Isothermal ignition delays, max d[H2O]/dt, microseconds:

| T (K) | Sounio | Cantera 3.2 |
|-------|--------|-------------|
| 1400  | 169    | 169.66      |
| 1500  | 126    | 126.34      |
| 1600  | 98     | 98.29       |
| 1700  | 79     | 79.00       |
| 1800  | 65     | 65.08       |

Pre-front checkpoint at t = 1e-4 s, T = 1500 K (Sounio fixed-step RK4
dt = 1e-8 vs Cantera CVODE adaptive): major species within 0.2–2%, radical
pools within ~3%, H2O2 (1e-13 level) within ~16% — solver-type difference,
not mechanism difference (Python dt-convergence at 1e-8 vs 5e-9 is identical
to 4 significant figures).

Native 1-sigma band at the same checkpoint (Sounio vs Python replica, same
first-order diagonal GUM formula): u(H2) = 1.4484e-9 both; u(H2O), u(OH),
u(H) agree to 3 significant figures. Cantera has no native equivalent —
UQ there requires an external Monte Carlo driver.

The Cantera cross-check caught a real bug: the first version of the Sounio
module missed `2 O + M <=> O2 + M` (GRI-Mech Reaction 1) due to an
off-by-one in the extractor; the module now carries all 29 H/O reactions.

## Demo

`examples/chemistry/h2_ignition_uq_demo.sio` prints the checkpoint table with
the native 1-sigma band and the ignition-delay curve (~70 s runtime):

```
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run examples/chemistry/h2_ignition_uq_demo.sio
```

## Notes

- The ignition front itself is exponentially phase-sensitive; only pre-front
  checkpoints are cross-language parity targets.
- Sounio wall time is dominated by the fixed-step RK4 (10k steps at dt = 1e-8);
  Cantera's CVODE takes adaptive steps. This is a correctness/UQ benchmark,
  not a speed benchmark.
