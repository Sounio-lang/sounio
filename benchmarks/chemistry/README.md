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
- `gri30_h2_adiabatic_replica.py` — independent pure-Python recomputation of
  the ADIABATIC (constant-U,V) path: NASA-7 cp/h, the coupled energy equation
  with Kc recomputed every RHS, RK4 sweep 1000–2000 K, dt-convergence.
  No dependencies. Run: `python3 gri30_h2_adiabatic_replica.py` (~6 min)
- `gri30_h2_adiabatic_cantera.py` — Cantera 3.2 reference for the adiabatic
  path: same sub-mechanism, constant-V `IdealGasReactor(energy="on")` (Cantera
  has no `IdealGasConstVolumeReactor`; `IdealGasReactor` IS the constant-V
  one), identical mixture/seed/horizon protocol.
  Run: `PYTHONPATH=/tmp/pylibs python3 gri30_h2_adiabatic_cantera.py`

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

## Adiabatic (constant-U,V) ignition-delay curve (2026-07-31)

`stdlib/chemistry/gri30_h2.sio` additionally exposes the adiabatic reactor
path (`g30_cp_r`, `g30_h_rt`, `g30_uv_rhs`, `g30_adiabatic_delay`): sensible
energy conservation couples T to the kinetics,

  dT/dt = -T * sum(h_rt_i - 1) * omega_i / sum c_i * (cp_r_i - 1),  u_mix = const

with Kc(T) recomputed on EVERY RHS from one shared per-species g_rt pass (the
isothermal path precomputes Kc once). Same mixture/seed as the isothermal
protocol, T0 sweep 1000–2000 K, fixed-step RK4 dt = 1e-8, delays = mid-step
times of max d[H2O]/dt (primary) and max dT/dt (secondary), uniform 3 ms
horizon (the early-exit certificate needs t > 2x peak, so a short horizon can
false-negative slow-but-real ignitions).

Ignition delays (microseconds), all three implementations on the identical
protocol:

| T0 (K) | 1000/T | Sounio (lean) t_h2o/t_dT | Python replica t_h2o/t_dT | Cantera 3.2 t_h2o/t_dT |
|--------|--------|--------------------------|---------------------------|------------------------|
| 1000   | 1.0000 | no-ign (3 ms)            | no-ign (3 ms)             | no-ign (3 ms)          |
| 1100   | 0.9091 | 674 / 718                | 674.96 / 718.67           | 674.96 / 718.68        |
| 1200   | 0.8333 | 376 / 415                | 376.31 / 415.84           | 376.30 / 415.84        |
| 1300   | 0.7692 | 241 / 271                | 241.95 / 271.88           | 241.95 / 271.88        |
| 1400   | 0.7143 | 169 / 189                | 169.66 / 189.44           | 169.66 / 189.43        |
| 1500   | 0.6667 | 126 / 138                | 126.31 / 138.25           | 126.31 / 138.25        |
| 1600   | 0.6250 | 98 / 105                 | 98.25 / 105.15            | 98.25 / 105.14         |
| 1700   | 0.5882 | 78 / 82                  | 78.98 / 82.75             | 78.97 / 82.74          |
| 1800   | 0.5556 | 65 / 66                  | 65.06 / 66.88             | 65.06 / 66.88          |
| 1900   | 0.5263 | 54 / 55                  | 54.55 / 55.11             | 54.55 / 55.11          |
| 2000   | 0.5000 | 46 / 46                  | 46.32 / 46.05             | 46.31 / 46.05          |

Sounio prints integer microseconds (truncation); replica and Cantera agree to
~0.01–0.1% at every point, and Sounio is within the truncation window of both.
Pre-front checkpoint at t = 10 us, T0 = 2000 K: replica T = 1999.99133 K,
H2O = 5.4779e-11 vs Cantera 1999.9913 K, 5.4797e-11 (0.03%). dt-convergence
(replica, t_h2o, dt = 1e-8 vs 5e-9): 126.315/126.317 us at 1500 K,
78.975/78.978 us at 1700 K — converged to 4 significant figures.

Physics: monotonic Arrhenius decrease 1100 -> 2000 K (no NTC for H2), no
ignition at 1000 K within 3 ms — consistent with the second-explosion-limit
crossover near ~1000–1050 K at 1 atm for dilute H2/O2. Literature anchors
(bibliography verified via Crossref; trend comparison only, no numeric
correlation constants are quoted because the full texts were not fetched):
Schott G.L., Kinsey J.L., "Kinetic Studies of Hydroxyl Radicals in Shock
Waves. II. Induction Times in the Hydrogen-Oxygen Reaction", J. Chem. Phys.
29(5):1177-1182 (1958), DOI 10.1063/1.1744674 — classic ~1 atm dilute H2/O2
induction times; Keromnes A., Metcalfe W.K., Heufer K.A., Donohoe N., et al.,
"An experimental and detailed chemical kinetic modeling study of hydrogen and
syngas mixture oxidation at elevated pressures", Combustion and Flame
160(6):995-1011 (2013), DOI 10.1016/j.combustflame.2013.01.001 — modern
shock-tube H2 oxidation reference.

The Cantera cross-check caught a real bug here too: the first version of the
adiabatic Python replica computed reverse rates with product exponents derived
as `reac - nu` (should be the explicit product stoichiometry), which silently
zeroed the reverse channels of H2+O2 -> H+HO2 and the back-dissociation
reactions at states where only major species are present. The error grew with
temperature (0.1% at 1100 K -> 8.6% at 2000 K on the delay) and was invisible
to every Sounio<->replica pin (both used to agree, each wrong differently).
Per-reaction rate comparison against Cantera at t0 exposed it in one shot;
after the fix all three implementations agree as tabulated above. Lesson:
cross-validate per-reaction rates, not just integrated observables. The
Sounio module itself used the explicit product table throughout and was never
affected.

Engine/parity status:
- Suite tests: `tests/stdlib/chemistry/test_gri30_h2_adiabatic.sio`
  (run-pass: thermo pins, single-shot UV RHS, 1000-step pre-front checkpoint)
  and `test_gri30_h2_adiabatic_delay.sio` (full fronts at 1500/2000 K) both
  PASS under lean_single/native (the CI suite engine); the delay test carries
  `//@ known-failure` for madaros.
- Byte parity madaros x lean_single: a 16-pin probe (12 NASA-7 thermo values
  + 4 single-shot UV RHS values, 9 significant digits) is byte-identical
  between engines (strip the madaros banner with
  `awk '/^   Output: /{f=1; next} f'`).
- The full adiabatic demo and the >100-step integration loops do NOT run on
  madaros (pre-existing engine bugs, isothermal path affected the same way):
  (a) cumulative per-process limit — a single process performing more than
  ~4k–28k `g30_uv_rhs` evaluations exits silently with rc=182 (the existing
  isothermal `test_gri30_h2.sio` and `h2_ignition_uq_demo.sio` hit the same
  wall); (b) the 1000-step UV RK4 loop produces deterministic wrong values on
  madaros (T diverges even at 100 steps) while single RHS evaluations are
  correct; (c) multiplying a tuple-destructured array element in the caller
  (e.g. `dc[0] * 1.0e6` after `let (dt, dc) = g30_uv_rhs(...)`) miscompiles
  on madaros, while the same expression on a plain local is correct. All
  three are lean_single-clean. Run the demo with
  `SOUNIO_SOUC_ENGINE=lean_single` (see below).

## Adiabatic demo

`examples/chemistry/h2_adiabatic_shocktube_demo.sio` prints the 11-point
adiabatic delay curve above (~45 min runtime on lean_single):

```
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run examples/chemistry/h2_adiabatic_shocktube_demo.sio
```
