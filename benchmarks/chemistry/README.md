# Chemistry benchmarks: Sounio vs Cantera (GRI-Mech 3.0)

Cross-validation harnesses for the 10-species/29-reaction H/O sub-mechanism in
`stdlib/chemistry/gri30_h2.sio` and the 53-species/325-reaction mechanism in
`stdlib/chemistry/gri30_full.sio`. Cantera 3.2.0 with its bundled `gri30.yaml`
is the external reference; the in-repo Python replicas are diagnostics, not
oracles.

## Files

- `extract_gri30_h2.py` — extracts the H/O sub-mechanism (default) or the full
  53-species/325-reaction mechanism (`--full`) from Cantera's `gri30.yaml`
  into JSON. Usage: `GRI30_YAML=/path/to/gri30.yaml python3 extract_gri30_h2.py [--full]`
- `gri30_h2_mechanism.json` — the extracted H/O sub-mechanism (input to both
  the Sounio generator and the Python replica)
- `gri30_full_mechanism.json` — the full GRI-Mech 3.0 (53 species, 325 reactions)
- `gri30_h2_python_replica.py` and `gri30_full_python_replica.py` — pure-Python
  diagnostic replicas of the generated Sounio math. They are useful for tracing
  individual operations but must not supply regression reference values.
- `gri30_full_cantera_parity.py` — Cantera 3.2 reference for the FULL
  mechanism (53 species, 325 reactions), added 2026-09-01. Same isothermal
  protocol, checkpoints at t = 4e-6 s and t = 2e-5 s. Before this, only the
  H/O sub-mechanism had a Cantera oracle and the full-mechanism results had
  no reproduction path. Parity table in `RESULTS.md` section 6.
  Run: `python3 gri30_full_cantera_parity.py`
- `cpp/gri30_h2_band_crosscheck.cpp` — independent C++20 third
  implementation (no dependencies) of the H/O kinetics and the GUM band,
  written from the protocol rather than translated. Reproduces the Python
  replica's deterministic checkpoint to all 17 digits; used to settle the
  band-scaling law in `RESULTS.md` section 5.
  Run: `g++ -std=c++20 -O2 -o band_crosscheck cpp/gri30_h2_band_crosscheck.cpp && ./band_crosscheck gri30_h2_mechanism.json`
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
- `gri30_full_cantera_uq_reference.py` — independent full-mechanism UQ referee.
  It central-differences all 325 persistent reaction-rate parameters and the H2
  and O2 initial conditions, then assembles the first-order covariance.

Install and run the pinned referee from this directory:

```sh
python3 -m pip install --target /tmp/sounio-cantera-py cantera==3.2.0
PYTHONPATH=/tmp/sounio-cantera-py python3 gri30_full_cantera_uq_reference.py \
  --jobs 4 --json-out /tmp/gri30_full_cantera_uq_reference.json
```

## Shared reactor protocol

2% H2 / 1% O2 / 97% N2 nominally at 1 atm and 1500 K, using exactly
`1/(82.057*T)` mol/cm3 plus an additive H-atom seed of 1e-11 mol/cm3. The seed
makes the actual initial pressure 101325.576758 Pa. Cantera is initialized
through `TDY` so those concentrations are not renormalized. Epistemic runs add
1% standard uncertainty on initial H2/O2 and one persistent multiplier per
complete reaction. The multiplier scales the effective forward and reverse
rates together, including falloff reactions; it does not separately model
uncertainty in `k_inf`, `k_0`, or Troe parameters. The per-reaction standard
uncertainties are representative magnitudes from Baulch 2005, Konnov 2008, and
Hong 2011, not a refit.

## Headline results

Isothermal ignition delays, max d[H2O]/dt, microseconds:

| T (K) | Sounio | Cantera 3.2 |
|-------|--------|-------------|
| 1400  | 169.665 | 169.66      |
| 1500  | 126.345 | 126.34      |
| 1600  | 98.295  | 98.29       |
| 1700  | 79.015  | 79.00       |
| 1800  | 65.085  | 65.08       |

At the H/O pre-front checkpoint, t=1e-4 s and T=1500 K, the deterministic
Sounio trajectory agrees with Cantera to 2e-7 through 6e-6 relative for the
major species and radicals; H2O2 agrees to 5.9e-3 relative.

> **Superseded 2026-09-01** (see `RESULTS.md` section 1 for the commands, the
> commit and the full three-way table). The two sentences above are kept for
> the record and corrected as follows.
>
> 1. The 2e-7 through 6e-6 figure is **confirmed** for the majors and
>    radicals: re-measured 2.247e-07 to 2.660e-06 across all eight species.
> 2. The H2O2 figure of **5.9e-3 is wrong by three orders of magnitude**.
>    H2O2 agrees to **1.891e-06** — within the same range as every other
>    species, not an outlier. There is no H2O2 discrepancy to explain.
> 3. These figures hold **only** under the TDY (non-renormalising)
>    initialisation this README documents below. Until 2026-09-01 the shipped
>    `gri30_h2_cantera_parity.py` used `gas.TPX`, which renormalises the
>    H-atom seed away, shifts every initial concentration by -5.692129e-06
>    and inflates the checkpoint deviation ~15x, to 1.7e-06 .. 3.9e-05. That
>    script now implements the documented protocol and asserts the realised
>    initial state matches intent to exactly 0.0e+00.
> 4. Sounio agrees with the **Python replica** (same RK4, same dt) to
>    **1.8e-16 .. 4.6e-15 — 1 to 30 ULP, i.e. 15 significant figures.** The
>    demo's apparent 3.3e-12 .. 2.6e-10 is purely its print resolution;
>    `examples/chemistry/h2_precision_probe.sio` re-prints the same checkpoint
>    at 16 digits and settles it. There is no cross-language discrepancy.
> 5. The 2e-7 .. 6e-6 band is **not** the RK4-vs-CVODE integrator difference,
>    contrary to what the Notes below have said. Measured by halving the step
>    (`examples/chemistry/h2_probe2.sio`), RK4 truncation at dt = 1e-8 is at
>    most **2.3e-14** -- seven orders of magnitude too small to explain the
>    gap. The gap is the Arrhenius activation-energy gas constant: the
>    CHEMKIN-conventional `R = 1.9872041` cal/mol/K against Cantera's
>    `8.31446261815324/4.184 = 1.9872042586408316`, a 7.98e-08 difference
>    inside `exp(-Ea/(R*T))`. Substituting Cantera's value collapses the gap
>    to **8.9e-13 .. 1.2e-11** (a 155,000x-296,000x improvement), at the floor
>    of CVODE's own `rtol = 1e-12`. The shipped constant is deliberately left
>    at the CHEMKIN value -- see `RESULTS.md` section 1.5.
>
> No claim of the form "majors 0.2-2%, radicals ~3%, H2O2 ~16%" has ever
> appeared in this file or anywhere else in this repository, and no pairing
> measured on 2026-09-01 produces percent-level deviations. If you have seen
> that attribution (to "fixed-step RK4 vs CVODE"), it has no provenance here.

For the full mechanism at t=4e-7 s, Sounio's coherent forward-sensitivity
band agrees with the independent Cantera central-difference referee in all
eight H/O species. The largest relative sigma deviation is 8.724e-7 (H); the
other seven range from 1.559e-7 to 7.171e-7. The Sounio H2 value at this
checkpoint is 1.624886332731288e-7 mol/cm3 versus
1.624886332731066e-7 mol/cm3 in Cantera.

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

- The ignition front itself is exponentially phase-sensitive; pre-front
  checkpoints are the strict cross-language concentration targets.
- The UQ implementation propagates persistent rate-parameter sensitivities
  coherently. A quadrature source added independently at every time step would
  incorrectly make radical uncertainties scale with the square root of `dt`.
  **Measured 2026-09-01** (`RESULTS.md` section 5, two independent
  implementations agreeing to six digits): the Python replica does add such a
  source, and its radical band scales as sqrt(dt) exactly as predicted --
  a factor 2 in dt gives 1.41418-1.41817 and a factor 4 gives 1.99994-2.00841.
  The law applies only to species whose band is *generated* by accumulation:
  H2 and O2, whose variance is dominated by the 1% initial-condition
  uncertainty, give a ratio of exactly 1.000000 under every dt change. The
  same sqrt(T*dt) reasoning does **not** survive the induction period: past
  t ~ 1e-5 s the Jacobian terms dominate and the measured band ratio exceeds
  the sqrt(T/dt) prediction by 59x-242x per decade.
- First-order symmetric bands describe the covariance, not the full sampling
  distribution. Near ignition, skewed species distributions can require
  quantiles to describe asymmetric tails.
- Sounio wall time is dominated by the fixed-step RK4 (10k steps at dt = 1e-8);
  Cantera's CVODE takes adaptive steps. This is a correctness/UQ benchmark,
  not a speed benchmark. **Note (2026-09-01):** the step-size difference costs
  wall time but *not* accuracy -- RK4 truncation at dt = 1e-8 is 2.3e-14, and
  the observed parity gap has a different cause entirely (`RESULTS.md` 1.5).

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

## Laminar flame speed (2026-07-31)

Third leg of the trilogy: a 1-D premixed freely-propagating H2/air flame
(phi=1, T_u=300 K, p=1 atm) in `stdlib/chemistry/flame1d.sio`, validated
against a Cantera 3.2.0 free-flame solution computed with MATCHED kinetics
AND MATCHED transport (`flame1d_cantera.py`, 29-reaction H/O sub-mechanism,
`transport_model="unity-Lewis-number"`).

### Transport decision (the match target)

Unity Lewis number: one shared diffusivity for heat and all species,

  D(T) = 4.9518e-05 * (T/300)^1.4435  m2/s   (lambda = rho cp D)

fit to Cantera's operative lambda/(rho cp) sampled along its own unity-Lewis
free-flame solution (fit residuals -6.9%..+8.9% over 300-2300 K). The Cantera
reference was run with the SAME unity-Lewis transport, so the match target
uses identical physics:

  S_L = 1.6543 m/s   (Cantera 3.2.0, 29 rxn, unity-Lewis, phi=1; T_ad=2384.7 K)

Mixture-averaged transport gives 2.3345 m/s at phi=1 (preferential H2
diffusion — real physics) and the phi sweep under unity-Lewis is
0.6->0.9652, 0.8->1.3810, 1.0->1.6543, 1.2->1.8143, 1.4->1.8848 m/s; both
are CONTEXT ONLY, never the match target. Implementing mixture-averaged
multicomponent transport is out of scope for a kinetics-validation
benchmark.

### Model

Low-Mach planar flame, Godunov operator split per macro-step dt_m=5e-7 s:
(1) stage-clamped explicit RK4 isobaric chemistry (substeps <= 1e-8 s;
50 substeps above 2200 K, 12 below) on REACTIVE cells — T > 1200 K or any
cell carrying a radical pool (H+O+OH+HO2 > 1e-8); (2) explicit FTCS
transport: central diffusion + first-order upwind advection with the
enthalpy-flux correction -sum(j_k cp_k) dT/dx, then Y clip/renorm.
Flame-fixed frame with prescribed uniform mass flux rhou = -rho_u S_L_G
(S_L_G = 1.6543 m/s): fresh inflow on the right, burned outflow on the
left. IC: the converged Cantera unity-Lewis profile (301 pts, 20 um),
anchored with its T=1500 K point at x0=1 mm; domain L=8 mm.

Two honesty notes, both caught by the replica before the Sounio run:
- Pinned hot-bath zone: cells more than 0.6 mm behind the anchor are frozen
  at the profile state (Dirichlet burned reservoir). Unpinned, the ~1 mm of
  burned gas drains out the left boundary at |u| ~ 11 m/s in ~90 us, the
  plateau cools, and the front is flushed downstream (drift -1.4 m/s).
- The chemistry mask MUST include the radical-pool clause: a pure T cutoff
  masks the radical-seeded heat release of the 700-1200 K preheat foot that
  the Cantera profile carries (the foot's radical cells release
  +7..+17 K/macro-step, substep-converged) and the flame blows off — foot
  dies, front collapses, drift -1.5 m/s.
The exported Cantera profile is truncated on the burned side (its hot end
is 2102 K, not T_ad = 2384.7 K; the recombination tail burns for ~cm beyond
the export window). Consequence: the integral estimator is biased low
(~10%) and T_max of the run is 2102 K by construction. This biases only
the integral cross-check, not the front eigenvalue.

### Speed extraction

S_L = S_L_G + dx_f/dt: least-squares drift of the front position
x_f = centroid of (max(-dT/dx,0))^2 over the last 60% of the run. In the
prescribed-flux frame a consistent eigenvalue gives drift ~ 0; any
systematic model bias shows up as nonzero drift. Cross-check: mass-flux
integral S_L = integral(-omega_H2 W_H2) dx / (rho_u Y_H2,u) over the final
state (exact for a steady flame in any frame; ~10% low here from the
truncated tail).

### Results

Sounio (lean_single, 800 macro-steps = 0.4 ms, 35 min wall) vs the
dependency-free Python replica (`flame1d_replica.py`, identical model):

| quantity        | Sounio  | replica | agreement |
|-----------------|---------|---------|-----------|
| drift (m/s)     | +0.3388 | +0.3389 | 4 sig fig |
| S_L(slope) m/s  | 1.9931  | 1.9932  | 4 sig fig |
| S_L(integral)   | 1.6943  | 1.6944  | 4 sig fig |
| T_max (K)       | 2102    | 2102.1  | -         |

dx-convergence (replica; Sounio runs the base grid for runtime):

| grid  | dx   | dt_m    | t_end  | drift   | S_L(slope) | S_L(integral) |
|-------|------|---------|--------|---------|------------|---------------|
| coarse| 80um | 2.0e-6  | 0.4 ms | +0.6857 | 2.3400     | 2.1984        |
| base  | 40um | 5.0e-7  | 0.4 ms | +0.3389 | 1.9932     | 1.6944        |
| fine  | 20um | 1.25e-7 | 0.2 ms | +0.1027 | 1.7570     | 1.5270        |

S_L(slope) decreases monotonically toward the Cantera reference under
refinement (excess +41% / +20.5% / +6.2% at 80/40/20 um) — the signature of
the first-order upwind advection, whose numerical diffusivity
D_num = |u| dx/2 is 20-30% of the physical D at the front on the base grid
and inflates the discrete eigenvalue. The two estimators bracket the
reference on the fine grid (1.527 < 1.6543 < 1.757); linear extrapolation
of the slope estimator through coarse+base lands at 1.646 (-0.5%), through
base+fine at 1.521 — the trend is convex, so we report the measured grid
values rather than an extrapolated point estimate. Remaining biases:
upwind diffusion (above), the D(T) fit band (±9% -> ±4.5% on S_L), and the
truncated-tail bias on the integral estimator. The mixture-averaged value
2.3345 m/s and the literature anchors below are preferential-diffusion
regime numbers — CONTEXT ONLY.

Literature (bibliography verified via Crossref; abstracts carry no phi=1
300 K point values, so no numeric correlation constants are quoted —
ILLUSTRATIVE regime context only): Aung K.T., Hassan M.I., Faeth G.M.,
"Flame stretch interactions of laminar premixed hydrogen/air flames at
normal temperature and pressure", Combustion and Flame 109:1-24 (1997),
DOI 10.1016/S0010-2180(96)00151-4; Krejci M.C., Mathieu O., Vissotski A.J.,
Ravi S., Sikes T.G., Petersen E.L., et al., "Laminar Flame Speed and
Ignition Delay Time Data for the Kinetic Modeling of Hydrogen and Syngas
Fuel Blends", ASME Turbo Expo GT2012-69290 (2012), DOI 10.1115/GT2012-69290.

### Engine/parity status

- Suite test `tests/stdlib/chemistry/test_flame1d.sio` (thermo/D pins +
  clamped-RK4 substep pins vs the replica) PASS on BOTH lean_single and
  madaros.
- Byte-parity probe (9 integer-scaled pins: 2 substep states, D(1000),
  rho(1500), cp(1500)) byte-identical between madaros and lean_single
  (strip the madaros banner with `awk '/^   Output: /{f=1; next} f'`).
- `gri30_h2.sio` gained `g30_cp_rhs` (isobaric RHS, pinned vs Cantera at
  0.036%) and `g30_cp_rhs_out` (out-param sibling): madaros miscompiles
  tuple-array destructures reached from loop bodies (same engine class as
  the documented 1000-step UV loop issue), so the flame substepper uses the
  out-param form on both engines. The full demo runs on lean_single only
  (madaros rc=182 wall on heavy loops, pre-existing).
- New engine quirks found this leg: madaros E038 rejects `&y`+`&!y`
  same-call aliasing that lean accepts (worked around with a separate input
  array); lean_single aliases two simultaneously-held tuple-array returns
  (avoided via the out-param substep signature).

### Runs

```
# Sounio demo (base grid, ~35 min on lean_single)
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/chemistry/h2_flame_speed_demo.sio
# Cantera reference (matched kinetics + transport; ~25 s)
PYTHONPATH=/tmp/pylibs python3 benchmarks/chemistry/flame1d_cantera.py
# Python replica: base + fine + coarse grids (~2.5 h)
python3 benchmarks/chemistry/flame1d_replica.py
```
