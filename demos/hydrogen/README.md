# demos/hydrogen — Metal-Hydride Hydrogen Compression, Uncertainty-Quantified

A Sounio demonstration written for **Dr. Emmanuel Stamatakis** (NCSR Demokritos,
Integrated Hydrogen Laboratory / H2Lab; CYRUS P.C.).

It takes the single-stage core of the metal-hydride (MH) thermal compression
concept he has published on for a decade — and shows what Sounio adds on top of
a conventional implementation in Python/MATLAB: **uncertainty propagation and
reproducibility living inside the language itself**, not in an external toolbox.

## Run

```bash
bin/souc run demos/hydrogen/mh_stage_uq.sio                              # stage model (any engine)
SOUNIO_SOUC_ENGINE=lean_single bin/souc run demos/hydrogen/mh_cascade_uq.sio  # cascade (see note)
```

Deterministic (seeded xorshift PRNG): every run prints the same numbers and
ends with `MH_STAGE_UQ_OK` / `MH_CASCADE_UQ_OK`.

**Engine note (hit live while building this demo):** the cascade file imports
Sounio's own `stdlib/epistemic/pce.sio`, which calls libm through
`extern "C"`. On the default Madaros engine the native path currently
mis-lowers extern f64 returns (exp→0, observed 2026-07-26; the suite keeps
its FFI tests on the lean_single stage2 engine). Until that lands, run the
cascade with `SOUNIO_SOUC_ENGINE=lean_single` as above. The stage file uses
pure-Sounio math only and runs on both engines with identical numbers.

## The model (`mh_stage_uq.sio`)

Single-stage MH thermal compressor with a LaNi5-type hydride:

- van't Hoff equilibrium: `ln P = ΔS/R − ΔH/(R·T)` (P in bar)
- ΔH = 30.8 ± 1.5 kJ/mol H2, ΔS = 108 ± 2 J/(mol·K) (1σ)
- cycle: absorb at 293.15 K (20 °C water), desorb at 353.15 K (80 °C water)
- thermal demand per cycle: `Q = ΔH + C_eff·ΔT`, C_eff = 400 ± 80 J/(mol H2·K)

Nominal results (physically sane vs. the MH-compression literature):

| quantity | value |
| --- | --- |
| P_eq at 20 °C (absorption) | 1.42 bar |
| P_eq at 80 °C (delivery) | 12.18 bar |
| single-stage compression ratio | 8.56 |
| thermal demand | 54.8 kJ/mol H2 = **7.6 kWh/kg H2** |

## What the demo proves about UQ

The same model is evaluated three ways, in one file, in one deterministic run:

1. **Nominal point** — what a conventional script gives you.
2. **GUM first-order propagation** (JCGM linear law) — analytic; exact for
   `ln P`, but only linear through `exp()`.
3. **20,000-sample Monte Carlo** through the identical van't Hoff map.

The comparison is the punchline:

- **Jensen effect**: `exp()` amplifies and skews — the *mean* delivery
  pressure sits ~17 % above the nominal point. A nominal-point design
  systematically misjudges the stage.
- **Linear GUM underestimates σ_P by ~20 %** (GUM/MC ≈ 0.80) and its
  2σ band covers only ~91 % of MC samples instead of the nominal ~95 %.
- Energy demand, being linear in the parameters, is where GUM is exact
  (σ_Q = 5.0 kJ/mol) — the demo says so, honestly.

Conclusion printed by the program itself: first-order UQ is not enough for
van't Hoff; higher-order UQ is the right tool — and Sounio's stdlib carries a
complete Polynomial Chaos Expansion implementation (`stdlib/epistemic/pce.sio`)
for exactly that, alongside GUM and p-boxes.

## The cascade model (`mh_cascade_uq.sio`)

Three identical LaNi5 stages chained from a 1 bar electrolyser supply to the
200 bar HRS delivery target, following the multi-stage architecture of his
2020 paper. Per stage: `ln r = ΔH/R·(1/T_cold − 1/T_hot) + ln η` — ΔS cancels
in the ratio; η = 0.80 ± 0.05 lumps hysteresis, plateau slope and drops.

Crucially, the demo models **batch correlation**: one alloy batch fills the
whole cascade, so ΔH (±2.5 kJ/mol) is sampled once per virtual batch and
shared by all stages. Independent-stage math underestimates σ_P3 by ~2x.

Three UQ levels in one deterministic run, validated against a SciPy oracle:

| level | mean P3 | std P3 | note |
| --- | --- | --- | --- |
| nominal | 321.1 bar | — | "comfortable margin" |
| GUM (first-order) | 321.1 (pinned) | 171.4 bar | 2σ band dips to 110 bar |
| **PCE (`stdlib/epistemic`)** | **370.3 bar** | **212.7 bar** | exact lognormal moments |
| Monte Carlo (20 000) | 367.3 bar | 206.4 bar | **reliability ≥ 200 bar: 81.1 %** |

The design statement: the nominal point promises 321 bar; uncertainty says
**~19 % of alloy batches miss the 200 bar target**, and the skew-correct
mean sits +15 % above nominal. Reliability numbers like this — not nominal
margins — are what HRS procurement and his techno-economic studies need.

## The caprock model (`caprock_seal_pbox.sio`) — his newest research line

His latest paper (*Hydrogen* 6(4):91, 2025) reviews caprock integrity for
underground hydrogen storage: wettability, interfacial tension and diffusion
control the seal — and H2 contact-angle data is scarce and conflicting.
That is an **epistemic** uncertainty problem, and it gets the deepest tool:
a **p-box** that separates aleatory scatter from epistemic ignorance.

Young–Laplace breakthrough pressure `P_c = 2γ·cos(θ)/r` with γ ~ N(70,5) mN/m
and shale pore throats r ~ lognormal(15 nm) as aleatory; the contact angle
θ ∈ [10°, 40°] as an **interval** (treating it as Gaussian would be a
category error — the data is too scarce to name a distribution). Question:
does the caprock seal an 800 m column?

| analysis | result |
| --- | --- |
| GUM at θ = 25° (standard practice) | P_c = 8.46 ± 4.27 MPa — one number, ignorance hidden |
| MC per θ scenario | reliability 66.5 % (θ=10°) … 59.4 % (θ=25°) … 46.8 % (θ=40°) |
| **p-box on reliability** | **[46.8 %, 66.5 %] — the 19.7-point width IS the ignorance** |
| **value of information** | measuring θ to ±5° collapses the width 19.7 → 5.7 pp |

The closing statement is a research-management argument, not just physics:
*the ~20-point gap is pure epistemic ignorance about wettability — one ±5°
contact-angle campaign cuts it ~3x. That is the experiment to fund next.*
For the author of a review that says "wettability data for H2 is scarce and
conflicting", this prices exactly that gap.

Runs on the default engine (pure-Sounio math, no extern):
`bin/souc run demos/hydrogen/caprock_seal_pbox.sio` → `CAPROCK_SEAL_PBOX_OK`.

## The deep caprock model (`caprock_integrity_v2.sio`)

The robust version of the caprock analysis — depth-coupled physics, two
failure mechanisms, three candidate formations, and self-verifying epistemic
bounds, in one deterministic run on the default engine:

- **Depth coupling**: z = 800 m → T = 35 °C, P ≈ 8.1 MPa (geothermal +
  hydrostatic); γ(P,T) correlation; threshold Δρ·g·z = 7.46 MPa.
- **Two mechanisms**: capillary seal (Young–Laplace) AND diffusive lifetime
  τ = L²/(π²·D_eff) vs a 30-year design life, intact vs micro-fractured.
- **Three formations** from the review's ranking, each with its own
  epistemic θ interval and pore-throat distribution.
- **Corner-exact p-boxes**: P_c is strictly decreasing in θ, so the interval
  corners are exact bounds — and the run *verifies the monotonicity
  numerically before relying on it*.
- **Robustness receipts**: MC convergence (N=1k/10k/50k: 60.5 → 58.5 →
  58.1 %) and seed invariance (58.10 vs 58.20 %) printed by the run itself.

| formation | seal p-box (800 m) | verdict |
| --- | --- | --- |
| anhydrite (evaporite) | [99.4 %, 99.7 %] | robust seal — the review's favorite |
| shale | [42.4 %, 66.6 %] | **indeterminate — 24 pp of pure ignorance** |
| mudstone | [14.9 %, 29.3 %] | fails the 800 m column at these throats |

Diffusion: intact caprock holds for 8 000–800 000 yr; micro-fracturing
(D_eff ~ 1e-7) collapses τ to 13–80 yr — below the design life. The closing
statement: *capillary wettability decides the site, fractures decide the
lifetime; measure θ and image the fractures — that is where certainty is
bought.*

Runs on the default engine: `bin/souc run demos/hydrogen/caprock_integrity_v2.sio`
→ `CAPROCK_INTEGRITY_V2_OK`. (v1 kept for its GUM-vs-p-box teaching contrast.)

## The measurement-strategy model (`sobol_voi.sio`) — the deepest cut

For the indeterminate formation (shale), which experiment buys certainty —
and in which order? Two parts, both validated against independent oracles:

1. **Sobol variance decomposition** of ln P_c: analytic log-space indices vs
   Monte Carlo Saltelli/Jansen total-order estimators (N = 10 000, two
   independent sample matrices). Result: pore-throat spread carries ~98 %
   of the aleatory variance; IFT only ~2 %.
2. **Value-of-information sequencing**: baseline shale p-box [42.4 %, 66.6 %]
   (width 24.3 pp). Option A — measure θ to ±7.5°: width halves to 12.0 pp.
   Option B — image pore throats (σ_log r halved): mid reliability rises
   58 → 66 %, **but the p-box WIDENS to 44.3 pp**.

The non-commutative insight: with less aleatory noise the answer becomes
*more* sensitive to the unknown wettability, so reducing the "wrong"
uncertainty first backfires epistemically. Correct order: **measure θ first,
then image the pores.** That is a research-portfolio argument no
point-estimate workflow can even express.

Runs on the default engine: `bin/souc run demos/hydrogen/sobol_voi.sio` →
`SOBOL_VOI_OK`.

## The techno-economic model (`smr_h2_lcoh.sio`) — feasibility as a portfolio

His Energies 2023 SMR/H2 feasibility study, redone the Sounio way:
`LCOH = (CAPEX·CRF + O&M) / annual production` with CAPEX ±20 %, discount
rate ±1.5 pp, specific energy ±2.5 kWh/kg as aleatory, and the capacity
factor as an epistemic **interval** [0.75, 0.90] — site/grid data too scarce
for a distribution. Uses the new **`stdlib/epistemic::pbox`** module.

| analysis | result |
| --- | --- |
| nominal | 3.11 €/kg |
| GUM σ | 0.74 €/kg |
| Monte Carlo (nominal, ±0.02 aleatory) | mean 3.12, σ 0.72, P(LCOH < 4 €/kg) = 88.5 % |
| **p-box on P(< 4 €/kg)** | **[72.3 %, 93.1 %] — 20.8 pp of priced ignorance** |
| Sobol first-order | **CAPEX 71.4 %**, discount rate 17.9 %, CF 6.2 %, efficiency 4.5 % |

The statement a point-estimate study cannot make: *the probability of
beating 4 €/kg is an interval, 71 % of cost variance is CAPEX (negotiate
there, not on electrolyser efficiency at 4.5 %), and the 21-point gap is
what missing capacity-factor data costs.* Runs on the default engine:
`bin/souc run demos/hydrogen/smr_h2_lcoh.sio` → `SMR_H2_LCOH_OK`.

## The stdlib modules (new, reusable)

- **`stdlib/epistemic/pbox.sio`** — the p-box type: corner-exact interval
  bounds on probabilities, width/midpoint/union/intersect/scale/add/mul_pos,
  ignorance ratio. Pure Sounio, green on **both** engines
  (`tests/run-pass/epistemic_pbox_selftest.sio`).
- **`stdlib/epistemic/sobol_indices.sio`** — variance estimators on
  precomputed outputs: Jansen total/first-order, exact log-linear shares.
  Pure Sounio; self-test green under lean_single
  (`tests/run-pass/epistemic_sobol_indices_selftest.sio`). Complements the
  existing full-pipeline `epistemic::sobol` module (Sobol sequences,
  Saltelli sampling, dominance gates) with a dependency-free estimator
  core. Engine note: passing a
  caller array as a slice argument *into* an imported module currently
  segfaults on the Madaros native path (minimal witness: 64-element array —
  same handle-vs-raw family as the syscall6 raw-ref fix in #1455; tracked
  separately). The math is engine-independent; Madaros runs resume once the
  compiler bug lands.

## Why this maps to his work

- *Renewable Energy 148 (2020) 1118–1130* (seven-stage MH compression for
  HRS) and *IJHE 2021* (dual-stage MH2C under thermal management): the stage
  and cascade files are the per-stage core of those system models.
- *Hydrogen* 6(4):91 (2025) (caprock integrity for UHS): the p-box demo
  quantifies the exact measurement gap the review identifies.
- His techno-economic analyses (SMR/H2 feasibility, MH2C market benchmark)
  run sensitivity by hand; here uncertainty is part of the program's value,
  and the run is a reproducible receipt.
- For dual-use / safety contexts (INRASTES is an Energy & **Safety**
  institute; CALIPSO is EDF): deterministic receipts plus epistemic labels
  are the audit trail.

## Honest boundaries

- One stage, flat plateau, no hysteresis/slope, constant C_eff — the
  extension path (multi-stage cascade, plateau slope, heat-recovery) is
  straightforward.
- The MC uses a CLT normal (sum-of-12 uniforms), not an exact Gaussian;
  agreement with a SciPy oracle is within ~2 % on σ_P.
- Units are encoded by naming/comment discipline in this file; Sounio's
  compile-time units binding (`stdlib/units`, QUDT) is landing separately —
  when it does, every equation here becomes dimension-checked by the
  compiler.
