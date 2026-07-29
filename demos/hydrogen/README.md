# demos/hydrogen — Metal-Hydride Hydrogen Compression, Uncertainty-Quantified

A Sounio demonstration written for **Dr. Emmanuel Stamatakis** (NCSR Demokritos,
Integrated Hydrogen Laboratory / H2Lab; CYRUS P.C.).

It takes the single-stage core of the metal-hydride (MH) thermal compression
concept he has published on for a decade — and shows what Sounio adds on top of
a conventional implementation in Python/MATLAB: **uncertainty propagation and
reproducibility living inside the language itself**, not in an external toolbox.

## Run

```bash
bin/souc run demos/hydrogen/mh_stage_uq.sio                              # stage model
bin/souc run demos/hydrogen/mh_cascade_uq.sio                            # cascade
bin/souc run demos/hydrogen/bayes_pilot.sio                              # value of pilot data (IDM)
bin/souc run demos/hydrogen/hub_chain.sio                                # full chain: delivered EUR/kg
```

Deterministic (seeded xorshift PRNG): every run prints the same numbers and
ends with `MH_STAGE_UQ_OK` / `MH_CASCADE_UQ_OK`. All demos run on the default
Madaros engine as well as lean_single. (Historical note: the cascade imports
`stdlib/epistemic/pce.sio`, which calls libm through `extern "C"`; until
#1550 the Madaros native path dropped all but the first extern decl and
mis-evaluated the exp/log builtins — issue #1547, fixed.)

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
what missing capacity-factor data costs.* Runs on both engines:
`bin/souc run demos/hydrogen/smr_h2_lcoh.sio` → `SMR_H2_LCOH_OK`.

## The pilot-data pricer (`bayes_pilot.sio`) — the value of data, in advance

A UHS pilot runs cyclic pressure-hold tests; before any data the per-cycle
hold probability is **vacuous** (p ∈ [0, 1], no prior shape). Three
inference styles are compared cycle-by-cycle as a deterministic pilot log
accumulates: frequentist k/n, Bayesian Beta(1,1) mean, and the **Imprecise
Dirichlet Model** (Walley 1996, JRSS-B 58:3) whose posterior is an exact
interval `[k/(n+s), (k+s)/(n+s)]` — no sampling, no prior invented.

| after 30 cycles (27 holds) | value |
| --- | --- |
| frequentist | 0.900 — "certified!" |
| Beta(1,1) mean | 0.875 |
| **IDM guarantee (lower bound)** | **0.844 — not yet** |
| cycles to a 0.90 *guarantee* at assumed fleet rate 0.93 | ~60 |

*The 30 cycles between the point estimate's "yes" and the interval's
"yes" are the price of the prior you refused to invent — and the exact
size of the pilot campaign to fund. At the pilot's own 0.90 rate the
guarantee never reaches 0.90 at any campaign size — the honest caveat.* Pure rational arithmetic, both engines:
`bin/souc run demos/hydrogen/bayes_pilot.sio` → `BAYES_PILOT_OK`.

## The full chain (`hub_chain.sio`) — delivered cost, decision-grade

The chain nobody closes: production → compression → storage → **delivered
€/kg**, every number from the Demokritos/FORTH literature (Energies
16:6257 Crete case: 50 MW PEM, 46.4 kWh/kg, €1500/kW, LCOE 0.046–0.052
€/kWh; MH compression energy 44–89 kWh_th/kg spanning his dual-stage and
seven-stage papers; €500/kg tank storage cycling 37.29×/yr nominal, epistemic interval [30, 45]).

Delivered cost is **monotone in every epistemic input**, so the p-box is
exact by corner evaluation — machine-checked in Lean 4
(`formal/lean4/SounioHydrogenPbox.lean`, `monotone_event_equiv`).

| analysis | result |
| --- | --- |
| stage intervals | production [3.92, 4.66], compression [0.22, 1.78], storage [0.95, 1.43] €/kg |
| delivered interval | **[5.11, 7.92] €/kg** (nominal 6.28) |
| **p-box on P(< 6 €/kg)** | **[0 %, 100 %] — the decision is undetermined** |
| ignorance decomposition | heat price alone: 93.6 % at best corner; compression energy: 50.5 %; **the pair: 98.9 %**; UHS loss 3.3/2.5 % and tank cycling 29.3/0.0 % (negligible for cost) |

*The point estimate says "barely misses 6 €/kg". The chain says the
decision is undetermined, then names the two knobs that decide it:
compression technology choice and the waste-heat contract — his two
compressor papers are literally the two endpoints of the decisive
interval.* Both engines: `bin/souc run demos/hydrogen/hub_chain.sio` →
`HUB_CHAIN_OK`.

## The machine-checked algebra (`formal/lean4/SounioHydrogenPbox.lean`)

Three facts the demos rely on, proved in core Lean 4 (no Mathlib, no
`sorry`, built by the CI lean-proofs job):

1. **Jensen / variance gap (n = 3)**: `(Σxᵢ)² ≤ 3·Σxᵢ²` via the Lagrange
   identity — why nominal-point designs underestimate the mean.
2. **Correlated-sum variance**: `var9(X+Y+Z) = Σvar9 + 2·Σcov9`, with the
   corollary that nonnegative batch covariance can only inflate cascade
   variance — the algebra behind "batch ΔH correlation doubles σ".
3. **Monotone p-box propagation**: on a 2-point support, the sub-level
   event of `f(X)` at `f(x₁)` equals that of `X` at `x₁` — endpoint
   evaluation transfers p-box bounds through a monotone chain with **no
   independence assumption**.

## The receipt (`formal/lean4/SounioHydrogenReceipt.lean`)

The Python oracle layer is retired: the demo **numbers themselves** are
theorems. Over exact rationals (`Rat`), closed by `native_decide` on
bignum arithmetic — no floats anywhere:

- `crf_bracket`: CRF(7 %, 25 y) = `(7/100)·(107/100)²⁵/((107/100)²⁵−1)`
  lies in `[0.0857, 0.0859]` — proved, not asserted; nothing
  transcendental enters the receipt unproven.
- `idm_final` / `idm_width`: the guarantee interval is exactly
  `[27/32, 29/32]`, width exactly `1/16`.
- `crossing_order`: the frequentist certifies exactly at cycle 30
  (`24/27 < 0.90 ≤ 27/30`), the guarantee lags.
- `campaign_size`: `0.90·2/(0.93−0.90) = 60` — exact.
- `idm_never_crosses`: at a true 0.90 rate the lower bound never reaches
  the 0.90 gate — proved for **all** `k, n` by `omega`, not numerically.
- `delivered_bracket` / `nominal_bracket`: the chain's
  `[5.113, 5.114]` / `[7.920, 7.921]` corner sandwiches and the
  `[6.277, 6.278]` nominal — the `[5.11, 7.92] €/kg` the demo prints,
  proved.

What a script checks, this file proves.

## The extrapolation gate (`vanthoff_gate.sio`) — his own failure mode, armed

His GHG 2025 paper showed PHREEQC defaults + van't Hoff extrapolation of
the methanation equilibrium constant produce **misleading** results; his
2026 geothermal paper models calcite scaling with the same class of
extrapolation. This demo takes the shared core — calcite solubility,
pK0 = 8.48 at 25 °C, ΔH ∈ [−12, −7] kJ/mol epistemic — and asks when an
extrapolation stops being a fact and becomes a guess.

Working in pK units, the arithmetic spine is **exactly rational** — and
machine-checked in `formal/lean4/SounioHydrogenVanthoff.lean`
(corner-exactness of linear maps, per-mille pK sandwiches, the SI
straddle, the gate verdicts).

| temperature | pK interval | width | SI interval (marginal brine) | verdict |
| --- | --- | --- | --- | --- |
| 25 °C | [8.480, 8.480] | 0 | [−0.300, −0.300] | CERTAIN_NO_SCALE |
| 60 °C | [8.609, 8.701] | 0.09 | [−0.171, −0.079] | CERTAIN_NO_SCALE |
| 90 °C | [8.699, 8.856] | 0.16 | **[−0.081, +0.076]** | **UNDETERMINED** |
| 150 °C | [8.842, 9.101] | 0.26 | [+0.062, +0.321] | CERTAIN_SCALE |

The point estimate (ΔH = −9.61) at 90 °C says SI = **+0.001** — a point
geochem code reports "scale" on a rounding artifact. The p-box on the
scaling decision is **[2.2 %, 97.2 %]**. At 150 °C the constant-ΔH and
ΔCp-corrected models' intervals **don't overlap** — model-form
uncertainty, invisible to any single-code run. *His GHG-2025 fix was a
better correlation; the deeper fix is a code that knows when
extrapolation has become a guess.* Both engines → `VANTHOFF_GATE_OK`.

## The stdlib modules (new, reusable)

- **`stdlib/epistemic/pbox.sio`** — the p-box type: corner-exact interval
  bounds on probabilities, width/midpoint/union/intersect/scale/add/mul_pos,
  ignorance ratio. Pure Sounio, green on **both** engines
  (`tests/run-pass/epistemic_pbox_selftest.sio`).
- **`stdlib/epistemic/sobol_indices.sio`** — variance estimators on
  precomputed outputs: Jansen total/first-order, exact log-linear shares.
  Pure Sounio; self-test green on **both** engines
  (`tests/run-pass/epistemic_sobol_indices_selftest.sio`). Complements the
  existing full-pipeline `epistemic::sobol` module (Sobol sequences,
  Saltelli sampling, dominance gates) with a dependency-free estimator
  core. (The Madaros slice-arg segfault this module surfaced — implicit
  borrow of a caller array into a `&[f64; N]` parameter — was fixed in
  #1545, closing issue #1510.)

## Why this maps to his work

- *Renewable Energy 147 (2020) 164–178*, DOI 10.1016/j.renene.2019.08.104
  (seven-stage MH compression for HRS) and *IJHE 46 (2021) 29272–29287*
  (dual-stage MH2C under thermal management): the stage
  and cascade files are the per-stage core of those system models, and
  their thermal-energy figures (44–89 kWh_th/kg) are the two endpoints of
  the decisive interval in `hub_chain.sio`.
- *Hydrogen* 6(4):91 (2025) (caprock integrity for UHS): the p-box demo
  quantifies the exact measurement gap the review identifies.
- *Energies* 16:6257 (2023) (SMR/H2 feasibility, Crete): the hub-chain
  demo is built entirely from its published parameters — and computes the
  delivered-€/kg uncertainty the paper does not report.
- *GHG: Sci. & Technol.* (2025) (H2–brine–calcite geochemistry) and the
  2026 geothermal scaling paper: the van't Hoff gate demo arms the exact
  failure mode they report — extrapolated equilibrium constants used
  past their evidence — with an interval and a refusal instead of a
  silent wrong answer.
- His techno-economic analyses run sensitivity by hand; here uncertainty
  is part of the program's value, and the run is a reproducible receipt.
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
