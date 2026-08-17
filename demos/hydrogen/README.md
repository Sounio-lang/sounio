# demos/hydrogen — Metal-Hydride Hydrogen Compression, Uncertainty-Quantified

A Sounio demonstration written for **Dr. Emmanuel Stamatakis** (NCSR Demokritos,
Integrated Hydrogen Laboratory / H2Lab; CYRUS S.A.).

It takes the single-stage core of the metal-hydride (MH) thermal compression
concept he has published on for a decade — and shows what Sounio adds on top of
a conventional implementation in Python/MATLAB: **uncertainty propagation and
reproducibility living inside the language itself**, not in an external toolbox.

## Run

```bash
bin/souc run demos/hydrogen/mh7_reliability.sio                         # 7-stage HRS reliability (flagship)
bin/souc run demos/hydrogen/trieres_chain.sio                           # TRIERES valley: dispensed EUR/kg p-box
bin/souc run demos/hydrogen/valley_chain_epistemic.sio                  # composed twin: subsurface -> compressor -> dispenser
bin/souc run demos/hydrogen/mh_stage_uq.sio                              # stage model
bin/souc run demos/hydrogen/mh_cascade_uq.sio                            # cascade
bin/souc run demos/hydrogen/bayes_pilot.sio                              # value of pilot data (IDM)
bin/souc run demos/hydrogen/hub_chain.sio                                # full chain: delivered EUR/kg
bin/souc run demos/hydrogen/methanation_logk_gate.sio                   # methanation log-K gate
SOUNIO_SOUC_ENGINE=lean_single bin/souc run demos/hydrogen/uhs_brine_calcite.sio  # UHS H2-brine-calcite network (lean_single only, see below)
SOUNIO_SOUC_ENGINE=lean_single bin/souc run demos/hydrogen/site_screening.sio     # epistemic UHS site screening, sourced Greek sites (lean_single only)
```

Deterministic (seeded xorshift PRNG): every run prints the same numbers and
ends with `MH_STAGE_UQ_OK` / `MH_CASCADE_UQ_OK`. All demos run on the default
Madaros engine as well as lean_single. (Historical note: the cascade imports
`stdlib/epistemic/pce.sio`, which calls libm through `extern "C"`; until
#1550 the Madaros native path dropped all but the first extern decl and
mis-evaluated the exp/log builtins — issue #1547, fixed.)

## The flagship (`mh7_reliability.sio`) — his seven-stage HRS compressor, reliability-quantified

His first-author seven-stage paper (*Renewable Energy* 147 (2020) 164–178,
DOI 10.1016/j.renene.2019.08.104) chains seven MH stages on 80 °C heat to
reach 365 bar for 350-bar dispensing — and offers itself explicitly as a
model and tool for sensitivity analysis. This demo takes the offer
literally: the whole chain, plus the batch-to-batch alloy scatter no
nominal-point run can see, as one deterministic receipt on both engines.

The paper's per-stage alloy internals (its Table 3) sit behind a paywall,
so the per-stage ΔH ladder (24–36 kJ/mol H2, batch half-widths
±1.5–2.5 kJ/mol assigned by alloy maturity) is **representative — ours,
labeled as such in the file header, with a swap slot for the real values**.
What is his: both published system-level oracles reproduce exactly.

| oracle (his paper) | demo |
| --- | --- |
| overall compression ratio 18.7 @ 80 °C | 18.700002 |
| delivery pressure 365 bar | 365.000037 bar |
| ratio 41.5 @ 120 °C | 41.500004 |

Nominal chain: per-stage ratios 1.464–1.565 across overlapping temperature
windows (20→35 … 65→80 °C), cumulative pressure 28.6 → 42.5 → 64.0 → 97.6 →
150.2 → 233.3 → 365.0 bar from the implied 19.5 bar supply.

Then the question HRS procurement actually asks: per-stage batch ΔH shifts
as **intervals** (no published batch distributions), lognormal efficiency
scatter (σ_ln η = 0.035) — what is P(P7 ≥ 350 bar), the dispensing gate?

| analysis | P(≥ 350 bar) |
| --- | --- |
| batches at nominal ΔH (η scatter only) | 67.3 % |
| independent batches | 65.3 % (GUM cross-check: 65.27 %) |
| one batch fills all stages (full correlation) | 59.0 % (GUM: 59.85 %) |
| **dependence cost — unobservable without batch data** | **6.2 pp** |
| **distribution-free corner p-box** | **[1.3 %, 99.9 %] — the reliability is an interval, not a number** |

Sobol first-order shares of Var(ln P7) — exact, because the model *is*
linear in log space, cross-checked by a Saltelli/Jansen Monte Carlo whose
indices sum to 99.995 %: efficiency scatter carries **75.0 %** of the
variance; among alloy batches the high-pressure stages dominate (stage 6:
4.7 %, stage 7: 4.3 %; stage 1: 2.7 %). The measurement priority falls out
for free: *efficiency is the big lever; among the alloys, characterize
stages 6–7 first.* And the honest floor: the longest published industrial
MH-compressor campaigns run ~1 year / 10 000 cycles (Tarasov et al.,
*J. Phys.: Energy* 2, 2020), so fleet batch data barely exists — which is
exactly why the deliverable is a p-box, not a point reliability. The
sensitivity analysis his paper offered itself as the tool for, run as one
reproducible receipt: `bin/souc run demos/hydrogen/mh7_reliability.sio` →
`MH7_RELIABILITY_OK`.

## The valley chain (`trieres_chain.sio`) — TRIERES wellhead-to-dispensed cost as a p-box

TRIERES is the EU hydrogen-valley project Demokritos is a paid beneficiary
of (Grant Agreement 101112056, ~€197.5k). This demo prices the valley's core
claim — hybridization lifts utilization — by chaining wellhead production →
compression → storage → **dispensing** and asking the procurement question:
does valley-scale green H2 beat the **€6/kg** dispensing gate?

Sourced inputs are his literature exactly as in `hub_chain.sio` (Energies
2023 CAPEX/O&M/LCOE/specific energy; the 44–89 kWh_th/kg compression span
of his two compressor papers; heat price and tank cycling intervals). The
valley-specific knobs are **illustrative assumptions, labeled in the file
header**: CF ∈ [0.55, 0.80] (the utilization claim on trial — his published
Crete electrolyser CF is [0.35, 0.44]), CAPEX ×[0.8, 1.2], specific energy
±2.5 kWh/kg, dispensing [0.50, 1.50] €/kg (no citable source; swap slots).

| analysis | result |
| --- | --- |
| delivered cost interval | **[4.46, 9.02] €/kg** (nominal 6.42 — misses the gate) |
| conventional independence MC (n = 20 000) | mean 6.44, σ 0.52, **P(< €6/kg) = 20.8 %** — one number, pure assumption |
| GUM first-order cross-check | σ 0.52, P(< 6) = 21.2 % — agrees |
| **distribution-free corner p-box** | **[0 %, 100 %] — undetermined, no independence assumption** |

The one-at-a-time corner table names the flippers: the **waste-heat
contract** (78.6 % at its favourable end) and the **dispensing tariff**
(76.2 %) each carry the gate alone; utilization CF (1.9 %) cannot — but the
coalition CF + heat + dispensing flips it to 100 %. Sobol first-order
shares of Var(D) (GUM-linear exact vs centered Jansen MC, sum 99.2 % — the
small gap is genuine interaction share from the energy×price products):
**heat price 32.2 % and dispensing 32.4 % dominate**, compression energy
10.3 %, CF 6.1 %, and the electrolyser specs that dominate the literature
(LCOE 2.5 %, CAPEX 5.3 %, specific energy 4.5 %) are minority shareholders.
*For TRIERES: price the heat contract and the dispensing business model
first; CF is a multiplier, not a saviour.* The shares are a map drawn under
the assumption the p-box refuses — the interval is the deliverable, the map
is how you shrink it. Monotonicity machine-checked in Lean 4
(`SounioHydrogenPbox.monotone_event_equiv`). Both engines →
`TRIERES_CHAIN_OK`.

## The composed valley chain (`valley_chain_epistemic.sio`) — the epistemic digital twin

One receipt composing the three house models into a single chain:
**subsurface geochemical loss → compressor reliability → dispensed
EUR/kg**, with p-boxes propagated end-to-end. Composition is by
construction — the component demos are not modified; their models are
re-derived in-file with the same formulas, constants, seeds, and
integer-decision idioms, and Sections B and C of the receipt reproduce
`mh7_reliability.sio` (65.255 / 59.030 %, corner p-box [1.31, 99.89] %)
and `trieres_chain.sio` (20.765 %, [0, 100]) digit for digit. The UHS
brine-calcite 30-yr loss p-box **[0.458073, 2.630814] %** enters as a
*pinned input* (re-derived by `uhs_brine_calcite.sio`, merged #1585),
which keeps the twin chemistry-free.

Both couplings are **ILLUSTRATIVE, labeled, with swap slots**: storage
residence τ = 1 yr gives an availability factor
`f_s = 1 − (L30/100)·(τ/30)`, and compressor reliability enters as an
availability derate `f_c = R` on delivered kg:
`CF_eff = f_s · f_c · CF`. D is decreasing in CF_eff and CF_eff is
increasing in both factors, so the per-stage monotone chains keep the
composed corner p-box **exact** — and it still spans **[0, 100] %**.

The headline is what the coupling does to the *conventional* number:

| analysis | P(dispensed < €6/kg) |
| --- | --- |
| conventional, no-coupling baseline | 20.765 % |
| conventional, composed valley-chain | **3.630 %** (−17.1 pp) |
| decomposition: subsurface only (R = 1) | 20.495 % |
| decomposition: compressor only (L30 = 0) | 3.635 % |
| distribution-free p-box, baseline and composed | [0, 100] % both |

Closing the loop moves the conventional number violently but does not
shrink the honest interval: the compressor's corner p-box on reliability
is itself [1.3, 99.9] %. The one-at-a-time table says the same thing
harsher — post-coupling, **only compressor R can flip the gate alone**
(the eight economic intervals and the subsurface loss each move 0.00
points); Sobol agrees (R carries ~94 % of Var(D) under the conventional
measure, Jansen). *The euro is not in the rock at 1-yr residence — it is
in the reliability data.* Reference engine: lean_single →
`VALLEY_CHAIN_OK`. On Madaros the receipt is byte-identical through
Section H, then hits the same pre-existing Saltelli-machinery SIGSEGV
(#1570 family) as both component demos. Suite coverage:
`tests/run-pass/valley_chain_epistemic_selftest.sio` (both engines).

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

## The methanation log-K gate (`methanation_logk_gate.sio`) — the constant he had to calibrate by hand

The companion to the calcite gate, aimed at the exact constant his
GHG-2025 paper (DOI 10.1002/ghg.2368) had to fix: the methanation
equilibrium. phreeqc.dat anchors `CO3-2 + 10H+ + 8e- = CH4 + 3H2O` at
log K0 = 41.071, ΔH = −61.039 kcal/mol (25 °C); a naive constant-ΔH
van't Hoff then extrapolates it to storage temperatures without
protest. This demo puts a gate on that extrapolation — an honest,
labeled heuristic: inside ±40 K of the anchor, report the point (with a
0.25-log-unit anchor band); past it, **refuse the point** and report an
interval that widens 0.05 log units per kelvin of overreach.

| temperature | naive log K | honest answer | width |
| --- | --- | --- | --- |
| 50 °C | 37.610 ± 0.25 | in-window: trusted point | 0.5 |
| 90 °C | 33.063 | **REFUSED** — [31.563, 34.563] | 3.0 |
| 120 °C | 30.260 | **REFUSED** — [27.260, 33.260] | 6.0 |
| 150 °C | 27.854 | **REFUSED** — [23.354, 32.354] | 9.0 |

A toy first-order 30-yr H2-loss proxy (labeled ILLUSTRATIVE in the
file; rate anchored to Bo et al. 2021's 0.72–2.76 % field range, DOI
10.1016/j.ijhydene.2021.03.116) shows what the refusal is worth: the
naive workflow prints 2.00 / 0.40 / 0.10 % at 90/120/150 °C and moves
on; the same proxy through the honest band spans **[0.85, 4.68] %**
(5.5×) at 90 °C, [0.072, 2.24] % (31×) at 120 °C and
[0.0076, 1.33] % (177×) at 150 °C. The mapping is not the problem —
the extrapolated log K is. A p-box on breaching Bo's worst field case
(2.76 %) comes back **[0 %, 100 %] at 90 °C: undetermined.**

The file's center is a **calibration slot**: his paper's fix was a
hand-calibrated log K(T) predicting *less* methanation than the
database default — and that expression is paywalled. So the demo
encodes only the abstract-level finding as a labeled placeholder
interval (calibrated log K = naive − [2, 6]; toy 90 °C loss drops from
2.0 % to [0.064, 0.637] %), with a two-line swap point inviting his
real expression. And it closes with why the phantom CH4 dies either
way: even the *low* end of the 150 °C band is ~10^23 — near-total
conversion on thermodynamics alone, which is exactly why an
equilibrium constant cannot price the kinetic losses that real UHS
sites see (DOI 10.1016/j.est.2023.106737).

Reproducibility is engineered, not hoped for: the two compiler
backends' f64 printers round differently and float contraction can flip
a borderline Monte Carlo comparison, so every number is emitted by a
digit-wise integer printer and the MC counts with a pure-integer
comparison — **byte-identical output on Madaros and lean_single, by
construction**. Both engines → `METHANATION_LOGK_GATE_OK`.

## The UHS geochemistry network (`uhs_brine_calcite.sio`) — the whole H2–brine–calcite system, with bands

The two gate demos gate *constants*. This one integrates the *network*
his GHG-2025 paper models — H2(aq) methanation coupled to PWP calcite
dissolution/precipitation — as a Sounio epistemic CRN
(`chemistry::kinetics::simulate_general_epistemic`), delivering what a
point-value PHREEQC run cannot: **a native GUM 1σ band and interval /
p-box corner bounds carried through the 30-year integration.**

The paper is closed access, so the file is explicit about provenance
(labels in the header): the **network skeleton** comes from the abstract
(negligible interactions above ~70 °C; considerable H2 consumption
possible at 25–50 °C in a limited-H2 model) plus **public anchors** —
the Plummer-Wigley-Parkhurst calcite rate constants transcribed in
phreeqc.dat, the phreeqc.dat calcite log K = −8.45 with van't Hoff ΔH,
a public H2 Henry constant, and Bo et al. 2021's 0.72–2.76 % 30-yr field
losses. Everything the paper holds that we cannot read is an
**AWAITING-AUTHOR-DATA slot**: the calibrated log K(T) expression, the
kinetic methanation rate law (a 2-line slot function, currently an
ILLUSTRATIVE Bo-anchored pseudo-sink switched off above 70 °C as an
abstract-level encoding), the brine-solubility tuning, and the published
trajectories — so parity is abstract-level only and says so.

Headline numbers (central parameters; brine and reactive area labeled
OUR CHOICE in the file):

| output (30 yr) | 25 °C | 50 °C | 90 °C |
| --- | --- | --- | --- |
| H2 loss, central | 1.710 % | 1.172 % | 0.000 % (slot: k_m = 0 > 70 °C) |
| H2 loss p-box | [0.458, 2.631] % | [0.194, 2.324] % | [0, 0] % |
| calcite net p-box (mmol/L) | [0.683, 5.383] | [−0.838, 3.683] | [−0.940, −0.670] |
| native GUM σ(calcite) | 3.33 µmol | 7.32 µmol | UNSTABLE-WITHHELD |

Two findings the file prints instead of hiding. First, the calcite net
balance is **non-monotone in the reactive area** (fast-dissolution
corners equilibrate early at *lower* cumulative net loss; the maximum is
interior — see the A-scan), so pure corner transfer would be unsound and
the p-box is built from an enumerated corner + interior scan, labeled as
grid evidence rather than proof; H2-loss monotonicity in k_m is
comparison-arguable and labeled unproven. Second, the house engine's
per-step GUM propagator has its own caveats — the band scales with dt
and does not accumulate, sub-1e-6 σ values hit a sqrt-convergence floor
(flagged ENGINE-FLOOR), and at 90 °C the J² amplification diverges, so
the band is **refused** there rather than printed. All of this is
labeled E1–E6 in the file header (E6: lean_single aliases returned
arrays across calls, so the demo extracts every run's scalars before the
next call — do not refactor it to collect-then-print).

Engine coverage: **lean_single only** — on current main, Madaros fails
"visibility preflight" on *any* `chemistry::kinetics` import (reproduced
with the repo's own `tests/stdlib/chemistry/test_kinetics_epistemic_ensemble.sio`;
pre-existing blocker, not from this demo). Run:

```bash
SOUNIO_SOUC_ENGINE=lean_single bin/souc run demos/hydrogen/uhs_brine_calcite.sio
```

Suite coverage: `tests/run-pass/uhs_brine_calcite_selftest.sio` pins the
25 °C trajectory and band against an independent Python delta-method
replica (`UHS_BRINE_CALCITE_SELFTEST_OK`).

## The site screen (`site_screening.sio`) — real Greek UHS candidates, sourced, through the network and the chain

The epistemic UHS site-screening brief package: three **real, public,
cited** Greek H₂-storage candidate formations — S1 South Kavala depleted
gas field (T = 95 °C *measured*, HRADF 2020; 2 TWh H₂ per HyUSPRe D1.3),
S2 Pentalofos and S3 Eptachori saline-aquifer formations in the
Mesohellenic Trough (Koukouzas 2021, *Energies* 14:3321; T brackets
gradient-derived and *labeled* as such) — each run through the
brine–calcite network at its own temperature bracket, then through the
TRIERES valley chain. Every geological value carries a citation or an
explicit label in `site_screening_data.md` (including a NOT-FOUND list:
no measured formation-water salinity exists for any Greek candidate;
TRIERES GA 101112056 is Corinth-anchored with no public geological
deliverables — stated in the data file and the brief).

Headline (deterministic receipt `SITE_SCREENING_OK`): the sourced
brackets put the three sites in **three different kinetic regimes** —
S1 entirely above the abstract-level 70 °C interaction cutoff (30-yr
loss exactly [0, 0] %), S2 entirely below it ([0.085, 2.276] %), S3
**straddling** it ([0, 1.981] %). At the valley's 1-yr residence all
sites leave f_s within ~0.1 % of 1 and the composed gate probability is
~3.6 % for every site — which candidate hosts the store does not move
the gate; the compressor p-box does. The τ = 10 yr analytic sensitivity
shows where site choice starts to matter.

**The k_m(T) law path (added 2026-08-01).** The original k_m was an
ILLUSTRATIVE pseudo-sink slot with a hard 70 °C step. A dedicated
literature hunt (verdict + citations in `site_screening_data.md`)
supported a sourced temperature law: k_m_eff(T) = k_m × f(T) with f the
**Rosso 1993 CTMI** cardinal-temperature growth model (DOI
10.1006/jtbi.1993.1099) and a cardinal p-box — Tmin [25, 40], Topt
[65, 70], Tmax [75, 90] °C (Zeikus & Wolfe 1972; Tyne 2021; Head 2014 /
Wilhelms 2001). The demo prints **both** paths side by side: the slot is
untouched; the law replaces the S3 cliff with a physical thermal-death
slide to zero inside [75, 90] °C, leaves S1's zero unchanged (95 °C is
above the whole Tmax bracket), narrows S2's p-box to [0.041, 2.041] %
and FLIPS its worst-case corner from cold to warm — with an interior
loss maximum near 60 °C that T-corner extrema miss (caught by the dense
2.5 °C T grid and printed). Ghaedi 2025's own ~70 °C threshold is an
equilibrium-calibration result, not a kinetic law; their rate law is
closed-access (searched, NOT FOUND). The composed gate probability is
unchanged (3.635 %, all sites, both paths) — the compressor still gates.

**Field validation (added 2026-08-01).** The sourced law is checked
against the only field-scale observations with a MEASURED reservoir
temperature AND measured methanation extent: Lehen (Underground Sun
Storage, RAG Austria: 40 °C, ~3 % of injected H₂ converted over 285 d —
Hellerschmied et al. 2024, *Nat. Energy*, DOI
10.1038/s41560-024-01458-1) and Lobodice (town-gas aquifer: 25–45 °C,
H₂ 54→37 % over one 7-month season — Šmigáň 1990 / Buzek 1994 via
Tremosa 2023). The receipt prints per-site observation, source,
predicted p-box, and a BRACKETED / NOT-BRACKETED verdict. Honest
result: the CTMI temperature **shape** is consistent (f > 0 exactly in
the measured 25–45 °C window), but the Bo-2021-anchored ILLUSTRATIVE
magnitude **under-brackets both field extents** (observed lower edges
≈ 107× and ≈ 540× the predicted upper edges) — additive evidence for
recalibrating the magnitude, not the shape; both validated paths are
untouched. A labeled stress test annualizes the observed extents
(upper bound) through the same seeded chain: the τ=1 headline survives
a Lehen-class bloom (gate 3.055 % vs 3.635 %) but moves under a
Lobodice-class one (0.110 %) — the rounding-term conclusion is
conditional, printed as such. Tyne 2021 documents no lag phases
(searched); reservoir Monod parameters exist only as FITTED values —
both negatives documented in `site_screening_data.md`.

**Field calibration (added 2026-08-01, stacked on the field-validation
commit).** The field-falsified Bo-2021 magnitude anchor is replaced by a
FIELD-CALIBRATED p-box, built three ways and printed beside both older
paths (additive only — every validated slot/law/field-validation number
is untouched). (i) FIELD-DERIVED inverse calibration: the network is
bisected (80 steps per corner, no closed form) until it reproduces each
observed extent — LEHEN k_eff ∈ [0.765606, 0.894709], LOBODICE k_eff ∈
[6.308105, 14.708991]. (ii) IN-SITU-MEASURED: Tyne et al. 2021
(*Nature*, DOI 10.1038/s41586-021-04153-3, OA) measured an in-situ
methanogenesis rate of 73–109 mmol CH₄ m⁻³(STP) yr⁻¹ at 29.2–50.7 °C —
bridged to model units as k_eff ∈ [0.499145, 1.064713], with the
normalization-volume ambiguity (never defined in the paper) documented
as a ~1–2-order bridge caveat — larger than the box width itself; the
box is conditional on the per-m³-water reading. (iii) Overlap: LEHEN ∩ TYNE =
[0.765606, 0.894709] is NONEMPTY — two independent in-situ evidences
are mutually consistent at ~40 °C (weak corroboration only). The
calibrated k_m p-box at Topt is [2.041617, 382.433772] —
109×–20451× the falsified lab anchor — with the interpretive layer
(volumetric biomass: Gray 2009 lab-vs-in-situ, Thaysen 2021 bulk-vs-
near-well, Tremosa 2023 ÷50 rescale, Haddad 2022 acceleration)
sourced in `site_screening_data.md` §C1–C6. Effect: 30-yr loss p-boxes
S1 [0, 0] (thermal death, anchor-independent), S2 [15.475, 100],
S3 [0, 100]; S2 f_s(10) falls to [0.666667, 0.948416]; composed τ=1
gate 3.635 / 3.355 / 3.370 % — the headline survives the re-anchoring
under the receipt's linear f_s mapping, but the conditional claim is
kept and sharpened (mapping-limited; annualized reading 0.110 %).
Bo 2021 is additionally documented as an abiotic-isothermal study
(provenance caveat, changes no numbers). A bisection-bracket bug
(hi = 1e4 broke RK4 monotonicity; bracket capped at 100) was caught by
the selftest and fixed in all three files.

Artifacts:

- `site_screening_data.md` — the sourced parameter table (HRADF,
  Energies, Hystories, HyUSPRe, Sci. Rep., CORDIS; accessed 2026-07-31).
- `SITE_SCREENING_BRIEF.md` — the 2–3 page brief draft (figures,
  honest-limitations section, AWAITING-AUTHOR-DATA swap invitation).
- `figures/` — publication-quality PNGs rendered from the demo's own
  stdout by `tools/render_site_figures.py` (matplotlib in the repo
  `.venv`; captions state data provenance; regenerate from a fresh run
  to byte-verify). Figs a–c parse only pre-[A4] FIGDATA kinds and
  regenerate byte-identical from the field-calibrated stdout; fig d
  renders the new FIELD-CALIBRATED band (FANF/PBOXF) against the
  sourced-law band.
- `tools/replica_60c_pins.py` — independent Python RK4 replica used for
  the selftest pins.
- `tools/km_law_predict.py` — independent CTMI + law-path replica
  (corner-vs-dense scan check, law p-box predictions, field-validation
  predictions, field-calibration inverse boxes + KMF, selftest pins).

Suite coverage: `tests/run-pass/site_screening_selftest.sio` pins the
60 °C trajectory against the independent replica, the A2 regime switch,
the k_m corner ordering, the chain analytics, a small seeded MC count,
the CTMI law path (f(Topt) = 1, the 11³ scan's f = 1 catch at
69 °C, a law-scaled trajectory, and the 80 °C law run staying active
above the slot step), and the field-validation path (cardinal-scan pins
at 40/45 °C, a 15-step sub-year law trajectory at Lehen's 40 °C vs the
replica, τ-monotonicity across the 285 d step bracket), and the
field-calibration path (inverse-calibration pins for Lehen and
Lobodice, Tyne bridge arithmetic, CTMI scan at Tyne's 50.7 °C, and the
KMF lower edge above 100× the falsified lab anchor)
(`SITE_SCREENING_SELFTEST_OK`). lean_single only
(same Madaros chemistry-import blocker as the network demo).

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
  and cascade files are the per-stage core of those system models,
  `mh7_reliability.sio` reproduces the seven-stage paper's published
  system-level oracles exactly and computes the batch-uncertainty
  reliability it does not report, and
  their thermal-energy figures (44–89 kWh_th/kg) are the two endpoints of
  the decisive interval in `hub_chain.sio`.
- *Hydrogen* 6(4):91 (2025) (caprock integrity for UHS): the p-box demo
  quantifies the exact measurement gap the review identifies.
- *Energies* 16:6257 (2023) (SMR/H2 feasibility, Crete): the hub-chain
  demo is built entirely from its published parameters — and computes the
  delivered-€/kg uncertainty the paper does not report.
- TRIERES (EU hydrogen valley, Grant Agreement 101112056; Demokritos is a
  paid beneficiary): `trieres_chain.sio` prices the valley's utilization
  claim — the wellhead-to-dispensed €/kg p-box says the €6/kg gate is
  decided by the heat contract and the dispensing tariff, not by the
  electrolyser spec sheet.
- *GHG: Sci. & Technol.* (2025), DOI 10.1002/ghg.2368 (H2–brine–calcite
  geochemistry) and the 2026 geothermal scaling paper: the two gate demos
  arm the exact failure mode they report — extrapolated equilibrium
  constants used past their evidence — with an interval and a refusal
  instead of a silent wrong answer. `vanthoff_gate.sio` gates the calcite
  pK; `methanation_logk_gate.sio` gates the very log K his GHG paper had
  to hand-calibrate, and carries a two-line slot for his expression.
- His techno-economic analyses run sensitivity by hand; here uncertainty
  is part of the program's value, and the run is a reproducible receipt.
- For dual-use / safety contexts (INRASTES is an Energy & **Safety**
  institute, and he leads innovation at CYRUS S.A.): deterministic receipts
  plus epistemic labels are the audit trail.

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
