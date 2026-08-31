# Epistemic screening of Greek underground H₂-storage candidates

**A deterministic, fully-sourced first look for the TRIERES conversation**

*Draft brief — all numbers re-derive from `demos/hydrogen/site_screening.sio`
(lean_single, seeded, receipt `SITE_SCREENING_OK`); geology table with
per-value citations in `demos/hydrogen/site_screening_data.md`. Figures
rendered from the demo's own stdout by
`demos/hydrogen/tools/render_site_figures.py`.
Revision 2026-08-01: the k_m slot is now printed side-by-side with a
sourced methanation temperature law (Rosso 1993 CTMI; §3, §6.1), and the
law is checked against measured field-scale in-situ methanation
(§4, field validation). Same-day second revision: the falsified
magnitude anchor is REPLACED by a field-calibrated k_m p-box built from
inverse calibration plus Tyne 2021's measured in-situ rate
(§4, field calibration; §5, §6.1 re-checked).*

## 1. Executive summary

We screened three real, publicly documented Greek H₂-storage candidate
formations through the validated H₂–brine–calcite kinetic network
(Ghaedi et al. 2025 skeleton, public PHREEQC/PWP rates) **at each site's
own sourced temperature bracket**, and pushed the resulting 30-year
H₂-loss p-boxes through the TRIERES wellhead-to-dispensed cost chain to
the 6 EUR/kg gate.

The headline is a negative result of the useful kind: **at the valley's
1-year storage residence, whichever candidate hosts the store barely moves
any number that matters** — the availability factor f_s stays within
~0.1 % of 1 for every site, the gate probability sits at ~3.6 %
(composed conventional) for all three, and the distribution-free p-box
on beating 6 EUR/kg is [0, 100] % everywhere because the compressor
reliability p-box, not the rock, is the gate. Site choice starts to
matter at longer residence (τ = 10 yr sensitivity, §5) and in the
kinetic-regime structure the sourced temperatures reveal (§4): the
three sites sit in three different regimes of the same public network.
**This conclusion is conditional on the reservoir not hosting a
Lobodice-class active methanogen bloom (§4, field validation).**
A 2026-08-01 revision adds a **sourced methanation temperature law**
(Rosso CTMI with a cardinal-temperature p-box from primary literature)
printed side-by-side with the original slot: S1's zero is unchanged
(now from thermal death, not a flag), S3's hard 70 °C cliff becomes a
smooth physical slide to zero inside [75, 90] °C, and S2's worst-case
corner flips from cold to warm — while the gate probability stays
3.635 % on **both** paths for every site. The same revision adds a
**field validation** of that law against the only measured in-situ
methanation datasets with a sourced temperature (Lehen, 40 °C;
Lobodice, 25–45 °C): the temperature **shape** survives field contact,
while the illustrative k_m **magnitude** under-brackets both measured
field extents by ~2–3 orders of magnitude — an honest, quantified
pointer at the one piece still awaiting real data (§4, §6.1). The
receipt's stress test computes what that gap means for the headline:
at τ = 1 the gate survives a Lehen-class bloom (3.055 % vs 3.635 %)
but **not** a Lobodice-class one (0.110 %) — the rounding-term
conclusion is conditional on no Lobodice-class bloom, and we print it
that way. These numbers come from **two different mechanisms answering
two different questions** — do not conflate them: the screening gates
(3.635 %, and 3.355/3.370 % below) use the receipt's linear f_s
mapping applied to the integrated 30-yr loss p-boxes; the 0.110 % is a
labeled **upper-bound what-if** that directly annualizes the OBSERVED
7-month Lobodice extent (17–31.5 % → 29.6–54.8 %/yr), bypassing both
the calibrated p-box and the mapping. It is not the calibrated model's
τ = 1 prediction; it is the bounding answer to "what if a
Lobodice-class bloom is active". A second, same-day revision then
**re-anchors the magnitude from the field evidence itself** (§4, field
calibration): inverse calibration on the network against the two field
observations gives effective-rate boxes ≈ **[0.77, 0.89]** (Lehen,
40 °C) and ≈ **[6.3, 15]** (Lobodice, leakage-caveated), and Tyne et
al. 2021's measured in-situ rate — 73–109 mmol CH₄ m⁻³ (STP) yr⁻¹ at
29.2–50.7 °C, the only such measurement in the peer-reviewed
literature — bridges (labeled assumptions) to ≈ **[0.50, 1.1]**,
**overlapping the Lehen inverse**. The resulting field-calibrated k_m
p-box ≈ **[2.0, 380]** (109×–20451× the falsified lab anchor; values
rounded to 2 significant figures here — full precision in
`site_screening_data.md` and the receipt) pushed through the same CTMI
law moves the 30-yr loss boxes from ≤ 2.04 % to **[15.475235, 100] %
(S2)** and **[0, 100] % (S3)** — yet the τ = 1 composed gate moves
only to 3.355 % / 3.370 %, because the linear availability mapping is
shallow at τ = 1. All τ = 1 gates read ≈ 3.6 % (per-gate SE ≈ 0.1 %
at n = 20000). The 0.005 pp delta between S1's [0, 0]-loss gate
(3.635 %) and the valley baseline (3.630 %) is sampling noise; the
field-path deltas (S2 3.355 %, S3 3.370 % — ≈ 0.3 pp below S1) are
**systematic, not noise** (they are driven by the wider loss boxes;
seed-scan: the same-seed difference reproduces at 0.31 ± 0.04 pp) and
decision-irrelevant at this magnitude — the distribution-free corner
p-box on the same quantity is [0, 100] %. The
concavity-honest annualized reading of the same kinetics (the stress
test) still collapses it to 0.110 %. **The conditional framing stands,
sharpened: short-horizon economics are robust to the re-anchored
kinetics under the receipt's mapping; long-horizon recoverability and
any Lobodice-class bloom are not.** Every old number stands, printed
beside the new.

**What changes the answer.** Three levers, in order of leverage: (1)
**compressor reliability evidence** — the gate is set by the R p-box,
not the rock, so alloy-batch test data moves the headline the most;
(2) **biological site characterization** — the k_m *magnitude* is now
field-calibrated (§4): the box ≈ [2.0, 380] spans two orders
on its own because the field evidence bounds the magnitude only
one-sidedly at each site (the CTMI shape allows f → 0 at field
temperatures). Ruling a Lobodice-class bloom in or out at the actual
candidate collapses that spread — and it is the same condition the
τ = 1 headline already hangs on; (3)
**measured S2/S3 formation temperatures** — the gradient-derived
brackets are the widest honest input intervals in the screen.

## 2. The sites (every value cited — `site_screening_data.md`)

| | S1 South Kavala | S2 Pentalofos Fm | S3 Eptachori Fm |
|---|---|---|---|
| Type | depleted gas field, turbiditic sands in Miocene anhydrite/salt | turbiditic sandstone saline aquifer, Tsotyli caprock | conglomerates/sandstones, marine shale top |
| Where | offshore N. Aegean, ~6 km off Thassos | Mesohellenic Trough, Grevena sub-basin (W. Macedonia) | same trough |
| Depth | top < 1630 m ss; GWC 1723 m | avg 1500 m (deepest −2544 m) | avg 2000 m |
| **Temperature** | **95 °C, MEASURED** (HRADF 2020 tender document — provenance caveat §6.2) | **[52.5, 69.0] °C** — gradient-derived union bracket (labeled) | **[65, 87] °C** — same construction (labeled) |
| Porosity | 20 % max / 22 % avg | 7–25 % (ftot 0.15) | ~12 % |
| Permeability | 50→100 mD | not published (Hystories default 100 mD, labeled) | not published (same default) |
| Salinity | **not published** — Hystories 100 g/L default, labeled | same | same |
| Sources | HRADF Annex B (2020); HyUSPRe D1.3: 2 TWh H₂ | Koukouzas 2021, Energies 14:3321 T6; Hystories D1.4 §12; Dotsika 2021, Sci. Rep. 11:16291 | same as S2 |

**Geography honesty.** TRIERES (GA 101112056) is anchored at the Motor
Oil **Corinth** refinery and has no public geological deliverables
(CORDIS, accessed 2026-07-31); its storage scope is above-ground. The
three candidates are what the public literature and EU project
databases (Hystories, HyUSPRe) actually document for Greece; the
economics they feed are the TRIERES chain. No salt structures are
documented in Western Macedonia; the Greek evaporite-cavern candidates
(Trifos, Achira, Kefalonia, Corfu) sit in the Ionian zone.

## 3. Method (composition by construction)

- **Loss network**: the H₂–brine–calcite CRN of
  `uhs_brine_calcite.sio` (#1585), re-derived in
  `site_screening.sio` with identical formulas/constants — public PWP
  calcite kinetics k(T), phreeqc.dat calcite equilibrium, and a
  methanation sink k_m run **two ways, both printed**:
  (a) the original ILLUSTRATIVE pseudo-sink **slot** (abstract-level A2
  of Ghaedi 2025: constant below a hard 70 °C step, zero above), and
  (b) a **sourced temperature law** — k_m_eff(T) = k_m × f(T) with f the
  Rosso 1993 cardinal-temperature (CTMI) microbial-growth model (DOI
  10.1006/jtbi.1993.1099) and a cardinal p-box Tmin [25, 40], Topt
  [65, 70], Tmax [75, 90] °C (Zeikus & Wolfe 1972; Tyne 2021; Head 2014
  / Wilhelms 2001 — full citations and the hunt's negative results in
  `site_screening_data.md`). The k_m magnitude interval [0.0048, 0.0187]
  is unchanged (Bo 2021 anchor, ILLUSTRATIVE, now read as the value at
  Topt). Per site and path, the 8 (k_m × A × salt) epistemic corners are
  run across the site's T bracket, giving the site's 30-yr loss
  **p-box** — no independence assumption anywhere.
- **Chain**: `valley_chain_epistemic.sio` (#1587) machinery — f_s =
  1 − (L30/100)(τ/30), f_c = R with the pinned compressor p-box
  R ∈ [0.0131, 0.9989], CF_eff = f_s·f_c·CF into the TRIERES cost
  chain; conventional (independent-uniform, seeded MC n = 20000) and
  distribution-free corner answers. The min–max composition of the
  site-loss and compressor p-box intervals at the epistemic corners is
  mechanically verified in `formal/lean4/SounioHydrogenValleyPbox.lean`
  — a sanity check of the interval arithmetic only, **not** a proof of
  correctness of the reaction network, the cost chain, or the Monte
  Carlo.
  Two stated assumptions: **f_s is linear in τ** — extrapolating the
  30-yr loss to τ ≪ 30 linearly *over*estimates short-residence losses
  within this network, because the modeled cumulative loss is concave
  in time (fastest early, slowing as H₂ depletes). Concavity is
  **verified numerically, not assumed**: integrating the network itself
  to τ = 1…30 at the field-calibrated p-box corners (RK4, same code as
  the demo) gives non-positive second differences everywhere — the
  LO-edge corner is strictly concave (year-1 increment 6.34 %
  decreasing to 2.20 % by year 5) and the HI-edge corners saturate
  (≈100 % in year 1 — itself confirmation that the linear mapping
  understates bloom-speed loss, which is exactly why the stress test
  annualizes directly). Convex early-time (lag-phase) profiles are
  outside this abiotic batch network's form and remain a labeled
  model-form limitation, not a modeled outcome; and **the R p-box is
  representative, not measured** — it is a corner p-box on P(P7 ≥
  350 bar) from
  `mh7_reliability.sio`'s 7-stage compressor ladder under alloy batch
  uncertainty (the source's Table 3 failure data are paywalled), not a
  fitted field-failure distribution.
- **Only temperature is site-specific in this machinery** — depth,
  porosity, permeability shape capacity/injectivity (§6) but have no
  slot in the loss network; no measured Greek formation-water salinity
  exists, so the salting-out interval stays the component demo's
  illustrative [0.70, 1.00]. Stated, not smoothed over.

## 4. Results

**Figure A — per-site loss p-box fan, SLOT vs LAW**
(`figures/fig_a_loss_pbox_fan.png`).
The sourced T brackets put the sites in **three different kinetic
regimes**, now shown on both k_m paths. On the **slot** path: S1
(measured 95 °C) lies entirely above the 70 °C interaction cutoff —
30-yr kinetic loss **exactly [0, 0] %**; S2 (52.5–69 °C) lies entirely
below it — loss p-box **[0.0845, 2.2760] %**; S3 (65–87 °C) **straddles**
the cutoff, so its p-box **[0, 1.9806] %** honestly contains the
interaction-free regime.
On the **law** path (Rosso CTMI, dense 2.5 °C grid): S1 is unchanged —
**[0, 0] %**, but the zero now comes from thermal death (95 °C is above
the whole Tmax bracket), not from a flag; S3's outer p-box is unchanged
(**[0, 1.9806] %**) but the hard 70 °C cliff is replaced by a **smooth
thermal-death slide** — the fan's upper edge runs 1.8392 % at 70 °C →
1.6081 → 1.2119 → 0.6665 → 0.4125 % at 87 °C through the Tmax bracket,
reaching zero only at 90 °C; S2's p-box narrows to **[0.0409, 2.0408] %**
and its worst-case corner **flips from cold to warm**: the slot's maximum
sat at the cold corner (constant k_m, network CO2 supply favored cold),
while the law's sits at an **interior maximum at 60 °C** (biology warming
toward Topt now outweighs the cooling network) — an extremum the
T-corner grid would have missed; the dense grid catches it and the
receipt prints the comparison ("dense grid EXCEEDS the T-corner
envelope").
Mechanism: the network's sole H₂-consuming pathway is the methanation
sink; the calcite dissolution/precipitation loop shifts brine chemistry
but is not itself an H₂ sink, which is why the k_m temperature treatment
dominates the fan (slot seams and law seams: §6.1).
(Scan honesty: all fan extrema are *labeled grid evidence* — the slot
path prints a corner-envelope consistency check, the law path a
dense-grid-vs-corner comparison, for every site.)

**Figure B — gate probability vs baselines**
(`figures/fig_b_pgate_vs_baselines.png`). Conventional composed
P(<6 EUR/kg): **≈ 3.6 % for all three sites** (3.635 % printed; valley
25 °C baseline 3.630 %; no-coupling baseline 20.765 %). The 20.765 →
3.630 % baseline-to-baseline drop is entirely the compressor factor
f_c = R entering CF_eff; the subsurface contributes no visible signal
at τ = 1 given the compressor p-box's width. Read all of these as
**≈ 3.6 % (per-gate SE ≈ 0.1 % at n = 20000)**: on the slot and law
paths the three sites print the same 3.635 % — per-site separation is
below the sampling floor, and S1's 0.005 pp delta against the 3.630 %
baseline is pure sampling noise. (The field-calibrated path's ≈ 0.3 pp
deltas below S1 are larger than that floor — they are systematic,
driven by the wider loss boxes, and decision-irrelevant; §4 field
calibration.)
The subsurface-only rows
(20.505–20.530 %) are within MC noise of the no-coupling baseline —
per-site separation at τ = 1 yr does not survive n = 20000 sampling
error, and we say so in the receipt. The corner p-box on beating the
gate is **[0, 100] % for every site**. The **law path changes none of
this**: composed 3.635 % for all three sites on both paths, subonly
rows identical within noise — the gate does not see the subsurface on
either k_m treatment.

**Figure C — composed-chain build-up** (`figures/fig_c_chain_waterfall.png`).
At interval mids: nominal 6.4160 EUR/kg → +subsurface **+0.0000–0.0005
EUR/kg** → +compressor → ~7.556 EUR/kg. The subsurface step is
sub-cent **on both k_m paths** (law: +0.00000/+0.00040/+0.00039 for
S1/S2/S3 — plotted as black diamonds); the compressor availability
(R mid = 0.506) is the cost driver.

**Field validation — the sourced law vs measured in-situ methanation**
(receipt section FIELD VALIDATION; sources in `site_screening_data.md`).
Only two field-scale datasets exist with **both** a measured reservoir
temperature and a measured methanation extent. **Lehen** (Underground
Sun Storage, RAG Austria 2016–2017 [16]): T = 40 °C measured; ~3 % of
the injected H₂ converted to CH₄ over 285 days (interval [3.0, 3.2] %:
the RAG-report figure via [18] and a stoichiometric bound from the
measured 960 m³ CO₂ consumption). **Lobodice** (Czech town-gas aquifer
1965–1991 [17, 19]): T = 25–45 °C seasonal; H₂ 54 → 37 % and CH₄
22 → 40 % over one 7-month season (extent interval [17.0, 31.5] % —
percentage-points vs fraction-of-injected-H₂ readings; a full
stoichiometric balance — 4 H₂ + CO₂ → CH₄ + 2 H₂O — over the same
source's measured initial/final compositions implies ≈ 50 % of the
injected H₂ consumed, so the demo's box is if anything LOW (data doc,
Lobodice row); isotope work
attributes part of the drop to caprock leakage, so the true microbial
extent is ≤ this, and we still bracket the full drop). For each site
the law network is re-run at the field temperature over the field
horizon and the predicted loss p-box is printed against the observed
extent. Verdicts: **NOT-BRACKETED at both sites** — the Bo-2021-
anchored magnitude interval's upper prediction (0.028 % and 0.031 %)
sits ≈ **107×** (Lehen) and ≈ **540×** (Lobodice) below the observed
lower edges. The **temperature shape, in contrast, is consistent**:
field methanation is measured exactly in the 25–45 °C window where the
CTMI f p-box is nonzero-but-suboptimal (f(40 °C) ∈ [0, 0.375]).
Reading: the shape survives field contact; the illustrative magnitude
anchor — calibrated on slow, largely abiotic field loss — is now
**empirically** flagged, not just labeled (§6.1). Nothing is
recalibrated: both validated paths stand, and the gap is the
quantified AWAITING-AUTHOR-DATA target. **Headline stress test**
(labeled what-if, same seeded chain): annualizing the observed extents
linearly — an upper bound, kinetic loss is concave in time — a
Lehen-class bloom gives 1-yr loss [3.84, 4.10] %, f_s(1) ∈
[0.959, 0.962], composed gate **3.055 %** (vs 3.635 %): the τ=1
headline **survives**; a Lobodice-class bloom gives 1-yr loss
[29.6, 54.8] %, f_s(1) ∈ [0.452, 0.704], composed gate **0.110 %**:
the gate **moves** — a Lobodice-class bloom would make the subsurface
visible at τ = 1. Mechanism note (read this before comparing numbers):
the stress test does **not** apply the linear f_s mapping to the
calibrated p-box — it **directly annualizes the observed 7-month
extents** (×365.25/210 — the 7-month season taken as 210 d) as a
labeled upper-bound what-if, bypassing both the
30-yr integration and the mapping. The screening gates above and the
stress test therefore answer different questions: "what does the
calibrated model say at τ = 1 under the receipt's mapping" vs "what if
an observed Lobodice-class season happens at the candidate". There is
no contradiction between 3.635 % and 0.110 % — and there is also no
comfort in it: the mapping-based gate is exactly the quantity the
stress test probes. Scoping: Lobodice is a shallow (500 m), near-fresh
(0.03 M), pH 6.7 aquifer storing 54 % H₂ town gas — Thaysen calls its
conditions "highly favorable" and its 17 % "exceptional"; S1 is
measured 95 °C (sterile-hot on both paths) and S2/S3 are deeper,
warmer, saline, where active blooms can be absent entirely (§6.1(d)).
The τ=1 rounding-term conclusion is therefore **conditional on no
Lobodice-class bloom** — and that condition is biological site
characterization, the identified target. Context rows without verdicts
(no sourced T, or outside the network's scope): Beynes (no detected
H₂ consumption [20]), Ketzin town-gas era (61 % H₂ volume lost [18]),
the salt-cavern pure-H₂ stores (no reactivity loss [18]), and the Olla
CO₂-EOR (a different process [11]).

**Field calibration — the replacement magnitude anchor** (receipt
section [A4]; method and quotes in `site_screening_data.md` §C1–C6).
The falsified magnitude is rebuilt from the field evidence, three ways,
each labeled by class. (i) **FIELD-DERIVED (inverse)**: bisecting the
network itself (bracket [1e-12, 100], 80 bisection steps per corner —
residual bracket width ≈ 100·2⁻⁸⁰ ≈ 1e-22, machine-level; the
practical error floor is the RK4 discretization, pinned independently
in the selftest; no closed form, no
hidden constants) until it reproduces each observed extent, enveloped
over the extent × horizon × A × salt corners — LEHEN k_eff ≈
**[0.77, 0.89]** (40 °C), LOBODICE k_eff ≈
**[6.3, 15]** (25–45 °C envelope; leakage-caveated). (Calibrated
values are quoted to 2 significant figures in this brief; full
precision lives in `site_screening_data.md` and the receipt.)
(ii) **IN-SITU-MEASURED**: Tyne et al. 2021 [11] measured "an in situ
microbial methanogenesis rate from within a natural system of 73–109
millimoles of CH₄ per cubic metre (standard temperature and pressure)
per year" at 29.2–50.7 °C in the Olla CO₂-EOR field — the only such
measurement in the literature, and the authors call it a conservative
minimum. Bridged to the model's units (4 H₂ : 1 CH₄; rate
ks[0]·[H₂]·[CO₂]; [H₂] = the screening charge; "per cubic metre" read
as per m³ water — the paper never defines the normalization volume, a
documented ~1–2-order bridge ambiguity that EXCEEDS the width of the
resulting box; the box is conditional on this reading): k_eff ≈
**[0.50, 1.1]**. (iii) **Overlap**: LEHEN ∩ TYNE =
**[0.765606, 0.894709], NONEMPTY** (exact edges) — two independent
in-situ evidences
are mutually consistent at ~40 °C (weak corroboration only; it does
not explain Lobodice, which sits ~6× above Tyne's upper edge,
consistent with its leakage + bloom-condition caveats). The
**field-calibrated k_m p-box at Topt** is then ≈ **[2.0, 380]**: LO = Lehen k_lo ÷ f_hi(40 °C) (caveat-free site,
biology at its p-box-best shape — the weakest magnitude Lehen allows);
HI = Lobodice k_hi ÷ f_lo(45 °C) (caveated site, biology at its
p-box-worst shape at the warmest reported T — a **labeled** edge, not
a strict bound: arbitrarily small f at cooler T would push k_m
higher); Tyne's minimal k_m = 0.659188 ≤ LO — consistent. This is
**109×–20451× the LAB-FALSIFIED Bo-2021 anchor** — replaced, not
adjusted. The interpretive layer (sourced, not a fudge factor): the
gap is a volumetric-biomass effect — Tyne itself quotes lab microcosm
rates of 0.01–0.15 vs its in-situ 73–109 mmol m⁻³ yr⁻¹ (Gray et al.
2009 [21]); Thaysen 2021 documents 0.7×–7-orders bulk-vs-lab spreads
with near-well rates up to 4533 nM/h; Tremosa had to rescale lab Monod
k_max by ~1/50 to fit Lobodice ("it could also be that not kmax, but
the concentration of bacteria is lower — or both"); Haddad 2022
watched ~40 % of injected H₂ go in <90 days at 47 °C, *accelerating*
after day 52. The calibrated k_m is therefore an **effective bulk
constant absorbing local attached-biomass density** — anchored on
active-bloom sites, which is the conservative screening direction.
Pushed through the same CTMI machinery (same seeded chain, n = 20000),
the **FIELD-CALIBRATED LAW PATH** prints beside both older paths:
30-yr loss p-boxes S1 **[0, 0]** (thermal death unchanged), S2
**[15.475235, 100.000000]**, S3 **[0, 100.000000]** — the 100 %
endpoints are **model-bounds of the constant-pH₂ batch sink, not
physical predictions**: the network holds pH₂ = 15 atm and brine
composition fixed, so nothing throttles the sink as H₂ would actually
deplete in a closed reservoir; read "100 %" as "beyond this model's
ability to bound" (§6.1(h)); f_s(1) ≥
[0.966667, …]; composed gate **3.635 / 3.355 / 3.370 %** for S1/S2/S3
(vs 3.635 % on both older paths — the ≈ 0.3 pp deltas are systematic,
not MC noise, but decision-irrelevant; per-gate SE ≈ 0.1 %) — and
f_s(10) at S2 drops to
**[0.666667, 0.948416]**. The τ = 1 headline **survives the
re-anchoring under the receipt's linear f_s mapping** — but that
mapping is exactly what the stress test showed to be optimistic for an
active bloom (annualized reading: 0.110 %), so the conditional claim
is kept and sharpened (§5). **Figure D** (`figures/fig_d_field_calibrated.png`)
plots the field-calibrated band against the sourced-law band per site:
S1's thermal death is anchor-independent ([0, 0] on every path), while
S2/S3 widen to the KMF magnitude — the visual statement of what the
re-anchoring costs at long residence.

## 5. Where site choice starts to matter

The τ = 10 yr analytic sensitivity: f_s intervals drop to
S2 [0.9924, 0.9997] and S3 [0.9934, 1.0000] (S1 stays [1, 1]) on the
slot path; the law path is nearly identical — S2 [0.9932, 0.9999],
S3 [0.9934, 1.0000], S1 [1, 1] — because the outer p-boxes barely move
at the site level. The **field-calibrated law path moves this**: with
the KMF magnitude the S2 τ = 10 availability falls to
**[0.666667, 0.948416]** (S3 [0.666667, 1.0000]) — multi-year residence
is exactly where the re-anchored kinetics bite, while S1 stays [1, 1]
(thermal death is anchor-independent). Interpretability caveat (same
one as §4, restated where the numbers are): these τ = 10 intervals
combine the un-throttled constant-pH₂ batch sink with the linear f_s
mapping — both known-simplification machinery (§3, §6.1) — so they are
**model-bound screening intervals, not reservoir predictions**; the
identified upgrade is direct kinetic integration to the target τ
(§6.1(i)). A seasonal-plus storage mandate
(multi-year
residence, or a strategic reserve) is where the warmer onshore
aquifers' kinetic losses become visible against South Kavala's
measured-hot, interaction-free regime (slot: a cutoff; law: thermal
death — same screening conclusion either way). If the valley's storage
residence is closer to τ = 1 (our
ILLUSTRATIVE default), the subsurface is a rounding term at every
candidate and the characterization euros belong to the compressor
alloys and the heat/dispensing contracts — same conclusion as the
valley-chain receipt, now backed by real site temperatures **and** a
sourced methanation temperature law — **with one condition the field
validation makes precise, and the field calibration now stress-tests
directly.** For the stated k_m interval the claim is
closed: the dominant subsurface uncertainty is parameterized, so it
cannot be overturned by an uncharacterized k_m *shape*. The magnitude
interval is no longer a caveat but a calibrated p-box — and pushing
the field-calibrated box through the composed chain moves the τ = 1
gate only to 3.355 % / 3.370 % (from 3.635 % — a systematic ≈ 0.3 pp
delta, larger than the per-gate SE ≈ 0.1 % but decision-irrelevant at
this magnitude), because the receipt's
linear f_s mapping is shallow at τ = 1 (f_s ≥ 0.9667 even at the
caveated HI edge). **But that survival is mapping-limited, not
kinetics-limited**: the same kinetics read through the
concavity-honest annualized lens (the stress test) collapse the gate
to 0.110 % under a Lobodice-class bloom — and the KMF HI edge is
harsher still at warm corners (30-yr loss saturates at 100 %). The
honest statement is therefore unchanged and now anchored: *at τ = 1
the subsurface is a rounding term unless the reservoir hosts a
Lobodice-class active methanogen bloom* — and ruling that in or out is
biological site
characterization, the top subsurface characterization target.

## 6. Honest limitations

1. **k_m: sourced law now, with labeled seams.** The 2026-08-01
   revision adds the sourced LAW PATH (§3): a Rosso CTMI temperature
   shape with a cardinal p-box from primary literature (Zeikus & Wolfe
   1972's ≤ 5 °C pure-culture cutoff; the 80–90 °C field biosphere
   cutoff). The old slot is kept and printed alongside — the comparison
   is in the receipt, not hidden. Remaining honest seams: (a) the k_m
   **magnitude** now exists in two printed forms — the Bo-2021 anchor
   (LAB-FALSIFIED: it under-brackets both measured in-situ extents by
   ~10²–10³×, and an independent 2026-08-01 re-check found the named
   paper abiotic, so its provenance as a *microbial* anchor is doubly
   weak — data file §C6) and the FIELD-CALIBRATED box ≈ [2.0, 380]
   that replaces it (§4; full precision in the data file). The replacement's own seams: the
   HI edge is a *labeled* edge, not a strict bound (the field evidence
   bounds k_m only one-sidedly at each site, because the CTMI shape
   allows f → 0 at field temperatures); the Tyne bridge carries a
   ~1–2-order normalization-volume ambiguity the paper never resolves;
   and the box is an *effective* bulk constant that absorbs local
   attached-biomass density — calibrated on active-bloom sites, the
   conservative screening direction, not a universal reservoir rate;
   (b) the
   CTMI is a microbial *growth* model applied to the pseudo-sink *rate* — a
   labeled form transfer (hydrogenotrophic methanogenesis is the
   biological H₂ sink; the network itself stays abiotic-batch); (c)
   Ghaedi 2025's own kinetic law remains closed-access — note their
   ~70 °C threshold is an *equilibrium-calibration* result, not
   kinetics; (d) microbial H₂ consumption can be **zero** even when
   thermodynamics smiles (Berta 2018: sulfate reducers outcompete at
   2–15 bar H₂; salinity > 35 g/L shuts consumption down) — our p-box
   spans active-methanogen conditions, which is the conservative
   screening direction. The apparent salinity paradox is likewise a
   stated conservative choice, not a contradiction: the [0.70, 1.00]
   salting-out interval enters only *physical* H₂ solubility, while the
   k_m sink deliberately models an *active* methanogen community at the
   (unmeasured, Hystories-default) site salinity; (e) Thaysen 2021's compiled rates carry a 2023
   corrigendum and no T correction — used as context only; (f) the field
   validation's NOT-BRACKETED verdicts inherit the network's screening
   assumptions (batch reactor, fixed pH₂ = 15 atm) and the observations'
   own caveats (Lobodice's caprock-leakage attribution; Lehen's
   unpartitioned 15.7 % unrecovered H₂) — the verdict scopes to the
   *magnitude interval*, stated as such; it is not a refutation of the
   CTMI shape or of the network. The stress test (§4) computes the
   headline consequence: the τ=1 rounding-term conclusion survives a
   Lehen-class bloom but not a Lobodice-class one, so it is a
   conditional claim — printed as such; (g) the field-calibrated
   gates inherit the receipt's linear f_s mapping (f_s = 1 −
   L30·τ/30), which is optimistic at short horizons when the loss box
   is wide — the annualized reading of the same kinetics (stress
   test) is the honest short-horizon lens and is why the conditional
   framing survives the re-anchoring unchanged; (h) the 100 % loss
   endpoints of the field-calibrated p-boxes are **model-bounds of the
   constant-pH₂ batch sink, not physical predictions** — the network
   holds pH₂ = 15 atm and brine composition fixed, so nothing throttles
   the sink as a real closed reservoir would; "100 %" means "beyond
   this model's ability to bound", and the derived f_s(10) intervals
   inherit that reading; (i) the identified **methodological upgrade**
   is direct kinetic integration of the network to the target residence
   time (τ = 1, τ = 10), bypassing the linear f_s mapping entirely —
   the mapping-based gates and the direct-annualization stress test
   answer different questions (§4 mechanism note), and until that
   integration replaces the mapping, every short-horizon field-
   calibrated number is read through both lenses.
   AWAITING-AUTHOR-DATA: drop the paper's law into the 2-line `km_at`
   slot (or recalibrate the k_m magnitude) and every number re-derives.
2. **Two of three temperatures are gradient-derived**, not measured —
   labeled union brackets (Hystories 30 ± 3 °C/km model default ∪
   Strymon-basin measured 25–36 °C/km). The union bracket is a
   deliberate maximum-bounds construction, not a prediction. No
   measured Mesohellenic gradient exists in the public record. The
   third temperature — S1's 95 °C — is **measured but tender-sourced**:
   it comes from the HRADF 2020 invitation for expression of interest
   (a tender document, not a primary reservoir-engineering report), and
   we do not know whether it is a corrected bottom-hole or an
   undisturbed formation temperature. The [0, 0] % claim survives a few
   degrees of downward revision — the Tₘₐₓ p-box's upper edge is
   90 °C, so S1 stays interaction-free down to 90 °C — **but if the
   true formation temperature is < 90 °C, S1 enters the thermal-death
   slide window and the hard zero must be re-read through the law fan,
   not quoted as [0, 0]**. Hedged, not hidden.
3. **No measured salinity for any site** (Hystories 100 g/L default is
   a screening assumption); salinity enters the network only through
   the illustrative salting-out interval.
4. **Batch network, fixed pH 7, pH₂ = 15 atm, fixed brine** — a
   screening reactor, not a reservoir model; perfect mixing and the
   fixed pH₂ = 15 atm are conservative *upper* bounds on H₂
   availability to the sink. Depth/porosity/
   permeability (the capacity and injectivity drivers) are out of its
   scope and reported as context only.
5. **MC noise**: per-gate sampling error is SE ≈ 0.1 % at n = 20000.
   On the slot and law paths the three sites print the same composed
   gate (3.635 %) — per-site separation is below that floor. The
   field-calibrated path's ≈ 0.3 pp deltas below S1 are **systematic**
   (driven by the wider loss boxes; same-seed difference 0.31 ± 0.04 pp
   across a seed scan) — stated as such, and decision-irrelevant: only
   p-box structure and corner results
   should be read quantitatively.
6. TRIERES itself is Corinth-anchored with above-ground storage scope
   (§2) — this brief screens Greek formations, not TRIERES deliverables.

## 7. Invitation — swap in your data

Every labeled slot is a 2-line edit: measured site T / salinity /
brine chemistry into `site_screening_data.md`'s AWAITING-DATA rows and
the per-site constants in `site_screening.sio`; the group's own
kinetic law into `km_at`; the valley's true residence time into `tau`.
Deterministic seeds mean the receipt, the figures, and this brief's
numbers regenerate byte-identically.

## 8. Reproducibility

```
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH SOUNIO_SOUC_ENGINE=lean_single \
    bin/souc run demos/hydrogen/site_screening.sio          # ~10–13 min
.venv/bin/python demos/hydrogen/tools/render_site_figures.py \
    <demo_stdout> demos/hydrogen/figures                    # seconds
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH SOUNIO_SOUC_ENGINE=lean_single \
    bin/souc run tests/run-pass/site_screening_selftest.sio # ~30 s
```

## References

1. HRADF, *UGS South Kavala — Invitation for EOI*, Annex B (2020).
2. Koukouzas, Ritss et al. 2021, *Energies* 14:3321 (DOI 10.3390/en14113321).
3. Hystories D1.4 (2023), GA 101007176, §12 Greece.
4. HyUSPRe D1.3 (2022), GA 101006632, p. 31.
5. Dotsika et al. 2021, *Sci. Rep.* 11:16291 (DOI 10.1038/s41598-021-95656-6).
6. Ghaedi, Gholami, Bellas & Stamatakis 2025, *GHG: Sci. Technol.*
   15(6):757–768 (DOI 10.1002/ghg.2368) — closed access; abstract-level use only.
7. Arvanitis et al. 2020, *Energies* 13:2707 (DOI 10.3390/en13112707).
8. CORDIS TRIERES GA 101112056 project page (accessed 2026-07-31).
9. Rosso, Lobry & Flandrois 1993, *J. Theor. Biol.* 162:447–463
   (DOI 10.1006/jtbi.1993.1099) — CTMI cardinal-temperature model form.
10. Zeikus & Wolfe 1972, *J. Bacteriol.* 109(2):707–713
    (DOI 10.1128/jb.109.2.707-713.1972) — methanogen cardinal
    temperatures 40 / 65–70 / 75 °C; "nothing occurred above 75 C".
11. Tyne et al. 2021, *Nature* 600:670–674 (DOI 10.1038/s41586-021-04153-3)
    — in-situ reservoir methanogenesis at 29.2–50.7 °C.
12. Head, Gray & Larter 2014, *Front. Microbiol.* 5:566
    (DOI 10.3389/fmicb.2014.00566) — 80–90 °C field biosphere cutoff.
13. Wilhelms et al. 2001, *Nature* 411:1034–1037 (DOI 10.1038/35082535)
    — palaeopasteurization primary.
14. Bo, Zeng & Chen 2021, *Int. J. Hydrogen Energy* 46(38):19998–20009
    (DOI 10.1016/j.ijhydene.2021.03.116) — 30-yr field-loss k_m anchor.
15. Berta et al. 2018, *Environ. Sci. Technol.* 52:4937–4949
    (DOI 10.1021/acs.est.7b05467) — negative control (no methanogenesis
    at 2–15 bar H₂).
16. Hellerschmied et al. 2024, *Nature Energy* 9:333–344
    (DOI 10.1038/s41560-024-01458-1) — Lehen field-trial primary:
    T = 40 °C, 84.3 % H₂ recovery, 960 m³ CO₂ consumed.
17. Šmigáň et al. 1990, *FEMS Microbiol. Lett.* 73:221–224
    (DOI 10.1016/0378-1097(90)90733-7) — Lobodice primary (paywalled;
    numbers via [18] and [20]).
18. Tremosa, Jakobsen & Le Gallo 2023, *Front. Energy Res.* 11:1145978
    (DOI 10.3389/fenrg.2023.1145978) — Lobodice verbatim composition
    table; Ketzin, Beynes, salt-cavern context.
19. Buzek et al. 1994, *Fuel* 73:747–752
    (DOI 10.1016/0016-2361(94)90019-1) — Lobodice isotope attribution
    (part of the H₂ loss is caprock leakage).
20. Heinemann et al. 2021, *Energy Environ. Sci.* 14:853–864
    (DOI 10.1039/d0ee03536j) — "no detected hydrogen consumption in
    Beynes ... up to a 17% decrease ... in Lobodice over a seven month
    cycle".
21. Gray et al. 2009, *Extremophiles* 13:511–519
    (DOI 10.1007/s00792-009-0237-3) — laboratory microcosm methanogenesis
    rates 0.01–0.15 mmol m⁻³ yr⁻¹, quoted via [11].
