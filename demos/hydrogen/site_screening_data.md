# Sourced geological parameters — candidate Greek H₂-storage formations

Data basis for `demos/hydrogen/site_screening.sio` and
`demos/hydrogen/SITE_SCREENING_BRIEF.md`. Every value is either cited to a
public source actually accessed (2026-07-31) or labeled as what it is
(model default, union bracket, AWAITING-AUTHOR-DATA). Nothing here is
fabricated; the NOT-FOUND list at the bottom is part of the deliverable.

## Geography correction (read first)

TRIERES (GA 101112056) is **anchored around the Motor Oil Corinth Refinery
complex** (Ag. Theodoroi), not Western Macedonia — CORDIS project page and
the Clean Hydrogen Partnership presentation `TRIERES-2025.pdf` agree, and
CORDIS shows no public geological deliverables (storage scope is
above-ground). The formations screened below are the Greek UHS candidates
documented in the public literature and EU project databases; the
valley-chain economics they feed are the TRIERES chain of
`valley_chain_epistemic.sio`. Western Macedonia / Mesohellenic Trough is
covered because it is the onshore region with published storage-relevant
characterization (Hystories D1.4 §12, research by CERTH, data copyright
IGME/CERTH).

---

## Site S1 — South Kavala depleted gas field (offshore N. Aegean)

Type: **depleted gas field** — turbiditic sands encased in Miocene
anhydrite/salt; Greece's concessioned underground gas storage (UGS) site,
publicly reported as an H₂-storage conversion candidate.

| Parameter | Value | Source (accessed 2026-07-31) |
|---|---|---|
| Location / water depth | ~6 km off west Thassos, 52 m water | HRADF, *Invitation for Expression of Interest, UGS South Kavala*, Annex B (2020), pp. 36–40: "located in the south western part of the Prinos-Kavala sub-basin, in 52 meters of water depth" |
| Depth to top reservoir | below 1630 m subsea | HRADF Annex B: "below 1630m (5363 ft) meters at SK-1" |
| Gas–water contact | 1723 m below sea level | HRADF Annex B: "The original gas-water contact was reported at 1723m (5654 feet) below sea level" |
| **Reservoir temperature** | **95 °C (measured, tender-sourced)** | HRADF Annex B: "Reservoir Temperature: 95 °C". PROVENANCE CAVEAT (added 2026-08-01, reviewer finding): the source is an invitation-for-EOI tender document, not a primary reservoir-engineering report; whether 95 °C is a corrected bottom-hole or an undisturbed formation temperature is not stated. The demo's [0, 0] % loss survives downward revision to 90 °C (the Tₘₐₓ p-box upper edge); below 90 °C S1 enters the thermal-death slide window and the hard zero must be re-read through the law fan. |
| Porosity | max 20 % (1995); avg 22 % (Energean) | HRADF Annex B: "maximum porosity 20%… (Georgakopoulos, et al., 1995)"; "Average porosity: 22%" |
| Permeability | 50 mD (1995) → 100 mD (Energean 2011) | HRADF Annex B: "average permeability of 50mD"; "Average permeability: 100 md" |
| Pressure | 182 bar initial; 27 bar after depletion | HRADF Annex B: "Initial reservoir pressure: 182 bars / Reservoir pressure after 850 million m3: 27 bars" |
| Gas quality | 83 % CH₄, no H₂S | HRADF Annex B: "characterized by the complete absence of H2S" |
| Net pay / closure | ~11–20 m net; ~4.5 km² closure, ~90 m vertical | Proedrou 2001, BGSG 34(3):1221–1228, DOI 10.12681/bgsg.17198 ("net pay thickness… up to eleven meters"); HRADF Annex B |
| H₂ storage potential | 2 TWh (WGC-90) | HyUSPRe D1.3 (2022), p. 31: "Greece has planned a small storage site, South Kavala, 4 TWh… The hydrogen storage potential is 2 TWh (WGC-90)" |
| Salinity | **NOT PUBLISHED** (water saturation 33 % is saturation, not salinity) | — |

## Site S2 — Pentalofos Formation, Mesohellenic Trough (Grevena sub-basin, W. Macedonia)

Type: **porous turbiditic sandstone saline aquifer** (with conglomerates/
shales), Tsarnos & Kalloni members, capped by the Tsotyli Fm
("effective cap rock", Hystories D1.4).

| Parameter | Value | Source |
|---|---|---|
| Average depth | 1500 m | Koukouzas et al. 2021, *Energies* 14:3321, DOI 10.3390/en14113321, Table 6 |
| Deepest storage point | −2544 m (base of Tsarnos member) | same paper; Tasianas & Koukouzas 2016, *Energy Procedia* 86:334–341, DOI 10.1016/j.egypro.2016.01.034 |
| Porosity | 7–25 % range; ftot = 0.15 used (P90-conservative) | Koukouzas et al. 2021: "with a porosity ranging from 7% to 25%" |
| Gross thickness | avg 2500 m, max 4000 m | Koukouzas et al. 2021; Hystories D1.4 §12 |
| Caprock | Tsotyli Fm, 1500–2000 m | Hystories D1.4 §12: "characterised as an effective cap rock for trapping buoyant fluids" |
| **Temperature** | **not measured — gradient-derived bracket [52.5, 69.0] °C at 1500 m** | union of: Hystories D1.4 model default 30 °C/km ±10 % (→ 55.5–64.5 °C, surface 15 °C); Strymon-basin measured gradient 25–36 °C/km (Dotsika et al. 2021, *Sci. Rep.* 11:16291, DOI 10.1038/s41598-021-95656-6 → 52.5–69.0 °C). The paper's own T estimate cites Jongsma 1974 Aegean heat flow without printing a number. LABELED UNION BRACKET, not a measurement. |
| Salinity | **NOT PUBLISHED** — Hystories screening default 100 g/L | Hystories D1.4 p. 24: "Salinity (assume 100 g/L if not provided)". MODEL DEFAULT, labeled. |
| Permeability | **NOT PUBLISHED** — Hystories default 100 mD | same: "Permeability (assume 100 mD if not provided)" |

## Site S3 — Eptachori Formation, Mesohellenic Trough

Type: **conglomerates/sandstones with marine turbiditic shale top** —
deeper saline-aquifer candidate in the same trough.

| Parameter | Value | Source |
|---|---|---|
| Average depth | 2000 m | Koukouzas et al. 2021, Table 6 |
| Porosity | ~12 % (ftot = 0.12; Hystories range 0.07–0.18) | Koukouzas et al. 2021: "Its porosity ranges around 12%" |
| Thickness / dip | ~1100 m, dipping 60–70° E | same: "a thickness of about 1100 m with a dipping 60–70° to the east" |
| **Temperature** | **not measured — gradient-derived bracket [65, 87] °C at 2000 m** | same construction as S2 (union of Hystories 30 ± 3 °C/km → 69–81 °C and Strymon 25–36 °C/km → 65–87 °C). LABELED UNION BRACKET. Straddles the 70 °C interaction cutoff of the kinetic network (abstract-level A2) — this is a *result-relevant* epistemic fact, see the demo. |
| Salinity / permeability | **NOT PUBLISHED** — same labeled defaults as S2 | Hystories D1.4 p. 24 |

---

## Context formations (not screened, documented for completeness)

- **Prinos oil field / basin** (offshore Kavala): oil trapped 2490–2790 m
  TVDSS (Energies 2023, 16:2392, DOI 10.3390/en16052392); Miocene
  sandstones ~1600 m; saline aquifers ~2400 m; **sour gas up to 60 % H₂S**
  (HRADF Annex B) — a purity red flag any UHS concept must price in.
  Nestos sub-basin geothermal gradient "very high" (Proedrou &
  Papaconstantinou 2004, BGSG 36(1), DOI 10.12681/bgsg.16675).
- **External Hellenides Triassic evaporites / diapirs** (Ionian zone,
  W/SW Greece — Trifos, Achira, Kefalonia, Corfu): salt-cavern candidates
  per Arvanitis et al. 2020 (*Energies* 13:2707, DOI 10.3390/en13112707);
  capsule-cavern scenario at 650 m with a generic 25 °C/km gradient.
  NOT in Western Macedonia — no halite/diapir is documented there in the
  accessed literature.
- **Epanomi gas field** (W. Thessaloniki): depleted, 2600 m limestones,
  500 Mm³ total / 250 Mm³ working gas (Energies 2020, Table 4).

## The k_m(T) methanation law — literature verdict (added 2026-08-01)

The demo's original k_m was a pseudo-sink **slot** (constant below a hard
70 °C step, zero above — an abstract-level encoding of Ghaedi 2025's A2).
A dedicated literature hunt (scite, Crossref, WebSearch, full texts;
every DOI below Crossref-verified) asked: does the literature support a
sourced temperature-dependent methanation rate law with p-box parameters?

**Verdict: YES for the temperature SHAPE, with the magnitude interval
kept as the labeled Bo-2021 anchor.** Implemented in
`site_screening.sio` as the LAW PATH alongside the untouched slot:

k_m_eff(T) = k_m × f(T), where f is the **Rosso 1993
cardinal-temperature (CTMI)** microbial-growth model (normalized
f(Topt) = 1, f = 0 outside [Tmin, Tmax]):

| Parameter | Bracket | Source |
|---|---|---|
| model form | CTMI | Rosso, Lobry & Flandrois 1993, *J. Theor. Biol.* 162:447–463, DOI 10.1006/jtbi.1993.1099 |
| Tmin | [25, 40] °C | 40: Zeikus & Wolfe 1972 (*M. thermautotrophicus* "minimum about 40 C"), *J. Bacteriol.* 109(2):707–713, DOI 10.1128/jb.109.2.707-713.1972. 25: **LABELED community floor** — in-situ reservoir methanogenesis measured at 29.2–50.7 °C (Tyne et al. 2021, *Nature* 600:670–674, DOI 10.1038/s41586-021-04153-3), so the community Tmin < 29; 25 is a round labeled floor, not a measurement |
| Topt | [65, 70] °C | Zeikus & Wolfe 1972, abstract verbatim: "The optimal temperature for growth was 65 to 70 C" |
| Tmax | [75, 90] °C | 75: Zeikus & Wolfe 1972: "nothing occurred above 75 C" — pure-culture transition ≤ 5 °C wide, the sharpest sourced cutoff found. 90: field-scale biosphere cutoff 80–90 °C (palaeopasteurization) — Head, Gray & Larter 2014, *Front. Microbiol.* 5:566, DOI 10.3389/fmicb.2014.00566 ("no reports of methanogenic oil degradation at such high temperatures"); Wilhelms et al. 2001, *Nature* 411:1034–1037, DOI 10.1038/35082535 |
| k_m magnitude | [0.0048, 0.0187] **unchanged** | Bo et al. 2021 30-yr field-loss anchor (0.72 % Tubridgi / 2.76 % Mondarra, DOI 10.1016/j.ijhydene.2021.03.116) — ILLUSTRATIVE as before, now read as the value AT Topt |

**Physics watch (the S3 cliff):** a sharp high-temperature cutoff is
PHYSICAL, not a slot artifact — but it is a smooth slide, not a step:
pure-culture growth goes from full rate at 70 °C to zero above 75 °C
(≤ 5 °C, Zeikus & Wolfe 1972); the field-scale biosphere cutoff sits at
80–90 °C (Head 2014 / Wilhelms 2001). The law therefore replaces the
hard 70 °C step with a steep smooth transition to zero inside [75, 90] °C.

**What Ghaedi 2025 actually says (abstract-level, verified):** their
~70 °C negligibility threshold is an **equilibrium-calibration** result —
"a new temperature-dependent analytical expression for the equilibrium
constant ... calibrated against the experimental observations" — NOT a
kinetic rate law. Their kinetic methanation rate law exists only in the
paywalled body (searched: no preprint/repository copy exists; OpenAlex,
Unpaywall, Semantic Scholar, EarthArXiv, SSRN all negative). The LAW
PATH here is independently sourced content, not the paper's.

**Supporting kinetic anchors (not needed by the CTMI form, documented):**
hydrogenotrophic Ks(H₂) 2.5–13 µM across four methanogens (Robinson &
Tiedje 1984, DOI 10.1007/bf00425803); µmax = 0.69 h⁻¹, Ks(H₂) ≈ 20 % gas
phase at 65 °C for *M. thermautotrophicum* (Schönheit, Moll & Thauer
1980, DOI 10.1007/bf00414356); mesocosm hydrogenotrophic rate 0.26 mmol
L⁻¹ h⁻¹ at 40 °C/40 bar in reservoir sandstone+brine (Hellerschmied et
al. 2024, *Nat. Energy*, DOI 10.1038/s41560-024-01458-1); in-situ 73–109
mmol CH₄ m⁻³ yr⁻¹ at 29–51 °C (Tyne 2021); Arrhenius Ea 40–75 kJ/mol
species / ~110 kJ/mol community (Price & Sowers 2004, DOI
10.1073/pnas.0400522101); H₂ threshold 6.5–12 Pa (Lovley 1985, DOI
10.1128/aem.49.6.1530-1531.1985).

**Negative controls / caveats:** no methanogenesis at all at 2–15 bar H₂
in groundwater columns — sulfate reducers outcompete (Berta et al. 2018,
DOI 10.1021/acs.est.7b05467); salinity > 35 g/L shuts H₂ consumption
down. The UHS Monod-modeling school (Hagemann et al. 2016, DOI
10.1007/s10596-015-9515-6; Eddaoui et al. 2026, DOI 10.2516/stet/2026026)
publishes only dimensionless parameters "estimated with a very high
uncertainty" — unusable as sourced numbers. Thaysen et al. 2021 (DOI
10.1016/j.rser.2021.111481) compiles per-cell consumption rates but
applies NO temperature correction and has a 2023 corrigendum (DOI
10.1016/j.rser.2022.113039, closed) touching its estimate equations —
treated as context, not as a parameter source.

## FIELD VALIDATION of the k_m(T) law — measured in-situ methanation (added 2026-08-01)

The sourced law is checked against the only field-scale underground gas
storage observations with a MEASURED reservoir temperature AND a measured
methanation extent. Primaries were hunted down; where paywalled, numbers
come from multiple independent open reviews that agree (stated per number).

### Field site F1 — Lehen (Underground Sun Storage, RAG Austria, 2016–2017)

Primary: Hellerschmied et al. 2024 (source 18/31; OA, full text quoted).

- **T = 40 °C MEASURED**: "The temperature in the reservoir is 40 °C."
  Reservoir: 2-m sandstone, Hall formation, Upper Austrian Molasse;
  1,027 m TVD; initial 107 bar, trial up to 78 bar; brine ~40,000 m³.
- **Gas**: co-storage of natural gas + "H2 (9.9% (v/v))"; 119,353 m³ H₂
  injected over 96 d, 112 d shut-in, 76 d withdrawal — 285 d total.
- **MEASURED H₂ fate**: "We successfully recovered 84.3% of the injected
  H2" — 15.7 % unrecovered, NOT partitioned (cushion-gas migration ~40 %
  of the unaccounted, decline-curve estimate; dissolution ≤ 3 % of
  injected; microbial consumption the remainder).
- **MEASURED methanation evidence**: "960 m3 of CO2 was consumed
  throughout the field trial" with "an 87.4% decrease in CO2" during
  shut-in; δ¹³C_CH4 shift −3.8 ± 0.4 ‰ during shut-in "supports the
  hypothesis of biological geo-methanation"; methanogenic genera 7.3 →
  17.0 % relative abundance (61 % of the active community at production;
  mostly *Methanobacteriaceae*, strictly hydrogenotrophic).
- **Observed extent used in the demo: [3.0, 3.2] % of injected H₂** —
  3.0 = "only 3% of the hydrogen converted into methane" (source 28,
  citing the RAG 2017 project report); 3.2 = DERIVED stoichiometric
  bound: 960 m³ CO₂ × 4 H₂/CO₂ ÷ 119,353 m³ H₂.
- Lab mesocosm anchor (reservoir cores+brine, 40 °C, 40 bar): "95% of
  the CH4 production was completed within 4.9 days", turnover rate
  0.008 h⁻¹ — LAB, not field.

### Field site F2 — Lobodice town-gas aquifer (Czech Republic, 1965–1991)

Primaries: sources 26/27 — BOTH PAYWALLED; numbers below are verbatim
from four independent OA reviews that agree (sources 24, 28, 29, 30).

- **T = 25–45 °C seasonal MEASURED** (Thaysen tabulates 20–45 °C):
  "town gas containing 54% H2 was stored in a sandstone reservoir at a
  depth of 500 m (pressure of 4 MPa and temperature of 25°C–45°C)"
  (source 28). Very low salinity 0.03 M, pH 6.7 (source 24).
- **MEASURED composition shift over one 7-month season**: "The stored
  gas initially composed of 54% H2, 22% CH4, 12% CO2, 9% CO and 2.5% N2
  evolved after being stored during 7 months to 40% CH4, 37% H2, 9% CO2,
  9% N2 and 3% CO", plus "10%–20% of the gas volume was lost" (28).
- **Observed extent used in the demo: [17.0, 31.5] %** — 17.0 = "A H2
  consumption of 17 % by methanogens at the Lobodice town gas storage
  site over a time span of seven months" (24; 29: "up to a 17% decrease
  in hydrogen ... over a seven month cycle"); 31.5 = DERIVED: the same
  17 percentage points as a fraction of the injected H₂ (17/54).
- **STOICHIOMETRIC TRANSLATION (added 2026-08-01, reviewer finding)** —
  4 H₂ + CO₂ → CH₄ + 2 H₂O over the same source's measured initial
  (54 % H₂, 22 % CH₄, 12 % CO₂) and final (37 % H₂, 40 % CH₄, 9 % CO₂)
  compositions. Per 100 initial mol, let x = mol H₂ consumed; CO₂
  falls by x/4, CH₄ rises by x/4, total gas falls to 100 − x. Then
  (54 − x)/(100 − x) = 0.37 → x ≈ 27.0 mol, i.e. **≈ 50 % of the
  injected H₂ consumed** (cross-checks: CH₄ (22 + x/4)/(100 − x) ≈
  39 % vs measured 40 %; CO₂ (12 − x/4)/(100 − x) ≈ 7 % vs measured
  9 %). The demo's [17.0, 31.5] % box therefore UNDERSTATES the
  stoichiometric extent — the inverse-calibrated Lobodice k_eff box is,
  if anything, LOW; the caprock-leakage caveat (next bullet) cuts the
  other way, and both are printed.
- **CAVEAT (isotope attribution)**: "Buzek et al. (1994) showed that
  some hydrogen losses are also linked to cap-rock heterogeneities"
  (28) — the true microbial extent is ≤ the observed drop; the demo
  brackets the FULL observed drop (conservative for the check).
- Thaysen's own honesty note: the 17 % "seems exceptional in the light
  of our calculations and the reported SSR and methanogenesis rates from
  the field" (24) — the field is FASTER than model expectations, which
  is exactly what our NOT-BRACKETED verdict reproduces.

### Verdict (printed by the demo, receipt section FIELD VALIDATION)

- LEHEN: predicted law p-box [0, 0.027964] % over 274–292 d vs observed
  [3.0, 3.2] % → **NOT-BRACKETED**, observed lower edge ≈ 107× the
  p-box upper edge.
- LOBODICE: predicted law p-box [0, 0.031462] % over the 25–45 °C ×
  201–219 d envelope vs observed [17.0, 31.5] % → **NOT-BRACKETED**,
  observed lower edge ≈ 540× the p-box upper edge.
- **SHAPE consistent, MAGNITUDE falsified**: the CTMI f p-box is nonzero
  exactly in the measured 25–45 °C window (f(40 °C) ∈ [0, 0.375]; f up
  to [0.038, 0.563] at 45 °C) — the temperature shape survives field
  contact. The Bo-2021-anchored ILLUSTRATIVE magnitude interval
  [0.0048, 0.0187] does NOT bracket either field extent — it was
  calibrated on slow (largely abiotic) field loss, not an active
  methanogen bloom. The demo's validated slot/law paths are UNCHANGED;
  this is additive evidence pointing at brief limitation §6.1(a).
- **HEADLINE STRESS TEST (labeled what-if, same seeded chain)**:
  observed extents annualized LINEARLY (upper bound — the network's
  cumulative loss is concave in time, VERIFIED numerically 2026-08-01
  by integrating the network to τ = 1…30 at the field-calibrated
  corners: non-positive second differences everywhere; strictly concave
  at the LO edge; HI-edge corners saturate ≈100 % in year 1. Convex
  lag-phase profiles are outside this abiotic batch network's form —
  labeled model-form limitation). MECHANISM NOTE: this test does NOT
  apply the linear f_s mapping to the calibrated p-box — it directly
  annualizes the observed 7-month extents (×365.25/210 — 7 months as
  210 d); the two answer
  different questions (screening gates vs bounding bloom what-if):
  LEHEN-like → 1-yr loss [3.844737, 4.101053] %, f_s(1) ∈
  [0.958989, 0.961553], composed P(<6) = **3.055 %** (vs 3.635 %) —
  the τ=1 headline survives; LOBODICE-like → 1-yr loss [29.567857,
  54.787500] %, f_s(1) ∈ [0.452125, 0.704321], composed P(<6) =
  **0.110 %** — the gate moves. Scoping printed in the receipt:
  Lobodice's conditions are exceptionally favorable (shallow,
  near-fresh, pH 6.7, 54 % H₂ town gas); the τ=1 rounding-term
  conclusion is CONDITIONAL on no Lobodice-class bloom.

### Context observations (no verdict — missing T or out of scope)

- BEYNES (France, 1956–1972): "no detected hydrogen consumption in
  Beynes (France)" (29); H₂S up to 40 ppm (Foh et al. 1979, grey).
  Reservoir T NOT FOUND → no bracket check possible.
- KETZIN (Germany, town-gas era): "61% of the volume of hydrogen was
  lost, corresponding to 8 million m3/year" (source 35, via 28); era
  reservoir T NOT FOUND → no bracket check possible.
- SALT CAVERNS (Teesside, Clemens Dome, Moss Bluff, Spindletop, Kiel,
  Yakshunovskoe): gases "containing up to 95% hydrogen ... since the
  1970s ... No loss of hydrogen due to its reactivity was observed"
  (28) — outside this network's scope (no brine-rock loop).
- OLLA CO₂-EOR (Tyne et al. 2021, source 14): CO₂ injection with H₂
  generated in-situ from oil degradation — a different process; used
  only as the community-floor anchor of the [A2] cardinal p-box.

## FIELD CALIBRATION — the replacement k_m magnitude anchor (added 2026-08-01)

The field validation above falsified the Bo-2021 magnitude anchor. This
section documents its replacement: a FIELD-CALIBRATED k_m p-box built
from (1) inverse calibration against the two validated field
observations and (2) the only measured in-situ methanogenesis rate in
the peer-reviewed literature (Tyne 2021). Every number below is printed
by the demo's [A4] section from the network itself (bisection), and
predicted first by `tools/km_law_predict.py`.

### C1 — Inverse calibration (FIELD-DERIVED; transparent)

Method: `field_invert` in the demo — 80 bisection steps of
`law_loss_steps` (the network itself; no closed form, no hidden
constants), bracket [1e-12, 100] chosen inside the RK4-stable monotone
regime (above ~10³ the 0.05-yr step overshoots the H₂ charge and the
loss goes non-monotone — caught by the selftest during development).
Convergence: residual bracket width ≈ 100·2⁻⁸⁰ ≈ 1e-22 — machine-level;
the practical error floor is the RK4 discretization, pinned
independently in the selftest.
Enveloped over (extent edge × horizon-step edge × A × salt) corners:
k_lo = least k consistent (low extent, LONG horizon), k_hi = most
(high extent, SHORT horizon).

- LEHEN k_eff box: **[0.765606, 0.894709]** (model units
  1/(mol/L)/yr at 40 °C; extent [3.0, 3.2] %, steps 15/16).
- LOBODICE k_eff envelope: **[6.308105, 14.708991]** (T-grid
  25:5:45 °C × steps 11/12). Buzek caprock-leakage caveat RETAINED —
  leakage-inflated, used only for the HI edge.
- Decoupling check printed: the fractional H₂ loss is nearly
  independent of the A × salt corners (R0 = ks[0]·[H₂]·[CO₂] is the
  only H₂ sink); the residual ~2 % spread enters through the network's
  CO₂ supply and is enveloped, not assumed away.

### C2 — Tyne et al. 2021 in-situ rate + unit bridge (IN-SITU-MEASURED)

Primary, open access (CC-BY, PMC8695373; source 14 full text read
2026-08-01). The ONLY measured in-situ rate in the paper:

> "We calculate an in situ microbial methanogenesis rate from within a
> natural system of 73–109 millimoles of CH₄ per cubic metre (standard
> temperature and pressure) per year for the Olla Field." (abstract)

> "By extrapolating our results over the 29 years between the cessation
> of injection (1986) and sampling (2015), and assuming 13–19% microbial
> consumption of CO₂ since injection … we calculate that a minimum of
> 1.15 × 10⁷–1.72 × 10⁷ m³ (STP) of microbial CH₄ has been produced at
> a minimum rate of 73–109 mmol CH₄ m⁻³ (STP) yr⁻¹." (main text — note
> "minimum", twice: the rate is a conservative LOWER bound)

Temperature window (verbatim): "the current temperatures (29.2–50.7 °C)
in the Olla reservoirs". Process: hydrogenotrophic (their eq. 1:
CO₂ + 4H₂ → CH₄ + 2H₂O), H₂ "sourced from the hydrocarbons and water".

UNIT BRIDGE to the model's ks[0] (labeled assumptions, printed in the
receipt):
- stoichiometry 4 H₂ : 1 CH₄ → r_H2 = [292, 436] mmol m⁻³ yr⁻¹;
- model R0 volumetric rate = ks[0]·[H₂]·[CO₂] mol/L/yr → ×10⁶
  mmol/m³/yr, with [CO₂] = 0.05 mol/L (the network's charge) and
  [H₂] = the screening charge 7.8e-4·15·salt mol/L, salt ∈ [0.70, 1.00];
- "per cubic metre" read as per m³ of reservoir WATER. **The paper
  never defines the normalization volume** (Methods searched — NOT
  FOUND; the derivation reads as per remaining injected-CO₂ volume at
  STP). This is a documented bridge ambiguity of ~1–2 orders of
  magnitude — LARGER than the width of the bridged box itself; the
  box is conditional on the per-m³-water reading.
- Olla's actual dissolved [H₂] is not reported (H₂ is internally
  generated); using the screening charge is a labeled assumption, and
  k scales inversely with the true [H₂].

Bridged TYNE k_eff box: **[0.499145, 1.064713]** (29.2–50.7 °C).

### C3 — Overlap analysis (k_eff at field temperature)

- LEHEN [0.765606, 0.894709] ∩ TYNE [0.499145, 1.064713] =
  **[0.765606, 0.894709] — NONEMPTY**. Two independent in-situ
  evidences (H₂-storage trial at a measured 40 °C; CO₂-EOR field at
  29.2–50.7 °C) are mutually consistent on the effective rate at
  ~40 °C. Read as WEAK corroboration only — a nonempty interval
  intersection does not address the differing site conditions, and
  does not explain Lobodice.
- LOBODICE [6.308105, 14.708991] sits ~5.9× above the Tyne upper edge —
  consistent with its caveats (leakage + exceptionally favorable bloom
  conditions); retained as the conservative HI edge only.

### C4 — The field-calibrated k_m p-box at Topt (f = 1)

- LO = LEHEN k_lo / f_hi(40 °C = 0.375) = **2.041617** — caveat-free
  site, biology at its p-box-BEST shape (the weakest magnitude the
  Lehen observation allows).
- HI = LOBODICE k_hi / f_lo(45 °C = 0.038461538462) = **382.433772** —
  caveated site, biology at its p-box-WORST shape at the warmest
  reported T. LABELED edge, NOT a strict bound: arbitrarily small f at
  cooler T is p-box-allowed and would push k_m higher.
- TYNE minimal k_m = TYNE k_lo / f_hi(50.7 °C = 0.757212864644) =
  **0.659188 ≤ LO** — the third, independent evidence is consistent.
- vs the LAB-FALSIFIED Bo-2021 anchor [0.0048, 0.0187]: the field box
  is **109× (LO) to 20451× (HI)** larger. The magnitude is REPLACED,
  not adjusted. The calibrated k_m is an EFFECTIVE bulk first-order
  constant that absorbs local attached-biomass density (see C5).

### C5 — Interpretive layer: why field ≫ the old anchor (sourced; NOT a fudge factor)

- Tyne 2021 itself quotes the lab-vs-field gap (verbatim): "Previous
  estimates for CO₂ reduction following methanogenic oil degradation by
  hydrogenotrophic methanogens in lab microcosm incubations at similar
  temperatures are significantly lower (about 0.01–0.15 mmol CH₄ m⁻³
  (STP) yr⁻¹) by comparison." (their ref. 38 = source 37, Gray et al.
  2009) — in-situ 73–109 vs lab 0.01–0.15: ~500–10000×.
- Thaysen et al. 2021 (source 24; EarthArXiv preprint full text):
  field bulk-reservoir methanogenesis 0–1185 nM/h vs near-well rates up
  to 4533 nM/h — "H2 consumption rates by SSR and methanogenesis were
  up to 2544 and 4533 nM h-1, respectively, which falls within the
  lower range of the values reported from laboratory studies" — and
  "the field H2 consumption by SSR is 1.5 times to eight orders of
  magnitude lower, and 0.7 times to 7 orders of magnitude lower for
  methanogenesis" (vs lab). The variance across spatial niche (bulk vs
  colonized near-well) is the documented 2–8-orders effect.
- Tremosa et al. 2023 (source 28): reproducing Lobodice required
  "dividing the rate of methanogenesis by about 4" (zero-order model)
  or "net rates divided by 50 for methanogenesis" (Monod model) —
  "it could also be that not kmax, but the concentration of bacteria is
  lower–or both": biomass density is the free parameter that absorbs
  lab-to-field discrepancies.
- Haddad et al. 2022 (source 30): with real reservoir rock +
  autochthonous formation water at 47 °C, "nearly 40% of injected H2
  transformed" in <90 days and "the rate of H2 decline between days 53
  and 84 was remarkably greater than before day 52" — conversion
  ACCELERATES with colonization; methanogenesis switched on when
  sulfate fell below 0.08 mmol (mcrA transcripts peaking
  1.8×10⁴ copies/mL).
- Berta et al. 2018 (source 19): in sediment columns at pH₂ 2–15 bar,
  H₂ consumed via sulfate reduction at 18 ± 5 µM/h with acetate
  production at 0.030 ± 0.006 h⁻¹ — but "no methanogenesis took place":
  active blooms can be ABSENT even under excess H₂ (the demo's seam
  (d), and the reason the calibrated magnitude must not be read as
  universal).
- Honest framing constraint (from the same evidence): the literature
  does NOT support a universally faster field — bulk-reservoir rates
  can be far LOWER than lab optima (Thaysen); the calibrated box here
  is anchored on ACTIVE-BLOOM field sites, which is exactly the
  conservative screening question.

### C6 — Provenance caveat on the OLD anchor (documentation hygiene)

An independent re-check (2026-08-01, during this calibration work)
found that Bo, Zeng & Chen 2021 (source 17) is an ABIOTIC PHREEQC
geochemistry study whose abstract reports no microbial methanogenesis
rate constants; the [0.0048, 0.0187] interval's provenance inside that
paper could not be re-verified from accessible text. This weakens the
"lab anchor" narrative but changes NO number: the interval was already
labeled ILLUSTRATIVE, is now empirically falsified by two field
observations, and is replaced by the C4 box. Flagged for a future
sourcing pass; the validated slot/law numbers that use it are byte-
identical and untouched.

## NOT FOUND (searched, not sourced — do not use without new data)

- Formation-water **salinity (TDS)** for every screened site (only the
  Hystories 100 g/L model default exists).
- A **measured geothermal gradient for the Mesohellenic Trough** proper
  (nearest measured: Strymon basin 25–36 °C/km; Nestos "very high",
  qualitative).
- **TRIERES geological deliverables** — none public; project scope is
  above-ground H₂ at Corinth (CORDIS, reporting tab empty as of access).
- **IGME primary public reports** — Greek onshore seismic/borehole data
  "not publicly available as yet" (Hystories D1.4); IGME appears as data
  copyright holder only.
- Permeability for the Mesohellenic formations; porosity of the
  W. Thessaloniki aquifers; depth/T/salinity beyond the abstract for the
  South Kavala reservoir (BGSG 2001 full text not downloadable).
- **Ghaedi 2025's kinetic methanation rate law** (functional form,
  constants, units) — closed-access body text; no preprint, repository,
  or indexed open copy exists (OpenAlex/Unpaywall/S2/EarthArXiv/SSRN
  searched 2026-08-01). Only the equilibrium-side log K(T) calibration
  is abstract-visible.
- A **Ratkowsky square-root fit or any published Arrhenius fit specific
  to methanogens** — none exists in Crossref/OpenAlex (Ratkowsky-model
  literature is food/soil bacteria only); the CTMI cardinal form is the
  available sourced alternative.
- **Methanogen thermal-death kinetics** (D-values/decimal reduction
  times) — not published; the sharpest sourced cutoff statement is
  Zeikus & Wolfe 1972's "nothing occurred above 75 C" (≤ 5 °C wide).
- Extractable parameter numbers from Dopffel et al. 2021 (DOI
  10.1016/j.ijhydene.2020.12.058) and from the published Thaysen et al.
  2021 + its 2023 corrigendum — all closed; preprint numbers carry the
  corrigendum caveat.
- **Tyne et al. 2021 "lag phases"** — do not exist: full-text + SI
  search finds no occurrence of "lag" and no lag times anywhere in the
  paper (checked 2026-08-01). Closest measured onset data: first
  detectable CH₄ after 17–254 days in 37 °C bottle incubations of
  aquifer formation water (source 32) — bottle onset times, not a fitted
  lag parameter; NOT implemented as a lag path.
- **MEASURED reservoir-condition Monod parameters** — none: the only
  reservoir-tuned values are Tremosa 2023's FITTED Lobodice set (28,
  Table 3: methanogenesis kmax = 7.96e-2 mol/molC/s, K½ = 1.5e-4 mol/L;
  the lab kmax = 4 mol/molC/s must be divided by ~50 to match the
  field). Pure-culture measured Ks(H₂)/µmax papers are paywalled with no
  extractable numbers (21, 33, 20). No separate Monod path implemented —
  documented negative.
- **Beynes and Ketzin (town-gas era) reservoir temperatures** — not
  stated in any accessible source (Foh 1979 and source 35 are offline
  grey literature), so their observed H₂ fates cannot be bracket-checked.
- **Underground Sun Conversion (Pilsbach) reservoir temperature + final
  reports** — USC/USC-FlexStore final reports are gated behind
  email-request walls; the "USC base" case (20 % H₂ → 11.1 % H₂ after
  conversion, source 34) is derived from the test series but states no
  reservoir T. Site is co-located with the USS trial (~40 °C inferred,
  NOT sourced — not used).
- **Rubensdorf / Underground Sun Storage 2030 quantitative results** —
  two pure-H₂ cycles completed April 2025; final report pending; the
  peer-reviewed account (source 36) has no accessible text yet.
- **Tyne 2021's rate normalization volume** — "per cubic metre (STP)"
  is never defined (per m³ reservoir bulk? pore water? remaining
  injected-CO₂ gas at STP?): the PMC full text + Methods contain no
  reservoir/pore-volume normalization (searched 2026-08-01). The unit
  bridge in C2 carries this ~1–2-order ambiguity explicitly.
- **Tyne 2021 per-cell rates, absolute cell densities, H₂ consumption
  rates, doubling/turnover numbers, Monod parameters** — none reported
  anywhere in the paper or its SI (only 16S relative abundances; "the
  abundance of methanogens was 100 times higher than ANME").
- **A controlled biofilm-vs-planktonic hydrogenotrophic-methanogenesis
  rate ratio** (same inoculum, same chemistry, attached vs suspended) —
  no peer-reviewed measurement exists (the one "suspended vs biofilm"
  rate study cited inside Thaysen 2021 is a 1992 PhD thesis, excluded).
  The interpretive layer in C5 uses volumetric niche contrasts instead.
- **Heinemann et al. 2021 full text** (source 29) — hybrid OA but the
  publisher blocks automated retrieval (HTTP 403 HTML + PDF, also via
  reader proxy, 2026-08-01); only the one-sentence abstract obtainable.
  Numbers attributed to it here remain via Tremosa 2023/Thaysen 2021.
- **Dopffel et al. 2021** (source 38) — closed access, no OA copy
  (OpenAlex/Unpaywall checked); abstract qualitative, no extractable
  rate numbers. Not used.
- **Ebigbo, Golfier & Quintard 2013** (pore-scale biofilm
  methanogenesis model, source 39) — identified as a lead for a
  mechanistic biomass-density path; not fetched, not used.

## Source list (all accessed 2026-07-31)

1. HRADF, *UGS South Kavala — Invitation for Expression of Interest*,
   Annex B "Technical Description of the Project" (2020).
   https://hradf.com/wp-content/uploads/2021/12/UGS-South-Kavala-Invitation-for-Expression-of-Interest-ENG_22_06_2020.pdf
2. Koukouzas, Ritss et al. 2021, *Energies* 14:3321. DOI 10.3390/en14113321
3. Hystories D1.4 (2023), *Opportunities in Europe for H₂ geological
   storage in depleted fields and aquifers* (GA 101007176), §12 Greece.
   https://hystories.eu/wp-content/uploads/2023/12/Hystories_D1.4-0-Opportunities-in-Europe-for-H2-geological-storage-in-depleted-fields-and-aquifers.pdf
4. HyUSPRe D1.3 (2022), *Hydrogen storage potential of existing European
   gas storage sites* (GA 101006632), p. 31.
   https://www.hyuspre.eu/wp-content/uploads/2022/06/HyUSPRe_D1.3_Hydrogen-storage-potential-of-existing-European-gas-storage-sites_2022.06.29.pdf
5. Dotsika et al. 2021, *Scientific Reports* 11:16291.
   DOI 10.1038/s41598-021-95656-6
6. Arvanitis et al. 2020, *Energies* 13:2707. DOI 10.3390/en13112707
7. Proedrou 2001, *Bull. Geol. Soc. Greece* 34(3):1221–1228.
   DOI 10.12681/bgsg.17198
8. Proedrou & Papaconstantinou 2004, *Bull. Geol. Soc. Greece* 36(1):327.
   DOI 10.12681/bgsg.16675
9. Energies 2023, 16:2392 (Prinos AGI). DOI 10.3390/en16052392
10. Tasianas & Koukouzas 2016, *Energy Procedia* 86:334–341.
    DOI 10.1016/j.egypro.2016.01.034
11. CORDIS TRIERES project page (GA 101112056).
    https://cordis.europa.eu/project/id/101112056

k_m(T) law-path sources (all accessed 2026-08-01, DOIs Crossref-verified):

12. Rosso, Lobry & Flandrois 1993, *J. Theor. Biol.* 162:447–463.
    DOI 10.1006/jtbi.1993.1099 (CTMI model form)
13. Zeikus & Wolfe 1972, *J. Bacteriol.* 109(2):707–713.
    DOI 10.1128/jb.109.2.707-713.1972 (cardinal temps 40 / 65–70 / 75 °C;
    OA: Europe PMC PMC285196)
14. Tyne et al. 2021, *Nature* 600:670–674. DOI 10.1038/s41586-021-04153-3
    (in-situ reservoir methanogenesis 73–109 mmol CH₄ m⁻³ yr⁻¹ at
    29.2–50.7 °C; OA: PMC8695373)
15. Head, Gray & Larter 2014, *Front. Microbiol.* 5:566.
    DOI 10.3389/fmicb.2014.00566 (80–90 °C field biosphere cutoff; OA)
16. Wilhelms et al. 2001, *Nature* 411:1034–1037. DOI 10.1038/35082535
    (palaeopasteurization primary; closed)
17. Bo, Zeng & Chen 2021, *Int. J. Hydrogen Energy* 46(38):19998–20009.
    DOI 10.1016/j.ijhydene.2021.03.116 (30-yr field-loss k_m anchor)
18. Hellerschmied et al. 2024, *Nat. Energy*. DOI 10.1038/s41560-024-01458-1
    (0.26 mmol L⁻¹ h⁻¹ at 40 °C/40 bar mesocosm; OA)
19. Berta et al. 2018, *Environ. Sci. Technol.* 52:4937–4949.
    DOI 10.1021/acs.est.7b05467 (negative control; closed)
20. Robinson & Tiedje 1984, *Arch. Microbiol.* 137:26–32.
    DOI 10.1007/bf00425803 (Km(H₂) 2.5–13 µM)
21. Schönheit, Moll & Thauer 1980, *Arch. Microbiol.* 127:59–65.
    DOI 10.1007/bf00414356 (µmax 0.69 h⁻¹, Ks(H₂) ~20 % gas at 65 °C)
22. Price & Sowers 2004, *PNAS* 101:4631–4636. DOI 10.1073/pnas.0400522101
    (Arrhenius Ea 40–75 / ~110 kJ mol⁻¹; OA: PMC384798)
23. Lovley 1985, *Appl. Environ. Microbiol.* 49:1530–1531.
    DOI 10.1128/aem.49.6.1530-1531.1985 (H₂ threshold 6.5–12 Pa; OA)
24. Thaysen et al. 2021, *RSER* 151:111481. DOI 10.1016/j.rser.2021.111481
    (+ corrigendum 2023, DOI 10.1016/j.rser.2022.113039) — context only
25. Hagemann et al. 2016, *Comput. Geosci.* 20:595–606.
    DOI 10.1007/s10596-015-9515-6 (dimensionless Monod, "very high
    uncertainty" — context only)

Field-validation sources (all accessed 2026-08-01, DOIs Crossref-verified):

26. Šmigáň, Greksák, Kozánková, Buzek, Onderka & Wolf 1990, *FEMS
    Microbiol. Lett.* 73:221–224. DOI 10.1016/0378-1097(90)90733-7
    (Lobodice primary; paywalled — numbers via 24/28/29/30)
27. Buzek, Onderka, Vančura & Wolf 1994, *Fuel* 73:747–752.
    DOI 10.1016/0016-2361(94)90019-1 (Lobodice isotope attribution,
    incl. caprock leakage; paywalled)
28. Tremosa, Jakobsen & Le Gallo 2023, *Front. Energy Res.* 11:1145978.
    DOI 10.3389/fenrg.2023.1145978 (Lobodice verbatim composition table;
    Ketzin 61 %; Beynes; salt-cavern negatives; OA)
29. Heinemann et al. 2021, *Energy Environ. Sci.* 14:853–864.
    DOI 10.1039/d0ee03536j ("no detected hydrogen consumption in Beynes
    ... up to a 17% decrease ... in Lobodice over a seven month cycle";
    OA)
30. Haddad et al. 2022, *Energy Environ. Sci.* 15:3400–3415.
    DOI 10.1039/d2ee00765g (Lobodice H₂ 54→37 %, CH₄ 21.9→40 %; OA)
31. Hellerschmied et al. 2024, *Nature Energy* 9:333–344.
    DOI 10.1038/s41560-024-01458-1 (Lehen field-trial primary — the
    same paper as source 18, which cited its mesocosm anchor; OA)
32. Ranchou-Peyruse et al. 2024, *FEMS Microbiol. Ecol.* 100:fiae066.
    DOI 10.1093/femsec/fiae066 (first-CH₄ onset 17–254 d at 37 °C in
    aquifer formation-water incubations; OA)
33. Kristjansson, Schönheit & Thauer 1982, *Arch. Microbiol.*
    131:278–282. DOI 10.1007/BF00405893 (canonical Ks(H₂)
    methanogen-vs-sulfate-reducer comparison; paywalled)
34. Energies 2022, 15:1021 (Underground Sun Conversion TEA, "USC base"
    case from the Pilsbach test series). DOI 10.3390/en15031021 (OA)
35. MARCOGAZ 2017, *Guidance: Injection of hydrogen/natural gas
    admixtures in underground gas storage (UGS)* (grey literature, no
    DOI — Ketzin 61 % H₂ loss, quoted via 28)
36. Andiappan et al. 2026, *Int. J. Hydrogen Energy* (Rubensdorf /
    USS-2030 field demonstrator account). DOI 10.1016/j.ijhydene.2026.156784
    (registered; no accessible text yet)
37. Gray, Sherwood Lollar, Ballentine et al. 2009, *Extremophiles*
    13:511–519. DOI 10.1007/s00792-009-0237-3 (lab microcosm
    methanogenesis rates 0.01–0.15 mmol CH₄ m⁻³ (STP) yr⁻¹ for North
    Sea formation waters — quoted VERBATIM inside Tyne 2021's main
    text, source 14; primary paywalled, numbers via 14)
38. Dopffel, Jansen & Gerritse 2021, *Int. J. Hydrogen Energy*
    46:8591–8606. DOI 10.1016/j.ijhydene.2020.12.058 (UHS microbiology
    review; closed access — abstract only, NO numbers used)
39. Ebigbo, Golfier & Quintard 2013, *Adv. Water Resour.* DOI
    10.1016/j.advwatres.2013.09.004 (pore-scale biofilm methanogenesis
    model — a LEAD for a mechanistic biomass path; not fetched, not
    used)
40. Tyne et al. 2021 open-access full text: PMC8695373 (CC-BY) — the
    copy actually read for the C2 quotes (same DOI as source 14).
