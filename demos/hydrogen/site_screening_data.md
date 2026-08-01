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
| **Reservoir temperature** | **95 °C (measured)** | HRADF Annex B: "Reservoir Temperature: 95 °C" |
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
  observed extents annualized LINEARLY (upper bound — loss is concave
  in time): LEHEN-like → 1-yr loss [3.844737, 4.101053] %, f_s(1) ∈
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
