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
