# Gkanas et al. 2020 — the seven-stage MHHC, as published

**Recovered 2026-09-09 from the open-access accepted post-print**, which
`demos/hydrogen/README.md` previously recorded as paywalled. Every number
below is transcribed from that document; nothing here is reconstructed.

| field | value |
|---|---|
| authors | E. Gkanas, C. Christodoulou, G. Tzamalis, **E. Stamatakis**, A. Chroneos, K. Deligiannis, G. Karagiorgis, A. K. Stubos |
| title | Numerical Investigation on the Operation and Energy Demand of a Seven-Stage Metal Hydride Hydrogen Compression System for Hydrogen Refuelling Stations |
| journal | Renewable Energy **147** (2020) 164–178 |
| DOI | 10.1016/j.renene.2019.08.104 |
| open copy | `https://pure.coventry.ac.uk/ws/files/30889947/Binder5.pdf` — Coventry University repository, author post-print, CC-BY-NC-ND (GREEN OA) |
| pdf sha256 | `b2fa55296105da59c5744874a15a2fb5de0d7f583f78ab8968d08ad1ba56d5c4` |

**Authorship correction.** `demos/hydrogen/README.md` called this "his
first-author seven-stage paper". Dr. Emmanuel Stamatakis is the **fourth**
author; E. Gkanas is first. Corrected in that file.

---

## Table 3 — per-stage materials and thermodynamics

Measured by the authors on a Sievert-type apparatus (Hidden Isochema), with
XRD (Bruker AXS D8-Advance) and SEM (Zeiss NEON 40 EsB) characterisation;
ΔH and ΔS obtained from van 't Hoff plots over at least three isotherms.

| stage | type | ΔH abs (J/mol H₂) | ΔS abs (J/mol H₂·K) | ΔH des (J/mol H₂) | ΔS des (J/mol H₂·K) |
|---|---|---|---|---|---|
| S1 | AB5 (Mm-based) | 25242 | 104.6 | 28195 | 106.8 |
| S2 | AB2 | 21466 | 94.7 | 26133 | 107.1 |
| S3 | AB2 | 20354 | 101.1 | 24823 | 108.7 |
| S4 | AB2 | 19991 | 100.2 | 20252 | 100.8 |
| S5 | AB2 | 18198 | 98.12 | 19856 | 101.4 |
| S6 | AB2 | 16232 | 98.05 | 19125 | 101.5 |
| S7 | AB2 | 14702 | 98.1 | 18916 | 106.2 |

AB2 alloys are Zr-Ti-Mn-Co-Cr-Fe-V; all intermetallics arc-melted.

**Two things this table settles that a reconstruction got wrong.**

1. **The ΔH ladder DECREASES along the cascade** (25.2 → 14.7 kJ/mol). It
   must: van 't Hoff gives `ln P = ΔS/R − ΔH/(R·T)`, so the high-pressure
   stage needs the *lowest* ΔH to reach its plateau. `mh7_reliability.sio`
   used an increasing representative ladder (24 → 36 kJ/mol), which is the
   physics backwards.
2. **Hysteresis is per-stage and measured**, as the gap between absorption
   and desorption enthalpy — 3.0 kJ/mol at S1, widening to 4.2 kJ/mol at
   S7. It does not need to be lumped into an efficiency factor.

## Geometry and inventory

| quantity | value |
|---|---|
| H₂ compressed per cycle | 0.425 kg |
| alloy mass, stage 1 (AB5) | 31 kg |
| alloy mass, stages 2–7 (AB2) | ≈29.5 kg each |
| reactor length | L = 12 m (simulated as 6 m to bound mesh size) |
| internal radius (hydride bed) | R = 21.4 mm |
| outer radius of heated shell | 30.15 mm |
| external SS 316L wall | 2 mm |
| internal SS wall | 8.75 mm |
| heating/cooling shell thickness | 3.85 mm |
| bed porosity | 0.5 (tanks filled to 50 %) |
| SS 316L shell mass heated/cooled | 136 kg per stage |

## Table 4 — compression cases (initial pressure 20 bar in every case)

| case | T abs (°C) | T des (°C) | final P (bar) | ratio | **cycle time (s)** |
|---|---|---|---|---|---|
| 1 | 10 | 80 | 374 | 18.7 | 6625 |
| 2 | 10 | 90 | 483.2 | 24.16 | 5760 |
| 3 | 10 | 100 | 606.4 | 30.32 | 5090 |
| 4 | 10 | 105 | 682 | 34.06 | 4720 |
| 5 | 10 | 110 | 742 | 36.9 | 4530 |
| 6 | 10 | 120 | 830 | 41.5 | 3760 |

**Attribution correction.** The 365 bar figure quoted in
`demos/hydrogen/README.md` as "delivery pressure 365 bar" is **not** the
delivery pressure. It is the pressure reached at the **third coupling
process** at 120 °C (the paper contrasts 180 bar at 80 °C with 365 bar at
120 °C for that coupling). System delivery is 374 bar in Case 1 and 830 bar
in Case 6.

**The cycle time is the column no Sounio demo currently reproduces**, and it
is what sets throughput: 0.425 kg / 6625 s = 0.231 kg/h in Case 1 against
0.425 kg / 3760 s = 0.407 kg/h in Case 6.

## Table 5 — sensible thermal energy per cycle

| case | SS 316L walls (MJ) | alloy powder (MJ) | total sensible (MJ) |
|---|---|---|---|
| 1 | 63.31 | 8.45 | 71.76 |
| 2 | 72.35 | 9.66 | 82.02 |
| 3 | 81.41 | 10.87 | 92.27 |
| 4 | 85.92 | 11.47 | 97.39 |
| 5 | 90.44 | 12.08 | 102.52 |
| 6 | 99.48 | 13.28 | 112.77 |

**The steel, not the hydride, is the thermal load** — 88 % of the sensible
heat in every case. A design conclusion the equilibrium-only models cannot
reach.

## Table 6 — total thermal energy per cycle

| case | latent (MJ) | sensible (MJ) | H₂ heating (MJ) | total (MJ) | total (kWh) | ratio |
|---|---|---|---|---|---|---|
| 1 | 61.83 | 71.76 | 3.08 | 136.67 | 37.96 | 18.7 |
| 2 | 67.34 | 82.02 | 3.52 | 152.88 | 42.47 | 24.16 |
| 3 | 72.88 | 92.27 | 3.96 | 169.11 | 46.98 | 30.32 |
| 4 | 75.94 | 97.39 | 4.18 | 177.51 | 49.31 | 34.06 |
| 5 | 79.04 | 102.52 | 4.41 | 185.97 | 51.66 | 36.9 |
| 6 | 84.17 | 112.77 | 4.84 | 201.78 | 56.05 | 41.5 |

Per kilogram: 37.96 kWh / 0.425 kg = **89.3 kWh_th/kg** (Case 1), which is
the upper end of the 44–89 kWh_th/kg span `hub_chain.sio` already cites.

## Table 7 — efficiencies

| case | isothermal η | isothermal n_c | isentropic η | isentropic n_c | polytropic η | polytropic n_c | adiabatic irrev. | isentropic rev. n_c |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.78 | 14.05 | 4.39 | 22.16 | 3.98 | 20.01 | 3.29 | 16.62 |
| 2 | 2.71 | 12.28 | 4.45 | 19.99 | 3.99 | 17.93 | 3.34 | 14.99 |
| 3 | 2.62 | 10.87 | 4.48 | 18.59 | 3.98 | 16.50 | 3.36 | 13.94 |
| 4 | 2.58 | 10.28 | 4.51 | 17.93 | 3.99 | 15.88 | 3.38 | 13.45 |
| 5 | 2.52 | 9.66 | 4.46 | 17.08 | 3.93 | 15.01 | 3.34 | 12.81 |
| 6 | 2.40 | 8.58 | 4.33 | 15.47 | 3.81 | 13.59 | 3.25 | 11.60 |

## The model the paper solves

- **Absorption rate (eq. 9):** `ṁ_abs = C_a · exp(−E_a/(R_g·T_s)) · ln(p_g/P_eq) · (ρ_ss − ρ_s)`
- **Desorption rate (eq. 10):** `ṁ_des = C_d · exp(−E_d/(R_g·T_s)) · ((p_g − P_eq)/P_eq) · ρ_s`
- **Equilibrium pressure (eq. 11):** van 't Hoff **with a plateau-slope and
  hysteresis correction** — `ln P_eq = (ΔH/(R_g·T) − ΔS/R_g) + (σ_s + σ_0)·tan(π·(X/X_max − ½)) ± Y/2`,
  where σ_s and σ_0 are plateau-slope flatness factors and Y is the isotherm
  hysteresis.
- **Interconnector coupling (eq. 12):** `n_t = n_in + n_des − n_abs`, then
  the interconnector pressure from the ideal-gas relation. **This is what
  makes a cascade a cascade** and why the achieved ratio is not the product
  of free per-stage van 't Hoff ratios.
- Effective thermal conductivity by the Zehner–Bauer–Schlünder packed-bed
  model; Darcy flow with Kozeny–Carman permeability.

**What the paper does NOT give numerically**, and therefore what any
reproduction must source elsewhere or declare: the values of `C_a`, `C_d`,
`E_a`, `E_d` (cited to its ref. [95]), and the values of `σ_s`, `σ_0`, `Y`.
The forms are given; the constants are not.

## Why the naive cascade model cannot reproduce this

Taking the free per-stage van 't Hoff ratio `P_des(T_hot)/P_abs(T_cold)`
from the Table 3 values and multiplying over seven stages gives **6640×**
for Case 1, not 18.7×. The stages do not each achieve their free ratio: they
are coupled through the interconnector, and each stage absorbs at whatever
pressure the previous stage can deliver into it. Reproducing 374 bar and
6625 s therefore requires the coupled model, not a ratio chain — which is
why a ratio chain can only match the published numbers by fitting a lumped
efficiency to them.
