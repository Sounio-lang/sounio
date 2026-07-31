# Epistemic screening of Greek underground H₂-storage candidates

**A deterministic, fully-sourced first look for the TRIERES conversation**

*Draft brief — all numbers re-derive from `demos/hydrogen/site_screening.sio`
(lean_single, seeded, receipt `SITE_SCREENING_OK`); geology table with
per-value citations in `demos/hydrogen/site_screening_data.md`. Figures
rendered from the demo's own stdout by
`demos/hydrogen/tools/render_site_figures.py`.*

## 1. Executive summary

We screened three real, publicly documented Greek H₂-storage candidate
formations through the validated H₂–brine–calcite kinetic network
(Ghaedi et al. 2025 skeleton, public PHREEQC/PWP rates) **at each site's
own sourced temperature bracket**, and pushed the resulting 30-year
H₂-loss p-boxes through the TRIERES wellhead-to-dispensed cost chain to
the 6 EUR/kg gate.

The headline is a negative result of the useful kind: **at the valley's
1-year storage residence, which candidate hosts the store barely moves
any number that matters** — the availability factor f_s stays within
~0.1 % of 1 for every site, the gate probability sits at ~3.6 %
(composed conventional) for all three, and the distribution-free p-box
on beating 6 EUR/kg is [0, 100] % everywhere because the compressor
reliability p-box, not the rock, is the gate. Site choice starts to
matter at longer residence (τ = 10 yr sensitivity, §5) and in the
kinetic-regime structure the sourced temperatures reveal (§4): the
three sites sit in three different regimes of the same public network.

**What changes the answer.** Three levers, in order of leverage: (1)
**compressor reliability evidence** — the gate is set by the R p-box,
not the rock, so alloy-batch test data moves the headline the most;
(2) **a calibrated k_m(T)** — Ghaedi 2025's T-dependent kinetic law
(closed-access) would replace the hard 70 °C step and reshape every
below-cutoff number; (3) **measured S2/S3 formation temperatures** —
the gradient-derived brackets are the widest honest input intervals in
the screen.

## 2. The sites (every value cited — `site_screening_data.md`)

| | S1 South Kavala | S2 Pentalofos Fm | S3 Eptachori Fm |
|---|---|---|---|
| Type | depleted gas field, turbiditic sands in Miocene anhydrite/salt | turbiditic sandstone saline aquifer, Tsotyli caprock | conglomerates/sandstones, marine shale top |
| Where | offshore N. Aegean, ~6 km off Thassos | Mesohellenic Trough, Grevena sub-basin (W. Macedonia) | same trough |
| Depth | top < 1630 m ss; GWC 1723 m | avg 1500 m (deepest −2544 m) | avg 2000 m |
| **Temperature** | **95 °C, MEASURED** (HRADF 2020) | **[52.5, 69.0] °C** — gradient-derived union bracket (labeled) | **[65, 87] °C** — same construction (labeled) |
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
  calcite kinetics k(T), phreeqc.dat calcite equilibrium, an
  ILLUSTRATIVE methanation pseudo-sink k_m that is **zero above 70 °C**
  (abstract-level A2 slot of Ghaedi 2025), 30-yr RK4 integration. Per
  site, the 8 (k_m × A × salt) epistemic corners are run at the site's
  T-bracket corners (and mid), giving the site's 30-yr loss **p-box** —
  no independence assumption anywhere.
- **Chain**: `valley_chain_epistemic.sio` (#1587) machinery — f_s =
  1 − (L30/100)(τ/30), f_c = R with the pinned compressor p-box
  R ∈ [0.0131, 0.9989], CF_eff = f_s·f_c·CF into the TRIERES cost
  chain; conventional (independent-uniform, seeded MC n = 20000) and
  distribution-free corner answers. Corner-exactness of this
  composition is machine-checked in
  `formal/lean4/SounioHydrogenValleyPbox.lean`.
  Two stated assumptions: **f_s is linear in τ** — extrapolating the
  30-yr loss to τ ≪ 30 linearly *over*estimates short-residence losses
  (kinetic loss is concave in time: fastest early), so this choice
  strengthens, not weakens, the "subsurface is a rounding term at
  τ = 1" conclusion; and **the R p-box is representative, not
  measured** — it is a corner p-box on P(P7 ≥ 350 bar) from
  `mh7_reliability.sio`'s 7-stage compressor ladder under alloy batch
  uncertainty (the source's Table 3 failure data are paywalled), not a
  fitted field-failure distribution.
- **Only temperature is site-specific in this machinery** — depth,
  porosity, permeability shape capacity/injectivity (§6) but have no
  slot in the loss network; no measured Greek formation-water salinity
  exists, so the salting-out interval stays the component demo's
  illustrative [0.70, 1.00]. Stated, not smoothed over.

## 4. Results

**Figure A — per-site loss p-box fan** (`figures/fig_a_loss_pbox_fan.png`).
The sourced T brackets put the sites in **three different kinetic
regimes**: S1 (measured 95 °C) lies entirely above the 70 °C
interaction cutoff — 30-yr kinetic loss **exactly [0, 0] %** by the
slot; S2 (52.5–69 °C) lies entirely below it — loss p-box
**[0.0845, 2.2760] %**; S3 (65–87 °C) **straddles** the cutoff, so its
p-box **[0, 1.9806] %** honestly contains the interaction-free regime.
Mechanism: the network's sole H₂-consuming pathway is the methanation
pseudo-sink; the calcite dissolution/precipitation loop shifts brine
chemistry but is not itself an H₂ sink, which is why the k_m slot's
70 °C step dominates the fan (the step is a slot artifact; see §6.1).
Below the cutoff, loss **decreases** with temperature (the pseudo-sink
kinetics slow as T falls), so S2's colder bracket corner carries the
higher upper bound.
(Straddle honesty: the fan resolves the 70 °C cliff with corner runs at
69.95/70.05 °C. T-corner
extrema are *labeled scan evidence*, and the mid-bracket fan box the
demo runs anyway lies inside the corner envelope for every site —
printed in the receipt.)

**Figure B — gate probability vs baselines**
(`figures/fig_b_pgate_vs_baselines.png`). Conventional composed
P(<6 EUR/kg): **3.635 % for all three sites** (valley 25 °C baseline
3.630 %; no-coupling baseline 20.765 %). The 20.765 → 3.630 %
baseline-to-baseline drop is entirely the compressor factor f_c = R
entering CF_eff; the subsurface contributes nothing visible at τ = 1.
The subsurface-only rows
(20.505–20.530 %) are within MC noise of the no-coupling baseline —
per-site separation at τ = 1 yr does not survive n = 20000 sampling
error, and we say so in the receipt. The corner p-box on beating the
gate is **[0, 100] % for every site**.

**Figure C — composed-chain build-up** (`figures/fig_c_chain_waterfall.png`).
At interval mids: nominal 6.4160 EUR/kg → +subsurface **+0.0000–0.0005
EUR/kg** → +compressor → ~7.556 EUR/kg. The subsurface step is
sub-cent; the compressor availability (R mid = 0.506) is the cost
driver.

## 5. Where site choice starts to matter

The τ = 10 yr analytic sensitivity: f_s intervals drop to
S2 [0.9924, 0.9997] and S3 [0.9934, 1.0000] (S1 stays [1, 1] by the
slot). A seasonal-plus storage mandate (multi-year residence, or a
strategic reserve) is where the warmer onshore aquifers' kinetic losses
become visible against South Kavala's measured-hot, interaction-free
regime. If the valley's storage residence is closer to τ = 1 (our
ILLUSTRATIVE default), the subsurface is a rounding term at every
candidate and the characterization euros belong to the compressor
alloys and the heat/dispensing contracts — same conclusion as the
valley-chain receipt, now backed by real site temperatures.

## 6. Honest limitations

1. **k_m is a slot, not a law.** The methanation pseudo-sink is
   ILLUSTRATIVE (anchored to Bo 2021 field losses) with an
   abstract-level 70 °C cutoff; Ghaedi 2025's calibrated T-dependent
   kinetic law is closed-access. Every S1/S3 "interaction-free" number
   inherits this, and **the hard step itself is a modeling choice, not
   derived from the underlying kinetics** — the correct epistemic
   response is exactly what we print: the straddling site's p-box
   contains both regimes. AWAITING-AUTHOR-DATA: drop the paper's law
   into the 2-line `km_at` slot and every number re-derives.
2. **Two of three temperatures are gradient-derived**, not measured —
   labeled union brackets (Hystories 30 ± 3 °C/km model default ∪
   Strymon-basin measured 25–36 °C/km). No measured Mesohellenic
   gradient exists in the public record.
3. **No measured salinity for any site** (Hystories 100 g/L default is
   a screening assumption); salinity enters the network only through
   the illustrative salting-out interval.
4. **Batch network, fixed pH 7, pH₂ = 15 atm, fixed brine** — a
   screening reactor, not a reservoir model; depth/porosity/
   permeability (the capacity and injectivity drivers) are out of its
   scope and reported as context only.
5. **MC noise**: per-site conventional differences at τ = 1 are within
   n = 20000 sampling error; only p-box structure and corner results
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
