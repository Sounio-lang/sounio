<!-- docs:meta
topic_id: repo.docs.dissertation.advisor-handoff
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.advisor-handoff
-->

# Dissertation Viewer — Advisor & Committee Walkthrough

**Audience:** advisor (pharmacologist) and committee members (no
programming background expected).

**Live URL:** <https://www.souniolang.org/dissertation/>

**Time needed:** ~10 minutes for the six guided tours (three per drug),
plus as long as you want to explore.

---

## What you are looking at

The page opens to a 3D anatomical scene of a 70 kg adult. The 14
spheres are the compartments of the PBPK model — heart, lung, liver,
kidney, brain, gut, muscle, adipose, skin, bone, spleen, pancreas,
blood, "other tissues." A chip-row selector at the top of the canvas
(or the <kbd>D</kbd> key) toggles the active drug between **Rapamycin**
(Cypher coronary stent — the original dissertation drug) and
**Semaglutide** (Ozempic / Wegovy — a 4114 g/mol GLP-1R peptide).

For **Rapamycin**, the small silver cylinder in the chest is the
Cypher drug-eluting stent at its coronary location; yellow particles
streaming upward from it are rapamycin being eluted at the Higuchi
rate *dQ/dt = K<sub>H</sub> / (2√t)* (140 µg total, ~80% over 30 days,
per the Cordis 2003 IFU).

For **Semaglutide**, the chest cylinder is replaced by a lower-abdomen
ellipsoid representing the subcutaneous depot of a 1 mg weekly dose;
cyan particles trace first-order absorption at rate
*F · Dose · k<sub>a</sub> · e<sup>−k<sub>a</sub>t</sup>* with
*k<sub>a</sub> = ln 2 / 60 h*, *F = 0.89* (Overgaard 2019,
Carlsson 2020).

The colour of each organ tracks the **current drug concentration**:
blue = empty, red = at the literature C<sub>max</sub> for that organ.
The translucent shell around each organ is the **GUM uncertainty
cone** — its radius is the propagated standard uncertainty *u* at the
current parameters. As you change a parameter slider, you should see
the shells widen or shrink.

A clock at top-left tracks simulation time (0 → 30 days). Top-right
is a **📸 Snapshot PNG** button — one click produces a print-quality
handout of the current frame.

---

## The three contributions, visualised

### 1. GUM-through-ODE (the GUM uncertainty cone)

Conventional PBPK reports a single concentration curve. The viewer
shows the *cone of uncertainty around the curve* — propagated from
parameter uncertainty through the ODE step-by-step. Move the
**patient profile** selector at the bottom-right to "low CL" (CYP3A4
poor metaboliser): watch the liver and kidney cones expand visibly.
Switch back to "typical" and they shrink. **The cone widening *is*
the GUM-through-ODE result.**

For the canonical case (10⁷ patients, seed 42), the budget on the
right shows that 53.2% of the total variance comes from dosing
variability (Type B, literature prior) and 46.8% from population
variation (Type A, statistical). Combined u<sub>c</sub> = 0.409 mg/L,
expanded U<sub>95</sub> = 0.818 mg/L.

### 2. Compile-time confidence gates (the traffic light)

Above the GUM bar is a traffic light. The evidence band is **per
drug** (the dissertation's contribution #2 generalises to multi-drug):

- **Rapamycin** — green inside ±50% of CL<sub>hep</sub> (Ferron 1997,
  n=24, CV=58%) and [0.7, 1.3] on the stent-release scale (Cordis
  2003 IFU coating-thickness ±30%).
- **Semaglutide** — green inside ±40% of CL<sub>prot</sub> (Overgaard
  2019, n=72, CV=15%) and [0.75, 1.25] on the SC dose multiplier
  (Carlsson 2020, F = 0.89 ± 0.05, k<sub>a</sub> CV = 22%). The band
  is *tighter* than rapamycin's because Overgaard's cohort was larger
  and more uniform — exactly the kind of fact the dissertation argues
  the gate should encode.

Drag the release slider into the red band: the light flips to
**TRIPPED** with a one-sentence explanation naming the correct
clearance route ("hepatic" vs "proteolytic") and the correct
literature anchor. The key claim: a kernel that would extrapolate
past its evidence base *does not compile* — it is rejected by
`kretikos_kaxi_phase_j_gate.sh` with the failure reason attached,
before any patient simulation runs.

### 3. ISO uncertainty budgets (the Hessian heatmap)

The bottom-right panel shows the 2×2 Hessian for the CL<sub>hep</sub>
× f<sub>u,plasma</sub> scan. Diagonal cells are pure curvature;
off-diagonal is parameter interaction. The strip below reports the
1st-order and 2nd-order GUM variance — their difference is the
correction the second-order term adds. For non-linear PBPK this
correction matters; first-order GUM alone can understate uncertainty.

---

## The six tours — three per drug

Click any of the buttons under **Guided tours** in the right panel.
The chip row swaps with the active drug; <kbd>T</kbd> cycles tours
within the current drug, <kbd>D</kbd> cycles drug and resets to the
first tour. Each tour drives the camera and (where annotated) the
patient profile, with a one-sentence narration card across the bottom
of the canvas.

**Rapamycin:**

1. **Cypher → blood → liver** — 30 s. Zooms into the stent at the
   coronary location, follows the eluted drug into the blood pool,
   then to the liver (Kp = 5.4, tissue concentration steady-states
   ~5× plasma).
2. **BBB closeup — Kp = 0.10** — 20 s. Highlights P-gp efflux per
   Lampen 1998. The thin GUM shell is the visual proof that, when
   the partition coefficient is small, the uncertainty envelope is
   also small.
3. **GUM cone widening under CL<sub>hep</sub> variability** — 20 s.
   Focuses on the liver and kidney while switching from typical →
   poor metaboliser → typical. The cones visibly breathe. This is
   the **contribution-#1 money shot** in 20 seconds.

**Semaglutide:**

4. **SC depot → blood → pancreas** — 30 s. Zooms into the abdominal
   depot, follows first-order absorption into blood, then to the
   pancreas (R<sub>total</sub> = 5 nM — the dominant TMDD sink).
5. **GLP-1R occupancy — brain, gut, pancreas** — 24 s. Walks the
   three GLP-1R sites in turn while the occupancy bars on the right
   fill. Narrates the appetite-suppression / gastric-emptying /
   insulinotropic pathway at each site.
6. **Bergman PD — ΔG falls as ΔI rises** — 22 s. Focuses on the
   pancreas while switching mid-tour from typical → slow CL. ΔG
   deepens visibly under increased exposure — this is the
   **GUM-through-PD money shot** for the peptide side.

---

## Try the other drug (Stage G multi-drug A/B)

The same engine drives both drugs:

|  | Rapamycin | Semaglutide |
|---|---|---|
| Release | Cypher stent, Higuchi diffusion | Subcutaneous depot, k<sub>a</sub> = ln 2 / 60 h |
| Distribution | Lipophilic (Kp up to 5.4 in liver) | Vascular-confined peptide (Kp < 1, BBB ≈ 0.05) |
| Clearance | Hepatic CYP3A4, CL = 12.4 L/h | Proteolytic, CL = 0.077 L/h |
| Target (TMDD) | FKBP12 / mTORC1 — liver, heart, gut | GLP-1R — brain, gut, pancreas |
| PD readout | mTORC1 active fraction + neointimal index | Plasma glucose (ΔG) + insulin (ΔI) — Bergman |
| Clinical endpoint | Late lumen loss (restenosis suppression) | Fasting plasma glucose reduction |

When you switch to semaglutide, the entire UI swaps in lockstep:

- **Release-source visual** — Cypher coronary cylinder ↔ lower-abdomen
  SC depot ellipsoid (cyan particles).
- **Receptor-occupancy panel** (right side) — relabels to GLP-1R and
  bars now show brain / gut / pancreas occupancy instead of liver /
  heart / gut.
- **PD readout panel** — switches from "mTORC1 active fraction +
  neointimal index" to live "plasma glucose / insulin" with the
  Bergman minimal-model coupling explained inline.
- **Patient-profile dropdown** — repopulates with semaglutide's
  proteolytic-clearance phenotypes (slow CL / fast CL / lean / obese
  per Overgaard 2019), replacing the rapamycin CYP3A4 bands.
- **Release-scale slider** — becomes the "SC depot dose" slider
  (multiplier on the 1 mg weekly dose); slider colour shifts to cyan.
- **Phase J evidence band** — swaps to Overgaard / Carlsson and
  *tightens* (±40% on clearance instead of ±50%).
- **Snapshot PNG filename** — embeds the active drug id, so multi-drug
  demos remain unambiguous in committee handouts.

The dissertation message: a single permeability-limited **PBPK28 +
TMDD + PD** framework — A-stable fully-coupled Crank-Nicolson on the
27-state arrow matrix — handles a 914 g/mol lipophilic small molecule
**and** a 4114 g/mol peptide without changes to the numerical core.
The framework's *generality* is the unifying contribution. The
nine-case `dissertation_pbpk28_parity_gate.sh` enforces ≤ 1 % RMSE
between the in-browser engine and the Sounio reference solver across
both drugs and all three layers (PBPK, TMDD, PD) on every push to
`main`.

## Try the receptor-saturation slider

Push the release-scale slider toward its upper bound on either drug
and watch the **receptor-occupancy panel** on the right. Two things
happen visibly:

- **At low exposure**, occupancy bars are short and free receptor
  (R<sub>free</sub>) dominates — TMDD is a small perturbation on the
  PK.
- **As exposure rises**, the bars approach saturation; the
  drug-receptor complex DR becomes the dominant pool, and any
  further dose increment buys progressively less PD effect. This is
  the **target-mediated non-linearity** that classical 2-compartment
  PK models cannot represent — and it is precisely what the
  dissertation's Mager 2004 TMDD layer adds to the kernel.

Click the pancreas (for semaglutide) or the heart (for rapamycin):
the organ modal pops out the Mager TMDD ODE in KaTeX, plus the live
R<sub>free</sub>, DR, and occupancy % at the current frame.

## Try the PD readout

The **PD readout panel** (bottom-right, under the GUM bar) is the
final downstream consequence of everything upstream. On semaglutide,
switch the patient profile from typical → slow CL: the GLP-1R
occupancy on the pancreas rises, drives insulin secretion, and ΔG
deepens below zero with a delay set by the Bergman p<sub>2</sub>
constant. On rapamycin, run the 30-day Cypher tour to the end and
watch the neointimal index N(t = 90 d) — the mapped clinical
endpoint for late-lumen-loss / restenosis suppression.

This is the **GUM-through-PD** result: parameter uncertainty
propagates not only through the PBPK ODE but all the way to the
clinical readout, end-to-end, with the uncertainty cone widening on
each layer's contribution.

---

## Controls reference

| Control | What it does |
|---|---|
| **Drug selector** (top of canvas) | Toggles Rapamycin ⇄ Semaglutide. Resets patient profile and tour selection. |
| **Patient profile** (bottom-centre dropdown) | Drug-aware: rapamycin = typical / low CL / high CL / lean / obese (CYP3A4 envelope); semaglutide = typical / slow CL / fast CL / lean / obese (Overgaard envelope). |
| **Release-scale slider** | Drug-aware: rapamycin = Higuchi K<sub>H</sub> multiplier (coating-thickness CV per Cordis 2003 IFU); semaglutide = SC dose multiplier on the 1 mg weekly dose. |
| **Time slider** | Scrubs the simulation between t = 0.1 h and t = 30 d. Toggle log/linear scale below the slider. |
| **Play / Pause / Restart / Speed** | Wall-clock playback rate; 1× = 60 s wall time spans the entire 30-day window. |
| **Click an organ** | Opens a modal with the mass-balance ODE (LaTeX), current concentration in ng/L, mass in pg, Kp / Q — and, if the organ is in the active drug's TMDD or PD set, the Mager TMDD ODE or per-drug PD ODE plus their live state. |
| **📸 Snapshot PNG** | Downloads the current frame as a timestamped PNG; filename embeds the active drug. |

### Keyboard

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `R` | Restart at t = 0 |
| `S` | Snapshot PNG |
| `D` | Cycle drug (rapamycin ⇄ semaglutide) |
| `T` | Cycle tours within the active drug |
| `Esc` | Stop the active tour |

---

## Underlying numerics

Everything visible on the page is generated from the same code that
emits the dissertation's submitted PDF figures. The in-browser
integrator is a **fully-coupled Crank-Nicolson step on the 27-state
PBPK28 arrow matrix** (per-organ Schur complement, O(N) per step,
A-stable, 2nd-order) — *not* RK4, after a Stage G-α audit found that
explicit-RK4 on the brain compartment is numerically unstable at the
default time step (mass growth ~172× over 30 d). The CN step is
**parity-locked** to the Sounio reference solver via two CI gates on
every push to `main`:

- `scripts/ci/dissertation_frontend_parity_gate.sh` — the original
  Stage C well-stirred PBPK14 anchor, kept as a regression test.
- `scripts/ci/dissertation_pbpk28_parity_gate.sh` — the Stage G
  **nine-case** gate covering: PBPK28 Node↔Sounio (rapamycin),
  PBPK28-degenerate ↔ 1-state QSS analytical (reporting), PBPK28
  literature ↔ PBPK14 well-stirred (reporting — feeds the Type B
  model-form contribution into the §5 GUM budget per JCGM 100:2008
  §4.3), total-mass monotonic decay, rapamycin TMDD parity (R<sub>free</sub>,
  DR) at liver / heart / gut, rapamycin PD parity (A, N) at heart,
  semaglutide PBPK28 Node↔Sounio, semaglutide GLP-1R TMDD parity at
  brain / gut / pancreas, semaglutide glucose-insulin PD parity
  (ΔG, ΔI) at pancreas.

If either gate fails CI, the viewer is not promoted to production.

The CSVs the side panels read (`/dissertation/gum_budget.csv`,
`/dissertation/hessian_budget.csv`) are bit-identical copies of
`benchmarks/pbpk/gum_budget.csv` and `benchmarks/pbpk/hessian_budget.csv`
in the repository — the same files that are referenced from the
ISO 17025 dossier (§5, §6) and the dissertation manuscript.

---

## If anything looks wrong

The viewer was built by Demetrios Chiuratto Agourakis as part of the
Master's dissertation. Issues / suggestions are welcome at:

- GitHub: <https://github.com/Sounio-lang/sounio/issues>
- Direct: demetrios@agourakis.med.br
