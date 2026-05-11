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

**Time needed:** ~5 minutes for the three guided tours, plus as long
as you want to explore.

---

## What you are looking at

The page opens to a 3D anatomical scene of a 70 kg adult. The 14
spheres are the compartments of the rapamycin PBPK model — heart,
lung, liver, kidney, brain, gut, muscle, adipose, skin, bone, spleen,
pancreas, blood, "other tissues." The small silver cylinder in the
chest is the Cypher drug-eluting stent at its coronary location;
yellow particles streaming upward from it are rapamycin being eluted
into the bloodstream at the Higuchi rate
*dQ/dt = K<sub>H</sub> / (2√t)* (140 µg total, ~80% over 30 days,
per the Cordis 2003 IFU).

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

Above the GUM bar is a traffic light. It will be **green** when the
current patient profile sits inside the documented evidence band
(±50% of CL<sub>hep</sub> per Ferron 1997, ±30% of stent release scale
per Cordis 2003 IFU). Drag the stent release slider to 0.5×: the
light should flip to **TRIPPED** with a one-sentence explanation. The
key claim: a kernel that would extrapolate past its evidence base
*does not compile* — it is rejected by `kretikos_kaxi_phase_j_gate.sh`
with the failure reason attached, before any patient simulation runs.

### 3. ISO uncertainty budgets (the Hessian heatmap)

The bottom-right panel shows the 2×2 Hessian for the CL<sub>hep</sub>
× f<sub>u,plasma</sub> scan. Diagonal cells are pure curvature;
off-diagonal is parameter interaction. The strip below reports the
1st-order and 2nd-order GUM variance — their difference is the
correction the second-order term adds. For non-linear PBPK this
correction matters; first-order GUM alone can understate uncertainty.

---

## The three tours

Click any of the buttons under **Guided tours** in the right panel.
Each tour drives the camera and (for tour 3) the patient profile, with
a one-sentence narration card across the bottom of the canvas.

1. **Cypher → blood → liver** — 30 s. Zooms into the stent at the
   coronary location, follows the eluted drug into the blood pool,
   then to the liver (Kp = 5.4, tissue concentration steady-states
   ~5× plasma).
2. **BBB closeup** — 20 s. Highlights the brain's Kp = 0.10 (P-gp
   efflux per Lampen 1998). The thin GUM shell is the visual proof
   that, when the partition coefficient is small, the uncertainty
   envelope is also small.
3. **GUM cone widening under CL<sub>hep</sub> variability** — 20 s.
   Focuses on the liver and kidney while switching from typical →
   poor metaboliser → typical. The cones visibly breathe. This is
   the **contribution-#1 money shot** in 20 seconds.

---

## Try the other drug (G-δ, multi-drug A/B toggle)

A drug-selector chip row sits at the top of the canvas — or press
<kbd>D</kbd> to cycle. The default is **Rapamycin** (Cypher coronary stent,
the original dissertation drug). The alternative is **Semaglutide** —
the GLP-1 receptor agonist used as Ozempic / Wegovy.

The same engine drives both drugs:

|  | Rapamycin | Semaglutide |
|---|---|---|
| Release | Cypher stent, Higuchi diffusion | Subcutaneous depot, k<sub>a</sub> = ln 2 / 60 h |
| Distribution | Lipophilic (Kp up to 5.4 in liver) | Vascular-confined peptide (Kp < 1, BBB ≈ 0.05) |
| Clearance | Hepatic CYP3A4, CL = 12.4 L/h | Proteolytic, CL = 0.077 L/h (FcRn recycling) |
| Target | FKBP12 / mTORC1 in coronary smooth muscle | GLP-1R on pancreatic β-cells |
| Clinical endpoint | Late lumen loss (restenosis suppression) | Fasting plasma glucose reduction |

When you switch to semaglutide:
- The release source visual swaps from the chest-mounted Cypher
  cylinder to a subcutaneous depot ellipsoid (planned G-ε-2).
- The patient sliders reset to semaglutide defaults — the existing
  CL<sub>hep</sub> slider is calibrated for rapamycin and would corrupt
  the semaglutide PK if carried over.
- Pancreas, gut, and brain become the high-occupancy organs (instead
  of liver / heart / gut for rapamycin). All currently rendered as
  organ-colour intensity; explicit receptor-occupancy and PD readout
  panels follow in G-ε-2.

The dissertation message: a single permeability-limited PBPK28 +
TMDD + PD framework, A-stable Crank-Nicolson integrator, handles a
914 g/mol lipophilic small molecule **and** a 4114 g/mol peptide
without changes to the numerical core. The framework's *generality*
is the unifying contribution.

---

## Controls reference

| Control | What it does |
|---|---|
| **Patient profile** (bottom-centre dropdown) | CL<sub>hep</sub> &/or volume of distribution. Switches between "typical" (population mean), "low CL" / "high CL" (CYP3A4 poor / ultrarapid), "lean" / "obese" (BMI). |
| **Stent release scale** | Multiplies Higuchi K<sub>H</sub>. Reflects coating-thickness and solubility CV (Cordis 2003 IFU). |
| **Time slider** | Scrubs the simulation between t = 0.1 h and t = 30 d. Toggle log/linear scale below the slider. |
| **Play / Pause / Restart / Speed** | Wall-clock playback rate; 1× = 60 s wall time spans the entire 30-day window. |
| **Click an organ** | Opens a modal with the mass-balance ODE for that compartment (LaTeX-rendered), the current concentration in ng/L, mass in pg, and Kp / Q. |
| **📸 Snapshot PNG** | Downloads the current frame as a timestamped PNG. |

### Keyboard

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `R` | Restart at t = 0 |
| `S` | Snapshot PNG |
| `Esc` | Stop the active tour |

---

## Underlying numerics

Everything visible on the page is generated from the same code that
emits the dissertation's submitted PDF figures. The in-browser RK4
integrator is **parity-locked** to the Sounio `tsit5_pbpk14` reference
solver via `scripts/ci/dissertation_frontend_parity_gate.sh` — the
two are required to agree at < 1% RMSE per compartment on every
push to `main`. If they ever diverge, the gate fails CI and the
viewer is not promoted to production.

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
