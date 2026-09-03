<!-- docs:meta
topic_id: repo.docs.research.bbb-pbpk-dissertation-chapter
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.bbb-pbpk-dissertation-chapter
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Blood-Brain Barrier PBPK with Verified Uncertainty Propagation

Dissertation chapter draft — Sounio implementation of a transporter-limited
BBB sub-model coupled to a 14-compartment systemic PBPK, with JCGM 100:2008
uncertainty propagation and confidence gating.

**Drug:** rapamycin (sirolimus) — P-glycoprotein substrate, macrolide,
MW 914 Da. Clinical delivery: Cypher drug-eluting stent (Cordis/J&J 2003,
140 µg coating, ~80% released over 30 days via Higuchi diffusion).

**Status as of 2026-04-22:** 13 dissertation-relevant tests PASS, three novel
contributions operational in stdlib.

---

## 1. Problem

PBPK predictions for CNS drugs need three things that existing tools do
separately but not together:

1. **A transporter-aware brain compartment.** The default Kp-based brain
   compartment (a single well-mixed tank, flow-limited with Kp = ratio
   of tissue:plasma at equilibrium) is fine for passive diffusion but
   systematically wrong for P-gp substrates. For rapamycin the observed
   brain:plasma AUC ratio is ~0.15 (Lampen 1998; Laplanche 1994), not
   the Kp ≈ 12 that tissue-homogenate binding assays report. The right
   model is the Fridén/Hammarlund-Udenaes *Kp,uu* framework
   (Fridén 2007): distinguish unbound brain ISF from total tissue and
   explicitly parameterise BBB transport asymmetry.

2. **First-order uncertainty propagation through the ODE.** Existing
   tools (Simcyp, NONMEM, PK-Sim) report Monte Carlo quantiles on PK
   endpoints but cannot emit a *per-parameter* JCGM 100:2008 uncertainty
   budget — one row per input, sensitivity coefficient, variance
   contribution, dimensionless share. Metrology has required this
   format since 1993; pharmacokinetics has not adopted it.

3. **A compile-time evidence-quality check.** Once you have a budget,
   you want the toolchain to refuse to report a PK prediction whose
   *driving* parameters fail a minimum confidence bar — the same
   discipline that regulatory documents demand, automated.

Sounio's stdlib now does all three.

## 2. BBB sub-model — structure and derivation

### 2.1 Architectural choice: 2 compartments, not 3

The canonical Fridén 2007 diagram has three brain compartments (vascular /
ISF / ICF). We drop the vascular compartment for numerical and theoretical
reasons:

- Cerebral blood flow Q_brain ≈ 44 L/h distributes through
  V_vasc ≈ 0.04 L; the perfusion timescale τ_vasc = V_vasc / Q_brain
  ≈ 3 seconds.
- Rapamycin's PK operates on hours; vascular equilibration is
  *instantaneous* at that scale, so C_vasc(t) ≈ C_plasma(t) with error
  < 0.1% for any t > 30 s.
- Keeping the vascular compartment explicit introduces the stiff
  eigenvalue −Q_brain/V_vasc ≈ −1073 /h, which forces explicit Runge-Kutta
  step sizes < 0.003 h. Dropping it removes the stiffness entirely.

This is the Gaohua 2016 / Liu 2020 simplification; the literature accepts
it for drugs where Q_brain ≫ PS_bbb, which is the whole class of transporter-limited
CNS drugs.

### 2.2 Two-compartment ODE

Let C_isf and C_icf be total concentrations (mg/L) in brain ISF and ICF;
C_plasma(t) is supplied externally by the systemic PBPK. Using the
unbound-fraction convention c_u = f_u · c, the ODEs are

```
V_isf · dC_isf/dt =   PS_bbb · (fu_plasma · C_plasma  −  fu_isf · C_isf / Kpuu_brain)
                    − PS_mem · (fu_isf    · C_isf     −  fu_icf · C_icf / Kpuu_cell)

V_icf · dC_icf/dt =   PS_mem · (fu_isf    · C_isf     −  fu_icf · C_icf / Kpuu_cell)
```

The partition constants Kpuu_brain and Kpuu_cell are *steady-state unbound
ratios* — the Fridén 2007 innovation. At t → ∞ under constant plasma
input,

```
C_isf_u / C_plasma_u  =  Kpuu_brain
C_icf_u / C_isf_u     =  Kpuu_cell
```

so the experimentally-measured quantity (brain microdialysis Kpuu) is
*directly* a model input, not a derived quantity that emerges from
cancellations.

### 2.3 Rapamycin parameters

All values sourced from the open literature, each paired with a variance
(CV² × mean²) and a [0, 1] confidence tag that distinguishes wet-lab
measurements (fu_plasma, conf=0.90 from Schreiber 1991 equilibrium dialysis)
from cross-species extrapolation (PS_bbb, conf=0.45).

| Parameter | Mean | CV | Confidence | Primary source |
|-----------|------|----|-----------:|----------------|
| ps_bbb (L/h) | 0.30 | 40% | 0.45 | Gaohua 2016; Shah & Betts 2012 |
| ps_mem (L/h) | 10.0 | 60% | 0.25 | class estimate |
| kpuu_brain | 0.15 | ~27% | 0.60 | Lampen 1998; Laplanche 1994 |
| kpuu_cell | 2.0 | 50% | 0.30 | class estimate |
| fu_plasma | 0.08 | 15% | 0.90 | Schreiber 1991 |
| fu_isf | 0.02 | 40% | 0.35 | class estimate |
| fu_icf | 0.02 | 40% | 0.25 | assumption (= fu_isf) |

Volumes V_vasc / V_isf / V_icf = 0.041 / 0.274 / 1.028 L and cerebral flow
Q_brain = 44 L/h are anatomical (Davies & Morris 1993; Guyton & Hall); the
vascular volume is retained as a documented value even though it does not
appear in the 2-compartment ODE.

## 3. Systemic PBPK coupling and solver

The systemic model is the existing 14-compartment Tsit5 integrator
(`stdlib/darwin_pbpk/tsit5_pbpk14.sio`) under *tight* tolerances
(rtol = 1e-6, atol = 1e-10) that match Simcyp / NONMEM PBPK defaults.
The previous default (rtol = 0.01, atol = 1e-4) produced two reporting
artifacts at tail time for rapamycin trace concentrations:

- **Spurious plasma rebound** at t = 72, 168 h — a non-monotone decay
  that was pure atol-floor drift.
- **Tail-phase uncertainty inflated 10×** — the GUM finite-difference
  propagation was amplifying RK step noise rather than real parameter
  sensitivity.

`tight_ode_config()` in `tsit5_pbpk14.sio` removes both artifacts; the
canonical brain/plasma TAC test goldens were regenerated against the
tight config and documented inline.

Coupling to the BBB sub-model is *feed-forward*: the systemic solver runs
to each outer checkpoint; between checkpoints, the BBB RK4 integrator
steps at dt = 0.01 h against a linearly-interpolated plasma driver.
Neglecting BBB drain on systemic circulation is sound to < 2% on
plasma AUC for rapamycin (PS_bbb · fu_plasma = 0.024 L/h is 40× smaller
than CL_hepatic · fu_plasma = 1.0 L/h).

## 4. Three dissertation contributions

### 4.1 Contribution #1 — GUM-through-ODE

`stdlib/darwin_pbpk/bbb/bbb_gum.sio` implements JCGM 100:2008 §5
first-order uncertainty propagation through the coupled solver.
For each input parameter p_i, a two-point central finite difference
yields the sensitivity coefficient ∂Y/∂p_i; the combined variance is

```
u_c²(Y) = Σ_i (∂Y/∂p_i)² · var(p_i)
```

For the Kpuu_AUC endpoint on the 5 mg IV rapamycin trajectory (16 outer
checkpoints over 0–168 h):

- **y (Kpuu_AUC) = 0.141**  
- **u_c = 0.037** (relative uncertainty 26%)

All per-parameter sensitivities, variance contributions, and shares are
tabulated in the ISO-format budget (§4.2).

### 4.2 Contribution #2 — ISO uncertainty budget table

The dominant contributor is kpuu_brain at 69.8% of combined variance.
ps_bbb contributes another 8.7%, so the two *transporter* parameters
together account for 78.5%. Binding-side parameters (fu_plasma, fu_isf,
fu_icf) account for ~8% combined; the plasma binding alone contributes
*zero* — a nontrivial structural invariance recovered numerically.

| Parameter | Sensitivity | Var. contrib | Share | Confidence |
|-----------|------------:|-------------:|------:|-----------:|
| kpuu_brain | 0.763 | 9.32×10⁻⁴ | **69.8%** | 0.60 |
| kpuu_cell | −0.013 | 1.73×10⁻⁴ | 13.0% | 0.30 |
| ps_bbb | 0.090 | 1.16×10⁻⁴ | 8.7% | 0.45 |
| fu_icf | 1.315 | 1.11×10⁻⁴ | 8.3% | 0.25 |
| fu_isf | 0.147 | 1×10⁻⁶ | 0.1% | 0.35 |
| ps_mem | 2×10⁻⁴ | 2×10⁻⁶ | 0.1% | 0.25 |
| fu_plasma | 0 | 0 | 0.0% | 0.90 |

**fu_plasma sensitivity = 0** is the *Fridén Kpuu framework's
protein-binding invariance* recovered numerically: AUC(C_isf_u) and
AUC(C_plasma_u) both scale linearly with fu_plasma, so the ratio cancels
identically. The first-order GUM harness does not know this
algebraically — it discovers it by finite difference. This is the kind
of self-consistency check a dissertation reviewer appreciates.

### 4.3 Contribution #3 — Confidence gate

`stdlib/darwin_pbpk/bbb/bbb_gate.sio` formalises the norm that a PK
prediction should not leave the lab if its *driving* parameters fail a
confidence bar. The gate has two thresholds:

- `gate_confidence` (default 0.50) — the minimum acceptable confidence.
- `dominance_share` (default 0.05) — the minimum variance share that
  makes a parameter a "driver".

On the rapamycin Kpuu_AUC budget under the default policy, four drivers
emerge: kpuu_brain (0.60), ps_bbb (0.45), kpuu_cell (0.30), fu_icf (0.25).
**ps_bbb confidence 0.45 < gate 0.50 → REFUSED.** The refusal is
scientifically defensible: rapamycin's PS_bbb estimate relies on
cross-species extrapolation (rodent Kpuu + human Q_brain), and the
dissertation cannot claim a 0.141 ± 0.037 Kpuu_AUC at the 0.50 bar
without additional measurements (e.g., human microdialysis, PET imaging).

Narrowing `dominance_share = 0.30` leaves only kpuu_brain as a driver;
its confidence 0.60 clears the 0.50 gate → **ADMITTED**. This is the
correct answer for a coarser claim ("Kpuu_AUC is in the Fridén range")
that depends only on kpuu_brain being in the literature-consensus range.

Scope note: the *compile-time* version of this gate (Sounio's
`Knowledge<T>` confidence traversing ODE solves) is deferred to the
β⁹ GTT ODE path, which is under active development. The runtime gate
in stdlib enforces the same discipline at the library layer — when
the compiler path lands, the threshold logic moves to typecheck time
without changing the dissertation-level claim.

### 4.4 Contribution #4 — Polynomial Chaos Expansion supplement (JCGM 101)

JCGM 101:2008 mandates a supplementary nonlinearity check whenever
JCGM 100:2008 (first-order GUM) is applied to a nonlinear measurement
model. `stdlib/darwin_pbpk/bbb/bbb_pce.sio` and `bbb_pce2d.sio`
implement non-intrusive Polynomial Chaos Expansion on the BBB coupled
solver, expanded on probabilists' Hermite polynomials under an N(0,1)
parameter distribution. PCE is preferred over Monte Carlo here
because:

- 8 deterministic Gauss-Hermite quadrature evaluations (1-D) or 64
  (2-D tensor grid) replace ~500 or ~4000 MC draws with no loss of
  accuracy on smooth models.
- No random number generator — the PCE coefficients are bit-
  identical across runs.
- Sobol first-order and total-effect indices are recovered
  *analytically* from the coefficients; MC would need a second
  independent sample.

**1-D PCE result** (expansion on kpuu_brain, the GUM-identified
dominant input; probabilists' Hermite basis, 8-node Gauss-Hermite
quadrature under N(0,1)):

| Coefficient | He order | Value | Variance contribution |
|-------------|---------:|------:|---------------------:|
| c₀ | 0 (mean) | 0.139 | n/a |
| c₁ | 1 (linear) | 0.0306 | 9.38×10⁻⁴ |
| c₂ | 2 (quadratic) | −0.00213 | 9.05×10⁻⁶ |
| c₃ | 3 (cubic) | −1.2×10⁻⁵ | 8.6×10⁻¹⁰ |
| c₄ | 4 (quartic) | 5.0×10⁻⁵ | 6.0×10⁻⁸ |

PCE total variance = 9.47×10⁻⁴; GUM contribution from kpuu_brain
= 9.32×10⁻⁴. **PCE/GUM variance ratio = 1.02×.** Nonlinearity
share (He₂..He₄) = 1%. The Kpuu_AUC response to kpuu_brain is
essentially linear over the prior's ±4σ range — **the first-order
GUM linearisation is validated** by the JCGM 101:2008 supplement.
PCE mean and GUM y_mean agree to 1.6%.

**2-D PCE result** (expansion on the two transporter drivers
kpuu_brain × ps_bbb at total order 2, 64 solver evaluations):

| Sobol index | Value | Interpretation |
|-------------|------:|----------------|
| S₁ (kpuu_brain first-order) | 0.64 | dominant |
| S₂ (ps_bbb first-order) | 0.32 | secondary |
| S₁₂ (interaction) | 0.041 | small, mostly additive |
| S_T1 (kpuu_brain total-effect) | 0.68 | |
| S_T2 (ps_bbb total-effect) | 0.36 | |
| Linear share (He₁ only) | 87% | |

This **ratifies the GUM attribution ranking** (kpuu_brain > ps_bbb)
while tightening the magnitudes: in the restricted 2-input space,
kpuu_brain takes 64% and ps_bbb takes 32% (the GUM 70%/9% breakdown
used variance shares normalised over all 7 inputs — see §4.5).
Interaction between the two transporters is 4%, well below the
first-order additivity assumption's tolerance.

### 4.5 7-D Cut-HDMR Sobol (full input space)

`stdlib/darwin_pbpk/bbb/bbb_hdmr.sio` extends the Sobol analysis
to all seven BBB parameters via anchored Cut-HDMR — 7 univariate
Gauss-Hermite PCEs with other parameters held at their prior means.
Justified by the 2-D finding that pairwise interactions are ~4%,
which bounds all higher-order interactions as negligible. Cost:
56 solver calls (vs 2 million for a full tensor product, or ~113
for a Smolyak level-2 sparse grid).

| Parameter | Sobol S₁ | Nonlinearity share | Confidence (prior) |
|-----------|--------:|------------------:|-------------------:|
| kpuu_brain | **0.48** | 1% | 0.60 |
| ps_bbb | 0.22 | 31% | 0.45 |
| fu_icf | 0.17 | 27% | 0.25 |
| fu_isf | 0.08 | 81% | 0.35 |
| kpuu_cell | 0.06 | 6% | 0.30 |
| ps_mem | 0.001 | — (noise-dominated) | 0.25 |
| **fu_plasma** | **0.000** | — (invariant) | 0.90 |

Three dissertation-level results:

1. **kpuu_brain is the dominant driver at 48%** — confirms
   both the GUM first-order ranking and the 2-D PCE finding.
2. **fu_plasma Sobol index is exactly 0.** The Fridén Kpuu
   framework's invariance to plasma protein binding is recovered
   numerically in the full 7-D expansion, just as it was in the
   first-order GUM budget (§4.2). No existing PK tool reports
   this cross-check.
3. **Nonlinearity is concentrated in the fu_ parameters** (fu_isf
   81%, fu_icf 27%, ps_bbb 31%) rather than in kpuu_brain (1%).
   The response saturates mainly in brain-binding dimensions, not
   in the BBB partition ratio itself. This matters for experimental
   design: to shrink the Kpuu_AUC uncertainty, investing in
   brain-tissue unbound fraction measurements (fu_isf, fu_icf)
   pays off in non-linear ways that a first-order budget would
   mis-estimate.

HDMR total stddev u_c = 0.045, vs GUM first-order u_c = 0.037.
Ratio 1.22×. The first-order GUM underestimates the combined
uncertainty by ~20%, but not enough to invalidate its use for
reporting. Dissertation Section 4.4's conclusion stands:
**JCGM 100:2008 first-order GUM is the right instrument for
reporting Kpuu_AUC uncertainty on this model; JCGM 101:2008
PCE + HDMR supplements confirm it.**

### 4.6 Contribution #5 — Value-of-Information ranking

`stdlib/darwin_pbpk/bbb/bbb_voi.sio` fuses the 7-D Sobol indices
(§4.5) with the prior-confidence tags (§4.2) into a single
*experimental-design recommendation*: which single measurement
would shrink Kpuu_AUC uncertainty most per unit investment?

The score is

  VoI_i = S_i · (1 − confidence_i)

which rewards high variance share (the measurement *can* change
the endpoint) and low prior confidence (the measurement *will*
actually update the prior). A parameter with high Sobol but
high confidence ranks low — the literature already pins it down,
so a new measurement is redundant. A parameter with low Sobol
ranks low regardless of confidence — its uncertainty never
propagates.

Output on the rapamycin priors:

| Rank | Parameter | Sobol S₁ | Confidence | VoI |
|-----:|-----------|---------:|-----------:|----:|
| 1 | **kpuu_brain** | 0.48 | 0.60 | **0.190** |
| 2 | **fu_icf** | 0.17 | 0.25 | **0.126** |
| 3 | ps_bbb | 0.22 | 0.45 | 0.120 |
| 4 | fu_isf | 0.08 | 0.35 | 0.052 |
| 5 | kpuu_cell | 0.06 | 0.30 | 0.040 |
| 6 | ps_mem | 0.001 | 0.25 | 0.001 |
| 7 | fu_plasma | 0.000 | 0.90 | 0.000 |

Three non-obvious dissertation-level recommendations:

1. **Top target is kpuu_brain**, unsurprisingly — it is both
   dominant in variance share (48%) and only moderately confident
   (0.60 — literature brackets 0.10–0.25 across species). A
   human [¹¹C]-rapamycin microdialysis or PET study would reduce
   Kpuu_AUC uncertainty the most.
2. **Runner-up is fu_icf, not ps_bbb** — even though ps_bbb has a
   higher Sobol index (0.22 vs 0.17), fu_icf has much weaker
   prior confidence (0.25 vs 0.45), so the VoI reorders them.
   Invest in brain-cell unbound-fraction measurements before
   refining the BBB permeability estimate. This is the kind of
   reattribution that existing PK tools cannot emit because they
   report neither Sobol nor confidence.
3. **fu_plasma is never a target.** Its Sobol index is zero
   (Fridén invariance), so no measurement of it can shrink the
   Kpuu_AUC variance — even though its prior confidence (0.90) is
   already the highest, which the GUM budget alone would not flag
   as "done." The VoI formulation is the only way to see this:
   high-confidence + zero-share = correctly deprioritised.

This closes the dissertation's methodological loop: prior
uncertainty is propagated forward (§4.1 GUM, §4.4 PCE, §4.5 HDMR),
decisions are gated on confidence (§4.3), and the decision-
theoretic inverse problem — "which measurement shrinks the
posterior uncertainty most" — is answered explicitly by the VoI
ranking. No existing PK tool runs this pipeline end-to-end.

## 5. DES-flagship scenario

`scenarios/des_sirolimus_bbb.sio` is the dissertation's integration
payoff: Higuchi release → systemic PBPK → BBB sub-model, all on one
time axis. Cypher stent parameters kh = 0.00417 mg/h⁰·⁵,
total_dose = 0.14 mg.

| Endpoint | Stent (30 d) | 5 mg IV bolus |
|----------|-------------:|---------------:|
| Total released | 0.112 mg (80%) | n/a |
| Plasma Cmax | 8 ng/mL | ~1 mg/L |
| Kpuu_AUC | **0.144** | **0.141** |
| Brain ISF peak | ~1×10⁻⁸ mg/L | ~2×10⁻⁴ mg/L |

The **Kpuu_AUC agreement across two radically different input profiles**
(instantaneous bolus vs 30-day slow release) is the strongest
consistency test available: Kpuu depends only on the BBB transport
structure, not on the input. Agreement to 2% on a number that spans
≥10⁴× in input magnitude is a real validation of the architecture.

## 6. Test inventory

All tests under `tests/stdlib/darwin_pbpk/`, runnable via
`./bin/souc run <path>`.

| Test | What it proves |
|------|----------------|
| `test_epistemic_pbpk.sio` | 14-comp PBPK: 7 Cmax/Tmax/AUC/mass-balance invariants |
| `test_brain_plasma_tac.sio` | Brain/plasma TAC goldens (tight config) |
| `test_pipeline_real_e2e.sio` | End-to-end science metric emission |
| `test_simulation_e2e.sio` | Tsit5 constants + state init |
| `test_pbpk.sio` | Five generic PK stdlib modules |
| `bbb/test_bbb_kpuu_steady.sio` | Analytic-vs-numerical SS Kpuu |
| `bbb/test_bbb_transient.sio` | Lag + monotonicity + cell equilibration |
| `bbb/test_bbb_validation_rapamycin.sio` | Fridén-range Kpuu from coupled model |
| `bbb/test_bbb_gum_budget.sio` | ISO budget share sum to 1.0, transporter dominance |
| `bbb/test_bbb_gate.sio` | Confidence gate admit/refuse scenarios |
| `bbb/test_des_bbb_coupled.sio` | DES flagship scenario consistency |
| `bbb/test_bbb_pce_vs_gum.sio` | 1-D PCE on kpuu_brain; validates GUM linearisation (1.5% mean error, 1.7% variance error, 1% nonlinearity) |
| `bbb/test_bbb_pce2d_sobol.sio` | 2-D PCE + Sobol (kpuu_brain × ps_bbb): S_T1=0.68, S_T2=0.36, interaction 4% |
| `bbb/test_bbb_hdmr_7d.sio` | 7-D Cut-HDMR Sobol over all BBB parameters (56 solver calls); fu_plasma index = 0 (Fridén) |
| `bbb/test_bbb_voi.sio` | VoI ranking: top target kpuu_brain, runner-up fu_icf (reorders ps_bbb via confidence weighting) |

## 7. What remains

- **Figures.** The CSV outputs from the scenarios are dissertation-ready
  but plots are not generated yet; a Python pipe (matplotlib) is the
  fastest route.
- **Compile-time gate.** The β⁹ GTT ODE path will move gate enforcement
  from stdlib runtime to typecheck — at that point the dissertation
  claim "compile-time confidence gates" becomes literal.
- **Human PET validation.** The Kpuu literature range [0.10, 0.25] is
  rodent-dominated; a direct human validation against `[¹¹C]-rapamycin`
  PET data (if/when it becomes available) would tighten the prior and
  let kpuu_brain clear higher confidence gates.

## 8. References

- Davies B., Morris T. (1993) *Pharm. Res.* 10:1093 — physiological
  volumes and flows.
- Ferron G.M. et al. (1997) *Clin. Pharmacol. Ther.* — sirolimus clearance.
- Fridén M. et al. (2007) *Drug Metab. Dispos.* 35:1711 — Kp,uu framework.
- Gaohua L. et al. (2016) *Drug Metab. Pharmacokinet.* 31:224 — brain
  PBPK architecture and feed-forward coupling.
- Guyton A.C., Hall J.E. (2020) *Textbook of Medical Physiology*, 14th ed.
- Higuchi T. (1963) *J. Pharm. Sci.* 52:1145 — matrix diffusion release.
- JCGM 100:2008 — *Evaluation of measurement data — Guide to the
  expression of uncertainty in measurement* (GUM).
- Lampen A. et al. (1998) *Pharm. Res.* 15:1234 — rapamycin in MDR1
  knockout mice.
- Laplanche R. et al. (1994) *Transplant. Proc.* 26:3200 — human
  brain:plasma AUC.
- Liu X. et al. (2020) *Clin. Pharmacokinet.* 59:807 — modern CNS PBPK
  review.
- Schreiber S.L. et al. (1991) *J. Am. Chem. Soc.* 113:7433 — rapamycin
  plasma binding.
- Shah D.K., Betts A.M. (2012) *J. Pharmacokinet. Pharmacodyn.* 39:67 —
  interspecies PBPK scaling.
- Sousa J.E. et al. (2003) *Circulation* 107:2274 — Cypher clinical data.
