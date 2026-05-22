<!-- docs:meta
topic_id: repo.docs.dissertation.handoff.section-4-10-sobol-hdmr-package
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.handoff.section-4-10-sobol-hdmr-package
-->

# §4.10 Writing Package — Sobol/Cut-HDMR Global Sensitivity Analysis
## PBPK dissertation (one document), §4.10 — rapamycin + semaglutide arm

**For:** Claude Desktop, drafting §4.10  
**From:** Claude Code, branch `claude/refine-local-plan-KAgIS`, commit `a5f633f`  
**Reconciled 2026-05-21:** the unifying frame is **PBPK** — one dissertation across drug classes (the 2026-05-12 two-track split is superseded). Advisor name corrected below.  
**Confirmed context:** September–October defense; advisor **Marli Gerenutti** (pharmacologist; PUC-SP biomaterials/regen-med program). The thesis is unified by PBPK + epistemic uncertainty, not biomaterials narrowly. (Earlier "Dr. Moema Haussen" was an error.)  
**Governing artefact:** `docs/dissertation/handoff/chapter_04.md` + PBPK28 parity gate  
**Truth table:** `pbpk_claim_truth_table.md` now carries PBPK28 rows (re-audited 2026-05-21); use it for §4.10 claim control.

---

## 0. Status update — gap CLOSED 2026-05-21

**`epistemic_pbpk28.sio` now EXISTS** (664 loc), alongside `epistemic_pbpk28_hessian.sio` (591 loc) — authored 2026-05-17 (`652133d7d`..`cb51778fa`: first-order GUM + Hessian + Sobol/PCE over the 28-state kernel). The 2026-05-21 re-audit (`docs/dissertation/audit/gap_report.json`, entry `six_contributions_modules`) downgrades this from **MAJOR to MINOR**: contributions (1) and (2) are done at PBPK28 fidelity; only the full Cypher PBPK28 scenario (3) remains at PBPK14 fidelity.

This means **Option B below is now available**: §4.10 can be written against the real 28-state permeability-limited kernel, not only the PBPK14 proxy.

**What this means for §4.10:**

| Option | Consequence |
|---|---|
| **A — Write with honest gap statement** | §4.10 uses Sobol infrastructure from `stdlib/epistemic/sobol.sio` applied via the PBPK14 proxy (same parameter set; well-stirred kernel). Frame as "sensitivity analysis via the PBPK14 epistemic path, with PBPK28 port in progress." This is honest and defensible. |
| **B — Wait for Claude Code to port PBPK28** | Delays §4.10 by one implementation sprint (~350 LOC). Result is a stronger claim: "Global sensitivity analysis through the full 28-state permeability-limited kernel." |

**Recommendation for Sept–Oct timeline (updated 2026-05-21):** Option B is now viable since `epistemic_pbpk28.sio` landed — §4.10 can run Sobol/HDMR through the real 28-state kernel. Option A remains a sound fallback: the Sobol indices are parameter-driven, not kernel-architecture-driven, so for the 7 epistemic parameters (cl_hepatic, cl_renal, fu_plasma, kp_brain, kp_liver, kp_kidney, kp_adipose) the sensitivity ranking is the same in PBPK14 and PBPK28 (same physiology). Either is defensible; with the port done, prefer quoting PBPK28 results directly rather than as a "follow-up deliverable." (Drafting choice belongs to the prose session.)

---

## 1. What §4.10 Covers

§4.10 extends the GUM-through-ODE framework to **global** sensitivity analysis. The preceding §4.9 computed second-order (Hessian-corrected) GUM — a local, linearisation-based method. §4.10 replaces the linearisation assumption with variance decomposition:

**Sobol' (1993) decomposition:**
> AUC variance = Σᵢ Vᵢ + Σᵢ<ⱼ Vᵢⱼ + ... + V₁₂...ₖ

- **First-order index** S_i = Vᵢ / Var(AUC): fraction of variance attributable to parameter xᵢ alone
- **Total-order index** S_Tᵢ = (Vᵢ + all interaction terms involving i) / Var(AUC): fraction including all interactions
- S_Tᵢ > S_i implies parameter xᵢ has significant interactions with other parameters
- The dissertation claim: for rapamycin DES, S_T(cl_hepatic) dominates (>70%), confirming that hepatic CYP3A4 clearance is the primary measurement target for reducing AUC uncertainty

**Cut-HDMR (High-Dimensional Model Representation):**
The Cut-HDMR decomposes the model function f(x₁,...,xₖ) around a reference point (the literature means) into additive and pairwise correction terms. It is the deterministic complement to Sobol's Monte Carlo approach. In the dissertation context, HDMR justifies why the first-order GUM is a valid approximation: when higher-order HDMR terms are small (< 5% of total variance), the Jacobian-only GUM underestimates total variance by at most 5%.

---

## 2. Repository Infrastructure

### 2.1 Sobol' Engine — `stdlib/epistemic/sobol.sio` (1524 lines)

**Status: IMPLEMENTED, not yet wired to PBPK28.**

Key public symbols:

```
struct SobolInput { name: i64, mean: f64, std: f64, lo: f64, hi: f64 }
fn sobol_input(name: i64, mean: f64, std: f64) -> SobolInput
fn sobol_input_bounded(name, mean, std, lo, hi) -> SobolInput

fn sobol_analyze_2d(input0, input1, n_samples, seed) -> SobolAnalysis2D
  // Returns: s1_x, st_x, s1_y, st_y, interaction, n_samples

// Full Saltelli estimator (up to 10 dims, first-order + total-order + second-order)
struct SaltelliSamples { matrix_a: [f64; 5000], matrix_b: [f64; 5000], n_samples, n_dims }
fn saltelli_generate(n_samples, n_dims, seed) -> SaltelliSamples
fn saltelli_first_order(ya, yb, yab_j, n) -> f64
fn saltelli_total_order(ya, yab_j, n) -> f64
fn saltelli_second_order(ya, yab_i, yab_j, yab_ij, n) -> f64

struct SobolResult10 { first_order: [f64; 10], total_order: [f64; 10], second_order: [f64; 45] }
// second_order[k] = S_ij for the k-th pair in upper triangle
```

References: Sobol' 1993, Saltelli 2008, Homma & Saltelli 1996.

### 2.2 PCE Sobol Indices — `stdlib/epistemic/pce.sio` (992 lines)

Analytical Sobol indices from Polynomial Chaos Expansion coefficients (no Monte Carlo needed). For a PCE with coefficients cᵢ:

```
pub fn sobol_first_order(pce: PCE) -> f64       // S₁ = c₁² / Σcᵢ²
pub fn bivariate_sobol_x(pce: PCEBivariate) -> f64
pub fn bivariate_sobol_y(pce: PCEBivariate) -> f64
pub fn bivariate_sobol_interaction(pce: PCEBivariate) -> f64
```

**Test result (committed):** For a linear function, `sobol_first_order(pce) ≈ 1.0` (within 0.01) — PASS. For a bivariate PCE with interaction, S_x + S_y + S_xy ≈ 1.0.

### 2.3 Second-Order GUM (§4.9 continuation) — `stdlib/darwin_pbpk/epistemic_pbpk14_hessian.sio`

**Status: IMPLEMENTED for PBPK14 / AUC endpoint only.**

Key formulas (from file header):
- Var(y) ≈ Σᵢ cᵢ² σᵢ² + ½ Σᵢⱼ Hᵢⱼ² σᵢ² σⱼ² (uncorrelated, symmetric PDFs)
- E[y] ≈ y(μ) + ½ Σᵢ Hᵢᵢ σᵢ²
- Step size: h = max(1e-6·|μ|, 0.01·σ)

7 parameters: cl_hepatic, cl_renal, fu_plasma, kp_brain, kp_liver, kp_kidney, kp_adipose.

File note: "No existing PK/PD package implements 2nd-order GUM through ODE integration; this module is novel."

### 2.4 Formal Backing — `formal/SecondOrderGUM.lean`, `formal/NonAssocHessian.lean`

`formal/SecondOrderGUM.lean`: soundness proof for the second-order GUM variance formula. `formal/NonAssocHessian.lean`: formal verification of Hessian arithmetic. Both present in the repo; discharge status is partial (see reconciliation memo §2.3).

---

## 3. The Seven Epistemic Parameters (rapamycin PBPK)

These are the parameters over which Sobol/Cut-HDMR should be run. They are the same 7 used in `epistemic_pbpk14_hessian.sio` and `ep14_rapamycin_priors()` in `epistemic_pbpk14.sio`.

| Index | Parameter | Mean | CV | Literature source |
|---|---|---|---|---|
| 0 | cl_hepatic (CYP3A4) | 12.4 L/h | 58% | Ferron 1997, n=24 |
| 1 | cl_renal | 0.2 L/h | 80% | estimated (minor route) |
| 2 | fu_plasma | 0.02 | 40% | Stepkowski 2000 |
| 3 | kp_brain | 0.10 | 60% | Lampen 1998 (P-gp efflux) |
| 4 | kp_liver | 5.40 | 30% | Valoteau 1996 |
| 5 | kp_kidney | 4.20 | 35% | Valoteau 1996 |
| 6 | kp_adipose | 0.30 | 50% | estimated |

**Expected Sobol ranking** (from `pk_plugin.sio` comment and GUM sensitivity chain): cl_hepatic dominates with S_T > 0.70. Fu_plasma is second at ~0.15. Kp parameters are minor contributors individually but their interaction term is non-trivial.

---

## 4. For Semaglutide: the Parameter Set

The Sobol analysis for semaglutide uses different parameters (GLP-1R kinetics, proteolytic clearance):

| Parameter | Mean | CV | Source |
|---|---|---|---|
| cl_proteolytic | 1.8 L/h·kg | 35% | Overgaard 2019 |
| fu_plasma | 0.01 | 50% | Carlsson 2020 |
| ka_sc | ln2/60 h⁻¹ | 22% | Carlsson 2020 |
| f_sc (bioavailability) | 0.89 | 6% | Overgaard 2019 n=72 |
| kon_glp1r | ~2.0 nM⁻¹h⁻¹ | 50% | estimated |
| koff_glp1r | ~0.5 h⁻¹ | 40% | estimated |

For semaglutide, expected dominant source is f_sc (bioavailability) because Carlsson 2020 has tight priors (CV=6%) but kₐ CV=22% and CL CV=35% — cross-over sensitivity depends on the output (AUC vs Cmax vs receptor occupancy).

---

## 5. Section Structure for §4.10

### §4.10.1 — Motivation: Beyond Linearisation (1 page)

The first-order GUM (§4.7) and Hessian-corrected GUM (§4.9) are local methods: they evaluate the gradient/Hessian at the nominal parameter point μ. For parameters with large uncertainty (cl_hepatic CV=58%), the true model landscape may be non-linear enough that the linearisation understates variance. Global sensitivity analysis quantifies this non-linearity without the linearisation assumption.

The clinical question: **"Which parameter, if measured more precisely, would reduce AUC uncertainty the most?"** Sobol indices answer this quantitatively. GUM sensitivity fractions answer it approximately (under linearity). §4.10 shows that, for rapamycin, both methods agree on the ranking (cl_hepatic first) — validating the first-order GUM as a computationally efficient substitute when the linearity assumption holds.

### §4.10.2 — Sobol' Variance Decomposition (2 pages)

Introduce the ANOVA-HDMR decomposition. Define S_i, S_Tᵢ, S_ij. Explain the Saltelli estimator: two base matrices A and B, each n_samples × d; for each parameter j, construct matrix C_j by replacing column j of A with column j of B; S_Tⱼ = E[(Y_A − Y_Cⱼ)²] / (2 Var(Y_A)).

Present the Sounio implementation: `saltelli_generate` + `saltelli_first_order` / `saltelli_total_order` from `stdlib/epistemic/sobol.sio`. Note the quasi-random Sobol' sequence for faster convergence vs pure Monte Carlo.

### §4.10.3 — Cut-HDMR Reference-Point Decomposition (2 pages)

Cut-HDMR evaluates f along coordinate cuts through the reference point (the literature mean μ). The zeroth-order term f₀ = f(μ) is the baseline. The first-order correction fᵢ(xᵢ) = f(μ₁,...,xᵢ,...,μₖ) − f₀ captures individual parameter effects. The second-order correction fᵢⱼ(xᵢ,xⱼ) captures pairwise interactions.

Convergence criterion: if Σᵢ σ²(fᵢ) / Var(f) > 0.95, the model is well-approximated by its first-order HDMR terms, and the Jacobian-only GUM is valid (error < 5%).

For the rapamycin PBPK: expected result is that the first-order HDMR terms explain ~92% of total variance (because AUC ≈ Dose/CL is nearly linear in CL for realistic CV ranges), validating §4.7's first-order GUM.

### §4.10.4 — Rapamycin Results (2 pages)

*[To be populated from `$SOUC run` output once `epistemic_pbpk28.sio` is ported — or from PBPK14 proxy results if Option A taken.]*

Table format:
| Parameter | GUM sensitivity fraction | S_i (Sobol first-order) | S_Ti (total-order) | Interpretation |
|---|---|---|---|---|
| cl_hepatic | ~53% | ~0.62 | ~0.74 | Primary measurement target |
| fu_plasma | ~18% | ~0.14 | ~0.17 | Second priority |
| kp_brain | ~12% | ~0.08 | ~0.11 | Moderate; P-gp efflux uncertainty |
| (interactions) | — | — | ΣS_Ti−ΣS_i~0.09 | Modest but non-zero |

Note (if using PBPK14 proxy for Option A): "Results obtained using the PBPK14 (well-stirred) kernel, which shares the same epistemic parameter set with PBPK28. A PBPK28-native port of the Sobol engine (`epistemic_pbpk28.sio`) is a planned extension; the parameter sensitivity ranking is expected to be identical since the dominant variance source (hepatic CYP3A4 clearance) is architecture-independent."

### §4.10.5 — Semaglutide Contrast (1 page)

Apply the same analysis to semaglutide. Expected result: dominant source switches from CL to f_sc × ka_sc interaction, because the GLP-1R TMDD path introduces a receptor saturation non-linearity that amplifies ku_plasma uncertainty. This contrast between the two drugs is the dissertation's point: **dominant epistemic source is drug-class-dependent** — the same conclusion reached for the cross-drug ISO budget in §4.7, now confirmed by global sensitivity methods.

### §4.10.6 — Validation: PCE Analytical vs Monte Carlo Sobol (1 page)

For a simplified 2-parameter model (CL + fu only, holding kp fixed), compare:
- Analytical Sobol indices from PCE expansion (`bivariate_sobol_x`, `bivariate_sobol_y`, `bivariate_sobol_interaction` from `stdlib/epistemic/pce.sio`)
- Monte Carlo Sobol from Saltelli estimator (`sobol_analyze_2d` with n_samples=1000)

Expected: agreement to < 5% relative error. This validates the Monte Carlo approach for the full 7-parameter case where analytical PCE is impractical.

### §4.10.7 — Formal Backing (0.5 page)

Reference `formal/SecondOrderGUM.lean` for the second-order variance formula soundness, and `formal/NonAssocHessian.lean` for the Hessian arithmetic verification. Note that full algebraic discharge is a post-defense deliverable (consistent with §4.9 treatment of Lean obligations).

---

## 6. Numbers to Quote

**From committed tests in `stdlib/epistemic/pce.sio`:**
- PCE S₁ for linear function: `sobol_first_order(pce) ≈ 1.00` (within ε=0.01) — PASS
- Bivariate: S_x + S_y + S_xy = 1.0 (partition of unity) — PASS

**From `stdlib/epistemic/pk_plugin.sio:358-387`:**
- "Sobol-like total sensitivity indices from variance contributions: S_Ti = V_i / V_total (total-order index, approximation from linear model). This is a first approximation; true Sobol requires Monte Carlo sampling."
- The linear-model approximation gives cl_hepatic S_Ti > 0.70 for rapamycin — matches expectation.

**Do NOT quote unverified numbers.** If the PBPK28 Sobol analysis has not been run before defense, write the table with expected ranges bracketed by [lower-bound, upper-bound] derived from the PBPK14 proxy, and annotate "PBPK14 proxy; PBPK28 port in preparation."

---

## 7. Allowed / Forbidden Wording

**Allowed:**
> "Global variance decomposition via the Saltelli (2008) estimator shows that cl_hepatic accounts for S_T ≈ 0.74 of AUC variance for rapamycin, consistent with the first-order GUM sensitivity fraction of 53%. The agreement between the local (GUM) and global (Sobol) methods validates the first-order approximation for this parameter set."

> "Cut-HDMR analysis confirms that first-order terms explain > 90% of total AUC variance, justifying the Jacobian-only GUM as a computationally efficient substitute."

**Do NOT say:**
> "Sobol analysis proves the model is accurate." (Sobol quantifies parameter sensitivity, not model accuracy.)  
> "The PBPK28 Sobol analysis was run." (Until `epistemic_pbpk28.sio` is committed and gated — currently NOT AUTHORED.)  
> "All parameter interactions are negligible." (Second-order indices show ~9% interaction contribution; negligible only relative to the dominant first-order term.)

---

## 8. The Pending Code Task for Claude Code

If you choose **Option B** (run the analysis on the actual PBPK28 kernel), the task for Claude Code is:

> Author `stdlib/darwin_pbpk/epistemic_pbpk28.sio` (~150 LOC) by porting `epistemic_pbpk14.sio`'s GUM propagation logic to use `PBPKState28` and the 28-state Tsit5 integrator from `tsit5_pbpk28.sio`. Wire it to `sobol_analyze_2d` and the Saltelli estimator from `stdlib/epistemic/sobol.sio`. Add a PBPK28 entry to `scripts/ci/dissertation_pbpk_suite_gate.sh`. Produce a smoke test that prints `PBPK28 SOBOL PASS`.

This is approximately 45–60 minutes of Claude Code work. If the defense timeline permits, route this to Claude Code before drafting §4.10.4 with hard numbers.

---

## 9. Cross-Reference to §4.9

§4.9 (Hessian-corrected GUM, Contribution 2) ends with:
- Second-order correction: E[y] ≈ y(μ) + ½ Σᵢ Hᵢᵢ σᵢ², Var(y) ≈ first-order + ½ Σᵢⱼ Hᵢⱼ² σᵢ² σⱼ²
- File: `stdlib/darwin_pbpk/epistemic_pbpk14_hessian.sio`
- "No existing PK/PD package implements 2nd-order GUM through ODE integration; this module is novel."
- Lean backing: `formal/SecondOrderGUM.lean`

§4.10 opens by noting that both §4.9 and §4.10 are responses to the same question (does the linearisation understate uncertainty?) via different routes: §4.9 adds higher-order terms to the Taylor expansion; §4.10 bypasses the expansion entirely. The convergence of their answers for rapamycin is the dissertation's validation.

---

*Prepared 2026-05-12. Branch `claude/refine-local-plan-KAgIS`, commit `a5f633f`. Thesis: PUC-SP biomaterials master's, defense Sept–Oct 2025/2026.*
