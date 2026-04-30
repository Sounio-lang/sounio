# NRM 2026 — Submission Text

> Derived from `NRM2026_ABSTRACT.md` and `AUDIT_REPORT.md` in the same
> directory. No code, numerical results, or scientific claims were altered
> in the preparation of this submission text. Portal word limits and
> formatting vary by track; trim where required.

---

## Title

**Executable epistemic PET kinetics: GUM-compliant uncertainty propagation from PBPK-informed priors to V_T and BP_ND in a two-tissue compartment model**

## Authors & affiliations

**Demetrios Chiuratto Agourakis** — independent, Sounio Language Project (corresponding author).

> *Note:* update affiliation and contact email on portal submission; no
> co-authors are listed in the audited source tree as of commit
> `2e817fcbde01b14ac3524c09e4ae0d88d72d83c3`.

## Submission category

Late-breaking abstract — **methodological proof-of-concept**.
Not a clinical PET fitting package.

---

## Abstract (≈ 285 words)

**Background.** Quantitative PET neuroreceptor analyses report the distribution volume (V_T) and the binding potential (BP_ND) with statistical uncertainty from bootstrap or asymptotic covariance. These approaches seldom make explicit the *epistemic* contribution of upstream physiological inputs — such as unbound plasma fraction (*fu*) and blood-brain-barrier transport — even though such inputs are known to dominate inter-study variability.

**Objective.** To present a minimal, reproducible, fully auditable proof-of-concept that propagates GUM-compliant (JCGM 100:2008 §5.1.3) epistemic uncertainty from eight priors (K1, k2, k3, k4, plasma amplitude, plasma decay, *fu*_plasma, *bbb*_scalar) through a standard two-tissue compartment model (2TCM) to V_T and BP_ND via a finite-difference Jacobian.

**Methods.** The 2TCM is integrated with classical RK4 (dt = 0.05 min, t = 0–60 min) against a synthetic exponential plasma input, using the Lammertsma / Innis consensus formulation: V_T = (K1_eff/k2)·(1 + k3/k4), BP_ND = k3/k4. Forward-difference sensitivities use `h_i = max(10⁻⁶|μ_i|, 10⁻²·σ_i)`; the combined variance `Var(y) = Σ c_i²·Var(θ_i)` is computed for TAC AUC, TAC peak, V_T and BP_ND, together with normalised sensitivity fractions. Priors are set at the centre of published [11C]raclopride human-striatum ranges, giving reference V_T = 2.25 and BP_ND = 2.00.

**Results.** GUM-propagated uncertainty agrees with the analytic delta-method to ≤ 0.5 % (V_T SD 0.696 vs 0.695; BP_ND SD 0.565 vs 0.566). The structural insensitivity ∂BP_ND/∂*fu* = ∂BP_ND/∂*bbb* = 0 is recovered to machine precision. All 12 internal acceptance tests pass; per-file numerical audit is released with the source.

**Conclusion.** Executable, auditable epistemic PET uncertainty propagation is feasible as a self-contained artefact and may serve as a methodological building block for coupling PBPK priors into neuroreceptor metrics.

---

## Keywords

PET kinetic modelling; two-tissue compartment model; uncertainty propagation; GUM; PBPK-informed priors; reproducibility

---

## Short methodological note

The implementation is a single-compilation-unit 2TCM + GUM propagation, released in source form. Integration is a fixed-step classical RK4 with dt = 0.05 min over 60 min; AUC uses composite trapezoidal integration on the time-activity curve. Sensitivities are computed by one-sided finite differences with a per-parameter step `h_i = max(10⁻⁶|μ_i|, 10⁻²·σ_i)` chosen to balance truncation and cancellation error; combined variance follows JCGM 100:2008 §5.1.3. Every numerical claim in the abstract is checked by an in-source acceptance test against the analytic delta-method and against the closed-form values of V_T and BP_ND for the chosen priors; the full pass/fail table is included in `AUDIT_REPORT.md`.

Supplementary illustrative files exercise the same machinery across four literature-informed parameter sets (raclopride, flumazenil, nominal DASB-like, nominal PK11195-like); a side-by-side SRTM vs 2TCM demonstration (Lammertsma & Hume 1996); a coordinate-descent Monte-Carlo parameter-recovery stress test on a synthetic noisy TAC; and an algebraic consistency check of the Innis 2007 consensus relation `BP_ND = V_T_target/V_T_reference − 1` against the *published aggregate V_T and BP_ND values* from Lammertsma 1996 Tables 2–3 (this is **not** a re-analysis of dynamic PET data).

---

## Reproducibility statement

All code, synthetic priors, acceptance tests, stdout captures and the numerical audit are released under the Sounio repository:

- **Repository:** Sounio-lang/sounio — https://github.com/Sounio-lang/sounio
- **Branch:** `integration/sounio-dev-ready-base`
- **Audited commit:** `2e817fcbde01b14ac3524c09e4ae0d88d72d83c3`
- **Provenance head (post-audit metadata fixes):** `2aa6859093c149d1b931dd5413c707513353f7f6`
- **Path:** `examples/neuroreceptor_pet/`
- **Audit report:** `examples/neuroreceptor_pet/AUDIT_REPORT.md`

The implementation is written in the Sounio language and uses no external numerical libraries. A single command (`./bin/souc run examples/neuroreceptor_pet/pet_2tcm_epistemic.sio`) executes the full 12-test acceptance audit.

---

## Scope statement

This submission is a methodological **proof-of-concept**. It is explicitly **not**:

- a clinical PET fitting or quantification package;
- equivalent to, or a replacement for, PMOD, AMIDE, PNEURO, or any peer-reviewed kinetic-modelling software;
- a re-analysis of, or a fit to, any real dynamic PET dataset;
- validated against in-vivo test-retest data for any tracer;
- intended for diagnostic, regulatory, dosimetric, or clinical decision-making use.

Priors are plausible and literature-anchored; they are **not** fitted. The Lammertsma 1996 consistency check operates on *published aggregate summary statistics* (Tables 2–3), not on dynamic TAC time series.

---

## Key references

1. Lammertsma AA *et al.* *J Cereb Blood Flow Metab* 1996; 16: 42–52.
2. Farde L *et al.* *J Cereb Blood Flow Metab* 1989; 9: 696–708.
3. Innis RB *et al.* *J Cereb Blood Flow Metab* 2007; 27: 1533–1539.
4. Lammertsma AA, Hume SP. *Neuroimage* 1996; 4: 153–158.
5. Gunn RN *et al.* *Neuroimage* 1997; 6: 279–287.
6. JCGM 100:2008. *Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM).* BIPM.

---

## Disclosure

No external funding. No conflicts of interest. No patient data.
