# Executable Epistemic PET Kinetics: GUM-Compliant Propagation of PBPK-Informed Uncertainty into Receptor Binding Metrics

**Authors:** Demetrios Chiuratto Agourakis (Sounio Language Project)

**Submission type:** Late-Breaking Abstract, NeuroReceptor Mapping 2026

**Category:** Proof-of-concept / methodological demonstration. **Not a clinical PET fitting package.**

---

## Abstract (approx. 300 words)

**Background.** Quantitative PET neuroreceptor analyses report distribution volume (V_T) and binding potential (BP_ND) with statistical uncertainty obtained from bootstrap or asymptotic covariance. These approaches rarely make explicit the *epistemic* contribution of upstream physiological inputs — unbound plasma fraction, blood-brain barrier transport — even though such inputs dominate inter-study variability.

**Objective.** We present a minimal, reproducible, *executable* proof-of-concept that propagates GUM-compliant (JCGM 100:2008 §5.1.3) epistemic uncertainty from eight priors (K1, k2, k3, k4, plasma amplitude, plasma decay, fu_plasma, bbb_scalar) through a standard two-tissue compartment model (2TCM) to V_T and BP_ND, using finite-difference Jacobian sensitivity — the same methodology previously exercised in a 14-compartment PBPK example in the Sounio language.

**Methods.** The 2TCM is integrated with fixed-step classical RK4 (dt = 0.05 min, t = 0..60 min) against a synthetic exponential plasma input. Equations follow the Lammertsma / Innis consensus formulation [Lammertsma 1996; Innis 2007]: `V_T = (K₁_eff/k₂)·(1 + k₃/k₄)`, `BP_ND = k₃/k₄`. Forward-difference sensitivities use `h_i = max(10⁻⁶|μ_i|, 10⁻²·σ_i)`; combined variance `Var(y) = Σ c_i² Var(θ_i)` is computed for TAC AUC, peak, V_T, BP_ND plus normalized sensitivity fractions. Synthetic priors are set at the centre of the published [11C]raclopride human-striatum range (K₁ 0.15, k₂ 0.20, k₃ 0.10, k₄ 0.05), giving reference V_T = 2.25 and BP_ND = 2.00. No real patient data are used.

**Results (verified internally by the audit in `AUDIT_REPORT.md`).** GUM-propagated uncertainty matches the analytic delta-method to ≤ 0.5 %: V_T SD 0.696 (analytic 0.695), BP_ND SD 0.565 (analytic 0.566); structural insensitivity ∂BP_ND/∂fu = ∂BP_ND/∂bbb = 0 is recovered exactly; all 12 internal acceptance tests pass. Supplementary examples exercise the same machinery across four tracer parameter sets and demonstrate a side-by-side SRTM vs 2TCM comparison; a coordinate-descent Monte-Carlo recovery on synthetic noisy TACs is provided as an illustrative stress test of the optimiser.

**Conclusion.** Executable, language-level epistemic PET modelling with analytic-level numerical fidelity is feasible. This slice is a methodological building block, not a clinical fitter; it is offered as an auditable starting point for coupling PBPK priors into neuroreceptor metrics.

---

## Framing

This submission is **not** a clinical fitting tool, a validated imaging pipeline, or a substitute for PMOD / AMIDE / PNEURO. It is a minimal, fully auditable proof-of-concept that demonstrates:

1. **Analytic-level fidelity** of GUM finite-difference derivatives (≤ 0.5 % relative error vs the analytic delta-method on V_T and BP_ND variance).
2. **Literature-anchored priors** — synthetic priors chosen inside the published [11C]raclopride human-striatum range (Lammertsma 1996; Farde 1989). Priors are *not* fitted to any real data.
3. **Structural insensitivity** ∂BP_ND/∂fu_plasma = ∂BP_ND/∂bbb_scalar = 0 recovered numerically to machine precision — a verifiable correctness property of the GUM propagation.
4. **Multi-tracer parameter-set portability** — the same 2TCM + GUM code exercised with four literature-informed prior sets (raclopride, flumazenil, nominal DASB-like, nominal PK11195-like). This is a code-portability demonstration, not a tracer-specific validation.
5. **SRTM example** — an SRTM solver (Lammertsma & Hume 1996) is included and shown to approximate 2TCM TACs under the rapid-equilibrium regime and to deviate under slow binding. This is a textbook illustration of the SRTM approximation, not a new result.
6. **Monte-Carlo stress test** — 20 independent LCG + Irwin-Hall noise realisations, fit by coordinate descent, show that V_T and BP_ND have lower CV than individual k3, k4 *for this particular optimiser and noise model*. Not a clinical identifiability claim.
7. **Formula consistency check** — the Innis 2007 relation `BP_ND = V_T_target/V_T_reference − 1` is re-computed from the aggregate V_T values published in Lammertsma 1996 Table 2 and reproduces that paper's own Table 3 BP_ND values to within two-decimal rounding (r = 0.9995 across the 8 tabulated subjects). This is an **algebraic consistency check on published aggregate metrics**, *not* a re-analysis of dynamic PET data.
8. **Full reproducibility** — source, deterministic integrator, deterministic LCG seed, captured stdout of every acceptance run in `results/`, and a standalone `AUDIT_REPORT.md`.

## Artifacts

- `pet_2tcm_epistemic.sio` — main 2TCM + GUM audit (12 internal tests)
- `pet_2tcm_export.sio` — CSV TAC curve exporter
- `pet_tracer_variants.sio` — four parameter-set variants
- `pet_srtm.sio` — SRTM vs 2TCM side-by-side demo (both regimes)
- `pet_fit_validation.sio` — single-realisation recovery stress test
- `pet_fit_montecarlo.sio` — 20-realisation Monte-Carlo stress test
- `pet_lammertsma1996_analysis.sio` — Innis 2007 formula consistency check against published aggregate V_T/BP from Lammertsma 1996
- `LITERATURE_VALIDATION.md` — prior-range citations
- `AUDIT_REPORT.md` — pass/fail audit of every numerical claim above
- `results/*.txt` — captured stdout for each acceptance run
- Repository: Sounio-lang/sounio (https://github.com/Sounio-lang/sounio) @ `integration/sounio-dev-ready-base`
- Audited commit: `2e817fcbde01b14ac3524c09e4ae0d88d72d83c3`

## Key References

1. Lammertsma AA *et al.* Comparison of methods for analysis of clinical [11C]raclopride studies. *J Cereb Blood Flow Metab* 1996; 16: 42–52.
2. Farde L *et al.* Kinetic analysis of central [11C]raclopride binding to D2-dopamine receptors. *J Cereb Blood Flow Metab* 1989; 9: 696–708.
3. Innis RB *et al.* Consensus nomenclature for in vivo imaging of reversibly binding radioligands. *J Cereb Blood Flow Metab* 2007; 27: 1533–1539.
4. Lammertsma AA, Hume SP. Simplified reference tissue model for PET receptor studies. *Neuroimage* 1996; 4: 153–158.
5. Gunn RN *et al.* Parametric imaging of ligand-receptor binding in PET using a simplified reference region model. *Neuroimage* 1997; 6: 279–287.
6. JCGM 100:2008. Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM). BIPM.

## Limitations (explicitly acknowledged)

- Synthetic exponential plasma input, not arterial sampling.
- Priors are plausible and literature-anchored, *not* fitted to any patient dataset.
- No hierarchical modelling, no partial-volume correction, no metabolite correction, no delay/dispersion modelling, no frame weighting.
- Monte-Carlo uses a simple coordinate-descent optimiser on a discrete multiplier grid, not Levenberg-Marquardt; the reported bias on individual k3, k4 reflects that optimiser, not a general identifiability claim.
- The "Lammertsma 1996 agreement" in `pet_lammertsma1996_analysis.sio` is an algebraic consistency check on *published summary statistics* (Tables 2 and 3), not a fit to or re-analysis of dynamic TAC data.
- Not intended for clinical, diagnostic, regulatory, or dosimetric decision-making.

## Disclosure

No external funding. No conflicts of interest. No patient data.

---

*Submitted as a proof-of-concept vertical slice for methodological discussion at NRM 2026.*
