# Executable Epistemic PET Kinetics: GUM-Compliant Propagation of PBPK-Informed Uncertainty into Receptor Binding Metrics

**Authors:** Demetrios Chiuratto Agourakis (Sounio Language Project)

**Submission type:** Late-Breaking Abstract, NeuroReceptor Mapping 2026

---

## Abstract (approx. 300 words)

**Background.** Quantitative PET neuroreceptor studies rely on kinetic models whose outputs — distribution volume (V_T), binding potential (BP_ND) — are typically reported with statistical uncertainty from bootstrap or asymptotic covariance. These approaches rarely quantify the **epistemic** contribution of upstream physiological inputs such as unbound plasma fraction (fu) and blood-brain barrier transport, even though these are known to dominate inter-study variability.

**Objective.** We present a reproducible, executable vertical slice that propagates GUM-compliant (JCGM 100:2008 §5.1.3) epistemic uncertainty from eight priors (K1, k2, k3, k4, plasma input amplitude, plasma decay, fu_plasma, bbb_scalar) through a standard two-tissue compartment model (2TCM) to V_T and BP_ND, using the same finite-difference Jacobian methodology already validated in a 14-compartment PBPK model in the same Sounio language codebase.

**Methods.** The 2TCM is integrated with a fixed-step classical RK4 (dt = 0.05 min, t = 0..60 min) against a synthetic exponential plasma input. The model equations are exactly those of the Lammertsma / Innis consensus formulation [Lammertsma 1996; Innis 2007], with `V_T = (K₁_eff/k₂)·(1 + k₃/k₄)` and `BP_ND = k₃/k₄`. Sensitivity coefficients `c_i = ∂y/∂θ_i` are obtained by forward differences with step `h_i = max(10⁻⁶|μ_i|, 10⁻²·σ_i)`. The combined variance `Var(y) = Σ c_i² Var(θ_i)` is computed for TAC AUC, TAC peak, V_T, and BP_ND, together with normalized sensitivity fractions and an evidence-weighted confidence score. Synthetic priors are chosen at the centre of the published [11C]raclopride human-striatum range (K₁ 0.15, k₂ 0.20, k₃ 0.10, k₄ 0.05), giving reference V_T = 2.25 and BP_ND = 2.00 — both inside the published range 2–4 and 1.5–3.0 respectively.

**Results.** All internal audits pass. (i) GUM-propagated uncertainty agrees with the analytic delta-method to ≤ 0.5 %: V_T SD 0.696 (analytic 0.695), BP_ND SD 0.565 (analytic 0.566). (ii) The structural insensitivity of BP_ND to fu_plasma and bbb_scalar is recovered exactly (d = 0). (iii) Four tracer variants ([11C]raclopride, flumazenil, DASB-like, PK11195-like) all place V_T and BP_ND inside the published human-brain range. (iv) An SRTM implementation (Lammertsma & Hume 1996) reproduces the documented scientific behaviour — accurate under rapid equilibrium ((k3+k4)/k2 >> 1, max err 10 %), biased when (k3+k4)/k2 ≈ 1 (max err 63 %). (v) Monte-Carlo recovery (20 noise realizations, 5 % of peak): convergence 20/20; V_T bias 3.7 %, CV 2.5 %; BP_ND bias 8 %, CV 3.5 %; k3 CV 4.2 %, k4 CV 5.9 % — reproducing the canonical identifiability hierarchy of Lammertsma 1996 and Hume 1992. **(vi) Real-data validation against Lammertsma 1996 JCBFM 16:42–52**: computing BP_ND = V_T_striatum/V_T_cerebellum − 1 (Innis 2007) from the paper's published Table 2 V_T values across all 8 normal subjects reproduces the paper's own Table 3 BP values with Pearson r = 0.9995, mean bias 0.005, RMS error 0.026 — an independent algebraic check against canonical clinical data.

**Conclusion.** Executable, language-level epistemic PET kinetics is feasible. This slice provides an auditable path for coupling PBPK priors into neuroreceptor metrics for future clinical and methodological work.

---

## Framing

This submission is **not** a clinical fitting tool. It is a minimal, fully auditable vertical slice that demonstrates the feasibility of executable epistemic PET modeling with analytic-level numerical fidelity in a strongly-typed, self-hosted language (Sounio).

The scientific differentiation:
1. **Analytic-level fidelity** of GUM finite-difference derivatives (≤ 0.5 % error vs delta-method).
2. **Literature-anchored priors** at the centre of the published [11C]raclopride human-striatum range (Lammertsma 1996; Farde 1989).
3. **Structural insensitivity** of BP_ND to fu_plasma and bbb_scalar correctly recovered (d = 0 exactly).
4. **Multi-tracer portability** — four tracer variants all inside published V_T/BP_ND ranges.
5. **SRTM regime discrimination** — implementation reproduces the canonical "accurate under rapid equilibrium, biased under slow binding" behaviour (Lammertsma & Hume 1996).
6. **Parameter recovery** from synthetic noisy TAC reproduces the identifiability hierarchy (V_T, BP_ND stable; individual k3, k4 less stable) — matching Lammertsma 1996 and Hume 1992.
7. **Real-data validation**: algebraically reproduces the BP_ND values published in Lammertsma 1996 Table 3 from the Table 2 V_T values (r = 0.9995 across 8 subjects).
8. **Full reproducibility**: source + deterministic integrator + all audit outputs + CSV export + literature-comparison table.

## Artifacts

- `pet_2tcm_epistemic.sio` — main 2TCM + GUM audit (12 tests)
- `pet_2tcm_export.sio` — CSV TAC curve exporter
- `pet_tracer_variants.sio` — raclopride / flumazenil / DASB / PK11195 variants
- `pet_srtm.sio` — SRTM vs 2TCM validation, both regimes
- `pet_fit_validation.sio` — single-realization parameter recovery
- `pet_fit_montecarlo.sio` — Monte Carlo bias/precision (20 realizations)
- `pet_lammertsma1996_analysis.sio` — real-data reproduction of Lammertsma 1996 Table 3
- `LITERATURE_VALIDATION.md` — comparison table with citations
- `results/*.txt` — captured stdout for each audit
- `results/tac_curve.csv` — generated TAC
- Repository: Sounio-lang/darwin-pbpk @ `integration/sounio-dev-ready-base`

## Key References

1. Lammertsma AA *et al.* Comparison of methods for analysis of clinical [11C]raclopride studies. *J Cereb Blood Flow Metab* 1996; 16: 42–52.
2. Farde L *et al.* Kinetic analysis of central [11C]raclopride binding to D2-dopamine receptors. *J Cereb Blood Flow Metab* 1989; 9: 696–708.
3. Innis RB *et al.* Consensus nomenclature for in vivo imaging of reversibly binding radioligands. *J Cereb Blood Flow Metab* 2007; 27: 1533–1539.
4. Lammertsma AA, Hume SP. Simplified reference tissue model for PET receptor studies. *Neuroimage* 1996; 4: 153–158.
5. Gunn RN *et al.* Parametric imaging of ligand-receptor binding in PET using a simplified reference region model. *Neuroimage* 1997; 6: 279–287.
6. Hume SP *et al.* Quantitation of [11C]raclopride in rat striatum using PET. *Synapse* 1992; 12: 47–54.
7. Price JC *et al.* Measurement of benzodiazepine receptor number and affinity using [11C]flumazenil. *J Cereb Blood Flow Metab* 1993; 13: 656–667.
8. Koeppe RA *et al.* Compartmental analysis of [11C]flumazenil kinetics. *J Cereb Blood Flow Metab* 1991; 11: 735–744.
9. JCGM 100:2008. Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM). BIPM.

## Limitations (explicitly acknowledged)

- Synthetic plasma input, not arterial sampling.
- Priors are plausible, not fitted to any real dataset.
- No hierarchical modeling, no partial volume correction, no fitting.
- Not intended for clinical or regulatory decision-making.

## Disclosure

No external funding. No conflicts of interest. No patient data.

---

*Submitted as a proof-of-concept vertical slice for discussion purposes at NRM 2026.*
