# Executable Epistemic PET Kinetics: GUM-Compliant Propagation of PBPK-Informed Uncertainty into Receptor Binding Metrics

**Authors:** Demetrios Chiuratto Agourakis (Sounio Language Project)

**Submission type:** Late-Breaking Abstract, NeuroReceptor Mapping 2026

---

## Abstract (approx. 300 words)

**Background.** Quantitative PET neuroreceptor studies rely on kinetic models whose outputs — distribution volume (V_T), binding potential (BP_ND) — are typically reported with statistical uncertainty from bootstrap or asymptotic covariance. These approaches rarely quantify the **epistemic** contribution of upstream physiological inputs such as unbound plasma fraction (fu) and blood-brain barrier transport, even though these are known to dominate inter-study variability.

**Objective.** We present a reproducible, executable vertical slice that propagates GUM-compliant (JCGM 100:2008 §5.1.3) epistemic uncertainty from eight priors (K1, k2, k3, k4, plasma input amplitude, plasma decay, fu_plasma, bbb_scalar) through a standard two-tissue compartment model (2TCM) to V_T and BP_ND, using the same finite-difference Jacobian methodology already validated in a 14-compartment PBPK model in the same Sounio language codebase.

**Methods.** The 2TCM is integrated with a fixed-step classical RK4 (dt = 0.05 min, t = 0..60 min) against a synthetic exponential plasma input. Sensitivity coefficients c_i = ∂y/∂θ_i are obtained by symmetric forward differences with step h_i = max(10⁻⁶|μ_i|, 10⁻²·σ_i). The combined variance Var(y) = Σ c_i² Var(θ_i) is computed for TAC AUC, TAC peak, V_T, and BP_ND, together with normalized sensitivity fractions and an evidence-weighted confidence score.

**Results.** All 12 internal numerical audits pass. Computed values agree with analytic delta-method predictions to ≤ 0.5 %: V_T SD 0.696 (analytic 0.695), BP_ND SD 0.565 (analytic 0.566), d(BP_ND)/dk3 20.00 (analytic 20), d(V_T)/dfu 2.250 (analytic 2.25). The model correctly recovers the structural insensitivity of BP_ND to both fu_plasma and bbb_scalar (d = 0), demonstrating that GUM propagation separates kinetic-binding from PBPK-input uncertainty at the level of individual partial derivatives.

**Conclusion.** Executable, language-level epistemic PET kinetics is feasible. This slice provides an auditable path for coupling PBPK priors into neuroreceptor metrics for future clinical and methodological work.

---

## Framing

This submission is **not** a clinical fitting tool. It is a minimal, fully auditable vertical slice that demonstrates the feasibility of executable epistemic PET modeling with analytic-level numerical fidelity in a strongly-typed, self-hosted language (Sounio).

The scientific differentiation:
1. **Analytic-level fidelity** of finite-difference derivatives (≤ 0.5% error vs delta-method).
2. **Structural insensitivity** of BP_ND to fu_plasma and bbb_scalar correctly recovered.
3. **PBPK-PET coupling** at the level of sensitivity fractions (10.4% of V_T variance attributed to fu_plasma, 10.4% to bbb_scalar in the synthetic prior set).
4. **Full reproducibility**: single-file source + deterministic integrator + 12 internal audits + CSV export.

## Artifacts

- Source: `examples/neuroreceptor_pet/pet_2tcm_epistemic.sio` (audit, 12 tests)
- Export: `examples/neuroreceptor_pet/pet_2tcm_export.sio` (CSV TAC curve)
- Audit log: `examples/neuroreceptor_pet/results/audit_output.txt`
- TAC curve: `examples/neuroreceptor_pet/results/tac_curve.csv`
- Repository: Sounio-lang/darwin-pbpk @ `integration/sounio-dev-ready-base`
- Commit context: numerical audit commit (see `git log examples/neuroreceptor_pet/`)

## Limitations (explicitly acknowledged)

- Synthetic plasma input, not arterial sampling.
- Priors are plausible, not fitted to any real dataset.
- No hierarchical modeling, no partial volume correction, no fitting.
- Not intended for clinical or regulatory decision-making.

## Disclosure

No external funding. No conflicts of interest. No patient data.

---

*Submitted as a proof-of-concept vertical slice for discussion purposes at NRM 2026.*
