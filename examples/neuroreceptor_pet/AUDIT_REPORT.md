# Audit Report — `examples/neuroreceptor_pet/`

**Date:** 2026-04-28
**Repository:** Sounio-lang/darwin-pbpk
**Branch:** `integration/sounio-dev-ready-base`
**Audit scope:** every `.sio` file and every quantitative claim in
`examples/neuroreceptor_pet/`.
**Audit mode:** no new features. Verification only.

---

## 1. Files audited

| File | Type |
|------|------|
| `pet_2tcm_epistemic.sio` | 2TCM + GUM propagation, 12 acceptance tests |
| `pet_2tcm_export.sio` | CSV TAC exporter |
| `pet_tracer_variants.sio` | Four parameter-set variants |
| `pet_srtm.sio` | SRTM vs 2TCM side-by-side demo |
| `pet_fit_validation.sio` | Single-realisation recovery stress test |
| `pet_fit_montecarlo.sio` | 20-realisation Monte-Carlo stress test |
| `pet_lammertsma1996_analysis.sio` | Innis 2007 formula consistency check on published aggregates |

Plus prose: `README.md`, `NRM2026_ABSTRACT.md`, `LITERATURE_VALIDATION.md`.

## 2. Commands run

```bash
export SOUC_BIN=./bin/souc
export SOUNIO_STDLIB_PATH=./stdlib

# Type-check every file
for f in examples/neuroreceptor_pet/*.sio; do $SOUC_BIN check "$f"; done

# Run every file, capture stdout in /tmp/audit_pet/
for f in examples/neuroreceptor_pet/*.sio; do
  $SOUC_BIN run "$f" > /tmp/audit_pet/$(basename $f .sio).out 2>&1
done
```

All files were checked and executed from a clean `integration/sounio-dev-ready-base`
working tree prior to writing this report. Captured outputs are also stored in
`examples/neuroreceptor_pet/results/*.txt`.

## 3. Pass / fail table

| File | `check` | `run` exit | Internal acceptance | Status |
|------|:-------:|:---------:|:-------------------:|:------:|
| `pet_2tcm_epistemic.sio` | 0 | 0 | 12 / 12 PASS | **PASS** |
| `pet_2tcm_export.sio` | 0 | 0 | n/a (CSV exporter) | **PASS** |
| `pet_tracer_variants.sio` | 0 | 0 | 5 / 5 PASS | **PASS** |
| `pet_srtm.sio` | 0 | 0 | 3 / 3 PASS | **PASS** |
| `pet_fit_validation.sio` | 0 | 0 | 5 / 5 PASS | **PASS** |
| `pet_fit_montecarlo.sio` | 0 | 0 | 5 / 5 PASS | **PASS** (see caveats §6) |
| `pet_lammertsma1996_analysis.sio` | 0 | 0 | 6 / 6 PASS | **PASS** (see downgrade §7) |

## 4. Numerical sanity results

### 4.1 2TCM equations

Verified in `pet_2tcm_epistemic.sio` around line 157 (`fn rk4_step`) and
line 187 (`fn simulate_pet`):

```
dC1/dt = K1_eff * Cp(t) − (k2 + k3) * C1 + k4 * C2
dC2/dt = k3 * C1 − k4 * C2
CT(t)  = C1(t) + C2(t)     (recomputed, not accumulated)
K1_eff = K1 * fu_plasma * bbb_scalar
```

Match the Lammertsma / Innis consensus 2TCM exactly.

- RK4 is a standard classical 4-stage explicit Runge-Kutta.
- AUC uses trapezoidal integration: `auc += 0.5 * (ct + prev_ct) * dt`.
- `t` advances by `dt = 0.05` min each step, max 10000 steps over 60 min
  → loop bound 1200 steps, not pathological.
- Off-by-one at the end: AUC covers roughly `[0, t_end − dt]` rather than
  `[0, t_end]`. Impact at `dt = 0.05` is ≤ 0.1 %, far below all acceptance
  thresholds. Documented here.

### 4.2 Plasma input

`Cp(t) = cp_amp * exp_f64(−cp_decay * t)`. Captured values at default
amp = 1, decay = 0.20:

| t (min) | Cp, computed | Cp, analytic | rel. err |
|:-------:|:------------:|:------------:|:--------:|
| 0  | 1.000000     | 1.000000     | 0 |
| 1  | 0.818731     | 0.818731     | < 1e-6 |
| 5  | 0.367879     | 0.367879     | < 1e-6 |
| 10 | 0.135335     | 0.135335     | < 1e-6 |
| 60 | 6.14 × 10⁻⁶  | 6.14 × 10⁻⁶  | < 1e-6 |

No explosion, no NaN, no negative values.

### 4.3 TAC AUC / peak plausibility

Audit-run output for default priors (K1 = 0.15, k2 = 0.20, k3 = 0.10,
k4 = 0.05, cp_amp = 1.00, cp_decay = 0.20):

| Metric | Computed | Sanity |
|--------|---------:|:------:|
| TAC AUC | 9.469517 | in expected range [5, 20] |
| TAC peak | 0.308191 | in expected range [0.05, 1.0] |
| V_T | 2.250000 | **matches analytic 2.25 exactly** |
| BP_ND | 2.000000 | **matches analytic 2.00 exactly** |

### 4.4 GUM finite-difference vs analytic delta-method

BP_ND = k3 / k4  ⇒  ∂BP_ND/∂k3 = 1/k4 = 20, ∂BP_ND/∂k4 = −k3/k4² = −40.
V_T = (K1·fu·bbb/k2)·(1 + k3/k4). At fu = bbb = 1, K1 = 0.15, k2 = 0.20,
BP_ND = 2: ∂V_T/∂K1 = 15, ∂V_T/∂k2 = −11.25, ∂V_T/∂fu = ∂V_T/∂bbb = 2.25.

Analytic variance predictions with σ²(K1) = 4e-4, σ²(k2) = 9e-4,
σ²(k3) = 4e-4, σ²(k4) = 1e-4, σ²(fu) = σ²(bbb) = 0.01:

- Var(BP_ND) = 20²·4e-4 + 40²·1e-4 = 0.160 + 0.160 = **0.320**
- SD(BP_ND) ≈ **0.5657**
- Var(V_T) kinetic part ≈ 15²·4e-4 + 11.25²·9e-4 + 2·2.25²·0.01 ≈ 0.484
- SD(V_T) ≈ **0.695**

Audited computed values (from `results/audit_output.txt`):

| Quantity | Analytic | Computed | Rel. error |
|----------|---------:|---------:|:----------:|
| ∂BP_ND/∂k3 | 20 | **20.000** | < 1e-4 |
| ∂BP_ND/∂k4 | −40 | **−39.920** | 0.2 % (forward-difference bias) |
| ∂BP_ND/∂fu | 0 | **0.000** | exact |
| ∂BP_ND/∂bbb | 0 | **0.000** | exact |
| ∂V_T/∂K1 | 15 | **15.000** | < 1e-4 |
| ∂V_T/∂k2 | −11.25 | **−11.233** | 0.15 % |
| ∂V_T/∂fu | 2.25 | **2.250** | < 1e-4 |
| ∂V_T/∂bbb | 2.25 | **2.250** | < 1e-4 |
| SD(BP_ND) | 0.566 | **0.565** | 0.18 % |
| SD(V_T) | 0.695 | **0.696** | 0.14 % |

BP_ND variance is **non-zero** and numerically agrees with the analytic
delta-method to < 0.5 %. Structural insensitivity of BP_ND to `fu_plasma`
and `bbb_scalar` is recovered to machine precision.

**Numerical audit: PASS.**

## 5. Hardcoded-output audit

Search targets: `397.49`, `10.40`, `9.47`, `0.31`, `2.25`, `2.00`,
`Lammertsma`, `Hume`, `PMOD`, `complete pipeline`, `validated`.

Classification of every match:

| Kind of match | Action |
|---|---|
| `V_T = 2.25`, `BP_ND = 2.00`, `AUC ≈ 9.47`, `peak ≈ 0.31` used as *expected-value* acceptance tests, the comparison is `if abs(result − expected) < tol` | **Legitimate**: these are analytically correct values for the default priors (`2.25 = 0.15/0.20 × 3`, `2.00 = 0.10/0.05`). Kept. |
| `V_T = 2.25` and `BP_ND = 2.00` appearing in `println` labels such as `"expected exact 2.25"` | **Legitimate comparison prompts**, not hardcoded simulation outputs. Kept. |
| `Lammertsma`, `Hume`, `Farde`, `Innis`, `Price`, `Koeppe`, `Gunn` in comments and printed references | **Legitimate citations**. Each is traceable (see §7). Kept. |
| `"real-data validation"`, `"REAL-DATA"`, `"REAL DATA ANALYSIS"`, `"canonical identifiability hierarchy"` | **Overclaim** — aggregated published V_T/BP ≠ dynamic TAC; coordinate-descent bias ≠ canonical identifiability result. **Downgraded** (see §7). |
| `"PMOD"` | No matches. |
| `"complete pipeline"` | No matches. |
| `"validated"` | Appears only inside ``LITERATURE_VALIDATION.md`` as `"✓ Not validated against in-vivo test-retest..."` style *limitation disclaimers*. Kept as-is. |

No suspicious hardcoded simulation outputs were found in the code: the
values `2.25` and `2.00` appear only as *expected* values in acceptance
tolerances, and they equal the closed-form algebraic result
`V_T = (K1/k2)(1 + k3/k4) = 2.25`, `BP_ND = k3/k4 = 2.00`.

## 6. Monte Carlo audit (`pet_fit_montecarlo.sio`)

- **Determinism.** Per-realisation seed is `(12345 + rep * 1009) as i64`
  → fully deterministic and reproducible. Same command produces the same
  numbers on every run.
- **Randomness.** LCG (Knuth constants) + Irwin-Hall (sum of 12 uniforms
  − 6) for an approximate N(0,1). Standard, not a fake PRNG.
- **Fitter does real work.** Starting guess `(0.12, 0.25, 0.07, 0.04)`
  differs from truth `(0.15, 0.20, 0.10, 0.05)` and from recovered means
  `(0.155, 0.189, 0.076, 0.041)` — the fit is not simply returning the
  initial guess, nor the priors.
- **Convergence criterion.** `rel_improve < 1.0e-5` after each coordinate
  sweep, max 30 iterations. `20 / 20 converged` means every run hit that
  plateau criterion; it does **not** mean every run hit the global
  minimum. The observed bias on `k3` and `k4` (≈ −24 % and −17 %) is
  consistent with plateauing of a 9-point discrete multiplier search on
  a poorly-conditioned ridge — an artefact of this particular optimiser.
- **Conclusion.** The MC demonstrates that the *pipeline* (noise gen →
  RK4 sim → coordinate-descent fit → statistics) runs end-to-end and
  that macroscopic combinations (V_T, BP_ND) have lower CV than
  individual rates `k3, k4` **for this optimiser and noise model**.
  This is an observation, not a clinical identifiability result.
- **Claim in the file downgraded** accordingly (see §7).

## 7. Literature / real-data audit

### `pet_lammertsma1996_analysis.sio`

- **What the data are.** Published **aggregate kinetic metrics** (V_T
  for cerebellum and striatum, and BP_ND summary values, method A 2TCM)
  for 8 normal subjects, transcribed to two decimal places from
  Lammertsma 1996, Tables 2 and 3.
  **Not** raw dynamic TAC time series.
- **Citation.** Exact: Lammertsma AA *et al.*, *J Cereb Blood Flow
  Metab* 1996; 16: 42–52, doi:10.1097/00004647-199601000-00005.
  Traceable to specific tables.
- **Copyright / data-use.** Aggregate summary numbers transcribed from
  a peer-reviewed 1996 paper are standard for methodological replication
  citations. No raw patient-level data, no PHI, no copyrighted figures
  reproduced.
- **What the check actually does.** Re-computes
  `BP_ND = V_T_striatum / V_T_cerebellum − 1` (Innis 2007 consensus)
  from the paper's own Table 2 V_T pairs, and compares the result to
  the paper's own Table 3 BP_ND column. Agreement (r = 0.9995,
  mean bias 0.005, max error 0.040) is expected to be essentially
  perfect up to two-decimal rounding — this is an **algebraic
  consistency check** on published aggregate summary statistics.
- **What the check does NOT do.** It does **not** fit any dynamic TAC,
  does not validate the Sounio 2TCM simulator against patient data,
  and does not exercise any sampling, noise, or identifiability claim.

### Claims downgraded in this audit

| Previous wording | Corrected wording |
|---|---|
| "REAL DATA ANALYSIS — Lammertsma 1996" | "ALGEBRAIC CONSISTENCY CHECK — Innis 2007 formula against published aggregate metrics (Lammertsma 1996, Tables 2 + 3)" |
| "REAL-DATA analysis, not a synthetic simulation" | "purely algebraic verification on published aggregate summary statistics; not a fit to real dynamic PET data" |
| "validates the Innis 2007 consensus relation ... against real clinical data" | "verifies the Innis 2007 consensus FORMULA against a published aggregate-metric table; does NOT validate any dynamic PET fit" |
| "Real-data validation against Lammertsma 1996 JCBFM" (abstract) | "Formula consistency check on published aggregate metrics" |
| "reproducing the canonical identifiability hierarchy of Lammertsma 1996 and Hume 1992" (Monte-Carlo abstract bullet) | "V_T and BP_ND show lower CV than individual k3, k4 for this particular optimiser and noise model; an observation consistent with, but not a reproduction of, the published general PET-identifiability behaviour" |
| "Real-data validation" (Framing §7 in abstract) | "Formula consistency check on published aggregate metrics" |
| "real-data reproduction of Lammertsma 1996 Table 3" (artifacts list) | "Innis 2007 formula consistency check against published aggregate V_T/BP from Lammertsma 1996" |
| README: "`pet_lammertsma1996_analysis.sio` — **Real-data** analysis: reproduces Lammertsma 1996 Table 3" | README: "`pet_lammertsma1996_analysis.sio` — algebraic consistency check of the Innis 2007 formula against published aggregate V_T/BP summary values from Lammertsma 1996 Tables 2 and 3. Not a fit to real dynamic TAC data." |

### Claims kept as-is (after review)

- "12 / 12 acceptance tests pass" — numerically verified in §4.
- "GUM ≤ 0.5 % agreement with analytic delta-method" — verified in §4.
- "Priors at centre of published [11C]raclopride human-striatum range" —
  `LITERATURE_VALIDATION.md` provides the per-parameter table with
  citations; each prior falls inside the cited range.
- "SRTM accurate under rapid equilibrium, biased under slow binding" —
  published behaviour (Lammertsma & Hume 1996; Slifstein & Laruelle 2001),
  verified numerically at the two regimes tested.
- All bibliographic citations — traceable to PubMed + DOI.

## 8. Known limitations (reproduced from README / abstract)

- Synthetic exponential plasma input; no arterial sampling, no metabolite
  correction, no delay/dispersion, no frame weighting.
- Priors are plausible and literature-anchored, **not** fitted to any
  patient dataset.
- Monte-Carlo uses a simple coordinate-descent optimiser on a discrete
  9-point multiplier grid, not Levenberg-Marquardt or TRM.
- Tracer variants other than raclopride use illustrative nominal priors;
  the code demonstrates parameter-set portability, not tracer-specific
  validation.
- `pet_lammertsma1996_analysis.sio` operates on **published aggregate
  summary statistics**, not on raw dynamic PET data.
- Not intended for clinical, diagnostic, regulatory, or dosimetric use.

## 9. Final recommended NRM wording

Short form suitable for the abstract body:

> "Proof-of-concept: an executable two-tissue compartment PET kinetic
> model in Sounio with GUM-compliant finite-difference uncertainty
> propagation across eight epistemic priors. Numerical fidelity of the
> GUM Jacobian is ≤ 0.5 % relative to the analytic delta-method on V_T
> and BP_ND variance, and the structural insensitivity
> ∂BP_ND/∂fu = ∂BP_ND/∂bbb = 0 is recovered to machine precision.
> Supplementary demonstrations include a multi-prior-set sweep across
> four literature-informed parameter sets, a side-by-side SRTM vs 2TCM
> example, and a coordinate-descent Monte-Carlo stress test on a
> synthetic noisy TAC. An algebraic consistency check of the Innis 2007
> formula against published aggregate V_T/BP values from Lammertsma
> 1996 Tables 2 and 3 is included. This is a methodological building
> block; it is not a fit to real dynamic PET data, not a clinical
> fitting package, and not equivalent to PMOD/AMIDE/PNEURO."

## 10. Acceptance criteria — status

| Criterion | Status |
|-----------|:------:|
| All `.sio` files compile and run, or failures are documented | **PASS** — 7/7 compile and run clean |
| No suspicious hardcoded simulation outputs remain | **PASS** — all numeric matches are analytic expected values |
| BP_ND and V_T GUM variance are non-zero and analytically plausible | **PASS** — 0.565 / 0.696 within 0.2 % of analytic |
| TAC AUC and peak are numerically plausible | **PASS** — 9.47 and 0.31 inside expected ranges |
| Monte Carlo claims are modest and reproducible | **PASS** (after downgrade in §7) |
| Literature claims are traceable and not overstated | **PASS** (after downgrade in §7) |
| README and abstract explicitly state this is not a clinical PET fitting package | **PASS** |

**Overall audit verdict: PASS with documented downgrades applied to
claims about "real-data validation" and "canonical identifiability
hierarchy". The slice is acceptable as NRM 2026 proof-of-concept
abstract support after the wording corrections in §7.**
