<!-- docs:meta
topic_id: repo.docs.dissertation.results.determinism-audit-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.determinism-audit-v1
-->

# PBPK28 MC Numerical-Determinism Audit — v1

**Date:** 2026-05-13  
**Branch:** `codex/pbpk28-numerical-determinism-audit`  
**Purpose:** Characterise and resolve the u_MC discrepancy between E1 (standalone
MC cross-validation, rel_Hess = 0.155) and E4 (prior-family sweep, LogNormal
rel_Hess = 0.026) reported in the prior-evolution sprint summary v1.  
**Status:** Root cause identified by code inspection; fix pending (D3).

---

## 1. Background

The prior-evolution sprint produced:

| Harness | u_MC (mg·h/L) | u_Hessian | rel_Hess |
|---|---|---|---|
| E1: `pbpk28_mc_cross_validation.sio` | 0.549197 | 0.464032 | 0.155 |
| E4: `pbpk28_mc_prior_family_sweep.sio` (LogNormal) | 0.476630 | 0.464032 | 0.026 |

Both runs use: seed = 1729, N = 2000, LogNormal prior (`ep28_rapamycin_priors()`),
same base parameters (`ep28_rapamycin_params()`), same AUC integrator
(`pbpk28_full_cn_step`, Crank-Nicolson, dt = 0.5 h, t ∈ [0, 168] h).

Same inputs + same seed + same algorithm should produce the same u_MC. The
discrepancy (0.549 vs 0.477, Δ = 14.8 %) is systematic, not stochastic.

The prior sprint self-audit attributed this to a "JIT compilation context effect."
**This attribution is incorrect.** Sounio is fully native-compiled; context cannot
alter arithmetic.

---

## 2. Reproducibility measurements

### 2.1 Intra-process determinism

The probe binary (`pbpk28_mc_determinism_probe.sio`) calls the lognormal MC
function three times in sequence within a single compiled process (seed = 1729,
N = 2000, CL_hep ~ LN(mean=15, var=77)):

| Run | exp_buggy mean | exp_buggy sd | exp_correct mean | exp_correct sd |
|---|---|---|---|---|
| 1 | 12.494538 | 7.224006 | 14.740416 | 8.412056 |
| 2 | 12.494538 | 7.224006 | 14.740416 | 8.412056 |
| 3 | 12.494538 | 7.224006 | 14.740416 | 8.412056 |

**Conclusion:** Intra-process determinism is intact for both implementations.
Non-determinism hypothesis (a) is **rejected**.

### 2.2 Inter-process determinism

E1 and E4 binaries run three times each in separate processes:

| Process | E1 u_MC | E4 (LogNormal) u_MC |
|---|---|---|
| 1 | 0.549197 | 0.476630 |
| 2 | 0.549197 | 0.476630 |
| 3 | 0.549197 | 0.476630 |

**Conclusion:** Both harnesses are fully deterministic across processes.
Non-determinism hypothesis (b) is **rejected**.

### 2.3 Cross-module comparison

With identical seed, N, and priors, E1 produces u_MC = 0.549197 and E4
LogNormal produces u_MC = 0.476630, a 14.8 % systematic difference.

**Conclusion:** The divergence is cross-module (hypothesis c), caused by
different implementations of the `exp()` function in the two harnesses.

### 2.4 RNG isolation (D2)

Each call to `ms28_run_family` in E4 initialises `var rng = ms28_rng_new(seed)`
independently (lines 360, 366, 372 of sweep harness all pass `seed = 1729`).
No RNG state is shared across the three family calls.

**Conclusion:** Per-family RNG isolation is already present. No code change
required for D2.  
**Gate:** `MC_PBPK28_DETERMINISM_RNG_ISOLATED_PASS`

---

## 3. Root cause — Taylor-series defect in `ms28_exp`

### 3.1 Diagnosis

Code inspection of `pbpk28_mc_prior_family_sweep.sio` (E4), function `ms28_exp`,
line 81:

```sounio
// E4 (BUGGY) — pbpk28_mc_prior_family_sweep.sio:81
var t = rx          // wrong: starts Taylor series at rx, not 1.0
```

```sounio
// E1 (CORRECT) — pbpk28_mc_cross_validation.sio:99
var t: f64 = 1.0    // correct: starts Taylor series at constant 1
```

Both use range reduction: `rx = x − n·ln2`, so `rx ∈ (−ln2, ln2)` ≈ (−0.693, 0.693).

**Correct** Taylor series for exp(rx):

    result = 1 + rx + rx²/2! + rx³/3! + ...

**Buggy** Taylor series (E4, `var t = rx`):

    t = rx; k=1: t ← rx·rx/1 = rx²; k=2: t ← rx³/2; ...
    result = 1 + rx² + rx³/2! + rx⁴/3! + ...

The first term `rx` is missing. The buggy implementation computes:

    f_buggy(rx) = 1 + rx·(exp(rx) − 1)

instead of `exp(rx)`.

### 3.2 Quantified error

| x | exp_correct(x) | exp_buggy(x) | relative error |
|---|---|---|---|
| 0.0 | 1.000000 | 1.000000 | 0 |
| 0.5 | 1.648721 | 1.324361 | 19.7 % |
| 1.0 | 2.718282 | 2.220407 | 18.3 % |
| 2.0 | 7.389056 | 6.079883 | 17.7 % |
| −0.5 | 0.606531 | 1.196735 | −97.3 % |

For `x = 1.0`, the relative error is **18.3 %**.

Note on `exp_buggy(−0.5)`: the range-reduction loop drives `rx` toward zero from
the negative side; the missing linear term causes the result to exceed 1 even for
negative input, which is qualitatively wrong.

### 3.3 Effect on lognormal sampling

The `ms28_sample_lognormal` function computes:

    sample = ms28_exp(mu_log + sig_log · z)

where `mu_log = ln(mean) − σ²_log/2` and `z` is a standard normal deviate.

For CL_hep ~ LN(mean = 15, var = 77, CV = 58 %):
- mu_log ≈ 2.44, sig_log ≈ 0.54
- Typical argument to exp: 2.44 ± 0.54·|z|, range ≈ [1.4, 3.6]
- After range reduction n = 2–5, rx ≈ 0.3–0.5 → error ≈ 17–20 %

The buggy exp systematically underestimates sample values, compressing the
entire lognormal distribution:

| Statistic | exp_buggy | exp_correct | ratio |
|---|---|---|---|
| Sample mean of CL_hep | 12.49 L/h | 14.74 L/h | 0.848 |
| Sample SD of CL_hep   | 7.22 L/h  | 8.41 L/h  | 0.859 |

The compressed distribution propagates through the PBPK28 ODE, producing
systematically lower AUC values and hence lower u_MC.

### 3.4 Why E1 is unaffected

E1 uses `mc28_exp` (defined in `pbpk28_mc_cross_validation.sio`) with
`var t: f64 = 1.0` — the correct initialisation. The `ms28_` prefix in E4 was
introduced to avoid symbol clashes when both harnesses are compiled separately;
it was not a deliberate implementation divergence, it was a copy-paste error
in the initial value of `t`.

---

## 4. Classification

| Hypothesis | Description | Verdict |
|---|---|---|
| (a) Intra-process | RNG state shared across calls in same binary | REJECTED — intra-process deterministic |
| (b) Inter-process | Env vars, allocator, ASLR | REJECTED — inter-process deterministic |
| **(c) Cross-module** | **Different exp implementations in two harnesses** | **CONFIRMED** |
| (d) Compiler | FMA contraction, unsafe-math flags | REJECTED — no such flags in Sounio |

**Conclusion:** The non-determinism is (c) cross-module: `ms28_exp` in E4 has a
Taylor-series defect (missing linear term) not present in `mc28_exp` in E1.

---

## 5. Corrective actions

| Deliverable | Action | File |
|---|---|---|
| D3 | Fix `var t = rx` → `var t: f64 = 1.0` in `ms28_exp` | `pbpk28_mc_prior_family_sweep.sio:81` |
| D3 | Add Welford accumulator to both harnesses (numerical stability) | both harnesses |
| D4 | Document Sounio IEEE 754 posture | `docs/compiler/numerical_determinism.md` |
| D5 | Re-run both harnesses and verify LogNormal u_MC match | `results/runs/` |

After D3, both harnesses should produce the same u_MC for LogNormal, seed = 1729,
N = 2000. The `--post-fix` mode of `scripts/audit/mc_determinism_probe.sh`
asserts agreement to 6 significant figures.

---

## 6. Gate markers from D1/D2

```
PBPK28_MC_DETERMINISM_PROBE_PASS
MC_PBPK28_DETERMINISM_RNG_ISOLATED_PASS
```
