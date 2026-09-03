<!-- docs:meta
topic_id: repo.docs.dissertation.results.m6-prior-update-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.m6-prior-update-v1
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: rapamycin
model: PBPK28
status: implementation-complete
version: m6-v1
date: 2026-05-14
---

# M6 Prior Update: CL_hep Variability and Julia Binding Reconciliation

## 1. Background

M6 updates the rapamycin PBPK28 hepatic-clearance uncertainty prior used by the
dissertation PBPK28 epistemic stack. The historical prior used
`CV(CL_hep) = 0.58`, encoded as `mean = 12.4 L/h` and `variance = 51.85`.
That value came from older small-cohort pharmacokinetic context and was too
broad for the current adult transplant population-PK evidence base.

The M6 update keeps the mean unchanged and narrows only the hepatic-clearance
coefficient of variation:

```text
mean(CL_hep) = 12.4 L/h
CV_old       = 0.58
CV_new       = 0.38
variance_new = (12.4 * 0.38)^2 = 22.202944
```

The requested file `stdlib/darwin_pbpk/ep28_rapamycin_priors.sio` is not present
on this HEAD. The live prior source is:

```text
stdlib/darwin_pbpk/epistemic_pbpk28.sio
```

## 2. Literature Table

Six references were checked for access. Some DOI publisher endpoints reject
non-browser `curl` requests with HTTP 403, but each reference was reachable via
an open metadata, PubMed, PMC, Frontiers, or JPTCP page.

| Study | Access checked | n / data | Population | CV(CL/F) / IIV | CV(V/F) / IIV | Residual | M6 use |
|---|---:|---:|---|---:|---:|---:|---|
| Jiao/Zhang et al. 2009, Br J Clin Pharmacol, PMID 19660003, DOI `10.1111/j.1365-2125.2009.03392.x` | PMC/PubMed accessible | 112 / 804 troughs | De novo Chinese adult renal transplant | 23.8% | 56.7% | 29.9% | Primary adult anchor |
| Goyal et al. 2013, BBMT, DOI `10.1016/j.bbmt.2012.12.015` | PMC/DOI accessible | 40 enrolled; 37 in PopPK | Pediatric early postmyeloablative BMT | 78% PopPK IIV | 91% Vc IIV | 21% proportional + 0.84 ng/mL additive | High-variance special population, not adult anchor |
| Zhang et al. 2022, Xenobiotica, PMID 34983304, DOI `10.1080/00498254.2022.2025628` | PubMed accessible; TandF publisher page restricted | 63 / 116 troughs | Chinese adult liver transplant | Not reported in PubMed abstract | Not reported in PubMed abstract | Not reported in PubMed abstract | Supports modern adult transplant PopPK context; exact variance requires full text |
| Sabo/Marquet et al. 2021, Pharmaceutics, PMCID PMC8067051, DOI `10.3390/pharmaceutics13040470` | PMC/MDPI accessible | 42 trial patients; 27 D1 and 34 D8 PK profiles | Pediatric oncology | D8 final-model CL/F IIV 50.1% | D8 final-model Vd/F IIV 29.4% | D8 proportional residual 0.306 | Pediatric oncology context; higher variability than adult transplant |
| Fan et al. 2024, Front Pharmacol, PMCID PMC11458483, DOI `10.3389/fphar.2024.1457614` | PMC/Frontiers accessible | 49 / 134 concentrations | Children with vascular anomalies | 25.94% | V/F IIV not estimated from trough-only data | 3.41 ng/mL additive | Close CL IIV to adult anchor; pediatric setting |
| Methaneethorn et al. 2022, JPTCP 29(4):11-29, DOI `10.47750/jptcp.2022.940` | JPTCP/Crossref accessible | 20 PopPK studies | Systematic review | No pooled numeric CV reported on article page | No pooled numeric CV reported on article page | No pooled numeric residual reported on article page | Confirms bodyweight, age, and CYP3A5 as recurring clearance predictors |

Bibliographic corrections from the launch brief:

- PMID 19660003 is Jiao/Zhang et al. 2009, DOI
  `10.1111/j.1365-2125.2009.03392.x`, not `...03459.x`.
- Goyal et al. 2013 resolves as DOI `10.1016/j.bbmt.2012.12.015`.
- Zhang et al. 2022 resolves as DOI `10.1080/00498254.2022.2025628`.
- PMCID PMC8067051 is Sabo et al. 2021 in Pharmaceutics, not Therapeutic
  Drug Monitoring.

## 3. Derivation

The M6 canonical adult-transplant prior is anchored on the Jiao/Zhang 2009
adult renal transplant PopPK model because it is open, numerically explicit,
and matches the dissertation's adult transplant rapamycin use case better than
early postmyeloablative BMT or pediatric oncology cohorts.

Jiao/Zhang 2009 reports:

```text
CL/F IIV     = 23.8% = 0.238
Residual CV  = 29.9% = 0.299
Combined CV  = sqrt(0.238^2 + 0.299^2)
             = sqrt(0.056644 + 0.089401)
             = sqrt(0.146045)
             = 0.382158...
```

Rounded canonical value:

```text
CV(CL_hep)_M6 = 0.38
```

The newer open pediatric vascular-anomaly study reports CL/F IIV of 25.94%,
which is close to the Jiao/Zhang 2009 CL/F IIV component. The pediatric BMT and
oncology cohorts show substantially higher variability and are retained as
population-specific caveats rather than used to widen the adult transplant
prior. Therefore M6 should be described as an adult-transplant canonical prior,
not as a universal sirolimus variability constant.

## 4. Julia Reconciliation

The Julia reference files were read from the recovered Darwin PBPK platform:

```text
/tmp/physiology-recovery-audit/repos/agourakis82_darwin-pbpk-platform/julia-migration/src/DarwinPBPK/compartments/lipoprotein_binding.jl
/tmp/physiology-recovery-audit/repos/agourakis82_darwin-pbpk-platform/julia-migration/src/DarwinPBPK/compartments/blood_binding.jl
```

The Julia lipoprotein database defines sirolimus with:

```text
kp_hdl = 10.0
kp_ldl = 18.0
kp_vldl = 15.0
logP = 4.3
fu_reference = 0.08
```

For the Julia normal lipoprotein profile:

```text
HDL-C  = 55 mg/dL
LDL-C  = 100 mg/dL
VLDL-C = 20 mg/dL
```

the implemented partition equations give:

```text
f_hdl               = 0.002696
f_ldl               = 0.014120
f_vldl              = 0.002647
f_total_lipoprotein = 0.019464
f_free_lipoprotein  = 0.980536
fu_adjusted         = 0.08 * 0.980536 = 0.078443
```

Comparison to the Sounio prior:

```text
Sounio fu_plasma = 0.080000
Julia adjusted   = 0.078443
relative delta   = 1.9464%
```

Verdict: consistent within the requested 5% tolerance. No `fu_plasma` prior
change is warranted. The Julia blood-binding module also models RBC partitioning
and hematocrit-dependent blood-to-plasma behavior, but there is no sirolimus-
specific transplant disease-state lipoprotein profile in the recovered Julia
source. Disease-state lipoprotein refinement remains future work.

## 5. Implementation

Changed source files:

```text
stdlib/darwin_pbpk/epistemic_pbpk28.sio
stdlib/darwin_pbpk/epistemic_pbpk28_hessian.sio
stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio
scripts/ci/verify_compiler_pin.sh
```

Numerical prior edit:

```text
v[0]: 51.85 -> 22.202944
```

The hepatic clearance mean, confidence weight, and all other priors are
unchanged. The Hessian dual-rho selftest range was updated because its previous
normalized rho expectation was explicitly tied to the deprecated 58% CV prior.
Under M6 the observed CL_hep normalized rho is 0.072365, so the selftest now
checks the M6 editorial range `[0.05, 0.20]`.

The compiler pin helper was added on this branch because it was absent on this
HEAD but required by the launch contract. It verifies the wrapper-selected
native compiler:

```text
/workspace/sounio/bin/souc-linux-x86_64
sha256 = 3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

## 6. New Canonical Numbers

> **Engine dependency (verified 2026-08-17).** This section is dated 2026-05-14, a month
> before `bin/souc` switched its default engine to Madaros (2026-06-14). The command below was
> accurate when written and is silently wrong today: under default Madaros, this file compiles
> clean but **crashes at runtime with `rc=182`** (`madaros: handles full`, a resource-ceiling
> abort) partway through the N=2000 Monte Carlo loop. It still runs to completion (`rc=0`,
> `PASS`) under `SOUNIO_SOUC_ENGINE=lean_single`. The numbers below have not been reproduced
> under the project's current default engine.

Command:

```bash
export SOUC_BIN=/workspace/sounio/bin/souc
bash scripts/ci/verify_compiler_pin.sh
/workspace/sounio/bin/souc run stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio
```

MC settings:

```text
N = 2000
seed = 1729
prior family = LogNormal
dt = 0.5 h
```

Updated M6 canonical output:

| Quantity | M6 value | Legacy value |
|---|---:|---:|
| AUC_ref (GUM) | 0.516790 mg.h/L | 0.516790 mg.h/L |
| u_GUM | 0.228175 mg.h/L | 0.317093 mg.h/L |
| u_Hessian | 0.295160 mg.h/L | 0.464032 mg.h/L |
| MC mean AUC | 1.086296 mg.h/L | 1.204004 mg.h/L |
| u_MC | 0.357945 mg.h/L | 0.549197 mg.h/L |
| MC variance | 0.128124623 (mg.h/L)^2 | 0.301617 (mg.h/L)^2 |
| rel_GUM | 0.362543 | 0.422624 |
| rel_Hess(LogNormal) | 0.175405 | 0.155073 |
| epsilon(Y) / rho_normalized(CL_hep) | 0.072365 | 0.169 legacy-doc value |
| rho_literal(CL_hep) | 0.380433 | 0.581 legacy-doc value |
| brain/blood ratio at 24 h | 0.016587 | 0.016587 |

Sensitivity shares derived from the M6 Hessian CSV first-order budget:

| Parameter | Share |
|---|---:|
| CL_hep | 0.697243 |
| CL_renal | 0.000452 |
| fu_plasma | 0.301780 |
| Kp_brain | 0.000001 |
| Kp_liver | 0.000498 |
| Kp_kidney | 0.000024 |
| Kp_adipose | 0.000003 |

The first-order PBPK28 selftest independently reports `max(sensitivity) =
0.697376`, consistent with CL_hep dominance.

## 7. Comparison and Interpretation

M6 lowers absolute uncertainty:

```text
u_MC: 0.549197 -> 0.357945 mg.h/L
ratio: 0.651761
```

However, the Hessian/MC residual does not improve:

```text
rel_Hess: 0.155073 -> 0.175405
```

This is an important negative result. The master briefing prediction that
`rel_Hess(LogNormal)` would drop to 5-8% did not hold in the executed code.
The M6 prior update therefore does not close the Section 4.9.9 <=10% criterion
at canonical `dt = 0.5 h`. M5 fourth-order / higher-moment closure remains
essential rather than merely cosmetic.

Canonical update statement:

```text
The canonical M6 prior is CV(CL_hep) = 0.38.
The legacy Schreiber/Ferron-era CV=0.58 result is retained only as historical
comparison. All downstream PBPK28 numerical results should be reported under
the M6 prior, with the caveat that rel_Hess(LogNormal) remains above 10%.
```

## 8. Section 4.7.2 Prose Draft

The previous specification of CV(CL_hep) = 0.58 was inherited from older
small-cohort sirolimus pharmacokinetic evidence. While historically important,
that prior is no longer the best adult-transplant variance estimate for the
PBPK28 rapamycin case study.

Post-1991 population pharmacokinetic literature provides a modern basis for
hepatic-clearance variability. Jiao/Zhang et al. (2009; PMID 19660003; n=112
Chinese adult renal transplant recipients) report CL/F inter-individual
variability of 23.8% and residual variability of 29.9%. Combining these terms
by variance quadrature gives
`sqrt(0.238^2 + 0.299^2) = 0.382`, rounded to a canonical
`CV(CL_hep) = 0.38`. More recent studies support this as a reasonable adult
transplant anchor while showing that pediatric BMT, oncology, and vascular-
anomaly populations may require population-specific priors.

Reconciliation with the candidate's preliminary Julia implementation
(`julia-migration/src/DarwinPBPK/compartments/lipoprotein_binding.jl` and
`blood_binding.jl`) demonstrates consistency of the plasma unbound fraction.
The Julia lipoprotein binding database uses `fu_reference = 0.08` for
sirolimus. Applying the Julia normal-lipoprotein profile gives an adjusted
`fu_plasma = 0.078443`, only 1.95% below the Sounio prior value of 0.08 and
within the 5% metrological tolerance. The Julia implementation therefore
confirms the `fu_plasma` prior but does not justify changing it in M6.

Hence the canonical M6-updated prior is:

```text
mean(CL_hep): unchanged
CV(CL_hep): 0.38
legacy CV(CL_hep): 0.58, deprecated but preserved for historical comparison
fu_plasma: unchanged at 0.08
```

The dissertation should report downstream GUM-Hessian, Sobol/PCE, and Monte
Carlo results under the M6-updated prior. The updated prior reduces the absolute
Monte Carlo standard uncertainty from 0.549197 to 0.357945 mg.h/L, but the
canonical LogNormal Hessian/MC residual remains 17.5405%. Therefore M6 narrows
the prior but does not by itself satisfy the <=10% Hessian/MC criterion; M5
higher-moment closure remains a required follow-up.

## 9. Validation and Offload

Validation artifacts:

```text
docs/dissertation/results/runs/m6_literature_access_v1.txt
docs/dissertation/results/runs/m6_julia_reconciliation_v1.txt
docs/dissertation/results/runs/m6_epistemic_pbpk28_v1.txt
docs/dissertation/results/runs/m6_hessian_pbpk28_v1.txt
docs/dissertation/results/runs/m6_full_stack_v1.txt
docs/dissertation/results/runs/m6_dissertation_pbpk_suite_gate_v1.txt
docs/dissertation/results/runs/m6_dissertation_pbpk28_parity_gate_v1.txt
docs/dissertation/results/runs/m6_dissertation_pbpk_hessian_gate_v1.txt
```

Compiler pin:

```text
verify_compiler_pin: PASS
sha256=3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

Gates:

```text
epistemic_pbpk28: PASS
epistemic_pbpk28_hessian: PASS
pbpk28_mc_cross_validation: PASS
dissertation_pbpk_suite_gate: PASS (50/50)
dissertation_pbpk28_parity_gate: PASS
dissertation_pbpk_hessian_gate: PASS (5/5)
```

Offload review:

```text
Status: WAIVED in this session because keys are absent
Target: docs/dissertation/results/m6_prior_update_v1.md
Policy: external-facing dissertation artifact + math/PK claims
Log: .claude/llm_offload_log.md, 2026-05-14 M6 PBPK28 prior update row
Required follow-up: re-run fan-out review when keys are available before
external dissertation submission.
```

Gate marker:

```text
M6_PRIOR_UPDATED_PASS
```
