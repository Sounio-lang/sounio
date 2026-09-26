# Venlafaxine CYP2D6 PGx — structural audit of the parent → ODV model

Date: 2026-09-26 · Branch: `claude/pbpk28-cn-portal-routing` · Status: **findings +
proposal; no scenario or drug parameter has been changed.** The scenario lane
(`stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio`) is owned by other sessions and
the parameters are dissertation-facing, so every change below is an operator
decision.

## 1. Summary

The ODV/parent ratio of the venlafaxine scenario under-reads the CYP2D6 effect.
The shortfall is a structural defect, not a numerical one. Numerical issues were
fixed earlier: saturating explicit sink (`3c0bbc50c`), CN floor clamp (TR-BDF2,
other lane) and the gut F bookkeeping (other lane). Even with all three fixed, the
AUC ratio is PM/IM/NM/UM = 0.346 / 1.126 / 1.912 / 2.501, against the scenario's
own literature targets of 0.25 / 1.16 / 3.45 / 10.3. UM is 4.1× low.

Three causes, each measured:

1. **Oral apparent clearances are used as systemic clearances.** The 100 L/h and
   43 L/h anchors are CL/F values from oral dosing (Lessard 1999), yet the model
   places them in the systemic circulation **and** multiplies the dose by
   F = 0.45. Implied oral CL/F is 246 L/h at NM, not 100 L/h. That 246 is the
   joint effect of causes 1–3; the F double count alone gives 100/0.45 ≈ 222.
2. **There is no portal first pass.** Absorbed drug enters systemic blood, so
   hepatic formation is flow-limited: CL_H = Q·X/(Q+X) ≤ Q_liver = 90 L/h. For an
   oral drug the ratio is linear in the formation intrinsic clearance X_form
   (§3), with no saturating Q·X/(Q+X) term. That flow limit is what compresses UM
   toward NM.
3. **The sink is referenced to the wrong concentration.** The sink acts on the
   liver's average concentration C_avg. For Kp = 4.2, C_avg ≈ κ·C_v with
   κ = f + (1−f)·Kp = 3.528. The effective venous-referenced clearance is
   therefore 3.5× the literature value (43 L/h behaves as 152 L/h).

The measured correction adds portal input (new kernel capability, §4), sinks
referenced to venous concentration, Lessard's clearance split, and a
*calibrated* ODV clearance (§3). Ratios become **0.279 / 1.269 / 3.589 / 9.361**,
within about ±12% of the targets for every phenotype. How the literature
clearances resolve into intrinsic clearances moves the absolute values by about
20% (§5.2): UM spans 7.7–9.4 across the variants, against 2.5 today.

The absolute NM value is **not** identified by the cited abstracts. The one direct
observation, AUC_ODV/AUC_V = 2–3 in healthy EMs (Klamerus 1992), sits below
config C (3.59) and brackets variant E (2.87). The ODV systemic clearance is the
open parameter (§3, P5). What is robust is the structure: with portal input the
ratio grows linearly with CYP2D6 activity; without it, it saturates.

## 2. Literature audit (PubMed-verified, 2026-09-26)

| Code says | Actually | Evidence |
|---|---|---|
| "Klamerus 1999, Pharmacogenetics 9:435-43" (CL_form 43 L/h, CL/F 100 L/h) | **Lessard E et al. 1999**, Pharmacogenetics 9(4):435-43, PMID 10780263 | Abstract: EM oral CL 100 ± 62 L/h; metabolic clearance to O-desmethylated metabolites 43 ± 32 L/h; with quinidine these fall to 17 ± 5 and 2 ± 1 L/h; renal CL 4 ± 1 L/h; PM oral CL "more than fourfold less". Every value is an **oral** (apparent) clearance at steady state. |
| "Kirchheiner 2006, Ther Drug Monit 28:493-502" (ratios EM 3.45, IM 1.16, PM 0.25, UM 10.3) | No such paper. The paper at 493-502 in 2006 is **Shams ME et al.**, J Clin Pharm Ther 31(5):493-502, PMID 16958828, doi:10.1111/j.1365-2710.2006.00763.x | Abstract (100 patients): median ODV/V 1.8, 10th–90th percentile 0.3–5.2; PM < 0.3; UM > 5.2; IM 1.1 ± 0.8. The four target values are **unsourced**. A separate forensics pass (`docs/audit/VENLAFAXINE_ODV_RATIO_CITATION_FORENSICS_2026-09-26.md`, branch `claude/mystifying-goldberg-ccfc8a`) found them in no publication, CPIC 2023 or DPWG 2024; they entered the repo already carrying the bogus citation. |
| — | Nichols AI et al. 2011, Clin Drug Investig 31(3):155-67, PMID 21288052, doi:10.2165/11586630-000000000-00000 | Venlafaxine XR 75 mg, genotyped healthy subjects: ODV:V AUC(∞) ratio **6.2 in EMs, 0.21 in PMs**; venlafaxine AUC 445% higher in PMs. |
| Klamerus (venlafaxine PK) | Klamerus KJ et al. 1992, J Clin Pharmacol 32:716-24, PMID 1487561, doi:10.1002/j.1552-4604.1992.tb03875.x | Venlafaxine t½ 3–4 h, ODV t½ 10 h; ODV **apparent** clearance 0.21–0.66 L/h/kg; ODV AUC 2–3× venlafaxine AUC. |
| ODV CL 28 L/h ("Wyeth label 0.4 L/h/kg") | Klamerus KJ et al. 1996, Pharmacotherapy 16:915-23, PMID 8888087 | Young adults: ODV apparent CL 0.38 L/h/kg, t½ 10.3 h. |
| F = 0.45 ("Wang 2022") | Patat A et al. 1998, J Clin Pharmacol 38:256-67, doi:10.1002/j.1552-4604.1998.tb04423.x | Absolute bioavailability 40–45% (IR and XR alike). |
| — | Troy SM et al. 1996, J Clin Pharmacol 36:175-81, doi:10.1002/j.1552-4604.1996.tb04183.x | Venlafaxine renal CL 0.053 L/h/kg ≈ 3.7 L/h. |

The **direct** literature anchor for the EM ratio is Klamerus 1992: ODV AUC is
2–3× venlafaxine AUC in healthy young men. If CL_ODV,app is the parent dose over
AUC_ODV, then AUC_ODV/AUC_V = (CL/F)_V/CL_ODV,app by definition. That gives
100/26.6 = 3.76 (Lessard 1999, Klamerus 1996) or 1.3/0.4 = 3.25 (label apparent
clearances). Those figures come from different studies and depend on how each
defined "apparent" ODV clearance, so they are weaker than the direct 2–3. The
code's 3.45 lies just above that range. The patient median of 1.8 in Shams 2006
is a mixed, phenoconverted population (Preskorn 2013, J Clin Psychiatry 74:614:
24% phenoconversion to PM), not an EM reference.

## 3. Pharmacology (well-stirred liver, oral dose)

Let X be the hepatic intrinsic clearance referenced to the venous (outflow)
concentration and Q the liver blood flow. At quasi-steady state:

    F_H = Q/(Q+X),   CL_H = Q·X/(Q+X),   CL/F (all-hepatic, F_abs = 1) = CL_H/F_H = X

With F_abs = 1 and no renal clearance, Lessard's oral clearances **are** the X
values. In general, with renal clearance CL_R on the systemic side:

    CL/F = X/F_abs + CL_R/(F_abs·F_H)

Solving this together with F = F_abs·F_H = 0.45 (Patat 1998), CL/F = 100 and
CL_R = 4 (Lessard 1999), and Q = 90 gives the self-consistent X_hep = 75.3 L/h,
F_H = 0.544 and F_abs = 0.827 (§5.2, variant E). With F_abs = 1 the same
equation gives X_hep = 96·90/94 = 91.9 L/h.

Configuration C below is a simpler, **approximate** reading. It takes
X_hep = 100 − 4 = 96, which is 4% above the F_abs = 1 solution (infinite-PS
CL/F would be 104.3; finite PS brings the realised value to 95.7). It is split
as:

- X_form = 43 L/h (CYP2D6 → ODV);
- X_other,2D6 = 40 L/h (not recovered as O-desmethylated metabolites);
- X_non-2D6 = 13 L/h;
- renal CL = 4 L/h, systemic.

Read consistently, the "incomplete inhibition" interpretation of the 2 L/h of
O-desmethylation that survives quinidine gives 43 s │ 42 s + 11. The 2 L/h is
inside the CYP2D6 term, so it must leave the quinidine-resistant 17. C's
40 s + 13 therefore holds 2 L/h in the non-scalable floor: NM is unaffected, and
PM's total X is about 2(1 − s) L/h high. The alternative reading, 41 s + 2 │
42 s + 11 (the residual is CYP2D6-independent), is variant D in §5.2.

**Oral ratio.** For the well-stirred liver with infinite PS, F_abs = 1, formed ODV
fully available and renal clearance on blood, the exact result is

    AUC_ODV/AUC_V = X_form·(1 + CL_R/Q)/CL_ODV

(derivation: formed = D·(X_form/X)·[(1−F_H) + F_H·CL_H/(CL_H+CL_R)] and
AUC_V = D·F_H/(CL_H+CL_R), with F_H = Q/(Q+X)). It is linear in X_form, which is
what CYP2D6 phenotype changes, and has no saturating Q·X/(Q+X) term. The only Q
dependence is the (1 + CL_R/Q) factor. When the input bypasses the liver (the
current model), the ratio follows Q·X/(Q+X) and saturates at Q. Finite PS makes
the model mildly nonlinear: config C's UM/NM is 9.361/3.589 = 2.61, against
s_UM = 2.99.

In PBPK28 the sink flux is cl_sink·C_avg. At equilibrium C_avg = κ·C_v, so the
venous-referenced clearance is X = κ·cl_sink, and a literature X enters the kernel
as cl_sink = X/κ. Finite PS lowers the realised value: under the sink,
C_t/(Kp·C_v) = 0.909, so X_eff/X = [f + 0.909·(κ − f)]/κ = 0.914 and
X_eff = 87.8 L/h for X = 96 L/h.

**ODV clearance: a calibration, not a measurement.** The model uses 28 L/h
(0.4 L/h/kg), an apparent value. Configs C/E use CL_ODV = 0.43 × 26.6 = 11.4 L/h:
here 0.43 = X_form/(X_hep + CL_R) = 43/100, and 26.6 L/h is Klamerus 1996's
0.38 L/h/kg. That choice forces the NM ratio toward (CL/F)_V/CL_ODV,app by
construction, so it is **not** a measured systemic clearance. The standard
conversion is CL = F_ODV·(CL/F)_ODV, and it needs ODV's own bioavailability.
Alternatively, calibrate CL_ODV so that the NM ratio matches the direct Klamerus
1992 observation of 2–3. The cited abstracts identify neither; see P5.

## 4. New kernel capability (this branch)

`pbpk28_full_cn_step_routed_mut(state, p, rel_mid, input_organ, dt, sink_organ,
cl_sink)` in `stdlib/darwin_pbpk/tsit5_pbpk28.sio`:

- `rel_mid` enters the vascular space of `input_organ`; organ 1 is the liver,
  i.e. portal delivery. The input adds dt·rel_mid/V_v to that organ's explicit
  side.
- With the default `input_organ = 0` (blood route), existing callers are
  bit-identical.

`tests/run-pass/darwin_pbpk28_portal_first_pass.sio` checks, to 1e-9 relative, at
the CN fixed point: F_H = Q/(Q+X), CL_H = Q·X/(Q+X), CL/F = CL_sys/F_H, and
steady-state mass balance for both routes. It also checks bit-identity of the
blood route. It passes on lean_single and on Madaros (built from 98315edcdb, via
`bin/souc`) with identical digits: X = 87.7685 L/h, CL_sys = 48.4351 L/h,
F_H = 0.506276, CL/F = 95.6694 L/h.

## 5. Measurements

Method: under linear kinetics the single-dose AUC ratio equals the C_ss ratio for
a constant input through the same route. The probe uses constant infusion: 400 h,
dt = 0.05 h, 0 clamped states in every run, dt-independent. The single-dose oral
runs give the same ratios to 3 digits.

Probe: `docs/audit/repro/venlafaxine_pgx_structure_ss.sio`.

| | PM | IM | NM | UM |
|---|---|---|---|---|
| A: current (systemic input × F, CL_c 57, sink 43·s on C_avg, ODV CL 28) | 0.346 | 1.126 | 1.912 | 2.501 |
| B: portal + X/κ + Lessard split, ODV CL 28 | 0.114 | 0.518 | 1.466 | 3.824 |
| **C: B with ODV CL = 11.4** | **0.279** | **1.269** | **3.589** | **9.361** |
| Targets (code; provenance §2) | 0.25 | 1.16 | 3.45 | 10.3 |
| C: oral CL/F, L/h (F_abs = 1) | 23.5 | 45.1 | 95.7 | 221.5 |
| C: F_H | 0.828 | 0.695 | 0.506 | 0.301 |
| A: oral CL/F implied, L/h | 148 | 197 | 246 | 282 |

Config A reproduces the TR-BDF2 scenario on `claude/competent-mcclintock-b137de`
digit for digit (0.346539 / 1.126788 / 1.912260 / 2.501897). That is a
cross-check between two kernels and two independent implementations.

### What is independent evidence and what is not

- **Not independent:** the phenotype scale s = r_i/3.45 is *defined* from the
  target ratios. Config C tracks the targets because its ratio is nearly
  proportional to X_form (0.279/3.589 = 0.078 vs s_PM = 0.072). This shows the
  structure *can* express the scale; config A cannot.
- **Not independent — NM ratio:** 3.589 follows from the ODV-clearance
  calibration of §3, which is built to reproduce (CL/F)_V/CL_ODV,app. It is above
  the direct Klamerus 1992 range (2–3); variant E (2.87) is inside it.
- **Independent 1 — EM/PM oral CL/F:** 95.7/23.5 = **4.07**. Lessard 1999: PM oral
  clearance "more than fourfold less". The PM arm of Lessard was not used.
- **Independent 2 — bioavailability:** in the self-consistent solve (variant E),
  F = 0.45 (Patat 1998), CL/F = 100 and CL_R = 4 give F_H = 0.544 and
  F_abs = 0.827, a physically admissible value (≤ 1). Config C's 0.45/0.506 =
  0.89 is not a check, because that F_H was produced under F_abs = 1. Under
  config A the question cannot be posed.
- **Independent 3 — the phenotype scale itself.** With portal input the ratio is
  linear in X_form (§3), so literature ratios fix s directly. Two independent
  sources agree: Nichols 2011 gives s_PM ≈ 0.21/6.2 = 0.034, and Lessard 1999
  (quinidine: O-desmethylation 43 → 2 L/h) gives 2/43 = 0.047. Both are below the
  code's s_PM = 0.25/3.45 = 0.0725, which comes from the unsourced targets.
  Correspondingly the model's EM/PM ratio contrast (C: 3.589/0.279 = 12.9; E: 13.1)
  is well below Nichols 2011's 6.2/0.21 = 29.5. The EM absolute ratio is not
  pinned either: Klamerus 1992 gives 2–3 (IR, healthy men), Nichols 2011 gives
  6.2 (XR, genotyped EMs), Shams 2006 a patient median of 1.8.
- **Conditional — PM ratio vs the clearance split** (§5.2): Shams 2006 reports
  PM < 0.3. This discriminates the split only given the PM scale s = 0.25/3.45,
  which comes from the unverified targets. As s → 0, variant D's 2 L/h floor
  alone gives ≈ 2/11.4 = 0.18 < 0.3.
- **Prediction (untested here):** F_oral(PM)/F_oral(NM) = 0.828/0.506 = 1.64, a
  higher PM bioavailability that follows from first pass.

### 5.2 Sensitivity to how the literature clearances resolve

Same probe with the variant configurations (sources in the probe header):

| Variant | PM | IM | NM | UM | NM oral CL/F (F_abs = 1) |
|---|---|---|---|---|---|
| C: split 43 s │ 40 s + 13 (approximate, §3); ODV 0.38 L/h/kg | 0.279 | 1.269 | 3.589 | 9.361 | 95.7 |
| D: split 41 s + 2 │ 42 s + 11 (quinidine residual CYP2D6-independent) | **0.445** | 1.385 | 3.589 | 9.071 | 95.7 |
| E: C with every X × 75.3/96 (self-consistent F, CL/F, CL_R, Q) | 0.219 | 1.003 | 2.868 | 7.671 | 77.3 (÷ F_abs 0.83 = 93) |
| F: C with ODV 0.40 L/h/kg (label) | 0.265 | 1.205 | 3.410 | 8.893 | 95.7 |
| A: current model | 0.346 | 1.126 | 1.912 | 2.501 | 246 |

- **D vs Shams is conditional:** PM 0.445 against Shams 2006's PM < 0.3, given
  s_PM = 0.0725 (see §5).
- **E is the self-consistent resolution.** It lowers every absolute ratio by
  about 20%; NM 2.87 lies inside the direct Klamerus 1992 range of 2–3.
- **Robust across all variants:** UM ≥ 7.7 and UM/PM ≥ 20 (C 33.6, D 20.4,
  E 35.0, F 33.6), against 7.2 today.

## 6. Other findings

- **Distribution volumes are too small.** Model Vss is 107.8 L for venlafaxine and
  86.4 L for ODV; these are exact sums of V_i(f_i + (1−f_i)Kp_i). Literature
  half-lives with the clearances above imply V_z ≈ t½·CL/ln 2:
  - venlafaxine: t½ 3–5 h, CL_sys ≈ F·CL/F ≈ 45 L/h → 195–325 L;
  - ODV: 10.3 h × 11.4 L/h → ≈ 170 L.

  The Kp sets need recalibration for the time course, Css fluctuation and the XR
  profile. AUC ratios are unaffected, since they depend on clearances only.
- **The CN floor clamp biases IV-bolus AUCs.** Venlafaxine config B, 75 mg IV
  bolus, CN dt = 0.05 h: 12,160 zeroed-state events, D/AUC = 35.96 L/h vs the
  exact steady-state CL of 48.44 L/h, an **AUC +34.7%**. Oral and infusion inputs
  on the same parameters produce 0 events. Reported to the floor-clamp dispatch
  (`docs/audit/PBPK28_CN_FLOOR_CLAMP_MASS_INJECTION_DISPATCH_2026-09-26.md`,
  other lane).

## 7. Proposal (operator decisions)

| # | Change | Where | Lane |
|---|---|---|---|
| P1 | Correct citations (Lessard 1999; Shams 2006). Treat 0.25 / 1.16 / 3.45 / 10.3 as unsourced and **replace the phenotype scale s**. It should not be r/3.45; derive it from formation-clearance data: s_PM ≈ 0.03–0.05 from Nichols 2011 and Lessard 1999; IM/UM need sourcing (e.g. an activity-score model). Validate against Nichols 2011 (EM 6.2, PM 0.21), Klamerus 1992 (2–3) and the Shams 2006 bands | `pgx/cyp2d6_venlafaxine.sio`, `drugs/venlafaxine.sio`, scenario header | pgx/drugs: mystifying-goldberg / competent-mcclintock (comments already in flight); scenario: competent-mcclintock |
| P2 | Route oral absorption to the liver (`input_organ = 1`) | scenario; TR-BDF2 step needs the same input term | modest-heisenberg (theta kernel), scenario lane |
| P3 | All hepatic clearance in the liver sink, venous-referenced: cl_sink = X_hep/κ with formed = removed·X_form/X_hep; renal 4 L/h on blood. Resolve X_hep from {F, CL/F, CL_R, Q, PS} (variant E), not by reading CL/F as X; use the consistent split 43 s │ 42 s + 11 (or D's reading, stated) | scenario + `drugs/venlafaxine.sio` | scenario lane |
| P4 | Gut factor F → F_abs (0.827 in the self-consistent solve); F_H emerges | scenario | scenario lane |
| P5 | ODV clearance 28 L/h is apparent, so replace it. Preferred: ODV's own systemic CL, i.e. F_ODV × (CL/F)_ODV from desvenlafaxine PK (to be sourced). Otherwise: calibrate so the NM ratio matches Klamerus 1992's 2–3, stated as a calibration. fm × apparent (11.4) is a calibration to (CL/F)_V/CL_ODV,app, not a measurement | `drugs/venlafaxine.sio` | free |
| P6 | Recalibrate Kp for Vss (venlafaxine ~200–320 L, ODV ~170 L) against the t½ values | `core/pbpk28_params.sio` | needs its own dispatch |

P2–P5 together are config C / variant E. P6 is independent of the ratios.

## 8. Review

- `bin/llm-offload -t math-review -p xai` (grok-4.6), 2026-09-26. Well-stirred
  identities, κ, finite-PS X_eff, the portal fixed point, the ratio identity,
  V_z, the clamp-bias arithmetic and the Table §5 relations were all judged
  correct. Its corrections are incorporated: the renal/F_abs form of CL/F (§3,
  variant E), the NM ratio relabelled as consistency (§5), the alternative
  clearance split (variant D), the 0.38 vs 0.40 L/h/kg ODV clearance (variant F),
  and the 246 L/h attribution (§1).
- Re-reviewed on the canonical route: `bin/llm-offload -t math-review` (default
  fan-out) from branch `chore/llm-offload-llmgateway-grok47`, Grok 4.7 (gateway
  timed out at 900 s; answered via xAI direct), 2026-09-26. Confirmed: the
  well-stirred block and the renal/F_abs form of CL/F, κ, the portal fixed
  point, variant E (X = 75.31, F_H = 0.544, F_abs = 0.827), the table
  relations, the clamp-bias arithmetic, V_z, and the systemic-input compression
  mechanism. Corrected here:
  - the exact oral ratio X_form(1 + CL_R/Q)/CL_ODV, replacing fm·(CL/F)/CL_ODV;
  - the ODV clearance is a calibration, not "true systemic";
  - config C's X = 96 is approximate (91.9 at F_abs = 1);
  - UM/PM is ≥ 20 with D, not 33–35;
  - the consistent incomplete-inhibition split is 43 s │ 42 s + 11;
  - F_abs is 0.827, not 0.89;
  - the X_eff/X arithmetic, and fm = 43/100;
  - D vs Shams is conditional on s;
  - the direct Klamerus 1992 anchor of 2–3 is added.
- After that review, the unsourced status of the targets (forensics branch
  above) and Nichols 2011 (PubMed-verified) were added to §2, §5 and P1. Those
  additions are ratio arithmetic on the reviewed linear-in-X_form result; they
  have not had a separate model review.
- The second vendor leg (Kimi K3 via the gateway) returned nothing: an upstream
  502, then a 900 s timeout. zai is rate-limited (1313), the on-prem `local` leg
  is down, and the DeepSeek direct key is rejected. **This document has had a
  single-vendor review.**

## 9. Reproduction

    SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/darwin_pbpk28_portal_first_pass.sio
    SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run docs/audit/repro/venlafaxine_pgx_structure_ss.sio
    SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run docs/audit/repro/venlafaxine_pgx_sensitivity_ss.sio
    # and each with SOUNIO_SOUC_ENGINE=lean_single

Run Madaros through `bin/souc` (512 MiB stack), never the raw
`artifacts/self-hosted/madaros` ELF.
