<!-- docs:meta
topic_id: repo.docs.research.mercyful-clinical-integration-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-clinical-integration-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning × clinical PK twins — treatment-sequencing scheduler

**Date:** 2026-07-25
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)
**Parents:** `docs/research/mercyful_sounio_port_spec_2026-07-25.md` (Sounio-native scheduler), `docs/research/mercyful_runtime_spec_2026-07-25.md` (Python contract, M_GREEN)
**Harness:** `tests/run-pass/mercyful_clinical_sequencing.sio`
**Gate:** `scripts/ci/mercyful_clinical_sequencing_gate.sh`
**Modules used (unmodified):** `stdlib/clinical/mercyful.sio`, `stdlib/clinical/vancomycin_pbpk.sio`, `stdlib/clinical/tacrolimus_oral_safety.sio`, `stdlib/epistemic/knightian.sio`

---

## 1. What this is

The Mercyful Learning scheduler (`mercyful_schedule`) minimizes

```
cost(γ; μ) = Σ_{(u,v)∈γ} s(u)·ℓ((u,v)) + μ · max_{v∈γ} s(v)
```

over paths `γ` that **must reach a target state** within a length budget (the anti-Goodhart constraint). Until now the suffering field `s` was hand-set (exposure-therapy toy). This rung connects the scheduler to the two clinical PK digital twins so that **the suffering field is computed from Knightian PK bands**, and **drug–drug interaction (DDI) gates act as additional Goodhart constraints on edge admission**.

Everything is synthetic and bounded: two synthetic patients (78.5 kg / CrCl 65; 70 kg / CrCl 80), fixed doses, no patient data, no dosing recommendation.

---

## 2. Suffering fields from PK constraints (Cmin, Cmax, AUC)

For a state representing a dosing regimen, the suffering value is a weighted sum of three normalized worst-case band violations. For a concentration PBox `[lo, hi]` and a window `[a, b]`:

```
s_win([lo,hi], [a,b]) = max(0, a − lo)/a + max(0, hi − b)/b
```

The first term is the worst-case sub-therapeutic shortfall (efficacy risk), the second the worst-case supra-therapeutic exceedance (toxicity risk), each normalized by the window bound. A band fully inside the window contributes 0.

### 2.1 Cmin term

Directly from the twins' public APIs:

- vancomycin: `predict_cmin_knightian(78.5, 65.0, 1000.0, 12.0, tdm)` → PBox over Cmin_ss (mg/L), window `[10, 20]` (Rybak 2020 trough screen).
- tacrolimus: `predict_c_trough_knightian(70.0, 80.0, 5.0, 12.0, tdm)` → PBox over C24h trough (ng/mL), window `[5, 15]` (Prograf label).

Both bands are Fréchet outer enclosures valid for any joint distribution of the PK parameters (monotone-corner argument, see the module headers).

### 2.2 AUC term

Per-interval steady-state AUC for the 1-compartment models is

```
AUC_ss = F · D / CL        (vancomycin IV: F = 1)
```

strictly **increasing in F** and **decreasing in CL**, so the corner enclosure is

```
AUC ∈ [F_lo·D/CL_hi,  F_hi·D/CL_lo]
```

computed from the public `vp_cl_to_pbox` / `tp_cl_to_pbox` / `tp_f_to_pbox` bands. The AUC windows used are **synthetic** (vancomycin per-interval `[250, 350]` mg·h/L; tacrolimus per-interval `[80, 200]` ng·h/mL): they are shaped after AUC-guided practice but are not a clinical target claim.

### 2.3 Cmax term (sound but loose proxy)

For the 1-compartment IV-bolus steady state there is an exact identity

```
Cmax_ss = Cmin_ss + D/Vc
```

(because `Cmax − Cmin = (D/Vc)·(1 − e^{−θ})/(1 − e^{−θ}) = D/Vc`, θ = CL·τ/Vc). Since max of a sum ≤ sum of maxes over the parameter rectangle,

```
Cmax_hi ≤ Cmin_hi + D/Vc_lo            (vancomycin)
Cmax_hi ≤ Cmin_hi + F_hi·D/Vc_lo       (tacrolimus, oral F·D swing)
```

This is a **sound outer bound, not a tight enclosure**; the spec deliberately does not claim Fréchet tightness for Cmax. The tacrolimus form inherits the same bound through the ka >> ke reduction documented in `tacrolimus_oral_safety.sio`.

For reference, the exact Cmax is *strictly decreasing in both* Vc and CL:

```
Cmax = D / (Vc·(1 − e^{−θ})),   θ = CL·τ/Vc
∂ ln Cmax/∂Vc = (1/Vc)·(θ/(e^θ − 1) − 1) < 0   ∀θ > 0
∂ ln Cmax/∂CL = −(τ/Vc)·e^{−θ}/(1 − e^{−θ}) < 0
```

(θ/(e^θ−1) < 1 for θ > 0 since e^θ > 1 + θ.) The test uses the loose bound above instead, because it needs no exponential and composes with the already-enclosed Cmin band.

### 2.4 Aggregation and contract-violation penalty

```
s(state) = s_cmin + s_auc + 0.5 · s_peak(Cmax_hi, ceiling)
```

A regimen that violates a twin's refinement-type contract (vacuous PBox, confidence 0 — e.g. dose 5000 mg > 4000 mg cap) is assigned `s = S_MAX = 100`, roughly 40× the worst in-contract suffering achievable with the synthetic windows above (≈ 1 + 1 + 0.5 per drug), so the scheduler treats contract-violating regimens as near-prohibitive without making them topologically impossible.

*(Notation note: the `800` / `700` arguments in the `is_safe_dose` / `is_safe_tac_dose` calls in §3 are the twins' structural provenance confidence thresholds (`min_conf`), not concentration or AUC ceilings — see the module headers.)*

---

## 3. Scenario graph and DDI gates as Goodhart constraints

States (suffering computed at runtime from §2):

| # | State | s |
|---|---|---|
| 0 | START (untreated) | 0 — the Goodhart trap: staying here has zero cost |
| 1 | VANCO_PRE (1000 mg q12h, pre-TDM) | s_vanco(tdm=0) |
| 2 | VANCO_POST (post-TDM verified) | s_vanco(tdm=3) |
| 3 | TAC_PRE (5 mg q12h, pre-TDM) | s_tac(tdm=0) |
| 4 | TAC_POST (post-TDM verified) | s_tac(tdm=3) |
| 5 | BAD_DOSE (5000 mg — contract violation) | S_MAX |
| 6 | TARGET (dual therapy verified) | 0.1 synthetic co-therapy burden |

Edges (all length 1.0): `0→0` (trap), `0→1`, `0→3`, `0→5`, `1→2`, `3→4`, `5→6`, and two **gated** transitions into the target.

### Gates

- **G_VERIFY (structural anti-Goodhart).** There is *no* edge `0→6`: the target "dual therapy verified" cannot be reached from the untreated state directly. Target-incoming edges exist only out of post-TDM states, and each is admitted only if the corresponding twin's safety gate passes:
  - `2→6` admitted iff `is_safe_dose(vanco_post_band, 10, 20, 800)`;
  - `4→6` admitted iff `is_safe_tac_dose(tac_post_band, 5, 15, 700)` **and** the AUC DDI gate (G_CYP) passes.
  A scheduler that minimizes raw suffering without the target constraint would loop at `0→0` forever; the mercyful scheduler must traverse positive-suffering pre-TDM states to reach the target.
- **G_NEPHROTOXIN (DDI).** A synthetic nephrotoxic co-medication flag removes all edges into vancomycin-active states. In a graph whose only route to the target is the vancomycin route, the scheduler must report **INFEASIBLE** rather than silently relax the gate.
- **G_CYP (DDI).** A synthetic CYP3A4-inhibitor flag scales tacrolimus CL by ×0.5 (inhibition raises exposure). The per-interval AUC enclosure `F·D/CL` doubles exactly: post-TDM AUC_hi ≈ 172 → ≈ 344 ng·h/mL, which crosses the synthetic AUC ceiling 200, so `4→6` is **not admitted** and the tacrolimus-only route becomes INFEASIBLE. The gate is evaluated on the recomputed band, not on a hardcoded verdict.

---

## 4. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **C1_PATH_FOUND** | In the healthy scenario the scheduler finds a path 0→…→6 with positive integral suffering (it must pass through pre-TDM states). | `mercyful_schedule` returns found, integral > 0. |
| **C2_GOODHART_SHORTCUT_BLOCKED** | A graph whose only target route is an unverified shortcut (G_VERIFY refuses admission) yields `found = false`. | Scheduler reports infeasible. |
| **C3_TDM_REDUCES_SUFFERING** | For both drugs, pre-TDM suffering strictly exceeds post-TDM suffering (Bayesian band narrowing lowers the field). | `s_pre > s_post` for vanco and tac. |
| **C4_CONTRACT_VIOLATION_PENALTY** | A contract-violating regimen (5000 mg) maps to `S_MAX` and is avoided by the chosen path. | `s_bad == S_MAX`; C1 path cost << S_MAX. |
| **C5_DDI_NEPHROTOXIN_INFEASIBLE** | With the nephrotoxin gate active and only the vanco route present, scheduler reports INFEASIBLE. | `found = false`. |
| **C6_DDI_CYP_GATE_BLOCKS** | With the CYP3A4-inhibitor flag, the recomputed tac AUC band crosses the ceiling and the gate blocks `4→6`; tac-only graph is INFEASIBLE. | adjusted AUC_hi > ceiling; `found = false`. |

Verdict marker: `MERCYFUL_CLINICAL_SEQ_PASS` (all six), gate marker `MERCYFUL_CLINICAL_SEQ_GATE_OK`.

---

## 5. What this is NOT

- **Not medical guidance, not a treatment recommendation, not a clinical decision-support tool.** Both patients, all doses, all DDI flags, and the AUC/Cmax windows are synthetic.
- **Not a new PK model.** All concentration bands come from the existing twins; this rung adds only the suffering-field map and the edge-admission gates.
- **Not a tight Cmax enclosure.** §2.3 is a sound loose bound by subadditivity of max; tightness is future work.
- **Not a learned scheduler.** The path search is the existing exact enumeration over simple paths.

---

## 6. Reproduce

```bash
scripts/dev/run_clinical_twin.sh tests/run-pass/mercyful_clinical_sequencing.sio
# expect: C1..C6 PASS lines, then MERCYFUL_CLINICAL_SEQ_PASS

bash scripts/ci/mercyful_clinical_sequencing_gate.sh
# expect: MERCYFUL_CLINICAL_SEQ_GATE_OK
```

Execution path: the test imports three stdlib modules plus `epistemic::knightian`, so it runs through `scripts/dev/run_clinical_twin.sh` (lean_single bootstrap engine), **not** the default Madaros `bin/souc run` — see `stdlib/clinical/README.md` and `docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`.

---

## 7. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical or patient-level claim. GAIDeT-ICMJE 2025.
