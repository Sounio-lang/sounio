<!-- docs:meta
topic_id: repo.docs.research.mimic-iv-sensitivity-analysis-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mimic-iv-sensitivity-analysis-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# MIMIC-IV × Mercyful Learning — sensitivity analysis of the TDM structural validation

**Date:** 2026-07-26
**Branch:** research/self-falsifying-compilation-line-20260726
**Parent:** `docs/research/mimic_iv_mercyful_validation_2026-07-26.md` (V_GREEN 7/7, POSITIVE)
**Contract:** `scripts/research/mercyful_mimic_iv_sensitivity_contract.py` (S1..S7, **S_GREEN 7/7**)
**Gate:** `scripts/ci/mercyful_mimic_iv_sensitivity_gate.sh` (**MERCYFUL_MIMIC_IV_SENSITIVITY_GATE_OK**)
**Verdict: ROBUST POSITIVE under the model's semantics; conclusion CHANGES under the
verification-blind counterfactual.** Under G_VERIFY, the TDM-guided route is selected in
**1792/1792** varied cells — the POSITIVE verdict is invariant to every knob varied. Under the
open-gate counterfactual (the parent contract's V4 probe, swept here), the conclusion does not
hold universally: FIXED_STD wins at every tested μ in 110/256 cells, and the twin-anchored
reference point has crossover μ\* = 1.443156. Both facts are reported, not rounded off.

> **Scope.** This is a sensitivity analysis of a formal scheduling framework's synthetic
> dosing graph. Windows, doses, p-boxes, and suffering values are synthetic declarations
> (except where marked measured). This is not medical guidance, not a treatment
> recommendation, not a dosing suggestion, and not a clinical decision-support tool.

---

## 1. What is being stress-tested

The parent validation's model-side claim is: *on a fixed synthetic graph, the mercyful
scheduler's unique feasible optimum is the TDM-guided, window-verified course.* A fair
objection is that the declared knobs (therapeutic window [10, 20] mg/L, FIXED_STD band
(6, 26), the linear window-violation functional, μ = 1) were chosen to make that true.
This analysis varies all of them:

| Axis | Values | Reference point (parent contract) |
|---|---|---|
| Therapeutic window [a, b] (mg/L) | [10,15], [15,20], [10,20], [15,25] | [10,20] |
| Toxicity level τ (FIXED_STD supra-window violation) | 0.1, 0.2, 0.3, 0.6 | 0.3 → band (6,26) |
| TDM residual ρ (residual suffering of the narrowed band) | 0.0, 0.03, 0.059420, 0.1 | 0.059420 (measured, clause C3) |
| Peak weight μ | 0, 0.5, 1, 2, 5, 10, 20 | 1 |
| Field shape f (applied to violation fractions) | linear, quadratic v², softplus ln(1+v), sqrt-hinge √v | linear |

FIXED_LOW (4.0, 9.0) and VANCO_PRE (6.0, 24.0) bands are held absolute (mg/L) across
windows — monitoring and dosing hardware do not rescale when the target window moves. The
TDM band is the middle 40% of each window (dose adjusted to the measured level — that is
what TDM *is*), so the gate's admission condition holds by construction and is checked
per cell (S1). FIXED_STD(τ) = (6.0, 20·(1+τ)).

The scheduler (`mercyful_runtime_contract.py`, M_GREEN 6/6) is imported unchanged. Cost of
a path: `total = ∫s dℓ + μ·peak(s)` with source-node quadrature.

## 2. GATED landscape (the model's actual semantics) — claim invariant

G_VERIFY admits an edge to TARGET only for window-contained bands. In every varied cell,
FIXED_LOW is sub-window (no edge), FIXED_STD straddles on the low side (refused), and the
TDM band is window-contained (admitted): the TDM-guided route is the *unique* feasible path,
and the scheduler selects it:

- **S2: 1792/1792 scheduler runs select `START → VANCO_PRE → TDM_GUIDED → TARGET`**
  (4 windows × 4 τ × 4 ρ × 4 shapes × 7 μ; feasibility uniqueness asserted separately per
  cell — the selection is not a tie-breaking artifact).
- **S1:** the naive toxicity-only minimizer still picks the unreachable FIXED_LOW in all
  64 (window × τ × shape) declaration cells — the V1 under-dosing hazard is shape- and
  window-robust.
- **S3:** at the twin-anchored reference cell (measured suffering 0.675679/0.059420), the
  parent's V5 canonical numbers reproduce exactly: ∫s = 0.735099, peak 0.675679,
  total 1.410778 at μ = 1.

**The POSITIVE verdict of the parent validation does not depend on any of the varied
knobs.** This is the honest meaning of "robust" here: the conclusion is carried by the
verification constraint on the feasible set, not by the field's numeric details.

## 3. OPEN-GATE landscape (verification-blind counterfactual) — conclusion changes

Repeating the sweep with the counterfactual edge FIXED_STD → TARGET admitted (the parent
V4 probe, now across all axes), the TDM conclusion is **not** universal. Winners are
classified by closed-form route costs — both are linear in μ, so equality is exact — with
an explicit TIE class; the scheduler is cross-checked against this classification on every
strict (non-tie) point (1,648/1,648 agreements, S4):

| μ | TDM wins (strict) | FIXED_STD wins (strict) | Exact ties |
|---|---|---|---|
| 0 | 110 | 124 | 22 |
| 0.5 | 120 | 116 | 20 |
| 1 | 121 | 113 | 22 |
| 2 | 125 | 111 | 20 |
| 5 | 125 | 111 | 20 |
| 10 | 125 | 111 | 20 |
| 20 | 126 | 110 | 20 |

(of 256 cells per row). Cell classification (frozen as S4 regression expectations):

- **110/256 cells: FIXED_STD wins strictly at every tested μ.** Concentrated where
  unmonitored dosing barely violates the window: τ = 0.1 cells except the four
  [15,25]/ρ = 0 ties, all τ = 0.2 cells with ρ > 0, and two high-ρ quadratic cells at
  [15,25], τ = 0.3. When the straddle is cheap and the TDM route pays the VANCO_PRE
  transit suffering plus residual ρ, a verification-blind optimizer never profits from
  monitoring. The wide window [15,25] is the most hostile: there the FIXED_STD high edge
  20(1+τ) stays in-window for τ ≤ 0.25, so FIXED_STD's only violation is the low edge it
  shares with VANCO_PRE — and it then dominates the TDM route whenever ρ > 0.
- **110/256 cells: TDM wins strictly at every tested μ.**
- **20/256 cells: EXACT TIES at every μ.** Two structural degeneracies: τ = 0.2, ρ = 0
  makes FIXED_STD's band (6, 24) identical to VANCO_PRE's declared band (16 cells), and
  [15,25], τ = 0.1, ρ = 0 leaves both bands violating only the shared low edge (4 cells).
  There the scheduler's pick is an enumeration-order artifact and is counted for neither
  route. (An earlier draft of this contract counted tie cells by scheduler output; the
  math-review offload — Z.AI — independently derived the tie structure and caught the
  resulting 114-vs-110 prose discrepancy. Classification is now closed-form.)
- **16/256 cells: a single STD→TDM crossover as μ grows** (S4 asserts monotonicity: no
  cell flips back; 2 cells are tied exactly at μ = 0 and 2 exactly at the μ = 1 grid
  point). Both route costs are linear in μ, so at most one crossing exists; the crossover
  is μ\* = (∫_tdm − ∫_std)/(peak_std − peak_tdm) when defined.

**Twin-anchored reference point** (measured field, window [10,20], linear, τ = 0.3,
ρ = 0.059420): open-gate selection is FIXED_STD for μ ∈ {0, 0.5, 1} and TDM for
μ ∈ {2, 5, 10, 20}; analytic **μ\* = (0.735099 − 0.7)/(0.7 − 0.675679) = 1.443156** (S3).
So even at the measured reference point, the TDM superiority is visible to a
verification-blind optimizer only once peak suffering is weighted more than ≈1.44×.

**Interpretation (model-internal, not clinical).** Section 2 and 3 together sharpen the
parent's V4 into a quantitative statement: *the benefit of TDM is carried by the
verification constraint, and it is robust — 1792/1792 — precisely because it does not
depend on the suffering field's shape or magnitude. Remove verification, and whether
monitoring looks worthwhile depends on μ, the window, the toxicity level, and the field
shape (strict TDM wins range 110–126 of 256 cells across μ, with 20–22 exact ties).* This is the framework's formal shadow of the cohort's
confounding-by-indication finding (unadjusted AKI OR 2.98 flipping to 0.580 after
adjustment): metric-watching without the right constraint inverts the answer.

## 4. Frozen expectations (regression contract)

Any change to these numbers fails the gate and requires a spec amendment:

- S2: gated TDM selections = 1792/1792; feasible path unique per cell.
- S3: gated reference (∫s, peak, total) at μ=1 = (0.735099, 0.675679, 1.410778);
  open-gate sequence over μ ∈ {0, 0.5, 1, 2, 5, 10, 20} =
  [FIXED_STD ×3, TDM ×4]; μ\* = 1.443156 ± 5e-7.
- S4: strict TDM wins per μ = {0:110, 0.5:120, 1:121, 2:125, 5:125, 10:125, 20:126};
  strict FIXED_STD wins per μ = {0:124, 0.5:116, 1:113, 2:111, 5:111, 10:111, 20:110};
  ties per μ = {0:22, 0.5:20, 1:22, 2:20, 5:20, 10:20, 20:20};
  const-STD = 110, const-TDM = 110, const-TIE = 20, flips = 16 (all single STD→TDM
  crossover; 2 tied exactly at μ=0, 2 at μ=1); scheduler cross-check 1648/1648 on
  strict points.
- S5: gated TDM at μ=1 = 256/256; open strict FIXED_STD at μ=1 = 113/256.
- S6: MIMIC-IV block identical to parent V6 (28,451; 10,758 + 17,693; 37.8%;
  all three 95% CIs exclude 1.0).

## 5. Assumptions declared (all synthetic, all disclosed)

1. The varied windows are Cmin targets shaping structure only (after Rybak et al., cited
   in the paper as ref [13] — not a target claim). [10,15], [15,20], [15,25] are
   sensitivity probes, not dosing guidance.
2. FIXED_LOW / VANCO_PRE bands stay absolute across windows (§1 rationale); the TDM band
   tracks the window (middle 40%) because window-targeted adjustment is the definition of
   TDM being modeled. τ = 0.3 on [10,20] reproduces the parent's declared (6, 26).
3. The TDM residual ρ is added shape-independently to the in-window band (the measured
   0.059420 is a Knightian p-box residual, not a window-violation term). ρ = 0 isolates
   the pure functional.
4. Non-reference cells use the functional field on declared bands, not the twin's
   measured values (which exist only for [10,20], linear, 1000 mg q12h); the reference
   cell anchors the two representations (S3).
5. The open-gate counterfactual is a model-internal causality probe (edge admission),
   not a clinical counterfactual — same reading as parent V4.
6. Grid boundaries are declarations: τ > 0.6, ρ > 0.1, μ > 20, windows outside
   [10,25] mg/L, and other field shapes are out of scope for this contract.

## 6. Falsifiers

| Clause | Falsifier |
|---|---|
| S1 | Any cell where FIXED_LOW reaches the window, FIXED_STD stops straddling, the TDM band escapes the window, or the naive tox-minimizer picks a target-reaching arm |
| S2 | Any gated cell selects anything but the TDM route, or feasibility is non-unique (**RED**: the robustness claim itself breaks) |
| S3 | Reference-cell numbers deviate from the parent's canonical V5/V4 values, or μ\* moves (**RED**) |
| S4 | Frozen open-gate counts move, or any cell shows a non-monotone (re-entrant) μ sequence |
| S5 | Gate stops being causal anywhere on the grid (open = gated selections everywhere) |
| S6 | Any cited CI includes 1.0, or cohort arithmetic fails (correspondence basis gone) |
| S7 | Scope guards stripped from the contract |

Global: S2/S3 failures are RED (the sensitivity claim breaks); S4/S5 failures are RED for
this document's frozen-landscape claim but do not touch the parent verdict's gated claim;
S6 is inherited RED from the parent.

## 7. Commands run

```bash
python3 scripts/research/mercyful_mimic_iv_sensitivity_contract.py   # S_GREEN 7/7
bash scripts/ci/mercyful_mimic_iv_sensitivity_gate.sh                # MERCYFUL_MIMIC_IV_SENSITIVITY_GATE_OK
```

LLM-offload: `bin/llm-offload -t math-review -i docs/research/mimic_iv_sensitivity_analysis_2026-07-26.md`
(bundle with the contract script). Outcome: Grok [OK] on all items; Z.AI (truncated at
token cap) independently derived the exact-tie structure (τ = 0.2/ρ = 0 and
[15,25]/τ = 0.1/ρ = 0 cells) and caught the resulting 114-vs-110 prose discrepancy in the
first draft — ADDRESSED by reclassifying S4 closed-form with an explicit TIE class and
cross-checking the scheduler only on strict points. Logged in `.claude/llm_offload_log.md`.

## 8. Verdict

**ROBUST POSITIVE, with the counterfactual landscape honestly reported.**

- Under the model's semantics (G_VERIFY), the TDM-guided course is the unique feasible
  optimum in 1792/1792 varied cells across windows [10,15]/[15,20]/[10,20]/[15,25] mg/L,
  toxicity levels τ ∈ {0.1, 0.2, 0.3, 0.6}, residuals ρ ∈ {0, 0.03, 0.059420, 0.1},
  μ ∈ {0, …, 20}, and four field shapes. The parent's POSITIVE verdict is not an artifact
  of its declared knobs.
- Under the verification-blind counterfactual the conclusion changes, as the parent V4
  already showed at one point: FIXED_STD wins strictly everywhere in 110/256 cells,
  20/256 cells are exact structural ties, and the measured reference point requires
  μ > 1.443156 for TDM to win. The sensitivity analysis therefore *strengthens* the
  original interpretation: verification, not the field's numeric details, is what makes
  TDM the optimum — and that statement is now frozen against 1,792 gated scheduler runs
  plus 1,648 strict-point scheduler cross-checks in CI.
