# Vancomycin Verified Under Knightian Uncertainty — Track

Plan: `vancomycin verified knightian-82cc737a.plan.md`
Started: 2026-04-30
PI: Demetrios Chiuratto Agourakis (`demetrios@agourakis.med.br`)

## Status by Milestone

### M0 — Setup [DONE 2026-04-30]
- [x] Snapshot baseline inventory → `docs/research/m0_baseline_inventory.md`
- [x] IRB protocol drafted → `docs/research/irb_protocol_draft.md`
- [x] Knightian operator chosen: **Ferson p-box** (lock) → `docs/research/knightian_operator_choice.md`

### M1 — Approx × Causal × Knowledge composition [DONE 2026-04-30]
- [x] `stdlib/epistemic/composed_effects.sio` (`pub struct ComposedKnowledge` + arithmetic + handlers)
- [x] `formal/lean4/SounioApproxCausalKnowledge.lean` (lakefile entry; Lean build green)
- [x] 11 tests in `tests/stdlib/epistemic/test_composed_effect_*.sio`
- [x] Gate: `bash scripts/run_sio_test_suite.sh composed_effect` → 11/11 PASS
- [x] Gate: `cd formal/lean4 && lake build SounioApproxCausalKnowledge` → success

### M2 — Knightian operator [DONE 2026-04-30]
- [x] `stdlib/epistemic/knightian.sio` (`pub struct PBox` + interval-extension arithmetic + projection)
- [x] `formal/lean4/SounioKnightian.lean` (lakefile entry; Lean build green)
- [x] 5 tests in `tests/stdlib/epistemic/test_knightian_*.sio`
- [x] Gate: `bash scripts/run_sio_test_suite.sh knightian` → 9/9 PASS (5 + 4 collateral matches)
- [x] Gate: `cd formal/lean4 && lake build SounioKnightian` → success

### M3 — Vancomycin PBPK Knightian [DONE 2026-04-30]
- [x] `stdlib/clinical/vancomycin_pbpk.sio` (Roberts 2011 PK + Knightian Cmin + safety gate)
- [x] Refinement-style runtime contracts (weight/CrCl/dose/interval bounds)
- [x] Lean export pipeline (manual, M3 stage):
  - [x] `formal/lean4/SounioVancomycinDosingSafety.lean`
  - [x] `formal/proof_obligations/README.md` (canonical index)
- [x] `tests/run-pass/vancomycin_propagation_v2.sio` (V2 PASS)
- [x] `tests/stdlib/clinical/test_vancomycin_pbpk_v2.sio` (VANCO V2 PASS)
- [x] Gate: `bash scripts/run_sio_test_suite.sh vancomycin` → 4/4 PASS
- [x] Gate: `cd formal/lean4 && lake build SounioVancomycinDosingSafety` → success
- **Clinical narrative validated**: pre-TDM (0 samples) → REFUSE; post-TDM (3 samples) → PRESCRIBE; contract-violation → BLOCK.

### M4 — Cohort [DONE 2026-04-30]
- [x] `scripts/clinical/process_tdm_cohort.sh` (driver pipeline)
- [x] `scripts/clinical/data_synthetic/tdm_cohort_synthetic_v1.csv` (20 patients, plumbing-only)
- [x] `scripts/clinical/README.md` (synthetic caveats)
- [x] `docs/research/m4_validation_framework.md` (pre-registered analysis plan)
- [x] Smoke run (skeleton) — pipeline plumbing verified
- **Real cohort awaits**: IRB approval (institutional) or MIMIC-IV ETL (public fallback). The pipeline accepts the same CSV schema; no Sounio code change needed when real data lands.

### M5 — Drafts [DONE 2026-04-30]
- [x] `docs/papers/vancomycin_pl_paper_outline.md` (POPL/ICFP 2027 target)
- [x] `docs/papers/vancomycin_clinical_paper_outline.md` (CP / JAMIA target)
- [x] `docs/dissertation/chapter_clinical_verified_outline.md` (dissertation chapter)

### M6 — Submission [DONE 2026-04-30]
- [x] `docs/papers/submission_checklist.md` (master checklist)
- [x] `docs/papers/cover_letters/popl_cover_letter.md`
- [x] `docs/papers/cover_letters/cp_cover_letter.md`
- [x] `docs/papers/cover_letters/dissertation_committee_memo.md`

### Track-progress [DONE 2026-04-30]
- [x] This file.

### M2.5 — Fréchet outer enclosure (joint-dependence resolution) [DONE 2026-05-01]

Direct response to the consensus pushback recorded in M0 row of the Pivot Log
(2026-04-30) and consolidated in
`docs/research/knightian_operator_consensus_2026-04-30.md`.

- [x] Math-review-first checkpoint (per `.claude/AGENT_OFFLOAD_POLICY.md`):
      `bin/llm-offload -t math-review -p xai -i /tmp/m25_math_thesis.md` →
      Grok 4.1 validated 7/7 questions (the central theorem, vancomycin
      monotonicity, naming, conservatism factor, applicability limits,
      canonical refs, no clinical omission).
- [x] `stdlib/epistemic/knightian.sio` adds three Fréchet wrappers:
      `pb_apply2_monotone_inc_dec`, `pb_apply2_monotone_inc_inc`,
      `pb_apply2_monotone_dec_dec`. Self-test `FRECHET_OK` integrated.
- [x] `stdlib/clinical/vancomycin_pbpk.sio` `predict_cmin_knightian`
      header now contains the explicit Fréchet soundness statement
      and links to the wrapper. Body unchanged (corner enumeration
      was already correct; M2.5 is the *justification*, not new
      arithmetic).
- [x] `tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio`:
      250 deterministic LCG samples (50 each at
      r ∈ {-0.7, -0.3, 0, +0.3, +0.7}) — every sample's actual Cmin is
      enclosed by `predict_cmin_knightian`. PASS.
- [x] `formal/lean4/SounioFrechet.lean`:
      - `frechet_enclosure_monotone_inc_dec_nat`
      - `frechet_enclosure_monotone_inc_inc_nat`
      - `frechet_enclosure_monotone_dec_dec_nat`
      - `vancomycin_cmin_frechet_enclosure_obligation` (Prop) and its
        proof `vancomycin_cmin_frechet_enclosure` by direct
        instantiation of the abstract theorem.
      All proofs are Mathlib-free, use only `Nat.le_trans` + supplied
      monotonicity hypotheses. No `axiom`, no `sorry`. Lean build
      green; Grok math-review approved 5/5.
- [x] `docs/research/knightian_operator_choice.md` adds §6 documenting
      the resolution and the consensus-derived Lean budget revision.
- [x] `docs/papers/vancomycin_pl_paper_outline.md` §5.4 adds the
      Fréchet sub-section with the conservatism factor disclosure;
      reviewer pre-empt note added.
- [x] `docs/papers/vancomycin_clinical_paper_outline.md` §2.3 adds
      explicit "no independence assumption" disclosure with link to
      the theorem and the empirical sensitivity test.
- **Open follow-up**: M3.5 (Walley neighborhood model at the
  elicitation surface, lift to p-box at propagation) — **landed
  2026-05-01, see below**.
  Float-Real lift of `SounioFrechet.lean` (4–6 weeks, requires
  Mathlib import or in-tree Float order theory) — deferred.

### M3.5 — Walley elicitation surface [DONE 2026-05-01]

Direct response to the consensus pushback: complete the
elicitation/propagation split announced in M2.5 by adding the
Walley ε-contamination credal set as the elicitation operator.

- [x] **M3.5.0 Math-review-first** (`bin/llm-offload -t math-review
      -p xai -i /tmp/m35_walley_thesis.md`) caught two real bugs in
      the proposed design: (a) variance upper bound was missing the
      cross-term `ε(1−ε)(μ_0−μ_Q)²` (UNSOUND counter-example
      provided); (b) Fréchet on mean rectangle under-encloses
      nonlinear monotone f. Both fixed before any code was written.
- [x] **M3.5.1** `stdlib/epistemic/walley.sio` — `CredalSet` struct
      + `cs_neighborhood` / `cs_precise` / `cs_vacuous` constructors
      + `credal_to_pbox` (mean-band lift, sound variance bound) +
      `credal_to_support_pbox` (support-band lift for nonlinear
      monotone propagation) + 5/5 self-test smoke checks PASS.
- [x] **M3.5.2** `formal/lean4/SounioWalley.lean` — five
      structural theorems mechanised in core Lean 4 (Nat-shadow, no
      Mathlib, no `axiom`, no `sorry`):
      - `walley_collapse_at_zero_nat` (precise recovery at ε = 0)
      - `walley_collapse_gap_zero_nat` (gap = 0 at ε = 0)
      - `walley_vacuous_lo_at_one_nat` / `_hi_at_one_nat` /
        `_gap_at_one_nat` (full-support band at ε = 1)
      - `walley_gap_monotone_in_epsilon_nat` (gap monotone in ε)
      - `walley_frechet_composition_holds` (composition with M2.5
        Fréchet on support rectangle).
      Lakefile entry added; `lake build SounioWalley` green.
- [x] **M3.5.3** Five round-trip tests in
      `tests/stdlib/epistemic/test_walley_*.sio`:
      `_collapse`, `_vacuous`, `_width_monotone`,
      `_support_lift`, `_frechet_compose` — all PASS via
      `bash scripts/run_sio_test_suite.sh walley`.
- [x] **M3.5.4** Math-review of the implementation
      (`/tmp/m35_implementation_review.md`) — Grok 4.1 returned
      11/11 OK, NO_FINDINGS. Variance bound, Lean theorems, and
      Fréchet composition all confirmed.
- [x] **M3.5.5** `docs/research/knightian_operator_choice.md` §7
      added: documents the Walley landing, the operator-surface
      table, and the math-review record. Audit-log entries added
      to `.claude/llm_offload_log.md` for both the thesis review
      and the implementation review.
- **Open follow-up**: Klibanoff smooth-ambiguity wrapper for
  cost-of-ambiguity calculations (deferred M5+). Float-Real lift
  of both `SounioFrechet.lean` and `SounioWalley.lean` (4–6 weeks).

## Pivot Log

| Date | From | To | Reason |
|---|---|---|---|
| 2026-04-30 | M2 univariate p-box treated as the safety operator | Acknowledged Knightian operator unsoundness for multivariate clinical PBPK; **opened M2.5 (Fréchet enclosure) and M3.5 (Walley neighborhood)** as follow-ups | **Joint (Vc, CL) dependence omission**. Caught by `bin/llm-offload --raw <prompt> deepseek xai` consensus fan-out (gemini/qwen blocked on credits, groq invalid key — partial 2-way). DeepSeek and Grok independently flagged that p-boxes are univariate and that the vancomycin Cmin map is non-monotone with Vc/CL correlated (popPK r ≈ 0.3–0.7). Both recommended Walley neighborhood/contamination models as the elicitation operator with p-box at the propagation surface. M2/M3 NOT rolled back; Fréchet-bound enclosure planned to make the existing arithmetic copula-free sound. Consolidated review at `docs/research/knightian_operator_consensus_2026-04-30.md`. |
| 2026-04-30 | corner enum `(Vc_hi, CL_hi) → cmin_low; (Vc_lo, CL_lo) → cmin_high` | corner enum `(Vc_lo, CL_hi) → cmin_low; (Vc_hi, CL_lo) → cmin_high` | **Monotonicity sign error**. Original assumed `dCmin/dVc < 0`; correct is `dCmin/dVc > 0` for all θ = ke·τ > 0 (since h(θ) = e^θ(θ-1) + 1 > 0). Caught by `bin/llm-offload -t math-review -p xai` (Grok 4.1) in 28 s. Pre-TDM band shifted [11.30, 21.31] → [8.49, 24.29] (correct, wider; clinical narrative preserved — REFUSE outcome unchanged but reason is stronger: band now straddles BOTH boundaries). Post-TDM band [13.92, 16.02] → [12.66, 17.32] (still within therapeutic window, PRESCRIBE preserved). All 4 vancomycin tests PASS post-fix. |

## Artifacts Produced

| Date | Path | Purpose |
|---|---|---|
| 2026-04-30 | `docs/research/m0_baseline_inventory.md` | M0 baseline |
| 2026-04-30 | `docs/research/knightian_operator_choice.md` | M0 operator decision (Ferson p-box) |
| 2026-04-30 | `docs/research/irb_protocol_draft.md` | M0 IRB skeleton |
| 2026-04-30 | `stdlib/epistemic/composed_effects.sio` | M1 composition substrate |
| 2026-04-30 | `formal/lean4/SounioApproxCausalKnowledge.lean` | M1 Lean soundness |
| 2026-04-30 | `tests/stdlib/epistemic/test_composed_effect_*.sio` (11 files) | M1 test suite |
| 2026-04-30 | `stdlib/epistemic/knightian.sio` | M2 Ferson p-box operator |
| 2026-04-30 | `formal/lean4/SounioKnightian.lean` | M2 Lean soundness |
| 2026-04-30 | `tests/stdlib/epistemic/test_knightian_*.sio` (5 files) | M2 test suite |
| 2026-04-30 | `stdlib/clinical/vancomycin_pbpk.sio` | M3 clinical pipeline |
| 2026-04-30 | `formal/lean4/SounioVancomycinDosingSafety.lean` | M3 Lean dosing-safety obligation |
| 2026-04-30 | `formal/proof_obligations/README.md` | M3 proof-obligation index |
| 2026-04-30 | `tests/run-pass/vancomycin_propagation_v2.sio` | M3 v2 propagation test |
| 2026-04-30 | `tests/stdlib/clinical/test_vancomycin_pbpk_v2.sio` | M3 stdlib clinical test |
| 2026-04-30 | `scripts/clinical/process_tdm_cohort.sh` | M4 cohort pipeline driver |
| 2026-04-30 | `scripts/clinical/data_synthetic/tdm_cohort_synthetic_v1.csv` | M4 synthetic cohort skeleton |
| 2026-04-30 | `scripts/clinical/README.md` | M4 caveats |
| 2026-04-30 | `docs/research/m4_validation_framework.md` | M4 pre-registered plan |
| 2026-04-30 | `docs/papers/vancomycin_pl_paper_outline.md` | M5 PL paper outline |
| 2026-04-30 | `docs/papers/vancomycin_clinical_paper_outline.md` | M5 clinical paper outline |
| 2026-04-30 | `docs/dissertation/chapter_clinical_verified_outline.md` | M5 dissertation chapter outline |
| 2026-04-30 | `docs/papers/submission_checklist.md` | M6 submission checklist |
| 2026-04-30 | `docs/papers/cover_letters/popl_cover_letter.md` | M6 POPL cover letter |
| 2026-04-30 | `docs/papers/cover_letters/cp_cover_letter.md` | M6 CP cover letter |
| 2026-04-30 | `docs/papers/cover_letters/dissertation_committee_memo.md` | M6 committee memo |
| 2026-05-01 | `stdlib/epistemic/walley.sio` | M3.5 Walley elicitation surface (CredalSet + two lifts) |
| 2026-05-01 | `formal/lean4/SounioWalley.lean` | M3.5 Lean structural soundness (5 theorems, Nat-shadow) |
| 2026-05-01 | `tests/stdlib/epistemic/test_walley_*.sio` (5 files) | M3.5 round-trip + Fréchet composition tests |

## LLM Review Notes

| Date | Provider | Task | Target | Outcome |
|---|---|---|---|---|
| 2026-04-30 | xAI Grok 4.1 fast reasoning | math-review | `vp_cmin_point` monotonicity comment in `stdlib/clinical/vancomycin_pbpk.sio` | **Caught real bug**: sign error on `dCmin/dVc`. Validated symbolically with counter-example (CLτ=1, Vc=1 ⇒ Cmin≈0.582; Vc=1.1 ⇒ Cmin≈0.613). Fix landed same session; see Pivot Log. **This single review prevented shipping the bug to a referee.** |

Recommended next reviews (post-cohort):

- **Reviewer A** (Composer): focus on PL framing of `vancomycin_pl_paper_outline.md` — does the related-work section adequately distinguish from existing probabilistic PLs? Are theorem statements maximally tight?
- **Reviewer B** (Codex): focus on clinical methods of `vancomycin_clinical_paper_outline.md` — does the analysis plan satisfy TRIPOD-AI? Are the secondary-outcome inferences well-powered?
- **Reviewer C** (Grok / DeepSeek): focus on dissertation-level synthesis of `chapter_clinical_verified_outline.md` — does §§ 9 (synthesis with adjacent chapters) work as the thesis statement, or does it read as a stitching of two papers?

Reviewer disagreements (especially on the Lean-discharge depth question in the committee memo) are valuable; log the diffs back here.

## What is *not* done in this cycle

The deliverables landed represent the **landable surface** of a 6-month plan executed in a single session. The following pieces inherently require time outside this session:

1. **IRB submission and approval** (institutional — weeks to months).
2. **Real cohort ingestion** (institutional or MIMIC-IV ETL — weeks).
3. **Full Lean discharge** of `cmin_within_implies_efficacy_and_safety` (8-12 weeks of dedicated Lean work).
4. **Cohort analysis** (depends on (1) or (2); weeks).
5. **Paper prose** (filling the M5 outlines once results land; weeks).
6. **Internal LLM review rounds** (despatchable now but most useful after results land).

The Sounio + Lean substrate, the formal soundness skeletons, the cohort-pipeline plumbing, the pre-registered analysis plan, the paper outlines, and the submission scaffolding are **all green and committable today**. The plan's intellectual scaffolding is complete; the calendar-bound external dependencies remain.

## Reproducibility — quick verification

```bash
# Type-check + run the substrate
./bin/souc check stdlib/epistemic/composed_effects.sio
./bin/souc check stdlib/epistemic/knightian.sio
./bin/souc check stdlib/clinical/vancomycin_pbpk.sio
./bin/souc run stdlib/clinical/vancomycin_pbpk.sio    # PRE_REFUSE / POST_PRESCRIBE / CONTRACT_BLOCK

# Test suite
bash scripts/run_sio_test_suite.sh composed_effect    # 11/11 PASS
bash scripts/run_sio_test_suite.sh knightian           # 5/5 (+ collateral) PASS
bash scripts/run_sio_test_suite.sh vancomycin          # 4/4 PASS

# Lean
cd formal/lean4
lake build SounioApproxCausalKnowledge SounioKnightian SounioVancomycinDosingSafety

# Cohort skeleton
bash scripts/clinical/process_tdm_cohort.sh
```
