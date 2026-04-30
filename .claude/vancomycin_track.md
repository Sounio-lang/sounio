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

## Pivot Log

| Date | From | To | Reason |
|---|---|---|---|
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
