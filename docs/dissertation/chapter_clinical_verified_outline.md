<!-- docs:meta
topic_id: repo.docs.dissertation.chapter-clinical-verified-outline
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.chapter-clinical-verified-outline
-->

# Dissertation Chapter Outline — *Verified Clinical Decision Support under Knightian Uncertainty*

**Working title (chapter)**: *Sounio for Verified Clinical Decision Support: Vancomycin under Knightian Uncertainty*

**Position in dissertation**: chapter aligned with `bbb_pbpk_dissertation_chapter.md` template (rapamycin BBB).

**Status**: M5 outline. Synthesises the PL paper (`vancomycin_pl_paper_outline.md`) and the clinical paper (`vancomycin_clinical_paper_outline.md`) into the dissertation argument.

**PBPK claim boundary**: when this chapter cross-references the rapamycin/PBPK dissertation lane, use `pbpk_claim_truth_table.md` as the claim-control artifact. Current safe wording is that Sounio demonstrates GUM-through-ODE in the PBPK14 stdlib and GPU-validated K-AXI kernels for narrower PBPK/GUM witnesses. Do not claim PBPK14 Tsit5 currently compiles to one GPU kernel.

**Estimated length**: 60-80 pages.

## Chapter abstract (1 page)

This chapter develops Sounio's substrate for **formally verified clinical decision support under Knightian uncertainty**, instantiated on vancomycin therapeutic drug monitoring (TDM). Three threads compose:

1. **Algebraic effects** (`Approx`, `Causal`, `Knowledge`, `Observe`) with Lean 4 soundness.
2. **Knightian operator** as Ferson p-boxes with parenthesization-invariance lever (linking to the sedenion-uncertainty result).
3. **Refinement-typed clinical pipeline** with manual Lean export, validated retrospectively on an ICU cohort.

The chapter argues that Sounio is **necessary** for this construction: no other PL combines monomorphic generics, refinement types, algebraic effects, and Lean export in the way the construction requires. The chapter concludes with the implications for prospective regulatory submission and the open questions (full Lean discharge, broader antibiotic panel, prospective trial).

## Chapter structure

### Section 1 — Setting

- 1.1 Why uncertainty in clinical computing: the ICU setting and TDM as a target case.
- 1.2 Knightian uncertainty: ontological status and operational consequence.
- 1.3 Why Sounio: the PL gap analysis.
- 1.4 Roadmap of the chapter.

### Section 2 — Background

- 2.1 Algebraic effects in Sounio (compressed from Sounio Effects chapter elsewhere in dissertation).
- 2.2 Imprecise probability primer.
- 2.3 Vancomycin pharmacology and TDM.
- 2.4 Existing CDS landscape (BestDose, InsightRX, PrecisePK).

### Section 3 — Composition of `Approx × Causal × Knowledge`

[Mirrors PL paper §4 + extra dissertation context.]

- 3.1 The `ComposedKnowledge` representation as a flat 64-byte struct.
- 3.2 Arithmetic preserving GUM, Approx, and Beta-edge channels.
- 3.3 Lean soundness: `composition_soundness` theorem.
- 3.4 Connection to the broader Sounio epistemic substrate (`stdlib/epistemic/`).

### Section 4 — Knightian Operator: Ferson p-boxes

[Mirrors PL paper §5 + extra rationale doc context.]

- 4.1 Decision: p-box vs Walley vs Klibanoff (cite `knightian_operator_choice.md`).
- 4.2 Operational arithmetic; corner-enumeration multiplication; vacuous division.
- 4.3 The directional-projection lever; sedenion-uncertainty parallel.
- 4.4 Lean soundness: containment preservation.

### Section 5 — Refinement Types and Lean Export

[Mirrors PL paper §6.]

- 5.1 Sounio refinement-style runtime gates.
- 5.2 The manual export pipeline.
- 5.3 Future: an automated extractor scanning `pub fn ... with ...`.

### Section 6 — Vancomycin Case Study

[Mirrors clinical paper §2-3 + PL paper §7.]

- 6.1 The Roberts 2011 popPK model.
- 6.2 The Knightian-Cmin pipeline (pre-TDM vs post-TDM band widths).
- 6.3 The contract gates and refusal as a first-class outcome.
- 6.4 The Lean dosing-safety theorem (statement and discharge plan).
- 6.5 **Tacrolimus and cross-drug synthesis** — exercising F_oral as
  the dominant GUM source.
  - 6.5.1 Closed-form 1-compartment oral steady-state trough Knightian
    gate (`stdlib/clinical/tacrolimus_oral_safety.sio`,
    `predict_c_trough_knightian`).
  - 6.5.2 14-compartment Tsit5 validation against Jusko 1995, Kershner
    1996, and Staatz 2004
    (`stdlib/darwin_pbpk/validation/tacrolimus_oral_pbpk.sio` — eight
    gates, GMFE ≤ 3.0).
  - 6.5.3 JCGM 100:2008 GUM budget showing F_oral dominance
    (`stdlib/darwin_pbpk/pd/tacrolimus_trough_gum.sio`).
  - 6.5.4 Lean obligation
    (`formal/lean4/SounioTacrolimusDosingSafety.lean`).
  - 6.5.5 Cross-drug ISO budget synthesis demonstrating that the
    dominant uncertainty source is drug-class-dependent
    (`stdlib/darwin_pbpk/validation/cross_drug_iso_budget.sio`,
    findings tabulated in
    `docs/dissertation/cross_drug_iso_budget_findings.md`).

- 6.6 **Drug-drug interactions as an irreducible GUM class** —
  tacrolimus + sirolimus (Cypher DES) co-administration.
  - 6.6.1 Mechanistic intestinal P-gp inhibition model
    (`stdlib/darwin_pbpk/ddi/tacrolimus_sirolimus_ddi.sio`).
  - 6.6.2 Clinical validation under Knightian-conservative F_combo
    propagation
    (`stdlib/darwin_pbpk/validation/tacrolimus_sirolimus_ddi_clinical.sio` —
    six gates including a "transitions from PRESCRIBE to ADJUST"
    decision-flip check on co-administration).
  - 6.6.3 Lean obligation
    (`formal/lean4/SounioTacrolimusDDI.lean`).
  - 6.6.4 Dissertation argument: DDI magnitude is bounded below by the
    Undre 1999 (n=12) single-study floor — an *irreducible* epistemic
    uncertainty that no analytical chemistry improvement can collapse.

- 6.7 Compile-time confidence gate exercised on a clinical refusal
  pathway (`tests/run-pass/tac_compile_gate_pass.sio` +
  `tests/compile-fail/tac_compile_gate_refuse.sio`). The compiler's
  `EpistemicComplete` enforcement at `lean_single.sio:20950-21020`
  rejects under-confident clinical pathways before binary emission —
  a guarantee that runtime gates cannot give.

### Section 7 — Empirical Evaluation

[Mirrors clinical paper §3-4. Filled when M4 cohort completes.]

- 7.1 Cohort and methods.
- 7.2 MAE non-inferiority.
- 7.3 Knightian coverage and refusal rate.
- 7.4 Refusal-AKI correlation.
- 7.5 Subgroup analyses.

### Section 8 — Discussion (chapter-level)

- 8.1 What the Knightian framework reveals that Bayesian point estimates miss.
- 8.2 Operational implications: the "do not prescribe yet" signal.
- 8.3 Generalisation: piperacillin-tazobactam, aminoglycosides, anticoagulants.
- 8.4 Regulatory pathway: SaMD framework, EMA AI Act, FDA pre-cert.
- 8.5 Limitations and open questions.

### Section 9 — Synthesis with Adjacent Chapters

- 9.1 Connection to the rapamycin / BBB chapter (`bbb_pbpk_dissertation_chapter.md`).
- 9.2 PBPK/GUM claim boundary: cite `pbpk_claim_truth_table.md` before importing any GPU or speedup language.
- 9.3 Connection to the surgical-interventions program.
- 9.4 The Sounio thesis: epistemic computing as a unifying substrate.

### Section 10 — Future Work

- 10.1 Full Lean discharge of the probabilistic obligation.
- 10.2 Prospective trial design (post-validation).
- 10.3 Broader antibiotic / drug class generalisation.
- 10.4 LLM-assisted refinement-type elicitation for new domains.

## Chapter cross-references

This chapter is cross-cited from:

- `bbb_pbpk_dissertation_chapter.md` (rapamycin) — shared Sounio epistemic substrate.
- `pbpk_claim_truth_table.md` — allowed/forbidden PBPK, GUM, and GPU claim language.
- `HYPER_UNCERTAINTY_PARENTHESIZATION_REPORT.md` — directional-projection lever.
- (Other dissertation chapters TBD as the dissertation table-of-contents stabilises.)

## Status checkpoints

- M5 deliverable: this outline + matched PL and clinical paper outlines.
- M6 deliverable: chapter draft submitted to dissertation committee.
- Post-defence: chapter republished as a standalone monograph or merged sections into PL/clinical journals.

## Internal review notes (parallel-LLM strategy)

Following the user's M5 protocol, this chapter is offered to multiple LLMs in parallel for blind review:

- **Reviewer A** (e.g., Composer): focus on PL framing, theorem statements, related work coverage.
- **Reviewer B** (e.g., Codex): focus on clinical methods, statistical analysis plan, reviewer-pre-empt rigor.
- **Reviewer C** (e.g., Grok / DeepSeek): focus on synthesis, dissertation-level argument, broader implications.

Reviewer disagreements are valuable — log to `.claude/vancomycin_track.md` LLM Review Notes section.
