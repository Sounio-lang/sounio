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
