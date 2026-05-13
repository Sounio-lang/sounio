<!-- docs:meta
topic_id: repo.docs.papers.cover-letters.popl-cover-letter
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.cover-letters.popl-cover-letter
-->

# Cover Letter — POPL 2027 Submission (Skeleton)

**Date**: [submission date]
**To**: POPL 2027 Programme Chairs
**Re**: Submission of *Algebraic Effects for Knightian Uncertainty: A Verified Clinical Decision Support Foundation*

Dear Chairs,

We are pleased to submit the attached manuscript for consideration at POPL 2027.

## Summary of contribution

The paper introduces an algebraic-effect system that composes cleanly with a Knightian-uncertainty operator (Ferson probability boxes) and is mechanised in Lean 4. We demonstrate the construction on a clinical case study — vancomycin therapeutic drug monitoring — where the framework's conservative refusals correlate with subsequent acute-kidney-injury events at non-inferior rates compared to the Bayesian state of the art.

To our knowledge no prior work combines algebraic effects, imprecise probability, and Lean-verified soundness in a single integrated PL substrate, nor validates the construction on a real clinical cohort.

## Why POPL

The contributions sit squarely within POPL's interests:

- **Core PL theory**: a new effect-row composition, mechanised in Lean.
- **Type systems**: refinement-typed runtime gates with manual proof-obligation export.
- **Empirical evaluation**: the construction does measurable clinical work on real data, going beyond the typical PL paper's microbenchmarks.

We believe POPL is the right forum because the paper's primary contribution is the formal-PL substrate; the clinical results validate but do not subsume the PL claim.

## Anonymisation

The submission is fully double-blind. The Sounio codebase is publicly available; for the review version we have anonymised the project name as `Lang` and replaced GitHub URLs with anonymous-link archive snapshots.

## Conflicts of interest

The author is the creator of the language. The clinical cohort analysis was performed under independent statistician oversight per the IRB protocol; the PI did not have access to the primary outcome data until the analysis was locked.

## Artifact track

We will submit a reproducibility artifact to the AEC track immediately after the paper deadline, including:

- Sounio toolchain (self-hosted, x86-64 Linux)
- Lean 4 formal proofs (lake-buildable)
- Cohort analysis pipeline (with deidentified MIMIC-IV subset)
- Pre-registration and analysis-plan provenance

We thank the chairs and reviewers for their consideration.

Sincerely,
[Anonymised authors]

---

**Internal note**: tighten / expand / re-tone before actual submission. POPL chairs typically prefer concise letters (≤ 1 page); this skeleton is ~1.5 pages and should be cut.
