<!-- docs:meta
topic_id: repo.docs.papers.cover-letters.cp-cover-letter
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.cover-letters.cp-cover-letter
-->

# Cover Letter — *Clinical Pharmacokinetics* Submission (Skeleton)

**Date**: [submission date]
**To**: Editor-in-Chief, *Clinical Pharmacokinetics*
**Re**: Submission of *Formally Verified Vancomycin Dosing under Knightian Uncertainty*

Dear Editor,

I am pleased to submit the attached manuscript for consideration in *Clinical Pharmacokinetics*.

## Why this paper, why this journal

Therapeutic drug monitoring of vancomycin sits at the centre of an unresolved trade-off: efficacy thresholds (AUC₂₄/MIC ≥ 400) versus AKI risk (7-19% in modern ICU cohorts). Existing decision-support tools (BestDose, InsightRX, PrecisePK) handle Bayesian uncertainty in population PK parameters but do not handle **Knightian uncertainty** — the doubt about the underlying probability distribution itself, arising from assay bias and inter-individual variability beyond what the population PK can quantify.

This paper presents the first **formally verified** clinical decision support system that propagates Knightian uncertainty through a 2-compartment population PK model and reports clinical correlates from a retrospective cohort. The verification component (Lean 4 theorems mirroring the runtime safety gate) is novel for clinical pharmacology; we believe *Clinical Pharmacokinetics* is the appropriate forum because the work's primary clinical claim — Knightian-conservative refusals correlate with subsequent AKI at non-inferior rates compared to Bayesian point estimates — is squarely within the journal's TDM and CDS focus.

## Pre-registration and conflicts

The analysis plan was pre-registered at OSF [link] before primary outcome data were unblinded. The PI is the creator of the underlying programming language; per the IRB protocol the primary outcome analysis was performed by an independent biostatistician.

## Concurrent submission disclosure

A companion programming-languages paper (*Algebraic Effects for Knightian Uncertainty*) is under review at POPL 2027. The two manuscripts have **non-overlapping** primary contributions: the PL paper presents the formal substrate; this paper presents the empirical clinical validation. Section 1 of this manuscript references the PL paper for readers who wish to inspect the underlying machinery; sections 2-5 are self-contained for the clinical audience.

## Suggested reviewers

[3 international reviewers, no PI conflicts; TBD]

## Reproducibility

All Sounio code, Lean theorems, analysis pipeline, and (deidentified) MIMIC-IV-derived validation cohort will be released under permissive licences upon acceptance. Institutional cohort raw data cannot be released (IRB restriction); the supplement provides extraction queries and aggregate descriptives sufficient to reproduce the qualitative findings.

Thank you for your consideration.

Sincerely,
Demetrios Chiuratto Agourakis, MD
Principal Investigator

---

**Internal note**: clinical journal cover letters benefit from explicitly framing why the work matters to a clinician (efficacy/AKI trade-off) before any PL framing. Keep PL machinery as a "we did the verification" line, not the headline.
