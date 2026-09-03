<!-- docs:meta
topic_id: repo.docs.papers.submission-checklist
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.submission-checklist
-->

# Submission Checklist (M6)

**Status**: M6 deliverable. Tracks the three artifacts (PL paper, clinical paper, dissertation chapter) through their respective submission flows.

## Three submission tracks

| Artifact | Venue | Deadline | Status |
|---|---|---|---|
| PL paper | POPL 2027 (or ICFP 2027) | Jul 2026 / Sep 2026 | Outline locked (M5) |
| Clinical paper | *Clinical Pharmacokinetics* (or JAMIA) | Rolling | Outline locked (M5) |
| Dissertation chapter | Internal committee review | Per programme calendar | Outline locked (M5) |

## Pre-submission gates (all artifacts)

- [ ] Cohort analysis complete (M4 milestone closed)
- [ ] Lean theorems built without `sorry` (or explicit `sorry` count documented)
- [ ] All Sounio code passes `bash scripts/run_sio_test_suite.sh`
- [ ] Pre-registration filed at OSF
- [ ] Independent biostatistician sign-off on primary outcome
- [ ] Internal LLM review (≥ 2 reviewers per artifact, per `.claude/vancomycin_track.md`)
- [ ] Co-author sign-off on each artifact

## PL paper (POPL/ICFP)

- [ ] LaTeX template: ACM acmart-sigplan
- [ ] Page count: 25 pages excluding references
- [ ] Anonymisation: double-blind (POPL); single-blind (ICFP)
- [ ] Artifact submission (typically due 1 week post-paper): Sounio docker image with reproducible build of the cohort analysis (with deidentified MIMIC-IV subset)
- [ ] Cover letter skeleton: `cover_letters/popl_cover_letter.md`
- [ ] Suggested reviewers (optional): N/A for POPL

## Clinical paper (CP / JAMIA)

- [ ] Manuscript template: per journal style
- [ ] Word count: 4500-6000
- [ ] Figures / tables: as planned in `vancomycin_clinical_paper_outline.md`
- [ ] Pre-registration link: from OSF
- [ ] CONSORT / TRIPOD checklist: TRIPOD-AI for the CDS validation
- [ ] Cover letter: `cover_letters/cp_cover_letter.md`
- [ ] Suggested reviewers: 3 (TBD; international, no PI conflicts)
- [ ] Funding statement
- [ ] CRediT contributor roles

## Dissertation chapter

- [ ] Committee review: by month 6 of programme calendar
- [ ] Cover memo: `cover_letters/dissertation_committee_memo.md`
- [ ] Defence date scheduling
- [ ] Open-access deposit pre-defense

## Reproducibility package (all artifacts)

| Item | Location |
|---|---|
| Sounio source | `/workspace/sounio` (this repo, tagged at submission) |
| Lean modules | `formal/lean4/Sounio*.lean` |
| Cohort code | `scripts/clinical/process_tdm_cohort.sh` + `data_synthetic/` |
| Real cohort data | not released (IRB restriction); MIMIC-IV instructions in supplement |
| Pre-registration | OSF [link to add at submission] |
| Docker image | `docker pull sounio/vanco-knightian:<tag>` (TBD) |

## Post-submission tracking

For each submission, log to `.claude/vancomycin_track.md`:

- Date submitted
- Submission ID / DOI / etc.
- Editor / handling-editor name
- Reviewer comments (when received)
- Major-revision turn-around plan

## Status

**Skeleton.** Cover-letter content + final pre-flight TBD when M4 cohort closes.
