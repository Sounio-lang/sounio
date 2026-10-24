<!-- docs:meta
topic_id: repo.docs.audit.venlafaxine-odv-ratio-citation-forensics-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.venlafaxine-odv-ratio-citation-forensics-2026-09-26
-->

# Venlafaxine ODV/parent phenotype ratios — citation forensics

Date: 2026-09-26 · Branch: `claude/mystifying-goldberg-ccfc8a` · Status: **finding only.
No numeric constant has been changed.** Only the citations in the two source files
were corrected (comment text), because the constants feed `CL_form` and a
dissertation readout and so are an operator decision (CLAUDE.md principles 5, 6, 9).

Companion document: `docs/audit/VENLAFAXINE_PGX_STRUCTURE_AUDIT_2026-09-26.md`
(`claude/determined-kilby-3fc1ca`), which found the same broken citation from the
structural side. This document answers the question that one left open: **where do
the four numbers actually come from?**

## 1. The claim under audit

`stdlib/darwin_pbpk/pgx/cyp2d6_venlafaxine.sio` and `stdlib/darwin_pbpk/drugs/venlafaxine.sio`
state, as "reference phenotype ratios (ODV/parent Css)":

| PM | IM | EM/NM | UM |
|---:|---:|---:|---:|
| 0.25 | 1.16 | 3.45 | 10.3 |

attributed to *"Kirchheiner J et al. (2006) Ther Drug Monit 28:493-502; CPIC
venlafaxine evidence table"*. They also define the whole phenotype scale:
`vfx_cyp2d6_cl_form_scale(p) = r_p / 3.45`, hence
`vfx_cl_form_odv_adjusted(p) = 43 · r_p / 3.45`.

## 2. Verdict

**No published source reports this quadruple.** The citation is not merely
mis-attributed — the values themselves have no locatable origin, in the literature
or in this repository.

### 2.1 The cited paper does not exist

`Ther Drug Monit` 2006;28:493-502 does not resolve on PubMed (citation lookup by
journal/year/volume/page returns NOT_FOUND). Julia Kirchheiner has **no**
venlafaxine pharmacokinetics paper: `Kirchheiner J[Author] AND venlafaxine` returns
exactly two records, both on antidepressant *response* genetics with venlafaxine
only as one drug among several — Pharmacogenomics 2008;9:841-6 (PMID 18597649,
doi:10.2217/14622416.9.7.841) and Pharmacogenomics J 2006;7:48-55 (PMID 16702979,
doi:10.1038/sj.tpj.6500398). Her two real dose-adjustment papers — Acta Psychiatr
Scand 2001;104:173-92 (PMID 11531654, doi:10.1034/j.1600-0447.2001.00299.x) and
Mol Psychiatry 2004;9:442-73 (PMID 15037866, doi:10.1038/sj.mp.4001494) — report
**dose-adjustment percentages and AUC/clearance ratios, not metabolite/parent
concentration ratios**, so they are the wrong kind of source regardless. Their
venlafaxine table rows are paywalled and were **not verified**; they are not
claimed here as a source either way.

The page range 493-502 belongs to **Shams ME et al. 2006, J Clin Pharm Ther
31(5):493-502** (PMID 16958828, doi:10.1111/j.1365-2710.2006.00763.x) — different
journal, different authors. That paper's numbers (n = 100 patients) are: median
ODV/V **1.8**, 10th-90th percentile **0.3-5.2**, PM **< 0.3**, UM **> 5.2**, IM
**1.1 ± 0.8**. None of 3.45, 10.3 or 0.25 appears in it.

### 2.2 "CPIC venlafaxine evidence table" does not contain them either

The CPIC 2023 guideline (PMID 37032427, doi:10.1002/cpt.2903) main text and its
165-page supplement v2.0 were downloaded from `files.cpicpgx.org` and
text-searched: **`3.45` and `10.3` have zero hits in either document**; every
`0.25` hit is a CYP2D6 *activity score* ("2D6IM (AS 0.25-1)") and every `1.16` hit
is a confidence-interval bound in an unrelated *HTR2A*/*SLC6A4* meta-analysis.
CPIC's venlafaxine evidence rows are purely directional prose ("Positive
correlation between CYP2D6 activity score and the ratio of
O-desmethylvenlafaxine/venlafaxine"). CPIC publishes **no** numeric ODV/VEN ratio
series.

The DPWG 2024 guideline (PMID 38956296, doi:10.1038/s41431-024-01648-1) contains
exactly one ODV/VEN number, a responder *threshold* of **> 4**, and states
explicitly that the literature does not support quantitative dose advice.

Both guideline bodies therefore decline to publish what the code attributes to
them. **Any four-point PM/IM/NM/UM ratio series presented as guideline-derived is
unsupportable.**

### 2.3 Repository provenance: born with the citation already attached

`git log -S` over all refs shows the four values first appear in **`2423d5ee0`**
("wip(hyper-epistemic): snapshot Knowledge<T> runtime-variance ABI lowering + PBPK
venlafaxine", 2026-06-28, AI-authored), cherry-picked onto `main` as
**`7cb5fad91`**. The file arrives in that commit with the bogus citation already in
the header comment. There is no earlier commit, data file or table in the
repository they were transcribed from.

### 2.4 Internal evidence that the series is constructed, not measured

Normalised to NM = 1: PM 0.0725, IM 0.3362, NM 1, UM 2.9855. The upper three bins
are a near-exact geometric ladder:

    3.45 / 1.16 = 2.974        10.3 / 3.45 = 2.986
    1.16 x 2.98^n  ->  1.16, 3.457, 10.301   (= the stated 1.16 / 3.45 / 10.3 to 3 s.f.)

Therapeutic-drug-monitoring ratios are log-normal with heavy overlap between
adjacent phenotype bins; three consecutive bins do not land on a constant x2.98
step in real cohort data. A UM step of x3 is what an activity-score construction
produces. PM breaks the ladder (IM/PM = 4.64), consistent with a fourth value
chosen separately.

Two further coincidences, offered as hypotheses only, not as findings: `3.45`
equals the clearance identity `(CL/F)_V / CL_ODV,app` to three digits for
plausible inputs (96.6/28 = 3.450; 100/29 = 3.448), and `10.3` is the ODV
half-life in hours reported by Klamerus 1996 — a number already inside this
model's own reference set.

## 3. What the real literature supports

Values below were read from PubMed abstracts on 2026-09-26.

| Source | What it actually reports |
|---|---|
| **Nichols AI et al. 2011**, Clin Drug Investig 31(3):155-67, PMID 21288052, doi:10.2165/11586630-000000000-00000 | Randomised crossover, genotyped, 7 EM + 7 PM, venlafaxine ER 75 mg single dose: ODV:venlafaxine **AUC(inf) ratio 6.2 (EM) vs 0.21 (PM)**; C_max ratio 3.3 vs 0.22 (p <= 0.001). Hard point estimates — but **EM and PM only**, single dose, AUC-based. Companion: Preskorn S et al. 2009, J Clin Psychopharmacol 29(1):39-43, PMID 19142106, doi:10.1097/JCP.0b013e318192e4c1. |
| **Shams ME et al. 2006**, J Clin Pharm Ther 31(5):493-502, PMID 16958828, doi:10.1111/j.1365-2710.2006.00763.x | Steady-state patients, n = 100: median **1.8**, 10th-90th pct **0.3-5.2**, PM **< 0.3**, IM **1.1 ± 0.8**, UM **> 5.2**. Covers all four bins, but PM and UM are *cut-offs*, not group means, and the population is mixed/phenoconverted. |
| **CPIC 2023** / **DPWG 2024** | Directional only; DPWG's single number is a **> 4** responder threshold. No phenotype series. |
| **McAlpine 2011**, Ther Drug Monit 33(1):14-20, PMID 21099743, doi:10.1097/FTD.0b013e3181fcf94d | Activity-score regressions (higher AS -> lower venlafaxine, p < 0.001; only CYP2D6 associated with ODV). No binned ratios. |
| **Veefkind 2000**, Ther Drug Monit 22(2):202-8, PMID 10774634, doi:10.1097/00007691-200004000-00011 | Qualitative: the ratio distinguishes UM from PM. No binned values in the abstract. |
| **Kandasamy M et al. 2010**, Eur J Clin Pharmacol 66(9):879-87, PMID 20446083, doi:10.1007/s00228-010-0829-y | Classified 141 healthy subjects (PM 18, EM 118, UM 5) by ODV/VEN **AUC** ratio; supports ODV/VEN for identifying UM. **Per-group values are paywalled and were not verified** — this is the one place a genuine three-point series may exist, and is the recommended manual follow-up if institutional access is available. |

## 4. Consequences for the model

1. The four constants are **unsourced**. They should be labelled as such wherever
   they appear, and must not be described as literature targets, as CPIC-derived,
   or as Kirchheiner's, in the dissertation or in any external artefact.
2. The scale is **circular**: `s_p = r_p / 3.45` is *defined* from the quadruple, so
   a model that reproduces the quadruple through `s_p` is not evidence for it. The
   structural audit (`...PGX_STRUCTURE_AUDIT...`, §5) makes the same point from the
   measurement side, and `competent-mcclintock`'s portal result
   (0.2588/1.2008/3.5714/10.6625 = `s·100/28`) is circular for exactly this reason.
3. The NM value alone survives on independent grounds: `(CL/F)_V / CL_ODV,app` from
   Lessard 1999 and Klamerus 1996 gives 3.76, and the Nichols 2011 EM point
   estimate is 6.2 for a single ER dose. Neither underwrites IM, PM or UM.
4. **Replacement is an operator decision.** Moving any of the four moves `CL_form`
   and the dissertation ODV/parent readout. Two defensible options, neither adopted
   here:
   - **Shams 2006 bands** (0.3 / 1.1 / 1.8 / 5.2), honestly labelled as
     threshold-and-median rather than group means — this is what
     `bba219013` already compares against in
     `tests/run-pass/darwin_venlafaxine_xr_odv_ratio_literature.sio`;
   - **Nichols 2011 point estimates** (EM 6.2, PM 0.21) as a two-point AUC anchor,
     with IM and UM left unconstrained or derived explicitly from activity score
     and declared as such.

## 5. Method

- PubMed via MCP (`search_articles`, `get_article_metadata`,
  `lookup_article_by_citation`); CPIC PDFs from `files.cpicpgx.org`, text-extracted
  and searched; `git log -S` / `git show` over all refs for repository provenance.
- Everything marked *not verified* above was blocked by a paywall, a 403 or a
  CAPTCHA (Kirchheiner 2001/2004 venlafaxine rows; Kandasamy 2010 per-group values;
  PharmGKB's JS-rendered annotations; the NCBI Bookshelf Dean chapter). No
  remembered or inferred number was substituted for any of them.
- The arithmetic in §2.4 is reproducible from the four constants alone.
