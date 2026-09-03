<!-- docs:meta
topic_id: repo.docs.research.faers-mercyful-analysis-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.faers-mercyful-analysis-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# FAERS × Mercyful Learning — chemotherapy toxicity feasibility analysis

**Date:** 2026-07-26
**Branch:** research/self-falsifying-compilation-line-20260726
**Verdict: NEGATIVE** — the FAERS data in this repository cannot build a chemotherapy suffering field, cannot rank chemotherapy toxicity, and cannot validate the Mercyful Learning scheduler's dose-dense vs stop-and-go predictions. Details and the path to a positive outcome below.

> **Scope.** This is a data-feasibility audit, not a clinical claim. Nothing here is medical guidance or a treatment recommendation.

---

## 1. What the in-repo FAERS data actually is

Provenance (from `experiments/faers_fano_order_asymmetry/README.md`): these files were built for the
**Fano-plane order-asymmetry DDI experiment** — a test of whether temporal start-order asymmetry of
drug triples in FAERS co-reports correlates with Fano-plane structure over CYP450 enzymes. That
experiment's own verdict is **INCONCLUSIVE** (v3, one-sided permutation p ≈ 0.37, B = 5000).

The five files:

| File | Rows | Content |
|---|---|---|
| `data/faers_168_results.csv` | 35 | Co-report case counts and start-order stats for 35 specific triples of 7 probe drugs |
| `data/faers_expanded.csv` | 35 | Same triples with openFDA totals (one row capped: warfarin+omeprazole+simvastatin 1619 total → 1000 retrieved) |
| `data/faers_concomitants.csv` | 35 | Counts of concomitant substrates per other CYP family, per triple |
| `data/faers_demographics.csv` | 35 | Aggregated age/sex/reporter-qualification/country fractions per triple |
| `data/faers_drugbank.csv` | 35 | Same statistics aggregated over all DrugBank triples per CYP triple |
| `data/cyp_drug_mapping.csv` | 39 | CYP substrate/inhibitor/inducer annotation for 39 drugs |

The 7 probe drugs in every FAERS row are: **theophylline, warfarin, repaglinide, bupropion,
omeprazole, codeine, simvastatin** — the standard CYP probe cocktail (CYP1A2, 2C9, 2C8, 2B6, 2C19,
2D6, 3A4 respectively).

---

## 2. Task 1 — chemotherapy adverse events in this data

### 2.1 There are none

The three oncology-relevant drugs in `data/cyp_drug_mapping.csv` — **cyclophosphamide** (CYP2B6
substrate), **paclitaxel** (CYP2C8 substrate), **tamoxifen** (CYP2D6 substrate) — appear in **zero**
FAERS rows. The FAERS files contain only the 7 probe drugs above. No antimetabolites, no
anthracyclines, no platinums, no alkylators.

### 2.2 No toxicity fields exist in any file

Across all six CSVs the complete field list is: CYP labels, Fano flag, drug names, case/report
counts, start-order counts (`a_first`, `b_first`, `temporal`), an asymmetry ratio, concomitant-substrate
counts, and demographic aggregates. There are **no MedDRA reaction terms, no seriousness
flags (death/hospitalization), no toxicity grades (CTCAE), no doses, no durations, no outcomes**.

Concretely, for the three Task-1 questions:

- **"Which drugs have the most severe toxicity profiles?"** — Unanswerable. Severity is not
  measured. The only available proxy is co-report volume, which measures reporting frequency, not
  severity: warfarin+omeprazole+simvastatin (1000 reports, capped retrieval), omeprazole+codeine+
  simvastatin (978), bupropion+omeprazole+simvastatin (645). These are elderly-polypharmacy
  triples (mean age 61–72), not chemotherapy.
- **"What are the common toxicity patterns?"** — Unanswerable. No reaction data was extracted.
- **"Are there dose-reduction patterns associated with toxicity?"** — Unanswerable, and
  structurally so: FAERS spontaneous reports carry no reliable dose-longitudinal data, and this
  extract kept no dose fields at all. RDI (relative dose intensity) — the quantity the Mercyful
  chemo benchmark cares about — is not representable.

### 2.3 What the data does show (honestly)

For completeness, the strongest descriptive signals in the data as it stands:

- Total probe-triple co-report cases: 1,239 across 35 triples; 6 triples have zero cases; 23/35
  have a defined asymmetry ratio.
- Per-drug case attribution (summed over triples containing the drug): omeprazole 902, simvastatin
  788, warfarin 644, bupropion 513, codeine 381, theophylline 355, repaglinide 134.
- In `faers_drugbank.csv` (all-DrugBank aggregation), the largest CYP triples are
  2C19+2D6+3A4 (9,715 cases), 2C9+2C19+2D6 (7,320), 2C9+2C19+3A4 (7,254).
- Chemo-metabolism-relevant CYP overlap (2B6 = cyclophosphamide, 2C8 = paclitaxel, 2D6 = tamoxifen):
  the triple 2B6+2C8+2D6 itself aggregates 916 cases (asymmetry 0.238); 2B6+2C19+2D6 has 3,346.
  These are **DDI co-reporting burdens among non-chemo drugs metabolized by the same enzymes** —
  a statement about shared metabolic pathways, not about chemotherapy toxicity.

---

## 3. Task 2 — mapping to Mercyful Learning

### 3.1 Can this data build a real suffering field for chemotherapy? **No.**

The Mercyful chemo benchmark (`docs/research/mercyful_chemo_sequencing_spec_2026-07-26.md`) needs a
state graph whose suffering values `s` are *treatment-attributable toxicity burdens* of chemotherapy
regimens. Building that from FAERS requires, per regimen/agent: reaction-term distributions (which
toxicities), seriousness/grade (how bad — the `s` value), and some exposure normalization (how often
per patient-course). This extract contains none of the three, for none of the chemo agents.

Additionally, two structural FAERS limitations apply even to a fresh extract: spontaneous reporting
has **no denominator** (no incidence, only disproportionality), and **no efficacy/outcome linkage**
(no remission, no RDI) — so the anti-Goodhart target constraint (`REMISSION` reached, RDI ≥ 85%)
can never be observed from FAERS alone.

### 3.2 Does the scheduler predict lower toxicity dose-dense vs stop-and-go? **Unchanged, synthetic.**

The existing benchmark's result stands exactly as before, on its synthetic graph: the Pareto frontier
is {(∫s=48, peak=8) dose-dense, (∫s=81, peak=5) stop-and-go}, continuous (84, 8) is dominated, and
the crossover is μ\* = 11. **This data adds no evidence for or against those numbers** — it neither
calibrates the suffering values nor the edge lengths. The model's prediction (stop-and-go trades +33
integral units for a peak cap of 5 vs 8) remains a synthetic-graph theorem awaiting real calibration.

### 3.3 Is there a positive outcome signal? **No.**

The only chemo-adjacent signal is indirect: the CYP families that activate/clear cyclophosphamide
(2B6) and paclitaxel (2C8) carry measurable DDI co-report burden in this data (§2.3). That could
serve as a **structural prior for an interaction-hazard layer** in a future suffering field (e.g.,
weighting edges where a chemo substrate shares its enzyme with common concomitants), but it is not a
toxicity measurement and does not move any benchmark number.

---

## 4. Task 4 — what a positive outcome would require

Since the outcome is negative, no mechanistic PK/PD model spec is warranted yet (building one on
absent data would be fabrication, not modeling). The honest gap list, in dependency order:

1. **Reaction-level FAERS extracts for chemotherapy agents.** Per-agent (cyclophosphamide,
   doxorubicin, paclitaxel, oxaliplatin, 5-FU, …): MedDRA preferred-term histograms, seriousness
   outcomes (death, life-threatening, hospitalization), and — critically — role code (primary
   suspect vs concomitant). This yields the *shape* of each agent's toxicity profile and an ordinal
   severity weighting for `s`. Source: openFDA API; the repo already has fetch scripts
   (`scripts/fetch_faers_*.py`) that show the pattern.
2. **A denominator.** FAERS alone gives disproportionality, never rates. Options: FAERS + drug
   utilization data (MEPS/IMS), or abandon FAERS for graded-toxicity sources — CTCAE tables from
   trial publications, or the regimens' label data. For `s` in "toxicity burden units," published
   grade-3/4 incidence per regimen (the same literature the synthetic benchmark cites: CALGB 9741,
   OPTIMOX1) is a more direct calibration source than FAERS.
3. **The anti-Goodhart target.** Remission/response and RDI are unobservable in FAERS. The RDI ≥ 85%
   threshold and its outcome link come from the RDI literature (Bonadonna & Valagussa; Lyman 2003)
   — i.e., the constraint parameters are literature-derived, and FAERS can at most inform the
   toxicity side.
4. **PK parameters for the mechanistic layer.** A two-compartment model per agent with CYP-mediated
   clearance (the repo's `data/cyp_drug_mapping.csv` and the existing vancomycin/tacrolimus
   PK-integration rung show the pattern), so that DDI inhibition/induction shifts exposure and
   hence shifts `s` dynamically. This is where §3.3's interaction-burden prior would plug in.

Only after (1)–(3) exist does the mechanistic PK/PD spec (Task 3) become meaningful: its structure
would be *PK layer (exposure under DDI) → PD layer (toxicity-grade hazard per cycle) → suffering
field on the regimen graph → Mercyful scheduler with anti-Goodhart remission constraint*, validated
by reproducing the CALGB 9741 / OPTIMOX1 peak–integral trade-offs from calibrated inputs rather than
declared ones.

---

## 5. CI gate decision

**No new CI gate.** A gate guards a contract; this analysis builds no contract and changes no code
or model. The existing gates are untouched: `scripts/ci/mercyful_chemo_sequencing_gate.sh` (H1–H8,
synthetic) remains the authoritative check for the chemo benchmark, and
`scripts/ci/deep_four_lane_gate.sh` covers the FAERS Fano experiment. If gap item (1) is executed
later, the natural gate is an extension of the chemo contract (H9+: "calibrated `s` values fall
inside declared Knightian bands derived from the FAERS extract"), not a new standalone gate.

---

## 6. Assumptions and limitations

- Assumed the six CSVs listed in the task (plus `faers_expanded.csv`, found adjacent) are the
  complete in-repo FAERS surface; `case_level_sample`-level JSON was not treated as primary because
  the task scoped the CSVs.
- Case counts are co-report counts from openFDA queries, with one known retrieval cap (1,000) on the
  largest triple; all volume comparisons are therefore approximate.
- Asymmetry ratios were computed for the Fano geometry experiment and are not toxicity measures;
  they are reported here only as descriptive statistics of the data as found.
- Descriptive arithmetic in §2 was recomputed from the CSVs (commands in §7); no new inferential
  statistics were run, so no new statistical claims are made.

## 7. Reproducibility

```bash
# Descriptive recomputation used for §2 (read-only, no files written):
python3 - <<'EOF'
import csv, collections
rows = list(csv.DictReader(open('data/faers_168_results.csv')))
print(sum(int(r['cases']) for r in rows))                      # 1239
all_drugs = set()
for r in rows: all_drugs.update(r['drugs'].split('+'))
print(sorted(all_drugs))                                       # 7 probe drugs, no chemo
print(sorted(all_drugs & {'cyclophosphamide','paclitaxel','tamoxifen'}))  # []
EOF
```

## 8. LLM-offload review

Per `.claude/AGENT_OFFLOAD_POLICY.md` (math claims → math-review): submitted to
`bin/llm-offload -t math-review -p xai` on 2026-07-26. Outcome logged in
`.claude/llm_offload_log.md`. The only mathematical content restated from prior work is the
synthetic benchmark's frontier and μ\* = 11; all other numbers are descriptive counts.
