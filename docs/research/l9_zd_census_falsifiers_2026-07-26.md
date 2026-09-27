<!-- docs:meta
topic_id: repo.docs.research.l9-zd-census-falsifiers-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.l9-zd-census-falsifiers-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Level-9 ZD census — falsifiers and stop rules

**Companion to:** `docs/research/l9_zd_census_spec_2026-07-26.md`
**Harness:** `scripts/research/l9_zd_census_fast.c`
**Gate:** `scripts/ci/l9_zd_census_gate.sh`

---

## Clause-level falsifiers

### C1_CENSUS_LAW (growth law at level 9)

**Falsifier:** The exact level-9 census deviates from `Z(9) = 249084` canonical ZD triples (124542 index pairs × 2 signs).

**Stop rule:** This is the growth law's **third** out-of-sample test (`Z(7) = 13884`, `Z(8) = 59772` previously confirmed). A deviation falsifies `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4` beyond level 8; demote the law to "exact at b ≤ 8", report the deviation as a finding, and audit whether the defect is in the census (cross-check Method 1 against Method 2 — if they disagree it is a harness bug, not a law failure) or genuine.

### C2_HISTOGRAM_LAW (nullity-histogram counting law at level 9)

**Falsifier:** The multiset of multiplicities of the level-9 nullity histogram deviates from `{1344×32, 2016×16, 2736×8, 3480×4, 4236×2, 4998×1}` (distinct nullity values 63, total mass 124542) — equivalently, `μ_s(9) = 2^(9−s+1)·c⁰(s−1)` fails for some terminal level `s ∈ {4..9}`, or an unexpected multiplicity value occurs.

**Stop rule:** This closes the **live falsification target** of `docs/research/nullity_histogram_law_falsifiers_2026-07-26.md` (global stop rules, last row). A deviation falsifies the counting law beyond L8: demote it to "exact at b ≤ 8", report the deviating terminal level `(s, μ_s, want, got)` as a finding; per the parent falsifiers doc this is a `C_AMBER` event — partial laws plus characterized exceptions remain publishable structure results. Do not patch the reference to match the scan without a derivation-level explanation.

### C3_FIBER_LAWS

**Falsifier:** The fiber count deviates from `F(9) = 498 = 2^9 − 9 − 5`; the label set deviates from `{l ∈ [8, 512) : l not a power of 2}`; or an `m`-born fiber deviates from `2^9 − 2^(9−m+2)` triples.

**Stop rule:** Fiber labels and sizes are proved corollaries of the master recursion (L3/L4 of the nullity-law spec). A count deviation with C1 green indicates a label-classification bug in the harness; a size deviation with correct labels is structural — isolate the deviating label and its birth class before proceeding.

### C4_NULLITY_ODD_PART_LAW

**Falsifier:** Any `m`-born pair at level 9 whose nullity is not `2^(9−m+2)·t` with `t` odd, `1 ≤ t ≤ 2^(m−3)−1`; or an allowed odd part that never occurs in some birth class `m = 4..9` (completeness failure); or max nullity ≠ `252 = 2^8 − 4`.

**Stop rule:** The odd-part law is the proved C9 set statement of the parent contracts. A form violation (wrong valuation, even `t`, out-of-range `t`) breaks the per-pair counting law at its base — isolate the pair. A completeness failure with the form intact only weakens the counting law's surjectivity clause; report the missing `(m, t)`.

### C5_EXACT_VERIFICATION (GF(65521) rank audit)

**Falsifier:** Any of the 260610 pair-sign rank computations (130305 candidate pairs × 2 signs) whose GF(65521) nullity differs from the census nullity of Method 1.

**Stop rule:** Method 2 is generic exact linear algebra (dense Gaussian elimination with partial pivoting, no closed-form nullity formula); the block-decomposition argument proves GF(65521) rank = Q rank. A mismatch means either a GE bug (audit against a single pair by hand or against the L8 harness at level 8) or a sign-table divergence — check the FNV-1a cross-hash against the NumPy reference first. All law clauses (C1–C4) are void as *verified* statements until resolved, though they may stand as census observations.

### C6_SIGN_TABLE_CROSS_HASH

**Falsifier:** The FNV-1a-64 hash of the C sign table differs from the NumPy reference (`routon_zd_contract.py:get_sign_matrix(9)`).

**Stop rule:** The two implementations share the recursion structure but not code. A mismatch is a harness divergence, not mathematics: diff `cds()` against the Python `cds()` bit by bit before trusting any downstream number.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| Parent contract (`routon_zd_contract.py`) or L8 gate red | `C_RED` | Shared-scan regression; L9 results void until the parents are green. |
| C6 fails | `C_RED` | Sign-table divergence; no downstream number is trustworthy. |
| C5 fails | `C_RED` | Verification audit broken or census wrong; isolate a single pair-sign. |
| C1 fails with C5 green | `C_AMBER` | Growth law falsified beyond L8; demote to "exact at b ≤ 8", report as a finding. |
| C2 fails with C5 green | `C_AMBER` | Counting law falsified beyond L8 (the parent falsifiers doc's live target); demote, report deviating terminal level. |
| C3/C4 fail with C5 green | `C_AMBER` | Isolate deviating label/class; partial law plus characterized exception is a finding. |
| Level-9 scan matches all predictions | — | **Closed 2026-07-26: confirmed exactly** (all clauses green; see spec §4). Next live target: level 10 (1024-dim, `Z(10) = 4^10 − 29·2^10 + 2^9 − 4 = 1019388`; histogram law `μ_s(10)`, `s = 4..10`: 2688×64, 4032×32, 5472×16, 6960×8, 8472×4, 9996×2, 11526×1). |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-26). GAIDeT-ICMJE 2025.
