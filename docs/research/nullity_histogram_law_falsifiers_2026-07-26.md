<!-- docs:meta
topic_id: repo.docs.research.nullity-histogram-law-falsifiers-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.nullity-histogram-law-falsifiers-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Nullity-histogram counting law — falsifiers and stop rules

**Companion to:** `docs/research/nullity_histogram_law_spec_2026-07-26.md`  
**Harness:** `scripts/research/nullity_histogram_law_contract.py`

---

## Clause-level falsifiers

### L1_EPS_IDENTITY

**Falsifier:** Any candidate pair `{i, j}` at any level `b = 3..7` with `S(i,ℓ)S(j,ℓ) ≠ −1`, `ℓ = i⊕j`.

**Stop rule:** The ε-identity is the load-bearing sign lemma (it kills the `2ε − 2` correction in the master recursion). A violation is either a sign-table regression — audit `cds()` against the L4/L5/L6 contracts first — or a genuinely new sign phenomenon; isolate the pair before proceeding. All downstream clauses are void until resolved.

### L2_LEFT_RIGHT_NULLITY

**Falsifier:** Any candidate pair with `nullity(L_a) ≠ nullity(R_a)` at `b = 4..7`.

**Stop rule:** The pointwise identity `pL ≡ pR` is proved from Lemma A with only the four exceptional points `k ∈ {0, i, j, ℓ}` needing direct evaluation; a mismatch indicates a harness bug in the 2-cycle formulas (compare against the parent contract's exact scan, C8 there) — do not average left and right counts.

### L3_NATIVE_RECURSION

**Falsifier:** Any native pair at `m = 4..7` with nullity `≠ 2^(m−1) − 2ν − 4`.

**Stop rule:** The boundary computation (8 exceptional `k`, correction `4ε − 4`) is wrong or incomplete. Isolate the pair, recompute its p-function by hand, and check whether the failure is at an exceptional point (bookkeeping bug) or a generic one (structural — report as a finding, the whole counting law collapses to the parts not using L3).

### L4_DOUBLING

**Falsifier:** Any embedded pair `{i, j}` or high pair `{h+i, h+j}` whose level-`b` nullity is not `2ν` (including `ν = 0`: an invertible pair whose lift becomes a ZD, or vice versa).

**Stop rule:** An embedded-pair failure contradicts the parent contract's C9 corollary — audit the harness. A high-pair failure only affects the counting recursion; report the deviating pair. An invertible-to-ZD lift would be a genuinely novel object (a zero divisor "born" above the birth level of its label) — isolate before proceeding.

### C1_INVERTIBLE_CENSUS

**Falsifier:** Invertible count at any level `b = 3..7` differs from `c⁰(b) = 3(2b−3)2^(b−2)+3` (21, 63, 171, 435, 1059).

**Stop rule:** Since `c⁰(b) = C(2^b−1,2) − Z(b)/2`, a mismatch with the parent contract's C1/C2 green means an arithmetic bug in this harness; with the parent's C2 red, the census law itself is at stake — reconcile against the parent first.

### C2_DESCENT_LAW

**Falsifier:** Any birth class `m` and odd `t` at any level `b = 4..7` with `N(m,b,t)` different from `2^(b−m+V+1)·c⁰(m_s−1)`, or a descent that fails to terminate.

**Stop rule:** The 2-adic descent is the content of the law. A single deviating class is a characterized exception, not a bug: report `(m, b, t)`, its descent trace, and the deviation; partial laws plus characterized exceptions remain publishable structure results.

### C3_TERMINAL_STRUCTURE

**Falsifier:** The multiset of multiplicities at any level `b = 4..7` differs from `{μ_s attained 2^(b−s) times : s = 4..b}`.

**Stop rule:** The terminal-level form is a corollary of C2 plus the composition bijection; a failure with C2 green is an arithmetic bug in the bijection counting, a failure with C2 red inherits the C2 stop rule.

### C4_L7_HEADLINE

**Falsifier:** The level-7 histogram differs from `{4:684, 8:504, 12:504, 16..44:336, 48:504, 52:504, 56:684, 60:870}` (total 6942).

**Stop rule:** This is the parent contract's tabulated data (its C9, SVD cross-checked). A divergence means a regression in the shared scan (`exact_nullity_index_pairs`) — audit against `scripts/research/routon_zd_contract.py` C1/C8/C9 before touching this contract.

### C5_L8_OUT_OF_SAMPLE

**Falsifier:** The law's level-8 prediction (672×16, 1008×8, 1368×4, 1740×2, 2118×1 distinct nullity values, total 29886) deviates from the tabulated L8 census; or, with `NULLITY_LAW_L8_EXACT=1`, from the raw exact L8 scan.

**Stop rule:** **Closed 2026-07-26 — confirmed, not falsified** (both modes green). A future regression here means either the tabulated reference in this harness drifted from `docs/research/l8_zd_census_benchmark_spec_2026-07-26.md` §4 (sync the reference block) or the shared scan regressed (audit the parent contracts). The live falsification target is now level 9 (see global stop rules).

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| Parent contract (`routon_zd_contract.py`) red | `C_RED` | Shared scan regression; this contract's results are void until the parent is green. |
| L1 or L2 fails | `C_RED` | Sign-lemma failure; the derivation chain is broken at its base. Audit `cds()` and the 2-cycle formulas. |
| L3 or L4 fails | `C_AMBER` | Isolate deviating pair(s); partial law plus characterized exception is a finding. |
| C1 fails with parent green | `C_RED` | Arithmetic bug in this harness. |
| C2/C3/C4 fail | `C_AMBER` | Report the deviating class/descent trace; do not patch over. |
| Level-8 scan deviates from 672×16, 1008×8, 1368×4, 1740×2, 2118×1 | — | **Closed 2026-07-26: confirmed exactly** (C5; tabulated L8 census and raw exact scan both match). |
| Level-9 scan (future) deviates from 1344×32, 2016×16, 2736×8, 3480×4, 4236×2, 4998×1 | `C_AMBER` | Law falsified beyond L8; demote to "exact at b ≤ 8", report as a finding. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-26). GAIDeT-ICMJE 2025.
