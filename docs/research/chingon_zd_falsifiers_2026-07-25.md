<!-- docs:meta
topic_id: repo.docs.research.chingon-zd-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.chingon-zd-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Chingon ZD structure — falsifiers and stop rules

**Companion to:** `docs/research/chingon_zd_spec_2026-07-25.md`  
**Harness:** `scripts/research/chingon_zd_contract.py`

---

## Clause-level falsifiers

### C1_ZD_CENSUS

**Falsifier:** Fewer or more than 3036 canonical zero divisors found at level 6.

**Stop rule:** The census is wrong; the fiber analysis cannot proceed. Re-audit the sign law `cds()` against the L4/L5 contracts before touching anything else.

---

### C2_GROWTH_LAW

**Falsifier:** `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4` fails to reproduce any of 84 (L4), 588 (L5), 3036 (L6).

**Stop rule:** The law is inductive (derived from the fiber-count and fiber-size laws, each verified fiber-by-fiber at b = 4, 5, 6), not a theorem. A mismatch at L4/L5 means the harness regressed (C_RED); a mismatch appearing only when the scan is extended to L7 — where the law predicts **Z(7) = 13884** — would falsify the law as stated and demote it to "exact at b ≤ 6" (report as a finding, not a bug).

---

### C3_FIBER_DECOMPOSITION

**Falsifier:** Fiber count at any level deviates from `F(b) = 2^b − b − 5`, or the L6 label set is not exactly `{9..15} ∪ {17..31} ∪ {33..63}`.

**Stop rule:** The "labels = {ℓ ≥ 8, not a power of two}" characterization is wrong; fiber structure claims in C4/C7 are void.

---

### C4_FIBER_SIZE_LAW

**Falsifier:** Any fiber at any level has size ≠ `2^b − 2^(b−m+2)` for its birth level `m`.

**Stop rule:** The Schafer-doubling size heuristic fails; report the deviating label and birth level. A single deviating fiber is itself a novel object — isolate it before proceeding.

---

### C5_TOWER_EMBEDDING

**Falsifier:** The restricted censuses differ (`𝕊` census ≠ `𝕋` census below index 16, or `𝕋` census ≠ `𝕀` census below index 32), or labels are not nested.

**Stop rule:** The tower structure is broken; this would contradict the L4/L5 contracts too, so treat as a harness-level bug first.

---

### C6_DENSITY_GROWTH

**Falsifier:** Densities not strictly increasing across 0.4000 (L4), 0.6323 (L5), 0.7773 (L6).

**Stop rule:** Follows automatically from C1–C2; a failure here with C1/C2 green indicates an arithmetic bug in the density computation.

---

### C7_NATIVE_DEFECT

**Falsifier:** Any L6 fiber's missing index pairs deviate from the defect diagonal `{{a, a⊕ℓ} : a ∈ span_F2{r, 2^m, …, 2^(b−1)} ∖ {0}}`; any fiber's defect count deviates from `2^(b−m+1) − 1`; any present index pair lacks one of its two signs (sign-duality failure); or any of the 31 native fibers misses a pair other than `{ℓ−32, 32}`.

**Stop rule:** The "invertible generator pair propagates upward" mechanism is wrong or incomplete. A sign-duality failure would be especially interesting: it would mean `e_i + e_j` and `e_i − e_j` can differ in ZD status at level 6, contradicting the L4/L5 pattern — report as a novel asymmetry, do not patch over.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| C1 fails | `C_RED` | Foundation broken; do not proceed. |
| C3 or C5 fails | `C_RED` | Tower/fiber structure broken; audit harness. |
| C2 fails only at L7 extension | `C_AMBER` | Growth law falsified beyond L6; demote to "exact at b ≤ 6", report as finding. |
| C4 or C7 fails | `C_AMBER` | Isolate the deviating fiber(s); a partial law plus a characterized exception is a publishable structure result, not a bug. |
| C6 fails with C1/C2 green | `C_RED` | Arithmetic bug in density clause. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). GAIDeT-ICMJE 2025.
