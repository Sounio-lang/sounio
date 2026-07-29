<!-- docs:meta
topic_id: repo.docs.research.routon-zd-falsifiers-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.routon-zd-falsifiers-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Routon ZD structure — falsifiers and stop rules

**Companion to:** `docs/research/routon_zd_spec_2026-07-26.md`  
**Harness:** `scripts/research/routon_zd_contract.py`

---

## Clause-level falsifiers

### C1_ZD_CENSUS

**Falsifier:** Fewer or more than 13884 canonical zero divisors (6942 index pairs × 2 signs) found at level 7.

**Stop rule:** The census is wrong; the fiber analysis cannot proceed. First check C8: if the exact criterion and the SVD scan agree with each other but not with 13884, the level-7 reality differs from the L4–L6 extrapolation — that is a *finding* (see C2), not a harness bug. If the two methods disagree, treat as a harness bug and re-audit the sign law `cds()` against the L4/L5/L6 contracts before touching anything else.

---

### C2_GROWTH_LAW

**Falsifier:** `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4` fails to reproduce any of 84 (L4), 588 (L5), 3036 (L6), 13884 (L7).

**Stop rule:** The law is inductive (derived from the fiber-count and fiber-size laws, each verified fiber-by-fiber at b = 4..7), not a theorem. A mismatch at L4/L5/L6 means the harness regressed (C_RED). A mismatch only at L7 falsifies the law as stated and demotes it to "exact at b ≤ 6" (report as a finding, not a bug). Note the asymmetric outcome already realized at L7: the law's prediction 13884 was *confirmed*, while the competing quadratic-in-`2^b` interpolant (`15·4^(b−2) − 27·2^(b−1) + 60`, prediction 13692) is falsified by the same scan.

---

### C3_FIBER_DECOMPOSITION

**Falsifier:** Fiber count at any level deviates from `F(b) = 2^b − b − 5` (7, 22, 53, 116), or the L7 label set is not exactly `{9..15} ∪ {17..31} ∪ {33..63} ∪ {65..127}`.

**Stop rule:** The "labels = {ℓ ≥ 8, not a power of two}" characterization is wrong; fiber structure claims in C4/C7/C9 are void. A new label outside the characterization (e.g. a power of two supporting ZDs at L7) would be a genuinely novel object — isolate it before proceeding.

---

### C4_FIBER_SIZE_LAW

**Falsifier:** Any fiber at any level has size ≠ `2^b − 2^(b−m+2)` for its birth level `m` (at L7: 96/112/120/124 triples for birth levels 4/5/6/7).

**Stop rule:** The Schafer-doubling size heuristic fails; report the deviating label and birth level. A single deviating fiber is itself a novel object — isolate it before proceeding.

---

### C5_TOWER_EMBEDDING

**Falsifier:** The `𝕀` census differs from the routon census restricted to indices `< 64`, or labels are not nested across `𝕊 ⊂ 𝕋 ⊂ 𝕀 ⊂` routons.

**Stop rule:** The tower structure is broken; this would contradict the L4/L5/L6 contracts too, so treat as a harness-level bug first.

---

### C6_DENSITY_GROWTH

**Falsifier:** Densities not strictly increasing across 0.4000 (L4), 0.6323 (L5), 0.7773 (L6), 0.8676 (L7).

**Stop rule:** Follows automatically from C1–C2; a failure here with C1/C2 green indicates an arithmetic bug in the density computation.

---

### C7_NATIVE_DEFECT

**Falsifier:** Any L7 fiber's missing index pairs deviate from the defect diagonal `{{a, a⊕ℓ} : a ∈ span_F2{r, 2^m, …, 2^(b−1)} ∖ {0}}`; any fiber's defect count deviates from `2^(b−m+1) − 1`; any present index pair lacks one of its two signs (sign-duality failure); or any of the 63 native fibers misses a pair other than `{ℓ−64, 64}`.

**Stop rule:** The "invertible generator pair propagates upward" mechanism is wrong or incomplete. A sign-duality failure is especially interesting: the 2-cycle criterion makes sign duality a theorem of the scan (`sgn` cancels in every determinant factor), so an observed failure would mean the harness's two methods diverge — audit C8 first, and if the divergence is real, report as a novel asymmetry, do not patch over.

---

### C8_EXACT_SVD_CROSSCHECK

**Falsifier:** The exact 2-cycle census differs from the SVD census at any level b = 4..7.

**Stop rule:** Method-level divergence. Since the criterion is derived exactly (determinant factorization over signed 2-cycles), a divergence most likely means an implementation bug in either scan or a floating-point artifact in the SVD path (e.g. the `1e-9` threshold at 128×128). Do not average the two methods; find the diverging pair(s) and decide from the exact arithmetic. If the divergence survives auditing, the signed-permutation structure of `L_a` itself is in doubt — that would invalidate the shared foundation of all four levels and must be reported as C_RED.

---

### C9_NULLITY_LAW

**Falsifier:** Any canonical ZD at any level b = 4..7 has nullity outside `{2^(b−m+2)·t : t odd, 1 ≤ t ≤ 2^(m−3)−1}` for its birth level `m`; any allowed value fails to occur in any L7 birth class; the L7 maximum differs from 60; or the SVD spot-check disagrees with an exact nullity.

**Stop rule:** The odd-part law is wrong or incomplete. A nullity with 2-adic valuation ≠ `b − m + 2` would break the "doubling exactly doubles the kernel" mechanism — isolate the pair. A missing allowed value breaks completeness but not the bound; report as a refinement of the law, not a bug. An SVD/exact disagreement on a nullity (as opposed to the census) indicates a numerical-rank threshold problem; re-check with a gap analysis of the singular values before concluding anything.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| C1 fails with C8 green | `C_AMBER` | Growth-law extrapolation failed at L7; demote laws to "exact at b ≤ 6", report as finding. |
| C1 fails with C8 red | `C_RED` | Method divergence; audit harness, do not proceed. |
| C3 or C5 fails | `C_RED` | Tower/fiber structure broken; audit harness. |
| C8 fails | `C_RED` | Method-level divergence; invalidates all other clauses until resolved. |
| C2 fails only at L8 extension | `C_AMBER` | Growth law falsified beyond L7; demote to "exact at b ≤ 7", report as finding. |
| C4, C7, or C9 fails | `C_AMBER` | Isolate the deviating fiber(s)/pair(s); a partial law plus a characterized exception is a publishable structure result, not a bug. |
| C6 fails with C1/C2 green | `C_RED` | Arithmetic bug in density clause. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-26). GAIDeT-ICMJE 2025.
