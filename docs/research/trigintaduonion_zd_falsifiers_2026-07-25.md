<!-- docs:meta
topic_id: repo.docs.research.trigintaduonion-zd-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.trigintaduonion-zd-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Trigintaduonion ZD structure — falsifiers and stop rules

**Companion to:** `docs/research/trigintaduonion_zd_spec_2026-07-25.md`  
**Harness:** `scripts/research/trigintaduonion_zd_contract.py`

---

## Clause-level falsifiers

### T1_ZD_CENSUS

**Falsifier:** Fewer than 588 or more than 588 canonical zero divisors found.

**Stop rule:** The census is wrong; the fiber analysis cannot proceed.

---

### T2_FIBER_DECOMPOSITION

**Falsifier:** The zero divisors do not decompose into fibers by xor-label.

**Stop rule:** The fiber structure is not well-defined.

---

### T3_SEDENION_EMBEDDING

**Falsifier:** The `𝕊` zero divisors do not embed into `𝕋`.

**Stop rule:** The tower structure is broken.

---

### T4_FIBER_GROWTH

**Falsifier:** The number of `𝕋` fibers is not greater than 7.

**Stop rule:** The doubling does not produce new fiber structure.

---

### T5_G2_EXTENSION

**Falsifier:** The `G₂` action does not extend to `𝕋` fibers.

**Stop rule:** Report the obstruction; the extension is a novel finding.

---

### T6_NOVEL_STRUCTURE

**Falsifier:** The `𝕋` fiber structure has no features beyond `𝕊`.

**Stop rule:** The computation is not novel; report as such.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| T1 fails | `T_RED` | Foundation broken; do not proceed. |
| T3 or T4 fails | `T_RED` | Tower structure broken. |
| T5 fails | `T_AMBER` | Report obstruction as novel finding. |
| T6 fails | `T_AMBER` | Report as `T_CHARACTERISED` not `T_GREEN`. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). GAIDeT-ICMJE 2025.
