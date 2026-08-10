<!-- docs:meta
topic_id: repo.docs.research.g2-zd-fibers-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.g2-zd-fibers-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# G₂ action on ZD fibers — falsifiers and stop rules

**Companion to:** `docs/research/g2_zd_fibers_spec_2026-07-25.md`  
**Harness:** `scripts/research/g2_zd_fibers_contract.py`

---

## Clause-level falsifiers

### G1_FIBER_DECOMPOSITION

**Falsifier:** The 84 canonical ZD pairs do not decompose into 7 fibers of size 12.

**Stop rule:** The R2 result is wrong; do not proceed.

---

### G2_G2_TRANSITIVE

**Falsifier:** The PSL(2,7) action on fibers is not transitive.

**Stop rule:** The G₂ action does not lift to fibers; the connection is broken.

---

### G3_GENERATORS

**Falsifier:** The explicit generators do not have the expected cycle structure.

**Stop rule:** The group is not PSL(2,7); the computation is wrong.

---

### G4_ORBIT_STRUCTURE

**Falsifier:** The 84 canonical ZD pairs do not form a single orbit.

**Stop rule:** The action is not transitive on ZD pairs; the geometry is richer than expected.

---

### G5_STABILIZER

**Falsifier:** The stabilizer of a fiber does not have order 24.

**Stop rule:** The action is not the natural PSL(2,7) action on the Fano plane.

---

### G6_INCIDENCE_PRESERVED

**Falsifier:** The action does not preserve the Fano incidence structure.

**Stop rule:** The fiber correspondence to Fano lines is wrong.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| G1 fails | `G_RED` | Foundation broken; do not proceed. |
| G2 or G4 fails | `G_RED` | Action not transitive; novel structure found. |
| G3 or G5 fails | `G_AMBER` | Group action misidentified. |
| G6 fails | `G_RED` | Fiber-line correspondence broken. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). GAIDeT-ICMJE 2025.
