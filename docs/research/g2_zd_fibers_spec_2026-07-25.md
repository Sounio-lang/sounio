<!-- docs:meta
topic_id: repo.docs.research.g2-zd-fibers-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.g2-zd-fibers-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# G₂ action on the seven sedenion ZD fibers

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parents:** `docs/research/rupture-r2-full-tubular_2026-07-25.md` (R2_FULL_MEASURED), `docs/research/functor_f_g2_equivariance_spec_2026-07-25.md` (H_CHARACTERISED), `docs/research/rupture-r4-fano-field_2026-07-25.md` (R4_GREEN)  
**Harness:** `scripts/research/g2_zd_fibers_contract.py`  
**Gate:** `scripts/ci/g2_zd_fibers_gate.sh`

---

## 1. What this is

The sedenion zero-divisor locus decomposes into 7 fibers (xor-labels 9–15), each of size 12, corresponding to the 7 Fano lines. The automorphism group `G₂ = Aut(𝕆)` acts on the 7 Fano lines via the projective special linear group `PSL(2,7) ≅ GL(3,2)` of order 168. This contract computes the **permutation representation** of `G₂` on the 7 ZD fibers, classifies its orbits, and verifies the connection to the Fano plane incidence structure.

This is a **novel computation**: the G₂ action on the ZD fibers has not been explicitly computed in the literature.

---

## 2. Mathematical setup

### ZD fibers

The 84 canonical 2-unit zero divisors `e_i ± e_j` decompose into 7 fibers by xor-label `i ⊕ j ∈ {9, ..., 15}`. Each fiber has 12 elements and corresponds to a Fano line.

### G₂ action on Fano lines

`G₂` acts transitively on the 7 Fano lines. The action factors through `PSL(2,7)`, the automorphism group of the Fano plane `PG(2,2)`. The homomorphism `G₂ → PSL(2,7)` has kernel the stabilizer of the Fano plane structure.

### Permutation representation

The action on the 7 Fano lines gives a permutation representation `ρ : PSL(2,7) → S₇`. We compute:
- the cycle structure of generators;
- the orbits of the 84 canonical ZD pairs under the induced action;
- the stabilizer of a single fiber.

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **G1_FIBER_DECOMPOSITION** | The 84 canonical ZD pairs decompose into 7 fibers of size 12 by xor-label. | 7 fibers, each size 12. |
| **G2_G2_TRANSITIVE** | The PSL(2,7) action on the 7 fibers is transitive. | Single orbit on fibers. |
| **G3_GENERATORS** | Explicit generators of PSL(2,7) have the expected cycle structure. | Generators match literature. |
| **G4_ORBIT_STRUCTURE** | The 84 canonical ZD pairs form a single orbit under PSL(2,7). | One orbit of size 84. |
| **G5_STABILIZER** | The stabilizer of a single fiber has order 24 (S₄). | Stabilizer order computed. |
| **G6_INCIDENCE_PRESERVED** | The action preserves the Fano incidence structure: two fibers meet in exactly one unit iff the corresponding Fano lines meet. | Incidence check passes. |

---

## 4. What this is NOT

- **Not a construction of G₂.** We use the known isomorphism `G₂ / Stab ≅ PSL(2,7)`.
- **Not a proof of new algebra.** The computation verifies the expected structure; novelty is the explicit connection to ZD fibers.
- **Not a clinical claim.**

---

## 5. Reproduce

```bash
python3 scripts/research/g2_zd_fibers_contract.py
# expect: G1..G6 PASS, G2_ZD_FIBERS_VERDICT G_GREEN

bash scripts/ci/g2_zd_fibers_gate.sh
# expect: G2_ZD_FIBERS_GATE_OK
```

Pure Python, self-contained.

---

## 6. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical content. GAIDeT-ICMJE 2025.
