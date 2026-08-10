<!-- docs:meta
topic_id: repo.docs.research.trigintaduonion-zd-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.trigintaduonion-zd-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Trigintaduonion zero-divisor structure — level-5 annihilation geometry

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parents:** `docs/research/catastrophe_cd.py` (CD tower scan), `docs/research/g2_zd_fibers_spec_2026-07-25.md` (G2 ZD fibers), `docs/research/rupture-r2-full-tubular_2026-07-25.md` (R2_FULL_MEASURED)  
**Harness:** `scripts/research/trigintaduonion_zd_contract.py`  
**Gate:** `scripts/ci/trigintaduonion_zd_gate.sh`

---

## 1. What this is

The Cayley–Dickson tower continues past the sedenions to the trigintaduonions `𝕋` (level 5, dimension 32). The catastrophe scan already found `588/930` singular 2-unit sums in `𝕋`. This contract analyzes the **fiber structure** of the `𝕋` zero-divisor locus: how the 588 canonical zero divisors decompose by xor-label, how the fibers relate to the sedenion `𝕊` fibers, and how the `G₂` action extends from `𝕊` to `𝕋`.

This is a **novel computation**: the trigintaduonion zero-divisor fiber structure has not been explicitly computed in the literature.

---

## 2. Mathematical setup

### The doubling tower

```
𝕆 (8) → 𝕊 (16) → 𝕋 (32)
```

Each doubling adds `N` new imaginary basis units and doubles the dimension (e.g., 8 for `𝕊`, 16 for `𝕋`). The zero-divisor locus is born at `𝕊` and grows under further doubling.

### Zero-divisor census

From `catastrophe_cd.py`:
- `𝕊`: 84/210 singular 2-unit sums
- `𝕋`: 588/930 singular 2-unit sums

The ratio grows from `84/210 ≈ 0.40` to `588/930 ≈ 0.63`. The zero-divisor set becomes dominant.

### G₂ action

For standard Cayley–Dickson algebras of dimension `≥ 16`, the automorphism group is isomorphic to `G₂` (Eakin & Sathaye, 1970). Thus `G₂` naturally acts on `𝕋` as its full automorphism group. The action from `𝕊` *extends* to `𝕋` because `𝕋` is the larger algebra containing `𝕊`; it does not "descend" from `𝕊`.

### Fiber structure

We decompose the 588 `𝕋` zero divisors by xor-label `i ⊕ j` and analyze:
- how many fibers exist;
- their sizes;
- how they embed the `𝕊` fibers;
- whether `G₂` acts on them.

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **T1_ZD_CENSUS** | The 588 canonical `𝕋` zero divisors are correctly enumerated. | 588 pairs found. |
| **T2_FIBER_DECOMPOSITION** | The zero divisors decompose into fibers by xor-label. | Fiber count and sizes reported. |
| **T3_SEDENION_EMBEDDING** | The `𝕊` zero divisors embed into `𝕋` zero divisors. | Embedding verified. |
| **T4_FIBER_GROWTH** | The number of `𝕋` fibers exceeds the number of `𝕊` fibers. | `𝕋` fibers > 7. |
| **T5_G2_EXTENSION** | The `G₂` action on `𝕊` fibers extends to `𝕋` fibers. | Extension verified or obstruction reported. |
| **T6_NOVEL_STRUCTURE** | The `𝕋` fiber structure has features not present in `𝕊`. | New structure reported. |

---

## 4. What this is NOT

- **Not a proof of new algebra.** The computation verifies the expected structure; novelty is the explicit `𝕋` fiber analysis.
- **Not a G₂ action on `𝕋`.** `G₂` is `Aut(𝕆)`, not `Aut(𝕋)`; the descent is through the embedding `𝕊 → 𝕋`.
- **Not a clinical claim.**

---

## 5. Reproduce

```bash
python3 scripts/research/trigintaduonion_zd_contract.py
# expect: T1..T6 PASS, TRIGINTADUONION_ZD_VERDICT T_GREEN

bash scripts/ci/trigintaduonion_zd_gate.sh
# expect: TRIGINTADUONION_ZD_GATE_OK
```

Pure Python, self-contained.

---

## 6. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical content. GAIDeT-ICMJE 2025.
