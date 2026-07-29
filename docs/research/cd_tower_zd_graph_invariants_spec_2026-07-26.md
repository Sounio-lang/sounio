<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-graph-invariants-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-graph-invariants-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The canonical Cayley–Dickson zero-divisor graph is exactly solvable at every level

**Date:** 2026-07-26
**Status:** `EXECUTABLE` (C_GREEN reached 2026-07-26, 8/8 clauses)
**Parents:** `docs/research/routon_zd_spec_2026-07-26.md` (2-cycle criterion, fiber structure), `docs/research/nullity_histogram_law_spec_2026-07-26.md` (Lemmas L1–L4, native/embedded/high trichotomy), `docs/research/cd_tower_nullity_histogram_law_spec_2026-07-26.md` (fiber census), `docs/research/l8_zd_census_benchmark_spec_2026-07-26.md` (L8 census), `docs/research/zd_qec_prediction_spec_2026-07-26.md` (sedenion crown graph)
**Harness:** `scripts/research/cd_zd_graph_invariants_contract.py`
**Gate:** `scripts/ci/cd_zd_graph_invariants_gate.sh`

---

## 1. What this is

Define the **canonical zero-divisor graph** `G_b` of the level-`b` Cayley–Dickson algebra (dimension `2^b`): vertices are the imaginary units `e_1, …, e_{2^b−1}`, and `{i, j}` is an edge iff `a = e_i ± e_j` is a zero divisor (by sign duality, both signs always occur together — routon contract). `G_4` is the sedenion ZD graph: the crown graph `K_{7,7}` minus a perfect matching, plus the isolated vertex `e_8` (the repo's QEC crown graph).

This document proves that `G_b` is **exactly solvable at every level**: its adjacency relation, degree sequence, independence number, clique number and chromatic number are all given by closed forms, derived from the L3/L4 fiber trichotomy of the nullity-histogram law (native birth + embedded/high doubling). The invariants:

```
edges(G_b)      = Z(b)/2 = (4^b − (3b−1)·2^b + 2^(b−1) − 4)/2     (census law)
deg_b(i)        = (2^b − 2b − 2) − max(0, 2^(v₂(i)+1) − 2·v₂(i) − 4)
isolated vertex = e_{2^(b−1)}        (the last generator; unique)
α(G_b)          = b + 4              (b ≥ 3)
ω(G_b)          = 2^(b−3)            (b ≥ 4)
χ(G_b)          = 2^(b−3)            (b ≥ 4)
```

All statements are proved for all `b` in §3 (modulo the established L3/L4 fiber structure) and verified exhaustively against the audited exact 2-cycle scan at levels 4–8 and **out of sample at level 9** (512 dimensions, 124542 index pairs, `Z(9) = 249084` reproduced). Exact branch-and-bound confirms `α` and `ω` independently at `b = 4..8`.

To our knowledge the graph-theoretic invariant theory of the canonical ZD graph across the tower is new. Prior literature treats the sedenion level only (Moreno's zero-divisor classification; Cawagas–de Marrais box-kites; the "42 assessors" orthogonality graph of `𝕊`) and zero-divisor graphs of commutative rings (Anderson–Livingston); no degree law, chromatic number, independence number, or clique number for `G_b`, `b ≥ 4`, appears there, and the level-4 crown graph itself appears only in this repository's QEC note.

---

## 2. Setup

Recall (routon spec §2, histogram-law spec §2): a ZD label is `ℓ ≥ 8`, not a power of two, written `ℓ = 2^(m−1) + r`, `1 ≤ r ≤ 2^(m−1) − 1`, with **birth level** `m = ⌊log2 ℓ⌋ + 1`. The fiber of `ℓ` at level `b ≥ m` is the set of ZD pairs with `i ⊕ j = ℓ`. Lemmas L3/L4 of `docs/research/nullity_histogram_law_spec_2026-07-26.md` establish:

- (L3) the birth fiber of `ℓ` at level `m` is exactly
  `{{i0, 2^(m−1) + (i0 ⊕ r)} : i0 ∈ [1, 2^(m−1)) ∖ {r}}` — `2^(m−1) − 2` pairs, all zero divisors; the missing `i0 = r` is the fundamental defect-diagonal pair `{r, 2^(m−1)}`;
- (L4) doubling maps each level-`(t−1)` ZD pair `{i, j}` to exactly two level-`t` ZD pairs, the embedded `{i, j}` and the high `{i + 2^(t−1), j + 2^(t−1)}`, and every old-label pair at level `t` arises this way; invertible pairs stay invertible.

Consequently the fiber of `ℓ` at level `b` is exactly

```
{ {i0 + v, 2^(m−1) + (i0 ⊕ r) + v} : i0 ∈ [1, 2^(m−1)) ∖ {r},  v ∈ 2^m · [0, 2^(b−m)) },
```

`(2^(m−1) − 2)·2^(b−m) = 2^(b−1) − 2^(b−m+1)` pairs (the fiber-size law).

---

## 3. The theorems

### Theorem B (pair criterion)

Let `ℓ = x ⊕ y = 2^(m−1) + r` be a ZD label (`m ≥ 4`). Then `{x, y}` is a zero-divisor pair iff

```
x mod 2^(m−1) ∉ {0, r}   (equivalently y mod 2^(m−1) ∉ {0, r}).
```

*Proof.* `{x, y}` is in the fiber iff `x = i0 + v` or `x = j0 + v` for a birth pair and a doubling offset `v ∈ 2^m·[0, 2^(b−m))`. If bit `m−1` of `x` is 0, necessarily `i0 = x mod 2^(m−1)`, `v = x − i0`, valid iff `i0 ∉ {0, r}`; then `x ⊕ ℓ = 2^(m−1) + (i0 ⊕ r) + v = j0 + v` automatically. If bit `m−1` of `x` is 1, `x = j0 + v` forces `i0 = (x mod 2^(m−1)) ⊕ r`, valid iff `i0 ∉ {0, r}` ⟺ `x mod 2^(m−1) ∉ {r, 0}` — the same condition. ∎

*Verification:* exhaustive agreement with the audited exact 2-cycle scan over **all** pairs (ZD and non-ZD) at `b = 4..8`, and out of sample at `b = 9` (130816 pairs, 0 mismatches; contract T1).

### Theorem A (crown-join recursion)

With `h = 2^(b−1)`, `L = [1, h)`, `H = [h, 2^b)`:

1. `G_b` restricted to `L` is `G_{b−1}`; restricted to `H` it is `G_{b−1}` shifted by `h` (embedded/high doubling, L4);
2. the cross pair `{x, h + y}` (`x ∈ [1, h)`, `y ∈ [0, h)`) is a ZD pair iff `y ≠ 0` and `x ≠ y` — the cross edges are the **crown graph** `K_{h−1,h−1}` minus the perfect matching `{{x, h + x}}` (the matched pairs have label `h`, a non-ZD label, so they are invertible);
3. `e_h` is isolated (case `y = 0` of 2; also Theorem C).

*Proof.* 1 is L4. For 2, if `x = y` then `ℓ = h`, a power of two — not a ZD label. Otherwise `ℓ = h + (x ⊕ y)` is `b`-born with `r = x ⊕ y ∈ [1, h)`, and Theorem B gives ZD ⟺ `x mod 2^(b−1) = x ∉ {0, x ⊕ y}` ⟺ `y ≠ 0` (using `x ≥ 1`). ∎

At `b = 4` clauses 1–3 say `G_4` is exactly the crown graph plus isolated `e_8` — recovering the QEC observation as the base case of a tower law.

*Verification:* all three clauses at `b = 9` (contract T2, 0 mismatches).

### Theorem C (degree law)

For every `b ≥ 4` and `i ∈ [1, 2^b)`, with `s = v₂(i)`:

```
deg_b(i) = Σ_{m = max(4, s+2)}^{b} (2^(m−1) − 2)
         = (2^b − 2b − 2) − max(0, 2^(s+1) − 2s − 4).
```

*Proof.* By Theorem B, vertex `i` has exactly one fiber-`ℓ` partner (`i ⊕ ℓ`) for each `m`-born label with `i mod 2^(m−1) ∉ {0, r}`, and none otherwise. `i mod 2^(m−1) = 0` ⟺ `m ≤ s + 1`. For `m ≥ max(4, s+2)`, `i mod 2^(m−1) ≠ 0`, and the constraint excludes exactly one of the `2^(m−1) − 1` values of `r`, giving `2^(m−1) − 2` neighbors per birth level. Sum; the geometric series closes as stated. ∎

Consequences. The degree distribution is closed form: `7·2^(b−3)` vertices have maximal degree `2^b − 2b − 2` (`s ≤ 2`), and `2^(b−s−1)` vertices have degree `d0(b) − d0(s+1)` for `3 ≤ s ≤ b−2` (`d0(t) = 2^t − 2t − 2`).

**Corollary D (generator isolation).** `deg_b(2^(b−1)) = 0`: the last Cayley–Dickson generator `e_{2^(b−1)}` is the unique isolated vertex of `G_b` — it is in *no* canonical zero divisor at all. (The sum is empty for `s = b − 1`.)

**Corollary (census identity).** `Σ_i deg_b(i) = Z(b)`: grouping by `s`,
`Σ_{s=0}^{b−2} 2^(b−s−1)·(d0(b) − max(0, d0(s+1)))` reproduces the census law `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4` — checked as an exact integer identity through `b = 64` (contract T8); at `b = 9` it gives `249084`.

*Verification:* exhaustive per-vertex equality with the scan-derived adjacency at `b = 4..9`, 0 violations (contract T3, T4).

### Theorem E (independence law)

`α(G_b) = b + 4` for all `b ≥ 3`, realized by

```
A_b = {1, …, 7} ∪ {2^t : 3 ≤ t ≤ b−1}   (octonion units + tower generators).
```

*Proof.* Consider the complement `Ḡ_b` (the "invertible graph": pairs generating division-quaternionic subalgebras). Negating Theorem A(2), the invertible cross pairs `{x, h + y}` are exactly those with `y = 0` — the star `{{x, h} : x ∈ [1, h)}` — or with `x = y` — the perfect matching `{{x, h + x} : x ∈ [1, h)}`. Both families have label `h`, a non-ZD label. Hence `Ḡ_b` is two disjoint copies of `Ḡ_{b−1}` joined by that star and that matching. Note `h` is **universal** in `Ḡ_b` (it is isolated in `G_b`, so all its pairs are invertible).

Let `C` be a clique of `Ḡ_b` meeting both halves. Each `x ∈ C ∩ L` is adjacent to `h + y ∈ C ∩ H` only when `y ∈ {0, x}`. So if `|C ∩ L| ≥ 2` then `C ∩ H ⊆ {h}` and `|C| ≤ ω(Ḡ_{b−1}) + 1`; if `C ∩ L = {x}` then `C ∩ H ⊆ {h, h + x}` and `|C| ≤ 3 ≤ ω(Ḡ_{b−1}) + 1` for `b ≥ 4` (since `ω(Ḡ_{b−1}) ≥ ω(Ḡ_3) = 7`). Pure-half cliques have size `≤ ω(Ḡ_{b−1})`. Universality of `h` gives the matching upper bound's attainment: `h` plus a maximum clique of the `L`-copy is a clique. Therefore

```
ω(Ḡ_b) = ω(Ḡ_{b−1}) + 1,   ω(Ḡ_3) = 7  (octonions are a division algebra),
```

so `α(G_b) = ω(Ḡ_b) = 7 + (b − 3) = b + 4`. The set `A_b` achieves the bound: pairs within `{1,…,7}` are octonionic (invertible); pairs `{2^s, 2^t}` and `{j, 2^s}` (`j < 8 ≤ 2^s`) are exactly the defect-diagonal pairs `{r, 2^(m−1)}` of their fibers (missing from every fiber by L3), hence invertible at birth and forever after (L4). ∎

*Verification:* the recursion `ω(Ḡ_b) = ω(Ḡ_{b−1}) + 1` and the value `b + 4` confirmed by exact branch-and-bound max clique of `Ḡ_b` at `b = 4..8` (values 8, 9, 10, 11, 12); independence of `A_b` checked through `b = 12` (contract T5).

### Theorem F (clique and chromatic law)

`ω(G_b) = χ(G_b) = 2^(b−3)` for all `b ≥ 4`.

*Proof.* *Upper bound (χ):* every ZD label satisfies `ℓ ≥ 8`, so every edge crosses distinct aligned 8-blocks; coloring `i ↦ i >> 3` is a proper coloring with `2^(b−3)` colors. *Lower bound (ω):* the **Thue–Morse clique**

```
C_b = { c(k) = 8k + 6 + TM(k) : 0 ≤ k < 2^(b−3) },   TM(k) = popcount(k) mod 2,
```

is a clique. For `k ≠ k'`, `m = k ⊕ k' ≥ 1`, the pair label is
`c(k) ⊕ c(k') = 8m + TM(m)` (using `TM(k ⊕ k') = TM(k) ⊕ TM(k')` and `6 ⊕ 6 = 0`), born at level `m' = bit_length(m) + 3 ≥ 4` with `r = 8(m − 2^(m'−4)) + TM(m) ≡ TM(m) (mod 8)`. Since `c(k) mod 8 = 6 + TM(k) ∈ {6, 7}` and `m' − 1 ≥ 3`, the residue `c(k) mod 2^(m'−1)` is `≡ 6 or 7 (mod 8)` — never `0`, and never equal to `r` (which is `≡ 0 or 1 (mod 8)`). Theorem B then makes every pair a ZD pair. Then `2^(b−3) ≤ ω(G_b) ≤ χ(G_b) ≤ 2^(b−3)`. ∎

*Verification:* `C_b` pairwise-ZD through `b = 12`; exact branch-and-bound `ω(G_b) = 2^(b−3)` at `b = 4..8` (values 2, 4, 8, 16, 32); block coloring proper at `b = 4..9` (contract T6, T7).

---

## 4. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **T1_PAIR_CRITERION** | Theorem B == exact 2-cycle scan, all pairs, `b = 4..9` (b=9 out of sample). | 0 mismatches over 175216 pairs. |
| **T2_CROWN_JOIN_RECURSION** | Theorem A: within-L/within-H recursion, crown cross edges, `e_h` isolated, `b = 9`. | 0 mismatches. |
| **T3_DEGREE_LAW** | `deg_b(i) = d0(b) − max(0, d0(v₂(i)+1))`, every vertex, `b = 4..9`. | 0 violations. |
| **T4_GENERATOR_ISOLATION** | Unique isolated vertex `= 2^(b−1)`, `b = 4..9`. | Exact match. |
| **T5_INDEPENDENCE_LAW** | `α(G_b) = b + 4`: `A_b` independent (`b = 4..12`); exact B&B `ω(Ḡ_b)` (`b = 4..8`); recursion `ω(Ḡ_b) = ω(Ḡ_{b−1}) + 1` holds. | Exact match. |
| **T6_CLIQUE_LAW** | `ω(G_b) = 2^(b−3)`: Thue–Morse clique valid (`b = 4..12`); exact B&B (`b = 4..8`). | Exact match. |
| **T7_CHROMATIC_LAW** | Block coloring `i ↦ i >> 3` proper, `b = 4..9` ⟹ `χ = ω = 2^(b−3)`. | 0 monochromatic edges. |
| **T8_CENSUS_IDENTITY** | `Σ_i deg_b(i) = Z(b)` as closed forms (`b = 4..64`); criterion census `Z(9) = 249084`. | Exact equality. |

---

## 5. What this is NOT

- **Not the full zero-divisor locus.** `G_b` sees only canonical 2-unit sums `e_i ± e_j`, as in all parent contracts.
- **Not a statement about other sign conventions.** The Cayley–Dickson sign table here is the repo's standard doubling convention (`cds()` in `scripts/research/routon_zd_contract.py`); other conventions give isomorphic sign tables only in low levels.
- **The L3/L4 fiber structure is inherited**, not re-proved here; the all-`b` induction status of the parents applies (L1–L4 are proved from the block recursion and verified exhaustively at `b ≤ 8`; see the nullity-histogram spec §4).
- **Not a clinical claim.**

## 6. Reproduce

```bash
python3 scripts/research/cd_zd_graph_invariants_contract.py
# expect: T1..T8 PASS, CD_ZD_GRAPH_INVARIANTS_VERDICT C_GREEN   (~5 s)

bash scripts/ci/cd_zd_graph_invariants_gate.sh
# expect: CD_ZD_GRAPH_GATE_OK
```

Pure Python + NumPy, self-contained; reuses the audited sign table and exact 2-cycle scan of `scripts/research/routon_zd_contract.py`. Exact integer arithmetic throughout; no SVD, no floating point. The branch-and-bound certifier (`max_clique_exact`) is a generic exact coloring-bound clique algorithm, independent of the closed forms it audits.

## 7. AI disclosure

Spec and harness drafted under human direction (2026-07-26). No clinical content. GAIDeT-ICMJE 2025.
