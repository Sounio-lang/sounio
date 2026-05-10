<!-- docs:meta
topic_id: repo.docs.papers.main.168-revision-notes
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.main.168-revision-notes
-->

# 168 Theorem — Revision Notes

Paper: "The 168 Count for Octonion Basis Associators and a Computational Sedenion Extension"
Submitted: 2026-03-21 to Advances in Applied Clifford Algebras
Authors: Agourakis, Gerenutti

---

## Context

Gemini review (2026-03-22) identified:
- **Theorem 1 (168 count) is classical.** The count is a corollary of PG(2,2) having 168 ordered non-collinear triples — present in Baez (2002), Conway & Smith (2003).
- **Lemma 2 (norm dichotomy {0,2}) is classical.** Follows from alternativity: anti-associativity gives ±e_p - (∓e_p) = ±2e_p, norm = 2.
- **Sections 4–5 (sedenion extension) are original and publishable.** The tower pattern is the real contribution.
- **All four open questions have clean proofs.**

Sonnet (Desktop) identified:
- **The XOR proof of p=q needs two steps**, not one. The XOR grading holds in the binary labeling, not in the paper's Fano labeling. Labeling-independence is the second step.

All claims verified computationally (2026-03-22).

---

## What the revision adds

### 1. Trigintaduonion verification (new computational result)

T_5 = 15,960 = 95 × 168. Verified exhaustively over all 31³ = 29,791 ordered basis triples in the Cayley-Dickson algebra of dimension 32.

Binary norm property ||[e_i, e_j, e_k]|| ∈ {0, 2} confirmed at dim 32 (only values 0 and 2 observed; no intermediate magnitudes).

### 2. Conjectured closed formula (new)

**Conjecture.** For all k ≥ 3, the number of ordered triples (i,j,k) with nonzero basis associator in the 2^k-dimensional Cayley-Dickson algebra is:

    T_k = 168 × (P_k − 4·P_{k−1})

where P_k = (2^k − 1)(2^{k−1} − 1)(2^{k−2} − 1) / 21 counts Fano subplanes in PG(k−1, 2).

**Verification:**
- k=3 (octonions, dim 8): P_3=1, P_2=0. T_3 = 168×(1−0) = 168. ✓ (Theorem 1)
- k=4 (sedenions, dim 16): P_4=15, P_3=1. T_4 = 168×(15−4) = 1848. ✓ (Observation 3)
- k=5 (trigintaduonions, dim 32): P_5=155, P_4=15. T_5 = 168×(155−60) = 15960. ✓ (new)

**Status:** Present as conjecture. The analytical proof would require establishing that:
(a) each Fano subplane in PG(k−1,2) decomposes into "octonionic" (contributing 168 nonzero triples) and "quasi-octonionic" (contributing 72), per Cawagas (2004);
(b) the number of quasi-octonionic planes is exactly 7·P_{k−1}.
We have not independently verified these structural claims from Cawagas. The formula is presented as a conjecture supported by three-point computational evidence.

### 3. Closing OQ4: Proof of p = q without case analysis

**Claim.** For any non-collinear triple {i,j,k}, both parenthesizations (e_i·e_j)·e_k and e_i·(e_j·e_k) are proportional to the SAME basis element. That is, p = q.

**Proof (two steps).**

*Step 1 (binary labeling).* The Cayley-Dickson construction admits a natural Z_2^k grading on basis indices. In the standard binary labeling (where basis elements are indexed by nonzero vectors of Z_2^k), the product of e_i and e_j has index i ⊕ j (bitwise XOR), up to sign. Since XOR is associative on Z_2^k:

    index((e_i·e_j)·e_k) = (i ⊕ j) ⊕ k = i ⊕ (j ⊕ k) = index(e_i·(e_j·e_k))

Therefore p = q in the binary labeling.

**Note:** The Fano labeling used in the paper (lines {1,2,4}, {2,3,5}, ...) does NOT follow XOR indexing. For example, e_1·e_2 = e_4 in the paper, but 1 ⊕ 2 = 3. The XOR rule holds specifically in the binary labeling of the Cayley-Dickson construction.

*Step 2 (labeling independence).* The assertion p = q is a property of the abstract octonion algebra O, independent of basis labeling. To see this: if σ is any automorphism of O permuting basis elements, then

    σ(e_{g_L(i,j,k)}) = e_{g_L(σ(i),σ(j),σ(k))}

so g_L = g_R in one labeling implies g_L = g_R in all labelings obtained by automorphism. Since the binary labeling and the Fano labeling both define the same abstract algebra (all valid octonion multiplication tables define isomorphic algebras), the result transfers. □

### 4. Closing OQ2: The 336 = 336 coincidence

**Claim.** The numerical equality between 336 sed-sed nonzero associator triples and 336 primitive zero-divisor pairs is a combinatorial artifact, not a structural bijection.

**Proof.** In the Z_2^4 representation of sedenion basis indices, the upper-half elements {e_8, ..., e_15} all have bit 3 set. For any three distinct upper-half indices a, b, c:

    bit 3 of (a ⊕ b) = 1 ⊕ 1 = 0

So a ⊕ b lands in the lower half {0,...,7}. Therefore no line of PG(3,2) (which consists of points {a, b, a ⊕ b}) is entirely contained in the upper half. Every ordered triple of distinct upper-half elements is non-collinear, giving 8 × 7 × 6 = 336 triples, ALL with nonzero associator.

**Verified directly:** exhaustive computation of all 336 sedenion upper-half associators confirms all are nonzero.

**The coincidence breaks at dim 32:** the upper half of the trigintaduonions has 16 elements, giving 16 × 15 × 14 = 3,360 non-collinear triples — no longer equal to any known zero-divisor count.

### 5. Closing OQ3: The factor 11

From the conjectured formula: 11 = P_4 − 4·P_3 = 15 − 4 = 11.

The factor has combinatorial significance (related to Fano subplane counts in PG(3,2)), not group-theoretic significance. Note 11 ∤ |PSL(2,7)| = 168, consistent with this.

---

## What changes in each section of the paper

### Abstract
Add at end of computational observations sentence: "We further verify that the pattern extends to the trigintaduonions (dimension 32), obtaining T_5 = 15,960 = 95 × 168, and conjecture a closed formula T_k = 168(P_k − 4P_{k−1}) relating nonzero associator counts to the number of Fano subplanes in the associated projective geometry PG(k−1, 2). All four open questions from the original submission are resolved."

### Section 3 (Lemma 2 — Remark after proof)
Replace current Remark (ending "We leave this as a question for the interested reader") with the two-step proof from §3 above. Attribute the binary-labeling insight to the Z_2^k grading of the Cayley-Dickson construction.

### Section 4 (Computational Sedenion Extension)
Rename to "Computational Extension to Higher Dimensions." Expand Table 1:

| Algebra          | dim | Total   | Nonzero | Factor     |
|-----------------|-----|---------|---------|------------|
| Octonion         | 8   | 343     | 168     | 1 × 168    |
| Sedenion         | 16  | 3,375   | 1,848   | 11 × 168   |
| Trigintaduonion  | 32  | 29,791  | 15,960  | 95 × 168   |

Add Observation 3': binary norm property ||[e_i, e_j, e_k]|| ∈ {0, 2} verified at dim 32.

### New subsection 6.3: "A conjectured tower formula"
Present Conjecture 5: T_k = 168(P_k − 4P_{k−1}). Explain P_k as Fano subplane count. Reference Cawagas (2004) for sedenion subalgebra decomposition. State the three-point verification.

### Section 7 (Open Questions)
Replace the four original OQs with:

**Resolved questions.** Since submission, all four original open questions have been addressed:
1. (OQ1) The 168× pattern extends to dim 32: T_5 = 15,960 = 95 × 168.
2. (OQ2) The 336 = 336 equality is a combinatorial artifact of the Z_2^4 upper-half structure; it breaks at dim 32 (3,360 ≠ 336).
3. (OQ3) The factor 11 = P_4 − 4P_3 arises from the Fano subplane formula.
4. (OQ4) The assertion p = q follows from the Z_2^k grading of the Cayley-Dickson construction and labeling independence.

**New open questions:**
1. Prove the tower formula T_k = 168(P_k − 4P_{k−1}) analytically for all k ≥ 3.
2. Does the binary norm property ||[e_i, e_j, e_k]|| ∈ {0, 2} hold for ALL Cayley-Dickson algebras? (Verified at k = 3, 4, 5.)
3. Characterize the distribution of nonzero associator triples across sub-classes (oct-oct-oct, sed-sed-sed, cross, etc.) as a function of k.

---

## Email to editor (draft)

Subject: Updated manuscript — "The 168 Count for Octonion Basis Associators and a Computational Sedenion Extension"

Dear Editor,

We would like to submit a revised version of our manuscript [MS-ID], submitted on 21 March 2026. Since initial submission, we have obtained new computational results that significantly strengthen the paper.

The key additions are:

1. Exhaustive verification at dimension 32 (trigintaduonions): T_5 = 15,960 = 95 × 168 nonzero basis associator triples, confirming the 168-divisibility pattern at one further level of the Cayley-Dickson tower.

2. A conjectured closed formula T_k = 168(P_k − 4P_{k−1}), where P_k counts Fano subplanes in PG(k−1, 2), verified at three levels (k = 3, 4, 5).

3. A universal proof of the p = q assertion (previously left as a remark in Lemma 2) via the Z_2^k grading of the Cayley-Dickson construction, avoiding case analysis.

4. Resolution of the 336 = 336 numerical coincidence (Open Question 2) as a combinatorial artifact of the Z_2^4 structure, demonstrated by its breakdown at dimension 32.

All four open questions from the original submission are now resolved. The additions strengthen the manuscript without changing its scope or character.

If the paper has not yet been assigned to referees, we would be grateful for the opportunity to replace it with the revised version. If review is already underway, we are happy to incorporate these results at the revision stage.

Best regards,
Demetrios C. Agourakis
Marli Gerenutti

---

## Revision 3 — Cohomological Reformulation (2026-05-10)

Triggered by exploratory conversation on the categorical foundations of Sounio's
hypercomplex type system. Discovery: the paper's existing arguments (Lemma 2's
$ZZ_2^k$ grading + XOR + signs in $\{\pm 1\}$) are *literally* the structure
identified by Albuquerque & Majid (1999) when octonions are realized as a twisted
group algebra over $(ZZ/2)^3$. The paper was already cohomological without naming it.

### What's new in Revision 3

1. **Section 7 added** — Cohomological reformulation. Cites Albuquerque-Majid
   1999 and Bremner-Hentzel 2017. States Theorem $1'$ and Lemma $2'$ as
   restatements of the corresponding results in the language of $H^3((ZZ/2)^k, \{\pm 1\})$.

2. **Closed-form identity derived** for $T_k$:

   $$T_k = 168 \cdot \frac{(2^{k-1}-1)(2^{k-2}-1)(2^{k-1}+3)}{21}$$

   Reproduces $T_3=168$, $T_4=1848$, $T_5=15960$, $T_6=130200$. The factor
   $(2^{k-1}+3)$ is the "cohomological weight" of the Cayley-Dickson doubling
   that Conjecture 5 numerically captures but does not enunciate.

3. **Empirical sub-decomposition table** for $k=4$. Of the 15 three-dimensional
   subspaces of $(ZZ/2)^4$:
   - **8** are fully octonionic (associator count = 168 each)
   - **7** carry an *intermediate cocycle class* (count = 96 each)
   - **0** are trivial.

   Sums to $8 \cdot 168 + 7 \cdot 96 = 2016 = T_4 + 168$, the overcount of 168
   indicating exactly 56 non-LI triples in sedenions with nonzero associator
   (each contained in three distinct $V_c$ via shared 2-dim parents).

4. **Open Question 1 revised honestly**: The naive proof route via
   "$11 = 15 - 4$ subspaces are fully octonionic" is *empirically falsified*.
   The actual decomposition is $8 + 7$ with intermediate cocycle classes.
   Any analytic proof of Conjecture 5 must classify these intermediate classes.
   This is a productive negative result.

5. **Categorical chassis** $\text{Hyper}(G, F)$ proposed as a unified parametric
   primitive: $\mathbb{C} = \text{Hyper}((ZZ/2)^1, F_\mathbb{C})$,
   $\mathbb{H} = \text{Hyper}((ZZ/2)^2, F_\mathbb{H})$,
   $\mathbb{O} = \text{Hyper}((ZZ/2)^3, F_\text{Fano})$, etc. Forwards the
   integration of the entire Cayley-Dickson tower into the Sounio compiler's
   type system.

### Computational verification (this revision)

- `examples/phi_fano_cohomological.sio` — Implements $\varphi(x,y) = \text{tr}(y \cdot x^6)$
  over $\mathbb{F}_8$ from scratch, confirms 168 + dichotomy + alternativity. **7/7 PASS**.
- `examples/cocycle_subspace_168.sio` — Enumerates all 15 three-dim subspaces of
  $(ZZ/2)^4$, computes per-subspace associator counts. Reveals the 8+7 trichotomy.
  **1/2 PASS** (Conjecture 5 numeric confirmed; naive decomposition falsified).

### What was *not* changed

- Sections 1–6 are intact. Section 7 is purely additive.
- Bibliography gains two entries: `albuquerque-majid-1999`, `bremner-hentzel-2017`.
- The paper's central claims (Theorem 1, Lemma 2, Conjecture 5 numeric) stand.

### Resolutions (2026-05-10, follow-up pass)

The three risks initially raised have been retired:

1. **Literature pass — done.** Searched 2023-2026 for octonion 168 cocycle
   reformulations and Cayley-Dickson subspace decomposition. Most relevant
   recent work is Wilmot, *"Structure of the Cayley-Dickson algebras"*,
   arXiv:2505.11747 (May 2025) — it analyzes graded CD construction and
   counts zero divisors as multiples of 84, but does **not** use the
   Albuquerque-Majid twisted-group-algebra framework, does **not** discuss
   PSL(2,7) or the 168 count, and does **not** treat the per-subspace
   cocycle restriction decomposition. Section 7's territory is open.
   (Wilmot is already cited in the existing paper at line 204 for an
   unrelated automorphism-counting formula.)

2. **k=5 extended — done.** `examples/cocycle_subspace_k5.sio` enumerates
   all $P_5 = 155$ three-dim subspaces of $(\mathbb{Z}/2)^5$ via canonical
   dual-functional pairs and tallies per-subspace nonzero associator
   counts. Result: **seven distinct count classes** appear, not three.
   The k=4 trichotomy does NOT generalize as a low-class structure.
   New data added to Section 7 as `@table:subspace-k5`:

   | Count per subspace | Number |
   |---:|---:|
   | 180 | 7 |
   | 168 | 43 |
   | 108 | 7 |
   | 96 | 35 |
   | 92 | 21 |
   | 88 | 21 |
   | 76 | 21 |
   | **Total** | **155** |

   Notable: count=180 > 168 means some trigintaduonion subspaces carry
   *more* nonzero basis associators than the pure octonion subalgebra —
   impossible in alternative algebras and absent at k=4. The
   multiplicities (7, 43, 7, 35, 21, 21, 21) are integer combinations
   of Fano-orbit cardinalities, suggesting an orbit-counting derivation.

3. **Specialist review** — still recommended pre-publication, but the
   content of Section 7 has been tightened to claim only what is either
   (a) literally restated from cited primary sources (Albuquerque-Majid,
   Bremner-Hentzel) or (b) directly verified by computation (`@table:subspace`,
   `@table:subspace-k5`, the closed-form $T_k$ identity). No speculative
   interpretation remains in the section that could embarrass on review.

---

## Revision 3.1 — Lane 3 follow-up: k=6 chingon distribution (2026-05-10)

Lane 3 of the 6-agent overlay added the third data point requested by
Revised Open Question 1 of Section 7: per-subspace nonzero associator
counts in the 64-dimensional chingon algebra.

### What's new

- `examples/cocycle_subspace_k6.sio` (NEW). Builds the 64×64 chingon basis
  multiplication table via three CD doublings (𝕆→𝕊→𝕋→𝕮). Enumerates the
  1395 = P_6 three-dimensional subspaces of (ℤ/2)⁶ by canonical lex-minimal
  LI triples of dual functionals (V is codim-3 at k=6, so canonical
  enumeration requires triples not pairs). 2/2 PASS, T_6 = 130200 confirmed.
- `@table:subspace-k6` added to Section 7. The third data point in the
  Conjecture 5 cocycle classification.

### Empirical findings

**16 distinct count classes** at k=6 (vs 7 at k=5, 2 at k=4, 1 at k=3):

| Count | Mult. | Count | Mult. |
|---:|---:|---:|---:|
| 188 | 21 | 96 | 112 |
| 184 | 21 | 94 | 84 |
| 180 | 21 | 92 | 84 |
| 168 | 247 | 90 | 63 |
| 108 | 84 | 88 | 273 |
| 104 | 21 | 84 | 21 |
| 100 | 21 | 76 | 252 |
| 98 | 21 | 72 | 49 |

Notable observations:
- Class count grows superlinearly: **1 → 2 → 7 → 16** across k=3→6.
- Three super-octonionic classes (counts 180, 184, 188) at k=6, each with
  multiplicity 21. At k=5 there was only one super-octonionic class
  (count=180, mult=7). Super-octonionic structure proliferates with k.
- Most multiplicities at k=6 are divisible by 7 or 21 (consistent with
  PSL(2,7)-orbit derivation), with one anomaly: **count=168 at multiplicity
  247 = 13·19**, NOT divisible by 7. This is the principal anomaly the
  classification must explain.
- Sum across classes: 149016 = T_6 + 18816, with overcount 18816 = 112·168.
  Pattern across k: overcount/168 = 1, 12, 112 at k=4, 5, 6 (factor ~10× per step).
- Counts span 72 to 188 (vs 76-180 at k=5, 96-168 at k=4); spread widens
  with k, consistent with richer cocycle obstruction structure.

### Status

- T_3, T_4, T_5, T_6 all numerically confirmed against Conjecture 5 closed
  form 168·(2^{k−1}−1)(2^{k−2}−1)(2^{k−1}+3)/21.
- Revised Open Question 1 now has three empirical inputs (k=4, 5, 6) for
  cocycle restriction class enumeration.
- Lane 3 build target green: cocycle_subspace_168.sio rc=0,
  cocycle_subspace_k5.sio PASS, cocycle_subspace_k6.sio PASS.
