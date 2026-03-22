<!-- docs:meta
topic_id: repo.docs.papers.main.168-theorem-preprint
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.main.168-theorem-preprint
-->

# The 168 Theorem: PSL(2,7) Governs Non-Associativity and Zero-Divisor Structure in the Cayley–Dickson Tower

**Demetrios C. Agourakis**

Biomaterials and Regenerative Medicine Post-Graduate Program,
Pontifícia Universidade Católica de São Paulo (PUC-SP), Sorocaba, SP, Brazil
Faculdade São Leopoldo Mandic, Campinas, SP, Brazil

D.C.A. ORCID: 0009-0001-8671-8878

March 2026

---

## Abstract

We prove that the number of non-associative basis triples in the octonion algebra is exactly 168 = |PSL(2,7)|, the order of the automorphism group of the Fano plane. The result follows from the regular (sharply transitive) action of PSL(2,7) on ordered non-collinear triples of the Fano plane, which are precisely the triples whose associator is nonzero. We establish a *binary property*: for unit octonion basis elements $e_i, e_j, e_k$ ($i,j,k \in \{1,\ldots,7\}$), the associator norm $\|[e_i, e_j, e_k]\|$ takes only the values $\{0, 2\}$ — no intermediate magnitudes exist. This sharp dichotomy extends to sedenions (the next Cayley–Dickson algebra, dimension 16), where all nonzero associator counts among basis elements are multiples of 168: the full sedenion count is $1848 = 11 \times 168$. Furthermore, the number of sedenion zero-divisor pairs of the form $(e_i \pm e_j)(e_k \pm e_l) = 0$ is exactly $336 = 2 \times 168$, which equals the number of nonzero cross-half associator triples — establishing a precise numerical link between the two fundamental algebraic breakdowns in the Cayley–Dickson tower. All computations are verified exhaustively in the Sounio programming language; the path product norm invariance theorem is machine-checked in Lean 4.

**Keywords:** octonions, sedenions, associator, Fano plane, PSL(2,7), Cayley–Dickson, zero divisors, non-associativity

**MSC 2020:** 17A75 (Composition algebras), 20B25 (Finite automorphism groups of algebraic, geometric, or combinatorial structures), 17D05 (Alternative rings)

---

## 1. Introduction

The Cayley–Dickson construction produces a tower of algebras by iterated doubling:

$$\mathbb{R} \xrightarrow{\times 2} \mathbb{C} \xrightarrow{\times 2} \mathbb{H} \xrightarrow{\times 2} \mathbb{O} \xrightarrow{\times 2} \mathbb{S} \xrightarrow{\times 2} \cdots$$

At each step, a structural property is lost: commutativity at the quaternions $\mathbb{H}$, associativity at the octonions $\mathbb{O}$, and the division property at the sedenions $\mathbb{S}$ (which acquire zero divisors) [1, 2]. The octonions are the last normed division algebra (Hurwitz, 1898) and have found deep applications in string theory [3], exceptional Lie groups [4], and quantum information [5].

The multiplication table of the octonions is encoded by the Fano plane — the unique projective plane of order 2, with 7 points and 7 lines (each containing 3 points). The automorphism group of the Fano plane is PSL(2,7) $\cong$ GL(3, $\mathbb{F}_2$), a simple group of order 168 [6]. This group acts on the octonion basis elements by permuting them while preserving the multiplication structure.

While the role of PSL(2,7) in octonion multiplication has been extensively studied [1, 4, 7], a remarkably simple combinatorial fact appears to have escaped explicit statement in the literature:

> **The number of ordered triples $(i,j,k) \in \{1,\ldots,7\}^3$ for which the octonion basis associator $[e_i, e_j, e_k] = (e_i e_j)e_k - e_i(e_j e_k)$ is nonzero is exactly 168.**

This is not a coincidence — it is a consequence of PSL(2,7) acting *regularly* (sharply transitively) on ordered non-collinear triples of the Fano plane. We develop this observation and trace it through the Cayley–Dickson tower, finding that the number 168 serves as the fundamental quantum of non-associativity at every level.

### Contributions

1. **The 168 Theorem** (Section 2): The nonzero basis associator count equals |PSL(2,7)|, with a complete decomposition: 343 = 133 (alternativity zeros) + 42 (Fano-line zeros) + 168 (nonzero).

2. **The Binary Property** (Section 3): $\|[e_i, e_j, e_k]\| \in \{0, 2\}$ exactly, for both octonions and sedenions. No intermediate values exist.

3. **The Tower Extension** (Section 4): All sedenion nonzero associator counts are multiples of 168. Sub-decomposition: oct-oct = 168, sed-sed = 336 = 2 $\times$ 168, cross = 1344 = 8 $\times$ 168, total = 1848 = 11 $\times$ 168.

4. **The Zero-Divisor Coincidence** (Section 5): The number of sedenion zero-divisor pairs $(e_i \pm e_j)(e_k \pm e_l) = 0$ is 336 = 2 $\times$ 168, exactly matching the sed-sed nonzero associator count — linking non-associativity to zero divisors through PSL(2,7).

---

## 2. The 168 Theorem

### 2.1 Setup

Let $\{e_0, e_1, \ldots, e_7\}$ denote the standard octonion basis, where $e_0 = 1$ is the identity and $e_1, \ldots, e_7$ are imaginary units. The multiplication is determined by the Fano plane: for each oriented line $(i,j,k)$, we have $e_i e_j = e_k$ and $e_j e_i = -e_k$.

The **associator** of three octonions is defined as:
$$[a, b, c] = (ab)c - a(bc)$$

The octonions are *alternative*: $[a, a, b] = [a, b, b] = 0$ for all $a, b$. By the Artin theorem, any subalgebra generated by two elements is associative [1].

### 2.2 Counting nonzero basis associators

Consider all $7^3 = 343$ ordered triples $(i,j,k) \in \{1,\ldots,7\}^3$. We classify them:

**Layer 1: Repeated indices (133 zeros).** If any two of $\{i,j,k\}$ coincide, the associator vanishes by alternativity ($[a,a,b] = 0$) or flexibility ($[a,b,a] = 0$, which follows from alternativity in alternative algebras [1]). By inclusion-exclusion over the three pairwise-equality conditions:

$$|\{i=j\}| + |\{j=k\}| + |\{i=k\}| - |\{i=j=k\}| \cdot \binom{3}{2} + |\{i=j=k\}| = 3(49) - 3(7) + 7 = 133$$

**Layer 2: Fano-line triples (42 zeros).** Among the $7 \times 6 \times 5 = 210$ triples with all-distinct indices, those where $\{i,j,k\}$ forms a line of the Fano plane generate a quaternion subalgebra, which is associative. There are 7 lines, each contributing $3! = 6$ ordered triples:

$$7 \times 6 = 42$$

**Layer 3: Non-collinear triples (168 nonzero).** The remaining triples have all-distinct indices and do not lie on any Fano line:

$$343 - 133 - 42 = 168$$

### 2.3 The group-theoretic explanation

**Theorem 1.** *The number of nonzero basis associator triples equals* $|\text{PSL}(2,7)|$.

*Proof.* The Fano plane $\mathbb{F}$ has 7 points and 7 lines. An **ordered non-collinear triple** is an ordered triple $(p,q,r)$ of distinct points not all on a line. The count of such triples is $7 \times 6 \times 5 - 7 \times 6 = 168$ (total ordered distinct triples minus those on a common line).

PSL(2,7) $\cong$ Aut($\mathbb{F}$) acts on ordered non-collinear triples. We claim this action is **regular** (free and transitive):
- *Transitive*: PSL(2,7) acts transitively on ordered bases of $\mathbb{F}_2^3$ (ordered triples of vectors spanning the space), and non-collinear triples in $\mathbb{F}$ correspond to ordered bases of $\mathbb{F}_2^3$ under the identification $\mathbb{F} \cong \mathbb{P}(\mathbb{F}_2^3)$.
- *Free*: The stabilizer of an ordered basis of $\mathbb{F}_2^3$ in GL(3, $\mathbb{F}_2$) is trivial (an invertible linear map is determined by its action on a basis).

Since PSL(2,7) $\cong$ GL(3, $\mathbb{F}_2$), the action is regular, and the number of orbits is 1. Therefore:

$$|\{\text{nonzero associator triples}\}| = |\{\text{ordered non-collinear triples}\}| = |\text{PSL}(2,7)| = 168. \qquad \square$$

---

## 3. The Binary Property

**Theorem 2.** *For all $i, j, k \in \{1, \ldots, 7\}$:*
$$\|[e_i, e_j, e_k]\| \in \{0, 2\}$$

*Proof.* For basis elements $e_i, e_j$ with $i \neq j$ and both $\geq 1$, the product $e_i e_j = \pm e_m$ for some $m \in \{1,\ldots,7\}$ (by the Fano plane multiplication rule). Therefore:

$$(e_i e_j) e_k = (\pm e_m) e_k = \pm e_p \quad \text{for some } p$$
$$e_i (e_j e_k) = e_i (\pm e_n) = \pm e_q \quad \text{for some } q$$

Each term is $\pm$ a basis element. The associator $[e_i, e_j, e_k] = \pm e_p \mp e_q$, which is either:

- **Zero**, if $p = q$ and the signs match, or
- **$\pm 2 e_r$**, if $p = q$ and the signs oppose (giving $\|\pm 2e_r\| = 2$).

It remains to show that $p \neq q$ never occurs (i.e., the two parenthesizations always land on the *same* basis element). This follows from the Moufang identity structure: in any alternative algebra, the subalgebra generated by any two elements is associative [1, Artin's theorem], and the "destination" index depends only on $\{i,j,k\}$ (as an unordered set), not on the parenthesization. $\square$

**Remark.** The binary property means that non-associativity for basis elements is a **parity phenomenon** — a $\mathbb{Z}_2$ flip, not a continuous rotation. The associator does not move the result to a different dimension of the algebra; it reflects it within the same dimension.

---

## 4. The Tower Extension

We extend the analysis to the sedenions $\mathbb{S}$ (dimension 16), constructed by Cayley–Dickson doubling of $\mathbb{O}$: $\mathbb{S} = \mathbb{O} \oplus \mathbb{O}\ell$, where $\ell = e_8$ and $e_9, \ldots, e_{15}$ are the "sedenion extension" basis elements.

### 4.1 Sub-decomposition

Among all $15^3 = 3375$ ordered triples of imaginary sedenion basis elements:

| Sub-class | Range | Total triples | Nonzero | Factor of 168 |
|-----------|-------|---------------|---------|---------------|
| Oct-oct-oct | $i,j,k \in \{1,\ldots,7\}$ | 343 | 168 | $1 \times 168$ |
| Sed-sed-sed | $i,j,k \in \{8,\ldots,15\}$ | 512 | 336 | $2 \times 168$ |
| Cross (mixed) | at least one from each half | 2520 | 1344 | $8 \times 168$ |
| **Total** | $i,j,k \in \{1,\ldots,15\}$ | **3375** | **1848** | **$11 \times 168$** |

**Theorem 3.** *Every entry in the "Nonzero" column is a multiple of 168.*

The binary property (Theorem 2) extends to sedenions: exhaustive computation over all 3375 triples confirms $\|[e_i, e_j, e_k]\| \in \{0, 2\}$ for all sedenion basis elements.

### 4.2 The sed-sed count

The sed-sed count of 336 = 2 $\times$ 168 admits a structural explanation. The sedenion "upper half" $\{e_8, \ldots, e_{15}\}$ carries an induced multiplication structure from the Cayley–Dickson doubling. The product $e_{i+8} \cdot e_{j+8}$ involves two octonion multiplications (from the Cayley–Dickson formula $(a,b)(c,d) = (ac - \bar{d}b, da + b\bar{c})$), introducing a factor of 2 in the non-associativity count relative to the octonionic base.

---

## 5. The Zero-Divisor Coincidence

The sedenions are the first Cayley–Dickson algebra containing zero divisors: nonzero elements $z, w$ with $zw = 0$ [2, 8].

### 5.1 Enumeration

We exhaustively search all pairs of the form $z = e_i + s_1 e_j$, $w = e_k + s_2 e_l$ with $i < j$, $k < l$, $s_1, s_2 \in \{+1, -1\}$, and $i, j, k, l \in \{0, \ldots, 15\}$.

**Result:** Exactly **336** such pairs satisfy $zw = 0$.

$$336 = 2 \times 168 = 2 \times |\text{PSL}(2,7)|$$

### 5.2 The coincidence

The sedenion zero-divisor count (336) exactly equals the sed-sed nonzero associator count (336). Both are $2 \times |\text{PSL}(2,7)|$.

This establishes a precise numerical link between the two fundamental breakdowns in the Cayley–Dickson tower:

- **Non-associativity** (loss of $(ab)c = a(bc)$): governed by ordered non-collinear triples of the Fano plane.
- **Zero divisors** (existence of $ab = 0$ with $a, b \neq 0$): governed by pairs of elements in orthogonal sub-octonions.

Both are controlled by the same group-theoretic structure — PSL(2,7), the automorphism group of the Fano plane — appearing at scale 1$\times$ for octonionic non-associativity and at scale 2$\times$ for sedenion zero divisors and upper-half non-associativity.

**Remark.** de Marrais [9] independently counted 168 "primitive unit zero-divisors" in the sedenions, arranged in 42 "Assessors." Our 336 counts *ordered pairs*, so $336 = 2 \times 168$ is consistent with his enumeration (each de Marrais zero-divisor generates two ordered pairs with opposite sign choices). The fact that this number equals the upper-half nonzero associator count appears to be new.

---

## 6. Computational Verification

All results are verified by exhaustive computation in the Sounio programming language [10], a systems language with native support for Cayley–Dickson algebras and epistemic types.

- Octonion multiplication uses the Fano plane triples (1,2,4)(2,3,5)(3,4,6)(4,5,7)(5,6,1)(6,7,2)(7,1,3).
- Sedenion multiplication uses Cayley–Dickson: $(a,b)(c,d) = (ac - \bar{d}b, da + b\bar{c})$.
- All $7^3 = 343$ octonion and $15^3 = 3375$ sedenion associator norms are computed and classified.
- All $\binom{16}{2}^2 \times 4 = 57{,}120$ candidate zero-divisor pairs are tested.
- The path product norm invariance theorem (a consequence of norm multiplicativity applied to labeled graphs) is machine-checked in Lean 4 with 0 `sorry` statements.

Source code: [github.com/agourakis82/sounio](https://github.com/agourakis82/sounio), files `examples/oct_conjecture_test.sio`, `examples/sedenion_zero_div_hunt.sio`, `examples/sedenion_168_verify.sio`, `formal/OctonionGraph.lean`.

---

## 7. Discussion

### 7.1 PSL(2,7) as a universal constant

The number 168 appears at three distinct levels in the Cayley–Dickson tower:

1. As the count of nonzero octonion basis associators (Theorem 1).
2. As the denominator in Wilmot's formula $T_n = (2^n - 1)(2^n - 2)(2^n - 4)/168$ for Cayley–Dickson automorphism counting [11].
3. As one-half the sedenion zero-divisor pair count and the sed-sed nonzero associator count.

This suggests that PSL(2,7) is not merely the automorphism group of the Fano plane — it is the **fundamental structural constant** governing algebraic breakdown in the Cayley–Dickson tower. The Fano plane encodes how seven imaginary units can multiply consistently; PSL(2,7) counts the ways they can *fail* to associate consistently.

### 7.2 Physical interpretation

The Cayley–Dickson tower corresponds to physical regimes of decreasing algebraic regularity:

| Level | Algebra | Physics | Breakdown |
|-------|---------|---------|-----------|
| $\mathbb{R}$ | Reals | Classical mechanics | — |
| $\mathbb{C}$ | Complex | Quantum amplitudes | Commutativity |
| $\mathbb{H}$ | Quaternions | SU(2) gauge (weak force) | Ordering |
| $\mathbb{O}$ | Octonions | String theory transverse dimensions | Associativity |
| $\mathbb{S}$ | Sedenions | Beyond the division algebra boundary | Zero divisors |

The binary property (Theorem 2) implies that non-associativity at the basis level is a **discrete, $\mathbb{Z}_2$ phenomenon**, not a continuous deformation. In the octonion-to-sedenion transition, both non-associativity and zero divisors emerge with counts controlled by PSL(2,7), suggesting a unified algebraic mechanism underlying both breakdowns.

### 7.3 Open questions

1. Does the 168$\times$ pattern persist in the trigintaduonions ($\mathbb{T}$, dimension 32) and beyond? Specifically, are all nonzero basis associator counts in $\mathbb{T}$ multiples of 168?

2. Is there a *bijection* (not just a numerical coincidence) between sedenion zero-divisor pairs and nonzero sed-sed associator triples?

3. The factor 11 in 1848 = 11 $\times$ 168: does it have group-theoretic significance? Note that 11 does not divide |PSL(2,7)| = 168.

---

## References

[1] J. C. Baez, "The Octonions," *Bull. Amer. Math. Soc.*, vol. 39, no. 2, pp. 145–205, 2002.

[2] G. Moreno, "The zero divisors of the Cayley–Dickson algebras over the real numbers," *Bol. Soc. Mat. Mex.*, vol. 4, no. 1, pp. 13–28, 1998.

[3] M. B. Green, J. H. Schwarz, and E. Witten, *Superstring Theory*, Cambridge University Press, 1987.

[4] J. H. Conway and D. A. Smith, *On Quaternions and Octonions*, A.K. Peters, 2003.

[5] P. Lévay, M. Saniga, and P. Vrana, "Three-qubit operators, the split Cayley hexagon of order two and black holes," *Phys. Rev. D*, vol. 78, 124022, 2008. arXiv:0808.3849.

[6] L. E. Dickson, "The abstract group $G$ simply isomorphic with the alternating group on seven letters," *Bull. Amer. Math. Soc.*, vol. 5, pp. 120–124, 1899.

[7] B. L. Cerchiai, P. Fré, and M. Trigiante, "The role of PSL(2,7) in M-theory," *Fortschr. Phys.*, vol. 67, no. 8–9, 1900020, 2019.

[8] R. E. Cawagas, "On the structure and zero divisors of the Cayley–Dickson sedenion algebra," *Discuss. Math. Gen. Algebra Appl.*, vol. 24, pp. 251–265, 2004.

[9] R. P. C. de Marrais, "The 42 Assessors and the Box-Kites they fly: Diagonal axis-pair systems of zero-divisors in the sedenions' 16 dimensions," 2000. arXiv:math/0011260.

[10] D. C. Agourakis, "Sounio: A systems programming language for epistemic computing," 2026. https://github.com/agourakis82/sounio.

[11] A. Wilmot, "Automorphisms of sedenions," 2025. arXiv:2512.07210.
