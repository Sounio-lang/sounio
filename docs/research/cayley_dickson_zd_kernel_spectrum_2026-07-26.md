<!-- docs:meta
topic_id: repo.docs.research.cayley-dickson-zd-kernel-spectrum-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cayley-dickson-zd-kernel-spectrum-2026-07-26
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The Kernel Spectrum of Zero-Divisor Primitives in Cayley-Dickson Algebras

Status: result, 2026-07-26. Exhaustively verified to dimension 128; proofs below.

## Statement

Let `A_k` be the Cayley-Dickson algebra of dimension `n = 2^k` over ℝ, with basis `e_0 … e_{n-1}`.
Call `u = e_a + s·e_b` (with `1 ≤ a < n/2 ≤ b < n`, `s = ±1`) a **primitive**, and let
`L_u(v) = u·v` be left multiplication.

> **Theorem.** For every primitive `u` of `A_k` (`k ≥ 4`),
> ```
> dim ker(L_u) ∈ { 0 } ∪ { 4 + 8i : 0 ≤ i ≤ 2^(k−4) − 1 },
> ```
> the maximum being `n/2 − 4`, and there are exactly `2^(k−4)` non-zero values.
> `dim ker(L_u) = 0` **iff** `b = n/2` or `a = b − n/2`.

Verified exhaustively for `n = 8, 16, 32, 64, 128`:

| `n` | non-zero kernel dimensions | count | degenerate primitives |
|---|---|---|---|
| 8 (𝕆) | — (none) | 0 | all (𝕆 is a division algebra) |
| 16 (𝕊) | {4} | 1 = 2⁰ | 14 = n−2 |
| 32 | {4, 12} | 2 = 2¹ | 30 = n−2 |
| 64 | {4, 12, 20, 28} | 4 = 2² | 62 = n−2 |
| 128 | {4, 12, …, 60} | 8 = 2³ | 126 = n−2 |

## Proof

Write `e_i·e_j = ε(i,j)·e_{i⊕j}` (the index law; immediate from the CD recursion),
`cj(0) = +1`, `cj(x) = −1` for `x > 0`, and
`χ(x,y) := ε(x,y)ε(y,x) = −1` if `x,y ≠ 0` and `x ≠ y`, else `+1`.

**Lemma 1.** `L_{e_a}² = R_{e_a}² = −I` for all `a > 0`.
*Induction on `k`.* Base `A_1 = ℂ`: `L_{e_1}²(z) = i(iz) = −z`.
Step, with `(p,q)(r,t) = (pr − t̄q, tp + q r̄)`:
- `u = (x,0)`, `x = e_a`, `a>0`: `L_u(r,t) = (xr, tx)`, so
  `L_u²(r,t) = (x(xr), (tx)x) = (−r,−t)` by the induction hypothesis.
- `u = (0,y)`, `y = e_b` (including `y = 1`): `L_u(r,t) = (−t̄y, y r̄)`; using that conjugation is
  an anti-automorphism, the components come to `−(rȳ)y = R_y²(r) = −r` and `y(ȳt) = −L_y²(t) = −t`. ∎

The proof uses **no associativity** — only the inductive identities. This is why it survives in 𝕊 and
above, where even alternativity has failed.

**Lemma 2 (reduction to an eigenspace).** Let `Q := −L_{e_a}∘L_{e_b}`. Then
`ker(L_u)` is the `(−s)`-eigenspace of `Q`.
*Proof.* `L_u v = 0 ⟺ e_a v = −s·e_b v`. Apply `L_{e_a}` and use Lemma 1:
`−v = −s·e_a(e_b v)`, hence `e_a(e_b v) = s·v` (as `s² = 1`), i.e. `Qv = −s·v`. ∎

**Lemma 3 (cycle structure).** By the index law the permutation underlying `L_{e_a}` is
`σ_a(j) = a ⊕ j`, so that of `Q` is `σ_aσ_b(j) = a⊕b⊕j`. Hence `(σ_aσ_b)² = id`: an **involution**.
A fixed point would need `a⊕b = 0`, i.e. `a = b`, impossible since `a < n/2 ≤ b`. Therefore `Q` has
exactly `n/2` cycles, **all of length 2**. ∎

**Theorem A.** On a 2-cycle `{j, Qj}` with sign product `pr`, `Q²|span = pr·I`; so `pr = +1` gives
eigenvalues `±1` (contributing exactly 1 dimension to the `(−s)`-eigenspace) and `pr = −1` gives
`±i` (contributing 0 over ℝ). With Lemma 2:
> `dim ker(L_u) = c₊ :=` number of 2-cycles of `Q` with sign product `+1`.
In particular `dim ker` does not depend on `s`. ∎

### The sign combinatorics

Expanding by the four CD sign rules, with `m′ = a⊕b′` where `b = n/2 + b′` (verified exact):
```
pr(t)   = −cj(t)cj(m′⊕t) · ε(b′,t) ε(b′⊕t,a) ε(m′⊕t,b′) ε(a,a⊕t)      (t < n/2)
pr(n+t) = −cj(t)cj(m′⊕t) · ε(t,b′) ε(a,b′⊕t) ε(b′,m′⊕t) ε(a⊕t,a)
```

**Lemma P (periodicity).** The `cj` factors agree, so
`pr(t)·pr(n+t) = χ(b′,t)·χ(b′⊕t,a)·χ(m′⊕t,b′)·χ(a,a⊕t)`.
Each factor is `+1` exactly on: ① `t∈{0,b′}` ② `t∈{b′,m′}` ③ `t∈{m′,a}` ④ `t∈{a,0}`.
If `t ∈ {0,a,b′,m′}` **exactly two** factors are `+1`; otherwise all four are `−1`. Either way the
product is `+1`. Hence `pr` is `n`-periodic and `c₊ = #{t < n : pr(t) = +1}`. ∎

**Lemma W (the twist is a Klein subgroup).** Squares cancel, leaving
`w(t) := pr⁽ᵏ⁺¹⁾(t)·pr⁽ᵏ⁾(t) = −cj(t)cj(m′⊕t)·χ(b′⊕t,a)·χ(m′⊕t,b′)`, and evaluating the five cases
gives `w = +1` **exactly** on
```
W₊ = {0, a, b′, a⊕b′} = ⟨a, b′⟩ ⊆ (𝔽₂ⁿ, ⊕),   |W₊| = 4.
```
Moreover `pr⁽ᵏ⁾ = −1` on all of `W₊`: at `t=0`, `pr⁽ᵏ⁾(0) = −ε(a,b′)ε(b′,m′) = ε(a,b′)ε(m′,b′) = −1`
by `R² = −I`; at `t=a`, `pr⁽ᵏ⁾(a) = −ε(b′,a)ε(a,m′) = −ε(a,b′)² = −1` by `L² = −I`; the remaining two
follow from `pr(t) = pr(m′⊕t)`. ∎

**Recurrence.** Split `{t < n}`. On `W₊` (4 points) `pr⁽ᵏ⁾ = −1` and `w = +1`, so `pr⁽ᵏ⁺¹⁾ = −1`:
they contribute nothing. Off `W₊`, `w = −1` so `pr⁽ᵏ⁺¹⁾ = −pr⁽ᵏ⁾`, and since `pr⁽ᵏ⁾` is `−1`
throughout `W₊`, all `2c₊⁽ᵏ⁾` of its `+1`s lie outside. Hence
> **`c₊⁽ᵏ⁺¹⁾ = (n − 4) − 2·c₊⁽ᵏ⁾`.** ∎

**Closed form.** With `S_k` the value set over all index pairs `≥1`, the two transfer rules
(`2c` within a half, `(n−4) − 2c` across halves) give by induction
`S_k = {4t : 0 ≤ t ≤ 2^(k−3) − 1}`, and restricting to primitives (across halves only)
```
{ (2^(k−1) − 4) − 8t : 0 ≤ t ≤ 2^(k−4) − 1 } = { 4 + 8i : 0 ≤ i ≤ 2^(k−4) − 1 }.  ∎
```

**Where the constants come from.** The `−4` in the recurrence is `|W₊|`, the order of the Klein
subgroup `⟨a,b′⟩`; the `8` in `{4+8i}` is `2·|W₊|`. The arithmetic of the spectrum is the order of a
group.

## Remarks

**Degeneracy is derived, not postulated.** `c₊ = 0 ⟺ b′ = 0 or a = b′` — exactly `n−2` primitives.
For `n = 16` this reproduces the validity conditions used in `formal/lean4/SounioZeroDivisorBridge.lean`
(`hi ≠ 8` and `lo⊕hi ≠ 8`), which the repository states as a definition.

**Cross-validation.** An independent reconstruction of the algebra in exact rational arithmetic
reproduces the repository's Lean results: 84 valid primitives in 𝕊, each with exactly 4 annihilators
(`prim_count_84`, `every_primitive_has_4_annihilators`). It further shows `dim ker(L_u) = 0` for a
generic element such as `1 + e₃` — annihilation is a rare, structured property, not a generic one.

**Structure of the subspaces (n = 16).** The 84 primitives induce only **42 distinct** 4-dimensional
annihilated subspaces — exactly two primitives per subspace. The mutual-annihilation graph has degree
4 and **maximum clique 2**: no three primitives annihilate one another pairwise. At most **3**
kernels are linearly independent (spanning 12 of 16 dimensions); a full decomposition into four
independent annihilated subspaces does **not** exist.

**Scaling.** The number of pairwise-independent kernels of the smallest class grows 3 → 5 → 13 for
`n = 16, 32, 64`, i.e. roughly `n/5`, at about 75% of the `n/4` bound.

**Kernel-dimension classes.** The observed dimension classes are `{4 + 8i}` with `2^(k−4)` of them —
homogeneous in 𝕊 (all kernels 4-dimensional), bimodal at `n = 32` ({4, 12}), four classes at
`n = 64`. We did not find this spectrum characterised in the literature on Cayley-Dickson zero
divisors, which treats the geometry of the 𝕊 zero-divisor set (e.g. arXiv:2411.18881) rather than
kernel dimensions up the tower.

## Reproduction

Exact-arithmetic scripts (independent of the repository's Lean corpus) build the CD table by
recursion, compute `dim ker(L_u)` by rank over a prime field (entries are `0, ±1`, so the rank equals
the rational rank), and enumerate cycle structures. Verification covers **all** primitives for
`n ≤ 128`.
