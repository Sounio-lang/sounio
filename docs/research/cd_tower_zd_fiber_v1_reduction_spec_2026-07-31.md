<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-v1-reduction-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-v1-reduction-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — V1 reduced: ∀n spectral completeness needs only two integers

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `C_CLOSED__V1_REDUCED_TO_D_ALONE__NOT_CLOSED`
**Update 2026-08-02:** **(c) is CLOSED** — `SounioZDCollapse.parity_collapse`, kernel-checked ∀n. **V1 is now (d) alone.**
**Parents:** `cd_tower_zd_fiber_antisymmetry_lemma_spec_2026-07-31.md`, `cd_tower_zd_fiber_spectral_forall_n_progress_2026-07-26.md`, `cd_tower_zd_fiber_spectral_classifier_2026-07-26.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_v1_reduction_contract.py`

---

## 0. What this does and does not do

**V1 is not closed.** `#distinct A_σ spectra = 3·2^{n-5}` for all `n` remains **OPEN**. This
rung takes it apart and shrinks the hard half.

Two things come out:

1. **The count is orbit arithmetic, not a mystery.** Three of its four ingredients were already
   settled; only two statements carry content.
2. **The hard half needs only two integers.** What looked like a cospectrality problem — "no two
   of the `3·2^{n-5}` classes share a spectrum", the shape that is notoriously hard because
   cospectral graphs are common — is really a question about `tr(A_σ²)` and `tr(A_σ³)`.

---

## 1. The decomposition

```
#classes = (#Fano orbits) + (#seam orbits) − (#merges)
         =    2^{n-4}     +  (2^{n-4} − 1)  − (2^{n-5} − 1)
         = 2^{n-3} − 2^{n-5} = 3·2^{n-5}
```

| ingredient | status |
|---|---|
| (a) the orbit split `2^{n-4}` Fano + `2^{n-4}−1` fixed seams | **proven ∀n** — the orbit theorem |
| (b) the spectrum is constant on each orbit | **proven ∀n** — the automorphism is an algebra map, hence a graph isomorphism |
| (c) the even-weight seams merge, exactly `2^{n-5}−1` of them | parity-collapse law — **PROVEN ∀n 2026-08-02** (`SounioZDCollapse.parity_collapse`; both σ-lemmas discharged: (★) = `star_forall`, L2 = `L2_forall`) |
| (d) **nothing else merges** | **OPEN** — this rung reduces it |

The arithmetic between them is trivial. So **V1 ∀n = (c) ∀n and (d) ∀n**, and nothing else.
`W1` measures the whole decomposition at n = 6..9; `W2` measures (b).

---

## 2. The reduction

> **The pair `(tr A_σ², tr A_σ³)` induces exactly the spectral partition.**

`tr(A_σ²)` is the number of nonzero entries — twice the edge count, since entries are `±1`.
`tr(A_σ³)` is the signed triangle count. Both are computable without an eigendecomposition.

- **`W3`** — the partition into blocks of equal `(tr A², tr A³)` is **identical, block for
  block**, to the partition by full spectrum, at n = 6..9. This is the clause that matters:
  two partitions can have the same *number* of blocks and still differ, so an equal count would
  not have been enough.
- **`W4`** — `#distinct (tr A², tr A³) = 3·2^{n-5}` at **n = 6, 7, 8, 9, 10, 11** — six levels.

So (d) becomes: *is a two-integer invariant injective on the classes, for all n?* That is a
closed-form question, not a spectral one.

**`W5` — the pair is not padded.** `tr(A¹) = 0` for every fiber (the diagonal vanishes — itself
a theorem, `Asig_diag` in the parent rung), and `tr(A²)` **alone** gives exactly `2^{n-4}`
values, strictly fewer than `3·2^{n-5}`. Neither trace alone does the job.

**`W7` — the reduction is not vacuous.** The weaker invariant `(#vertices, tr A²)` under-
separates at every level. If a weaker invariant separated too, `W4` would carry no information.

---

## 3. Where the closed form stops — and it stops early

`W6`. On the stratum where `tr(A²)` is **constant** — the `y = 0` Fano class together with the
**weight-1 seams only** (`7 + (n−3)` fibers) — it equals

```
tr(A²) = (2^{n-1} − 2) · 4(2^{n-3} − 1)
```

i.e. the graph is `Dmax`-regular off its single isolated vertex, with `Dmax = 4(2^{n-3}−1)` the
core law already proven ∀n. Measured: `840, 3720, 15624, 64008` at n = 6,7,8,9.

**Off that stratum `tr(A²)` genuinely varies, and no closed form is derived here.** `W6` asserts
the variation as a clause so the boundary cannot be read as an oversight. Fitting the general
values appears to need the degree-histogram induction, which this lane records as **open**. Two corrections, both caught by the clause rather than by inspection:

1. An earlier draft called this closed form "evident". It is evident only on the stratum where
   the quantity does not vary — not the same thing.
2. That draft also put **all** seams in the stratum. `W6` failed (`rc=1`). The stratum is
   `y = 0` Fano plus the **weight-1** seams; the higher-weight seams take other values. The
   mistake came from reading a table whose top block happened to contain only weight-1 seams
   and generalising from it.

`tr(A³)` has no closed form here at all.

---

## 4. Not claimed

- **V1 is not closed** — but as of 2026-08-02 it is **(d) alone**. (c) is proven ∀n
  (`W8`: the theorem's hypotheses hold on *every* even-weight seam, the merge it forces
  really happens, and the count is `2^{n-5}−1`). (d) is untouched.
- **(d) is reduced, not proven.** Closing it needs **two** things, and both are open: a closed
  form for `(tr A², tr A³)` in terms of the fiber label, *and* a proof that that form is
  injective on the `3·2^{n-5}` classes for all `n`. Six levels of agreement is evidence, not a
  proof — a form that collides at some larger `n` would leave (d) exactly where it was.
- **(c) is CLOSED** (2026-08-02). `Φ` is an isomorphism of the *signed* annihilation
  graph between an even-weight seam fiber and its Fano partner, ∀n, with both sign
  identities discharged. An isomorphism of the signed graph forces equal spectra, which
  is the merge; the count `2^{n-5}−1` is the elementary fact that half of `y ∈ [1,2^{n-4})`
  have even popcount. `j = lsb W ≥ 3` for a seam and `j ≤ n−3` is automatic, since
  `j = n−2` forces `W = 2^{n-2}`, which has **odd** weight — so no stratum is left out.

- ⚠ **The merge map is NOT injective, and assuming it were would be wrong.** `τ` clears the
  **lowest set bit** of `y = W ≫ 3`, so `y = 5` and `y = 6` both land on `y = 4`: the
  `2^{n-5}−1` even-weight seams collapse onto only `2^{n-6}` Fano orbits. The subtraction is
  still right — each merged seam is removed *once*, regardless of where it lands — but the
  natural reading "each even-weight seam merges into its own Fano orbit" is false. `W8` asserts
  the non-injectivity so nobody re-derives the arithmetic from it.

- **What is left of V1 is (d) alone**: that the `2^{n-4}` Fano orbits are pairwise
  non-cospectral, and that no odd-weight seam merges.

---

## 6. Attacking (d): the degree splits, and its bulk has a closed form (`W9`, 2026-08-02)

§3 recorded that `tr(A²)` has a closed form only on the narrow stratum where it is constant, and
that the general case "appears to need the degree-histogram induction, which is OPEN". Two
structural facts move it.

**(i) Edge ⟺ resonance.** `A_sig`'s definition carries three conditions — `P1` symmetric, `P3`
symmetric, `P1 = P3` — and **the first two are automatic**. Each is a product of *two* commutation
signs, `χ(a,b)·χ(a⊕L,b⊕L)` and `χ(a,b⊕L)·χ(a⊕L,b)`, and `χ(x,y) = −1` for distinct nonzero `x,y`,
so both products are `+1` identically. `W9(a)`: 0 mismatches over 2 328 221 pairs. The edge
relation *is* resonance.

**(ii) So the degree is a resonance count, and the collapse theorem applies to it.**
`Qred_hi_ll` turns `Q(L,a,b) = 1` into `Qgen'(Llo,a,b,n−1) = −1` one level down — the exact object
this session's `collapse` governs. Off the six degeneracy lines it is a function of the **bottom
residues alone**, and every residue class has exactly `H/M` representatives (`M = 2^{j+2}`,
`j = lsb Llo`). Hence

```
deg(a)  =  (H/M) · #{ b₀ off the lines : ε·Qgen(Llo₀, a₀, b₀, j+2) = −1 }   +   R(a)
```

where `ε = (−1)^{popcount(Llo ≫ (j+2))}` and the bottom `Qgen` is known in closed form for **both**
label classes — `Qgen_pow2` (`≡ −1`) and `Q_three_pow2` (`−1` iff a low part vanishes or the two
agree), both Lean-proven this session.

`W9(b)`: the split is **exact** — 0 wrong `(a,Llo)` pairs. `W9(c)`: the first term equals that
closed form — 0 wrong out of 53 208 pairs checked (`n = 7,8,9`).

**Coverage.** The closed form holds for `j = 0` as well (0/8064 at `n ≤ 8`) — odd `Llo` is *half*
the labels, and an earlier pass had skipped it out of caution. So it covers **every fiber except
the single label `Llo = 2^{n-2}`** per level, where `j + 2 > n − 1` puts the bottom above the
resonance level.

### `R(a)` is not unknown — it is *determined*, but stratified (`W10`)

The collapse theorem takes `j` as a **free parameter**: it never requires `j = lsb Y`. So a pair
degenerate at `j = lsb Llo` — exactly where `R(a)` lives — may be **non-degenerate at a larger
`j'`**, and then the collapse determines its value at the finer bottom level `j'+2`. Running that
on every ordered pair:

| | |
|---|---|
| resolved at some `j' ≥ j`, collapse value exactly right | **0 wrong** |
| coset partners `b = a ⊕ Llo`, `Qgen'` value | **always `+1`** — never an edge |
| left over | **0** |

**Nothing is left over.** The minimal resolving level distributes as `j+0: 28.6%`, `j+1: ~30%`,
`j+2: ~21%`, `j+3: ~12%`, `j+4: ~5%`, `j+5: 2.4%` — and it **grows with `n`**.

So the honest state of (d)'s first half: `tr(A²)` is computable level by level from a *proven*
theorem plus one explicit never-an-edge class. What it is **not** is a *bounded* closed form —
that would need the strata to telescope.

---

## 7. The injectivity decomposes, and `tr(A²)` is parity-blind (`W11`)

(d)'s second half is "the pair is injective on the surviving classes". Its **fibre structure** is
now known exactly:

| | |
|---|---|
| **(I)** `tr(A²)` is **injective on the `2^{n-4}` Fano orbits** | verified `n = 6…9` |
| **(II)** `tr(A²)(seam y) = tr(A²)(Fano τy)`, `τy = y` with its **lowest set bit cleared** — for **every** seam, even weight and odd alike | 0 exceptions |
| **(III)** `tr(A³)` then separates the odd-weight seams inside those fibres | 0 exceptions |

So each `tr(A²)`-fibre is one Fano orbit `y` together with the seams `y + 2^i`, `i < lsb(y)`, that
`τ` maps onto it. The injectivity is therefore **(I) ∧ (III)** — two statements about a *known*
structure rather than one about an unknown one.

**And the conceptual point.** `τ` **always** preserves `tr(A²)`, but preserves the **spectrum**
only for even weight. So **`tr(A²)` is parity-blind** — it is an L1-level invariant, and the parity
the collapse law turns on is carried entirely by `tr(A³)`. That is the trace-side echo of this
lane's own `C1`/`C2` finding: L1 holds for every seam and does not see the parity; L2 does.

### ★ (I) is no longer a measurement — it *follows* from a recursion, by parity (`W12`)

With `T(n,y) = tr(A²)(n, Fano orbit y)/24`, `h = 2^{n-5}`, `c_n = 2^{n-3}−1`, `A(n) = T(n,0)`:

```
T(n, y)      =  4·T(n-1, y) + c_n          (y < h)      ← 0 wrong, n = 7…10
T(n, y + h)  =  A(n) − 4·T(n-1, y)         (y < h)      ← 0 wrong, n = 7…10
```

Given that, **injectivity is an induction**, and the only interesting case dies by parity:

- lower half — `y ↦ 4·T(n-1,y) + c_n` is affine, hence injective whenever `T(n-1,·)` is;
- upper half — `y ↦ A(n) − 4·T(n-1,y)` likewise;
- **cross** — a lower value equals an upper one iff `4T(n-1,y) + c_n = A(n) − 4T(n-1,y'')`, and
  `A(n) = 4A(n-1) + c_n`, so that is `T(n-1,y) + T(n-1,y'') = A(n-1)`. **Every `T` is odd**, so the
  left side is **even** and the right side is **odd**. Impossible.
- base `n = 6`: `T = [35, 19, 7, 23]`, distinct.

Oddness propagates: `4·odd + odd = odd`, `odd − 4·odd = odd`, and `c_n` is odd for `n ≥ 4`.

So (I) is not a coincidence checked at four levels — it is a **consequence of one recursion**, and
that recursion is now the single thing (I) needs. It is measured at `n = 7…10`.

`A(n)` solves to `T(n,0) = (2^{n-1}−2)(2^{n-3}−1)/6`, i.e. `tr(A²)(n,0) = 4(2^{n-1}−2)(2^{n-3}−1)`
— exactly §3's `W6` formula, which the recursion therefore reproduces.

### ★★ And the recursion is *derived* — four quadrants, eight rows, one sign law (`W13`)

It is not a coincidence about Fano representatives. It comes from a **raw-count** recursion that
holds for **every** label. With `N(m,W) = #{(a,b) ∈ [1,2^m)² : a ≠ b, Qgen'(W,a,b,m) = −1}` and
`e = 2^{m-1}`:

```
N(m, W)      =  4·N(m-1, W)                +  10e − 18       (W < e, label LOW)
N(m, W + e)  =  4(e−1)(e−2) − 4·N(m-1, W)  +   6e − 10       (label HIGH)
```

0 wrong at `m = 5,6,7`, over **all** `2^{m-1}−1` labels each.

**Mechanism.** Split `(a,b)` by the top bit at level `m` — four quadrants. The eight `Q'red` rows
send each to level `m−1`, and `N11`'s sign law says the sign is `−1` **exactly when the label is
high**. A `−1` sign turns "count the `−1`s" into "count the `+1`s" `= total − (−1)s` — which **is**
the reflection in the upper-half formula. The `ll` quadrant is `Q'red_low_ll`, which is
*unconditional*, so it contributes `N(m−1,W)` on the nose; the other three differ by constants
that depend **only on the level, never on the label**.

Subtracting the isolated-vertex pairs (`2^m − 2`, since `a = W` and `b = W` are excluded from
`A_sig`) gives `W12`'s form exactly, including `12(2^{n-2}−2) = 24·c_n` and
`A(n) = 4(2^{n-2}−1)(2^{n-2}−2)`.

### ★★★ And the low-label constant is derived too, from four closed forms (`W14`)

`10e − 18` is not a fitted integer. Quadrant by quadrant:

| quadrant | value | priced by |
|---|---|---|
| `ll` | `N(m-1,W)` **exactly** | `Q'red_low_ll` is *unconditional* |
| `lu`, `uu` | `N(m-1,W) + 3(e−2)` | three pieces, below |
| `ul` | `lu + e` | the `u = 0` slice (`e−1` terms, all `−1`) plus one boundary term |

and the three `(e−2)`s of `lu` are:

1. `v = 0` → `Qgen(W,0,u) = −1` — **`Qgen_zero_left`**
2. `v = u` → `Qgen(W,u,u) = −1` — **`Qgen_diag_neg`**
3. `v = u ⊕ W` → the **asymmetry**: unprimed `−1` (`Qgen_coset_left` + `Qgen_diag_neg`) but
   **primed `+1`** — **`Qgen'_coset_partner`**

with the `u = W` row-failure contributing **zero**. Summing: `4N(m-1,W) + 9(e−2) + e = 4N + 10e −
18`.

All four are Lean-proven ∀n, and the last one — *the coset partner is never an edge* — was `W10`'s
measured fact until today:

```lean
Qgen'_coset_partner : a ≠ 0 → a ⊕ W ≠ 0 → Qgen' W a (a ⊕ W) m = 1
```

two lines, the same shape as `Qgen'_diag`: two factors square away by `cdSq`, the other two are
self-pairings pinned by `sigma_self`.

**What remained unpriced** — the single `+1` boundary term in `ul` and the high-label constant
`6e − 10` — is closed in §8. Both fell to the same observation.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 8. Both constants, derived: one table prices every slice (`W15`, 2026-08-02)

`W14` priced the low-label constant piece by piece but left two things measured: a `+1` boundary
term in `ul`, and the whole high-label constant `6e − 10`. Both close at once, and the reason they
resisted is that I had been pricing slices *one at a time*.

### 8.1 The observation

Each of the eight reduction rows carries side conditions, and where they fail the row says nothing.
Those failure slices are exactly what the constants are made of. The point is:

> **Every failure slice lies on the twelve-condition locus where `Qgen = −1`.**

That locus was already closed ∀n, in two halves this lane proved long ago: the six `= 0`
degeneracies (`Qgen_degen`) and the six `= H` gap roots (`Qgen_H_left_*`, `Qgen_H_right_*`,
`Qgen_H_diff_*`, Tiers 9–10). Checking the fifteen low-label and sixteen high-label slices against
it is mechanical — `a = W′` means `a ⊕ W = H`, `u = 0` means `a = H`, `u = b` means `a ⊕ b = H`,
`u ⊕ v = W′` means `a ⊕ b ⊕ W = H`, and the rest are plain `= 0` degeneracies.

And on that locus `Q′` has a closed form, because `Qgen'_eq_chi` factors it through `Q` and two
commutation signs, both of which `chi_char` makes explicit. That is one rewrite:

```lean
Qgen'_on_neg : Qgen W a b m = -1 → Qgen' W a b m = -(chi (a ^^^ W) (b ^^^ W) m * chi a (b ^^^ W) m)
```

so, writing `c₁ = (a = W ∨ b = W ∨ a = b)` and `c₂ = (a = 0 ∨ b = W ∨ b = a ⊕ W)`,

> `Q′ = +1` **exactly when precisely one of `c₁`, `c₂` holds.**

One table, thirty-one slices. In particular the six degeneracy lines read `+1, +1, −1, −1, +1, +1`
for `a = 0`, `a = W`, `b = 0`, `b = W`, `a = b`, `b = a ⊕ W` — four of which are now named
theorems (`Qgen'_zero_left`, `Qgen'_label_left`, `Qgen'_zero_right`, `Qgen'_label_right`; the other
two were already `Qgen'_diag` and `Qgen'_coset_partner`).

### 8.2 The high-label ledger

With `e = 2^{m−1}`, `W = W′ + e`, `P′ = (e−1)(e−2)`, `N′ = N(m−1, W′)`, every high row carries a
**minus** sign (N11: the label is high), so counting `−1`s at level `m` becomes counting `+1`s at
level `m−1` — that is the reflection.

| quadrant | applies-part | failure `−1`s | total |
|---|---|---|---|
| `ll` | `(e−2)² − N′` | `2(e−2)` | `e² − 2e − N′` |
| `ul` | `(e−2)(e−3) − N′` | `3(e−2)` | `e² − 2e − N′` |
| `lu` | `(e−2)(e−3) − N′` | `4e − 7` | `e² − e − 1 − N′` |
| `uu` | `(e−2)(e−3) − N′` | `4e − 7` | `e² − e − 1 − N′` |

Summing: `4e² − 6e − 2 − 4N′ = 4P′ − 4N′ + 6e − 10`. **The high constant is derived.**

The one non-obvious step is the bridge the `lu` and `ul` quadrants need:

> `M = N′`, where `M` counts `Qgen(W′,·,·) = −1` over the five-line-free box.

`M` is counted over `(e−2)(e−3)` pairs and `N′` over `(e−1)(e−2)`, so this is not "same function,
same box". It holds because two `(e−2)`s cancel.

*Off* the sixth line `a ⊕ v = W′` the two functions agree, and that is a **theorem ∀n**, not a
measurement: `Qgen_symm` then `Qgen_eq_Qgen'`, whose five hypotheses — `a ≠ 0`, `a ≠ W`, `v ≠ W`,
`a ≠ v`, `a ⊕ v ≠ W` — are *exactly* the box, plus `Qgen_symm`'s `v ≠ 0`. *On* the sixth line they
differ: `Qgen = −1` while `Q′ = +1` (`Qgen'_coset_partner`) — and that surplus of `(e−2)` exactly
replaces the `(e−2)` that lemma A's `b = W′` row contributes (`Qgen'_label_right`, all `−1`) while
its `a = W′` row contributes none (`Qgen'_label_left`, all `+1`). So the bridge has **no measured
pointwise ingredient** — only the counting `|on6| = e−2`.

### 8.3 The low-label leftover

The `+1` boundary term was an artifact of writing `ul = lu + e`. The honest form is

```
ll = N′        lu = uu = N′ + 3e − 6        ul = N′ + 4e − 6        total = 4N′ + 10e − 18
```

and the gap between `ul` and `lu` is structural, not a fitted offset: `ul`'s `u = 0` slice is
`a = H`, fully degenerate over all `e−1` values of `b` (`Qgen_H_left_low`), whereas `lu`'s `v = 0`
sits *inside* the per-`a` degenerate set and so is already counted.

### 8.4 Scope, honestly

The pointwise closed forms are Lean ∀n and kernel-clean. The **counting** — slice sizes,
disjointness, coverage — is on paper, and `W15` pins it at `m = 5, 6, 7`: 55 labels × 2 parities,
0 violations, and it checks coverage (`Σ|slice| = |quadrant|`), not just the values. Writing that
check is what caught a genuine hole in my first pass: the high `ul` failure set was one pair short
(`(u,b) = (W′,W′)`, a `+1` pair, so the totals had survived the omission).

`W′ = 0` is the null control — `Qgen_degen` needs `W ≠ 0` — and the ledger fails there with 74
violations, so the clause is not vacuously wide. Even `W′` (6, 22) pass, confirming the derivation
uses only `W′ ≠ 0` and not the lane's `Llo = 8y+1` oddness. The `m = 7` sweep covers 9 of 63
labels; the clause string says so.

**Next step, and it is not a defect in the above.** The slice arithmetic needs `e ≥ 4`
(`(e−2)(e−4)` and friends), so `W13`–`W15` determine the recursion's *step* but pin no **base
level**. §9 supplies it.

---

## 9. The base case, and how far the closed form reaches (`W16`, 2026-08-02)

### 9.1 Where the descent bottoms out

The recursion sends `(m, W) → (m−1, W mod 2^{m−1})`. For an **odd** label — the lane's
`Llo = 8y+1` — the reduced label is odd at every level (`odd_stays_odd`), hence **never `0`**,
which is exactly the hypothesis `W15`'s null control violates; and at level 1 the box is empty
(`base_box_empty`). So every odd chain bottoms out in the **label-`2^k` family**.

### 9.2 The base case, proven ∀n

```lean
Qgen'_pow2_eq : k < m → 1 ≤ a,b < 2^m → a ≠ b →
    Qgen' (2^k) a b m = if a = 2^k ∨ b = a ^^^ 2^k then 1 else -1
```

`Q′` is `+1` on exactly two lines — `a = 2^k` (`Qgen'_label_left`) and `b = a ⊕ 2^k`
(`Qgen'_coset_partner`) — which are **disjoint** and of size `2^m − 2` each, and `−1` on
everything else (`Qgen'_off_lines`, new, plus `Qgen_pow2`, long in the tree). Hence

> `N(m, 2^k) = (2^m−1)(2^m−2) − 2(2^m−2) = (2^m−2)(2^m−3)`, **independent of `k`**.

### 9.3 The bridge to the graph — and the isolated vertex, derived

`A_sig`'s edge test is `Qgen(Llo | 2^{n−1}, a, b, n) = +1`. **This needs `Llo < 2^{n−1}`**, which
holds throughout the lane (`A_sig_fast` ranges `Llo` over `[1, 2^{n−1})`): only then is the `OR`
an addition, `Llo | 2^{n−1} = Llo + 2^{n−1}`, which is the shape `Qred_hi_ll` requires — without
it the conversion below is simply unavailable. `Qred_hi_ll` instantiated at `m−1`, whose only side
conditions are `b ≠ 0` and `b ≠ Llo`, converts the test to `Qgen'(Llo,a,b,m) = −1` — and it covers the
**whole `a = Llo` row**, where `Qgen'_label_left` gives `+1`. So the isolated vertex is now a
*consequence*, not measured structure. The one column the row cannot reach, `b = Llo`, is zero by
`A`'s symmetry while contributing exactly `2^m − 2` to `N` (`Qgen'_label_right`). Therefore

> `tr(A²) = N(m, Llo) − (2^m − 2)`,

checked on **both** label families, `n = 6..9`, 0 violations. At `Llo = 2^k` this is
`(2^m−2)(2^m−4)` — precisely `W6`'s constant stratum, whose members turn out to be exactly the
`y = 0` Fano class and the **pure-power-of-two seams** (`n−4` per level). `W6` had that measured;
it is now derived.

### 9.4 The closed form — Fano only, and the seam half is a declared negative

Unrolling `W15`'s recursion from this base collapses both branches into one homogeneous rule and
gives a **signed base-4 digit sum**:

> `tr(A²) = (2^m − 2)(2^m − 4) − E(m, Llo)`, `m = n−1`,
> `E(m,W) = Σ_{i = 2..m, bit_{i−1}(W)=1} (2^i−4)(2^i−8)·4^{m−i}·(−1)^{popcount(W ≫ i)}`
>
> **The lower bound `i ≥ 2` is part of the definition** (`contract.py:1006`,
> `range(2, m+1)`). This line previously omitted it. It is not cosmetic: an `i = 1` term
> for odd `W` would contribute `(2−4)(2−8)·4^{m−1} = 12·4^{m−1} ≠ 0`, and a reader
> reconstructing `E` from this line alone gets a formula that fails on every label —
> which is exactly what happened to me on 2026-08-03 while planning §26.
>
> Note also that the `i = 2` and `i = 3` terms vanish identically (`2^i−4` and `2^i−8`
> respectively), so **`E` depends only on bits ≥ 3 of `W`**, and `E(m, 8g+1) = E(m, 8g)`.

**Exact on the Fano family** — every label, `n = 6..10` in the clause and `n = 11` in-session,
1764 fibers, 0 mismatches, against the lane's own `A_sig_fast`/`traces23`. Fed the **raw** label
it is **false on every seam**, and `W16` asserts that as a *declared negative* rather than leaving
it to be found later.

The reason the raw label fails is structural: an even label's descent hits `W′ = 0` at an
**intermediate** step — `24 % 16 = 8`, then `8 % 8 = 0` — where the recursion is simply
inapplicable. Not at the bottom, and not only for pure powers of two.

**§10 supersedes the conclusion I drew from this.** It was the *argument*, not the formula.

### 9.5 Scope

The formula was verified *directly* against raw counts (all odd labels, `m = 2..8`) and against
`traces23` (`n = 6..11`); those checks do not route through the recursion. The recursion + base is
the **derivation**: its base is Lean ∀n, its counting step is on paper and pinned by `W15`.

The `n = 6..11` agreement is also the strongest evidence yet for the chain's one measured
pointwise link — that `A_sig`'s two symmetry conditions `P1 = P1ᵀ`, `P3 = P3ᵀ` are automatic. If
either ever failed, `res` would be strictly smaller than `{Qgen = +1}` and `tr(A²)` would come in
*below* the closed form. It does not, at six levels.

**This does not narrow (d).** `tr(A²)` is parity-blind (`W11`), so (d) still needs `tr(A³)`'s form
**and** the seam family. **(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 10. The seam half closes: one invariant covers every label (`W17`, 2026-08-03)

§9 reported the seam family open. That conclusion was wrong, and instructively so: **the formula
was right and the argument I fed it was wrong.**

### 10.1 The invariant

`N(m,W)` depends on `W` only through

> `g(W) = (W & (W−1)) ≫ 3` — **clear the lowest set bit, then take bits ≥ 3.**

For odd `W`, `W & (W−1) = W − 1`, so `g(8y+1) = y`: **`g` generalises the lane's `y`.** Every
label — seam included — therefore reduces to a Fano label `8·g(W)+1`, and §9.4's closed form
applies verbatim:

> `tr(A²)(n,W) = (2^m − 2)(2^m − 4) − E(m, 8·g(W)+1)`, `m = n−1`.

**487/487 labels, both families, `n = 6..9`, 0 mismatches** against `A_sig_fast`/`traces23`.
(This line previously read `987/987 … n = 6..10`. The clause loops `range(6, 10)`
(`contract.py:1115`), i.e. `n = 6..9`; 987 is the `n = 6..10` label total. `n = 10` was
verified in-session and the cap is declared in the clause text, but it is not what the
gate executes.)

### 10.2 It was read off the block structure, not fitted

`N` is constant on explicit blocks, and printing them shows the rule directly. At `m = 7` the
block of `65` is `{65..72, 80, 96}` — and `72 = 64+8`, `80 = 64+16`, `96 = 64+32` all clear to
`64`, exactly as `65` does. The pure powers of two all clear to `0` and so join the `y = 0` block,
which is precisely `Qgen'_pow2_eq`, the Lean base case of §9.2.

Two measured **negatives** got me here and both are kept in the clause:

* the natural scaling law `N(m, 2^t·V) = N(m−t, V)` is **refuted**, 0 of 31 at `m = 6`;
* plain `y = Llo ≫ 3` fails on **every** seam — so `g` is doing real work, not decoration.

### 10.3 Status

**`W6`'s open general form for `tr(A²)` is now closed on the whole label set.** What is verified
directly is the *formula*; the *derivation* is still base (Lean ∀n) + recursion (paper, `W15`).

**Still not (d):** `tr(A²)` is parity-blind (`W11`), so (d) needs `tr(A³)`. **(III) is untouched.
(d) is not closed, and V1 is not proven.**

---

## 11. Half of `g` is now a theorem: `Q′` is τ-equivariant (`W18`, 2026-08-03)

§10 found `g` empirically. Half of it is now proven ∀n, and the half that is *not* is stated
sharply rather than left implicit.

### 11.1 The theorem

```lean
Qgen'_tau : j < m → Y < 2^m → Y ≠ 0 → Y % 2^j = 0 → a,b < 2^m →
    Qgen' Y a b m = Qgen' (tau j Y) (tau j a) (tau j b) m
```

Three lines, because the tree already had every ingredient: **`star_forall`** gives
τ-equivariance for `Q`, **`tau_xor`** moves `τ` through the xors, and **`chi_tau`** says the two
commutation signs cannot see `τ` at all. The hypothesis `Y % 2^j = 0` is exactly *`j` at or below
the lowest set bit*.

`tau j` swaps bits `0` and `j`, so at `j = lsb(W)` it **moves the lowest set bit to position 0** —
precisely what `g` does before the `≫3` — and normalises the label to an **odd** one
(`tau_lsb_odd`, also proven), which is what keeps `odd_stays_odd`, hence `W′ ≠ 0`, available at
every level below. So

> `g W = (tau (lsb W) W) ≫ 3`, and `tau (lsb W) W = (W & (W−1)) + 1`.

### 11.2 What is *not* proven, plainly

1. **The counting step.** From the pointwise identity to `N(m,W) = N(m, τ_j W)` needs a
   bijection-to-cardinality argument — Finset territory this Mathlib-free file does not have.
2. **τ is sound but not complete.** Each `N`-block is **exactly four** τ-orbits (`m = 5,6,7`:
   16/4, 32/8, 64/16). The residual — that bits 1 and 2 of an *already odd* label are irrelevant
   — has no proof anywhere yet. That factor of four is the open half of `g`.
3. **The additive identity** `τ_lsb W = (W & (W−1)) + 1` is pure bit arithmetic, pinned in `W18`
   rather than proven in Lean.

### 11.3 A kept negative

My first guess at the residual — that the Fano/168 action lets the low 3 bits of *any* label vary
freely — is **refuted**. Adding that relation collapses **all** labels into a single orbit, merging
labels with demonstrably different `N` (21 witnessing pairs at `m = 5`). The factor of four is
*not* the naive Fano action.

`W18` also pins the Lean `tau` definition against the clause's own, because `K7` in this lane once
drew a wrong conclusion from a mismatched τ.

**§12 closes this factor of four.**

---

## 12. The factor of four is `GL(3,2)`, and a coboundary kills it (`W19`, 2026-08-03)

### 12.1 The group

The missing mechanism is **`GL(3,2)` acting on bits 0,1,2, identity above** — order 168, the
lane's own group, and **transitive on the seven nonzero low patterns**, so it merges the four odd
residues `1,3,5,7` that τ could not reach. Unlike §11.3's refuted guess it acts on the **label and
both points at once**, exactly as τ does.

> `⟨GL(3,2), τ_lsb⟩` is **sound and complete** against the `N`-block partition at `m = 5,6,7`
> (4/4, 8/8, 16/16 orbits vs blocks).

So `g` is now fully explained: `GL(3,2)` merges the odd residues, τ normalises every even label to
an odd one.

### 12.2 Why it costs no hypothesis — and this part is a theorem

`σ` itself is **not** invariant under these maps. It moves by a **coboundary**:

> `σ(p x, p y) = σ(x,y) · λ(x) · λ(y) · λ(x ⊕ y)`

`Q` and `Q′` are each a product of **four** σ's over a coset square in which the six λ values occur
exactly **twice** — so every λ squares away:

```lean
Qgen_of_coboundary  : (∀ x y, p (x ⊕ y) = p x ⊕ p y) → (∀ x, lam x = 1 ∨ lam x = -1) →
                      (∀ x y, cdSigma (p x) (p y) m = cdSigma x y m * lam x * lam y * lam (x ⊕ y)) →
                      Qgen  (p W) (p a) (p b) m = Qgen  W a b m
Qgen'_of_coboundary : … → Qgen' (p W) (p a) (p b) m = Qgen' W a b m
```

Both ∀n, kernel-clean, for an **arbitrary** F2-linear `p` and an **arbitrary** sign `λ` — they do
not even need `Classical.choice`. That is the cancellation, proven.

### 12.3 What is still measured

That `σ` **does** move by a coboundary under `GL(3,2)`. That is now the single open statement
behind `g` — one clean σ-level fact rather than a vague factor of four — verified **168/168** at
`m = 4,5` by solving the F2 system for λ, with a **non-linear null control** that fails both the
equivariance test and the coboundary test.

The counting step (§11.2 item 1) is also still outside Lean.

### 12.5 The coboundary itself is now proven ∀n — see §13.

### 12.4 By-catch: `star_forall`'s hypothesis is not tight

The bit-swap `(0↔1)` **is** `tau_1`, and it is equivariant for **odd** `Y` too — which
`star_forall`'s `Y % 2^j = 0` excludes. 0 mismatches at `m = 5,6`. The theorem is true more widely
than it is stated; nothing downstream depends on the gap, but it is worth knowing.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

**(III) is still untouched. (d) is not closed, and V1 is not proven.**

---

## 5. Reproduce

```sh
python3 scripts/research/cd_tower_zd_fiber_v1_reduction_contract.py
```

`W0` pins this rung's builders to the in-tree `sign_table`/`A_sig` entrywise, so every clause is
measured against the lane's own generator. `W4` at n = 11 dominates the runtime (~4 min solo);
the spectral clauses stop at n = 9 because they need eigendecompositions and the traces do not —
which is precisely the point of the reduction.

---

## 13. The coboundary, proven ∀n (`W20`, 2026-08-03)

§12 left exactly one measured statement behind `g`: that σ **does** move by a coboundary. The
∀n content of that is now a theorem, and for concrete maps it closes completely.

### 13.1 Level 3 decides every level

`cdSigma`'s recursion strips the **top** bit and recurses on the residues. A map confined to bits
0,1,2 commutes with that split *entirely* — it preserves the `≥ half` tests, the `= 0` tests and
the residues. So the coboundary property is **inherited** from each level to the next, and the
whole ∀n statement collapses to a check at **level 3**:

```lean
sigma_coboundary_up :
  (p 0 = 0) → (∀ x ≠ 0, p x ≠ 0) → (lam 0 = 1) → (∀ x, lam x = ±1) →
  (p preserves levels) → (p commutes with the seam) → (lam ignores the seam) →
  (level-3 base) →
  ∀ k x y, x,y < 2^(k+3) → cdSigma (p x) (p y) (k+3)
                         = cdSigma x y (k+3) * lam x * lam y * lam (x ⊕ y)
```

Its four branches are exactly `R_ll`, `R_lu`, `R_ul`, `R_uu` — already in the tree.

### 13.2 Two generators, closed completely

The level-3 base is **finite**, so for a concrete map it falls to `decide`. Writing a low-block
map as `lowMap t x = 8·(x/8) + t (x % 8)` and a low-block sign as `lowSign l x = l (x % 8)` makes
every structural hypothesis `omega`-arithmetic instead of bit-fiddling. Both generators close:

| generator | table | λ |
|---|---|---|
| `sigma_coboundary_trans` — transvection `e₂ ↦ e₂ ⊕ e₀` | `(0,1,2,3,5,4,7,6)` | `−1` on `{5,7}` |
| `sigma_coboundary_cyc` — 7-cycle `e₀↦e₁, e₁↦e₂, e₂↦e₀⊕e₁` | `(0,2,4,6,3,1,7,5)` | `−1` on `{6,7}` |

Both ∀n, kernel-clean, plain `decide` (no `native_decide`).

### 13.3 What is still not a single Lean statement

`W20` checks the two links the Lean does not:

* these two generators **generate `GL(3,2)`** — closure gives exactly 168, equal to the group
  `W19` enumerates;
* the coboundary property is **closed under composition**, `λ_{p∘q}(x) = λ_q(x)·λ_p(q x)`,
  verified on the product of the two.

Those two facts plus the two theorems give all 168. But the composition step needs `lowMap`'s
F2-linearity inside Lean, which is bit-work not done here — so **"all 168" is not yet one Lean
statement**, even though every piece of it is either proven or checked.

The counting step (§11.2 item 1) also remains outside Lean.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 14. All 168, closed in Lean (`W21`, 2026-08-03)

§13 proved the coboundary ∀n for the two generators and left "all 168" outside Lean, because
composing two coboundaries needs `lowMap`'s F2-linearity. That is now proven, and the chain closes.

### 14.1 `lowMap` is linear

```lean
lowMap_lin : (∀ v < 8, t v < 8) → (∀ u v < 8, t (u ⊕ v) = t u ⊕ t v) →
             lowMap t (x ⊕ y) = lowMap t x ⊕ lowMap t y
```

It follows from four core bit facts — `shiftRight_xor_distrib`, `shiftLeft_xor_distrib`,
`testBit_mod_two_pow`, `two_pow_add_eq_or_of_lt` — once `8·a + b` with `b < 8` is recognised as a
**disjoint xor**. That last step (`add8_xor`) is the one that makes the whole thing go.

### 14.2 The class, and the payoff

| theorem | content |
|---|---|
| `sigma_coboundary_comp` | the coboundary composes: `λ_{p∘q}(v) = l₂ v · l₁ (t₂ v)` |
| `LowCob` | inductive class: the two generators, closed under composition |
| `lowCob_sigma` | **every** member carries the coboundary, at every level |
| `Qgen'_lowCob` | **`Q′` is invariant under every member, ∀n** |

`Qgen'_of_coboundary_lt` is the bounded restatement the last step needs — the coboundary is only
available on the box, so the unbounded form of §12.2 could not be applied directly.

All kernel-clean. So the chain is complete:

> σ moves by a coboundary (∀n) → the four σ's of `Q′` cancel it (∀n) → `Q′` is invariant under the
> class (∀n) → the seven nonzero low residues merge, which **contains** §11's residual factor of
> four.

`W21` verifies the merge itself: `N(m, 8y+r)` is **constant in `r = 1..7`** for every `y`, at
`m = 5,6,7`; and it pins the Lean `lowMap` against `W19`'s index permutation, checks linearity for
all 168 tables, checks `lowMap t₁ ∘ lowMap t₂ = lowMap (t₁∘t₂)`, and fails on a non-linear control.

### 14.3 What is still not Lean

* that `LowCob` is **exactly** `GL(3,2)` — a finite closure computation, done in `W20`
  (168 elements, equal to the enumerated group);
* the **counting step**, from pointwise `Q′`-invariance to equality of `N` — Finset cardinality,
  which this Mathlib-free file does not have.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 15. The counting step, proven — `g` closes end to end (`W22`, 2026-08-03)

`Qgen'_lowCob` is **pointwise**; `g` is about the **count**. Bridging them normally means
`Finset` cardinality, which this Mathlib-free file does not have. It does not need it.

### 15.1 The machinery

```lean
sumLt : Nat → (Nat → Nat) → Nat          -- plain recursive bounded sum
sumLt_add     : sumLt (n+m) f = sumLt n f + sumLt m (fun i => f (n+i))
sum8_perm     : the 8-term block sum is invariant under every table in the class
sumLt_lowMap  : reindexing a bounded sum by `lowMap t` changes nothing, ∀n
lowMap_inj    : `lowMap t` is injective (linearity + trivial kernel)
Ncnt_lowCob   : Ncnt (lowMap t W) (k+3) = Ncnt W (k+3)
```

`sum8_perm` is proven by induction over the class: the two generators are **concrete**, so each
base case is eight terms reordered and closes by `omega`. `sumLt_lowMap` is then an induction on
the level — base is the 8-block permutation, step is `lowMap_seam` plus `sumLt_add`. Applying it
once per argument, with injectivity and `lowMap t 0 = 0` to transport the guards, gives the result.

### 15.2 `g` is now proven end to end

> σ moves by a coboundary (§13) → the four σ's of `Q′` cancel it (§12) → `Q′` is invariant under
> the class (§14) → **the count is invariant** (here).

With `Qgen'_tau` (§11) handling the even labels, **both halves of
`g(W) = (W & (W−1)) ≫ 3` are theorems.**

### 15.3 An honest weakening, caught by this clause's own null control

My first null control here was **vacuous**: I asserted that the non-linear low permutation would
break count-invariance. It does not. Count-invariance is *weaker* than pointwise `Q′`-invariance
and holds for **all 5040** permutations of the low block fixing 0, not merely the 168 linear ones.

So `LowCob` is **sufficient but not necessary** for this conclusion. It *is* necessary for the
pointwise statement — `W19` pins that exactly 168 of the low maps are `Q′`-equivariant, and the
non-linear ones fail pointwise — and the pointwise statement is what the proof actually uses.

The load-bearing hypothesis is **confinement to the low block**: a map touching bit 3 breaks the
count at *every* label (62/62 at `m = 6`). That is now the null control.

### 15.4 What remains outside Lean

Only that the class `LowCob` is **exactly** `GL(3,2)` — a closure computation over 8-element
tables, done in `W20` (168 elements, equal to the enumerated group). It is a finite check, not an
analytic gap.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 16. `LowCob` is exactly `GL(3,2)` — nothing about `g` is outside Lean (`W23`, 2026-08-03)

§15 left one item: that the inductive class is exactly `GL(3,2)`. Both directions are now proven.

### 16.1 Soundness

```lean
lowCob_isGL : LowCob t l →
  (∀ v < 8, t v < 8) ∧ t 0 = 0 ∧ (linear on the low block) ∧ (injective on the low block)
```

Assembled from `lowCob_lt`, `lowCob_t0`, `lowCob_lin` and the new `lowCob_inj8` (injectivity from
linearity plus trivial kernel). An injective linear endomorphism of `F2³` **is** an element of
`GL(3,2)`.

### 16.2 Completeness

For each of the 168 elements, an **explicit word** in the two generators, found by breadth-first
search — longest word **12**. Emitted as `LowCob.comp` terms:

```lean
lowCob_covers : ∀ i < 168, ∃ t l, LowCob t l ∧ ∀ v < 8, t v = linMap (glTable i)… v
lowCob_eq_GL  : glIndep a b c → ∃ t l, LowCob t l ∧ ∀ v < 8, t v = linMap a b c v
```

The dispatch is a 168-way match on the index; each case closes by `decide` on the eight low
values. `glIdx_lt`/`glIdx_eq` compute the index, `glList_indep` checks every listed triple is a
basis. **All plain `decide`, no `native_decide`** — so no extra trust axiom.

### 16.3 The chain, complete

> σ moves by a coboundary (§13) → the four σ's of `Q′` cancel it (§12) → `Q′` is invariant under
> the class (§14) → **the count is invariant** (§15) → **and the class is `GL(3,2)`** (here),
> whose transitivity on the seven nonzero low patterns is what merges the residues.

With `Qgen'_tau` (§11) for the even labels, **both halves of `g(W) = (W & (W−1)) ≫ 3` are
theorems, end to end.**

### 16.4 A second null-control catch

`W23`'s first run **failed**: it re-derived the generated group with a *stack* rather than a
*queue*, so it reported a longest word of 72 against the claimed 12. The reachable set was right
either way — only the word-length bound differed. The clause was re-deriving a quantity by a
different search order than the generator used, and said so. Fixed to breadth-first.

That is the second time in two rungs that a clause, not a reviewer, caught the defect.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 17. `tr(A³)`: why it is the finer invariant, and its constant stratum (`W24`, 2026-08-03)

### 17.1 The mechanism, proven

`A_sig`'s **entry** is not the resonance predicate but the **sign**
`−P1 = −σ(a,b)·σ(a⊕L,b⊕L)`. Under a class member the coboundary does **not** cancel there: only
`λ(a⊕b)` squares away, and what survives **factors**:

> `P1(p a, p b) = P1(a,b) · μ(a) · μ(b)`,  `μ(x) = λ(x)·λ(x⊕L)`

That is a **diagonal similarity** `A′ = D A D` with `D = diag μ`, `D² = I` — so `tr(A′ᵏ) = tr(Aᵏ)`
for **every** `k`. `P1_of_coboundary` and `P1_lowCob` are proven ∀n, kernel-clean.

`tau` admits no such factorisation, and it **measurably changes** `tr(A³)` (14 merges at
`n = 6,7,8`), while leaving `tr(A²)` alone.

> `tr(A²)` is invariant under **both** `GL(3,2)` and `τ`. `tr(A³)` is invariant under `GL(3,2)`
> **only**.

That asymmetry is the structural reason the *pair* separates strictly more than `tr(A²)` alone
(`W5`) — and therefore the reason (d) needs the second trace at all. Measured: 0 of 28 GL-orbits
have non-constant `tr(A³)`.

### 17.2 Closed form, on its stratum only

> `tr(A³) = (2/7)(2^m−2)(2^m−4)(2^m−15) = (2/7)·tr(A²)·(2^m−15)` on the `y = 0` class,
> exact at `n = 6..11`.

**Off that stratum the form fails** — exactly **7** labels satisfy it at every level, and those
are precisely the seven members of the `y = 0` GL-orbit. `W24` asserts that as a **declared
negative**. The deviation is constant on GL-orbits, which is where the next rung starts.

### 17.3 Not claimed

**No general closed form for `tr(A³)`.** This is exactly where `tr(A²)` stood at `W6` — a closed
form on the one stratum where the quantity is constant, and honest variation off it. Closing that
one took `W13`–`W17`.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 18. The `tr(A³)` deviation: one exact recursion, two impossibilities (`W25`, 2026-08-03)

§17 left the deviation off the `y=0` stratum open, noting only that it is GL-constant. Splitting
the label by its **top bit** — the split that cracked `tr(A²)` — resolves half of it and rules out
the two obvious ansätze for the other half.

### 18.1 The positive: the low branch is exact

> `t3(n,W) = 8·t3(n−1,W′) + 24·t2(n−1,W′) − 12(2^m − 4)`,  `m = n−1`, `W′ = W mod 2^{m−1}`

Verified in **exact integer arithmetic** over every low label at `n = 7..10` — 31/63/127/255
labels, **0 failures**. The constant `−12(2^m−4)` is closed.

### 18.2 First impossibility: the pair is not self-propagating

On the **high** branch the pair `(t2′, t3′)` does **not** determine `t3(n,W)`. Witness at `n = 7`:
the key `(t2′,t3′) = (168, −336)` carries **both** `−92112` and `18480`.

> So no recursion for `tr(A³)` on the pair `(tr A², tr A³)` alone can exist in general — a third
> level-quantity is required.

That matters for the lane's strategy: the pair *separates* the classes (`W3`/`W4`), but it is not
**self-propagating**.

### 18.3 …but the collisions are entirely seam-borne

Restricted to **odd** labels the pair **does** determine `t3` on the high branch — **0 collisions**
at all four levels. The even labels are what break it, which is the same place `τ` and the whole
`tr(A²)` story needed separate treatment.

### 18.4 Second impossibility: not even affine

Even on the odd high branch the dependence is **not affine** in `(t2′,t3′)`: an affine fit through
three points misses the rest by `1e4`–`1e7`, against **exact zero** on the low branch. So it is a
genuine function of the pair on the Fano family, but not a linear one.

### 18.5 Status

**`tr(A³)` is not closed.** What this rung buys is one exact half of the recursion and two
impossibility results that rule out the two obvious ansätze — a low-branch closed constant, a
proof that the pair cannot propagate itself in general, and a proof that the surviving case is
non-linear.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 19. The third level-quantity is `lsb(W)` (`W26`, 2026-08-03)

`W25` proved the pair `(t2,t3)` is not self-propagating on the high branch. The missing datum is

> **`lsb(W)` — the 2-adic valuation of the label.**

### 19.1 It is the one the structure predicted

`τ` moves the **lowest set bit** to position 0, so `lsb` is precisely the datum `τ` destroys. And
`W24` proved:

* `tr(A²)` is invariant under **both** `GL(3,2)` and `τ` — so it depends only on
  `g(W) = (W & (W−1)) ≫ 3`, which **discards** the lowest set bit;
* `tr(A³)` is invariant under `GL(3,2)` **only**.

So `tr(A³)` must see exactly what `g` threw away. It does: adjoining `lsb(W′)` kills **every**
high-branch collision at `n = 7,8,9,10`, where the pair alone collides at 1, 2, 4 and 8 keys.

### 19.2 It propagates for free

`lsb` is **label data**, not a graph invariant. On the high branch `W = W′ + e` with `W′ ≠ 0`, so
`lsb(W) = lsb(W′)`. Hence the **triple `(t2, t3, lsb)` is self-propagating** where the pair is not.

### 19.3 The sharp negative: no spectral invariant would have done

Adding `tr(A⁴)` or `tr(A⁵)` leaves the collision count **exactly unchanged** (1, 2, 4 at
`n = 7,8,9` — identical to the pair). The colliding labels agree across the whole trace family, so
**no level-(n−1) spectral invariant can supply the missing datum**. It had to be
label-arithmetic, and it is.

### 19.4 Still not a formula

At **fixed** `lsb′` the high branch is determined but **not affine** in `(t2′,t3′)` — an affine fit
misses by `1e5`–`1e6`, against exact zero on the low branch.

**`tr(A³)` is not closed.** What is now known is exactly *which three quantities* a closed form may
use, and that no fourth trace would help.

**(III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 20. No closed form from the triple yet — and why (`W27`, 2026-08-03)

I went looking for the closed form using `(t2, t3, lsb)` and **did not find it**. Three things are
recorded so the next rung does not repeat the search.

### 20.1 The high branch is not a low-degree polynomial in the pair

Even **stratified by `lsb′`**, adding the quadratic terms `t2′²` and `t2′t3′` improves the relative
residual from ≈ 0.46 to ≈ 0.06 but does **not** close it — against **exact zero** on the low
branch.

### 20.2 `W26`'s "determines" is real but modest — measured

The triple is **not** injective on the high labels: ≈ 2 labels per key, at most 4, carrying
127 labels → 63 keys → 23 distinct `t3` values at `n = 9`. So the agreement is not an artefact of
a fine partition.

**But determining a value on a finite set is not evidence that a formula exists**, and §19 should
not be read as if it were. That caveat belongs with the claim.

### 20.3 The structural reason — the useful part

> A triangle at level `n` whose vertices **straddle** the level split does not reduce to a
> level-(n−1) **triangle**. It reduces to a **path** — and path counts are not traces.

That is why no additional trace helps (§19.3) and why no polynomial in the traces closes it
(§20.1).

> ⚠ **§21 RETRACTS the recommendation I drew from this.** I concluded "carry a path-count" and
> then counted them: they give nothing, because the colliding fibers are **cospectral and agree on
> the non-spectral invariants too**. The straddling observation is still true; the inference from
> it was not.

**`tr(A³)` is not closed. (III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 21. I counted the paths. They give nothing — and §20.3 is retracted (`W28`, 2026-08-03)

### 21.1 The count

I took the path-counts that are invariant under the class action `A′ = D A D` (`D² = I`, so `|A|`
and hence the **degree sequence** is untouched) and are **not** traces:

* `Σ_a deg_a²` — 2-paths through a vertex
* `Σ_a deg_a³`
* `Σ_{a,b} A_ab·(A²)_ab` — the Hadamard edge/2-path contraction

Every one leaves the high-branch collision count **exactly unchanged**: 1, 2, 4 at `n = 7,8,9` —
identical to the pair alone. Adding **all of them together with the full spectrum** still changes
nothing.

### 21.2 Why — and the retraction

The colliding level-(n−1) labels have **identical full spectra**: one distinct spectrum among the
eight labels `{17..24}` at every level tested.

> So **no invariant of the level-(n−1) fiber, spectral or not, can supply the missing datum.**

§20.3 concluded that straddling triangles become paths and recommended the next rung carry a
path-count. **That recommendation is wrong and is withdrawn.** The straddling observation is still
true; the inference — that a fiber invariant could therefore express the correction — does not
follow, and the measurement refutes it.

### 21.3 The correct statement

> The level-(n−1) fiber's **isomorphism class does not determine** `tr(A³)` at level `n`. The
> missing datum is **label arithmetic, not graph structure**.

Which is exactly what §19 found the hard way, when every spectral candidate failed and `lsb(W)`
worked. So this is **not** a graph recursion awaiting a richer invariant — it is a **label
recursion**.

**`tr(A³)` is not closed. (III) is untouched. (d) is not closed, and V1 is not proven.**

---

## 22. The counting recursion enters Lean (`W29`, Tier 29, 2026-08-03)

The `W15` ledger — the step that turns `Ncnt` at level `m+2` into `Ncnt` at level `m+1` — has been
carried **on paper and pinned by clause** since it was found. Every commit had to say so. Tier 29
starts formalising it.

### 22.1 The toolkit

All by induction on `n`, all kernel-clean: `sumLt_zero`, `sumLt_const`, `sumLt_pair`,
`sumLt_split_if`, `sumLt_single`, `sumLt_single'`.

> **Split by predicate, extract singletons, evaluate constants** — no `Finset`, exactly as Tier 27
> avoided cardinality for the counting step.

### 22.2 Three of the four LOW quadrants

| theorem | content |
|---|---|
| `Ncnt_quad` | the level-`m+2` box splits into its four quadrants at the seam `2^{m+1}` |
| `Ncnt_ll_low` | **`ll` IS the level-`m+1` count** — `Q'red_low_ll` is unconditional, no slices |
| `Ncnt_ul_low` | `ul` reduces to the **unprimed** count `Mcnt` |
| `Ncnt_lu_low` | `lu` reduces to the transposed unprimed count, `a ≠ 0`, `a ≠ W` |
| `Ncnt_uu_low` | **`uu`** — five side conditions, and they collapse (below) |

`ul` and `lu` land on `Qgen`, not `Qgen'`, because those two low rows do. `lu`'s single
row-failure `a = W` contributes **nothing**: `Qgen'_label_left` makes the value `+1` there, so the
indicator is `0`. `W29`'s null control confirms that dropping the `a ≠ W` guard genuinely changes
the count.

### 22.3 `uu` closes the set

Its five side conditions **collapse**. On **every** failure slice `Qgen = −1`:

* `u = 0` and `v = 0` are the gap roots `a = H` and `b = H`;
* `u = W` and `v = W` are `a ⊕ W = H` and `b ⊕ W = H`.

So `Qgen'_off_lines` converts all four at once. The fifth, `v = u ⊕ W`, is exactly
`Qgen'_coset_partner` — value `+1`, contributing **nothing**. **All four LOW quadrants are now
Lean theorems.**

(`by_contra` bit here, as it always does: it does not exist Mathlib-free and silently becomes
`sorryAx`. Two uses were caught by the `unknown tactic` error and replaced with `by_cases`.)

### 22.4 What is not done

The **bridge from the unprimed counts back to `Ncnt`** — the six-line slice arithmetic, over
**six overlapping lines**, which is the part that needs care.

Until it lands, **the LOW recursion is not yet a Lean theorem** and the closed form's derivation
still rests on the contract clause. The caveat stays in place.

**(III) is untouched. `tr(A³)` is not closed. (d) is not closed, and V1 is not proven.**

---

## 23. The bridge's core: `Ncnt = OffCnt + (2^M − 2)` (`W30`, 2026-08-03)

§22.4 named the remaining obstacle as "the six-line slice arithmetic over **six overlapping**
lines". The overlaps turn out never to arise.

### 23.1 The factoring

All four quadrants differ from `Ncnt` only **on the six lines**, so they all factor through one
quantity — `OffCnt`, the count **off** the lines. `Ncnt_eq_OffCnt` is the first of those
factorings:

> `Ncnt W M + 2 = OffCnt W M + 2^M`

### 23.2 No inclusion–exclusion is needed

`nInd_split` is a **pointwise** identity —

> `nInd = [the b = W column] + [off-lines]`

— and summing a pointwise identity keeps the pieces disjoint for free. The overlapping-lines
problem was an artifact of thinking in sets rather than in summands.

The three lines `Ncnt` sees and `OffCnt` does not:

| line | value | contributes |
|---|---|---|
| `a = W` row | `+1` (`Qgen'_label_left`) | **nothing** |
| coset diagonal `b = a ⊕ W` | `+1` (`Qgen'_coset_partner`) | **nothing** |
| `b = W` column | `−1` (`Qgen'_label_right`) | exactly `2^M − 2` |

### 23.3 A caught vacuity — the third in this lane

`W30`'s first null control dropped the `b ≠ a⊕W` guard expecting the bridge to break. It **did
not**: on the coset diagonal `Q′ = +1`, so those pairs contribute `0` with or without the guard.
Same for the `a ≠ W` row. Both guards are load-bearing for the **proof's case analysis** and not
for the **value**; only `b ≠ W` carries the count.

The clause now asserts both: dropping `b ≠ W` breaks it, dropping `b ≠ a⊕W` does not.

> A guard needed by a proof is not automatically load-bearing for the quantity the proof computes.
> That is now three separate rungs where a null control caught exactly this.

### 23.4 `uu` is factored too

`uuInd_split` factors the `uu` summand as `OffCnt`'s — **with the arguments swapped**, which is
why `sumLt_swap` was needed — plus four boundary lines contributing exactly `4(2^M − 2)`.

### 23.5 `ul` and `lu` factor too — all four are proven

Both land on the **unprimed** `Qgen`, so besides the line values they need `Qgen_eq_Qgen'` off the
lines. Every line value was already in the tree (`Qgen_zero_left`, `Qgen_diag_neg`,
`Qgen_coset_left`/`_right`, `Qgen_degen`).

* `qInd_split` — `ul`, **five** boundary lines, contributing `5·2^M − 8`
* `luInd_split` — `lu`, **four** lines, contributing `4·2^M − 8`. Only four: `Qgen`'s
  `b ⊕ W = 0` degeneracy is `a = W`, which that quadrant's own guard already excludes.

### 23.6 What is not done

**Counting** those boundary sets in Lean, and the arithmetic assembly. All four *pointwise*
factorings are proven; the constants below are verified by `W30` but not yet derived in Lean:

| quadrant | factoring | theorem |
|---|---|---|
| `ll` | `OffCnt + (e−2)` | `nInd_split` |
| `ul` | `OffCnt + 5e − 8` | `qInd_split` |
| `lu` | `OffCnt + 4e − 8` | `luInd_split` |
| `uu` | `OffCnt + 4e − 8` | `uuInd_split` |

summing to `4·OffCnt + 14e − 26 = 4·Ncnt + 10e − 18` ✓ — the LOW recursion.

Until the counting lands, **the LOW recursion is not yet a Lean theorem**.

**(III) is untouched. `tr(A³)` is not closed. (d) is not closed, and V1 is not proven.**

---

## 24. The LOW recursion is a Lean theorem (`W31`, 2026-08-03)

> `Ncnt_low : Ncnt W (m+2) + 18 = 4 · Ncnt W (m+1) + 10 · 2^{m+1}`

The `W15` ledger's LOW half is **no longer paper — it is derived**. Kernel-clean, no `sorryAx`,
no `native_decide`.

### 24.1 The route

1. **Split** — the box splits into four quadrants at the seam (`Ncnt_quad`).
2. **Reduce** — each quadrant drops a level through its reduction row (`Ncnt_ll_low`,
   `Ncnt_ul_low`, `Ncnt_lu_low`, `Ncnt_uu_low`).
3. **Factor** — each reduces *pointwise* to a boundary term plus the shared off-lines core
   `OffCnt` (`nInd_split`, `qInd_split`, `luInd_split`, `uuInd_split`).
4. **Count** — the boundary terms, via `sumLt_cons`, `sumLt_two`/`_three`/`_four`,
   `sumLt_scale`, `count_off1`/`_off2`:

   | quadrant | boundary |
   |---|---|
   | `ll` | `2^M − 2` |
   | `ul` | `5·2^M − 8` |
   | `lu` | `4·2^M − 8` |
   | `uu` | `4·2^M − 8` |

5. **Tie back** — `Ncnt_eq_OffCnt` returns the core to `Ncnt`.

### 24.2 The obstacle that dissolved

The "six **overlapping** lines" that looked like the hard part never arose. **Summing a pointwise
identity keeps the pieces disjoint for free**, so no inclusion–exclusion was needed anywhere. The
difficulty was in the set-theoretic framing, not the mathematics.

None of it needed `Finset` — the same route Tier 27 took for the counting step.

### 24.3 What is still paper

The **HIGH branch** — the reflection `4P′ − 4N′ + 6e − 10`. Until it lands, the closed form for
`tr(A²)` is still not fully derived in Lean.

**(III) is untouched. `tr(A³)` is not closed. (d) is not closed, and V1 is not proven.**


## §25 — The HIGH recursion is a Lean theorem (both halves of W15 now derived)

`Ncnt_hi` (Tier 31, `formal/lean4/SounioZDFiberAntisym.lean`), for `W < e`, `W ≠ 0`, `m ≥ 1`,
`e = 2^(m+1)`:

```
Ncnt (W + e) (m+2) + 6e + 2 + 4 * Ncnt W (m+1) = 4 * e * e
```

which is the paper's `N = 4P' − 4N' + 6e − 10` with `P' = (e−1)(e−2)`. Stated additively so that
no `Nat` subtraction enters and `omega` stays linear with `e*e` as a single atom. With §24's
`Ncnt_low`, **both halves of the W15 ledger are now theorems ∀n**, kernel-clean (`propext`,
`Classical.choice`, `Quot.sound` only — no `sorryAx`, no `native_decide`).

### The content is the asymmetry, not the total

The four quadrants do **not** factor through a common core. Writing `P = OffCntP W (m+1)` for the
positive core (the count of `Q' = +1` off all six lines — new here, because each `hi` row carries
a global minus sign, so counting `−1` at level `m+2` means counting `+1` at level `m+1`):

| quadrant | total | slices |
|---|---|---|
| `ll` | `P + 3(e−2)` | `a=W` row, `b=W` column, **coset diagonal** |
| `ul` | `P + 3(e−2)` | `u=0` row, `b=W` column, diagonal `b=u` |
| `lu` | `P + 4e − 7` | `v=0`, **`v=W` column (e−1 points)**, `a=W`, diagonal `a=v` |
| `uu` | `P + 4e − 7` | **`u=0` row (e−1 points)**, `v=0`, `v=W`, coset diagonal |

Two facts do the work, and both were errors in the first attempt at this branch:

1. **`ll`'s coset diagonal counts; `ul`'s does not.** `Q'red_hi_ll` does not exclude `b = a^W`, and
   the reflected remainder counts `Q' = +1`, which is exactly the coset-partner value. `Q'red_hi_ul`
   lands on the **unprimed** `Qgen`, where the same diagonal is `−1` (`Qgen_degen`). Same total,
   different anatomy.
2. **`lu` and `uu` each carry one extra point.** `Qgen'_label_right` holds with *no hypothesis on
   its other argument* (unlike `label_left`), so the `v = W` column is `−1` on all `e−1` of its
   points while every other slice loses two to the lines.

### Method note

The 15 pointwise boundary values were each verified numerically (m = 1..4, all W, ~1200 points
each) **before** any Lean was written. Two of the fifteen needed no new lemma: `a = Wh` is the
whole `label_left` row and `b = Wh` the whole `label_right` column — but only once the split order
puts `a = Wh` **before** `v = 0`, which is what removes the two exceptional cells that an earlier,
non-partitioning probe had missed. Relaxing the four `hi` rows' hypotheses was tested and
**refuted** (only `hi_ll`'s `a ≠ W` and `hi_ul`'s `u ≠ W` are individually removable), so the
boundary lemmas are genuinely needed.

### `m = 0` is CLOSED — `Ncnt_hi` is unconditional

An earlier version of `Ncnt_hi` carried `hm : 1 ≤ m`. That hypothesis was a proof artifact, not
mathematics: it came from `OffCnt_add_OffCntP` needing `4 ≤ 2^M` for its `∃ k, 2^M = k + 4` step.
**It has been removed.**

The box has only two possible sizes. A nonempty label forces `2 ≤ 2^M` (from `W < 2^M`, `W ≠ 0`),
and a power of two that is not `2` is at least `4` — and `M = 0` is impossible, since `W < 2^0 = 1`
contradicts `W ≠ 0`. So:

- **`2^M = 2`** — the `m = 0` bottom. `count_off2` makes the off-lines count `0`, the product
  vanishes (`Nat.mul_zero`), and the identity reads `12 = 12`. It holds by `Nat` truncation rather
  than by the `(e−2)(e−4)` expansion, which is exactly why it needs its own line.
- **`4 ≤ 2^M`** — the original argument, unchanged.

`OffCntP_eq` and `Ncnt_hi` drop the hypothesis as a consequence. `Ncnt_hi_bottom` records the
closed instance `Ncnt 3 2 + 12 + 2 + 4·Ncnt 1 1 = 16`, elaborated as `Ncnt_hi 0 1`, so that the
removal is **checked by the kernel** rather than asserted in a signature.

This was load-bearing for the unrolling, not cosmetic: descending an odd non-power-of-two label
uses LOW while `W < 2^(m+1)` and then lands on HIGH at `m = 0`. `W = 3` bottoms out on exactly
the case the old hypothesis excluded.

**Unrolled in §26** (`Ncnt_eq_Nclosed`, same day). **Still open:** the explicit digit-sum form
`E` and its dependence on `g(W)` alone; (III); `tr(A³)`'s general closed form; (d); V1.


## §26 — The unrolling: `Ncnt` is a closed evaluator

`Ncnt_eq_Nclosed` (Tier 32, `formal/lean4/SounioZDFiberAntisym.lean`):

```lean
theorem Ncnt_eq_Nclosed : ∀ (m W : Nat), W < 2^m → W ≠ 0 → (Ncnt W m : Int) = Nclosed m W
```

where `Nclosed` recurses on the **level** and mentions **no `Qgen'`, no `cdSigma`, no sum**:

```lean
def Nclosed : Nat → Nat → Int
  | 0, _ => 0
  | 1, _ => 0
  | (n+2), W =>
      if W < 2^(n+1) then 4 * Nclosed (n+1) W + 10 * ((2^(n+1) : Nat) : Int) - 18
      else if W = 2^(n+1) then ((2^(n+2) * 2^(n+2) : Nat) : Int) + 6 - 5 * ((2^(n+2) : Nat) : Int)
      else 4 * ((2^(n+1) * 2^(n+1) : Nat) : Int) - 6 * ((2^(n+1) : Nat) : Int) - 2
             - 4 * Nclosed (n+1) (W - 2^(n+1))
```

This discharges the caveat W31, W32 and §25 all carried: *"what is still NOT derived in Lean is
the UNROLLING of the two recursions into the closed form."* Kernel-clean; the three equation
lemmas for `Nclosed` depend on **no axioms at all** (`rfl`), so the recursor is definitional.

### The floor, and why it is four lines

The descent needed a floor, and `Qgen'_pow2_eq` is pointwise — its docstring asserted the count
`(2^m−2)(2^m−3)` without proving it. `Ncnt_pow2` proves it, via:

> **`OffCntP_pow2 : OffCntP (2^k) m = 0`** — `OffCntP` already excludes `a = W` and `b = a^W`, and
> by `Qgen'_pow2_eq` those two lines are *exactly* where `Q' = +1` at a power-of-two label. So
> nothing is left to count.

Feeding that to `OffCnt_add_OffCntP` and `Ncnt_eq_OffCnt` (both proven 2026-08-03) leaves a linear
`omega`. **The power-of-two labels are the saturated case**, and every other label's count is that
value minus a deficit — which is the structural reason the closed form
`Ncnt W m = (2^m−2)(2^m−3) − E(m, 8·g(W)+1)` has the base value as its leading term.

I had planned this as a ~200-line two-disjoint-lines counting argument. It needed none of
`sumLt_scale`, `count_off1`/`count_off2`, or any disjointness lemma.

### No power-of-two test

`Nclosed` has no `isPow2` branch. A label `2^k` with `k < n+1` is already below the seam, so the
LOW branch handles it — and the two agree, because

```
(2e − 2)(2e − 3) = 4(e − 2)(e − 3) + 10e − 18
```

identically. Only `W = 2^(n+1)` needs its own branch, and that is decidable `Nat` equality. This
deletes the `W &&& (W−1) = 0 → ∃k, W = 2^k` characterisation, 60–150 lines of bit induction in a
Mathlib-free file. Verified before writing any Lean: the descent *without* any `isPow2` test
reproduces `Ncnt` with **0 failures over 1013 pairs, m = 1..9**.

### The failure that actually cost time

Not `Int`-vs-`Nat`, which was anticipated and handled by forming every product and power on the
`Nat` side and casting (so `omega` sees the same atoms it already sees inside `Ncnt_hi`). The real
one: **the induction leaves the level as `k+1+1` while every recursion theorem states it as
`k+2`, and `omega` treats those as different atoms although they are definitionally equal.** It
does not error — it silently has no equation relating them, and reports an unprovable goal with a
missing atom. Fixed by a `show` that normalises the index.

**Closed in §27.** The digit sum is now a Lean theorem — and it did **not** need W17's residual.
(III), (d), V1 and `tr(A³)`'s general form are all unchanged.


## §27 — The closed form is derived

`Ncnt_closed` (Tier 33, `formal/lean4/SounioZDFiberAntisym.lean`):

```lean
theorem Ncnt_closed (m W : Nat) (hW : W < 2^m) (hW0 : W ≠ 0) :
    ((Ncnt W m : Nat) : Int) + Ddig m W + 5 * ((2^m : Nat) : Int)
      = ((2^m * 2^m : Nat) : Int) + 6
```

i.e. **`Ncnt W m = (2^m−2)(2^m−3) − E(m,W)`** with `E = Ddig` a **finite, non-recursive** signed
base-4 digit sum over the set bits of `W`. Stated additively so no `Nat` subtraction enters.
Kernel-clean.

W17 has carried this since it was written as *"the DERIVATION remains base (Lean ∀n) + recursion
(paper); what is verified directly here is the FORMULA."* That caveat is discharged.

### The route avoids W17's residual entirely — and that was the surprise

§26 predicted the hard obstacle would be the dependence on `g(W)` alone: W17's residual, which the
Lean file itself calls `Finset` territory (`:4626`). **It is not needed.** That obstacle only
exists if you target the contract's form, which is stated on the *normalised* label `8·g(W)+1`.

Writing the **deficit** `base − Ncnt` on `W`'s **own** bits gives a digit sum that matches
directly:

```
Ddig m W = Σ_{i : bit_{i−1}(W)=1 and W mod 2^(i−1) ≠ 0}  (2^i−4)(2^i−8) · 4^(m−i) · (−1)^popcount(W ≫ i)
```

The second guard conjunct is the whole point: the descent **stops at the lowest set bit**, so the
term there is omitted. **W16's declared negative — "the raw label FAILS on every seam" — is
exactly that missing exclusion, not a missing normalisation.** Both forms agree label-for-label
(`Ddig m W = E(m, 8·g(W)+1)`, pinned in W34 at m = 1..8, and measured at m = 1..12 before any Lean
was written).

### Proof structure

- `dterm_step` — every digit below the seam scales by `4` and picks up the seam bit's sign. The
  `4` is the LOW multiplier; the sign is `psg (W >>> (n+1))`, `+1` when the seam bit is clear and
  `−1` when set. Sign factorisation is `psg_split` via `shift_mod_pow`/`shift_div_pow`; the guard
  is unchanged below the seam by `mod_pow_mod`.
- `Ddig_step` — hence the digit sum obeys the same descent as `Nclosed`.
- `Nclosed_add_Ddig` — induction on the level, three branches, with every nonlinear fact supplied
  explicitly (`two_mul_mul`, `sub_prod_pow`, `dcoef_top`), since `omega` derives none of them.

`dcoef` uses **truncated** `Nat` subtraction and that is exactly right: at `i = 2, 3` the true
coefficient is genuinely `0` and truncation gives `0`; at `i = 0, 1` truncation is wrong but the
guard always excludes those. No index is both included and mis-valued.

### Two traps, both silent

`omega` treats `k+1+1` and `k+2` as **different atoms** although they are definitionally equal, and
likewise `Nclosed (k+1) (W − 2^(k+1))` and the IH's `Nclosed (k+1) W'`. Neither errors — omega
simply loses the equation and reports an unprovable goal with a missing atom. Fixed by `show` and
by `subst` + `Nat.add_sub_cancel`.

**Still open:** (III); `tr(A³)` has no general closed form; (d) is not closed. `tr(A²)` is
parity-blind (W11), so deriving its closed form does **not** narrow (d).


## §28 — The bridge is proven: W17's residual is a theorem

`Ddig_gnorm (m W) (hW : W ≠ 0) : Ddig m W = Ddig m (gnorm W)`, with
`gnorm W = 8 * ((W &&& (W−1)) >>> 3) + 1` — exactly the contract's `8·g(W)+1`. Plus
`Ncnt_closed_gnorm`, the closed form on the **normalised** label, i.e. the contract's own form
of `E`.

§27 proved the closed form on `W`'s own bits and left `Ddig m W = E(m, 8·g(W)+1)` **measured**
(W34 pins it at m = 1..8). It is now **proven ∀n**. This is the residual W17 has carried unproven
— "bits 1 and 2 of an already-odd label do not matter" — which the Lean file calls `Finset`
territory at `:4626`.

**Why it was reachable, and it was not by attacking the invariance head-on.** The digit sum
**excludes the lowest set bit**, so every included digit sits strictly above it — therefore
neither its guard nor its sign can see the bit that normalisation moves. The invariance is
**termwise**: no bijection, no cardinality, no `Finset`. The same exclusion that made §27 provable
on `W`'s own bits is what makes the bridge provable.

The crux is one bit fact — `testBit_pred`: subtracting one flips bit `j` exactly when nothing lies
below it. Its consequence `and_pred_testBit` says `(W &&& (W−1)).testBit j` **is** `dterm`'s
guard. The rest is bookkeeping: `guard_iff`, `shift_and_pred`, `gnorm_shift`/`_testBit`/`_odd`.

Note `(2t+1) % 2^(j+1) ≠ 0` is **not** omega-provable — the witness needs `2^j · q` and omega
rejects nonlinear terms. Parity extracted via `mod_pow_mod 1 (j+1)`.

**Still open:** (III); `tr(A³)`'s general closed form; (d); V1. `tr(A²)` is parity-blind (W11), so
none of this narrows (d).


## §29 — Tier 35 review: both hypotheses are sharp

`math-review` (policy M3) on Tier 35 returned PASS on all six declarations, and volunteered two
sharpness results not in the statements. Both independently re-verified against the Python oracle
before being accepted:

- **`2 ≤ i` in `dcoef_spec` is sharp.** At `i = 1` the `Nat` coefficient truncates to `0` while the
  true value is `(2−4)(2−8) = 12`, and the subtraction-free identity would read
  `24·4^(m−1) = 36·4^(m−1)`. Measured: 24 vs 36 at `m=1`, 96 vs 144 at `m=2`, 384 vs 576 at `m=3`.
- **Oddness in `Ddig_eq_Edig` is sharp.** For **even** `V` with `lsb(V) = i−1 ≥ 1`, `testBit (i−1)`
  holds but `V % 2^(i−1) = 0`, so `Edig` includes a term `dterm` drops. Measured: 31 even labels
  differ at `m = 8`; at `V = 16`, `Ddig = 0` while `Edig = 43008`.

So neither hypothesis is convenience — each is necessary, and the second is the precise reason the
contract states `E` on the **normalised** (hence odd) label rather than on `W`.


## §30 — §7's indexing reconciled with `Ncnt`, and (II) is now a theorem

§7 is written in **fibre coordinates**; `Ncnt` is written in **raw labels**. The dictionary:

| §7 | `Ncnt` side |
|---|---|
| level `n` | `Ncnt` runs at `m = n − 1`; the level-`n` label is `Llo \| 2^(n−1)` and `Llo` is the raw label |
| `tr(A²)` | `tr(A²) = Ncnt W m − (2^m − 2)` (the W16 bridge) |
| fibre coordinate `y` | `g(W) = (W &&& (W−1)) >>> 3` — for odd `W`, `g(8y+1) = y` |
| Fano orbit `y` | the representative label `8y + 1` |
| the fibre of `y` | `{8y+1 … 8y+7}` ∪ the even labels clearing to `8y` |

**`τ` lands on the Fano representative, not on `W &&& (W−1)`.** Measured, decisively:

- reading A, `Ncnt W = Ncnt (W &&& (W−1))` — **1972/4012 FAIL**;
- reading B, `Ncnt W = Ncnt (8·g(W)+1)` — **0/4072**, and this *is* `Ddig_gnorm`.

So **(II) is not a measurement any more: it is `Ddig_gnorm`, proven ∀n today.** The earlier
"0 exceptions" in §7's table can be replaced by a citation.

A structural check rules reading A out independently of the counts: `tr(A²)` is **injective in
`g`** — 8→8 fibres/values at `m=6`, 64→64 at `m=9`, 256→256 at `m=11` — so no reading of (II) may
collapse two distinct `g`-values, which reading A would require.

### What this does to (I) and (III)

- **(I)** ("`tr(A²)` injective on the Fano orbits") is *weaker* than what is measured: `tr(A²)` is
  injective in `g` on **all** labels, Fano and seam alike. It is now a statement purely about the
  closed form `Ddig` — no `Qgen'` — hence attackable by the same machinery. Still measured.
- **(III)** is sharpened, not helped. Inside one fibre every label has the **same** `tr(A²)` — at
  `m=9`, `g=5`, labels 41…47 all give `142344`, and that is now a *theorem*, not an observation.
  So `tr(A²)` provably contributes nothing inside a fibre, and (III)'s separation must come
  entirely from `tr(A³)`. This is the trace-side restatement of the parity-blindness (W11).

**Step 0 of any attack on (III) is therefore closed.** What remains open is unchanged: (III)
itself, `tr(A³)`'s general closed form, (d), V1.


## §31 — (I) reduced: `F` splits into three recursive pieces (MEASURED, not proven)

Everything in this section is **measurement**, recorded so it is not lost. Nothing here is a Lean
theorem. The Lean targets are written out as signatures so a later session starts from types.

With `F(m,y) = Ddig m (8y+1)` — i.e. §30's fibre-coordinate form of the closed form — write
`s_j = (−1)^popcount(y ≫ (j+1))` and

```
S(y)    = Σ_{j ∈ bits(y)} s_j
P(m,y)  = Σ_{j ∈ bits(y)} s_j · 2^(2m−j−4)
Q(m,y)  = Σ_{j ∈ bits(y)} s_j · 2^(2m−2j−8)
```

**The decomposition** (0 mismatches, m = 5..10), from `dcoef(m,i) = 4^m − 12·2^(2m−i) + 32·2^(2m−2i)`:

```
F(m,y) = 4^m · S(y) − 12 · P(m,y) + 32 · Q(m,y)
```

**The six identities** (0 mismatches; even branch m = 8..11, odd branch m = 8..11). `psg` is the
file's existing `(−1)^popcount`:

| | even | odd |
|---|---|---|
| `S` | `S(2t) = S(t)` | `S(2t+1) = psg t + S(t)` |
| `P` | `P(m,2t) = 2·P(m−1,t)` | `P(m,2t+1) = psg t · 2^(2m−4) + 2·P(m−1,t)` |
| `Q` | `Q(m,2t) = Q(m−1,t)` | `Q(m,2t+1) = psg t · 2^(2m−8) + Q(m−1,t)` |

They were **derived from the bit shift before being measured**: for `y = 2t+1` bit 0 contributes
`psg t` at top weight, and the remaining bits are those of `t` shifted with the *same* signs, which
is why the residual is exactly the level-below value.

**Why this reduces (I).** `S` takes only two values (`S(y) ∈ {0,1}`, checked m = 4..14) so it is
not injective, but **`P` and `Q` are each injective** (32/32 at m=8, 128/128 at m=10, 512/512 at
m=12). So (I) does not need `F`: it needs `P`.

And `P`'s weights are **pure powers of two**, so `2^k > Σ_{j<k} 2^j` — the dominance that the raw
`dcoef` weights do *not* admit (their ratio `c_{j+1}/c_j → 1`). (I) therefore reduces to
**uniqueness of a signed binary representation**, provable by induction on `t/2` in the style of
`psg` (`:3513`, well-founded, `decreasing_by`), recovering the leading bit by dominance at the odd
step and the rest by the induction hypothesis. That is a genuinely different proof shape from the
magnitude comparison I had wrongly assumed was required.

### Known obstacle before any of this becomes Lean

`S`, `P`, `Q` are sums over the bits of `y`, and each identity **reindexes a bounded sum under a
bit shift**. This file has no `Finset`; the precedent is `sumLt_lowMap` (`:5240`), built by
induction on the level specifically to do one reindexing. Expect that cost per identity, not
lemma-sized effort.

**Status: all of §31 is MEASURED.** Open, unchanged: (I), (III), `tr(A³)`'s general closed form,
(d), V1.

## §32 — (I) is a theorem: the 2-adic valuation, not magnitude (`W35`, Tier 36, 2026-08-04)

**(I) is PROVEN ∀n.** `Ncnt_inj_g` / `Ncnt_inj_gnorm` in `SounioZDFiberAntisym.lean`: `Ncnt` is
injective in the fibre coordinate `g`. Kernel-clean `[propext, Classical.choice, Quot.sound]`.
§30 had reduced (I) to a statement about the closed form alone (no `Qgen'`); it is now closed.

**Stated on `Ncnt`, not on `tr(A²)`.** `tr(A²) = Ncnt W m − (2^m − 2)` is §7's dictionary (the W16
bridge), and `tr(A²)` is **not a Lean object** in that file. The offset depends only on `m`, so
the two injectivity statements are equivalent — but that equivalence is documentation, not a
theorem.

### §31's reduction does not close, and its target was false

Recorded because §31 asks the next session to distrust its own diagnosis, and it was right to:

- **"`P` injective ⟹ (I)" is a non-sequitur.** (I) is about `F`, and nothing in §31 recovers `P`
  from `F = 4^m·S − 12P + 32Q`.
- **"uniqueness of a signed binary representation" is FALSE.** Signed sums of distinct powers of
  two are not unique: `2^k − 2^(k−1) = 2^(k−1)`. What makes `P` injective is the **sign
  structure** (`s_j` is determined by the higher bits), which §31's framing discards.

The measurement (`P` injective, 512/512) is real; it was attached to the wrong object.

### What carries it

**Factor the coefficient instead of expanding it.** With `i = p+1` (so `2^i ≥ 16`, no
truncated-subtraction trap):

```
2^(p+1) − 4 = 4(2^(p−1) − 1),  2^(p+1) − 8 = 8(2^(p−2) − 1),  4^(m−p−1) = 2^(2m−2p−2)
⟹  dcoef m (p+1) = 2^(2m−2p+3) · (2^(p−1) − 1)(2^(p−2) − 1),  the cofactor ODD
```

So `v₂(dcoef m (p+1)) = 2m − 2p + 3`: **strictly decreasing in the bit position, gap exactly 2.**
That gap is the whole proof. (`dcoef_factor`, `dcoef_cofactor_odd`.)

Two theorems then prove each other in one induction:

| | statement |
|---|---|
| `Ddig_peel` | `Ddig m (2^p + W) = dcoef m (p+1) − Ddig m W` for `0 < W < 2^p`, `p+1 ≤ m` |
| `Ddig_val` | `W ≡ 1 (mod 8)` with top bit `p` ⟹ `Ddig m W = 2^(2m−2p+3) · odd` |

`Ddig_peel` is **termwise**: at the peeled bit the sign is `psg 0 = 1`, and every lower digit
keeps its guard (`2^p` is divisible by `2^i`) and flips its sign by `psg_top` (`:3525`, already
proven). `Ddig_val` follows because the peeled remainder's valuation is at least **two** higher,
so it cannot cancel the odd cofactor.

Injectivity then falls out: equal digit sums force equal valuations (`pow_mul_odd_inj`), hence
equal top bits; the peel descends one bit; `Ddig_ne_zero` handles the base. `Ddig_inj_gnorm`
lifts it to arbitrary labels via `Ddig_gnorm` (§28) and `gnorm W ≤ W`.

**§31's stated obstacle is dodged entirely** — no `Finset`, no sum reindexed under a bit shift,
no `m`-index shift; `m` stays fixed throughout. This is the same termwise move that made §28
reachable: the descent never looks below the lowest set bit, so nothing has to be reindexed.
Two whole sessions' worth of predicted cost evaporated because the object was factored rather
than expanded. That is now the third time in this lane that a "no route" diagnosis came from
looking at the object in the wrong form (§31 lists the other two).

### Measured before any Lean was written

`scripts/research/zd_v1_I_valuation_probe.py`, against the same definitions the Lean file uses:

- factorisation: `m = 6..14`, 63 checks, **0 mismatches**;
- peel: `m = 6..12`, 1009 checks, **0 mismatches**;
- valuation `v₂(F(m,y)) = 2m − 2k − 3` and injectivity: `m = 4..14`, 4094 labels,
  **0 mismatches, 0 collisions** — `m = 4, 5` included deliberately, where the only label
  `≡ 1 (mod 8)` below `2^m` is `W = 1` and injectivity is vacuous, to confirm there is no silent
  gap between the vacuous and the measured range.

The oracle is pinned to §30's independently recorded number — `Ddig(9,41) = 116736`, constant
across the fibre `41…47`.

**Still open, unchanged:** (III); `tr(A³)`'s general closed form; (d); V1. §30 already showed
`tr(A²)` contributes nothing inside a fibre, so (III) is untouched by this.

## §33 — (III) reduced to ONE arithmetic identity: the within-fibre deviation of `tr(A³)` (`W36`, 2026-08-04)

**Everything in this section is MEASURED.** Nothing here is a Lean theorem and nothing here is
derived. (III) is **not** proven; it is reduced.

### 33.1 The reformulation: inside a fibre, `lsb(W)` indexes the classes — with `{0,1,2}` collapsed

Tier 36 proved `tr(A²)` injective in `g(W) = (W ∧ (W−1)) ≫ 3`, so the `tr(A²)`-fibre of `y` is
*exactly* `{W : g(W) = y}`, and §30's dictionary makes its members explicit:

| member | labels | `lsb(W)` |
|---|---|---|
| the Fano orbit `y` | `W = 8y + r`, `r = 1…7` | `0, 1, 2` — one `GL(3,2)` orbit |
| the seams | `W = 8(y + 2^i)`, `i < lsb(y)` | `i + 3` |

⚠ **`lsb` indexes classes only after collapsing `{0,1,2}`.** The seven Fano labels realise three
different `lsb` values and are ONE class (`GL(3,2)`-constancy of `tr(A³)`, §17.1). So a fibre holds
`1 + lsb(y)` classes, **not** `3 + lsb(y)`. §19's third level-quantity is `lsb(W)` on the nose; the
class index is `lsb` post-collapse. The two are not the same object and this section had run them
together.

The class count below was computed with the collapsed reading (`1 + lsb(y)`), so it is unaffected;
`3·2^(n−5)` would fail immediately under the naive one.

⚠ **The parity variable is `popcount(g)`, not `popcount(y_seam)` and not `popcount(W)`.** The
earlier phrasing — "every seam has popcount `popcount(y)+1`, so the fibre carries one parity" —
is a non-sequitur: the Fano members `8y+r` carry popcounts `popcount(y) + {1,2,3}`, i.e. three
parities. The correct statement is trivial and in the other variable: `popcount(g)` is constant on
a fibre **by definition of the fibre**. The translation to (c)'s variable, verified for
`y' = 1…4095`:

> for a seam `W = 8y'`, `g = y' ∧ (y'−1)`, so `popcount(g) = popcount(y') − 1`, hence
> **`popcount(g)` ODD ⟺ the seam's own weight `popcount(y')` is EVEN** — which is exactly the
> class (c) quantifies (§1, §4: "the even-weight seams merge, exactly `2^(n−5)−1` of them").

With that pinned:

- `popcount(g)` **odd** (= even-weight seam) → every seam in the fibre merges with the Fano class —
  this is (c), PROVEN;
- `popcount(g)` **even** (= odd-weight seam) → (III) must separate them.

Consistency check, independent of any trace: counting `1 + lsb(y)` classes on even-`popcount(g)`
fibres and `1` on odd ones reproduces `3·2^(n−5)` exactly at `n = 6…12`.

Hence (III) is a statement about the **deviation**

```
D(W) = tr(A³)(W) − tr(A³)(8·g(W) + 1)
```

and nothing else. The lane spent `W24`–`W28` chasing a closed form for `tr(A³)` *absolutely* and
did not find one. (III) never needed it.

### 33.2 The measurement: the deviation does not depend on `g`

> **`D(W) = 0` when `popcount(g(W))` is odd, and `D(W) = δ(n, lsb W)` — a function of the LEVEL
> and the LOWEST SET BIT ALONE — when it is even.**

That the correction is **independent of `y`** is the surprising part and the load-bearing claim.
Checked at `n = 6…11`, **all 2010 labels, 0 mismatches**, together with `D = 0` on every Fano
label (`GL(3,2)`-constancy) and 0 collisions of `D` in `lsb` on every even-popcount fibre.

### 33.3 The closed form for `δ` — it is not a cubic, it is a subspace count

```
δ(n, j) = −(9/56) · u³ · (2^j − 1)(2^j − 2)(2^j − 4),        u = 2^(n−j)
        = −(9/56) · (2^n − u)(2^n − 2u)(2^n − 4u)
```

**Regrouped, the cubic disappears.** `(2^j−1)(2^j−2)(2^j−4) = |Inj(𝔽₂³, 𝔽₂^j)|` — the number of
ordered linearly independent triples in `𝔽₂^j` — which is `168 · [j choose 3]₂` with
`168 = |GL(3,2)|`. Since `(9/56)·168 = 27`:

```
δ(n, j) = − 27 · 8^(n−j) · [j choose 3]₂
```

Verified identical to the first form at `(n,j) = (6,3), (6,4), (7,5), (8,6), (9,7), (10,8), (11,9)`.

Three things this buys, none of which the cubic form showed:

- **The vanishing below `j = 3` stops being a coincidence.** It is not "a cubic that happens to
  have roots at `1, 2, 4`" — it is that `𝔽₂^j` has no 3-dimensional subspace when `j < 3`. **The
  Fano orbit merges because there is no Fano plane below the seam.** (c) and the `j ≥ 3` case
  become one statement evaluated in two regimes, which is strictly stronger than two coordinated
  lemmas.
- **Monotonicity becomes trivial, and the margin becomes explicit.**
  `|δ(n,j)| = (9/7)·2^(3n−3)·(1−2^(−j))(1−2^(1−j))(1−2^(2−j))` is strictly increasing in `j` ✓ but
  **asymptotically constant**: it converges from below to `(9/7)·2^(3n−3)`, with consecutive gaps
  `O(2^(3n−3−j))`. In exact arithmetic the separation is real; it is not numerically comfortable,
  and any float route would lose it.
- **A candidate mechanism for the open residual.** `[j choose 3]₂` counts subspaces of the `j` bit
  positions **below** the seam — precisely the coordinates `g` forgets. So "`D` does not depend on
  `g`" stops being a brute fact and becomes: the triangle deficit is a count over independent
  triples strictly below the seam, and translation by `8y` is an `𝔽₂`-affine map fixing those `j`
  coordinates. **The proof to build is a sign-preserving bijection between the deficit triangles of
  `8y + 2^j` and those of `2^j`, given by `a ↦ a ⊕ 8y`** — the only real burden being that the sign,
  restricted to the deficit set, depends on the low part alone. That is much smaller than deriving
  a cubic.

⚠ **The naive edge-level form of that bijection is REFUTED, measured.** `a ↦ a ⊕ 8y` does **not**
carry the edge deficit `A(n, 8(y+2^i)) − A(n, 8y+1)` onto `A(n, 2^(i+3)) − A(n, 1)`: 0 of 41
`(y,i)` pairs at `n = 7,8,9`, and the deficits are not even the same *size* — at `n = 7`,
`|D_W| ∈ {424, 1152, 2296}` against a fixed `|D_0| = 2160`. So whatever preserves `D` is **not** a
relabelling of the deficit edge set. A bijection at the level of deficit *triangles* is not
excluded by this test, but it cannot be induced by one on edges, which makes the `g`-independence
harder to explain, not easier. Recorded so the next rung does not start from the edge-level form.

**(III) follows from either form immediately**: `δ` is nonzero (Fano vs seam) and injective in `j`
(seam vs seam), which is exactly what (III) asserts.

Status of the form: **fitted**, on `n = 6…9`, `j = 3…7`, in the four-parameter family
`α·8^j + β·4^j + γ·2^j + ε`, then checked at `n = 10` (new `j = 8`) and `n = 11` (new `j = 9`) —
0 mismatches. `j` ranges `3 … n−2`, and `j = n−2` is the maximum possible (labels are
`< 2^(n−1)`), so the boundary is exercised at every level, not extrapolated.

`tr(A³)` was computed in float64 and **verified against exact `int64`** at `n = 11` on 13 labels
including every `j`: 0 disagreements. The law is an exact integer identity with a `/56`, so this
check is not cosmetic.

### 33.4 What this does and does not settle

- **(III) is reduced to one arithmetic identity**, the deviation law of §33.2. It is not proven.
- **`δ`'s closed form is a fitted cubic in `2^j` that factors nicely.** That it factors as
  `(x−1)(x−2)(x−4)` — the same shape as `dcoef`'s `(2^i−4)(2^i−8)` and §17.2's
  `(2^m−2)(2^m−4)(2^m−15)` — is suggestive, not evidence. A fitted cubic that factors is still a
  fitted cubic.
- **A derivation lead, recorded and not pursued:** `(2^j−2)(2^j−4)` is exactly `tr(A²)` of the
  `y = 0` class at level `j+1` (§3's `W6` form), and `(2^j−1)` is the number of nonzero elements of
  a block of size `2^j`. So `δ(n,j) = −(9/56)·2^(3(n−j))·(2^j−1)·tr(A²)(j+1, 0)`. The `7` in the
  denominator echoes §17.2's `2/7`. The competing reading — that `δ` is the level-`j` `y = 0`
  *`tr(A³)`* rescaled — is **refuted**: that form carries a `−15` and `δ` does not.
- **Why `W24`–`W28` stalled is now explained.** They asked for `tr(A³)` as a function of the label.
  The fibre-*relative* quantity is the simple one, and it is simple precisely because it forgets
  `g` — the same locality that made `Ddig_peel` termwise in Tier 36.

**Still open:** (III) itself (= the deviation law), `tr(A³)`'s absolute closed form, (d), V1.
(d) is now exactly **(I) ∧ the deviation law**, and (I) is a theorem.

Reproduce: `python3 scripts/research/zd_v1_III_deviation_probe.py 6 7 8 9 10 11`.

### 33.5 The law decomposes — and one third of it is a *derivation*, not a fit

The fitted cubic of §33.3 is not the irreducible content. Three pieces, with different status:

**(A) The `popcount(g)` odd half is already a theorem.** `D = 0` there is exactly (c): the seam has
even weight, `Φ` is an isomorphism of the *signed* graph onto its Fano partner, so the spectra —
hence `tr(A³)` — agree. Nothing to prove.

**(B) The `8^(n−j)` scaling follows from facts already on the books.** §18.1's low-branch
recursion, applied to `W` and to its own reference `8·g(W)+1` (also low) and subtracted:

```
D(n, W) = 8·D(n−1, W) + 24·[t2(n−1,W) − t2(n−1, 8g+1)]  − (the constant, which cancels)
        = 8·D(n−1, W)
```

because the bracket is **zero by the proven fibre-constancy of `tr(A²)`** (§30 + Tier 36). So the
whole `n`-dependence is one line, given §18.1 (itself MEASURED, `n = 7..10`). The descent bottoms
out when `W` stops being low — at level `t+2`, where `t` is `W`'s **top** bit.

**(C) For the `y = 0` fibre the base case is DERIVED.** There the seams are `W = 2^j`, top bit =
lowest bit, so the base is the single family `W = 2^(n−2)`, and its graph is completely explicit:

- one isolated vertex, `a = W` (the known isolated vertex);
- on the remaining `N = 2^(n−1) − 2` vertices, **`K_N` minus the perfect matching `a ↔ a ⊕ W`** —
  every vertex has exactly one non-neighbour, and `t2 = N(N−2)` confirms the regularity.
  ⚠ **This is a THEOREM, not a measurement** — `Qgen'_pow2_eq` (`:4587`), see §39.2; this section
  originally mislabelled it;
- **every triangle has sign product `−1`** — ✅ **now a THEOREM ∀n**, `Asig_pow2_top` (§42): the
  entry is `−μ(l)μ(y)` with `μ(a) = −1 ⟺ a` carries the label's bit, i.e. `A_σ` is **antibalanced**,
  and every triangle is negative in one line. (Note the sign: all triangles `−1` is the OPPOSITE of
  *balanced*; a balanced signed graph has every cycle `+1`.)

Counting ordered triangles in `K_N` minus a perfect matching is elementary — no triple of distinct
vertices can contain two matched pairs, so

```
#ordered triangles = N(N−1)(N−2) − 3N(N−2) = N(N−2)(N−4)
```

and with `N = 2(q−1)`, `q = 2^(n−2)`, that is `8(q−1)(q−2)(q−3)`. Hence

```
tr(A³)(n, 2^(n−2)) = −8(2^(n−2)−1)(2^(n−2)−2)(2^(n−2)−3)
```

— exact at `n = 6..11`, and **the factor 8 is the doubling `N = 2(q−1)`, not a fitted constant.**
Subtracting §17.2's `y = 0` value gives `δ(n, n−2) = −(72/7)(2^(n−2)−1)(2^(n−2)−2)(2^(n−2)−4)`,
which is §33.3's form at `j = n−2`. The two sides agree because

```
(2/7)(2q−2)(2q−4)(2q−15)  −  (72/7)(q−1)(q−2)(q−4)
    = (8/7)(q−1)(q−2)·[(2q−15) − 9(q−4)]
    = (8/7)(q−1)(q−2)(21 − 7q)  =  −8(q−1)(q−2)(q−3)
```

i.e. `t3(n,1) + δ(n,n−2) = t3(n,2^(n−2))`, with the `7` cancelling against the `21 − 7q`. That is
why both §17.2 and `δ` carry a `/7` and the answer does not. (Checked as an exact rational
identity at `n = 6..11`.)

⚠ **(B) is conditional.** §18.1's low-branch recursion is itself MEASURED, so (B) does not make
the scaling a theorem; it makes it a consequence of an earlier unproven identity. Proving §18.1 is
a load-bearing target, not a nicety. **The range gap is closed** (§34): §18.1 was measured at
`n = 7…10` when this was written and is now measured at `n = 6…11`, the same range as the law it
explains. §34 also decomposes it into four smaller statements, one of them pure algebra.

**So what is genuinely open is narrower than §33.3 suggests:** not the cubic, but

> the base case for a **general** seam at its own top-bit level — equivalently, the statement that
> `D` does not depend on `g` at all.

For `y = 0` that base is derived; for general `y` it is measured.

**The derived base and the measured law are coupled — and the coupling checks out.** At `y = 0` the
base label `W = 2^(n−2)` has `j = lsb(W) = n−2`, the MAXIMUM `j`, so §33.5(C) and §33.3 meet at a
single point and the meeting is a closed prediction for `t3(1)` — the normalising label, which is
the *subtrahend* of `D` and never its subject, hence not implied by the 2010 measurements:

```
t3(1)|ₙ = t3(2^(n−2)) − δ(n, n−2) = (16/7)·(2^(n−2) − 1)(2^(n−3) − 1)(2^(n−1) − 15)
```

Measured `4080, 52080, 504432, 4407408, 36789360, 300520560` at `n = 6…11` — **exact at every
level**. (Integrality is itself non-trivial: `7` divides the product through the `2^k − 1` factors.)
This is the closest thing to a derivation available before the §33.3 bijection.

**Dependency, stated so it lands in the graph and not only in the prose:** step (B) works *because*
the `24·t2` term dies, and it dies by the constancy of `tr(A²)` on the fibre — which is **(I)**,
`Ncnt_inj_gnorm`, a theorem as of Tier 36. (B) is not available without (I). A failed lead, recorded — **and it was refutable a priori, from data already in hand**: `P1` is
not a coboundary `μ(a)μ(b)` for `W = 2^(n−2)`. The measurement (misses on `2(N−2)` support entries
at every level) was unnecessary: if `σ(a,b) = μ(a)μ(b)` then every triangle is
`μ(a)²μ(b)²μ(c)² = +1`, and "every triangle is `−1`" was already established. **All-triangles-`−1`
IMPLIES not-a-coboundary, in one line.** The earlier write-up called the graph
"triangle-balanced", which is the inverted descriptor, and that inversion is what let a
one-line-refutable ansatz be spent on.

⚠ **Method note — this is the second instance of the `W16` pathology in this lane:** a negative
archived under the wrong descriptor, then a lead pursued that the existing record already refuted.
The standing amendment (negatives carry their residual; the descriptor enters as a hypothesis) now
gains a step: **check the descriptor against the record's own data before spending a rung.**

## §34 — §18.1 is a four-term trace expansion, not a fitted identity (`W37`, 2026-08-04)

§33.5(B) leaned on §18.1's low-branch recursion and flagged it as MEASURED and load-bearing. It is
**not proven here either** — but it stops being an opaque integer identity. Every one of its three
summands is now a separate, much smaller statement, and the leading one is pure algebra.

### 34.1 The level map is a 2-fold blow-up

With `h = 2^(n−2)`, the level-`n` vertex set `[1, 2^(n−1))` is the level-`(n−1)` vertex set `[1,h)`
**doubled** (`a₀` and `a₀+h`) plus the single extra vertex `h`. Measured, over **every** low label
at `n = 7…10` (38M entry comparisons, **0 violations**):

> **Wherever `A′(a₀,b₀) ≠ 0`, all four blocks of `A(n,W)` carry exactly `A′(a₀,b₀)`** — same sign,
> all four. I.e. `A = B + E` with `B = J₂ ⊗ A′` and `E` supported where `A′` vanishes.

`E`'s support is exactly four families, `12(h−2)` ordered pairs in total, and its shape does not
depend on the label:

| family | pairs | sign |
|---|---|---|
| (i) the matching `a₀ ↔ a₀+h` | `2(h−2)` | **`+1`**, always |
| (ii) the coset `a₀ ↔ (a₀⊕W)+h`, cross blocks only | `2(h−2)` | **`−1`**, always |
| (iii) the two copies of the isolated vertex `a₀ = W` | `4(h−2)` | mixed |
| (iv) the extra vertex `h` itself | `4(h−2)` | mixed |

Both sign functions are **constant** — measured over every low label at `n = 7…10`.

### 34.2 The recursion is the expansion of `tr((B+E)³)`, term for term

`B` and `E` are symmetric, so `tr(A³) = tr(B³) + 3tr(B²E) + 3tr(BE²) + tr(E³)`. Measured over
**every** low label at `n = 7, 8, 9` — 0 violations on each line:

| term | value | status |
|---|---|---|
| `tr(B³)` | `8·t3′` | ✅ **PROVEN ∀n as a SUM** — `tri3_Asig_blow` (Tier 43); axioms `[propext, Quot.sound]`, no `Classical.choice` |
| `3·tr(B²E)` | `24·t2′` | **DERIVED** — see §34.3 |
| `3·tr(BE²)` | `0` | MEASURED |
| `tr(E³)` | `−24(h−2) = −12(2^(n−1)−4)` | MEASURED — **this is the constant** |

So the `8` is the Kronecker factor `tr(J₂³)`, and the constant `−12(2^m−4)` is nothing but the
signed triangle sum of `E` — a `12(h−2)`-edge graph whose shape is label-independent. Neither is
fitted.

### 34.3 The `24·t2′` term, derived

`B² = J₂²⊗A′² = 2J₂⊗A′²`, so
`tr(B²E) = 2·Σ_{ε,ε'} Σ_{a₀,b₀} (A′²)_{a₀b₀} · E_{(b₀,ε'),(a₀,ε)}`.

- families **(iii)** and **(iv)** contribute **zero**, and this is an *argument*, not a
  measurement: row `W` of `A′` is zero (the isolated vertex), hence so is row `W` of `A′²`; and the
  vertex `h` lies outside `B`'s index range entirely;
- family **(i)**: `b₀ = a₀`, two `(ε,ε')` orders, sign `+1` → `4·tr(A′²) = 4·t2′`;
- family **(ii)**: `b₀ = a₀⊕W`, two orders, sign `−1` → `−4·S(W)` with `S(W) = Σ_a (A′²)[a, a⊕W]`.

and the missing ingredient is one new identity, measured over **every** label (not just low ones)
at `n = 6…9`, **0 violations**:

> **`Σ_a (A²)[a, a⊕W] = −tr(A²)`** — the signed count of 2-paths from a vertex to its coset
> partner is minus the edge count.

That gives `−4·S(W) = +4·t2′`, hence `tr(B²E) = 8·t2′` and `3·tr(B²E) = 24·t2′`. The `24` is
`3 × (4 + 4)`: three positions in the trace, one `4·t2′` from the matching and one from the coset.

### 34.4 Status

**§18.1 is not proven.** What it now rests on is four statements, each far smaller than itself:

1. the block identity `A = J₂⊗A′ + E` and `E`'s four families — MEASURED, all low labels `n = 7…10`;
2. the two constant signs `+1` (matching) and `−1` (coset) — MEASURED, same coverage;
3. `Σ_a (A²)[a, a⊕W] = −tr(A²)` — MEASURED, all labels `n = 6…9`;
4. `tr(BE²) = 0` and `tr(E³) = −24(h−2)` — MEASURED, all low labels `n = 6…11`.

**Edge coverage — both edges now closed.** §18.1 and all four terms were re-measured at `n = 6`
(from level 5; 15 low labels) and at `n = 10` (255) and `n = 11` (511): **0 violations on every
line at every level**. §33.5(B) previously leaned on §18.1 over `n = 7…10` while the (III)
deviation law is measured over `n = 6…11`, so the derivation did not reach the boundary levels and
its epistemic support there was strictly that of §18.1, not more. It now spans the same range as
the law it explains:

| level | low labels | §18.1 | `tr(B³)` | `3tr(B²E)` | `tr(BE²)` | `tr(E³)` |
|---|---|---|---|---|---|---|
| 6 | 15 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | 31 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8 | 63 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | 127 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | 255 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 11 | 511 | ✓ | ✓ | ✓ | ✓ | ✓ |

(`n = 10, 11` were run in float64, which is exact at these magnitudes — pinned by the float64
vs `int64` agreement recorded in §33.3.)

Given 1–3 the `8·t3′` and `24·t2′` terms are *derived*; only the two `E`-internal facts in 4 remain
purely measured, and they are statements about a label-independent graph on `12(h−2)` edges — the
kind of object the lane has closed before.

**Consequence for the chain.** §33.5(B) — the `8^(n−j)` scaling of the (III) deviation — now
depends on 1–4 rather than on an unexplained integer identity. (III) is still reduced, not proven;
(d) is not closed; V1 is not proven.

Reproduce: `python3 scripts/research/zd_v1_18_1_decomposition_probe.py 7 8 9`.

## §35 — The `g`-independence is a `k = 3` fact, and that kills every isomorphism route (`W38`, 2026-08-04)

> ⚠ **PARTLY DEFLATED by the prior-art scan** (`docs/research/prior_art_k3_signed_switching_2026-08-04.md`).
> The conclusion stands; two of its framings do not. (1) `tr(A^k)` of a signed graph is the signed
> closed-`k`-walk count (Belardo–Cioabă–Koolen–Wang, arXiv:1907.04349, Thm 2.2), so `k = 2` is
> `2·|E|` — it depends **only on the underlying graph** and is forced by §37's unsigned isomorphism.
> The table below over-reads it as a datum about the signature. (2) `k = 3` is not a razor-thin
> coincidence: it is the **first** moment that can see a signature at all, and by Seidel's
> two-graph correspondence the triangle signs are the **complete** invariant of the switching class
> (Brouwer, *Two-graphs*; Seidel 1968, 1976). So "the register is `k = 3`" is 1968–76 theory. What
> stays candidate-novel is the **universality in `g`**, not the register.

§33.3 proposed closing (III) by a sign-preserving bijection `a ↦ a ⊕ 8y` between the deficit
triangles of `8(y+2^i)` and those of `2^(i+3)`, and recorded that its **edge-level** form is
refuted. This rung tests the repaired forms and finds a sharper obstruction underneath all of them.

### 35.1 The contrast that frames it

(c) — the `popcount(g)` **odd** regime — is proven by `Φ`, an **isomorphism of the signed graph**.
An isomorphism preserves *every* trace, so (c)'s deviation must vanish at every power. (In the
standard vocabulary: the two signatures are **switching-equivalent**, and "switching isomorphic
signed graphs are cospectral" — Belardo et al., §2. §37.2's dichotomy is exactly the statement that
switching-equivalence holds iff `popcount(g)` is odd.) Measured:
`tr(A_W^k) = tr(A_(8g+1)^k)` for `k = 1…7`, **0 deviations**, `n = 7, 8, 9`. That is the shape of
a proof by isomorphism, and it is what one would hope to imitate in the even regime.

### 35.2 The even regime has no such object — three refutations, one cause

| candidate mechanism | verdict |
|---|---|
| edge-level transport `a ↦ a⊕8y` (§33.3) | **REFUTED** — 0/41 pairs, deficits not even the same size |
| **triple-level** transport of the per-triple deficit tensor | **REFUTED** — e.g. `n = 8, y = 6, i = 0`: `\|supp F_y\| = 1001304` vs `\|supp F_0\| = 915864` |
| `A(8(y+2^i)) ⊕ A(1)` trace-equal to `A(8·2^i) ⊕ A(8y+1)` at every `k` | **holds at `k = 1,2,3`, FAILS at `k = 4,5,6,7`** — every tested pair, `n = 7, 8` |

And the sharp statement behind all three. With `D_k(W) = tr(A_W^k) − tr(A_(8g+1)^k)` on the seams
of even-`popcount(g)` fibres:

> **`D_k` is determined by `lsb(W)` alone for `k = 2` and `k = 3`, and NOT for `k ≥ 4`.**

`k = 2` is (I), a theorem. `k = 3` is the deviation law. At `k = 4` the number of distinct values
per `lsb` is `2, 4, 8` at `n = 7, 8, 9` — i.e. `2^(n−6)`, growing with the fibre count, so the
failure is **complete, not marginal**.

### 35.3 What this kills, and what survives

> **No proof of the deviation law can go through a graph isomorphism, a switching equivalence, or
> any spectral identity.** Every such argument delivers all powers `k` at once, and `k = 4` is
> false.

In particular **(c)'s technique is not adaptable to (III)**, and that is now a measured fact rather
than a suspicion. The two halves of the fibre are structurally *unlike*: the odd half merges by an
isomorphism, the even half separates by a quantity that exists only in the closed-3-walk register.

What survives is the register §34 already works in: **a counting identity for closed 3-walks**.
That is what produced the `n`-scaling (`tr(B³) = 8t3′` and the term-by-term split of §18.1), and it
is the only style of argument the `k = 4` failure does not veto. Concretely, the next rung should
carry the §34 block decomposition to the **HIGH** branch, since §33.5(B) already reduces `D` at any
level to its value at the label's own top-bit level, where the label is high.

**(III) is still reduced, not proven. `tr(A³)`'s absolute closed form, (d) and V1 are untouched.**

Reproduce: `python3 scripts/research/zd_v1_III_mechanism_probe.py 7 8 9`.

## §36 — §34 on the HIGH branch: the deviation is created there, and it is a pure SIGN defect (`W39`, 2026-08-04)

§35 ruled out every isomorphism/spectral route and left "a counting identity for closed 3-walks",
with the instruction to carry §34's block decomposition to the high branch — which is where
§33.5(B)'s descent bottoms out. Three findings, all MEASURED (`n = 7, 8, 9`).

### 36.1 The high branch is the COMPLEMENT, not a blow-up

For a high label `W = W′ + h` (`h = 2^(n−2)`), the level-`n` resonance on the doubled vertices is
the **complement** of the level-`(n−1)` one: exact on the `(0,0)` block (0 mismatches) and off by
exactly `2(h−2)` entries on each of the other three — the same correction size as `E` in §34. This
is the graph-level content of the reflection in the proven `t2` high recursion ("counting `−1`s
becomes counting `+1`s").

> **Consequence, and it is structural rather than statistical:** the level-`n` edges sit exactly
> where level `n−1` had **no** edge, so their signs are **not determined by `A′` at all**. The
> level-`(n−1)` signed graph therefore cannot determine `tr(A³)` at level `n`.

That is §21.3's conclusion — reached there by exhausting invariants, and now visible directly in
the block structure. Two ansätze died on the way and are recorded: the per-block sign law is not a
constant (both signs occur in every block), and the ratio `P1ₙ·P1ₙ₋₁` is **not** a coboundary
`μ(a)μ(b)` — mismatches ≈ the whole block, so this is not the `GL(3,2)` situation of §17.1.

### 36.2 The top bit flips the parity — the high step CREATES the deviation

`g(W′+h) = g(W′) + 2^(n−5)`, so `popcount(g)` changes by exactly one, and the reference lifts the
same way (`8·g(W′+h)+1 = (8·g(W′)+1) + h`). Hence, measured with 0 violations:

> `D(n−1, W′) ≠ 0 ⟺ D(n, W′+h) = 0`, and when `D(n, W′+h) ≠ 0` it equals `δ(n, lsb W′)`.

So the high step is **not** a scaling recursion. The pair `(W′, 8g(W′)+1)` that lifts to a nonzero
deviation is precisely the pair whose level-`(n−1)` deviation is **zero** — i.e. a pair that (c)
has *merged*, cospectral by the proven isomorphism `Φ`. The whole of `δ` is generated in one step
from an isomorphic pair.

> **The clean restatement of (III)'s open core:** `δ(n,j)` is the **obstruction to lifting `Φ`
> through the high step**. `Φ` is explicit and proven ∀n; what is open is the defect of its lift.

That is also why §21.2 found the colliding level-`(n−1)` labels to have identical full spectra —
they are (c)-merged by construction. §21's negative and this rung are the same phenomenon.

### 36.3 The deviation is a PURE SIGN defect

A seam and its Fano reference have the same edge count `t2` (that is (I), a theorem) — and also:

> the **unsigned** graphs `|A|` are **cospectral at every power** `k = 1…6` and share degree
> sequences, for **every** seam at `n = 7, 8, 9` (7/7, 15/15, 31/31), while the **signed**
> `tr(A³)` differs by `δ` on exactly the even-`popcount(g)` ones.

(The unsigned graphs are not entrywise equal — 0/53 — so this is cospectrality, not identity.)

**So the entire deviation lives in the signs.** The unsigned graph is blind to it at every order;
the signed graph sees it only at `k = 3` (§35). Together:

| | seam vs its Fano reference |
|---|---|
| unsigned `tr(\|A\|^k)`, all `k` | **equal** |
| signed `tr(A^k)`, `k = 1, 2` | equal (`k=2` is (I), a theorem) |
| signed `tr(A³)` | differs by `δ(n, lsb)` — universal in `g` |
| signed `tr(A^k)`, `k ≥ 4` | differs, and `g`-dependently (§35) |

`δ` is therefore a **signature invariant, not a graph invariant**, and the sign difference is not a
switching (a switching would preserve every `k`). The next object to study is that sign difference
itself: a `±1` perturbation on a cospectral pair whose only `g`-universal footprint is its
closed-3-walk sum.

**(III) is still reduced, not proven.** `tr(A³)`'s absolute closed form, (d) and V1 are untouched.

Reproduce: `python3 scripts/research/zd_v1_III_high_branch_probe.py 7 8 9`.

## §37 — The sign defect named: `Φ = τ_j`, and its curvature lives BELOW the seam (`W40`, 2026-08-04)

§36 showed the deviation is a pure sign phenomenon on a cospectral pair but did not name the map.
It is the lane's own `τ` — `tau j x = x` if `bit₀(x) = bit_j(x)`, else `x ⊕ (1 ∣ 2^j)`, the same
`τ` as `Qgen'_tau` (`:1714`) — with `j = lsb(W)`, and note `τ_j(W) = 8·g(W)+1` is *exactly* the
Fano representative. All MEASURED, `n = 7, 8, 9`.

### 37.1 `Φ = τ_j` is an unsigned isomorphism — always

> `|A_W|(a,b) = |A_(τ_j W)|(τ_j a, τ_j b)` for **every** seam, **both parities**. 0 violations.

That is §36.3's cospectrality upgraded to an explicit map, and it means the seam and its Fano
reference differ **only** in signs, on a common support:

```
ε(a,b) = A_W(a,b) · A_(τW)(τa, τb)   ∈ {±1} on the support
```

### 37.2 The dichotomy: `ε` is balanced exactly in the merging regime

> **Vocabulary, from the prior-art scan.** `ε(a,b) = μ(a)μ(b)` is Zaslavsky's **switching function**
> (`σ^θ(vw) = θ(v)σ(vw)θ(w)`, arXiv:1303.3083 §I.G) verbatim; "balanced" is Harary's Balance Theorem
> (ibid., Thm I.2); and §39.3's `A_σ(a,b) = −μ(a)μ(b)` says `A_σ` is **antibalanced**, from which
> `tr(A³) = −#triangles` is routine. These are not new objects — state them with the standard names
> and the 1968–76 attributions.

> **`ε` is balanced (every triangle `+1`) ⟺ `popcount(g(W))` is ODD.** 0 violations.

Balanced means `ε` is a coboundary, i.e. `A_W = D·(Φ*A_f)·D` with `D = diag(±1)` — a **switching**,
which preserves every trace. So this *is* (c), rederived from the sign side, and it explains §35's
`C1` exactly: (c) holds at every power `k` because a switching is a similarity, while the even
regime differs only at `k = 3` because there the failure is a **curvature**, and curvature is a
3-cycle invariant.

Measured curvature counts (ordered triangles with `ε_T = −1`), `n = 7`: `0` for every
odd-`popcount(g)` seam; `69120, 111744, 133920, 59904` for the even ones. The dichotomy is total,
not statistical.

### 37.3 The mechanism: the curvature is determined BELOW the seam

> **`ε_T = ε(a,b)ε(b,c)ε(c,a)` is determined by the low `j+1` bits of `(a,b,c)` alone.**
> `n = 7`: `W = 8` (`j = 3`) → 4096 classes, **0 ambiguous**; `W = 16` (`j = 4`) → 24410 classes,
> 0 ambiguous. `n = 8`: same, 0 ambiguous over 317688 sampled triangles.

This is the mechanism the lane has been looking for since §33:

- **why `δ` does not depend on `g`** — `g` is the bits **above** the seam, and the curvature cannot
  see them. The `g`-independence stops being a brute fact;
- **why `[j choose 3]₂`** — the curvature is a function on the `j+1` coordinates at and below the
  seam, so its triangle sum is a count over that space, and `δ`'s closed form is
  `δ(n,j) = (2^(n−j−2))³ · (−1728 · [j choose 3]₂)`: the cube of the class size times a
  **level-independent** constant. That is exactly the shape a low-determined curvature forces;
- **why `k = 3`** — curvature is defined on 3-cycles. At `k ≥ 4` the sum is not a curvature sum and
  §35 measured that it is `g`-dependent.

### 37.4 What is now open, and it is finite per `j`

(III) reduces to two statements about `ε`, both about an object on `j+1` bits rather than on the
whole label:

1. `ε` is balanced ⟺ `popcount(g)` odd (the (c)/¬(c) dichotomy, from the sign side);
2. the curvature's triangle sum over the low classes is `−1728·[j choose 3]₂` — a **fixed finite
   computation for each `j`**, independent of `n`.
   ⚠ **§39: (2) is not a separate item.** It is `δ(j+2,j)`, hence §17.2 — §33.5's collapse identity
   already contained it. Stating it here as newly-open was the error that produced §38's ladder.

Neither is proven. But (2) is no longer a statement about an unbounded family: it is one number per
`j`. **(III) is still reduced, not proven.** `tr(A³)`'s absolute closed form, (d) and V1 untouched.

Reproduce: `python3 scripts/research/zd_v1_III_sign_defect_probe.py 7 8 9`.

## §38 — The curvature sum, computed for `j = 3…13`: `K_j` is a level-independent tensor and `Σ K_j = −1728·[j,3]₂` (`W41`, 2026-08-04)

> ⚠⚠ **DEFLATED BY §39 — read this first.** The clause this section computes eleven times is not
> independent content: at `cls = 1` it is `δ(j+2, j)` **by definition**, and §33.5's own collapse
> identity already gave `δ(j+2, j) = −(72/7)(q−1)(q−2)(q−4)`, which *is* `−1728·[j,3]₂`. So the
> `j`-ladder below is **eleven evaluations of one cubic identity, not eleven independent
> confirmations**, and its evidential weight is that of a single item. §37.4 introduced it as "the
> open content, finite per `j`" when the line had already been written in §33.5. The engineering
> the ladder forced (four implementations, a 550-billion-entry tensor at 1873 s) was spent
> recomputing a quantity that has a closed form and costs two traces.
>
> What survives as **new** in this section: the tensor's *structure* — values exactly `{0, −2}`,
> the flipped count `864·[j,3]₂`, the support-independence condition, and level-independence entry
> by entry. Those are not implied by the sum. The *sum* was never open.

§37.4 left (III)'s open content as **one number per `j`**. That number is now computed for
`j = 3 … 13`, and it comes with more structure than was asked for.

Partition the vertices by their low `j+1` bits — `M = 2^(j+1)` classes, each of size
`cls = 2^(n−j−2)` — and with `A` the seam's graph and `Bp = Φ*A_(τW)` the transported reference:

```
K_j(u,v,w) := [ tr(A_uv A_vw A_wu) − tr(Bp_uv Bp_vw Bp_wu) ] / cls³
```

| `j` | `Σ K_j` | `−1728·[j,3]₂` | flipped classes = `864·[j,3]₂` | values taken |
|---|---|---|---|---|
| 3 | `−1728` | `−1728·1` | `864` | `{0, −2}` |
| 4 | `−25920` | `−1728·15` | `12960` | `{0, −2}` |
| 5 | `−267840` | `−1728·155` | `133920` | `{0, −2}` |
| 6 | `−2410560` | `−1728·1395` | `1205280` | `{0, −2}` |
| 7 | `−20409408` | `−1728·11811` | `10204704` | `{0, −2}` |
| 8 | `−167883840` | `−1728·97155` | `83941920` | `{0, −2}` |
| 9 | `−1361724480` | `−1728·788035` | `680862240` | `{0, −2}` |
| 10 | `−10968851520` | `−1728·6347715` | `5484425760` | `{0, −2}` |
| 11 | `−88051917888` | `−1728·50955971` | `44025958944` | `{0, −2}` |
| 12 | `−705621533760` | `−1728·408345795` | `352810766880` | `{0, −2}` |
| 13 | `−5649800569920` | `−1728·3269560515` | `2824900284960` ⁽*⁾ | `{0, −2}` ⁽*⁾ |

⁽*⁾ `j = 13` is the first row where the tensor was **not** enumerated: at `M = 16384` it has
4.4 **trillion** entries (≈ 4 h). The **sum** is exact — it is `δ(15,13)`, two traces, 181 s — and
the values were checked on **16 of 16384** `u`-slices, all `{0, −2}` with 0 support violations.
The flipped count is then `−ΣK/2`, exact *given* that every value is `0` or `−2`; that hypothesis
is sampled at `j = 13`, exhaustive at `j ≤ 12`.

Four facts, all measured, `n = 7…15` (each `j` at `n = j+2`, where `cls = 1`, and at `n = j+3`;
from `j = 11` the second level is covered by the `δ` cross-check rather than by the tensor):

1. **`K_j` is level-independent — the whole tensor, entry by entry**, not merely its sum
   (`np.array_equal` across consecutive levels, exact, `n = 7…10`). This is the factorisation
   §37.3 predicted and did not verify: `δ` really is `cls³ × (a fixed object on `(𝔽₂^(j+1))³`)`.
   It holds at levels where the label is **high** as well as low (`j = 6` at `n = 8`, where
   `W = 2^6 = 2^(n−2)`), so `K_j` does not depend on the branch either.

   **And it identifies `K_j`.** At `n = j+2` the class size is `cls = 1`, so `K_j` *is* the raw
   per-triangle defect table at that level. Level-independence then says every higher level is a
   uniform `cls`-fold blow-up of that one table — which is exactly why `δ`'s only `n`-dependence is
   the factor `cls³`. `j = 7` at `n = 9` is that base case computed directly.
2. **`K_j` takes only the values `0` and `−2`.** Every contributing class is a *single sign flip* —
   there is no cancellation and no higher multiplicity to explain.
3. **The number of flipped classes is exactly `864·[j choose 3]₂`.** The `q`-binomial is not an
   artefact of the closed form; it counts them.
4. **Every nonzero entry has `(u⊕v, v⊕w)` linearly independent** — the support sits inside the
   non-degenerate triples, as the `Inj(𝔽₂³, ·)` reading of §33.3 predicted.

Hence, assembled rather than fitted:

```
δ(n,j) = cls³ · Σ K_j = (2^(n−j−2))³ · (−1728) · [j choose 3]₂ = −27 · 8^(n−j) · [j choose 3]₂
```

verified against the measured `δ` at every `(j, n)` in the table.

### 38.1 What is left

The `n`-dependence of `δ` is now **structural**: it is `cls³`, the cube of the class size, and it is
forced by `K_j` being level-independent. What remains open is one clause:

> `Σ K_j = −1728·[j,3]₂` for **all** `j` — computed here for `j = 3 … 13`, not proven.

`j = 6` needed a different implementation: `M = 2^7 = 128` makes the `M³` triple loop unworkable,
so the probe blocks the matrices by residue and contracts over the block indices (`M³·cls³` work,
vertex `0` added as isolated so the reshape is uniform). The fast path reproduces the
`j = 3, 4, 5` numbers of the slow one exactly, which is the cross-check that makes `j = 6` and
`j = 7` trustworthy. `j = 7` (`M = 256`, a 16.8M-entry tensor) additionally needs the contraction
accumulated one block-triple at a time so only one `M³` temporary is live. `j = 8` (`M = 512`,
**134M entries**) cannot hold the tensor at all: the probe's `stream` path walks `u` and keeps only
one `M×M` slice, accumulating the sum, the value histogram, the support check and a weighted linear
**checksum**. Equal checksums plus equal histograms pin the tensor across levels without ever
holding it — `n = 10` and `n = 11` both give `2294549568037756351`. `j = 9` (`M = 1024`,
**1.07 BILLION** entries) runs the same way in 43 s at `n = 11` and 68 s at `n = 12`, both
`1573899305250049471`. `j = 10` is **8.6 billion** entries and needed one more idea: the checksum
is computed from row and column sums rather than by extracting nonzeros
(`Σ K·(uM²+vM+w) = uM²·ΣK + M·⟨v, rowsums⟩ + ⟨w, colsums⟩`), which drops it to ~40 s per level;
the support check, which does need indices, is then sampled every 64 slices. `n = 12` and `n = 13`
are levels the lane's trace measurements had never reached — this needs one label and its
reference, not a sweep or an eigendecomposition.

`j = 11` is 68.7 billion entries and takes 458 s at `n = 13`; `j = 12` is **550 billion** and takes
1873 s at `n = 14` (`int8` slices — the values never leave `{−2…2}`). The `cls = 2` tensors above
`j = 10` were not run.

**Cost note, and it is the useful one.** At `cls = 1` — i.e. at `n = j+2`, the canonical level —
`Σ K_j` is *by definition* `δ(j+2, j)`, which is a **two-trace computation**: 25 s at `j = 12`
against 1873 s for the tensor, and 181 s at `j = 13` against an estimated 4 h. So the *sum* never
needs the tensor at all. What the tensor buys is the **structure** — that the values are exactly
`{0, −2}`, that the flipped count is `864·[j,3]₂`, and that the support satisfies the independence
condition. And the first of those implies the second, given the sum, so from `j = 13` the probe
computes the sum exactly and samples the value/support checks.

The level-independence cross-check stops at `j = 12`: `δ(16,13)` would need `n = 16`, where
`A_sig`'s `int16` intermediates alone come to ~17 GB. Reaching it needs a chunked generator, which
is not written.

Independently, `δ` itself was computed straight from the traces, which is *much* cheaper and
confirms the `cls³` blow-up without touching the tensor:

| | measured `δ` | `−27·8^(n−j)·[j,3]₂` | cost |
|---|---|---|---|
| `δ(12,10)` | `−10968851520` | ✓ | 1 s |
| `δ(13,10)` | `−87750812160` | ✓ | 4 s |
| `δ(13,11)` | `−88051917888` | ✓ | 4 s |
| `δ(14,11)` | `−704415343104` | ✓ | 21 s |
| `δ(14,12)` | `−705621533760` | ✓ | 25 s |
| `δ(15,12)` | `−5644972270080` | ✓ | 147 s |

`n = 15` needs a chunked `float32` product for `tr(A³)` (entries of `A²` stay below `2^24`, so the
matmul is exact; the outer accumulation is `float64`).

`864 = 27·32` flips per 3-subspace is the shape to explain; the linear-independence of the support
(fact 4) is the visible half of it. Note this is a statement about a **finite** object at each `j`,
with no level parameter, so it is the first piece of (III) that is checkable by direct enumeration
rather than by a family of measurements.

**(III) is still reduced, not proven.** `tr(A³)`'s absolute closed form, (d) and V1 untouched.

Reproduce: `python3 scripts/research/zd_v1_III_curvature_sum_probe.py`.

## §39 — The finite clause is not new content, and half of it was already a theorem (`W42`, 2026-08-04)

§38 left `Σ K_j = −1728·[j,3]₂` as the open clause. Attacking it directly collapses it.

### 39.1 The clause IS §17.2

At `n = j+2` the class size is `1`, so by definition

```
Σ K_j = δ(j+2, j) = tr(A³)(j+2, 2^j) − tr(A³)(j+2, 1)
```

— the seam label `2^j = 2^(n−2)` against the `y = 0` Fano representative, at the same level. §33.5(C)
gives the first term as `−N(N−2)(N−4)`, `N = 2^(n−1) − 2 = 2q − 2`, `q = 2^j`. Substituting and
comparing coefficients in `q` (exact, over ℚ):

```
−1728·[j,3]₂           has q-coefficients ( 576/7, −144,  72, −72/7 )
tr(A³)(2^j) − tr(A³)(1) has the same, iff  tr(A³)(1) = (2/7)(2q−2)(2q−4)(2q−15)
```

and that right-hand side **is §17.2**, the `y = 0` closed form, measured since `W24`. So:

> **Given §33.5(C), the finite clause and §17.2 are the same statement**, an identity of cubics in
> `q = 2^j`. Checked as a polynomial identity and numerically at `j = 3…13`.

The clause was never independent open content. What §38's ladder measured, eleven times, was §17.2
in another coordinate.

### 39.2 §33.5(C)'s first half is a THEOREM, already in the file

§33.5(C) listed the graph structure of `W = 2^(n−2)` as measured. It is not — it is
`Qgen'_pow2_eq` (`:4587`), proven ∀n:

```lean
Qgen'_pow2_eq : a ≠ 0 → b ≠ 0 → a ≠ b →
    Qgen' (2^k) a b m = if a = 2^k ∨ b = a ^^^ 2^k then 1 else -1
```

`+1` on exactly two lines — `a = W` (the isolated vertex) and `b = a ⊕ W` (the perfect matching) —
and `−1`, i.e. an edge, everywhere else. That is precisely "`K_N` minus the coset matching, plus one
isolated vertex", with no counting argument needed. The edge count it forces,
`(H−2)(H−4) = N(N−2)`, is `Ncnt_pow2`'s value, which is the consistency check.

**Correction to §33.5(C):** that bullet is proven, not measured, and this spec said otherwise.

### 39.3 The second half reduces to one explicit entry formula

What remained was "every triangle has sign product `−1`" — a global statement. It is a coboundary,
and the coboundary is **trivial**:

> **`A_σ(a,b) = −μ(a)·μ(b)` on every edge, with `μ(a) = −1 iff a ∧ W ≠ 0`.**
> In words: the entry is `−1` when `a` and `b` lie on the **same side of the label's bit** and `+1`
> when they **straddle** it.

Measured at `n = 6…12` — **5.5M edges, 0 exceptions**. (Found by building `μ` along a spanning tree
and reading it off; the earlier failed coboundary test in §33.5 used a base vertex whose one
non-neighbour poisoned the propagation.)

Given that formula, `tr(A³) = Σ_T (−μμ)³ = −Σ_T μ(a)²μ(b)²μ(c)² = −#T` in one line, which is
§33.5(C)'s value.

### 39.4 The ledger, after this rung

| statement | status |
|---|---|
| the graph of `W = 2^(n−2)` is `K_N` minus the coset matching, plus one isolated vertex | **THEOREM ∀n** (`Qgen'_pow2_eq`) |
| `A_σ(a,b) = −μ(a)μ(b)`, `μ(a) = −1 iff a ∧ W ≠ 0` | ✅ **THEOREM ∀n** — `Asig_pow2_top` (§42) |
| ⟹ `tr(A³)(2^(n−2)) = −N(N−2)(N−4)` | one line from the two above |
| §17.2: `tr(A³)(y=0) = (2/7)(2^m−2)(2^m−4)(2^m−15)` | MEASURED |
| **`Σ K_j = −1728·[j,3]₂`** | **⟺ §17.2**, given the two above — proven equivalence, this rung |

So the finite clause is not proven, but it is no longer a separate target: it is §17.2, and the
route to it is one explicit sign identity plus a closed form the lane has carried since `W24`.
The `j`-ladder can stop; extending it re-measures §17.2.

**(III) is still reduced, not proven.** (d) and V1 untouched.

## §40 — The annihilation graph IS the zero-divisor graph, and `A_σ`'s sign is the annihilating sign (`W43`, 2026-08-04)

The prior-art scan (§3A.4 of `prior_art_k3_signed_switching_2026-08-04.md`) left the lane owing a
definitional check: Zhilina (IJAC 31(4), 2021) states that for the **main sequence** the zero-divisor
graph coincides with the orthogonality graph, so if the lane's object is that graph, its *support*
is a named, studied thing. It is — and the check also gives the signature an algebraic meaning it
did not have before.

### 40.1 What a vertex is

`A_σ` is indexed by `l ∈ [1, 2^(n−1))` with `hi(l) = l ⊕ L`, `L = Llo | 2^(n−1)`. So a vertex is a
**pencil element** `x_l = e_l + ε_l·e_(l⊕L)` — a two-term element whose two indices XOR to the label.
(Moreno's opening example is of exactly this shape: `x = e₁ + e₁₀`, `y = e₁₅ − e₄`, and
`1⊕10 = 15⊕4 = 11`.)

### 40.2 The identification is a theorem, not a measurement

```
x_l x_y = [ σ(l,y)   + ε_l ε_y σ(l⊕L, y⊕L) ] · e_(l⊕y)
        + [ ε_y σ(l, y⊕L) + ε_l σ(l⊕L, y) ] · e_(l⊕y⊕L)
```

The first bracket vanishes iff `ε_l ε_y = −P1(l,y)`; the second iff `ε_l ε_y = −P3(l,y)`. So

> **`x_l x_y = 0` for some choice of relative sign ⟺ `P1 = P3`, and then the relative sign is
> `ε_l ε_y = −P1(l,y) = A_σ(l,y)`.**

`A_σ` demands `P1 = P3` **plus** the two symmetry clauses — and those are `P1_symm` and `P3_symm`,
**proven ∀n in this file**. Hence `supp(A_σ) = {P1 = P3} =` the annihilation relation, by lemmas
already in the tree.

> ⚠ **Corrected in §41 after reading their Definition 2.3.** The symmetry clauses are not "extra":
> Guterman–Zhilina's adjacency is **two-sided** (`ab = ba = 0`), and reversing the product turns the
> two conditions into `ε_l ε_y = −P1(y,l)` and `= −P3(y,l)`. So `resB` — `P1` symmetric, `P3`
> symmetric, `P1 = P3` — **is exactly the two-sided condition**, and it collapses to the one-sided one
> only because `P1_symm`/`P3_symm` are theorems.

Two consequences, and the second is the one that changes how the object should be read:

1. **The support is an induced subgraph of the Cayley–Dickson zero-divisor graph** — of the
   orthogonality graph too, by Zhilina's coincidence for the main sequence — induced on the pencil
   of two-term zero divisors with a **fixed XOR label**. It is not a new graph and must be cited as
   theirs.
2. **`A_σ`'s sign is not decoration: it is the relative sign that makes the product vanish.** So the
   *signed* graph is the natural object here, and its **balance** has an algebraic reading:

   > `A_σ` is balanced ⟺ the signs `ε_l` can be chosen **globally** so that every one of these
   > products vanishes simultaneously. The frustration of `A_σ` is exactly the obstruction to a
   > coherent choice.

   §39.3's `A_σ(a,b) = −μ(a)μ(b)` for `W = 2^(n−2)` says the obstruction there is *total*: over ℝ a
   global `ε` would need `ε_l = c·μ(l)` with `c² = −1`.

### 40.3 Verification

| clause | result |
|---|---|
| `C1` `sign_table_fast` really is CD basis multiplication, vs an **independent** doubling recursion | `n = 4,5,6` — **0 mismatches** |
| `C2a` `x_l x_y = 0` by **real CD multiplication** vs `A_σ`, sedenions | 294 ordered pairs, 7 labels — **0 support, 0 sign mismatches**, and no pair annihilates for both relative signs |
| `C2b` annihilation `⟺ P1 = P3` and `A_σ = −P1` on the support | `n = 5…8`, 2.33M entries over 236 labels — **0** |
| `C3` at `n = 4` every label gives `K₆` minus the coset matching plus the isolated vertex `l = Llo` — the **octahedron** | **7 of 7** |

`C3` is the point of contact with the published `n = 4` theory: 6 vertices, 4-regular, 12 edges, and
non-adjacency exactly the coset pairing is the shape of the **double hexagon** that Guterman &
Zhilina obtain from any pair of sedenion zero divisors (Zap. Nauchn. Sem. POMI **496** (2020) 61–86).
Their theorem plausibly *is* our `n = 4` base case; asserting identity needs their definition read in
full, which is still owed.

### 40.4 The novelty ledger

*Rewritten 2026-08-04 after §41. The first version of this subsection read "the graph is theirs, the
signature is ours"; §41.3 shows that is too generous, because a double cover determines its signature
up to switching. This is the single place to check before any write-up.*

| item | status | who to cite | confidence |
|---|---|---|---|
| the underlying graph — `supp(A_σ)` | **not ours.** It is the CD zero-divisor graph (= orthogonality graph on the main sequence, Zhilina IJAC 2021), induced on the fixed-XOR-label pencil of two-term zero divisors | Guterman–Zhilina; Zhilina | **high** — proved in §40.2, verified against real CD multiplication |
| the two-sided adjacency `ab = ba = 0` ≡ `resB` | **not ours** — it is their Definition 2.3 | Guterman & Zhilina (2020), Def. 2.3 | **high** — §41.1 |
| the **signature** `A_σ`, as data | **not independent.** `Γ_O` on a pencil is its double cover (§41.2), and a double cover fixes the signature up to switching — so the *switching class* is equivalent to their unsigned line-graph wherever their results reach | as above | **high** — verified edge-for-edge at `n = 4` |
| `n = 4`: double hexagons, components ↔ Fano lines, diameters | **not ours** | Guterman & Zhilina, Zap. Nauchn. Sem. POMI **496** (2020) 61–86 | **high** — their abstract, verbatim |
| the `k = 3` register; "balanced", "switching", "two-graph", "antibalanced" | **not ours** — classical | Seidel (1968, 1976); Zaslavsky; Harary; Belardo et al. Thm 2.2 | **high** — see the prior-art report |
| the **spectral** layer: `tr(A²)`, `tr(A³)` as invariants of the label, their closed forms | **not located** in the CD graph literature — `спектр`/`spectr` appears **0 times** in either Guterman–Zhilina paper; they compute components, diameters, cliques, hexagons, never a spectrum | — | **medium** — keyword sweep over full texts, bodies not read in detail |
| the fibre/seam classification, `g`, `τ`-equivariance, the counting recursions, the deviation law, `δ = −27·8^(n−j)[j,3]₂` | **not located** | — | **medium-low**, see the risk below |

> ⚠ **The top novelty risk is not `n = 4`, it is ∀n.** Zhilina, *Orthogonality graphs of real
> Cayley–Dickson algebras. Part I: doubly alternative zero divisors and their hexagons* (IJAC
> **31**(4), 2021) is about **arbitrary real Cayley–Dickson algebras**, not just sedenions, and it is
> **unread** — I could not obtain it. Its abstract already asserts, for the main sequence, that the
> zero-divisor graph coincides with the orthogonality graph and that hexagons extend to double
> hexagons *in an arbitrary real CD algebra*. Until it is read, **no ∀n structural claim in this lane
> should be presented as new.**

**Owed before any write-up, in order:** (1) obtain and read Zhilina's IJAC Part I; (2) read the
Russian bodies of both Guterman–Zhilina papers in detail, not just their abstracts; (3) MathSciNet /
zbMATH on the `g`-universality and on `[j,3]₂` as a signed-triangle count. Only item (3) is a search;
(1) and (2) are readings, and (1) is the one that can move the ledger.

Reproduce: `python3 scripts/research/zd_annihilation_is_orthogonality_probe.py`.


## §41 — Their Definition 2.3, read: `Γ_O` restricted to a pencil is the DOUBLE COVER of `A_σ` (`W44`, 2026-08-04)

Definition 2.3 of Guterman & Zhilina (Zap. Nauchn. Sem. POMI **496** (2020) 61–86), verbatim from
the Russian original:

> **Граф ортогональности `Γ_O(A)`** определяется следующим образом: его вершины — прямые в
> `Z_LR(A)`, то есть `V(Γ_O(A)) = P(Z_LR(A))`, причём различные вершины `[a]` и `[b]` соединены
> ребром, если и только если **`ab = ba = 0`**.

Two features that §40 did not have right.

### 41.1 The adjacency is TWO-SIDED, and `resB` is exactly that

Reversing the product gives `ε_l ε_y = −P1(y,l)` and `ε_l ε_y = −P3(y,l)`. So two-sided annihilation
with one relative sign is `P1 = P3 = P1ᵀ = P3ᵀ` — **precisely `resB`**. The lane's resonance
predicate is not "`P1 = P3` plus two convenience clauses"; it *is* `ab = ba = 0`. The clauses
collapse only because `P1_symm`/`P3_symm` are proven. Measured at `n = 4` with real CD
multiplication: `A_σ` vs the two-sided relation, **0 mismatches**, and one-sided ≡ two-sided, **0**.

### 41.2 Their vertices are LINES — so their graph is our double cover

`[a] = ℝa`, so each pencil pair `{l, l⊕L}` contributes **two** vertices, `[e_l + e_(l⊕L)]` and
`[e_l − e_(l⊕L)]`, where the lane has **one** vertex carrying a sign. That is the classical
signed-graph ↔ double-cover correspondence, and it holds on the nose:

> **`Γ_O` restricted to a fixed-XOR-label pencil is the canonical double cover of `A_σ`.**
> Verified at `n = 4`, all 7 labels: 12 zero-divisor lines of the 14 candidates, **24 edges**, and
> the edge set is **identical** to the double cover `(l,ε) ∼ (y, ε·A_σ(l,y))` — 7 of 7, exact.

This also identifies the point of contact by name. Our per-label graph at `n = 4` is the octahedron
(6 vertices, 12 edges); its double cover is 4-regular on **12** vertices, which is the shape of their
**double hexagon**.

### 41.3 The consequence, and it cuts

A double cover determines its signature **up to switching**, and switching class is exactly the
spectrally meaningful datum (§35, §37). Therefore:

> **The lane's switching class is equivalent data to their (published) unsigned line-graph.** The
> signature is a choice of representative in it, not independent content.

So §40.4's split — "the graph is theirs, the signature is ours" — is too generous to us. The honest
version:

- **theirs:** the underlying graph *and*, up to switching, the signature too, wherever their results
  reach — which at `n = 4` is everything;
- **not located in their work, on present evidence:** the ∀`n` statements; the **spectral** layer
  (`спектр`/`spectr` appears **0 times** in either paper — they compute components, diameters,
  cliques, hexagons, never a spectrum); `tr(A²)`/`tr(A³)` as invariants; the fibre/seam
  classification; the deviation law.

The `n = 4` base case of this lane should be cited to Guterman–Zhilina 2020, not presented as found
here.

Reproduce: `python3 scripts/research/zd_annihilation_is_orthogonality_probe.py` (clauses A and B
added).


## §42 — The sign formula is a theorem ∀n: `A_σ` is antibalanced on the top-bit label (`W45`, 2026-08-04)

`Asig_pow2_top`, kernel-clean `[propext, Classical.choice, Quot.sound]`:

```lean
Asig_pow2_top (n l y : Nat) : l < 2^(n+1) → y < 2^(n+1) → l ≠ 0 → y ≠ 0 →
    l ≠ 2^n → y ≠ 2^n → resB l y (2^n) n = true →
    Asig l y (2^n) n = - (muTop l n * muTop y n)
```

with `muTop a n = if a < 2^n then 1 else -1`. In words: **the entry is `−1` when `l` and `y` lie on
the same side of the label's bit and `+1` when they straddle it.** §39.3 had this measured over 5.5M
edges at `n = 6…12`; it is now proven.

In signed-graph language `muTop` is Zaslavsky's **switching function** (arXiv:1303.3083 §I.G) and the
statement is that **`A_σ` is antibalanced** — `−A_σ` switches to all-positive. Every triangle is then
negative immediately.

### 42.1 The proof, and why the coset line had to be handled

`P1_red` drops the level; then four cases on which side of the bit `l` and `y` lie, through the
existing branch reductions:

| case | `cdSigma l y` | `cdSigma (y⊕2^n) (l⊕2^n)` | product |
|---|---|---|---|
| both below | `R_ll` → `σ(l,y)` | `R_uu` → `σ(l,y)` | `σ² = +1` |
| `l` below, `y` above | `R_lu` → `σ(y′,l)` | `R_lu` → `σ(l,y′)` | `−1` by `antisym` |
| `l` above, `y` below | `R_ul` → `−σ(l′,y)` | `R_ul` → `−σ(y,l′)` | `−1` by `antisym` |
| both above | `R_uu` → `σ(y′,l′)` | `R_ll` → `σ(y′,l′)` | `σ² = +1` |

and `μ(l)μ(y)` is `+1`, `−1`, `−1`, `+1` in those four cases. The mixed cases need `antisym`, which
requires `u ≠ v` — and `u = v` is **exactly** `y = l ⊕ 2^n`, the coset line.

That line is excluded by proof, not by hypothesis: `P1_coset` gives `P1 = 1` (it is a square,
`cdSq`) and `P3_coset` gives `P3 = −1` (two applications of `sigma_self`), so `resB_coset` shows the
line is not in the support. Hence `Asig_pow2_top` assumes only that `(l,y)` is an edge.

### 42.2 Where this leaves the chain

§33.5(C)'s value `tr(A³)(2^(n−2)) = −N(N−2)(N−4)` now rests on **two theorems**:

| ingredient | status |
|---|---|
| the graph is `K_N` minus the coset matching, plus one isolated vertex | **THEOREM** — `Qgen'_pow2_eq` (§39.2) |
| every entry is `−μ(l)μ(y)`, hence every triangle is `−1` | **THEOREM** — `Asig_pow2_top` (this rung) |
| antibalance ⟹ `tr(A³) = −#triangles`, and `#triangles = N(N−2)(N−4)` | elementary counting, **not yet in Lean** |
| §17.2: `tr(A³)(y = 0) = (2/7)(2^m−2)(2^m−4)(2^m−15)` | ✅ **DERIVED from §18.1** — see §45 |

and by §39.1 the finite clause `Σ K_j = −1728·[j,3]₂` is *equivalent* to §17.2 given the first three.
So **§17.2 is now the only measured statement left in that chain**, and it is the next target.

**(III) is still reduced, not proven.** (d) and V1 untouched.

## §43 — Why §17.2 does NOT yield to the §42 technique: the absolute triangle sign has no bounded window (`W46`, 2026-08-04)

§42 left §17.2 as the only measured statement in the chain. It does not fall to the same method, and
the obstruction is measurable rather than a matter of effort.

### 43.1 What carries over, and what does not

Carries over: the **unsigned** graph of the `y = 0` Fano label `Llo = 1` is again `K_N` minus a
perfect matching — its unsigned triangle count is exactly `N(N−2)(N−4)` at `n = 6…10` — which is
§37's `τ`-isomorphism, already known.

Does **not** carry over: `A_σ(1)` is **not antibalanced** (its negative-triangle count is not `T`,
else `δ` would vanish), and its signature is **not bilinear** — the multiplicativity test
`s(l⊕l′,y) = s(l,y)·s(l′,y)` fails on **half** the triples tested. So there is no coboundary and no
`(−1)^{B(l,y)}` form to exploit; §42's whole route is unavailable.

### 43.2 The measured obstruction

The technique behind §38 and §42 is *locality*: §37.3 found the **curvature** `ε_T` determined by the
low `j+1` bits of `(a,b,c)` — a window fixed by the label, which is what makes `K_j` a finite object
and the whole `n`-dependence a `cls³` blow-up. The same question for the **absolute** triangle sign
of `A_σ(1)`:

| level | smallest `k` such that the triangle sign is determined by the low `k` bits |
|---|---|
| `n = 7` | **5** = `n − 2` |
| `n = 8` | **6** = `n − 2` |

> **The window is `n − 2` — it grows with the level, i.e. the absolute triangle sign needs
> essentially every bit of the label.** There is no bounded local rule, so there is no finite object
> per level to enumerate, and the §38/§42 method has nothing to bite on.

This is the sharpest statement yet of a pattern the lane has been living with since `W24`:
**relative quantities here are local; absolute ones are not.** §33's headline (the deviation forgets
`g`) and this negative are two faces of the same fact, and it is now measured on both sides.

### 43.3 The route that remains, and it is well-defined

By §39.1, §17.2 ⟺ `Σ K_j = −1728·[j,3]₂`, and by §38 that is equivalent to

> the number of curvature-flipped classes in `(𝔽₂^(j+1))³` is exactly `864·[j,3]₂`.

That **is** a bounded-window statement — §37.3 guarantees the window — so it is finite combinatorics
on `𝔽₂^(j+1)`, with no Cayley–Dickson algebra left in it, once one has the curvature as a **formula**
rather than as a table. §38 established the table (values `{0,−2}`, support inside the
non-degenerate triples, `864·[j,3]₂` of them) but never extracted the rule.

**Next target, concrete:** extract the explicit local rule for `ε_T` as a function on
`(𝔽₂^(j+1))³`, then count.

> ⚠ **WITHDRAWN by §44.** The rule was extracted, and it is `K = −1 − Bp_T`: the curvature is
> *minus the base-level triangle sign*. So "extract the rule, then count" is the same statement as
> §17.2, not a route to it. §44 replaces this recommendation.

**§17.2 is not proven, and this rung does not attempt it.** (III) is still reduced; (d) and V1
untouched.


## §44 — The local rule, extracted: `K = −1 − Bp_T`, and the chain closes on itself (`W47`, 2026-08-04)

§43.3 proposed extracting the curvature's local rule and then counting. The rule is now extracted,
and it withdraws that recommendation.

### 44.1 The rule

At the base level `n = j+2` — the only level where `K_j` is defined as the raw per-triple table
(`cls = 1`) — the label `W = 2^j` *is* the top bit, so §42 applies: `A_σ = −μ(l)μ(y)`. Hence, per
triangle,

```
ε_T = ∏ (−μ(a)μ(b)) · ∏ Bp = (−1)³ · μ(a)²μ(b)²μ(c)² · Bp_T = − Bp_T
```

and therefore, since `A_T = ε_T·Bp_T`,

> **`K = A_T − Bp_T = −1 − Bp_T`** — the curvature class is `−2` **exactly on the POSITIVE triangles
> of the base-level `y = 0` graph**, and `0` on the negative ones.

Measured at the base level `j = 3, 4, 5, 6`: **0 exceptions** on both identities.

⚠ The derivation needs `W` to be the *top bit*. A first pass tested `(n,W) = (7,8), (8,8), (8,16)`,
none of which is the top-bit case, and the identity failed on 20–40% of triangles there — correctly,
because §42 does not apply. The identity is a base-level statement.

### 44.2 What this derives — §38's structure was not independent

Three of §38's four measured facts follow immediately:

| §38 fact | now |
|---|---|
| `K_j` takes only the values `{0, −2}` | **DERIVED**: `K = −1 − Bp_T` with `Bp_T = ±1` |
| the support sits in the non-degenerate triples | **DERIVED**: those are exactly the edge-triples |
| `K_j` is fully symmetric (cyclic + swap) | **DERIVED**: triangle signs are |
| level-independence of the whole tensor | still measured |

### 44.3 What this costs — the chain is circular

`K = −2` on positive triangles means

> `#flipped classes = #positive triangles of the base-level graph = (T + tr(A³)(1))/2`.

Verified as an exact arithmetic identity at `j = 3…9`: `(T + t3(1))/2 = 864·[j,3]₂` at every one.
So the finite clause is *literally* "the base-level `y = 0` graph has `864·[j,3]₂` positive
triangles" — which is §17.2 rearranged.

> **The chain closes on itself:** §17.2 ⟺ the finite clause ⟺ the base-level positive-triangle count
> ⟺ §17.2. There is no reduction left inside it.

That is a better description of the open state than the four-item list §42.2 gave: **the chain is one
statement, not four**, and it is a single signed-triangle count for which §43 measured that no
bounded window exists. Anything that closes it has to come from outside this chain — the `σ`
recursion itself, or the classical `q`-analogue reading of `[j,3]₂`, not from further rearrangement.

> ✅ **§45 did exactly that.** The `σ` recursion is §18.1, `Llo = 1` is low at every level, and
> §17.2 is the *solution* of that recursion with the proven `t2` closed form as inhomogeneity. The
> single statement of this section is therefore not open on its own — it inherits from §18.1.

**Three self-deflations today, and this is the third:** §31's route (false target), §38's ladder
(one cubic, not eleven confirmations), and now §43.3's recommendation (circular). All three were
rearrangements of something already in the file.

## §45 — §17.2 dissolves: it is the SOLUTION of §18.1's recursion (`W48`, 2026-08-04)

§44 left the chain closed on itself, with §17.2 as its single open statement, and said anything
that moves it must come from outside — "the `σ` recursion itself". The `σ` recursion *is* §18.1, and
it dissolves §17.2 in one step. The connection was available since `W25` and nobody made it.

### 45.1 The observation

> **`Llo = 1` is a LOW label at every level** (`1 < 2^(n−2)` for `n ≥ 3`), so §18.1's low-branch
> recursion applies to the `y = 0` class **directly**:
>
> `t3(n,1) = 8·t3(n−1,1) + 24·t2(n−1,1) − 12(2^(n−1) − 4)`
>
> and `t2(n,1) = (2^(n−1)−2)(2^(n−1)−4)` is the **proven** closed form (§3/§9, `Ncnt_closed`).

That is a first-order linear recursion with a known inhomogeneity. Solving it:

```
with q = 2^(n−1) and f(q) = (2/7)(q−2)(q−4)(q−15),
  8·f(q/2) + 24·(q/2−2)(q/2−4) − 12(q−4)
    = (q−4)·[ (2/7)(q−8)(q−30) + 6(q−8) − 12 ]
    = (q−4)·(2q² − 34q + 60)/7
    = (2/7)(q−4)(q−2)(q−15)  =  f(q)
```

Verified as an exact identity of polynomials in `q` (coefficients `−240/7, 28, −6, 2/7` on both
sides) and numerically: the recursion reproduces `t3(n,1)` at `n = 5…11`, and the closed form matches
the measured value at `n = 4…11` including the base `t3(4,1) = −48`.

> **So §17.2 is not an independent measured statement.** It is §18.1 + the proven `t2` closed form +
> one finite base case at `n = 4`.

### 45.2 What the ledger looks like now

Everything in the (III) chain rests on **§18.1**, and §34 already took §18.1 apart:

| ingredient | status |
|---|---|
| `tr(B³) = 8·t3′` | **ALGEBRA** (`tr(J₂³) = 8`) |
| `3·tr(B²E) = 24·t2′` | **DERIVED** from the block identity, the two sign lemmas and `Σ_a (A²)[a,a⊕W] = −tr(A²)` |
| the block identity `A = J₂⊗A′ + E`, and `E`'s four families | ✅ **THEOREM ∀n** — `Asig_block` (§47) |
| the two constant signs `+1` (matching), `−1` (coset) | ✅ **THEOREMS ∀n** — Tier 40 |
| `Σ_a (A²)[a, a⊕W] = −tr(A²)` | ✅ **PROVEN ∀n as a SUM** — `cosetSum_eq` (Tier 42), denotation cross-checked against the builder at `n = 5…9` |
| `tr(BE²) = 0`, `tr(E³) = −24(h−2)` | MEASURED (`n = 6…11`) |
| the `y = 0` base case `t3(4,1) = −48` | finite check on the sedenions |
| `Qgen'_pow2_eq`; `Asig_pow2_top` | **THEOREMS ∀n** |

**This is a strictly better position than §44's.** §43 measured that the absolute triangle sign has
no bounded window — but none of the items above is that. They are statements about the block
decomposition and about `E`, a graph whose **shape is label-independent** with `12(h−2)` edges. The
obstruction §43 found does not apply to them.

> **(III)'s entire chain now reduces to §18.1, and §18.1 to the six measured lines above.** The next
> target is the cheapest of them, and `tr(E³) = −24(h−2)` is the natural one: `E` is explicit, its
> four families are named, and two of their signs are already pinned.

**§18.1 is still not proven, so (III) is still reduced, not proven.** (d) and V1 untouched.

## §46 — `tr(E³) = −24(h−2)` is DERIVED: `E` is two hubs, a matching and a coset (`W49`, 2026-08-04)

§45 named `tr(E³)` the cheapest of the six lines §18.1 rests on. It is no longer a measured line: it
follows from `E`'s structure by counting.

### 46.1 What `E` is

Measured over **every** low label at `n = 6, 7, 8` (109 labels), **0 exceptions on each clause**:

| | |
|---|---|
| **S1** | `W` is **isolated** in `E`; the two **hubs** `h` and `W+h` are each adjacent to everything except `W` and the other hub |
| **S2** | the only other edges are the **matching** `{a, a+h}` with sign `+1` and the **coset** `{a, (a⊕W)+h}` with sign `−1`, for `a ∉ {0, W}` |
| **S3** | the hub signs satisfy `s(X, a+h) = − s(X, a)` and `s(X, (a⊕W)+h) = + s(X, a)`, for both hubs `X` |

So `E` has exactly `6(h−2)` edges in six families of `h−2` each — matching, coset, and two families
per hub — which is §34's count, now with the families identified rather than merely counted.

### 46.2 The count, and it is elementary

There are no low–low and no high–high edges, and the two hubs are **not** adjacent to each other.
So every triangle has exactly one hub, one low vertex and one high vertex, and the low–high edge must
be a matching or a coset edge. Hence

```
#triangles = (2 hubs) × (h−2 low vertices a ∉ {0,W}) × (matching or coset) = 4(h−2)
```

and each is negative, by **S3** against **S2**:

```
matching:  s(X, a+h)·s(X, a)·s(a, a+h)      = (−s)·s·(+1) = −1
coset:     s(X, (a⊕W)+h)·s(X, a)·s(a, ·)    = ( s)·s·(−1) = −1
```

Therefore `tr(E³) = 6 · 4(h−2) · (−1) = **−24(h−2)**`, which is §34's constant — **derived, not
fitted**, and it explains why the constant is label-independent: `E`'s *shape* is.

### 46.3 What is left, and it is one relation

The hub rows are not really about `E`: the blow-up `B` has **zero rows** at `h` (outside its index
range) and at `W+h` (`A′`'s row at `W` is zero), so on those rows `E = A`. Hence S1 and S3 are
statements about `A_σ` itself:

- `A_σ`'s row at `W` vanishes — that is **`Asig_isolated`, a theorem ∀n**;
- S3's second relation follows from the first plus **`A1`** (`Asig (l ⊕ L_lo) y = − Asig l y`,
  a theorem ∀n), because `(a+h) ⊕ W = (a⊕W)+h` when `W < h`;
- what remains genuinely new is **one relation**:

> **`A_σ(X, y + h) = − A_σ(X, y)` for `X ∈ {h, W+h}`** — the hub rows are antisymmetric under adding
> the top bit of the vertex range.
>
> ✅ **PROVEN ∀n** — `Asig_hub0` / `Asig_hubL` (Tier 38), kernel-clean. Four `cdSigma` evaluations
> through `P1_red`/`P3_red`: `P1` **and** `P3` both flip, `antisym` supplies the sign where the
> arguments swap, and `resB` cannot see the flip because `P1_symm`/`P3_symm` kill its first two
> clauses and the third compares `−P1` with `−P3`.

Plus S1's adjacency for the two hubs.

> ✅ **BOTH ARE NOW THEOREMS ∀n.** The relation is `Asig_hub0`/`Asig_hubL` (Tier 38); S1 is
> `resB_hub0_low`, `resB_hubL_low` and `resB_hub_hub`. For a **low** vertex the resonance reduces to
> a single identity — the four value lemmas give
> `P1(2^n,y) = −σ(L,y⊕L)`, `P3(2^n,y) = σ(L,y)`, `P1(L+2^n,y) = −σ(L,y)`, `P3(L+2^n,y) = σ(L,y⊕L)`,
> so `P1 = P3` on both hub rows **iff** `σ(L,y) = −σ(L, y⊕L)`, which is **`A4_sub'`**, the swapped
> form already proven in the file. For a **high** vertex it follows from the hub relation. And the
> two hubs miss each other because there `P1 = 1` while `P3 = −1`, both by `sigma_self`.

**So §46's derivation of `tr(E³) = −24(h−2)` now rests only on theorems plus §34's matching/coset
signs.** The `E`-internal content of §18.1 is closed.

> **Net effect on §45's ledger:** `tr(E³) = −24(h−2)` and `tr(BE²) = 0` are no longer independent
> measured lines; the first reduces to S1 + one hub relation, with the matching/coset signs already
> on the list. The open content of §18.1 is now the **block identity** and **the hub row**.

**Not proven in Lean yet.** §18.1, and with it (III), is still reduced, not proven.


## §47 — The block identity is a theorem, and it IS `E`'s family list (`W50`, 2026-08-04)

`Asig_block`, kernel-clean, ∀n:

```lean
Asig_block (k a b W e f : Nat) : e = 0 ∨ e = 1 → f = 0 ∨ f = 1 → BlkStd k a b W →
    Asig (a + e·2^(k+1)) (b + f·2^(k+1)) W (k+1) = Asig a b W k
```

with `BlkStd`: `a, b, W < 2^(k+1)`; `a, b ≠ 0`; `a ≠ W`, `b ≠ W`; **`a ≠ b`**; **`a ≠ b ⊕ W`**.

This is §34's `C1`, the foundation everything in §18.1 rests on, measured there over 38M entries.

### 47.1 The proof

`P1` and `P3` are computed on each block through `P1_red`/`P3_red` and the branch reductions:

| block | `P1` and `P3` via | closed by |
|---|---|---|
| `(0,0)` | `R_ll` twice | nothing — no side condition beyond the ranges |
| `(0,1)` | `R_lu` + `R_ul` | `antisym` on `(b,a)` |
| `(1,1)` | `R_uu` twice | `antisym` twice |
| `(1,0)` | — | `P1_symm`/`P3_symm` from `(0,1)` |

`resB` then agrees because its first two clauses are the symmetry lemmas at each level and the third
compares the two values just shown equal.

### 47.2 The observation worth keeping

> **The hypotheses `antisym` demands are exactly the lines `E` keeps.** `a ≠ b` is §34's *matching*
> family and `a ≠ b ⊕ W` its *coset* family.

So the block identity and `E`'s four families are **one computation seen from two sides** — which is
why §34's measurement found `E` supported precisely there, and why the matching and coset signs were
the two constants it could pin. The `(0,0)` block needs no exclusion at all, which is why it was the
one block §34 measured as exact.

### 47.3 The ledger for §18.1

| ingredient | status |
|---|---|
| `tr(B³) = 8·t3′` | ALGEBRA |
| the block identity | ✅ **THEOREM** (§47) |
| the hub row and hub adjacency | ✅ **THEOREMS** (Tier 38) |
| `tr(E³) = −24(h−2)` | ✅ **DERIVED** (§46) from the above + the matching/coset signs |
| the matching sign `+1` and the coset sign `−1` | ✅ **THEOREMS ∀n** — `Asig_matching` / `Asig_coset` (Tier 40) |
| `Σ_a (A²)[a, a⊕W] = −tr(A²)` | ✅ **PROVEN ∀n as a SUM** — `cosetSum_eq` (Tier 42) |
| `3·tr(B²E) = 24·t2′`, `tr(BE²) = 0` | the first DERIVED from the above; the second MEASURED |

**No measured line is left in §18.1's structural inputs.** The two `E` signs are `Asig_matching` and
`Asig_coset` (Tier 40); the coset 2-path identity's content is `Asig_coset_step` (Tier 41), which is
`A1` composed with `Asig_symm`. So §46's derivation of `tr(E³) = −24(h−2)` rests entirely on theorems,
and each of §34's four terms now has a proven structural basis.

⚠ **What is still NOT formalised, and it is bookkeeping rather than content: three of the four
summations.** Tier 42 carries `Σ_a (A²)[a, a⊕W] = −tr(A²)` all the way (`cosetSum_eq`), with the
denotation cross-checked against the builder — a true theorem about the wrong sum would be worthless,
so `degSum == tr(A²)` and `cosetSum == §34's S(W)` were verified at `n = 5…9` over all 491 labels.
Tier 43 then **defines the blow-up indexing** (`blow m A x y = A (x % m) (y % m)`, which is §34's
`J₂⊗A′`) and carries the algebraic term: `tri3 (m+m) (blow m A) = 8·tri3 m A`, with new
`Finset`-free infrastructure (`sumLtI_shift`, `sumLtI_double`). No property of `A_σ` enters — the
`8 = 2³` is the blow-up's, which is exactly why §34 called this term ALGEBRA.

Tier 44 then supplies the lemma Tier 43 named as missing: `sumLtI_swap` (`Σ_a Σ_b = Σ_b Σ_a`, plain
induction), `sumLtI3_cyc` (`Σ F a b c = Σ F b c a`, two swaps) and `tri3_cyc` — the three cyclic
rotations of a triple matrix product have the same triple sum. With `f = g = B`, `h = E` that
identifies the three `BBE` terms of `tr((B+E)³)`; with `f = B`, `g = h = E`, the three `BEE` terms.
**The `8 → 4` collapse is therefore available.**

**Two of §34's four sums remain uncarried:** `3·tr(B²E) = 24·t2′` and `tr(BE²) = 0`. Tier 45 supplies
their first move — `Esig_vanishes`, which discards the generic part (`BlkStd → E = 0`, one direction
only; the values on the families are Tiers 38/40 and are *not* assembled into a characterisation).

> ⚠⚠ **WITHDRAWN — see §49.** The paragraph below reasons from the density of `E`'s hub rows to the
> shape of the argument, without checking whether those rows can reach the OUTER indices of
> `tr(BE²)`. They cannot: `B` is zero on the rows *and* columns of all three special vertices, so a
> hub can only ever be the MIDDLE index, where the hub relation makes its collapsed contribution
> vanish outright. `tr(BE²) = 0` is a support argument after all. Kept in place, uncorrected, as the
> record of the misdiagnosis.
>
> ⚠ **Why the assembly did not close, and it is a finding rather than an excuse.** `E`'s **hub rows
> are dense** — each hub touches ≈ `2(h−2)` vertices — so `tr(BE²)` has `O(h²)` nonzero terms and its
> vanishing is a **cancellation, not a sparsity argument**. Every earlier tier closed by restricting
> to a handful of indices and finishing with `sumLtI_single`; that move does not bite here. The two
> missing mechanical steps are splitting a `sumLtI` range by a **predicate** rather than an interval,
> and reparameterising a sparse family as a sum over its index — neither hard alone — but `tr(BE²)`
> needs the cancellation on top of both.
>
> ✅ **Both mechanics are now proven** (Tiers 46–47): `sumLtI_split_pred` / `sumLtI_of_support` /
> `sumLtI_of_cosupport` cut a range by membership, and `sumLtI_eq_at` / `sumLtI_eq_at2` collapse a
> sum onto one or two support points — the shape the matching and coset families give, since for a
> fixed low vertex both partners are determined. ~~The general injective reindexing is still not
> proven~~ — **it is now proven, Tier 48, see §48 below** — and more to the point **neither mechanic
> reaches the dense hub rows.** The blocker for
> `tr(BE²)` is unchanged and is not a missing tool.

So the honest position: every *pointwise* and *structural* fact §18.1 rests on is a theorem, the
infrastructure (`sumLtI_shift`/`_double`/`_swap`/`_add`, `tri3_cyc`) is in place, one of the four sums
is carried and a second has its gate — and the last two need an argument of a kind this file has not
yet had to make.

**§18.1 is not yet proven, so (III) is still reduced, not proven.** (d) and V1 untouched.


---

## §48 — reindexing along an injection (Tier 48): the lemma §47 should not have skipped

`5e0a008df6`. Two theorems in `formal/lean4/SounioZDFiberAntisym.lean`, both `[propext, Quot.sound]`,
full-file build green:

```lean
theorem sumLtI_peel : ∀ (n k : Nat) (f : Nat → Int), k < n →
    sumLtI n f = f k + sumLtI n (fun i => if i = k then 0 else f i)

theorem sumLtI_reindex : ∀ (m n : Nat) (i : Nat → Nat) (f : Nat → Int),
    (∀ a, a < m → i a < n) →                              -- ι lands in the range
    (∀ a b, a < m → b < m → i a = i b → a = b) →           -- ι injective
    (∀ x, x < n → (∀ a, a < m → i a ≠ x) → f x = 0) →      -- f supported on the image
    sumLtI n f = sumLtI m (fun a => f (i a))
```

### §48.1 — this is a self-correction, and the correction is the point

§47 wrote that the general reindexing was "not proven — without `Finset` it needs index deletion from
a `sumLtI` range, which is awkward", and scoped it out. **That was a wrong call rather than a real
obstruction.** `sumLtI_peel` *is* the index deletion, it is a short induction on the range, and with
it the reindexing goes by induction on `m`, stripping the image one point at a time. The whole thing
compiled first try.

The general shape of the error is worth recording because it is cheap to repeat: **"awkward without
X" is a prediction about a proof I had not attempted, and I filed it as if it were a finding.** The
file's own precedent (`sumLt_lowMap` at `:5240`, and every `sumLtI_*` lemma in Tiers 43–47) already
showed that this file does index surgery without `Finset` routinely.

### §48.2 — what it does not do

**It does not move `tr(BE²)`.** There is no sparse family to reindex on the hub rows — they are
dense, `≈2(h−2)` entries each — so the blocker stated in §47 stands verbatim: the vanishing is a
**cancellation**, and no reindexing, splitting or collapsing lemma produces one. What Tier 48 removes
is the *tool gap the earlier tier claimed*, not the mathematical one.

Reviewed per M1 before commit: grok-4.5 `[OK]` on both statements; Z.AI truncated at 81 diff lines,
consistent with the measured ~50-line threshold for that provider.

**§18.1 is still not proven, so (III) is still reduced, not proven.** (d) and V1 untouched.


---

## §49 — `tr(BE²) = 0`: §47's diagnosis was about the wrong index (`W43`, 2026-08-05)

§47 filed the blocker like this:

> `E`'s **hub rows are dense** — each hub touches ≈ `2(h−2)` vertices — so `tr(BE²)` has `O(h²)`
> nonzero terms and its vanishing is a **cancellation, not a sparsity argument**.

The premise is true. The conclusion does not follow, and the reason is an index the diagnosis never
looked at.

### 49.1 The collapse

`B = J₂⊗A′` is zero on the **rows and the columns** of all three special vertices: row `W` of `A′` is
the isolated vertex (`Asig_isolated_row`), and `h` lies outside `B`'s index range entirely. So in

```
tr(BE²) = Σ_{u,v,w} B[u,v] · E[v,w] · E[w,u]
```

`u` and `v` are both non-special and only the **middle** index `w` is free. Summing over `u` and `v`
first, and using that `B` acts on its two arguments through the `2×2` block, the whole trace becomes
one quadratic form per vertex:

> **`tr(BE²) = Σ_w yᵂ_w ᵀ A′ yᵂ_w`, where `y_w(a) = E[w,a] + E[w,a+h]` for `a ∈ [1,h)`.**

A dense hub row never appears as an outer index; as a middle index it contributes exactly **one**
vector `y_w`. Density is irrelevant to the count.

### 49.2 Every form vanishes, and each for a theorem already in the file

| `w` | `y_w` | reason |
|---|---|---|
| `W` | `0` | row `W` of `A` is zero — `Asig_isolated_row` |
| `h` | `0` | `A[h, y+h] = −A[h, y]` — **`Asig_hub0`** (Tier 38) |
| `W+h` | `0` | `A[W+h, y+h] = −A[W+h, y]` — **`Asig_hubL`** (Tier 38) |
| `b + δh`, generic | `e_b − e_{b⊕W}` off the index `W` | `Asig_matching` (`+1`), `Asig_coset` (`−1`), `Asig_block` (the rest cancels against `B`), `resB_coset` (the within-block coset entry is `0`) |

**The hub case is the punchline.** The hub relation is precisely a *block-flip antisymmetry* of the
hub row, so collapsing the block adds a number to its own negative: `y_hub = 0` **outright**, not
after a cancellation between distinct terms. The two theorems that kill the dense rows were proven in
Tier 38, before the obstruction was even written down.

For a generic `w` the form is
`A′(b,b) − 2A′(b, b⊕W) + A′(b⊕W, b⊕W) = 0` — zero diagonal (`Asig_diag`) and the fact that a vertex is
never adjacent to its coset partner (`resB_coset`), which is exactly *why* `E` has a coset family in
the first place.

So `tr(BE²) = 0` **is** a support argument — at the collapsed index, not the ambient one.

### 49.3 Measured

`scripts/research/zd_v1_trBE2_probe.py 6 7 8 9 10`, every low label at each level (15/31/63/127/255):
**0 violations** on all four checks — the collapse identity, `y = 0` on the three special vertices
together with the three separate reasons, the `e_b − e_{b⊕W}` shape on all `129540` generic rows at
`n = 10`, and `A′`'s zero diagonal plus coset non-adjacency.

### 49.4 Status — honest

**`tr(BE²) = 0` is NOT yet proven in Lean.** What §49 changes is the *kind* of gap: every pointwise
ingredient is already a kernel-clean `∀n` theorem, and what remains is the **sum bookkeeping** — the
collapse of §49.1 — for which Tiers 43–48 supply the tools (`sumLtI_swap`, `sumLtI_add`,
`sumLtI_split_pred`, `sumLtI_reindex`). It is no longer "an argument of a kind this file has not yet
had to make".

**Correction to §47, stated plainly:** the sentence quoted at the top is withdrawn. It reasoned from
the density of a row to the shape of the argument without checking whether that row could reach the
outer indices — it cannot, because `B` kills it on both sides. This is the same failure shape as
Tier 47's "awkward without `Finset`" (§48.1): a property of the object was read off as a property of
the proof, without attempting the proof.

### 49.5 — the same lemma closes the OTHER remaining sum, and it is one lemma, not two

The collapsed row `y_w` does not depend on `δ` — the two blocks give the same vector. So the `2×2`
block sum of `E` is `Z(b,a) = 2[(a=b) − (a=b⊕W)]`, and since `B² = 2·J₂⊗A′²`,

```
tr(B²E) = 2 Σ_{a,b} (A′²)(a,b)·Z(b,a) = 4[t2′ − S(W)] = 8·t2′
```

through the already-proven `S(W) = −t2′` (`cosetSum_eq`, Tier 42). So **§34's last two sums are ONE
lemma apart, not two**, and §34.3's `24 = 3 × (4+4)` is exactly the two support points of the
collapsed row. Measured: `0` violations on both `Z`'s closed form and `tr(B²E) = 8·t2′` at
`n = 6,7,8`, every low label (`scripts/research/zd_v1_yrow_probe.py`).

### 49.6 — a gap the §49 draft did not see: the index `0`

Lean's sum ranges are `[0, 2^(n+1))`, one element **wider** than the contract's vertex set
`[1, 2^(n+1))`, and Tier 43's `blow m A x y = A (x % m) (y % m)` sends the hub `2^(k+1)` to `0`. So
every denotation claim in this arc — "this `sumLtI` **is** `t3′`", `blow = B` — silently needs row and
column `0` of `A_σ` to be zero. **`Asig_isolated` does not cover it: that theorem requires `l ≠ 0`.**
This is the "true theorem about the wrong sum" hazard, and it was live.

It is closed, in the good direction: the resonance predicate **fails at `l = 0` for every column, and
at `y = 0` for every row**.
Verified first against a Lean-faithful re-implementation of `cdSigma`/`P1`/`P3`/`resB`/`Asig` (not the
fast builder, which never constructs the index at all) over every label at `n = 2…5`, then proven —
`resB_zero_row` / `Asig_zero_row`, Tier 49. My own first hand-derivation of the diagonal entry got it
wrong (`Asig 0 0 = −1`, from `cdSigma L L = +1`); the true value is `−1` and the entry is `0`. The
measurement caught it, which is the reason to run one.

### 49.7 — what Tier 49 puts in Lean, and what it does not

`resB_zero_row`, `resB_zero_col`, `Asig_zero_row`, `Asig_zero_col`, `yrow_gen` — the **pointwise**
inputs.  (The row alone is not enough: index `0` occurs in all three positions `u,v,w` of the
collapse, and `E = A − blow` at index `0` needs the column too.  `Asig_symm` cannot supply it — it
requires both indices nonzero, which is exactly the excluded boundary — so both come from the same
fact, that the two orders of `P1` disagree at index `0`, which is `resB`'s FIRST clause and therefore
reads the same in either order.)  The assembly's tool is `sumLtI_eq_at2`, whose hypothesis is
summand-vanishing rather than vector-support: at the index `W` the zero comes from `A′`'s null
column, not from `y_w`.  `sumLtI_of_cosupport` is not needed either. The assembly (§49.1's
collapse) is still not formalised; the route is per-`w`: split the `u,v` range into four blocks with
`sumLtI_shift`/`_double` (Tier 43), then collapse the inner sum onto `{b, b⊕W}` with `sumLtI_eq_at2`
(Tier 47) — the summand vanishes at the index `W` through `A′`'s null column, not through `y`.
`sumLtI_reindex` (Tier 48) is **not** needed here; §49.4's tool list named it, and that was the long
way round.

### 49.8 — the assembly is DONE: `tr(BE²) = 0` is a theorem ∀n (Tier 50)

`0bd1e19121`. `trBE2_zero`, build green, no `sorry`, axioms `[propext, Classical.choice, Quot.sound]`
(the lane's `Asig`-level baseline; `quad_vanish` alone is `[propext, Quot.sound]`).

```lean
theorem trBE2_zero (W k : Nat) (hW : W < 2^(k+1)) (hW0 : W ≠ 0) :
    sumLtI (2^(k+1) + 2^(k+1)) (fun a => sumLtI (2^(k+1) + 2^(k+1)) (fun b =>
      sumLtI (2^(k+1) + 2^(k+1)) (fun c =>
        blow (2^(k+1)) (fun u v => Asig u v W k) a b
          * (Esig W k b c * Esig W k c a)))) = 0
```

Route: `sumLtI3_cyc` puts the free vertex outermost → `Esig_symm` (new, via a new `Asig_symm_full`
extending `Asig_symm` to the boundary `0`/`L_lo`, where both entries vanish) puts `w` first in both
`E` factors → `blowsum_r`/`sumLtI_shift_inv` collapse both of `B`'s indices onto the low half →
`quad_vanish`. `yr_support` does the case split on `w`.

**Two corrections to what §49 predicted.**

1. **§49.7 named `sumLtI_eq_at2` as the assembly's tool. It was not needed.** The summand vanishes
   *termwise* — `sumLtI_congr` + `sumLtI_zero` suffice. A two-point collapse would have worked but
   was strictly more work.
2. **A pointwise input was missing and neither §49 nor its review saw it: `Asig_isolated_diag`.**
   `Asig_isolated` and `Asig_isolated_row` both require `l ⊕ L_lo ≠ 0`, so neither reaches the corner
   `(L_lo, L_lo)`; there `P1 = 1` but `P3 = −1`, so `resB` fails and the entry is `0`. The assembly
   needs the isolated row null on the *whole* low range, corner included.

**The denotation was CHECKED, not assumed — in TWO steps, and the first was nearly skipped.**

- **C-1, the bridge.** The Lean `Asig` is a *transcription*; §34's whole ledger is stated about the
  builder `A_sig_fast`. Comparing the theorem's sum to another sum computed from the *same*
  transcription would have proved only that the padding is inert, not that either object is the
  contract's matrix. So: `Asig x y W m` vs `A_sig_fast(m+2, W)[x−1, y−1]` — **283 606 entries over
  `m = 2,3,4,5`, every label, 0 mismatches.** (`n = m+2` because the Lean index range is
  `[0, 2^(m+1))` and the builder's vertex set is `[1, 2^(n−1))`.) This is the
  `zd_annihilation_is_orthogonality_probe.py` pattern — cross-check the transcription before
  building on it.
- **C-2, the padding.** Lean's ranges are `[0, 2^(k+2))`, one element wider than the vertex set. The
  theorem's own sum over `[0,N)` vs the contract's over `[1,N)`, `k = 1,2,3`, every label: equal, and
  both `0`.

Both are `scripts/research/zd_v1_yrow_probe.py`, run before anything else in that file.

⚠ **The M1 review DEGRADED on this tier** — grok-4.5 returned unterminated chain-of-thought with no
verdict on both the 424-line diff and a 45-line statements-only extract; what it emitted flagged no
defect and independently re-derived the `p, q` choice. The denotation measurement is the substitute
evidence. Single provider, disclosed here and in the commit.

### 49.8b — a free strengthening of §34.2

§34.2 wrote `tr(A³) = tr(B³) + 3tr(B²E) + 3tr(BE²) + tr(E³)` and justified the `3×` coefficients with
"`B` and `E` are symmetric" — in prose. Both halves of that are now theorems: `Esig_symm` (Tier 50)
gives `E`'s symmetry ∀n, `B`'s follows from `Asig_symm_full` through `blow`, and `tri3_cyc` (Tier 44)
is what turns symmetry into the collapse of the six mixed terms into `3 + 3`. The expansion §18.1
rests on is therefore no longer prose at any point.

### 49.9 — what §34 still owes

| term | status |
|---|---|
| `tr(B³) = 8·t3′` | ✅ PROVEN ∀n (`tri3_Asig_blow`, Tier 43) |
| `3·tr(B²E) = 24·t2′` | ✅ **PROVEN ∀n, in exactly this form** — `trB2E_three` (Tier 51b), over `trB2E_eq` (Tier 51) |
| `3·tr(BE²) = 0` | ✅ **PROVEN ∀n (`trBE2_zero`, Tier 50)** |
| `tr(E³) = −24(h−2)` | ✅ **PROVEN ∀n (`trE3`, Tier 53)** |

**§18.1 is still not proven, so (III) is still reduced, not proven.** (d) and V1 untouched.

---

## §50 — `tr(B²E) = 8·t2′` (Tier 51): three of §34's four terms are now theorems

`e1eac7c886`. Build green **first try** in the real file; no `sorry`; axioms
`[propext, Classical.choice, Quot.sound]` (`yrow_esig` alone is `[propext, Quot.sound]`).

```lean
theorem trB2E_split : Σ_{a,b,c < 2^(k+2)} B a b * (B b c * Esig c a) = 4 * (t2′ − S(W))
theorem trB2E_eq    : ...                                            = 8 * degSum W k
```

Same shape as Tier 50, but with a **value** rather than a zero, so it has to *land on* Tier 42's
guarded sums instead of merely vanishing.

- `P2s_full` — summing `B²`'s middle index over the full range **doubles** the low-half two-path
  matrix. That `2` is the blow-up's, like Tier 43's `8`.
- `blk_eq` — after both outer indices collapse, what is left is `E`'s `2×2` block sum, and it is
  `2·(e_a − e_{a⊕W})`: **the same collapsed row as Tier 50, doubled by the two blocks.** This is
  §49.5's prediction, now a theorem: one lemma settles both mixed sums.
- `c_step` — here the two support points are **not** discarded. `sumLtI_eq_at2` reads them off, and
  they are the diagonal and the coset entry of `P2s`. **That is the tool §49.7 wrongly predicted for
  `tr(BE²)`; it is the right tool here.** So §49.7's error was not "the tool is useless" but "the
  tool belongs to the other term".
- `T2_eq`/`S_eq` — the bridge. Tier 42's sums carry the guards `a ≠ 0 ∧ a ⊕ W ≠ 0`; dropping them is
  legitimate exactly because those rows and columns of `A′` are null. **`Asig_zero_row`/`_col`
  (Tier 49b) and `Asig_isolated_diag` (Tier 50) are what make this step legal** — both were added
  because an earlier tier needed them for something else, and both pay for themselves here.
- `cosetSum_eq` (`S(W) = −t2′`, Tier 42) turns `4·(t2′ − S)` into `8·t2′`.

**Denotation measured against the CONTRACT's builder, not just the transcription:** the theorem's sum
over `[0,N)`, the contract's over `[1,N)`, and `8·tr(A′²)` computed from `A_sig_fast` all agree at
`k = 1,2,3`, every label.

**Method note that paid off immediately.** Tier 50 cost three build cycles because the scratch
axiomatized `yrow_gen` with its LHS *expanded* while the file states it about the **defined** `yrow`
— so `omega` saw an opaque atom and silently ignored the hypothesis. Tier 51's scratch copied every
imported statement **verbatim**, and the port compiled first try.

⚠ The M1 review was again **partial** (grok-4.5, no terminating verdict). What it produced derived
`B² = [[2A², 2A²], [2A², 2A²]]` independently and concluded "`trB2E_split` is perfectly correct"; its
single doubt — can row `0` of `A′` really be null? — came from context I failed to give it: index `0`
is not a vertex of the fiber graph, and `W` is the isolated vertex.

### §50.2 — the coefficient `3` is a theorem too (Tier 51b, `c9f56307e4`)

§34.2 justified the `3×` in prose and §49.8b noted the ingredients were theorems — but **no theorem
in the file stated the collapsed form.** Tier 51 proved `tr(B²E) = 8·t2′`; the file said nothing
about `3·tr(B²E)`. Now it does:

```lean
theorem trB2E_three : T(B,B,E) + T(B,E,B) + T(E,B,B) = 24 * degSum W k
```

Two applications of `tri3_cyc` (Tier 44) make the three mixed `BBE` positions of `tr((B+E)³)` equal;
`trB2E_eq` gives each `= 8·degSum`. No new mathematics — it closes the gap between what the file
proves and what §34's table claims, which is the kind of gap this lane has been caught by before.

### §50.1 — what is left of §18.1

**Only `tr(E³) = −24(h−2)`.** It is a statement about a label-independent graph on `12(h−2)` edges,
and unlike the three carried terms it involves no `B` at all — so none of the blow-up collapse
machinery applies to it. That is the next target, and it is the last one before §18.1 itself.

**§18.1 is still not proven** — three of its four terms are, the fourth is measured. (III) is still
reduced; (d) and V1 untouched.

---

## §51 — `tr(E³) = −24(h−2)`: the constant is a TRIANGLE COUNT, and the sign is `A1` (`W44`, 2026-08-05)

§34's fourth term, and the only one with no `B` in it — so none of the blow-up machinery that
carried the other three applies. Measured over every low label at `n = 5,6,7`, **0 violations on
every line** (`scripts/research/zd_v1_trE3_probe.py`):

### 51.1 The generic subgraph has no triangles

On the generic vertices `(a,δ)`, `a ∈ [1,h)∖{W}`, `E` has **exactly two** edges: the matching
partner `(a,1−δ)` with sign `+1`, and the coset partner `(a⊕W,1−δ)` with sign `−1`. That is
2-regular, and the component of `a` is the 4-cycle

```
(a,0) — (a,1) — (a⊕W,0) — (a⊕W,1) — (a,0)
```

so it contains no triangle.

### 51.2 There is no 2-hub or 3-hub triangle either

Row `W` of `E` is identically zero, and the three special vertices `W`, `W+h`, `h` are **pairwise
non-adjacent** (`resB_hub_hub`, Tier 38, is exactly `E(h, W+h) = 0`). So there are only **two** real
hubs, and **every triangle of `E` uses exactly one of them.** Measured: the `0`-hub, `2`-hub and
`3`-hub parts of `tr(E³)` are each exactly `0` at every level and label.

### 51.3 Each hub sees every generic, and every hub-edge triangle has sign `−1`

Each hub has degree exactly `2(h−2)` — it is adjacent to **all** generic vertices (the S1 section,
"the hub rows are FULL"). So each of the `2(h−2)` generic edges closes a triangle with each of the
two hubs: `4(h−2)` triangles.

Every one of them has sign product `−1`, and **for two different reasons**. Writing `s = E(H,·)`:

| edge | product | why |
|---|---|---|
| matching | `s(a)·(+1)·(−s(a)) = −1` | the hub row flips under the **block flip** — `Asig_hub0`/`Asig_hubL` (Tier 38) |
| coset | `s(a)·(−1)·(−s(a⊕W)) = −1` | the hub row also flips under **`a ↦ a⊕W`** — **`A1`** |

> **★ `A1` is this file's own headline lemma** — `Asig (l ⊕ L_lo) y = − Asig l y`, the fiber
> antisymmetry the file is named after — **and none of §34's other three terms had needed it.**
> `tr(B³)` is pure blow-up algebra, and `tr(BE²)`/`tr(B²E)` run on the collapsed row. The last term
> is the one that reaches for the lemma the whole file was built to prove.

### 51.4 The count

`2` hubs × `2(h−2)` generic edges = `4(h−2)` triangles, each contributing `6` ordered terms (three
rotations × two orientations, `E` symmetric) of `−1`:

> **`tr(E³) = 6 · 4(h−2) · (−1) = −24(h−2)`** — so the `24` is `6 × 4`, not a fitted constant.

### 51.5 Route to Lean, and the one new ingredient

The clean formal route is not the raw triangle enumeration but a two-point collapse. For a hub `H`
and a generic `b`, the only `c` with both `E(b,c) ≠ 0` and `E(c,H) ≠ 0` are `b`'s two generic
neighbours (the hubs are excluded because they are non-adjacent to `H` and to themselves), so

> **`(E²)(b,H) = E(b,mat b)·E(mat b,H) + E(b,cos b)·E(cos b,H) = (+1)(−s(b)) + (−1)(+s(b)) = −2·E(H,b)`**

— a `sumLtI_eq_at2` collapse, the tool that is already in the file. Then

`Σ_{b,c} E(H,b)E(b,c)E(c,H) = Σ_b E(H,b)·(−2·E(H,b)) = −2·Σ_b E(H,b)² = −2·2(h−2) = −4(h−2)`,

and the two hubs plus the three cyclic positions give `3 × 2 × (−4(h−2)) = −24(h−2)`.

**The one genuinely new ingredient is a DEGREE COUNT:** `Σ_{b<2h} E(H,b)² = 2(h−2)`. Every earlier
tier evaluated sums whose summands vanished or collapsed onto named points; this one has to *count*
a support. It is `sumLtI_peel` twice (Tier 48 — the lemma §47 nearly skipped) against
`Σ_{i<m} 1 = m`, on each half of the range.

**`tr(E³)` is NOT yet proven in Lean.** §34's other three terms are. (III) is still reduced; (d) and
V1 untouched.

### §51.6 — what is in Lean (Tiers 52, 52b, 52c) and what is not

| piece | status |
|---|---|
| `HubOf` + `hubOf_hub0` / `hubOf_hubL` — the two hubs | ✅ Tier 52 |
| `E2_hub_gen`, `E2_hub` — `(E²)(b,H) = −2·E(H,b)` | ✅ Tier 52 / 52c |
| `hub_deg` — the degree count `Σ_b E(H,b)² = 2(h−2)` | ✅ Tier 52b |
| `Esig_gen_gen`, `Esig_gen_supp` — the support discharge | ✅ Tier 52c |
| `trE3_hub` — the hub-position part, `−4(h−2)` per hub | ✅ Tier 52c |
| **the classification + cyclic split → `tr(E³)`** | ❌ **NOT PROVEN** |

**What is left, precisely.** Write `T(a,b,c) = E(a,b)E(b,c)E(c,a)` and `χ(x) = [x is a hub]`. The
missing step is the pointwise identity

```
T(a,b,c) = (χ a + χ b + χ c) · T(a,b,c)
```

— i.e. **every nonzero term has exactly one hub index.** Its `≥ 2 hubs` half is immediate: any two
hub indices among `a,b,c` are adjacent in the cycle, and hub–hub entries vanish. Its `0 hubs` half is
§51.1's 4-cycle argument and needs `Esig_gen_supp` twice, which now exists. With that identity,
`sumLtI_add` splits the trace into three pieces, `sumLtI3_cyc` identifies them, a two-point collapse
evaluates the first as `trE3_hub(2^(k+1)) + trE3_hub(W+2^(k+1)) = −8(h−2)`, and the total is
`3 × (−8(h−2)) = −24(h−2)`.

⚠ **A wording correction inside Tier 52c.** Its lemma is stated as "vanishes outside SIX named
columns" — the matching partner, the coset partner, the two hubs, and `0` and `W`. The row's true
support is the first four; `0` and `W` are *excluded by the lemma* rather than covered by it, and are
discharged directly inside `E2_hub`. The docstring originally said "exactly four nonzeros", which is
true of the object but not of the theorem; fixed.

---

## §52 — `tr(E³) = −24(h−2)` (Tier 53): §34's LAST term, and §18.1's four summands are ALL theorems

`trE3`, build green, no `sorry`, axioms `[propext, Classical.choice, Quot.sound]` (`sumLtI3_add`
alone is choice-free).

```lean
theorem trE3 (W k : Nat) (hW : W < 2^(k+1)) (hW0 : W ≠ 0) :
    Σ_{a,b,c < 2^(k+2)} Esig W k a b * (Esig W k b c * Esig W k c a)
      = -24 * (((2^(k+1) : Nat) : Int) - 2)
```

### 52.1 — the classification is SHORTER than §51.5 predicted, and the prediction erred the other way

§51.5 said the `0 hubs` half would be "§51.1's 4-cycle argument and needs `Esig_gen_supp` twice". It
needs neither. A nonzero entry between two **generic** vertices forces **different blocks** — within
a block `E` is either the diagonal or the within-block coset entry, both zero, which is a one-line
corollary of `Esig_gen_gen` (`Esig_same_block`). So walking `a → b → c` flips the block **twice** and
lands back in `a`'s block, whence `E(c,a) = 0`. **Two flips are the whole contradiction**; the
4-cycle never has to be spelled out, and no low-index arithmetic appears.

> This is the first prediction in this arc that erred on the side of **too hard**. §47 and §49.7 both
> erred the other way — a property of the object read as a property of the proof. Worth noting that
> the failure mode is not one-directional: what they share is that I recorded a guess about a proof I
> had not attempted.

`≥ 2` hubs is immediate: two hub indices among `a,b,c` are adjacent in the cycle, and hub–hub entries
vanish. (The argument does need the vertices to avoid `0` and `W` as well — derived inside
`tri_one_hub` from the null rows, not assumed.)

### 52.2 — the chain

```
tr(E³) = Σ (χa+χb+χc)·T                        tri_one_hub
       = 3 · Σ χa·T                             sumLtI3_add ×2, sumLtI3_cyc ×2
       = 3 · (F(2^(k+1)) + F(W+2^(k+1)))        sumLtI_eq_at2 on the two hubs
       = 3 · (−8(h−2)) = −24(h−2)               trE3_hub (Tier 52c)
```

The `24` is `3 × 8` and the `8` is `2 hubs × 4` — matching §51.4's `6 × 4` count from the other
direction (`6` ordered terms per triangle × `4(h−2)` triangles).

### 52.3 — denotation, measured against the CONTRACT

The theorem's own sum over `[0,N)`, the contract's over `[1,N)`, and `tr(E³)` computed from the
builder `A_sig_fast` all agree with `−24(h−2)` at `k = 1,2,3`, every label. `h = 2^(k+1)` because the
Lean level `k+1` is the builder's level `n = k+3`; and `−12(2^(n−1)−4) = −24(h−2)`, so this is §34's
constant in §34's own form.

### 52.4 — §18.1's ledger is CLOSED

| term | status |
|---|---|
| `tr(B³) = 8·t3′` | ✅ Tier 43 |
| `3·tr(B²E) = 24·t2′` | ✅ Tiers 51 / 51b |
| `3·tr(BE²) = 0` | ✅ Tier 50 |
| `tr(E³) = −24(h−2)` | ✅ **Tier 53** |

**All four summands of §34's expansion are now theorems ∀n.** What §18.1 still needs to be a theorem
is the *expansion itself* — that `tr(A³) = tr(B³) + 3tr(B²E) + 3tr(BE²) + tr(E³)` for `A = B + E` at
the level of the file's `sumLtI`, which is the multinomial expansion of a triple sum and has not been
written. Everything it would multiply out into is proven. (III) is still reduced; (d) and V1 untouched.

---

## §53 — §18.1 IS A THEOREM ∀n (Tier 54)

```lean
theorem section_18_1 (W k : Nat) (hW : W < 2^(k+1)) (hW0 : W ≠ 0) :
    tri3 (2^(k+1)+2^(k+1)) (fun x y => Asig x y W (k+1))
      = 8 * tri3 (2^(k+1)) (fun x y => Asig x y W k)
        + 24 * degSum W k
        - 24 * (((2^(k+1) : Nat) : Int) - 2)
```

Build green, no `sorry`, **no `axiom` anywhere in the file**; axioms
`[propext, Classical.choice, Quot.sound]` (`tri3_expand` alone is choice-free).

That is §18.1's low-branch recursion, in §18.1's own form: `h = 2^(k+1)` and
`−24(h−2) = −12(2^(n−1)−4)` under `n = k+3`.

### 53.1 — the last step was the smallest

`A = B + E` is *definitional* (`Esig := Asig − blow`), the cube expands into eight words by three
linearities of the mixed triple sum (`tri3m_add1/2/3`), and the four groups are the four theorems:

| group | value | tier |
|---|---|---|
| `BBB` | `8·t3′` | 43 |
| `BBE + BEB + EBB` | `24·t2′` | 51b |
| `BEE + EBE + EEB` | `0` | 54, over 50 |
| `EEE` | `−24(h−2)` | 53 |

No property of `A_σ` enters the expansion itself — it is `Int` algebra, and would hold for any
`A = B + E`. All the content is in the four values.

**Not circular.** Substituting `Asig = blow + Esig` is rewriting by the definition of `Esig`; the
four values were established without reference to the expansion.

### 53.2 — denotation, measured against the CONTRACT

At `k = 1,2,3`, every label: the theorem's own `t3`, `t3′` and `degSum` (computed from the
Lean-faithful transcription over `[0,·)`) each equal the builder's, **and** the recursion
`t3 = 8·t3′ + 24·t2′ − 24(h−2)` holds. So the Lean statement is §18.1 about the contract's matrices,
not about a paraphrase.

### 53.3 — what this closes, and what it does not

**§18.1 was the load-bearing MEASURED premise of §33.5(B)** — the `8^(n−j)` scaling of the (III)
deviation. It is now a theorem ∀n, so §33.5(B) no longer rests on an unexplained integer identity at
any point.

**(III) is still reduced, not proven.** §32–§33 reduce it to the deviation law, and §33.5(B) explains
the scaling; what remains open is the law's own base content, plus (d) and V1. What changed here is
that the recursion underneath is no longer measured.

---

## §54 — the deviation law: (B) is now UNCONDITIONAL, and the obvious route to the base is REFUTED

### 54.1 — cashing in §18.1

§33.5(B) flagged itself: *"(B) is conditional. §18.1's low-branch recursion is itself MEASURED, so
(B) does not make the scaling a theorem... Proving §18.1 is a load-bearing target, not a nicety."*
§18.1 is now `section_18_1` (Tier 54), so the flag comes down. Tier 55:

```lean
theorem deviation_descent (W V k : Nat) (hW hW0 hV hV0)
    (hfib : degSum W k = degSum V k) :
    tri3 (2^(k+1)+2^(k+1)) A_W − tri3 (2^(k+1)+2^(k+1)) A_V
      = 8 * (tri3 (2^(k+1)) A′_W − tri3 (2^(k+1)) A′_V)
```

Two instances of §18.1 subtracted: the constant `−24(h−2)` cancels because it does not depend on the
label, and the `24·t2′` terms cancel against each other exactly when the two labels share their
level-`k` `tr(A²)`. Nothing is left over. **The `8^(n−j)` scaling is a theorem.**

**★ VACUITY CHECKED — the reviewer's question, and it needed answering.** If `degSum` were
injective in the LABEL, `hfib` would force `W = V` and `deviation_descent` would be the empty
statement `0 = 0`. It is not: at levels `k = 2,3,4,5` the `2^(k+1)−1` labels fall into
`1, 2, 4, 8` fibres of sizes `7 … 10`, **every one of them with more than one label**. So the
hypothesis is satisfiable everywhere it is used, and the theorem has content. (Tier 36 proved
`tr(A²)` injective in **`g`**, not in `W` — precisely the distinction that makes this non-vacuous.)

⚠ `hfib` is a **hypothesis**, not discharged. `degSum` fibre-constancy — that `tr(A²)` depends only
on `g` — is §30's, and Tier 36 proved the *other* direction (injectivity, `Ncnt_inj_g`) about a
different counting object. Measured here at `n = 6,7,8`, 0 violations, but not proven.

### 54.2 — the base case, and a route CLOSED

What the law still needs is the value at the level where `W` stops being low, `n = top(W) + 2` —
where `W` is **high**. Measured (`scripts/research/zd_v1_deviation_base_probe.py`, `n = 6,7,8`,
every label): the base value matches `−27·8^(n−j)·[j,3]₂` (or `0` on odd parity) with **0
violations**, and depends only on `(lsb, top, popcount(g) mod 2)` — 0 violations.

> ⚠ **My first version of that last check keyed on `(lsb, top)` ALONE and reported violations.** The
> check was wrong, not the law: the parity half is §33.5(A), already a theorem. Recorded because a
> mis-specified check that *fails* is more dangerous than one that passes.

**★ REFUTED: the naive high-branch twin of §18.1.** The base lives on the high branch, where no
recursion is on the books, so the obvious move is to look for one of the same shape. There is none:
for `W` with top bit `2^(n−2)` and `W_lo = W − 2^(n−2)`,

```
t3(n,W) − 8·t3(n−1,W_lo) − 24·t2(n−1,W_lo)
```

takes **7 distinct values at `n = 7` and 15 at `n = 8`**, not one. And the residual is **not** a
function of `g(W)` either (2 of 4 `g`-classes at `n = 7` carry two values), nor of `g(W_lo)` (same
counts). So the base case will not come from a `§18.1`-shaped high-branch recursion, and it will not
come from one repaired by a `g`-dependent correction of that shape.

**The reviewer sharpened this further, and the sharpening is right.** The refutation is not limited
to the coefficients `(8, 24)`: the number of residual values follows `2^(n−4)−1`, i.e. it **grows
with `n`**, so no *constant* pair `(a,b)` can absorb it either — a two-term constant-coefficient
recurrence of any coefficients is dead, not just this one. What is not excluded is a **third
invariant** alongside `t3′` and `t2′`, or coefficients indexed by a fibre parameter; the Mersenne
shape `2^(n−4)−1` suggests the residual classes are indexed by something like `𝔽₂^(n−4)∖{0}`, which
is where a third term would have to be keyed. Recorded as the live lead, not as a result.

This is the second route closed on the base case, after §33.3's edge-level bijection. Both are
recorded so the next rung does not restart from them.

**(III) is still reduced, not proven.** What is open is now exactly: **the base value for a general
seam at its own top-bit level**. Its `n`-dependence is a theorem; its `g`-independence is not.

### 54.3 — the third invariant: the natural candidates are REFUTED, and the target is reformulated

`Ncnt_hi` (Tier 31) is the tr(A²) high recursion, and it reads

```
N(m+2, W+e) = 4e² − 6e − 2 − 4·N(m+1, W),      e = 2^(m+1)
```

— **a minus**. The low branch is `+4` (`Ncnt_low`), and `+4 = tr(J₂²)`, `+8 = tr(J₂³)`: a Kronecker
blow-up. `−4` is not `tr(K²)` for any real sign matrix `K`, so the high branch **complements**
rather than blows up. A complement's triangle count needs a third moment beyond the edge count, so
that is where the third invariant should be.

**It is not there.** Measured at `n = 7,8,9` over every high label:

| candidate | result |
|---|---|
| exact affine `t3(n,W) = a·t3′ + b·t2′ + c`, **any** `(a,b,c)` | **none** (exact rational search) |
| the same plus `Σ deg²`, or `Σ deg³`, or a signed 2-path count | **none** |
| `1ᵀA1`, `1ᵀA²1`, `1ᵀA³1`, third moment of the signed row sums | **label-independent constants** — they carry no information at all |

That last row is the informative one: the corrections a complement formula would need are *identically
constant* on this family, so the analogy that motivated the search does not transfer. The
two-term refutation is now direct (an exact search found no coefficients), not only the growth
argument of §54.2.

**What the measurements did establish, and it reframes the target.** `t3(n,·)` restricted to the
high labels is **exactly a function of `(g, lsb)`** — 0 splits at `n = 7,8,9`. Since the fibre
reference `8g+1` is low, its own `g`-dependence descends by §18.1. So the open content of the
deviation law is not "a missing recursion" but:

> **`t3(n, ·)` is additively separable in `(g, j)` on the even-parity fibres:
> `t3(g, j) = f₀(g) + δ(n, j)`.**

That is a sharper statement than "the base value does not depend on `g`", and it is the shape any
proof would have to produce: not a recursion, a **separation of variables**.

⚠ **Scope:** a handful of candidate invariants were refuted, not all of them. "No third invariant
exists" is NOT claimed.

⚠ **I made the parity-keying mistake a SECOND time.** Grouping `t3(W) − t3(8g+1)` by `lsb` alone
showed 1/2/3 splits at `n = 7,8,9` and looked like a failure of separability; the splits are just the
two `popcount(g)` parities sharing an `lsb`. Same error as §54.2, in a fresh check. Recording the
repeat: when a check on this law reports splits, **look for the missing parity key before looking
for a finding.**

### 54.4 — §33.3's CANDIDATE MECHANISM IS REFUTED, in both of its forms

§33.3 recorded, as "a candidate mechanism for the open residual":

> `[j choose 3]₂` counts subspaces of the `j` bit positions **below** the seam — precisely the
> coordinates `g` forgets. So "`D` does not depend on `g`" stops being a brute fact and becomes: the
> triangle deficit is a count over independent triples strictly below the seam... **The proof to
> build is a sign-preserving bijection between the deficit triangles of `8y + 2^j` and those of
> `2^j`, given by `a ↦ a ⊕ 8y`.**

Both halves are now tested, and both fail.

**(i) The deficit is NOT classified by independence below the seam.** Split the signed triangle sum
by whether the three vertices' low `j` bits are linearly independent over `𝔽₂`, for a seam and for
its fibre reference, and take the difference. At `n = 8`, `j = 3`, the total is `−884736` for every
`g` — the law — but the split is not:

| `g` | independent class | dependent class |
|---|---|---|
| `0` | `−866304` | `−18432` |
| `6` | `−792576` | `−92160` |
| `10` | `−755712` | `−129024` |
| `12` | `−829440` | `−55296` |

**The `g`-dependence is present in each class and cancels only in the total.** So the deficit is not
a count over independent triples below the seam; that classification does not see the law.

**(ii) The bijection `a ↦ a ⊕ 8y` fails at TRIANGLE level too.** §33.3 refuted it at edge level and
explicitly left the triangle-level form open ("A bijection at the level of deficit *triangles* is not
excluded by this test"). It is now excluded: comparing the triple products of `A(n, 8(g+2^j))` and
`A(n, 2^j)` under `a ↦ a ⊕ 8g`, **69072 of 238328 triples mismatch at `n = 7`** (≈29%), and
320784–886608 of 2000376 at `n = 8` depending on `g`.

So the lane's recorded lead for the open residual is dead. Four routes are now closed: the
edge-level bijection (§33.3), the triangle-level bijection (here), the two-term high-branch
recursion for any coefficients (§54.2–54.3), and the complementation third invariant (§54.3).

**What survives, and is worth carrying:**

- `t3(n,·)` on the high labels is exactly a function of `(g, lsb)` — 0 splits.
- The law has a **local form**: for two seams in the SAME fibre with consecutive `lsb`,
  `t3(W_j) − t3(W_{j−1}) = δ(n,j) − δ(n,j−1)`, and `[j,3]₂`'s Pascal identity gives
  `δ(n,j) − δ(n,j−1) = −27·8^(n−j)·[j−1,2]₂`. 0 violations at `n = 6,7,8`. This compares two labels
  differing in one bit position rather than a seam against a Fano reference — a smaller object than
  anything §33 considered. ⚠ But the two matrices differ in thousands of entries (424–35592 of
  `N²`), so it is *not* a local perturbation of the graph.

⚠⚠ **THE PARITY WAS DROPPED BY HAND A THIRD TIME.** Re-deriving the prediction inline for the
consecutive-`j` check produced 1/2/3 violations at `n = 7,8,9` that looked like a refutation of
separability; the check had lost the `popcount(g)` parity. **Fixed structurally, not by resolve**:
the law is now one function, `dev_pred`, in `scripts/research/zd_v1_separability_probe.py`, and the
docstring says not to inline it again. Three occurrences in one session is a tooling problem, not an
attention problem.

### 54.5 — the local form attacked: it is TRUE and it does not localize

The fibre-`g` family is one parameter, and writing it that way is the clearest thing to come out of
this rung:

> **`W_j = 8g + 2^j`, `j = 0 … lsb(8g)−1`** — `j = 0,1,2` are Fano members, `j ≥ 3` are the seams,
> and the fibre reference `8g+1` is simply the `j = 0` member. The law is `D(W_j) = δ(n,j)` with
> `δ = 0` for `j < 3` because `[j,3]₂ = 0` there: **one statement, no case split.**

Consecutive members differ in exactly **two bits**. The local form
`t3(W_j) − t3(W_{j−1}) = −27·8^(n−j)·[j−1,2]₂` holds with 0 violations at `n = 6,7,8`. Attacked:

- **No locality.** Is `A_{W_j} − A_{W_{j−1}}` a function of the low `m` bits of the two vertices?
  **No, for every `m < n−1`** — i.e. for every non-vacuous `m`. The difference is global.
- **The perturbative split fails to isolate anything.** With `Δ = A_{W_j} − A_{W_{j−1}}`,
  `t3(W_j) − t3(W_{j−1}) = 3tr(A′²Δ) + 3tr(A′Δ²) + tr(Δ³)`. **All three pieces are `g`-dependent**,
  and each is of the same order as the total (e.g. `n = 8, j = 3`: `3tr(A′²Δ)` runs
  `−1335288, −522360, −144504, −879096` over `g = 0,6,10,12` while the total is `−884736`
  throughout).

**One regularity survived, and it is a strange one.** `nnz(A_{W_j} − A_{W_{j−1}})` depends only on
`(n, g)` — it is the *same for every `j`* in a fibre's family (`2160` for `g = 0` at `n = 7`, `2296`
for `g = 6`, `8432`/`9120`/`9224`/`8824` for `g = 0,6,10,12` at `n = 8`). So the two quantities have
**complementary dependences**: the size of the perturbation sees only `g`, the triangle count sees
only `j`.

### 54.6 — what five failed decompositions have in common

| decomposition | `g`-dependence |
|---|---|
| independence of low `j` bits (§54.4) | present in each class, cancels in the total |
| `a ↦ a ⊕ 8g` on triangles (§54.4) | ~29% of triples mismatch |
| high-branch two-term recursion (§54.2–3) | no coefficients exist |
| complementation third invariant (§54.3) | the candidate moments are label-constant |
| perturbative `3tr(A′²Δ) + 3tr(A′Δ²) + tr(Δ³)` (§54.5) | present in all three pieces |

**In every one of them the `g`-dependence is present at the fine level and cancels only in the
signed triangle count.** That is now a pattern across five independent attempts, not an accident of
one. It says the mechanism is unlikely to be a combinatorial bijection or a term-by-term
decomposition at all — the natural remaining reading is **spectral**: `tr(A³)` is a symmetric
function of the spectrum, and the lane's own prior-art work (§40–41) established that the
**switching class** is the spectrally meaningful datum. A route through switching equivalence is the
one thing this section has NOT tried, and it is what §35 said `k = 3` is the first moment able to
see.

⚠ **A range error caught in this rung, worth recording.** The first version of the local-form probe
took `j` up to `n−2` instead of `j < lsb(8g)`, which silently compares labels in DIFFERENT fibres
(`8·10 + 2^4 = 96 = 8·12` has `g = 8`, not `10`). It produced one anomalous row that briefly looked
like structure. Caught by recomputing `g(W)` from the label rather than trusting the loop bound.

**(III) is still reduced, not proven.**

## §55 — the SPECTRAL route: an exact halving valid for EVERY label

§54.6 said the remaining reading is spectral. It is, and the first thing it gives is a factorisation
better than §18.1's, because it does not care whether the label is low.

### 55.1 — the fold

`A1` — this file's headline lemma, `A_σ(l ⊕ W, y) = −A_σ(l, y)` — pairs the vertices `{l, l⊕W}`; the
isolated vertex is `W` itself. Choosing the representative with `W`'s **top** bit clear and deleting
that bit maps the representatives onto `[1, 2^(n−2))`, and the matrix becomes a Kronecker product:

```
A  =  M ⊗ K   (plus the isolated row/column),        K = [[1,−1],[−1,1]]
```

`K² = 2K`, so `tr(K²) = 4` and `tr(K³) = 8`, and therefore

> **`tr(A²) = 4·tr(M²)` and `tr(A³) = 8·tr(M³)` — for EVERY label.**

Measured: **0 violations at `n = 6,7,8`, all `2^(n−1)−1` labels each.** §18.1 is the low branch only;
this holds on the high branch too, which is exactly where the deviation's base case lives.

**So the ubiquitous `8` is `tr(K³)`, once.** It is *not* iterated folding: `M` is **nonsingular**
(rank `2^(n−2)−1`, full) and admits no second signed antisymmetry, so the fold is exactly one level
deep. The `8^(n−j)` in `δ` still comes from the §18.1 descent, which is a separate mechanism.

### 55.2 — what the spectra show

Along the one-parameter family `W_j = 8g + 2^j` of §54.5:

| `n` | `g` | `(j, #distinct eigenvalues of M, tr(M³))` |
|---|---|---|
| 7 | 0 | `(0,8,6510) (1,8,6510) (2,8,6510) (3,7,−7314) (4,5,−19410) (5,2,−26970)` |
| 7 | 6 | `(0,8,2310) (1,8,2310) (2,8,2310) (3,5,−11514)` |

- **`j = 0,1,2` are COSPECTRAL** — the Fano orbit, seen spectrally. That is §33.5(A) as a statement
  about eigenvalues rather than about `Φ`.
- **The spectrum simplifies monotonically as `j` grows**, collapsing to two distinct eigenvalues at
  the maximal seam.

### 55.3 — the maximal seam is `I − J`, and §33.5(C) falls out in one line

At `W = 2^(n−2)` (the maximal seam, `j = n−2`), **`M = I − J` exactly** — the complete graph with
every edge `−1` — verified at `n = 5,6,7,8,9`. Its spectrum is `1` with multiplicity `s−1` and
`1−s` once, `s = 2^(n−2)−1`, so `tr(M³) = (s−1) − (s−1)³` and

```
tr(A³)(n, 2^(n−2)) = 8·[(s−1) − (s−1)³] = −8(q−1)(q−2)(q−3),      q = 2^(n−2)
```

which is **§33.5(C) exactly**, obtained here in one line instead of via the `K_N`-minus-a-matching
count plus the antibalance theorem. The `8` that §33.5(C) attributed to "the doubling `N = 2(q−1)`"
is the same `tr(K³)`.

### 55.4 — status

**This does not prove the deviation law.** What it changes:

- the base case, which lives on the high branch where §18.1 says nothing, now has an exact
  factorisation of its own — `tr(A³) = 8·tr(M³)` on a matrix of half the size that is nonsingular;
- `δ`'s endpoint `j = n−2` is now completely explicit (`M = I − J`), and the previously derived base
  case is a one-line corollary rather than a separate argument;
- §33.5(A)'s `D = 0` half restates as *cospectrality* of `j = 0,1,2`.

What is open is unchanged in substance: the value at a general seam. But it is now a question about
the spectrum of an explicit nonsingular half-size matrix, and the two endpoints of the `j`-family
(`j ≤ 2` cospectral, `j = n−2` equal to `I − J`) are both settled.

**(III) is still reduced, not proven.**

## §56 — §33.5(C)'s BASE CASE IS A THEOREM (Tiers 57–58)

```lean
theorem tri3_pow2_value (n : Nat) (hn : n ≠ 0) :
    tri3 (2^(n+1)) (fun x y => Asig x y (2^n) n)
      = -8 * ((2^n : Int) − 1) * ((2^n : Int) − 2) * ((2^n : Int) − 3)
```

Build green, no `sorry`, no `axiom` in the file.

§33.5(C) derived this on paper from the `K_N`-minus-a-perfect-matching structure plus the
antibalance theorem, and called it "DERIVED". It is now proven, and by a different route — the fold.

### 56.1 — the one missing lemma was `P3_pow2_top`

`Asig_pow2_top` (Tier 37) carries `resB … = true` as a **hypothesis**, so on its own it says nothing
about *where* the graph has edges. Supplying that hypothesis needs `P3` at the maximal-seam label,
which the file did not have. Tier 57 adds it:

> `P3_pow2_top : P3 l y (2^n) n = μ(l)·μ(y)` — **with `l ≠ y`, which `P1_pow2_top` does not need.**
> On the diagonal `P1 = 1` but `P3 = −1`, and that disagreement is exactly `Asig_diag`. The extra
> hypothesis is the content, not a weakening.

With both, `resB_pow2_top` gives a FULL mask off the four excluded lines, and `Asig_pow2_rep` gives
`−1` on the representative box: **the fold of the maximal seam is `I − J`, proven.**

### 56.2 — the count

`tri3_kron` (Tier 56) reduces the trace to `8 ×` the box sum. The box summand is `(−1)³ = −1`
exactly on ordered triples of distinct nonzero representatives and `0` otherwise — zero rows at `0`,
`Asig_diag` whenever two indices coincide. Verified exhaustive: over `n = 1…4` there is **no**
distinct nonzero pair in the box with entry `≠ −1`.

The count itself is three nested instances of one pattern — a constant over a range minus a measured
excluded set (`sumLtI_const_excl` against `cnt1`/`cnt2`/`cnt3`) — giving
`(2^n−1)(2^n−2)(2^n−3)`.

Measured: `n = 1,2,3,4,5` → `0, −48, −1680, −21840, −215760`, all matching, including the degenerate
`n = 1` where no such triples exist and both sides are `0`.

### 56.3 — status

**This closes the `y = 0` base case only.** §33.5 already had it on paper; what is new is that it is
now a theorem, that it follows from the fold rather than from a bespoke count, and that the fold is
available at **every** label. The **general seam's** base case — the actual open content of the
deviation law — is untouched.

**(III) is still reduced, not proven.**

### 56.4 — the general seam: two facts, and a stop

Attacking the general seam's base case with the fold. **I did not crack it.** Two things are worth
keeping, and then a deliberate stop.

**(1) A seam and its Fano reference fold onto the SAME box.** The representative predicate
`a < a ⊕ W` is decided by the highest bit where `a` and `a ⊕ W` differ, which is `W`'s top bit. So
for **every** label with top bit `n` the representatives are `[0, 2^n)` — independent of the label's
low part. At the base level both the seam `W` and its fibre reference `8g+1` are high with the same
top bit, so

> `D(W) = 8 · ( t3(box_W) − t3(box_{W₀}) )` — **a difference of two triangle sums on one index set.**

That is a real simplification of the base case: before, the seam and the reference lived on
different vertex sets and the comparison was between objects of different shape. ~~(Measured; the Lean `rep_iff` is currently stated for `W = 2^n` only, and generalising it needs a
highest-differing-bit lemma about `Nat.xor` that the file does not have.)~~ — **⚠ WRONG COST
ESTIMATE, corrected in §57: the missing fact is two lines of `xor_pow_low` plus associativity, and
both this and the same-box fold are now theorems (`rep_iff_gen`, `tri3_fold_high`).**

**(2) The general high box is `I − J` plus a sparse correction.** Entries lie in `{−1, 0, +1}`.
~~and the `+1` entries number exactly `2(2^n − 2)` at `n = 3,4,5` for every low part tested — which
is §34's matching/coset family size.~~ **⚠ FALSE, corrected in §57.3: I had sampled only
`W_lo ∈ {1,2,3,5}`, which all sit in the `g = 0` fibre. Sweeping every `W_lo` gives 2 values at
`n = 4` and 8 at `n = 6`; `2(2^n−2)` is only the minimum.** The correction is *not* supported on the coset line `l ⊕ y = W_lo`
(tested: mixed), so the obvious identification is wrong.

⚠ **And the "stop" below was called one lemma too early** — see §57. The reason to stop was sound
(measurements without a proof plan); the specific claim that (1) could not be stated in Lean was
not, and it was again a guess about a proof I had not attempted. Fourth occurrence in this session.

**Stopping the MEASUREMENT SWEEP on purpose.** Six angles have now been tried on this base case — edge bijection,
triangle bijection, high-branch two-term recursion, complementation third invariant, perturbative
split, spectral fold — five closed and one (the fold) turned into the `y = 0` proof. What is
accumulating now is measurements without a proof plan, which is how this lane has previously spent
rungs for nothing. The next rung should start from (1) — the same-box reformulation — and should
have a target statement before it starts measuring.

**(III) is still reduced, not proven.** (d) and V1 untouched.

## §57 — the same-box fold is a theorem (Tier 59)

```lean
theorem rep_iff_gen (n W x) (hn : n ≠ 0) (2^n ≤ W) (W < 2^(n+1)) (x < 2^(n+1)) :
    (x < x ^^^ W) ↔ (x < 2^n)
theorem tri3_fold_high (W n) (hn : n ≠ 0) (2^n ≤ W) (W < 2^(n+1)) :
    tri3 (2^(n+1)) (Asig · · W n) = 8 * (the triple sum over [0, 2^n))
```

Build green, no `sorry`, no `axiom`. Denotation measured: `n = 2,3,4`, **every** high label, 0
violations.

**§56.4(1) is now stateable and stated.** Every label with top bit `2^n` folds onto the SAME box
`[0, 2^n)` — independent of the label's low part — so at the deviation law's base level a seam and
its fibre reference fold onto ONE index set and

> `D(W) = 8 · ( t3(box_W) − t3(box_{W₀}) )`.

`rep_iff` (Tier 58) is the special case `W = 2^n`.

### 57.1 — the cost estimate in §56.4 was wrong, and that is the fourth time

§56.4 said generalising `rep_iff` "needs a highest-differing-bit lemma about `Nat.xor` that the file
does not have", and stopped there. The fact needed is:

```
u ^^^ (v + 2^n) = (u ^^^ v) + 2^n        (low against shifted keeps the top bit)
(u + 2^n) ^^^ (v + 2^n) = u ^^^ v        (two shifted give a low)
```

— two lines each from `xor_pow_low` and associativity. **Fourth recorded cost estimate in this
session about a proof I had not attempted** (§47 "dense rows ⇒ cancellation", §49.7 "needs
`sumLtI_eq_at2`", §51.5 "needs the 4-cycle argument", and this). Three erred toward *too easy* and
two toward *too hard*; the direction is not the pattern — the guess is. The rule stands: **do not
record a difficulty you have not attempted.**

The decision to stop the measurement sweep was still right; what was wrong was bundling a
un-attempted-proof claim into it.

### 57.2 — the general box: a pattern that died at the next level

With `tri3_fold_high` the base case is a difference of two triangle sums on the box `[0, 2^n)`, so
the question is what the box of a general high label `W = 2^n + W_lo` looks like. Measured:

- entries lie in `{−1, 0, +1}`; relative to `I − J` the correction takes values `{0,1,2}`;
- grouping the off-diagonal entries by `x = l ⊕ y`, each `x` carries values from exactly one of
  three sets: `{1}` when `x = W_lo` (the coset line), or `{0,1}`, or `{−1,0}`. ~~— **⚠ the second
  and third classes are an `n = 3` artefact; see §57.3.** Only the coset-line half survives.~~

**At `n = 3` the involution `x ↦ x ⊕ W_lo` swaps the `{0,1}` and `{−1,0}` classes — perfectly, for
every `W_lo`.** That is a clean coset structure and it is exactly the shape a proof would want.

**It is false at `n = 4` (84 of 210 pairs) and `n = 5` (588 of 930).** The pattern is an artefact of
the smallest level. The entry is also not a function of `(l ⊕ y, ⟨l, W_lo⟩)` at any level tested.

⚠ **I was one measurement away from recording this as a finding.** The `n = 3` table is small enough
to read by eye and the swap is exact there; the check that killed it was running the same test at
`n = 4,5`. This is the mirror of the parity-keying mistakes earlier in this session — those were
mis-specified checks that FAILED and looked like refutations; this is a correctly-specified check
that PASSED at one level and looked like a discovery. **Both failure modes are cured by the same
discipline: never conclude from a single level.**

So the general box has no characterisation by the two natural candidates, and the base case remains
open. What is now solid, and is the place to start from: the base case is
`D(W) = 8·(t3(box_W) − t3(box_{W₀}))` on one index set (`tri3_fold_high`, Tier 59), with `box` known
exactly at `W_lo = 0` (`I − J`, Tier 57) and unclassified otherwise.

**(III) is still reduced, not proven.**

### 57.3 — the box at `n = 6`: two of my claims die, and the survivor is the whole problem

Swept every high label at `n = 3,4,5,6` (`scripts/research/zd_v1_general_box_probe.py`).

**Two earlier claims are FALSE and are struck above.**

1. §56.4(2): "the `+1` entries number exactly `2(2^n−2)` … for every low part tested". I had sampled
   `W_lo ∈ {1,2,3,5}` — all in the `g = 0` fibre. Sweeping everything: **2 values at `n = 4`, 4 at
   `n = 5`, 8 at `n = 6`.** `2(2^n−2)` is the minimum, attained on `g = 0`.
2. §57.2's class shape ("every other `x` carries `{0,1}` or `{−1,0}`") — 0 violations at `n = 3`,
   then **42 / 336 / 1938** at `n = 4,5,6`. The `x ↦ x ⊕ W_lo` swap I had already retracted; the
   *class shape itself* is equally an `n = 3` artefact. My retraction in §57.2 did not go far enough.

**What survives every level.** Entry values are exactly `{−1,0,+1}`, and **the coset line
`l ⊕ y = W_lo` carries exactly `{1}`** — `7/7`, `15/15`, `31/31`, `63/63` labels at `n = 3,4,5,6`.

**★ And the real finding: the box reproduces the whole phenomenon one level down.** Grouping the high
labels by their fibre (`g(2^n + W_lo) = 8 + g(W_lo)`, so the fibres of `W` are the fibres of `W_lo`):

| box invariant | constant on the fibre? |
|---|---|
| `#(+1)`, `#0`, `t2(box)` | **yes** — 0 splits at `n = 4,5,6` |
| **`t3(box)`** | **NO** — 1 of 4 fibres at `n = 5`, **2 of 8 at `n = 6`** |

That is exactly the lane's own signature — `tr(A²)` a fibre invariant, `tr(A³)` not — now visible on
an object of **half the size**. And the split is the law itself: **`8·(t3(box_W) − t3(box_ref))`
equals the deviation `δ`, 0 violations over all 31 labels at `n = 5` and all 63 at `n = 6`.**

**Status.** The base case is unchanged in substance, but it now lives on the box: prove that
`t3(box)`'s variation within a fibre is `δ/8`, on a `(2^n−1)`-square matrix whose entry values are
`{−1,0,+1}` and whose coset line is uniformly `+1`. Nothing smaller has been available before.

⚠ **Method note, third occurrence in two rungs.** §57.2's lesson was "never conclude from a single
level"; this rung shows the sampling version of the same error — §56.4(2) concluded from four values
of `W_lo` that happened to share a fibre. **Sweep the parameter, not a sample of it.**

**(III) is still reduced, not proven.**

### 57.4 — the box vs. the level below: `P3` is INVARIANT, `P1` FLIPS, and the graphs are edge-disjoint

Attacking the variation of `t3(box)` inside a fibre. The box of the high label `2^n + W_lo` sits on
`[1, 2^n)`, which is the level-`(n−1)` vertex set, so the first question is how it relates to the
level-`(n−1)` matrix of the **low** label `W_lo`. Measured at `n = 3,4,5`, every `(W_lo, l, y)`:

| | |
|---|---|
| `P3 l y (2^n+W_lo) n` vs `P3 l y W_lo (n−1)` | **identical**, 100% |
| `P1`, off the diagonal | **exactly negated**, 100% |
| `P1`, on the diagonal | `= 1` at both levels |
| the two masks `resB`, off the diagonal | **never both true** — `0/28830` at `n = 5` |

The last line follows from the first two in one line: `resB`'s third clause is `P1 = P3`; if `P3` is
unchanged and `P1` flips, then `P1 = P3` at one level forces `P1 = −P3` at the other, and both are
`±1`. **So the box and the level-below matrix are EDGE-DISJOINT graphs on the same vertex set.**

The `P1` flip has a paper derivation: `P1_red` turns `P1 l y (2^n+W_lo) n` into
`cdSigma l y n · cdSigma (l⊕W_lo) (y⊕W_lo) n` through `R_uu` (both `l⊕W` and `y⊕W` carry the top
bit), while `P1_red` at level `n−1` gives the same product with the **second factor's arguments in
the other order** — and `antisym` supplies the sign. **The `P3` invariance is derived too** (Tier 60
below): it is the same two lines with `R_lu`/`R_ul` in place of `R_ll`/`R_uu`, where the branch
reduction contributes one extra minus on each factor and the two cancel. When I first wrote this
section I recorded it as "measured, not derived" — that was a statement about what I had done, and
the derivation turned out to be the same size as `P1`'s.

**★ This is what `Ncnt_hi`'s `−4` was.** §54.3 read the minus as "the high branch COMPLEMENTS rather
than blows up", went looking for a complement's third moment, and found nothing — the candidate
moments were all label-constant. The complementation is real but it lives at the level of the
**support**, not of a moment: the high label's graph avoids every edge of the low label's. That
retro-explains both the `−4` and why the third-moment search was bound to fail.

**Status.** This does not yet give `t3(box)`'s variation. Edge-disjointness constrains the pair but
does not determine either triangle count, and the two graphs do not cover everything (they miss the
diagonal and, off it, the pairs where both masks are false). What it does is replace "the box is
unclassified" with a precise relation to an object one level down that the lane already understands.

### 57.5 — Tier 60: all three statements are now theorems ∀n

The two lemmas §57.4 named, plus the disjointness they were wanted for, are in
`SounioZDFiberAntisym.lean`, kernel-clean (`[propext, Classical.choice, Quot.sound]`):

| theorem | statement | hypotheses |
|---|---|---|
| `hi_shift` | `hi l (W + 2^(m+1)) (m+1) = hi l W m + 2^(m+2)` | `l, W < 2^(m+1)` |
| `P1_hi_lo` | `P1 l y (W + 2^(m+1)) (m+1) = − P1 l y W m` | `+ l ≠ y` |
| `P3_hi_lo` | `P3 l y (W + 2^(m+1)) (m+1) = P3 l y W m` | `+ l ≠ 0`, `y ≠ 0` |
| `resB_hi_lo_disjoint` | `¬(resB … (m+1) ∧ resB … m)` | `+ l ≠ y`, `l ≠ 0`, `y ≠ 0` |

`hi_shift` is the whole content: the `⊕` with the new label's top bit is a `+ 2^(m+2)` because
`l < 2^(m+1)` puts the two bits in disjoint positions, so both `hi` images move up one storey
together and the four branch reductions apply unchanged. The asymmetry between the two conclusions
is then just which branches each product lands on — `P1` on `(R_ll, R_uu)` (one `antisym`, one
minus), `P3` on `(R_lu, R_ul)` (two minuses that cancel). The disjointness needs no new work: clause
`P1 = P3` at both levels plus `P1_pm` gives `P1 = −P1` with `P1 = ±1`.

Dual-provider review `[OK]/[OK]` on the sign asymmetry and on both hypothesis sets. Measured before
asking, at the Lean levels `m = 2,3,4`: `P1_hi_lo` 0 violations in 36032 instances, `P3_hi_lo` 0 in
34744.

**What this does NOT give.** `t3(box)`'s variation inside the fibre. Edge-disjointness restricts the
pair of graphs without determining either triangle count. `Ncnt_hi`'s `−4` is now a theorem at the
level of the support; the moment is still open.

### 57.6 — ⚠ CORRECTION to §57.4, and what the disjointness actually buys

§57.4 wrote, of the two graphs, that "they miss the diagonal and, off it, the pairs where both masks
are false". **The second clause is FALSE.** Measured at `m = 2,3,4` over every label and every
off-diagonal pair of nonzero indices: the number of pairs where both masks are false is **0**. The
two graphs do not merely avoid each other, they **PARTITION** the edges.

The reason, now a theorem (Tier 61 `resB_hi_or_lo`): `resB`'s first two clauses are the A2_VACUITY
pair, and at level `m+1` they are unconditional — `P1_symm` needs `x ⊕ L ≠ 0`, and the high label's
`⊕` on a low index is `(x ⊕ W) + 2^(m+1)`, never `0`. So the high mask **is** its third clause,
which Tier 60 turns into `−P1_low = P3_low`, the exact negation of the low third clause (both are
`±1`). The single place the low mask fails for another reason is the low label's ISOLATED vertex
`l = W` — precisely where `P1_symm`'s hypothesis fails — and there `P1 = −P3` outright, so the high
mask carries that row and that column. That identity was already inside `Asig_isolated_row` and
`Asig_isolated`, used only to conclude the mask fails; Tier 61 extracts it as `P1_iso_row` /
`P1_iso_col`.

**The object §57.4 wanted.** Subtracting kills the mask:

| | |
|---|---|
| `Asig_hi_lo_diff` | `Asig l y W m − Asig l y (W+2^(m+1)) (m+1) = − P1 l y W m` |

`−P1` is symmetric (`P1_symm`), `±1` (`P1_pm`), zero on the diagonal — **a Seidel matrix**, of which
the two `Asig` are the two edge classes. That is the lane's own vocabulary: writing `P = −P1`,

    tr(P³) = 6·( C(2^(m+1)−1, 3) − 2·|Ω_W| )      0 violations at m = 2,3,4

where `Ω_W` is the **two-graph** of the label (the triples with `P_ij P_jk P_ki = −1`), so the
`P`-term is a switching-class invariant and the mask-dependence of the deviation lives entirely in
the mixed traces.

*Measured and unexplained, recorded so it is not mistaken later for a finding:* the within-fibre
deviations of `|Ω_W|` itself are `+168` at `m = 3` (`j = 3`) and `+1152 / +2328 / −1152` at `m = 4`
(`j = 3, 4, 3`). The `168` is the same integer as `|GL(3,2)|`, which is all over this lane; at
`m = 4` the `j = 3` values are `±1152`, not multiples of `168`, so the coincidence does **not**
extend. No claim is attached to these numbers.

**The four-term high-branch descent.** With `P = Alo − Ahi`, trilinearity gives, exactly,

    t3(box) = t3(Alo) − 3·tr(Alo² P) + 3·tr(Alo P²) − t3(P)

0 violations at `m = 2,3,4`. **This does not overturn §54.3.** What §54.3 refuted was a *two-term
affine* recursion in `(t3′, t2′)`; the extra terms here are mixed traces, not multiples of those two.

**Where the box sits.** Measured at `m = 3,4,5`, over every high label:

| | |
|---|---|
| `t3(box_W) − t3(box_{8g+1})` | `= −27·8^(n−j)·[j,3]₂` with `n = m+2`, `j = lsb W`, parity on `g(W+2^(m+1))` — 0 viol |
| `D(full level-(m+1) matrix)` | `= 8 · D(box)` — 0 viol / 63 at `m = 5`, with 3 nonzero instances |

So the box **is** the level-`n` object and the `8` is `tri3_kron`'s, now checked on general labels
and not only on the maximal seam. Before this, the factor was a theorem for the `2^n` label alone.

**The parity corollary — and why it is NOT a finding.** `g(W + 2^(m+1)) = g(W) + 2^(m−2)`
(0 violations), so `popcount`'s parity FLIPS: a low label and its box sit in fibres of opposite
parity, and therefore exactly one of the two deviations is zero (0 violations). Hence

    D_low + D_box = −27·8^(n−j)·[j,3]₂     with no parity case split

which looks like a cleaner law than either half. It is not new content: it is (A) — the parity half,
already a theorem — composed with the `g`-shift. Recorded as a corollary, per the lane's rule about
checking whether a pattern is forced by cheaper structure before calling it a fingerprint.

### 57.7 — Tier 62: the mask is an ARTEFACT

The partition has a consequence stronger than the difference identity. The mask holds exactly where
`P1 = P3` and fails exactly where `P1 = −P3` — the two are exhaustive because both products are
`±1`, and the isolated row/column (the only place the first clause can fail on its own) is a
`P1 = −P3` locus. Both cases are then the same formula:

| | |
|---|---|
| `Asig_no_mask` | `2 · Asig l y Llo n = − (P1 l y Llo n + P3 l y Llo n)` |

with hypotheses `l, y, Llo < 2^(n+1)`, `l ≠ 0`, `y ≠ 0`, `l ≠ y` — **`Llo ≠ 0` is not assumed**; it
is derived in the two branches that use it. (The M1 reviewer found that same hypothesis superfluous
in Tier 61's `resB_hi_or_lo`/`Asig_hi_lo_diff`, where the two providers disagreed; the argument that
it is derivable — `l ⊕ W = 0` gives `l = W`, and `l ≠ 0` gives `W ≠ 0` — is correct, so both were
tightened before committing.) Measured over every index including `0` and the diagonal: 0 failures
at `m = 2,3,4`.

`A_σ` is therefore a **polynomial in two `±1` matrices**, and `tri3` of it expands by trilinearity
into eight words in `(P1, P3)` with no `resB` anywhere:

    8 · tri3(A_σ) = − tri3(P1 + P3)        0 violations, m = 2,3, every label

**⚠ A corollary I wrote down and then refuted.** Substituting Tier 60's `P1_hi = −P1_lo`,
`P3_hi = P3_lo` into the box's copy gives `8·t3(box) = tri3(P1 − P3)` and hence the clean-looking
`4·(t3(Alo) + t3(box)) = −tri3(P3) − 3·tri3m(P3,P1,P1)`. **This is FALSE at every label of
`m = 2,3,4`.** Both substitutions carry hypotheses — `l ≠ y` for `P1_hi_lo`, `l,y ≠ 0` for both —
and measuring `(P1_hi + P3_hi) − (P3_lo − P1_lo)` locates the damage exactly there:

| level | off-diagonal | diagonal | touching index 0 |
|---|---|---|---|
| `m = 2` | **0** | 49 | 105 |
| `m = 3` | **0** | 225 | 465 |
| `m = 4` | **0** | 961 | 1953 |

`tri3` sums over both of those loci. The per-level identity is right; the level LINK needs that
correction, not computed here. Recorded because it was written into the Lean docstring before being
measured, and the measurement is what caught it. **A first version of this paragraph said the
damage was "exactly the diagonal"** — that was measured about `P1` alone; the index-0 row and column
differ too, because both `hi_lo` lemmas exclude the index 0.

### 57.8 — Tier 63: the diagonal correction, and the refuted corollary comes back exact

The correction is one line. `P1` is `1` on the **whole** diagonal — `P1_diag` off the isolated
corner, and at the corner the computation already inside `Asig_isolated_diag` — and `P3` is `−1`
there. So

| | | |
|---|---|---|
| `P1_diag_full` | `P1 l l Llo n = 1` | `l ≠ 0` only |
| `diag_sum_zero` | `P1 + P3 = 0` on the diagonal | ← the combination `A_σ` is built from |
| `diag_diff_two` | `P1 − P3 = 2` on the diagonal | ← the combination the level link needs |
| `P1_add_P3_zero_row` / `_col` | `P1 + P3 = 0` on the index-`0` line | |

**That asymmetry is the whole bug.** `2·A_σ = −(P1+P3)`, and the sum vanishes exactly where `A_σ`
does — the diagonal and the index-`0` line — so masking `P1+P3` is a no-op. The box's copy needs
`P1−P3`, which is `2` at each of the `2^(n+1)−1` diagonal entries and `±2` on the index-`0` line,
and `tri3` sums over both loci. All five statements are Lean theorems, kernel-clean.

With the mask put back, the corollary refuted in §57.7 is **exact**, 0 violations at `m = 2,3,4`,
every label — writing `X̃` for `X` zeroed on the diagonal and on the index-`0` line:

    8·t3(A_lo)  = − tri3(P1̃ + P3̃)        (the mask is a no-op here)
    8·t3(A_box) =   tri3(P1̃ − P3̃)        (the mask is load-bearing here)
    4·(t3(A_lo) + t3(A_box)) = − ( tri3(P3̃) + 3·tri3m(P3̃, P1̃, P1̃) )

**What that buys.** The deviation law now reads entirely inside level `m`, on two Seidel matrices:

    D[tri3(P3̃)] + 3·D[tri3m(P3̃,P1̃,P1̃)] = −4·(−27·8^(n−j)·[j,3]₂)     0 viol, m = 3,4,5

and measuring the two terms separately: **every label with `j ≥ 3` moves, every label with `j < 3`
does not, and off the maximal seam the two deviations are EQUAL**, each `= 27·8^(n−j)·[j,3]₂`. The
single exception at each level is `W = 2^m`, where they differ (`m = 3`: `2016` vs `1632`; `m = 4`:
`27936` vs `25248`; `m = 5`: `277920` vs `264480` — the identity still holds, the split does not).

So the law is now equivalent to a statement about **one** moment of **one** Seidel matrix,
`tri3(P3̃)` — i.e. about the two-graph of `P3` alone, with no `P1` and no mask. `tri3(P3̃)` takes
only `2^(m−2)` distinct values per level.

### 57.9 — Tier 64: `tri3(P3̃)`, and the label's TOP BIT is a SWITCHING

Attacking the one moment §57.8 reduced the law to. Two things came out, both with the lane's own
signed-graph vocabulary.

**(1) The maximal seam is the empty two-graph.** At `W = 2^m` the spectrum of `P3̃` is exactly
`{N−1 (×1), −1 (×N−1)}` — the spectrum of `J − I` — at `m = 2,3,4,5`. That is not a coincidence:
`P3_pow2_top` (Tier 57) already says `P3 l y (2^n) n = μ(l)·μ(y)`, a RANK-ONE sign matrix, so every
triple product is a square and `|Ω| = 0`. Hence

    tri3(P3̃)|_{W = 2^m} = N(N−1)(N−2),  N = 2^(m+1)−1

measured exactly: `210, 2730, 26970, 238266` at `m = 2,3,4,5`. (Verified rank-one on *all* distinct
nonzero triples, including those through the seam vertex, where `P3_pow2_top`'s hypotheses exclude
it: 0 bad triples.)

**(2) The label's top bit acts by SWITCHING — now a theorem.** `tri3(P3̃)` depends only on
`W mod 2^m` (0 violations, `m = 2,3,4,5`). The reason:

| | |
|---|---|
| `epsTop x m` | `= −1` iff `2^m < x` — the switching vector |
| `sigma_top_flip` | `cdSigma (c ⊕ 2^(k+1)) l = (−epsTop l)·cdSigma c l` |
| `P3_top_switch` | `P3 l y (W + 2^(k+1)) = epsTop l · epsTop y · P3 l y W` |

Switching is `S ↦ diag(ε)·S·diag(ε)`; it preserves every triple product `S_ab S_bc S_ca`, hence the
two-graph, hence `tri3`. `P3` is a product of two `cdSigma` factors (`P3_red`), each picking up
`−epsTop`, and the two minus signs cancel. Measured first: 0 violations in 139176 instances at
nonzero indices.

Two sharp boundaries, both measured and both reflected in the hypotheses:

* **the index-`0` line FAILS** the switching law (42/210/930/3906 entries at `m = 2..5`) — `epsTop 0`
  would have to be `−1` and it is `+1`. Tier 63 masks that line out of `tri3(P3̃)`, so the corollary
  survives; the theorem carries `l ≠ 0`, `y ≠ 0` for exactly this reason.
* **`P1` is NOT switched by the top bit** — not by `epsTop`, and not by any vector: at every label
  of `m = 2,3,4` some triple product of `P1(W)·P1(W+2^m)` is `−1`. The property is specific to `P3`.

### 57.10 — `D[tri3(P3̃)]`: the reference is a REGULAR two-graph, and the exception has a closed form

**The fibre reference is regular.** Switching `P3̃` so that vertex `1`'s row is all `+1` and reading
off the descendant graph at `W = 8g+1`:

| `m` | vertices | degree | edges |
|---|---|---|---|
| 3 | 14 = 2 isolated + 12 | 4 | 24 |
| 4 | 30 = 2 isolated + 28 | 12 | 168 |
| 5 | 62 = 2 isolated + 60 | 28 | 840 |

i.e. **regular of degree `2^m − 4` on `2^(m+1) − 4` vertices**, plus exactly two isolated ones, with
`E = (2^(m+1)−4)(2^m−4)/2`. (At `m = 4` that edge count is `168` — noted, not claimed: the lane's
`168`s have burned me before, and this one is a product of two binomials that happens to land there.)

**The reduction.** For a two-graph with descendant graph `G` on `N` vertices, elementary counting
gives `|Ω| = E·(N−2) − 2·Σ_v C(d_v,2) + 4·t(G)` with `t` the triangle count, so

    tri3(P3̃) = 6·C(N,3) − 12·|Ω|

turns `D[tri3(P3̃)]` into the deviation of three graph statistics. Measuring them separately shows
the expected thing and is a useful control: **`ΔE`, `Δpaths` and `Δtriangles` are all
root-dependent** (at `m = 5`, `W = 8` gives `ΔE = −204` while `W = 24` gives `ΔE = −432`) **while
`Δ|Ω|` is not** — both give `−9216`. Only the switching-class invariant is stable, which is exactly
what the theory demands.

**The maximal-seam exception, in closed form.** §57.9 left it open. The excess of the measured
deviation over the plain law is `288, 2016, 10080` at `m = 3,4,5` — that is `288·[m−1,2]₂`. So

    D[tri3(P3̃)]|_{W = 2^m} = 27·8²·[m,3]₂ + 288·[m−1,2]₂

**Confirmed out of sample at `m = 6`**, a level not used to find the pattern — three independent
predictions, all exact:

| prediction | value | |
|---|---|---|
| `tri3(P3̃)|_{W=2^m} = N(N−1)(N−2)`, `N = 127` | `2000250` | ✓ |
| `D` at `j = 3` `= 27·8^(n−3)` | `884736` | ✓ |
| `D` at the maximal seam `= 27·8²·[6,3]₂ + 288·[5,2]₂` | `2455200` | ✓ |

Equivalently, the `g = 0` reference itself has a closed form,
`tri3(P3̃)|_{W=1} = N(N−1)(N−2) − 1728·[m,3]₂ − 288·[m−1,2]₂`, checked at `m = 3,4,5,6`
(`714, −966, −39654, −454950`).

**Tier 65** makes the geometric half a theorem: `P3_pow2_coherent`, every triple product is `+1`
at the maximal seam — because `P3_pow2_top` makes `P3` rank-one and each product becomes a product
of three squares. **Scope, and it is narrower than "the two-graph is empty":** the theorem covers
only the triples that AVOID the seam vertex `2^n`, which `P3_pow2_top`'s hypotheses exclude. The
triples through `2^n` are coherent too, but that is measured (0 bad triples), not proved — the M1
reviewer flagged the overstatement and both the docstring and this line were corrected before
committing. The counting half (empty two-graph ⇒ `tri3 = N(N−1)(N−2)`) is the `sumLtI` argument of
Tier 58 and was not redone.

### 57.11 — `tri3(P3̃)` at EVERY fibre reference: a closed form, confirmed out of sample

§57.10 left this open: the `g ≠ 0` references had no formula (`−39654, 15642, −7398, 11034` at
`m = 5`). They do now, and the right coordinate is the WALSH basis.

Fix a level `m`, write `b = m−3`, `N = 2^(m+1)−1`. The references are `W = 8g+1`, and
`P3_top_switch` (Tier 64) makes `g` and `g + 2^b` switching-equivalent, so `g` runs over `[0, 2^b)`.
Expand `g ↦ tri3(P3̃)(8g+1)` in characters of `(ℤ/2)^b`:

    tri3(P3̃)(8g+1) = Σ_k w_k · (−1)^popcount(g ∧ k)

| | |
|---|---|
| **(1) support** | `w_k = 0` unless the set bits of `k` form a **contiguous block** of positions — only `b(b+1)/2` of the `2^b` characters survive |
| **(2) value** | for the block `[i, i+L−1]`: `w_k = −2304·(2^(i+1)−1)·8^(m−4−i) / 2^(L−1)` |
| **(3) mean** | `w_0 = N(N−1)(N−2) − 1728·[m,3]₂ − 288·[m−1,2]₂ − Σ_{k≠0} w_k` (§57.10's `g=0` form) |

Everything is fixed by the single-bit coefficients `s_i = −2304·(2^(i+1)−1)·8^(m−4−i)`; lengthening
a block by one bit **halves** its coefficient. The level-to-level rule is `s_i(m+1) = 8·s_i(m)`, with
one new coefficient appearing per level, `s_{m−4} = −2304·(2^(m−3)−1)`.

**Evidence, and the order it came in.** Discovered on `m = 4,5,6` (1, 3 and 6 nonzero characters).
Then, before extending the formula:

| test | prediction | result |
|---|---|---|
| `m = 7`, the seven characters already present at `m = 6` | `w_7[k] = 8·w_6[k]` | exact, 7/7 |
| `m = 7`, `k = 5 = 0b101` (not a block) | `w = 0` | exact |
| **`m = 8`, ALL 32 reference values**, none used to build the formula | full closed form | **0 mismatches / 32** |

The `m = 8` run is the real test: `N = 511`, thirty-two references, every value predicted before
being computed. Probe: `scripts/research/zd_v1_p3_twograph_probe.py`.

**What this is NOT.** A closed form for the REFERENCES, not a proof of the deviation law. The law
`D[tri3(P3̃)] = 27·8^(n−j)·[j,3]₂` compares `W = 8g+2^j` with `8g+1`; this settles only the second
term. And none of it is in Lean — the statement is a global count over `~N³` triples, not a
pointwise identity like the tiers above, so it is a different kind of formalisation job.

### 57.12 — attacking the contiguous-block law: two mechanisms REFUTED, one reformulation

**(a) "each triple's coherence is `±` a character of `g`."** If that held, the support would be the
set of characters realised by triples and the whole law would be a counting statement. **Refuted:**
`10464 / 39711` triples at `m = 5` and `139032 / 333375` at `m = 6` have a coherence vector that is
not `±` a character.

**(b) "the entries are characters and the block structure is inherited."** **Refuted twice over.**
Only `13164 / 16002` entries at `m = 6` are single characters at all (the rest have Walsh support of
size 3, 5 or 7), and — decisively — those that are **realise the non-block character**: `k = 5 =
0b101` occurs at `1624` entries. Block-ness is invisible at the entry level, so it cannot be
inherited from there.

**(c) What is true.** Splitting the triple sum at `m = 6` into the two classes of (a):

| `k` | | from character triples | from the rest | total |
|---|---|---|---|---|
| 3 | block | `0` | `−73728` | `−73728` |
| **5** | **NOT a block** | **`0`** | **`0`** | **`0`** |
| 6 | block | `0` | `−27648` | `−27648` |
| 7 | block | `0` | `−36864` | `−36864` |

The non-block coefficient vanishes **in each class separately** — it is not a conspiracy between
them, and any proof has to explain a cancellation that already holds inside each class. Note also
that `k = 3, 6, 7` get *nothing* from the character triples: their whole coefficient comes from the
triples that are not characters.

**The reformulation.** Writing `x_t = (−1)^(bit t of g)` and

    R_i = Σ_{L≥1} 2^{−(L−1)}·x_i x_{i+1} ⋯ x_{i+L−1}      equivalently   R_i = x_i·(1 + R_{i+1}/2)

the two halves of the law — interval support **and** the halving — are together *equivalent* to

    tri3(P3̃)(8g+1) = w_0 + Σ_i s_i·R_i(g)

(algebra, not measurement; checked at `m = 5,6,7,8`, 0 violations). So the whole `g`-dependence
enters through `b` nested dyadic quantities, one per bit position, each an affine function of the
binary fraction whose digits are the **prefix parities** of `g` from that position on. That is the
shape a proof has to produce, and the `2^{−(L−1)}` is a place value, not a coincidence.

### 57.13 — the IDENTIFICATION check: the two standard families are EXCLUDED

Consulted an outside reviewer on where the new knowledge in this lane is. Its first instruction was
a novelty check to run **before** spending another rung, on the grounds that the lane's memory
already records two label-drift near-misses and a firewall (the `168` freezing is Kirshtein 2012).
The check has two halves and **both came back negative**, which is the outcome that keeps the object
interesting rather than the one that deflates it.

**(1) Not a bilinear-form two-graph.** If `P3̃` were `(−1)^(B(l,y)+f(l)+f(y))` for an `F₂`-bilinear
`B` on the index space — the symplectic / quadratic-form two-graphs of Seidel, whose triple
invariants and Seidel spectra are classical — then switching away `f` and taking the `F₂` logarithm
would leave a matrix of rank `≤ m+1`. Measured instead:

| `m` | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| `rank_F₂` of the switched sign-log | 4 | 12 | 28 | 60 | 124 |
| `2^m − 4` | 4 | 12 | 28 | 60 | 124 |
| bilinear-form bound `m+1` | 4 | 5 | 6 | 7 | 8 |
| `N = 2^(m+1)−1` | 15 | 31 | 63 | 127 | 255 |

**`rank_F₂ = 2^m − 4` exactly**, at every label of `m = 3,4,5` (all of them) and at six labels of
`m = 6,7` — which is the descendant graph's DEGREE, and which exceeds the bilinear bound from `m = 4`
on. The one exception is the maximal seam, where the rank is **`0`**: consistent, since there the
two-graph is empty (Tier 65). So the family is not a bilinear-form two-graph, and `m = 3` is the
coincidence level where `2^m − 4 = m + 1` — exactly the level that would have made this look settled.

**(2) Not a regular two-graph.** The descendant graph at a reference is regular (§57.10), which
would allow it, but it is not STRONGLY regular: at `m = 5`, `λ` takes the values `12,14,16,18,20,…`
and `μ` takes `0,2,4,6,8,…`. So this is not an equiangular line system at the relative bound.

**What this does and does not license.** It excludes the two identifications that would have made
`tri3(P3̃)` a known computation, and it hands over a clean invariant (`rank_F₂ = 2^m − 4`, `0` at the
seam) that a literature search can be run against. It does **not** establish novelty — the two-graph
catalogues have not been searched, and that search is owed before any claim is made.

### 57.14 — Tier 66: the level-transfer backbone, and where the defect lives

Attacking the reviewer's recommended target. The recursion's `8` is supposed to be "each level-`m`
triangle lifts `2³` ways across the doubling". Measuring the four blocks of the level-`(m+1)` matrix
against the level-`m` one, at a label `W < 2^(m+1)` (valid at **both** levels), gives exactly that —
and the `(0,0)` block is not approximately the level below, **it is it**:

| block | agreement with `P3 l y W m` |
|---|---|
| `(0,0)` | **exact** — every `l, y, W`, index `0` and the diagonal included, 0 violations |
| `(0,1)` | flips exactly on `l = W` and on the coset line `l ⊕ y = W` |
| `(1,0)` | flips exactly on `y = W` and on `l ⊕ y = W` |
| `(1,1)` | flips exactly on the isolated line `l = W` or `y = W` |

The last three are measured at `m = 3,4`, **every** label (15/15 and 31/31), with `2(2^(m+1)−2)`
exceptions each. So the defect is supported on the two loci this lane already has names and theorems
for — the **isolated vertex** and the **coset line** — which is what makes the recursion look
provable rather than merely true.

**Tier 66 proves the first row:**

| | |
|---|---|
| `P3_level_stable` | `P3 l y W (m+1) = P3 l y W m` for `l, y, W < 2^(m+1)` |

no nonzero hypothesis, no `l ≠ y`. The proof is four branch reductions and nothing else: `R_lu` and
`R_ul` peel the `hi` argument at both levels, `R_ll` identifies the two results, and the `if y = 0`
branches that `R_ul` produces **match on the nose** on both sides — which is why the statement is
hypothesis-free. Kernel-clean.

**⚠ The `8×` shape is REFUTED — and both M1 providers called it before I measured.** Asked whether
the block table licenses "each of the eight `(λ_a,λ_b,λ_c)` orthants contributes a copy of `tri3_m`",
grok and zai both answered `[PROBLEM]`, with the same reason: `R_lu` transposes its arguments and
`R_ul` adds a minus, so the cross-block products need not assemble that way. Measuring the eight
orthant sums directly at `m = 3`, `W = 1` (`tri3_m = 714`):

| orthant weight | sum | vs `tri3_m` |
|---|---|---|
| 0 | `714` | **exact** |
| 1 (×3) | `−384` | `−1098` |
| 2 (×3) | `−528` | `−1242` |
| 3 | `1056` | `+342` |

So only the all-low orthant is a copy. **But the failure is structured**: the orthant sum depends
only on the **weight** of `(λ_a,λ_b,λ_c)`, at every label of `m = 3` and `m = 4` (15/15, 31/31), and
the weight-0 orthant is exactly `tri3_m` (15/15, 31/31). The transfer is therefore not `8×` a copy
but the lane's own **1+3+3+1 word decomposition** — the same shape as §34's `B`/`E` ledger and Tier
54's `tri3_expand`:

    tri3(P3̃)_{m+1} = O_0 + 3·O_1 + 3·O_2 + O_3,        O_0 = tri3(P3̃)_m

Consistent with this, the defect `T_{m+1}(W) − 8·T_m(W)` is **not** a function of `j = lsb W` — at
`m = 5` it takes eight distinct values. It is a function of the fibre, and by §57.11's closed form a
combination of exactly the `b` **suffix** characters.

The three off-blocks and the three unknown orthants `O_1, O_2, O_3` are measured, not proved.

### 57.15 — Tier 67: the three off-blocks are theorems, and the exceptions ARE `antisym`'s failures

The three off-blocks reduce exactly like the `(0,0)` one, and they all end at the same place: after
the two branch reductions peel the `hi` arguments, what is left is a `cdSigma` pair **in the wrong
order**, and `antisym` puts it right. So the exceptional loci measured in §57.14 are not a separate
phenomenon — **they are precisely where `antisym`'s hypotheses fail**:

| block | surviving pair | `antisym` applied to | breaks when |
|---|---|---|---|
| `(0,1)` | `σ(y⊕W, l)·σ(y, l⊕W)` | `σ(y, l⊕W)` | `l = W` or `l ⊕ y = W` |
| `(1,0)` | `σ(l, y⊕W)·σ(l⊕W, y)` | `σ(l, y⊕W)` | `y = W` or `l ⊕ y = W` |
| `(1,1)` | `−σ(l, y⊕W)·σ(y, l⊕W)` | **both** | `l = W` or `y = W` |

which is the measured table exactly — the isolated vertex and the coset line, nothing else. Note the
asymmetry the table explains: `(0,1)` does **not** need `y ≠ W`, because the factor `y ⊕ W` sits in
the pair that cancels rather than the one `antisym` touches.

All three are now Lean theorems (`P3_block01`, `P3_block10`, `P3_block11`), kernel-clean, with the
hypotheses `l,y ≠ 0` plus the failure locus of the `antisym` each one uses. Measured first, with
exactly the stated hypotheses: 0 violations in `252/2940/27900` (blocks 01 and 10) and
`210/2730/26970` (block 11) instances at `m = 2,3,4`.

⚠ **`(1,1)`'s theorem is NARROWER than the measured statement.** Its proof applies `antisym` twice
and the coset line breaks both, so it carries `l ⊕ y ≠ W` — but the measurement says the identity
still holds there, the two breakages cancelling (on that line both sides collapse to
`−σ(l,l)·σ(y,y)`). That case is not proved.

### 57.16 — `O_1, O_2, O_3`: two level constants, and the transfer collapses to ONE unknown

`O_0 = tri3(P3̃)_m` was already known. The other three are not independent:

| | value | scope |
|---|---|---|
| `O_1 − O_2` | `26·2^m − 64` | every label except the maximal seam |
| `O_3 − O_0` | `54·2^m − 90` | every label except the maximal seam |
| at `W = 2^m` | **both shift by `+288·[m−1,2]₂`** | |

Found on `m = 3,4,5` (both constants are affine in `2^m`, the successive differences doubling) and
then **confirmed out of sample at `m = 6`** — `c_1 = 1600` and `c_3 = 3366` at three different
labels, and the seam shift `44640 = 288·[5,2]₂` for **both** constants, all exact.

**★ The seam shift is the same number as §57.10's.** The maximal-seam excess in the deviation law
and the maximal-seam anomaly in the transfer are the identical constant `288·[m−1,2]₂`. Two
independent places, one number — which says the seam exception is a single phenomenon, not two.

**The transfer, off the seam.** Substituting into `1+3+3+1`:

    tri3(P3̃)_{m+1} = 2·tri3(P3̃)_m + 6·O_1 − 24·2^m + 102        0 violations, m = 4,5,6

so the recursion has ONE unknown left. (Note `2 + 6 = 8`: the `8` heuristic survives, but split
`2` on the level below and `6` on the weight-1 orthant, which is exactly what the refutation in
§57.14 forced.)

**And `O_1` is explicit at the fibre references.** Solving the line above with §57.11's closed form
at both levels:

    O_1(8g+1) = [ tri3(P3̃)_{m+1}(g) − 2·tri3(P3̃)_m(g) + 24·2^m − 102 ] / 6

0 violations at `m = 4,5,6`. This is a closed form *derived from* two measured closed forms, so it
adds no independent evidence — what it does is make `O_1` explicit wherever the deviation law lives,
which is exactly the references.

**Status of (III).** Still reduced, not proven. What moved: the transfer is now one recursion with a
single unknown, that unknown is explicit at the references, and the seam exception is identified
across two independent computations. What has not been attempted: a DERIVATION of `O_1` from the
block identities (the honest route — the one above is bookkeeping on top of measured closed forms),
the two level constants `26·2^m − 64` and `54·2^m − 90` from the exceptional-loci counts, the
coset-line case of block `(1,1)`, the `q`-binomial finish that would produce `[j,3]₂`, and the fibre
variation of `tr(Alo²P)` / `tr(AloP²)`. No difficulty is recorded for any of them.

**(III) is still reduced, not proven.**
