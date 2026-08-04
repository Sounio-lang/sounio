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

**(I) is PROVEN ∀n.** `Ncnt_inj_g` / `Ncnt_inj_gnorm` in `SounioZDFiberAntisym.lean`: `tr(A²)` —
equivalently `Ncnt` — is injective in the fibre coordinate `g`. Kernel-clean
`[propext, Classical.choice, Quot.sound]`. §30 had reduced (I) to a statement about the closed
form alone (no `Qgen'`); it is now closed.

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
- valuation `v₂(F(m,y)) = 2m − 2k − 3` and injectivity: `m = 6..14`, 4088 labels,
  **0 mismatches, 0 collisions**.

The oracle is pinned to §30's independently recorded number — `Ddig(9,41) = 116736`, constant
across the fibre `41…47`.

**Still open, unchanged:** (III); `tr(A³)`'s general closed form; (d); V1. §30 already showed
`tr(A²)` contributes nothing inside a fibre, so (III) is untouched by this.
