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
> `E(m,W) = Σ_{i : bit_{i−1}(W)=1} (2^i−4)(2^i−8)·4^{m−i}·(−1)^{popcount(W ≫ i)}`

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

**987/987 labels, both families, `n = 6..10`, 0 mismatches** against `A_sig_fast`/`traces23`.

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
