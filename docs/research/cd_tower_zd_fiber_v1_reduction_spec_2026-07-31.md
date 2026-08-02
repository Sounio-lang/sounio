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
same box". It holds because two `(e−2)`s cancel: off the sixth line `a ⊕ v = W′` the two agree, but
**on** it `Qgen = −1` while `Q′ = +1` (`Qgen'_coset_partner`) — and that surplus exactly replaces
the `(e−2)` that lemma A's `b = W′` row contributes (`Qgen'_label_right`, all `−1`) while its
`a = W′` row contributes none (`Qgen'_label_left`, all `+1`).

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
uses only `W′ ≠ 0` and not the lane's `Llo = 8y+1` oddness.

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
