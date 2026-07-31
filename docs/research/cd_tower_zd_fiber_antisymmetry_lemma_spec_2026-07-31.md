<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-antisymmetry-lemma-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-antisymmetry-lemma-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — the fiber antisymmetry lemma: the explicit low-rank factorisation, found

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ZD_FIBER_ANTISYMMETRY_LEMMA__FACTORISATION_FOUND_LOWRANK_BOUND_PROVEN`
**Parents:** `cd_tower_zd_fiber_spectral_forall_n_progress_2026-07-26.md`, `cd_tower_zd_fiber_signed_localization_2026-07-26.md`, `cd_tower_zd_fiber_spectral_classifier_2026-07-26.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_antisymmetry_lemma_contract.py`

---

## 0. The result

The prior rung named exactly one missing object, twice, verbatim:

> "The ∀n proof needs the explicit algebraic low-rank factorisation (Walsh/character-sum type)
> — **OPEN**." — `cd_tower_zd_fiber_spectral_forall_n_progress_contract.py`, verdict note
>
> "The remaining route is the explicit algebraic factorisation `A_σ = Cᵀ S C` (Walsh /
> character-sum type) — **not found in-session**." —
> `cd_tower_zd_fiber_spectral_forall_n_progress_2026-07-26.md`, §2

**It is found, and it is elementary.** The factorisation is not a Walsh character sum — it is a
**rank-2 folding induced by a sign involution on the fiber**:

> **Fiber antisymmetry lemma.** For `n ≥ 6` and every fiber `L = L_lo | H` (`H = 2^{n-1}`),
> the signed resonance matrix satisfies
>
> `A_σ(l ⊕ L_lo, y) = − A_σ(l, y)`   for all lo-labels `l, y ∈ [1,H)` with `l ⊕ L_lo ≠ 0`.

Each row is the negative of its partner row under the involution `l ↦ l ⊕ L_lo`. That single
identity supplies everything the prior rung was missing:

| | |
|---|---|
| the factorisation | `A_σ = Jᵀ M J` with `J Jᵀ = 2I`, `J` a `(2^{n-2}−1)×(2^{n-1}−1)` 0/±1 incidence matrix |
| the rank bound | `rank(A_σ) ≤ 2^{n-2}−1` — **derived, ∀n** |
| the `−1` | the isolated vertex sits at exactly `l = L_lo`, **derived**, not observed |
| the spectrum | nonzero `spec(A_σ) = spec(2M)` — an **exact halving**, ∀n |

**Evidence.** The lemma is **proven ∀n in Lean** (§8, `A1`, kernel-checked, no `sorry`, no
`native_decide`). Independently, `A1_LEMMA` runs it over **all fibers at n = 6..11** — `0`
violations in `1 221 547 860` entry comparisons. `A5`'s rank equality is confirmed over **all** fibers at
n ≤ 10 (the prior rung's `V3` reached only n ≤ 8).

---

## 1. Clause table

| clause | statement | status |
|---|---|---|
| `A0_PARITY` | the vectorised builders reproduce the in-tree `sign_table`/`A_sig` entrywise | measured |
| `A1_LEMMA` | `A_σ(l⊕L_lo, y) = −A_σ(l, y)`, all fibers, n=6..11 | measured, 0/1.2e9 |
| `A2_MASK` | `res(l⊕L_lo, y) = res(l, y)` for `y ≠ L_lo` | measured + derived (§2) |
| `A2_VACUITY` | `P1` and `P3` are *always* symmetric ⇒ 2 of the 3 clauses of the resonance predicate in the `A_sig` builder are vacuous | measured |
| `A2_SIBLING` | the same vacuity, measured on the sibling rung's own four-term `resonant()` | measured |
| `A3_CORE` | `P1(l⊕L_lo, y) = −P3(l, y)` and `P3(l⊕L_lo, y) = −P1(l, y)` | **derived** (§2), measured |
| `A4_ISOLATED` | the unique zero row/column is at `l = L_lo`; level-`(n−1)` sub-lemma `τ(l,L_lo) = −τ(l⊕L_lo, L_lo)` | **derived** (§3), measured |
| `A5_FACTOR` | `A_σ = Jᵀ M J`, `J Jᵀ = 2I`; `rank ≤ 2^{n-2}−1` ∀n | factorisation **derived**; equality measured n≤10 |
| `A6_HALVING` | nonzero `spec(A_σ) = spec(2M)` | **derived** from `A5`, measured n≤9 |
| `A7_DEFLATE` | the lemma is not implied by the prior rung's `V2` doubling containment | measured (§5) |
| `A8_NULL_a` | a perturbed *pairing* `L' ≠ L_lo` always **breaks** the antisymmetry | measured |
| `A8_NULL_b` | a foreign fiber's *matrix* never satisfies `L_lo`'s antisymmetry | measured |
| `A9_LEAN` | the Lean file's `cdSigma`/`hi`/`P1`/`P3` are entrywise the ones measured here | measured (§8) |
| `A10_DIAG` | the builder's `np.fill_diagonal(A, 0)` is a **no-op** — resonance already fails on the diagonal | measured, **Lean-proven** (§8) |

---

## 2. The proof

Write `τ(x,y) = σ(x,y,n−1)`, `m = l ⊕ L_lo`, `m' = y ⊕ L_lo`, `h(x) = x ⊕ L`. Since
`l, y, m, m' < H` all have top bit `0`, and `h(l) = m + H`, `h(y) = m' + H`, the CD top-bit
recursion gives, **for `m' ≠ 0`** (i.e. `y ≠ L_lo`):

```
σ(l, y)      = τ(l, y)          σ(h(l), h(y)) =  τ(m', m)
σ(l, h(y))   = τ(m', l)         σ(h(l), y)    = −τ(m, y)
```

so `P1(l,y) = τ(l,y)·τ(m',m)` and `P3(l,y) = −τ(m',l)·τ(m,y)`. Substituting `l ↦ m` — whose
partner is `l` — and reading the same four lines back:

```
P1(m, y) =  τ(m,y)·τ(m',l) = −P3(l, y)
P3(m, y) = −τ(m',m)·τ(l,y) = −P1(l, y)
```

**The involution swaps `P1` and `P3` and negates them.** No case analysis; it is one
substitution in the recursion. Everything follows:

- **Mask.** `A2_VACUITY` shows `P1` and `P3` are always symmetric, so
  `res = (P1 = P3)` — the other two clauses never bind. (`P3` is symmetric by inspection:
  `P3(y,l) = −τ(m,y)τ(m',l) = P3(l,y)`. `P1`'s symmetry is the CD basis-unit anticommutation
  sign `c(x,y) = τ(x,y)τ(y,x)`, which equals `−1` for distinct nonzero arguments, giving
  `c(l,y) = c(m,m')` identically.) Then
  `res(m,y) ⟺ P1(m,y) = P3(m,y) ⟺ −P3(l,y) = −P1(l,y) ⟺ res(l,y)`. **The mask is invariant.**
- **Sign.** On the support, `A(m,y) = −P1(m,y) = P3(l,y) = P1(l,y) = −A(l,y)`. ∎

This whole argument is formalised: `core_P1`/`core_P3` (the substitution), `P1_symm`/`P3_symm`
(the vacuity), `resB_inv` (the mask), `A1` (the conclusion) — see §8.

The lemma has the shape of the seam-flip law because it *is* a seam flip: `L_lo ⊕ L = H`
exactly, so `l ↦ l ⊕ L_lo` acts on the pair `(l, h(l))` as `(l ⊕ L_lo, l ⊕ H)` — a flip of the
outer seam bit on the hi coordinate.

---

## 3. Why the isolated vertex is at `l = L_lo` — derived

The one column excluded above, `y = L_lo`, has `m' = 0`, which flips the `bL == 0` branch of the
recursion. There `res` reduces to `τ(l, L_lo) = τ(l ⊕ L_lo, L_lo)`, while the level-`(n−1)`
sub-lemma

> `τ(l, L_lo) = −τ(l ⊕ L_lo, L_lo)`

holds (`A4`, all fibers, n ≤ 10). So `res` fails **identically** on that column: row and column
`L_lo` are zero, in every fiber. The isolated vertex is not an accident of small `n` — it is the
same antisymmetry one level down, and it is the source of the `−1` in `2^{n-2}−1`.

Counting. The involution `l ↦ l ⊕ L_lo` is **fixed-point-free** on `[1,H) \ {L_lo}`: a fixed
point would need `l ⊕ L_lo = l`, i.e. `L_lo = 0`, which is excluded — and `l = L_lo` is the one
label whose image leaves the index set (`L_lo ⊕ L_lo = 0`). So the `2^{n-1}−2` labels `l ≠ L_lo`
fall into **disjoint** pairs, exactly `2^{n-2}−1` of them, whose rows agree up to sign, plus one
zero row. Hence `rank(A_σ) ≤ 2^{n-2}−1` **for all n**.

---

## 4. The factorisation, and the spectral halving

Let `reps` be one label per pair (`|reps| = 2^{n-2}−1`), `M = A_σ[reps, reps]`, and

```
J[k, rep_k] = +1,   J[k, rep_k ⊕ L_lo] = −1,   0 elsewhere.
```

Then `A_σ = Jᵀ M J` exactly, and — because each row of `J` has two nonzeros of modulus 1 with
pairwise disjoint supports — `J Jᵀ = 2I`. This is the `Cᵀ S C` the prior rung asked for, with
`C = J`. Since `spec(XY) \ {0} = spec(YX) \ {0}`,

> **nonzero `spec(A_σ) = spec(2M)`.**

The eigenproblem descends, for all `n`, to an explicit matrix of half the dimension. Measured at
n = 6..9 over all fibers, together with the count that matters: `#distinct spec(M)` equals
`#distinct spec(A_σ)` — `6, 12, 24` at n = 6,7,8 — so the reduction is **spectrum-faithful** and
the `V1` question transfers to `M` without loss.

---

## 5. Not claimed

- **This does not close ∀n spectral completeness.** `#distinct spectra = 3·2^{n-5}` remains
  **OPEN**. An explicit rank and factorisation do not exclude cospectral fibers at large `n`;
  §4 *reduces* that question to `M`, it does not answer it. Any reading of this rung as closing
  `V1` is wrong.
- **The rank equality is not derived.** `rank ≤ 2^{n-2}−1` is proven ∀n; `rank = 2^{n-2}−1`
  needs the independence of the `2^{n-2}−1` class representatives, which is **measured**
  (all fibers, n ≤ 10) and not proven.
- **Not implied by the prior rung.** `A7`: `V2` states that the top-left block of `A_σ(n)` on
  lo-labels `[1, 2^{n-2})` equals `A_σ(n−1)`. For the `64` fibers at n = 8 with
  `L_lo ≥ 2^{n-2}`, the pairing `l ↔ l ⊕ L_lo` carries `4032` inside-block labels *outside*
  that block, so a statement about the block alone cannot yield the lemma.
- **Not vacuous — two arms, and neither is the whole claim.** `A8_NULL_a` holds the matrix at
  `L_lo` and perturbs the *pairing* (`L' ≠ L_lo`): always breaks. `A8_NULL_b` holds the pairing
  at `L_lo` and perturbs the *matrix* (a foreign fiber's `A_σ(L'')`): always breaks. Together
  they show the identity binds this fiber's matrix to this fiber's involution, and is not a
  property of the ambient sign table. They do not, by themselves, establish the lemma — that is
  `A1` plus §2.
- **A dropped pattern, recorded so it is not re-chased.** The irrational eigenvalues
  `−1 ± √57` (n=6) and `±2√57` (n=7) looked like a closed-form signature. At n = 8, `4√57` does
  appear — but so does a **new** irrationality `√249`. `√57` is therefore *not* a universal
  fingerprint, and the closed-form-spectrum route is not supported by it. Two data points were
  not enough; the third killed it.
- **Numerical certificate.** Exact integer sign table, `D3` class — same standing as the prior
  rung. For what *is* kernel-checked, see §8; the numerical clauses are not.

---

## 6. A finding about the in-tree predicate

Of the three clauses in the resonance predicate **as used in the `A_sig` builder** of
`cd_tower_zd_fiber_spectral_forall_n_progress_contract.py` (`P1 == P1ᵀ`, `P3 == P3ᵀ`,
`P1 == P3`), the first two are **always true** — `A2_VACUITY`, measured over all fibers at
n = 6..10. `res` reduces to `P1 == P3`.

The claim is *not* asserted for the lane at large; it is measured a second time, on the other
rung's own code. `A2_SIBLING` runs the four-term form
`P1 == P2 == P3 == P4` of `cd_tower_zd_fiber_signed_localization_contract.py`'s `resonant()` —
where `P2`, `P4` are the transposes of `P1`, `P3` — and finds `P1 ≠ P2` in `0/279 838` and
`P3 ≠ P4` in `0/279 838` (n = 6,7, all fibers). The vacuity therefore transfers to that rung
as written. No claim is made about the classifier rung, which builds the full annihilation
graph rather than `A_σ` and was not measured here.

This is reported, **not repaired**: changing the predicate would change the object every prior
rung measured.

---

## 7. Reproduce

```sh
python3 scripts/research/cd_tower_zd_fiber_antisymmetry_lemma_contract.py   # rc=0
python3 scripts/research/cd_tower_zd_fiber_spectral_forall_n_progress_contract.py  # unchanged
python3 scripts/research/cd_tower_zd_fiber_spectral_classifier_contract.py         # unchanged
```

Runtime is load-sensitive: **123 s solo**, **983 s** measured while the two prior contracts were
running concurrently on the same box. `A1` at n = 11 is the dominant term. The two prior
contracts are **not modified by this rung** — only the prior *spec* gained a forward pointer —
and both were re-run to confirm it.

`A0` pins this rung's vectorised builders to the in-tree ones entrywise, so the certificate is
verified against the lane's own generator rather than a reimplementation.

---

## 8. What is Lean-proven, by tier

`formal/lean4/SounioZDFiberAntisym.lean` — self-contained, Mathlib-free, **no `sorry`, no
`native_decide`**, axioms `[propext, Quot.sound]` or `[propext, Classical.choice, Quot.sound]`
(Classical enters via `simp`; a logical axiom, not a computational trust anchor).

| statement | Lean name | status |
|---|---|---|
| `P1 (l ⊕ L_lo) y = − P3 l y` (`A3`) | `core_P1` | **kernel-checked ∀n** |
| `P3 (l ⊕ L_lo) y = − P1 l y` (`A3`) | `core_P3` | **kernel-checked ∀n** |
| `P3 l y = P3 y l` (`A2_VACUITY`) | `P3_symm` | **kernel-checked ∀n** |
| `P1 l y = P1 y l` (`A2_VACUITY`) | `P1_symm` | **kernel-checked ∀n** (uses `antisym`) |
| `resB (l ⊕ L_lo) y = resB l y` (`A2_MASK`) | `resB_inv` | **kernel-checked ∀n** |
| **`Asig (l ⊕ L_lo) y = − Asig l y`** (`A1`) | `A1` | **kernel-checked ∀n** |
| `Asig l l = 0` — `fill_diagonal` is a no-op (`A10`) | `Asig_diag` | **kernel-checked ∀n** |
| the additive form *is* the XOR form | `hi_eq_xor` | **kernel-checked ∀n** |
| `A4`'s sub-lemma `τ(l,L_lo) = −τ(l ⊕ L_lo, L_lo)` | — | **not formalised** |
| rank, spectra, the factorisation `Jᵀ M J` | — | numerical only |

All 15 theorems in the file report `[propext, Quot.sound]` or
`[propext, Classical.choice, Quot.sound]` under `#print axioms` — no `sorryAx`, no
`native_decide`.

**So the lemma itself is proven, not only its core.** The earlier version of this rung stopped
at `A3` because `P1`-symmetry reduces to `antisym`, which lived only on the unmerged branch
`lean/cd-seamflip-forall-n`. That branch is now merged (additively, 13 files, no deletions), and
the gap closed.

**Two obligations that came with it, both discharged.** (i) `Asig_diag`: the Lean `Asig` does
not zero the diagonal while the builder calls `np.fill_diagonal(A, 0)`. Resonance already fails
on the diagonal (`P1 = +1`, `P3 = −1`), so the call is a no-op and the matrices coincide —
a **third vacuity** in the builder, after `A2`'s two. (ii) The hypotheses `l ⊕ L_lo ≠ 0` and
`y ⊕ L_lo ≠ 0` are not a weakening: they are exactly §3's excluded row and column.

The side conditions in the Lean statements are exactly §2's: `y ≠ 0` and `y ⊕ L_lo ≠ 0` — the
latter being `y ≠ L_lo`, §3's excluded column.

**The Lean object is the measured object.** Clause `A9` transcribes the Lean file's `cdSigma`
into Python and checks it entrywise against the **in-tree** `cd_sigma` — not this rung's
vectorised rewrite, since `A0` pins that only at n = 6,7 while `A9` also runs at n = 4,5,8 —
over all pairs, then checks Lean's `hi`/`P1`/`P3` against the builder's over all fibers at
n = 6,7,8. It carries its own negative control: two **wrong** hi-maps (`l + H`, i.e. forgetting
the XOR, and `(l ⊕ L_lo ⊕ 1) + H`) must *disagree* with `P1`, so the arm can fail. Without this
bridge the formalisation could be about a lookalike; without the control, the bridge could not
fail.

**A provenance note worth recording.** `antisym` and the four branch reductions
`R_ll/R_lu/R_ul/R_uu` were cited by this lane as "proven ∀n" — they live in
`SounioSeamFlip.lean` on a branch that has **never been merged**. They were re-verified to
compile clean here (no `sorryAx`) and the branch reductions are now carried into
`SounioZDFiberAntisym.lean`, so they exist in the tree for the first time. `antisym` itself is
still only on that branch. This is the R20 dangling-citation pattern, found again.

---
