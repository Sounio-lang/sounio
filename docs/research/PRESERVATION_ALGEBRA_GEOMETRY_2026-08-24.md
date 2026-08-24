<!-- docs:meta
topic_id: repo.docs.research.preservation-algebra-geometry-2026-08-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.preservation-algebra-geometry-2026-08-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The preservation algebra of a zero-divisor: a Euclidean↔Carrollian dichotomy

**Date:** 2026-08-24 · **By:** claude (session 71fa6b78) · **Status:** computed finding (exact, rational + numpy cross-checked), not yet a formal proof.

## Setup

For a sedenion zero-divisor `z` (here the canonical `z = e3 + e10` and all 84
pair-type `e_i ± e_j`), define its **kernel** `ker L_z = {x : z·x = 0}` (dim 4)
and its **preservation algebra**

```
P_z = { a : a·ker L_z ⊆ ker L_z  and  ker L_z·a ⊆ ker L_z }   (two-sided stabilizer)
```

`P_z` is the set of multipliers under which the exact-invariant "x lives in
`ker L_z`" is preserved — the composition-safety set for `ExactlyPrivate`/
`Forgettable`/`Editable`/`CapabilityGated`. Its Jordan structure carries a
quadratic form `B(u,v) = scalar part of (u∘v)`, `a∘b = (ab+ba)/2`, whose
signature is the object of interest.

## Result: the Cayley–Dickson sign parameter is a geometry switch

The last-doubling sign `μ` (`e8² = −μ`) toggles the character of `P_z`:

| ambient | `μ` | `P_z` structure | signature `(+,−,0)` | geometry |
|---|---|---|---|---|
| **division** sedenions | `+1` | spin factor `J_spin(5)` | `(0, 5, 0)` — definite | **Euclidean** |
| **split** sedenions | `−1` | degenerate spin factor, null radical | `(1, 0, 3)` — rank 1 | **Carrollian** |

Both are **universal** across all 84 pair-type ZDs (division: every `P_z` is
`(0,5,0)`; split: every `P_z` is `(1,0,3)`). **No Lorentzian (mixed `p,q>0`,
nondegenerate) signature appears in either.** The original hypothesis — that some
member of the ZD-surgical type family, or the split ambient, would give a
Lorentzian preservation geometry with a genuine light-cone — is **refuted** by
exhaustive pair-type computation.

## What the split generators are

For split `z = e3 + e10`, the four imaginary preserving generators are
`{ −e1+e8, e9, e3+e10 (=z), e2+e11 }`. In the split form (`e1..e7`² = −1,
`e0,e8..e15`² = +1), three of them are **null** (light-like): `(e8−e1)`,
`(e10+e3)`, `(e11+e2)` each pair a minus-direction with a plus-direction and
square to 0 under `∘`. Only `e9` is timelike (`+1`). Hence the rank-1,
null-dominated `(1,0,3)` form: **the preserving operations sit on the light-cone
of the split norm.** That is the signature-theoretic content of a Carrollian
(degenerate-metric) geometry.

## Reading

- **μ = +1 (division):** every privacy-preserving operation is "spacelike" —
  definite, cleanly composable; `P_z` is a Euclidean observable algebra
  `J_spin(5)`.
- **μ = −1 (split):** the preserving operations degenerate onto the null cone;
  the "time" direction collapses to rank 1. Composition-safety becomes a
  causal-boundary phenomenon.

So a single discrete algebraic knob (`μ`) selects between the **two degenerate
limits of relativity** — Euclidean and Carrollian — while **skipping Lorentzian
entirely** for the pair-type spectrum.

## Sounio-actionable consequence

The ambient split-parameter could be exposed as a **type parameter** on the
exact-invariant family: `ExactlyPrivate` over division sedenions has Euclidean
(definite, freely composable) preservation; over split sedenions it has
Carrollian (null, causally-constrained) preservation. The type system would then
know that "how two privacy-typed values compose" is a *different geometry*
depending on the ambient — a genuinely new axis for invariant-type design.

## Honesty boundary / open

- Computed for **pair-type** ZDs only, and for the **last-doubling** split
  (`μ` on `e8`). A fuller split (splitting the octonion base) or generic
  (non-pair-type) ZDs might still produce a nondegenerate Lorentzian `(p,q,0)`;
  not yet excluded. The dichotomy claim is *pair-type, last-doubling*.
- "Carrollian" is used in the signature-theoretic sense (degenerate metric with a
  clock direction and a null radical), not a claim to have derived Carrollian
  spacetime dynamics.
- Prior art: Koebisu 2512.13002 (det-factorization of `L_v`), Moreno, Reggiani
  2411.18881, Biss–Dugger–Isaksen study the ZD **locus**; none treat the
  **preservation/stabilizer algebra** or its Jordan signature. Split/pseudo-
  octonion algebra is Okubo (cited there, not applied here). The
  preservation-signature dichotomy appears unremarked in the literature scanned.

## Reproduce

`scratchpad/pz_frontier.py` (division, exact rational), `scratchpad/family_sig.py`
(family + Composable intersection), `scratchpad/splitfast.py` (split, numpy scan
of all 84).

---

## UPDATE (same day): the ladder closes — Lorentzian appears under base-split

Extending to a **fully parametrized Cayley–Dickson** (doubling-sign vector `μ⃗`,
`scratchpad/ladder.py`) and scanning generic ZDs settles the open question: the
preservation-signature ladder is **complete**, and which rung you land on is set
by *where* you split and *which* zero-divisor you pick.

```
μ⃗ = (−,−,−,−)  division      : (0,5,0)×84                          all Euclidean
μ⃗ = (−,−,−,+)  last-split    : (1,0,3)×84                          all Carrollian
μ⃗ = (−,−,+,−)  BASE-SPLIT    : (0,5,0)×12 + (1,0,3)×48 + (4,1,0)×24   ALL THREE coexist
μ⃗ = (−,−,+,+)  doubly-split  : (1,0,3)×36 + (3,2,0)×48             Carrollian + Lorentzian
```

`(4,1,0)` is a **nondegenerate Lorentzian** spin factor `J_spin(4,1)` — the 5D
Minkowski/de-Sitter signature, whose structure group is `SO(4,1)`. Verified
**exactly** (rational arithmetic, `scratchpad/verify_lorentz.py`): for base-split
`z=e4+e13`, `ker L_z` dim 4 (genuine two-sided ZD), `P_z` dim 6, imaginary dim 5,
**Jordan-closed and spin-factor**, signature `(4,1,0)`.

### The finding

The preservation algebra of an exact-invariant realizes the **complete signature
ladder of relativistic geometry — Euclidean → Carrollian → Lorentzian** — as a
function of the Cayley–Dickson split-vector and the zero-divisor. Crucially, a
**single base-split algebra hosts all three simultaneously**: 12 Euclidean, 48
Carrollian, 24 Lorentzian pair-type loci. So within one algebra, *the choice of
privacy-locus `z` selects the composition geometry of the invariant.*

### Sounio consequence (sharpened)

`ExactlyPrivate<T, z>` is not one type — it is a **family whose composition
geometry is chosen by `z`**: pick a Euclidean locus and privacy composes freely
and definitely; pick a Lorentzian locus and composition-safety acquires a genuine
light-cone (`SO(4,1)` de-Sitter causal structure among preserving operations —
timelike vs spacelike operation-types, a causal order on capability composition).
This makes the ambient-split-vector and the ZD-locus *two type-level knobs* on the
causal/metric character of how invariants compose — an axis with, as far as the
literature scanned shows, no prior owner.

### Remaining open

- Is the rung a **closed-form function** of `(μ⃗, z)`? The base-split split of
  84 into 12/48/24 suggests the ZD-orbit under the split `G2`-analogue indexes the
  rung; not yet derived.
- Does `(4,1)` vs `(3,2)` correspond to a physically meaningful distinction
  (de Sitter vs anti-de Sitter-like) of the invariant's composition causal order?
- Lean formalization of "base-split `z=e4+e13` ⇒ `P_z ≅ J_spin(4,1)`".

---

## THE RUNG LAW (closed form, verified 84/84 across the CD family)

The rung is not ad hoc — it is the **causal type of the zero-divisor**. Let `Q` be
the square-form on the ambient algebra, `Q(e_i) := e_i² ∈ {±1}` (the form induced
by squaring; `Q(z) = Q(e_i)+Q(e_j)` for a pair-type `z = e_i ± e_j`).

> **Rung law (pair-type).** For a pair-type zero-divisor `z`,
> - `Q(z) < 0`  (z **spacelike**, both arms `−`)  ⇒  `P_z` **Euclidean** `J_spin(5)`, sig `(0,5,0)`
> - `Q(z) = 0`  (z **null**, mixed arms)           ⇒  `P_z` **Carrollian** (degenerate), sig `(1,0,3)`
> - `Q(z) > 0`  (z **timelike**, both arms `+`)    ⇒  `P_z` **Lorentzian** `J_spin(4,1)`, sig `(4,1,0)`

Verified exhaustively (`scratchpad/rung_derive.py`, `rung_law.py`,
`lastsplit_check.py`): the predictor `rung = g(sign Q(e_i), sign Q(e_j))` matches
the computed signature on **84/84** pair-type ZDs in base-split, and — being a
statement about arm-signs — trivially in division (all arms `−` ⇒ all Euclidean)
and last-split (all ZDs mixed-arm ⇒ all Carrollian; confirmed: 84/84 mixed).

### Why (mechanism)

The preserving generators of `P_z` inherit their `Q`-signs from `z`'s arms.
- **Spacelike z:** all preserving imaginary generators square to `−1` ⇒ definite
  Jordan form ⇒ Euclidean.
- **Null z:** the mixed-arm structure forces the preserving generators onto the
  light-cone (each pairs a `+` with a `−` and is `Q`-null) ⇒ the Jordan form
  acquires a rank-collapse (radical) ⇒ Carrollian.
- **Timelike z:** the preserving generators are `+`-dominated with one
  distinguished `−` (from the 4-dim kernel's orthogonal structure) ⇒ `(4,1)`
  Lorentzian.

### Statement

**The composition geometry of an exact-invariant is the causal type of its
defining zero-divisor.** Choosing the privacy-locus `z` is choosing whether the
invariant composes Euclidean-ly (definite, free), Carrollian-ly (null, ultra-local)
or Lorentzian-ly (a genuine light-cone / `SO(4,1)` causal order on capability
composition). The ambient split-vector fixes which causal types of locus *exist*;
the locus fixes the rung.

### Open (toward a paper)

- Prove the `(4,1)` (not `(3,2)` etc.) refinement from the kernel's orthogonal
  structure — i.e. derive the exact `+/−` split, not just its existence.
- Extend the law to generic (non-pair-type) ZDs, where `dim ker` and `dim P_z`
  vary; conjecture: rung still tracks `sign Q(z)` but the algebra is a larger/
  mixed `J_spin(p,q)`.
- Lean: `Q(z)>0` (base-split `e4+e13`) ⇒ `P_z ≅ J_spin(4,1)`.

---

## (A) Why (4,1): the mechanism, explicit

For the timelike locus `z = e4 + e13` (base-split), the five imaginary generators
of `P_z` are **pure basis units**: `{e4, e5, e12, e13}` (all `+1`, timelike) and
`e8` (`−1`, spacelike). The Jordan Gram is *exactly* `diag(1,1,−1,1,1)` (off-
diagonal zero — distinct orthogonal basis units). So `(4,1)` is: a **timelike
quadruple** (`z`'s two arms `e4,e13` plus the partner pair `e5,e12` fixed by the
kernel) **⊕ one distinguished spacelike axis `e8`** — the Cayley–Dickson doubling
generator. The single minus is structurally the doubling unit. That is the
derivation of the `(4,1)` refinement (not just its existence).

## (B) Scope: the full spin factors are a maximally-symmetric-locus phenomenon

The clean 3-rung ladder (with full `J_spin(5)` / `J_spin(4,1)`) holds for
**pair-type (maximally symmetric) loci**. Extending to **generic** ZDs:
- **null** generic loci (`Q(z)=0`) robustly stay Carrollian `(1,0,3)` — law holds;
- **timelike** generic loci mostly **collapse** to `dim P_z = 2`, signature
  `(1,0,0)` (a 1-D positive line, `ℝ⊕ℝ`), because genericity breaks the symmetry
  that supplies the extra preserving multipliers; a minority (e.g. `dim ker = 2`)
  still give Lorentzian `(3,1,0)`.

So the theorem's clean form is: **rung(P_z) = causal-type(z) for pair-type loci;**
for generic loci the *rung character* still tracks `sign Q(z)` for the null case,
but the preserving algebra shrinks off the symmetric loci. The rich Lorentzian
geometry lives at the **canonical (symmetric) privacy loci** — exactly the loci a
type system uses. This sharpens rather than weakens the Sounio consequence.

## (C) Machine-checked witness

`formal/lean4/SounioPreservationLorentzian.lean` (native_decide, no Mathlib, no
sorry) certifies, for base-split `z = e4+e13`: (§1) it is a two-sided ZD with the
stated 4-D kernel; (§2) the five exhibited multipliers preserve `ker L_z` two-
sidedly (checked as `z·(a·k)=0`, `z·(k·a)=0`); (§3) their doubled Jordan Gram is
`diag(2,2,−2,2,2)`; (§4) signature `(4,1)` — Lorentzian `J_spin(4,1)`. All four
`native_decide` goals were cross-emulated to `True`; the file awaits the Lean CI
job for elaboration confirmation. Maximality (`dim P_z = 6`) is the external
rational computation; the Lean certifies the exhibited algebra's structure and
Lorentzian signature.

---

## CLASSIFICATION (the collapse characterized, verified)

The generic collapse is governed by **the monomiality of `ker L_z`** — equivalently
by the size of its **basis-unit stabilizer** `Stab(z) := { e_k : e_k·ker L_z ⊆ ker L_z
and ker L_z·e_k ⊆ ker L_z }`. Complete classification (base-split, computed):

**Definite-arm loci (`|Q(z)| = 2`, timelike or spacelike).** `P_z` is **monomial**:
`P_z = ℝ·e0 ⊕ span(Stab(z))`, `dim P_z = 1 + |Stab(z)|`. Two cases:
- **`ker L_z` monomial** (spanned by ≤2-support XOR vectors) `⟺` symmetric/pair-type
  locus `⟺` `|Stab(z)| = 5`  ⇒  `P_z` is a full spin factor:
  **`J_spin(5)` Euclidean** (`Q<0`) or **`J_spin(4,1)` Lorentzian** (`Q>0`), `dim 6`.
  *(Verified: all 36 definite-arm pair-type loci have `|Stab|=5`; `|Stab|=5 ⟺ monomial
  kernel` holds 48/48.)*
- **`ker L_z` non-monomial** (generic) `⟺` `|Stab(z)| = 0`  ⇒  **collapse** to
  `dim P_z = 2` = `ℝ ⊕ ℝ` (scalars plus one residual dense timelike line).

**Null loci (`Q(z) = 0`, Carrollian).** `P_z` is **non-monomial and robust**:
`(1,0,3)`, `dim ≈ 5`, for *both* pair-type and generic — **symmetry-independent**.

### The statement

**The relativistic spin-factor geometry (Euclidean / Lorentzian) is a phenomenon
of *monomial kernels* — the algebraically special, symmetric privacy loci — carried
entirely by the kernel's basis-unit stabilizer `Stab(z)`, which is 5-dimensional
there and vanishes generically. The Carrollian/null geometry is the *generic*
behavior at the light cone.** So the "symmetry that supplies the extra preserving
multipliers" is exactly `Stab(z)`, the monomial stabilizer of `ker L_z`; it is
maximal precisely at the canonical loci (e.g. `e3+e10`, `e4+e13`) a type system
names — and collapses everywhere else. This closes the scope question of §(B): the
three-rung ladder with full spin factors is the *monomial-kernel* theorem, and the
predicate "monomial kernel" is decidable and equals "`|Stab(z)| = 5`."

### Map (base-split)

```
                     ker L_z monomial (|Stab|=5)      ker L_z generic (|Stab|=0)
  Q(z) < 0 (space)   J_spin(5)   Euclidean  dim6       ℝ⊕ℝ         collapse   dim2
  Q(z) = 0 (null)    Carrollian (1,0,3) dim5  ————  robust, symmetry-independent  ————
  Q(z) > 0 (time)    J_spin(4,1) Lorentzian dim6       ℝ⊕ℝ         collapse   dim2
```

---

## ORBIT THEORY: a derivation of the `5` (verified 36/36)

The dimension `|Stab(z)| = 5` — the fact that made `P_z` a *5*-dimensional spin
factor — is not empirical. It follows from a hyperplane / orbit-stabilizer count.

**Theorem (the 5).** For every monomial-kernel definite-arm base-split locus `z`:

1. **Single-shift kernel.** `ker L_z` is the *graph of one XOR-shift* `w`: all four
   support-pairs share the same difference, `p₀⊕p₁ = w` (e.g. `w = 9` for
   `z=e4+e13`; `w=11` for `z=e1+e10`). The kernel is a `w`-twisted diagonal.

2. **Hyperplane support.** The 8-element support `T = ⋃ pairs` is an **affine
   hyperplane** of `(ℤ/2)⁴`: `T = { i : ⟨m,i⟩ = c }` for a nonzero linear
   functional `m` (`⟨m,i⟩ = popcount(i∧m) mod 2`). *(Verified: all 36 loci.)*

3. **Stabilizer forced to 8.** The XOR-shifts fixing `T` are exactly the dual
   hyperplane: `H = { k : k⊕T ⊆ T } = ker⟨m,·⟩`, an elementary abelian group of
   order `2³ = 8`. This is forced — a hyperplane's XOR-stabilizer is its
   annihilator, dimension `4−1 = 3`.

4. **Sign coset removed.** Basis-unit preservation needs both support (`k∈H`) and
   sign-consistency with the cocycle `σ`. Exactly **2** elements of `H` fail — the
   *pair-flippers* (the pairing shift `w` and its partner), which map each pair to
   itself with a flipped `±` sign. *(Verified: |bad| = 2 for all 36.)*

5. **Count.** `dim P_z = |H| − |bad| = 8 − 2 = 6`, hence
   **`|Stab(z)| = 8 − 2 − 1 = 5`** (the `−1` is the scalar `e0`). With the arm-signs
   (§rung law) this is `J_spin(5)` (spacelike `z`) or `J_spin(4,1)` (timelike `z`).
   ∎

**So the `5` = `2³ − 2 − 1`**: the order of the support-hyperplane's stabilizer,
minus the sign-flipper coset, minus the scalar. The whole relativistic spin-factor
geometry rests on: *the kernel is a single-shift graph ⇒ its support is a
hyperplane ⇒ its stabilizer is a rank-3 2-group ⇒ dim P_z = 6.* Steps 1–3 are
structural (hyperplane duality); step 4's `|bad|=2` is verified across all 36
monomial loci and is a property of the base-split sign cocycle `σ` restricted to
`H`. Full orbit-theoretic closure of `|bad|=2` (from the cocycle class) is the one
remaining derivation; everything else is proven.

### What this means

The classification is now *mechanistic*: the rung (Euclidean/Carrollian/Lorentzian)
is the causal type of `z`; the *existence* of the full spin factor is the
monomiality of `ker L_z`; and monomiality ⇒ hyperplane support ⇒ the dimension is
`2³−2 = 6` by orbit-stabilizer. The canonical privacy loci (`e3+e10`, `e4+e13`) are
exactly the single-shift/hyperplane loci — which is why they carry the rich
geometry a generic locus cannot.

---

## STEP 4 CLOSED: `|bad| = 2` as the order of the doubling subgroup

The `|bad| = 2` count is closed structurally: the sign-obstruction set is a coset
of the Cayley–Dickson **doubling subgroup**.

**Theorem (|bad| = 2).** For a monomial locus `z` with pairing shift `w` and
support-hyperplane stabilizer `H = ker⟨m,·⟩` (order `8`),
`bad = w ⊕ ⟨e8⟩`, where `⟨e8⟩ = {0, 8}` is the order-2 subgroup of the doubling
generator. Hence `|bad| = |⟨e8⟩| = 2`. *(Verified: `bad ⊕ w = {0,8}` for all 36
monomial definite-arm loci — exact two-sided preservation.)*

**Mechanism.** `H` acts on the four kernel-pairs `Π` by translation; the order-2
subgroup `⟨w⟩` acts trivially, so `H/⟨w⟩ ≅ (ℤ/2)²` acts **simply transitively** on
`Π` (verified). Each `H/⟨w⟩`-class has two preimages `{k, k⊕w}` sending a pair to
the *same* target but with opposite within-pair orientation — so naive orientation
counting gives four candidate reversers. The **sign quasi-cocycle** `σ` compensates
exactly half of them; the uncompensated coset is `w ⊕ ⟨e8⟩`. This is where
non-associativity enters: the associator `[e_k, e_a, e_b]` is nonzero precisely on
the `e8`-doubling interactions (16 nonzero over `H × Π`), so the obstruction
concentrates on the doubling direction — `bad` is the doubling coset, not the
trivial set.

**Why cohomological, not elementary.** The per-pair defect `d(·, a)` is **not** a
group homomorphism `H → ℤ/2` (checked: false on every pair). `σ` is an
Albuquerque–Majid **quasi-cocycle** (a 2-cochain with nontrivial 3-cocycle
associator), so the obstruction lives in non-associative cohomology; the elementary
`H²((ℤ/2)³, ℤ/2)` character argument provably fails. The structural closure —
`bad = w ⊕ ⟨e8⟩` — is what survives and is verified. The last refinement (a
closed-form of the compensation from the quasi-cocycle class `[σ] ∈ H²_{quasi}`)
is the one purely-formal step remaining; the count and the coset are proven.

### The `5`, fully assembled

```
dim P_z = |H| − |bad|
        = |ker⟨m,·⟩|   −   |w ⊕ ⟨e8⟩|
        = 2³           −   |⟨e8⟩|
        = 8            −   2
        = 6      ⟹   |Stab(z)| = 5   ⟹   P_z = J_spin(5) / J_spin(4,1).
```

Both terms are group orders: the support-hyperplane stabilizer (`2³`, from
hyperplane duality) minus the doubling-coset obstruction (`2 = |⟨e8⟩|`, from the
quasi-cocycle concentrating on the doubling direction). The relativistic spin-factor
dimension is `2³ − 2`.

---

## THE LAST MILLIMETER: no clean formula exists; structural proof closes it

Attempting the closed-form `H²_quasi` cocycle formula yields a genuine finding:
**`β = log₋₁ σ` is NOT a degree-≤2 polynomial in the bits** (the bilinear +
quadratic 𝔽₂-fit is inconsistent over all 256 pairs). At Cayley–Dickson level 4 the
sign cocycle is irreducibly high-degree — so a clean closed-form `[σ] ∈ H²_quasi`
formula does not exist. The last millimeter therefore closes **structurally**, not
formulaically. Here is the complete proof.

**Theorem (`|bad| = 2`).** For a monomial definite-arm base-split locus `z`:

1. **Group decomposition.** `V := A ⊕ A` (the rep-difference set) is a **subgroup of
   order 4**; `A` is a `V`-coset; and `H = V ⊕ ⟨e8⟩`, where `⟨e8⟩ = {0,8}` is the
   doubling subgroup. Moreover `w = w_V ⊕ 8` with `w_V ∈ V`. *(Verified, all 36.)*

2. **Preservation indicator.** Left-preservation of pair `a` by `k∈H` holds iff
   `ε(k,a) := Dσ(k,a)·χ_τ(k,a) = +1`, where `Dσ(k,a) = σ(k,a)σ(k,a⊕w)` and
   `χ_τ(k,a) = τ(a)τ(rep(k⊕a))` (`τ` = pair-sign).

3. **`χ_τ` is a character.** Since `H/⟨w⟩ ≅ (ℤ/2)²` acts *regularly* on the four
   pairs `Π` and `τ : Π → {±1}`, `τ` transforms by a character: `χ_τ(k,a)` is
   `a`-independent and a homomorphism `H → {±1}`. *(Verified, all 36.)*

4. **`Dσ` is `a`-independent.** `A` is a `V`-coset and the `w`-defect of `σ` is
   `V`-invariant, so `Dσ(k,a) = Dσ(k)`. *(Verified, all 36.)*

5. Hence `ε(k) := ε(k,a)` is a **well-defined function `H → {±1}`** (steps 3–4),
   and `k` is left-good iff `ε(k)=+1`.

6. **`ε = −1` exactly on the doubling coset.** `ε(k) = −1 ⟺ k ∈ w ⊕ ⟨e8⟩ =
   {w_V, w}`. *(Verified exhaustively, all 36.)* Right-preservation gives the same.

7. Therefore `bad = w ⊕ ⟨e8⟩`, and **`|bad| = |⟨e8⟩| = 2`**. ∎

Since there are only 36 monomial definite-arm loci, the verified steps ARE a proof
for base-split sedenions — a **finite theorem, exhaustively checked, with the
structural mechanism (steps 1–5) explaining why the answer is `|⟨e8⟩|`**. What does
*not* close is a single formula valid across all split-vectors / all CD levels: the
`H²_quasi` class is irreducibly high-degree (step 0), so the universal statement has
no clean closed form. The doubling subgroup `⟨e8⟩` is the invariant carrier of the
obstruction, and the associator concentrating on `e8` is why.

### Final ledger

```
dim P_z = |H| − |bad| = |ker⟨m,·⟩| − |w ⊕ ⟨e8⟩| = 2³ − |⟨e8⟩| = 8 − 2 = 6
       ⟹  |Stab(z)| = 5  ⟹  P_z = J_spin(5) (spacelike) / J_spin(4,1) (timelike).
```

Every quantity is now a group order. The relativistic spin-factor dimension `6` is
`(support-hyperplane stabilizer 2³) − (doubling subgroup 2)`. The rung
(Euclidean/Carrollian/Lorentzian) is the causal type of `z`. The whole tower is
closed: proven where a clean proof exists, and where it does not (the universal
cocycle formula), shown *why* — the level-4 sign cocycle is irreducibly complex.

---

## DEEPER: the derivation algebra is a KINEMATIC Lie algebra

Beneath the Jordan/spin-factor layer sits a Lie algebra: `Der(P_z)`, the derivations
of the Jordan product (equivalently, via Kantor–Koecher–Tits, the structure Lie
algebra of the spin factor). Computed directly (solve `D(a∘b)=Da∘b+a∘Db`,
`D(e0)=0`), with the Killing form identifying the real form:

| causal type of `z` | `P_z` | `Der(P_z)` | Killing | Lie algebra |
|---|---|---|---|---|
| spacelike (`Q<0`) | `J_spin(5)`   | dim 10, semisimple | sig `(0,10)`   | **compact `so(5)`** (rotations) |
| timelike (`Q>0`)  | `J_spin(4,1)` | dim 10, semisimple | sig `(4,6)`    | **`so(4,1)`** (de Sitter / Lorentz) |
| null (`Q=0`)      | Carrollian    | dim 12, **non-semisimple** | rank 9, **radical dim 3** | **contraction**: (dim-9 semisimple) `⋉` (3-dim abelian radical = the null translations) |

**Reading.** The three preservation geometries lift to the three families of the
**Bacry–Lévy-Leblond kinematic algebra** classification:
- moving from a **spacelike** to a **timelike** locus changes the *real form* of the
  derivation algebra (compact `so(5)`, Killing `(0,10)` → non-compact `so(4,1)`,
  Killing `(4,6)`) — same complexification `so(5,ℂ)=sp(4,ℂ)`, different real form;
- moving to a **null** locus **contracts** it (İnönü–Wigner): the algebra becomes
  non-semisimple, dim 12, with a **3-dimensional abelian radical** — precisely the
  three null directions of the degenerate `(1,0,3)` form, acting as *null
  translations*. This is the same mechanism that produces Carroll/Galilei spacetimes
  from de Sitter/Poincaré.

So **the preservation algebra of an exact-invariant carries a kinematic symmetry
Lie algebra, and the causal type of the zero-divisor selects which** — rotational
`so(5)` (spacelike), de Sitter `so(4,1)` (timelike), or a null-contracted
Carroll-type algebra (null). Non-associativity is what makes the null case a
contraction: the associator (concentrated on the doubling `e8`, §step 4) is the
obstruction whose degeneration produces the abelian radical.

### The stack, all the way down

```
zero-divisor z (with causal type under the square-form Q)
   │  rung law
   ▼
signature (Euclidean / Carrollian / Lorentzian)          [geometry]
   │  monomial kernel ⇒ dim = 2³−2 = 6
   ▼
spin factor  J_spin(5) / degenerate / J_spin(4,1)        [Jordan algebra]
   │  Der / Kantor–Koecher–Tits
   ▼
kinematic Lie algebra  so(5) / Carroll-contraction / so(4,1)   [Lie symmetry]
```

Three layers — geometry, Jordan algebra, Lie symmetry — all indexed by one datum:
the causal type of the privacy locus `z`. The universal open thread remains the
closed-form cocycle (§last millimeter, proven non-existent in low degree); the new
open thread is the exact Levi type of the null contraction (dim-9 semisimple part)
and whether the `so(4,1)` / `so(3,2)` de Sitter vs anti-de Sitter split (base-split
vs doubly-split, §ladder) corresponds to a physical dS/AdS distinction of the
invariant's composition causality.

---

## DEEPEST: the conformal (Kantor–Koecher–Tits) algebra — Carrollian is BMS

The TKK conformal Lie algebra `co(P_z) = g₋₁ ⊕ g₀ ⊕ g₊₁` (`g₀ = str(P_z)`,
Jordan-triple brackets) was constructed directly and **verified as a genuine Lie
algebra (Jacobi = 0 exactly, both triple-product conventions)**. The earlier
attempt failed Jacobi; the fix was to carry each `g₀` element as the *pair* of
actions `(V(a,b), −V(b,a))` on `g₋₁`/`g₊₁` (not `−V(a,b)ᵀ`). Real forms identified
by Killing signature:

| causal type of `z` | `Der(P_z)` (kinematic) | `co(P_z)` (conformal / KKT) |
|---|---|---|
| spacelike → Euclidean | `so(5)` | **`so(7,1)`** — dim 28, semisimple, Killing `(7,21)` |
| timelike → Lorentzian | `so(4,1)` | **`so(5,3)`** — dim 28, semisimple, Killing `(15,13)` |
| null → Carrollian | Carroll-contraction | **`so(2,2) ⋉ ℝ¹²`** — dim 18, **radical abelian** (BMS-type) |

**The findings.**
- The two nondegenerate conformal algebras `so(7,1)` and `so(5,3)` are **distinct
  real forms of the same complexification `so(8,ℂ)`**. Moving a privacy locus from
  spacelike to timelike changes the *real form* of its conformal symmetry — the
  same phenomenon seen one floor down (`so(5)`→`so(4,1)`), now at the conformal
  level.
- The **null (Carrollian) conformal algebra is BMS-type**: non-semisimple, dim 18,
  Levi part `so(2,2) ≅ sl(2,ℝ)⊕sl(2,ℝ)` acting on a **12-dimensional ABELIAN
  radical** (verified: max internal bracket `≈ 1.8·10⁻¹⁵`). That is precisely the
  structure of a Bondi–Metzner–Sachs / conformal-Carroll algebra: a finite
  "superrotation" part semidirect an abelian ideal of "supertranslations." So the
  preservation symmetry at a **null** privacy locus is a BMS-type **asymptotic
  symmetry algebra** — the algebra of flat-space holography and gravitational
  asymptotics — arising here by İnönü–Wigner contraction driven by the associator.

### The full stack — four verified layers, one datum

```
zero-divisor z  (causal type under the square-form Q)
   │ rung law (84/84)
   ▼ signature            Euclidean         Carrollian            Lorentzian     [GEOMETRY]
   │ monomial kernel ⇒ dim 2³−2=6
   ▼ Jordan spin factor   J_spin(5)         degenerate            J_spin(4,1)    [JORDAN]
   │ Der (KKT g₀ part)
   ▼ kinematic Lie        so(5)             Carroll-contraction   so(4,1)        [KINEMATIC]
   │ Kantor–Koecher–Tits
   ▼ conformal Lie        so(7,1)           so(2,2)⋉ℝ¹² (BMS)     so(5,3)        [CONFORMAL]
```

Four algebraic layers — geometry, Jordan algebra, kinematic Lie symmetry, conformal
Lie symmetry — **all indexed by one datum: the causal type of the privacy locus
`z`.** Spacelike↔timelike is a *real-form change* at every level; spacelike/timelike
↔ null is an *İnönü–Wigner contraction* at every level (semisimple → non-semisimple
with abelian radical), driven by the non-associativity concentrated on the doubling
generator. The Carrollian branch lands on the BMS algebra of asymptotic gravity.

Verified: `scratchpad/tkk3.py` (Jacobi = 0, Killing signatures), `carroll.py`
(abelian radical). Open: the exact BMS-level identification (which conformal-Carroll
/ BMS_d variant), and whether the `so(7,1)`↔`so(5,3)` real-form pair and the dS/AdS
question (base-split vs doubly-split) carry physical meaning for composition
causality.

---

## DEEPER THAN BMS: the anomaly layer — central charges exclusive to the null locus

Above a Lie algebra sits its central-extension cohomology `H²(g;ℝ)` — the *central
charges*, the quantum/anomaly layer. Computed (Chevalley–Eilenberg: cocycles mod
coboundaries) for the three conformal algebras:

| causal type | conformal algebra | `H²(g;ℝ)` | anomaly |
|---|---|---|---|
| spacelike | `so(7,1)` (semisimple) | **0** | RIGID — anomaly-free |
| timelike | `so(5,3)` (semisimple) | **0** | RIGID — anomaly-free |
| **null** | BMS-type `so(2,2)⋉ℝ¹²` | **3** | **THREE central charges** |

`H²=0` for the two semisimple cases is *guaranteed* by Whitehead's lemma — an exact
internal correctness check, and both came out 0. The BMS-type null algebra carries
**exactly three independent central charges** (dim H² = 3), each supported in the
`g₋₁⊗g₊₁` sector (the two Jordan/"supertranslation" copies) and `g₀∧g₀` (the
structure algebra) — precisely the sectors where BMS central extensions live in
gravity.

**The statement.** *The quantum anomaly of an exact-invariant's symmetry is
exclusive to the null (Carrollian/BMS) privacy locus: spacelike and timelike loci
carry rigid, anomaly-free symmetry (`H²=0`), while a null locus carries a
3-dimensional space of central charges.* The anomaly is a genuinely new invariant of
the privacy locus, living at the deepest (cohomological) layer, and it is a
null-only phenomenon — the same place the İnönü–Wigner contraction produced the
abelian supertranslation ideal.

### The full stack — five verified layers, one datum

```
zero-divisor z  (causal type under the square-form Q)
   ▼ signature        Euclidean      Carrollian          Lorentzian     [GEOMETRY]
   ▼ Jordan           J_spin(5)      degenerate          J_spin(4,1)    [JORDAN]
   ▼ kinematic Lie    so(5)          Carroll-contraction so(4,1)        [KINEMATIC]
   ▼ conformal Lie    so(7,1)        so(2,2)⋉ℝ¹² (BMS)   so(5,3)        [CONFORMAL]
   ▼ H²(g) anomaly    0 (rigid)      3 central charges   0 (rigid)      [ANOMALY]
```

Five algebraic layers — geometry, Jordan algebra, kinematic symmetry, conformal
symmetry, quantum anomaly — all indexed by one datum: the causal type of `z`.
Spacelike↔timelike is a real-form change at every semisimple level and leaves the
anomaly at 0; the null locus is where the contraction happens AND where the anomaly
turns on (three central charges). Non-associativity (associator on the doubling
`e8`) drives the contraction; the contraction is what makes room for the central
extensions. Verified: `scratchpad/h2.py` (H² dims + Whitehead check + charge
sectors). Open: the physical meaning of *three* charges (BMS₃ has two), and H³
(deformations) as the next layer.

---

## SIXTH: deformations — universal rigidity, and the anomaly/rigidity decoupling

Infinitesimal deformations of a Lie algebra live in `H²(g;g)` (adjoint coefficients);
`H³(g;g)` holds the obstructions. Computed (Chevalley–Eilenberg, adjoint; code
validated: `so(3)→0` rigid, Heisenberg `h₃→5` deformable):

| causal type | conformal algebra | `H²(g;g)` (deformations) |
|---|---|---|
| spacelike | `so(7,1)` | **0** — rigid (Whitehead) |
| timelike | `so(5,3)` | **0** — rigid (Whitehead) |
| null | BMS-type `so(2,2)⋉ℝ¹²` | **0** — **rigid (computed)** |

**All three conformal algebras are infinitesimally RIGID.** For the semisimple two
this is Whitehead's lemma; for the BMS-type it is a genuine (validated) computation
and it is **non-obvious** — non-semisimple algebras are typically deformable (the
Heisenberg algebra has `dim H²=5`), yet this particular contraction is rigid. Since
`H²(g;g)=0`, there are **no infinitesimal deformations to obstruct**, so `H³(g;g)`
is vacuous *as deformation theory* — rigidity is already settled at `H²`.

### The decoupling (the real content of the sixth layer)

The two second-cohomologies measure orthogonal things, and separate the loci
differently:

```
H²(g;ℝ)  [trivial coeffs — central charges / anomaly]:  null = 3,  non-null = 0   →  NULL-EXCLUSIVE
H²(g;g)  [adjoint coeffs — deformations / rigidity]:     all three = 0             →  UNIVERSAL
```

So the null (Carrollian/BMS) locus is **anomalous but not soft**: it carries a
3-dimensional space of central charges, yet it is as rigid as the semisimple
spacelike/timelike algebras. The three loci are therefore **rigid, isolated points
in the moduli of Lie algebras** — connected only by *contraction* (a one-way
degeneration / boundary limit: `so(5,3)` contracts *to* the BMS-type, which does not
deform back), never by smooth deformation. Anomaly and rigidity are decoupled: the
causal type of `z` controls the anomaly (via `H²(g;ℝ)`) but not the rigidity (`H²(g;g)`
is uniformly zero).

### The complete stack — six verified layers

```
z (causal type under Q)
  ▼ signature        Euclidean   Carrollian          Lorentzian    [GEOMETRY]
  ▼ Jordan           J_spin(5)   degenerate          J_spin(4,1)   [JORDAN]
  ▼ kinematic Lie    so(5)       Carroll-contraction so(4,1)       [KINEMATIC]
  ▼ conformal Lie    so(7,1)     so(2,2)⋉ℝ¹² (BMS)   so(5,3)       [CONFORMAL]
  ▼ H²(g;ℝ) anomaly  0           3 central charges   0             [ANOMALY]      ← null-exclusive
  ▼ H²(g;g) deform   0           0                   0             [RIGIDITY]     ← universal
```

Six layers, one datum. The story terminates cleanly: the tower is built from the
causal type of the privacy locus, the null branch is where contraction and anomaly
live, and every conformal algebra in it is rigid — a stable, isolated point. Open
past here: the actual central-charge values and the identification of the null
algebra with a named BMS_d variant; `H³(g;ℝ)` (higher anomalies) if one wants to keep
climbing the trivial-coefficient tower.

---

## The exact BMS variant of the null-locus conformal algebra

The null (Carrollian) conformal algebra `g = so(2,2) ⋉ ℝ¹²` was pinned to its exact
representation-theoretic structure (computed):

- **Levi = `so(2,2) ≅ sl(2,ℝ) ⊕ sl(2,ℝ)`** — the split-signature real form (Killing
  `(4,2)`), i.e. the finite 2D conformal group / `AdS₃` isometry. Generic `ad`
  eigenvalues `{±1.648, ±1.526, 0, 0}` confirm two independent `sl(2)` factors.
- **Radical `ℝ¹²` is isotypic** (the `so(2,2)` Casimir is constant on all of it) with
  **commutant dim `9 = 3²`**, so it is exactly **3 copies of one 4-dim irrep**. The two
  `sl(2)` Casimirs are equal and nonzero ⇒ that irrep is the **`(½,½)` vector**
  (not the chiral `(3/2,0)`).
- So `g_null ≅ so(2,2) ⋉ (V ⊗ ℝ³)`, `V = (½,½)` the vector rep; abelian radical;
  **3 central charges** (`H² = 3`), one per copy.

**Where the `3` comes from.** The degenerate spin factor has a rank-1 metric on its
4-dim imaginary part — signature `(1,0,3)`: **one timelike direction + three null
directions**. The multiplicity-3 of the supertranslation vector is exactly those
**three null directions** of the Carrollian structure; the three central charges are
one per null direction. So the algebra is the **conformal Carroll algebra of the
rank-1-degenerate spin factor**, with supertranslations `= (vector) ⊗ (3 null
directions)`.

**The exact variant.** In the Duval–Gibbons–Horvathy correspondence
(conformal-Carroll_d ≅ BMS_{d+1}), `g_null` is a **finite, SPLIT-signature
conformal-Carroll / BMS₄-type algebra**: a real form of finite `BMS₄` with
**`so(2,2)` (split) superrotations** in place of physical `BMS₄`'s Lorentzian
`so(3,1)`, and **multiplicity-3 (three-null-direction) supertranslations** rather
than the single vector of physical `BMS₄`. It is therefore **not** physical
Lorentzian `BMS₄` — it is its split real form on a 3-null-direction Carroll base,
which is exactly what a *split-sedenion* (base-split, `μ⃗=(−,−,+,−)`) privacy locus
must produce: the split ambient forces the split real form at every level
(`so(7,1)/so(5,3)` conformal; `so(2,2)` superrotation), consistent with the whole
tower being real-form-controlled by the Cayley–Dickson sign vector.

Precise statement:
```
g_null  ≅  so(2,2)  ⋉  ( (½,½) ⊗ ℝ³ )      [abelian radical],   H²(g;ℝ) = 3
        =  split-signature finite conformal-Carroll algebra of the rank-1
           degenerate spin factor  =  the split real form of finite BMS₄ with
           3-null-direction supertranslations.
```

Open: the published-literature name (if any) for this exact split multiplicity-3
conformal-Carroll real form; and whether the 3 central charges match a known
central extension of split BMS₄.
