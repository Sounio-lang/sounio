<!-- docs:meta
topic_id: repo.docs.research.epistemic-calculus-value-carrying-redesign
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.epistemic-calculus-value-carrying-redesign
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Design: value-carrying `Knowledge<T>` for the epistemic-effect calculus

**Status:** design spec (2026-06-02). Motivated by a machine-checked soundness
gap: subject reduction is *false* for the current scalar-`KCell` calculus
(`formal/lean4/EpistemicEffects.lean`), because `Knowledge<T>` is typed
generically but represented by a real-valued cell.
Counterexample (`EpistemicPreservationWIP_counterexample.lean`, compiles):
`measure (lit_nat 0) k : Knowledge<ℕ> ⇒ kraw k`, and `kraw` can only be typed
`Knowledge<ℝ>`. The same mismatch recurs in `kvalue` once `kraw` is
type-indexed. Root cause: a **scalar cell cannot carry a value of arbitrary
`T`**. This spec makes the runtime cell value-carrying so generic
`Knowledge<T>` is sound.

## The core idea

A `Knowledge<T>` value is a **stored value of type `T`** together with scalar
GUM metadata (variance, confidence). The base type `T` is recovered by typing
the stored value — no separate type tag is needed, so `measure` needs no
annotation.

```
KMeta := { gumVar : Int, conf : Int }          -- scalar metadata only
kvalid (m) := 0 ≤ m.gumVar ∧ 0 ≤ m.conf ∧ m.conf ≤ 1000
```

## Syntax changes (`Expr`)

```
| measure : Expr → KMeta → Expr     -- measure(e, meta); e : T  ⇒  Knowledge<T>
| kraw    : Expr → KMeta → Expr     -- runtime Knowledge value: STORES a value of type T
```
(`measure` keeps a metadata arg; `kraw` now stores a value `Expr` + metadata.)

## Typing (`HasTy`)

```
[T-Kraw]     HasTy Γ v T emptyE → IsValue v → kvalid m
               → HasTy Γ (kraw v m) (tknow T) emptyE          -- T = type of stored value
[T-Measure]  HasTy Γ e T emptyE → kvalid m
               → HasTy Γ (measure e m) (tknow T) (singleE eObserve)
[T-KValue]   HasTy Γ e (tknow T) E → HasTy Γ (kvalue e) T E   -- returns T  ✅ generic
[T-KUnc]     HasTy Γ e (tknow T) E → HasTy Γ (kunc e) treal E -- variance: real
[T-KConf]    HasTy Γ e (tknow T) E → HasTy Γ (kconf e) treal E
[T-KAdd]/[T-KMul]   restricted to Knowledge<ℝ>:
               HasTy Γ a (tknow treal) E₁ → HasTy Γ b (tknow treal) E₂
               → HasTy Γ (kadd a b) (tknow treal) (E₁ ∪ E₂)
```
**Key:** GUM arithmetic (`kadd`/`kmul`) is *numeric* — it is restricted to
`Knowledge<ℝ>`, which is correct (you cannot GUM-add `Knowledge<bool>`).
`measure`/`kvalue`/`kunc`/`kconf` remain generic in `T`.

## Values (`IsValue`)

```
v_kraw : IsValue v → IsValue (kraw v m)        -- kraw is a value iff its payload is
```

## Operational semantics (`Step`)

```
meas_red   : IsValue v → measure v m ⇒ kraw v m           -- stores the value; T implicit ✅
kvalue_red : kvalue (kraw v m) ⇒ v                        -- returns stored value : T  ✅
kunc_red   : kunc  (kraw v m) ⇒ lit_real m.gumVar
kconf_red  : kconf (kraw v m) ⇒ lit_real m.conf
kadd_red   : kadd (kraw (lit_real x) ma) (kraw (lit_real y) mb)
               ⇒ kraw (lit_real (x+y)) (gumAdd (x,ma) (y,mb))
kmul_red   : kmul (kraw (lit_real x) ma) (kraw (lit_real y) mb)
               ⇒ kraw (lit_real (x*y)) (gumMul (x,ma) (y,mb))
-- congruence rules unchanged; kraw payload is closed so it is inert under eval contexts
```
`gumAdd`/`gumMul` now take `(value, meta)` pairs (the variance rules need the
operand values for the product case): `gumVar(a·b) = y²·σ²ₐ + x²·σ²_b`.

## Impact on the existing Lean (`EpistemicEffects.lean`)

| Piece | Change |
|---|---|
| `Expr.measure`, `Expr.kraw` | signatures (above) |
| `KCell` → `KMeta` | drop `value` field; keep `gumVar`,`conf` |
| `gumAdd`/`gumMul`, §10/§11 thms | operate on `(Int × KMeta)`; conservativity/monotone restated on the real value |
| `HasTy` (5 rules) | as above; `[T-KAdd]/[T-KMul]` restricted to `treal` |
| `IsValue.v_kraw` | takes the payload-value hypothesis |
| `shift`/`subst` | `kraw v m ↦ kraw (shift/subst … v) m` — kraw is no longer a leaf |
| committed Progress proof | `genKraw`/`canon_know`/`progress'` kraw+measure+kadd/kmul cases |
| **effect machinery + substitution lemma** (`EpistemicPreservationWIP.lean`) | **survives**; only the `kraw` case moves from leaf to a one-line recursive case. The payload is a closed value, so `wellScoped` keeps its shifts inert. |

## What this buys

- `kvalue : Knowledge<T> → T` is **sound for every `T`** (returns the stored value).
- `meas_red` needs **no type annotation** (the value carries `T`).
- Subject reduction holds for the generic calculus; the WIP substitution lemma
  applies almost unchanged.
- The paper's §3/§5 (`[T-Kraw]`, `[T-KValue]`, `[E-Measure]`) become faithful to
  a value-carrying runtime, and §5.4's prose [E-Measure] is corrected.

## Execution decisions (advisor-vetted 2026-06-02)

- **New module, not in-place.** Build the value-carrying calculus in a fresh
  `formal/lean4/EpistemicEffectsV2.lean`, prove full type-safety there as a
  clean unit, math-review the whole module, then swap the paper's citation and
  deprecate the old file in a *separate* reviewed step. The committed,
  math-reviewed Progress proof (`a04efed98`) stays green throughout. This
  avoids rewriting committed work and colliding with parallel agents on the
  shared file.
- **`kraw` payload is a recursive sub-term with an IH** in `shift`/`subst` and
  every metatheory lemma — *not* shortcut as "closed, therefore inert."
  `[T-Kraw]` admits `HasTy Γ v T emptyE` with `v` typed in a non-empty `Γ` (a
  `lam` value captures context vars); closedness holds only of
  runtime-*arising* `kraw`s. So treat `kraw` like `lam`/`app`.
- **Re-derive Progress + restate §10/§11 in V2.** The `gumAdd/gumMul`
  restructure to `(value, meta)` makes conservativity/monotonicity new math
  claims; math-review the module whole. §5.4 prose [E-Measure] correction and
  the "`kadd`/`kmul` is ℝ-only" disclosure ship with the swap.
- **Calibration.** §5.4 type-safety is standard table-stakes rigor; the *novel*
  SOTA core is §6 (a compiler applying its epistemic type system to its own
  source). This re-architecture must not crowd out formalizing the
  self-application property. A crisp design + banked artifacts (counterexample,
  Progress proof, substitution machinery) is a clean checkpoint; the V2 build is
  a fresh focused effort, not something to force-complete under context
  pressure.

## Honest scope

This is a **new calculus** relative to the committed one — a multi-step
refactor of the shared `EpistemicEffects.lean` (which is cited by the paper and
may be touched by parallel agents). It must be implemented and re-verified
(math-review) as a unit; the committed Progress proof and the §10/§11 theorems
are updated in the same change. `kadd`/`kmul` being `ℝ`-only is a deliberate,
disclosed restriction (GUM arithmetic is numeric), not a soundness dodge.
