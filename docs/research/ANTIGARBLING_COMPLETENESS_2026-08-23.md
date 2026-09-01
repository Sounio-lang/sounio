<!-- docs:meta
topic_id: repo.docs.research.antigarbling-completeness-2026-08-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.antigarbling-completeness-2026-08-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

> **CORRECTION / SUPERSEDED IN PART — 2026-09-01.** The two-certificate soundness below is now a
> THEOREM in a typed calculus over every Cayley–Dickson algebra (`formal/lean4/EpistemicEffectsNSA.lean`,
> `reassoc_sound`), and the "exactly two structural axes" claim (C) is **corrected**: it holds for
> propagators that carry per-source *sensitivity vectors*; for *variance-only* propagators (the GUM
> shortcut `‖y‖²·Var x + ‖x‖²·Var y`, i.e. `ep_mul`) there is a THIRD structural axis —
> norm-multiplicativity — free on ℝ,ℂ,ℍ,𝕆 (Hurwitz) and FALSE on the sedenions (kernel-checked
> understatement 4 vs 8 with disjoint sources and no re-association). See
> `ANTIGARBLING_FUSION_THEOREM_2026-09-01.md`.

# Anti-Garbling Completeness: fabricated precision as the kernel of the free-algebra quotient

**Date:** 2026-08-23 · **Author:** fable-1 (agent=claude), ns-antigarbling-wire lane
**Status:** research note — the two-orthogonal-axes thesis, a falsifier set, and one
actionable soundness hardening for the live N2 encoding. **Machine-verified
(2026-08-23):** the full first-order proof spine is checked in
`docs/research/lean/SounioAntiGarblingModel.lean` (Lean 4.33.0, exit 0, zero
warnings, `sorry`-free) — both certificate lemmas, both axes of Theorem 4.1 with the
associator sensitivity identity, the completeness dimension count (Prop 2), and the
§5 non-separability caveat. See the proof companion §7 for the theorem map.
**Extends:** the NS anti-garbling wire (`self-hosted/check/noise_sets.sio`,
E230) and the associator-variance corpus (`stdlib/epistemic/product_nonassoc.sio`,
`order_spread_exact.sio`, `docs/research/variance_of_associator.md`).
**Feeds:** the CEI program (`silly-enchanting-newt.md`) — this names the two
structural side-conditions any uncertainty-effect handler must discharge.

---

## 0. The claim in one sentence

> **Fabricated precision — a compiler or library reporting *less* uncertainty than
> the epistemic object actually carries — arises, for product/bilinear
> propagation over a normed algebra, from exactly two structural sources:
> *support garbling* (assuming independence across shared measurement sources) and
> *order garbling* (assuming associativity in a non-associative algebra). Both are
> the same act — imposing an algebraic identity the object does not satisfy — and
> both are the kernel of one projection.**

NS already catches the first. The associator already measures the second. The new
content is that **they are two halves of one object**, that a covariance-sound
type system must carry a *checked certificate* for each, and that this dictates
the lattice orientation the live N2 encoding must adopt to be fail-closed.

---

## 1. The two garblings, precisely

Work in the standard **affine / noise-symbol model** (Stolfi–de Figueiredo,
generalised): a value is
`x = x₀ + Σ_k dₖ·εₖ`, where each `εₖ` is an independent unit noise symbol
identifying one measurement source; `Var(x) = Σ_k dₖ²`,
`Cov(x,y) = Σ_k dₖeₖ`.

**Support garbling.** `Var(x·y)` (or `x+y`) under the *independence* assumption
drops the cross term `2·Cov(x,y) = 2·Σ_{k∈supp(x)∩supp(y)} dₖeₖ`. This is sound
**iff `supp(x) ∩ supp(y) = ∅`** — the operands share no source symbol.
- This is *exactly* the NS certificate. `noise_set_id` interns `supp(x)` as a bit
  mask; `ns_disjoint` tests `supp(x) ∩ supp(y) = ∅`; E230 rejects `ep_add`/`ep_mul`
  when the supports are non-disjoint **or unknown**. NS is the *support*
  half of anti-garbling.

**Order garbling.** For a **non-associative** product, `(x·y)·z ≠ x·(y·z)`. Any
propagation that reports a single value for `x·y·z` has silently chosen one
parenthesization and dropped the **associator** `[x,y,z] = (x·y)·z − x·(y·z)`. The
verified augmentation `Var_total = Var_GUM + κ·‖[x,y,z]‖²`
(`product_nonassoc.sio`; Fano triple → 0.25, non-Fano → 4.25 at κ=1) is the
certificate that no order-freedom was hidden. This is the *order* half.

Neither subsumes the other. **Disjoint supports do not make a product associative**
(octonion basis elements have disjoint "sources" yet a non-vanishing associator),
and **associativity does not decorrelate shared sources**. They are orthogonal.

---

## 2. The unifying picture — two *orthogonal structural axes*

The honest home of an uncertain value carries two independent pieces of
structure, and standard propagation makes a simplifying assumption on **each**.
Getting the mathematics exactly right means keeping them separate — they live in
different objects, which is *why* neither garbling subsumes the other.

**Axis 1 — the multiplication (how products associate).** Let `F` be the **free
magma algebra** over the noise symbols `{εₖ}` (products with no associativity
imposed). GUM computes in the associative quotient `A = F / ⟨associators⟩`; the
projection `q : F → A` has kernel the two-sided **associator ideal** (standard:
the free associative algebra is `F` modulo associators). Understatement along this
axis = the (variance-)norm of the dropped associator component. **This axis is
non-trivial iff the algebra is non-associative** (octonions, sedenions).

**Axis 2 — the noise measure (how sources correlate).** Independence is not a fact
about the *multiplication* but about the *second-moment form* `⟨εᵢ,εⱼ⟩ = E[εᵢεⱼ]`.
GUM assumes this form is **diagonal** (`⟨εᵢ,εⱼ⟩ = δᵢⱼ`). Understatement along this
axis = the dropped off-diagonal `2·Σ_{k∈supp(x)∩supp(y)} dₖeₖ` — non-zero exactly
when the operand **supports overlap**.

These axes are **orthogonal by construction**: axis 1 is a property of `·`
(associator ideal), axis 2 is a property of `⟨·,·⟩` (off-diagonal covariance). A
product can be associative with correlated sources (axis-2 garbling only), or
non-associative with disjoint sources (axis-1 garbling only) — octonion basis
elements are the second case. That orthogonality is the precise reason NS and the
associator are **independent, both-required** certificates:

| Axis | Structure | Assumption imposed | Garbling | Sounio certificate |
|---|---|---|---|---|
| **1 multiplication** | free magma → associative | associativity | **order** (non-associativity) | associator norm `κ‖[x,y,z]‖²` |
| **2 noise measure** | second-moment form | diagonality (independence) | **support** (correlation) | **NS** disjointness (`ns_disjoint`, E230) |

**Completeness conjecture (C), sharpened.** For **product/bilinear** propagation
over a normed algebra, axis 1 and axis 2 are the *only two structural axes* of
identity-induced understatement: if every product site carries (i) an
NS-disjointness certificate (axis 2) and (ii) an associator-vanishing-or-
augmentation certificate (axis 1), the reported variance is a sound
over-approximation — **no third structural leak exists.** The claim is a
*dimension count*: bilinear propagation touches exactly the algebra's associativity
and the measure's diagonality, and nothing else structural. (Curvature is the
*next order*, not a third structural axis — §2.1.)

### 2.1 Why exactly two (the proof obligation, and its boundary)

A bilinear map `p(x,y)` is fixed by its action on generator pairs
`p(εᵢ,εⱼ)`. `Var(p(x,y))` is a quadratic form in the coefficients, and its value
depends on the propagator only through (a) how `p(εᵢ,εⱼ)` reassociates in a longer
chain — **axis 1**, the associator — and (b) the pairing `⟨p(εᵢ,εⱼ), p(εₖ,εₗ)⟩`,
which for a normed algebra factors through `⟨εᵢεⱼ, εₖεₗ⟩` and hence through the
second-moment form — **axis 2**. There is no third input to a *bilinear* value's
variance. **This makes (C) plausibly a theorem, not merely a conjecture**, for the
bilinear class — and it sharply names its boundary: `p` **non**-bilinear (a
transcendental map, `exp`, a ratio) introduces higher moments `E[εᵢεⱼεₖ]` and
curvature, which is the `C_H` axis (exact only under MC/p-box), *outside* (C) by
construction. (C) is therefore a completeness statement **for the bilinear
fragment**, with curvature explicitly ceded to the approximation lane.

**Boundary — what C deliberately excludes.** *Curvature / nonlinearity* (GUM
first-order under-covering a convex map) is **not** an identity garbling; it is an
*approximation-quality* axis, handled by a separate side-condition `C_H` (the
per-handler curvature bound in the CEI plan) with Monte-Carlo / p-box as the exact
fallbacks. C is a statement about *dropped algebraic identities*, not about
linearization error. Keeping these lanes separate is the honesty gate: NS +
associator certify *structure*; `C_H`/MC certify *shape*.

---

## 3. Why this matters for the live N2 encoding — the lattice-orientation lemma

The free-algebra picture forces a soundness requirement the grok-4.6 review
circled from the other side. NS is a **may-analysis of support**: to never falsely
certify independence, it must **over-approximate** `supp(x)` — when in doubt, the
support is "everything," so nothing is provably disjoint.

**Lemma (orientation).** For a may-support analysis to be covariance-sound, the
**⊤ (unknown/"all sources") element must be both the join-absorbing element and
the default for any unseeded value.** The empty set ⊥ (deterministic) is the join
*identity* and the *most permissive* disjointness answer, so ⊥ must **never** be
the value a value gets *by default*.

The current NS encoding gets the *operational* semantics right — `ns_union`
absorbs at `-1`, `ns_disjoint(-1, ·)=false`, `ns_unknown()` seeds unseeded
`TypeEntry` — but the *representation* is orientation-inverted: `ns_empty()=0`,
so the **memset/zero default of any `i64` field is ⊥ (empty), the most permissive
answer.** Soundness then rests entirely on *every* construction, copy, join, and
call-summary site explicitly overwriting the field with `-1`; a single unseeded
path is silently **fail-open** (`ns_disjoint(t, 0, S)=true` — a value certified
independent of everything). The grok-4.6 review flagged exactly this at
`ns_deref` (⊤/invalid/empty all → mask 0) and at the zero-default.

**Orientation-robust fix (recommended for N2/N3):** make **`0 = ⊤` (unknown)** the
zero-value, and reserve a distinct sentinel for ⊥ (empty/deterministic). Then the
memset-zero default is *fail-closed by construction* and NS soundness no longer
depends on exhaustive seeding discipline — it depends only on the *few* sites that
deliberately narrow a value to a known singleton/empty. This is the single
highest-leverage hardening the theory implies, and it is behaviour-neutral for
every already-seeded path.

---

## 4. Consequence for CEI — the two mandatory handler side-conditions

In the CEI thesis (`silly-enchanting-newt.md`), an uncertainty-effect handler `H`
is *sound* iff its `collapse` interval over-approximates the reference. Anti-Garbling
Completeness names the **two structural obligations** inside `Sound_H`:

1. **Support obligation** — for every `perform Epistemic.{add,mul}(x,y)`, either
   `supp(x) ∩ supp(y) = ∅` (NS certificate) or `H` adds the covariance correction
   `2·Cov(x,y)`.
2. **Order obligation** — for every non-associative `perform Epistemic.mul` chain,
   either the associator vanishes (associative/Fano certificate) or `H` adds the
   order-spread `κ·‖[·,·,·]‖²` (its exact N=4 form is the associahedron K₄ pentagon
   variance, `order_spread_exact.sio`).

A handler that discharges both is *anti-garbling sound*. This is what makes the
non-associative handler (WS-C) the discriminator: it is the **only** handler for
which obligation (2) is non-trivial, and no competing system (`Uncertain⟨T⟩`,
`Measurements.jl`, Stan) even represents it.

---

## 5. Proven vs conjectured (evidence discipline)

**Proven / execution-verified (do not re-litigate):**
- `Var_total = Var_GUM + κ‖[a,b,c]‖²` and the associator = order-spread at N=3
  (`product_nonassoc.sio`: 0.25 / 4.25).
- N=4 order-spread is the associahedron K₄ pentagon variance, parenthesization-
  independent (`order_spread_exact.sio` ≈ 2.0442), and the perturbation DAG is
  order-safe **iff N≤3** (`perturbation_graph.sio`).
- NS disjointness ⟺ zero shared interned bit; the encoding invariants
  (handle/deref/union/disjoint) are grok-4.6-reviewed
  (`NS_N1_GROK46_MATHREVIEW_2026-08-23.md`); `ns_handle_validity` 9/9.

**Conjectured (this note):**
- **(C)** completeness — support + order are the *only* two identity-garblings for
  bilinear propagation over a normed algebra.
- The free-algebra framing: fabricated precision = variance-norm of `ker q`, with
  `ker q` generated by the shared-support and associator families. (The
  ring-theoretic statement "the associative-commutative quotient of a free magma
  algebra is killed by the commutator+associator ideal" is standard; the *novel*
  step is the **epistemic identification** of those two ideal families with the
  two garblings, via the affine/covariance model.)

**Falsifiers (each kills a specific claim):**
- **F1 (kills C):** a product-only program with **disjoint** NS supports **and**
  an **associative** product whose GUM variance still *understates* the exact
  (MC/interval) variance by a structural (non-curvature) term. Search the darwin
  PBPK corpus and the octonion demos; if none exists after an adversarial sweep,
  C stands for that class.
- **F2 (kills the order half):** an order-garbling at N=3 **not** captured by the
  associator norm — refuted (proven identity).
- **F3 (kills the support half):** a shared-source correlation that
  `ns_disjoint`=∅ fails to reject — this is exactly the N3 negative-witness
  `ns_add_shared_source_rejected.sio` (E230); if it ever passes, the support
  certificate is unsound.
- **F4 (kills the orientation lemma):** a covariance-sound NS run where the
  zero-default ⊥ is *never* narrowed and yet no fabricated independence occurs —
  would show the orientation fix is unnecessary (expected to fail: the sabotage
  gate `ns_antigarbling_gate.sh` should exhibit a fail-open on an unseeded path
  under the current 0=empty encoding).

---

## 6. Immediate, buildable next steps (no new claims beyond §5)

1. **Formalise the anchor — DONE (machine-checked).**
   `docs/research/lean/SounioAntiGarblingModel.lean` now exists and typechecks
   clean under Lean 4.33.0 (`sorry`-free): the affine model, both
   *soundness-of-certificate* lemmas (NS-disjoint ⇒ Cov term absent;
   associator-vanish ⇒ order term absent), both axes of Theorem 4.1 with the §3(B)
   associator sensitivity identity, the completeness dimension count (Prop 2), and
   the §5 non-separability caveat. (C) is verified as a theorem **for the
   first-order structural model**; the F1 falsifier stands for the general-algebra
   claim, which the structural Lean model instantiates. See the proof companion §7.
2. **Adopt the orientation fix** (0=⊤) in the N2 encoding if the `ns_antigarbling_gate.sh`
   sabotage control exhibits an unseeded fail-open — turning a
   seed-discipline-dependent soundness into an encoding-guaranteed one.
3. **Wire the order obligation** alongside the support obligation at the
   `ep_mul` site for non-associative operands (the associator certificate), so a
   single E230 family covers *both* garblings — the first checker in existence
   that rejects correlation-fabrication and order-fabrication under one rule.
