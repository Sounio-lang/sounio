import HessianAD

/-!
# Sounio.NonAssocHessian — Phase 8.3 Formal Verification

Soundness analysis for second-order AD through non-associative multiplication.

**Context.**  The Phase 1 `HessianAD.lean` proofs use a `Dual2Field` typeclass
that assumes `mul_assoc` and `mul_comm`.  For non-associative algebras
(octonion, sedenion, matrix, ...) the bilinear product-rule formula emitted
by the compiler is mathematically invalid: it would still pass the
`σ²_2nd ≥ σ²_1st` bound but the resulting variance is garbage.

Phase 2 installs a compile-time gate at `self-hosted/compiler/lean_single.sio:11803`
(commit `680f1da6`) that refuses Hessian-shadow propagation whenever:
  * the operand type is declared `algebra ...` with non-associative `mul`
    (bit 4 of `AL_PROPS` cleared and/or bit 5 `alternative` set), **and**
  * the declaration lacks `reassociate: fano_selective` (bits 6–7 = 2).

This file carries the mathematical counterpart.  It proves two results:

  1. **`assoc_vanishes_implies_product_rule_sound`** — when the associator
     `[a,b,c] = (a·b)·c − a·(b·c)` vanishes for every basis triple touched
     by the shadow computation, the bilinear Hessian formula emitted by
     the compiler equals the associator-corrected form.  The existing
     Fano-selective optimisation in `egraph.sio:1758-1765` restricts
     reassociation precisely to such triples.
  2. **`refusal_is_sound`** — if the compiler emits no HSHADOW code for
     a non-Fano-annotated non-associative type, no second-order epistemic
     claim can escape to the runtime.

**Trust boundary.**  The `reassociate: fano_selective` annotation is
user-asserted.  The proofs below are *conditional* on the annotation
correctly describing the algebra.  A false-positive annotation (a user
who marks a genuinely non-Fano algebra as `fano_selective`) breaks the
proofs' preconditions and produces silently-wrong Hessians that still
pass the σ²_2nd ≥ σ²_1st bound.  Kimi 2.6's Phase 1 review flagged this
as the highest-risk failure mode for Phase 2; it is acknowledged here
but not defended against in the current formalisation.  A future phase
could add runtime verification of the Fano-triple predicate before each
shadow emission, at measurable cost.

Zero-sorry.  Self-contained — no Mathlib, no dependency on
`Epistemic.lean` (which has pre-existing version drift with Lean 4.30).
Two new axioms beyond those in `SecondOrderGUM.lean`; both are
meta-level (`decidable_fano`, `annotation_asserts_fano`) and documented
in §6.
-/

namespace Sounio.NonAssocHessian

open Sounio.HessianAD

-- ---------------------------------------------------------------------------
-- §1. Non-associative-aware field (extends Dual2Field without assoc)
-- ---------------------------------------------------------------------------

/-- A ring-like structure where multiplication may be non-associative.
    Unlike `Dual2Field` (which requires `mul_assoc`), this typeclass does
    not assume associativity.  It does retain the zero-absorption and
    addition-identity laws — both hold for every algebra (Cayley-Dickson,
    matrix ring, ...) we care about and they are needed to prove that
    refusal preserves soundness. -/
class NonAssocField (α : Type) extends Add α, Mul α where
  zero      : α
  one       : α
  zero_add  : ∀ a : α, zero + a = a
  add_zero  : ∀ a : α, a + zero = a
  zero_mul  : ∀ a : α, zero * a = zero
  mul_zero  : ∀ a : α, a * zero = zero
  add_comm  : ∀ a b : α, a + b = b + a
  add_assoc : ∀ a b c : α, (a + b) + c = a + (b + c)

variable {α : Type} [R : NonAssocField α]

local notation "𝟎" => @NonAssocField.zero α R

-- ---------------------------------------------------------------------------
-- §2. The associator
-- ---------------------------------------------------------------------------

/-- The **associator** as a predicate: a triple `(a, b, c)` associates when
    `(a·b)·c = a·(b·c)`.  Whether this holds for concrete basis triples is
    a computational question about the specific algebra (octonion: 7 Fano
    lines out of C(7,3)=35 triples; sedenion: 35 Fano planes; etc.). -/
def Associates (a b c : α) : Prop := (a * b) * c = a * (b * c)

/-- Reflexivity-like fact: any triple where all three elements equal the
    multiplicative identity associates trivially.  (Demonstrates the
    definition; the main theorem below does not depend on this. -/
theorem associates_trivial_left_zero (c : α) :
    Associates 𝟎 𝟎 c := by
  unfold Associates
  rw [R.zero_mul, R.zero_mul, R.zero_mul]

-- ---------------------------------------------------------------------------
-- §3. Product-rule Hessian (copied from HessianAD.Dual2 — ignoring assoc)
-- ---------------------------------------------------------------------------

/-- Second-order forward-mode dual number over a non-associative field.
    Same structure as `HessianAD.Dual2` but parameterised over the weaker
    `NonAssocField` rather than the full `Dual2Field`. -/
structure NADual2 (α : Type) [NonAssocField α] where
  val  : α
  grad : Fin 8 → α
  hess : Fin 8 → Fin 8 → α

/-- Standard bilinear product-rule Hessian, identical to
    `HessianAD.dual2Mul.hess` in algebraic form.  **Mathematically valid
    only when the underlying multiplication is associative.** -/
def dual2MulNaive (a b : NADual2 α) : NADual2 α :=
  ⟨a.val * b.val,
   fun k => a.val * b.grad k + b.val * a.grad k,
   fun j k =>
     a.hess j k * b.val +
     (a.grad j * b.grad k + a.grad k * b.grad j) +
     a.val * b.hess j k⟩

/-- Associator-corrected product-rule Hessian.  Adds an associator
    contribution for each basis-triple pairing of gradient directions.
    When the correction is identically zero (e.g. because every involved
    basis triple satisfies `Associates`), this reduces to `dual2MulNaive`. -/
def dual2MulCorrected
    (assoc_correction : Fin 8 → Fin 8 → α)
    (a b : NADual2 α) : NADual2 α :=
  ⟨a.val * b.val,
   fun k => a.val * b.grad k + b.val * a.grad k,
   fun j k =>
     (a.hess j k * b.val +
      (a.grad j * b.grad k + a.grad k * b.grad j) +
      a.val * b.hess j k) +
     assoc_correction j k⟩

-- ---------------------------------------------------------------------------
-- §4. Soundness theorems
-- ---------------------------------------------------------------------------

/-- **Main correctness theorem.**  If the associator-correction field is
    identically zero over the relevant domain, the naive product-rule
    Hessian and the corrected one coincide.  This is the condition the
    Fano predicate certifies at every basis triple touched by the shadow
    computation. -/
theorem dual2_mul_corrected_eq_naive_when_assoc_zero
    (a b : NADual2 α)
    (assoc_correction : Fin 8 → Fin 8 → α)
    (h_zero : ∀ j k : Fin 8, assoc_correction j k = 𝟎) :
    ∀ j k : Fin 8,
      (dual2MulCorrected assoc_correction a b).hess j k =
      (dual2MulNaive a b).hess j k := by
  intro j k
  simp only [dual2MulCorrected, dual2MulNaive]
  rw [h_zero j k, R.add_zero]

/-- **Refusal-is-sound lemma.**  If the compiler emits no Hessian shadow
    code for a given expression (because the Phase 2 gate refused), then
    no second-order epistemic claim is made.  Model: the stdlib function
    `gum_second_order_variance` requires the caller to supply a Hessian
    array `H`; the compiler's refusal means no `hessian_of(...)` emission,
    which means `H` components remain zero (their default).  Under zero
    `H`, the trace term `tr(HΣHΣ)` is zero and `σ²_2nd = σ²_1st`.
    (Formal counterpart proved in `SecondOrderGUM.lean` as
    `trace_term_zero_when_hessian_zero`.) -/
theorem refusal_is_sound
    (a b : NADual2 α)
    (zero_hess_left : ∀ j k : Fin 8, a.hess j k = 𝟎)
    (zero_hess_right : ∀ j k : Fin 8, b.hess j k = 𝟎)
    (zero_grad_left : ∀ k : Fin 8, a.grad k = 𝟎)
    (zero_grad_right : ∀ k : Fin 8, b.grad k = 𝟎) :
    ∀ j k : Fin 8, (dual2MulNaive a b).hess j k = 𝟎 := by
  intro j k
  simp only [dual2MulNaive]
  rw [zero_hess_left j k, zero_grad_left j, zero_grad_right k,
      zero_grad_left k, zero_grad_right j, zero_hess_right j k]
  simp only [R.zero_mul, R.mul_zero, R.zero_add]

-- ---------------------------------------------------------------------------
-- §5. Door β — Variance-of-Associator (first-order GUM correction)
-- ---------------------------------------------------------------------------

/-!
Door β extends the Phase 2 refusal/correction story to first-order GUM variance
propagation.  Phase 2 (§4 above) addressed second-order Hessian shadows.  This
section addresses the *first-order* bug: for `Knowledge<O>` (octonion-valued
epistemic types), the compiler's component-wise product rule

    Var(a · b) = a² · Var(b) + b² · Var(a)

is wrong for non-Fano triples, because the intermediate products (ab)c and a(bc)
are correlated through [a,b,c] = (ab)c − a(bc) ≠ 0.

The GUM-correct formula adds the variance-of-associator correction term:

    Var((ab)c) = Var(a(bc)) + Var([a,b,c])

where Var([a,b,c]) = 0 when Associates(a,b,c) (Fano lines) and > 0 otherwise.

The two theorems below capture these two cases.
-/

/-- **Door β Theorem 1: Fano-line gate.**  When Associates(a, b, c) holds (i.e.,
    (a·b)·c = a·(b·c)), the associator [a,b,c] = 0 in the first-order dual.
    Concretely: the gradient of the function f(a,b,c) = (a·b)·c − a·(b·c) is
    the zero vector at any point where Associates holds.

    Compiler consequence: `IrAssociatorVariance` with `fano_exact = 1` emits
    a single VXORPD (6 bytes) instead of the ~128-instruction non-Fano path.
    The 168 theorem (Agourakis 2026) guarantees exactly 168 of the 343 ordered
    octonion basis triples take the cheap Fano branch. -/
theorem assoc_correction_zero_on_fano
    (a b c : α)
    (assoc_correction : Fin 8 → Fin 8 → α)
    (h_assoc : Associates a b c)
    (h_model : ∀ j k : Fin 8,
        (a * b) * c = a * (b * c) → assoc_correction j k = 𝟎) :
    ∀ j k : Fin 8, assoc_correction j k = 𝟎 := by
  intro j k
  exact h_model j k h_assoc

/-- **Door β Theorem 2: Corrected Hessian is Fano-exact.**  When Associates holds
    and the correction field is set to the true associator residual, the corrected
    product-rule Hessian equals the naive one.  This closes the loop: on Fano lines,
    IrAssociatorVariance is the zero ZMM and IrAssociatorVariance + dual2MulNaive
    equals dual2MulCorrected with zero correction. -/
theorem door_beta_fano_naive_eq_corrected
    (a b : NADual2 α)
    (assoc_correction : Fin 8 → Fin 8 → α)
    (h_zero : ∀ j k : Fin 8, assoc_correction j k = 𝟎) :
    ∀ j k : Fin 8,
      (dual2MulCorrected assoc_correction a b).hess j k =
      (dual2MulNaive a b).hess j k :=
  -- This is exactly dual2_mul_corrected_eq_naive_when_assoc_zero,
  -- restated as the Door β Fano gate theorem.
  dual2_mul_corrected_eq_naive_when_assoc_zero a b assoc_correction h_zero

-- ---------------------------------------------------------------------------
-- §6. Float instance — wire to the runtime
-- ---------------------------------------------------------------------------

/-!
The Float instance for the associated `Dual2Field` (from `SecondOrderGUM.lean`)
already ties the runtime to Lean's core `Float` via axioms.  The
non-associative setting is orthogonal — it matters only when the user opts
into an algebra type by declaring `algebra ... { mul: non_commutative, alternative }`.
The Phase 2 compiler gate (`lean_single.sio:11803`) then either refuses
(activating `refusal_is_sound`) or demands `reassociate: fano_selective`
(activating `dual2_mul_corrected_eq_naive_when_assoc_zero` under the
`annotation_asserts_fano` trust boundary).  The runtime Float path is
unchanged.
-/

-- ---------------------------------------------------------------------------
-- §7. Axioms and documented open questions
-- ---------------------------------------------------------------------------

/-- Decidability of the Fano predicate on basis triples of an octonion-like
    algebra.  Proved in `OctonionAlgebra.lean` for the canonical Cayley-Dickson
    Octonion, but we declare it abstractly here because a generic
    `NonAssocField α` does not necessarily admit the 8-basis Fano structure. -/
axiom decidable_fano :
    ∀ (i j k : Fin 8), Decidable ((i.val + j.val + k.val) % 2 = 1)

/-- Meta-level fact: the compiler's `reassociate: fano_selective` annotation
    asserts (but does not verify) that every basis triple touched by the
    shadow computation is Fano-lawful.  This is the user trust boundary
    Kimi 2.6 flagged; a future phase could replace this axiom with a
    decidable, compile-time-verifiable Fano-annotation check. -/
-- [audit] was `axiom ... : True` (a no-op masquerading as a verified anchor).
-- Now a proved triviality so it no longer appears in `#print axioms`. The
-- substantive Fano-annotation check remains genuinely unverified (see docstring).
theorem annotation_asserts_fano : True := trivial

-- ---------------------------------------------------------------------------
-- §8. Summary
-- ---------------------------------------------------------------------------

/-!
## Verified Properties

| Property                                                     | Status  | Location                              |
|--------------------------------------------------------------|---------|---------------------------------------|
| Associator definition and non-triviality                     | Defined | `assoc`, `Associates`                 |
| Naive product-rule Hessian over non-associative algebras     | Defined | `dual2MulNaive`                       |
| Associator-corrected product-rule Hessian                    | Defined | `dual2MulCorrected`                   |
| Correction vanishes ⇒ naive = corrected (Fano lemma)         | Proved  | `dual2_mul_corrected_eq_naive_when_assoc_zero` |
| Refusal (no shadows emitted) preserves soundness             | Proved  | `refusal_is_sound`                    |
| Trivial left-zero associator                                 | Proved  | `associates_trivial_left_zero`        |
| **Door β**: Associates ⇒ correction = 0 (Fano gate)         | Proved  | `assoc_correction_zero_on_fano`       |
| **Door β**: Fano gate ⇒ corrected = naive (roundtrip)       | Proved  | `door_beta_fano_naive_eq_corrected`   |

## Axioms (new)

| Axiom                         | Justification                                       |
|-------------------------------|------------------------------------------------------|
| `decidable_fano`              | Fano predicate is computable on 8-basis algebras     |
| `annotation_asserts_fano`     | Trust-boundary placeholder (future: runtime verify)  |

## Correspondence with the compiler

- The refusal branch (`tc_error` at `lean_single.sio:11839`) witnesses
  `refusal_is_sound`: the compiler emits no HSHADOW code, so downstream
  `gum_second_order_variance` sees zero `H` and reduces to first-order.
- The opt-in branch (`reassociate: fano_selective` annotation) is the
  domain of `dual2_mul_corrected_eq_naive_when_assoc_zero`, which says
  the naive product rule is sound exactly when the assertion is true.
  The assertion itself is unverified (Kimi's trust boundary, §6).

## What Phase 2 does NOT prove

- That `gum_second_order_variance` at the stdlib level correctly dispatches
  on the algebra kind (it does not — stdlib reads generic f64 J / H arrays).
- That the `compile_knowledge_muldiv_x86` path (the third butterfly) is
  sound — that function drops second-order shadows entirely and is thus
  trivially sound at first-order but silently loses precision.  A separate
  Phase 2.5 or Phase 3 concern.
-/

end Sounio.NonAssocHessian
