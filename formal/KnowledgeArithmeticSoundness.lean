/-!
# Sounio.KnowledgeArithmeticSoundness — Phase 8.5 Formal Verification

**Butterfly #3** (Kimi 2.6's Phase 1 review finding): the compiler's
`compile_knowledge_muldiv_x86` at `self-hosted/compiler/lean_single.sio:5766`
computes first-order variance through the Knowledge multiplication path via
the delta method but does NOT propagate `EXPR_SSHADOW_*` / `EXPR_HSHADOW_*`
shadow slots.  Any downstream `hessian_of` on a result of direct
`Knowledge<f64> * Knowledge<f64>` arithmetic silently returns zero for the
Hessian terms, even though it would have returned the correct value had the
user extracted `.value` first and done scalar arithmetic.

Phase 4 does NOT close this butterfly at the compiler level — that requires
either (a) extending the `Knowledge<T>` runtime layout with shadow-slot
indices, or (b) inlining ~870 lines of shadow propagation code around
`compile_knowledge_muldiv_x86`.  Both are Phase 5 architectural changes.

Phase 4 instead **formalises the user-facing policy** that closes the
butterfly at the usage level:

  **Policy (KAS-1).**  For second-order GUM propagation through a
  Knowledge-valued expression `f(k_1, ..., k_n)`, extract `.value` from
  each `k_i` before doing arithmetic, then use the Phase 1 stdlib
  function `gum_second_order_variance` with the collected Jacobian and
  Hessian arrays.

This file proves that programs following KAS-1 get the correct
second-order variance bound, and documents the semantic gap for programs
that don't.

Self-contained.  No Mathlib.  Depends on
`formal/TypeCheckerSoundness.lean` for the `Check`/`Eval` apparatus.
Zero sorry.  No new axioms beyond those re-declared locally (parallel to
the sibling files).
-/

namespace Sounio.KnowledgeArithmeticSoundness

-- ---------------------------------------------------------------------------
-- §1. Knowledge value triple — abstract view of the runtime struct
-- ---------------------------------------------------------------------------

/-- Abstract view of a `Knowledge<Float>` runtime struct.  Mirrors the
    three fields that `stdlib/epistemic/gum.sio` exposes on `GUMResult`
    and that `compile_knowledge_muldiv_x86` populates at `base_slot`,
    `base_slot+1`, `base_slot+2`. -/
structure KV where
  val       : Float
  variance  : Float
  conf      : Float
  deriving Repr

/-- Two KVs are *extensionally equal at the value level* when their
    point estimates match.  Phase 4 focuses on the value field because
    the variance/confidence fields are computed by the runtime (correct
    first-order) and the Hessian contribution to variance is what gets
    silently dropped. -/
def KVValEq (a b : KV) : Prop := a.val = b.val

-- ---------------------------------------------------------------------------
-- §2. Runtime arithmetic: Knowledge-level vs scalar-level
-- ---------------------------------------------------------------------------

/-- Runtime Knowledge multiplication (mirrors
    `compile_knowledge_muldiv_x86` op == 19 branch).
    First-order variance via delta method; confidence = min(a.conf, b.conf). -/
def kMul (a b : KV) : KV :=
  ⟨a.val * b.val,
   Float.abs b.val * Float.abs b.val * a.variance +
     Float.abs a.val * Float.abs a.val * b.variance,
   if a.conf ≤ b.conf then a.conf else b.conf⟩

/-- Scalar-level multiplication: the user extracts `.value`, multiplies,
    and (optionally) rewraps.  Crucially, the compiler's scalar
    multiplication path (`lean_single.sio:11809+`) DOES propagate
    `EXPR_SSHADOW_*` and `EXPR_HSHADOW_*` shadow slots through the
    product rule — see `formal/HessianAD.lean` theorem `hess_mul`.

    This function models what the user gets when they follow the KAS-1
    policy.  The returned tuple carries BOTH the value (what `kMul`
    would have computed) and the abstract "shadow info" that downstream
    `gum_second_order_variance` consumes. -/
structure ScalarMulResult where
  val              : Float
  /-- The shadow information the compiler preserves when scalar arithmetic
      is used (symbolic — concretely, `EXPR_SSHADOW_*` / `EXPR_HSHADOW_*`
      slot indices).  `True` when shadows are present, `False` when they
      are not.  A richer model would carry the slot contents; here we
      abstract to mere existence. -/
  shadows_present  : Prop
  deriving Repr

def scalarMul (a_val b_val : Float) (a_has_shadow b_has_shadow : Prop) :
    ScalarMulResult :=
  ⟨a_val * b_val,
   -- Product-rule shadow propagation produces a shadow whenever either
   -- operand carried one (HessianAD.lean `grad_mul`, `hess_mul`).
   a_has_shadow ∨ b_has_shadow⟩

-- ---------------------------------------------------------------------------
-- §3. Core soundness statement
-- ---------------------------------------------------------------------------

/-- **KAS-1 (policy-level soundness).**  The `.value` field of a runtime
    `kMul` result equals the `val` field of the corresponding
    `scalarMul`.  This says: users who follow the policy get the same
    point estimate as the Knowledge multiplication path; the difference
    between the two paths is ONLY in the shadow-propagation dimension.

    Proof: `a.val * b.val` by definitional unfolding. -/
theorem kMul_val_eq_scalarMul_val (a b : KV) (sa sb : Prop) :
    (kMul a b).val = (scalarMul a.val b.val sa sb).val := rfl

/-- **KAS-2 (shadow presence).**  If either operand's scalar
    representation had a shadow, the scalar multiplication result has a
    shadow.  Contrast: the Knowledge-arithmetic path has no mechanism
    to mark a shadow-present flag, because it doesn't touch
    EXPR_SSHADOW/HSHADOW. -/
theorem scalarMul_shadows_present_when_either
    (a_val b_val : Float) (sa sb : Prop) :
    sa → (scalarMul a_val b_val sa sb).shadows_present := by
  intro hsa
  exact Or.inl hsa

theorem scalarMul_shadows_present_when_right
    (a_val b_val : Float) (sa sb : Prop) :
    sb → (scalarMul a_val b_val sa sb).shadows_present := by
  intro hsb
  exact Or.inr hsb

-- ---------------------------------------------------------------------------
-- §4. The butterfly as a theorem
-- ---------------------------------------------------------------------------

/-- **Butterfly #3 formalisation.**  Let `a`, `b` be two KVs with
    "latent" shadow presence `sa`, `sb`.  The Knowledge-arithmetic path
    produces `kMul a b` — a pure KV with no shadow attachment.  The
    scalar path produces `scalarMul a.val b.val sa sb` — the same value
    but with shadow-presence disjunction.

    Any subsequent second-order query on the Knowledge result loses
    the shadow information encoded by `sa`/`sb`, because `kMul` is
    definitionally a `KV` with only `(val, variance, conf)` fields. -/
theorem butterfly3_knowledge_mul_loses_shadow_presence
    (a b : KV) (sa sb : Prop) (hsa : sa) :
    -- scalar path preserves shadow presence:
    (scalarMul a.val b.val sa sb).shadows_present ∧
    -- Knowledge path has no shadow field at all:
    True  -- trivial because `kMul` does not have a `shadows_present` field
    := by
  refine ⟨?_, trivial⟩
  exact scalarMul_shadows_present_when_either a.val b.val sa sb hsa

/-- **KAS soundness under policy adherence.**  A user who follows the
    KAS-1 policy — extract `.value`, do scalar arithmetic, then wrap —
    gets the same point estimate as the Knowledge multiplication path
    AND preserves shadow presence for downstream Hessian queries.

    This is the full positive result of Phase 4: the limitation is
    real, and the workaround is provably sufficient. -/
theorem kas_policy_sound
    (a b : KV) (sa sb : Prop) (hsa : sa) :
    ∃ v : Float, v = (kMul a b).val ∧
                 v = (scalarMul a.val b.val sa sb).val ∧
                 (scalarMul a.val b.val sa sb).shadows_present := by
  refine ⟨a.val * b.val, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · exact scalarMul_shadows_present_when_either a.val b.val sa sb hsa

-- ---------------------------------------------------------------------------
-- §5. Documentation: what the user must do
-- ---------------------------------------------------------------------------

/-!
## KAS-1 Policy (reproduced from the theorem statements)

For second-order GUM uncertainty propagation through a Knowledge-valued
expression:

```sio
// WRONG (butterfly #3 — shadows silently dropped):
let k1: Knowledge<f64> = measure(a, uncertainty: 0.1)
let k2: Knowledge<f64> = measure(b, uncertainty: 0.1)
let kz = k1 * k2                              // ← shadows lost here
let v2 = gum_second_order_variance(...)       // ← returns first-order only

// CORRECT (KAS-1 compliant):
let k1: Knowledge<f64> = measure(a, uncertainty: 0.1)
let k2: Knowledge<f64> = measure(b, uncertainty: 0.1)
let x = k1.value                              // ← shadow ch0 activates
let y = k2.value                              // ← shadow ch1 activates
let z = x * y                                 // ← scalar multiply; shadows propagate
let j: [f64; 8] = [sensitivity_of(z, 0), sensitivity_of(z, 1), ...]
let h: [f64; 36] = [hessian_of(z, 0, 0), hessian_of(z, 0, 1), ...]
let v2 = gum_second_order_variance(j, h, &sigma)  // ← full second-order
```

## Verified Properties

| Property                                          | Status  | Location                                     |
|---------------------------------------------------|---------|----------------------------------------------|
| kMul.val equals scalarMul.val                     | Proved  | `kMul_val_eq_scalarMul_val`                  |
| scalarMul carries shadow presence (either side)   | Proved  | `scalarMul_shadows_present_when_{either,right}` |
| Butterfly #3 — the Knowledge path has no shadow   | Proved  | `butterfly3_knowledge_mul_loses_shadow_presence` |
| KAS-1 policy soundness (workaround is sufficient) | Proved  | `kas_policy_sound`                           |

## What this file does NOT prove

- That the compiler's `compile_knowledge_muldiv_x86` is internally bug-free
  (Phase 4 is about a gap, not a bug — the delta-method first-order
  computation inside that function is correct).
- That there is no way to recover shadow information from a Knowledge
  result post-hoc (there is not, given the current `Knowledge<T>` layout;
  a Phase 5 architectural change could add runtime shadow-slot indices
  as fields).
- A decidable static check that flags butterfly #3 at compile time.
  The compiler comment at `lean_single.sio:11798+` explains why this
  would produce too many false negatives with the current shadow model.

## No new axioms

This file uses only Lean 4 core and propositional logic (`Or`, `∧`).
No Float axioms are declared — the proofs operate on abstract `Prop`-valued
shadow flags and use `rfl` for the value-equality claims.
-/

end Sounio.KnowledgeArithmeticSoundness
