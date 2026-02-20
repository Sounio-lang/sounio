-- formal/lean4/SounioEffects.lean
/-!
# Sounio Effect System — Lean 4 Formalization

Formalizes the algebraic effect row system of Sounio.

Mirrors the Rust implementation in:
- `crates/souc/src/types/core.rs:973–1076`  (Effect names, EffectSet::subtract/union)

References:
- Plotkin, G. and Pretnar, M. (2009). "Handlers of Algebraic Effects." ESOP.
- Lindley, S. and Cheney, J. (2012). "Row-based Effect Types for Database Integration." TLDI.
- Leijen, D. (2014). "Koka: Programming with Row Polymorphic Effect Types." HOPE.

## Design

Effect rows are modeled as characteristic functions `Effect → Bool` rather than
`Finset Effect`, eliminating any Mathlib dependency. This is mathematically equivalent
to finite sets via function extensionality.

All proofs use `funext` + `by_cases` + `simp`. No `sorry`. No Mathlib.
-/

namespace Sounio.Effects

-- ================================================================
-- §1. Effect Enumeration
-- ================================================================

/-- Named effects in Sounio.
    Mirrors the constructors in `crates/souc/src/types/core.rs:973–1076`. -/
inductive Effect where
  | IO         -- console, file, network I/O
  | Mut        -- mutable state
  | Alloc      -- memory allocation
  | Prob       -- probabilistic sampling
  | GPU        -- GPU kernel launch
  | Epistemic  -- confidence/provenance operations
  | Div        -- division (may be by zero)
  | Exn        -- exceptions
  | Async      -- asynchronous computation
  | FFI        -- foreign function interface
  deriving DecidableEq, Repr

-- ================================================================
-- §2. Effect Rows as Characteristic Functions
-- ================================================================

/-- An effect row is a decidable predicate over effects.
    Mathematically equivalent to a finite set via `funext`. -/
abbrev EffectRow := Effect → Bool

/-- The empty row — represents pure (effect-free) functions. -/
def pureRow : EffectRow := fun _ => false

/-- A row containing exactly one effect. -/
def singleRow (e : Effect) : EffectRow := fun f =>
  if f = e then true else false

/-- Effect membership. -/
def memberOf (e : Effect) (row : EffectRow) : Prop := row e = true

notation:50 e " ∈ᵣ " row => memberOf e row

/-- Decidability of membership. -/
instance decMemberOf (e : Effect) (row : EffectRow) :
    Decidable (e ∈ᵣ row) :=
  Bool.decEq (row e) true

-- ================================================================
-- §3. Masking (Effect Handler Application)
-- ================================================================

/-- Masking removes effect `e` from the row.
    Models: a handler for `e` transforms `f : τ with (ρ ∪ {e})`
    into `f : τ with ρ`.
    Mirrors `EffectSet::subtract` in `crates/souc/src/types/core.rs`. -/
def mask (row : EffectRow) (e : Effect) : EffectRow := fun f =>
  if f = e then false else row f

-- §3.1  Masking theorems

/-- Masking is idempotent: applying the same handler twice = once. -/
theorem mask_idempotent (row : EffectRow) (e : Effect) :
    mask (mask row e) e = mask row e := by
  funext f; simp only [mask]
  by_cases h : f = e <;> simp [h]

/-- Two distinct handlers commute: order of application is irrelevant. -/
theorem mask_comm (row : EffectRow) (e₁ e₂ : Effect) :
    mask (mask row e₁) e₂ = mask (mask row e₂) e₁ := by
  funext f; simp only [mask]
  by_cases h1 : f = e₁ <;> by_cases h2 : f = e₂
  · subst h1; simp        -- both sides false: outer if-pos, inner if-pos → false
  · subst h1; simp [h2]   -- outer if-neg h2, inner if-pos → false; RHS if-pos → false
  · subst h2; simp [h1]   -- outer if-pos → false; RHS if-neg h1, inner if-pos → false
  · simp [h1, h2]          -- all ifs → else branch → row f

/-- Masking an effect not in the row has no effect. -/
theorem mask_absent_noop (row : EffectRow) (e : Effect)
    (hab : ¬(e ∈ᵣ row)) : mask row e = row := by
  funext f; simp only [mask]
  by_cases h : f = e
  · subst h
    cases hrow : row f
    · rfl
    · exact absurd hrow hab
  · simp [h]

/-- Masking removes exactly the target effect. -/
theorem mask_removes (row : EffectRow) (e : Effect) :
    ¬(e ∈ᵣ mask row e) := by
  simp [memberOf, mask]

/-- Masking preserves all other effects. -/
theorem mask_preserves_other (row : EffectRow) (e f : Effect) (hne : f ≠ e) :
    (f ∈ᵣ mask row e) ↔ (f ∈ᵣ row) := by
  simp [memberOf, mask, hne]

-- ================================================================
-- §4. Row Union
-- ================================================================

/-- Union of two effect rows.
    Mirrors `EffectSet::union` in `crates/souc/src/types/core.rs`. -/
def rowUnion (r₁ r₂ : EffectRow) : EffectRow := fun e => r₁ e || r₂ e

theorem rowUnion_comm (r₁ r₂ : EffectRow) :
    rowUnion r₁ r₂ = rowUnion r₂ r₁ := by
  funext e; simp only [rowUnion]
  cases r₁ e <;> cases r₂ e <;> rfl

theorem rowUnion_assoc (r₁ r₂ r₃ : EffectRow) :
    rowUnion (rowUnion r₁ r₂) r₃ = rowUnion r₁ (rowUnion r₂ r₃) := by
  funext e; simp only [rowUnion]
  cases r₁ e <;> cases r₂ e <;> cases r₃ e <;> rfl

theorem rowUnion_idempotent (r : EffectRow) :
    rowUnion r r = r := by
  funext e; simp only [rowUnion]; cases r e <;> rfl

theorem rowUnion_pure_left (r : EffectRow) :
    rowUnion pureRow r = r := by
  funext e; simp [rowUnion, pureRow]

theorem rowUnion_pure_right (r : EffectRow) :
    rowUnion r pureRow = r := by
  funext e; simp [rowUnion, pureRow]

-- ================================================================
-- §5. Subrow (Effect Subtyping)
-- ================================================================

/-- r₁ ⊆ r₂: r₁ requires fewer effects than r₂. -/
def effectSubrow (r₁ r₂ : EffectRow) : Prop :=
  ∀ e, r₁ e = true → r₂ e = true

theorem effectSubrow_refl (r : EffectRow) :
    effectSubrow r r :=
  fun _ h => h

theorem effectSubrow_trans (r₁ r₂ r₃ : EffectRow)
    (h₁ : effectSubrow r₁ r₂) (h₂ : effectSubrow r₂ r₃) :
    effectSubrow r₁ r₃ :=
  fun e h => h₂ e (h₁ e h)

theorem effectSubrow_antisymm (r₁ r₂ : EffectRow)
    (h₁ : effectSubrow r₁ r₂) (h₂ : effectSubrow r₂ r₁) :
    r₁ = r₂ := by
  funext e
  cases hr₁ : r₁ e <;> cases hr₂ : r₂ e
  · rfl
  · exact absurd (h₂ e hr₂) (by simp [hr₁])
  · exact absurd (h₁ e hr₁) (by simp [hr₂])
  · rfl

-- ================================================================
-- §6. Pure Functions
-- ================================================================

/-- The pure row has no effects. -/
theorem pureRow_is_pure : ∀ e, pureRow e = false := fun _ => rfl

/-- Masking any effect from the pure row yields the pure row. -/
theorem pure_mask_noop (e : Effect) :
    mask pureRow e = pureRow := by
  funext f; simp [mask, pureRow]

/-- Union with the pure row is the identity (left). -/
theorem pure_union_left (r : EffectRow) :
    rowUnion pureRow r = r :=
  rowUnion_pure_left r

/-- The pure row is a subrow of every row. -/
theorem pure_is_subrow (r : EffectRow) :
    effectSubrow pureRow r :=
  fun _ h => by simp [pureRow] at h

-- ================================================================
-- §7. Handler Soundness Theorems
-- ================================================================

/-! ### Handler correctness

    A handler for effect `e` applied to a computation with effect row `r`
    produces a computation with effect row `mask r e ⊆ r`.

    References: Plotkin & Pretnar (2009). "Handlers of Algebraic Effects." ESOP. -/

/-- Handling effect `e` reduces the effect row: `mask r e ⊆ r`. -/
theorem handler_reduces_effects (r : EffectRow) (e : Effect) :
    effectSubrow (mask r e) r := by
  intro f hf
  simp only [mask] at hf
  by_cases h : f = e
  · subst h; simp at hf
  · simp [h] at hf; exact hf

/-- Applying the same handler twice = applying it once. -/
theorem handler_idempotent (r : EffectRow) (e : Effect) :
    mask (mask r e) e = mask r e :=
  mask_idempotent r e

/-- Two handlers for distinct effects can be applied in either order. -/
theorem handler_order_independence (r : EffectRow) (e₁ e₂ : Effect) :
    mask (mask r e₁) e₂ = mask (mask r e₂) e₁ :=
  mask_comm r e₁ e₂

/-- Handling the sole effect of a single-effect row yields the pure row. -/
theorem single_mask_pure (e : Effect) :
    mask (singleRow e) e = pureRow := by
  funext f; simp only [mask, singleRow, pureRow]
  by_cases h : f = e <;> simp [h]

-- ================================================================
-- §8. Row Union as Lattice Upper Bound
-- ================================================================

/-- Row union is a least upper bound: r₁ ⊆ (r₁ ∪ r₂). -/
theorem effectSubrow_union_left (r₁ r₂ : EffectRow) :
    effectSubrow r₁ (rowUnion r₁ r₂) :=
  fun e h => by simp [rowUnion, h]

/-- Row union is a least upper bound: r₂ ⊆ (r₁ ∪ r₂). -/
theorem effectSubrow_union_right (r₁ r₂ : EffectRow) :
    effectSubrow r₂ (rowUnion r₁ r₂) :=
  fun e h => by simp [rowUnion, h]

/-- If r₁ ⊆ r and r₂ ⊆ r, then (r₁ ∪ r₂) ⊆ r. -/
theorem effectSubrow_union_lub (r₁ r₂ r : EffectRow)
    (h₁ : effectSubrow r₁ r) (h₂ : effectSubrow r₂ r) :
    effectSubrow (rowUnion r₁ r₂) r :=
  fun e he => by
    simp only [rowUnion] at he
    cases h_r1 : r₁ e
    · cases h_r2 : r₂ e
      · simp [h_r1, h_r2] at he
      · exact h₂ e h_r2
    · exact h₁ e h_r1

-- ================================================================
-- §9. Mask Distributes over Union
-- ================================================================

/-- Masking distributes over union when the masked effect is absent from r₁. -/
theorem mask_union_right (r₁ r₂ : EffectRow) (e : Effect)
    (h : r₁ e = false) :
    mask (rowUnion r₁ r₂) e = rowUnion r₁ (mask r₂ e) := by
  funext f; simp only [mask, rowUnion]
  by_cases heq : f = e
  · subst heq; simp [h]
  · simp [heq]

end Sounio.Effects
