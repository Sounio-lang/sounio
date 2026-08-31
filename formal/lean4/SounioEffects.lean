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
  | NonAssoc   -- non-associative algebraic operations (ID 14 in Sounio)
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
    · simp
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

-- ================================================================
-- §10. SingleRow Properties
-- ================================================================

/-- The target effect is a member of its own singleton row. -/
theorem singleRow_member (e : Effect) : e ∈ᵣ singleRow e := by
  simp [memberOf, singleRow]

/-- Any other effect is not a member of a singleton row. -/
theorem singleRow_not_member (e f : Effect) (hne : f ≠ e) :
    ¬(f ∈ᵣ singleRow e) := by
  simp [memberOf, singleRow, hne]

/-- Membership in a singleton row iff equal to the target. -/
theorem memberOf_single_iff (e f : Effect) :
    (e ∈ᵣ singleRow f) ↔ e = f := by
  constructor
  · intro h
    have h' : singleRow f e = true := h
    simp only [singleRow] at h'
    by_cases heq : e = f <;> simp_all
  · intro h; rw [h]; exact singleRow_member f

/-- Masking a different effect from a singleton row is a no-op. -/
theorem mask_single_other (e1 e2 : Effect) (hne : e1 ≠ e2) :
    mask (singleRow e1) e2 = singleRow e1 :=
  mask_absent_noop _ _ (singleRow_not_member e1 e2 hne.symm)

-- ================================================================
-- §11. Membership Characterizations
-- ================================================================

/-- No effect is a member of the pure row. -/
theorem memberOf_pure_false (e : Effect) : ¬(e ∈ᵣ pureRow) := by
  simp [memberOf, pureRow]

/-- Membership in a union iff membership in either component. -/
theorem memberOf_union_iff (e : Effect) (r1 r2 : EffectRow) :
    (e ∈ᵣ rowUnion r1 r2) ↔ ((e ∈ᵣ r1) ∨ (e ∈ᵣ r2)) := by
  constructor
  · intro h
    have h' : (r1 e || r2 e) = true := h
    cases hr1 : r1 e <;> cases hr2 : r2 e <;> simp_all [memberOf]
  · intro h
    show (r1 e || r2 e) = true
    rcases h with h1 | h2
    · simp [(show r1 e = true from h1)]
    · simp [(show r2 e = true from h2)]

/-- A row with no members is the pure row. -/
theorem pureRow_unique (r : EffectRow) (h : ∀ e, ¬(e ∈ᵣ r)) : r = pureRow := by
  funext e
  cases hr : r e
  · rfl
  · exact absurd hr (h e)

-- ================================================================
-- §12. Row Intersection
-- ================================================================

/-- Intersection of two effect rows: an effect is present iff in both. -/
def rowInter (r1 r2 : EffectRow) : EffectRow := fun e => r1 e && r2 e

theorem rowInter_comm (r1 r2 : EffectRow) :
    rowInter r1 r2 = rowInter r2 r1 := by
  funext e; simp only [rowInter]; cases r1 e <;> cases r2 e <;> rfl

theorem rowInter_assoc (r1 r2 r3 : EffectRow) :
    rowInter (rowInter r1 r2) r3 = rowInter r1 (rowInter r2 r3) := by
  funext e; simp only [rowInter]; cases r1 e <;> cases r2 e <;> cases r3 e <;> rfl

theorem rowInter_idempotent (r : EffectRow) :
    rowInter r r = r := by
  funext e; simp only [rowInter]; cases r e <;> rfl

theorem rowInter_pure_left (r : EffectRow) :
    rowInter pureRow r = pureRow := by
  funext e; simp [rowInter, pureRow]

theorem rowInter_pure_right (r : EffectRow) :
    rowInter r pureRow = pureRow := by
  funext e; simp [rowInter, pureRow]

/-- Membership in an intersection iff membership in both components. -/
theorem memberOf_inter_iff (e : Effect) (r1 r2 : EffectRow) :
    (e ∈ᵣ rowInter r1 r2) ↔ ((e ∈ᵣ r1) ∧ (e ∈ᵣ r2)) := by
  constructor
  · intro h
    have h' : (r1 e && r2 e) = true := h
    cases hr1 : r1 e <;> cases hr2 : r2 e <;> simp_all [memberOf]
  · intro h
    show (r1 e && r2 e) = true
    obtain ⟨h1, h2⟩ := h
    simp [(show r1 e = true from h1), (show r2 e = true from h2)]

/-- Intersection is a lower bound: r1 ∩ r2 ⊆ r1. -/
theorem effectSubrow_inter_left (r1 r2 : EffectRow) :
    effectSubrow (rowInter r1 r2) r1 := by
  intro e he
  simp only [rowInter] at he
  cases h1 : r1 e <;> cases h2 : r2 e <;> simp_all

/-- Intersection is a lower bound: r1 ∩ r2 ⊆ r2. -/
theorem effectSubrow_inter_right (r1 r2 : EffectRow) :
    effectSubrow (rowInter r1 r2) r2 := by
  intro e he
  simp only [rowInter] at he
  cases h1 : r1 e <;> cases h2 : r2 e <;> simp_all

/-- Intersection is the greatest lower bound: r ⊆ r1 ∧ r ⊆ r2 → r ⊆ r1 ∩ r2. -/
theorem effectSubrow_inter_glb (r r1 r2 : EffectRow)
    (h1 : effectSubrow r r1) (h2 : effectSubrow r r2) :
    effectSubrow r (rowInter r1 r2) := by
  intro e he
  simp only [rowInter]
  simp [h1 e he, h2 e he]

/-- Intersection distributes over union (Boolean law). -/
theorem rowInter_union_distrib (r1 r2 r3 : EffectRow) :
    rowInter r1 (rowUnion r2 r3) = rowUnion (rowInter r1 r2) (rowInter r1 r3) := by
  funext e; simp only [rowInter, rowUnion]
  cases r1 e <;> cases r2 e <;> cases r3 e <;> rfl

/-- Union distributes over intersection (Boolean law). -/
theorem rowUnion_inter_distrib (r1 r2 r3 : EffectRow) :
    rowUnion r1 (rowInter r2 r3) = rowInter (rowUnion r1 r2) (rowUnion r1 r3) := by
  funext e; simp only [rowUnion, rowInter]
  cases r1 e <;> cases r2 e <;> cases r3 e <;> rfl

/-- Absorption: r1 ∩ (r1 ∪ r2) = r1. -/
theorem rowInter_union_absorb (r1 r2 : EffectRow) :
    rowInter r1 (rowUnion r1 r2) = r1 := by
  funext e; simp only [rowInter, rowUnion]; cases r1 e <;> cases r2 e <;> rfl

/-- Absorption: r1 ∪ (r1 ∩ r2) = r1. -/
theorem rowUnion_inter_absorb (r1 r2 : EffectRow) :
    rowUnion r1 (rowInter r1 r2) = r1 := by
  funext e; simp only [rowUnion, rowInter]; cases r1 e <;> cases r2 e <;> rfl

-- ================================================================
-- §13. Monotonicity
-- ================================================================

/-- Union is monotone on the left: r1 ⊆ r2 implies r1 ∪ r3 ⊆ r2 ∪ r3. -/
theorem effectSubrow_union_mono_left (r1 r2 r3 : EffectRow)
    (h : effectSubrow r1 r2) :
    effectSubrow (rowUnion r1 r3) (rowUnion r2 r3) := by
  intro e he
  simp only [rowUnion] at *
  cases h1 : r1 e <;> cases h3 : r3 e <;> simp_all
  exact h e h1

/-- Union is monotone on the right: r1 ⊆ r2 implies r3 ∪ r1 ⊆ r3 ∪ r2. -/
theorem effectSubrow_union_mono_right (r1 r2 r3 : EffectRow)
    (h : effectSubrow r1 r2) :
    effectSubrow (rowUnion r3 r1) (rowUnion r3 r2) := by
  intro e he
  simp only [rowUnion] at *
  cases h3 : r3 e <;> cases h1 : r1 e <;> simp_all
  exact h e h1

/-- Masking is monotone: r1 ⊆ r2 implies mask r1 e ⊆ mask r2 e. -/
theorem effectSubrow_mask_mono (r1 r2 : EffectRow) (e : Effect)
    (h : effectSubrow r1 r2) :
    effectSubrow (mask r1 e) (mask r2 e) := by
  intro f hf
  simp only [mask] at *
  by_cases heq : f = e
  · subst heq; simp at hf
  · simp [heq] at hf ⊢; exact h f hf

/-- Intersection is monotone on the left: r1 ⊆ r2 implies r1 ∩ r3 ⊆ r2 ∩ r3. -/
theorem effectSubrow_inter_mono_left (r1 r2 r3 : EffectRow)
    (h : effectSubrow r1 r2) :
    effectSubrow (rowInter r1 r3) (rowInter r2 r3) := by
  intro e he
  simp only [rowInter] at *
  cases h1 : r1 e <;> cases h3 : r3 e <;> simp_all
  exact h e h1

-- ================================================================
-- §14. Row Complement (Boolean Algebra)
-- ================================================================

/-- The complement of an effect row: e is present iff absent in r. -/
def rowComplement (r : EffectRow) : EffectRow := fun e => !r e

/-- Double complement is the identity (involution). -/
theorem rowComplement_involution (r : EffectRow) :
    rowComplement (rowComplement r) = r := by
  funext e; simp [rowComplement]

/-- De Morgan: complement of union = intersection of complements. -/
theorem rowComplement_union (r1 r2 : EffectRow) :
    rowComplement (rowUnion r1 r2) = rowInter (rowComplement r1) (rowComplement r2) := by
  funext e; simp only [rowComplement, rowUnion, rowInter]
  cases r1 e <;> cases r2 e <;> rfl

/-- De Morgan: complement of intersection = union of complements. -/
theorem rowComplement_inter (r1 r2 : EffectRow) :
    rowComplement (rowInter r1 r2) = rowUnion (rowComplement r1) (rowComplement r2) := by
  funext e; simp only [rowComplement, rowInter, rowUnion]
  cases r1 e <;> cases r2 e <;> rfl

/-- A row unioned with its complement covers every effect. -/
theorem rowUnion_complement_full (r : EffectRow) :
    rowUnion r (rowComplement r) = fun _ => true := by
  funext e; simp only [rowUnion, rowComplement]; cases r e <;> rfl

/-- A row intersected with its complement is the pure row. -/
theorem rowInter_complement_pure (r : EffectRow) :
    rowInter r (rowComplement r) = pureRow := by
  funext e; simp only [rowInter, rowComplement, pureRow]; cases r e <;> rfl

/-- Complement is anti-monotone: r1 ⊆ r2 implies complement(r2) ⊆ complement(r1). -/
theorem effectSubrow_complement_antimono (r1 r2 : EffectRow)
    (h : effectSubrow r1 r2) :
    effectSubrow (rowComplement r2) (rowComplement r1) := by
  intro e he
  simp only [rowComplement] at *
  cases h1 : r1 e
  · rfl
  · exact absurd (h e h1) (by simp_all)

-- ================================================================
-- §15. All-Effects Row (Top Element)
-- ================================================================

/-- The all-effects row: every effect is present. -/
def allEffectsRow : EffectRow := fun _ => true

/-- allEffectsRow is the top of the effect lattice: everything is a subrow. -/
theorem effectSubrow_allEffects (r : EffectRow) :
    effectSubrow r allEffectsRow :=
  fun _ _ => rfl

/-- Union with allEffectsRow on the right absorbs everything. -/
theorem rowUnion_allEffects_right (r : EffectRow) :
    rowUnion r allEffectsRow = allEffectsRow := by
  funext e; simp [rowUnion, allEffectsRow]

/-- Union with allEffectsRow on the left absorbs everything. -/
theorem rowUnion_allEffects_left (r : EffectRow) :
    rowUnion allEffectsRow r = allEffectsRow := by
  funext e; simp [rowUnion, allEffectsRow]

/-- Intersection with allEffectsRow on the right is the identity. -/
theorem rowInter_allEffects_right (r : EffectRow) :
    rowInter r allEffectsRow = r := by
  funext e; simp [rowInter, allEffectsRow]

/-- Intersection with allEffectsRow on the left is the identity. -/
theorem rowInter_allEffects_left (r : EffectRow) :
    rowInter allEffectsRow r = r := by
  funext e; simp [rowInter, allEffectsRow]

/-- The complement of pureRow is allEffectsRow. -/
theorem rowComplement_pure_is_all :
    rowComplement pureRow = allEffectsRow := by
  funext e; simp [rowComplement, pureRow, allEffectsRow]

/-- The complement of allEffectsRow is pureRow. -/
theorem rowComplement_all_is_pure :
    rowComplement allEffectsRow = pureRow := by
  funext e; simp [rowComplement, allEffectsRow, pureRow]

-- ================================================================
-- §16. Disjoint Rows
-- ================================================================

/-- Two rows are disjoint if they share no effects. -/
def rowDisjoint (r1 r2 : EffectRow) : Prop :=
  rowInter r1 r2 = pureRow

theorem rowDisjoint_comm (r1 r2 : EffectRow)
    (h : rowDisjoint r1 r2) : rowDisjoint r2 r1 := by
  unfold rowDisjoint at *; rw [rowInter_comm]; exact h

theorem rowDisjoint_pure_left (r : EffectRow) : rowDisjoint pureRow r :=
  rowInter_pure_left r

theorem rowDisjoint_pure_right (r : EffectRow) : rowDisjoint r pureRow :=
  rowInter_pure_right r

/-- A singleton row is disjoint from any row not containing that effect. -/
theorem rowDisjoint_single_absent (e : Effect) (r : EffectRow)
    (hab : ¬(e ∈ᵣ r)) : rowDisjoint (singleRow e) r := by
  unfold rowDisjoint
  funext f; simp only [rowInter, pureRow]
  by_cases heq : f = e
  · subst heq
    simp only [singleRow, ite_true]
    cases hr : r f
    · rfl
    · exact absurd hr hab
  · simp [singleRow, heq]

/-- A singleton {e} is disjoint from the row produced by masking e. -/
theorem rowDisjoint_single_mask (e : Effect) (r : EffectRow) :
    rowDisjoint (singleRow e) (mask r e) :=
  rowDisjoint_single_absent e (mask r e) (mask_removes r e)

/-- Disjointness is preserved under union: (r1 ⊥ r3) ∧ (r2 ⊥ r3) → (r1 ∪ r2) ⊥ r3. -/
theorem rowDisjoint_union_inter (r1 r2 r3 : EffectRow)
    (h1 : rowDisjoint r1 r3) (h2 : rowDisjoint r2 r3) :
    rowDisjoint (rowUnion r1 r2) r3 := by
  unfold rowDisjoint at *
  funext e
  have he1 := congrFun h1 e
  have he2 := congrFun h2 e
  simp only [rowInter, pureRow, rowUnion] at he1 he2 ⊢
  cases hr1 : r1 e <;> cases hr2 : r2 e <;> cases hr3 : r3 e <;> simp_all

-- ================================================================
-- §17. maskAll — Handle All Effects
-- ================================================================

/-- Apply handlers for all 11 named effects in order. -/
def maskAll (r : EffectRow) : EffectRow :=
  mask (mask (mask (mask (mask (mask
  (mask (mask (mask (mask (mask r
    .IO) .Mut) .Alloc) .Prob) .GPU)
    .Epistemic) .Div) .Exn) .Async) .FFI) .NonAssoc

/-- Applying handlers for all effects always produces the pure row. -/
theorem maskAll_pure (r : EffectRow) : maskAll r = pureRow := by
  funext e; cases e <;> simp [maskAll, mask, pureRow]

-- ================================================================
-- §18. Additional Mask and Union Properties
-- ================================================================

/-- Masking distributes over union when the effect is absent from the left. -/
theorem mask_union_left (r1 r2 : EffectRow) (e : Effect)
    (h : r1 e = false) :
    mask (rowUnion r1 r2) e = rowUnion r1 (mask r2 e) := by
  funext f; simp only [mask, rowUnion]
  by_cases heq : f = e
  · subst heq; simp [h]
  · simp [heq]

/-- Masking distributes over intersection. -/
theorem mask_inter (r1 r2 : EffectRow) (e : Effect) :
    mask (rowInter r1 r2) e = rowInter (mask r1 e) (mask r2 e) := by
  funext f; simp only [mask, rowInter]
  by_cases h : f = e <;> simp [h]

/-- A chain of two handlers reduces effects twice (transitivity of masking). -/
theorem handler_chain_reduces (r : EffectRow) (e1 e2 : Effect) :
    effectSubrow (mask (mask r e1) e2) r :=
  effectSubrow_trans _ _ _
    (handler_reduces_effects (mask r e1) e2)
    (handler_reduces_effects r e1)

/-- Mask followed by union with the removed singleton recovers a superset. -/
theorem mask_union_single_superset (r : EffectRow) (e : Effect) :
    effectSubrow r (rowUnion (mask r e) (singleRow e)) := by
  intro f hf
  simp only [rowUnion, mask, singleRow]
  by_cases h : f = e
  · subst h; simp
  · simp [h, hf]

-- ================================================================
-- §19. Handler × payload interaction (identity handlers)
-- ================================================================

/-! ### Effectful computations

    Models a value paired with an effect row. Identity handlers (Plotkin–Pretnar
    structural clears) change only the row; they do not rewrite the payload.
    This is the missing interaction layer flagged against the NonUnitary×NWA
    leaf: commutation of masks alone does not state payload preservation or
    sequential discharge of membership. -/

/-- A computation carrying a payload and an effect row. -/
structure Effectful (α : Type) where
  value : α
  effects : EffectRow

/-- Handle effect `e`: clear it from the row; keep the value. -/
def handle (c : Effectful α) (e : Effect) : Effectful α :=
  { c with effects := mask c.effects e }

/-- Identity handlers preserve the computational payload. -/
theorem handle_preserves_value (c : Effectful α) (e : Effect) :
    (handle c e).value = c.value := by
  rfl

/-- Handling twice for distinct effects is order-independent on the whole
    `Effectful` (value + row), not merely on the row. -/
theorem handle_comm (c : Effectful α) (e₁ e₂ : Effect) :
    handle (handle c e₁) e₂ = handle (handle c e₂) e₁ := by
  cases c with
  | mk v r =>
    simp only [handle]
    rw [mask_comm]

/-- Sequential discharge: after handling `e`, `e` is absent from the row. -/
theorem handle_clears_target (c : Effectful α) (e : Effect) :
    ¬(e ∈ᵣ (handle c e).effects) := by
  simp [handle, memberOf, mask]

/-- Handling `e` preserves membership of every other effect. -/
theorem handle_preserves_other_member (c : Effectful α) (e f : Effect)
    (hne : f ≠ e) :
    (f ∈ᵣ (handle c e).effects) ↔ (f ∈ᵣ c.effects) := by
  simp [handle, memberOf, mask, hne]

/-- Two-handler discharge clears both targets and leaves the payload intact. -/
theorem discharge_pair_interaction (c : Effectful α) (e₁ e₂ : Effect) :
    (handle (handle c e₁) e₂).value = c.value ∧
    ¬(e₁ ∈ᵣ (handle (handle c e₁) e₂).effects) ∧
    ¬(e₂ ∈ᵣ (handle (handle c e₁) e₂).effects) := by
  refine ⟨by rfl, ?_, ?_⟩
  · -- e₁ cleared even after a subsequent distinct/same mask
    simp only [handle, memberOf, mask]
    by_cases h : e₁ = e₂ <;> simp [h]
  · exact handle_clears_target (handle c e₁) e₂

end Sounio.Effects
