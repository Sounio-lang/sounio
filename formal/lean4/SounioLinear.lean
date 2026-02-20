-- formal/lean4/SounioLinear.lean
/-!
# Sounio Linear Type System — Lean 4 Formalization

Formalizes the four-modality linear type system for Sounio.

Mirrors the Rust implementation in:
- `crates/souc/src/linear/modality.rs`  (combine/join/allows_weakening/allows_contraction/must_use)
- `crates/souc/src/linear/context.rs`   (UsageCount/LinearContext/check_exhausted/ContextSplit)
- `crates/souc/src/linear/exponentials.rs` (BangType::can_promote, Seely isomorphism)

References:
- Girard, J.-Y. (1987). "Linear Logic." Theoretical Computer Science 50(1):1–102.
- Wadler, P. (1990). "Linear Types Can Change the World!" IFIP TC 2 Working Conference.
- Bierman, G. (1994). "On Intuitionistic Linear Logic." PhD thesis, Cambridge.

## Modality lattice (ordered by resource permissiveness)

```
       Unrestricted  -- weakening + contraction
          /      \
      Affine    Relevant  -- incomparable
       (≤1)      (≥1)
          \      /
          Linear  -- neither structural rule
```

`meet` = greatest lower bound (most restrictive).
`join` = least upper bound (most permissive).
-/

namespace Sounio.Linear

-- ================================================================
-- §1. Modalities
-- ================================================================

/-- The four sub-structural modalities of resource usage.
    Mirrors `Modality` enum in `linear/modality.rs`. -/
inductive Modality where
  | Linear       -- use exactly once
  | Affine       -- use at most once    (weakening allowed)
  | Relevant     -- use at least once   (contraction allowed)
  | Unrestricted -- any number of times (both structural rules)
  deriving DecidableEq, Repr

open Modality

/-- Weakening: the variable may be dropped unused.
    Mirrors `Modality::allows_weakening` in `linear/modality.rs`. -/
def Modality.allowsWeakening : Modality → Bool
  | Affine       => true
  | Unrestricted => true
  | _            => false

/-- Contraction: the variable may be used more than once.
    Mirrors `Modality::allows_contraction` in `linear/modality.rs`. -/
def Modality.allowsContraction : Modality → Bool
  | Relevant     => true
  | Unrestricted => true
  | _            => false

/-- Whether the variable MUST be used at least once.
    Mirrors `Modality::must_use` in `linear/modality.rs`. -/
def Modality.mustUse : Modality → Bool
  | Linear   => true
  | Relevant => true
  | _        => false

-- ================================================================
-- §2. Modality Lattice
-- ================================================================

/-- Meet: greatest lower bound (most restrictive modality satisfying both).
    Mirrors `Modality::combine` in `linear/modality.rs`. -/
def Modality.meet : Modality → Modality → Modality
  | Unrestricted, x          => x
  | x,           Unrestricted => x
  | Linear,      _           => Linear
  | _,           Linear      => Linear
  | Affine,      Relevant    => Linear
  | Relevant,    Affine      => Linear
  | Affine,      Affine      => Affine
  | Relevant,    Relevant    => Relevant

/-- Join: least upper bound (least restrictive modality satisfying either).
    Mirrors `Modality::join` in `linear/modality.rs`. -/
def Modality.join : Modality → Modality → Modality
  | Linear,      x           => x
  | x,           Linear      => x
  | Unrestricted, _          => Unrestricted
  | _,           Unrestricted => Unrestricted
  | Affine,      Relevant    => Unrestricted
  | Relevant,    Affine      => Unrestricted
  | Affine,      Affine      => Affine
  | Relevant,    Relevant    => Relevant

-- ================================================================
-- §3. Modality Lattice Theorems
-- All are proved by exhaustive case analysis (16 cases each for 2-arg,
-- 64 for 3-arg). This is correct and complete for a 4-element type.
-- ================================================================

theorem modality_meet_comm (m n : Modality) :
    m.meet n = n.meet m := by
  cases m <;> cases n <;> rfl

theorem modality_meet_assoc (m n k : Modality) :
    (m.meet n).meet k = m.meet (n.meet k) := by
  cases m <;> cases n <;> cases k <;> rfl

theorem modality_join_comm (m n : Modality) :
    m.join n = n.join m := by
  cases m <;> cases n <;> rfl

theorem modality_join_assoc (m n k : Modality) :
    (m.join n).join k = m.join (n.join k) := by
  cases m <;> cases n <;> cases k <;> rfl

theorem modality_meet_idempotent (m : Modality) :
    m.meet m = m := by
  cases m <;> rfl

theorem modality_join_idempotent (m : Modality) :
    m.join m = m := by
  cases m <;> rfl

theorem modality_absorption_meet (m n : Modality) :
    m.meet (m.join n) = m := by
  cases m <;> cases n <;> rfl

theorem modality_absorption_join (m n : Modality) :
    m.join (m.meet n) = m := by
  cases m <;> cases n <;> rfl

/-- Linear is the bottom element: meet with anything gives Linear. -/
theorem modality_linear_is_bottom (m : Modality) :
    Linear.meet m = Linear := by
  cases m <;> rfl

/-- Unrestricted is the top element: join with anything gives Unrestricted. -/
theorem modality_unrestricted_is_top (m : Modality) :
    Unrestricted.join m = Unrestricted := by
  cases m <;> rfl

/-- Unrestricted is the identity for meet. -/
theorem modality_unrestricted_meet_id (m : Modality) :
    Unrestricted.meet m = m := by
  cases m <;> rfl

/-- Linear is the identity for join. -/
theorem modality_linear_join_id (m : Modality) :
    Linear.join m = m := by
  cases m <;> rfl

/-- Affine and Relevant are strictly incomparable:
    their meet is Linear (not either of them). -/
theorem modality_affine_relevant_incomparable :
    Affine.meet Relevant = Linear ∧ Relevant.meet Affine = Linear :=
  ⟨rfl, rfl⟩

-- ================================================================
-- §4. Structural Rule Properties
-- ================================================================

theorem linear_no_weakening    : Linear.allowsWeakening    = false := rfl
theorem linear_no_contraction  : Linear.allowsContraction  = false := rfl
theorem affine_allows_weakening : Affine.allowsWeakening   = true  := rfl
theorem affine_no_contraction  : Affine.allowsContraction  = false := rfl
theorem relevant_no_weakening  : Relevant.allowsWeakening  = false := rfl
theorem relevant_allows_contraction : Relevant.allowsContraction = true := rfl
theorem unrestricted_both_rules :
    Unrestricted.allowsWeakening = true ∧ Unrestricted.allowsContraction = true :=
  ⟨rfl, rfl⟩

/-- Weakening is monotone in the lattice:
    if m₁ ≤ m₂ (i.e., m₁.meet m₂ = m₁) and m₁ allows weakening,
    then m₂ also allows weakening. -/
theorem weakening_monotone (m₁ m₂ : Modality)
    (h_le : m₁.meet m₂ = m₁)
    (hw   : m₁.allowsWeakening = true) :
    m₂.allowsWeakening = true := by
  cases m₁ <;> cases m₂ <;>
    simp_all [Modality.meet, Modality.allowsWeakening]

-- ================================================================
-- §5. Usage Counts
-- ================================================================

/-- Three-valued usage count: unused / used once / used many times.
    Mirrors `UsageCount` in `linear/context.rs`. -/
inductive UsageCount where
  | Zero : UsageCount
  | One  : UsageCount
  | Many : UsageCount
  deriving DecidableEq, Repr

/-- Adding usage counts (for merging parallel branches).
    Zero is the identity; One + One = Many. -/
def UsageCount.add : UsageCount → UsageCount → UsageCount
  | .Zero, x     => x
  | x,    .Zero  => x
  | _,    _      => .Many

theorem usage_add_comm (u v : UsageCount) :
    u.add v = v.add u := by
  cases u <;> cases v <;> rfl

theorem usage_add_assoc (u v w : UsageCount) :
    (u.add v).add w = u.add (v.add w) := by
  cases u <;> cases v <;> cases w <;> rfl

theorem usage_zero_neutral_left (u : UsageCount) :
    UsageCount.Zero.add u = u := by
  cases u <;> rfl

theorem usage_zero_neutral_right (u : UsageCount) :
    u.add UsageCount.Zero = u := by
  cases u <;> rfl

-- ================================================================
-- §6. Well-Usedness Predicate
-- ================================================================

/-! A binding with modality `m` and actual usage count `u` is
    *well-used* iff `u` satisfies the resource discipline of `m`.

    Mirrors `LinearBinding::check_usage` in `linear/context.rs`. -/
def wellUsed (m : Modality) (u : UsageCount) : Prop :=
  match m, u with
  | Linear,       .One  => True
  | Affine,       .Zero => True
  | Affine,       .One  => True
  | Relevant,     .One  => True
  | Relevant,     .Many => True
  | Unrestricted, _     => True
  | _,            _     => False

instance decWellUsed (m : Modality) (u : UsageCount) : Decidable (wellUsed m u) :=
  match m, u with
  | Linear,       .One  => isTrue trivial
  | Affine,       .Zero => isTrue trivial
  | Affine,       .One  => isTrue trivial
  | Relevant,     .One  => isTrue trivial
  | Relevant,     .Many => isTrue trivial
  | Unrestricted, _     => isTrue trivial
  | Linear,       .Zero => isFalse id
  | Linear,       .Many => isFalse id
  | Affine,       .Many => isFalse id
  | Relevant,     .Zero => isFalse id

-- §6.1  Key structural rule theorems

/-- Affine allows zero use: weakening is sound. -/
theorem affine_zero_well_used : wellUsed Affine .Zero := trivial

/-- Linear with zero use is NOT well-used: no weakening. -/
theorem linear_zero_not_well_used : ¬wellUsed Linear .Zero := id

/-- Linear with many uses is NOT well-used: no contraction. -/
theorem linear_many_not_well_used : ¬wellUsed Linear .Many := id

/-- Relevant with zero use is NOT well-used: no weakening. -/
theorem relevant_zero_not_well_used : ¬wellUsed Relevant .Zero := id

/-- Unrestricted is always well-used, whatever the count. -/
theorem unrestricted_always_well_used (u : UsageCount) :
    wellUsed Unrestricted u := by
  cases u <;> exact trivial

-- ================================================================
-- §7. Typing Contexts
-- ================================================================

/-- A context entry: variable name, type (opaque), declared modality. -/
structure CtxEntry where
  name     : String
  ty       : String   -- type is opaque in the meta-theory
  modality : Modality

/-- A linear typing context is a list of entries. -/
abbrev Ctx := List CtxEntry

/-- A usage environment records how many times each variable was used. -/
abbrev UsageEnv := List (String × UsageCount)

/-- Look up a variable's usage count; default to Zero if absent. -/
def lookupUsage (ρ : UsageEnv) (x : String) : UsageCount :=
  match ρ.find? (fun e => e.1 == x) with
  | some (_, u) => u
  | none        => .Zero

/-- A context Γ is exhausted under ρ if every entry is well-used.
    Mirrors `LinearContext::check_exhausted` in `linear/context.rs`. -/
def Ctx.exhausted (Γ : Ctx) (ρ : UsageEnv) : Prop :=
  ∀ e ∈ Γ, wellUsed e.modality (lookupUsage ρ e.name)

-- ================================================================
-- §8. Structural Rule Theorems
-- ================================================================

/-! ### Weakening lemma

    If x has a weakenable modality and its usage is Zero,
    the entry can be removed from the context without violating exhaustion.

    Formally proves that Affine/Unrestricted variables may be dropped. -/
theorem weakening_lemma
    (Γ : Ctx) (ρ : UsageEnv) (x : String) (τ : String) (m : Modality)
    (_hm    : m.allowsWeakening = true)
    (_hzero : lookupUsage ρ x = .Zero)
    (hΓ    : Ctx.exhausted (⟨x, τ, m⟩ :: Γ) ρ) :
    Ctx.exhausted Γ ρ :=
  fun e he => hΓ e (List.mem_cons.mpr (.inr he))

/-! ### No-weakening lemma

    If x has a non-weakenable modality (Linear or Relevant),
    any exhausted context containing x forces a non-zero usage count. -/
theorem no_weakening_lemma
    (Γ : Ctx) (ρ : UsageEnv) (x : String) (τ : String) (m : Modality)
    (hm : m.allowsWeakening = false)
    (hΓ : Ctx.exhausted (⟨x, τ, m⟩ :: Γ) ρ) :
    lookupUsage ρ x ≠ .Zero := by
  have hu := hΓ ⟨x, τ, m⟩ List.mem_cons_self
  intro hzero
  rw [hzero] at hu
  cases m with
  | Linear       => exact hu
  | Affine       => simp [Modality.allowsWeakening] at hm
  | Relevant     => exact hu
  | Unrestricted => simp [Modality.allowsWeakening] at hm

-- ================================================================
-- §9. Bang (!) Promotion — Seely Isomorphism
-- ================================================================

/-! ### Promotion condition (Seely)

    `!Γ ⊢ !A` requires that all variables in Γ are Unrestricted.
    Mirrors `BangType::can_promote` in `linear/exponentials.rs`. -/
def Ctx.allUnrestricted (Γ : Ctx) : Prop :=
  ∀ e ∈ Γ, e.modality = Unrestricted

/-- If all context entries are Unrestricted, the context is exhausted
    under any usage environment — proving Bang promotion is always valid
    for Unrestricted contexts. -/
theorem promotion_exhausted_any
    (Γ : Ctx) (ρ : UsageEnv)
    (hall : Ctx.allUnrestricted Γ) :
    Ctx.exhausted Γ ρ :=
  fun e he => by
    have hm : e.modality = Unrestricted := hall e he
    rw [hm]
    exact unrestricted_always_well_used _

/-- Dereliction: Unrestricted variables are well-used at any count,
    in particular at One (a single use is always valid). -/
theorem dereliction_one_ok : wellUsed Unrestricted .One := trivial

-- ================================================================
-- §10. Distributive Lattice
-- ================================================================

/-- The modality lattice is distributive: meet distributes over join. -/
theorem modality_meet_distrib (m n k : Modality) :
    m.meet (n.join k) = (m.meet n).join (m.meet k) := by
  cases m <;> cases n <;> cases k <;> rfl

/-- The modality lattice is distributive: join distributes over meet. -/
theorem modality_join_distrib (m n k : Modality) :
    m.join (n.meet k) = (m.join n).meet (m.join k) := by
  cases m <;> cases n <;> cases k <;> rfl

-- ================================================================
-- §11. Extended Structural Rule Monotonicity
-- ================================================================

/-- Contraction is monotone in the lattice:
    if m1.meet m2 = m1 and m1 allows contraction, then m2 allows contraction. -/
theorem contraction_monotone (m1 m2 : Modality)
    (h_le : m1.meet m2 = m1)
    (hc   : m1.allowsContraction = true) :
    m2.allowsContraction = true := by
  cases m1 <;> cases m2 <;>
    simp_all [Modality.meet, Modality.allowsContraction]

/-- Weakening is preserved upward by join: if m allows weakening, so does m.join n. -/
theorem allowsWeakening_join (m n : Modality)
    (hw : m.allowsWeakening = true) :
    (m.join n).allowsWeakening = true := by
  cases m <;> cases n <;>
    simp_all [Modality.join, Modality.allowsWeakening]

/-- Contraction is preserved upward by join: if m allows contraction, so does m.join n. -/
theorem allowsContraction_join (m n : Modality)
    (hc : m.allowsContraction = true) :
    (m.join n).allowsContraction = true := by
  cases m <;> cases n <;>
    simp_all [Modality.join, Modality.allowsContraction]

/-- mustUse is preserved downward by meet:
    the meet of a must-use modality with anything stays must-use. -/
theorem mustUse_meet (m n : Modality)
    (hm : m.mustUse = true) :
    (m.meet n).mustUse = true := by
  cases m <;> cases n <;>
    simp_all [Modality.meet, Modality.mustUse]

-- ================================================================
-- §12. Additional WellUsed Properties
-- ================================================================

/-- Using a resource exactly once satisfies every modality. -/
theorem wellUsed_one_always (m : Modality) : wellUsed m .One := by
  cases m <;> exact trivial

/-- Relaxing a modality via join preserves well-usedness. -/
theorem wellUsed_join_left (m n : Modality) (u : UsageCount)
    (hw : wellUsed m u) : wellUsed (m.join n) u := by
  cases m <;> cases n <;> cases u <;>
    simp_all [wellUsed, Modality.join]

/-- Strengthening a modality via meet preserves non-well-usedness. -/
theorem not_wellUsed_meet (m n : Modality) (u : UsageCount)
    (hn : ¬wellUsed m u) : ¬wellUsed (m.meet n) u := by
  cases m <;> cases n <;> cases u <;>
    simp_all [wellUsed, Modality.meet]

-- ================================================================
-- §13. Usage Count Arithmetic
-- ================================================================

/-- Many is a left-absorbing element for usage addition. -/
theorem usage_many_absorb_left (u : UsageCount) :
    UsageCount.Many.add u = .Many := by
  cases u <;> rfl

/-- Many is a right-absorbing element for usage addition. -/
theorem usage_many_absorb_right (u : UsageCount) :
    u.add .Many = .Many := by
  cases u <;> rfl

/-- One plus One equals Many. -/
theorem usage_one_plus_one :
    UsageCount.One.add .One = .Many := rfl

/-- The sum is Zero iff both summands are Zero. -/
theorem usage_add_zero_iff (u v : UsageCount) :
    u.add v = .Zero ↔ u = .Zero ∧ v = .Zero := by
  cases u <;> cases v <;> simp [UsageCount.add]

/-- The sum is Many iff at least one is Many, or both are One. -/
theorem usage_add_many_iff (u v : UsageCount) :
    u.add v = .Many ↔ u = .Many ∨ v = .Many ∨ (u = .One ∧ v = .One) := by
  cases u <;> cases v <;> simp [UsageCount.add]


end Sounio.Linear
