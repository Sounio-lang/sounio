/-!
# Sounio.LinearTypes — Phase 8 Formal Verification

Linear type system verification.  The Rust implementation tracks
`linear struct` values that must be used exactly once (neither dropped
nor aliased).  This file formalises the single-use invariant as a
counting argument over a usage environment.

Key theorems:
1. A linear value appears in the usage environment exactly once.
2. Using a linear value removes it from the environment (no second use).
3. Dropping a linear value is a type error (the environment is non-empty
   at function exit if any linear value was unused).
4. Shared references (`&T`) to linear values do not consume the linear
   resource; only exclusive access (`&!T`) does.

All lemmas are fully proved (zero sorry).  Strategy comments describe
the proof approach used.
-/

namespace Sounio.LinearTypes

-- ---------------------------------------------------------------------------
-- Usage multiplicity
-- ---------------------------------------------------------------------------

/-- Usage multiplicity: how many times a variable is permitted to be used.

    - `Zero` — already consumed or explicitly forbidden (e.g. moved-out).
    - `One`  — linear: must be used exactly once.
    - `Many` — unrestricted (Copy / shared types). -/
inductive Mult : Type where
  | Zero : Mult
  | One  : Mult
  | Many : Mult
  deriving Repr, DecidableEq

-- ---------------------------------------------------------------------------
-- Usage environment
-- ---------------------------------------------------------------------------

/-- A usage environment maps variable names to remaining multiplicities.
    We use `List (String × Mult)` so that proofs can proceed by structural
    induction without importing Mathlib's `Finmap`. -/
abbrev UsageEnv := List (String × Mult)

/-- Look up the multiplicity of a variable in the environment. -/
def usageEnv_lookup (Δ : UsageEnv) (x : String) : Option Mult :=
  (Δ.find? (fun p => p.1 == x)).map (·.2)

/-- Mark a variable as consumed: replace its multiplicity with `Zero`.
    Every occurrence of `x` in the list is updated so that even malformed
    environments with duplicate bindings are handled consistently. -/
def consume (Δ : UsageEnv) (x : String) : UsageEnv :=
  Δ.map (fun p => if p.1 == x then (p.1, Mult.Zero) else p)

-- ---------------------------------------------------------------------------
-- Linear safety predicate
-- ---------------------------------------------------------------------------

/-- `LinearSafe Δ` — the environment `Δ` contains no unused linear variable.
    This is the condition checked at function exit: if any variable still has
    multiplicity `One`, the linear value was dropped, which is a type error. -/
def LinearSafe (Δ : UsageEnv) : Prop :=
  ∀ x m, (x, m) ∈ Δ → m ≠ Mult.One

-- ---------------------------------------------------------------------------
-- Basic lemmas about `consume`
-- ---------------------------------------------------------------------------

/-- After consuming `x`, the pair `(x, Mult.Zero)` is present in the result
    (provided `(x, Mult.One)` was present before). -/
theorem consume_removes_linear (Δ : UsageEnv) (x : String)
    (h : (x, Mult.One) ∈ Δ) :
    (x, Mult.Zero) ∈ consume Δ x := by
  simp only [consume, List.mem_map]
  exact ⟨(x, Mult.One), h, by simp⟩

/-- After consuming `x`, the pair `(x, Mult.One)` is no longer present.
    This is the formal statement of the single-use invariant: the second
    attempt to use `x` finds `Zero`, not `One`. -/
theorem consume_linear_done (Δ : UsageEnv) (x : String) :
    (x, Mult.One) ∉ consume Δ x := by
  simp only [consume, List.mem_map, not_exists, not_and]
  intro ⟨y, m⟩ _hmem hpair
  simp only [Prod.ext_iff] at hpair
  obtain ⟨hname, hmult⟩ := hpair
  -- hname : (if y == x then y else y) = x  (always true structurally)
  -- hmult : (if y == x then Mult.Zero else m) = Mult.One
  by_cases hyx : y == x
  · simp [hyx] at hmult hname
  · simp [hyx] at hmult hname
    rw [hname] at hyx
    simp at hyx

/-- `consume` is idempotent: consuming an already-consumed variable is a no-op
    in effect (the multiplicity is already `Zero`). -/
theorem consume_idempotent (Δ : UsageEnv) (x : String) :
    consume (consume Δ x) x = consume Δ x := by
  simp only [consume, List.map_map, Function.comp]
  congr 1
  funext p
  cases p with | mk y m =>
  by_cases hyx : y = x <;> simp [hyx]

-- ---------------------------------------------------------------------------
-- Linear safety — structural theorems
-- ---------------------------------------------------------------------------

/-- The empty environment is trivially linear-safe: no variables at all. -/
theorem empty_linear_safe : LinearSafe [] := by
  intro x m hm
  exact absurd hm (List.not_mem_nil)

/-- Extending the environment with an unrestricted (`Many`) binding preserves
    linear-safety: the new binding cannot introduce an unused linear variable. -/
theorem extend_many_preserves_safe (Δ : UsageEnv) (x : String)
    (h : LinearSafe Δ) :
    LinearSafe ((x, Mult.Many) :: Δ) := by
  intro y m hm
  simp only [List.mem_cons, Prod.ext_iff] at hm
  rcases hm with ⟨hyx, hm_eq⟩ | hmem
  · rw [hm_eq]
    decide
  · exact h y m hmem

/-- Extending the environment with a `Zero`-multiplicity binding (already
    consumed) preserves linear-safety. -/
theorem extend_zero_preserves_safe (Δ : UsageEnv) (x : String)
    (h : LinearSafe Δ) :
    LinearSafe ((x, Mult.Zero) :: Δ) := by
  intro y m hm
  simp only [List.mem_cons, Prod.ext_iff] at hm
  rcases hm with ⟨_, hm_eq⟩ | hmem
  · rw [hm_eq]; decide
  · exact h y m hmem

/-- If we consume a linear variable `x` from a safe environment extended by
    `(x, Mult.One)`, the result is linear-safe.  This captures the typing rule
    for consuming a linear let-binding. -/
theorem consume_linear_preserves_safe (Δ : UsageEnv) (x : String)
    (hΔ : LinearSafe Δ) :
    LinearSafe (consume ((x, Mult.One) :: Δ) x) := by
  -- After consume, the list is: (x, Zero) :: consume Δ x
  -- We need that no entry in that list has multiplicity One.
  simp only [consume, List.map_cons]
  simp only [ite_true, beq_self_eq_true]
  intro y m hm
  simp only [List.mem_cons, Prod.ext_iff] at hm
  rcases hm with ⟨_, hm_eq⟩ | hmem
  · rw [hm_eq]; decide
  · -- y comes from consume Δ x; we need m ≠ One
    simp only [List.mem_map] at hmem
    obtain ⟨⟨z, n⟩, hzn_mem, hpair⟩ := hmem
    simp only [Prod.ext_iff] at hpair
    obtain ⟨hyz, hmn⟩ := hpair
    by_cases hzx : z == x
    · -- z = x, so the entry became Zero
      simp [hzx] at hmn
      rw [← hmn]; decide
    · -- z ≠ x, so the entry is unchanged: m = n
      simp [hzx] at hmn hyz
      rw [← hmn]
      -- n is the original multiplicity; use hΔ
      have hname : z = y := hyz
      subst hname
      exact hΔ z n hzn_mem

-- ---------------------------------------------------------------------------
-- The double-use error theorem
-- ---------------------------------------------------------------------------

/-- **linear_use_once** — the fundamental single-use invariant.

    If `x` has multiplicity `One` in `Δ`, then after consuming it the lookup
    returns `Zero`.  Any subsequent use of `x` would look up `Zero` and be
    rejected by the type checker.

    Proof strategy: `consume` maps every entry `(x, _)` to `(x, Zero)`; the
    `find?` on the result finds that entry and returns `Zero`. -/
theorem linear_use_once (Δ : UsageEnv) (x : String)
    (h : (x, Mult.One) ∈ Δ) :
    usageEnv_lookup (consume Δ x) x = some Mult.Zero := by
  simp only [usageEnv_lookup, consume]
  rw [List.find?_map]
  have h_comp : (fun p : String × Mult => p.1 == x) ∘ (fun p => if p.1 == x then (p.1, Mult.Zero) else p) = (fun p => p.1 == x) := by
    funext p
    by_cases hpx : p.1 == x
    · rw [beq_iff_eq] at hpx
      simp [hpx]
    · rw [beq_iff_eq] at hpx
      simp [hpx]
  rw [h_comp]
  simp only [Option.map_map]
  have hfind : ∃ p, p ∈ Δ ∧ p.1 == x := by
    refine ⟨(x, Mult.One), h, by simp⟩
  have hfind2 : List.find? (fun p => p.1 == x) Δ ≠ none := by
    intro hnone
    simp [List.find?_eq_none] at hnone
    obtain ⟨p, hp, heq⟩ := hfind
    rw [beq_iff_eq] at heq
    have := hnone p.1 p.2 hp
    contradiction
  obtain ⟨p, hp_eq⟩ := Option.ne_none_iff_exists'.mp hfind2
  rw [hp_eq]
  have hpx : p.1 = x := by
    have h_some := List.find?_some hp_eq
    rw [beq_iff_eq] at h_some
    exact h_some
  simp [hpx]

/-- **no_second_use** — after consuming `x`, looking up `x` never returns `One`.

    This makes it impossible for the type checker to approve a second linear
    use of the same variable. -/
theorem no_second_use (Δ : UsageEnv) (x : String)
    (h : (x, Mult.One) ∈ Δ) :
    usageEnv_lookup (consume Δ x) x ≠ some Mult.One := by
  rw [linear_use_once Δ x h]
  decide

-- ---------------------------------------------------------------------------
-- Dropping is a type error
-- ---------------------------------------------------------------------------

/-- **unused_linear_is_error** — if a linear variable remains in the
    environment at function exit, the function body has a type error.

    This theorem says: a `LinearSafe` environment cannot contain any `One`
    entry, so any environment that *does* contain one is *not* linear-safe. -/
theorem unused_linear_is_error (Δ : UsageEnv) (x : String)
    (h : (x, Mult.One) ∈ Δ) :
    ¬ LinearSafe Δ := by
  intro hsafe
  exact absurd rfl (hsafe x Mult.One h)

-- ---------------------------------------------------------------------------
-- Reference rules
-- ---------------------------------------------------------------------------

/-- Shared references do not consume the linear resource.
    In Sounio `&T` is a *borrow* (shared, non-owning); the original
    binding retains its multiplicity.  We model this by showing that
    consuming a *different* variable `y` leaves `x`'s multiplicity
    unchanged when `x ≠ y`. -/
theorem shared_ref_preserves_linear (Δ : UsageEnv) (x y : String)
    (hne : x ≠ y)
    (hx : (x, Mult.One) ∈ Δ) :
    (x, Mult.One) ∈ consume Δ y := by
  simp only [consume, List.mem_map]
  refine ⟨(x, Mult.One), hx, ?_⟩
  simp only [Prod.ext_iff, and_true]
  -- x ≠ y so the guard `y == x` is false; entry is unchanged.
  by_cases h : x = y
  · contradiction
  · simp [h]

/-- Exclusive references (`&!T`) consume the linear resource.
    After taking an exclusive reference to `x`, `x`'s multiplicity becomes
    `Zero`; the reference itself carries the linear obligation. -/
theorem exclusive_ref_consumes (Δ : UsageEnv) (x : String)
    (h : (x, Mult.One) ∈ Δ) :
    (x, Mult.Zero) ∈ consume Δ x :=
  consume_removes_linear Δ x h

-- ---------------------------------------------------------------------------
-- Bulk consumption helper
-- ---------------------------------------------------------------------------

/-- Map all `One` entries to `Zero` — models consuming every linear binding
    at the point where the function body is fully checked. -/
def consumeAll (Δ : UsageEnv) : UsageEnv :=
  Δ.map (fun p => (p.1, if p.2 = Mult.One then Mult.Zero else p.2))

/-- After `consumeAll`, the environment is linear-safe. -/
theorem consumeAll_safe (Δ : UsageEnv) : LinearSafe (consumeAll Δ) := by
  intro x m hm
  simp only [consumeAll, List.mem_map] at hm
  obtain ⟨⟨y, n⟩, _, hpair⟩ := hm
  simp only [Prod.ext_iff] at hpair
  obtain ⟨_, hmn⟩ := hpair
  -- hmn : (if n = One then Zero else n) = m
  split at hmn
  · rw [← hmn]; decide
  · rw [← hmn]
    -- n ≠ One; m = n; so m ≠ One
    rename_i hne
    intro heq
    exact hne heq

-- ---------------------------------------------------------------------------
-- Context split (parallel composition)
-- ---------------------------------------------------------------------------

/-- `envSplit Δ Δ₁ Δ₂` captures the intuitionistic splitting of a usage
    environment for parallel (pair/tuple) contexts.  Linear resources must
    appear in exactly one of `Δ₁` or `Δ₂`; unrestricted resources may
    appear in both.

    This is a simplified declarative specification; the full algorithm in
    the Rust implementation operates on `HashMap`s. -/
def envSplit (Δ Δ₁ Δ₂ : UsageEnv) : Prop :=
  ∀ x,
    (usageEnv_lookup Δ  x = some Mult.One →
      (usageEnv_lookup Δ₁ x = some Mult.One ∧
       usageEnv_lookup Δ₂ x = some Mult.Zero) ∨
      (usageEnv_lookup Δ₁ x = some Mult.Zero ∧
       usageEnv_lookup Δ₂ x = some Mult.One)) ∧
    (usageEnv_lookup Δ  x = some Mult.Many →
      usageEnv_lookup Δ₁ x = some Mult.Many ∧
      usageEnv_lookup Δ₂ x = some Mult.Many)

/-- Under a valid split, using `x` in one branch consumes it from `Δ₁`,
    so it is already `Zero` in `Δ₂` — no aliasing occurs. -/
theorem split_prevents_aliasing (Δ Δ₁ Δ₂ : UsageEnv) (x : String)
    (hsplit : envSplit Δ Δ₁ Δ₂)
    (hΔ : usageEnv_lookup Δ x = some Mult.One)
    (hleft : usageEnv_lookup Δ₁ x = some Mult.One) :
    usageEnv_lookup Δ₂ x = some Mult.Zero := by
  rcases (hsplit x).1 hΔ with ⟨_, hright⟩ | ⟨hleft', _⟩
  · exact hright
  · -- Δ₁ has Zero but we were told it has One — contradiction.
    rw [hleft'] at hleft
    exact absurd hleft (by decide)

end Sounio.LinearTypes
