-- TACTICS_CORE_WITNESSES.lean
-- Compile-verified witnesses for formal/lean4/README.md's core-vs-Mathlib
-- tactic table.  Run:  lean TACTICS_CORE_WITNESSES.lean   (exit 0 = verified)
--
-- Every `theorem` below is the core-only replacement WORKING.  The
-- Mathlib-only tactics themselves are NOT written here — each one was
-- separately confirmed to fail with `unknown tactic` / `unknown constant`
-- in core 4.33.0 before this table was written (see README "how verified").
--
-- Lean core 4.33.0 (commit d8b1897), the lean-toolchain pin.  Verified
-- 2026-08-17.  Context: cursor-3 burned eight attempts on unknown-tactic
-- errors from `set`, `by_contra`, `push_cast` before this table existed.

namespace TacticsCoreWitnesses

-- ============================================================
-- 1. by_contra  →  cases Nat.lt_or_ge / rcases Int.le_total
-- ============================================================
-- by_contra is Mathlib (it pulls in Classical + push_neg normalization).
-- Core pattern for ≤ goals: case-split on the trichotomy, derive the good
-- branch constructively, kill the bad branch with omega.  Both witnesses
-- below are the exact patterns used in the shipped kernel-clean files
-- (EpistemicEffects.int_sq_nonneg, SounioGradedModal.div_le_of_divisor_le).

theorem witness_by_contra_Nat (a b : Nat) (h : a ≤ b) : a ≤ b + 1 := by
  cases Nat.lt_or_ge a (b + 1) with
  | inl hlt => exact Nat.le_of_lt hlt   -- good branch: strict < gives ≤
  | inr hge => omega                    -- bad branch: arithmetic contradiction

theorem witness_by_contra_Int (x : Int) : 0 ≤ x * x := by
  rcases Int.le_total 0 x with h | h
  · exact Int.mul_nonneg h h
  · have h2 : 0 ≤ -x := by omega
    have h3 := Int.mul_nonneg h2 h2
    rwa [Int.neg_mul_neg] at h3

-- For a general Prop P with a Decidable instance, `by_cases` (which IS core)
-- gives the same shape; without Decidable use `Classical.em` explicitly and
-- do the ¬¬-elimination by hand — there is no push_neg to normalize.

-- ============================================================
-- 2. set x := e with hx  →  have hx : x = e := rfl  +  rw
-- ============================================================
-- `set` is Mathlib.  Core: name the abstraction with `have ... := rfl`
-- (or `show` to restate the goal) and rewrite.  Closed numerics close with
-- `decide`, never `norm_num` (Mathlib).

theorem witness_set (a b : Nat) (w : a + b = 4) :
    (a + b) + (a + b) = 8 := by
  have h2 : (a + b) + (a + b) = (a + b) * 2 := by omega
  rw [h2, w]

-- ============================================================
-- 3. push_cast  →  simp
-- ============================================================
-- push_cast is Mathlib (norm_cast family).  In core 4.33.0 the cast lemmas
-- are NOT addressable constants — `Nat.cast_add` and `Int.ofNat_add` are
-- unknown identifiers to #check — but they ARE in the core simp set, so
-- plain `simp` closes cast goals.  Prefer hypothesis-`rw` first, then simp.

theorem witness_push_cast (n m : Nat) (h : n = m) :
    (n : Int) + (m : Int) = ((n + m : Nat) : Int) := by
  rw [h]
  simp

-- ============================================================
-- 4. nlinarith / positivity  →  monotone core lemma + omega
-- ============================================================
-- Nonlinear goals: establish the monotonicity facts with named core lemmas
-- (`Nat.mul_le_mul_left/right`, `Nat.pow_le_pow_right`), then omega finishes.
-- This witness IS SounioGradedModal.graded_app_rule, shipped kernel-clean.

theorem witness_nonlinear (a b : Nat) (h1 : a ≤ 1000) (h2 : b ≤ 1000) :
    a * b / 1000 ≤ min a b := by
  have ha : a * b ≤ a * 1000 := Nat.mul_le_mul_left _ h2
  have hb : a * b ≤ 1000 * b := Nat.mul_le_mul_right _ h1
  omega

-- ============================================================
-- 5. ring  →  distribute by named core lemma, finish with omega
-- ============================================================
-- `ring` is Mathlib.  Core: expand with Int.add_mul / Int.mul_add, commute
-- with Int.mul_comm, and state the target so every product is an OMEGA
-- ATOM — omega is linear, so keep `a*b` as an atom and never mix it with
-- literal-coefficient forms like `2*a*b` ((2*a)*b is nonlinear to omega;
-- a*b + a*b is the same fact in linear form).

theorem witness_ring (a b : Int) :
    (a + b) * (a + b) = a*a + (a*b + a*b) + b*b := by
  rw [Int.add_mul, Int.mul_add, Int.mul_add, Int.mul_comm b a]
  omega

-- ============================================================
-- 6. min / if under a bound  →  split <;> omega
-- ============================================================
-- Core omega does not split if/min by itself for non-goals; `split <;>
-- omega` does.  (Pattern from EpistemicEffects' confidence monotonicity,
-- shipped kernel-clean.)

theorem witness_min_split (a b : Int) : (if a ≤ b then a else b) ≤ a := by
  split <;> omega

-- ============================================================
-- 7. What IS core (don't replace these)
-- ============================================================
-- `rcases` / `obtain` / `rintro` / `by_cases` / `exfalso` / `omega` /
-- `simp` / `simp only` / `split` / `decide` / `cases ... with` / `induction
-- ... with` / `constructor` / `exact?`-free term proofs — all core.  The
-- failures cursor-3 hit were `set`, `by_contra`, `push_cast` specifically.

theorem witness_obtain_ok {p q : Prop} (h : p ∧ q) : q := by
  obtain ⟨_, hq⟩ := h
  exact hq

theorem witness_by_cases_ok (n : Nat) : n + 0 = n := by
  by_cases h : n = 0
  · rw [h]
  · omega

-- ---------------------------------------------------------------------------
-- `omega` is stronger than the `linarith` habit assumes.
-- Probes by kimi-cli1, re-verified here rather than trusted.

theorem witness_omega_min (a b : Int) : min a b ≤ a := by omega
theorem witness_omega_max (a b : Int) : max a b ≥ b := by omega
theorem witness_omega_sub_zero (a b : Nat) (h : a ≤ b) : a - b = 0 := by omega
theorem witness_omega_sub_add (a b : Nat) (h : b ≤ a) : (a - b) + b = a := by omega

/-- `omega` crosses `Nat → Int` unaided, truncated subtraction included, so the
whole `push_cast` question usually does not arise for linear goals. -/
theorem witness_omega_cast (a b : Nat) (h : a ≤ b) :
    ((b - a : Nat) : Int) = (b : Int) - (a : Int) := by omega

-- ---------------------------------------------------------------------------
-- A REFUSAL IS NOT A LIMITATION.
-- kimi-cli1 probed `0 ≤ a → 0 ≤ a * b / 2 + 1`; `omega` refused, and the first
-- read was an `omega` limitation.  Checking the goal reversed that: the goal is
-- FALSE for negative `b`, so the refusal was correct.  Witness the falsity with
-- a concrete counterexample rather than asserting it.

theorem witness_refusal_was_correct :
    ¬ (∀ a b : Int, 0 ≤ a → 0 ≤ a * b / 2 + 1) := by
  intro hall
  have h := hall 2 (0 - 2) (by decide)   -- a = 2, b = -2:  2*(-2)/2 + 1 = -1
  exact absurd h (by decide)

/-- The repaired statement, which `omega` does prove once `b` is constrained. -/
theorem witness_refusal_repaired (a b : Int) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    0 ≤ a * b / 2 + 1 := by
  have : 0 ≤ a * b := Int.mul_nonneg ha hb
  omega

-- ---------------------------------------------------------------------------
-- Core tactics kimi-cli1 probed and confirmed present (no `unknown tactic`).

theorem witness_subst (a b : Int) (h : a = b) : b = a := by subst h; rfl

theorem witness_generalize (a b : Nat) : (a + b) + 0 = a + b := by
  generalize a + b = s
  omega

end TacticsCoreWitnesses

-- NEGATIVE WITNESSES — each line below was actually compiled against core
-- 4.33.0 on 2026-08-17 and observed to fail.  Do not uncomment in this file
-- (it must stay exit-0); the failures were verified in a throwaway probe:
--   `by_contra h`             → unknown tactic
--   `set x := a + b with hx`  → unknown tactic
--   `nlinarith`               → unknown tactic
--   `linarith`                → unknown tactic
--   `push_neg`                → unknown tactic
--   `tauto`                   → unknown tactic
--   `positivity`              → unknown tactic
--   `ring`                    → unknown tactic
--   `field_simp`              → unknown tactic
--   `norm_num`                → unknown tactic
--   `interval_cases n`        → unknown tactic
--   `Nat.cast_add`            → unknown constant (cast lemmas: simp only)
--   `Int.ofNat_add`           → unknown constant (cast lemmas: simp only)
--
-- CORRECTED 2026-08-17: `push_cast` was listed here as `unknown tactic`.  That
-- is FALSE.  `push_cast`, `norm_cast` and `exact_mod_cast` are all in core and
-- all run under 4.33.0; so are `subst`, `generalize` and `exact?`.  The tell is
-- the KIND of error: an absent tactic gives `unknown tactic`, whereas a present
-- tactic that cannot finish gives `unsolved goals`.  A probe that only checks
-- that something FAILS cannot tell those apart, which is how the row got in.
-- Positive witnesses for all of the above: `CorePatternsWitnesses.lean` §P11.
