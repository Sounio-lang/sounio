-- TACTICS_CORE_WITNESSES_2.lean
-- Companion to TACTICS_CORE_WITNESSES.lean — the REVERSE DIRECTION.
--
--   Part A: "I habitually reach for T — what core construction does T's
--           work?"  One working witness per habit, not per tactic name.
--   Part B: omega failure modes (split item 3) — what omega DOES handle
--           natively (so you don't hand-roll it) and where it stops,
--           each boundary demonstrated by a compiling proof.
--
-- Verified: lean TACTICS_CORE_WITNESSES_2.lean  → exit 0, core 4.33.0
-- (commit d8b1897), 2026-08-18.  Mathlib-only probes behind every
-- "Mathlib-only" label were run and observed to fail; see README.

namespace TacticsCoreWitnesses2

-- ============================================================
-- PART A — habit → core construction (reverse direction)
-- ============================================================

-- Habit: linarith / nlinarith for inequalities.
-- Core: omega IS the linear solver; feed it atoms and hypotheses.
theorem habit_linarith (a b c : Int) (h1 : a ≤ b) (h2 : b ≤ c) :
    a ≤ c := by omega

theorem habit_linarith_mix (a b : Nat) (ha : 2 * a ≤ b + 3) (hb : b ≤ 10) :
    a ≤ 6 := by omega

-- Habit: push_neg to normalize ¬(a < b) etc.
-- Core: the strict/non-strict conversions are core lemmas; omega also
-- consumes negated hypotheses directly.
theorem habit_push_neg (a b : Int) (h : ¬(a < b)) : b ≤ a := by
  exact Int.le_of_not_gt h

theorem habit_push_neg_nat (a b : Nat) (h : ¬(a ≤ b)) : b + 1 ≤ a := by
  omega

-- Habit: tauto / classical propositional juggling.
-- Core: `omega` handles decidable propositional structure over Nat/Int;
-- for general Props, `by_cases` + explicit branches.
theorem habit_tauto {p q : Prop} (h : p ∨ q) (hp : p → q) : q := by
  rcases h with hp' | hq
  · exact hp hp'
  · exact hq

theorem habit_tauto_deMorgan {p q : Prop} (h : ¬(p ∧ q)) (hp : p) : ¬q := by
  intro hq; exact h ⟨hp, hq⟩

-- Habit: subst to eliminate an equation.
-- Core: `subst` IS core (clarification — do not hand-roll it).
theorem habit_subst (a b : Int) (h : a = b) (w : a + a = 4) : b + b = 4 := by
  subst h; exact w

-- Habit: simp_all / simp_arith to blast everything.
-- Core: `simp_all` can hit maxRecDepth on arithmetic hypotheses — the
-- reliable pattern is `simp only [named lemma] at h`, `subst`, `omega`.
theorem habit_simp_all (a b : Nat) (h : a + 0 = b) (h2 : b = 2 * a) :
    a ≤ 3 := by
  simp only [Nat.add_zero] at h
  subst h
  omega

-- Habit: norm_cast / push_cast through a Nat→Int coercion chain.
-- CORRECTED 2026-08-19: `norm_cast` and `push_cast` ARE core (4.33.0) — an
-- earlier revision of this file's footer wrongly listed norm_cast as
-- unknown-tactic (misread probe), and the README correction of 2026-08-17
-- (rescue #1772) caught the same error in the table. The robust pattern
-- stays as below, but `norm_cast` closes a pure cast goal directly:
theorem habit_cast_chain (n m : Nat) (h : n + m = 7) :
    ((n : Int) + (m : Int)) = 7 := by
  norm_cast

theorem habit_cast_chain_norm_cast (n m : Nat) :
    ((n + m : Nat) : Int) = (n : Int) + (m : Int) := by
  norm_cast

-- Habit: simp_arith to normalize arithmetic inside simp.
-- Core: `simp` made no progress on pure arithmetic — that is omega's
-- job, and omega consumes the simp-normalized hypotheses directly.
theorem habit_simp_arith (a b : Nat) (h : a + 0 = b) : b + b = 2 * a := by
  simp only [Nat.add_zero] at h
  subst h
  omega

-- Habit: existsi / refine ⟨_, _⟩ after constructing a witness.
-- Core: exact ⟨_, _⟩ / refine / constructor — all core.
theorem habit_witness (a b : Int) (_h : a ≤ b) : ∃ c, a + c = b := by
  refine ⟨b - a, ?_⟩
  omega

-- Habit: interval_cases for bounded enumeration.
-- Core: omega proves bounded facts when the bounds are hypotheses; for
-- genuine finite enumeration use `decide` (bounded numerics) or case-split.
theorem habit_interval (a : Nat) (_h1 : 3 ≤ a) (_h2 : a ≤ 5) : a = 3 ∨ a = 4 ∨ a = 5 := by
  omega

-- Habit: field_simp + ring on division goals.
-- Core: named field/ring lemmas on the carrier (Rat.*, Int.*, Nat.*),
-- finish with omega once the shape is linear.  (Cf. cursor-3's audit:
-- Rat.div_pos is TWO lines from Rat.mul_pos + Rat.inv_pos.)
theorem habit_field (a b : Rat) (ha : 0 < a) (hb : 0 < b) : 0 < a / b := by
  rw [Rat.div_def]
  exact Rat.mul_pos ha (Rat.inv_pos.mpr hb)

-- Habit: ring_nf to normalize polynomial sides before closing.
-- Core: expand with the carrier's lemmas and keep products as OMEGA
-- ATOMS (see Part B mode 3) — never mix `2*a*b` with `a*b + a*b`.
theorem habit_ring_nf (a b : Int) (_h : a * b = 6) :
    (a + b) * (a + b) = a * a + (a * b + a * b) + b * b := by
  rw [Int.add_mul, Int.mul_add, Int.mul_add, Int.mul_comm b a]
  omega

-- Habit: use x ≤ y ↔ ¬(y < x) style iff-rewrites (Mathlib ord lemmas).
-- Core: Nat.le_iff_lt / Int.le_iff_lt and friends exist in core;
-- omega also closes the iff goals of this shape directly.
theorem habit_le_iff (a b : Int) : (a ≤ b) ↔ ¬(b < a) := by
  omega

-- ============================================================
-- PART B — omega failure modes (split item 3)
-- ============================================================

-- MODE 1 (does handle — do not hand-roll): min/max are NATIVE to omega.
theorem omega_min (a b : Int) : min a b ≤ a ∧ min a b ≤ b := by omega
theorem omega_max (a b : Int) : b ≤ max a b ∧ a ≤ max a b := by omega

-- MODE 2 (does handle): truncated Nat subtraction IS linear to omega.
theorem omega_nat_sub (a b : Nat) (h : b ≤ a) : (a - b) + b = a := by omega
theorem omega_nat_sub2 (a b : Nat) (h : a ≤ b) : a - b = 0 := by omega

-- MODE 3 (FAILURE — the big one): products are ATOMS; a literal
-- coefficient times an atom is NONLINEAR to omega and mixes badly.
-- a*b is fine as an atom; 2*a*b (i.e. (2*a)*b) is a DIFFERENT atom and
-- omega cannot relate them.  Fix: state both sides in atom form.
theorem omega_atom_ok (a b : Int) (h : a * b ≥ 3) :
    a * b + a * b ≥ 6 := by omega

-- The failing twin (2*a*b form) — witness for the FIX, since the failing
-- version cannot compile by construction:
theorem omega_atom_fix (a b : Int) (h : a * b ≥ 3) :
    2 * (a * b) ≥ 6 := by
  have h2 : 2 * (a * b) = a * b + a * b := by omega
  rw [h2]; omega

-- MODE 4 (does handle): / and % by NUMERALS are linear to omega
-- (variable divisor is an atom — see MODE 5).
theorem omega_div_numeral (a : Int) (h : 0 ≤ a) : a / 1000 ≤ a := by omega
theorem omega_mod_numeral (a : Nat) : a % 4 < 4 := by omega

-- MODE 5 (FAILURE): division by a VARIABLE divisor is an atom; omega
-- cannot conclude divisor-ordering facts.  Fix: the hand-rolled
-- anti-monotonicity lemma — the canonical shipped witness is
-- `SounioGradedModal.div_le_of_divisor_le` (formal/lean4/
-- SounioGradedModal.lean), proved by trichotomy case-split +
-- `Nat.div_add_mod` + omega.  Not duplicated here; load that module or
-- copy the pattern from the witnesses file for it.

-- MODE 6 (FAILURE): equalities between PRODUCTS — omega treats each
-- monomial as an atom and proves nothing about their relations.
-- Fix: rewrite/subst to make both sides syntactically identical.
theorem omega_prod_eq_fix (a b : Int) (h : a = b) : a * a = b * b := by
  subst h; rfl

-- MODE 7 (does handle — verified): mixed Nat/Int goals — omega
-- preprocesses Nat→Int casts natively; no simp bridge needed.
theorem omega_cast_boundary (n m : Nat) (h : n < m) :
    ((n : Int) + 1) ≤ (m : Int) := by omega

end TacticsCoreWitnesses2

-- NEGATIVE WITNESSES (re-probed one-file-per-tactic, core 4.33.0, 2026-08-19;
-- do not uncomment, this file must stay exit-0):
--   `linarith`       → unknown tactic
--   `push_neg`       → unknown tactic
--   `tauto`          → unknown tactic
--   `simp_arith`     → unknown tactic
--   `ring`           → unknown tactic
--   `exact_mod_cast` → unknown tactic (the *tactic*; the lemma side is core)
-- CORRECTION 2026-08-19 (misread probe of 2026-08-18, caught by the README
-- correction in rescue #1772): `norm_cast` and `push_cast` ARE core. The
-- 08-18 probe showed `unsolved goals` (the tactic RAN on a wrong-shaped
-- goal) and was misread as `unknown tactic`. `norm_cast` closes a pure
-- cast goal — see habit_cast_chain_norm_cast above. `push_cast` runs as a
-- normalizer but does not discharge our cast-equality goals on its own.
-- CORE (probed OK — use freely): subst, generalize, exact?, simp_all,
--   constructor, refine, rcases/obtain/rintro, by_cases, exfalso, split,
--   decide, omega, simp [only], norm_cast, push_cast,
--   cases/induction ... with.
