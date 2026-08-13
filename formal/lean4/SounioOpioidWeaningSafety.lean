-- formal/lean4/SounioOpioidWeaningSafety.lean
/-!
# Sounio.Weaning — Opioid Weaning Safety (F5c) — Lean 4 Discharge

Formal core of the **P3 proof-carrying weaning protocol** (UTIp ed cohort,
N = 61, Campinas-SP). This file discharges, in **pure ℚ** (Lean 4 core
`Rat`, Mathlib-free, no `sorry`, no `native_decide`, no IEEE-754 axioms),
the structural theorem behind the E5 counterfactual arm:

    taper −20 %/day + methadone PCC from day 1  ⇒  W(t) = 0 ∀ t

## Model (mirrors `prontuarios/gen_replay_tol_e5.py`, TMDD-lite)

    occ(t) = load(t) / (load(t) + EC50_μ)          EC50_μ = 1 ng/mL fentanyl-eq
    A(t+1) = A(t) + k · (occ(t) − A(t))            k = dt/τ_adapt, τ_adapt = 72 h
    W(t)   = max(0, A(t) − occ(t))                 withdrawal pressure

`A` is the μ-opioid adaptation state (Kumar 2008); `W` is the modeled
abstinence pressure. The runtime integrates with Euler dt = 0.05 h; the
Lean model is the exact per-step recurrence with `k` universally
quantified — the theorems are timestep- and parameter-independent.

## Main results (all machine-checked below)

1. `gap_nonneg_of_bounded_drop` — **F5c, sufficient condition**.
   If at every step the occupancy drop is bounded by the contracted gap,
       occ(t) − occ(t+1) ≤ (1 − k) · (occ(t) − A(t)),
   and initially A(0) ≤ occ(0), then A(t) ≤ occ(t) for **all** t.
   Note: no sign hypothesis on `k` is needed — the per-step certificate
   alone is sufficient.
2. `f5c_withdrawal_zero` — corollary: W(t) = 0 for all t.
3. `gap_neg_of_large_drop` — **F5c, tightness (impossibility direction)**.
   If at step t the drop exceeds the contracted gap,
       (1 − k) · (occ(t) − A(t)) < occ(t) − occ(t+1),
   then A(t+1) > occ(t+1): withdrawal pressure appears at the next step.
   Together with (1) this shows the per-step certificate is the *exact*
   boundary between zero and positive withdrawal pressure — the
   mechanistic explanation of why abrupt interruption causes abstinence
   and why the E5 schedule (bounded hourly drop + methadone PCC) does not.
4. `wpress_pos_of_inverted` — positive pressure whenever the gap inverts.
5. `pbox_floor_sound` / `pbox_ceiling_sound` — Knightian gate transfer:
   a p-box whose mean-band floor clears the threshold certifies every
   contained point (links the E5-GUM coverage p-box, lo ≥ 1.37 > 1, to
   per-realization coverage ≥ 1 in every Knightian corner).

## Trust boundary

Everything here is pure `Rat` algebra from Lean 4 core: expected axioms
are `[propext, Classical.choice, Quot.sound]` only (see the `#print
axioms` audit at the end). The bridge to the f64 runtime is *not*
claimed here: the Sounio module
`stdlib/darwin_pbpk/proof_carrying_weaning.sio` re-verifies the per-step
certificate numerically at every step of every cohort corner, and this
file proves that wherever the certificate holds, W = 0 follows — for
any τ_adapt > 0 and any EC50_μ > 0 (both enter only through the
universally quantified `k` and `occ`).

Provenance: síntese-mestra §6c (F5c GUM argument), E5 arm 61/61 W_max = 0,
E5-GUM coverage p-box ≥ 1.37 in all corners; cohort data local-only
(CEP/CAAE pending).
-/

namespace Sounio.Weaning

-- ================================================================
-- §1. Dynamics (discrete per-step recurrence)
-- ================================================================

/-- Adaptation state: A(t+1) = A(t) + k·(occ(t) − A(t)).
    `occ` is total opioid occupancy (fentanyl + methadone-equivalent),
    `k = dt/τ_adapt`. -/
def aseq (occ : Nat → Rat) (k a0 : Rat) : Nat → Rat
  | 0 => a0
  | t + 1 => aseq occ k a0 t + k * (occ t - aseq occ k a0 t)

/-- Withdrawal pressure: W(t) = max(0, A(t) − occ(t)). -/
def wpress (occ : Nat → Rat) (k a0 : Rat) (t : Nat) : Rat :=
  if aseq occ k a0 t ≤ occ t then 0 else aseq occ k a0 t - occ t

-- ================================================================
-- §2. Rational algebra helpers (Lean 4 core lemmas only)
-- ================================================================

/-- k·(x − y) = k·x − k·y (derived: core exposes `mul_add`, `mul_neg`). -/
theorem mul_sub_rat (k x y : Rat) : k * (x - y) = k * x - k * y := by
  calc k * (x - y) = k * (x + -y) := by rw [Rat.sub_eq_add_neg]
    _ = k * x + k * -y := by rw [Rat.mul_add]
    _ = k * x + -(k * y) := by rw [Rat.mul_neg]
    _ = k * x - k * y := by rw [← Rat.sub_eq_add_neg]

/-- (1 − k)·x = x − k·x. -/
theorem one_sub_mul_rat (k x : Rat) : (1 - k) * x = x - k * x := by
  calc (1 - k) * x = (1 + -k) * x := by rw [Rat.sub_eq_add_neg]
    _ = 1 * x + -k * x := by rw [Rat.add_mul]
    _ = x + -(k * x) := by rw [Rat.one_mul, Rat.neg_mul]
    _ = x - k * x := by rw [← Rat.sub_eq_add_neg]

/-- Closed form of the adaptation step:
    a + k·(o₀ − a) = (1 − k)·a + k·o₀  (convex combination). -/
theorem step_closed (a k o0 : Rat) :
    a + k * (o0 - a) = (1 - k) * a + k * o0 := by
  calc a + k * (o0 - a) = a + (k * o0 - k * a) := by rw [mul_sub_rat]
    _ = a + (k * o0 + -(k * a)) := by rw [Rat.sub_eq_add_neg]
    _ = a + (-(k * a) + k * o0) := by rw [Rat.add_comm (k * o0) (-(k * a))]
    _ = a + -(k * a) + k * o0 := by rw [← Rat.add_assoc]
    _ = (a - k * a) + k * o0 := by rw [← Rat.sub_eq_add_neg]
    _ = (1 - k) * a + k * o0 := by rw [one_sub_mul_rat]

/-- Contracted-gap identity:
    o₀ − (1 − k)·(o₀ − a) = (1 − k)·a + k·o₀.
    Proved by cancelling (1 − k)·(o₀ − a) on both sides of `key`. -/
theorem contracted_gap_id (a k o0 : Rat) :
    o0 - (1 - k) * (o0 - a) = (1 - k) * a + k * o0 := by
  have hsum : k + (1 - k) = 1 := by
    calc k + (1 - k) = k + (1 + -k) := by rw [Rat.sub_eq_add_neg]
      _ = k + 1 + -k := by rw [← Rat.add_assoc]
      _ = 1 + k + -k := by rw [Rat.add_comm k 1]
      _ = 1 + (k + -k) := by rw [Rat.add_assoc]
      _ = 1 + 0 := by rw [Rat.add_neg_cancel]
      _ = 1 := by rw [Rat.add_zero]
  have key : (1 - k) * a + k * o0 + (1 - k) * (o0 - a) = o0 := by
    calc (1 - k) * a + k * o0 + (1 - k) * (o0 - a)
        = (1 - k) * a + k * o0 + ((1 - k) * o0 - (1 - k) * a) := by rw [mul_sub_rat]
      _ = (1 - k) * a + (k * o0 + ((1 - k) * o0 - (1 - k) * a)) := by rw [Rat.add_assoc]
      _ = (1 - k) * a + (k * o0 + ((1 - k) * o0 + -((1 - k) * a))) := by
          rw [Rat.sub_eq_add_neg ((1 - k) * o0) ((1 - k) * a)]
      _ = (1 - k) * a + (-((1 - k) * a) + (k * o0 + (1 - k) * o0)) := by
          rw [Rat.add_comm ((1 - k) * o0) (-((1 - k) * a)),
              ← Rat.add_assoc (k * o0) (-((1 - k) * a)) ((1 - k) * o0),
              Rat.add_comm (k * o0) (-((1 - k) * a)),
              Rat.add_assoc (-((1 - k) * a)) (k * o0) ((1 - k) * o0)]
      _ = (1 - k) * a + -((1 - k) * a) + (k * o0 + (1 - k) * o0) := by
          rw [← Rat.add_assoc ((1 - k) * a) (-((1 - k) * a)) (k * o0 + (1 - k) * o0)]
      _ = 0 + (k * o0 + (1 - k) * o0) := by rw [Rat.add_neg_cancel]
      _ = k * o0 + (1 - k) * o0 := by rw [Rat.zero_add]
      _ = (k + (1 - k)) * o0 := by rw [← Rat.add_mul]
      _ = 1 * o0 := by rw [hsum]
      _ = o0 := by rw [Rat.one_mul]
  have h1 : o0 - (1 - k) * (o0 - a) + (1 - k) * (o0 - a)
          = (1 - k) * a + k * o0 + (1 - k) * (o0 - a) := by
    rw [Rat.sub_add_cancel, key]
  exact Rat.add_right_cancel _ h1

-- ================================================================
-- §3. F5c — the weaning safety theorem
-- ================================================================

/-- **F5c (sufficient condition).** If the initial adaptation is covered
    (A(0) ≤ occ(0)) and every occupancy drop is bounded by the contracted
    gap — occ(t) − occ(t+1) ≤ (1 − k)·(occ(t) − A(t)) — then the gap never
    closes: A(t) ≤ occ(t) for all t.

    Clinical reading: with τ_adapt = 72 h, k = dt/72 is small, so the
    admissible drop per step is ≈ the full remaining gap; the E5 taper
    (−20 %/day ≈ −0.9 %/h on the fentanyl leg) plus methadone PCC keeps
    the total-occupancy drop far inside this bound in every corner of
    the cohort (verified numerically, 9/9 corners; síntese §6c). -/
theorem gap_nonneg_of_bounded_drop
    (occ : Nat → Rat) (k a0 : Rat)
    (h0 : a0 ≤ occ 0)
    (hdrop : ∀ t, occ t - occ (t + 1) ≤ (1 - k) * (occ t - aseq occ k a0 t)) :
    ∀ t, aseq occ k a0 t ≤ occ t := by
  intro t
  induction t with
  | zero => exact h0
  | succ t _ih =>
      show aseq occ k a0 t + k * (occ t - aseq occ k a0 t) ≤ occ (t + 1)
      rw [step_closed, ← contracted_gap_id]
      -- goal: occ t - (1 - k) * (occ t - aseq occ k a0 t) ≤ occ (t + 1)
      have h1 : occ t ≤ (1 - k) * (occ t - aseq occ k a0 t) + occ (t + 1) :=
        Rat.le_add_iff_sub_le.mpr (hdrop t)
      apply Rat.le_add_iff_sub_le.mp
      rw [Rat.add_comm]
      exact h1

/-- **F5c corollary — zero withdrawal pressure.** Under the bounded-drop
    certificate, W(t) = 0 at every step of the weaning window. -/
theorem f5c_withdrawal_zero
    (occ : Nat → Rat) (k a0 : Rat)
    (h0 : a0 ≤ occ 0)
    (hdrop : ∀ t, occ t - occ (t + 1) ≤ (1 - k) * (occ t - aseq occ k a0 t)) :
    ∀ t, wpress occ k a0 t = 0 := by
  intro t
  unfold wpress
  exact if_pos (gap_nonneg_of_bounded_drop occ k a0 h0 hdrop t)

/-- **F5c (tightness).** If at some step the occupancy drop exceeds the
    contracted gap, withdrawal pressure appears at the *next* step:
    A(t+1) > occ(t+1). The certificate of `gap_nonneg_of_bounded_drop`
    is therefore the exact boundary — abrupt interruption (drop ≫ gap)
    structurally produces W > 0, independent of PK parameters. -/
theorem gap_neg_of_large_drop
    (occ : Nat → Rat) (k a0 : Rat) (t : Nat)
    (h : (1 - k) * (occ t - aseq occ k a0 t) < occ t - occ (t + 1)) :
    occ (t + 1) < aseq occ k a0 (t + 1) := by
  show occ (t + 1) < aseq occ k a0 t + k * (occ t - aseq occ k a0 t)
  rw [step_closed, ← contracted_gap_id]
  -- goal: occ (t + 1) < occ t - (1 - k) * (occ t - aseq occ k a0 t)
  have h1 : (1 - k) * (occ t - aseq occ k a0 t) + occ (t + 1) < occ t :=
    Rat.lt_sub_right_iff_add_lt.mp h
  apply Rat.lt_sub_right_iff_add_lt.mpr
  rw [Rat.add_comm]
  exact h1

/-- Positive withdrawal pressure whenever the gap inverts. -/
theorem wpress_pos_of_inverted
    (occ : Nat → Rat) (k a0 : Rat) (t : Nat)
    (h : occ t < aseq occ k a0 t) :
    0 < wpress occ k a0 t := by
  unfold wpress
  rw [if_neg (Rat.not_le.mpr h)]
  exact (Rat.lt_iff_sub_pos _ _).mp h

-- ================================================================
-- §4. Knightian gate transfer (p-box floor ⇒ contained values)
-- ================================================================

/-- If the mean-band floor of a p-box clears the threshold, every value
    contained in the band clears it too. This is the transfer used by
    the E5-GUM claim: coverage p-box lo ≥ 1.37 > 1 (all Knightian
    corners) ⇒ per-realization coverage ≥ 1 ⇒ (by F5c) W = 0.

    ℚ counterpart of `Sounio.PBoxSemantics.containsR`; restated here
    unbundled so this file stays import-free (fast, toolchain-light). -/
theorem pbox_floor_sound (lo x thresh : Rat)
    (hfloor : thresh ≤ lo) (hx_lo : lo ≤ x) :
    thresh ≤ x :=
  Rat.le_trans hfloor hx_lo

/-- Upper-edge variant: hi below threshold ⇒ every contained value below. -/
theorem pbox_ceiling_sound (hi x thresh : Rat)
    (hceil : hi ≤ thresh) (hx_hi : x ≤ hi) :
    x ≤ thresh :=
  Rat.le_trans hx_hi hceil

-- ================================================================
-- §5. Axiom audit (build-log evidence)
-- ================================================================

-- Expected: [propext, Classical.choice, Quot.sound] only.
#print axioms gap_nonneg_of_bounded_drop
#print axioms f5c_withdrawal_zero
#print axioms gap_neg_of_large_drop
#print axioms wpress_pos_of_inverted
#print axioms pbox_floor_sound
#print axioms pbox_ceiling_sound

end Sounio.Weaning
