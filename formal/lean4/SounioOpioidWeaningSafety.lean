-- formal/lean4/SounioOpioidWeaningSafety.lean
/-!
# Sounio.Weaning — Opioid Weaning Safety (F5c) — Lean 4 Discharge

Formal core of the **P3 proof-carrying weaning protocol** (UTIped cohort,
N = 61, Campinas-SP). This file discharges, in **pure ℚ** (Lean 4 core
`Rat`, Mathlib-free, no `sorry`, no `native_decide`, no IEEE-754 axioms),
the structural theorem behind the E5/E6 counterfactual arms.

**Claim scope (do not overclaim):** W(t) = 0 holds **only while** the
per-step bounded-drop certificate holds. Empirically the E5 schedule
(taper −20 %/day + methadone PCC from day 1, then methadone −10 %/day
geometric after a 5-day hold) certifies the **conversion window**
(~120 h after fentanyl stop) but **not** the methadone geometric tail:
cohort replay shows certificate violations and W_global > 0 in 61/61
patients once the constant-percentage methadone taper begins. The
geometric-tail corollaries below prove that constant-% taper is
*structurally* unsafe once the adaptation gap has closed (fully adapted
state after a hold) — motivating the E6 certificate-clipped taper.

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
2. `f5c_withdrawal_zero` — corollary: W(t) = 0 for all t **under the
   certificate** (not an unconditional claim about any particular
   clinical schedule).
3. `gap_neg_of_large_drop` — **F5c, tightness (impossibility direction)**.
   If at step t the drop exceeds the contracted gap,
       (1 − k) · (occ(t) − A(t)) < occ(t) − occ(t+1),
   then A(t+1) > occ(t+1): withdrawal pressure appears at the next step.
   Together with (1) this shows the per-step certificate is the *exact*
   boundary between zero and positive withdrawal pressure.
4. `wpress_pos_of_inverted` — positive pressure whenever the gap inverts.
5. `pbox_floor_sound` / `pbox_ceiling_sound` — Knightian gate transfer:
   a p-box whose mean-band floor clears the threshold certifies every
   contained point (links the E5-GUM coverage p-box, lo ≥ 1.37 > 1, to
   per-realization coverage ≥ 1 in every Knightian corner).
6. **Geometric-tail corollaries (E6 motivation):**
   - `geometric_drop_eq`: under `occ(t+1) = r · occ(t)`, the drop equals
     `(1 − r) · occ(t)`.
   - `geometric_violates_small_gap`: if the contracted gap is strictly
     smaller than that geometric drop, tightness forces W at the next
     step.
   - `fully_adapted_geometric_withdraws`: after a hold that closes the
     gap (`A(0) = occ(0)`), any geometric factor `r < 1` with `occ > 0`
     immediately violates the certificate and produces W(1) > 0 —
     constant-% methadone taper is structurally unsafe at the start of
     the tail.

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

Provenance: síntese-mestra §6c/§12 (F5c + P3), E5 arm W≈0 in the 120 h
window but W_global > 0 in 61/61 on the geometric methadone tail;
E5-GUM coverage p-box ≥ 1.37 in conversion corners; cohort data
local-only (CEP/CAAE pending).
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
-- §5. Geometric-tail corollaries (constant-% taper is unsafe at gap=0)
-- ================================================================

/-- Geometric occupancy schedule: `occ(t+1) = r · occ(t)`.
    Models a constant-percentage dose reduction (E5 methadone −10 %/day
    after the hold), discretised at the adaptation timestep. -/
def geo_occ (r o0 : Rat) : Nat → Rat
  | 0 => o0
  | t + 1 => r * geo_occ r o0 t

/-- Under a geometric schedule the occupancy drop factors cleanly:
    `occ − r·occ = (1 − r)·occ`. -/
theorem geometric_drop_eq (r o : Rat) :
    o - r * o = (1 - r) * o := by
  calc o - r * o = o + -(r * o) := by rw [Rat.sub_eq_add_neg]
    _ = 1 * o + -(r * o) := by rw [Rat.one_mul]
    _ = 1 * o + (-r) * o := by rw [← Rat.neg_mul]
    _ = (1 + -r) * o := by rw [← Rat.add_mul]
    _ = (1 - r) * o := by rw [← Rat.sub_eq_add_neg]

/-- If the contracted gap is strictly smaller than the geometric drop,
    tightness forces withdrawal at the next step.
    Hypothesis form avoids division: `(1−k)·gap < (1−r)·occ`. -/
theorem geometric_violates_small_gap
    (r k o0 a0 : Rat) (t : Nat)
    (h : (1 - k) * (geo_occ r o0 t - aseq (geo_occ r o0) k a0 t)
          < (1 - r) * geo_occ r o0 t) :
    geo_occ r o0 (t + 1) < aseq (geo_occ r o0) k a0 (t + 1) := by
  have hdrop :
      (1 - k) * (geo_occ r o0 t - aseq (geo_occ r o0) k a0 t)
        < geo_occ r o0 t - geo_occ r o0 (t + 1) := by
    -- geo_occ (t+1) = r * geo_occ t, so RHS = (1-r)*geo_occ t
    show (1 - k) * (geo_occ r o0 t - aseq (geo_occ r o0) k a0 t)
          < geo_occ r o0 t - r * geo_occ r o0 t
    rw [geometric_drop_eq]
    exact h
  exact gap_neg_of_large_drop (geo_occ r o0) k a0 t hdrop

/-- **Fully-adapted geometric taper withdraws immediately.**
    After a hold that closes the adaptation gap (`A(0) = occ(0)`), any
    geometric factor `r < 1` with positive occupancy produces a positive
    drop against a zero contracted gap, so the certificate fails at step 0
    and `W(1) > 0`. This is the formal reason the E5 methadone −10 %/day
    tail (constant %) is unsafe once the patient is adapted — motivating
    the E6 certificate-clipped schedule. -/
theorem fully_adapted_geometric_withdraws
    (r k o0 : Rat)
    (hr : r < 1) (ho : 0 < o0) :
    geo_occ r o0 1 < aseq (geo_occ r o0) k o0 1 := by
  have hgap0 : geo_occ r o0 0 - aseq (geo_occ r o0) k o0 0 = 0 := by
    simp [geo_occ, aseq, Rat.sub_self]
  have hbound0 : (1 - k) * (geo_occ r o0 0 - aseq (geo_occ r o0) k o0 0) = 0 := by
    rw [hgap0, Rat.mul_zero]
  have hpos : 0 < (1 - r) * o0 := by
    have hr' : 0 < 1 - r := (Rat.lt_iff_sub_pos r 1).mp hr
    exact Rat.mul_pos hr' ho
  have h : (1 - k) * (geo_occ r o0 0 - aseq (geo_occ r o0) k o0 0)
            < (1 - r) * geo_occ r o0 0 := by
    rw [hbound0]
    simp [geo_occ]
    exact hpos
  exact geometric_violates_small_gap r k o0 o0 0 h

/-- Positive withdrawal pressure one step after a fully-adapted geometric
    cut — packages the previous theorem with `wpress_pos_of_inverted`. -/
theorem fully_adapted_geometric_wpress_pos
    (r k o0 : Rat)
    (hr : r < 1) (ho : 0 < o0) :
    0 < wpress (geo_occ r o0) k o0 1 :=
  wpress_pos_of_inverted (geo_occ r o0) k o0 1
    (fully_adapted_geometric_withdraws r k o0 hr ho)

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
#print axioms geometric_drop_eq
#print axioms geometric_violates_small_gap
#print axioms fully_adapted_geometric_withdraws
#print axioms fully_adapted_geometric_wpress_pos

end Sounio.Weaning
