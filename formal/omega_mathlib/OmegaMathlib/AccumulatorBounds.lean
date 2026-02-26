import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

namespace OmegaMathlib

/-- Hardware accumulator floor expression in Q32.32 log units. -/
def L_hw (fidelity provenance quantum : ℕ) : ℚ :=
  ((fidelity / 8 : ℕ) : ℚ) +
    ((provenance / 16 : ℕ) : ℚ) +
    ((quantum / 32 : ℕ) : ℚ)

/-- Reference (exact rational) accumulator expression. -/
def L_ref (fidelity provenance quantum : ℕ) : ℚ :=
  (fidelity : ℚ) / 8 + (provenance : ℚ) / 16 + (quantum : ℚ) / 32

/-- 1/32-scaled residual gap used by the hardware gate. -/
def delta32 (fidelity provenance quantum : ℕ) : ℕ :=
  4 * (fidelity % 8) + 2 * (provenance % 16) + (quantum % 32)

/-- Reference minus hardware gap. -/
def gap (fidelity provenance quantum : ℕ) : ℚ :=
  L_ref fidelity provenance quantum - L_hw fidelity provenance quantum

lemma lane_gap_eq_mod (n d : ℕ) (hd : 0 < d) :
    (n : ℚ) / (d : ℚ) - ((n / d : ℕ) : ℚ) = ((n % d : ℕ) : ℚ) / (d : ℚ) := by
  have hdq : (d : ℚ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hd)
  have hnat : (n / d) * d + n % d = n := by
    simpa [Nat.mul_comm, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
      (Nat.mod_add_div n d)
  have hrat : ((n / d : ℕ) : ℚ) * (d : ℚ) + ((n % d : ℕ) : ℚ) = (n : ℚ) := by
    exact_mod_cast hnat
  have hsplit :
      (n : ℚ) / (d : ℚ) =
        ((n / d : ℕ) : ℚ) + ((n % d : ℕ) : ℚ) / (d : ℚ) := by
    apply (div_eq_iff hdq).2
    calc
      (n : ℚ) = ((n / d : ℕ) : ℚ) * (d : ℚ) + ((n % d : ℕ) : ℚ) := by
        simp [hrat]
      _ = ((n / d : ℕ) : ℚ) * (d : ℚ) + (((n % d : ℕ) : ℚ) / (d : ℚ)) * (d : ℚ) := by
        rw [div_mul_cancel₀ _ hdq]
      _ = (((n / d : ℕ) : ℚ) + ((n % d : ℕ) : ℚ) / (d : ℚ)) * (d : ℚ) := by
        ring
  calc
    (n : ℚ) / (d : ℚ) - ((n / d : ℕ) : ℚ) =
        (((n / d : ℕ) : ℚ) + ((n % d : ℕ) : ℚ) / (d : ℚ)) - ((n / d : ℕ) : ℚ) := by
      simp [hsplit]
    _ = ((n % d : ℕ) : ℚ) / (d : ℚ) := by ring

theorem gap_eq_delta32_div32 (fidelity provenance quantum : ℕ) :
    gap fidelity provenance quantum = ((delta32 fidelity provenance quantum : ℕ) : ℚ) / (32 : ℚ) := by
  have hF := lane_gap_eq_mod fidelity 8 (by decide)
  have hP := lane_gap_eq_mod provenance 16 (by decide)
  have hQ := lane_gap_eq_mod quantum 32 (by decide)
  have hsum :
      gap fidelity provenance quantum =
        ((fidelity % 8 : ℕ) : ℚ) / (8 : ℚ) +
          ((provenance % 16 : ℕ) : ℚ) / (16 : ℚ) +
          ((quantum % 32 : ℕ) : ℚ) / (32 : ℚ) := by
    unfold gap L_ref L_hw
    linarith [hF, hP, hQ]
  calc
    gap fidelity provenance quantum =
      ((fidelity % 8 : ℕ) : ℚ) / (8 : ℚ) +
        ((provenance % 16 : ℕ) : ℚ) / (16 : ℚ) +
        ((quantum % 32 : ℕ) : ℚ) / (32 : ℚ) := hsum
    _ = ((4 : ℚ) * ((fidelity % 8 : ℕ) : ℚ) +
          (2 : ℚ) * ((provenance % 16 : ℕ) : ℚ) +
          ((quantum % 32 : ℕ) : ℚ)) / (32 : ℚ) := by
      ring
    _ = (((4 * (fidelity % 8) + 2 * (provenance % 16) + (quantum % 32) : ℕ) : ℚ)) / (32 : ℚ) := by
      have hcast :
          ((4 * (fidelity % 8) + 2 * (provenance % 16) + (quantum % 32) : ℕ) : ℚ) =
            (4 : ℚ) * ((fidelity % 8 : ℕ) : ℚ) +
              (2 : ℚ) * ((provenance % 16 : ℕ) : ℚ) +
              ((quantum % 32 : ℕ) : ℚ) := by
        norm_num [Nat.cast_add, Nat.cast_mul]
      rw [← hcast]
    _ = ((delta32 fidelity provenance quantum : ℕ) : ℚ) / (32 : ℚ) := by
      unfold delta32
      rfl

theorem delta32_lt_96 (fidelity provenance quantum : ℕ) :
    delta32 fidelity provenance quantum < 96 := by
  have hF : fidelity % 8 ≤ 7 := Nat.le_pred_of_lt (Nat.mod_lt _ (by decide))
  have hP : provenance % 16 ≤ 15 := Nat.le_pred_of_lt (Nat.mod_lt _ (by decide))
  have hQ : quantum % 32 ≤ 31 := Nat.le_pred_of_lt (Nat.mod_lt _ (by decide))
  have h1 : 4 * (fidelity % 8) ≤ 4 * 7 := Nat.mul_le_mul_left 4 hF
  have h2 : 2 * (provenance % 16) ≤ 2 * 15 := Nat.mul_le_mul_left 2 hP
  have h3 : quantum % 32 ≤ 31 := hQ
  have hsum :
      4 * (fidelity % 8) + 2 * (provenance % 16) + quantum % 32 ≤
        4 * 7 + 2 * 15 + 31 := by
    exact Nat.add_le_add (Nat.add_le_add h1 h2) h3
  unfold delta32
  exact lt_of_le_of_lt hsum (by decide)

theorem accumulator_bound (fidelity provenance quantum : ℕ) :
    0 ≤ gap fidelity provenance quantum ∧ gap fidelity provenance quantum < 3 := by
  rw [gap_eq_delta32_div32]
  constructor
  · have hδ0 : (0 : ℚ) ≤ (delta32 fidelity provenance quantum : ℚ) := by
      exact_mod_cast (Nat.zero_le (delta32 fidelity provenance quantum))
    have h32 : (0 : ℚ) ≤ (32 : ℚ) := by norm_num
    exact div_nonneg hδ0 h32
  · have hδ : (delta32 fidelity provenance quantum : ℚ) < 96 := by
      exact_mod_cast (delta32_lt_96 fidelity provenance quantum)
    have hdiv : (delta32 fidelity provenance quantum : ℚ) / 32 < (96 : ℚ) / 32 := by
      exact div_lt_div_of_pos_right hδ (by norm_num)
    norm_num at hdiv
    exact hdiv

theorem bound_is_tight :
    gap 7 15 31 = (89 : ℚ) / 32 := by
  rw [gap_eq_delta32_div32]
  norm_num [delta32]

end OmegaMathlib
