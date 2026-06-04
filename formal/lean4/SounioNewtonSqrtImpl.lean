import SounioSqrtFieldReal

set_option maxHeartbeats 0

namespace SounioSqrt.RealCauchyField

open Sounio.RealCauchy (SounioRealCauchy)

/-! ## Generic `Rat` algebra helpers (Mathlib-free)

All proofs use only the verified core `Rat` order/ring API plus the public toolkit
re-exported from `SounioSqrtFieldReal` (`rat_sub_split`, `rat_add_le_add`,
`rat_add_nonneg`). No `ring`/`linarith`/`field_simp`. -/

/-- Distribute multiplication over a subtraction on the right factor. -/
theorem rmul_sub (a b c : Rat) : a * (b - c) = a * b - a * c := by
  rw [Rat.sub_eq_add_neg, Rat.mul_add, Rat.mul_neg, ← Rat.sub_eq_add_neg]

/-- Distribute multiplication over a subtraction on the left factor. -/
theorem rsub_mul (a b c : Rat) : (a - b) * c = a * c - b * c := by
  rw [Rat.sub_eq_add_neg, Rat.add_mul, Rat.neg_mul, ← Rat.sub_eq_add_neg]

theorem eq_add_of_sub_eq {a b d : Rat} (h : a - b = d) : a = b + d := by
  rw [← h, Rat.add_comm, Rat.sub_add_cancel]

theorem sub_eq_of_add_eq {a b c : Rat} (h : a + b = c) : c - b = a := by
  rw [← h, Rat.add_sub_cancel]

/-- Difference of squares. -/
theorem diff_of_squares (a b : Rat) : a * a - b * b = (a + b) * (a - b) := by
  rw [Rat.add_mul, rmul_sub, rmul_sub, Rat.mul_comm b a, ← rat_sub_split]

/-- `(A·B)² = A²·B²` (mul commutes/associates). -/
theorem sq_prod (A B : Rat) : (A * B) * (A * B) = (A * A) * (B * B) := by
  rw [Rat.mul_assoc, ← Rat.mul_assoc B A B, Rat.mul_comm B A, Rat.mul_assoc A B B, ← Rat.mul_assoc]

theorem add_simp1 (y c : Rat) : (y + c) + (y - c) = y + y := by
  rw [Rat.sub_eq_add_neg]
  calc (y + c) + (y + -c)
      = y + (c + (y + -c)) := by rw [Rat.add_assoc]
    _ = y + ((c + y) + -c) := by rw [← Rat.add_assoc c y (-c)]
    _ = y + ((y + c) + -c) := by rw [Rat.add_comm c y]
    _ = y + (y + (c + -c)) := by rw [Rat.add_assoc y c (-c)]
    _ = y + (y + 0) := by rw [Rat.add_neg_cancel]
    _ = y + y := by rw [Rat.add_zero]

theorem sub_simp1 (y c : Rat) : (y + c) - (y - c) = c + c := by
  rw [Rat.sub_eq_add_neg (y + c) (y - c), Rat.neg_sub, Rat.sub_eq_add_neg c y]
  calc (y + c) + (c + -y)
      = y + (c + (c + -y)) := by rw [Rat.add_assoc]
    _ = y + ((c + c) + -y) := by rw [← Rat.add_assoc c c (-y)]
    _ = y + (-y + (c + c)) := by rw [Rat.add_comm (c + c) (-y)]
    _ = (y + -y) + (c + c) := by rw [← Rat.add_assoc y (-y) (c + c)]
    _ = 0 + (c + c) := by rw [Rat.add_neg_cancel]
    _ = c + c := by rw [Rat.zero_add]

/-- `(y+c)² = (y-c)² + (y+y)(c+c)`  (i.e. the `+4yc` cross-term). -/
theorem expand (y c : Rat) : (y + c) * (y + c) = (y - c) * (y - c) + (y + y) * (c + c) := by
  have hd := eq_add_of_sub_eq (diff_of_squares (y + c) (y - c))
  rw [hd, add_simp1, sub_simp1]

theorem prod_yy_cc (y c : Rat) : (y + y) * (c + c) = (y * c + y * c) + (y * c + y * c) := by
  rw [Rat.add_mul, Rat.mul_add]

theorem quarter_four (p : Rat) : ((p + p) + (p + p)) * (1 / 4) = p := by
  have hs : p * (1 / 4) + p * (1 / 4) = p * (1 / 2) := by
    rw [← Rat.mul_add, show (1 / 4 + 1 / 4 : Rat) = 1 / 2 from by native_decide]
  calc ((p + p) + (p + p)) * (1 / 4)
      = (p + p) * (1 / 4) + (p + p) * (1 / 4) := by rw [Rat.add_mul]
    _ = (p * (1 / 4) + p * (1 / 4)) + (p * (1 / 4) + p * (1 / 4)) := by rw [Rat.add_mul]
    _ = p * (1 / 2) + p * (1 / 2) := by rw [hs]
    _ = p * ((1 / 2) + (1 / 2)) := by rw [← Rat.mul_add]
    _ = p * 1 := by rw [show ((1 / 2 : Rat) + (1 / 2)) = 1 from by native_decide]
    _ = p := by rw [Rat.mul_one]

theorem valpart (p y : Rat) (hyc : y * (p * y⁻¹) = p) :
    ((y + y) * ((p * y⁻¹) + (p * y⁻¹))) * (1 / 4) = p := by
  rw [prod_yy_cc, hyc, quarter_four]

theorem add_swap4 (a b c d : Rat) : (a + b) + (c + d) = (a + c) + (b + d) := by
  rw [Rat.add_assoc, ← Rat.add_assoc b c d, Rat.add_comm b c, Rat.add_assoc c b d,
    ← Rat.add_assoc a c (b + d)]

/-! ## Order helpers over `Rat`. -/

theorem le_of_sub_nonneg {a b : Rat} (h : 0 ≤ a - b) : b ≤ a := by
  have h2 := (Rat.add_le_add_right (a := 0) (b := a - b) (c := b)).mpr h
  rwa [Rat.zero_add, Rat.sub_add_cancel] at h2

theorem sub_nonneg_of_le {a b : Rat} (h : b ≤ a) : 0 ≤ a - b := by
  have h2 := (Rat.add_le_add_right (a := b) (b := a) (c := -b)).mpr h
  rwa [Rat.add_neg_cancel, ← Rat.sub_eq_add_neg] at h2

theorem sub_le_self (a : Rat) {c : Rat} (hc : 0 ≤ c) : a - c ≤ a := by
  rw [Rat.sub_eq_add_neg]
  have hnp : -c ≤ 0 := by
    have h := Rat.neg_le_neg hc
    rwa [Rat.neg_zero] at h
  have h2 := (Rat.add_le_add_left (a := -c) (b := 0) (c := a)).mpr hnp
  rwa [Rat.add_zero] at h2

theorem sub_le_sub_of_le {a b c : Rat} (h : b ≤ c) : a - c ≤ a - b := by
  rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg]
  exact (Rat.add_le_add_left).mpr (Rat.neg_le_neg h)

theorem ge_one_of_sq {x : Rat} (h0 : 0 < x) (hsq : 1 ≤ x * x) : 1 ≤ x := by
  refine Classical.byContradiction (fun hc => ?_)
  have hx1 : x < 1 := Rat.not_le.mp hc
  have hxx : x * x ≤ x * 1 := Rat.mul_le_mul_of_nonneg_left (Rat.le_of_lt hx1) (Rat.le_of_lt h0)
  rw [Rat.mul_one] at hxx
  have hlt : x * x < 1 := Std.lt_of_le_of_lt hxx hx1
  exact absurd hsq (Rat.not_le.mpr hlt)

theorem inv_le_one {x : Rat} (hx : 1 ≤ x) : x⁻¹ ≤ 1 := by
  have h01 : (0 : Rat) < 1 := by native_decide
  have hx0 : 0 < x := Std.lt_of_lt_of_le h01 hx
  have hxne : x ≠ 0 := fun h => by rw [h] at hx0; exact Rat.lt_irrefl hx0
  have hq0 : 0 ≤ x⁻¹ := Rat.le_of_lt (Rat.inv_pos.mpr hx0)
  have hxx : x⁻¹ * x = 1 := by rw [Rat.mul_comm]; exact Rat.mul_inv_cancel x hxne
  have h := Rat.mul_le_mul_of_nonneg_left hx hq0
  rwa [Rat.mul_one, hxx] at h

theorem le_of_mul_le_mul_left {a b c : Rat} (hc : 0 < c) (h : c * a ≤ c * b) : a ≤ b := by
  have hcne : c ≠ 0 := fun hh => by rw [hh] at hc; exact Rat.lt_irrefl hc
  have hinv : c⁻¹ * c = 1 := by rw [Rat.mul_comm]; exact Rat.mul_inv_cancel c hcne
  have hq0 : 0 ≤ c⁻¹ := Rat.le_of_lt (Rat.inv_pos.mpr hc)
  have h2 := Rat.mul_le_mul_of_nonneg_left h hq0
  rwa [← Rat.mul_assoc, ← Rat.mul_assoc, hinv, Rat.one_mul, Rat.one_mul] at h2

theorem le_add_one (a : Rat) : a ≤ a + 1 := by
  have h := (Rat.add_le_add_left (a := 0) (b := 1) (c := a)).mpr (by native_decide : (0 : Rat) ≤ 1)
  rwa [Rat.add_zero] at h

theorem rat_add_pos {a b : Rat} (ha : 0 < a) (hb : 0 < b) : 0 < a + b := by
  have h : a + 0 ≤ a + b := (Rat.add_le_add_left (a := 0) (b := b) (c := a)).mpr (Rat.le_of_lt hb)
  rw [Rat.add_zero] at h
  exact Std.lt_of_lt_of_le ha h

/-- Archimedean property of `ℚ`: every rational is bounded above by a natural. -/
theorem exists_nat_ge (r : Rat) : ∃ n : Nat, r ≤ (n : Rat) := by
  refine ⟨r.ceil.toNat, ?_⟩
  have h1 : r ≤ ((r.ceil : Int) : Rat) := Rat.le_ceil
  have h2 : (r.ceil : Int) ≤ ((r.ceil.toNat : Int)) := Int.self_le_toNat _
  have h3 : ((r.ceil : Int) : Rat) ≤ (((r.ceil.toNat : Int)) : Rat) := Rat.intCast_le_intCast.mpr h2
  have h4 : (((r.ceil.toNat : Int)) : Rat) = ((r.ceil.toNat : Nat) : Rat) :=
    Rat.intCast_natCast r.ceil.toNat
  rw [h4] at h3
  exact Rat.le_trans h1 h3

/-! ## The Newton square-root iteration for a rational `p`. -/

/-- Newton iteration for `√p` starting at `x₀ = p`:
    `x_{n+1} = (x_n + p / x_n) / 2`. -/
def newton (p : Rat) : Nat → Rat
  | 0 => p
  | n + 1 => (newton p n + p * (newton p n)⁻¹) * (1 / 2)

/-- The same sequence as the carrier of a real number. -/
def newtonReal (p : Rat) : Nat → Rat := newton p

/-- **Key error identity.** With `y ≠ 0` (encoded by `y·y⁻¹ = 1`):
    `x_{n+1}² − p = (x_n² − p)² · ( (x_n⁻¹)² / 4 )`. -/
theorem err_step (p y : Rat) (hyq : y * y⁻¹ = 1) :
    ((y + p * y⁻¹) * (1 / 2)) * ((y + p * y⁻¹) * (1 / 2)) - p
      = ((y * y - p) * (y * y - p)) * ((y⁻¹ * y⁻¹) * (1 / 4)) := by
  have hyc : y * (p * y⁻¹) = p := by
    rw [← Rat.mul_assoc, Rat.mul_comm y p, Rat.mul_assoc, hyq, Rat.mul_one]
  have hsub : (y * y - p) * y⁻¹ = y - p * y⁻¹ := by
    rw [rsub_mul, Rat.mul_assoc, hyq, Rat.mul_one]
  calc ((y + p * y⁻¹) * (1 / 2)) * ((y + p * y⁻¹) * (1 / 2)) - p
      = ((y + p * y⁻¹) * (y + p * y⁻¹)) * ((1 / 2) * (1 / 2)) - p := by rw [sq_prod]
    _ = ((y + p * y⁻¹) * (y + p * y⁻¹)) * (1 / 4) - p := by
          rw [show ((1 / 2 : Rat) * (1 / 2)) = 1 / 4 from by native_decide]
    _ = ((y - p * y⁻¹) * (y - p * y⁻¹) + (y + y) * ((p * y⁻¹) + (p * y⁻¹))) * (1 / 4) - p := by
          rw [expand]
    _ = ((y - p * y⁻¹) * (y - p * y⁻¹)) * (1 / 4)
          + ((y + y) * ((p * y⁻¹) + (p * y⁻¹))) * (1 / 4) - p := by rw [Rat.add_mul]
    _ = ((y - p * y⁻¹) * (y - p * y⁻¹)) * (1 / 4) + p - p := by rw [valpart p y hyc]
    _ = ((y - p * y⁻¹) * (y - p * y⁻¹)) * (1 / 4) := by rw [Rat.add_sub_cancel]
    _ = (((y * y - p) * y⁻¹) * ((y * y - p) * y⁻¹)) * (1 / 4) := by rw [hsub]
    _ = ((y * y - p) * (y * y - p)) * (y⁻¹ * y⁻¹) * (1 / 4) := by rw [sq_prod]
    _ = ((y * y - p) * (y * y - p)) * ((y⁻¹ * y⁻¹) * (1 / 4)) := by rw [Rat.mul_assoc]

/-- **Monotone-step identity.** `x_n − x_{n+1} = (x_n² − p)·x_n⁻¹ / 2`. -/
theorem mono_eq_gen (p y : Rat) (hyq : y * y⁻¹ = 1) :
    y - (y + p * y⁻¹) * (1 / 2) = ((y * y - p) * y⁻¹) * (1 / 2) := by
  have hsub : (y * y - p) * y⁻¹ = y - p * y⁻¹ := by
    rw [rsub_mul, Rat.mul_assoc, hyq, Rat.mul_one]
  have hyhalf : y * (1 / 2) + y * (1 / 2) = y := by
    rw [← Rat.mul_add, show ((1 / 2 : Rat) + (1 / 2)) = 1 from by native_decide, Rat.mul_one]
  have hA : (y + p * y⁻¹) * (1 / 2) = y * (1 / 2) + (p * y⁻¹) * (1 / 2) := by rw [Rat.add_mul]
  have hB : ((y * y - p) * y⁻¹) * (1 / 2) = y * (1 / 2) - (p * y⁻¹) * (1 / 2) := by
    rw [hsub, rsub_mul]
  have hBA : (y * (1 / 2) - (p * y⁻¹) * (1 / 2)) + (y * (1 / 2) + (p * y⁻¹) * (1 / 2))
      = y * (1 / 2) + y * (1 / 2) := by
    rw [Rat.sub_eq_add_neg, Rat.add_assoc,
      ← Rat.add_assoc (-((p * y⁻¹) * (1 / 2))) (y * (1 / 2)) ((p * y⁻¹) * (1 / 2)),
      Rat.add_comm (-((p * y⁻¹) * (1 / 2))) (y * (1 / 2)),
      Rat.add_assoc (y * (1 / 2)) (-((p * y⁻¹) * (1 / 2))) ((p * y⁻¹) * (1 / 2)),
      Rat.add_comm (-((p * y⁻¹) * (1 / 2))) ((p * y⁻¹) * (1 / 2)), Rat.add_neg_cancel, Rat.add_zero]
  have hsum : ((y * y - p) * y⁻¹) * (1 / 2) + (y + p * y⁻¹) * (1 / 2) = y := by
    rw [hB, hA, hBA, hyhalf]
  exact sub_eq_of_add_eq hsum

/-! ## Invariants: `1 ≤ x_n` and `p ≤ x_n²`. -/

/-- Combined invariant proven by induction. -/
theorem newton_inv (p : Rat) (hp1 : 1 ≤ p) :
    ∀ n, 1 ≤ newton p n ∧ p ≤ newton p n * newton p n := by
  have h01 : (0 : Rat) < 1 := by native_decide
  have hp0 : (0 : Rat) ≤ p := Rat.le_trans (Rat.le_of_lt h01) hp1
  have hppos : 0 < p := Std.lt_of_lt_of_le h01 hp1
  intro n
  induction n with
  | zero =>
      refine ⟨hp1, ?_⟩
      show p ≤ p * p
      have h := Rat.mul_le_mul_of_nonneg_left hp1 hp0
      rwa [Rat.mul_one] at h
  | succ n ih =>
      obtain ⟨hx1, hsq⟩ := ih
      have hx0 : 0 < newton p n := Std.lt_of_lt_of_le h01 hx1
      have hxne : newton p n ≠ 0 := fun h => by rw [h] at hx0; exact Rat.lt_irrefl hx0
      have hyq : newton p n * (newton p n)⁻¹ = 1 := Rat.mul_inv_cancel _ hxne
      have hqpos : 0 < (newton p n)⁻¹ := Rat.inv_pos.mpr hx0
      have hdef : newton p (n + 1) = (newton p n + p * (newton p n)⁻¹) * (1 / 2) := rfl
      have hpos : 0 < newton p (n + 1) := by
        rw [hdef]
        exact Rat.mul_pos (rat_add_pos hx0 (Rat.mul_pos hppos hqpos)) (by native_decide)
      have he0 : 0 ≤ newton p n * newton p n - p := sub_nonneg_of_le hsq
      have hsqsub : newton p (n + 1) * newton p (n + 1) - p
          = ((newton p n * newton p n - p) * (newton p n * newton p n - p))
              * (((newton p n)⁻¹ * (newton p n)⁻¹) * (1 / 4)) := by
        rw [hdef]; exact err_step p (newton p n) hyq
      have hrhs : 0 ≤ ((newton p n * newton p n - p) * (newton p n * newton p n - p))
              * (((newton p n)⁻¹ * (newton p n)⁻¹) * (1 / 4)) := by
        apply Rat.mul_nonneg (Rat.mul_nonneg he0 he0)
        apply Rat.mul_nonneg (Rat.mul_nonneg (Rat.le_of_lt hqpos) (Rat.le_of_lt hqpos))
        native_decide
      have hsqn1 : p ≤ newton p (n + 1) * newton p (n + 1) := by
        apply le_of_sub_nonneg
        rw [hsqsub]; exact hrhs
      exact ⟨ge_one_of_sq hpos (Rat.le_trans hp1 hsqn1), hsqn1⟩

theorem newton_ge_one (p : Rat) (hp1 : 1 ≤ p) (n : Nat) : 1 ≤ newton p n :=
  (newton_inv p hp1 n).1

theorem newton_sq_ge (p : Rat) (hp1 : 1 ≤ p) (n : Nat) : p ≤ newton p n * newton p n :=
  (newton_inv p hp1 n).2

/-! ## Error contraction: `e_{n+1} ≤ e_n / 4`. -/

theorem regroup_gen (e b : Rat) : (e * e) * (b * (1 / 4)) = (e * (e * b)) * (1 / 4) := by
  rw [Rat.mul_assoc, ← Rat.mul_assoc e b (1 / 4), ← Rat.mul_assoc e (e * b) (1 / 4)]

theorem error_contract (p : Rat) (hp1 : 1 ≤ p) (n : Nat) :
    newton p (n + 1) * newton p (n + 1) - p ≤ (newton p n * newton p n - p) * (1 / 4) := by
  have h01 : (0 : Rat) < 1 := by native_decide
  have hp0 : (0 : Rat) ≤ p := Rat.le_trans (Rat.le_of_lt h01) hp1
  have hx1 : 1 ≤ newton p n := newton_ge_one p hp1 n
  have hsq : p ≤ newton p n * newton p n := newton_sq_ge p hp1 n
  have hx0 : 0 < newton p n := Std.lt_of_lt_of_le h01 hx1
  have hxne : newton p n ≠ 0 := fun h => by rw [h] at hx0; exact Rat.lt_irrefl hx0
  have hyq : newton p n * (newton p n)⁻¹ = 1 := Rat.mul_inv_cancel _ hxne
  have hqpos : 0 < (newton p n)⁻¹ := Rat.inv_pos.mpr hx0
  have he0 : 0 ≤ newton p n * newton p n - p := sub_nonneg_of_le hsq
  have hdef : newton p (n + 1) = (newton p n + p * (newton p n)⁻¹) * (1 / 2) := rfl
  have hsqsub : newton p (n + 1) * newton p (n + 1) - p
      = ((newton p n * newton p n - p) * (newton p n * newton p n - p))
          * (((newton p n)⁻¹ * (newton p n)⁻¹) * (1 / 4)) := by
    rw [hdef]; exact err_step p (newton p n) hyq
  have hqq_nonneg : 0 ≤ (newton p n)⁻¹ * (newton p n)⁻¹ :=
    Rat.mul_nonneg (Rat.le_of_lt hqpos) (Rat.le_of_lt hqpos)
  have hee : newton p n * newton p n - p ≤ newton p n * newton p n := sub_le_self _ hp0
  have hyy_qq : (newton p n * newton p n) * ((newton p n)⁻¹ * (newton p n)⁻¹) = 1 := by
    rw [← sq_prod, hyq, Rat.one_mul]
  have hle1 : (newton p n * newton p n - p) * ((newton p n)⁻¹ * (newton p n)⁻¹) ≤ 1 := by
    have h := Rat.mul_le_mul_of_nonneg_right hee hqq_nonneg
    rwa [hyy_qq] at h
  have hle2 : (newton p n * newton p n - p)
      * ((newton p n * newton p n - p) * ((newton p n)⁻¹ * (newton p n)⁻¹))
      ≤ (newton p n * newton p n - p) := by
    have h := Rat.mul_le_mul_of_nonneg_left hle1 he0
    rwa [Rat.mul_one] at h
  rw [hsqsub, regroup_gen (newton p n * newton p n - p) ((newton p n)⁻¹ * (newton p n)⁻¹)]
  exact Rat.mul_le_mul_of_nonneg_right hle2 (by native_decide)

/-! ## Geometric factor `q4 n = (1/4)^n` and its decay. -/

def q4 : Nat → Rat
  | 0 => 1
  | n + 1 => q4 n * (1 / 4)

theorem q4_nonneg : ∀ n, 0 ≤ q4 n
  | 0 => by native_decide
  | n + 1 => by
      show 0 ≤ q4 n * (1 / 4)
      exact Rat.mul_nonneg (q4_nonneg n) (by native_decide)

theorem q4_step_le (n : Nat) : q4 (n + 1) ≤ q4 n := by
  show q4 n * (1 / 4) ≤ q4 n
  have h := Rat.mul_le_mul_of_nonneg_left (show (1 / 4 : Rat) ≤ 1 from by native_decide) (q4_nonneg n)
  rwa [Rat.mul_one] at h

theorem q4_anti : ∀ {N n : Nat}, N ≤ n → q4 n ≤ q4 N := by
  intro N n h
  induction h with
  | refl => exact Rat.le_refl
  | step _ ih => exact Rat.le_trans (q4_step_le _) ih

/-- `(m+2)/4 ≤ m+1` for `m ≥ 0`. -/
theorem hquarter {m : Rat} (hm : 0 ≤ m) : (m + 2) * (1 / 4) ≤ m + 1 := by
  apply le_of_sub_nonneg
  have hid : (m + 2) * (1 / 4) + (m * (3 / 4) + 1 / 2) = m + 1 := by
    calc (m + 2) * (1 / 4) + (m * (3 / 4) + 1 / 2)
        = (m * (1 / 4) + 2 * (1 / 4)) + (m * (3 / 4) + 1 / 2) := by rw [Rat.add_mul]
      _ = (m * (1 / 4) + 1 / 2) + (m * (3 / 4) + 1 / 2) := by
            rw [show ((2 : Rat) * (1 / 4)) = 1 / 2 from by native_decide]
      _ = (m * (1 / 4) + m * (3 / 4)) + (1 / 2 + 1 / 2) := by rw [add_swap4]
      _ = m * ((1 / 4) + (3 / 4)) + (1 / 2 + 1 / 2) := by rw [← Rat.mul_add]
      _ = m * 1 + 1 := by
            rw [show ((1 / 4 : Rat) + (3 / 4)) = 1 from by native_decide,
              show ((1 / 2 : Rat) + (1 / 2)) = 1 from by native_decide]
      _ = m + 1 := by rw [Rat.mul_one]
  rw [← hid, Rat.add_comm ((m + 2) * (1 / 4)) (m * (3 / 4) + 1 / 2), Rat.add_sub_cancel]
  exact rat_add_nonneg (Rat.mul_nonneg hm (by native_decide)) (by native_decide)

theorem q4_bound (n : Nat) : ((n : Rat) + 1) * q4 n ≤ 1 := by
  induction n with
  | zero =>
      show (((0 : Nat) : Rat) + 1) * q4 0 ≤ 1
      rw [show ((0 : Nat) : Rat) = 0 from by simp, Rat.zero_add, Rat.one_mul]
      exact Rat.le_refl
  | succ n ih =>
      show (((n + 1 : Nat) : Rat) + 1) * q4 (n + 1) ≤ 1
      rw [show ((n + 1 : Nat) : Rat) = (n : Rat) + 1 from by simp]
      show (((n : Rat) + 1) + 1) * (q4 n * (1 / 4)) ≤ 1
      rw [Rat.add_assoc (n : Rat) 1 1, show ((1 : Rat) + 1) = 2 from by native_decide,
        show ((n : Rat) + 2) * (q4 n * (1 / 4)) = (((n : Rat) + 2) * (1 / 4)) * q4 n from by
          rw [Rat.mul_comm (q4 n) (1 / 4), ← Rat.mul_assoc]]
      exact Rat.le_trans
        (Rat.mul_le_mul_of_nonneg_right (hquarter (m := (n : Rat)) Rat.natCast_nonneg) (q4_nonneg n))
        ih

/-- For every `δ > 0` there is `N` with `q4 N ≤ δ` (geometric decay to `0`). -/
theorem exists_pow_le (δ : Rat) (hδ : 0 < δ) : ∃ N, q4 N ≤ δ := by
  obtain ⟨N, hN⟩ := exists_nat_ge δ⁻¹
  refine ⟨N, ?_⟩
  have hb := q4_bound N
  have hNpos : 0 < (N : Rat) + 1 := by
    have h1 : (1 : Rat) ≤ (N : Rat) + 1 := by
      have h := (Rat.add_le_add_right (a := 0) (b := (N : Rat)) (c := 1)).mpr Rat.natCast_nonneg
      rwa [Rat.zero_add] at h
    exact Std.lt_of_lt_of_le (by native_decide) h1
  have key : ((N : Rat) + 1) * q4 N ≤ ((N : Rat) + 1) * δ := by
    refine Rat.le_trans hb ?_
    have hδne : δ ≠ 0 := fun hh => by rw [hh] at hδ; exact Rat.lt_irrefl hδ
    have hdd : δ * δ⁻¹ = 1 := Rat.mul_inv_cancel δ hδne
    have hN1 : δ⁻¹ ≤ (N : Rat) + 1 := Rat.le_trans hN (le_add_one (N : Rat))
    have hmul : δ * δ⁻¹ ≤ δ * ((N : Rat) + 1) :=
      Rat.mul_le_mul_of_nonneg_left hN1 (Rat.le_of_lt hδ)
    rw [hdd] at hmul
    rwa [Rat.mul_comm δ ((N : Rat) + 1)] at hmul
  exact le_of_mul_le_mul_left hNpos key

/-! ## Decay of the error and the convergence modulus. -/

theorem decay (p : Rat) (hp1 : 1 ≤ p) (n : Nat) :
    newton p n * newton p n - p ≤ (p * p - p) * q4 n := by
  induction n with
  | zero =>
      show p * p - p ≤ (p * p - p) * q4 0
      rw [show q4 0 = 1 from rfl, Rat.mul_one]
      exact Rat.le_refl
  | succ n ih =>
      have hc := error_contract p hp1 n
      refine Rat.le_trans hc ?_
      have h1 := Rat.mul_le_mul_of_nonneg_right ih (show (0 : Rat) ≤ 1 / 4 from by native_decide)
      refine Rat.le_trans h1 ?_
      show ((p * p - p) * q4 n) * (1 / 4) ≤ (p * p - p) * q4 (n + 1)
      rw [show q4 (n + 1) = q4 n * (1 / 4) from rfl, ← Rat.mul_assoc]
      exact Rat.le_refl

theorem decay_modulus (p : Rat) (hp1 : 1 ≤ p) (δ : Rat) (hδ : 0 < δ) :
    ∃ N, ∀ m, N ≤ m → newton p m * newton p m - p ≤ δ := by
  have h01 : (0 : Rat) < 1 := by native_decide
  have hp0 : (0 : Rat) ≤ p := Rat.le_trans (Rat.le_of_lt h01) hp1
  have hE0 : 0 ≤ p * p - p := by
    have hpp : p ≤ p * p := by
      have h := Rat.mul_le_mul_of_nonneg_left hp1 hp0
      rwa [Rat.mul_one] at h
    exact sub_nonneg_of_le hpp
  have hCpos : 0 < (p * p - p) + 1 := by
    have h1 : (1 : Rat) ≤ (p * p - p) + 1 := by
      have h := (Rat.add_le_add_right (a := 0) (b := p * p - p) (c := 1)).mpr hE0
      rwa [Rat.zero_add] at h
    exact Std.lt_of_lt_of_le h01 h1
  have hCne : (p * p - p) + 1 ≠ 0 := fun h => by rw [h] at hCpos; exact Rat.lt_irrefl hCpos
  have hδ' : 0 < δ * ((p * p - p) + 1)⁻¹ := Rat.mul_pos hδ (Rat.inv_pos.mpr hCpos)
  obtain ⟨N, hN⟩ := exists_pow_le (δ * ((p * p - p) + 1)⁻¹) hδ'
  refine ⟨N, fun m hm => ?_⟩
  have hd := decay p hp1 m
  have hq : q4 m ≤ q4 N := q4_anti hm
  have hE0m : (p * p - p) * q4 m ≤ (p * p - p) * q4 N := Rat.mul_le_mul_of_nonneg_left hq hE0
  have hCN : (p * p - p) * q4 N ≤ ((p * p - p) + 1) * q4 N :=
    Rat.mul_le_mul_of_nonneg_right (le_add_one (p * p - p)) (q4_nonneg N)
  have hfin : ((p * p - p) + 1) * q4 N ≤ δ := by
    have h := Rat.mul_le_mul_of_nonneg_left hN (Rat.le_of_lt hCpos)
    have heq : ((p * p - p) + 1) * (δ * ((p * p - p) + 1)⁻¹) = δ := by
      rw [Rat.mul_comm δ ((p * p - p) + 1)⁻¹, ← Rat.mul_assoc, Rat.mul_inv_cancel _ hCne,
        Rat.one_mul]
    rwa [heq] at h
  exact Rat.le_trans hd (Rat.le_trans hE0m (Rat.le_trans hCN hfin))

/-! ## Monotonicity and the telescoping Cauchy bound. -/

theorem mono_step (p : Rat) (hp1 : 1 ≤ p) (n : Nat) : 0 ≤ newton p n - newton p (n + 1) := by
  have h01 : (0 : Rat) < 1 := by native_decide
  have hx1 : 1 ≤ newton p n := newton_ge_one p hp1 n
  have hx0 : 0 < newton p n := Std.lt_of_lt_of_le h01 hx1
  have hxne : newton p n ≠ 0 := fun h => by rw [h] at hx0; exact Rat.lt_irrefl hx0
  have hyq : newton p n * (newton p n)⁻¹ = 1 := Rat.mul_inv_cancel _ hxne
  have he0 : 0 ≤ newton p n * newton p n - p := sub_nonneg_of_le (newton_sq_ge p hp1 n)
  have hq0 : 0 ≤ (newton p n)⁻¹ := Rat.le_of_lt (Rat.inv_pos.mpr hx0)
  have hdef : newton p (n + 1) = (newton p n + p * (newton p n)⁻¹) * (1 / 2) := rfl
  rw [hdef, mono_eq_gen p (newton p n) hyq]
  exact Rat.mul_nonneg (Rat.mul_nonneg he0 hq0) (by native_decide)

theorem mono_step_bound (p : Rat) (hp1 : 1 ≤ p) (n : Nat) :
    newton p n - newton p (n + 1) ≤ (newton p n * newton p n - p) * (1 / 2) := by
  have h01 : (0 : Rat) < 1 := by native_decide
  have hx1 : 1 ≤ newton p n := newton_ge_one p hp1 n
  have hx0 : 0 < newton p n := Std.lt_of_lt_of_le h01 hx1
  have hxne : newton p n ≠ 0 := fun h => by rw [h] at hx0; exact Rat.lt_irrefl hx0
  have hyq : newton p n * (newton p n)⁻¹ = 1 := Rat.mul_inv_cancel _ hxne
  have he0 : 0 ≤ newton p n * newton p n - p := sub_nonneg_of_le (newton_sq_ge p hp1 n)
  have hinv1 : (newton p n)⁻¹ ≤ 1 := inv_le_one hx1
  have hdef : newton p (n + 1) = (newton p n + p * (newton p n)⁻¹) * (1 / 2) := rfl
  rw [hdef, mono_eq_gen p (newton p n) hyq]
  apply Rat.mul_le_mul_of_nonneg_right _ (by native_decide)
  have h := Rat.mul_le_mul_of_nonneg_left hinv1 he0
  rwa [Rat.mul_one] at h

/-- Per-step telescoping inequality: `x_k − x_{k+1} ≤ (2/3)(e_k − e_{k+1})`. -/
theorem per_step (p : Rat) (hp1 : 1 ≤ p) (k : Nat) :
    newton p k - newton p (k + 1)
      ≤ ((newton p k * newton p k - p) - (newton p (k + 1) * newton p (k + 1) - p)) * (2 / 3) := by
  have hmb := mono_step_bound p hp1 k
  have hc := error_contract p hp1 k
  refine Rat.le_trans hmb ?_
  have hstep1 : (newton p (k + 1) * newton p (k + 1) - p) * (2 / 3)
      ≤ (newton p k * newton p k - p) * (1 / 6) := by
    have h := Rat.mul_le_mul_of_nonneg_right hc (by native_decide : (0 : Rat) ≤ 2 / 3)
    rwa [Rat.mul_assoc, show ((1 / 4 : Rat) * (2 / 3)) = 1 / 6 from by native_decide] at h
  have h2 : (newton p k * newton p k - p) * (1 / 2)
      = (newton p k * newton p k - p) * (2 / 3) - (newton p k * newton p k - p) * (1 / 6) := by
    rw [← rmul_sub, show ((2 / 3 : Rat) - (1 / 6)) = 1 / 2 from by native_decide]
  have hR : ((newton p k * newton p k - p) - (newton p (k + 1) * newton p (k + 1) - p)) * (2 / 3)
      = (newton p k * newton p k - p) * (2 / 3)
          - (newton p (k + 1) * newton p (k + 1) - p) * (2 / 3) :=
    rsub_mul (newton p k * newton p k - p) (newton p (k + 1) * newton p (k + 1) - p) (2 / 3)
  rw [h2, hR]
  exact sub_le_sub_of_le hstep1

/-- Telescoped bound on `x_m − x_{m+d}`. -/
theorem telescope (p : Rat) (hp1 : 1 ≤ p) (m : Nat) : ∀ d,
    newton p m - newton p (m + d)
      ≤ ((newton p m * newton p m - p) - (newton p (m + d) * newton p (m + d) - p)) * (2 / 3) := by
  intro d
  induction d with
  | zero =>
      rw [Nat.add_zero, Rat.sub_self, Rat.sub_self, Rat.zero_mul]
      exact Rat.le_refl
  | succ d ih =>
      have hps := per_step p hp1 (m + d)
      have hadd := rat_add_le_add ih hps
      have hL : (newton p m - newton p (m + d)) + (newton p (m + d) - newton p ((m + d) + 1))
          = newton p m - newton p ((m + d) + 1) :=
        (rat_sub_split (newton p m) (newton p (m + d)) (newton p ((m + d) + 1))).symm
      have hR : ((newton p m * newton p m - p) - (newton p (m + d) * newton p (m + d) - p)) * (2 / 3)
            + ((newton p (m + d) * newton p (m + d) - p)
                - (newton p ((m + d) + 1) * newton p ((m + d) + 1) - p)) * (2 / 3)
          = ((newton p m * newton p m - p)
              - (newton p ((m + d) + 1) * newton p ((m + d) + 1) - p)) * (2 / 3) := by
        rw [← Rat.add_mul, ← rat_sub_split]
      rw [hL, hR] at hadd
      show newton p m - newton p (m + d + 1)
          ≤ ((newton p m * newton p m - p)
              - (newton p (m + d + 1) * newton p (m + d + 1) - p)) * (2 / 3)
      exact hadd

theorem newton_anti (p : Rat) (hp1 : 1 ≤ p) : ∀ a d, newton p (a + d) ≤ newton p a := by
  intro a d
  induction d with
  | zero => rw [Nat.add_zero]; exact Rat.le_refl
  | succ d ih =>
      have hstep : newton p ((a + d) + 1) ≤ newton p (a + d) :=
        le_of_sub_nonneg (mono_step p hp1 (a + d))
      show newton p (a + d + 1) ≤ newton p a
      exact Rat.le_trans hstep ih

/-! ## The three deliverables.

`newton_ge_one` (deliverable (c)) is proven above as a projection of `newton_inv`. -/

/-- (b) `x_n² → p`, i.e. `x_n² − p → 0`. -/
theorem newton_sq_tendsto (p : Rat) (hp1 : 1 ≤ p) :
    TendsToZero (fun n => newton p n * newton p n - p) := by
  intro ε hε
  obtain ⟨N, hN⟩ := decay_modulus p hp1 ε hε
  refine ⟨N, fun n hn => ?_⟩
  have he : 0 ≤ newton p n * newton p n - p := sub_nonneg_of_le (newton_sq_ge p hp1 n)
  refine ⟨hN n hn, ?_⟩
  have hneg : -(newton p n * newton p n - p) ≤ 0 := by
    have h := Rat.neg_le_neg he
    rwa [Rat.neg_zero] at h
  exact Rat.le_trans hneg (Rat.le_of_lt hε)

/-- (a) The Newton sequence is Cauchy. -/
theorem newton_cauchy (p : Rat) (hp1 : 1 ≤ p) :
    Sounio.RealCauchy.IsCauchy (newton p) := by
  intro ε hε
  have hδ : 0 < ε * (3 / 2) := Rat.mul_pos hε (by native_decide)
  obtain ⟨N, hN⟩ := decay_modulus p hp1 (ε * (3 / 2)) hδ
  refine ⟨N, fun m n hm hn => ?_⟩
  have key : ∀ a b : Nat, a ≤ b → N ≤ a → newton p a - newton p b ≤ ε := by
    intro a b hab ha
    have ht := telescope p hp1 a (b - a)
    rw [Nat.add_sub_cancel' hab] at ht
    have heb : 0 ≤ newton p b * newton p b - p := sub_nonneg_of_le (newton_sq_ge p hp1 b)
    have hstep : ((newton p a * newton p a - p) - (newton p b * newton p b - p)) * (2 / 3)
        ≤ (newton p a * newton p a - p) * (2 / 3) :=
      Rat.mul_le_mul_of_nonneg_right (sub_le_self _ heb) (by native_decide)
    have hea : (newton p a * newton p a - p) * (2 / 3) ≤ ε := by
      have h := Rat.mul_le_mul_of_nonneg_right (hN a ha) (by native_decide : (0 : Rat) ≤ 2 / 3)
      rwa [Rat.mul_assoc, show ((3 / 2 : Rat) * (2 / 3)) = 1 from by native_decide, Rat.mul_one] at h
    exact Rat.le_trans ht (Rat.le_trans hstep hea)
  rcases Nat.le_total m n with hmn | hnm
  · refine ⟨key m n hmn hm, ?_⟩
    have hle : newton p n ≤ newton p m := by
      have h := newton_anti p hp1 m (n - m)
      rwa [Nat.add_sub_cancel' hmn] at h
    have hz : newton p n - newton p m ≤ 0 := by
      have h := Rat.neg_le_neg (sub_nonneg_of_le hle)
      rw [Rat.neg_zero, Rat.neg_sub] at h
      exact h
    exact Rat.le_trans hz (Rat.le_of_lt hε)
  · refine ⟨?_, key n m hnm hn⟩
    have hle : newton p m ≤ newton p n := by
      have h := newton_anti p hp1 n (m - n)
      rwa [Nat.add_sub_cancel' hnm] at h
    have hz : newton p m - newton p n ≤ 0 := by
      have h := Rat.neg_le_neg (sub_nonneg_of_le hle)
      rw [Rat.neg_zero, Rat.neg_sub] at h
      exact h
    exact Rat.le_trans hz (Rat.le_of_lt hε)

#print axioms newton_cauchy
#print axioms newton_sq_tendsto
#print axioms newton_ge_one

end SounioSqrt.RealCauchyField
