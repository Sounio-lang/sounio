import SounioSqrtFieldReal

set_option maxHeartbeats 0

namespace SounioSqrt.RealCauchyField

open Sounio.RealCauchy (SounioRealCauchy)

/-!
# Sounio — the canonical ε-order on Cauchy-sequence reals (Mathlib-free)

This file defines the canonical **ε-eventual order** `leR` on representatives of the
Cauchy-sequence reals (`SounioRealCauchy`) and discharges the ordered-field order axioms at the
representative level, so that they descend to the quotient `ℝ := Quotient realSetoid`.

All proofs are Mathlib-free core Lean 4. The toolkit (`TendsToZero`, `RealEq`, `ratAbs`,
`cauchy_bounded`, `isCauchy_abs`, the `Rat` order micro-lemmas) is imported from
`SounioSqrtFieldReal`. `native_decide` is used only for closed scalar `Rat` facts (e.g.
`0 < (1:Rat)`, `¬ ((1:Rat) ≤ 1/2)`), exactly as the imported toolkit already does — `decide`
stalls on `Rat` division and core Lean ships no `Rat.zero_lt_one`.
-/

/-! ## The canonical ε-eventual order on representatives -/

/-- The canonical ε-eventual order on Cauchy representatives: `a ≤ b` iff for every positive `Rat`
    slack `ε`, eventually `aₙ ≤ bₙ + ε`. This is the order ℝ inherits on the quotient. -/
def leR (a b : SounioRealCauchy) : Prop :=
  ∀ ε : Rat, 0 < ε → ∃ N : Nat, ∀ n : Nat, N ≤ n → a.seq n ≤ b.seq n + ε

/-! ## Classical logic micro-helpers (core Lean, no `by_contra`, no Mathlib `push_neg`) -/

/-- Classical `not_forall`: `¬ ∀ x, p x → ∃ x, ¬ p x`. -/
private theorem not_forall_exists {α : Sort u} {p : α → Prop} (h : ¬ ∀ x, p x) : ∃ x, ¬ p x :=
  Classical.byContradiction (fun hc => h (fun x => Classical.byContradiction (fun hx => hc ⟨x, hx⟩)))

/-- Classical `not_imp`: `¬ (P → Q) → P ∧ ¬ Q`. -/
private theorem not_imp {P Q : Prop} (h : ¬ (P → Q)) : P ∧ ¬ Q :=
  ⟨Classical.byContradiction (fun hp => h (fun p => absurd p hp)), fun hq => h (fun _ => hq)⟩

/-- `not_exists`: `¬ (∃ x, p x) → ∀ x, ¬ p x`. -/
private theorem not_exists_forall {α : Sort u} {p : α → Prop} (h : ¬ ∃ x, p x) : ∀ x, ¬ p x :=
  fun x hx => h ⟨x, hx⟩

/-! ## Order axiom: reflexivity -/

/-- `leR` is **reflexive**: take `N = 0`; `aₙ = aₙ + 0 ≤ aₙ + ε` since `0 ≤ ε`. -/
theorem leR_refl (a : SounioRealCauchy) : leR a a := by
  intro ε hε
  refine ⟨0, fun n _ => ?_⟩
  have h0 : a.seq n + 0 ≤ a.seq n + ε := Rat.add_le_add_left.mpr (Rat.le_of_lt hε)
  rwa [Rat.add_zero] at h0

/-! ## Order axiom: transitivity (ε/2 triangle) -/

/-- `leR` is **transitive**: split the slack as `ε/2 + ε/2`. From `aₙ ≤ bₙ + ε/2` and
    `bₙ ≤ cₙ + ε/2` conclude `aₙ ≤ cₙ + ε`. -/
theorem leR_trans {a b c : SounioRealCauchy} (hab : leR a b) (hbc : leR b c) : leR a c := by
  intro ε hε
  obtain ⟨N1, hN1⟩ := hab (ε * (1/2)) (rat_half_pos hε)
  obtain ⟨N2, hN2⟩ := hbc (ε * (1/2)) (rat_half_pos hε)
  refine ⟨max N1 N2, fun n hn => ?_⟩
  have h1 := hN1 n (Nat.le_trans (Nat.le_max_left N1 N2) hn)
  have h2 := hN2 n (Nat.le_trans (Nat.le_max_right N1 N2) hn)
  -- bₙ + ε/2 ≤ (cₙ + ε/2) + ε/2
  have step1 : b.seq n + ε * (1/2) ≤ (c.seq n + ε * (1/2)) + ε * (1/2) :=
    Rat.add_le_add_right.mpr h2
  have chain : a.seq n ≤ (c.seq n + ε * (1/2)) + ε * (1/2) := Rat.le_trans h1 step1
  have heq : (c.seq n + ε * (1/2)) + ε * (1/2) = c.seq n + ε := by
    rw [Rat.add_assoc, rat_add_halves]
  rwa [heq] at chain

/-! ## Order axiom: antisymmetry (descends to `RealEq`) -/

/-- `leR` is **antisymmetric up to `RealEq`**: `leR a b` and `leR b a` force the difference to tend
    to `0`, i.e. `RealEq a b`. From `aₙ ≤ bₙ + ε` get `aₙ - bₙ ≤ ε`; from `bₙ ≤ aₙ + ε` get
    `-(aₙ - bₙ) = bₙ - aₙ ≤ ε`. -/
theorem leR_antisymm {a b : SounioRealCauchy} (hab : leR a b) (hba : leR b a) : RealEq a b := by
  intro ε hε
  obtain ⟨N1, hN1⟩ := hab ε hε
  obtain ⟨N2, hN2⟩ := hba ε hε
  refine ⟨max N1 N2, fun n hn => ?_⟩
  have h1 := hN1 n (Nat.le_trans (Nat.le_max_left N1 N2) hn)
  have h2 := hN2 n (Nat.le_trans (Nat.le_max_right N1 N2) hn)
  show a.seq n - b.seq n ≤ ε ∧ -(a.seq n - b.seq n) ≤ ε
  refine ⟨?_, ?_⟩
  · have h1' : a.seq n ≤ ε + b.seq n := by rw [Rat.add_comm ε (b.seq n)]; exact h1
    exact Rat.sub_right_le_iff_le_add.mpr h1'
  · rw [Rat.neg_sub]
    have h2' : b.seq n ≤ ε + a.seq n := by rw [Rat.add_comm ε (a.seq n)]; exact h2
    exact Rat.sub_right_le_iff_le_add.mpr h2'

/-! ## Order axiom: compatibility with addition (representative / `.seq` level) -/

/-- `leR` is **compatible with right addition**, at the `.seq` level (addition on representatives is
    pointwise). From `aₙ ≤ bₙ + ε` add `cₙ` to both sides and reorder. -/
theorem leR_add_le_add_right_seq {a b c : SounioRealCauchy} (h : leR a b) :
    ∀ ε, 0 < ε → ∃ N, ∀ n, N ≤ n → a.seq n + c.seq n ≤ (b.seq n + c.seq n) + ε := by
  intro ε hε
  obtain ⟨N, hN⟩ := h ε hε
  refine ⟨N, fun n hn => ?_⟩
  have hab := hN n hn
  have step : a.seq n + c.seq n ≤ (b.seq n + ε) + c.seq n := Rat.add_le_add_right.mpr hab
  have heq : (b.seq n + ε) + c.seq n = (b.seq n + c.seq n) + ε := by
    rw [Rat.add_assoc, Rat.add_comm ε (c.seq n), ← Rat.add_assoc]
  rwa [heq] at step

/-! ## Non-degeneracy: `0 ≠ 1` -/

/-- The reals are **non-degenerate**: `0 ≠ 1`. `RealEq 0 1` would force the constant sequence
    `0 - 1 = -1` into every band `[-ε, ε]`; at `ε = 1/2` the band-half `-(-1) = 1 ≤ 1/2` is false. -/
theorem zero_ne_oneR :
    ¬ RealEq Sounio.RealCauchy.SounioRealCauchy.zero Sounio.RealCauchy.SounioRealCauchy.one := by
  intro h
  obtain ⟨N, hN⟩ := h (1/2) (by native_decide)
  obtain ⟨_, h2⟩ := hN N (Nat.le_refl N)
  -- h2 : -(zero.seq N - one.seq N) ≤ 1/2, with zero.seq N = 0, one.seq N = 1
  have hbad : ¬ (-(Sounio.RealCauchy.SounioRealCauchy.zero.seq N
      - Sounio.RealCauchy.SounioRealCauchy.one.seq N) ≤ (1/2 : Rat)) := by
    show ¬ (-((0 : Rat) - 1) ≤ (1/2 : Rat))
    native_decide
  exact hbad h2

/-! ## Order axiom: totality (the fragile one — proved via the Cauchy modulus) -/

/-- A `Rat` squeeze used in `leR_total`: if `aₙ₀ - aₙ ≤ h`, `bₙ - bₙ₀ ≤ h`, `h + h = ε`, and
    `bₙ₀ + ε < aₙ₀`, then `bₙ < aₙ`. (The two Cauchy moduli pin `aₙ, bₙ` within `h` of `aₙ₀, bₙ₀`;
    the strict gap `ε` at `n₀` then survives.) -/
private theorem rat_squeeze {an0 an bn0 bn ε half : Rat}
    (hhalf : half + half = ε)
    (ha : an0 - an ≤ half) (hb : bn - bn0 ≤ half)
    (hd : bn0 + ε < an0) : bn < an := by
  have hS : (an0 - an) + (bn - bn0) ≤ ε := by
    have hsum := rat_add_le_add ha hb
    rwa [hhalf] at hsum
  have h1 : ((an0 - an) + (bn - bn0)) + bn0 ≤ ε + bn0 := Rat.add_le_add_right.mpr hS
  have h2 : ε + bn0 < an0 := by rw [Rat.add_comm ε bn0]; exact hd
  have h3 : ((an0 - an) + (bn - bn0)) + bn0 < an0 := Std.lt_of_le_of_lt h1 h2
  have hsimp : ((an0 - an) + (bn - bn0)) + bn0 = (an0 - an) + bn := by
    rw [Rat.add_assoc, Rat.sub_add_cancel]
  rw [hsimp] at h3
  have hsimp2 : (an0 - an) + bn = an0 + (bn - an) := by
    rw [Rat.sub_eq_add_neg an0 an, Rat.add_assoc, Rat.add_comm (-an) bn,
      ← Rat.sub_eq_add_neg bn an]
  rw [hsimp2] at h3
  have h4 : an0 + (bn - an) < an0 + 0 := by rw [Rat.add_zero]; exact h3
  have h5 : bn - an < 0 := Rat.add_lt_add_left.mp h4
  have h6 : bn < 0 + an := Rat.sub_lt_iff.mp h5
  rwa [Rat.zero_add] at h6

/-- `leR` is **total**: `leR a b ∨ leR b a`. If `leR a b` fails, the negation yields a positive
    `ε₀` with `aₙ > bₙ + ε₀` infinitely often. Pinning `aₙ, bₙ` within `ε₀/2` of a witness `n₀`
    (Cauchy moduli of `a`, `b`) forces `bₙ < aₙ` for all `n ≥ M`, hence `leR b a`. -/
theorem leR_total (a b : SounioRealCauchy) : leR a b ∨ leR b a := by
  cases Classical.em (leR a b) with
  | inl hP => exact Or.inl hP
  | inr hP =>
    refine Or.inr ?_
    -- Extract a positive ε₀ that `leR a b` fails to satisfy, infinitely often.
    have hnab' : ¬ (∀ ε : Rat, 0 < ε → ∃ N : Nat, ∀ n : Nat, N ≤ n → a.seq n ≤ b.seq n + ε) := hP
    obtain ⟨ε₀, hε₀imp⟩ := not_forall_exists hnab'
    obtain ⟨hε₀pos, hnex⟩ := not_imp hε₀imp
    have hall := not_exists_forall hnex
    have hio : ∀ N, ∃ n, N ≤ n ∧ b.seq n + ε₀ < a.seq n := by
      intro N
      obtain ⟨n, hn⟩ := not_forall_exists (hall N)
      obtain ⟨hNn, hnle⟩ := not_imp hn
      exact ⟨n, hNn, Rat.not_le.mp hnle⟩
    -- Cauchy moduli of a and b at ε₀/2.
    obtain ⟨Ma, hMa⟩ := isCauchy_abs a.cauchy (ε₀ * (1/2)) (rat_half_pos hε₀pos)
    obtain ⟨Mb, hMb⟩ := isCauchy_abs b.cauchy (ε₀ * (1/2)) (rat_half_pos hε₀pos)
    obtain ⟨n₀, hn₀ge, hn₀lt⟩ := hio (max Ma Mb)
    have hn₀Ma : Ma ≤ n₀ := Nat.le_trans (Nat.le_max_left Ma Mb) hn₀ge
    have hn₀Mb : Mb ≤ n₀ := Nat.le_trans (Nat.le_max_right Ma Mb) hn₀ge
    -- Eventually b ≤ a.
    have hev : ∀ n, max Ma Mb ≤ n → b.seq n ≤ a.seq n := by
      intro n hn
      have hnMa : Ma ≤ n := Nat.le_trans (Nat.le_max_left Ma Mb) hn
      have hnMb : Mb ≤ n := Nat.le_trans (Nat.le_max_right Ma Mb) hn
      have ha1 := (abs_two_sided (hMa n₀ n hn₀Ma hnMa)).1
      have hb1 := (abs_two_sided (hMb n n₀ hnMb hn₀Mb)).1
      exact Rat.le_of_lt (rat_squeeze (rat_add_halves ε₀) ha1 hb1 hn₀lt)
    -- Conclude leR b a.
    intro ε hε
    refine ⟨max Ma Mb, fun n hn => ?_⟩
    have hba := hev n hn
    have h2 : a.seq n ≤ a.seq n + ε := by
      have h0 : a.seq n + 0 ≤ a.seq n + ε := Rat.add_le_add_left.mpr (Rat.le_of_lt hε)
      rwa [Rat.add_zero] at h0
    exact Rat.le_trans hba h2

/-! ## Order axiom: `mul_nonneg` (the second hard one — via boundedness + a shrinking `δ`) -/

/-- Binomial expansion over `Rat`: `(x+d)(y+d) = xy + (xd + yd + d²)`. -/
private theorem rat_expand (x y d : Rat) :
    (x + d) * (y + d) = x * y + (x * d + y * d + d * d) := by
  rw [Rat.add_mul, Rat.mul_add, Rat.mul_add, Rat.mul_comm d y,
    Rat.add_assoc (x*y) (x*d) (y*d + d*d), ← Rat.add_assoc (x*d) (y*d) (d*d)]

/-- `Ba·d + Bb·d + d = (Ba+Bb+1)·d`. -/
private theorem factorK (Ba Bb d : Rat) : (Ba * d + Bb * d) + d = (Ba + Bb + 1) * d := by
  rw [Rat.add_mul, Rat.add_mul, Rat.one_mul]

/-- Pure-`Rat` core of `mul_nonneg`: if `x, y` are bounded above by `Ba, Bb`, are each `≥ -d`
    (i.e. `0 ≤ x+d`, `0 ≤ y+d`) for a slack `0 ≤ d ≤ 1` with `(Ba+Bb+1)·d ≤ ε`, then
    `0 ≤ x·y + ε`. The witness `(x+d)(y+d) ≥ 0` expands to `xy + (xd+yd+d²)`, and the error
    `xd+yd+d² ≤ (Ba+Bb+1)·d ≤ ε`. -/
private theorem rat_mul_nonneg_core {x y d Ba Bb ε : Rat}
    (hxd : 0 ≤ x + d) (hyd : 0 ≤ y + d)
    (hxB : x ≤ Ba) (hyB : y ≤ Bb)
    (hdnn : 0 ≤ d) (hdle1 : d ≤ 1)
    (hKd : (Ba + Bb + 1) * d ≤ ε) :
    0 ≤ x * y + ε := by
  have hP : 0 ≤ (x + d) * (y + d) := Rat.mul_nonneg hxd hyd
  rw [rat_expand x y d] at hP
  have hb1 : x * d ≤ Ba * d := Rat.mul_le_mul_of_nonneg_right hxB hdnn
  have hb2 : y * d ≤ Bb * d := Rat.mul_le_mul_of_nonneg_right hyB hdnn
  have hb3 : d * d ≤ d := by
    have h := Rat.mul_le_mul_of_nonneg_right hdle1 hdnn
    rwa [Rat.one_mul] at h
  have hRsum : (x * d + y * d) + d * d ≤ (Ba * d + Bb * d) + d :=
    rat_add_le_add (rat_add_le_add hb1 hb2) hb3
  have hR : (x * d + y * d) + d * d ≤ ε := by
    refine Rat.le_trans hRsum ?_
    rw [factorK Ba Bb d]; exact hKd
  have hchain : x * y + (x * d + y * d + d * d) ≤ x * y + ε := Rat.add_le_add_left.mpr hR
  exact Rat.le_trans hP hchain

/-- `leR` satisfies **`mul_nonneg`** at the `.seq` level: if `0 ≤ a` and `0 ≤ b` (in the `leR`
    sense), then `0 ≤ aₙ·bₙ + ε` eventually, for every `ε > 0`. Both reps are eventually bounded
    (`cauchy_bounded`), and `aₙ, bₙ ≥ -δ` for the shrinking slack `δ = ε·(Ba+Bb+1+ε)⁻¹ < 1`, so
    `rat_mul_nonneg_core` applies. This is the representative form of `0 ≤ a → 0 ≤ b → 0 ≤ a·b`. -/
theorem leR_mul_nonneg {a b : SounioRealCauchy}
    (ha : leR Sounio.RealCauchy.SounioRealCauchy.zero a)
    (hb : leR Sounio.RealCauchy.SounioRealCauchy.zero b) :
    ∀ ε, 0 < ε → ∃ N, ∀ n, N ≤ n → (0 : Rat) ≤ a.seq n * b.seq n + ε := by
  intro ε hε
  obtain ⟨Ba, Na, hBa, haB⟩ := cauchy_bounded a.cauchy
  obtain ⟨Bb, Nb, hBb, hbB⟩ := cauchy_bounded b.cauchy
  have hK : 0 < Ba + Bb + 1 := rat_K_pos hBa hBb
  -- The shrinking divisor Ba+Bb+1+ε is positive and nonzero.
  have hKle : Ba + Bb + 1 ≤ Ba + Bb + 1 + ε := by
    have h0 : Ba + Bb + 1 + 0 ≤ Ba + Bb + 1 + ε := Rat.add_le_add_left.mpr (Rat.le_of_lt hε)
    rwa [Rat.add_zero] at h0
  have hKε : 0 < Ba + Bb + 1 + ε := Std.lt_of_lt_of_le hK hKle
  have hKεne : Ba + Bb + 1 + ε ≠ 0 := fun h => by rw [h] at hKε; exact Rat.lt_irrefl hKε
  -- δ = ε·(Ba+Bb+1+ε)⁻¹ : positive, ≤ 1, and (Ba+Bb+1)·δ ≤ ε.
  have hδpos : 0 < ε * (Ba + Bb + 1 + ε)⁻¹ := Rat.mul_pos hε (Rat.inv_pos.mpr hKε)
  have hδnn : 0 ≤ ε * (Ba + Bb + 1 + ε)⁻¹ := Rat.le_of_lt hδpos
  have hδle1 : ε * (Ba + Bb + 1 + ε)⁻¹ ≤ 1 := by
    have hεle : ε ≤ Ba + Bb + 1 + ε := by
      have h0 : 0 + ε ≤ (Ba + Bb + 1) + ε := Rat.add_le_add_right.mpr (Rat.le_of_lt hK)
      rwa [Rat.zero_add] at h0
    have hmul : ε * (Ba + Bb + 1 + ε)⁻¹ ≤ (Ba + Bb + 1 + ε) * (Ba + Bb + 1 + ε)⁻¹ :=
      Rat.mul_le_mul_of_nonneg_right hεle (Rat.le_of_lt (Rat.inv_pos.mpr hKε))
    rwa [Rat.mul_inv_cancel _ hKεne] at hmul
  have hKδ : (Ba + Bb + 1) * (ε * (Ba + Bb + 1 + ε)⁻¹) ≤ ε := by
    have hmul : (Ba + Bb + 1) * (ε * (Ba + Bb + 1 + ε)⁻¹)
        ≤ (Ba + Bb + 1 + ε) * (ε * (Ba + Bb + 1 + ε)⁻¹) :=
      Rat.mul_le_mul_of_nonneg_right hKle hδnn
    rwa [rat_K_delta hKεne] at hmul
  -- aₙ ≥ -δ and bₙ ≥ -δ eventually (leR 0 a / leR 0 b at slack δ).
  obtain ⟨Nad, hNad⟩ := ha (ε * (Ba + Bb + 1 + ε)⁻¹) hδpos
  obtain ⟨Nbd, hNbd⟩ := hb (ε * (Ba + Bb + 1 + ε)⁻¹) hδpos
  refine ⟨max (max Na Nb) (max Nad Nbd), fun n hn => ?_⟩
  have hnNa : Na ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_left _ _)) hn
  have hnNb : Nb ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_left _ _)) hn
  have hnNad : Nad ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_right _ _)) hn
  have hnNbd : Nbd ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_right _ _)) hn
  have hxd : 0 ≤ a.seq n + ε * (Ba + Bb + 1 + ε)⁻¹ := hNad n hnNad
  have hyd : 0 ≤ b.seq n + ε * (Ba + Bb + 1 + ε)⁻¹ := hNbd n hnNbd
  have hxB : a.seq n ≤ Ba := Rat.le_trans (le_ratAbs (a.seq n)) (haB n hnNa)
  have hyB : b.seq n ≤ Bb := Rat.le_trans (le_ratAbs (b.seq n)) (hbB n hnNb)
  exact rat_mul_nonneg_core hxd hyd hxB hyB hδnn hδle1 hKδ

#print axioms leR_refl
#print axioms leR_trans
#print axioms leR_antisymm
#print axioms leR_add_le_add_right_seq
#print axioms zero_ne_oneR
#print axioms leR_total
#print axioms leR_mul_nonneg

end SounioSqrt.RealCauchyField
