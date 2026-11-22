import EpistemicStat

/-!
# Correlated sources in the statistical layer (no Mathlib)

`EpistemicStat` decides `E_n ≤ num/den` for two independent sources. Two
sources that share an uncertainty component (the same reference material, the
same force-field parameters) are correlated, and the uncertainty of their
difference is not `U₁² + U₂²` but

  D = U₁₁ + U₂₂ − 2·U₁₂,

with `Uᵢⱼ` the expanded covariance (`k²·cov`). The policy is decided exactly
on squares, with no real or float:

  E_n ≤ num/den  ⇔  (x₁ − x₂)² · den² ≤ num² · D.

The generalised-least-squares fusion of the pair weighs source 1 by
`(U₂₂ − U₁₂)/D` and source 2 by `(U₁₁ − U₁₂)/D`. A strong positive correlation
makes one weight negative (Peelle's puzzle) and puts the fused value outside
both sources; `meet` refuses that case too (ADR-016, decision 7).

Theorems:
* `meet_sound`, `meet_refuses_inconsistent`, `meet_refuses_peelle`: a
  successful `meet` satisfies the policy and has non-negative weights; each
  failure is an explicit, named refusal.
* `consistent_comm`: the order of the sources does not change the decision.
* `consistent_mono`: loosening the policy never turns acceptance into refusal.
* `indep_iff`: with zero covariance the predicate is `Epi.Stat.consistent`.
* `ignore_pos_corr`: for positively correlated sources, the independent test
  accepts everything the correlated test accepts. Dropping a shared component
  can only hide an inconsistency.
* `fused_between`: an accepted fusion lies between the two values.
* `neff_pair_bounds`: two sources with correlation `ρ = r/s ∈ [0, 1)` count as
  `n_eff = 2/(1+ρ)` sources, between one and two.
-/


namespace Epi.Corr

open Epi.Stat (Policy)

/-- Two sources and their expanded covariance: `u11 = U₁²`, `u22 = U₂²`,
    `u12 = k²·cov(x₁, x₂)`. `dpos`: the difference has positive variance. -/
structure Pair where
  x1 : Int
  x2 : Int
  u11 : Int
  u22 : Int
  u12 : Int
  dpos : 0 < u11 + u22 - 2 * u12

/-- Squared expanded uncertainty of the difference. -/
def Pair.D (p : Pair) : Int := p.u11 + p.u22 - 2 * p.u12

/-- The policy `E_n ≤ num/den` on the correlated difference. -/
def consistent (pol : Policy) (p : Pair) : Prop :=
  (p.x1 - p.x2) * (p.x1 - p.x2) * ((pol.den : Int) * pol.den)
    ≤ ((pol.num : Int) * pol.num) * p.D

instance (pol : Policy) (p : Pair) : Decidable (consistent pol p) := by
  unfold consistent; exact inferInstance

/-- Weight numerators of the GLS fusion: `w₁ = n1/D`, `w₂ = n2/D`. -/
def Pair.n1 (p : Pair) : Int := p.u22 - p.u12
def Pair.n2 (p : Pair) : Int := p.u11 - p.u12

theorem Pair.n_sum (p : Pair) : p.n1 + p.n2 = p.D := by
  unfold Pair.n1 Pair.n2 Pair.D; omega

/-- Both GLS weights are non-negative. -/
def weights_ok (p : Pair) : Prop := 0 ≤ p.n1 ∧ 0 ≤ p.n2

instance (p : Pair) : Decidable (weights_ok p) := by
  unfold weights_ok; exact inferInstance

inductive Refusal | inconsistent | negativeWeight
deriving DecidableEq

/-- The fused value as an exact rational `num / den`. -/
structure Fused where
  num : Int
  den : Int

def fuse (p : Pair) : Fused := ⟨p.x1 * p.n1 + p.x2 * p.n2, p.D⟩

def meet (pol : Policy) (p : Pair) : Except Refusal Fused :=
  if consistent pol p then
    if weights_ok p then .ok (fuse p) else .error .negativeWeight
  else .error .inconsistent

/-- Soundness: a successful `meet` satisfies the policy, has non-negative
    weights, and returns the GLS fusion. -/
theorem meet_sound {pol : Policy} {p : Pair} {v : Fused}
    (h : meet pol p = .ok v) : consistent pol p ∧ weights_ok p ∧ v = fuse p := by
  unfold meet at h
  split at h
  · split at h
    · cases h; exact ⟨by assumption, by assumption, rfl⟩
    · cases h
  · cases h

theorem meet_refuses_inconsistent {pol : Policy} {p : Pair}
    (h : ¬ consistent pol p) : meet pol p = .error .inconsistent := by
  simp [meet, h]

theorem meet_refuses_peelle {pol : Policy} {p : Pair}
    (hc : consistent pol p) (hw : ¬ weights_ok p) :
    meet pol p = .error .negativeWeight := by
  simp [meet, hc, hw]

/-- The pair with its sources swapped. -/
def Pair.swap (p : Pair) : Pair :=
  ⟨p.x2, p.x1, p.u22, p.u11, p.u12, by have := p.dpos; omega⟩

theorem sq_sub_comm (a b : Int) : (a - b) * (a - b) = (b - a) * (b - a) := by
  have : b - a = -(a - b) := by omega
  rw [this, Int.neg_mul_neg]

/-- Symmetry: the order of the sources does not change the decision. -/
theorem consistent_comm (pol : Policy) (p : Pair) :
    consistent pol p ↔ consistent pol p.swap := by
  unfold consistent Pair.swap Pair.D
  simp only
  rw [sq_sub_comm p.x1 p.x2]
  have : p.u11 + p.u22 - 2 * p.u12 = p.u22 + p.u11 - 2 * p.u12 := by omega
  rw [this]


/-- With zero covariance the test is the independent one of `EpistemicStat`. -/
theorem indep_iff (pol : Policy) (a b : Epi.Stat.Src)
    (h : 0 < ((a.U * a.U : Nat) : Int) + ((b.U * b.U : Nat) : Int) - 2 * 0) :
    consistent pol ⟨a.x, b.x, (a.U * a.U : Nat), (b.U * b.U : Nat), 0, h⟩
      ↔ Epi.Stat.consistent pol a b := by
  unfold consistent Pair.D Epi.Stat.consistent Epi.Stat.dist2 Epi.Stat.var2 Epi.Stat.sq
  simp only
  have hx : (a.x - b.x) * (a.x - b.x) = (((a.x - b.x).natAbs * (a.x - b.x).natAbs : Nat) : Int) := by
    exact Int.natAbs_mul_self.symm
  rw [hx]
  simp only [Int.mul_zero, Int.sub_zero]
  constructor
  · intro hc; exact_mod_cast hc
  · intro hc; exact_mod_cast hc

/-- `x ≤ y → c ≥ 0 → c·x ≤ c·y` on `Int`, written once. -/
theorem mul_le_left {x y c : Int} (h : x ≤ y) (hc : 0 ≤ c) : c * x ≤ c * y :=
  Int.mul_le_mul_of_nonneg_left h hc

/-- Ignoring a positive correlation can only accept more: whatever the
    correlated test accepts, the independent test accepts. -/
theorem ignore_pos_corr (pol : Policy) (p : Pair) (hc : 0 ≤ p.u12)
    (hi : 0 < p.u11 + p.u22 - 2 * 0) (h : consistent pol p) :
    consistent pol ⟨p.x1, p.x2, p.u11, p.u22, 0, hi⟩ := by
  unfold consistent Pair.D at *
  simp only at *
  have hD : p.u11 + p.u22 - 2 * p.u12 ≤ p.u11 + p.u22 - 2 * 0 := by omega
  have hn : (0 : Int) ≤ (pol.num : Int) * pol.num := Int.mul_nonneg (by omega) (by omega)
  exact Int.le_trans h (mul_le_left hD hn)

/-- Loosening the policy never turns acceptance into refusal. -/
theorem consistent_mono {p q : Policy} {pr : Pair}
    (hpq : p.le q) (h : consistent p pr) : consistent q pr := by
  unfold consistent at *
  unfold Policy.le at hpq
  have hD : 0 < pr.D := pr.dpos
  have hpd : (0 : Int) < (p.den : Int) * p.den := Int.mul_pos (by have := p.den_pos; omega) (by have := p.den_pos; omega)
  have hqd : (0 : Int) ≤ (q.den : Int) * q.den := Int.mul_nonneg (by omega) (by omega)
  have hpq' : ((p.num : Int) * q.den) ≤ (q.num : Int) * p.den := by exact_mod_cast hpq
  have hsq : ((p.num : Int) * q.den) * ((p.num : Int) * q.den)
      ≤ ((q.num : Int) * p.den) * ((q.num : Int) * p.den) :=
    Int.mul_le_mul hpq' hpq' (Int.mul_nonneg (by omega) (by omega)) (Int.mul_nonneg (by omega) (by omega))
  -- s := (x1 − x2)², chain: s·q.den²·p.den² ≤ p.num²·D·q.den² ≤ q.num²·p.den²·D
  have s1 : (pr.x1 - pr.x2) * (pr.x1 - pr.x2) * ((p.den : Int) * p.den) * ((q.den : Int) * q.den)
      ≤ (p.num : Int) * p.num * pr.D * ((q.den : Int) * q.den) :=
    Int.mul_le_mul_of_nonneg_right h hqd
  have e3 : (p.num : Int) * p.num * pr.D * ((q.den : Int) * q.den)
      = ((p.num : Int) * q.den) * ((p.num : Int) * q.den) * pr.D := by
    simp only [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
  have s2 : ((p.num : Int) * q.den) * ((p.num : Int) * q.den) * pr.D
      ≤ ((q.num : Int) * p.den) * ((q.num : Int) * p.den) * pr.D :=
    Int.mul_le_mul_of_nonneg_right hsq (Int.le_of_lt hD)
  have e4 : ((q.num : Int) * p.den) * ((q.num : Int) * p.den) * pr.D
      = (q.num : Int) * q.num * pr.D * ((p.den : Int) * p.den) := by
    simp only [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
  have e0 : (pr.x1 - pr.x2) * (pr.x1 - pr.x2) * ((p.den : Int) * p.den) * ((q.den : Int) * q.den)
      = (pr.x1 - pr.x2) * (pr.x1 - pr.x2) * ((q.den : Int) * q.den) * ((p.den : Int) * p.den) := by
    simp only [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
  have key : (pr.x1 - pr.x2) * (pr.x1 - pr.x2) * ((q.den : Int) * q.den) * ((p.den : Int) * p.den)
      ≤ (q.num : Int) * q.num * pr.D * ((p.den : Int) * p.den) := by
    rw [← e0, ← e4]; rw [e3] at s1; exact Int.le_trans s1 s2
  exact Int.le_of_mul_le_mul_right key hpd

/-- An accepted fusion lies between the two values:
    `min(x₁, x₂)·D ≤ num ≤ max(x₁, x₂)·D`. -/
theorem fused_between (p : Pair) (hw : weights_ok p) (hle : p.x1 ≤ p.x2) :
    p.x1 * p.D ≤ (fuse p).num ∧ (fuse p).num ≤ p.x2 * p.D := by
  unfold fuse
  simp only
  rw [← p.n_sum]
  obtain ⟨h1, h2⟩ := hw
  constructor
  · have : p.x1 * p.n2 ≤ p.x2 * p.n2 := Int.mul_le_mul_of_nonneg_right hle h2
    rw [Int.mul_add]; omega
  · have : p.x1 * p.n1 ≤ p.x2 * p.n1 := Int.mul_le_mul_of_nonneg_right hle h1
    rw [Int.mul_add]; omega

/-- `n_eff = 2/(1 + ρ)` for two sources with `ρ = r/s`, `0 ≤ r < s`:
    `n_eff = 2s/(s + r)` lies in `(1, 2]`, stated on cross products. -/
theorem neff_pair_bounds (r s : Nat) (hr : r < s) :
    (s + r) < 2 * s ∧ 2 * s ≤ 2 * (s + r) := by
  omega

/-! ### Witness: H₂ in sII, 200 K, 300 MPa, TIP4P/Ice against SPC/E

Values in 10⁻³ H₂ per cell; expanded (k = 2) covariance in 10⁻⁶ from the
shared H₂ Lennard-Jones channels and each run's Monte Carlo error
(oracle `tests/oracles/epistemic_meet/h2_water_models.cpp`). Correlated, `E_n ≈ 9.0` and `meet`
refuses; taken as independent, `E_n ≈ 0.64` and it would accept. -/

def h2Tip4Spce : Pair := ⟨33080, 31820, 1988035, 1935398, 1951967, by decide⟩

example : meet Epi.Stat.iso13528 h2Tip4Spce = .error .inconsistent :=
  meet_refuses_inconsistent (by decide)

example : consistent Epi.Stat.iso13528 ⟨33080, 31820, 1988035, 1935398, 0, by decide⟩ := by
  decide

/-- Even under `E_n ≤ 10` the pair is refused, now by Peelle: the weight of
    TIP4P/Ice is negative (`U₂₂ − U₁₂ < 0`). -/
example : meet ⟨10, 1, by decide⟩ h2Tip4Spce = .error .negativeWeight :=
  meet_refuses_peelle (by decide) (by decide)

end Epi.Corr
