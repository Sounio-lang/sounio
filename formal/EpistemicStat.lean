/-!
# Statistical layer of the Knowledge/Contest calculus (no Mathlib)

A statistical source is a value `x` with expanded uncertainty `U`. The
consistency policy is `E_n ≤ num/den` (ISO 13528 default: 1/1), checked
exactly on squares so that no real number or float enters the decision:

  E_n ≤ num/den  ⇔  (x₁ − x₂)² · den² ≤ num² · (U₁² + U₂²).

Theorems: soundness of acceptance, explicit refusal, symmetry, and
monotonicity in the policy (loosening never turns acceptance into refusal).
-/

namespace Epi.Stat

structure Src where
  x : Int
  U : Nat

/-- Policy `E_n ≤ num / den`, with `den > 0`. -/
structure Policy where
  num : Nat
  den : Nat
  den_pos : 0 < den

def sq (n : Nat) : Nat := n * n

/-- Squared distance between the two values. -/
def dist2 (a b : Src) : Nat := sq (a.x - b.x).natAbs

/-- Combined squared uncertainty. -/
def var2 (a b : Src) : Nat := sq a.U + sq b.U

/-- The exact consistency predicate. -/
def consistent (p : Policy) (a b : Src) : Prop :=
  dist2 a b * sq p.den ≤ sq p.num * var2 a b

instance (p : Policy) (a b : Src) : Decidable (consistent p a b) := by
  unfold consistent; exact inferInstance

inductive Refusal | inconsistent
deriving DecidableEq

/-- Fused value as an exact rational: inverse-variance weighted mean
    `(x₁·U₂² + x₂·U₁²) / (U₁² + U₂²)`, kept as numerator and denominator. -/
structure Fused where
  num : Int
  den : Nat

def fuse (a b : Src) : Fused :=
  ⟨a.x * (sq b.U : Int) + b.x * (sq a.U : Int), var2 a b⟩

def meet (p : Policy) (a b : Src) : Except Refusal Fused :=
  if consistent p a b then .ok (fuse a b) else .error .inconsistent

/-- Soundness: a successful `meet` satisfies the declared policy. -/
theorem meet_sound {p : Policy} {a b : Src} {v : Fused}
    (h : meet p a b = .ok v) : consistent p a b := by
  unfold meet at h
  split at h
  · assumption
  · cases h

/-- Refusal: above the policy, `meet` refuses explicitly. -/
theorem meet_refuses {p : Policy} {a b : Src} (h : ¬ consistent p a b) :
    meet p a b = .error .inconsistent := by
  simp [meet, h]

theorem dist2_comm (a b : Src) : dist2 a b = dist2 b a := by
  unfold dist2
  have : (a.x - b.x).natAbs = (b.x - a.x).natAbs := by
    rw [← Int.natAbs_neg]; congr 1; omega
  rw [this]

theorem var2_comm (a b : Src) : var2 a b = var2 b a := by
  unfold var2; exact Nat.add_comm _ _

/-- Symmetry: the order of the sources does not change the decision. -/
theorem consistent_comm (p : Policy) (a b : Src) :
    consistent p a b ↔ consistent p b a := by
  unfold consistent; rw [dist2_comm, var2_comm]

/-- `p ≤ q` as rationals: `p.num / p.den ≤ q.num / q.den`. -/
def Policy.le (p q : Policy) : Prop := p.num * q.den ≤ q.num * p.den

/-- Monotonicity: acceptance under a policy implies acceptance under any looser one. -/
theorem consistent_mono {p q : Policy} {a b : Src}
    (hpq : p.le q) (h : consistent p a b) : consistent q a b := by
  unfold consistent sq at *
  unfold Policy.le at hpq
  -- Square the policy inequality.
  have hsq : (p.num * q.den) * (p.num * q.den) ≤ (q.num * p.den) * (q.num * p.den) :=
    Nat.mul_le_mul hpq hpq
  have hd : 0 < p.den * p.den := Nat.mul_pos p.den_pos p.den_pos
  -- Multiply the hypothesis by q.den² and chain.
  have step1 : dist2 a b * (p.den * p.den) * (q.den * q.den)
      ≤ p.num * p.num * var2 a b * (q.den * q.den) :=
    Nat.mul_le_mul_right _ h
  have step2 : p.num * p.num * var2 a b * (q.den * q.den)
      = (p.num * q.den) * (p.num * q.den) * var2 a b := by
    simp only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
  have step3 : (p.num * q.den) * (p.num * q.den) * var2 a b
      ≤ (q.num * p.den) * (q.num * p.den) * var2 a b :=
    Nat.mul_le_mul_right _ hsq
  have step4 : (q.num * p.den) * (q.num * p.den) * var2 a b
      = q.num * q.num * var2 a b * (p.den * p.den) := by
    simp only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
  have step0 : dist2 a b * (p.den * p.den) * (q.den * q.den)
      = dist2 a b * (q.den * q.den) * (p.den * p.den) := by
    simp only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
  have key : dist2 a b * (q.den * q.den) * (p.den * p.den)
      ≤ q.num * q.num * var2 a b * (p.den * p.den) := by
    rw [← step0, ← step4]; rw [step2] at step1; exact Nat.le_trans step1 step3
  exact Nat.le_of_mul_le_mul_right key hd

/-- The ISO 13528 default policy `E_n ≤ 1`. -/
def iso13528 : Policy := ⟨1, 1, by decide⟩

/-- Witness: the venlafaxine example (100 ± 4 vs 112 ± 5, E_n ≈ 1.87) is refused
    under the default policy and accepted under `E_n ≤ 2`. -/
example : meet iso13528 ⟨100, 4⟩ ⟨112, 5⟩ = .error .inconsistent :=
  meet_refuses (by decide)
example : consistent ⟨2, 1, by decide⟩ ⟨100, 4⟩ ⟨112, 5⟩ := by decide

end Epi.Stat
