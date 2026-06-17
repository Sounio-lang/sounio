import SounioMultiquadIndep

/-!
# SounioRealPlaneGeometry

Canonical Mathlib-free `Real x Real` squared-distance spelling used by public
unit-distance promotion gates.

The chi>=6 promotion validator imports this file and requires candidate-owned
Real-plane unit relations to prove equivalence to `standardRealPlaneDist2 = 1`.
This keeps the public plane formula out of verifier-local temporary code.
-/

namespace SounioSqrt.RealCauchyField

/-- Difference in the Mathlib-free Cauchy-quotient reals. -/
def standardSubR (x y : Real) : Real :=
  addR x (negR y)

/-- Square in the Mathlib-free Cauchy-quotient reals. -/
def standardSqR (x : Real) : Real :=
  mulR x x

/-- The repository-standard squared Euclidean distance on `Real x Real`. -/
def standardRealPlaneDist2 (p q : Real × Real) : Real :=
  addR
    (standardSqR (standardSubR p.1 q.1))
    (standardSqR (standardSubR p.2 q.2))

/-- The repository-standard unit-distance relation on `Real x Real`. -/
def standardRealPlaneUnit (p q : Real × Real) : Prop :=
  standardRealPlaneDist2 p q = qR (1 : Rat)

theorem standardRealPlaneUnit_iff_standard_dist2 (p q : Real × Real) :
    standardRealPlaneUnit p q ↔ standardRealPlaneDist2 p q = qR (1 : Rat) :=
  Iff.rfl

#print axioms standardRealPlaneDist2
#print axioms standardRealPlaneUnit
#print axioms standardRealPlaneUnit_iff_standard_dist2

end SounioSqrt.RealCauchyField
