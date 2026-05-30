/-
Spike: internalise a Sounio/souc_sat UNSAT certificate inside Lean core's
verified LRAT checker (Std.Tactic.BVDecide.LRAT.check / check_sound).
No Mathlib. Tiny instance first to validate the mechanism.

Tiny UNSAT CNF over one variable (CNF Nat var 0 = DIMACS literal 1):
    (x0) ∧ (¬x0)
LRAT refutation: derive the empty clause by RUP from clauses 1 and 2.
-/
import Std.Tactic.BVDecide.LRAT.Checker

open Std.Sat
open Std.Tactic.BVDecide.LRAT

/-- `(x0) ∧ (¬x0)` as a `CNF Nat`. Literal `(v, b)`: variable `v`, polarity `b`. -/
def tinyCnf : CNF Nat := { clauses := #[ [(0, true)], [(0, false)] ] }

/-- LRAT proof: clause id 3 is the empty clause, RUP-derived from clauses 1 and 2. -/
def tinyProof : Array IntAction := #[ Action.addEmpty 3 #[1, 2] ]

/-- The verified checker accepts the certificate (computed by reflection). -/
example : check tinyProof tinyCnf = true := by native_decide

/-- Therefore the CNF is genuinely unsatisfiable — proved inside Lean. -/
theorem tinyCnf_unsat : tinyCnf.Unsat :=
  check_sound tinyProof tinyCnf (by native_decide)

#print axioms tinyCnf_unsat
