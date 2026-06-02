import EpistemicEffects
namespace Sounio.EpistemicEffects

-- a valid cell
def k0 : KCell := ⟨0, 0, 1000⟩
theorem k0_valid : kvalid k0 := by unfold kvalid k0; decide

-- measure of a NAT literal is well-typed at Knowledge<Nat>
theorem meas_nat_typed : HasTy [] (.measure (.lit_nat 0) k0) (.tknow .tnat) (singleE .eObserve) :=
  .t_measure _ _ _ _ (.t_lit_nat _ _) k0_valid

-- it steps to kraw k0
theorem meas_nat_steps : (Expr.measure (.lit_nat 0) k0) ⇒ (.kraw k0) :=
  .meas_red (.v_nat 0)

-- but kraw k0 can ONLY have type Knowledge<Real>, never Knowledge<Nat>
theorem kraw_not_nat : ¬ ∃ E, HasTy [] (.kraw k0) (.tknow .tnat) E := by
  rintro ⟨E, h⟩
  have := genKraw h rfl          -- tknow tnat = tknow treal
  injection this with hh          -- tnat = treal
  exact Ty.noConfusion hh

-- CONCLUSION: subject reduction is FALSE for this calculus as written —
-- the reduct cannot be retyped at the original type (Knowledge<Nat>), at ANY effect.
theorem preservation_is_false :
    ∃ e e' T, HasTy [] e (.tknow T) (singleE .eObserve) ∧ (e ⇒ e') ∧ ¬ ∃ E, HasTy [] e' (.tknow T) E :=
  ⟨_, _, .tnat, meas_nat_typed, meas_nat_steps, kraw_not_nat⟩
