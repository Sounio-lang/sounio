import Init

/-!
Structural parity for the frozen Sounio Spark pair decommission frame 9026.

Sounio remains the semantic authority. This file models only the admitted
control-flow shape after all Sounio fact guards have passed. It proves that the
shape cannot materialize an effect, that action 41 is non-terminal, and that
only action 49 can confirm legacy-host custody.
-/

namespace SounioSparkPairDecommissionParity

inductive State where
  | slurmOwned
  | decommissionDraining
  | decommissionFenced
  | schedulersWithdrawn
  | legacyRestoring
  | legacyHostOwned
  | recommissionFencing
  | recommissionSchedulersRestoring
  | recommissionPhase1Ready
  | decommissionRecoveryRequired
  deriving DecidableEq, Repr

inductive Custody where
  | fenced
  | slurm
  | k8s
  | legacyHost
  deriving DecidableEq, Repr

inductive Action where
  | statusDecommission
  | beginDecommission
  | confirmDecommissionFenced
  | withdrawSchedulers
  | confirmSchedulersWithdrawn
  | prepareLegacyPair
  | commitLegacyPair
  | restoreLegacyServices
  | retireArbiterSurfaces
  | beginRecommission
  | confirmRecommissionFenced
  | restorePhase1Surfaces
  | confirmRecommissionPhase1
  | confirmRecommissionSlurm
  | enterDecommissionRecovery
  | recoverDecommissionFenced
  | confirmLegacyHost
  deriving DecidableEq, Repr

inductive PlanEffect where
  | none
  deriving DecidableEq, Repr

structure Plan where
  next : State
  effect : PlanEffect
  deriving DecidableEq, Repr

def admittedShape (action : Action) (state : State) (custody : Custody) : Option Plan :=
  match action, state, custody with
  | .statusDecommission, .slurmOwned, .slurm =>
      some ⟨.slurmOwned, .none⟩
  | .statusDecommission, .decommissionDraining, .fenced =>
      some ⟨.decommissionDraining, .none⟩
  | .statusDecommission, .decommissionDraining, .slurm =>
      some ⟨.decommissionDraining, .none⟩
  | .statusDecommission, .decommissionFenced, .fenced =>
      some ⟨.decommissionFenced, .none⟩
  | .statusDecommission, .schedulersWithdrawn, .fenced =>
      some ⟨.schedulersWithdrawn, .none⟩
  | .statusDecommission, .legacyRestoring, .legacyHost =>
      some ⟨.legacyRestoring, .none⟩
  | .statusDecommission, .legacyHostOwned, .legacyHost =>
      some ⟨.legacyHostOwned, .none⟩
  | .statusDecommission, .recommissionFencing, .fenced =>
      some ⟨.recommissionFencing, .none⟩
  | .statusDecommission, .recommissionFencing, .legacyHost =>
      some ⟨.recommissionFencing, .none⟩
  | .statusDecommission, .recommissionSchedulersRestoring, .fenced =>
      some ⟨.recommissionSchedulersRestoring, .none⟩
  | .statusDecommission, .recommissionPhase1Ready, .fenced =>
      some ⟨.recommissionPhase1Ready, .none⟩
  | .statusDecommission, .decommissionRecoveryRequired, _ =>
      some ⟨.decommissionRecoveryRequired, .none⟩
  | .beginDecommission, .slurmOwned, .slurm =>
      some ⟨.decommissionDraining, .none⟩
  | .confirmDecommissionFenced, .decommissionDraining, .fenced =>
      some ⟨.decommissionFenced, .none⟩
  | .withdrawSchedulers, .decommissionFenced, .fenced =>
      some ⟨.decommissionFenced, .none⟩
  | .confirmSchedulersWithdrawn, .decommissionFenced, .fenced =>
      some ⟨.schedulersWithdrawn, .none⟩
  | .prepareLegacyPair, .schedulersWithdrawn, .fenced =>
      some ⟨.schedulersWithdrawn, .none⟩
  | .commitLegacyPair, .schedulersWithdrawn, .fenced =>
      some ⟨.legacyRestoring, .none⟩
  | .restoreLegacyServices, .legacyRestoring, .legacyHost =>
      some ⟨.legacyRestoring, .none⟩
  | .retireArbiterSurfaces, .legacyRestoring, .legacyHost =>
      some ⟨.legacyRestoring, .none⟩
  | .confirmLegacyHost, .legacyRestoring, .legacyHost =>
      some ⟨.legacyHostOwned, .none⟩
  | .beginRecommission, .legacyHostOwned, .legacyHost =>
      some ⟨.recommissionFencing, .none⟩
  | .confirmRecommissionFenced, .recommissionFencing, .fenced =>
      some ⟨.recommissionSchedulersRestoring, .none⟩
  | .restorePhase1Surfaces, .recommissionSchedulersRestoring, .fenced =>
      some ⟨.recommissionSchedulersRestoring, .none⟩
  | .confirmRecommissionPhase1, .recommissionSchedulersRestoring, .fenced =>
      some ⟨.recommissionPhase1Ready, .none⟩
  | .confirmRecommissionSlurm, .recommissionPhase1Ready, .slurm =>
      some ⟨.slurmOwned, .none⟩
  | .enterDecommissionRecovery, .decommissionDraining, _ =>
      some ⟨.decommissionRecoveryRequired, .none⟩
  | .enterDecommissionRecovery, .decommissionFenced, _ =>
      some ⟨.decommissionRecoveryRequired, .none⟩
  | .enterDecommissionRecovery, .schedulersWithdrawn, _ =>
      some ⟨.decommissionRecoveryRequired, .none⟩
  | .enterDecommissionRecovery, .legacyRestoring, _ =>
      some ⟨.decommissionRecoveryRequired, .none⟩
  | .enterDecommissionRecovery, .recommissionFencing, _ =>
      some ⟨.decommissionRecoveryRequired, .none⟩
  | .enterDecommissionRecovery, .recommissionSchedulersRestoring, _ =>
      some ⟨.decommissionRecoveryRequired, .none⟩
  | .enterDecommissionRecovery, .recommissionPhase1Ready, _ =>
      some ⟨.decommissionRecoveryRequired, .none⟩
  | .recoverDecommissionFenced, .decommissionRecoveryRequired, .fenced =>
      some ⟨.decommissionFenced, .none⟩
  | _, _, _ => none

theorem action41_is_not_terminal :
    admittedShape .retireArbiterSurfaces .legacyRestoring .legacyHost =
      some ⟨.legacyRestoring, .none⟩ := rfl

theorem action49_confirms_legacy_host :
    admittedShape .confirmLegacyHost .legacyRestoring .legacyHost =
      some ⟨.legacyHostOwned, .none⟩ := rfl

theorem all_admitted_shapes_are_effect_free
    (action : Action) (state : State) (custody : Custody) (plan : Plan)
    (h : admittedShape action state custody = some plan) :
    plan.effect = .none := by
  cases action <;> cases state <;> cases custody <;>
    simp [admittedShape] at h <;> simp_all

theorem only_action49_confirms_legacy_host
    (action : Action) (state : State) (custody : Custody) (plan : Plan)
    (h : admittedShape action state custody = some plan)
    (notAlreadyOwned : state != .legacyHostOwned)
    (terminal : plan.next = .legacyHostOwned) :
    action = .confirmLegacyHost := by
  cases action <;> cases state <;> cases custody <;>
    simp [admittedShape] at h
  all_goals subst plan
  all_goals simp_all

theorem recovery_can_only_return_to_fenced
    (state : State) (custody : Custody) (plan : Plan)
    (h : admittedShape .recoverDecommissionFenced state custody = some plan) :
    plan.next = .decommissionFenced := by
  cases state <;> cases custody <;> simp [admittedShape] at h
  all_goals subst plan
  all_goals rfl

theorem recommission_returns_to_slurm_only_through_confirmation
    (action : Action) (state : State) (custody : Custody) (plan : Plan)
    (h : admittedShape action state custody = some plan)
    (notAlreadySlurm : state != .slurmOwned)
    (slurm : plan.next = .slurmOwned) :
    action = .confirmRecommissionSlurm := by
  cases action <;> cases state <;> cases custody <;>
    simp [admittedShape] at h
  all_goals subst plan
  all_goals simp_all

#print axioms all_admitted_shapes_are_effect_free
#print axioms only_action49_confirms_legacy_host
#print axioms recovery_can_only_return_to_fenced
#print axioms recommission_returns_to_slurm_only_through_confirmation

#eval "SOUNIO_SPARK_PAIR_DECOMMISSION_LEAN_PARITY_PASS frame=9026 scope=STRUCTURAL effect=NONE"

end SounioSparkPairDecommissionParity
