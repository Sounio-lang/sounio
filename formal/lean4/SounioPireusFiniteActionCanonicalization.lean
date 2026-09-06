/-
  Reusable FORMAL_PARITY theorem for Pireus finite-action canonicalizers.

  For any explicitly enumerated action system closed under identity,
  composition, and inverse, whose action respects identity/composition and is
  undone by the declared inverse, the minimum of the enumerated orbit is equal
  for two states exactly when one state is reachable from the other. The state
  type must have decidable equality and a lawful linear-order minimum. The
  theorem is independent of Mathlib and of the concrete V13 representation.

  This is the abstract half of V13 formal target 03. It does not yet prove
  that the executed GL(4,F2) x input-swap x tree-gauge normalizer realizes the
  abstract action system or the same minimum, so concrete parity remains open.
-/
import Std

namespace SounioPireusFiniteActionCanonicalization

structure FiniteActionSystem
    (Action State : Type) [DecidableEq Action] where
  actions : List Action
  identity : Action
  compose : Action -> Action -> Action
  inverse : Action -> Action
  act : Action -> State -> State
  identity_mem : identity ∈ actions
  compose_mem :
    ∀ {outer inner : Action}, outer ∈ actions -> inner ∈ actions ->
      compose outer inner ∈ actions
  inverse_mem : ∀ {action : Action}, action ∈ actions -> inverse action ∈ actions
  act_identity : ∀ state : State, act identity state = state
  act_compose :
    ∀ (outer inner : Action) (state : State),
      act (compose outer inner) state = act outer (act inner state)
  act_inverse :
    ∀ (action : Action) (state : State),
      act (inverse action) (act action state) = state

section FiniteAction

variable {Action State : Type}
variable [DecidableEq Action]

def FiniteActionSystem.orbit
    (system : FiniteActionSystem Action State) (state : State) : List State :=
  system.actions.map fun action => system.act action state

def FiniteActionSystem.sameOrbit
    (system : FiniteActionSystem Action State) (left right : State) : Prop :=
  ∃ action, action ∈ system.actions ∧ system.act action left = right

theorem orbit_mem_iff_sameOrbit
    [DecidableEq State]
    (system : FiniteActionSystem Action State) (source target : State) :
    target ∈ system.orbit source ↔ system.sameOrbit source target := by
  simp [FiniteActionSystem.orbit, FiniteActionSystem.sameOrbit]

theorem sameOrbit_refl
    (system : FiniteActionSystem Action State) (state : State) :
    system.sameOrbit state state := by
  exact ⟨system.identity, system.identity_mem, system.act_identity state⟩

theorem sameOrbit_symm
    (system : FiniteActionSystem Action State) {left right : State}
    (relation : system.sameOrbit left right) :
    system.sameOrbit right left := by
  rcases relation with ⟨action, actionMem, acts⟩
  refine ⟨system.inverse action, system.inverse_mem actionMem, ?_⟩
  rw [← acts]
  exact system.act_inverse action left

theorem sameOrbit_trans
    (system : FiniteActionSystem Action State) {first second third : State}
    (firstRelation : system.sameOrbit first second)
    (secondRelation : system.sameOrbit second third) :
    system.sameOrbit first third := by
  rcases firstRelation with ⟨firstAction, firstMem, firstActs⟩
  rcases secondRelation with ⟨secondAction, secondMem, secondActs⟩
  refine
    ⟨system.compose secondAction firstAction,
      system.compose_mem secondMem firstMem, ?_⟩
  calc
    system.act (system.compose secondAction firstAction) first =
        system.act secondAction (system.act firstAction first) :=
      system.act_compose secondAction firstAction first
    _ = system.act secondAction second := congrArg (system.act secondAction) firstActs
    _ = third := secondActs

theorem orbit_membership_iff_of_sameOrbit
    [DecidableEq State]
    (system : FiniteActionSystem Action State) {left right : State}
    (relation : system.sameOrbit left right) (candidate : State) :
    candidate ∈ system.orbit left ↔ candidate ∈ system.orbit right := by
  rw [orbit_mem_iff_sameOrbit, orbit_mem_iff_sameOrbit]
  constructor
  · intro leftToCandidate
    exact sameOrbit_trans system (sameOrbit_symm system relation) leftToCandidate
  · intro rightToCandidate
    exact sameOrbit_trans system relation rightToCandidate

theorem minOption_eq_of_membership_iff
    [DecidableEq State] [Min State] [LE State]
    [Std.IsLinearOrder State] [Std.LawfulOrderMin State]
    (left right : List State)
    (membership : ∀ candidate : State, candidate ∈ left ↔ candidate ∈ right) :
    left.min? = right.min? := by
  cases leftMin : left.min? with
  | none =>
      have leftNil : left = [] := List.min?_eq_none_iff.mp leftMin
      have rightNil : right = [] := by
        cases right with
        | nil => rfl
        | cons head tail =>
            exfalso
            have headInLeft : head ∈ left := (membership head).mpr (by simp)
            simp [leftNil] at headInLeft
      simp [rightNil]
  | some minimum =>
      have leftFacts := List.min?_eq_some_iff.mp leftMin
      have rightMin : right.min? = some minimum :=
        List.min?_eq_some_iff.mpr
          ⟨(membership minimum).mp leftFacts.1,
            fun candidate candidateInRight =>
              leftFacts.2 candidate ((membership candidate).mpr candidateInRight)⟩
      exact rightMin.symm

def FiniteActionSystem.canonicalOption
    [Min State] (system : FiniteActionSystem Action State)
    (state : State) : Option State :=
  (system.orbit state).min?

theorem canonicalOption_eq_of_sameOrbit
    [DecidableEq State] [Min State] [LE State]
    [Std.IsLinearOrder State] [Std.LawfulOrderMin State]
    (system : FiniteActionSystem Action State) {left right : State}
    (relation : system.sameOrbit left right) :
    system.canonicalOption left = system.canonicalOption right := by
  apply minOption_eq_of_membership_iff
  exact orbit_membership_iff_of_sameOrbit system relation

theorem orbit_contains_source
    [DecidableEq State]
    (system : FiniteActionSystem Action State) (state : State) :
    state ∈ system.orbit state := by
  exact (orbit_mem_iff_sameOrbit system state state).mpr (sameOrbit_refl system state)

theorem sameOrbit_of_canonicalOption_eq
    [DecidableEq State] [Min State] [LE State]
    [Std.IsLinearOrder State] [Std.LawfulOrderMin State]
    (system : FiniteActionSystem Action State) {left right : State}
    (canonicalEqual : system.canonicalOption left = system.canonicalOption right) :
    system.sameOrbit left right := by
  cases leftMinimum : system.canonicalOption left with
  | none =>
      have orbitMinimumNone : (system.orbit left).min? = none := by
        simpa [FiniteActionSystem.canonicalOption] using leftMinimum
      have orbitNil : system.orbit left = [] :=
        List.min?_eq_none_iff.mp orbitMinimumNone
      have sourceMem := orbit_contains_source system left
      rw [orbitNil] at sourceMem
      simp at sourceMem
  | some minimum =>
      have leftMinimum' : (system.orbit left).min? = some minimum := by
        simpa [FiniteActionSystem.canonicalOption] using leftMinimum
      have rightMinimum : system.canonicalOption right = some minimum :=
        canonicalEqual.symm.trans leftMinimum
      have rightMinimum' : (system.orbit right).min? = some minimum := by
        simpa [FiniteActionSystem.canonicalOption] using rightMinimum
      have leftToMinimum : system.sameOrbit left minimum :=
        (orbit_mem_iff_sameOrbit system left minimum).mp
          (List.min?_mem leftMinimum')
      have rightToMinimum : system.sameOrbit right minimum :=
        (orbit_mem_iff_sameOrbit system right minimum).mp
          (List.min?_mem rightMinimum')
      exact
        sameOrbit_trans system leftToMinimum
          (sameOrbit_symm system rightToMinimum)

theorem canonicalOption_eq_iff_sameOrbit
    [DecidableEq State] [Min State] [LE State]
    [Std.IsLinearOrder State] [Std.LawfulOrderMin State]
    (system : FiniteActionSystem Action State) (left right : State) :
    system.canonicalOption left = system.canonicalOption right ↔
      system.sameOrbit left right := by
  constructor
  · exact sameOrbit_of_canonicalOption_eq system
  · exact canonicalOption_eq_of_sameOrbit system

end FiniteAction

structure V13InstantiationBoundary where
  genericFiniteActionTheoremProved : Bool
  concreteGL4ActionLawsInstantiated : Bool
  concreteGaugeCoboundaryActionInstantiated : Bool
  concreteExecutedNormalizerEqualsAbstractMinimumProved : Bool
  concreteCanonicalEqualityIffDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def v13InstantiationBoundary : V13InstantiationBoundary :=
  { genericFiniteActionTheoremProved := true
  , concreteGL4ActionLawsInstantiated := false
  , concreteGaugeCoboundaryActionInstantiated := false
  , concreteExecutedNormalizerEqualsAbstractMinimumProved := false
  , concreteCanonicalEqualityIffDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem generic_theorem_does_not_close_v13_without_concrete_instantiation :
    v13InstantiationBoundary.genericFiniteActionTheoremProved &&
      !v13InstantiationBoundary.concreteGL4ActionLawsInstantiated &&
      !v13InstantiationBoundary.concreteGaugeCoboundaryActionInstantiated &&
      !v13InstantiationBoundary.concreteExecutedNormalizerEqualsAbstractMinimumProved &&
      !v13InstantiationBoundary.concreteCanonicalEqualityIffDeclaredOrbitProved &&
      !v13InstantiationBoundary.formalTarget03Closed &&
      !v13InstantiationBoundary.formalParityClosed &&
      !v13InstantiationBoundary.claimReady := by
  decide

end SounioPireusFiniteActionCanonicalization
