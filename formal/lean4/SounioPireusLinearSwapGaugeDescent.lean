/-
  FORMAL_PARITY for the outer XOR-linear and input-swap transport used by
  frozen Pireus V13.

  A raw view precomposes a 16 x 16 sign table with one XOR-linear lane
  equivalence and optionally transposes its arguments. These views form a
  lawful action (with the pullback/opposite composition order). They transport
  every unrestricted coboundary by pulling back its potential; input swap is
  already a concrete involution and preserves each basis-fixed coboundary.

  The remaining bridge is precise: after an arbitrary GL(4,F2) map, the
  pulled potential must be rebased by a linear character and encoded back into
  the frozen 11-bit basis-fixed GaugeWord. This file does not yet connect the
  65536 matrix-code predicate to XorLaneEquiv, enumerate 20160 such witnesses,
  or prove the outer 40320-view minimum.
-/
import SounioPireusGaugeSectionCanonicalization

namespace SounioPireusLinearSwapGaugeDescent

open SounioPireusGaugeCoboundaryAction

abbrev Lane := Fin 16
abbrev Potential := Lane -> Bool

theorem lane_xor_val (left right : Lane) :
    (left ^^^ right).val = left.val ^^^ right.val := by
  exact Fin.xor_val_of_two_pow (w := 4) left right

theorem lane_xor_comm (left right : Lane) :
    left ^^^ right = right ^^^ left := by
  apply Fin.ext
  simp only [lane_xor_val]
  exact Nat.xor_comm left.val right.val

structure XorLaneEquiv where
  toFun : Lane -> Lane
  invFun : Lane -> Lane
  mapZero : toFun 0 = 0
  mapXor : ∀ left right : Lane,
    toFun (left ^^^ right) = (toFun left ^^^ toFun right)
  invMapZero : invFun 0 = 0
  invMapXor : ∀ left right : Lane,
    invFun (left ^^^ right) = (invFun left ^^^ invFun right)
  leftInverse : ∀ lane : Lane, invFun (toFun lane) = lane
  rightInverse : ∀ lane : Lane, toFun (invFun lane) = lane

def identityLinear : XorLaneEquiv :=
  { toFun := id
  , invFun := id
  , mapZero := rfl
  , mapXor := by intros; rfl
  , invMapZero := rfl
  , invMapXor := by intros; rfl
  , leftInverse := by intros; rfl
  , rightInverse := by intros; rfl }

-- Raw table views are pullbacks, so action composition reverses map order.
def composeLinear (outer inner : XorLaneEquiv) : XorLaneEquiv :=
  { toFun := fun lane => inner.toFun (outer.toFun lane)
  , invFun := fun lane => outer.invFun (inner.invFun lane)
  , mapZero := by simp [outer.mapZero, inner.mapZero]
  , mapXor := by
      intro left right
      rw [outer.mapXor, inner.mapXor]
  , invMapZero := by simp [inner.invMapZero, outer.invMapZero]
  , invMapXor := by
      intro left right
      rw [inner.invMapXor, outer.invMapXor]
  , leftInverse := by
      intro lane
      rw [inner.leftInverse, outer.leftInverse]
  , rightInverse := by
      intro lane
      rw [outer.rightInverse, inner.rightInverse] }

def inverseLinear (linear : XorLaneEquiv) : XorLaneEquiv :=
  { toFun := linear.invFun
  , invFun := linear.toFun
  , mapZero := linear.invMapZero
  , mapXor := linear.invMapXor
  , invMapZero := linear.mapZero
  , invMapXor := linear.mapXor
  , leftInverse := linear.rightInverse
  , rightInverse := linear.leftInverse }

structure LinearSwapAction where
  linear : XorLaneEquiv
  swap : Bool

def identityAction : LinearSwapAction :=
  { linear := identityLinear, swap := false }

def composeAction (outer inner : LinearSwapAction) : LinearSwapAction :=
  { linear := composeLinear outer.linear inner.linear
  , swap := outer.swap ^^ inner.swap }

def inverseAction (action : LinearSwapAction) : LinearSwapAction :=
  { linear := inverseLinear action.linear
  , swap := action.swap }

def rawAct (action : LinearSwapAction) (table : SignTable) : SignTable :=
  fun cell =>
    let mappedLeft := action.linear.toFun cell.1
    let mappedRight := action.linear.toFun cell.2
    if action.swap then table (mappedRight, mappedLeft)
    else table (mappedLeft, mappedRight)

theorem raw_action_identity (table : SignTable) :
    rawAct identityAction table = table := by
  funext cell
  rfl

theorem raw_action_compose
    (outer inner : LinearSwapAction) (table : SignTable) :
    rawAct (composeAction outer inner) table =
      rawAct outer (rawAct inner table) := by
  funext cell
  cases outerSwap : outer.swap <;>
    cases innerSwap : inner.swap <;>
      simp [rawAct, composeAction, composeLinear, outerSwap, innerSwap]

theorem raw_action_inverse (action : LinearSwapAction) (table : SignTable) :
    rawAct (inverseAction action) (rawAct action table) = table := by
  funext cell
  cases actionSwap : action.swap <;>
    simp [rawAct, inverseAction, inverseLinear, actionSwap,
      action.linear.rightInverse]

def unrestrictedCoboundary (potential : Potential) (cell : Cell) : Bool :=
  potential cell.1 ^^ potential cell.2 ^^ potential (cell.1 ^^^ cell.2)

def potentialGaugeAct (potential : Potential) (table : SignTable) : SignTable :=
  fun cell => table cell ^^ unrestrictedCoboundary potential cell

def pullPotential (linear : XorLaneEquiv) (potential : Potential) : Potential :=
  fun lane => potential (linear.toFun lane)

theorem unrestricted_coboundary_pullback
    (linear : XorLaneEquiv) (potential : Potential) (cell : Cell) :
    unrestrictedCoboundary (pullPotential linear potential) cell =
      unrestrictedCoboundary potential
        (linear.toFun cell.1, linear.toFun cell.2) := by
  simp [unrestrictedCoboundary, pullPotential, linear.mapXor]

theorem raw_action_coboundary_covariant
    (action : LinearSwapAction) (potential : Potential) (table : SignTable) :
    rawAct action (potentialGaugeAct potential table) =
      potentialGaugeAct (pullPotential action.linear potential)
        (rawAct action table) := by
  funext cell
  cases actionSwap : action.swap <;>
    simp [rawAct, potentialGaugeAct, unrestrictedCoboundary,
      pullPotential, actionSwap, action.linear.mapXor,
      lane_xor_comm, Bool.xor_comm]

def gaugeWordPotential (word : GaugeWord) : Potential :=
  fun lane => gaugeValue word lane.val

theorem unrestricted_coboundary_of_gauge_word (word : GaugeWord) :
    unrestrictedCoboundary (gaugeWordPotential word) = gaugeCoboundary word := by
  funext cell
  simp [unrestrictedCoboundary, gaugeWordPotential, gaugeCoboundary,
    lane_xor_val]

theorem basis_fixed_gauge_action_is_potential_action
    (word : GaugeWord) (table : SignTable) :
    gaugeAct word table = potentialGaugeAct (gaugeWordPotential word) table := by
  funext cell
  simp [gaugeAct, potentialGaugeAct, unrestrictedCoboundary,
    gaugeWordPotential, gaugeCoboundary, lane_xor_val]

theorem raw_action_transports_basis_fixed_gauge_to_unrestricted_potential
    (action : LinearSwapAction) (word : GaugeWord) (table : SignTable) :
    rawAct action (gaugeAct word table) =
      potentialGaugeAct (pullPotential action.linear (gaugeWordPotential word))
        (rawAct action table) := by
  rw [basis_fixed_gauge_action_is_potential_action]
  exact raw_action_coboundary_covariant action (gaugeWordPotential word) table

def inputSwapAction : LinearSwapAction :=
  { linear := identityLinear, swap := true }

theorem input_swap_action_is_transpose (table : SignTable) (cell : Cell) :
    rawAct inputSwapAction table cell = table (cell.2, cell.1) := by
  rfl

theorem input_swap_action_is_involution (table : SignTable) :
    rawAct inputSwapAction (rawAct inputSwapAction table) = table := by
  exact raw_action_inverse inputSwapAction table

theorem input_swap_commutes_with_basis_fixed_gauge
    (word : GaugeWord) (table : SignTable) :
    rawAct inputSwapAction (gaugeAct word table) =
      gaugeAct word (rawAct inputSwapAction table) := by
  funext cell
  simp [rawAct, inputSwapAction, identityLinear, gaugeAct, gaugeCoboundary,
    Nat.xor_comm, Bool.xor_comm]

structure LinearSwapDescentBoundary where
  parentBasisFixedGaugeCanonicalizationProved : Bool
  xorLinearRawActionLawsProved : Bool
  concreteInputSwapActionInstantiated : Bool
  inputSwapGaugeCommutationProved : Bool
  unrestrictedCoboundaryTransportProved : Bool
  basisFixedGaugeRebaseAfterLinearMapProved : Bool
  concreteMatrixCodeToXorEquivBridgeProved : Bool
  concreteGL4ActionInstantiated : Bool
  outer40320ViewMinimumProved : Bool
  concreteCanonicalEqualityIffFullDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def linearSwapDescentBoundary : LinearSwapDescentBoundary :=
  { parentBasisFixedGaugeCanonicalizationProved := true
  , xorLinearRawActionLawsProved := true
  , concreteInputSwapActionInstantiated := true
  , inputSwapGaugeCommutationProved := true
  , unrestrictedCoboundaryTransportProved := true
  , basisFixedGaugeRebaseAfterLinearMapProved := false
  , concreteMatrixCodeToXorEquivBridgeProved := false
  , concreteGL4ActionInstantiated := false
  , outer40320ViewMinimumProved := false
  , concreteCanonicalEqualityIffFullDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem linear_swap_descent_progress_does_not_close_v13_target03 :
    linearSwapDescentBoundary.parentBasisFixedGaugeCanonicalizationProved &&
      linearSwapDescentBoundary.xorLinearRawActionLawsProved &&
      linearSwapDescentBoundary.concreteInputSwapActionInstantiated &&
      linearSwapDescentBoundary.inputSwapGaugeCommutationProved &&
      linearSwapDescentBoundary.unrestrictedCoboundaryTransportProved &&
      !linearSwapDescentBoundary.basisFixedGaugeRebaseAfterLinearMapProved &&
      !linearSwapDescentBoundary.concreteMatrixCodeToXorEquivBridgeProved &&
      !linearSwapDescentBoundary.concreteGL4ActionInstantiated &&
      !linearSwapDescentBoundary.outer40320ViewMinimumProved &&
      !linearSwapDescentBoundary.concreteCanonicalEqualityIffFullDeclaredOrbitProved &&
      !linearSwapDescentBoundary.formalTarget03Closed &&
      !linearSwapDescentBoundary.formalParityClosed &&
      !linearSwapDescentBoundary.claimReady := by
  decide

end SounioPireusLinearSwapGaugeDescent
