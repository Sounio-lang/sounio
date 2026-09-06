/-
  Concrete FORMAL_PARITY instantiation of the 2048 basis-fixed Pireus V13
  gauge words as a finite action on Boolean sign tables.

  Gauge words compose by XOR and are self-inverse. Their action adds the
  coboundary g(left) XOR g(right) XOR g(left XOR right) to every sign cell.
  This proves the concrete gauge action laws required by the abstract finite
  action theorem. It does not yet prove that Sounio's tree section is the
  lawful lexicographic minimum of each gauge orbit.
-/
import SounioPireusFiniteActionCanonicalization

namespace SounioPireusGaugeCoboundaryAction

open SounioPireusFiniteActionCanonicalization

def gaugeBits : Nat := 11
def gaugeWords : Nat := 2 ^ gaugeBits

abbrev GaugeWord := Fin (2 ^ gaugeBits)
abbrev Cell := Fin 16 × Fin 16
abbrev SignTable := Cell -> Bool

def gaugeVectors : List Nat := [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]

def gaugeRank? (lane : Nat) : Option Nat := gaugeVectors.idxOf? lane

def gaugeValue (word : GaugeWord) (lane : Nat) : Bool :=
  match gaugeRank? lane with
  | none => false
  | some rank => word.val.testBit rank

def zeroGauge : GaugeWord := ⟨0, by decide⟩

def composeGauge (outer inner : GaugeWord) : GaugeWord := outer ^^^ inner

def inverseGauge (word : GaugeWord) : GaugeWord := word

theorem gauge_value_zero (lane : Nat) : gaugeValue zeroGauge lane = false := by
  cases rank : gaugeRank? lane <;> simp [gaugeValue, zeroGauge, rank]

theorem gauge_value_xor (outer inner : GaugeWord) (lane : Nat) :
    gaugeValue (composeGauge outer inner) lane =
      (gaugeValue outer lane ^^ gaugeValue inner lane) := by
  cases rank : gaugeRank? lane <;>
    simp [gaugeValue, composeGauge, rank, Fin.xor_val_of_two_pow,
      Nat.testBit_xor]

def gaugeCoboundary (word : GaugeWord) (cell : Cell) : Bool :=
  gaugeValue word cell.1.val ^^
    gaugeValue word cell.2.val ^^
    gaugeValue word (cell.1.val ^^^ cell.2.val)

theorem gauge_coboundary_zero (cell : Cell) :
    gaugeCoboundary zeroGauge cell = false := by
  simp [gaugeCoboundary, gauge_value_zero]

theorem gauge_coboundary_xor
    (outer inner : GaugeWord) (cell : Cell) :
    gaugeCoboundary (composeGauge outer inner) cell =
      (gaugeCoboundary outer cell ^^ gaugeCoboundary inner cell) := by
  simp only [gaugeCoboundary, gauge_value_xor]
  simp only [Bool.xor_assoc, Bool.xor_left_comm]

def gaugeAct (word : GaugeWord) (table : SignTable) : SignTable :=
  fun cell => table cell ^^ gaugeCoboundary word cell

theorem gauge_action_identity (table : SignTable) :
    gaugeAct zeroGauge table = table := by
  funext cell
  simp [gaugeAct, gauge_coboundary_zero]

theorem gauge_action_compose
    (outer inner : GaugeWord) (table : SignTable) :
    gaugeAct (composeGauge outer inner) table =
      gaugeAct outer (gaugeAct inner table) := by
  funext cell
  rw [gaugeAct, gauge_coboundary_xor]
  simp only [gaugeAct, Bool.xor_left_comm, Bool.xor_comm]

theorem gauge_action_inverse (word : GaugeWord) (table : SignTable) :
    gaugeAct (inverseGauge word) (gaugeAct word table) = table := by
  funext cell
  simp [inverseGauge, gaugeAct]

def gaugeActionSystem : FiniteActionSystem GaugeWord SignTable :=
  { actions := List.finRange (2 ^ gaugeBits)
  , identity := zeroGauge
  , compose := composeGauge
  , inverse := inverseGauge
  , act := gaugeAct
  , identity_mem := List.mem_finRange zeroGauge
  , compose_mem := fun {_outer _inner} _ _ => List.mem_finRange _
  , inverse_mem := fun {_word} _ => List.mem_finRange _
  , act_identity := gauge_action_identity
  , act_compose := gauge_action_compose
  , act_inverse := gauge_action_inverse }

theorem gauge_word_enumeration_has_exactly_2048_actions :
    gaugeActionSystem.actions.length = 2048 := by
  unfold gaugeActionSystem
  rw [List.length_finRange]
  decide

theorem gauge_vectors_are_exactly_the_11_nonbasis_lanes :
    gaugeVectors =
        ((List.range 16).filter fun lane =>
          lane != 0 && lane != 1 && lane != 2 && lane != 4 && lane != 8) ∧
      gaugeVectors.length = 11 ∧ gaugeVectors.eraseDups.length = 11 ∧
      gaugeVectors.all fun lane =>
        lane < 16 && lane != 0 && lane != 1 && lane != 2 && lane != 4 && lane != 8 := by
  decide

theorem gauge_action_system_satisfies_concrete_finite_action_laws :
    (∀ table, gaugeActionSystem.act gaugeActionSystem.identity table = table) ∧
      (∀ outer inner table,
        gaugeActionSystem.act (gaugeActionSystem.compose outer inner) table =
          gaugeActionSystem.act outer (gaugeActionSystem.act inner table)) ∧
      (∀ word table,
        gaugeActionSystem.act (gaugeActionSystem.inverse word)
            (gaugeActionSystem.act word table) = table) := by
  exact ⟨gauge_action_identity, gauge_action_compose, gauge_action_inverse⟩

structure GaugeInstantiationBoundary where
  genericFiniteActionTheoremProved : Bool
  gaugeWordEnumerationComplete : Bool
  gaugeCoboundaryActionInstantiated : Bool
  gaugeTableLawfulMinimumInstantiated : Bool
  treeSectionEqualsGaugeOrbitMinimumProved : Bool
  concreteGL4ActionInstantiated : Bool
  concreteExecutedNormalizerEqualsAbstractMinimumProved : Bool
  concreteCanonicalEqualityIffDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def gaugeInstantiationBoundary : GaugeInstantiationBoundary :=
  { genericFiniteActionTheoremProved := true
  , gaugeWordEnumerationComplete := true
  , gaugeCoboundaryActionInstantiated := true
  , gaugeTableLawfulMinimumInstantiated := false
  , treeSectionEqualsGaugeOrbitMinimumProved := false
  , concreteGL4ActionInstantiated := false
  , concreteExecutedNormalizerEqualsAbstractMinimumProved := false
  , concreteCanonicalEqualityIffDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem gauge_action_progress_does_not_close_v13_target03 :
    gaugeInstantiationBoundary.genericFiniteActionTheoremProved &&
      gaugeInstantiationBoundary.gaugeWordEnumerationComplete &&
      gaugeInstantiationBoundary.gaugeCoboundaryActionInstantiated &&
      !gaugeInstantiationBoundary.gaugeTableLawfulMinimumInstantiated &&
      !gaugeInstantiationBoundary.treeSectionEqualsGaugeOrbitMinimumProved &&
      !gaugeInstantiationBoundary.concreteGL4ActionInstantiated &&
      !gaugeInstantiationBoundary.concreteExecutedNormalizerEqualsAbstractMinimumProved &&
      !gaugeInstantiationBoundary.concreteCanonicalEqualityIffDeclaredOrbitProved &&
      !gaugeInstantiationBoundary.formalTarget03Closed &&
      !gaugeInstantiationBoundary.formalParityClosed &&
      !gaugeInstantiationBoundary.claimReady := by
  decide

end SounioPireusGaugeCoboundaryAction
