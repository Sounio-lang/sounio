/-
  FORMAL_PARITY enumeration of the concrete V13 GL(4,F2) matrix witnesses
  and their two input-swap views.

  Each element of the frozen `invertibleMatrixCodes` scan is converted into an
  `InvertibleMatrixCode`, so its range and invertibility proofs travel with the
  code. The resulting witnesses instantiate concrete `LinearSwapAction`s.

  The construction and membership proofs are kernel checked without consuming
  the earlier `native_decide` census. This module exposes the exact subtype of
  frozen scan members as typed witnesses and exactly two views per member.
  Materialized 20160/40320 lists, their analytic census, action distinctness,
  the outer minimum, and V13 Target-03 remain open.
-/
import SounioPireusMatrixCodeXorEquiv

namespace SounioPireusGL4ActionEnumeration

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

open SounioPireusOperatorOrbitCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusMatrixCodeXorEquiv

theorem scan_membership_facts {code : Nat}
    (membership : code ∈ invertibleMatrixCodes) :
    code < matrixCodes ∧ matrixInvertible code = true := by
  simpa [invertibleMatrixCodes] using membership

abbrev GL4ScanEntry := {code // code ∈ invertibleMatrixCodes}

def matrixWitnessOfScanEntry
    (entry : GL4ScanEntry) : InvertibleMatrixCode :=
  { code := ⟨entry.val, (scan_membership_facts entry.property).1⟩
  , invertible := (scan_membership_facts entry.property).2 }

@[simp] theorem matrix_witness_of_scan_entry_code
    (entry : GL4ScanEntry) :
    (matrixWitnessOfScanEntry entry).code.val = entry.val := by
  rfl

theorem typed_gl4_witness_originates_in_frozen_scan
    (entry : GL4ScanEntry) :
    (matrixWitnessOfScanEntry entry).code.val ∈ invertibleMatrixCodes := by
  simpa only [matrix_witness_of_scan_entry_code] using entry.property

theorem every_admitted_matrix_code_has_typed_entry
    (code : Nat) (inRange : code < matrixCodes)
    (invertible : matrixInvertible code = true) :
    ∃ entry : GL4ScanEntry, (matrixWitnessOfScanEntry entry).code.val = code := by
  have membership :=
    every_invertible_4x4_code_is_in_the_scan code inRange invertible
  exact ⟨⟨code, membership⟩, rfl⟩

structure MatrixLinearSwapView where
  matrixEntry : GL4ScanEntry
  swap : Bool

def viewWitness (view : MatrixLinearSwapView) : InvertibleMatrixCode :=
  matrixWitnessOfScanEntry view.matrixEntry

def concreteLinearSwapActionAt
    (view : MatrixLinearSwapView) : LinearSwapAction :=
  matrixCodeLinearSwapAction (viewWitness view) view.swap

def unswappedViewOf (entry : GL4ScanEntry) : MatrixLinearSwapView :=
  { matrixEntry := entry, swap := false }

def swappedViewOf (entry : GL4ScanEntry) : MatrixLinearSwapView :=
  { matrixEntry := entry, swap := true }

theorem each_scan_entry_has_both_concrete_actions (entry : GL4ScanEntry) :
    concreteLinearSwapActionAt (unswappedViewOf entry) =
        matrixCodeLinearSwapAction (matrixWitnessOfScanEntry entry) false ∧
      concreteLinearSwapActionAt (swappedViewOf entry) =
        matrixCodeLinearSwapAction (matrixWitnessOfScanEntry entry) true := by
  exact ⟨rfl, rfl⟩

theorem the_two_views_have_the_same_matrix_witness (entry : GL4ScanEntry) :
    viewWitness (unswappedViewOf entry) = viewWitness (swappedViewOf entry) := by
  rfl

structure GL4ActionEnumerationBoundary where
  parentMatrixCodeBridgeProved : Bool
  typedWitnessConstructionKernelChecked : Bool
  typedWitnessSubtypeInstantiated : Bool
  everyScanEntryHasTypedWitness : Bool
  everyPredicateWitnessHasEntry : Bool
  twoViewsPerEntryInstantiated : Bool
  concreteLinearSwapActionFamilyInstantiated : Bool
  typedWitnessListInstantiated : Bool
  typedWitnessCount20160Proved : Bool
  importedNativeCensusConsumed : Bool
  outer40320ViewListInstantiated : Bool
  concreteLinearSwapActionListInstantiated : Bool
  outer40320ViewCountProved : Bool
  actionListDistinctnessProved : Bool
  outer40320ViewMinimumProved : Bool
  concreteCanonicalEqualityIffFullDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def gl4ActionEnumerationBoundary : GL4ActionEnumerationBoundary :=
  { parentMatrixCodeBridgeProved := true
  , typedWitnessConstructionKernelChecked := true
  , typedWitnessSubtypeInstantiated := true
  , everyScanEntryHasTypedWitness := true
  , everyPredicateWitnessHasEntry := true
  , twoViewsPerEntryInstantiated := true
  , concreteLinearSwapActionFamilyInstantiated := true
  , typedWitnessListInstantiated := false
  , typedWitnessCount20160Proved := false
  , importedNativeCensusConsumed := false
  , outer40320ViewListInstantiated := false
  , concreteLinearSwapActionListInstantiated := false
  , outer40320ViewCountProved := false
  , actionListDistinctnessProved := false
  , outer40320ViewMinimumProved := false
  , concreteCanonicalEqualityIffFullDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem gl4_action_enumeration_does_not_close_v13_target03 :
    gl4ActionEnumerationBoundary.parentMatrixCodeBridgeProved &&
      gl4ActionEnumerationBoundary.typedWitnessConstructionKernelChecked &&
      gl4ActionEnumerationBoundary.typedWitnessSubtypeInstantiated &&
      gl4ActionEnumerationBoundary.everyScanEntryHasTypedWitness &&
      gl4ActionEnumerationBoundary.everyPredicateWitnessHasEntry &&
      gl4ActionEnumerationBoundary.twoViewsPerEntryInstantiated &&
      gl4ActionEnumerationBoundary.concreteLinearSwapActionFamilyInstantiated &&
      !gl4ActionEnumerationBoundary.typedWitnessListInstantiated &&
      !gl4ActionEnumerationBoundary.typedWitnessCount20160Proved &&
      !gl4ActionEnumerationBoundary.importedNativeCensusConsumed &&
      !gl4ActionEnumerationBoundary.outer40320ViewListInstantiated &&
      !gl4ActionEnumerationBoundary.concreteLinearSwapActionListInstantiated &&
      !gl4ActionEnumerationBoundary.outer40320ViewCountProved &&
      !gl4ActionEnumerationBoundary.actionListDistinctnessProved &&
      !gl4ActionEnumerationBoundary.outer40320ViewMinimumProved &&
      !gl4ActionEnumerationBoundary.concreteCanonicalEqualityIffFullDeclaredOrbitProved &&
      !gl4ActionEnumerationBoundary.formalTarget03Closed &&
      !gl4ActionEnumerationBoundary.formalParityClosed &&
      !gl4ActionEnumerationBoundary.claimReady := by
  decide

end SounioPireusGL4ActionEnumeration
