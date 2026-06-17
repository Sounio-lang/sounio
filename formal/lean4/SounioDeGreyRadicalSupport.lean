import SounioDeGreyUnitDistance
import SounioMultiquadFaithful

set_option maxHeartbeats 0

/-!
# SounioDeGreyRadicalSupport — exact radical support of the G529 coordinate tables

This file theoremizes the finite support probe for the current de Grey G529 coordinate tables.
It is intentionally scoped: it proves which radicals are used by this coordinate witness, not that
no alternate embedding could exist over a smaller field.
-/

namespace DeGrey529.Support

open DeGrey529

/-- Distinct vertices that appear in at least one edge. -/
def vertsTouched : List Nat :=
  edges.toList.foldl (fun acc e =>
    let acc := if acc.contains e.1 then acc else e.1 :: acc
    if acc.contains e.2 then acc else e.2 :: acc) []

def coeffAt (q : QF) (i : Nat) : Int := gi q.1 i

/-- A basis index is used if some touched vertex has a nonzero coefficient there in X or Y. -/
def idxUsedX (i : Nat) : Bool :=
  vertsTouched.any (fun v => coeffAt (X.getD v ([], 1)) i != 0)

def idxUsedY (i : Nat) : Bool :=
  vertsTouched.any (fun v => coeffAt (Y.getD v ([], 1)) i != 0)

def idxUsed (i : Nat) : Bool := idxUsedX i || idxUsedY i

def usedIndices : List Nat := (List.range 16).filter idxUsed

def unusedIndices : List Nat := (List.range 16).filter (fun i => !(usedIndices.contains i))

def primesOfIdx (i : Nat) : List Nat :=
  (if i % 2 = 1 then [3] else []) ++
    (if (i / 2) % 2 = 1 then [5] else []) ++
    (if (i / 4) % 2 = 1 then [7] else []) ++
    (if (i / 8) % 2 = 1 then [11] else [])

def primesUsed : List Nat :=
  (usedIndices.foldl (fun acc i => acc ++ primesOfIdx i) []).foldl
    (fun acc p => if acc.contains p then acc else acc ++ [p]) []

/-- An edge touches basis index `i` if either endpoint coordinate has a nonzero coefficient there. -/
def edgeUsesIdx (e : Nat × Nat) (i : Nat) : Bool :=
  coeffAt (X.getD e.1 ([], 1)) i != 0 || coeffAt (Y.getD e.1 ([], 1)) i != 0 ||
  coeffAt (X.getD e.2 ([], 1)) i != 0 || coeffAt (Y.getD e.2 ([], 1)) i != 0

def edgeCntAt (i : Nat) : Nat :=
  (edges.toList.filter (fun e => edgeUsesIdx e i)).length

/-- An edge touches prime `p` if some endpoint coordinate uses a radical containing `p`. -/
def edgeUsesPrime (e : Nat × Nat) (p : Nat) : Bool :=
  (List.range 16).any (fun i => edgeUsesIdx e i && (primesOfIdx i).contains p)

def edgeCntPrime (p : Nat) : Nat :=
  (edges.toList.filter (fun e => edgeUsesPrime e p)).length

def basisSupport3511Masks : List Nat := [0, 1, 2, 3, 8, 9, 10, 11]

def primeSupport3511 : List Nat := [3, 5, 11]

def vertexCoordinateSupportedByBasis (v : Nat) (is : List Nat) : Bool :=
  (List.range 16).all (fun i =>
    ((coeffAt (X.getD v ([], 1)) i == 0) && (coeffAt (Y.getD v ([], 1)) i == 0)) ||
      is.contains i)

/-- Direct finite check: every touched vertex coordinate uses only basis masks in `is`. -/
def currentCoordinateTableSupportedByBasis (is : List Nat) : Bool :=
  vertsTouched.all (fun v => vertexCoordinateSupportedByBasis v is)

def idxSupportedByPrimes (i : Nat) (ps : List Nat) : Bool :=
  (primesOfIdx i).all (fun p => ps.contains p)

def vertexCoordinateSupportedByPrimes (v : Nat) (ps : List Nat) : Bool :=
  (List.range 16).all (fun i =>
    ((coeffAt (X.getD v ([], 1)) i == 0) && (coeffAt (Y.getD v ([], 1)) i == 0)) ||
      idxSupportedByPrimes i ps)

/-- Direct finite check: every touched vertex coordinate uses only radicals in `ps`. -/
def currentCoordinateTableSupportedByPrimes (ps : List Nat) : Bool :=
  vertsTouched.all (fun v => vertexCoordinateSupportedByPrimes v ps)

def basisSubsupportFromBits (m : Nat) : List Nat :=
  (if m % 2 = 1 then [0] else []) ++
    (if (m / 2) % 2 = 1 then [1] else []) ++
    (if (m / 4) % 2 = 1 then [2] else []) ++
    (if (m / 8) % 2 = 1 then [3] else []) ++
    (if (m / 16) % 2 = 1 then [8] else []) ++
    (if (m / 32) % 2 = 1 then [9] else []) ++
    (if (m / 64) % 2 = 1 then [10] else []) ++
    (if (m / 128) % 2 = 1 then [11] else [])

def primeSubsupportFromBits (m : Nat) : List Nat :=
  (if m % 2 = 1 then [3] else []) ++
    (if (m / 2) % 2 = 1 then [5] else []) ++
    (if (m / 4) % 2 = 1 then [11] else [])

def sqrt7MaskIndices : List Nat := [4, 5, 6, 7, 12, 13, 14, 15]

def sqrt7FreeNum (l : List Int) : Prop :=
  gi l 4 = 0 ∧ gi l 5 = 0 ∧ gi l 6 = 0 ∧ gi l 7 = 0 ∧
    gi l 12 = 0 ∧ gi l 13 = 0 ∧ gi l 14 = 0 ∧ gi l 15 = 0

def sqrt7FreeQF (q : QF) : Prop := sqrt7FreeNum q.1

def sqrt7FreeNumBool (l : List Int) : Bool :=
  sqrt7MaskIndices.all (fun i => gi l i == 0)

def sqrt7FreeQFBool (q : QF) : Bool := sqrt7FreeNumBool q.1

def qfSupportedByBasis (q : QF) (is : List Nat) : Bool :=
  (List.range 16).all (fun i => coeffAt q i == 0 || is.contains i)

def qfSupportedByPrimes (q : QF) (ps : List Nat) : Bool :=
  (List.range 16).all (fun i => coeffAt q i == 0 || idxSupportedByPrimes i ps)

def edgeDx (e : Nat × Nat) : QF :=
  qsub (X.getD e.1 ([], 1)) (X.getD e.2 ([], 1))

def edgeDy (e : Nat × Nat) : QF :=
  qsub (Y.getD e.1 ([], 1)) (Y.getD e.2 ([], 1))

def edgeDistanceTermsSupportedByBasis (e : Nat × Nat) (is : List Nat) : Bool :=
  qfSupportedByBasis (edgeDx e) is &&
    qfSupportedByBasis (edgeDy e) is &&
    qfSupportedByBasis (qmul (edgeDx e) (edgeDx e)) is &&
    qfSupportedByBasis (qmul (edgeDy e) (edgeDy e)) is &&
    qfSupportedByBasis (dist2 e.1 e.2) is

def edgeDistanceTermsSupportedByPrimes (e : Nat × Nat) (ps : List Nat) : Bool :=
  qfSupportedByPrimes (edgeDx e) ps &&
    qfSupportedByPrimes (edgeDy e) ps &&
    qfSupportedByPrimes (qmul (edgeDx e) (edgeDx e)) ps &&
    qfSupportedByPrimes (qmul (edgeDy e) (edgeDy e)) ps &&
    qfSupportedByPrimes (dist2 e.1 e.2) ps

/-- The exact basis masks used by the current G529 coordinate witness. -/
theorem usedIndices_eq : usedIndices = [0, 1, 2, 3, 8, 9, 10, 11] := by native_decide

/-- The exact basis masks not used by the current G529 coordinate witness. -/
theorem unusedIndices_eq : unusedIndices = [4, 5, 6, 7, 12, 13, 14, 15] := by native_decide

/-- The exact prime radical support used by the current G529 coordinate witness. -/
theorem primesUsed_eq : primesUsed = [3, 5, 11] := by native_decide

/-- No touched edge endpoint uses a radical mask containing `√7`. -/
theorem sqrt7_touching_unused : edgeCntPrime 7 = 0 := by native_decide

/-- The current coordinate witness genuinely uses `√3`. -/
theorem prime3_used : 0 < edgeCntPrime 3 := by native_decide

/-- The current coordinate witness genuinely uses `√5`. -/
theorem prime5_used : 0 < edgeCntPrime 5 := by native_decide

/-- The current coordinate witness genuinely uses `√11`. -/
theorem prime11_used : 0 < edgeCntPrime 11 := by native_decide

/-- Every used basis mask is touched by at least one edge endpoint coordinate. -/
theorem used_basis_masks_touched :
    (0 < edgeCntAt 0) ∧
    (0 < edgeCntAt 1) ∧
    (0 < edgeCntAt 2) ∧
    (0 < edgeCntAt 3) ∧
    (0 < edgeCntAt 8) ∧
    (0 < edgeCntAt 9) ∧
    (0 < edgeCntAt 10) ∧
    (0 < edgeCntAt 11) := by native_decide

/-- Every unused basis mask has zero touched edge endpoint coordinates. -/
theorem unused_basis_masks_untouched :
    edgeCntAt 4 = 0 ∧
    edgeCntAt 5 = 0 ∧
    edgeCntAt 6 = 0 ∧
    edgeCntAt 7 = 0 ∧
    edgeCntAt 12 = 0 ∧
    edgeCntAt 13 = 0 ∧
    edgeCntAt 14 = 0 ∧
    edgeCntAt 15 = 0 := by native_decide

/-- The exact eight-mask support supports every touched coordinate in the current table. -/
theorem exact_basis_supports_current_coordinates :
    currentCoordinateTableSupportedByBasis basisSupport3511Masks = true := by native_decide

/-- Any proper sub-support of the eight used masks fails for the current coordinate table. -/
theorem every_proper_basis_subsupport_rejected :
    (List.range 256).all (fun m =>
      let is := basisSubsupportFromBits m
      if is == basisSupport3511Masks then true
      else !currentCoordinateTableSupportedByBasis is) = true := by native_decide

/-- The exact `{3,5,11}` radical support supports every touched coordinate in the current table. -/
theorem exact_prime_supports_current_coordinates :
    currentCoordinateTableSupportedByPrimes primeSupport3511 = true := by native_decide

/-- Any proper sub-support of `{3,5,11}` fails for the current coordinate table. -/
theorem every_proper_prime_subsupport_rejected :
    (List.range 8).all (fun m =>
      let ps := primeSubsupportFromBits m
      if ps == primeSupport3511 then true
      else !currentCoordinateTableSupportedByPrimes ps) = true := by native_decide

/-- The prime sub-supports that directly support the current coordinate table.

This is the compact decision-summary form of `every_proper_prime_subsupport_rejected`. -/
theorem accepted_prime_subsupports_eq :
    ((List.range 8).filter (fun m =>
      currentCoordinateTableSupportedByPrimes (primeSubsupportFromBits m))).map
        primeSubsupportFromBits = [[3, 5, 11]] := by native_decide

/-- The basis-mask sub-supports that directly support the current coordinate table.

Among all 256 sub-supports of the eight non-`√7` masks, only the full one works. -/
theorem accepted_basis_subsupports_eq :
    ((List.range 256).filter (fun m =>
      currentCoordinateTableSupportedByBasis (basisSubsupportFromBits m))).map
        basisSubsupportFromBits = [[0, 1, 2, 3, 8, 9, 10, 11]] := by native_decide

/-- Direct decision table for the eight prime sub-supports of `{3,5,11}`.

Only the full support `[3,5,11]` supports the current coordinate table. -/
theorem direct_prime_support_decision_table :
    currentCoordinateTableSupportedByPrimes [] = false ∧
    currentCoordinateTableSupportedByPrimes [3] = false ∧
    currentCoordinateTableSupportedByPrimes [5] = false ∧
    currentCoordinateTableSupportedByPrimes [11] = false ∧
    currentCoordinateTableSupportedByPrimes [3, 5] = false ∧
    currentCoordinateTableSupportedByPrimes [3, 11] = false ∧
    currentCoordinateTableSupportedByPrimes [5, 11] = false ∧
    currentCoordinateTableSupportedByPrimes [3, 5, 11] = true := by native_decide

/-- Every touched X-coordinate numerator in the current table is `√7`-free. -/
theorem touched_X_sqrt7Free_all :
    vertsTouched.all (fun v => sqrt7FreeQFBool (X.getD v ([], 1))) = true := by native_decide

/-- Every touched Y-coordinate numerator in the current table is `√7`-free. -/
theorem touched_Y_sqrt7Free_all :
    vertsTouched.all (fun v => sqrt7FreeQFBool (Y.getD v ([], 1))) = true := by native_decide

/-- Faithfulness wrapper for numerators supported on the current `√7`-free mask family. -/
theorem evalNum_faithful_of_sqrt7FreeNum (l : List Int)
    (hfree : sqrt7FreeNum l)
    (hz : (@SounioMultiquadHom.evalNum
      SounioSqrt.RealCauchyField.rootedFieldReal l) =
        SounioSqrt.RealCauchyField.zeroR') :
    gi l 0 = 0 ∧ gi l 1 = 0 ∧ gi l 2 = 0 ∧ gi l 3 = 0 ∧
    gi l 8 = 0 ∧ gi l 9 = 0 ∧ gi l 10 = 0 ∧ gi l 11 = 0 :=
  SounioSqrt.RealCauchyField.evalNum_faithful_on_support l hfree hz

/-- Every edge's `dx`, `dy`, square terms, and squared distance stay in the exact basis support. -/
theorem exact_basis_supports_all_edge_distance_terms :
    edges.toList.all (fun e =>
      edgeDistanceTermsSupportedByBasis e basisSupport3511Masks) = true := by native_decide

/-- Every edge's `dx`, `dy`, square terms, and squared distance stay in `{3,5,11}`. -/
theorem exact_prime_supports_all_edge_distance_terms :
    edges.toList.all (fun e =>
      edgeDistanceTermsSupportedByPrimes e primeSupport3511) = true := by native_decide

/-- Direct decision table for prime supports over all edge-distance expressions. -/
theorem edge_distance_terms_prime_support_decision_table :
    edges.toList.all (fun e => edgeDistanceTermsSupportedByPrimes e []) = false ∧
    edges.toList.all (fun e => edgeDistanceTermsSupportedByPrimes e [3]) = false ∧
    edges.toList.all (fun e => edgeDistanceTermsSupportedByPrimes e [5]) = false ∧
    edges.toList.all (fun e => edgeDistanceTermsSupportedByPrimes e [11]) = false ∧
    edges.toList.all (fun e => edgeDistanceTermsSupportedByPrimes e [3, 5]) = false ∧
    edges.toList.all (fun e => edgeDistanceTermsSupportedByPrimes e [3, 11]) = false ∧
    edges.toList.all (fun e => edgeDistanceTermsSupportedByPrimes e [5, 11]) = false ∧
    edges.toList.all (fun e => edgeDistanceTermsSupportedByPrimes e [3, 5, 11]) = true := by
  native_decide

/-! ## Scoped minimality of the current coordinate table

The next lemmas are deliberately about the *current coordinate table*. They say that any radical
support list capable of containing the table's used prime support must contain all of `3,5,11`.
This is the formal version of "no proper sub-support works for these coordinates", not a claim
about every possible embedding of the same abstract graph.
-/

/-- A candidate radical support contains every prime radical used by the current coordinate table. -/
def currentCoordinateSupportContainedIn (ps : List Nat) : Prop :=
  ∀ p, p ∈ primesUsed → p ∈ ps

/-- A candidate basis-mask support contains every basis mask used by the current coordinate table. -/
def currentCoordinateBasisSupportContainedIn (is : List Nat) : Prop :=
  ∀ i, i ∈ usedIndices → i ∈ is

/-- A proper sub-support of `{3,5,11}` is a list using only these primes and omitting at least one. -/
def properPrimeSubsupport3511 (ps : List Nat) : Prop :=
  (∀ p, p ∈ ps → p ∈ [3, 5, 11]) ∧ ¬ (3 ∈ ps ∧ 5 ∈ ps ∧ 11 ∈ ps)

/-- A proper basis-mask sub-support of the used coordinate table masks omits at least one mask. -/
def properBasisSubsupportForCurrentCoordinates (is : List Nat) : Prop :=
  (∀ i, i ∈ is → i ∈ [0, 1, 2, 3, 8, 9, 10, 11]) ∧
    ¬ (0 ∈ is ∧ 1 ∈ is ∧ 2 ∈ is ∧ 3 ∈ is ∧
      8 ∈ is ∧ 9 ∈ is ∧ 10 ∈ is ∧ 11 ∈ is)

/-- Containing the current coordinate support is equivalent to containing all of `3`, `5`, `11`. -/
theorem current_support_contained_iff (ps : List Nat) :
    currentCoordinateSupportContainedIn ps ↔ 3 ∈ ps ∧ 5 ∈ ps ∧ 11 ∈ ps := by
  unfold currentCoordinateSupportContainedIn
  rw [primesUsed_eq]
  constructor
  · intro h
    exact ⟨h 3 (by simp), h 5 (by simp), h 11 (by simp)⟩
  · rintro ⟨h3, h5, h11⟩ p hp
    simp at hp
    rcases hp with rfl | rfl | rfl
    · exact h3
    · exact h5
    · exact h11

/-- Headline form: the current coordinate table requires exactly the `3/5/11` prime support. -/
theorem current_coordinate_table_requires_exact_3511_support (ps : List Nat) :
    currentCoordinateSupportContainedIn ps ↔ 3 ∈ ps ∧ 5 ∈ ps ∧ 11 ∈ ps :=
  current_support_contained_iff ps

/-- Containing the current coordinate table at basis-mask level means containing exactly the eight
    masks without the `√7` bit. -/
theorem current_basis_support_contained_iff (is : List Nat) :
    currentCoordinateBasisSupportContainedIn is ↔
      0 ∈ is ∧ 1 ∈ is ∧ 2 ∈ is ∧ 3 ∈ is ∧
        8 ∈ is ∧ 9 ∈ is ∧ 10 ∈ is ∧ 11 ∈ is := by
  unfold currentCoordinateBasisSupportContainedIn
  rw [usedIndices_eq]
  constructor
  · intro h
    exact ⟨h 0 (by simp), h 1 (by simp), h 2 (by simp), h 3 (by simp),
      h 8 (by simp), h 9 (by simp), h 10 (by simp), h 11 (by simp)⟩
  · rintro ⟨h0, h1, h2, h3, h8, h9, h10, h11⟩ i hi
    simp at hi
    rcases hi with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact h0
    · exact h1
    · exact h2
    · exact h3
    · exact h8
    · exact h9
    · exact h10
    · exact h11

/-- Headline form: the current coordinate table requires exactly the eight non-`√7` basis masks. -/
theorem current_coordinate_table_requires_exact_basis_support (is : List Nat) :
    currentCoordinateBasisSupportContainedIn is ↔
      0 ∈ is ∧ 1 ∈ is ∧ 2 ∈ is ∧ 3 ∈ is ∧
        8 ∈ is ∧ 9 ∈ is ∧ 10 ∈ is ∧ 11 ∈ is :=
  current_basis_support_contained_iff is

/-- Every support containing the current coordinate table must contain `√3`. -/
theorem current_support_requires_3 (ps : List Nat)
    (h : currentCoordinateSupportContainedIn ps) : 3 ∈ ps :=
  (current_support_contained_iff ps).mp h |>.1

/-- Every support containing the current coordinate table must contain `√5`. -/
theorem current_support_requires_5 (ps : List Nat)
    (h : currentCoordinateSupportContainedIn ps) : 5 ∈ ps :=
  (current_support_contained_iff ps).mp h |>.2.1

/-- Every support containing the current coordinate table must contain `√11`. -/
theorem current_support_requires_11 (ps : List Nat)
    (h : currentCoordinateSupportContainedIn ps) : 11 ∈ ps :=
  (current_support_contained_iff ps).mp h |>.2.2

/-- No proper sub-support of `{3,5,11}` contains the radical support of the current coordinates. -/
theorem no_proper_subsupport_for_current_coordinates (ps : List Nat)
    (hps : properPrimeSubsupport3511 ps) :
    ¬ currentCoordinateSupportContainedIn ps := by
  intro h
  exact hps.2 ((current_support_contained_iff ps).mp h)

/-- No proper basis-mask sub-support contains the current coordinate table. -/
theorem no_proper_basis_subsupport_for_current_coordinates (is : List Nat)
    (his : properBasisSubsupportForCurrentCoordinates is) :
    ¬ currentCoordinateBasisSupportContainedIn is := by
  intro h
  exact his.2 ((current_basis_support_contained_iff is).mp h)

/-- The seven concrete proper subsets of `{3,5,11}` all fail for the current coordinate table. -/
theorem concrete_proper_subsupports_fail :
    (¬ currentCoordinateSupportContainedIn []) ∧
    (¬ currentCoordinateSupportContainedIn [3]) ∧
    (¬ currentCoordinateSupportContainedIn [5]) ∧
    (¬ currentCoordinateSupportContainedIn [11]) ∧
    (¬ currentCoordinateSupportContainedIn [3, 5]) ∧
    (¬ currentCoordinateSupportContainedIn [3, 11]) ∧
    (¬ currentCoordinateSupportContainedIn [5, 11]) := by
  have h_nil : ¬ currentCoordinateSupportContainedIn [] := by
    intro h
    have h3 := current_support_requires_3 [] h
    simp at h3
  have h_3 : ¬ currentCoordinateSupportContainedIn [3] := by
    intro h
    have h5 := current_support_requires_5 [3] h
    simp at h5
  have h_5 : ¬ currentCoordinateSupportContainedIn [5] := by
    intro h
    have h3 := current_support_requires_3 [5] h
    simp at h3
  have h_11 : ¬ currentCoordinateSupportContainedIn [11] := by
    intro h
    have h3 := current_support_requires_3 [11] h
    simp at h3
  have h_35 : ¬ currentCoordinateSupportContainedIn [3, 5] := by
    intro h
    have h11 := current_support_requires_11 [3, 5] h
    simp at h11
  have h_311 : ¬ currentCoordinateSupportContainedIn [3, 11] := by
    intro h
    have h5 := current_support_requires_5 [3, 11] h
    simp at h5
  have h_511 : ¬ currentCoordinateSupportContainedIn [5, 11] := by
    intro h
    have h3 := current_support_requires_3 [5, 11] h
    simp at h3
  exact ⟨h_nil, h_3, h_5, h_11, h_35, h_311, h_511⟩

#print axioms usedIndices_eq
#print axioms unusedIndices_eq
#print axioms primesUsed_eq
#print axioms sqrt7_touching_unused
#print axioms prime3_used
#print axioms prime5_used
#print axioms prime11_used
#print axioms used_basis_masks_touched
#print axioms unused_basis_masks_untouched
#print axioms exact_basis_supports_current_coordinates
#print axioms every_proper_basis_subsupport_rejected
#print axioms exact_prime_supports_current_coordinates
#print axioms every_proper_prime_subsupport_rejected
#print axioms accepted_prime_subsupports_eq
#print axioms accepted_basis_subsupports_eq
#print axioms direct_prime_support_decision_table
#print axioms touched_X_sqrt7Free_all
#print axioms touched_Y_sqrt7Free_all
#print axioms evalNum_faithful_of_sqrt7FreeNum
#print axioms exact_basis_supports_all_edge_distance_terms
#print axioms exact_prime_supports_all_edge_distance_terms
#print axioms edge_distance_terms_prime_support_decision_table
#print axioms current_support_contained_iff
#print axioms current_coordinate_table_requires_exact_3511_support
#print axioms current_basis_support_contained_iff
#print axioms current_coordinate_table_requires_exact_basis_support
#print axioms no_proper_subsupport_for_current_coordinates
#print axioms no_proper_basis_subsupport_for_current_coordinates
#print axioms concrete_proper_subsupports_fail

#eval IO.println "SounioDeGreyRadicalSupport: G529 coordinate-table basis masks are exactly [0,1,2,3,8,9,10,11]; radical support exactly {3,5,11}; √7 masks unused; no proper sub-support contains these coordinates."

end DeGrey529.Support
