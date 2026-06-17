import SounioDeGreyChi5TransferWf
import SounioDeGreyChi5Param

set_option maxHeartbeats 0

/-!
# Sounio — de Grey G529 transfer through the checked `{3,5,11}` fragment

`SounioDeGreyChi5TransferWf` proves the abstract chromatic transfer for any guarded
`QFTransferWf`. This file adds the review-facing support boundary for the current G529 embedding:
the edge endpoints and edge-distance terms used by the reflected LRAT obstruction all lie in the
prime-radical support `{3,5,11}`.

The theorem here is intentionally scoped. This lower-level file does not itself introduce the
three-root field interface; that public target surface lives in `SounioDeGreyChi5Rooted3511`. It
also does not claim universal minimality over all possible G529 embeddings. It says that any target
receiving exactly the checked QF operations needed by the current `{3,5,11}` fragment carries the
current no-4-colouring obstruction.
-/

namespace DeGrey529.Transfer3511

open UnitDistanceChromatic
open DeGrey529
open DeGrey529.Concrete

/-- Stable spelling for the prime-radical support of the current G529 embedding. -/
def primeSupport : List Nat := DeGrey529.Support.primeSupport3511

/-- Local well-formedness for the checked `{3,5,11}` QF fragment:
nonzero denominator plus support by the three prime radicals. -/
def qf3511Wf (x : QF) : Prop :=
  DeGrey529.TransferWf.qfWf x ∧
    DeGrey529.Support.qfSupportedByPrimes x primeSupport = true

/-- A transfer target receiving the QF operations needed by the checked `{3,5,11}` fragment.

The homomorphism laws are only required on locally well-formed inputs. The support fields record
the finite audit for the current G529 embedding and LRAT obstruction. -/
structure QF3511TransferWf where
  F : Type
  add : F → F → F
  mul : F → F → F
  sub : F → F → F
  phi : QF → F
  isUnitVal : F → Prop
  hadd : ∀ a b, qf3511Wf a → qf3511Wf b → phi (qadd a b) = add (phi a) (phi b)
  hmul : ∀ a b, qf3511Wf a → qf3511Wf b → phi (qmul a b) = mul (phi a) (phi b)
  hsub : ∀ a b, qf3511Wf a → qf3511Wf b → phi (qsub a b) = sub (phi a) (phi b)
  hunit : ∀ d, qf3511Wf d → isOne d = true → isUnitVal (phi d)
  hcoord_edge_endpoints3511 :
    DeGrey529.Param.edgeEndpointsInPrimeSubplane primeSupport = true
  hedge_terms3511 :
    DeGrey529.edges.toList.all (fun e =>
      DeGrey529.Support.edgeDistanceTermsSupportedByPrimes e primeSupport) = true
  hcurrent_lrat_support :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane primeSupport

namespace QF3511TransferWf

variable (T : QF3511TransferWf)

/-- Squared-distance unit relation transported to the target fragment. -/
def unit (p q : T.F × T.F) : Prop :=
  T.isUnitVal (T.add (T.mul (T.sub p.1 q.1) (T.sub p.1 q.1))
                     (T.mul (T.sub p.2 q.2) (T.sub p.2 q.2)))

/-- `φ`-image of the exact symbolic embedding of vertex `v`. -/
def embF (v : Nat) : T.F × T.F := (T.phi (emb v).1, T.phi (emb v).2)

private theorem edge_endpoint_qf3511 (T : QF3511TransferWf)
    (e : Nat × Nat) (he : e ∈ DeGrey529.edges.toList) :
    qf3511Wf (emb e.1).1 ∧ qf3511Wf (emb e.1).2 ∧
      qf3511Wf (emb e.2).1 ∧ qf3511Wf (emb e.2).2 := by
  have hbool :
      (DeGrey529.Param.primeSubplane primeSupport (emb e.1) &&
        DeGrey529.Param.primeSubplane primeSupport (emb e.2)) = true :=
    (List.all_eq_true.mp T.hcoord_edge_endpoints3511) e he
  have hends :
      DeGrey529.Param.primeSubplane primeSupport (emb e.1) = true ∧
        DeGrey529.Param.primeSubplane primeSupport (emb e.2) = true := by
    simpa [Bool.and_eq_true] using hbool
  have hleft :
      DeGrey529.Support.qfSupportedByPrimes (emb e.1).1 primeSupport = true ∧
        DeGrey529.Support.qfSupportedByPrimes (emb e.1).2 primeSupport = true := by
    simpa [DeGrey529.Param.primeSubplane, Bool.and_eq_true] using hends.1
  have hright :
      DeGrey529.Support.qfSupportedByPrimes (emb e.2).1 primeSupport = true ∧
        DeGrey529.Support.qfSupportedByPrimes (emb e.2).2 primeSupport = true := by
    simpa [DeGrey529.Param.primeSubplane, Bool.and_eq_true] using hends.2
  have hw1 := DeGrey529.TransferWf.emb_den_ne_zero e.1
  have hw2 := DeGrey529.TransferWf.emb_den_ne_zero e.2
  exact ⟨⟨hw1.1, hleft.1⟩, ⟨hw1.2, hleft.2⟩,
    ⟨hw2.1, hright.1⟩, ⟨hw2.2, hright.2⟩⟩

private theorem edge_terms_supported (T : QF3511TransferWf)
    (e : Nat × Nat) (he : e ∈ DeGrey529.edges.toList) :
    DeGrey529.Support.qfSupportedByPrimes (DeGrey529.Support.edgeDx e) primeSupport = true ∧
      DeGrey529.Support.qfSupportedByPrimes (DeGrey529.Support.edgeDy e) primeSupport = true ∧
      DeGrey529.Support.qfSupportedByPrimes
        (qmul (DeGrey529.Support.edgeDx e) (DeGrey529.Support.edgeDx e)) primeSupport = true ∧
      DeGrey529.Support.qfSupportedByPrimes
        (qmul (DeGrey529.Support.edgeDy e) (DeGrey529.Support.edgeDy e)) primeSupport = true ∧
      DeGrey529.Support.qfSupportedByPrimes (dist2 e.1 e.2) primeSupport = true := by
  have hbool :=
    (List.all_eq_true.mp T.hedge_terms3511) e he
  have hflat :
      (((DeGrey529.Support.qfSupportedByPrimes (DeGrey529.Support.edgeDx e) primeSupport = true ∧
          DeGrey529.Support.qfSupportedByPrimes (DeGrey529.Support.edgeDy e) primeSupport = true) ∧
          DeGrey529.Support.qfSupportedByPrimes
            (qmul (DeGrey529.Support.edgeDx e) (DeGrey529.Support.edgeDx e)) primeSupport = true) ∧
          DeGrey529.Support.qfSupportedByPrimes
            (qmul (DeGrey529.Support.edgeDy e) (DeGrey529.Support.edgeDy e)) primeSupport = true) ∧
          DeGrey529.Support.qfSupportedByPrimes (dist2 e.1 e.2) primeSupport = true := by
    simpa [DeGrey529.Support.edgeDistanceTermsSupportedByPrimes] using hbool
  exact ⟨hflat.1.1.1.1, hflat.1.1.1.2, hflat.1.1.2, hflat.1.2, hflat.2⟩

/-- Geometry transfer using only the checked current `{3,5,11}` QF fragment. -/
theorem geom_transfer_3511
    (e : Nat × Nat) (he : e ∈ DeGrey529.edges.toList) :
    T.unit (T.embF e.1) (T.embF e.2) := by
  have hgeom : unitFP (emb e.1) (emb e.2) :=
    geom_all_edges_unitFP e he
  unfold unitFP at hgeom
  unfold unit embF
  have hc := T.edge_endpoint_qf3511 e he
  have h1x : qf3511Wf (emb e.1).1 := hc.1
  have h1y : qf3511Wf (emb e.1).2 := hc.2.1
  have h2x : qf3511Wf (emb e.2).1 := hc.2.2.1
  have h2y : qf3511Wf (emb e.2).2 := hc.2.2.2
  have ht := T.edge_terms_supported e he
  have hdx : qf3511Wf (qsub (emb e.1).1 (emb e.2).1) := by
    refine ⟨DeGrey529.TransferWf.qfWf_qsub h1x.1 h2x.1, ?_⟩
    simpa [DeGrey529.Support.edgeDx, emb] using ht.1
  have hdy : qf3511Wf (qsub (emb e.1).2 (emb e.2).2) := by
    refine ⟨DeGrey529.TransferWf.qfWf_qsub h1y.1 h2y.1, ?_⟩
    simpa [DeGrey529.Support.edgeDy, emb] using ht.2.1
  have hdx2 : qf3511Wf
      (qmul (qsub (emb e.1).1 (emb e.2).1)
        (qsub (emb e.1).1 (emb e.2).1)) := by
    refine ⟨DeGrey529.TransferWf.qfWf_qmul hdx.1 hdx.1, ?_⟩
    simpa [DeGrey529.Support.edgeDx, emb] using ht.2.2.1
  have hdy2 : qf3511Wf
      (qmul (qsub (emb e.1).2 (emb e.2).2)
        (qsub (emb e.1).2 (emb e.2).2)) := by
    refine ⟨DeGrey529.TransferWf.qfWf_qmul hdy.1 hdy.1, ?_⟩
    simpa [DeGrey529.Support.edgeDy, emb] using ht.2.2.2.1
  have hdist : qf3511Wf
      (qadd
        (qmul (qsub (emb e.1).1 (emb e.2).1)
          (qsub (emb e.1).1 (emb e.2).1))
        (qmul (qsub (emb e.1).2 (emb e.2).2)
          (qsub (emb e.1).2 (emb e.2).2))) := by
    refine ⟨DeGrey529.TransferWf.qfWf_qadd hdx2.1 hdy2.1, ?_⟩
    simpa [dist2, emb] using ht.2.2.2.2
  rw [← T.hsub _ _ h1x h2x, ← T.hsub _ _ h1y h2y,
      ← T.hmul _ _ hdx hdx, ← T.hmul _ _ hdy hdy,
      ← T.hadd _ _ hdx2 hdy2]
  exact T.hunit _ hdist hgeom

/-- The current G529 LRAT obstruction transfers through any target receiving the checked
`{3,5,11}` QF fragment. -/
theorem chi_ge_5_current_embedding :
    ¬ Nonempty (PlaneColouring (T.F × T.F) T.unit 4) := by
  rintro ⟨κ, hκ⟩
  apply DeGrey529.Closed.not_VColourable
  refine ⟨fun v => κ (T.embF v), ?_⟩
  intro e he
  exact hκ (T.embF e.1) (T.embF e.2) (T.geom_transfer_3511 e he)

end QF3511TransferWf

/-- Any existing guarded full QF transfer restricts to the checked current `{3,5,11}` fragment. -/
def ofQFTransferWf (T : DeGrey529.TransferWf.QFTransferWf) : QF3511TransferWf where
  F := T.F
  add := T.add
  mul := T.mul
  sub := T.sub
  phi := T.phi
  isUnitVal := T.isUnitVal
  hadd := fun a b ha hb => T.hadd a b ha.1 hb.1
  hmul := fun a b ha hb => T.hmul a b ha.1 hb.1
  hsub := fun a b ha hb => T.hsub a b ha.1 hb.1
  hunit := fun d hd hone => T.hunit d hd.1 hone
  hcoord_edge_endpoints3511 := by
    simpa [primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_contains_all_edge_endpoints
  hedge_terms3511 := by
    simpa [primeSupport] using
      DeGrey529.Param.current_g529_edge_distances_supported_in_3511
  hcurrent_lrat_support := by
    simpa [primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_carries_current_lrat_obstruction_support

/-- The proved fraction homomorphism from any `RootedField` restricted to the current
`{3,5,11}` G529 fragment. -/
def rootedTransfer3511 (R : SounioSqrt.RootedField) : QF3511TransferWf :=
  ofQFTransferWf (DeGrey529.TransferWf.rootedTransfer R)

/-- One-line theorem form of the current-fragment transfer. -/
theorem qf3511Transfer_chi_ge_5_current_embedding (T : QF3511TransferWf) :
    ¬ Nonempty (PlaneColouring (T.F × T.F) T.unit 4) :=
  T.chi_ge_5_current_embedding

/-- Rooted-field corollary of the current `{3,5,11}` fragment transfer. The target still uses the
repository's existing four-root `RootedField` interface; the support boundary is the checked
current embedding and LRAT obstruction. -/
theorem rootedField_chi_ge_5_current_3511 (R : SounioSqrt.RootedField) :
    ¬ Nonempty (PlaneColouring
      ((rootedTransfer3511 R).F × (rootedTransfer3511 R).F)
      (rootedTransfer3511 R).unit 4) :=
  qf3511Transfer_chi_ge_5_current_embedding (rootedTransfer3511 R)

/-- Review-facing package for the current `{3,5,11}` transfer boundary. -/
structure QF3511TransferCurrentEmbeddingCertificate where
  transfer_no_four_colouring :
    ∀ T : QF3511TransferWf,
      ¬ Nonempty (PlaneColouring (T.F × T.F) T.unit 4)
  endpoint_support :
    DeGrey529.Param.edgeEndpointsInPrimeSubplane primeSupport = true
  edge_distance_support :
    DeGrey529.edges.toList.all (fun e =>
      DeGrey529.Support.edgeDistanceTermsSupportedByPrimes e primeSupport) = true
  full_current_lrat_support :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane primeSupport
  no_proper_current_lrat_support :
    ∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps

/-- Single object exposing the current `{3,5,11}` QF-fragment transfer theorem and its support
audit. -/
def qf3511TransferCurrentEmbeddingCertificate :
    QF3511TransferCurrentEmbeddingCertificate where
  transfer_no_four_colouring := qf3511Transfer_chi_ge_5_current_embedding
  endpoint_support := by
    simpa [primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_contains_all_edge_endpoints
  edge_distance_support := by
    simpa [primeSupport] using
      DeGrey529.Param.current_g529_edge_distances_supported_in_3511
  full_current_lrat_support := by
    simpa [primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_carries_current_lrat_obstruction_support
  no_proper_current_lrat_support :=
    DeGrey529.Param.no_proper_prime_subsupport_carries_current_g529_lrat_obstruction

#check @QF3511TransferWf
#check @qf3511Transfer_chi_ge_5_current_embedding
#check @rootedTransfer3511
#check @rootedField_chi_ge_5_current_3511
#check @QF3511TransferCurrentEmbeddingCertificate
#check @qf3511TransferCurrentEmbeddingCertificate

#print axioms qf3511Transfer_chi_ge_5_current_embedding
#print axioms rootedField_chi_ge_5_current_3511
#print axioms qf3511TransferCurrentEmbeddingCertificate

#eval IO.println "SounioDeGreyChi5Transfer3511: current G529 obstruction transfers through any target receiving the checked {3,5,11} QF fragment; support/minimality boundary is scoped to the current embedding."

end DeGrey529.Transfer3511
