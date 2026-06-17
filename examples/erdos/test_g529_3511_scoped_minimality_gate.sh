#!/usr/bin/env bash
# Gate the scoped {3,5,11} minimal-support certificate for the current G529 embedding.
#
# This deliberately verifies the scoped theorem surface only:
#   * the current coordinate table and edge-distance terms live in {3,5,11},
#   * every proper prime sub-support of {3,5,11} is rejected for the current embedding,
#   * the full current subplane carries the reflected G529 no-4-colouring obstruction,
#   * the claim boundary says this is not universal minimality over all possible embeddings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN_DIR="$ROOT/formal/lean4"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi

[[ -x "$LOCK" ]] || { echo "error: missing build lock helper: $LOCK" >&2; exit 1; }
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

PARAM="$LEAN_DIR/SounioDeGreyChi5Param.lean"
VITRINE="$LEAN_DIR/DeGreyChi5Vitrine.lean"
PROOF_HOLE_RE='^\s*(sorry|admit)\b|:= by\s*(sorry|admit)\b|by\s+(sorry|admit)\b|#exit'

if rg -n "$PROOF_HOLE_RE" "$PARAM" "$VITRINE"; then
  echo "error: G529 scoped-minimality Lean surface contains an incomplete proof marker" >&2
  exit 1
fi

rg -q 'It is not a universal minimality theorem for every possible embedding' "$PARAM"
rg -q 'not to every possible embedding of G529' "$PARAM"

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" build SounioDeGreyChi5Param DeGreyChi5Vitrine
)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

cat > "$WORK/check_g529_3511_scoped_minimality.lean" <<'EOF'
import DeGreyChi5Vitrine

open UnitDistanceChromatic

noncomputable section

#check (DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate :
  DeGrey529.Param.CurrentEmbeddingG529MinimalPrimeSupportCertificate)
#check (DeGrey529.Param.current_g529_3511_only_full_prime_support_and_subplanes)
#check (DeGrey529.Param.current_g529_3511_is_minimal_coordinate_and_edge_support)
#check (DeGrey529.Param.current_g529_3511_is_minimal_obstructing_subplane)
#check (DeGrey529.Param.current_g529_3511_is_exact_lrat_obstruction_support)
#check (DeGrey529.Param.current_g529_3511_concrete_proper_lrat_obstruction_subsupports_fail)
#check (DeGrey529.Param.prime_subplane_3511_needs_5_colours_via_g529_lrat)
#check (DeGrey529.Param.prime_subplane_3511_is_exact_current_obstruction_surface)
#check (DeGrey529.Param.q3511_plane_needs_5_colours)
#check (DeGrey529.Showcase.explicit_prime_subplane_3511_needs_5_colours)
#check (DeGrey529.Showcase.explicit_prime_subplane_3511_exact_current_obstruction_surface)
#check (DeGrey529.Showcase.current_embedding_3511_concrete_proper_lrat_obstruction_subsupports_fail)
#check (DeGrey529.Showcase.RootedFieldTransferCurrent3511BoundaryCertificate)
#check (DeGrey529.Showcase.rootedField_transfer_with_current_3511_support_boundary)
#check (DeGrey529.Showcase.rootedFieldTransferCurrent3511BoundaryCertificate)
#check (MadoreSpindle.Showcase.madoreVitrineUnifiedPathsCertificate :
  MadoreSpindle.Showcase.MadoreVitrineUnifiedPathsCertificate)
#check (DeGrey529.Showcase.q311_base_obstruction_and_not_current_g529_lrat_support :
  (¬ Nonempty (PlaneColouring
    (MoserSpindleQ311.Qf × MoserSpindleQ311.Qf)
    MoserSpindleQ311.unitFP 3)) ∧
  ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11])
#check (DeGrey529.Showcase.q311_standard_real_base_and_not_current_g529_lrat_support :
  (¬ Nonempty (PlaneColouring
    (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
    SounioSqrt.RealCauchyField.standardRealPlaneUnit 3)) ∧
  ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11])
#check (DeGrey529.Showcase.current_embedding_3511_minimality_scope_boundary :
  DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
  (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) ∧
  DeGrey529.Support.properPrimeSubsupport3511 [3, 11] ∧
    (¬ Nonempty (PlaneColouring
      (MoserSpindleQ311.Qf × MoserSpindleQ311.Qf)
      MoserSpindleQ311.unitFP 3)) ∧
  ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11])
#check (DeGrey529.Showcase.current_embedding_3511_minimality_with_q311_real_boundary :
  DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
  (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) ∧
  DeGrey529.Support.properPrimeSubsupport3511 [3, 11] ∧
  (¬ Nonempty (PlaneColouring
    (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
    SounioSqrt.RealCauchyField.standardRealPlaneUnit 3)) ∧
  ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11])

example :
    DeGrey529.Support.currentCoordinateTableSupportedByPrimes
        DeGrey529.Support.primeSupport3511 = true :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.coordinates_supported

example :
    ((List.range 8).filter (fun m =>
      DeGrey529.Support.currentCoordinateTableSupportedByPrimes
        (DeGrey529.Support.primeSubsupportFromBits m))).map
        DeGrey529.Support.primeSubsupportFromBits = [[3, 5, 11]] :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.only_full_coordinate_support

example :
    ∀ ps : List Nat,
      DeGrey529.Support.currentCoordinateSupportContainedIn ps ↔
        3 ∈ ps ∧ 5 ∈ ps ∧ 11 ∈ ps :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.coordinate_support_exact

example :
    ∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Support.currentCoordinateSupportContainedIn ps :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.no_proper_coordinate_support

example :
    DeGrey529.edges.toList.all (fun e =>
      DeGrey529.Support.edgeDistanceTermsSupportedByPrimes e
        DeGrey529.Support.primeSupport3511) = true :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.edge_distances_supported

example :
    DeGrey529.Support.edgeCntPrime 7 = 0 :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.sqrt7_not_touched

example :
    ((List.range 8).filter (fun m =>
      DeGrey529.Param.touchedVerticesInPrimeSubplane
        (DeGrey529.Support.primeSubsupportFromBits m))).map
        DeGrey529.Support.primeSubsupportFromBits = [[3, 5, 11]] :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.only_full_touched_vertex_subplane

example :
    ∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      DeGrey529.Param.touchedVerticesInPrimeSubplane ps = false :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.no_proper_touched_vertex_subplane

example :
    ((List.range 8).filter (fun m =>
      DeGrey529.Param.edgeEndpointsInPrimeSubplane
        (DeGrey529.Support.primeSubsupportFromBits m))).map
        DeGrey529.Support.primeSubsupportFromBits = [[3, 5, 11]] :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.only_full_endpoint_subplane

example :
    ∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      DeGrey529.Param.edgeEndpointsInPrimeSubplane ps = false :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.no_proper_endpoint_subplane

example :
    ∀ e ∈ DeGrey529.edges.toList,
      DeGrey529.Param.currentUnit3511
        (DeGrey529.Param.embCurrent e.1) (DeGrey529.Param.embCurrent e.2) :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.current_subplane_edges_unit

example :
    ¬ Nonempty (PlaneColouring
      DeGrey529.Param.FieldPoint3511Current DeGrey529.Param.currentUnit3511 4) :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate.current_subplane_obstruction

example :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
    (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) ∧
    ¬ Nonempty (PlaneColouring
      DeGrey529.Param.FieldPoint3511Current DeGrey529.Param.currentUnit3511 4) :=
  DeGrey529.Param.current_g529_3511_is_exact_lrat_obstruction_support

example :
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane []) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [5]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [11]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [5, 11]) :=
  DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate
    |>.concrete_proper_lrat_obstruction_subsupports_fail

example :
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane []) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [5]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [11]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11]) ∧
    (¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [5, 11]) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.concrete_proper_lrat_obstruction_subsupports_fail

example :
    ¬ Nonempty (PlaneColouring
      (DeGrey529.Param.FieldPointPrimeSupport [3, 5, 11])
      DeGrey529.Param.primeUnit3511 4) :=
  DeGrey529.Param.prime_subplane_3511_needs_5_colours_via_g529_lrat

example :
    ¬ Nonempty (PlaneColouring
      (DeGrey529.Param.FieldPointPrimeSupport [3, 5, 11])
      DeGrey529.Param.primeUnit3511 4) ∧
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
    (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) :=
  DeGrey529.Param.prime_subplane_3511_is_exact_current_obstruction_surface

example :
    ¬ Nonempty (PlaneColouring
      (DeGrey529.Param.FieldPointPrimeSupport [3, 5, 11])
      DeGrey529.Param.primeUnit3511 4) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.explicit_prime_support_subplane_obstruction

example :
    ¬ Nonempty (PlaneColouring
      (DeGrey529.Param.FieldPointPrimeSupport [3, 5, 11])
      DeGrey529.Param.primeUnit3511 4) ∧
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
    (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.explicit_prime_support_exact_obstruction_surface

example :
    ¬ Nonempty (PlaneColouring DeGrey529.Concrete.FieldPoint DeGrey529.Concrete.unitFP 4) :=
  DeGrey529.Param.q3511_plane_needs_5_colours

example :
    ¬ Nonempty (PlaneColouring
      (MoserSpindleQ311.Qf × MoserSpindleQ311.Qf)
      MoserSpindleQ311.unitFP 3) :=
  DeGrey529.Showcase.q311_base_obstruction_and_not_current_g529_lrat_support.1

example :
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11] :=
  DeGrey529.Showcase.q311_base_obstruction_and_not_current_g529_lrat_support.2

example :
    MadoreSpindle.Showcase.MadoreVitrineUnifiedPathsCertificate :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.madore_q311_vitrine

example :
    ¬ Nonempty (PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      SounioSqrt.RealCauchyField.standardRealPlaneUnit 3) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.q311_standard_real_base_and_g529_boundary |>.1

example :
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11] :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.q311_standard_real_base_and_g529_boundary |>.2

example :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
    (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) ∧
    DeGrey529.Support.properPrimeSubsupport3511 [3, 11] ∧
    (¬ Nonempty (PlaneColouring
      (MoserSpindleQ311.Qf × MoserSpindleQ311.Qf)
      MoserSpindleQ311.unitFP 3)) ∧
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11] :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.current_embedding_minimality_scope_boundary

example :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
    (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) ∧
    DeGrey529.Support.properPrimeSubsupport3511 [3, 11] ∧
    (¬ Nonempty (PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      SounioSqrt.RealCauchyField.standardRealPlaneUnit 3)) ∧
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11] :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.current_embedding_minimality_with_q311_real_boundary

example :
    ∀ F : SounioSqrt.RootedField,
      ¬ Nonempty (PlaneColouring
        ((DeGrey529.TransferWf.rootedTransfer F).F × (DeGrey529.TransferWf.rootedTransfer F).F)
        (DeGrey529.TransferWf.rootedTransfer F).unit 4) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField_transfer_current_support_boundary.rootedField_transfer

example :
    DeGrey529.Support.edgeCntPrime 7 = 0 :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField_transfer_current_support_boundary.current_embedding_sqrt7_free

example :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField_transfer_current_support_boundary.full_current_lrat_support

example :
    ∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField_transfer_current_support_boundary.no_proper_current_lrat_support

end

#print axioms DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate
#print axioms DeGrey529.Param.current_g529_3511_only_full_prime_support_and_subplanes
#print axioms DeGrey529.Param.current_g529_3511_is_minimal_coordinate_and_edge_support
#print axioms DeGrey529.Param.current_g529_3511_is_minimal_obstructing_subplane
#print axioms DeGrey529.Param.current_g529_3511_is_exact_lrat_obstruction_support
#print axioms DeGrey529.Param.current_g529_3511_concrete_proper_lrat_obstruction_subsupports_fail
#print axioms DeGrey529.Param.prime_subplane_3511_needs_5_colours_via_g529_lrat
#print axioms DeGrey529.Param.prime_subplane_3511_is_exact_current_obstruction_surface
#print axioms DeGrey529.Showcase.explicit_prime_subplane_3511_needs_5_colours
#print axioms DeGrey529.Showcase.explicit_prime_subplane_3511_exact_current_obstruction_surface
#print axioms DeGrey529.Showcase.rootedField_transfer_with_current_3511_support_boundary
#print axioms DeGrey529.Showcase.rootedFieldTransferCurrent3511BoundaryCertificate
#print axioms DeGrey529.Showcase.current_embedding_3511_concrete_proper_lrat_obstruction_subsupports_fail
#print axioms DeGrey529.Showcase.q311_base_obstruction_and_not_current_g529_lrat_support
#print axioms DeGrey529.Showcase.q311_standard_real_base_and_not_current_g529_lrat_support
#print axioms DeGrey529.Showcase.current_embedding_3511_minimality_scope_boundary
#print axioms DeGrey529.Showcase.current_embedding_3511_minimality_with_q311_real_boundary
EOF

if rg -n "$PROOF_HOLE_RE" "$WORK/check_g529_3511_scoped_minimality.lean"; then
  echo "error: generated G529 scoped-minimality verifier contains an incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/check_g529_3511_scoped_minimality.lean" \
    > "$WORK/check.out" 2> "$WORK/check.err"
)

if rg -q 'error:' "$WORK/check.out" "$WORK/check.err"; then
  cat "$WORK/check.out" "$WORK/check.err" >&2
  exit 1
fi
if rg -q 'sorryAx' "$WORK/check.out" "$WORK/check.err"; then
  cat "$WORK/check.out" "$WORK/check.err" >&2
  echo "error: G529 scoped-minimality verifier reports sorryAx" >&2
  exit 1
fi

rg -q '^DeGrey529\.Param\.currentEmbeddingG529MinimalPrimeSupportCertificate : DeGrey529\.Param\.CurrentEmbeddingG529MinimalPrimeSupportCertificate$' \
  "$WORK/check.out"
rg -q '^DeGrey529\.Param\.current_g529_3511_only_full_prime_support_and_subplanes' "$WORK/check.out"
rg -q '^DeGrey529\.Param\.current_g529_3511_is_minimal_coordinate_and_edge_support' "$WORK/check.out"
rg -q '^DeGrey529\.Param\.current_g529_3511_is_minimal_obstructing_subplane' "$WORK/check.out"
rg -q '^DeGrey529\.Param\.current_g529_3511_is_exact_lrat_obstruction_support' "$WORK/check.out"
rg -q '^DeGrey529\.Param\.current_g529_3511_concrete_proper_lrat_obstruction_subsupports_fail' "$WORK/check.out"
rg -q '^DeGrey529\.Param\.prime_subplane_3511_needs_5_colours_via_g529_lrat' "$WORK/check.out"
rg -q '^DeGrey529\.Param\.prime_subplane_3511_is_exact_current_obstruction_surface' "$WORK/check.out"
rg -q '^DeGrey529\.Param\.q3511_plane_needs_5_colours' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.explicit_prime_subplane_3511_needs_5_colours' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.explicit_prime_subplane_3511_exact_current_obstruction_surface' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.rootedField_transfer_with_current_3511_support_boundary' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.rootedFieldTransferCurrent3511BoundaryCertificate' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.current_embedding_3511_concrete_proper_lrat_obstruction_subsupports_fail' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.q311_base_obstruction_and_not_current_g529_lrat_support' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.q311_standard_real_base_and_not_current_g529_lrat_support' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.current_embedding_3511_minimality_scope_boundary' "$WORK/check.out"
rg -q '^DeGrey529\.Showcase\.current_embedding_3511_minimality_with_q311_real_boundary' "$WORK/check.out"

echo "g529_3511_scoped_minimality_gate: PASS"
