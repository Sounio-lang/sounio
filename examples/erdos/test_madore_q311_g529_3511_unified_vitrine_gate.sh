#!/usr/bin/env bash
# Gate the unified Q311 + current {3,5,11} G529 vitrine surface.
#
# This is a packaging gate, not a new theorem:
#   * Q311/Madore closes the normalized real-plane 4-colour lower bound.
#   * The current G529 embedding transfers through the checked {3,5,11} fragment.
#   * The {3,5,11} minimal-support statement is scoped to the current embedding/LRAT support.
#   * Q311 is kept as a proper-support boundary, not as a G529 obstruction support.
#   * The chi>=6 surface is represented only by interface smokes, not a Euclidean witness.
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

LEAN_FILES=(
  "$LEAN_DIR/SounioRealPlaneGeometry.lean"
  "$LEAN_DIR/SounioMultiquadParam.lean"
  "$LEAN_DIR/SounioMoserSpindleQ311.lean"
  "$LEAN_DIR/SounioMoserSpindleQ311Real.lean"
  "$LEAN_DIR/SounioMoserSpindleQ311EuclideanGeometry.lean"
  "$LEAN_DIR/MadoreSpindleVitrine.lean"
  "$LEAN_DIR/SounioDeGreyChi5Param.lean"
  "$LEAN_DIR/SounioDeGreyChi5Transfer3511.lean"
  "$LEAN_DIR/SounioDeGreyChi5Rooted3511.lean"
  "$LEAN_DIR/SounioDeGreyChi5Eval3511.lean"
  "$LEAN_DIR/DeGreyChi5Vitrine.lean"
  "$LEAN_DIR/ErdosVitrine.lean"
)

PROOF_HOLE_RE='^\s*(sorry|admit)\b|:= by\s*(sorry|admit)\b|by\s+(sorry|admit)\b|#exit'

if rg -n "$PROOF_HOLE_RE" "${LEAN_FILES[@]}"; then
  echo "error: unified Q311/G529 Lean surface contains an incomplete proof marker" >&2
  exit 1
fi

rg -q 'It introduces no new mathematics' "$LEAN_DIR/MadoreSpindleVitrine.lean"
rg -q 'It introduces no new mathematics' "$LEAN_DIR/DeGreyChi5Vitrine.lean"
rg -q 'It is not a universal minimality theorem for every possible embedding' \
  "$LEAN_DIR/SounioDeGreyChi5Param.lean"
rg -q 'not this current G529 LRAT support' "$LEAN_DIR/DeGreyChi5Vitrine.lean"
rg -q 'standardRealPlaneUnit' "$LEAN_DIR/MadoreSpindleVitrine.lean"
rg -q 'no Euclidean no-five lower-bound witness is' "$LEAN_DIR/ErdosVitrine.lean"

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" build \
    SounioRealPlaneGeometry \
    SounioMultiquadParam \
    SounioMoserSpindleQ311 \
    SounioMoserSpindleQ311Real \
    SounioMoserSpindleQ311EuclideanGeometry \
    MadoreSpindleVitrine \
    SounioDeGreyChi5Param \
    SounioDeGreyChi5Transfer3511 \
    SounioDeGreyChi5Rooted3511 \
    SounioDeGreyChi5Eval3511 \
    DeGreyChi5Vitrine \
    ErdosVitrine
)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

cat > "$WORK/check_madore_q311_g529_3511_unified_vitrine.lean" <<'EOF'
import ErdosVitrine

open UnitDistanceChromatic

noncomputable section

#check SounioSqrt.RealCauchyField.Multiquad.indep_3_11
#check MoserSpindleQ311.spindle_not_3_colourable
#check MoserSpindleQ311.q311_plane_needs_4_colours
#check MoserSpindleQ311.RealPlane.phi311Param_eq_phi311
#check MoserSpindleQ311.RealPlane.evalS_3_11_coeff311_eq_phi311
#check MoserSpindleQ311.RealPlane.phi311Param_injective
#check MoserSpindleQ311.RealPlane.chi_R2_ge_4_param_unit
#check MadoreSpindle.Showcase.unitReal1_iff_standardRealPlaneDist2
#check MoserSpindleQ311.RealPlane.chi_R2_ge_4_param_standard_unit
#check MoserSpindleQ311.RealPlane.madoreQ311ParametricBaseCaseCertificate
#check MadoreSpindle.Showcase.madoreVitrineUnifiedPathsCertificate
#check MadoreSpindle.Showcase.madore_q311_parametric_base_case_standard_unit

#check DeGrey529.Transfer3511.QF3511TransferWf
#check DeGrey529.Transfer3511.qf3511Transfer_chi_ge_5_current_embedding
#check DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate
#check DeGrey529.Rooted3511.RootedField3511
#check DeGrey529.Rooted3511.rootedField3511_chi_ge_5_current_embedding
#check DeGrey529.Rooted3511.rootedField3511CurrentEmbeddingCertificate
#check DeGrey529.Rooted3511.RootedField3511.evalNum8_qmulNum3511
#check DeGrey529.Rooted3511.RootedField3511.evalNum3511_qmul_num_bridge
#check DeGrey529.Rooted3511.RootedField3511.evalNum3511_qmul
#check DeGrey529.Rooted3511.RootedField3511.phi3511_qmul
#check DeGrey529.Rooted3511.RootedField3511.phi3511_qadd
#check DeGrey529.Rooted3511.RootedField3511.phi3511_qsub
#check DeGrey529.Rooted3511.RootedField3511.phi3511_unit
#check DeGrey529.Rooted3511.RootedField3511.toDerivedQF3511TransferWf
#check DeGrey529.Rooted3511.RootedField3511.derived_phi3511_chi_ge_5_current_embedding
#check DeGrey529.Rooted3511.RootedField3511.phi3511DerivedTransferCertificate

#check DeGrey529.Showcase.qf3511_transfer_current_embedding_chi_ge_5
#check DeGrey529.Showcase.rootedField3511_transfer_current_embedding_chi_ge_5
#check DeGrey529.Showcase.rootedField3511DerivedPhiTransferWf
#check DeGrey529.Showcase.rootedField3511DerivedPhiTransferCertificate
#check DeGrey529.Showcase.rootedField3511_derived_phi_transfer_current_embedding_chi_ge_5

#check DeGrey529.Param.currentEmbeddingG529MinimalPrimeSupportCertificate
#check DeGrey529.Param.current_g529_3511_is_exact_lrat_obstruction_support
#check DeGrey529.Param.prime_subplane_3511_needs_5_colours_via_g529_lrat
#check DeGrey529.Param.prime_subplane_3511_is_exact_current_obstruction_surface
#check DeGrey529.Showcase.q311_base_obstruction_and_not_current_g529_lrat_support
#check DeGrey529.Showcase.q311_standard_real_base_and_not_current_g529_lrat_support
#check DeGrey529.Showcase.current_embedding_3511_minimality_scope_boundary
#check DeGrey529.Showcase.current_embedding_3511_minimality_with_q311_real_boundary
#check DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate

#check Erdos.Showcase.Q311G5293511UnifiedVitrineCertificate
#check Erdos.Showcase.q311G5293511UnifiedVitrineCertificate
#check Erdos.Showcase.Chi6InterfaceSmokeBoundary
#check Erdos.Showcase.chi6InterfaceSmokeBoundary
#check Erdos.Showcase.q311_standard_real_lower_bound
#check Erdos.Showcase.current_embedding_3511_minimality_with_q311_boundary
#check Erdos.Showcase.chi6_frontier_smokes_available

example :
    ¬ Nonempty (PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      SounioSqrt.RealCauchyField.standardRealPlaneUnit 3) :=
  MadoreSpindle.Showcase.madoreVitrineUnifiedPathsCertificate.parametric_standard_unit_path

example :
    MadoreSpindle.Showcase.MadoreVitrineUnifiedPathsCertificate :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate.madore_q311_vitrine

example :
    ¬ Nonempty (PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      SounioSqrt.RealCauchyField.standardRealPlaneUnit 3) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.madore_q311_vitrine |>.parametric_standard_unit_path

example :
    (¬ Nonempty (PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      SounioSqrt.RealCauchyField.standardRealPlaneUnit 3)) ∧
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11] :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.q311_standard_real_base_and_g529_boundary

example :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.exact_lrat_obstruction_support |>.1

example :
    ∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.no_proper_lrat_obstruction_support

example :
    ¬ Nonempty (PlaneColouring
      (DeGrey529.Param.FieldPointPrimeSupport [3, 5, 11])
      DeGrey529.Param.primeUnit3511 4) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.explicit_prime_support_subplane_obstruction

example :
    DeGrey529.Transfer3511.QF3511TransferCurrentEmbeddingCertificate :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.qf3511_transfer_current_embedding

example :
    DeGrey529.Rooted3511.RootedField3511CurrentEmbeddingCertificate :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField3511_transfer_current_embedding

example :
    ∀ R : DeGrey529.Rooted3511.RootedField3511, R.IntCastNonzero →
      R.Phi3511AddSubUnitCertificate :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField3511_eval_add_sub_unit

example :
    ∀ R : DeGrey529.Rooted3511.RootedField3511, ∀ hden : R.IntCastNonzero,
      R.Phi3511DerivedTransferCertificate hden :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField3511_derived_phi_transfer_certificate

example :
    ∀ R : DeGrey529.Rooted3511.RootedField3511, ∀ hden : R.IntCastNonzero,
      ¬ Nonempty (PlaneColouring
        (R.F × R.F) (DeGrey529.Showcase.rootedField3511DerivedPhiTransferWf R hden).unit 4) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField3511_derived_phi_transfer

example :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
    (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) ∧
    DeGrey529.Support.properPrimeSubsupport3511 [3, 11] ∧
    (¬ Nonempty (PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      SounioSqrt.RealCauchyField.standardRealPlaneUnit 3)) ∧
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11] :=
  DeGrey529.Showcase.current_embedding_3511_minimality_with_q311_real_boundary

example :
    Erdos.Showcase.Q311G5293511UnifiedVitrineCertificate :=
  Erdos.Showcase.q311G5293511UnifiedVitrineCertificate

example :
    ¬ Nonempty (PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      SounioSqrt.RealCauchyField.standardRealPlaneUnit 3) :=
  Erdos.Showcase.q311G5293511UnifiedVitrineCertificate.q311_standard_real_lower_bound

example :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 5, 11] ∧
    (∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps) ∧
    DeGrey529.Support.properPrimeSubsupport3511 [3, 11] ∧
    (¬ Nonempty (PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      SounioSqrt.RealCauchyField.standardRealPlaneUnit 3)) ∧
    ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane [3, 11] :=
  Erdos.Showcase.q311G5293511UnifiedVitrineCertificate
    |>.current_embedding_3511_minimality_with_q311_boundary

example :
    Nonempty (NatEdgeUnitDistanceCertificate.NoFiveColourWitness
        6 Nat UnitDistanceChromatic.Smoke.k6Unit) ∧
    (∃ G : EuclideanNatEdgeExactGeometry
        4 (Fin 4) UnitDistanceChromatic.Chi6EuclideanGeometrySmoke.squareUnit,
      G.exact.edges = [(0, 1), (1, 2), (2, 3), (3, 0)]) :=
  Erdos.Showcase.chi6_frontier_smokes_available

end

#print axioms MadoreSpindle.Showcase.madore_q311_parametric_base_case_standard_unit
#print axioms MadoreSpindle.Showcase.madoreVitrineUnifiedPathsCertificate
#print axioms DeGrey529.Transfer3511.qf3511Transfer_chi_ge_5_current_embedding
#print axioms DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate
#print axioms DeGrey529.Rooted3511.rootedField3511_chi_ge_5_current_embedding
#print axioms DeGrey529.Rooted3511.rootedField3511CurrentEmbeddingCertificate
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511_qmul
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511_qadd
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511_qsub
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511_unit
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511DerivedTransferCertificate
#print axioms DeGrey529.Showcase.rootedField3511_derived_phi_transfer_current_embedding_chi_ge_5
#print axioms DeGrey529.Showcase.current_embedding_3511_minimality_scope_boundary
#print axioms DeGrey529.Showcase.current_embedding_3511_minimality_with_q311_real_boundary
#print axioms DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
#print axioms Erdos.Showcase.q311G5293511UnifiedVitrineCertificate
#print axioms Erdos.Showcase.chi6InterfaceSmokeBoundary
#print axioms Erdos.Showcase.q311_standard_real_lower_bound
#print axioms Erdos.Showcase.current_embedding_3511_minimality_with_q311_boundary
#print axioms Erdos.Showcase.chi6_frontier_smokes_available
EOF

if rg -n "$PROOF_HOLE_RE" "$WORK/check_madore_q311_g529_3511_unified_vitrine.lean"; then
  echo "error: generated unified Q311/G529 verifier contains an incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/check_madore_q311_g529_3511_unified_vitrine.lean" \
    > "$WORK/check.out" 2> "$WORK/check.err"
)

if rg -q 'error:' "$WORK/check.out" "$WORK/check.err"; then
  cat "$WORK/check.out" "$WORK/check.err" >&2
  exit 1
fi
if rg -q 'sorryAx' "$WORK/check.out" "$WORK/check.err"; then
  cat "$WORK/check.out" "$WORK/check.err" >&2
  echo "error: unified Q311/G529 verifier reports sorryAx" >&2
  exit 1
fi

rg -q 'MadoreSpindle\.Showcase\.madore_q311_parametric_base_case_standard_unit' \
  "$WORK/check.out"
rg -q 'MadoreSpindle\.Showcase\.madoreVitrineUnifiedPathsCertificate' \
  "$WORK/check.out"
rg -q 'DeGrey529\.Showcase\.qf3511_transfer_current_embedding_chi_ge_5' \
  "$WORK/check.out"
rg -q 'DeGrey529\.Showcase\.rootedField3511_transfer_current_embedding_chi_ge_5' \
  "$WORK/check.out"
rg -q 'DeGrey529\.Showcase\.rootedField3511_derived_phi_transfer_current_embedding_chi_ge_5' \
  "$WORK/check.out"
rg -q 'DeGrey529\.Showcase\.current_embedding_3511_minimality_with_q311_real_boundary' \
  "$WORK/check.out"
rg -q 'DeGrey529\.Showcase\.scopedG5293511MinimalityShowcaseCertificate' \
  "$WORK/check.out"
rg -q 'Erdos\.Showcase\.q311G5293511UnifiedVitrineCertificate' \
  "$WORK/check.out"
rg -q 'Erdos\.Showcase\.chi6InterfaceSmokeBoundary' \
  "$WORK/check.out"
rg -q 'Erdos\.Showcase\.chi6_frontier_smokes_available' \
  "$WORK/check.out"

echo "madore_q311_g529_3511_unified_vitrine_gate: PASS"
