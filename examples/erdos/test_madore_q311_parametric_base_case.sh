#!/usr/bin/env bash
# Gate the Madore/Moser Q311 parametric base case end to end.
#
# This checks the actual Lean citation surface for the goal:
#   * the exact Q(sqrt3,sqrt11) spindle is not 3-colourable,
#   * the parametric evaluator `evalS [3,11]` replicates the legacy `phi311`,
#   * the real-plane theorem is exposed at normalized unit scale (`dist^2 = 1`),
#   * the vitrine packages legacy, parametric, and Euclidean-geometry paths together.
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
)

PROOF_HOLE_RE='^\s*(sorry|admit)\b|:= by\s*(sorry|admit)\b|by\s+(sorry|admit)\b|#exit'

if rg -n "$PROOF_HOLE_RE" "${LEAN_FILES[@]}"; then
  echo "error: Madore/Q311 Lean surface contains an incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" build \
    SounioRealPlaneGeometry \
    SounioMultiquadParam \
    SounioMoserSpindleQ311 \
    SounioMoserSpindleQ311Real \
    SounioMoserSpindleQ311EuclideanGeometry \
    MadoreSpindleVitrine
)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

cat > "$WORK/check_madore_q311_parametric_base_case.lean" <<'EOF'
import MadoreSpindleVitrine

open UnitDistanceChromatic
open MoserSpindleQ311
open MoserSpindleQ311.RealPlane
open MadoreSpindle.Showcase
open SounioSqrt.RealCauchyField
open SounioSqrt.RealCauchyField.Multiquad

noncomputable section

#check (SounioSqrt.RealCauchyField.Multiquad.indep_3_11 : IndepMultiquad [3, 11])
#check (spindle_not_3_colourable : ¬ Graph.Colourable spindleG 3)
#check (q311_plane_needs_4_colours : ¬ Nonempty (PlaneColouring (Qf × Qf) unitFP 3))
#check (phi311Param_eq_phi311 : ∀ q : Qf, phi311Param q = phi311 q)
#check (evalS_3_11_coeff311_eq_phi311 : ∀ q : Qf, evalS [3, 11] (coeff311 q) = phi311 q)
#check (phi311Param_injective : Function.Injective phi311Param)
#check (chi_R2_ge_4_param_unit : ¬ Nonempty (PlaneColouring (Real × Real) unitReal1 3))
#check (MadoreSpindle.Showcase.unitReal1_iff_standardRealPlaneDist2 :
  ∀ p q : Real × Real,
    unitReal1 p q ↔ standardRealPlaneDist2 p q = qR (1 : Rat))
#check (chi_R2_ge_4_param_standard_unit :
  ¬ Nonempty (PlaneColouring (Real × Real) standardRealPlaneUnit 3))
#check (madoreQ311ParametricBaseCaseCertificate : MadoreQ311ParametricBaseCaseCertificate)
#check (madoreVitrineUnifiedPathsCertificate : MadoreVitrineUnifiedPathsCertificate)
#check (MadoreSpindle.Showcase.madore_q311_parametric_base_case_standard_unit :
  ¬ Nonempty (PlaneColouring (Real × Real) standardRealPlaneUnit 3))

example :
    HasRadicals [3, 11] :=
  madoreQ311ParametricBaseCaseCertificate.evalS_genericity.support

example :
    MultiquadField [3, 11] :=
  madoreQ311ParametricBaseCaseCertificate.evalS_genericity.field_package

example :
    TowerHasInverses [3, 11] :=
  madoreQ311ParametricBaseCaseCertificate.evalS_genericity.inverses

example :
    IndepMultiquad [3, 11] :=
  madoreQ311ParametricBaseCaseCertificate.evalS_genericity.independence

example :
    (∀ q : Qf, phi311Param q = phi311 q) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.evaluator_eq

example :
    (∀ q : Qf, evalS [3, 11] (coeff311 q) = phi311 q) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.evalS_eq_legacy

example :
    Function.Injective phi311Param :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.parametric_injective

example :
    (∀ i : Fin 7, embRealParam i = embReal i) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.embedding_eq

example :
    (∀ x y : Qf, phi311Param (qadd x y) = addR (phi311Param x) (phi311Param y)) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.add_compat

example :
    (∀ x : Qf, phi311Param (qneg x) = negR (phi311Param x)) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.neg_compat

example :
    (∀ x y : Qf, phi311Param (qsub x y) = addR (phi311Param x) (negR (phi311Param y))) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.sub_compat

example :
    (∀ x y : Qf, phi311Param (qmul x y) = mulR (phi311Param x) (phi311Param y)) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.mul_compat

example :
    (∀ e : Fin 7 × Fin 7, e ∈ spindleG.edges →
      standardRealPlaneUnit (embRealParamUnit e.1) (embRealParamUnit e.2)) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.parametric_standard_edge_units

example :
    ¬ Nonempty (PlaneColouring (Real × Real) standardRealPlaneUnit 3) :=
  madoreQ311ParametricBaseCaseCertificate.legacy_parametric_replication.parametric_standard_lower_bound

example :
    ¬ Nonempty (PlaneColouring (Real × Real) unitReal144 3) :=
  madoreQ311ParametricBaseCaseCertificate.real_parametric_native_lower_bound

example :
    ¬ Nonempty (PlaneColouring (Real × Real) unitReal1 3) :=
  madoreQ311ParametricBaseCaseCertificate.real_parametric_unit_lower_bound

example :
    ¬ Nonempty (PlaneColouring (Real × Real) unitReal144 3) :=
  madoreVitrineUnifiedPathsCertificate.legacy_native_path

example :
    ¬ Nonempty (PlaneColouring (Real × Real) unitReal144 3) :=
  madoreVitrineUnifiedPathsCertificate.parametric_native_path

example :
    ¬ Nonempty (PlaneColouring (Real × Real) unitReal1 3) :=
  madoreVitrineUnifiedPathsCertificate.parametric_unit_path

example :
    ∀ p q : Real × Real,
      unitReal1 p q ↔ standardRealPlaneDist2 p q = qR (1 : Rat) :=
  madoreVitrineUnifiedPathsCertificate.parametric_unit_uses_standard_dist2

example :
    Function.Injective phi311Param :=
  madoreVitrineUnifiedPathsCertificate.parametric_evaluator_injective

example :
    ¬ Nonempty (PlaneColouring (Real × Real) standardRealPlaneUnit 3) :=
  madoreVitrineUnifiedPathsCertificate.parametric_standard_unit_path

example :
    ¬ Nonempty (PlaneColouring (Real × Real) standardRealPlaneUnit 3) :=
  MadoreSpindle.Showcase.madore_q311_parametric_base_case_standard_unit

example :
    EuclideanNatEdgeExactGeometry 7 (Fin 7) moserQ311Unit :=
  madoreVitrineUnifiedPathsCertificate.normalized_euclidean_geometry

example :
    MoserQ311Dist2ZeroSeparatesPoints :=
  madoreVitrineUnifiedPathsCertificate.zero_distance_separation

end

#print axioms chi_R2_ge_4_param_unit
#print axioms chi_R2_ge_4_param_standard_unit
#print axioms chi_R2_ge_4_param_unit_via_witness
#print axioms chi_R2_ge_4_param_standard_unit_via_witness
#print axioms MadoreSpindle.Showcase.unitReal1_iff_standardRealPlaneDist2
#print axioms phi311Param_injective
#print axioms parametricReplicatesLegacyCertificate
#print axioms madoreQ311ParametricBaseCaseCertificate
#print axioms MadoreSpindle.Showcase.madore_q311_parametric_base_case_standard_unit
#print axioms MadoreSpindle.Showcase.madoreVitrineUnifiedPathsCertificate
EOF

if rg -n "$PROOF_HOLE_RE" "$WORK/check_madore_q311_parametric_base_case.lean"; then
  echo "error: generated Madore/Q311 verifier contains an incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/check_madore_q311_parametric_base_case.lean" \
    > "$WORK/check.out" 2> "$WORK/check.err"
)

if rg -q 'error:' "$WORK/check.out" "$WORK/check.err"; then
  cat "$WORK/check.out" "$WORK/check.err" >&2
  exit 1
fi
if rg -q 'sorryAx' "$WORK/check.out" "$WORK/check.err"; then
  cat "$WORK/check.out" "$WORK/check.err" >&2
  echo "error: Madore/Q311 verifier reports sorryAx" >&2
  exit 1
fi

rg -q '^indep_3_11 : IndepMultiquad \[3, 11\]$' "$WORK/check.out"
rg -q '^spindle_not_3_colourable : ¬spindleG\.Colourable 3$' "$WORK/check.out"
rg -q '^q311_plane_needs_4_colours : ¬Nonempty \(PlaneColouring \(Qf × Qf\) unitFP 3\)$' \
  "$WORK/check.out"
rg -q '^phi311Param_eq_phi311 : ∀ \(q : Qf\), phi311Param q = phi311 q$' "$WORK/check.out"
rg -q '^evalS_3_11_coeff311_eq_phi311 : ∀ \(q : Qf\), evalS \[3, 11\] \(coeff311 q\) = phi311 q$' \
  "$WORK/check.out"
rg -q '^phi311Param_injective : Function.Injective phi311Param$' "$WORK/check.out"
rg -q '^chi_R2_ge_4_param_unit : ¬Nonempty \(PlaneColouring \(Real × Real\) unitReal1 3\)$' \
  "$WORK/check.out"
rg -q '^MadoreSpindle\.Showcase\.unitReal1_iff_standardRealPlaneDist2 : ∀ \(p q : Real × Real\),$' \
  "$WORK/check.out"
rg -q '^  unitReal1 p q ↔ standardRealPlaneDist2 p q = qR 1$' \
  "$WORK/check.out"
rg -q '^chi_R2_ge_4_param_standard_unit : ¬Nonempty \(PlaneColouring \(Real × Real\) standardRealPlaneUnit 3\)$' \
  "$WORK/check.out"
rg -q '^madoreQ311ParametricBaseCaseCertificate : MadoreQ311ParametricBaseCaseCertificate$' \
  "$WORK/check.out"
rg -q '^madoreVitrineUnifiedPathsCertificate : MadoreVitrineUnifiedPathsCertificate$' \
  "$WORK/check.out"
rg -q '^madore_q311_parametric_base_case_standard_unit : ¬Nonempty \(PlaneColouring \(Real × Real\) standardRealPlaneUnit 3\)$' \
  "$WORK/check.out"

if rg -q 'dist²=1 exact' "$LEAN_DIR/lakefile.lean"; then
  echo "error: stale Q311 lakefile wording claims native dist²=1 instead of scaled dist²=144" >&2
  exit 1
fi
rg -q 'dist²=144 at ×12 scale' "$LEAN_DIR/lakefile.lean"

echo "madore_q311_parametric_base_case_gate: PASS"
