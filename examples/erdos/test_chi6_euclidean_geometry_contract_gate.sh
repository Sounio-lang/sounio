#!/usr/bin/env bash
# Build gate for the chi>=6 Euclidean geometry contract smoke.
#
# This inhabits the EuclideanNatEdgeExactGeometry type with a two-point segment
# and a four-edge unit square over Rat^2. It is a sanity check for the contract
# surface, not a scalability demonstration and not a chi>=6/no-5 witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

MOD="$ROOT/formal/lean4/SounioFiniteUnitDistanceEuclideanSmoke.lean"
[[ -s "$MOD" ]] || { echo "error: missing module: $MOD" >&2; exit 2; }
if rg -q '\b(noFiveWitness|noFivePlaneColouring|PlaneColouring|colourCNF|chi_ge_6_euclidean_plugin_contract)\b' "$MOD"; then
  echo "error: geometry smoke must not define or call no-5/chi>=6 certificate terms" >&2
  exit 1
fi

if rg -q '\b(sorry|admit)\b|#exit' "$MOD"; then
  echo "error: sorry/admit/#exit found in Euclidean geometry smoke" >&2
  exit 1
fi

if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
mkdir -p "$WORK"
VERIFY="$WORK/Chi6EuclideanGeometrySmokeVerify.lean"

echo "chi6_euclidean_geometry_contract_gate: building SounioFiniteUnitDistanceEuclideanSmoke"
(
  cd "$ROOT/formal/lean4"
  ../../scripts/dev/souc-build-lock.sh "$LAKE" build \
    SounioFiniteUnitDistanceWitness SounioFiniteUnitDistanceEuclideanSmoke
)

cat > "$VERIFY" <<'EOF'
import SounioFiniteUnitDistanceEuclideanSmoke
import Init.Data.Rat.Lemmas

open UnitDistanceChromatic
open UnitDistanceChromatic.Chi6EuclideanGeometrySmoke

#check (twoPointEuclideanGeometry : EuclideanNatEdgeExactGeometry 2 (Fin 2) twoPointUnit)
#check (twoPointExactGeometry : NatEdgeExactGeometry 2 (Fin 2) twoPointUnit)
#check (twoPointPlane : ExactSquaredDistancePlane (Fin 2) twoPointUnit)
#check (twoPointGeometryHasEuclideanContract :
  ∃ G : EuclideanNatEdgeExactGeometry 2 (Fin 2) twoPointUnit, G.exact.edges = [(0, 1)])
#check (squareEuclideanGeometry : EuclideanNatEdgeExactGeometry 4 (Fin 4) squareUnit)
#check (squareExactGeometry : NatEdgeExactGeometry 4 (Fin 4) squareUnit)
#check (squarePlane : ExactSquaredDistancePlane (Fin 4) squareUnit)
#check (squareGeometryHasEuclideanContract :
  ∃ G : EuclideanNatEdgeExactGeometry 4 (Fin 4) squareUnit,
    G.exact.edges = [(0, 1), (1, 2), (2, 3), (3, 0)])

example :
    ∃ G : EuclideanNatEdgeExactGeometry 2 (Fin 2) twoPointUnit, G.exact.edges = [(0, 1)] :=
  twoPointGeometryHasEuclideanContract
example :
    ∃ G : EuclideanNatEdgeExactGeometry 4 (Fin 4) squareUnit,
      G.exact.edges = [(0, 1), (1, 2), (2, 3), (3, 0)] :=
  squareGeometryHasEuclideanContract

example : twoPointEuclideanGeometry.exact.edges = [(0, 1)] := rfl
example : twoPointEuclideanGeometry.exact.emb = twoPointEmb := rfl
example : twoPointEuclideanGeometry.plane.dist2 = twoPointDist2 := rfl
example : twoPointEuclideanGeometry.plane.unit_irrefl = twoPointUnit_irrefl := rfl
example : twoPointUnit (twoPointEmb 0) (twoPointEmb 1) := by
  simp [twoPointEmb, twoPointUnit, twoPointDist2, twoPointX, twoPointY]
  native_decide
example : ¬ twoPointUnit (twoPointEmb 0) (twoPointEmb 0) := by
  exact twoPointUnit_irrefl (twoPointEmb 0)
example : twoPointEmb 0 ≠ twoPointEmb 1 := by
  native_decide
example : twoPointDist2 (twoPointEmb 0) (twoPointEmb 1) = (1 : Rat) := by
  simp [twoPointEmb, twoPointDist2, twoPointX, twoPointY]
  native_decide
example : ∀ e ∈ twoPointExactGeometry.edges, twoPointUnit (twoPointEmb e.1) (twoPointEmb e.2) := by
  intro e he
  exact twoPointExactGeometry.unit_edges e he

example : squareEuclideanGeometry.exact.edges = [(0, 1), (1, 2), (2, 3), (3, 0)] := rfl
example : squareEuclideanGeometry.exact.emb = squareEmb := rfl
example : squareEuclideanGeometry.plane.dist2 = squareDist2 := rfl
example : squareEuclideanGeometry.plane.unit_symm = squareUnit_symm := rfl
example : squareEuclideanGeometry.plane.unit_irrefl = squareUnit_irrefl := rfl
example (p q : Fin 4) :
    squareEuclideanGeometry.plane.dist2 p q =
      squareEuclideanGeometry.plane.scalar.add
        (squareEuclideanGeometry.plane.scalar.mul
          (squareEuclideanGeometry.plane.scalar.sub
            (squareEuclideanGeometry.plane.x p)
            (squareEuclideanGeometry.plane.x q))
          (squareEuclideanGeometry.plane.scalar.sub
            (squareEuclideanGeometry.plane.x p)
            (squareEuclideanGeometry.plane.x q)))
        (squareEuclideanGeometry.plane.scalar.mul
          (squareEuclideanGeometry.plane.scalar.sub
            (squareEuclideanGeometry.plane.y p)
            (squareEuclideanGeometry.plane.y q))
          (squareEuclideanGeometry.plane.scalar.sub
            (squareEuclideanGeometry.plane.y p)
            (squareEuclideanGeometry.plane.y q))) := by
  exact squareEuclideanGeometry.plane.dist2_formula p q
example (p q : Fin 4) : squareUnit p q → squareUnit q p := by
  exact squareEuclideanGeometry.plane.unit_symm p q
example (p : Fin 4) : ¬ squareUnit p p := by
  exact squareEuclideanGeometry.plane.unit_irrefl p
example : squareUnit (squareEmb 0) (squareEmb 1) → squareUnit (squareEmb 1) (squareEmb 0) := by
  exact squareEuclideanGeometry.plane.unit_symm (squareEmb 0) (squareEmb 1)
example : squareUnit (squareEmb 0) (squareEmb 1) := by
  simp [squareEmb, squareUnit, squareDist2, squarePointX, squarePointY]
  native_decide
example : squareUnit (squareEmb 1) (squareEmb 2) := by
  simp [squareEmb, squareUnit, squareDist2, squarePointX, squarePointY]
  native_decide
example : squareDist2 (squareEmb 0) (squareEmb 1) = (1 : Rat) := by
  simp [squareEmb, squareDist2, squarePointX, squarePointY]
  native_decide
example : squareDist2 (squareEmb 1) (squareEmb 2) = (1 : Rat) := by
  simp [squareEmb, squareDist2, squarePointX, squarePointY]
  native_decide
example : squareDist2 (squareEmb 2) (squareEmb 3) = (1 : Rat) := by
  simp [squareEmb, squareDist2, squarePointX, squarePointY]
  native_decide
example : squareDist2 (squareEmb 3) (squareEmb 0) = (1 : Rat) := by
  simp [squareEmb, squareDist2, squarePointX, squarePointY]
  native_decide
example : squareDist2 (squareEmb 0) (squareEmb 2) = (2 : Rat) := by
  simp [squareEmb, squareDist2, squarePointX, squarePointY]
  native_decide
example : ¬ squareUnit (squareEmb 0) (squareEmb 0) := by
  exact squareUnit_irrefl (squareEmb 0)
example : ∀ e ∈ squareExactGeometry.edges, squareUnit (squareEmb e.1) (squareEmb e.2) := by
  intro e he
  exact squareExactGeometry.unit_edges e he
EOF

if rg -q '\b(sorry|admit)\b|#exit' "$VERIFY"; then
  echo "error: sorry/admit/#exit found in generated Euclidean geometry verifier" >&2
  exit 1
fi

(
  cd "$ROOT/formal/lean4"
  "$LAKE" env lean "$VERIFY" > "$WORK/lean-verify.out" 2> "$WORK/lean-verify.err"
)
if rg -q 'error:' "$WORK/lean-verify.out" "$WORK/lean-verify.err"; then
  cat "$WORK/lean-verify.out" >&2
  cat "$WORK/lean-verify.err" >&2
  exit 1
fi
rg -q '^twoPointEuclideanGeometry : EuclideanNatEdgeExactGeometry 2' "$WORK/lean-verify.out"
rg -Fq 'twoPointGeometryHasEuclideanContract : ∃ G, G.exact.edges = [(0, 1)]' \
  "$WORK/lean-verify.out"
rg -q '^squareEuclideanGeometry : EuclideanNatEdgeExactGeometry 4' "$WORK/lean-verify.out"
rg -Fq 'squareGeometryHasEuclideanContract : ∃ G, G.exact.edges = [(0, 1), (1, 2), (2, 3), (3, 0)]' \
  "$WORK/lean-verify.out"

sha256sum "$MOD"
echo "chi6_euclidean_geometry_contract_gate: PASS"
