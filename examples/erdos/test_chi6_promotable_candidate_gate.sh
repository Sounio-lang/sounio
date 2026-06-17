#!/usr/bin/env bash
# Negative gate for promotable chi>=6 manifests.
#
# A marker-only geometry file can satisfy lightweight hash/text checks, but it
# must fail the generated Lean verifier because the named terms do not have the
# Euclidean geometry / no-five witness / final theorem types.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
PROMOTABLE_GATE="$ROOT/examples/erdos/validate_chi6_promotable_candidate.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }

rg -q 'chi6StandardRealNoFiveColouring' "$PROMOTABLE_GATE"
rg -Fq 'PlaneColouring (Real × Real) standardRealPlaneUnit 5' "$PROMOTABLE_GATE"

OWN_WORK=0
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  OWN_WORK=1
fi
mkdir -p "$WORK"

SAT_MOD="$ROOT/formal/lean4/SounioChi6FakePromotableSat.lean"
BAD_SAT_MOD="$ROOT/formal/lean4/SounioChi6FakePromotableSatUnsafe.lean"
GEOM_MOD="$ROOT/formal/lean4/SounioChi6FakePromotable.lean"
BAD_ROUTE_GEOM_MOD="$ROOT/formal/lean4/SounioChi6FakePromotableNoRouteAdapter.lean"
HELPER_AXIOM_MOD="$ROOT/formal/lean4/SounioChi6FakePromotableHelper.lean"
HIDDEN_AXIOM_GEOM_MOD="$ROOT/formal/lean4/SounioChi6FakePromotableHiddenAxiom.lean"
cleanup() {
  rm -f "$SAT_MOD" "$BAD_SAT_MOD" "$GEOM_MOD" "$BAD_ROUTE_GEOM_MOD" \
    "$HELPER_AXIOM_MOD" "$HIDDEN_AXIOM_GEOM_MOD"
  if [[ "$OWN_WORK" == "1" ]]; then
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT INT TERM

for mod in "$SAT_MOD" "$BAD_SAT_MOD" "$GEOM_MOD" "$BAD_ROUTE_GEOM_MOD" \
    "$HELPER_AXIOM_MOD" "$HIDDEN_AXIOM_GEOM_MOD"; do
  [[ ! -e "$mod" ]] || { echo "error: temp Lean module path already exists: $mod" >&2; exit 1; }
done

cat > "$WORK/k6.edge" <<'EOF'
p edge 6 15
e 1 2
e 1 3
e 1 4
e 1 5
e 1 6
e 2 3
e 2 4
e 2 5
e 2 6
e 3 4
e 3 5
e 3 6
e 4 5
e 4 6
e 5 6
EOF

cat > "$WORK/fake.cnf" <<'EOF'
p cnf 30 0
EOF

cat > "$WORK/fake.lrat" <<'EOF'
0
EOF

cat > "$WORK/offload-review.txt" <<'EOF'
negative fake promotable fixture; must not pass the Lean verifier
EOF

cat > "$SAT_MOD" <<'EOF'
import SounioSatColouringBridge
import SounioSatColouringSB

open Std.Sat
open SounioSatColouring
open SounioSatColouringSB

def fake_edges : List (Nat × Nat) :=
  [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
   (1, 2), (1, 3), (1, 4), (1, 5),
   (2, 3), (2, 4), (2, 5),
   (3, 4), (3, 5),
   (4, 5)]

def fake_cnf : CNF Nat := colourCNFsb5 0 1 2 6 fake_edges
EOF

cat > "$GEOM_MOD" <<'EOF'
import SounioFiniteUnitDistanceWitness
import SounioRootedFieldReal
import SounioMultiquadIndep

open UnitDistanceChromatic
open SounioSqrt.RealCauchyField

namespace SounioChi6FakePromotable

def fakeUnit : Nat → Nat → Prop := fun _ _ => True

def fakeRealUnit : Real × Real → Real × Real → Prop := fun _ _ => True

theorem fakeRealUnitIffStandard :
    ∀ p q : Real × Real, fakeRealUnit p q ↔ fakeRealUnit p q := by
  intro p q
  rfl

theorem fakeEdgesSync : True := by
  trivial

def fakeGeometry :=
  "EuclideanNatEdgeExactGeometry ExactFieldLike chi_ge_6_euclidean_plugin_contract noFiveWitnessOfColourCNFsb5UnsatTri"

def fakeNoFiveWitness :=
  "NatEdgeUnitDistanceCertificate.NoFiveColourWitness"

theorem fakeFinal : True := by
  trivial

theorem fakeRealFinal : True := by
  trivial

end SounioChi6FakePromotable
EOF

cat > "$HELPER_AXIOM_MOD" <<'EOF'
import SounioFiniteUnitDistanceWitness
import SounioRootedFieldReal
import SounioMultiquadIndep
import SounioRealPlaneGeometry

open UnitDistanceChromatic
open SounioSqrt.RealCauchyField

namespace SounioChi6FakePromotableHelper

axiom fakeUnit : Nat → Nat → Prop
axiom fakeRealUnit : Real × Real → Real × Real → Prop
axiom fakeRealUnitIffStandard :
  ∀ p q : Real × Real,
    fakeRealUnit p q ↔
      standardRealPlaneDist2 p q = qR (1 : Rat)
axiom fakeGeometry : EuclideanNatEdgeExactGeometry 6 Nat fakeUnit
axiom fakeEdgesSync :
  fakeGeometry.exact.edges =
    [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
     (1, 2), (1, 3), (1, 4), (1, 5),
     (2, 3), (2, 4), (2, 5),
     (3, 4), (3, 5),
     (4, 5)]
axiom fakeNoFiveWitness :
  NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat fakeUnit
axiom fakeFinal :
  ¬ Nonempty (PlaneColouring Nat fakeUnit 5)
axiom fakeRealFinal :
  ¬ Nonempty (PlaneColouring (Real × Real) fakeRealUnit 5)

end SounioChi6FakePromotableHelper
EOF

cat > "$HIDDEN_AXIOM_GEOM_MOD" <<'EOF'
import SounioChi6FakePromotableHelper

open UnitDistanceChromatic
open SounioSqrt.RealCauchyField

namespace SounioChi6FakePromotableHiddenAxiom

def fakeUnit : Nat → Nat → Prop :=
  SounioChi6FakePromotableHelper.fakeUnit

def fakeRealUnit : Real × Real → Real × Real → Prop :=
  SounioChi6FakePromotableHelper.fakeRealUnit

theorem fakeRealUnitIffStandard :
    ∀ p q : Real × Real,
      fakeRealUnit p q ↔
        standardRealPlaneDist2 p q = qR (1 : Rat) := by
  simpa [fakeRealUnit] using SounioChi6FakePromotableHelper.fakeRealUnitIffStandard

noncomputable def fakeGeometry : EuclideanNatEdgeExactGeometry 6 Nat fakeUnit := by
  simpa [fakeUnit] using SounioChi6FakePromotableHelper.fakeGeometry

theorem fakeEdgesSync :
    fakeGeometry.exact.edges =
      [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
       (1, 2), (1, 3), (1, 4), (1, 5),
       (2, 3), (2, 4), (2, 5),
       (3, 4), (3, 5),
       (4, 5)] := by
  simpa [fakeGeometry] using SounioChi6FakePromotableHelper.fakeEdgesSync

noncomputable def fakeNoFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat fakeUnit := by
  simpa [fakeUnit] using SounioChi6FakePromotableHelper.fakeNoFiveWitness

theorem fakeFinal : ¬ Nonempty (PlaneColouring Nat fakeUnit 5) := by
  simpa [fakeUnit] using SounioChi6FakePromotableHelper.fakeFinal

theorem fakeRealFinal : ¬ Nonempty (PlaneColouring (Real × Real) fakeRealUnit 5) := by
  simpa [fakeRealUnit] using SounioChi6FakePromotableHelper.fakeRealFinal

/- ExactFieldLike EuclideanNatEdgeExactGeometry chi_ge_6_euclidean_plugin_contract
   noFiveWitnessOfColourCNFsb5UnsatTri marker strings are not enough. -/

end SounioChi6FakePromotableHiddenAxiom
EOF

cat > "$BAD_ROUTE_GEOM_MOD" <<'EOF'
import SounioFiniteUnitDistanceWitness
import SounioRootedFieldReal
import SounioMultiquadIndep

open UnitDistanceChromatic
open SounioSqrt.RealCauchyField

namespace SounioChi6FakePromotableNoRouteAdapter

def fakeUnit : Nat → Nat → Prop := fun _ _ => True

def fakeRealUnit : Real × Real → Real × Real → Prop := fun _ _ => True

theorem fakeRealUnitIffStandard :
    ∀ p q : Real × Real, fakeRealUnit p q ↔ fakeRealUnit p q := by
  intro p q
  rfl

theorem fakeEdgesSync : True := by
  trivial

def fakeGeometry :=
  "EuclideanNatEdgeExactGeometry ExactFieldLike chi_ge_6_euclidean_plugin_contract"

def fakeNoFiveWitness :=
  "NatEdgeUnitDistanceCertificate.NoFiveColourWitness"

theorem fakeFinal : True := by
  trivial

theorem fakeRealFinal : True := by
  trivial

end SounioChi6FakePromotableNoRouteAdapter
EOF

cat > "$WORK/manifest.env" <<EOF
candidate_manifest_version=1
promotable=1
candidate_id=fake_textual_promotable
n=6
m=15
k=5
edge_path=k6.edge
edge_sha256=$(sha256sum "$WORK/k6.edge" | awk '{print $1}')
cnf_path=fake.cnf
cnf_sha256=$(sha256sum "$WORK/fake.cnf" | awk '{print $1}')
drat_or_lrat_path=fake.lrat
drat_or_lrat_sha256=$(sha256sum "$WORK/fake.lrat" | awk '{print $1}')
lean_sat_module_path=$SAT_MOD
lean_sat_module_sha256=$(sha256sum "$SAT_MOD" | awk '{print $1}')
geometry_module_path=$GEOM_MOD
geometry_module_sha256=$(sha256sum "$GEOM_MOD" | awk '{print $1}')
geometry_proof_type=euclidean
sat_proof_route=triangle_sb5_lrat
triangle_sb=0,1,2
generator_commit=$(git -C "$ROOT" rev-parse --verify HEAD)
producer_command=negative fake fixture
lean_build_command=lake build SounioChi6FakePromotableSat SounioChi6FakePromotable
offload_review_raw=offload-review.txt
offload_review_sha256=$(sha256sum "$WORK/offload-review.txt" | awk '{print $1}')
lean_module=SounioChi6FakePromotable
lean_sat_edges_term=fake_edges
lean_point_type=Nat
lean_unit_term=SounioChi6FakePromotable.fakeUnit
lean_geometry_term=SounioChi6FakePromotable.fakeGeometry
lean_edges_sync_term=SounioChi6FakePromotable.fakeEdgesSync
lean_no_five_witness_term=SounioChi6FakePromotable.fakeNoFiveWitness
lean_final_theorem=SounioChi6FakePromotable.fakeFinal
lean_real_unit_term=SounioChi6FakePromotable.fakeRealUnit
lean_real_unit_iff_standard=SounioChi6FakePromotable.fakeRealUnitIffStandard
lean_real_final_theorem=SounioChi6FakePromotable.fakeRealFinal
EOF

"$VALIDATOR" "$WORK/manifest.env" | tee "$WORK/validator.out"
rg -q '^chi6_manifest: VALID_PROMOTABLE_FORMAT candidate=fake_textual_promotable$' \
  "$WORK/validator.out"

cp "$WORK/manifest.env" "$WORK/bad-sat-edges.env"
sed -i '/^lean_sat_edges_term=/d' "$WORK/bad-sat-edges.env"
if "$VALIDATOR" "$WORK/bad-sat-edges.env" > "$WORK/bad-sat-edges.out" 2>&1; then
  echo "error: promotable manifest without SAT edge term unexpectedly validated" >&2
  exit 1
fi
rg -q 'promotable=1 requires lean_sat_edges_term' "$WORK/bad-sat-edges.out"

cp "$WORK/manifest.env" "$WORK/bad-edge-sync.env"
sed -i '/^lean_edges_sync_term=/d' "$WORK/bad-edge-sync.env"
if "$VALIDATOR" "$WORK/bad-edge-sync.env" > "$WORK/bad-edge-sync.out" 2>&1; then
  echo "error: promotable manifest without edge sync term unexpectedly validated" >&2
  exit 1
fi
rg -q 'promotable=1 requires lean_edges_sync_term' "$WORK/bad-edge-sync.out"

cp "$WORK/manifest.env" "$WORK/bad-real-bridge.env"
sed -i '/^lean_real_final_theorem=/d' "$WORK/bad-real-bridge.env"
if "$VALIDATOR" "$WORK/bad-real-bridge.env" > "$WORK/bad-real-bridge.out" 2>&1; then
  echo "error: promotable manifest without Real-plane theorem unexpectedly validated" >&2
  exit 1
fi
rg -q 'promotable=1 requires lean_real_final_theorem' "$WORK/bad-real-bridge.out"

cp "$WORK/manifest.env" "$WORK/bad-route-adapter.env"
sed -i "s|^geometry_module_path=.*|geometry_module_path=$BAD_ROUTE_GEOM_MOD|" \
  "$WORK/bad-route-adapter.env"
sed -i "s/^geometry_module_sha256=.*/geometry_module_sha256=$(sha256sum "$BAD_ROUTE_GEOM_MOD" | awk '{print $1}')/" \
  "$WORK/bad-route-adapter.env"
sed -i 's/^lean_module=.*/lean_module=SounioChi6FakePromotableNoRouteAdapter/' \
  "$WORK/bad-route-adapter.env"
sed -i 's/^lean_build_command=.*/lean_build_command=lake build SounioChi6FakePromotableSat SounioChi6FakePromotableNoRouteAdapter/' \
  "$WORK/bad-route-adapter.env"
if "$VALIDATOR" "$WORK/bad-route-adapter.env" > "$WORK/bad-route-adapter.out" 2>&1; then
  echo "error: promotable manifest without route adapter marker unexpectedly validated" >&2
  exit 1
fi
rg -q 'sat_proof_route=triangle_sb5_lrat requires SB5 witness adapter in geometry module' \
  "$WORK/bad-route-adapter.out"

cp "$WORK/manifest.env" "$WORK/bad-generator-commit.env"
sed -i 's/^generator_commit=.*/generator_commit=0000000000000000000000000000000000000000/' \
  "$WORK/bad-generator-commit.env"
if "$VALIDATOR" "$WORK/bad-generator-commit.env" > "$WORK/bad-generator-commit.out" 2>&1; then
  echo "error: promotable manifest with absent generator_commit unexpectedly validated" >&2
  exit 1
fi
rg -q 'promotable=1 generator_commit is not present in this repo' \
  "$WORK/bad-generator-commit.out"

cp "$WORK/manifest.env" "$WORK/bad-build-target.env"
sed -i 's/^lean_build_command=.*/lean_build_command=lake build SounioChi6FakePromotableSat/' \
  "$WORK/bad-build-target.env"
if "$VALIDATOR" "$WORK/bad-build-target.env" > "$WORK/bad-build-target.out" 2>&1; then
  echo "error: promotable manifest with missing geometry build target unexpectedly validated" >&2
  exit 1
fi
rg -q 'lean_build_command must include lean_module target SounioChi6FakePromotable' \
  "$WORK/bad-build-target.out"

: > "$WORK/empty-offload-review.txt"
cp "$WORK/manifest.env" "$WORK/bad-empty-offload.env"
sed -i 's/^offload_review_raw=.*/offload_review_raw=empty-offload-review.txt/' \
  "$WORK/bad-empty-offload.env"
sed -i "s/^offload_review_sha256=.*/offload_review_sha256=$(sha256sum "$WORK/empty-offload-review.txt" | awk '{print $1}')/" \
  "$WORK/bad-empty-offload.env"
if "$VALIDATOR" "$WORK/bad-empty-offload.env" > "$WORK/bad-empty-offload.out" 2>&1; then
  echo "error: promotable manifest with empty offload review unexpectedly validated" >&2
  exit 1
fi
rg -q 'missing/empty offload_review_raw artifact|offload_review_raw artifact must be non-empty text' \
  "$WORK/bad-empty-offload.out"

cp "$SAT_MOD" "$BAD_SAT_MOD"
printf '\n#check Nat\n' >> "$BAD_SAT_MOD"
cp "$WORK/manifest.env" "$WORK/bad-lean-surface.env"
sed -i "s|^lean_sat_module_path=.*|lean_sat_module_path=$BAD_SAT_MOD|" \
  "$WORK/bad-lean-surface.env"
sed -i "s/^lean_sat_module_sha256=.*/lean_sat_module_sha256=$(sha256sum "$BAD_SAT_MOD" | awk '{print $1}')/" \
  "$WORK/bad-lean-surface.env"
if "$VALIDATOR" "$WORK/bad-lean-surface.env" > "$WORK/bad-lean-surface.out" 2>&1; then
  echo "error: promotable manifest with unsafe Lean surface unexpectedly validated" >&2
  exit 1
fi
rg -q 'unsafe/partial/#eval/#check/#exit|sorry/admit/#exit/#eval/#check' \
  "$WORK/bad-lean-surface.out"

if "$PROMOTABLE_GATE" "$WORK/manifest.env" > "$WORK/promotable-gate.out" 2>&1; then
  echo "error: fake textual promotable manifest unexpectedly passed Lean verifier" >&2
  exit 1
fi
rg -q 'type mismatch|failed to synthesize|error:' "$WORK/promotable-gate.out"

cp "$WORK/manifest.env" "$WORK/hidden-axiom.env"
sed -i "s|^geometry_module_path=.*|geometry_module_path=$HIDDEN_AXIOM_GEOM_MOD|" \
  "$WORK/hidden-axiom.env"
sed -i "s/^geometry_module_sha256=.*/geometry_module_sha256=$(sha256sum "$HIDDEN_AXIOM_GEOM_MOD" | awk '{print $1}')/" \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_module=.*/lean_module=SounioChi6FakePromotableHiddenAxiom/' \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_unit_term=.*/lean_unit_term=SounioChi6FakePromotableHiddenAxiom.fakeUnit/' \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_geometry_term=.*/lean_geometry_term=SounioChi6FakePromotableHiddenAxiom.fakeGeometry/' \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_edges_sync_term=.*/lean_edges_sync_term=SounioChi6FakePromotableHiddenAxiom.fakeEdgesSync/' \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_no_five_witness_term=.*/lean_no_five_witness_term=SounioChi6FakePromotableHiddenAxiom.fakeNoFiveWitness/' \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_final_theorem=.*/lean_final_theorem=SounioChi6FakePromotableHiddenAxiom.fakeFinal/' \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_real_unit_term=.*/lean_real_unit_term=SounioChi6FakePromotableHiddenAxiom.fakeRealUnit/' \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_real_final_theorem=.*/lean_real_final_theorem=SounioChi6FakePromotableHiddenAxiom.fakeRealFinal/' \
  "$WORK/hidden-axiom.env"
sed -i 's/^lean_build_command=.*/lean_build_command=lake build SounioChi6FakePromotableSat SounioChi6FakePromotableHiddenAxiom/' \
  "$WORK/hidden-axiom.env"
mkdir -p "$WORK/leanlib"
(
  cd "$ROOT/formal/lean4"
  "$LAKE" env lean -o "$WORK/leanlib/SounioChi6FakePromotableHelper.olean" "$HELPER_AXIOM_MOD"
  LEAN_PATH="$WORK/leanlib${LEAN_PATH:+:$LEAN_PATH}" \
    "$LAKE" env lean -o "$WORK/leanlib/SounioChi6FakePromotableHiddenAxiom.olean" \
      "$HIDDEN_AXIOM_GEOM_MOD"
)
if LEAN_PATH="$WORK/leanlib${LEAN_PATH:+:$LEAN_PATH}" \
    "$PROMOTABLE_GATE" "$WORK/hidden-axiom.env" > "$WORK/hidden-axiom.out" 2>&1; then
  echo "error: promotable manifest with helper-owned axioms unexpectedly passed Lean verifier" >&2
  exit 1
fi
rg -q 'unexpected axiom dependencies|SounioChi6FakePromotableHelper.fake' \
  "$WORK/hidden-axiom.out"

echo "chi6_promotable_candidate_gate: PASS"
