#!/usr/bin/env bash
# Temp-only end-to-end candidate-shaped gate for the chi>=6 lane.
#
# This uses K6/no-5 as a finite smoke to exercise the exact contract shape:
# SB5 reflected SAT module -> NatEdgeExactGeometry -> NoFiveColourWitness ->
# generic no-5 obstruction. It is not a Euclidean chi>=6 witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

OWN_WORK=0
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  OWN_WORK=1
fi

mkdir -p "$WORK"
SAT_MOD="$ROOT/formal/lean4/SounioChi6TempSat.lean"
CAND_MOD="$ROOT/formal/lean4/SounioChi6TempCandidateGate.lean"
FAKE_NO_FIELD_MOD="$ROOT/formal/lean4/SounioChi6FakeNoFieldLaws.lean"
FAKE_AXIOM_MOD="$ROOT/formal/lean4/SounioChi6FakeAxiom.lean"
cleanup() {
  rm -f "$SAT_MOD" "$CAND_MOD" "$FAKE_NO_FIELD_MOD" "$FAKE_AXIOM_MOD"
  if [[ "$OWN_WORK" == "1" ]]; then
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT INT TERM

for mod in "$SAT_MOD" "$CAND_MOD" "$FAKE_NO_FIELD_MOD" "$FAKE_AXIOM_MOD"; do
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

echo "chi6_temp_candidate_gate: workdir=$WORK"
SB_MODE=1 WORK="$WORK" \
  "$ROOT/examples/erdos/make_graph_reflect_certificate.sh" \
    "$WORK/k6.edge" 5 "$SAT_MOD" chi6temp SounioChi6TempSat "$WORK" \
  > "$WORK/producer.log"

{
cat <<'EOF'
import SounioFiniteUnitDistanceWitness
EOF
cat "$SAT_MOD"
cat <<'EOF'
open UnitDistanceChromatic

namespace SounioChi6TempCandidateGate

def tempUnit : Nat → Nat → Prop := fun _ _ => True
def tempEmb : Nat → Nat := id

/-- K6 finite smoke geometry. This is intentionally not Euclidean geometry. -/
def tempGeometry : NatEdgeExactGeometry 6 Nat tempUnit where
  edges := chi6temp_edges
  emb := tempEmb
  emb_injective := by
    intro i j _hi _hj h
    exact h
  endpoints := by native_decide
  unit_edges := by
    intro e he
    trivial

theorem tempEdgesSync : tempGeometry.edges = chi6temp_edges := rfl

/-- Candidate-shaped no-5 witness smoke, routed through the same SB5 interface as a real candidate. -/
def tempNoFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat tempUnit :=
  tempGeometry.noFiveWitnessOfColourCNFsb5UnsatTri
    0 1 2
    (by decide) (by decide) (by decide)
    (by native_decide) (by native_decide) (by native_decide)
    chi6temp_unsat

/-- The generic no-5 obstruction exercised on a finite non-Euclidean smoke relation. -/
theorem temp_candidate_contract_smoke :
    ¬ Nonempty (PlaneColouring Nat tempUnit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction tempNoFiveWitness

#print axioms temp_candidate_contract_smoke

end SounioChi6TempCandidateGate
EOF
} > "$CAND_MOD"

cat > "$WORK/manifest.env" <<EOF
candidate_manifest_version=1
promotable=0
candidate_id=chi6_temp_k6_sb5_not_planar
n=6
m=15
k=5
edge_path=k6.edge
edge_sha256=$(sha256sum "$WORK/k6.edge" | awk '{print $1}')
cnf_path=souc_sat_worker.cnf
cnf_sha256=$(sha256sum "$WORK/souc_sat_worker.cnf" | awk '{print $1}')
drat_or_lrat_path=chi6temp.lrat
drat_or_lrat_sha256=$(sha256sum "$WORK/chi6temp.lrat" | awk '{print $1}')
lean_sat_module_path=$SAT_MOD
lean_sat_module_sha256=$(sha256sum "$SAT_MOD" | awk '{print $1}')
geometry_module_path=$CAND_MOD
geometry_module_sha256=$(sha256sum "$CAND_MOD" | awk '{print $1}')
geometry_proof_type=finite_smoke
sat_proof_route=triangle_sb5_lrat
triangle_sb=0,1,2
generator_commit=$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null || echo UNKNOWN)
producer_command=SB_MODE=1 examples/erdos/make_graph_reflect_certificate.sh k6.edge 5 SounioChi6TempSat.lean chi6temp SounioChi6TempSat
lean_build_command=lake env lean SounioChi6TempSat.lean && lake env lean SounioChi6TempCandidateGate.lean
offload_review_raw=NONE
offload_review_sha256=NONE
EOF

"$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh" "$WORK/manifest.env"
cp "$WORK/manifest.env" "$WORK/bad-promotable-finite-smoke.env"
sed -i 's/^promotable=.*/promotable=1/' "$WORK/bad-promotable-finite-smoke.env"
if "$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh" \
    "$WORK/bad-promotable-finite-smoke.env" > "$WORK/bad-promotable-finite-smoke.out" 2>&1; then
  echo "error: promotable finite_smoke manifest unexpectedly validated" >&2
  exit 1
fi
rg -q 'promotable=1 requires geometry_proof_type=euclidean' \
  "$WORK/bad-promotable-finite-smoke.out"

cp "$WORK/manifest.env" "$WORK/bad-promotable-old-geometry.env"
printf 'temp smoke review placeholder for negative validator fixture\n' > "$WORK/offload-review.txt"
sed -i 's/^promotable=.*/promotable=1/' "$WORK/bad-promotable-old-geometry.env"
sed -i 's/^geometry_proof_type=.*/geometry_proof_type=euclidean/' "$WORK/bad-promotable-old-geometry.env"
sed -i 's|^offload_review_raw=.*|offload_review_raw=offload-review.txt|' \
  "$WORK/bad-promotable-old-geometry.env"
sed -i "s|^offload_review_sha256=.*|offload_review_sha256=$(sha256sum "$WORK/offload-review.txt" | awk '{print $1}')|" \
  "$WORK/bad-promotable-old-geometry.env"
sed -i 's|^lean_build_command=.*|lean_build_command=lake build SounioChi6TempSat SounioChi6TempCandidateGate|' \
  "$WORK/bad-promotable-old-geometry.env"
cp "$WORK/bad-promotable-old-geometry.env" "$WORK/bad-promotable-missing-terms.env"
if "$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh" \
    "$WORK/bad-promotable-missing-terms.env" > "$WORK/bad-promotable-missing-terms.out" 2>&1; then
  echo "error: promotable manifest without Lean term names unexpectedly validated" >&2
  exit 1
fi
rg -q 'promotable=1 requires lean_module' "$WORK/bad-promotable-missing-terms.out"

cat >> "$WORK/bad-promotable-old-geometry.env" <<'EOF'
lean_module=SounioChi6TempCandidateGate
lean_sat_edges_term=chi6temp_edges
lean_point_type=Nat
lean_unit_term=SounioChi6TempCandidateGate.tempUnit
lean_geometry_term=SounioChi6TempCandidateGate.tempGeometry
lean_edges_sync_term=SounioChi6TempCandidateGate.tempEdgesSync
lean_no_five_witness_term=SounioChi6TempCandidateGate.tempNoFiveWitness
lean_final_theorem=SounioChi6TempCandidateGate.temp_candidate_contract_smoke
lean_real_unit_term=SounioChi6TempCandidateGate.tempRealUnit
lean_real_unit_iff_standard=SounioChi6TempCandidateGate.tempRealUnitIffStandard
lean_real_final_theorem=SounioChi6TempCandidateGate.temp_real_candidate_contract_smoke
EOF
if "$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh" \
    "$WORK/bad-promotable-old-geometry.env" > "$WORK/bad-promotable-old-geometry.out" 2>&1; then
  echo "error: promotable manifest without EuclideanNatEdgeExactGeometry unexpectedly validated" >&2
  exit 1
fi
rg -q 'lacks EuclideanNatEdgeExactGeometry shape' "$WORK/bad-promotable-old-geometry.out"

cp "$WORK/bad-promotable-old-geometry.env" "$WORK/bad-promotable-no-field-laws.env"
cat > "$FAKE_NO_FIELD_MOD" <<'EOF'
import SounioRootedFieldReal

def fakeRealUnit : Real × Real → Real × Real → Prop := fun _ _ => True

theorem fakeRealFinal : True := by
  trivial

def fakeEuclideanMarker :=
  "EuclideanNatEdgeExactGeometry chi_ge_6_euclidean_plugin_contract"
EOF
sed -i "s|^geometry_module_path=.*|geometry_module_path=$FAKE_NO_FIELD_MOD|" \
  "$WORK/bad-promotable-no-field-laws.env"
sed -i "s|^geometry_module_sha256=.*|geometry_module_sha256=$(sha256sum "$FAKE_NO_FIELD_MOD" | awk '{print $1}')|" \
  "$WORK/bad-promotable-no-field-laws.env"
sed -i 's/^lean_module=.*/lean_module=SounioChi6FakeNoFieldLaws/' \
  "$WORK/bad-promotable-no-field-laws.env"
sed -i 's/^lean_build_command=.*/lean_build_command=lake build SounioChi6FakeNoFieldLaws/' \
  "$WORK/bad-promotable-no-field-laws.env"
sed -i 's/^lean_real_unit_term=.*/lean_real_unit_term=fakeRealUnit/' \
  "$WORK/bad-promotable-no-field-laws.env"
sed -i 's/^lean_real_unit_iff_standard=.*/lean_real_unit_iff_standard=fakeRealUnitIffStandard/' \
  "$WORK/bad-promotable-no-field-laws.env"
sed -i 's/^lean_real_final_theorem=.*/lean_real_final_theorem=fakeRealFinal/' \
  "$WORK/bad-promotable-no-field-laws.env"
if "$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh" \
    "$WORK/bad-promotable-no-field-laws.env" > "$WORK/bad-promotable-no-field-laws.out" 2>&1; then
  echo "error: promotable manifest without ExactFieldLike unexpectedly validated" >&2
  exit 1
fi
rg -q 'lacks ExactFieldLike scalar-law shape' "$WORK/bad-promotable-no-field-laws.out"

cp "$WORK/bad-promotable-old-geometry.env" "$WORK/bad-promotable-axiom.env"
cat > "$FAKE_AXIOM_MOD" <<'EOF'
import SounioFiniteUnitDistanceWitness
import SounioRootedFieldReal

open UnitDistanceChromatic

def fakeUnit : Nat → Nat → Prop := fun _ _ => True
def fakeRealUnit : Real × Real → Real × Real → Prop := fun _ _ => True
axiom fakeGeometry : EuclideanNatEdgeExactGeometry 6 Nat (fun _ _ => True)
axiom fakeNoFiveWitness :
  NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat (fun _ _ => True)
axiom fakeFinal :
  ¬ Nonempty (PlaneColouring Nat (fun _ _ => True) 5)
axiom fakeRealFinal :
  ¬ Nonempty (PlaneColouring (Real × Real) fakeRealUnit 5)

/- ExactFieldLike chi_ge_6_euclidean_plugin_contract marker strings are not enough. -/
EOF
sed -i "s|^geometry_module_path=.*|geometry_module_path=$FAKE_AXIOM_MOD|" \
  "$WORK/bad-promotable-axiom.env"
sed -i "s|^geometry_module_sha256=.*|geometry_module_sha256=$(sha256sum "$FAKE_AXIOM_MOD" | awk '{print $1}')|" \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_module=.*/lean_module=SounioChi6FakeAxiom/' \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_build_command=.*/lean_build_command=lake build SounioChi6FakeAxiom/' \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_geometry_term=.*/lean_geometry_term=fakeGeometry/' \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_no_five_witness_term=.*/lean_no_five_witness_term=fakeNoFiveWitness/' \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_final_theorem=.*/lean_final_theorem=fakeFinal/' \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_unit_term=.*/lean_unit_term=fakeUnit/' \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_real_unit_term=.*/lean_real_unit_term=fakeRealUnit/' \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_real_unit_iff_standard=.*/lean_real_unit_iff_standard=fakeRealUnitIffStandard/' \
  "$WORK/bad-promotable-axiom.env"
sed -i 's/^lean_real_final_theorem=.*/lean_real_final_theorem=fakeRealFinal/' \
  "$WORK/bad-promotable-axiom.env"
if "$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh" \
    "$WORK/bad-promotable-axiom.env" > "$WORK/bad-promotable-axiom.out" 2>&1; then
  echo "error: promotable manifest with axiom-backed Lean terms unexpectedly validated" >&2
  exit 1
fi
rg -q 'declares axiom/constant/opaque' "$WORK/bad-promotable-axiom.out"

rg -q '^def chi6temp_cnf : CNF Nat := colourCNFsb5 0 1 2 6 chi6temp_edges$' "$SAT_MOD"
rg -q 'NatEdgeExactGeometry 6 Nat tempUnit' "$CAND_MOD"
rg -q 'generic_no_five_colour_obstruction tempNoFiveWitness' "$CAND_MOD"

(
  cd "$ROOT/formal/lean4"
  "$LAKE" env lean "$SAT_MOD"
  "$LAKE" env lean "$CAND_MOD"
)

if rg -q '\b(sorry|admit)\b' "$SAT_MOD" "$CAND_MOD"; then
  echo "error: sorry/admit found in temp candidate gate" >&2
  exit 1
fi

sha256sum "$WORK/manifest.env" "$SAT_MOD" "$CAND_MOD"
echo "chi6_temp_candidate_gate: PASS"
