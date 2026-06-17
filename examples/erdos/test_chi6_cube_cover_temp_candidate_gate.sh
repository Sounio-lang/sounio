#!/usr/bin/env bash
# Temp-only candidate-shaped gate for the generic cube-cover route.
#
# This is not Euclidean geometry evidence. It joins a generated arbitrary
# complement cube-cover SAT module to a candidate-owned finite geometry object
# with an explicit generated edge-list identity, then routes through
# NatEdgeExactGeometry.noFiveWitnessOfCubeCoverUnsat.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN_DIR="$ROOT/formal/lean4"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COMP_CNF="$ROOT/examples/erdos/cube_cover_complement_cnf.py"
CONVERTER="$ROOT/examples/erdos/drup_to_lrat_rup.py"
GEN="$ROOT/examples/erdos/gen_lean_cube_cover_reflect.py"

if [[ ! -x "$LOCK" ]]; then
  echo "error: missing build lock helper: $LOCK" >&2
  exit 1
fi
if [[ ! -x "$LAKE" ]]; then
  echo "error: missing lake executable: $LAKE" >&2
  exit 1
fi
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$REFUTER" "$COMP_CNF" "$CONVERTER" "$GEN"

OWN_WORK=0
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)" || { echo "error: failed to create temp dir" >&2; exit 1; }
  OWN_WORK=1
fi
mkdir -p "$WORK/refute"

SAT_MOD="$LEAN_DIR/SounioChi6TempCubeCoverSat.lean"
CAND_MOD="$LEAN_DIR/SounioChi6TempCubeCoverCandidateGate.lean"
cleanup() {
  rm -f "$SAT_MOD" "$CAND_MOD"
  if [[ "$OWN_WORK" == "1" ]]; then
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT INT TERM

for mod in "$SAT_MOD" "$CAND_MOD"; do
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

for c0 in 0 1 2 3 4; do
  for c1 in 0 1 2 3 4; do
    printf 'v0_c%s_v1_c%s: 0:%s 1:%s\n' "$c0" "$c1" "$c0" "$c1"
  done
done > "$WORK/k6_v0_v1_cover.cubes"

echo "chi6_cube_cover_temp_candidate_gate: workdir=$WORK"
python3 "$REFUTER" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/refute" \
  > "$WORK/refute.out"
python3 "$COMP_CNF" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/cover_complement.cnf" \
  > "$WORK/cover_complement.out"

cat > "$WORK/cover_complement.drup" <<'EOF'
-1 0
-2 0
-3 0
-4 0
-5 0
0
EOF

python3 "$CONVERTER" "$WORK/cover_complement.cnf" "$WORK/cover_complement.drup" \
  "$WORK/cover_complement.lrat" > "$WORK/cover_lrat.out" 2> "$WORK/cover_lrat.err"
[[ -s "$WORK/cover_complement.lrat" ]]

python3 "$GEN" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/refute.out" \
  "$SAT_MOD" \
  --module SounioChi6TempCubeCoverSat \
  --prefix chi6cube \
  --composition arbitrary \
  --cover-cnf "$WORK/cover_complement.cnf" \
  --cover-lrat "$WORK/cover_complement.lrat" \
  > "$WORK/gen.out"

rg -q '^leaf_count=25$' "$WORK/gen.out"
rg -q '^composition=arbitrary$' "$WORK/gen.out"
rg -q '^theorem chi6cube_unsat_from_arbitrary_cube_cover' "$SAT_MOD"
rg -q '^theorem chi6cube_leaf24_check' "$SAT_MOD"
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' "$SAT_MOD"
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' "$SAT_MOD"
if rg -q '\b(sorry|admit)\b' "$SAT_MOD"; then
  echo "error: generated SAT module contains incomplete proof marker" >&2
  exit 1
fi

cat > "$CAND_MOD" <<'EOF'
import SounioFiniteUnitDistanceWitness
import SounioSatCubeCover
import SounioSatReflect
EOF
rg -v '^import ' "$SAT_MOD" >> "$CAND_MOD"
cat >> "$CAND_MOD" <<'EOF'

open UnitDistanceChromatic

namespace SounioChi6TempCubeCoverCandidateGate

/-- Finite graph-unit relation induced by the generated edge list. This is still
not Euclidean geometry, but it is not the universal relation. -/
def tempUnit (p q : Nat) : Prop := (p, q) ∈ chi6cube_edges ∨ (q, p) ∈ chi6cube_edges
def tempEmb : Nat → Nat := id

theorem tempUnit_symm {p q : Nat} : tempUnit p q → tempUnit q p := by
  intro h
  rcases h with h | h
  · exact Or.inr h
  · exact Or.inl h

/-- Candidate-owned finite smoke geometry. This is intentionally not Euclidean geometry. -/
def tempCubeGeometry : NatEdgeExactGeometry 6 Nat tempUnit where
  edges := chi6cube_edges
  emb := tempEmb
  emb_injective := by
    intro i j _hi _hj h
    exact h
  endpoints := by native_decide
  unit_edges := by
    intro e he
    exact Or.inl he

/-- The candidate geometry consumes exactly the generated SAT edge list. -/
theorem temp_edges_match_generated : tempCubeGeometry.edges = chi6cube_edges := rfl

/-- Candidate-shaped no-5 witness routed through the generic cube-cover adapter. -/
def tempCubeNoFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat tempUnit :=
  tempCubeGeometry.noFiveWitnessOfCubeCoverUnsat
    chi6cube_cubes
    (by simpa [tempCubeGeometry] using chi6cube_cube_cover)
    (by
      intro cube hcube
      simpa [tempCubeGeometry] using chi6cube_cube_unsat cube hcube)

/-- The finite non-Euclidean candidate attachment smoke. -/
theorem temp_cube_cover_candidate_contract_smoke :
    ¬ Nonempty (PlaneColouring Nat tempUnit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction tempCubeNoFiveWitness

#print axioms temp_edges_match_generated
#print axioms tempUnit_symm
#print axioms tempCubeNoFiveWitness
#print axioms temp_cube_cover_candidate_contract_smoke

end SounioChi6TempCubeCoverCandidateGate
EOF

rg -q 'temp_edges_match_generated' "$CAND_MOD"
rg -q 'noFiveWitnessOfCubeCoverUnsat' "$CAND_MOD"
rg -q 'chi6cube_cube_cover' "$CAND_MOD"
rg -q 'chi6cube_cube_unsat' "$CAND_MOD"
if rg -q '\b(sorry|admit)\b' "$CAND_MOD"; then
  echo "error: temp cube-cover candidate module contains incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$SAT_MOD" > "$WORK/sat_lean.log" 2>&1
  "$LOCK" "$LAKE" env lean "$CAND_MOD" > "$WORK/cand_lean.log" 2>&1
)
if rg -q 'sorryAx' "$WORK/sat_lean.log" "$WORK/cand_lean.log"; then
  echo "error: temp cube-cover candidate gate depends on sorryAx" >&2
  exit 1
fi

echo "chi6_cube_cover_temp_candidate_gate: PASS"
