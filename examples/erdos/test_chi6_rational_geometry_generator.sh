#!/usr/bin/env bash
# Gate the data-driven rational-coordinate Euclidean geometry generator.
#
# This is geometry-only. It proves that a DIMACS edge list plus exact rational
# coordinates can generate a Lean `EuclideanNatEdgeExactGeometry` object and an
# edge-sync theorem, but it does not attach SAT/LRAT and does not claim chi>=6.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEN="$ROOT/examples/erdos/gen_lean_rational_geometry.py"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
mkdir -p "$WORK"

EDGE="$WORK/square.edge"
COORD="$WORK/square.coords.csv"
OUT="$WORK/SounioChi6RationalGeometrySmoke.lean"
VERIFY="$WORK/VerifyRationalGeometrySmoke.lean"
LEANLIB="$WORK/leanlib"

cat > "$EDGE" <<'EOF'
p edge 4 4
e 1 2
e 2 3
e 3 4
e 4 1
EOF

cat > "$COORD" <<'EOF'
id,x,y
0,0,0
1,1,0
2,1,1
3,0,1
EOF

echo "chi6_rational_geometry_generator_gate: workdir=$WORK"
python3 "$GEN" "$EDGE" "$COORD" "$OUT" \
  --module SounioChi6RationalGeometrySmoke \
  --namespace SounioChi6RationalGeometrySmoke \
  --prefix square > "$WORK/gen.out"
rg -q '^gen_lean_rational_geometry v1$' "$WORK/gen.out"
rg -q '^geometry_claim=exact_rational_squared_distance_edges_only$' "$WORK/gen.out"
rg -q '^sat_claim=none$' "$WORK/gen.out"
rg -q '^chromatic_claim=none$' "$WORK/gen.out"
rg -q '^promotable=0$' "$WORK/gen.out"
rg -q '^status=lean_rational_geometry_emitted$' "$WORK/gen.out"

rg -q '^def euclideanGeometry : EuclideanNatEdgeExactGeometry 4' "$OUT"
rg -q '^theorem edgesSyncSelf : euclideanGeometry\.exact\.edges = edges := rfl$' "$OUT"
rg -q '^abbrev realUnit : Real × Real → Real × Real → Prop := standardRealPlaneUnit$' "$OUT"
rg -q '^theorem realUnit_iff_standard :' "$OUT"
rg -q '^theorem realUnitEdges : ∀ e ∈ edges, realUnit \(realEmb e\.1\) \(realEmb e\.2\) := by$' "$OUT"
if rg -q '\b(sorry|admit)\b|#exit|#eval|#check' "$OUT"; then
  echo "error: generated rational geometry module contains forbidden proof/debug marker" >&2
  exit 1
fi

(
  cd "$ROOT/formal/lean4"
  ../../scripts/dev/souc-build-lock.sh "$LAKE" env lean "$OUT" > "$WORK/lean.out" 2> "$WORK/lean.err"
)
if rg -q 'error:' "$WORK/lean.out" "$WORK/lean.err"; then
  cat "$WORK/lean.out" "$WORK/lean.err" >&2
  exit 1
fi
if rg -q 'sorryAx' "$WORK/lean.out" "$WORK/lean.err"; then
  cat "$WORK/lean.out" "$WORK/lean.err" >&2
  echo "error: generated rational geometry module depends on sorryAx" >&2
  exit 1
fi

mkdir -p "$LEANLIB"
(
  cd "$ROOT/formal/lean4"
  ../../scripts/dev/souc-build-lock.sh "$LAKE" env lean \
    -R "$WORK" \
    -o "$LEANLIB/SounioChi6RationalGeometrySmoke.olean" \
    "$OUT" > "$WORK/olean.out" 2> "$WORK/olean.err"
)
if rg -q 'error:' "$WORK/olean.out" "$WORK/olean.err"; then
  cat "$WORK/olean.out" "$WORK/olean.err" >&2
  exit 1
fi

cat > "$VERIFY" <<EOF
import SounioChi6RationalGeometrySmoke

open UnitDistanceChromatic
open SounioSqrt.RealCauchyField
open UnitDistanceChromatic.SounioChi6RationalGeometrySmoke

#check (euclideanGeometry : EuclideanNatEdgeExactGeometry 4 (Fin 4) unit)
#check (plane : ExactSquaredDistancePlane (Fin 4) unit)
#check (edgesSyncSelf : euclideanGeometry.exact.edges = edges)
#check (realUnit : Real × Real → Real × Real → Prop)
#check (realUnit_iff_standard :
  ∀ p q : Real × Real, realUnit p q ↔
    standardRealPlaneDist2 p q = qR (1 : Rat))
#check (realUnitEdges : ∀ e ∈ edges, realUnit (realEmb e.1) (realEmb e.2))

example : euclideanGeometry.exact.edges = [(0, 1), (1, 2), (2, 3), (0, 3)] := rfl
example : euclideanGeometry.exact.emb = emb := rfl
example : euclideanGeometry.plane.dist2 = dist2 := rfl
example : realUnit (realEmb 0) (realEmb 1) := by
  exact realUnitEdges (0, 1) (by decide)
example : realDist2 (realEmb 0) (realEmb 2) = qR (2 : Rat) := by
  unfold realEmb realPoint realPointX realPointY
  simp
  rw [realDist2_qR]
  congr 1
  native_decide
example : unit (emb 0) (emb 1) := by
  simp [emb, unit, dist2, pointX, pointY]
  native_decide
example : unit (emb 1) (emb 2) := by
  simp [emb, unit, dist2, pointX, pointY]
  native_decide
example : dist2 (emb 0) (emb 2) = (2 : Rat) := by
  simp [emb, dist2, pointX, pointY]
  native_decide
example : ¬ unit (emb 0) (emb 0) := by
  exact unit_irrefl (emb 0)
example : ∀ e ∈ exactGeometry.edges, unit (emb e.1) (emb e.2) := by
  intro e he
  exact exactGeometry.unit_edges e he
EOF

if rg -q '\b(sorry|admit)\b|#exit' "$VERIFY"; then
  echo "error: generated verifier contains sorry/admit/#exit" >&2
  exit 1
fi

(
  cd "$ROOT/formal/lean4"
  LEAN_PATH="$LEANLIB${LEAN_PATH:+:$LEAN_PATH}" \
    ../../scripts/dev/souc-build-lock.sh "$LAKE" env lean "$VERIFY" \
      > "$WORK/verify.out" 2> "$WORK/verify.err"
)
if rg -q 'error:' "$WORK/verify.out" "$WORK/verify.err"; then
  cat "$WORK/verify.out" "$WORK/verify.err" >&2
  exit 1
fi
rg -q '^euclideanGeometry : EuclideanNatEdgeExactGeometry 4' "$WORK/verify.out"
rg -qF 'plane : ExactSquaredDistancePlane (Fin 4) unit' "$WORK/verify.out"
rg -qF 'realUnit : Real × Real → Real × Real → Prop' "$WORK/verify.out"
rg -q '^realUnitEdges : ∀ \(e : Nat × Nat\), e ∈ edges → realUnit \(realEmb e\.fst\) \(realEmb e\.snd\)$' \
  "$WORK/verify.out"

BAD_COORD="$WORK/bad.coords.csv"
cp "$COORD" "$BAD_COORD"
sed -i 's/^1,1,0/1,2,0/' "$BAD_COORD"
if python3 "$GEN" "$EDGE" "$BAD_COORD" "$WORK/bad.lean" \
    --module BadRationalGeometry > "$WORK/bad.out" 2>&1; then
  echo "error: rational geometry generator accepted a non-unit edge" >&2
  exit 1
fi
rg -q 'edge 0,1 has dist2=4, expected 1' "$WORK/bad.out"

echo "chi6_rational_geometry_generator_gate: PASS"
