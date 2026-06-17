#!/usr/bin/env bash
# Produce a concrete, non-promotable chi>=6 candidate-shaped manifest.
#
# This wraps the existing K6/k=5 SB reflected-certificate smoke pipeline and
# emits candidate.manifest with real hashes for the edge, CNF, LRAT, and Lean SAT
# module artifacts. It is deliberately finite-only and geometry-free:
# promotable=0, geometry_proof_type=none.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$(mktemp -d)}"
SOUC="${SOUC:-$ROOT/bin/souc}"
LAKE="${LAKE:-$(command -v lake || true)}"
K=5
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi

[[ -x "$SOUC" ]] || { echo "error: SOUC is not executable: $SOUC" >&2; exit 127; }
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }

mkdir -p "$WORK"
EDGE="$WORK/k6.edge"
OUT_LEAN="$WORK/SounioSatK65SmokeReflect.lean"
MANIFEST="$WORK/candidate.manifest"

cat > "$EDGE" <<'EOF'
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

edge_has() {
  local u="$(( $1 + 1 ))"
  local v="$(( $2 + 1 ))"
  awk -v u="$u" -v v="$v" '
    $1 == "e" && (($2 == u && $3 == v) || ($2 == v && $3 == u)) { found = 1 }
    END { exit found ? 0 : 1 }
  ' "$EDGE"
}

for pair in "0 1" "0 2" "1 2"; do
  read -r a b <<< "$pair"
  if ! edge_has "$a" "$b"; then
    echo "error: generated K6 smoke edge file is missing SB triangle edge $a-$b" >&2
    exit 1
  fi
done

echo "chi6_smoke_candidate_manifest: workdir=$WORK"
SB_MODE=1 SOUC="$SOUC" WORK="$WORK" \
  "$ROOT/examples/erdos/make_graph_reflect_certificate.sh" \
    "$EDGE" "$K" "$OUT_LEAN" k65smoke SounioSatK65SmokeReflect "$WORK" \
  > "$WORK/producer.log"

if ! (
  cd "$ROOT/formal/lean4"
  "$LAKE" build SounioSatColouringSB
  "$LAKE" env lean "$OUT_LEAN"
) > "$WORK/lean_build.log" 2>&1; then
  echo "error: Lean smoke-certificate build failed; see $WORK/lean_build.log" >&2
  exit 1
fi

if grep -Eq '(^|[^[:alnum:]_])(sorry|admit)([^[:alnum:]_]|$)' "$OUT_LEAN"; then
  echo "error: generated Lean SAT smoke module contains sorry/admit" >&2
  exit 1
fi
if ! grep -Fq "colourCNFsb5 0 1 2 6 " "$OUT_LEAN"; then
  echo "error: generated Lean SAT smoke module does not match triangle_sb=0,1,2" >&2
  exit 1
fi

sha() {
  [[ -f "$1" ]] || { echo "error: file not found for hashing: $1" >&2; exit 1; }
  sha256sum "$1" | awk '{print $1}'
}

GENERATOR_COMMIT="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null || echo UNKNOWN)"
EDGE_SHA="$(sha "$EDGE")"
CNF_SHA="$(sha "$WORK/souc_sat_worker.cnf")"
LRAT_SHA="$(sha "$WORK/k65smoke.lrat")"
LEAN_SAT_SHA="$(sha "$OUT_LEAN")"

cat > "$MANIFEST" <<EOF
candidate_manifest_version=1
promotable=0
candidate_id=k6_sb5_smoke_not_planar
n=6
m=15
k=$K
edge_path=k6.edge
edge_sha256=$EDGE_SHA
cnf_path=souc_sat_worker.cnf
cnf_sha256=$CNF_SHA
drat_or_lrat_path=k65smoke.lrat
drat_or_lrat_sha256=$LRAT_SHA
lean_sat_module_path=SounioSatK65SmokeReflect.lean
lean_sat_module_sha256=$LEAN_SAT_SHA
geometry_module_path=NONE
geometry_module_sha256=NONE
geometry_proof_type=none
sat_proof_route=triangle_sb5_lrat
triangle_sb=0,1,2
generator_commit=$GENERATOR_COMMIT
producer_command=SB_MODE=1 examples/erdos/make_graph_reflect_certificate.sh k6.edge $K SounioSatK65SmokeReflect.lean k65smoke SounioSatK65SmokeReflect
lean_build_command=lake env lean SounioSatK65SmokeReflect.lean
offload_review_raw=NONE
offload_review_sha256=NONE
EOF

"$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh" "$MANIFEST" \
  | tee "$WORK/manifest_validator.log"

echo "manifest=$MANIFEST"
echo "edge=$EDGE"
echo "cnf=$WORK/souc_sat_worker.cnf"
echo "lrat=$WORK/k65smoke.lrat"
echo "lean_sat_module=$OUT_LEAN"
echo "chi6_smoke_candidate_manifest: PASS"
