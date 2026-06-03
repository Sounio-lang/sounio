#!/usr/bin/env bash
# Temp-only gate for the reflected triangle-precolour certificate producer.
#
# This exercises the SB_MODE=1 path with K6/k=5 and checks that the generated
# Lean file uses `colourCNFsb5` plus the unconditional WLOG bridge. It remains a
# finite SAT smoke, not a Euclidean chi>=6 witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
OUT_LEAN="$WORK/SounioSatK65SBReflect.lean"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
if [[ -z "$LAKE" ]]; then
  echo "error: lake not found; set LAKE=/path/to/lake" >&2
  exit 127
fi
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

mkdir -p "$WORK"
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

echo "graph_reflect_certificate_sb: workdir=$WORK"
rg -q '^e 1 2$' "$WORK/k6.edge"
rg -q '^e 1 3$' "$WORK/k6.edge"
rg -q '^e 2 3$' "$WORK/k6.edge"

SB_MODE=1 WORK="$WORK" \
  "$ROOT/examples/erdos/make_graph_reflect_certificate.sh" \
    "$WORK/k6.edge" 5 "$OUT_LEAN" k65sb SounioSatK65SBReflect "$WORK" \
  > "$WORK/producer.log"

[[ -s "$WORK/souc_sat.stdout" ]] || { echo "error: missing souc_sat.stdout" >&2; exit 1; }
rg -q '^\[precolour triangle 0,1,2\]$' "$WORK/souc_sat.stdout"
rg -q '^sb_mode=1$' "$WORK/producer.log"
rg -q '^sb_triangle=0,1,2$' "$WORK/producer.log"
rg -q '^import SounioSatColouringSB$' "$OUT_LEAN"
edge_count="$(rg -c '^[[:space:]]+\([0-5], [0-5]\),' "$OUT_LEAN")"
if [[ "$edge_count" != "15" ]]; then
  echo "error: expected 15 generated K6 edges, got $edge_count" >&2
  exit 1
fi
rg -q '^[[:space:]]+\(0, 1\),' "$OUT_LEAN"
rg -q '^[[:space:]]+\(4, 5\),' "$OUT_LEAN"
rg -q '^def k65sb_cnf : CNF Nat := colourCNFsb5 0 1 2 6 k65sb_edges$' "$OUT_LEAN"
rg -q '^theorem k65sb_unsat : k65sb_cnf\.Unsat :=' "$OUT_LEAN"
rg -q '^theorem k65sb_not_colourable :' "$OUT_LEAN"
rg -q 'not_colourable5_of_unsat_tri' "$OUT_LEAN"

(
  cd "$ROOT/formal/lean4"
  "$LAKE" build SounioSatColouringSB SounioSatReflect
  "$LAKE" env lean "$OUT_LEAN"
)

if rg -q '\b(sorry|admit)\b' "$OUT_LEAN"; then
  echo "error: sorry/admit found in generated SB reflect module" >&2
  exit 1
fi

sha256sum "$OUT_LEAN"
echo "graph_reflect_certificate_sb: PASS"
