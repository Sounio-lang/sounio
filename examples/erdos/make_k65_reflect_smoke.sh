#!/usr/bin/env bash
# Reproduce the finite K6/no-5 reflected-LRAT smoke module.
#
# This script is a producer only. The generated Lean module still has to be
# built, and soundness lives in Lean's verified LRAT checker, not in this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_LEAN="${1:-$ROOT/formal/lean4/SounioSatK65Reflect.lean}"
WORK="${WORK:-$(mktemp -d)}"

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

"$ROOT/examples/erdos/make_graph_reflect_certificate.sh" \
  "$WORK/k6.edge" 5 "$OUT_LEAN" k65 SounioSatK65Reflect "$WORK"
