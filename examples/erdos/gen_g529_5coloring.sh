#!/usr/bin/env bash
# examples/erdos/gen_g529_5coloring.sh
#
# Reproducible pipeline for χ(G₅₂₉) = 5.
#
# Steps:
#   1. Compile souc_sat.sio with the legacy self-hosted compiler.
#   2. Run souc_sat on G₅₂₉ with k=5 and triangle symmetry-breaking (SB=1).
#   3. Extract the SAT 5-colouring.
#   4. Emit formal/lean4/SounioDeGreyChi529Exact.lean with the colouring table.
#   5. Build the Lean library SounioDeGreyChi529Exact.
#
# The resulting theorem DeGrey529.g529_chi_eq_5 states:
#   - there exists a proper 5-colouring of G₅₂₉, and
#   - no proper 4-colouring exists (uses SounioSatG529.g529_not_colourable).
#
# All arithmetic is exact integer; no floats, no Mathlib, no sorry.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUNIO_SOUC_BIN:-$ROOT/artifacts/self-hosted/souc-self-hosted-x86_64}"
ELAN_HOME="${ELAN_HOME:-$ROOT/formal/lean4/.elan}"
export PATH="$ELAN_HOME/bin:$PATH"

WORK="${WORK:-$(mktemp -d)}"
trap 'rm -rf "$WORK"' EXIT INT TERM

EDGEFILE="$ROOT/examples/erdos/data/degrey_529.edge"
LEANOUT="$ROOT/formal/lean4/SounioDeGreyChi529Exact.lean"

[[ -x "$SOUC" ]] || { echo "error: SOUC is not executable: $SOUC" >&2; exit 2; }
command -v lake >/dev/null 2>&1 || { echo "error: lake not in PATH (set ELAN_HOME)" >&2; exit 2; }
[[ -s "$EDGEFILE" ]] || { echo "error: missing edge file $EDGEFILE" >&2; exit 2; }

echo "[1/4] compile souc_sat.sio ..."
"$SOUC" "$ROOT/examples/erdos/souc_sat.sio" "$WORK/souc_sat.elf"
chmod +x "$WORK/souc_sat.elf"

echo "[2/4] solve 5-colouring of G529 (SAT expected, rc=1) ..."
"$WORK/souc_sat.elf" 0 5 1 1 "$EDGEFILE" > "$WORK/souc_sat.stdout" 2>&1 || true
if ! grep -q 'SAT colouring=' "$WORK/souc_sat.stdout"; then
  echo "error: souc_sat did not return a 5-colouring" >&2
  cat "$WORK/souc_sat.stdout" >&2
  exit 1
fi

echo "[3/4] extract and validate 5-colouring ..."
python3 - "$WORK/souc_sat.stdout" "$EDGEFILE" "$LEANOUT" <<'PY'
import sys, re
stdout_path, edge_path, lean_out = sys.argv[1:]
with open(stdout_path) as f:
    text = f.read()
m = re.search(r'SAT colouring=(.+?)\]', text)
if not m:
    print("error: no SAT colouring line found", file=sys.stderr)
    sys.exit(1)
parts = m.group(1).split(',')
colors = []
for p in parts:
    v, c = p.split(':')
    colors.append((int(v), int(c)))
assert len(colors) == 529, f"expected 529 vertices, got {len(colors)}"
for i, (v, c) in enumerate(colors):
    assert v == i, f"vertex out of order at index {i}: {v}"
    assert 0 <= c <= 4, f"invalid colour {c} at vertex {v}"
with open(edge_path) as f:
    header = f.readline()
    edges = [tuple(int(x)-1 for x in line.split()[1:]) for line in f]
violations = [(u, v, colors[u][1]) for u, v in edges if colors[u][1] == colors[v][1]]
if violations:
    print(f"error: {len(violations)} monochromatic edges", file=sys.stderr)
    print(violations[:10], file=sys.stderr)
    sys.exit(1)
print(f"validated proper 5-colouring of G529 ({len(edges)} edges)")

lines = ['import SounioSatG529', '', 'namespace DeGrey529', '', 'def g529_color_table : Array (Fin 5) := #[']
for i in range(0, 529, 16):
    chunk = colors[i:i+16]
    vals = ', '.join(str(c) for _, c in chunk)
    sep = ',' if i + 16 < 529 else ''
    lines.append(f'  {vals}{sep}')
lines.append(']')
lines.append('')
lines.append('def g529_color (v : Fin 529) : Fin 5 :=')
lines.append('  g529_color_table.getD v.val 0')
lines.append('')
lines.append('theorem g529_proper_5colouring :')
lines.append('    ∀ e ∈ g529_edges, ∀ (h1 : e.1 < 529) (h2 : e.2 < 529),')
lines.append('      g529_color ⟨e.1, h1⟩ ≠ g529_color ⟨e.2, h2⟩ := by')
lines.append('  native_decide')
lines.append('')
lines.append('theorem g529_not_4colourable :')
lines.append('    ¬ ∃ f : Fin 529 → Fin 4,')
lines.append('        ∀ e ∈ g529_edges, ∀ (h1 : e.1 < 529) (h2 : e.2 < 529),')
lines.append('          f ⟨e.1, h1⟩ ≠ f ⟨e.2, h2⟩ := g529_not_colourable')
lines.append('')
lines.append('/- The chromatic number of the de Grey graph G529 is exactly 5. -/')
lines.append('theorem g529_chi_eq_5 :')
lines.append('    (∃ f : Fin 529 → Fin 5,')
lines.append('       ∀ e ∈ g529_edges, ∀ (h1 : e.1 < 529) (h2 : e.2 < 529),')
lines.append('         f ⟨e.1, h1⟩ ≠ f ⟨e.2, h2⟩) ∧')
lines.append('    (¬ ∃ f : Fin 529 → Fin 4,')
lines.append('       ∀ e ∈ g529_edges, ∀ (h1 : e.1 < 529) (h2 : e.2 < 529),')
lines.append('         f ⟨e.1, h1⟩ ≠ f ⟨e.2, h2⟩) :=')
lines.append('  ⟨⟨g529_color, g529_proper_5colouring⟩, g529_not_4colourable⟩')
lines.append('')
lines.append('#print axioms g529_chi_eq_5')
lines.append('')
lines.append('end DeGrey529')
lines.append('')

with open(lean_out, 'w') as f:
    f.write('\n'.join(lines))
print(f"wrote {lean_out}")
PY

echo "[4/4] build SounioDeGreyChi529Exact ..."
cd "$ROOT/formal/lean4"
lake build SounioDeGreyChi529Exact

echo "gen_g529_5coloring.sh: PASS — χ(G529) = 5 formalized in $LEANOUT"
