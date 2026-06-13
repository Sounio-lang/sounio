#!/usr/bin/env python3
"""Generate Lean witness from erdos90_grid_export.sio stdout."""
import re
import sys
from pathlib import Path

def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/grid_witness.log")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "formal/lean4/SounioErdos90GridWitness.lean"
    )
    text = log_path.read_text()
    m = re.search(r"edges=(\d+)", text)
    if not m:
        print("missing edges=", file=sys.stderr)
        return 1
    edges = int(m.group(1))
    unit_m = re.search(r"unitN=(\d+)", text)
    unit_n = int(unit_m.group(1)) if unit_m else 5
    pts = []
    for x, y in re.findall(
        r"WITNESS \d+\s+x=(-?\d+)\s+y=(-?\d+)", text, flags=re.MULTILINE
    ):
        pts.append((int(x), int(y)))
    if len(pts) != 100:
        print(f"expected 100 points, got {len(pts)}", file=sys.stderr)
        return 1
    if len(set(pts)) != 100:
        print(f"witness has duplicate coordinates: {len(set(pts))} unique", file=sys.stderr)
        return 1
    entries = ",\n  ".join(f"({x}, {y})" for x, y in pts)
    harb100 = 265
    grid_full = 288
    beats_harb = edges > harb100
    beats_full = edges > grid_full
    extra = ""
    if beats_harb:
        extra += f"""
theorem grid_n100_beats_harb : harb 100 < countGridUnit{unit_n} witness100 := by
  native_decide
"""
    if beats_full:
        extra += f"""
theorem grid_n100_beats_full10x10 : 288 < countGridUnit{unit_n} witness100 := by
  native_decide
"""
    lean = f"""/-!
# Erdős [90] — ℤ² grid witness at n=100 (machine-checked)

Generated from `stdlib/research/erdos90_grid_export.sio` via
`scripts/gates/gen_lean_grid_witness.py`. Full 10×10 grid in ℤ²; unit edge iff
`(dx)²+(dy)² = {unit_n}.
-/

namespace Sounio.Erdos90Grid

def gridUnitSq (n : Nat) (p q : Int × Int) : Int :=
  let dx := p.1 - q.1
  let dy := p.2 - q.2
  dx * dx + dy * dy

def countGridUnit (n : Nat) (pts : List (Int × Int)) : Nat :=
  ((pts.zipIdx).flatMap (fun (p, i) => (pts.zipIdx).filterMap (fun (q, j) =>
    if i < j && gridUnitSq n p q == (n : Int) then some (1 : Nat) else none))).length

def countGridUnit{unit_n} := countGridUnit {unit_n}

def isqrt (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc k => if k * k ≤ n then k else acc) 0

def harb (n : Nat) : Nat :=
  let s := isqrt (12 * n - 3)
  3 * n - (if s * s < 12 * n - 3 then s + 1 else s)

def witness100 : List (Int × Int) := [
  {entries}
]

theorem grid_n100_count : countGridUnit{unit_n} witness100 = {edges} := by
  native_decide
{extra}
theorem grid_n100_meets_harb : harb 100 ≤ countGridUnit{unit_n} witness100 := by
  native_decide

theorem grid_n100_meets_full10x10 : 288 ≤ countGridUnit{unit_n} witness100 := by
  native_decide

/-- Explicit lower bound: u(100) ≥ {edges} (full 10×10 ℤ² grid, unit²={unit_n}). -/
theorem u100_grid_lower_bound : {edges} ≤ countGridUnit{unit_n} witness100 := by
  native_decide

end Sounio.Erdos90Grid
"""
    out_path.write_text(lean)
    print(f"wrote {out_path} ({len(pts)} pts, edges={edges}, N={unit_n})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())