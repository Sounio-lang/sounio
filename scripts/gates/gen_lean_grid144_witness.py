#!/usr/bin/env python3
"""Generate Lean witness from erdos90_grid144_export.sio stdout."""
import re
import sys
from pathlib import Path

def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/grid144_witness.log")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "formal/lean4/SounioErdos90Grid144Witness.lean"
    )
    text = log_path.read_text()
    m = re.search(r"edges=(\d+)", text)
    if not m:
        print("missing edges=", file=sys.stderr)
        return 1
    edges = int(m.group(1))
    unit_m = re.search(r"unitN=(\d+)", text)
    unit_n = int(unit_m.group(1)) if unit_m else 25
    pts = []
    for x, y in re.findall(
        r"WITNESS \d+\s+x=(-?\d+)\s+y=(-?\d+)", text, flags=re.MULTILINE
    ):
        pts.append((int(x), int(y)))
    if len(pts) != 144:
        print(f"expected 144 points, got {len(pts)}", file=sys.stderr)
        return 1
    if len(set(pts)) != 144:
        print(f"witness has duplicate coordinates: {len(set(pts))} unique", file=sys.stderr)
        return 1
    entries = ",\n  ".join(f"({x}, {y})" for x, y in pts)
    harb144 = 390
    extra = ""
    if edges > harb144:
        extra += """
theorem grid_n144_beats_harb : harb 144 < countGridUnit25 witness144 := by
  native_decide
"""
    lean = f"""/-!
# Erdős [90] — ℤ² grid witness at n=144 (machine-checked)

Generated from `stdlib/research/erdos90_grid144_export.sio`. Full 12×12 grid in ℤ²;
unit edge iff `(dx)²+(dy)² = {unit_n}.
-/

set_option maxHeartbeats 1000000

namespace Sounio.Erdos90Grid144

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

def witness144 : List (Int × Int) := [
  {entries}
]

theorem grid_n144_count : countGridUnit{unit_n} witness144 = {edges} := by
  native_decide
{extra}
theorem grid_n144_meets_harb : harb 144 ≤ countGridUnit{unit_n} witness144 := by
  native_decide

/-- Explicit lower bound: u(144) ≥ {edges} (full 12×12 ℤ² grid, unit²={unit_n}). -/
theorem u144_grid_lower_bound : {edges} ≤ countGridUnit{unit_n} witness144 := by
  native_decide

end Sounio.Erdos90Grid144
"""
    out_path.write_text(lean)
    print(f"wrote {out_path} ({len(pts)} pts, edges={edges}, N={unit_n})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())