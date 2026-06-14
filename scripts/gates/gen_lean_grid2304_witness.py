#!/usr/bin/env python3
"""Generate Lean witness from erdos90_grid2304_export.sio stdout."""
import re
import sys
from pathlib import Path

def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/grid2304_witness.log")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "formal/lean4/SounioErdos90Grid2304Witness.lean"
    )
    text = log_path.read_text()
    m = re.search(r"edges=(\d+)", text)
    if not m:
        print("missing edges=", file=sys.stderr)
        return 1
    edges = int(m.group(1))
    unit_n = 25
    pts = []
    for x, y in re.findall(
        r"WITNESS \d+\s+x=(-?\d+)\s+y=(-?\d+)", text, flags=re.MULTILINE
    ):
        pts.append((int(x), int(y)))
    if len(pts) != 2304:
        print(f"expected 2304 points, got {len(pts)}", file=sys.stderr)
        return 1
    if len(set(pts)) != 2304:
        print(f"witness has duplicate coordinates: {len(set(pts))} unique", file=sys.stderr)
        return 1
    entries = ",\n  ".join(f"({x}, {y})" for x, y in pts)
    lean = f"""/-!
# Erdős [90] — ℤ² grid witness at n=2304 (machine-checked)

Generated from `stdlib/research/erdos90_grid2304_export.sio`. Full 48x48 grid; unit²={unit_n}.
-/

set_option maxHeartbeats 11520000

namespace Sounio.Erdos90Grid2304

def gridUnitSq (n : Nat) (p q : Int × Int) : Int :=
  let dx := p.1 - q.1
  let dy := p.2 - q.2
  dx * dx + dy * dy

def countGridUnit25 (pts : List (Int × Int)) : Nat :=
  ((pts.zipIdx).flatMap (fun (p, i) => (pts.zipIdx).filterMap (fun (q, j) =>
    if i < j && gridUnitSq 25 p q == 25 then some (1 : Nat) else none))).length

def isqrt (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc k => if k * k ≤ n then k else acc) 0

def harb (n : Nat) : Nat :=
  let s := isqrt (12 * n - 3)
  3 * n - (if s * s < 12 * n - 3 then s + 1 else s)

def witness2304 : List (Int × Int) := [
  {entries}
]

theorem grid_n2304_count : countGridUnit25 witness2304 = {edges} := by
  native_decide

theorem grid_n2304_beats_harb : harb 2304 < countGridUnit25 witness2304 := by
  native_decide

theorem grid_n2304_meets_harb : harb 2304 ≤ countGridUnit25 witness2304 := by
  native_decide

theorem u2304_grid_lower_bound : {edges} ≤ countGridUnit25 witness2304 := by
  native_decide

end Sounio.Erdos90Grid2304
"""
    out_path.write_text(lean)
    print(f"wrote {out_path} ({len(pts)} pts, edges={edges})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())