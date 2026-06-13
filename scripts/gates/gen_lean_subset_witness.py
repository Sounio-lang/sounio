#!/usr/bin/env python3
"""Generate Lean witness from erdos90_subset_export.sio stdout."""
import re
import sys
from pathlib import Path

def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/subset_witness.log")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "formal/lean4/SounioErdos90SubsetWitness.lean"
    )
    text = log_path.read_text()
    m = re.search(r"edges=(\d+)", text)
    if not m:
        print("missing edges=", file=sys.stderr)
        return 1
    edges = int(m.group(1))
    unit_m = re.search(r"unitN=(\d+)", text)
    unit_n = int(unit_m.group(1)) if unit_m else 5
    seed_m = re.search(r"EXPORT seed=(\d+)", text)
    seed = int(seed_m.group(1)) if seed_m else 0
    winr_m = re.search(r"winR=(\d+)", text)
    win_r = int(winr_m.group(1)) if winr_m else 0
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
    grid288 = 288
    beats_harb = edges > harb100
    beats_grid = edges > grid288
    extra = ""
    if beats_harb:
        extra += """
theorem subset_n100_beats_harb : harb 100 < countGridUnit5 witness100 := by
  native_decide
"""
    if beats_grid:
        extra += """
theorem subset_n100_beats_grid10x10 : 288 < countGridUnit5 witness100 := by
  native_decide
"""
    lean = f"""/-!
# Erdős [90] — ℤ² subset witness at n=100 (machine-checked)

Generated from `stdlib/research/erdos90_subset_export.sio` via
`scripts/gates/gen_lean_subset_witness.py`. Densest-k-subgraph hill-climb on a
ℤ² pool (seed={seed}, winR={win_r}); unit edge iff `(dx)²+(dy)² = {unit_n}.
Honest finite witness only — not a claim of global optimality.
-/

set_option maxHeartbeats 1000000

namespace Sounio.Erdos90Subset

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

theorem subset_n100_count : countGridUnit{unit_n} witness100 = {edges} := by
  native_decide
{extra}
theorem subset_n100_meets_harb : harb 100 ≤ countGridUnit{unit_n} witness100 := by
  native_decide

theorem subset_n100_meets_grid10x10 : 288 ≤ countGridUnit{unit_n} witness100 := by
  native_decide

/-- Explicit lower bound: u(100) ≥ {edges} (ℤ² subset search, unit²={unit_n}). -/
theorem u100_subset_lower_bound : {edges} ≤ countGridUnit{unit_n} witness100 := by
  native_decide

end Sounio.Erdos90Subset
"""
    out_path.write_text(lean)
    print(f"wrote {out_path} ({len(pts)} pts, edges={edges}, N={unit_n})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())