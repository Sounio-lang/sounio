#!/usr/bin/env python3
"""Generate Lean witness from erdos90_subset225_export.sio stdout."""
import re
import sys
from pathlib import Path

def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/subset225.log")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "formal/lean4/SounioErdos90Subset225Witness.lean"
    )
    text = log_path.read_text()
    m = re.search(r"edges=(\d+)", text)
    if not m:
        print("missing edges=", file=sys.stderr)
        return 1
    edges = int(m.group(1))
    unit_n = 25
    seed_m = re.search(r"EXPORT seed=(\d+)", text)
    seed = int(seed_m.group(1)) if seed_m else 0
    winr_m = re.search(r"winR=(\d+)", text)
    win_r = int(winr_m.group(1)) if winr_m else 0
    pts = []
    for x, y in re.findall(
        r"WITNESS \d+\s+x=(-?\d+)\s+y=(-?\d+)", text, flags=re.MULTILINE
    ):
        pts.append((int(x), int(y)))
    if len(pts) != 225:
        print(f"expected 225 points, got {len(pts)}", file=sys.stderr)
        return 1
    if len(set(pts)) != 225:
        print(f"witness has duplicate coordinates: {len(set(pts))} unique", file=sys.stderr)
        return 1
    entries = ",\n  ".join(f"({x}, {y})" for x, y in pts)
    lean = f"""/-!
# Erdős [90] — ℤ² subset witness at n=225, u(225) ≥ {edges} (machine-checked)

Generated from `stdlib/research/erdos90_subset225_export.sio`.
Seed={seed}, winR={win_r}, unit²={unit_n}. Not a claim of global optimality.
-/

namespace Sounio.Erdos90Subset225

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

def witness225 : List (Int × Int) := [
  {entries}
]

theorem subset_n225_count : countGridUnit25 witness225 = {edges} := by
  native_decide

theorem subset_n225_beats_grid15x15 : 828 < countGridUnit25 witness225 := by
  native_decide

theorem subset_n225_meets_harb : harb 225 ≤ countGridUnit25 witness225 := by
  native_decide

theorem u225_subset_lower_bound : {edges} ≤ countGridUnit25 witness225 := by
  native_decide

end Sounio.Erdos90Subset225
"""
    out_path.write_text(lean)
    print(f"wrote {out_path} ({len(pts)} pts, edges={edges})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())