#!/usr/bin/env python3
"""Generate Lean witness from erdos90_disk225_export.sio stdout."""
import re
import sys
from pathlib import Path

def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/disk225_witness.log")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "formal/lean4/SounioErdos90Disk225Witness.lean"
    )
    text = log_path.read_text()
    m = re.search(r"edges=(\d+)", text)
    if not m:
        print("missing edges=", file=sys.stderr)
        return 1
    edges = int(m.group(1))
    unit_n = 25
    rr_m = re.search(r"disk rr=(\d+)", text)
    rr = int(rr_m.group(1)) if rr_m else 72
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
# Erdős [90] — compact ℤ² disk witness at n=225, u(225) ≥ {edges} (machine-checked)

Generated from `stdlib/research/erdos90_disk225_export.sio`.
Disk rr={rr}, unit²={unit_n}. Full compact disk (not k-subset). Below subset witness 856.
-/

namespace Sounio.Erdos90Disk225

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

theorem disk_n225_count : countGridUnit25 witness225 = {edges} := by
  native_decide

theorem disk_n225_beats_grid15x15 : 828 < countGridUnit25 witness225 := by
  native_decide

theorem disk_n225_below_subset : countGridUnit25 witness225 < 856 := by
  native_decide

theorem disk_n225_meets_harb : harb 225 ≤ countGridUnit25 witness225 := by
  native_decide

theorem u225_disk_lower_bound : {edges} ≤ countGridUnit25 witness225 := by
  native_decide

end Sounio.Erdos90Disk225
"""
    out_path.write_text(lean)
    print(f"wrote {out_path} ({len(pts)} pts, edges={edges})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())