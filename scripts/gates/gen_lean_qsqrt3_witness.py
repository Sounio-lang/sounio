#!/usr/bin/env python3
"""Generate Lean witness from erdos90_unified_export.sio stdout."""
import re
import sys
from pathlib import Path

def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/witness.log")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "formal/lean4/SounioErdos90UnifiedQsqrt3Witness.lean"
    )
    text = log_path.read_text()
    m = re.search(r"edges=(\d+)", text)
    if not m:
        print("missing edges=", file=sys.stderr)
        return 1
    edges = int(m.group(1))
    pts = []
    blocks = re.findall(
        r"WITNESS \d+\s+px=(-?\d+)\s+pu=(-?\d+)\s+pv=(-?\d+)",
        text,
        flags=re.MULTILINE,
    )
    for px, pu, pv in blocks:
        pts.append((int(px), int(pu), int(pv)))
    if len(pts) != 100:
        print(f"expected 100 points, got {len(pts)}", file=sys.stderr)
        return 1
    if len(set(pts)) != 100:
        print(f"witness has duplicate coordinates: {len(set(pts))} unique", file=sys.stderr)
        return 1
    entries = ",\n  ".join(f"({px}, ({pu}, {pv}))" for px, pu, pv in pts)
    harb100 = 265
    beats_harb = edges > harb100
    beats_block = ""
    if beats_harb:
        beats_block = f"""
theorem unified_qsqrt3_n100_beats_harb : harb 100 < countUnitQ witness100 := by
  native_decide
"""
    lean = f"""/-!
# Erdős [90] — unified ℚ(√3) witness at n=100 (machine-checked)

Generated from `stdlib/research/erdos90_unified_export.sio` via
`scripts/gates/gen_lean_qsqrt3_witness.py`. Points `(px, (pu, pv))` represent
physical `(px/2, (pu + pv·√3)/2)`. Unit edge iff `(dpx)²+(dpu)²+3·(dpv)² = 4`
and `(dpu)·(dpv) = 0`.
-/

set_option maxHeartbeats 1000000

namespace Sounio.Erdos90Unified

def QSqrt3Pt := Int × (Int × Int)

def unitSq (p q : QSqrt3Pt) : Int :=
  let dpx := p.1 - q.1
  let dpu := p.2.1 - q.2.1
  let dpv := p.2.2 - q.2.2
  if dpu * dpv != 0 then 0 else dpx * dpx + dpu * dpu + 3 * dpv * dpv

def countUnitQ (pts : List QSqrt3Pt) : Nat :=
  ((pts.zipIdx).flatMap (fun (p, i) => (pts.zipIdx).filterMap (fun (q, j) =>
    if i < j && unitSq p q == 4 then some (1 : Nat) else none))).length

def isqrt (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc k => if k * k ≤ n then k else acc) 0

def harb (n : Nat) : Nat :=
  let s := isqrt (12 * n - 3)
  3 * n - (if s * s < 12 * n - 3 then s + 1 else s)

def witness100 : List QSqrt3Pt := [
  {entries}
]

theorem unified_qsqrt3_n100_count : countUnitQ witness100 = {edges} := by
  native_decide
{beats_block}
theorem unified_qsqrt3_n100_meets_harb : harb 100 ≤ countUnitQ witness100 := by
  native_decide

/-- Explicit finite witness: {edges} unit pairs among 100 distinct ℚ(√3) lattice points. -/
theorem u100_witness_edges : countUnitQ witness100 = {edges} := by
  native_decide

end Sounio.Erdos90Unified
"""
    out_path.write_text(lean)
    print(f"wrote {out_path} ({len(pts)} pts, edges={edges})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())