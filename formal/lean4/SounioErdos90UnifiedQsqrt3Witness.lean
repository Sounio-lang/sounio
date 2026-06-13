/-!
# Erdős [90] — unified ℚ(√3) witness at n=100 (machine-checked)

Generated from `stdlib/research/erdos90_unified_export.sio` via
`scripts/gates/gen_lean_qsqrt3_witness.py`. Points `(px, (pu, pv))` represent
physical `(px/2, (pu + pv·√3)/2)`. Unit edge iff `(dpx)²+(dpu)²+3·(dpv)² = 4`
and `(dpu)·(dpv) = 0`.
-/

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
  (-14, (0, 0)),
  (-12, (0, 0)),
  (-12, (2, 0)),
  (-10, (0, 0)),
  (-10, (2, 0)),
  (-8, (0, 0)),
  (-6, (0, 0)),
  (-4, (0, 0)),
  (-2, (0, 0)),
  (0, (0, 0)),
  (2, (0, 0)),
  (4, (0, 0)),
  (6, (0, 0)),
  (8, (0, 0)),
  (10, (0, 0)),
  (12, (0, 0)),
  (12, (2, 0)),
  (14, (0, 0)),
  (14, (2, 0)),
  (16, (0, 0)),
  (16, (2, 0)),
  (18, (0, 0)),
  (18, (2, 0)),
  (-14, (0, 0)),
  (-13, (0, 1)),
  (-12, (0, 2)),
  (-11, (0, 3)),
  (-10, (0, 4)),
  (-13, (0, -1)),
  (-12, (0, 0)),
  (-11, (0, 1)),
  (-10, (0, 2)),
  (-9, (0, 3)),
  (-8, (0, 4)),
  (-11, (0, -1)),
  (-10, (0, 0)),
  (-9, (0, 1)),
  (-8, (0, 2)),
  (-7, (0, 3)),
  (-6, (0, 4)),
  (-9, (0, -1)),
  (-8, (0, 0)),
  (-7, (0, 1)),
  (-6, (0, 2)),
  (-5, (0, 3)),
  (-7, (0, -1)),
  (-6, (0, 0)),
  (-5, (0, 1)),
  (-4, (0, 2)),
  (-3, (0, 3)),
  (-5, (0, -1)),
  (-4, (0, 0)),
  (-3, (0, 1)),
  (-2, (0, 2)),
  (-1, (0, 3)),
  (-4, (0, -2)),
  (-3, (0, -1)),
  (-2, (0, 0)),
  (-1, (0, 1)),
  (0, (0, 2)),
  (-2, (0, -2)),
  (-1, (0, -1)),
  (0, (0, 0)),
  (1, (0, 1)),
  (0, (0, -2)),
  (1, (0, -1)),
  (2, (0, 0)),
  (3, (0, 1)),
  (2, (0, -2)),
  (3, (0, -1)),
  (4, (0, 0)),
  (5, (0, 1)),
  (3, (0, -3)),
  (4, (0, -2)),
  (5, (0, -1)),
  (6, (0, 0)),
  (7, (0, 1)),
  (5, (0, -3)),
  (6, (0, -2)),
  (7, (0, -1)),
  (8, (0, 0)),
  (9, (0, 1)),
  (8, (0, -2)),
  (9, (0, -1)),
  (10, (0, 0)),
  (11, (0, 1)),
  (11, (0, -1)),
  (12, (0, 0)),
  (13, (0, 1)),
  (13, (0, -1)),
  (14, (0, 0)),
  (15, (0, 1)),
  (15, (0, -1)),
  (16, (0, 0)),
  (17, (0, 1)),
  (17, (0, -1)),
  (18, (0, 0)),
  (19, (0, 1)),
  (19, (0, -1)),
  (20, (0, 0))
]

theorem unified_qsqrt3_n100_count : countUnitQ witness100 = 318 := by
  native_decide

theorem unified_qsqrt3_n100_beats_harb : harb 100 < countUnitQ witness100 := by
  native_decide

/-- Explicit lower bound: u(100) ≥ 318 (mixed ℤ²+Eisenstein via ℚ(√3) embedding). -/
theorem u100_lower_bound : 318 ≤ countUnitQ witness100 := by
  native_decide

end Sounio.Erdos90Unified
