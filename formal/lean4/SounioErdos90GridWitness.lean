/-!
# Erdős [90] — ℤ² grid witness at n=100 (machine-checked)

Generated from `stdlib/research/erdos90_grid_export.sio` via
`scripts/gates/gen_lean_grid_witness.py`. Full 10×10 grid in ℤ²; unit edge iff
`(dx)²+(dy)² = 5.
-/

namespace Sounio.Erdos90Grid

def gridUnitSq (n : Nat) (p q : Int × Int) : Int :=
  let dx := p.1 - q.1
  let dy := p.2 - q.2
  dx * dx + dy * dy

def countGridUnit (n : Nat) (pts : List (Int × Int)) : Nat :=
  ((pts.zipIdx).flatMap (fun (p, i) => (pts.zipIdx).filterMap (fun (q, j) =>
    if i < j && gridUnitSq n p q == (n : Int) then some (1 : Nat) else none))).length

def countGridUnit5 := countGridUnit 5

def isqrt (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc k => if k * k ≤ n then k else acc) 0

def harb (n : Nat) : Nat :=
  let s := isqrt (12 * n - 3)
  3 * n - (if s * s < 12 * n - 3 then s + 1 else s)

def witness100 : List (Int × Int) := [
  (0, 0),
  (0, 1),
  (0, 2),
  (0, 3),
  (0, 4),
  (0, 5),
  (0, 6),
  (0, 7),
  (0, 8),
  (0, 9),
  (1, 0),
  (1, 1),
  (1, 2),
  (1, 3),
  (1, 4),
  (1, 5),
  (1, 6),
  (1, 7),
  (1, 8),
  (1, 9),
  (2, 0),
  (2, 1),
  (2, 2),
  (2, 3),
  (2, 4),
  (2, 5),
  (2, 6),
  (2, 7),
  (2, 8),
  (2, 9),
  (3, 0),
  (3, 1),
  (3, 2),
  (3, 3),
  (3, 4),
  (3, 5),
  (3, 6),
  (3, 7),
  (3, 8),
  (3, 9),
  (4, 0),
  (4, 1),
  (4, 2),
  (4, 3),
  (4, 4),
  (4, 5),
  (4, 6),
  (4, 7),
  (4, 8),
  (4, 9),
  (5, 0),
  (5, 1),
  (5, 2),
  (5, 3),
  (5, 4),
  (5, 5),
  (5, 6),
  (5, 7),
  (5, 8),
  (5, 9),
  (6, 0),
  (6, 1),
  (6, 2),
  (6, 3),
  (6, 4),
  (6, 5),
  (6, 6),
  (6, 7),
  (6, 8),
  (6, 9),
  (7, 0),
  (7, 1),
  (7, 2),
  (7, 3),
  (7, 4),
  (7, 5),
  (7, 6),
  (7, 7),
  (7, 8),
  (7, 9),
  (8, 0),
  (8, 1),
  (8, 2),
  (8, 3),
  (8, 4),
  (8, 5),
  (8, 6),
  (8, 7),
  (8, 8),
  (8, 9),
  (9, 0),
  (9, 1),
  (9, 2),
  (9, 3),
  (9, 4),
  (9, 5),
  (9, 6),
  (9, 7),
  (9, 8),
  (9, 9)
]

theorem grid_n100_count : countGridUnit5 witness100 = 288 := by
  native_decide

theorem grid_n100_beats_harb : harb 100 < countGridUnit5 witness100 := by
  native_decide

theorem grid_n100_meets_harb : harb 100 ≤ countGridUnit5 witness100 := by
  native_decide

theorem grid_n100_meets_full10x10 : 288 ≤ countGridUnit5 witness100 := by
  native_decide

/-- Explicit lower bound: u(100) ≥ 288 (full 10×10 ℤ² grid, unit²=5). -/
theorem u100_grid_lower_bound : 288 ≤ countGridUnit5 witness100 := by
  native_decide

end Sounio.Erdos90Grid
