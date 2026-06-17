/-!
# Erdős [90] — ℤ² grid witness at n=144 (machine-checked)

Generated from `stdlib/research/erdos90_grid144_export.sio`. Full 12×12 grid in ℤ²;
unit edge iff `(dx)²+(dy)² = 25.
-/

namespace Sounio.Erdos90Grid144

def gridUnitSq (n : Nat) (p q : Int × Int) : Int :=
  let dx := p.1 - q.1
  let dy := p.2 - q.2
  dx * dx + dy * dy

def countGridUnit (n : Nat) (pts : List (Int × Int)) : Nat :=
  ((pts.zipIdx).flatMap (fun (p, i) => (pts.zipIdx).filterMap (fun (q, j) =>
    if i < j && gridUnitSq n p q == (n : Int) then some (1 : Nat) else none))).length

def countGridUnit25 := countGridUnit 25

def isqrt (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc k => if k * k ≤ n then k else acc) 0

def harb (n : Nat) : Nat :=
  let s := isqrt (12 * n - 3)
  3 * n - (if s * s < 12 * n - 3 then s + 1 else s)

def witness144 : List (Int × Int) := [
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
  (0, 10),
  (0, 11),
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
  (1, 10),
  (1, 11),
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
  (2, 10),
  (2, 11),
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
  (3, 10),
  (3, 11),
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
  (4, 10),
  (4, 11),
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
  (5, 10),
  (5, 11),
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
  (6, 10),
  (6, 11),
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
  (7, 10),
  (7, 11),
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
  (8, 10),
  (8, 11),
  (9, 0),
  (9, 1),
  (9, 2),
  (9, 3),
  (9, 4),
  (9, 5),
  (9, 6),
  (9, 7),
  (9, 8),
  (9, 9),
  (9, 10),
  (9, 11),
  (10, 0),
  (10, 1),
  (10, 2),
  (10, 3),
  (10, 4),
  (10, 5),
  (10, 6),
  (10, 7),
  (10, 8),
  (10, 9),
  (10, 10),
  (10, 11),
  (11, 0),
  (11, 1),
  (11, 2),
  (11, 3),
  (11, 4),
  (11, 5),
  (11, 6),
  (11, 7),
  (11, 8),
  (11, 9),
  (11, 10),
  (11, 11)
]

theorem grid_n144_count : countGridUnit25 witness144 = 456 := by
  native_decide

theorem grid_n144_beats_harb : harb 144 < countGridUnit25 witness144 := by
  native_decide

theorem grid_n144_meets_harb : harb 144 ≤ countGridUnit25 witness144 := by
  native_decide

/-- Explicit lower bound: u(144) ≥ 456 (full 12×12 ℤ² grid, unit²=25). -/
theorem u144_grid_lower_bound : 456 ≤ countGridUnit25 witness144 := by
  native_decide

end Sounio.Erdos90Grid144
