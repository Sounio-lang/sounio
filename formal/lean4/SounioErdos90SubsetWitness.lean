/-!
# Erdős [90] — ℤ² subset witness at n=100 (machine-checked)

Generated from `stdlib/research/erdos90_subset_export.sio` via
`scripts/gates/gen_lean_subset_witness.py`. Densest-k-subgraph hill-climb on a
ℤ² pool (seed=1000003, winR=20); unit edge iff `(dx)²+(dy)² = 5.
Honest finite witness only — not a claim of global optimality.
-/

namespace Sounio.Erdos90Subset

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
  (1, 5),
  (2, 3),
  (2, 4),
  (2, 7),
  (3, 1),
  (3, 2),
  (3, 3),
  (3, 4),
  (3, 5),
  (3, 6),
  (3, 7),
  (3, 9),
  (4, 0),
  (4, 2),
  (4, 3),
  (4, 4),
  (4, 5),
  (4, 6),
  (4, 7),
  (4, 8),
  (5, 0),
  (5, 1),
  (5, 2),
  (5, 3),
  (5, 4),
  (5, 5),
  (5, 6),
  (5, 7),
  (5, 8),
  (5, 10),
  (6, -1),
  (6, 1),
  (6, 2),
  (6, 3),
  (6, 4),
  (6, 5),
  (6, 6),
  (6, 7),
  (6, 8),
  (6, 9),
  (7, -1),
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
  (7, 11),
  (8, -2),
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
  (9, -1),
  (9, 0),
  (9, 1),
  (9, 2),
  (9, 3),
  (9, 4),
  (9, 5),
  (9, 6),
  (9, 7),
  (9, 8),
  (9, 10),
  (10, -1),
  (10, 1),
  (10, 2),
  (10, 3),
  (10, 4),
  (10, 5),
  (10, 6),
  (10, 7),
  (10, 8),
  (10, 9),
  (11, 1),
  (11, 2),
  (11, 3),
  (11, 4),
  (11, 5),
  (11, 6),
  (11, 7),
  (11, 9),
  (12, 3),
  (12, 4),
  (12, 5),
  (12, 7),
  (12, 8),
  (13, 5),
  (13, 6)
]

theorem subset_n100_count : countGridUnit5 witness100 = 302 := by
  native_decide

theorem subset_n100_beats_harb : harb 100 < countGridUnit5 witness100 := by
  native_decide

theorem subset_n100_beats_grid10x10 : 288 < countGridUnit5 witness100 := by
  native_decide

theorem subset_n100_meets_harb : harb 100 ≤ countGridUnit5 witness100 := by
  native_decide

theorem subset_n100_meets_grid10x10 : 288 ≤ countGridUnit5 witness100 := by
  native_decide

/-- Explicit lower bound: u(100) ≥ 302 (ℤ² subset search, unit²=5). -/
theorem u100_subset_lower_bound : 302 ≤ countGridUnit5 witness100 := by
  native_decide

end Sounio.Erdos90Subset
