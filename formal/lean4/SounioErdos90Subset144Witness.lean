/-!
# Erdős [90] — ℤ² subset witness at n=144, u(144) ≥ 493 (machine-checked)

Generated from `stdlib/research/erdos90_subset144_export.sio`.
Seed=9000023, winR=22, unit²=25. Not a claim of global optimality.
-/

namespace Sounio.Erdos90Subset144

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

def witness144 : List (Int × Int) := [
  (-18, 5),
  (-18, 10),
  (-15, 1),
  (-15, 4),
  (-15, 6),
  (-15, 9),
  (-15, 14),
  (-14, 2),
  (-14, 3),
  (-14, 7),
  (-14, 8),
  (-14, 13),
  (-13, 5),
  (-13, 10),
  (-12, 0),
  (-12, 5),
  (-12, 8),
  (-12, 9),
  (-12, 10),
  (-11, -2),
  (-11, 1),
  (-11, 3),
  (-11, 4),
  (-11, 6),
  (-11, 7),
  (-11, 9),
  (-11, 11),
  (-11, 12),
  (-11, 17),
  (-10, 0),
  (-10, 1),
  (-10, 4),
  (-10, 5),
  (-10, 6),
  (-10, 9),
  (-10, 10),
  (-10, 14),
  (-9, 2),
  (-9, 3),
  (-9, 4),
  (-9, 5),
  (-9, 7),
  (-9, 8),
  (-9, 9),
  (-9, 10),
  (-9, 12),
  (-9, 13),
  (-8, -3),
  (-8, 2),
  (-8, 3),
  (-8, 5),
  (-8, 6),
  (-8, 7),
  (-8, 8),
  (-8, 10),
  (-8, 11),
  (-8, 13),
  (-8, 15),
  (-7, 0),
  (-7, 1),
  (-7, 4),
  (-7, 5),
  (-7, 6),
  (-7, 8),
  (-7, 9),
  (-7, 10),
  (-7, 14),
  (-6, -2),
  (-6, 1),
  (-6, 3),
  (-6, 4),
  (-6, 6),
  (-6, 7),
  (-6, 8),
  (-6, 9),
  (-6, 11),
  (-6, 12),
  (-6, 13),
  (-6, 17),
  (-5, 0),
  (-5, 1),
  (-5, 4),
  (-5, 5),
  (-5, 6),
  (-5, 7),
  (-5, 9),
  (-5, 10),
  (-5, 11),
  (-5, 14),
  (-4, 0),
  (-4, 2),
  (-4, 4),
  (-4, 5),
  (-4, 7),
  (-4, 8),
  (-4, 9),
  (-4, 10),
  (-4, 12),
  (-4, 13),
  (-3, -3),
  (-3, 2),
  (-3, 3),
  (-3, 5),
  (-3, 7),
  (-3, 8),
  (-3, 11),
  (-3, 13),
  (-2, 1),
  (-2, 4),
  (-2, 5),
  (-2, 6),
  (-2, 9),
  (-2, 10),
  (-2, 14),
  (-1, -2),
  (-1, 3),
  (-1, 4),
  (-1, 6),
  (-1, 8),
  (-1, 9),
  (-1, 11),
  (-1, 12),
  (-1, 17),
  (0, 1),
  (0, 5),
  (0, 6),
  (0, 7),
  (0, 10),
  (1, 0),
  (1, 5),
  (1, 8),
  (1, 10),
  (2, 2),
  (2, 7),
  (2, 8),
  (2, 13),
  (3, 1),
  (3, 6),
  (3, 9),
  (3, 14),
  (4, 4),
  (4, 9),
  (6, 5),
  (6, 10)
]

theorem subset_n144_count : countGridUnit25 witness144 = 493 := by
  native_decide

theorem subset_n144_beats_grid12x12 : 456 < countGridUnit25 witness144 := by
  native_decide

theorem subset_n144_meets_harb : harb 144 ≤ countGridUnit25 witness144 := by
  native_decide

theorem u144_subset_lower_bound : 493 ≤ countGridUnit25 witness144 := by
  native_decide

end Sounio.Erdos90Subset144
