/-!
# Erdős [90] — ℤ² subset witness at n=100, u(100) ≥ 303 (saturation run)

Generated from `stdlib/research/erdos90_subset_saturation_export.sio`.
Seed=1000003, winR=18, unit²=5. Not a claim of global optimality.
-/

namespace Sounio.Erdos90Subset303

def gridUnitSq (n : Nat) (p q : Int × Int) : Int :=
  let dx := p.1 - q.1
  let dy := p.2 - q.2
  dx * dx + dy * dy

def countGridUnit5 (pts : List (Int × Int)) : Nat :=
  ((pts.zipIdx).flatMap (fun (p, i) => (pts.zipIdx).filterMap (fun (q, j) =>
    if i < j && gridUnitSq 5 p q == 5 then some (1 : Nat) else none))).length

def isqrt (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc k => if k * k ≤ n then k else acc) 0

def harb (n : Nat) : Nat :=
  let s := isqrt (12 * n - 3)
  3 * n - (if s * s < 12 * n - 3 then s + 1 else s)

def witness100 : List (Int × Int) := [
  (-1, -12),
  (0, -14),
  (0, -10),
  (1, -16),
  (1, -13),
  (1, -12),
  (1, -11),
  (1, -8),
  (2, -15),
  (2, -14),
  (2, -13),
  (2, -12),
  (2, -11),
  (2, -10),
  (2, -9),
  (3, -17),
  (3, -16),
  (3, -15),
  (3, -14),
  (3, -13),
  (3, -12),
  (3, -11),
  (3, -10),
  (3, -9),
  (3, -7),
  (4, -16),
  (4, -15),
  (4, -14),
  (4, -13),
  (4, -12),
  (4, -11),
  (4, -10),
  (4, -9),
  (4, -8),
  (5, -18),
  (5, -17),
  (5, -16),
  (5, -15),
  (5, -14),
  (5, -13),
  (5, -12),
  (5, -11),
  (5, -10),
  (5, -9),
  (5, -8),
  (5, -6),
  (6, -17),
  (6, -16),
  (6, -15),
  (6, -14),
  (6, -13),
  (6, -12),
  (6, -11),
  (6, -10),
  (6, -9),
  (6, -8),
  (6, -7),
  (7, -18),
  (7, -17),
  (7, -16),
  (7, -15),
  (7, -14),
  (7, -13),
  (7, -12),
  (7, -11),
  (7, -10),
  (7, -9),
  (7, -7),
  (8, -16),
  (8, -15),
  (8, -14),
  (8, -13),
  (8, -12),
  (8, -11),
  (8, -10),
  (8, -9),
  (8, -8),
  (9, -17),
  (9, -16),
  (9, -15),
  (9, -14),
  (9, -13),
  (9, -12),
  (9, -11),
  (9, -10),
  (9, -8),
  (10, -15),
  (10, -14),
  (10, -13),
  (10, -12),
  (10, -11),
  (10, -10),
  (10, -9),
  (11, -16),
  (11, -13),
  (11, -12),
  (11, -11),
  (12, -14),
  (12, -10),
  (13, -12)
]

theorem subset_n100_count : countGridUnit5 witness100 = 303 := by
  native_decide

theorem subset_n100_beats_prior302 : 302 < countGridUnit5 witness100 := by
  native_decide

theorem subset_n100_beats_grid10x10 : 288 < countGridUnit5 witness100 := by
  native_decide

theorem subset_n100_meets_harb : harb 100 ≤ countGridUnit5 witness100 := by
  native_decide

theorem u100_subset_lower_bound : 303 ≤ countGridUnit5 witness100 := by
  native_decide

end Sounio.Erdos90Subset303
