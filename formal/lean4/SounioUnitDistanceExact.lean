/-!
# Sounio — Exact Unit-Distance Certifier (Hadwiger–Nelson #508 foundation)

First stone of the #508 thrust (6-chromatic unit-distance graph). This file builds
the **certifier half** — exact, float-free unit-distance + chromatic verification —
and validates it on known ground: the **Moser spindle** (7 vertices, 11 unit edges,
χ = 4), the smallest unit-distance graph whose realization *requires* genuine
irrational arithmetic.

## Why exact arithmetic is the point

Every claimed unit-distance construction must answer one question without error:
*is this distance exactly 1?* Floating point cannot; rounding has invalidated claimed
graphs in the literature. We work in the real number field **ℚ(√3, √11)**, represented
exactly as `(a₀ + a₁√3 + a₂√11 + a₃√33)` with **integer** components (coordinates scaled
by 12 to clear denominators; a unit edge then has squared length exactly `144`). All
arithmetic is exact integer arithmetic, so `native_decide` certifies every distance.

## What this proves (`native_decide`, no `sorry`, no Mathlib)

* `unit_pairs_are_the_11_edges` — the exact unit-distance graph on the 7 spindle
  vertices is *exactly* the 11 claimed edges (no spurious unit pair, none missing).
* `spindle_edge_count` — exactly 11 unit edges.
* `spindle_chromatic_eq_4` — its chromatic number is exactly 4 (brute force over all
  `4⁷` colorings).
* `moser_spindle_certified` — the three together.

## Honest scope

This is the **certifier**, proven correct on the χ = 4 landmark. It is *not* progress on
#508 by itself — χ(plane) ≥ 5 already needs ~500-vertex graphs (de Grey 2018, Heule's
SAT reductions), and χ ≥ 6 is wide open and may be false. The path from here:
1. (done) exact unit-distance + χ verification over a number field — this file.
2. scale the colorability check to a verified **UNSAT certificate** (non-k-colorability),
   the back-end where machine-code control is the real edge over float/script pipelines.
3. GPU + ML-guided search (FunSearch-style) over algebraically-structured vertex families
   for a graph forcing χ ≥ 6 — the discovery half, the genuine long shot.

Reference: L. Moser & W. Moser (1961); A. de Grey (2018), "The chromatic number of the
plane is at least 5"; M. Heule, SAT reductions.
-/

namespace Sounio.UnitDistanceExact

-- ===========================================================================
-- §1. Exact arithmetic in K = ℚ(√3, √11), integer components.
--     x = (a₀, a₁, a₂, a₃)  ↦  a₀ + a₁√3 + a₂√11 + a₃√33,   √33 = √3·√11.
-- ===========================================================================

abbrev K := Int × Int × Int × Int

def kadd (x y : K) : K := (x.1+y.1, x.2.1+y.2.1, x.2.2.1+y.2.2.1, x.2.2.2+y.2.2.2)
def ksub (x y : K) : K := (x.1-y.1, x.2.1-y.2.1, x.2.2.1-y.2.2.1, x.2.2.2-y.2.2.2)

/-- Product in K using √3√3=3, √11√11=11, √33√33=33, √3√11=√33, √3√33=3√11,
    √11√33=11√3. -/
def kmul (x y : K) : K :=
  ( x.1*y.1 + 3*x.2.1*y.2.1 + 11*x.2.2.1*y.2.2.1 + 33*x.2.2.2*y.2.2.2,
    x.1*y.2.1 + x.2.1*y.1 + 11*x.2.2.1*y.2.2.2 + 11*x.2.2.2*y.2.2.1,
    x.1*y.2.2.1 + x.2.2.1*y.1 + 3*x.2.1*y.2.2.2 + 3*x.2.2.2*y.2.1,
    x.1*y.2.2.2 + x.2.2.2*y.1 + x.2.1*y.2.2.1 + x.2.2.1*y.2.1 )

/-- A point in the plane: a pair of K-coordinates. -/
abbrev Pt := K × K

/-- Exact squared Euclidean distance, in K. -/
def d2 (p q : Pt) : K :=
  let dx := ksub p.1 q.1
  let dy := ksub p.2 q.2
  kadd (kmul dx dx) (kmul dy dy)

/-- Squared length of a unit edge, in scaled coordinates (×12 ⇒ 12² = 144). -/
def unit144 : K := (144, 0, 0, 0)

-- ===========================================================================
-- §2. The Moser spindle, exact coordinates (×12, integer components).
--     Two unit rhombi sharing vertex A, the second rotated by θ, cosθ = 5/6,
--     sinθ = √11/6; the far vertices D, G then lie at unit distance.
-- ===========================================================================

def coords : Nat → Pt
  | 0 => ((0,0,0,0),  (0,0,0,0))    -- A = (0,0)
  | 1 => ((0,6,0,0),  (6,0,0,0))    -- B = (√3/2, 1/2)·12
  | 2 => ((0,6,0,0),  (-6,0,0,0))   -- C = (√3/2,-1/2)·12
  | 3 => ((0,12,0,0), (0,0,0,0))    -- D = (√3, 0)·12
  | 4 => ((0,5,-1,0), (5,0,0,1))    -- E = ((5√3-√11)/12, (√33+5)/12)·12
  | 5 => ((0,5,1,0),  (-5,0,0,1))   -- F = ((5√3+√11)/12, (√33-5)/12)·12
  | 6 => ((0,10,0,0), (0,0,0,2))    -- G = (5√3/6, √33/6)·12
  | _ => ((0,0,0,0),  (0,0,0,0))

def verts : List Nat := [0,1,2,3,4,5,6]

/-- Unordered vertex pairs i < j. -/
def pairs : List (Nat × Nat) :=
  verts.flatMap (fun i => verts.filterMap (fun j => if i < j then some (i, j) else none))

/-- Exact unit-distance test. -/
def isUnit (i j : Nat) : Bool := d2 (coords i) (coords j) == unit144

/-- The 11 claimed spindle edges (lex order by (i,j)). -/
def claimedEdges : List (Nat × Nat) :=
  [(0,1),(0,2),(0,4),(0,5),(1,2),(1,3),(2,3),(3,6),(4,5),(4,6),(5,6)]

-- ===========================================================================
-- §3. The exact unit-distance graph IS the spindle.
-- ===========================================================================

/-- **Exactness.** The exact unit-distance pairs among the 7 vertices are exactly the
    11 claimed edges — no float, no spurious or missing edge. -/
theorem unit_pairs_are_the_11_edges :
    pairs.filter (fun p => isUnit p.1 p.2) = claimedEdges := by native_decide

/-- Exactly 11 unit edges. -/
theorem spindle_edge_count :
    (pairs.filter (fun p => isUnit p.1 p.2)).length = 11 := by native_decide

-- ===========================================================================
-- §4. Exact chromatic number (brute force over all colorings).
-- ===========================================================================

def nverts : Nat := 7
def colorOf (a k v : Nat) : Nat := (a / (k ^ v)) % k
def properAssign (a k : Nat) : Bool :=
  pairs.all (fun p => (! isUnit p.1 p.2) || colorOf a k p.1 != colorOf a k p.2)
def kColorable (k : Nat) : Bool :=
  (List.range (k ^ nverts)).any (fun a => properAssign a k)
def chromaticNumber : Nat :=
  ((List.range (nverts + 1)).find? (fun k => kColorable k)).getD nverts

/-- **χ = 4.** The Moser spindle needs 4 colors (3 do not suffice; 4 do). -/
theorem spindle_chromatic_eq_4 : chromaticNumber = 4 := by native_decide

-- ===========================================================================
-- §5. Summary.
-- ===========================================================================

/-- **Moser spindle, exactly certified.** Its exact unit-distance graph over ℚ(√3,√11)
    is precisely the 11-edge spindle, and its chromatic number is exactly 4. The exact
    arithmetic + chromatic verifier — the certifier half of the #508 program — works on
    the χ = 4 landmark. -/
theorem moser_spindle_certified :
    (pairs.filter (fun p => isUnit p.1 p.2) = claimedEdges) ∧
    ((pairs.filter (fun p => isUnit p.1 p.2)).length = 11) ∧
    (chromaticNumber = 4) := by
  exact ⟨unit_pairs_are_the_11_edges, spindle_edge_count, spindle_chromatic_eq_4⟩

end Sounio.UnitDistanceExact
