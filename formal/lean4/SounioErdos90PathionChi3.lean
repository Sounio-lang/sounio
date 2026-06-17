import SounioPathionBridge
import SounioAssociatorShadow

/-! 
# Sounio — Erdős [90]: pathion (32D) associator-conflict χ ≥ 3 witness

Lifts the non-distance associator-conflict construction of
`SounioErdosUnitDistance.lean` (§10) from sedenions (16D) to pathions (32D).
The associator `[x,y,c] = (x·y)·c − x·(y·c)` is bilinear in (x,y), so the
conflict graph edge relation `‖[x,y,c]‖² = 4` is not translation-invariant
and need not inherit the parity bipartition that traps every distance-based
ZD surgery. This file certifies the first pathion-level χ ≥ 3 witness.

## What this proves (`native_decide`, no `sorry`, no Mathlib)

* `pathion_conflict_triangle` — on the unit basis vectors `{e₁, e₂, e₃}` in
  32D pathion coordinates, with the first valid pathion primitive
  `c = e₁ + e₁₈`, all three pairwise associator conflicts have squared norm 4.
  Hence the induced conflict graph contains a triangle.
* `pathion_chi_ge_3` — the 3-vertex conflict graph is not 2-colorable,
  therefore its chromatic number is at least 3.

This is the pathion analogue of the sedenion `associator_conflict_triangle`
theorem; the witness uses the same three basis indices {1,2,3}, and the
primitive `c = e₁ + e₁₈` is the first valid pathion primitive (the lower-half
index 1 pairs with the first non-diagonal upper-half index 18).

## Honest scope

Unconditional finite-model statement in 32D pathion coordinates. It is not a
planar unit-distance claim; no ℝ^d embedding is asserted.
-/ 

namespace Sounio.Erdos90PathionChi3

open Sounio.PathionBridge
open Sounio.AssociatorShadow (SVec combine)

-- ===========================================================================
-- §1. 32-D pathion product and associator conflict edge.
-- ===========================================================================

/-- Sparse pathion product via the verified `pathSigma = cdSigma · 5`. -/
def pSmulC (a b : SVec) : SVec :=
  combine (a.flatMap (fun p => b.map (fun q => (p.1 ^^^ q.1, p.2 * q.2 * pathSigma p.1 q.1))))

/-- A primitive pathion element `e_lo ± e_hi` as a sparse vector. -/
def pVecC (u : PrimPath) : SVec :=
  [(u.lo, (1 : Int)), (u.hi, if u.neg then (-1 : Int) else 1)]

/-- Squared norm of a sparse pathion vector. -/
def pNsqC (v : SVec) : Int :=
  (combine v).foldl (fun a q => a + q.2 * q.2) 0

/-- Associator `[x,y,c] = (x·y)·c − x·(y·c)` as a sparse pathion vector. -/
def pAssoc (x y c : SVec) : SVec :=
  combine (pSmulC (pSmulC x y) c ++ (pSmulC x (pSmulC y c)).map (fun p => (p.1, -p.2)))

/-- Squared norm of the associator `‖[x,y,c]‖²`. -/
def pAssocNormSq (x y : SVec) (c : PrimPath) : Int :=
  pNsqC (pAssoc x y (pVecC c))

/-- Unit basis vector `e_i` in 32D pathion coordinates. -/
def e32 (i : Nat) : SVec := [(i, (1 : Int))]

/-- Symmetrized associator-conflict edge predicate on basis vectors:
    edge (i,j) iff `‖[e_i,e_j,c]‖² = 4` or `‖[e_j,e_i,c]‖² = 4`.
    (The two norms are equal because the associator is antisymmetric in the
    first two arguments, but we keep the symmetrized form to match the
    sedenion construction literally.) -/
def pAssocEdge (c : PrimPath) (i j : Nat) : Bool :=
  pAssocNormSq (e32 i) (e32 j) c == 4 || pAssocNormSq (e32 j) (e32 i) c == 4

-- ===========================================================================
-- §2. The witness primitive and triangle certificate.
-- ===========================================================================

/-- The first valid pathion primitive: `e₁ + e₁₈`.
    `pathPrims` enumerates `lo ∈ {1..15}`, `hi ∈ {17..31}` excluding the
    diagonal `lo ⊕ hi = 16`. The first candidate `(1,17,+)` is diagonal and
    excluded; `(1,18,+)` is therefore the head of `pathPrims`. -/
def cPathWit : PrimPath := ⟨1, 18, false⟩

/-- `cPathWit` is indeed the first valid pathion primitive. -/
theorem cPathWit_is_first : pathPrims.head? = some cPathWit := by native_decide

/-- **Triangle witness.** On the unit basis vectors `{e₁, e₂, e₃}` with
    `c = e₁ + e₁₈`, all three pairwise associator conflicts have squared
    norm exactly 4. A triangle is not 2-colorable, so the associator conflict
    graph has χ ≥ 3. -/
theorem pathion_conflict_triangle :
    pAssocEdge cPathWit 1 2 = true
    ∧ pAssocEdge cPathWit 1 3 = true
    ∧ pAssocEdge cPathWit 2 3 = true := by native_decide

-- ===========================================================================
-- §3. Finite chromatic-number lower bound (3 vertices).
-- ===========================================================================

/-- Unordered pairs of vertices 0..2. -/
def vpairs3 : List (Nat × Nat) :=
  (List.range 3).flatMap (fun i =>
    (List.range 3).filterMap (fun j => if i < j then some (i, j) else none))

/-- Color of vertex `v` in the `a`-th coloring written in base `k`. -/
def colorOf (a k v : Nat) : Nat := (a / (k ^ v)) % k

/-- Does coloring index `a` (base `k`) properly color the graph? -/
def properAssign (a k : Nat) (edge : Nat → Nat → Bool) : Bool :=
  vpairs3.all (fun p =>
    (! edge p.1 p.2) || (colorOf a k p.1 != colorOf a k p.2))

/-- Is the 3-vertex graph properly `k`-colorable? (Brute force over `k³` colorings.) -/
def kColorable (k : Nat) (edge : Nat → Nat → Bool) : Bool :=
  (List.range (k ^ 3)).any (fun a => properAssign a k edge)

/-- Exact chromatic number of a 3-vertex graph. -/
def chromaticNumber3 (edge : Nat → Nat → Bool) : Nat :=
  ((List.range (3 + 1)).find? (fun k => kColorable k edge)).getD 3

/-- The 3-vertex conflict graph induced by `{e₁, e₂, e₃}` on vertices 1,2,3.
    We remap them to vertices 0,1,2 for the chromatic-number enumeration. -/
def pathionConflictGraph (i j : Nat) : Bool :=
  let edge : Nat → Nat → Bool := fun x y =>
    if x = 0 then      if y = 1 then pAssocEdge cPathWit 1 2
                    else if y = 2 then pAssocEdge cPathWit 1 3
                    else false
    else if x = 1 then if y = 0 then pAssocEdge cPathWit 1 2
                    else if y = 2 then pAssocEdge cPathWit 2 3
                    else false
    else if x = 2 then if y = 0 then pAssocEdge cPathWit 1 3
                    else if y = 1 then pAssocEdge cPathWit 2 3
                    else false
    else false
  edge i j

/-- **Main theorem.** The pathion associator-conflict graph on `{e₁,e₂,e₃}`
    has chromatic number at least 3. -/
theorem pathion_chi_ge_3 : chromaticNumber3 pathionConflictGraph ≥ 3 := by native_decide

end Sounio.Erdos90PathionChi3
