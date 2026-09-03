/-
  SounioSedenionDynamics — the substrate DYNAMICS (Frente B, vector 4/2): the sedenion ZD-geometry
  graphs carry an exact spanning-tree complexity (Matrix-Tree theorem) and random-walk return structure.
  tau = det of a principal (n-1)x(n-1) Laplacian cofactor, computed by fraction-free (Bareiss) integer
  Gaussian elimination. fiber K_{6,6}-3K_{2,2}: tau=393216; 2*K_7: tau=1075648. Mathlib-free, no sorry.
-/
namespace SounioSedenionDynamics

def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        let aHi := a ≥ half; let bHi := b ≥ half; let aLo := a % half; let bLo := b % half
        if !aHi && !bHi then cdSigma aLo bLo (n+1)
        else if !aHi && bHi then cdSigma bLo aLo (n+1)
        else if aHi && !bHi then (if bLo == 0 then cdSigma aLo 0 (n+1) else - cdSigma aLo bLo (n+1))
        else (if bLo == 0 then - cdSigma 0 aLo (n+1) else cdSigma bLo aLo (n+1))
def sedSigma (a b : Nat) : Int := cdSigma a b 4
def primProd (ulo uhi : Nat) (un : Bool) (vlo vhi : Nat) (vn : Bool) (k : Nat) : Int :=
  let su : Int := if un then -1 else 1
  let sv : Int := if vn then -1 else 1
  (if (ulo ^^^ vlo) == k then sedSigma ulo vlo else 0)
  + (if (ulo ^^^ vhi) == k then sv * sedSigma ulo vhi else 0)
  + (if (uhi ^^^ vlo) == k then su * sedSigma uhi vlo else 0)
  + (if (uhi ^^^ vhi) == k then su * sv * sedSigma uhi vhi else 0)
def isZeroPair (ulo uhi : Nat) (un : Bool) (vlo vhi : Nat) (vn : Bool) : Bool :=
  (List.range 16).all (fun k => primProd ulo uhi un vlo vhi vn k == 0)
abbrev V := Nat × Nat × Bool
-- fiber L=9 vertices: mixed-half primitives with lo^hi=9 and hi != 8
def fiber : List V :=
  ((List.range 7).flatMap (fun i => (List.range 8).flatMap (fun j =>
     let lo := i+1; let hi := j+8
     if (lo ^^^ hi) == 9 && hi != 8 then [(lo,hi,false),(lo,hi,true)] else []))).eraseDups
-- integer adjacency matrix (row-major over `fiber`)
def adjMat : List (List Int) :=
  fiber.map (fun v => fiber.map (fun w =>
    if v == w then 0
    else if isZeroPair v.1 v.2.1 v.2.2 w.1 w.2.1 w.2.2 then 1 else 0))
-- 2*K_7: 0 on diagonal, 2 off-diagonal
def k7 : List (List Int) := (List.range 7).map (fun i => (List.range 7).map (fun j => if i == j then 0 else 2))

-- ── random walk: (A^k)_{00} via integer matrix powers ─────────────────────────
def matmul (A B : List (List Int)) : List (List Int) :=
  let n := A.length
  A.map (fun row => (List.range n).map (fun j =>
    (List.range n).foldl (fun s t => s + (row.getD t 0) * ((B.getD t []).getD j 0)) 0))
def power (A : List (List Int)) : Nat → List (List Int)
  | 0 => (List.range A.length).map (fun i => (List.range A.length).map (fun j => if i == j then (1:Int) else 0))
  | (k+1) => matmul (power A k) A
def walk (A : List (List Int)) (k : Nat) : Int := ((power A k).getD 0 []).getD 0 0

-- ── Matrix-Tree: tau = det of a principal (n-1)x(n-1) Laplacian cofactor ───────
def degOf (A : List (List Int)) (i : Nat) : Int := (A.getD i []).foldl (·+·) 0
-- Laplacian L = D - A, then delete row/col 0 (rows/cols 1..n-1).
def lapCofactor (A : List (List Int)) : List (List Int) :=
  let n := A.length
  (List.range (n-1)).map (fun r => (List.range (n-1)).map (fun c =>
    let gr := r+1; let gc := c+1
    (if gr == gc then degOf A gr else 0) - (A.getD gr []).getD gc 0))
-- element access / update on List (List Int)
def getE (M : List (List Int)) (i j : Nat) : Int := (M.getD i []).getD j 0
def setE (M : List (List Int)) (i j : Nat) (v : Int) : List (List Int) :=
  M.set i ((M.getD i []).set j v)
-- Fraction-free (Bareiss) integer Gaussian elimination. The reduced Laplacian of a connected graph is
-- positive-definite, so no pivot is ever zero; we run pivot-free.
def bareissStep (M : List (List Int)) (k n : Nat) (prev : Int) : List (List Int) :=
  (List.range n).foldl (fun M1 i =>
    if i ≤ k then M1 else
    (List.range n).foldl (fun M2 j =>
      if j ≤ k then M2 else
      setE M2 i j ((getE M2 i j * getE M2 k k - getE M2 i k * getE M2 k j) / prev)) M1) M
-- structural recursion on `fuel` (fuel = n suffices: at most n-1 elimination steps).
def bareissGo (M : List (List Int)) (k n : Nat) (prev : Int) : Nat → Int
  | 0 => getE M (n-1) (n-1)
  | (fuel+1) =>
      if k + 1 ≥ n then getE M (n-1) (n-1)
      else
        let M' := bareissStep M k n prev
        bareissGo M' (k+1) n (getE M' k k) fuel
def bareissDet (M : List (List Int)) : Int :=
  let n := M.length
  bareissGo M 0 n 1 n
def spanningTrees (A : List (List Int)) : Int := bareissDet (lapCofactor A)

theorem fiber_verts : fiber.length = 12 := by native_decide
-- spanning-tree counts (headline)
theorem fiber_spantree : spanningTrees adjMat = 393216 := by native_decide
theorem k7_spantree : spanningTrees k7 = 1075648 := by native_decide
-- random-walk return counts (secondary)
theorem fiber_walk2 : walk adjMat 2 = 4 := by native_decide
theorem fiber_walk4 : walk adjMat 4 = 48 := by native_decide
theorem k7_walk2 : walk k7 2 = 24 := by native_decide

end SounioSedenionDynamics
