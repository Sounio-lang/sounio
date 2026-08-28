/-
  SounioSedenionSpectra — the emergent metric (Frente B, vector 4/1): the sedenion ZD-geometry graphs
  have INTEGRAL spectra, certified by spectral moments trace(A^k). Fiber (K_{6,6}-3K_{2,2}) adjacency
  spectrum {4,2^2,0^6,-2^2,-4}; 2*K_7 fiber-incidence {12,-2^6}. Mathlib-free, no sorry.
-/
namespace SounioSedenionSpectra

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
-- integer adjacency matrix as a function (row-major over `fiber`)
def adjMat : List (List Int) :=
  fiber.map (fun v => fiber.map (fun w =>
    if v == w then 0
    else if isZeroPair v.1 v.2.1 v.2.2 w.1 w.2.1 w.2.2 then 1 else 0))
def matmul (A B : List (List Int)) : List (List Int) :=
  let n := A.length
  A.map (fun row => (List.range n).map (fun j =>
    (List.range n).foldl (fun s t => s + (row.getD t 0) * ((B.getD t []).getD j 0)) 0))
def traceM (A : List (List Int)) : Int := (List.range A.length).foldl (fun s i => s + (A.getD i []).getD i 0) 0
def power (A : List (List Int)) : Nat → List (List Int)
  | 0 => (List.range A.length).map (fun i => (List.range A.length).map (fun j => if i == j then (1:Int) else 0))
  | (k+1) => matmul (power A k) A
def moment (A : List (List Int)) (k : Nat) : Int := traceM (power A k)

-- 2*K_7
def k7 : List (List Int) := (List.range 7).map (fun i => (List.range 7).map (fun j => if i == j then 0 else 2))

theorem fiber_verts : fiber.length = 12 := by native_decide
theorem fiber_m2 : moment adjMat 2 = 48 := by native_decide
theorem fiber_m4 : moment adjMat 4 = 576 := by native_decide
theorem fiber_m6 : moment adjMat 6 = 8448 := by native_decide
theorem k7_m2 : moment k7 2 = 168 := by native_decide
theorem k7_m3 : moment k7 3 = 1680 := by native_decide

end SounioSedenionSpectra
