/-
  SounioFureyOctonion — Sounio reproduces Furey's octonion -> one Standard-Model generation
  (Frente B, vector 4/3 Part A). Left-multiplication operators L_a (8x8 integer matrices) over Sounio's
  octonion convention (Cayley-Dickson sign cdSigma at bits=3). Furey's ladder operators
  alpha_i = 1/2(-L_{a_i} + i L_{b_i}) for pairs (a_i,b_i) = (1,2),(3,4),(5,6); to stay exact over Z[i]
  we use A_i = 2*alpha_i = (Re,Im) = (-L_{a_i}, L_{b_i}).

  CLAIM 1 (native_decide): the three modes close a fermionic ladder algebra over Z[i]:
    {A_i, A_j} = 0  and  {A_i, A_j^dag} = 4*delta_ij*I   for all i,j in {1,2,3}.
  CLAIM 2 (decide): the fermionic Fock-space occupation multiplicities C(3,n) = [1,3,3,1] -> one
  generation of electric charges Q=N/3 in {0,1/3,2/3,1}, the x3 being SU(3) colour.

  Mathlib-free, no sorry.
-/
namespace SounioFureyOctonion

-- Cayley-Dickson sign, recursion on `bits`; octonions at bits=3.
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
def octSigma (a b : Nat) : Int := cdSigma a b 3

-- L_a entry: e_a * e_c = octSigma(a,c) * e_{a^c}; nonzero only when r = a^c.
def lentry (a r c : Nat) : Int := if r == (a ^^^ c) then octSigma a c else 0

-- 8x8 integer matrices as List (List Int).
def mk (f : Nat → Nat → Int) : List (List Int) :=
  (List.range 8).map (fun r => (List.range 8).map (fun c => f r c))

-- A_i = 2*alpha_i, pairs (a_i,b_i) = (2i-1, 2i); Re = -L_{a_i}, Im = L_{b_i}.
def Are (i : Nat) : List (List Int) := mk (fun r c => - lentry (2*i - 1) r c)
def Aim (i : Nat) : List (List Int) := mk (fun r c => lentry (2*i) r c)

def matmul (A B : List (List Int)) : List (List Int) :=
  let n := A.length
  A.map (fun row => (List.range n).map (fun j =>
    (List.range n).foldl (fun s t => s + (row.getD t 0) * ((B.getD t []).getD j 0)) 0))
def madd (A B : List (List Int)) : List (List Int) :=
  (List.range A.length).map (fun i =>
    (List.range ((A.getD i []).length)).map (fun j => (A.getD i []).getD j 0 + (B.getD i []).getD j 0))
def smul (s : Int) (A : List (List Int)) : List (List Int) :=
  A.map (fun row => row.map (fun x => s * x))
def tpose (A : List (List Int)) : List (List Int) :=
  (List.range 8).map (fun i => (List.range 8).map (fun j => (A.getD j []).getD i 0))

-- complex 8x8 product (Xr+iXi)(Yr+iYi) = (XrYr - XiYi) + i(XrYi + XiYr).
def prodRe (Xr Xi Yr Yi : List (List Int)) : List (List Int) := madd (matmul Xr Yr) (smul (-1) (matmul Xi Yi))
def prodIm (Xr Xi Yr Yi : List (List Int)) : List (List Int) := madd (matmul Xr Yi) (matmul Xi Yr)
def anticommRe (Xr Xi Yr Yi : List (List Int)) : List (List Int) :=
  madd (prodRe Xr Xi Yr Yi) (prodRe Yr Yi Xr Xi)
def anticommIm (Xr Xi Yr Yi : List (List Int)) : List (List Int) :=
  madd (prodIm Xr Xi Yr Yi) (prodIm Yr Yi Xr Xi)

def isZero (M : List (List Int)) : Bool := M.all (fun row => row.all (fun x => x == 0))
def isScalar (M : List (List Int)) (v : Int) : Bool :=
  (List.range 8).all (fun r => (List.range 8).all (fun c =>
    (M.getD r []).getD c 0 == (if r == c then v else 0)))

-- {A_i,A_j}=0 and {A_i,A_j^dag}=4 delta_ij I.
def checkPair (i j : Nat) : Bool :=
  let Xr := Are i; let Xi := Aim i
  let Yr := Are j; let Yi := Aim j
  let acR := anticommRe Xr Xi Yr Yi; let acI := anticommIm Xr Xi Yr Yi
  let YrD := tpose (Are j); let YiD := smul (-1) (tpose (Aim j))
  let adR := anticommRe Xr Xi YrD YiD; let adI := anticommIm Xr Xi YrD YiD
  let want : Int := if i == j then 4 else 0
  isZero acR && isZero acI && isScalar adR want && isZero adI
def checkAll : Bool :=
  ((List.range 3).map (· + 1)).all (fun i => ((List.range 3).map (· + 1)).all (fun j => checkPair i j))

-- CLAIM 1: the fermionic ladder algebra closes exactly over Z[i].
theorem ladder_closes : checkAll = true := by native_decide

-- CLAIM 2: Fock-space occupation multiplicities C(3,n) = one generation {1,3,3,1}.
def fact : Nat → Nat
  | 0 => 1
  | (n+1) => (n+1) * fact n
def binom (n k : Nat) : Nat := fact n / (fact k * fact (n - k))
theorem charge_multiplicities : ((List.range 4).map (fun n => binom 3 n)) = [1, 3, 3, 1] := by decide

end SounioFureyOctonion
