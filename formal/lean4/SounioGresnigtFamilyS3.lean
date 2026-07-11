/-
  SounioGresnigtFamilyS3 — Gresnigt's family S₃ generator ψ, reconstructed exactly and shown to be
  NON-MONOMIAL, by native_decide over ℚ(√3). ψ is the order-3 automorphism of the sedenions from
  Gresnigt arXiv:2306.13098 §5 eq (70)-(76): a 120° rotation in each plane {e_i,e_{i+8}} (i=1..7),
  a=(√3−1)/2, b=(−√3−1)/2. Certified: ψ is a genuine sedenion automorphism, order 3, maps the gen-1
  ladder A†_1 to the gen-2 ladder B†_1, and is NON-MONOMIAL (uses √3). Hence the fermion FAMILY symmetry
  is not a signed permutation and lies OUTSIDE the zero-divisor monomial-168 — the decisive answer:
  no bridge from the ZD-168 to fermion generations. Erratum E1 vindicated with the family generator
  explicit. Frame-relative mechanism: the monomial φ (color-triplet Weyl-S₃) commutes with the gen-1
  number operator N; ψ carries N to the gen-2 operator (equal spectrum, different operator). Mathlib-free.
-/
namespace SounioGresnigtFamilyS3

abbrev Q3 := Rat × Rat            -- p + q√3
def qadd (x y : Q3) : Q3 := (x.1 + y.1, x.2 + y.2)
def qsub (x y : Q3) : Q3 := (x.1 - y.1, x.2 - y.2)
def qmul (x y : Q3) : Q3 := (x.1*y.1 + 3*x.2*y.2, x.1*y.2 + x.2*y.1)
def Q0 : Q3 := (0,0)
def Q1 : Q3 := (1,0)

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
def sgn (a b : Nat) : Q3 := ((cdSigma a b 4 : Int), 0)

-- sedenion element = Nat → Q3 (coeff of e_k). product of two elements:
def smul (u v : Nat → Q3) : Nat → Q3 := fun k =>
  (List.range 16).foldl (fun acc a =>
    (List.range 16).foldl (fun acc2 b =>
      if (a ^^^ b) == k then qadd acc2 (qmul (qmul (u a) (v b)) (sgn a b)) else acc2) acc) Q0
def eunit (i : Nat) : Nat → Q3 := fun k => if k == i then Q1 else Q0

-- ψ : 120° rotation in {e_i, e_{i+8}}, i=1..7 ; fix e_0, e_8.
def psi (j : Nat) : Nat → Q3 := fun k =>
  if j == 0 || j == 8 then (if k == j then Q1 else Q0)
  else if 1 ≤ j && j ≤ 7 then
    (if k == j then (-1/2, 0) else if k == j+8 then (0, -1/2) else Q0)
  else -- 9..15
    (if k == j-8 then (0, 1/2) else if k == j then (-1/2, 0) else Q0)
def psiVec (u : Nat → Q3) : Nat → Q3 := fun k =>
  (List.range 16).foldl (fun acc j => qadd acc (qmul (u j) (psi j k))) Q0
def veq (u v : Nat → Q3) : Bool := (List.range 16).all (fun k => u k == v k)

/-- ψ is a genuine sedenion automorphism: ψ(e_j e_k) = ψ(e_j) ψ(e_k) for all 256 basis pairs. -/
theorem psi_automorphism :
    (List.range 16).all (fun j => (List.range 16).all (fun k =>
      veq (psiVec (smul (eunit j) (eunit k))) (smul (psi j) (psi k)))) = true := by native_decide

/-- ψ has order 3. -/
theorem psi_order_3 :
    (List.range 16).all (fun i => veq (psiVec (psiVec (psi i))) (eunit i))
    && (List.range 16).any (fun i => ! veq (psi i) (eunit i)) = true := by native_decide

/-- ψ is NON-MONOMIAL: some ψ(e_j) has a nonzero √3 component (so it is not a signed permutation). -/
theorem psi_non_monomial :
    (List.range 16).any (fun j => (List.range 16).any (fun k => (psi j k).2 != 0)) = true := by native_decide

/-- ψ maps the gen-1 raising ladder A†_1 = e₁+ie₅+e₉+ie₁₃ to the gen-2 ladder
    B†_1 = a e₁ + i a e₅ + b e₉ + i b e₁₃, a=(√3−1)/2, b=(−√3−1)/2 (Gresnigt eq 70-76).
    Real and imaginary parts are ℝ-linear under ψ; check both components. -/
def Ad1_re : Nat → Q3 := fun k => if k == 1 then Q1 else if k == 9 then Q1 else Q0
def Ad1_im : Nat → Q3 := fun k => if k == 5 then Q1 else if k == 13 then Q1 else Q0
def a3 : Q3 := (-1/2, 1/2)     -- (√3-1)/2
def b3 : Q3 := (-1/2, -1/2)    -- (-√3-1)/2
def Bd1_re : Nat → Q3 := fun k => if k == 1 then a3 else if k == 9 then b3 else Q0
def Bd1_im : Nat → Q3 := fun k => if k == 5 then a3 else if k == 13 then b3 else Q0
theorem psi_maps_A_to_B :
    (veq (psiVec Ad1_re) Bd1_re) && (veq (psiVec Ad1_im) Bd1_im) = true := by native_decide

-- === Frame-relative mechanism: [phi_color, N] = 0 and [psi_family, N] != 0 (16-dim Gresnigt charge) ===
abbrev CQ := Q3 × Q3                                  -- (re, im) over Q(sqrt3)
def cq0 : CQ := (Q0, Q0)
def cqadd (x y : CQ) : CQ := (qadd x.1 y.1, qadd x.2 y.2)
def cqmul (x y : CQ) : CQ := (qsub (qmul x.1 y.1) (qmul x.2 y.2), qadd (qmul x.1 y.2) (qmul x.2 y.1))
def cqI : CQ := (Q0, Q1)
abbrev Mat := Nat → Nat → CQ
def Lu (a : Nat) : Mat := fun r c => if r == (a ^^^ c) then ((( (cdSigma a c 4 : Int) : Rat), 0), Q0) else cq0
def mmul (A B : Mat) : Mat := fun r c => (List.range 16).foldl (fun acc t => cqadd acc (cqmul (A r t) (B t c))) cq0
def madd (A B : Mat) : Mat := fun r c => cqadd (A r c) (B r c)
def msc (z : CQ) (A : Mat) : Mat := fun r c => cqmul z (A r c)
def Ldag (i : Nat) : Mat := madd (madd (Lu i) (msc cqI (Lu (i+4)))) (madd (Lu (i+8)) (msc cqI (Lu (i+12))))
def Llow (i : Nat) : Mat := madd (madd (msc ((-1,0),Q0) (Lu i)) (msc cqI (Lu (i+4)))) (madd (msc ((-1,0),Q0) (Lu (i+8))) (msc cqI (Lu (i+12))))
def Nop : Mat := madd (madd (mmul (Ldag 1) (Llow 1)) (mmul (Ldag 2) (Llow 2))) (mmul (Ldag 3) (Llow 3))
def gp : List Nat := [0,2,3,1,4,6,7,5,8,10,11,9,12,14,15,13]
def Phi : Mat := fun r c => if r == gp.getD c 0 then (Q1, Q0) else cq0
def Psi : Mat := fun r c => ((psi c r), Q0)
def commZero (X : Mat) : Bool :=
  (List.range 16).all (fun r => (List.range 16).all (fun c =>
    (mmul X Nop) r c == (mmul Nop X) r c))
/-- The monomial color-Weyl element phi COMMUTES with the gen-1 number operator N. -/
theorem phi_commutes_charge : commZero Phi = true := by native_decide
/-- The family generator psi does NOT commute with N (it carries N to the gen-2 operator). -/
theorem psi_not_commute_charge : commZero Psi = false := by native_decide

end SounioGresnigtFamilyS3
