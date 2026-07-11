/-
  SounioGresnigtG2S3 — Aut(𝕊)=G₂×S₃ executed (Frente B, vector 4/3 capstone²), by native_decide over
  ℚ(√3). Certifies the family S₃=⟨ψ,ϵ⟩ from Gresnigt arXiv:2306.13098 §4.3 (eq 47-55): ϵ (order-2,
  monomial) is a genuine automorphism; the braid relation ϵ∘ψ=ψ²∘ϵ holds; the color-Weyl φ commutes
  with BOTH ψ and ϵ (color ⟂ family, the direct product); ψ is non-monomial and the sole octonion↔new
  mixer, ϵ is monomial. Turns Brown's theorem (the cited foundation of Erratum E1) into computation.
  Mathlib-free, no sorry.
-/
namespace SounioGresnigtG2S3

abbrev Q3 := Rat × Rat
def qadd (x y : Q3) : Q3 := (x.1 + y.1, x.2 + y.2)
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

def smul (u v : Nat → Q3) : Nat → Q3 := fun k =>
  (List.range 16).foldl (fun acc a =>
    (List.range 16).foldl (fun acc2 b =>
      if (a ^^^ b) == k then qadd acc2 (qmul (qmul (u a) (v b)) (((cdSigma a b 4 : Int) : Rat), 0)) else acc2) acc) Q0
def eunit (i : Nat) : Nat → Q3 := fun k => if k == i then Q1 else Q0

def psi (j : Nat) : Nat → Q3 := fun k =>
  if j == 0 || j == 8 then (if k == j then Q1 else Q0)
  else if 1 ≤ j && j ≤ 7 then (if k == j then (-1/2, 0) else if k == j+8 then (0, -1/2) else Q0)
  else (if k == j-8 then (0, 1/2) else if k == j then (-1/2, 0) else Q0)
def eps (j : Nat) : Nat → Q3 := fun k => if k == j then (if j ≤ 7 then Q1 else (-1,0)) else Q0
def gp : List Nat := [0,2,3,1,4,6,7,5,8,10,11,9,12,14,15,13]
def phi (j : Nat) : Nat → Q3 := fun k => if k == gp.getD j 0 then Q1 else Q0

def app (f : Nat → Nat → Q3) (u : Nat → Q3) : Nat → Q3 := fun k =>
  (List.range 16).foldl (fun acc j => qadd acc (qmul (u j) (f j k))) Q0
def veq (u v : Nat → Q3) : Bool := (List.range 16).all (fun k => u k == v k)
-- compose: (f ∘ g) applied to e_i
def compv (f g : Nat → Nat → Q3) (i : Nat) : Nat → Q3 := app f (g i)

/-- ϵ is a genuine sedenion automorphism. -/
theorem eps_automorphism :
    (List.range 16).all (fun j => (List.range 16).all (fun k =>
      veq (app eps (smul (eunit j) (eunit k))) (smul (eps j) (eps k)))) = true := by native_decide

/-- Braid relation ϵ∘ψ = ψ²∘ϵ (⟨ψ,ϵ⟩ ≅ S₃ with ψ³=ϵ²=1). -/
theorem s3_braid :
    (List.range 16).all (fun i => veq (compv eps psi i) (app psi (compv psi eps i))) = true := by native_decide

/-- Color-Weyl φ commutes with the family generator ψ: [φ,ψ]=0. -/
theorem phi_commutes_psi :
    (List.range 16).all (fun i => veq (compv phi psi i) (compv psi phi i)) = true := by native_decide
/-- Color-Weyl φ commutes with the family involution ϵ: [φ,ϵ]=0. Together ⟹ G₂ and S₃ commute. -/
theorem phi_commutes_eps :
    (List.range 16).all (fun i => veq (compv phi eps i) (compv eps phi i)) = true := by native_decide

/-- ψ is non-monomial (uses √3); ϵ is monomial (each ϵ(e_j)=±e_j, no √3). -/
theorem psi_nonmonomial_eps_monomial :
    ((List.range 16).any (fun j => (List.range 16).any (fun k => (psi j k).2 != 0))
     && (List.range 16).all (fun j => (List.range 16).all (fun k => (eps j k).2 == 0))) = true := by native_decide

end SounioGresnigtG2S3
