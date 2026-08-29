/-
  SounioFureyChargeG2 — the Furey Cℓ(6) charge operator and the G₂ automorphism φ, by native_decide.
  Builds (over Gaussian integers ℤ[i], scaled) the Furey ladder operators 2α_i = −L_{a_i} + i L_{b_i} of
  the complex octonions, verifies the Witt relations {2α_i,(2α_j)†}=4δ_ij·I and {2α_i,2α_j}=0, forms the
  (scaled) charge D = 12Q = Σ_i (2α_i)†(2α_i), and shows the G₂ automorphism φ = (e₁e₂e₃)(e₅e₆e₇)
  (from SounioSedenionGresnigtOctonions) does NOT commute with D: [P_φ, D] ≠ 0. So φ does not preserve
  the charge — it is not a charge-preserving (family) symmetry. Rules out THIS φ; the general bridge
  question needs Brown's rotational S₃ (open). Mathlib-free, no sorry.
-/
namespace SounioFureyChargeG2

abbrev C := Int × Int          -- Gaussian integer (re, im)
def cadd (x y : C) : C := (x.1 + y.1, x.2 + y.2)
def cmul (x y : C) : C := (x.1 * y.1 - x.2 * y.2, x.1 * y.2 + x.2 * y.1)
def cconj (x : C) : C := (x.1, - x.2)
def N : Nat := 8

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
def og (a b : Nat) : Int := cdSigma a b 3   -- octonion sign

-- matrices as functions Nat → Nat → C  (row, col), on 0..7
abbrev M := Nat → Nat → C
def Lmat (a : Nat) : M := fun r c => if r == (a ^^^ c) then (og a c, 0) else (0,0)
def mmul (A B : M) : M := fun r c => (List.range N).foldl (fun acc t => cadd acc (cmul (A r t) (B t c))) (0,0)
def madd (A B : M) : M := fun r c => cadd (A r c) (B r c)
def mscale (z : C) (A : M) : M := fun r c => cmul z (A r c)
def mdag (A : M) : M := fun r c => cconj (A c r)
def meq (A B : M) : Bool := (List.range N).all (fun r => (List.range N).all (fun c => A r c == B r c))
def mzero : M := fun _ _ => (0,0)

-- 2α_i = -L_{a_i} + i L_{b_i} for pairs (1,2),(3,4),(5,6)
def A2 (i : Nat) : M :=
  let ab := [(1,2),(3,4),(5,6)].getD i (0,0)
  madd (mscale (-1,0) (Lmat ab.1)) (mscale (0,1) (Lmat ab.2))
def anti (A B : M) : M := madd (mmul A B) (mmul B A)
def I4 : M := fun r c => if r == c then (4,0) else (0,0)

/-- Witt relations: {2α_i,(2α_j)†} = 4δ_ij·I and {2α_i,2α_j}=0. -/
theorem witt_relations :
    ([0,1,2].all (fun i => [0,1,2].all (fun j =>
      meq (anti (A2 i) (mdag (A2 j))) (if i == j then I4 else mzero)
      && meq (anti (A2 i) (A2 j)) mzero))) = true := by native_decide

def D : M := (List.range 3).foldl (fun acc i => madd acc (mmul (mdag (A2 i)) (A2 i))) mzero
-- φ on the octonion: g=(1 2 3)(5 6 7), fix 0,4,7? g fixes 0,4; permutation P e_j = e_{g j}
def gperm : Nat → Nat := fun x => [0,2,3,1,4,6,7,5].getD x 0
def P : M := fun r c => if r == gperm c then (1,0) else (0,0)

/-- φ does NOT commute with the charge: [P_φ, D] ≠ 0. -/
theorem phi_does_not_preserve_charge : meq (mmul P D) (mmul D P) = false := by native_decide

/-- One generation's charge multiplicities (Fock, ×3): 3·Q ∈ {0,1,1,1,2,2,2,3}. -/
def charge3 : List Nat := (List.range 8).map (fun s => (s % 2) + ((s/2) % 2) + ((s/4) % 2))
theorem charge_multiplicities :
    (charge3.filter (· == 0)).length = 1 ∧ (charge3.filter (· == 1)).length = 3
    ∧ (charge3.filter (· == 2)).length = 3 ∧ (charge3.filter (· == 3)).length = 1 := by native_decide

end SounioFureyChargeG2
