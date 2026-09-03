/-
  SounioCDTowerSeam — the e_top-seam bridge across the Cayley-Dickson tower, by native_decide.
  The forward obstruction ({L_l,L_u}=0 ⟹ e_l+e_u not a ZD) is dimension-independent linear algebra GIVEN
  the cocycle lemma L_i²=−I, i.e. σ(i,j)·σ(i,i⊕j)=−1. That lemma is certified here at dim 16/32/64
  (n=4,5,6); a Mathlib-free general induction on `bits`, and the CONVERSE (off-seam ⟹ ZD) for all n,
  are left open (a tower-wide conjecture). The operator/zero-divisor/off-seam coincidence is verified at
  dim 16 (four members incl. the explicit ZD scan) and dim 32 (operator/seam; the 32- and 64-dim ZD scan
  is carried by the Python oracle). Mathlib-free, no sorry.
-/
namespace SounioCDTowerSeam

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

-- cocycle lemma L_i^2 = -I : σ(i,j)·σ(i,i⊕j) = -1 for all i≥1, j (< 2^bits)
def lsqOK (bits : Nat) : Bool :=
  let n := 2 ^ bits
  (List.range n).all (fun i => i == 0 || (List.range n).all (fun j =>
    cdSigma i j bits * cdSigma i (i ^^^ j) bits == -1))
theorem lsq_16 : lsqOK 4 = true := by native_decide
theorem lsq_32 : lsqOK 5 = true := by native_decide
theorem lsq_64 : lsqOK 6 = true := by native_decide

-- operator / seam / zero-divisor predicates at a given dim
def anti0 (bits l u : Nat) : Bool :=
  let n := 2 ^ bits
  (List.range n).all (fun c => cdSigma l (u ^^^ c) bits * cdSigma u c bits
    + cdSigma u (l ^^^ c) bits * cdSigma l c bits == 0)
def llsqNegI (bits l u : Nat) : Bool :=
  let n := 2 ^ bits
  (List.range n).all (fun c => cdSigma l (l ^^^ c) bits * cdSigma u (l ^^^ u ^^^ c) bits
    * cdSigma l (u ^^^ c) bits * cdSigma u c bits == -1)
def offSeam (bits l u : Nat) : Bool := let top := 2 ^ (bits - 1); ! (u == top || (l ^^^ u) == top)
def annih (bits l u a b : Nat) (s : Int) : Bool :=
  let n := 2 ^ bits
  (List.range n).all (fun k =>
    (if (l ^^^ a) == k then cdSigma l a bits else 0) + (if (l ^^^ b) == k then s * cdSigma l b bits else 0)
    + (if (u ^^^ a) == k then cdSigma u a bits else 0) + (if (u ^^^ b) == k then s * cdSigma u b bits else 0) == 0)
def isZD (bits l u : Nat) : Bool :=
  let n := 2 ^ bits
  (List.range n).any (fun a => (List.range n).any (fun b =>
    a < b && (annih bits l u a b 1 || annih bits l u a b (-1))))
def loHi (bits : Nat) : List (Nat × Nat) :=
  let top := 2 ^ (bits - 1)
  (List.range (top - 1)).flatMap (fun l => (List.range top).map (fun u => (l + 1, u + top)))

/-- Dim 16: the four members coincide — {L_l,L_u}=0 ⟺ (L_lL_u)²=−I ⟺ NOT off-seam ⟺ NOT a ZD. -/
theorem coincidence_16 :
    (loHi 4).all (fun p =>
      (anti0 4 p.1 p.2 == llsqNegI 4 p.1 p.2)
      && (anti0 4 p.1 p.2 == ! offSeam 4 p.1 p.2)
      && (anti0 4 p.1 p.2 == ! isZD 4 p.1 p.2)) = true := by native_decide
/-- Dim 32: the operator/(L L)²/seam members coincide (the ZD scan is carried by the oracle). -/
theorem coincidence_32 :
    (loHi 5).all (fun p =>
      (anti0 5 p.1 p.2 == llsqNegI 5 p.1 p.2)
      && (anti0 5 p.1 p.2 == ! offSeam 5 p.1 p.2)) = true := by native_decide

end SounioCDTowerSeam
