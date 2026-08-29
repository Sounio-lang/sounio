/-
  SounioSeamBridge — the e₈-seam bridge (Frente B), by native_decide. For every lower×upper index-pair
  (l,u) of 𝕊 the sparse equivalences hold: {L_l,L_u}=0 ⟺ (L_lL_u)²=−I ⟺ e_l+e_u is NOT a zero divisor
  ⟺ (l,u) is on the e₈ seam. Hence operator non-alternativity and state-level zero-division are ONE
  locus (the 42 off-seam pairs). The anticommutator is the exact obstruction to zero-division (forward
  proof: {L_l,L_u}=0 ⟹ (L_lL_u)²=−I ⟹ +1 ∉ spec ⟹ not a ZD). Mathlib-free, no sorry.
-/
namespace SounioSeamBridge

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
def sg (a b : Nat) : Int := cdSigma a b 4

def anti0 (l u : Nat) : Bool :=
  (List.range 16).all (fun c => sg l (u ^^^ c) * sg u c + sg u (l ^^^ c) * sg l c == 0)
def llsqNegI (l u : Nat) : Bool :=
  (List.range 16).all (fun c => sg l (l ^^^ c) * sg u (l ^^^ u ^^^ c) * sg l (u ^^^ c) * sg u c == -1)
def annih (l u a b : Nat) (s : Int) : Bool :=
  (List.range 16).all (fun k =>
    (if (l ^^^ a) == k then sg l a else 0) + (if (l ^^^ b) == k then s * sg l b else 0)
    + (if (u ^^^ a) == k then sg u a else 0) + (if (u ^^^ b) == k then s * sg u b else 0) == 0)
def isZD (l u : Nat) : Bool :=
  (List.range 16).any (fun a => (List.range 16).any (fun b =>
    a < b && (annih l u a b 1 || annih l u a b (-1))))
def offSeam (l u : Nat) : Bool := ! (u == 8 || (l ^^^ u) == 8)

def loHi : List (Nat × Nat) :=
  (List.range 7).flatMap (fun l => (List.range 8).map (fun u => (l+1, u+8)))

/-- The four sparse conditions are equivalent for all 56 lower×upper pairs:
    {L_l,L_u}=0 ⟺ (L_lL_u)²=−I ⟺ (l,u) NOT off-seam ⟺ e_l+e_u NOT a zero divisor. -/
theorem seam_equivalence :
    loHi.all (fun p =>
      (anti0 p.1 p.2 == llsqNegI p.1 p.2)
      && (anti0 p.1 p.2 == ! offSeam p.1 p.2)
      && (anti0 p.1 p.2 == ! isZD p.1 p.2)) = true := by native_decide

/-- The three 42-element sets coincide: non-anticommuting pairs = off-seam pairs = zero-divisor directions. -/
theorem three_42_sets :
    ((loHi.filter (fun p => ! anti0 p.1 p.2)).length = 42)
    ∧ ((loHi.filter (fun p => offSeam p.1 p.2)).length = 42)
    ∧ ((loHi.filter (fun p => isZD p.1 p.2)).length = 42) := by native_decide

end SounioSeamBridge
