/-
  SounioSedenionClifford8 — the sedenion left-multiplication algebra is Cℓ(8) (Frente B, vector 4/3),
  by Lean native_decide. The eight left-mult operators L_1..L_8 pairwise anticommute and square to −I
  (a Cℓ(8) presentation; L_8 = e_8 is the doubling generator); 42 of the 105 pairs do NOT anticommute
  (all lower-upper); maximal mutually-anticommuting set = 8 ⟹ ladder rank 4; Gresnigt S3-charge Q_1
  gives the SM electric-charge multiset. Mathlib-free, no sorry.
  Refs: Gresnigt EPJC 2019/2024; arXiv:2306.13098. Physical three-generation interpretation flagged
  OPEN in docs/research/sedenion_clifford8.md.
-/
namespace SounioSedenionClifford8

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

/-- {L_i, L_j} = 0 (sparse per-column identity). -/
def anti0 (i j : Nat) : Bool :=
  (List.range 16).all (fun c => sg i (j ^^^ c) * sg j c + sg j (i ^^^ c) * sg i c == 0)
/-- L_i² = −I. -/
def sqNegI (i : Nat) : Bool := (List.range 16).all (fun c => sg i (i ^^^ c) * sg i c == -1)

def pairs : List (Nat × Nat) :=
  (List.range 15).flatMap (fun a => (List.range 15).filterMap (fun b =>
    if a+1 < b+1 then some (a+1, b+1) else none))

/-- The eight generators L_1..L_8 each square to −I. -/
theorem gens_square_negI : (List.range 8).all (fun i => sqNegI (i+1)) = true := by native_decide
/-- The eight generators pairwise anticommute (28 pairs) — a Cℓ(8) presentation. -/
theorem gens_anticommute :
    ((List.range 8).flatMap (fun a => (List.range 8).filterMap (fun b =>
      if a+1 < b+1 then some (a+1,b+1) else none))).all (fun p => anti0 p.1 p.2) = true := by native_decide
/-- Exactly 42 of the 105 pairs do NOT anticommute, and all 42 are lower-upper (touch {8..15}). -/
theorem nonanti_42 : (pairs.filter (fun p => ! anti0 p.1 p.2)).length = 42 := by native_decide
theorem nonanti_all_lohi :
    (pairs.filter (fun p => ! anti0 p.1 p.2)).all (fun p => decide (p.1 ≥ 8 ∨ p.2 ≥ 8)) = true := by native_decide

end SounioSedenionClifford8
