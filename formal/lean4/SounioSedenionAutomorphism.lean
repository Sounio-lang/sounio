/-
  SounioSedenionAutomorphism — the group behind every 168 (Frente B), by Lean native_decide.

  The 168 = |PSL(2,7)| that pervades the sedenion geometry IS a genuine group: the sedenion
  SIGNED-AUTOMORPHISM group. A linear M in GL(d,2) (acting on indices 1..2^d-1 as F_2^d vectors,
  preserving xor) is a signed automorphism iff sigma(Mi,Mj)*sigma(i,j) is a coboundary. We count them
  by rigorous highest-bit-pivot F_2 elimination:
    * octonions {1..7}: 168 = |GL(3,2)|          (oct_168)
    * sedenions {1..15}: 168 (of |GL(4,2)|=20160)  (sed_168)  — the SAME 168
    * ALL sedenion autos FIX e8 (index 8)          (sed_fix_e8_168) — e8 is the unique fixed point.
  So 168 is Aut ~ PSL(2,7); 1848 = 11*168 is NOT a group order (the 11 is combinatorial).

  NOT a @[default_target]: three native_decide sweeps over GL(4,2)=65536 matrices take ~1 min; build
  on demand with `lake build SounioSedenionAutomorphism` (mirrors SounioSatK76/DeGreyUnitDistance).
  Mathlib-free, no sorry. Companion to tests/run-pass/sedenion_automorphism_168.sio.
-/
namespace SounioSedenionAutomorphism

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
def par (x : Nat) : Nat := (List.range 8).foldl (fun c b => if x &&& (2^b) != 0 then c+1 else c) 0 % 2
def apply (M : List Nat) (x d : Nat) : Nat :=
  (List.range d).foldl (fun y r => if par ((M.getD r 0) &&& x) == 1 then y ||| (2^r) else y) 0
def invertible (M : List Nat) (d : Nat) : Bool := Id.run do
  let mut rows := M; let mut rank := 0
  for col in List.range d do
    match (List.range d).find? (fun r => r ≥ rank && (rows.getD r 0) &&& (2^col) != 0) with
    | some piv =>
        let rr := rows.getD rank 0; let rp := rows.getD piv 0
        rows := (rows.set rank rp).set piv rr
        for r in List.range d do
          if r != rank && (rows.getD r 0) &&& (2^col) != 0 then rows := rows.set r ((rows.getD r 0) ^^^ (rows.getD rank 0))
        rank := rank + 1
    | none => pure ()
  return rank == d
/-- rigorous highest-variable-bit pivoting (order-independent). -/
def isAuto (M : List Nat) (d : Nat) : Bool := Id.run do
  let n := 2^d
  let mut piv : Array Nat := Array.replicate 16 0
  for i in List.range n do
    if i ≥ 1 then
      for j in List.range n do
        if j > i then
          let Mi := apply M i d; let Mj := apply M j d; let k := i ^^^ j
          let s := cdSigma Mi Mj d * cdSigma i j d
          let mut m := (2^i) ||| (2^j) ||| (2^k) ||| (if s < 0 then 1 else 0)
          let mut done := false
          for _ in List.range (n+2) do
            if !done then
              let vpart := m >>> 1
              if vpart == 0 then
                if m &&& 1 != 0 then return false
                done := true
              else
                let h := Nat.log2 vpart
                if piv.getD h 0 == 0 then piv := piv.set! h m; done := true
                else m := m ^^^ (piv.getD h 0)
  return true
def matsOf (d : Nat) : List (List Nat) :=
  let n := 2^d
  match d with
  | 3 => (List.range n).flatMap (fun a => (List.range n).flatMap (fun b => (List.range n).map (fun c => [a,b,c])))
  | _ => (List.range n).flatMap (fun a => (List.range n).flatMap (fun b =>
           (List.range n).flatMap (fun c => (List.range n).map (fun e => [a,b,c,e]))))
def octAutos : Nat := ((matsOf 3).filter (fun M => invertible M 3 && isAuto M 3)).length
def sedAutos : Nat := ((matsOf 4).filter (fun M => invertible M 4 && isAuto M 4)).length
def sedFixE8 : Nat := ((matsOf 4).filter (fun M => invertible M 4 && isAuto M 4 && apply M 8 4 == 8)).length

theorem oct_168 : octAutos = 168 := by native_decide
theorem sed_168 : sedAutos = 168 := by native_decide
theorem sed_fix_e8_168 : sedFixE8 = 168 := by native_decide

end SounioSedenionAutomorphism
