/-
  SounioSedenionQuartets — independent-spec (Lean native_decide) leg for the 42 support-quartets of
  the sedenion zero-divisor geometry (Frente B). The 168 unordered ZD pairs group by support-union
  into exactly 42 quartets, each a 4-set with 2 lower + 2 upper indices, each hosting exactly 4 pairs.
  Companion to tests/run-pass/sedenion_zd_quartets.sio + docs/research/sedenion_zd_quartets.md.
  Mathlib-free, no sorry.
-/
namespace SounioSedenionQuartets

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
def cands : List V :=
  (List.range 7).flatMap (fun i => (List.range 8).flatMap (fun j => [(i+1, j+8, false), (i+1, j+8, true)]))
def adjacent (u v : V) : Bool := isZeroPair u.1 u.2.1 u.2.2 v.1 v.2.1 v.2.2
def participates (c : V) : Bool := cands.any (fun d => adjacent c d)
def parts : List V := cands.filter participates

/-- support-quartet bitmask of a pair (union of the 4 support indices). -/
def qmaskOf (a b : V) : Nat := (2^a.1) ||| (2^a.2.1) ||| (2^b.1) ||| (2^b.2.1)

/-- the 168 unordered annihilation-pair masks. -/
def masks : List Nat :=
  (List.range parts.length).flatMap (fun i => (List.range parts.length).filterMap (fun j =>
    if i < j && adjacent parts[i]! parts[j]! then some (qmaskOf parts[i]! parts[j]!) else none))

def popc (m : Nat) : Nat := (List.range 16).foldl (fun c b => if m &&& (2^b) != 0 then c+1 else c) 0
def lowerc (m : Nat) : Nat := (List.range 8).foldl (fun c b => if b ≥ 1 && m &&& (2^b) != 0 then c+1 else c) 0

theorem pairs_168 : masks.length = 168 := by native_decide
theorem quartets_42 : masks.eraseDups.length = 42 := by native_decide
/-- every quartet is a 4-set with exactly 2 lower indices, and hosts exactly 4 pairs. -/
theorem quartets_structure :
    masks.eraseDups.all (fun m => popc m == 4 && lowerc m == 2 && (masks.filter (· == m)).length == 4) = true := by
  native_decide

end SounioSedenionQuartets
