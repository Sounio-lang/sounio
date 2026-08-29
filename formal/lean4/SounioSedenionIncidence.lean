/-
  SounioSedenionIncidence — independent-spec (Lean native_decide) leg for the quartet<->fiber
  incidence of the sedenion ZD geometry (Frente B): the 42 support-quartets, as edges on the 7
  fibers L = lo^hi in {9..15}, form exactly 2*K_7 — every C(7,2)=21 fiber-pair joined by exactly 2
  quartets, every fiber of incidence-degree 12. Companion to
  tests/run-pass/sedenion_quartet_fiber_incidence.sio. Mathlib-free, no sorry.
-/
namespace SounioSedenionIncidence

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
def qmaskOf (a b : V) : Nat := (2^a.1) ||| (2^a.2.1) ||| (2^b.1) ||| (2^b.2.1)

/-- (mask, fiber-label) for each of the 168 unordered annihilation pairs (intra-fiber: both share L). -/
def pairData : List (Nat × Nat) :=
  (List.range parts.length).flatMap (fun i => (List.range parts.length).filterMap (fun j =>
    if i < j && adjacent parts[i]! parts[j]! then
      some (qmaskOf parts[i]! parts[j]!, parts[i]!.1 ^^^ parts[i]!.2.1) else none))

def masks : List Nat := (pairData.map (·.1)).eraseDups

/-- the 2 fiber-labels a quartet touches, as (min,max). -/
def fibersOf (mask : Nat) : Nat × Nat :=
  let ls := (pairData.filter (·.1 == mask)).map (·.2) |>.eraseDups
  (ls.foldl Nat.min 99, ls.foldl Nat.max 0)

def fiberPairs : List (Nat × Nat) := (masks.map fibersOf).eraseDups

theorem pairs_168 : pairData.length = 168 := by native_decide
/-- every quartet spans exactly 2 distinct fibers. -/
theorem each_quartet_spans_2 :
    masks.all (fun m => ((pairData.filter (·.1 == m)).map (·.2)).eraseDups.length == 2) = true := by native_decide
/-- the 42 quartets form 2*K_7 on the 7 fibers: all 21 fiber-pairs, each carrying exactly 2 quartets. -/
theorem fiberpairs_21 : fiberPairs.length = 21 := by native_decide
theorem two_per_fiberpair :
    fiberPairs.all (fun fp => (masks.filter (fun m => fibersOf m == fp)).length == 2) = true := by native_decide

end SounioSedenionIncidence
