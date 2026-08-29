/-
  SounioSedenionFiberIdentity — independent-spec (Lean `native_decide`) leg for the ISOMORPHISM
  TYPE of the sedenion zero-divisor fibers (Frente B, brick 3).

  Companion to tests/run-pass/sedenion_zd_fiber_identity.sio + docs/research/sedenion_zd_fiber_identity.md
  + scripts/research/sedenion_zd_fiber_identity_oracle.py. Each of the 7 fibers is isomorphic to
  K_{6,6} minus a 2-factor of three disjoint 4-cycles; here Lean's kernel evaluator verifies the
  BFS-free signature — the common-neighbor profile (4:6, 2:24, 0:36) — over every fiber. Mathlib-free,
  no `sorry`. (The rigorous "complement = three 4-cycles" traversal is discharged by the Python oracle.)
-/
namespace SounioSedenionFiberIdentity

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
def fiberOf (L : Nat) : List V := parts.filter (fun c => (c.1 ^^^ c.2.1) == L)

/-- Number of common annihilators of `u,v` within the fiber `vs`. -/
def commonN (vs : List V) (u v : V) : Nat := (vs.filter (fun w => adjacent u w && adjacent v w)).length

/-- Common-neighbor profile `(n4, n2, n0)` over the unordered vertex-pairs of a fiber. -/
def profile (vs : List V) : Nat × Nat × Nat :=
  let ips := (List.range vs.length).flatMap (fun i => (List.range vs.length).filterMap (fun j =>
              if i < j then some (vs[i]!, vs[j]!) else none))
  ips.foldl (fun (acc : Nat × Nat × Nat) (p : V × V) =>
     let c := commonN vs p.1 p.2
     if c == 4 then (acc.1+1, acc.2.1, acc.2.2)
     else if c == 2 then (acc.1, acc.2.1+1, acc.2.2)
     else if c == 0 then (acc.1, acc.2.1, acc.2.2+1)
     else acc) (0,0,0)

/-- Every fiber `L = lo xor hi ∈ {9..15}` has common-neighbor profile `(6, 24, 36)` — the signature
    of `K_{6,6}` minus three disjoint 4-cycles (given brick 2's 4-regular bipartite 6+6 structure). -/
theorem fiber_profile :
    (List.range 7).all (fun t => profile (fiberOf (t+9)) == (6, 24, 36)) = true := by native_decide

end SounioSedenionFiberIdentity
