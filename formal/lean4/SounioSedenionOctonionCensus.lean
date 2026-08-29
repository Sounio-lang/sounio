/-
  SounioSedenionOctonionCensus — the basis-aligned octonion census in the sedenion (Frente B, vector
  4/3), by Lean native_decide. Of the 15 basis-aligned 3-dim F₂ index-subspaces: 8 are zero-divisor-free
  (genuine octonions / division algebras), 7 are quasi-octonions carrying zero divisors (Cawagas 2004),
  and exactly ONE is Clifford-pure (all 21 internal ambient left-mult pairs anticommute) — the base
  octonion {1..7}. Through the base quaternion {1,2,3} the three copies give L-non-anti counts {0,6,12}.
  A verified fingerprint consistent with (illustrating, not proving) Erratum E1. Mathlib-free, no sorry.
-/
namespace SounioSedenionOctonionCensus

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
def antiL (i j : Nat) : Bool :=
  (List.range 16).all (fun c => sg i (j ^^^ c) * sg j c + sg j (i ^^^ c) * sg i c == 0)

/-- (e_ul + us·e_uh)(e_vl + vs·e_vh) component on e_k. -/
def pprod (ul uh us vl vh vs k : Int) : Int :=
  (if (ul.toNat ^^^ vl.toNat) == k.toNat then sg ul.toNat vl.toNat else 0)
  + (if (ul.toNat ^^^ vh.toNat) == k.toNat then vs * sg ul.toNat vh.toNat else 0)
  + (if (uh.toNat ^^^ vl.toNat) == k.toNat then us * sg uh.toNat vl.toNat else 0)
  + (if (uh.toNat ^^^ vh.toNat) == k.toNat then us * vs * sg uh.toNat vh.toNat else 0)

/-- all 2-support primitives (a<b in S, sign ±1). -/
def prims (S : List Nat) : List (Nat × Nat × Int) :=
  let n := S.length
  (List.range n).flatMap (fun i => (List.range n).flatMap (fun j =>
    if i < j then [(S.getD i 0, S.getD j 0, (1:Int)), (S.getD i 0, S.getD j 0, (-1:Int))] else []))

/-- does the subspace S contain a zero-divisor pair? -/
def hasZD (S : List Nat) : Bool :=
  let ps := prims S
  let m := ps.length
  (List.range m).any (fun i => (List.range m).any (fun j =>
    if i < j then
      let u := ps.getD i (0,0,1); let v := ps.getD j (0,0,1)
      (List.range 16).all (fun k =>
        pprod (Int.ofNat u.1) (Int.ofNat u.2.1) u.2.2 (Int.ofNat v.1) (Int.ofNat v.2.1) v.2.2 (Int.ofNat k) == 0)
    else false))

/-- non-anticommuting internal ambient-L pairs of a 7-element set. -/
def nonanti (S : List Nat) : Nat :=
  let n := S.length
  ((List.range n).flatMap (fun i => (List.range n).filterMap (fun j =>
    if i < j then some (S.getD i 0, S.getD j 0) else none))).filter (fun p => ! antiL p.1 p.2) |>.length

def span7 (a b c : Nat) : List Nat := [a, b, a ^^^ b, c, a ^^^ c, b ^^^ c, a ^^^ b ^^^ c]
def maskOf (S : List Nat) : Nat := S.foldl (fun m x => m ||| (2 ^ x)) 0

/-- the 15 basis-aligned 3-dim F₂ subspaces (deduped by membership mask). -/
def subspaces : List (List Nat) :=
  let raw := (List.range 16).flatMap (fun a => (List.range 16).flatMap (fun b =>
    (List.range 16).filterMap (fun c =>
      if 1 ≤ a ∧ a < b ∧ b < c ∧ c ≠ (a ^^^ b) then some (span7 a b c) else none)))
  (raw.foldl (fun acc S => if acc.any (fun T => maskOf T == maskOf S) then acc else S :: acc) []).reverse

theorem nsub_15 : subspaces.length = 15 := by native_decide
/-- Exactly 8 are zero-divisor-free (genuine octonions); the other 7 are quasi-octonions (Cawagas). -/
theorem zdfree_8 : (subspaces.filter (fun S => ! hasZD S)).length = 8 := by native_decide
/-- Exactly one subspace is Clifford-pure, and it is the base octonion {1..7}. -/
theorem pure_1 : (subspaces.filter (fun S => nonanti S == 0)).length = 1 := by native_decide
theorem base_octonion_pure : nonanti (span7 1 2 4) = 0 := by native_decide
/-- Through the base quaternion {1,2,3}, the three octonion copies give L-non-anti counts {0,6,12}. -/
theorem quaternion_triple_0_6_12 :
    (nonanti [1,2,3,4,5,6,7], nonanti [1,2,3,8,9,10,11], nonanti [1,2,3,12,13,14,15]) = (0,6,12) := by
  native_decide

end SounioSedenionOctonionCensus
