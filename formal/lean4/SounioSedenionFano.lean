/-
  SounioSedenionFano — the 7 fibers ARE the Fano plane PG(2,2), and Aut(S) is its collineation group
  (Frente B, vector 1). Corollary of SounioSedenionAutomorphism: the 168 signed automorphisms fix e8,
  so they act on the fiber labels L = lo^hi in {9..15} = {8 XOR t : t in 1..7} via t -> M(t) on the
  lower 3 bits = F_2^3 \ {0} = the 7 Fano points. This file native_decide-proves that action is
  faithful (168 distinct fiber-permutations), transitive, and permutes the 7 Fano lines {a,b,a^b}.
  Mathlib-free, no sorry. NON-default_target (re-sweeps GL(4,2); ~1 min): `lake build SounioSedenionFano`.
-/
import SounioSedenionAutomorphism
open SounioSedenionAutomorphism

namespace SounioSedenionFano

/-- the 168 sedenion signed automorphisms, as matrices. -/
def autos : List (List Nat) := (matsOf 4).filter (fun M => invertible M 4 && isAuto M 4)

/-- action of M on the 7 fibers, as the tuple of images of fiber labels 9..15, reduced to lower 3 bits. -/
def fiberPerm (M : List Nat) : List Nat := (List.range 7).map (fun t => (apply M (8 ||| (t+1)) 4) &&& 7)

/-- the 7 Fano lines {a, b, a^b} of F_2^3 \ {0}. -/
def fanoLines : List (List Nat) :=
  (([(1,2),(1,4),(2,4),(1,6),(3,4),(1,7),(2,5)] : List (Nat × Nat)).map
    (fun (a,b) => let l := [a, b, a ^^^ b]; l)).eraseDups

def lineImg (M : List Nat) (l : List Nat) : List Nat := (l.map (fun t => (apply M (8 ||| t) 4) &&& 7))

-- helper: is a 3-set (as sorted list) a Fano line?
def isLine (s : List Nat) : Bool := s.length == 3 && ((s.foldl (· ^^^ ·) 0) == 0)

/-- faithful: the 168 automorphisms give 168 distinct permutations of the 7 fibers. -/
theorem fibers_faithful : (autos.map fiberPerm).eraseDups.length = 168 := by native_decide

/-- transitive: fiber 1 reaches all 7 fibers. -/
theorem fibers_transitive : (autos.map (fun M => (apply M (8 ||| 1) 4) &&& 7)).eraseDups.length = 7 := by native_decide

/-- the group permutes the 7 Fano lines: for every auto and every triple {a,b,a^b}, the image xors to 0. -/
theorem fano_lines_preserved :
    autos.all (fun M => (List.range 7).all (fun a => (List.range 7).all (fun b =>
      (a == b) || (((apply M (8 ||| (a+1)) 4) &&& 7) ^^^ ((apply M (8 ||| (b+1)) 4) &&& 7)
                   ^^^ ((apply M (8 ||| ((a+1) ^^^ (b+1))) 4) &&& 7) == 0)))) = true := by
  native_decide

end SounioSedenionFano
