/-
  SounioSedenionAssociator1848 — independent-spec (Lean native_decide) leg for the ASSOCIATOR side
  of the sedenion tower: 1848 = 11 * 168 ordered non-associative basis triples (Frente B).

  Confirms the open conjecture of SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT.md — that the factor 11 in
  1848 = 11*168 lives on the associator side, not the zero-divisor side. Companion to
  tests/run-pass/sedenion_associator_1848.sio + docs/research/sedenion_associator_1848.md.
  Mathlib-free, no sorry. cdSigma aligned to SounioCayleyDickson.lean.
-/
namespace SounioSedenionAssociator1848

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

def sig (a b : Nat) : Int := cdSigma a b 4

/-- Associator coefficient of `e_(i^j^k)` in `[e_i,e_j,e_k]`; in `{-2,0,+2}`. -/
def assoc (i j k : Nat) : Int := sig i j * sig (i ^^^ j) k - sig j k * sig i (j ^^^ k)

def units : List Nat := (List.range 15).map (· + 1)
def triples : List (Nat × Nat × Nat) :=
  units.flatMap (fun i => units.flatMap (fun j => units.filterMap (fun k =>
    if i != j && j != k && i != k then some (i, j, k) else none)))
def nonassoc : List (Nat × Nat × Nat) := triples.filter (fun t => assoc t.1 t.2.1 t.2.2 != 0)

def unordered : List (Nat × Nat × Nat) :=
  (List.range 15).flatMap (fun i => (List.range 15).flatMap (fun j => (List.range 15).filterMap (fun k =>
    if i+1 < j+1 && j+1 < k+1 then some (i+1, j+1, k+1) else none)))
def orderCount (i j k : Nat) : Nat :=
  (if assoc i j k != 0 then 1 else 0) + (if assoc i k j != 0 then 1 else 0)
  + (if assoc j i k != 0 then 1 else 0) + (if assoc j k i != 0 then 1 else 0)
  + (if assoc k i j != 0 then 1 else 0) + (if assoc k j i != 0 then 1 else 0)

/-- 1848 = 11 * 168 ordered non-associative sedenion basis triples. -/
theorem total_1848 : nonassoc.length = 1848 := by native_decide
/-- The doubling grade `i^j^k = 8` carries exactly 168 (= the octonion associator count). -/
theorem grade8_168 : (nonassoc.filter (fun t => (t.1 ^^^ t.2.1 ^^^ t.2.2) == 8)).length = 168 := by native_decide
/-- The other 14 grades carry 1680 = 10 * 168 together, so 1848 = (10+1)*168. -/
theorem other_1680 : (nonassoc.filter (fun t => (t.1 ^^^ t.2.1 ^^^ t.2.2) != 8)).length = 1680 := by native_decide
/-- The octonion sub-tower {1..7} contributes exactly 168 ordered non-associative triples. -/
theorem oct_168 : (nonassoc.filter (fun t => t.1 ≤ 7 && t.2.1 ≤ 7 && t.2.2 ≤ 7)).length = 168 := by native_decide

/-- Ordering-class decomposition of the 455 = C(15,3) unordered triples: 35 associative, 168 semi
    (exactly 2 non-associative orderings), 252 fully non-associative (all 6). -/
theorem class0_35  : (unordered.filter (fun t => orderCount t.1 t.2.1 t.2.2 == 0)).length = 35  := by native_decide
theorem class2_168 : (unordered.filter (fun t => orderCount t.1 t.2.1 t.2.2 == 2)).length = 168 := by native_decide
theorem class6_252 : (unordered.filter (fun t => orderCount t.1 t.2.1 t.2.2 == 6)).length = 252 := by native_decide
/-- The doubling grade 8 is fully non-associative on ALL its support-triples (this is why it carries
    168 = 28*6 while the other grades carry 120 = 16*6 + 12*2). -/
theorem grade8_all_full :
    (unordered.filter (fun t => (t.1 ^^^ t.2.1 ^^^ t.2.2) == 8 && orderCount t.1 t.2.1 t.2.2 != 6)).length = 0 := by
  native_decide

end SounioSedenionAssociator1848
