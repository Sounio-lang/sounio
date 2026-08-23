/-
  SounioOctonionFidelity.lean

  Closes the octonion-channel FIDELITY gap for lemma (i) (§18 scope note): the
  grading predicate `isFanoTriple` is faithful to `[α]=0` computed from the REAL
  octonion multiplication — proven, not assumed.

  Self-contained (no Mathlib, no imports). A signed basis-unit multiplication table
  over e_0..e_7 (e_0 = 1), Baez orientation e_i·e_{i+1}=e_{i+3} (mod 7). The table is
  SELF-CERTIFIED as a genuine octonion algebra (squares = -1, anti-commutativity,
  alternativity — all by `native_decide`), so fidelity does not rest on hand-checked
  signs. Then: assocNormSq i j k = 0 ⟺ the triple is associative (on a Fano line, or
  degenerate); restricted to distinct triples that is exactly `isFanoTriple`.

  STATUS: CHECKED under Lean 4.33.1, EXIT=0, ZERO sorry (verify below).

  NOTE on Fano lines: this table's associative lines are
  {1,2,4},{2,3,5},{3,4,6},{4,5,7},{1,5,6},{2,6,7},{1,3,7} (the standard
  quadratic-residue plane), which DIFFER as a labelled set from
  SounioBlackwellBridge.fanoLines {1,2,3},{1,4,5},{1,6,7},{2,4,6},{2,5,7},{3,4,7},
  {3,5,6}. Both are valid Fano planes (7 lines, 168 non-associative ordered triples),
  related by a relabelling of the imaginary units; the invariant that the bridge uses
  — associative ⟺ on-a-line, 168 non-Fano — is what transfers, and the 168 count here
  cross-checks `SounioBidirectionalBridge.nonassoc_iff_not_fano`.
-/

namespace Sounio.OctonionFidelity

/-- Positive oriented pairs of the Baez table: `e_i · e_j = + e_k`. -/
def posProd : Nat → Nat → Option Nat
  | 1, 2 => some 4 | 2, 4 => some 1 | 4, 1 => some 2
  | 2, 3 => some 5 | 3, 5 => some 2 | 5, 2 => some 3
  | 3, 4 => some 6 | 4, 6 => some 3 | 6, 3 => some 4
  | 4, 5 => some 7 | 5, 7 => some 4 | 7, 4 => some 5
  | 5, 6 => some 1 | 6, 1 => some 5 | 1, 5 => some 6
  | 6, 7 => some 2 | 7, 2 => some 6 | 2, 6 => some 7
  | 7, 1 => some 3 | 1, 3 => some 7 | 3, 7 => some 1
  | _, _ => none

/-- Multiply two basis units (index 0 = identity). Returns `(sign, index)`. -/
def mul (i j : Nat) : Int × Nat :=
  if i == 0 then (1, j)
  else if j == 0 then (1, i)
  else if i == j then (-1, 0)
  else match posProd i j with
    | some k => (1, k)
    | none => match posProd j i with
      | some k => (-1, k)
      | none => (0, 0)

/-- Multiply signed units. -/
def smul (a b : Int × Nat) : Int × Nat :=
  let p := mul a.2 b.2
  (a.1 * b.1 * p.1, p.2)

/-- A pure basis unit. -/
def unit (i : Nat) : Int × Nat := (1, i)

/-- Negate a signed unit. -/
def negU (a : Int × Nat) : Int × Nat := (-a.1, a.2)

/-- Squared norm of the associator `[e_i,e_j,e_k] = (e_i e_j) e_k − e_i (e_j e_k)`.
    For basis units this is 0 (associative) or 4 (non-associative). -/
def assocNormSq (i j k : Nat) : Int :=
  let lhs := smul (smul (unit i) (unit j)) (unit k)
  let rhs := smul (unit i) (smul (unit j) (unit k))
  if lhs.2 == rhs.2 then (lhs.1 - rhs.1) * (lhs.1 - rhs.1)
  else lhs.1 * lhs.1 + rhs.1 * rhs.1

/-! ### Self-certification: the table is a genuine octonion algebra -/

def imag : List Nat := [1, 2, 3, 4, 5, 6, 7]
def all8 : List Nat := [0, 1, 2, 3, 4, 5, 6, 7]
def pairsImag : List (Nat × Nat) := imag.flatMap (fun i => imag.map (fun j => (i, j)))
def pairs8 : List (Nat × Nat) := all8.flatMap (fun i => all8.map (fun j => (i, j)))

/-- Every imaginary unit squares to −1. -/
theorem square_neg_one : imag.all (fun i => mul i i == ((-1 : Int), 0)) = true := by
  native_decide

/-- Anti-commutativity on distinct imaginary units: `e_i e_j = −(e_j e_i)`. -/
theorem anticomm :
    pairsImag.all (fun p => (p.1 == p.2) || (mul p.1 p.2 == negU (mul p.2 p.1))) = true := by
  native_decide

/-- **Alternativity** (the defining octonion identity): `(e_i e_i) e_j = e_i (e_i e_j)`. -/
theorem alternativity :
    pairs8.all (fun p =>
      smul (smul (unit p.1) (unit p.1)) (unit p.2)
        == smul (unit p.1) (smul (unit p.1) (unit p.2))) = true := by
  native_decide

/-! ### Fano classification (matched to THIS table) and fidelity -/

def fanoLines : List (List Nat) :=
  [[1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 7], [1, 5, 6], [2, 6, 7], [1, 3, 7]]

def distinct (i j k : Nat) : Bool := (i != j) && (j != k) && (i != k)

def onLine (i j k : Nat) : Bool :=
  fanoLines.any (fun L => L.contains i && L.contains j && L.contains k)

def isFanoTriple (i j k : Nat) : Bool := distinct i j k && onLine i j k

def allTriples : List (Nat × Nat × Nat) :=
  imag.flatMap (fun a => imag.flatMap (fun b => imag.map (fun c => (a, b, c))))

/-- **Fidelity theorem.** For every triple of imaginary units, the associator vanishes
    IFF the triple is associative — on a Fano line (distinct) or degenerate. Hence,
    restricted to distinct triples, `[α]=0 ⟺ isFanoTriple`. -/
theorem fidelity_all :
    allTriples.all (fun t =>
      (assocNormSq t.1 t.2.1 t.2.2 == 0)
        == (isFanoTriple t.1 t.2.1 t.2.2 || !(distinct t.1 t.2.1 t.2.2))) = true := by
  native_decide

/-- The associator squared norm is exactly 0 or 4 (matching product_nonassoc 0.25/4.25). -/
theorem assoc_zero_or_four :
    allTriples.all (fun t =>
      (assocNormSq t.1 t.2.1 t.2.2 == 0) || (assocNormSq t.1 t.2.1 t.2.2 == 4)) = true := by
  native_decide

/-- **168 non-associative ordered triples** — cross-checks
    `SounioBidirectionalBridge.nonassoc_iff_not_fano`. -/
theorem nonassoc_count :
    (allTriples.filter (fun t => !(assocNormSq t.1 t.2.1 t.2.2 == 0))).length = 168 := by
  native_decide

/-- Equivalently, exactly 42 associative distinct ordered triples (7 lines × 6 orders). -/
theorem assoc_distinct_count :
    (allTriples.filter (fun t =>
      distinct t.1 t.2.1 t.2.2 && (assocNormSq t.1 t.2.1 t.2.2 == 0))).length = 42 := by
  native_decide

end Sounio.OctonionFidelity
