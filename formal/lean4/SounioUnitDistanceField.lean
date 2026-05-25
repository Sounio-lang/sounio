/-!
# Sounio — Exact Certifier over K = ℚ(√3,√5,√7,√11) (Hadwiger–Nelson #508, Phase B)

The full-field exact unit-distance certifier. The base certifier
(`SounioUnitDistanceExact`) worked in ℚ(√3,√11); de Grey's construction — and
therefore the 510-vertex record graph that descends from it — actually lives in the
**degree-16** field

    K = ℚ(√3, √5, √7, √11),

with the four radicands entering one per assembly stage:
  √3  — the hexagonal/Eisenstein base lattice (H, J);
  √11 — the spindle-dense stage M/V (cos arcsin(1/√12) = √11/√12);
  √5  — the K-assembly rotation 2·arcsin(1/4): cos = 7/8, sin = √15/8, √15 = √3·√5;
  √7  — the L-assembly rotation 2·arcsin(1/8): cos arcsin(1/8) = √63/8 = 3√7/8.

(The √7 stage is easy to miss from secondary summaries; a certifier over the
degree-8 subfield ℚ(√3,√5,√11) would silently mis-measure every √7-bearing edge.
Field derived from de Grey, *The chromatic number of the plane is at least 5*,
arXiv:1804.02385.)

## Representation

K is the compositum of four linearly-disjoint real quadratic fields, Gal(K/ℚ) ≅
(ℤ/2)⁴. We index the 16-element ℚ-basis `{√d : d | 1155 squarefree}` by a 4-bit mask
over the primes (bit0=3, bit1=5, bit2=7, bit3=11): basis index `m` ↦ `√(∏ primes set
in m)`. Multiplication is the **XOR-graded** rule

    √a · √b = (∏ primes in a∩b) · √(a △ b),

i.e. result index `i ⊕ j` with rational factor `primeProduct(i &&& j)`. A K-element
is a sparse `List (index, ℤ-coeff)`; `kCoeff` sums a basis component (order/duplicate
independent), so all arithmetic is exact and float-free.

## What this proves (`native_decide`, no `sorry`, no Mathlib)

Field correctness, then the self-test fixture ladder climbing the field axis by axis
(the discipline that must pass *before* the 510-vertex graph is loaded):
* `field_roots_square`, `field_cross_15_21` — the multiplication table is correct.
* `spindle_*` — Moser spindle (ℚ(√3,√11)) recertified in K: 11 exact unit edges
  (converse-audited) and χ = 4.
* `degreyH_*` — de Grey's base graph H (ℚ(√3)): 12 edges, χ = 3.
* `witness_sqrt5/7/11` — three explicit unit edges whose exact certification *requires*
  the √5, √7, √11 axes respectively (a unit distance living in each new axis).

When the 510-vertex `.vtx` coordinates arrive, the same `isUnit` / converse-audit
machinery certifies every edge over this exact K; the realised field is then asserted
from the radicands the data actually uses (audit), not assumed.
-/

namespace Sounio.UnitDistanceField

-- ===========================================================================
-- §1. Exact arithmetic over K (sparse, XOR-graded, degree 16).
-- ===========================================================================

/-- Sparse K-element: list of (basis index 0..15, rational... here ℤ) terms. -/
abbrev KV := List (Nat × Int)

/-- Product of the primes selected by the 4-bit mask (bit0=3,1=5,2=7,3=11). -/
def primeProduct (mask : Nat) : Int :=
  (if mask &&& 1 != 0 then 3 else 1) * (if mask &&& 2 != 0 then 5 else 1) *
  (if mask &&& 4 != 0 then 7 else 1) * (if mask &&& 8 != 0 then 11 else 1)

def kadd (x y : KV) : KV := x ++ y
def kneg (x : KV) : KV := x.map (fun p => (p.1, -p.2))
def ksub (x y : KV) : KV := kadd x (kneg y)

/-- XOR-graded product: √a·√b = primeProduct(a∩b)·√(a△b). -/
def kmul (x y : KV) : KV :=
  x.flatMap (fun p => y.map (fun q => (p.1 ^^^ q.1, p.2 * q.2 * primeProduct (p.1 &&& q.1))))

/-- Exact coefficient of basis component `i` (sums duplicates; order-independent). -/
def kCoeff (x : KV) (i : Nat) : Int :=
  (x.filter (fun p => p.1 == i)).foldl (fun a p => a + p.2) 0

/-- `x` equals the rational scalar `v` exactly: component 0 is `v`, all 15 others 0. -/
def isScalar (x : KV) (v : Int) : Bool :=
  (List.range 16).all (fun i => kCoeff x i == (if i == 0 then v else 0))

/-- Basis element √(radicand of mask). -/
def root (mask : Nat) : KV := [(mask, 1)]
def scal (v : Int) : KV := if v == 0 then [] else [(0, v)]

-- ===========================================================================
-- §2. Field multiplication correctness.
-- ===========================================================================

/-- (√3)²=3, (√5)²=5, (√7)²=7, (√11)²=11. (bit0=3,1=5,2=7,3=11) -/
theorem field_roots_square :
    isScalar (kmul (root 1) (root 1)) 3 &&
    isScalar (kmul (root 2) (root 2)) 5 &&
    isScalar (kmul (root 4) (root 4)) 7 &&
    isScalar (kmul (root 8) (root 8)) 11 := by native_decide

/-- √15·√21 = 3√35: index (0b0011 ⊕ 0b0101)=0b0110 (=35), factor primeProduct(0b0001)=3. -/
theorem field_cross_15_21 :
    kCoeff (kmul (root 3) (root 5)) 6 == 3 &&
    (List.range 16).all (fun i => i == 6 || kCoeff (kmul (root 3) (root 5)) i == 0) := by
  native_decide

-- ===========================================================================
-- §3. Geometry over K.
-- ===========================================================================

abbrev Pt := KV × KV
def d2 (p q : Pt) : KV :=
  let dx := ksub p.1 q.1
  let dy := ksub p.2 q.2
  kadd (kmul dx dx) (kmul dy dy)
def isUnit (p q : Pt) (s2 : Int) : Bool := isScalar (d2 p q) s2

-- ===========================================================================
-- §4. Fixture 1 — Moser spindle (ℚ(√3,√11) ⊂ K), recertified. coords ×12, s²=144.
--     basis indices: 1 = √3, 8 = √11, 9 = √33.
-- ===========================================================================

def spindle : Nat → Pt
  | 0 => ([],            [])               -- A
  | 1 => ([(1,6)],       [(0,6)])          -- B = (√3/2,1/2)·12
  | 2 => ([(1,6)],       [(0,-6)])         -- C
  | 3 => ([(1,12)],      [])               -- D = (√3,0)·12
  | 4 => ([(1,5),(8,-1)],[(0,5),(9,1)])    -- E = ((5√3-√11)/12,(√33+5)/12)·12
  | 5 => ([(1,5),(8,1)], [(0,-5),(9,1)])   -- F
  | 6 => ([(1,10)],      [(9,2)])          -- G = (5√3/6,√33/6)·12
  | _ => ([],            [])

def verts : List Nat := [0,1,2,3,4,5,6]
def pairs : List (Nat × Nat) :=
  verts.flatMap (fun i => verts.filterMap (fun j => if i < j then some (i,j) else none))
def spindleEdges : List (Nat × Nat) :=
  [(0,1),(0,2),(0,4),(0,5),(1,2),(1,3),(2,3),(3,6),(4,5),(4,6),(5,6)]

theorem spindle_unit_edges :
    pairs.filter (fun p => isUnit (spindle p.1) (spindle p.2) 144) = spindleEdges := by
  native_decide

def colorOf (a k v : Nat) : Nat := (a / (k ^ v)) % k
def properSpindle (a k : Nat) : Bool :=
  pairs.all (fun p => (! isUnit (spindle p.1) (spindle p.2) 144) ||
                       colorOf a k p.1 != colorOf a k p.2)
def chiSpindle : Nat :=
  ((List.range 8).find? (fun k => (List.range (k ^ 7)).any (fun a => properSpindle a k))).getD 7
theorem spindle_chromatic_eq_4 : chiSpindle = 4 := by native_decide

-- ===========================================================================
-- §5. Fixture 2 — de Grey base graph H (ℚ(√3) ⊂ K). coords ×2, s²=4.
-- ===========================================================================

def degreyH : Nat → Pt
  | 0 => ([],       [])         -- O
  | 1 => ([(0,2)],  [])         -- (1,0)·2
  | 2 => ([(0,1)],  [(1,1)])    -- (1,√3)
  | 3 => ([(0,-1)], [(1,1)])    -- (-1,√3)
  | 4 => ([(0,-2)], [])         -- (-1,0)·2
  | 5 => ([(0,-1)], [(1,-1)])   -- (-1,-√3)
  | 6 => ([(0,1)],  [(1,-1)])   -- (1,-√3)
  | _ => ([],       [])

def degreyHEdges : List (Nat × Nat) :=
  [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(1,6),(2,3),(3,4),(4,5),(5,6)]

theorem degreyH_unit_edges :
    pairs.filter (fun p => isUnit (degreyH p.1) (degreyH p.2) 4) = degreyHEdges := by
  native_decide

def properH (a k : Nat) : Bool :=
  pairs.all (fun p => (! isUnit (degreyH p.1) (degreyH p.2) 4) ||
                       colorOf a k p.1 != colorOf a k p.2)
def chiH : Nat :=
  ((List.range 8).find? (fun k => (List.range (k ^ 7)).any (fun a => properH a k))).getD 7
theorem degreyH_chromatic_eq_3 : chiH = 3 := by native_decide

-- ===========================================================================
-- §6. Per-axis unit-edge witnesses (each unit distance LIVES in a new axis).
-- ===========================================================================

/-- √5 axis: rotation 2·arcsin(1/4), R·(1,0)=(7/8,√15/8); ×8 ⇒ (7,√15), d²=64.
    Certifying this unit edge requires the √15 = √3·√5 component (index 3). -/
theorem witness_sqrt5 : isUnit ([], []) ([(0,7)], [(3,1)]) 64 := by native_decide

/-- √7 axis: rotation arcsin(1/8), R·(1,0)=(3√7/8,1/8); ×8 ⇒ (3√7,1), d²=64.
    Requires the √7 component (index 4) — the axis a degree-8 field would miss. -/
theorem witness_sqrt7 : isUnit ([], []) ([(4,3)], [(0,1)]) 64 := by native_decide

/-- √11 axis: R·(1,0)=(√11/√12,1/√12); ×√12 ⇒ (√11,1), d²=12.
    Requires the √11 component (index 8). -/
theorem witness_sqrt11 : isUnit ([], []) ([(8,1)], [(0,1)]) 12 := by native_decide

-- ===========================================================================
-- §7. Summary — certifier ready for the 510-vertex graph.
-- ===========================================================================

/-- **Phase B fixture ladder, certified over the full field K = ℚ(√3,√5,√7,√11).**
    Multiplication table correct; Moser spindle (χ=4) and de Grey H (χ=3) recertified
    with converse audit; and a unit edge requiring each of √5, √7, √11 is certified
    exactly. The exact certifier is validated axis-by-axis and ready to be pointed at
    the 510-vertex record graph's coordinates. -/
theorem phaseB_field_ladder_certified :
    (isScalar (kmul (root 4) (root 4)) 7) ∧
    (pairs.filter (fun p => isUnit (spindle p.1) (spindle p.2) 144) = spindleEdges) ∧
    (chiSpindle = 4) ∧
    (pairs.filter (fun p => isUnit (degreyH p.1) (degreyH p.2) 4) = degreyHEdges) ∧
    (chiH = 3) ∧
    (isUnit ([], []) ([(0,7)], [(3,1)]) 64) ∧   -- √5
    (isUnit ([], []) ([(4,3)], [(0,1)]) 64) ∧   -- √7
    (isUnit ([], []) ([(8,1)], [(0,1)]) 12) := by  -- √11
  refine ⟨?_, spindle_unit_edges, spindle_chromatic_eq_4, degreyH_unit_edges,
          degreyH_chromatic_eq_3, witness_sqrt5, witness_sqrt7, witness_sqrt11⟩
  native_decide

end Sounio.UnitDistanceField
