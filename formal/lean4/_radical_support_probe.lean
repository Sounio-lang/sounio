import SounioDeGreyUnitDistance

/-! Scratch probe: extract the exact radical support of the G₅₂₉ coordinate tables.
    QF = (List Int (16 coeffs), Int denom); basis index i over {3,5,7,11} bitmask,
    radicand = bcoeff i (i=0 → rational 1). -/

open DeGrey529

/-- Distinct vertices that appear in at least one edge. -/
def vertsTouched : List Nat :=
  edges.toList.foldl (fun acc e =>
    let acc := if acc.contains e.1 then acc else e.1 :: acc
    if acc.contains e.2 then acc else e.2 :: acc) []

def coeffAt (q : QF) (i : Nat) : Int := gi q.1 i

/-- A basis index is "used" if some touched vertex has a nonzero coeff there in X or Y. -/
def idxUsedX (i : Nat) : Bool :=
  vertsTouched.any (fun v => coeffAt (X.getD v ([], 1)) i != 0)
def idxUsedY (i : Nat) : Bool :=
  vertsTouched.any (fun v => coeffAt (Y.getD v ([], 1)) i != 0)
def idxUsed (i : Nat) : Bool := idxUsedX i || idxUsedY i

def usedIndices : List Nat := (List.range 16).filter (fun i => idxUsed i)

def primesOfIdx (i : Nat) : List Nat :=
  (if i % 2 = 1 then [3] else []) ++ (if (i / 2) % 2 = 1 then [5] else [])
    ++ (if (i / 4) % 2 = 1 then [7] else []) ++ (if (i / 8) % 2 = 1 then [11] else [])

def primesUsed : List Nat :=
  (usedIndices.foldl (fun acc i => acc ++ primesOfIdx i) []).foldl
    (fun acc p => if acc.contains p then acc else acc ++ [p]) []

/-- distinct touched vertices with nonzero coeff at index i (in X or Y). -/
def cntAt (i : Nat) : Nat :=
  (vertsTouched.filter (fun v =>
    coeffAt (X.getD v ([], 1)) i != 0 || coeffAt (Y.getD v ([], 1)) i != 0)).length

#eval ("total edges", edges.size, "distinct verts touched", vertsTouched.length, "X size", X.size, "Y size", Y.size)
#eval ("used basis indices", usedIndices)
#eval ("index -> radicand (bcoeff) -> #verts", (List.range 16).map (fun i => (i, bcoeff i, cntAt i)))
#eval ("PRIMES USED (set S)", primesUsed)
#eval ("X-only used idx", (List.range 16).filter (fun i => idxUsedX i))
#eval ("Y-only used idx", (List.range 16).filter (fun i => idxUsedY i))

/-- An edge "touches" basis index i if either endpoint coordinate (X or Y) has a nonzero coeff there. -/
def edgeUsesIdx (e : Nat × Nat) (i : Nat) : Bool :=
  coeffAt (X.getD e.1 ([], 1)) i != 0 || coeffAt (Y.getD e.1 ([], 1)) i != 0 ||
  coeffAt (X.getD e.2 ([], 1)) i != 0 || coeffAt (Y.getD e.2 ([], 1)) i != 0
def edgeCntAt (i : Nat) : Nat := (edges.toList.filter (fun e => edgeUsesIdx e i)).length

/-- An edge touches prime p if some endpoint coordinate uses a radical whose radicand contains p. -/
def edgeUsesPrime (e : Nat × Nat) (p : Nat) : Bool :=
  (List.range 16).any (fun i => edgeUsesIdx e i && (primesOfIdx i).contains p)
def edgeCntPrime (p : Nat) : Nat := (edges.toList.filter (fun e => edgeUsesPrime e p)).length

/-- Edges all of whose endpoint coordinates are purely rational (only index 0). -/
def rationalEdges : Nat :=
  (edges.toList.filter (fun e => !((List.range 16).any (fun i => i != 0 && edgeUsesIdx e i)))).length

#eval ("index -> radicand -> #edges (of 2670)", (List.range 16).map (fun i => (i, bcoeff i, edgeCntAt i)))
#eval ("prime -> #edges touching it", [3, 5, 7, 11].map (fun p => (p, edgeCntPrime p)))
#eval ("edges with both endpoints purely rational", rationalEdges, "of", edges.size)
