/-
  SounioSedenionE8Fibers — the independent-spec (Lean `native_decide`) leg for the sedenion
  zero-divisor e8-boundary and its 7-fiber decomposition.

  Companion to the executed-in-Sounio results:
    * tests/run-pass/sedenion_e8_boundary.sio  + docs/research/sedenion_e8_boundary.md
    * tests/run-pass/sedenion_zd_fibers.sio    + docs/research/sedenion_zd_fibers.md
  and to the Python oracles (scripts/research/sedenion_e8_boundary_oracle.py, ..._zd_fibers_oracle.py).

  Those two legs (souc, Python) both TRANSCRIBE the Cayley-Dickson sign law, so their agreement
  guards against a souc miscompile but not against a spec error. This file is the third leg the
  operator's firewall asks for: Lean's compiled kernel evaluator (`native_decide`) — an independent
  checker — verifies the same combinatorial claims. Mathlib-free; no `sorry`; aligned with the
  `cdSigma` recursion of SounioCayleyDickson.lean / SounioZeroDivisorBridge.lean.
-/

namespace SounioSedenionE8Fibers

/-- Cayley-Dickson sign at tower level `bits`: `e_a * e_b = cdSigma a b bits * e_(a xor b)`, in `{-1,+1}`. -/
def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        let aHi := a ≥ half
        let bHi := b ≥ half
        let aLo := a % half
        let bLo := b % half
        if !aHi && !bHi then cdSigma aLo bLo (n+1)
        else if !aHi && bHi then cdSigma bLo aLo (n+1)
        else if aHi && !bHi then (if bLo == 0 then cdSigma aLo 0 (n+1) else - cdSigma aLo bLo (n+1))
        else (if bLo == 0 then - cdSigma 0 aLo (n+1) else cdSigma bLo aLo (n+1))

def sedSigma (a b : Nat) : Int := cdSigma a b 4

/-- `k`-th integer coefficient of the product of the two-support primitives
    `u = e_ulo (±) e_uhi` and `v = e_vlo (±) e_vhi` (`un`/`vn = true` means the minus sign). -/
def primProd (ulo uhi : Nat) (un : Bool) (vlo vhi : Nat) (vn : Bool) (k : Nat) : Int :=
  let su : Int := if un then -1 else 1
  let sv : Int := if vn then -1 else 1
  (if (ulo ^^^ vlo) == k then sedSigma ulo vlo else 0)
  + (if (ulo ^^^ vhi) == k then sv * sedSigma ulo vhi else 0)
  + (if (uhi ^^^ vlo) == k then su * sedSigma uhi vlo else 0)
  + (if (uhi ^^^ vhi) == k then su * sv * sedSigma uhi vhi else 0)

/-- The exact product is the zero vector across all 16 components (decidable). -/
def isZeroPair (ulo uhi : Nat) (un : Bool) (vlo vhi : Nat) (vn : Bool) : Bool :=
  (List.range 16).all (fun k => primProd ulo uhi un vlo vhi vn k == 0)

/-- The 112 mixed-half signed two-support primitives: `lo ∈ 1..7`, `hi ∈ 8..15`, sign in `{+,-}`. -/
def cands : List (Nat × Nat × Bool) :=
  (List.range 7).flatMap (fun i => (List.range 8).flatMap (fun j => [(i+1, j+8, false), (i+1, j+8, true)]))

/-- A primitive participates iff some candidate annihilates it. -/
def participates (c : Nat × Nat × Bool) : Bool :=
  cands.any (fun d => isZeroPair c.1 c.2.1 c.2.2 d.1 d.2.1 d.2.2)

/-- Number of participating vertices in the fiber labeled `L = lo xor hi`. -/
def fiberSize (L : Nat) : Nat :=
  (cands.filter (fun c => participates c && (c.1 ^^^ c.2.1) == L)).length

-- ── Theorems (Lean-kernel verified; independent of souc and of the Python oracle) ──────────────

/-- There are 112 mixed-half signed two-support primitives. -/
theorem cands_count : cands.length = 112 := by native_decide

/-- Exactly 84 participate in a sedenion zero-divisor pair. -/
theorem participate_count : (cands.filter participates).length = 84 := by native_decide

/-- Exactly 28 participate in none. -/
theorem excluded_count : (cands.filter (fun c => ! participates c)).length = 28 := by native_decide

/-- The e8-BOUNDARY: a mixed-half primitive is excluded ⟺ it touches `e8` (`hi = 8`)
    or lies on the xor-grade-8 diagonal (`lo xor hi = 8`) — the octonion→sedenion doubling seam. -/
theorem e8_invariant :
    cands.all (fun c => (! participates c) == decide (c.2.1 = 8 ∨ (c.1 ^^^ c.2.1) = 8)) = true := by
  native_decide

/-- The 7-FIBER structure: exactly 7 fibers, one per `L = lo xor hi ∈ {9..15}`, each of 12 vertices. -/
theorem fibers_7x12 : (List.range 7).all (fun t => fiberSize (t+9) == 12) = true := by native_decide

/-- Annihilation never crosses fibers: `a*b = 0` (participating, distinct) ⟹ `L(a) = L(b)`. -/
theorem intra_fiber :
    cands.all (fun a => cands.all (fun b =>
      (! (participates a && participates b && isZeroPair a.1 a.2.1 a.2.2 b.1 b.2.1 b.2.2))
      || ((a.1 ^^^ a.2.1) == (b.1 ^^^ b.2.1)))) = true := by
  native_decide

end SounioSedenionE8Fibers
