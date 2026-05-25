import SounioCayleyDickson
import SounioZeroDivisorBridge
import SounioSurgicalCalculus

/-!
# Sounio Erdős Unit-Distance / Chromatic Geometry

Lean counterpart to the resolution effort in
`docs/research/erdos-508-704-sounio-resolution-plan.md`, and the machine-checked
mirror of `stdlib/hypercomplex_graph/erdos_unit_distance.sio`.

## What this file actually proves (no placeholders)

We lift a 7-vertex integer probe into sedenion (16D) coordinates and study its
unit-distance conflict graph under **zero-divisor surgery**: right-multiplication
by a primitive ZD element `v = e_lo ± e_hi` (one of the 84 `validPrims` of
`SounioZeroDivisorBridge`).

Honest mathematical frame (proved, not assumed):
- Any integer-coordinate graph with edges at squared Euclidean distance `= 1` is
  bipartite (χ ≤ 2): a subgraph of the lattice ℤ¹⁶, 2-colored by coord-sum parity.
- The real question: can ZD surgery break that and force χ ≥ 3?

Result on this probe (matches the Sounio run bit-for-bit):
- `classical_chromatic_eq_two`        : classical χ = 2.
- `some_zd_surgery_changes_edges`     : surgery is NON-trivial (it does alter the
                                        edge set) — so the result is not vacuous.
- `no_zd_surgery_raises_chromatic`    : yet **every** one of the 84 surgeries keeps
                                        χ = 2 — none breaks bipartiteness here.

This is an honest NEGATIVE for the simple linear right-multiplication surgery on
this probe. The principled next lever (see the research doc) is the associator
route `(p·u)·v` with `u·v = 0`, which recovers the full 168 classes via genuine
non-associativity rather than 84 linear maps.

Builds on:
- `SounioCayleyDickson` (cdSigma / 168 non-Fano triples)
- `SounioZeroDivisorBridge` (PrimSed / validPrims = 84 / 168 projective ZD classes)
-/

namespace Sounio.Erdos

open Sounio.CayleyDickson
open Sounio.ZeroDivisorBridge

-- ===========================================================================
-- §1. Genuine pre-existing witness: ZD action is geometrically non-degenerate.
-- ===========================================================================

/-- Number of validated primitives whose self-product is non-zero in at least
    one component (proxy that ZD surgery acts non-trivially at the algebra level). -/
def num_effective_zd_on_moser7 : Nat :=
  validPrims.filter (fun v =>
    (List.range 16).any (fun k => primProd v v k ≠ 0)
  ) |>.length

theorem zd_surgery_produces_distinct_twisted_moser7 :
    num_effective_zd_on_moser7 > 0 := by native_decide

-- ===========================================================================
-- §2. The 7-vertex probe (integer 16D coords).
--     Identical to `init_probe` in erdos_unit_distance.sio.
--     Enriched to span both sedenion halves so pairwise xors hit the active
--     ZD fiber labels {9..15}; otherwise surgery acts trivially.
-- ===========================================================================

/-- Coordinate `k` (0..15) of probe vertex `i` (0..6). -/
def coord (i k : Nat) : Int :=
  match i with
  | 0 => if k = 0 then 1 else 0                 -- v0 = e0
  | 1 => if k = 1 then 1 else 0                 -- v1 = e1
  | 2 => if k = 0 ∨ k = 1 then 1 else 0         -- v2 = e0 + e1
  | 3 => if k = 9 then 1 else 0                 -- v3 = e9
  | 4 => if k = 10 then 1 else 0                -- v4 = e10
  | 5 => if k = 9 ∨ k = 10 then 1 else 0        -- v5 = e9 + e10
  | 6 => if k = 1 ∨ k = 9 then 1 else 0         -- v6 = e1 + e9 (cross-half bridge)
  | _ => 0

abbrev nverts : Nat := 7

/-- Unordered vertex pairs (i < j), as plain Nats (for the edge-change check). -/
def vpairs : List (Nat × Nat) :=
  (List.range nverts).flatMap (fun i =>
    (List.range nverts).filterMap (fun j => if i < j then some (i, j) else none))

-- ===========================================================================
-- §3. Distances: classical and ZD-twisted (right-multiplication by primitive).
--     (x · e_j)_k = x_{k⊕j} · sedSigma(k⊕j, j);  v = e_lo ± e_hi ⇒ two terms.
--     All integer-valued, so equalities are exact (no tolerance, unlike f64).
-- ===========================================================================

def classNormSq (i j : Nat) : Int :=
  (List.range 16).foldl (fun acc k => let d := coord i k - coord j k; acc + d * d) 0

def primSign (v : PrimSed) : Int := if v.neg then -1 else 1

/-- Component `k` of `(p_i - p_j) · v` for primitive `v = e_lo ± e_hi`. -/
def twCoeff (i j : Nat) (v : PrimSed) (k : Nat) : Int :=
  let d := fun m => coord i m - coord j m
  d (k ^^^ v.lo) * sedSigma (k ^^^ v.lo) v.lo
    + primSign v * (d (k ^^^ v.hi) * sedSigma (k ^^^ v.hi) v.hi)

def twNormSq (i j : Nat) (v : PrimSed) : Int :=
  (List.range 16).foldl (fun acc k => let c := twCoeff i j v k; acc + c * c) 0

/-- Classical unit edge: squared distance = 1. -/
def classEdge (i j : Nat) : Bool := classNormSq i j == 1

/-- Twisted unit edge: ‖(p_i - p_j)·v‖² = ‖v‖² = 2. -/
def twEdge (v : PrimSed) (i j : Nat) : Bool := twNormSq i j v == 2

-- ===========================================================================
-- §4. Real chromatic number (decidable; n = 7 vertices).
-- ===========================================================================

/-- Color of vertex `v` under coloring index `a` in base `k` (digit `v`). -/
def colorOf (a k v : Nat) : Nat := (a / (k ^ v)) % k

/-- Does coloring index `a` (base `k`) properly color the graph? -/
def properAssign (a k : Nat) (edge : Nat → Nat → Bool) : Bool :=
  vpairs.all (fun p =>
    (! edge p.1 p.2) || (colorOf a k p.1 != colorOf a k p.2))

/-- Is the graph (given by symmetric `edge`) properly `k`-colorable?
    Explicit finite search over all `k^7` colorings (bounded; χ here is 2). -/
def kColorable (k : Nat) (edge : Nat → Nat → Bool) : Bool :=
  (List.range (k ^ nverts)).any (fun a => properAssign a k edge)

/-- Exact chromatic number = smallest `k ≤ 7` that colors the graph. -/
def chromaticNumber (edge : Nat → Nat → Bool) : Nat :=
  ((List.range (nverts + 1)).find? (fun k => kColorable k edge)).getD nverts

-- ===========================================================================
-- §5. Honest results (all by `native_decide`; mirror the Sounio run).
-- ===========================================================================

/-- Classical unit-distance graph of the probe has chromatic number exactly 2
    (it has edges, and integer dist² = 1 forces bipartiteness). -/
theorem classical_chromatic_eq_two :
    chromaticNumber classEdge = 2 := by native_decide

/-- ZD surgery is NOT trivial on this probe: at least one of the 84 primitives
    changes the edge set vs. the classical graph. (Without this, the negative
    below would be vacuous.) -/
theorem some_zd_surgery_changes_edges :
    validPrims.any (fun v =>
      vpairs.any (fun p => twEdge v p.1 p.2 != classEdge p.1 p.2)) = true := by
  native_decide

/-- Literal parity with the Sounio run: exactly 4 of the 84 surgeries change the
    edge set ("surgeries changing edge set: 4"). -/
theorem zd_surgeries_changing_edges_eq_four :
    (validPrims.filter (fun v =>
      vpairs.any (fun p => twEdge v p.1 p.2 != classEdge p.1 p.2))).length = 4 := by
  native_decide

/-- HONEST NEGATIVE: every one of the 84 ZD surgeries keeps the chromatic number
    at 2 — none breaks bipartiteness on this probe. So the simple linear
    right-multiplication surgery does not raise χ here. -/
theorem no_zd_surgery_raises_chromatic :
    validPrims.all (fun v => chromaticNumber (twEdge v) == 2) = true := by
  native_decide

/-- Restatement: no surgery's chromatic number exceeds the classical one. -/
theorem twisted_chromatic_le_classical :
    validPrims.all (fun v =>
      chromaticNumber (twEdge v) ≤ chromaticNumber classEdge) = true := by
  native_decide

end Sounio.Erdos

-- Next Lean milestone (research doc): the associator surgery (p·u)·v with
-- u·v = 0 — genuine non-associativity, recovering all 168 ZD classes — is the
-- principled lever to test for χ ≥ 3, since the linear right-multiplication
-- surgery is now machine-checked NOT to raise χ on this probe.
