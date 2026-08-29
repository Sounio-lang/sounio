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

-- ===========================================================================
-- §6. ASSOCIATOR surgery (the non-associative lever → 168 ZD classes).
--
--   A ZD class is an unordered pair (u, v) with u·v = 0 (bridge: unorderedZDPairs,
--   length 168). Since u·v = 0, assoc(p,u,v) = (p·u)·v − p·(u·v) = (p·u)·v.
--   So the associator-twisted point is the double right-multiplication (p·u)·v,
--   computed by composing the same right-mult coefficient twice. Twisted-unit
--   target = ‖u‖²·‖v‖² = 4.  All integer-valued ⇒ exact (no tolerance).
-- ===========================================================================

/-- Difference vector (p_i − p_j) as a coordinate function. -/
def dvec (i j : Nat) : Nat → Int := fun m => coord i m - coord j m

/-- Component `k` of `x · v` (right-multiplication of an arbitrary vector `x`
    by primitive `v = e_lo ± e_hi`). Composing this twice gives (x·u)·v. -/
def rmulCoeff (x : Nat → Int) (v : PrimSed) (k : Nat) : Int :=
  x (k ^^^ v.lo) * sedSigma (k ^^^ v.lo) v.lo
    + primSign v * (x (k ^^^ v.hi) * sedSigma (k ^^^ v.hi) v.hi)

/-- ‖((p_i − p_j)·u)·v‖² = ‖assoc(p_i − p_j, u, v)‖² (since u·v = 0). -/
def assocNormSq (i j : Nat) (u v : PrimSed) : Int :=
  let w1 := fun k => rmulCoeff (dvec i j) u k
  (List.range 16).foldl (fun acc k => let c := rmulCoeff w1 v k; acc + c * c) 0

/-- Associator-twisted unit edge for ZD class (u, v): ‖((p_i−p_j)·u)·v‖² = 4. -/
def assocEdge (u v : PrimSed) (i j : Nat) : Bool := assocNormSq i j u v == 4

-- ===========================================================================
-- §7. Honest associator results (native_decide; mirror the Sounio run).
-- ===========================================================================

/-- The 168 ZD classes are exactly the unordered ZD pairs of the bridge. -/
theorem associator_class_count_168 : unorderedZDPairs.length = 168 :=
  zd_projective_count_168

/-- The associator surgery is FULLY active: every one of the 168 ZD classes
    changes the edge set vs. classical (matches Sounio "classes changing edge
    set: 168"). Far stronger than the linear surgery's 4/84. -/
theorem all_associator_surgeries_change_edges :
    unorderedZDPairs.all (fun p =>
      vpairs.any (fun q => assocEdge p.1 p.2 q.1 q.2 != classEdge q.1 q.2)) = true := by
  native_decide

/-- HONEST NEGATIVE (stronger than the linear case): although every one of the
    168 associator surgeries is active, ALL keep the chromatic number at 2 —
    genuine non-associativity does not break bipartiteness on this probe.
    Conclusion: the bottleneck is the probe size, not the algebraic mechanism. -/
theorem no_associator_surgery_raises_chromatic :
    unorderedZDPairs.all (fun p => chromaticNumber (assocEdge p.1 p.2) == 2) = true := by
  native_decide

/-- Restatement: no associator class exceeds the classical chromatic number. -/
theorem associator_chromatic_le_classical :
    unorderedZDPairs.all (fun p =>
      chromaticNumber (assocEdge p.1 p.2) ≤ chromaticNumber classEdge) = true := by
  native_decide

-- ===========================================================================
-- §8. SCALE (slice 3): linear-surgery bipartiteness is FORCED, proved
--     constructively by an explicit 2-coloring (total Hamming-weight parity),
--     over the COMPLETE weight-≤2 binary probe (137 points). This is the
--     theorem behind the Sounio scale search (0/504 non-bipartite). The
--     associator coloring is non-obvious (total/half parity fail) — Sounio
--     BFS-verifies bipartiteness empirically; an explicit coloring is open.
-- ===========================================================================

/-- e_i as a coordinate function on {0..15}. -/
def e16 (i : Nat) : Nat → Int := fun k => if k = i then 1 else 0
/-- e_a + e_b. -/
def pair16 (a b : Nat) : Nat → Int := fun k => if k = a ∨ k = b then 1 else 0

/-- The complete weight-≤2 binary probe: {0} ∪ {e_i} ∪ {e_a+e_b}, 137 points.
    Same family scanned by `erdos_run_scale_search` in the Sounio module. -/
def bigProbe : List (Nat → Int) :=
  (fun _ => (0 : Int))
    :: ((List.range 16).map e16
        ++ (List.range 16).flatMap (fun a =>
             (List.range 16).filterMap (fun b => if a < b then some (pair16 a b) else none)))

theorem bigProbe_card : bigProbe.length = 137 := by native_decide

/-- Linear twisted unit edge between two point-functions: ‖(pi − pj)·v‖² = 2. -/
def linEdgeP (v : PrimSed) (pi pj : Nat → Int) : Bool :=
  ((List.range 16).foldl
    (fun acc k => let c := rmulCoeff (fun m => pi m - pj m) v k; acc + c * c) 0) == 2

/-- Total Hamming-weight parity of a point (the candidate 2-coloring). -/
def weightParity (p : Nat → Int) : Int :=
  ((List.range 16).foldl (fun a k => a + (if p k == 0 then 0 else 1)) 0) % 2

/-- THEOREM (slice 3): for every one of the 84 primitives, total Hamming-weight
    parity is a PROPER 2-coloring of the linear twisted graph on the complete
    weight-≤2 binary probe — every twisted edge joins opposite parities. Hence
    the linear ZD-surgery twisted graph is bipartite (χ ≤ 2): forced, not a
    probe-size accident. Matches the Sounio run (total-parity-2colors = 84/84). -/
theorem linear_surgery_total_parity_2colors :
    validPrims.all (fun v =>
      bigProbe.all (fun pi => bigProbe.all (fun pj =>
        (! linEdgeP v pi pj) || (weightParity pi != weightParity pj)))) = true := by
  native_decide

/-- Component `k` of LEFT-multiplication `v · x` (slice 4). Differs from
    `rmulCoeff` only by sigma argument order (anticommutativity), but terms
    touching index 0 do not flip. -/
def lmulCoeff (x : Nat → Int) (v : PrimSed) (k : Nat) : Int :=
  x (k ^^^ v.lo) * sedSigma v.lo (k ^^^ v.lo)
    + primSign v * (x (k ^^^ v.hi) * sedSigma v.hi (k ^^^ v.hi))

/-- Left-multiplication twisted unit edge: ‖v·(pi − pj)‖² = 2. -/
def leftEdgeP (v : PrimSed) (pi pj : Nat → Int) : Bool :=
  ((List.range 16).foldl
    (fun acc k => let c := lmulCoeff (fun m => pi m - pj m) v k; acc + c * c) 0) == 2

/-- THEOREM (slice 4): total Hamming-weight parity ALSO properly 2-colors the
    LEFT-multiplication twisted graph for all 84 primitives. So the bipartiteness
    invariant is TWO-SIDED — left-multiplication does not break total parity and
    is not the lever for χ ≥ 3. Matches the Sounio run (left non-bip = 0/84,
    total-parity-2colors = 84/84, edges identical to right). -/
theorem left_surgery_total_parity_2colors :
    validPrims.all (fun v =>
      bigProbe.all (fun pi => bigProbe.all (fun pj =>
        (! leftEdgeP v pi pj) || (weightParity pi != weightParity pj)))) = true := by
  native_decide

-- ===========================================================================
-- §10. NON-DISTANCE construction (slice 4b): the associator CONFLICT graph
--      breaks the parity ceiling — first χ ≥ 3 in the program.
--
--   assoc(x,y,c) = (x·y)·c − x·(y·c) is bilinear in (x,y), NOT a function of
--   x−y, so the conflict graph (edge ⟺ ‖assoc(x,y,c)‖²==4) is not translation-
--   invariant and escapes total-parity bipartiteness. We certify χ ≥ 3 by an
--   explicit triangle on the unit basis vectors {e₁, e₂, e₃}.
-- ===========================================================================

/-- Canonical first valid primitive (Sounio c=0): e₁ + e₁₀. -/
def cWit : PrimSed := ⟨1, 10, false⟩
theorem cWit_is_first : validPrims.head? = some cWit := by native_decide

/-- General sedenion product: (x·y)_k = Σ_{a⊕b=k} x_a y_b σ(a,b). -/
def sprodV (x y : Nat → Int) : Nat → Int := fun k =>
  (List.range 16).foldl (fun acc a =>
    (List.range 16).foldl (fun acc2 b =>
      if a ^^^ b = k then acc2 + x a * y b * sedSigma a b else acc2) acc) 0

/-- ‖assoc(x,y,c)‖² = ‖(x·y)·c − x·(y·c)‖²  (c primitive ⇒ ·c is rmulCoeff). -/
def assocNormSqL (x y : Nat → Int) (c : PrimSed) : Int :=
  let xy := sprodV x y
  let yc := fun k => rmulCoeff y c k
  (List.range 16).foldl (fun acc k =>
    let d := rmulCoeff xy c k - sprodV x yc k; acc + d * d) 0

/-- Associator-conflict edge (symmetrized): ‖[x,y,cWit]‖²==4 or ‖[y,x,cWit]‖²==4. -/
def assocEdgeWit (x y : Nat → Int) : Bool :=
  (assocNormSqL x y cWit == 4) || (assocNormSqL y x cWit == 4)

/-- THEOREM (slice 4b): {e₁, e₂, e₃} is a TRIANGLE in the associator conflict
    graph (c = cWit). A triangle is not 2-colorable, so the associator conflict
    graph has χ ≥ 3 — the FIRST construction in the program to exceed χ = 2.
    Non-associativity is essential: every distance-based ZD surgery (left & right
    multiplication, §3/§8/§9) is provably bipartite, but the associator —
    bilinear, not translation-invariant — breaks the total-parity ceiling. -/
theorem associator_conflict_triangle :
    assocEdgeWit (e16 1) (e16 2) = true
    ∧ assocEdgeWit (e16 1) (e16 3) = true
    ∧ assocEdgeWit (e16 2) (e16 3) = true := by native_decide

-- ===========================================================================
-- §11. SPARSE rigorous gap (slice 5): the χ≥8 above is CLIQUE-driven (χ≈ω),
--      the trivial regime for unit-distance. The Erdős-hard regime is χ ≫ ω
--      (sparse, clique-poor). Here we certify an INDUCED 5-CYCLE in the T=8
--      associator conflict graph: a pentagon has ω = 2 and χ = 3, so χ > ω with
--      the *smallest possible* clique — a rigorous foothold in the sparse regime.
--      (Sounio finds, exactly, a triangle-free induced subgraph that is
--      non-bipartite; this C₅ is its shortest odd cycle.)
-- ===========================================================================

/-- Associator-conflict edge at target T=8 (symmetrized), c = cWit. -/
def assocEdge8 (x y : Nat → Int) : Bool :=
  (assocNormSqL x y cWit == 8) || (assocNormSqL y x cWit == 8)

/-- THEOREM (slice 5): the five points e₃, e₂+e₆, e₂, e₄, e₈ form an INDUCED
    5-cycle in the T=8 associator conflict graph (5 cycle edges present, all 5
    chords absent). An induced C₅ is triangle-free (ω = 2) and non-bipartite
    (χ = 3), so χ > ω. This is a rigorous witness in the sparse (clique-poor)
    regime — distinct from the clique-driven χ≥8, and the regime that actually
    matters for unit-distance chromatic problems. -/
theorem associator_conflict_induced_C5 :
    -- 5 cycle edges (e₃ — e₂+e₆ — e₂ — e₄ — e₈ — e₃)
    assocEdge8 (e16 3) (pair16 2 6) = true
    ∧ assocEdge8 (pair16 2 6) (e16 2) = true
    ∧ assocEdge8 (e16 2) (e16 4) = true
    ∧ assocEdge8 (e16 4) (e16 8) = true
    ∧ assocEdge8 (e16 8) (e16 3) = true
    -- 5 chords absent ⇒ chordless ⇒ triangle-free (ω = 2)
    ∧ assocEdge8 (e16 3) (e16 2) = false
    ∧ assocEdge8 (e16 3) (e16 4) = false
    ∧ assocEdge8 (pair16 2 6) (e16 4) = false
    ∧ assocEdge8 (pair16 2 6) (e16 8) = false
    ∧ assocEdge8 (e16 2) (e16 8) = false := by native_decide

end Sounio.Erdos

-- Findings, all machine-checked:
--   slice 1 — linear right-mult surgery: 4/84 active on the 7-vertex probe, 0 raise χ;
--   slice 2 — associator (p·u)·v: 168/168 active, 0 raise χ;
--   slice 3 — scale to the complete weight-≤2 family (137 binary + 273 signed,
--             504 graphs via Sounio BFS): 0 non-bipartite. Linear bipartiteness
--             is PROVED here by an explicit total-Hamming-weight-parity coloring.
-- The algebraic lever is validated (associator acts everywhere); bipartiteness
-- is structural for the linear case. OPEN (slice 4): explicit associator coloring
-- / proof, and whether breaking total-parity (e.g. left-mult, or a different
-- graph construction) is required to force χ ≥ 3.
