import SounioCayleyDickson
import SounioZeroDivisorBridge
import SounioSurgicalCalculus

/-!
# Sounio — The Associator-Shadow Experiment (the non-associative lever)

Lean counterpart to `docs/research/fano-arcs-blocking-sounio-note.md` §5 and
`docs/research/associator-shadow-experiment.md`. This is the *deferred next lever* from
`SounioErdosUnitDistance.lean` and `SounioFanoArcsBlocking.lean`, now executed.

## The experiment

`SounioFanoArcsBlocking` proved that the **linear** zero-divisor surgery (right
multiplication, `UNLEARN`) is trapped: its Fano point-shadows are *always* one of the
7 hyperovals (the 7 line-complements, all 4-arcs). The principled question left open
was whether **genuine non-associativity** does more.

For a zero-divisor pair `u · v = 0` the associator collapses to a single product:
  `[p, u, v] = (p·u)·v − p·(u·v) = (p·u)·v`   (since `u·v = 0`).
So the *associator shadow* of a triple `(p, u, v)` with `u · v = 0` is the Fano
point-shadow (lo-class image, in `{1..7}`) of the sedenion `(p·u)·v`.

## What this file proves (`native_decide`, no `sorry`, no Mathlib)

Ranging `p` over the 84 `validPrims` and `(u,v)` over the 336 `orderedZDPairs`:

* `assoc_reaches_every_fano_line` + `witness_lines_are_the_seven` — for **each** of the
  7 Fano lines there is a ZD-pair triple whose associator shadow is *exactly that line*.
* `assoc_escapes_hyperovals` — those 7 shadows are **not** hyperovals and are **not
  arcs** (each contains a full Fano line). The linear surgery can *never* produce a
  line; the associator does.
* `assoc_shadow_max_size_le_3` — **the honest ceiling.** Every associator shadow over
  all `84 × 336` triples has at most 3 points; the lever never produces a 4-set.

## Honest reading (Level 2, not Level 3)

The non-associative lever has real geometric teeth: it escapes the 7-hyperoval trap and
realizes the entire sub-maximal skeleton of `PG(2,2)` — all points, pairs, and triples
(the full characterization, family = `{S ⊆ {1..7} : |S| ≤ 3}`, is Python-validated in
`scripts/research/associator_shadow_validate.py`), **including the 7 Fano lines the
linear surgery structurally avoids.** But it escapes *downward*: it never produces a new
arc of size ≥ 4. So it changes the incidence *type* (arc → line) without producing any
new maximal/forbidden configuration in this fully-understood plane. This is a clean
Level-2 separation between the linear and non-associative regimes — **not** a Level-3
bound. The two regimes are complementary co-flats: linear = the 7 maximal arcs (top),
associator = the whole `|S| ≤ 3` skeleton (bottom).

Builds on `SounioZeroDivisorBridge` (`sedSigma`, `PrimSed`, `validPrims = 84`,
`orderedZDPairs = 336`, `isZeroPair`) and the Fano structure of `SounioCayleyDickson`.
-/

namespace Sounio.AssociatorShadow

open Sounio.CayleyDickson
open Sounio.ZeroDivisorBridge

-- ===========================================================================
-- §1. Sparse sedenion multiplication (keeps native_decide fast).
-- ===========================================================================

/-- Sparse 16-D sedenion: a list of `(index, coefficient)` terms. -/
abbrev SVec := List (Nat × Int)

/-- Add coefficient `c` at index `k` (merging with an existing term). -/
def bump (acc : List (Nat × Int)) (k : Nat) (c : Int) : List (Nat × Int) :=
  if acc.any (fun p => p.1 == k)
  then acc.map (fun p => if p.1 == k then (p.1, p.2 + c) else p)
  else acc ++ [(k, c)]

/-- Merge duplicate indices and drop zero coefficients. -/
def combine (v : SVec) : SVec :=
  (v.foldl (fun acc p => bump acc p.1 p.2) []).filter (fun p => p.2 != 0)

/-- Sedenion product via the verified `sedSigma` (Cayley–Dickson level 4):
    `(eᵢ·eⱼ) = sedSigma(i,j) · e_{i⊕j}`. -/
def smul (a b : SVec) : SVec :=
  combine (a.flatMap (fun p => b.map (fun q => (p.1 ^^^ q.1, p.2 * q.2 * sedSigma p.1 q.1))))

/-- A primitive `e_lo ± e_hi` as a sparse vector. -/
def primVec (u : PrimSed) : SVec :=
  [(u.lo, (1 : Int)), (u.hi, if u.neg then (-1 : Int) else 1)]

/-- De-duplicate a `List Nat`. -/
def nubNat (l : List Nat) : List Nat :=
  l.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []

/-- Fano point-shadow: the lo-class (`index &&& 7`) of each nonzero support index,
    kept in `{1..7}` (`0` = real/`e8` slot, excluded). -/
def loShadow (v : SVec) : List Nat :=
  nubNat ((combine v).filterMap (fun p =>
    let l := p.1 &&& 7
    if l == 0 then none else some l))

/-- The associator shadow of `(p,u,v)`: for `u·v = 0`, `[p,u,v] = (p·u)·v`. -/
def assocShadow (p u v : PrimSed) : List Nat :=
  loShadow (smul (smul (primVec p) (primVec u)) (primVec v))

-- ===========================================================================
-- §2. Fano lines, hyperovals, arcs (Bool predicates on point sets).
-- ===========================================================================

def fanoPoints : List Nat := [1, 2, 3, 4, 5, 6, 7]
def fanoLines7 : List (List Nat) :=
  [[1,2,3], [1,4,5], [1,6,7], [2,4,6], [2,5,7], [3,4,7], [3,5,6]]

def seteq (a b : List Nat) : Bool :=
  a.all (fun x => b.contains x) && b.all (fun x => a.contains x)

/-- The 7 hyperovals = the 7 line-complements (what the LINEAR surgery is trapped in). -/
def hyperovals7 : List (List Nat) :=
  fanoLines7.map (fun L => fanoPoints.filter (fun p => !(L.contains p)))

def isHyperoval (s : List Nat) : Bool := hyperovals7.any (fun H => seteq s H)
def containsLine (s : List Nat) : Bool :=
  fanoLines7.any (fun L => L.all (fun p => s.contains p))
def isArc (s : List Nat) : Bool := !containsLine s

-- ===========================================================================
-- §3. Per-line witnesses (extracted by the Python census).
--     Each `(L, p, u, v)`: `u·v = 0` and `assocShadow p u v = L`.
-- ===========================================================================

/-- One ZD-pair triple per Fano line, whose associator shadow is exactly that line. -/
def lineWitnesses : List (List Nat × PrimSed × PrimSed × PrimSed) :=
  [ ([1,2,3], PrimSed.mk 1 10 false, PrimSed.mk 4 13 false, PrimSed.mk 6 15 false),
    ([1,4,5], PrimSed.mk 1 12 false, PrimSed.mk 2 11 false, PrimSed.mk 6 15 false),
    ([1,6,7], PrimSed.mk 1 14 false, PrimSed.mk 2 11 false, PrimSed.mk 4 13 true),
    ([2,4,6], PrimSed.mk 2 12 false, PrimSed.mk 1 11 false, PrimSed.mk 5 15 false),
    ([2,5,7], PrimSed.mk 2 13 false, PrimSed.mk 1 11 false, PrimSed.mk 4 14 false),
    ([3,4,7], PrimSed.mk 3 12 false, PrimSed.mk 1 10 false, PrimSed.mk 5 14 false),
    ([3,5,6], PrimSed.mk 3 13 false, PrimSed.mk 1 10 false, PrimSed.mk 4 15 true) ]

/-- The witness lines are exactly the 7 Fano lines. -/
theorem witness_lines_are_the_seven :
    lineWitnesses.map (fun w => w.1) = fanoLines7 := by native_decide

/-- **The lever reaches every Fano line.** For each Fano line `L` there is a genuine ZD
    pair `u·v = 0` and a probe `p` whose associator shadow `(p·u)·v` is exactly `L`. -/
theorem assoc_reaches_every_fano_line :
    lineWitnesses.all (fun w =>
      let p := w.2.1; let u := w.2.2.1; let v := w.2.2.2
      isZeroPair u v && seteq (assocShadow p u v) w.1) := by
  native_decide

/-- **The lever escapes the hyperoval trap.** Each witness shadow is neither a hyperoval
    nor an arc — it is a full Fano line. The linear `UNLEARN` surgery (always a hyperoval,
    `SounioFanoArcsBlocking.unlearn_kernel_is_complement_of_line`) can never do this. -/
theorem assoc_escapes_hyperovals :
    lineWitnesses.all (fun w =>
      let s := assocShadow w.2.1 w.2.2.1 w.2.2.2
      !isHyperoval s && !isArc s && containsLine s) := by
  native_decide

-- ===========================================================================
-- §4. The honest ceiling: the lever escapes downward, never to a 4-set.
-- ===========================================================================

/-- **Associator-shadow ceiling.** Over all `84 × 336` triples `(p, u·v=0)`, every
    associator shadow has at most 3 points. The non-associative lever never produces a
    4-set (hence never a new arc/hyperoval): it escapes into the `|S| ≤ 3` skeleton, not
    upward. This is the honest Level-2 limit of the lever on `PG(2,2)`. -/
theorem assoc_shadow_max_size_le_3 :
    validPrims.all (fun p => orderedZDPairs.all (fun pr =>
      (assocShadow p pr.1 pr.2).length ≤ 3)) := by
  native_decide

/-- No associator shadow is ever a hyperoval (immediate corollary of the size ceiling:
    hyperovals have 4 points). Direct contrast with the linear surgery. -/
theorem assoc_shadow_never_hyperoval :
    validPrims.all (fun p => orderedZDPairs.all (fun pr =>
      !isHyperoval (assocShadow p pr.1 pr.2))) := by
  native_decide

-- ===========================================================================
-- §5. Summary.
-- ===========================================================================

/-- **Associator-shadow separation summary.** The non-associative lever (i) reaches every
    Fano line via a genuine ZD pair, (ii) thereby escapes the 7-hyperoval trap with
    non-arc shadows the linear surgery cannot produce, yet (iii) never produces a shadow
    of size > 3 and (iv) never a hyperoval. Linear and associator regimes are
    complementary co-flats of `PG(2,2)`. -/
theorem associator_shadow_summary :
    (lineWitnesses.map (fun w => w.1) = fanoLines7) ∧
    (lineWitnesses.all (fun w =>
      isZeroPair w.2.2.1 w.2.2.2
      && seteq (assocShadow w.2.1 w.2.2.1 w.2.2.2) w.1
      && !isArc (assocShadow w.2.1 w.2.2.1 w.2.2.2))) ∧
    (validPrims.all (fun p => orderedZDPairs.all (fun pr =>
      (assocShadow p pr.1 pr.2).length ≤ 3))) := by
  refine ⟨witness_lines_are_the_seven, ?_, assoc_shadow_max_size_le_3⟩
  native_decide

end Sounio.AssociatorShadow
