import SounioPathionBridge
import SounioAssociatorShadow

/-!
# Sounio — Erdős [90]: lifting the sedenion ceiling at the pathion level

Refinement of `SounioErdos90UnitSpectrum.lean`. There the associator lever
`((p·u)·v, u·v=0)` applied to an interleaved-star probe produced a unit-pair count
that grows super-linearly with probe size but **saturates at 40**, because the
sedenion basis offers only 7 + 7 = 14 ZD-active imaginary directions. This file shows
that saturation is an artifact of the 16-dimensional algebra, not of the lever: read
the *same* construction one Cayley–Dickson level up, in the **pathions** (32-D, level
5, `Sounio.PathionBridge`), and the growth continues past 40.

## Method (all `native_decide`, no Mathlib, no `sorry`)

* `pSmul` — the 32-D pathion product on sparse vectors, built from the verified
  `pathSigma = cdSigma · 5` (`SounioPathionBridge`). The direct analogue of
  `Sounio.AssociatorShadow.smul`, which used `sedSigma` over 16 components.
* `pathZD` — the intra-fiber ordered pathion zero-divisor pairs, computed and
  **verified** here (`pZero u v` checks `u·v = 0` in all 32 components). The pathion
  ZD-pair set was not previously enumerated (`SounioPathionBridge` stops at the 420
  primitives / 15 fibers). `pathZD_count` certifies there are **3696** of them
  (15 fibers × 28 primitives × … ; 11× the sedenion 336).
* The interleaved-star probe and `pAssocEdge` (associator squared norm `= 4`, same
  normalization as the sedenion file) are the literal 32-D lift of §7 there.

## What this proves

* `pathZD_count`                     : 3696 verified pathion ZD pairs.
* `path_classical_count`             : the `m`-leaf pathion star has `m` classical unit
                                       pairs (linear/classical baseline, as before).
* `path_assoc_max_{6,14,18}`         : associator max unit-pair count = 12, 44, 72 at
                                       7 / 15 / 19 vertices.
* `pathion_breaks_sedenion_ceiling`  : at 15 vertices the pathion associator reaches
                                       **44 > 40** — strictly above the sedenion ceiling
                                       proved in `SounioErdos90UnitSpectrum`.
* `pathion_growth_summary`           : 12 < 44 < 72 and 44 > 40 — growth continues past
                                       the lifted ceiling.

The full observed curve (reproducible by raising the leaf bound; the larger sizes are
omitted as proven theorems only to keep this library's build cheap) is
`(vertices, classical, assoc-max) = (7,6,12) (11,10,30) (15,14,44) (19,18,72)
(23,22,104) (27,26,136) (31,30,168)` — the 32-D associator star reaches **168** unit
pairs at 31 vertices while the linear/classical route stays pinned at `n−1`. (The 168
here is coincidental: n=31 = 15+15+1 is the full ZD-active pathion basis, so 168 is that
star's unit-pair count — not connected to `non_fano_count_168` / the 168 sedenion ZD
classes.)

## Honest scope

Unchanged from the sedenion file: this is an **unconditional finite-model** statement
about star probes in 32-D pathion coordinates. It demonstrates that the count-growth
separation is not capped by the sedenion algebra — it is a property of the
non-associative lever that survives moving up the Cayley–Dickson tower. It is **not** a
bound on the classical planar `u(n)`; no planar embedding is claimed.
-/

namespace Sounio.Erdos90Pathion

open Sounio.PathionBridge
open Sounio.AssociatorShadow (SVec combine nubNat)

-- §1. 32-D pathion product and verified zero-divisor pairs.
def pSmul (a b : SVec) : SVec :=
  combine (a.flatMap (fun p => b.map (fun q => (p.1 ^^^ q.1, p.2 * q.2 * pathSigma p.1 q.1))))
def pVec (u : PrimPath) : SVec := [(u.lo, (1 : Int)), (u.hi, if u.neg then (-1 : Int) else 1)]
def pZero (u v : PrimPath) : Bool := (combine (pSmul (pVec u) (pVec v))).all (fun p => p.2 == 0)

/-- Ordered pathion zero-divisor pairs (intra-fiber, as at the sedenion level). -/
def pathZD : List (PrimPath × PrimPath) :=
  pathPrims.flatMap (fun u => pathPrims.filterMap (fun v =>
    if u ≠ v && pathXorLabel u == pathXorLabel v && pZero u v then some (u, v) else none))

/-- There are exactly 3696 ordered pathion ZD pairs (11× the sedenion 336). -/
theorem pathZD_count : pathZD.length = 3696 := by native_decide

-- §2. The interleaved-star probe, lifted to 32-D.
def pLeafIdx (t : Nat) : Nat := if t % 2 == 1 then (t + 1) / 2 else 16 + t / 2
def pCoord (m i k : Nat) : Int :=
  if i == 0 then 0 else if i ≤ m then (if k == pLeafIdx i then 1 else 0) else 0
def pVpairs (m : Nat) : List (Nat × Nat) :=
  (List.range (m+1)).flatMap (fun i =>
    (List.range (m+1)).filterMap (fun j => if i < j then some (i, j) else none))
def pProbeVec (m i : Nat) : SVec :=
  (List.range 32).filterMap (fun k => let c := pCoord m i k; if c == 0 then none else some (k, c))
def pDiff (m i j : Nat) : SVec :=
  combine (pProbeVec m i ++ (pProbeVec m j).map (fun p => (p.1, -p.2)))
def pNsq (v : SVec) : Int := (combine v).foldl (fun a q => a + q.2 * q.2) 0
def pUdc (m : Nat) (edge : Nat → Nat → Bool) : Nat := ((pVpairs m).filter (fun p => edge p.1 p.2)).length
def pClassEdge (m i j : Nat) : Bool := pNsq (pDiff m i j) == 1
def pAssocEdge (m : Nat) (u v : PrimPath) (i j : Nat) : Bool :=
  pNsq (pSmul (pSmul (pDiff m i j) (pVec u)) (pVec v)) == 4
def pClassCount (m : Nat) : Nat := pUdc m (pClassEdge m)
def pAssocMax (m : Nat) : Nat := (pathZD.map (fun pr => pUdc m (pAssocEdge m pr.1 pr.2))).foldl max 0

-- §3. Theorems.
/-- The `m`-leaf pathion star has exactly `m` classical unit pairs. -/
theorem path_classical_count : [6, 14, 18].all (fun m => pClassCount m == m) = true := by
  native_decide

theorem path_assoc_max_6  : pAssocMax 6  = 12 := by native_decide
theorem path_assoc_max_14 : pAssocMax 14 = 44 := by native_decide
theorem path_assoc_max_18 : pAssocMax 18 = 72 := by native_decide

/-- **The sedenion ceiling is lifted.** At 15 vertices the pathion associator star
    reaches 44 unit pairs — strictly above 40, the star-family sedenion ceiling proved
    in `SounioErdos90UnitSpectrum.star_assoc_max_14` (= 40 at the same 15 vertices). -/
theorem pathion_breaks_sedenion_ceiling : pAssocMax 14 > 40 := by
  rw [path_assoc_max_14]; decide

/-- **Growth continues past the lifted ceiling.** 12 < 44 < 72, and 44 > 40: the
    non-associative count-growth is not capped by the sedenion algebra. -/
theorem pathion_growth_summary :
    pAssocMax 6 < pAssocMax 14 ∧ pAssocMax 14 < pAssocMax 18 ∧ pAssocMax 14 > 40 := by
  rw [path_assoc_max_6, path_assoc_max_14, path_assoc_max_18]
  decide

end Sounio.Erdos90Pathion
