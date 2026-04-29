/-!
# Sounio.FanoLabellingOrbits — Prereg §7.1 labelling control

Formalisation of the 30 inequivalent labellings of the seven imaginary
octonion basis elements under the Fano-preserving subgroup of `S_7`.

This is the artefact whose production is deferred in
`docs/papers/preregistrations/DEVIATIONS.md` entry **0001** (annotation,
not a protocol deviation). See also
`scripts/research/ossm_168_dryrun/fano_orbits.py` for the companion
Python enumeration that agrees modulo choice of coset representative.

Proved here, all by `native_decide`:

  * `fano_automorphism_group_card : (fanoAuts).length = 168`
    — `|Aut(Fano)| = |PGL(3, 𝔽₂)| = |PSL(2, 7)| = 168`

  * `s7_card : (allPerms7).length = 5040`
    — `|S₇| = 5040`

  * `fano_labelling_orbits_count : (cosetReps).length = 30`
    — `|S₇ / Aut(Fano)| = 30`, equivalently the number of inequivalent
    Fano labellings, which is what the prereg text refers to.

File is deliberately self-contained (no `import`, no Mathlib, no
`OctonionAlgebra` dependency) so that it survives independent of the
rest of `formal/` and can be checked in isolation by:

    lean formal/FanoLabellingOrbits.lean

References:
  - Conway & Smith 2003, "On Quaternions and Octonions", §11
  - Baez 2002, "The Octonions", §2.2
  - Preregistration v3 §7.1; DEVIATIONS entry 0001
-/

namespace Sounio.FanoLabellingOrbits

/-- The seven Fano lines on `{1..7}` under the XOR characterisation
    `i XOR j XOR k = 0` for `i < j < k` distinct in `{1..7}`. -/
def fanoLines : List (Nat × Nat × Nat) :=
  [(1,2,3), (1,4,5), (1,6,7), (2,4,6), (2,5,7), (3,4,7), (3,5,6)]

/-- Sort a triple of naturals ascending. -/
@[inline] def sortTriple (a b c : Nat) : Nat × Nat × Nat :=
  let (x, y) := if a ≤ b then (a, b) else (b, a)
  let (x, z) := if x ≤ c then (x, c) else (c, x)
  let (y, z) := if y ≤ z then (y, z) else (z, y)
  (x, y, z)

/-- Membership test for the Fano line set (as unordered 3-subsets). -/
def isFanoLine (a b c : Nat) : Bool :=
  fanoLines.contains (sortTriple a b c)

/-- Insert `x` at every position of `ys`; produces `ys.length + 1`
    lists. -/
def insertEverywhere (x : Nat) : List Nat → List (List Nat)
  | []      => [[x]]
  | y :: ys => (x :: y :: ys) :: (insertEverywhere x ys).map (y :: ·)

/-- All permutations of a list. Deterministic: for distinct inputs
    produces `n!` lists. -/
def perms : List Nat → List (List Nat)
  | []      => [[]]
  | x :: xs => (perms xs).flatMap (insertEverywhere x)

/-- The 5040 permutations of `{1..7}`, 1-indexed. Ordering is determined
    by `perms` and is independent of Lean build settings. -/
def allPerms7 : List (List Nat) := perms [1,2,3,4,5,6,7]

/-- Apply a length-7 permutation σ (1-indexed) to a line, returning the
    sorted image. `σ.getD (i-1) 0` gives the image of `i` under `σ`. -/
def applyPerm (σ : List Nat) (line : Nat × Nat × Nat) : Nat × Nat × Nat :=
  let (a, b, c) := line
  let a' := σ.getD (a - 1) 0
  let b' := σ.getD (b - 1) 0
  let c' := σ.getD (c - 1) 0
  sortTriple a' b' c'

/-- σ is a Fano automorphism iff it maps every Fano line (setwise) to a
    Fano line. -/
def isFanoAut (σ : List Nat) : Bool :=
  fanoLines.all (fun line =>
    let (a, b, c) := applyPerm σ line
    fanoLines.contains (a, b, c))

/-- Enumeration of `Aut(Fano) ≤ S₇` as the subgroup preserving the line
    set. -/
def fanoAuts : List (List Nat) := allPerms7.filter isFanoAut

/-- **|Aut(Fano)| = 168.** -/
theorem fano_automorphism_group_card : fanoAuts.length = 168 := by native_decide

/-- **|S₇| = 5040.** -/
theorem s7_card : allPerms7.length = 5040 := by native_decide

/-- Compose two length-7 1-indexed permutations: `(σ ∘ τ)(i) = σ(τ(i))`. -/
def composePerm (σ τ : List Nat) : List Nat :=
  τ.map (fun v => σ.getD (v - 1) 0)

/-- Lex-minimum canonical representative of the left coset `σ · H`, where
    `H = Aut(Fano)`. Two permutations share a `cosetRep` iff they lie in
    the same left coset. -/
def cosetRep (σ : List Nat) : List Nat :=
  fanoAuts.foldl (fun acc h =>
    let p := composePerm σ h
    if p < acc then p else acc) σ

/-- Deduplicate a list. Structurally recursive on the tail; keeps the
    last occurrence of each element, which is sufficient for counting. -/
def dedup : List (List Nat) → List (List Nat)
  | []      => []
  | x :: xs => if xs.contains x then dedup xs else x :: dedup xs

/-- One representative per left coset of `Aut(Fano)` in `S₇`, in the order
    produced by `allPerms7`. Agrees with
    `scripts/research/ossm_168_dryrun/fano_orbits.py::coset_representatives_S7_mod_Aut`
    modulo the choice of representative per coset. -/
def cosetReps : List (List Nat) := dedup (allPerms7.map cosetRep)

/-- **|S₇ / Aut(Fano)| = 30.** This is the theorem named
    `fano_labelling_orbits_count` referenced in the frozen
    pre-registration §7.1 and satisfying DEVIATIONS entry 0001. -/
theorem fano_labelling_orbits_count : cosetReps.length = 30 := by native_decide

/-- Sanity: by Lagrange, the coset count is `|S₇| / |Aut(Fano)|`. -/
theorem lagrange_consistency : allPerms7.length = cosetReps.length * fanoAuts.length := by
  native_decide

end Sounio.FanoLabellingOrbits
