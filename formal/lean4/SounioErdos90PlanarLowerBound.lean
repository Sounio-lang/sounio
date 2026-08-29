/-!
# Sounio — Erdős [90] planar unit distances: the exact lattice lower bound

`u(n)` = the maximum number of unit-distance pairs among `n` points in ℝ². The
extremal lower-bound configurations live on the triangular lattice (Eisenstein
integers ℤ[ω]). In axial integer coordinates `(a,b)` (point `= a·(1,0) + b·(1/2,√3/2)`),
the squared Euclidean distance is the **integer** Eisenstein norm

> `‖(a,b) − (c,d)‖² = x² + xy + y²,  x=a−c, y=b−d`

so a *unit distance* is exactly `x²+xy+y² = 1` (the six units of ℤ[ω]). Exact integer
arithmetic, fully `native_decide`. (ℤ[ω] ⊂ ℂ = level-1 Cayley–Dickson algebra.)

`greedy R n` selects `n` lattice cells by maximal marginal unit-edge gain (deterministic,
exact). `harb n = ⌊3n − √(12n−3)⌋` is Harborth's triangular-lattice formula.

* `lattice_achieves_harborth` : for `n = 1..18` the greedy config has **exactly** `harb n`
  unit-distance pairs — an explicit, exact, planar witness. Hence `u(n) ≥ ⌊3n−√(12n−3)⌋`,
  machine-checked with concrete configurations.

This is the rigorous **baseline** the cluster search must beat. The CPU search pilot
(`stdlib/research/erdos90_search.sio`) shows the Erdős ℤ² grid construction exceeds this
triangular nearest-neighbour bound for `n ≳ 225` (e.g. n=225: 828 > 623), reproducing the
known crossover explicitly and exactly. See `docs/research/erdos-90-planar-search-plan.md`.
-/

namespace Sounio.Erdos90Planar

/-- Squared Euclidean distance on the triangular lattice (Eisenstein norm), exact integer. -/
def d2 (a b : Int × Int) : Int := let x := a.1 - b.1; let y := a.2 - b.2; x*x + x*y + y*y

/-- Hex (graph) distance of a lattice cell from the origin. -/
def hexdist (p : Int × Int) : Int :=
  (Int.natAbs p.1 + Int.natAbs p.2 + Int.natAbs (p.1 + p.2)) / 2

/-- All lattice cells within hex-radius `R` of the origin. -/
def cells (R : Nat) : List (Int × Int) :=
  (List.range (2*R+1)).flatMap (fun i => (List.range (2*R+1)).filterMap (fun j =>
    let a : Int := (Int.ofNat i) - (Int.ofNat R)
    let b : Int := (Int.ofNat j) - (Int.ofNat R)
    if hexdist (a,b) ≤ (Int.ofNat R) then some (a,b) else none))

/-- Number of already-chosen cells at unit distance from `p`. -/
def deg (p : Int × Int) (chosen : List (Int × Int)) : Nat :=
  (chosen.filter (fun q => d2 p q == 1)).length

/-- Greedy: repeatedly add the unchosen cell of `cells R` gaining the most unit edges
    (ties → smaller hex-distance, then lexicographic). Deterministic. -/
def greedy (R : Nat) (n : Nat) : List (Int × Int) :=
  (List.range n).foldl (fun chosen _ =>
    let cand := (cells R).filter (fun p => ¬ chosen.contains p)
    match cand.foldl (fun best p =>
      let g := deg p chosen
      match best with
      | none => some (p, g)
      | some (bp, bg) =>
        if g > bg || (g == bg && (hexdist p < hexdist bp ||
          (hexdist p == hexdist bp && (p.1 < bp.1 || (p.1 == bp.1 && p.2 < bp.2)))))
        then some (p, g) else some (bp, bg)) none with
    | some (p, _) => chosen ++ [p]
    | none => chosen) []

/-- Count of unit-distance pairs in a point list (unordered). -/
def countUnit (pts : List (Int × Int)) : Nat :=
  ((pts.zipIdx).flatMap (fun (p, i) => (pts.zipIdx).filterMap (fun (q, j) =>
    if i < j && d2 p q == 1 then some (1 : Nat) else none))).length

/-- Integer square root (Mathlib-free). -/
def isqrt (n : Nat) : Nat := (List.range (n+1)).foldl (fun acc k => if k*k ≤ n then k else acc) 0

/-- Harborth's triangular-lattice maximum: `⌊3n − √(12n−3)⌋`. -/
def harb (n : Nat) : Nat := 3*n - (let s := isqrt (12*n - 3); if s*s < 12*n - 3 then s+1 else s)

/-- **Planar lower bound `u(n) ≥ ⌊3n − √(12n−3)⌋`, witnessed.** For every `n = 1..18` the
    greedy triangular-lattice configuration realizes *exactly* `harb n` unit distances
    (exact Eisenstein arithmetic). So `u(n)` is at least the Harborth value, certified by
    explicit configurations. The rigorous baseline for the cluster search. -/
theorem lattice_achieves_harborth :
    ((List.range 18).map (·+1)).all (fun n => countUnit (greedy 4 n) == harb n) = true := by
  native_decide

/-! ## Erdős ℤ² compact-disk construction

The CPU search (`stdlib/research/erdos90_optimize.sio`) finds that the compact
disk of integer points with x²+y² ≤ 5000, using unit distance² = 1105, gives a
large explicit lower bound. We certify the count here by pure integer
computation in Lean core (`native_decide`). -/

/-- All integer points inside the origin-centred disk of radius² = rr. -/
def compactDiskZ2 (rr : Nat) : List (Int × Int) :=
  let R : Int := isqrt rr
  (List.range (2 * R.toNat + 1)).flatMap (fun i =>
    (List.range (2 * R.toNat + 1)).filterMap (fun j =>
      let x : Int := (Int.ofNat i) - R
      let y : Int := (Int.ofNat j) - R
      if x*x + y*y ≤ (Int.ofNat rr) then some (x, y) else none))

/-- Count unordered pairs at squared Euclidean distance exactly `nsq`. -/
def countUnitSq (pts : List (Int × Int)) (nsq : Nat) : Nat :=
  ((pts.zipIdx).flatMap (fun (p, i) => (pts.zipIdx).filterMap (fun (q, j) =>
    if i < j then
      let dx := p.1 - q.1
      let dy := p.2 - q.2
      if dx*dx + dy*dy == (Int.ofNat nsq) then some (1 : Nat) else none
    else none))).length

/-- **New lower bound:** among the 15705 integer points with x²+y² ≤ 5000 there are
    at least 176768 pairs at squared distance 1105, so u(15705) ≥ 176768. -/
theorem erdos90_compact_disk_u15705 :
    countUnitSq (compactDiskZ2 5000) 1105 ≥ 176768 := by
  native_decide

/-- **New lower bound:** among the 31417 integer points with x²+y² ≤ 10000 there are
    at least 405648 pairs at squared distance 5525, so u(31417) ≥ 405648. -/
theorem erdos90_compact_disk_u31417 :
    countUnitSq (compactDiskZ2 10000) 5525 ≥ 405648 := by
  native_decide

end Sounio.Erdos90Planar
