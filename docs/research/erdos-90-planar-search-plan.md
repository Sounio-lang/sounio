# Erdős [90] — planar `u(n)` cluster search plan

*Companion to `formal/lean4/SounioErdos90PlanarLowerBound.lean`. Goal: attack the
classical open problem — the maximum number `u(n)` of unit distances among `n` points in
ℝ² — with the Sounio compiler, K-AXI→PTX, and the GPU cluster. Honest framing below.*

## What is open, exactly

* **Lower bound (baseline, now verified).** The triangular lattice (Eisenstein integers
  ℤ[ω]) gives `u(n) ≥ ⌊3n − √(12n−3)⌋` =: `harb(n)` (Harborth 1974). Verified for n=1..18
  with explicit exact configs in `lattice_achieves_harborth` — exact integer arithmetic,
  `x²+xy+y² = 1` for unit distance, no floats.
* **The open question.** Is `u(n) = harb(n)` for all `n`, or can a **non-lattice**
  configuration beat the lattice for some `n`? Exact `u(n)` is settled only for small `n`
  (exhaustive search). Lattice-optimality is folklore/conjecture, unproven in general.
* **Upper bound.** `u(n) = O(n^{4/3})` (Spencer–Szemerédi–Trotter 1984); the exponent gap
  to the `n^{1+c/loglog n}` lower bound is THE famous open part and is **not** closable by
  finite search. We do not target the exponent.

**Honest success ladder.** L1: confirm `u(n) = harb(n)` over a searched pool for a range
of n (modest, verifies lattice-optimality empirically). L2: find a config **= harb(n)**
that is provably non-lattice (structural interest). L3: find a config with **> harb(n)**
unit distances for some n — a genuine new result (would refute lattice-optimality for that
n). L3 is hard; L1 is the realistic near-term cluster output.

## The exact, GPU-able search (no floats, fully certifiable)

Keep everything in exact integer arithmetic so any candidate is rigorously verifiable
(and re-checkable in Lean):

1. **Point pool.** Fix a finite pool `P` of exact points whose pairwise squared distances
   are integers (or integers after a common scaling): e.g. a bounded region of a *fine*
   triangular/Eisenstein lattice, or a **union of lattices** (triangular ∪ rotated/scaled
   copies) — the natural place non-lattice optima hide. `|P|` ~ 10²–10⁴.
2. **Unit predicate.** Edge iff scaled squared distance equals the chosen unit² (integer
   compare). Precompute the pool adjacency once.
3. **Objective.** Over size-`n` subsets `S ⊆ P`, maximize the induced unit-edge count.
   This is "max edges in an `n`-vertex induced subgraph of the pool graph."
4. **Search (GPU-parallel).** Subset space is huge, so run many parallel guided searches
   (simulated annealing / genetic / tabu local moves: add/drop/swap a point) — thousands
   of independent search threads per GPU block, each carrying a bitmask of `S` and an
   incremental edge count. This is a K-AXI-shaped integer kernel (fixed pool table,
   per-thread state, no dynamic allocation) — emit via `kretikos_emit_kaxi` → PTX, launch
   through `scripts/ci/kretikos_kaxi_l4_launch_gate.sh` or BeagleCockpit MCP.
5. **Certify.** Any subset beating/matching `harb(n)` is dumped as explicit integer
   coordinates and re-verified by `countUnit` in Lean (`native_decide`) — same machinery
   as the baseline file. No trust in the GPU float path; the GPU only *proposes*, Lean
   *certifies*.

## Build steps

1. **Sounio kernel** `stdlib/.../erdos90_search.sio`: pool builder (exact lattice-union
   coords), adjacency precompute, incremental local-move objective, RNG-seeded annealing.
   CPU-run first (souc) to reproduce `harb(n)` on the pure triangular pool (sanity = the
   verified baseline), then search lattice-unions.
2. **K-AXI port**: lift the per-thread search loop to a K-AXI kernel; validate bit-exact
   vs the CPU run on a small pool before scaling (cf. `feedback_l4_gate_catches_what_local_misses`).
3. **Cluster sweep**: launch many seeds × `n` values on the L4/A5000 lane via MCP; collect
   any `≥ harb(n)` configs.
4. **Lean certification**: fold collected configs into `SounioErdos90PlanarLowerBound` as
   new witnessed theorems (extend `lattice_achieves_harborth` / add `beats_harborth` if any).

## Honest caveats

* The triangular lattice is conjectured optimal and verified for small n, so a pool search
  will most likely **confirm** `u(n)=harb(n)` for the searched range — a real but modest
  (L1) outcome. A genuine beat (L3) is unlikely at small n and is the high-risk prize.
* The GPU is a **proposer**, not a prover: every reported config is re-certified exactly in
  Lean. Float realizability heuristics are never trusted as results.
* This does **not** touch the `n^{4/3}` exponent gap; that is not finite-search-accessible.
