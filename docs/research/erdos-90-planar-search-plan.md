<!-- docs:meta
topic_id: repo.docs.research.erdos-90-planar-search-plan
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.erdos-90-planar-search-plan
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Erdős [90] — planar `u(n)` cluster search plan + pilot results

*Companion to `formal/lean4/SounioErdos90PlanarLowerBound.lean` and the CPU search kernel
`stdlib/research/erdos90_search.sio`. Goal: attack the classical open problem — the
maximum number `u(n)` of unit distances among `n` points in ℝ² — with the Sounio
compiler, K-AXI→PTX, and the GPU cluster. Honest framing throughout.*

## What is open, exactly

* **Lower bound (baseline, verified).** The triangular lattice (Eisenstein integers ℤ[ω])
  gives `u(n) ≥ ⌊3n − √(12n−3)⌋` =: `harb(n)` (Harborth 1974). Verified for n=1..18 with
  explicit exact configs in `lattice_achieves_harborth` (unit distance ⟺ `x²+xy+y²=1`,
  exact integer arithmetic, no floats).
* **The open question.** Is `u(n) = harb(n)` for all `n`, or can a different construction
  beat it? Exact `u(n)` is settled only for small `n`. Lattice-NN-optimality is folklore,
  unproven in general; and asymptotically the Erdős grid is known to win.
* **Upper bound.** `u(n) = O(n^{4/3})` (Spencer–Szemerédi–Trotter 1984); the exponent gap
  to the `n^{1+c/loglog n}` lower bound is THE famous open part and is **not** closable by
  finite search. We do not target the exponent.

## Pilot results (CPU, `erdos90_search.sio`, exact integer)

The pilot takes `n = W×W` points of a square ℤ² region and chooses unit² = `N` with many
sum-of-two-squares representations `r₂(N)` (interior points then have `r₂(N)` unit
neighbours), comparing to the triangular `harb(n)`:

```
n      harb(n)  bestGrid  unit²N   winner
25     57       48        5        triangular
225    623      828       25       GRID (beats)
625    1788     3144      65       GRID
1225   3553     7144      —        GRID
2025   5919     13320     325      GRID
3025   8884     22640     325      GRID
```

Clean exact **crossover**: tiny `n` favours the triangular nearest-neighbour lattice, but
by `n ≈ 225` the Erdős grid (many-representation distance) decisively wins, gap widening
to ~2.5×, with the optimal `N` climbing 5→25→65→325 as `r₂(N)` grows (4→8→12→16→24). So
`u(n) > harb(n)` for `n ≳ 225` — the triangular NN lattice is provably **not** optimal
there. (This reproduces the known Erdős phenomenon *explicitly and exactly*; the value of
the pilot is a **validated, certifiable search engine**, not a new theorem.)

Gotcha recorded: the native backend BSS-zeroes global `var`-array initializers, so the
kernel tests candidate `N` as literals (`consider(N)`), not via an `NCAND[]` array.

## The exact, GPU-able search (no floats, fully certifiable)

1. **Pool.** A finite pool `P` of exact points (bounded ℤ²/Eisenstein region, or a union
   of lattices — where non-trivial optima hide). Squared distances are integers.
2. **Unit predicate.** Edge iff `dx²+dy² == N` (integer compare).
3. **Objective.** Over size-`n` subsets `S ⊆ P`, maximize induced unit-edge count.
4. **Search (GPU-parallel).** Many parallel guided searches (simulated annealing / tabu:
   add/drop/swap a point), per-thread bitmask + incremental edge count — a K-AXI-shaped
   integer kernel → `kretikos_emit_kaxi` → PTX, launched via
   `scripts/ci/kretikos_kaxi_l4_launch_gate.sh` or BeagleCockpit MCP.
5. **Certify.** Any subset matching/beating the baseline is dumped as explicit integer
   coordinates and re-verified by `countUnit` in Lean (`native_decide`) — the GPU only
   *proposes*, Lean *certifies*. No trust in any float path.

## Build status / next steps

* DONE: exact lattice lower-bound (Lean, verified) + CPU search kernel (Sounio, runs,
  pilot crossover reproduced).
* NEXT: (a) lift the per-thread subset search to a K-AXI kernel; validate bit-exact vs the
  CPU run on a small pool before scaling (cf. `feedback_l4_gate_catches_what_local_misses`);
  (b) cluster sweep over seeds × N × region shapes on the L4/A5000 lane via MCP; (c) fold
  any record configs back into the Lean file as witnessed theorems (`countUnit … = …`).

## Honest caveats

* The grid-beats-triangular crossover is **known** (Erdős); the pilot makes it explicit
  and exact. A genuinely new result needs configs beating *all* known constructions for a
  specific `n`, or extending exact `u(n)` records — the high-risk prize.
* The GPU is a **proposer**, not a prover: every reported config is re-certified in Lean.
* This does **not** touch the `n^{4/3}` exponent gap; that is not finite-search-accessible.

## Cluster run (validated, 2026-05-25)

The Sounio search ELF ran on the live Slurm cluster via `slurm-jobs/erdos90/run_on_cluster.sh`
(direct-srun fallback; the BeagleCockpit MCP tools were not loaded this session). Topology
discovered: compute nodes do **not** mount `/workspace` — the only shared FS is OrangeFS at
`/orangefs/training` (pvfs2), invisible to the login container. Path used: compile to a
static self-contained ELF locally → ship bytes over `srun` stdin as base64 → decode + run on
the node → results on stdout.

Scaled cluster run on `cpuops-t560-proxmox` (8 cores), n up to 16384, wall = 8s:

```
n       harb(n)   bestGrid   N(unit²)   ratio
9216    27315     88320      1105       3.23x
12544   37244     130816     1105       3.51x
16384   48708     181504     1105       3.69x   (N=1105=5·13·17, r₂=32)
```

The grid-over-triangular margin grows with `n` (the winning many-representation distance
climbs 5→25→65→325→1105 as `r₂(N)` increases), exactly the Erdős phenomenon at scale — an
explicit large-`n` lower bound `u(16384) ≥ 181504`, in exact integer arithmetic.

Next (genuine open frontier, needs the array-job/optimizer build): replace the full-square
enumeration with a per-thread subset local-search (anneal/tabu) over a large exact pool, run
as a Slurm array (many seeds × n), hunting for configs that beat *all* known constructions —
then re-certify any record in Lean. The K-AXI/PTX GPU port (idle gpu-orangefs nodes r740/5860)
is the throughput multiplier for that search.

## Optimizer array sweep (3 nodes, 2026-05-25)

`stdlib/research/erdos90_optimize.sio` improves the pilot via (1) COMPACT DISK regions
(x²+y² ≤ rr) instead of squares — fewer boundary-deficient points — and (2) a broad
unit-distance² sweep. Compiled to 3 banded static ELFs and run in PARALLEL across the
three idle nodes via `srun` (the same compile→base64→srun path):

```
band  node                       peak n   count   N(unit²)  count/harb
A     cpuops-t560-proxmox         2693    19848   325       2.51x
B     gpuorangefs-r740-proxmox    5369    46688   325       2.94x
C     gpuorangefs-5860-proxmox    7845    73376   1105      3.15x   (harb=23228)
```

Headline: an explicit exact compact-disk configuration of **7845 points with 73376 unit
distances** — `u(7845) ≥ 73376`, **3.15×** the triangular nearest-neighbour bound. The
ratio grows with n and the winning many-representation distance climbs (325 → 1105) as
larger disks admit denser N. The disk shape strictly improves the square pilot's per-point
efficiency. Every config is exact integer and re-certifiable in Lean (`countUnit`).

Honest standing: this is the strongest *explicit* lower bound from the Erdős grid family
with optimized shape + distance — it does **not** beat the asymptotic Erdős construction or
touch the n^{4/3} exponent gap. Genuinely-new territory (beating *all* known constructions
for a specific n, or extending exact small-n u(n) records) needs a richer-than-lattice pool
and a true subset local-search, the next build. The cluster path and certifiable engine are
now proven end-to-end.

## Subset optimizer — densest-k-subgraph search (3 nodes × 3 seeds, 2026-05-25)

`stdlib/research/erdos90_subset.sio`: hill-climb densest-k-subgraph over the ℤ² distance-N
graph (random-swap moves: drop a member, add a non-member, accept if internal edge count
rises; Park–Miller RNG), restarted from the compact disk + random seeds. Compares best
found to the compact-disk baseline. Run as 3 independent seed variants across the three
nodes; broad N sweep including ANISOTROPIC distances where the optimal shape need not be a
disk.

Result (identical across all 3 seeds — robust):

* **Isotropic N (2, 5, 10, 13, 25, 65):** the compact disk *is* the densest k-subgraph —
  every case "disk-optimal", no subset beats it. Expected: ℤ² is vertex-transitive, so the
  densest induced subgraph of its unit-distance graph is the compact region.
* **Anisotropic N = 50** (neighbours (±1,±7),(±7,±1),(±5,±5)): the disk is genuinely
  **suboptimal** — the search robustly reshapes it `2447 → 2767` unit pairs (+13%), the same
  on all three seeds. A real shape finding: for anisotropic distances the optimal region is
  not a disk.
* **No record, though:** the reshaped N=50 config (2767) still trails the N=25 disk
  (~2780) at n=600. The optimal-N grid/disk construction stands; **no `u(n)` record beaten.**

Honest conclusion: comprehensive densest-k-subgraph search confirms the compact-disk /
optimal-N grid is locally optimal and robust, and pinpoints where the disk assumption breaks
(anisotropic N) without yielding a better global construction. This is consistent with why
the problem is hard — no exact periodic-pool subset search can beat the grid, because
(i) lattices are vertex-transitive and (ii) square ℤ² and triangular ℤ[ω] share no integer
Cartesian frame, so heterogeneous exact pools with cross-lattice unit edges don't exist.
Genuinely beating the grid would require non-lattice rational configs / the additive-energy
frontier — beyond exact lattice search. The cluster engine, however, is fully proven:
build → ship ELF over srun → 3-node parallel search → exact, Lean-certifiable configs.

## UPDATE 2026-05-29: the exponent gap has (allegedly) been closed — OpenAI 2026

Throughout this plan we said the `n^{4/3}`-vs-`n^{1+c/loglog n}` **exponent** gap is
"not finite-search-accessible" and "we do not target the exponent." On 2026-05-28 a
Lean 4 formalization appeared claiming exactly that result:

> **OpenAI (2026), *Disproof of Erdős's planar unit-distance conjecture*** —
> `github.com/logical-intelligence/erdos-unit-distance` (Apache-2.0, Lean 4.29.1 +
> Mathlib). The headline `main_theorem` proves
> `∃ δ > 0, { n | νn(n) ≥ n^(1+δ) }.Infinite`, where `νn(n) = u(n)`. A **fixed** `δ>0`
> infinitely often **contradicts** `u(n) = n^{1+o(1)}` (Erdős's conjecture).

**Trust base (as published).** Conditional on two classical, well-cited hypotheses
isolated in `main_theorem`'s signature: the Golod–Shafarevich inequality
(`d(Q)² < 4 r(Q)`, NSW 3.9.7 / Serre) and Shafarevich's relation-rank bound (Mayer
Thm 5.1, S=∅ / Koch 11.5+11.8). `#print axioms` reduces to `propext`,
`Classical.choice`, `Quot.sound` — no bespoke axioms, no `sorry`; CI runs `lake build`
+ `leanchecker`, with a heavier `lean4export`+nanoda kernel pass locally.

**The mechanism (decoded from the Lean source).** Not geometry of lattices —
**class field towers.** Pick a cyclic cubic totally-real field `F` (a subfield of
`ℚ(ζ_r)`, `r` prime `≡ 1 mod 3`) whose maximal **unramified pro-3 extension is
infinite** (Golod–Shafarevich: `r ≤ d+2` from Shafarevich forces `d² < 4r` to fail
for any finite quotient). This gives a tower of fields `L_j/F`, all unramified,
`deg → ∞`. Set `K_j = L_j(i)` — that is the *plane* (the `i² = −1` complex place),
`AdmissibleDatum` in the Lean. Fix `t` rational primes `q ≡ 1 (mod 4)` that split
completely in `L`. The exponent excess is

```
γ = t·log 2 − log H > 0,      H = class-number growth base (h(K_j) ≤ H^f).
```

Each split prime `q ≡ 1 (mod 4)` **doubles** the count of algebraic elements of a
given norm (factor 2 ⇒ the `t·log 2` term); the class number is the loss (`log H`).
When `t·log 2 > log H`, the unit-distance count on `n` algebraic points of `K_j`
beats `n^{1+o(1)}`. A geometric double-count (`overlapArea(R) = 2R²·arccos(1/2R) −
½√(4R²−1)`, the lens area of two unit-separated disks) turns the algebraic
representation count into the planar `u(n)` bound.

**This is the infinite-degree lift of our pilot.** Our search repeatedly found the
winning distance `N` climbing `5 → 65 → 325 → 1105` as the **sum-of-two-squares count
`r₂(N)` grows** (the doc's own `r₂(N) = 32` at `N=1105=5·13·17`). That `r₂` doubling
*is* the `t·log 2` engine — in the base plane `ℤ[i]`, `r₂(∏ q_i) = 4·2^t` for `t`
distinct primes `q_i ≡ 1 (mod 4)`. We verify this exact-arithmetic core independently
in `examples/erdos/erdos90_repcount_engine.sio` (8→16→32→64; and any `≡ 3 (mod 4)`
factor sends `r₂` to 0 — the construction's `q_mod4` hypothesis). What our finite /
lattice search **structurally cannot** do — and the OpenAI result makes precise — is
let the *field degree* grow: the exponent lever is unbounded-degree number fields, not
a cleverer planar lattice. Our "honest caveat" (`exponent gap is not
finite-search-accessible`) is corroborated, and the actual key is now identified.

**NB / correction.** The cross-lattice "foreclosure" remark in the densest-k-subgraph
section above (ℤ² and ℤ[ω] sharing no integer Cartesian frame ⇒ heterogeneous unit
edges "don't exist") was already found **invalid** in a 2026-05-28 `math-review`
(logged in `.claude/llm_offload_log.md`); the OpenAI construction reinforces the
correction — exact unit-distance richness lives in number fields of growing degree,
not in any fixed integer frame.

**Honest standing.** This is a **3-days-old, not-yet-peer-reviewed** artifact; the
"OpenAI 2026" attribution and the result itself await community scrutiny, and two
classical inputs are assumed (not proved) in Lean. We verify only the **finite
combinatorial core** (`erdos90_repcount_engine.sio`), not the class field tower, which
is infinitary and outside exact computation. We make **no** independent claim on the
exponent.
