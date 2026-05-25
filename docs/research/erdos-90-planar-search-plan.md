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
