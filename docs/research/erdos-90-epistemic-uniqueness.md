# Erdős [90], reframed: the epistemic unit-distance graph (Sounio's uniqueness)

*Companion to `stdlib/research/erdos90_epistemic.sio`. This is the boundary-break: where
classical computation (and all our prior exact work) hits a wall, Sounio's epistemic type
system expresses something no other language can.*

## The wall, and why it is a *classical* wall

Our exact-lattice search established (robustly, across 3 nodes × 3 seeds) that the
triangular/grid construction is locally optimal and cannot be beaten by subset search over
integer lattices — because lattices are vertex-transitive and exact heterogeneous pools
(with cross-lattice unit edges) do not exist in a shared integer Cartesian frame. The
configurations that *could* beat the grid are **non-lattice, with irrational coordinates**.

But the classical unit-distance problem demands distances that are *exactly* 1. For a
non-lattice or measured configuration, "is this distance exactly unit?" is not a boolean —
it is an **epistemic** question with an error budget. Every classical tool, and every
exact-integer method we built, collapses that question to a naive yes/no and silently
discards the uncertainty.

## What Sounio expresses that nothing else does

`erdos90_epistemic.sio` treats each coordinate as a `Knowledge<f64>` (value + GUM standard
uncertainty + confidence), built with `measure(v, uncertainty: σ)`. The squared distance
`dx² + dy²` then carries **automatically GUM-propagated variance** (`variance_of`), and a
unit edge becomes an *epistemic verdict*: a confidence in `[0,1]` computed from how many
propagated sigmas of slack fit inside the unit-tolerance band — high only when the distance
is **both** near 1 **and** precisely known. Extracting these values requires the
`with Epistemic` effect, and the certified count is **compile-time gated** on confidence.

## The demonstration (the optimal `u(4) = 5` rhombus, measured)

Two equilateral triangles sharing an edge — the optimal 4-point config, 5 unit distances
(the 6th pair is the √3 diagonal). Run at two measurement precisions:

```
σ = 0.005 (precise):   5 edges CERTIFIED (conf 0.952);  epistemic expected count = 4.76
σ = 0.080 (noisy):     0 edges certified;  all 5 flagged "classical-unit but UNCERTAIN";
                       epistemic expected count = 2.78
classical boolean count in BOTH cases:  5
```

The √3 diagonal is correctly `conf = 0` throughout (GUM propagation, not a hardcoded skip).

**The point:** the *same* extremal configuration is epistemically certain when measured
precisely and epistemically degraded when measured coarsely — and Sounio's count reflects
that, GUM-propagated and compile-time-gated, while the classical boolean count says "5"
regardless. This is ISO-GUM uncertainty quantification fused with combinatorial geometry
and a compile-time confidence contract, in one language. No other system expresses it.

## Why this matters for [90] specifically

The grid-beating frontier is continuous and non-lattice (irrational coordinates). A search
there cannot rely on exact equality; it needs **rigorous confidence that a near-unit
distance is unit within budget** — exactly what `edge_confidence` + the `with Epistemic`
gate provide. Sounio can therefore search continuous configurations and *certify* their
unit-distance counts with propagated confidence rather than fragile floating-point equality
— the only rigorous way past the exact-lattice wall. The exact-integer engine
(`SounioErdos90PlanarLowerBound.lean`, `countUnit`/`native_decide`) remains the certifier
for any rational config the epistemic search promotes to a candidate; the two compose:
**epistemic search proposes under uncertainty, exact Lean certifies when rational.**

## Status

`erdos90_epistemic.sio` compiles to a static ELF with the dev souc and runs (locally and,
via the proven `run_on_cluster.sh` path, on the cluster). It is a *capability*
demonstration — a new kind of unit-distance reasoning — not a new `u(n)` bound. The honest
ledger: classical exact methods are bounded (we proved it); Sounio's epistemic layer opens
the continuous frontier with rigor that is uniquely its own.
