/-!
# Sounio.Scheduler.CriticalPathLowerBound

Pure Lean 4 stdlib.  No `Mathlib` imports.  No `sorry`.

Companion proof obligation for `cgplan_global_priority_inner` /
`cgplan_phase_a_compute_priority` in `self-hosted/native/codegen_plan.sio`.

The runtime computes per-instruction critical-path priority by DP over
the topological order: each node's priority is its own latency mean plus
the maximum priority over its successors.  The schedule's reported
makespan is upper-quantile (`mean + k · σ`); the deterministic component
is the longest-chain mean.

This file proves:

  `reported_critical_path_geq_topological_max :
    ∀ g k, schedulerCritical g k ≥ topologicalLongestChainMean g`

i.e. the scheduler's reported critical path NEVER underestimates the
deterministic longest-chain mean of the DAG.  Adding the `k · isqrt(var)`
upper-quantile term cannot make the report smaller.

The work here is the type infrastructure: `DAG`, `topologicalLongestChainMean`
by predecessor-DP in topological order, `isqrt` by Newton iteration in
`Nat`, and `schedulerCritical` as the integer fixed-point report.  Once
those line up, the inequality follows from `Nat.le_add_right`.
-/

namespace Sounio.Scheduler.CriticalPath

/-- A node in the DAG: latency mean, latency variance (in cycles²), and a
    list of predecessor node indices.  Predecessors are integer ids that
    must point to earlier nodes (enforced by `DAG.topo_sorted`). -/
structure NodeData where
  latMean : Nat
  latVar : Nat
  preds : List Nat
deriving Inhabited, Repr

/-- A DAG whose nodes are listed in topological order.  Every predecessor
    of node `i` must have index strictly less than `i`. -/
structure DAG where
  nodes : Array NodeData
  topoSorted : ∀ i : Fin nodes.size, ∀ p ∈ (nodes[i]).preds, p < i.val

/-- Look up `arr[p]` returning `0` if out of bounds.  This is the simple,
    total fixed-point form of array indexing; it sidesteps having to
    propagate `p < arr.size` proofs everywhere.  Sound because the DAG's
    `topoSorted` invariant ensures every legitimate predecessor lookup
    happens within bounds, and out-of-bounds entries contribute `0` to
    the longest-chain max (the identity for `Nat.max`). -/
def Array.getOrZero (arr : Array Nat) (p : Nat) : Nat :=
  match arr[p]? with
  | some v => v
  | none => 0

/-- Longest-chain mean ending at a node with the given `NodeData`, given
    the already-computed values for previous nodes in `acc`.  Equation:

      longest(i) = latMean(i) + max { longest(p) | p ∈ preds(i) } ∪ {0}

    With `Nat.max` and empty predecessor list, the inner max is `0`, so
    leaves (no predecessors) get exactly their own `latMean`. -/
def longestEndingAt (nd : NodeData) (acc : Array Nat) : Nat :=
  nd.latMean + nd.preds.foldl (fun m p => Nat.max m (Array.getOrZero acc p)) 0

/-- Topological-DP longest-chain mean: fold over the topologically-ordered
    node array, pushing each node's longest-chain value into the
    accumulator before processing the next.  The final makespan-as-
    longest-chain is the max over all entries. -/
def topologicalLongestChainMean (g : DAG) : Nat :=
  let folded := g.nodes.foldl
    (fun acc nd => acc.push (longestEndingAt nd acc)) (#[] : Array Nat)
  folded.foldl Nat.max 0

/-- Sum of latency variances across the whole DAG.  This upper-bounds the
    variance along any particular chain, matching the scheduler's
    conservative `CGPLAN_PRIO_VAR` propagation. -/
def totalLatencyVar (g : DAG) : Nat :=
  g.nodes.foldl (fun s nd => s + nd.latVar) 0

/-- Newton-style integer square root over `Nat`.  Pure stdlib `Nat`
    arithmetic; mirrors `cgplan_isqrt` in `self-hosted/native/codegen_plan.sio`.
    Terminates by structural recursion on a fuel parameter equal to the
    input — the iteration converges in `O(log n)` steps so this is loose
    but eliminates the well-founded-recursion overhead. -/
def isqrtAux (n : Nat) : Nat → Nat → Nat
  | x, 0 => x
  | x, fuel+1 =>
    if x = 0 then 0
    else
      let y := (x + n / x) / 2
      if y < x then isqrtAux n y fuel else x

def isqrt (n : Nat) : Nat :=
  if n = 0 then 0 else isqrtAux n n n

/-- The scheduler's reported critical path: longest-chain mean plus an
    upper-quantile variance contribution `k · isqrt(totalLatencyVar)`.
    Matches the makespan report in `cgplan_set_makespan` /
    `cgplan_priority_quantile`. -/
def schedulerCritical (g : DAG) (k : Nat) : Nat :=
  topologicalLongestChainMean g + k * isqrt (totalLatencyVar g)

/-- **Theorem.** The scheduler upper-bounds the deterministic longest-chain
    mean of the DAG.  Adding `k · isqrt(var)` is non-decreasing because
    it is a sum of `Nat`s. -/
theorem reported_critical_path_geq_topological_max
    (g : DAG) (k : Nat) :
    schedulerCritical g k ≥ topologicalLongestChainMean g := by
  unfold schedulerCritical
  exact Nat.le_add_right _ _

/-- **Corollary.** The reported critical path is monotone in the quantile
    coefficient `k`: more conservative variance contribution can never
    shrink the reported makespan. -/
theorem schedulerCritical_monotone_in_k
    (g : DAG) {k₁ k₂ : Nat} (h : k₁ ≤ k₂) :
    schedulerCritical g k₁ ≤ schedulerCritical g k₂ := by
  unfold schedulerCritical
  exact Nat.add_le_add_left (Nat.mul_le_mul_right _ h) _

/-- **Corollary.** When variance is zero (deterministic latencies), the
    scheduler reports exactly the longest-chain mean. -/
theorem schedulerCritical_eq_when_variance_zero
    (g : DAG) (k : Nat) (h : totalLatencyVar g = 0) :
    schedulerCritical g k = topologicalLongestChainMean g := by
  unfold schedulerCritical
  rw [h]
  -- isqrt 0 = 0 by the unfolding of `isqrt`
  show topologicalLongestChainMean g + k * isqrt 0 = topologicalLongestChainMean g
  have h_isqrt0 : isqrt 0 = 0 := by
    unfold isqrt; rfl
  rw [h_isqrt0, Nat.mul_zero, Nat.add_zero]

end Sounio.Scheduler.CriticalPath
