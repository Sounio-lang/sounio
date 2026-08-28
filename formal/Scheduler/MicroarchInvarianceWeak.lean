/-!
# Sounio.Scheduler.MicroarchInvarianceWeak

Pure Lean 4 stdlib.  No `Mathlib` imports.  No `sorry`.  No `Classical`
(axiom footprint matches `CriticalPathLowerBound` / `PortMatchingSound`
/ `ClarkFoldDeterminism`).

Companion proof obligation for the abstention discipline of the
variance-aware priority comparator in `cgplan_global_pick_best_pp_n`
(self-hosted/native/codegen_plan.sio): the epistemic scheduler operates
on `(mean, variance)` distributions, not bare cycles, so any two
instructions whose `k·σ` latency intervals overlap should not be
reordered on priority alone — the scheduler must *abstain*.

**Weak invariance (abstention version).**  When two instructions have
overlapping latency intervals — i.e. `|μ_A − μ_B| ≤ k·σ_A + k·σ_B` —
the abstention scheduler preserves their program order:

      weak_invariance_overlap_preserves_order :
        overlap a b → scheduleTwo a b = (a, b)

The "weak" qualifier vs. the strong microarch invariance (every
instruction pair is strictly ordered by priority) reflects the
abstention rule: overlapping-interval pairs are *equivalent* under the
priority relation and therefore stable-sorted on their original index.

The complementary fact — when intervals are strictly apart, the
schedule reflects the strict order — falls out of the same definitional
machinery and is recorded as a corollary.

Integer overflow & 64-bit truncation are not modelled here (Sounio uses
`i64`; this file works over `Int`).  Bridging the bit-width gap is a
separate operational claim, alongside the Hoare-classical split for the
rest of the scheduler.
-/

namespace Sounio.Scheduler.MicroarchInvariance

/-- An instruction with epistemic latency: integer mean (in cycles),
    non-negative variance (in cycles²), and a program-order index used
    as the stable-sort tie-breaker. -/
structure Instr where
  mean    : Int
  variance : Nat
  origIdx : Nat
deriving Inhabited, Repr

/-- Newton-style integer square root over `Nat`.  Mirrors the structure
    of `cgplan_isqrt` and the analogous helper in `CriticalPathLowerBound`.
    Terminates by structural recursion on a fuel parameter equal to the
    input (converges in `O(log n)` so this is loose but cheap). -/
def isqrtAux (n : Nat) : Nat → Nat → Nat
  | x, 0 => x
  | x, fuel+1 =>
    if x = 0 then 0
    else
      let y := (x + n / x) / 2
      if y < x then isqrtAux n y fuel else x

def isqrt (n : Nat) : Nat :=
  if n = 0 then 0 else isqrtAux n n n

/-- The k coefficient for the latency confidence interval.  Matches
    `CGPLAN_QUANTILE_K` in the Sounio scheduler — a `2σ` band gives
    `~95%` interval under the normal approximation. -/
def K : Int := 2

/-- Half-width of an instruction's `k·σ` confidence interval, in cycles. -/
def halfWidth (i : Instr) : Int := K * (isqrt i.variance : Int)

/-- Two instructions' latency intervals overlap iff their means lie
    within the sum of their half-widths — i.e. neither interval is
    strictly above the other. -/
def overlap (a b : Instr) : Prop :=
  (a.mean - b.mean) ≤ (halfWidth a + halfWidth b)
  ∧ (b.mean - a.mean) ≤ (halfWidth a + halfWidth b)

/-- Abstention comparator: returns `.eq` when intervals overlap, `.gt`
    when `a`'s interval is strictly above `b`'s, `.lt` otherwise.
    Mirrors what an abstention-aware refinement of
    `cgplan_priority_quantile` would produce. -/
def abstainOrdering (a b : Instr) : Ordering :=
  let delta := a.mean - b.mean
  let combined := halfWidth a + halfWidth b
  if delta > combined then .gt
  else if (b.mean - a.mean) > combined then .lt
  else .eq

/-- Stable two-instruction abstention schedule: on `.eq` (intervals
    overlap), preserves the input order — the canonical "abstain"
    behaviour. -/
def scheduleTwo (a b : Instr) : Instr × Instr :=
  match abstainOrdering a b with
  | .gt => (a, b)
  | .lt => (b, a)
  | .eq => (a, b)

/-- **Theorem (MicroarchInvarianceWeak, abstention version).**
    Overlapping latency intervals imply that the abstention scheduler
    preserves the program order on the pair.  Direct consequence of the
    definitions: `overlap a b` rules out both strict-above branches of
    `abstainOrdering`, so the comparator returns `.eq`, and `scheduleTwo`
    leaves the input untouched. -/
theorem weak_invariance_overlap_preserves_order
    (a b : Instr) (h : overlap a b) :
    scheduleTwo a b = (a, b) := by
  obtain ⟨hab, hba⟩ := h
  -- Both strict-above guards in `abstainOrdering` are false.
  have h_not_gt : ¬(a.mean - b.mean > halfWidth a + halfWidth b) := by omega
  have h_not_lt : ¬(b.mean - a.mean > halfWidth a + halfWidth b) := by omega
  show (match abstainOrdering a b with
        | .gt => (a, b) | .lt => (b, a) | .eq => (a, b)) = (a, b)
  unfold abstainOrdering
  rw [if_neg h_not_gt, if_neg h_not_lt]

/-- **Corollary.**  The abstention comparator yields `.eq` on
    overlapping intervals.  This is the underlying mechanism the
    invariance proof relies on. -/
theorem abstainOrdering_eq_of_overlap
    (a b : Instr) (h : overlap a b) :
    abstainOrdering a b = .eq := by
  obtain ⟨hab, hba⟩ := h
  have h_not_gt : ¬(a.mean - b.mean > halfWidth a + halfWidth b) := by omega
  have h_not_lt : ¬(b.mean - a.mean > halfWidth a + halfWidth b) := by omega
  unfold abstainOrdering
  rw [if_neg h_not_gt, if_neg h_not_lt]

/-- **Corollary (strict-order contrapositive).**  When `a`'s interval is
    strictly above `b`'s — i.e. `a.mean − b.mean > halfWidth a + halfWidth b`
    — the schedule places `a` before `b`.  Confirms that abstention only
    fires when the comparator genuinely cannot tell the operands apart. -/
theorem schedule_respects_strict_above
    (a b : Instr) (h : a.mean - b.mean > halfWidth a + halfWidth b) :
    scheduleTwo a b = (a, b) := by
  show (match abstainOrdering a b with
        | .gt => (a, b) | .lt => (b, a) | .eq => (a, b)) = (a, b)
  unfold abstainOrdering
  rw [if_pos h]

end Sounio.Scheduler.MicroarchInvariance
