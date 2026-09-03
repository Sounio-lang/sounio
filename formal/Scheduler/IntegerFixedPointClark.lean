/-!
# Sounio.Scheduler.IntegerFixedPointClark

Pure Lean 4 stdlib.  No `Mathlib` imports.  No `sorry`.  No `Classical`
(axiom footprint matches the other three scheduler-invariant files).

Companion proof obligation for the Newton-style integer-square-root
iteration inside `cgplan_clark_fold` and `cgplan_priority_quantile`
(self-hosted/native/codegen_plan.sio):

```sounio
pub fn cgplan_isqrt(n: i64) -> i64 with Div, Mut {
    if n <= 0 { return 0 }
    var x: i64 = n
    var y: i64 = (x + 1) / 2
    while y < x {
        x = y
        y = (x + n / x) / 2
    }
    x
}
```

Every step of `cgplan_clark_fold` calls `cgplan_isqrt(sum_v * 1000000)`
to compute the fixed-point representation of `a × 1000`, and every
priority quantile in `cgplan_phase_b_quantize` calls
`cgplan_isqrt(variance)`.  Termination of the outer Clark iteration in
`cgplan_finalize_makespan_clark` therefore rests on termination of the
inner Newton iteration in integer arithmetic.

**Termination by monotonicity of the rounding error.**  In integer
arithmetic, the Newton step `(x + n/x)/2` either strictly decreases the
iterate or stops (the rounding error is monotone — it never inverts
the direction of progress).  Strict decrease on `Nat` is well-founded,
so the iteration reaches a fixed-point in at most `x` steps:

      newtonIter_reaches_fixed_point :
        ∀ (x fuel : Nat), fuel ≥ x →
          newtonIter n x fuel = 0 ∨ IsFixedPoint n (newtonIter n x fuel)

The "fuel = x" instance of this is what `cgplan_isqrt` and the
`CriticalPathLowerBound.isqrt` helper both use: they pass the input as
their fuel because the strict-decrease invariant guarantees the
iteration finishes inside that budget.

The bridge to `i64` overflow — that `n ≤ 2⁶³ − 1` keeps the iterate
inside the representable range throughout — is a separate operational
claim (paper-side, alongside the Hoare-classical split).
-/

namespace Sounio.Scheduler.IntegerFixedPoint

/-- The integer Newton step for `√n`: `(x + n/x)/2`.  Both `cgplan_isqrt`
    and `CriticalPathLowerBound.isqrt` use this exact recurrence. -/
def newtonStep (n x : Nat) : Nat := (x + n / x) / 2

/-- An iterate `x` is a Newton fixed-point for `n` when the step cannot
    strictly decrease it — i.e. the rounding-error monotonicity has
    arrested the iteration. -/
def IsFixedPoint (n x : Nat) : Prop := newtonStep n x ≥ x

/-- The Newton iteration over `Nat`, fuel-bounded for structural
    termination.  Mirrors the `isqrtAux` shape from
    `CriticalPathLowerBound`. -/
def newtonIter (n : Nat) : Nat → Nat → Nat
  | x, 0       => x
  | x, fuel+1 =>
    if x = 0 then 0
    else
      let y := newtonStep n x
      if y < x then newtonIter n y fuel else x

/-- Top-level integer square root: pass `n` as fuel — sufficient by the
    termination theorem below. -/
def isqrt (n : Nat) : Nat :=
  if n = 0 then 0 else newtonIter n n n

/-- **Theorem (IntegerFixedPointClark — termination by monotonicity).**
    Whenever the fuel exceeds the current iterate, the Newton iteration
    is guaranteed to either return `0` or terminate at a Newton
    fixed-point.  Induction on `fuel`; the strict-decrease invariant of
    the `if y < x then …` branch is exactly the "monotonicity of the
    rounding error" — the step cannot oscillate.

    The hypothesis `fuel ≥ x` captures the standing convention that
    fuel is passed equal to the input iterate.  Each recursive call's
    new iterate is strictly less than the current one, so `fuel − 1`
    still suffices. -/
theorem newtonIter_reaches_fixed_point
    (n : Nat) :
    ∀ (x fuel : Nat), fuel ≥ x →
      newtonIter n x fuel = 0 ∨ IsFixedPoint n (newtonIter n x fuel) := by
  intro x fuel hf
  induction fuel generalizing x with
  | zero =>
    -- fuel = 0 with `fuel ≥ x` forces `x = 0`.  Return value is `x = 0`.
    have hx : x = 0 := by omega
    subst hx
    left
    unfold newtonIter
    rfl
  | succ k ih =>
    -- Three cases: x = 0, strict-decrease, or fixed-point reached.
    unfold newtonIter
    if hx : x = 0 then
      rw [if_pos hx]
      left
      rfl
    else
      rw [if_neg hx]
      if hy : newtonStep n x < x then
        rw [if_pos hy]
        -- Recurse on the smaller iterate; fuel = k still suffices.
        have hk : k ≥ newtonStep n x := by omega
        exact ih (newtonStep n x) hk
      else
        rw [if_neg hy]
        -- Returns `x`, which is now a fixed-point by `¬(step < x)`.
        right
        show newtonStep n x ≥ x
        omega

/-- **Corollary.**  The top-level `isqrt` always lands on a Newton
    fixed-point (or `0`), bounded by `fuel = n`. -/
theorem isqrt_reaches_fixed_point (n : Nat) :
    isqrt n = 0 ∨ IsFixedPoint n (isqrt n) := by
  unfold isqrt
  if h : n = 0 then
    rw [if_pos h]
    left
    rfl
  else
    rw [if_neg h]
    exact newtonIter_reaches_fixed_point n n n (Nat.le_refl n)

end Sounio.Scheduler.IntegerFixedPoint
