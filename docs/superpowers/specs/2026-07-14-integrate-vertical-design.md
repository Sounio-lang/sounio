<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-integrate-vertical-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-integrate-vertical-design
-->

# Design — Harden the integrate::epistemic_ode vertical

**Status:** approved · **Date:** 2026-07-14 · **Constraint:** no compiler changes.
**Coordination:** `integrate/` is cold (stable, no open PR) — disjoint from active stats/IR/special lanes.
This lane **edits no source** — it adds an independent run-proof + gate + docs. EN-UK.

## 1. Why
Eighth playbook application (GUM #860, units #873, linalg #892, prob #902, stats #909, FFT #917).
`integrate::epistemic_ode` is a self-contained, epistemic-aware ODE solver (GUM through-line) that
native-compiles under the default Madaros engine.

## 2. Verified starting state
- `stdlib/integrate/epistemic_ode.sio` — GUM-uncertainty ODE solver (`epistemic_rk4_step`,
  `epistemic_euler_solve`, `epistemic_solve_decay`); `EpistemicState { pub values[16], pub uncertainties[16], pub n }`.
  Self-contained, green; returns `EpistemicState` by value across import correctly (verified — the #916
  fixed-array-value-semantics fix applies).
- **Runs under default Madaros** (verified): dy/dt=−ky, y₀=1,k=1 → y(1)→e⁻¹ as steps grow.
- **Method note (honest):** `epistemic_solve_decay` feeds a fixed per-step derivative to `rk4_step`, so
  the decay path is **first-order (Euler-equivalent)**: y(1) error 9.2e-4 (n=200) → 9.2e-5 (n=2000), O(1/n).
  Uncertainty uses a **step-wise additive GUM model** (u(y_{n+1})²=u(y_n)²+(dt·u(f))²) → u stays ≈u₀
  rather than the analytic correlated e^{−kt}·u₀. Both are properties of the existing code, asserted as-is.

## 3. Goal
An independent run-proof asserts the solver converges to the closed-form exponential-decay solution and
propagates a stable uncertainty, gated under the default engine — no source edits.

## 4. Scope
### In: run-proof driver (convergence to e⁻¹, half-life, decay monotonicity, uncertainty stability),
consumer example (first-order elimination report), gate, math-review.
### Out: no source edits; no new solver; no compiler edits.

## 5. Design — assertions (closed form y(t)=y₀e^{−kt})
- **Convergence:** |y(1;n=2000) − e⁻¹| < |y(1;n=200) − e⁻¹|, and < 2e-4.
- **Half-life:** y(ln2; n=2000) = 0.5 ± 2e-4.
- **Decay:** y(1) < y₀.
- **Value:** y(2) for y₀=2,k=0.5 = 2e⁻¹ = 0.735759 ± 5e-4.
- **Uncertainty:** for u₀=0.05, u stays in [0.049, 0.051] (step-wise additive model), > 0.
All inline in `main`; read `EpistemicState.values[0]`/`.uncertainties[0]`.

## 6. Layout
```
tests/stdlib/integrate/test_integrate_stdlib.sio  (new)
examples/integrate/decay_report.sio               (new)
scripts/integrate_gate.sh                          (new)
```

## 7. Verification
`souc check` green; `souc compile … && ./elf` (default Madaros) driver+example; gate → `INTEGRATE_GATE_OK`;
math-review logged.

## 8. Success criteria
Run-proof asserts convergence + known decay values + uncertainty stability and passes under default Madaros;
gate green; no source/compiler files touched (disjoint).
