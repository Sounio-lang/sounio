# Autodiff

## Overview

Automatic differentiation: tape-based reverse-mode, dual numbers for forward-mode, linear AD, and epistemic dual numbers for uncertainty-aware gradients.

## Epistemic Differentiators

- [`EpistemicDual`](./epistemic_dual.sio) combines AD with uncertainty propagation
- Gradient provenance and confidence tracking
- [`Tape`](./tape.sio) with uncertainty accumulation through backward pass
- Linear AD for efficient Jacobian-vector products

## Quickstart

`stdlib/autodiff/tape.sio` is free functions over a by-value `Tape`, not
`Tape::new()` / `push_var()` / `grad()` methods. Each operation returns the
updated tape. **These names are internal to `tape.sio` (none are exported):**
the sketch below is implementation-internal pseudocode, not a public API to
import from user code:

```sio
var tape = new_tape()
tape = tape_new_var(tape, 3.0)
let x = tape_last_var(tape)
tape = tape_mul(tape, x, x)   // y = x^2
let y = tape_last_var(tape)
tape = backward(tape, y)

let dx = get_grad(tape, x)    // dy/dx = 2x = 6
```

These names — `new_tape`, `tape_new_var`, `tape_last_var`, `tape_mul`,
`backward`, `get_grad`, and the `Var` type — are private to `tape.sio` and
cannot be imported. The tape module is internal; `epistemic_dual`
(e.g. `edual_new` / `edual_mul`) and `grad` (the public `Dual` type) expose the
public AD surface.

More in [`TAPE_IMPLEMENTATION.md`](./TAPE_IMPLEMENTATION.md). For uncertainty-aware gradients, `stdlib/autodiff/epistemic_dual.sio` builds values with `edual_new(val, dot, unc, unc_dot)` and `edual_mul` — there is no `EpistemicDual::new` and no `Knowledge::measured`.

## Benchmarks

See [`BENCHMARKS.md`](../../benchmarks/README.md) for performance data.

## Validation Status

See [`VALIDATION_REPORT.md`](../../benchmarks/stdlib_validation/VALIDATION_REPORT.md) for test coverage.

## Modules

| Module | Description |
|--------|-------------|
| [`tape`](./tape.sio) | Reverse-mode AD via Wengert tape |
| [`dual`](./dual.sio) | Forward-mode AD with dual numbers |
| [`epistemic_dual`](./epistemic_dual.sio) | Dual numbers with uncertainty |
| [`linear_ad`](./linear_ad.sio) | Linear AD for Jacobians |
| [`grad`](./grad.sio) | High-level gradient API |
| [`differentiable`](./differentiable.sio) | Differentiable function traits |

## Tape Implementation Details

The tape-based reverse-mode AD follows:

1. **Forward pass**: Record operations on tape with values
2. **Backward pass**: Traverse tape in reverse, accumulating adjoints

Supported operations:
- Add, Sub, Mul, Div
- Neg, Sqrt, Exp, Ln
- Sin, Cos, Tanh
- Pow, ReLU, Sigmoid

## License

MIT / Apache-2.0 (same as Sounio)
