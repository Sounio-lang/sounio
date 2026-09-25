# Autodiff

## Overview

Automatic differentiation: tape-based reverse-mode, dual numbers for forward-mode, linear AD, and epistemic dual numbers for uncertainty-aware gradients.

## Epistemic Differentiators

- [`EpistemicDual`](./epistemic_dual.sio) combines AD with uncertainty propagation
- Gradient provenance and confidence tracking
- [`Tape`](./tape.sio) with uncertainty accumulation through backward pass
- Linear AD for efficient Jacobian-vector products

## Quickstart

```sio
use autodiff::tape::Tape;

// Reverse-mode AD with tape
let mut tape = Tape::new();
let x = tape.push_var(3.0);
let y = tape.push_mul(x, x);  // y = x²
tape.backward();              // Compute gradients

let dx = tape.grad(x);        // dy/dx = 2x = 6
```

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
