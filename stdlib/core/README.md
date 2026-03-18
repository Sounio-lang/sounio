# Core

## Overview

The `core` module provides foundational data structures and utilities essential for all Sounio programs: `Option`, `Result`, prelude imports, and basic error handling.

## Epistemic Differentiators

- `Option<Knowledge<T>>` enables optional uncertain computations with automatic propagation.
- `Result<T, E>` with epistemic types tracks confidence degradation through error paths.
- Pattern matching preserves provenance and uncertainty bounds.

## Quickstart

```sio
use core::prelude::*;
use core::option::Option;
use core::result::Result;

// Basic utilities from prelude
let x = clamp_f64(3.14, 0.0, 1.0);  // 1.0
let y = abs_f64(-5.0);               // 5.0
```

## Benchmarks

See [`BENCHMARKS.md`](../../benchmarks/README.md) for performance data.

## Validation Status

See [`VALIDATION_REPORT.md`](../../benchmarks/stdlib_validation/VALIDATION_REPORT.md) for test coverage.

## Modules

| Module | Description |
|--------|-------------|
| [`prelude`](./prelude.sio) | Common utilities (abs, min, max, clamp) |
| [`option`](./option.sio) | Optional values |
| [`result`](./result.sio) | Error handling with Result<T, E> |

## License

MIT / Apache-2.0 (same as Sounio)
