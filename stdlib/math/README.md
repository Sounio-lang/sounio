# Math

## Overview

Advanced mathematics: Cayley-Dickson algebras (quaternion → octonion → sedenion), Lie groups, finite element methods, FFT, spectral methods, geometric algebra, and differential geometry.

## Epistemic Differentiators

- Uncertainty propagation in hypercomplex operations (quaternion, octonion)
- [`ga/epistemic`](./ga/epistemic.sio): Epistemic multivectors in geometric algebra
- Propagation through differential geometry and integral equations
- Stochastic calculus with uncertainty bounds

## Quickstart

```sio
use math::ga::quaternion::{quat_new, quat_mul, quat_rotate_vec3};

// Create quaternion for 90° rotation around Z-axis
let q = quat_new(0.7071, 0.0, 0.0, 0.7071);

// Rotate vector
let rotated = quat_rotate_vec3(q, 1.0, 0.0, 0.0);
```

## Benchmarks

See [`BENCHMARKS.md`](../../benchmarks/README.md) for performance data.

## Validation Status

See [`VALIDATION_REPORT.md`](../../benchmarks/stdlib_validation/VALIDATION_REPORT.md) for test coverage.

## Modules

| Module | Description |
|--------|-------------|
| [`ga/`](./ga/) | Geometric Algebra (Clifford) |
| [`ga/quaternion`](./ga/quaternion.sio) | Quaternions as Cl(0,2) |
| [`ga/epistemic`](./ga/epistemic.sio) | Epistemic multivectors |
| [`cayley_dickson`](./cayley_dickson.sio) | Cayley-Dickson construction |
| [`octonion`](./octonion.sio) | Octonion algebra |
| [`sedenion`](./sedenion.sio) | Sedenion algebra |
| [`sedenion64`](./sedenion64.sio) | 64-bit sedenions |
| [`lie`](./lie.sio) | Lie groups and algebras |
| [`fft`](./fft.sio) | Fast Fourier Transform |
| [`spectral`](./spectral.sio) | Spectral methods |
| [`fem`](./fem.sio) | Finite Element Method |
| [`diffgeo`](./diffgeo.sio) | Differential geometry |
| [`stochastic_calc`](./stochastic_calc.sio) | Stochastic calculus |
| [`functional`](./functional.sio) | Functional analysis |
| [`interpolation`](./interpolation.sio) | Interpolation methods |
| [`convex`](./convex.sio) | Convex optimization |
| [`number_theory`](./number_theory.sio) | Number theory |

## License

MIT / Apache-2.0 (same as Sounio)
