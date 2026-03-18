# NN

## Overview

Neural network building blocks: dense/quaternion layers, activations, optimizers, equivariant convolutions, and autograd integration.

## Epistemic Differentiators

- [`EpistemicLayer`](./epistemic_layer.sio) propagates uncertainty through forward/backward passes
- Knowledge-aware loss functions and gradients
- Quaternion and octonion layers for geometric deep learning
- G₂-equivariant hyperspectral convolutions

## Quickstart

```sio
use nn::dense::Dense;
use nn::activation::sigmoid;

// Create dense layer
let layer = Dense::new(2, 1);

// Forward pass
let input = [1.0, 0.5];
let output = layer.forward(input);
```

## Benchmarks

See [`BENCHMARKS.md`](../../benchmarks/README.md) for performance data.

## Validation Status

See [`VALIDATION_REPORT.md`](../../benchmarks/stdlib_validation/VALIDATION_REPORT.md) for test coverage.

## Modules

| Module | Description |
|--------|-------------|
| [`dense`](./dense.sio) | Fully connected layers |
| [`dense_quaternion`](./dense_quaternion.sio) | Quaternion-valued dense layers |
| [`epistemic_layer`](./epistemic_layer.sio) | Layers with uncertainty propagation |
| [`activation`](./activation.sio) | Activation functions (ReLU, sigmoid, tanh) |
| [`autograd`](./autograd.sio) | Automatic differentiation for NN |
| [`optimizers_quaternion`](./optimizers_quaternion.sio) | Quaternion optimizers (AdamQ, etc.) |
| [`g2_equivariant`](./g2_equivariant.sio) | G₂-equivariant layers |
| [`g2_hyperspectral_conv`](./g2_hyperspectral_conv.sio) | Hyperspectral convolutions |
| [`tensor`](./tensor.sio) | Tensor operations for NN |

## License

MIT / Apache-2.0 (same as Sounio)
