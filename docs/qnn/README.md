<!-- docs:meta
topic_id: repo.docs.qnn.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.qnn.readme
-->

# Sounio QNN Documentation

Comprehensive documentation for Quaternion Neural Networks in Sounio.

## Quick Links

| Document | Description |
|----------|-------------|
| [Programming Guide](PROGRAMMING_GUIDE.md) | Tutorial-style introduction to QNNs |
| [Performance Handbook](PERFORMANCE_HANDBOOK.md) | Optimization techniques for CPU/GPU |
| [Architecture Deep-Dive](ARCHITECTURE_DEEP_DIVE.md) | Internal implementation details |
| [Migration Guide](MIGRATION_GUIDE.md) | Converting float networks to QNN |

## Getting Started

```sounio
// Create a quaternion linear layer
let layer = quat_linear_new(64, 32)
let weights = quat_xavier_init(64, 32, seed: 42)
let bias = quat_xavier_init(1, 32, seed: 43)

// Forward pass with ReLU activation
let output = quat_relu(quat_linear_forward(&layer, &weights, &input, &bias))
```

## Key Features

- **4× parameter efficiency** through quaternion algebra
- **Native SIMD optimization** (AVX2, AVX-512, NEON)
- **GPU acceleration** via Tensor Cores (WMMA)
- **INT8 quantization** for deployment

## Examples

See `/examples/qnn/` for working demonstrations:

- `01_hello_quaternion.sio` — Simplest introduction
- `02_basic_linear.sio` — Single layer demo

## References

- Gaudet & Maida (2018). "Deep Quaternion Networks." [arXiv:1705.07944](https://arxiv.org/abs/1705.07944)
- Parcollet et al. (2019). "Quaternion Recurrent Neural Networks." [arXiv:1903.08478](https://arxiv.org/abs/1903.08478)
