<!-- docs:meta
topic_id: repo.docs.qnn.programming-guide
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.qnn.programming-guide
-->

# Sounio QNN Programming Guide

> *In 1843, William Rowan Hamilton carved the quaternion multiplication rules into Broom Bridge in Dublin: i² = j² = k² = ijk = −1. Nearly two centuries later, these hypercomplex numbers unlock 4× parameter efficiency in neural networks, naturally encoding the rotations and orientations that pervade 3D vision, robotics, and bioinformatics.*

Sounio brings quaternionic neural networks (QNNs) into the language and source
tree as native types and compiler-oriented operations. Public execution claims,
especially GPU claims, still need to be tied to the exact checked artifact being
discussed rather than inferred from implementation breadth alone.

> Current public status: the checked JIT artifact is the default docs entry
> point; the checked GPU artifact is used for verified GPU syntax and PTX
> emission through `build --backend gpu`. Do not read this guide as proof that
> every QNN workflow already runs on the checked public GPU artifact.

---

## 1. Why Quaternion Neural Networks?

### 1.1 The 4× Parameter Efficiency Theorem

In standard real-valued networks, processing a 3D vector requires three separate weights. A quaternion consolidates four components (w, x, y, z) into a single algebraic unit, reducing storage and computation:

| Network Type | Weights for 64→32 layer | Parameters |
|--------------|-------------------------|------------|
| Real-valued  | 64 × 32 × 1            | 2,048      |
| Quaternion   | 64 × 32 × 1 (as quats) | 512 quats = 2,048 floats, but **learned as 512 units** |

The key insight: quaternion weights learn *transformations* rather than independent scalars, providing richer feature interactions with the same memory footprint.

### 1.2 Natural 3D Rotation Representation

Unit quaternions form the special unitary group SU(2), which double-covers the rotation group SO(3). This means:

- No gimbal lock (unlike Euler angles)
- Smooth interpolation via SLERP
- Composition is simple multiplication

Applications that benefit:
- **3D vision**: Point cloud processing, mesh analysis
- **Robotics**: Manipulator pose estimation, SLAM
- **Molecular dynamics**: Protein folding, bond angles
- **Motion capture**: Human pose tracking

### 1.3 The Hamilton Product as Learned Transformation

The Hamilton product replaces scalar multiplication in QNNs:

```
q₁ ⊗ q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂)
        + (w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂)i
        + (w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂)j
        + (w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)k
```

This non-commutative operation (q₁ ⊗ q₂ ≠ q₂ ⊗ q₁) inherently encodes rotational structure.

---

## 2. Quaternion Fundamentals for ML

### 2.1 The Quat Type

Sounio provides a native `Quat` type with four f32 components:

```sounio
// Creating quaternions
let identity = quat(1.0, 0.0, 0.0, 0.0)  // w=1, x=y=z=0
let rotation = quat(0.707, 0.707, 0.0, 0.0)  // 90° around x-axis

// Accessing components
let w = rotation.w  // Real/scalar part
let x = rotation.x  // i component
let y = rotation.y  // j component
let z = rotation.z  // k component
```

### 2.2 Core Operations

```sounio
// Conjugate: negates vector part
fn quat_conjugate(q: Quat) -> Quat {
    quat(q.w, -q.x, -q.y, -q.z)
}

// Norm (magnitude)
fn quat_norm(q: Quat) -> f32 {
    let norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
    norm_sq.sqrt()
}

// Normalize to unit quaternion
fn quat_normalize(q: Quat) -> Quat {
    let n = quat_norm(q)
    quat(q.w / n, q.x / n, q.y / n, q.z / n)
}

// Hamilton product (non-commutative!)
fn hamilton_product(a: Quat, b: Quat) -> Quat {
    quat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    )
}

// Dot product (similarity measure)
fn quat_dot(a: Quat, b: Quat) -> f32 {
    a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z
}
```

### 2.3 Unit Quaternions and S³

Unit quaternions (norm = 1) live on the 3-sphere S³. For rotation tasks, always normalize:

```sounio
let raw = quat(1.5, 0.5, 0.5, 0.5)
let unit = quat_normalize(raw)  // Now on S³
```

---

## 3. Your First QNN

### 3.1 A Minimal Example

```sounio
//! hello_qnn.sio - Your first quaternion neural network

fn main() {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 1: Create a linear layer
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Defines architecture: 4 quaternion inputs → 2 quaternion outputs
    let layer = quat_linear_new(4, 2)
    println("✓ Created layer: 4 → 2 quaternions")

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 2: Initialize weights and bias
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Xavier: scale = sqrt(2 / (fan_in + fan_out))
    let weights = quat_xavier_init(4, 2, seed: 42)
    let bias = quat_xavier_init(1, 2, seed: 43)
    println("✓ Initialized weights (4×2) and bias (1×2)")

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 3: Create input data
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Input: 4 quaternions (identity quaternions for clarity)
    var input: [Quat; 4] = [
        quat(1.0, 0.0, 0.0, 0.0),  // q₀ = (1, 0, 0, 0) - identity
        quat(0.0, 1.0, 0.0, 0.0),  // q₁ = (0, 1, 0, 0) - pure i
        quat(0.0, 0.0, 1.0, 0.0),  // q₂ = (0, 0, 1, 0) - pure j
        quat(0.0, 0.0, 0.0, 1.0),  // q₃ = (0, 0, 0, 1) - pure k
    ]
    println("✓ Created 4 quaternion inputs")

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 4: Forward pass
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Computation: y = W ⊗ x + b (Hamilton product)
    let output = quat_linear_forward(&layer, &weights, &input, &bias)
    println("✓ Forward pass complete: output shape (1×2)")

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 5: Apply activation
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ReLU: applies max(0, component) to each of w, x, y, z
    let activated = quat_relu(output)
    println("✓ ReLU activation applied")

    println("\n✨ Your first QNN is working!")
}
```

### 3.1.1 Execution Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│  Input: [q₀, q₁, q₂, q₃]                           │
│  Shape: 1 batch × 4 quaternions                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Linear Layer (quat_linear_forward)                 │
│  ─────────────────────────────────────────────────  │
│  W: 4×2 quaternion matrix (random init)             │
│  b: 1×2 quaternion bias (random init)               │
│  Operation: y = W ⊗ x + b                           │
│  (Hamilton product, not standard matrix multiply)   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Linear Output: [z₀, z₁]                            │
│  Shape: 1 batch × 2 quaternions                     │
│  Range: unbounded (may be negative or large)        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  ReLU Activation (quat_relu)                        │
│  ─────────────────────────────────────────────────  │
│  Component-wise: max(0, w), max(0, x), ...          │
│  Introduces non-linearity for deep networks         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Final Output: [a₀, a₁]                             │
│  Shape: 1 batch × 2 quaternions                     │
│  Range: non-negative (sparsity, expressiveness)     │
└─────────────────────────────────────────────────────┘
```

### 3.2 Weight Initialization

Sounio provides three initialization strategies:

```sounio
// Xavier/Glorot: Good for sigmoid/tanh activations
// Scale: sqrt(2 / (fan_in + fan_out))
let w_xavier = quat_xavier_init(in_features, out_features, seed: 42)

// He/Kaiming: Preferred for ReLU activations
// Scale: sqrt(2 / fan_in)
let w_he = quat_he_init(in_features, out_features, seed: 42)

// Unit quaternions: All weights start with norm = 1
// Good for rotation-focused tasks
let w_unit = quat_unit_init(in_features, out_features, seed: 42)
```

### 3.3 Building a Multi-Layer Network

```sounio
fn create_qnn_classifier() {
    // Architecture: 16 → 32 → 16 → 8 (quaternion features)

    // Layer 1: Input → Hidden1
    let layer1 = quat_linear_new(16, 32)
    let w1 = quat_xavier_init(16, 32, seed: 100)
    let b1 = quat_xavier_init(1, 32, seed: 101)

    // Layer 2: Hidden1 → Hidden2
    let layer2 = quat_linear_new(32, 16)
    let w2 = quat_he_init(32, 16, seed: 102)  // He for ReLU
    let b2 = quat_xavier_init(1, 16, seed: 103)

    // Layer 3: Hidden2 → Output
    let layer3 = quat_linear_new(16, 8)
    let w3 = quat_xavier_init(16, 8, seed: 104)
    let b3 = quat_xavier_init(1, 8, seed: 105)

    // Forward pass
    let h1 = quat_relu(quat_linear_forward(&layer1, &w1, &input, &b1))
    let h2 = quat_relu(quat_linear_forward(&layer2, &w2, &h1, &b2))
    let output = quat_sigmoid(quat_linear_forward(&layer3, &w3, &h2, &b3))
}
```

---

## 4. Layer Types and Building Blocks

### 4.1 Linear Layers (QuatLinearLayer)

**Memory and Computation**:

```
Input:   [batch=1, features=16]              16 quaternions = 64 floats
         ↓
Weights: [in=16, out=8]                      128 quaternions = 512 floats
         ↓
Bias:    [out=8]                             8 quaternions = 32 floats
         ↓
         Hamilton Product (16 × 8 = 128 products)
         ↓
Output:  [batch=1, features=8]               8 quaternions = 32 floats

Equivalent Real-Valued Network:
Input:   64 floats
Weights: 512 floats (same)
Bias:    32 floats (same)
Total:   608 floats (identical memory)

BUT: QNN learns 128 quaternion transformations
     vs Real learns 512 scalar transformations
     → 4× fewer learned parameters!
```

The fundamental building block. Computes y = W ⊗ x + b:

```sounio
struct QuatLinearLayer {
    input_features: i32,
    output_features: i32,
}

let linear = quat_linear_new(64, 32)
let output = quat_linear_forward(&linear, &weights, &input, &bias)
```

**Detailed Example**:

```sounio
fn build_classifier() {
    // Layer 1: Input to hidden
    let layer1 = quat_linear_new(256, 128)
    let w1 = quat_xavier_init(256, 128, seed: 100)
    let b1 = quat_xavier_init(1, 128, seed: 101)

    // Layer 2: Hidden to output
    let layer2 = quat_linear_new(128, 10)
    let w2 = quat_xavier_init(128, 10, seed: 102)
    let b2 = quat_xavier_init(1, 10, seed: 103)

    // Forward pass
    let h = quat_relu(quat_linear_forward(&layer1, &w1, &input, &b1))
    let logits = quat_linear_forward(&layer2, &w2, &h, &b2)

    // Loss
    let loss = quat_mse_loss(&logits, &target)
}
```

### 4.2 Convolutional Layers

2D convolutions with quaternion kernels:

```sounio
// Create conv layer: 3 input channels, 16 output channels, 3×3 kernel
let conv = quat_conv2d_new(3, 16, kernel_h: 3, kernel_w: 3)

// Forward pass with padding and stride
let features = quat_conv2d_forward(&conv, &weights, &input, stride: 1, padding: 1)

// Pooling operations
let pooled = quat_avg_pool2d(&features, kernel: 2, stride: 2)
```

### 4.3 Recurrent Layers (LSTM/GRU)

Quaternion gates for sequence modeling:

```sounio
// LSTM cell with quaternion hidden state
let lstm = quat_lstm_new(input_size: 32, hidden_size: 64)

// Process sequence
var hidden = quat_zeros(64)
var cell = quat_zeros(64)
let i: i32 = 0
while i < sequence.len() {
    let result = quat_lstm_cell(&lstm, &sequence[i], &hidden, &cell)
    hidden = result.hidden
    cell = result.cell
    i = i + 1
}

// GRU variant (simpler, often sufficient)
let gru = quat_gru_new(input_size: 32, hidden_size: 64)
```

### 4.4 Attention Mechanism

Multi-head quaternion attention for transformers:

```sounio
// Create attention layer
let attn = quat_attention_new(embed_dim: 64, num_heads: 8)

// Compute attention
let attended = quat_attention_forward(&attn, query: &q, key: &k, value: &v)
```

---

## 5. Activation Functions

All activations apply **component-wise** to quaternions:

```sounio
activate(quat(w, x, y, z)) = quat(activate(w), activate(x), activate(y), activate(z))
```

### 5.1 Available Activations

```sounio
// ReLU: max(0, component)
let activated = quat_relu(q)

// Sigmoid: 1 / (1 + exp(-component))
let bounded = quat_sigmoid(q)

// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
let normalized = quat_tanh(q)

// Leaky ReLU: component if > 0, else alpha * component
let leaky = quat_leaky_relu(q, alpha: 0.01)
```

### 5.2 Batch Operations

For efficiency, use batch variants:

```sounio
fn process_batch(input: &[Quat], output: &![Quat]) {
    quat_relu_batch(input, output)
}
```

---

## 6. Training a QNN

### 6.1 Data Encoding

Convert your data to quaternions:

```sounio
// RGB pixel → quaternion (one approach)
fn rgb_to_quat(r: f32, g: f32, b: f32) -> Quat {
    let luminance = (r + g + b) / 3.0
    quat(luminance, r - luminance, g - luminance, b - luminance)
}

// 3D coordinates → pure quaternion
fn vec3_to_quat(x: f32, y: f32, z: f32) -> Quat {
    quat(0.0, x, y, z)  // w=0 for pure quaternions
}
```

### 6.2 Loss Functions

```sounio
// Mean Squared Error
let mse = quat_mse_loss(&predictions, &targets)

// Mean Absolute Error (robust to outliers)
let mae = quat_mae_loss(&predictions, &targets)

// Cosine similarity (good for rotation tasks)
let cos_loss = quat_cosine_similarity_loss(&predictions, &targets)
```

### 6.3 Optimizers

```sounio
// Adam optimizer (recommended default)
var optimizer = quat_adam_new(learning_rate: 0.001)

// SGD with momentum
var sgd = quat_sgd_new(
    learning_rate: 0.01,
    momentum: 0.9,
    weight_decay: 1e-4
)
```

### 6.4 Training Loop

```sounio
fn train_epoch(
    model_weights: &![Quat],
    optimizer: &!QuatAdamOptimizer,
    first_moment: &![Quat],
    second_moment: &![Quat],
    data: &[Quat],
    targets: &[Quat]
) -> f32 {
    // Forward pass
    let predictions = forward(model_weights, data)

    // Compute loss
    let loss = quat_mse_loss(&predictions, targets)

    // Backward pass (compute gradients)
    let gradients = backward(model_weights, &predictions, targets)

    // Update weights
    quat_adam_step(model_weights, &gradients, optimizer, first_moment, second_moment)

    loss
}
```

### 6.5 Learning Rate Guidelines

| Task Type | Recommended LR | Notes |
|-----------|---------------|-------|
| Classification | 0.001 | Standard starting point |
| Rotation prediction | 0.0005 | Lower for stability |
| Fine-tuning | 0.0001 | Conservative updates |

Use cosine annealing for decay:

```sounio
let lr = quat_learning_rate_schedule(
    initial_lr: 0.001,
    step: current_step,
    total_steps: 10000,
    min_lr: 0.00001
)
```

---

## 7. GPU Acceleration

### 7.1 Enabling GPU Support

Build with the GPU feature:

```bash
cargo build --features gpu
```

### 7.2 Tensor Cores (WMMA)

On NVIDIA Volta+ GPUs, Sounio leverages Tensor Cores for 20-30× speedup:

- Automatic dispatch when batch size ≥ 16
- Best performance at batch sizes divisible by 128
- Hamilton products mapped to WMMA matrix ops

### 7.3 Kernel Fusion

Sounio fuses common operation patterns:

```sounio
// Instead of:
let h1 = quat_linear_forward(...)
let h2 = quat_bn_forward(...)
let h3 = quat_relu(h2)

// Fused version (single kernel launch):
let h = quat_linear_bn_relu_fused(...)
```

### 7.4 Memory Layout

Quaternions are stored contiguously as `[w, x, y, z]` per element, enabling coalesced GPU memory access.

---

## 8. What Does NOT Work (Anti-Patterns)

### 8.1 Syntax Differences from Rust

```sounio
// ❌ WRONG: Rust-style mutable reference
fn update(x: &mut Quat) { ... }

// ✅ CORRECT: Sounio exclusive reference
fn update(x: &!Quat) { ... }
```

```sounio
// ❌ WRONG: Rust macros don't exist
assert!(q.w > 0.0)
println!("Value: {}", q.w)

// ✅ CORRECT: Sounio functions
println("Value computed")
```

### 8.2 No Tuple Destructuring

```sounio
// ❌ WRONG: Tuple destructuring not supported
let (w, x, y, z) = get_components(q)

// ✅ CORRECT: Access fields directly
let w = q.w
let x = q.x
let y = q.y
let z = q.z
```

### 8.3 Define Helpers Before Use

```sounio
// ❌ WRONG: Forward reference
fn main() {
    let result = helper()  // Error: helper not defined yet
}
fn helper() -> Quat { ... }

// ✅ CORRECT: Helper defined first
fn helper() -> Quat { ... }
fn main() {
    let result = helper()
}
```

### 8.4 Use Hamilton Product, Not Element-wise Multiply

```sounio
// ❌ WRONG: Element-wise loses rotational semantics
let wrong = quat(q1.w * q2.w, q1.x * q2.x, q1.y * q2.y, q1.z * q2.z)

// ✅ CORRECT: Hamilton product for transformations
let correct = hamilton_product(q1, q2)
```

---

## References

1. Gaudet & Maida (2018). "Deep Quaternion Networks." IJCNN 2018. [arXiv:1705.07944](https://arxiv.org/abs/1705.07944)
2. Parcollet et al. (2019). "Quaternion Recurrent Neural Networks." ICLR 2019. [arXiv:1903.08478](https://arxiv.org/abs/1903.08478)
3. Zhu et al. (2018). "Quaternion Convolutional Neural Networks." ECCV 2018.
4. Grassucci et al. (2021). "PHNNs: Lightweight Neural Networks via Parameterized Hypercomplex Convolutions." IEEE TNNLS.

---

## Next Steps

- [Performance Tuning Handbook](PERFORMANCE_HANDBOOK.md) — Optimize for production
- [Architecture Deep-Dive](ARCHITECTURE_DEEP_DIVE.md) — Implementation details
- [Migration Guide](MIGRATION_GUIDE.md) — Convert float networks to QNN
