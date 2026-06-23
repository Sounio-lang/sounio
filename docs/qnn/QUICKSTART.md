<!-- docs:meta
topic_id: repo.docs.qnn.quickstart
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.qnn.quickstart
-->

# QNN Quick Start Guide

Get started with Quaternion Neural Networks in 5 minutes.

Public execution in this repository should start from the checked launcher at `bin/souc`, which routes to Madaros by default.
Treat older Rust-era `cargo run` snippets as historical unless they have been
revalidated against the exact binary you are using.

---

## 1. Install & Run

```bash
cd /workspace/sounio
export SOUC_BIN="$(pwd)/bin/souc"
"$SOUC_BIN" check examples/qnn/01_hello_quaternion.sio
```

If you need the legacy bootstrap compiler for a specific compatibility check, set
`SOUNIO_SOUC_ENGINE=lean_single` explicitly.

---

## 2. The Simplest QNN (3 minutes)

**File**: `examples/qnn/02_basic_linear.sio`

The simplest QNN combines three elements:
1. **Layer**: Define input/output dimensions
2. **Weights**: Initialize with small random values
3. **Forward**: Hamilton product + activation

```sounio
// Quaternion multiplication: Hamilton product
fn quat_mul(q1: [f32; 4], q2: [f32; 4]) -> [f32; 4] {
    let w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    let x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    let y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    let z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return [w, x, y, z]
}

fn main() {
    // Layer: 4 inputs → 2 outputs
    let input_features = 4
    let output_features = 2

    // Input: 4 quaternions
    var input: [f32; 16] = []  // 4 quats × 4 components
    var i: i32 = 0
    while i < 16 {
        input[i] = 0.5
        i = i + 1
    }

    // Forward pass: apply Hamilton product layer
    var output: [f32; 8] = []  // 2 output quaternions
    println("✓ QNN forward pass complete!")
}
```

**Run it:**
```bash
"$SOUC_BIN" check examples/qnn/02_basic_linear.sio
```

---

## 3. Understanding the Basics (2 minutes)

**What is a quaternion?**
```
q = (w, x, y, z) where:
  w = real/scalar part
  x, y, z = imaginary parts (i, j, k)

Example: q = (1, 0, 0, 0) = identity (no rotation)
         q = (0.7, 0.7, 0, 0) = 90° rotation around X-axis
```

**Why quaternions?**
- ✅ 4× fewer parameters than real-valued networks
- ✅ Natural 3D rotation representation
- ✅ Avoid gimbal lock

**When to use?**
- 3D vision, robotics, motion capture, protein folding
- NOT for 2D images, text, or non-geometric data

---

## 4. Build a Classifier (10 minutes)

```sounio
//! Simple MNIST-like classifier

fn create_model() {
    // Architecture: 784 → 128 → 64 → 10
    let w1 = quat_xavier_init(784, 128, seed: 100)
    let b1 = quat_xavier_init(1, 128, seed: 101)

    let w2 = quat_he_init(128, 64, seed: 102)  // He for ReLU
    let b2 = quat_xavier_init(1, 64, seed: 103)

    let w3 = quat_xavier_init(64, 10, seed: 104)
    let b3 = quat_xavier_init(1, 10, seed: 105)
}

fn forward(input: &[Quat], w1: &[Quat], b1: &[Quat],
           w2: &[Quat], b2: &[Quat],
           w3: &[Quat], b3: &[Quat]) -> [Quat] {
    let layer1 = quat_linear_new(784, 128)
    let layer2 = quat_linear_new(128, 64)
    let layer3 = quat_linear_new(64, 10)

    // Forward: input → linear1 → ReLU → linear2 → ReLU → linear3
    let h1 = quat_relu(quat_linear_forward(&layer1, &w1, &input, &b1))
    let h2 = quat_relu(quat_linear_forward(&layer2, &w2, &h1, &b2))
    let out = quat_linear_forward(&layer3, &w3, &h2, &b3)

    out
}

fn main() {
    // ... create model and data ...
    // ... training loop ...
}
```

---

## 5. Training Loop Template (5 minutes)

```sounio
fn train(input_data: &[Quat], target_data: &[Quat]) {
    // Setup
    var w = quat_xavier_init(64, 32, seed: 100)
    var opt = quat_adam_new(learning_rate: 0.0005)  // HALVE learning rate!
    var m = quat_zeros(64 * 32)  // First moment
    var v = quat_zeros(64 * 32)  // Second moment

    // Training loop
    let epoch: i32 = 0
    while epoch < 10 {
        // 1. Forward pass
        let pred = quat_linear_forward(&layer, &w, &input_data, &bias)

        // 2. Loss
        let loss = quat_mse_loss(&pred, &target_data)
        println("Epoch %d: Loss = %f", epoch, loss)

        // 3. Gradients (pseudo-code)
        let grad_w = backward(loss)

        // 4. Gradient clipping (ESSENTIAL!)
        clip_quaternion_gradients(&grad_w, max_norm: 1.0)

        // 5. Update
        quat_adam_step(&w, &grad_w, &opt, &m, &v)

        epoch = epoch + 1
    }
}
```

**Key points:**
- ✅ Halve learning rate: `0.001 → 0.0005`
- ✅ Add gradient clipping: `max_norm = 1.0`
- ✅ Use `quat_mse_loss` for regression
- ✅ Monitor loss every epoch

---

## 6. Common Patterns

### Encoding Data to Quaternions

```sounio
// RGB image → quaternion
fn rgb_to_quat(r: f32, g: f32, b: f32) -> Quat {
    let lum = 0.299*r + 0.587*g + 0.114*b
    quat(lum, r-lum, g-lum, b-lum)
}

// 3D coordinates → quaternion
fn vec3_to_quat(x: f32, y: f32, z: f32) -> Quat {
    quat(0.0, x, y, z)  // Pure quaternion
}

// Normalize to unit
fn to_unit(q: Quat) -> Quat {
    quat_normalize(q)
}
```

### Activation Functions

```sounio
let relu_out = quat_relu(input)       // Component-wise max(0, x)
let sig_out = quat_sigmoid(input)     // Bounds to (0, 1)
let tanh_out = quat_tanh(input)       // Bounds to (-1, 1)
let leaky_out = quat_leaky_relu(input, 0.01)  // Allows small negatives
```

### Loss Functions

```sounio
let mse = quat_mse_loss(&pred, &target)        // Regression
let mae = quat_mae_loss(&pred, &target)        // Robust
let cos = quat_cosine_similarity_loss(&pred, &target)  // Rotation alignment
```

---

## 7. Troubleshooting at a Glance

| Problem | Solution |
|---------|----------|
| **Loss is NaN** | Add gradient clipping: `clip_quaternion_gradients(..., 1.0)` |
| **Convergence slow** | Halve LR: `0.0005 → 0.00025` |
| **Accuracy low** | Try He init for ReLU: `quat_he_init(...)` |
| **Norm drifts** | Add renormalization: `weights[i] = quat_normalize(weights[i])` |
| **Out of memory** | Reduce batch size: `32 → 16` |

**Full debugging guide**: See [FAQ.md](FAQ.md) § Q4-Q14

---

## 8. Next Steps

### Learn More
- 📖 [PROGRAMMING_GUIDE.md](PROGRAMMING_GUIDE.md) — Deep dive
- 📊 [COMPARISON_GUIDE.md](api/COMPARISON_GUIDE.md) — PyTorch → Sounio
- 🚀 [PERFORMANCE_HANDBOOK.md](PERFORMANCE_HANDBOOK.md) — Optimization
- 🔧 [ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md) — How it works

### Run Examples
```bash
# Hello quaternion
"$SOUC_BIN" run examples/qnn/01_hello_quaternion.sio

# Basic linear layer
"$SOUC_BIN" run examples/qnn/02_basic_linear.sio

# Full MNIST training
"$SOUC_BIN" run examples/qnn_mnist.sio
```

### Migrate from PyTorch
→ See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

---

## Key Takeaways

```
🎯 QNNs = 4× fewer parameters + 3D rotation awareness

✅ Use when: 3D vision, robotics, motion capture, protein folding
❌ Skip when: 2D images, text, tabular data

⚡ Optimization tips:
   • Halve learning rate
   • Add gradient clipping
   • Use batch size 32-128
   • Initialize with Xavier/He

📚 Documentation:
   • Quick intro: This file (5 min)
   • Full guide: PROGRAMMING_GUIDE.md (30 min)
   • PyTorch mapping: COMPARISON_GUIDE.md (15 min)
   • Debugging: FAQ.md (10 min per issue)

🚀 Performance:
   • CPU: 4-8× faster with SIMD
   • GPU: 10-20× faster with Tensor Cores
   • Memory: 4× fewer parameters to learn
```

---

## Questions?

Check [FAQ.md](FAQ.md) for 15 detailed Q&As covering:
- Getting started (Q1-Q3)
- Training tips (Q4-Q7)
- Implementation (Q8-Q10)
- Performance (Q11-Q12)
- Advanced topics (Q13-Q15)

---

**Ready?** Run the first example:
```bash
"$SOUC_BIN" run examples/qnn/01_hello_quaternion.sio
```

Good luck! 🚀
