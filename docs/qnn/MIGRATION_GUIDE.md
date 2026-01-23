# Migration Guide: Float to Quaternion Networks

> *Converting real-valued neural networks to quaternion-based architectures—when it helps, when it doesn't, and how to do it right.*

This guide walks through migrating traditional float-based networks to Sounio's Quaternion Neural Networks (QNNs), covering data encoding, layer conversion, and optimization adjustments.

---

## 1. When to Migrate

### Use Cases That Benefit

QNNs excel when your data has inherent **3D rotational structure**:

| Domain | Why QNNs Help |
|--------|--------------|
| **3D Vision** | Point clouds, meshes, depth estimation—rotations are natural |
| **Robotics** | Pose estimation, SLAM, orientation tracking |
| **Molecular Dynamics** | Protein folding, bond angles, conformational search |
| **Motion Capture** | Skeletal animation, pose interpolation, IMU processing |
| **Satellite/Drone** | Attitude estimation, gimbal control |

### When to Stay Real-Valued

QNNs add complexity without benefit when:

- **2D Image Classification**: No inherent rotation semantics
- **NLP/Text**: Sequential, non-geometric structure
- **Tabular Data**: Features without rotational meaning
- **Audio Processing**: Unless dealing with 3D spatial audio

**Rule of Thumb**: If your data doesn't naturally involve rotations or 3D orientations, stay with real-valued networks.

---

## 2. Data Encoding Strategies

### 2.1 RGB Images → Quaternions

**Approach 1: Luminance + Color Differences**

```sounio
fn rgb_to_quat(r: f32, g: f32, b: f32) -> Quat {
    let luminance = 0.299 * r + 0.587 * g + 0.114 * b
    let r_diff = r - luminance
    let g_diff = g - luminance
    let b_diff = b - luminance
    quat(luminance, r_diff, g_diff, b_diff)
}
```

**Approach 2: Direct RGBA Mapping**

```sounio
fn rgba_to_quat(r: f32, g: f32, b: f32, a: f32) -> Quat {
    quat(a, r, g, b)  // Alpha as scalar, RGB as vector
}
```

### 2.2 3D Coordinates → Quaternions

**Pure Quaternion (Position)**

```sounio
fn position_to_quat(x: f32, y: f32, z: f32) -> Quat {
    quat(0.0, x, y, z)  // w=0 for pure quaternions
}
```

**Scaled Quaternion (Position + Magnitude)**

```sounio
fn position_scaled_to_quat(x: f32, y: f32, z: f32) -> Quat {
    let magnitude = (x*x + y*y + z*z).sqrt()
    if magnitude > 0.0 {
        quat(magnitude, x/magnitude, y/magnitude, z/magnitude)
    } else {
        quat(0.0, 0.0, 0.0, 0.0)
    }
}
```

### 2.3 Time Series / Trajectories

For sequential 3D data (IMU, motion capture):

```sounio
// Window of quaternion frames
fn encode_trajectory(positions: &[[f32; 3]], window: i32) -> [Quat] {
    var encoded: [Quat; window] = []
    let i: i32 = 0
    while i < window {
        let p = positions[i]
        encoded[i] = quat(0.0, p[0], p[1], p[2])
        i = i + 1
    }
    encoded
}
```

---

## 3. Layer Conversion Patterns

### 3.1 Linear Layers

**PyTorch**:
```python
self.fc = nn.Linear(64, 32)
```

**Sounio QNN**:
```sounio
let layer = quat_linear_new(64, 32)
let weights = quat_xavier_init(64, 32, seed: 42)
let bias = quat_xavier_init(1, 32, seed: 43)

// Forward pass
let output = quat_linear_forward(&layer, &weights, &input, &bias)
```

### 3.2 Convolutional Layers

**PyTorch**:
```python
self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

**Sounio QNN**:
```sounio
let conv = quat_conv2d_new(3, 16, kernel_h: 3, kernel_w: 3)
let output = quat_conv2d_forward(&conv, &weights, &input, stride: 1, padding: 1)
```

### 3.3 Recurrent Layers (LSTM)

**PyTorch**:
```python
self.lstm = nn.LSTM(input_size=32, hidden_size=64)
```

**Sounio QNN**:
```sounio
let lstm = quat_lstm_new(input_size: 32, hidden_size: 64)

// Process sequence
var hidden = quat_zeros(64)
var cell = quat_zeros(64)
let result = quat_lstm_cell(&lstm, &input, &hidden, &cell)
```

### 3.4 Attention Layers

**PyTorch**:
```python
self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=8)
```

**Sounio QNN**:
```sounio
let attn = quat_attention_new(embed_dim: 64, num_heads: 8)
let output = quat_attention_forward(&attn, query: &q, key: &k, value: &v)
```

---

## 4. Optimizer and Loss Migration

### 4.1 Learning Rate Adjustment

**Critical**: Quaternions have 4 components, so gradients are ~2× larger. Halve your learning rate.

| Real-Valued | QNN Equivalent |
|------------|----------------|
| `lr = 0.001` | `lr = 0.0005` |
| `lr = 0.01` | `lr = 0.005` |
| `lr = 0.1` | `lr = 0.05` |

### 4.2 Optimizer Conversion

**PyTorch Adam**:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Sounio QNN Adam**:
```sounio
var optimizer = quat_adam_new(learning_rate: 0.0005)  // Half the LR!
```

**PyTorch SGD**:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Sounio QNN SGD**:
```sounio
var optimizer = quat_sgd_new(
    learning_rate: 0.005,  // Half the LR!
    momentum: 0.9,
    weight_decay: 0.0001
)
```

### 4.3 Loss Function Conversion

| PyTorch | Sounio QNN |
|---------|-----------|
| `nn.MSELoss()` | `quat_mse_loss(&pred, &target)` |
| `nn.L1Loss()` | `quat_mae_loss(&pred, &target)` |
| `nn.CosineSimilarity()` | `quat_cosine_similarity_loss(&pred, &target)` |

---

## 5. Step-by-Step Example: MNIST

### 5.1 Original PyTorch

```python
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MNISTClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

### 5.2 Sounio QNN Equivalent

```sounio
//! qnn_mnist_classifier.sio

fn main() {
    // Layer definitions
    let layer1 = quat_linear_new(784, 256)
    let layer2 = quat_linear_new(256, 128)
    let layer3 = quat_linear_new(128, 10)

    // Weight initialization
    var w1 = quat_xavier_init(784, 256, seed: 100)
    var b1 = quat_xavier_init(1, 256, seed: 101)
    var w2 = quat_he_init(256, 128, seed: 102)  // He for ReLU
    var b2 = quat_xavier_init(1, 128, seed: 103)
    var w3 = quat_xavier_init(128, 10, seed: 104)
    var b3 = quat_xavier_init(1, 10, seed: 105)

    // Optimizer: HALF the learning rate
    var optimizer = quat_adam_new(learning_rate: 0.0005)

    // Training loop
    let epoch: i32 = 0
    while epoch < 10 {
        // Forward pass
        let h1 = quat_relu(quat_linear_forward(&layer1, &w1, &input, &b1))
        let h2 = quat_relu(quat_linear_forward(&layer2, &w2, &h1, &b2))
        let output = quat_linear_forward(&layer3, &w3, &h2, &b3)

        // Loss
        let loss = quat_mse_loss(&output, &target)

        // Backward + Update
        // ... gradient computation and optimizer step

        epoch = epoch + 1
    }
}
```

### 5.3 Key Differences

| Aspect | PyTorch | Sounio QNN |
|--------|---------|-----------|
| Learning rate | 0.001 | 0.0005 |
| Weight init | Default | Xavier/He explicit |
| Activation | `F.relu()` | `quat_relu()` |
| Parameters | 235K | 235K (as quaternions) |

---

## 6. Troubleshooting

### 6.1 Gradient Explosion

**Symptom**: Loss becomes NaN or Inf after a few epochs.

**Cause**: Hamilton product can amplify gradients.

**Solution**: Apply gradient clipping:

```sounio
fn clip_gradients(grads: &![Quat], max_norm: f32) {
    let total_norm = compute_grad_norm(grads)
    if total_norm > max_norm {
        let scale = max_norm / total_norm
        let i: i32 = 0
        while i < grads.len() {
            grads[i] = quat(
                grads[i].w * scale,
                grads[i].x * scale,
                grads[i].y * scale,
                grads[i].z * scale
            )
            i = i + 1
        }
    }
}

// Use max_norm = 1.0 as starting point
```

### 6.2 Slow Convergence

**Symptom**: Loss decreases much slower than real-valued baseline.

**Solutions**:
1. **Further reduce LR**: Try `lr / 4` instead of `lr / 2`
2. **Use cosine annealing**:
   ```sounio
   let lr = quat_learning_rate_schedule(
       initial_lr: 0.0005,
       step: current_step,
       total_steps: total_steps,
       min_lr: 0.00001
   )
   ```
3. **Increase batch size**: QNNs benefit from larger batches

### 6.3 Unit Quaternion Drift

**Symptom**: Rotation predictions become unstable over time.

**Cause**: Quaternion norms drift from 1.0 during training.

**Solutions**:

1. **Use Riemannian SGD** (respects S³ manifold):
   ```sounio
   var optimizer = quat_riemannian_sgd_new(learning_rate: 0.001)
   ```

2. **Periodic renormalization**:
   ```sounio
   fn renormalize_weights(weights: &![Quat]) {
       let i: i32 = 0
       while i < weights.len() {
           weights[i] = quat_normalize(weights[i])
           i = i + 1
       }
   }
   // Call every N epochs
   ```

### 6.4 Shape Mismatch Errors

**Symptom**: "Dimension mismatch" errors during forward pass.

**Cause**: Quaternion layers expect 4-channel data.

**Solution**: Ensure input encoding produces quaternions:

```sounio
// Wrong: Raw grayscale (1 channel)
let input = load_grayscale_image(path)

// Right: Encoded as quaternions (4 components)
let input = encode_grayscale_as_quat(load_grayscale_image(path))
```

---

## 7. Migration Checklist

### Before Migration

- [ ] Verify your task has rotational/3D structure
- [ ] Identify data encoding strategy
- [ ] Plan layer-by-layer conversion
- [ ] Estimate memory requirements (same as real-valued)

### During Migration

- [ ] Convert layers systematically
- [ ] Halve learning rate
- [ ] Add gradient clipping
- [ ] Test each layer individually

### After Migration

- [ ] Compare accuracy to baseline
- [ ] Profile performance (should be similar or faster)
- [ ] Monitor gradient norms during training
- [ ] Test with rotated/augmented data

---

## See Also

- [Programming Guide](PROGRAMMING_GUIDE.md) — QNN fundamentals
- [Performance Handbook](PERFORMANCE_HANDBOOK.md) — Optimization techniques
- [Architecture Deep-Dive](ARCHITECTURE_DEEP_DIVE.md) — Implementation details
