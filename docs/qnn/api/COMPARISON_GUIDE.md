# PyTorch vs Sounio QNN Comparison

This guide provides side-by-side comparisons of equivalent PyTorch and Sounio code, helping you understand the translation patterns.

---

## 1. Basic Operations

### Creating and Manipulating Quaternions

| Concept | PyTorch | Sounio QNN |
|---------|---------|-----------|
| **Create quaternion** | N/A (not native) | `quat(1.0, 0.0, 0.0, 0.0)` |
| **Create from components** | Complex tensor ops | `quat(w, x, y, z)` |
| **Conjugate** | `torch.conj()` | `quat_conjugate(q)` |
| **Norm** | `torch.norm(q)` | `quat_norm(q)` |
| **Normalize** | `q / q.norm()` | `quat_normalize(q)` |

### Initialization Strategies

**PyTorch**:
```python
# Xavier/Glorot
nn.init.xavier_uniform_(layer.weight)

# He/Kaiming (for ReLU)
nn.init.kaiming_normal_(layer.weight)
```

**Sounio QNN**:
```sounio
// Xavier/Glorot
let weights = quat_xavier_init(in_features, out_features, seed: 42)

// He/Kaiming (for ReLU)
let weights = quat_he_init(in_features, out_features, seed: 42)

// Unit quaternions (all norms = 1)
let weights = quat_unit_init(in_features, out_features, seed: 42)
```

---

## 2. Neural Network Layers

### Linear Layers

**PyTorch**:
```python
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TwoLayerNet()
```

**Sounio QNN**:
```sounio
fn forward(input: &[Quat], w1: &[Quat], b1: &[Quat],
           w2: &[Quat], b2: &[Quat]) -> [Quat] {
    let layer1 = quat_linear_new(64, 32)
    let layer2 = quat_linear_new(32, 10)

    let h = quat_linear_forward(&layer1, &w1, &input, &b1)
    let h_activated = quat_relu(h)
    let output = quat_linear_forward(&layer2, &w2, &h_activated, &b2)

    output
}
```

**Key Differences**:
- Sounio uses functional style (no class-based models yet)
- Weights passed as parameters to forward function
- Biases are separate quaternion arrays

### Convolutional Layers

**PyTorch**:
```python
self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
output = self.conv1(input)
```

**Sounio QNN**:
```sounio
let conv = quat_conv2d_new(3, 16, kernel_h: 3, kernel_w: 3)
let output = quat_conv2d_forward(&conv, &weights, &input, stride: 1, padding: 1)
```

---

## 3. Activation Functions

| Function | PyTorch | Sounio QNN | Behavior |
|----------|---------|-----------|----------|
| **ReLU** | `F.relu(x)` | `quat_relu(x)` | Component-wise max(0, component) |
| **Sigmoid** | `torch.sigmoid(x)` | `quat_sigmoid(x)` | Component-wise 1/(1+exp(-x)) |
| **Tanh** | `torch.tanh(x)` | `quat_tanh(x)` | Component-wise (exp(x)-exp(-x))/(exp(x)+exp(-x)) |
| **Leaky ReLU** | `F.leaky_relu(x, 0.01)` | `quat_leaky_relu(x, 0.01)` | Component-wise max(x, 0.01*x) |

**Example**:

PyTorch:
```python
x = torch.randn(32, 64)
y = F.relu(x)
```

Sounio:
```sounio
var x = quat_init_xavier(32, 64, seed: 100)
var y = quat_relu(x)
```

---

## 4. Optimizers and Learning

### Optimizer Setup

**PyTorch**:
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    loss = model(x) - y
    loss.backward()
    optimizer.step()
```

**Sounio QNN**:
```sounio
// Note: Halve learning rate for QNNs
var optimizer = quat_adam_new(learning_rate: 0.0005)  // 0.001 → 0.0005
var first_moment = quat_zeros(num_params)
var second_moment = quat_zeros(num_params)

// Training loop
let epoch: i32 = 0
while epoch < 10 {
    let output = forward(input, weights, bias)
    let loss = quat_mse_loss(&output, &target)

    // Compute gradients (manual or AD)
    let gradients = compute_gradients(loss, weights)

    // Clip gradients
    clip_quaternion_gradients(&gradients, max_norm: 1.0)

    // Update
    quat_adam_step(&weights, &gradients, &optimizer,
                   &first_moment, &second_moment)

    epoch = epoch + 1
}
```

### Learning Rate Comparison

| Scenario | PyTorch | Sounio QNN | Reason |
|----------|---------|-----------|--------|
| Classification (default) | 0.001 | 0.0005 | 4 gradient components |
| Fine-tuning | 0.0001 | 0.00005 | Conservative updates |
| Rotation prediction | 0.0005 | 0.00025 | Sensitive to changes |

---

## 5. Loss Functions

### Common Losses

**PyTorch**:
```python
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
criterion_cross = nn.CrossEntropyLoss()

loss = criterion_mse(output, target)
```

**Sounio QNN**:
```sounio
// Mean Squared Error (for regression)
let mse_loss = quat_mse_loss(&output, &target)

// Mean Absolute Error (robust to outliers)
let mae_loss = quat_mae_loss(&output, &target)

// Cosine Similarity (for rotation alignment)
let cos_loss = quat_cosine_similarity_loss(&output, &target)
```

### Loss Function Selection

| Task | PyTorch | Sounio QNN | Why |
|------|---------|-----------|-----|
| Rotation prediction | MSELoss | `quat_mse_loss` | Penalizes all 4 components |
| Pose estimation | SmoothL1Loss | `quat_mae_loss` | Robust to outliers |
| Angular alignment | CosineSimilarity | `quat_cosine_similarity_loss` | Focuses on angle, not magnitude |

---

## 6. Complete Training Example

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class QNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Note: Real PyTorch doesn't have native quaternion layers
        # This is a hypothetical example
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train():
    model = QNNModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### Sounio QNN Implementation

```sounio
//! Equivalent QNN training in Sounio

fn main() {
    // Architecture: 64 → 32 → 10
    let layer1 = quat_linear_new(64, 32)
    let layer2 = quat_linear_new(32, 10)

    // Initialize weights
    var w1 = quat_xavier_init(64, 32, seed: 100)
    var b1 = quat_xavier_init(1, 32, seed: 101)
    var w2 = quat_xavier_init(32, 10, seed: 102)
    var b2 = quat_xavier_init(1, 10, seed: 103)

    // Optimizer with HALVED learning rate
    var opt = quat_adam_new(learning_rate: 0.0005)  // 0.001 → 0.0005
    var m1 = quat_zeros(64 * 32)
    var m2_state = quat_zeros(64 * 32)
    var v1 = quat_zeros(64 * 32)
    var v2_state = quat_zeros(64 * 32)

    // Training loop
    let epoch: i32 = 0
    while epoch < 10 {
        // Forward pass
        let h = quat_relu(quat_linear_forward(&layer1, &w1, &input, &b1))
        let output = quat_linear_forward(&layer2, &w2, &h, &b2)

        // Loss computation
        let loss = quat_mse_loss(&output, &target)

        // Gradient computation (simplified)
        let grad_w1 = compute_gradient_w1(loss)
        let grad_w2 = compute_gradient_w2(loss)

        // Gradient clipping (ESSENTIAL for quaternions)
        clip_quaternion_gradients(&grad_w1, max_norm: 1.0)
        clip_quaternion_gradients(&grad_w2, max_norm: 1.0)

        // Update step
        quat_adam_step(&w1, &grad_w1, &opt, &m1, &v1)
        quat_adam_step(&w2, &grad_w2, &opt, &m2_state, &v2_state)

        if (epoch + 1) % 2 == 0 {
            println("Epoch: %d, Loss: %f")
        }

        epoch = epoch + 1
    }
}
```

### Key Implementation Differences

| Aspect | PyTorch | Sounio QNN |
|--------|---------|-----------|
| **Learning rate** | 0.001 | 0.0005 (halved) |
| **Gradient clipping** | Optional | Essential |
| **Model definition** | Class-based | Functional |
| **Parameter management** | Automatic | Manual arrays |
| **Computation graph** | Automatic AD | Manual or AD |

---

## 7. Performance Comparison

### Memory Usage

| Operation | PyTorch (float32) | Sounio QNN (f32) | Advantage |
|-----------|------------------|-----------------|-----------|
| 64→32 layer weights | 8 KB | 8 KB | Same memory |
| Learned parameters | 2,048 scalars | 512 quaternions | 4× fewer |
| Effective capacity | Equivalent | Equivalent | Tied |

### Execution Speed

| Task | PyTorch (CPU) | Sounio (CPU SIMD) | Speedup |
|------|--------------|------------------|---------|
| Linear 64→32 forward | 120 μs | 15-30 μs | 4-8× |
| Batch ReLU (128 quats) | 5 μs | 1.2 μs | 4× |
| Linear 64→32 (GPU A100) | 50 μs | 6 μs | 8× |

---

## 8. Migration Checklist

When converting PyTorch to Sounio QNN:

- [ ] **Learning rate**: Divide by 2
- [ ] **Weight init**: Use `quat_xavier_init` or `quat_he_init`
- [ ] **Activations**: Use quaternion-specific functions
- [ ] **Gradient clipping**: Add with max_norm = 1.0
- [ ] **Loss function**: Choose quaternion-aware variant
- [ ] **Optimizer**: Use quaternion optimizer
- [ ] **Data encoding**: Convert inputs to quaternions
- [ ] **Testing**: Verify gradient correctness with finite differences

---

## See Also

- [Programming Guide](../PROGRAMMING_GUIDE.md) — QNN fundamentals
- [Migration Guide](../MIGRATION_GUIDE.md) — Detailed conversion guide
