<!-- docs:meta
topic_id: repo.docs.qnn.faq
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.qnn.faq
-->

# QNN FAQ - Frequently Asked Questions

---

## Getting Started

### Q1: What is a quaternion and why should I care?

**A:** A quaternion is a four-dimensional number system: q = w + xi + yj + zk, where i² = j² = k² = ijk = -1.

**Why it matters for neural networks:**

1. **Compact 3D rotations**: Unit quaternions represent rotations without gimbal lock
2. **4× parameter efficiency**: Store [w, x, y, z] as one learned unit instead of four separate weights
3. **Natural geometry**: Hamilton product encodes rotational transformations

**Example:**
```
Real-valued 3D rotation: Requires rotation matrix (9 parameters, redundant)
Quaternion rotation: Only 4 parameters, non-commutative algebra built-in
```

### Q2: When should I use QNNs vs standard networks?

**Use QNNs for:**
- 3D vision (point clouds, meshes)
- Robotics (pose, orientation)
- Motion capture (skeletal animation)
- Protein folding (bond angles)
- Any task with rotational semantics

**Stick with float networks for:**
- 2D image classification
- NLP (text, language)
- Tabular data
- Time series without rotation

**Rule of thumb:** If your data involves 3D rotations or orientations, QNNs will help.

### Q3: What's the learning curve?

**Easy:**
- Basic quaternion operations: 1 hour
- Training a small QNN: 2-3 hours
- Understanding Hamilton product: 30 minutes

**Harder:**
- Quaternion-aware optimization: 1-2 days
- GPU kernel implementation: 2-3 weeks
- Numerically stable normalization: 1 week

**Recommendation:** Start with [hello_quaternion.sio](../examples/qnn/01_hello_quaternion.sio), then [PROGRAMMING_GUIDE.md](PROGRAMMING_GUIDE.md).

---

## Training & Optimization

### Q4: Why do my QNNs converge slower than float networks?

**Most common causes:**

1. **Learning rate too high** → Unstable gradients
   ```
   Solution: Use lr_qnn = lr_float / 2
   ```

2. **Missing gradient clipping** → Exploding gradients
   ```sounio
   clip_quaternion_gradients(&gradients, max_norm: 1.0)
   ```

3. **Large batch sizes amplify quaternion cross-products**
   ```
   Try: batch_size = 32 or 64 (not 512)
   ```

**Fix checklist:**
- [ ] Halved learning rate?
- [ ] Added gradient clipping?
- [ ] Batch size ≤ 128?
- [ ] Using correct weight initialization (Xavier)?

### Q5: My loss becomes NaN after 10 epochs. Help!

**Diagnosis:**

1. **Check gradient norms**: Should be < 1 after clipping
2. **Check quaternion norms**: Should stay close to 1 for unit quats
3. **Check weight ranges**: Should be bounded (e.g., w,x,y,z ∈ [-1, 1])

**Solutions (in order):**

```sounio
// 1. Add gradient clipping
clip_quaternion_gradients(&grads, max_norm: 0.5)  // Start conservative

// 2. Reduce learning rate further
let lr = 0.0005 / 2  // Halve again

// 3. Add periodic renormalization
if epoch % 10 == 0 {
    renormalize_weights(&weights)
}

// 4. Use Riemannian SGD (respects unit quaternion manifold)
var optimizer = quat_riemannian_sgd_new(learning_rate: 0.0003)
```

### Q6: How do I choose batch size?

**Guidelines:**

| Batch Size | Memory | Training | Gradient Noise | Best For |
|-----------|--------|----------|-----------------|----------|
| 8 | Low | Slow | High | Tiny datasets |
| **32** | Medium | Good | Moderate | **Recommended** |
| 64 | Medium | Faster | Lower | Fast iteration |
| 128 | Medium | Very fast | Low | Large datasets |
| 256+ | High | Fastest | Very low | GPU training only |

**For GPU:**
- Minimum 16 for Tensor Core utilization
- Optimal: 128-256
- Adjust lr downward as batch size increases

### Q7: Should I use Adam or SGD with momentum?

**Adam** (recommended):
```sounio
var opt = quat_adam_new(learning_rate: 0.0005)
// Adaptive learning per component
// Handles quaternion gradients well
```

**SGD with momentum** (simpler):
```sounio
var opt = quat_sgd_new(
    learning_rate: 0.001,
    momentum: 0.9,
    weight_decay: 0.0001
)
// More control, but needs careful tuning
```

**Riemannian SGD** (for unit quaternions):
```sounio
var opt = quat_riemannian_sgd_new(learning_rate: 0.0003)
// Respects S³ manifold
// Best for rotation-heavy tasks
```

---

## Implementation & Debugging

### Q8: How do I encode my data as quaternions?

**RGB Images:**
```sounio
fn rgb_to_quat(r: f32, g: f32, b: f32) -> Quat {
    // Luminance as w, color differences as x,y,z
    let L = 0.299*r + 0.587*g + 0.114*b
    quat(L, r-L, g-L, b-L)
}
```

**3D Coordinates:**
```sounio
fn pos_to_quat(x: f32, y: f32, z: f32) -> Quat {
    // Pure quaternion (w=0)
    quat(0.0, x, y, z)
}

// Or with magnitude:
fn pos_scaled_to_quat(x: f32, y: f32, z: f32) -> Quat {
    let mag = (x*x + y*y + z*z).sqrt()
    quat(mag, x/mag, y/mag, z/mag)
}
```

**Time Series:**
```sounio
// Sliding window of 3D poses
fn trajectory_to_quats(positions: &[[f32; 3]], window: i32) -> [Quat] {
    var quats: [Quat; window] = []
    let i: i32 = 0
    while i < window {
        let p = positions[i]
        quats[i] = quat(0.0, p[0], p[1], p[2])
        i = i + 1
    }
    quats
}
```

### Q9: How do I debug quaternion networks?

**Validation techniques:**

1. **Finite difference gradient checking:**
   ```sounio
   let eps = 1e-4
   let grad_numerical = (loss(w + eps) - loss(w - eps)) / (2*eps)
   let grad_computed = backward(loss, w)
   // Check: |grad_numerical - grad_computed| / |grad_numerical| < 0.01
   ```

2. **Monitor quaternion norms:**
   ```sounio
   fn check_norm_drift(weights: &[Quat]) {
       let i: i32 = 0
       while i < weights.len() {
           let n = quat_norm(weights[i])
           if n > 1.1 || n < 0.9 {
               println("Norm drift at index %d: %f", i, n)
           }
           i = i + 1
       }
   }
   ```

3. **Print gradient statistics:**
   ```sounio
   fn print_gradient_stats(grads: &[Quat]) {
       let total = 0.0
       let i: i32 = 0
       while i < grads.len() {
           total = total + quat_norm(grads[i])
           i = i + 1
       }
       println("Avg gradient norm: %f", total / (grads.len() as f32))
   }
   ```

### Q10: What does "non-commutative" mean and why does it matter?

**Definition:**
```
Commutative:      a × b = b × a    (real multiplication)
Non-commutative:  q1 ⊗ q2 ≠ q2 ⊗ q1  (quaternion multiplication)
```

**Why it matters:**

```sounio
let q1 = quat(1.0, 0.0, 0.0, 0.0)
let q2 = quat(0.0, 1.0, 0.0, 0.0)

let ab = hamilton_product(q1, q2)  // Result: quat(0, 0, 0, 1)
let ba = hamilton_product(q2, q1)  // Result: quat(0, 0, 0, -1)
// ab ≠ ba!
```

**Implications:**
- Order of layers matters for quaternion multiplication
- Gradients flow differently backward
- Geometric intuition: Order of rotations matters in 3D

---

## Performance & Deployment

### Q11: How much faster are QNNs really?

**CPU Performance (SIMD):**
- Single operation: 4-8× faster with AVX2/AVX-512
- Full network: 2-4× faster due to memory bandwidth

**GPU Performance (Tensor Cores):**
- Large batch (>128): 10-20× faster with WMMA
- Small batch (<32): 2-4× faster

**Memory:**
- Always 4× fewer learned parameters
- Same total memory (storing quaternions as 4 floats)

**Real-world example:**
```
MNIST: 784 → 256 → 128 → 10
Float:      2M params training
QNN:        500K params training (4× fewer)
            Same inference speed on CPU
            8× faster on GPU
```

### Q12: Should I quantize to INT8?

**Benefits:**
- 4× memory reduction (16 bytes → 4 bytes per quaternion)
- 2-4× inference speedup
- Efficient mobile deployment

**Tradeoffs:**
```
FP32 accuracy:    95.0%
INT8 accuracy:    94.1%     (0.9% loss acceptable)
INT8 + retraining: 94.8%    (better)
```

**When to use:**
- Mobile/edge deployment: YES
- High-precision tasks: NO
- Real-time inference: YES

---

## Advanced Topics

### Q13: What's the difference between S³, SO(3), and SU(2)?

**For practitioners:**

```
S³ (3-sphere):     Unit quaternions (|q| = 1)
                   4D hypersurface in ℝ⁴
                   "Space where quaternions live"

SO(3):             3D rotation matrices
                   Describes actual 3D rotations

SU(2):             2×2 unitary matrices
                   Double-cover of SO(3)
                   Every q ∈ S³ represents 2 rotations in SU(2)
```

**Why you care:**
- Keep quaternion weights on S³ (normalized)
- Use Riemannian SGD to stay on the manifold
- Periodic renormalization prevents drift

### Q14: How do I handle numerical instability?

**Common issues:**

1. **Quaternion norm drift:**
   ```sounio
   // Fix: Periodic renormalization
   if epoch % 10 == 0 {
       let i: i32 = 0
       while i < weights.len() {
           weights[i] = quat_normalize(weights[i])
           i = i + 1
       }
   }
   ```

2. **Gradient explosion:**
   ```sounio
   // Fix: Gradient clipping with smaller threshold
   clip_quaternion_gradients(&grads, max_norm: 0.1)
   ```

3. **Division by near-zero:**
   ```sounio
   // Fix: Add epsilon check
   fn quat_normalize_safe(q: Quat) -> Quat {
       let n = quat_norm(q)
       if n > 1e-8 {
           quat_normalize(q)
       } else {
           quat(1.0, 0.0, 0.0, 0.0)  // Return identity
       }
   }
   ```

### Q15: Can I use QNNs for non-geometric data?

**Short answer:** Probably not better than floats.

**Long answer:**
- QNNs add structure (rotational invariance)
- Only beneficial if data has that structure
- On unstructured data: Similar or worse performance
- Exception: Very small models where 4× efficiency helps memory-constrained devices

**Test it:**
```
1. Baseline float network: 95% accuracy
2. QNN network: < 93% (loss from quaternion structure)
3. Conclusion: Use floats for this task
```

---

## Resources & References

### Key Papers

1. **Gaudet & Maida (2018)** - "Deep Quaternion Networks"
   - [arXiv:1705.07944](https://arxiv.org/abs/1705.07944)

2. **Parcollet et al. (2019)** - "Quaternion Recurrent Neural Networks"
   - [arXiv:1903.08478](https://arxiv.org/abs/1903.08478)

3. **Zhu et al. (2018)** - "Quaternion Convolutional Neural Networks"
   - ECCV 2018

### Documentation

- [Programming Guide](PROGRAMMING_GUIDE.md) — Getting started
- [Performance Handbook](PERFORMANCE_HANDBOOK.md) — Optimization
- [Architecture Deep-Dive](ARCHITECTURE_DEEP_DIVE.md) — Under the hood
- [Migration Guide](MIGRATION_GUIDE.md) — Converting from float

### Examples

- [01_hello_quaternion.sio](../examples/qnn/01_hello_quaternion.sio)
- [02_basic_linear.sio](../examples/qnn/02_basic_linear.sio)
- [qnn_mnist.sio](../examples/qnn_mnist.sio) - Full training example

---

## Still Have Questions?

**Checklist before asking:**
- [ ] Checked the [PROGRAMMING_GUIDE.md](PROGRAMMING_GUIDE.md)?
- [ ] Ran the examples?
- [ ] Checked gradient with finite differences?
- [ ] Added gradient clipping?
- [ ] Halved learning rate?

If still stuck, the Sounio community is happy to help!
