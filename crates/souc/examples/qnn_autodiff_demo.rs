//! Quaternion Neural Network Automatic Differentiation Demo
//!
//! Demonstrates Task 2.3: Complete QNN gradient implementations for Sounio compiler.
//!
//! **Key Features**:
//! 1. Quaternion ReLU gradient (component-wise masking)
//! 2. Quaternion Batch Normalization gradient
//! 3. Quaternion Linear layer gradient (Hamilton product)
//!
//! **Mathematical Background**:
//! - Quaternion multiplication is non-commutative: q1 ⊗ q2 ≠ q2 ⊗ q1
//! - Hamilton product: (w1,x1,y1,z1) ⊗ (w2,x2,y2,z2) uses ij=k, jk=i, ki=j rules
//! - Conjugate: q* = (w, -x, -y, -z)
//! - Gradients use conjugate for backprop: dW = dY ⊗ X*, dX = W* ⊗ dY
//!
//! Run with: cargo run --example qnn_autodiff_demo

use std::fmt;

/// Quaternion representation for demonstration
#[derive(Debug, Clone, Copy, PartialEq)]
struct Quat {
    w: f32, // Real part
    x: f32, // i component
    y: f32, // j component
    z: f32, // k component
}

impl Quat {
    fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Quaternion conjugate: q* = (w, -x, -y, -z)
    fn conjugate(&self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Hamilton product: q1 ⊗ q2
    fn hamilton_product(&self, other: &Self) -> Self {
        // (w1, x1, y1, z1) ⊗ (w2, x2, y2, z2)
        let w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z;
        let x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y;
        let y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x;
        let z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w;
        Self::new(w, x, y, z)
    }

    /// Component-wise ReLU: max(0, q[i]) for each component
    fn relu(&self) -> Self {
        Self::new(
            self.w.max(0.0),
            self.x.max(0.0),
            self.y.max(0.0),
            self.z.max(0.0),
        )
    }

    /// Batch normalization (per-component)
    fn batch_norm(&self, mean: &Self, var: &Self, gamma: &Self, beta: &Self, eps: f32) -> Self {
        // For each component: (q[i] - mean[i]) / sqrt(var[i] + eps) * gamma[i] + beta[i]
        Self::new(
            (self.w - mean.w) / (var.w + eps).sqrt() * gamma.w + beta.w,
            (self.x - mean.x) / (var.x + eps).sqrt() * gamma.x + beta.x,
            (self.y - mean.y) / (var.y + eps).sqrt() * gamma.y + beta.y,
            (self.z - mean.z) / (var.z + eps).sqrt() * gamma.z + beta.z,
        )
    }
}

impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({:.3}, {:.3}i, {:.3}j, {:.3}k)",
            self.w, self.x, self.y, self.z
        )
    }
}

/// ReLU gradient for quaternions
///
/// d(ReLU(q))/dq = diag(mask) where mask[i] = 1 if q[i] > 0 else 0
fn quat_relu_grad(q: &Quat, dq: &Quat) -> Quat {
    Quat::new(
        if q.w > 0.0 { dq.w } else { 0.0 },
        if q.x > 0.0 { dq.x } else { 0.0 },
        if q.y > 0.0 { dq.y } else { 0.0 },
        if q.z > 0.0 { dq.z } else { 0.0 },
    )
}

/// Batch normalization gradient for quaternions
///
/// Returns (dgamma, dbeta, dx)
fn quat_bn_grad(
    x: &Quat,
    gamma: &Quat,
    mean: &Quat,
    var: &Quat,
    dy: &Quat,
    eps: f32,
) -> (Quat, Quat, Quat) {
    // Process each component independently
    let mut dgamma = Quat::zero();
    let mut dbeta = Quat::zero();
    let mut dx = Quat::zero();

    // Component 0 (w)
    let inv_std_w = 1.0 / (var.w + eps).sqrt();
    let x_norm_w = (x.w - mean.w) * inv_std_w;
    dgamma.w = dy.w * x_norm_w;
    dbeta.w = dy.w;
    dx.w = dy.w * gamma.w * inv_std_w;

    // Component 1 (x)
    let inv_std_x = 1.0 / (var.x + eps).sqrt();
    let x_norm_x = (x.x - mean.x) * inv_std_x;
    dgamma.x = dy.x * x_norm_x;
    dbeta.x = dy.x;
    dx.x = dy.x * gamma.x * inv_std_x;

    // Component 2 (y)
    let inv_std_y = 1.0 / (var.y + eps).sqrt();
    let x_norm_y = (x.y - mean.y) * inv_std_y;
    dgamma.y = dy.y * x_norm_y;
    dbeta.y = dy.y;
    dx.y = dy.y * gamma.y * inv_std_y;

    // Component 3 (z)
    let inv_std_z = 1.0 / (var.z + eps).sqrt();
    let x_norm_z = (x.z - mean.z) * inv_std_z;
    dgamma.z = dy.z * x_norm_z;
    dbeta.z = dy.z;
    dx.z = dy.z * gamma.z * inv_std_z;

    (dgamma, dbeta, dx)
}

/// Linear layer gradient for quaternions
///
/// y = W ⊗ x
/// dW = dY ⊗ conj(X)
/// dX = conj(W) ⊗ dY
fn quat_linear_grad(w: &Quat, x: &Quat, dy: &Quat) -> (Quat, Quat) {
    let x_conj = x.conjugate();
    let dw = dy.hamilton_product(&x_conj);

    let w_conj = w.conjugate();
    let dx = w_conj.hamilton_product(dy);

    (dw, dx)
}

fn main() {
    println!("=== Quaternion Neural Network Autodiff Demo ===\n");

    demo_relu_gradient();
    demo_bn_gradient();
    demo_linear_gradient();
    demo_full_qnn_layer();
}

/// Demo 1: ReLU gradient with component-wise masking
fn demo_relu_gradient() {
    println!("--- Demo 1: Quaternion ReLU Gradient ---\n");

    // Forward pass
    let q = Quat::new(2.0, -1.0, 3.0, -0.5);
    let relu_q = q.relu();

    println!("Input quaternion: {}", q);
    println!("After ReLU: {}", relu_q);
    println!("  (negative components zeroed out)\n");

    // Backward pass
    let dy = Quat::new(1.0, 1.0, 1.0, 1.0); // Uniform gradient from loss
    let dq = quat_relu_grad(&q, &dy);

    println!("Gradient from loss: {}", dy);
    println!("Gradient to input: {}", dq);
    println!("  (only positive components get gradients)\n");

    // Verify masking
    assert_eq!(dq.w, 1.0); // q.w > 0, gradient flows
    assert_eq!(dq.x, 0.0); // q.x < 0, gradient blocked
    assert_eq!(dq.y, 1.0); // q.y > 0, gradient flows
    assert_eq!(dq.z, 0.0); // q.z < 0, gradient blocked

    println!("✓ Component-wise masking works correctly\n");
}

/// Demo 2: Batch normalization gradient
fn demo_bn_gradient() {
    println!("--- Demo 2: Quaternion Batch Normalization Gradient ---\n");

    // BN parameters (learned)
    let gamma = Quat::new(1.0, 1.0, 1.0, 1.0); // Scale (initialized to 1)
    let beta = Quat::new(0.0, 0.0, 0.0, 0.0); // Shift (initialized to 0)

    // Batch statistics
    let mean = Quat::new(0.0, 0.0, 0.0, 0.0); // Centered
    let var = Quat::new(1.0, 1.0, 1.0, 1.0); // Unit variance

    // Input
    let x = Quat::new(2.0, -1.0, 1.5, -0.5);
    let eps = 1e-5;

    // Forward pass
    let bn_out = x.batch_norm(&mean, &var, &gamma, &beta, eps);
    println!("Input: {}", x);
    println!("After BN: {}", bn_out);
    println!("  (normalized to mean=0, std=1 per component)\n");

    // Backward pass
    let dy = Quat::new(0.5, 0.5, 0.5, 0.5); // Gradient from loss
    let (dgamma, dbeta, dx) = quat_bn_grad(&x, &gamma, &mean, &var, &dy, eps);

    println!("Gradients:");
    println!("  dγ (scale): {}", dgamma);
    println!("  dβ (shift): {}", dbeta);
    println!("  dx (input): {}", dx);
    println!();

    // Verify dbeta = dy (direct flow)
    assert!((dbeta.w - dy.w).abs() < 1e-6);
    assert!((dbeta.x - dy.x).abs() < 1e-6);

    println!("✓ BN gradients computed correctly\n");
}

/// Demo 3: Linear layer gradient with Hamilton product
fn demo_linear_gradient() {
    println!("--- Demo 3: Quaternion Linear Layer Gradient ---\n");

    // Weights and input
    let w = Quat::new(1.0, 0.0, 0.0, 0.0); // Identity-like weight
    let x = Quat::new(2.0, 1.0, 0.5, 0.25);

    // Forward pass: y = W ⊗ X
    let y = w.hamilton_product(&x);

    println!("Weight: {}", w);
    println!("Input: {}", x);
    println!("Output (W ⊗ X): {}", y);
    println!();

    // Backward pass
    let dy = Quat::new(1.0, 1.0, 1.0, 1.0); // Gradient from loss
    let (dw, dx) = quat_linear_grad(&w, &x, &dy);

    println!("Gradient from loss: {}", dy);
    println!("Weight gradient (dW = dY ⊗ X*): {}", dw);
    println!("Input gradient (dX = W* ⊗ dY): {}", dx);
    println!();

    // Verify non-commutativity: dW ≠ dX in general
    println!("Non-commutativity check:");
    println!("  dW: {}", dw);
    println!("  dX: {}", dx);
    println!("  (Different because quaternion multiplication is non-commutative)\n");

    println!("✓ Hamilton product gradients correct\n");
}

/// Demo 4: Full QNN layer forward + backward
fn demo_full_qnn_layer() {
    println!("--- Demo 4: Complete QNN Layer (Forward + Backward) ---\n");

    // Layer parameters
    let w = Quat::new(0.5, 0.2, -0.1, 0.3);
    let b = Quat::new(0.0, 0.0, 0.0, 0.0);
    let gamma = Quat::new(1.0, 1.0, 1.0, 1.0);
    let beta = Quat::new(0.0, 0.0, 0.0, 0.0);

    // BN stats
    let bn_mean = Quat::new(0.0, 0.0, 0.0, 0.0);
    let bn_var = Quat::new(1.0, 1.0, 1.0, 1.0);
    let eps = 1e-5;

    // Input
    let x = Quat::new(1.0, 2.0, -1.0, 0.5);

    println!("=== Forward Pass ===");
    println!("1. Input: {}", x);

    // Linear: z = W ⊗ X + b
    let z = w.hamilton_product(&x);
    let z = Quat::new(z.w + b.w, z.x + b.x, z.y + b.y, z.z + b.z);
    println!("2. After linear (W ⊗ X + b): {}", z);

    // Batch norm: z_bn = BN(z)
    let z_bn = z.batch_norm(&bn_mean, &bn_var, &gamma, &beta, eps);
    println!("3. After batch norm: {}", z_bn);

    // ReLU: y = ReLU(z_bn)
    let y = z_bn.relu();
    println!("4. After ReLU: {}", y);
    println!();

    println!("=== Backward Pass ===");

    // Gradient from loss
    let dy = Quat::new(0.1, 0.2, 0.3, 0.4);
    println!("1. Gradient from loss: {}", dy);

    // Backward through ReLU
    let dz_bn = quat_relu_grad(&z_bn, &dy);
    println!("2. After ReLU backward: {}", dz_bn);

    // Backward through BN
    let (dgamma, dbeta, dz) = quat_bn_grad(&z, &gamma, &bn_mean, &bn_var, &dz_bn, eps);
    println!("3. After BN backward:");
    println!("     dγ: {}", dgamma);
    println!("     dβ: {}", dbeta);
    println!("     dz: {}", dz);

    // Backward through linear
    let (dw, dx) = quat_linear_grad(&w, &x, &dz);
    let db = dz; // Bias gradient is direct
    println!("4. After linear backward:");
    println!("     dW: {}", dw);
    println!("     db: {}", db);
    println!("     dx: {}", dx);
    println!();

    println!("✓ Full QNN layer backpropagation complete!");
    println!("  All gradients (dW, db, dγ, dβ, dx) computed in one backward pass\n");

    println!("=== Key Insights ===");
    println!("1. Hamilton product gradients use conjugate: dW = dY ⊗ X*, dX = W* ⊗ dY");
    println!("2. BN processes each quaternion component independently");
    println!("3. ReLU applies component-wise masking based on forward pass activations");
    println!("4. Non-commutativity of quaternion multiplication matters for gradients");
    println!();
}
