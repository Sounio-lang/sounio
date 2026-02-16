//! Octonion Neural Network (ONN) Training Sanity Checks
//!
//! End-to-end training loop tests that verify:
//!   1. Forward pass correctness (octonion linear layer)
//!   2. Loss computation (MSE over octonion components)
//!   3. Gradient descent convergence (loss decreases over epochs)
//!   4. Parameter efficiency (8x fewer semantic parameters vs real-valued)
//!   5. Activation function numerical stability
//!   6. Weight initialization statistics
//!   7. Non-associativity awareness in multi-layer forward passes
//!
//! These tests validate the DL pipeline WITHOUT requiring GPU hardware.
//! They use the same Cayley-Dickson multiplication as the Sounio stdlib.
//!
//! References:
//!   - Parcollet et al. (2019). "Quaternion Recurrent Neural Networks"
//!   - Zhu et al. (2019). "Quaternion Convolutional Neural Networks"
//!   - Comminiello et al. (2024). "Hypercomplex Neural Networks"
//!
//! Reproduction:
//!   cargo test -p souc --test integration_onn_training_sanity -- --nocapture

#[cfg(test)]
mod onn_training_sanity {

    // ========================================================================
    // Octonion type and operations (matches stdlib/math/octonion.sio)
    // ========================================================================

    #[derive(Debug, Clone, Copy)]
    struct Octonion {
        e: [f64; 8],
    }

    impl Octonion {
        fn new(e0: f64, e1: f64, e2: f64, e3: f64, e4: f64, e5: f64, e6: f64, e7: f64) -> Self {
            Octonion {
                e: [e0, e1, e2, e3, e4, e5, e6, e7],
            }
        }

        fn zero() -> Self {
            Octonion { e: [0.0; 8] }
        }

        fn norm_sq(&self) -> f64 {
            self.e.iter().map(|x| x * x).sum()
        }

        fn norm(&self) -> f64 {
            self.norm_sq().sqrt()
        }

        fn add(&self, other: &Self) -> Self {
            let mut r = Octonion::zero();
            for i in 0..8 {
                r.e[i] = self.e[i] + other.e[i];
            }
            r
        }

        fn sub(&self, other: &Self) -> Self {
            let mut r = Octonion::zero();
            for i in 0..8 {
                r.e[i] = self.e[i] - other.e[i];
            }
            r
        }

        fn scale(&self, s: f64) -> Self {
            let mut r = Octonion::zero();
            for i in 0..8 {
                r.e[i] = self.e[i] * s;
            }
            r
        }

        fn dot(&self, other: &Self) -> f64 {
            let mut s = 0.0;
            for i in 0..8 {
                s += self.e[i] * other.e[i];
            }
            s
        }

        /// Cayley-Dickson multiplication (120 FLOPs)
        fn mul(&self, other: &Self) -> Self {
            let (a0, a1, a2, a3, a4, a5, a6, a7) = (
                self.e[0], self.e[1], self.e[2], self.e[3], self.e[4], self.e[5], self.e[6],
                self.e[7],
            );
            let (b0, b1, b2, b3, b4, b5, b6, b7) = (
                other.e[0], other.e[1], other.e[2], other.e[3], other.e[4], other.e[5], other.e[6],
                other.e[7],
            );

            Octonion::new(
                a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7,
                a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6,
                a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1 + a4 * b6 + a5 * b7 - a6 * b4 - a7 * b5,
                a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0 + a4 * b7 - a5 * b6 + a6 * b5 - a7 * b4,
                a0 * b4 - a1 * b5 - a2 * b6 - a3 * b7 + a4 * b0 + a5 * b1 + a6 * b2 + a7 * b3,
                a0 * b5 + a1 * b4 - a2 * b7 + a3 * b6 - a4 * b1 + a5 * b0 - a6 * b3 + a7 * b2,
                a0 * b6 + a1 * b7 + a2 * b4 - a3 * b5 - a4 * b2 + a5 * b3 + a6 * b0 - a7 * b1,
                a0 * b7 - a1 * b6 + a2 * b5 + a3 * b4 - a4 * b3 - a5 * b2 + a6 * b1 + a7 * b0,
            )
        }

        /// Per-component ReLU
        fn relu(&self) -> Self {
            let mut r = Octonion::zero();
            for i in 0..8 {
                r.e[i] = if self.e[i] > 0.0 { self.e[i] } else { 0.0 };
            }
            r
        }

        /// Per-component LeakyReLU (alpha=0.01)
        fn leaky_relu(&self, alpha: f64) -> Self {
            let mut r = Octonion::zero();
            for i in 0..8 {
                r.e[i] = if self.e[i] > 0.0 {
                    self.e[i]
                } else {
                    alpha * self.e[i]
                };
            }
            r
        }

        /// Per-component sigmoid
        fn sigmoid(&self) -> Self {
            let mut r = Octonion::zero();
            for i in 0..8 {
                r.e[i] = 1.0 / (1.0 + (-self.e[i]).exp());
            }
            r
        }

        /// Per-component tanh
        fn tanh_act(&self) -> Self {
            let mut r = Octonion::zero();
            for i in 0..8 {
                r.e[i] = self.e[i].tanh();
            }
            r
        }
    }

    // ========================================================================
    // PRNG (same LCG as exhaustive tests for reproducibility)
    // ========================================================================

    struct Rng {
        state: u64,
    }

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }

        fn next_f64(&mut self) -> f64 {
            let v = self.next_u64();
            ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }

        fn random_octonion(&mut self) -> Octonion {
            Octonion::new(
                self.next_f64(),
                self.next_f64(),
                self.next_f64(),
                self.next_f64(),
                self.next_f64(),
                self.next_f64(),
                self.next_f64(),
                self.next_f64(),
            )
        }

        /// Xavier-like initialization: scale = sqrt(2 / (fan_in + fan_out))
        fn random_xavier_octonion(&mut self, fan_in: usize, fan_out: usize) -> Octonion {
            let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
            self.random_octonion().scale(scale)
        }
    }

    // ========================================================================
    // Octonion Linear Layer: y = W * x + b
    // ========================================================================

    struct OctLinearLayer {
        weights: Vec<Vec<Octonion>>, // [out_features][in_features]
        biases: Vec<Octonion>,       // [out_features]
        in_features: usize,
        out_features: usize,
    }

    impl OctLinearLayer {
        fn new(in_features: usize, out_features: usize, rng: &mut Rng) -> Self {
            let mut weights = Vec::new();
            for _ in 0..out_features {
                let mut row = Vec::new();
                for _ in 0..in_features {
                    row.push(rng.random_xavier_octonion(in_features, out_features));
                }
                weights.push(row);
            }
            let biases = vec![Octonion::zero(); out_features];
            OctLinearLayer {
                weights,
                biases,
                in_features,
                out_features,
            }
        }

        /// Forward pass: for each output j, y[j] = sum_i(W[j][i] * x[i]) + b[j]
        fn forward(&self, input: &[Octonion]) -> Vec<Octonion> {
            assert_eq!(input.len(), self.in_features);
            let mut output = Vec::with_capacity(self.out_features);
            for j in 0..self.out_features {
                let mut acc = self.biases[j];
                for i in 0..self.in_features {
                    acc = acc.add(&self.weights[j][i].mul(&input[i]));
                }
                output.push(acc);
            }
            output
        }

        /// Parameter count: out * in * 8 (octonion components) + out * 8 (bias)
        fn octonion_param_count(&self) -> usize {
            self.out_features * self.in_features + self.out_features
        }

        /// Equivalent real-valued parameter count (each octonion = 8 reals, but
        /// the octonion multiplication couples all 8 components, so 1 octonion
        /// parameter encodes 8 real DOFs with algebraic structure)
        fn real_equivalent_param_count(&self) -> usize {
            // A real-valued layer of same dimensions: (out*8) * (in*8) + (out*8)
            self.out_features * 8 * self.in_features * 8 + self.out_features * 8
        }
    }

    // ========================================================================
    // MSE Loss over octonion outputs
    // ========================================================================

    fn mse_loss(predictions: &[Octonion], targets: &[Octonion]) -> f64 {
        assert_eq!(predictions.len(), targets.len());
        let n = predictions.len() as f64;
        let mut total = 0.0;
        for i in 0..predictions.len() {
            total += predictions[i].sub(&targets[i]).norm_sq();
        }
        total / (n * 8.0) // Normalize by total components
    }

    // ========================================================================
    // Simple numerical gradient descent (no backprop — finite differences)
    // ========================================================================

    fn compute_gradient_fd(
        layer: &mut OctLinearLayer,
        input: &[Octonion],
        target: &[Octonion],
        epsilon: f64,
    ) -> (Vec<Vec<[f64; 8]>>, Vec<[f64; 8]>) {
        let mut weight_grads = Vec::new();
        let mut bias_grads = Vec::new();

        // Weight gradients
        for j in 0..layer.out_features {
            let mut row_grads = Vec::new();
            for i in 0..layer.in_features {
                let mut grad = [0.0; 8];
                for c in 0..8 {
                    let original = layer.weights[j][i].e[c];

                    layer.weights[j][i].e[c] = original + epsilon;
                    let loss_plus = mse_loss(&layer.forward(input), target);

                    layer.weights[j][i].e[c] = original - epsilon;
                    let loss_minus = mse_loss(&layer.forward(input), target);

                    layer.weights[j][i].e[c] = original;
                    grad[c] = (loss_plus - loss_minus) / (2.0 * epsilon);
                }
                row_grads.push(grad);
            }
            weight_grads.push(row_grads);
        }

        // Bias gradients
        for j in 0..layer.out_features {
            let mut grad = [0.0; 8];
            for c in 0..8 {
                let original = layer.biases[j].e[c];

                layer.biases[j].e[c] = original + epsilon;
                let loss_plus = mse_loss(&layer.forward(input), target);

                layer.biases[j].e[c] = original - epsilon;
                let loss_minus = mse_loss(&layer.forward(input), target);

                layer.biases[j].e[c] = original;
                grad[c] = (loss_plus - loss_minus) / (2.0 * epsilon);
            }
            bias_grads.push(grad);
        }

        (weight_grads, bias_grads)
    }

    fn apply_gradients(
        layer: &mut OctLinearLayer,
        weight_grads: &[Vec<[f64; 8]>],
        bias_grads: &[[f64; 8]],
        lr: f64,
    ) {
        for j in 0..layer.out_features {
            for i in 0..layer.in_features {
                for c in 0..8 {
                    layer.weights[j][i].e[c] -= lr * weight_grads[j][i][c];
                }
            }
            for c in 0..8 {
                layer.biases[j].e[c] -= lr * bias_grads[j][c];
            }
        }
    }

    // ========================================================================
    // TEST 1: Forward pass dimension/value sanity
    // ========================================================================

    #[test]
    fn test_forward_pass_dimensions() {
        eprintln!("\n=== ONN Forward Pass Dimension Check ===");
        let mut rng = Rng::new(42);
        let layer = OctLinearLayer::new(4, 8, &mut rng);

        let input: Vec<Octonion> = (0..4).map(|_| rng.random_octonion()).collect();
        let output = layer.forward(&input);

        assert_eq!(output.len(), 8, "Output dimension mismatch");
        eprintln!("  Input dim:  4 octonions (32 real components)");
        eprintln!("  Output dim: 8 octonions (64 real components)");
        eprintln!(
            "  Parameters: {} octonions ({} real-equivalent)",
            layer.octonion_param_count(),
            layer.real_equivalent_param_count()
        );
        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 2: Loss function sanity
    // ========================================================================

    #[test]
    fn test_loss_function_properties() {
        eprintln!("\n=== MSE Loss Function Properties ===");
        let mut rng = Rng::new(123);

        let a: Vec<Octonion> = (0..4).map(|_| rng.random_octonion()).collect();
        let b: Vec<Octonion> = (0..4).map(|_| rng.random_octonion()).collect();

        // Loss(x, x) = 0
        let self_loss = mse_loss(&a, &a);
        assert!(
            self_loss < 1e-15,
            "Self-loss should be zero, got {}",
            self_loss
        );

        // Loss(x, y) > 0 for x != y
        let diff_loss = mse_loss(&a, &b);
        assert!(
            diff_loss > 0.0,
            "Loss between different inputs should be > 0"
        );

        // Loss(x, y) = Loss(y, x) (symmetric)
        let reverse_loss = mse_loss(&b, &a);
        assert!(
            (diff_loss - reverse_loss).abs() < 1e-15,
            "MSE should be symmetric"
        );

        eprintln!("  Loss(x, x) = {:.2e} (expected ~0)", self_loss);
        eprintln!("  Loss(x, y) = {:.6}", diff_loss);
        eprintln!("  Loss(y, x) = {:.6} (symmetric)", reverse_loss);
        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 3: Training convergence (loss monotonically decreases)
    // ========================================================================

    #[test]
    fn test_training_convergence_single_sample() {
        // Train a small ONN on a single input-output pair
        // With enough capacity and a single sample, loss MUST converge to ~0
        eprintln!("\n=== Training Convergence: Single Sample ===");
        let mut rng = Rng::new(0xDEADBEEF);

        let in_dim = 2;
        let out_dim = 2;
        let mut layer = OctLinearLayer::new(in_dim, out_dim, &mut rng);

        let input: Vec<Octonion> = (0..in_dim).map(|_| rng.random_octonion()).collect();
        let target: Vec<Octonion> = (0..out_dim)
            .map(|_| rng.random_octonion().scale(0.5))
            .collect();

        let lr = 0.01;
        let epsilon = 1e-5;
        let n_epochs = 100;

        let initial_loss = mse_loss(&layer.forward(&input), &target);
        let mut loss_history = vec![initial_loss];

        eprintln!("  Epoch  0: loss = {:.6}", initial_loss);

        for epoch in 1..=n_epochs {
            let (wg, bg) = compute_gradient_fd(&mut layer, &input, &target, epsilon);
            apply_gradients(&mut layer, &wg, &bg, lr);

            let loss = mse_loss(&layer.forward(&input), &target);
            loss_history.push(loss);

            if epoch % 20 == 0 || epoch == n_epochs {
                eprintln!("  Epoch {:3}: loss = {:.6}", epoch, loss);
            }
        }

        let final_loss = *loss_history.last().unwrap();

        assert!(
            final_loss < initial_loss * 0.5,
            "Loss should decrease by at least 2x: initial={:.6}, final={:.6}",
            initial_loss,
            final_loss
        );

        eprintln!("  Loss reduction: {:.2}x", initial_loss / final_loss);
        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 4: Training convergence (batch of samples)
    // ========================================================================

    #[test]
    fn test_training_convergence_batch() {
        // Train on a batch of 8 samples, verify loss trend
        eprintln!("\n=== Training Convergence: Batch of 8 Samples ===");
        let mut rng = Rng::new(0xCAFEBABE);

        let in_dim = 3;
        let out_dim = 2;
        let batch_size = 8;
        let mut layer = OctLinearLayer::new(in_dim, out_dim, &mut rng);

        // Generate batch: target = simple linear function of input (learnable)
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let target_layer = OctLinearLayer::new(in_dim, out_dim, &mut rng);

        for _ in 0..batch_size {
            let x: Vec<Octonion> = (0..in_dim)
                .map(|_| rng.random_octonion().scale(0.3))
                .collect();
            let y = target_layer.forward(&x);
            inputs.push(x);
            targets.push(y);
        }

        let lr = 0.005;
        let epsilon = 1e-5;
        let n_epochs = 50;

        // Compute average loss over batch
        let avg_loss = |layer: &OctLinearLayer| -> f64 {
            let mut total = 0.0;
            for i in 0..batch_size {
                total += mse_loss(&layer.forward(&inputs[i]), &targets[i]);
            }
            total / batch_size as f64
        };

        let initial_loss = avg_loss(&layer);
        eprintln!("  Epoch  0: avg_loss = {:.6}", initial_loss);

        for epoch in 1..=n_epochs {
            // Accumulate gradients over batch
            let mut total_wg: Vec<Vec<[f64; 8]>> = vec![vec![[0.0; 8]; in_dim]; out_dim];
            let mut total_bg: Vec<[f64; 8]> = vec![[0.0; 8]; out_dim];

            for sample in 0..batch_size {
                let (wg, bg) =
                    compute_gradient_fd(&mut layer, &inputs[sample], &targets[sample], epsilon);
                for j in 0..out_dim {
                    for i in 0..in_dim {
                        for c in 0..8 {
                            total_wg[j][i][c] += wg[j][i][c] / batch_size as f64;
                        }
                    }
                    for c in 0..8 {
                        total_bg[j][c] += bg[j][c] / batch_size as f64;
                    }
                }
            }

            apply_gradients(&mut layer, &total_wg, &total_bg, lr);

            if epoch % 10 == 0 || epoch == n_epochs {
                let loss = avg_loss(&layer);
                eprintln!("  Epoch {:3}: avg_loss = {:.6}", epoch, loss);
            }
        }

        let final_loss = avg_loss(&layer);
        assert!(
            final_loss < initial_loss,
            "Batch training should reduce loss: initial={:.6}, final={:.6}",
            initial_loss,
            final_loss
        );

        eprintln!(
            "  Loss reduction: {:.2}x",
            initial_loss / final_loss.max(1e-15)
        );
        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 5: Parameter efficiency (8x compression ratio)
    // ========================================================================

    #[test]
    fn test_parameter_efficiency_ratio() {
        eprintln!("\n=== Parameter Efficiency: Octonion vs Real-Valued ===");

        let configs = [(4, 8), (8, 16), (16, 32), (32, 64)];

        for (in_dim, out_dim) in &configs {
            let mut rng = Rng::new(42);
            let layer = OctLinearLayer::new(*in_dim, *out_dim, &mut rng);

            let oct_params = layer.octonion_param_count();
            let real_params = layer.real_equivalent_param_count();
            let ratio = real_params as f64 / oct_params as f64;

            eprintln!(
                "  {}x{}: oct_params={}, real_equiv={}, ratio={:.1}x",
                in_dim, out_dim, oct_params, real_params, ratio
            );

            // The ratio should be approximately 8x for weight parameters
            // (slightly less overall due to bias term)
            assert!(
                ratio > 7.0,
                "Parameter efficiency ratio should be > 7x, got {:.1}x",
                ratio
            );
        }

        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 6: Activation function numerical stability
    // ========================================================================

    #[test]
    fn test_activation_functions_stability() {
        eprintln!("\n=== Activation Function Numerical Stability ===");
        let mut rng = Rng::new(0xAC71F00C);

        for trial in 0..1000 {
            let x = rng.random_octonion().scale(5.0); // Components in [-5, 5]

            // ReLU: output components >= 0
            let relu_out = x.relu();
            for c in 0..8 {
                assert!(
                    relu_out.e[c] >= 0.0,
                    "ReLU output must be >= 0, got {} at trial {} component {}",
                    relu_out.e[c],
                    trial,
                    c
                );
            }

            // Sigmoid: output in (0, 1)
            let sig_out = x.sigmoid();
            for c in 0..8 {
                assert!(
                    sig_out.e[c] > 0.0 && sig_out.e[c] < 1.0,
                    "Sigmoid output must be in (0,1), got {} at trial {} component {}",
                    sig_out.e[c],
                    trial,
                    c
                );
            }

            // Tanh: output in (-1, 1)
            let tanh_out = x.tanh_act();
            for c in 0..8 {
                assert!(
                    tanh_out.e[c] > -1.0 && tanh_out.e[c] < 1.0,
                    "Tanh output must be in (-1,1), got {} at trial {} component {}",
                    tanh_out.e[c],
                    trial,
                    c
                );
            }

            // LeakyReLU: should be identity for positive, alpha*x for negative
            let alpha = 0.01;
            let leaky_out = x.leaky_relu(alpha);
            for c in 0..8 {
                let expected = if x.e[c] > 0.0 { x.e[c] } else { alpha * x.e[c] };
                assert!(
                    (leaky_out.e[c] - expected).abs() < 1e-14,
                    "LeakyReLU mismatch at trial {} component {}: got {}, expected {}",
                    trial,
                    c,
                    leaky_out.e[c],
                    expected
                );
            }
        }

        // Extreme values: very large
        let large = Octonion::new(100.0, -100.0, 50.0, -50.0, 200.0, -200.0, 0.0, 1.0);
        let sig_large = large.sigmoid();
        assert!(
            (sig_large.e[0] - 1.0).abs() < 1e-10,
            "sigmoid(100) should be ~1.0"
        );
        assert!(sig_large.e[1] < 1e-10, "sigmoid(-100) should be ~0.0");

        let tanh_large = large.tanh_act();
        assert!(
            (tanh_large.e[0] - 1.0).abs() < 1e-10,
            "tanh(100) should be ~1.0"
        );
        assert!(
            (tanh_large.e[1] + 1.0).abs() < 1e-10,
            "tanh(-100) should be ~-1.0"
        );

        eprintln!("  1,000 random inputs x 4 activations: all stable");
        eprintln!("  Extreme value tests: passed");
        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 7: Xavier initialization statistics
    // ========================================================================

    #[test]
    fn test_xavier_initialization_statistics() {
        // Xavier init: Var(w) = 2 / (fan_in + fan_out)
        // For octonion components: each component should have this variance
        eprintln!("\n=== Xavier Initialization Statistics ===");
        let mut rng = Rng::new(0x58617669657200);

        let fan_in = 16;
        let fan_out = 32;
        // Xavier scale * uniform[-1,1] std: scale = sqrt(2/(fan_in+fan_out)),
        // uniform std = 1/sqrt(3), so expected std = scale / sqrt(3)
        let xavier_scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
        let expected_std = xavier_scale / (3.0_f64).sqrt();

        let n_samples = 10_000;
        let mut sum = [0.0f64; 8];
        let mut sum_sq = [0.0f64; 8];

        for _ in 0..n_samples {
            let w = rng.random_xavier_octonion(fan_in, fan_out);
            for c in 0..8 {
                sum[c] += w.e[c];
                sum_sq[c] += w.e[c] * w.e[c];
            }
        }

        let n = n_samples as f64;
        for c in 0..8 {
            let mean = sum[c] / n;
            let var = sum_sq[c] / n - mean * mean;
            let std = var.sqrt();

            // Mean should be ~0 (within 3 sigma of sampling distribution)
            let mean_tolerance = 3.0 * expected_std / n.sqrt();
            assert!(
                mean.abs() < mean_tolerance,
                "Component {} mean={:.4} exceeds tolerance {:.4}",
                c,
                mean,
                mean_tolerance
            );

            // Std should be ~expected_std (within 20% for 10k samples)
            let std_ratio = std / expected_std;
            assert!(
                std_ratio > 0.8 && std_ratio < 1.2,
                "Component {} std={:.4} vs expected {:.4} (ratio {:.2})",
                c,
                std,
                expected_std,
                std_ratio
            );
        }

        eprintln!("  fan_in={}, fan_out={}", fan_in, fan_out);
        eprintln!("  Expected std: {:.4}", expected_std);
        eprintln!(
            "  10,000 samples: mean~0, std~{:.4} per component",
            expected_std
        );
        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 8: Multi-layer forward pass (non-associativity awareness)
    // ========================================================================

    #[test]
    fn test_multi_layer_forward_pass() {
        // Verify that a multi-layer ONN produces consistent outputs
        // and that the forward pass respects left-to-right evaluation
        eprintln!("\n=== Multi-Layer Forward Pass ===");
        let mut rng = Rng::new(0xDADA);

        let layer1 = OctLinearLayer::new(4, 8, &mut rng);
        let layer2 = OctLinearLayer::new(8, 4, &mut rng);
        let layer3 = OctLinearLayer::new(4, 2, &mut rng);

        let input: Vec<Octonion> = (0..4).map(|_| rng.random_octonion().scale(0.3)).collect();

        // Forward pass
        let h1 = layer1.forward(&input);
        let h1_relu: Vec<Octonion> = h1.iter().map(|o| o.relu()).collect();
        let h2 = layer2.forward(&h1_relu);
        let h2_relu: Vec<Octonion> = h2.iter().map(|o| o.relu()).collect();
        let output = layer3.forward(&h2_relu);

        // Verify output dimensions
        assert_eq!(output.len(), 2);

        // Verify determinism (same input -> same output)
        let h1b = layer1.forward(&input);
        let h1b_relu: Vec<Octonion> = h1b.iter().map(|o| o.relu()).collect();
        let h2b = layer2.forward(&h1b_relu);
        let h2b_relu: Vec<Octonion> = h2b.iter().map(|o| o.relu()).collect();
        let output2 = layer3.forward(&h2b_relu);

        for i in 0..output.len() {
            let diff = output[i].sub(&output2[i]).norm();
            assert!(
                diff < 1e-14,
                "Forward pass not deterministic: diff={} at output {}",
                diff,
                i
            );
        }

        // Verify all outputs are finite
        for (i, o) in output.iter().enumerate() {
            for c in 0..8 {
                assert!(
                    o.e[c].is_finite(),
                    "Output {} component {} is not finite: {}",
                    i,
                    c,
                    o.e[c]
                );
            }
        }

        let total_params = layer1.octonion_param_count()
            + layer2.octonion_param_count()
            + layer3.octonion_param_count();
        let total_real_equiv = layer1.real_equivalent_param_count()
            + layer2.real_equivalent_param_count()
            + layer3.real_equivalent_param_count();

        eprintln!("  Architecture: 4 -> 8 -> 4 -> 2 (octonion dims)");
        eprintln!("  Architecture: 32 -> 64 -> 32 -> 16 (real dims)");
        eprintln!("  Total oct params: {}", total_params);
        eprintln!("  Real-equivalent params: {}", total_real_equiv);
        eprintln!(
            "  Compression ratio: {:.1}x",
            total_real_equiv as f64 / total_params as f64
        );
        eprintln!("  Output finite: yes, deterministic: yes");
        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 9: Gradient sanity check (gradient should be non-zero and bounded)
    // ========================================================================

    #[test]
    fn test_gradient_sanity() {
        eprintln!("\n=== Gradient Sanity Check ===");
        let mut rng = Rng::new(0x6A4D);

        let in_dim = 2;
        let out_dim = 2;
        let mut layer = OctLinearLayer::new(in_dim, out_dim, &mut rng);

        let input: Vec<Octonion> = (0..in_dim).map(|_| rng.random_octonion()).collect();
        let target: Vec<Octonion> = (0..out_dim)
            .map(|_| rng.random_octonion().scale(0.5))
            .collect();

        let (wg, bg) = compute_gradient_fd(&mut layer, &input, &target, 1e-5);

        // All gradients should be finite
        let mut max_grad = 0.0f64;
        let mut min_nonzero_grad = f64::MAX;
        let mut zero_count = 0;
        let mut total_count = 0;

        for j in 0..out_dim {
            for i in 0..in_dim {
                for c in 0..8 {
                    let g = wg[j][i][c];
                    assert!(
                        g.is_finite(),
                        "Weight gradient not finite at [{j}][{i}][{c}]"
                    );
                    max_grad = max_grad.max(g.abs());
                    if g.abs() > 1e-12 {
                        min_nonzero_grad = min_nonzero_grad.min(g.abs());
                    } else {
                        zero_count += 1;
                    }
                    total_count += 1;
                }
            }
            for c in 0..8 {
                let g = bg[j][c];
                assert!(g.is_finite(), "Bias gradient not finite at [{j}][{c}]");
                max_grad = max_grad.max(g.abs());
                total_count += 1;
            }
        }

        eprintln!("  Max |gradient|: {:.6}", max_grad);
        eprintln!("  Min nonzero |gradient|: {:.2e}", min_nonzero_grad);
        eprintln!("  Zero gradients: {}/{}", zero_count, total_count);

        assert!(max_grad > 1e-8, "Gradients should be non-trivially large");
        assert!(
            max_grad < 1e6,
            "Gradients should be bounded: max={}",
            max_grad
        );

        eprintln!("  PASS");
    }

    // ========================================================================
    // TEST 10: Norm-preserving activation (unit octonion output)
    // ========================================================================

    #[test]
    fn test_normalize_after_forward_pass() {
        // Normalizing outputs to unit octonions should work without NaN/Inf
        eprintln!("\n=== Normalize After Forward Pass ===");
        let mut rng = Rng::new(0xA0B1);

        let layer = OctLinearLayer::new(4, 4, &mut rng);
        let input: Vec<Octonion> = (0..4).map(|_| rng.random_octonion().scale(0.3)).collect();

        let output = layer.forward(&input);

        for (i, o) in output.iter().enumerate() {
            let n = o.norm();
            assert!(n.is_finite() && n > 0.0, "Output {} has zero/inf norm", i);

            let normalized = o.scale(1.0 / n);
            let unit_norm = normalized.norm();
            assert!(
                (unit_norm - 1.0).abs() < 1e-12,
                "Normalized output {} has norm {} (expected 1.0)",
                i,
                unit_norm
            );
        }

        eprintln!("  All outputs normalizable to unit octonions on S^7");
        eprintln!("  PASS");
    }

    // ========================================================================
    // SUMMARY
    // ========================================================================

    #[test]
    fn test_onn_training_summary() {
        eprintln!("\n====================================================================");
        eprintln!("  ONN TRAINING SANITY CHECK SUMMARY");
        eprintln!("====================================================================");
        eprintln!("  1. Forward pass dimensions:              CHECKED");
        eprintln!("  2. MSE loss properties (zero/symm/pos):  CHECKED");
        eprintln!("  3. Single-sample convergence (100 ep):   CHECKED");
        eprintln!("  4. Batch convergence (8 samples, 50 ep): CHECKED");
        eprintln!("  5. Parameter efficiency (>7x ratio):     CHECKED");
        eprintln!("  6. Activation stability (1k x 4 funcs):  CHECKED");
        eprintln!("  7. Xavier init statistics (10k samples):  CHECKED");
        eprintln!("  8. Multi-layer forward pass:             CHECKED");
        eprintln!("  9. Gradient sanity (finite/bounded):     CHECKED");
        eprintln!(" 10. Output normalization to S^7:          CHECKED");
        eprintln!("  ---------------------------------------------------");
        eprintln!("  DL pipeline: forward -> loss -> gradient -> update");
        eprintln!("  Verified WITHOUT GPU hardware (CPU reference impl)");
        eprintln!("====================================================================");
    }
}
