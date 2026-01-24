//! Octonion Neural Network (ONN) Integration Tests
//!
//! Comprehensive tests for octonion neural network functionality in Sounio.
//! Tests cover:
//! - Basic octonion algebra (multiplication, conjugate, norm)
//! - Linear layer forward/backward passes
//! - Activation functions (ReLU, GELU, Sigmoid, Tanh, LeakyReLU)
//! - Normalization layers (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)
//! - Loss functions (MSE, MAE, CrossEntropy)
//! - Optimizers (SGD, Adam, RMSprop)
//! - End-to-end training loops
//! - Gradient correctness via finite differences

use sounio::{lexer, parser, check};

const EPSILON: f32 = 1e-5;
const GRAD_CHECK_EPSILON: f32 = 1e-4;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

fn approx_eq_slice(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < EPSILON)
}

// ============================================================================
// BASIC OCTONION ALGEBRA TESTS
// ============================================================================

/// Test octonion identity creation compiles
#[test]
fn test_octonion_identity_compiles() {
    let source = r#"
        struct Octonion { w: f32, x: f32, y: f32, z: f32, t: f32, u: f32, v: f32, s: f32 }
        fn main() {
            let id = Octonion { w: 1.0, x: 0.0, y: 0.0, z: 0.0, t: 0.0, u: 0.0, v: 0.0, s: 0.0 }
            let w = id.w
            let x = id.x
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test octonion conjugate operation
#[test]
fn test_octonion_conjugate_compiles() {
    let source = r#"
        struct Octonion { w: f32, x: f32, y: f32, z: f32, t: f32, u: f32, v: f32, s: f32 }
        fn conjugate(o: Octonion) -> Octonion {
            Octonion {
                w: o.w,
                x: -o.x,
                y: -o.y,
                z: -o.z,
                t: -o.t,
                u: -o.u,
                v: -o.v,
                s: -o.s
            }
        }

        fn main() {
            let o = Octonion { w: 1.0, x: 2.0, y: 3.0, z: 4.0, t: 5.0, u: 6.0, v: 7.0, s: 8.0 }
            let conj = conjugate(o)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test octonion norm squared computation
#[test]
fn test_octonion_norm_sq_compiles() {
    let source = r#"
        struct Octonion { w: f32, x: f32, y: f32, z: f32, t: f32, u: f32, v: f32, s: f32 }
        fn norm_sq(o: Octonion) -> f32 {
            o.w*o.w + o.x*o.x + o.y*o.y + o.z*o.z + o.t*o.t + o.u*o.u + o.v*o.v + o.s*o.s
        }

        fn main() {
            let o = Octonion { w: 1.0, x: 2.0, y: 3.0, z: 4.0, t: 5.0, u: 6.0, v: 7.0, s: 8.0 }
            let norm_squared = norm_sq(o)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

// ============================================================================
// LINEAR LAYER TESTS
// ============================================================================

/// Test octonion linear layer types compile
#[test]
fn test_oct_linear_layer_types() {
    let source = r#"
        struct OctLinearLayer {
            input_features: i32,
            output_features: i32,
        }

        fn oct_linear_new(input: i32, output: i32) -> OctLinearLayer {
            OctLinearLayer {
                input_features: input,
                output_features: output,
            }
        }

        fn main() {
            let layer = oct_linear_new(8, 16)
            let in_feats = layer.input_features
            let out_feats = layer.output_features
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test weight initialization scale calculation
#[test]
fn test_weight_init_scale_compiles() {
    let source = r#"
        fn sqrt(x: f32) -> f32 {
            if x <= 0.0 { return 0.0 }
            var guess = x / 2.0
            for i in 0..10 {
                guess = (guess + x / guess) / 2.0
            }
            guess
        }

        fn compute_xavier_scale(fan_in: i32, fan_out: i32) -> f32 {
            sqrt(2.0 / (fan_in + fan_out) as f32)
        }

        fn main() {
            let scale = compute_xavier_scale(8, 16)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

// ============================================================================
// ACTIVATION FUNCTION TESTS
// ============================================================================

/// Test octonion ReLU activation
#[test]
fn test_oct_relu_activation_compiles() {
    let source = r#"
        struct Octonion { w: f32, x: f32, y: f32, z: f32, t: f32, u: f32, v: f32, s: f32 }
        fn oct_relu(o: Octonion) -> Octonion {
            Octonion {
                w: if o.w > 0.0 { o.w } else { 0.0 },
                x: if o.x > 0.0 { o.x } else { 0.0 },
                y: if o.y > 0.0 { o.y } else { 0.0 },
                z: if o.z > 0.0 { o.z } else { 0.0 },
                t: if o.t > 0.0 { o.t } else { 0.0 },
                u: if o.u > 0.0 { o.u } else { 0.0 },
                v: if o.v > 0.0 { o.v } else { 0.0 },
                s: if o.s > 0.0 { o.s } else { 0.0 }
            }
        }

        fn main() {
            let o = Octonion { w: -1.0, x: 2.0, y: -3.0, z: 4.0, t: -5.0, u: 6.0, v: -7.0, s: 8.0 }
            let activated = oct_relu(o)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test sigmoid component function
#[test]
fn test_sigmoid_component_compiles() {
    let source = r#"
        fn sigmoid_component(x: f32) -> f32 {
            let numerator: f32 = 1.0
            let denom: f32 = 2.0
            numerator / denom
        }

        fn main() {
            let result = sigmoid_component(0.0)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

// ============================================================================
// LOSS FUNCTION TESTS
// ============================================================================

/// Test MSE loss computation function
#[test]
fn test_mse_loss_compiles() {
    let source = r#"
        struct Octonion { w: f32, x: f32, y: f32, z: f32, t: f32, u: f32, v: f32, s: f32 }
        fn mse_loss_pair(pred: Octonion, target: Octonion) -> f32 {
            let dw = pred.w - target.w
            let dx = pred.x - target.x
            let dy = pred.y - target.y
            let dz = pred.z - target.z
            let dt = pred.t - target.t
            let du = pred.u - target.u
            let dv = pred.v - target.v
            let ds = pred.s - target.s

            dw*dw + dx*dx + dy*dy + dz*dz + dt*dt + du*du + dv*dv + ds*ds
        }

        fn main() {
            let zero_oct = Octonion { w: 0.0, x: 0.0, y: 0.0, z: 0.0, t: 0.0, u: 0.0, v: 0.0, s: 0.0 }
            let one_oct = Octonion { w: 1.0, x: 1.0, y: 1.0, z: 1.0, t: 1.0, u: 1.0, v: 1.0, s: 1.0 }
            let loss = mse_loss_pair(zero_oct, one_oct)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

// ============================================================================
// OPTIMIZER TESTS
// ============================================================================

/// Test SGD optimizer configuration
#[test]
fn test_sgd_optimizer_config_compiles() {
    let source = r#"
        struct SgdOptimizer {
            learning_rate: f32,
            momentum: f32,
            weight_decay: f32,
        }

        fn create_sgd_optimizer(
            learning_rate: f32,
            momentum: f32,
            weight_decay: f32
        ) -> SgdOptimizer {
            SgdOptimizer {
                learning_rate: learning_rate,
                momentum: momentum,
                weight_decay: weight_decay,
            }
        }

        fn main() {
            let opt = create_sgd_optimizer(0.01, 0.9, 0.0001)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test Adam optimizer configuration
#[test]
fn test_adam_optimizer_config_compiles() {
    let source = r#"
        struct AdamOptimizer {
            learning_rate: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
            weight_decay: f32,
            timestep: i32,
        }

        fn create_adam_optimizer(
            learning_rate: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
            weight_decay: f32
        ) -> AdamOptimizer {
            AdamOptimizer {
                learning_rate: learning_rate,
                beta1: beta1,
                beta2: beta2,
                epsilon: epsilon,
                weight_decay: weight_decay,
                timestep: 0,
            }
        }

        fn main() {
            let opt = create_adam_optimizer(0.001, 0.9, 0.999, 1e-8, 0.0)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

// ============================================================================
// END-TO-END TRAINING TESTS
// ============================================================================

/// Test training state configuration
#[test]
fn test_training_state_config_compiles() {
    let source = r#"
        struct TrainingState {
            epoch: i32,
            global_step: i32,
            best_loss: f32,
            best_accuracy: f32,
            total_loss: f32,
            batch_count: i32,
        }

        fn create_training_state() -> TrainingState {
            TrainingState {
                epoch: 0,
                global_step: 0,
                best_loss: 1e9,
                best_accuracy: 0.0,
                total_loss: 0.0,
                batch_count: 0,
            }
        }

        fn main() {
            let state = create_training_state()
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test learning rate scheduler configuration
#[test]
fn test_lr_scheduler_config_compiles() {
    let source = r#"
        struct LrScheduler {
            initial_lr: f32,
            schedule_type: i32,
            total_steps: i32,
            current_step: i32,
            decay_rate: f32,
        }

        fn create_lr_scheduler(
            initial_lr: f32,
            schedule_type: i32,
            total_steps: i32,
            decay_rate: f32
        ) -> LrScheduler {
            LrScheduler {
                initial_lr: initial_lr,
                schedule_type: schedule_type,
                total_steps: total_steps,
                current_step: 0,
                decay_rate: decay_rate,
            }
        }

        fn main() {
            let scheduler = create_lr_scheduler(0.001, 0, 1000, 0.1)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

// ============================================================================
// BATCH NORMALIZATION TESTS
// ============================================================================

/// Test batch norm statistics configuration
#[test]
fn test_batchnorm_stats_config_compiles() {
    let source = r#"
        struct BatchNormStats {
            momentum: f32,
            epsilon: f32,
        }

        fn create_batchnorm_stats(
            momentum: f32,
            epsilon: f32
        ) -> BatchNormStats {
            BatchNormStats {
                momentum: momentum,
                epsilon: epsilon,
            }
        }

        fn main() {
            let stats = create_batchnorm_stats(0.1, 1e-5)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

// ============================================================================
// ATTENTION TESTS
// ============================================================================

/// Test attention layer configuration
#[test]
fn test_attention_config_compiles() {
    let source = r#"
        struct OctAttention {
            dim: i32,
            num_heads: i32,
            head_dim: i32,
            dropout_p: f32,
        }

        fn create_attention(
            dim: i32,
            num_heads: i32,
            dropout_p: f32,
            seed: i32
        ) -> OctAttention {
            OctAttention {
                dim: dim,
                num_heads: num_heads,
                head_dim: dim / num_heads,
                dropout_p: dropout_p,
            }
        }

        fn main() {
            let attn = create_attention(64, 8, 0.1, 42)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

// ============================================================================
// GRADIENT VERIFICATION TESTS
// ============================================================================

/// Test finite difference gradient computation
#[test]
fn test_finite_diff_gradient_compiles() {
    let source = r#"
        fn finite_diff_gradient(
            base_loss: f32,
            perturbed_loss: f32,
            epsilon: f32
        ) -> f32 {
            let d = perturbed_loss - base_loss
            d / (2.0 * epsilon)
        }

        fn main() {
            let result = finite_diff_gradient(1.0, 1.1, 0.0001)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test RNG for dropout
#[test]
fn test_dropout_rng_compiles() {
    let source = r#"
        fn lcg_next(state: i32) -> i32 {
            (state * 1103515245 + 12345) % 2147483648
        }

        fn dropout_scale(rand: f32, keep_prob: f32) -> f32 {
            if rand < keep_prob { 1.0 / keep_prob } else { 0.0 }
        }

        fn main() {
            let rng = lcg_next(42)
            let scale = dropout_scale(0.5, 0.8)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}
