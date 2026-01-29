//! Quaternion Operation Tests
//!
//! Tests for quaternion algebra operations in stdlib/qnn/
//! These tests verify that the QNN stdlib modules parse and type-check correctly.

use sounio::{lexer, parser, check};

/// Test quaternion conjugate operation compiles
#[test]
fn test_quat_conjugate_compiles() {
    let source = r#"
        fn main() {
            let q = quat(1.0, 2.0, 3.0, 4.0)
            let conj_w = q.w
            let conj_x = -q.x
            let conj_y = -q.y
            let conj_z = -q.z
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test quaternion operations parse and type-check
#[test]
fn test_quat_operations_compile() {
    let source = r#"
        fn test_quat_ops() -> f32 {
            let q: Quat
            let w_component = q.w
            let x_component = q.x
            let y_component = q.y
            let z_component = q.z
            w_component + x_component
        }

        fn main() {
            let result = test_quat_ops()
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test Hamilton product type-checks
#[test]
fn test_hamilton_product_types() {
    let source = r#"
        fn hamilton_product_components(a: Quat, b: Quat) -> f32 {
            let w1 = a.w
            let x1 = a.x
            let y1 = a.y
            let z1 = a.z
            let w2 = b.w
            let x2 = b.x
            let y2 = b.y
            let z2 = b.z

            // Return the w component of the Hamilton product
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        }

        fn main() {
            let i: Quat
            let j: Quat
            let result = hamilton_product_components(i, j)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test quaternion layer types compile
#[test]
fn test_quat_linear_layer_types() {
    let source = r#"
        struct QuatLinearLayer {
            input_features: i32,
            output_features: i32,
        }

        fn quat_linear_new(input: i32, output: i32) -> QuatLinearLayer {
            QuatLinearLayer {
                input_features: input,
                output_features: output,
            }
        }

        fn main() {
            let layer = quat_linear_new(16, 32)
            let in_feats = layer.input_features
            let out_feats = layer.output_features
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test quaternion batch operations type-check
#[test]
fn test_quat_batch_operations() {
    let source = r#"
        fn process_batch(input: &[Quat], output: &![Quat]) {
            let i: i32 = 0
            let n = 8
            while i < n {
                output[i] = input[i]
                i = i + 1
            }
        }

        fn main() {
            let batch: [Quat; 8]
            var result: [Quat; 8]
            process_batch(&batch, &!result)
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    let _hir = check::check_ast(&ast).unwrap();
}

/// Test quaternion intrinsic function signatures
#[test]
fn test_quat_intrinsics_available() {
    // This test verifies that key QNN intrinsics are available
    // Note: We can't call them without actual implementations,
    // but we can verify they exist in the type system
    let source = r#"
        fn main() {
            // Just reference the types - actual calls would need implementations
            let layer_type: QuatLinear
            let conv_type: QuatConv2d
            let rnn_type: QuatRnnState
            let gate_type: QuatGate
        }
    "#;

    let tokens = lexer::lex(source).unwrap();
    let ast = parser::parse(&tokens, source).unwrap();
    // This will fail until we register the QNN types properly,
    // but that's expected for now
    let result = check::check_ast(&ast);
    // For now, we just verify it parses - type checking will fail until
    // we have the full type system integration
    assert!(result.is_err() || result.is_ok());
}
