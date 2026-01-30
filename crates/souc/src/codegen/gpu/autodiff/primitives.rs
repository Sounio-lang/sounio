//! Backward Rules for Differentiable GPU Primitives
//!
//! This module defines the backward (adjoint) rules for each differentiable primitive.
//! These rules implement the chain rule for reverse-mode automatic differentiation.
//!
//! # Notation
//!
//! For a function z = f(x, y):
//! - `dz` is the incoming gradient (adjoint) from upstream
//! - `dx` is the gradient contribution to x: dx = dz * dz/dx
//! - `dy` is the gradient contribution to y: dy = dz * dz/dy
//!
//! # Gradient Accumulation
//!
//! When a value is used multiple times, gradients must be accumulated:
//! ```text
//! If z1 = f(x) and z2 = g(x), then:
//! dx = dz1 * df/dx + dz2 * dg/dx
//! ```

use std::collections::HashMap;

use super::super::ir::GpuOp;
use super::tape::TapeOp;

/// Differentiable primitive operation (subset of GpuOp that supports differentiation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffPrimitive {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Neg,

    // Math functions
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Pow,
    Rsqrt,
    Abs,

    // Activation functions
    Relu,
    Sigmoid,
    Tanh,
    Gelu,

    // Reductions
    Sum,
    Mean,
    Max,
    Min,

    // Matrix operations
    MatMul,
    Transpose,
    BatchMatMul,

    // Fused operations
    Fma,

    // Quantization operations (straight-through estimator)
    FakeQuantize,
    FakeQuantizePerChannel,
    FakeQuantizeQuat,
}

impl DiffPrimitive {
    /// Convert to TapeOp
    pub fn to_tape_op(&self) -> TapeOp {
        match self {
            DiffPrimitive::Add => TapeOp::Add,
            DiffPrimitive::Sub => TapeOp::Sub,
            DiffPrimitive::Mul => TapeOp::Mul,
            DiffPrimitive::Div => TapeOp::Div,
            DiffPrimitive::Neg => TapeOp::Neg,
            DiffPrimitive::Sin => TapeOp::Sin,
            DiffPrimitive::Cos => TapeOp::Cos,
            DiffPrimitive::Tan => TapeOp::Tan,
            DiffPrimitive::Exp => TapeOp::Exp,
            DiffPrimitive::Log => TapeOp::Log,
            DiffPrimitive::Sqrt => TapeOp::Sqrt,
            DiffPrimitive::Pow => TapeOp::Pow,
            DiffPrimitive::Rsqrt => TapeOp::Rsqrt,
            DiffPrimitive::Abs => TapeOp::Abs,
            DiffPrimitive::Relu => TapeOp::Relu,
            DiffPrimitive::Sigmoid => TapeOp::Sigmoid,
            DiffPrimitive::Tanh => TapeOp::Tanh,
            DiffPrimitive::Gelu => TapeOp::Gelu,
            DiffPrimitive::Sum => TapeOp::Sum,
            DiffPrimitive::Mean => TapeOp::Mean,
            DiffPrimitive::Max => TapeOp::Max,
            DiffPrimitive::Min => TapeOp::Min,
            DiffPrimitive::MatMul => TapeOp::MatMul,
            DiffPrimitive::Transpose => TapeOp::Transpose,
            DiffPrimitive::BatchMatMul => TapeOp::BatchMatMul,
            DiffPrimitive::Fma => TapeOp::Fma,
            DiffPrimitive::FakeQuantize => TapeOp::FakeQuantize,
            DiffPrimitive::FakeQuantizePerChannel => TapeOp::FakeQuantizePerChannel,
            DiffPrimitive::FakeQuantizeQuat => TapeOp::FakeQuantizeQuat,
        }
    }
}

/// A backward rule specifies how to compute gradients for an operation
#[derive(Debug, Clone)]
pub struct BackwardRule {
    /// Operation this rule applies to
    pub op: DiffPrimitive,
    /// Number of input operands
    pub num_inputs: usize,
    /// Description of the backward computation
    pub description: &'static str,
    /// GPU operations to generate for the backward pass
    /// Format: (result_name, operation_expr)
    pub backward_ops: Vec<BackwardOp>,
}

/// A single backward operation
#[derive(Debug, Clone)]
pub enum BackwardOp {
    /// dx = dz (gradient passes through unchanged)
    PassThrough { input_idx: usize },

    /// dx = dz * constant
    Scale { input_idx: usize, scale: f64 },

    /// dx = -dz (negate gradient)
    Negate { input_idx: usize },

    /// dx = dz * other_input (for multiplication)
    MulByInput {
        input_idx: usize,
        other_input_idx: usize,
    },

    /// dx = dz * f(inputs) where f is a custom expression
    Custom {
        input_idx: usize,
        expr: BackwardExpr,
    },

    /// dx = dz / input (for division numerator)
    DivByInput {
        input_idx: usize,
        divisor_idx: usize,
    },

    /// dx = -dz * numerator / (input^2) (for division denominator)
    DivDenomGrad {
        input_idx: usize,
        numerator_idx: usize,
    },
}

/// Expression for custom backward computation
#[derive(Debug, Clone)]
pub enum BackwardExpr {
    /// cos(x) - for d/dx sin(x)
    Cos(usize),
    /// -sin(x) - for d/dx cos(x)
    NegSin(usize),
    /// 1/cos^2(x) - for d/dx tan(x)
    SecSquared(usize),
    /// exp(x) - for d/dx exp(x)
    Exp(usize),
    /// 1/x - for d/dx log(x)
    Reciprocal(usize),
    /// 1/(2*sqrt(x)) - for d/dx sqrt(x)
    HalfRsqrt(usize),
    /// n * x^(n-1) - for d/dx x^n
    PowGrad { base_idx: usize, exponent: f64 },
    /// -1/(2*x^(3/2)) - for d/dx rsqrt(x)
    RsqrtGrad(usize),
    /// sign(x) - for d/dx abs(x)
    Sign(usize),
    /// step(x) - for d/dx relu(x)
    Step(usize),
    /// sigmoid(x) * (1 - sigmoid(x)) - for d/dx sigmoid(x)
    SigmoidGrad(usize),
    /// 1 - tanh^2(x) - for d/dx tanh(x)
    TanhGrad(usize),
    /// Complex GELU gradient
    GeluGrad(usize),
    /// Broadcast (for reduction backward)
    Broadcast { shape: Vec<u32> },
    /// Indicator (1 where max/min occurred)
    Indicator {
        input_idx: usize,
        reduction_result_idx: usize,
    },
    /// 1/n (for mean gradient)
    MeanScale { count: usize },
    /// Matrix transpose (for matmul gradient)
    Transpose(usize),
    /// Matrix multiply for gradient computation
    MatMulGrad {
        grad_idx: usize,
        other_idx: usize,
        transpose_first: bool,
        transpose_second: bool,
    },
}

/// Registry of backward rules for all supported primitives
#[derive(Debug, Clone)]
pub struct PrimitiveRegistry {
    /// Map from primitive to backward rule
    rules: HashMap<DiffPrimitive, BackwardRule>,
}

impl PrimitiveRegistry {
    /// Create a new registry with all standard backward rules
    pub fn new() -> Self {
        let mut registry = Self {
            rules: HashMap::new(),
        };
        registry.register_all_rules();
        registry
    }

    /// Register all standard backward rules
    fn register_all_rules(&mut self) {
        // Arithmetic operations
        self.register_add();
        self.register_sub();
        self.register_mul();
        self.register_div();
        self.register_neg();

        // Math functions
        self.register_sin();
        self.register_cos();
        self.register_tan();
        self.register_exp();
        self.register_log();
        self.register_sqrt();
        self.register_pow();
        self.register_rsqrt();
        self.register_abs();

        // Activation functions
        self.register_relu();
        self.register_sigmoid();
        self.register_tanh();
        self.register_gelu();

        // Reductions
        self.register_sum();
        self.register_mean();
        self.register_max();
        self.register_min();

        // Matrix operations
        self.register_matmul();
        self.register_transpose();
        self.register_batch_matmul();

        // Fused operations
        self.register_fma();

        // Quantization operations (straight-through estimator)
        self.register_fake_quantize();
        self.register_fake_quantize_per_channel();
        self.register_fake_quantize_quat();
    }

    // === Arithmetic Operations ===

    /// z = x + y => dx = dz, dy = dz
    fn register_add(&mut self) {
        self.rules.insert(
            DiffPrimitive::Add,
            BackwardRule {
                op: DiffPrimitive::Add,
                num_inputs: 2,
                description: "dx = dz, dy = dz",
                backward_ops: vec![
                    BackwardOp::PassThrough { input_idx: 0 },
                    BackwardOp::PassThrough { input_idx: 1 },
                ],
            },
        );
    }

    /// z = x - y => dx = dz, dy = -dz
    fn register_sub(&mut self) {
        self.rules.insert(
            DiffPrimitive::Sub,
            BackwardRule {
                op: DiffPrimitive::Sub,
                num_inputs: 2,
                description: "dx = dz, dy = -dz",
                backward_ops: vec![
                    BackwardOp::PassThrough { input_idx: 0 },
                    BackwardOp::Negate { input_idx: 1 },
                ],
            },
        );
    }

    /// z = x * y => dx = dz * y, dy = dz * x
    fn register_mul(&mut self) {
        self.rules.insert(
            DiffPrimitive::Mul,
            BackwardRule {
                op: DiffPrimitive::Mul,
                num_inputs: 2,
                description: "dx = dz * y, dy = dz * x",
                backward_ops: vec![
                    BackwardOp::MulByInput {
                        input_idx: 0,
                        other_input_idx: 1,
                    },
                    BackwardOp::MulByInput {
                        input_idx: 1,
                        other_input_idx: 0,
                    },
                ],
            },
        );
    }

    /// z = x / y => dx = dz / y, dy = -dz * x / y^2
    fn register_div(&mut self) {
        self.rules.insert(
            DiffPrimitive::Div,
            BackwardRule {
                op: DiffPrimitive::Div,
                num_inputs: 2,
                description: "dx = dz / y, dy = -dz * x / y^2",
                backward_ops: vec![
                    BackwardOp::DivByInput {
                        input_idx: 0,
                        divisor_idx: 1,
                    },
                    BackwardOp::DivDenomGrad {
                        input_idx: 1,
                        numerator_idx: 0,
                    },
                ],
            },
        );
    }

    /// z = -x => dx = -dz
    fn register_neg(&mut self) {
        self.rules.insert(
            DiffPrimitive::Neg,
            BackwardRule {
                op: DiffPrimitive::Neg,
                num_inputs: 1,
                description: "dx = -dz",
                backward_ops: vec![BackwardOp::Negate { input_idx: 0 }],
            },
        );
    }

    // === Math Functions ===

    /// z = sin(x) => dx = dz * cos(x)
    fn register_sin(&mut self) {
        self.rules.insert(
            DiffPrimitive::Sin,
            BackwardRule {
                op: DiffPrimitive::Sin,
                num_inputs: 1,
                description: "dx = dz * cos(x)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Cos(0),
                }],
            },
        );
    }

    /// z = cos(x) => dx = -dz * sin(x)
    fn register_cos(&mut self) {
        self.rules.insert(
            DiffPrimitive::Cos,
            BackwardRule {
                op: DiffPrimitive::Cos,
                num_inputs: 1,
                description: "dx = -dz * sin(x)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::NegSin(0),
                }],
            },
        );
    }

    /// z = tan(x) => dx = dz / cos^2(x) = dz * (1 + tan^2(x))
    fn register_tan(&mut self) {
        self.rules.insert(
            DiffPrimitive::Tan,
            BackwardRule {
                op: DiffPrimitive::Tan,
                num_inputs: 1,
                description: "dx = dz * sec^2(x)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::SecSquared(0),
                }],
            },
        );
    }

    /// z = exp(x) => dx = dz * exp(x) = dz * z
    fn register_exp(&mut self) {
        self.rules.insert(
            DiffPrimitive::Exp,
            BackwardRule {
                op: DiffPrimitive::Exp,
                num_inputs: 1,
                description: "dx = dz * exp(x)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Exp(0),
                }],
            },
        );
    }

    /// z = log(x) => dx = dz / x
    fn register_log(&mut self) {
        self.rules.insert(
            DiffPrimitive::Log,
            BackwardRule {
                op: DiffPrimitive::Log,
                num_inputs: 1,
                description: "dx = dz / x",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Reciprocal(0),
                }],
            },
        );
    }

    /// z = sqrt(x) => dx = dz / (2 * sqrt(x))
    fn register_sqrt(&mut self) {
        self.rules.insert(
            DiffPrimitive::Sqrt,
            BackwardRule {
                op: DiffPrimitive::Sqrt,
                num_inputs: 1,
                description: "dx = dz / (2 * sqrt(x))",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::HalfRsqrt(0),
                }],
            },
        );
    }

    /// z = x^n => dx = dz * n * x^(n-1)
    fn register_pow(&mut self) {
        self.rules.insert(
            DiffPrimitive::Pow,
            BackwardRule {
                op: DiffPrimitive::Pow,
                num_inputs: 1, // Exponent is assumed constant
                description: "dx = dz * n * x^(n-1)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::PowGrad {
                        base_idx: 0,
                        exponent: 0.0, // Will be filled in from tape
                    },
                }],
            },
        );
    }

    /// z = rsqrt(x) = 1/sqrt(x) => dx = -dz / (2 * x^(3/2))
    fn register_rsqrt(&mut self) {
        self.rules.insert(
            DiffPrimitive::Rsqrt,
            BackwardRule {
                op: DiffPrimitive::Rsqrt,
                num_inputs: 1,
                description: "dx = -dz / (2 * x^(3/2))",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::RsqrtGrad(0),
                }],
            },
        );
    }

    /// z = |x| => dx = dz * sign(x)
    fn register_abs(&mut self) {
        self.rules.insert(
            DiffPrimitive::Abs,
            BackwardRule {
                op: DiffPrimitive::Abs,
                num_inputs: 1,
                description: "dx = dz * sign(x)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Sign(0),
                }],
            },
        );
    }

    // === Activation Functions ===

    /// z = relu(x) = max(0, x) => dx = dz * (x > 0)
    fn register_relu(&mut self) {
        self.rules.insert(
            DiffPrimitive::Relu,
            BackwardRule {
                op: DiffPrimitive::Relu,
                num_inputs: 1,
                description: "dx = dz * (x > 0)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Step(0),
                }],
            },
        );
    }

    /// z = sigmoid(x) => dx = dz * z * (1 - z)
    fn register_sigmoid(&mut self) {
        self.rules.insert(
            DiffPrimitive::Sigmoid,
            BackwardRule {
                op: DiffPrimitive::Sigmoid,
                num_inputs: 1,
                description: "dx = dz * sigmoid(x) * (1 - sigmoid(x))",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::SigmoidGrad(0),
                }],
            },
        );
    }

    /// z = tanh(x) => dx = dz * (1 - z^2)
    fn register_tanh(&mut self) {
        self.rules.insert(
            DiffPrimitive::Tanh,
            BackwardRule {
                op: DiffPrimitive::Tanh,
                num_inputs: 1,
                description: "dx = dz * (1 - tanh^2(x))",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::TanhGrad(0),
                }],
            },
        );
    }

    /// z = gelu(x) = x * Phi(x) => complex gradient
    fn register_gelu(&mut self) {
        self.rules.insert(
            DiffPrimitive::Gelu,
            BackwardRule {
                op: DiffPrimitive::Gelu,
                num_inputs: 1,
                description: "dx = dz * gelu'(x)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::GeluGrad(0),
                }],
            },
        );
    }

    // === Reduction Operations ===

    /// z = sum(x) => dx = broadcast(dz)
    fn register_sum(&mut self) {
        self.rules.insert(
            DiffPrimitive::Sum,
            BackwardRule {
                op: DiffPrimitive::Sum,
                num_inputs: 1,
                description: "dx = broadcast(dz)",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Broadcast { shape: vec![] },
                }],
            },
        );
    }

    /// z = mean(x) => dx = broadcast(dz) / n
    fn register_mean(&mut self) {
        self.rules.insert(
            DiffPrimitive::Mean,
            BackwardRule {
                op: DiffPrimitive::Mean,
                num_inputs: 1,
                description: "dx = broadcast(dz) / n",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::MeanScale { count: 0 },
                }],
            },
        );
    }

    /// z = max(x) => dx = dz * (x == z)
    fn register_max(&mut self) {
        self.rules.insert(
            DiffPrimitive::Max,
            BackwardRule {
                op: DiffPrimitive::Max,
                num_inputs: 1,
                description: "dx = dz * (x == max(x))",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Indicator {
                        input_idx: 0,
                        reduction_result_idx: 0,
                    },
                }],
            },
        );
    }

    /// z = min(x) => dx = dz * (x == z)
    fn register_min(&mut self) {
        self.rules.insert(
            DiffPrimitive::Min,
            BackwardRule {
                op: DiffPrimitive::Min,
                num_inputs: 1,
                description: "dx = dz * (x == min(x))",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Indicator {
                        input_idx: 0,
                        reduction_result_idx: 0,
                    },
                }],
            },
        );
    }

    // === Matrix Operations ===

    /// C = A @ B => dA = dC @ B^T, dB = A^T @ dC
    fn register_matmul(&mut self) {
        self.rules.insert(
            DiffPrimitive::MatMul,
            BackwardRule {
                op: DiffPrimitive::MatMul,
                num_inputs: 2,
                description: "dA = dC @ B^T, dB = A^T @ dC",
                backward_ops: vec![
                    BackwardOp::Custom {
                        input_idx: 0,
                        expr: BackwardExpr::MatMulGrad {
                            grad_idx: 0,
                            other_idx: 1,
                            transpose_first: false,
                            transpose_second: true,
                        },
                    },
                    BackwardOp::Custom {
                        input_idx: 1,
                        expr: BackwardExpr::MatMulGrad {
                            grad_idx: 0,
                            other_idx: 0,
                            transpose_first: true,
                            transpose_second: false,
                        },
                    },
                ],
            },
        );
    }

    /// z = x^T => dx = dz^T
    fn register_transpose(&mut self) {
        self.rules.insert(
            DiffPrimitive::Transpose,
            BackwardRule {
                op: DiffPrimitive::Transpose,
                num_inputs: 1,
                description: "dx = dz^T",
                backward_ops: vec![BackwardOp::Custom {
                    input_idx: 0,
                    expr: BackwardExpr::Transpose(0),
                }],
            },
        );
    }

    /// Batch matrix multiply: same as matmul but batched
    fn register_batch_matmul(&mut self) {
        self.rules.insert(
            DiffPrimitive::BatchMatMul,
            BackwardRule {
                op: DiffPrimitive::BatchMatMul,
                num_inputs: 2,
                description: "dA = dC @ B^T (batched), dB = A^T @ dC (batched)",
                backward_ops: vec![
                    BackwardOp::Custom {
                        input_idx: 0,
                        expr: BackwardExpr::MatMulGrad {
                            grad_idx: 0,
                            other_idx: 1,
                            transpose_first: false,
                            transpose_second: true,
                        },
                    },
                    BackwardOp::Custom {
                        input_idx: 1,
                        expr: BackwardExpr::MatMulGrad {
                            grad_idx: 0,
                            other_idx: 0,
                            transpose_first: true,
                            transpose_second: false,
                        },
                    },
                ],
            },
        );
    }

    // === Fused Operations ===

    /// z = a * b + c => da = dz * b, db = dz * a, dc = dz
    fn register_fma(&mut self) {
        self.rules.insert(
            DiffPrimitive::Fma,
            BackwardRule {
                op: DiffPrimitive::Fma,
                num_inputs: 3,
                description: "da = dz * b, db = dz * a, dc = dz",
                backward_ops: vec![
                    BackwardOp::MulByInput {
                        input_idx: 0,
                        other_input_idx: 1,
                    },
                    BackwardOp::MulByInput {
                        input_idx: 1,
                        other_input_idx: 0,
                    },
                    BackwardOp::PassThrough { input_idx: 2 },
                ],
            },
        );
    }

    // === Quantization Operations (Straight-Through Estimator) ===
    //
    // QAT uses the straight-through estimator: during forward pass, values are
    // fake-quantized (quantize then dequantize), but during backward pass,
    // gradients pass through unchanged as if quantization didn't happen.
    // This allows gradients to flow and weights to update during QAT.

    /// FakeQuantize: z = dequant(quant(x)) => dx = dz (straight-through)
    fn register_fake_quantize(&mut self) {
        self.rules.insert(
            DiffPrimitive::FakeQuantize,
            BackwardRule {
                op: DiffPrimitive::FakeQuantize,
                num_inputs: 1,
                description: "dx = dz (straight-through estimator)",
                backward_ops: vec![BackwardOp::PassThrough { input_idx: 0 }],
            },
        );
    }

    /// FakeQuantizePerChannel: same as FakeQuantize but per-channel
    fn register_fake_quantize_per_channel(&mut self) {
        self.rules.insert(
            DiffPrimitive::FakeQuantizePerChannel,
            BackwardRule {
                op: DiffPrimitive::FakeQuantizePerChannel,
                num_inputs: 1,
                description: "dx = dz (straight-through estimator, per-channel)",
                backward_ops: vec![BackwardOp::PassThrough { input_idx: 0 }],
            },
        );
    }

    /// FakeQuantizeQuat: quaternion-aware fake quantization with STE
    fn register_fake_quantize_quat(&mut self) {
        self.rules.insert(
            DiffPrimitive::FakeQuantizeQuat,
            BackwardRule {
                op: DiffPrimitive::FakeQuantizeQuat,
                num_inputs: 1,
                description: "dx = dz (straight-through estimator, quaternion-aware)",
                backward_ops: vec![BackwardOp::PassThrough { input_idx: 0 }],
            },
        );
    }

    /// Get the backward rule for a primitive
    pub fn get_rule(&self, op: DiffPrimitive) -> Option<&BackwardRule> {
        self.rules.get(&op)
    }

    /// Check if a primitive is supported
    pub fn is_supported(&self, op: DiffPrimitive) -> bool {
        self.rules.contains_key(&op)
    }

    /// Classify a GpuOp as a differentiable primitive
    pub fn classify_op(&self, op: &GpuOp) -> Option<DiffPrimitive> {
        match op {
            GpuOp::FAdd(_, _) | GpuOp::Add(_, _) => Some(DiffPrimitive::Add),
            GpuOp::FSub(_, _) | GpuOp::Sub(_, _) => Some(DiffPrimitive::Sub),
            GpuOp::FMul(_, _) | GpuOp::Mul(_, _) => Some(DiffPrimitive::Mul),
            GpuOp::FDiv(_, _) | GpuOp::Div(_, _) => Some(DiffPrimitive::Div),
            GpuOp::FNeg(_) | GpuOp::Neg(_) => Some(DiffPrimitive::Neg),
            GpuOp::FastSin(_) => Some(DiffPrimitive::Sin),
            GpuOp::FastCos(_) => Some(DiffPrimitive::Cos),
            GpuOp::FastExp(_) => Some(DiffPrimitive::Exp),
            GpuOp::FastLog(_) => Some(DiffPrimitive::Log),
            GpuOp::FastSqrt(_) => Some(DiffPrimitive::Sqrt),
            GpuOp::FastRsqrt(_) => Some(DiffPrimitive::Rsqrt),
            GpuOp::FMulAdd(_, _, _) => Some(DiffPrimitive::Fma),
            GpuOp::AbsF32(_) | GpuOp::AbsF64(_) => Some(DiffPrimitive::Abs),
            GpuOp::WarpReduce(super::super::ir::WarpReduceOp::Add, _) => Some(DiffPrimitive::Sum),
            GpuOp::WarpReduce(super::super::ir::WarpReduceOp::Max, _) => Some(DiffPrimitive::Max),
            GpuOp::WarpReduce(super::super::ir::WarpReduceOp::Min, _) => Some(DiffPrimitive::Min),
            GpuOp::TileMma { .. } => Some(DiffPrimitive::MatMul),
            GpuOp::TileTranspose(_) => Some(DiffPrimitive::Transpose),
            GpuOp::FakeQuantize { .. } => Some(DiffPrimitive::FakeQuantize),
            GpuOp::FakeQuantizePerChannel { .. } => Some(DiffPrimitive::FakeQuantizePerChannel),
            GpuOp::FakeQuantizeQuat { .. } => Some(DiffPrimitive::FakeQuantizeQuat),
            _ => None,
        }
    }

    /// Get all supported primitives
    pub fn supported_primitives(&self) -> impl Iterator<Item = &DiffPrimitive> {
        self.rules.keys()
    }
}

impl Default for PrimitiveRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::ir::ValueId;
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = PrimitiveRegistry::new();
        assert!(registry.is_supported(DiffPrimitive::Add));
        assert!(registry.is_supported(DiffPrimitive::Mul));
        assert!(registry.is_supported(DiffPrimitive::Sin));
        assert!(registry.is_supported(DiffPrimitive::MatMul));
    }

    #[test]
    fn test_add_rule() {
        let registry = PrimitiveRegistry::new();
        let rule = registry.get_rule(DiffPrimitive::Add).unwrap();

        assert_eq!(rule.num_inputs, 2);
        assert_eq!(rule.backward_ops.len(), 2);

        // Both gradients should pass through
        match &rule.backward_ops[0] {
            BackwardOp::PassThrough { input_idx } => assert_eq!(*input_idx, 0),
            _ => panic!("Expected PassThrough"),
        }
    }

    #[test]
    fn test_mul_rule() {
        let registry = PrimitiveRegistry::new();
        let rule = registry.get_rule(DiffPrimitive::Mul).unwrap();

        assert_eq!(rule.num_inputs, 2);

        // dx = dz * y
        match &rule.backward_ops[0] {
            BackwardOp::MulByInput {
                input_idx,
                other_input_idx,
            } => {
                assert_eq!(*input_idx, 0);
                assert_eq!(*other_input_idx, 1);
            }
            _ => panic!("Expected MulByInput"),
        }
    }

    #[test]
    fn test_div_rule() {
        let registry = PrimitiveRegistry::new();
        let rule = registry.get_rule(DiffPrimitive::Div).unwrap();

        assert_eq!(rule.num_inputs, 2);

        // dx = dz / y
        match &rule.backward_ops[0] {
            BackwardOp::DivByInput {
                input_idx,
                divisor_idx,
            } => {
                assert_eq!(*input_idx, 0);
                assert_eq!(*divisor_idx, 1);
            }
            _ => panic!("Expected DivByInput"),
        }
    }

    #[test]
    fn test_sin_rule() {
        let registry = PrimitiveRegistry::new();
        let rule = registry.get_rule(DiffPrimitive::Sin).unwrap();

        assert_eq!(rule.num_inputs, 1);

        // dx = dz * cos(x)
        match &rule.backward_ops[0] {
            BackwardOp::Custom { input_idx, expr } => {
                assert_eq!(*input_idx, 0);
                match expr {
                    BackwardExpr::Cos(_) => {}
                    _ => panic!("Expected Cos expression"),
                }
            }
            _ => panic!("Expected Custom"),
        }
    }

    #[test]
    fn test_matmul_rule() {
        let registry = PrimitiveRegistry::new();
        let rule = registry.get_rule(DiffPrimitive::MatMul).unwrap();

        assert_eq!(rule.num_inputs, 2);
        assert_eq!(rule.backward_ops.len(), 2);

        // dA = dC @ B^T
        match &rule.backward_ops[0] {
            BackwardOp::Custom { expr, .. } => match expr {
                BackwardExpr::MatMulGrad {
                    transpose_second, ..
                } => {
                    assert!(*transpose_second);
                }
                _ => panic!("Expected MatMulGrad"),
            },
            _ => panic!("Expected Custom"),
        }
    }

    #[test]
    fn test_classify_op() {
        let registry = PrimitiveRegistry::new();

        assert_eq!(
            registry.classify_op(&GpuOp::FAdd(ValueId(0), ValueId(1))),
            Some(DiffPrimitive::Add)
        );
        assert_eq!(
            registry.classify_op(&GpuOp::FMul(ValueId(0), ValueId(1))),
            Some(DiffPrimitive::Mul)
        );
        assert_eq!(
            registry.classify_op(&GpuOp::FastSin(ValueId(0))),
            Some(DiffPrimitive::Sin)
        );

        // Non-differentiable ops should return None
        assert_eq!(registry.classify_op(&GpuOp::ThreadIdX), None);
        assert_eq!(registry.classify_op(&GpuOp::SyncThreads), None);
    }

    #[test]
    fn test_diff_primitive_to_tape_op() {
        assert_eq!(DiffPrimitive::Add.to_tape_op(), TapeOp::Add);
        assert_eq!(DiffPrimitive::Sin.to_tape_op(), TapeOp::Sin);
        assert_eq!(DiffPrimitive::MatMul.to_tape_op(), TapeOp::MatMul);
    }
}
