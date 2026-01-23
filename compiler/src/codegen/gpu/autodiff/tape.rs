//! GPU Tape for Reverse-Mode Automatic Differentiation
//!
//! The tape (also known as Wengert list) records operations during the forward pass
//! to enable efficient gradient computation during the backward pass.
//!
//! # Memory Layout
//!
//! The GPU tape is stored in global memory with the following layout:
//!
//! ```text
//! ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
//! │  Tape Header    │  Tape Entries   │  Value Buffer   │  Gradient Buffer│
//! │  (metadata)     │  (operations)   │  (forward vals) │  (adjoint vals) │
//! └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
//! ```
//!
//! # Thread Safety
//!
//! Each GPU thread records its own operations to the tape using atomic counters.
//! The backward pass processes entries in reverse order.

use std::collections::HashMap;

use super::super::ir::{GpuType, ValueId};

/// Configuration for the GPU tape
#[derive(Debug, Clone)]
pub struct GpuTapeConfig {
    /// Maximum number of tape entries
    pub max_entries: usize,
    /// Maximum number of values to track
    pub max_values: usize,
    /// Size of shared memory buffer (bytes)
    pub shared_buffer_size: usize,
    /// Enable tape compression (deduplicate common patterns)
    pub enable_compression: bool,
    /// Use thread-local tape sections
    pub thread_local_sections: bool,
}

impl Default for GpuTapeConfig {
    fn default() -> Self {
        Self {
            max_entries: 64 * 1024,    // 64K entries per thread block
            max_values: 16 * 1024,     // 16K values
            shared_buffer_size: 48 * 1024, // 48KB shared memory
            enable_compression: false,
            thread_local_sections: true,
        }
    }
}

/// Operation types recorded on the tape
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TapeOp {
    // Arithmetic operations
    /// Addition: z = x + y
    Add,
    /// Subtraction: z = x - y
    Sub,
    /// Multiplication: z = x * y
    Mul,
    /// Division: z = x / y
    Div,
    /// Negation: z = -x
    Neg,

    // Mathematical functions
    /// Sine: z = sin(x)
    Sin,
    /// Cosine: z = cos(x)
    Cos,
    /// Tangent: z = tan(x)
    Tan,
    /// Exponential: z = exp(x)
    Exp,
    /// Natural logarithm: z = ln(x)
    Log,
    /// Square root: z = sqrt(x)
    Sqrt,
    /// Power: z = x^y
    Pow,
    /// Reciprocal square root: z = 1/sqrt(x)
    Rsqrt,
    /// Absolute value: z = |x|
    Abs,

    // Activation functions (common in deep learning)
    /// ReLU: z = max(0, x)
    Relu,
    /// Sigmoid: z = 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: z = tanh(x)
    Tanh,
    /// GELU: z = x * Phi(x)
    Gelu,
    /// Softmax (requires special handling)
    Softmax,

    // Reduction operations
    /// Sum reduction
    Sum,
    /// Mean reduction
    Mean,
    /// Max reduction
    Max,
    /// Min reduction
    Min,

    // Matrix operations
    /// Matrix multiplication: C = A @ B
    MatMul,
    /// Matrix transpose
    Transpose,
    /// Batch matrix multiplication
    BatchMatMul,

    // Memory operations (for tracking data flow)
    /// Load from global memory
    Load,
    /// Store to global memory
    Store,
    /// Copy operation
    Copy,

    // Fused operations (for efficiency)
    /// Fused multiply-add: z = a * b + c
    Fma,
    /// Fused add-multiply: z = (a + b) * c
    AddMul,

    // Quantization operations (QAT with straight-through estimator)
    /// Fake quantization: output = dequant(quant(x)) - gradient passes through
    FakeQuantize,
    /// Per-channel fake quantization
    FakeQuantizePerChannel,
    /// Quaternion-aware fake quantization
    FakeQuantizeQuat,

    // Special operations
    /// Constant (no gradient)
    Const,
    /// Input variable (gradient endpoint)
    Input,
    /// Output variable (gradient source)
    Output,
    /// No operation (placeholder)
    Nop,
}

impl TapeOp {
    /// Check if this operation is differentiable
    pub fn is_differentiable(&self) -> bool {
        !matches!(self, TapeOp::Const | TapeOp::Load | TapeOp::Store | TapeOp::Copy | TapeOp::Nop)
    }

    /// Get the number of operands for this operation
    pub fn num_operands(&self) -> usize {
        match self {
            TapeOp::Neg | TapeOp::Sin | TapeOp::Cos | TapeOp::Tan |
            TapeOp::Exp | TapeOp::Log | TapeOp::Sqrt | TapeOp::Rsqrt |
            TapeOp::Abs | TapeOp::Relu | TapeOp::Sigmoid | TapeOp::Tanh |
            TapeOp::Gelu | TapeOp::Transpose | TapeOp::Sum | TapeOp::Mean |
            TapeOp::Max | TapeOp::Min | TapeOp::Softmax |
            TapeOp::FakeQuantize | TapeOp::FakeQuantizePerChannel |
            TapeOp::FakeQuantizeQuat => 1,

            TapeOp::Add | TapeOp::Sub | TapeOp::Mul | TapeOp::Div |
            TapeOp::Pow | TapeOp::MatMul | TapeOp::BatchMatMul => 2,

            TapeOp::Fma | TapeOp::AddMul => 3,

            TapeOp::Const | TapeOp::Input | TapeOp::Output |
            TapeOp::Load | TapeOp::Store | TapeOp::Copy | TapeOp::Nop => 0,
        }
    }

    /// Get a human-readable name for this operation
    pub fn name(&self) -> &'static str {
        match self {
            TapeOp::Add => "add",
            TapeOp::Sub => "sub",
            TapeOp::Mul => "mul",
            TapeOp::Div => "div",
            TapeOp::Neg => "neg",
            TapeOp::Sin => "sin",
            TapeOp::Cos => "cos",
            TapeOp::Tan => "tan",
            TapeOp::Exp => "exp",
            TapeOp::Log => "log",
            TapeOp::Sqrt => "sqrt",
            TapeOp::Pow => "pow",
            TapeOp::Rsqrt => "rsqrt",
            TapeOp::Abs => "abs",
            TapeOp::Relu => "relu",
            TapeOp::Sigmoid => "sigmoid",
            TapeOp::Tanh => "tanh",
            TapeOp::Gelu => "gelu",
            TapeOp::Softmax => "softmax",
            TapeOp::Sum => "sum",
            TapeOp::Mean => "mean",
            TapeOp::Max => "max",
            TapeOp::Min => "min",
            TapeOp::MatMul => "matmul",
            TapeOp::Transpose => "transpose",
            TapeOp::BatchMatMul => "batch_matmul",
            TapeOp::Load => "load",
            TapeOp::Store => "store",
            TapeOp::Copy => "copy",
            TapeOp::Fma => "fma",
            TapeOp::AddMul => "add_mul",
            TapeOp::FakeQuantize => "fake_quantize",
            TapeOp::FakeQuantizePerChannel => "fake_quantize_per_channel",
            TapeOp::FakeQuantizeQuat => "fake_quantize_quat",
            TapeOp::Const => "const",
            TapeOp::Input => "input",
            TapeOp::Output => "output",
            TapeOp::Nop => "nop",
        }
    }
}

/// A single entry in the tape
#[derive(Debug, Clone)]
pub struct TapeEntry {
    /// Operation type
    pub op: TapeOp,
    /// Result value index
    pub result: u32,
    /// First operand index (or 0 for nullary ops)
    pub operand1: u32,
    /// Second operand index (or 0 for unary ops)
    pub operand2: u32,
    /// Third operand index (or 0 for binary ops)
    pub operand3: u32,
    /// Value type
    pub value_type: GpuType,
    /// Extra data (e.g., constant value, reduction axis)
    pub extra: TapeExtra,
}

/// Extra data associated with a tape entry
#[derive(Debug, Clone)]
pub enum TapeExtra {
    /// No extra data
    None,
    /// Constant float value
    ConstFloat(f64),
    /// Constant integer value
    ConstInt(i64),
    /// Reduction axis
    ReductionAxis(u32),
    /// Matrix dimensions (M, N, K)
    MatrixDims(u32, u32, u32),
    /// Power exponent
    Exponent(f64),
    /// Shape information
    Shape(Vec<u32>),
}

impl Default for TapeExtra {
    fn default() -> Self {
        TapeExtra::None
    }
}

impl TapeEntry {
    /// Create a new tape entry
    pub fn new(op: TapeOp, result: u32, value_type: GpuType) -> Self {
        Self {
            op,
            result,
            operand1: 0,
            operand2: 0,
            operand3: 0,
            value_type,
            extra: TapeExtra::None,
        }
    }

    /// Create a unary operation entry
    pub fn unary(op: TapeOp, result: u32, operand: u32, value_type: GpuType) -> Self {
        Self {
            op,
            result,
            operand1: operand,
            operand2: 0,
            operand3: 0,
            value_type,
            extra: TapeExtra::None,
        }
    }

    /// Create a binary operation entry
    pub fn binary(op: TapeOp, result: u32, left: u32, right: u32, value_type: GpuType) -> Self {
        Self {
            op,
            result,
            operand1: left,
            operand2: right,
            operand3: 0,
            value_type,
            extra: TapeExtra::None,
        }
    }

    /// Create a ternary operation entry
    pub fn ternary(op: TapeOp, result: u32, a: u32, b: u32, c: u32, value_type: GpuType) -> Self {
        Self {
            op,
            result,
            operand1: a,
            operand2: b,
            operand3: c,
            value_type,
            extra: TapeExtra::None,
        }
    }

    /// Set extra data
    pub fn with_extra(mut self, extra: TapeExtra) -> Self {
        self.extra = extra;
        self
    }
}

/// GPU tape for recording operations during forward pass
#[derive(Debug, Clone)]
pub struct GpuTape {
    /// Configuration
    config: GpuTapeConfig,
    /// Recorded entries
    entries: Vec<TapeEntry>,
    /// Value buffer (stores intermediate values for backward pass)
    values: Vec<f64>,
    /// Gradient buffer (stores accumulated gradients)
    gradients: Vec<f64>,
    /// Value type tracking
    value_types: Vec<GpuType>,
    /// Map from ValueId to tape value index
    value_map: HashMap<ValueId, u32>,
    /// Next value index
    next_value_idx: u32,
    /// Input value indices (endpoints for gradient computation)
    input_indices: Vec<u32>,
    /// Output value indices (sources of gradients)
    output_indices: Vec<u32>,
}

impl GpuTape {
    /// Create a new GPU tape with the given configuration
    pub fn new(config: GpuTapeConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            values: Vec::new(),
            gradients: Vec::new(),
            value_types: Vec::new(),
            value_map: HashMap::new(),
            next_value_idx: 0,
            input_indices: Vec::new(),
            output_indices: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(GpuTapeConfig::default())
    }

    /// Allocate a new value index
    fn allocate_value(&mut self, ty: GpuType) -> u32 {
        let idx = self.next_value_idx;
        self.next_value_idx += 1;
        self.values.push(0.0);
        self.gradients.push(0.0);
        self.value_types.push(ty);
        idx
    }

    /// Map a ValueId to a tape value index
    pub fn map_value(&mut self, value_id: ValueId, ty: GpuType) -> u32 {
        if let Some(&idx) = self.value_map.get(&value_id) {
            return idx;
        }
        let idx = self.allocate_value(ty);
        self.value_map.insert(value_id, idx);
        idx
    }

    /// Get the tape value index for a ValueId
    pub fn get_value_index(&self, value_id: ValueId) -> Option<u32> {
        self.value_map.get(&value_id).copied()
    }

    /// Record a constant value
    pub fn record_const(&mut self, result: ValueId, value: f64, ty: GpuType) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        self.values[result_idx as usize] = value;
        let entry = TapeEntry::new(TapeOp::Const, result_idx, ty)
            .with_extra(TapeExtra::ConstFloat(value));
        self.entries.push(entry);
        result_idx
    }

    /// Record an input variable
    pub fn record_input(&mut self, result: ValueId, ty: GpuType) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        self.input_indices.push(result_idx);
        let entry = TapeEntry::new(TapeOp::Input, result_idx, ty);
        self.entries.push(entry);
        result_idx
    }

    /// Record an output variable
    pub fn record_output(&mut self, result: ValueId, source: u32, ty: GpuType) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        self.output_indices.push(result_idx);
        let entry = TapeEntry::unary(TapeOp::Output, result_idx, source, ty);
        self.entries.push(entry);
        result_idx
    }

    /// Record a unary operation
    pub fn record_unary(&mut self, op: TapeOp, result: ValueId, operand: u32, ty: GpuType) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        let entry = TapeEntry::unary(op, result_idx, operand, ty);
        self.entries.push(entry);
        result_idx
    }

    /// Record a binary operation
    pub fn record_binary(
        &mut self,
        op: TapeOp,
        result: ValueId,
        left: u32,
        right: u32,
        ty: GpuType,
    ) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        let entry = TapeEntry::binary(op, result_idx, left, right, ty);
        self.entries.push(entry);
        result_idx
    }

    /// Record a ternary operation (e.g., FMA)
    pub fn record_ternary(
        &mut self,
        op: TapeOp,
        result: ValueId,
        a: u32,
        b: u32,
        c: u32,
        ty: GpuType,
    ) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        let entry = TapeEntry::ternary(op, result_idx, a, b, c, ty);
        self.entries.push(entry);
        result_idx
    }

    /// Record a reduction operation
    pub fn record_reduction(
        &mut self,
        op: TapeOp,
        result: ValueId,
        operand: u32,
        axis: u32,
        ty: GpuType,
    ) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        let entry = TapeEntry::unary(op, result_idx, operand, ty)
            .with_extra(TapeExtra::ReductionAxis(axis));
        self.entries.push(entry);
        result_idx
    }

    /// Record a matrix multiplication
    pub fn record_matmul(
        &mut self,
        result: ValueId,
        a: u32,
        b: u32,
        m: u32,
        n: u32,
        k: u32,
        ty: GpuType,
    ) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        let entry = TapeEntry::binary(TapeOp::MatMul, result_idx, a, b, ty)
            .with_extra(TapeExtra::MatrixDims(m, n, k));
        self.entries.push(entry);
        result_idx
    }

    /// Record a power operation with constant exponent
    pub fn record_pow_const(
        &mut self,
        result: ValueId,
        base: u32,
        exponent: f64,
        ty: GpuType,
    ) -> u32 {
        let result_idx = self.map_value(result, ty.clone());
        let entry = TapeEntry::unary(TapeOp::Pow, result_idx, base, ty)
            .with_extra(TapeExtra::Exponent(exponent));
        self.entries.push(entry);
        result_idx
    }

    /// Get the number of entries in the tape
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the tape is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries
    pub fn entries(&self) -> &[TapeEntry] {
        &self.entries
    }

    /// Get entries in reverse order (for backward pass)
    pub fn entries_reversed(&self) -> impl Iterator<Item = &TapeEntry> {
        self.entries.iter().rev()
    }

    /// Get input indices
    pub fn input_indices(&self) -> &[u32] {
        &self.input_indices
    }

    /// Get output indices
    pub fn output_indices(&self) -> &[u32] {
        &self.output_indices
    }

    /// Get value at index
    pub fn get_value(&self, idx: u32) -> f64 {
        self.values.get(idx as usize).copied().unwrap_or(0.0)
    }

    /// Set value at index
    pub fn set_value(&mut self, idx: u32, value: f64) {
        if let Some(v) = self.values.get_mut(idx as usize) {
            *v = value;
        }
    }

    /// Get gradient at index
    pub fn get_gradient(&self, idx: u32) -> f64 {
        self.gradients.get(idx as usize).copied().unwrap_or(0.0)
    }

    /// Set gradient at index
    pub fn set_gradient(&mut self, idx: u32, gradient: f64) {
        if let Some(g) = self.gradients.get_mut(idx as usize) {
            *g = gradient;
        }
    }

    /// Accumulate gradient at index (for gradient accumulation)
    pub fn accumulate_gradient(&mut self, idx: u32, gradient: f64) {
        if let Some(g) = self.gradients.get_mut(idx as usize) {
            *g += gradient;
        }
    }

    /// Clear all gradients
    pub fn clear_gradients(&mut self) {
        for g in &mut self.gradients {
            *g = 0.0;
        }
    }

    /// Get the type of a value
    pub fn get_value_type(&self, idx: u32) -> Option<&GpuType> {
        self.value_types.get(idx as usize)
    }

    /// Clear the tape (but keep configuration)
    pub fn clear(&mut self) {
        self.entries.clear();
        self.values.clear();
        self.gradients.clear();
        self.value_types.clear();
        self.value_map.clear();
        self.next_value_idx = 0;
        self.input_indices.clear();
        self.output_indices.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &GpuTapeConfig {
        &self.config
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let entries_size = self.entries.len() * std::mem::size_of::<TapeEntry>();
        let values_size = self.values.len() * std::mem::size_of::<f64>();
        let gradients_size = self.gradients.len() * std::mem::size_of::<f64>();
        let types_size = self.value_types.len() * std::mem::size_of::<GpuType>();
        entries_size + values_size + gradients_size + types_size
    }
}

/// Builder for constructing a tape from kernel analysis
#[derive(Debug)]
pub struct TapeBuilder {
    tape: GpuTape,
}

impl TapeBuilder {
    /// Create a new tape builder
    pub fn new(config: GpuTapeConfig) -> Self {
        Self {
            tape: GpuTape::new(config),
        }
    }

    /// Add an input
    pub fn add_input(&mut self, value_id: ValueId, ty: GpuType) -> u32 {
        self.tape.record_input(value_id, ty)
    }

    /// Add a constant
    pub fn add_const(&mut self, value_id: ValueId, value: f64, ty: GpuType) -> u32 {
        self.tape.record_const(value_id, value, ty)
    }

    /// Add an operation
    pub fn add_op(&mut self, entry: TapeEntry) {
        self.tape.entries.push(entry);
    }

    /// Mark an output
    pub fn mark_output(&mut self, value_id: ValueId, source: u32, ty: GpuType) -> u32 {
        self.tape.record_output(value_id, source, ty)
    }

    /// Build the tape
    pub fn build(self) -> GpuTape {
        self.tape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_creation() {
        let tape = GpuTape::with_defaults();
        assert!(tape.is_empty());
        assert_eq!(tape.len(), 0);
    }

    #[test]
    fn test_record_const() {
        let mut tape = GpuTape::with_defaults();
        let idx = tape.record_const(ValueId(0), 3.14, GpuType::F32);
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.get_value(idx), 3.14);
    }

    #[test]
    fn test_record_binary() {
        let mut tape = GpuTape::with_defaults();
        let x = tape.record_input(ValueId(0), GpuType::F32);
        let y = tape.record_input(ValueId(1), GpuType::F32);
        let z = tape.record_binary(TapeOp::Add, ValueId(2), x, y, GpuType::F32);

        assert_eq!(tape.len(), 3);
        assert_eq!(tape.input_indices().len(), 2);

        let entry = &tape.entries()[2];
        assert_eq!(entry.op, TapeOp::Add);
        assert_eq!(entry.operand1, x);
        assert_eq!(entry.operand2, y);
        assert_eq!(entry.result, z);
    }

    #[test]
    fn test_gradient_accumulation() {
        let mut tape = GpuTape::with_defaults();
        let x = tape.record_input(ValueId(0), GpuType::F32);

        tape.set_gradient(x, 1.0);
        assert_eq!(tape.get_gradient(x), 1.0);

        tape.accumulate_gradient(x, 2.0);
        assert_eq!(tape.get_gradient(x), 3.0);

        tape.clear_gradients();
        assert_eq!(tape.get_gradient(x), 0.0);
    }

    #[test]
    fn test_tape_op_properties() {
        assert!(TapeOp::Add.is_differentiable());
        assert!(TapeOp::Mul.is_differentiable());
        assert!(TapeOp::Sin.is_differentiable());
        assert!(!TapeOp::Const.is_differentiable());
        assert!(!TapeOp::Load.is_differentiable());

        assert_eq!(TapeOp::Add.num_operands(), 2);
        assert_eq!(TapeOp::Sin.num_operands(), 1);
        assert_eq!(TapeOp::Fma.num_operands(), 3);
    }

    #[test]
    fn test_entries_reversed() {
        let mut tape = GpuTape::with_defaults();
        tape.record_input(ValueId(0), GpuType::F32);
        tape.record_input(ValueId(1), GpuType::F32);
        tape.record_binary(TapeOp::Add, ValueId(2), 0, 1, GpuType::F32);

        let ops: Vec<_> = tape.entries_reversed().map(|e| e.op).collect();
        assert_eq!(ops, vec![TapeOp::Add, TapeOp::Input, TapeOp::Input]);
    }
}
