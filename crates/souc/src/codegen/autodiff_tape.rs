//! Tape Data Structure for Reverse-Mode Automatic Differentiation
//!
//! This module implements the tape recording infrastructure for backpropagation.
//! The tape records all operations during the forward pass, then plays them back
//! in reverse order during the backward pass to compute gradients.
//!
//! # Architecture
//!
//! - **TapeOp**: Enum of differentiable operations (add, mul, sin, etc.)
//! - **TapeEntry**: Single recorded operation with operands and result
//! - **Tape**: Collection of entries with value/gradient buffers
//!
//! # Memory Optimization
//!
//! For long computations, the tape can grow very large. We implement:
//! - Value reuse: Only store values needed for backward pass
//! - Binomial checkpointing: Trade recomputation for memory (future)
//!
//! # References
//!
//! - TapeFlow (CGO 2024): Streaming gradient tapes
//! - MimIrADe (CC 2025): MIR-level autodiff integration

use std::collections::HashMap;

/// Operations that can be recorded on the tape
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TapeOp {
    /// Input variable (leaf node)
    Input,
    /// Constant value (leaf node)
    Const,

    // Binary operations
    Add,
    Sub,
    Mul,
    Div,

    // Unary operations
    Neg,

    // Transcendental functions
    Exp,
    Log,
    Sin,
    Cos,
    Tan,

    // Hyperbolic functions
    Sinh,
    Cosh,
    Tanh,

    // Power functions
    Sqrt,
    /// Power with constant exponent
    PowConst,
    /// General power (both operands can be variables)
    Pow,
}

/// A single entry in the autodiff tape
#[derive(Debug, Clone)]
pub struct TapeEntry {
    /// Operation type
    pub op: TapeOp,
    /// Index of the result value
    pub result_idx: usize,
    /// Indices of operands (up to 2 for binary ops)
    pub operand_indices: Vec<usize>,
    /// Additional data (e.g., constant exponent for PowConst)
    pub extra_data: Option<f64>,
}

impl TapeEntry {
    /// Create a new tape entry for a unary operation
    pub fn unary(op: TapeOp, result_idx: usize, operand_idx: usize) -> Self {
        Self {
            op,
            result_idx,
            operand_indices: vec![operand_idx],
            extra_data: None,
        }
    }

    /// Create a new tape entry for a binary operation
    pub fn binary(op: TapeOp, result_idx: usize, left_idx: usize, right_idx: usize) -> Self {
        Self {
            op,
            result_idx,
            operand_indices: vec![left_idx, right_idx],
            extra_data: None,
        }
    }

    /// Create a new tape entry for a leaf node (input/constant)
    pub fn leaf(op: TapeOp, result_idx: usize, value: Option<f64>) -> Self {
        Self {
            op,
            result_idx,
            operand_indices: vec![],
            extra_data: value,
        }
    }

    /// Add extra data (e.g., constant exponent)
    pub fn with_extra(mut self, data: f64) -> Self {
        self.extra_data = Some(data);
        self
    }
}

/// Configuration for the tape
#[derive(Debug, Clone)]
pub struct TapeConfig {
    /// Initial capacity for tape entries
    pub initial_capacity: usize,
    /// Initial capacity for value buffer
    pub value_capacity: usize,
    /// Whether to enable checkpointing (future)
    pub enable_checkpointing: bool,
}

impl Default for TapeConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            value_capacity: 1024,
            enable_checkpointing: false,
        }
    }
}

/// Tape for recording computation graph
///
/// The tape records all operations during the forward pass. Each operation
/// stores:
/// - The operation type (add, mul, sin, etc.)
/// - Indices of operands
/// - Index where result is stored
///
/// During backward pass, we traverse the tape in reverse order and accumulate
/// gradients using the chain rule.
#[derive(Debug, Clone)]
pub struct Tape {
    /// Configuration
    config: TapeConfig,

    /// Recorded operations in forward order
    entries: Vec<TapeEntry>,

    /// Value buffer: stores intermediate values for backward pass
    /// Index i contains the value computed by the operation that produced index i
    values: Vec<f64>,

    /// Gradient buffer: stores accumulated gradients
    /// Index i contains ∂L/∂value[i] where L is the loss
    gradients: Vec<f64>,

    /// Map from value index to whether it's needed for backward pass
    /// (memory optimization - can discard values not needed)
    value_needed: Vec<bool>,

    /// Input variable indices (for returning gradients)
    input_indices: Vec<usize>,

    /// Next available value index
    next_value_idx: usize,
}

impl Tape {
    /// Create a new tape with default configuration
    pub fn new() -> Self {
        Self::with_config(TapeConfig::default())
    }

    /// Create a new tape with custom configuration
    pub fn with_config(config: TapeConfig) -> Self {
        Self {
            entries: Vec::with_capacity(config.initial_capacity),
            values: Vec::with_capacity(config.value_capacity),
            gradients: Vec::with_capacity(config.value_capacity),
            value_needed: Vec::with_capacity(config.value_capacity),
            input_indices: Vec::new(),
            next_value_idx: 0,
            config,
        }
    }

    /// Allocate a new value index
    fn allocate_value(&mut self) -> usize {
        let idx = self.next_value_idx;
        self.next_value_idx += 1;

        // Extend buffers if needed
        if idx >= self.values.len() {
            self.values.resize(idx + 1, 0.0);
            self.gradients.resize(idx + 1, 0.0);
            self.value_needed.resize(idx + 1, false);
        }

        idx
    }

    /// Record an input variable
    pub fn input(&mut self, value: f64) -> usize {
        let idx = self.allocate_value();
        self.values[idx] = value;
        self.value_needed[idx] = true; // Inputs always needed for gradients
        self.input_indices.push(idx);

        let entry = TapeEntry::leaf(TapeOp::Input, idx, None);
        self.entries.push(entry);

        idx
    }

    /// Record a constant value
    pub fn constant(&mut self, value: f64) -> usize {
        let idx = self.allocate_value();
        self.values[idx] = value;
        self.value_needed[idx] = false; // Constants don't need gradients

        let entry = TapeEntry::leaf(TapeOp::Const, idx, Some(value));
        self.entries.push(entry);

        idx
    }

    /// Get value at index
    pub fn get_value(&self, idx: usize) -> f64 {
        self.values[idx]
    }

    /// Get gradient at index
    pub fn get_gradient(&self, idx: usize) -> f64 {
        self.gradients[idx]
    }

    /// Get all input indices
    pub fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }

    /// Get number of recorded operations
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if tape is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the tape for reuse
    pub fn clear(&mut self) {
        self.entries.clear();
        self.gradients.fill(0.0);
        self.input_indices.clear();
        self.next_value_idx = 0;
    }

    /// Clear gradients only (for multiple backward passes)
    pub fn clear_gradients(&mut self) {
        self.gradients.fill(0.0);
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let entries_size = self.entries.len() * std::mem::size_of::<TapeEntry>();
        let values_size = self.values.len() * std::mem::size_of::<f64>();
        let gradients_size = self.gradients.len() * std::mem::size_of::<f64>();
        let needed_size = self.value_needed.len() * std::mem::size_of::<bool>();

        entries_size + values_size + gradients_size + needed_size
    }

    /// Record a unary operation
    pub fn record_unary(&mut self, op: TapeOp, operand_idx: usize) -> usize {
        let result_idx = self.allocate_value();
        self.value_needed[operand_idx] = true; // Mark operand as needed

        let entry = TapeEntry::unary(op, result_idx, operand_idx);
        self.entries.push(entry);

        result_idx
    }

    /// Record a binary operation
    pub fn record_binary(&mut self, op: TapeOp, left_idx: usize, right_idx: usize) -> usize {
        let result_idx = self.allocate_value();
        self.value_needed[left_idx] = true; // Mark operands as needed
        self.value_needed[right_idx] = true;

        let entry = TapeEntry::binary(op, result_idx, left_idx, right_idx);
        self.entries.push(entry);

        result_idx
    }

    /// Record a power operation with constant exponent
    pub fn record_pow_const(&mut self, base_idx: usize, exponent: f64) -> usize {
        let result_idx = self.allocate_value();
        self.value_needed[base_idx] = true;

        let entry = TapeEntry::unary(TapeOp::PowConst, result_idx, base_idx).with_extra(exponent);
        self.entries.push(entry);

        result_idx
    }

    /// Get entries for iteration (forward order)
    pub fn entries(&self) -> &[TapeEntry] {
        &self.entries
    }

    /// Get entries in reverse order (for backward pass)
    pub fn entries_reversed(&self) -> impl Iterator<Item = &TapeEntry> {
        self.entries.iter().rev()
    }

    /// Set value at index (used during forward pass execution)
    pub fn set_value(&mut self, idx: usize, value: f64) {
        self.values[idx] = value;
    }

    /// Accumulate gradient at index (used during backward pass)
    pub fn accumulate_gradient(&mut self, idx: usize, grad: f64) {
        self.gradients[idx] += grad;
    }

    /// Get gradients for all inputs
    pub fn input_gradients(&self) -> HashMap<usize, f64> {
        self.input_indices
            .iter()
            .map(|&idx| (idx, self.gradients[idx]))
            .collect()
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_allocation() {
        let mut tape = Tape::new();

        let idx1 = tape.input(5.0);
        assert_eq!(idx1, 0);
        assert_eq!(tape.get_value(0), 5.0);

        let idx2 = tape.constant(2.0);
        assert_eq!(idx2, 1);
        assert_eq!(tape.get_value(1), 2.0);
    }

    #[test]
    fn test_tape_recording() {
        let mut tape = Tape::new();

        let x = tape.input(3.0);
        let c = tape.constant(2.0);
        let _result = tape.record_binary(TapeOp::Mul, x, c);

        assert_eq!(tape.len(), 3); // Input, Const, Mul
    }

    #[test]
    fn test_tape_clear() {
        let mut tape = Tape::new();

        let _x = tape.input(5.0);
        let _y = tape.constant(3.0);
        assert_eq!(tape.len(), 2);

        tape.clear();
        assert_eq!(tape.len(), 0);
        assert_eq!(tape.next_value_idx, 0);
    }
}
