//! Reverse-Mode Automatic Differentiation (Backpropagation)
//!
//! This module implements tape-based reverse-mode AD for efficient gradient computation.
//! Reverse-mode AD is optimal for functions R^n → R^m where n >> m (many inputs, few outputs),
//! such as neural network training where we compute gradients of a scalar loss with respect
//! to millions of parameters.
//!
//! # Algorithm
//!
//! 1. **Forward Pass**: Execute computation and record all operations on a tape
//! 2. **Backward Pass**: Traverse tape in reverse, applying chain rule to compute gradients
//! 3. **Result**: Gradients of output(s) with respect to all inputs
//!
//! # Complexity
//!
//! - Time: O(forward_cost) + O(backward_cost) ≈ 3× forward cost
//! - Space: O(tape_size) = O(number of operations)
//!
//! # Example
//!
//! ```rust,ignore
//! use sounio::codegen::autodiff_reverse::{Tape, TapeValue};
//!
//! let mut tape = Tape::new();
//!
//! // f(x) = x² at x=3
//! let x = TapeValue::variable(&mut tape, 3.0);
//! let result = x.mul(x);
//!
//! // Compute df/dx
//! let gradients = result.backward(&mut tape);
//! assert_eq!(gradients[&x.idx], 6.0);  // df/dx = 2x = 6
//! ```
//!
//! # References
//!
//! - TapeFlow (CGO 2024): Streaming gradient tapes
//! - MimIrADe (CC 2025): MIR-level autodiff integration

use super::autodiff_tape::{Tape, TapeEntry, TapeOp};
use std::cell::RefCell;
use std::rc::Rc;

/// A value tracked by the tape for automatic differentiation
///
/// This wraps an index into the tape's value buffer. All arithmetic operations
/// on TapeValue automatically record themselves on the tape.
#[derive(Debug, Clone)]
pub struct TapeValue {
    /// Reference to the tape
    tape: Rc<RefCell<Tape>>,
    /// Index in the tape's value buffer
    pub idx: usize,
}

impl TapeValue {
    /// Create a new variable (input with gradient tracking)
    pub fn variable(tape: &Rc<RefCell<Tape>>, value: f64) -> Self {
        let idx = tape.borrow_mut().input(value);
        Self {
            tape: Rc::clone(tape),
            idx,
        }
    }

    /// Create a constant (no gradient tracking)
    pub fn constant(tape: &Rc<RefCell<Tape>>, value: f64) -> Self {
        let idx = tape.borrow_mut().constant(value);
        Self {
            tape: Rc::clone(tape),
            idx,
        }
    }

    /// Get the current value
    pub fn value(&self) -> f64 {
        self.tape.borrow().get_value(self.idx)
    }

    /// Get the gradient (only valid after backward pass)
    pub fn gradient(&self) -> f64 {
        self.tape.borrow().get_gradient(self.idx)
    }

    // ======================
    // Arithmetic Operations
    // ======================

    /// Addition: self + other
    pub fn add(&self, other: &Self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_binary(TapeOp::Add, self.idx, other.idx);

            // Execute forward pass
            let val = tape.get_value(self.idx) + tape.get_value(other.idx);
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Subtraction: self - other
    pub fn sub(&self, other: &Self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_binary(TapeOp::Sub, self.idx, other.idx);

            let val = tape.get_value(self.idx) - tape.get_value(other.idx);
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Multiplication: self * other
    pub fn mul(&self, other: &Self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_binary(TapeOp::Mul, self.idx, other.idx);

            let val = tape.get_value(self.idx) * tape.get_value(other.idx);
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Division: self / other
    pub fn div(&self, other: &Self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_binary(TapeOp::Div, self.idx, other.idx);

            let val = tape.get_value(self.idx) / tape.get_value(other.idx);
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Negation: -self
    pub fn neg(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Neg, self.idx);

            let val = -tape.get_value(self.idx);
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    // ==========================
    // Transcendental Functions
    // ==========================

    /// Exponential: e^self
    pub fn exp(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Exp, self.idx);

            let val = libm::exp(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Natural logarithm: ln(self)
    pub fn log(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Log, self.idx);

            let val = libm::log(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Sine: sin(self)
    pub fn sin(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Sin, self.idx);

            let val = libm::sin(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Cosine: cos(self)
    pub fn cos(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Cos, self.idx);

            let val = libm::cos(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Tangent: tan(self)
    pub fn tan(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Tan, self.idx);

            let val = libm::tan(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    // ========================
    // Hyperbolic Functions
    // ========================

    /// Hyperbolic sine: sinh(self)
    pub fn sinh(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Sinh, self.idx);

            let val = libm::sinh(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Hyperbolic cosine: cosh(self)
    pub fn cosh(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Cosh, self.idx);

            let val = libm::cosh(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Hyperbolic tangent: tanh(self)
    pub fn tanh(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Tanh, self.idx);

            let val = libm::tanh(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    // =================
    // Power Functions
    // =================

    /// Square root: √self
    pub fn sqrt(&self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_unary(TapeOp::Sqrt, self.idx);

            let val = libm::sqrt(tape.get_value(self.idx));
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// Power with constant exponent: self^exponent
    pub fn pow_const(&self, exponent: f64) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_pow_const(self.idx, exponent);

            let val = libm::pow(tape.get_value(self.idx), exponent);
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    /// General power: self^other (both can be variables)
    pub fn pow(&self, other: &Self) -> Self {
        let result_idx = {
            let mut tape = self.tape.borrow_mut();
            let result_idx = tape.record_binary(TapeOp::Pow, self.idx, other.idx);

            let base = tape.get_value(self.idx);
            let exp = tape.get_value(other.idx);
            let val = libm::pow(base, exp);
            tape.set_value(result_idx, val);

            result_idx
        };

        Self {
            tape: Rc::clone(&self.tape),
            idx: result_idx,
        }
    }

    // ====================
    // Backward Pass
    // ====================

    /// Perform backward pass to compute gradients
    ///
    /// This computes ∂(self)/∂x for all inputs x.
    /// After calling this, use `gradient()` on input TapeValues to get their gradients.
    pub fn backward(&self) {
        let mut tape = self.tape.borrow_mut();

        // Seed gradient: ∂self/∂self = 1
        tape.accumulate_gradient(self.idx, 1.0);

        // Collect entries to avoid holding immutable borrow during mutation
        let entries: Vec<TapeEntry> = tape.entries().iter().rev().cloned().collect();

        // Traverse tape in reverse order
        for entry in &entries {
            let output_grad = tape.get_gradient(entry.result_idx);

            // Skip if no gradient flows through this operation
            if output_grad == 0.0 {
                continue;
            }

            match entry.op {
                TapeOp::Input | TapeOp::Const => {
                    // Leaf nodes - gradients already accumulated
                }

                // Binary operations
                TapeOp::Add => {
                    // d(x + y)/dx = 1, d(x + y)/dy = 1
                    let left_idx = entry.operand_indices[0];
                    let right_idx = entry.operand_indices[1];

                    tape.accumulate_gradient(left_idx, output_grad);
                    tape.accumulate_gradient(right_idx, output_grad);
                }

                TapeOp::Sub => {
                    // d(x - y)/dx = 1, d(x - y)/dy = -1
                    let left_idx = entry.operand_indices[0];
                    let right_idx = entry.operand_indices[1];

                    tape.accumulate_gradient(left_idx, output_grad);
                    tape.accumulate_gradient(right_idx, -output_grad);
                }

                TapeOp::Mul => {
                    // d(x * y)/dx = y, d(x * y)/dy = x
                    let left_idx = entry.operand_indices[0];
                    let right_idx = entry.operand_indices[1];

                    let left_val = tape.get_value(left_idx);
                    let right_val = tape.get_value(right_idx);

                    tape.accumulate_gradient(left_idx, output_grad * right_val);
                    tape.accumulate_gradient(right_idx, output_grad * left_val);
                }

                TapeOp::Div => {
                    // d(x / y)/dx = 1/y, d(x / y)/dy = -x/y²
                    let left_idx = entry.operand_indices[0];
                    let right_idx = entry.operand_indices[1];

                    let left_val = tape.get_value(left_idx);
                    let right_val = tape.get_value(right_idx);

                    tape.accumulate_gradient(left_idx, output_grad / right_val);
                    tape.accumulate_gradient(
                        right_idx,
                        -output_grad * left_val / (right_val * right_val),
                    );
                }

                // Unary operations
                TapeOp::Neg => {
                    // d(-x)/dx = -1
                    let input_idx = entry.operand_indices[0];
                    tape.accumulate_gradient(input_idx, -output_grad);
                }

                TapeOp::Exp => {
                    // d(e^x)/dx = e^x
                    let input_idx = entry.operand_indices[0];
                    let result_val = tape.get_value(entry.result_idx);
                    tape.accumulate_gradient(input_idx, output_grad * result_val);
                }

                TapeOp::Log => {
                    // d(ln(x))/dx = 1/x
                    let input_idx = entry.operand_indices[0];
                    let input_val = tape.get_value(input_idx);
                    tape.accumulate_gradient(input_idx, output_grad / input_val);
                }

                TapeOp::Sin => {
                    // d(sin(x))/dx = cos(x)
                    let input_idx = entry.operand_indices[0];
                    let input_val = tape.get_value(input_idx);
                    tape.accumulate_gradient(input_idx, output_grad * libm::cos(input_val));
                }

                TapeOp::Cos => {
                    // d(cos(x))/dx = -sin(x)
                    let input_idx = entry.operand_indices[0];
                    let input_val = tape.get_value(input_idx);
                    tape.accumulate_gradient(input_idx, -output_grad * libm::sin(input_val));
                }

                TapeOp::Tan => {
                    // d(tan(x))/dx = sec²(x) = 1/cos²(x)
                    let input_idx = entry.operand_indices[0];
                    let input_val = tape.get_value(input_idx);
                    let cos_val = libm::cos(input_val);
                    tape.accumulate_gradient(input_idx, output_grad / (cos_val * cos_val));
                }

                TapeOp::Sinh => {
                    // d(sinh(x))/dx = cosh(x)
                    let input_idx = entry.operand_indices[0];
                    let input_val = tape.get_value(input_idx);
                    tape.accumulate_gradient(input_idx, output_grad * libm::cosh(input_val));
                }

                TapeOp::Cosh => {
                    // d(cosh(x))/dx = sinh(x)
                    let input_idx = entry.operand_indices[0];
                    let input_val = tape.get_value(input_idx);
                    tape.accumulate_gradient(input_idx, output_grad * libm::sinh(input_val));
                }

                TapeOp::Tanh => {
                    // d(tanh(x))/dx = sech²(x) = 1/cosh²(x)
                    let input_idx = entry.operand_indices[0];
                    let input_val = tape.get_value(input_idx);
                    let cosh_val = libm::cosh(input_val);
                    tape.accumulate_gradient(input_idx, output_grad / (cosh_val * cosh_val));
                }

                TapeOp::Sqrt => {
                    // d(√x)/dx = 1/(2√x)
                    let input_idx = entry.operand_indices[0];
                    let result_val = tape.get_value(entry.result_idx);
                    tape.accumulate_gradient(input_idx, output_grad / (2.0 * result_val));
                }

                TapeOp::PowConst => {
                    // d(x^n)/dx = n*x^(n-1)
                    let input_idx = entry.operand_indices[0];
                    let input_val = tape.get_value(input_idx);
                    let exponent = entry.extra_data.unwrap();

                    let grad = output_grad * exponent * libm::pow(input_val, exponent - 1.0);
                    tape.accumulate_gradient(input_idx, grad);
                }

                TapeOp::Pow => {
                    // d(f^g)/df = g*f^(g-1)
                    // d(f^g)/dg = f^g * ln(f)
                    let base_idx = entry.operand_indices[0];
                    let exp_idx = entry.operand_indices[1];

                    let base_val = tape.get_value(base_idx);
                    let exp_val = tape.get_value(exp_idx);
                    let result_val = tape.get_value(entry.result_idx);

                    let base_grad = output_grad * exp_val * libm::pow(base_val, exp_val - 1.0);
                    let exp_grad = output_grad * result_val * libm::log(base_val);

                    tape.accumulate_gradient(base_idx, base_grad);
                    tape.accumulate_gradient(exp_idx, exp_grad);
                }
            }
        }
    }
}

/// Compute Jacobian matrix using reverse-mode AD
///
/// For f: R^n → R^m, computes the m×n Jacobian matrix where J[i][j] = ∂f_i/∂x_j.
///
/// Reverse-mode is efficient when m << n (few outputs, many inputs).
pub fn compute_jacobian_reverse<F>(f: &F, x: &[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[TapeValue]) -> Vec<TapeValue>,
{
    let tape = Rc::new(RefCell::new(Tape::new()));

    // Create input variables
    let inputs: Vec<TapeValue> = x.iter().map(|&xi| TapeValue::variable(&tape, xi)).collect();

    // Evaluate function
    let outputs = f(&inputs);

    let n_inputs = inputs.len();
    let n_outputs = outputs.len();

    // Compute one row of Jacobian per output
    let mut jacobian = vec![vec![0.0; n_inputs]; n_outputs];

    for (i, output) in outputs.iter().enumerate() {
        // Reset gradients
        tape.borrow_mut().clear_gradients();

        // Backward pass from this output
        output.backward();

        // Extract gradients for this row
        for (j, input) in inputs.iter().enumerate() {
            jacobian[i][j] = input.gradient();
        }
    }

    jacobian
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_backward_add() {
        let tape = Rc::new(RefCell::new(Tape::new()));

        let x = TapeValue::variable(&tape, 3.0);
        let y = TapeValue::variable(&tape, 4.0);
        let z = x.add(&y); // z = x + y = 7

        z.backward();

        assert_eq!(x.gradient(), 1.0); // dz/dx = 1
        assert_eq!(y.gradient(), 1.0); // dz/dy = 1
    }

    #[test]
    fn test_backward_mul() {
        let tape = Rc::new(RefCell::new(Tape::new()));

        let x = TapeValue::variable(&tape, 3.0);
        let y = TapeValue::variable(&tape, 4.0);
        let z = x.mul(&y); // z = x * y = 12

        z.backward();

        assert_eq!(x.gradient(), 4.0); // dz/dx = y = 4
        assert_eq!(y.gradient(), 3.0); // dz/dy = x = 3
    }

    #[test]
    fn test_backward_chain() {
        let tape = Rc::new(RefCell::new(Tape::new()));

        // f(x) = (x + 2) * (x + 3) at x=1
        let x = TapeValue::variable(&tape, 1.0);
        let two = TapeValue::constant(&tape, 2.0);
        let three = TapeValue::constant(&tape, 3.0);

        let a = x.add(&two); // a = x + 2 = 3
        let b = x.add(&three); // b = x + 3 = 4
        let result = a.mul(&b); // result = a * b = 12

        result.backward();

        // df/dx = (x+3) + (x+2) = 2x + 5 = 7
        assert_eq!(x.gradient(), 7.0);
    }

    #[test]
    fn test_backward_exp() {
        let tape = Rc::new(RefCell::new(Tape::new()));

        let x = TapeValue::variable(&tape, 0.0);
        let y = x.exp(); // y = e^x at x=0, so y = 1

        y.backward();

        assert!((x.gradient() - 1.0).abs() < 1e-10); // d(e^x)/dx at x=0 = 1
    }

    #[test]
    fn test_backward_sin() {
        let tape = Rc::new(RefCell::new(Tape::new()));

        let x = TapeValue::variable(&tape, 0.0);
        let y = x.sin(); // y = sin(x) at x=0

        y.backward();

        assert!((x.gradient() - 1.0).abs() < 1e-10); // d(sin(x))/dx at x=0 = cos(0) = 1
    }

    #[test]
    fn test_jacobian_simple() {
        // f(x,y) = [x² + y, x*y]
        let f = |inputs: &[TapeValue]| {
            let x = &inputs[0];
            let y = &inputs[1];

            let out1 = x.mul(x).add(y); // x² + y
            let out2 = x.mul(y); // x * y

            vec![out1, out2]
        };

        let jacobian = compute_jacobian_reverse(&f, &[2.0, 3.0]);

        // Expected:
        // ∂(x²+y)/∂x = 2x = 4, ∂(x²+y)/∂y = 1
        // ∂(xy)/∂x = y = 3, ∂(xy)/∂y = x = 2

        assert!((jacobian[0][0] - 4.0).abs() < 1e-10);
        assert!((jacobian[0][1] - 1.0).abs() < 1e-10);
        assert!((jacobian[1][0] - 3.0).abs() < 1e-10);
        assert!((jacobian[1][1] - 2.0).abs() < 1e-10);
    }
}
