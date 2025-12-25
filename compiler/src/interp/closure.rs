//! Closure adapter for bridging D closures with Rust function traits
//!
//! This module provides the InterpreterClosure type which adapts D function values
//! (Value::Function) to be callable from Rust code, particularly for scientific
//! computing callbacks (ODE solvers, optimization, etc).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use miette::{Result, miette};

use crate::hir::HirFn;

use super::eval::Interpreter;
use super::value::{ControlFlow, Value};

/// A D closure adapted for use in Rust code
///
/// This wrapper allows D functions to be called from Rust code that expects
/// standard Rust closures. It handles:
/// - Type conversion (f64 ↔ Value)
/// - Interpreter state management
/// - Error propagation
///
/// # Example
///
/// In D code:
/// ```d
/// let f = |t: f64, y: [f64]| -> [f64] { [...] }
/// ```
///
/// This can be wrapped and called from Rust:
/// ```ignore
/// let closure = InterpreterClosure::new(func, captures, interpreter);
/// let result = closure.call_with_floats(&[1.0, 2.0])?;
/// ```
pub struct InterpreterClosure {
    /// The HIR function to call
    pub func: Rc<HirFn>,
    /// Captured variables from the closure environment
    pub captures: HashMap<String, Value>,
    /// Shared mutable reference to the interpreter
    pub interpreter: Rc<RefCell<Interpreter>>,
}

impl InterpreterClosure {
    /// Create a new closure adapter
    pub fn new(
        func: Rc<HirFn>,
        captures: HashMap<String, Value>,
        interpreter: Rc<RefCell<Interpreter>>,
    ) -> Self {
        InterpreterClosure {
            func,
            captures,
            interpreter,
        }
    }

    /// Call the closure with f64 arguments (for ODE solvers)
    ///
    /// Converts f64 inputs to the appropriate Value types based on the
    /// function signature, executes the function, and converts the result back.
    ///
    /// # Arguments
    ///
    /// * `args` - Floating point arguments to pass to the closure
    ///
    /// # Returns
    ///
    /// Vector of f64 results from the closure
    pub fn call_with_floats(&self, args: &[f64]) -> Result<Vec<f64>> {
        // Convert f64 arguments to Values
        let values: Vec<Value> = args.iter().map(|&f| Value::Float(f)).collect();

        // Call the function
        let result = {
            let mut interp = self.interpreter.borrow_mut();
            self.call_function(&mut interp, &values)?
        };

        // Convert result back to f64
        match result {
            Value::Float(f) => Ok(vec![f]),
            Value::Array(arr) => {
                let arr = arr.borrow();
                let mut floats = Vec::new();
                for v in arr.iter() {
                    match v {
                        Value::Float(f) => floats.push(*f),
                        Value::Int(n) => floats.push(*n as f64),
                        _ => {
                            return Err(miette!("Expected numeric array, got {:?}", v.type_name()));
                        }
                    }
                }
                Ok(floats)
            }
            Value::Tuple(vals) => {
                let mut floats = Vec::new();
                for v in vals.iter() {
                    match v {
                        Value::Float(f) => floats.push(*f),
                        Value::Int(n) => floats.push(*n as f64),
                        _ => {
                            return Err(miette!("Expected numeric tuple, got {:?}", v.type_name()));
                        }
                    }
                }
                Ok(floats)
            }
            _ => Err(miette!(
                "Closure must return float or array, got {}",
                result.type_name()
            )),
        }
    }

    /// Call the closure with arbitrary Value arguments
    pub fn call_with_values(&self, args: &[Value]) -> Result<Value> {
        let mut interp = self.interpreter.borrow_mut();
        self.call_function(&mut interp, args)
    }

    /// Internal function call implementation
    fn call_function(&self, interp: &mut Interpreter, args: &[Value]) -> Result<Value> {
        interp.env_mut().push_scope();

        // Define captured variables in the new scope
        for (name, value) in &self.captures {
            interp.env_mut().define(name.clone(), value.clone());
        }

        // Bind function parameters
        for (param, arg) in self.func.ty.params.iter().zip(args.iter()) {
            interp.env_mut().define(param.name.clone(), arg.clone());
        }

        // Execute function body
        let result = match interp.eval_block_internal(&self.func.body) {
            Ok(v) => Ok(v),
            Err(ControlFlow::Return(v)) => Ok(v),
            Err(ControlFlow::Break(_)) => Err(miette!("break outside loop")),
            Err(ControlFlow::Continue) => Err(miette!("continue outside loop")),
        };

        interp.env_mut().pop_scope();

        result
    }
}

/// Helper trait for callable D closures
pub trait DCallable {
    /// Call with f64 array arguments
    fn call_f64(&self, args: &[f64]) -> Result<Vec<f64>>;

    /// Call with Value arguments
    fn call_values(&self, args: &[Value]) -> Result<Value>;
}

impl DCallable for InterpreterClosure {
    fn call_f64(&self, args: &[f64]) -> Result<Vec<f64>> {
        self.call_with_floats(args)
    }

    fn call_values(&self, args: &[Value]) -> Result<Value> {
        self.call_with_values(args)
    }
}

/// Extract a closure from a Value
///
/// # Arguments
///
/// * `value` - The value to extract the closure from
/// * `interpreter` - The interpreter instance
///
/// # Returns
///
/// An InterpreterClosure if successful, error otherwise
pub fn extract_closure(
    value: &Value,
    interpreter: Rc<RefCell<Interpreter>>,
) -> Result<InterpreterClosure> {
    match value {
        Value::Function { func, captures } => Ok(InterpreterClosure::new(
            func.clone(),
            captures.clone(),
            interpreter,
        )),
        _ => Err(miette!("Expected function, got {}", value.type_name())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests will be added when builtins are integrated
}
