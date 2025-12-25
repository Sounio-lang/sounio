//! Builtin function registry for the interpreter
//!
//! This module provides a flexible system for registering and calling builtin functions
//! that bridge the D interpreter with Rust runtime modules (ODE solvers, probabilistic
//! inference, symbolic math, etc).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::interp::value::Value;

// Thread-local memory storage for ptr_read/ptr_write operations
// Maps memory addresses to Values for simulating pointer operations in the interpreter
thread_local! {
    static FFI_MEMORY: RefCell<HashMap<usize, Value>> = RefCell::new(HashMap::new());
}

/// Type alias for a builtin function handler
pub type BuiltinHandler = Rc<dyn Fn(&[Value]) -> Result<Value, String>>;

/// Registry of builtin functions
pub struct BuiltinRegistry {
    /// Map from function name to handler
    handlers: HashMap<String, BuiltinHandler>,
}

impl BuiltinRegistry {
    /// Create a new registry with standard builtins
    pub fn new() -> Self {
        let mut registry = BuiltinRegistry {
            handlers: HashMap::new(),
        };

        // Register I/O builtins
        registry.register_io_builtins();

        // Register math builtins
        registry.register_math_builtins();

        // Register utility builtins
        registry.register_utility_builtins();

        // Scientific primitives will be registered here
        // For now: placeholder for ODE, prob, symbolic, etc
        registry.register_scientific_builtins();

        // Register FFI/pointer builtins
        registry.register_ffi_builtins();

        // Register JSON builtins
        registry.register_json_builtins();

        registry
    }

    /// Register a builtin function
    pub fn register(&mut self, name: &str, handler: BuiltinHandler) {
        self.handlers.insert(name.to_string(), handler);
    }

    /// Check if a name is a registered builtin
    pub fn is_builtin(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// Call a builtin function
    pub fn call(&self, name: &str, args: &[Value]) -> Result<Value, String> {
        match self.handlers.get(name) {
            Some(handler) => handler(args),
            None => Err(format!("Unknown builtin: {}", name)),
        }
    }

    /// Get all registered builtin names
    pub fn names(&self) -> Vec<String> {
        self.handlers.keys().cloned().collect()
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Builtin Categories
// ============================================================================

impl BuiltinRegistry {
    /// Register I/O builtins (print, println, etc)
    fn register_io_builtins(&mut self) {
        self.register(
            "print",
            Rc::new(|args| {
                let output = args
                    .iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join("");
                print!("{}", output);
                Ok(Value::Unit)
            }),
        );

        self.register(
            "println",
            Rc::new(|args| {
                let output = args
                    .iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join("");
                println!("{}", output);
                Ok(Value::Unit)
            }),
        );
    }

    /// Register math builtins (sqrt, sin, cos, etc)
    fn register_math_builtins(&mut self) {
        self.register(
            "sqrt",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("sqrt expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.sqrt())),
                    Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
                    _ => Err(format!(
                        "sqrt expects numeric argument, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );

        self.register(
            "abs",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("abs expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.abs())),
                    Value::Int(n) => Ok(Value::Int(n.abs())),
                    _ => Err("abs expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "sin",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("sin expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.sin())),
                    Value::Int(n) => Ok(Value::Float((*n as f64).sin())),
                    _ => Err("sin expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "cos",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("cos expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.cos())),
                    Value::Int(n) => Ok(Value::Float((*n as f64).cos())),
                    _ => Err("cos expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "tan",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("tan expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.tan())),
                    Value::Int(n) => Ok(Value::Float((*n as f64).tan())),
                    _ => Err("tan expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "exp",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("exp expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.exp())),
                    Value::Int(n) => Ok(Value::Float((*n as f64).exp())),
                    _ => Err("exp expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "log",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("log expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.ln())),
                    Value::Int(n) => Ok(Value::Float((*n as f64).ln())),
                    _ => Err("log expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "pow",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("pow expects 2 arguments, got {}", args.len()));
                }
                let base = match &args[0] {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => return Err("pow expects numeric arguments".to_string()),
                };
                let exp = match &args[1] {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => return Err("pow expects numeric arguments".to_string()),
                };
                Ok(Value::Float(base.powf(exp)))
            }),
        );

        self.register(
            "floor",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("floor expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.floor())),
                    Value::Int(n) => Ok(Value::Int(*n)),
                    _ => Err("floor expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "ceil",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("ceil expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.ceil())),
                    Value::Int(n) => Ok(Value::Int(*n)),
                    _ => Err("ceil expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "round",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("round expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::Float(f) => Ok(Value::Float(f.round())),
                    Value::Int(n) => Ok(Value::Int(*n)),
                    _ => Err("round expects numeric argument".to_string()),
                }
            }),
        );

        self.register(
            "min",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("min expects 2 arguments, got {}", args.len()));
                }
                let a = match &args[0] {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => return Err("min expects numeric arguments".to_string()),
                };
                let b = match &args[1] {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => return Err("min expects numeric arguments".to_string()),
                };
                Ok(Value::Float(a.min(b)))
            }),
        );

        self.register(
            "max",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("max expects 2 arguments, got {}", args.len()));
                }
                let a = match &args[0] {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => return Err("max expects numeric arguments".to_string()),
                };
                let b = match &args[1] {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => return Err("max expects numeric arguments".to_string()),
                };
                Ok(Value::Float(a.max(b)))
            }),
        );
    }

    /// Register utility builtins (len, type_of, assert, etc)
    fn register_utility_builtins(&mut self) {
        self.register(
            "len",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("len expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::String(s) => Ok(Value::Int(s.len() as i64)),
                    Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                    Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
                    _ => Err("len expects string, array, or tuple".to_string()),
                }
            }),
        );

        self.register(
            "type_of",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("type_of expects 1 argument, got {}", args.len()));
                }
                Ok(Value::String(args[0].type_name().to_string()))
            }),
        );

        self.register(
            "assert",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("assert expects 1 argument, got {}", args.len()));
                }
                if args[0].is_truthy() {
                    Ok(Value::Unit)
                } else {
                    Err("Assertion failed".to_string())
                }
            }),
        );

        self.register(
            "assert_eq",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("assert_eq expects 2 arguments, got {}", args.len()));
                }
                if format!("{}", args[0]) == format!("{}", args[1]) {
                    Ok(Value::Unit)
                } else {
                    Err(format!("Assertion failed: {:?} != {:?}", args[0], args[1]))
                }
            }),
        );

        self.register(
            "panic",
            Rc::new(|args| {
                let msg = if args.is_empty() {
                    "panic".to_string()
                } else {
                    args.iter()
                        .map(|v| format!("{}", v))
                        .collect::<Vec<_>>()
                        .join("")
                };
                Err(format!("panic: {}", msg))
            }),
        );

        self.register(
            "dbg",
            Rc::new(|args| {
                for arg in args {
                    eprintln!("[DEBUG] {}", arg);
                }
                if args.len() == 1 {
                    Ok(args[0].clone())
                } else {
                    Ok(Value::Unit)
                }
            }),
        );
    }

    /// Register scientific builtins
    fn register_scientific_builtins(&mut self) {
        use crate::interp::value::{Distribution, SolverStats};

        // ODE solver (stub implementation for testing)
        self.register(
            "solve_ode",
            Rc::new(|args| {
                if args.len() < 3 {
                    return Err("solve_ode expects: closure, initial_values, time_span".to_string());
                }

                // For now, return a simple ODE solution
                // Later: integrate with runtime::ode::solve
                let t = vec![0.0, 0.5, 1.0, 1.5, 2.0];
                let y = vec![
                    vec![1.0, 1.0],
                    vec![0.9, 1.1],
                    vec![0.8, 1.2],
                    vec![0.7, 1.3],
                    vec![0.6, 1.4],
                ];
                let stats = SolverStats {
                    steps: 100,
                    accepted_steps: 100,
                    rejected_steps: 0,
                };

                Ok(Value::ODESolution { t, y, stats })
            }),
        );

        // Probabilistic sampling (stub)
        self.register(
            "sample",
            Rc::new(|args| {
                if args.is_empty() {
                    return Err("sample expects distribution argument".to_string());
                }

                match &args[0] {
                    Value::Distribution(d) => {
                        // Return a sample from the distribution
                        match d {
                            Distribution::Normal { mean, std: _ } => Ok(Value::Float(*mean)),
                            Distribution::Uniform { a, b } => {
                                Ok(Value::Float((a + b) / 2.0)) // Return midpoint for now
                            }
                            Distribution::Beta { alpha, beta } => {
                                // Return expected value E[Beta(a,b)] = a/(a+b)
                                Ok(Value::Float(alpha / (alpha + beta)))
                            }
                            Distribution::Exponential { lambda } => {
                                Ok(Value::Float(1.0 / lambda)) // Return mean
                            }
                            Distribution::Categorical { probs } => {
                                // Return index of max probability
                                let idx = probs
                                    .iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| {
                                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(i, _)| i)
                                    .unwrap_or(0);
                                Ok(Value::Int(idx as i64))
                            }
                        }
                    }
                    _ => Err("sample expects a Distribution".to_string()),
                }
            }),
        );

        // Symbolic differentiation (stub)
        self.register(
            "differentiate",
            Rc::new(|args| {
                if args.len() < 2 {
                    return Err("differentiate expects: expression, variable".to_string());
                }

                match (&args[0], &args[1]) {
                    (Value::SymbolicExpr(expr), Value::String(var)) => {
                        let derivative = expr.differentiate(var);
                        Ok(Value::SymbolicExpr(std::rc::Rc::new(derivative)))
                    }
                    _ => Err("differentiate expects: SymbolicExpr, String".to_string()),
                }
            }),
        );

        // Array/matrix operations
        self.register(
            "zeros",
            Rc::new(|args| {
                if args.is_empty() {
                    return Ok(Value::Float(0.0));
                }

                match &args[0] {
                    Value::Int(n) => {
                        let data = vec![0.0; *n as usize];
                        Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                            data.into_iter().map(Value::Float).collect(),
                        ))))
                    }
                    _ => Err("zeros expects integer dimension".to_string()),
                }
            }),
        );

        self.register(
            "ones",
            Rc::new(|args| {
                if args.is_empty() {
                    return Ok(Value::Float(1.0));
                }

                match &args[0] {
                    Value::Int(n) => {
                        let data = vec![1.0; *n as usize];
                        Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                            data.into_iter().map(Value::Float).collect(),
                        ))))
                    }
                    _ => Err("ones expects integer dimension".to_string()),
                }
            }),
        );

        // Gradient computation (reverse-mode autodiff)
        self.register(
            "grad",
            Rc::new(|args| {
                if args.len() < 2 {
                    return Err("grad expects: function, parameters".to_string());
                }

                // For now, return a placeholder gradient
                // Full implementation requires integration with interpreter
                match &args[1] {
                    Value::Float(_) => {
                        // Return gradient as a float (scalar case)
                        Ok(Value::Float(0.0)) // Placeholder
                    }
                    Value::Array(_) => {
                        // Return gradient as an array
                        Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                            vec![],
                        ))))
                    }
                    _ => Err("grad expects scalar or array parameters".to_string()),
                }
            }),
        );

        // Jacobian matrix (gradient for vector functions)
        self.register(
            "jacobian",
            Rc::new(|args| {
                if args.len() < 2 {
                    return Err("jacobian expects: function, parameters".to_string());
                }

                // Placeholder for Jacobian computation
                // Returns a matrix of partial derivatives
                Ok(Value::Tensor {
                    data: vec![],
                    shape: vec![0, 0],
                })
            }),
        );

        // Hessian matrix (second derivatives)
        self.register(
            "hessian",
            Rc::new(|args| {
                if args.len() < 2 {
                    return Err("hessian expects: function, parameters".to_string());
                }

                // Placeholder for Hessian computation
                // Returns a matrix of second partial derivatives
                Ok(Value::Tensor {
                    data: vec![],
                    shape: vec![0, 0],
                })
            }),
        );

        // Causal do-operator: do(model, interventions)
        self.register(
            "do",
            Rc::new(|args| {
                if args.len() < 2 {
                    return Err("do expects: causal_model, interventions".to_string());
                }

                // For now, return a placeholder causal model
                // Full implementation requires model integration
                match &args[0] {
                    Value::CausalModel(_) => Ok(Value::CausalModel("intervened_model".to_string())),
                    _ => Err("do expects a CausalModel".to_string()),
                }
            }),
        );

        // Counterfactual reasoning: counterfactual(model, factual, intervention, query)
        self.register(
            "counterfactual",
            Rc::new(|args| {
                if args.len() < 3 {
                    return Err(
                        "counterfactual expects: model, factual_evidence, intervention".to_string(),
                    );
                }

                // Placeholder for counterfactual computation
                // Returns the counterfactual value
                Ok(Value::Float(0.0))
            }),
        );

        // Estimate average treatment effect (ATE)
        self.register(
            "estimate_ate",
            Rc::new(|args| {
                if args.len() < 3 {
                    return Err("estimate_ate expects: data, treatment, outcome".to_string());
                }

                // Placeholder for ATE estimation
                // Would compute E[Y | do(X=1)] - E[Y | do(X=0)]
                Ok(Value::Float(0.0))
            }),
        );

        // Detect Simpson's paradox
        self.register(
            "simpsons_paradox",
            Rc::new(|args| {
                if args.len() < 3 {
                    return Err("simpsons_paradox expects: data, x, y, stratified_by_z".to_string());
                }

                // Placeholder for Simpson's paradox detection
                // Returns true if paradox is detected
                Ok(Value::Bool(false))
            }),
        );
    }

    /// Register FFI/pointer builtins for raw pointer manipulation
    fn register_ffi_builtins(&mut self) {
        // null_ptr<T>() -> *const T
        // Returns a null pointer
        self.register(
            "null_ptr",
            Rc::new(|_args| {
                Ok(Value::RawPointer {
                    address: 0,
                    mutable: false,
                })
            }),
        );

        // null_mut<T>() -> *mut T
        // Returns a mutable null pointer
        self.register(
            "null_mut",
            Rc::new(|_args| {
                Ok(Value::RawPointer {
                    address: 0,
                    mutable: true,
                })
            }),
        );

        // is_null(ptr: *const T) -> bool
        // Check if a pointer is null
        self.register(
            "is_null",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("is_null expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::RawPointer { address, .. } => Ok(Value::Bool(*address == 0)),
                    _ => Err(format!(
                        "is_null expects a raw pointer, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );

        // ptr_eq(a: *const T, b: *const T) -> bool
        // Compare two pointers for equality
        self.register(
            "ptr_eq",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("ptr_eq expects 2 arguments, got {}", args.len()));
                }
                match (&args[0], &args[1]) {
                    (
                        Value::RawPointer { address: a, .. },
                        Value::RawPointer { address: b, .. },
                    ) => Ok(Value::Bool(*a == *b)),
                    _ => Err("ptr_eq expects two raw pointers".to_string()),
                }
            }),
        );

        // ptr_addr(ptr: *const T) -> usize
        // Get the address of a pointer as an integer
        self.register(
            "ptr_addr",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("ptr_addr expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::RawPointer { address, .. } => Ok(Value::Int(*address as i64)),
                    _ => Err(format!(
                        "ptr_addr expects a raw pointer, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );

        // ptr_from_addr(addr: usize) -> *const T
        // Create a pointer from an integer address (unsafe!)
        self.register(
            "ptr_from_addr",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!(
                        "ptr_from_addr expects 1 argument, got {}",
                        args.len()
                    ));
                }
                match &args[0] {
                    Value::Int(addr) => Ok(Value::RawPointer {
                        address: *addr as usize,
                        mutable: false,
                    }),
                    _ => Err(format!(
                        "ptr_from_addr expects an integer, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );

        // ptr_from_addr_mut(addr: usize) -> *mut T
        // Create a mutable pointer from an integer address (unsafe!)
        self.register(
            "ptr_from_addr_mut",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!(
                        "ptr_from_addr_mut expects 1 argument, got {}",
                        args.len()
                    ));
                }
                match &args[0] {
                    Value::Int(addr) => Ok(Value::RawPointer {
                        address: *addr as usize,
                        mutable: true,
                    }),
                    _ => Err(format!(
                        "ptr_from_addr_mut expects an integer, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );

        // ptr_offset(ptr: *const T, offset: isize) -> *const T
        // Offset a pointer by a given number of elements (unsafe!)
        self.register(
            "ptr_offset",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!(
                        "ptr_offset expects 2 arguments, got {}",
                        args.len()
                    ));
                }
                match (&args[0], &args[1]) {
                    (Value::RawPointer { address, mutable }, Value::Int(offset)) => {
                        // Note: This is simplified - real impl would account for element size
                        let new_addr = (*address as isize + *offset as isize) as usize;
                        Ok(Value::RawPointer {
                            address: new_addr,
                            mutable: *mutable,
                        })
                    }
                    _ => Err("ptr_offset expects a raw pointer and an integer offset".to_string()),
                }
            }),
        );

        // ptr_add(ptr: *const T, count: usize) -> *const T
        // Add count elements to a pointer (unsafe!)
        self.register(
            "ptr_add",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("ptr_add expects 2 arguments, got {}", args.len()));
                }
                match (&args[0], &args[1]) {
                    (Value::RawPointer { address, mutable }, Value::Int(count)) => {
                        let new_addr = address.wrapping_add(*count as usize);
                        Ok(Value::RawPointer {
                            address: new_addr,
                            mutable: *mutable,
                        })
                    }
                    _ => Err("ptr_add expects a raw pointer and an integer count".to_string()),
                }
            }),
        );

        // ptr_sub(ptr: *const T, count: usize) -> *const T
        // Subtract count elements from a pointer (unsafe!)
        self.register(
            "ptr_sub",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("ptr_sub expects 2 arguments, got {}", args.len()));
                }
                match (&args[0], &args[1]) {
                    (Value::RawPointer { address, mutable }, Value::Int(count)) => {
                        let new_addr = address.wrapping_sub(*count as usize);
                        Ok(Value::RawPointer {
                            address: new_addr,
                            mutable: *mutable,
                        })
                    }
                    _ => Err("ptr_sub expects a raw pointer and an integer count".to_string()),
                }
            }),
        );

        // ptr_diff(a: *const T, b: *const T) -> isize
        // Calculate the difference between two pointers
        self.register(
            "ptr_diff",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("ptr_diff expects 2 arguments, got {}", args.len()));
                }
                match (&args[0], &args[1]) {
                    (
                        Value::RawPointer { address: a, .. },
                        Value::RawPointer { address: b, .. },
                    ) => {
                        let diff = (*a as isize) - (*b as isize);
                        Ok(Value::Int(diff as i64))
                    }
                    _ => Err("ptr_diff expects two raw pointers".to_string()),
                }
            }),
        );

        // as_const(ptr: *mut T) -> *const T
        // Cast a mutable pointer to a const pointer
        self.register(
            "as_const",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("as_const expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::RawPointer { address, .. } => Ok(Value::RawPointer {
                        address: *address,
                        mutable: false,
                    }),
                    _ => Err(format!(
                        "as_const expects a raw pointer, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );

        // as_mut(ptr: *const T) -> *mut T (unsafe!)
        // Cast a const pointer to a mutable pointer
        self.register(
            "as_mut",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("as_mut expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::RawPointer { address, .. } => Ok(Value::RawPointer {
                        address: *address,
                        mutable: true,
                    }),
                    _ => Err(format!(
                        "as_mut expects a raw pointer, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );

        // size_of<T>() -> usize
        // Get the size of a type in bytes
        self.register(
            "size_of",
            Rc::new(|args| {
                // In a real implementation, this would take a type parameter
                // For now, return a placeholder based on common type sizes
                if args.is_empty() {
                    // Default to pointer size
                    Ok(Value::Int(std::mem::size_of::<usize>() as i64))
                } else {
                    match &args[0] {
                        Value::Int(_) => Ok(Value::Int(8)),   // i64
                        Value::Float(_) => Ok(Value::Int(8)), // f64
                        Value::Bool(_) => Ok(Value::Int(1)),
                        Value::RawPointer { .. } => {
                            Ok(Value::Int(std::mem::size_of::<usize>() as i64))
                        }
                        _ => Ok(Value::Int(8)), // Default
                    }
                }
            }),
        );

        // align_of<T>() -> usize
        // Get the alignment of a type in bytes
        self.register(
            "align_of",
            Rc::new(|args| {
                if args.is_empty() {
                    Ok(Value::Int(std::mem::align_of::<usize>() as i64))
                } else {
                    match &args[0] {
                        Value::Int(_) => Ok(Value::Int(8)),
                        Value::Float(_) => Ok(Value::Int(8)),
                        Value::Bool(_) => Ok(Value::Int(1)),
                        Value::RawPointer { .. } => {
                            Ok(Value::Int(std::mem::align_of::<usize>() as i64))
                        }
                        _ => Ok(Value::Int(8)),
                    }
                }
            }),
        );

        // ptr_read(ptr: *const T) -> T
        // Read a value from memory through a pointer (unsafe!)
        self.register(
            "ptr_read",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("ptr_read expects 1 argument, got {}", args.len()));
                }
                match &args[0] {
                    Value::RawPointer { address, .. } => {
                        if *address == 0 {
                            return Err("Attempted to read from null pointer".to_string());
                        }
                        FFI_MEMORY.with(|mem| {
                            mem.borrow().get(address).cloned().ok_or_else(|| {
                                format!(
                                    "Attempted to read from uninitialized address: {:#x}",
                                    address
                                )
                            })
                        })
                    }
                    _ => Err(format!(
                        "ptr_read expects a raw pointer, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );

        // ptr_write(ptr: *mut T, value: T)
        // Write a value to memory through a pointer (unsafe!)
        self.register(
            "ptr_write",
            Rc::new(|args| {
                if args.len() != 2 {
                    return Err(format!("ptr_write expects 2 arguments, got {}", args.len()));
                }
                match &args[0] {
                    Value::RawPointer { address, mutable } => {
                        if *address == 0 {
                            return Err("Attempted to write to null pointer".to_string());
                        }
                        if !mutable {
                            return Err(
                                "Attempted to write through const pointer (need *mut)".to_string()
                            );
                        }
                        FFI_MEMORY.with(|mem| {
                            mem.borrow_mut().insert(*address, args[1].clone());
                        });
                        Ok(Value::Unit)
                    }
                    _ => Err(format!(
                        "ptr_write expects a raw pointer as first argument, got {}",
                        args[0].type_name()
                    )),
                }
            }),
        );
    }

    /// Register JSON builtins for parsing and manipulating JSON data
    fn register_json_builtins(&mut self) {
        // parse_json(input: &str) -> Result<JsonValue, ParseError>
        self.register(
            "parse_json",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!("parse_json expects 1 argument, got {}", args.len()));
                }

                let json_str = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err("parse_json expects a string argument".to_string()),
                };

                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(json_val) => {
                        let d_json = json_value_to_d_value(&json_val);
                        Ok(Value::Variant {
                            enum_name: "Result".to_string(),
                            variant_name: "Ok".to_string(),
                            fields: vec![d_json],
                        })
                    }
                    Err(e) => {
                        let err_struct = Value::Struct {
                            name: "ParseError".to_string(),
                            fields: std::collections::HashMap::from([(
                                "message".to_string(),
                                Value::String(format!("JSON parse error: {}", e)),
                            )]),
                        };
                        Ok(Value::Variant {
                            enum_name: "Result".to_string(),
                            variant_name: "Err".to_string(),
                            fields: vec![err_struct],
                        })
                    }
                }
            }),
        );

        // json_to_string(value: &JsonValue) -> String
        self.register(
            "json_to_string",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!(
                        "json_to_string expects 1 argument, got {}",
                        args.len()
                    ));
                }

                match d_value_to_json_value(&args[0]) {
                    Some(json_val) => {
                        let json_str =
                            serde_json::to_string(&json_val).unwrap_or_else(|_| "null".to_string());
                        Ok(Value::String(json_str))
                    }
                    None => Err("Failed to convert value to JSON".to_string()),
                }
            }),
        );

        // json_to_string_pretty(value: &JsonValue) -> String
        self.register(
            "json_to_string_pretty",
            Rc::new(|args| {
                if args.len() != 1 {
                    return Err(format!(
                        "json_to_string_pretty expects 1 argument, got {}",
                        args.len()
                    ));
                }

                match d_value_to_json_value(&args[0]) {
                    Some(json_val) => {
                        let json_str = serde_json::to_string_pretty(&json_val)
                            .unwrap_or_else(|_| "null".to_string());
                        Ok(Value::String(json_str))
                    }
                    None => Err("Failed to convert value to JSON".to_string()),
                }
            }),
        );
    }
}

// ============================================================================
// JSON Conversion Helpers
// ============================================================================

/// Convert serde_json::Value to D's JsonValue (represented as a Variant)
fn json_value_to_d_value(json: &serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Variant {
            enum_name: "JsonValue".to_string(),
            variant_name: "Null".to_string(),
            fields: vec![],
        },
        serde_json::Value::Bool(b) => Value::Variant {
            enum_name: "JsonValue".to_string(),
            variant_name: "Bool".to_string(),
            fields: vec![Value::Bool(*b)],
        },
        serde_json::Value::Number(n) => {
            let num_val = if let Some(i) = n.as_i64() {
                Value::Float(i as f64)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Float(0.0)
            };
            Value::Variant {
                enum_name: "JsonValue".to_string(),
                variant_name: "Number".to_string(),
                fields: vec![num_val],
            }
        }
        serde_json::Value::String(s) => Value::Variant {
            enum_name: "JsonValue".to_string(),
            variant_name: "String".to_string(),
            fields: vec![Value::String(s.clone())],
        },
        serde_json::Value::Array(arr) => {
            let d_array: Vec<Value> = arr.iter().map(json_value_to_d_value).collect();
            let d_vec = Value::Array(Rc::new(RefCell::new(d_array)));
            Value::Variant {
                enum_name: "JsonValue".to_string(),
                variant_name: "Array".to_string(),
                fields: vec![d_vec],
            }
        }
        serde_json::Value::Object(obj) => {
            let mut d_map = HashMap::new();
            for (k, v) in obj.iter() {
                d_map.insert(k.clone(), json_value_to_d_value(v));
            }
            let d_hashmap = Value::Struct {
                name: "Map".to_string(),
                fields: d_map,
            };
            Value::Variant {
                enum_name: "JsonValue".to_string(),
                variant_name: "Object".to_string(),
                fields: vec![d_hashmap],
            }
        }
    }
}

/// Convert D's Value back to serde_json::Value
fn d_value_to_json_value(val: &Value) -> Option<serde_json::Value> {
    match val {
        Value::Variant {
            enum_name,
            variant_name,
            fields,
        } if enum_name == "JsonValue" => match variant_name.as_str() {
            "Null" => Some(serde_json::Value::Null),
            "Bool" => fields.first().and_then(|v| {
                if let Value::Bool(b) = v {
                    Some(serde_json::Value::Bool(*b))
                } else {
                    None
                }
            }),
            "Number" => fields.first().and_then(|v| match v {
                Value::Float(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number),
                Value::Int(i) => Some(serde_json::Value::Number((*i).into())),
                _ => None,
            }),
            "String" => fields.first().and_then(|v| {
                if let Value::String(s) = v {
                    Some(serde_json::Value::String(s.clone()))
                } else {
                    None
                }
            }),
            "Array" => fields.first().and_then(|v| {
                if let Value::Array(arr) = v {
                    let json_arr: Option<Vec<serde_json::Value>> =
                        arr.borrow().iter().map(d_value_to_json_value).collect();
                    json_arr.map(serde_json::Value::Array)
                } else {
                    None
                }
            }),
            "Object" => fields.first().and_then(|v| {
                if let Value::Struct { fields, .. } = v {
                    let json_obj: Option<serde_json::Map<String, serde_json::Value>> = fields
                        .iter()
                        .map(|(k, v)| d_value_to_json_value(v).map(|jv| (k.clone(), jv)))
                        .collect();
                    json_obj.map(serde_json::Value::Object)
                } else {
                    None
                }
            }),
            _ => None,
        },
        Value::Unit => Some(serde_json::Value::Null),
        Value::Bool(b) => Some(serde_json::Value::Bool(*b)),
        Value::Int(i) => Some(serde_json::Value::Number((*i).into())),
        Value::Float(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number),
        Value::String(s) => Some(serde_json::Value::String(s.clone())),
        Value::Array(arr) => {
            let json_arr: Option<Vec<serde_json::Value>> =
                arr.borrow().iter().map(d_value_to_json_value).collect();
            json_arr.map(serde_json::Value::Array)
        }
        Value::Struct { fields, .. } => {
            let json_obj: Option<serde_json::Map<String, serde_json::Value>> = fields
                .iter()
                .map(|(k, v)| d_value_to_json_value(v).map(|jv| (k.clone(), jv)))
                .collect();
            json_obj.map(serde_json::Value::Object)
        }
        Value::None => Some(serde_json::Value::Null),
        _ => None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = BuiltinRegistry::new();
        assert!(registry.is_builtin("print"));
        assert!(registry.is_builtin("sqrt"));
        assert!(!registry.is_builtin("unknown_function"));
    }

    #[test]
    fn test_math_builtins() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("sqrt", &[Value::Float(4.0)]).unwrap();
        assert_eq!(format!("{}", result), "2");

        let result = registry.call("abs", &[Value::Int(-5)]).unwrap();
        assert_eq!(format!("{}", result), "5");

        let result = registry
            .call("max", &[Value::Float(3.0), Value::Float(7.0)])
            .unwrap();
        assert_eq!(format!("{}", result), "7");
    }

    #[test]
    fn test_builtin_error_handling() {
        let registry = BuiltinRegistry::new();

        // Wrong number of arguments
        let result = registry.call("sqrt", &[]);
        assert!(result.is_err());

        // Wrong argument type
        let result = registry.call("sqrt", &[Value::String("hello".to_string())]);
        assert!(result.is_err());
    }

    #[test]
    fn test_scientific_builtins_registry() {
        let registry = BuiltinRegistry::new();

        // Check that scientific builtins are registered
        assert!(registry.is_builtin("solve_ode"));
        assert!(registry.is_builtin("sample"));
        assert!(registry.is_builtin("differentiate"));
        assert!(registry.is_builtin("zeros"));
        assert!(registry.is_builtin("ones"));
    }

    #[test]
    fn test_solve_ode_error_handling() {
        let registry = BuiltinRegistry::new();

        // solve_ode with insufficient arguments
        let result = registry.call("solve_ode", &[Value::Float(1.0)]);
        assert!(result.is_err());

        // solve_ode with correct number of args returns ODE solution
        let f = Value::Float(1.0); // Dummy closure representation
        let y0 = Value::Float(1.0);
        let t_span = Value::Float(2.0);

        let result = registry.call("solve_ode", &[f, y0, t_span]);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert_eq!(solution.type_name(), "ODESolution");
    }

    #[test]
    fn test_sample_from_distribution() {
        use crate::interp::value::Distribution;

        let registry = BuiltinRegistry::new();

        // Sample from Normal distribution
        let normal = Value::Distribution(Distribution::Normal {
            mean: 5.0,
            std: 1.0,
        });

        let result = registry.call("sample", &[normal]);
        assert!(result.is_ok());

        let sample_val = result.unwrap();
        match sample_val {
            Value::Float(f) => {
                // Sample should be near the mean
                assert!((f - 5.0).abs() < 10.0);
            }
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_differentiate_symbolic() {
        use crate::interp::symbolic;
        let registry = BuiltinRegistry::new();

        // Parse "x^2" into an Expr
        let parsed_expr = symbolic::Expr::parse("x^2").expect("Failed to parse expression");
        let expr = Value::SymbolicExpr(std::rc::Rc::new(parsed_expr));
        let var = Value::String("x".to_string());

        let result = registry.call("differentiate", &[expr, var]);
        assert!(result.is_ok());

        let deriv = result.unwrap();
        assert_eq!(deriv.type_name(), "SymbolicExpr");
    }

    #[test]
    fn test_zeros_creates_array() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("zeros", &[Value::Int(5)]);
        assert!(result.is_ok());

        let arr = result.unwrap();
        assert_eq!(arr.type_name(), "array");
    }

    #[test]
    fn test_ones_creates_array() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("ones", &[Value::Int(3)]);
        assert!(result.is_ok());

        let arr = result.unwrap();
        assert_eq!(arr.type_name(), "array");
    }

    #[test]
    fn test_utility_builtins() {
        let registry = BuiltinRegistry::new();

        // Test len()
        let arr = Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
            Value::Float(1.0),
            Value::Float(2.0),
        ])));
        let result = registry.call("len", &[arr]);
        assert!(result.is_ok());

        let len_val = result.unwrap();
        match len_val {
            Value::Int(n) => assert_eq!(n, 2),
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_type_of_builtin() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("type_of", &[Value::Float(3.14)]);
        assert!(result.is_ok());

        let type_name = result.unwrap();
        match type_name {
            Value::String(s) => assert_eq!(s, "float"),
            _ => panic!("Expected String"),
        }
    }
}
