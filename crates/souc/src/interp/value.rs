//! Runtime values for the interpreter

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::hir::HirFn;
use crate::interp::symbolic::Expr as SymbolicExprType;
use crate::runtime::async_runtime::{SounioFuture, TaskHandle, TaskId};

/// ODE solver statistics
#[derive(Clone, Debug, Default)]
pub struct SolverStats {
    /// Number of steps taken
    pub steps: usize,
    /// Number of successful steps
    pub accepted_steps: usize,
    /// Number of rejected steps
    pub rejected_steps: usize,
}

/// Probabilistic distributions
#[derive(Clone, Debug)]
pub enum Distribution {
    /// Normal distribution with mean and std dev
    Normal { mean: f64, std: f64 },
    /// Uniform distribution between a and b
    Uniform { a: f64, b: f64 },
    /// Beta distribution with shape parameters
    Beta { alpha: f64, beta: f64 },
    /// Exponential distribution with rate
    Exponential { lambda: f64 },
    /// Categorical/Discrete distribution
    Categorical { probs: Vec<f64> },
}

/// Strategy for fusing neural and symbolic components in hybrid models
#[derive(Clone, Debug, PartialEq)]
pub enum HybridFusion {
    /// Weighted sum: α * neural + (1-α) * symbolic, where α is learned
    WeightedSum,
    /// Learned gating: g(x) * neural + (1-g(x)) * symbolic, where g is a learned gate function
    LearnedGate,
    /// Element-wise product: neural * symbolic
    Product,
    /// Product then sum: (neural * symbolic) + residual
    ProductResidual,
}

/// Runtime value
#[derive(Clone)]
pub enum Value {
    Dummy,
    /// Unit value `()`
    Unit,
    /// Boolean
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit float
    Float(f64),
    /// String
    String(String),
    /// Array (mutable interior)
    Array(Rc<RefCell<Vec<Value>>>),
    /// Tuple
    Tuple(Vec<Value>),
    /// Struct instance
    Struct {
        name: String,
        fields: HashMap<String, Value>,
    },
    /// Enum variant
    Variant {
        enum_name: String,
        variant_name: String,
        fields: Vec<Value>,
    },
    /// Function closure
    Function {
        func: Rc<HirFn>,
        /// Captured environment for closures
        captures: HashMap<String, Value>,
    },
    /// Reference to a value
    Ref(Rc<RefCell<Value>>),
    /// Reference to an array element (array + index) - allows mutation of elements in place
    ArrayRef {
        array: Rc<RefCell<Vec<Value>>>,
        index: usize,
    },
    /// Raw pointer (for FFI) - stores address and mutability
    RawPointer { address: usize, mutable: bool },
    /// Option::None
    None,
    /// Option::Some(value)
    Some(Box<Value>),
    /// Result::Ok(value)
    Ok(Box<Value>),
    /// Result::Err(value)
    Err(Box<Value>),
    /// Builtin function (by name)
    Builtin(String),

    // Scientific types
    /// ODE solution: time points, solution trajectories, solver stats
    ODESolution {
        t: Vec<f64>,
        y: Vec<Vec<f64>>,
        stats: SolverStats,
    },
    /// Probability distribution
    Distribution(Distribution),
    /// Symbolic mathematical expression
    SymbolicExpr(Rc<SymbolicExprType>),
    /// Multi-dimensional tensor/array
    Tensor { data: Vec<f64>, shape: Vec<usize> },
    /// Sparse tensor in various formats
    SparseTensor(Rc<crate::codegen::gpu::sparse::SparseTensor>),
    /// Value with uncertainty bounds (mean ± std)
    Uncertain { mean: f64, std: f64 },
    /// Causal model representation
    CausalModel(String),
    /// Hybrid neural-symbolic model
    HybridModel {
        /// Neural component parameters
        neural_params: Rc<RefCell<Vec<f64>>>,
        /// Symbolic expression component
        symbolic_expr: Rc<SymbolicExprType>,
        /// Fusion strategy: how to combine neural and symbolic outputs
        fusion: HybridFusion,
    },

    // Async types
    /// A future that represents an async computation
    Future {
        /// The underlying Sounio future
        future: SounioFuture,
    },
    /// A handle to a spawned async task
    Task {
        /// The task handle
        handle: TaskHandle,
    },
    /// Epistemic value with confidence tracking
    Knowledge {
        /// The wrapped value
        value: Box<Value>,
        /// Confidence level (0.0 to 1.0); meaningless when knightian=true
        confidence: f64,
        /// Knightian undefined: true means ε=⊥ (no probability model applies).
        /// Distinct from confidence=0.0 ("known false"). Propagates through computation.
        knightian: bool,
        /// Provenance ID for tracking
        provenance_id: u32,
    },
}

impl Value {
    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Dummy => "dummy",
            Value::Unit => "unit",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Tuple(_) => "tuple",
            Value::Struct { .. } => "struct",
            Value::Variant { .. } => "variant",
            Value::Function { .. } => "function",
            Value::Ref(_) => "ref",
            Value::ArrayRef { .. } => "array_ref",
            Value::RawPointer { .. } => "raw_pointer",
            Value::None => "None",
            Value::Some(_) => "Some",
            Value::Ok(_) => "Ok",
            Value::Err(_) => "Err",
            Value::Builtin(_) => "builtin",
            Value::ODESolution { .. } => "ODESolution",
            Value::Distribution(_) => "Distribution",
            Value::SymbolicExpr(_) => "SymbolicExpr",
            Value::Tensor { .. } => "Tensor",
            Value::SparseTensor(_) => "SparseTensor",
            Value::Uncertain { .. } => "Uncertain",
            Value::CausalModel(_) => "CausalModel",
            Value::HybridModel { .. } => "HybridModel",
            Value::Future { .. } => "Future",
            Value::Task { .. } => "Task",
            Value::Knowledge { .. } => "Knowledge",
        }
    }

    /// Deep clone: recursively copies arrays so struct copies don't alias.
    /// Regular `clone()` on `Value::Array` just bumps the Rc refcount,
    /// which means `var copy = original_struct` shares array storage.
    pub fn deep_clone(&self) -> Value {
        match self {
            Value::Array(arr) => {
                let borrowed = arr.borrow();
                Value::Array(Rc::new(RefCell::new(
                    borrowed.iter().map(|v| v.deep_clone()).collect(),
                )))
            }
            Value::Struct { name, fields } => Value::Struct {
                name: name.clone(),
                fields: fields.iter().map(|(k, v)| (k.clone(), v.deep_clone())).collect(),
            },
            Value::Ref(r) => {
                let inner = r.borrow();
                Value::Ref(Rc::new(RefCell::new(inner.deep_clone())))
            }
            Value::Tuple(elems) => {
                Value::Tuple(elems.iter().map(|v| v.deep_clone()).collect())
            }
            Value::Variant { enum_name, variant_name, fields } => Value::Variant {
                enum_name: enum_name.clone(),
                variant_name: variant_name.clone(),
                fields: fields.iter().map(|v| v.deep_clone()).collect(),
            },
            Value::Some(v) => Value::Some(Box::new(v.deep_clone())),
            Value::Ok(v) => Value::Ok(Box::new(v.deep_clone())),
            Value::Err(v) => Value::Err(Box::new(v.deep_clone())),
            // Scalars and other types — shallow clone is fine
            other => other.clone(),
        }
    }

    /// Check if value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Unit => false,
            Value::None => false,
            _ => true,
        }
    }

    /// Try to get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            _ => None,
        }
    }

    /// Try to get as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    /// Try to get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as ODE solution
    pub fn as_ode_solution(&self) -> Option<(&[f64], &[Vec<f64>], &SolverStats)> {
        match self {
            Value::ODESolution { t, y, stats } => Some((t, y, stats)),
            _ => None,
        }
    }

    /// Try to get as distribution
    pub fn as_distribution(&self) -> Option<&Distribution> {
        match self {
            Value::Distribution(d) => Some(d),
            _ => None,
        }
    }

    /// Try to get as symbolic expression
    pub fn as_symbolic_expr(&self) -> Option<Rc<SymbolicExprType>> {
        match self {
            Value::SymbolicExpr(e) => Some(e.clone()),
            _ => None,
        }
    }

    /// Try to get as hybrid model
    pub fn as_hybrid_model(
        &self,
    ) -> Option<(Rc<RefCell<Vec<f64>>>, Rc<SymbolicExprType>, HybridFusion)> {
        match self {
            Value::HybridModel {
                neural_params,
                symbolic_expr,
                fusion,
            } => Some((neural_params.clone(), symbolic_expr.clone(), fusion.clone())),
            _ => None,
        }
    }

    /// Try to get as tensor
    pub fn as_tensor(&self) -> Option<(&[f64], &[usize])> {
        match self {
            Value::Tensor { data, shape } => Some((data, shape)),
            _ => None,
        }
    }

    /// Try to get as sparse tensor
    pub fn as_sparse_tensor(&self) -> Option<Rc<crate::codegen::gpu::sparse::SparseTensor>> {
        match self {
            Value::SparseTensor(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Try to get as uncertain value
    pub fn as_uncertain(&self) -> Option<(f64, f64)> {
        match self {
            Value::Uncertain { mean, std } => Some((*mean, *std)),
            _ => None,
        }
    }

    /// Try to get as causal model
    pub fn as_causal_model(&self) -> Option<&str> {
        match self {
            Value::CausalModel(m) => Some(m),
            _ => None,
        }
    }

    /// Try to get as future
    pub fn as_future(&self) -> Option<&SounioFuture> {
        match self {
            Value::Future { future } => Some(future),
            _ => None,
        }
    }

    /// Try to get as task handle
    pub fn as_task(&self) -> Option<&TaskHandle> {
        match self {
            Value::Task { handle } => Some(handle),
            _ => None,
        }
    }

    /// Check if this value is a future
    pub fn is_future(&self) -> bool {
        matches!(self, Value::Future { .. })
    }

    /// Check if this value is a task
    pub fn is_task(&self) -> bool {
        matches!(self, Value::Task { .. })
    }

    /// Get the task ID if this is a future or task
    pub fn task_id(&self) -> Option<TaskId> {
        match self {
            Value::Future { future } => Some(future.task_id()),
            Value::Task { handle } => Some(handle.task_id()),
            _ => None,
        }
    }

    /// Pretty print complex mathematical structures (tensors, distributions, etc.)
    pub fn pretty_print(&self) -> String {
        match self {
            Value::Tensor { data, shape } => {
                if shape.is_empty() {
                    return format!("Scalar({})", data.first().unwrap_or(&0.0));
                }
                if shape.len() == 1 {
                    let max_disp = 10;
                    let display_len = data.len().min(max_disp);
                    let mut items: Vec<String> = data[..display_len].iter().map(|v| format!("{:.4}", v)).collect();
                    if data.len() > max_disp {
                        items.push("...".to_string());
                    }
                    return format!("Tensor[{}]: [{}]", shape[0], items.join(", "));
                }
                if shape.len() == 2 {
                    let rows = shape[0];
                    let cols = shape[1];
                    let max_rows = 5;
                    let max_cols = 5;
                    let disp_rows = rows.min(max_rows);
                    let disp_cols = cols.min(max_cols);
                    
                    let mut out = format!("Tensor[{}×{}]:\n", rows, cols);
                    out.push_str("┌ ");
                    out.push_str(&" ".repeat(disp_cols * 9));
                    out.push_str(" ┐\n");
                    
                    for r in 0..disp_rows {
                        out.push_str("│ ");
                        let mut row_items = Vec::new();
                        for c in 0..disp_cols {
                            row_items.push(format!("{:>8.4}", data[r * cols + c]));
                        }
                        out.push_str(&row_items.join(" "));
                        if cols > max_cols {
                            out.push_str(" ... ");
                        }
                        out.push_str(" │\n");
                    }
                    if rows > max_rows {
                        out.push_str("│ ");
                        out.push_str(&"   ...   ".repeat(disp_cols));
                        out.push_str(" │\n");
                    }
                    out.push_str("└ ");
                    out.push_str(&" ".repeat(disp_cols * 9));
                    out.push_str(" ┘");
                    return out;
                }
                format!("Tensor{:?} ({} elements)", shape, data.len())
            }
            Value::ODESolution { t, y, stats } => {
                let mut out = format!("ODESolution (steps: {})\n", stats.steps);
                let display_len = t.len().min(3);
                out.push_str("Time        Trajectories\n");
                for i in 0..display_len {
                    let t_val = t[i];
                    let vals: Vec<String> = y[i].iter().take(3).map(|v| format!("{:.4}", v)).collect();
                    let ellipsis = if y[i].len() > 3 { ", ..." } else { "" };
                    out.push_str(&format!("{:<10.4} [{}{}]\n", t_val, vals.join(", "), ellipsis));
                }
                if t.len() > 3 {
                    out.push_str("...\n");
                    let last = t.len() - 1;
                    let t_val = t[last];
                    let vals: Vec<String> = y[last].iter().take(3).map(|v| format!("{:.4}", v)).collect();
                    let ellipsis = if y[last].len() > 3 { ", ..." } else { "" };
                    out.push_str(&format!("{:<10.4} [{}{}]", t_val, vals.join(", "), ellipsis));
                }
                out
            }
            Value::Distribution(d) => match d {
                Distribution::Normal { mean, std } => format!("𝒩(μ={:.4}, σ={:.4})", mean, std),
                Distribution::Uniform { a, b } => format!("𝒰(a={:.4}, b={:.4})", a, b),
                Distribution::Beta { alpha, beta } => format!("Beta(α={:.4}, β={:.4})", alpha, beta),
                Distribution::Exponential { lambda } => format!("Exp(λ={:.4})", lambda),
                Distribution::Categorical { probs } => {
                    let max_disp = 5;
                    let display_len = probs.len().min(max_disp);
                    let mut items: Vec<String> = probs[..display_len].iter().map(|p| format!("{:.4}", p)).collect();
                    if probs.len() > max_disp {
                        items.push("...".to_string());
                    }
                    format!("Cat([{}])", items.join(", "))
                }
            },
            Value::CausalModel(m) => format!("CausalModel {{ {} }}", m),
            _ => format!("{}", self)
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Dummy => write!(f, "<dummy>"),
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{:?}", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                let arr = arr.borrow();
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, "]")
            }
            Value::Tuple(vals) => {
                write!(f, "(")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {:?}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::Variant {
                enum_name,
                variant_name,
                fields,
            } => {
                write!(f, "{}::{}", enum_name, variant_name)?;
                if !fields.is_empty() {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:?}", v)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::Function { func, .. } => write!(f, "<fn {}>", func.name),
            Value::Ref(r) => write!(f, "&{:?}", r.borrow()),
            Value::ArrayRef { array, index } => {
                let arr = array.borrow();
                if let Some(v) = arr.get(*index) {
                    write!(f, "&arr[{}]={:?}", index, v)
                } else {
                    write!(f, "&arr[{}]=<out of bounds>", index)
                }
            }
            Value::None => write!(f, "None"),
            Value::Some(v) => write!(f, "Some({:?})", v),
            Value::Ok(v) => write!(f, "Ok({:?})", v),
            Value::Err(v) => write!(f, "Err({:?})", v),
            Value::Builtin(name) => write!(f, "<builtin {}>", name),
            Value::ODESolution { t, y, stats } => {
                write!(
                    f,
                    "ODESolution {{ t: [{}], y: [{}], steps: {} }}",
                    t.len(),
                    y.len(),
                    stats.steps
                )
            }
            Value::Distribution(d) => write!(f, "{:?}", d),
            Value::SymbolicExpr(e) => write!(f, "SymbolicExpr({:?})", e),
            Value::Tensor { data, shape } => {
                write!(f, "Tensor {{ shape: {:?}, data: [{}] }}", shape, data.len())
            }
            Value::SparseTensor(s) => {
                let (r, c) = s.dims();
                let nnz = s.nnz();
                write!(f, "SparseTensor({}x{}, nnz={})", r, c, nnz)
            }
            Value::Uncertain { mean, std } => write!(f, "{} ± {}", mean, std),
            Value::CausalModel(m) => write!(f, "CausalModel({})", m),
            Value::HybridModel {
                neural_params,
                symbolic_expr,
                fusion,
            } => {
                write!(
                    f,
                    "HybridModel {{ neural_params: [{}], symbolic: {:?}, fusion: {:?} }}",
                    neural_params.borrow().len(),
                    symbolic_expr,
                    fusion
                )
            }
            Value::RawPointer { address, mutable } => {
                if *mutable {
                    write!(f, "*mut 0x{:x}", address)
                } else {
                    write!(f, "*const 0x{:x}", address)
                }
            }
            Value::Future { future } => {
                if future.is_completed() {
                    write!(f, "Future(completed)")
                } else {
                    write!(f, "Future(pending)")
                }
            }
            Value::Task { handle } => {
                if handle.is_completed() {
                    write!(f, "Task({:?}, completed)", handle.task_id())
                } else {
                    write!(f, "Task({:?}, pending)", handle.task_id())
                }
            }
            Value::Knowledge {
                value,
                confidence,
                knightian,
                provenance_id,
            } => {
                if *knightian {
                    write!(f, "Knowledge({:?}, ε=⊥, prov={})", value, provenance_id)
                } else {
                    write!(
                        f,
                        "Knowledge({:?}, conf={:.3}, prov={})",
                        value, confidence, provenance_id
                    )
                }
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Dummy => write!(f, "<dummy>"),
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                let arr = arr.borrow();
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Tuple(vals) => {
                write!(f, "(")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::Variant {
                variant_name,
                fields,
                ..
            } => {
                write!(f, "{}", variant_name)?;
                if !fields.is_empty() {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::Function { func, .. } => write!(f, "<fn {}>", func.name),
            Value::Ref(r) => write!(f, "{}", r.borrow()),
            Value::ArrayRef { array, index } => {
                let arr = array.borrow();
                if let Some(v) = arr.get(*index) {
                    write!(f, "{}", v)
                } else {
                    write!(f, "<out of bounds>")
                }
            }
            Value::None => write!(f, "None"),
            Value::Some(v) => write!(f, "Some({})", v),
            Value::Ok(v) => write!(f, "Ok({})", v),
            Value::Err(v) => write!(f, "Err({})", v),
            Value::Builtin(name) => write!(f, "<builtin {}>", name),
            Value::ODESolution { t, y, stats } => {
                write!(
                    f,
                    "ODESolution(t: {} points, y: {} trajectories, steps: {})",
                    t.len(),
                    y.len(),
                    stats.steps
                )
            }
            Value::Distribution(d) => match d {
                Distribution::Normal { mean, std } => write!(f, "Normal({}, {})", mean, std),
                Distribution::Uniform { a, b } => write!(f, "Uniform({}, {})", a, b),
                Distribution::Beta { alpha, beta } => write!(f, "Beta({}, {})", alpha, beta),
                Distribution::Exponential { lambda } => write!(f, "Exponential({})", lambda),
                Distribution::Categorical { probs } => write!(f, "Categorical([{}])", probs.len()),
            },
            Value::SymbolicExpr(e) => write!(f, "Symbolic({})", e),
            Value::Tensor { data, shape } => {
                write!(f, "Tensor({:?}, {} elements)", shape, data.len())
            }
            Value::SparseTensor(s) => {
                let (r, c) = s.dims();
                let nnz = s.nnz();
                write!(f, "SparseTensor({}x{}, nnz={})", r, c, nnz)
            }
            Value::Uncertain { mean, std } => write!(f, "{} ± {}", mean, std),
            Value::CausalModel(m) => write!(f, "CausalModel({})", m),
            Value::HybridModel {
                neural_params,
                symbolic_expr,
                fusion,
            } => {
                write!(
                    f,
                    "HybridModel(params: {}, expr: {}, fusion: {:?})",
                    neural_params.borrow().len(),
                    symbolic_expr,
                    fusion
                )
            }
            Value::RawPointer { address, mutable } => {
                if *mutable {
                    write!(f, "*mut 0x{:x}", address)
                } else {
                    write!(f, "*const 0x{:x}", address)
                }
            }
            Value::Future { future } => {
                if future.is_completed() {
                    write!(f, "Future(completed)")
                } else {
                    write!(f, "Future(pending)")
                }
            }
            Value::Task { handle } => {
                if handle.is_completed() {
                    write!(f, "Task(completed)")
                } else {
                    write!(f, "Task(pending)")
                }
            }
            Value::Knowledge {
                value, confidence, knightian, ..
            } => {
                if *knightian {
                    write!(f, "Knowledge({}, ε=⊥)", value)
                } else {
                    write!(f, "Knowledge({}, {:.3})", value, confidence)
                }
            }
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Dummy, Value::Dummy) => true,
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::None, Value::None) => true,
            (Value::Some(a), Value::Some(b)) => a == b,
            (Value::Ok(a), Value::Ok(b)) => a == b,
            (Value::Err(a), Value::Err(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => *a.borrow() == *b.borrow(),
            (
                Value::Struct {
                    name: n1,
                    fields: f1,
                },
                Value::Struct {
                    name: n2,
                    fields: f2,
                },
            ) => n1 == n2 && f1 == f2,
            (
                Value::Variant {
                    enum_name: e1,
                    variant_name: v1,
                    fields: f1,
                },
                Value::Variant {
                    enum_name: e2,
                    variant_name: v2,
                    fields: f2,
                },
            ) => e1 == e2 && v1 == v2 && f1 == f2,
            (Value::Builtin(a), Value::Builtin(b)) => a == b,
            (Value::Uncertain { mean: m1, std: s1 }, Value::Uncertain { mean: m2, std: s2 }) => {
                m1 == m2 && s1 == s2
            }
            (Value::SymbolicExpr(a), Value::SymbolicExpr(b)) => a == b,
            (Value::CausalModel(a), Value::CausalModel(b)) => a == b,
            _ => false,
        }
    }
}

/// Unique identifier for a suspended computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SuspensionId(pub u64);

impl SuspensionId {
    /// Create a new suspension ID from a raw value
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SuspensionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "suspension_{}", self.0)
    }
}

/// Control flow signal (not an error, just flow control)
#[derive(Debug, Clone)]
pub enum ControlFlow {
    /// Return from function with value
    Return(Value),
    /// Break from loop with optional value
    Break(Option<Value>),
    /// Continue to next iteration
    Continue,
    /// Suspend execution, waiting for handler to resume
    ///
    /// This is used when an effect handler returns `HandlerResult::Suspend`.
    /// The computation is suspended and the continuation is stored for later resumption.
    ///
    /// - `suspension_id`: Unique ID identifying this suspended computation
    /// - `continuation_id`: ID of the captured continuation in the continuation store
    Suspend {
        /// Unique ID for this suspension (used by handler to resume)
        suspension_id: SuspensionId,
        /// ID of the continuation in the store
        continuation_id: crate::effects::continuation::ContinuationId,
    },
}
