//! Compile-Time Function Execution (CTFE)
//!
//! Implements a subset of D execution at compile time for:
//! - Const evaluation
//! - Static assertions
//! - Type-level computation

use std::collections::HashMap;
use std::fmt;

use crate::common::Span;

/// A compile-time value
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Unit,
    Bool(bool),
    Int(i128),
    Uint(u128),
    Float(f64),
    String(String),
    Char(char),
    Array(Vec<ConstValue>),
    Tuple(Vec<ConstValue>),
    Struct {
        name: String,
        fields: HashMap<String, ConstValue>,
    },
    Enum {
        name: String,
        variant: String,
        value: Option<Box<ConstValue>>,
    },
    Function {
        name: String,
        module: String,
    },
    Type(String),
    Error(String),
}

impl ConstValue {
    pub fn is_truthy(&self) -> bool {
        match self {
            ConstValue::Bool(b) => *b,
            ConstValue::Int(i) => *i != 0,
            ConstValue::Uint(u) => *u != 0,
            _ => true,
        }
    }

    pub fn as_int(&self) -> Option<i128> {
        match self {
            ConstValue::Int(i) => Some(*i),
            ConstValue::Uint(u) => Some(*u as i128),
            _ => None,
        }
    }
}

impl fmt::Display for ConstValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstValue::Unit => write!(f, "()"),
            ConstValue::Bool(b) => write!(f, "{}", b),
            ConstValue::Int(i) => write!(f, "{}", i),
            ConstValue::Uint(u) => write!(f, "{}", u),
            ConstValue::Float(fl) => write!(f, "{}", fl),
            ConstValue::String(s) => write!(f, "\"{}\"", s),
            ConstValue::Char(c) => write!(f, "'{}'", c),
            ConstValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ConstValue::Tuple(tup) => {
                write!(f, "(")?;
                for (i, v) in tup.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
            ConstValue::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, " }}")
            }
            ConstValue::Enum {
                name,
                variant,
                value,
            } => {
                write!(f, "{}::{}", name, variant)?;
                if let Some(v) = value {
                    write!(f, "({})", v)?;
                }
                Ok(())
            }
            ConstValue::Function { name, module } => {
                write!(f, "{}::{}", module, name)
            }
            ConstValue::Type(ty) => write!(f, "type({})", ty),
            ConstValue::Error(e) => write!(f, "error({})", e),
        }
    }
}

/// CTFE evaluation context
pub struct CtfeContext {
    locals: Vec<HashMap<String, ConstValue>>,
    fuel: usize,
    max_fuel: usize,
    depth: usize,
    max_depth: usize,
}

/// CTFE error
#[derive(Debug, Clone)]
pub struct CtfeError {
    pub message: String,
    pub span: Option<Span>,
    pub backtrace: Vec<String>,
}

impl CtfeError {
    pub fn new(message: impl Into<String>) -> Self {
        CtfeError {
            message: message.into(),
            span: None,
            backtrace: Vec::new(),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

impl fmt::Display for CtfeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if !self.backtrace.is_empty() {
            write!(f, "\nbacktrace:")?;
            for frame in &self.backtrace {
                write!(f, "\n  {}", frame)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for CtfeError {}

impl CtfeContext {
    pub fn new() -> Self {
        CtfeContext {
            locals: vec![HashMap::new()],
            fuel: 1_000_000,
            max_fuel: 1_000_000,
            depth: 0,
            max_depth: 128,
        }
    }

    pub fn consume_fuel(&mut self) -> Result<(), CtfeError> {
        if self.fuel == 0 {
            return Err(CtfeError::new("const evaluation exceeded step limit"));
        }
        self.fuel -= 1;
        Ok(())
    }

    pub fn lookup_var(&self, name: &str) -> Result<ConstValue, CtfeError> {
        for scope in self.locals.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Ok(value.clone());
            }
        }
        Err(CtfeError::new(format!(
            "undefined variable in const context: {}",
            name
        )))
    }

    pub fn set_var(&mut self, name: String, value: ConstValue) {
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name, value);
        }
    }

    pub fn push_scope(&mut self) {
        self.locals.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.locals.len() > 1 {
            self.locals.pop();
        }
    }

    pub fn eval_binary_op(
        &self,
        op: &str,
        left: &ConstValue,
        right: &ConstValue,
    ) -> Result<ConstValue, CtfeError> {
        match (op, left, right) {
            // Integer arithmetic
            ("+", ConstValue::Int(a), ConstValue::Int(b)) => {
                Ok(ConstValue::Int(a.checked_add(*b).ok_or_else(|| {
                    CtfeError::new("integer overflow in const context")
                })?))
            }
            ("-", ConstValue::Int(a), ConstValue::Int(b)) => {
                Ok(ConstValue::Int(a.checked_sub(*b).ok_or_else(|| {
                    CtfeError::new("integer underflow in const context")
                })?))
            }
            ("*", ConstValue::Int(a), ConstValue::Int(b)) => {
                Ok(ConstValue::Int(a.checked_mul(*b).ok_or_else(|| {
                    CtfeError::new("integer overflow in const context")
                })?))
            }
            ("/", ConstValue::Int(a), ConstValue::Int(b)) => {
                if *b == 0 {
                    Err(CtfeError::new("division by zero in const context"))
                } else {
                    Ok(ConstValue::Int(a / b))
                }
            }
            ("%", ConstValue::Int(a), ConstValue::Int(b)) => {
                if *b == 0 {
                    Err(CtfeError::new("modulo by zero in const context"))
                } else {
                    Ok(ConstValue::Int(a % b))
                }
            }

            // Float arithmetic
            ("+", ConstValue::Float(a), ConstValue::Float(b)) => Ok(ConstValue::Float(a + b)),
            ("-", ConstValue::Float(a), ConstValue::Float(b)) => Ok(ConstValue::Float(a - b)),
            ("*", ConstValue::Float(a), ConstValue::Float(b)) => Ok(ConstValue::Float(a * b)),
            ("/", ConstValue::Float(a), ConstValue::Float(b)) => Ok(ConstValue::Float(a / b)),

            // Comparisons
            ("==", _, _) => Ok(ConstValue::Bool(left == right)),
            ("!=", _, _) => Ok(ConstValue::Bool(left != right)),
            ("<", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Bool(a < b)),
            ("<=", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Bool(a <= b)),
            (">", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Bool(a > b)),
            (">=", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Bool(a >= b)),

            // Boolean operations
            ("&&", ConstValue::Bool(a), ConstValue::Bool(b)) => Ok(ConstValue::Bool(*a && *b)),
            ("||", ConstValue::Bool(a), ConstValue::Bool(b)) => Ok(ConstValue::Bool(*a || *b)),

            // Bitwise operations
            ("&", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(a & b)),
            ("|", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(a | b)),
            ("^", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(a ^ b)),
            ("<<", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(a << b)),
            (">>", ConstValue::Int(a), ConstValue::Int(b)) => Ok(ConstValue::Int(a >> b)),

            // String concatenation
            ("+", ConstValue::String(a), ConstValue::String(b)) => {
                Ok(ConstValue::String(format!("{}{}", a, b)))
            }

            _ => Err(CtfeError::new(format!(
                "unsupported binary operation {} on {:?} and {:?}",
                op, left, right
            ))),
        }
    }

    pub fn eval_unary_op(&self, op: &str, operand: &ConstValue) -> Result<ConstValue, CtfeError> {
        match (op, operand) {
            ("-", ConstValue::Int(i)) => Ok(ConstValue::Int(-i)),
            ("-", ConstValue::Float(f)) => Ok(ConstValue::Float(-f)),
            ("!", ConstValue::Bool(b)) => Ok(ConstValue::Bool(!b)),
            ("~", ConstValue::Int(i)) => Ok(ConstValue::Int(!i)),
            _ => Err(CtfeError::new(format!(
                "unsupported unary operation {} on {:?}",
                op, operand
            ))),
        }
    }
}

impl Default for CtfeContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Static assertion
pub fn static_assert(condition: bool, message: &str) -> Result<(), CtfeError> {
    if condition {
        Ok(())
    } else {
        Err(CtfeError::new(format!(
            "static assertion failed: {}",
            message
        )))
    }
}
