//! Tree-walking interpreter for HIR

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use miette::{Result, miette};

use crate::hir::*;

use super::builtins::BuiltinRegistry;
use super::env::Environment;
use super::value::{ControlFlow, Value};

/// Tree-walking interpreter
pub struct Interpreter {
    /// Variable environment
    env: Environment,
    /// Function definitions (by name)
    functions: HashMap<String, Rc<HirFn>>,
    /// Struct definitions (by name)
    structs: HashMap<String, HirStruct>,
    /// Enum definitions (by name)
    enums: HashMap<String, HirEnum>,
    /// Builtin function registry
    builtins: BuiltinRegistry,
    /// Output buffer for testing
    output: Vec<String>,
}

impl Interpreter {
    /// Create a new interpreter
    pub fn new() -> Self {
        Interpreter {
            env: Environment::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            builtins: BuiltinRegistry::new(),
            output: Vec::new(),
        }
    }

    /// Get captured output (for testing)
    pub fn get_output(&self) -> &[String] {
        &self.output
    }

    /// Clear output buffer
    pub fn clear_output(&mut self) {
        self.output.clear();
    }

    /// Get mutable access to environment (for REPL)
    pub fn env_mut(&mut self) -> &mut Environment {
        &mut self.env
    }

    /// Evaluate a block (internal API for closures)
    pub(crate) fn eval_block_internal(&mut self, block: &HirBlock) -> Result<Value, ControlFlow> {
        self.eval_block(block)
    }

    /// Alias for interpret (for API compatibility)
    pub fn run(&mut self, hir: &Hir) -> Result<Value> {
        self.interpret(hir)
    }

    /// Interpret an HIR program
    pub fn interpret(&mut self, hir: &Hir) -> Result<Value> {
        // First pass: collect all definitions
        for item in &hir.items {
            match item {
                HirItem::Function(f) => {
                    self.functions.insert(f.name.clone(), Rc::new(f.clone()));
                }
                HirItem::Struct(s) => {
                    self.structs.insert(s.name.clone(), s.clone());
                }
                HirItem::Enum(e) => {
                    self.enums.insert(e.name.clone(), e.clone());
                }
                _ => {}
            }
        }

        // Look for main function
        if let Some(main_fn) = self.functions.get("main").cloned() {
            self.call_function(&main_fn, vec![])
        } else {
            // No main, evaluate last expression or return unit
            Ok(Value::Unit)
        }
    }

    /// Call a function with arguments
    fn call_function(&mut self, func: &HirFn, args: Vec<Value>) -> Result<Value> {
        self.env.push_scope();

        // Bind parameters
        for (param, arg) in func.ty.params.iter().zip(args.into_iter()) {
            self.env.define(param.name.clone(), arg);
        }

        // Execute body
        let result = self.eval_block(&func.body);

        self.env.pop_scope();

        match result {
            Ok(v) => Ok(v),
            Err(ControlFlow::Return(v)) => Ok(v),
            Err(ControlFlow::Break(_)) => Err(miette!("break outside loop")),
            Err(ControlFlow::Continue) => Err(miette!("continue outside loop")),
        }
    }

    /// Evaluate a block
    fn eval_block(&mut self, block: &HirBlock) -> Result<Value, ControlFlow> {
        self.env.push_scope();

        let result = self.eval_block_inner(block);

        // Always pop scope, even on early return/break/continue
        self.env.pop_scope();

        result
    }

    /// Inner block evaluation (without scope management)
    fn eval_block_inner(&mut self, block: &HirBlock) -> Result<Value, ControlFlow> {
        let mut result = Value::Unit;

        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == block.stmts.len() - 1;

            match stmt {
                HirStmt::Let {
                    name,
                    value,
                    is_mut,
                    ..
                } => {
                    let val = if let Some(expr) = value {
                        self.eval_expr(expr)?
                    } else {
                        Value::Unit
                    };
                    // Wrap mutable structs in Ref to allow field mutation
                    let val = if *is_mut && matches!(val, Value::Struct { .. }) {
                        Value::Ref(Rc::new(RefCell::new(val)))
                    } else {
                        val
                    };
                    self.env.define(name.clone(), val);
                }
                HirStmt::Expr(expr) => {
                    result = self.eval_expr(expr)?;
                }
                HirStmt::Assign { target, value } => {
                    let val = self.eval_expr(value)?;
                    self.assign_target(target, val)?;
                }
            }
        }

        Ok(result)
    }

    /// Evaluate an expression
    fn eval_expr(&mut self, expr: &HirExpr) -> Result<Value, ControlFlow> {
        match &expr.kind {
            HirExprKind::Literal(lit) => Ok(self.eval_literal(lit)),

            HirExprKind::Local(name) => {
                // First check local variables
                if let Some(val) = self.env.get(name) {
                    return Ok(val);
                }
                // Then check if it's a function name
                if let Some(func) = self.functions.get(name).cloned() {
                    return Ok(Value::Function {
                        func,
                        captures: HashMap::new(),
                    });
                }
                // Check if it's a builtin function
                if self.is_builtin(name) {
                    return Ok(Value::Builtin(name.clone()));
                }
                // Not found
                Err(ControlFlow::Return(Value::Unit))
            }

            HirExprKind::Global(name) => {
                // Check if it's a function
                if let Some(func) = self.functions.get(name).cloned() {
                    Ok(Value::Function {
                        func,
                        captures: HashMap::new(),
                    })
                } else if self.is_builtin(name) {
                    // Check if it's a builtin function
                    Ok(Value::Builtin(name.clone()))
                } else {
                    self.env.get(name).ok_or(ControlFlow::Return(Value::Unit))
                }
            }

            HirExprKind::Binary { op, left, right } => {
                let lhs = self.eval_expr(left)?;

                // Short-circuit for And/Or
                match op {
                    HirBinaryOp::And => {
                        if !lhs.is_truthy() {
                            return Ok(Value::Bool(false));
                        }
                        let rhs = self.eval_expr(right)?;
                        return Ok(Value::Bool(rhs.is_truthy()));
                    }
                    HirBinaryOp::Or => {
                        if lhs.is_truthy() {
                            return Ok(Value::Bool(true));
                        }
                        let rhs = self.eval_expr(right)?;
                        return Ok(Value::Bool(rhs.is_truthy()));
                    }
                    _ => {}
                }

                let rhs = self.eval_expr(right)?;
                self.eval_binary(*op, lhs, rhs)
            }

            HirExprKind::Unary { op, expr: inner } => {
                // Special case: Ref/RefMut of array index creates ArrayRef
                if matches!(op, HirUnaryOp::Ref | HirUnaryOp::RefMut) {
                    if let HirExprKind::Index { base, index } = &inner.kind {
                        let base_val = self.eval_expr(base)?;
                        let idx_val = self.eval_expr(index)?;
                        let idx =
                            idx_val.as_int().ok_or(ControlFlow::Return(Value::Unit))? as usize;

                        if let Value::Array(arr) = base_val {
                            return Ok(Value::ArrayRef {
                                array: arr,
                                index: idx,
                            });
                        }
                    }
                }
                let val = self.eval_expr(inner)?;
                self.eval_unary(*op, val)
            }

            HirExprKind::Call { func, args } => {
                let callee = self.eval_expr(func)?;
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.eval_expr(arg)?);
                }
                self.eval_call(callee, arg_values)
            }

            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond = self.eval_expr(condition)?;
                if cond.is_truthy() {
                    self.eval_block(then_branch)
                } else if let Some(else_expr) = else_branch {
                    self.eval_expr(else_expr)
                } else {
                    Ok(Value::Unit)
                }
            }

            HirExprKind::Block(block) => self.eval_block(block),

            HirExprKind::Loop(block) => loop {
                match self.eval_block(block) {
                    Ok(_) => continue,
                    Err(ControlFlow::Continue) => continue,
                    Err(ControlFlow::Break(val)) => {
                        return Ok(val.unwrap_or(Value::Unit));
                    }
                    Err(ControlFlow::Return(v)) => {
                        return Err(ControlFlow::Return(v));
                    }
                }
            },

            // While loop - condition re-evaluated FRESH on each iteration
            HirExprKind::While { condition, body } => loop {
                // Evaluate condition fresh each iteration (fixes the bug!)
                let cond_val = self.eval_expr(condition)?;
                if !cond_val.is_truthy() {
                    break Ok(Value::Unit);
                }

                // Execute body
                match self.eval_block(body) {
                    Ok(_) => continue,
                    Err(ControlFlow::Continue) => continue,
                    Err(ControlFlow::Break(val)) => {
                        return Ok(val.unwrap_or(Value::Unit));
                    }
                    Err(ControlFlow::Return(v)) => {
                        return Err(ControlFlow::Return(v));
                    }
                }
            },

            HirExprKind::Return(value) => {
                let val = if let Some(expr) = value {
                    self.eval_expr(expr)?
                } else {
                    Value::Unit
                };
                Err(ControlFlow::Return(val))
            }

            HirExprKind::Break(value) => {
                let val = if let Some(expr) = value {
                    Some(self.eval_expr(expr)?)
                } else {
                    None
                };
                Err(ControlFlow::Break(val))
            }

            HirExprKind::Continue => Err(ControlFlow::Continue),

            HirExprKind::Tuple(elements) => {
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_expr(elem)?);
                }
                Ok(Value::Tuple(values))
            }

            HirExprKind::Array(elements) => {
                let mut values = Vec::new();
                for elem in elements {
                    values.push(self.eval_expr(elem)?);
                }
                Ok(Value::Array(Rc::new(RefCell::new(values))))
            }

            HirExprKind::Range {
                start,
                end,
                inclusive,
            } => {
                let start_val = match start {
                    Some(s) => self.eval_expr(s)?,
                    None => Value::Int(0),
                };
                let end_val = match end {
                    Some(e) => self.eval_expr(e)?,
                    None => Value::Int(i64::MAX),
                };
                // Return as a struct-like value
                let mut fields = HashMap::new();
                fields.insert("start".to_string(), start_val);
                fields.insert("end".to_string(), end_val);
                fields.insert("inclusive".to_string(), Value::Bool(*inclusive));
                Ok(Value::Struct {
                    name: "Range".to_string(),
                    fields,
                })
            }

            HirExprKind::Struct { name, fields } => {
                let mut field_values = HashMap::new();
                for (field_name, field_expr) in fields {
                    field_values.insert(field_name.clone(), self.eval_expr(field_expr)?);
                }
                Ok(Value::Struct {
                    name: name.clone(),
                    fields: field_values,
                })
            }

            HirExprKind::Variant {
                enum_name,
                variant,
                fields,
            } => {
                let mut field_values = Vec::new();
                for field_expr in fields {
                    field_values.push(self.eval_expr(field_expr)?);
                }
                Ok(Value::Variant {
                    enum_name: enum_name.clone(),
                    variant_name: variant.clone(),
                    fields: field_values,
                })
            }

            HirExprKind::Field { base, field } => {
                let base_val = self.eval_expr(base)?;
                match base_val {
                    Value::Struct { fields, .. } => fields
                        .get(field)
                        .cloned()
                        .ok_or(ControlFlow::Return(Value::Unit)),
                    Value::Ref(r) => {
                        let inner = r.borrow();
                        if let Value::Struct { ref fields, .. } = *inner {
                            fields
                                .get(field)
                                .cloned()
                                .ok_or(ControlFlow::Return(Value::Unit))
                        } else {
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
                    _ => Err(ControlFlow::Return(Value::Unit)),
                }
            }

            HirExprKind::TupleField { base, index } => {
                let base_val = self.eval_expr(base)?;
                match base_val {
                    Value::Tuple(elements) => elements
                        .get(*index)
                        .cloned()
                        .ok_or(ControlFlow::Return(Value::Unit)),
                    _ => Err(ControlFlow::Return(Value::Unit)),
                }
            }

            HirExprKind::Index { base, index } => {
                let base_val = self.eval_expr(base)?;
                let idx_val = self.eval_expr(index)?;

                let idx = idx_val.as_int().ok_or(ControlFlow::Return(Value::Unit))? as usize;

                match base_val {
                    Value::Array(arr) => {
                        let arr = arr.borrow();
                        arr.get(idx)
                            .cloned()
                            .ok_or(ControlFlow::Return(Value::Unit))
                    }
                    Value::String(s) => s
                        .chars()
                        .nth(idx)
                        .map(|c| Value::String(c.to_string()))
                        .ok_or(ControlFlow::Return(Value::Unit)),
                    // ArrayRef: reference to array element - index into the inner array
                    Value::ArrayRef {
                        array,
                        index: arr_idx,
                    } => {
                        let arr = array.borrow();
                        if let Some(inner) = arr.get(arr_idx) {
                            match inner {
                                Value::Array(inner_arr) => {
                                    let inner_arr = inner_arr.borrow();
                                    inner_arr
                                        .get(idx)
                                        .cloned()
                                        .ok_or(ControlFlow::Return(Value::Unit))
                                }
                                Value::String(s) => s
                                    .chars()
                                    .nth(idx)
                                    .map(|c| Value::String(c.to_string()))
                                    .ok_or(ControlFlow::Return(Value::Unit)),
                                _ => Err(ControlFlow::Return(Value::Unit)),
                            }
                        } else {
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
                    _ => Err(ControlFlow::Return(Value::Unit)),
                }
            }

            HirExprKind::Ref {
                mutable: _,
                expr: inner,
            } => {
                // Special case: reference to array element creates ArrayRef
                if let HirExprKind::Index { base, index } = &inner.kind {
                    let base_val = self.eval_expr(base)?;
                    let idx_val = self.eval_expr(index)?;
                    let idx = idx_val.as_int().ok_or(ControlFlow::Return(Value::Unit))? as usize;

                    if let Value::Array(arr) = base_val {
                        return Ok(Value::ArrayRef {
                            array: arr,
                            index: idx,
                        });
                    }
                }
                // Default: evaluate and wrap in Ref
                let val = self.eval_expr(inner)?;
                Ok(Value::Ref(Rc::new(RefCell::new(val))))
            }

            HirExprKind::Deref(inner) => {
                let val = self.eval_expr(inner)?;
                match val {
                    Value::Ref(r) => Ok(r.borrow().clone()),
                    Value::ArrayRef { array, index } => {
                        let arr = array.borrow();
                        arr.get(index)
                            .cloned()
                            .ok_or(ControlFlow::Return(Value::Unit))
                    }
                    _ => Err(ControlFlow::Return(Value::Unit)),
                }
            }

            HirExprKind::Match { scrutinee, arms } => {
                let val = self.eval_expr(scrutinee)?;

                for arm in arms {
                    if let Some(bindings) = self.match_pattern(&arm.pattern, &val) {
                        // Check guard if present
                        if let Some(guard) = &arm.guard {
                            self.env.push_scope();
                            for (name, value) in &bindings {
                                self.env.define(name.clone(), value.clone());
                            }
                            let guard_result = self.eval_expr(guard)?;
                            self.env.pop_scope();

                            if !guard_result.is_truthy() {
                                continue;
                            }
                        }

                        // Execute arm body with bindings
                        self.env.push_scope();
                        for (name, value) in bindings {
                            self.env.define(name, value);
                        }
                        let result = self.eval_expr(&arm.body);
                        self.env.pop_scope();
                        return result;
                    }
                }

                // No match found - this should be a runtime error
                Err(ControlFlow::Return(Value::Unit))
            }

            HirExprKind::Cast { expr: inner, target } => {
                let val = self.eval_expr(inner)?;
                self.cast_value(val, target)
            }

            HirExprKind::Closure { params, body } => {
                // Capture current environment
                let captures = self.env.capture_all();

                // Create a synthetic HirFn for the closure
                let closure_fn = HirFn {
                    id: crate::common::NodeId::dummy(),
                    name: "<closure>".to_string(),
                    ty: HirFnType {
                        params: params.clone(),
                        return_type: Box::new(body.ty.clone()),
                        effects: Vec::new(),
                    },
                    body: HirBlock {
                        stmts: vec![HirStmt::Expr(body.as_ref().clone())],
                        ty: body.ty.clone(),
                    },
                    abi: crate::ast::Abi::Rust,
                    is_exported: false,
                };

                Ok(Value::Function {
                    func: Rc::new(closure_fn),
                    captures,
                })
            }

            HirExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let recv = self.eval_expr(receiver)?;
                let mut arg_values = vec![recv.clone()];
                for arg in args {
                    arg_values.push(self.eval_expr(arg)?);
                }

                // Handle built-in methods
                match (recv, method.as_str()) {
                    (Value::Array(arr), "len") => Ok(Value::Int(arr.borrow().len() as i64)),
                    (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
                    (Value::String(s), "slice") => {
                        // String slice: s.slice(start, end) -> substring
                        let start = match arg_values.get(1) {
                            Some(Value::Int(i)) => (*i).max(0) as usize,
                            _ => 0,
                        };
                        let end = match arg_values.get(2) {
                            Some(Value::Int(i)) => (*i).max(0) as usize,
                            _ => s.len(),
                        };
                        // Handle UTF-8: use char_indices for safe slicing
                        let chars: Vec<char> = s.chars().collect();
                        let start = start.min(chars.len());
                        let end = end.min(chars.len()).max(start);
                        let result: String = chars[start..end].iter().collect();
                        Ok(Value::String(result))
                    }
                    (Value::String(s), "byte_at") => {
                        // Get byte at index as integer
                        let idx = match arg_values.get(1) {
                            Some(Value::Int(i)) => (*i).max(0) as usize,
                            _ => 0,
                        };
                        if idx < s.len() {
                            Ok(Value::Int(s.as_bytes()[idx] as i64))
                        } else {
                            Ok(Value::Int(0))
                        }
                    }
                    (Value::Array(arr), "push") => {
                        if let Some(val) = arg_values.get(1) {
                            arr.borrow_mut().push(val.clone());
                        }
                        Ok(Value::Unit)
                    }
                    (Value::Array(arr), "pop") => Ok(arr.borrow_mut().pop().unwrap_or(Value::None)),
                    // ArrayRef: reference to array element - delegate to the inner array
                    (Value::ArrayRef { array, index }, method_name) => {
                        let arr = array.borrow();
                        if let Some(inner) = arr.get(index) {
                            match (inner, method_name) {
                                (Value::Array(inner_arr), "len") => {
                                    Ok(Value::Int(inner_arr.borrow().len() as i64))
                                }
                                (Value::Array(inner_arr), "push") => {
                                    drop(arr); // Release borrow
                                    let arr = array.borrow();
                                    if let Some(Value::Array(inner_arr)) = arr.get(index) {
                                        if let Some(val) = arg_values.get(1) {
                                            inner_arr.borrow_mut().push(val.clone());
                                        }
                                    }
                                    Ok(Value::Unit)
                                }
                                (Value::Array(inner_arr), "pop") => {
                                    Ok(inner_arr.borrow_mut().pop().unwrap_or(Value::None))
                                }
                                (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
                                _ => Err(ControlFlow::Return(Value::Unit)),
                            }
                        } else {
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
                    _ => {
                        // Try to find a function with method name
                        if let Some(func) = self.functions.get(method).cloned() {
                            self.call_function(&func, arg_values)
                                .map_err(|e| ControlFlow::Return(Value::Unit))
                        } else {
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
                }
            }

            // Effect operations - not fully implemented
            HirExprKind::Perform { .. } | HirExprKind::Handle { .. } | HirExprKind::Sample(_) => {
                Ok(Value::Unit)
            }

            // ==================== EPISTEMIC EXPRESSIONS ====================
            HirExprKind::Knowledge {
                value,
                epsilon,
                validity,
                provenance,
            } => {
                // For interpreter, evaluate the inner value and wrap with epistemic metadata
                let val = self.eval_expr(value)?;
                let eps = self.eval_expr(epsilon)?;
                // Return the value for now - a full implementation would return a Knowledge struct
                Ok(val)
            }

            HirExprKind::Do { variable, value } => {
                // Do intervention - in interpreter, just evaluate the value
                // A full implementation would track causal graph modifications
                let val = self.eval_expr(value)?;
                Ok(val)
            }

            HirExprKind::Counterfactual {
                factual,
                intervention,
                outcome,
            } => {
                // Counterfactual reasoning requires:
                // 1. Abduction: infer latent variables from factual
                // 2. Action: apply intervention
                // 3. Prediction: compute outcome under modified model
                // For now, just evaluate the outcome expression
                self.eval_expr(outcome)
            }

            HirExprKind::Query {
                target,
                given,
                interventions,
            } => {
                // Probabilistic query - P(target | given, do(interventions))
                // For interpreter, just evaluate the target
                // A full implementation would compute the probability
                self.eval_expr(target)
            }

            HirExprKind::Observe { variable, value } => {
                // Observe for probabilistic programming
                // Just evaluate the value for now
                self.eval_expr(value)
            }

            HirExprKind::EpsilonOf(expr) => {
                // Extract epsilon - return 1.0 (perfect confidence) for now
                Ok(Value::Float(1.0))
            }

            HirExprKind::ProvenanceOf(expr) => {
                // Extract provenance - return unit for now
                Ok(Value::Unit)
            }

            HirExprKind::ValidityOf(expr) => {
                // Extract validity - return unit for now
                Ok(Value::Unit)
            }

            HirExprKind::Unwrap(expr) => {
                // Unwrap Knowledge to get inner value
                self.eval_expr(expr)
            }

            HirExprKind::OntologyTerm { namespace, term } => {
                // Ontology terms are represented as strings at runtime
                Ok(Value::String(format!("{}:{}", namespace, term)))
            }
        }
    }

    /// Evaluate a literal
    fn eval_literal(&self, lit: &HirLiteral) -> Value {
        match lit {
            HirLiteral::Unit => Value::Unit,
            HirLiteral::Bool(b) => Value::Bool(*b),
            HirLiteral::Int(n) => Value::Int(*n),
            HirLiteral::Float(f) => Value::Float(*f),
            HirLiteral::Char(c) => Value::String(c.to_string()),
            HirLiteral::String(s) => Value::String(Self::process_escape_sequences(s)),
        }
    }

    /// Process escape sequences in a string literal
    fn process_escape_sequences(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('0') => result.push('\0'),
                    Some('"') => result.push('"'),
                    Some('\'') => result.push('\''),
                    Some(other) => {
                        // Unknown escape, keep as-is
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Cast a value to a target type
    fn cast_value(&self, val: Value, target: &HirType) -> Result<Value, ControlFlow> {
        match (val, target) {
            // Integer to integer casts (with proper truncation/sign extension)
            (Value::Int(n), HirType::I8) => Ok(Value::Int((n as i8) as i64)),
            (Value::Int(n), HirType::I16) => Ok(Value::Int((n as i16) as i64)),
            (Value::Int(n), HirType::I32) => Ok(Value::Int((n as i32) as i64)),
            (Value::Int(n), HirType::I64) => Ok(Value::Int(n)),
            (Value::Int(n), HirType::I128) => Ok(Value::Int(n)), // Can't represent i128 fully
            (Value::Int(n), HirType::Isize) => Ok(Value::Int((n as isize) as i64)),
            (Value::Int(n), HirType::U8) => Ok(Value::Int((n as u8) as i64)),
            (Value::Int(n), HirType::U16) => Ok(Value::Int((n as u16) as i64)),
            (Value::Int(n), HirType::U32) => Ok(Value::Int((n as u32) as i64)),
            (Value::Int(n), HirType::U64) => Ok(Value::Int(n as i64)),
            (Value::Int(n), HirType::U128) => Ok(Value::Int(n)), // Can't represent u128 fully
            (Value::Int(n), HirType::Usize) => Ok(Value::Int((n as usize) as i64)),

            // Float to integer casts
            (Value::Float(f), HirType::I8) => Ok(Value::Int((f as i8) as i64)),
            (Value::Float(f), HirType::I16) => Ok(Value::Int((f as i16) as i64)),
            (Value::Float(f), HirType::I32) => Ok(Value::Int((f as i32) as i64)),
            (Value::Float(f), HirType::I64) => Ok(Value::Int(f as i64)),
            (Value::Float(f), HirType::Isize) => Ok(Value::Int((f as isize) as i64)),
            (Value::Float(f), HirType::U8) => Ok(Value::Int((f as u8) as i64)),
            (Value::Float(f), HirType::U16) => Ok(Value::Int((f as u16) as i64)),
            (Value::Float(f), HirType::U32) => Ok(Value::Int((f as u32) as i64)),
            (Value::Float(f), HirType::U64) => Ok(Value::Int(f as i64)),
            (Value::Float(f), HirType::Usize) => Ok(Value::Int((f as usize) as i64)),

            // Integer to float casts
            (Value::Int(n), HirType::F32) => Ok(Value::Float((n as f32) as f64)),
            (Value::Int(n), HirType::F64) => Ok(Value::Float(n as f64)),

            // Float to float casts
            (Value::Float(f), HirType::F32) => Ok(Value::Float((f as f32) as f64)),
            (Value::Float(f), HirType::F64) => Ok(Value::Float(f)),

            // Bool conversions
            (Value::Bool(b), HirType::I8 | HirType::I16 | HirType::I32 | HirType::I64 | HirType::Isize) => {
                Ok(Value::Int(if b { 1 } else { 0 }))
            }
            (Value::Bool(b), HirType::U8 | HirType::U16 | HirType::U32 | HirType::U64 | HirType::Usize) => {
                Ok(Value::Int(if b { 1 } else { 0 }))
            }

            // Int/Float/Bool to String casts
            (Value::Int(n), HirType::String) => Ok(Value::String(n.to_string())),
            (Value::Float(f), HirType::String) => Ok(Value::String(format!("{}", f))),
            (Value::Bool(b), HirType::String) => Ok(Value::String(if b { "true" } else { "false" }.to_string())),

            // Identity cast - value stays the same
            (v, _) => Ok(v),
        }
    }

    /// Evaluate a binary operation
    fn eval_binary(&self, op: HirBinaryOp, lhs: Value, rhs: Value) -> Result<Value, ControlFlow> {
        match op {
            HirBinaryOp::Add => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Sub => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 - b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - b as f64)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Mul => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 * b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * b as f64)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Div => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => {
                    if b == 0 {
                        Err(ControlFlow::Return(Value::Unit)) // Division by zero
                    } else {
                        Ok(Value::Int(a / b))
                    }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 / b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / b as f64)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Rem => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => {
                    if b == 0 {
                        Err(ControlFlow::Return(Value::Unit))
                    } else {
                        Ok(Value::Int(a % b))
                    }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Eq => Ok(Value::Bool(lhs == rhs)),
            HirBinaryOp::Ne => Ok(Value::Bool(lhs != rhs)),
            HirBinaryOp::Lt => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a < b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Le => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a <= b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Gt => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a > b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Ge => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a >= b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::And => Ok(Value::Bool(lhs.is_truthy() && rhs.is_truthy())),
            HirBinaryOp::Or => Ok(Value::Bool(lhs.is_truthy() || rhs.is_truthy())),
            HirBinaryOp::BitAnd => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a & b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::BitOr => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a | b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::BitXor => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a ^ b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Shl => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a << b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Shr => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a >> b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::PlusMinus => match (lhs, rhs) {
                // Treat PlusMinus as Add (uncertainty is handled at type-check time)
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Concat => match (lhs, rhs) {
                // Concatenate arrays
                (Value::Array(a), Value::Array(b)) => {
                    let mut result = a.borrow().clone();
                    result.extend(b.borrow().iter().cloned());
                    Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(
                        result,
                    ))))
                }
                // Concatenate strings
                (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
        }
    }

    /// Evaluate a unary operation
    fn eval_unary(&self, op: HirUnaryOp, val: Value) -> Result<Value, ControlFlow> {
        match op {
            HirUnaryOp::Neg => match val {
                Value::Int(n) => Ok(Value::Int(-n)),
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirUnaryOp::Not => match val {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                Value::Int(n) => Ok(Value::Int(!n)),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirUnaryOp::Ref | HirUnaryOp::RefMut => Ok(Value::Ref(Rc::new(RefCell::new(val)))),
            HirUnaryOp::Deref => match val {
                Value::Ref(r) => Ok(r.borrow().clone()),
                Value::ArrayRef { array, index } => {
                    let arr = array.borrow();
                    arr.get(index)
                        .cloned()
                        .ok_or(ControlFlow::Return(Value::Unit))
                }
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
        }
    }

    /// Evaluate a function call
    fn eval_call(&mut self, callee: Value, args: Vec<Value>) -> Result<Value, ControlFlow> {
        match callee {
            Value::Function { func, captures } => {
                // Set up environment with captures
                self.env.push_scope();
                for (name, value) in captures {
                    self.env.define(name, value);
                }

                // Bind parameters
                for (param, arg) in func.ty.params.iter().zip(args.into_iter()) {
                    self.env.define(param.name.clone(), arg);
                }

                // Execute body
                let result = self.eval_block(&func.body);

                self.env.pop_scope();

                match result {
                    Ok(v) => Ok(v),
                    Err(ControlFlow::Return(v)) => Ok(v),
                    Err(cf) => Err(cf),
                }
            }
            Value::Builtin(name) => {
                // Call the builtin function by name
                self.call_builtin(&name, args)
            }
            _ => {
                // Check if it's a builtin by looking at the callee name
                // For now, handle common cases
                self.call_builtin_by_args(&args)
            }
        }
    }

    /// Check if a name is a builtin function
    fn is_builtin(&self, name: &str) -> bool {
        matches!(
            name,
            "print"
                | "println"
                | "assert"
                | "assert_eq"
                | "len"
                | "type_of"
                | "Some"
                | "None"
                | "Ok"
                | "Err"
                | "dbg"
                | "panic"
                | "format"
                | "read_line"
                | "parse_int"
                | "parse_float"
                | "to_string"
                | "sqrt"
                | "abs"
                | "sin"
                | "cos"
                | "tan"
                | "exp"
                | "log"
                | "pow"
                | "floor"
                | "ceil"
                | "round"
                | "min"
                | "max"
                // FFI / Raw pointer operations
                | "null_ptr"
                | "null_mut"
                | "is_null"
                | "ptr_eq"
                | "ptr_addr"
                | "ptr_from_addr"
                | "ptr_from_addr_mut"
                | "ptr_offset"
                | "ptr_add"
                | "ptr_sub"
                | "ptr_diff"
                | "as_const"
                | "as_mut"
                | "size_of"
                | "align_of"
        )
    }

    /// Try calling a builtin function by examining arguments
    fn call_builtin_by_args(&mut self, args: &[Value]) -> Result<Value, ControlFlow> {
        // Default: return unit
        Ok(Value::Unit)
    }

    /// Call a named builtin function
    pub fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, ControlFlow> {
        // First, try the builtin registry
        if self.builtins.is_builtin(name) {
            match self.builtins.call(name, &args) {
                Ok(v) => return Ok(v),
                Err(e) => return Err(ControlFlow::Return(Value::String(e))),
            }
        }

        // Fall back to hardcoded implementations
        match name {
            "print" => {
                let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
                let line = output.join(" ");
                print!("{}", line);
                self.output.push(line);
                Ok(Value::Unit)
            }
            "println" => {
                let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
                let line = output.join(" ");
                println!("{}", line);
                self.output.push(line);
                Ok(Value::Unit)
            }
            "assert" => {
                if let Some(val) = args.first()
                    && !val.is_truthy()
                {
                    return Err(ControlFlow::Return(Value::Unit)); // Assertion failed
                }
                Ok(Value::Unit)
            }
            "assert_eq" => {
                if args.len() >= 2 && args[0] != args[1] {
                    return Err(ControlFlow::Return(Value::Unit)); // Assertion failed
                }
                Ok(Value::Unit)
            }
            "len" => {
                if let Some(val) = args.first() {
                    match val {
                        Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                        Value::String(s) => Ok(Value::Int(s.len() as i64)),
                        Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
                        _ => Ok(Value::Int(0)),
                    }
                } else {
                    Ok(Value::Int(0))
                }
            }
            "type_of" => {
                if let Some(val) = args.first() {
                    Ok(Value::String(val.type_name().to_string()))
                } else {
                    Ok(Value::String("unknown".to_string()))
                }
            }
            "Some" => {
                if let Some(val) = args.into_iter().next() {
                    Ok(Value::Some(Box::new(val)))
                } else {
                    Ok(Value::Some(Box::new(Value::Unit)))
                }
            }
            "None" => Ok(Value::None),
            "Ok" => {
                if let Some(val) = args.into_iter().next() {
                    Ok(Value::Ok(Box::new(val)))
                } else {
                    Ok(Value::Ok(Box::new(Value::Unit)))
                }
            }
            "Err" => {
                if let Some(val) = args.into_iter().next() {
                    Ok(Value::Err(Box::new(val)))
                } else {
                    Ok(Value::Err(Box::new(Value::Unit)))
                }
            }
            "dbg" => {
                // Debug print - shows value with type info
                for arg in &args {
                    let line = format!("[dbg] {:?}", arg);
                    eprintln!("{}", line);
                    self.output.push(line);
                }
                Ok(args.into_iter().next().unwrap_or(Value::Unit))
            }
            "panic" => {
                let msg = args
                    .first()
                    .map(|v| format!("{}", v))
                    .unwrap_or_else(|| "panic!".to_string());
                eprintln!("panic: {}", msg);
                Err(ControlFlow::Return(Value::Unit))
            }
            "format" => {
                let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
                Ok(Value::String(output.join("")))
            }
            "to_string" => {
                let s = args.first().map(|v| format!("{}", v)).unwrap_or_default();
                Ok(Value::String(s))
            }
            "parse_int" => {
                if let Some(Value::String(s)) = args.first() {
                    match s.trim().parse::<i64>() {
                        Ok(n) => Ok(Value::Some(Box::new(Value::Int(n)))),
                        Err(_) => Ok(Value::None),
                    }
                } else {
                    Ok(Value::None)
                }
            }
            "parse_float" => {
                if let Some(Value::String(s)) = args.first() {
                    match s.trim().parse::<f64>() {
                        Ok(f) => Ok(Value::Some(Box::new(Value::Float(f)))),
                        Err(_) => Ok(Value::None),
                    }
                } else {
                    Ok(Value::None)
                }
            }
            "read_line" => {
                use std::io::{self, BufRead};
                let mut line = String::new();
                if io::stdin().lock().read_line(&mut line).is_ok() {
                    // Remove trailing newline
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    Ok(Value::String(line))
                } else {
                    Ok(Value::String(String::new()))
                }
            }
            // Math functions
            "sqrt" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                Ok(Value::Float(x.sqrt()))
            }
            "abs" => match args.first() {
                Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
                Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
                _ => Ok(Value::Float(0.0)),
            },
            "sin" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                Ok(Value::Float(x.sin()))
            }
            "cos" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                Ok(Value::Float(x.cos()))
            }
            "tan" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                Ok(Value::Float(x.tan()))
            }
            "exp" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                Ok(Value::Float(x.exp()))
            }
            "log" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(1.0);
                Ok(Value::Float(x.ln()))
            }
            "pow" => {
                let base = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                let exp = args.get(1).and_then(|v| v.as_float()).unwrap_or(1.0);
                Ok(Value::Float(base.powf(exp)))
            }
            "floor" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                Ok(Value::Float(x.floor()))
            }
            "ceil" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                Ok(Value::Float(x.ceil()))
            }
            "round" => {
                let x = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                Ok(Value::Float(x.round()))
            }
            "min" => match (args.first(), args.get(1)) {
                (Some(Value::Int(a)), Some(Value::Int(b))) => Ok(Value::Int(*a.min(b))),
                (Some(Value::Float(a)), Some(Value::Float(b))) => Ok(Value::Float(a.min(*b))),
                _ => Ok(Value::Float(0.0)),
            },
            "max" => match (args.first(), args.get(1)) {
                (Some(Value::Int(a)), Some(Value::Int(b))) => Ok(Value::Int(*a.max(b))),
                (Some(Value::Float(a)), Some(Value::Float(b))) => Ok(Value::Float(a.max(*b))),
                _ => Ok(Value::Float(0.0)),
            },
            _ => {
                // Try to find function by name
                if let Some(func) = self.functions.get(name).cloned() {
                    self.call_function(&func, args)
                        .map_err(|_| ControlFlow::Return(Value::Unit))
                } else {
                    Ok(Value::Unit)
                }
            }
        }
    }

    /// Match a pattern against a value, returning bindings if successful
    fn match_pattern(&self, pattern: &HirPattern, value: &Value) -> Option<Vec<(String, Value)>> {
        match pattern {
            HirPattern::Wildcard => Some(vec![]),

            HirPattern::Binding { name, .. } => Some(vec![(name.clone(), value.clone())]),

            HirPattern::Literal(lit) => {
                let lit_val = self.eval_literal(lit);
                if lit_val == *value {
                    Some(vec![])
                } else {
                    None
                }
            }

            HirPattern::Tuple(patterns) => {
                if let Value::Tuple(values) = value {
                    if patterns.len() != values.len() {
                        return None;
                    }
                    let mut bindings = Vec::new();
                    for (pat, val) in patterns.iter().zip(values.iter()) {
                        bindings.extend(self.match_pattern(pat, val)?);
                    }
                    Some(bindings)
                } else {
                    None
                }
            }

            HirPattern::Struct { name, fields } => {
                if let Value::Struct {
                    name: struct_name,
                    fields: struct_fields,
                } = value
                {
                    if name != struct_name {
                        return None;
                    }
                    let mut bindings = Vec::new();
                    for (field_name, field_pat) in fields {
                        let field_val = struct_fields.get(field_name)?;
                        bindings.extend(self.match_pattern(field_pat, field_val)?);
                    }
                    Some(bindings)
                } else {
                    None
                }
            }

            HirPattern::Variant {
                enum_name,
                variant,
                patterns,
            } => {
                if let Value::Variant {
                    enum_name: e,
                    variant_name: v,
                    fields,
                } = value
                {
                    if enum_name != e || variant != v {
                        return None;
                    }
                    if patterns.len() != fields.len() {
                        return None;
                    }
                    let mut bindings = Vec::new();
                    for (pat, val) in patterns.iter().zip(fields.iter()) {
                        bindings.extend(self.match_pattern(pat, val)?);
                    }
                    Some(bindings)
                } else {
                    None
                }
            }

            HirPattern::Or(patterns) => {
                for pat in patterns {
                    if let Some(bindings) = self.match_pattern(pat, value) {
                        return Some(bindings);
                    }
                }
                None
            }
        }
    }

    /// Assign to a target expression
    fn assign_target(&mut self, target: &HirExpr, value: Value) -> Result<(), ControlFlow> {
        match &target.kind {
            HirExprKind::Local(name) => {
                self.env.assign(name, value);
                Ok(())
            }
            HirExprKind::Field { base, field } => {
                let base_val = self.eval_expr(base)?;
                if let Value::Ref(r) = base_val
                    && let Value::Struct { ref mut fields, .. } = *r.borrow_mut()
                {
                    fields.insert(field.clone(), value);
                }
                Ok(())
            }
            HirExprKind::Index { base, index } => {
                let base_val = self.eval_expr(base)?;
                let idx = self.eval_expr(index)?.as_int().unwrap_or(0) as usize;

                if let Value::Array(arr) = base_val {
                    let mut arr = arr.borrow_mut();
                    if idx < arr.len() {
                        arr[idx] = value;
                    }
                }
                Ok(())
            }
            HirExprKind::Deref(inner) => {
                let inner_val = self.eval_expr(inner)?;
                if let Value::Ref(r) = inner_val {
                    *r.borrow_mut() = value;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}
