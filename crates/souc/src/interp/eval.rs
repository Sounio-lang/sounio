//! Tree-walking interpreter for HIR

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use miette::{Result, miette};

use crate::hir::*;
use crate::runtime::async_runtime::{SounioFuture, SounioRuntime, SounioValue};

use super::builtins::{BuiltinRegistry, set_user_args};
use super::effect_dispatch::{
    CapabilityAdapter, DispatchResult, EffectContext, EffectError, EffectHandler, EffectKind,
};
use super::env::Environment;
use super::value::{ControlFlow, SuspensionId, Value};
use crate::effects::continuation::ContinuationId;
use crate::effects::handlers::HandlerRegistry;

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
    /// Async runtime for spawn/await
    async_runtime: SounioRuntime,
    /// Effect handler context for Prob/Causal/etc effects
    effect_ctx: EffectContext,
    /// Emit interpreter trace logs to stderr
    trace: bool,
    /// User arguments passed from CLI (after `--`)
    user_args: Vec<String>,
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
            async_runtime: SounioRuntime::new(),
            effect_ctx: EffectContext::new(),
            trace: false,
            user_args: Vec::new(),
        }
    }

    /// Create a new interpreter with a specific random seed (for reproducibility)
    pub fn with_seed(seed: u64) -> Self {
        Interpreter {
            env: Environment::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            builtins: BuiltinRegistry::new(),
            output: Vec::new(),
            async_runtime: SounioRuntime::new(),
            effect_ctx: EffectContext::with_seed(seed),
            trace: false,
            user_args: Vec::new(),
        }
    }

    /// Create a new interpreter with all effect handlers from the registry.
    ///
    /// This is the recommended way to create an interpreter that supports all
    /// built-in effects (IO, Mut, Alloc, Panic, Prob, GPU, Div, etc.).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut interp = Interpreter::with_all_effects();
    /// // Now can dispatch to any registered effect
    /// ```
    pub fn with_all_effects() -> Self {
        Interpreter {
            env: Environment::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            builtins: BuiltinRegistry::new(),
            output: Vec::new(),
            async_runtime: SounioRuntime::new(),
            effect_ctx: EffectContext::with_all_handlers(),
            trace: false,
            user_args: Vec::new(),
        }
    }

    /// Create a new interpreter with a custom handler registry.
    ///
    /// Use this when you want to provide a custom set of effect handlers.
    pub fn with_registry(registry: HandlerRegistry) -> Self {
        Interpreter {
            env: Environment::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            builtins: BuiltinRegistry::new(),
            output: Vec::new(),
            async_runtime: SounioRuntime::new(),
            effect_ctx: EffectContext::with_registry(registry),
            trace: false,
            user_args: Vec::new(),
        }
    }

    /// Create a new interpreter with trace logging enabled/disabled.
    pub fn with_trace(trace: bool) -> Self {
        let mut interp = Self::new();
        interp.trace = trace;
        interp
    }

    /// Set user args passed after `--` in CLI mode.
    pub fn set_user_args(&mut self, args: Vec<String>) {
        self.user_args = args.clone();
        set_user_args(args);
    }

    /// Get mutable access to effect context
    pub fn effect_ctx_mut(&mut self) -> &mut EffectContext {
        &mut self.effect_ctx
    }

    /// Get read-only access to effect context
    pub fn effect_ctx(&self) -> &EffectContext {
        &self.effect_ctx
    }

    /// Get the async runtime
    pub fn async_runtime(&self) -> &SounioRuntime {
        &self.async_runtime
    }

    // =========================================================================
    // Effect Dispatch Helpers
    // =========================================================================

    /// Dispatch an effect operation by name.
    ///
    /// This is the primary method for invoking effect operations from the interpreter.
    pub fn perform_effect(
        &mut self,
        effect: &str,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<Value, EffectError> {
        self.effect_ctx.dispatch_by_name(effect, operation, args)
    }

    /// Dispatch an IO effect operation.
    pub fn perform_io(&mut self, operation: &str, args: Vec<Value>) -> Result<Value, EffectError> {
        self.effect_ctx.dispatch_by_name("IO", operation, args)
    }

    /// Dispatch a Mut effect operation.
    pub fn perform_mut(&mut self, operation: &str, args: Vec<Value>) -> Result<Value, EffectError> {
        self.effect_ctx.dispatch_by_name("Mut", operation, args)
    }

    /// Dispatch a Div effect operation (safe division).
    pub fn perform_div(&mut self, operation: &str, args: Vec<Value>) -> Result<Value, EffectError> {
        self.effect_ctx.dispatch_by_name("Div", operation, args)
    }

    /// Dispatch a Panic effect operation.
    pub fn perform_panic(
        &mut self,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<Value, EffectError> {
        self.effect_ctx.dispatch_by_name("Panic", operation, args)
    }

    /// Push an effect handler onto the stack.
    ///
    /// Handlers pushed later take precedence over earlier ones.
    pub fn push_handler(&mut self, handler: EffectHandler) {
        self.effect_ctx.push_handler(handler);
    }

    /// Pop an effect handler from the stack.
    pub fn pop_handler(&mut self) -> Option<EffectHandler> {
        self.effect_ctx.pop_handler()
    }

    /// Resume a suspended computation.
    ///
    /// When an effect handler returns `HandlerResult::Suspend`, the computation
    /// is paused and a `ControlFlow::Suspend` is propagated up the call stack.
    /// This method allows resuming that suspended computation with a value.
    ///
    /// # Arguments
    /// * `suspension_id` - The ID of the suspension (from ControlFlow::Suspend)
    /// * `continuation_id` - The continuation ID (from ControlFlow::Suspend)
    /// * `value` - The value to resume with
    ///
    /// # Returns
    /// The result of resuming the continuation
    ///
    /// # Example
    ///
    /// ```ignore
    /// // When evaluation suspends:
    /// match interp.eval_expr(&expr) {
    ///     Err(ControlFlow::Suspend { suspension_id, continuation_id }) => {
    ///         // Do something (e.g., wait for async completion)
    ///         let resume_value = Value::Int(42);
    ///         let result = interp.resume_suspension(suspension_id, continuation_id, resume_value)?;
    ///     }
    ///     // ...
    /// }
    /// ```
    pub fn resume_suspension(
        &mut self,
        suspension_id: SuspensionId,
        continuation_id: ContinuationId,
        value: Value,
    ) -> Result<Value, EffectError> {
        self.effect_ctx
            .resume_suspension(suspension_id, continuation_id, value)
    }

    /// Check if there are any active suspensions.
    pub fn has_active_suspensions(&self) -> bool {
        self.effect_ctx.active_continuation_count() > 0
    }

    /// Get the number of active suspensions.
    pub fn active_suspension_count(&self) -> usize {
        self.effect_ctx.active_continuation_count()
    }

    /// Clear all suspensions (useful for cleanup/reset).
    pub fn clear_suspensions(&mut self) {
        self.effect_ctx.clear_continuations();
    }

    /// Get captured output (for testing)
    pub fn get_output(&self) -> &[String] {
        &self.output
    }

    /// Clear output buffer
    pub fn clear_output(&mut self) {
        self.output.clear();
    }

    fn trace_call(&self, name: &str, args: &[Value]) {
        if !self.trace {
            return;
        }
        let rendered = args
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("[TRACE] call {}({})", name, rendered);
    }

    fn trace_return(&self, value: &Value) {
        if self.trace {
            eprintln!("[TRACE] return {}", value);
        }
    }

    fn trace_undefined(&self, name: &str) {
        if self.trace {
            eprintln!("[TRACE] undefined: {}", name);
        }
    }

    fn trace_binary_mismatch(&self, op: HirBinaryOp, lhs_ty: &str, rhs_ty: &str) {
        if self.trace {
            eprintln!("[TRACE] binary {:?}: {} vs {}", op, lhs_ty, rhs_ty);
        }
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
                HirItem::Impl(imp) => {
                    let type_name = match &imp.self_ty {
                        HirType::Named { name, .. } => name.clone(),
                        _ => continue,
                    };
                    for method in &imp.methods {
                        let qualified = format!("{}::{}", type_name, method.name);
                        self.functions.insert(qualified, Rc::new(method.clone()));
                    }
                }
                _ => {}
            }
        }

        // Second pass: evaluate global constants
        for item in &hir.items {
            if let HirItem::Global(g) = item {
                let val = self.eval_expr(&g.value).unwrap_or(Value::Unit);
                self.env.define(g.name.clone(), val);
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
        self.trace_call(&func.name, &args);
        self.env.push_scope();

        // Bind parameters
        for (param, arg) in func.ty.params.iter().zip(args.into_iter()) {
            self.env.define(param.name.clone(), arg);
        }

        // Execute body
        let result = self.eval_block(&func.body);

        self.env.pop_scope();

        let final_result = match result {
            Ok(v) => Ok(v),
            Err(ControlFlow::Return(v)) => Ok(v),
            Err(ControlFlow::Break(_)) => Err(miette!("break outside loop")),
            Err(ControlFlow::Continue) => Err(miette!("continue outside loop")),
            Err(ControlFlow::Suspend {
                suspension_id,
                continuation_id,
            }) => {
                // Suspension reached the top of a function call
                // This should be handled by an enclosing handle expression
                Err(miette!(
                    "unhandled suspension {} (continuation {}) in function {}",
                    suspension_id,
                    continuation_id,
                    func.name
                ))
            }
        };

        if let Ok(ref value) = final_result {
            self.trace_return(value);
        }

        final_result
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
                    // Wrap mutable structs in Ref to allow field mutation.
                    // Unwrap any existing Ref to prevent double-wrapping.
                    let val = if *is_mut {
                        let inner = match val {
                            Value::Ref(r) => r.borrow().clone(),
                            other => other,
                        };
                        if matches!(inner, Value::Struct { .. }) {
                            Value::Ref(Rc::new(RefCell::new(inner)))
                        } else {
                            inner
                        }
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
                self.trace_undefined(name);
                Err(ControlFlow::Return(Value::Unit))
            }

            HirExprKind::Global(name) => {
                // Resolve Option/Result constructors to native values
                if name == "None" {
                    return Ok(Value::None);
                }
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
                    match self.env.get(name) {
                        Some(v) => Ok(v),
                        None => {
                            self.trace_undefined(name);
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
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
                let lhs_ty = lhs.type_name().to_string();
                let rhs_ty = rhs.type_name().to_string();
                match self.eval_binary(*op, lhs, rhs) {
                    Ok(v) => Ok(v),
                    Err(cf) => {
                        self.trace_binary_mismatch(*op, &lhs_ty, &rhs_ty);
                        Err(cf)
                    }
                }
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
                    Err(cf @ ControlFlow::Suspend { .. }) => {
                        // Propagate suspension up the call stack
                        return Err(cf);
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
                    Err(cf @ ControlFlow::Suspend { .. }) => {
                        // Propagate suspension up the call stack
                        return Err(cf);
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

            HirExprKind::ArrayRepeat { value, count } => {
                let val = self.eval_expr(value)?;
                let values = vec![val; *count];
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

                // Unwrap Value::Ref to get the inner value for indexing
                let base_val = match base_val {
                    Value::Ref(r) => r.borrow().clone(),
                    other => other,
                };

                // Check if this is a Range-based slice operation
                if let Value::Struct { name, fields } = &idx_val {
                    if name == "Range" {
                        // Extract range bounds
                        let start =
                            fields.get("start").and_then(|v| v.as_int()).unwrap_or(0) as usize;
                        let end_val = fields
                            .get("end")
                            .and_then(|v| v.as_int())
                            .unwrap_or(i64::MAX);
                        let inclusive = fields
                            .get("inclusive")
                            .and_then(|v| {
                                if let Value::Bool(b) = v {
                                    Some(*b)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(false);

                        return match base_val {
                            Value::Array(arr) => {
                                let arr = arr.borrow();
                                let len = arr.len();
                                // Handle end bound (clamp to length, handle i64::MAX for open ranges)
                                let end = if end_val >= len as i64 || end_val == i64::MAX {
                                    len
                                } else if inclusive {
                                    (end_val as usize).saturating_add(1).min(len)
                                } else {
                                    (end_val as usize).min(len)
                                };
                                let start = start.min(len);
                                let slice: Vec<Value> = arr[start..end].to_vec();
                                Ok(Value::Array(Rc::new(RefCell::new(slice))))
                            }
                            Value::String(s) => {
                                let chars: Vec<char> = s.chars().collect();
                                let len = chars.len();
                                let end = if end_val >= len as i64 || end_val == i64::MAX {
                                    len
                                } else if inclusive {
                                    (end_val as usize).saturating_add(1).min(len)
                                } else {
                                    (end_val as usize).min(len)
                                };
                                let start = start.min(len);
                                let slice: String = chars[start..end].iter().collect();
                                Ok(Value::String(slice))
                            }
                            _ => Err(ControlFlow::Return(Value::Unit)),
                        };
                    }
                }

                // Regular integer index
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
                    // Box::new is transparent in the interpreter, so deref on
                    // a non-ref value (e.g. a struct from Box) passes through
                    other => Ok(other),
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

            HirExprKind::Cast {
                expr: inner,
                target,
            } => {
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
                    extern_name: None,
                    doc: None,
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
                    (Value::Array(arr), "reverse") => {
                        let reversed: Vec<Value> = arr.borrow().iter().rev().cloned().collect();
                        Ok(Value::Array(Rc::new(RefCell::new(reversed))))
                    }
                    (Value::Array(arr), "map") => {
                        // arr.map(fn) - apply function to each element
                        if let Some(Value::Function { func, captures }) = arg_values.get(1) {
                            let arr_ref = arr.borrow();
                            let mut results = Vec::with_capacity(arr_ref.len());
                            for elem in arr_ref.iter() {
                                let call_args = vec![elem.clone()];
                                match self.call_function(func, call_args) {
                                    Ok(result) => results.push(result),
                                    Err(_) => results.push(Value::Unit),
                                }
                            }
                            Ok(Value::Array(Rc::new(RefCell::new(results))))
                        } else {
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
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
                    // Value::Ref: unwrap the reference and dispatch method to inner value
                    (Value::Ref(r), method_name) => {
                        let inner = r.borrow().clone();
                        match (inner, method_name) {
                            (Value::Array(arr), "len") => Ok(Value::Int(arr.borrow().len() as i64)),
                            (Value::Array(arr), "push") => {
                                if let Some(val) = arg_values.get(1) {
                                    arr.borrow_mut().push(val.clone());
                                }
                                Ok(Value::Unit)
                            }
                            (Value::Array(arr), "pop") => {
                                Ok(arr.borrow_mut().pop().unwrap_or(Value::None))
                            }
                            (Value::Array(arr), "reverse") => {
                                let reversed: Vec<Value> =
                                    arr.borrow().iter().rev().cloned().collect();
                                Ok(Value::Array(Rc::new(RefCell::new(reversed))))
                            }
                            (Value::Array(arr), "map") => {
                                if let Some(Value::Function { func, captures: _ }) =
                                    arg_values.get(1)
                                {
                                    let arr_ref = arr.borrow();
                                    let mut results = Vec::with_capacity(arr_ref.len());
                                    for elem in arr_ref.iter() {
                                        let call_args = vec![elem.clone()];
                                        match self.call_function(func, call_args) {
                                            Ok(result) => results.push(result),
                                            Err(_) => results.push(Value::Unit),
                                        }
                                    }
                                    Ok(Value::Array(Rc::new(RefCell::new(results))))
                                } else {
                                    Err(ControlFlow::Return(Value::Unit))
                                }
                            }
                            (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
                            (Value::String(s), "slice") => {
                                let start = match arg_values.get(1) {
                                    Some(Value::Int(i)) => (*i).max(0) as usize,
                                    _ => 0,
                                };
                                let end = match arg_values.get(2) {
                                    Some(Value::Int(i)) => (*i).max(0) as usize,
                                    _ => s.len(),
                                };
                                let chars: Vec<char> = s.chars().collect();
                                let start = start.min(chars.len());
                                let end = end.min(chars.len()).max(start);
                                let result: String = chars[start..end].iter().collect();
                                Ok(Value::String(result))
                            }
                            // For struct method calls on references, look up as function
                            (Value::Struct { name, .. }, _) => {
                                let qname = format!("{}::{}", name, method_name);
                                if let Some(func) = self.functions.get(&qname).cloned() {
                                    self.call_function(&func, arg_values)
                                        .map_err(|_| ControlFlow::Return(Value::Unit))
                                } else if let Some(func) = self.functions.get(method_name).cloned()
                                {
                                    self.call_function(&func, arg_values)
                                        .map_err(|_| ControlFlow::Return(Value::Unit))
                                } else {
                                    Err(ControlFlow::Return(Value::Unit))
                                }
                            }
                            _ => Err(ControlFlow::Return(Value::Unit)),
                        }
                    }
                    _ => {
                        // Try qualified name (Type::method) first for struct receivers
                        let qualified = match &arg_values[0] {
                            Value::Struct { name, .. } => Some(format!("{}::{}", name, method)),
                            Value::Ref(r) => {
                                if let Value::Struct { name, .. } = &*r.borrow() {
                                    Some(format!("{}::{}", name, method))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        if let Some(ref qname) = qualified {
                            if let Some(func) = self.functions.get(qname).cloned() {
                                return self
                                    .call_function(&func, arg_values)
                                    .map_err(|_| ControlFlow::Return(Value::Unit));
                            }
                        }
                        // Fall back to bare method name
                        if let Some(func) = self.functions.get(method).cloned() {
                            self.call_function(&func, arg_values)
                                .map_err(|_| ControlFlow::Return(Value::Unit))
                        } else {
                            Err(ControlFlow::Return(Value::Unit))
                        }
                    }
                }
            }

            // ==================== EFFECT OPERATIONS ====================
            HirExprKind::Perform { effect, op, args } => {
                // Perform an effect operation by dispatching to the handler stack
                let mut arg_values = Vec::with_capacity(args.len());
                for arg in args {
                    arg_values.push(self.eval_expr(arg)?);
                }

                // Dispatch to effect handler with suspension support
                match self
                    .effect_ctx
                    .dispatch_by_name_with_continuation(effect, op, arg_values)
                {
                    Ok(DispatchResult::Completed(result)) => Ok(result),
                    Ok(DispatchResult::Suspended {
                        suspension_id,
                        continuation_id,
                    }) => {
                        // Handler suspended - propagate up the call stack
                        Err(ControlFlow::Suspend {
                            suspension_id,
                            continuation_id,
                        })
                    }
                    Err(e) => {
                        // Convert effect error to control flow error
                        // In production, this would generate proper diagnostics
                        eprintln!("Effect error: {}", e);
                        Ok(Value::Unit)
                    }
                }
            }

            HirExprKind::Handle { expr, handler } => {
                // Handle wraps an expression with a named handler
                // The handler name corresponds to an effect name (e.g., "IO", "Mut", "Panic")
                //
                // If we have a registry, create a handler from it and push it onto the stack.
                // This allows the expression to use the handler for effect operations.
                let handler_pushed = if let Some(registry) = self.effect_ctx.registry() {
                    if let Some(capability_handler) = registry.get(handler) {
                        // Clone the Arc and create an adapter to convert to EffectHandler
                        let cloned_handler = capability_handler.clone();
                        let adapter = CapabilityAdapter::from_arc(cloned_handler);
                        let effect_handler = adapter.to_effect_handler();
                        self.effect_ctx.push_handler(effect_handler);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                // Evaluate the wrapped expression, catching suspensions
                let result = self.eval_expr(expr);

                // Pop the handler if we pushed one
                if handler_pushed {
                    self.effect_ctx.pop_handler();
                }

                // Handle suspensions from the inner expression
                match result {
                    Err(ControlFlow::Suspend {
                        suspension_id,
                        continuation_id,
                    }) => {
                        // The inner expression suspended.
                        // For now, we propagate the suspension up.
                        // A full implementation would allow the handler to:
                        // 1. Resume immediately with a value
                        // 2. Store the suspension for later resumption
                        // 3. Abort the computation
                        //
                        // The handler could inspect suspension_id to decide what to do.
                        // For now, we just propagate it.
                        Err(ControlFlow::Suspend {
                            suspension_id,
                            continuation_id,
                        })
                    }
                    other => other,
                }
            }

            HirExprKind::Resume { value } => {
                // Resume a suspended continuation with a value
                // This is typically used inside effect handlers
                let resume_value = self.eval_expr(value)?;

                // Look up the current active continuation from the handler context
                if let Some(cont_id) = self.effect_ctx.current_continuation() {
                    // Resume the continuation with the value
                    match self.effect_ctx.resume_continuation(cont_id, resume_value) {
                        Ok(result) => Ok(result),
                        Err(e) => {
                            // Convert continuation error to control flow error
                            // In production, this would generate proper diagnostics
                            eprintln!("Continuation error: {}", e);
                            Ok(Value::Unit)
                        }
                    }
                } else {
                    // No active continuation - this happens when resume is used outside
                    // a handler or when the handler doesn't set up a continuation.
                    // For simple cases where resume is the last expression in a handler,
                    // we can just return the value directly.
                    Ok(resume_value)
                }
            }

            HirExprKind::Sample(dist_expr) => {
                // Sample from a probability distribution (Prob effect)
                let dist_val = self.eval_expr(dist_expr)?;

                match self.effect_ctx.sample(dist_val) {
                    Ok(result) => Ok(result),
                    Err(e) => {
                        eprintln!("Sample error: {}", e);
                        Ok(Value::Float(0.0))
                    }
                }
            }

            // ==================== EPISTEMIC EXPRESSIONS ====================
            HirExprKind::Knowledge {
                value,
                epsilon,
                validity,
                provenance,
            } => {
                // Evaluate the inner value
                let val = self.eval_expr(value)?;

                // Extract confidence from epsilon (uncertainty)
                let eps = self.eval_expr(epsilon)?;
                let confidence = match eps {
                    Value::Float(e) => (1.0 - e).max(0.0).min(1.0),
                    Value::Int(e) => (1.0 - (e as f64)).max(0.0).min(1.0),
                    _ => 1.0, // Default to certain if epsilon is not numeric
                };

                // Generate provenance ID from provenance expression
                let provenance_id = if let Some(prov) = provenance {
                    // Hash the provenance expression to get an ID
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    format!("{:?}", prov).hash(&mut hasher);
                    (hasher.finish() & 0xFFFFFFFF) as u32
                } else {
                    0 // No provenance
                };

                Ok(Value::Knowledge {
                    value: Box::new(val),
                    confidence,
                    provenance_id,
                })
            }

            HirExprKind::Do { variable, value } => {
                // Do intervention - dispatch to Causal effect handler
                let val = self.eval_expr(value)?;
                let var_val = Value::String(variable.clone());

                match self.effect_ctx.do_intervention(var_val, val.clone()) {
                    Ok(result) => Ok(result),
                    Err(e) => {
                        eprintln!("Do intervention error: {}", e);
                        Ok(val)
                    }
                }
            }

            HirExprKind::Counterfactual {
                factual,
                intervention,
                outcome,
            } => {
                // Counterfactual reasoning via Causal effect
                let factual_val = self.eval_expr(factual)?;
                let intervention_val = self.eval_expr(intervention)?;
                let outcome_val = self.eval_expr(outcome)?;

                match self.effect_ctx.counterfactual(
                    factual_val,
                    intervention_val,
                    outcome_val.clone(),
                ) {
                    Ok(result) => Ok(result),
                    Err(e) => {
                        eprintln!("Counterfactual error: {}", e);
                        Ok(outcome_val)
                    }
                }
            }

            HirExprKind::Query {
                target,
                given,
                interventions,
            } => {
                // Probabilistic query - P(target | given, do(interventions))
                // Dispatch to Causal effect handler
                let target_val = self.eval_expr(target)?;

                // Build query arguments
                let mut query_args = vec![target_val.clone()];
                for g in given {
                    query_args.push(self.eval_expr(g)?);
                }

                match self
                    .effect_ctx
                    .dispatch(EffectKind::Causal, "query", query_args)
                {
                    Ok(result) => Ok(result),
                    Err(e) => {
                        eprintln!("Query error: {}", e);
                        Ok(target_val)
                    }
                }
            }

            HirExprKind::Observe { variable, value } => {
                // Observe for probabilistic programming - dispatch to Prob effect
                let val = self.eval_expr(value)?;

                // For probabilistic observe, we need a distribution
                // Create a simple observation struct
                let mut fields = HashMap::new();
                fields.insert("variable".to_string(), Value::String(variable.clone()));
                fields.insert("value".to_string(), val.clone());
                let obs_struct = Value::Struct {
                    name: "Observation".to_string(),
                    fields,
                };

                // Record the observation in effect context
                self.effect_ctx
                    .state
                    .observations
                    .push((variable.clone(), val.clone()));

                Ok(val)
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

            // ==================== ASYNC EXPRESSIONS ====================
            HirExprKind::Await { future } => {
                // Evaluate the future expression
                let future_val = self.eval_expr(future)?;

                // If it's a Value::Future, block until completion and return result
                match future_val {
                    Value::Future {
                        future: sounio_future,
                    } => {
                        // Block on the future and get the result
                        let result = self.async_runtime.block_on_future(&sounio_future);
                        Ok(self.sounio_value_to_value(result))
                    }
                    Value::Task { handle } => {
                        // For a task, wait for it to complete
                        loop {
                            if let Some(result) = handle.try_get() {
                                return Ok(self.sounio_value_to_value(result));
                            }
                            // Yield to allow other work
                            std::thread::yield_now();
                        }
                    }
                    // If it's not a future, just return the value
                    other => Ok(other),
                }
            }

            HirExprKind::Spawn { expr } => {
                // Create a future for the spawned expression
                let future = SounioFuture::new();
                let future_clone = future.clone();

                // Evaluate the expression and complete the future
                let result = self.eval_expr(expr)?;
                let sounio_result = self.value_to_sounio_value(&result);
                future_clone.complete(sounio_result);

                // Return a task handle
                Ok(Value::Task {
                    handle: self.async_runtime.spawn_future(future),
                })
            }

            HirExprKind::AsyncBlock { body } => {
                // Create a future for the async block
                let future = SounioFuture::new();
                let future_clone = future.clone();

                // Evaluate the block
                let result = match self.eval_block(body) {
                    Ok(v) => v,
                    Err(ControlFlow::Return(v)) => v,
                    Err(e) => return Err(e),
                };

                // Complete the future with the result
                let sounio_result = self.value_to_sounio_value(&result);
                future_clone.complete(sounio_result);

                // Return the future as a Value
                Ok(Value::Future { future })
            }

            HirExprKind::Join { futures } => {
                // Evaluate all future expressions
                let mut results = Vec::new();
                for future_expr in futures {
                    let future_val = self.eval_expr(future_expr)?;
                    // Await each future (in order for now - a real implementation would be concurrent)
                    match future_val {
                        Value::Future {
                            future: sounio_future,
                        } => {
                            let result = self.async_runtime.block_on_future(&sounio_future);
                            results.push(self.sounio_value_to_value(result));
                        }
                        Value::Task { handle } => loop {
                            if let Some(result) = handle.try_get() {
                                results.push(self.sounio_value_to_value(result));
                                break;
                            }
                            std::thread::yield_now();
                        },
                        other => {
                            results.push(other);
                        }
                    }
                }
                // Return results as a tuple
                Ok(Value::Tuple(results))
            }

            HirExprKind::Select { arms } => {
                // Simple select implementation: check each arm in order
                // A real implementation would use proper async polling
                for arm in arms {
                    let future_val = self.eval_expr(&arm.future)?;

                    let result = match future_val {
                        Value::Future {
                            future: ref sounio_future,
                        } if sounio_future.is_completed() => sounio_future.try_get(),
                        Value::Task { ref handle } if handle.is_completed() => handle.try_get(),
                        _ => continue, // Not ready, try next arm
                    };

                    if let Some(sounio_val) = result {
                        let val = self.sounio_value_to_value(sounio_val);
                        // Bind the result to the pattern
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
                }
                // No arm was ready - in a real implementation we'd block
                // For now, return unit
                Ok(Value::Unit)
            }
        }
    }

    /// Convert a SounioValue to a Value
    fn sounio_value_to_value(&self, sounio: SounioValue) -> Value {
        match sounio {
            SounioValue::Unit => Value::Unit,
            SounioValue::Bool(b) => Value::Bool(b),
            SounioValue::Int(n) => Value::Int(n),
            SounioValue::Float(f) => Value::Float(f),
            SounioValue::String(s) => Value::String(s),
            SounioValue::Future(id) => {
                // Try to get the future from the runtime
                if let Some(future) = self.async_runtime.get_task(id) {
                    Value::Future { future }
                } else {
                    Value::None
                }
            }
            SounioValue::Array(arr) => {
                let values: Vec<Value> = arr
                    .into_iter()
                    .map(|v| self.sounio_value_to_value(v))
                    .collect();
                Value::Array(Rc::new(RefCell::new(values)))
            }
            SounioValue::Tuple(t) => {
                let values: Vec<Value> = t
                    .into_iter()
                    .map(|v| self.sounio_value_to_value(v))
                    .collect();
                Value::Tuple(values)
            }
            SounioValue::Struct { name, fields } => {
                let converted_fields: HashMap<String, Value> = fields
                    .into_iter()
                    .map(|(k, v)| (k, self.sounio_value_to_value(v)))
                    .collect();
                Value::Struct {
                    name,
                    fields: converted_fields,
                }
            }
            SounioValue::Variant {
                enum_name,
                variant_name,
                fields,
            } => {
                let converted_fields: Vec<Value> = fields
                    .into_iter()
                    .map(|v| self.sounio_value_to_value(v))
                    .collect();
                Value::Variant {
                    enum_name,
                    variant_name,
                    fields: converted_fields,
                }
            }
            SounioValue::None => Value::None,
            SounioValue::Some(v) => Value::Some(Box::new(self.sounio_value_to_value(*v))),
            SounioValue::Ok(v) => Value::Ok(Box::new(self.sounio_value_to_value(*v))),
            SounioValue::Err(v) => Value::Err(Box::new(self.sounio_value_to_value(*v))),
        }
    }

    /// Convert a Value to a SounioValue
    fn value_to_sounio_value(&self, value: &Value) -> SounioValue {
        match value {
            Value::Unit => SounioValue::Unit,
            Value::Bool(b) => SounioValue::Bool(*b),
            Value::Int(n) => SounioValue::Int(*n),
            Value::Float(f) => SounioValue::Float(*f),
            Value::String(s) => SounioValue::String(s.clone()),
            Value::Future { future } => SounioValue::Future(future.task_id()),
            Value::Task { handle } => SounioValue::Future(handle.task_id()),
            Value::Array(arr) => {
                let values: Vec<SounioValue> = arr
                    .borrow()
                    .iter()
                    .map(|v| self.value_to_sounio_value(v))
                    .collect();
                SounioValue::Array(values)
            }
            Value::Tuple(t) => {
                let values: Vec<SounioValue> =
                    t.iter().map(|v| self.value_to_sounio_value(v)).collect();
                SounioValue::Tuple(values)
            }
            Value::Struct { name, fields } => {
                let converted_fields: HashMap<String, SounioValue> = fields
                    .iter()
                    .map(|(k, v)| (k.clone(), self.value_to_sounio_value(v)))
                    .collect();
                SounioValue::Struct {
                    name: name.clone(),
                    fields: converted_fields,
                }
            }
            Value::Variant {
                enum_name,
                variant_name,
                fields,
            } => {
                let converted_fields: Vec<SounioValue> = fields
                    .iter()
                    .map(|v| self.value_to_sounio_value(v))
                    .collect();
                SounioValue::Variant {
                    enum_name: enum_name.clone(),
                    variant_name: variant_name.clone(),
                    fields: converted_fields,
                }
            }
            Value::None => SounioValue::None,
            Value::Some(v) => SounioValue::Some(Box::new(self.value_to_sounio_value(v))),
            Value::Ok(v) => SounioValue::Ok(Box::new(self.value_to_sounio_value(v))),
            Value::Err(v) => SounioValue::Err(Box::new(self.value_to_sounio_value(v))),
            // For other types, convert to a sensible default
            _ => SounioValue::Unit,
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
            // C string literal: for interpreter, treat as a pointer to a null-terminated string
            // In a real FFI context this would be a raw pointer; here we use RawPointer with
            // the string content stored in a global
            HirLiteral::CString(s) => {
                // For interpreter purposes, we represent C strings as strings with null terminator
                let mut with_null = Self::process_escape_sequences(s);
                with_null.push('\0');
                Value::String(with_null)
            }
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
            (
                Value::Bool(b),
                HirType::I8 | HirType::I16 | HirType::I32 | HirType::I64 | HirType::Isize,
            ) => Ok(Value::Int(if b { 1 } else { 0 })),
            (
                Value::Bool(b),
                HirType::U8 | HirType::U16 | HirType::U32 | HirType::U64 | HirType::Usize,
            ) => Ok(Value::Int(if b { 1 } else { 0 })),

            // Int/Float/Bool to String casts
            (Value::Int(n), HirType::String) => Ok(Value::String(n.to_string())),
            (Value::Float(f), HirType::String) => Ok(Value::String(format!("{}", f))),
            (Value::Bool(b), HirType::String) => {
                Ok(Value::String(if b { "true" } else { "false" }.to_string()))
            }

            // Identity cast - value stays the same
            (v, _) => Ok(v),
        }
    }

    /// Unwrap Knowledge values and extract their metadata
    fn unwrap_knowledge(val: Value) -> (Value, f64, u32) {
        match val {
            Value::Knowledge {
                value,
                confidence,
                provenance_id,
            } => (*value, confidence, provenance_id),
            _ => (val, 1.0, 0), // Non-Knowledge values have confidence 1.0
        }
    }

    /// Combine two confidence values for binary operations
    fn combine_confidence(c1: f64, c2: f64) -> f64 {
        // For now, use multiplication (confidence degrades)
        (c1 * c2).max(0.0).min(1.0)
    }

    /// Combine two provenance IDs
    fn combine_provenance(p1: u32, p2: u32) -> u32 {
        // XOR for simple combination
        p1 ^ p2
    }

    /// Wrap result in Knowledge if either operand was Knowledge
    fn wrap_knowledge(result: Value, conf1: f64, conf2: f64, prov1: u32, prov2: u32) -> Value {
        // If either input was Knowledge (confidence < 1.0 or prov != 0), wrap result
        if conf1 < 1.0 || conf2 < 1.0 || prov1 != 0 || prov2 != 0 {
            Value::Knowledge {
                value: Box::new(result),
                confidence: Self::combine_confidence(conf1, conf2),
                provenance_id: Self::combine_provenance(prov1, prov2),
            }
        } else {
            result
        }
    }

    /// Evaluate a binary operation
    fn eval_binary(&self, op: HirBinaryOp, lhs: Value, rhs: Value) -> Result<Value, ControlFlow> {
        // Unwrap Knowledge values if present
        let (lhs, lhs_conf, lhs_prov) = Self::unwrap_knowledge(lhs);
        let (rhs, rhs_conf, rhs_prov) = Self::unwrap_knowledge(rhs);

        match op {
            HirBinaryOp::Add => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a + b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a + b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Int(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a as f64 + b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a + b as f64),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::String(a), Value::String(b)) => Ok(Self::wrap_knowledge(
                    Value::String(a + &b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Sub => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a - b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a - b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Int(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a as f64 - b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a - b as f64),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Mul => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a * b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a * b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Int(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a as f64 * b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a * b as f64),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Div => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => {
                    if b == 0 {
                        Err(ControlFlow::Return(Value::Unit)) // Division by zero
                    } else {
                        Ok(Self::wrap_knowledge(
                            Value::Int(a / b),
                            lhs_conf,
                            rhs_conf,
                            lhs_prov,
                            rhs_prov,
                        ))
                    }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a / b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Int(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a as f64 / b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a / b as f64),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Rem => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => {
                    if b == 0 {
                        Err(ControlFlow::Return(Value::Unit))
                    } else {
                        Ok(Self::wrap_knowledge(
                            Value::Int(a % b),
                            lhs_conf,
                            rhs_conf,
                            lhs_prov,
                            rhs_prov,
                        ))
                    }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a % b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Eq => Ok(Self::wrap_knowledge(
                Value::Bool(lhs == rhs),
                lhs_conf,
                rhs_conf,
                lhs_prov,
                rhs_prov,
            )),
            HirBinaryOp::Ne => Ok(Self::wrap_knowledge(
                Value::Bool(lhs != rhs),
                lhs_conf,
                rhs_conf,
                lhs_prov,
                rhs_prov,
            )),
            HirBinaryOp::Lt => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a < b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a < b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::String(a), Value::String(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a < b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Le => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a <= b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a <= b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::String(a), Value::String(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a <= b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Gt => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a > b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a > b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::String(a), Value::String(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a > b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Ge => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a >= b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a >= b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::String(a), Value::String(b)) => Ok(Self::wrap_knowledge(
                    Value::Bool(a >= b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::And => Ok(Value::Bool(lhs.is_truthy() && rhs.is_truthy())),
            HirBinaryOp::Or => Ok(Value::Bool(lhs.is_truthy() || rhs.is_truthy())),
            HirBinaryOp::BitAnd => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a & b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::BitOr => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a | b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::BitXor => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a ^ b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Shl => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a << b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::Shr => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a >> b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                _ => Err(ControlFlow::Return(Value::Unit)),
            },
            HirBinaryOp::PlusMinus => match (lhs, rhs) {
                // Treat PlusMinus as Add (uncertainty is handled at type-check time)
                (Value::Int(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Int(a + b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a + b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Int(a), Value::Float(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a as f64 + b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
                (Value::Float(a), Value::Int(b)) => Ok(Self::wrap_knowledge(
                    Value::Float(a + b as f64),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
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
                (Value::String(a), Value::String(b)) => Ok(Self::wrap_knowledge(
                    Value::String(a + &b),
                    lhs_conf,
                    rhs_conf,
                    lhs_prov,
                    rhs_prov,
                )),
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
                // Box deref is transparent — pass through non-ref values
                other => Ok(other),
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
                | "print_int"
                | "print_char"
                | "assert"
                | "assert_eq"
                | "len"
                | "type_of"
                | "Some"
                | "None"
                | "Ok"
                | "Err"
                | "Box::new"
                | "dbg"
                | "panic"
                | "format"
                | "read_byte"
                | "read_file"
                | "file_size"
                | "write_bytes"
                | "arg_count"
                | "get_arg"
                | "str_len"
                | "str_eq"
                | "str_concat"
                | "str_slice"
                | "str_char_at"
                | "str_from_bytes"
                | "int_to_str"
                | "starts_with"
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
                // Slice construction intrinsics
                | "__builtin_slice_from_raw_parts"
                | "__builtin_slice_from_raw_parts_mut"
        )
    }

    /// Try calling a builtin function by examining arguments
    fn call_builtin_by_args(&mut self, args: &[Value]) -> Result<Value, ControlFlow> {
        // Default: return unit
        Ok(Value::Unit)
    }

    /// Call a named builtin function
    pub fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, ControlFlow> {
        self.trace_call(name, &args);

        // First, try the builtin registry
        if self.builtins.is_builtin(name) {
            match self.builtins.call(name, &args) {
                Ok(v) => {
                    self.trace_return(&v);
                    return Ok(v);
                }
                Err(e) => return Err(ControlFlow::Return(Value::String(e))),
            }
        }

        // Fall back to hardcoded implementations
        let result = match name {
            "print" => {
                let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
                let line = output.join(" ");
                print!("{}", line);
                self.output.push(line);
                Ok(Value::Unit)
            }
            "print_char" => {
                if let Some(Value::Int(code)) = args.first() {
                    let ch = (*code as u8) as char;
                    print!("{}", ch);
                    self.output.push(ch.to_string());
                }
                Ok(Value::Unit)
            }
            "print_int" => {
                if let Some(Value::Int(n)) = args.first() {
                    print!("{}", n);
                    self.output.push(n.to_string());
                }
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
            // Box::new(v) is transparent in the interpreter — just return the value
            "Box::new" => {
                if let Some(val) = args.into_iter().next() {
                    match val {
                        // Avoid leaking internal mutable-cell refs into boxed value positions.
                        // This preserves value semantics for self-hosted AST nodes that use
                        // Box<T> pervasively and prevents accidental self-referential cycles.
                        Value::Ref(r) => Ok(r.borrow().clone()),
                        other => Ok(other),
                    }
                } else {
                    Ok(Value::Unit)
                }
            }
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
            "read_byte" => {
                use std::io::Read;
                let mut buf = [0u8; 1];
                match std::io::stdin().lock().read(&mut buf) {
                    Ok(1) => Ok(Value::Int(buf[0] as i64)),
                    _ => Ok(Value::Int(-1)), // EOF
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
            // Slice construction intrinsics - for interpreter, just return the array
            // In interpreter, ptr is simulated as an array/array-ref, len is the length
            // We just return the array data directly since interpreter doesn't have real pointers
            "__builtin_slice_from_raw_parts" | "__builtin_slice_from_raw_parts_mut" => {
                match args.first() {
                    Some(Value::Array(arr)) => {
                        // Return the array directly (simulating a slice)
                        Ok(Value::Array(arr.clone()))
                    }
                    Some(Value::Ref(r)) => {
                        // Return the referenced value
                        Ok(r.borrow().clone())
                    }
                    _ => {
                        // No valid args - return empty array
                        use std::cell::RefCell;
                        use std::rc::Rc;
                        let arr = Rc::new(RefCell::new(Vec::<Value>::new()));
                        Ok(Value::Array(arr))
                    }
                }
            }
            _ => {
                // Try to find function by name
                if let Some(func) = self.functions.get(name).cloned() {
                    self.call_function(&func, args)
                        .map_err(|_| ControlFlow::Return(Value::Unit))
                } else {
                    Ok(Value::Unit)
                }
            }
        };

        if let Ok(ref value) = result {
            self.trace_return(value);
        }

        result
    }

    /// Match a pattern against a value, returning bindings if successful
    fn match_pattern(&self, pattern: &HirPattern, value: &Value) -> Option<Vec<(String, Value)>> {
        match pattern {
            HirPattern::Wildcard => Some(vec![]),

            HirPattern::Binding { name, .. } => {
                // HIR lowering may emit `None` as a Binding instead of a Variant.
                // Treat binding named "None" as a variant match when the value is
                // Option::None (either Value::None or Value::Variant with variant_name "None").
                if name == "None" {
                    match value {
                        Value::None => return Some(vec![]),
                        Value::Some(_) => return None,
                        Value::Variant {
                            variant_name,
                            fields,
                            ..
                        } => {
                            if variant_name == "None" && fields.is_empty() {
                                return Some(vec![]);
                            } else {
                                return None;
                            }
                        }
                        _ => {} // Fall through to normal binding
                    }
                }
                Some(vec![(name.clone(), value.clone())])
            }

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
                // Handle builtin Value::None and Value::Some for Option pattern matching.
                // The HIR may emit Some(v) as Variant { enum_name: "", variant: "Some", patterns: [v] }
                // but the runtime uses Value::Some(inner) / Value::None instead of Value::Variant.
                if variant == "None" {
                    if matches!(value, Value::None) && patterns.is_empty() {
                        return Some(vec![]);
                    }
                    // Also handle Value::Variant with variant_name "None"
                    if let Value::Variant {
                        variant_name,
                        fields,
                        ..
                    } = value
                    {
                        if variant_name == "None" && fields.is_empty() && patterns.is_empty() {
                            return Some(vec![]);
                        }
                    }
                    return None;
                }
                if variant == "Some" {
                    if let Value::Some(inner) = value {
                        if patterns.len() == 1 {
                            return self.match_pattern(&patterns[0], inner);
                        }
                        return if patterns.is_empty() {
                            Some(vec![])
                        } else {
                            None
                        };
                    }
                }

                if let Value::Variant {
                    enum_name: e,
                    variant_name: v,
                    fields,
                } = value
                {
                    // Allow empty enum_name in pattern to match any enum type
                    // (HIR lowering doesn't always resolve the enum name for
                    //  well-known types like Option)
                    if (!enum_name.is_empty() && enum_name != e) || variant != v {
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
                // If existing binding is a Ref (mutable struct variable), update its
                // content rather than replacing the binding. This preserves the Ref
                // wrapper so subsequent field assignments continue to work.
                if let Some(existing) = self.env.get(name) {
                    if let Value::Ref(r) = existing {
                        // Unwrap incoming Ref to prevent double-wrapping
                        let inner = match value {
                            Value::Ref(inner_r) => inner_r.borrow().clone(),
                            other => other,
                        };
                        *r.borrow_mut() = inner;
                        return Ok(());
                    }
                }
                self.env.assign(name, value);
                Ok(())
            }
            HirExprKind::Field { base, field } => {
                // Special case: arr[i].field = value - need to modify array element in-place
                if let HirExprKind::Index {
                    base: arr_base,
                    index,
                } = &base.kind
                {
                    let arr_val = self.eval_expr(arr_base)?;
                    let idx = self.eval_expr(index)?.as_int().unwrap_or(0) as usize;

                    if let Value::Array(arr) = arr_val {
                        let mut arr = arr.borrow_mut();
                        if idx < arr.len() {
                            if let Value::Struct { ref mut fields, .. } = arr[idx] {
                                fields.insert(field.clone(), value);
                            }
                        }
                    }
                    return Ok(());
                }

                let base_val = self.eval_expr(base)?;
                if let Value::Ref(r) = base_val
                    && let Value::Struct { ref mut fields, .. } = *r.borrow_mut()
                {
                    fields.insert(field.clone(), value);
                }
                Ok(())
            }
            HirExprKind::Index { base, index } => {
                let idx = self.eval_expr(index)?.as_int().unwrap_or(0) as usize;

                // Special case: struct_var.array_field[idx] = value
                // Need to modify the array in-place within the Ref-wrapped struct
                if let HirExprKind::Field {
                    base: struct_base,
                    field,
                } = &base.kind
                {
                    let struct_val = self.eval_expr(struct_base)?;
                    if let Value::Ref(r) = struct_val {
                        let mut inner = r.borrow_mut();
                        if let Value::Struct { ref mut fields, .. } = *inner {
                            if let Some(Value::Array(arr)) = fields.get(field) {
                                let mut arr = arr.borrow_mut();
                                if idx < arr.len() {
                                    arr[idx] = value;
                                }
                                return Ok(());
                            }
                        }
                    }
                }

                let base_val = self.eval_expr(base)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpreter_with_all_effects() {
        let interp = Interpreter::with_all_effects();
        assert!(interp.effect_ctx().registry().is_some());
        assert_eq!(interp.effect_ctx().registry().unwrap().len(), 12);
    }

    #[test]
    fn test_interpreter_with_registry() {
        use crate::effects::handlers::HandlerRegistryBuilder;

        let registry = HandlerRegistryBuilder::new().with_io().with_mut().build();

        let interp = Interpreter::with_registry(registry);
        assert!(interp.effect_ctx().registry().is_some());
        assert_eq!(interp.effect_ctx().registry().unwrap().len(), 2);
    }

    #[test]
    fn test_perform_effect_io() {
        let mut interp = Interpreter::with_all_effects();

        // Dispatch IO.print
        let result = interp.perform_io("print", vec![Value::String("test".into())]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_perform_effect_div() {
        let mut interp = Interpreter::with_all_effects();

        // Dispatch Div.div
        let result = interp.perform_div("div", vec![Value::Float(10.0), Value::Float(2.0)]);
        assert!(result.is_ok());
        if let Ok(Value::Float(v)) = result {
            assert!((v - 5.0).abs() < 0.001);
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_perform_effect_mut() {
        let mut interp = Interpreter::with_all_effects();

        // Set a value
        let set_result = interp.perform_mut(
            "set",
            vec![Value::String("counter".into()), Value::Int(100)],
        );
        assert!(set_result.is_ok());

        // Get the value back
        let get_result = interp.perform_mut("get", vec![Value::String("counter".into())]);
        assert!(get_result.is_ok());
        assert_eq!(get_result.unwrap(), Value::Int(100));
    }

    #[test]
    fn test_perform_effect_panic() {
        let mut interp = Interpreter::with_all_effects();

        // Assert true should succeed
        let result = interp.perform_panic("assert", vec![Value::Bool(true)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_perform_effect_generic() {
        let mut interp = Interpreter::with_all_effects();

        // Use the generic perform_effect method
        let result = interp.perform_effect("GPU", "sync", vec![]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_push_pop_handler() {
        let mut interp = Interpreter::with_all_effects();

        // Create a custom handler
        let custom_handler = EffectHandler::new(EffectKind::IO, "custom_io")
            .with_case("print", |_args, _state| Ok(Value::String("custom".into())));

        // Push the handler
        interp.push_handler(custom_handler);

        // Pop should return something
        let popped = interp.pop_handler();
        assert!(popped.is_some());
    }

    #[test]
    fn test_method_call_assignment_with_print_int_keeps_struct_state() {
        let source = r#"
struct Span {
    start: i64,
    end: i64,
}

struct Checker {
    error_count: i64,
    had_error: bool,
}

impl Checker {
    fn report_mismatch(self, span: Span, code: i64) -> Checker with IO, Mut, Panic, Div, Alloc {
        var c = self
        c.error_count = c.error_count + 1
        c.had_error = true
        print("error[E")
        if code < 100 { print("0") }
        if code < 10 { print("0") }
        print_int(code)
        print("]\n")
        c
    }
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    var c = Checker { error_count: 0, had_error: false }
    let sp = Span { start: 0, end: 0 }
    c = c.report_mismatch(sp, 8)
    print_int(c.error_count)
    print("\n")
    0
}
"#;

        let tokens = crate::lexer::lex(source).expect("lex should succeed");
        let ast = crate::parser::parse(&tokens, source).expect("parse should succeed");
        let hir = crate::check::check_ast(&ast).expect("typecheck should succeed");

        let mut interp = Interpreter::new();
        let result = interp.interpret(&hir).expect("interpret should succeed");

        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_interpreter_default_uses_new() {
        let interp = Interpreter::default();
        // Default uses new() which doesn't have a registry
        assert!(interp.effect_ctx().registry().is_none());
    }

    #[test]
    fn test_interpreter_with_trace_constructor() {
        let _interp = Interpreter::with_trace(true);
    }
}
