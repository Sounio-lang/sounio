//! Async Function Transformation for HIR
//!
//! This module transforms async functions into state machines.
//! Each await point becomes a state transition, allowing the function
//! to be suspended and resumed.
//!
//! # Transformation Overview
//!
//! An async function like:
//! ```d
//! async fn fetch_and_process(url: string) -> string with Async {
//!     let data = http_get(url).await
//!     let processed = process(data)
//!     processed
//! }
//! ```
//!
//! Is transformed into a state machine:
//! ```text
//! State 0 (Start):
//!     - Capture locals: url
//!     - Call http_get(url)
//!     - Transition to State 1 (awaiting result)
//!
//! State 1 (After first await):
//!     - Resume with data = awaited_value
//!     - Call process(data)
//!     - Store result in processed
//!     - Transition to Completed
//! ```

use crate::common::NodeId;
use crate::hir::*;
use std::collections::HashMap;

/// A transformed async function as a state machine
#[derive(Debug, Clone)]
pub struct AsyncStateMachine {
    /// The original function name
    pub name: String,
    /// Parameters from the original function
    pub params: Vec<HirParam>,
    /// Return type of the future's output
    pub output_type: HirType,
    /// The states in the state machine
    pub states: Vec<AsyncStateNode>,
    /// Local variables that need to be captured across await points
    pub captured_locals: Vec<CapturedLocal>,
    /// Effects from the original function
    pub effects: Vec<HirEffect>,
}

/// A single state in the async state machine
#[derive(Debug, Clone)]
pub struct AsyncStateNode {
    /// State index
    pub index: u32,
    /// The kind of state
    pub kind: AsyncStateKind,
    /// Statements to execute in this state (before any transition)
    pub stmts: Vec<HirStmt>,
    /// The transition to take after executing statements
    pub transition: StateTransition,
}

/// The kind of async state
#[derive(Debug, Clone)]
pub enum AsyncStateKind {
    /// Initial entry state
    Start,
    /// Resumed after an await point
    ResumePoint {
        /// The local variable to store the awaited result
        result_binding: Option<String>,
    },
    /// Final state (function has completed)
    Completed,
}

/// A transition between states
#[derive(Debug, Clone)]
pub enum StateTransition {
    /// Move directly to the next state
    Goto(u32),
    /// Await an expression and transition to the next state when ready
    Await {
        /// The expression being awaited
        future_expr: HirExpr,
        /// The state to transition to when the await completes
        resume_state: u32,
    },
    /// Return with a value (completes the async function)
    Return(HirExpr),
    /// Conditional transition
    Branch {
        condition: HirExpr,
        if_true: u32,
        if_false: u32,
    },
    /// No transition (terminal state)
    Terminal,
}

/// A local variable that needs to be captured across await points
#[derive(Debug, Clone)]
pub struct CapturedLocal {
    /// Variable name
    pub name: String,
    /// Variable type
    pub ty: HirType,
    /// Whether the variable is mutable
    pub is_mut: bool,
    /// The state where this variable is first defined
    pub defined_in_state: u32,
    /// States where this variable is used
    pub used_in_states: Vec<u32>,
}

/// Information about a single extracted await
#[derive(Debug, Clone)]
pub struct AwaitExtraction {
    /// The future expression being awaited
    pub future: HirExpr,
    /// Temporary variable name to store the result
    pub temp_name: String,
    /// The type of the awaited result
    pub result_type: HirType,
    /// The evaluation context where this await appears
    pub context: EvalContext,
}

/// Represents the position of an await in an expression tree
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalContext {
    /// Await is at the root of the expression
    Root,
    /// Await is a function call argument (index)
    CallArg(usize),
    /// Await is the function being called
    CallFunc,
    /// Await is the left operand of a binary operation
    BinaryLeft,
    /// Await is the right operand of a binary operation
    BinaryRight,
    /// Await is inside a unary operation
    UnaryOperand,
    /// Await is the receiver of a method call
    MethodReceiver,
    /// Await is a method call argument (index)
    MethodArg(usize),
    /// Await is inside a block statement
    BlockStmt,
    /// Await is in the condition of an if expression
    IfCondition,
    /// Await is in the then branch of an if expression
    IfThen,
    /// Await is in the else branch of an if expression
    IfElse,
}

/// Where a binding result should be stored
#[derive(Debug, Clone)]
pub enum BindingTarget {
    /// Bind to a new let statement with the given name
    Let { name: String, is_mut: bool },
    /// Bind to an assignment target
    Assign(String),
    /// No binding, just execute for side effects
    None,
}

/// Result of extracting all awaits from an expression
#[derive(Debug, Clone)]
pub struct ExtractedAwaits {
    /// The expression with awaits replaced by temporaries
    pub expr: HirExpr,
    /// All extracted awaits in bottom-up order
    pub awaits: Vec<AwaitExtraction>,
}

impl ExtractedAwaits {
    /// Create an ExtractedAwaits with no extractions
    pub fn none(expr: HirExpr) -> Self {
        Self {
            expr,
            awaits: Vec::new(),
        }
    }

    /// Check if any awaits were extracted
    pub fn has_awaits(&self) -> bool {
        !self.awaits.is_empty()
    }
}

/// Async function transformer
///
/// Transforms HIR async functions into state machines.
pub struct AsyncTransformer {
    /// Counter for generating state indices
    state_counter: u32,
    /// Counter for generating temporary variable names
    temp_counter: u32,
    /// The states being built
    states: Vec<AsyncStateNode>,
    /// Captured locals discovered during transformation
    captured_locals: HashMap<String, CapturedLocal>,
    /// Current state index
    current_state: u32,
    /// Stack of statements for the current state
    current_stmts: Vec<HirStmt>,
}

impl AsyncTransformer {
    /// Create a new async transformer
    pub fn new() -> Self {
        Self {
            state_counter: 0,
            temp_counter: 0,
            states: Vec::new(),
            captured_locals: HashMap::new(),
            current_state: 0,
            current_stmts: Vec::new(),
        }
    }

    /// Transform an async function into a state machine
    pub fn transform(&mut self, func: &HirFn) -> AsyncStateMachine {
        self.reset();

        // Create the initial start state
        let start_state = self.new_state(AsyncStateKind::Start);
        self.current_state = start_state;

        // Transform the function body
        let final_transition = self.transform_block(&func.body);

        // Finalize the current state with the final transition
        self.finalize_state(final_transition);

        // Build captured locals list
        let captured_locals: Vec<CapturedLocal> = self.captured_locals.values().cloned().collect();

        AsyncStateMachine {
            name: func.name.clone(),
            params: func.ty.params.clone(),
            output_type: (*func.ty.return_type).clone(),
            states: std::mem::take(&mut self.states),
            captured_locals,
            effects: func.ty.effects.clone(),
        }
    }

    /// Reset the transformer for a new function
    fn reset(&mut self) {
        self.state_counter = 0;
        self.temp_counter = 0;
        self.states.clear();
        self.captured_locals.clear();
        self.current_state = 0;
        self.current_stmts.clear();
    }

    /// Create a new state and return its index
    fn new_state(&mut self, kind: AsyncStateKind) -> u32 {
        let index = self.state_counter;
        self.state_counter += 1;

        self.states.push(AsyncStateNode {
            index,
            kind,
            stmts: Vec::new(),
            transition: StateTransition::Terminal,
        });

        index
    }

    /// Finalize the current state with a transition
    fn finalize_state(&mut self, transition: StateTransition) {
        let state = &mut self.states[self.current_state as usize];
        state.stmts = std::mem::take(&mut self.current_stmts);
        state.transition = transition;
    }

    /// Transform a block of statements
    fn transform_block(&mut self, block: &HirBlock) -> StateTransition {
        for (i, stmt) in block.stmts.iter().enumerate() {
            match stmt {
                HirStmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    layout_hint,
                } => {
                    if let Some(expr) = value {
                        // Extract all awaits from the value expression
                        let extracted = self.extract_awaits(expr);

                        if extracted.has_awaits() {
                            // Build state sequence for awaits and bind result
                            self.build_await_sequence(
                                &extracted,
                                &BindingTarget::Let {
                                    name: name.clone(),
                                    is_mut: *is_mut,
                                },
                            );
                        } else {
                            // No awaits - just add the statement normally
                            self.current_stmts.push(HirStmt::Let {
                                name: name.clone(),
                                ty: ty.clone(),
                                value: Some(extracted.expr),
                                is_mut: *is_mut,
                                layout_hint: layout_hint.clone(),
                            });
                        }
                    } else {
                        // No value, just add as-is
                        self.current_stmts.push(stmt.clone());
                    }

                    // Track as captured local
                    self.captured_locals.insert(
                        name.clone(),
                        CapturedLocal {
                            name: name.clone(),
                            ty: ty.clone(),
                            is_mut: *is_mut,
                            defined_in_state: self.current_state,
                            used_in_states: Vec::new(),
                        },
                    );
                }

                HirStmt::Expr(expr) => {
                    // Handle return expressions specially
                    if let HirExprKind::Return(ret_val) = &expr.kind {
                        let return_expr = ret_val.as_ref().map_or_else(
                            || HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Literal(HirLiteral::Unit),
                                ty: HirType::Unit,
                            },
                            |e| (**e).clone(),
                        );

                        // Check if return value has awaits
                        let extracted = self.extract_awaits(&return_expr);
                        if extracted.has_awaits() {
                            // Handle awaits in return value - create states for them
                            // but the final transition should be Return instead of just an expr
                            for (i, await_info) in extracted.awaits.iter().enumerate() {
                                let next_state = self.new_state(AsyncStateKind::ResumePoint {
                                    result_binding: Some(await_info.temp_name.clone()),
                                });

                                self.finalize_state(StateTransition::Await {
                                    future_expr: await_info.future.clone(),
                                    resume_state: next_state,
                                });

                                self.current_state = next_state;
                            }
                            return StateTransition::Return(extracted.expr);
                        } else {
                            return StateTransition::Return(extracted.expr);
                        }
                    }

                    // Regular expression statement
                    let extracted = self.extract_awaits(expr);

                    if extracted.has_awaits() {
                        // Build state sequence for awaits without binding
                        self.build_await_sequence(&extracted, &BindingTarget::None);
                    } else {
                        // No awaits, just add expression
                        self.current_stmts.push(HirStmt::Expr(extracted.expr));
                    }
                }

                HirStmt::Assign { target, value } => {
                    // Extract awaits from the value
                    let extracted = self.extract_awaits(value);

                    if extracted.has_awaits() {
                        let target_name = self.extract_assign_target(target);
                        let binding_target = if let Some(name) = target_name {
                            BindingTarget::Assign(name)
                        } else {
                            BindingTarget::None
                        };

                        self.build_await_sequence(&extracted, &binding_target);
                    } else {
                        // No awaits, just add the statement
                        self.current_stmts.push(HirStmt::Assign {
                            target: target.clone(),
                            value: extracted.expr,
                        });
                    }
                }
            }
        }

        // If we reach here without an explicit return, return unit
        StateTransition::Return(HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Literal(HirLiteral::Unit),
            ty: block.ty.clone(),
        })
    }

    /// Check if an expression contains an await, and if so return the awaited expression
    fn check_for_await(&self, expr: &HirExpr) -> Option<HirExpr> {
        match &expr.kind {
            // Direct await - return the inner future expression
            HirExprKind::MethodCall {
                receiver,
                method,
                args: _,
            } if method == "await" => Some((**receiver).clone()),

            // Recursively check binary operations
            HirExprKind::Binary { left, right, .. } => self
                .check_for_await(left)
                .or_else(|| self.check_for_await(right)),

            // Recursively check unary operations
            HirExprKind::Unary { expr: inner, .. } => self.check_for_await(inner),

            // Recursively check call arguments
            HirExprKind::Call { func, args } => {
                if let Some(await_expr) = self.check_for_await(func) {
                    return Some(await_expr);
                }
                for arg in args {
                    if let Some(await_expr) = self.check_for_await(arg) {
                        return Some(await_expr);
                    }
                }
                None
            }

            // Method call (could be .await)
            HirExprKind::MethodCall {
                receiver,
                method: _,
                args,
            } => {
                if let Some(await_expr) = self.check_for_await(receiver) {
                    return Some(await_expr);
                }
                for arg in args {
                    if let Some(await_expr) = self.check_for_await(arg) {
                        return Some(await_expr);
                    }
                }
                None
            }

            // Block expressions
            HirExprKind::Block(block) => {
                for stmt in &block.stmts {
                    match stmt {
                        HirStmt::Expr(e) => {
                            if let Some(await_expr) = self.check_for_await(e) {
                                return Some(await_expr);
                            }
                        }
                        HirStmt::Let { value: Some(e), .. } => {
                            if let Some(await_expr) = self.check_for_await(e) {
                                return Some(await_expr);
                            }
                        }
                        HirStmt::Assign { value, .. } => {
                            if let Some(await_expr) = self.check_for_await(value) {
                                return Some(await_expr);
                            }
                        }
                        _ => {}
                    }
                }
                None
            }

            // If expressions
            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                if let Some(await_expr) = self.check_for_await(condition) {
                    return Some(await_expr);
                }
                for stmt in &then_branch.stmts {
                    if let HirStmt::Expr(e) = stmt {
                        if let Some(await_expr) = self.check_for_await(e) {
                            return Some(await_expr);
                        }
                    }
                }
                if let Some(else_expr) = else_branch {
                    if let Some(await_expr) = self.check_for_await(else_expr) {
                        return Some(await_expr);
                    }
                }
                None
            }

            // Other expressions don't contain awaits at the top level
            _ => None,
        }
    }

    /// Extract the target variable name from an assignment target
    fn extract_assign_target(&self, target: &HirExpr) -> Option<String> {
        match &target.kind {
            HirExprKind::Local(name) => Some(name.clone()),
            _ => None,
        }
    }

    /// Generate a fresh temporary variable name
    fn gen_temp(&mut self) -> String {
        let name = format!("__await_tmp_{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    /// Extract all awaits from an expression, replacing them with temporaries
    /// Returns the modified expression and a list of extracted awaits
    fn extract_awaits(&mut self, expr: &HirExpr) -> ExtractedAwaits {
        let mut awaits = Vec::new();
        let transformed = self.extract_awaits_recursive(expr, &mut awaits);
        ExtractedAwaits {
            expr: transformed,
            awaits,
        }
    }

    /// Recursively extract awaits from an expression
    /// Collects extracted awaits in the provided vector in post-order (bottom-up)
    fn extract_awaits_recursive(&mut self, expr: &HirExpr, awaits: &mut Vec<AwaitExtraction>) -> HirExpr {
        match &expr.kind {
            // Direct await - extract it
            HirExprKind::MethodCall {
                receiver,
                method,
                args: _,
            } if method == "await" => {
                let temp_name = self.gen_temp();
                let result_type = expr.ty.clone();

                awaits.push(AwaitExtraction {
                    future: (**receiver).clone(),
                    temp_name: temp_name.clone(),
                    result_type,
                    context: EvalContext::Root,
                });

                // Replace with a local reference to the temporary
                HirExpr {
                    id: expr.id,
                    kind: HirExprKind::Local(temp_name),
                    ty: expr.ty.clone(),
                }
            }

            // Binary operation - check both sides and rebuild
            HirExprKind::Binary { left, op, right } => {
                let left_awaits_before = awaits.len();
                let new_left = self.extract_awaits_recursive(left, awaits);
                let left_had_awaits = awaits.len() > left_awaits_before;

                let right_awaits_before = awaits.len();
                let new_right = self.extract_awaits_recursive(right, awaits);
                let right_had_awaits = awaits.len() > right_awaits_before;

                // Update context information for extracted awaits
                if left_had_awaits {
                    for await_info in &mut awaits[left_awaits_before..] {
                        if await_info.context == EvalContext::Root {
                            await_info.context = EvalContext::BinaryLeft;
                        }
                    }
                }
                if right_had_awaits {
                    for await_info in &mut awaits[right_awaits_before..] {
                        if await_info.context == EvalContext::Root {
                            await_info.context = EvalContext::BinaryRight;
                        }
                    }
                }

                HirExpr {
                    id: expr.id,
                    kind: HirExprKind::Binary {
                        left: Box::new(new_left),
                        op: op.clone(),
                        right: Box::new(new_right),
                    },
                    ty: expr.ty.clone(),
                }
            }

            // Unary operation
            HirExprKind::Unary { op, expr: inner } => {
                let new_inner = self.extract_awaits_recursive(inner, awaits);

                // Update context for extracted awaits
                if !awaits.is_empty() {
                    if let Some(last_await) = awaits.last_mut() {
                        if last_await.context == EvalContext::Root {
                            last_await.context = EvalContext::UnaryOperand;
                        }
                    }
                }

                HirExpr {
                    id: expr.id,
                    kind: HirExprKind::Unary {
                        op: op.clone(),
                        expr: Box::new(new_inner),
                    },
                    ty: expr.ty.clone(),
                }
            }

            // Function call - check function and all arguments
            HirExprKind::Call { func, args } => {
                let new_func = self.extract_awaits_recursive(func, awaits);

                let mut new_args = Vec::new();
                for (i, arg) in args.iter().enumerate() {
                    let arg_awaits_before = awaits.len();
                    let new_arg = self.extract_awaits_recursive(arg, awaits);

                    // Update context for this argument's awaits
                    for await_info in &mut awaits[arg_awaits_before..] {
                        if await_info.context == EvalContext::Root {
                            await_info.context = EvalContext::CallArg(i);
                        }
                    }

                    new_args.push(new_arg);
                }

                HirExpr {
                    id: expr.id,
                    kind: HirExprKind::Call {
                        func: Box::new(new_func),
                        args: new_args,
                    },
                    ty: expr.ty.clone(),
                }
            }

            // Method call - check receiver and arguments
            HirExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let new_receiver = self.extract_awaits_recursive(receiver, awaits);

                let mut new_args = Vec::new();
                for (i, arg) in args.iter().enumerate() {
                    let arg_awaits_before = awaits.len();
                    let new_arg = self.extract_awaits_recursive(arg, awaits);

                    // Update context for this argument's awaits
                    for await_info in &mut awaits[arg_awaits_before..] {
                        if await_info.context == EvalContext::Root {
                            await_info.context = EvalContext::MethodArg(i);
                        }
                    }

                    new_args.push(new_arg);
                }

                HirExpr {
                    id: expr.id,
                    kind: HirExprKind::MethodCall {
                        receiver: Box::new(new_receiver),
                        method: method.clone(),
                        args: new_args,
                    },
                    ty: expr.ty.clone(),
                }
            }

            // Block expressions - check all statements
            HirExprKind::Block(block) => {
                let mut new_stmts = Vec::new();
                for stmt in &block.stmts {
                    match stmt {
                        HirStmt::Let {
                            name,
                            ty,
                            value: Some(val),
                            is_mut,
                            layout_hint,
                        } => {
                            let new_val = self.extract_awaits_recursive(val, awaits);
                            new_stmts.push(HirStmt::Let {
                                name: name.clone(),
                                ty: ty.clone(),
                                value: Some(new_val),
                                is_mut: *is_mut,
                                layout_hint: layout_hint.clone(),
                            });
                        }
                        HirStmt::Assign { target, value } => {
                            let new_val = self.extract_awaits_recursive(value, awaits);
                            new_stmts.push(HirStmt::Assign {
                                target: target.clone(),
                                value: new_val,
                            });
                        }
                        HirStmt::Expr(e) => {
                            let new_expr = self.extract_awaits_recursive(e, awaits);
                            new_stmts.push(HirStmt::Expr(new_expr));
                        }
                        _ => {
                            new_stmts.push(stmt.clone());
                        }
                    }
                }

                HirExpr {
                    id: expr.id,
                    kind: HirExprKind::Block(HirBlock {
                        stmts: new_stmts,
                        ty: block.ty.clone(),
                    }),
                    ty: expr.ty.clone(),
                }
            }

            // If expressions - check condition and branches
            HirExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let new_condition = self.extract_awaits_recursive(condition, awaits);

                let mut new_then_stmts = Vec::new();
                for stmt in &then_branch.stmts {
                    match stmt {
                        HirStmt::Expr(e) => {
                            let new_expr = self.extract_awaits_recursive(e, awaits);
                            new_then_stmts.push(HirStmt::Expr(new_expr));
                        }
                        _ => {
                            new_then_stmts.push(stmt.clone());
                        }
                    }
                }

                let new_else = else_branch.as_ref().map(|e| {
                    Box::new(self.extract_awaits_recursive(e, awaits))
                });

                HirExpr {
                    id: expr.id,
                    kind: HirExprKind::If {
                        condition: Box::new(new_condition),
                        then_branch: HirBlock {
                            stmts: new_then_stmts,
                            ty: then_branch.ty.clone(),
                        },
                        else_branch: new_else,
                    },
                    ty: expr.ty.clone(),
                }
            }

            // Other expressions - return as-is
            _ => expr.clone(),
        }
    }

    /// Build a sequence of states for multiple extracted awaits
    /// Returns the next state to continue from after all awaits are processed
    fn build_await_sequence(
        &mut self,
        extracted: &ExtractedAwaits,
        binding: &BindingTarget,
    ) -> u32 {
        if extracted.awaits.is_empty() {
            // No awaits to process - just add the expression as a statement
            match binding {
                BindingTarget::Let { name, is_mut } => {
                    self.current_stmts.push(HirStmt::Let {
                        name: name.clone(),
                        ty: extracted.expr.ty.clone(),
                        value: Some(extracted.expr.clone()),
                        is_mut: *is_mut,
                        layout_hint: None,
                    });
                }
                BindingTarget::Assign(target) => {
                    self.current_stmts.push(HirStmt::Assign {
                        target: HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::Local(target.clone()),
                            ty: extracted.expr.ty.clone(),
                        },
                        value: extracted.expr.clone(),
                    });
                }
                BindingTarget::None => {
                    self.current_stmts.push(HirStmt::Expr(extracted.expr.clone()));
                }
            }
            return self.current_state;
        }

        // Process awaits in order
        for (i, await_info) in extracted.awaits.iter().enumerate() {
            let is_last = i == extracted.awaits.len() - 1;

            if i == 0 {
                // First await - finalize current state with await transition
                let next_state = self.new_state(AsyncStateKind::ResumePoint {
                    result_binding: Some(await_info.temp_name.clone()),
                });

                self.finalize_state(StateTransition::Await {
                    future_expr: await_info.future.clone(),
                    resume_state: next_state,
                });

                self.current_state = next_state;
            } else {
                // Subsequent awaits - create transition from previous resume point
                let next_state = self.new_state(AsyncStateKind::ResumePoint {
                    result_binding: Some(await_info.temp_name.clone()),
                });

                self.finalize_state(StateTransition::Await {
                    future_expr: await_info.future.clone(),
                    resume_state: next_state,
                });

                self.current_state = next_state;
            }
        }

        // After all awaits, add the final expression with binding
        match binding {
            BindingTarget::Let { name, is_mut } => {
                self.current_stmts.push(HirStmt::Let {
                    name: name.clone(),
                    ty: extracted.expr.ty.clone(),
                    value: Some(extracted.expr.clone()),
                    is_mut: *is_mut,
                    layout_hint: None,
                });
            }
            BindingTarget::Assign(target) => {
                self.current_stmts.push(HirStmt::Assign {
                    target: HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Local(target.clone()),
                        ty: extracted.expr.ty.clone(),
                    },
                    value: extracted.expr.clone(),
                });
            }
            BindingTarget::None => {
                self.current_stmts.push(HirStmt::Expr(extracted.expr.clone()));
            }
        }

        self.current_state
    }
}

impl Default for AsyncTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Transform all async functions in an HIR module
pub fn transform_async_functions(hir: &Hir) -> Vec<AsyncStateMachine> {
    let mut transformer = AsyncTransformer::new();
    let mut machines = Vec::new();

    for item in &hir.items {
        if let HirItem::Function(func) = item {
            // Check if this is an async function by looking for Async effect
            let is_async = func.ty.effects.iter().any(|e| e.name == "Async");
            if is_async {
                machines.push(transformer.transform(func));
            }
        }
    }

    machines
}

/// Check if an HIR function is async
pub fn is_async_function(func: &HirFn) -> bool {
    func.ty.effects.iter().any(|e| e.name == "Async")
}

/// Add await expression kind to HIR (for use in parser/lowering)
/// This represents expr.await in the surface syntax
#[derive(Debug, Clone)]
pub struct AwaitExpr {
    pub id: NodeId,
    pub future: Box<HirExpr>,
    pub ty: HirType,
}

impl AwaitExpr {
    pub fn new(id: NodeId, future: HirExpr, output_ty: HirType) -> Self {
        Self {
            id,
            future: Box::new(future),
            ty: output_ty,
        }
    }

    /// Convert to a HirExpr (method call style)
    pub fn to_hir_expr(self) -> HirExpr {
        HirExpr {
            id: self.id,
            kind: HirExprKind::MethodCall {
                receiver: self.future,
                method: "await".to_string(),
                args: Vec::new(),
            },
            ty: self.ty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Abi;

    fn make_test_async_fn() -> HirFn {
        HirFn {
            id: NodeId::dummy(),
            name: "test_async".to_string(),
            ty: HirFnType {
                params: vec![HirParam {
                    id: NodeId::dummy(),
                    name: "url".to_string(),
                    ty: HirType::String,
                    is_mut: false,
                }],
                return_type: Box::new(HirType::String),
                effects: vec![HirEffect {
                    id: NodeId::dummy(),
                    name: "Async".to_string(),
                    operations: Vec::new(),
                    effect_var: None,
                }],
            },
            body: HirBlock {
                stmts: vec![
                    HirStmt::Let {
                        name: "data".to_string(),
                        ty: HirType::String,
                        value: Some(HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::MethodCall {
                                receiver: Box::new(HirExpr {
                                    id: NodeId::dummy(),
                                    kind: HirExprKind::Call {
                                        func: Box::new(HirExpr {
                                            id: NodeId::dummy(),
                                            kind: HirExprKind::Local("http_get".to_string()),
                                            ty: HirType::Fn {
                                                params: vec![HirType::String],
                                                return_type: Box::new(HirType::String),
                                            },
                                        }),
                                        args: vec![HirExpr {
                                            id: NodeId::dummy(),
                                            kind: HirExprKind::Local("url".to_string()),
                                            ty: HirType::String,
                                        }],
                                    },
                                    ty: HirType::Named {
                                        name: "Future".to_string(),
                                        args: vec![HirType::String],
                                    },
                                }),
                                method: "await".to_string(),
                                args: Vec::new(),
                            },
                            ty: HirType::String,
                        }),
                        is_mut: false,
                        layout_hint: None,
                    },
                    HirStmt::Expr(HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Return(Some(Box::new(HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::Local("data".to_string()),
                            ty: HirType::String,
                        }))),
                        ty: HirType::Never,
                    }),
                ],
                ty: HirType::String,
            },
            abi: Abi::Rust,
            is_exported: false,
            extern_name: None,
        }
    }

    #[test]
    fn test_async_transform_basic() {
        let func = make_test_async_fn();
        let mut transformer = AsyncTransformer::new();
        let machine = transformer.transform(&func);

        assert_eq!(machine.name, "test_async");
        assert_eq!(machine.output_type, HirType::String);
        // Should have at least 2 states: start and resume after await
        assert!(machine.states.len() >= 2);
    }

    #[test]
    fn test_is_async_function() {
        let func = make_test_async_fn();
        assert!(is_async_function(&func));

        // Non-async function
        let sync_func = HirFn {
            id: NodeId::dummy(),
            name: "sync_fn".to_string(),
            ty: HirFnType {
                params: Vec::new(),
                return_type: Box::new(HirType::Unit),
                effects: Vec::new(),
            },
            body: HirBlock {
                stmts: Vec::new(),
                ty: HirType::Unit,
            },
            abi: Abi::Rust,
            is_exported: false,
            extern_name: None,
        };
        assert!(!is_async_function(&sync_func));
    }

    #[test]
    fn test_extract_simple_await() {
        let mut transformer = AsyncTransformer::new();

        // Create a simple await expression: foo().await
        let foo_call = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Call {
                func: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("foo".to_string()),
                    ty: HirType::String,
                }),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let await_expr = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::MethodCall {
                receiver: Box::new(foo_call.clone()),
                method: "await".to_string(),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let extracted = transformer.extract_awaits(&await_expr);

        // Should have extracted one await
        assert_eq!(extracted.awaits.len(), 1);
        assert_eq!(extracted.awaits[0].context, EvalContext::Root);

        // The result should be a reference to a temporary
        match &extracted.expr.kind {
            HirExprKind::Local(name) => {
                assert!(name.starts_with("__await_tmp_"));
            }
            _ => panic!("Expected local reference to temporary"),
        }
    }

    #[test]
    fn test_extract_nested_awaits_in_call() {
        let mut transformer = AsyncTransformer::new();

        // Create: foo(bar().await)
        let bar_call = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Call {
                func: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("bar".to_string()),
                    ty: HirType::String,
                }),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let await_bar = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::MethodCall {
                receiver: Box::new(bar_call),
                method: "await".to_string(),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let foo_call = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Call {
                func: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("foo".to_string()),
                    ty: HirType::String,
                }),
                args: vec![await_bar],
            },
            ty: HirType::String,
        };

        let extracted = transformer.extract_awaits(&foo_call);

        // Should have extracted one await
        assert_eq!(extracted.awaits.len(), 1);
        assert_eq!(extracted.awaits[0].context, EvalContext::CallArg(0));

        // The result expression should still be foo(temp)
        match &extracted.expr.kind {
            HirExprKind::Call { func: _, args } => {
                assert_eq!(args.len(), 1);
                match &args[0].kind {
                    HirExprKind::Local(name) => {
                        assert!(name.starts_with("__await_tmp_"));
                    }
                    _ => panic!("Expected local reference to temporary in argument"),
                }
            }
            _ => panic!("Expected call expression"),
        }
    }

    #[test]
    fn test_extract_multiple_awaits() {
        let mut transformer = AsyncTransformer::new();

        // Create: foo(bar().await, baz().await)
        let bar_call = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Call {
                func: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("bar".to_string()),
                    ty: HirType::String,
                }),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let await_bar = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::MethodCall {
                receiver: Box::new(bar_call),
                method: "await".to_string(),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let baz_call = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Call {
                func: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("baz".to_string()),
                    ty: HirType::String,
                }),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let await_baz = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::MethodCall {
                receiver: Box::new(baz_call),
                method: "await".to_string(),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let foo_call = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Call {
                func: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("foo".to_string()),
                    ty: HirType::String,
                }),
                args: vec![await_bar, await_baz],
            },
            ty: HirType::String,
        };

        let extracted = transformer.extract_awaits(&foo_call);

        // Should have extracted two awaits
        assert_eq!(extracted.awaits.len(), 2);
        assert_eq!(extracted.awaits[0].context, EvalContext::CallArg(0));
        assert_eq!(extracted.awaits[1].context, EvalContext::CallArg(1));

        // The result should be foo(temp0, temp1)
        match &extracted.expr.kind {
            HirExprKind::Call { func: _, args } => {
                assert_eq!(args.len(), 2);
                for i in 0..2 {
                    match &args[i].kind {
                        HirExprKind::Local(name) => {
                            assert!(name.starts_with("__await_tmp_"));
                        }
                        _ => panic!("Expected local reference to temporary in argument {}", i),
                    }
                }
            }
            _ => panic!("Expected call expression"),
        }
    }

    #[test]
    fn test_eval_context_tracking() {
        let mut transformer = AsyncTransformer::new();

        // Create: a + b.await
        let b_call = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Call {
                func: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("b".to_string()),
                    ty: HirType::String,
                }),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let await_b = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::MethodCall {
                receiver: Box::new(b_call),
                method: "await".to_string(),
                args: Vec::new(),
            },
            ty: HirType::String,
        };

        let binary_expr = HirExpr {
            id: NodeId::dummy(),
            kind: HirExprKind::Binary {
                left: Box::new(HirExpr {
                    id: NodeId::dummy(),
                    kind: HirExprKind::Local("a".to_string()),
                    ty: HirType::String,
                }),
                op: HirBinaryOp::Add,
                right: Box::new(await_b),
            },
            ty: HirType::String,
        };

        let extracted = transformer.extract_awaits(&binary_expr);

        // Should have one await with BinaryRight context
        assert_eq!(extracted.awaits.len(), 1);
        assert_eq!(extracted.awaits[0].context, EvalContext::BinaryRight);
    }
}
