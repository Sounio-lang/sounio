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
//! ```
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

/// Async function transformer
///
/// Transforms HIR async functions into state machines.
pub struct AsyncTransformer {
    /// Counter for generating state indices
    state_counter: u32,
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
            let is_last = i == block.stmts.len() - 1;

            match stmt {
                HirStmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    layout_hint,
                } => {
                    if let Some(expr) = value {
                        // Check if the value expression contains an await
                        if let Some(await_transition) = self.check_for_await(expr) {
                            // We need to split here
                            // First, finalize the current state with await transition
                            let next_state = self.new_state(AsyncStateKind::ResumePoint {
                                result_binding: Some(name.clone()),
                            });

                            // Track this as a captured local
                            self.captured_locals.insert(
                                name.clone(),
                                CapturedLocal {
                                    name: name.clone(),
                                    ty: ty.clone(),
                                    is_mut: *is_mut,
                                    defined_in_state: self.current_state,
                                    used_in_states: vec![next_state],
                                },
                            );

                            self.finalize_state(StateTransition::Await {
                                future_expr: await_transition,
                                resume_state: next_state,
                            });

                            // Continue from the new state
                            self.current_state = next_state;
                            continue;
                        }
                    }

                    // No await - just add the statement normally
                    self.current_stmts.push(stmt.clone());

                    // Track as captured if might be used across await
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
                    // Check for await in the expression
                    if let Some(await_expr) = self.check_for_await(expr) {
                        let next_state = self.new_state(AsyncStateKind::ResumePoint {
                            result_binding: None,
                        });

                        self.finalize_state(StateTransition::Await {
                            future_expr: await_expr,
                            resume_state: next_state,
                        });

                        self.current_state = next_state;
                        continue;
                    }

                    // Handle return expressions
                    if let HirExprKind::Return(ret_val) = &expr.kind {
                        let return_expr = ret_val.as_ref().map_or_else(
                            || HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Literal(HirLiteral::Unit),
                                ty: HirType::Unit,
                            },
                            |e| (**e).clone(),
                        );
                        return StateTransition::Return(return_expr);
                    }

                    // Normal expression statement
                    self.current_stmts.push(stmt.clone());
                }

                HirStmt::Assign { target, value } => {
                    // Check for await in the value
                    if let Some(await_expr) = self.check_for_await(value) {
                        // Need to handle this specially - store the target info
                        let next_state = self.new_state(AsyncStateKind::ResumePoint {
                            result_binding: self.extract_assign_target(target),
                        });

                        self.finalize_state(StateTransition::Await {
                            future_expr: await_expr,
                            resume_state: next_state,
                        });

                        self.current_state = next_state;
                        continue;
                    }

                    self.current_stmts.push(stmt.clone());
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
        };
        assert!(!is_async_function(&sync_func));
    }
}
