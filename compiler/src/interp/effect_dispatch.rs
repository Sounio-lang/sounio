//! Effect handler dispatch for interpreter
//!
//! Implements algebraic effect semantics using a handler stack.
//! This module provides the runtime mechanism for dispatching effect
//! operations (Prob, Causal, IO, etc.) to their registered handlers.
//!
//! # Algebraic Effects Overview
//!
//! Effects in Sounio are first-class: operations like `sample`, `observe`,
//! `do` are effect operations that suspend computation and delegate to
//! a handler. Handlers define how to interpret these operations.
//!
//! # Example
//!
//! ```text
//! // Sounio code:
//! fn coin_model() with Prob {
//!     let p = sample(Beta(1.0, 1.0));  // Prob effect operation
//!     observe(Bernoulli(p), true);      // Prob effect operation
//!     p
//! }
//!
//! // Run with MH handler:
//! handle coin_model() with mh_handler { ... }
//! ```

use std::collections::HashMap;
use std::fmt;

use thiserror::Error;

use super::value::{Distribution, SuspensionId, Value};
use crate::effects::continuation::{
    CapturedContinuation, ContinuationError, ContinuationId, ContinuationStore, ResumePoint,
};

/// Result of dispatching an effect operation with suspension support
#[derive(Debug)]
pub enum DispatchResult {
    /// The operation completed with a value
    Completed(Value),
    /// The operation suspended execution
    Suspended {
        /// The suspension ID from the handler
        suspension_id: SuspensionId,
        /// The continuation ID for resuming
        continuation_id: ContinuationId,
    },
}
use crate::effects::handler_capability::{
    Continuation as CapabilityContinuation, HandlerCapability, HandlerResult as CapabilityResult,
    HandlerState as CapabilityHandlerState,
};
use crate::effects::handlers::HandlerRegistry;
use crate::effects::EpistemicImpactRegistry;
use crate::runtime::causal;
use crate::runtime::prob::{self, ProbContext, Rng};

/// Errors that can occur during effect dispatch
#[derive(Error, Debug)]
pub enum EffectError {
    /// No handler found for the requested effect
    #[error("unhandled effect `{effect}`: no handler for operation `{operation}`")]
    UnhandledEffect { effect: String, operation: String },

    /// Handler returned an error
    #[error("handler error for `{effect}.{operation}`: {message}")]
    HandlerError {
        effect: String,
        operation: String,
        message: String,
    },

    /// Invalid arguments passed to effect operation
    #[error("invalid arguments for `{effect}.{operation}`: expected {expected}, got {got}")]
    InvalidArguments {
        effect: String,
        operation: String,
        expected: usize,
        got: usize,
    },

    /// Type mismatch in effect operation
    #[error("type mismatch in `{effect}.{operation}`: {message}")]
    TypeMismatch {
        effect: String,
        operation: String,
        message: String,
    },
}

/// Effect kinds supported by the interpreter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectKind {
    /// Probabilistic programming effect
    Prob,
    /// Causal inference effect
    Causal,
    /// Input/Output effect
    IO,
    /// Mutable state effect
    Mut,
    /// Memory allocation effect
    Alloc,
    /// Exception effect
    Exn,
    /// Async effect
    Async,
    /// GPU effect
    GPU,
    /// Epistemic (knowledge tracking) effect
    Epistemic,
    /// Division effect (may fail)
    Div,
    /// Panic/unrecoverable failure effect
    Panic,
    /// Network I/O effect
    Network,
    /// Sensor/measurement effect
    Sensor,
}

impl EffectKind {
    /// Parse effect kind from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Prob" => Some(EffectKind::Prob),
            "Causal" => Some(EffectKind::Causal),
            "IO" => Some(EffectKind::IO),
            "Mut" => Some(EffectKind::Mut),
            "Alloc" => Some(EffectKind::Alloc),
            "Exn" => Some(EffectKind::Exn),
            "Async" => Some(EffectKind::Async),
            "GPU" => Some(EffectKind::GPU),
            "Epistemic" => Some(EffectKind::Epistemic),
            "Div" => Some(EffectKind::Div),
            "Panic" => Some(EffectKind::Panic),
            "Network" => Some(EffectKind::Network),
            "Sensor" => Some(EffectKind::Sensor),
            _ => None,
        }
    }

    /// Get the string name of this effect
    pub fn as_str(&self) -> &'static str {
        match self {
            EffectKind::Prob => "Prob",
            EffectKind::Causal => "Causal",
            EffectKind::IO => "IO",
            EffectKind::Mut => "Mut",
            EffectKind::Alloc => "Alloc",
            EffectKind::Exn => "Exn",
            EffectKind::Async => "Async",
            EffectKind::GPU => "GPU",
            EffectKind::Epistemic => "Epistemic",
            EffectKind::Div => "Div",
            EffectKind::Panic => "Panic",
            EffectKind::Network => "Network",
            EffectKind::Sensor => "Sensor",
        }
    }
}

impl fmt::Display for EffectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single handler case for an effect operation
pub struct HandlerCase {
    /// Operation name (e.g., "sample", "observe")
    pub operation: String,
    /// Handler function: takes arguments, returns result
    pub handler_fn: Box<dyn Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError>>,
}

impl fmt::Debug for HandlerCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerCase")
            .field("operation", &self.operation)
            .finish()
    }
}

/// State shared across handler invocations
pub struct HandlerState {
    /// Random number generator for probabilistic operations
    pub rng: Rng,
    /// Probabilistic programming context
    pub prob_ctx: ProbContext,
    /// Causal model state (DAG name -> DAG)
    pub causal_models: HashMap<String, causal::DAG>,
    /// Intervention stack for causal operations
    pub interventions: Vec<(String, Value)>,
    /// Observation log for probabilistic conditioning
    pub observations: Vec<(String, Value)>,
    /// Custom state for user-defined handlers
    pub custom: HashMap<String, Value>,
}

impl HandlerState {
    /// Create new handler state with default seed
    pub fn new() -> Self {
        Self::with_seed(42)
    }

    /// Create new handler state with specified seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            prob_ctx: ProbContext::new(seed),
            causal_models: HashMap::new(),
            interventions: Vec::new(),
            observations: Vec::new(),
            custom: HashMap::new(),
        }
    }

    /// Reset state for a new execution
    pub fn reset(&mut self) {
        self.prob_ctx.reset();
        self.interventions.clear();
        self.observations.clear();
    }
}

impl Default for HandlerState {
    fn default() -> Self {
        Self::new()
    }
}

/// An effect handler with cases for each operation
pub struct EffectHandler {
    /// Which effect this handler handles
    pub effect: EffectKind,
    /// Handler cases for each operation
    pub cases: Vec<HandlerCase>,
    /// Handler name (for debugging/display)
    pub name: String,
}

impl EffectHandler {
    /// Create a new effect handler
    pub fn new(effect: EffectKind, name: &str) -> Self {
        Self {
            effect,
            cases: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Add a handler case
    pub fn with_case<F>(mut self, operation: &str, handler: F) -> Self
    where
        F: Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError> + 'static,
    {
        self.cases.push(HandlerCase {
            operation: operation.to_string(),
            handler_fn: Box::new(handler),
        });
        self
    }

    /// Find a handler case for an operation
    pub fn find_case(&self, operation: &str) -> Option<&HandlerCase> {
        self.cases.iter().find(|c| c.operation == operation)
    }
}

impl fmt::Debug for EffectHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectHandler")
            .field("effect", &self.effect)
            .field("name", &self.name)
            .field("cases", &self.cases.iter().map(|c| &c.operation).collect::<Vec<_>>())
            .finish()
    }
}

// =============================================================================
// HandlerCapability Adapter
// =============================================================================

/// Adapter to use HandlerCapability-based handlers with the existing EffectHandler system
///
/// This enables gradual migration from the old callback-based handlers to the new
/// trait-based HandlerCapability system. The new system supports:
/// - Continuations for proper algebraic effect semantics
/// - Epistemic impact tracking (Track B)
/// - Multi-shot continuations for effects like Amb
///
/// # Example
///
/// ```ignore
/// use sounio::effects::handler_capability::HandlerCapability;
///
/// #[derive(Debug)]
/// struct MyIOHandler;
///
/// impl HandlerCapability for MyIOHandler {
///     fn effect_name(&self) -> &str { "IO" }
///     fn handler_name(&self) -> &str { "MyIOHandler" }
///     fn operations(&self) -> &[OperationSpec] { &[] }
///     fn handle(&self, op: &str, args: &[Value], cont: Continuation, state: &mut HandlerState)
///         -> HandlerResult {
///         // Handle the operation
///         HandlerResult::Resume(Value::Unit)
///     }
/// }
///
/// // Convert to EffectHandler for use with EffectContext
/// let adapter = CapabilityAdapter::new(Box::new(MyIOHandler));
/// let effect_handler = adapter.to_effect_handler();
/// ctx.push_handler(effect_handler);
/// ```
pub struct CapabilityAdapter {
    /// The underlying HandlerCapability implementation
    handler: std::sync::Arc<dyn HandlerCapability + Send + Sync>,
    /// Epistemic impact registry for tracking confidence
    impact_registry: EpistemicImpactRegistry,
}

impl CapabilityAdapter {
    /// Create a new adapter from a HandlerCapability implementation
    pub fn new(handler: Box<dyn HandlerCapability + Send + Sync>) -> Self {
        Self {
            handler: std::sync::Arc::from(handler),
            impact_registry: EpistemicImpactRegistry::new(),
        }
    }

    /// Create adapter with custom impact registry
    pub fn with_registry(
        handler: Box<dyn HandlerCapability + Send + Sync>,
        registry: EpistemicImpactRegistry,
    ) -> Self {
        Self {
            handler: std::sync::Arc::from(handler),
            impact_registry: registry,
        }
    }

    /// Create a new adapter from an Arc<dyn HandlerCapability + Send + Sync>
    ///
    /// This is useful when the handler is already wrapped in an Arc,
    /// such as when retrieved from a HandlerRegistry.
    pub fn from_arc(handler: std::sync::Arc<dyn HandlerCapability + Send + Sync>) -> Self {
        Self {
            handler,
            impact_registry: EpistemicImpactRegistry::new(),
        }
    }

    /// Get the effect name this adapter handles
    pub fn effect_name(&self) -> &str {
        self.handler.effect_name()
    }

    /// Convert to an EffectHandler for use with EffectContext
    ///
    /// This creates an EffectHandler that wraps the HandlerCapability,
    /// translating between the two interfaces.
    pub fn to_effect_handler(&self) -> EffectHandler {
        let effect_kind = EffectKind::from_str(self.handler.effect_name())
            .unwrap_or(EffectKind::IO); // Default to IO if unknown

        let mut handler = EffectHandler::new(effect_kind, self.handler.handler_name());

        // Create a handler case for each operation
        for op_spec in self.handler.operations() {
            let op_name = op_spec.name.clone();
            let op_name_for_closure = op_name.clone();
            let capability_handler = self.handler.clone();

            handler = handler.with_case(&op_name, move |args: &[Value], _state: &mut HandlerState| {
                let op_name = op_name_for_closure.clone();
                // Create a placeholder continuation
                let continuation = CapabilityContinuation::new();

                // Create a capability handler state
                let mut cap_state = CapabilityHandlerState::new();

                // Call the capability handler
                let result = capability_handler.handle(&op_name, args, continuation, &mut cap_state);

                // Convert result back to our format
                match result {
                    CapabilityResult::Return(v) | CapabilityResult::Resume(v) => Ok(v),
                    CapabilityResult::Abort(err) => Err(EffectError::HandlerError {
                        effect: err.effect,
                        operation: err.operation,
                        message: err.message,
                    }),
                    CapabilityResult::Suspend(_) => Err(EffectError::HandlerError {
                        effect: capability_handler.effect_name().to_string(),
                        operation: op_name.clone(),
                        message: "Suspension not yet supported in adapter".to_string(),
                    }),
                }
            });
        }

        handler
    }

    /// Dispatch directly using the HandlerCapability interface
    ///
    /// This bypasses the EffectHandler conversion and calls the handler directly.
    /// Use this when you want full access to the continuation and epistemic features.
    pub fn dispatch_direct(
        &self,
        operation: &str,
        args: &[Value],
        continuation: CapabilityContinuation,
    ) -> Result<Value, EffectError> {
        let mut state = CapabilityHandlerState::new();

        let result = self.handler.handle(operation, args, continuation, &mut state);

        match result {
            CapabilityResult::Return(v) | CapabilityResult::Resume(v) => Ok(v),
            CapabilityResult::Abort(err) => Err(EffectError::HandlerError {
                effect: err.effect,
                operation: err.operation,
                message: err.message,
            }),
            CapabilityResult::Suspend(_) => Err(EffectError::HandlerError {
                effect: self.handler.effect_name().to_string(),
                operation: operation.to_string(),
                message: "Suspension not yet supported".to_string(),
            }),
        }
    }
}

impl fmt::Debug for CapabilityAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapabilityAdapter")
            .field("effect", &self.handler.effect_name())
            .field("handler", &self.handler.handler_name())
            .finish()
    }
}

/// Wrapper to make Arc<dyn HandlerCapability> implement HandlerCapability
#[derive(Debug)]
struct CloneableHandlerWrapper(std::sync::Arc<dyn HandlerCapability + Send + Sync>);

impl HandlerCapability for CloneableHandlerWrapper {
    fn effect_name(&self) -> &str {
        self.0.effect_name()
    }

    fn handler_name(&self) -> &str {
        self.0.handler_name()
    }

    fn operations(&self) -> &[crate::effects::handler_capability::OperationSpec] {
        self.0.operations()
    }

    fn handle(
        &self,
        operation: &str,
        args: &[Value],
        continuation: CapabilityContinuation,
        state: &mut CapabilityHandlerState,
    ) -> CapabilityResult {
        self.0.handle(operation, args, continuation, state)
    }

    fn epistemic_impact(
        &self,
        operation: &str,
    ) -> crate::effects::handler_capability::EpistemicImpact {
        self.0.epistemic_impact(operation)
    }

    fn supports_multi_shot(&self) -> bool {
        self.0.supports_multi_shot()
    }
}

/// Effect handler context managing a stack of handlers
pub struct EffectContext {
    /// Stack of active handlers (most recent on top)
    handler_stack: Vec<EffectHandler>,
    /// Shared state across handlers
    pub state: HandlerState,
    /// Store for captured continuations
    pub continuation_store: ContinuationStore,
    /// Registry of HandlerCapability-based handlers for fallback dispatch
    registry: Option<HandlerRegistry>,
    /// Capability handler state (separate from legacy HandlerState)
    capability_state: CapabilityHandlerState,
}

impl EffectContext {
    /// Create a new effect context with default handlers
    pub fn new() -> Self {
        let mut ctx = Self {
            handler_stack: Vec::new(),
            state: HandlerState::new(),
            continuation_store: ContinuationStore::new(),
            registry: None,
            capability_state: CapabilityHandlerState::new(),
        };

        // Install default handlers for Prob and Causal effects
        ctx.push_handler(default_prob_handler());
        ctx.push_handler(default_causal_handler());

        ctx
    }

    /// Create a new effect context with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        let mut ctx = Self {
            handler_stack: Vec::new(),
            state: HandlerState::with_seed(seed),
            continuation_store: ContinuationStore::new(),
            registry: None,
            capability_state: CapabilityHandlerState::new(),
        };

        ctx.push_handler(default_prob_handler());
        ctx.push_handler(default_causal_handler());

        ctx
    }

    /// Create a new effect context with a handler registry.
    ///
    /// The registry provides fallback handlers for effects not found in the
    /// handler stack. This is useful for dispatching to HandlerCapability-based
    /// handlers without converting them to EffectHandlers.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use sounio::effects::handlers::HandlerRegistry;
    ///
    /// let registry = HandlerRegistry::with_defaults();
    /// let mut ctx = EffectContext::with_registry(registry);
    ///
    /// // Can now dispatch to any of the 12 registered effects
    /// let result = ctx.dispatch_by_name("IO", "print", vec![Value::String("Hello".into())]);
    /// ```
    pub fn with_registry(registry: HandlerRegistry) -> Self {
        let mut ctx = Self {
            handler_stack: Vec::new(),
            state: HandlerState::new(),
            continuation_store: ContinuationStore::new(),
            registry: Some(registry),
            capability_state: CapabilityHandlerState::new(),
        };

        // Still install Prob and Causal for specialized handling
        ctx.push_handler(default_prob_handler());
        ctx.push_handler(default_causal_handler());

        ctx
    }

    /// Create a new effect context with all default handlers from the registry.
    ///
    /// This is the recommended way to create an EffectContext that supports
    /// all built-in effects (IO, Mut, Alloc, Panic, Prob, GPU, etc.).
    pub fn with_all_handlers() -> Self {
        Self::with_registry(HandlerRegistry::with_defaults())
    }

    /// Set the handler registry for fallback dispatch.
    pub fn set_registry(&mut self, registry: HandlerRegistry) {
        self.registry = Some(registry);
    }

    /// Get a reference to the handler registry, if set.
    pub fn registry(&self) -> Option<&HandlerRegistry> {
        self.registry.as_ref()
    }

    /// Install handlers from a registry into the handler stack.
    ///
    /// This converts HandlerCapability handlers to EffectHandlers and pushes
    /// them onto the stack. Useful when you want stack-based handler shadowing.
    pub fn install_from_registry(&mut self, registry: &HandlerRegistry) {
        for handler in registry.handlers() {
            let adapter = CapabilityAdapter::new(Box::new(CloneableHandlerWrapper(handler.clone())));
            self.push_handler(adapter.to_effect_handler());
        }
    }

    /// Push a handler onto the stack
    pub fn push_handler(&mut self, handler: EffectHandler) {
        self.handler_stack.push(handler);
    }

    /// Pop a handler from the stack
    pub fn pop_handler(&mut self) -> Option<EffectHandler> {
        self.handler_stack.pop()
    }

    /// Find a handler for the given effect and operation
    fn find_handler(&self, effect: EffectKind, operation: &str) -> Option<&HandlerCase> {
        // Search from top of stack (most recently pushed) to bottom
        for handler in self.handler_stack.iter().rev() {
            if handler.effect == effect {
                if let Some(case) = handler.find_case(operation) {
                    return Some(case);
                }
            }
        }
        None
    }

    /// Dispatch an effect operation to the appropriate handler.
    ///
    /// First searches the handler stack (most recent first), then falls back
    /// to the registry if set.
    pub fn dispatch(
        &mut self,
        effect: EffectKind,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<Value, EffectError> {
        // Try the handler stack first
        let handler_idx = self
            .handler_stack
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, handler)| {
                if handler.effect == effect && handler.find_case(operation).is_some() {
                    Some(idx)
                } else {
                    None
                }
            });

        if let Some(idx) = handler_idx {
            // Get pointer to handler function to avoid holding borrow on handler_stack
            // SAFETY: We know the index is valid and we don't modify handler_stack
            // during the call
            let handler_fn_ptr: *const dyn Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError> = {
                let handler = &self.handler_stack[idx];
                let case = handler.find_case(operation).unwrap();
                &*case.handler_fn as *const _
            };

            // Now call with mutable state - the handler_stack borrow is released
            // SAFETY: handler_fn_ptr is valid for the duration of this call
            return unsafe { (*handler_fn_ptr)(&args, &mut self.state) };
        }

        // Fall back to registry if available
        let effect_name = effect.as_str();
        self.dispatch_via_registry(effect_name, operation, args)
    }

    /// Dispatch using effect name as string.
    ///
    /// First searches the handler stack, then falls back to the registry.
    pub fn dispatch_by_name(
        &mut self,
        effect_name: &str,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<Value, EffectError> {
        // First try the handler stack if we can parse the effect kind
        if let Some(effect) = EffectKind::from_str(effect_name) {
            // Try the handler stack first
            let handler_idx = self
                .handler_stack
                .iter()
                .enumerate()
                .rev()
                .find_map(|(idx, handler)| {
                    if handler.effect == effect && handler.find_case(operation).is_some() {
                        Some(idx)
                    } else {
                        None
                    }
                });

            if let Some(idx) = handler_idx {
                let handler_fn_ptr: *const dyn Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError> = {
                    let handler = &self.handler_stack[idx];
                    let case = handler.find_case(operation).unwrap();
                    &*case.handler_fn as *const _
                };
                return unsafe { (*handler_fn_ptr)(&args, &mut self.state) };
            }
        }

        // Fall back to registry
        self.dispatch_via_registry(effect_name, operation, args)
    }

    /// Dispatch an operation through the registry.
    ///
    /// This is called when no handler is found in the stack.
    fn dispatch_via_registry(
        &mut self,
        effect_name: &str,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<Value, EffectError> {
        match self.dispatch_via_registry_with_suspension(effect_name, operation, args)? {
            DispatchResult::Completed(v) => Ok(v),
            DispatchResult::Suspended { suspension_id, .. } => Err(EffectError::HandlerError {
                effect: effect_name.to_string(),
                operation: operation.to_string(),
                message: format!(
                    "Suspension {} returned but not handled - use dispatch_with_continuation for suspension support",
                    suspension_id
                ),
            }),
        }
    }

    /// Dispatch an operation through the registry with suspension support.
    ///
    /// This method properly handles `HandlerResult::Suspend` by capturing the
    /// continuation and returning a `DispatchResult::Suspended`.
    ///
    /// # Returns
    ///
    /// - `DispatchResult::Completed(value)` - The operation completed with a value
    /// - `DispatchResult::Suspended { suspension_id, continuation_id }` - The operation
    ///   suspended; use `resume_suspension` with the suspension_id to continue
    fn dispatch_via_registry_with_suspension(
        &mut self,
        effect_name: &str,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<DispatchResult, EffectError> {
        let registry = self.registry.as_ref().ok_or_else(|| EffectError::UnhandledEffect {
            effect: effect_name.to_string(),
            operation: operation.to_string(),
        })?;

        let handler = registry.get(effect_name).ok_or_else(|| EffectError::UnhandledEffect {
            effect: effect_name.to_string(),
            operation: operation.to_string(),
        })?;

        // Create a continuation for the handler
        let continuation = CapabilityContinuation::new();

        // Call the handler
        let result = handler.handle(operation, &args, continuation, &mut self.capability_state);

        // Convert result with suspension support
        match result {
            CapabilityResult::Return(v) | CapabilityResult::Resume(v) => {
                Ok(DispatchResult::Completed(v))
            }
            CapabilityResult::Abort(err) => Err(EffectError::HandlerError {
                effect: err.effect,
                operation: err.operation,
                message: err.message,
            }),
            CapabilityResult::Suspend(handler_suspension_id) => {
                // Convert the handler's SuspensionId to our SuspensionId
                let suspension_id = SuspensionId::new(handler_suspension_id.0);

                // Capture a continuation for later resumption
                // The continuation will be resumed when resume_suspension is called
                let cont_id = self.capture_continuation(Some(&format!(
                    "{}::{}::suspended_{}",
                    effect_name, operation, suspension_id
                )));

                Ok(DispatchResult::Suspended {
                    suspension_id,
                    continuation_id: cont_id,
                })
            }
        }
    }

    /// Dispatch an effect operation with full suspension support.
    ///
    /// Unlike `dispatch`, this method properly handles handler suspensions by:
    /// 1. Capturing the current continuation before dispatching
    /// 2. If the handler suspends, storing the continuation and returning a DispatchResult::Suspended
    /// 3. If the handler completes, returning DispatchResult::Completed
    ///
    /// Use this method when you need to handle asynchronous or multi-shot effects.
    ///
    /// # Example
    ///
    /// ```ignore
    /// match ctx.dispatch_with_continuation(EffectKind::Async, "await", vec![future_val])? {
    ///     DispatchResult::Completed(value) => {
    ///         // Handler completed immediately
    ///         Ok(value)
    ///     }
    ///     DispatchResult::Suspended { suspension_id, continuation_id } => {
    ///         // Handler suspended - store this info for later resumption
    ///         Err(ControlFlow::Suspend { suspension_id, continuation_id })
    ///     }
    /// }
    /// ```
    pub fn dispatch_with_continuation(
        &mut self,
        effect: EffectKind,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<DispatchResult, EffectError> {
        // Try the handler stack first
        let handler_idx = self
            .handler_stack
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, handler)| {
                if handler.effect == effect && handler.find_case(operation).is_some() {
                    Some(idx)
                } else {
                    None
                }
            });

        if let Some(idx) = handler_idx {
            // Get pointer to handler function to avoid holding borrow on handler_stack
            let handler_fn_ptr: *const dyn Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError> = {
                let handler = &self.handler_stack[idx];
                let case = handler.find_case(operation).unwrap();
                &*case.handler_fn as *const _
            };

            // Call the handler - stack handlers currently don't support suspension
            let result = unsafe { (*handler_fn_ptr)(&args, &mut self.state) };
            return result.map(DispatchResult::Completed);
        }

        // Fall back to registry with suspension support
        let effect_name = effect.as_str();
        self.dispatch_via_registry_with_suspension(effect_name, operation, args)
    }

    /// Dispatch by name with suspension support.
    ///
    /// See `dispatch_with_continuation` for details on suspension handling.
    pub fn dispatch_by_name_with_continuation(
        &mut self,
        effect_name: &str,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<DispatchResult, EffectError> {
        // First try the handler stack if we can parse the effect kind
        if let Some(effect) = EffectKind::from_str(effect_name) {
            let handler_idx = self
                .handler_stack
                .iter()
                .enumerate()
                .rev()
                .find_map(|(idx, handler)| {
                    if handler.effect == effect && handler.find_case(operation).is_some() {
                        Some(idx)
                    } else {
                        None
                    }
                });

            if let Some(idx) = handler_idx {
                let handler_fn_ptr: *const dyn Fn(&[Value], &mut HandlerState) -> Result<Value, EffectError> = {
                    let handler = &self.handler_stack[idx];
                    let case = handler.find_case(operation).unwrap();
                    &*case.handler_fn as *const _
                };
                let result = unsafe { (*handler_fn_ptr)(&args, &mut self.state) };
                return result.map(DispatchResult::Completed);
            }
        }

        // Fall back to registry with suspension support
        self.dispatch_via_registry_with_suspension(effect_name, operation, args)
    }

    /// Resume a suspended computation.
    ///
    /// When a handler returns `HandlerResult::Suspend`, the computation is paused
    /// and a suspension_id is returned. Call this method with that ID and a value
    /// to resume the computation.
    ///
    /// # Arguments
    /// * `suspension_id` - The ID returned from the suspended dispatch
    /// * `continuation_id` - The continuation ID returned from the suspended dispatch
    /// * `value` - The value to resume with
    ///
    /// # Returns
    /// The result of resuming the continuation
    pub fn resume_suspension(
        &mut self,
        suspension_id: SuspensionId,
        continuation_id: ContinuationId,
        value: Value,
    ) -> Result<Value, EffectError> {
        // Verify the continuation exists
        if !self.continuation_store.contains(continuation_id) {
            return Err(EffectError::HandlerError {
                effect: "continuation".to_string(),
                operation: "resume".to_string(),
                message: format!(
                    "Continuation {} for suspension {} not found",
                    continuation_id, suspension_id
                ),
            });
        }

        // Resume the continuation with the provided value
        self.continuation_store
            .resume_and_remove(continuation_id, value)
            .map_err(|e| EffectError::HandlerError {
                effect: "continuation".to_string(),
                operation: "resume".to_string(),
                message: format!("Failed to resume suspension {}: {}", suspension_id, e),
            })
    }

    /// Convenience method for Prob.sample
    pub fn sample(&mut self, distribution: Value) -> Result<Value, EffectError> {
        self.dispatch(EffectKind::Prob, "sample", vec![distribution])
    }

    /// Convenience method for Prob.observe
    pub fn observe(&mut self, distribution: Value, value: Value) -> Result<Value, EffectError> {
        self.dispatch(EffectKind::Prob, "observe", vec![distribution, value])
    }

    /// Convenience method for Causal.do
    pub fn do_intervention(&mut self, variable: Value, value: Value) -> Result<Value, EffectError> {
        self.dispatch(EffectKind::Causal, "do", vec![variable, value])
    }

    /// Convenience method for Causal.counterfactual
    pub fn counterfactual(
        &mut self,
        factual: Value,
        intervention: Value,
        query: Value,
    ) -> Result<Value, EffectError> {
        self.dispatch(
            EffectKind::Causal,
            "counterfactual",
            vec![factual, intervention, query],
        )
    }

    // =========================================================================
    // Continuation Support
    // =========================================================================

    /// Capture a continuation at the current point (stub version)
    ///
    /// This creates a stub continuation that can be used for testing.
    /// For real interpreter continuations, use `capture_interpreter_continuation`.
    ///
    /// # Arguments
    /// * `label` - Optional label for debugging
    ///
    /// # Returns
    /// The ID of the captured continuation
    pub fn capture_continuation(&mut self, label: Option<&str>) -> ContinuationId {
        let mut cont = CapturedContinuation::new_one_shot(ResumePoint::Stub);
        if let Some(l) = label {
            cont = cont.with_label(l);
        }
        self.continuation_store.store(cont)
    }

    /// Capture an interpreter continuation with a resume closure.
    ///
    /// This creates a real continuation that can be resumed by calling
    /// the provided closure with the resume value.
    ///
    /// # Arguments
    /// * `effect` - The effect name (for debugging/tracking)
    /// * `operation` - The operation name (for debugging/tracking)
    /// * `is_multi_shot` - Whether this continuation can be resumed multiple times
    /// * `resume_fn` - Closure that will be called when the continuation is resumed
    ///
    /// # Returns
    /// The ID of the captured continuation
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cont_id = ctx.capture_interpreter_continuation(
    ///     "Prob",
    ///     "sample",
    ///     false, // one-shot
    ///     |value| {
    ///         // Continue execution with the sampled value
    ///         Ok(compute_result(value))
    ///     },
    /// );
    /// ```
    pub fn capture_interpreter_continuation<F>(
        &mut self,
        effect: &str,
        operation: &str,
        is_multi_shot: bool,
        resume_fn: F,
    ) -> ContinuationId
    where
        F: FnOnce(Value) -> Result<Value, ContinuationError> + 'static,
    {
        let label = format!("{}::{}", effect, operation);
        let resume_point = ResumePoint::interpreter_closure(resume_fn, Some(&label));

        let cont = if is_multi_shot {
            CapturedContinuation::new_multi_shot(resume_point).with_label(&label)
        } else {
            CapturedContinuation::new_one_shot(resume_point).with_label(&label)
        };

        self.continuation_store.store(cont)
    }

    /// Capture a multi-shot continuation (stub version)
    ///
    /// Multi-shot continuations can be resumed multiple times, which is
    /// useful for effects like non-determinism or backtracking.
    pub fn capture_multi_shot_continuation(&mut self, label: Option<&str>) -> ContinuationId {
        let mut cont = CapturedContinuation::new_multi_shot(ResumePoint::Stub);
        if let Some(l) = label {
            cont = cont.with_label(l);
        }
        self.continuation_store.store(cont)
    }

    /// Capture a multi-shot interpreter continuation with a reusable closure.
    ///
    /// Unlike `capture_interpreter_continuation`, this creates a continuation
    /// that can be resumed multiple times. The closure is wrapped in an `Arc`
    /// and uses `Fn` instead of `FnOnce`, allowing it to be invoked repeatedly.
    ///
    /// This is essential for effects like:
    /// - **Prob (multi-world inference)**: Explore multiple probabilistic branches
    /// - **Amb (non-determinism)**: Try different choices
    /// - **Backtracking search**: Retry with different values
    ///
    /// # Arguments
    /// * `effect` - The effect name (for debugging/tracking)
    /// * `operation` - The operation name (for debugging/tracking)
    /// * `resume_fn` - Closure that will be called when the continuation is resumed.
    ///                 Must be `Fn` (not `FnOnce`) and `Send + Sync` for thread safety.
    ///
    /// # Returns
    /// The ID of the captured continuation
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create a multi-shot continuation for probabilistic sampling
    /// let cont_id = ctx.capture_multi_shot_interpreter_continuation(
    ///     "Prob",
    ///     "sample",
    ///     |value| {
    ///         // This can be called multiple times with different sampled values
    ///         Ok(compute_result(value))
    ///     },
    /// );
    ///
    /// // Resume multiple times to explore different branches
    /// let result1 = ctx.resume_continuation(cont_id, Value::Float(0.3))?;
    /// let result2 = ctx.resume_continuation(cont_id, Value::Float(0.7))?;
    /// let result3 = ctx.resume_continuation(cont_id, Value::Float(0.9))?;
    /// ```
    pub fn capture_multi_shot_interpreter_continuation<F>(
        &mut self,
        effect: &str,
        operation: &str,
        resume_fn: F,
    ) -> ContinuationId
    where
        F: Fn(Value) -> Result<Value, ContinuationError> + Send + Sync + 'static,
    {
        let label = format!("{}::{} (multi-shot)", effect, operation);
        let resume_point = ResumePoint::interpreter_multi_shot(resume_fn, Some(&label));

        let cont = CapturedContinuation::new_multi_shot(resume_point).with_label(&label);

        self.continuation_store.store(cont)
    }

    /// Resume a captured continuation with a value
    ///
    /// For one-shot continuations, this removes the continuation from the store.
    /// For multi-shot continuations, the continuation remains available.
    ///
    /// When the continuation was captured with `capture_interpreter_continuation`,
    /// this will execute the resume closure with the provided value.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The continuation is not found
    /// - The continuation is one-shot and has already been resumed
    /// - The resume closure returns an error
    pub fn resume_continuation(
        &mut self,
        id: ContinuationId,
        value: Value,
    ) -> Result<Value, EffectError> {
        self.continuation_store
            .resume_and_remove(id, value)
            .map_err(|e| EffectError::HandlerError {
                effect: "continuation".to_string(),
                operation: "resume".to_string(),
                message: e.to_string(),
            })
    }

    /// Perform an effect operation with continuation capture
    ///
    /// This is the core method for algebraic effects. It:
    /// 1. Captures the current continuation
    /// 2. Dispatches to the handler with the continuation
    /// 3. Returns the result (or suspends for async handlers)
    ///
    /// NOTE: This is a foundation - full implementation will involve
    /// actual continuation capture and handler invocation.
    pub fn perform(
        &mut self,
        effect: EffectKind,
        operation: &str,
        args: Vec<Value>,
    ) -> Result<PerformResult, EffectError> {
        // Capture a continuation for this perform
        let cont_id = self.capture_continuation(Some(&format!("{}::{}", effect, operation)));

        // Dispatch to the handler
        let result = self.dispatch(effect, operation, args)?;

        // For now, we complete immediately
        // Full implementation would allow the handler to suspend or resume
        Ok(PerformResult::Completed {
            value: result,
            continuation_id: cont_id,
        })
    }

    /// Check if a continuation exists
    pub fn has_continuation(&self, id: ContinuationId) -> bool {
        self.continuation_store.contains(id)
    }

    /// Get the number of active continuations
    pub fn active_continuation_count(&self) -> usize {
        self.continuation_store.len()
    }

    /// Clear all captured continuations
    ///
    /// This is useful for cleanup after an effect scope ends.
    pub fn clear_continuations(&mut self) {
        self.continuation_store.clear();
    }
}

/// Result of performing an effect operation
#[derive(Debug)]
pub enum PerformResult {
    /// The effect completed immediately with a value
    Completed {
        value: Value,
        continuation_id: ContinuationId,
    },
    /// The effect is suspended, waiting to be resumed
    /// (Not yet implemented - placeholder for async effects)
    Suspended {
        continuation_id: ContinuationId,
    },
}

impl PerformResult {
    /// Get the value if completed, None if suspended
    pub fn value(&self) -> Option<&Value> {
        match self {
            PerformResult::Completed { value, .. } => Some(value),
            PerformResult::Suspended { .. } => None,
        }
    }

    /// Get the continuation ID
    pub fn continuation_id(&self) -> ContinuationId {
        match self {
            PerformResult::Completed { continuation_id, .. } => *continuation_id,
            PerformResult::Suspended { continuation_id } => *continuation_id,
        }
    }

    /// Check if the effect completed
    pub fn is_completed(&self) -> bool {
        matches!(self, PerformResult::Completed { .. })
    }
}

impl Default for EffectContext {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for EffectContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectContext")
            .field("handler_stack_size", &self.handler_stack.len())
            .field(
                "handlers",
                &self
                    .handler_stack
                    .iter()
                    .map(|h| format!("{}:{}", h.effect, h.name))
                    .collect::<Vec<_>>(),
            )
            .field("active_continuations", &self.continuation_store.len())
            .finish()
    }
}

// =============================================================================
// Default Handlers
// =============================================================================

/// Create the default handler for the Prob effect
pub fn default_prob_handler() -> EffectHandler {
    EffectHandler::new(EffectKind::Prob, "default_prob")
        .with_case("sample", |args, state| {
            // Sample from a distribution
            if args.is_empty() {
                return Err(EffectError::InvalidArguments {
                    effect: "Prob".to_string(),
                    operation: "sample".to_string(),
                    expected: 1,
                    got: 0,
                });
            }

            let dist = &args[0];
            match dist {
                Value::Distribution(d) => {
                    let value = sample_from_distribution(d, &mut state.rng);
                    Ok(Value::Float(value))
                }
                Value::Struct { name, fields } => {
                    // Handle distribution structs (e.g., Normal { mean: 0.0, std: 1.0 })
                    let prob_dist = value_to_prob_distribution(name, fields)?;
                    let value = prob_dist.sample(&mut state.rng);
                    Ok(Value::Float(value))
                }
                _ => Err(EffectError::TypeMismatch {
                    effect: "Prob".to_string(),
                    operation: "sample".to_string(),
                    message: format!("expected Distribution, got {:?}", dist.type_name()),
                }),
            }
        })
        .with_case("observe", |args, state| {
            // Observe (condition on) a value
            if args.len() < 2 {
                return Err(EffectError::InvalidArguments {
                    effect: "Prob".to_string(),
                    operation: "observe".to_string(),
                    expected: 2,
                    got: args.len(),
                });
            }

            let dist = &args[0];
            let observed_value = &args[1];

            let obs_float = match observed_value {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                _ => {
                    return Err(EffectError::TypeMismatch {
                        effect: "Prob".to_string(),
                        operation: "observe".to_string(),
                        message: "observed value must be numeric".to_string(),
                    });
                }
            };

            match dist {
                Value::Distribution(d) => {
                    let prob_dist = interpreter_dist_to_prob(d);
                    state.prob_ctx.observe(&prob_dist, obs_float);
                    Ok(Value::Unit)
                }
                Value::Struct { name, fields } => {
                    let prob_dist = value_to_prob_distribution(name, fields)?;
                    state.prob_ctx.observe(&prob_dist, obs_float);
                    Ok(Value::Unit)
                }
                _ => Err(EffectError::TypeMismatch {
                    effect: "Prob".to_string(),
                    operation: "observe".to_string(),
                    message: format!("expected Distribution, got {:?}", dist.type_name()),
                }),
            }
        })
        .with_case("score", |args, state| {
            // Add log probability to the trace
            if args.is_empty() {
                return Err(EffectError::InvalidArguments {
                    effect: "Prob".to_string(),
                    operation: "score".to_string(),
                    expected: 1,
                    got: 0,
                });
            }

            let log_prob = match &args[0] {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => {
                    return Err(EffectError::TypeMismatch {
                        effect: "Prob".to_string(),
                        operation: "score".to_string(),
                        message: "log_prob must be numeric".to_string(),
                    });
                }
            };

            // Add to log probability in context
            // (ProbContext tracks this internally)
            state.prob_ctx.observe(
                &prob::Distribution::Normal { mean: 0.0, std: 1.0 },
                0.0, // Dummy observation
            );
            // Note: Full implementation would directly add log_prob

            Ok(Value::Float(log_prob))
        })
}

/// Create the default handler for the Causal effect
pub fn default_causal_handler() -> EffectHandler {
    EffectHandler::new(EffectKind::Causal, "default_causal")
        .with_case("do", |args, state| {
            // do(variable, value) - intervene on a variable
            if args.len() < 2 {
                return Err(EffectError::InvalidArguments {
                    effect: "Causal".to_string(),
                    operation: "do".to_string(),
                    expected: 2,
                    got: args.len(),
                });
            }

            let variable = match &args[0] {
                Value::String(s) => s.clone(),
                _ => {
                    return Err(EffectError::TypeMismatch {
                        effect: "Causal".to_string(),
                        operation: "do".to_string(),
                        message: "variable must be a string".to_string(),
                    });
                }
            };

            let value = args[1].clone();

            // Record intervention
            state.interventions.push((variable.clone(), value.clone()));

            // Return the intervention value
            Ok(value)
        })
        .with_case("counterfactual", |args, state| {
            // counterfactual(factual, intervention, query)
            // Three-step process: Abduction, Action, Prediction
            if args.len() < 3 {
                return Err(EffectError::InvalidArguments {
                    effect: "Causal".to_string(),
                    operation: "counterfactual".to_string(),
                    expected: 3,
                    got: args.len(),
                });
            }

            // For the skeleton, return the query value directly
            // Full implementation would:
            // 1. Abduction: infer latent variables from factual
            // 2. Action: apply intervention
            // 3. Prediction: compute outcome under modified model
            let query = args[2].clone();
            Ok(query)
        })
        .with_case("observe_causal", |args, state| {
            // Observe a causal variable (for abduction)
            if args.len() < 2 {
                return Err(EffectError::InvalidArguments {
                    effect: "Causal".to_string(),
                    operation: "observe_causal".to_string(),
                    expected: 2,
                    got: args.len(),
                });
            }

            let variable = match &args[0] {
                Value::String(s) => s.clone(),
                _ => {
                    return Err(EffectError::TypeMismatch {
                        effect: "Causal".to_string(),
                        operation: "observe_causal".to_string(),
                        message: "variable must be a string".to_string(),
                    });
                }
            };

            let value = args[1].clone();
            state.observations.push((variable, value.clone()));

            Ok(value)
        })
        .with_case("query", |args, _state| {
            // P(target | given, do(interventions))
            // Simplified: return target expression value
            if args.is_empty() {
                return Ok(Value::Unit);
            }
            Ok(args[0].clone())
        })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Sample from an interpreter Distribution value
fn sample_from_distribution(dist: &Distribution, rng: &mut Rng) -> f64 {
    match dist {
        Distribution::Normal { mean, std } => mean + std * rng.next_normal(),
        Distribution::Uniform { a, b } => a + (b - a) * rng.next_f64(),
        Distribution::Beta { alpha, beta } => {
            // Use runtime prob module
            let prob_dist = prob::Distribution::Beta {
                alpha: *alpha,
                beta: *beta,
            };
            prob_dist.sample(rng)
        }
        Distribution::Exponential { lambda } => {
            let prob_dist = prob::Distribution::Exponential { rate: *lambda };
            prob_dist.sample(rng)
        }
        Distribution::Categorical { probs } => {
            let prob_dist = prob::Distribution::Categorical {
                probs: probs.clone(),
            };
            prob_dist.sample(rng)
        }
    }
}

/// Convert interpreter Distribution to runtime prob Distribution
fn interpreter_dist_to_prob(dist: &Distribution) -> prob::Distribution {
    match dist {
        Distribution::Normal { mean, std } => prob::Distribution::Normal {
            mean: *mean,
            std: *std,
        },
        Distribution::Uniform { a, b } => prob::Distribution::Uniform {
            low: *a,
            high: *b,
        },
        Distribution::Beta { alpha, beta } => prob::Distribution::Beta {
            alpha: *alpha,
            beta: *beta,
        },
        Distribution::Exponential { lambda } => prob::Distribution::Exponential { rate: *lambda },
        Distribution::Categorical { probs } => prob::Distribution::Categorical {
            probs: probs.clone(),
        },
    }
}

/// Convert a Value struct to a prob::Distribution
fn value_to_prob_distribution(
    name: &str,
    fields: &HashMap<String, Value>,
) -> Result<prob::Distribution, EffectError> {
    let get_float = |field: &str| -> Result<f64, EffectError> {
        fields
            .get(field)
            .and_then(|v| match v {
                Value::Float(f) => Some(*f),
                Value::Int(i) => Some(*i as f64),
                _ => None,
            })
            .ok_or_else(|| EffectError::TypeMismatch {
                effect: "Prob".to_string(),
                operation: "sample".to_string(),
                message: format!("missing or invalid field `{}`", field),
            })
    };

    match name {
        "Normal" => Ok(prob::Distribution::Normal {
            mean: get_float("mean")?,
            std: get_float("std").or_else(|_| get_float("stddev"))?,
        }),
        "Uniform" => Ok(prob::Distribution::Uniform {
            low: get_float("low").or_else(|_| get_float("a"))?,
            high: get_float("high").or_else(|_| get_float("b"))?,
        }),
        "Beta" => Ok(prob::Distribution::Beta {
            alpha: get_float("alpha")?,
            beta: get_float("beta")?,
        }),
        "Gamma" => Ok(prob::Distribution::Gamma {
            shape: get_float("shape")?,
            rate: get_float("rate")?,
        }),
        "Exponential" => Ok(prob::Distribution::Exponential {
            rate: get_float("rate").or_else(|_| get_float("lambda"))?,
        }),
        "Bernoulli" => Ok(prob::Distribution::Bernoulli {
            p: get_float("p")?,
        }),
        "Poisson" => Ok(prob::Distribution::Poisson {
            lambda: get_float("lambda")?,
        }),
        _ => Err(EffectError::TypeMismatch {
            effect: "Prob".to_string(),
            operation: "sample".to_string(),
            message: format!("unknown distribution type `{}`", name),
        }),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_context_creation() {
        let ctx = EffectContext::new();
        assert!(ctx.handler_stack.len() >= 2); // Prob and Causal handlers
    }

    #[test]
    fn test_effect_kind_from_str() {
        assert_eq!(EffectKind::from_str("Prob"), Some(EffectKind::Prob));
        assert_eq!(EffectKind::from_str("Causal"), Some(EffectKind::Causal));
        assert_eq!(EffectKind::from_str("IO"), Some(EffectKind::IO));
        assert_eq!(EffectKind::from_str("Unknown"), None);
    }

    #[test]
    fn test_prob_sample_dispatch() {
        let mut ctx = EffectContext::with_seed(42);

        // Create a Normal distribution value
        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        // Sample should succeed
        let result = ctx.sample(dist);
        assert!(result.is_ok());

        if let Ok(Value::Float(v)) = result {
            // Value should be reasonable for N(0, 1)
            assert!(v > -10.0 && v < 10.0);
        } else {
            panic!("Expected Float value from sample");
        }
    }

    #[test]
    fn test_prob_observe_dispatch() {
        let mut ctx = EffectContext::with_seed(42);

        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        let result = ctx.observe(dist, Value::Float(0.5));
        assert!(result.is_ok());
    }

    #[test]
    fn test_causal_do_dispatch() {
        let mut ctx = EffectContext::new();

        let result = ctx.do_intervention(
            Value::String("X".to_string()),
            Value::Float(1.0),
        );

        assert!(result.is_ok());
        assert_eq!(ctx.state.interventions.len(), 1);
        assert_eq!(ctx.state.interventions[0].0, "X");
    }

    #[test]
    fn test_unhandled_effect() {
        let mut ctx = EffectContext::new();

        // Try to dispatch an operation that doesn't exist
        let result = ctx.dispatch(EffectKind::Prob, "nonexistent_op", vec![]);
        assert!(result.is_err());

        if let Err(EffectError::UnhandledEffect { effect, operation }) = result {
            assert_eq!(effect, "Prob");
            assert_eq!(operation, "nonexistent_op");
        }
    }

    #[test]
    fn test_handler_stack() {
        let mut ctx = EffectContext::new();
        let initial_count = ctx.handler_stack.len();

        // Push a custom handler
        let custom_handler = EffectHandler::new(EffectKind::Prob, "custom_prob")
            .with_case("sample", |_args, _state| Ok(Value::Float(42.0)));

        ctx.push_handler(custom_handler);
        assert_eq!(ctx.handler_stack.len(), initial_count + 1);

        // Custom handler should take precedence
        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        let result = ctx.sample(dist);
        assert!(matches!(result, Ok(Value::Float(v)) if (v - 42.0).abs() < 0.001));

        // Pop and verify default handler is used again
        ctx.pop_handler();
        assert_eq!(ctx.handler_stack.len(), initial_count);
    }

    #[test]
    fn test_dispatch_by_name() {
        let mut ctx = EffectContext::with_seed(123);

        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(5.0));
        fields.insert("std".to_string(), Value::Float(0.1));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        let result = ctx.dispatch_by_name("Prob", "sample", vec![dist]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_handler_state_reset() {
        let mut state = HandlerState::new();

        state.interventions.push(("X".to_string(), Value::Float(1.0)));
        state.observations.push(("Y".to_string(), Value::Float(2.0)));

        assert!(!state.interventions.is_empty());
        assert!(!state.observations.is_empty());

        state.reset();

        assert!(state.interventions.is_empty());
        assert!(state.observations.is_empty());
    }

    #[test]
    fn test_distribution_conversion() {
        // Test value_to_prob_distribution for various distribution types
        let mut fields = HashMap::new();
        fields.insert("alpha".to_string(), Value::Float(2.0));
        fields.insert("beta".to_string(), Value::Float(5.0));

        let result = value_to_prob_distribution("Beta", &fields);
        assert!(result.is_ok());

        if let Ok(prob::Distribution::Beta { alpha, beta }) = result {
            assert!((alpha - 2.0).abs() < 0.001);
            assert!((beta - 5.0).abs() < 0.001);
        }
    }

    // =========================================================================
    // Continuation Tests
    // =========================================================================

    #[test]
    fn test_continuation_store_in_context() {
        let ctx = EffectContext::new();
        assert_eq!(ctx.active_continuation_count(), 0);
    }

    #[test]
    fn test_capture_continuation() {
        let mut ctx = EffectContext::new();

        let id = ctx.capture_continuation(Some("test"));
        assert!(ctx.has_continuation(id));
        assert_eq!(ctx.active_continuation_count(), 1);

        let id2 = ctx.capture_continuation(None);
        assert!(ctx.has_continuation(id2));
        assert_eq!(ctx.active_continuation_count(), 2);
    }

    #[test]
    fn test_resume_continuation() {
        let mut ctx = EffectContext::new();

        let id = ctx.capture_continuation(Some("test_resume"));

        // Resume with a value - using stub, it should just return the value
        let result = ctx.resume_continuation(id, Value::Int(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Int(42));

        // One-shot continuation should be removed after resume
        assert!(!ctx.has_continuation(id));
    }

    #[test]
    fn test_resume_nonexistent_continuation() {
        let mut ctx = EffectContext::new();

        // Create an ID but don't store it
        let fake_id = ContinuationId::new();

        let result = ctx.resume_continuation(fake_id, Value::Unit);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_shot_continuation() {
        let mut ctx = EffectContext::new();

        let id = ctx.capture_multi_shot_continuation(Some("multi_shot"));

        // First resume
        let result1 = ctx.resume_continuation(id, Value::Int(1));
        assert!(result1.is_ok());

        // Multi-shot should still be in the store
        assert!(ctx.has_continuation(id));

        // Second resume should also work
        let result2 = ctx.resume_continuation(id, Value::Int(2));
        assert!(result2.is_ok());
    }

    #[test]
    fn test_clear_continuations() {
        let mut ctx = EffectContext::new();

        ctx.capture_continuation(None);
        ctx.capture_continuation(None);
        ctx.capture_continuation(None);

        assert_eq!(ctx.active_continuation_count(), 3);

        ctx.clear_continuations();

        assert_eq!(ctx.active_continuation_count(), 0);
    }

    #[test]
    fn test_perform_operation() {
        let mut ctx = EffectContext::with_seed(42);

        // Create a distribution for testing
        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        // Perform a sample operation
        let result = ctx.perform(EffectKind::Prob, "sample", vec![dist]);
        assert!(result.is_ok());

        let perform_result = result.unwrap();
        assert!(perform_result.is_completed());
        assert!(perform_result.value().is_some());

        // The continuation should have been captured
        // (though for stub, it gets removed immediately after resume)
    }

    #[test]
    fn test_perform_result_accessors() {
        let completed = PerformResult::Completed {
            value: Value::Float(3.14),
            continuation_id: ContinuationId::new(),
        };

        assert!(completed.is_completed());
        assert!(completed.value().is_some());

        let suspended = PerformResult::Suspended {
            continuation_id: ContinuationId::new(),
        };

        assert!(!suspended.is_completed());
        assert!(suspended.value().is_none());
    }

    #[test]
    fn test_effect_context_debug_includes_continuations() {
        let mut ctx = EffectContext::new();
        ctx.capture_continuation(Some("debug_test"));

        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("active_continuations"));
    }

    // =========================================================================
    // HandlerCapability Tests
    // =========================================================================

    #[test]
    fn test_capability_adapter_creation() {
        use crate::effects::handler_capability::{
            Continuation, HandlerError, HandlerResult, HandlerState as CapHandlerState,
            OperationSpec,
        };

        #[derive(Debug)]
        struct TestCapHandler;

        impl HandlerCapability for TestCapHandler {
            fn effect_name(&self) -> &str {
                "Test"
            }
            fn handler_name(&self) -> &str {
                "TestCapHandler"
            }
            fn operations(&self) -> &[OperationSpec] {
                &[]
            }
            fn handle(
                &self,
                operation: &str,
                _args: &[Value],
                _continuation: Continuation,
                _state: &mut CapHandlerState,
            ) -> HandlerResult {
                match operation {
                    "noop" => HandlerResult::Resume(Value::Unit),
                    _ => HandlerResult::Abort(HandlerError::new("Test", operation, "Unknown")),
                }
            }
        }

        let adapter = CapabilityAdapter::new(Box::new(TestCapHandler));
        assert_eq!(adapter.effect_name(), "Test");
    }

    #[test]
    fn test_capability_adapter_direct_dispatch() {
        use crate::effects::handler_capability::{
            Continuation, HandlerError, HandlerResult, HandlerState as CapHandlerState,
            OperationSpec,
        };

        #[derive(Debug)]
        struct AddHandler;

        impl HandlerCapability for AddHandler {
            fn effect_name(&self) -> &str {
                "Math"
            }
            fn handler_name(&self) -> &str {
                "AddHandler"
            }
            fn operations(&self) -> &[OperationSpec] {
                &[]
            }
            fn handle(
                &self,
                operation: &str,
                args: &[Value],
                _continuation: Continuation,
                _state: &mut CapHandlerState,
            ) -> HandlerResult {
                match operation {
                    "add" => {
                        if args.len() >= 2 {
                            match (&args[0], &args[1]) {
                                (Value::Int(a), Value::Int(b)) => {
                                    HandlerResult::Resume(Value::Int(a + b))
                                }
                                _ => HandlerResult::Abort(HandlerError::new(
                                    "Math",
                                    "add",
                                    "Expected integers",
                                )),
                            }
                        } else {
                            HandlerResult::Abort(HandlerError::new(
                                "Math",
                                "add",
                                "Expected 2 arguments",
                            ))
                        }
                    }
                    _ => HandlerResult::Abort(HandlerError::new("Math", operation, "Unknown")),
                }
            }
        }

        let adapter = CapabilityAdapter::new(Box::new(AddHandler));
        let continuation = CapabilityContinuation::new();

        let result =
            adapter.dispatch_direct("add", &[Value::Int(2), Value::Int(3)], continuation);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Int(5));
    }

    // =========================================================================
    // Handler Registry Integration Tests
    // =========================================================================

    #[test]
    fn test_effect_context_with_registry() {
        let ctx = EffectContext::with_registry(HandlerRegistry::with_defaults());
        assert!(ctx.registry().is_some());
        assert_eq!(ctx.registry().unwrap().len(), 12);
    }

    #[test]
    fn test_effect_context_with_all_handlers() {
        let ctx = EffectContext::with_all_handlers();
        assert!(ctx.registry().is_some());
    }

    #[test]
    fn test_dispatch_via_registry_io() {
        let mut ctx = EffectContext::with_all_handlers();

        // Dispatch IO.print through the registry
        let result = ctx.dispatch_by_name("IO", "print", vec![Value::String("hello".into())]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispatch_via_registry_div() {
        let mut ctx = EffectContext::with_all_handlers();

        // Dispatch Div.div through the registry
        let result = ctx.dispatch_by_name("Div", "div", vec![Value::Float(10.0), Value::Float(2.0)]);
        assert!(result.is_ok());
        if let Ok(Value::Float(v)) = result {
            assert!((v - 5.0).abs() < 0.001);
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_dispatch_via_registry_mut() {
        let mut ctx = EffectContext::with_all_handlers();

        // Set a value
        let set_result = ctx.dispatch_by_name(
            "Mut",
            "set",
            vec![Value::String("x".into()), Value::Int(42)],
        );
        assert!(set_result.is_ok());

        // Get the value back
        let get_result = ctx.dispatch_by_name("Mut", "get", vec![Value::String("x".into())]);
        assert!(get_result.is_ok());
        assert_eq!(get_result.unwrap(), Value::Int(42));
    }

    #[test]
    fn test_dispatch_via_registry_panic() {
        let mut ctx = EffectContext::with_all_handlers();

        // Dispatch Panic.assert (should succeed for true)
        let result = ctx.dispatch_by_name("Panic", "assert", vec![Value::Bool(true)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_handler_stack_takes_precedence_over_registry() {
        let mut ctx = EffectContext::with_all_handlers();

        // Push a custom Prob handler that always returns 99.0
        let custom_handler = EffectHandler::new(EffectKind::Prob, "custom_prob")
            .with_case("sample", |_args, _state| Ok(Value::Float(99.0)));
        ctx.push_handler(custom_handler);

        // Create a distribution
        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        // The custom handler should take precedence
        let result = ctx.sample(dist);
        assert!(matches!(result, Ok(Value::Float(v)) if (v - 99.0).abs() < 0.001));
    }

    #[test]
    fn test_set_registry_after_creation() {
        let mut ctx = EffectContext::new();
        assert!(ctx.registry().is_none());

        ctx.set_registry(HandlerRegistry::with_defaults());
        assert!(ctx.registry().is_some());

        // Now IO dispatch should work
        let result = ctx.dispatch_by_name("IO", "print", vec![Value::String("test".into())]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispatch_unhandled_without_registry() {
        let mut ctx = EffectContext::new();

        // Without a registry, unknown effects should fail
        let result = ctx.dispatch_by_name("Network", "fetch", vec![Value::String("http://example.com".into())]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatch_with_registry_fallback() {
        let mut ctx = EffectContext::with_all_handlers();

        // Network effect is not in the handler stack, only in registry
        let result = ctx.dispatch_by_name(
            "Network",
            "fetch",
            vec![Value::String("http://example.com".into())],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_effect_kind_new_variants() {
        assert_eq!(EffectKind::from_str("Panic"), Some(EffectKind::Panic));
        assert_eq!(EffectKind::from_str("Network"), Some(EffectKind::Network));
        assert_eq!(EffectKind::from_str("Sensor"), Some(EffectKind::Sensor));

        assert_eq!(EffectKind::Panic.as_str(), "Panic");
        assert_eq!(EffectKind::Network.as_str(), "Network");
        assert_eq!(EffectKind::Sensor.as_str(), "Sensor");
    }

    #[test]
    fn test_dispatch_sensor_via_registry() {
        let mut ctx = EffectContext::with_all_handlers();

        let result = ctx.dispatch_by_name("Sensor", "is_available", vec![Value::String("temp_sensor".into())]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispatch_gpu_via_registry() {
        let mut ctx = EffectContext::with_all_handlers();

        // Test GPU sync operation
        let result = ctx.dispatch_by_name("GPU", "sync", vec![]);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Interpreter Continuation Tests
    // =========================================================================

    #[test]
    fn test_capture_interpreter_continuation_basic() {
        let mut ctx = EffectContext::new();

        // Capture a continuation that doubles the input
        let id = ctx.capture_interpreter_continuation(
            "Test",
            "double",
            false, // one-shot
            |value| {
                match value {
                    Value::Int(n) => Ok(Value::Int(n * 2)),
                    _ => Ok(value),
                }
            },
        );

        assert!(ctx.has_continuation(id));
        assert_eq!(ctx.active_continuation_count(), 1);

        // Resume with a value
        let result = ctx.resume_continuation(id, Value::Int(21));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Int(42));

        // Should be removed after resume (one-shot)
        assert!(!ctx.has_continuation(id));
    }

    #[test]
    fn test_capture_interpreter_continuation_with_closure_state() {
        let mut ctx = EffectContext::new();

        // Capture a continuation that adds a captured value
        let captured_value = 100;
        let id = ctx.capture_interpreter_continuation(
            "Test",
            "add_captured",
            false,
            move |value| {
                match value {
                    Value::Int(n) => Ok(Value::Int(n + captured_value)),
                    _ => Ok(value),
                }
            },
        );

        let result = ctx.resume_continuation(id, Value::Int(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Int(142));
    }

    #[test]
    fn test_capture_interpreter_continuation_error() {
        use crate::effects::continuation::ContinuationError;

        let mut ctx = EffectContext::new();

        // Capture a continuation that returns an error
        let id = ctx.capture_interpreter_continuation(
            "Test",
            "fail",
            false,
            |_value| {
                Err(ContinuationError::NotImplemented("test error".to_string()))
            },
        );

        let result = ctx.resume_continuation(id, Value::Unit);
        assert!(result.is_err());
    }

    #[test]
    fn test_capture_interpreter_continuation_one_shot_enforcement() {
        let mut ctx = EffectContext::new();

        // Capture a one-shot continuation
        let id = ctx.capture_interpreter_continuation(
            "Test",
            "one_shot",
            false,
            |value| Ok(value),
        );

        // First resume should succeed
        let result1 = ctx.resume_continuation(id, Value::Int(1));
        assert!(result1.is_ok());

        // Continuation should be removed, second resume should fail
        assert!(!ctx.has_continuation(id));
    }

    #[test]
    fn test_interpreter_continuation_label() {
        let mut ctx = EffectContext::new();

        let id = ctx.capture_interpreter_continuation(
            "MyEffect",
            "myOp",
            false,
            |value| Ok(value),
        );

        // The continuation should have a label with format "effect::operation"
        let cont = ctx.continuation_store.get(id);
        assert!(cont.is_some());
        assert_eq!(cont.unwrap().label, Some("MyEffect::myOp".to_string()));
    }

    #[test]
    fn test_interpreter_continuation_chaining() {
        let mut ctx = EffectContext::new();

        // Simulate a chain of effect operations
        // First effect: multiply by 2
        let id1 = ctx.capture_interpreter_continuation(
            "Math",
            "mul2",
            false,
            |value| {
                match value {
                    Value::Int(n) => Ok(Value::Int(n * 2)),
                    _ => Ok(value),
                }
            },
        );

        // Second effect: add 10
        let id2 = ctx.capture_interpreter_continuation(
            "Math",
            "add10",
            false,
            |value| {
                match value {
                    Value::Int(n) => Ok(Value::Int(n + 10)),
                    _ => Ok(value),
                }
            },
        );

        // Resume second effect first (stack order)
        let result2 = ctx.resume_continuation(id2, Value::Int(5));
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), Value::Int(15)); // 5 + 10

        // Resume first effect
        let result1 = ctx.resume_continuation(id1, Value::Int(15));
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), Value::Int(30)); // 15 * 2
    }
}
