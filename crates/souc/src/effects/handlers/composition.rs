//! Handler Composition for Multi-Effect Functions
//!
//! This module implements handler composition based on "Soundly Handling Linearity" (POPL 2024).
//! Enables `with IO, Mut, Epistemic` multi-effect functions by composing multiple handlers
//! into a single composed handler with configurable dispatch order.
//!
//! # Architecture
//!
//! - `ComposedHandler`: Wraps multiple handlers with configurable dispatch order
//! - `CompositionOrder`: Specifies how effects are dispatched (left-to-right, right-to-left, priority)
//! - `HandlerStack`: Linear stack for one-time handler consumption (supports linear types)
//! - `HandlerComposer`: Builder API for creating composed handlers
//!
//! # Example
//!
//! ```ignore
//! use sounio::effects::handlers::{HandlerComposer, CompositionOrder};
//!
//! // Compose IO, Mut, and Epistemic handlers
//! let composed = HandlerComposer::new()
//!     .with(Box::new(IOHandler::new()))
//!     .with(Box::new(MutHandler::new()))
//!     .with(Box::new(EpistemicHandler::new()))
//!     .order(CompositionOrder::LeftToRight)
//!     .build()
//!     .expect("No duplicate effects");
//!
//! // Now the composed handler dispatches to the first matching handler
//! ```

use crate::effects::{
    HandlerCapability, HandlerError, HandlerResult, HandlerState, OperationSpec,
    handler_capability::Continuation,
};
use crate::interp::Value;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;

/// Error during handler composition
#[derive(Debug, Clone)]
pub enum ComposeError {
    /// Duplicate effect name in composition
    DuplicateEffect(String),
    /// Empty composition (no handlers)
    EmptyComposition,
}

impl std::fmt::Display for ComposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComposeError::DuplicateEffect(eff) => {
                write!(f, "Duplicate effect '{}' in handler composition", eff)
            }
            ComposeError::EmptyComposition => {
                write!(f, "Cannot create composed handler with zero handlers")
            }
        }
    }
}

impl std::error::Error for ComposeError {}

/// Composition order for dispatching effect operations.
///
/// Determines which handler gets priority when multiple handlers are composed.
///
/// # Semantics
///
/// - **LeftToRight**: First added handler gets first chance (outermost to innermost)
/// - **RightToLeft**: Last added handler gets first chance (innermost to outermost)
/// - **Priority**: Explicit priority list by effect name
///
/// # Example
///
/// ```ignore
/// // State + Error composition: order matters!
/// // Left-to-right: state is preserved on error
/// // Right-to-left: state is lost on error
/// let order = CompositionOrder::LeftToRight;
/// ```
#[derive(Debug, Clone)]
pub enum CompositionOrder {
    /// First handler added gets first chance to handle operations
    LeftToRight,
    /// Last handler added gets first chance (reverse order)
    RightToLeft,
    /// Explicit priority by effect names (first in list = highest priority)
    Priority(Vec<String>),
}

impl Default for CompositionOrder {
    fn default() -> Self {
        CompositionOrder::LeftToRight
    }
}

/// A composed handler that dispatches to multiple sub-handlers.
///
/// This is the core composition primitive enabling `with IO, Mut, Epistemic` syntax.
/// Effect operations are dispatched to the first matching handler based on composition order.
///
/// # Handler Dispatch
///
/// 1. Determine dispatch order based on `CompositionOrder`
/// 2. For each candidate handler (in order):
///    - Check if the operation matches the handler's operations
///    - If match, dispatch and return result
/// 3. If no match, return Abort with UnknownOperation error
///
/// # State Propagation
///
/// `HandlerState` is shared across all handlers, enabling:
/// - Epistemic tracking through composition
/// - Named state sharing
/// - Custom data propagation
#[derive(Debug)]
pub struct ComposedHandler {
    /// Sub-handlers in the composition
    handlers: Vec<Arc<dyn HandlerCapability + Send + Sync>>,
    /// Dispatch order configuration
    composition_order: CompositionOrder,
    /// Cached name for the composition (for debugging)
    name: String,
}

impl ComposedHandler {
    /// Create a composed handler from a list of handlers
    pub fn new(
        handlers: Vec<Arc<dyn HandlerCapability + Send + Sync>>,
        order: CompositionOrder,
    ) -> Self {
        let name = handlers
            .iter()
            .map(|h| h.effect_name())
            .collect::<Vec<_>>()
            .join("+");

        Self {
            handlers,
            composition_order: order,
            name,
        }
    }

    /// Get the number of composed handlers
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Get the composition order
    pub fn order(&self) -> &CompositionOrder {
        &self.composition_order
    }

    /// Get the effect names in this composition
    pub fn effect_names(&self) -> Vec<&str> {
        self.handlers.iter().map(|h| h.effect_name()).collect()
    }

    /// Find a handler by effect name
    pub fn find_handler(
        &self,
        effect_name: &str,
    ) -> Option<&Arc<dyn HandlerCapability + Send + Sync>> {
        self.handlers
            .iter()
            .find(|h| h.effect_name() == effect_name)
    }
}

impl HandlerCapability for ComposedHandler {
    fn effect_name(&self) -> &str {
        &self.name
    }

    fn handler_name(&self) -> &str {
        "ComposedHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        // For composed handlers, we don't pre-collect operations
        // Instead, we dynamically dispatch to sub-handlers
        // Return empty slice to indicate "check sub-handlers"
        &[]
    }

    fn handle(
        &self,
        operation: &str,
        args: &[Value],
        continuation: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        // Determine dispatch order based on composition order
        let candidates: Vec<&Arc<dyn HandlerCapability + Send + Sync>> =
            match &self.composition_order {
                CompositionOrder::LeftToRight => self.handlers.iter().collect(),
                CompositionOrder::RightToLeft => self.handlers.iter().rev().collect(),
                CompositionOrder::Priority(priority) => {
                    let mut cands: Vec<_> = self.handlers.iter().collect();
                    cands.sort_by(|a, b| {
                        let pos_a = priority
                            .iter()
                            .position(|e| e == a.effect_name())
                            .unwrap_or(usize::MAX);
                        let pos_b = priority
                            .iter()
                            .position(|e| e == b.effect_name())
                            .unwrap_or(usize::MAX);
                        pos_a.cmp(&pos_b)
                    });
                    cands
                }
            };

        // Dispatch to first matching handler
        for candidate in candidates {
            // Check if operation matches this handler's operations
            let matches = candidate
                .operations()
                .iter()
                .any(|spec| spec.name == operation);

            if matches {
                // Clone continuation for multi-shot if needed
                // For single-shot, we consume it here
                return candidate.handle(operation, args, continuation, state);
            }
        }

        // No matching handler found
        HandlerResult::Abort(HandlerError::new(
            self.effect_name(),
            operation,
            format!(
                "No handler in composition supports operation '{}'",
                operation
            ),
        ))
    }

    fn supports_multi_shot(&self) -> bool {
        // Composition supports multi-shot if any sub-handler does
        self.handlers.iter().any(|h| h.supports_multi_shot())
    }
}

/// A linear stack for managing handlers with one-time consumption semantics.
///
/// Supports linear type integration: handlers can be marked as consumed exactly once.
/// This prevents resource leaks (e.g., file handles, GPU contexts).
///
/// # Example
///
/// ```ignore
/// let mut stack = HandlerStack::new();
/// stack.push_handler(Arc::new(IOHandler::new()));
/// stack.push_handler(Arc::new(MutHandler::new()));
///
/// // Consume handlers linearly
/// while let Some(handler) = stack.pop_handler() {
///     // Handler is moved out, consumed exactly once
/// }
/// ```
#[derive(Debug, Default)]
pub struct HandlerStack {
    /// Stack of handlers (top = last pushed)
    stack: Vec<Arc<dyn HandlerCapability + Send + Sync>>,
    /// Set of consumed handler IDs (for linear type checking)
    consumed: HashSet<String>,
}

impl HandlerStack {
    /// Create a new empty handler stack
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a handler onto the stack
    pub fn push_handler(&mut self, handler: Arc<dyn HandlerCapability + Send + Sync>) {
        self.stack.push(handler);
    }

    /// Pop and consume the top handler (linear semantics)
    ///
    /// Returns None if stack is empty.
    /// Marks the handler as consumed for linear type checking.
    pub fn pop_handler(&mut self) -> Option<Arc<dyn HandlerCapability + Send + Sync>> {
        let handler = self.stack.pop()?;
        self.consumed.insert(handler.effect_name().to_string());
        Some(handler)
    }

    /// Find a handler by effect name (top-down search)
    ///
    /// Returns a reference to the topmost matching handler.
    pub fn find_handler(
        &self,
        effect_name: &str,
    ) -> Option<&Arc<dyn HandlerCapability + Send + Sync>> {
        self.stack
            .iter()
            .rev()
            .find(|h| h.effect_name() == effect_name)
    }

    /// Check if a handler has been consumed (for linear type checking)
    pub fn is_consumed(&self, effect_name: &str) -> bool {
        self.consumed.contains(effect_name)
    }

    /// Get the stack depth
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Check if the stack is empty
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Clear all handlers and reset consumed set
    pub fn clear(&mut self) {
        self.stack.clear();
        self.consumed.clear();
    }

    /// Get all effect names in the stack (top to bottom)
    pub fn effect_names(&self) -> Vec<&str> {
        self.stack.iter().rev().map(|h| h.effect_name()).collect()
    }
}

/// Builder for creating composed handlers with validation.
///
/// Provides a fluent API for composing multiple handlers with configurable dispatch order.
/// Validates that no duplicate effect names exist in the composition.
///
/// # Example
///
/// ```ignore
/// let composed = HandlerComposer::new()
///     .with(Arc::new(IOHandler::new()))
///     .with(Arc::new(MutHandler::new()))
///     .with(Arc::new(EpistemicHandler::new()))
///     .order(CompositionOrder::Priority(vec!["Epistemic".into(), "IO".into(), "Mut".into()]))
///     .build()?;
/// ```
#[derive(Debug, Default)]
pub struct HandlerComposer {
    handlers: Vec<Arc<dyn HandlerCapability + Send + Sync>>,
    order: Option<CompositionOrder>,
}

impl HandlerComposer {
    /// Create a new handler composer
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a handler to the composition
    pub fn with(mut self, handler: Arc<dyn HandlerCapability + Send + Sync>) -> Self {
        self.handlers.push(handler);
        self
    }

    /// Set the composition order
    pub fn order(mut self, order: CompositionOrder) -> Self {
        self.order = Some(order);
        self
    }

    /// Build the composed handler
    ///
    /// Returns an error if:
    /// - No handlers were added (empty composition)
    /// - Duplicate effect names exist
    pub fn build(self) -> Result<ComposedHandler, ComposeError> {
        // Validate non-empty
        if self.handlers.is_empty() {
            return Err(ComposeError::EmptyComposition);
        }

        // Validate unique effect names
        let mut effects = HashSet::new();
        for handler in &self.handlers {
            let effect = handler.effect_name();
            if !effects.insert(effect.to_string()) {
                return Err(ComposeError::DuplicateEffect(effect.to_string()));
            }
        }

        let order = self.order.unwrap_or_default();

        Ok(ComposedHandler::new(self.handlers, order))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::handlers::{IOHandler, MutHandler, PanicHandler};

    #[test]
    fn test_empty_composition_error() {
        let result = HandlerComposer::new().build();
        assert!(matches!(result, Err(ComposeError::EmptyComposition)));
    }

    #[test]
    fn test_duplicate_effect_error() {
        let result = HandlerComposer::new()
            .with(Arc::new(IOHandler::new()))
            .with(Arc::new(IOHandler::new())) // Duplicate!
            .build();

        assert!(matches!(result, Err(ComposeError::DuplicateEffect(_))));
    }

    #[test]
    fn test_composition_basic() {
        let composed = HandlerComposer::new()
            .with(Arc::new(IOHandler::new()))
            .with(Arc::new(MutHandler::new()))
            .with(Arc::new(PanicHandler::new()))
            .build()
            .expect("Valid composition");

        assert_eq!(composed.handler_count(), 3);
        assert_eq!(composed.effect_names(), vec!["IO", "Mut", "Panic"]);
    }

    #[test]
    fn test_composition_left_to_right() {
        let composed = HandlerComposer::new()
            .with(Arc::new(IOHandler::new()))
            .with(Arc::new(MutHandler::new()))
            .order(CompositionOrder::LeftToRight)
            .build()
            .expect("Valid composition");

        assert!(matches!(composed.order(), CompositionOrder::LeftToRight));
    }

    #[test]
    fn test_composition_right_to_left() {
        let composed = HandlerComposer::new()
            .with(Arc::new(IOHandler::new()))
            .with(Arc::new(MutHandler::new()))
            .order(CompositionOrder::RightToLeft)
            .build()
            .expect("Valid composition");

        assert!(matches!(composed.order(), CompositionOrder::RightToLeft));
    }

    #[test]
    fn test_composition_priority() {
        let priority = vec!["Mut".to_string(), "IO".to_string()];
        let composed = HandlerComposer::new()
            .with(Arc::new(IOHandler::new()))
            .with(Arc::new(MutHandler::new()))
            .order(CompositionOrder::Priority(priority.clone()))
            .build()
            .expect("Valid composition");

        if let CompositionOrder::Priority(p) = composed.order() {
            assert_eq!(p, &priority);
        } else {
            panic!("Expected Priority order");
        }
    }

    #[test]
    fn test_find_handler() {
        let composed = HandlerComposer::new()
            .with(Arc::new(IOHandler::new()))
            .with(Arc::new(MutHandler::new()))
            .build()
            .expect("Valid composition");

        assert!(composed.find_handler("IO").is_some());
        assert!(composed.find_handler("Mut").is_some());
        assert!(composed.find_handler("NonExistent").is_none());
    }

    #[test]
    fn test_handler_stack_push_pop() {
        let mut stack = HandlerStack::new();
        assert!(stack.is_empty());

        stack.push_handler(Arc::new(IOHandler::new()));
        stack.push_handler(Arc::new(MutHandler::new()));

        assert_eq!(stack.len(), 2);
        assert_eq!(stack.effect_names(), vec!["Mut", "IO"]); // Top to bottom

        let h1 = stack.pop_handler().expect("Should have handler");
        assert_eq!(h1.effect_name(), "Mut");
        assert!(stack.is_consumed("Mut"));

        let h2 = stack.pop_handler().expect("Should have handler");
        assert_eq!(h2.effect_name(), "IO");
        assert!(stack.is_consumed("IO"));

        assert!(stack.is_empty());
        assert!(stack.pop_handler().is_none());
    }

    #[test]
    fn test_handler_stack_find() {
        let mut stack = HandlerStack::new();
        stack.push_handler(Arc::new(IOHandler::new()));
        stack.push_handler(Arc::new(MutHandler::new()));
        stack.push_handler(Arc::new(PanicHandler::new()));

        // Find searches top-down
        assert!(stack.find_handler("Panic").is_some());
        assert!(stack.find_handler("Mut").is_some());
        assert!(stack.find_handler("IO").is_some());
        assert!(stack.find_handler("NonExistent").is_none());
    }

    #[test]
    fn test_handler_stack_clear() {
        let mut stack = HandlerStack::new();
        stack.push_handler(Arc::new(IOHandler::new()));
        stack.push_handler(Arc::new(MutHandler::new()));

        stack.pop_handler(); // Mark Mut as consumed

        assert_eq!(stack.len(), 1);
        assert!(stack.is_consumed("Mut"));

        stack.clear();
        assert!(stack.is_empty());
        assert!(!stack.is_consumed("Mut")); // Cleared
    }

    #[test]
    fn test_composed_handler_invocation() {
        let composed = HandlerComposer::new()
            .with(Arc::new(IOHandler::new()))
            .with(Arc::new(PanicHandler::new()))
            .build()
            .expect("Valid composition");

        let mut state = HandlerState::new();
        let cont = Continuation::new();

        // Test IO operation dispatch
        let result = composed.handle("print", &[Value::String("test".into())], cont, &mut state);

        // Should dispatch to IOHandler and return Resume
        match result {
            HandlerResult::Resume(_) => {}
            _ => panic!("Expected Resume from IO handler"),
        }
    }

    #[test]
    fn test_composed_handler_unknown_operation() {
        let composed = HandlerComposer::new()
            .with(Arc::new(IOHandler::new()))
            .build()
            .expect("Valid composition");

        let mut state = HandlerState::new();
        let cont = Continuation::new();

        // Test unknown operation
        let result = composed.handle("unknown_op", &[], cont, &mut state);

        // Should return Abort
        match result {
            HandlerResult::Abort(_) => {}
            _ => panic!("Expected Abort for unknown operation"),
        }
    }
}
