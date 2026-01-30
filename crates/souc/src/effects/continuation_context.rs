//! Continuation context tracking for effect handlers
//!
//! This module provides infrastructure for tracking which continuation
//! is currently "active" during handler execution. This enables the
//! `resume` expression to know which continuation to invoke.

use crate::effects::continuation::ContinuationId;

/// Context for tracking active continuations during handler execution.
///
/// When an effect operation is performed, a continuation is captured.
/// That continuation becomes "active" while the handler processes the operation.
/// The `resume` expression within the handler can then access and invoke
/// this active continuation.
///
/// For nested handlers, continuations are stacked - inner handlers have
/// their own active continuation while outer continuations remain on the stack.
#[derive(Debug, Clone, Default)]
pub struct ContinuationContext {
    /// Stack of active continuations (innermost/current last)
    active_stack: Vec<ContinuationId>,
}

impl ContinuationContext {
    /// Create a new continuation context
    pub fn new() -> Self {
        Self {
            active_stack: Vec::new(),
        }
    }

    /// Push a continuation onto the active stack.
    ///
    /// This is called when entering a handler with a captured continuation.
    pub fn push(&mut self, id: ContinuationId) {
        self.active_stack.push(id);
    }

    /// Pop the current continuation from the active stack.
    ///
    /// This is called when exiting a handler.
    pub fn pop(&mut self) -> Option<ContinuationId> {
        self.active_stack.pop()
    }

    /// Get the current (innermost) active continuation.
    ///
    /// Returns `None` if no continuation is currently active.
    /// This is used by the `resume` expression to find which
    /// continuation to invoke.
    pub fn current(&self) -> Option<ContinuationId> {
        self.active_stack.last().copied()
    }

    /// Check if any continuation is currently active
    pub fn has_active(&self) -> bool {
        !self.active_stack.is_empty()
    }

    /// Get the depth of the continuation stack.
    ///
    /// Useful for debugging nested handler scenarios.
    pub fn depth(&self) -> usize {
        self.active_stack.len()
    }

    /// Clear all active continuations.
    ///
    /// This should only be called when resetting the context
    /// or in error recovery scenarios.
    pub fn clear(&mut self) {
        self.active_stack.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuation_context_basic() {
        let mut ctx = ContinuationContext::new();

        assert!(!ctx.has_active());
        assert_eq!(ctx.current(), None);
        assert_eq!(ctx.depth(), 0);

        let id1 = ContinuationId::new();
        ctx.push(id1);

        assert!(ctx.has_active());
        assert_eq!(ctx.current(), Some(id1));
        assert_eq!(ctx.depth(), 1);

        assert_eq!(ctx.pop(), Some(id1));
        assert!(!ctx.has_active());
        assert_eq!(ctx.current(), None);
    }

    #[test]
    fn test_continuation_context_nested() {
        let mut ctx = ContinuationContext::new();

        let id1 = ContinuationId::new();
        let id2 = ContinuationId::new();
        let id3 = ContinuationId::new();

        ctx.push(id1);
        ctx.push(id2);
        ctx.push(id3);

        assert_eq!(ctx.depth(), 3);
        assert_eq!(ctx.current(), Some(id3)); // Innermost

        assert_eq!(ctx.pop(), Some(id3));
        assert_eq!(ctx.current(), Some(id2));

        assert_eq!(ctx.pop(), Some(id2));
        assert_eq!(ctx.current(), Some(id1));

        assert_eq!(ctx.pop(), Some(id1));
        assert_eq!(ctx.current(), None);
    }

    #[test]
    fn test_continuation_context_clear() {
        let mut ctx = ContinuationContext::new();

        ctx.push(ContinuationId::new());
        ctx.push(ContinuationId::new());

        assert_eq!(ctx.depth(), 2);

        ctx.clear();

        assert!(!ctx.has_active());
        assert_eq!(ctx.depth(), 0);
        assert_eq!(ctx.current(), None);
    }
}
