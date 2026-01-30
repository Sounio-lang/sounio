//! Continuation infrastructure for effect handlers
//!
//! This module provides the foundational data structures for implementing
//! continuation-passing style (CPS) transformation for algebraic effect handlers.
//!
//! # Theory Background
//!
//! Based on Plotkin & Pretnar (2009) "Handlers of Algebraic Effects" and
//! Leijen (2017) "Type Directed Compilation of Row-typed Algebraic Effects" (Koka).
//!
//! When an effect operation is performed, the current continuation (the rest of
//! the computation) must be captured and passed to the handler. The handler can
//! then choose to:
//! - Resume the continuation with a value (normal effect handling)
//! - Discard the continuation (aborting the computation)
//! - Resume multiple times (for non-determinism, backtracking)
//! - Store the continuation for later use (delimited continuations)
//!
//! # One-shot vs Multi-shot Continuations
//!
//! - **One-shot**: Can only be resumed once (more efficient, covers most use cases)
//! - **Multi-shot**: Can be resumed multiple times (needed for backtracking, amb)
//!
//! Most effect handlers only need one-shot continuations. Multi-shot continuations
//! require copying the entire stack/state, which is expensive.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::interp::Value;

/// A boxed closure that can be called to resume a continuation (one-shot).
///
/// The closure takes a resume value and returns the final computation result.
/// Note: This does not require `Send` because interpreter `Value` uses `Rc<RefCell<...>>`
/// which is not thread-safe. For cross-thread continuations, use JIT-based resumption.
pub type OneShotResumeFn = Box<dyn FnOnce(Value) -> Result<Value, ContinuationError> + 'static>;

/// Legacy alias for OneShotResumeFn
pub type ResumeFn = OneShotResumeFn;

/// A multi-shot resume function that can be called multiple times.
///
/// Uses `Arc` to allow cloning and `Fn` instead of `FnOnce` so the closure
/// can be invoked repeatedly. Requires `Send + Sync` for thread safety.
pub type MultiShotResumeFn =
    Arc<dyn Fn(Value) -> Result<Value, ContinuationError> + Send + Sync + 'static>;

/// Global counter for generating unique continuation IDs
static CONTINUATION_ID: AtomicU64 = AtomicU64::new(0);

/// Unique continuation identifier
///
/// Each captured continuation gets a unique ID for tracking and debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContinuationId(u64);

impl ContinuationId {
    /// Create a new unique continuation ID
    pub fn new() -> Self {
        Self(CONTINUATION_ID.fetch_add(1, Ordering::SeqCst))
    }

    /// Get the raw ID value (for debugging)
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for ContinuationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ContinuationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cont_{}", self.0)
    }
}

/// Where to resume execution when a continuation is invoked
///
/// This enum represents different execution contexts that can be resumed:
/// - InterpreterClosure: Resume with a one-shot closure (FnOnce)
/// - InterpreterMultiShot: Resume with a multi-shot closure (Fn, can be called multiple times)
/// - Interpreter: Resume in the tree-walking interpreter (description-only, not executable)
/// - JIT: Resume in JIT-compiled code with saved machine state
/// - Stub: Placeholder for not-yet-implemented backends
pub enum ResumePoint {
    /// Resume in interpreter using a captured one-shot closure.
    ///
    /// The closure captures the interpreter's evaluation context and can be
    /// called with the resume value to continue execution from where the
    /// effect was performed.
    ///
    /// NOTE: This closure is consumed on first call (FnOnce). For multi-shot
    /// continuations, use `InterpreterMultiShot` instead.
    InterpreterClosure {
        /// The continuation closure that resumes execution
        resume_fn: OneShotResumeFn,
        /// Optional description for debugging
        description: Option<String>,
    },

    /// Resume in interpreter using a multi-shot closure.
    ///
    /// Unlike `InterpreterClosure`, this variant uses `Arc<dyn Fn>` so the
    /// closure can be called multiple times without being consumed. This is
    /// essential for effects like:
    /// - Non-determinism (Amb): explore multiple branches
    /// - Backtracking search: retry with different values
    /// - Probabilistic effects: multi-world inference (SMC, enumeration)
    ///
    /// The closure is cloned on each resume, allowing the same continuation
    /// to be invoked multiple times with different values.
    InterpreterMultiShot {
        /// The continuation closure that resumes execution (can be called multiple times)
        resume_fn: MultiShotResumeFn,
        /// Optional description for debugging
        description: Option<String>,
    },

    /// Resume in interpreter with a description (not executable).
    ///
    /// This variant is used when we want to record a continuation point
    /// but the actual resume logic is handled elsewhere.
    Interpreter {
        /// The continuation closure - currently a stub that just returns the value
        /// Full implementation will capture the interpreter's evaluation context
        #[allow(dead_code)]
        description: String,
    },

    /// Resume in JIT-compiled code with saved machine state
    ///
    /// This captures the low-level execution state needed to resume
    /// native code execution.
    Jit {
        /// Return address in the JIT-compiled code
        return_address: usize,
        /// Saved register values (architecture-dependent)
        saved_registers: Vec<u64>,
        /// Snapshot of the stack at capture point
        stack_snapshot: Vec<u8>,
    },

    /// Placeholder for not-yet-implemented resume mechanisms
    ///
    /// Used during incremental development - simply returns the
    /// resume value as the result.
    Stub,
}

impl ResumePoint {
    /// Extract the description/label from this resume point for debugging
    pub fn description(&self) -> Option<String> {
        match self {
            ResumePoint::InterpreterClosure { description, .. } => description.clone(),
            ResumePoint::InterpreterMultiShot { description, .. } => description.clone(),
            ResumePoint::Interpreter { description } => Some(description.clone()),
            ResumePoint::Jit { .. } => None,
            ResumePoint::Stub => None,
        }
    }

    /// Create a new interpreter resume point with a closure.
    ///
    /// The closure captures the interpreter state and will be called
    /// when the continuation is resumed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let resume_point = ResumePoint::interpreter_closure(
    ///     |value| {
    ///         // Continue computation with the provided value
    ///         Ok(Value::Int(value.as_int().unwrap() * 2))
    ///     },
    ///     Some("doubling continuation"),
    /// );
    /// ```
    pub fn interpreter_closure<F>(resume_fn: F, description: Option<&str>) -> Self
    where
        F: FnOnce(Value) -> Result<Value, ContinuationError> + 'static,
    {
        ResumePoint::InterpreterClosure {
            resume_fn: Box::new(resume_fn),
            description: description.map(String::from),
        }
    }

    /// Create a new interpreter resume point with a multi-shot closure.
    ///
    /// Unlike `interpreter_closure`, this creates a continuation that can be
    /// resumed multiple times. The closure is wrapped in an `Arc` so it can
    /// be cloned on each resume.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let resume_point = ResumePoint::interpreter_multi_shot(
    ///     |value| {
    ///         // This can be called multiple times
    ///         Ok(Value::Int(value.as_int().unwrap() * 2))
    ///     },
    ///     Some("multi-shot doubling continuation"),
    /// );
    /// ```
    pub fn interpreter_multi_shot<F>(resume_fn: F, description: Option<&str>) -> Self
    where
        F: Fn(Value) -> Result<Value, ContinuationError> + Send + Sync + 'static,
    {
        ResumePoint::InterpreterMultiShot {
            resume_fn: Arc::new(resume_fn),
            description: description.map(String::from),
        }
    }

    /// Create a new interpreter resume point (description only, not executable)
    pub fn interpreter(description: impl Into<String>) -> Self {
        ResumePoint::Interpreter {
            description: description.into(),
        }
    }

    /// Create a new JIT resume point
    pub fn jit(return_address: usize, saved_registers: Vec<u64>, stack_snapshot: Vec<u8>) -> Self {
        ResumePoint::Jit {
            return_address,
            saved_registers,
            stack_snapshot,
        }
    }

    /// Check if this is a stub resume point
    pub fn is_stub(&self) -> bool {
        matches!(self, ResumePoint::Stub)
    }

    /// Check if this resume point is executable (can actually resume)
    pub fn is_executable(&self) -> bool {
        matches!(
            self,
            ResumePoint::InterpreterClosure { .. }
                | ResumePoint::InterpreterMultiShot { .. }
                | ResumePoint::Stub
        )
    }

    /// Check if this is a multi-shot resume point
    pub fn is_multi_shot(&self) -> bool {
        matches!(self, ResumePoint::InterpreterMultiShot { .. })
    }
}

impl fmt::Debug for ResumePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResumePoint::InterpreterClosure { description, .. } => f
                .debug_struct("InterpreterClosure")
                .field("description", description)
                .field("resume_fn", &"<one-shot closure>")
                .finish(),
            ResumePoint::InterpreterMultiShot { description, .. } => f
                .debug_struct("InterpreterMultiShot")
                .field("description", description)
                .field("resume_fn", &"<multi-shot closure>")
                .finish(),
            ResumePoint::Interpreter { description } => f
                .debug_struct("Interpreter")
                .field("description", description)
                .finish(),
            ResumePoint::Jit {
                return_address,
                saved_registers,
                stack_snapshot,
            } => f
                .debug_struct("Jit")
                .field("return_address", &format_args!("0x{:x}", return_address))
                .field("saved_registers", &saved_registers.len())
                .field("stack_snapshot", &stack_snapshot.len())
                .finish(),
            ResumePoint::Stub => write!(f, "Stub"),
        }
    }
}

/// Captured continuation representing a suspended computation
///
/// When an effect operation is performed, the continuation captures
/// "the rest of the computation" so that the handler can resume it
/// (possibly multiple times) with different values.
#[derive(Debug)]
pub struct CapturedContinuation {
    /// Unique identifier for this continuation
    pub id: ContinuationId,
    /// Where to resume execution
    pub resume_point: ResumePoint,
    /// Whether this continuation can be called multiple times
    pub is_multi_shot: bool,
    /// Number of times this continuation has been resumed
    pub resume_count: usize,
    /// Optional label for debugging
    pub label: Option<String>,
}

impl CapturedContinuation {
    /// Create a new one-shot continuation
    ///
    /// One-shot continuations can only be resumed once. Attempting to
    /// resume a second time will return an error. This is the most
    /// common case and is more efficient than multi-shot.
    pub fn new_one_shot(resume: ResumePoint) -> Self {
        let label = resume.description();
        Self {
            id: ContinuationId::new(),
            resume_point: resume,
            is_multi_shot: false,
            resume_count: 0,
            label,
        }
    }

    /// Create a new multi-shot continuation
    ///
    /// Multi-shot continuations can be resumed multiple times,
    /// which is needed for effects like non-determinism (amb) or
    /// backtracking search.
    pub fn new_multi_shot(resume: ResumePoint) -> Self {
        let label = resume.description();
        Self {
            id: ContinuationId::new(),
            resume_point: resume,
            is_multi_shot: true,
            resume_count: 0,
            label,
        }
    }

    /// Create a stub continuation (for testing/development)
    pub fn stub() -> Self {
        Self::new_one_shot(ResumePoint::Stub)
    }

    /// Create a stub multi-shot continuation (for testing/development)
    pub fn stub_multi_shot() -> Self {
        Self::new_multi_shot(ResumePoint::Stub)
    }

    /// Add a label to this continuation (for debugging)
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Check if this continuation has been resumed
    pub fn has_been_resumed(&self) -> bool {
        self.resume_count > 0
    }

    /// Check if this continuation can still be resumed
    pub fn can_resume(&self) -> bool {
        self.is_multi_shot || self.resume_count == 0
    }

    /// Resume the continuation with a value
    ///
    /// For one-shot continuations, this can only be called once.
    /// For multi-shot continuations, this can be called multiple times.
    ///
    /// # Errors
    ///
    /// Returns `ContinuationError::AlreadyResumed` if this is a one-shot
    /// continuation that has already been resumed.
    ///
    /// Returns `ContinuationError::NotImplemented` for resume points
    /// that are not yet fully implemented.
    pub fn resume(&mut self, value: Value) -> Result<Value, ContinuationError> {
        // Check one-shot enforcement
        if !self.is_multi_shot && self.resume_count > 0 {
            return Err(ContinuationError::AlreadyResumed {
                id: self.id,
                label: self.label.clone(),
            });
        }

        self.resume_count += 1;

        // Handle multi-shot continuations specially: clone the Arc and call without consuming
        if let ResumePoint::InterpreterMultiShot { ref resume_fn, .. } = self.resume_point {
            // Clone the Arc to allow multiple calls
            let fn_clone = Arc::clone(resume_fn);
            return fn_clone(value);
        }

        // For one-shot closures, we need to take ownership of the closure
        // (since it's FnOnce). We replace the resume_point with a Stub after
        // extracting the closure.
        let resume_point = std::mem::replace(&mut self.resume_point, ResumePoint::Stub);

        match resume_point {
            ResumePoint::InterpreterClosure { resume_fn, .. } => {
                // Execute the resume closure with the provided value (consumes the closure)
                resume_fn(value)
            }
            ResumePoint::InterpreterMultiShot { .. } => {
                // Already handled above, this branch should be unreachable
                unreachable!("InterpreterMultiShot should be handled before mem::replace")
            }
            ResumePoint::Interpreter { description } => {
                // Restore the resume point since we didn't consume it
                self.resume_point = ResumePoint::Interpreter {
                    description: description.clone(),
                };
                // This variant is not executable - it's just a description
                Err(ContinuationError::NotImplemented(format!(
                    "interpreter resume (description only): {}",
                    description
                )))
            }
            ResumePoint::Jit {
                return_address,
                saved_registers,
                stack_snapshot,
            } => {
                // Create a JIT resume context from the saved state
                let mut jit_ctx = super::jit_resume::JitResumeContext::new(
                    return_address,
                    saved_registers.clone(),
                    stack_snapshot.clone(),
                );

                // Set multi-shot if this continuation is multi-shot
                if self.is_multi_shot {
                    jit_ctx.is_multi_shot = true;
                }

                // Attempt to resume using the JIT resume infrastructure
                let result = super::jit_resume::resume_jit_continuation(&mut jit_ctx, value);

                // Restore the resume point for potential re-use (multi-shot)
                // or for debugging purposes
                self.resume_point = ResumePoint::Jit {
                    return_address,
                    saved_registers: saved_registers.clone(),
                    stack_snapshot: stack_snapshot.clone(),
                };

                result
            }
            ResumePoint::Stub => {
                // Stub implementation: just return the value directly
                // This allows testing the continuation infrastructure
                // before the full implementation is ready
                Ok(value)
            }
        }
    }
}

/// Errors that can occur when working with continuations
#[derive(Debug, Clone)]
pub enum ContinuationError {
    /// Attempted to resume a one-shot continuation more than once
    AlreadyResumed {
        id: ContinuationId,
        label: Option<String>,
    },

    /// The requested continuation was not found in the store
    NotFound(ContinuationId),

    /// Feature not yet implemented
    NotImplemented(String),
}

impl fmt::Display for ContinuationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContinuationError::AlreadyResumed { id, label } => {
                if let Some(label) = label {
                    write!(
                        f,
                        "continuation {} ({}) has already been resumed",
                        id, label
                    )
                } else {
                    write!(f, "continuation {} has already been resumed", id)
                }
            }
            ContinuationError::NotFound(id) => {
                write!(f, "continuation {} not found", id)
            }
            ContinuationError::NotImplemented(msg) => {
                write!(f, "not implemented: {}", msg)
            }
        }
    }
}

impl std::error::Error for ContinuationError {}

/// Store for managing active continuations
///
/// The continuation store holds captured continuations that are waiting
/// to be resumed. It provides operations for storing, retrieving, and
/// removing continuations.
#[derive(Debug, Default)]
pub struct ContinuationStore {
    /// Map from continuation ID to the continuation itself
    continuations: HashMap<ContinuationId, CapturedContinuation>,
}

impl ContinuationStore {
    /// Create a new empty continuation store
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a continuation and return its ID
    ///
    /// The continuation can later be retrieved using the returned ID.
    pub fn store(&mut self, cont: CapturedContinuation) -> ContinuationId {
        let id = cont.id;
        self.continuations.insert(id, cont);
        id
    }

    /// Get a reference to a continuation by ID
    pub fn get(&self, id: ContinuationId) -> Option<&CapturedContinuation> {
        self.continuations.get(&id)
    }

    /// Get a mutable reference to a continuation by ID
    pub fn get_mut(&mut self, id: ContinuationId) -> Option<&mut CapturedContinuation> {
        self.continuations.get_mut(&id)
    }

    /// Remove a continuation from the store
    ///
    /// Returns the continuation if it was found, None otherwise.
    pub fn remove(&mut self, id: ContinuationId) -> Option<CapturedContinuation> {
        self.continuations.remove(&id)
    }

    /// Check if a continuation exists in the store
    pub fn contains(&self, id: ContinuationId) -> bool {
        self.continuations.contains_key(&id)
    }

    /// Get the number of continuations in the store
    pub fn len(&self) -> usize {
        self.continuations.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.continuations.is_empty()
    }

    /// Clear all continuations from the store
    pub fn clear(&mut self) {
        self.continuations.clear();
    }

    /// Resume a continuation by ID and remove it from the store (one-shot pattern)
    ///
    /// This is a convenience method that combines get_mut, resume, and remove
    /// for the common one-shot continuation pattern.
    pub fn resume_and_remove(
        &mut self,
        id: ContinuationId,
        value: Value,
    ) -> Result<Value, ContinuationError> {
        let cont = self
            .continuations
            .get_mut(&id)
            .ok_or(ContinuationError::NotFound(id))?;

        // For one-shot, we'll remove after resuming
        let is_one_shot = !cont.is_multi_shot;
        let result = cont.resume(value)?;

        if is_one_shot {
            self.continuations.remove(&id);
        }

        Ok(result)
    }

    /// Get an iterator over all continuation IDs
    pub fn ids(&self) -> impl Iterator<Item = ContinuationId> + '_ {
        self.continuations.keys().copied()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuation_id_uniqueness() {
        let id1 = ContinuationId::new();
        let id2 = ContinuationId::new();
        let id3 = ContinuationId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_continuation_id_display() {
        let id = ContinuationId::new();
        let display = format!("{}", id);
        assert!(display.starts_with("cont_"));
    }

    #[test]
    fn test_one_shot_continuation_single_resume() {
        let mut cont = CapturedContinuation::new_one_shot(ResumePoint::Stub);

        assert!(!cont.has_been_resumed());
        assert!(cont.can_resume());

        let result = cont.resume(Value::Int(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Int(42));

        assert!(cont.has_been_resumed());
        assert!(!cont.can_resume());
    }

    #[test]
    fn test_one_shot_continuation_double_resume_fails() {
        let mut cont = CapturedContinuation::new_one_shot(ResumePoint::Stub);

        // First resume succeeds
        let result1 = cont.resume(Value::Int(1));
        assert!(result1.is_ok());

        // Second resume fails
        let result2 = cont.resume(Value::Int(2));
        assert!(matches!(
            result2,
            Err(ContinuationError::AlreadyResumed { .. })
        ));
    }

    #[test]
    fn test_multi_shot_continuation_multiple_resumes() {
        let mut cont = CapturedContinuation::new_multi_shot(ResumePoint::Stub);

        assert!(cont.can_resume());

        // First resume
        let result1 = cont.resume(Value::Int(1));
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), Value::Int(1));

        // Second resume - should also succeed for multi-shot
        assert!(cont.can_resume());
        let result2 = cont.resume(Value::Int(2));
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), Value::Int(2));

        // Third resume
        let result3 = cont.resume(Value::Int(3));
        assert!(result3.is_ok());

        assert_eq!(cont.resume_count, 3);
    }

    #[test]
    fn test_continuation_with_label() {
        let cont = CapturedContinuation::stub().with_label("test_continuation");
        assert_eq!(cont.label, Some("test_continuation".to_string()));
    }

    #[test]
    fn test_continuation_store_basic_operations() {
        let mut store = ContinuationStore::new();

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        // Store a continuation
        let cont = CapturedContinuation::stub();
        let id = store.store(cont);

        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
        assert!(store.contains(id));

        // Retrieve it
        let retrieved = store.get(id);
        assert!(retrieved.is_some());

        // Remove it
        let removed = store.remove(id);
        assert!(removed.is_some());
        assert!(store.is_empty());
        assert!(!store.contains(id));
    }

    #[test]
    fn test_continuation_store_not_found() {
        let mut store = ContinuationStore::new();
        let fake_id = ContinuationId::new();

        let result = store.resume_and_remove(fake_id, Value::Unit);
        assert!(matches!(result, Err(ContinuationError::NotFound(_))));
    }

    #[test]
    fn test_continuation_store_resume_and_remove() {
        let mut store = ContinuationStore::new();

        // Store a one-shot continuation
        let cont = CapturedContinuation::new_one_shot(ResumePoint::Stub);
        let id = store.store(cont);

        assert_eq!(store.len(), 1);

        // Resume and remove
        let result = store.resume_and_remove(id, Value::Float(3.14));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Float(3.14));

        // Should be removed after resume
        assert!(store.is_empty());
    }

    #[test]
    fn test_continuation_store_multi_shot_not_removed() {
        let mut store = ContinuationStore::new();

        // Store a multi-shot continuation
        let cont = CapturedContinuation::new_multi_shot(ResumePoint::Stub);
        let id = store.store(cont);

        // Resume - should NOT be removed for multi-shot
        let result = store.resume_and_remove(id, Value::Int(1));
        assert!(result.is_ok());

        // Should still be in store
        assert!(!store.is_empty());
        assert!(store.contains(id));
    }

    #[test]
    fn test_continuation_store_clear() {
        let mut store = ContinuationStore::new();

        store.store(CapturedContinuation::stub());
        store.store(CapturedContinuation::stub());
        store.store(CapturedContinuation::stub());

        assert_eq!(store.len(), 3);

        store.clear();

        assert!(store.is_empty());
    }

    #[test]
    fn test_continuation_store_ids_iterator() {
        let mut store = ContinuationStore::new();

        let id1 = store.store(CapturedContinuation::stub());
        let id2 = store.store(CapturedContinuation::stub());

        let ids: Vec<_> = store.ids().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_resume_point_helpers() {
        let stub = ResumePoint::Stub;
        assert!(stub.is_stub());

        let interp = ResumePoint::interpreter("test continuation");
        assert!(!interp.is_stub());

        let jit = ResumePoint::jit(0x1000, vec![1, 2, 3], vec![0u8; 64]);
        assert!(!jit.is_stub());
    }

    #[test]
    fn test_continuation_error_display() {
        let id = ContinuationId::new();

        let err1 = ContinuationError::AlreadyResumed { id, label: None };
        let display1 = format!("{}", err1);
        assert!(display1.contains("already been resumed"));

        let err2 = ContinuationError::AlreadyResumed {
            id,
            label: Some("my_cont".to_string()),
        };
        let display2 = format!("{}", err2);
        assert!(display2.contains("my_cont"));

        let err3 = ContinuationError::NotFound(id);
        let display3 = format!("{}", err3);
        assert!(display3.contains("not found"));

        let err4 = ContinuationError::NotImplemented("test feature".to_string());
        let display4 = format!("{}", err4);
        assert!(display4.contains("not implemented"));
    }

    #[test]
    fn test_interpreter_resume_not_implemented() {
        let mut cont =
            CapturedContinuation::new_one_shot(ResumePoint::interpreter("test computation"));

        let result = cont.resume(Value::Unit);
        assert!(matches!(result, Err(ContinuationError::NotImplemented(_))));
    }

    #[test]
    fn test_jit_resume_returns_value() {
        // JIT resume now uses the cooperative model and returns the value
        let mut cont = CapturedContinuation::new_one_shot(ResumePoint::jit(0x1234, vec![], vec![]));

        let result = cont.resume(Value::Float(42.0));
        // The cooperative resume model returns the value as a Float
        assert!(result.is_ok());
        match result.unwrap() {
            Value::Float(f) => assert!((f - 42.0).abs() < 0.001),
            _ => panic!("Expected Float value from JIT resume"),
        }
    }

    #[test]
    fn test_jit_resume_multi_shot() {
        // Test that multi-shot JIT continuations can be resumed multiple times
        let mut cont = CapturedContinuation::new_multi_shot(ResumePoint::jit(
            0x5678,
            vec![1, 2, 3],
            vec![4, 5, 6],
        ));

        // First resume
        let result1 = cont.resume(Value::Float(1.0));
        assert!(result1.is_ok());

        // Second resume - should also work for multi-shot
        let result2 = cont.resume(Value::Float(2.0));
        assert!(result2.is_ok());

        // Third resume
        let result3 = cont.resume(Value::Float(3.0));
        assert!(result3.is_ok());

        assert_eq!(cont.resume_count, 3);
    }

    // =========================================================================
    // InterpreterClosure Tests
    // =========================================================================

    #[test]
    fn test_interpreter_closure_basic_resume() {
        // Create a continuation that doubles the input integer
        let resume_point = ResumePoint::interpreter_closure(
            |value| {
                if let Value::Int(n) = value {
                    Ok(Value::Int(n * 2))
                } else {
                    Err(ContinuationError::NotImplemented("expected Int".into()))
                }
            },
            Some("doubler"),
        );

        let mut cont = CapturedContinuation::new_one_shot(resume_point);

        let result = cont.resume(Value::Int(21));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Int(42));
    }

    #[test]
    fn test_interpreter_closure_one_shot_consumed() {
        // After one-shot resume, the closure should be consumed
        let resume_point = ResumePoint::interpreter_closure(|value| Ok(value), Some("identity"));

        let mut cont = CapturedContinuation::new_one_shot(resume_point);

        // First resume succeeds
        let result1 = cont.resume(Value::Int(1));
        assert!(result1.is_ok());

        // Second resume fails (one-shot)
        let result2 = cont.resume(Value::Int(2));
        assert!(matches!(
            result2,
            Err(ContinuationError::AlreadyResumed { .. })
        ));
    }

    #[test]
    fn test_interpreter_closure_captures_environment() {
        // Test that closures can capture values from their environment
        let multiplier = 10;
        let resume_point = ResumePoint::interpreter_closure(
            move |value| {
                if let Value::Int(n) = value {
                    Ok(Value::Int(n * multiplier))
                } else {
                    Err(ContinuationError::NotImplemented("expected Int".into()))
                }
            },
            Some("multiplier"),
        );

        let mut cont = CapturedContinuation::new_one_shot(resume_point);

        let result = cont.resume(Value::Int(5));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Int(50));
    }

    #[test]
    fn test_interpreter_closure_error_propagation() {
        // Test that errors from the closure are properly propagated
        let resume_point = ResumePoint::interpreter_closure(
            |_value| {
                Err(ContinuationError::NotImplemented(
                    "intentional error".into(),
                ))
            },
            Some("error generator"),
        );

        let mut cont = CapturedContinuation::new_one_shot(resume_point);

        let result = cont.resume(Value::Unit);
        assert!(
            matches!(result, Err(ContinuationError::NotImplemented(msg)) if msg.contains("intentional"))
        );
    }

    #[test]
    fn test_interpreter_closure_is_executable() {
        let closure_point = ResumePoint::interpreter_closure(|v| Ok(v), None);
        assert!(closure_point.is_executable());
        assert!(!closure_point.is_stub());

        let desc_only = ResumePoint::interpreter("description only");
        assert!(!desc_only.is_executable());
        assert!(!desc_only.is_stub());

        let jit = ResumePoint::jit(0x1000, vec![], vec![]);
        assert!(!jit.is_executable());

        let stub = ResumePoint::Stub;
        assert!(stub.is_executable());
        assert!(stub.is_stub());
    }

    #[test]
    fn test_interpreter_closure_debug_format() {
        let resume_point = ResumePoint::interpreter_closure(|v| Ok(v), Some("test closure"));

        let debug = format!("{:?}", resume_point);
        assert!(debug.contains("InterpreterClosure"));
        assert!(debug.contains("test closure"));
        assert!(debug.contains("<one-shot closure>"));
    }

    #[test]
    fn test_interpreter_closure_in_store() {
        let mut store = ContinuationStore::new();

        // Store a continuation with a closure
        let resume_point = ResumePoint::interpreter_closure(
            |value| {
                if let Value::Int(n) = value {
                    Ok(Value::String(format!("Result: {}", n)))
                } else {
                    Ok(Value::String("unknown".into()))
                }
            },
            Some("formatter"),
        );

        let cont = CapturedContinuation::new_one_shot(resume_point);
        let id = store.store(cont);

        // Resume through the store
        let result = store.resume_and_remove(id, Value::Int(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("Result: 42".into()));

        // Should be removed (one-shot)
        assert!(store.is_empty());
    }

    // =========================================================================
    // InterpreterMultiShot Tests
    // =========================================================================

    #[test]
    fn test_interpreter_multi_shot_basic_resume() {
        // Create a multi-shot continuation that triples the input integer
        let resume_point = ResumePoint::interpreter_multi_shot(
            |value| {
                if let Value::Int(n) = value {
                    Ok(Value::Int(n * 3))
                } else {
                    Err(ContinuationError::NotImplemented("expected Int".into()))
                }
            },
            Some("tripler"),
        );

        let mut cont = CapturedContinuation::new_multi_shot(resume_point);

        // First resume
        let result1 = cont.resume(Value::Int(7));
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), Value::Int(21));

        // Second resume - should also work for multi-shot
        let result2 = cont.resume(Value::Int(14));
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), Value::Int(42));

        // Third resume
        let result3 = cont.resume(Value::Int(100));
        assert!(result3.is_ok());
        assert_eq!(result3.unwrap(), Value::Int(300));

        assert_eq!(cont.resume_count, 3);
    }

    #[test]
    fn test_interpreter_multi_shot_resume_point_preserved() {
        // Verify that multi-shot resume point is NOT replaced with Stub after resume
        let resume_point = ResumePoint::interpreter_multi_shot(|value| Ok(value), Some("identity"));

        let mut cont = CapturedContinuation::new_multi_shot(resume_point);

        // Resume multiple times
        for i in 0..5 {
            let result = cont.resume(Value::Int(i));
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), Value::Int(i));
        }

        // Resume point should still be InterpreterMultiShot (not Stub)
        assert!(cont.resume_point.is_multi_shot());
        assert!(!cont.resume_point.is_stub());
    }

    #[test]
    fn test_interpreter_multi_shot_with_captured_state() {
        use std::sync::atomic::{AtomicI64, Ordering};

        // Test that closures can capture state and it's shared across resumes
        let counter = Arc::new(AtomicI64::new(0));
        let counter_clone = Arc::clone(&counter);

        let resume_point = ResumePoint::interpreter_multi_shot(
            move |value| {
                if let Value::Int(n) = value {
                    let old_count = counter_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(Value::Int(n + old_count))
                } else {
                    Ok(value)
                }
            },
            Some("stateful"),
        );

        let mut cont = CapturedContinuation::new_multi_shot(resume_point);

        // Each resume should see the updated counter
        let result1 = cont.resume(Value::Int(100));
        assert_eq!(result1.unwrap(), Value::Int(100)); // 100 + 0

        let result2 = cont.resume(Value::Int(100));
        assert_eq!(result2.unwrap(), Value::Int(101)); // 100 + 1

        let result3 = cont.resume(Value::Int(100));
        assert_eq!(result3.unwrap(), Value::Int(102)); // 100 + 2

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_interpreter_multi_shot_error_recovery() {
        // Multi-shot continuations can recover from errors
        let resume_point = ResumePoint::interpreter_multi_shot(
            |value| match value {
                Value::Int(n) if n == 0 => {
                    Err(ContinuationError::NotImplemented("divide by zero".into()))
                }
                Value::Int(n) => Ok(Value::Int(100 / n)),
                _ => Ok(value),
            },
            Some("divider"),
        );

        let mut cont = CapturedContinuation::new_multi_shot(resume_point);

        // Success
        let result1 = cont.resume(Value::Int(5));
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), Value::Int(20)); // 100 / 5

        // Error
        let result2 = cont.resume(Value::Int(0));
        assert!(result2.is_err());

        // Resume point should still work after error
        assert!(cont.can_resume());

        // Success again
        let result3 = cont.resume(Value::Int(4));
        assert!(result3.is_ok());
        assert_eq!(result3.unwrap(), Value::Int(25)); // 100 / 4
    }

    #[test]
    fn test_interpreter_multi_shot_is_executable_and_multi_shot() {
        let multi_shot_point = ResumePoint::interpreter_multi_shot(|v| Ok(v), Some("test"));
        assert!(multi_shot_point.is_executable());
        assert!(multi_shot_point.is_multi_shot());
        assert!(!multi_shot_point.is_stub());

        let one_shot_point = ResumePoint::interpreter_closure(|v| Ok(v), Some("test"));
        assert!(one_shot_point.is_executable());
        assert!(!one_shot_point.is_multi_shot());
        assert!(!one_shot_point.is_stub());
    }

    #[test]
    fn test_interpreter_multi_shot_debug_format() {
        let resume_point =
            ResumePoint::interpreter_multi_shot(|v| Ok(v), Some("debug test multi-shot"));

        let debug = format!("{:?}", resume_point);
        assert!(debug.contains("InterpreterMultiShot"));
        assert!(debug.contains("debug test multi-shot"));
        assert!(debug.contains("multi-shot closure"));
    }

    #[test]
    fn test_interpreter_multi_shot_in_store_not_removed() {
        let mut store = ContinuationStore::new();

        // Store a multi-shot continuation
        let resume_point = ResumePoint::interpreter_multi_shot(
            |value| {
                if let Value::Int(n) = value {
                    Ok(Value::Int(n * 2))
                } else {
                    Ok(value)
                }
            },
            Some("doubler"),
        );

        let cont = CapturedContinuation::new_multi_shot(resume_point);
        let id = store.store(cont);

        // Resume multiple times through the store
        let result1 = store.resume_and_remove(id, Value::Int(5));
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), Value::Int(10));
        assert!(!store.is_empty()); // Should NOT be removed

        let result2 = store.resume_and_remove(id, Value::Int(10));
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), Value::Int(20));
        assert!(!store.is_empty()); // Should NOT be removed

        let result3 = store.resume_and_remove(id, Value::Int(21));
        assert!(result3.is_ok());
        assert_eq!(result3.unwrap(), Value::Int(42));
        assert!(!store.is_empty()); // Should NOT be removed
    }

    #[test]
    fn test_interpreter_multi_shot_vs_one_shot_in_store() {
        let mut store = ContinuationStore::new();

        // One-shot continuation
        let one_shot_point = ResumePoint::interpreter_closure(|value| Ok(value), Some("one-shot"));
        let one_shot_cont = CapturedContinuation::new_one_shot(one_shot_point);
        let one_shot_id = store.store(one_shot_cont);

        // Multi-shot continuation
        let multi_shot_point =
            ResumePoint::interpreter_multi_shot(|value| Ok(value), Some("multi-shot"));
        let multi_shot_cont = CapturedContinuation::new_multi_shot(multi_shot_point);
        let multi_shot_id = store.store(multi_shot_cont);

        assert_eq!(store.len(), 2);

        // Resume one-shot
        let _ = store.resume_and_remove(one_shot_id, Value::Int(1));
        assert_eq!(store.len(), 1); // One-shot removed
        assert!(!store.contains(one_shot_id));

        // Resume multi-shot multiple times
        let _ = store.resume_and_remove(multi_shot_id, Value::Int(1));
        assert_eq!(store.len(), 1); // Multi-shot still there
        assert!(store.contains(multi_shot_id));

        let _ = store.resume_and_remove(multi_shot_id, Value::Int(2));
        assert_eq!(store.len(), 1); // Multi-shot still there
        assert!(store.contains(multi_shot_id));
    }
}
