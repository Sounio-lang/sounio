//! Allocation Effect Handler
//!
//! This module implements the `Alloc` effect handler for tracked memory allocation.
//! The Alloc effect enables safe memory management through algebraic effects,
//! providing allocation tracking and automatic cleanup semantics.
//!
//! # Effect Operations
//!
//! | Operation        | Arguments                    | Return Type | Description                      |
//! |-----------------|------------------------------|-------------|----------------------------------|
//! | `alloc`         | `(size: I64)`               | `I64`       | Allocate bytes, return pointer   |
//! | `alloc_array`   | `(count: I64, elem_size: I64)` | `I64`    | Allocate array, return pointer   |
//! | `dealloc`       | `(pointer: I64)`            | `Unit`      | Free allocation                  |
//! | `realloc`       | `(pointer: I64, new_size: I64)` | `I64`   | Resize allocation                |
//! | `size`          | `(pointer: I64)`            | `I64`       | Get allocation size              |
//! | `is_valid`      | `(pointer: I64)`            | `Bool`      | Check if pointer is valid        |
//! | `allocated_bytes` | `()`                      | `I64`       | Total allocated bytes            |
//!
//! # Epistemic Impact
//!
//! Allocation operations do not degrade epistemic confidence. Memory allocation
//! is a deterministic operation that does not introduce uncertainty about values.
//!
//! # Safety Model
//!
//! This handler tracks all allocations and validates pointer operations:
//! - Double-free detection via allocation tracking
//! - Use-after-free detection via validity checks
//! - Memory leak detection via allocated_bytes tracking
//!
//! # Example
//!
//! ```text
//! // Sounio code using Alloc effect:
//! fn create_buffer(size: i64) -> i64 with Alloc {
//!     let ptr = alloc(size)
//!     // ... use buffer ...
//!     dealloc(ptr)
//!     0
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use std::sync::OnceLock;

/// Key prefix for storing allocation metadata in HandlerState
const ALLOC_PREFIX: &str = "__alloc_";

/// Key for storing the next pointer ID
const NEXT_PTR_KEY: &str = "__alloc_next_ptr";

/// Key for storing total allocated bytes
const TOTAL_BYTES_KEY: &str = "__alloc_total_bytes";

/// Static operation specifications for AllocHandler
static ALLOC_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_alloc_operations() -> &'static [OperationSpec] {
    ALLOC_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("alloc", "I64").with_params(vec!["I64"]),
            OperationSpec::new("alloc_array", "I64").with_params(vec!["I64", "I64"]),
            OperationSpec::new("dealloc", "Unit").with_params(vec!["I64"]),
            OperationSpec::new("realloc", "I64").with_params(vec!["I64", "I64"]),
            OperationSpec::new("size", "I64").with_params(vec!["I64"]),
            OperationSpec::new("is_valid", "Bool").with_params(vec!["I64"]),
            OperationSpec::new("allocated_bytes", "I64").with_params(vec![]),
        ]
    })
}

/// Handler for the Alloc (memory allocation) effect
///
/// AllocHandler provides tracked memory allocation through algebraic effects.
/// Allocations are tracked in `HandlerState.named_state` with metadata including:
/// - Allocation size
/// - Validity status
///
/// # Pointer Representation
///
/// Pointers are represented as unique i64 values. In a real runtime, these would
/// be actual memory addresses. For the effect handler, they serve as identifiers
/// into the allocation tracking table.
///
/// # Thread Safety
///
/// AllocHandler is stateless - all allocation state is stored in HandlerState.
/// This enables safe concurrent use when each thread has its own HandlerState.
#[derive(Debug)]
pub struct AllocHandler {
    _private: (),
}

impl Default for AllocHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocHandler {
    /// Create a new AllocHandler
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Get the allocation key for a pointer
    fn alloc_key(ptr: i64) -> String {
        format!("{}{}", ALLOC_PREFIX, ptr)
    }

    /// Get the next pointer ID and increment
    fn next_pointer(state: &mut HandlerState) -> i64 {
        let current = match state.named_state.get(NEXT_PTR_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 1, // Start at 1 so 0 can be null
        };
        state
            .named_state
            .insert(NEXT_PTR_KEY.to_string(), Value::Int(current + 1));
        current
    }

    /// Get total allocated bytes
    fn get_total_bytes(state: &HandlerState) -> i64 {
        match state.named_state.get(TOTAL_BYTES_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        }
    }

    /// Set total allocated bytes
    fn set_total_bytes(state: &mut HandlerState, bytes: i64) {
        state
            .named_state
            .insert(TOTAL_BYTES_KEY.to_string(), Value::Int(bytes));
    }

    /// Handle alloc operation
    fn handle_alloc(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "alloc",
                "alloc requires a size argument",
            ));
        }

        let size = match &args[0] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "alloc",
                    format!("expected I64 for size, got {:?}", args[0].type_name()),
                ));
            }
        };

        if size < 0 {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "alloc",
                format!("allocation size must be non-negative, got {}", size),
            ));
        }

        let ptr = Self::next_pointer(state);
        let key = Self::alloc_key(ptr);

        // Store allocation metadata as the size
        state.named_state.insert(key, Value::Int(size));

        // Update total bytes
        let total = Self::get_total_bytes(state) + size;
        Self::set_total_bytes(state, total);

        HandlerResult::Resume(Value::Int(ptr))
    }

    /// Handle alloc_array operation
    fn handle_alloc_array(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "alloc_array",
                format!(
                    "alloc_array requires 2 arguments (count, elem_size), got {}",
                    args.len()
                ),
            ));
        }

        let count = match &args[0] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "alloc_array",
                    format!("expected I64 for count, got {:?}", args[0].type_name()),
                ));
            }
        };

        let elem_size = match &args[1] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "alloc_array",
                    format!("expected I64 for elem_size, got {:?}", args[1].type_name()),
                ));
            }
        };

        if count < 0 || elem_size < 0 {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "alloc_array",
                format!(
                    "count and elem_size must be non-negative, got count={}, elem_size={}",
                    count, elem_size
                ),
            ));
        }

        // Check for overflow
        let total_size = count.checked_mul(elem_size).ok_or_else(|| {
            HandlerError::new(
                "Alloc",
                "alloc_array",
                format!(
                    "allocation size overflow: {} * {} exceeds i64 range",
                    count, elem_size
                ),
            )
        });

        let total_size = match total_size {
            Ok(size) => size,
            Err(e) => return HandlerResult::Abort(e),
        };

        let ptr = Self::next_pointer(state);
        let key = Self::alloc_key(ptr);

        state.named_state.insert(key, Value::Int(total_size));

        let total = Self::get_total_bytes(state) + total_size;
        Self::set_total_bytes(state, total);

        HandlerResult::Resume(Value::Int(ptr))
    }

    /// Handle dealloc operation
    fn handle_dealloc(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "dealloc",
                "dealloc requires a pointer argument",
            ));
        }

        let ptr = match &args[0] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "dealloc",
                    format!("expected I64 for pointer, got {:?}", args[0].type_name()),
                ));
            }
        };

        let key = Self::alloc_key(ptr);

        // Check if allocation exists (detect double-free)
        let size = match state.named_state.get(&key) {
            Some(Value::Int(n)) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "dealloc",
                    format!(
                        "invalid dealloc: pointer {} was not allocated or already freed",
                        ptr
                    ),
                ));
            }
        };

        // Remove allocation
        state.named_state.remove(&key);

        // Update total bytes
        let total = Self::get_total_bytes(state) - size;
        Self::set_total_bytes(state, total);

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle realloc operation
    fn handle_realloc(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "realloc",
                format!(
                    "realloc requires 2 arguments (pointer, new_size), got {}",
                    args.len()
                ),
            ));
        }

        let ptr = match &args[0] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "realloc",
                    format!("expected I64 for pointer, got {:?}", args[0].type_name()),
                ));
            }
        };

        let new_size = match &args[1] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "realloc",
                    format!("expected I64 for new_size, got {:?}", args[1].type_name()),
                ));
            }
        };

        if new_size < 0 {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "realloc",
                format!("new_size must be non-negative, got {}", new_size),
            ));
        }

        let key = Self::alloc_key(ptr);

        // Check if allocation exists
        let old_size = match state.named_state.get(&key) {
            Some(Value::Int(n)) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "realloc",
                    format!("invalid realloc: pointer {} was not allocated", ptr),
                ));
            }
        };

        // For simplicity, realloc returns the same pointer with updated size
        // In a real implementation, this might return a new pointer
        state.named_state.insert(key, Value::Int(new_size));

        // Update total bytes
        let total = Self::get_total_bytes(state) - old_size + new_size;
        Self::set_total_bytes(state, total);

        HandlerResult::Resume(Value::Int(ptr))
    }

    /// Handle size operation
    fn handle_size(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "size",
                "size requires a pointer argument",
            ));
        }

        let ptr = match &args[0] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "size",
                    format!("expected I64 for pointer, got {:?}", args[0].type_name()),
                ));
            }
        };

        let key = Self::alloc_key(ptr);

        match state.named_state.get(&key) {
            Some(Value::Int(size)) => HandlerResult::Resume(Value::Int(*size)),
            _ => HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "size",
                format!("invalid pointer: {} was not allocated", ptr),
            )),
        }
    }

    /// Handle is_valid operation
    fn handle_is_valid(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Alloc",
                "is_valid",
                "is_valid requires a pointer argument",
            ));
        }

        let ptr = match &args[0] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Alloc",
                    "is_valid",
                    format!("expected I64 for pointer, got {:?}", args[0].type_name()),
                ));
            }
        };

        let key = Self::alloc_key(ptr);
        let is_valid = state.named_state.contains_key(&key);

        HandlerResult::Resume(Value::Bool(is_valid))
    }

    /// Handle allocated_bytes operation
    fn handle_allocated_bytes(&self, state: &HandlerState) -> HandlerResult {
        let total = Self::get_total_bytes(state);
        HandlerResult::Resume(Value::Int(total))
    }
}

impl HandlerCapability for AllocHandler {
    fn effect_name(&self) -> &str {
        "Alloc"
    }

    fn handler_name(&self) -> &str {
        "AllocHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_alloc_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "alloc" => self.handle_alloc(args, state),
            "alloc_array" => self.handle_alloc_array(args, state),
            "dealloc" => self.handle_dealloc(args, state),
            "realloc" => self.handle_realloc(args, state),
            "size" => self.handle_size(args, state),
            "is_valid" => self.handle_is_valid(args, state),
            "allocated_bytes" => self.handle_allocated_bytes(state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Alloc",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        false
    }

    fn epistemic_impact(&self, _operation: &str) -> EpistemicImpact {
        // Allocation operations do not degrade confidence
        // Memory allocation is deterministic and introduces no uncertainty
        EpistemicImpact::none()
    }

    fn operation_linearity(&self, operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        // Memory allocation/deallocation modify the heap - must be one-shot
        // to prevent double-frees or allocation leaks
        match operation {
            "alloc" | "alloc_array" | "dealloc" | "realloc" => Linearity::ExactlyOnce,
            // Read-only queries are safe for multi-shot
            "size" | "is_valid" | "allocated_bytes" => Linearity::MultiShot,
            _ => Linearity::ExactlyOnce,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_state() -> HandlerState {
        HandlerState::new()
    }

    fn cont() -> Continuation {
        Continuation::new()
    }

    #[test]
    fn test_handler_identity() {
        let handler = AllocHandler::new();
        assert_eq!(handler.effect_name(), "Alloc");
        assert_eq!(handler.handler_name(), "AllocHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = AllocHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"alloc"));
        assert!(names.contains(&"alloc_array"));
        assert!(names.contains(&"dealloc"));
        assert!(names.contains(&"realloc"));
        assert!(names.contains(&"size"));
        assert!(names.contains(&"is_valid"));
        assert!(names.contains(&"allocated_bytes"));
    }

    #[test]
    fn test_does_not_support_multi_shot() {
        let handler = AllocHandler::new();
        assert!(!handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact_is_identity() {
        let handler = AllocHandler::new();

        // All Alloc operations should have no epistemic impact (confidence factor = 1.0)
        let alloc_impact = handler.epistemic_impact("alloc");
        assert!((alloc_impact.confidence_factor - 1.0).abs() < 0.001);
        assert!(alloc_impact.provenance_tag.is_none());
        assert!(!alloc_impact.crosses_firewall);

        let dealloc_impact = handler.epistemic_impact("dealloc");
        assert!((dealloc_impact.confidence_factor - 1.0).abs() < 0.001);

        let realloc_impact = handler.epistemic_impact("realloc");
        assert!((realloc_impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_alloc_basic() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("alloc", &[Value::Int(100)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(ptr)) => {
                assert!(ptr > 0, "pointer should be positive");
            }
            other => panic!("expected Resume(Int), got {:?}", other),
        }
    }

    #[test]
    fn test_alloc_requires_argument() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("alloc", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("requires a size argument"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_alloc_type_check() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("alloc", &[Value::String("100".into())], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("expected I64"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_alloc_negative_size() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("alloc", &[Value::Int(-10)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("non-negative"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_alloc_array_basic() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "alloc_array",
            &[Value::Int(10), Value::Int(8)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Int(ptr)) => {
                assert!(ptr > 0);
                // Check that size is count * elem_size
                let size_result = handler.handle("size", &[Value::Int(ptr)], cont(), &mut state);
                match size_result {
                    HandlerResult::Resume(Value::Int(size)) => {
                        assert_eq!(size, 80); // 10 * 8
                    }
                    other => panic!("expected size=80, got {:?}", other),
                }
            }
            other => panic!("expected Resume(Int), got {:?}", other),
        }
    }

    #[test]
    fn test_alloc_array_requires_two_args() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("alloc_array", &[Value::Int(10)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("requires 2 arguments"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_dealloc_basic() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        // Allocate first
        let alloc_result = handler.handle("alloc", &[Value::Int(100)], cont(), &mut state);
        let ptr = match alloc_result {
            HandlerResult::Resume(Value::Int(p)) => p,
            _ => panic!("alloc failed"),
        };

        // Dealloc
        let result = handler.handle("dealloc", &[Value::Int(ptr)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        // Verify pointer is no longer valid
        let valid_result = handler.handle("is_valid", &[Value::Int(ptr)], cont(), &mut state);
        match valid_result {
            HandlerResult::Resume(Value::Bool(false)) => {}
            other => panic!("expected is_valid=false, got {:?}", other),
        }
    }

    #[test]
    fn test_dealloc_double_free() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        // Allocate
        let ptr = match handler.handle("alloc", &[Value::Int(100)], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(p)) => p,
            _ => panic!("alloc failed"),
        };

        // First dealloc succeeds
        handler.handle("dealloc", &[Value::Int(ptr)], cont(), &mut state);

        // Second dealloc fails (double-free)
        let result = handler.handle("dealloc", &[Value::Int(ptr)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("not allocated or already freed"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_dealloc_invalid_pointer() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("dealloc", &[Value::Int(9999)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("not allocated"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_realloc_basic() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        // Allocate 100 bytes
        let ptr = match handler.handle("alloc", &[Value::Int(100)], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(p)) => p,
            _ => panic!("alloc failed"),
        };

        // Realloc to 200 bytes
        let result = handler.handle(
            "realloc",
            &[Value::Int(ptr), Value::Int(200)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Int(new_ptr)) => {
                assert_eq!(new_ptr, ptr, "realloc should return same pointer");
                // Check new size
                let size_result =
                    handler.handle("size", &[Value::Int(new_ptr)], cont(), &mut state);
                match size_result {
                    HandlerResult::Resume(Value::Int(size)) => {
                        assert_eq!(size, 200);
                    }
                    _ => panic!("size check failed"),
                }
            }
            other => panic!("expected Resume(Int), got {:?}", other),
        }
    }

    #[test]
    fn test_realloc_invalid_pointer() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "realloc",
            &[Value::Int(9999), Value::Int(200)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("was not allocated"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_size_operation() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let ptr = match handler.handle("alloc", &[Value::Int(256)], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(p)) => p,
            _ => panic!("alloc failed"),
        };

        let result = handler.handle("size", &[Value::Int(ptr)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(size)) => {
                assert_eq!(size, 256);
            }
            other => panic!("expected Resume(Int(256)), got {:?}", other),
        }
    }

    #[test]
    fn test_size_invalid_pointer() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("size", &[Value::Int(9999)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("was not allocated"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_is_valid_true() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let ptr = match handler.handle("alloc", &[Value::Int(100)], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(p)) => p,
            _ => panic!("alloc failed"),
        };

        let result = handler.handle("is_valid", &[Value::Int(ptr)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Bool(true)) => {}
            other => panic!("expected Resume(Bool(true)), got {:?}", other),
        }
    }

    #[test]
    fn test_is_valid_false() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("is_valid", &[Value::Int(9999)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Bool(false)) => {}
            other => panic!("expected Resume(Bool(false)), got {:?}", other),
        }
    }

    #[test]
    fn test_allocated_bytes_tracking() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        // Initially 0
        let result = handler.handle("allocated_bytes", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Int(0)) => {}
            other => panic!("expected 0, got {:?}", other),
        }

        // Allocate 100
        let ptr1 = match handler.handle("alloc", &[Value::Int(100)], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(p)) => p,
            _ => panic!("alloc failed"),
        };

        let result = handler.handle("allocated_bytes", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Int(100)) => {}
            other => panic!("expected 100, got {:?}", other),
        }

        // Allocate 50 more
        let ptr2 = match handler.handle("alloc", &[Value::Int(50)], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(p)) => p,
            _ => panic!("alloc failed"),
        };

        let result = handler.handle("allocated_bytes", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Int(150)) => {}
            other => panic!("expected 150, got {:?}", other),
        }

        // Dealloc first
        handler.handle("dealloc", &[Value::Int(ptr1)], cont(), &mut state);

        let result = handler.handle("allocated_bytes", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Int(50)) => {}
            other => panic!("expected 50, got {:?}", other),
        }

        // Dealloc second
        handler.handle("dealloc", &[Value::Int(ptr2)], cont(), &mut state);

        let result = handler.handle("allocated_bytes", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Int(0)) => {}
            other => panic!("expected 0, got {:?}", other),
        }
    }

    #[test]
    fn test_realloc_updates_total_bytes() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        // Allocate 100
        let ptr = match handler.handle("alloc", &[Value::Int(100)], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(p)) => p,
            _ => panic!("alloc failed"),
        };

        // Realloc to 200 (+100 bytes)
        handler.handle(
            "realloc",
            &[Value::Int(ptr), Value::Int(200)],
            cont(),
            &mut state,
        );

        let result = handler.handle("allocated_bytes", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Int(200)) => {}
            other => panic!("expected 200, got {:?}", other),
        }

        // Realloc to 50 (-150 bytes)
        handler.handle(
            "realloc",
            &[Value::Int(ptr), Value::Int(50)],
            cont(),
            &mut state,
        );

        let result = handler.handle("allocated_bytes", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Int(50)) => {}
            other => panic!("expected 50, got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let result = handler.handle("unknown_op", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("unknown operation"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_multiple_allocations_unique_pointers() {
        let handler = AllocHandler::new();
        let mut state = new_state();

        let mut pointers = Vec::new();
        for i in 0..10 {
            let result = handler.handle("alloc", &[Value::Int(i * 10)], cont(), &mut state);
            match result {
                HandlerResult::Resume(Value::Int(ptr)) => {
                    assert!(!pointers.contains(&ptr), "pointers should be unique");
                    pointers.push(ptr);
                }
                _ => panic!("alloc failed"),
            }
        }

        assert_eq!(pointers.len(), 10);
    }

    #[test]
    fn test_default_impl() {
        let handler = AllocHandler::default();
        assert_eq!(handler.effect_name(), "Alloc");
    }
}
