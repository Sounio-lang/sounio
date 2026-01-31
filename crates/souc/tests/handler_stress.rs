//! Phase D.5: Production Testing - Handler Stress Tests
//!
//! Tests for concurrent dispatch, high-volume operations, and resource cleanup.

use sounio::effects::handler_capability::{
    Continuation, ContinuationId, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use sounio::interp::Value;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// Mock handler that tracks call counts
#[derive(Debug)]
struct CountingHandler {
    calls: AtomicU32,
}

impl CountingHandler {
    fn new() -> Self {
        Self {
            calls: AtomicU32::new(0),
        }
    }
}

impl HandlerCapability for CountingHandler {
    fn effect_name(&self) -> &str {
        "Test"
    }

    fn handler_name(&self) -> &str {
        "CountingHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        &[]
    }

    fn handle(
        &self,
        operation: &str,
        _args: &[Value],
        _continuation: Continuation,
        _state: &mut HandlerState,
    ) -> HandlerResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match operation {
            "get" => HandlerResult::Resume(Value::Int(self.calls.load(Ordering::SeqCst) as i64)),
            _ => HandlerResult::Resume(Value::Unit),
        }
    }
}

/// Test high-volume sequential handler dispatch
#[test]
fn test_high_volume_sequential_dispatch() {
    let handler = CountingHandler::new();
    let mut state = HandlerState::new();

    for _ in 0..1000 {
        let cont = Continuation::new();
        let result = handler.handle("increment", &[], cont, &mut state);
        assert!(matches!(result, HandlerResult::Resume(Value::Unit)));
    }

    assert_eq!(handler.calls.load(Ordering::SeqCst), 1000);
}

/// Test concurrent handler dispatch from multiple threads
#[test]
fn test_concurrent_handler_dispatch() {
    let handler = Arc::new(CountingHandler::new());
    let num_threads = 10;
    let calls_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let handler = Arc::clone(&handler);
            thread::spawn(move || {
                for _ in 0..calls_per_thread {
                    let mut state = HandlerState::new();
                    let cont = Continuation::new();
                    let result = handler.handle("increment", &[], cont, &mut state);
                    assert!(matches!(result, HandlerResult::Resume(_)));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        handler.calls.load(Ordering::SeqCst),
        (num_threads * calls_per_thread) as u32
    );
}

/// Test that handler state is isolated per invocation
#[test]
fn test_handler_state_isolation() {
    #[derive(Debug)]
    struct StatefulHandler;

    impl HandlerCapability for StatefulHandler {
        fn effect_name(&self) -> &str {
            "Stateful"
        }
        fn handler_name(&self) -> &str {
            "StatefulHandler"
        }
        fn operations(&self) -> &[OperationSpec] {
            &[]
        }
        fn handle(
            &self,
            _operation: &str,
            _args: &[Value],
            _continuation: Continuation,
            state: &mut HandlerState,
        ) -> HandlerResult {
            // Increment a counter in state
            let count = match state.named_state.get("count") {
                Some(Value::Int(n)) => *n + 1,
                _ => 1,
            };
            state
                .named_state
                .insert("count".to_string(), Value::Int(count));
            HandlerResult::Resume(Value::Int(count))
        }
    }

    let handler = StatefulHandler;

    // Each invocation with fresh state should start at 1
    for _ in 0..10 {
        let mut state = HandlerState::new();
        let cont = Continuation::new();
        match handler.handle("inc", &[], cont, &mut state) {
            HandlerResult::Resume(Value::Int(1)) => {}
            other => panic!("Expected Resume(Int(1)), got {:?}", other),
        }
    }

    // Shared state should accumulate
    let mut shared_state = HandlerState::new();
    for i in 1..=5 {
        let cont = Continuation::new();
        match handler.handle("inc", &[], cont, &mut shared_state) {
            HandlerResult::Resume(Value::Int(n)) => assert_eq!(n, i),
            other => panic!("Expected Resume(Int({})), got {:?}", i, other),
        }
    }
}

/// Test continuation ID uniqueness under load
#[test]
fn test_continuation_id_uniqueness_under_load() {
    let mut ids: HashSet<u64> = HashSet::new();

    for _ in 0..1000 {
        let cont = Continuation::new();
        let ContinuationId(id) = cont.id();
        assert!(ids.insert(id), "Duplicate continuation ID: {}", id);
    }

    assert_eq!(ids.len(), 1000);
}

/// Test concurrent continuation ID uniqueness
#[test]
fn test_concurrent_continuation_id_uniqueness() {
    let ids = Arc::new(Mutex::new(Vec::new()));
    let num_threads = 10;
    let ids_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let ids = Arc::clone(&ids);
            thread::spawn(move || {
                let mut local_ids = Vec::new();
                for _ in 0..ids_per_thread {
                    let cont = Continuation::new();
                    let ContinuationId(id) = cont.id();
                    local_ids.push(id);
                }
                ids.lock().unwrap().extend(local_ids);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let all_ids = ids.lock().unwrap();
    let unique: HashSet<_> = all_ids.iter().collect();
    assert_eq!(
        unique.len(),
        (num_threads * ids_per_thread) as usize,
        "Found duplicate continuation IDs"
    );
}

/// Test handler cleanup on abort
#[test]
fn test_handler_cleanup_on_abort() {
    #[derive(Debug)]
    struct ResourceHandler {
        active_resources: AtomicU32,
    }

    impl HandlerCapability for ResourceHandler {
        fn effect_name(&self) -> &str {
            "Resource"
        }
        fn handler_name(&self) -> &str {
            "ResourceHandler"
        }
        fn operations(&self) -> &[OperationSpec] {
            &[]
        }
        fn handle(
            &self,
            operation: &str,
            _args: &[Value],
            _continuation: Continuation,
            _state: &mut HandlerState,
        ) -> HandlerResult {
            match operation {
                "allocate" => {
                    self.active_resources.fetch_add(1, Ordering::SeqCst);
                    HandlerResult::Resume(Value::Unit)
                }
                "fail" => {
                    // Abort should trigger cleanup
                    let count = self.active_resources.load(Ordering::SeqCst);
                    HandlerResult::Abort(HandlerError::new(
                        "Resource",
                        "fail",
                        format!("Intentional failure with {} resources", count),
                    ))
                }
                "cleanup" => {
                    if self.active_resources.load(Ordering::SeqCst) > 0 {
                        self.active_resources.fetch_sub(1, Ordering::SeqCst);
                    }
                    HandlerResult::Resume(Value::Unit)
                }
                _ => HandlerResult::Resume(Value::Unit),
            }
        }
    }

    let handler = ResourceHandler {
        active_resources: AtomicU32::new(0),
    };
    let mut state = HandlerState::new();

    // Allocate some resources
    for _ in 0..5 {
        handler.handle("allocate", &[], Continuation::new(), &mut state);
    }
    assert_eq!(handler.active_resources.load(Ordering::SeqCst), 5);

    // Trigger abort
    let result = handler.handle("fail", &[], Continuation::new(), &mut state);
    assert!(matches!(result, HandlerResult::Abort(_)));

    // Cleanup resources
    for _ in 0..5 {
        handler.handle("cleanup", &[], Continuation::new(), &mut state);
    }
    assert_eq!(handler.active_resources.load(Ordering::SeqCst), 0);
}

/// Test epistemic confidence accumulation over many operations
#[test]
fn test_epistemic_accumulation_many_ops() {
    let mut state = HandlerState::new();

    // Record 100 IO read operations (each degrades by 0.95)
    for _ in 0..100 {
        state.record_effect("IO", "read_file");
    }

    let final_conf = state.final_confidence(1.0);

    // Expected: 0.95^100 ≈ 0.00592
    let expected = 0.95_f64.powi(100);
    assert!(
        (final_conf - expected).abs() < 0.001,
        "Expected ~{}, got {}",
        expected,
        final_conf
    );
    assert!(state.has_degradation());
}

/// Test handler with high-volume abort and recovery
#[test]
fn test_high_volume_abort_recovery() {
    #[derive(Debug)]
    struct FailingHandler {
        fail_rate: u32, // Fail every N calls
        call_count: AtomicU32,
    }

    impl HandlerCapability for FailingHandler {
        fn effect_name(&self) -> &str {
            "Failing"
        }
        fn handler_name(&self) -> &str {
            "FailingHandler"
        }
        fn operations(&self) -> &[OperationSpec] {
            &[]
        }
        fn handle(
            &self,
            _operation: &str,
            _args: &[Value],
            _continuation: Continuation,
            _state: &mut HandlerState,
        ) -> HandlerResult {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count % self.fail_rate == 0 {
                HandlerResult::Abort(HandlerError::new("Failing", "op", "Periodic failure"))
            } else {
                HandlerResult::Resume(Value::Int(count as i64))
            }
        }
    }

    let handler = FailingHandler {
        fail_rate: 10,
        call_count: AtomicU32::new(0),
    };
    let mut state = HandlerState::new();

    let mut successes = 0;
    let mut failures = 0;

    for _ in 0..100 {
        match handler.handle("op", &[], Continuation::new(), &mut state) {
            HandlerResult::Resume(_) => successes += 1,
            HandlerResult::Abort(_) => failures += 1,
            _ => {}
        }
    }

    // Should have 10 failures (0, 10, 20, ..., 90) and 90 successes
    assert_eq!(failures, 10);
    assert_eq!(successes, 90);
}

/// Test multi-shot continuation cloning under load
#[test]
fn test_multishot_continuation_under_load() {
    let mut cloned_count = 0;

    for _ in 0..100 {
        let cont = Continuation::multi_shot(1, |v| v);
        assert!(cont.is_multi_shot());

        // Should be able to clone multi-shot continuations
        if let Some(cloned) = cont.try_clone() {
            cloned_count += 1;
            assert!(cloned.is_multi_shot());
            assert_eq!(cloned.id(), cont.id());
        }
    }

    assert_eq!(cloned_count, 100);
}

/// Test single-shot continuation cannot be cloned
#[test]
fn test_singleshot_continuation_no_clone() {
    for _ in 0..100 {
        let cont = Continuation::with_resume(1, |v| v);
        assert!(!cont.is_multi_shot());
        assert!(cont.try_clone().is_none());
    }
}
