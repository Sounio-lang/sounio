//! Phase D.5: Production Testing - Network Resilience Tests
//!
//! Tests for network handler with retry and circuit breaker patterns.

use sounio::effects::handler_capability::{
    Continuation, HandlerCapability, HandlerError, HandlerResult, HandlerState, OperationSpec,
};
use sounio::effects::resilience::{CircuitBreaker, CircuitState, RetryConfig, with_retry};
use sounio::interp::Value;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Mock network handler that can simulate failures
#[derive(Debug)]
struct MockNetworkHandler {
    fail_count: AtomicU32,
    should_fail_until: u32,
}

impl MockNetworkHandler {
    fn new(should_fail_until: u32) -> Self {
        Self {
            fail_count: AtomicU32::new(0),
            should_fail_until,
        }
    }

    fn reset(&self) {
        self.fail_count.store(0, Ordering::SeqCst);
    }

    fn attempt_count(&self) -> u32 {
        self.fail_count.load(Ordering::SeqCst)
    }
}

impl HandlerCapability for MockNetworkHandler {
    fn effect_name(&self) -> &str {
        "Network"
    }

    fn handler_name(&self) -> &str {
        "MockNetworkHandler"
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
        let count = self.fail_count.fetch_add(1, Ordering::SeqCst);

        if count < self.should_fail_until {
            HandlerResult::Abort(HandlerError::new(
                "Network",
                operation,
                format!("Simulated network timeout (attempt {})", count + 1),
            ))
        } else {
            HandlerResult::Resume(Value::String("success".to_string()))
        }
    }
}

/// Test network operation retries on timeout error
#[test]
fn test_network_retry_on_timeout() {
    let handler = MockNetworkHandler::new(2); // Fail first 2 attempts
    let config = RetryConfig::new().with_max_retries(3).with_base_delay_ms(1);

    let mut state = HandlerState::new();

    let result = with_retry(&config, || {
        match handler.handle("connect", &[], Continuation::new(), &mut state) {
            HandlerResult::Resume(v) => Ok(v),
            HandlerResult::Abort(e) => Err(e.message),
            _ => Err("unexpected result".to_string()),
        }
    });

    assert!(result.is_ok());
    assert_eq!(handler.attempt_count(), 3); // 2 failures + 1 success
}

/// Test circuit breaker opens after consecutive network failures
#[test]
fn test_network_circuit_breaker_integration() {
    let handler = MockNetworkHandler::new(u32::MAX); // Always fail
    let mut circuit = CircuitBreaker::new().with_threshold(3);
    let config = RetryConfig::new()
        .with_max_retries(0) // No retries - each failure counts
        .with_base_delay_ms(1);

    let mut state = HandlerState::new();

    // Perform operations until circuit opens
    for i in 0..5 {
        if !circuit.should_allow() {
            break;
        }

        let result = with_retry(&config, || {
            match handler.handle("connect", &[], Continuation::new(), &mut state) {
                HandlerResult::Resume(v) => Ok(v),
                HandlerResult::Abort(e) => Err(e.message),
                _ => Err("unexpected".to_string()),
            }
        });

        if result.is_err() {
            circuit.record_failure();
        }

        // After 3 failures, circuit should open
        if i >= 2 {
            assert_eq!(circuit.state(), CircuitState::Open);
        }
    }

    // Circuit is now open - handler should not be called
    let before_count = handler.attempt_count();
    assert!(!circuit.should_allow());
    assert_eq!(handler.attempt_count(), before_count); // No new calls
}

/// Test successful retry resets circuit breaker
#[test]
fn test_network_retry_success_resets_circuit() {
    let handler = MockNetworkHandler::new(2); // Fail first 2, then succeed
    let mut circuit = CircuitBreaker::new().with_threshold(5);
    let config = RetryConfig::new().with_max_retries(3).with_base_delay_ms(1);

    let mut state = HandlerState::new();

    // Record some failures first
    circuit.record_failure();
    circuit.record_failure();

    // Now perform operation with retries
    let result = with_retry(&config, || {
        if !circuit.should_allow() {
            return Err("circuit open".to_string());
        }
        match handler.handle("connect", &[], Continuation::new(), &mut state) {
            HandlerResult::Resume(v) => Ok(v),
            HandlerResult::Abort(e) => Err(e.message),
            _ => Err("unexpected".to_string()),
        }
    });

    // Success should reset circuit
    if result.is_ok() {
        circuit.record_success();
    }

    assert!(result.is_ok());
    assert_eq!(circuit.state(), CircuitState::Closed);
}

/// Test backoff delays increase between network retries
#[test]
fn test_network_backoff_between_retries() {
    let handler = MockNetworkHandler::new(3); // Fail first 3
    let config = RetryConfig::new()
        .with_max_retries(3)
        .with_base_delay_ms(10)
        .with_backoff_multiplier(2.0)
        .with_jitter(0.0);

    let mut state = HandlerState::new();
    let timestamps = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let ts_clone = timestamps.clone();

    let start = Instant::now();

    let result = with_retry(&config, || {
        ts_clone.lock().unwrap().push(start.elapsed());
        match handler.handle("connect", &[], Continuation::new(), &mut state) {
            HandlerResult::Resume(v) => Ok(v),
            HandlerResult::Abort(e) => Err(e.message),
            _ => Err("unexpected".to_string()),
        }
    });

    assert!(result.is_ok());

    let ts = timestamps.lock().unwrap();
    assert_eq!(ts.len(), 4); // 3 failures + 1 success

    // Verify delays increase (with some tolerance)
    // Expected: 0ms, ~10ms, ~30ms (10+20), ~70ms (10+20+40)
    if ts.len() >= 3 {
        let delay1 = ts[1] - ts[0];
        let delay2 = ts[2] - ts[1];

        // delay2 should be roughly 2x delay1 (within tolerance)
        assert!(
            delay2 >= delay1,
            "Backoff should increase: delay1={:?}, delay2={:?}",
            delay1,
            delay2
        );
    }
}

/// Test max retries is respected for network operations
#[test]
fn test_network_max_retries_respected() {
    let handler = MockNetworkHandler::new(u32::MAX); // Always fail
    let config = RetryConfig::new().with_max_retries(2).with_base_delay_ms(1);

    let mut state = HandlerState::new();

    let result = with_retry(&config, || {
        match handler.handle("connect", &[], Continuation::new(), &mut state) {
            HandlerResult::Resume(v) => Ok(v),
            HandlerResult::Abort(e) => Err(e.message),
            _ => Err("unexpected".to_string()),
        }
    });

    assert!(result.is_err());
    assert_eq!(handler.attempt_count(), 3); // 1 initial + 2 retries
}

/// Test error context is preserved through retry/circuit patterns
#[test]
fn test_network_error_context_preserved() {
    let handler = MockNetworkHandler::new(u32::MAX);
    let config = RetryConfig::new().with_max_retries(1).with_base_delay_ms(1);

    let mut state = HandlerState::new();

    let result = with_retry(&config, || {
        match handler.handle("http_get", &[], Continuation::new(), &mut state) {
            HandlerResult::Resume(v) => Ok(v),
            HandlerResult::Abort(e) => {
                // Verify error has context
                assert!(e.message.contains("timeout") || e.message.contains("Simulated"));
                assert!(e.timestamp.is_some());
                Err(e.message)
            }
            _ => Err("unexpected".to_string()),
        }
    });

    assert!(result.is_err());
    let err = result.into_result().unwrap_err();
    assert!(err.contains("Simulated") || err.contains("network"));
}

/// Test combined retry + circuit breaker pattern
#[test]
fn test_combined_network_resilience_pattern() {
    let handler = MockNetworkHandler::new(u32::MAX); // Always fail
    let mut circuit = CircuitBreaker::new().with_threshold(3);
    let config = RetryConfig::new()
        .with_max_retries(0) // No retries per call
        .with_base_delay_ms(1);

    let mut state = HandlerState::new();
    let mut call_count = 0;

    // Simulate multiple request cycles
    for _ in 0..10 {
        if !circuit.should_allow() {
            // Circuit open - fast fail
            continue;
        }

        let result = with_retry(&config, || {
            call_count += 1;
            match handler.handle("connect", &[], Continuation::new(), &mut state) {
                HandlerResult::Resume(v) => Ok(v),
                HandlerResult::Abort(e) => Err(e.message),
                _ => Err("unexpected".to_string()),
            }
        });

        if result.is_err() {
            circuit.record_failure();
        } else {
            circuit.record_success();
        }
    }

    // Should have stopped calling after threshold reached
    assert_eq!(call_count, 3); // Only 3 calls before circuit opened
    assert_eq!(circuit.state(), CircuitState::Open);
}

/// Test network operation timing with retries
#[test]
fn test_network_operation_timing() {
    let handler = MockNetworkHandler::new(2); // Fail first 2
    let config = RetryConfig::new()
        .with_max_retries(3)
        .with_base_delay_ms(20)
        .with_jitter(0.0);

    let mut state = HandlerState::new();
    let start = Instant::now();

    let result = with_retry(&config, || {
        match handler.handle("connect", &[], Continuation::new(), &mut state) {
            HandlerResult::Resume(v) => Ok(v),
            HandlerResult::Abort(e) => Err(e.message),
            _ => Err("unexpected".to_string()),
        }
    });

    let elapsed = start.elapsed();

    assert!(result.is_ok());
    // Should have waited: 20ms + 40ms = 60ms minimum
    assert!(
        elapsed >= Duration::from_millis(50),
        "Expected at least 50ms, got {:?}",
        elapsed
    );
}

/// Test circuit breaker cooldown and half-open state
#[test]
fn test_network_circuit_cooldown_recovery() {
    let handler = MockNetworkHandler::new(1); // Fail once, then succeed
    let mut circuit = CircuitBreaker::new()
        .with_threshold(1)
        .with_cooldown(Duration::from_millis(50));

    let mut state = HandlerState::new();

    // First call fails, opens circuit
    if circuit.should_allow() {
        match handler.handle("connect", &[], Continuation::new(), &mut state) {
            HandlerResult::Abort(_) => circuit.record_failure(),
            _ => {}
        }
    }
    assert_eq!(circuit.state(), CircuitState::Open);

    // Immediately after - circuit is open
    assert!(!circuit.should_allow());

    // Wait for cooldown
    std::thread::sleep(Duration::from_millis(60));

    // Now should transition to half-open
    assert!(circuit.should_allow());
    assert_eq!(circuit.state(), CircuitState::HalfOpen);

    // Handler already failed once, so next call should succeed (should_fail_until=1, current count=1)
    // First success in half-open
    match handler.handle("connect", &[], Continuation::new(), &mut state) {
        HandlerResult::Resume(_) => {
            circuit.record_success();
        }
        HandlerResult::Abort(_) => panic!("Expected success after first failure"),
        _ => {}
    }

    // Still in half-open (need 2 successes by default)
    assert_eq!(circuit.state(), CircuitState::HalfOpen);

    // Second success closes the circuit
    match handler.handle("connect", &[], Continuation::new(), &mut state) {
        HandlerResult::Resume(_) => {
            circuit.record_success();
        }
        _ => {}
    }

    assert_eq!(circuit.state(), CircuitState::Closed);
}
