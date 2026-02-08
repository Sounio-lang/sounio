//! Phase D.5: Production Testing - Resilience Integration Tests
//!
//! Tests for retry + circuit breaker composition patterns.

use sounio::effects::resilience::{
    CircuitBreaker, CircuitState, RetryConfig, RetryResult, with_retry,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Test that retry respects circuit breaker state
#[test]
fn test_retry_with_circuit_breaker_composition() {
    let mut circuit = CircuitBreaker::new().with_threshold(3);
    let config = RetryConfig::new().with_max_retries(2).with_base_delay_ms(1);

    let attempt_count = AtomicU32::new(0);

    // Operation that fails but respects circuit state
    let operation = || {
        if !circuit.should_allow() {
            return Err("circuit open");
        }
        attempt_count.fetch_add(1, Ordering::SeqCst);
        circuit.record_failure();
        Err("simulated failure")
    };

    // First retry cycle - should attempt and fail
    let result: RetryResult<(), &str> = with_retry(&config, operation);
    assert!(result.is_err());
    // 1 initial + 2 retries = 3 attempts, circuit now open
    assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    assert_eq!(circuit.state(), CircuitState::Open);

    // Second retry cycle - circuit is open, should not attempt
    let attempt_count2 = AtomicU32::new(0);
    let operation2 = || {
        if !circuit.should_allow() {
            return Err("circuit open");
        }
        attempt_count2.fetch_add(1, Ordering::SeqCst);
        Err("should not reach")
    };

    let result2: RetryResult<(), &str> = with_retry(&config, operation2);
    assert!(result2.is_err());
    // No attempts made because circuit is open
    assert_eq!(attempt_count2.load(Ordering::SeqCst), 0);
}

/// Test that circuit opens after retry exhaustion
#[test]
fn test_circuit_opens_after_retry_exhaustion() {
    let mut circuit = CircuitBreaker::new().with_threshold(5);
    let config = RetryConfig::new().with_max_retries(4).with_base_delay_ms(1);

    let attempt_count = AtomicU32::new(0);

    let result: RetryResult<(), &str> = with_retry(&config, || {
        attempt_count.fetch_add(1, Ordering::SeqCst);
        circuit.record_failure();
        Err("always fails")
    });

    assert!(result.is_err());
    // 1 initial + 4 retries = 5 attempts
    assert_eq!(attempt_count.load(Ordering::SeqCst), 5);
    // 5 failures = threshold reached, circuit should be open
    assert_eq!(circuit.state(), CircuitState::Open);
}

/// Test half-open state allows probe and success closes circuit
#[test]
fn test_half_open_recovery_with_retry() {
    let mut circuit = CircuitBreaker::new()
        .with_threshold(2)
        .with_cooldown(Duration::from_millis(10)); // Short cooldown for test

    // Open the circuit
    circuit.record_failure();
    circuit.record_failure();
    assert_eq!(circuit.state(), CircuitState::Open);

    // Wait for cooldown
    std::thread::sleep(Duration::from_millis(15));

    // Now should transition to half-open on should_allow()
    assert!(circuit.should_allow());
    assert_eq!(circuit.state(), CircuitState::HalfOpen);

    // Success in half-open should eventually close
    circuit.record_success();
    circuit.record_success(); // Need 2 successes by default
    assert_eq!(circuit.state(), CircuitState::Closed);
}

/// Test that jitter produces variation in delays
#[test]
fn test_jitter_prevents_thundering_herd() {
    // Use multiplier of 1.0 to keep base delay constant - we're testing jitter, not backoff
    let config = RetryConfig::new()
        .with_base_delay_ms(100)
        .with_backoff_multiplier(1.0)
        .with_jitter(0.5); // 50% jitter

    // Collect delays for different attempts
    let delays: Vec<Duration> = (0..10)
        .map(|attempt| config.delay_for_attempt(attempt))
        .collect();

    // With jitter, delays should vary due to the pseudo-random jitter
    assert!(delays.len() == 10, "Should have collected 10 delays");

    // Verify delays are within expected range (base +/- jitter_factor)
    // With multiplier=1.0 and base=100ms, jitter=0.5, range is 50-150ms
    for (i, delay) in delays.iter().enumerate() {
        let ms = delay.as_millis() as f64;
        assert!(
            ms >= 50.0 && ms <= 150.0,
            "Delay {} at attempt {} outside expected range [50, 150]",
            ms,
            i
        );
    }

    // Verify jitter produces some variation (not all identical)
    let first_delay = delays[0];
    let has_variation = delays.iter().any(|d| *d != first_delay);
    // Note: With deterministic jitter based on attempt number, we expect variation
    assert!(has_variation, "Expected variation in delays due to jitter");
}

/// Test that backoff respects max delay cap
#[test]
fn test_backoff_respects_max_delay() {
    let config = RetryConfig::new()
        .with_base_delay_ms(100)
        .with_backoff_multiplier(2.0)
        .with_max_delay_ms(500)
        .with_jitter(0.0); // No jitter for predictable test

    // Attempt 0: 100ms
    // Attempt 1: 200ms
    // Attempt 2: 400ms
    // Attempt 3: 800ms -> capped to 500ms
    // Attempt 4: 1600ms -> capped to 500ms

    assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
    assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
    assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
    assert_eq!(config.delay_for_attempt(3), Duration::from_millis(500)); // Capped
    assert_eq!(config.delay_for_attempt(4), Duration::from_millis(500)); // Capped
    assert_eq!(config.delay_for_attempt(10), Duration::from_millis(500)); // Still capped
}

/// Test that RetryResult accurately tracks attempts and duration
#[test]
fn test_retry_statistics_accurate() {
    let config = RetryConfig::new()
        .with_max_retries(3)
        .with_base_delay_ms(10)
        .with_jitter(0.0);

    let attempt_count = AtomicU32::new(0);
    let start = Instant::now();

    let result: RetryResult<i32, &str> = with_retry(&config, || {
        let count = attempt_count.fetch_add(1, Ordering::SeqCst);
        if count < 2 {
            Err("temporary failure")
        } else {
            Ok(42)
        }
    });

    let elapsed = start.elapsed();

    assert!(result.is_ok());
    assert_eq!(result.attempts, 3); // 2 failures + 1 success

    // Duration should include delays: 10ms + 20ms = 30ms minimum
    assert!(
        elapsed >= Duration::from_millis(25),
        "Expected at least 25ms delay, got {:?}",
        elapsed
    );
    assert!(
        result.total_duration >= Duration::from_millis(20),
        "Result duration too short"
    );
}

/// Test retry with all attempts exhausted
#[test]
fn test_retry_all_failures_exhausted() {
    let config = RetryConfig::new().with_max_retries(2).with_base_delay_ms(1);

    let attempt_count = AtomicU32::new(0);

    let result: RetryResult<(), &str> = with_retry(&config, || {
        attempt_count.fetch_add(1, Ordering::SeqCst);
        Err("always fails")
    });

    assert!(result.is_err());
    assert_eq!(result.attempts, 3); // 1 initial + 2 retries
    assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
}

/// Test circuit breaker failure counting and threshold
#[test]
fn test_circuit_breaker_threshold_behavior() {
    let mut circuit = CircuitBreaker::new().with_threshold(3);

    assert_eq!(circuit.state(), CircuitState::Closed);
    assert!(circuit.should_allow());

    // 2 failures - still closed
    circuit.record_failure();
    circuit.record_failure();
    assert_eq!(circuit.state(), CircuitState::Closed);

    // 3rd failure - opens
    circuit.record_failure();
    assert_eq!(circuit.state(), CircuitState::Open);
    assert!(!circuit.should_allow());
}

/// Test that success resets failure count
#[test]
fn test_circuit_success_resets_failures() {
    let mut circuit = CircuitBreaker::new().with_threshold(3);

    circuit.record_failure();
    circuit.record_failure();
    // 2 failures, 1 more would open

    circuit.record_success();
    // Failure count reset

    circuit.record_failure();
    circuit.record_failure();
    // Still closed - started fresh after success
    assert_eq!(circuit.state(), CircuitState::Closed);

    circuit.record_failure();
    // Now open
    assert_eq!(circuit.state(), CircuitState::Open);
}

/// Test retry with circuit breaker - combined resilience pattern
#[test]
fn test_combined_retry_circuit_pattern() {
    let mut circuit = CircuitBreaker::new().with_threshold(5);
    let config = RetryConfig::new().with_max_retries(2).with_base_delay_ms(1);

    let call_count = AtomicU32::new(0);

    // Helper function to perform resilient operation
    fn resilient_op(
        circuit: &mut CircuitBreaker,
        config: &RetryConfig,
        call_count: &AtomicU32,
    ) -> Result<i32, String> {
        if !circuit.should_allow() {
            return Err("circuit open".to_string());
        }

        let result: RetryResult<i32, String> = with_retry(config, || {
            call_count.fetch_add(1, Ordering::SeqCst);
            Err("network error".to_string())
        });

        if result.is_err() {
            circuit.record_failure();
        } else {
            circuit.record_success();
        }

        result.into_result()
    }

    // First call: 3 attempts (1 + 2 retries), 1 circuit failure
    let r1 = resilient_op(&mut circuit, &config, &call_count);
    assert!(r1.is_err());
    assert_eq!(call_count.load(Ordering::SeqCst), 3);

    // Second call: 3 more attempts, 2 circuit failures total
    let r2 = resilient_op(&mut circuit, &config, &call_count);
    assert!(r2.is_err());
    assert_eq!(call_count.load(Ordering::SeqCst), 6);

    // Continue until circuit opens (threshold is 5)
    let _ = resilient_op(&mut circuit, &config, &call_count); // 3 failures
    let _ = resilient_op(&mut circuit, &config, &call_count); // 4 failures
    let _ = resilient_op(&mut circuit, &config, &call_count); // 5 failures - opens

    assert_eq!(circuit.state(), CircuitState::Open);

    // Now circuit is open - no more calls should be made
    let before_count = call_count.load(Ordering::SeqCst);
    let r_open = resilient_op(&mut circuit, &config, &call_count);
    assert!(r_open.is_err());
    assert_eq!(call_count.load(Ordering::SeqCst), before_count); // No new calls
}
