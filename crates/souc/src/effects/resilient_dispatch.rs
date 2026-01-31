//! Resilient Effect Dispatch
//!
//! This module provides resilience patterns integrated with effect handler dispatch.
//! It wraps handlers with retry and circuit breaker logic, enabling automatic
//! recovery from transient failures.
//!
//! # Architecture
//!
//! ```text
//! Effect Operation
//!       │
//!       ▼
//! ┌─────────────────────┐
//! │ ResilientDispatcher │
//! │  ┌───────────────┐  │
//! │  │ CircuitBreaker│──┼── Fast-fail if open
//! │  └───────────────┘  │
//! │         │           │
//! │  ┌───────────────┐  │
//! │  │  RetryConfig  │──┼── Retry with backoff
//! │  └───────────────┘  │
//! │         │           │
//! │  ┌───────────────┐  │
//! │  │ safe_dispatch │──┼── Panic recovery
//! │  └───────────────┘  │
//! └─────────────────────┘
//!       │
//!       ▼
//!   Handler Result
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use sounio::effects::resilient_dispatch::{ResilientDispatcher, DispatchConfig};
//! use sounio::effects::handlers::NetworkHandler;
//!
//! // Create a resilient dispatcher for network operations
//! let dispatcher = ResilientDispatcher::new(NetworkHandler::new())
//!     .with_retry(3, 100)  // 3 retries, 100ms base delay
//!     .with_circuit_breaker(5, 30_000);  // 5 failures, 30s cooldown
//!
//! // Dispatch with automatic resilience
//! let result = dispatcher.dispatch("http_get", &[url], &mut state);
//! ```

use crate::effects::handler_capability::{
    safe_dispatch, Continuation, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::effects::linearity::Linearity;
use crate::effects::resilience::{CircuitBreaker, CircuitState, RetryConfig};
use crate::interp::Value;
use std::panic::RefUnwindSafe;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Configuration for resilient dispatch behavior.
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Retry configuration for transient failures
    pub retry: Option<RetryConfig>,
    /// Circuit breaker for cascading failure prevention
    pub circuit_breaker_threshold: Option<u32>,
    /// Cooldown period for circuit breaker (ms)
    pub circuit_breaker_cooldown_ms: Option<u64>,
    /// Whether to use panic-safe dispatch
    pub panic_safe: bool,
    /// Operations that should NOT be retried (side effects that can't be repeated)
    pub no_retry_operations: Vec<String>,
    /// Operations that should NOT trip the circuit breaker
    pub no_circuit_operations: Vec<String>,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            retry: None,
            circuit_breaker_threshold: None,
            circuit_breaker_cooldown_ms: None,
            panic_safe: true,
            no_retry_operations: vec![
                "send".to_string(),
                "write".to_string(),
                "delete".to_string(),
                "commit".to_string(),
            ],
            no_circuit_operations: vec![],
        }
    }
}

impl DispatchConfig {
    /// Create a new dispatch configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable retry with default settings.
    pub fn with_retry(mut self, max_retries: u32, base_delay_ms: u64) -> Self {
        self.retry = Some(
            RetryConfig::new()
                .with_max_retries(max_retries)
                .with_base_delay_ms(base_delay_ms),
        );
        self
    }

    /// Enable retry with full configuration.
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry = Some(config);
        self
    }

    /// Enable circuit breaker.
    pub fn with_circuit_breaker(mut self, threshold: u32, cooldown_ms: u64) -> Self {
        self.circuit_breaker_threshold = Some(threshold);
        self.circuit_breaker_cooldown_ms = Some(cooldown_ms);
        self
    }

    /// Set panic-safe dispatch mode.
    pub fn panic_safe(mut self, enabled: bool) -> Self {
        self.panic_safe = enabled;
        self
    }

    /// Add operations that should not be retried.
    pub fn no_retry_for(mut self, operations: &[&str]) -> Self {
        self.no_retry_operations
            .extend(operations.iter().map(|s| s.to_string()));
        self
    }

    /// Add operations that should not trip the circuit breaker.
    pub fn no_circuit_for(mut self, operations: &[&str]) -> Self {
        self.no_circuit_operations
            .extend(operations.iter().map(|s| s.to_string()));
        self
    }

    /// Check if an operation should be retried.
    pub fn should_retry(&self, operation: &str) -> bool {
        self.retry.is_some() && !self.no_retry_operations.contains(&operation.to_string())
    }

    /// Check if an operation should affect the circuit breaker.
    pub fn should_circuit(&self, operation: &str) -> bool {
        self.circuit_breaker_threshold.is_some()
            && !self.no_circuit_operations.contains(&operation.to_string())
    }
}

/// Outcome of a resilient dispatch operation.
#[derive(Debug)]
pub struct DispatchOutcome {
    /// The handler result
    pub result: HandlerResult,
    /// Number of attempts made
    pub attempts: u32,
    /// Whether the circuit breaker is now open
    pub circuit_open: bool,
    /// Total time spent (including retries)
    pub duration: Duration,
}

impl DispatchOutcome {
    /// Check if the dispatch succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self.result, HandlerResult::Resume(_))
    }

    /// Get the result value if successful.
    pub fn value(&self) -> Option<&Value> {
        match &self.result {
            HandlerResult::Resume(v) => Some(v),
            _ => None,
        }
    }

    /// Get the error if failed.
    pub fn error(&self) -> Option<&HandlerError> {
        match &self.result {
            HandlerResult::Abort(e) => Some(e),
            _ => None,
        }
    }
}

/// A resilient wrapper around an effect handler.
///
/// This wrapper adds retry logic, circuit breaker protection, and panic
/// recovery to any handler that implements `HandlerCapability`.
///
/// The `RefUnwindSafe` bound is required because we use `catch_unwind`
/// internally for panic recovery.
pub struct ResilientDispatcher<H: HandlerCapability + RefUnwindSafe> {
    /// The wrapped handler
    handler: H,
    /// Dispatch configuration
    config: DispatchConfig,
    /// Circuit breaker state (shared for thread safety)
    circuit: Arc<Mutex<CircuitBreaker>>,
}

impl<H: HandlerCapability + RefUnwindSafe> ResilientDispatcher<H> {
    /// Create a new resilient dispatcher with default configuration.
    pub fn new(handler: H) -> Self {
        Self {
            handler,
            config: DispatchConfig::default(),
            circuit: Arc::new(Mutex::new(CircuitBreaker::new())),
        }
    }

    /// Create a dispatcher with custom configuration.
    pub fn with_config(handler: H, config: DispatchConfig) -> Self {
        let mut circuit = CircuitBreaker::new();

        if let Some(threshold) = config.circuit_breaker_threshold {
            circuit = circuit.with_threshold(threshold);
        }

        if let Some(cooldown_ms) = config.circuit_breaker_cooldown_ms {
            circuit = circuit.with_cooldown(Duration::from_millis(cooldown_ms));
        }

        Self {
            handler,
            config,
            circuit: Arc::new(Mutex::new(circuit)),
        }
    }

    /// Set retry configuration.
    pub fn with_retry(mut self, max_retries: u32, base_delay_ms: u64) -> Self {
        self.config = self.config.with_retry(max_retries, base_delay_ms);
        self
    }

    /// Set circuit breaker configuration.
    pub fn with_circuit_breaker(mut self, threshold: u32, cooldown_ms: u64) -> Self {
        self.config = self.config.with_circuit_breaker(threshold, cooldown_ms);

        // Update the circuit breaker instance
        let mut circuit = CircuitBreaker::new()
            .with_threshold(threshold)
            .with_cooldown(Duration::from_millis(cooldown_ms));

        self.circuit = Arc::new(Mutex::new(circuit));
        self
    }

    /// Get the current circuit breaker state.
    pub fn circuit_state(&self) -> CircuitState {
        self.circuit.lock().unwrap().state()
    }

    /// Force the circuit breaker to close (for recovery/testing).
    pub fn reset_circuit(&self) {
        self.circuit.lock().unwrap().reset();
    }

    /// Access the underlying handler.
    pub fn handler(&self) -> &H {
        &self.handler
    }

    /// Dispatch an operation with resilience patterns.
    pub fn dispatch(
        &self,
        operation: &str,
        args: &[Value],
        state: &mut HandlerState,
    ) -> DispatchOutcome {
        let start = std::time::Instant::now();

        // Check circuit breaker first
        if self.config.should_circuit(operation) {
            let mut circuit = self.circuit.lock().unwrap();
            if !circuit.should_allow() {
                return DispatchOutcome {
                    result: HandlerResult::Abort(
                        HandlerError::new(
                            self.handler.effect_name(),
                            operation,
                            "circuit breaker is open - operation blocked",
                        )
                        .with_context("circuit_state", "open"),
                    ),
                    attempts: 0,
                    circuit_open: true,
                    duration: start.elapsed(),
                };
            }
        }

        // Determine if we should retry this operation
        let should_retry = self.config.should_retry(operation);
        let should_circuit = self.config.should_circuit(operation);

        let outcome = if should_retry {
            self.dispatch_with_retry(operation, args, state, should_circuit)
        } else {
            self.dispatch_once(operation, args, state, should_circuit)
        };

        DispatchOutcome {
            result: outcome.0,
            attempts: outcome.1,
            circuit_open: self.circuit.lock().unwrap().state() == CircuitState::Open,
            duration: start.elapsed(),
        }
    }

    /// Dispatch with retry logic.
    fn dispatch_with_retry(
        &self,
        operation: &str,
        args: &[Value],
        state: &mut HandlerState,
        should_circuit: bool,
    ) -> (HandlerResult, u32) {
        let retry_config = match &self.config.retry {
            Some(cfg) => cfg,
            None => return self.dispatch_once(operation, args, state, should_circuit),
        };

        let max_attempts = retry_config.max_retries + 1;
        let mut attempt = 0;

        loop {
            attempt += 1;

            // Check circuit breaker before each attempt
            if should_circuit {
                let mut circuit = self.circuit.lock().unwrap();
                if !circuit.should_allow() {
                    return (
                        HandlerResult::Abort(HandlerError::new(
                            self.handler.effect_name(),
                            operation,
                            "circuit breaker opened during retry",
                        )),
                        attempt,
                    );
                }
            }

            // Dispatch the operation
            let result = self.dispatch_inner(operation, args, state);

            match result {
                HandlerResult::Resume(value) => {
                    if should_circuit {
                        self.circuit.lock().unwrap().record_success();
                    }
                    return (HandlerResult::Resume(value), attempt);
                }
                HandlerResult::Abort(err) => {
                    if should_circuit {
                        self.circuit.lock().unwrap().record_failure();
                    }

                    // Check if we've exhausted retries
                    if attempt >= max_attempts {
                        return (HandlerResult::Abort(err), attempt);
                    }

                    // Wait before retrying
                    let delay = retry_config.delay_for_attempt(attempt - 1);
                    std::thread::sleep(delay);
                }
                // Return and Suspend are not retryable - pass through immediately
                other => {
                    return (other, attempt);
                }
            }
        }
    }

    /// Dispatch once without retry.
    fn dispatch_once(
        &self,
        operation: &str,
        args: &[Value],
        state: &mut HandlerState,
        should_circuit: bool,
    ) -> (HandlerResult, u32) {
        let result = self.dispatch_inner(operation, args, state);

        // Update circuit breaker
        if should_circuit {
            let mut circuit = self.circuit.lock().unwrap();
            match &result {
                HandlerResult::Resume(_) => circuit.record_success(),
                HandlerResult::Abort(_) => circuit.record_failure(),
                _ => {}
            }
        }

        (result, 1)
    }

    /// Inner dispatch logic with optional panic safety.
    fn dispatch_inner(
        &self,
        operation: &str,
        args: &[Value],
        state: &mut HandlerState,
    ) -> HandlerResult {
        if self.config.panic_safe {
            safe_dispatch(&self.handler, operation, args, Continuation::new(), state)
        } else {
            self.handler
                .handle(operation, args, Continuation::new(), state)
        }
    }
}

impl<H: HandlerCapability + RefUnwindSafe> std::fmt::Debug for ResilientDispatcher<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResilientDispatcher")
            .field("handler", &self.handler.handler_name())
            .field("effect", &self.handler.effect_name())
            .field("config", &self.config)
            .field("circuit_state", &self.circuit_state())
            .finish()
    }
}

// Implement HandlerCapability for ResilientDispatcher so it can be used as a handler
impl<H: HandlerCapability + Send + Sync + RefUnwindSafe> HandlerCapability
    for ResilientDispatcher<H>
{
    fn effect_name(&self) -> &str {
        self.handler.effect_name()
    }

    fn handler_name(&self) -> &str {
        // Indicate this is a resilient wrapper
        "ResilientDispatcher"
    }

    fn operations(&self) -> &[OperationSpec] {
        self.handler.operations()
    }

    fn handle(
        &self,
        operation: &str,
        args: &[Value],
        _continuation: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        self.dispatch(operation, args, state).result
    }

    fn supports_multi_shot(&self) -> bool {
        self.handler.supports_multi_shot()
    }

    fn operation_linearity(&self, operation: &str) -> Linearity {
        self.handler.operation_linearity(operation)
    }
}

/// Builder for creating resilient handlers with common configurations.
pub struct ResilientBuilder {
    config: DispatchConfig,
}

impl ResilientBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            config: DispatchConfig::default(),
        }
    }

    /// Use network-optimized defaults.
    ///
    /// Configures: 3 retries, 100ms base delay, circuit breaker at 5 failures.
    pub fn for_network() -> Self {
        Self {
            config: DispatchConfig::new()
                .with_retry(3, 100)
                .with_circuit_breaker(5, 30_000)
                .no_retry_for(&["send", "close"]),
        }
    }

    /// Use IO-optimized defaults.
    ///
    /// Configures: 2 retries, 50ms base delay, no circuit breaker.
    pub fn for_io() -> Self {
        Self {
            config: DispatchConfig::new()
                .with_retry(2, 50)
                .no_retry_for(&["write", "delete", "create"]),
        }
    }

    /// Use database-optimized defaults.
    ///
    /// Configures: 3 retries, 200ms base delay, circuit breaker at 10 failures.
    pub fn for_database() -> Self {
        Self {
            config: DispatchConfig::new()
                .with_retry(3, 200)
                .with_circuit_breaker(10, 60_000)
                .no_retry_for(&["commit", "rollback", "begin"]),
        }
    }

    /// Set custom retry configuration.
    pub fn retry(mut self, max_retries: u32, base_delay_ms: u64) -> Self {
        self.config = self.config.with_retry(max_retries, base_delay_ms);
        self
    }

    /// Set custom circuit breaker configuration.
    pub fn circuit_breaker(mut self, threshold: u32, cooldown_ms: u64) -> Self {
        self.config = self.config.with_circuit_breaker(threshold, cooldown_ms);
        self
    }

    /// Add operations that should not be retried.
    pub fn no_retry_for(mut self, operations: &[&str]) -> Self {
        self.config = self.config.no_retry_for(operations);
        self
    }

    /// Build a resilient dispatcher wrapping the given handler.
    pub fn wrap<H: HandlerCapability + RefUnwindSafe>(self, handler: H) -> ResilientDispatcher<H> {
        ResilientDispatcher::with_config(handler, self.config)
    }
}

impl Default for ResilientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Test handler that fails a configurable number of times
    #[derive(Debug)]
    struct FailingHandler {
        fail_count: AtomicU32,
        fail_until: u32,
    }

    // AtomicU32 is RefUnwindSafe, so this is safe
    impl RefUnwindSafe for FailingHandler {}

    impl FailingHandler {
        fn new(fail_until: u32) -> Self {
            Self {
                fail_count: AtomicU32::new(0),
                fail_until,
            }
        }
    }

    impl HandlerCapability for FailingHandler {
        fn effect_name(&self) -> &str {
            "Test"
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
            let count = self.fail_count.fetch_add(1, Ordering::SeqCst);
            if count < self.fail_until {
                HandlerResult::Abort(HandlerError::new(
                    "Test",
                    "op",
                    format!("failure {}", count + 1),
                ))
            } else {
                HandlerResult::Resume(Value::Int(42))
            }
        }
    }

    #[test]
    fn test_dispatch_success_no_retry() {
        let handler = FailingHandler::new(0); // Never fails
        let dispatcher = ResilientDispatcher::new(handler);
        let mut state = HandlerState::new();

        let outcome = dispatcher.dispatch("op", &[], &mut state);

        assert!(outcome.is_success());
        assert_eq!(outcome.attempts, 1);
        assert!(!outcome.circuit_open);
    }

    #[test]
    fn test_dispatch_with_retry_eventual_success() {
        let handler = FailingHandler::new(2); // Fail first 2 times
        let dispatcher = ResilientDispatcher::new(handler).with_retry(3, 1);
        let mut state = HandlerState::new();

        let outcome = dispatcher.dispatch("op", &[], &mut state);

        assert!(outcome.is_success());
        assert_eq!(outcome.attempts, 3); // 2 failures + 1 success
    }

    #[test]
    fn test_dispatch_retry_exhaustion() {
        let handler = FailingHandler::new(10); // Always fails within retry limit
        let dispatcher = ResilientDispatcher::new(handler).with_retry(2, 1);
        let mut state = HandlerState::new();

        let outcome = dispatcher.dispatch("op", &[], &mut state);

        assert!(!outcome.is_success());
        assert_eq!(outcome.attempts, 3); // 1 initial + 2 retries
    }

    #[test]
    fn test_circuit_breaker_opens() {
        let handler = FailingHandler::new(100); // Always fails
        let dispatcher = ResilientDispatcher::new(handler)
            .with_retry(0, 1)
            .with_circuit_breaker(3, 60_000);
        let mut state = HandlerState::new();

        // Trigger 3 failures
        for _ in 0..3 {
            let _ = dispatcher.dispatch("op", &[], &mut state);
        }

        // Circuit should now be open
        assert_eq!(dispatcher.circuit_state(), CircuitState::Open);

        // Next dispatch should fail immediately
        let outcome = dispatcher.dispatch("op", &[], &mut state);
        assert!(!outcome.is_success());
        assert!(outcome.circuit_open);
        assert_eq!(outcome.attempts, 0); // No attempt made
    }

    #[test]
    fn test_no_retry_operations() {
        let handler = FailingHandler::new(1); // Fail once
        let config = DispatchConfig::new()
            .with_retry(3, 1)
            .no_retry_for(&["send"]);
        let dispatcher = ResilientDispatcher::with_config(handler, config);
        let mut state = HandlerState::new();

        // "send" should not be retried
        let outcome = dispatcher.dispatch("send", &[], &mut state);
        assert!(!outcome.is_success());
        assert_eq!(outcome.attempts, 1);

        // But other operations should be retried
        let handler2 = FailingHandler::new(1);
        let config2 = DispatchConfig::new()
            .with_retry(3, 1)
            .no_retry_for(&["send"]);
        let dispatcher2 = ResilientDispatcher::with_config(handler2, config2);
        let outcome2 = dispatcher2.dispatch("read", &[], &mut state);
        assert!(outcome2.is_success());
        assert_eq!(outcome2.attempts, 2);
    }

    #[test]
    fn test_resilient_builder_for_network() {
        let handler = FailingHandler::new(2);
        let dispatcher = ResilientBuilder::for_network().wrap(handler);
        let mut state = HandlerState::new();

        let outcome = dispatcher.dispatch("connect", &[], &mut state);
        assert!(outcome.is_success());
        assert_eq!(outcome.attempts, 3);
    }

    #[test]
    fn test_dispatch_config_defaults() {
        let config = DispatchConfig::default();
        assert!(config.panic_safe);
        assert!(config.retry.is_none());
        assert!(config.no_retry_operations.contains(&"send".to_string()));
    }

    #[test]
    fn test_reset_circuit() {
        let handler = FailingHandler::new(100);
        let dispatcher = ResilientDispatcher::new(handler).with_circuit_breaker(1, 60_000);
        let mut state = HandlerState::new();

        // Open the circuit
        let _ = dispatcher.dispatch("op", &[], &mut state);
        assert_eq!(dispatcher.circuit_state(), CircuitState::Open);

        // Reset it
        dispatcher.reset_circuit();
        assert_eq!(dispatcher.circuit_state(), CircuitState::Closed);
    }

    #[test]
    fn test_dispatch_outcome_accessors() {
        let handler = FailingHandler::new(0);
        let dispatcher = ResilientDispatcher::new(handler);
        let mut state = HandlerState::new();

        let outcome = dispatcher.dispatch("op", &[], &mut state);

        assert!(outcome.is_success());
        assert_eq!(outcome.value(), Some(&Value::Int(42)));
        assert!(outcome.error().is_none());
    }

    #[test]
    fn test_as_handler_capability() {
        let handler = FailingHandler::new(0);
        let dispatcher: ResilientDispatcher<FailingHandler> = ResilientDispatcher::new(handler);

        // Test HandlerCapability trait methods
        assert_eq!(dispatcher.effect_name(), "Test");
        assert_eq!(dispatcher.handler_name(), "ResilientDispatcher");
    }

    #[test]
    fn test_debug_impl() {
        let handler = FailingHandler::new(0);
        let dispatcher = ResilientDispatcher::new(handler);
        let debug = format!("{:?}", dispatcher);

        assert!(debug.contains("ResilientDispatcher"));
        assert!(debug.contains("FailingHandler"));
    }
}
