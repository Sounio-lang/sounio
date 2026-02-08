//! Resilience patterns for effect handlers
//!
//! This module provides production-hardening utilities:
//! - Retry with exponential backoff
//! - Circuit breaker pattern
//! - Timeout enforcement
//!
//! # Example
//!
//! ```ignore
//! use sounio::effects::resilience::{RetryConfig, with_retry};
//!
//! let config = RetryConfig::default()
//!     .with_max_retries(3)
//!     .with_base_delay_ms(100);
//!
//! let result = with_retry(&config, || {
//!     // Potentially failing operation
//!     network_call()
//! });
//! ```

use std::time::Duration;
use tracing::{debug, warn};

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Base delay between retries in milliseconds
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds
    pub max_delay_ms: u64,
    /// Multiplier for exponential backoff (e.g., 2.0 for doubling)
    pub backoff_multiplier: f64,
    /// Add randomness to prevent thundering herd (0.0-1.0)
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 30_000, // 30 seconds max
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

impl RetryConfig {
    /// Create a new retry config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum retry attempts
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Set base delay in milliseconds
    pub fn with_base_delay_ms(mut self, ms: u64) -> Self {
        self.base_delay_ms = ms;
        self
    }

    /// Set maximum delay cap
    pub fn with_max_delay_ms(mut self, ms: u64) -> Self {
        self.max_delay_ms = ms;
        self
    }

    /// Set backoff multiplier
    pub fn with_backoff_multiplier(mut self, m: f64) -> Self {
        self.backoff_multiplier = m;
        self
    }

    /// Set jitter factor (0.0-1.0)
    pub fn with_jitter(mut self, j: f64) -> Self {
        self.jitter_factor = j.clamp(0.0, 1.0);
        self
    }

    /// Calculate delay for a given attempt number (0-indexed)
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base = self.base_delay_ms as f64;
        let multiplier = self.backoff_multiplier.powi(attempt as i32);
        let delay_ms = (base * multiplier).min(self.max_delay_ms as f64);

        // Add jitter
        let jitter_range = delay_ms * self.jitter_factor;
        let jitter = if jitter_range > 0.0 {
            // Simple pseudo-random based on attempt number
            let random_factor = ((attempt as f64 * 1.618033988749895) % 1.0) * 2.0 - 1.0;
            jitter_range * random_factor
        } else {
            0.0
        };

        let final_delay = (delay_ms + jitter).max(0.0) as u64;
        Duration::from_millis(final_delay)
    }
}

/// Result of a retry operation
#[derive(Debug)]
pub struct RetryResult<T, E> {
    /// The final result (success or last error)
    pub result: Result<T, E>,
    /// Number of attempts made
    pub attempts: u32,
    /// Total time spent retrying
    pub total_duration: Duration,
}

impl<T, E> RetryResult<T, E> {
    /// Check if the operation succeeded
    pub fn is_ok(&self) -> bool {
        self.result.is_ok()
    }

    /// Check if all retries failed
    pub fn is_err(&self) -> bool {
        self.result.is_err()
    }

    /// Get the result, consuming self
    pub fn into_result(self) -> Result<T, E> {
        self.result
    }
}

/// Execute an operation with retry logic
///
/// # Arguments
///
/// * `config` - Retry configuration
/// * `operation` - The operation to retry
///
/// # Returns
///
/// A `RetryResult` containing the outcome and retry statistics
pub fn with_retry<T, E, F>(config: &RetryConfig, mut operation: F) -> RetryResult<T, E>
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Debug,
{
    let start = std::time::Instant::now();
    let mut attempts = 0;

    loop {
        attempts += 1;
        debug!(
            attempt = attempts,
            max = config.max_retries + 1,
            "Executing operation"
        );

        match operation() {
            Ok(value) => {
                debug!(attempts, "Operation succeeded");
                return RetryResult {
                    result: Ok(value),
                    attempts,
                    total_duration: start.elapsed(),
                };
            }
            Err(e) => {
                warn!(
                    attempt = attempts,
                    max = config.max_retries + 1,
                    error = ?e,
                    "Operation failed"
                );

                if attempts > config.max_retries {
                    return RetryResult {
                        result: Err(e),
                        attempts,
                        total_duration: start.elapsed(),
                    };
                }

                // Calculate and apply delay
                let delay = config.delay_for_attempt(attempts - 1);
                debug!(delay_ms = delay.as_millis(), "Waiting before retry");
                std::thread::sleep(delay);
            }
        }
    }
}

/// Execute an async operation with retry logic
#[cfg(feature = "tokio")]
pub async fn with_retry_async<T, E, F, Fut>(
    config: &RetryConfig,
    mut operation: F,
) -> RetryResult<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let start = std::time::Instant::now();
    let mut attempts = 0;

    loop {
        attempts += 1;
        debug!(
            attempt = attempts,
            max = config.max_retries + 1,
            "Executing async operation"
        );

        match operation().await {
            Ok(value) => {
                debug!(attempts, "Async operation succeeded");
                return RetryResult {
                    result: Ok(value),
                    attempts,
                    total_duration: start.elapsed(),
                };
            }
            Err(e) => {
                warn!(
                    attempt = attempts,
                    max = config.max_retries + 1,
                    error = ?e,
                    "Async operation failed"
                );

                if attempts > config.max_retries {
                    return RetryResult {
                        result: Err(e),
                        attempts,
                        total_duration: start.elapsed(),
                    };
                }

                // Calculate and apply delay
                let delay = config.delay_for_attempt(attempts - 1);
                debug!(delay_ms = delay.as_millis(), "Waiting before async retry");
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, operations proceed normally
    Closed,
    /// Circuit is open, operations fail fast
    Open,
    /// Circuit is testing if the service has recovered
    HalfOpen,
}

/// Circuit breaker for preventing cascading failures
///
/// When a threshold of consecutive failures is reached, the circuit opens
/// and immediately rejects operations for a cooldown period.
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current state
    state: CircuitState,
    /// Consecutive failures
    failure_count: u32,
    /// Failure threshold to open circuit
    failure_threshold: u32,
    /// Time when circuit opened
    opened_at: Option<std::time::Instant>,
    /// Cooldown duration before trying half-open
    cooldown: Duration,
    /// Successes needed in half-open to close
    half_open_successes: u32,
    /// Current half-open success count
    half_open_count: u32,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold: 5,
            opened_at: None,
            cooldown: Duration::from_secs(30),
            half_open_successes: 2,
            half_open_count: 0,
        }
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new() -> Self {
        Self::default()
    }

    /// Set failure threshold
    pub fn with_threshold(mut self, n: u32) -> Self {
        self.failure_threshold = n;
        self
    }

    /// Set cooldown duration
    pub fn with_cooldown(mut self, d: Duration) -> Self {
        self.cooldown = d;
        self
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Check if operation should be allowed
    pub fn should_allow(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(opened) = self.opened_at {
                    if opened.elapsed() >= self.cooldown {
                        debug!("Circuit transitioning to half-open");
                        self.state = CircuitState::HalfOpen;
                        self.half_open_count = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.half_open_count += 1;
                if self.half_open_count >= self.half_open_successes {
                    debug!("Circuit closing after successful half-open test");
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.opened_at = None;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed operation
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    warn!(
                        failures = self.failure_count,
                        threshold = self.failure_threshold,
                        "Circuit opening due to failures"
                    );
                    self.state = CircuitState::Open;
                    self.opened_at = Some(std::time::Instant::now());
                }
            }
            CircuitState::HalfOpen => {
                warn!("Circuit re-opening after half-open failure");
                self.state = CircuitState::Open;
                self.opened_at = Some(std::time::Instant::now());
                self.half_open_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Reset the circuit breaker to closed state.
    ///
    /// This is useful for testing or manual recovery after
    /// the underlying issue has been resolved.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.opened_at = None;
        self.half_open_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 100);
    }

    #[test]
    fn test_retry_config_builder() {
        let config = RetryConfig::new()
            .with_max_retries(5)
            .with_base_delay_ms(200)
            .with_backoff_multiplier(3.0);

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.base_delay_ms, 200);
        assert_eq!(config.backoff_multiplier, 3.0);
    }

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig::new()
            .with_base_delay_ms(100)
            .with_backoff_multiplier(2.0)
            .with_jitter(0.0); // No jitter for predictable test

        // Attempt 0: 100ms
        // Attempt 1: 200ms
        // Attempt 2: 400ms
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
    }

    #[test]
    fn test_delay_respects_max() {
        let config = RetryConfig::new()
            .with_base_delay_ms(1000)
            .with_backoff_multiplier(10.0)
            .with_max_delay_ms(5000)
            .with_jitter(0.0);

        // Should be capped at 5000ms
        assert_eq!(config.delay_for_attempt(5), Duration::from_millis(5000));
    }

    #[test]
    fn test_with_retry_success_first_try() {
        let config = RetryConfig::new().with_max_retries(3);
        let mut call_count = 0;

        let result = with_retry(&config, || {
            call_count += 1;
            Ok::<_, &str>(42)
        });

        assert!(result.is_ok());
        assert_eq!(result.attempts, 1);
        assert_eq!(call_count, 1);
    }

    #[test]
    fn test_with_retry_success_after_failures() {
        let config = RetryConfig::new().with_max_retries(3).with_base_delay_ms(1); // Fast for testing

        let mut call_count = 0;

        let result = with_retry(&config, || {
            call_count += 1;
            if call_count < 3 {
                Err("temporary failure")
            } else {
                Ok(42)
            }
        });

        assert!(result.is_ok());
        assert_eq!(result.attempts, 3);
        assert_eq!(call_count, 3);
    }

    #[test]
    fn test_with_retry_all_failures() {
        let config = RetryConfig::new().with_max_retries(2).with_base_delay_ms(1);

        let mut call_count = 0;

        let result = with_retry(&config, || {
            call_count += 1;
            Err::<i32, _>("always fails")
        });

        assert!(result.is_err());
        assert_eq!(result.attempts, 3); // 1 initial + 2 retries
        assert_eq!(call_count, 3);
    }

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let mut cb = CircuitBreaker::new().with_threshold(3);

        assert!(cb.should_allow());
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_allow());
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new().with_threshold(3);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count, 2);

        cb.record_success();
        assert_eq!(cb.failure_count, 0);
        assert_eq!(cb.state(), CircuitState::Closed);
    }
}
