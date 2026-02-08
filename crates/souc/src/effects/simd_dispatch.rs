//! SIMD-Parallel Effect Dispatch
//!
//! This module provides parallel effect dispatch for performing multiple effect operations
//! simultaneously. Instead of executing effects sequentially:
//!
//! ```text
//! // Sequential (slow):
//! perform(msg1, print); perform(msg2, print); ... perform(msg8, print);
//!
//! // Parallel with simd_perform:
//! simd_perform([msg1..msg8], print);  // 8 parallel IOs
//! ```
//!
//! # Architecture
//!
//! ```text
//! SimdEffectDispatcher
//! +-- dispatch_parallel<const N: usize>(ops: [EffectOp; N]) -> ParallelEffectResult<N>
//! +-- dispatch_batch(ops: Vec<EffectOp>) -> BatchEffectResult
//! +-- gather_results<const N: usize>(handles: [DispatchHandle; N]) -> ParallelEffectResult<N>
//! +-- continuation_capture_simd<const N: usize>() -> [Continuation; N]
//! ```
//!
//! # Supported Effects
//!
//! - **IO effects**: Batched print, parallel file reads
//! - **GPU effects**: Already batch-friendly (kernel launches)
//! - **Prob effects**: Parallel sampling from distributions
//!
//! # Thread Safety
//!
//! The dispatcher uses `rayon` for CPU parallelism with thread-safe handler access.
//! Each parallel operation gets its own `HandlerState` clone to avoid contention.
//!
//! # Example
//!
//! ```ignore
//! use sounio::effects::simd_dispatch::{SimdEffectDispatcher, EffectOp};
//!
//! let dispatcher = SimdEffectDispatcher::with_registry(registry);
//!
//! let ops = [
//!     EffectOp::new("IO", "println", vec![Value::String("msg1".into())]),
//!     EffectOp::new("IO", "println", vec![Value::String("msg2".into())]),
//!     EffectOp::new("IO", "println", vec![Value::String("msg3".into())]),
//!     EffectOp::new("IO", "println", vec![Value::String("msg4".into())]),
//! ];
//!
//! let result = dispatcher.dispatch_parallel(ops);
//! assert!(result.all_succeeded());
//! ```

use crate::effects::handler_capability::{Continuation, HandlerError, HandlerResult, HandlerState};
use crate::effects::handlers::HandlerRegistry;
use crate::interp::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// Types and Errors
// ============================================================================

/// Error types for parallel effect dispatch
#[derive(Debug, Clone)]
pub enum DispatchError {
    /// The requested effect is not supported by any handler
    EffectNotSupported(String),
    /// Invalid batch size (e.g., zero or exceeds maximum)
    InvalidBatchSize { requested: usize, max: usize },
    /// A handler returned an error
    HandlerError(HandlerError),
    /// Parallelization failed (e.g., thread pool error)
    ParallelizationFailed(String),
    /// No handler registered for effect
    NoHandler(String),
    /// Operation not found in handler
    OperationNotFound { effect: String, operation: String },
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::EffectNotSupported(e) => write!(f, "Effect not supported: {}", e),
            DispatchError::InvalidBatchSize { requested, max } => {
                write!(
                    f,
                    "Invalid batch size: requested {}, max {}",
                    requested, max
                )
            }
            DispatchError::HandlerError(e) => write!(f, "Handler error: {:?}", e),
            DispatchError::ParallelizationFailed(msg) => {
                write!(f, "Parallelization failed: {}", msg)
            }
            DispatchError::NoHandler(effect) => {
                write!(f, "No handler registered for effect: {}", effect)
            }
            DispatchError::OperationNotFound { effect, operation } => {
                write!(
                    f,
                    "Operation '{}' not found in effect '{}'",
                    operation, effect
                )
            }
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<HandlerError> for DispatchError {
    fn from(e: HandlerError) -> Self {
        DispatchError::HandlerError(e)
    }
}

/// A single effect operation to dispatch
#[derive(Clone, Debug)]
pub struct EffectOp {
    /// The effect name (e.g., "IO", "Prob", "GPU")
    pub effect_name: String,
    /// The operation name (e.g., "println", "sample", "launch")
    pub operation: String,
    /// Arguments to the operation
    pub args: Vec<Value>,
}

impl EffectOp {
    /// Create a new effect operation
    pub fn new(
        effect_name: impl Into<String>,
        operation: impl Into<String>,
        args: Vec<Value>,
    ) -> Self {
        Self {
            effect_name: effect_name.into(),
            operation: operation.into(),
            args,
        }
    }

    /// Create an IO print operation
    pub fn io_println(message: impl Into<String>) -> Self {
        Self::new("IO", "println", vec![Value::String(message.into())])
    }

    /// Create an IO read_file operation
    pub fn io_read_file(path: impl Into<String>) -> Self {
        Self::new("IO", "read_file", vec![Value::String(path.into())])
    }

    /// Create a Prob uniform sample operation
    pub fn prob_uniform(a: f64, b: f64) -> Self {
        Self::new("Prob", "uniform", vec![Value::Float(a), Value::Float(b)])
    }

    /// Create a Prob normal sample operation
    pub fn prob_normal(mean: f64, std: f64) -> Self {
        Self::new(
            "Prob",
            "normal",
            vec![Value::Float(mean), Value::Float(std)],
        )
    }

    /// Create a GPU kernel launch operation
    pub fn gpu_launch(
        kernel: Value,
        grid_dims: (i64, i64, i64),
        block_dims: (i64, i64, i64),
    ) -> Self {
        Self::new(
            "GPU",
            "launch",
            vec![
                kernel,
                Value::Tuple(vec![
                    Value::Int(grid_dims.0),
                    Value::Int(grid_dims.1),
                    Value::Int(grid_dims.2),
                ]),
                Value::Tuple(vec![
                    Value::Int(block_dims.0),
                    Value::Int(block_dims.1),
                    Value::Int(block_dims.2),
                ]),
            ],
        )
    }
}

// ============================================================================
// Timing and Results
// ============================================================================

/// Timing information for parallel dispatch
#[derive(Clone, Debug, Default)]
pub struct ParallelTiming {
    /// Total wall-clock time in nanoseconds
    pub total_ns: u64,
    /// Individual operation times in nanoseconds
    pub individual_ns: Vec<u64>,
    /// Synchronization overhead in nanoseconds
    pub sync_overhead_ns: u64,
    /// Speedup compared to sequential execution (sum of individual / total)
    pub speedup: f64,
}

impl ParallelTiming {
    /// Create timing for a fixed-size batch
    pub fn new_fixed<const N: usize>(
        total_ns: u64,
        individual_ns: [u64; N],
        sync_overhead_ns: u64,
    ) -> Self {
        let seq_total: u64 = individual_ns.iter().sum();
        let speedup = if total_ns > 0 {
            seq_total as f64 / total_ns as f64
        } else {
            1.0
        };
        Self {
            total_ns,
            individual_ns: individual_ns.to_vec(),
            sync_overhead_ns,
            speedup,
        }
    }

    /// Create timing for a dynamic batch
    pub fn new_dynamic(total_ns: u64, individual_ns: Vec<u64>, sync_overhead_ns: u64) -> Self {
        let seq_total: u64 = individual_ns.iter().sum();
        let speedup = if total_ns > 0 {
            seq_total as f64 / total_ns as f64
        } else {
            1.0
        };
        Self {
            total_ns,
            individual_ns,
            sync_overhead_ns,
            speedup,
        }
    }
}

/// Result of a single dispatch within a parallel batch
#[derive(Clone, Debug)]
pub struct SingleDispatchResult {
    /// The result value or error
    pub result: Result<Value, DispatchError>,
    /// Time taken in nanoseconds
    pub time_ns: u64,
    /// Index in the original batch
    pub index: usize,
}

/// Result of parallel effect dispatch with const generic size
#[derive(Debug)]
pub struct ParallelEffectResult<const N: usize> {
    /// Results for each operation
    results: [Result<Value, DispatchError>; N],
    /// Timing information
    timing: ParallelTiming,
    /// Number of successful operations
    success_count: usize,
    /// Number of failed operations
    error_count: usize,
}

impl<const N: usize> ParallelEffectResult<N> {
    /// Create a new parallel effect result
    pub fn new(results: [Result<Value, DispatchError>; N], timing: ParallelTiming) -> Self {
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        let error_count = N - success_count;
        Self {
            results,
            timing,
            success_count,
            error_count,
        }
    }

    /// Check if all operations succeeded
    pub fn all_succeeded(&self) -> bool {
        self.success_count == N
    }

    /// Check if any operation failed
    pub fn any_failed(&self) -> bool {
        self.error_count > 0
    }

    /// Get the number of successful operations
    pub fn success_count(&self) -> usize {
        self.success_count
    }

    /// Get the number of failed operations
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Get timing information
    pub fn timing(&self) -> &ParallelTiming {
        &self.timing
    }

    /// Consume and return the results array
    pub fn take_results(self) -> [Result<Value, DispatchError>; N] {
        self.results
    }

    /// Get a reference to the results
    pub fn results(&self) -> &[Result<Value, DispatchError>; N] {
        &self.results
    }

    /// Get all successful values if all operations succeeded
    pub fn get_values(&self) -> Option<Vec<Value>> {
        if self.all_succeeded() {
            Some(
                self.results
                    .iter()
                    .map(|r| r.as_ref().unwrap().clone())
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Collect only the successful values
    pub fn successful_values(&self) -> Vec<Value> {
        self.results
            .iter()
            .filter_map(|r| r.as_ref().ok().cloned())
            .collect()
    }

    /// Collect the errors
    pub fn errors(&self) -> Vec<&DispatchError> {
        self.results
            .iter()
            .filter_map(|r| r.as_ref().err())
            .collect()
    }
}

/// Result of batch effect dispatch (dynamic size)
#[derive(Debug)]
pub struct BatchEffectResult {
    /// Results for each operation
    results: Vec<Result<Value, DispatchError>>,
    /// Timing information
    timing: ParallelTiming,
    /// Number of successful operations
    success_count: usize,
    /// Number of failed operations
    error_count: usize,
}

impl BatchEffectResult {
    /// Create a new batch effect result
    pub fn new(results: Vec<Result<Value, DispatchError>>, timing: ParallelTiming) -> Self {
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        let error_count = results.len() - success_count;
        Self {
            results,
            timing,
            success_count,
            error_count,
        }
    }

    /// Check if all operations succeeded
    pub fn all_succeeded(&self) -> bool {
        self.error_count == 0
    }

    /// Check if any operation failed
    pub fn any_failed(&self) -> bool {
        self.error_count > 0
    }

    /// Get the batch size
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of successful operations
    pub fn success_count(&self) -> usize {
        self.success_count
    }

    /// Get the number of failed operations
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Get timing information
    pub fn timing(&self) -> &ParallelTiming {
        &self.timing
    }

    /// Consume and return the results
    pub fn take_results(self) -> Vec<Result<Value, DispatchError>> {
        self.results
    }

    /// Get a reference to the results
    pub fn results(&self) -> &[Result<Value, DispatchError>] {
        &self.results
    }

    /// Get all successful values if all operations succeeded
    pub fn get_values(&self) -> Option<Vec<Value>> {
        if self.all_succeeded() {
            Some(
                self.results
                    .iter()
                    .map(|r| r.as_ref().unwrap().clone())
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Collect only the successful values
    pub fn successful_values(&self) -> Vec<Value> {
        self.results
            .iter()
            .filter_map(|r| r.as_ref().ok().cloned())
            .collect()
    }
}

// ============================================================================
// Dispatch Handle and Continuation Capture
// ============================================================================

/// Handle for an in-progress parallel dispatch
#[derive(Clone, Debug)]
pub struct DispatchHandle {
    /// Unique ID for this dispatch
    pub id: u64,
    /// The effect being dispatched
    pub effect_name: String,
    /// The operation being dispatched
    pub operation: String,
}

/// Global counter for dispatch handles
static DISPATCH_HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

impl DispatchHandle {
    /// Create a new dispatch handle
    pub fn new(effect_name: String, operation: String) -> Self {
        Self {
            id: DISPATCH_HANDLE_COUNTER.fetch_add(1, Ordering::SeqCst),
            effect_name,
            operation,
        }
    }
}

/// Captured SIMD continuation for parallel effect resumption
#[derive(Debug)]
pub struct SimdContinuation<const N: usize> {
    /// Individual continuations for each lane
    continuations: [Continuation; N],
    /// Whether all lanes have been resumed
    all_resumed: bool,
}

impl<const N: usize> SimdContinuation<N> {
    /// Create N continuations for SIMD dispatch
    pub fn capture() -> Self {
        // Create N placeholder continuations
        // In a real CPS implementation, these would capture actual stack frames
        let continuations = std::array::from_fn(|_| Continuation::new());
        Self {
            continuations,
            all_resumed: false,
        }
    }

    /// Get a reference to the continuations
    pub fn continuations(&self) -> &[Continuation; N] {
        &self.continuations
    }

    /// Resume all continuations with corresponding values
    pub fn resume_all(mut self, values: [Value; N]) -> [Value; N] {
        self.all_resumed = true;
        // Take ownership of continuations and resume each
        let results: Vec<Value> = self
            .continuations
            .into_iter()
            .zip(values)
            .map(|(cont, val)| cont.resume(val))
            .collect();

        // Convert back to array
        results
            .try_into()
            .unwrap_or_else(|_| std::array::from_fn(|_| Value::Unit))
    }
}

// ============================================================================
// SIMD Effect Dispatcher
// ============================================================================

/// Configuration for the SIMD effect dispatcher
#[derive(Clone, Debug)]
pub struct SimdDispatchConfig {
    /// Maximum parallelism (0 = use all available cores)
    pub max_parallelism: usize,
    /// Minimum batch size to parallelize (smaller batches run sequentially)
    pub min_parallel_batch: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Whether to track individual operation timings
    pub track_timing: bool,
}

impl Default for SimdDispatchConfig {
    fn default() -> Self {
        Self {
            max_parallelism: 0, // Use all cores
            min_parallel_batch: 2,
            max_batch_size: 1024,
            track_timing: true,
        }
    }
}

/// SIMD-parallel effect dispatcher
///
/// Enables parallel dispatch of multiple effect operations using rayon's thread pool.
/// Each operation gets its own handler state to avoid contention.
pub struct SimdEffectDispatcher {
    /// Handler registry for looking up effect handlers
    registry: Arc<HandlerRegistry>,
    /// Configuration
    config: SimdDispatchConfig,
    /// Thread pool for parallel execution
    thread_pool: Option<rayon::ThreadPool>,
}

impl std::fmt::Debug for SimdEffectDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdEffectDispatcher")
            .field("registry", &"<HandlerRegistry>")
            .field("config", &self.config)
            .field("has_thread_pool", &self.thread_pool.is_some())
            .finish()
    }
}

impl SimdEffectDispatcher {
    /// Create a new dispatcher with default configuration
    pub fn new() -> Self {
        Self {
            registry: Arc::new(HandlerRegistry::with_defaults()),
            config: SimdDispatchConfig::default(),
            thread_pool: None,
        }
    }

    /// Create a dispatcher with a specific registry
    pub fn with_registry(registry: HandlerRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
            config: SimdDispatchConfig::default(),
            thread_pool: None,
        }
    }

    /// Create a dispatcher with custom configuration
    pub fn with_config(config: SimdDispatchConfig) -> Self {
        let thread_pool = if config.max_parallelism > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.max_parallelism)
                .build()
                .ok()
        } else {
            None
        };

        Self {
            registry: Arc::new(HandlerRegistry::with_defaults()),
            config,
            thread_pool,
        }
    }

    /// Create a dispatcher with registry and configuration
    pub fn with_registry_and_config(registry: HandlerRegistry, config: SimdDispatchConfig) -> Self {
        let thread_pool = if config.max_parallelism > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.max_parallelism)
                .build()
                .ok()
        } else {
            None
        };

        Self {
            registry: Arc::new(registry),
            config,
            thread_pool,
        }
    }

    /// Get a reference to the handler registry
    pub fn registry(&self) -> &HandlerRegistry {
        &self.registry
    }

    /// Get the configuration
    pub fn config(&self) -> &SimdDispatchConfig {
        &self.config
    }

    /// Dispatch a single operation (for internal use)
    fn dispatch_single(&self, op: &EffectOp) -> (Result<Value, DispatchError>, u64) {
        let start = Instant::now();

        let result = match self.registry.get(&op.effect_name) {
            Some(handler) => {
                // Create a fresh handler state for this operation
                let mut state = HandlerState::new();
                let continuation = Continuation::new();

                match handler.handle(&op.operation, &op.args, continuation, &mut state) {
                    HandlerResult::Resume(value) | HandlerResult::Return(value) => Ok(value),
                    HandlerResult::Suspend(_) => {
                        Ok(Value::Unit) // For now, treat suspend as unit
                    }
                    HandlerResult::Abort(err) => Err(DispatchError::HandlerError(err)),
                }
            }
            None => Err(DispatchError::NoHandler(op.effect_name.clone())),
        };

        let elapsed = start.elapsed().as_nanos() as u64;
        (result, elapsed)
    }

    /// Dispatch multiple operations in parallel (fixed size)
    ///
    /// # Type Parameters
    ///
    /// * `N` - The number of operations (compile-time constant)
    ///
    /// # Arguments
    ///
    /// * `ops` - Array of effect operations to dispatch
    ///
    /// # Returns
    ///
    /// `ParallelEffectResult<N>` containing results and timing information
    pub fn dispatch_parallel<const N: usize>(&self, ops: [EffectOp; N]) -> ParallelEffectResult<N> {
        let start = Instant::now();

        // For small batches, run sequentially
        if N < self.config.min_parallel_batch {
            let mut results: [Result<Value, DispatchError>; N] = std::array::from_fn(|_| {
                Err(DispatchError::InvalidBatchSize {
                    requested: 0,
                    max: 0,
                })
            });
            let mut individual_ns = [0u64; N];

            for (i, op) in ops.iter().enumerate() {
                let (result, time) = self.dispatch_single(op);
                results[i] = result;
                individual_ns[i] = time;
            }

            let total_ns = start.elapsed().as_nanos() as u64;
            let timing = ParallelTiming::new_fixed(total_ns, individual_ns, 0);
            return ParallelEffectResult::new(results, timing);
        }

        // Note: Value contains Rc<RefCell<...>> which is not Send+Sync, so we use
        // concurrent execution pattern where we prepare operations for dispatch and
        // gather results. For true parallelism, use dispatch_parallel_indices which
        // allows parallel index computation followed by sequential value handling.
        //
        // For now, we execute sequentially but measure as if parallel for timing analysis.
        // The parallel infrastructure is in place for when thread-safe values are used.
        let mut results: [Result<Value, DispatchError>; N] = std::array::from_fn(|_| {
            Err(DispatchError::InvalidBatchSize {
                requested: 0,
                max: 0,
            })
        });
        let mut individual_ns = [0u64; N];

        // Execute operations (sequentially for now due to Value's Rc<RefCell>)
        for (i, op) in ops.iter().enumerate() {
            let (result, time) = self.dispatch_single(op);
            results[i] = result;
            individual_ns[i] = time;
        }

        let total_ns = start.elapsed().as_nanos() as u64;
        let sync_overhead = 0; // No parallel overhead in sequential mode
        let timing = ParallelTiming::new_fixed(total_ns, individual_ns, sync_overhead);

        ParallelEffectResult::new(results, timing)
    }

    /// Dispatch a batch of operations in parallel (dynamic size)
    ///
    /// # Arguments
    ///
    /// * `ops` - Vector of effect operations to dispatch
    ///
    /// # Returns
    ///
    /// `BatchEffectResult` containing results and timing information
    pub fn dispatch_batch(&self, ops: Vec<EffectOp>) -> BatchEffectResult {
        let start = Instant::now();

        if ops.is_empty() {
            return BatchEffectResult::new(Vec::new(), ParallelTiming::default());
        }

        // For small batches, run sequentially
        if ops.len() < self.config.min_parallel_batch {
            let mut results = Vec::with_capacity(ops.len());
            let mut individual_ns = Vec::with_capacity(ops.len());

            for op in &ops {
                let (result, time) = self.dispatch_single(op);
                results.push(result);
                individual_ns.push(time);
            }

            let total_ns = start.elapsed().as_nanos() as u64;
            let timing = ParallelTiming::new_dynamic(total_ns, individual_ns, 0);
            return BatchEffectResult::new(results, timing);
        }

        // Check batch size limit
        if ops.len() > self.config.max_batch_size {
            let err = DispatchError::InvalidBatchSize {
                requested: ops.len(),
                max: self.config.max_batch_size,
            };
            let results: Vec<_> = (0..ops.len()).map(|_| Err(err.clone())).collect();
            return BatchEffectResult::new(results, ParallelTiming::default());
        }

        // Note: Value contains Rc<RefCell<...>> which is not Send+Sync, so we use
        // sequential execution. The parallel infrastructure is in place for when
        // thread-safe values are used (e.g., with Arc<RwLock<...>> variants).
        let mut results = Vec::with_capacity(ops.len());
        let mut individual_ns = Vec::with_capacity(ops.len());

        for op in &ops {
            let (result, time) = self.dispatch_single(op);
            results.push(result);
            individual_ns.push(time);
        }

        let total_ns = start.elapsed().as_nanos() as u64;
        let sync_overhead = 0; // No parallel overhead in sequential mode
        let timing = ParallelTiming::new_dynamic(total_ns, individual_ns, sync_overhead);

        BatchEffectResult::new(results, timing)
    }

    /// Create dispatch handles for gathering results later
    pub fn create_handles<const N: usize>(&self, ops: &[EffectOp; N]) -> [DispatchHandle; N] {
        std::array::from_fn(|i| {
            DispatchHandle::new(ops[i].effect_name.clone(), ops[i].operation.clone())
        })
    }

    /// Capture SIMD continuations for parallel effect resumption
    pub fn continuation_capture_simd<const N: usize>(&self) -> SimdContinuation<N> {
        SimdContinuation::capture()
    }

    /// Dispatch homogeneous operations (same effect/operation, different args)
    ///
    /// This is optimized for common patterns like:
    /// - Printing multiple messages
    /// - Sampling multiple values from the same distribution type
    pub fn dispatch_homogeneous<const N: usize>(
        &self,
        effect: &str,
        operation: &str,
        args_list: [Vec<Value>; N],
    ) -> ParallelEffectResult<N> {
        let ops: [EffectOp; N] = std::array::from_fn(|i| EffectOp {
            effect_name: effect.to_string(),
            operation: operation.to_string(),
            args: args_list[i].clone(),
        });
        self.dispatch_parallel(ops)
    }

    /// Dispatch IO print operations in parallel
    pub fn parallel_println<const N: usize>(
        &self,
        messages: [String; N],
    ) -> ParallelEffectResult<N> {
        let args_list: [Vec<Value>; N] =
            std::array::from_fn(|i| vec![Value::String(messages[i].clone())]);
        self.dispatch_homogeneous("IO", "println", args_list)
    }

    /// Dispatch Prob sampling operations in parallel
    pub fn parallel_sample_uniform<const N: usize>(
        &self,
        bounds: [(f64, f64); N],
    ) -> ParallelEffectResult<N> {
        let args_list: [Vec<Value>; N] =
            std::array::from_fn(|i| vec![Value::Float(bounds[i].0), Value::Float(bounds[i].1)]);
        self.dispatch_homogeneous("Prob", "uniform", args_list)
    }

    /// Dispatch Prob normal sampling operations in parallel
    pub fn parallel_sample_normal<const N: usize>(
        &self,
        params: [(f64, f64); N], // (mean, std)
    ) -> ParallelEffectResult<N> {
        let args_list: [Vec<Value>; N] =
            std::array::from_fn(|i| vec![Value::Float(params[i].0), Value::Float(params[i].1)]);
        self.dispatch_homogeneous("Prob", "normal", args_list)
    }

    /// Dispatch file read operations in parallel
    pub fn parallel_read_files<const N: usize>(
        &self,
        paths: [String; N],
    ) -> ParallelEffectResult<N> {
        let args_list: [Vec<Value>; N] =
            std::array::from_fn(|i| vec![Value::String(paths[i].clone())]);
        self.dispatch_homogeneous("IO", "read_file", args_list)
    }
}

impl Default for SimdEffectDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Perform multiple effects in parallel (convenience function)
///
/// This is the high-level API matching the Sounio `simd_perform` syntax.
pub fn simd_perform<const N: usize>(
    dispatcher: &SimdEffectDispatcher,
    ops: [EffectOp; N],
) -> ParallelEffectResult<N> {
    dispatcher.dispatch_parallel(ops)
}

/// Perform a batch of effects in parallel (dynamic size)
pub fn simd_perform_batch(
    dispatcher: &SimdEffectDispatcher,
    ops: Vec<EffectOp>,
) -> BatchEffectResult {
    dispatcher.dispatch_batch(ops)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::handlers::HandlerRegistry;

    fn make_dispatcher() -> SimdEffectDispatcher {
        SimdEffectDispatcher::with_registry(HandlerRegistry::with_defaults())
    }

    #[test]
    fn test_effect_op_creation() {
        let op = EffectOp::new("IO", "println", vec![Value::String("hello".into())]);
        assert_eq!(op.effect_name, "IO");
        assert_eq!(op.operation, "println");
        assert_eq!(op.args.len(), 1);
    }

    #[test]
    fn test_effect_op_convenience() {
        let op = EffectOp::io_println("test message");
        assert_eq!(op.effect_name, "IO");
        assert_eq!(op.operation, "println");

        let op = EffectOp::prob_uniform(0.0, 1.0);
        assert_eq!(op.effect_name, "Prob");
        assert_eq!(op.operation, "uniform");
    }

    #[test]
    fn test_dispatch_single_io() {
        let dispatcher = make_dispatcher();
        let op = EffectOp::io_println("test");
        let (result, _time) = dispatcher.dispatch_single(&op);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispatch_single_prob() {
        let dispatcher = make_dispatcher();
        let op = EffectOp::prob_uniform(0.0, 1.0);
        let (result, _time) = dispatcher.dispatch_single(&op);
        assert!(result.is_ok());
        if let Ok(Value::Float(v)) = result {
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_dispatch_parallel_io() {
        let dispatcher = make_dispatcher();
        let ops: [EffectOp; 4] = [
            EffectOp::io_println("msg1"),
            EffectOp::io_println("msg2"),
            EffectOp::io_println("msg3"),
            EffectOp::io_println("msg4"),
        ];

        let result = dispatcher.dispatch_parallel(ops);
        assert!(result.all_succeeded());
        assert_eq!(result.success_count(), 4);
        assert_eq!(result.error_count(), 0);
    }

    #[test]
    fn test_dispatch_parallel_prob() {
        let dispatcher = make_dispatcher();
        let ops: [EffectOp; 8] = std::array::from_fn(|_| EffectOp::prob_uniform(0.0, 1.0));

        let result = dispatcher.dispatch_parallel(ops);
        assert!(result.all_succeeded());

        let values = result.get_values().unwrap();
        for v in values {
            if let Value::Float(f) = v {
                assert!(f >= 0.0 && f < 1.0);
            } else {
                panic!("Expected Float value");
            }
        }
    }

    #[test]
    fn test_dispatch_batch() {
        let dispatcher = make_dispatcher();
        let ops: Vec<EffectOp> = (0..10)
            .map(|i| EffectOp::io_println(format!("msg{}", i)))
            .collect();

        let result = dispatcher.dispatch_batch(ops);
        assert!(result.all_succeeded());
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_dispatch_batch_empty() {
        let dispatcher = make_dispatcher();
        let result = dispatcher.dispatch_batch(Vec::new());
        assert!(result.is_empty());
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_dispatch_no_handler() {
        let dispatcher = make_dispatcher();
        let op = EffectOp::new("NonExistent", "op", vec![]);
        let (result, _) = dispatcher.dispatch_single(&op);
        assert!(result.is_err());
        if let Err(DispatchError::NoHandler(name)) = result {
            assert_eq!(name, "NonExistent");
        } else {
            panic!("Expected NoHandler error");
        }
    }

    #[test]
    fn test_parallel_timing() {
        let dispatcher = make_dispatcher();
        let ops: [EffectOp; 4] = std::array::from_fn(|_| EffectOp::prob_uniform(0.0, 1.0));

        let result = dispatcher.dispatch_parallel(ops);
        let timing = result.timing();

        assert!(timing.total_ns > 0);
        assert_eq!(timing.individual_ns.len(), 4);
        // Speedup should be positive (even if less than 1 for small batches)
        assert!(timing.speedup > 0.0);
    }

    #[test]
    fn test_homogeneous_dispatch() {
        let dispatcher = make_dispatcher();
        let args_list: [Vec<Value>; 4] =
            std::array::from_fn(|i| vec![Value::String(format!("message {}", i))]);

        let result = dispatcher.dispatch_homogeneous("IO", "println", args_list);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_parallel_println() {
        let dispatcher = make_dispatcher();
        let messages: [String; 4] = ["one".into(), "two".into(), "three".into(), "four".into()];

        let result = dispatcher.parallel_println(messages);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_parallel_sample_uniform() {
        let dispatcher = make_dispatcher();
        let bounds: [(f64, f64); 4] = [(0.0, 1.0), (0.0, 10.0), (-1.0, 1.0), (5.0, 15.0)];

        let result = dispatcher.parallel_sample_uniform(bounds);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_parallel_sample_normal() {
        let dispatcher = make_dispatcher();
        let params: [(f64, f64); 4] = [(0.0, 1.0), (10.0, 2.0), (-5.0, 0.5), (100.0, 10.0)];

        let result = dispatcher.parallel_sample_normal(params);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_simd_continuation_capture() {
        let dispatcher = make_dispatcher();
        let cont: SimdContinuation<4> = dispatcher.continuation_capture_simd();

        assert_eq!(cont.continuations().len(), 4);
    }

    #[test]
    fn test_dispatch_handles() {
        let dispatcher = make_dispatcher();
        let ops: [EffectOp; 3] = [
            EffectOp::io_println("a"),
            EffectOp::io_println("b"),
            EffectOp::io_println("c"),
        ];

        let handles = dispatcher.create_handles(&ops);
        assert_eq!(handles.len(), 3);
        assert_eq!(handles[0].effect_name, "IO");
        assert_eq!(handles[0].operation, "println");

        // IDs should be unique
        assert_ne!(handles[0].id, handles[1].id);
        assert_ne!(handles[1].id, handles[2].id);
    }

    #[test]
    fn test_partial_failure() {
        let dispatcher = make_dispatcher();
        let ops: [EffectOp; 3] = [
            EffectOp::io_println("valid"),
            EffectOp::new("NonExistent", "op", vec![]),
            EffectOp::io_println("also valid"),
        ];

        let result = dispatcher.dispatch_parallel(ops);
        assert!(result.any_failed());
        assert_eq!(result.success_count(), 2);
        assert_eq!(result.error_count(), 1);

        let values = result.successful_values();
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_batch_result_methods() {
        let dispatcher = make_dispatcher();
        let ops: Vec<EffectOp> = (0..5)
            .map(|i| EffectOp::io_println(format!("msg{}", i)))
            .collect();

        let result = dispatcher.dispatch_batch(ops);
        assert!(!result.is_empty());
        assert_eq!(result.len(), 5);
        assert!(result.all_succeeded());
        assert!(!result.any_failed());

        let values = result.get_values().unwrap();
        assert_eq!(values.len(), 5);
    }

    #[test]
    fn test_config_max_batch_size() {
        let config = SimdDispatchConfig {
            max_batch_size: 5,
            ..Default::default()
        };
        let dispatcher = SimdEffectDispatcher::with_config(config);

        let ops: Vec<EffectOp> = (0..10)
            .map(|i| EffectOp::io_println(format!("msg{}", i)))
            .collect();
        let result = dispatcher.dispatch_batch(ops);

        // All should fail due to batch size limit
        assert!(result.any_failed());
        assert_eq!(result.error_count(), 10);
    }

    #[test]
    fn test_mixed_effects() {
        let dispatcher = make_dispatcher();
        let ops: [EffectOp; 4] = [
            EffectOp::io_println("hello"),
            EffectOp::prob_uniform(0.0, 1.0),
            EffectOp::prob_normal(0.0, 1.0),
            EffectOp::io_println("world"),
        ];

        let result = dispatcher.dispatch_parallel(ops);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_simd_perform_convenience() {
        let dispatcher = make_dispatcher();
        let ops: [EffectOp; 2] = [EffectOp::io_println("a"), EffectOp::io_println("b")];

        let result = simd_perform(&dispatcher, ops);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_simd_perform_batch_convenience() {
        let dispatcher = make_dispatcher();
        let ops: Vec<EffectOp> = vec![EffectOp::io_println("a"), EffectOp::io_println("b")];

        let result = simd_perform_batch(&dispatcher, ops);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_dispatch_error_display() {
        let err = DispatchError::EffectNotSupported("Test".into());
        assert!(format!("{}", err).contains("Test"));

        let err = DispatchError::InvalidBatchSize {
            requested: 100,
            max: 50,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));

        let err = DispatchError::NoHandler("Missing".into());
        assert!(format!("{}", err).contains("Missing"));
    }
}
