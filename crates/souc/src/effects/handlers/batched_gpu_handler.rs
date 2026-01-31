//! Batched GPU Effect Handler
//!
//! This module provides a batching wrapper around `RealGpuHandler` that collects
//! GPU operations during a `batch { ... }` block and submits them together with
//! a single synchronization point at the end.
//!
//! # Architecture
//!
//! Based on the ICFP 2024 paper "Parallel Algebraic Effect Handlers", this handler
//! implements effect batching for GPU operations:
//!
//! ```text
//! batch {
//!     let buf1 = GPU.allocate(1024);     // Queued
//!     let buf2 = GPU.allocate(2048);     // Queued
//!     GPU.copy_to_device(data1, buf1);   // Queued
//!     GPU.copy_to_device(data2, buf2);   // Queued
//!     GPU.launch(kernel, grid, block);   // Queued
//! }  // <-- All operations executed here with single sync
//! ```
//!
//! # Benefits
//!
//! - Reduces GPU synchronization overhead by batching operations
//! - Enables command buffer coalescing for WGPU
//! - Allows for potential operation reordering and optimization
//! - Provides transactional semantics (all-or-nothing execution)
//!
//! # Usage
//!
//! ```ignore
//! use sounio::effects::handlers::BatchedGpuHandler;
//!
//! let mut handler = BatchedGpuHandler::new();
//! handler.begin_batch();
//! // ... queue operations ...
//! handler.end_batch(&mut state)?;  // Executes all operations
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;
use thiserror::Error;

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::effects::linearity::Linearity;
use crate::interp::Value;

use super::gpu_handler_real::RealGpuHandler;

/// Confidence factor for GPU compute operations (matches RealGpuHandler)
const GPU_COMPUTE_CONFIDENCE_FACTOR: f64 = 0.99;

/// Static operation specifications for batched GPU handler
static BATCHED_GPU_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_batched_gpu_operations() -> &'static [OperationSpec] {
    BATCHED_GPU_OPERATIONS.get_or_init(|| {
        vec![
            // Standard GPU operations (delegated or batched)
            OperationSpec::new("compile_kernel", "KernelId").with_params(vec!["String", "String"]),
            OperationSpec::new("launch", "Unit")
                .with_params(vec!["KernelId", "Tuple", "Tuple", "Array"])
                .with_confidence_factor(GPU_COMPUTE_CONFIDENCE_FACTOR),
            OperationSpec::new("allocate", "BufferId").with_params(vec!["Int"]),
            OperationSpec::new("copy_to_device", "Unit").with_params(vec!["Array", "BufferId"]),
            OperationSpec::new("copy_to_host", "Array")
                .with_params(vec!["BufferId", "Int"])
                .with_confidence_factor(GPU_COMPUTE_CONFIDENCE_FACTOR),
            OperationSpec::new("free", "Unit").with_params(vec!["BufferId"]),
            OperationSpec::new("get_device_info", "DeviceInfo").with_params(vec![]),
            OperationSpec::new("sync", "Unit").with_params(vec![]),
            // Batch control operations
            OperationSpec::new("begin_batch", "Unit").with_params(vec![]),
            OperationSpec::new("end_batch", "Unit").with_params(vec![]),
            OperationSpec::new("is_batching", "Bool").with_params(vec![]),
            OperationSpec::new("batch_size", "Int").with_params(vec![]),
            OperationSpec::new("abort_batch", "Unit").with_params(vec![]),
        ]
    })
}

/// Enum representing batched GPU operations.
#[derive(Debug, Clone)]
pub enum BatchedOp {
    /// Launch a kernel with specified dimensions and buffer bindings.
    Launch {
        kernel_id: i64,
        grid_dims: (u32, u32, u32),
        block_dims: (u32, u32, u32),
        buffer_bindings: Vec<i64>,
    },
    /// Allocate GPU memory with deferred ID assignment.
    Alloc {
        size_bytes: i64,
        placeholder_id: i64,
    },
    /// Copy data from host to device.
    CopyHtoD { host_data: Vec<f64>, buffer_id: i64 },
    /// Copy data from device to host, storing result at a pending index.
    CopyDtoH {
        buffer_id: i64,
        size_bytes: i64,
        result_index: usize,
    },
    /// Free GPU memory for the given buffer.
    Free { buffer_id: i64 },
}

/// Error types for batched GPU operations.
#[derive(Error, Debug, Clone)]
pub enum BatchError {
    /// Invalid buffer ID provided.
    #[error("invalid buffer ID: {0}")]
    InvalidBufferId(i64),
    /// Attempted operation while batch is not active.
    #[error("batch is not active")]
    BatchNotActive,
    /// Failed to flush the batch to the underlying handler.
    #[error("flush failed: {0}")]
    FlushFailed(String),
    /// Buffer was not allocated with the given ID.
    #[error("buffer not allocated with ID: {0}")]
    BufferNotAllocated(i64),
    /// Batch is already active.
    #[error("batch is already active")]
    BatchAlreadyActive,
    /// Handler error from underlying GPU handler.
    #[error("handler error: {0}")]
    HandlerError(String),
}

impl From<BatchError> for HandlerError {
    fn from(err: BatchError) -> Self {
        HandlerError::new("GPU", "batch", err.to_string())
    }
}

/// A GPU handler that batches operations for efficiency, deferring execution until flush.
///
/// This wrapper collects GPU operations during an active batch and executes them
/// all at once when `end_batch()` is called. Operations performed outside of a
/// batch are passed through immediately to the underlying `RealGpuHandler`.
///
/// # Example
///
/// ```ignore
/// let mut handler = BatchedGpuHandler::new();
/// let mut state = HandlerState::new();
///
/// // Start batching
/// handler.begin_batch();
///
/// // These are queued, not executed yet
/// let buf_id = handler.queue_allocate(1024);
/// handler.queue_copy_to_device(&data, buf_id);
/// handler.queue_launch(kernel, grid, block);
///
/// // Execute all operations with single sync
/// handler.end_batch(&mut state)?;
/// ```
#[derive(Debug)]
pub struct BatchedGpuHandler {
    /// The underlying real GPU handler.
    inner: RealGpuHandler,
    /// Whether a batch is currently active.
    batch_active: bool,
    /// Pending batched operations.
    pending_ops: Vec<BatchedOp>,
    /// Pending results from CopyDtoH operations (filled during flush).
    pending_results: Vec<Option<Value>>,
    /// Mapping from placeholder allocation IDs to real buffer IDs (assigned on flush).
    deferred_allocs: HashMap<i64, i64>,
    /// Next placeholder ID for deferred allocations.
    next_placeholder_id: i64,
}

impl BatchedGpuHandler {
    /// Create a new batched GPU handler.
    pub fn new() -> Self {
        Self {
            inner: RealGpuHandler::new(),
            batch_active: false,
            pending_ops: Vec::new(),
            pending_results: Vec::new(),
            deferred_allocs: HashMap::new(),
            next_placeholder_id: -1, // Negative IDs for placeholders
        }
    }

    /// Create a new batched GPU handler wrapping an existing RealGpuHandler.
    pub fn with_inner(inner: RealGpuHandler) -> Self {
        Self {
            inner,
            batch_active: false,
            pending_ops: Vec::new(),
            pending_results: Vec::new(),
            deferred_allocs: HashMap::new(),
            next_placeholder_id: -1,
        }
    }

    /// Check if batching is currently active.
    pub fn is_batching(&self) -> bool {
        self.batch_active
    }

    /// Get the number of pending operations in the current batch.
    pub fn batch_size(&self) -> usize {
        self.pending_ops.len()
    }

    /// Begin a new batch of operations.
    ///
    /// After calling this, GPU operations will be queued instead of executed
    /// immediately. Call `end_batch()` to execute all queued operations.
    ///
    /// # Errors
    ///
    /// Returns an error if a batch is already active.
    pub fn begin_batch(&mut self) -> Result<(), BatchError> {
        if self.batch_active {
            return Err(BatchError::BatchAlreadyActive);
        }

        self.batch_active = true;
        self.pending_ops.clear();
        self.pending_results.clear();
        self.deferred_allocs.clear();
        self.next_placeholder_id = -1;

        Ok(())
    }

    /// Abort the current batch without executing any operations.
    ///
    /// All queued operations are discarded.
    pub fn abort_batch(&mut self) {
        self.batch_active = false;
        self.pending_ops.clear();
        self.pending_results.clear();
        self.deferred_allocs.clear();
        self.next_placeholder_id = -1;
    }

    /// End the current batch and execute all queued operations.
    ///
    /// This method:
    /// 1. Executes all allocations first (to get real buffer IDs)
    /// 2. Remaps placeholder IDs to real IDs in all operations
    /// 3. Executes all copy and launch operations
    /// 4. Performs a single GPU synchronization
    /// 5. Collects results from device-to-host copies
    ///
    /// # Errors
    ///
    /// Returns an error if no batch is active or if any operation fails.
    pub fn end_batch(&mut self, state: &mut HandlerState) -> Result<Vec<Value>, BatchError> {
        if !self.batch_active {
            return Err(BatchError::BatchNotActive);
        }

        let ops = std::mem::take(&mut self.pending_ops);
        let mut results = Vec::new();

        // Phase 1: Execute all allocations to get real buffer IDs
        for op in &ops {
            if let BatchedOp::Alloc {
                size_bytes,
                placeholder_id,
            } = op
            {
                let result = self.inner.handle(
                    "allocate",
                    &[Value::Int(*size_bytes)],
                    Continuation::new(),
                    state,
                );

                match result {
                    HandlerResult::Resume(Value::Int(real_id)) => {
                        self.deferred_allocs.insert(*placeholder_id, real_id);
                    }
                    HandlerResult::Abort(e) => {
                        self.abort_batch();
                        return Err(BatchError::FlushFailed(e.message));
                    }
                    _ => {
                        self.abort_batch();
                        return Err(BatchError::FlushFailed(
                            "unexpected result from allocate".to_string(),
                        ));
                    }
                }
            }
        }

        // Phase 2: Execute copies and launches with remapped buffer IDs
        for op in &ops {
            match op {
                BatchedOp::Alloc { .. } => {
                    // Already handled in phase 1
                }
                BatchedOp::CopyHtoD {
                    host_data,
                    buffer_id,
                } => {
                    let real_id = self.resolve_buffer_id(*buffer_id)?;
                    let data_values: Vec<Value> =
                        host_data.iter().map(|&f| Value::Float(f)).collect();
                    let data = Value::Array(Rc::new(RefCell::new(data_values)));

                    let result = self.inner.handle(
                        "copy_to_device",
                        &[data, Value::Int(real_id)],
                        Continuation::new(),
                        state,
                    );

                    if let HandlerResult::Abort(e) = result {
                        self.abort_batch();
                        return Err(BatchError::FlushFailed(e.message));
                    }
                }
                BatchedOp::Launch {
                    kernel_id,
                    grid_dims,
                    block_dims,
                    buffer_bindings,
                } => {
                    // Remap buffer bindings
                    let mut remapped_bindings = Vec::with_capacity(buffer_bindings.len());
                    for &bid in buffer_bindings {
                        remapped_bindings.push(Value::Int(self.resolve_buffer_id(bid)?));
                    }

                    let grid = Value::Tuple(vec![
                        Value::Int(grid_dims.0 as i64),
                        Value::Int(grid_dims.1 as i64),
                        Value::Int(grid_dims.2 as i64),
                    ]);
                    let block = Value::Tuple(vec![
                        Value::Int(block_dims.0 as i64),
                        Value::Int(block_dims.1 as i64),
                        Value::Int(block_dims.2 as i64),
                    ]);
                    let bindings = Value::Array(Rc::new(RefCell::new(remapped_bindings)));

                    let result = self.inner.handle(
                        "launch",
                        &[Value::Int(*kernel_id), grid, block, bindings],
                        Continuation::new(),
                        state,
                    );

                    if let HandlerResult::Abort(e) = result {
                        self.abort_batch();
                        return Err(BatchError::FlushFailed(e.message));
                    }
                }
                BatchedOp::CopyDtoH {
                    buffer_id,
                    size_bytes,
                    result_index,
                } => {
                    let real_id = self.resolve_buffer_id(*buffer_id)?;

                    let result = self.inner.handle(
                        "copy_to_host",
                        &[Value::Int(real_id), Value::Int(*size_bytes)],
                        Continuation::new(),
                        state,
                    );

                    match result {
                        HandlerResult::Resume(value) => {
                            // Store result at the specified index
                            while results.len() <= *result_index {
                                results.push(Value::Unit);
                            }
                            results[*result_index] = value;
                        }
                        HandlerResult::Abort(e) => {
                            self.abort_batch();
                            return Err(BatchError::FlushFailed(e.message));
                        }
                        _ => {}
                    }
                }
                BatchedOp::Free { buffer_id } => {
                    let real_id = self.resolve_buffer_id(*buffer_id)?;

                    let result = self.inner.handle(
                        "free",
                        &[Value::Int(real_id)],
                        Continuation::new(),
                        state,
                    );

                    if let HandlerResult::Abort(e) = result {
                        self.abort_batch();
                        return Err(BatchError::FlushFailed(e.message));
                    }
                }
            }
        }

        // Phase 3: Synchronize GPU
        let sync_result = self.inner.handle("sync", &[], Continuation::new(), state);

        if let HandlerResult::Abort(e) = sync_result {
            self.abort_batch();
            return Err(BatchError::FlushFailed(format!(
                "sync failed: {}",
                e.message
            )));
        }

        // Reset batch state
        self.batch_active = false;
        self.pending_ops.clear();
        self.pending_results.clear();
        self.deferred_allocs.clear();
        self.next_placeholder_id = -1;

        Ok(results)
    }

    /// Resolve a buffer ID (either a placeholder or a real ID).
    fn resolve_buffer_id(&self, buffer_id: i64) -> Result<i64, BatchError> {
        if buffer_id < 0 {
            // Placeholder ID - look up the real ID
            self.deferred_allocs
                .get(&buffer_id)
                .copied()
                .ok_or(BatchError::BufferNotAllocated(buffer_id))
        } else {
            // Already a real ID
            Ok(buffer_id)
        }
    }

    /// Queue an allocation operation (returns placeholder ID).
    fn queue_allocate(&mut self, size_bytes: i64) -> i64 {
        let placeholder_id = self.next_placeholder_id;
        self.next_placeholder_id -= 1;

        self.pending_ops.push(BatchedOp::Alloc {
            size_bytes,
            placeholder_id,
        });

        placeholder_id
    }

    /// Queue a host-to-device copy operation.
    fn queue_copy_to_device(&mut self, data: Vec<f64>, buffer_id: i64) {
        self.pending_ops.push(BatchedOp::CopyHtoD {
            host_data: data,
            buffer_id,
        });
    }

    /// Queue a device-to-host copy operation.
    fn queue_copy_to_host(&mut self, buffer_id: i64, size_bytes: i64) -> usize {
        let result_index = self.pending_results.len();
        self.pending_results.push(None);

        self.pending_ops.push(BatchedOp::CopyDtoH {
            buffer_id,
            size_bytes,
            result_index,
        });

        result_index
    }

    /// Queue a kernel launch operation.
    fn queue_launch(
        &mut self,
        kernel_id: i64,
        grid_dims: (u32, u32, u32),
        block_dims: (u32, u32, u32),
        buffer_bindings: Vec<i64>,
    ) {
        self.pending_ops.push(BatchedOp::Launch {
            kernel_id,
            grid_dims,
            block_dims,
            buffer_bindings,
        });
    }

    /// Queue a free operation.
    fn queue_free(&mut self, buffer_id: i64) {
        self.pending_ops.push(BatchedOp::Free { buffer_id });
    }

    /// Extract integer from Value.
    fn extract_int(value: &Value, op: &str, param: &str) -> Result<i64, HandlerResult> {
        match value {
            Value::Int(n) => Ok(*n),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "GPU",
                op,
                format!("expected Int for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Extract grid/block dimensions from a tuple.
    fn extract_dims(value: &Value, name: &str) -> Result<(u32, u32, u32), HandlerResult> {
        match value {
            Value::Tuple(elems) if elems.len() >= 3 => {
                let x = Self::extract_u32(&elems[0], &format!("{}.x", name))?;
                let y = Self::extract_u32(&elems[1], &format!("{}.y", name))?;
                let z = Self::extract_u32(&elems[2], &format!("{}.z", name))?;
                Ok((x, y, z))
            }
            Value::Tuple(elems) if elems.len() == 1 => {
                let x = Self::extract_u32(&elems[0], &format!("{}.x", name))?;
                Ok((x, 1, 1))
            }
            Value::Int(n) => Ok((*n as u32, 1, 1)),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "GPU",
                "launch",
                format!(
                    "expected tuple of 3 u32 for {}, got {:?}",
                    name,
                    value.type_name()
                ),
            ))),
        }
    }

    /// Extract u32 from Value.
    fn extract_u32(value: &Value, name: &str) -> Result<u32, HandlerResult> {
        match value {
            Value::Int(n) if *n >= 0 => Ok(*n as u32),
            Value::Int(n) => Err(HandlerResult::Abort(HandlerError::new(
                "GPU",
                "launch",
                format!("{} must be non-negative, got {}", name, n),
            ))),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "GPU",
                "launch",
                format!("expected Int for {}, got {:?}", name, value.type_name()),
            ))),
        }
    }

    /// Extract f64 array from Value.
    fn extract_f64_array(value: &Value, op: &str) -> Result<Vec<f64>, HandlerResult> {
        match value {
            Value::Array(arr) => {
                let borrowed = arr.borrow();
                let mut result = Vec::with_capacity(borrowed.len());
                for v in borrowed.iter() {
                    match v {
                        Value::Float(f) => result.push(*f),
                        Value::Int(i) => result.push(*i as f64),
                        _ => {
                            return Err(HandlerResult::Abort(HandlerError::new(
                                "GPU",
                                op,
                                format!(
                                    "array element must be Float or Int, got {:?}",
                                    v.type_name()
                                ),
                            )));
                        }
                    }
                }
                Ok(result)
            }
            Value::Tensor { data, .. } => Ok(data.clone()),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "GPU",
                op,
                format!("expected Array or Tensor, got {:?}", value.type_name()),
            ))),
        }
    }

    /// Extract buffer bindings from an array of buffer IDs.
    fn extract_buffer_bindings(value: &Value, op: &str) -> Result<Vec<i64>, HandlerResult> {
        match value {
            Value::Array(arr) => {
                let borrowed = arr.borrow();
                let mut result = Vec::with_capacity(borrowed.len());
                for v in borrowed.iter() {
                    match v {
                        Value::Int(id) => result.push(*id),
                        _ => {
                            return Err(HandlerResult::Abort(HandlerError::new(
                                "GPU",
                                op,
                                format!("buffer binding must be Int, got {:?}", v.type_name()),
                            )));
                        }
                    }
                }
                Ok(result)
            }
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "GPU",
                op,
                format!("expected Array of buffer IDs, got {:?}", value.type_name()),
            ))),
        }
    }

    // === Handler Operations ===

    fn handle_begin_batch(&mut self) -> HandlerResult {
        match self.begin_batch() {
            Ok(()) => HandlerResult::Resume(Value::Unit),
            Err(e) => HandlerResult::Abort(e.into()),
        }
    }

    fn handle_end_batch(&mut self, state: &mut HandlerState) -> HandlerResult {
        match self.end_batch(state) {
            Ok(results) => {
                if results.is_empty() {
                    HandlerResult::Resume(Value::Unit)
                } else {
                    HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(results))))
                }
            }
            Err(e) => HandlerResult::Abort(e.into()),
        }
    }

    fn handle_abort_batch(&mut self) -> HandlerResult {
        self.abort_batch();
        HandlerResult::Resume(Value::Unit)
    }

    fn handle_is_batching(&self) -> HandlerResult {
        HandlerResult::Resume(Value::Bool(self.batch_active))
    }

    fn handle_batch_size(&self) -> HandlerResult {
        HandlerResult::Resume(Value::Int(self.pending_ops.len() as i64))
    }

    fn handle_allocate(&mut self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "allocate",
                "allocate requires size_bytes argument",
            ));
        }

        let size = match Self::extract_int(&args[0], "allocate", "size_bytes") {
            Ok(s) => s,
            Err(result) => return result,
        };

        if self.batch_active {
            // Queue the allocation and return a placeholder ID
            let placeholder_id = self.queue_allocate(size);
            HandlerResult::Resume(Value::Int(placeholder_id))
        } else {
            // Passthrough to inner handler
            self.inner
                .handle("allocate", args, Continuation::new(), state)
        }
    }

    fn handle_copy_to_device(&mut self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "copy_to_device",
                "copy_to_device requires host_data and buffer_id",
            ));
        }

        let data = match Self::extract_f64_array(&args[0], "copy_to_device") {
            Ok(d) => d,
            Err(result) => return result,
        };

        let buffer_id = match Self::extract_int(&args[1], "copy_to_device", "buffer_id") {
            Ok(id) => id,
            Err(result) => return result,
        };

        if self.batch_active {
            self.queue_copy_to_device(data, buffer_id);
            HandlerResult::Resume(Value::Unit)
        } else {
            self.inner
                .handle("copy_to_device", args, Continuation::new(), state)
        }
    }

    fn handle_copy_to_host(&mut self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "copy_to_host",
                "copy_to_host requires buffer_id and size_bytes",
            ));
        }

        let buffer_id = match Self::extract_int(&args[0], "copy_to_host", "buffer_id") {
            Ok(id) => id,
            Err(result) => return result,
        };

        let size = match Self::extract_int(&args[1], "copy_to_host", "size_bytes") {
            Ok(s) => s,
            Err(result) => return result,
        };

        if self.batch_active {
            // Queue and return a promise/future-like value
            // The actual result will be available after end_batch
            let result_index = self.queue_copy_to_host(buffer_id, size);
            // Return a "pending result" marker
            HandlerResult::Resume(Value::Tuple(vec![
                Value::String("pending_result".to_string()),
                Value::Int(result_index as i64),
            ]))
        } else {
            self.inner
                .handle("copy_to_host", args, Continuation::new(), state)
        }
    }

    fn handle_launch(&mut self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 3 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "launch",
                "launch requires kernel_id, grid_dims, and block_dims",
            ));
        }

        let kernel_id = match Self::extract_int(&args[0], "launch", "kernel_id") {
            Ok(id) => id,
            Err(result) => return result,
        };

        let grid_dims = match Self::extract_dims(&args[1], "grid_dims") {
            Ok(dims) => dims,
            Err(result) => return result,
        };

        let block_dims = match Self::extract_dims(&args[2], "block_dims") {
            Ok(dims) => dims,
            Err(result) => return result,
        };

        // Extract buffer bindings if provided
        let buffer_bindings = if args.len() > 3 {
            match Self::extract_buffer_bindings(&args[3], "launch") {
                Ok(bindings) => bindings,
                Err(result) => return result,
            }
        } else {
            Vec::new()
        };

        if self.batch_active {
            self.queue_launch(kernel_id, grid_dims, block_dims, buffer_bindings);
            HandlerResult::Resume(Value::Unit)
        } else {
            self.inner
                .handle("launch", args, Continuation::new(), state)
        }
    }

    fn handle_free(&mut self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "free",
                "free requires buffer_id argument",
            ));
        }

        let buffer_id = match Self::extract_int(&args[0], "free", "buffer_id") {
            Ok(id) => id,
            Err(result) => return result,
        };

        if self.batch_active {
            self.queue_free(buffer_id);
            HandlerResult::Resume(Value::Unit)
        } else {
            self.inner.handle("free", args, Continuation::new(), state)
        }
    }
}

impl Default for BatchedGpuHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl HandlerCapability for BatchedGpuHandler {
    fn effect_name(&self) -> &str {
        "GPU"
    }

    fn handler_name(&self) -> &str {
        "BatchedGpuHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_batched_gpu_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        // We need mutable access for batching operations
        // In a real implementation, we'd use RefCell or similar
        // For now, create a mutable copy of self's state
        let mut handler = Self::new();
        handler.batch_active = self.batch_active;
        handler.pending_ops = self.pending_ops.clone();
        handler.pending_results = self.pending_results.clone();
        handler.deferred_allocs = self.deferred_allocs.clone();
        handler.next_placeholder_id = self.next_placeholder_id;

        match op {
            // Batch control operations
            "begin_batch" => handler.handle_begin_batch(),
            "end_batch" => handler.handle_end_batch(state),
            "abort_batch" => handler.handle_abort_batch(),
            "is_batching" => handler.handle_is_batching(),
            "batch_size" => handler.handle_batch_size(),

            // GPU operations (batched or passthrough)
            "allocate" => handler.handle_allocate(args, state),
            "copy_to_device" => handler.handle_copy_to_device(args, state),
            "copy_to_host" => handler.handle_copy_to_host(args, state),
            "launch" => handler.handle_launch(args, state),
            "free" => handler.handle_free(args, state),

            // Passthrough operations (never batched)
            "compile_kernel" | "get_device_info" | "sync" => {
                handler.inner.handle(op, args, Continuation::new(), state)
            }

            _ => HandlerResult::Abort(HandlerError::new(
                "GPU",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        false // GPU operations have side effects
    }

    fn operation_linearity(&self, operation: &str) -> Linearity {
        match operation {
            // Batch control operations are exactly once
            "begin_batch" | "end_batch" | "abort_batch" => Linearity::ExactlyOnce,
            // Query operations are safe for multi-shot
            "is_batching" | "batch_size" | "get_device_info" => Linearity::MultiShot,
            // GPU operations modify state - exactly once
            _ => Linearity::ExactlyOnce,
        }
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        match operation {
            // Compute operations have small precision loss
            "launch" | "copy_to_host" => {
                EpistemicImpact::with_confidence(GPU_COMPUTE_CONFIDENCE_FACTOR)
            }
            // Batch operations don't affect confidence
            "begin_batch" | "end_batch" | "abort_batch" | "is_batching" | "batch_size" => {
                EpistemicImpact::none()
            }
            // Other operations don't affect confidence
            _ => EpistemicImpact::none(),
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
    fn test_handler_creation() {
        let handler = BatchedGpuHandler::new();
        assert_eq!(handler.effect_name(), "GPU");
        assert_eq!(handler.handler_name(), "BatchedGpuHandler");
        assert!(!handler.is_batching());
        assert_eq!(handler.batch_size(), 0);
    }

    #[test]
    fn test_begin_batch() {
        let mut handler = BatchedGpuHandler::new();
        assert!(!handler.is_batching());

        handler.begin_batch().expect("begin_batch should succeed");
        assert!(handler.is_batching());
    }

    #[test]
    fn test_begin_batch_already_active() {
        let mut handler = BatchedGpuHandler::new();
        handler
            .begin_batch()
            .expect("first begin_batch should succeed");

        let result = handler.begin_batch();
        assert!(matches!(result, Err(BatchError::BatchAlreadyActive)));
    }

    #[test]
    fn test_abort_batch() {
        let mut handler = BatchedGpuHandler::new();
        handler.begin_batch().expect("begin_batch should succeed");
        assert!(handler.is_batching());

        handler.abort_batch();
        assert!(!handler.is_batching());
        assert_eq!(handler.batch_size(), 0);
    }

    #[test]
    fn test_queue_allocate() {
        let mut handler = BatchedGpuHandler::new();
        handler.begin_batch().expect("begin_batch should succeed");

        let placeholder1 = handler.queue_allocate(1024);
        let placeholder2 = handler.queue_allocate(2048);

        // Placeholders should be negative
        assert!(placeholder1 < 0);
        assert!(placeholder2 < 0);
        assert_ne!(placeholder1, placeholder2);
        assert_eq!(handler.batch_size(), 2);
    }

    #[test]
    fn test_queue_operations() {
        let mut handler = BatchedGpuHandler::new();
        handler.begin_batch().expect("begin_batch should succeed");

        let buf_id = handler.queue_allocate(1024);
        handler.queue_copy_to_device(vec![1.0, 2.0, 3.0], buf_id);
        handler.queue_launch(1, (64, 1, 1), (256, 1, 1), vec![buf_id]);
        handler.queue_copy_to_host(buf_id, 12);
        handler.queue_free(buf_id);

        assert_eq!(handler.batch_size(), 5);
    }

    #[test]
    fn test_operations_list() {
        let handler = BatchedGpuHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();

        // Standard GPU operations
        assert!(names.contains(&"compile_kernel"));
        assert!(names.contains(&"launch"));
        assert!(names.contains(&"allocate"));
        assert!(names.contains(&"copy_to_device"));
        assert!(names.contains(&"copy_to_host"));
        assert!(names.contains(&"free"));
        assert!(names.contains(&"sync"));

        // Batch control operations
        assert!(names.contains(&"begin_batch"));
        assert!(names.contains(&"end_batch"));
        assert!(names.contains(&"is_batching"));
        assert!(names.contains(&"batch_size"));
        assert!(names.contains(&"abort_batch"));
    }

    #[test]
    fn test_no_multi_shot() {
        let handler = BatchedGpuHandler::new();
        assert!(!handler.supports_multi_shot());
    }

    #[test]
    fn test_operation_linearity() {
        let handler = BatchedGpuHandler::new();

        assert_eq!(
            handler.operation_linearity("begin_batch"),
            Linearity::ExactlyOnce
        );
        assert_eq!(
            handler.operation_linearity("end_batch"),
            Linearity::ExactlyOnce
        );
        assert_eq!(
            handler.operation_linearity("is_batching"),
            Linearity::MultiShot
        );
        assert_eq!(
            handler.operation_linearity("launch"),
            Linearity::ExactlyOnce
        );
    }

    #[test]
    fn test_epistemic_impact_launch() {
        let handler = BatchedGpuHandler::new();
        let impact = handler.epistemic_impact("launch");
        assert!((impact.confidence_factor - 0.99).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_impact_batch_ops() {
        let handler = BatchedGpuHandler::new();

        let begin_impact = handler.epistemic_impact("begin_batch");
        assert!((begin_impact.confidence_factor - 1.0).abs() < 0.001);

        let end_impact = handler.epistemic_impact("end_batch");
        assert!((end_impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_handler_via_trait() {
        let handler = BatchedGpuHandler::new();
        let mut state = new_state();

        // Test is_batching via trait interface
        let result = handler.handle("is_batching", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Bool(false)) => {}
            other => panic!("expected Resume(Bool(false)), got {:?}", other),
        }

        // Test batch_size via trait interface
        let result = handler.handle("batch_size", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Int(0)) => {}
            other => panic!("expected Resume(Int(0)), got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = BatchedGpuHandler::new();
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
    fn test_default_impl() {
        let handler = BatchedGpuHandler::default();
        assert_eq!(handler.effect_name(), "GPU");
        assert!(!handler.is_batching());
    }

    #[test]
    fn test_batch_error_display() {
        let err = BatchError::InvalidBufferId(42);
        assert_eq!(err.to_string(), "invalid buffer ID: 42");

        let err = BatchError::BatchNotActive;
        assert_eq!(err.to_string(), "batch is not active");

        let err = BatchError::FlushFailed("test error".to_string());
        assert_eq!(err.to_string(), "flush failed: test error");

        let err = BatchError::BufferNotAllocated(-1);
        assert_eq!(err.to_string(), "buffer not allocated with ID: -1");

        let err = BatchError::BatchAlreadyActive;
        assert_eq!(err.to_string(), "batch is already active");
    }

    #[test]
    fn test_handler_error_conversion() {
        let batch_err = BatchError::FlushFailed("test".to_string());
        let handler_err: HandlerError = batch_err.into();

        assert_eq!(handler_err.effect, "GPU");
        assert_eq!(handler_err.operation, "batch");
        assert!(handler_err.message.contains("flush failed"));
    }

    #[test]
    fn test_resolve_buffer_id_real() {
        let handler = BatchedGpuHandler::new();
        // Real IDs (positive) should pass through
        let result = handler.resolve_buffer_id(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_resolve_buffer_id_placeholder_not_found() {
        let handler = BatchedGpuHandler::new();
        // Placeholder IDs (negative) without mapping should fail
        let result = handler.resolve_buffer_id(-1);
        assert!(matches!(result, Err(BatchError::BufferNotAllocated(-1))));
    }
}
