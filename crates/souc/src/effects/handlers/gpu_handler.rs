//! GPU Effect Handler
//!
//! This module implements the `GPU` effect handler for GPU compute operations
//! through algebraic effects. The GPU effect enables kernel launches, memory
//! transfers, and synchronization with explicit effect tracking.
//!
//! # Effect Operations
//!
//! | Operation         | Arguments                              | Return Type | Confidence |
//! |------------------|----------------------------------------|-------------|------------|
//! | `launch`         | `(kernel, grid_dims, block_dims, args)`| `Unit`      | 0.99       |
//! | `sync`           | `()`                                   | `Unit`      | 1.0        |
//! | `alloc_device`   | `(size: Int)`                          | `DevicePtr` | 1.0        |
//! | `free_device`    | `(ptr: DevicePtr)`                     | `Unit`      | 1.0        |
//! | `copy_to_device` | `(host_data, device_ptr, size)`        | `Unit`      | 1.0        |
//! | `copy_from_device`| `(device_ptr, size)`                  | `Array`     | 0.99       |
//! | `get_device_count`| `()`                                  | `Int`       | 1.0        |
//! | `set_device`     | `(device_id: Int)`                     | `Unit`      | 1.0        |
//! | `stream_create`  | `()`                                   | `StreamId`  | 1.0        |
//! | `stream_sync`    | `(stream: StreamId)`                   | `Unit`      | 1.0        |
//! | `stream_destroy` | `(stream: StreamId)`                   | `Unit`      | 1.0        |
//!
//! # Epistemic Impact
//!
//! GPU operations have a small confidence degradation (0.99) due to:
//! - Floating-point precision differences between CPU and GPU
//! - Non-deterministic execution order in parallel kernels
//! - Potential for silent numerical errors
//!
//! Synchronization and memory management operations don't affect confidence
//! as they don't modify computational values.
//!
//! # Example
//!
//! ```text
//! // Sounio code using GPU effect:
//! fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32> with GPU {
//!     let n = a.len()
//!     let d_a = alloc_device(n * 4)
//!     let d_b = alloc_device(n * 4)
//!     let d_c = alloc_device(n * 4)
//!
//!     copy_to_device(a, d_a, n * 4)
//!     copy_to_device(b, d_b, n * 4)
//!
//!     let grid = (n + 255) / 256
//!     launch(vector_add_kernel, (grid, 1, 1), (256, 1, 1), [d_a, d_b, d_c, n])
//!     sync()
//!
//!     let result = copy_from_device(d_c, n * 4)
//!     free_device(d_a)
//!     free_device(d_b)
//!     free_device(d_c)
//!     result
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use std::sync::OnceLock;

/// Confidence factor for GPU compute operations (small numerical precision loss)
const GPU_COMPUTE_CONFIDENCE_FACTOR: f64 = 0.99;

/// Key for storing device allocations in HandlerState
const GPU_ALLOCATIONS_KEY: &str = "__gpu_allocations";

/// Key for storing current device ID
const GPU_CURRENT_DEVICE_KEY: &str = "__gpu_current_device";

/// Key for storing stream counter
const GPU_STREAM_COUNTER_KEY: &str = "__gpu_stream_counter";

/// Key for storing active streams
const GPU_ACTIVE_STREAMS_KEY: &str = "__gpu_active_streams";

/// Key for storing pending kernel launches
const GPU_PENDING_KERNELS_KEY: &str = "__gpu_pending_kernels";

/// Key for tracking total allocated bytes
const GPU_TOTAL_ALLOCATED_KEY: &str = "__gpu_total_allocated";

/// Static operation specifications for GpuHandler
static GPU_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_gpu_operations() -> &'static [OperationSpec] {
    GPU_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("launch", "Unit")
                .with_params(vec!["Kernel", "Tuple", "Tuple", "Array"])
                .with_confidence_factor(GPU_COMPUTE_CONFIDENCE_FACTOR),
            OperationSpec::new("sync", "Unit").with_params(vec![]),
            OperationSpec::new("alloc_device", "DevicePtr").with_params(vec!["Int"]),
            OperationSpec::new("free_device", "Unit").with_params(vec!["DevicePtr"]),
            OperationSpec::new("copy_to_device", "Unit").with_params(vec![
                "Array",
                "DevicePtr",
                "Int",
            ]),
            OperationSpec::new("copy_from_device", "Array")
                .with_params(vec!["DevicePtr", "Int"])
                .with_confidence_factor(GPU_COMPUTE_CONFIDENCE_FACTOR),
            OperationSpec::new("get_device_count", "Int").with_params(vec![]),
            OperationSpec::new("set_device", "Unit").with_params(vec!["Int"]),
            OperationSpec::new("stream_create", "StreamId").with_params(vec![]),
            OperationSpec::new("stream_sync", "Unit").with_params(vec!["StreamId"]),
            OperationSpec::new("stream_destroy", "Unit").with_params(vec!["StreamId"]),
            OperationSpec::new("get_device_properties", "DeviceProperties")
                .with_params(vec!["Int"]),
            OperationSpec::new("memset_device", "Unit").with_params(vec![
                "DevicePtr",
                "Int",
                "Int",
            ]),
        ]
    })
}

/// Simulated GPU device pointer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(u64);

/// Simulated GPU stream ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(u64);

/// GPU allocation tracking
#[derive(Debug, Clone)]
struct GpuAllocation {
    ptr: u64,
    size: i64,
    device_id: i64,
}

/// Handler for the GPU effect
///
/// GpuHandler provides GPU compute operations through algebraic effects.
/// It simulates GPU memory management and kernel execution for testing
/// and interpretation purposes.
///
/// # State Tracking
///
/// The handler tracks:
/// - Device memory allocations
/// - Active streams
/// - Pending kernel launches
/// - Current device selection
///
/// # Thread Safety
///
/// The handler is designed for single-threaded interpretation. For actual
/// GPU execution, this handler would delegate to CUDA/Metal/Vulkan backends.
#[derive(Debug)]
pub struct GpuHandler {
    /// Simulated number of GPU devices
    device_count: i64,
}

impl Default for GpuHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuHandler {
    /// Create a new GpuHandler with default settings
    pub fn new() -> Self {
        Self { device_count: 1 }
    }

    /// Create a GpuHandler with a specific device count (for testing)
    pub fn with_device_count(device_count: i64) -> Self {
        Self { device_count }
    }

    /// Get or initialize allocation counter
    fn get_alloc_counter(state: &mut HandlerState) -> u64 {
        match state.named_state.get(GPU_ALLOCATIONS_KEY) {
            Some(Value::Int(n)) => *n as u64,
            _ => {
                state
                    .named_state
                    .insert(GPU_ALLOCATIONS_KEY.to_string(), Value::Int(1000));
                1000
            }
        }
    }

    /// Increment allocation counter and return new pointer
    fn next_device_ptr(state: &mut HandlerState) -> u64 {
        let ptr = Self::get_alloc_counter(state);
        state.named_state.insert(
            GPU_ALLOCATIONS_KEY.to_string(),
            Value::Int((ptr + 1) as i64),
        );
        ptr
    }

    /// Get current device ID
    fn get_current_device(state: &HandlerState) -> i64 {
        match state.named_state.get(GPU_CURRENT_DEVICE_KEY) {
            Some(Value::Int(d)) => *d,
            _ => 0,
        }
    }

    /// Get stream counter
    fn get_stream_counter(state: &mut HandlerState) -> u64 {
        match state.named_state.get(GPU_STREAM_COUNTER_KEY) {
            Some(Value::Int(n)) => *n as u64,
            _ => {
                state
                    .named_state
                    .insert(GPU_STREAM_COUNTER_KEY.to_string(), Value::Int(1));
                1
            }
        }
    }

    /// Increment stream counter
    fn next_stream_id(state: &mut HandlerState) -> u64 {
        let id = Self::get_stream_counter(state);
        state.named_state.insert(
            GPU_STREAM_COUNTER_KEY.to_string(),
            Value::Int((id + 1) as i64),
        );
        id
    }

    /// Get pending kernel count
    fn get_pending_kernels(state: &HandlerState) -> i64 {
        match state.named_state.get(GPU_PENDING_KERNELS_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        }
    }

    /// Add pending kernel
    fn add_pending_kernel(state: &mut HandlerState) {
        let current = Self::get_pending_kernels(state);
        state
            .named_state
            .insert(GPU_PENDING_KERNELS_KEY.to_string(), Value::Int(current + 1));
    }

    /// Clear pending kernels
    fn clear_pending_kernels(state: &mut HandlerState) {
        state
            .named_state
            .insert(GPU_PENDING_KERNELS_KEY.to_string(), Value::Int(0));
    }

    /// Get total allocated bytes
    fn get_total_allocated(state: &HandlerState) -> i64 {
        match state.named_state.get(GPU_TOTAL_ALLOCATED_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        }
    }

    /// Update total allocated bytes
    fn update_total_allocated(state: &mut HandlerState, delta: i64) {
        let current = Self::get_total_allocated(state);
        state.named_state.insert(
            GPU_TOTAL_ALLOCATED_KEY.to_string(),
            Value::Int(current + delta),
        );
    }

    /// Extract grid/block dimensions from a tuple
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

    /// Extract u32 from Value
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

    /// Extract integer from Value
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

    /// Handle launch operation
    fn handle_launch(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 3 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "launch",
                "launch requires at least kernel, grid_dims, and block_dims",
            ));
        }

        // Validate kernel (we accept any value for simulation)
        let _kernel = &args[0];

        // Extract grid dimensions
        let grid_dims = match Self::extract_dims(&args[1], "grid_dims") {
            Ok(dims) => dims,
            Err(result) => return result,
        };

        // Extract block dimensions
        let block_dims = match Self::extract_dims(&args[2], "block_dims") {
            Ok(dims) => dims,
            Err(result) => return result,
        };

        // Validate dimensions
        if grid_dims.0 == 0 || grid_dims.1 == 0 || grid_dims.2 == 0 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "launch",
                "grid dimensions must be non-zero",
            ));
        }

        if block_dims.0 == 0 || block_dims.1 == 0 || block_dims.2 == 0 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "launch",
                "block dimensions must be non-zero",
            ));
        }

        // Check block size limits (typical GPU limit)
        let threads_per_block = block_dims.0 as u64 * block_dims.1 as u64 * block_dims.2 as u64;
        if threads_per_block > 1024 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "launch",
                format!(
                    "threads per block ({}) exceeds maximum (1024)",
                    threads_per_block
                ),
            ));
        }

        // Record pending kernel
        Self::add_pending_kernel(state);

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle sync operation
    fn handle_sync(&self, state: &mut HandlerState) -> HandlerResult {
        // Clear all pending kernels
        Self::clear_pending_kernels(state);
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle alloc_device operation
    fn handle_alloc_device(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "alloc_device",
                "alloc_device requires size argument",
            ));
        }

        let size = match Self::extract_int(&args[0], "alloc_device", "size") {
            Ok(s) => s,
            Err(result) => return result,
        };

        if size <= 0 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "alloc_device",
                format!("allocation size must be positive, got {}", size),
            ));
        }

        let ptr = Self::next_device_ptr(state);
        Self::update_total_allocated(state, size);

        // Return pointer as Int (simulated device pointer)
        HandlerResult::Resume(Value::Int(ptr as i64))
    }

    /// Handle free_device operation
    fn handle_free_device(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "free_device",
                "free_device requires device pointer argument",
            ));
        }

        let _ptr = match Self::extract_int(&args[0], "free_device", "ptr") {
            Ok(p) => p,
            Err(result) => return result,
        };

        // In a real implementation, we would track allocations and validate
        // For simulation, we just accept the operation

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle copy_to_device operation
    fn handle_copy_to_device(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 3 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "copy_to_device",
                "copy_to_device requires host_data, device_ptr, and size",
            ));
        }

        // Validate host data is an array
        match &args[0] {
            Value::Array(_) => {}
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "GPU",
                    "copy_to_device",
                    format!(
                        "expected Array for host_data, got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        }

        // Validate device pointer
        let _device_ptr = match Self::extract_int(&args[1], "copy_to_device", "device_ptr") {
            Ok(p) => p,
            Err(result) => return result,
        };

        // Validate size
        let size = match Self::extract_int(&args[2], "copy_to_device", "size") {
            Ok(s) => s,
            Err(result) => return result,
        };

        if size < 0 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "copy_to_device",
                format!("size must be non-negative, got {}", size),
            ));
        }

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle copy_from_device operation
    fn handle_copy_from_device(&self, args: &[Value]) -> HandlerResult {
        use std::cell::RefCell;
        use std::rc::Rc;

        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "copy_from_device",
                "copy_from_device requires device_ptr and size",
            ));
        }

        // Validate device pointer
        let _device_ptr = match Self::extract_int(&args[0], "copy_from_device", "device_ptr") {
            Ok(p) => p,
            Err(result) => return result,
        };

        // Validate size
        let size = match Self::extract_int(&args[1], "copy_from_device", "size") {
            Ok(s) => s,
            Err(result) => return result,
        };

        if size < 0 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "copy_from_device",
                format!("size must be non-negative, got {}", size),
            ));
        }

        // Return a simulated array (in real implementation, this would copy actual data)
        let data = vec![Value::Float(0.0); size as usize];
        HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(data))))
    }

    /// Handle get_device_count operation
    fn handle_get_device_count(&self) -> HandlerResult {
        HandlerResult::Resume(Value::Int(self.device_count))
    }

    /// Handle set_device operation
    fn handle_set_device(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "set_device",
                "set_device requires device_id argument",
            ));
        }

        let device_id = match Self::extract_int(&args[0], "set_device", "device_id") {
            Ok(d) => d,
            Err(result) => return result,
        };

        if device_id < 0 || device_id >= self.device_count {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "set_device",
                format!(
                    "device_id {} out of range [0, {})",
                    device_id, self.device_count
                ),
            ));
        }

        state
            .named_state
            .insert(GPU_CURRENT_DEVICE_KEY.to_string(), Value::Int(device_id));
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle stream_create operation
    fn handle_stream_create(&self, state: &mut HandlerState) -> HandlerResult {
        let stream_id = Self::next_stream_id(state);
        HandlerResult::Resume(Value::Int(stream_id as i64))
    }

    /// Handle stream_sync operation
    fn handle_stream_sync(&self, args: &[Value]) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "stream_sync",
                "stream_sync requires stream_id argument",
            ));
        }

        let _stream_id = match Self::extract_int(&args[0], "stream_sync", "stream_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        // In real implementation, would wait for stream to complete
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle stream_destroy operation
    fn handle_stream_destroy(&self, args: &[Value]) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "stream_destroy",
                "stream_destroy requires stream_id argument",
            ));
        }

        let _stream_id = match Self::extract_int(&args[0], "stream_destroy", "stream_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        // In real implementation, would destroy the stream
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_device_properties operation
    fn handle_get_device_properties(&self, args: &[Value]) -> HandlerResult {
        use std::cell::RefCell;
        use std::rc::Rc;

        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "get_device_properties",
                "get_device_properties requires device_id argument",
            ));
        }

        let device_id = match Self::extract_int(&args[0], "get_device_properties", "device_id") {
            Ok(d) => d,
            Err(result) => return result,
        };

        if device_id < 0 || device_id >= self.device_count {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "get_device_properties",
                format!(
                    "device_id {} out of range [0, {})",
                    device_id, self.device_count
                ),
            ));
        }

        // Return simulated device properties as a record/tuple
        // In real implementation, would query actual GPU properties
        let properties = vec![
            (
                "name".to_string(),
                Value::String("Simulated GPU".to_string()),
            ),
            (
                "total_memory".to_string(),
                Value::Int(8 * 1024 * 1024 * 1024),
            ), // 8GB
            ("compute_capability_major".to_string(), Value::Int(8)),
            ("compute_capability_minor".to_string(), Value::Int(6)),
            ("max_threads_per_block".to_string(), Value::Int(1024)),
            ("max_grid_size_x".to_string(), Value::Int(2147483647)),
            ("max_grid_size_y".to_string(), Value::Int(65535)),
            ("max_grid_size_z".to_string(), Value::Int(65535)),
            ("warp_size".to_string(), Value::Int(32)),
            ("multiprocessor_count".to_string(), Value::Int(84)),
        ];

        // Return as array of tuples for simplicity
        let props_values: Vec<Value> = properties
            .into_iter()
            .map(|(k, v)| Value::Tuple(vec![Value::String(k), v]))
            .collect();

        HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(props_values))))
    }

    /// Handle memset_device operation
    fn handle_memset_device(&self, args: &[Value]) -> HandlerResult {
        if args.len() < 3 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "memset_device",
                "memset_device requires device_ptr, value, and size",
            ));
        }

        let _device_ptr = match Self::extract_int(&args[0], "memset_device", "device_ptr") {
            Ok(p) => p,
            Err(result) => return result,
        };

        let _value = match Self::extract_int(&args[1], "memset_device", "value") {
            Ok(v) => v,
            Err(result) => return result,
        };

        let size = match Self::extract_int(&args[2], "memset_device", "size") {
            Ok(s) => s,
            Err(result) => return result,
        };

        if size < 0 {
            return HandlerResult::Abort(HandlerError::new(
                "GPU",
                "memset_device",
                format!("size must be non-negative, got {}", size),
            ));
        }

        HandlerResult::Resume(Value::Unit)
    }

    /// Get the number of pending kernel launches (for testing)
    pub fn pending_kernel_count(state: &HandlerState) -> i64 {
        Self::get_pending_kernels(state)
    }

    /// Get total allocated device memory (for testing)
    pub fn total_allocated_memory(state: &HandlerState) -> i64 {
        Self::get_total_allocated(state)
    }

    /// Get current device ID (for testing)
    pub fn current_device(state: &HandlerState) -> i64 {
        Self::get_current_device(state)
    }
}

impl HandlerCapability for GpuHandler {
    fn effect_name(&self) -> &str {
        "GPU"
    }

    fn handler_name(&self) -> &str {
        "GpuHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_gpu_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "launch" => self.handle_launch(args, state),
            "sync" => self.handle_sync(state),
            "alloc_device" => self.handle_alloc_device(args, state),
            "free_device" => self.handle_free_device(args, state),
            "copy_to_device" => self.handle_copy_to_device(args),
            "copy_from_device" => self.handle_copy_from_device(args),
            "get_device_count" => self.handle_get_device_count(),
            "set_device" => self.handle_set_device(args, state),
            "stream_create" => self.handle_stream_create(state),
            "stream_sync" => self.handle_stream_sync(args),
            "stream_destroy" => self.handle_stream_destroy(args),
            "get_device_properties" => self.handle_get_device_properties(args),
            "memset_device" => self.handle_memset_device(args),
            _ => HandlerResult::Abort(HandlerError::new(
                "GPU",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        // GPU operations are generally not safe for multi-shot due to
        // side effects on device memory
        false
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        match operation {
            // Compute operations have small precision loss
            "launch" | "copy_from_device" => {
                EpistemicImpact::with_confidence(GPU_COMPUTE_CONFIDENCE_FACTOR)
            }
            // Memory and sync operations don't affect confidence
            _ => EpistemicImpact::none(),
        }
    }

    fn operation_linearity(&self, operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        // GPU operations modify device state - mostly one-shot
        match operation {
            "launch" | "sync" | "alloc_device" | "free_device" | "copy_to_device"
            | "copy_from_device" | "set_device" | "stream_create" | "stream_sync"
            | "stream_destroy" | "memset_device" => Linearity::ExactlyOnce,
            // Query operations are safe for multi-shot
            "get_device_count" | "get_device_properties" => Linearity::MultiShot,
            _ => Linearity::ExactlyOnce,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn new_state() -> HandlerState {
        HandlerState::new()
    }

    fn cont() -> Continuation {
        Continuation::new()
    }

    #[test]
    fn test_handler_identity() {
        let handler = GpuHandler::new();
        assert_eq!(handler.effect_name(), "GPU");
        assert_eq!(handler.handler_name(), "GpuHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = GpuHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"launch"));
        assert!(names.contains(&"sync"));
        assert!(names.contains(&"alloc_device"));
        assert!(names.contains(&"free_device"));
        assert!(names.contains(&"copy_to_device"));
        assert!(names.contains(&"copy_from_device"));
        assert!(names.contains(&"get_device_count"));
        assert!(names.contains(&"set_device"));
    }

    #[test]
    fn test_no_multi_shot() {
        let handler = GpuHandler::new();
        assert!(!handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact_launch() {
        let handler = GpuHandler::new();
        let impact = handler.epistemic_impact("launch");
        assert!((impact.confidence_factor - 0.99).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_impact_sync() {
        let handler = GpuHandler::new();
        let impact = handler.epistemic_impact("sync");
        assert!((impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_launch_basic() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let kernel = Value::String("test_kernel".to_string());
        let grid = Value::Tuple(vec![Value::Int(16), Value::Int(1), Value::Int(1)]);
        let block = Value::Tuple(vec![Value::Int(256), Value::Int(1), Value::Int(1)]);

        let result = handler.handle("launch", &[kernel, grid, block], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        assert_eq!(GpuHandler::pending_kernel_count(&state), 1);
    }

    #[test]
    fn test_launch_with_args() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let kernel = Value::String("vector_add".to_string());
        let grid = Value::Tuple(vec![Value::Int(4), Value::Int(1), Value::Int(1)]);
        let block = Value::Tuple(vec![Value::Int(128), Value::Int(1), Value::Int(1)]);
        let args = Value::Array(Rc::new(RefCell::new(vec![
            Value::Int(1000), // d_a
            Value::Int(1001), // d_b
            Value::Int(1002), // d_c
        ])));

        let result = handler.handle("launch", &[kernel, grid, block, args], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_launch_zero_grid() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let kernel = Value::String("test".to_string());
        let grid = Value::Tuple(vec![Value::Int(0), Value::Int(1), Value::Int(1)]);
        let block = Value::Tuple(vec![Value::Int(256), Value::Int(1), Value::Int(1)]);

        let result = handler.handle("launch", &[kernel, grid, block], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("non-zero"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_launch_exceeds_block_limit() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let kernel = Value::String("test".to_string());
        let grid = Value::Tuple(vec![Value::Int(1), Value::Int(1), Value::Int(1)]);
        // 32 * 32 * 2 = 2048 > 1024
        let block = Value::Tuple(vec![Value::Int(32), Value::Int(32), Value::Int(2)]);

        let result = handler.handle("launch", &[kernel, grid, block], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("exceeds maximum"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_sync_clears_pending() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        // Launch a few kernels
        let kernel = Value::String("test".to_string());
        let grid = Value::Int(16);
        let block = Value::Int(256);

        handler.handle(
            "launch",
            &[kernel.clone(), grid.clone(), block.clone()],
            cont(),
            &mut state,
        );
        handler.handle(
            "launch",
            &[kernel.clone(), grid.clone(), block.clone()],
            cont(),
            &mut state,
        );

        assert_eq!(GpuHandler::pending_kernel_count(&state), 2);

        // Sync
        let result = handler.handle("sync", &[], cont(), &mut state);
        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        assert_eq!(GpuHandler::pending_kernel_count(&state), 0);
    }

    #[test]
    fn test_alloc_device() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let result = handler.handle("alloc_device", &[Value::Int(1024)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(ptr)) => {
                assert!(ptr >= 1000); // Our starting pointer
            }
            other => panic!("expected Resume(Int), got {:?}", other),
        }

        assert_eq!(GpuHandler::total_allocated_memory(&state), 1024);
    }

    #[test]
    fn test_alloc_device_zero() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let result = handler.handle("alloc_device", &[Value::Int(0)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("positive"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_free_device() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let result = handler.handle("free_device", &[Value::Int(1000)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_copy_to_device() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let data = Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(1.0),
            Value::Float(2.0),
        ])));
        let ptr = Value::Int(1000);
        let size = Value::Int(8);

        let result = handler.handle("copy_to_device", &[data, ptr, size], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_copy_from_device() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let ptr = Value::Int(1000);
        let size = Value::Int(4);

        let result = handler.handle("copy_from_device", &[ptr, size], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Array(arr)) => {
                assert_eq!(arr.borrow().len(), 4);
            }
            other => panic!("expected Resume(Array), got {:?}", other),
        }
    }

    #[test]
    fn test_get_device_count() {
        let handler = GpuHandler::with_device_count(4);
        let mut state = new_state();

        let result = handler.handle("get_device_count", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(count)) => {
                assert_eq!(count, 4);
            }
            other => panic!("expected Resume(Int(4)), got {:?}", other),
        }
    }

    #[test]
    fn test_set_device() {
        let handler = GpuHandler::with_device_count(4);
        let mut state = new_state();

        let result = handler.handle("set_device", &[Value::Int(2)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        assert_eq!(GpuHandler::current_device(&state), 2);
    }

    #[test]
    fn test_set_device_out_of_range() {
        let handler = GpuHandler::with_device_count(2);
        let mut state = new_state();

        let result = handler.handle("set_device", &[Value::Int(5)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("out of range"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_stream_create() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let result = handler.handle("stream_create", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(id)) => {
                assert!(id >= 1);
            }
            other => panic!("expected Resume(Int), got {:?}", other),
        }
    }

    #[test]
    fn test_stream_sync() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let result = handler.handle("stream_sync", &[Value::Int(1)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_stream_destroy() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let result = handler.handle("stream_destroy", &[Value::Int(1)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_get_device_properties() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "get_device_properties",
            &[Value::Int(0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Array(props)) => {
                assert!(!props.borrow().is_empty());
            }
            other => panic!("expected Resume(Array), got {:?}", other),
        }
    }

    #[test]
    fn test_memset_device() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "memset_device",
            &[Value::Int(1000), Value::Int(0), Value::Int(1024)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = GpuHandler::new();
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
        let handler = GpuHandler::default();
        assert_eq!(handler.effect_name(), "GPU");
    }

    #[test]
    fn test_multiple_allocations() {
        let handler = GpuHandler::new();
        let mut state = new_state();

        // Allocate multiple buffers
        handler.handle("alloc_device", &[Value::Int(1024)], cont(), &mut state);
        handler.handle("alloc_device", &[Value::Int(2048)], cont(), &mut state);
        handler.handle("alloc_device", &[Value::Int(512)], cont(), &mut state);

        assert_eq!(
            GpuHandler::total_allocated_memory(&state),
            1024 + 2048 + 512
        );
    }
}
