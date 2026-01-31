//! GPU Epistemic Kernels - WGSL Compute Shaders for Knowledge<T> Operations
//!
//! This module provides GPU-accelerated epistemic operations using WGPU/WGSL.
//! It enables data-parallel uncertainty propagation for large arrays of Knowledge values.
//!
//! # Features
//!
//! - Parallel uncertainty propagation for thousands of Knowledge values
//! - GPU-accelerated confidence computation with workgroup reductions
//! - Element-wise epistemic operations (add, multiply, transform)
//! - Integration with RealGpuHandler for effect system
//!
//! # Architecture
//!
//! ```text
//! Knowledge<T> Arrays -> GPU Buffers -> WGSL Shaders -> Result Arrays
//!                           |                |
//!                      GpuEpistemicOps   EpistemicKernelId
//! ```

use std::collections::HashMap;

/// Size of a Knowledge struct in bytes (4 x f32 = 16 bytes)
pub const KNOWLEDGE_STRUCT_SIZE: usize = 16;

/// Default workgroup size for epistemic kernels
pub const DEFAULT_WORKGROUP_SIZE: u32 = 256;

/// Confidence factor for GPU epistemic operations (numerical precision)
pub const GPU_EPISTEMIC_CONFIDENCE_FACTOR: f64 = 0.99;

// ============================================================================
// WGSL SHADER SOURCES
// ============================================================================

/// WGSL shader for element-wise uncertainty addition: sqrt(sigma1^2 + sigma2^2)
pub const WGSL_UNCERTAINTY_ADD: &str = r#"
struct Knowledge {
    value: f32,
    uncertainty: f32,
    confidence: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> input_a: array<Knowledge>;
@group(0) @binding(1) var<storage, read> input_b: array<Knowledge>;
@group(0) @binding(2) var<storage, read_write> output: array<Knowledge>;

@compute @workgroup_size(256)
fn uncertainty_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&output);
    if (idx >= len) {
        return;
    }

    let a = input_a[idx];
    let b = input_b[idx];

    // Value addition
    output[idx].value = a.value + b.value;

    // Uncertainty propagation: sqrt(sigma_a^2 + sigma_b^2)
    let unc_sq = a.uncertainty * a.uncertainty + b.uncertainty * b.uncertainty;
    output[idx].uncertainty = sqrt(max(unc_sq, 0.0));

    // Confidence: minimum of inputs
    output[idx].confidence = min(a.confidence, b.confidence);
    output[idx]._padding = 0.0;
}
"#;

/// WGSL shader for element-wise uncertainty multiplication (relative error propagation)
pub const WGSL_UNCERTAINTY_MUL: &str = r#"
struct Knowledge {
    value: f32,
    uncertainty: f32,
    confidence: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> input_a: array<Knowledge>;
@group(0) @binding(1) var<storage, read> input_b: array<Knowledge>;
@group(0) @binding(2) var<storage, read_write> output: array<Knowledge>;

@compute @workgroup_size(256)
fn uncertainty_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&output);
    if (idx >= len) {
        return;
    }

    let a = input_a[idx];
    let b = input_b[idx];

    // Value multiplication
    let result_value = a.value * b.value;
    output[idx].value = result_value;

    // Relative error propagation: sqrt((da/a)^2 + (db/b)^2) * |result|
    let abs_a = abs(a.value);
    let abs_b = abs(b.value);
    let eps = 1e-10;

    let rel_a = select(0.0, a.uncertainty / abs_a, abs_a > eps);
    let rel_b = select(0.0, b.uncertainty / abs_b, abs_b > eps);

    let rel_sq = rel_a * rel_a + rel_b * rel_b;
    output[idx].uncertainty = sqrt(max(rel_sq, 0.0)) * abs(result_value);

    // Confidence: minimum of inputs
    output[idx].confidence = min(a.confidence, b.confidence);
    output[idx]._padding = 0.0;
}
"#;

/// WGSL shader for parallel confidence aggregation (reduction)
pub const WGSL_CONFIDENCE_AGGREGATE: &str = r#"
struct Knowledge {
    value: f32,
    uncertainty: f32,
    confidence: f32,
    _padding: f32,
}

struct AggregateParams {
    mode: u32,      // 0=min, 1=product, 2=quadrature
    count: u32,     // number of elements
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<Knowledge>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: AggregateParams;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn confidence_aggregate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let mode = params.mode;
    let count = params.count;

    // Load confidence value or identity
    var val: f32;
    if (gid < count) {
        val = input[gid].confidence;
    } else {
        // Identity values for each mode
        if (mode == 0u) {
            val = 1.0; // min identity
        } else if (mode == 1u) {
            val = 1.0; // product identity
        } else {
            val = 0.0; // quadrature identity (squared)
        }
    }

    // For quadrature, store squared value
    if (mode == 2u && gid < count) {
        val = val * val;
    }

    shared_data[tid] = val;
    workgroupBarrier();

    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let other = shared_data[tid + stride];
            if (mode == 0u) {
                shared_data[tid] = min(shared_data[tid], other);
            } else if (mode == 1u) {
                shared_data[tid] = shared_data[tid] * other;
            } else {
                shared_data[tid] = shared_data[tid] + other;
            }
        }
        workgroupBarrier();
    }

    // First thread writes result
    if (tid == 0u) {
        var result = shared_data[0];
        if (mode == 2u) {
            // Quadrature: sqrt of mean of squares
            let n = f32(min(256u, count - wg_id.x * 256u));
            result = sqrt(result / max(n, 1.0));
        }
        output[wg_id.x] = result;
    }
}
"#;

/// WGSL shader for applying scalar transformation with uncertainty propagation
pub const WGSL_KNOWLEDGE_TRANSFORM: &str = r#"
struct Knowledge {
    value: f32,
    uncertainty: f32,
    confidence: f32,
    _padding: f32,
}

struct TransformParams {
    scale: f32,
    offset: f32,
    confidence_factor: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> input: array<Knowledge>;
@group(0) @binding(1) var<storage, read_write> output: array<Knowledge>;
@group(0) @binding(2) var<uniform> params: TransformParams;

@compute @workgroup_size(256)
fn knowledge_transform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&output);
    if (idx >= len) {
        return;
    }

    let k = input[idx];

    // Apply affine transformation: y = scale * x + offset
    output[idx].value = params.scale * k.value + params.offset;

    // Uncertainty propagation: |scale| * uncertainty
    output[idx].uncertainty = abs(params.scale) * k.uncertainty;

    // Apply confidence degradation factor
    output[idx].confidence = k.confidence * params.confidence_factor;
    output[idx]._padding = 0.0;
}
"#;

/// WGSL shader for element-wise uncertainty subtraction
pub const WGSL_UNCERTAINTY_SUB: &str = r#"
struct Knowledge {
    value: f32,
    uncertainty: f32,
    confidence: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> input_a: array<Knowledge>;
@group(0) @binding(1) var<storage, read> input_b: array<Knowledge>;
@group(0) @binding(2) var<storage, read_write> output: array<Knowledge>;

@compute @workgroup_size(256)
fn uncertainty_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&output);
    if (idx >= len) {
        return;
    }

    let a = input_a[idx];
    let b = input_b[idx];

    // Value subtraction
    output[idx].value = a.value - b.value;

    // Uncertainty propagation: sqrt(sigma_a^2 + sigma_b^2) (same as addition)
    let unc_sq = a.uncertainty * a.uncertainty + b.uncertainty * b.uncertainty;
    output[idx].uncertainty = sqrt(max(unc_sq, 0.0));

    // Confidence: minimum of inputs
    output[idx].confidence = min(a.confidence, b.confidence);
    output[idx]._padding = 0.0;
}
"#;

/// WGSL shader for element-wise uncertainty division
pub const WGSL_UNCERTAINTY_DIV: &str = r#"
struct Knowledge {
    value: f32,
    uncertainty: f32,
    confidence: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> input_a: array<Knowledge>;
@group(0) @binding(1) var<storage, read> input_b: array<Knowledge>;
@group(0) @binding(2) var<storage, read_write> output: array<Knowledge>;

@compute @workgroup_size(256)
fn uncertainty_div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&output);
    if (idx >= len) {
        return;
    }

    let a = input_a[idx];
    let b = input_b[idx];

    let eps = 1e-10;
    let abs_b = abs(b.value);

    // Handle division by zero
    if (abs_b < eps) {
        output[idx].value = 0.0;
        output[idx].uncertainty = 0.0;
        output[idx].confidence = 0.0; // Zero confidence for invalid operation
        output[idx]._padding = 0.0;
        return;
    }

    // Value division
    let result_value = a.value / b.value;
    output[idx].value = result_value;

    // Relative error propagation: sqrt((da/a)^2 + (db/b)^2) * |result|
    let abs_a = abs(a.value);
    let rel_a = select(0.0, a.uncertainty / abs_a, abs_a > eps);
    let rel_b = b.uncertainty / abs_b;

    let rel_sq = rel_a * rel_a + rel_b * rel_b;
    output[idx].uncertainty = sqrt(max(rel_sq, 0.0)) * abs(result_value);

    // Confidence: minimum of inputs
    output[idx].confidence = min(a.confidence, b.confidence);
    output[idx]._padding = 0.0;
}
"#;

// ============================================================================
// KERNEL REGISTRY
// ============================================================================

/// Identifier for compiled epistemic kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicKernelId {
    /// Element-wise uncertainty addition
    UncertaintyAdd,
    /// Element-wise uncertainty subtraction
    UncertaintySub,
    /// Element-wise uncertainty multiplication
    UncertaintyMul,
    /// Element-wise uncertainty division
    UncertaintyDiv,
    /// Parallel confidence aggregation
    ConfidenceAggregate,
    /// Scalar transformation with uncertainty propagation
    KnowledgeTransform,
}

impl EpistemicKernelId {
    /// Get the WGSL source for this kernel
    pub fn wgsl_source(&self) -> &'static str {
        match self {
            Self::UncertaintyAdd => WGSL_UNCERTAINTY_ADD,
            Self::UncertaintySub => WGSL_UNCERTAINTY_SUB,
            Self::UncertaintyMul => WGSL_UNCERTAINTY_MUL,
            Self::UncertaintyDiv => WGSL_UNCERTAINTY_DIV,
            Self::ConfidenceAggregate => WGSL_CONFIDENCE_AGGREGATE,
            Self::KnowledgeTransform => WGSL_KNOWLEDGE_TRANSFORM,
        }
    }

    /// Get the entry point name for this kernel
    pub fn entry_point(&self) -> &'static str {
        match self {
            Self::UncertaintyAdd => "uncertainty_add",
            Self::UncertaintySub => "uncertainty_sub",
            Self::UncertaintyMul => "uncertainty_mul",
            Self::UncertaintyDiv => "uncertainty_div",
            Self::ConfidenceAggregate => "confidence_aggregate",
            Self::KnowledgeTransform => "knowledge_transform",
        }
    }

    /// Get all kernel IDs
    pub fn all() -> &'static [EpistemicKernelId] {
        &[
            Self::UncertaintyAdd,
            Self::UncertaintySub,
            Self::UncertaintyMul,
            Self::UncertaintyDiv,
            Self::ConfidenceAggregate,
            Self::KnowledgeTransform,
        ]
    }
}

// ============================================================================
// KNOWLEDGE VALUE (GPU-compatible representation)
// ============================================================================

/// GPU-compatible Knowledge value representation
///
/// This struct is designed to match the WGSL Knowledge struct exactly,
/// with 16-byte alignment for efficient GPU memory access.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuKnowledge {
    /// The measured/computed value
    pub value: f32,
    /// Uncertainty (standard deviation)
    pub uncertainty: f32,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Padding for 16-byte alignment
    pub _padding: f32,
}

impl GpuKnowledge {
    /// Create a new GPU-compatible Knowledge value
    pub fn new(value: f32, uncertainty: f32, confidence: f32) -> Self {
        Self {
            value,
            uncertainty,
            confidence,
            _padding: 0.0,
        }
    }

    /// Create from a measurement with given uncertainty
    pub fn from_measurement(value: f32, uncertainty: f32) -> Self {
        Self::new(value, uncertainty, 0.95) // 95% confidence default
    }

    /// Create an axiomatic value (known constant, no uncertainty)
    pub fn axiomatic(value: f32) -> Self {
        Self::new(value, 0.0, 1.0)
    }

    /// Check if this knowledge value is valid
    pub fn is_valid(&self) -> bool {
        self.confidence > 0.0
            && self.uncertainty >= 0.0
            && self.value.is_finite()
            && self.uncertainty.is_finite()
    }

    /// Convert to byte slice for GPU transfer
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// Convert a slice of GpuKnowledge to bytes
pub fn knowledge_slice_to_bytes(knowledge: &[GpuKnowledge]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            knowledge.as_ptr() as *const u8,
            knowledge.len() * std::mem::size_of::<GpuKnowledge>(),
        )
    }
}

/// Convert bytes back to GpuKnowledge slice
///
/// # Safety
/// The byte slice must be properly aligned and have correct length
pub unsafe fn bytes_to_knowledge_slice(bytes: &[u8]) -> &[GpuKnowledge] {
    let count = bytes.len() / std::mem::size_of::<GpuKnowledge>();
    // SAFETY: Caller ensures bytes is properly aligned and has valid length
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const GpuKnowledge, count) }
}

// ============================================================================
// AGGREGATION MODE
// ============================================================================

/// Mode for confidence aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateMode {
    /// Take minimum confidence (most conservative)
    Minimum = 0,
    /// Multiply confidences (independent events)
    Product = 1,
    /// Root mean square (quadrature)
    Quadrature = 2,
}

impl AggregateMode {
    /// Get the mode value for GPU uniform buffer
    pub fn as_u32(&self) -> u32 {
        *self as u32
    }
}

// ============================================================================
// TRANSFORM PARAMETERS
// ============================================================================

/// Parameters for knowledge transformation
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TransformParams {
    /// Scale factor for value
    pub scale: f32,
    /// Offset to add after scaling
    pub offset: f32,
    /// Confidence degradation factor (0.0 to 1.0)
    pub confidence_factor: f32,
    /// Padding for alignment
    pub _padding: f32,
}

impl TransformParams {
    /// Create scale-only transformation
    pub fn scale(scale: f32) -> Self {
        Self {
            scale,
            offset: 0.0,
            confidence_factor: 1.0,
            _padding: 0.0,
        }
    }

    /// Create affine transformation
    pub fn affine(scale: f32, offset: f32) -> Self {
        Self {
            scale,
            offset,
            confidence_factor: 1.0,
            _padding: 0.0,
        }
    }

    /// Create transformation with confidence degradation
    pub fn with_confidence_factor(mut self, factor: f32) -> Self {
        self.confidence_factor = factor.clamp(0.0, 1.0);
        self
    }
}

/// Parameters for confidence aggregation
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AggregateParams {
    /// Aggregation mode (0=min, 1=product, 2=quadrature)
    pub mode: u32,
    /// Number of elements to aggregate
    pub count: u32,
    /// Padding
    pub _pad0: u32,
    pub _pad1: u32,
}

impl AggregateParams {
    /// Create aggregation parameters
    pub fn new(mode: AggregateMode, count: u32) -> Self {
        Self {
            mode: mode.as_u32(),
            count,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

// ============================================================================
// GPU EPISTEMIC OPS (High-level API)
// ============================================================================

/// High-level API for GPU epistemic operations
///
/// This struct manages kernel compilation and execution for epistemic
/// operations on GPU. It integrates with the WGPU runtime.
#[derive(Debug)]
pub struct GpuEpistemicOps {
    /// Compiled kernel IDs (maps EpistemicKernelId to runtime kernel ID)
    compiled_kernels: HashMap<EpistemicKernelId, i64>,
    /// Whether kernels have been compiled
    initialized: bool,
}

impl GpuEpistemicOps {
    /// Create a new GPU epistemic operations manager
    pub fn new() -> Self {
        Self {
            compiled_kernels: HashMap::new(),
            initialized: false,
        }
    }

    /// Check if a kernel is compiled
    pub fn is_kernel_compiled(&self, kernel: EpistemicKernelId) -> bool {
        self.compiled_kernels.contains_key(&kernel)
    }

    /// Get the runtime kernel ID for a compiled kernel
    pub fn get_kernel_id(&self, kernel: EpistemicKernelId) -> Option<i64> {
        self.compiled_kernels.get(&kernel).copied()
    }

    /// Register a compiled kernel
    pub fn register_kernel(&mut self, kernel: EpistemicKernelId, runtime_id: i64) {
        self.compiled_kernels.insert(kernel, runtime_id);
    }

    /// Calculate workgroup count for given element count
    pub fn workgroup_count(element_count: u32) -> u32 {
        (element_count + DEFAULT_WORKGROUP_SIZE - 1) / DEFAULT_WORKGROUP_SIZE
    }

    /// Calculate buffer size for Knowledge array
    pub fn buffer_size(element_count: usize) -> usize {
        element_count * KNOWLEDGE_STRUCT_SIZE
    }

    /// Get all WGSL shader sources
    pub fn all_shader_sources() -> Vec<(&'static str, &'static str)> {
        EpistemicKernelId::all()
            .iter()
            .map(|k| (k.entry_point(), k.wgsl_source()))
            .collect()
    }
}

impl Default for GpuEpistemicOps {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EPISTEMIC IMPACT
// ============================================================================

/// Calculate epistemic impact for GPU operations
pub fn gpu_epistemic_confidence_factor() -> f64 {
    GPU_EPISTEMIC_CONFIDENCE_FACTOR
}

/// Propagate uncertainty for addition/subtraction
pub fn propagate_uncertainty_add(sigma_a: f64, sigma_b: f64) -> f64 {
    (sigma_a * sigma_a + sigma_b * sigma_b).sqrt()
}

/// Propagate uncertainty for multiplication/division (relative error)
pub fn propagate_uncertainty_mul(value_a: f64, sigma_a: f64, value_b: f64, sigma_b: f64) -> f64 {
    let result = value_a * value_b;
    if result.abs() < 1e-10 {
        return 0.0;
    }

    let rel_a = if value_a.abs() > 1e-10 {
        sigma_a / value_a.abs()
    } else {
        0.0
    };
    let rel_b = if value_b.abs() > 1e-10 {
        sigma_b / value_b.abs()
    } else {
        0.0
    };

    (rel_a * rel_a + rel_b * rel_b).sqrt() * result.abs()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_knowledge_size() {
        assert_eq!(std::mem::size_of::<GpuKnowledge>(), KNOWLEDGE_STRUCT_SIZE);
        assert_eq!(std::mem::size_of::<GpuKnowledge>(), 16);
    }

    #[test]
    fn test_gpu_knowledge_creation() {
        let k = GpuKnowledge::new(1.0, 0.1, 0.95);
        assert_eq!(k.value, 1.0);
        assert_eq!(k.uncertainty, 0.1);
        assert_eq!(k.confidence, 0.95);
        assert!(k.is_valid());
    }

    #[test]
    fn test_gpu_knowledge_axiomatic() {
        let k = GpuKnowledge::axiomatic(42.0);
        assert_eq!(k.value, 42.0);
        assert_eq!(k.uncertainty, 0.0);
        assert_eq!(k.confidence, 1.0);
    }

    #[test]
    fn test_kernel_ids() {
        let all = EpistemicKernelId::all();
        assert_eq!(all.len(), 6);

        for kernel in all {
            assert!(!kernel.wgsl_source().is_empty());
            assert!(!kernel.entry_point().is_empty());
        }
    }

    #[test]
    fn test_aggregate_mode() {
        assert_eq!(AggregateMode::Minimum.as_u32(), 0);
        assert_eq!(AggregateMode::Product.as_u32(), 1);
        assert_eq!(AggregateMode::Quadrature.as_u32(), 2);
    }

    #[test]
    fn test_transform_params() {
        let p = TransformParams::scale(2.0);
        assert_eq!(p.scale, 2.0);
        assert_eq!(p.offset, 0.0);
        assert_eq!(p.confidence_factor, 1.0);

        let p = TransformParams::affine(2.0, 1.0).with_confidence_factor(0.9);
        assert_eq!(p.scale, 2.0);
        assert_eq!(p.offset, 1.0);
        assert_eq!(p.confidence_factor, 0.9);
    }

    #[test]
    fn test_workgroup_count() {
        assert_eq!(GpuEpistemicOps::workgroup_count(1), 1);
        assert_eq!(GpuEpistemicOps::workgroup_count(256), 1);
        assert_eq!(GpuEpistemicOps::workgroup_count(257), 2);
        assert_eq!(GpuEpistemicOps::workgroup_count(1000), 4);
    }

    #[test]
    fn test_buffer_size() {
        assert_eq!(GpuEpistemicOps::buffer_size(1), 16);
        assert_eq!(GpuEpistemicOps::buffer_size(100), 1600);
    }

    #[test]
    fn test_uncertainty_propagation_add() {
        let sigma = propagate_uncertainty_add(3.0, 4.0);
        assert!((sigma - 5.0).abs() < 1e-10); // 3-4-5 triangle
    }

    #[test]
    fn test_uncertainty_propagation_mul() {
        // 10 +/- 1 * 5 +/- 0.5 = 50 +/- ?
        // rel_a = 0.1, rel_b = 0.1
        // rel_result = sqrt(0.01 + 0.01) = 0.1414
        // sigma_result = 0.1414 * 50 = 7.07
        let sigma = propagate_uncertainty_mul(10.0, 1.0, 5.0, 0.5);
        assert!((sigma - 7.071).abs() < 0.01);
    }

    #[test]
    fn test_knowledge_slice_bytes() {
        let data = vec![
            GpuKnowledge::new(1.0, 0.1, 0.9),
            GpuKnowledge::new(2.0, 0.2, 0.8),
        ];
        let bytes = knowledge_slice_to_bytes(&data);
        assert_eq!(bytes.len(), 32);

        let recovered = unsafe { bytes_to_knowledge_slice(bytes) };
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].value, 1.0);
        assert_eq!(recovered[1].value, 2.0);
    }

    #[test]
    fn test_wgsl_shader_validity() {
        // Basic syntax checks for WGSL shaders
        for kernel in EpistemicKernelId::all() {
            let source = kernel.wgsl_source();
            assert!(source.contains("struct Knowledge"));
            assert!(source.contains("@compute @workgroup_size(256)"));
            assert!(source.contains(&format!("fn {}", kernel.entry_point())));
        }
    }
}
