//! GPU Intrinsic Functions
//!
//! Built-in functions available in GPU kernels

use std::collections::HashMap;

/// GPU intrinsic function
#[derive(Debug, Clone)]
pub struct GpuIntrinsic {
    pub name: &'static str,
    pub short_name: &'static str,
    pub param_count: usize,
    pub return_type: IntrinsicType,
    pub description: &'static str,
    pub category: IntrinsicCategory,
}

/// Intrinsic type (simplified)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicType {
    Void,
    Bool,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Ptr,
    // Extended types for ML quantization
    U8,
    U16,
}

/// Intrinsic category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicCategory {
    /// Thread identification
    ThreadId,
    /// Block identification
    BlockId,
    /// Dimension queries
    Dimension,
    /// Synchronization
    Sync,
    /// Warp operations
    Warp,
    /// Atomic operations
    Atomic,
    /// Fast math
    FastMath,
    /// Memory operations
    Memory,
    /// Kernel launch
    Launch,
    /// Debug/profiling
    Debug,
    /// Quantization/ML types (FP8/BF16/F4)
    Quantization,
}

impl IntrinsicCategory {
    pub fn all() -> &'static [IntrinsicCategory] {
        &[
            IntrinsicCategory::ThreadId,
            IntrinsicCategory::BlockId,
            IntrinsicCategory::Dimension,
            IntrinsicCategory::Sync,
            IntrinsicCategory::Warp,
            IntrinsicCategory::Atomic,
            IntrinsicCategory::FastMath,
            IntrinsicCategory::Memory,
            IntrinsicCategory::Launch,
            IntrinsicCategory::Debug,
            IntrinsicCategory::Quantization,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            IntrinsicCategory::ThreadId => "Thread Identification",
            IntrinsicCategory::BlockId => "Block Identification",
            IntrinsicCategory::Dimension => "Dimension Queries",
            IntrinsicCategory::Sync => "Synchronization",
            IntrinsicCategory::Warp => "Warp Operations",
            IntrinsicCategory::Atomic => "Atomic Operations",
            IntrinsicCategory::FastMath => "Fast Math",
            IntrinsicCategory::Memory => "Memory Operations",
            IntrinsicCategory::Launch => "Kernel Launch",
            IntrinsicCategory::Debug => "Debug/Profiling",
            IntrinsicCategory::Quantization => "Quantization/ML Types",
        }
    }
}

/// All GPU intrinsics
pub fn all_intrinsics() -> Vec<GpuIntrinsic> {
    vec![
        // === Thread identification ===
        GpuIntrinsic {
            name: "gpu.thread_id.x",
            short_name: "thread_id_x",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Thread index within block (X dimension)",
            category: IntrinsicCategory::ThreadId,
        },
        GpuIntrinsic {
            name: "gpu.thread_id.y",
            short_name: "thread_id_y",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Thread index within block (Y dimension)",
            category: IntrinsicCategory::ThreadId,
        },
        GpuIntrinsic {
            name: "gpu.thread_id.z",
            short_name: "thread_id_z",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Thread index within block (Z dimension)",
            category: IntrinsicCategory::ThreadId,
        },
        // === Block identification ===
        GpuIntrinsic {
            name: "gpu.block_id.x",
            short_name: "block_id_x",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Block index within grid (X dimension)",
            category: IntrinsicCategory::BlockId,
        },
        GpuIntrinsic {
            name: "gpu.block_id.y",
            short_name: "block_id_y",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Block index within grid (Y dimension)",
            category: IntrinsicCategory::BlockId,
        },
        GpuIntrinsic {
            name: "gpu.block_id.z",
            short_name: "block_id_z",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Block index within grid (Z dimension)",
            category: IntrinsicCategory::BlockId,
        },
        // === Dimension queries ===
        GpuIntrinsic {
            name: "gpu.block_dim.x",
            short_name: "block_dim_x",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Block size (X dimension)",
            category: IntrinsicCategory::Dimension,
        },
        GpuIntrinsic {
            name: "gpu.block_dim.y",
            short_name: "block_dim_y",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Block size (Y dimension)",
            category: IntrinsicCategory::Dimension,
        },
        GpuIntrinsic {
            name: "gpu.block_dim.z",
            short_name: "block_dim_z",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Block size (Z dimension)",
            category: IntrinsicCategory::Dimension,
        },
        GpuIntrinsic {
            name: "gpu.grid_dim.x",
            short_name: "grid_dim_x",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Grid size (X dimension)",
            category: IntrinsicCategory::Dimension,
        },
        GpuIntrinsic {
            name: "gpu.grid_dim.y",
            short_name: "grid_dim_y",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Grid size (Y dimension)",
            category: IntrinsicCategory::Dimension,
        },
        GpuIntrinsic {
            name: "gpu.grid_dim.z",
            short_name: "grid_dim_z",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Grid size (Z dimension)",
            category: IntrinsicCategory::Dimension,
        },
        GpuIntrinsic {
            name: "gpu.warp_id",
            short_name: "warp_id",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Warp index within block",
            category: IntrinsicCategory::Dimension,
        },
        GpuIntrinsic {
            name: "gpu.lane_id",
            short_name: "lane_id",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Lane index within warp (0-31)",
            category: IntrinsicCategory::Dimension,
        },
        GpuIntrinsic {
            name: "gpu.warp_size",
            short_name: "warp_size",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Number of threads per warp (typically 32)",
            category: IntrinsicCategory::Dimension,
        },
        // === Synchronization ===
        GpuIntrinsic {
            name: "gpu.sync_threads",
            short_name: "sync_threads",
            param_count: 0,
            return_type: IntrinsicType::Void,
            description: "Block-level barrier synchronization",
            category: IntrinsicCategory::Sync,
        },
        GpuIntrinsic {
            name: "gpu.sync_warp",
            short_name: "sync_warp",
            param_count: 0,
            return_type: IntrinsicType::Void,
            description: "Warp-level synchronization",
            category: IntrinsicCategory::Sync,
        },
        GpuIntrinsic {
            name: "gpu.memory_fence",
            short_name: "memory_fence",
            param_count: 0,
            return_type: IntrinsicType::Void,
            description: "Memory fence (ensures visibility)",
            category: IntrinsicCategory::Sync,
        },
        GpuIntrinsic {
            name: "gpu.memory_fence_block",
            short_name: "memory_fence_block",
            param_count: 0,
            return_type: IntrinsicType::Void,
            description: "Block-level memory fence",
            category: IntrinsicCategory::Sync,
        },
        GpuIntrinsic {
            name: "gpu.memory_fence_device",
            short_name: "memory_fence_device",
            param_count: 0,
            return_type: IntrinsicType::Void,
            description: "Device-level memory fence",
            category: IntrinsicCategory::Sync,
        },
        // === Warp operations ===
        GpuIntrinsic {
            name: "gpu.warp_shuffle",
            short_name: "warp_shuffle",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Exchange values between warp lanes",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_shuffle_down",
            short_name: "warp_shuffle_down",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Get value from lane + delta",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_shuffle_up",
            short_name: "warp_shuffle_up",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Get value from lane - delta",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_shuffle_xor",
            short_name: "warp_shuffle_xor",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Get value from lane XOR mask",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_vote_all",
            short_name: "warp_vote_all",
            param_count: 1,
            return_type: IntrinsicType::Bool,
            description: "True if all warp lanes have true predicate",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_vote_any",
            short_name: "warp_vote_any",
            param_count: 1,
            return_type: IntrinsicType::Bool,
            description: "True if any warp lane has true predicate",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_ballot",
            short_name: "warp_ballot",
            param_count: 1,
            return_type: IntrinsicType::U32,
            description: "Bitmask of lanes with true predicate",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_reduce_add",
            short_name: "warp_reduce_add",
            param_count: 1,
            return_type: IntrinsicType::I32,
            description: "Sum values across warp lanes",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_reduce_min",
            short_name: "warp_reduce_min",
            param_count: 1,
            return_type: IntrinsicType::I32,
            description: "Minimum value across warp lanes",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_reduce_max",
            short_name: "warp_reduce_max",
            param_count: 1,
            return_type: IntrinsicType::I32,
            description: "Maximum value across warp lanes",
            category: IntrinsicCategory::Warp,
        },
        GpuIntrinsic {
            name: "gpu.warp_match",
            short_name: "warp_match",
            param_count: 1,
            return_type: IntrinsicType::U32,
            description: "Find lanes with matching values",
            category: IntrinsicCategory::Warp,
        },
        // === Atomic operations ===
        GpuIntrinsic {
            name: "gpu.atomic_add",
            short_name: "atomic_add",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Atomic add, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        GpuIntrinsic {
            name: "gpu.atomic_sub",
            short_name: "atomic_sub",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Atomic subtract, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        GpuIntrinsic {
            name: "gpu.atomic_min",
            short_name: "atomic_min",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Atomic minimum, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        GpuIntrinsic {
            name: "gpu.atomic_max",
            short_name: "atomic_max",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Atomic maximum, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        GpuIntrinsic {
            name: "gpu.atomic_and",
            short_name: "atomic_and",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Atomic AND, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        GpuIntrinsic {
            name: "gpu.atomic_or",
            short_name: "atomic_or",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Atomic OR, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        GpuIntrinsic {
            name: "gpu.atomic_xor",
            short_name: "atomic_xor",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Atomic XOR, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        GpuIntrinsic {
            name: "gpu.atomic_exch",
            short_name: "atomic_exch",
            param_count: 2,
            return_type: IntrinsicType::I32,
            description: "Atomic exchange, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        GpuIntrinsic {
            name: "gpu.atomic_cas",
            short_name: "atomic_cas",
            param_count: 3,
            return_type: IntrinsicType::I32,
            description: "Atomic compare-and-swap, returns old value",
            category: IntrinsicCategory::Atomic,
        },
        // === Fast math ===
        GpuIntrinsic {
            name: "gpu.fast_sin",
            short_name: "fast_sin",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Fast sine (reduced precision)",
            category: IntrinsicCategory::FastMath,
        },
        GpuIntrinsic {
            name: "gpu.fast_cos",
            short_name: "fast_cos",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Fast cosine (reduced precision)",
            category: IntrinsicCategory::FastMath,
        },
        GpuIntrinsic {
            name: "gpu.fast_exp",
            short_name: "fast_exp",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Fast exponential (reduced precision)",
            category: IntrinsicCategory::FastMath,
        },
        GpuIntrinsic {
            name: "gpu.fast_log",
            short_name: "fast_log",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Fast logarithm (reduced precision)",
            category: IntrinsicCategory::FastMath,
        },
        GpuIntrinsic {
            name: "gpu.fast_sqrt",
            short_name: "fast_sqrt",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Fast square root (reduced precision)",
            category: IntrinsicCategory::FastMath,
        },
        GpuIntrinsic {
            name: "gpu.fast_rsqrt",
            short_name: "fast_rsqrt",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Fast reciprocal square root (1/sqrt(x))",
            category: IntrinsicCategory::FastMath,
        },
        GpuIntrinsic {
            name: "gpu.fma",
            short_name: "fma",
            param_count: 3,
            return_type: IntrinsicType::F32,
            description: "Fused multiply-add (a * b + c)",
            category: IntrinsicCategory::FastMath,
        },
        GpuIntrinsic {
            name: "gpu.fast_pow",
            short_name: "fast_pow",
            param_count: 2,
            return_type: IntrinsicType::F32,
            description: "Fast power (reduced precision)",
            category: IntrinsicCategory::FastMath,
        },
        // === Memory operations ===
        GpuIntrinsic {
            name: "gpu.alloc",
            short_name: "alloc",
            param_count: 1,
            return_type: IntrinsicType::Ptr,
            description: "Allocate device memory",
            category: IntrinsicCategory::Memory,
        },
        GpuIntrinsic {
            name: "gpu.free",
            short_name: "free",
            param_count: 1,
            return_type: IntrinsicType::Void,
            description: "Free device memory",
            category: IntrinsicCategory::Memory,
        },
        GpuIntrinsic {
            name: "gpu.copy_to_device",
            short_name: "copy_to_device",
            param_count: 2,
            return_type: IntrinsicType::Void,
            description: "Copy data from host to device",
            category: IntrinsicCategory::Memory,
        },
        GpuIntrinsic {
            name: "gpu.copy_to_host",
            short_name: "copy_to_host",
            param_count: 2,
            return_type: IntrinsicType::Void,
            description: "Copy data from device to host",
            category: IntrinsicCategory::Memory,
        },
        // === Kernel launch ===
        GpuIntrinsic {
            name: "gpu.launch",
            short_name: "launch",
            param_count: 3,
            return_type: IntrinsicType::Void,
            description: "Launch a GPU kernel",
            category: IntrinsicCategory::Launch,
        },
        GpuIntrinsic {
            name: "gpu.synchronize",
            short_name: "synchronize",
            param_count: 0,
            return_type: IntrinsicType::Void,
            description: "Wait for all GPU operations to complete",
            category: IntrinsicCategory::Launch,
        },
        // === Debug/Profiling ===
        GpuIntrinsic {
            name: "gpu.printf",
            short_name: "printf",
            param_count: 255, // Variadic
            return_type: IntrinsicType::I32,
            description: "Print formatted output from GPU (CUDA printf/vprintf)",
            category: IntrinsicCategory::Debug,
        },
        GpuIntrinsic {
            name: "gpu.assert",
            short_name: "assert",
            param_count: 1,
            return_type: IntrinsicType::Void,
            description: "Assert condition on GPU (triggers trap if false)",
            category: IntrinsicCategory::Debug,
        },
        GpuIntrinsic {
            name: "gpu.trap",
            short_name: "trap",
            param_count: 0,
            return_type: IntrinsicType::Void,
            description: "Trigger GPU trap/breakpoint",
            category: IntrinsicCategory::Debug,
        },
        GpuIntrinsic {
            name: "gpu.brkpt",
            short_name: "brkpt",
            param_count: 0,
            return_type: IntrinsicType::Void,
            description: "Software breakpoint (for debugger)",
            category: IntrinsicCategory::Debug,
        },
        GpuIntrinsic {
            name: "gpu.clock",
            short_name: "clock",
            param_count: 0,
            return_type: IntrinsicType::U64,
            description: "Read per-SM clock counter (for profiling)",
            category: IntrinsicCategory::Debug,
        },
        GpuIntrinsic {
            name: "gpu.clock_hi",
            short_name: "clock_hi",
            param_count: 0,
            return_type: IntrinsicType::U32,
            description: "Read high 32 bits of clock counter",
            category: IntrinsicCategory::Debug,
        },
        GpuIntrinsic {
            name: "gpu.globaltimer",
            short_name: "globaltimer",
            param_count: 0,
            return_type: IntrinsicType::U64,
            description: "Read global timer (nanoseconds since reset)",
            category: IntrinsicCategory::Debug,
        },
        GpuIntrinsic {
            name: "gpu.pm_event",
            short_name: "pm_event",
            param_count: 1,
            return_type: IntrinsicType::Void,
            description: "Record performance monitoring event",
            category: IntrinsicCategory::Debug,
        },
        // === Quantization/ML Type Intrinsics ===
        GpuIntrinsic {
            name: "gpu.f32_to_bf16",
            short_name: "f32_to_bf16",
            param_count: 1,
            return_type: IntrinsicType::U16, // BF16 represented as u16
            description: "Convert F32 to BF16 (BFloat16)",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.bf16_to_f32",
            short_name: "bf16_to_f32",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Convert BF16 to F32",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.f32_to_f8e4m3",
            short_name: "f32_to_f8e4m3",
            param_count: 1,
            return_type: IntrinsicType::U8, // FP8 E4M3 as u8
            description: "Convert F32 to FP8 E4M3 (higher precision, ±448 range)",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.f8e4m3_to_f32",
            short_name: "f8e4m3_to_f32",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Convert FP8 E4M3 to F32",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.f32_to_f8e5m2",
            short_name: "f32_to_f8e5m2",
            param_count: 1,
            return_type: IntrinsicType::U8, // FP8 E5M2 as u8
            description: "Convert F32 to FP8 E5M2 (larger range, ±57344)",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.f8e5m2_to_f32",
            short_name: "f8e5m2_to_f32",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Convert FP8 E5M2 to F32",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.f32_to_f4",
            short_name: "f32_to_f4",
            param_count: 1,
            return_type: IntrinsicType::U8, // F4 as nibble in u8
            description: "Convert F32 to 4-bit float (extreme quantization)",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.f4_to_f32",
            short_name: "f4_to_f32",
            param_count: 1,
            return_type: IntrinsicType::F32,
            description: "Convert 4-bit float to F32",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.pack_f8x2",
            short_name: "pack_f8x2",
            param_count: 2,
            return_type: IntrinsicType::U16, // Two FP8 values packed
            description: "Pack two FP8 values into u16",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.unpack_f8x2",
            short_name: "unpack_f8x2",
            param_count: 1,
            return_type: IntrinsicType::U16, // Returns packed, caller extracts
            description: "Unpack u16 into two FP8 values",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.pack_f4x2",
            short_name: "pack_f4x2",
            param_count: 2,
            return_type: IntrinsicType::U8, // Two F4 values in one byte
            description: "Pack two F4 values into u8",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.unpack_f4x2",
            short_name: "unpack_f4x2",
            param_count: 1,
            return_type: IntrinsicType::U8, // Returns packed, caller extracts
            description: "Unpack u8 into two F4 values",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.quantize_block",
            short_name: "quantize_block",
            param_count: 3,                  // ptr to f32[], size, ptr to f8[]
            return_type: IntrinsicType::F32, // Returns scale factor
            description: "Quantize F32 block to F8 with per-block scaling",
            category: IntrinsicCategory::Quantization,
        },
        GpuIntrinsic {
            name: "gpu.dequantize_block",
            short_name: "dequantize_block",
            param_count: 3, // ptr to f8[], size, scale factor
            return_type: IntrinsicType::Void,
            description: "Dequantize F8 block to F32 with scale",
            category: IntrinsicCategory::Quantization,
        },
    ]
}

/// Check if a name is a GPU intrinsic
pub fn is_gpu_intrinsic(name: &str) -> bool {
    name.starts_with("gpu.")
}

/// Get intrinsic by name
pub fn get_intrinsic(name: &str) -> Option<GpuIntrinsic> {
    all_intrinsics().into_iter().find(|i| i.name == name)
}

/// Get intrinsic by short name
pub fn get_intrinsic_by_short_name(short_name: &str) -> Option<GpuIntrinsic> {
    all_intrinsics()
        .into_iter()
        .find(|i| i.short_name == short_name)
}

/// Get all intrinsics by category
pub fn get_intrinsics_by_category(category: IntrinsicCategory) -> Vec<GpuIntrinsic> {
    all_intrinsics()
        .into_iter()
        .filter(|i| i.category == category)
        .collect()
}

/// Get intrinsic count by category
pub fn intrinsic_count_by_category() -> HashMap<IntrinsicCategory, usize> {
    let mut counts = HashMap::new();
    for cat in IntrinsicCategory::all() {
        counts.insert(*cat, get_intrinsics_by_category(*cat).len());
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gpu_intrinsic() {
        assert!(is_gpu_intrinsic("gpu.thread_id.x"));
        assert!(is_gpu_intrinsic("gpu.sync_threads"));
        assert!(!is_gpu_intrinsic("thread_id"));
        assert!(!is_gpu_intrinsic("cuda.sync"));
    }

    #[test]
    fn test_get_intrinsic() {
        let intrinsic = get_intrinsic("gpu.thread_id.x").unwrap();
        assert_eq!(intrinsic.name, "gpu.thread_id.x");
        assert_eq!(intrinsic.return_type, IntrinsicType::U32);
        assert_eq!(intrinsic.param_count, 0);

        assert!(get_intrinsic("gpu.nonexistent").is_none());
    }

    #[test]
    fn test_get_by_short_name() {
        let intrinsic = get_intrinsic_by_short_name("sync_threads").unwrap();
        assert_eq!(intrinsic.name, "gpu.sync_threads");
        assert_eq!(intrinsic.return_type, IntrinsicType::Void);
    }

    #[test]
    fn test_get_by_category() {
        let thread_intrinsics = get_intrinsics_by_category(IntrinsicCategory::ThreadId);
        assert_eq!(thread_intrinsics.len(), 3);

        let sync_intrinsics = get_intrinsics_by_category(IntrinsicCategory::Sync);
        assert!(sync_intrinsics.len() >= 3);
    }

    #[test]
    fn test_all_intrinsics_count() {
        let intrinsics = all_intrinsics();
        assert!(intrinsics.len() >= 50);
    }

    #[test]
    fn test_intrinsic_categories() {
        let counts = intrinsic_count_by_category();
        assert!(counts[&IntrinsicCategory::ThreadId] >= 3);
        assert!(counts[&IntrinsicCategory::BlockId] >= 3);
        assert!(counts[&IntrinsicCategory::Warp] >= 5);
        assert!(counts[&IntrinsicCategory::Atomic] >= 5);
    }

    #[test]
    fn test_category_names() {
        assert_eq!(IntrinsicCategory::ThreadId.name(), "Thread Identification");
        assert_eq!(IntrinsicCategory::Sync.name(), "Synchronization");
        assert_eq!(IntrinsicCategory::Warp.name(), "Warp Operations");
    }
}
