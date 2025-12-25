//! Tile Programming Utilities for CUDA 13+
//!
//! This module provides helper functions for tile-based GPU programming:
//! - Validating tile dimensions against WMMA hardware constraints
//! - Selecting optimal WMMA/WGMMA instructions based on tile shape and data type
//! - Computing shared memory requirements with proper alignment
//! - Generating swizzle patterns for bank conflict avoidance
//!
//! # Architecture Support
//!
//! | Architecture | Tensor Core Gen | Supported Shapes |
//! |--------------|-----------------|------------------|
//! | Turing (sm_75) | 2nd gen | 16x16x16 (f16) |
//! | Ampere (sm_80+) | 3rd gen | 16x16x16, 32x8x16, 8x32x16 |
//! | Ada (sm_89) | 4th gen | + FP8 support |
//! | Hopper (sm_90) | 4th gen | + TMA, thread block clusters |
//! | Blackwell (sm_100+) | 5th gen | + FP4, WGMMA, decompression |
//!
//! # Example
//!
//! ```ignore
//! use sounio::codegen::gpu::tile::*;
//!
//! // Validate a 16x16x16 BF16 tile
//! validate_tile_dims(16, 16, 16, &GpuType::BF16)?;
//!
//! // Get shared memory size needed
//! let bytes = shared_memory_bytes(16, 16, &GpuType::BF16);
//!
//! // Select optimal WMMA instruction
//! let op = select_wmma_op(&GpuType::BF16, &GpuType::BF16, &GpuType::F32, 16, 16, 16);
//! ```

use super::ir::*;

/// Validate tile dimensions for WMMA/WGMMA hardware compatibility.
///
/// Returns Ok(()) if the dimensions are valid, or an error message if not.
///
/// # Constraints
///
/// 1. All dimensions must be powers of 2
/// 2. All dimensions must be ≤64
/// 3. The (M, N, K) shape must be supported by the data type's WMMA implementation
///
/// # Supported Shapes by Data Type
///
/// | Data Type | Supported (M, N, K) shapes |
/// |-----------|---------------------------|
/// | F16, BF16 | (16,16,16), (32,8,16), (8,32,16) |
/// | F8E4M3, F8E5M2 | (16,16,32), (32,8,32), (8,32,32) |
/// | F4 | (16,16,64), (32,8,64), (8,32,64) |
/// | F32 (TF32) | (16,16,8), (32,8,8), (8,32,8) |
pub fn validate_tile_dims(m: u32, n: u32, k: u32, dtype: &GpuType) -> Result<(), String> {
    // Must be powers of 2
    if !m.is_power_of_two() || !n.is_power_of_two() || !k.is_power_of_two() {
        return Err(format!(
            "Tile dimensions must be powers of 2, got {}x{}x{}",
            m, n, k
        ));
    }

    // CUDA hardware constraints
    if m > 64 || n > 64 || k > 64 {
        return Err(format!(
            "Tile dimensions must be ≤64, got {}x{}x{}",
            m, n, k
        ));
    }

    // Supported WMMA shapes from CUDA documentation
    let supported: Vec<(u32, u32, u32)> = match dtype {
        GpuType::F16 => vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)],
        GpuType::BF16 => vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)],
        GpuType::F8E4M3 | GpuType::F8E5M2 => vec![(16, 16, 32), (32, 8, 32), (8, 32, 32)],
        GpuType::F4 => vec![(16, 16, 64), (32, 8, 64), (8, 32, 64)],
        GpuType::F32 => vec![(16, 16, 8), (32, 8, 8), (8, 32, 8)], // TF32 mode
        _ => {
            return Err(format!(
                "Unsupported element type for WMMA: {:?}. Use F16, BF16, F8, F4, or F32.",
                dtype
            ));
        }
    };

    if !supported.contains(&(m, n, k)) {
        return Err(format!(
            "Tile shape {}x{}x{} not supported for {:?}. Supported shapes: {:?}",
            m, n, k, dtype, supported
        ));
    }

    Ok(())
}

/// Validate 2D tile dimensions (M x N) for non-MMA operations.
///
/// Used for TileCreate, TileLoad, TileStore where K dimension is not relevant.
pub fn validate_tile_dims_2d(m: u32, n: u32, dtype: &GpuType) -> Result<(), String> {
    if !m.is_power_of_two() || !n.is_power_of_two() {
        return Err(format!(
            "Tile dimensions must be powers of 2, got {}x{}",
            m, n
        ));
    }

    if m > 64 || n > 64 {
        return Err(format!("Tile dimensions must be ≤64, got {}x{}", m, n));
    }

    // Verify element type is supported
    match dtype {
        GpuType::F16
        | GpuType::BF16
        | GpuType::F32
        | GpuType::F8E4M3
        | GpuType::F8E5M2
        | GpuType::F4 => Ok(()),
        _ => Err(format!(
            "Unsupported element type for tile: {:?}. Use F16, BF16, F8, F4, or F32.",
            dtype
        )),
    }
}

/// Select the optimal WMMA/WGMMA GpuOp for the given tile shape and data types.
///
/// This function maps high-level tile MMA requests to the correct low-level
/// Tensor Core operation based on the input/output types and dimensions.
///
/// # Returns
///
/// A `GpuOp` with placeholder ValueIds (must be filled in by the caller).
///
/// # Panics
///
/// Panics if the type combination is not supported for WMMA operations.
pub fn select_wmma_op(
    a_type: &GpuType,
    b_type: &GpuType,
    _c_type: &GpuType,
    m: u32,
    n: u32,
    k: u32,
) -> GpuOp {
    // Placeholder ValueIds - caller must fill these in
    let placeholder = ValueId(0);

    match (a_type, b_type) {
        (GpuType::F4, GpuType::F4) => GpuOp::WgmmaFp4 {
            a: placeholder,
            b: placeholder,
            c: placeholder,
            m,
            n,
            k,
            scale_a: placeholder,
            scale_b: placeholder,
        },
        (GpuType::F8E4M3, GpuType::F8E4M3) => GpuOp::WgmmaFp8 {
            a: placeholder,
            b: placeholder,
            c: placeholder,
            m,
            n,
            k,
            format: Fp8Format::E4M3,
        },
        (GpuType::F8E5M2, GpuType::F8E5M2) => GpuOp::WgmmaFp8 {
            a: placeholder,
            b: placeholder,
            c: placeholder,
            m,
            n,
            k,
            format: Fp8Format::E5M2,
        },
        (GpuType::BF16, GpuType::BF16) => GpuOp::WgmmaBf16 {
            a: placeholder,
            b: placeholder,
            c: placeholder,
            m,
            n,
            k,
        },
        (GpuType::F16, GpuType::F16) => {
            // F16 WMMA - use TileMma as it maps to mma.sync on Ampere
            GpuOp::TileMma {
                c: placeholder,
                a: placeholder,
                b: placeholder,
                tile_m: m,
                tile_n: n,
                tile_k: k,
            }
        }
        (GpuType::F32, GpuType::F32) => {
            // TF32 mode for F32 inputs
            GpuOp::TileMma {
                c: placeholder,
                a: placeholder,
                b: placeholder,
                tile_m: m,
                tile_n: n,
                tile_k: k,
            }
        }
        _ => panic!(
            "Unsupported type combination for WMMA: {:?} x {:?}. \
             Supported: F4xF4, F8xF8, BF16xBF16, F16xF16, F32xF32 (TF32)",
            a_type, b_type
        ),
    }
}

/// Calculate the shared memory bytes required for a tile.
///
/// Returns the number of bytes needed, aligned to 128 bytes for optimal
/// memory access patterns and bank conflict avoidance.
///
/// # Arguments
///
/// * `m` - Tile height
/// * `n` - Tile width
/// * `dtype` - Element data type
///
/// # Returns
///
/// Aligned byte count (multiple of 128 for cache line optimization)
pub fn shared_memory_bytes(m: u32, n: u32, dtype: &GpuType) -> u32 {
    let element_bytes = element_size_bytes(dtype);
    let raw_bytes = m * n * element_bytes;

    // Align to 128 bytes (cache line size on modern GPUs)
    // Also ensures proper alignment for vectorized loads
    align_to(raw_bytes, 128)
}

/// Get the size in bytes of a single element of the given type.
pub fn element_size_bytes(dtype: &GpuType) -> u32 {
    match dtype {
        GpuType::F64 | GpuType::I64 | GpuType::U64 => 8,
        GpuType::F32 | GpuType::I32 | GpuType::U32 => 4,
        GpuType::F16 | GpuType::BF16 | GpuType::I16 | GpuType::U16 => 2,
        GpuType::F8E4M3 | GpuType::F8E5M2 | GpuType::I8 | GpuType::U8 | GpuType::Bool => 1,
        GpuType::F4 => 1, // Packed 2 per byte, but we allocate 1 byte per element for simplicity
        GpuType::Void => 0,
        GpuType::Ptr(_, _) => 8, // 64-bit pointers
        GpuType::Array(inner, _) => element_size_bytes(inner),
        GpuType::Vec2(inner) => element_size_bytes(inner) * 2,
        GpuType::Vec3(inner) => element_size_bytes(inner) * 3,
        GpuType::Vec4(inner) => element_size_bytes(inner) * 4,
        GpuType::Struct { .. } => 0, // Structs need explicit size calculation
    }
}

/// Align a value up to the nearest multiple of alignment.
#[inline]
pub fn align_to(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Generate an optimal swizzle pattern for bank conflict avoidance.
///
/// Shared memory is organized into 32 banks on modern GPUs. When multiple
/// threads access the same bank simultaneously, bank conflicts cause
/// serialization. Swizzling reorders the data layout to minimize conflicts.
///
/// # Arguments
///
/// * `m` - Tile height
/// * `n` - Tile width
/// * `dtype` - Element data type
///
/// # Returns
///
/// A tuple (block_m, block_n) defining the swizzle block granularity.
/// Use with `TileLayout::Swizzled { block_m, block_n }`.
pub fn compute_swizzle_pattern(m: u32, n: u32, dtype: &GpuType) -> (u32, u32) {
    // Shared memory has 32 banks, each 4 bytes wide
    // Bank conflict occurs when two threads access the same bank
    let bank_count = 32u32;
    let bank_width_bytes = 4u32;
    let total_bank_width = bank_count * bank_width_bytes; // 128 bytes

    let element_bytes = element_size_bytes(dtype);

    // Elements per bank cycle (128 bytes)
    let elements_per_bank_cycle = if element_bytes > 0 {
        total_bank_width / element_bytes
    } else {
        64 // Default for unknown types
    };

    // Swizzle block width should span one bank cycle to avoid conflicts
    let block_n = elements_per_bank_cycle.min(n);

    // Swizzle block height is typically warp-aligned (8 or 16)
    // This ensures consecutive warps access different banks
    let block_m = if m >= 16 { 8 } else { m.min(8) };

    (block_m, block_n)
}

/// Check if a given compute capability supports tile operations.
///
/// Tile operations require at least Turing (sm_75) for basic Tensor Cores,
/// with newer features requiring newer architectures.
pub fn supports_tiles(sm_version: (u32, u32)) -> bool {
    let sm = sm_version.0 * 10 + sm_version.1;
    sm >= 75 // Turing and newer
}

/// Check if a given compute capability supports TMA (Tensor Memory Accelerator).
///
/// TMA provides async bulk memory transfers and is available on Hopper (sm_90+).
pub fn supports_tma(sm_version: (u32, u32)) -> bool {
    let sm = sm_version.0 * 10 + sm_version.1;
    sm >= 90 // Hopper and newer
}

/// Check if a given compute capability supports WGMMA (Warpgroup MMA).
///
/// WGMMA is a Blackwell feature providing 5th-gen Tensor Core operations.
pub fn supports_wgmma(sm_version: (u32, u32)) -> bool {
    let sm = sm_version.0 * 10 + sm_version.1;
    sm >= 100 // Blackwell and newer
}

/// Get the recommended tile size for a given data type and architecture.
///
/// Returns (M, N, K) dimensions optimized for the target architecture.
pub fn recommended_tile_size(dtype: &GpuType, sm_version: (u32, u32)) -> (u32, u32, u32) {
    let sm = sm_version.0 * 10 + sm_version.1;

    match dtype {
        GpuType::F4 if sm >= 100 => (16, 16, 64), // Blackwell FP4
        GpuType::F8E4M3 | GpuType::F8E5M2 if sm >= 89 => (16, 16, 32), // Ada/Hopper FP8
        GpuType::BF16 | GpuType::F16 => (16, 16, 16), // Standard WMMA shape
        GpuType::F32 => (16, 16, 8),              // TF32 mode
        _ => (16, 16, 16),                        // Default
    }
}

/// Calculate the number of threads required for a tile operation.
///
/// Different tile sizes require different thread configurations.
pub fn threads_per_tile(m: u32, n: u32) -> u32 {
    // Standard: one thread per element, but capped at warp size multiples
    let elements = m * n;

    // Round up to warp size (32)
    let warps = elements.div_ceil(32);
    warps * 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tile_dims_valid() {
        // Standard 16x16x16 shapes
        assert!(validate_tile_dims(16, 16, 16, &GpuType::F16).is_ok());
        assert!(validate_tile_dims(16, 16, 16, &GpuType::BF16).is_ok());

        // Alternative shapes
        assert!(validate_tile_dims(32, 8, 16, &GpuType::F16).is_ok());
        assert!(validate_tile_dims(8, 32, 16, &GpuType::BF16).is_ok());

        // FP8 shapes
        assert!(validate_tile_dims(16, 16, 32, &GpuType::F8E4M3).is_ok());
        assert!(validate_tile_dims(16, 16, 32, &GpuType::F8E5M2).is_ok());

        // FP4 shapes (Blackwell)
        assert!(validate_tile_dims(16, 16, 64, &GpuType::F4).is_ok());

        // TF32 shapes
        assert!(validate_tile_dims(16, 16, 8, &GpuType::F32).is_ok());
    }

    #[test]
    fn test_validate_tile_dims_invalid_not_power_of_2() {
        let result = validate_tile_dims(15, 16, 16, &GpuType::F16);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("powers of 2"));
    }

    #[test]
    fn test_validate_tile_dims_invalid_too_large() {
        let result = validate_tile_dims(128, 16, 16, &GpuType::F16);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("≤64"));
    }

    #[test]
    fn test_validate_tile_dims_invalid_unsupported_shape() {
        // 16x16x32 is not valid for F16 (only for FP8)
        let result = validate_tile_dims(16, 16, 32, &GpuType::F16);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not supported"));
    }

    #[test]
    fn test_validate_tile_dims_invalid_unsupported_type() {
        let result = validate_tile_dims(16, 16, 16, &GpuType::I32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported element type"));
    }

    #[test]
    fn test_shared_memory_bytes() {
        // 16x16 F16 tile = 16*16*2 = 512 bytes, aligned to 512
        let bytes = shared_memory_bytes(16, 16, &GpuType::F16);
        assert!(bytes >= 512);
        assert_eq!(bytes % 128, 0);

        // 16x16 F32 tile = 16*16*4 = 1024 bytes
        let bytes = shared_memory_bytes(16, 16, &GpuType::F32);
        assert!(bytes >= 1024);
        assert_eq!(bytes % 128, 0);

        // 32x8 BF16 tile = 32*8*2 = 512 bytes
        let bytes = shared_memory_bytes(32, 8, &GpuType::BF16);
        assert!(bytes >= 512);
        assert_eq!(bytes % 128, 0);

        // 16x16 F8 tile = 16*16*1 = 256 bytes, aligned to 256 (multiple of 128)
        let bytes = shared_memory_bytes(16, 16, &GpuType::F8E4M3);
        assert!(bytes >= 256);
        assert_eq!(bytes % 128, 0);
    }

    #[test]
    fn test_element_size_bytes() {
        assert_eq!(element_size_bytes(&GpuType::F64), 8);
        assert_eq!(element_size_bytes(&GpuType::F32), 4);
        assert_eq!(element_size_bytes(&GpuType::F16), 2);
        assert_eq!(element_size_bytes(&GpuType::BF16), 2);
        assert_eq!(element_size_bytes(&GpuType::F8E4M3), 1);
        assert_eq!(element_size_bytes(&GpuType::F8E5M2), 1);
        assert_eq!(element_size_bytes(&GpuType::F4), 1);
    }

    #[test]
    fn test_compute_swizzle_pattern() {
        // F16: 128 bytes / 2 bytes = 64 elements per bank cycle
        let (block_m, block_n) = compute_swizzle_pattern(16, 16, &GpuType::F16);
        assert_eq!(block_m, 8);
        assert_eq!(block_n, 16); // min(64, 16) = 16

        // F32: 128 bytes / 4 bytes = 32 elements per bank cycle
        let (block_m, block_n) = compute_swizzle_pattern(16, 16, &GpuType::F32);
        assert_eq!(block_m, 8);
        assert_eq!(block_n, 16); // min(32, 16) = 16

        // Larger tile
        let (block_m, block_n) = compute_swizzle_pattern(32, 64, &GpuType::F16);
        assert_eq!(block_m, 8);
        assert_eq!(block_n, 64); // min(64, 64) = 64
    }

    #[test]
    fn test_supports_tiles() {
        // Pre-Turing doesn't support tiles
        assert!(!supports_tiles((7, 0))); // Volta

        // Turing and newer support tiles
        assert!(supports_tiles((7, 5))); // Turing
        assert!(supports_tiles((8, 0))); // Ampere
        assert!(supports_tiles((8, 9))); // Ada
        assert!(supports_tiles((9, 0))); // Hopper
        assert!(supports_tiles((10, 0))); // Blackwell
    }

    #[test]
    fn test_supports_tma() {
        // Pre-Hopper doesn't support TMA
        assert!(!supports_tma((8, 9))); // Ada

        // Hopper and newer support TMA
        assert!(supports_tma((9, 0))); // Hopper
        assert!(supports_tma((10, 0))); // Blackwell
    }

    #[test]
    fn test_supports_wgmma() {
        // Pre-Blackwell doesn't support WGMMA
        assert!(!supports_wgmma((9, 0))); // Hopper

        // Blackwell and newer support WGMMA
        assert!(supports_wgmma((10, 0))); // Blackwell
        assert!(supports_wgmma((12, 0))); // Blackwell Ultra
    }

    #[test]
    fn test_recommended_tile_size() {
        // Standard shapes for different types
        assert_eq!(recommended_tile_size(&GpuType::F16, (8, 0)), (16, 16, 16));
        assert_eq!(recommended_tile_size(&GpuType::BF16, (8, 0)), (16, 16, 16));
        assert_eq!(recommended_tile_size(&GpuType::F32, (8, 0)), (16, 16, 8));

        // FP8 on Ada/Hopper
        assert_eq!(
            recommended_tile_size(&GpuType::F8E4M3, (9, 0)),
            (16, 16, 32)
        );

        // FP4 on Blackwell
        assert_eq!(recommended_tile_size(&GpuType::F4, (10, 0)), (16, 16, 64));
    }

    #[test]
    fn test_threads_per_tile() {
        // 16x16 = 256 elements = 8 warps
        assert_eq!(threads_per_tile(16, 16), 256);

        // 8x8 = 64 elements = 2 warps
        assert_eq!(threads_per_tile(8, 8), 64);

        // 32x8 = 256 elements = 8 warps
        assert_eq!(threads_per_tile(32, 8), 256);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(100, 128), 128);
        assert_eq!(align_to(128, 128), 128);
        assert_eq!(align_to(129, 128), 256);
        assert_eq!(align_to(0, 128), 0);
        assert_eq!(align_to(1, 128), 128);
    }

    #[test]
    fn test_validate_tile_dims_2d() {
        assert!(validate_tile_dims_2d(16, 16, &GpuType::F16).is_ok());
        assert!(validate_tile_dims_2d(32, 32, &GpuType::BF16).is_ok());
        assert!(validate_tile_dims_2d(8, 64, &GpuType::F32).is_ok());

        // Invalid: not power of 2
        assert!(validate_tile_dims_2d(15, 16, &GpuType::F16).is_err());

        // Invalid: too large
        assert!(validate_tile_dims_2d(128, 16, &GpuType::F16).is_err());

        // Invalid: unsupported type
        assert!(validate_tile_dims_2d(16, 16, &GpuType::I32).is_err());
    }
}
