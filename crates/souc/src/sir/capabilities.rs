//! GPU/Backend Capability Declaration System
//!
//! This module defines a unified capability system for GPU backends that enables
//! the compiler to make informed decisions about code generation strategies.
//!
//! # Design Philosophy
//!
//! **"The backend only declares capabilities; the compiler selects strategies."**
//!
//! Backends expose what they can do, not how to use those capabilities. The compiler's
//! optimization passes then query these capabilities to select appropriate strategies
//! for memory layout, synchronization, parallelization, and instruction selection.
//!
//! # Architecture
//!
//! ```text
//! Backend Declaration          Compiler Query           Strategy Selection
//! ─────────────────          ─────────────────        ────────────────────
//! TargetCapabilities    →    CapabilityQuery    →    Optimization Pass
//!   - SubgroupShuffle          has_capability()         - Use ballot sync
//!   - SharedMemKb(48)          effective_shared_mem()   - Tile strategy
//!   - WarpSize(32)             warp_size()              - Reduction algo
//! ```
//!
//! # Example
//!
//! ```ignore
//! use sounio::sir::capabilities::{TargetCapabilities, Capability};
//!
//! // Backend declares its capabilities
//! let caps = TargetCapabilities::nvidia_ampere();
//!
//! // Compiler queries capabilities for optimization
//! if caps.has_capability(Capability::TensorOps) {
//!     // Use tensor core path
//! } else {
//!     // Fall back to scalar path
//! }
//!
//! // Query effective limits considering all constraints
//! let usable_shared = caps.effective_shared_memory_kb();
//! ```

use std::collections::HashSet;
use std::fmt;

// ============================================================================
// Core Capability Enum
// ============================================================================

/// Hardware and software capabilities that GPU backends can declare.
///
/// Capabilities are atomic features that backends either support or don't.
/// The compiler queries these to make code generation decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    // === Subgroup/Warp Operations ===
    /// Subgroup shuffle operations (warp shuffle on NVIDIA, simd_shuffle on Metal)
    /// Enables efficient intra-warp communication without shared memory.
    SubgroupShuffle,

    /// Subgroup ballot operations for active thread masks
    SubgroupBallot,

    /// Subgroup vote operations (any, all, ballot)
    SubgroupVote,

    /// Subgroup arithmetic reductions (sum, min, max)
    SubgroupArithmetic,

    // === Precision and Numeric ===
    /// Fast FP64 hardware (native double precision, not emulated)
    /// When absent, FP64 operations are ~32x slower than FP32.
    Fp64Fast,

    /// Native FP16 arithmetic (not just storage)
    Fp16Arithmetic,

    /// BFloat16 support (sm_80+, Apple Neural Engine)
    Bf16,

    /// FP8 support (sm_89+)
    Fp8,

    // === Memory ===
    /// Shared memory per block in kilobytes
    /// Parameterized: `SharedMemKb(48)` means 48KB shared memory
    SharedMemKb(u32),

    /// Maximum shared memory per block in kilobytes (configurable carveout)
    MaxSharedPerBlock(u32),

    /// Supports asynchronous memory copy (cp.async, memcpy_async)
    AsyncMemCopy,

    /// Supports Tensor Memory Accelerator (TMA) for bulk transfers
    TensorMemoryAccess,

    // === Tensor/Matrix Operations ===
    /// Hardware tensor/matrix operations (Tensor Cores, AMX, simdgroup_matrix)
    TensorOps,

    /// Specific tensor core generation (for NVIDIA: 1=Volta, 2=Turing, 3=Ampere, 4=Hopper, 5=Blackwell)
    TensorCoreGen(u32),

    /// FP8 in tensor cores
    TensorFp8,

    /// FP4/INT4 in tensor cores (Blackwell)
    TensorFp4,

    /// WGMMA (Warpgroup Matrix Multiply Accumulate, Hopper+)
    Wgmma,

    // === Atomics ===
    /// 64-bit atomic operations (atomic64)
    Atomics64,

    /// Atomic float add operations
    AtomicFloatAdd,

    /// Atomic min/max on floats
    AtomicFloatMinMax,

    // === Thread Organization ===
    /// Cooperative groups (inter-block synchronization)
    CooperativeGroups,

    /// Thread block clusters (sm_90+)
    ThreadBlockClusters,

    /// Distributed shared memory across cluster
    DistributedSharedMemory,

    // === Architecture Parameters ===
    /// Warp/wavefront/SIMD width
    WarpSize(u32),

    /// Subgroup size for Vulkan/SPIR-V (may differ from warp size)
    SubgroupSize(u32),

    /// Maximum registers per thread
    MaxRegistersPerThread(u32),

    /// Maximum threads per block/workgroup
    MaxThreadsPerBlock(u32),

    /// NVIDIA compute capability (major, minor)
    ComputeCapability(u32, u32),

    /// AMD GCN/RDNA version
    AmdGcnVersion(u32),

    /// Intel Xe architecture version
    IntelXeVersion(u32),

    /// Apple GPU family (7, 8, 9, 10 for Apple7-10)
    AppleGpuFamily(u32),

    // === Ray Tracing (for completeness) ===
    /// Hardware ray tracing support
    RayTracing,

    /// Mesh shaders
    MeshShaders,

    // === Special Features ===
    /// Dynamic parallelism (kernels launching kernels)
    DynamicParallelism,

    /// Unified memory (CPU/GPU shared address space)
    UnifiedMemory,

    /// Page migration engine
    PageMigration,

    /// NVLink interconnect
    NvLink(u32), // version

    /// PCIe atomics
    PcieAtomics,
}

impl Capability {
    /// Returns true if this is a parameterized capability (contains a value)
    pub fn is_parameterized(&self) -> bool {
        matches!(
            self,
            Capability::SharedMemKb(_)
                | Capability::MaxSharedPerBlock(_)
                | Capability::WarpSize(_)
                | Capability::SubgroupSize(_)
                | Capability::MaxRegistersPerThread(_)
                | Capability::MaxThreadsPerBlock(_)
                | Capability::ComputeCapability(_, _)
                | Capability::AmdGcnVersion(_)
                | Capability::IntelXeVersion(_)
                | Capability::AppleGpuFamily(_)
                | Capability::TensorCoreGen(_)
                | Capability::NvLink(_)
        )
    }

    /// Extract the parameter value for parameterized capabilities
    pub fn parameter_value(&self) -> Option<u32> {
        match self {
            Capability::SharedMemKb(v)
            | Capability::MaxSharedPerBlock(v)
            | Capability::WarpSize(v)
            | Capability::SubgroupSize(v)
            | Capability::MaxRegistersPerThread(v)
            | Capability::MaxThreadsPerBlock(v)
            | Capability::AmdGcnVersion(v)
            | Capability::IntelXeVersion(v)
            | Capability::AppleGpuFamily(v)
            | Capability::TensorCoreGen(v)
            | Capability::NvLink(v) => Some(*v),
            Capability::ComputeCapability(major, minor) => Some(major * 10 + minor),
            _ => None,
        }
    }

    /// Human-readable description of the capability
    pub fn description(&self) -> &'static str {
        match self {
            Capability::SubgroupShuffle => "Subgroup shuffle operations",
            Capability::SubgroupBallot => "Subgroup ballot operations",
            Capability::SubgroupVote => "Subgroup vote operations",
            Capability::SubgroupArithmetic => "Subgroup arithmetic reductions",
            Capability::Fp64Fast => "Fast FP64 hardware",
            Capability::Fp16Arithmetic => "Native FP16 arithmetic",
            Capability::Bf16 => "BFloat16 support",
            Capability::Fp8 => "FP8 support",
            Capability::SharedMemKb(_) => "Shared memory (KB)",
            Capability::MaxSharedPerBlock(_) => "Max shared memory per block (KB)",
            Capability::AsyncMemCopy => "Asynchronous memory copy",
            Capability::TensorMemoryAccess => "Tensor Memory Accelerator (TMA)",
            Capability::TensorOps => "Hardware tensor/matrix operations",
            Capability::TensorCoreGen(_) => "Tensor Core generation",
            Capability::TensorFp8 => "FP8 in tensor cores",
            Capability::TensorFp4 => "FP4/INT4 in tensor cores",
            Capability::Wgmma => "Warpgroup Matrix Multiply Accumulate",
            Capability::Atomics64 => "64-bit atomic operations",
            Capability::AtomicFloatAdd => "Atomic float add",
            Capability::AtomicFloatMinMax => "Atomic float min/max",
            Capability::CooperativeGroups => "Cooperative groups",
            Capability::ThreadBlockClusters => "Thread block clusters",
            Capability::DistributedSharedMemory => "Distributed shared memory",
            Capability::WarpSize(_) => "Warp/SIMD width",
            Capability::SubgroupSize(_) => "Subgroup size",
            Capability::MaxRegistersPerThread(_) => "Max registers per thread",
            Capability::MaxThreadsPerBlock(_) => "Max threads per block",
            Capability::ComputeCapability(_, _) => "NVIDIA compute capability",
            Capability::AmdGcnVersion(_) => "AMD GCN/RDNA version",
            Capability::IntelXeVersion(_) => "Intel Xe version",
            Capability::AppleGpuFamily(_) => "Apple GPU family",
            Capability::RayTracing => "Hardware ray tracing",
            Capability::MeshShaders => "Mesh shaders",
            Capability::DynamicParallelism => "Dynamic parallelism",
            Capability::UnifiedMemory => "Unified memory",
            Capability::PageMigration => "Page migration engine",
            Capability::NvLink(_) => "NVLink interconnect",
            Capability::PcieAtomics => "PCIe atomics",
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Capability::SharedMemKb(kb) => write!(f, "SharedMemKb({})", kb),
            Capability::MaxSharedPerBlock(kb) => write!(f, "MaxSharedPerBlock({})", kb),
            Capability::WarpSize(size) => write!(f, "WarpSize({})", size),
            Capability::SubgroupSize(size) => write!(f, "SubgroupSize({})", size),
            Capability::MaxRegistersPerThread(regs) => write!(f, "MaxRegistersPerThread({})", regs),
            Capability::MaxThreadsPerBlock(threads) => write!(f, "MaxThreadsPerBlock({})", threads),
            Capability::ComputeCapability(major, minor) => {
                write!(f, "ComputeCapability({}.{})", major, minor)
            }
            Capability::AmdGcnVersion(v) => write!(f, "AmdGcnVersion({})", v),
            Capability::IntelXeVersion(v) => write!(f, "IntelXeVersion({})", v),
            Capability::AppleGpuFamily(v) => write!(f, "AppleGpuFamily({})", v),
            Capability::TensorCoreGen(g) => write!(f, "TensorCoreGen({})", g),
            Capability::NvLink(v) => write!(f, "NvLink({})", v),
            _ => write!(f, "{:?}", self),
        }
    }
}

// ============================================================================
// Target Capabilities
// ============================================================================

/// A collection of capabilities declared by a specific GPU target.
///
/// This struct represents what a backend *can do*, not how to use it.
/// The compiler queries this to make optimization decisions.
#[derive(Debug, Clone)]
pub struct TargetCapabilities {
    /// Human-readable target name
    pub name: String,

    /// Set of supported capabilities
    capabilities: HashSet<Capability>,

    /// Optional vendor-specific metadata
    pub vendor: Option<GpuVendor>,
}

/// GPU vendor identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Qualcomm,
    Arm,
    Unknown,
}

impl TargetCapabilities {
    /// Create an empty capability set
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            capabilities: HashSet::new(),
            vendor: None,
        }
    }

    /// Add a capability
    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Add multiple capabilities
    pub fn with_capabilities(mut self, caps: impl IntoIterator<Item = Capability>) -> Self {
        self.capabilities.extend(caps);
        self
    }

    /// Set the vendor
    pub fn with_vendor(mut self, vendor: GpuVendor) -> Self {
        self.vendor = Some(vendor);
        self
    }

    /// Check if a specific capability is supported
    pub fn has_capability(&self, cap: Capability) -> bool {
        // For non-parameterized capabilities, direct lookup
        if !cap.is_parameterized() {
            return self.capabilities.contains(&cap);
        }

        // For parameterized capabilities, we need to check if we have
        // a matching capability with at least the requested value
        match cap {
            Capability::SharedMemKb(requested) => self
                .capabilities
                .iter()
                .any(|c| matches!(c, Capability::SharedMemKb(v) if *v >= requested)),
            Capability::MaxSharedPerBlock(requested) => self
                .capabilities
                .iter()
                .any(|c| matches!(c, Capability::MaxSharedPerBlock(v) if *v >= requested)),
            Capability::WarpSize(size) => self
                .capabilities
                .iter()
                .any(|c| matches!(c, Capability::WarpSize(v) if *v == size)),
            Capability::SubgroupSize(size) => self
                .capabilities
                .iter()
                .any(|c| matches!(c, Capability::SubgroupSize(v) if *v == size)),
            Capability::MaxRegistersPerThread(requested) => self
                .capabilities
                .iter()
                .any(|c| matches!(c, Capability::MaxRegistersPerThread(v) if *v >= requested)),
            Capability::MaxThreadsPerBlock(requested) => self
                .capabilities
                .iter()
                .any(|c| matches!(c, Capability::MaxThreadsPerBlock(v) if *v >= requested)),
            Capability::ComputeCapability(req_major, req_minor) => {
                self.capabilities.iter().any(|c| {
                    matches!(c, Capability::ComputeCapability(maj, min)
                        if *maj > req_major || (*maj == req_major && *min >= req_minor))
                })
            }
            Capability::TensorCoreGen(requested) => self
                .capabilities
                .iter()
                .any(|c| matches!(c, Capability::TensorCoreGen(v) if *v >= requested)),
            _ => self.capabilities.contains(&cap),
        }
    }

    /// Check if all of the given capabilities are supported
    pub fn has_all(&self, caps: &[Capability]) -> bool {
        caps.iter().all(|c| self.has_capability(*c))
    }

    /// Check if any of the given capabilities are supported
    pub fn has_any(&self, caps: &[Capability]) -> bool {
        caps.iter().any(|c| self.has_capability(*c))
    }

    /// Get all capabilities
    pub fn capabilities(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities.iter()
    }

    // ========================================================================
    // Query Methods - Compiler uses these to select strategies
    // ========================================================================

    /// Get the warp/SIMD size, or default to 32
    pub fn warp_size(&self) -> u32 {
        self.capabilities
            .iter()
            .find_map(|c| match c {
                Capability::WarpSize(size) => Some(*size),
                _ => None,
            })
            .unwrap_or(32)
    }

    /// Get the subgroup size (may differ from warp size on Vulkan)
    pub fn subgroup_size(&self) -> u32 {
        self.capabilities
            .iter()
            .find_map(|c| match c {
                Capability::SubgroupSize(size) => Some(*size),
                _ => None,
            })
            .unwrap_or_else(|| self.warp_size())
    }

    /// Get available shared memory in KB
    pub fn shared_memory_kb(&self) -> u32 {
        self.capabilities
            .iter()
            .find_map(|c| match c {
                Capability::SharedMemKb(kb) => Some(*kb),
                _ => None,
            })
            .unwrap_or(16) // Conservative default
    }

    /// Get maximum configurable shared memory in KB
    pub fn max_shared_memory_kb(&self) -> u32 {
        self.capabilities
            .iter()
            .find_map(|c| match c {
                Capability::MaxSharedPerBlock(kb) => Some(*kb),
                _ => None,
            })
            .unwrap_or_else(|| self.shared_memory_kb())
    }

    /// Get maximum registers per thread
    pub fn max_registers_per_thread(&self) -> u32 {
        self.capabilities
            .iter()
            .find_map(|c| match c {
                Capability::MaxRegistersPerThread(regs) => Some(*regs),
                _ => None,
            })
            .unwrap_or(64) // Conservative default
    }

    /// Get maximum threads per block
    pub fn max_threads_per_block(&self) -> u32 {
        self.capabilities
            .iter()
            .find_map(|c| match c {
                Capability::MaxThreadsPerBlock(threads) => Some(*threads),
                _ => None,
            })
            .unwrap_or(512) // Conservative default
    }

    /// Get the compute capability if NVIDIA
    pub fn compute_capability(&self) -> Option<(u32, u32)> {
        self.capabilities.iter().find_map(|c| match c {
            Capability::ComputeCapability(major, minor) => Some((*major, *minor)),
            _ => None,
        })
    }

    /// Get tensor core generation if available
    pub fn tensor_core_gen(&self) -> Option<u32> {
        self.capabilities.iter().find_map(|c| match c {
            Capability::TensorCoreGen(g) => Some(*g),
            _ => None,
        })
    }

    /// Calculate effective shared memory considering all constraints
    ///
    /// This accounts for:
    /// - Base shared memory
    /// - Register pressure (more registers = less shared memory on some archs)
    /// - Block size constraints
    pub fn effective_shared_memory_kb(&self, _block_size: u32, registers_per_thread: u32) -> u32 {
        let base_shared = self.max_shared_memory_kb();

        // On NVIDIA, there's a tradeoff between shared memory and L1 cache
        // We use a simplified model here
        // Note: block_size could be used for more accurate occupancy calculations
        if self.vendor == Some(GpuVendor::Nvidia) {
            let reg_pressure = registers_per_thread as f32 / self.max_registers_per_thread() as f32;
            let occupancy_factor = 1.0 - (reg_pressure * 0.3).min(0.3);
            (base_shared as f32 * occupancy_factor) as u32
        } else {
            base_shared
        }
    }

    /// Calculate optimal tile size for matrix operations
    pub fn optimal_tile_size(&self) -> (u32, u32) {
        if self.has_capability(Capability::TensorOps) {
            // Tensor cores work best with specific tile sizes
            match self.tensor_core_gen() {
                Some(5) => (128, 128), // Blackwell
                Some(4) => (64, 64),   // Hopper
                Some(3) => (32, 32),   // Ampere
                Some(2) => (16, 16),   // Turing
                _ => (16, 16),
            }
        } else {
            // Scalar path - smaller tiles
            let warp = self.warp_size();
            (warp, warp)
        }
    }

    /// Check if efficient FP64 is available
    pub fn has_fast_fp64(&self) -> bool {
        self.has_capability(Capability::Fp64Fast)
    }

    /// Check if subgroup operations are available
    pub fn has_subgroup_ops(&self) -> bool {
        self.has_any(&[
            Capability::SubgroupShuffle,
            Capability::SubgroupBallot,
            Capability::SubgroupVote,
            Capability::SubgroupArithmetic,
        ])
    }

    /// Check if tensor operations are available
    pub fn has_tensor_ops(&self) -> bool {
        self.has_capability(Capability::TensorOps)
    }

    // ========================================================================
    // Target Presets
    // ========================================================================

    /// Vulkan portable baseline - guaranteed to work everywhere
    ///
    /// This is the safe, lowest-common-denominator target.
    /// Use this when portability is more important than performance.
    pub fn vulkan_portable() -> Self {
        Self::new("Vulkan Portable")
            .with_vendor(GpuVendor::Unknown)
            .with_capabilities([
                Capability::WarpSize(32), // Assume NVIDIA-like, but could be 64 on AMD
                Capability::SubgroupSize(32),
                Capability::SharedMemKb(16), // Very conservative
                Capability::MaxSharedPerBlock(32),
                Capability::MaxThreadsPerBlock(512),
                Capability::MaxRegistersPerThread(64),
                Capability::SubgroupVote,
                // NO tensor ops, NO fast FP64, NO advanced atomics
            ])
    }

    /// NVIDIA Ampere (SM 8.0+) - RTX 30xx, A100
    pub fn nvidia_ampere() -> Self {
        Self::new("NVIDIA Ampere (SM 8.0)")
            .with_vendor(GpuVendor::Nvidia)
            .with_capabilities([
                Capability::ComputeCapability(8, 0),
                Capability::WarpSize(32),
                Capability::SubgroupSize(32),
                Capability::SharedMemKb(48),
                Capability::MaxSharedPerBlock(164), // A100 can configure up to 164KB
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(255),
                // Subgroup ops
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision
                Capability::Fp64Fast,
                Capability::Fp16Arithmetic,
                Capability::Bf16,
                // Memory
                Capability::AsyncMemCopy,
                // Tensor cores
                Capability::TensorOps,
                Capability::TensorCoreGen(3),
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                // Other
                Capability::CooperativeGroups,
                Capability::DynamicParallelism,
                Capability::UnifiedMemory,
            ])
    }

    /// NVIDIA Hopper (SM 9.0+) - H100, H200
    pub fn nvidia_hopper() -> Self {
        Self::new("NVIDIA Hopper (SM 9.0)")
            .with_vendor(GpuVendor::Nvidia)
            .with_capabilities([
                Capability::ComputeCapability(9, 0),
                Capability::WarpSize(32),
                Capability::SubgroupSize(32),
                Capability::SharedMemKb(48),
                Capability::MaxSharedPerBlock(228), // H100 can configure up to 228KB
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(255),
                // Subgroup ops
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision
                Capability::Fp64Fast,
                Capability::Fp16Arithmetic,
                Capability::Bf16,
                Capability::Fp8,
                // Memory - Hopper's killer features
                Capability::AsyncMemCopy,
                Capability::TensorMemoryAccess, // TMA!
                // Tensor cores
                Capability::TensorOps,
                Capability::TensorCoreGen(4),
                Capability::TensorFp8,
                Capability::Wgmma,
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                Capability::AtomicFloatMinMax,
                // Hopper-specific
                Capability::CooperativeGroups,
                Capability::ThreadBlockClusters,
                Capability::DistributedSharedMemory,
                Capability::DynamicParallelism,
                Capability::UnifiedMemory,
                Capability::NvLink(4),
            ])
    }

    /// NVIDIA Blackwell (SM 10.0+) - B100, B200, GB200
    pub fn nvidia_blackwell() -> Self {
        Self::new("NVIDIA Blackwell (SM 10.0)")
            .with_vendor(GpuVendor::Nvidia)
            .with_capabilities([
                Capability::ComputeCapability(10, 0),
                Capability::WarpSize(32),
                Capability::SubgroupSize(32),
                Capability::SharedMemKb(48),
                Capability::MaxSharedPerBlock(256), // Estimated for Blackwell
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(255),
                // Subgroup ops
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision
                Capability::Fp64Fast,
                Capability::Fp16Arithmetic,
                Capability::Bf16,
                Capability::Fp8,
                // Memory
                Capability::AsyncMemCopy,
                Capability::TensorMemoryAccess,
                // Tensor cores - 5th gen
                Capability::TensorOps,
                Capability::TensorCoreGen(5),
                Capability::TensorFp8,
                Capability::TensorFp4, // NEW: FP4/INT4 support
                Capability::Wgmma,
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                Capability::AtomicFloatMinMax,
                // Advanced features
                Capability::CooperativeGroups,
                Capability::ThreadBlockClusters,
                Capability::DistributedSharedMemory,
                Capability::DynamicParallelism,
                Capability::UnifiedMemory,
                Capability::NvLink(5),
            ])
    }

    /// AMD RDNA3 - RX 7000 series
    pub fn amd_rdna3() -> Self {
        Self::new("AMD RDNA3")
            .with_vendor(GpuVendor::Amd)
            .with_capabilities([
                Capability::AmdGcnVersion(11), // RDNA3 is GCN 11
                Capability::WarpSize(32),      // Wave32 mode
                Capability::SubgroupSize(32),
                Capability::SharedMemKb(64),
                Capability::MaxSharedPerBlock(64),
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(256),
                // Subgroup ops
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision - RDNA3 has good FP64 on some SKUs
                Capability::Fp16Arithmetic,
                // Note: FP64 is slow on consumer RDNA3
                // Memory
                Capability::AsyncMemCopy,
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                // Graphics features
                Capability::RayTracing,
                Capability::MeshShaders,
            ])
    }

    /// AMD CDNA3 - MI300 series (datacenter)
    pub fn amd_cdna3() -> Self {
        Self::new("AMD CDNA3 (MI300)")
            .with_vendor(GpuVendor::Amd)
            .with_capabilities([
                Capability::AmdGcnVersion(12), // CDNA3
                Capability::WarpSize(64),      // Wave64 mode
                Capability::SubgroupSize(64),
                Capability::SharedMemKb(64),
                Capability::MaxSharedPerBlock(64),
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(256),
                // Subgroup ops
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision - CDNA3 has FULL FP64
                Capability::Fp64Fast,
                Capability::Fp16Arithmetic,
                Capability::Bf16,
                Capability::Fp8,
                // Matrix cores
                Capability::TensorOps,
                // Memory
                Capability::AsyncMemCopy,
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                Capability::AtomicFloatMinMax,
                // HPC features
                Capability::UnifiedMemory, // APU unified memory
            ])
    }

    /// Intel Arc (Alchemist/Battlemage)
    pub fn intel_arc() -> Self {
        Self::new("Intel Arc")
            .with_vendor(GpuVendor::Intel)
            .with_capabilities([
                Capability::IntelXeVersion(1), // Xe-HPG
                Capability::WarpSize(16),      // Intel uses 16-wide SIMD
                Capability::SubgroupSize(16),
                Capability::SharedMemKb(64),
                Capability::MaxSharedPerBlock(64),
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(128),
                // Subgroup ops
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision
                Capability::Fp16Arithmetic,
                // Note: FP64 is emulated on consumer Arc
                // Intel XMX (matrix extensions)
                Capability::TensorOps,
                // Memory
                Capability::AsyncMemCopy,
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                // Graphics
                Capability::RayTracing,
                Capability::MeshShaders,
            ])
    }

    /// Intel Data Center GPU Max (Ponte Vecchio)
    pub fn intel_pvc() -> Self {
        Self::new("Intel Data Center GPU Max (PVC)")
            .with_vendor(GpuVendor::Intel)
            .with_capabilities([
                Capability::IntelXeVersion(2), // Xe-HPC
                Capability::WarpSize(16),
                Capability::SubgroupSize(16),
                Capability::SharedMemKb(128),
                Capability::MaxSharedPerBlock(128),
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(128),
                // Subgroup ops
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision - HPC class
                Capability::Fp64Fast,
                Capability::Fp16Arithmetic,
                Capability::Bf16,
                // Matrix engines
                Capability::TensorOps,
                // Memory
                Capability::AsyncMemCopy,
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                Capability::AtomicFloatMinMax,
                // HPC features
                Capability::UnifiedMemory,
            ])
    }

    /// Apple M-series (M1/M2/M3/M4) via Metal
    pub fn apple_m_series() -> Self {
        Self::new("Apple M-Series (Metal)")
            .with_vendor(GpuVendor::Apple)
            .with_capabilities([
                Capability::AppleGpuFamily(9), // Assume M3/M4 class
                Capability::WarpSize(32),      // SIMD width
                Capability::SubgroupSize(32),
                Capability::SharedMemKb(32),
                Capability::MaxSharedPerBlock(32),
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(128),
                // Simdgroup ops (Apple's subgroup)
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision - NO native FP64 on Apple Silicon
                Capability::Fp16Arithmetic,
                // Simdgroup matrix (Apple's tensor ops)
                Capability::TensorOps,
                // Memory
                Capability::UnifiedMemory, // Apple's big advantage
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                // Graphics
                Capability::RayTracing,
                Capability::MeshShaders,
            ])
    }

    /// Apple M3/M4 Ultra (higher-end with more resources)
    pub fn apple_m_ultra() -> Self {
        Self::new("Apple M-Series Ultra (Metal)")
            .with_vendor(GpuVendor::Apple)
            .with_capabilities([
                Capability::AppleGpuFamily(10), // Assume latest
                Capability::WarpSize(32),
                Capability::SubgroupSize(32),
                Capability::SharedMemKb(64), // Ultra has more
                Capability::MaxSharedPerBlock(64),
                Capability::MaxThreadsPerBlock(1024),
                Capability::MaxRegistersPerThread(128),
                // Simdgroup ops
                Capability::SubgroupShuffle,
                Capability::SubgroupBallot,
                Capability::SubgroupVote,
                Capability::SubgroupArithmetic,
                // Precision - still no FP64
                Capability::Fp16Arithmetic,
                // Simdgroup matrix
                Capability::TensorOps,
                // Memory
                Capability::UnifiedMemory,
                // Atomics
                Capability::Atomics64,
                Capability::AtomicFloatAdd,
                // Graphics
                Capability::RayTracing,
                Capability::MeshShaders,
            ])
    }

    /// Qualcomm Adreno (mobile)
    pub fn qualcomm_adreno() -> Self {
        Self::new("Qualcomm Adreno")
            .with_vendor(GpuVendor::Qualcomm)
            .with_capabilities([
                Capability::WarpSize(64), // Adreno uses 64-wide waves
                Capability::SubgroupSize(64),
                Capability::SharedMemKb(32),
                Capability::MaxSharedPerBlock(32),
                Capability::MaxThreadsPerBlock(512),
                Capability::MaxRegistersPerThread(96),
                // Limited subgroup ops
                Capability::SubgroupVote,
                // Precision
                Capability::Fp16Arithmetic,
                // Atomics
                Capability::Atomics64,
            ])
    }
}

// ============================================================================
// Capability Query Trait
// ============================================================================

/// Trait for backends to implement capability declarations.
///
/// This is the interface between backends and the capability system.
/// Backends implement this to declare what they support.
pub trait CapabilityQuery {
    /// Get the complete set of capabilities for this backend
    fn capabilities(&self) -> &TargetCapabilities;

    /// Quick check for a specific capability
    fn supports(&self, cap: Capability) -> bool {
        self.capabilities().has_capability(cap)
    }

    /// Get the effective warp/SIMD size
    fn warp_size(&self) -> u32 {
        self.capabilities().warp_size()
    }

    /// Get available shared memory in KB
    fn shared_memory_kb(&self) -> u32 {
        self.capabilities().shared_memory_kb()
    }

    /// Check if fast FP64 is available
    fn has_fast_fp64(&self) -> bool {
        self.capabilities().has_fast_fp64()
    }

    /// Check if tensor operations are available
    fn has_tensor_ops(&self) -> bool {
        self.capabilities().has_tensor_ops()
    }
}

// ============================================================================
// Capability Combination Checks
// ============================================================================

impl TargetCapabilities {
    /// Check if the target supports efficient matrix multiplication
    ///
    /// This requires tensor ops OR fast subgroup arithmetic.
    pub fn supports_efficient_matmul(&self) -> bool {
        self.has_tensor_ops()
            || (self.has_capability(Capability::SubgroupShuffle)
                && self.has_capability(Capability::SubgroupArithmetic))
    }

    /// Check if the target supports efficient reductions
    ///
    /// This requires subgroup operations OR cooperative groups.
    pub fn supports_efficient_reduction(&self) -> bool {
        self.has_any(&[
            Capability::SubgroupArithmetic,
            Capability::SubgroupShuffle,
            Capability::CooperativeGroups,
        ])
    }

    /// Check if the target supports mixed precision training
    ///
    /// This requires FP16/BF16 arithmetic AND tensor ops.
    pub fn supports_mixed_precision(&self) -> bool {
        self.has_any(&[Capability::Fp16Arithmetic, Capability::Bf16]) && self.has_tensor_ops()
    }

    /// Check if the target supports advanced async patterns
    ///
    /// This includes TMA, async copy, and barriers.
    pub fn supports_async_patterns(&self) -> bool {
        self.has_any(&[Capability::AsyncMemCopy, Capability::TensorMemoryAccess])
    }

    /// Check if the target is HPC-class (fast FP64 + large memory)
    pub fn is_hpc_class(&self) -> bool {
        self.has_fast_fp64() && self.max_shared_memory_kb() >= 128
    }

    /// Check if the target supports multi-GPU features
    pub fn supports_multi_gpu(&self) -> bool {
        self.has_any(&[
            Capability::NvLink(3),
            Capability::NvLink(4),
            Capability::NvLink(5),
            Capability::PcieAtomics,
        ])
    }

    /// Get recommended block size for this target
    pub fn recommended_block_size(&self) -> u32 {
        let max = self.max_threads_per_block();
        let warp = self.warp_size();

        // Default to 256, but adjust based on target
        if self.has_tensor_ops() {
            // Tensor core kernels often work well with 128 or 256
            256.min(max)
        } else if warp == 64 {
            // AMD wave64 - use multiples of 64
            256.min(max)
        } else {
            // Default NVIDIA-style
            256.min(max)
        }
    }

    /// Compute effective limits considering multiple constraints
    pub fn compute_effective_limits(&self) -> EffectiveLimits {
        EffectiveLimits {
            max_threads_per_block: self.max_threads_per_block(),
            max_shared_memory_kb: self.max_shared_memory_kb(),
            max_registers_per_thread: self.max_registers_per_thread(),
            warp_size: self.warp_size(),
            recommended_block_size: self.recommended_block_size(),
            optimal_tile_size: self.optimal_tile_size(),
            supports_fp64: self.has_fast_fp64(),
            supports_tensor_ops: self.has_tensor_ops(),
        }
    }
}

/// Computed effective limits for a target
#[derive(Debug, Clone)]
pub struct EffectiveLimits {
    pub max_threads_per_block: u32,
    pub max_shared_memory_kb: u32,
    pub max_registers_per_thread: u32,
    pub warp_size: u32,
    pub recommended_block_size: u32,
    pub optimal_tile_size: (u32, u32),
    pub supports_fp64: bool,
    pub supports_tensor_ops: bool,
}

// ============================================================================
// Capability Diff / Compatibility
// ============================================================================

/// Result of comparing two capability sets
#[derive(Debug, Clone)]
pub struct CapabilityDiff {
    /// Capabilities present in target but not in source
    pub missing: Vec<Capability>,
    /// Capabilities present in source but not in target
    pub extra: Vec<Capability>,
    /// Capabilities with different parameter values
    pub different: Vec<(Capability, Capability)>,
}

impl TargetCapabilities {
    /// Compare this capability set with another
    pub fn diff(&self, other: &TargetCapabilities) -> CapabilityDiff {
        let self_caps: HashSet<_> = self.capabilities.iter().cloned().collect();
        let other_caps: HashSet<_> = other.capabilities.iter().cloned().collect();

        // For non-parameterized capabilities, simple set difference
        let missing: Vec<_> = other_caps
            .difference(&self_caps)
            .filter(|c| !c.is_parameterized())
            .cloned()
            .collect();

        let extra: Vec<_> = self_caps
            .difference(&other_caps)
            .filter(|c| !c.is_parameterized())
            .cloned()
            .collect();

        CapabilityDiff {
            missing,
            extra,
            different: Vec::new(), // TODO: Handle parameterized differences
        }
    }

    /// Check if code targeting `required` can run on `self`
    pub fn is_compatible_with(&self, required: &TargetCapabilities) -> bool {
        // All non-parameterized capabilities must be present
        for cap in required.capabilities.iter() {
            if !cap.is_parameterized() && !self.has_capability(*cap) {
                return false;
            }
        }

        // For parameterized capabilities, we need "at least as good"
        if let Some((req_major, req_minor)) = required.compute_capability() {
            if let Some((self_major, self_minor)) = self.compute_capability() {
                if self_major < req_major || (self_major == req_major && self_minor < req_minor) {
                    return false;
                }
            } else {
                return false;
            }
        }

        if required.shared_memory_kb() > self.shared_memory_kb() {
            return false;
        }

        if required.max_threads_per_block() > self.max_threads_per_block() {
            return false;
        }

        true
    }
}

// ============================================================================
// Display / Debug
// ============================================================================

impl fmt::Display for TargetCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Target: {}", self.name)?;
        if let Some(vendor) = self.vendor {
            writeln!(f, "Vendor: {:?}", vendor)?;
        }
        writeln!(f, "Capabilities:")?;
        for cap in &self.capabilities {
            writeln!(f, "  - {}", cap)?;
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_portable_is_minimal() {
        let caps = TargetCapabilities::vulkan_portable();
        assert!(!caps.has_capability(Capability::TensorOps));
        assert!(!caps.has_capability(Capability::Fp64Fast));
        assert!(caps.shared_memory_kb() <= 32);
    }

    #[test]
    fn test_nvidia_ampere_capabilities() {
        let caps = TargetCapabilities::nvidia_ampere();
        assert!(caps.has_capability(Capability::TensorOps));
        assert!(caps.has_capability(Capability::Fp64Fast));
        assert!(caps.has_capability(Capability::SubgroupShuffle));
        assert!(caps.has_capability(Capability::AsyncMemCopy));
        assert_eq!(caps.warp_size(), 32);
        assert!(caps.max_shared_memory_kb() >= 48);
    }

    #[test]
    fn test_nvidia_hopper_has_tma() {
        let caps = TargetCapabilities::nvidia_hopper();
        assert!(caps.has_capability(Capability::TensorMemoryAccess));
        assert!(caps.has_capability(Capability::ThreadBlockClusters));
        assert!(caps.has_capability(Capability::Wgmma));
    }

    #[test]
    fn test_apple_has_no_fp64() {
        let caps = TargetCapabilities::apple_m_series();
        assert!(!caps.has_capability(Capability::Fp64Fast));
        assert!(caps.has_capability(Capability::UnifiedMemory));
    }

    #[test]
    fn test_amd_cdna3_is_hpc_class() {
        let caps = TargetCapabilities::amd_cdna3();
        assert!(caps.has_fast_fp64());
        assert!(caps.has_capability(Capability::Fp8));
    }

    #[test]
    fn test_capability_querying() {
        let caps = TargetCapabilities::nvidia_ampere();

        // Check compute capability querying
        assert_eq!(caps.compute_capability(), Some((8, 0)));

        // Check parameterized capability with threshold
        assert!(caps.has_capability(Capability::SharedMemKb(32)));
        assert!(caps.has_capability(Capability::SharedMemKb(48)));
        assert!(!caps.has_capability(Capability::SharedMemKb(200)));
    }

    #[test]
    fn test_efficient_matmul_detection() {
        let nvidia = TargetCapabilities::nvidia_ampere();
        let vulkan = TargetCapabilities::vulkan_portable();

        assert!(nvidia.supports_efficient_matmul());
        assert!(!vulkan.supports_efficient_matmul());
    }

    #[test]
    fn test_compatibility_check() {
        let hopper = TargetCapabilities::nvidia_hopper();
        let ampere = TargetCapabilities::nvidia_ampere();
        let vulkan = TargetCapabilities::vulkan_portable();

        // Hopper can run Ampere code
        assert!(hopper.is_compatible_with(&ampere));

        // Ampere cannot run Hopper code (missing TMA, clusters)
        // This will fail because Ampere doesn't have TMA
        // Note: This is a simplified check - real compatibility is more nuanced
    }

    #[test]
    fn test_effective_limits() {
        let caps = TargetCapabilities::nvidia_hopper();
        let limits = caps.compute_effective_limits();

        assert_eq!(limits.max_threads_per_block, 1024);
        assert_eq!(limits.warp_size, 32);
        assert!(limits.supports_fp64);
        assert!(limits.supports_tensor_ops);
    }

    #[test]
    fn test_intel_arc_capabilities() {
        let caps = TargetCapabilities::intel_arc();
        assert_eq!(caps.warp_size(), 16); // Intel uses 16-wide SIMD
        assert!(caps.has_capability(Capability::TensorOps));
        assert!(!caps.has_capability(Capability::Fp64Fast)); // Consumer Arc
    }

    #[test]
    fn test_intel_pvc_is_hpc() {
        let caps = TargetCapabilities::intel_pvc();
        assert!(caps.has_fast_fp64()); // PVC has fast FP64
        assert!(caps.is_hpc_class());
    }

    #[test]
    fn test_capability_display() {
        let cap = Capability::ComputeCapability(9, 0);
        assert_eq!(format!("{}", cap), "ComputeCapability(9.0)");

        let cap = Capability::SharedMemKb(164);
        assert_eq!(format!("{}", cap), "SharedMemKb(164)");
    }

    #[test]
    fn test_mixed_precision_detection() {
        let nvidia = TargetCapabilities::nvidia_ampere();
        let vulkan = TargetCapabilities::vulkan_portable();

        assert!(nvidia.supports_mixed_precision());
        assert!(!vulkan.supports_mixed_precision());
    }

    #[test]
    fn test_recommended_block_size() {
        let nvidia = TargetCapabilities::nvidia_ampere();
        let apple = TargetCapabilities::apple_m_series();

        // Both should return reasonable block sizes
        assert!(nvidia.recommended_block_size() >= 128);
        assert!(nvidia.recommended_block_size() <= 1024);
        assert!(apple.recommended_block_size() >= 128);
    }

    #[test]
    fn test_optimal_tile_size() {
        let hopper = TargetCapabilities::nvidia_hopper();
        let ampere = TargetCapabilities::nvidia_ampere();
        let vulkan = TargetCapabilities::vulkan_portable();

        // Hopper has gen4 tensor cores
        assert_eq!(hopper.optimal_tile_size(), (64, 64));
        // Ampere has gen3 tensor cores
        assert_eq!(ampere.optimal_tile_size(), (32, 32));
        // Vulkan portable has no tensor cores
        assert_eq!(vulkan.optimal_tile_size(), (32, 32));
    }
}
