//! Instruction Cost Database (Phase 12)
//!
//! Provides architecture-specific instruction costs including latency,
//! throughput, and resource usage for performance modeling.

use std::collections::HashMap;

use super::ir::{CudaArch, GpuKernel, GpuOp, MemorySpace};

// ============================================================================
// Instruction Cost Types
// ============================================================================

/// Cost of a single GPU instruction
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstructionCost {
    /// Latency in cycles (time to first result)
    pub latency: u32,
    /// Throughput in ops/cycle/SM
    pub throughput: f64,
    /// Memory bandwidth consumed (bytes/op, 0 for compute)
    pub memory_bytes: u32,
    /// Registers consumed by operation
    pub registers: u32,
    /// Requires special function unit (SFU)
    pub uses_sfu: bool,
    /// Requires tensor core
    pub uses_tensor_core: bool,
}

impl Default for InstructionCost {
    fn default() -> Self {
        Self {
            latency: 4,
            throughput: 64.0,
            memory_bytes: 0,
            registers: 1,
            uses_sfu: false,
            uses_tensor_core: false,
        }
    }
}

impl InstructionCost {
    /// Create a compute instruction cost
    pub fn compute(latency: u32, throughput: f64) -> Self {
        Self {
            latency,
            throughput,
            memory_bytes: 0,
            registers: 1,
            uses_sfu: false,
            uses_tensor_core: false,
        }
    }

    /// Create an SFU instruction cost
    pub fn sfu(latency: u32, throughput: f64) -> Self {
        Self {
            latency,
            throughput,
            memory_bytes: 0,
            registers: 1,
            uses_sfu: true,
            uses_tensor_core: false,
        }
    }

    /// Create a memory instruction cost
    pub fn memory(latency: u32, throughput: f64, bytes: u32) -> Self {
        Self {
            latency,
            throughput,
            memory_bytes: bytes,
            registers: 1,
            uses_sfu: false,
            uses_tensor_core: false,
        }
    }

    /// Create a tensor core instruction cost
    pub fn tensor_core(latency: u32, throughput: f64) -> Self {
        Self {
            latency,
            throughput,
            memory_bytes: 0,
            registers: 4,
            uses_sfu: false,
            uses_tensor_core: true,
        }
    }
}

// ============================================================================
// Instruction Classification
// ============================================================================

/// Instruction classification for cost lookup
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum InstructionClass {
    // Integer arithmetic
    IntAdd,
    IntMul,
    IntDiv,
    IntMod,
    IntBitwise,
    IntShift,
    IntCompare,

    // FP32 arithmetic
    Fp32Add,
    Fp32Mul,
    Fp32Fma,
    Fp32Div,
    Fp32Sqrt,
    Fp32Rsqrt,

    // FP16 arithmetic
    Fp16Add,
    Fp16Mul,
    Fp16Fma,

    // BF16 arithmetic
    Bf16Add,
    Bf16Mul,
    Bf16Fma,

    // FP64 arithmetic
    Fp64Add,
    Fp64Mul,
    Fp64Fma,
    Fp64Div,

    // Special functions (SFU)
    Fp32Sin,
    Fp32Cos,
    Fp32Exp,
    Fp32Log,
    Fp32Pow,
    Fp32Tanh,

    // Memory operations
    GlobalLoad,
    GlobalStore,
    SharedLoad,
    SharedStore,
    LocalLoad,
    LocalStore,
    ConstantLoad,

    // Tensor Core operations
    TensorMmaFp16,
    TensorMmaFp8,
    TensorMmaInt8,
    TensorMmaFp4,
    TensorMmaBf16,

    // Quantization operations
    Dp4a,
    QuantizeInt8,
    DequantizeInt8,
    QuantizeInt4,
    DequantizeInt4,

    // Synchronization
    SyncThreads,
    SyncWarp,
    MemoryFence,

    // Atomic operations
    AtomicAdd,
    AtomicMin,
    AtomicMax,
    AtomicCas,

    // Control flow
    Branch,
    PredicatedExec,
    Call,
    Return,

    // Warp operations
    WarpShuffle,
    WarpVote,
    WarpReduce,

    // Miscellaneous
    Convert,
    Select,
    Nop,
}

// ============================================================================
// Architecture Performance Data
// ============================================================================

/// Peak performance characteristics for an architecture
#[derive(Debug, Clone, Copy)]
pub struct ArchPeakPerf {
    /// Peak FP32 throughput (TFLOPS)
    pub fp32_tflops: f64,
    /// Peak FP16 throughput (TFLOPS)
    pub fp16_tflops: f64,
    /// Peak Tensor Core FP16 throughput (TFLOPS)
    pub tensor_fp16_tflops: f64,
    /// Peak INT8 throughput (TOPS)
    pub int8_tops: f64,
    /// Peak memory bandwidth (GB/s)
    pub memory_bandwidth_gbs: f64,
    /// L2 cache bandwidth (GB/s)
    pub l2_bandwidth_gbs: f64,
    /// Shared memory bandwidth per SM (GB/s)
    pub shared_bandwidth_gbs: f64,
    /// Number of SMs
    pub num_sms: u32,
    /// Clock frequency (GHz)
    pub clock_ghz: f64,
}

impl ArchPeakPerf {
    /// Get peak performance for Turing (RTX 2080 Ti reference)
    pub fn turing() -> Self {
        Self {
            fp32_tflops: 16.3,
            fp16_tflops: 32.6,
            tensor_fp16_tflops: 130.0,
            int8_tops: 260.0,
            memory_bandwidth_gbs: 616.0,
            l2_bandwidth_gbs: 2000.0,
            shared_bandwidth_gbs: 12000.0,
            num_sms: 68,
            clock_ghz: 1.545,
        }
    }

    /// Get peak performance for Ampere (A100 reference)
    pub fn ampere() -> Self {
        Self {
            fp32_tflops: 19.5,
            fp16_tflops: 39.0,
            tensor_fp16_tflops: 312.0,
            int8_tops: 624.0,
            memory_bandwidth_gbs: 1555.0,
            l2_bandwidth_gbs: 4000.0,
            shared_bandwidth_gbs: 19000.0,
            num_sms: 108,
            clock_ghz: 1.41,
        }
    }

    /// Get peak performance for Ada (RTX 4090 reference)
    pub fn ada() -> Self {
        Self {
            fp32_tflops: 82.6,
            fp16_tflops: 165.0,
            tensor_fp16_tflops: 660.0,
            int8_tops: 1320.0,
            memory_bandwidth_gbs: 1008.0,
            l2_bandwidth_gbs: 6000.0,
            shared_bandwidth_gbs: 25000.0,
            num_sms: 128,
            clock_ghz: 2.52,
        }
    }

    /// Get peak performance for Hopper (H100 reference)
    pub fn hopper() -> Self {
        Self {
            fp32_tflops: 67.0,
            fp16_tflops: 134.0,
            tensor_fp16_tflops: 989.0,
            int8_tops: 1979.0,
            memory_bandwidth_gbs: 3350.0,
            l2_bandwidth_gbs: 12000.0,
            shared_bandwidth_gbs: 33000.0,
            num_sms: 132,
            clock_ghz: 1.83,
        }
    }

    /// Get peak performance for Blackwell (B200 reference)
    pub fn blackwell() -> Self {
        Self {
            fp32_tflops: 90.0,
            fp16_tflops: 180.0,
            tensor_fp16_tflops: 2500.0,
            int8_tops: 5000.0,
            memory_bandwidth_gbs: 8000.0,
            l2_bandwidth_gbs: 20000.0,
            shared_bandwidth_gbs: 50000.0,
            num_sms: 192,
            clock_ghz: 2.1,
        }
    }

    /// Get peak performance for an architecture
    pub fn for_arch(arch: CudaArch) -> Self {
        match arch {
            CudaArch::Turing => Self::turing(),
            CudaArch::Ampere => Self::ampere(),
            CudaArch::Ada => Self::ada(),
            CudaArch::Hopper => Self::hopper(),
            CudaArch::Blackwell | CudaArch::BlackwellUltra => Self::blackwell(),
        }
    }
}

// ============================================================================
// Cost Database
// ============================================================================

/// Architecture-specific instruction cost database
pub struct CostDatabase {
    arch: CudaArch,
    /// Peak performance characteristics
    pub peak: ArchPeakPerf,
    /// Instruction costs
    costs: HashMap<InstructionClass, InstructionCost>,
}

impl CostDatabase {
    /// Create database for specific architecture
    pub fn for_arch(arch: CudaArch) -> Self {
        let peak = ArchPeakPerf::for_arch(arch);
        let costs = Self::build_cost_table(arch);

        Self { arch, peak, costs }
    }

    /// Build the cost table for an architecture
    fn build_cost_table(arch: CudaArch) -> HashMap<InstructionClass, InstructionCost> {
        let mut costs = HashMap::new();

        // Base throughput multiplier based on architecture
        let throughput_mult = match arch {
            CudaArch::Turing => 1.0,
            CudaArch::Ampere => 1.2,
            CudaArch::Ada => 1.5,
            CudaArch::Hopper => 1.8,
            CudaArch::Blackwell | CudaArch::BlackwellUltra => 2.0,
        };

        // Integer operations
        costs.insert(
            InstructionClass::IntAdd,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::IntMul,
            InstructionCost::compute(4, 16.0 * throughput_mult),
        );
        costs.insert(InstructionClass::IntDiv, InstructionCost::compute(32, 2.0));
        costs.insert(InstructionClass::IntMod, InstructionCost::compute(32, 2.0));
        costs.insert(
            InstructionClass::IntBitwise,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::IntShift,
            InstructionCost::compute(4, 32.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::IntCompare,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );

        // FP32 operations
        costs.insert(
            InstructionClass::Fp32Add,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Mul,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Fma,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Div,
            InstructionCost::compute(16, 4.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Sqrt,
            InstructionCost::sfu(8, 8.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Rsqrt,
            InstructionCost::sfu(8, 8.0 * throughput_mult),
        );

        // FP16 operations (2x throughput of FP32)
        costs.insert(
            InstructionClass::Fp16Add,
            InstructionCost::compute(4, 128.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp16Mul,
            InstructionCost::compute(4, 128.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp16Fma,
            InstructionCost::compute(4, 128.0 * throughput_mult),
        );

        // BF16 operations (same as FP16 on modern archs)
        costs.insert(
            InstructionClass::Bf16Add,
            InstructionCost::compute(4, 128.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Bf16Mul,
            InstructionCost::compute(4, 128.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Bf16Fma,
            InstructionCost::compute(4, 128.0 * throughput_mult),
        );

        // FP64 operations (1/2 to 1/32 of FP32 throughput)
        let fp64_mult = match arch {
            CudaArch::Turing | CudaArch::Ada => 0.03125, // 1/32
            CudaArch::Ampere
            | CudaArch::Hopper
            | CudaArch::Blackwell
            | CudaArch::BlackwellUltra => 0.5,
        };
        costs.insert(
            InstructionClass::Fp64Add,
            InstructionCost::compute(8, 32.0 * fp64_mult * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp64Mul,
            InstructionCost::compute(8, 32.0 * fp64_mult * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp64Fma,
            InstructionCost::compute(8, 32.0 * fp64_mult * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp64Div,
            InstructionCost::compute(32, 2.0 * fp64_mult),
        );

        // Special functions (SFU)
        costs.insert(
            InstructionClass::Fp32Sin,
            InstructionCost::sfu(8, 8.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Cos,
            InstructionCost::sfu(8, 8.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Exp,
            InstructionCost::sfu(8, 8.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Log,
            InstructionCost::sfu(8, 8.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Pow,
            InstructionCost::sfu(16, 4.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Fp32Tanh,
            InstructionCost::sfu(8, 8.0 * throughput_mult),
        );

        // Memory operations (latency varies significantly)
        costs.insert(
            InstructionClass::GlobalLoad,
            InstructionCost::memory(300, 32.0, 4),
        );
        costs.insert(
            InstructionClass::GlobalStore,
            InstructionCost::memory(300, 32.0, 4),
        );
        costs.insert(
            InstructionClass::SharedLoad,
            InstructionCost::memory(25, 128.0, 4),
        );
        costs.insert(
            InstructionClass::SharedStore,
            InstructionCost::memory(25, 128.0, 4),
        );
        costs.insert(
            InstructionClass::LocalLoad,
            InstructionCost::memory(300, 32.0, 4),
        );
        costs.insert(
            InstructionClass::LocalStore,
            InstructionCost::memory(300, 32.0, 4),
        );
        costs.insert(
            InstructionClass::ConstantLoad,
            InstructionCost::memory(4, 128.0, 4),
        );

        // Tensor Core operations
        let tensor_throughput = match arch {
            CudaArch::Turing => 512.0,
            CudaArch::Ampere => 1024.0,
            CudaArch::Ada => 2048.0,
            CudaArch::Hopper => 4096.0,
            CudaArch::Blackwell | CudaArch::BlackwellUltra => 8192.0,
        };
        costs.insert(
            InstructionClass::TensorMmaFp16,
            InstructionCost::tensor_core(8, tensor_throughput),
        );
        costs.insert(
            InstructionClass::TensorMmaBf16,
            InstructionCost::tensor_core(8, tensor_throughput),
        );
        costs.insert(
            InstructionClass::TensorMmaFp8,
            InstructionCost::tensor_core(8, tensor_throughput * 2.0),
        );
        costs.insert(
            InstructionClass::TensorMmaInt8,
            InstructionCost::tensor_core(8, tensor_throughput * 2.0),
        );
        costs.insert(
            InstructionClass::TensorMmaFp4,
            InstructionCost::tensor_core(8, tensor_throughput * 4.0),
        );

        // Quantization operations
        costs.insert(
            InstructionClass::Dp4a,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::QuantizeInt8,
            InstructionCost::compute(8, 32.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::DequantizeInt8,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::QuantizeInt4,
            InstructionCost::compute(12, 16.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::DequantizeInt4,
            InstructionCost::compute(8, 32.0 * throughput_mult),
        );

        // Synchronization
        costs.insert(
            InstructionClass::SyncThreads,
            InstructionCost::compute(20, 1.0),
        );
        costs.insert(
            InstructionClass::SyncWarp,
            InstructionCost::compute(4, 32.0),
        );
        costs.insert(
            InstructionClass::MemoryFence,
            InstructionCost::compute(10, 8.0),
        );

        // Atomic operations
        costs.insert(
            InstructionClass::AtomicAdd,
            InstructionCost::memory(100, 4.0, 4),
        );
        costs.insert(
            InstructionClass::AtomicMin,
            InstructionCost::memory(100, 4.0, 4),
        );
        costs.insert(
            InstructionClass::AtomicMax,
            InstructionCost::memory(100, 4.0, 4),
        );
        costs.insert(
            InstructionClass::AtomicCas,
            InstructionCost::memory(100, 4.0, 8),
        );

        // Control flow
        costs.insert(InstructionClass::Branch, InstructionCost::compute(4, 32.0));
        costs.insert(
            InstructionClass::PredicatedExec,
            InstructionCost::compute(0, 64.0),
        );
        costs.insert(InstructionClass::Call, InstructionCost::compute(8, 16.0));
        costs.insert(InstructionClass::Return, InstructionCost::compute(4, 32.0));

        // Warp operations
        costs.insert(
            InstructionClass::WarpShuffle,
            InstructionCost::compute(4, 32.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::WarpVote,
            InstructionCost::compute(4, 32.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::WarpReduce,
            InstructionCost::compute(8, 16.0 * throughput_mult),
        );

        // Miscellaneous
        costs.insert(
            InstructionClass::Convert,
            InstructionCost::compute(4, 32.0 * throughput_mult),
        );
        costs.insert(
            InstructionClass::Select,
            InstructionCost::compute(4, 64.0 * throughput_mult),
        );
        costs.insert(InstructionClass::Nop, InstructionCost::compute(0, 64.0));

        costs
    }

    /// Get the architecture this database was built for
    pub fn arch(&self) -> CudaArch {
        self.arch
    }

    /// Get cost for an instruction class
    pub fn get_cost(&self, class: InstructionClass) -> InstructionCost {
        self.costs.get(&class).copied().unwrap_or_default()
    }

    /// Classify a GpuOp into an instruction class
    pub fn classify_op(op: &GpuOp) -> InstructionClass {
        match op {
            // Integer arithmetic
            GpuOp::Add(_, _) | GpuOp::Sub(_, _) => InstructionClass::IntAdd,
            GpuOp::Mul(_, _) => InstructionClass::IntMul,
            GpuOp::Div(_, _) => InstructionClass::IntDiv,
            GpuOp::Rem(_, _) => InstructionClass::IntMod,
            GpuOp::BitAnd(_, _) | GpuOp::BitOr(_, _) | GpuOp::BitXor(_, _) | GpuOp::BitNot(_) => {
                InstructionClass::IntBitwise
            }
            GpuOp::Shl(_, _) | GpuOp::Shr(_, _) | GpuOp::LShr(_, _) => InstructionClass::IntShift,
            GpuOp::Neg(_) => InstructionClass::IntAdd,

            // Floating-point arithmetic (assume FP32 by default)
            GpuOp::FAdd(_, _) | GpuOp::FSub(_, _) | GpuOp::FNeg(_) => InstructionClass::Fp32Add,
            GpuOp::FMul(_, _) => InstructionClass::Fp32Mul,
            GpuOp::FDiv(_, _) => InstructionClass::Fp32Div,
            GpuOp::FMulAdd(_, _, _) => InstructionClass::Fp32Fma,

            // Special functions (SFU)
            GpuOp::FastSin(_) => InstructionClass::Fp32Sin,
            GpuOp::FastCos(_) => InstructionClass::Fp32Cos,
            GpuOp::FastExp(_) => InstructionClass::Fp32Exp,
            GpuOp::FastLog(_) => InstructionClass::Fp32Log,
            GpuOp::FastSqrt(_) => InstructionClass::Fp32Sqrt,
            GpuOp::FastRsqrt(_) => InstructionClass::Fp32Rsqrt,

            // Comparisons
            GpuOp::Eq(_, _)
            | GpuOp::Ne(_, _)
            | GpuOp::Lt(_, _)
            | GpuOp::Le(_, _)
            | GpuOp::Gt(_, _)
            | GpuOp::Ge(_, _) => InstructionClass::IntCompare,
            GpuOp::FEq(_, _)
            | GpuOp::FNe(_, _)
            | GpuOp::FLt(_, _)
            | GpuOp::FLe(_, _)
            | GpuOp::FGt(_, _)
            | GpuOp::FGe(_, _) => InstructionClass::Fp32Add, // Comparisons are cheap

            // Memory operations
            GpuOp::Load(_, space) => match space {
                MemorySpace::Global => InstructionClass::GlobalLoad,
                MemorySpace::Shared => InstructionClass::SharedLoad,
                MemorySpace::Local => InstructionClass::LocalLoad,
                MemorySpace::Constant => InstructionClass::ConstantLoad,
                _ => InstructionClass::GlobalLoad,
            },
            GpuOp::Store(_, _, space) => match space {
                MemorySpace::Global => InstructionClass::GlobalStore,
                MemorySpace::Shared => InstructionClass::SharedStore,
                MemorySpace::Local => InstructionClass::LocalStore,
                _ => InstructionClass::GlobalStore,
            },

            // Tensor Core operations
            GpuOp::TileMma { .. } => InstructionClass::TensorMmaFp16,
            GpuOp::WgmmaFp8 { .. } => InstructionClass::TensorMmaFp8,
            GpuOp::WgmmaBf16 { .. } => InstructionClass::TensorMmaBf16,
            GpuOp::WgmmaFp4 { .. } => InstructionClass::TensorMmaFp4,

            // Quantization
            GpuOp::Dp4a { .. } => InstructionClass::Dp4a,
            GpuOp::QuantizeF32ToInt8 { .. } => InstructionClass::QuantizeInt8,
            GpuOp::DequantizeInt8ToF32 { .. } => InstructionClass::DequantizeInt8,
            GpuOp::QuantizeF32ToInt4 { .. } => InstructionClass::QuantizeInt4,
            GpuOp::DequantizeInt4ToF32Lo { .. } | GpuOp::DequantizeInt4ToF32Hi { .. } => {
                InstructionClass::DequantizeInt4
            }

            // Synchronization
            GpuOp::SyncThreads => InstructionClass::SyncThreads,
            GpuOp::SyncWarp(_) => InstructionClass::SyncWarp,
            GpuOp::MemoryFence(_) => InstructionClass::MemoryFence,

            // Atomics
            GpuOp::AtomicAdd(_, _) => InstructionClass::AtomicAdd,
            GpuOp::AtomicMin(_, _) => InstructionClass::AtomicMin,
            GpuOp::AtomicMax(_, _) => InstructionClass::AtomicMax,
            GpuOp::AtomicCas(_, _, _) => InstructionClass::AtomicCas,

            // Warp operations
            GpuOp::WarpShuffle(_, _)
            | GpuOp::WarpShuffleUp(_, _)
            | GpuOp::WarpShuffleDown(_, _)
            | GpuOp::WarpShuffleXor(_, _) => InstructionClass::WarpShuffle,
            GpuOp::WarpVote(_, _) => InstructionClass::WarpVote,
            GpuOp::WarpReduce(_, _) => InstructionClass::WarpReduce,
            GpuOp::WarpMatch(_) => InstructionClass::WarpVote,

            // Control flow
            GpuOp::Select(_, _, _) => InstructionClass::Select,

            // Type conversions
            GpuOp::Trunc(_, _)
            | GpuOp::ZExt(_, _)
            | GpuOp::SExt(_, _)
            | GpuOp::FpTrunc(_, _)
            | GpuOp::FpExt(_, _)
            | GpuOp::FpToSi(_, _)
            | GpuOp::FpToUi(_, _)
            | GpuOp::SiToFp(_, _)
            | GpuOp::UiToFp(_, _)
            | GpuOp::Bitcast(_, _) => InstructionClass::Convert,

            // BF16/FP8 conversions
            GpuOp::F32ToBF16(_)
            | GpuOp::BF16ToF32(_)
            | GpuOp::F32ToF8E4M3(_)
            | GpuOp::F8E4M3ToF32(_)
            | GpuOp::F32ToF8E5M2(_)
            | GpuOp::F8E5M2ToF32(_)
            | GpuOp::F32ToF4(_)
            | GpuOp::F4ToF32(_) => InstructionClass::Convert,

            // Constants (not FLOPS - just loading immediate values)
            GpuOp::ConstInt(_, _) | GpuOp::ConstFloat(_, _) | GpuOp::ConstBool(_) => {
                InstructionClass::ConstantLoad
            }

            // Default for unclassified operations (not counted as FLOPS)
            _ => InstructionClass::Nop,
        }
    }

    /// Compute total cost estimate for a kernel
    pub fn estimate_kernel_cycles(&self, kernel: &GpuKernel) -> KernelCostEstimate {
        let mut compute_cycles = 0u64;
        let mut memory_cycles = 0u64;
        let mut sync_cycles = 0u64;
        let mut total_ops = 0u64;

        for block in &kernel.blocks {
            for (_, op) in &block.instructions {
                let class = Self::classify_op(op);
                let cost = self.get_cost(class);

                total_ops += 1;

                if cost.memory_bytes > 0 {
                    memory_cycles += cost.latency as u64;
                } else if matches!(
                    class,
                    InstructionClass::SyncThreads
                        | InstructionClass::SyncWarp
                        | InstructionClass::MemoryFence
                ) {
                    sync_cycles += cost.latency as u64;
                } else {
                    compute_cycles += cost.latency as u64;
                }
            }
        }

        let total_cycles = compute_cycles.max(memory_cycles) + sync_cycles;

        let limiting_resource = if memory_cycles > compute_cycles {
            LimitingResource::MemoryBandwidth
        } else if sync_cycles > compute_cycles / 4 {
            LimitingResource::Synchronization
        } else {
            LimitingResource::Compute
        };

        KernelCostEstimate {
            total_cycles,
            compute_cycles,
            memory_cycles,
            sync_cycles,
            limiting_resource,
            total_ops,
        }
    }

    /// Count FLOPS for a kernel
    pub fn count_flops(&self, kernel: &GpuKernel) -> FlopsCount {
        let mut fp32_flops = 0u64;
        let mut fp16_flops = 0u64;
        let mut fp64_flops = 0u64;
        let mut int_ops = 0u64;
        let mut tensor_flops = 0u64;

        for block in &kernel.blocks {
            for (_, op) in &block.instructions {
                let class = Self::classify_op(op);
                match class {
                    InstructionClass::Fp32Add
                    | InstructionClass::Fp32Mul
                    | InstructionClass::Fp32Div
                    | InstructionClass::Fp32Sqrt
                    | InstructionClass::Fp32Rsqrt => fp32_flops += 1,

                    InstructionClass::Fp32Fma => fp32_flops += 2, // FMA = 2 FLOPS

                    InstructionClass::Fp32Sin
                    | InstructionClass::Fp32Cos
                    | InstructionClass::Fp32Exp
                    | InstructionClass::Fp32Log
                    | InstructionClass::Fp32Tanh => fp32_flops += 8, // Approximation for SFU

                    InstructionClass::Fp32Pow => fp32_flops += 16,

                    InstructionClass::Fp16Add | InstructionClass::Fp16Mul => fp16_flops += 1,
                    InstructionClass::Fp16Fma => fp16_flops += 2,

                    InstructionClass::Bf16Add | InstructionClass::Bf16Mul => fp16_flops += 1,
                    InstructionClass::Bf16Fma => fp16_flops += 2,

                    InstructionClass::Fp64Add
                    | InstructionClass::Fp64Mul
                    | InstructionClass::Fp64Div => fp64_flops += 1,
                    InstructionClass::Fp64Fma => fp64_flops += 2,

                    InstructionClass::IntAdd
                    | InstructionClass::IntMul
                    | InstructionClass::IntDiv
                    | InstructionClass::IntMod
                    | InstructionClass::IntBitwise
                    | InstructionClass::IntShift
                    | InstructionClass::IntCompare => int_ops += 1,

                    InstructionClass::TensorMmaFp16
                    | InstructionClass::TensorMmaBf16
                    | InstructionClass::TensorMmaFp8
                    | InstructionClass::TensorMmaFp4 => {
                        // Tensor core ops do MxNxK operations
                        tensor_flops += 256; // Approximate for 16x16x16 MMA
                    }
                    InstructionClass::TensorMmaInt8 => tensor_flops += 512, // INT8 has higher throughput

                    InstructionClass::Dp4a => int_ops += 8, // 4 multiplies + 3 adds + 1 accumulate

                    _ => {}
                }
            }
        }

        let total_flops = fp32_flops + fp16_flops + fp64_flops + tensor_flops;

        FlopsCount {
            fp32_flops,
            fp16_flops,
            fp64_flops,
            int_ops,
            tensor_flops,
            total_flops,
        }
    }

    /// Count memory traffic for a kernel
    pub fn count_memory_bytes(&self, kernel: &GpuKernel) -> MemoryTraffic {
        let mut global_loads = 0u64;
        let mut global_stores = 0u64;
        let mut shared_loads = 0u64;
        let mut shared_stores = 0u64;

        for block in &kernel.blocks {
            for (_, op) in &block.instructions {
                match op {
                    GpuOp::Load(_, space) => {
                        let bytes = 4u64; // Assume 4 bytes per load (no type info available)
                        match space {
                            MemorySpace::Global => global_loads += bytes,
                            MemorySpace::Shared => shared_loads += bytes,
                            _ => global_loads += bytes,
                        }
                    }
                    GpuOp::Store(_, _, space) => {
                        let bytes = 4u64; // Assume 4 bytes per store
                        match space {
                            MemorySpace::Global => global_stores += bytes,
                            MemorySpace::Shared => shared_stores += bytes,
                            _ => global_stores += bytes,
                        }
                    }
                    _ => {}
                }
            }
        }

        let total_bytes = global_loads + global_stores + shared_loads + shared_stores;

        MemoryTraffic {
            global_loads,
            global_stores,
            shared_loads,
            shared_stores,
            total_bytes,
        }
    }
}

// ============================================================================
// Cost Estimate Types
// ============================================================================

/// What resource is limiting kernel performance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitingResource {
    Compute,
    MemoryBandwidth,
    MemoryLatency,
    Synchronization,
    Occupancy,
}

/// Kernel-level cost estimate
#[derive(Debug, Clone)]
pub struct KernelCostEstimate {
    /// Total estimated cycles
    pub total_cycles: u64,
    /// Cycles spent on compute
    pub compute_cycles: u64,
    /// Cycles spent waiting for memory
    pub memory_cycles: u64,
    /// Cycles spent on synchronization
    pub sync_cycles: u64,
    /// What resource is limiting performance
    pub limiting_resource: LimitingResource,
    /// Total number of operations
    pub total_ops: u64,
}

/// FLOPS breakdown by precision
#[derive(Debug, Clone, Default)]
pub struct FlopsCount {
    /// FP32 floating-point operations
    pub fp32_flops: u64,
    /// FP16 floating-point operations
    pub fp16_flops: u64,
    /// FP64 floating-point operations
    pub fp64_flops: u64,
    /// Integer operations
    pub int_ops: u64,
    /// Tensor core operations (counted as equivalent FLOPS)
    pub tensor_flops: u64,
    /// Total floating-point operations
    pub total_flops: u64,
}

impl FlopsCount {
    /// Get total FLOPS including all precisions
    pub fn total(&self) -> u64 {
        self.fp32_flops + self.fp16_flops + self.fp64_flops + self.tensor_flops
    }

    /// Get effective FP32-equivalent FLOPS (FP64 counts as 2x)
    pub fn fp32_equivalent(&self) -> u64 {
        self.fp32_flops + self.fp16_flops + self.fp64_flops * 2 + self.tensor_flops
    }
}

/// Memory traffic breakdown
#[derive(Debug, Clone, Default)]
pub struct MemoryTraffic {
    /// Bytes loaded from global memory
    pub global_loads: u64,
    /// Bytes stored to global memory
    pub global_stores: u64,
    /// Bytes loaded from shared memory
    pub shared_loads: u64,
    /// Bytes stored to shared memory
    pub shared_stores: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
}

impl MemoryTraffic {
    /// Get total global memory traffic
    pub fn global_traffic(&self) -> u64 {
        self.global_loads + self.global_stores
    }

    /// Get total shared memory traffic
    pub fn shared_traffic(&self) -> u64 {
        self.shared_loads + self.shared_stores
    }

    /// Calculate arithmetic intensity (FLOPS per byte)
    pub fn arithmetic_intensity(&self, flops: &FlopsCount) -> f64 {
        if self.global_traffic() == 0 {
            f64::INFINITY
        } else {
            flops.total() as f64 / self.global_traffic() as f64
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arch_peak_perf() {
        let turing = ArchPeakPerf::turing();
        assert!(turing.fp32_tflops > 10.0);
        assert!(turing.memory_bandwidth_gbs > 500.0);

        let hopper = ArchPeakPerf::hopper();
        assert!(hopper.tensor_fp16_tflops > turing.tensor_fp16_tflops);
    }

    #[test]
    fn test_cost_database_creation() {
        let db = CostDatabase::for_arch(CudaArch::Ampere);
        assert_eq!(db.arch(), CudaArch::Ampere);

        let fp32_add = db.get_cost(InstructionClass::Fp32Add);
        assert_eq!(fp32_add.latency, 4);
        assert!(fp32_add.throughput > 0.0);
    }

    #[test]
    fn test_instruction_costs_vary_by_arch() {
        let turing = CostDatabase::for_arch(CudaArch::Turing);
        let blackwell = CostDatabase::for_arch(CudaArch::Blackwell);

        let turing_fp32 = turing.get_cost(InstructionClass::Fp32Fma);
        let blackwell_fp32 = blackwell.get_cost(InstructionClass::Fp32Fma);

        // Blackwell should have higher throughput
        assert!(blackwell_fp32.throughput > turing_fp32.throughput);
    }

    #[test]
    fn test_memory_vs_compute_latency() {
        let db = CostDatabase::for_arch(CudaArch::Ampere);

        let global_load = db.get_cost(InstructionClass::GlobalLoad);
        let fp32_add = db.get_cost(InstructionClass::Fp32Add);

        // Memory should have much higher latency than compute
        assert!(global_load.latency > fp32_add.latency * 10);
    }

    #[test]
    fn test_sfu_operations() {
        let db = CostDatabase::for_arch(CudaArch::Ada);

        let sin = db.get_cost(InstructionClass::Fp32Sin);
        assert!(sin.uses_sfu);
        assert!(sin.throughput < db.get_cost(InstructionClass::Fp32Add).throughput);
    }

    #[test]
    fn test_tensor_core_operations() {
        let db = CostDatabase::for_arch(CudaArch::Hopper);

        let tensor_mma = db.get_cost(InstructionClass::TensorMmaFp16);
        assert!(tensor_mma.uses_tensor_core);
        assert!(tensor_mma.throughput > 1000.0); // Very high throughput
    }

    #[test]
    fn test_flops_count() {
        let flops = FlopsCount {
            fp32_flops: 100,
            fp16_flops: 200,
            fp64_flops: 50,
            int_ops: 30,
            tensor_flops: 1000,
            total_flops: 1350,
        };

        assert_eq!(flops.total(), 1350);
        assert_eq!(flops.fp32_equivalent(), 100 + 200 + 100 + 1000); // FP64 counts 2x
    }

    #[test]
    fn test_memory_traffic() {
        let traffic = MemoryTraffic {
            global_loads: 1000,
            global_stores: 500,
            shared_loads: 2000,
            shared_stores: 1000,
            total_bytes: 4500,
        };

        assert_eq!(traffic.global_traffic(), 1500);
        assert_eq!(traffic.shared_traffic(), 3000);

        let flops = FlopsCount {
            fp32_flops: 15000, // Use fp32_flops so total() returns 15000
            ..Default::default()
        };
        let intensity = traffic.arithmetic_intensity(&flops);
        assert!((intensity - 10.0).abs() < 0.01); // 15000 / 1500 = 10
    }
}
