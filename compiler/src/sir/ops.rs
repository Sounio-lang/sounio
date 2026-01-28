//! SIR Operations
//!
//! Operations in SIR are designed to be:
//! 1. Low-level enough for efficient code generation
//! 2. High-level enough to preserve domain semantics for optimization
//!
//! # Operation Categories
//!
//! - **Arithmetic**: Standard math operations with overflow semantics
//! - **Memory**: Load, store, alloc with aliasing info
//! - **Control**: Branch, call, return
//! - **Vector**: SIMD operations
//! - **Epistemic**: Confidence propagation (domain-specific)
//! - **Probability**: Distribution sampling (domain-specific)
//! - **Scientific**: ODE steps, matrix ops (domain-specific)

use super::types::{CallingConv, DistributionKind, SirType};
use super::values::{BlockId, Constant, FuncId, ValueId};
use std::fmt;

/// Arithmetic binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    // Integer operations
    Add,
    Sub,
    Mul,
    SDiv, // Signed division
    UDiv, // Unsigned division
    SRem, // Signed remainder
    URem, // Unsigned remainder

    // Floating point operations
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem, // IEEE 754 remainder

    // Bitwise operations
    And,
    Or,
    Xor,
    Shl,  // Shift left
    LShr, // Logical shift right
    AShr, // Arithmetic shift right
}

/// Comparison operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    // Integer comparisons
    Eq,
    Ne,
    SLt, // Signed less than
    SLe, // Signed less or equal
    SGt, // Signed greater than
    SGe, // Signed greater or equal
    ULt, // Unsigned less than
    ULe,
    UGt,
    UGe,

    // Floating point comparisons (IEEE 754 ordered)
    FOEq, // Ordered equal
    FONe, // Ordered not equal
    FOLt,
    FOLe,
    FOGt,
    FOGe,

    // Floating point unordered (NaN-aware)
    FUEq,
    FUNe,
    FULt,
    FULe,
    FUGt,
    FUGe,
}

/// Cast/conversion operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastOp {
    // Integer casts
    Trunc, // Truncate to smaller type
    ZExt,  // Zero extend to larger type
    SExt,  // Sign extend to larger type

    // Float casts
    FPTrunc, // Float to smaller float
    FPExt,   // Float to larger float

    // Int <-> Float
    FPToSI, // Float to signed int
    FPToUI, // Float to unsigned int
    SIToFP, // Signed int to float
    UIToFP, // Unsigned int to float

    // Pointer casts
    PtrToInt,
    IntToPtr,
    BitCast, // Reinterpret bits
}

/// Unary integer operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryIntOp {
    Neg,
    Not,
}

/// Unary floating point operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFloatOp {
    Neg,
    Abs,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Exp,
    Exp2,
    Log,
    Log2,
    Log10,
    Floor,
    Ceil,
    Round,
    Trunc,
}

/// Binary floating point operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFloatOp {
    Pow,
    Min,
    Max,
    CopySign,
    Atan2,
}

/// SIMD/Vector operations
#[derive(Debug, Clone, PartialEq)]
pub enum VectorOp {
    /// Create vector from scalars
    Splat(ValueId),
    /// Extract element at index
    Extract { vec: ValueId, idx: u8 },
    /// Insert element at index
    Insert { vec: ValueId, val: ValueId, idx: u8 },
    /// Shuffle elements
    Shuffle {
        v1: ValueId,
        v2: ValueId,
        mask: Vec<u8>,
    },
    /// Horizontal add (sum all elements)
    HAdd(ValueId),
    /// Horizontal multiply (product of all elements)
    HMul(ValueId),
    /// Fused multiply-add: a * b + c
    Fma { a: ValueId, b: ValueId, c: ValueId },
    /// Reduction (apply binary op across all lanes)
    Reduce { op: ArithOp, vec: ValueId },
}

/// Memory operations
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryOp {
    /// Stack allocation
    Alloca {
        ty: SirType,
        count: Option<ValueId>, // For dynamic arrays
        align: Option<u32>,
    },
    /// Load from memory
    Load {
        ptr: ValueId,
        ty: SirType,
        volatile: bool,
        align: Option<u32>,
    },
    /// Store to memory
    Store {
        ptr: ValueId,
        val: ValueId,
        volatile: bool,
        align: Option<u32>,
    },
    /// Get element pointer (for struct/array access)
    GetElementPtr {
        ptr: ValueId,
        ty: SirType,
        indices: Vec<GepIndex>,
    },
    /// Memory copy
    Memcpy {
        dst: ValueId,
        src: ValueId,
        len: ValueId,
        volatile: bool,
    },
    /// Memory set
    Memset {
        dst: ValueId,
        val: ValueId,
        len: ValueId,
        volatile: bool,
    },
}

/// GEP index - can be constant or dynamic
#[derive(Debug, Clone, PartialEq)]
pub enum GepIndex {
    Const(i64),
    Value(ValueId),
}

/// Function call information
#[derive(Debug, Clone, PartialEq)]
pub struct CallInfo {
    pub callee: Callee,
    pub args: Vec<ValueId>,
    pub ret_ty: SirType,
    pub cc: CallingConv,
    /// Is this a tail call?
    pub tail: bool,
    /// Is this call known to not throw?
    pub nounwind: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Callee {
    /// Direct call to known function
    Direct(FuncId),
    /// Call through function name (external)
    Named(String),
    /// Indirect call through pointer
    Indirect(ValueId),
}

// ============================================================================
// DOMAIN-SPECIFIC OPERATIONS
// ============================================================================

/// Epistemic operations for confidence propagation
#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicOp {
    /// Create epistemic value from value + confidence
    Create {
        value: ValueId,
        confidence: ValueId,
    },
    /// Extract the raw value
    ExtractValue(ValueId),
    /// Extract the confidence
    ExtractConfidence(ValueId),

    /// Propagate confidence through arithmetic
    /// This is the key optimization target!
    PropagateAdd {
        conf_a: ValueId,
        conf_b: ValueId,
    },
    PropagateSub {
        conf_a: ValueId,
        conf_b: ValueId,
    },
    PropagateMul {
        val_a: ValueId,
        conf_a: ValueId,
        val_b: ValueId,
        conf_b: ValueId,
    },
    PropagateDiv {
        val_a: ValueId,
        conf_a: ValueId,
        val_b: ValueId,
        conf_b: ValueId,
    },

    /// Fused epistemic arithmetic (optimized)
    /// Computes both value and confidence in one operation
    FusedMul {
        val_a: ValueId,
        conf_a: ValueId,
        val_b: ValueId,
        conf_b: ValueId,
    },

    /// Meet operation (minimum confidence)
    Meet {
        conf_a: ValueId,
        conf_b: ValueId,
    },
    /// Join operation (for branches)
    Join {
        conf_a: ValueId,
        conf_b: ValueId,
    },

    /// Conditional confidence update
    /// If condition is certain (conf=1), propagate full confidence
    CondPropagate {
        condition_conf: ValueId,
        then_conf: ValueId,
        else_conf: ValueId,
    },
}

/// Probability distribution operations
#[derive(Debug, Clone, PartialEq)]
pub enum ProbOp {
    /// Create a distribution
    CreateDist {
        kind: DistributionKind,
        params: Vec<ValueId>,
    },
    /// Sample from distribution
    Sample {
        dist: ValueId,
        /// RNG state pointer
        rng: ValueId,
    },
    /// Vectorized sampling (for population simulations)
    SampleN {
        dist: ValueId,
        count: ValueId,
        rng: ValueId,
    },
    /// PDF evaluation
    Pdf { dist: ValueId, point: ValueId },
    /// Log-PDF (more numerically stable)
    LogPdf { dist: ValueId, point: ValueId },
    /// CDF evaluation
    Cdf { dist: ValueId, point: ValueId },
    /// Quantile function (inverse CDF)
    Quantile { dist: ValueId, prob: ValueId },
    /// Analytical distribution combination (e.g., Normal + Normal = Normal)
    Combine {
        op: DistCombineOp,
        dist_a: ValueId,
        dist_b: ValueId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistCombineOp {
    /// Sum of independent RVs
    Add,
    /// Scaling
    Scale,
    /// Mixture
    Mixture,
}

/// Scientific computing operations
#[derive(Debug, Clone, PartialEq)]
pub enum ScientificOp {
    /// Single ODE solver step (domain-aware fusion target)
    OdeStep {
        method: OdeMethod,
        state: ValueId,      // Current state vector
        derivatives: FuncId, // Derivative function
        t: ValueId,          // Current time
        dt: ValueId,         // Time step
    },

    /// Matrix-vector multiply (optimized for small matrices)
    MatVecMul {
        matrix: ValueId,
        vector: ValueId,
        rows: u32,
        cols: u32,
    },

    /// Dot product (with optional epistemic tracking)
    DotProduct {
        a: ValueId,
        b: ValueId,
        len: u32,
        epistemic: bool,
    },

    /// Linear interpolation with bounds
    Lerp {
        a: ValueId,
        b: ValueId,
        t: ValueId, // Must be in [0, 1] - refinement type info
    },

    /// Compartment model step (PBPK-specific)
    CompartmentStep {
        concentrations: ValueId,
        flows: ValueId,
        volumes: ValueId,
        dt: ValueId,
    },

    /// Automatic differentiation operation
    Autodiff {
        mode: AutodiffMode,
        function: FuncId,      // Function to differentiate
        input: ValueId,        // Input value(s)
        output: ValueId,       // Output: gradient(s)
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeMethod {
    /// Euler method (1st order)
    Euler,
    /// Midpoint method (2nd order)
    Midpoint,
    /// Classic Runge-Kutta (4th order)
    RK4,
    /// Dormand-Prince 5(4) with error estimate
    DoPri5,
    /// Cash-Karp 5(4)
    CashKarp,
    /// BDF (Backward Differentiation Formula) for stiff systems
    BDF,
    /// LSODA (automatic stiff/non-stiff switching)
    LSODA,
}

/// Automatic differentiation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutodiffMode {
    /// Forward-mode (dual numbers) - efficient for functions with few inputs
    Forward,
    /// Reverse-mode (tape-based) - efficient for functions with many inputs (R^n -> R)
    Reverse,
}

/// Failure mode for assertion violations
#[derive(Debug, Clone, PartialEq)]
pub enum FailureMode {
    /// Emit trap instruction (ud2 on x86)
    Trap,
    /// Call runtime panic handler
    Panic,
    /// Degrade confidence by a factor (epistemic fallback)
    DegradeConfidence(f64),
}

impl Default for FailureMode {
    fn default() -> Self {
        FailureMode::Panic
    }
}

// ============================================================================
// MAIN INSTRUCTION TYPE
// ============================================================================

/// A single SIR instruction
#[derive(Debug, Clone, PartialEq)]
pub enum SirInst {
    // === Basic Operations ===
    /// Binary arithmetic: result = op(lhs, rhs)
    BinOp {
        op: ArithOp,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Comparison: result = cmp(lhs, rhs)
    Cmp {
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Cast/convert: result = cast(val)
    Cast {
        op: CastOp,
        val: ValueId,
        to_ty: SirType,
    },

    /// Unary float operation
    UnaryFloat {
        op: UnaryFloatOp,
        val: ValueId,
    },

    /// Unary integer operation
    UnaryInt {
        op: UnaryIntOp,
        val: ValueId,
    },

    /// Binary float operation
    BinaryFloat {
        op: BinaryFloatOp,
        lhs: ValueId,
        rhs: ValueId,
    },

    /// Select (ternary): result = cond ? then_val : else_val
    Select {
        cond: ValueId,
        then_val: ValueId,
        else_val: ValueId,
    },

    /// Phi node for SSA
    Phi {
        ty: SirType,
        incoming: Vec<(BlockId, ValueId)>,
    },

    /// Load constant
    Const(Constant),

    // === Memory ===
    Memory(MemoryOp),

    // === Vector/SIMD ===
    Vector(VectorOp),

    // === Function Calls ===
    Call(CallInfo),

    // === Domain-Specific ===
    Epistemic(EpistemicOp),
    Prob(ProbOp),
    Scientific(ScientificOp),

    // === Debugging/Metadata ===
    /// Debug value intrinsic
    DebugValue {
        value: ValueId,
        name: String,
    },

    /// Assertion (for refinement type checks and runtime verification)
    Assert {
        cond: ValueId,
        message: String,
        failure_mode: FailureMode,
    },

    /// Assume (optimization hint)
    Assume(ValueId),

    /// Build an aggregate (struct/tuple) from field values
    /// Used for slice_from_raw_parts and other struct construction
    BuildAggregate {
        /// Field values in order
        fields: Vec<ValueId>,
        /// The aggregate type being constructed
        ty: SirType,
    },
}

impl SirInst {
    /// Does this instruction have side effects?
    pub fn has_side_effects(&self) -> bool {
        match self {
            SirInst::Memory(MemoryOp::Store { .. })
            | SirInst::Memory(MemoryOp::Memcpy { .. })
            | SirInst::Memory(MemoryOp::Memset { .. })
            | SirInst::Call(_)
            | SirInst::DebugValue { .. }
            | SirInst::Assert { .. } => true,
            SirInst::Prob(ProbOp::Sample { .. } | ProbOp::SampleN { .. }) => true,
            _ => false,
        }
    }

    /// Can this instruction trap/throw?
    pub fn can_trap(&self) -> bool {
        match self {
            SirInst::BinOp {
                op: ArithOp::SDiv | ArithOp::UDiv | ArithOp::SRem | ArithOp::URem,
                ..
            } => true,
            SirInst::Memory(MemoryOp::Load { .. } | MemoryOp::Store { .. }) => true,
            SirInst::Assert { .. } => true,
            _ => false,
        }
    }

    /// Get operands (for dataflow analysis)
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            SirInst::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            SirInst::Cmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            SirInst::Cast { val, .. } => vec![*val],
            SirInst::UnaryFloat { val, .. } => vec![*val],
            SirInst::UnaryInt { val, .. } => vec![*val],
            SirInst::BinaryFloat { lhs, rhs, .. } => vec![*lhs, *rhs],
            SirInst::Select {
                cond,
                then_val,
                else_val,
            } => vec![*cond, *then_val, *else_val],
            SirInst::Phi { incoming, .. } => incoming.iter().map(|(_, v)| *v).collect(),
            SirInst::Const(_) => vec![],
            SirInst::Memory(mem) => match mem {
                MemoryOp::Alloca { count, .. } => count.iter().copied().collect(),
                MemoryOp::Load { ptr, .. } => vec![*ptr],
                MemoryOp::Store { ptr, val, .. } => vec![*ptr, *val],
                MemoryOp::GetElementPtr { ptr, indices, .. } => {
                    let mut ops = vec![*ptr];
                    for idx in indices {
                        if let GepIndex::Value(v) = idx {
                            ops.push(*v);
                        }
                    }
                    ops
                }
                MemoryOp::Memcpy { dst, src, len, .. } => vec![*dst, *src, *len],
                MemoryOp::Memset { dst, val, len, .. } => vec![*dst, *val, *len],
            },
            SirInst::Vector(vec_op) => match vec_op {
                VectorOp::Splat(v) => vec![*v],
                VectorOp::Extract { vec, .. } => vec![*vec],
                VectorOp::Insert { vec, val, .. } => vec![*vec, *val],
                VectorOp::Shuffle { v1, v2, .. } => vec![*v1, *v2],
                VectorOp::HAdd(v) | VectorOp::HMul(v) => vec![*v],
                VectorOp::Fma { a, b, c } => vec![*a, *b, *c],
                VectorOp::Reduce { vec, .. } => vec![*vec],
            },
            SirInst::Call(CallInfo { args, callee, .. }) => {
                let mut ops = args.clone();
                if let Callee::Indirect(v) = callee {
                    ops.push(*v);
                }
                ops
            }
            SirInst::Epistemic(ep) => match ep {
                EpistemicOp::Create { value, confidence } => vec![*value, *confidence],
                EpistemicOp::ExtractValue(v) | EpistemicOp::ExtractConfidence(v) => vec![*v],
                EpistemicOp::PropagateAdd { conf_a, conf_b }
                | EpistemicOp::PropagateSub { conf_a, conf_b }
                | EpistemicOp::Meet { conf_a, conf_b }
                | EpistemicOp::Join { conf_a, conf_b } => vec![*conf_a, *conf_b],
                EpistemicOp::PropagateMul {
                    val_a,
                    conf_a,
                    val_b,
                    conf_b,
                }
                | EpistemicOp::PropagateDiv {
                    val_a,
                    conf_a,
                    val_b,
                    conf_b,
                }
                | EpistemicOp::FusedMul {
                    val_a,
                    conf_a,
                    val_b,
                    conf_b,
                } => {
                    vec![*val_a, *conf_a, *val_b, *conf_b]
                }
                EpistemicOp::CondPropagate {
                    condition_conf,
                    then_conf,
                    else_conf,
                } => {
                    vec![*condition_conf, *then_conf, *else_conf]
                }
            },
            SirInst::Prob(prob) => match prob {
                ProbOp::CreateDist { params, .. } => params.clone(),
                ProbOp::Sample { dist, rng } => vec![*dist, *rng],
                ProbOp::SampleN { dist, count, rng } => vec![*dist, *count, *rng],
                ProbOp::Pdf { dist, point }
                | ProbOp::LogPdf { dist, point }
                | ProbOp::Cdf { dist, point }
                | ProbOp::Quantile { dist, prob: point } => vec![*dist, *point],
                ProbOp::Combine { dist_a, dist_b, .. } => vec![*dist_a, *dist_b],
            },
            SirInst::Scientific(sci) => match sci {
                ScientificOp::OdeStep { state, t, dt, .. } => vec![*state, *t, *dt],
                ScientificOp::MatVecMul { matrix, vector, .. } => vec![*matrix, *vector],
                ScientificOp::DotProduct { a, b, .. } => vec![*a, *b],
                ScientificOp::Lerp { a, b, t } => vec![*a, *b, *t],
                ScientificOp::CompartmentStep {
                    concentrations,
                    flows,
                    volumes,
                    dt,
                } => {
                    vec![*concentrations, *flows, *volumes, *dt]
                }
                ScientificOp::Autodiff {
                    function: _,
                    input,
                    output: _,
                    ..
                } => vec![*input],
            },
            SirInst::DebugValue { value, .. } => vec![*value],
            SirInst::Assert { cond, .. } | SirInst::Assume(cond) => vec![*cond],
            SirInst::BuildAggregate { fields, .. } => fields.clone(),
        }
    }
}

impl fmt::Display for SirInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SirInst::BinOp { op, lhs, rhs } => write!(f, "{:?} {}, {}", op, lhs, rhs),
            SirInst::Cmp { op, lhs, rhs } => write!(f, "{:?} {}, {}", op, lhs, rhs),
            SirInst::Cast { op, val, to_ty } => write!(f, "{:?} {} to {}", op, val, to_ty),
            SirInst::UnaryFloat { op, val } => write!(f, "{:?} {}", op, val),
            SirInst::UnaryInt { op, val } => write!(f, "{:?} {}", op, val),
            SirInst::BinaryFloat { op, lhs, rhs } => write!(f, "{:?} {}, {}", op, lhs, rhs),
            SirInst::Select {
                cond,
                then_val,
                else_val,
            } => {
                write!(f, "select {}, {}, {}", cond, then_val, else_val)
            }
            SirInst::Phi { ty, incoming } => {
                write!(f, "phi {} [", ty)?;
                for (i, (blk, val)) in incoming.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", blk, val)?;
                }
                write!(f, "]")
            }
            SirInst::Const(c) => write!(f, "const {}", c),
            SirInst::Memory(mem) => write!(f, "{:?}", mem),
            SirInst::Vector(vec) => write!(f, "{:?}", vec),
            SirInst::Call(info) => write!(f, "call {:?}({:?})", info.callee, info.args),
            SirInst::Epistemic(ep) => write!(f, "epistemic.{:?}", ep),
            SirInst::Prob(prob) => write!(f, "prob.{:?}", prob),
            SirInst::Scientific(sci) => write!(f, "sci.{:?}", sci),
            SirInst::DebugValue { value, name } => write!(f, "debug.value {} = {}", name, value),
            SirInst::Assert { cond, message, failure_mode } => write!(f, "assert {}, \"{}\", mode={:?}", cond, message, failure_mode),
            SirInst::Assume(cond) => write!(f, "assume {}", cond),
            SirInst::BuildAggregate { fields, ty } => {
                write!(f, "build_aggregate {:?} {{ {:?} }}", ty, fields)
            }
        }
    }
}
