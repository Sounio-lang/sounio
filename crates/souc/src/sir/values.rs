//! SIR Values
//!
//! Values in SIR are SSA (Static Single Assignment) - each value is defined
//! exactly once. Values carry both their type and optional domain metadata.

use super::types::{DistributionKind, SirType};
use std::fmt;

/// Unique identifier for a value in SSA form
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl ValueId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Unique identifier for a basic block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Unique identifier for a function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

impl fmt::Display for FuncId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@func{}", self.0)
    }
}

/// Constant values
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    /// Boolean constant
    Bool(bool),
    /// Integer constants
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    /// Float constants
    F32(f32),
    F64(f64),
    /// Null pointer
    NullPtr,
    /// Undefined value (for uninitialized memory)
    Undef(SirType),
    /// Zero-initialized aggregate
    ZeroInit(SirType),
    /// Array/vector of constants
    Aggregate(Vec<Constant>),

    // === Domain-Specific Constants ===
    /// Epistemic constant with known confidence
    Epistemic {
        value: Box<Constant>,
        confidence: f64,
    },
    /// Known distribution parameters
    Distribution {
        kind: DistributionKind,
        params: Vec<f64>,
    },
}

impl Constant {
    pub fn ty(&self) -> SirType {
        match self {
            Constant::Bool(_) => SirType::bool(),
            Constant::I8(_) => SirType::Scalar(super::types::ScalarType::I8),
            Constant::I16(_) => SirType::Scalar(super::types::ScalarType::I16),
            Constant::I32(_) => SirType::i32(),
            Constant::I64(_) => SirType::i64(),
            Constant::F32(_) => SirType::f32(),
            Constant::F64(_) => SirType::f64(),
            Constant::NullPtr => SirType::ptr(SirType::Void),
            Constant::Undef(ty) | Constant::ZeroInit(ty) => ty.clone(),
            Constant::Aggregate(elems) => {
                if elems.is_empty() {
                    SirType::Array(super::types::ArrayType::new(SirType::Void, 0))
                } else {
                    let elem_ty = elems[0].ty();
                    SirType::Array(super::types::ArrayType::new(elem_ty, elems.len()))
                }
            }
            Constant::Epistemic { value, .. } => SirType::Epistemic(Box::new(value.ty())),
            Constant::Distribution { kind, .. } => SirType::Distribution(*kind),
        }
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Bool(b) => write!(f, "{}", b),
            Constant::I8(v) => write!(f, "{}i8", v),
            Constant::I16(v) => write!(f, "{}i16", v),
            Constant::I32(v) => write!(f, "{}i32", v),
            Constant::I64(v) => write!(f, "{}i64", v),
            Constant::F32(v) => write!(f, "{}f32", v),
            Constant::F64(v) => write!(f, "{}f64", v),
            Constant::NullPtr => write!(f, "null"),
            Constant::Undef(ty) => write!(f, "undef {}", ty),
            Constant::ZeroInit(ty) => write!(f, "zeroinit {}", ty),
            Constant::Aggregate(elems) => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Constant::Epistemic { value, confidence } => {
                write!(f, "epistemic({}, conf={})", value, confidence)
            }
            Constant::Distribution { kind, params } => {
                write!(f, "dist::{:?}({:?})", kind, params)
            }
        }
    }
}

/// Domain-specific metadata attached to values
#[derive(Debug, Clone, Default)]
pub struct ValueMetadata {
    /// Known confidence bounds [min, max]
    pub confidence_bounds: Option<(f64, f64)>,

    /// Provenance chain (for debugging/auditing)
    pub provenance: Option<ProvenanceInfo>,

    /// Known value bounds (for refinement type optimization)
    pub value_bounds: Option<ValueBounds>,

    /// Physical unit (for dimensional analysis optimization)
    pub unit: Option<PhysicalUnit>,

    /// Loop-invariant flag
    pub loop_invariant: bool,

    /// Pure computation (no side effects)
    pub pure: bool,

    /// Vectorization hint
    pub vectorizable: bool,
}

/// Provenance tracking for epistemic values
#[derive(Debug, Clone)]
pub struct ProvenanceInfo {
    /// Source of the data
    pub source: ProvenanceSource,
    /// Timestamp or version
    pub timestamp: Option<u64>,
    /// Transformation chain
    pub transforms: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ProvenanceSource {
    /// User input
    Input(String),
    /// Loaded from file/database
    File(String),
    /// Computed from other values
    Computed,
    /// External API
    External(String),
    /// Literature reference
    Literature { doi: String },
}

/// Value bounds for refinement type optimization
#[derive(Debug, Clone)]
pub struct ValueBounds {
    /// Lower bound (if known)
    pub min: Option<f64>,
    /// Upper bound (if known)
    pub max: Option<f64>,
    /// Is the value guaranteed non-negative?
    pub non_negative: bool,
    /// Is the value guaranteed non-zero?
    pub non_zero: bool,
    /// Is the value guaranteed to be an integer (even if f64)?
    pub integral: bool,
}

impl ValueBounds {
    pub fn positive() -> Self {
        Self {
            min: Some(0.0),
            max: None,
            non_negative: true,
            non_zero: true,
            integral: false,
        }
    }

    pub fn non_negative() -> Self {
        Self {
            min: Some(0.0),
            max: None,
            non_negative: true,
            non_zero: false,
            integral: false,
        }
    }

    pub fn unit_interval() -> Self {
        Self {
            min: Some(0.0),
            max: Some(1.0),
            non_negative: true,
            non_zero: false,
            integral: false,
        }
    }

    pub fn range(min: f64, max: f64) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
            non_negative: min >= 0.0,
            non_zero: min > 0.0 || max < 0.0,
            integral: false,
        }
    }
}

/// Physical units for dimensional analysis
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalUnit {
    /// SI base unit exponents: [m, kg, s, A, K, mol, cd]
    /// e.g., velocity m/s = [1, 0, -1, 0, 0, 0, 0]
    pub dimensions: [i8; 7],
    /// Scale factor (e.g., 0.001 for mm)
    pub scale: f64,
}

impl PhysicalUnit {
    /// Dimensionless
    pub fn dimensionless() -> Self {
        Self {
            dimensions: [0; 7],
            scale: 1.0,
        }
    }

    /// Common units
    pub fn meter() -> Self {
        Self {
            dimensions: [1, 0, 0, 0, 0, 0, 0],
            scale: 1.0,
        }
    }

    pub fn kilogram() -> Self {
        Self {
            dimensions: [0, 1, 0, 0, 0, 0, 0],
            scale: 1.0,
        }
    }

    pub fn second() -> Self {
        Self {
            dimensions: [0, 0, 1, 0, 0, 0, 0],
            scale: 1.0,
        }
    }

    /// Pharmacokinetic units
    pub fn milligram() -> Self {
        Self {
            dimensions: [0, 1, 0, 0, 0, 0, 0],
            scale: 1e-6, // kg
        }
    }

    pub fn liter() -> Self {
        Self {
            dimensions: [3, 0, 0, 0, 0, 0, 0],
            scale: 0.001, // m³
        }
    }

    pub fn hour() -> Self {
        Self {
            dimensions: [0, 0, 1, 0, 0, 0, 0],
            scale: 3600.0, // seconds
        }
    }

    /// Multiply units
    pub fn multiply(&self, other: &PhysicalUnit) -> PhysicalUnit {
        let mut dims = [0i8; 7];
        for i in 0..7 {
            dims[i] = self.dimensions[i] + other.dimensions[i];
        }
        PhysicalUnit {
            dimensions: dims,
            scale: self.scale * other.scale,
        }
    }

    /// Divide units
    pub fn divide(&self, other: &PhysicalUnit) -> PhysicalUnit {
        let mut dims = [0i8; 7];
        for i in 0..7 {
            dims[i] = self.dimensions[i] - other.dimensions[i];
        }
        PhysicalUnit {
            dimensions: dims,
            scale: self.scale / other.scale,
        }
    }
}

/// A value definition in SIR
#[derive(Debug, Clone)]
pub struct Value {
    /// Unique identifier
    pub id: ValueId,
    /// Type of the value
    pub ty: SirType,
    /// Optional name (for debugging)
    pub name: Option<String>,
    /// Domain-specific metadata
    pub metadata: ValueMetadata,
}

impl Value {
    pub fn new(id: ValueId, ty: SirType) -> Self {
        Self {
            id,
            ty,
            name: None,
            metadata: ValueMetadata::default(),
        }
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_metadata(mut self, metadata: ValueMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}
