//! SIR Type System
//!
//! SIR uses machine-level types with domain-specific extensions.
//! Unlike HLIR types which mirror Sounio source types, SIR types
//! directly correspond to machine representations.
//!
//! # Type Categories
//!
//! 1. **Scalar Types**: i8, i16, i32, i64, f32, f64, bool
//! 2. **Pointer Types**: Explicit memory addresses with aliasing info
//! 3. **Vector Types**: SIMD vectors for parallel operations
//! 4. **Aggregate Types**: Structs and arrays with known layout
//! 5. **Domain Types**: Epistemic pairs, distributions, units (metadata)

use std::fmt;

/// Machine-level scalar types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// Boolean (1 bit, stored as i8)
    Bool,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 32-bit IEEE 754 float
    F32,
    /// 64-bit IEEE 754 float
    F64,
}

impl ScalarType {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            ScalarType::Bool | ScalarType::I8 => 1,
            ScalarType::I16 => 2,
            ScalarType::I32 | ScalarType::F32 => 4,
            ScalarType::I64 | ScalarType::F64 => 8,
        }
    }

    /// Alignment in bytes
    pub fn align_bytes(&self) -> usize {
        self.size_bytes()
    }

    /// Is this a floating point type?
    pub fn is_float(&self) -> bool {
        matches!(self, ScalarType::F32 | ScalarType::F64)
    }

    /// Is this an integer type?
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            ScalarType::Bool | ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        )
    }
}

/// SIMD vector types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorType {
    /// Element type
    pub elem: ScalarType,
    /// Number of lanes (must be power of 2: 2, 4, 8, 16, etc.)
    pub lanes: u8,
}

impl VectorType {
    pub fn new(elem: ScalarType, lanes: u8) -> Self {
        debug_assert!(lanes.is_power_of_two(), "Vector lanes must be power of 2");
        Self { elem, lanes }
    }

    /// Common SIMD types
    pub fn f64x2() -> Self {
        Self::new(ScalarType::F64, 2)
    }
    pub fn f64x4() -> Self {
        Self::new(ScalarType::F64, 4)
    }
    pub fn f32x4() -> Self {
        Self::new(ScalarType::F32, 4)
    }
    pub fn f32x8() -> Self {
        Self::new(ScalarType::F32, 8)
    }
    /// 16-lane f32 for sedenions (512 bits, AVX-512 or 2x AVX-256)
    pub fn f32x16() -> Self {
        Self::new(ScalarType::F32, 16)
    }
    pub fn i32x4() -> Self {
        Self::new(ScalarType::I32, 4)
    }
    pub fn i64x4() -> Self {
        Self::new(ScalarType::I64, 4)
    }

    pub fn size_bytes(&self) -> usize {
        self.elem.size_bytes() * self.lanes as usize
    }
}

/// Pointer aliasing information for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AliasInfo {
    /// Unknown aliasing (conservative)
    #[default]
    Unknown,
    /// Pointer does not alias with any other pointer (like C's restrict)
    NoAlias,
    /// Pointer is derived from a specific allocation
    Allocation(u32),
    /// Pointer is a function argument with specific index
    Argument(u16),
}

/// Pointer type with aliasing metadata
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointerType {
    /// What this pointer points to
    pub pointee: Box<SirType>,
    /// Address space (0 = generic, 1 = stack, 2 = heap, 3 = shared GPU)
    pub address_space: u8,
    /// Aliasing information for optimization
    pub alias: AliasInfo,
    /// Is this a mutable pointer?
    pub mutable: bool,
}

impl PointerType {
    pub fn new(pointee: SirType) -> Self {
        Self {
            pointee: Box::new(pointee),
            address_space: 0,
            alias: AliasInfo::Unknown,
            mutable: true,
        }
    }

    pub fn immutable(pointee: SirType) -> Self {
        Self {
            pointee: Box::new(pointee),
            address_space: 0,
            alias: AliasInfo::Unknown,
            mutable: false,
        }
    }

    pub fn no_alias(mut self) -> Self {
        self.alias = AliasInfo::NoAlias;
        self
    }

    pub fn stack(mut self) -> Self {
        self.address_space = 1;
        self
    }

    pub fn heap(mut self) -> Self {
        self.address_space = 2;
        self
    }
}

/// Struct field with offset
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField {
    pub name: Option<String>,
    pub ty: SirType,
    pub offset: usize,
}

/// Struct type with explicit layout
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructType {
    pub name: Option<String>,
    pub fields: Vec<StructField>,
    pub size: usize,
    pub align: usize,
    /// Is this struct packed (no padding)?
    pub packed: bool,
}

impl StructType {
    pub fn new(fields: Vec<(Option<String>, SirType)>) -> Self {
        let mut offset = 0;
        let mut max_align = 1;
        let mut struct_fields = Vec::with_capacity(fields.len());

        for (name, ty) in fields {
            let align = ty.align_bytes();
            max_align = max_align.max(align);

            // Align offset
            offset = (offset + align - 1) & !(align - 1);

            struct_fields.push(StructField {
                name,
                ty: ty.clone(),
                offset,
            });

            offset += ty.size_bytes();
        }

        // Final size aligned to struct alignment
        let size = (offset + max_align - 1) & !(max_align - 1);

        Self {
            name: None,
            fields: struct_fields,
            size,
            align: max_align,
            packed: false,
        }
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Array type with known size
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayType {
    pub elem: Box<SirType>,
    pub len: usize,
}

impl ArrayType {
    pub fn new(elem: SirType, len: usize) -> Self {
        Self {
            elem: Box::new(elem),
            len,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.elem.size_bytes() * self.len
    }
}

/// Function type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    pub params: Vec<SirType>,
    pub ret: Box<SirType>,
    /// Calling convention
    pub cc: CallingConv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CallingConv {
    /// Platform default (System V AMD64 on Linux, Win64 on Windows)
    #[default]
    Default,
    /// C calling convention
    C,
    /// Fast calling convention (more registers)
    Fast,
    /// Tail call optimized
    Tail,
    /// Vector calling convention
    Vector,
}

/// The main SIR type enum
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SirType {
    /// Void type (for functions returning nothing)
    Void,
    /// Scalar types (integers, floats, bool)
    Scalar(ScalarType),
    /// SIMD vector types
    Vector(VectorType),
    /// Pointer types
    Pointer(PointerType),
    /// Struct types
    Struct(StructType),
    /// Array types
    Array(ArrayType),
    /// Function types
    Function(FunctionType),

    // === Domain-Specific Types ===
    // These preserve Sounio semantics for optimization
    /// Epistemic pair: (value, confidence)
    /// Represented as a struct but tagged for special optimization
    Epistemic(Box<SirType>),

    /// Distribution handle for probability operations
    /// Allows distribution-specific optimizations
    Distribution(DistributionKind),
}

/// Known probability distributions for specialized codegen
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionKind {
    /// Normal(mean, stddev)
    Normal,
    /// Uniform(low, high)
    Uniform,
    /// LogNormal(mu, sigma)
    LogNormal,
    /// Exponential(lambda)
    Exponential,
    /// Beta(alpha, beta)
    Beta,
    /// Gamma(shape, scale)
    Gamma,
    /// Poisson(lambda)
    Poisson,
    /// Generic/unknown distribution
    Generic,
}

impl SirType {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            SirType::Void => 0,
            SirType::Scalar(s) => s.size_bytes(),
            SirType::Vector(v) => v.size_bytes(),
            SirType::Pointer(_) => 8, // 64-bit pointers
            SirType::Struct(s) => s.size,
            SirType::Array(a) => a.size_bytes(),
            SirType::Function(_) => 8, // Function pointer
            SirType::Epistemic(inner) => inner.size_bytes() + 8, // value + f64 confidence
            SirType::Distribution(_) => 24, // Distribution state (type + 2 params)
        }
    }

    /// Alignment in bytes
    pub fn align_bytes(&self) -> usize {
        match self {
            SirType::Void => 1,
            SirType::Scalar(s) => s.align_bytes(),
            SirType::Vector(v) => v.size_bytes().min(32), // Max 32-byte alignment for AVX
            SirType::Pointer(_) => 8,
            SirType::Struct(s) => s.align,
            SirType::Array(a) => a.elem.align_bytes(),
            SirType::Function(_) => 8,
            SirType::Epistemic(inner) => inner.align_bytes().max(8),
            SirType::Distribution(_) => 8,
        }
    }

    /// Helper constructors
    pub fn i32() -> Self {
        SirType::Scalar(ScalarType::I32)
    }
    pub fn i64() -> Self {
        SirType::Scalar(ScalarType::I64)
    }
    pub fn f32() -> Self {
        SirType::Scalar(ScalarType::F32)
    }
    pub fn f64() -> Self {
        SirType::Scalar(ScalarType::F64)
    }
    pub fn bool() -> Self {
        SirType::Scalar(ScalarType::Bool)
    }
    pub fn ptr(pointee: SirType) -> Self {
        SirType::Pointer(PointerType::new(pointee))
    }

    /// Construct a Knowledge type with appropriate struct layout based on mode
    pub fn knowledge(inner: SirType, mode: crate::runtime::EpistemicMode) -> Self {
        use crate::runtime::EpistemicMode;

        match mode {
            EpistemicMode::Full => {
                // Full: { value: T, confidence: f64, lower: f64, upper: f64, provenance_id: u32, flags: u32, _pad: [u8; 16] }
                let fields = vec![
                    (Some("value".to_string()), inner),
                    (Some("confidence".to_string()), SirType::f64()),
                    (Some("lower_bound".to_string()), SirType::f64()),
                    (Some("upper_bound".to_string()), SirType::f64()),
                    (Some("provenance_id".to_string()), SirType::i32()),
                    (Some("flags".to_string()), SirType::i32()),
                    // Padding to reach 64 bytes total (exact size depends on T)
                ];
                SirType::Struct(StructType::new(fields).named("Knowledge_Full"))
            }
            EpistemicMode::Compact => {
                // Compact: { value: T, confidence: f64 }
                let fields = vec![
                    (Some("value".to_string()), inner),
                    (Some("confidence".to_string()), SirType::f64()),
                ];
                SirType::Struct(StructType::new(fields).named("Knowledge_Compact"))
            }
            EpistemicMode::Erased => {
                // Erased: just the value, no metadata
                inner
            }
        }
    }

    /// Is this a floating point scalar?
    pub fn is_float(&self) -> bool {
        matches!(self, SirType::Scalar(s) if s.is_float())
    }

    /// Is this an integer scalar?
    pub fn is_integer(&self) -> bool {
        matches!(self, SirType::Scalar(s) if s.is_integer())
    }
}

impl fmt::Display for SirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SirType::Void => write!(f, "void"),
            SirType::Scalar(s) => write!(f, "{:?}", s),
            SirType::Vector(v) => write!(f, "<{} x {:?}>", v.lanes, v.elem),
            SirType::Pointer(p) => {
                if p.mutable {
                    write!(f, "*mut {}", p.pointee)
                } else {
                    write!(f, "*{}", p.pointee)
                }
            }
            SirType::Struct(s) => {
                if let Some(name) = &s.name {
                    write!(f, "%{}", name)
                } else {
                    write!(f, "{{...}}")
                }
            }
            SirType::Array(a) => write!(f, "[{} x {}]", a.len, a.elem),
            SirType::Function(func) => {
                write!(f, "fn(")?;
                for (i, p) in func.params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", func.ret)
            }
            SirType::Epistemic(inner) => write!(f, "epistemic<{}>", inner),
            SirType::Distribution(kind) => write!(f, "dist<{:?}>", kind),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_sizes() {
        assert_eq!(ScalarType::I32.size_bytes(), 4);
        assert_eq!(ScalarType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_struct_layout() {
        // Struct { i32, f64 } should have padding
        let s = StructType::new(vec![
            (Some("x".into()), SirType::i32()),
            (Some("y".into()), SirType::f64()),
        ]);

        assert_eq!(s.fields[0].offset, 0); // i32 at 0
        assert_eq!(s.fields[1].offset, 8); // f64 at 8 (aligned)
        assert_eq!(s.size, 16); // Total size
        assert_eq!(s.align, 8); // Alignment from f64
    }

    #[test]
    fn test_vector_types() {
        let v = VectorType::f64x4();
        assert_eq!(v.size_bytes(), 32); // 4 * 8 bytes
    }
}

#[test]
fn test_knowledge_full_mode() {
    use crate::runtime::EpistemicMode;

    let inner = SirType::f64();
    let knowledge_ty = SirType::knowledge(inner, EpistemicMode::Full);

    // Full mode should create a struct with value + metadata fields
    match knowledge_ty {
        SirType::Struct(st) => {
            assert!(
                st.name
                    .as_ref()
                    .map_or(false, |n| n.contains("Knowledge_Full"))
            );
            assert!(st.fields.len() >= 6); // value, confidence, lower, upper, provenance, flags
        }
        _ => panic!("Expected Struct type for Full mode"),
    }
}

#[test]
fn test_knowledge_compact_mode() {
    use crate::runtime::EpistemicMode;

    let inner = SirType::i32();
    let knowledge_ty = SirType::knowledge(inner, EpistemicMode::Compact);

    // Compact mode should create a struct with value + confidence
    match knowledge_ty {
        SirType::Struct(st) => {
            assert!(
                st.name
                    .as_ref()
                    .map_or(false, |n| n.contains("Knowledge_Compact"))
            );
            assert_eq!(st.fields.len(), 2); // value, confidence
        }
        _ => panic!("Expected Struct type for Compact mode"),
    }
}

#[test]
fn test_knowledge_erased_mode() {
    use crate::runtime::EpistemicMode;

    let inner = SirType::f32();
    let knowledge_ty = SirType::knowledge(inner.clone(), EpistemicMode::Erased);

    // Erased mode should return just the inner type
    assert_eq!(knowledge_ty, inner);
}
