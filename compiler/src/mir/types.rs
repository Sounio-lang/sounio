//! MIR Types
//!
//! This module defines the type system for MIR (Mid-level IR).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Value identifier in SSA form
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueId(pub usize);

/// Basic block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(pub usize);

/// Function identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FuncId(pub usize);

/// MIR Type system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MirType {
    /// Unit type ()
    Unit,
    /// Boolean type
    Bool,
    /// Integer types
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    /// Floating point types
    F32,
    F64,
    /// Character type
    Char,
    /// String type (pointer to null-terminated string)
    String,
    /// Pointer type
    Ptr(Box<MirType>),
    /// Array type with fixed size
    Array(Box<MirType>, usize),
    /// Tuple type
    Tuple(Vec<MirType>),
    /// Function type
    Function {
        params: Vec<MirType>,
        return_type: Box<MirType>,
    },
    /// Named struct type - references a type definition by name
    /// Field layout is resolved from the type definition table
    Struct {
        name: std::string::String,
        /// Optional inline field definitions for simple structs
        /// that don't need a separate type definition lookup
        fields: Vec<(std::string::String, MirType)>,
    },
    /// Void type (for functions that don't return)
    Void,
    /// Error type (for type checking failures)
    Error,
}

impl MirType {
    /// Check if this is the unit type
    pub fn is_unit(&self) -> bool {
        matches!(self, MirType::Unit)
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            MirType::I8
                | MirType::I16
                | MirType::I32
                | MirType::I64
                | MirType::I128
                | MirType::Isize
                | MirType::U8
                | MirType::U16
                | MirType::U32
                | MirType::U64
                | MirType::U128
                | MirType::Usize
        )
    }

    /// Check if this is a signed integer type
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            MirType::I8
                | MirType::I16
                | MirType::I32
                | MirType::I64
                | MirType::I128
                | MirType::Isize
        )
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, MirType::F32 | MirType::F64)
    }

    /// Check if this is a pointer type
    pub fn is_pointer(&self) -> bool {
        matches!(self, MirType::Ptr(_))
    }

    /// Get the pointed-to type if this is a pointer
    pub fn pointee_type(&self) -> Option<&MirType> {
        match self {
            MirType::Ptr(inner) => Some(inner),
            _ => None,
        }
    }

    /// Get the size in bytes (if known statically)
    pub fn size_bytes(&self) -> Option<usize> {
        match self {
            MirType::Unit | MirType::Bool | MirType::I8 | MirType::U8 => Some(1),
            MirType::I16 | MirType::U16 => Some(2),
            MirType::I32 | MirType::U32 | MirType::F32 => Some(4),
            MirType::I64 | MirType::U64 | MirType::F64 | MirType::Isize | MirType::Usize => Some(8),
            MirType::I128 | MirType::U128 => Some(16),
            MirType::Char => Some(4),   // Unicode code point
            MirType::Ptr(_) => Some(8), // Pointer size on 64-bit
            MirType::Array(elem_ty, size) => elem_ty.size_bytes().map(|s| s * size),
            MirType::Tuple(types) => types
                .iter()
                .filter_map(|t| t.size_bytes())
                .sum::<usize>()
                .into(),
            MirType::Struct { fields, .. } => {
                // Compute struct size as sum of field sizes (simplified - ignores alignment)
                if fields.is_empty() {
                    None // Need type definition lookup for named structs without inline fields
                } else {
                    Some(fields.iter().filter_map(|(_, t)| t.size_bytes()).sum())
                }
            }
            MirType::Function { .. } | MirType::String | MirType::Void | MirType::Error => None,
        }
    }

    /// Convert from HIR type to MIR type
    pub fn from_hir(hir_ty: &crate::hir::HirType) -> Self {
        use crate::hir::HirType;

        match hir_ty {
            HirType::Unit => MirType::Unit,
            HirType::Bool => MirType::Bool,
            HirType::I8 => MirType::I8,
            HirType::I16 => MirType::I16,
            HirType::I32 => MirType::I32,
            HirType::I64 => MirType::I64,
            HirType::I128 => MirType::I128,
            HirType::Isize => MirType::Isize,
            HirType::U8 => MirType::U8,
            HirType::U16 => MirType::U16,
            HirType::U32 => MirType::U32,
            HirType::U64 => MirType::U64,
            HirType::U128 => MirType::U128,
            HirType::Usize => MirType::Usize,
            HirType::F32 => MirType::F32,
            HirType::F64 => MirType::F64,
            HirType::Char => MirType::Char,
            HirType::String => MirType::String,

            HirType::Ref { inner, .. } => MirType::Ptr(Box::new(MirType::from_hir(inner))),
            HirType::RawPointer { inner, .. } => MirType::Ptr(Box::new(MirType::from_hir(inner))),
            HirType::Array { element, size } => {
                MirType::Array(Box::new(MirType::from_hir(element)), size.unwrap_or(0))
            }
            HirType::Tuple(types) => MirType::Tuple(types.iter().map(MirType::from_hir).collect()),
            HirType::Fn {
                params,
                return_type,
                ..
            } => MirType::Function {
                params: params.iter().map(MirType::from_hir).collect(),
                return_type: Box::new(MirType::from_hir(return_type)),
            },

            // Epistemic types lower to their inner type
            HirType::Knowledge { inner, .. } => MirType::from_hir(inner),
            HirType::Quantity { numeric, .. } => MirType::from_hir(numeric),

            // Named types (structs, enums, type aliases)
            HirType::Named { name, .. } => {
                // Create a struct reference by name
                // Field resolution happens during codegen when type definitions are available
                MirType::Struct {
                    name: name.clone(),
                    fields: Vec::new(), // Fields resolved from type definition table
                }
            }

            HirType::Future { output } => MirType::Ptr(Box::new(MirType::from_hir(output))),

            // Linear algebra types - lower to arrays of floats
            HirType::Vec2 => MirType::Array(Box::new(MirType::F32), 2),
            HirType::Vec3 => MirType::Array(Box::new(MirType::F32), 4), // Padded for SIMD
            HirType::Vec4 => MirType::Array(Box::new(MirType::F32), 4),
            HirType::Mat2 => MirType::Array(Box::new(MirType::F32), 4),
            HirType::Mat3 => MirType::Array(Box::new(MirType::F32), 9),
            HirType::Mat4 => MirType::Array(Box::new(MirType::F32), 16),
            HirType::Quat => MirType::Array(Box::new(MirType::F32), 4),

            // f64 vector types
            HirType::Vec2d => MirType::Array(Box::new(MirType::F64), 2),
            HirType::Vec3d => MirType::Array(Box::new(MirType::F64), 4), // Padded for SIMD
            HirType::Vec4d => MirType::Array(Box::new(MirType::F64), 4),

            // Octonion: 8 x f32
            HirType::Octonion => MirType::Array(Box::new(MirType::F32), 8),
            // Quaternionic Neural Network types - treat as arrays for now
            HirType::QuatLinear { .. } => MirType::Array(Box::new(MirType::F32), 0),
            HirType::QuatConv2d { .. } => MirType::Array(Box::new(MirType::F32), 0),
            HirType::QuatRnnState { .. } => MirType::Array(Box::new(MirType::F32), 0),
            HirType::QuatGate { .. } => MirType::Array(Box::new(MirType::F32), 0),

            // Automatic differentiation
            HirType::Dual => MirType::Tuple(vec![MirType::F64, MirType::F64]),

            // Tensor types - lower to struct representation
            HirType::Tensor { element, dims } => {
                // Tensor<T, N> => struct with data pointer and shape
                let elem_ty = MirType::from_hir(element);
                let ndims = dims.len();
                MirType::Struct {
                    name: format!("Tensor<{}, {}>", elem_ty, ndims),
                    fields: vec![
                        ("data".to_string(), MirType::Ptr(Box::new(elem_ty))),
                        (
                            "shape".to_string(),
                            MirType::Array(Box::new(MirType::Usize), ndims),
                        ),
                        (
                            "strides".to_string(),
                            MirType::Array(Box::new(MirType::Usize), ndims),
                        ),
                    ],
                }
            }

            // Ontology types - lower to struct representation
            HirType::Ontology { namespace, term } => {
                MirType::Struct {
                    name: format!("Ontology<{}:{}>", namespace, term),
                    fields: Vec::new(), // Opaque - fields resolved at runtime
                }
            }

            // Scientific array types - lower to struct representation
            HirType::ScientificArray { element, dim } => {
                let elem_ty = MirType::from_hir(element);
                let size = match dim {
                    crate::hir::HirTensorDim::Fixed(n) => *n,
                    _ => 0, // Dynamic size
                };
                MirType::Struct {
                    name: format!("Array<{}, {}>", elem_ty, size),
                    fields: vec![
                        ("data".to_string(), MirType::Ptr(Box::new(elem_ty))),
                        ("len".to_string(), MirType::Usize),
                    ],
                }
            }

            // Matrix types - lower to struct representation
            HirType::Matrix {
                element,
                rows,
                cols,
            } => {
                let elem_ty = MirType::from_hir(element);
                let r = match rows {
                    crate::hir::HirTensorDim::Fixed(n) => *n,
                    _ => 0,
                };
                let c = match cols {
                    crate::hir::HirTensorDim::Fixed(n) => *n,
                    _ => 0,
                };
                MirType::Struct {
                    name: format!("Matrix<{}, {}, {}>", elem_ty, r, c),
                    fields: vec![
                        ("data".to_string(), MirType::Ptr(Box::new(elem_ty))),
                        ("rows".to_string(), MirType::Usize),
                        ("cols".to_string(), MirType::Usize),
                    ],
                }
            }

            // Type variable (should be resolved by now)
            HirType::Var(_) => MirType::Error,

            HirType::Never | HirType::Error => MirType::Error,
        }
    }
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirType::Unit => write!(f, "()"),
            MirType::Bool => write!(f, "bool"),
            MirType::I8 => write!(f, "i8"),
            MirType::I16 => write!(f, "i16"),
            MirType::I32 => write!(f, "i32"),
            MirType::I64 => write!(f, "i64"),
            MirType::I128 => write!(f, "i128"),
            MirType::Isize => write!(f, "isize"),
            MirType::U8 => write!(f, "u8"),
            MirType::U16 => write!(f, "u16"),
            MirType::U32 => write!(f, "u32"),
            MirType::U64 => write!(f, "u64"),
            MirType::U128 => write!(f, "u128"),
            MirType::Usize => write!(f, "usize"),
            MirType::F32 => write!(f, "f32"),
            MirType::F64 => write!(f, "f64"),
            MirType::Char => write!(f, "char"),
            MirType::String => write!(f, "string"),
            MirType::Ptr(inner) => write!(f, "*{}", inner),
            MirType::Array(elem, size) => write!(f, "[{}; {}]", elem, size),
            MirType::Tuple(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            MirType::Function {
                params,
                return_type,
            } => {
                write!(f, "fn(")?;
                for (i, ty) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ") -> {}", return_type)
            }
            MirType::Struct { name, fields } => {
                if fields.is_empty() {
                    write!(f, "struct {}", name)
                } else {
                    write!(f, "struct {} {{ ", name)?;
                    for (i, (field_name, ty)) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", field_name, ty)?;
                    }
                    write!(f, " }}")
                }
            }
            MirType::Void => write!(f, "!"),
            MirType::Error => write!(f, "error"),
        }
    }
}

/// MIR constant values
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MirConstant {
    /// Unit constant
    Unit,
    /// Boolean constant
    Bool(bool),
    /// Integer constant
    Int(i64),
    /// Unsigned integer constant
    UInt(u64),
    /// Float constant (stored as string for hashing)
    Float(String),
    /// String constant (pointer to string data)
    String(String),
    /// Null pointer
    Null,
    /// Function reference
    FunctionRef(String),
    /// Global variable reference
    GlobalRef(String),
}

impl MirConstant {
    /// Get the type of this constant
    pub fn ty(&self) -> MirType {
        match self {
            MirConstant::Unit => MirType::Unit,
            MirConstant::Bool(_) => MirType::Bool,
            MirConstant::Int(_) => MirType::I64,
            MirConstant::UInt(_) => MirType::U64,
            MirConstant::Float(_) => MirType::F64,
            MirConstant::String(_) => MirType::Ptr(Box::new(MirType::U8)),
            MirConstant::Null => MirType::Ptr(Box::new(MirType::Void)),
            MirConstant::FunctionRef(_) => MirType::Ptr(Box::new(MirType::Void)),
            MirConstant::GlobalRef(_) => MirType::Ptr(Box::new(MirType::Void)),
        }
    }
}

/// Type definitions in MIR
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MirTypeDef {
    Struct {
        name: String,
        fields: Vec<(String, MirType)>,
    },
    Enum {
        name: String,
        variants: Vec<(String, Vec<MirType>)>,
    },
}

impl MirTypeDef {
    /// Get the name of this type definition
    pub fn name(&self) -> &str {
        match self {
            MirTypeDef::Struct { name, .. } => name,
            MirTypeDef::Enum { name, .. } => name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_conversions() {
        use crate::hir::HirType;

        // Test basic type conversion
        let hir_unit = HirType::Unit;
        let mir_unit = MirType::from_hir(&hir_unit);
        assert_eq!(mir_unit, MirType::Unit);
        assert!(mir_unit.is_unit());

        let hir_i32 = HirType::I32;
        let mir_i32 = MirType::from_hir(&hir_i32);
        assert_eq!(mir_i32, MirType::I32);
        assert!(mir_i32.is_integer());
    }

    #[test]
    fn test_constant_types() {
        let unit_const = MirConstant::Unit;
        assert_eq!(unit_const.ty(), MirType::Unit);

        let bool_const = MirConstant::Bool(true);
        assert_eq!(bool_const.ty(), MirType::Bool);

        let int_const = MirConstant::Int(42);
        assert_eq!(int_const.ty(), MirType::I64);

        let float_const = MirConstant::Float("3.14".to_string());
        assert_eq!(float_const.ty(), MirType::F64);
    }

    #[test]
    fn test_type_sizes() {
        assert_eq!(MirType::I32.size_bytes(), Some(4));
        assert_eq!(MirType::F64.size_bytes(), Some(8));
        assert_eq!(MirType::Bool.size_bytes(), Some(1));
        assert_eq!(MirType::Ptr(Box::new(MirType::I32)).size_bytes(), Some(8));
    }
}
