//! High-Level IR (HLIR) - SSA-based intermediate representation
//!
//! HLIR is an SSA-based intermediate representation suitable for optimization
//! and code generation. It features:
//! - Static Single Assignment (SSA) form
//! - Basic blocks with explicit control flow
//! - Explicit memory operations
//! - Type-safe operations

pub mod builder;
pub mod ir;
pub mod lower;

// Re-export main types
pub use builder::{FunctionBuilder, ModuleBuilder};
pub use ir::*;
pub use lower::lower;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::HirType;

    #[test]
    fn test_hlir_type_conversion() {
        assert_eq!(HlirType::from_hir(&HirType::I32), HlirType::I32);
        assert_eq!(HlirType::from_hir(&HirType::F64), HlirType::F64);
        assert_eq!(HlirType::from_hir(&HirType::Unit), HlirType::Void);
    }

    #[test]
    fn test_hlir_type_properties() {
        assert!(HlirType::I64.is_integer());
        assert!(HlirType::I64.is_signed());
        assert!(!HlirType::U64.is_signed());
        assert!(HlirType::F64.is_float());
        assert!(!HlirType::I64.is_float());
    }
}
