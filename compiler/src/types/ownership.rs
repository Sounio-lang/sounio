//! Ownership and borrowing rules
//!
//! D supports three ownership modes:
//! - **Copy**: Value can be freely copied (primitives, small structs)
//! - **Affine**: Value can be used at most once (default for most types)
//! - **Linear**: Value must be used exactly once (resources that must be cleaned up)

use super::Type;

/// Ownership mode for a type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ownership {
    /// Can be freely copied
    Copy,
    /// Can be used at most once (can be dropped without using)
    #[default]
    Affine,
    /// Must be used exactly once (cannot be dropped implicitly)
    Linear,
}

/// Determine the ownership mode of a type
pub fn ownership_of(ty: &Type) -> Ownership {
    match ty {
        // Primitives are Copy
        Type::Unit
        | Type::Bool
        | Type::I8
        | Type::I16
        | Type::I32
        | Type::I64
        | Type::I128
        | Type::Isize
        | Type::U8
        | Type::U16
        | Type::U32
        | Type::U64
        | Type::U128
        | Type::Usize
        | Type::F32
        | Type::F64
        | Type::Char => Ownership::Copy,

        // References are Copy (the reference itself, not the data)
        Type::Ref { .. } => Ownership::Copy,

        // Raw pointers are Copy (like references, the pointer itself is copied)
        Type::RawPointer { .. } => Ownership::Copy,

        // Function pointers are Copy
        Type::Function { .. } => Ownership::Copy,

        // Strings are Affine (heap allocated)
        Type::Str | Type::String => Ownership::Affine,

        // Arrays depend on element type
        Type::Array { element, .. } => {
            let elem_own = ownership_of(element);
            if elem_own == Ownership::Linear {
                Ownership::Linear
            } else {
                elem_own
            }
        }

        // Tuples depend on element types
        Type::Tuple(elems) => {
            let mut result = Ownership::Copy;
            for elem in elems {
                let elem_own = ownership_of(elem);
                if elem_own == Ownership::Linear {
                    return Ownership::Linear;
                }
                if elem_own == Ownership::Affine {
                    result = Ownership::Affine;
                }
            }
            result
        }

        // Named types default to Affine (could be overridden by type definition)
        Type::Named { .. } => Ownership::Affine,

        // Quantity types: ownership follows the numeric type (usually Copy)
        Type::Quantity { numeric, .. } => ownership_of(numeric),

        // Type variables are Affine by default
        Type::Var(_) | Type::Forall { .. } => Ownership::Affine,

        // Error types and SelfType (resolved later)
        Type::Never | Type::Unknown | Type::Error | Type::SelfType => Ownership::Copy,

        // Ontology types are Copy (they're essentially interned string identifiers)
        Type::Ontology { .. } => Ownership::Copy,

        // Linear algebra primitives are Copy (fixed-size, stack-allocated)
        Type::Vec2 | Type::Vec3 | Type::Vec4 => Ownership::Copy,
        Type::Mat2 | Type::Mat3 | Type::Mat4 => Ownership::Copy,
        Type::Quat => Ownership::Copy,

        // Dual numbers are Copy (two f64 values, stack-allocated)
        Type::Dual => Ownership::Copy,
    }
}

/// Check if a type can be moved
pub fn can_move(ty: &Type) -> bool {
    // All types can be moved
    true
}

/// Check if a type can be copied
pub fn can_copy(ty: &Type) -> bool {
    ownership_of(ty) == Ownership::Copy
}

/// Check if a type must be used (linear)
pub fn must_use(ty: &Type) -> bool {
    ownership_of(ty) == Ownership::Linear
}

/// Borrow state tracking
#[derive(Debug, Clone, Default)]
pub struct BorrowState {
    /// Active shared borrows
    pub shared_borrows: Vec<BorrowInfo>,
    /// Active mutable borrow (only one allowed)
    pub mutable_borrow: Option<BorrowInfo>,
    /// Whether the value has been moved
    pub moved: bool,
}

/// Information about a borrow
#[derive(Debug, Clone)]
pub struct BorrowInfo {
    pub location: String,
    pub mutable: bool,
}

impl BorrowState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to create a shared borrow
    pub fn borrow_shared(&mut self, location: String) -> Result<(), BorrowError> {
        if self.moved {
            return Err(BorrowError::UseAfterMove);
        }
        if self.mutable_borrow.is_some() {
            return Err(BorrowError::BorrowConflict {
                existing: "mutable".to_string(),
                requested: "shared".to_string(),
            });
        }
        self.shared_borrows.push(BorrowInfo {
            location,
            mutable: false,
        });
        Ok(())
    }

    /// Try to create a mutable borrow
    pub fn borrow_mut(&mut self, location: String) -> Result<(), BorrowError> {
        if self.moved {
            return Err(BorrowError::UseAfterMove);
        }
        if self.mutable_borrow.is_some() {
            return Err(BorrowError::BorrowConflict {
                existing: "mutable".to_string(),
                requested: "mutable".to_string(),
            });
        }
        if !self.shared_borrows.is_empty() {
            return Err(BorrowError::BorrowConflict {
                existing: "shared".to_string(),
                requested: "mutable".to_string(),
            });
        }
        self.mutable_borrow = Some(BorrowInfo {
            location,
            mutable: true,
        });
        Ok(())
    }

    /// Release a shared borrow
    pub fn release_shared(&mut self) {
        self.shared_borrows.pop();
    }

    /// Release a mutable borrow
    pub fn release_mut(&mut self) {
        self.mutable_borrow = None;
    }

    /// Mark the value as moved
    pub fn mark_moved(&mut self) -> Result<(), BorrowError> {
        if self.moved {
            return Err(BorrowError::UseAfterMove);
        }
        if self.mutable_borrow.is_some() || !self.shared_borrows.is_empty() {
            return Err(BorrowError::MoveWhileBorrowed);
        }
        self.moved = true;
        Ok(())
    }

    /// Check if there are any active borrows
    pub fn is_borrowed(&self) -> bool {
        self.mutable_borrow.is_some() || !self.shared_borrows.is_empty()
    }
}

/// Borrow checker error
#[derive(Debug, Clone)]
pub enum BorrowError {
    UseAfterMove,
    MoveWhileBorrowed,
    BorrowConflict { existing: String, requested: String },
    LinearNotUsed,
}

impl std::fmt::Display for BorrowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BorrowError::UseAfterMove => write!(f, "use of moved value"),
            BorrowError::MoveWhileBorrowed => write!(f, "cannot move while borrowed"),
            BorrowError::BorrowConflict {
                existing,
                requested,
            } => {
                write!(
                    f,
                    "cannot borrow as {} while {} borrow is active",
                    requested, existing
                )
            }
            BorrowError::LinearNotUsed => write!(f, "linear value must be used"),
        }
    }
}

impl std::error::Error for BorrowError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ownership_primitives() {
        assert_eq!(ownership_of(&Type::I32), Ownership::Copy);
        assert_eq!(ownership_of(&Type::Bool), Ownership::Copy);
        assert_eq!(ownership_of(&Type::F64), Ownership::Copy);
    }

    #[test]
    fn test_ownership_string() {
        assert_eq!(ownership_of(&Type::String), Ownership::Affine);
    }

    #[test]
    fn test_borrow_state() {
        let mut state = BorrowState::new();

        // Can borrow shared multiple times
        assert!(state.borrow_shared("loc1".into()).is_ok());
        assert!(state.borrow_shared("loc2".into()).is_ok());

        // Cannot borrow mutably while shared
        assert!(state.borrow_mut("loc3".into()).is_err());

        // Release shared borrows
        state.release_shared();
        state.release_shared();

        // Now can borrow mutably
        assert!(state.borrow_mut("loc3".into()).is_ok());

        // Cannot borrow shared while mutably borrowed
        assert!(state.borrow_shared("loc4".into()).is_err());
    }
}
