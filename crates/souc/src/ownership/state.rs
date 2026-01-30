//! Ownership state tracking
//!
//! Tracks the state of values and borrows during ownership checking.

use crate::common::Span;
use crate::resolve::DefId;
use std::collections::HashMap;

/// A place (lvalue) in memory
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Place {
    /// Base variable
    pub base: DefId,
    /// Projections (field access, index, deref)
    pub projections: Vec<Projection>,
}

/// A projection from a base place
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Projection {
    Field(String),
    Index,
    Deref,
}

impl Place {
    pub fn var(def_id: DefId) -> Self {
        Self {
            base: def_id,
            projections: Vec::new(),
        }
    }

    pub fn field(mut self, name: impl Into<String>) -> Self {
        self.projections.push(Projection::Field(name.into()));
        self
    }

    pub fn index(mut self) -> Self {
        self.projections.push(Projection::Index);
        self
    }

    pub fn deref(mut self) -> Self {
        self.projections.push(Projection::Deref);
        self
    }
}

impl std::fmt::Display for Place {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.base.0)?;
        for proj in &self.projections {
            match proj {
                Projection::Field(name) => write!(f, ".{}", name)?,
                Projection::Index => write!(f, "[_]")?,
                Projection::Deref => write!(f, ".*")?,
            }
        }
        Ok(())
    }
}

/// Unique place ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaceId(pub u32);

/// State of ownership for a place
#[derive(Debug, Clone)]
pub enum OwnershipState {
    /// Owned, not borrowed
    Owned,
    /// Moved to another location
    Moved { to: Span },
    /// Borrowed (shared)
    BorrowedShared {
        /// Spans of active borrows
        borrows: Vec<Span>,
    },
    /// Borrowed (exclusive)
    BorrowedExclusive { borrow: Span },
    /// Dropped/out of scope
    Dropped,
}

impl OwnershipState {
    pub fn is_usable(&self) -> bool {
        matches!(
            self,
            OwnershipState::Owned | OwnershipState::BorrowedShared { .. }
        )
    }

    pub fn is_movable(&self) -> bool {
        matches!(self, OwnershipState::Owned)
    }

    pub fn can_borrow_shared(&self) -> bool {
        matches!(
            self,
            OwnershipState::Owned | OwnershipState::BorrowedShared { .. }
        )
    }

    pub fn can_borrow_exclusive(&self) -> bool {
        matches!(self, OwnershipState::Owned)
    }
}

/// State of a borrow
#[derive(Debug, Clone)]
pub struct BorrowState {
    /// Place being borrowed
    pub place: Place,
    /// Is this an exclusive borrow?
    pub exclusive: bool,
    /// Span where borrow occurred
    pub span: Span,
    /// Is the borrow still active?
    pub active: bool,
}

/// Linear/affine tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Linearity {
    /// Normal (unrestricted) type - can be used any number of times
    #[default]
    Unrestricted,
    /// Linear (must use exactly once)
    Linear,
    /// Affine (may use at most once)
    Affine,
}

/// Tracked value
#[derive(Debug, Clone)]
pub struct TrackedValue {
    pub def_id: DefId,
    pub name: String,
    pub linearity: Linearity,
    pub state: OwnershipState,
    pub use_count: u32,
    pub decl_span: Span,
    pub uses: Vec<Span>,
}

impl TrackedValue {
    pub fn new(def_id: DefId, name: String, linearity: Linearity, decl_span: Span) -> Self {
        Self {
            def_id,
            name,
            linearity,
            state: OwnershipState::Owned,
            use_count: 0,
            decl_span,
            uses: Vec::new(),
        }
    }

    pub fn record_use(&mut self, span: Span) {
        self.use_count += 1;
        self.uses.push(span);
    }

    pub fn check_linearity(&self) -> Result<(), LinearityError> {
        match self.linearity {
            Linearity::Linear => {
                if self.use_count == 0 {
                    Err(LinearityError::NotConsumed {
                        def_id: self.def_id,
                        name: self.name.clone(),
                        decl_span: self.decl_span,
                    })
                } else if self.use_count > 1 {
                    Err(LinearityError::MultipleUse {
                        def_id: self.def_id,
                        name: self.name.clone(),
                        first: self.uses[0],
                        second: self.uses[1],
                    })
                } else {
                    Ok(())
                }
            }
            Linearity::Affine => {
                if self.use_count > 1 {
                    Err(LinearityError::MultipleUse {
                        def_id: self.def_id,
                        name: self.name.clone(),
                        first: self.uses[0],
                        second: self.uses[1],
                    })
                } else {
                    Ok(())
                }
            }
            Linearity::Unrestricted => Ok(()),
        }
    }
}

/// Linearity error
#[derive(Debug, Clone)]
pub enum LinearityError {
    NotConsumed {
        def_id: DefId,
        name: String,
        decl_span: Span,
    },
    MultipleUse {
        def_id: DefId,
        name: String,
        first: Span,
        second: Span,
    },
}

/// Ownership state for entire scope
#[derive(Debug, Default)]
pub struct ScopeState {
    /// Tracked values by DefId
    values: HashMap<DefId, TrackedValue>,
    /// Active borrows
    borrows: Vec<BorrowState>,
}

impl ScopeState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn track(&mut self, value: TrackedValue) {
        self.values.insert(value.def_id, value);
    }

    pub fn get(&self, def_id: DefId) -> Option<&TrackedValue> {
        self.values.get(&def_id)
    }

    pub fn get_mut(&mut self, def_id: DefId) -> Option<&mut TrackedValue> {
        self.values.get_mut(&def_id)
    }

    pub fn add_borrow(&mut self, borrow: BorrowState) {
        self.borrows.push(borrow);
    }

    pub fn end_borrow(&mut self, place: &Place) {
        for borrow in &mut self.borrows {
            if &borrow.place == place {
                borrow.active = false;
            }
        }
    }

    pub fn active_borrows(&self, place: &Place) -> Vec<&BorrowState> {
        self.borrows
            .iter()
            .filter(|b| b.active && &b.place == place)
            .collect()
    }

    pub fn check_all_linear(&self) -> Vec<LinearityError> {
        let mut errors = Vec::new();
        for value in self.values.values() {
            if let Err(e) = value.check_linearity() {
                errors.push(e);
            }
        }
        errors
    }

    /// Get all tracked values
    pub fn values(&self) -> impl Iterator<Item = &TrackedValue> {
        self.values.values()
    }
}
