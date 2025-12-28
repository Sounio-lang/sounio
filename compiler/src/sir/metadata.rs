//! SIR Metadata
//!
//! Metadata provides additional information for optimization and debugging
//! without affecting program semantics.

use super::values::ValueId;
use std::collections::HashMap;

/// Metadata kinds
#[derive(Debug, Clone, PartialEq)]
pub enum Metadata {
    /// Source location for debugging
    SourceLocation {
        file: String,
        line: u32,
        column: u32,
    },

    /// Variable name for debugging
    VariableName(String),

    /// Loop metadata for optimization
    Loop(LoopMetadata),

    /// Branch probability
    BranchWeights(Vec<u32>),

    /// Alias scope for memory disambiguation
    AliasScope(String),

    /// No-alias declaration
    NoAlias(Vec<String>),

    /// Invariant load (value won't change during function)
    InvariantLoad,

    /// Dereferenceable (pointer is valid for N bytes)
    Dereferenceable(usize),

    /// Non-null pointer
    NonNull,

    /// Range of values (for integer types)
    Range { min: i64, max: i64 },

    // === Domain-Specific Metadata ===
    /// Epistemic metadata
    Epistemic(EpistemicMetadata),

    /// Physical unit
    Unit(super::values::PhysicalUnit),

    /// Probability distribution info
    Distribution(DistributionMetadata),

    /// ODE solver metadata
    OdeSolver(OdeSolverMetadata),
}

/// Loop optimization metadata
#[derive(Debug, Clone, PartialEq)]
pub struct LoopMetadata {
    /// Vectorization hint
    pub vectorize: Option<bool>,
    /// Unroll factor hint
    pub unroll: Option<u32>,
    /// Distribute hint
    pub distribute: bool,
    /// Loop is known to execute at least once
    pub non_zero_trip_count: bool,
    /// Loop bound is known
    pub known_trip_count: Option<u64>,
    /// Inner loop of a loop nest
    pub inner_loop: bool,
    /// Parallel loop (no loop-carried dependencies)
    pub parallel: bool,
}

impl Default for LoopMetadata {
    fn default() -> Self {
        Self {
            vectorize: None,
            unroll: None,
            distribute: false,
            non_zero_trip_count: false,
            known_trip_count: None,
            inner_loop: false,
            parallel: false,
        }
    }
}

/// Epistemic-specific metadata
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicMetadata {
    /// Confidence is statically known
    pub known_confidence: Option<f64>,
    /// Confidence is loop-invariant
    pub confidence_invariant: bool,
    /// Value is fully certain (confidence = 1.0)
    pub certain: bool,
    /// Provenance source ID
    pub provenance_id: Option<u32>,
}

impl Default for EpistemicMetadata {
    fn default() -> Self {
        Self {
            known_confidence: None,
            confidence_invariant: false,
            certain: false,
            provenance_id: None,
        }
    }
}

impl EpistemicMetadata {
    pub fn certain() -> Self {
        Self {
            known_confidence: Some(1.0),
            certain: true,
            ..Default::default()
        }
    }

    pub fn with_confidence(conf: f64) -> Self {
        Self {
            known_confidence: Some(conf),
            certain: conf == 1.0,
            ..Default::default()
        }
    }
}

/// Distribution metadata for optimization
#[derive(Debug, Clone, PartialEq)]
pub struct DistributionMetadata {
    /// Distribution kind (if known statically)
    pub kind: Option<super::types::DistributionKind>,
    /// Parameters are constants
    pub constant_params: bool,
    /// Number of samples to be drawn
    pub sample_count: Option<u64>,
    /// Can use vectorized sampling
    pub vectorizable: bool,
}

impl Default for DistributionMetadata {
    fn default() -> Self {
        Self {
            kind: None,
            constant_params: false,
            sample_count: None,
            vectorizable: true,
        }
    }
}

/// ODE solver metadata for optimization
#[derive(Debug, Clone, PartialEq)]
pub struct OdeSolverMetadata {
    /// Number of state variables
    pub state_size: u32,
    /// Derivative function is pure
    pub pure_derivatives: bool,
    /// Jacobian is available
    pub has_jacobian: bool,
    /// System is stiff (suggests implicit methods)
    pub stiff: bool,
    /// Can use adaptive stepping
    pub adaptive: bool,
    /// Error tolerance
    pub tolerance: Option<f64>,
}

impl Default for OdeSolverMetadata {
    fn default() -> Self {
        Self {
            state_size: 0,
            pure_derivatives: true,
            has_jacobian: false,
            stiff: false,
            adaptive: true,
            tolerance: None,
        }
    }
}

/// Metadata storage attached to values and instructions
#[derive(Debug, Clone, Default)]
pub struct MetadataStore {
    /// Value-level metadata
    value_metadata: HashMap<ValueId, Vec<Metadata>>,
    /// Named metadata (module-level)
    named_metadata: HashMap<String, Vec<Metadata>>,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach metadata to a value
    pub fn attach(&mut self, value: ValueId, meta: Metadata) {
        self.value_metadata.entry(value).or_default().push(meta);
    }

    /// Get metadata for a value
    pub fn get(&self, value: ValueId) -> Option<&[Metadata]> {
        self.value_metadata.get(&value).map(|v| v.as_slice())
    }

    /// Check if value has specific metadata kind
    pub fn has_metadata<F>(&self, value: ValueId, predicate: F) -> bool
    where
        F: Fn(&Metadata) -> bool,
    {
        self.value_metadata
            .get(&value)
            .map(|metas| metas.iter().any(predicate))
            .unwrap_or(false)
    }

    /// Get epistemic metadata for a value
    pub fn epistemic(&self, value: ValueId) -> Option<&EpistemicMetadata> {
        self.get(value).and_then(|metas| {
            metas.iter().find_map(|m| {
                if let Metadata::Epistemic(e) = m {
                    Some(e)
                } else {
                    None
                }
            })
        })
    }

    /// Check if value is marked as certain
    pub fn is_certain(&self, value: ValueId) -> bool {
        self.epistemic(value).map(|e| e.certain).unwrap_or(false)
    }

    /// Add named metadata
    pub fn add_named(&mut self, name: impl Into<String>, meta: Metadata) {
        self.named_metadata
            .entry(name.into())
            .or_default()
            .push(meta);
    }

    /// Get named metadata
    pub fn get_named(&self, name: &str) -> Option<&[Metadata]> {
        self.named_metadata.get(name).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_store() {
        let mut store = MetadataStore::new();
        let val = ValueId::new(0);

        store.attach(val, Metadata::Epistemic(EpistemicMetadata::certain()));

        assert!(store.is_certain(val));
    }

    #[test]
    fn test_loop_metadata() {
        let meta = LoopMetadata {
            vectorize: Some(true),
            parallel: true,
            known_trip_count: Some(1000),
            ..Default::default()
        };

        assert!(meta.parallel);
        assert_eq!(meta.known_trip_count, Some(1000));
    }
}
