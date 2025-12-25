//! Semantic Type System Integration
//!
//! This module integrates the ontology infrastructure into Sounio' type system,
//! enabling type compatibility based on semantic distance rather than structural equality.
//!
//! # The Paradigm Shift
//!
//! Traditional type systems: Types are compatible iff structurally equal
//! Sounio semantic types: Types are compatible based on semantic distance
//!
//! # Example
//!
//! ```sounio
//! // Aspirin is semantically close to Drug (is_a relationship)
//! fn process_drug(drug: ChEBI.Drug) { ... }
//!
//! let aspirin: ChEBI.Aspirin = ...;
//! process_drug(aspirin);  // Compiles! Semantic distance < threshold
//!
//! // Dog is semantically far from Drug
//! let fido: NCBITaxon.Dog = ...;
//! process_drug(fido);  // Error! Semantic distance too large
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::ontology::distance::{PhysicalCost, SemanticDistance, SemanticDistanceIndex};
use crate::ontology::loader::IRI;

/// A semantic type that wraps an ontological term
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticType {
    /// The ontological IRI for this type
    pub iri: IRI,

    /// Human-readable name
    pub name: String,

    /// Ontology prefix (e.g., "ChEBI", "GO")
    pub ontology: String,

    /// Local ID within the ontology
    pub local_id: String,

    /// Compatibility threshold for implicit coercion
    pub implicit_threshold: f64,

    /// Compatibility threshold for explicit cast
    pub explicit_threshold: f64,
}

impl SemanticType {
    /// Create a semantic type from an IRI
    pub fn from_iri(iri: IRI, name: String) -> Self {
        let (ontology, local_id) = iri
            .to_curie()
            .unwrap_or_else(|| ("UNKNOWN".to_string(), iri.as_str().to_string()));

        Self {
            iri,
            name,
            ontology,
            local_id,
            implicit_threshold: 0.3, // Allow implicit coercion if distance < 0.3
            explicit_threshold: 0.7, // Allow explicit cast if distance < 0.7
        }
    }

    /// Create from CURIE format (e.g., "ChEBI:15365")
    pub fn from_curie(prefix: &str, local: &str, name: String) -> Self {
        let iri = IRI::from_curie(prefix, local);
        Self {
            iri,
            name,
            ontology: prefix.to_uppercase(),
            local_id: local.to_string(),
            implicit_threshold: 0.3,
            explicit_threshold: 0.7,
        }
    }

    /// Get the CURIE representation
    pub fn curie(&self) -> String {
        format!("{}:{}", self.ontology, self.local_id)
    }

    /// Set custom compatibility thresholds
    pub fn with_thresholds(mut self, implicit: f64, explicit: f64) -> Self {
        self.implicit_threshold = implicit;
        self.explicit_threshold = explicit;
        self
    }
}

/// Result of semantic compatibility check
#[derive(Debug, Clone)]
pub struct SemanticCompatibility {
    /// The semantic distance between types
    pub distance: SemanticDistance,

    /// Whether implicit coercion is allowed
    pub implicit_compatible: bool,

    /// Whether explicit cast is allowed
    pub explicit_compatible: bool,

    /// Conversion path if one exists (for error messages)
    pub conversion_path: Option<Vec<IRI>>,

    /// Reason for incompatibility (if any)
    pub incompatibility_reason: Option<String>,
}

impl SemanticCompatibility {
    /// Create a compatible result
    pub fn compatible(distance: SemanticDistance) -> Self {
        Self {
            implicit_compatible: distance.is_implicitly_compatible(),
            explicit_compatible: distance.is_explicitly_compatible(),
            distance,
            conversion_path: None,
            incompatibility_reason: None,
        }
    }

    /// Create an incompatible result
    pub fn incompatible(reason: String) -> Self {
        Self {
            distance: SemanticDistance::MAX,
            implicit_compatible: false,
            explicit_compatible: false,
            conversion_path: None,
            incompatibility_reason: Some(reason),
        }
    }

    /// Check if types are compatible at all
    pub fn is_compatible(&self) -> bool {
        self.implicit_compatible || self.explicit_compatible
    }
}

/// Configuration for semantic type checking
#[derive(Debug, Clone)]
pub struct SemanticTypeConfig {
    /// Maximum semantic distance for implicit coercion
    pub implicit_coercion_threshold: f64,

    /// Maximum semantic distance for explicit cast
    pub explicit_cast_threshold: f64,

    /// Whether to allow cross-ontology coercion
    pub allow_cross_ontology: bool,

    /// Minimum SSSOM mapping confidence for cross-ontology
    pub min_mapping_confidence: f64,

    /// Whether to compute physical cost
    pub compute_physical_cost: bool,

    /// Whether to track provenance through conversions
    pub track_provenance: bool,
}

impl Default for SemanticTypeConfig {
    fn default() -> Self {
        Self {
            implicit_coercion_threshold: 0.3,
            explicit_cast_threshold: 0.7,
            allow_cross_ontology: true,
            min_mapping_confidence: 0.5,
            compute_physical_cost: true,
            track_provenance: true,
        }
    }
}

/// Semantic type checker that uses ontological distance
pub struct SemanticTypeChecker {
    /// The semantic distance index
    distance_index: Arc<RwLock<SemanticDistanceIndex>>,

    /// Configuration
    config: SemanticTypeConfig,

    /// Cache of type compatibility results
    compatibility_cache: RwLock<HashMap<(IRI, IRI), SemanticCompatibility>>,

    /// Statistics
    stats: RwLock<SemanticTypeStats>,
}

/// Statistics about semantic type checking
#[derive(Debug, Clone, Default)]
pub struct SemanticTypeStats {
    /// Number of compatibility checks
    pub checks: usize,

    /// Number of cache hits
    pub cache_hits: usize,

    /// Number of implicit coercions allowed
    pub implicit_coercions: usize,

    /// Number of explicit casts required
    pub explicit_casts: usize,

    /// Number of incompatible type pairs
    pub incompatible: usize,

    /// Total physical cost of all coercions
    pub total_physical_cost: u64,
}

impl SemanticTypeChecker {
    /// Create a new semantic type checker
    pub fn new(distance_index: Arc<RwLock<SemanticDistanceIndex>>) -> Self {
        Self::with_config(distance_index, SemanticTypeConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(
        distance_index: Arc<RwLock<SemanticDistanceIndex>>,
        config: SemanticTypeConfig,
    ) -> Self {
        Self {
            distance_index,
            config,
            compatibility_cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(SemanticTypeStats::default()),
        }
    }

    /// Check semantic compatibility between two types
    pub fn check_compatibility(
        &self,
        from: &SemanticType,
        to: &SemanticType,
    ) -> SemanticCompatibility {
        // Update stats
        {
            if let Ok(mut stats) = self.stats.write() {
                stats.checks += 1;
            }
        }

        // Check cache first
        {
            if let Ok(cache) = self.compatibility_cache.read()
                && let Some(result) = cache.get(&(from.iri.clone(), to.iri.clone()))
            {
                if let Ok(mut stats) = self.stats.write() {
                    stats.cache_hits += 1;
                }
                return result.clone();
            }
        }

        // Same type = exact match
        if from.iri == to.iri {
            let result = SemanticCompatibility::compatible(SemanticDistance::ZERO);
            self.cache_result(&from.iri, &to.iri, result.clone());
            return result;
        }

        // Check cross-ontology
        if from.ontology != to.ontology && !self.config.allow_cross_ontology {
            let result = SemanticCompatibility::incompatible(format!(
                "Cross-ontology coercion disabled: {} -> {}",
                from.ontology, to.ontology
            ));
            self.cache_result(&from.iri, &to.iri, result.clone());
            return result;
        }

        // Compute semantic distance
        let distance = {
            if let Ok(index) = self.distance_index.read() {
                index.distance(&from.iri, &to.iri)
            } else {
                SemanticDistance::MAX
            }
        };

        // Determine compatibility based on thresholds
        let implicit_compatible = distance.conceptual <= self.config.implicit_coercion_threshold
            && distance.conceptual <= from.implicit_threshold;
        let explicit_compatible = distance.conceptual <= self.config.explicit_cast_threshold
            && distance.conceptual <= from.explicit_threshold;

        // Get conversion path for error messages
        let conversion_path = if !implicit_compatible {
            if let Ok(index) = self.distance_index.read() {
                index.get_subsumption_path(&from.iri, &to.iri)
            } else {
                None
            }
        } else {
            None
        };

        // Update stats
        {
            if let Ok(mut stats) = self.stats.write() {
                if implicit_compatible {
                    stats.implicit_coercions += 1;
                } else if explicit_compatible {
                    stats.explicit_casts += 1;
                } else {
                    stats.incompatible += 1;
                }
                if self.config.compute_physical_cost {
                    stats.total_physical_cost += distance.physical_cost.cycles;
                }
            }
        }

        let result = SemanticCompatibility {
            distance,
            implicit_compatible,
            explicit_compatible,
            conversion_path,
            incompatibility_reason: if !explicit_compatible {
                Some(format!(
                    "Semantic distance {} exceeds threshold {} for {} -> {}",
                    distance.conceptual,
                    self.config.explicit_cast_threshold,
                    from.curie(),
                    to.curie()
                ))
            } else {
                None
            },
        };

        self.cache_result(&from.iri, &to.iri, result.clone());
        result
    }

    /// Check if implicit coercion is allowed
    pub fn allows_implicit_coercion(&self, from: &SemanticType, to: &SemanticType) -> bool {
        self.check_compatibility(from, to).implicit_compatible
    }

    /// Check if explicit cast is allowed
    pub fn allows_explicit_cast(&self, from: &SemanticType, to: &SemanticType) -> bool {
        self.check_compatibility(from, to).explicit_compatible
    }

    /// Get the semantic distance between types
    pub fn semantic_distance(&self, from: &SemanticType, to: &SemanticType) -> SemanticDistance {
        self.check_compatibility(from, to).distance
    }

    /// Check if types are in a subsumption relationship (is_a)
    pub fn is_subtype(&self, sub: &SemanticType, sup: &SemanticType) -> bool {
        if let Ok(index) = self.distance_index.read() {
            index.is_subtype(&sub.iri, &sup.iri)
        } else {
            false
        }
    }

    /// Get the physical cost of a type conversion
    pub fn conversion_cost(&self, from: &SemanticType, to: &SemanticType) -> PhysicalCost {
        self.check_compatibility(from, to).distance.physical_cost
    }

    /// Get confidence retention for a type conversion
    pub fn confidence_retention(&self, from: &SemanticType, to: &SemanticType) -> f64 {
        self.check_compatibility(from, to)
            .distance
            .confidence_retention
    }

    /// Cache a compatibility result
    fn cache_result(&self, from: &IRI, to: &IRI, result: SemanticCompatibility) {
        if let Ok(mut cache) = self.compatibility_cache.write() {
            cache.insert((from.clone(), to.clone()), result);
        }
    }

    /// Clear the compatibility cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.compatibility_cache.write() {
            cache.clear();
        }
    }

    /// Get statistics
    pub fn stats(&self) -> SemanticTypeStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = SemanticTypeStats::default();
        }
    }
}

/// Error type for semantic type checking
#[derive(Debug, Clone)]
pub enum SemanticTypeError {
    /// Types are semantically incompatible
    Incompatible {
        from: String,
        to: String,
        distance: f64,
        threshold: f64,
    },

    /// Cross-ontology coercion not allowed
    CrossOntologyDisabled {
        from_ontology: String,
        to_ontology: String,
    },

    /// Unknown ontology term
    UnknownTerm(String),

    /// Mapping confidence too low
    LowMappingConfidence {
        from: String,
        to: String,
        confidence: f64,
        required: f64,
    },
}

impl std::fmt::Display for SemanticTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incompatible {
                from,
                to,
                distance,
                threshold,
            } => {
                write!(
                    f,
                    "Cannot convert {} to {}: semantic distance {:.3} exceeds threshold {:.3}",
                    from, to, distance, threshold
                )
            }
            Self::CrossOntologyDisabled {
                from_ontology,
                to_ontology,
            } => {
                write!(
                    f,
                    "Cross-ontology coercion from {} to {} is disabled",
                    from_ontology, to_ontology
                )
            }
            Self::UnknownTerm(term) => {
                write!(f, "Unknown ontology term: {}", term)
            }
            Self::LowMappingConfidence {
                from,
                to,
                confidence,
                required,
            } => {
                write!(
                    f,
                    "Mapping from {} to {} has confidence {:.3}, required {:.3}",
                    from, to, confidence, required
                )
            }
        }
    }
}

impl std::error::Error for SemanticTypeError {}

/// Builder for semantic type expressions in the DSL
pub struct SemanticTypeBuilder {
    ontology: String,
    local_id: String,
    name: Option<String>,
    implicit_threshold: Option<f64>,
    explicit_threshold: Option<f64>,
}

impl SemanticTypeBuilder {
    /// Start building a semantic type
    pub fn new(ontology: &str, local_id: &str) -> Self {
        Self {
            ontology: ontology.to_uppercase(),
            local_id: local_id.to_string(),
            name: None,
            implicit_threshold: None,
            explicit_threshold: None,
        }
    }

    /// Set the human-readable name
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set the implicit coercion threshold
    pub fn implicit_threshold(mut self, threshold: f64) -> Self {
        self.implicit_threshold = Some(threshold);
        self
    }

    /// Set the explicit cast threshold
    pub fn explicit_threshold(mut self, threshold: f64) -> Self {
        self.explicit_threshold = Some(threshold);
        self
    }

    /// Build the semantic type
    pub fn build(self) -> SemanticType {
        let name = self
            .name
            .unwrap_or_else(|| format!("{}:{}", self.ontology, self.local_id));

        let mut ty = SemanticType::from_curie(&self.ontology, &self.local_id, name);

        if let Some(t) = self.implicit_threshold {
            ty.implicit_threshold = t;
        }
        if let Some(t) = self.explicit_threshold {
            ty.explicit_threshold = t;
        }

        ty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_type_creation() {
        let aspirin = SemanticType::from_curie("ChEBI", "15365", "Aspirin".to_string());
        assert_eq!(aspirin.ontology, "CHEBI");
        assert_eq!(aspirin.local_id, "15365");
        assert_eq!(aspirin.curie(), "CHEBI:15365");
    }

    #[test]
    fn test_semantic_type_builder() {
        let ty = SemanticTypeBuilder::new("GO", "0008150")
            .name("biological_process")
            .implicit_threshold(0.2)
            .build();

        assert_eq!(ty.ontology, "GO");
        assert_eq!(ty.name, "biological_process");
        assert_eq!(ty.implicit_threshold, 0.2);
    }

    #[test]
    fn test_compatibility_same_type() {
        let index = Arc::new(RwLock::new(SemanticDistanceIndex::new()));
        let checker = SemanticTypeChecker::new(index);

        let aspirin = SemanticType::from_curie("ChEBI", "15365", "Aspirin".to_string());
        let compat = checker.check_compatibility(&aspirin, &aspirin);

        assert!(compat.implicit_compatible);
        assert!(compat.explicit_compatible);
        assert!(compat.distance.is_exact());
    }
}
