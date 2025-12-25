//! Semantic Distance Calculation
//!
//! This module implements semantic distance as a proper metric space,
//! enabling type compatibility based on conceptual distance.
//!
//! # Distance Metrics
//!
//! We support multiple distance metrics:
//!
//! 1. **Path Distance**: Shortest path through is-a hierarchy
//! 2. **Information Content**: Based on corpus statistics
//! 3. **Graph Embedding**: Vector similarity in embedding space
//! 4. **SSSOM Confidence**: Based on curated mappings
//!
//! # The Key Insight
//!
//! These are not just similarity scores — they are compiled into the type system.
//! A function that accepts `Disease` can accept `Cancer` because the compiler
//! knows the semantic distance is within bounds.
//!
//! # Physical Cost Model
//!
//! Semantic distance maps to physical resources:
//! - Cache hierarchy (L1 → L2 → L3 → RAM → SSD → Network)
//! - CPU cycles for conversion
//! - Memory allocation for intermediate representations
//! - Network latency for federated resolution

pub mod cache;
pub mod information_content;
pub mod path;
pub mod sssom;

use std::collections::HashMap;
use std::sync::RwLock;

use super::alignment::{AlignmentIndex, AlignmentResult};
use super::loader::{IRI, LoadedTerm};

/// Semantic distance between two ontological terms
#[derive(Debug, Clone, Copy)]
pub struct SemanticDistance {
    /// Conceptual distance (0.0 = identical, 1.0 = maximally distant)
    pub conceptual: f64,

    /// Physical cost to bridge this distance
    pub physical_cost: PhysicalCost,

    /// Confidence retained after conversion
    pub confidence_retention: f64,

    /// How many provenance steps this conversion adds
    pub provenance_depth: u32,
}

impl SemanticDistance {
    /// Zero distance (identical types)
    pub const ZERO: Self = Self {
        conceptual: 0.0,
        physical_cost: PhysicalCost::ZERO,
        confidence_retention: 1.0,
        provenance_depth: 0,
    };

    /// Maximum distance (completely unrelated)
    pub const MAX: Self = Self {
        conceptual: 1.0,
        physical_cost: PhysicalCost::MAX,
        confidence_retention: 0.0,
        provenance_depth: u32::MAX,
    };

    /// Create a new semantic distance
    pub fn new(conceptual: f64) -> Self {
        Self {
            conceptual: conceptual.clamp(0.0, 1.0),
            physical_cost: PhysicalCost::from_conceptual(conceptual),
            confidence_retention: 1.0 - (conceptual * 0.2),
            provenance_depth: if conceptual > 0.0 { 1 } else { 0 },
        }
    }

    /// Compose two distances (for A → B → C)
    pub fn compose(self, other: Self) -> Self {
        Self {
            // Distances add (with ceiling at 1.0)
            conceptual: (self.conceptual + other.conceptual).min(1.0),

            // Costs add
            physical_cost: self.physical_cost.add(other.physical_cost),

            // Confidences multiply
            confidence_retention: self.confidence_retention * other.confidence_retention,

            // Provenance depths add
            provenance_depth: self.provenance_depth.saturating_add(other.provenance_depth),
        }
    }

    /// Check if this distance is within a threshold
    pub fn within_threshold(&self, threshold: f64) -> bool {
        self.conceptual <= threshold
    }

    /// Is this an exact match?
    pub fn is_exact(&self) -> bool {
        self.conceptual == 0.0
    }

    /// Is this within subsumption distance (direct is-a)?
    pub fn is_subsumption(&self) -> bool {
        self.conceptual <= 0.1
    }

    /// Is this within implicit coercion distance?
    pub fn is_implicitly_compatible(&self) -> bool {
        self.conceptual <= 0.3
    }

    /// Is this within explicit cast distance?
    pub fn is_explicitly_compatible(&self) -> bool {
        self.conceptual <= 0.7
    }
}

impl Default for SemanticDistance {
    fn default() -> Self {
        Self::MAX
    }
}

/// Physical cost to perform a type conversion
#[derive(Debug, Clone, Copy)]
pub struct PhysicalCost {
    /// Estimated CPU cycles
    pub cycles: u64,

    /// Memory tier required (0=register, 6=network)
    pub memory_tier: u8,

    /// Network round-trips needed
    pub network_hops: u8,

    /// Bytes of temporary allocation
    pub allocation: u64,
}

impl PhysicalCost {
    pub const ZERO: Self = Self {
        cycles: 0,
        memory_tier: 0,
        network_hops: 0,
        allocation: 0,
    };

    pub const MAX: Self = Self {
        cycles: u64::MAX,
        memory_tier: 7,
        network_hops: u8::MAX,
        allocation: u64::MAX,
    };

    /// Create physical cost from conceptual distance
    pub fn from_conceptual(distance: f64) -> Self {
        if distance == 0.0 {
            return Self::ZERO;
        }

        Self {
            cycles: (distance * 1000.0) as u64,
            memory_tier: if distance < 0.1 {
                1 // L1 cache
            } else if distance < 0.3 {
                2 // L2 cache
            } else if distance < 0.5 {
                3 // L3 cache
            } else if distance < 0.7 {
                4 // RAM
            } else {
                5 // SSD/Network
            },
            network_hops: if distance > 0.5 { 1 } else { 0 },
            allocation: if distance > 0.3 { 64 } else { 0 },
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            cycles: self.cycles.saturating_add(other.cycles),
            memory_tier: self.memory_tier.max(other.memory_tier),
            network_hops: self.network_hops.saturating_add(other.network_hops),
            allocation: self.allocation.saturating_add(other.allocation),
        }
    }

    /// Memory tier as human-readable string
    pub fn memory_tier_name(&self) -> &'static str {
        match self.memory_tier {
            0 => "register",
            1 => "L1 cache",
            2 => "L2 cache",
            3 => "L3 cache",
            4 => "RAM",
            5 => "SSD",
            6 => "Network",
            _ => "Unknown",
        }
    }
}

/// Configuration for distance calculation
#[derive(Debug, Clone)]
pub struct DistanceConfig {
    /// Weight for path-based distance
    pub path_weight: f64,

    /// Weight for IC-based distance
    pub ic_weight: f64,

    /// Weight for embedding-based distance
    pub embedding_weight: f64,

    /// Weight for SSSOM mapping confidence
    pub sssom_weight: f64,

    /// Maximum path length to consider (longer = unrelated)
    pub max_path_length: u32,

    /// Default distance for unknown pairs
    pub unknown_distance: f64,

    /// Minimum confidence to consider a mapping valid
    pub min_mapping_confidence: f64,

    /// Whether to use embeddings in distance calculation
    pub use_embeddings: bool,
}

impl Default for DistanceConfig {
    fn default() -> Self {
        Self {
            path_weight: 0.35,
            ic_weight: 0.25,
            embedding_weight: 0.40, // Embeddings get highest weight when available
            sssom_weight: 0.3,
            max_path_length: 20,
            unknown_distance: 0.9,
            min_mapping_confidence: 0.5,
            use_embeddings: true,
        }
    }
}

/// SSSOM mapping entry
#[derive(Debug, Clone)]
pub struct SSSOMMapping {
    /// Subject (source) IRI
    pub subject_id: IRI,

    /// Predicate (mapping relation)
    pub predicate_id: IRI,

    /// Object (target) IRI
    pub object_id: IRI,

    /// Mapping confidence (0.0 - 1.0)
    pub confidence: f64,

    /// Justification for the mapping
    pub mapping_justification: String,
}

/// Index for fast SSSOM mapping lookup
pub struct SSSOMIndex {
    /// Subject → Object mappings
    by_subject: HashMap<IRI, Vec<SSSOMMapping>>,

    /// Object → Subject mappings (reverse)
    by_object: HashMap<IRI, Vec<SSSOMMapping>>,
}

impl SSSOMIndex {
    pub fn new() -> Self {
        Self {
            by_subject: HashMap::new(),
            by_object: HashMap::new(),
        }
    }

    pub fn add(&mut self, mapping: SSSOMMapping) {
        self.by_subject
            .entry(mapping.subject_id.clone())
            .or_default()
            .push(mapping.clone());
        self.by_object
            .entry(mapping.object_id.clone())
            .or_default()
            .push(mapping);
    }

    pub fn find(&self, from: &IRI, to: &IRI) -> Option<&SSSOMMapping> {
        // Check direct mapping
        if let Some(mappings) = self.by_subject.get(from)
            && let Some(m) = mappings.iter().find(|m| &m.object_id == to)
        {
            return Some(m);
        }

        // Check reverse mapping
        if let Some(mappings) = self.by_object.get(from)
            && let Some(m) = mappings.iter().find(|m| &m.subject_id == to)
        {
            return Some(m);
        }

        None
    }

    pub fn get_mappings_for(&self, iri: &IRI) -> Vec<&SSSOMMapping> {
        let mut result = Vec::new();

        if let Some(mappings) = self.by_subject.get(iri) {
            result.extend(mappings.iter());
        }

        if let Some(mappings) = self.by_object.get(iri) {
            result.extend(mappings.iter());
        }

        result
    }

    pub fn len(&self) -> usize {
        self.by_subject.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_subject.is_empty()
    }
}

impl Default for SSSOMIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Index structure for fast semantic distance lookup
pub struct SemanticDistanceIndex {
    /// Graph structure for path-based distance
    hierarchy_graph: path::HierarchyGraph,

    /// Information content values for IC-based similarity
    ic_values: HashMap<IRI, f64>,

    /// SSSOM mappings for cross-ontology distance
    sssom_mappings: SSSOMIndex,

    /// Unified alignment index (SSSOM + CUI + LOOM + embeddings)
    alignment_index: Option<AlignmentIndex>,

    /// Embedding space for geometric distance (optional)
    embedding_space: Option<super::embedding::EmbeddingSpace>,

    /// Precomputed distances for common pairs
    distance_cache: RwLock<lru::LruCache<(IRI, IRI), SemanticDistance>>,

    /// Configuration
    config: DistanceConfig,
}

impl SemanticDistanceIndex {
    /// Create a new index
    pub fn new() -> Self {
        Self::with_config(DistanceConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DistanceConfig) -> Self {
        Self {
            hierarchy_graph: path::HierarchyGraph::new(),
            ic_values: HashMap::new(),
            sssom_mappings: SSSOMIndex::new(),
            alignment_index: None,
            embedding_space: None,
            distance_cache: RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(100_000).unwrap(),
            )),
            config,
        }
    }

    /// Set the unified alignment index for cross-ontology resolution
    pub fn set_alignment_index(&mut self, index: AlignmentIndex) {
        self.alignment_index = Some(index);
    }

    /// Get a reference to the alignment index
    pub fn alignment_index(&self) -> Option<&AlignmentIndex> {
        self.alignment_index.as_ref()
    }

    /// Get a mutable reference to the alignment index
    pub fn alignment_index_mut(&mut self) -> Option<&mut AlignmentIndex> {
        self.alignment_index.as_mut()
    }

    /// Build index from loaded terms
    pub fn build_from_terms(&mut self, terms: &[LoadedTerm]) {
        // Build hierarchy graph
        for term in terms {
            self.hierarchy_graph.add_term(&term.iri, &term.superclasses);
        }

        // Compute information content
        self.compute_information_content(terms);
    }

    /// Add a single term to the index
    pub fn add_term(&mut self, term: &LoadedTerm) {
        self.hierarchy_graph.add_term(&term.iri, &term.superclasses);
    }

    /// Load SSSOM mappings
    pub fn load_sssom(&mut self, mappings: &[SSSOMMapping]) {
        for mapping in mappings {
            self.sssom_mappings.add(mapping.clone());
        }
    }

    /// Attach an embedding space for geometric distance calculation
    pub fn set_embedding_space(&mut self, space: super::embedding::EmbeddingSpace) {
        self.embedding_space = Some(space);
    }

    /// Check if embedding space is available
    pub fn has_embeddings(&self) -> bool {
        self.embedding_space.is_some()
    }

    /// Get a reference to the embedding space
    pub fn embedding_space(&self) -> Option<&super::embedding::EmbeddingSpace> {
        self.embedding_space.as_ref()
    }

    /// Find nearest neighbors using embeddings
    pub fn nearest_neighbors(
        &self,
        iri: &IRI,
        k: usize,
    ) -> Result<Vec<(IRI, f64)>, super::embedding::EmbeddingError> {
        self.embedding_space
            .as_ref()
            .ok_or(super::embedding::EmbeddingError::NotInitialized)?
            .nearest_neighbors(iri, k)
    }

    /// Calculate semantic distance between two terms
    pub fn distance(&self, from: &IRI, to: &IRI) -> SemanticDistance {
        // Check cache first
        {
            if let Ok(cache) = self.distance_cache.read()
                && let Some(d) = cache.peek(&(from.clone(), to.clone()))
            {
                return *d;
            }
        }

        // Compute distance
        let distance = self.compute_distance(from, to);

        // Cache result
        {
            if let Ok(mut cache) = self.distance_cache.write() {
                cache.put((from.clone(), to.clone()), distance);
            }
        }

        distance
    }

    fn compute_distance(&self, from: &IRI, to: &IRI) -> SemanticDistance {
        if from == to {
            return SemanticDistance::ZERO;
        }

        // Same ontology: use hierarchy
        if from.ontology() == to.ontology() {
            return self.compute_intra_ontology_distance(from, to);
        }

        // Different ontologies: use SSSOM + path
        self.compute_cross_ontology_distance(from, to)
    }

    fn compute_intra_ontology_distance(&self, from: &IRI, to: &IRI) -> SemanticDistance {
        // Check for direct subsumption
        if self.hierarchy_graph.is_ancestor(to, from) {
            // to is an ancestor of from (from is more specific)
            let depth = self.hierarchy_graph.path_length(from, to).unwrap_or(10);
            let distance = (depth as f64 / self.config.max_path_length as f64).min(0.3);
            return SemanticDistance::new(distance);
        }

        if self.hierarchy_graph.is_ancestor(from, to) {
            // from is an ancestor of to (to is more specific)
            // This is a downcast - higher distance
            let depth = self.hierarchy_graph.path_length(to, from).unwrap_or(10);
            let distance = (depth as f64 / self.config.max_path_length as f64).min(0.5) + 0.1;
            return SemanticDistance::new(distance);
        }

        // Path-based distance via LCA
        let path_distance = if let Some(lca) = self.hierarchy_graph.lowest_common_ancestor(from, to)
        {
            let from_to_lca = self.hierarchy_graph.path_length(from, &lca).unwrap_or(10);
            let to_to_lca = self.hierarchy_graph.path_length(to, &lca).unwrap_or(10);
            let total = from_to_lca + to_to_lca;
            (total as f64 / (2.0 * self.config.max_path_length as f64)).min(1.0)
        } else {
            self.config.unknown_distance
        };

        // IC-based distance
        let ic_distance = self.compute_ic_distance(from, to);

        // Embedding-based distance (if available)
        let embedding_distance = self.compute_embedding_distance(from, to);

        // Weighted combination based on available metrics
        let conceptual = self.combine_distances(path_distance, ic_distance, embedding_distance);

        SemanticDistance::new(conceptual)
    }

    /// Compute embedding-based semantic distance
    fn compute_embedding_distance(&self, from: &IRI, to: &IRI) -> Option<f64> {
        if !self.config.use_embeddings {
            return None;
        }

        self.embedding_space
            .as_ref()
            .and_then(|space| space.embedding_distance(from, to).ok())
    }

    /// Combine multiple distance metrics with configured weights
    fn combine_distances(
        &self,
        path_distance: f64,
        ic_distance: f64,
        embedding_distance: Option<f64>,
    ) -> f64 {
        match (ic_distance > 0.0, embedding_distance) {
            // All three metrics available
            (true, Some(emb_dist)) => {
                // Normalize weights to sum to 1.0
                let total_weight =
                    self.config.path_weight + self.config.ic_weight + self.config.embedding_weight;
                let path_w = self.config.path_weight / total_weight;
                let ic_w = self.config.ic_weight / total_weight;
                let emb_w = self.config.embedding_weight / total_weight;

                path_w * path_distance + ic_w * ic_distance + emb_w * emb_dist
            }
            // Path and embedding only
            (false, Some(emb_dist)) => {
                let total_weight = self.config.path_weight + self.config.embedding_weight;
                let path_w = self.config.path_weight / total_weight;
                let emb_w = self.config.embedding_weight / total_weight;

                path_w * path_distance + emb_w * emb_dist
            }
            // Path and IC only
            (true, None) => {
                let total_weight = self.config.path_weight + self.config.ic_weight;
                let path_w = self.config.path_weight / total_weight;
                let ic_w = self.config.ic_weight / total_weight;

                path_w * path_distance + ic_w * ic_distance
            }
            // Path only
            (false, None) => path_distance,
        }
    }

    fn compute_cross_ontology_distance(&self, from: &IRI, to: &IRI) -> SemanticDistance {
        // Try unified alignment index first (includes SSSOM, CUI, LOOM)
        if let Some(ref alignment_index) = self.alignment_index
            && let Some(result) = alignment_index.find_alignment(from, to)
        {
            return self.alignment_result_to_distance(&result);
        }

        // Fallback to direct SSSOM mappings
        if let Some(mapping) = self.sssom_mappings.find(from, to)
            && mapping.confidence >= self.config.min_mapping_confidence
        {
            let conceptual = 1.0 - mapping.confidence;
            return SemanticDistance {
                conceptual,
                physical_cost: PhysicalCost {
                    cycles: 50,
                    memory_tier: 2,
                    network_hops: 0,
                    allocation: 0,
                },
                confidence_retention: mapping.confidence,
                provenance_depth: 1,
            };
        }

        // Try embedding-based similarity for cross-ontology
        if let Some(emb_distance) = self.compute_embedding_distance(from, to)
            && emb_distance < self.config.unknown_distance
        {
            return SemanticDistance {
                conceptual: emb_distance,
                physical_cost: PhysicalCost {
                    cycles: 200,
                    memory_tier: 3,
                    network_hops: 0,
                    allocation: 512,
                },
                confidence_retention: 1.0 - (emb_distance * 0.3),
                provenance_depth: 1,
            };
        }

        // No alignment found - high distance
        SemanticDistance {
            conceptual: self.config.unknown_distance,
            physical_cost: PhysicalCost {
                cycles: 1000,
                memory_tier: 4,
                network_hops: 1,
                allocation: 1024,
            },
            confidence_retention: 0.5,
            provenance_depth: 2,
        }
    }

    /// Convert an alignment result to semantic distance
    fn alignment_result_to_distance(&self, result: &AlignmentResult) -> SemanticDistance {
        use super::alignment::AlignmentMethod;

        let conceptual = result.alignment.to_distance();
        let confidence = result.alignment.effective_confidence();

        // Physical cost depends on alignment method
        let physical_cost = match result.method {
            AlignmentMethod::Direct => PhysicalCost {
                cycles: 50,
                memory_tier: 2,
                network_hops: 0,
                allocation: 0,
            },
            AlignmentMethod::CUIBridge => PhysicalCost {
                cycles: 100,
                memory_tier: 2,
                network_hops: 0,
                allocation: 64,
            },
            AlignmentMethod::LOOM => PhysicalCost {
                cycles: 150,
                memory_tier: 3,
                network_hops: 0,
                allocation: 128,
            },
            AlignmentMethod::Embedding => PhysicalCost {
                cycles: 200,
                memory_tier: 3,
                network_hops: 0,
                allocation: 512,
            },
            AlignmentMethod::Transitive => PhysicalCost {
                cycles: 300,
                memory_tier: 3,
                network_hops: 0,
                allocation: 256,
            },
            AlignmentMethod::Combined => PhysicalCost {
                cycles: 250,
                memory_tier: 3,
                network_hops: 0,
                allocation: 256,
            },
        };

        // Provenance depth from transitive hops
        let provenance_depth = result.alignment.provenance.len() as u32 + 1;

        SemanticDistance {
            conceptual,
            physical_cost,
            confidence_retention: confidence,
            provenance_depth,
        }
    }

    fn compute_ic_distance(&self, from: &IRI, to: &IRI) -> f64 {
        let ic_from = self.ic_values.get(from).copied().unwrap_or(0.0);
        let ic_to = self.ic_values.get(to).copied().unwrap_or(0.0);

        if ic_from == 0.0 || ic_to == 0.0 {
            return 0.0; // No IC data available
        }

        // Find LCA IC
        let ic_lca = if let Some(lca) = self.hierarchy_graph.lowest_common_ancestor(from, to) {
            self.ic_values.get(&lca).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        // Lin similarity: 2 * IC(LCA) / (IC(a) + IC(b))
        if ic_from + ic_to > 0.0 {
            let similarity = 2.0 * ic_lca / (ic_from + ic_to);
            1.0 - similarity // Convert similarity to distance
        } else {
            1.0
        }
    }

    fn compute_information_content(&mut self, terms: &[LoadedTerm]) {
        // Count occurrences of each term and its ancestors
        let total = terms.len() as f64;
        let mut counts: HashMap<IRI, usize> = HashMap::new();

        for term in terms {
            // Count the term itself
            *counts.entry(term.iri.clone()).or_default() += 1;

            // Count all ancestors
            let ancestors = self.hierarchy_graph.get_ancestors(&term.iri);
            for ancestor in ancestors {
                *counts.entry(ancestor).or_default() += 1;
            }
        }

        // Compute IC = -log(p(term))
        for (iri, count) in counts {
            let probability = count as f64 / total;
            let ic = -probability.ln();
            self.ic_values.insert(iri, ic);
        }
    }

    /// Check if 'to' is an ancestor of 'from' (subsumption)
    pub fn is_subtype(&self, from: &IRI, to: &IRI) -> bool {
        self.hierarchy_graph.is_ancestor(to, from)
    }

    /// Get the subsumption path if one exists
    pub fn get_subsumption_path(&self, from: &IRI, to: &IRI) -> Option<Vec<IRI>> {
        self.hierarchy_graph.find_path(from, to)
    }

    /// Find SSSOM mapping between two terms
    pub fn find_sssom_mapping(&self, from: &IRI, to: &IRI) -> Option<&SSSOMMapping> {
        self.sssom_mappings.find(from, to)
    }

    /// Clear the distance cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.distance_cache.write() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        if let Ok(cache) = self.distance_cache.read() {
            (cache.len(), 100_000)
        } else {
            (0, 100_000)
        }
    }
}

impl Default for SemanticDistanceIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_distance_zero() {
        let d = SemanticDistance::ZERO;
        assert_eq!(d.conceptual, 0.0);
        assert!(d.is_exact());
    }

    #[test]
    fn test_semantic_distance_compose() {
        let d1 = SemanticDistance::new(0.2);
        let d2 = SemanticDistance::new(0.3);
        let composed = d1.compose(d2);

        assert!((composed.conceptual - 0.5).abs() < 0.01);
        assert!(composed.confidence_retention < d1.confidence_retention);
    }

    #[test]
    fn test_physical_cost() {
        let cost = PhysicalCost::from_conceptual(0.2);
        assert_eq!(cost.memory_tier, 2); // L2 cache

        let cost = PhysicalCost::from_conceptual(0.6);
        assert_eq!(cost.memory_tier, 4); // RAM
    }

    #[test]
    fn test_sssom_index() {
        let mut index = SSSOMIndex::new();

        index.add(SSSOMMapping {
            subject_id: IRI::from_curie("CHEBI", "15365"),
            predicate_id: IRI::new("skos:exactMatch"),
            object_id: IRI::from_curie("DRUGBANK", "DB00945"),
            confidence: 0.95,
            mapping_justification: "manual curation".to_string(),
        });

        let mapping = index.find(
            &IRI::from_curie("CHEBI", "15365"),
            &IRI::from_curie("DRUGBANK", "DB00945"),
        );

        assert!(mapping.is_some());
        assert_eq!(mapping.unwrap().confidence, 0.95);
    }

    #[test]
    fn test_semantic_distance_thresholds() {
        let exact = SemanticDistance::new(0.0);
        assert!(exact.is_exact());
        assert!(exact.is_subsumption());
        assert!(exact.is_implicitly_compatible());

        let subsumption = SemanticDistance::new(0.05);
        assert!(!subsumption.is_exact());
        assert!(subsumption.is_subsumption());
        assert!(subsumption.is_implicitly_compatible());

        let implicit = SemanticDistance::new(0.25);
        assert!(!implicit.is_subsumption());
        assert!(implicit.is_implicitly_compatible());
        assert!(implicit.is_explicitly_compatible());

        let explicit = SemanticDistance::new(0.5);
        assert!(!explicit.is_implicitly_compatible());
        assert!(explicit.is_explicitly_compatible());

        let incompatible = SemanticDistance::new(0.8);
        assert!(!incompatible.is_explicitly_compatible());
    }
}
