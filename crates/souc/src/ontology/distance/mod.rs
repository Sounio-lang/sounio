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
use super::loader::{LoadedTerm, IRI};

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

/// IC-based similarity metric to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ICSimilarityMetric {
    /// Lin similarity: 2*IC(LCA) / (IC(a) + IC(b))
    /// Normalized to [0, 1], commonly used
    #[default]
    Lin,
    /// Resnik similarity: IC(LCA)
    /// Not normalized, represents shared information
    Resnik,
    /// Jiang-Conrath distance: IC(a) + IC(b) - 2*IC(LCA)
    /// Lower is more similar
    JiangConrath,
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

    /// Which IC-based similarity metric to use
    pub ic_metric: ICSimilarityMetric,

    /// Whether to use intrinsic IC as fallback when corpus IC unavailable
    pub use_intrinsic_ic_fallback: bool,
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
            ic_metric: ICSimilarityMetric::Lin,
            use_intrinsic_ic_fallback: true,
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

/// Cached term text information for embedding-based similarity
#[derive(Debug, Clone)]
pub struct TermText {
    /// Human-readable label
    pub label: String,
    /// Definition text (if available)
    pub definition: Option<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
}

impl TermText {
    /// Create from a loaded term
    pub fn from_loaded_term(term: &LoadedTerm) -> Self {
        Self {
            label: term.label.clone(),
            definition: term.definition.clone(),
            synonyms: term.synonyms.iter().map(|s| s.text.clone()).collect(),
        }
    }

    /// Concatenate all text for embedding
    pub fn to_embedding_text(&self) -> String {
        let mut parts = vec![self.label.clone()];
        if let Some(ref def) = self.definition {
            parts.push(def.clone());
        }
        for syn in &self.synonyms {
            parts.push(syn.clone());
        }
        parts.join(" . ")
    }
}

/// Index structure for fast semantic distance lookup
pub struct SemanticDistanceIndex {
    /// Graph structure for path-based distance
    hierarchy_graph: path::HierarchyGraph,

    /// Information content values for IC-based similarity (corpus-based)
    ic_values: HashMap<IRI, f64>,

    /// Full IC index with intrinsic IC fallback support
    ic_index: information_content::ICIndex,

    /// SSSOM mappings for cross-ontology distance
    sssom_mappings: SSSOMIndex,

    /// Unified alignment index (SSSOM + CUI + LOOM + embeddings)
    alignment_index: Option<AlignmentIndex>,

    /// Embedding space for geometric distance (optional)
    embedding_space: Option<super::embedding::EmbeddingSpace>,

    /// Term text cache for label-based similarity fallback
    /// Used when full embedding space is not available
    term_texts: HashMap<IRI, TermText>,

    /// Precomputed distances for common pairs
    distance_cache: RwLock<lru::LruCache<(IRI, IRI), SemanticDistance>>,

    /// Configuration
    config: DistanceConfig,

    /// Maximum IC value for normalization (Resnik)
    max_ic: f64,
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
            ic_index: information_content::ICIndex::new(),
            sssom_mappings: SSSOMIndex::new(),
            alignment_index: None,
            embedding_space: None,
            term_texts: HashMap::new(),
            distance_cache: RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(100_000).unwrap(),
            )),
            config,
            max_ic: 0.0,
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

    /// Get a reference to the hierarchy graph for path-based operations
    pub fn hierarchy_graph(&self) -> &path::HierarchyGraph {
        &self.hierarchy_graph
    }

    /// Build index from loaded terms
    pub fn build_from_terms(&mut self, terms: &[LoadedTerm]) {
        // Build hierarchy graph and term text cache
        for term in terms {
            self.hierarchy_graph.add_term(&term.iri, &term.superclasses);

            // Cache term text for embedding-based similarity fallback
            if self.config.use_embeddings {
                self.term_texts
                    .insert(term.iri.clone(), TermText::from_loaded_term(term));
            }
        }

        // Compute information content (both corpus-based and intrinsic)
        self.compute_information_content(terms);

        // Compute intrinsic IC as fallback (based on hierarchy structure)
        if self.config.use_intrinsic_ic_fallback {
            self.ic_index.compute_intrinsic_ic(&self.hierarchy_graph);
        }
    }

    /// Add a single term to the index
    pub fn add_term(&mut self, term: &LoadedTerm) {
        self.hierarchy_graph.add_term(&term.iri, &term.superclasses);

        // Cache term text for embedding-based similarity fallback
        if self.config.use_embeddings {
            self.term_texts
                .insert(term.iri.clone(), TermText::from_loaded_term(term));
        }
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
    ///
    /// This function uses a multi-tier fallback strategy:
    /// 1. Full embedding space (if loaded) - highest quality
    /// 2. Label/definition text similarity - fallback when embeddings not loaded
    ///
    /// The text similarity uses a hash-based embedding approach that captures
    /// lexical overlap between term labels and definitions.
    fn compute_embedding_distance(&self, from: &IRI, to: &IRI) -> Option<f64> {
        if !self.config.use_embeddings {
            return None;
        }

        // Try full embedding space first (highest quality)
        if let Some(ref space) = self.embedding_space {
            if let Ok(distance) = space.embedding_distance(from, to) {
                return Some(distance);
            }
        }

        // Fallback: compute text-based similarity from cached term texts
        self.compute_text_similarity(from, to)
    }

    /// Compute text-based similarity between two terms using their labels/definitions
    ///
    /// Uses a hash-based embedding approach similar to the TextualGenerator,
    /// computing cosine similarity between the embedded text representations.
    fn compute_text_similarity(&self, from: &IRI, to: &IRI) -> Option<f64> {
        let from_text = self.term_texts.get(from)?;
        let to_text = self.term_texts.get(to)?;

        let from_embedding = self.embed_text(&from_text.to_embedding_text());
        let to_embedding = self.embed_text(&to_text.to_embedding_text());

        // Compute cosine similarity
        let similarity = self.cosine_similarity(&from_embedding, &to_embedding);

        // Convert similarity [-1, 1] to distance [0, 1]
        // sim=1 -> dist=0, sim=0 -> dist=0.5, sim=-1 -> dist=1
        Some(((1.0 - similarity) / 2.0).clamp(0.0, 1.0))
    }

    /// Generate a hash-based text embedding vector
    ///
    /// This is a lightweight embedding approach that captures lexical features.
    /// For production use with high accuracy requirements, this would be replaced
    /// with a proper sentence transformer (e.g., via ONNX runtime or external API).
    fn embed_text(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Use 128 dimensions for lightweight text embeddings
        const DIMENSIONS: usize = 128;
        let mut vector = vec![0.0f32; DIMENSIONS];

        // Tokenize and embed each word with position-weighted contribution
        for (word_idx, word) in text.split_whitespace().enumerate() {
            let normalized = word.to_lowercase();

            let mut hasher = DefaultHasher::new();
            normalized.hash(&mut hasher);
            let hash = hasher.finish();

            // Spread the word's contribution across dimensions
            for j in 0..DIMENSIONS {
                let idx = (hash.wrapping_add(j as u64) as usize) % DIMENSIONS;
                let sign = if (hash >> (j % 64)) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                // Earlier words (label) have higher weight than later words (synonyms)
                let magnitude = 1.0 / (word_idx as f32 + 1.0).sqrt();

                vector[idx] += sign * magnitude;
            }
        }

        // Normalize to unit length
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut vector {
                *x /= norm;
            }
        }

        vector
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        // Vectors are already normalized, so just return dot product
        dot_product as f64
    }

    /// Get term text for an IRI (public API for introspection)
    pub fn get_term_text(&self, iri: &IRI) -> Option<&TermText> {
        self.term_texts.get(iri)
    }

    /// Check if term texts are available for embedding similarity
    pub fn has_term_texts(&self) -> bool {
        !self.term_texts.is_empty()
    }

    /// Get the number of cached term texts
    pub fn term_text_count(&self) -> usize {
        self.term_texts.len()
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

    /// Get IC value for a term, with intrinsic IC fallback if configured
    fn get_ic_value(&self, iri: &IRI) -> Option<f64> {
        // Try corpus-based IC first
        if let Some(ic) = self.ic_values.get(iri).copied() {
            return Some(ic);
        }

        // Try intrinsic IC fallback if enabled
        if self.config.use_intrinsic_ic_fallback {
            if let Some(ic) = self.ic_index.get_intrinsic_ic(iri) {
                return Some(ic);
            }
        }

        None
    }

    fn compute_ic_distance(&self, from: &IRI, to: &IRI) -> f64 {
        // Get IC values with fallback support
        let ic_from = match self.get_ic_value(from) {
            Some(ic) => ic,
            None => return 0.0, // No IC data, signal unavailable
        };

        let ic_to = match self.get_ic_value(to) {
            Some(ic) => ic,
            None => return 0.0, // No IC data, signal unavailable
        };

        // Find LCA and its IC
        let lca = match self.hierarchy_graph.lowest_common_ancestor(from, to) {
            Some(lca) => lca,
            None => {
                // No common ancestor found - use path-based fallback
                // Return high distance (dissimilar)
                return 1.0;
            }
        };

        let ic_lca = self.get_ic_value(&lca).unwrap_or(0.0);

        // Compute distance based on configured metric
        match self.config.ic_metric {
            ICSimilarityMetric::Lin => {
                // Lin similarity: 2 * IC(LCA) / (IC(a) + IC(b))
                let denominator = ic_from + ic_to;
                if denominator == 0.0 {
                    return 0.0; // Both are root concepts, identical
                }
                let similarity = 2.0 * ic_lca / denominator;
                1.0 - similarity.clamp(0.0, 1.0) // Convert to distance [0, 1]
            }
            ICSimilarityMetric::Resnik => {
                // Resnik similarity: IC(LCA)
                // Need to normalize to [0, 1] using max_ic
                if self.max_ic == 0.0 {
                    return 0.0;
                }
                let normalized_similarity = ic_lca / self.max_ic;
                1.0 - normalized_similarity.clamp(0.0, 1.0) // Convert to distance
            }
            ICSimilarityMetric::JiangConrath => {
                // Jiang-Conrath distance: IC(a) + IC(b) - 2*IC(LCA)
                // Already a distance metric, but unbounded - need to normalize
                let jc_distance = ic_from + ic_to - 2.0 * ic_lca;
                // Normalize by 2*max_ic (theoretical maximum)
                if self.max_ic == 0.0 {
                    return 0.0;
                }
                (jc_distance / (2.0 * self.max_ic)).clamp(0.0, 1.0)
            }
        }
    }

    /// Compute Resnik similarity directly: IC(LCA)
    /// Returns None if IC data unavailable
    pub fn resnik_similarity(&self, from: &IRI, to: &IRI) -> Option<f64> {
        let lca = self.hierarchy_graph.lowest_common_ancestor(from, to)?;
        self.get_ic_value(&lca)
    }

    /// Compute Lin similarity directly: 2*IC(LCA) / (IC(a) + IC(b))
    /// Returns None if IC data unavailable
    pub fn lin_similarity(&self, from: &IRI, to: &IRI) -> Option<f64> {
        let ic_from = self.get_ic_value(from)?;
        let ic_to = self.get_ic_value(to)?;
        let lca = self.hierarchy_graph.lowest_common_ancestor(from, to)?;
        let ic_lca = self.get_ic_value(&lca).unwrap_or(0.0);

        let denominator = ic_from + ic_to;
        if denominator == 0.0 {
            return Some(1.0); // Both are root concepts
        }

        Some(2.0 * ic_lca / denominator)
    }

    /// Compute Jiang-Conrath distance directly: IC(a) + IC(b) - 2*IC(LCA)
    /// Returns None if IC data unavailable
    pub fn jiang_conrath_distance(&self, from: &IRI, to: &IRI) -> Option<f64> {
        let ic_from = self.get_ic_value(from)?;
        let ic_to = self.get_ic_value(to)?;
        let lca = self.hierarchy_graph.lowest_common_ancestor(from, to)?;
        let ic_lca = self.get_ic_value(&lca).unwrap_or(0.0);

        Some(ic_from + ic_to - 2.0 * ic_lca)
    }

    /// Get IC value for a term (public API)
    pub fn get_ic(&self, iri: &IRI) -> Option<f64> {
        self.get_ic_value(iri)
    }

    /// Check if IC data is available for distance calculations
    pub fn has_ic_data(&self) -> bool {
        !self.ic_values.is_empty() || self.ic_index.stats().num_intrinsic > 0
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
        self.max_ic = 0.0;
        for (iri, count) in &counts {
            let probability = *count as f64 / total;
            let ic = -probability.ln();
            self.ic_values.insert(iri.clone(), ic);

            // Track max IC for Resnik normalization
            if ic > self.max_ic {
                self.max_ic = ic;
            }

            // Also populate the IC index for fallback support
            self.ic_index.add_frequency(iri.clone(), *count as u64);
        }

        // Compute IC values in the IC index
        self.ic_index
            .compute_ic(&information_content::ICConfig::default());
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

    fn make_test_terms() -> Vec<LoadedTerm> {
        use super::super::loader::OntologyId;

        // Create a simple hierarchy:
        //        Thing
        //       /     \
        //    Animal   Plant
        //    /    \
        //  Dog    Cat
        let thing = IRI::new("http://example.org/Thing");
        let animal = IRI::new("http://example.org/Animal");
        let plant = IRI::new("http://example.org/Plant");
        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        vec![
            LoadedTerm {
                iri: thing.clone(),
                label: "Thing".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![],
                subclasses: vec![animal.clone(), plant.clone()],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: None,
                synonyms: vec![],
                hierarchy_depth: 0,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
            LoadedTerm {
                iri: animal.clone(),
                label: "Animal".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![thing.clone()],
                subclasses: vec![dog.clone(), cat.clone()],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: None,
                synonyms: vec![],
                hierarchy_depth: 1,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
            LoadedTerm {
                iri: plant.clone(),
                label: "Plant".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![thing.clone()],
                subclasses: vec![],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: None,
                synonyms: vec![],
                hierarchy_depth: 1,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
            LoadedTerm {
                iri: dog.clone(),
                label: "Dog".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![animal.clone()],
                subclasses: vec![],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: None,
                synonyms: vec![],
                hierarchy_depth: 2,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
            LoadedTerm {
                iri: cat.clone(),
                label: "Cat".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![animal.clone()],
                subclasses: vec![],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: None,
                synonyms: vec![],
                hierarchy_depth: 2,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
        ]
    }

    #[test]
    fn test_ic_distance_lin() {
        let terms = make_test_terms();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        // Lin similarity should work
        let lin_sim = index.lin_similarity(&dog, &cat);
        assert!(lin_sim.is_some());
        let sim = lin_sim.unwrap();
        assert!(
            sim > 0.0 && sim < 1.0,
            "Lin similarity should be between 0 and 1"
        );

        // IC distance should contribute to overall distance
        let ic_distance = index.compute_ic_distance(&dog, &cat);
        assert!(
            ic_distance > 0.0,
            "IC distance should be positive for non-identical terms"
        );
        assert!(
            ic_distance < 1.0,
            "IC distance should be < 1 for related terms"
        );
    }

    #[test]
    fn test_ic_distance_resnik() {
        let terms = make_test_terms();
        let mut config = DistanceConfig::default();
        config.ic_metric = ICSimilarityMetric::Resnik;

        let mut index = SemanticDistanceIndex::with_config(config);
        index.build_from_terms(&terms);

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        // Resnik similarity (IC of LCA)
        let resnik = index.resnik_similarity(&dog, &cat);
        assert!(resnik.is_some());
        assert!(
            resnik.unwrap() > 0.0,
            "LCA (Animal) should have positive IC"
        );

        // Distance should be normalized
        let ic_distance = index.compute_ic_distance(&dog, &cat);
        assert!(ic_distance >= 0.0 && ic_distance <= 1.0);
    }

    #[test]
    fn test_ic_distance_jiang_conrath() {
        let terms = make_test_terms();
        let mut config = DistanceConfig::default();
        config.ic_metric = ICSimilarityMetric::JiangConrath;

        let mut index = SemanticDistanceIndex::with_config(config);
        index.build_from_terms(&terms);

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        // Jiang-Conrath distance
        let jc = index.jiang_conrath_distance(&dog, &cat);
        assert!(jc.is_some());
        assert!(jc.unwrap() >= 0.0, "JC distance should be non-negative");

        // Normalized distance
        let ic_distance = index.compute_ic_distance(&dog, &cat);
        assert!(ic_distance >= 0.0 && ic_distance <= 1.0);
    }

    #[test]
    fn test_ic_fallback_unavailable() {
        // Test with empty index (no IC data)
        let index = SemanticDistanceIndex::new();

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        // Should return 0.0 when IC data unavailable (signals to use other metrics)
        let ic_distance = index.compute_ic_distance(&dog, &cat);
        assert_eq!(
            ic_distance, 0.0,
            "Should return 0.0 when no IC data available"
        );

        assert!(!index.has_ic_data());
    }

    #[test]
    fn test_ic_contributes_to_distance() {
        let terms = make_test_terms();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        assert!(index.has_ic_data());

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        // Overall distance should be computed (combining path and IC)
        let distance = index.distance(&dog, &cat);
        assert!(distance.conceptual > 0.0);
        assert!(distance.conceptual < 1.0);
    }

    #[test]
    fn test_ic_identical_terms() {
        let terms = make_test_terms();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        let dog = IRI::new("http://example.org/Dog");

        // Same term should have zero distance
        let distance = index.distance(&dog, &dog);
        assert_eq!(distance.conceptual, 0.0);
    }

    #[test]
    fn test_ic_metric_enum() {
        assert_eq!(ICSimilarityMetric::default(), ICSimilarityMetric::Lin);
    }

    // ============== Embedding Similarity Tests ==============

    fn make_test_terms_with_definitions() -> Vec<LoadedTerm> {
        use super::super::loader::{OntologyId, Synonym, SynonymScope};

        let heart = IRI::new("http://example.org/Heart");
        let cardiac = IRI::new("http://example.org/Cardiac");
        let liver = IRI::new("http://example.org/Liver");
        let software = IRI::new("http://example.org/Software");

        vec![
            LoadedTerm {
                iri: heart.clone(),
                label: "Heart".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![],
                subclasses: vec![],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: Some(
                    "A muscular organ that pumps blood through the circulatory system".to_string(),
                ),
                synonyms: vec![Synonym {
                    text: "cardiac organ".to_string(),
                    scope: SynonymScope::Exact,
                }],
                hierarchy_depth: 0,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
            LoadedTerm {
                iri: cardiac.clone(),
                label: "Cardiac".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![],
                subclasses: vec![],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: Some("Relating to or affecting the heart".to_string()),
                synonyms: vec![],
                hierarchy_depth: 0,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
            LoadedTerm {
                iri: liver.clone(),
                label: "Liver".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![],
                subclasses: vec![],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: Some(
                    "A large organ in the abdomen that produces bile and processes nutrients"
                        .to_string(),
                ),
                synonyms: vec![Synonym {
                    text: "hepatic organ".to_string(),
                    scope: SynonymScope::Exact,
                }],
                hierarchy_depth: 0,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
            LoadedTerm {
                iri: software.clone(),
                label: "Software".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![],
                subclasses: vec![],
                properties: vec![],
                restrictions: vec![],
                xrefs: vec![],
                definition: Some("Computer programs and related data".to_string()),
                synonyms: vec![Synonym {
                    text: "application".to_string(),
                    scope: SynonymScope::Broad,
                }],
                hierarchy_depth: 0,
                information_content: 0.0,
                is_obsolete: false,
                replaced_by: None,
            },
        ]
    }

    #[test]
    fn test_term_text_caching() {
        let terms = make_test_terms_with_definitions();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        // Term texts should be cached
        assert!(index.has_term_texts());
        assert_eq!(index.term_text_count(), 4);

        // Check that term text is retrievable
        let heart = IRI::new("http://example.org/Heart");
        let term_text = index.get_term_text(&heart);
        assert!(term_text.is_some());

        let text = term_text.unwrap();
        assert_eq!(text.label, "Heart");
        assert!(text.definition.is_some());
        assert!(!text.synonyms.is_empty());
    }

    #[test]
    fn test_embedding_text_generation() {
        let terms = make_test_terms_with_definitions();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        let heart = IRI::new("http://example.org/Heart");
        let term_text = index.get_term_text(&heart).unwrap();

        let embedding_text = term_text.to_embedding_text();
        assert!(embedding_text.contains("Heart"));
        assert!(embedding_text.contains("muscular organ"));
        assert!(embedding_text.contains("cardiac organ"));
    }

    #[test]
    fn test_text_similarity_related_terms() {
        let terms = make_test_terms_with_definitions();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        let heart = IRI::new("http://example.org/Heart");
        let cardiac = IRI::new("http://example.org/Cardiac");
        let software = IRI::new("http://example.org/Software");

        // Heart and Cardiac should have embedding similarity (shared word "heart")
        let heart_cardiac_dist = index.compute_embedding_distance(&heart, &cardiac);
        assert!(heart_cardiac_dist.is_some());
        let hc_dist = heart_cardiac_dist.unwrap();

        // Heart and Software should have low similarity (unrelated domains)
        let heart_software_dist = index.compute_embedding_distance(&heart, &software);
        assert!(heart_software_dist.is_some());
        let hs_dist = heart_software_dist.unwrap();

        // Heart-Cardiac should be closer than Heart-Software
        assert!(
            hc_dist < hs_dist,
            "Heart-Cardiac distance ({}) should be less than Heart-Software distance ({})",
            hc_dist,
            hs_dist
        );
    }

    #[test]
    fn test_text_similarity_organs() {
        let terms = make_test_terms_with_definitions();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        let heart = IRI::new("http://example.org/Heart");
        let liver = IRI::new("http://example.org/Liver");
        let software = IRI::new("http://example.org/Software");

        // Heart and Liver are both organs - should have some similarity
        let heart_liver_dist = index.compute_embedding_distance(&heart, &liver);
        assert!(heart_liver_dist.is_some());
        let hl_dist = heart_liver_dist.unwrap();

        // Heart and Software should have low similarity
        let heart_software_dist = index.compute_embedding_distance(&heart, &software);
        assert!(heart_software_dist.is_some());
        let hs_dist = heart_software_dist.unwrap();

        // Heart-Liver (both organs) should be closer than Heart-Software
        assert!(
            hl_dist < hs_dist,
            "Heart-Liver distance ({}) should be less than Heart-Software distance ({})",
            hl_dist,
            hs_dist
        );
    }

    #[test]
    fn test_embedding_distance_bounds() {
        let terms = make_test_terms_with_definitions();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        let heart = IRI::new("http://example.org/Heart");
        let cardiac = IRI::new("http://example.org/Cardiac");

        let distance = index.compute_embedding_distance(&heart, &cardiac);
        assert!(distance.is_some());
        let dist = distance.unwrap();

        // Distance should be in [0, 1] range
        assert!(
            dist >= 0.0 && dist <= 1.0,
            "Distance {} should be in [0, 1]",
            dist
        );
    }

    #[test]
    fn test_embedding_contributes_to_overall_distance() {
        let terms = make_test_terms_with_definitions();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        let heart = IRI::new("http://example.org/Heart");
        let cardiac = IRI::new("http://example.org/Cardiac");

        // Get the overall semantic distance
        let distance = index.distance(&heart, &cardiac);

        // Distance should be computed (not MAX or some default)
        assert!(
            distance.conceptual < 1.0,
            "Terms with shared content should have distance < 1.0"
        );
    }

    #[test]
    fn test_embedding_disabled() {
        let terms = make_test_terms_with_definitions();
        let mut config = DistanceConfig::default();
        config.use_embeddings = false;

        let mut index = SemanticDistanceIndex::with_config(config);
        index.build_from_terms(&terms);

        // Term texts should not be cached when embeddings are disabled
        assert!(!index.has_term_texts());
        assert_eq!(index.term_text_count(), 0);

        let heart = IRI::new("http://example.org/Heart");
        let cardiac = IRI::new("http://example.org/Cardiac");

        // Embedding distance should return None
        let distance = index.compute_embedding_distance(&heart, &cardiac);
        assert!(
            distance.is_none(),
            "Embedding distance should be None when embeddings disabled"
        );
    }

    #[test]
    fn test_embedding_distance_unknown_term() {
        let terms = make_test_terms_with_definitions();
        let mut index = SemanticDistanceIndex::new();
        index.build_from_terms(&terms);

        let heart = IRI::new("http://example.org/Heart");
        let unknown = IRI::new("http://example.org/UnknownTerm");

        // Distance to unknown term should return None
        let distance = index.compute_embedding_distance(&heart, &unknown);
        assert!(
            distance.is_none(),
            "Distance to unknown term should be None"
        );
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        let index = SemanticDistanceIndex::new();

        // Test with normalized vectors
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let c = vec![1.0f32, 0.0, 0.0];

        // Orthogonal vectors should have 0 similarity
        let sim_ab = index.cosine_similarity(&a, &b);
        assert!(
            sim_ab.abs() < 0.001,
            "Orthogonal vectors should have ~0 similarity"
        );

        // Identical vectors should have 1.0 similarity
        let sim_ac = index.cosine_similarity(&a, &c);
        assert!(
            (sim_ac - 1.0).abs() < 0.001,
            "Identical vectors should have ~1.0 similarity"
        );
    }
}
