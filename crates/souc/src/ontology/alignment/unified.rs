//! Unified Alignment Index
//!
//! Combines multiple alignment sources (SSSOM, UMLS, LOOM, embeddings) into
//! a single queryable index for cross-ontology type compatibility.
//!
//! # Query Priority
//!
//! When looking up alignments between two terms, sources are queried in order:
//!
//! 1. Direct SSSOM mappings (highest confidence)
//! 2. UMLS CUI bridging
//! 3. BioPortal LOOM mappings
//! 4. Embedding similarity (fallback)
//! 5. Transitive inference (computed on-demand)
//!
//! # Caching
//!
//! Results are cached to avoid repeated computation, especially for
//! transitive alignments which require graph traversal.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ontology::distance::sssom::{MappingPredicate, SSSOMIndex, SSSOMMapping};
use crate::ontology::loader::IRI;

use super::cui::CUIBridge;
use super::loom::LOOMClient;
use super::{
    Alignment, AlignmentSource, AlignmentStats, MAX_TRANSITIVE_HOPS, MIN_MAPPING_CONFIDENCE,
};

/// How an alignment was found
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentMethod {
    /// Direct lookup (single hop)
    Direct,
    /// Via UMLS CUI bridging
    CUIBridge,
    /// Via BioPortal LOOM
    LOOM,
    /// Embedding similarity
    Embedding,
    /// Transitive path (multiple hops)
    Transitive,
    /// Combined from multiple sources
    Combined,
}

/// Result of an alignment lookup
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// The alignment found
    pub alignment: Alignment,
    /// How it was found
    pub method: AlignmentMethod,
    /// Alternative alignments (if multiple found)
    pub alternatives: Vec<Alignment>,
}

impl AlignmentResult {
    /// Get the best confidence across all results
    pub fn best_confidence(&self) -> f64 {
        let mut best = self.alignment.effective_confidence();
        for alt in &self.alternatives {
            best = best.max(alt.effective_confidence());
        }
        best
    }

    /// Get minimum distance
    pub fn min_distance(&self) -> f64 {
        let mut min = self.alignment.to_distance();
        for alt in &self.alternatives {
            min = min.min(alt.to_distance());
        }
        min
    }
}

/// Unified cross-ontology alignment index
pub struct AlignmentIndex {
    /// SSSOM mappings
    sssom: SSSOMIndex,
    /// UMLS CUI bridge
    cui_bridge: CUIBridge,
    /// BioPortal LOOM mappings
    loom: LOOMClient,
    /// Embedding similarity threshold
    embedding_threshold: f64,
    /// Cache for computed alignments
    cache: HashMap<(IRI, IRI), Option<AlignmentResult>>,
    /// Enable transitive inference
    transitive_enabled: bool,
    /// Statistics
    stats: AlignmentStats,
}

impl AlignmentIndex {
    /// Create an empty alignment index
    pub fn new() -> Self {
        Self {
            sssom: SSSOMIndex::new(),
            cui_bridge: CUIBridge::new(),
            loom: LOOMClient::new(),
            embedding_threshold: 0.85,
            cache: HashMap::new(),
            transitive_enabled: true,
            stats: AlignmentStats::default(),
        }
    }

    /// Set SSSOM index
    pub fn with_sssom(mut self, sssom: SSSOMIndex) -> Self {
        self.sssom = sssom;
        self
    }

    /// Set CUI bridge
    pub fn with_cui_bridge(mut self, bridge: CUIBridge) -> Self {
        self.cui_bridge = bridge;
        self
    }

    /// Set LOOM client
    pub fn with_loom(mut self, loom: LOOMClient) -> Self {
        self.loom = loom;
        self
    }

    /// Set embedding similarity threshold
    pub fn with_embedding_threshold(mut self, threshold: f64) -> Self {
        self.embedding_threshold = threshold;
        self
    }

    /// Enable/disable transitive inference
    pub fn with_transitive(mut self, enabled: bool) -> Self {
        self.transitive_enabled = enabled;
        self
    }

    /// Look up alignment between two terms
    pub fn find_alignment(&self, source: &IRI, target: &IRI) -> Option<AlignmentResult> {
        // Check cache first
        let cache_key = Self::cache_key(source, target);
        if let Some(cached) = self.cache.get(&cache_key) {
            return cached.clone();
        }

        // Same term = identical
        if source == target {
            return Some(AlignmentResult {
                alignment: Alignment::direct(
                    source.clone(),
                    target.clone(),
                    MappingPredicate::ExactMatch,
                    1.0,
                    AlignmentSource::Manual,
                ),
                method: AlignmentMethod::Direct,
                alternatives: Vec::new(),
            });
        }

        // Try each source in priority order
        let mut results = Vec::new();

        // 1. SSSOM
        if let Some(alignment) = self.find_sssom_alignment(source, target) {
            results.push((alignment, AlignmentMethod::Direct));
        }

        // 2. UMLS CUI
        if let Some(alignment) = self.find_cui_alignment(source, target) {
            results.push((alignment, AlignmentMethod::CUIBridge));
        }

        // 3. LOOM
        if let Some(alignment) = self.find_loom_alignment(source, target) {
            results.push((alignment, AlignmentMethod::LOOM));
        }

        // 4. Transitive (if enabled and no direct mapping found)
        if self.transitive_enabled
            && results.is_empty()
            && let Some(alignment) = self.find_transitive_alignment(source, target)
        {
            results.push((alignment, AlignmentMethod::Transitive));
        }

        if results.is_empty() {
            return None;
        }

        // Sort by effective confidence, take best
        results.sort_by(|a, b| {
            b.0.effective_confidence()
                .partial_cmp(&a.0.effective_confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let (best, method) = results.remove(0);
        let alternatives = results.into_iter().map(|(a, _)| a).collect();

        Some(AlignmentResult {
            alignment: best,
            method,
            alternatives,
        })
    }

    /// Find SSSOM-based alignment
    fn find_sssom_alignment(&self, source: &IRI, target: &IRI) -> Option<Alignment> {
        // Check direct mapping
        for mapping in self.sssom.get_mappings_from(source) {
            if &mapping.object_id == target && mapping.confidence >= MIN_MAPPING_CONFIDENCE {
                return Some(Alignment::direct(
                    source.clone(),
                    target.clone(),
                    mapping.predicate,
                    mapping.confidence,
                    AlignmentSource::SSSOM,
                ));
            }
        }

        // Check reverse mapping
        for mapping in self.sssom.get_mappings_to(source) {
            if &mapping.subject_id == target && mapping.confidence >= MIN_MAPPING_CONFIDENCE {
                return Some(Alignment::direct(
                    source.clone(),
                    target.clone(),
                    mapping.predicate,
                    mapping.confidence,
                    AlignmentSource::SSSOM,
                ));
            }
        }

        None
    }

    /// Find CUI-based alignment
    fn find_cui_alignment(&self, source: &IRI, target: &IRI) -> Option<Alignment> {
        if let Some((distance, confidence, cui)) = self.cui_bridge.cui_distance(source, target)
            && confidence >= MIN_MAPPING_CONFIDENCE
        {
            return Some(Alignment::via_cui(
                source.clone(),
                target.clone(),
                cui,
                confidence,
            ));
        }
        None
    }

    /// Find LOOM-based alignment
    fn find_loom_alignment(&self, source: &IRI, target: &IRI) -> Option<Alignment> {
        if let Some((distance, confidence)) = self.loom.loom_distance(source, target)
            && confidence >= MIN_MAPPING_CONFIDENCE
        {
            return Some(Alignment::direct(
                source.clone(),
                target.clone(),
                MappingPredicate::CloseMatch,
                confidence,
                AlignmentSource::LOOM,
            ));
        }
        None
    }

    /// Find transitive alignment via BFS
    fn find_transitive_alignment(&self, source: &IRI, target: &IRI) -> Option<Alignment> {
        // BFS to find path from source to target
        let mut visited: HashSet<IRI> = HashSet::new();
        let mut queue: VecDeque<(IRI, Vec<IRI>, f64)> = VecDeque::new();

        queue.push_back((source.clone(), vec![source.clone()], 1.0));
        visited.insert(source.clone());

        while let Some((current, path, confidence)) = queue.pop_front() {
            if path.len() > MAX_TRANSITIVE_HOPS {
                continue;
            }

            // Get all neighbors
            let neighbors = self.get_aligned_terms(&current);

            for (neighbor, edge_confidence) in neighbors {
                if &neighbor == target {
                    // Found path to target
                    let mut full_path = path.clone();
                    full_path.push(neighbor.clone());

                    let total_confidence = confidence * edge_confidence * 0.9; // Penalty for each hop

                    return Some(Alignment::transitive(
                        source.clone(),
                        target.clone(),
                        total_confidence,
                        full_path,
                    ));
                }

                if !visited.contains(&neighbor) {
                    visited.insert(neighbor.clone());
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    let new_confidence = confidence * edge_confidence * 0.9;
                    queue.push_back((neighbor, new_path, new_confidence));
                }
            }
        }

        None
    }

    /// Get all terms aligned with a given term
    fn get_aligned_terms(&self, term: &IRI) -> Vec<(IRI, f64)> {
        let mut aligned = Vec::new();

        // From SSSOM
        for mapping in self.sssom.get_mappings_from(term) {
            aligned.push((mapping.object_id.clone(), mapping.effective_confidence()));
        }
        for mapping in self.sssom.get_mappings_to(term) {
            aligned.push((mapping.subject_id.clone(), mapping.effective_confidence()));
        }

        // From CUI bridge
        for (equiv, _, confidence) in self.cui_bridge.find_equivalents(term) {
            aligned.push((equiv, confidence));
        }

        // From LOOM
        for mapping in self.loom.get_all_mappings(term) {
            let target = if &mapping.source_class == term {
                mapping.target_class.clone()
            } else {
                mapping.source_class.clone()
            };
            aligned.push((
                target,
                mapping.confidence * mapping.mapping_source.reliability(),
            ));
        }

        // Deduplicate
        let mut seen = HashSet::new();
        aligned.retain(|(iri, _)| seen.insert(iri.clone()));

        aligned
    }

    /// Compute alignment distance between two terms
    pub fn alignment_distance(&self, source: &IRI, target: &IRI) -> f64 {
        if let Some(result) = self.find_alignment(source, target) {
            result.min_distance()
        } else {
            // No alignment found - return high distance
            0.95
        }
    }

    /// Check if two terms are aligned above a confidence threshold
    pub fn are_aligned(&self, source: &IRI, target: &IRI, min_confidence: f64) -> bool {
        if let Some(result) = self.find_alignment(source, target) {
            result.best_confidence() >= min_confidence
        } else {
            false
        }
    }

    /// Get all alignments for a term
    pub fn get_alignments(&self, term: &IRI) -> Vec<Alignment> {
        let mut alignments = Vec::new();

        // SSSOM
        for mapping in self.sssom.get_mappings_from(term) {
            alignments.push(Alignment::direct(
                term.clone(),
                mapping.object_id.clone(),
                mapping.predicate,
                mapping.confidence,
                AlignmentSource::SSSOM,
            ));
        }
        for mapping in self.sssom.get_mappings_to(term) {
            alignments.push(Alignment::direct(
                mapping.subject_id.clone(),
                term.clone(),
                mapping.predicate,
                mapping.confidence,
                AlignmentSource::SSSOM,
            ));
        }

        // CUI
        for (equiv, cui, confidence) in self.cui_bridge.find_equivalents(term) {
            alignments.push(Alignment::via_cui(term.clone(), equiv, cui, confidence));
        }

        // LOOM
        for mapping in self.loom.get_mappings_from(term) {
            alignments.push(Alignment::direct(
                term.clone(),
                mapping.target_class.clone(),
                mapping.relation,
                mapping.confidence,
                AlignmentSource::LOOM,
            ));
        }

        alignments
    }

    /// Cache key for alignment lookup
    fn cache_key(a: &IRI, b: &IRI) -> (IRI, IRI) {
        if a.as_str() <= b.as_str() {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        }
    }

    /// Clear the alignment cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &AlignmentStats {
        &self.stats
    }

    /// Recompute statistics
    pub fn recompute_stats(&mut self) {
        let sssom_stats = self.sssom.stats();
        let cui_stats = self.cui_bridge.stats();
        let loom_stats = self.loom.stats();

        self.stats = AlignmentStats {
            total_alignments: sssom_stats.total_mappings
                + cui_stats.total_mappings
                + loom_stats.total_mappings,
            sssom_alignments: sssom_stats.total_mappings,
            umls_alignments: cui_stats.total_mappings,
            loom_alignments: loom_stats.total_mappings,
            embedding_alignments: 0,  // TODO: add when embeddings integrated
            transitive_alignments: 0, // Computed on-demand
            unique_terms: sssom_stats.unique_subjects
                + sssom_stats.unique_objects
                + cui_stats.unique_terms,
            ontology_pairs: loom_stats.ontology_pairs,
        };
    }

    /// Add SSSOM mapping directly
    pub fn add_sssom_mapping(&mut self, mapping: SSSOMMapping) {
        self.sssom.add(mapping);
    }

    /// Add CUI mapping directly
    pub fn add_cui_mapping(&mut self, term: IRI, cui: String) {
        self.cui_bridge.add_mapping(term, cui);
    }

    /// Get access to SSSOM index
    pub fn sssom(&self) -> &SSSOMIndex {
        &self.sssom
    }

    /// Get access to CUI bridge
    pub fn cui_bridge(&self) -> &CUIBridge {
        &self.cui_bridge
    }

    /// Get access to LOOM client
    pub fn loom(&self) -> &LOOMClient {
        &self.loom
    }
}

impl Default for AlignmentIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating an AlignmentIndex
pub struct AlignmentIndexBuilder {
    index: AlignmentIndex,
}

impl AlignmentIndexBuilder {
    pub fn new() -> Self {
        Self {
            index: AlignmentIndex::new(),
        }
    }

    /// Load SSSOM file
    pub fn load_sssom(
        mut self,
        path: &std::path::Path,
    ) -> Result<Self, crate::ontology::distance::sssom::SSSOMParseError> {
        let mut parser = crate::ontology::distance::sssom::SSSOMParser::new();
        let set = parser.parse_file(path)?;
        self.index.sssom.add_set(set);
        Ok(self)
    }

    /// Load UMLS MRCONSO
    pub fn load_umls(mut self, path: &std::path::Path) -> Result<Self, super::cui::CUILoadError> {
        self.index.cui_bridge.load_from_rrf(path)?;
        Ok(self)
    }

    /// Load LOOM TSV
    pub fn load_loom(mut self, path: &std::path::Path) -> Result<Self, super::loom::LOOMError> {
        self.index.loom.load_tsv(path)?;
        Ok(self)
    }

    /// Set embedding threshold
    pub fn embedding_threshold(mut self, threshold: f64) -> Self {
        self.index.embedding_threshold = threshold;
        self
    }

    /// Enable transitive inference
    pub fn transitive(mut self, enabled: bool) -> Self {
        self.index.transitive_enabled = enabled;
        self
    }

    /// Build the index
    pub fn build(mut self) -> AlignmentIndex {
        self.index.recompute_stats();
        self.index
    }
}

impl Default for AlignmentIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> AlignmentIndex {
        let mut index = AlignmentIndex::new();

        // Add some SSSOM mappings
        index.add_sssom_mapping(SSSOMMapping::exact(
            IRI::from_curie("MONDO", "0005148"),
            IRI::from_curie("DOID", "9352"),
            0.95,
        ));

        // Add CUI mappings
        index.add_cui_mapping(IRI::from_curie("CHEBI", "15365"), "C0004057".to_string());
        index.add_cui_mapping(
            IRI::from_curie("DrugBank", "DB00945"),
            "C0004057".to_string(),
        );

        index
    }

    #[test]
    fn test_find_sssom_alignment() {
        let index = create_test_index();

        let mondo = IRI::from_curie("MONDO", "0005148");
        let doid = IRI::from_curie("DOID", "9352");

        let result = index.find_alignment(&mondo, &doid);
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.method, AlignmentMethod::Direct);
        assert!(result.alignment.confidence > 0.9);
    }

    #[test]
    fn test_find_cui_alignment() {
        let index = create_test_index();

        let chebi = IRI::from_curie("CHEBI", "15365");
        let drugbank = IRI::from_curie("DrugBank", "DB00945");

        let result = index.find_alignment(&chebi, &drugbank);
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.method, AlignmentMethod::CUIBridge);
    }

    #[test]
    fn test_alignment_distance() {
        let index = create_test_index();

        let mondo = IRI::from_curie("MONDO", "0005148");
        let doid = IRI::from_curie("DOID", "9352");

        let distance = index.alignment_distance(&mondo, &doid);
        assert!(distance < 0.1); // Should be very close (exact match)

        // Unknown alignment
        let unknown1 = IRI::from_curie("TEST", "1");
        let unknown2 = IRI::from_curie("TEST", "2");
        let distance = index.alignment_distance(&unknown1, &unknown2);
        assert!(distance > 0.9); // Should be far apart
    }

    #[test]
    fn test_are_aligned() {
        let index = create_test_index();

        let mondo = IRI::from_curie("MONDO", "0005148");
        let doid = IRI::from_curie("DOID", "9352");

        assert!(index.are_aligned(&mondo, &doid, 0.5));
        assert!(index.are_aligned(&mondo, &doid, 0.9));
    }

    #[test]
    fn test_get_alignments() {
        let index = create_test_index();

        let chebi = IRI::from_curie("CHEBI", "15365");
        let alignments = index.get_alignments(&chebi);

        assert!(!alignments.is_empty());
    }

    #[test]
    fn test_transitive_alignment() {
        let mut index = AlignmentIndex::new();

        // A -> B -> C chain
        index.add_sssom_mapping(SSSOMMapping::exact(
            IRI::from_curie("ONT", "A"),
            IRI::from_curie("ONT", "B"),
            0.9,
        ));
        index.add_sssom_mapping(SSSOMMapping::exact(
            IRI::from_curie("ONT", "B"),
            IRI::from_curie("ONT", "C"),
            0.9,
        ));

        let a = IRI::from_curie("ONT", "A");
        let c = IRI::from_curie("ONT", "C");

        let result = index.find_alignment(&a, &c);
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.method, AlignmentMethod::Transitive);
        assert!(result.alignment.provenance.len() > 2);
    }
}
