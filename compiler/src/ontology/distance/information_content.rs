// Sounio Compiler - Information Content Based Semantic Similarity
// Corpus-based statistics for semantic distance calculation

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use super::path::HierarchyGraph;
use crate::ontology::loader::IRI;

/// Information Content (IC) index for corpus-based similarity
///
/// IC measures how specific a concept is based on corpus statistics.
/// IC(c) = -log(p(c)) where p(c) is probability of encountering concept c
pub struct ICIndex {
    /// IC values for each term
    ic_values: HashMap<IRI, f64>,

    /// Raw frequency counts (before normalization)
    frequencies: HashMap<IRI, u64>,

    /// Total annotations in corpus
    total_annotations: u64,

    /// Maximum IC value (for normalization)
    max_ic: f64,

    /// Intrinsic IC values (structure-based, not corpus-based)
    intrinsic_ic: HashMap<IRI, f64>,
}

/// Configuration for IC calculation
#[derive(Debug, Clone)]
pub struct ICConfig {
    /// Smoothing factor for zero frequencies (Laplace smoothing)
    pub smoothing: f64,

    /// Whether to use intrinsic IC as fallback
    pub use_intrinsic_fallback: bool,

    /// Minimum IC value
    pub min_ic: f64,
}

impl Default for ICConfig {
    fn default() -> Self {
        Self {
            smoothing: 1.0,
            use_intrinsic_fallback: true,
            min_ic: 0.0,
        }
    }
}

/// Result of IC-based similarity computation
#[derive(Debug, Clone)]
pub struct ICSimilarity {
    /// Resnik similarity: IC(LCA)
    pub resnik: f64,

    /// Lin similarity: 2*IC(LCA) / (IC(a) + IC(b))
    pub lin: f64,

    /// Jiang-Conrath distance: IC(a) + IC(b) - 2*IC(LCA)
    pub jiang_conrath: f64,

    /// IC of the LCA
    pub ic_lca: f64,

    /// IC of term a
    pub ic_a: f64,

    /// IC of term b
    pub ic_b: f64,
}

impl ICIndex {
    /// Create empty IC index
    pub fn new() -> Self {
        Self {
            ic_values: HashMap::new(),
            frequencies: HashMap::new(),
            total_annotations: 0,
            max_ic: 0.0,
            intrinsic_ic: HashMap::new(),
        }
    }

    /// Create IC index with estimated capacity
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            ic_values: HashMap::with_capacity(cap),
            frequencies: HashMap::with_capacity(cap),
            total_annotations: 0,
            max_ic: 0.0,
            intrinsic_ic: HashMap::with_capacity(cap),
        }
    }

    /// Add annotation frequency for a term
    pub fn add_frequency(&mut self, iri: IRI, count: u64) {
        *self.frequencies.entry(iri).or_insert(0) += count;
        self.total_annotations += count;
    }

    /// Build IC values from frequency counts
    pub fn compute_ic(&mut self, config: &ICConfig) {
        self.ic_values.clear();
        self.max_ic = 0.0;

        let total_f64 = self.total_annotations as f64;

        for (iri, &count) in &self.frequencies {
            // Apply Laplace smoothing
            let smoothed_count = count as f64 + config.smoothing;
            let smoothed_total = total_f64 + config.smoothing * self.frequencies.len() as f64;

            // IC = -log(p(c))
            let p = smoothed_count / smoothed_total;
            let ic = (-p.ln()).max(config.min_ic);

            self.ic_values.insert(iri.clone(), ic);
            self.max_ic = self.max_ic.max(ic);
        }
    }

    /// Compute intrinsic IC based on hierarchy structure
    /// Uses Seco et al. formula: IC(c) = 1 - log(hypo(c) + 1) / log(max_nodes)
    pub fn compute_intrinsic_ic(&mut self, hierarchy: &HierarchyGraph) {
        self.intrinsic_ic.clear();

        let stats = hierarchy.stats();
        let max_nodes = stats.num_terms as f64;
        let log_max = max_nodes.ln();

        if log_max == 0.0 {
            return;
        }

        // For each term, count descendants (hyponyms)
        for (iri, _) in hierarchy
            .stats()
            .num_terms
            .checked_sub(0)
            .map(|_| {
                // Iterate through all terms by checking leaves and roots
                Vec::<(IRI, u64)>::new()
            })
            .unwrap_or_default()
        {
            let _ = iri; // Placeholder
        }

        // Actually iterate through the graph
        // We need to get all IRIs from the hierarchy
        // For now, use the frequencies as proxy for known terms
        for iri in self.frequencies.keys() {
            let descendants = hierarchy.get_descendants(iri);
            let hypo_count = descendants.len() as f64;

            // Seco formula: IC(c) = 1 - log(hypo(c) + 1) / log(max_nodes)
            let ic = 1.0 - (hypo_count + 1.0).ln() / log_max;
            self.intrinsic_ic.insert(iri.clone(), ic);
        }
    }

    /// Get IC value for a term
    pub fn get_ic(&self, iri: &IRI) -> Option<f64> {
        self.ic_values.get(iri).copied()
    }

    /// Get IC with intrinsic fallback
    pub fn get_ic_with_fallback(&self, iri: &IRI) -> f64 {
        self.ic_values
            .get(iri)
            .or_else(|| self.intrinsic_ic.get(iri))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get intrinsic IC value
    pub fn get_intrinsic_ic(&self, iri: &IRI) -> Option<f64> {
        self.intrinsic_ic.get(iri).copied()
    }

    /// Get maximum IC value for normalization
    pub fn max_ic(&self) -> f64 {
        self.max_ic
    }

    /// Compute Resnik similarity: IC(LCA)
    pub fn resnik_similarity(&self, a: &IRI, b: &IRI, hierarchy: &HierarchyGraph) -> Option<f64> {
        let lca_iri = hierarchy.lowest_common_ancestor(a, b)?;
        self.get_ic(&lca_iri)
    }

    /// Compute Lin similarity: 2*IC(LCA) / (IC(a) + IC(b))
    pub fn lin_similarity(&self, a: &IRI, b: &IRI, hierarchy: &HierarchyGraph) -> Option<f64> {
        let lca_iri = hierarchy.lowest_common_ancestor(a, b)?;

        let ic_lca = self.get_ic_with_fallback(&lca_iri);
        let ic_a = self.get_ic_with_fallback(a);
        let ic_b = self.get_ic_with_fallback(b);

        let denominator = ic_a + ic_b;
        if denominator == 0.0 {
            return Some(1.0); // Both are root concepts
        }

        Some(2.0 * ic_lca / denominator)
    }

    /// Compute Jiang-Conrath distance: IC(a) + IC(b) - 2*IC(LCA)
    /// Note: This is a distance (lower = more similar), not similarity
    pub fn jiang_conrath_distance(
        &self,
        a: &IRI,
        b: &IRI,
        hierarchy: &HierarchyGraph,
    ) -> Option<f64> {
        let lca_iri = hierarchy.lowest_common_ancestor(a, b)?;

        let ic_lca = self.get_ic_with_fallback(&lca_iri);
        let ic_a = self.get_ic_with_fallback(a);
        let ic_b = self.get_ic_with_fallback(b);

        Some(ic_a + ic_b - 2.0 * ic_lca)
    }

    /// Compute all IC-based metrics at once
    pub fn compute_similarity(
        &self,
        a: &IRI,
        b: &IRI,
        hierarchy: &HierarchyGraph,
    ) -> Option<ICSimilarity> {
        let lca_iri = hierarchy.lowest_common_ancestor(a, b)?;

        let ic_lca = self.get_ic_with_fallback(&lca_iri);
        let ic_a = self.get_ic_with_fallback(a);
        let ic_b = self.get_ic_with_fallback(b);

        let lin = if ic_a + ic_b == 0.0 {
            1.0
        } else {
            2.0 * ic_lca / (ic_a + ic_b)
        };

        Some(ICSimilarity {
            resnik: ic_lca,
            lin,
            jiang_conrath: ic_a + ic_b - 2.0 * ic_lca,
            ic_lca,
            ic_a,
            ic_b,
        })
    }

    /// Load IC values from annotation file
    /// Format: <IRI>\t<count>
    pub fn load_from_file(&mut self, path: &Path, config: &ICConfig) -> Result<(), ICLoadError> {
        let file = File::open(path).map_err(ICLoadError::IoError)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.map_err(ICLoadError::IoError)?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 2 {
                continue;
            }

            let iri = IRI::new(parts[0]);
            let count: u64 = parts[1].parse().map_err(|_| ICLoadError::ParseError)?;

            self.add_frequency(iri, count);
        }

        self.compute_ic(config);
        Ok(())
    }

    /// Save IC values to file
    pub fn save_to_file(&self, path: &Path) -> Result<(), ICLoadError> {
        let mut file = File::create(path).map_err(ICLoadError::IoError)?;

        writeln!(file, "# IC values for Sounio ontology index").map_err(ICLoadError::IoError)?;
        writeln!(file, "# Format: IRI<TAB>IC_value").map_err(ICLoadError::IoError)?;

        let mut entries: Vec<_> = self.ic_values.iter().collect();
        entries.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));

        for (iri, ic) in entries {
            writeln!(file, "{}\t{:.6}", iri.as_str(), ic).map_err(ICLoadError::IoError)?;
        }

        Ok(())
    }

    /// Get statistics about IC index
    pub fn stats(&self) -> ICStats {
        let ic_values: Vec<f64> = self.ic_values.values().copied().collect();

        let mean = if ic_values.is_empty() {
            0.0
        } else {
            ic_values.iter().sum::<f64>() / ic_values.len() as f64
        };

        let std_dev = if ic_values.len() < 2 {
            0.0
        } else {
            let variance =
                ic_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / ic_values.len() as f64;
            variance.sqrt()
        };

        ICStats {
            num_terms: self.ic_values.len(),
            total_annotations: self.total_annotations,
            max_ic: self.max_ic,
            mean_ic: mean,
            std_dev_ic: std_dev,
            num_intrinsic: self.intrinsic_ic.len(),
        }
    }
}

impl Default for ICIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Error loading IC values
#[derive(Debug)]
pub enum ICLoadError {
    IoError(std::io::Error),
    ParseError,
}

impl std::fmt::Display for ICLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ParseError => write!(f, "Parse error"),
        }
    }
}

impl std::error::Error for ICLoadError {}

/// Statistics about IC index
#[derive(Debug, Clone)]
pub struct ICStats {
    pub num_terms: usize,
    pub total_annotations: u64,
    pub max_ic: f64,
    pub mean_ic: f64,
    pub std_dev_ic: f64,
    pub num_intrinsic: usize,
}

/// Corpus annotation builder for computing IC from real data
pub struct AnnotationCorpus {
    /// Term frequencies across all documents
    term_counts: HashMap<IRI, u64>,

    /// Number of documents
    num_documents: u64,

    /// Document frequency (number of docs containing each term)
    doc_frequency: HashMap<IRI, u64>,
}

impl AnnotationCorpus {
    pub fn new() -> Self {
        Self {
            term_counts: HashMap::new(),
            num_documents: 0,
            doc_frequency: HashMap::new(),
        }
    }

    /// Add a document's annotations
    pub fn add_document(&mut self, annotations: &[IRI]) {
        self.num_documents += 1;

        // Count term frequencies
        for ann in annotations {
            *self.term_counts.entry(ann.clone()).or_insert(0) += 1;
        }

        // Count document frequency (unique terms in this doc)
        let unique: std::collections::HashSet<_> = annotations.iter().collect();
        for ann in unique {
            *self.doc_frequency.entry(ann.clone()).or_insert(0) += 1;
        }
    }

    /// Add annotations with ancestor propagation
    /// Each annotation also counts for all its ancestors
    pub fn add_document_with_ancestors(&mut self, annotations: &[IRI], hierarchy: &HierarchyGraph) {
        self.num_documents += 1;

        let mut all_terms = Vec::new();
        for ann in annotations {
            all_terms.push(ann.clone());
            // Add all ancestors
            for ancestor in hierarchy.get_ancestors(ann) {
                all_terms.push(ancestor);
            }
        }

        for term in &all_terms {
            *self.term_counts.entry(term.clone()).or_insert(0) += 1;
        }

        let unique: std::collections::HashSet<_> = all_terms.iter().collect();
        for term in unique {
            *self.doc_frequency.entry(term.clone()).or_insert(0) += 1;
        }
    }

    /// Build IC index from corpus
    pub fn build_ic_index(&self, config: &ICConfig) -> ICIndex {
        let mut index = ICIndex::with_capacity(self.term_counts.len());

        for (iri, &count) in &self.term_counts {
            index.add_frequency(iri.clone(), count);
        }

        index.compute_ic(config);
        index
    }

    /// Get TF-IDF weighted IC
    /// Combines term frequency with inverse document frequency
    pub fn tfidf_ic(&self, iri: &IRI) -> f64 {
        let tf = *self.term_counts.get(iri).unwrap_or(&0) as f64;
        let df = *self.doc_frequency.get(iri).unwrap_or(&1) as f64;
        let n = self.num_documents as f64;

        if tf == 0.0 || n == 0.0 {
            return 0.0;
        }

        // TF-IDF = tf * log(N / df)
        tf * (n / df).ln()
    }
}

impl Default for AnnotationCorpus {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate similarity calculator for sets of terms
pub struct SetSimilarity;

impl SetSimilarity {
    /// Best Match Average (BMA) similarity between two term sets
    /// For each term in set A, find best match in set B, then average
    pub fn bma(
        set_a: &[IRI],
        set_b: &[IRI],
        ic_index: &ICIndex,
        hierarchy: &HierarchyGraph,
    ) -> f64 {
        if set_a.is_empty() || set_b.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;

        // Average of best matches from A to B
        for a in set_a {
            let best = set_b
                .iter()
                .filter_map(|b| ic_index.lin_similarity(a, b, hierarchy))
                .fold(0.0f64, |acc, x| acc.max(x));
            sum += best;
        }

        // Average of best matches from B to A
        for b in set_b {
            let best = set_a
                .iter()
                .filter_map(|a| ic_index.lin_similarity(a, b, hierarchy))
                .fold(0.0f64, |acc, x| acc.max(x));
            sum += best;
        }

        sum / (set_a.len() + set_b.len()) as f64
    }

    /// Maximum similarity between any pair
    pub fn max_similarity(
        set_a: &[IRI],
        set_b: &[IRI],
        ic_index: &ICIndex,
        hierarchy: &HierarchyGraph,
    ) -> f64 {
        let mut max_sim: f64 = 0.0;

        for a in set_a {
            for b in set_b {
                if let Some(sim) = ic_index.lin_similarity(a, b, hierarchy) {
                    max_sim = max_sim.max(sim);
                }
            }
        }

        max_sim
    }

    /// Average pairwise similarity
    pub fn avg_similarity(
        set_a: &[IRI],
        set_b: &[IRI],
        ic_index: &ICIndex,
        hierarchy: &HierarchyGraph,
    ) -> f64 {
        if set_a.is_empty() || set_b.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for a in set_a {
            for b in set_b {
                if let Some(sim) = ic_index.lin_similarity(a, b, hierarchy) {
                    sum += sim;
                    count += 1;
                }
            }
        }

        if count == 0 { 0.0 } else { sum / count as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::distance::path::HierarchyGraph;

    fn make_test_setup() -> (HierarchyGraph, ICIndex) {
        let mut graph = HierarchyGraph::new();

        let thing = IRI::new("http://example.org/Thing");
        let animal = IRI::new("http://example.org/Animal");
        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        graph.add_is_a(&animal, &thing);
        graph.add_is_a(&dog, &animal);
        graph.add_is_a(&cat, &animal);

        let mut ic = ICIndex::new();
        ic.add_frequency(thing.clone(), 100);
        ic.add_frequency(animal.clone(), 50);
        ic.add_frequency(dog.clone(), 20);
        ic.add_frequency(cat.clone(), 25);
        ic.compute_ic(&ICConfig::default());

        (graph, ic)
    }

    #[test]
    fn test_ic_values() {
        let (_, ic) = make_test_setup();

        // More specific terms should have higher IC
        let ic_thing = ic.get_ic(&IRI::new("http://example.org/Thing")).unwrap();
        let ic_dog = ic.get_ic(&IRI::new("http://example.org/Dog")).unwrap();

        assert!(ic_dog > ic_thing, "Dog should have higher IC than Thing");
    }

    #[test]
    fn test_lin_similarity() {
        let (graph, ic) = make_test_setup();

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        let sim = ic.lin_similarity(&dog, &cat, &graph).unwrap();

        // Dog and Cat share Animal as LCA, so should have reasonable similarity
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_jiang_conrath() {
        let (graph, ic) = make_test_setup();

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        let dist = ic.jiang_conrath_distance(&dog, &cat, &graph).unwrap();

        // Distance should be positive
        assert!(dist >= 0.0);
    }
}
