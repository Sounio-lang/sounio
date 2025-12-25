//! Optimized Ontology Resolution
//!
//! Key optimizations:
//! 1. Bloom filter for fast "definitely not present" checks
//! 2. Interned IRIs to avoid string allocations
//! 3. Pre-computed subsumption index
//! 4. SIMD-accelerated embedding distance (see embedding/simd.rs)

use super::{IRI, LoadedTerm};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

/// Error type for optimized loader operations
#[derive(Debug, Clone)]
pub enum OptimizedLoaderError {
    /// Term not found in the ontology
    TermNotFound(IRI),
    /// Internal error
    Internal(String),
}

impl std::fmt::Display for OptimizedLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TermNotFound(iri) => write!(f, "Term not found: {}", iri),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for OptimizedLoaderError {}

/// Simple bloom filter for fast negative lookups
pub struct BloomFilter {
    bits: Vec<u64>,
    num_hashes: usize,
    num_bits: usize,
}

impl BloomFilter {
    /// Create a new bloom filter with given capacity and false positive rate
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        // Calculate optimal number of bits and hashes
        let num_bits = Self::optimal_bits(expected_items.max(1), fp_rate);
        let num_hashes = Self::optimal_hashes(num_bits, expected_items.max(1));
        let num_words = num_bits.div_ceil(64);

        Self {
            bits: vec![0u64; num_words],
            num_hashes,
            num_bits,
        }
    }

    fn optimal_bits(n: usize, p: f64) -> usize {
        let ln2_sq = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let p_clamped = p.max(0.0001).min(0.5);
        (-(n as f64) * p_clamped.ln() / ln2_sq).ceil() as usize
    }

    fn optimal_hashes(m: usize, n: usize) -> usize {
        ((m as f64 / n as f64) * std::f64::consts::LN_2).ceil() as usize
    }

    /// Insert an item into the bloom filter
    pub fn insert(&mut self, item: &str) {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let bit_pos = hash % self.num_bits;
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if word_idx < self.bits.len() {
                self.bits[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    /// Check if an item might be in the set (false positives possible)
    pub fn might_contain(&self, item: &str) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let bit_pos = hash % self.num_bits;
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if word_idx >= self.bits.len() || self.bits[word_idx] & (1u64 << bit_idx) == 0 {
                return false;
            }
        }
        true
    }

    fn hash(&self, item: &str, seed: usize) -> usize {
        // FNV-1a hash with seed mixing
        let mut hash = 0xcbf29ce484222325u64.wrapping_add(seed as u64);
        for byte in item.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash as usize
    }
}

/// Interned IRI for efficient comparison and storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedIRI(u32);

/// String interner for IRI deduplication
pub struct IRIInterner {
    strings: RwLock<Vec<String>>,
    indices: RwLock<HashMap<String, InternedIRI>>,
}

impl IRIInterner {
    pub fn new() -> Self {
        Self {
            strings: RwLock::new(Vec::new()),
            indices: RwLock::new(HashMap::new()),
        }
    }

    /// Intern a string, returning its interned ID
    pub fn intern(&self, s: &str) -> InternedIRI {
        // Check if already interned
        if let Ok(indices) = self.indices.read()
            && let Some(&idx) = indices.get(s)
        {
            return idx;
        }

        // Intern new string
        let mut strings = self.strings.write().unwrap();
        let mut indices = self.indices.write().unwrap();

        // Double-check after acquiring write lock
        if let Some(&idx) = indices.get(s) {
            return idx;
        }

        let idx = InternedIRI(strings.len() as u32);
        strings.push(s.to_string());
        indices.insert(s.to_string(), idx);

        idx
    }

    /// Get an already-interned string's ID
    pub fn get(&self, s: &str) -> Option<InternedIRI> {
        self.indices.read().ok()?.get(s).copied()
    }

    /// Get the string for an interned ID
    pub fn resolve(&self, id: InternedIRI) -> Option<String> {
        self.strings.read().ok()?.get(id.0 as usize).cloned()
    }

    /// Get or intern a string
    pub fn get_or_intern(&self, s: &str) -> InternedIRI {
        self.get(s).unwrap_or_else(|| self.intern(s))
    }

    /// Number of interned strings
    pub fn len(&self) -> usize {
        self.strings.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for IRIInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-computed subsumption index for O(1) subtype checks
pub struct SubsumptionIndex {
    /// If (A, B) is in this set, then A is-a B (transitively)
    pairs: HashSet<(InternedIRI, InternedIRI)>,
    /// Direct parents for each term
    parents: HashMap<InternedIRI, Vec<InternedIRI>>,
    /// Direct children for each term
    children: HashMap<InternedIRI, Vec<InternedIRI>>,
}

impl SubsumptionIndex {
    pub fn new() -> Self {
        Self {
            pairs: HashSet::new(),
            parents: HashMap::new(),
            children: HashMap::new(),
        }
    }

    /// Build the index from loaded terms
    pub fn build(terms: &[LoadedTerm], interner: &IRIInterner) -> Self {
        let mut index = Self::new();

        // First pass: collect direct relationships
        for term in terms {
            let term_id = interner.get_or_intern(&term.iri.to_string());

            for parent_iri in &term.superclasses {
                let parent_id = interner.get_or_intern(&parent_iri.to_string());

                index.parents.entry(term_id).or_default().push(parent_id);
                index.children.entry(parent_id).or_default().push(term_id);
            }
        }

        // Second pass: compute transitive closure
        for term in terms {
            let term_id = interner.get_or_intern(&term.iri.to_string());

            // Self-subsumption
            index.pairs.insert((term_id, term_id));

            // Compute all ancestors
            let mut to_visit: Vec<InternedIRI> =
                index.parents.get(&term_id).cloned().unwrap_or_default();
            let mut visited: HashSet<InternedIRI> = HashSet::new();

            while let Some(ancestor) = to_visit.pop() {
                if visited.contains(&ancestor) {
                    continue;
                }
                visited.insert(ancestor);

                // term is-a ancestor
                index.pairs.insert((term_id, ancestor));

                // Continue to ancestor's parents
                if let Some(grandparents) = index.parents.get(&ancestor) {
                    for gp in grandparents {
                        if !visited.contains(gp) {
                            to_visit.push(*gp);
                        }
                    }
                }
            }
        }

        index
    }

    /// Check if sub is a subtype of sup (transitively)
    pub fn is_subtype(&self, sub: InternedIRI, sup: InternedIRI) -> bool {
        self.pairs.contains(&(sub, sup))
    }

    /// Get direct parents
    pub fn direct_parents(&self, term: InternedIRI) -> &[InternedIRI] {
        self.parents.get(&term).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get direct children
    pub fn direct_children(&self, term: InternedIRI) -> &[InternedIRI] {
        self.children
            .get(&term)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get path length between two terms (-1 if unrelated)
    pub fn path_length(&self, from: InternedIRI, to: InternedIRI) -> i32 {
        if from == to {
            return 0;
        }

        // BFS to find shortest path
        let mut queue = std::collections::VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((from, 0i32));
        visited.insert(from);

        while let Some((current, dist)) = queue.pop_front() {
            // Check parents
            for &parent in self.direct_parents(current) {
                if parent == to {
                    return dist + 1;
                }
                if !visited.contains(&parent) {
                    visited.insert(parent);
                    queue.push_back((parent, dist + 1));
                }
            }

            // Check children
            for &child in self.direct_children(current) {
                if child == to {
                    return dist + 1;
                }
                if !visited.contains(&child) {
                    visited.insert(child);
                    queue.push_back((child, dist + 1));
                }
            }
        }

        -1 // Not connected
    }
}

impl Default for SubsumptionIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the optimized loader
#[derive(Debug, Default, Clone)]
pub struct LoaderStats {
    pub bloom_checks: u64,
    pub bloom_negatives: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Optimized term cache with bloom filter
pub struct OptimizedTermCache {
    /// Bloom filter for fast negative lookups (0.1% false positive rate)
    bloom: BloomFilter,

    /// Interned IRI strings
    interner: IRIInterner,

    /// Pre-computed subsumption closure
    subsumption_index: SubsumptionIndex,

    /// Term cache
    terms: RwLock<HashMap<InternedIRI, LoadedTerm>>,

    /// Statistics
    stats: RwLock<LoaderStats>,
}

impl OptimizedTermCache {
    /// Create a new optimized cache from loaded terms
    pub fn new(terms: Vec<LoadedTerm>) -> Self {
        // Build bloom filter
        let mut bloom = BloomFilter::new(terms.len().max(100), 0.001);
        for term in &terms {
            bloom.insert(&term.iri.to_string());
        }

        // Build interner
        let interner = IRIInterner::new();
        for term in &terms {
            interner.intern(&term.iri.to_string());
        }

        // Build subsumption index
        let subsumption_index = SubsumptionIndex::build(&terms, &interner);

        // Populate cache
        let mut term_map = HashMap::new();
        for term in terms {
            let interned = interner.get_or_intern(&term.iri.to_string());
            term_map.insert(interned, term);
        }

        Self {
            bloom,
            interner,
            subsumption_index,
            terms: RwLock::new(term_map),
            stats: RwLock::new(LoaderStats::default()),
        }
    }

    /// Fast lookup with bloom filter pre-check
    pub fn get(&self, iri: &IRI) -> Result<LoadedTerm, OptimizedLoaderError> {
        let iri_str = iri.to_string();

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.bloom_checks += 1;
        }

        // Bloom filter check
        if !self.bloom.might_contain(&iri_str) {
            if let Ok(mut stats) = self.stats.write() {
                stats.bloom_negatives += 1;
            }
            return Err(OptimizedLoaderError::TermNotFound(iri.clone()));
        }

        // Get interned IRI
        let interned = match self.interner.get(&iri_str) {
            Some(i) => i,
            None => return Err(OptimizedLoaderError::TermNotFound(iri.clone())),
        };

        // Cache lookup
        let terms = self
            .terms
            .read()
            .map_err(|e| OptimizedLoaderError::Internal(e.to_string()))?;

        if let Some(term) = terms.get(&interned) {
            if let Ok(mut stats) = self.stats.write() {
                stats.cache_hits += 1;
            }
            return Ok(term.clone());
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.cache_misses += 1;
        }

        Err(OptimizedLoaderError::TermNotFound(iri.clone()))
    }

    /// Fast subsumption check via pre-computed index
    pub fn is_subtype(&self, sub: &IRI, sup: &IRI) -> bool {
        let sub_interned = match self.interner.get(&sub.to_string()) {
            Some(s) => s,
            None => return false,
        };
        let sup_interned = match self.interner.get(&sup.to_string()) {
            Some(s) => s,
            None => return false,
        };

        self.subsumption_index
            .is_subtype(sub_interned, sup_interned)
    }

    /// Get path length between two IRIs
    pub fn path_length(&self, from: &IRI, to: &IRI) -> i32 {
        let from_interned = match self.interner.get(&from.to_string()) {
            Some(s) => s,
            None => return -1,
        };
        let to_interned = match self.interner.get(&to.to_string()) {
            Some(s) => s,
            None => return -1,
        };

        self.subsumption_index
            .path_length(from_interned, to_interned)
    }

    /// Get loader statistics
    pub fn stats(&self) -> LoaderStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = LoaderStats::default();
        }
    }

    /// Number of interned IRIs
    pub fn interned_count(&self) -> usize {
        self.interner.len()
    }

    /// Number of cached terms
    pub fn cached_count(&self) -> usize {
        self.terms.read().map(|c| c.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::OntologyId;
    use super::*;

    #[test]
    fn test_bloom_filter() {
        let mut bloom = BloomFilter::new(1000, 0.001);

        bloom.insert("hello");
        bloom.insert("world");

        assert!(bloom.might_contain("hello"));
        assert!(bloom.might_contain("world"));
    }

    #[test]
    fn test_iri_interner() {
        let interner = IRIInterner::new();

        let id1 = interner.intern("http://example.org/A");
        let id2 = interner.intern("http://example.org/B");
        let id3 = interner.intern("http://example.org/A");

        assert_eq!(id1, id3); // Same string should get same ID
        assert_ne!(id1, id2); // Different strings should get different IDs

        assert_eq!(
            interner.resolve(id1),
            Some("http://example.org/A".to_string())
        );
    }

    #[test]
    fn test_subsumption_index() {
        let interner = IRIInterner::new();

        // Create a simple hierarchy: A -> B -> C
        let terms = vec![
            LoadedTerm {
                iri: IRI::new("http://example.org/A"),
                label: "A".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![],
                subclasses: vec![],
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
                iri: IRI::new("http://example.org/B"),
                label: "B".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![IRI::new("http://example.org/A")],
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
                iri: IRI::new("http://example.org/C"),
                label: "C".to_string(),
                ontology: OntologyId::Unknown,
                superclasses: vec![IRI::new("http://example.org/B")],
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
        ];

        let index = SubsumptionIndex::build(&terms, &interner);

        let a = interner.get("http://example.org/A").unwrap();
        let b = interner.get("http://example.org/B").unwrap();
        let c = interner.get("http://example.org/C").unwrap();

        // Direct subsumption
        assert!(index.is_subtype(b, a));
        assert!(index.is_subtype(c, b));

        // Transitive subsumption
        assert!(index.is_subtype(c, a));

        // Self subsumption
        assert!(index.is_subtype(a, a));

        // Not subtype
        assert!(!index.is_subtype(a, b));
        assert!(!index.is_subtype(a, c));

        // Path lengths
        assert_eq!(index.path_length(c, a), 2);
        assert_eq!(index.path_length(b, a), 1);
        assert_eq!(index.path_length(a, a), 0);
    }
}
