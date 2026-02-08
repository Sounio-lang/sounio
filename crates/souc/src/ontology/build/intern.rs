//! String Interner with Prefix Compression
//!
//! Provides efficient string storage for ontology IRIs with:
//! - Deduplication via interning
//! - Prefix compression for common URL bases (10x memory savings)
//! - O(1) lookup by handle
//! - Thread-safe access

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Handle to an interned string
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedString {
    /// Index into the string table
    index: u32,
    /// Prefix table index (0 = no prefix)
    prefix_id: u16,
    /// Length of the suffix
    suffix_len: u16,
}

impl InternedString {
    /// Create a new interned string handle
    fn new(index: u32, prefix_id: u16, suffix_len: u16) -> Self {
        Self {
            index,
            prefix_id,
            suffix_len,
        }
    }

    /// Get the index in the string table
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Check if this string uses prefix compression
    pub fn is_compressed(&self) -> bool {
        self.prefix_id > 0
    }
}

/// Entry in the prefix table
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    /// The prefix string
    pub prefix: String,
    /// Number of strings using this prefix
    pub count: usize,
    /// Bytes saved by using this prefix
    pub bytes_saved: usize,
}

/// Table of common prefixes for compression
#[derive(Debug, Default)]
pub struct PrefixTable {
    /// Prefix string to ID mapping
    prefix_to_id: HashMap<String, u16>,
    /// ID to prefix entry mapping
    entries: Vec<PrefixEntry>,
}

impl PrefixTable {
    /// Create a new prefix table with common ontology prefixes
    pub fn new() -> Self {
        let mut table = Self::default();

        // Pre-register common ontology prefixes
        let common_prefixes = [
            "http://purl.obolibrary.org/obo/",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "http://www.w3.org/2000/01/rdf-schema#",
            "http://www.w3.org/2002/07/owl#",
            "http://www.w3.org/2001/XMLSchema#",
            "https://schema.org/",
            "http://hl7.org/fhir/",
            "http://snomed.info/id/",
            "http://purl.bioontology.org/ontology/",
            "http://www.geneontology.org/formats/oboInOwl#",
        ];

        for prefix in common_prefixes {
            table.register_prefix(prefix);
        }

        table
    }

    /// Register a new prefix, returns its ID
    pub fn register_prefix(&mut self, prefix: &str) -> u16 {
        if let Some(&id) = self.prefix_to_id.get(prefix) {
            return id;
        }

        let id = (self.entries.len() + 1) as u16; // 0 is reserved for "no prefix"
        self.prefix_to_id.insert(prefix.to_string(), id);
        self.entries.push(PrefixEntry {
            prefix: prefix.to_string(),
            count: 0,
            bytes_saved: 0,
        });
        id
    }

    /// Find the best matching prefix for a string
    /// Returns (prefix_id, prefix_length) if found
    pub fn find_prefix(&self, s: &str) -> Option<(u16, usize)> {
        // Find the longest matching prefix
        let mut best: Option<(u16, usize)> = None;

        for (prefix, &id) in &self.prefix_to_id {
            if s.starts_with(prefix) && (best.is_none() || prefix.len() > best.unwrap().1) {
                best = Some((id, prefix.len()));
            }
        }

        best
    }

    /// Get a prefix by ID
    pub fn get_prefix(&self, id: u16) -> Option<&str> {
        if id == 0 || id as usize > self.entries.len() {
            return None;
        }
        Some(&self.entries[id as usize - 1].prefix)
    }

    /// Record usage of a prefix
    pub fn record_usage(&mut self, id: u16, suffix_len: usize) {
        if id > 0 && (id as usize) <= self.entries.len() {
            let entry = &mut self.entries[id as usize - 1];
            entry.count += 1;
            // Bytes saved = prefix length (since we only store suffix)
            entry.bytes_saved += entry.prefix.len();
        }
    }

    /// Get total bytes saved by prefix compression
    pub fn total_bytes_saved(&self) -> usize {
        self.entries.iter().map(|e| e.bytes_saved).sum()
    }

    /// Get statistics about prefix usage
    pub fn stats(&self) -> Vec<(&str, usize, usize)> {
        self.entries
            .iter()
            .map(|e| (e.prefix.as_str(), e.count, e.bytes_saved))
            .collect()
    }

    /// Number of registered prefixes
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Thread-safe string interner with prefix compression
pub struct StringInterner {
    /// String storage (suffix only if compressed)
    strings: RwLock<Vec<String>>,
    /// String to handle mapping
    lookup: RwLock<HashMap<String, InternedString>>,
    /// Prefix compression table
    prefix_table: RwLock<PrefixTable>,
    /// Whether prefix compression is enabled
    compression_enabled: bool,
    /// Total bytes of original strings
    total_original_bytes: AtomicUsize,
    /// Total bytes actually stored
    total_stored_bytes: AtomicUsize,
}

impl StringInterner {
    /// Create a new interner without compression
    pub fn new() -> Self {
        Self {
            strings: RwLock::new(Vec::new()),
            lookup: RwLock::new(HashMap::new()),
            prefix_table: RwLock::new(PrefixTable::default()),
            compression_enabled: false,
            total_original_bytes: AtomicUsize::new(0),
            total_stored_bytes: AtomicUsize::new(0),
        }
    }

    /// Create a new interner with prefix compression
    pub fn with_prefix_compression(enabled: bool) -> Self {
        Self {
            strings: RwLock::new(Vec::new()),
            lookup: RwLock::new(HashMap::new()),
            prefix_table: RwLock::new(if enabled {
                PrefixTable::new()
            } else {
                PrefixTable::default()
            }),
            compression_enabled: enabled,
            total_original_bytes: AtomicUsize::new(0),
            total_stored_bytes: AtomicUsize::new(0),
        }
    }

    /// Intern a string, returning a handle
    pub fn intern(&self, s: &str) -> InternedString {
        // Check if already interned
        {
            let lookup = self.lookup.read().unwrap();
            if let Some(&handle) = lookup.get(s) {
                return handle;
            }
        }

        // Need to intern - acquire write locks
        let mut strings = self.strings.write().unwrap();
        let mut lookup = self.lookup.write().unwrap();

        // Double-check after acquiring write lock
        if let Some(&handle) = lookup.get(s) {
            return handle;
        }

        let original_len = s.len();
        self.total_original_bytes
            .fetch_add(original_len, Ordering::Relaxed);

        let (prefix_id, stored_string) = if self.compression_enabled {
            let mut prefix_table = self.prefix_table.write().unwrap();
            if let Some((id, prefix_len)) = prefix_table.find_prefix(s) {
                let suffix = &s[prefix_len..];
                prefix_table.record_usage(id, suffix.len());
                (id, suffix.to_string())
            } else {
                (0, s.to_string())
            }
        } else {
            (0, s.to_string())
        };

        self.total_stored_bytes
            .fetch_add(stored_string.len(), Ordering::Relaxed);

        let index = strings.len() as u32;
        let suffix_len = stored_string.len() as u16;
        strings.push(stored_string);

        let handle = InternedString::new(index, prefix_id, suffix_len);
        lookup.insert(s.to_string(), handle);

        handle
    }

    /// Resolve an interned string handle to the original string
    pub fn resolve(&self, handle: InternedString) -> Option<String> {
        let strings = self.strings.read().unwrap();
        let suffix = strings.get(handle.index as usize)?;

        if handle.prefix_id == 0 {
            Some(suffix.clone())
        } else {
            let prefix_table = self.prefix_table.read().unwrap();
            let prefix = prefix_table.get_prefix(handle.prefix_id)?;
            Some(format!("{}{}", prefix, suffix))
        }
    }

    /// Get the number of interned strings
    pub fn len(&self) -> usize {
        self.strings.read().unwrap().len()
    }

    /// Check if interner is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get bytes saved by interning and compression
    pub fn bytes_saved(&self) -> usize {
        let original = self.total_original_bytes.load(Ordering::Relaxed);
        let stored = self.total_stored_bytes.load(Ordering::Relaxed);
        original.saturating_sub(stored)
    }

    /// Get compression ratio (stored / original)
    pub fn compression_ratio(&self) -> f64 {
        let original = self.total_original_bytes.load(Ordering::Relaxed);
        let stored = self.total_stored_bytes.load(Ordering::Relaxed);
        if original == 0 {
            1.0
        } else {
            stored as f64 / original as f64
        }
    }

    /// Register a new prefix for compression
    pub fn register_prefix(&self, prefix: &str) -> u16 {
        self.prefix_table.write().unwrap().register_prefix(prefix)
    }

    /// Get prefix table statistics
    pub fn prefix_stats(&self) -> Vec<(String, usize, usize)> {
        self.prefix_table
            .read()
            .unwrap()
            .stats()
            .into_iter()
            .map(|(p, c, b)| (p.to_string(), c, b))
            .collect()
    }

    /// Analyze strings and suggest new prefixes
    pub fn suggest_prefixes(&self, min_frequency: usize) -> Vec<(String, usize)> {
        let lookup = self.lookup.read().unwrap();
        let mut prefix_counts: HashMap<String, usize> = HashMap::new();

        // Count potential prefixes (up to last / or #)
        for s in lookup.keys() {
            if let Some(pos) = s.rfind('/').or_else(|| s.rfind('#')) {
                let potential_prefix = &s[..=pos];
                *prefix_counts
                    .entry(potential_prefix.to_string())
                    .or_default() += 1;
            }
        }

        // Filter by frequency and sort
        let mut suggestions: Vec<_> = prefix_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_frequency)
            .collect();

        suggestions.sort_by(|a, b| b.1.cmp(&a.1));
        suggestions
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch interner for parallel processing
pub struct BatchInterner {
    /// Local buffer of strings to intern
    buffer: Vec<String>,
    /// Capacity before flushing
    capacity: usize,
}

impl BatchInterner {
    /// Create a new batch interner
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a string to the batch
    pub fn add(&mut self, s: String) -> bool {
        self.buffer.push(s);
        self.buffer.len() >= self.capacity
    }

    /// Flush the batch to the main interner
    pub fn flush(&mut self, interner: &StringInterner) -> Vec<InternedString> {
        let handles: Vec<_> = self.buffer.drain(..).map(|s| interner.intern(&s)).collect();
        handles
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_interning() {
        let interner = StringInterner::new();

        let h1 = interner.intern("hello");
        let h2 = interner.intern("world");
        let h3 = interner.intern("hello"); // duplicate

        assert_eq!(h1, h3);
        assert_ne!(h1, h2);
        assert_eq!(interner.len(), 2);

        assert_eq!(interner.resolve(h1), Some("hello".to_string()));
        assert_eq!(interner.resolve(h2), Some("world".to_string()));
    }

    #[test]
    fn test_prefix_compression() {
        let interner = StringInterner::with_prefix_compression(true);

        let h1 = interner.intern("http://purl.obolibrary.org/obo/CHEBI_15365");
        let h2 = interner.intern("http://purl.obolibrary.org/obo/GO_0008150");

        // Both should be compressed
        assert!(h1.is_compressed());
        assert!(h2.is_compressed());

        // Should resolve correctly
        assert_eq!(
            interner.resolve(h1),
            Some("http://purl.obolibrary.org/obo/CHEBI_15365".to_string())
        );
        assert_eq!(
            interner.resolve(h2),
            Some("http://purl.obolibrary.org/obo/GO_0008150".to_string())
        );

        // Should have saved bytes
        assert!(interner.bytes_saved() > 0);
    }

    #[test]
    fn test_prefix_table() {
        let mut table = PrefixTable::new();

        // Pre-registered prefixes
        assert!(table.len() > 0);

        // Find prefix
        let test_iri = "http://purl.obolibrary.org/obo/CHEBI_123";
        let result = table.find_prefix(test_iri);
        assert!(result.is_some());
        let (id, prefix_len) = result.unwrap();
        let expected_prefix = "http://purl.obolibrary.org/obo/";
        assert_eq!(prefix_len, expected_prefix.len());
        assert_eq!(&test_iri[prefix_len..], "CHEBI_123");

        // Register custom prefix
        let custom_id = table.register_prefix("http://example.org/");
        assert!(custom_id > 0);

        // Find custom prefix
        let test_iri2 = "http://example.org/term1";
        let result = table.find_prefix(test_iri2);
        assert!(result.is_some());
        let (_, prefix_len) = result.unwrap();
        assert_eq!(&test_iri2[prefix_len..], "term1");
    }

    #[test]
    fn test_compression_ratio() {
        let interner = StringInterner::with_prefix_compression(true);

        // Intern many strings with common prefix
        for i in 0..100 {
            interner.intern(&format!("http://purl.obolibrary.org/obo/TEST_{:05}", i));
        }

        let ratio = interner.compression_ratio();
        assert!(
            ratio < 0.5,
            "Expected >50% compression, got {:.2}%",
            (1.0 - ratio) * 100.0
        );
    }

    #[test]
    fn test_suggest_prefixes() {
        let interner = StringInterner::new(); // no compression for analysis

        // Intern strings with common prefixes
        for i in 0..10 {
            interner.intern(&format!("http://example.org/terms/{}", i));
        }
        for i in 0..5 {
            interner.intern(&format!("http://other.org/items/{}", i));
        }

        let suggestions = interner.suggest_prefixes(3);

        // Should suggest the common prefixes
        assert!(suggestions.iter().any(|(p, _)| p.contains("example.org")));
        assert!(suggestions.iter().any(|(p, _)| p.contains("other.org")));
    }

    #[test]
    fn test_batch_interner() {
        let interner = StringInterner::new();
        let mut batch = BatchInterner::new(5);

        // Add strings
        for i in 0..4 {
            assert!(!batch.add(format!("string_{}", i)));
        }
        assert_eq!(batch.len(), 4);

        // Fifth string triggers capacity
        assert!(batch.add("string_4".to_string()));

        // Flush
        let handles = batch.flush(&interner);
        assert_eq!(handles.len(), 5);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let interner = Arc::new(StringInterner::with_prefix_compression(true));
        let mut handles = vec![];

        for t in 0..4 {
            let interner = Arc::clone(&interner);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    interner.intern(&format!("http://purl.obolibrary.org/obo/T{}_{}", t, i));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Should have 400 unique strings
        assert_eq!(interner.len(), 400);
    }
}
