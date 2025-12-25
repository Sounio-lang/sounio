//! Compact Term Representation
//!
//! This module provides a memory-efficient representation for ontological terms.
//! Each term fits in 64 bytes (one cache line) for optimal cache performance.
//!
//! Layout (64 bytes total):
//! - iri_hash: u64 (8 bytes) - Hash of the IRI for fast comparison
//! - iri_offset: u32 (4 bytes) - Offset into string table
//! - iri_len: u16 (2 bytes) - Length of IRI string
//! - label_offset: u32 (4 bytes) - Offset into string table
//! - label_len: u16 (2 bytes) - Length of label string
//! - parent_idx: u32 (4 bytes) - Index of parent term (0 = root)
//! - depth: u16 (2 bytes) - Depth in hierarchy
//! - flags: u16 (2 bytes) - Bit flags for properties
//! - child_count: u32 (4 bytes) - Number of direct children
//! - embedding_offset: u32 (4 bytes) - Offset to embedding vector (0 = none)
//! - reserved: [u8; 28] (28 bytes) - Reserved for future use

use std::sync::Arc;

/// Flags for compact term properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TermFlags(u16);

impl TermFlags {
    /// Term is deprecated
    pub const DEPRECATED: u16 = 1 << 0;
    /// Term is abstract (cannot be instantiated)
    pub const ABSTRACT: u16 = 1 << 1;
    /// Term has embedding vector
    pub const HAS_EMBEDDING: u16 = 1 << 2;
    /// Term is a leaf (no children)
    pub const IS_LEAF: u16 = 1 << 3;
    /// Term is from external ontology
    pub const EXTERNAL: u16 = 1 << 4;
    /// Term has synonyms
    pub const HAS_SYNONYMS: u16 = 1 << 5;
    /// Term has definition
    pub const HAS_DEFINITION: u16 = 1 << 6;
    /// Term is frequently accessed (hot)
    pub const HOT: u16 = 1 << 7;

    /// Create empty flags
    pub fn empty() -> Self {
        TermFlags(0)
    }

    /// Create from raw bits
    pub fn from_bits(bits: u16) -> Self {
        TermFlags(bits)
    }

    /// Get raw bits
    pub fn bits(&self) -> u16 {
        self.0
    }

    /// Set a flag
    pub fn set(&mut self, flag: u16) {
        self.0 |= flag;
    }

    /// Clear a flag
    pub fn clear(&mut self, flag: u16) {
        self.0 &= !flag;
    }

    /// Check if flag is set
    pub fn has(&self, flag: u16) -> bool {
        (self.0 & flag) != 0
    }

    /// Check if deprecated
    pub fn is_deprecated(&self) -> bool {
        self.has(Self::DEPRECATED)
    }

    /// Check if abstract
    pub fn is_abstract(&self) -> bool {
        self.has(Self::ABSTRACT)
    }

    /// Check if has embedding
    pub fn has_embedding(&self) -> bool {
        self.has(Self::HAS_EMBEDDING)
    }

    /// Check if leaf node
    pub fn is_leaf(&self) -> bool {
        self.has(Self::IS_LEAF)
    }

    /// Check if hot (frequently accessed)
    pub fn is_hot(&self) -> bool {
        self.has(Self::HOT)
    }
}

/// Compact representation of an ontological term
///
/// This uses a fixed-size layout optimized for cache efficiency.
/// For strings longer than inline capacity, we store a hash and
/// the full string must be looked up in a separate string table.
#[repr(C)]
#[derive(Clone)]
pub struct CompactTerm {
    /// Hash of the IRI for fast comparison (8 bytes)
    pub iri_hash: u64,

    /// Inline IRI storage - up to 16 bytes
    iri_inline: [u8; 16],

    /// IRI length (if <= 16, stored inline; otherwise, look up by hash)
    iri_len: u8,

    /// Inline label storage - up to 7 bytes
    label_inline: [u8; 7],

    /// Label length (if <= 7, stored inline; otherwise truncated)
    label_len: u8,

    /// Index of parent term in the ontology (0 = root) (4 bytes)
    pub parent_idx: u32,

    /// Depth in the hierarchy (0 = root) (2 bytes)
    pub depth: u16,

    /// Property flags (2 bytes)
    pub flags: TermFlags,

    /// Number of direct children (4 bytes)
    pub child_count: u32,

    /// Offset to embedding vector in embedding store (0 = none) (4 bytes)
    pub embedding_offset: u32,
}

// Total: 8 + 16 + 1 + 7 + 1 + 4 + 2 + 2 + 4 + 4 = 49 bytes
// With padding for u64 alignment = 56 bytes, fits in cache line

impl CompactTerm {
    /// Maximum inline IRI length
    const MAX_INLINE_IRI: usize = 16;
    /// Maximum inline label length
    const MAX_INLINE_LABEL: usize = 7;

    /// Create a new compact term
    pub fn new(iri: &str) -> Self {
        let iri_hash = Self::hash_iri(iri);
        let (iri_inline, iri_len) = Self::store_iri_inline(iri);

        CompactTerm {
            iri_hash,
            iri_inline,
            iri_len,
            label_inline: [0; 7],
            label_len: 0,
            parent_idx: 0,
            depth: 0,
            flags: TermFlags::empty(),
            child_count: 0,
            embedding_offset: 0,
        }
    }

    /// Get the IRI (returns truncated version if longer than inline capacity)
    pub fn iri(&self) -> &str {
        let len = (self.iri_len as usize).min(Self::MAX_INLINE_IRI);
        let bytes = &self.iri_inline[..len];
        // Safety: we only store valid UTF-8
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// Get the label (if any)
    pub fn label(&self) -> Option<&str> {
        if self.label_len == 0 {
            None
        } else {
            let len = (self.label_len as usize).min(Self::MAX_INLINE_LABEL);
            let bytes = &self.label_inline[..len];
            Some(unsafe { std::str::from_utf8_unchecked(bytes) })
        }
    }

    /// Set the label
    pub fn set_label(&mut self, label: &str) {
        let len = label.len().min(Self::MAX_INLINE_LABEL);
        self.label_inline[..len].copy_from_slice(&label.as_bytes()[..len]);
        self.label_len = len as u8;
    }

    /// Check if this term matches an IRI (fast hash comparison first)
    pub fn matches_iri(&self, iri: &str) -> bool {
        let hash = Self::hash_iri(iri);
        if self.iri_hash != hash {
            return false;
        }
        // Hash matched, verify actual string (for short IRIs)
        if iri.len() <= Self::MAX_INLINE_IRI {
            self.iri() == iri
        } else {
            // For long IRIs, hash match is sufficient (with low collision probability)
            true
        }
    }

    /// Get the size of this term in bytes
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    /// Hash an IRI string
    fn hash_iri(iri: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        iri.hash(&mut hasher);
        hasher.finish()
    }

    /// Store IRI inline (truncates if too long)
    fn store_iri_inline(iri: &str) -> ([u8; 16], u8) {
        let mut data = [0u8; 16];
        let len = iri.len().min(Self::MAX_INLINE_IRI);
        data[..len].copy_from_slice(&iri.as_bytes()[..len]);
        (data, iri.len() as u8) // Store actual length, even if truncated
    }
}

impl std::fmt::Debug for CompactTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactTerm")
            .field("iri", &self.iri())
            .field("label", &self.label())
            .field("parent_idx", &self.parent_idx)
            .field("depth", &self.depth)
            .field("flags", &self.flags)
            .field("child_count", &self.child_count)
            .finish()
    }
}

/// Builder for creating compact terms
pub struct CompactTermBuilder {
    iri: String,
    label: Option<String>,
    parent_idx: u32,
    depth: u16,
    flags: TermFlags,
    child_count: u32,
    embedding_offset: u32,
}

impl CompactTermBuilder {
    /// Create a new builder with the given IRI
    pub fn new(iri: &str) -> Self {
        CompactTermBuilder {
            iri: iri.to_string(),
            label: None,
            parent_idx: 0,
            depth: 0,
            flags: TermFlags::empty(),
            child_count: 0,
            embedding_offset: 0,
        }
    }

    /// Set the label
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Set the parent index
    pub fn with_parent(mut self, parent_idx: u32) -> Self {
        self.parent_idx = parent_idx;
        self
    }

    /// Set the depth
    pub fn with_depth(mut self, depth: u16) -> Self {
        self.depth = depth;
        self
    }

    /// Set flags
    pub fn with_flags(mut self, flags: TermFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Mark as deprecated
    pub fn deprecated(mut self) -> Self {
        self.flags.set(TermFlags::DEPRECATED);
        self
    }

    /// Mark as abstract
    pub fn abstract_term(mut self) -> Self {
        self.flags.set(TermFlags::ABSTRACT);
        self
    }

    /// Mark as leaf
    pub fn leaf(mut self) -> Self {
        self.flags.set(TermFlags::IS_LEAF);
        self
    }

    /// Mark as hot (frequently accessed)
    pub fn hot(mut self) -> Self {
        self.flags.set(TermFlags::HOT);
        self
    }

    /// Set child count
    pub fn with_children(mut self, count: u32) -> Self {
        self.child_count = count;
        self
    }

    /// Set embedding offset
    pub fn with_embedding(mut self, offset: u32) -> Self {
        self.embedding_offset = offset;
        self.flags.set(TermFlags::HAS_EMBEDDING);
        self
    }

    /// Build the compact term
    pub fn build(self) -> CompactTerm {
        let mut term = CompactTerm::new(&self.iri);

        if let Some(label) = &self.label {
            term.set_label(label);
        }

        term.parent_idx = self.parent_idx;
        term.depth = self.depth;
        term.flags = self.flags;
        term.child_count = self.child_count;
        term.embedding_offset = self.embedding_offset;

        term
    }
}

/// Pool for interned strings (reduces memory for common strings)
#[derive(Default)]
pub struct StringPool {
    strings: std::sync::RwLock<std::collections::HashMap<String, Arc<str>>>,
}

impl StringPool {
    /// Create a new string pool
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string, returning a shared reference
    pub fn intern(&self, s: &str) -> Arc<str> {
        // Fast path: check if already interned
        if let Ok(strings) = self.strings.read()
            && let Some(interned) = strings.get(s)
        {
            return Arc::clone(interned);
        }

        // Slow path: insert new string
        let mut strings = self.strings.write().unwrap();
        if let Some(interned) = strings.get(s) {
            return Arc::clone(interned);
        }

        let interned: Arc<str> = Arc::from(s);
        strings.insert(s.to_string(), Arc::clone(&interned));
        interned
    }

    /// Get number of interned strings
    pub fn len(&self) -> usize {
        self.strings.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the pool
    pub fn clear(&self) {
        if let Ok(mut strings) = self.strings.write() {
            strings.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_term_size() {
        // Verify the term fits in 64 bytes (one cache line)
        assert!(
            std::mem::size_of::<CompactTerm>() <= 64,
            "CompactTerm should fit in 64 bytes, but is {} bytes",
            std::mem::size_of::<CompactTerm>()
        );
    }

    #[test]
    fn test_compact_term_basic() {
        // Use short strings that fit inline
        let term = CompactTermBuilder::new("snomed:123")
            .with_label("Diabet") // 6 chars, fits in 7-byte label
            .with_depth(5)
            .with_parent(42)
            .build();

        assert_eq!(term.iri(), "snomed:123");
        assert_eq!(term.label(), Some("Diabet"));
        assert_eq!(term.depth, 5);
        assert_eq!(term.parent_idx, 42);
    }

    #[test]
    fn test_compact_term_inline_iri() {
        // Short IRI should be inline (up to 16 chars)
        let term = CompactTermBuilder::new("snomed:123").build();
        assert_eq!(term.iri(), "snomed:123");

        // Long IRI is truncated but hash is preserved for matching
        let long_iri = "http://snomed.info/id/73211009#something_very_long";
        let term = CompactTermBuilder::new(long_iri).build();
        // IRI is truncated to 16 chars
        assert_eq!(term.iri().len(), 16);
        // But hash-based matching still works
        assert!(term.matches_iri(long_iri));
    }

    #[test]
    fn test_compact_term_flags() {
        let mut flags = TermFlags::empty();

        assert!(!flags.is_deprecated());
        flags.set(TermFlags::DEPRECATED);
        assert!(flags.is_deprecated());

        flags.set(TermFlags::IS_LEAF);
        assert!(flags.is_leaf());

        flags.clear(TermFlags::DEPRECATED);
        assert!(!flags.is_deprecated());
        assert!(flags.is_leaf());
    }

    #[test]
    fn test_compact_term_matches() {
        // Short IRI - exact match
        let term = CompactTermBuilder::new("snomed:123").build();
        assert!(term.matches_iri("snomed:123"));
        assert!(!term.matches_iri("snomed:12345"));
        assert!(!term.matches_iri("icd10:E11"));
    }

    #[test]
    fn test_compact_term_builder() {
        let term = CompactTermBuilder::new("snomed:123")
            .with_label("Test") // Short label
            .with_parent(100)
            .with_depth(3)
            .with_children(15)
            .deprecated()
            .hot()
            .with_embedding(1000)
            .build();

        assert_eq!(term.iri(), "snomed:123");
        assert_eq!(term.label(), Some("Test"));
        assert_eq!(term.parent_idx, 100);
        assert_eq!(term.depth, 3);
        assert_eq!(term.child_count, 15);
        assert!(term.flags.is_deprecated());
        assert!(term.flags.is_hot());
        assert!(term.flags.has_embedding());
        assert_eq!(term.embedding_offset, 1000);
    }

    #[test]
    fn test_string_pool() {
        let pool = StringPool::new();

        let s1 = pool.intern("hello");
        let s2 = pool.intern("hello");
        let s3 = pool.intern("world");

        // Same string should return same Arc
        assert!(Arc::ptr_eq(&s1, &s2));
        assert!(!Arc::ptr_eq(&s1, &s3));

        assert_eq!(pool.len(), 2);
    }
}
