//! B-tree Index Builder for Ontology Terms
//!
//! Provides O(log n) lookup for 15M+ ontology term IRIs.
//!
//! # Design
//!
//! The index is built in two phases:
//! 1. **Accumulation**: Collect all (IRI hash, term data) pairs
//! 2. **Bulk construction**: Sort and build balanced B-tree
//!
//! # Storage Format
//!
//! ```text
//! Header:
//!   magic: u32 (0x44454D54 = "DEMT")
//!   version: u32
//!   entry_count: u64
//!   root_offset: u64
//!
//! Nodes:
//!   is_leaf: u8
//!   count: u16
//!   keys: [u64; B-1]      // IRI hashes
//!   values: [u64; B-1]    // Offsets (leaf) or child offsets (internal)
//!   children: [u64; B]    // Only for internal nodes
//! ```

use std::hash::{Hash, Hasher};
use std::io::{Read, Write};

/// Magic number for index files
const INDEX_MAGIC: u32 = 0x44454D54; // "DEMT"

/// Current index format version
const INDEX_VERSION: u32 = 1;

/// B-tree branching factor
const B: usize = 64;

/// Entry in the term index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    /// Hash of the IRI for fast comparison
    pub iri_hash: u64,
    /// Offset into the term data store
    pub data_offset: u64,
    /// Size of the term data
    pub data_size: u32,
    /// Interned IRI handle (for verification)
    pub iri_handle: u32,
}

impl IndexEntry {
    /// Create a new index entry
    pub fn new(iri_hash: u64, data_offset: u64, data_size: u32, iri_handle: u32) -> Self {
        Self {
            iri_hash,
            data_offset,
            data_size,
            iri_handle,
        }
    }
}

/// B-tree node
#[derive(Debug, Clone)]
enum BTreeNode {
    /// Leaf node containing entries
    Leaf { entries: Vec<IndexEntry> },
    /// Internal node with keys and children
    Internal {
        keys: Vec<u64>,
        children: Vec<Box<BTreeNode>>,
    },
}

impl BTreeNode {
    fn new_leaf() -> Self {
        BTreeNode::Leaf {
            entries: Vec::new(),
        }
    }

    fn new_internal() -> Self {
        BTreeNode::Internal {
            keys: Vec::new(),
            children: Vec::new(),
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, BTreeNode::Leaf { .. })
    }

    fn len(&self) -> usize {
        match self {
            BTreeNode::Leaf { entries } => entries.len(),
            BTreeNode::Internal { keys, .. } => keys.len(),
        }
    }
}

/// B-tree index for ontology term lookup
pub struct BTreeIndex {
    /// Root node of the tree
    root: Option<Box<BTreeNode>>,
    /// Number of entries in the index
    entry_count: usize,
    /// Height of the tree
    height: usize,
}

impl BTreeIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self {
            root: None,
            entry_count: 0,
            height: 0,
        }
    }

    /// Build an index from a sorted iterator of entries
    pub fn build_from_sorted<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = IndexEntry>,
    {
        let entries: Vec<_> = entries.into_iter().collect();
        let entry_count = entries.len();

        if entries.is_empty() {
            return Self::new();
        }

        // Build leaf nodes
        let mut leaves: Vec<Box<BTreeNode>> = entries
            .chunks(B - 1)
            .map(|chunk| {
                Box::new(BTreeNode::Leaf {
                    entries: chunk.to_vec(),
                })
            })
            .collect();

        let mut height = 1;

        // Build internal nodes bottom-up
        while leaves.len() > 1 {
            let mut new_level = Vec::new();

            for chunk in leaves.chunks(B) {
                let mut keys = Vec::new();
                let mut children = Vec::new();

                for (i, child) in chunk.iter().enumerate() {
                    if i > 0 {
                        // Key is the minimum key of this child
                        let min_key = Self::get_min_key(child);
                        keys.push(min_key);
                    }
                    children.push(child.clone());
                }

                new_level.push(Box::new(BTreeNode::Internal { keys, children }));
            }

            leaves = new_level;
            height += 1;
        }

        Self {
            root: leaves.into_iter().next(),
            entry_count,
            height,
        }
    }

    /// Get the minimum key in a subtree
    fn get_min_key(node: &BTreeNode) -> u64 {
        match node {
            BTreeNode::Leaf { entries } => entries.first().map(|e| e.iri_hash).unwrap_or(0),
            BTreeNode::Internal { children, .. } => {
                children.first().map(|c| Self::get_min_key(c)).unwrap_or(0)
            }
        }
    }

    /// Look up an entry by IRI hash
    pub fn lookup(&self, iri_hash: u64) -> Option<&IndexEntry> {
        let mut node = self.root.as_ref()?;

        loop {
            match node.as_ref() {
                BTreeNode::Leaf { entries } => {
                    // Binary search in leaf
                    return entries
                        .binary_search_by_key(&iri_hash, |e| e.iri_hash)
                        .ok()
                        .map(|i| &entries[i]);
                }
                BTreeNode::Internal { keys, children } => {
                    // Find the appropriate child
                    let idx = keys.partition_point(|&k| k <= iri_hash);
                    node = &children[idx];
                }
            }
        }
    }

    /// Look up all entries with the given IRI hash (handles collisions)
    pub fn lookup_all(&self, iri_hash: u64) -> Vec<&IndexEntry> {
        let mut results = Vec::new();

        if let Some(ref root) = self.root {
            self.collect_matching(root, iri_hash, &mut results);
        }

        results
    }

    fn collect_matching<'a>(
        &'a self,
        node: &'a BTreeNode,
        iri_hash: u64,
        results: &mut Vec<&'a IndexEntry>,
    ) {
        match node {
            BTreeNode::Leaf { entries } => {
                for entry in entries {
                    if entry.iri_hash == iri_hash {
                        results.push(entry);
                    }
                }
            }
            BTreeNode::Internal { keys, children } => {
                // Need to check all potentially matching children
                let start = keys.partition_point(|&k| k < iri_hash);
                let end = keys.partition_point(|&k| k <= iri_hash);

                for i in start..=end.min(children.len() - 1) {
                    self.collect_matching(&children[i], iri_hash, results);
                }
            }
        }
    }

    /// Get the number of entries in the index
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Get the height of the tree
    pub fn height(&self) -> usize {
        self.height
    }

    /// Iterate over all entries in order
    pub fn iter(&self) -> impl Iterator<Item = &IndexEntry> {
        IndexIterator::new(self)
    }
}

impl Default for BTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over index entries
struct IndexIterator<'a> {
    stack: Vec<(&'a BTreeNode, usize)>,
    current_leaf: Option<(&'a [IndexEntry], usize)>,
}

impl<'a> IndexIterator<'a> {
    fn new(index: &'a BTreeIndex) -> Self {
        let mut iter = Self {
            stack: Vec::new(),
            current_leaf: None,
        };

        if let Some(ref root) = index.root {
            iter.descend_left(root);
        }

        iter
    }

    fn descend_left(&mut self, mut node: &'a BTreeNode) {
        loop {
            match node {
                BTreeNode::Leaf { entries } => {
                    if !entries.is_empty() {
                        self.current_leaf = Some((entries, 0));
                    }
                    break;
                }
                BTreeNode::Internal { children, .. } => {
                    if children.is_empty() {
                        break;
                    }
                    self.stack.push((node, 0));
                    node = &children[0];
                }
            }
        }
    }
}

impl<'a> Iterator for IndexIterator<'a> {
    type Item = &'a IndexEntry;

    fn next(&mut self) -> Option<Self::Item> {
        // Try to get from current leaf
        if let Some((entries, idx)) = self.current_leaf.take() {
            let entry = &entries[idx];

            if idx + 1 < entries.len() {
                self.current_leaf = Some((entries, idx + 1));
            } else {
                // Need to move to next leaf
                self.advance_to_next_leaf();
            }

            return Some(entry);
        }

        None
    }
}

impl<'a> IndexIterator<'a> {
    fn advance_to_next_leaf(&mut self) {
        while let Some((node, child_idx)) = self.stack.pop() {
            if let BTreeNode::Internal { children, .. } = node
                && child_idx + 1 < children.len()
            {
                self.stack.push((node, child_idx + 1));
                self.descend_left(&children[child_idx + 1]);
                return;
            }
        }
    }
}

/// Builder for constructing B-tree indices incrementally
pub struct BTreeIndexBuilder {
    entries: Vec<IndexEntry>,
    sorted: bool,
}

impl BTreeIndexBuilder {
    /// Create a new index builder
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            sorted: true,
        }
    }

    /// Create a builder with expected capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            sorted: true,
        }
    }

    /// Add an entry to the builder
    pub fn add(&mut self, entry: IndexEntry) {
        if let Some(last) = self.entries.last()
            && entry.iri_hash < last.iri_hash
        {
            self.sorted = false;
        }
        self.entries.push(entry);
    }

    /// Add an entry from components
    pub fn add_term(&mut self, iri_hash: u64, data_offset: u64, data_size: u32, iri_handle: u32) {
        self.add(IndexEntry::new(
            iri_hash,
            data_offset,
            data_size,
            iri_handle,
        ));
    }

    /// Build the index
    pub fn build(mut self) -> BTreeIndex {
        if !self.sorted {
            self.entries.sort_by_key(|e| e.iri_hash);
        }
        BTreeIndex::build_from_sorted(self.entries)
    }

    /// Get current entry count
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if builder is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for BTreeIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a hash for an IRI string
pub fn hash_iri(iri: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    iri.hash(&mut hasher);
    hasher.finish()
}

/// Serializable index format for disk storage
pub struct SerializedIndex {
    /// Header information
    pub magic: u32,
    pub version: u32,
    pub entry_count: u64,
    /// Entries in sorted order
    pub entries: Vec<IndexEntry>,
}

impl SerializedIndex {
    /// Create from a B-tree index
    pub fn from_index(index: &BTreeIndex) -> Self {
        Self {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION,
            entry_count: index.len() as u64,
            entries: index.iter().cloned().collect(),
        }
    }

    /// Write to a file
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Write header
        writer.write_all(&self.magic.to_le_bytes())?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.entry_count.to_le_bytes())?;

        // Write entries
        for entry in &self.entries {
            writer.write_all(&entry.iri_hash.to_le_bytes())?;
            writer.write_all(&entry.data_offset.to_le_bytes())?;
            writer.write_all(&entry.data_size.to_le_bytes())?;
            writer.write_all(&entry.iri_handle.to_le_bytes())?;
        }

        Ok(())
    }

    /// Read from a file
    pub fn read_from<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Read header
        reader.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);

        if magic != INDEX_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid index magic number",
            ));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);

        if version != INDEX_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported index version: {}", version),
            ));
        }

        reader.read_exact(&mut buf8)?;
        let entry_count = u64::from_le_bytes(buf8);

        // Read entries
        let mut entries = Vec::with_capacity(entry_count as usize);
        for _ in 0..entry_count {
            reader.read_exact(&mut buf8)?;
            let iri_hash = u64::from_le_bytes(buf8);

            reader.read_exact(&mut buf8)?;
            let data_offset = u64::from_le_bytes(buf8);

            reader.read_exact(&mut buf4)?;
            let data_size = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let iri_handle = u32::from_le_bytes(buf4);

            entries.push(IndexEntry::new(
                iri_hash,
                data_offset,
                data_size,
                iri_handle,
            ));
        }

        Ok(Self {
            magic,
            version,
            entry_count,
            entries,
        })
    }

    /// Convert to a B-tree index
    pub fn into_index(self) -> BTreeIndex {
        BTreeIndex::build_from_sorted(self.entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_iri() {
        let hash1 = hash_iri("http://example.org/term1");
        let hash2 = hash_iri("http://example.org/term2");
        let hash3 = hash_iri("http://example.org/term1");

        assert_ne!(hash1, hash2);
        assert_eq!(hash1, hash3);
    }

    #[test]
    fn test_empty_index() {
        let index = BTreeIndex::new();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(index.lookup(12345).is_none());
    }

    #[test]
    fn test_single_entry() {
        let mut builder = BTreeIndexBuilder::new();
        builder.add_term(hash_iri("http://example.org/term1"), 0, 100, 1);

        let index = builder.build();

        assert_eq!(index.len(), 1);

        let entry = index.lookup(hash_iri("http://example.org/term1"));
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().data_offset, 0);
    }

    #[test]
    fn test_many_entries() {
        let mut builder = BTreeIndexBuilder::with_capacity(1000);

        for i in 0..1000 {
            let iri = format!("http://example.org/term{}", i);
            builder.add_term(hash_iri(&iri), i as u64 * 100, 50, i as u32);
        }

        let index = builder.build();

        assert_eq!(index.len(), 1000);
        assert!(
            index.height() > 1,
            "Expected tree height > 1 for 1000 entries"
        );

        // Look up some entries
        for i in [0, 100, 500, 999] {
            let iri = format!("http://example.org/term{}", i);
            let entry = index.lookup(hash_iri(&iri));
            assert!(entry.is_some(), "Entry {} not found", i);
            assert_eq!(entry.unwrap().iri_handle, i as u32);
        }

        // Look up non-existent
        assert!(
            index
                .lookup(hash_iri("http://nonexistent.org/term"))
                .is_none()
        );
    }

    #[test]
    fn test_iteration() {
        let mut builder = BTreeIndexBuilder::new();

        for i in 0..100 {
            builder.add_term(i as u64, i as u64, 10, i as u32);
        }

        let index = builder.build();
        let entries: Vec<_> = index.iter().collect();

        assert_eq!(entries.len(), 100);

        // Should be in sorted order
        for i in 1..entries.len() {
            assert!(entries[i].iri_hash >= entries[i - 1].iri_hash);
        }
    }

    #[test]
    fn test_unsorted_builder() {
        let mut builder = BTreeIndexBuilder::new();

        // Add in reverse order
        for i in (0..100).rev() {
            builder.add_term(i as u64, i as u64, 10, i as u32);
        }

        let index = builder.build();

        // Should still work
        assert_eq!(index.len(), 100);

        let entry = index.lookup(50);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().iri_handle, 50);
    }

    #[test]
    fn test_serialization() {
        let mut builder = BTreeIndexBuilder::new();

        for i in 0..100 {
            let iri = format!("http://example.org/term{}", i);
            builder.add_term(hash_iri(&iri), i as u64 * 100, 50, i as u32);
        }

        let index = builder.build();

        // Serialize
        let serialized = SerializedIndex::from_index(&index);
        let mut buffer = Vec::new();
        serialized.write_to(&mut buffer).unwrap();

        // Deserialize
        let mut cursor = std::io::Cursor::new(buffer);
        let loaded = SerializedIndex::read_from(&mut cursor).unwrap();
        let loaded_index = loaded.into_index();

        // Verify
        assert_eq!(loaded_index.len(), 100);

        for i in 0..100 {
            let iri = format!("http://example.org/term{}", i);
            let entry = loaded_index.lookup(hash_iri(&iri));
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().iri_handle, i as u32);
        }
    }

    #[test]
    fn test_collision_handling() {
        // Create entries with same hash (simulated collision)
        let hash = 12345u64;

        let mut builder = BTreeIndexBuilder::new();
        builder.add_term(hash, 0, 100, 1);
        builder.add_term(hash, 100, 100, 2);
        builder.add_term(hash, 200, 100, 3);

        let index = builder.build();

        let all = index.lookup_all(hash);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_large_index_height() {
        // B-tree with B=64 should have height ceil(log_64(n))
        let mut builder = BTreeIndexBuilder::with_capacity(10000);

        for i in 0..10000 {
            builder.add_term(i as u64, i as u64, 10, i as u32);
        }

        let index = builder.build();

        // 10000 entries with B=64 should have height around 3
        // ceil(log_64(10000)) ≈ 2.2, so height should be 2-3
        assert!(
            index.height() <= 4,
            "Height {} is too large for 10000 entries",
            index.height()
        );
        assert!(
            index.height() >= 2,
            "Height {} is too small for 10000 entries",
            index.height()
        );
    }
}
