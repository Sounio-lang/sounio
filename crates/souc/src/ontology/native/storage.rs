//! Native Ontology Storage
//!
//! Compact binary format for ontology data with string interning.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;

use crate::ontology::{OntologyError, OntologyResult};

use super::{DONTOLOGY_MAGIC, DONTOLOGY_VERSION};

/// Header of a .dontology file
#[derive(Debug, Clone)]
pub struct NativeStoreHeader {
    /// Ontology identifier
    pub ontology_id: String,
    /// Version string
    pub version: String,
    /// Number of concepts
    pub concept_count: u32,
}

/// A single concept entry in the store
#[derive(Debug, Clone)]
pub struct ConceptEntry {
    /// Index into string table for CURIE
    pub curie_idx: u32,
    /// Index into string table for label (None = no label)
    pub label_idx: Option<u32>,
    /// Index into string table for definition (None = no definition)
    pub definition_idx: Option<u32>,
    /// Index into string table for parent CURIE (None = root)
    pub parent_idx: Option<u32>,
    /// Bit flags (deprecated, abstract, etc.)
    pub flags: u32,
}

impl ConceptEntry {
    /// Write entry to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.curie_idx.to_le_bytes())?;
        writer.write_all(&self.label_idx.unwrap_or(u32::MAX).to_le_bytes())?;
        writer.write_all(&self.definition_idx.unwrap_or(u32::MAX).to_le_bytes())?;
        writer.write_all(&self.parent_idx.unwrap_or(u32::MAX).to_le_bytes())?;
        writer.write_all(&self.flags.to_le_bytes())?;
        Ok(())
    }

    /// Read entry from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf = [0u8; 4];

        reader.read_exact(&mut buf)?;
        let curie_idx = u32::from_le_bytes(buf);

        reader.read_exact(&mut buf)?;
        let label_idx = u32::from_le_bytes(buf);
        let label_idx = if label_idx == u32::MAX {
            None
        } else {
            Some(label_idx)
        };

        reader.read_exact(&mut buf)?;
        let definition_idx = u32::from_le_bytes(buf);
        let definition_idx = if definition_idx == u32::MAX {
            None
        } else {
            Some(definition_idx)
        };

        reader.read_exact(&mut buf)?;
        let parent_idx = u32::from_le_bytes(buf);
        let parent_idx = if parent_idx == u32::MAX {
            None
        } else {
            Some(parent_idx)
        };

        reader.read_exact(&mut buf)?;
        let flags = u32::from_le_bytes(buf);

        Ok(Self {
            curie_idx,
            label_idx,
            definition_idx,
            parent_idx,
            flags,
        })
    }
}

/// Interned string table
#[derive(Debug, Clone, Default)]
pub struct StringTable {
    /// All strings concatenated
    data: Vec<u8>,
    /// Offset and length for each string index
    offsets: Vec<(u32, u32)>,
    /// Reverse lookup: string -> index
    lookup: HashMap<String, u32>,
}

impl StringTable {
    /// Create a new empty string table
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string, returning its index
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.lookup.get(s) {
            return idx;
        }

        let idx = self.offsets.len() as u32;
        let offset = self.data.len() as u32;
        let len = s.len() as u32;

        self.data.extend_from_slice(s.as_bytes());
        self.offsets.push((offset, len));
        self.lookup.insert(s.to_string(), idx);

        idx
    }

    /// Get a string by index
    pub fn get(&self, idx: u32) -> Option<&str> {
        let (offset, len) = self.offsets.get(idx as usize)?;
        let bytes = &self.data[*offset as usize..(*offset + *len) as usize];
        std::str::from_utf8(bytes).ok()
    }

    /// Number of strings
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Write to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write string count
        writer.write_all(&(self.offsets.len() as u32).to_le_bytes())?;

        // Write data length
        writer.write_all(&(self.data.len() as u32).to_le_bytes())?;

        // Write offsets
        for (offset, len) in &self.offsets {
            writer.write_all(&offset.to_le_bytes())?;
            writer.write_all(&len.to_le_bytes())?;
        }

        // Write data
        writer.write_all(&self.data)?;

        Ok(())
    }

    /// Read from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];

        // Read string count
        reader.read_exact(&mut buf4)?;
        let count = u32::from_le_bytes(buf4) as usize;

        // Read data length
        reader.read_exact(&mut buf4)?;
        let data_len = u32::from_le_bytes(buf4) as usize;

        // Read offsets
        let mut offsets = Vec::with_capacity(count);
        for _ in 0..count {
            reader.read_exact(&mut buf4)?;
            let offset = u32::from_le_bytes(buf4);
            reader.read_exact(&mut buf4)?;
            let len = u32::from_le_bytes(buf4);
            offsets.push((offset, len));
        }

        // Read data
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        // Build lookup
        let mut lookup = HashMap::new();
        for (idx, (offset, len)) in offsets.iter().enumerate() {
            let s = std::str::from_utf8(&data[*offset as usize..(*offset + *len) as usize])
                .unwrap_or("")
                .to_string();
            lookup.insert(s, idx as u32);
        }

        Ok(Self {
            data,
            offsets,
            lookup,
        })
    }
}

/// The native ontology store
#[derive(Debug)]
pub struct NativeStore {
    /// File header
    pub header: NativeStoreHeader,
    /// String table
    pub strings: StringTable,
    /// Concept entries
    pub concepts: Vec<ConceptEntry>,
    /// Index: CURIE -> concept index
    curie_index: HashMap<String, usize>,
    /// Index: label prefix -> concept indices (for search)
    label_index: Vec<(String, usize)>,
}

impl NativeStore {
    /// Load from a .dontology file
    pub fn load(path: &Path) -> OntologyResult<Self> {
        let file = File::open(path).map_err(|e| {
            OntologyError::DatabaseError(format!("Failed to open {:?}: {}", path, e))
        })?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| OntologyError::DatabaseError(format!("Failed to read magic: {}", e)))?;
        if &magic != DONTOLOGY_MAGIC {
            return Err(OntologyError::DatabaseError(format!(
                "Invalid magic bytes: expected {:?}, got {:?}",
                DONTOLOGY_MAGIC, magic
            )));
        }

        // Read and verify version
        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| OntologyError::DatabaseError(format!("Failed to read version: {}", e)))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != DONTOLOGY_VERSION {
            return Err(OntologyError::DatabaseError(format!(
                "Unsupported version: expected {}, got {}",
                DONTOLOGY_VERSION, version
            )));
        }

        // Read ontology ID
        let mut len_bytes = [0u8; 4];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        let id_len = u32::from_le_bytes(len_bytes) as usize;
        let mut id_bytes = vec![0u8; id_len];
        reader
            .read_exact(&mut id_bytes)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        let ontology_id =
            String::from_utf8(id_bytes).map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        // Read version string
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        let ver_len = u32::from_le_bytes(len_bytes) as usize;
        let mut ver_bytes = vec![0u8; ver_len];
        reader
            .read_exact(&mut ver_bytes)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        let ont_version = String::from_utf8(ver_bytes)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        // Read concept count
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        let concept_count = u32::from_le_bytes(len_bytes);

        let header = NativeStoreHeader {
            ontology_id,
            version: ont_version,
            concept_count,
        };

        // Read string table
        let strings = StringTable::read_from(&mut reader).map_err(|e| {
            OntologyError::DatabaseError(format!("Failed to read string table: {}", e))
        })?;

        // Read concepts
        let mut concepts = Vec::with_capacity(concept_count as usize);
        for _ in 0..concept_count {
            let entry = ConceptEntry::read_from(&mut reader).map_err(|e| {
                OntologyError::DatabaseError(format!("Failed to read concept: {}", e))
            })?;
            concepts.push(entry);
        }

        // Build indices
        let mut curie_index = HashMap::new();
        let mut label_index = Vec::new();

        for (idx, entry) in concepts.iter().enumerate() {
            if let Some(curie) = strings.get(entry.curie_idx) {
                curie_index.insert(curie.to_string(), idx);
            }
            if let Some(label_idx) = entry.label_idx
                && let Some(label) = strings.get(label_idx)
            {
                label_index.push((label.to_lowercase(), idx));
            }
        }

        // Sort label index for binary search
        label_index.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(Self {
            header,
            strings,
            concepts,
            curie_index,
            label_index,
        })
    }

    /// Get concept by CURIE
    pub fn get_concept(&self, curie: &str) -> Option<&ConceptEntry> {
        let idx = self.curie_index.get(curie)?;
        self.concepts.get(*idx)
    }

    /// Get label for a concept
    pub fn get_label(&self, curie: &str) -> Option<&str> {
        let entry = self.get_concept(curie)?;
        let label_idx = entry.label_idx?;
        self.strings.get(label_idx)
    }

    /// Get definition for a concept
    pub fn get_definition(&self, curie: &str) -> Option<&str> {
        let entry = self.get_concept(curie)?;
        let def_idx = entry.definition_idx?;
        self.strings.get(def_idx)
    }

    /// Get parent CURIE for a concept
    pub fn get_parent(&self, curie: &str) -> Option<&str> {
        let entry = self.get_concept(curie)?;
        let parent_idx = entry.parent_idx?;
        self.strings.get(parent_idx)
    }

    /// Search by label prefix
    pub fn search_by_label(&self, prefix: &str, limit: usize) -> Vec<(&str, &str)> {
        let prefix_lower = prefix.to_lowercase();
        let mut results = Vec::new();

        // Binary search for starting position
        let start = self
            .label_index
            .partition_point(|(label, _)| label.as_str() < prefix_lower.as_str());

        for (label, idx) in self.label_index[start..].iter() {
            if !label.starts_with(&prefix_lower) {
                break;
            }
            if results.len() >= limit {
                break;
            }

            if let Some(entry) = self.concepts.get(*idx)
                && let Some(curie) = self.strings.get(entry.curie_idx)
                && let Some(label_idx) = entry.label_idx
                && let Some(label) = self.strings.get(label_idx)
            {
                results.push((curie, label));
            }
        }

        results
    }

    /// Get CURIE for a concept at index
    pub fn curie_at(&self, idx: usize) -> Option<&str> {
        let entry = self.concepts.get(idx)?;
        self.strings.get(entry.curie_idx)
    }

    /// Get parent index for a concept
    pub fn parent_idx(&self, idx: usize) -> Option<usize> {
        let entry = self.concepts.get(idx)?;
        let parent_curie_idx = entry.parent_idx?;
        let parent_curie = self.strings.get(parent_curie_idx)?;
        self.curie_index.get(parent_curie).copied()
    }

    /// Create an empty store (for testing)
    pub fn empty() -> Self {
        Self {
            header: NativeStoreHeader {
                ontology_id: String::new(),
                version: String::new(),
                concept_count: 0,
            },
            strings: StringTable::new(),
            concepts: Vec::new(),
            curie_index: HashMap::new(),
            label_index: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_table() {
        let mut table = StringTable::new();

        let idx1 = table.intern("hello");
        let idx2 = table.intern("world");
        let idx3 = table.intern("hello"); // Should reuse

        assert_eq!(idx1, idx3);
        assert_ne!(idx1, idx2);
        assert_eq!(table.get(idx1), Some("hello"));
        assert_eq!(table.get(idx2), Some("world"));
    }

    #[test]
    fn test_string_table_roundtrip() {
        let mut table = StringTable::new();
        table.intern("foo");
        table.intern("bar");
        table.intern("baz");

        let mut buf = Vec::new();
        table.write_to(&mut buf).unwrap();

        let loaded = StringTable::read_from(&mut buf.as_slice()).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.get(0), Some("foo"));
        assert_eq!(loaded.get(1), Some("bar"));
        assert_eq!(loaded.get(2), Some("baz"));
    }

    #[test]
    fn test_concept_entry_roundtrip() {
        let entry = ConceptEntry {
            curie_idx: 0,
            label_idx: Some(1),
            definition_idx: None,
            parent_idx: Some(2),
            flags: 0,
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();

        let loaded = ConceptEntry::read_from(&mut buf.as_slice()).unwrap();
        assert_eq!(loaded.curie_idx, 0);
        assert_eq!(loaded.label_idx, Some(1));
        assert_eq!(loaded.definition_idx, None);
        assert_eq!(loaded.parent_idx, Some(2));
    }
}
