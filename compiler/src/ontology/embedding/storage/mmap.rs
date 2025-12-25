//! Memory-Mapped Embedding Storage
//!
//! Efficient storage for large embedding datasets using memory-mapped files.
//!
//! # File Format
//!
//! ```text
//! [Header: 64 bytes]
//!   - magic: u32 = 0x444D4245 ("DMBE")
//!   - version: u32
//!   - dimensions: u32
//!   - count: u64
//!   - index_offset: u64
//!   - reserved: [u8; 32]
//!
//! [Embeddings: count * (dimensions * 4) bytes]
//!   - embedding_0: [f32; dimensions]
//!   - embedding_1: [f32; dimensions]
//!   - ...
//!
//! [Index: variable]
//!   - IRI string table
//!   - IRI -> offset mapping
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use super::super::{Embedding, EmbeddingConfig, EmbeddingError, EmbeddingModel, EmbeddingStore};
use crate::ontology::loader::IRI;

const MAGIC: u32 = 0x444D4245; // "DMBE"
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;

/// Memory-mapped embedding store
///
/// For simplicity, this implementation uses file I/O rather than actual mmap.
/// In production, you'd use memmap2 crate for true memory mapping.
pub struct MmapStore {
    /// Path to the storage file
    path: PathBuf,

    /// Embedding dimensions
    dimensions: usize,

    /// Index: IRI -> offset in embeddings section
    index: RwLock<HashMap<IRI, u64>>,

    /// Number of embeddings
    count: RwLock<u64>,

    /// In-memory cache of embeddings (simplified version)
    cache: RwLock<HashMap<IRI, Embedding>>,
}

impl MmapStore {
    pub fn open(config: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
        std::fs::create_dir_all(&config.cache_dir)?;
        let path = config.cache_dir.join("embeddings.dmbe");
        Self::open_path(&path, config.dimensions)
    }

    pub fn open_path(path: &Path, dimensions: usize) -> Result<Self, EmbeddingError> {
        if path.exists() {
            Self::load(path, dimensions)
        } else {
            Self::create(path, dimensions)
        }
    }

    fn create(path: &Path, dimensions: usize) -> Result<Self, EmbeddingError> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create empty file with header
        let mut file = File::create(path)?;

        let header = Header {
            magic: MAGIC,
            version: VERSION,
            dimensions: dimensions as u32,
            count: 0,
            index_offset: HEADER_SIZE as u64,
        };

        file.write_all(&header.to_bytes())?;

        Ok(Self {
            path: path.to_path_buf(),
            dimensions,
            index: RwLock::new(HashMap::new()),
            count: RwLock::new(0),
            cache: RwLock::new(HashMap::new()),
        })
    }

    fn load(path: &Path, dimensions: usize) -> Result<Self, EmbeddingError> {
        let mut file = File::open(path)?;
        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;

        let header = Header::from_bytes(&header_bytes)?;

        if header.magic != MAGIC {
            return Err(EmbeddingError::InvalidFormat("Bad magic number".into()));
        }

        if header.dimensions as usize != dimensions {
            return Err(EmbeddingError::DimensionMismatch {
                expected: dimensions,
                found: header.dimensions as usize,
            });
        }

        // For simplicity, we'll load everything into the cache
        // In production, you'd use actual mmap
        let store = Self {
            path: path.to_path_buf(),
            dimensions,
            index: RwLock::new(HashMap::new()),
            count: RwLock::new(header.count),
            cache: RwLock::new(HashMap::new()),
        };

        // Load embeddings from file
        store.load_all_from_file(&mut file, &header)?;

        Ok(store)
    }

    fn load_all_from_file(&self, file: &mut File, header: &Header) -> Result<(), EmbeddingError> {
        use std::io::{Read, Seek, SeekFrom};

        let embedding_size = self.dimensions * 4; // f32 = 4 bytes
        let embeddings_section_size = header.count as usize * embedding_size;
        let index_offset = HEADER_SIZE + embeddings_section_size;

        // Seek to index section
        file.seek(SeekFrom::Start(index_offset as u64))?;

        let mut cache = self
            .cache
            .write()
            .map_err(|_| EmbeddingError::Io("Failed to acquire write lock".into()))?;
        let mut index = self
            .index
            .write()
            .map_err(|_| EmbeddingError::Io("Failed to acquire write lock".into()))?;

        // Read index entries
        for _ in 0..header.count {
            // Read IRI length (4 bytes)
            let mut len_bytes = [0u8; 4];
            if file.read_exact(&mut len_bytes).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;

            // Read IRI string
            let mut iri_bytes = vec![0u8; len];
            if file.read_exact(&mut iri_bytes).is_err() {
                break;
            }
            let iri_str = String::from_utf8_lossy(&iri_bytes).to_string();

            // Read embedding offset
            let mut offset_bytes = [0u8; 8];
            if file.read_exact(&mut offset_bytes).is_err() {
                break;
            }
            let offset = u64::from_le_bytes(offset_bytes);

            // Read the actual embedding vector
            let embedding_start = HEADER_SIZE as u64 + offset * embedding_size as u64;
            let current_pos = file.stream_position()?;
            file.seek(SeekFrom::Start(embedding_start))?;

            let mut vector_bytes = vec![0u8; embedding_size];
            if file.read_exact(&mut vector_bytes).is_ok() {
                let vector: Vec<f32> = vector_bytes
                    .chunks(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                let iri = IRI::new(&iri_str);
                let embedding = Embedding::new(iri.clone(), vector, EmbeddingModel::Pretrained);
                cache.insert(iri.clone(), embedding);
                index.insert(iri, offset);
            }

            // Seek back to continue reading index
            file.seek(SeekFrom::Start(current_pos))?;
        }

        Ok(())
    }

    /// Save all embeddings to file
    pub fn save(&self) -> Result<(), EmbeddingError> {
        let cache = self
            .cache
            .read()
            .map_err(|_| EmbeddingError::Io("Failed to acquire read lock".into()))?;

        let mut file = File::create(&self.path)?;

        // Write header
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            dimensions: self.dimensions as u32,
            count: cache.len() as u64,
            index_offset: 0, // Will update later
        };
        file.write_all(&header.to_bytes())?;

        // Write embeddings
        let mut offsets: Vec<(IRI, u64)> = Vec::new();
        for (idx, (iri, embedding)) in cache.iter().enumerate() {
            offsets.push((iri.clone(), idx as u64));
            for val in &embedding.vector {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // Write index
        for (iri, offset) in &offsets {
            let iri_bytes = iri.as_str().as_bytes();
            file.write_all(&(iri_bytes.len() as u32).to_le_bytes())?;
            file.write_all(iri_bytes)?;
            file.write_all(&offset.to_le_bytes())?;
        }

        Ok(())
    }
}

impl EmbeddingStore for MmapStore {
    fn get(&self, iri: &IRI) -> Result<Option<Embedding>, EmbeddingError> {
        if let Ok(cache) = self.cache.read() {
            Ok(cache.get(iri).cloned())
        } else {
            Ok(None)
        }
    }

    fn put(&mut self, iri: &IRI, embedding: &Embedding) -> Result<(), EmbeddingError> {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(iri.clone(), embedding.clone());
        }
        if let Ok(mut count) = self.count.write() {
            *count += 1;
        }
        Ok(())
    }

    fn delete(&mut self, iri: &IRI) -> Result<(), EmbeddingError> {
        if let Ok(mut cache) = self.cache.write()
            && cache.remove(iri).is_some()
            && let Ok(mut count) = self.count.write()
        {
            *count = count.saturating_sub(1);
        }
        Ok(())
    }

    fn all(&self) -> Result<Vec<Embedding>, EmbeddingError> {
        if let Ok(cache) = self.cache.read() {
            Ok(cache.values().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    fn count(&self) -> Result<usize, EmbeddingError> {
        if let Ok(cache) = self.cache.read() {
            Ok(cache.len())
        } else {
            Ok(0)
        }
    }

    fn contains(&self, iri: &IRI) -> Result<bool, EmbeddingError> {
        if let Ok(cache) = self.cache.read() {
            Ok(cache.contains_key(iri))
        } else {
            Ok(false)
        }
    }
}

#[derive(Debug)]
struct Header {
    magic: u32,
    version: u32,
    dimensions: u32,
    count: u64,
    index_offset: u64,
}

impl Header {
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.dimensions.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.count.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.index_offset.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, EmbeddingError> {
        if bytes.len() < HEADER_SIZE {
            return Err(EmbeddingError::InvalidFormat("Header too short".into()));
        }

        Ok(Self {
            magic: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            version: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            dimensions: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            count: u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
            index_offset: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
        })
    }
}
