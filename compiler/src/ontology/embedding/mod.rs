//! Ontological Embedding Space
//!
//! This module implements vector embeddings for ontological terms,
//! enabling geometric semantic distance calculation.
//!
//! # Embedding Models
//!
//! We support multiple embedding approaches:
//!
//! 1. **Structural**: Graph-based embeddings from ontology structure (OWL2Vec* style)
//! 2. **Textual**: From labels, definitions, synonyms
//! 3. **Hybrid**: Combination of structural and textual
//! 4. **Pretrained**: Load existing embeddings (BioConceptVec, etc.)
//!
//! # The Key Insight
//!
//! Embeddings capture semantic relationships that hierarchy misses:
//! - Synonymy: "heart" ~ "cardiac organ"
//! - Association: "aspirin" ~ "pain relief"
//! - Analogy: "DNA" - "nucleus" + "mitochondria" ~ "mtDNA"
//!
//! These relationships become part of the type system.

pub mod models;
pub mod simd;
pub mod storage;

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::RwLock;

use crate::ontology::loader::IRI;

/// Embedding-specific errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Embedding not found for IRI: {0}")]
    NotFound(IRI),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("Unknown format: {0}")]
    UnknownFormat(String),

    #[error("No pretrained path specified")]
    NoPretrained,

    #[error("Index not built")]
    IndexNotBuilt,

    #[error("Not initialized")]
    NotInitialized,

    #[error("Invalid offset: {0}")]
    InvalidOffset(u64),

    #[error("Operation not supported: {0}")]
    OperationNotSupported(String),

    #[error("Generation error: {0}")]
    GenerationError(String),
}

impl From<std::io::Error> for EmbeddingError {
    fn from(e: std::io::Error) -> Self {
        EmbeddingError::Io(e.to_string())
    }
}

/// Configuration for embedding generation and storage
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Embedding dimensionality
    pub dimensions: usize,

    /// Which model to use for generation
    pub model: EmbeddingModel,

    /// Storage backend type
    pub storage: EmbeddingStorageType,

    /// Cache size for hot embeddings
    pub cache_size: usize,

    /// Whether to generate embeddings on-demand
    pub lazy_generation: bool,

    /// Path to pre-trained embeddings
    pub pretrained_path: Option<PathBuf>,

    /// Cache directory for storage
    pub cache_dir: PathBuf,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimensions: 256,
            model: EmbeddingModel::Hybrid,
            storage: EmbeddingStorageType::Memory,
            cache_size: 100_000,
            lazy_generation: true,
            pretrained_path: None,
            cache_dir: PathBuf::from(".sounio/embeddings"),
        }
    }
}

/// Embedding model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// Graph-based: OWL2Vec* style random walks
    Structural,

    /// Text-based: from labels and definitions
    Textual,

    /// Combination of structural and textual
    Hybrid,

    /// Load from pre-trained file
    Pretrained,

    /// Random (for testing)
    Random,
}

/// Storage backend types
#[derive(Debug, Clone)]
pub enum EmbeddingStorageType {
    /// In-memory HashMap
    Memory,

    /// Memory-mapped file (efficient for large datasets)
    Mmap,

    /// Specialized vector database path
    VectorDb { path: PathBuf },
}

/// A semantic embedding vector
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The IRI this embedding represents
    pub iri: IRI,

    /// The vector (normalized to unit length)
    pub vector: Vec<f32>,

    /// Which model produced this
    pub model: EmbeddingModel,

    /// Confidence in this embedding (0.0 - 1.0)
    pub confidence: f64,
}

impl Embedding {
    /// Create a new embedding (auto-normalizes)
    pub fn new(iri: IRI, vector: Vec<f32>, model: EmbeddingModel) -> Self {
        let mut emb = Self {
            iri,
            vector,
            model,
            confidence: 1.0,
        };
        emb.normalize();
        emb
    }

    /// Create with specific confidence
    pub fn with_confidence(
        iri: IRI,
        vector: Vec<f32>,
        model: EmbeddingModel,
        confidence: f64,
    ) -> Self {
        let mut emb = Self {
            iri,
            vector,
            model,
            confidence,
        };
        emb.normalize();
        emb
    }

    /// Normalize to unit length
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
    }

    /// Get dimensionality
    pub fn dimensions(&self) -> usize {
        self.vector.len()
    }

    /// Cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        debug_assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "Embedding dimensions must match"
        );

        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Euclidean distance
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        debug_assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "Embedding dimensions must match"
        );

        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Convert cosine similarity to semantic distance (0 = identical, 1 = orthogonal)
    pub fn to_semantic_distance(&self, other: &Self) -> f64 {
        let sim = self.cosine_similarity(other);
        // Cosine similarity: -1 to 1
        // Semantic distance: 0 to 1
        // sim=1 -> dist=0, sim=0 -> dist=0.5, sim=-1 -> dist=1
        ((1.0 - sim) / 2.0).clamp(0.0, 1.0) as f64
    }

    /// Vector addition (for analogy operations)
    pub fn add(&self, other: &Self) -> Vec<f32> {
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Vector subtraction (for analogy operations)
    pub fn subtract(&self, other: &Self) -> Vec<f32> {
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a - b)
            .collect()
    }
}

/// Trait for embedding storage backends
pub trait EmbeddingStore: Send + Sync {
    fn get(&self, iri: &IRI) -> Result<Option<Embedding>, EmbeddingError>;
    fn put(&mut self, iri: &IRI, embedding: &Embedding) -> Result<(), EmbeddingError>;
    fn delete(&mut self, iri: &IRI) -> Result<(), EmbeddingError>;
    fn all(&self) -> Result<Vec<Embedding>, EmbeddingError>;
    fn count(&self) -> Result<usize, EmbeddingError>;
    fn contains(&self, iri: &IRI) -> Result<bool, EmbeddingError>;
}

/// Trait for embedding generation
pub trait EmbeddingGenerator: Send + Sync {
    fn generate(&self, iri: &IRI) -> Result<Embedding, EmbeddingError>;
    fn generate_batch(&self, iris: &[IRI]) -> Result<Vec<Embedding>, EmbeddingError>;
    fn dimensions(&self) -> usize;
}

/// The main embedding space
pub struct EmbeddingSpace {
    config: EmbeddingConfig,

    /// Storage backend
    storage: Box<dyn EmbeddingStore>,

    /// Hot cache for frequently accessed embeddings
    cache: RwLock<lru::LruCache<IRI, Embedding>>,

    /// Embedding generator (for lazy generation)
    generator: Option<Box<dyn EmbeddingGenerator>>,

    /// Approximate nearest neighbor index
    ann_index: Option<storage::ann::AnnIndex>,
}

impl EmbeddingSpace {
    /// Create a new embedding space
    pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let storage: Box<dyn EmbeddingStore> = match &config.storage {
            EmbeddingStorageType::Memory => Box::new(storage::memory::MemoryStore::new()),
            EmbeddingStorageType::Mmap => Box::new(storage::mmap::MmapStore::open(&config)?),
            EmbeddingStorageType::VectorDb { path } => Box::new(
                storage::mmap::MmapStore::open_path(path, config.dimensions)?,
            ),
        };

        let generator: Option<Box<dyn EmbeddingGenerator>> = if config.lazy_generation {
            Some(models::create_generator(&config)?)
        } else {
            None
        };

        let cache = RwLock::new(lru::LruCache::new(
            NonZeroUsize::new(config.cache_size).unwrap(),
        ));

        Ok(Self {
            config,
            storage,
            cache,
            generator,
            ann_index: None,
        })
    }

    /// Create with a specific generator
    pub fn with_generator(
        config: EmbeddingConfig,
        generator: Box<dyn EmbeddingGenerator>,
    ) -> Result<Self, EmbeddingError> {
        let storage: Box<dyn EmbeddingStore> = match &config.storage {
            EmbeddingStorageType::Memory => Box::new(storage::memory::MemoryStore::new()),
            EmbeddingStorageType::Mmap => Box::new(storage::mmap::MmapStore::open(&config)?),
            EmbeddingStorageType::VectorDb { path } => Box::new(
                storage::mmap::MmapStore::open_path(path, config.dimensions)?,
            ),
        };

        let cache = RwLock::new(lru::LruCache::new(
            NonZeroUsize::new(config.cache_size).unwrap(),
        ));

        Ok(Self {
            config,
            storage,
            cache,
            generator: Some(generator),
            ann_index: None,
        })
    }

    /// Get embedding for an IRI
    pub fn get(&self, iri: &IRI) -> Result<Embedding, EmbeddingError> {
        // Check cache first
        {
            if let Ok(cache) = self.cache.read()
                && let Some(emb) = cache.peek(iri)
            {
                return Ok(emb.clone());
            }
        }

        // Check storage
        if let Some(emb) = self.storage.get(iri)? {
            if let Ok(mut cache) = self.cache.write() {
                cache.put(iri.clone(), emb.clone());
            }
            return Ok(emb);
        }

        // Generate on demand
        if let Some(ref generator) = self.generator {
            let emb = generator.generate(iri)?;
            // Note: storage is immutable here, would need interior mutability for caching
            if let Ok(mut cache) = self.cache.write() {
                cache.put(iri.clone(), emb.clone());
            }
            return Ok(emb);
        }

        Err(EmbeddingError::NotFound(iri.clone()))
    }

    /// Try to get embedding, returning None if not found
    pub fn try_get(&self, iri: &IRI) -> Option<Embedding> {
        self.get(iri).ok()
    }

    /// Calculate semantic distance via embeddings
    pub fn embedding_distance(&self, from: &IRI, to: &IRI) -> Result<f64, EmbeddingError> {
        let from_emb = self.get(from)?;
        let to_emb = self.get(to)?;

        Ok(from_emb.to_semantic_distance(&to_emb))
    }

    /// Calculate cosine similarity
    pub fn cosine_similarity(&self, a: &IRI, b: &IRI) -> Result<f32, EmbeddingError> {
        let a_emb = self.get(a)?;
        let b_emb = self.get(b)?;

        Ok(a_emb.cosine_similarity(&b_emb))
    }

    /// Add an embedding to storage
    pub fn add(&mut self, embedding: Embedding) -> Result<(), EmbeddingError> {
        let iri = embedding.iri.clone();
        self.storage.put(&iri, &embedding)?;
        if let Ok(mut cache) = self.cache.write() {
            cache.put(iri, embedding);
        }
        Ok(())
    }

    /// Add multiple embeddings
    pub fn add_batch(&mut self, embeddings: Vec<Embedding>) -> Result<(), EmbeddingError> {
        for emb in embeddings {
            self.add(emb)?;
        }
        Ok(())
    }

    /// Find k nearest neighbors
    pub fn nearest_neighbors(
        &self,
        iri: &IRI,
        k: usize,
    ) -> Result<Vec<(IRI, f64)>, EmbeddingError> {
        let emb = self.get(iri)?;

        if let Some(ref index) = self.ann_index {
            // Use ANN index for efficiency
            index.search(&emb.vector, k)
        } else {
            // Brute force (slow for large datasets)
            self.brute_force_neighbors(&emb, k)
        }
    }

    /// Find nearest neighbors to a raw vector
    pub fn nearest_neighbors_to_vector(
        &self,
        vector: &[f32],
        k: usize,
    ) -> Result<Vec<(IRI, f64)>, EmbeddingError> {
        if let Some(ref index) = self.ann_index {
            index.search(vector, k)
        } else {
            // Create temporary embedding for comparison
            let query =
                Embedding::new(IRI::new("_query_"), vector.to_vec(), EmbeddingModel::Random);
            self.brute_force_neighbors(&query, k)
        }
    }

    /// Build ANN index for fast similarity search
    pub fn build_ann_index(&mut self) -> Result<(), EmbeddingError> {
        let all_embeddings = self.storage.all()?;

        let mut index = storage::ann::AnnIndex::new(self.config.dimensions);

        for emb in &all_embeddings {
            index.add(&emb.iri, &emb.vector)?;
        }

        index.build()?;
        self.ann_index = Some(index);

        Ok(())
    }

    /// Check if ANN index is built
    pub fn has_ann_index(&self) -> bool {
        self.ann_index.is_some()
    }

    fn brute_force_neighbors(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Result<Vec<(IRI, f64)>, EmbeddingError> {
        let all = self.storage.all()?;

        let mut distances: Vec<(IRI, f64)> = all
            .iter()
            .filter(|emb| emb.iri != query.iri)
            .map(|emb| (emb.iri.clone(), query.to_semantic_distance(emb)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        Ok(distances)
    }

    /// Get number of embeddings in storage
    pub fn count(&self) -> Result<usize, EmbeddingError> {
        self.storage.count()
    }

    /// Get configuration
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Perform analogy: a - b + c ~ ?
    /// Returns the vector for "a is to b as c is to ?"
    pub fn analogy(
        &self,
        a: &IRI,
        b: &IRI,
        c: &IRI,
        k: usize,
    ) -> Result<Vec<(IRI, f64)>, EmbeddingError> {
        let a_emb = self.get(a)?;
        let b_emb = self.get(b)?;
        let c_emb = self.get(c)?;

        // result = a - b + c
        let result: Vec<f32> = a_emb
            .vector
            .iter()
            .zip(b_emb.vector.iter())
            .zip(c_emb.vector.iter())
            .map(|((a, b), c)| a - b + c)
            .collect();

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        let result: Vec<f32> = if norm > 1e-10 {
            result.iter().map(|x| x / norm).collect()
        } else {
            result
        };

        self.nearest_neighbors_to_vector(&result, k)
    }
}

/// Statistics about the embedding space
#[derive(Debug, Clone, Default)]
pub struct EmbeddingStats {
    pub total_embeddings: usize,
    pub dimensions: usize,
    pub cache_size: usize,
    pub has_ann_index: bool,
    pub model: Option<EmbeddingModel>,
}

impl EmbeddingSpace {
    pub fn stats(&self) -> EmbeddingStats {
        EmbeddingStats {
            total_embeddings: self.storage.count().unwrap_or(0),
            dimensions: self.config.dimensions,
            cache_size: self.config.cache_size,
            has_ann_index: self.ann_index.is_some(),
            model: Some(self.config.model),
        }
    }
}
