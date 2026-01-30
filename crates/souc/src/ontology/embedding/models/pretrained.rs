//! Pre-trained Embedding Loader
//!
//! Load embeddings from existing sources:
//! - BioConceptVec (PubMed-trained biomedical embeddings)
//! - OPA2Vec (ontology embeddings)
//! - Word2Vec text/binary format
//! - Custom trained embeddings
//!
//! # File Formats Supported
//!
//! 1. Word2Vec text format: "IRI 0.1 0.2 0.3 ..."
//! 2. Word2Vec binary format
//! 3. Our native .dmbe format

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use super::super::{Embedding, EmbeddingError, EmbeddingGenerator, EmbeddingModel};
use crate::ontology::loader::IRI;

/// Pre-trained embedding loader/generator
pub struct PretrainedGenerator {
    embeddings: HashMap<IRI, Vec<f32>>,
    dimensions: usize,
}

impl PretrainedGenerator {
    /// Load from file (auto-detects format based on extension)
    pub fn load(path: &Path) -> Result<Self, EmbeddingError> {
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        match extension {
            "txt" | "vec" => Self::load_word2vec_text(path),
            "bin" => Self::load_word2vec_binary(path),
            "dmbe" => Self::load_native(path),
            _ => Err(EmbeddingError::UnknownFormat(extension.to_string())),
        }
    }

    /// Create empty generator (for testing or incremental loading)
    pub fn empty(dimensions: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dimensions,
        }
    }

    /// Add an embedding manually
    pub fn add(&mut self, iri: IRI, vector: Vec<f32>) {
        if self.dimensions == 0 && !vector.is_empty() {
            self.dimensions = vector.len();
        }
        self.embeddings.insert(iri, vector);
    }

    /// Get number of embeddings
    pub fn count(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if an IRI has an embedding
    pub fn contains(&self, iri: &IRI) -> bool {
        self.embeddings.contains_key(iri)
    }

    /// Load Word2Vec text format
    /// Format:
    /// ```text
    /// vocab_size dimensions
    /// word1 0.1 0.2 0.3 ...
    /// word2 0.4 0.5 0.6 ...
    /// ```
    fn load_word2vec_text(path: &Path) -> Result<Self, EmbeddingError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // First line: vocab_size dimensions
        let header = lines
            .next()
            .ok_or_else(|| EmbeddingError::InvalidFormat("Empty file".into()))??;

        let parts: Vec<&str> = header.split_whitespace().collect();

        let _vocab_size: usize = parts.first().and_then(|s| s.parse().ok()).ok_or_else(|| {
            EmbeddingError::InvalidFormat("Bad header: missing vocab size".into())
        })?;

        let dimensions: usize = parts.get(1).and_then(|s| s.parse().ok()).ok_or_else(|| {
            EmbeddingError::InvalidFormat("Bad header: missing dimensions".into())
        })?;

        let mut embeddings = HashMap::new();

        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() != dimensions + 1 {
                continue; // Skip malformed lines
            }

            let word = parts[0];
            let vector: Vec<f32> = parts[1..].iter().filter_map(|s| s.parse().ok()).collect();

            if vector.len() == dimensions {
                // Convert word to IRI
                let iri = if word.starts_with("http://") || word.starts_with("https://") {
                    IRI::new(word)
                } else {
                    // Assume it's a CURIE or local name
                    IRI::new(&format!("http://embedding.local/{}", word))
                };
                embeddings.insert(iri, vector);
            }
        }

        Ok(Self {
            embeddings,
            dimensions,
        })
    }

    /// Load Word2Vec binary format
    fn load_word2vec_binary(path: &Path) -> Result<Self, EmbeddingError> {
        let mut file = File::open(path)?;
        let mut header = String::new();

        // Read header line (ends with newline)
        loop {
            let mut byte = [0u8; 1];
            file.read_exact(&mut byte)?;
            if byte[0] == b'\n' {
                break;
            }
            header.push(byte[0] as char);
        }

        let parts: Vec<&str> = header.split_whitespace().collect();
        let vocab_size: usize = parts
            .first()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| EmbeddingError::InvalidFormat("Bad vocab size".into()))?;
        let dimensions: usize = parts
            .get(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| EmbeddingError::InvalidFormat("Bad dimensions".into()))?;

        let mut embeddings = HashMap::with_capacity(vocab_size);

        for _ in 0..vocab_size {
            // Read word (space or newline terminated)
            let mut word = String::new();
            loop {
                let mut byte = [0u8; 1];
                if file.read_exact(&mut byte).is_err() {
                    break;
                }
                if byte[0] == b' ' || byte[0] == b'\n' {
                    break;
                }
                word.push(byte[0] as char);
            }

            if word.is_empty() {
                break;
            }

            // Read vector (dimensions * 4 bytes)
            let bytes_to_read = dimensions * 4;
            let mut buffer = vec![0u8; bytes_to_read];

            if file.read_exact(&mut buffer).is_err() {
                break;
            }

            let vector: Vec<f32> = buffer
                .chunks(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            let iri = if word.starts_with("http://") || word.starts_with("https://") {
                IRI::new(&word)
            } else {
                IRI::new(&format!("http://embedding.local/{}", word))
            };

            embeddings.insert(iri, vector);
        }

        Ok(Self {
            embeddings,
            dimensions,
        })
    }

    /// Load native DMBE format
    fn load_native(path: &Path) -> Result<Self, EmbeddingError> {
        use super::super::EmbeddingStore;
        use super::super::storage::mmap::MmapStore;

        // Use MmapStore to load
        let store = MmapStore::open_path(path, 0)?; // 0 = auto-detect dimensions
        let all = store.all()?;

        let dimensions = all.first().map(|e| e.vector.len()).unwrap_or(256);

        let embeddings: HashMap<IRI, Vec<f32>> =
            all.into_iter().map(|e| (e.iri, e.vector)).collect();

        Ok(Self {
            embeddings,
            dimensions,
        })
    }

    /// Save to Word2Vec text format
    pub fn save_word2vec_text(&self, path: &Path) -> Result<(), EmbeddingError> {
        use std::io::Write;

        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "{} {}", self.embeddings.len(), self.dimensions)?;

        // Write embeddings
        for (iri, vector) in &self.embeddings {
            write!(file, "{}", iri.as_str())?;
            for val in vector {
                write!(file, " {}", val)?;
            }
            writeln!(file)?;
        }

        Ok(())
    }
}

impl EmbeddingGenerator for PretrainedGenerator {
    fn generate(&self, iri: &IRI) -> Result<Embedding, EmbeddingError> {
        self.embeddings
            .get(iri)
            .map(|vec| Embedding::new(iri.clone(), vec.clone(), EmbeddingModel::Pretrained))
            .ok_or_else(|| EmbeddingError::NotFound(iri.clone()))
    }

    fn generate_batch(&self, iris: &[IRI]) -> Result<Vec<Embedding>, EmbeddingError> {
        iris.iter().map(|iri| self.generate(iri)).collect()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Load pre-trained embeddings from various sources
pub fn load_pretrained_embeddings(path: &Path) -> Result<Vec<Embedding>, EmbeddingError> {
    let generator = PretrainedGenerator::load(path)?;

    generator
        .embeddings
        .iter()
        .map(|(iri, vec)| {
            Ok(Embedding::new(
                iri.clone(),
                vec.clone(),
                EmbeddingModel::Pretrained,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_empty_generator() {
        let generator = PretrainedGenerator::empty(128);
        assert_eq!(generator.dimensions(), 128);
        assert_eq!(generator.count(), 0);
    }

    #[test]
    fn test_add_embedding() {
        let mut generator = PretrainedGenerator::empty(3);
        let iri = IRI::new("http://example.org/test");

        generator.add(iri.clone(), vec![1.0, 0.0, 0.0]);

        assert!(generator.contains(&iri));
        assert_eq!(generator.count(), 1);

        let emb = generator.generate(&iri).unwrap();
        assert_eq!(emb.dimensions(), 3);
    }

    #[test]
    fn test_load_word2vec_text() {
        let mut file = NamedTempFile::with_suffix(".txt").unwrap();

        writeln!(file, "3 4").unwrap();
        writeln!(file, "http://example.org/term1 1.0 0.0 0.0 0.0").unwrap();
        writeln!(file, "http://example.org/term2 0.0 1.0 0.0 0.0").unwrap();
        writeln!(file, "http://example.org/term3 0.0 0.0 1.0 0.0").unwrap();

        file.flush().unwrap();

        let generator = PretrainedGenerator::load(file.path()).unwrap();

        assert_eq!(generator.dimensions(), 4);
        assert_eq!(generator.count(), 3);

        let emb = generator
            .generate(&IRI::new("http://example.org/term1"))
            .unwrap();
        assert_eq!(emb.model, EmbeddingModel::Pretrained);
    }

    #[test]
    fn test_save_and_load() {
        let mut generator = PretrainedGenerator::empty(4);
        generator.add(IRI::new("http://example.org/a"), vec![1.0, 0.0, 0.0, 0.0]);
        generator.add(IRI::new("http://example.org/b"), vec![0.0, 1.0, 0.0, 0.0]);

        let file = NamedTempFile::with_suffix(".txt").unwrap();
        generator.save_word2vec_text(file.path()).unwrap();

        let loaded = PretrainedGenerator::load(file.path()).unwrap();
        assert_eq!(loaded.count(), 2);
        assert_eq!(loaded.dimensions(), 4);
    }
}
