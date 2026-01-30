//! Embedding Model Implementations
//!
//! Multiple approaches to generating ontological embeddings:
//! - Structural: Graph-based random walks (OWL2Vec* style)
//! - Textual: From labels, definitions, synonyms
//! - Hybrid: Fusion of structural and textual
//! - Pretrained: Load from existing embedding files

pub mod hybrid;
pub mod pretrained;
pub mod structural;
pub mod textual;

use super::{Embedding, EmbeddingConfig, EmbeddingError, EmbeddingGenerator, EmbeddingModel};
use crate::ontology::loader::IRI;

/// Create a generator based on model type
pub fn create_generator(
    config: &EmbeddingConfig,
) -> Result<Box<dyn EmbeddingGenerator>, EmbeddingError> {
    match config.model {
        EmbeddingModel::Structural => Ok(Box::new(structural::StructuralGenerator::new(
            config.dimensions,
        ))),
        EmbeddingModel::Textual => Ok(Box::new(textual::TextualGenerator::new(config.dimensions))),
        EmbeddingModel::Hybrid => Ok(Box::new(hybrid::HybridGenerator::new(config)?)),
        EmbeddingModel::Pretrained => {
            let path = config
                .pretrained_path
                .as_ref()
                .ok_or(EmbeddingError::NoPretrained)?;
            Ok(Box::new(pretrained::PretrainedGenerator::load(path)?))
        }
        EmbeddingModel::Random => Ok(Box::new(RandomGenerator::new(config.dimensions))),
    }
}

/// Random embedding generator (for testing)
pub struct RandomGenerator {
    dimensions: usize,
}

impl RandomGenerator {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl EmbeddingGenerator for RandomGenerator {
    fn generate(&self, iri: &IRI) -> Result<Embedding, EmbeddingError> {
        // Use a deterministic hash-based approach for reproducibility
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        iri.as_str().hash(&mut hasher);
        let seed = hasher.finish();

        let vector: Vec<f32> = (0..self.dimensions)
            .map(|i| {
                // Simple LCG-style pseudorandom
                let x = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(i as u64);
                ((x >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        Ok(Embedding::new(iri.clone(), vector, EmbeddingModel::Random))
    }

    fn generate_batch(&self, iris: &[IRI]) -> Result<Vec<Embedding>, EmbeddingError> {
        iris.iter().map(|iri| self.generate(iri)).collect()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_generator_deterministic() {
        let generator = RandomGenerator::new(128);
        let iri = IRI::new("http://example.org/test");

        let emb1 = generator.generate(&iri).unwrap();
        let emb2 = generator.generate(&iri).unwrap();

        // Same IRI should produce same embedding
        assert_eq!(emb1.vector, emb2.vector);
    }

    #[test]
    fn test_random_generator_different_iris() {
        let generator = RandomGenerator::new(128);
        let iri1 = IRI::new("http://example.org/term1");
        let iri2 = IRI::new("http://example.org/term2");

        let emb1 = generator.generate(&iri1).unwrap();
        let emb2 = generator.generate(&iri2).unwrap();

        // Different IRIs should produce different embeddings
        assert_ne!(emb1.vector, emb2.vector);
    }
}
