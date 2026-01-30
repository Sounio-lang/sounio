//! Hybrid Embeddings
//!
//! Combines structural and textual embeddings for richer representation.
//!
//! # Fusion Strategies
//!
//! 1. **Concatenation**: [structural; textual] -> 2x dimensions, then project
//! 2. **Averaging**: (structural + textual) / 2
//! 3. **Weighted**: alpha * structural + (1-alpha) * textual
//! 4. **Learned**: MLP fusion (requires training data)

use super::super::{
    Embedding, EmbeddingConfig, EmbeddingError, EmbeddingGenerator, EmbeddingModel,
};
use super::structural::StructuralGenerator;
use super::textual::TextualGenerator;
use crate::ontology::loader::{IRI, LoadedTerm};

/// Hybrid embedding generator combining structural and textual approaches
pub struct HybridGenerator {
    structural: StructuralGenerator,
    textual: TextualGenerator,
    fusion: FusionStrategy,
    dimensions: usize,
    initialized: bool,
}

/// Strategy for combining structural and textual embeddings
#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    /// Concatenate and project back to original dimensions
    Concatenate,

    /// Simple average
    Average,

    /// Weighted combination
    Weighted { alpha: f32 },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        FusionStrategy::Weighted { alpha: 0.5 }
    }
}

impl HybridGenerator {
    pub fn new(config: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
        Ok(Self {
            structural: StructuralGenerator::new(config.dimensions),
            textual: TextualGenerator::new(config.dimensions),
            fusion: FusionStrategy::default(),
            dimensions: config.dimensions,
            initialized: false,
        })
    }

    pub fn with_fusion(dimensions: usize, fusion: FusionStrategy) -> Self {
        Self {
            structural: StructuralGenerator::new(dimensions),
            textual: TextualGenerator::new(dimensions),
            fusion,
            dimensions,
            initialized: false,
        }
    }

    /// Set fusion strategy
    pub fn set_fusion(&mut self, fusion: FusionStrategy) {
        self.fusion = fusion;
    }

    /// Initialize from ontology terms
    pub fn initialize(&mut self, terms: &[LoadedTerm]) -> Result<(), EmbeddingError> {
        // Build structural graph and train
        self.structural.build_graph(terms);
        self.structural.train()?;

        // Generate textual embeddings
        self.textual.embed_terms(terms)?;

        self.initialized = true;
        Ok(())
    }

    /// Add a single term (requires re-training structural for best results)
    pub fn add_term(&mut self, term: &LoadedTerm) {
        self.structural.add_term(term);
        self.textual.add_term(term);
    }

    /// Retrain structural embeddings (call after adding terms)
    pub fn retrain(&mut self) -> Result<(), EmbeddingError> {
        self.structural.train()?;
        self.initialized = true;
        Ok(())
    }

    fn fuse(&self, structural: &[f32], textual: &[f32]) -> Vec<f32> {
        match self.fusion {
            FusionStrategy::Concatenate => {
                // Concatenate and project (take interleaved elements)
                let mut result = Vec::with_capacity(self.dimensions);
                let half = self.dimensions / 2;

                for i in 0..half {
                    if i < structural.len() {
                        result.push(structural[i]);
                    } else {
                        result.push(0.0);
                    }
                }

                for i in 0..(self.dimensions - half) {
                    if i < textual.len() {
                        result.push(textual[i]);
                    } else {
                        result.push(0.0);
                    }
                }

                // Normalize
                let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in &mut result {
                        *x /= norm;
                    }
                }

                result
            }

            FusionStrategy::Average => {
                let mut result: Vec<f32> = structural
                    .iter()
                    .zip(textual.iter())
                    .map(|(s, t)| (s + t) / 2.0)
                    .collect();

                // Normalize
                let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in &mut result {
                        *x /= norm;
                    }
                }

                result
            }

            FusionStrategy::Weighted { alpha } => {
                let mut result: Vec<f32> = structural
                    .iter()
                    .zip(textual.iter())
                    .map(|(s, t)| alpha * s + (1.0 - alpha) * t)
                    .collect();

                // Normalize
                let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in &mut result {
                        *x /= norm;
                    }
                }

                result
            }
        }
    }

    /// Generate hybrid embedding (can fallback to single source if one is missing)
    fn generate_internal(&self, iri: &IRI) -> Result<Embedding, EmbeddingError> {
        let structural_result = self.structural.generate(iri);
        let textual_result = self.textual.generate(iri);

        match (structural_result, textual_result) {
            (Ok(s), Ok(t)) => {
                // Both available - fuse them
                let fused = self.fuse(&s.vector, &t.vector);
                let confidence = s.confidence.min(t.confidence);
                Ok(Embedding::with_confidence(
                    iri.clone(),
                    fused,
                    EmbeddingModel::Hybrid,
                    confidence,
                ))
            }
            (Ok(s), Err(_)) => {
                // Only structural available
                Ok(Embedding::with_confidence(
                    iri.clone(),
                    s.vector,
                    EmbeddingModel::Structural,
                    s.confidence * 0.8, // Reduced confidence
                ))
            }
            (Err(_), Ok(t)) => {
                // Only textual available
                Ok(Embedding::with_confidence(
                    iri.clone(),
                    t.vector,
                    EmbeddingModel::Textual,
                    t.confidence * 0.8, // Reduced confidence
                ))
            }
            (Err(e), Err(_)) => Err(e),
        }
    }

    /// Check if generator has been initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the fusion strategy
    pub fn fusion_strategy(&self) -> FusionStrategy {
        self.fusion
    }
}

impl EmbeddingGenerator for HybridGenerator {
    fn generate(&self, iri: &IRI) -> Result<Embedding, EmbeddingError> {
        self.generate_internal(iri)
    }

    fn generate_batch(&self, iris: &[IRI]) -> Result<Vec<Embedding>, EmbeddingError> {
        iris.iter().map(|iri| self.generate(iri)).collect()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Builder for hybrid generator with custom configuration
pub struct HybridGeneratorBuilder {
    dimensions: usize,
    fusion: FusionStrategy,
    structural_walks_per_node: usize,
    structural_walk_length: usize,
}

impl HybridGeneratorBuilder {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            fusion: FusionStrategy::default(),
            structural_walks_per_node: 10,
            structural_walk_length: 40,
        }
    }

    pub fn fusion(mut self, fusion: FusionStrategy) -> Self {
        self.fusion = fusion;
        self
    }

    pub fn structural_walks(mut self, walks_per_node: usize, walk_length: usize) -> Self {
        self.structural_walks_per_node = walks_per_node;
        self.structural_walk_length = walk_length;
        self
    }

    pub fn build(self) -> HybridGenerator {
        HybridGenerator {
            structural: StructuralGenerator::with_params(
                self.dimensions,
                self.structural_walks_per_node,
                self.structural_walk_length,
                5, // window_size
            ),
            textual: TextualGenerator::new(self.dimensions),
            fusion: self.fusion,
            dimensions: self.dimensions,
            initialized: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::loader::OntologyId;

    fn make_term(iri: &str, label: &str, superclasses: Vec<&str>) -> LoadedTerm {
        LoadedTerm {
            iri: IRI::new(iri),
            label: label.to_string(),
            ontology: OntologyId::Unknown,
            superclasses: superclasses.into_iter().map(|s| IRI::new(s)).collect(),
            subclasses: vec![],
            properties: vec![],
            restrictions: vec![],
            xrefs: vec![],
            definition: Some(format!("Definition of {}", label)),
            synonyms: vec![],
            hierarchy_depth: 0,
            information_content: 0.0,
            is_obsolete: false,
            replaced_by: None,
        }
    }

    #[test]
    fn test_hybrid_generator() {
        let config = EmbeddingConfig {
            dimensions: 64,
            ..Default::default()
        };

        let mut generator = HybridGenerator::new(&config).unwrap();

        let terms = vec![
            make_term("http://example.org/Animal", "Animal", vec![]),
            make_term(
                "http://example.org/Mammal",
                "Mammal",
                vec!["http://example.org/Animal"],
            ),
            make_term(
                "http://example.org/Dog",
                "Dog",
                vec!["http://example.org/Mammal"],
            ),
        ];

        generator.initialize(&terms).unwrap();

        let dog = generator
            .generate(&IRI::new("http://example.org/Dog"))
            .unwrap();
        assert_eq!(dog.dimensions(), 64);
        assert_eq!(dog.model, EmbeddingModel::Hybrid);
    }

    #[test]
    fn test_fusion_strategies() {
        let structural = vec![1.0, 0.0, 0.0, 0.0];
        let textual = vec![0.0, 1.0, 0.0, 0.0];

        // Weighted 50/50
        let hybrid = HybridGenerator::with_fusion(4, FusionStrategy::Weighted { alpha: 0.5 });
        let fused = hybrid.fuse(&structural, &textual);

        // Should be normalized [0.5, 0.5, 0, 0] -> [0.707..., 0.707..., 0, 0]
        assert!((fused[0] - fused[1]).abs() < 0.01);
        assert!(fused[0] > 0.5);

        // Average
        let hybrid2 = HybridGenerator::with_fusion(4, FusionStrategy::Average);
        let fused2 = hybrid2.fuse(&structural, &textual);
        assert!((fused2[0] - fused2[1]).abs() < 0.01);
    }

    #[test]
    fn test_builder() {
        let hybrid = HybridGeneratorBuilder::new(128)
            .fusion(FusionStrategy::Weighted { alpha: 0.7 })
            .structural_walks(20, 50)
            .build();

        assert_eq!(hybrid.dimensions(), 128);
        assert!(!hybrid.is_initialized());
    }
}
