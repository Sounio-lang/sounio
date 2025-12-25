//! Textual Embeddings
//!
//! Generate embeddings from term labels, definitions, and synonyms.
//!
//! # Approach
//!
//! 1. Concatenate label + definition + synonyms
//! 2. Apply a text embedding model
//! 3. Optionally fine-tune on ontology-specific corpus
//!
//! # Why This Complements Structural
//!
//! "Cardiac" and "heart" may not be directly connected in the hierarchy,
//! but their textual descriptions will be similar:
//! - "Cardiac: relating to the heart"
//! - "Heart: a muscular organ that pumps blood"

use std::collections::HashMap;

use super::super::{Embedding, EmbeddingError, EmbeddingGenerator, EmbeddingModel};
use crate::ontology::loader::{IRI, LoadedTerm};

/// Textual embedding generator
pub struct TextualGenerator {
    dimensions: usize,

    /// Precomputed embeddings
    embeddings: HashMap<IRI, Vec<f32>>,

    /// Term text cache
    texts: HashMap<IRI, String>,
}

impl TextualGenerator {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            embeddings: HashMap::new(),
            texts: HashMap::new(),
        }
    }

    /// Generate embeddings for all terms
    pub fn embed_terms(&mut self, terms: &[LoadedTerm]) -> Result<(), EmbeddingError> {
        for term in terms {
            let text = self.term_to_text(term);
            let vector = self.embed_text(&text);
            self.texts.insert(term.iri.clone(), text);
            self.embeddings.insert(term.iri.clone(), vector);
        }
        Ok(())
    }

    /// Add a single term
    pub fn add_term(&mut self, term: &LoadedTerm) {
        let text = self.term_to_text(term);
        let vector = self.embed_text(&text);
        self.texts.insert(term.iri.clone(), text);
        self.embeddings.insert(term.iri.clone(), vector);
    }

    /// Generate embedding for raw text
    pub fn embed_raw(&self, text: &str) -> Vec<f32> {
        self.embed_text(text)
    }

    fn term_to_text(&self, term: &LoadedTerm) -> String {
        let mut parts = vec![term.label.clone()];

        if let Some(ref def) = term.definition {
            parts.push(def.clone());
        }

        for syn in &term.synonyms {
            parts.push(syn.text.clone());
        }

        parts.join(" . ")
    }

    /// Simple hash-based text embedding
    ///
    /// In production, this would use:
    /// - ONNX runtime with sentence-transformers
    /// - Candle (Rust ML framework)
    /// - External API (OpenAI, Cohere, etc.)
    fn embed_text(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vector = vec![0.0f32; self.dimensions];

        // Tokenize and embed each word
        for (word_idx, word) in text.split_whitespace().enumerate() {
            let normalized = word.to_lowercase();

            let mut hasher = DefaultHasher::new();
            normalized.hash(&mut hasher);
            let hash = hasher.finish();

            // Spread the word's contribution across dimensions
            for j in 0..self.dimensions {
                let idx = (hash.wrapping_add(j as u64) as usize) % self.dimensions;
                let sign = if (hash >> (j % 64)) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                let magnitude = 1.0 / (word_idx as f32 + 1.0).sqrt();

                vector[idx] += sign * magnitude;
            }
        }

        // Normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut vector {
                *x /= norm;
            }
        }

        vector
    }

    /// Get embedding count
    pub fn count(&self) -> usize {
        self.embeddings.len()
    }
}

impl EmbeddingGenerator for TextualGenerator {
    fn generate(&self, iri: &IRI) -> Result<Embedding, EmbeddingError> {
        self.embeddings
            .get(iri)
            .map(|vec| Embedding::new(iri.clone(), vec.clone(), EmbeddingModel::Textual))
            .ok_or_else(|| EmbeddingError::NotFound(iri.clone()))
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
    use crate::ontology::loader::{OntologyId, Synonym, SynonymScope};

    fn make_term_with_text(iri: &str, label: &str, definition: Option<&str>) -> LoadedTerm {
        LoadedTerm {
            iri: IRI::new(iri),
            label: label.to_string(),
            ontology: OntologyId::Unknown,
            superclasses: vec![],
            subclasses: vec![],
            properties: vec![],
            restrictions: vec![],
            xrefs: vec![],
            definition: definition.map(String::from),
            synonyms: vec![],
            hierarchy_depth: 0,
            information_content: 0.0,
            is_obsolete: false,
            replaced_by: None,
        }
    }

    #[test]
    fn test_textual_generator_basic() {
        let mut generator = TextualGenerator::new(128);

        let terms = vec![
            make_term_with_text(
                "http://example.org/Heart",
                "Heart",
                Some("A muscular organ that pumps blood through the circulatory system"),
            ),
            make_term_with_text(
                "http://example.org/Cardiac",
                "Cardiac",
                Some("Relating to or affecting the heart"),
            ),
        ];

        generator.embed_terms(&terms).unwrap();

        let heart = generator
            .generate(&IRI::new("http://example.org/Heart"))
            .unwrap();
        let cardiac = generator
            .generate(&IRI::new("http://example.org/Cardiac"))
            .unwrap();

        // Both should have embeddings
        assert_eq!(heart.dimensions(), 128);
        assert_eq!(cardiac.dimensions(), 128);

        // They should be somewhat similar due to shared word "heart"
        let sim = heart.cosine_similarity(&cardiac);
        println!("Heart-Cardiac similarity: {}", sim);
        assert!(sim > 0.0); // At least some similarity
    }

    #[test]
    fn test_textual_similar_texts() {
        let generator = TextualGenerator::new(128);

        let emb1 = generator.embed_text("pain relief medication");
        let emb2 = generator.embed_text("pain relief drug");
        let emb3 = generator.embed_text("software development");

        let sim_12: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let sim_13: f32 = emb1.iter().zip(emb3.iter()).map(|(a, b)| a * b).sum();

        // "pain relief medication" should be more similar to "pain relief drug"
        // than to "software development"
        println!("Sim(pain-medication, pain-drug): {}", sim_12);
        println!("Sim(pain-medication, software): {}", sim_13);
        assert!(sim_12 > sim_13);
    }

    #[test]
    fn test_textual_with_synonyms() {
        let mut generator = TextualGenerator::new(128);

        let term = LoadedTerm {
            iri: IRI::new("http://example.org/Aspirin"),
            label: "Aspirin".to_string(),
            ontology: OntologyId::Unknown,
            superclasses: vec![],
            subclasses: vec![],
            properties: vec![],
            restrictions: vec![],
            xrefs: vec![],
            definition: Some("A pain reliever and fever reducer".to_string()),
            synonyms: vec![Synonym {
                text: "acetylsalicylic acid".to_string(),
                scope: SynonymScope::Exact,
            }],
            hierarchy_depth: 0,
            information_content: 0.0,
            is_obsolete: false,
            replaced_by: None,
        };

        generator.add_term(&term);

        let emb = generator
            .generate(&IRI::new("http://example.org/Aspirin"))
            .unwrap();
        assert_eq!(emb.dimensions(), 128);
    }
}
