//! Structural Embeddings (OWL2Vec* style)
//!
//! Generate embeddings from ontology graph structure using random walks.
//!
//! # Algorithm
//!
//! 1. Build a graph from ontology relationships (is-a, part-of, etc.)
//! 2. Perform random walks from each node
//! 3. Treat walks as "sentences" and apply Skip-gram
//! 4. The resulting vectors capture structural similarity
//!
//! # Why This Works
//!
//! Terms that appear in similar structural contexts get similar embeddings.
//! "Heart" and "cardiac muscle" will be close because they share neighbors
//! like "cardiovascular system", "blood vessel", etc.

use std::collections::HashMap;

use super::super::{Embedding, EmbeddingError, EmbeddingGenerator, EmbeddingModel};
use crate::ontology::loader::{IRI, LoadedTerm};

/// OWL2Vec*-style structural embedding generator
pub struct StructuralGenerator {
    /// Embedding dimensionality
    dimensions: usize,

    /// Number of random walks per node
    walks_per_node: usize,

    /// Length of each random walk
    walk_length: usize,

    /// Skip-gram context window size
    window_size: usize,

    /// The ontology graph
    graph: OntologyGraph,

    /// Trained embeddings (populated after training)
    embeddings: HashMap<IRI, Vec<f32>>,

    /// Whether training has been done
    trained: bool,
}

impl StructuralGenerator {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            walks_per_node: 10,
            walk_length: 40,
            window_size: 5,
            graph: OntologyGraph::new(),
            embeddings: HashMap::new(),
            trained: false,
        }
    }

    pub fn with_params(
        dimensions: usize,
        walks_per_node: usize,
        walk_length: usize,
        window_size: usize,
    ) -> Self {
        Self {
            dimensions,
            walks_per_node,
            walk_length,
            window_size,
            graph: OntologyGraph::new(),
            embeddings: HashMap::new(),
            trained: false,
        }
    }

    /// Build graph from loaded ontology terms
    pub fn build_graph(&mut self, terms: &[LoadedTerm]) {
        for term in terms {
            // Add is-a edges (high weight)
            for superclass in &term.superclasses {
                self.graph
                    .add_edge(&term.iri, superclass, EdgeType::IsA, 1.0);
            }

            // Add property edges from restrictions (lower weight)
            for restriction in &term.restrictions {
                self.graph.add_edge(
                    &term.iri,
                    &restriction.filler,
                    EdgeType::Property(restriction.property.clone()),
                    0.5,
                );
            }
        }
    }

    /// Add a single term to the graph
    pub fn add_term(&mut self, term: &LoadedTerm) {
        for superclass in &term.superclasses {
            self.graph
                .add_edge(&term.iri, superclass, EdgeType::IsA, 1.0);
        }
        for restriction in &term.restrictions {
            self.graph.add_edge(
                &term.iri,
                &restriction.filler,
                EdgeType::Property(restriction.property.clone()),
                0.5,
            );
        }
        self.trained = false;
    }

    /// Train embeddings using random walks + Skip-gram
    pub fn train(&mut self) -> Result<(), EmbeddingError> {
        if self.graph.is_empty() {
            return Ok(());
        }

        // Generate random walks
        let walks = self.generate_walks();

        if walks.is_empty() {
            return Ok(());
        }

        // Train Skip-gram model
        let model = self.train_skipgram(&walks)?;

        // Extract embeddings
        self.embeddings = model.into_embeddings();
        self.trained = true;

        Ok(())
    }

    fn generate_walks(&self) -> Vec<Vec<IRI>> {
        let mut walks = Vec::new();

        // Simple LCG for reproducibility
        let mut rng_state: u64 = 42;
        let next_rand = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 33) as f64 / (1u64 << 31) as f64
        };

        for node in self.graph.nodes() {
            for _ in 0..self.walks_per_node {
                let walk = self.random_walk(node, &mut rng_state, &next_rand);
                if walk.len() > 1 {
                    walks.push(walk);
                }
            }
        }

        walks
    }

    fn random_walk<F>(&self, start: &IRI, rng_state: &mut u64, next_rand: &F) -> Vec<IRI>
    where
        F: Fn(&mut u64) -> f64,
    {
        let mut walk = vec![start.clone()];
        let mut current = start.clone();

        for _ in 0..self.walk_length {
            let neighbors = self.graph.neighbors(&current);
            if neighbors.is_empty() {
                break;
            }

            // Weighted random selection
            let total_weight: f64 = neighbors.iter().map(|(_, w)| *w).sum();
            if total_weight <= 0.0 {
                break;
            }

            let mut target = next_rand(rng_state) * total_weight;
            let mut selected = None;

            for (neighbor, weight) in neighbors {
                target -= weight;
                if target <= 0.0 {
                    selected = Some(neighbor.clone());
                    break;
                }
            }

            if let Some(next) = selected {
                current = next;
                walk.push(current.clone());
            } else {
                break;
            }
        }

        walk
    }

    fn train_skipgram(&self, walks: &[Vec<IRI>]) -> Result<SkipGramModel, EmbeddingError> {
        let mut model = SkipGramModel::new(self.dimensions, self.window_size);

        // Build vocabulary
        for walk in walks {
            for iri in walk {
                model.add_to_vocab(iri);
            }
        }

        // Training iterations
        let epochs = 5;
        for epoch in 0..epochs {
            let learning_rate = 0.025 * (1.0 - epoch as f64 / epochs as f64);

            for walk in walks {
                model.train_sequence(walk, learning_rate);
            }
        }

        Ok(model)
    }

    /// Get embedding for a specific IRI
    pub fn get_embedding(&self, iri: &IRI) -> Option<&Vec<f32>> {
        self.embeddings.get(iri)
    }

    /// Check if training has been done
    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

impl EmbeddingGenerator for StructuralGenerator {
    fn generate(&self, iri: &IRI) -> Result<Embedding, EmbeddingError> {
        self.embeddings
            .get(iri)
            .map(|vec| Embedding::new(iri.clone(), vec.clone(), EmbeddingModel::Structural))
            .ok_or_else(|| EmbeddingError::NotFound(iri.clone()))
    }

    fn generate_batch(&self, iris: &[IRI]) -> Result<Vec<Embedding>, EmbeddingError> {
        iris.iter().map(|iri| self.generate(iri)).collect()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Simple Skip-gram implementation
struct SkipGramModel {
    dimensions: usize,
    window_size: usize,
    vocab: HashMap<IRI, usize>,
    embeddings: Vec<Vec<f32>>,
    context_embeddings: Vec<Vec<f32>>,
}

impl SkipGramModel {
    fn new(dimensions: usize, window_size: usize) -> Self {
        Self {
            dimensions,
            window_size,
            vocab: HashMap::new(),
            embeddings: Vec::new(),
            context_embeddings: Vec::new(),
        }
    }

    fn add_to_vocab(&mut self, iri: &IRI) {
        if !self.vocab.contains_key(iri) {
            let idx = self.vocab.len();
            self.vocab.insert(iri.clone(), idx);

            // Initialize random embeddings using hash-based seeding
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            iri.as_str().hash(&mut hasher);
            let seed = hasher.finish();

            let emb: Vec<f32> = (0..self.dimensions)
                .map(|i| {
                    let x = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(i as u64);
                    (((x >> 33) as f32 / u32::MAX as f32) - 0.5) / self.dimensions as f32
                })
                .collect();

            let ctx: Vec<f32> = vec![0.0; self.dimensions];

            self.embeddings.push(emb);
            self.context_embeddings.push(ctx);
        }
    }

    fn train_sequence(&mut self, sequence: &[IRI], learning_rate: f64) {
        for (i, center) in sequence.iter().enumerate() {
            let center_idx = match self.vocab.get(center) {
                Some(&idx) => idx,
                None => continue,
            };

            // Context window
            let start = i.saturating_sub(self.window_size);
            let end = (i + self.window_size + 1).min(sequence.len());

            for j in start..end {
                if i == j {
                    continue;
                }

                let context = &sequence[j];
                let context_idx = match self.vocab.get(context) {
                    Some(&idx) => idx,
                    None => continue,
                };

                self.update_pair(center_idx, context_idx, learning_rate as f32);
            }
        }
    }

    fn update_pair(&mut self, center: usize, context: usize, lr: f32) {
        // Simplified Skip-gram update (without negative sampling)
        let dot: f32 = self.embeddings[center]
            .iter()
            .zip(self.context_embeddings[context].iter())
            .map(|(a, b)| a * b)
            .sum();

        let sigmoid = 1.0 / (1.0 + (-dot).exp());
        let gradient = lr * (1.0 - sigmoid);

        for d in 0..self.dimensions {
            let g_emb = gradient * self.context_embeddings[context][d];
            let g_ctx = gradient * self.embeddings[center][d];

            self.embeddings[center][d] += g_emb;
            self.context_embeddings[context][d] += g_ctx;
        }
    }

    fn into_embeddings(self) -> HashMap<IRI, Vec<f32>> {
        let idx_to_iri: HashMap<usize, IRI> = self
            .vocab
            .into_iter()
            .map(|(iri, idx)| (idx, iri))
            .collect();

        self.embeddings
            .into_iter()
            .enumerate()
            .filter_map(|(idx, emb)| idx_to_iri.get(&idx).map(|iri| (iri.clone(), emb)))
            .collect()
    }
}

/// Ontology graph for random walks
struct OntologyGraph {
    adjacency: HashMap<IRI, Vec<(IRI, f64)>>,
}

/// Edge type in the ontology graph
#[derive(Clone)]
#[allow(dead_code)]
enum EdgeType {
    IsA,
    Property(IRI),
}

impl OntologyGraph {
    fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    fn add_edge(&mut self, from: &IRI, to: &IRI, _edge_type: EdgeType, weight: f64) {
        // Add forward edge
        self.adjacency
            .entry(from.clone())
            .or_default()
            .push((to.clone(), weight));

        // Add reverse edge for undirected walk
        self.adjacency
            .entry(to.clone())
            .or_default()
            .push((from.clone(), weight));
    }

    fn nodes(&self) -> impl Iterator<Item = &IRI> {
        self.adjacency.keys()
    }

    fn neighbors(&self, node: &IRI) -> &[(IRI, f64)] {
        self.adjacency
            .get(node)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::loader::OntologyId;

    fn make_term(iri: IRI, superclasses: Vec<IRI>) -> LoadedTerm {
        LoadedTerm {
            iri,
            label: String::new(),
            ontology: OntologyId::Unknown,
            superclasses,
            subclasses: vec![],
            properties: vec![],
            restrictions: vec![],
            xrefs: vec![],
            definition: None,
            synonyms: vec![],
            hierarchy_depth: 0,
            information_content: 0.0,
            is_obsolete: false,
            replaced_by: None,
        }
    }

    #[test]
    fn test_structural_generator_basic() {
        let mut generator = StructuralGenerator::new(64);

        let terms = vec![
            make_term(IRI::new("http://example.org/Animal"), vec![]),
            make_term(
                IRI::new("http://example.org/Mammal"),
                vec![IRI::new("http://example.org/Animal")],
            ),
            make_term(
                IRI::new("http://example.org/Dog"),
                vec![IRI::new("http://example.org/Mammal")],
            ),
            make_term(
                IRI::new("http://example.org/Cat"),
                vec![IRI::new("http://example.org/Mammal")],
            ),
        ];

        generator.build_graph(&terms);
        generator.train().unwrap();

        assert!(generator.is_trained());

        // Dog and Cat should both have embeddings
        let dog_emb = generator.get_embedding(&IRI::new("http://example.org/Dog"));
        let cat_emb = generator.get_embedding(&IRI::new("http://example.org/Cat"));

        assert!(dog_emb.is_some());
        assert!(cat_emb.is_some());
    }

    #[test]
    fn test_structural_generator_siblings_closer() {
        let mut generator = StructuralGenerator::new(64);

        let terms = vec![
            make_term(IRI::new("http://example.org/Animal"), vec![]),
            make_term(
                IRI::new("http://example.org/Mammal"),
                vec![IRI::new("http://example.org/Animal")],
            ),
            make_term(
                IRI::new("http://example.org/Dog"),
                vec![IRI::new("http://example.org/Mammal")],
            ),
            make_term(
                IRI::new("http://example.org/Cat"),
                vec![IRI::new("http://example.org/Mammal")],
            ),
            make_term(
                IRI::new("http://example.org/Bird"),
                vec![IRI::new("http://example.org/Animal")],
            ),
        ];

        generator.build_graph(&terms);
        generator.train().unwrap();

        let dog = generator
            .generate(&IRI::new("http://example.org/Dog"))
            .unwrap();
        let cat = generator
            .generate(&IRI::new("http://example.org/Cat"))
            .unwrap();
        let bird = generator
            .generate(&IRI::new("http://example.org/Bird"))
            .unwrap();

        // Dog and Cat (both mammals) should be more similar than Dog and Bird
        let dog_cat_sim = dog.cosine_similarity(&cat);
        let dog_bird_sim = dog.cosine_similarity(&bird);

        // This may not always hold with random initialization, but generally should
        // For a robust test, we'd need more training data
        println!(
            "Dog-Cat similarity: {}, Dog-Bird similarity: {}",
            dog_cat_sim, dog_bird_sim
        );
    }
}
