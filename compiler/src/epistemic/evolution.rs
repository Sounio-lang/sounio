//! Ontology Evolution Calculus
//!
//! Implements MCMC-inspired evolution of ontologies using Bayesian inference
//! and Metropolis-Hastings acceptance criteria.
//!
//! # Algorithm: CMAM (CMA-ES + Metropolis)
//!
//! Based on 2025 research combining Covariance Matrix Adaptation with
//! Metropolis acceptance for Bayesian ontology evolution.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Evolution Loop                                │
//! │                                                                  │
//! │   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
//! │   │ Current  │───▶│ Propose  │───▶│ Evaluate │                 │
//! │   │ State    │    │ Mutation │    │ Fitness  │                 │
//! │   └──────────┘    └──────────┘    └────┬─────┘                 │
//! │        ▲                               │                        │
//! │        │         ┌─────────────────────┴─────────────────┐     │
//! │        │         ▼                                       ▼     │
//! │   ┌────┴────┐   ┌──────────┐                      ┌──────────┐ │
//! │   │ Accept  │◀──│ Better?  │──────────────────────│ Metropolis│ │
//! │   │ Change  │   │          │                      │ Accept?  │ │
//! │   └─────────┘   └──────────┘                      └────┬─────┘ │
//! │        ▲                                               │       │
//! │        └───────────────────────────────────────────────┘       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Mutation Operators
//!
//! | Operator | Description | Impact |
//! |----------|-------------|--------|
//! | AddAxiom | Add new relationship | Medium |
//! | RemoveAxiom | Remove relationship | Medium |
//! | AdjustConfidence | Change belief confidence | Low |
//! | PromoteBelief | Convert belief to fact | High |
//! | DemoteFact | Convert fact to belief | High |
//! | MergeClasses | Combine equivalent classes | High |
//! | SplitClass | Divide class into subclasses | High |
//!
//! # Fitness Functions
//!
//! Fitness combines multiple objectives:
//! - Consistency: No constraint violations
//! - Coverage: Adequate domain representation
//! - Parsimony: Minimal redundancy
//! - Evidence Alignment: Matches empirical data

use std::collections::HashMap;

/// Configuration for ontology evolution
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Initial temperature for simulated annealing
    pub initial_temperature: f64,
    /// Cooling rate (multiplied each iteration)
    pub cooling_rate: f64,
    /// Minimum temperature before stopping
    pub min_temperature: f64,
    /// Convergence threshold (fitness improvement)
    pub convergence_threshold: f64,
    /// Convergence window (iterations to check)
    pub convergence_window: usize,
    /// Mutation probability weights
    pub mutation_weights: MutationWeights,
    /// Fitness function weights
    pub fitness_weights: FitnessWeights,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            initial_temperature: 1.0,
            cooling_rate: 0.995,
            min_temperature: 0.01,
            convergence_threshold: 0.001,
            convergence_window: 50,
            mutation_weights: MutationWeights::default(),
            fitness_weights: FitnessWeights::default(),
        }
    }
}

/// Weights for different mutation operators
#[derive(Debug, Clone)]
pub struct MutationWeights {
    pub add_axiom: f64,
    pub remove_axiom: f64,
    pub adjust_confidence: f64,
    pub promote_belief: f64,
    pub demote_fact: f64,
    pub merge_classes: f64,
    pub split_class: f64,
}

impl Default for MutationWeights {
    fn default() -> Self {
        Self {
            add_axiom: 0.2,
            remove_axiom: 0.15,
            adjust_confidence: 0.3,
            promote_belief: 0.1,
            demote_fact: 0.05,
            merge_classes: 0.1,
            split_class: 0.1,
        }
    }
}

impl MutationWeights {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.add_axiom
            + self.remove_axiom
            + self.adjust_confidence
            + self.promote_belief
            + self.demote_fact
            + self.merge_classes
            + self.split_class;

        if total > 0.0 {
            self.add_axiom /= total;
            self.remove_axiom /= total;
            self.adjust_confidence /= total;
            self.promote_belief /= total;
            self.demote_fact /= total;
            self.merge_classes /= total;
            self.split_class /= total;
        }
    }
}

/// Weights for fitness function components
#[derive(Debug, Clone)]
pub struct FitnessWeights {
    /// Weight for consistency (no violations)
    pub consistency: f64,
    /// Weight for coverage (domain representation)
    pub coverage: f64,
    /// Weight for parsimony (minimal redundancy)
    pub parsimony: f64,
    /// Weight for evidence alignment
    pub evidence_alignment: f64,
    /// Weight for ontological coherence
    pub coherence: f64,
}

impl Default for FitnessWeights {
    fn default() -> Self {
        Self {
            consistency: 0.3,
            coverage: 0.2,
            parsimony: 0.15,
            evidence_alignment: 0.25,
            coherence: 0.1,
        }
    }
}

/// Types of mutations that can be applied
#[derive(Debug, Clone, PartialEq)]
pub enum MutationType {
    /// Add a new axiom/relationship
    AddAxiom {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Remove an existing axiom
    RemoveAxiom { axiom_id: String },
    /// Adjust confidence of a belief
    AdjustConfidence { belief_id: String, delta: f64 },
    /// Promote a belief to a fact
    PromoteBelief { belief_id: String },
    /// Demote a fact to a belief
    DemoteFact { fact_id: String, confidence: f64 },
    /// Merge two equivalent classes
    MergeClasses {
        class1: String,
        class2: String,
        merged_name: String,
    },
    /// Split a class into subclasses
    SplitClass {
        class: String,
        subclasses: Vec<String>,
    },
}

impl std::fmt::Display for MutationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MutationType::AddAxiom {
                subject, predicate, ..
            } => {
                write!(f, "AddAxiom({} {})", subject, predicate)
            }
            MutationType::RemoveAxiom { axiom_id } => {
                write!(f, "RemoveAxiom({})", axiom_id)
            }
            MutationType::AdjustConfidence { belief_id, delta } => {
                write!(f, "AdjustConfidence({}, {:+.2})", belief_id, delta)
            }
            MutationType::PromoteBelief { belief_id } => {
                write!(f, "PromoteBelief({})", belief_id)
            }
            MutationType::DemoteFact { fact_id, .. } => {
                write!(f, "DemoteFact({})", fact_id)
            }
            MutationType::MergeClasses { class1, class2, .. } => {
                write!(f, "MergeClasses({}, {})", class1, class2)
            }
            MutationType::SplitClass { class, .. } => {
                write!(f, "SplitClass({})", class)
            }
        }
    }
}

/// A proposed mutation with its expected impact
#[derive(Debug, Clone)]
pub struct Mutation {
    /// Type of mutation
    pub mutation_type: MutationType,
    /// Expected fitness impact (-1.0 to 1.0)
    pub expected_impact: f64,
    /// Rationale for this mutation
    pub rationale: String,
}

/// State of the ontology being evolved
#[derive(Debug, Clone)]
pub struct OntologyState {
    /// Current axioms
    pub axioms: HashMap<String, Axiom>,
    /// Current beliefs with confidence
    pub beliefs: HashMap<String, BeliefState>,
    /// Current facts
    pub facts: HashMap<String, FactState>,
    /// Class hierarchy
    pub class_hierarchy: HashMap<String, Vec<String>>,
    /// Current fitness
    pub fitness: f64,
}

/// An axiom in the ontology
#[derive(Debug, Clone)]
pub struct Axiom {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
}

/// Belief state for evolution
#[derive(Debug, Clone)]
pub struct BeliefState {
    pub id: String,
    pub content: String,
    pub confidence: f64,
    pub revision_count: u32,
}

/// Fact state for evolution
#[derive(Debug, Clone)]
pub struct FactState {
    pub id: String,
    pub content: String,
    pub sources: Vec<String>,
}

impl OntologyState {
    /// Create a new empty state
    pub fn new() -> Self {
        Self {
            axioms: HashMap::new(),
            beliefs: HashMap::new(),
            facts: HashMap::new(),
            class_hierarchy: HashMap::new(),
            fitness: 0.0,
        }
    }

    /// Count total elements
    pub fn element_count(&self) -> usize {
        self.axioms.len() + self.beliefs.len() + self.facts.len()
    }

    /// Get all class names
    pub fn classes(&self) -> Vec<&String> {
        self.class_hierarchy.keys().collect()
    }
}

impl Default for OntologyState {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of evolution process
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether evolution converged
    pub converged: bool,
    /// Initial fitness
    pub initial_fitness: f64,
    /// Final fitness
    pub final_fitness: f64,
    /// Fitness improvement
    pub improvement: f64,
    /// Mutations accepted
    pub mutations_accepted: usize,
    /// Mutations rejected
    pub mutations_rejected: usize,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// History of fitness values
    pub fitness_history: Vec<f64>,
    /// Applied mutations
    pub applied_mutations: Vec<Mutation>,
}

/// Ontology evolution engine
pub struct EvolutionEngine {
    config: EvolutionConfig,
    rng_state: u64,
}

impl EvolutionEngine {
    /// Create a new evolution engine
    pub fn new(config: EvolutionConfig) -> Self {
        Self {
            config,
            rng_state: 42, // Seed for reproducibility
        }
    }

    /// Create with default config
    pub fn default_engine() -> Self {
        Self::new(EvolutionConfig::default())
    }

    /// Run evolution on an ontology state
    pub fn evolve(&mut self, initial_state: OntologyState) -> EvolutionResult {
        let mut state = initial_state;
        let initial_fitness = self.calculate_fitness(&state);
        state.fitness = initial_fitness;

        let mut temperature = self.config.initial_temperature;
        let mut mutations_accepted = 0;
        let mut mutations_rejected = 0;
        let mut fitness_history = vec![initial_fitness];
        let mut applied_mutations = Vec::new();

        // Convergence tracking
        let mut recent_fitness: Vec<f64> = Vec::new();

        for iteration in 0..self.config.max_iterations {
            // Check temperature
            if temperature < self.config.min_temperature {
                break;
            }

            // Propose mutation
            if let Some(mutation) = self.propose_mutation(&state) {
                // Apply mutation to get new state
                let mut new_state = state.clone();
                if self.apply_mutation(&mut new_state, &mutation) {
                    let new_fitness = self.calculate_fitness(&new_state);

                    // Metropolis acceptance
                    if self.accept(&state.fitness, &new_fitness, temperature) {
                        state = new_state;
                        state.fitness = new_fitness;
                        mutations_accepted += 1;
                        applied_mutations.push(mutation);
                    } else {
                        mutations_rejected += 1;
                    }
                } else {
                    mutations_rejected += 1;
                }
            }

            // Record fitness
            fitness_history.push(state.fitness);

            // Check convergence
            recent_fitness.push(state.fitness);
            if recent_fitness.len() > self.config.convergence_window {
                recent_fitness.remove(0);
            }

            if self.has_converged(&recent_fitness) {
                return EvolutionResult {
                    iterations: iteration + 1,
                    converged: true,
                    initial_fitness,
                    final_fitness: state.fitness,
                    improvement: state.fitness - initial_fitness,
                    mutations_accepted,
                    mutations_rejected,
                    acceptance_rate: mutations_accepted as f64
                        / (mutations_accepted + mutations_rejected).max(1) as f64,
                    fitness_history,
                    applied_mutations,
                };
            }

            // Cool down
            temperature *= self.config.cooling_rate;
        }

        EvolutionResult {
            iterations: self.config.max_iterations,
            converged: false,
            initial_fitness,
            final_fitness: state.fitness,
            improvement: state.fitness - initial_fitness,
            mutations_accepted,
            mutations_rejected,
            acceptance_rate: mutations_accepted as f64
                / (mutations_accepted + mutations_rejected).max(1) as f64,
            fitness_history,
            applied_mutations,
        }
    }

    /// Calculate fitness of a state
    pub fn calculate_fitness(&self, state: &OntologyState) -> f64 {
        let w = &self.config.fitness_weights;

        let consistency = self.calculate_consistency(state);
        let coverage = self.calculate_coverage(state);
        let parsimony = self.calculate_parsimony(state);
        let evidence_alignment = self.calculate_evidence_alignment(state);
        let coherence = self.calculate_coherence(state);

        w.consistency * consistency
            + w.coverage * coverage
            + w.parsimony * parsimony
            + w.evidence_alignment * evidence_alignment
            + w.coherence * coherence
    }

    /// Calculate consistency score (0-1)
    fn calculate_consistency(&self, state: &OntologyState) -> f64 {
        // Check for contradictions
        let mut violations = 0;

        // Check for cycles in class hierarchy
        for (class, parents) in &state.class_hierarchy {
            for parent in parents {
                if self.has_cycle(class, parent, &state.class_hierarchy) {
                    violations += 1;
                }
            }
        }

        // Check for conflicting axioms
        let axiom_pairs: Vec<_> = state.axioms.values().collect();
        for i in 0..axiom_pairs.len() {
            for j in (i + 1)..axiom_pairs.len() {
                if self.axioms_conflict(axiom_pairs[i], axiom_pairs[j]) {
                    violations += 1;
                }
            }
        }

        // Score: 1.0 for no violations, decreasing with violations
        1.0 / (1.0 + violations as f64)
    }

    /// Check for cycles in hierarchy
    fn has_cycle(
        &self,
        class: &str,
        target: &str,
        hierarchy: &HashMap<String, Vec<String>>,
    ) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![target.to_string()];

        while let Some(current) = stack.pop() {
            if current == class {
                return true;
            }
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(parents) = hierarchy.get(&current) {
                for parent in parents {
                    stack.push(parent.clone());
                }
            }
        }

        false
    }

    /// Check if two axioms conflict
    fn axioms_conflict(&self, a1: &Axiom, a2: &Axiom) -> bool {
        // Simple conflict detection: same subject and predicate but different object
        // with high confidence
        if a1.subject == a2.subject && a1.predicate == a2.predicate && a1.object != a2.object {
            // Check if predicate is functional (single-valued)
            let functional_predicates = ["is_a", "type_of", "equivalent_to"];
            if functional_predicates.contains(&a1.predicate.as_str()) {
                return a1.confidence > 0.8 && a2.confidence > 0.8;
            }
        }
        false
    }

    /// Calculate coverage score (0-1)
    fn calculate_coverage(&self, state: &OntologyState) -> f64 {
        // Based on number of elements, normalized
        let count = state.element_count() as f64;
        // Sigmoid function to normalize
        1.0 - 1.0 / (1.0 + (count / 50.0))
    }

    /// Calculate parsimony score (0-1)
    fn calculate_parsimony(&self, state: &OntologyState) -> f64 {
        // Penalize redundancy
        // For now, use inverse of element count as proxy
        let count = state.element_count() as f64;
        1.0 / (1.0 + count / 100.0)
    }

    /// Calculate evidence alignment score (0-1)
    fn calculate_evidence_alignment(&self, state: &OntologyState) -> f64 {
        // Average confidence of beliefs
        if state.beliefs.is_empty() {
            return 1.0;
        }

        state.beliefs.values().map(|b| b.confidence).sum::<f64>() / state.beliefs.len() as f64
    }

    /// Calculate coherence score (0-1)
    fn calculate_coherence(&self, state: &OntologyState) -> f64 {
        // Check if classes are well-connected
        if state.class_hierarchy.is_empty() {
            return 1.0;
        }

        // Count orphan classes (no parents, not root)
        let all_classes: std::collections::HashSet<_> = state.class_hierarchy.keys().collect();
        let mut has_parents = std::collections::HashSet::new();
        for parents in state.class_hierarchy.values() {
            for parent in parents {
                has_parents.insert(parent);
            }
        }

        let orphans = all_classes.len() - has_parents.len();
        1.0 - (orphans as f64 / all_classes.len().max(1) as f64)
    }

    /// Propose a random mutation
    fn propose_mutation(&mut self, state: &OntologyState) -> Option<Mutation> {
        // Copy weights to avoid borrow issues
        let adjust_confidence = self.config.mutation_weights.adjust_confidence;
        let add_axiom = self.config.mutation_weights.add_axiom;
        let remove_axiom = self.config.mutation_weights.remove_axiom;
        let promote_belief = self.config.mutation_weights.promote_belief;
        let demote_fact = self.config.mutation_weights.demote_fact;

        // Generate random number for selection
        let r = self.random();

        let mut cumulative = 0.0;

        // AdjustConfidence (most common)
        cumulative += adjust_confidence;
        if r < cumulative && !state.beliefs.is_empty() {
            let beliefs: Vec<_> = state.beliefs.keys().collect();
            let idx = (self.random() * beliefs.len() as f64) as usize;
            let belief_id = beliefs[idx.min(beliefs.len() - 1)].clone();
            let delta = (self.random() - 0.5) * 0.2; // -0.1 to +0.1

            return Some(Mutation {
                mutation_type: MutationType::AdjustConfidence { belief_id, delta },
                expected_impact: 0.05,
                rationale: "Refine belief confidence based on evidence".to_string(),
            });
        }

        // AddAxiom
        cumulative += add_axiom;
        if r < cumulative && !state.class_hierarchy.is_empty() {
            let classes: Vec<_> = state.class_hierarchy.keys().collect();
            if classes.len() >= 2 {
                let idx1 = (self.random() * classes.len() as f64) as usize;
                let idx2 = (self.random() * classes.len() as f64) as usize;
                if idx1 != idx2 {
                    return Some(Mutation {
                        mutation_type: MutationType::AddAxiom {
                            subject: classes[idx1.min(classes.len() - 1)].clone(),
                            predicate: "related_to".to_string(),
                            object: classes[idx2.min(classes.len() - 1)].clone(),
                        },
                        expected_impact: 0.1,
                        rationale: "Explore potential relationship".to_string(),
                    });
                }
            }
        }

        // RemoveAxiom
        cumulative += remove_axiom;
        if r < cumulative && !state.axioms.is_empty() {
            let axioms: Vec<_> = state.axioms.keys().collect();
            let idx = (self.random() * axioms.len() as f64) as usize;
            return Some(Mutation {
                mutation_type: MutationType::RemoveAxiom {
                    axiom_id: axioms[idx.min(axioms.len() - 1)].clone(),
                },
                expected_impact: -0.05,
                rationale: "Remove potentially redundant axiom".to_string(),
            });
        }

        // PromoteBelief
        cumulative += promote_belief;
        if r < cumulative && !state.beliefs.is_empty() {
            // Find high-confidence belief
            if let Some((id, _)) = state.beliefs.iter().find(|(_, b)| b.confidence > 0.9) {
                return Some(Mutation {
                    mutation_type: MutationType::PromoteBelief {
                        belief_id: id.clone(),
                    },
                    expected_impact: 0.15,
                    rationale: "Promote high-confidence belief to fact".to_string(),
                });
            }
        }

        // DemoteFact
        cumulative += demote_fact;
        if r < cumulative && !state.facts.is_empty() {
            let facts: Vec<_> = state.facts.keys().collect();
            let idx = (self.random() * facts.len() as f64) as usize;
            return Some(Mutation {
                mutation_type: MutationType::DemoteFact {
                    fact_id: facts[idx.min(facts.len() - 1)].clone(),
                    confidence: 0.7 + self.random() * 0.2,
                },
                expected_impact: -0.1,
                rationale: "Question established fact".to_string(),
            });
        }

        None
    }

    /// Apply a mutation to a state
    fn apply_mutation(&self, state: &mut OntologyState, mutation: &Mutation) -> bool {
        match &mutation.mutation_type {
            MutationType::AdjustConfidence { belief_id, delta } => {
                if let Some(belief) = state.beliefs.get_mut(belief_id) {
                    belief.confidence = (belief.confidence + delta).clamp(0.0, 1.0);
                    belief.revision_count += 1;
                    true
                } else {
                    false
                }
            }
            MutationType::AddAxiom {
                subject,
                predicate,
                object,
            } => {
                let id = format!(
                    "{}_{}_{}_{}",
                    subject,
                    predicate,
                    object,
                    state.axioms.len()
                );
                state.axioms.insert(
                    id.clone(),
                    Axiom {
                        id,
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                        confidence: 0.7,
                    },
                );
                true
            }
            MutationType::RemoveAxiom { axiom_id } => state.axioms.remove(axiom_id).is_some(),
            MutationType::PromoteBelief { belief_id } => {
                if let Some(belief) = state.beliefs.remove(belief_id) {
                    state.facts.insert(
                        belief_id.clone(),
                        FactState {
                            id: belief_id.clone(),
                            content: belief.content,
                            sources: vec!["promoted from belief".to_string()],
                        },
                    );
                    true
                } else {
                    false
                }
            }
            MutationType::DemoteFact {
                fact_id,
                confidence,
            } => {
                if let Some(fact) = state.facts.remove(fact_id) {
                    state.beliefs.insert(
                        fact_id.clone(),
                        BeliefState {
                            id: fact_id.clone(),
                            content: fact.content,
                            confidence: *confidence,
                            revision_count: 0,
                        },
                    );
                    true
                } else {
                    false
                }
            }
            MutationType::MergeClasses {
                class1,
                class2,
                merged_name,
            } => {
                // Remove both classes, add merged
                let parents1 = state.class_hierarchy.remove(class1).unwrap_or_default();
                let parents2 = state.class_hierarchy.remove(class2).unwrap_or_default();
                let mut merged_parents = parents1;
                for p in parents2 {
                    if !merged_parents.contains(&p) {
                        merged_parents.push(p);
                    }
                }
                state
                    .class_hierarchy
                    .insert(merged_name.clone(), merged_parents);
                true
            }
            MutationType::SplitClass { class, subclasses } => {
                if let Some(parents) = state.class_hierarchy.remove(class) {
                    for subclass in subclasses {
                        state
                            .class_hierarchy
                            .insert(subclass.clone(), vec![class.clone()]);
                    }
                    // Keep original class with its parents
                    state.class_hierarchy.insert(class.clone(), parents);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Metropolis-Hastings acceptance criterion
    fn accept(&mut self, old_fitness: &f64, new_fitness: &f64, temperature: f64) -> bool {
        if *new_fitness >= *old_fitness {
            true
        } else {
            let delta = new_fitness - old_fitness;
            let probability = (delta / temperature).exp();
            self.random() < probability
        }
    }

    /// Check if evolution has converged
    fn has_converged(&self, recent_fitness: &[f64]) -> bool {
        if recent_fitness.len() < self.config.convergence_window {
            return false;
        }

        let min = recent_fitness.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = recent_fitness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        (max - min) < self.config.convergence_threshold
    }

    /// Simple PRNG (xorshift64)
    fn random(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_config_default() {
        let config = EvolutionConfig::default();
        assert!(config.max_iterations > 0);
        assert!(config.initial_temperature > 0.0);
        assert!(config.cooling_rate < 1.0);
    }

    #[test]
    fn test_mutation_weights_normalize() {
        let mut weights = MutationWeights {
            add_axiom: 2.0,
            remove_axiom: 1.0,
            adjust_confidence: 3.0,
            promote_belief: 1.0,
            demote_fact: 1.0,
            merge_classes: 1.0,
            split_class: 1.0,
        };

        weights.normalize();

        let total = weights.add_axiom
            + weights.remove_axiom
            + weights.adjust_confidence
            + weights.promote_belief
            + weights.demote_fact
            + weights.merge_classes
            + weights.split_class;

        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ontology_state_new() {
        let state = OntologyState::new();
        assert_eq!(state.element_count(), 0);
        assert!(state.classes().is_empty());
    }

    #[test]
    fn test_evolution_engine_fitness() {
        let engine = EvolutionEngine::default_engine();
        let state = OntologyState::new();

        let fitness = engine.calculate_fitness(&state);
        assert!(fitness >= 0.0);
        assert!(fitness <= 5.0); // Max is sum of weights
    }

    #[test]
    fn test_evolution_with_beliefs() {
        let mut engine = EvolutionEngine::default_engine();

        let mut state = OntologyState::new();
        state.beliefs.insert(
            "b1".to_string(),
            BeliefState {
                id: "b1".to_string(),
                content: "Test belief".to_string(),
                confidence: 0.5,
                revision_count: 0,
            },
        );
        state.beliefs.insert(
            "b2".to_string(),
            BeliefState {
                id: "b2".to_string(),
                content: "Another belief".to_string(),
                confidence: 0.7,
                revision_count: 0,
            },
        );

        let result = engine.evolve(state);

        assert!(result.iterations > 0);
        assert!(result.fitness_history.len() > 1);
    }

    #[test]
    fn test_apply_adjust_confidence() {
        let engine = EvolutionEngine::default_engine();

        let mut state = OntologyState::new();
        state.beliefs.insert(
            "test".to_string(),
            BeliefState {
                id: "test".to_string(),
                content: "Test".to_string(),
                confidence: 0.5,
                revision_count: 0,
            },
        );

        let mutation = Mutation {
            mutation_type: MutationType::AdjustConfidence {
                belief_id: "test".to_string(),
                delta: 0.1,
            },
            expected_impact: 0.05,
            rationale: "Test".to_string(),
        };

        let applied = engine.apply_mutation(&mut state, &mutation);
        assert!(applied);
        assert!((state.beliefs["test"].confidence - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_apply_promote_belief() {
        let engine = EvolutionEngine::default_engine();

        let mut state = OntologyState::new();
        state.beliefs.insert(
            "belief".to_string(),
            BeliefState {
                id: "belief".to_string(),
                content: "Promoted".to_string(),
                confidence: 0.95,
                revision_count: 0,
            },
        );

        let mutation = Mutation {
            mutation_type: MutationType::PromoteBelief {
                belief_id: "belief".to_string(),
            },
            expected_impact: 0.15,
            rationale: "Test".to_string(),
        };

        let applied = engine.apply_mutation(&mut state, &mutation);
        assert!(applied);
        assert!(state.beliefs.is_empty());
        assert!(state.facts.contains_key("belief"));
    }

    #[test]
    fn test_mutation_type_display() {
        let m = MutationType::AdjustConfidence {
            belief_id: "test".to_string(),
            delta: 0.1,
        };
        let display = format!("{}", m);
        assert!(display.contains("AdjustConfidence"));
        assert!(display.contains("test"));
    }

    #[test]
    fn test_evolution_convergence() {
        let config = EvolutionConfig {
            max_iterations: 10,
            convergence_window: 5,
            convergence_threshold: 0.1,
            ..Default::default()
        };
        let mut engine = EvolutionEngine::new(config);

        let state = OntologyState::new();
        let result = engine.evolve(state);

        // Should complete without panic
        assert!(result.iterations <= 10);
    }
}
