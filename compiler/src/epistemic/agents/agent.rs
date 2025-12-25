//! Epistemic Agent Implementations
//!
//! Provides concrete agent types for different epistemic operations.

use super::{AgentError, AgentOpResult, KnowledgeBase, tasks::*};
use crate::llm::LLMClientRegistry;
use crate::ontology::llm_gen::{GenerationConfig, OntologyGenerator};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Task timeout in seconds
    pub task_timeout_secs: u64,
    /// Whether agent requires human approval
    pub requires_approval: bool,
    /// Minimum confidence for automatic actions
    pub min_auto_confidence: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 1,
            task_timeout_secs: 300,
            requires_approval: true,
            min_auto_confidence: 0.9,
        }
    }
}

/// Agent kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentKind {
    /// Query agent - read-only knowledge access
    Query,
    /// Revise agent - update knowledge with evidence
    Revise,
    /// Evolve agent - adapt ontologies over time
    Evolve,
    /// Generate agent - create new knowledge from text
    Generate,
}

impl std::fmt::Display for AgentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentKind::Query => write!(f, "Query"),
            AgentKind::Revise => write!(f, "Revise"),
            AgentKind::Evolve => write!(f, "Evolve"),
            AgentKind::Generate => write!(f, "Generate"),
        }
    }
}

/// Result from an agent operation
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// Success or failure
    pub success: bool,
    /// Message describing result
    pub message: String,
    /// Confidence in result
    pub confidence: f64,
    /// Any warnings
    pub warnings: Vec<String>,
}

impl AgentResult {
    /// Create a success result
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            confidence: 1.0,
            warnings: Vec::new(),
        }
    }

    /// Create a failure result
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            confidence: 0.0,
            warnings: Vec::new(),
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add warning
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }
}

/// Trait for epistemic agents
pub trait Agent: Send + Sync {
    /// Get the agent kind
    fn kind(&self) -> AgentKind;

    /// Check if agent is available
    fn is_available(&self) -> bool;

    /// Execute a task
    fn execute(&self, task: &mut AgentTask, kb: &mut KnowledgeBase) -> AgentOpResult<AgentResult>;
}

/// Combined epistemic agent with all capabilities
pub struct EpistemicAgent {
    /// Query agent
    query: QueryAgent,
    /// Revise agent
    revise: ReviseAgent,
    /// Evolve agent
    evolve: EvolveAgent,
    /// Generate agent (optional, requires LLM)
    generate: Option<GenerateAgent>,
    /// Task queue
    task_queue: Mutex<VecDeque<AgentTask>>,
    /// Shared knowledge base
    knowledge_base: Arc<Mutex<KnowledgeBase>>,
}

impl EpistemicAgent {
    /// Create a new epistemic agent
    pub fn new(knowledge_base: KnowledgeBase) -> Self {
        Self {
            query: QueryAgent::new(),
            revise: ReviseAgent::new(),
            evolve: EvolveAgent::new(),
            generate: None,
            task_queue: Mutex::new(VecDeque::new()),
            knowledge_base: Arc::new(Mutex::new(knowledge_base)),
        }
    }

    /// Add LLM capability for generation
    pub fn with_llm(mut self, registry: LLMClientRegistry) -> Self {
        self.generate = Some(GenerateAgent::new(registry));
        self
    }

    /// Submit a task
    pub fn submit(&self, task: AgentTask) -> TaskId {
        let id = task.id;
        let mut queue = self.task_queue.lock().unwrap();
        queue.push_back(task);
        id
    }

    /// Execute the next pending task
    pub fn execute_next(&self) -> AgentOpResult<Option<TaskResult>> {
        let mut task = {
            let mut queue = self.task_queue.lock().unwrap();
            match queue.pop_front() {
                Some(t) => t,
                None => return Ok(None),
            }
        };

        let mut kb = self.knowledge_base.lock().unwrap();
        task.start();

        let result = match &task.kind {
            TaskKind::Query(_) => self.query.execute(&mut task, &mut kb),
            TaskKind::Revise(_) => self.revise.execute(&mut task, &mut kb),
            TaskKind::Evolve(_) => self.evolve.execute(&mut task, &mut kb),
            TaskKind::Generate(_) => {
                if let Some(ref generator) = self.generate {
                    generator.execute(&mut task, &mut kb)
                } else {
                    Err(AgentError::AgentUnavailable(AgentKind::Generate))
                }
            }
        };

        match result {
            Ok(agent_result) => {
                let task_result = task.result.clone();
                Ok(task_result)
            }
            Err(e) => {
                task.fail(e.to_string());
                Err(e)
            }
        }
    }

    /// Get pending task count
    pub fn pending_count(&self) -> usize {
        self.task_queue.lock().unwrap().len()
    }

    /// Check if LLM generation is available
    pub fn has_llm(&self) -> bool {
        self.generate.is_some()
    }
}

// ============================================================================
// Query Agent
// ============================================================================

/// Agent for querying the knowledge base
pub struct QueryAgent {
    config: AgentConfig,
}

impl QueryAgent {
    /// Create a new query agent
    pub fn new() -> Self {
        Self {
            config: AgentConfig {
                requires_approval: false,
                ..Default::default()
            },
        }
    }
}

impl Default for QueryAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Agent for QueryAgent {
    fn kind(&self) -> AgentKind {
        AgentKind::Query
    }

    fn is_available(&self) -> bool {
        true
    }

    fn execute(&self, task: &mut AgentTask, kb: &mut KnowledgeBase) -> AgentOpResult<AgentResult> {
        let query_task = match &task.kind {
            TaskKind::Query(q) => q,
            _ => return Err(AgentError::QueryFailed("Not a query task".into())),
        };

        let start = std::time::Instant::now();

        let results = kb.search(&query_task.query, query_task.min_confidence);
        let limited: Vec<_> = results.into_iter().take(query_task.max_results).collect();

        let total = limited.len();
        let entries: Vec<_> = limited
            .iter()
            .map(|r| QueryMatch {
                id: r.id.clone(),
                relevance: r.relevance,
                confidence: r.entry.confidence(),
                summary: r.entry.content().chars().take(100).collect(),
            })
            .collect();

        let query_result = QueryResult {
            entries,
            total_matches: total,
            execution_time_ms: start.elapsed().as_millis() as u64,
        };

        task.complete(TaskResult::Query(query_result));

        Ok(AgentResult::success(format!("Found {} matches", total)))
    }
}

// ============================================================================
// Revise Agent
// ============================================================================

/// Agent for revising knowledge with new evidence
pub struct ReviseAgent {
    config: AgentConfig,
}

impl ReviseAgent {
    /// Create a new revise agent
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
        }
    }
}

impl Default for ReviseAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Agent for ReviseAgent {
    fn kind(&self) -> AgentKind {
        AgentKind::Revise
    }

    fn is_available(&self) -> bool {
        true
    }

    fn execute(&self, task: &mut AgentTask, kb: &mut KnowledgeBase) -> AgentOpResult<AgentResult> {
        let revise_task = match &task.kind {
            TaskKind::Revise(r) => r.clone(),
            _ => return Err(AgentError::RevisionConflict("Not a revise task".into())),
        };

        // Check if target exists
        let entry = kb
            .get(&revise_task.target)
            .ok_or_else(|| AgentError::KnowledgeError("Target not found".into()))?;

        let old_confidence = entry.confidence();

        // Calculate new confidence based on strategy and evidence
        let new_confidence = match revise_task.strategy {
            RevisionStrategy::Replace => revise_task.evidence.confidence,
            RevisionStrategy::Merge => (old_confidence + revise_task.evidence.confidence) / 2.0,
            RevisionStrategy::Bayesian => {
                // Simple Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
                // Simplified: weight by evidence confidence
                let prior = old_confidence;
                let likelihood = revise_task.evidence.confidence;

                (prior * likelihood) / (prior * likelihood + (1.0 - prior) * (1.0 - likelihood))
            }
            RevisionStrategy::Alternative => {
                // Keep old, add alternative (not implemented in simple KB)
                old_confidence
            }
        };

        // Apply revision
        kb.revise_belief(
            &revise_task.target,
            new_confidence,
            &revise_task.evidence.source,
        )?;

        let revision_result = RevisionResult {
            applied: true,
            old_confidence,
            new_confidence,
            explanation: format!(
                "Revised using {:?} strategy with evidence from {}",
                revise_task.strategy, revise_task.evidence.source
            ),
        };

        task.complete(TaskResult::Revise(revision_result));

        Ok(AgentResult::success("Revision applied").with_confidence(new_confidence))
    }
}

// ============================================================================
// Evolve Agent
// ============================================================================

/// Agent for evolving ontologies using MCMC-inspired methods
pub struct EvolveAgent {
    config: AgentConfig,
}

impl EvolveAgent {
    /// Create a new evolve agent
    pub fn new() -> Self {
        Self {
            config: AgentConfig {
                requires_approval: true,
                min_auto_confidence: 0.95,
                ..Default::default()
            },
        }
    }

    /// Calculate fitness of current knowledge base state
    fn calculate_fitness(&self, kb: &KnowledgeBase, objective: &EvolutionObjective) -> f64 {
        match objective {
            EvolutionObjective::Consistency => {
                let violations = kb.check_constraints();
                let penalty = violations.len() as f64 * 0.1;
                (1.0 - penalty).max(0.0)
            }
            EvolutionObjective::Coverage => {
                let stats = kb.stats();
                let total = stats.fact_count + stats.belief_count;
                (total as f64 / 100.0).min(1.0) // Normalize to 0-1
            }
            EvolutionObjective::Parsimony => {
                let stats = kb.stats();
                let redundancy_penalty = 0.0; // Would need duplicate detection
                1.0 - redundancy_penalty
            }
            EvolutionObjective::Alignment { reference: _ } => {
                // Would need external ontology comparison
                0.5
            }
            EvolutionObjective::Custom { name: _ } => {
                // Custom fitness not implemented
                0.5
            }
        }
    }

    /// Propose a mutation to the knowledge base
    fn propose_mutation(&self, kb: &mut KnowledgeBase) -> Option<EvolutionChange> {
        // Simple mutation: adjust a random belief's confidence
        let stats = kb.stats();
        if stats.belief_count == 0 {
            return None;
        }

        // In a real implementation, we'd randomly select and mutate
        // For now, just return None (no mutation)
        None
    }

    /// Metropolis acceptance criterion
    fn accept_mutation(&self, old_fitness: f64, new_fitness: f64, temperature: f64) -> bool {
        if new_fitness >= old_fitness {
            true
        } else {
            let delta = new_fitness - old_fitness;
            let probability = (delta / temperature).exp();
            rand_probability() < probability
        }
    }
}

/// Simple random probability (would use rand crate in production)
fn rand_probability() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f64) / (u32::MAX as f64)
}

impl Default for EvolveAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Agent for EvolveAgent {
    fn kind(&self) -> AgentKind {
        AgentKind::Evolve
    }

    fn is_available(&self) -> bool {
        true
    }

    fn execute(&self, task: &mut AgentTask, kb: &mut KnowledgeBase) -> AgentOpResult<AgentResult> {
        let evolve_task = match &task.kind {
            TaskKind::Evolve(e) => e.clone(),
            _ => return Err(AgentError::EvolutionFailed("Not an evolve task".into())),
        };

        let initial_fitness = self.calculate_fitness(kb, &evolve_task.objective);
        let mut current_fitness = initial_fitness;
        let mut changes = Vec::new();
        let mut temperature = 1.0;
        let cooling_rate = 0.99;

        for iteration in 0..evolve_task.max_iterations {
            // Propose mutation
            if let Some(change) = self.propose_mutation(kb) {
                let new_fitness = self.calculate_fitness(kb, &evolve_task.objective);

                if self.accept_mutation(current_fitness, new_fitness, temperature) {
                    current_fitness = new_fitness;
                    changes.push(change);
                }
            }

            // Cool down
            temperature *= cooling_rate;

            // Check convergence
            if (current_fitness - initial_fitness).abs() < evolve_task.convergence_threshold {
                break;
            }
        }

        let evolution_result = EvolutionResult {
            iterations: evolve_task.max_iterations.min(100),
            converged: (current_fitness - initial_fitness).abs()
                < evolve_task.convergence_threshold,
            initial_fitness,
            final_fitness: current_fitness,
            changes,
        };

        task.complete(TaskResult::Evolve(evolution_result.clone()));

        Ok(AgentResult::success(format!(
            "Evolution complete: fitness {} -> {}",
            initial_fitness, current_fitness
        ))
        .with_confidence(current_fitness))
    }
}

// ============================================================================
// Generate Agent
// ============================================================================

/// Agent for generating new knowledge using LLMs
pub struct GenerateAgent {
    registry: LLMClientRegistry,
    config: AgentConfig,
}

impl GenerateAgent {
    /// Create a new generate agent
    pub fn new(registry: LLMClientRegistry) -> Self {
        Self {
            registry,
            config: AgentConfig {
                requires_approval: true,
                min_auto_confidence: 0.85,
                ..Default::default()
            },
        }
    }
}

impl Agent for GenerateAgent {
    fn kind(&self) -> AgentKind {
        AgentKind::Generate
    }

    fn is_available(&self) -> bool {
        self.registry.is_available()
    }

    fn execute(&self, task: &mut AgentTask, kb: &mut KnowledgeBase) -> AgentOpResult<AgentResult> {
        let generate_task = match &task.kind {
            TaskKind::Generate(g) => g.clone(),
            _ => return Err(AgentError::LLMError("Not a generate task".into())),
        };

        if !self.is_available() {
            return Err(AgentError::AgentUnavailable(AgentKind::Generate));
        }

        // Use ontology generator
        let gen_config = GenerationConfig::for_domain(&generate_task.domain);
        let generator = OntologyGenerator::with_config(
            LLMClientRegistry::from_env(), // Would share registry in production
            gen_config,
        );

        let fragment = generator
            .generate_from_text(&generate_task.description, &generate_task.domain)
            .map_err(|e| AgentError::LLMError(e.to_string()))?;

        // Validate
        let validator = crate::ontology::llm_gen::OntologyValidator::new();
        let validation = validator.validate(&fragment);

        // Determine if we should integrate
        let should_integrate = match generate_task.integration {
            IntegrationMode::Preview => false,
            IntegrationMode::Supervised => false, // Would need UI interaction
            IntegrationMode::Automatic { min_confidence } => {
                fragment.average_confidence() >= min_confidence && validation.is_valid()
            }
        };

        // Integration would add to knowledge base here
        if should_integrate {
            for class in &fragment.classes {
                kb.add_belief(
                    &class.name,
                    super::knowledge_base::Belief::new(&class.label, class.confidence)
                        .in_domain(&generate_task.domain),
                );
            }
        }

        let generation_result = GenerationResult {
            integrated: should_integrate,
            classes_generated: fragment.classes.len(),
            relations_generated: fragment
                .axioms
                .iter()
                .filter(|a| {
                    matches!(
                        a,
                        crate::ontology::llm_gen::GeneratedAxiom::ObjectPropertyAssertion { .. }
                    )
                })
                .count(),
            average_confidence: fragment.average_confidence(),
            validation_passed: validation.is_valid(),
        };

        task.complete(TaskResult::Generate(generation_result.clone()));

        let message = if should_integrate {
            format!(
                "Generated and integrated {} classes",
                generation_result.classes_generated
            )
        } else {
            format!(
                "Generated {} classes (not integrated)",
                generation_result.classes_generated
            )
        };

        Ok(AgentResult::success(message).with_confidence(fragment.average_confidence()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_agent() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact(
            "scaffold",
            super::super::knowledge_base::Fact::new("Scaffolds are structures"),
        );

        let agent = QueryAgent::new();
        let mut task = AgentTask::query("scaffold");

        let result = agent.execute(&mut task, &mut kb).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_revise_agent() {
        let mut kb = KnowledgeBase::new();
        kb.add_belief(
            "hypothesis",
            super::super::knowledge_base::Belief::new("Initial hypothesis", 0.5),
        );

        let agent = ReviseAgent::new();
        let evidence =
            Evidence::new("experiment", "New data supports hypothesis").with_confidence(0.9);
        let mut task = AgentTask::revise("hypothesis", evidence);

        let result = agent.execute(&mut task, &mut kb).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_evolve_agent() {
        let mut kb = KnowledgeBase::new();
        kb.add_belief(
            "b1",
            super::super::knowledge_base::Belief::new("Belief 1", 0.7),
        );

        let agent = EvolveAgent::new();
        let mut task = AgentTask::evolve("test", EvolutionObjective::Consistency);

        let result = agent.execute(&mut task, &mut kb).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_epistemic_agent_submit() {
        let kb = KnowledgeBase::new();
        let agent = EpistemicAgent::new(kb);

        let task = AgentTask::query("test");
        let id = agent.submit(task);

        assert_eq!(agent.pending_count(), 1);
    }

    #[test]
    fn test_agent_result_builder() {
        let result = AgentResult::success("Test passed")
            .with_confidence(0.95)
            .with_warning("Minor issue");

        assert!(result.success);
        assert!((result.confidence - 0.95).abs() < f64::EPSILON);
        assert_eq!(result.warnings.len(), 1);
    }
}
