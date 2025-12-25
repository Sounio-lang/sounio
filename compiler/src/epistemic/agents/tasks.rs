//! Agent Task Definitions
//!
//! Defines the task structure for epistemic agents.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global task ID counter
static TASK_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Unique task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

impl TaskId {
    /// Generate a new unique task ID
    pub fn new() -> Self {
        TaskId(TASK_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "task-{}", self.0)
    }
}

/// Types of agent tasks
#[derive(Debug, Clone)]
pub enum TaskKind {
    /// Query the knowledge base
    Query(QueryTask),
    /// Revise existing knowledge
    Revise(ReviseTask),
    /// Evolve the ontology
    Evolve(EvolveTask),
    /// Generate new knowledge
    Generate(GenerateTask),
}

impl fmt::Display for TaskKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskKind::Query(_) => write!(f, "Query"),
            TaskKind::Revise(_) => write!(f, "Revise"),
            TaskKind::Evolve(_) => write!(f, "Evolve"),
            TaskKind::Generate(_) => write!(f, "Generate"),
        }
    }
}

/// A query task
#[derive(Debug, Clone)]
pub struct QueryTask {
    /// Natural language or structured query
    pub query: String,
    /// Domain to search in
    pub domain: Option<String>,
    /// Maximum results to return
    pub max_results: usize,
    /// Minimum confidence threshold
    pub min_confidence: f64,
}

impl QueryTask {
    /// Create a new query task
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            domain: None,
            max_results: 10,
            min_confidence: 0.5,
        }
    }

    /// Set domain filter
    pub fn in_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Set max results
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.max_results = limit;
        self
    }

    /// Set min confidence
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }
}

/// A revision task
#[derive(Debug, Clone)]
pub struct ReviseTask {
    /// Target knowledge entry to revise
    pub target: String,
    /// New evidence supporting revision
    pub evidence: Evidence,
    /// Revision strategy
    pub strategy: RevisionStrategy,
}

/// Evidence for knowledge revision
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Source of evidence
    pub source: String,
    /// Description of evidence
    pub description: String,
    /// Confidence in evidence
    pub confidence: f64,
    /// Type of evidence
    pub kind: EvidenceKind,
}

impl Evidence {
    /// Create new evidence
    pub fn new(source: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            description: description.into(),
            confidence: 0.8,
            kind: EvidenceKind::Observation,
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set kind
    pub fn with_kind(mut self, kind: EvidenceKind) -> Self {
        self.kind = kind;
        self
    }
}

/// Types of evidence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceKind {
    /// Direct observation or measurement
    Observation,
    /// Published research
    Publication,
    /// Expert assertion
    Expert,
    /// Computed derivation
    Computation,
    /// External system
    External,
}

/// Strategy for revising knowledge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RevisionStrategy {
    /// Replace old belief with new
    Replace,
    /// Merge with existing, preferring higher confidence
    Merge,
    /// Keep both as alternatives
    Alternative,
    /// Weighted combination
    #[default]
    Bayesian,
}

/// An evolution task
#[derive(Debug, Clone)]
pub struct EvolveTask {
    /// Target ontology or knowledge domain
    pub target: String,
    /// Evolution objective
    pub objective: EvolutionObjective,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

/// Objectives for ontology evolution
#[derive(Debug, Clone)]
pub enum EvolutionObjective {
    /// Maximize consistency
    Consistency,
    /// Maximize coverage of domain
    Coverage,
    /// Minimize redundancy
    Parsimony,
    /// Align with external ontology
    Alignment { reference: String },
    /// Custom fitness function
    Custom { name: String },
}

/// A generation task
#[derive(Debug, Clone)]
pub struct GenerateTask {
    /// Natural language description
    pub description: String,
    /// Target domain
    pub domain: String,
    /// What to generate
    pub target: GenerationTarget,
    /// Integration mode
    pub integration: IntegrationMode,
}

/// What to generate
#[derive(Debug, Clone)]
pub enum GenerationTarget {
    /// Generate classes/concepts
    Classes,
    /// Generate relationships
    Relations,
    /// Generate axioms
    Axioms,
    /// Generate complete fragment
    Fragment,
    /// Generate definitions
    Definitions,
}

/// How to integrate generated content
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IntegrationMode {
    /// Don't integrate, just return
    Preview,
    /// Integrate after human approval
    #[default]
    Supervised,
    /// Integrate if confidence above threshold
    Automatic { min_confidence: f64 },
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Priority {
    /// Background task, run when idle
    Low = 0,
    /// Normal priority
    #[default]
    Normal = 1,
    /// High priority, run soon
    High = 2,
    /// Critical, run immediately
    Critical = 3,
}

/// Task execution status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    /// Waiting to be executed
    Pending,
    /// Currently being executed
    Running,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed(String),
    /// Cancelled by user
    Cancelled,
}

impl TaskStatus {
    /// Check if task is finished (completed, failed, or cancelled)
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            TaskStatus::Completed | TaskStatus::Failed(_) | TaskStatus::Cancelled
        )
    }

    /// Check if task succeeded
    pub fn is_success(&self) -> bool {
        matches!(self, TaskStatus::Completed)
    }
}

/// Complete agent task
#[derive(Debug, Clone)]
pub struct AgentTask {
    /// Unique identifier
    pub id: TaskId,
    /// Task kind and parameters
    pub kind: TaskKind,
    /// Priority level
    pub priority: Priority,
    /// Current status
    pub status: TaskStatus,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Completion timestamp
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Result (if completed)
    pub result: Option<TaskResult>,
}

impl AgentTask {
    /// Create a new query task
    pub fn query(query: impl Into<String>) -> Self {
        Self::new(TaskKind::Query(QueryTask::new(query)))
    }

    /// Create a new revise task
    pub fn revise(target: impl Into<String>, evidence: Evidence) -> Self {
        Self::new(TaskKind::Revise(ReviseTask {
            target: target.into(),
            evidence,
            strategy: RevisionStrategy::default(),
        }))
    }

    /// Create a new evolve task
    pub fn evolve(target: impl Into<String>, objective: EvolutionObjective) -> Self {
        Self::new(TaskKind::Evolve(EvolveTask {
            target: target.into(),
            objective,
            max_iterations: 100,
            convergence_threshold: 0.01,
        }))
    }

    /// Create a new generate task
    pub fn generate(description: impl Into<String>, domain: impl Into<String>) -> Self {
        Self::new(TaskKind::Generate(GenerateTask {
            description: description.into(),
            domain: domain.into(),
            target: GenerationTarget::Fragment,
            integration: IntegrationMode::default(),
        }))
    }

    /// Create a new task with given kind
    pub fn new(kind: TaskKind) -> Self {
        Self {
            id: TaskId::new(),
            kind,
            priority: Priority::default(),
            status: TaskStatus::Pending,
            created_at: chrono::Utc::now(),
            completed_at: None,
            result: None,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Mark task as running
    pub fn start(&mut self) {
        self.status = TaskStatus::Running;
    }

    /// Mark task as completed with result
    pub fn complete(&mut self, result: TaskResult) {
        self.status = TaskStatus::Completed;
        self.completed_at = Some(chrono::Utc::now());
        self.result = Some(result);
    }

    /// Mark task as failed
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = TaskStatus::Failed(error.into());
        self.completed_at = Some(chrono::Utc::now());
    }

    /// Cancel the task
    pub fn cancel(&mut self) {
        self.status = TaskStatus::Cancelled;
        self.completed_at = Some(chrono::Utc::now());
    }
}

/// Result of a completed task
#[derive(Debug, Clone)]
pub enum TaskResult {
    /// Query results
    Query(QueryResult),
    /// Revision result
    Revise(RevisionResult),
    /// Evolution result
    Evolve(EvolutionResult),
    /// Generation result
    Generate(GenerationResult),
}

/// Result of a query task
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Matching knowledge entries
    pub entries: Vec<QueryMatch>,
    /// Total matches found
    pub total_matches: usize,
    /// Query execution time (ms)
    pub execution_time_ms: u64,
}

/// A single query match
#[derive(Debug, Clone)]
pub struct QueryMatch {
    /// Entry identifier
    pub id: String,
    /// Relevance score
    pub relevance: f64,
    /// Confidence in the entry
    pub confidence: f64,
    /// Summary of matched content
    pub summary: String,
}

/// Result of a revision task
#[derive(Debug, Clone)]
pub struct RevisionResult {
    /// Whether revision was applied
    pub applied: bool,
    /// Previous confidence
    pub old_confidence: f64,
    /// New confidence after revision
    pub new_confidence: f64,
    /// Explanation of revision
    pub explanation: String,
}

/// Result of an evolution task
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    /// Number of iterations performed
    pub iterations: usize,
    /// Converged (met threshold)
    pub converged: bool,
    /// Initial fitness
    pub initial_fitness: f64,
    /// Final fitness
    pub final_fitness: f64,
    /// Changes made
    pub changes: Vec<EvolutionChange>,
}

/// A single change made during evolution
#[derive(Debug, Clone)]
pub struct EvolutionChange {
    /// Type of change
    pub kind: String,
    /// Affected element
    pub element: String,
    /// Description
    pub description: String,
}

/// Result of a generation task
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Whether content was integrated
    pub integrated: bool,
    /// Number of classes generated
    pub classes_generated: usize,
    /// Number of relations generated
    pub relations_generated: usize,
    /// Average confidence
    pub average_confidence: f64,
    /// Validation passed
    pub validation_passed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_id_unique() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_query_task_builder() {
        let task = QueryTask::new("what is a scaffold")
            .in_domain("biomaterials")
            .with_limit(5)
            .with_min_confidence(0.8);

        assert_eq!(task.query, "what is a scaffold");
        assert_eq!(task.domain, Some("biomaterials".to_string()));
        assert_eq!(task.max_results, 5);
        assert!((task.min_confidence - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_agent_task_lifecycle() {
        let mut task = AgentTask::query("test query");

        assert!(matches!(task.status, TaskStatus::Pending));

        task.start();
        assert!(matches!(task.status, TaskStatus::Running));

        task.complete(TaskResult::Query(QueryResult {
            entries: vec![],
            total_matches: 0,
            execution_time_ms: 10,
        }));

        assert!(task.status.is_finished());
        assert!(task.status.is_success());
        assert!(task.completed_at.is_some());
    }

    #[test]
    fn test_task_failure() {
        let mut task = AgentTask::query("test");
        task.start();
        task.fail("Something went wrong");

        assert!(task.status.is_finished());
        assert!(!task.status.is_success());
        assert!(matches!(task.status, TaskStatus::Failed(_)));
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_evidence_builder() {
        let evidence = Evidence::new("experiment", "Observed phenomenon X")
            .with_confidence(0.95)
            .with_kind(EvidenceKind::Observation);

        assert_eq!(evidence.source, "experiment");
        assert!((evidence.confidence - 0.95).abs() < f64::EPSILON);
        assert!(matches!(evidence.kind, EvidenceKind::Observation));
    }
}
