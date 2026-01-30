//! Epistemic Agent System
//!
//! Provides autonomous agents that can query, revise, and evolve knowledge
//! within the Sounio type system.
//!
//! # Agent Types
//!
//! | Agent | Purpose | Autonomy Level |
//! |-------|---------|----------------|
//! | Query | Answer questions about knowledge | Read-only |
//! | Revise | Update knowledge with new evidence | Supervised |
//! | Evolve | Adapt ontologies over time | Semi-autonomous |
//! | Generate | Create new knowledge from descriptions | LLM-assisted |
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    EpistemicAgentSystem                         │
//! │                                                                 │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
//! │  │ TaskQueue   │  │ AgentPool   │  │    KnowledgeBase        │ │
//! │  │             │──│             │──│                         │ │
//! │  │ - pending   │  │ - query     │  │ - facts                 │ │
//! │  │ - running   │  │ - revise    │  │ - beliefs               │ │
//! │  │ - completed │  │ - evolve    │  │ - constraints           │ │
//! │  └─────────────┘  │ - generate  │  │ - provenance            │ │
//! │                   └─────────────┘  └─────────────────────────┘ │
//! │                                                                 │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                   AgentTask                              │   │
//! │  │  - id: TaskId                                            │   │
//! │  │  - kind: TaskKind                                        │   │
//! │  │  - priority: Priority                                    │   │
//! │  │  - status: TaskStatus                                    │   │
//! │  │  - result: Option<TaskResult>                            │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Inspiration
//!
//! Based on Richard Sutton's NeurIPS 2025 keynote insights:
//! > "We need agents that learn continually. We need world models and planning.
//! > We need knowledge that is high-level and learnable."

pub mod agent;
pub mod knowledge_base;
pub mod tasks;

pub use agent::{
    Agent, AgentConfig, AgentKind, AgentResult, EpistemicAgent, EvolveAgent, GenerateAgent,
    QueryAgent, ReviseAgent,
};
pub use knowledge_base::{Belief, Constraint, Fact, KnowledgeBase, KnowledgeEntry};
pub use tasks::{AgentTask, Priority, TaskId, TaskKind, TaskResult, TaskStatus};

use std::fmt;

/// Errors that can occur in the agent system
#[derive(Debug, Clone)]
pub enum AgentError {
    /// Task not found
    TaskNotFound(TaskId),
    /// Agent unavailable
    AgentUnavailable(AgentKind),
    /// Knowledge base error
    KnowledgeError(String),
    /// LLM error (for generate agent)
    LLMError(String),
    /// Revision conflict
    RevisionConflict(String),
    /// Evolution failed
    EvolutionFailed(String),
    /// Query failed
    QueryFailed(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::TaskNotFound(id) => write!(f, "Task not found: {:?}", id),
            AgentError::AgentUnavailable(kind) => write!(f, "Agent unavailable: {:?}", kind),
            AgentError::KnowledgeError(msg) => write!(f, "Knowledge error: {}", msg),
            AgentError::LLMError(msg) => write!(f, "LLM error: {}", msg),
            AgentError::RevisionConflict(msg) => write!(f, "Revision conflict: {}", msg),
            AgentError::EvolutionFailed(msg) => write!(f, "Evolution failed: {}", msg),
            AgentError::QueryFailed(msg) => write!(f, "Query failed: {}", msg),
        }
    }
}

impl std::error::Error for AgentError {}

/// Result type for agent operations
pub type AgentOpResult<T> = Result<T, AgentError>;
