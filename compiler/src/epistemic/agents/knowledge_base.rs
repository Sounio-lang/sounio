//! In-Memory Knowledge Base for Epistemic Agents
//!
//! Provides a lightweight knowledge store for agent operations.

use super::{AgentError, AgentOpResult};
use std::collections::HashMap;

/// In-memory knowledge base
#[derive(Debug, Default)]
pub struct KnowledgeBase {
    /// Facts (high-confidence, established knowledge)
    facts: HashMap<String, Fact>,
    /// Beliefs (medium-confidence, subject to revision)
    beliefs: HashMap<String, Belief>,
    /// Constraints (rules and invariants)
    constraints: Vec<Constraint>,
    /// Domain metadata
    domains: HashMap<String, DomainInfo>,
}

impl KnowledgeBase {
    /// Create a new empty knowledge base
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fact
    pub fn add_fact(&mut self, id: impl Into<String>, fact: Fact) {
        self.facts.insert(id.into(), fact);
    }

    /// Add a belief
    pub fn add_belief(&mut self, id: impl Into<String>, belief: Belief) {
        self.beliefs.insert(id.into(), belief);
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Register a domain
    pub fn register_domain(&mut self, id: impl Into<String>, info: DomainInfo) {
        self.domains.insert(id.into(), info);
    }

    /// Get a fact by ID
    pub fn get_fact(&self, id: &str) -> Option<&Fact> {
        self.facts.get(id)
    }

    /// Get a belief by ID
    pub fn get_belief(&self, id: &str) -> Option<&Belief> {
        self.beliefs.get(id)
    }

    /// Get any knowledge entry by ID
    pub fn get(&self, id: &str) -> Option<KnowledgeEntry> {
        if let Some(fact) = self.facts.get(id) {
            return Some(KnowledgeEntry::Fact(fact.clone()));
        }
        if let Some(belief) = self.beliefs.get(id) {
            return Some(KnowledgeEntry::Belief(belief.clone()));
        }
        None
    }

    /// Query facts by domain
    pub fn facts_in_domain(&self, domain: &str) -> Vec<&Fact> {
        self.facts
            .values()
            .filter(|f| f.domain.as_deref() == Some(domain))
            .collect()
    }

    /// Query beliefs by domain
    pub fn beliefs_in_domain(&self, domain: &str) -> Vec<&Belief> {
        self.beliefs
            .values()
            .filter(|b| b.domain.as_deref() == Some(domain))
            .collect()
    }

    /// Query by content (simple substring search)
    pub fn search(&self, query: &str, min_confidence: f64) -> Vec<SearchResult> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for (id, fact) in &self.facts {
            if fact.content.to_lowercase().contains(&query_lower) {
                results.push(SearchResult {
                    id: id.clone(),
                    entry: KnowledgeEntry::Fact(fact.clone()),
                    relevance: self.compute_relevance(&fact.content, &query_lower),
                });
            }
        }

        for (id, belief) in &self.beliefs {
            if belief.confidence >= min_confidence
                && belief.content.to_lowercase().contains(&query_lower)
            {
                results.push(SearchResult {
                    id: id.clone(),
                    entry: KnowledgeEntry::Belief(belief.clone()),
                    relevance: self.compute_relevance(&belief.content, &query_lower),
                });
            }
        }

        // Sort by relevance
        results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
        results
    }

    /// Compute simple relevance score
    fn compute_relevance(&self, content: &str, query: &str) -> f64 {
        let content_lower = content.to_lowercase();
        let words: Vec<&str> = query.split_whitespace().collect();
        let matched = words.iter().filter(|w| content_lower.contains(*w)).count();

        if words.is_empty() {
            0.0
        } else {
            matched as f64 / words.len() as f64
        }
    }

    /// Revise a belief with new evidence
    pub fn revise_belief(
        &mut self,
        id: &str,
        new_confidence: f64,
        source: impl Into<String>,
    ) -> AgentOpResult<RevisionRecord> {
        let belief = self
            .beliefs
            .get_mut(id)
            .ok_or_else(|| AgentError::KnowledgeError(format!("Belief not found: {}", id)))?;

        let old_confidence = belief.confidence;
        belief.confidence = new_confidence;
        belief.sources.push(source.into());
        belief.revision_count += 1;

        Ok(RevisionRecord {
            id: id.to_string(),
            old_confidence,
            new_confidence,
            revision_number: belief.revision_count,
        })
    }

    /// Promote a belief to a fact (if confidence is high enough)
    pub fn promote_to_fact(&mut self, id: &str, min_confidence: f64) -> AgentOpResult<bool> {
        let belief = self
            .beliefs
            .get(id)
            .ok_or_else(|| AgentError::KnowledgeError(format!("Belief not found: {}", id)))?;

        if belief.confidence >= min_confidence {
            let fact = Fact {
                content: belief.content.clone(),
                domain: belief.domain.clone(),
                sources: belief.sources.clone(),
                established_at: chrono::Utc::now(),
            };

            self.facts.insert(id.to_string(), fact);
            self.beliefs.remove(id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check all constraints
    pub fn check_constraints(&self) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        for constraint in &self.constraints {
            if let Some(violation) = self.check_constraint(constraint) {
                violations.push(violation);
            }
        }

        violations
    }

    /// Check a single constraint
    fn check_constraint(&self, constraint: &Constraint) -> Option<ConstraintViolation> {
        match &constraint.kind {
            ConstraintKind::MinConfidence { entry_id, min } => {
                if let Some(belief) = self.beliefs.get(entry_id)
                    && belief.confidence < *min
                {
                    return Some(ConstraintViolation {
                        constraint: constraint.name.clone(),
                        message: format!(
                            "Belief '{}' confidence {} below minimum {}",
                            entry_id, belief.confidence, min
                        ),
                    });
                }
            }
            ConstraintKind::Required { entry_ids } => {
                for id in entry_ids {
                    if self.get(id).is_none() {
                        return Some(ConstraintViolation {
                            constraint: constraint.name.clone(),
                            message: format!("Required entry '{}' not found", id),
                        });
                    }
                }
            }
            ConstraintKind::Exclusive { entry_ids } => {
                let present: Vec<_> = entry_ids
                    .iter()
                    .filter(|id| self.get(id).is_some())
                    .collect();
                if present.len() > 1 {
                    return Some(ConstraintViolation {
                        constraint: constraint.name.clone(),
                        message: format!("Mutually exclusive entries both present: {:?}", present),
                    });
                }
            }
            ConstraintKind::Custom { check } => {
                if !check(self) {
                    return Some(ConstraintViolation {
                        constraint: constraint.name.clone(),
                        message: "Custom constraint failed".to_string(),
                    });
                }
            }
        }
        None
    }

    /// Get statistics about the knowledge base
    pub fn stats(&self) -> KnowledgeStats {
        let total_beliefs = self.beliefs.len();
        let avg_confidence = if total_beliefs > 0 {
            self.beliefs.values().map(|b| b.confidence).sum::<f64>() / total_beliefs as f64
        } else {
            0.0
        };

        KnowledgeStats {
            fact_count: self.facts.len(),
            belief_count: total_beliefs,
            constraint_count: self.constraints.len(),
            domain_count: self.domains.len(),
            average_belief_confidence: avg_confidence,
        }
    }
}

/// A fact: established, high-confidence knowledge
#[derive(Debug, Clone)]
pub struct Fact {
    /// The content/assertion
    pub content: String,
    /// Domain this fact belongs to
    pub domain: Option<String>,
    /// Sources supporting this fact
    pub sources: Vec<String>,
    /// When the fact was established
    pub established_at: chrono::DateTime<chrono::Utc>,
}

impl Fact {
    /// Create a new fact
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            domain: None,
            sources: Vec::new(),
            established_at: chrono::Utc::now(),
        }
    }

    /// Set domain
    pub fn in_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Add source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.sources.push(source.into());
        self
    }
}

/// A belief: knowledge with associated confidence
#[derive(Debug, Clone)]
pub struct Belief {
    /// The content/assertion
    pub content: String,
    /// Confidence level (0.0-1.0)
    pub confidence: f64,
    /// Domain this belief belongs to
    pub domain: Option<String>,
    /// Sources supporting this belief
    pub sources: Vec<String>,
    /// Number of times revised
    pub revision_count: u32,
    /// Whether this belief is revisable
    pub revisable: bool,
}

impl Belief {
    /// Create a new belief
    pub fn new(content: impl Into<String>, confidence: f64) -> Self {
        Self {
            content: content.into(),
            confidence: confidence.clamp(0.0, 1.0),
            domain: None,
            sources: Vec::new(),
            revision_count: 0,
            revisable: true,
        }
    }

    /// Set domain
    pub fn in_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Add source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.sources.push(source.into());
        self
    }

    /// Set non-revisable
    pub fn non_revisable(mut self) -> Self {
        self.revisable = false;
        self
    }
}

/// A constraint on knowledge
pub struct Constraint {
    /// Constraint name
    pub name: String,
    /// Constraint kind
    pub kind: ConstraintKind,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("name", &self.name)
            .field("kind", &"<constraint>")
            .finish()
    }
}

/// Types of constraints
pub enum ConstraintKind {
    /// Entry must have minimum confidence
    MinConfidence { entry_id: String, min: f64 },
    /// Entries must exist
    Required { entry_ids: Vec<String> },
    /// At most one entry can exist
    Exclusive { entry_ids: Vec<String> },
    /// Custom check function
    Custom { check: fn(&KnowledgeBase) -> bool },
}

/// Domain information
#[derive(Debug, Clone)]
pub struct DomainInfo {
    /// Domain name
    pub name: String,
    /// Domain description
    pub description: String,
    /// Parent ontology
    pub ontology: Option<String>,
}

/// A knowledge entry (fact or belief)
#[derive(Debug, Clone)]
pub enum KnowledgeEntry {
    /// A fact
    Fact(Fact),
    /// A belief
    Belief(Belief),
}

impl KnowledgeEntry {
    /// Get the content
    pub fn content(&self) -> &str {
        match self {
            KnowledgeEntry::Fact(f) => &f.content,
            KnowledgeEntry::Belief(b) => &b.content,
        }
    }

    /// Get the confidence (1.0 for facts)
    pub fn confidence(&self) -> f64 {
        match self {
            KnowledgeEntry::Fact(_) => 1.0,
            KnowledgeEntry::Belief(b) => b.confidence,
        }
    }

    /// Get the domain
    pub fn domain(&self) -> Option<&str> {
        match self {
            KnowledgeEntry::Fact(f) => f.domain.as_deref(),
            KnowledgeEntry::Belief(b) => b.domain.as_deref(),
        }
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Entry ID
    pub id: String,
    /// The entry
    pub entry: KnowledgeEntry,
    /// Relevance score
    pub relevance: f64,
}

/// Record of a revision
#[derive(Debug, Clone)]
pub struct RevisionRecord {
    /// Entry ID
    pub id: String,
    /// Old confidence
    pub old_confidence: f64,
    /// New confidence
    pub new_confidence: f64,
    /// Revision number
    pub revision_number: u32,
}

/// Constraint violation
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// Constraint name
    pub constraint: String,
    /// Violation message
    pub message: String,
}

/// Knowledge base statistics
#[derive(Debug, Clone)]
pub struct KnowledgeStats {
    /// Number of facts
    pub fact_count: usize,
    /// Number of beliefs
    pub belief_count: usize,
    /// Number of constraints
    pub constraint_count: usize,
    /// Number of domains
    pub domain_count: usize,
    /// Average belief confidence
    pub average_belief_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_fact() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact("scaffold", Fact::new("A scaffold is a material entity"));

        let fact = kb.get_fact("scaffold").unwrap();
        assert!(fact.content.contains("scaffold"));
    }

    #[test]
    fn test_add_and_get_belief() {
        let mut kb = KnowledgeBase::new();
        kb.add_belief(
            "hypothesis",
            Belief::new("Scaffolds improve regeneration", 0.75),
        );

        let belief = kb.get_belief("hypothesis").unwrap();
        assert!((belief.confidence - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_search() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact(
            "f1",
            Fact::new("Scaffolds are porous structures").in_domain("biomaterials"),
        );
        kb.add_belief(
            "b1",
            Belief::new("Scaffolds support cell growth", 0.8).in_domain("biomaterials"),
        );
        kb.add_belief("b2", Belief::new("Something unrelated", 0.9));

        let results = kb.search("scaffold", 0.5);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_revise_belief() {
        let mut kb = KnowledgeBase::new();
        kb.add_belief("hyp", Belief::new("Initial hypothesis", 0.5));

        let record = kb.revise_belief("hyp", 0.8, "new experiment").unwrap();
        assert!((record.old_confidence - 0.5).abs() < f64::EPSILON);
        assert!((record.new_confidence - 0.8).abs() < f64::EPSILON);

        let belief = kb.get_belief("hyp").unwrap();
        assert!((belief.confidence - 0.8).abs() < f64::EPSILON);
        assert_eq!(belief.revision_count, 1);
    }

    #[test]
    fn test_promote_to_fact() {
        let mut kb = KnowledgeBase::new();
        kb.add_belief("candidate", Belief::new("High confidence belief", 0.95));

        let promoted = kb.promote_to_fact("candidate", 0.9).unwrap();
        assert!(promoted);

        assert!(kb.get_fact("candidate").is_some());
        assert!(kb.get_belief("candidate").is_none());
    }

    #[test]
    fn test_constraint_violation() {
        let mut kb = KnowledgeBase::new();
        kb.add_belief("low", Belief::new("Low confidence", 0.3));

        kb.add_constraint(Constraint {
            name: "min_conf".to_string(),
            kind: ConstraintKind::MinConfidence {
                entry_id: "low".to_string(),
                min: 0.5,
            },
        });

        let violations = kb.check_constraints();
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("below minimum"));
    }

    #[test]
    fn test_stats() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact("f1", Fact::new("Fact 1"));
        kb.add_fact("f2", Fact::new("Fact 2"));
        kb.add_belief("b1", Belief::new("Belief 1", 0.8));

        let stats = kb.stats();
        assert_eq!(stats.fact_count, 2);
        assert_eq!(stats.belief_count, 1);
    }
}
