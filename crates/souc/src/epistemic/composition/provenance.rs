//! Provenance Tracking for Epistemic Composition
//!
//! Tracks the complete derivation history of epistemic values,
//! including sources, transformations, and combination operations.
//!
//! # Provenance Tree Structure
//!
//! ```text
//! ProvenanceNode::Derived {
//!     operation: "tensor"
//!     children: [
//!         ProvenanceNode::Source { doi: "10.1234/..." },
//!         ProvenanceNode::Source { measurement: "TDM_001" }
//!     ]
//! }
//! ```

use std::fmt;

/// Source information for knowledge origin
#[derive(Debug, Clone, PartialEq)]
pub enum SourceInfo {
    /// From published literature
    Literature {
        /// Digital Object Identifier
        doi: String,
        /// Citation text
        citation: Option<String>,
    },

    /// From experimental measurement
    Measurement {
        /// Measurement identifier
        id: String,
        /// Timestamp of measurement
        timestamp: Option<chrono::DateTime<chrono::Utc>>,
        /// Instrument or method used
        method: Option<String>,
    },

    /// From computational inference
    Inference {
        /// Inference method name
        method: String,
        /// Model version
        version: Option<String>,
    },

    /// From LLM generation
    LLMGenerated {
        /// Model identifier
        model: String,
        /// Hash of the prompt used
        prompt_hash: u64,
        /// Confidence from linguistic markers
        linguistic_confidence: f64,
    },

    /// Primitive/axiomatic value
    Primitive,

    /// User assertion (explicit declaration)
    UserAssertion {
        /// Who made the assertion
        author: Option<String>,
        /// When it was made
        timestamp: Option<chrono::DateTime<chrono::Utc>>,
    },

    /// External API or database
    External {
        /// API or database name
        source: String,
        /// Query or endpoint
        query: String,
        /// Response timestamp
        timestamp: Option<chrono::DateTime<chrono::Utc>>,
    },
}

impl SourceInfo {
    /// Create a literature source from DOI
    pub fn from_doi(doi: impl Into<String>) -> Self {
        SourceInfo::Literature {
            doi: doi.into(),
            citation: None,
        }
    }

    /// Create a measurement source
    pub fn from_measurement(id: impl Into<String>) -> Self {
        SourceInfo::Measurement {
            id: id.into(),
            timestamp: Some(chrono::Utc::now()),
            method: None,
        }
    }

    /// Create an inference source
    pub fn from_inference(method: impl Into<String>) -> Self {
        SourceInfo::Inference {
            method: method.into(),
            version: None,
        }
    }

    /// Create an LLM source
    pub fn from_llm(model: impl Into<String>, prompt_hash: u64, confidence: f64) -> Self {
        SourceInfo::LLMGenerated {
            model: model.into(),
            prompt_hash,
            linguistic_confidence: confidence,
        }
    }

    /// Get a short description
    pub fn short_description(&self) -> String {
        match self {
            SourceInfo::Literature { doi, .. } => format!("doi:{}", doi),
            SourceInfo::Measurement { id, .. } => format!("meas:{}", id),
            SourceInfo::Inference { method, .. } => format!("inf:{}", method),
            SourceInfo::LLMGenerated { model, .. } => format!("llm:{}", model),
            SourceInfo::Primitive => "primitive".to_string(),
            SourceInfo::UserAssertion { author, .. } => {
                format!("user:{}", author.as_deref().unwrap_or("anonymous"))
            }
            SourceInfo::External { source, .. } => format!("ext:{}", source),
        }
    }
}

impl fmt::Display for SourceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_description())
    }
}

/// A node in the provenance tree
#[derive(Debug, Clone)]
pub enum ProvenanceNode {
    /// Leaf: Original source
    Source(SourceInfo),

    /// Internal node: Derived from operation
    Derived {
        /// Operation that produced this
        operation: String,
        /// Child provenances
        children: Vec<ProvenanceNode>,
        /// Additional metadata
        metadata: Option<String>,
    },

    /// Updated version of existing knowledge
    Updated {
        /// Original provenance
        original: Box<ProvenanceNode>,
        /// Update operation
        update_type: String,
        /// Timestamp of update
        timestamp: chrono::DateTime<chrono::Utc>,
    },
}

impl ProvenanceNode {
    /// Create a source node
    pub fn source(info: SourceInfo) -> Self {
        ProvenanceNode::Source(info)
    }

    /// Create a primitive source
    pub fn primitive() -> Self {
        ProvenanceNode::Source(SourceInfo::Primitive)
    }

    /// Create a derived node from an operation
    pub fn derived(operation: impl Into<String>, children: Vec<ProvenanceNode>) -> Self {
        ProvenanceNode::Derived {
            operation: operation.into(),
            children,
            metadata: None,
        }
    }

    /// Create a derived node with metadata
    pub fn derived_with_metadata(
        operation: impl Into<String>,
        children: Vec<ProvenanceNode>,
        metadata: impl Into<String>,
    ) -> Self {
        ProvenanceNode::Derived {
            operation: operation.into(),
            children,
            metadata: Some(metadata.into()),
        }
    }

    /// Create an updated node
    pub fn updated(original: ProvenanceNode, update_type: impl Into<String>) -> Self {
        ProvenanceNode::Updated {
            original: Box::new(original),
            update_type: update_type.into(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get all leaf sources
    pub fn sources(&self) -> Vec<&SourceInfo> {
        match self {
            ProvenanceNode::Source(info) => vec![info],
            ProvenanceNode::Derived { children, .. } => {
                children.iter().flat_map(|c| c.sources()).collect()
            }
            ProvenanceNode::Updated { original, .. } => original.sources(),
        }
    }

    /// Get the depth of the provenance tree
    pub fn depth(&self) -> usize {
        match self {
            ProvenanceNode::Source(_) => 1,
            ProvenanceNode::Derived { children, .. } => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
            ProvenanceNode::Updated { original, .. } => 1 + original.depth(),
        }
    }

    /// Get the number of source nodes
    pub fn source_count(&self) -> usize {
        self.sources().len()
    }

    /// Check if any source is from LLM
    pub fn has_llm_source(&self) -> bool {
        self.sources()
            .iter()
            .any(|s| matches!(s, SourceInfo::LLMGenerated { .. }))
    }

    /// Get the root operation name
    pub fn root_operation(&self) -> Option<&str> {
        match self {
            ProvenanceNode::Source(_) => None,
            ProvenanceNode::Derived { operation, .. } => Some(operation),
            ProvenanceNode::Updated { update_type, .. } => Some(update_type),
        }
    }
}

impl fmt::Display for ProvenanceNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvenanceNode::Source(info) => write!(f, "{}", info),
            ProvenanceNode::Derived {
                operation,
                children,
                ..
            } => {
                let child_strs: Vec<_> = children.iter().map(|c| c.to_string()).collect();
                write!(f, "{}({})", operation, child_strs.join(", "))
            }
            ProvenanceNode::Updated {
                original,
                update_type,
                ..
            } => {
                write!(f, "{}({})", update_type, original)
            }
        }
    }
}

/// A chain of derivations for linear provenance
#[derive(Debug, Clone, Default)]
pub struct DerivationChain {
    /// Steps in the derivation
    steps: Vec<DerivationStep>,
}

/// A single step in a derivation chain
#[derive(Debug, Clone)]
pub struct DerivationStep {
    /// Operation performed
    pub operation: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Confidence before this step
    pub confidence_before: f64,
    /// Confidence after this step
    pub confidence_after: f64,
}

impl DerivationChain {
    /// Create an empty chain
    pub fn new() -> Self {
        DerivationChain { steps: Vec::new() }
    }

    /// Add a step to the chain
    pub fn add_step(
        &mut self,
        operation: impl Into<String>,
        confidence_before: f64,
        confidence_after: f64,
    ) {
        self.steps.push(DerivationStep {
            operation: operation.into(),
            timestamp: chrono::Utc::now(),
            confidence_before,
            confidence_after,
        });
    }

    /// Get the number of steps
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get all steps
    pub fn steps(&self) -> &[DerivationStep] {
        &self.steps
    }

    /// Get the total confidence change
    pub fn total_confidence_change(&self) -> f64 {
        if self.steps.is_empty() {
            0.0
        } else {
            let first = self.steps.first().unwrap().confidence_before;
            let last = self.steps.last().unwrap().confidence_after;
            last - first
        }
    }
}

impl fmt::Display for DerivationChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops: Vec<_> = self.steps.iter().map(|s| s.operation.as_str()).collect();
        write!(f, "[{}]", ops.join(" → "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_info() {
        let lit = SourceInfo::from_doi("10.1234/example");
        assert!(lit.short_description().contains("10.1234"));

        let meas = SourceInfo::from_measurement("TDM_001");
        assert!(meas.short_description().contains("TDM_001"));
    }

    #[test]
    fn test_provenance_tree() {
        let source1 = ProvenanceNode::source(SourceInfo::from_doi("10.1234/a"));
        let source2 = ProvenanceNode::source(SourceInfo::from_measurement("M1"));

        let derived = ProvenanceNode::derived("tensor", vec![source1, source2]);

        assert_eq!(derived.source_count(), 2);
        assert_eq!(derived.depth(), 2);
        assert_eq!(derived.root_operation(), Some("tensor"));
    }

    #[test]
    fn test_derivation_chain() {
        let mut chain = DerivationChain::new();
        chain.add_step("tensor", 0.9, 0.81);
        chain.add_step("condition", 0.81, 0.85);

        assert_eq!(chain.len(), 2);
        assert!((chain.total_confidence_change() - (-0.05)).abs() < 0.01);
    }

    #[test]
    fn test_llm_source_detection() {
        let llm_source = ProvenanceNode::source(SourceInfo::from_llm("gpt-4", 12345, 0.8));
        assert!(llm_source.has_llm_source());

        let lit_source = ProvenanceNode::source(SourceInfo::from_doi("10.1234/x"));
        assert!(!lit_source.has_llm_source());
    }
}
