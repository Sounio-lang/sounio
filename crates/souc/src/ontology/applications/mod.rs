// Sounio Compiler - Practical Ontology Applications
// Knowledge base completion, concept drift detection, alignment, and advanced conformal methods

pub mod alignment;
pub mod completion;
pub mod drift;
pub mod transduction;

pub use alignment::{AlignmentMapping, FederatedOntologyAligner};
pub use completion::{CompletionCandidate, KnowledgeBaseCompletion};
pub use drift::{ConceptDriftDetector, DriftAlert};
pub use transduction::{EnsembleConformal, FullConformalTransduction};
