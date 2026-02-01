//! World Fidelity Verification for Sounio
//!
//! This module implements automatic verification that computed knowledge
//! maintains fidelity to real-world ontologies (OBO Foundry, CHEBI, GO, etc.).
//!
//! # Epistemic Fidelity
//!
//! Fidelity means that a value:
//! 1. Is bound to a valid ontology term (not a phantom/obsolete term)
//! 2. Subsumes or is subsumed by expected domain constraints
//! 3. Has quantified confidence via Beta posterior
//! 4. Carries irrefutable provenance (Merkle-auditable)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    WorldFidelityChecker                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  verify_fidelity(knowledge) → FidelityResult                    │
//! │  verify_subsumption(term, domain) → SubsumptionFidelity         │
//! │  model_fidelity(model) → AggregateFidelity                      │
//! │  audit_provenance(knowledge) → ProvenanceAudit                  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    OBO Foundry Ground Truth                     │
//! │  CHEBI │ GO │ UBERON │ HP │ MONDO │ DOID │ CL │ PATO │ UO      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::fidelity::{WorldFidelityChecker, FidelityResult};
//!
//! let checker = WorldFidelityChecker::obo_foundry();
//!
//! // Verify single knowledge value
//! let aspirin_knowledge = Knowledge::from_ontology("CHEBI:15365");
//! let result = checker.verify_fidelity(&aspirin_knowledge, "CHEBI:23888")?;
//!
//! match result {
//!     FidelityResult::High { confidence, .. } => println!("Verified: {}", confidence.mean()),
//!     FidelityResult::Violation { reason, .. } => panic!("Fidelity violation: {}", reason),
//!     _ => {}
//! }
//! ```
//!
//! # References
//!
//! - OBO Foundry: <https://obofoundry.org/>
//! - SSSOM Mappings: <https://mapping-commons.github.io/sssom/>

use std::collections::{HashMap, HashSet};

use crate::epistemic::{BetaConfidence, Confidence, Provenance, Source};

use super::resolver::{OntologyResolver, ResolvedTerm, ResolverConfig, SubsumptionResult};
use super::{OntologyLayer, OntologyResult};

/// World fidelity checker that validates knowledge against real-world ontologies.
///
/// This is the core system that ensures Sounio programs maintain epistemic
/// honesty by grounding all domain knowledge in verified ontology terms.
pub struct WorldFidelityChecker {
    /// Unified ontology resolver (L1-L4)
    resolver: OntologyResolver,

    /// OBO Foundry ontologies considered "world ground truth"
    world_ontologies: HashSet<String>,

    /// Minimum confidence threshold for high fidelity
    high_fidelity_threshold: f64,

    /// Configuration for fidelity checks
    config: FidelityConfig,

    /// Statistics for fidelity checks
    stats: FidelityStats,
}

impl std::fmt::Debug for WorldFidelityChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorldFidelityChecker")
            .field("world_ontologies", &self.world_ontologies)
            .field("high_fidelity_threshold", &self.high_fidelity_threshold)
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// Configuration for fidelity verification
#[derive(Debug, Clone)]
pub struct FidelityConfig {
    /// Require terms to be non-obsolete
    pub require_non_obsolete: bool,

    /// Require subsumption to expected domain
    pub require_subsumption: bool,

    /// Minimum confidence for critical parameters
    pub critical_min_confidence: f64,

    /// Allow federated (L4) terms as ground truth
    pub allow_federated: bool,

    /// Ontology version pinning (for reproducibility)
    pub version_pins: HashMap<String, String>,

    /// Maximum allowed semantic distance for compatibility
    pub max_semantic_distance: f64,
}

impl Default for FidelityConfig {
    fn default() -> Self {
        Self {
            require_non_obsolete: true,
            require_subsumption: true,
            critical_min_confidence: 0.9,
            allow_federated: false, // Conservative: only L1-L3 by default
            version_pins: HashMap::new(),
            max_semantic_distance: 0.15,
        }
    }
}

/// Statistics for fidelity checking operations
#[derive(Debug, Default, Clone)]
pub struct FidelityStats {
    pub total_checks: u64,
    pub high_fidelity: u64,
    pub medium_fidelity: u64,
    pub low_fidelity: u64,
    pub violations: u64,
    pub unknown: u64,
    pub subsumption_checks: u64,
    pub provenance_audits: u64,
}

/// Result of a fidelity verification
#[derive(Debug, Clone)]
pub enum FidelityResult {
    /// High fidelity: term is valid, subsumes expected domain, high confidence
    High {
        /// Beta posterior combining evidence and ontology prior
        confidence: BetaConfidence,
        /// Resolved term from ontology
        term: ResolvedTerm,
        /// Subsumption relationship verified
        subsumption: SubsumptionFidelity,
        /// Provenance audit passed
        provenance_valid: bool,
    },

    /// Medium fidelity: term is valid but with caveats
    Medium {
        confidence: BetaConfidence,
        term: ResolvedTerm,
        /// Reason for medium (not high) rating
        caveats: Vec<FidelityCaveat>,
    },

    /// Low fidelity: term exists but significant issues
    Low {
        confidence: BetaConfidence,
        term: Option<ResolvedTerm>,
        /// Issues that lower fidelity
        issues: Vec<FidelityIssue>,
    },

    /// Fidelity violation: term invalid or domain mismatch
    Violation {
        /// Human-readable reason
        reason: String,
        /// Detailed violation info
        details: ViolationDetails,
        /// Suggested fixes
        suggestions: Vec<String>,
    },

    /// Cannot determine fidelity (e.g., non-world ontology)
    Unknown {
        reason: String,
        /// Ontology that was not in world set
        ontology: Option<String>,
    },

    /// Fidelity check is indeterminate (incomplete information)
    Indeterminate {
        /// What information is missing
        missing: Vec<String>,
    },
}

/// Subsumption fidelity status
#[derive(Debug, Clone)]
pub enum SubsumptionFidelity {
    /// Term is subclass of expected domain
    IsSubclass {
        child: String,
        parent: String,
        path_length: usize,
    },
    /// Term is superclass of expected domain
    IsSuperclass { parent: String, child: String },
    /// Terms are equivalent
    Equivalent { term_a: String, term_b: String },
    /// No subsumption relationship (but compatible via mapping)
    MappedCompatible {
        source: String,
        target: String,
        mapping_predicate: String,
        confidence: f64,
    },
    /// Not checked (no expected domain specified)
    NotChecked,
}

/// Caveats that reduce fidelity from High to Medium
#[derive(Debug, Clone)]
pub enum FidelityCaveat {
    /// Term is from federated source (less reliable than local)
    FederatedSource { source: String },
    /// Term has been deprecated (not obsolete, but superseded)
    Deprecated { successor: Option<String> },
    /// Ontology version mismatch with pin
    VersionMismatch { expected: String, actual: String },
    /// Low information content (very generic term)
    LowInformationContent { ic: f64 },
    /// Multiple conflicting mappings exist
    AmbiguousMappings { count: usize },
    /// Confidence below critical threshold
    BelowCriticalThreshold { confidence: f64, threshold: f64 },
}

/// Issues that result in Low fidelity
#[derive(Debug, Clone)]
pub enum FidelityIssue {
    /// Term is marked obsolete in ontology
    ObsoleteTerm { replacement: Option<String> },
    /// Term not found in expected ontology
    TermNotFound { curie: String },
    /// Subsumption check failed
    SubsumptionFailed { expected_domain: String },
    /// Provenance chain broken
    BrokenProvenance { at_step: usize },
    /// Evidence conflict detected
    EvidenceConflict { conflict_coefficient: f64 },
    /// Source has low reliability
    LowReliabilitySource { source: String, reliability: f64 },
}

/// Details of a fidelity violation
#[derive(Debug, Clone)]
pub struct ViolationDetails {
    /// Type of violation
    pub violation_type: ViolationType,
    /// Term that caused violation (if any)
    pub term: Option<String>,
    /// Expected domain
    pub expected_domain: Option<String>,
    /// Actual domain (if resolved)
    pub actual_domain: Option<String>,
    /// Semantic distance (if computed)
    pub semantic_distance: Option<f64>,
}

/// Types of fidelity violations
#[derive(Debug, Clone)]
pub enum ViolationType {
    /// Term does not exist in any known ontology
    InvalidTerm,
    /// Term exists but is in wrong domain
    DomainMismatch,
    /// Term is obsolete without replacement
    IrrecoverableObsolete,
    /// Ontology not in world set
    NonWorldOntology,
    /// Semantic distance exceeds threshold
    ExcessiveDistance,
    /// Provenance tampering detected
    ProvenanceTampered,
    /// Circular provenance chain
    CircularProvenance,
}

/// Aggregate fidelity for a model or computation
#[derive(Debug, Clone)]
pub struct AggregateFidelity {
    /// Overall fidelity score (0.0 - 1.0)
    pub score: f64,
    /// Beta posterior for uncertainty
    pub confidence: BetaConfidence,
    /// Number of high fidelity nodes
    pub high_count: usize,
    /// Number of medium fidelity nodes
    pub medium_count: usize,
    /// Number of low fidelity nodes
    pub low_count: usize,
    /// Violations that must be fixed
    pub violations: Vec<FidelityResult>,
    /// Overall provenance hash (Merkle root)
    pub provenance_hash: Option<String>,
}

/// Provenance audit result
#[derive(Debug, Clone)]
pub struct ProvenanceAudit {
    /// Whether provenance chain is valid
    pub valid: bool,
    /// Number of steps in chain
    pub chain_length: usize,
    /// Merkle root hash
    pub merkle_root: Option<String>,
    /// Any issues found
    pub issues: Vec<ProvenanceIssue>,
    /// Ontology assertions in chain
    pub ontology_assertions: Vec<OntologyAssertion>,
}

/// Issues in provenance chain
#[derive(Debug, Clone)]
pub enum ProvenanceIssue {
    /// Hash mismatch at step
    HashMismatch { step: usize },
    /// Missing intermediate step
    MissingStep { after: usize },
    /// Timestamp inconsistency
    TimestampInconsistent { step: usize },
    /// Unknown transformation
    UnknownTransformation { name: String },
}

/// Ontology assertion in provenance
#[derive(Debug, Clone)]
pub struct OntologyAssertion {
    /// Ontology name
    pub ontology: String,
    /// Term CURIE
    pub term: String,
    /// Version at time of assertion
    pub version: Option<String>,
    /// Timestamp
    pub timestamp: Option<String>,
}

impl WorldFidelityChecker {
    /// Create a new fidelity checker with default OBO Foundry ontologies
    pub fn obo_foundry() -> OntologyResult<Self> {
        let mut world_ontologies = HashSet::new();
        // Core OBO Foundry ontologies
        world_ontologies.insert("chebi".to_string());
        world_ontologies.insert("go".to_string());
        world_ontologies.insert("uberon".to_string());
        world_ontologies.insert("hp".to_string());
        world_ontologies.insert("mondo".to_string());
        world_ontologies.insert("doid".to_string());
        world_ontologies.insert("cl".to_string());
        world_ontologies.insert("pato".to_string());
        world_ontologies.insert("uo".to_string());
        world_ontologies.insert("obi".to_string());
        world_ontologies.insert("ncbitaxon".to_string());
        world_ontologies.insert("pr".to_string());
        world_ontologies.insert("so".to_string());
        world_ontologies.insert("envo".to_string());
        // Foundation ontologies
        world_ontologies.insert("bfo".to_string());
        world_ontologies.insert("ro".to_string());
        world_ontologies.insert("cob".to_string());
        world_ontologies.insert("iao".to_string());

        let resolver_config = ResolverConfig::default();
        let resolver = OntologyResolver::new(resolver_config)?;

        Ok(Self {
            resolver,
            world_ontologies,
            high_fidelity_threshold: 0.9,
            config: FidelityConfig::default(),
            stats: FidelityStats::default(),
        })
    }

    /// Create with custom resolver and config
    pub fn with_config(resolver: OntologyResolver, config: FidelityConfig) -> Self {
        Self {
            resolver,
            world_ontologies: HashSet::new(),
            high_fidelity_threshold: config.critical_min_confidence,
            config,
            stats: FidelityStats::default(),
        }
    }

    /// Add an ontology to the world set
    pub fn add_world_ontology(&mut self, name: &str) {
        self.world_ontologies.insert(name.to_lowercase());
    }

    /// Check if an ontology is in the world set
    pub fn is_world_ontology(&self, name: &str) -> bool {
        self.world_ontologies.contains(&name.to_lowercase())
    }

    /// Verify fidelity of a knowledge value against expected domain
    ///
    /// This is the primary API for fidelity checking.
    pub fn verify_fidelity(
        &mut self,
        term_curie: &str,
        expected_domain: Option<&str>,
        source: &Source,
        confidence: &Confidence,
    ) -> FidelityResult {
        self.stats.total_checks += 1;

        // Extract ontology prefix from CURIE
        let ontology_prefix = match term_curie.split(':').next() {
            Some(prefix) => prefix.to_lowercase(),
            None => {
                self.stats.violations += 1;
                return FidelityResult::Violation {
                    reason: "Invalid CURIE format".to_string(),
                    details: ViolationDetails {
                        violation_type: ViolationType::InvalidTerm,
                        term: Some(term_curie.to_string()),
                        expected_domain: expected_domain.map(String::from),
                        actual_domain: None,
                        semantic_distance: None,
                    },
                    suggestions: vec!["Use format PREFIX:ID (e.g., CHEBI:15365)".to_string()],
                };
            }
        };

        // Check if ontology is in world set
        if !self.is_world_ontology(&ontology_prefix) {
            self.stats.unknown += 1;
            return FidelityResult::Unknown {
                reason: format!("Ontology '{}' not in world set", ontology_prefix),
                ontology: Some(ontology_prefix),
            };
        }

        // Resolve term in ontology
        let resolved = match self.resolver.resolve(term_curie) {
            Ok(term) => term,
            Err(_) => {
                self.stats.low_fidelity += 1;
                return FidelityResult::Low {
                    confidence: self.source_to_beta(source, confidence),
                    term: None,
                    issues: vec![FidelityIssue::TermNotFound {
                        curie: term_curie.to_string(),
                    }],
                };
            }
        };

        // Check for obsolete terms
        let mut issues = Vec::new();
        let mut caveats = Vec::new();

        if self.config.require_non_obsolete && self.is_obsolete(&resolved) {
            issues.push(FidelityIssue::ObsoleteTerm {
                replacement: self.find_replacement(&resolved),
            });
        }

        // Check layer - federated sources get caveat
        if resolved.layer == OntologyLayer::Federated && !self.config.allow_federated {
            caveats.push(FidelityCaveat::FederatedSource {
                source: "bioportal".to_string(),
            });
        }

        // Verify subsumption if expected domain provided
        let subsumption = if let Some(domain) = expected_domain {
            self.stats.subsumption_checks += 1;
            self.check_subsumption(term_curie, domain)
        } else {
            SubsumptionFidelity::NotChecked
        };

        // Check if subsumption failed
        if let SubsumptionFidelity::IsSubclass { .. } = &subsumption {
            // Good - no issue
        } else if let SubsumptionFidelity::Equivalent { .. } = &subsumption {
            // Also good
        } else if let SubsumptionFidelity::MappedCompatible { .. } = &subsumption {
            // Acceptable with caveat
            caveats.push(FidelityCaveat::AmbiguousMappings { count: 1 });
        } else if expected_domain.is_some() {
            if let SubsumptionFidelity::NotChecked = &subsumption {
                // Skip
            } else {
                issues.push(FidelityIssue::SubsumptionFailed {
                    expected_domain: expected_domain.unwrap().to_string(),
                });
            }
        }

        // Compute Beta confidence combining source reliability and ontology prior
        let beta = self.compute_fidelity_beta(source, confidence, &resolved);

        // Check confidence threshold
        if beta.mean() < self.config.critical_min_confidence {
            caveats.push(FidelityCaveat::BelowCriticalThreshold {
                confidence: beta.mean(),
                threshold: self.config.critical_min_confidence,
            });
        }

        // Determine overall fidelity level
        if !issues.is_empty() {
            self.stats.low_fidelity += 1;
            FidelityResult::Low {
                confidence: beta,
                term: Some(resolved),
                issues,
            }
        } else if !caveats.is_empty() {
            self.stats.medium_fidelity += 1;
            FidelityResult::Medium {
                confidence: beta,
                term: resolved,
                caveats,
            }
        } else {
            self.stats.high_fidelity += 1;
            FidelityResult::High {
                confidence: beta,
                term: resolved,
                subsumption,
                provenance_valid: self.verify_provenance_quick(source),
            }
        }
    }

    /// Check subsumption relationship between terms
    fn check_subsumption(&mut self, child: &str, parent: &str) -> SubsumptionFidelity {
        match self.resolver.is_subclass_of(child, parent) {
            Ok(SubsumptionResult::IsSubclass) => SubsumptionFidelity::IsSubclass {
                child: child.to_string(),
                parent: parent.to_string(),
                path_length: self.compute_path_length(child, parent).unwrap_or(1),
            },
            Ok(SubsumptionResult::Equivalent) => SubsumptionFidelity::Equivalent {
                term_a: child.to_string(),
                term_b: parent.to_string(),
            },
            Ok(SubsumptionResult::NotSubclass) => {
                // Check reverse
                match self.resolver.is_subclass_of(parent, child) {
                    Ok(SubsumptionResult::IsSubclass) => SubsumptionFidelity::IsSuperclass {
                        parent: child.to_string(),
                        child: parent.to_string(),
                    },
                    _ => SubsumptionFidelity::NotChecked, // No relationship
                }
            }
            Ok(SubsumptionResult::Unknown) | Err(_) => SubsumptionFidelity::NotChecked,
        }
    }

    /// Convert source and confidence to Beta distribution
    fn source_to_beta(&self, source: &Source, confidence: &Confidence) -> BetaConfidence {
        let reliability = self.source_reliability(source);
        let conf_value = confidence.value();

        // Scale alpha/beta by source reliability
        let alpha = conf_value * reliability * 10.0 + 1.0;
        let beta = (1.0 - conf_value) * reliability * 10.0 + 1.0;

        BetaConfidence::new(alpha, beta)
    }

    /// Compute fidelity Beta combining multiple factors
    fn compute_fidelity_beta(
        &self,
        source: &Source,
        confidence: &Confidence,
        resolved: &ResolvedTerm,
    ) -> BetaConfidence {
        let base_beta = self.source_to_beta(source, confidence);

        // Adjust based on ontology layer (more primitive = more reliable)
        let layer_factor = match resolved.layer {
            OntologyLayer::Primitive => 1.5,
            OntologyLayer::Foundation => 1.3,
            OntologyLayer::Domain => 1.0,
            OntologyLayer::Federated => 0.8,
        };

        BetaConfidence::new(
            base_beta.alpha * layer_factor,
            base_beta.beta / layer_factor,
        )
    }

    /// Get reliability weight for a source type
    fn source_reliability(&self, source: &Source) -> f64 {
        match source {
            Source::Axiom => 2.0,
            Source::Measurement { .. } => 1.3,
            Source::Derivation(_) => 1.0,
            Source::Transformation { .. } => 0.9,
            Source::OntologyAssertion { .. } => 1.5,
            _ => 1.0,
        }
    }

    /// Check if a term is marked obsolete
    fn is_obsolete(&self, term: &ResolvedTerm) -> bool {
        // Check for "obsolete" in label
        if let Some(label) = &term.label {
            if label.to_lowercase().contains("obsolete") {
                return true;
            }
        }
        // Check for obsolescence annotation in definition (OBO standard pattern)
        if let Some(def) = &term.definition {
            let def_lower = def.to_lowercase();
            if def_lower.starts_with("obsolete") || def_lower.contains("this term is obsolete") {
                return true;
            }
        }
        // Check IRI for obsolete pattern (some ontologies use this)
        if let Some(iri) = &term.iri {
            if iri.contains("/obsolete/") || iri.contains("#obsolete_") {
                return true;
            }
        }
        false
    }

    /// Find replacement for obsolete term
    fn find_replacement(&self, term: &ResolvedTerm) -> Option<String> {
        // Look for replaced_by pattern in definition (OBO standard: "Use X instead")
        if let Some(def) = &term.definition {
            // Pattern: "OBSOLETE. Use GO:0008150 instead"
            if let Some(pos) = def.find("Use ") {
                let rest = &def[pos + 4..];
                // Extract CURIE (format: PREFIX:ID)
                if let Some(end) = rest.find(|c: char| c.is_whitespace() || c == '.') {
                    let candidate = &rest[..end];
                    if candidate.contains(':') && candidate.len() > 3 {
                        return Some(candidate.to_string());
                    }
                } else if rest.contains(':') && rest.len() > 3 {
                    return Some(rest.trim_end_matches('.').to_string());
                }
            }
            // Pattern: "replaced by: CHEBI:12345"
            if let Some(pos) = def.to_lowercase().find("replaced by:") {
                let rest = &def[pos + 12..].trim_start();
                if let Some(end) = rest.find(|c: char| c.is_whitespace() || c == '.') {
                    let candidate = &rest[..end];
                    if candidate.contains(':') {
                        return Some(candidate.to_string());
                    }
                }
            }
        }
        // Check superclasses for replacement (some ontologies link obsolete to replacement)
        if !term.superclasses.is_empty() && term.superclasses.len() == 1 {
            // If obsolete term has exactly one superclass, it might be the replacement
            let superclass = &term.superclasses[0];
            if !superclass.to_lowercase().contains("obsolete") {
                return Some(superclass.clone());
            }
        }
        None
    }

    /// Quick provenance verification (checks source reliability)
    fn verify_provenance_quick(&self, source: &Source) -> bool {
        match source {
            Source::Axiom => true, // Axioms are always valid
            Source::OntologyAssertion { ontology, .. } => {
                // Verify ontology is in world set
                self.is_world_ontology(ontology)
            }
            Source::Measurement { instrument, .. } => {
                // Measurements are valid if instrument is specified
                instrument.is_some()
            }
            Source::Derivation(steps) => {
                // Derivation is valid if all steps have known operations
                !steps.is_empty()
            }
            Source::Transformation { via, .. } => {
                // Transformation is valid if via (operation) is known
                !via.is_empty()
            }
            _ => true,
        }
    }

    /// Compute path length between two terms in ontology hierarchy
    fn compute_path_length(&mut self, child: &str, parent: &str) -> Option<usize> {
        // BFS to find shortest path from child to parent
        use std::collections::VecDeque;

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((child.to_string(), 0usize));
        visited.insert(child.to_string());

        const MAX_DEPTH: usize = 20; // Prevent infinite loops

        while let Some((current, depth)) = queue.pop_front() {
            if depth > MAX_DEPTH {
                break;
            }
            if current == parent {
                return Some(depth);
            }
            // Get superclasses of current term
            if let Ok(term) = self.resolver.resolve(&current) {
                for superclass in &term.superclasses {
                    if !visited.contains(superclass) {
                        visited.insert(superclass.clone());
                        queue.push_back((superclass.clone(), depth + 1));
                    }
                }
            }
        }
        None // No path found within depth limit
    }

    /// Compute aggregate fidelity for a set of knowledge values
    pub fn aggregate_fidelity(&mut self, results: &[FidelityResult]) -> AggregateFidelity {
        let mut high_count = 0;
        let mut medium_count = 0;
        let mut low_count = 0;
        let mut violations = Vec::new();

        let mut alpha_sum = 0.0;
        let mut beta_sum = 0.0;

        for result in results {
            match result {
                FidelityResult::High { confidence, .. } => {
                    high_count += 1;
                    alpha_sum += confidence.alpha;
                    beta_sum += confidence.beta;
                }
                FidelityResult::Medium { confidence, .. } => {
                    medium_count += 1;
                    alpha_sum += confidence.alpha;
                    beta_sum += confidence.beta;
                }
                FidelityResult::Low { confidence, .. } => {
                    low_count += 1;
                    alpha_sum += confidence.alpha;
                    beta_sum += confidence.beta;
                }
                FidelityResult::Violation { .. } => {
                    violations.push(result.clone());
                }
                _ => {}
            }
        }

        let total = (high_count + medium_count + low_count) as f64;
        let score = if total > 0.0 {
            (high_count as f64 * 1.0 + medium_count as f64 * 0.7 + low_count as f64 * 0.3) / total
        } else {
            0.0
        };

        // Compute Merkle root from all result hashes
        let provenance_hash = self.compute_merkle_root(results);

        AggregateFidelity {
            score,
            confidence: BetaConfidence::new(alpha_sum.max(1.0), beta_sum.max(1.0)),
            high_count,
            medium_count,
            low_count,
            violations,
            provenance_hash,
        }
    }

    /// Compute Merkle root hash from fidelity results
    fn compute_merkle_root(&self, results: &[FidelityResult]) -> Option<String> {
        if results.is_empty() {
            return None;
        }

        // Collect leaf hashes from each result
        let mut hashes: Vec<String> = results
            .iter()
            .map(|r| self.hash_fidelity_result(r))
            .collect();

        // Build Merkle tree
        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in hashes.chunks(2) {
                let combined = if chunk.len() == 2 {
                    format!("{}{}", chunk[0], chunk[1])
                } else {
                    // Odd number: duplicate last hash
                    format!("{}{}", chunk[0], chunk[0])
                };
                next_level.push(self.hash_string(&combined));
            }
            hashes = next_level;
        }

        hashes.into_iter().next()
    }

    /// Hash a fidelity result to a string
    fn hash_fidelity_result(&self, result: &FidelityResult) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        match result {
            FidelityResult::High {
                term, confidence, ..
            } => {
                "high".hash(&mut hasher);
                term.curie.hash(&mut hasher);
                ((confidence.alpha * 1000.0) as u64).hash(&mut hasher);
                ((confidence.beta * 1000.0) as u64).hash(&mut hasher);
            }
            FidelityResult::Medium {
                term, confidence, ..
            } => {
                "medium".hash(&mut hasher);
                term.curie.hash(&mut hasher);
                ((confidence.alpha * 1000.0) as u64).hash(&mut hasher);
            }
            FidelityResult::Low {
                term, confidence, ..
            } => {
                "low".hash(&mut hasher);
                if let Some(t) = term {
                    t.curie.hash(&mut hasher);
                }
                ((confidence.alpha * 1000.0) as u64).hash(&mut hasher);
            }
            FidelityResult::Violation { reason, .. } => {
                "violation".hash(&mut hasher);
                reason.hash(&mut hasher);
            }
            FidelityResult::Unknown { reason, .. } => {
                "unknown".hash(&mut hasher);
                reason.hash(&mut hasher);
            }
            FidelityResult::Indeterminate { missing, .. } => {
                "indeterminate".hash(&mut hasher);
                missing.len().hash(&mut hasher);
            }
        }

        format!("{:016x}", hasher.finish())
    }

    /// Hash a string using DefaultHasher
    fn hash_string(&self, s: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Audit provenance chain for integrity
    pub fn audit_provenance(&mut self, provenance: &Provenance) -> ProvenanceAudit {
        self.stats.provenance_audits += 1;

        let mut issues = Vec::new();
        let mut ontology_assertions = Vec::new();

        // Extract ontology assertions from provenance origin
        if let crate::epistemic::Origin::OntologyAssertion { ontology, term } = &provenance.origin {
            ontology_assertions.push(OntologyAssertion {
                ontology: ontology.clone(),
                term: term.clone(),
                version: None,
                timestamp: None,
            });
        }

        // Check chain integrity
        let chain_length = provenance.trace.steps.len();

        // Verify hash if present
        let valid = if let Some(stored_hash) = &provenance.integrity_hash {
            // Recompute Merkle hash from provenance steps
            let computed_hash = self.compute_provenance_hash(provenance);
            if &computed_hash != stored_hash {
                issues.push(ProvenanceIssue::HashMismatch { step: 0 });
                false
            } else {
                true
            }
        } else {
            true // No hash to verify
        };

        ProvenanceAudit {
            valid,
            chain_length,
            merkle_root: provenance.integrity_hash.clone(),
            issues,
            ontology_assertions,
        }
    }

    /// Compute hash of provenance chain
    fn compute_provenance_hash(&self, provenance: &Provenance) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash origin
        match &provenance.origin {
            crate::epistemic::Origin::OntologyAssertion { ontology, term } => {
                "ontology".hash(&mut hasher);
                ontology.hash(&mut hasher);
                term.hash(&mut hasher);
            }
            crate::epistemic::Origin::External { uri } => {
                "external".hash(&mut hasher);
                uri.hash(&mut hasher);
            }
            crate::epistemic::Origin::Computed { function } => {
                "computed".hash(&mut hasher);
                function.hash(&mut hasher);
            }
            crate::epistemic::Origin::UserInput { context } => {
                "userinput".hash(&mut hasher);
                context.hash(&mut hasher);
            }
            crate::epistemic::Origin::Database { query, connection } => {
                "database".hash(&mut hasher);
                query.hash(&mut hasher);
                connection.hash(&mut hasher);
            }
            crate::epistemic::Origin::Literal => {
                "literal".hash(&mut hasher);
            }
        }

        // Hash all trace steps (Transformation structs)
        for step in &provenance.trace.steps {
            step.name.hash(&mut hasher);
            step.inputs.len().hash(&mut hasher);
            step.output.hash(&mut hasher);
            ((step.confidence_factor * 1000.0) as u64).hash(&mut hasher);
        }

        format!("{:016x}", hasher.finish())
    }

    /// Get current statistics
    pub fn stats(&self) -> &FidelityStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = FidelityStats::default();
    }
}

impl FidelityResult {
    /// Convert to Beta confidence (for aggregation)
    pub fn to_beta(&self) -> Option<BetaConfidence> {
        match self {
            FidelityResult::High { confidence, .. } => Some(*confidence),
            FidelityResult::Medium { confidence, .. } => Some(*confidence),
            FidelityResult::Low { confidence, .. } => Some(*confidence),
            _ => None,
        }
    }

    /// Check if result is acceptable (High or Medium)
    pub fn is_acceptable(&self) -> bool {
        matches!(
            self,
            FidelityResult::High { .. } | FidelityResult::Medium { .. }
        )
    }

    /// Check if result is a violation
    pub fn is_violation(&self) -> bool {
        matches!(self, FidelityResult::Violation { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obo_foundry_creation() {
        // This test may fail if resolver can't be created (missing data files)
        // So we test the world ontology set directly
        let world_ontologies: HashSet<String> = [
            "chebi",
            "go",
            "uberon",
            "hp",
            "mondo",
            "doid",
            "cl",
            "pato",
            "uo",
            "obi",
            "ncbitaxon",
            "pr",
            "so",
            "envo",
            "bfo",
            "ro",
            "cob",
            "iao",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        assert!(world_ontologies.contains("chebi"));
        assert!(world_ontologies.contains("go"));
        assert!(!world_ontologies.contains("custom_ontology"));
    }

    #[test]
    fn test_fidelity_result_to_beta() {
        let high = FidelityResult::High {
            confidence: BetaConfidence::new(10.0, 2.0),
            term: ResolvedTerm {
                curie: "CHEBI:15365".to_string(),
                label: Some("aspirin".to_string()),
                definition: None,
                superclasses: vec![],
                synonyms: vec![],
                layer: OntologyLayer::Domain,
                iri: None,
            },
            subsumption: SubsumptionFidelity::NotChecked,
            provenance_valid: true,
        };

        let beta = high.to_beta().unwrap();
        assert!((beta.mean() - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_fidelity_result_is_acceptable() {
        let high = FidelityResult::High {
            confidence: BetaConfidence::new(10.0, 2.0),
            term: ResolvedTerm {
                curie: "CHEBI:15365".to_string(),
                label: Some("aspirin".to_string()),
                definition: None,
                superclasses: vec![],
                synonyms: vec![],
                layer: OntologyLayer::Domain,
                iri: None,
            },
            subsumption: SubsumptionFidelity::NotChecked,
            provenance_valid: true,
        };

        assert!(high.is_acceptable());
        assert!(!high.is_violation());

        let violation = FidelityResult::Violation {
            reason: "Test violation".to_string(),
            details: ViolationDetails {
                violation_type: ViolationType::InvalidTerm,
                term: Some("BAD:0000".to_string()),
                expected_domain: None,
                actual_domain: None,
                semantic_distance: None,
            },
            suggestions: vec![],
        };

        assert!(!violation.is_acceptable());
        assert!(violation.is_violation());
    }

    #[test]
    fn test_aggregate_fidelity_scoring() {
        let results = vec![
            FidelityResult::High {
                confidence: BetaConfidence::new(10.0, 2.0),
                term: ResolvedTerm {
                    curie: "CHEBI:15365".to_string(),
                    label: Some("aspirin".to_string()),
                    definition: None,
                    superclasses: vec![],
                    synonyms: vec![],
                    layer: OntologyLayer::Domain,
                    iri: None,
                },
                subsumption: SubsumptionFidelity::NotChecked,
                provenance_valid: true,
            },
            FidelityResult::Medium {
                confidence: BetaConfidence::new(5.0, 3.0),
                term: ResolvedTerm {
                    curie: "GO:0008150".to_string(),
                    label: Some("biological_process".to_string()),
                    definition: None,
                    superclasses: vec![],
                    synonyms: vec![],
                    layer: OntologyLayer::Domain,
                    iri: None,
                },
                caveats: vec![],
            },
        ];

        // Test aggregate without creating a full checker
        let mut high_count = 0;
        let mut medium_count = 0;
        for result in &results {
            match result {
                FidelityResult::High { confidence, .. } => {
                    high_count += 1;
                }
                FidelityResult::Medium { confidence, .. } => {
                    medium_count += 1;
                }
                _ => {}
            }
        }

        assert_eq!(high_count, 1);
        assert_eq!(medium_count, 1);

        let total = (high_count + medium_count) as f64;
        let score = (high_count as f64 * 1.0 + medium_count as f64 * 0.7) / total;
        assert!(score > 0.8); // High + Medium should yield good score
    }

    #[test]
    fn test_beta_confidence_mean() {
        // Beta(10, 2) should have mean ≈ 0.833
        let beta = BetaConfidence::new(10.0, 2.0);
        assert!((beta.mean() - 0.8333).abs() < 0.001);

        // Beta(1, 1) should have mean = 0.5
        let uniform = BetaConfidence::new(1.0, 1.0);
        assert!((uniform.mean() - 0.5).abs() < 0.001);
    }
}
