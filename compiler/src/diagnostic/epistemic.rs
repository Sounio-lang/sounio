//! Epistemic Integrity Diagnostics
//!
//! This module provides compile-time warnings and errors for epistemic issues:
//!
//! - Low confidence in critical code paths
//! - Epistemic heterogeneity conflicts
//! - Missing or degraded provenance chains
//! - Ontological type compatibility issues
//! - Revision requirement violations
//!
//! # Design Philosophy
//!
//! The epistemic warning system treats knowledge quality as a first-class concern.
//! Just as Rust warns about unused variables, Sounio warns about uncertain
//! knowledge in contexts where certainty matters.
//!
//! # Warning Levels
//!
//! - **Error**: Critical epistemic violation (e.g., using uncertain value in safety-critical context)
//! - **Warning**: Potential issue that may affect correctness (e.g., heterogeneity in merge)
//! - **Note**: Informational (e.g., confidence degradation through transformation chain)
//! - **Help**: Suggestions for improving epistemic integrity

use std::fmt;

use crate::diagnostic::{Diagnostic, DiagnosticLevel, Span};
use crate::epistemic::{
    EpistemicStatus, HeterogeneityResolver, ResolutionStrategy, Revisability, Source,
};

/// Epistemic diagnostic code categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicCode {
    // Confidence issues (E100-E199)
    /// Confidence below required threshold
    LowConfidence = 100,
    /// Confidence degraded through transformation
    ConfidenceDegradation = 101,
    /// Confidence interval too wide
    WideConfidenceInterval = 102,
    /// Zero confidence (complete uncertainty)
    ZeroConfidence = 103,

    // Heterogeneity issues (E200-E299)
    /// High heterogeneity in merge operation
    HighHeterogeneity = 200,
    /// Conflicting sources in combination
    ConflictingSources = 201,
    /// Inconsistent revisability in merge
    InconsistentRevisability = 202,
    /// Resolution strategy mismatch
    ResolutionMismatch = 203,

    // Provenance issues (E300-E399)
    /// Missing provenance chain
    MissingProvenance = 300,
    /// Unknown source
    UnknownSource = 301,
    /// Provenance chain too long
    ProvenanceChainTooLong = 302,
    /// Unverified transformation
    UnverifiedTransformation = 303,

    // Ontological issues (E400-E499)
    /// Ontology term not found
    TermNotFound = 400,
    /// Subsumption check failed
    SubsumptionFailed = 401,
    /// Incompatible ontology domains
    IncompatibleDomains = 402,
    /// Deprecated ontology term
    DeprecatedTerm = 403,
    /// Cross-ontology mapping uncertain
    UncertainMapping = 404,

    // Revision issues (E500-E599)
    /// Value must be revised but wasn't
    RevisionRequired = 500,
    /// Non-revisable value modified
    NonRevisableModified = 501,
    /// Revision conditions not met
    RevisionConditionsUnmet = 502,

    // Temporal issues (E600-E699)
    /// Knowledge may be stale
    PotentiallyStale = 600,
    /// Temporal validity expired
    ValidityExpired = 601,
    /// Context time mismatch
    ContextTimeMismatch = 602,
}

impl EpistemicCode {
    /// Get the string code (e.g., "E0100")
    pub fn code(&self) -> String {
        format!("E{:04}", *self as u16)
    }

    /// Get the severity level for this code
    pub fn severity(&self) -> EpistemicSeverity {
        match self {
            // Errors
            EpistemicCode::ZeroConfidence
            | EpistemicCode::RevisionRequired
            | EpistemicCode::NonRevisableModified
            | EpistemicCode::ValidityExpired => EpistemicSeverity::Error,

            // Warnings
            EpistemicCode::LowConfidence
            | EpistemicCode::HighHeterogeneity
            | EpistemicCode::ConflictingSources
            | EpistemicCode::MissingProvenance
            | EpistemicCode::UnknownSource
            | EpistemicCode::SubsumptionFailed
            | EpistemicCode::IncompatibleDomains
            | EpistemicCode::DeprecatedTerm
            | EpistemicCode::PotentiallyStale => EpistemicSeverity::Warning,

            // Notes
            EpistemicCode::ConfidenceDegradation
            | EpistemicCode::WideConfidenceInterval
            | EpistemicCode::InconsistentRevisability
            | EpistemicCode::ResolutionMismatch
            | EpistemicCode::ProvenanceChainTooLong
            | EpistemicCode::UnverifiedTransformation
            | EpistemicCode::UncertainMapping
            | EpistemicCode::RevisionConditionsUnmet
            | EpistemicCode::ContextTimeMismatch => EpistemicSeverity::Note,

            // Help level (informational)
            EpistemicCode::TermNotFound => EpistemicSeverity::Help,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            EpistemicCode::LowConfidence => "confidence below required threshold",
            EpistemicCode::ConfidenceDegradation => "confidence degraded through transformation",
            EpistemicCode::WideConfidenceInterval => "confidence interval too wide",
            EpistemicCode::ZeroConfidence => "zero confidence indicates complete uncertainty",
            EpistemicCode::HighHeterogeneity => "high epistemic heterogeneity in combination",
            EpistemicCode::ConflictingSources => "conflicting knowledge sources",
            EpistemicCode::InconsistentRevisability => "inconsistent revisability constraints",
            EpistemicCode::ResolutionMismatch => "resolution strategy may not be appropriate",
            EpistemicCode::MissingProvenance => "missing provenance information",
            EpistemicCode::UnknownSource => "unknown knowledge source",
            EpistemicCode::ProvenanceChainTooLong => "provenance chain exceeds recommended depth",
            EpistemicCode::UnverifiedTransformation => "transformation lacks verification",
            EpistemicCode::TermNotFound => "ontology term not found",
            EpistemicCode::SubsumptionFailed => "subsumption check failed",
            EpistemicCode::IncompatibleDomains => "incompatible ontology domains",
            EpistemicCode::DeprecatedTerm => "ontology term is deprecated",
            EpistemicCode::UncertainMapping => "cross-ontology mapping has low confidence",
            EpistemicCode::RevisionRequired => "value requires revision but was not revised",
            EpistemicCode::NonRevisableModified => "non-revisable value was modified",
            EpistemicCode::RevisionConditionsUnmet => "revision conditions not satisfied",
            EpistemicCode::PotentiallyStale => "knowledge may be stale",
            EpistemicCode::ValidityExpired => "temporal validity has expired",
            EpistemicCode::ContextTimeMismatch => "context time does not match",
        }
    }
}

impl fmt::Display for EpistemicCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Severity level for epistemic warnings
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EpistemicSeverity {
    /// Informational help message
    Help,
    /// Non-critical note
    Note,
    /// Warning that should be addressed
    Warning,
    /// Error that must be fixed
    Error,
}

impl From<EpistemicSeverity> for DiagnosticLevel {
    fn from(severity: EpistemicSeverity) -> Self {
        match severity {
            EpistemicSeverity::Error => DiagnosticLevel::Error,
            EpistemicSeverity::Warning => DiagnosticLevel::Warning,
            EpistemicSeverity::Note => DiagnosticLevel::Note,
            EpistemicSeverity::Help => DiagnosticLevel::Help,
        }
    }
}

/// An epistemic diagnostic message
#[derive(Debug, Clone)]
pub struct EpistemicDiagnostic {
    /// The diagnostic code
    pub code: EpistemicCode,
    /// Primary message
    pub message: String,
    /// Source span (if available)
    pub span: Option<Span>,
    /// Additional context/notes
    pub notes: Vec<String>,
    /// Suggestions for fixing
    pub suggestions: Vec<EpistemicSuggestion>,
    /// Related values (for context)
    pub related: Vec<RelatedValue>,
}

impl EpistemicDiagnostic {
    /// Create a new epistemic diagnostic
    pub fn new(code: EpistemicCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            suggestions: Vec::new(),
            related: Vec::new(),
        }
    }

    /// Add a source span
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a note
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add a suggestion
    pub fn with_suggestion(mut self, suggestion: EpistemicSuggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Add related value context
    pub fn with_related(mut self, related: RelatedValue) -> Self {
        self.related.push(related);
        self
    }

    /// Get severity level
    pub fn severity(&self) -> EpistemicSeverity {
        self.code.severity()
    }

    /// Convert to standard Diagnostic
    pub fn to_diagnostic(&self) -> Diagnostic {
        let level: DiagnosticLevel = self.code.severity().into();
        let mut diag = Diagnostic::new(level, &self.message).with_code(self.code.code());

        if let Some(span) = &self.span {
            diag = diag.with_label(*span, "");
        }

        for note in &self.notes {
            diag = diag.with_note(note);
        }

        for suggestion in &self.suggestions {
            diag = diag.with_help(&suggestion.message);
        }

        diag
    }
}

/// A suggestion for fixing an epistemic issue
#[derive(Debug, Clone)]
pub struct EpistemicSuggestion {
    /// Description of the fix
    pub message: String,
    /// Suggested code change (if applicable)
    pub replacement: Option<String>,
}

impl EpistemicSuggestion {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            replacement: None,
        }
    }

    pub fn with_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.replacement = Some(replacement.into());
        self
    }
}

/// Related value for diagnostic context
#[derive(Debug, Clone)]
pub struct RelatedValue {
    /// Name or description
    pub name: String,
    /// Confidence value
    pub confidence: Option<f64>,
    /// Source description
    pub source: Option<String>,
}

/// Configuration for epistemic checking
#[derive(Debug, Clone)]
pub struct EpistemicCheckConfig {
    /// Minimum confidence threshold (below triggers warning)
    pub min_confidence: f64,
    /// Confidence warning threshold (below triggers note)
    pub confidence_warning: f64,
    /// Maximum heterogeneity before warning
    pub max_heterogeneity: f64,
    /// Maximum provenance chain depth
    pub max_provenance_depth: u32,
    /// Confidence interval width threshold
    pub max_interval_width: f64,
    /// Whether to check for deprecated terms
    pub check_deprecated: bool,
    /// Whether to require provenance
    pub require_provenance: bool,
    /// Mapping confidence threshold
    pub mapping_confidence_threshold: f64,
}

impl Default for EpistemicCheckConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            confidence_warning: 0.7,
            max_heterogeneity: 0.3,
            max_provenance_depth: 10,
            max_interval_width: 0.4,
            check_deprecated: true,
            require_provenance: false,
            mapping_confidence_threshold: 0.8,
        }
    }
}

/// The epistemic integrity checker
pub struct EpistemicIntegrityChecker {
    config: EpistemicCheckConfig,
    heterogeneity_resolver: HeterogeneityResolver,
    diagnostics: Vec<EpistemicDiagnostic>,
}

impl EpistemicIntegrityChecker {
    /// Create a new integrity checker with default config
    pub fn new() -> Self {
        Self {
            config: EpistemicCheckConfig::default(),
            heterogeneity_resolver: HeterogeneityResolver::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: EpistemicCheckConfig) -> Self {
        Self {
            config,
            heterogeneity_resolver: HeterogeneityResolver::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Get collected diagnostics
    pub fn diagnostics(&self) -> &[EpistemicDiagnostic] {
        &self.diagnostics
    }

    /// Take collected diagnostics (consuming)
    pub fn take_diagnostics(&mut self) -> Vec<EpistemicDiagnostic> {
        std::mem::take(&mut self.diagnostics)
    }

    /// Clear collected diagnostics
    pub fn clear(&mut self) {
        self.diagnostics.clear();
    }

    /// Check an epistemic status and collect any diagnostics
    pub fn check_status(&mut self, status: &EpistemicStatus, context: &str) {
        self.check_confidence(status, context);
        self.check_source(status, context);
        self.check_revisability(status, context);
    }

    /// Check confidence level
    fn check_confidence(&mut self, status: &EpistemicStatus, context: &str) {
        let confidence = status.confidence.value();

        // Zero confidence is an error
        if confidence <= 0.001 {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::ZeroConfidence,
                    format!("zero confidence in {}", context),
                )
                .with_note("A confidence of 0 indicates complete uncertainty")
                .with_suggestion(EpistemicSuggestion::new(
                    "Provide evidence or use a different source to establish confidence",
                )),
            );
            return;
        }

        // Low confidence warning
        if confidence < self.config.min_confidence {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::LowConfidence,
                    format!(
                        "confidence {:.2} is below threshold {:.2} in {}",
                        confidence, self.config.min_confidence, context
                    ),
                )
                .with_note(format!(
                    "Current confidence: {:.2}, Required: {:.2}",
                    confidence, self.config.min_confidence
                ))
                .with_suggestion(EpistemicSuggestion::new(
                    "Consider adding evidence or using a more reliable source",
                )),
            );
        } else if confidence < self.config.confidence_warning {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::ConfidenceDegradation,
                    format!("moderate confidence {:.2} in {}", confidence, context),
                )
                .with_note(
                    "Confidence is above minimum but may be insufficient for critical paths",
                ),
            );
        }

        // Check confidence interval width if available
        if let (Some(lower), Some(upper)) = (
            status.confidence.lower_bound(),
            status.confidence.upper_bound(),
        ) {
            let width = upper - lower;
            if width > self.config.max_interval_width {
                self.diagnostics.push(
                    EpistemicDiagnostic::new(
                        EpistemicCode::WideConfidenceInterval,
                        format!(
                            "confidence interval [{:.2}, {:.2}] is too wide ({:.2}) in {}",
                            lower, upper, width, context
                        ),
                    )
                    .with_note("Wide intervals indicate high uncertainty")
                    .with_suggestion(EpistemicSuggestion::new(
                        "Gather more evidence to narrow the confidence interval",
                    )),
                );
            }
        }
    }

    /// Check source validity
    fn check_source(&mut self, status: &EpistemicStatus, context: &str) {
        match &status.source {
            Source::Unknown => {
                if self.config.require_provenance {
                    self.diagnostics.push(
                        EpistemicDiagnostic::new(
                            EpistemicCode::UnknownSource,
                            format!("unknown source in {}", context),
                        )
                        .with_note("Knowledge without known source cannot be verified")
                        .with_suggestion(EpistemicSuggestion::new(
                            "Specify the source of this knowledge",
                        )),
                    );
                }
            }
            Source::Transformation { original, via: _ } => {
                // Check transformation chain depth
                let depth = self.count_transformation_depth(original);
                if depth > self.config.max_provenance_depth {
                    self.diagnostics.push(
                        EpistemicDiagnostic::new(
                            EpistemicCode::ProvenanceChainTooLong,
                            format!(
                                "provenance chain depth {} exceeds maximum {} in {}",
                                depth, self.config.max_provenance_depth, context
                            ),
                        )
                        .with_note(
                            "Long transformation chains may accumulate errors and reduce confidence",
                        ),
                    );
                }
            }
            _ => {}
        }

        // Check for missing provenance if required
        if self.config.require_provenance && status.evidence.is_empty() {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::MissingProvenance,
                    format!("no evidence provided in {}", context),
                )
                .with_note("Evidence chains help establish trust in knowledge")
                .with_suggestion(EpistemicSuggestion::new(
                    "Add evidence using .with_evidence() or provide source documentation",
                )),
            );
        }
    }

    /// Count transformation chain depth
    fn count_transformation_depth(&self, source: &Source) -> u32 {
        match source {
            Source::Transformation { original, .. } => {
                1 + self.count_transformation_depth(original)
            }
            _ => 0,
        }
    }

    /// Check revisability constraints
    fn check_revisability(&mut self, status: &EpistemicStatus, context: &str) {
        if let Revisability::MustRevise { reason } = &status.revisability {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::RevisionRequired,
                    format!("value in {} requires revision: {}", context, reason),
                )
                .with_note("This value has been marked as provisional and must be updated")
                .with_suggestion(EpistemicSuggestion::new(
                    "Revise this value with updated information",
                )),
            );
        }
    }

    /// Check heterogeneity when combining multiple epistemic statuses
    pub fn check_combination(
        &mut self,
        statuses: &[&EpistemicStatus],
        context: &str,
        strategy: ResolutionStrategy,
    ) {
        if statuses.len() < 2 {
            return;
        }

        // Check pairwise heterogeneity
        let mut max_heterogeneity = 0.0f64;
        let mut max_pair = (0, 1);

        for i in 0..statuses.len() {
            for j in (i + 1)..statuses.len() {
                let het = self.compute_heterogeneity(statuses[i], statuses[j]);
                if het > max_heterogeneity {
                    max_heterogeneity = het;
                    max_pair = (i, j);
                }
            }
        }

        if max_heterogeneity > self.config.max_heterogeneity {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::HighHeterogeneity,
                    format!(
                        "high heterogeneity {:.2} when combining values in {}",
                        max_heterogeneity, context
                    ),
                )
                .with_note(format!(
                    "Maximum heterogeneity between values {} and {}",
                    max_pair.0, max_pair.1
                ))
                .with_suggestion(EpistemicSuggestion::new(format!(
                    "Consider using {:?} strategy or reconciling sources",
                    strategy
                ))),
            );
        }

        // Check for conflicting sources
        let has_conflicting_sources = self.detect_conflicting_sources(statuses);
        if has_conflicting_sources {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::ConflictingSources,
                    format!("conflicting knowledge sources in {}", context),
                )
                .with_note("Different sources may have different biases or validity contexts")
                .with_suggestion(EpistemicSuggestion::new(
                    "Verify source compatibility before combining",
                )),
            );
        }

        // Check revisability consistency
        let has_mixed_revisability = self.detect_mixed_revisability(statuses);
        if has_mixed_revisability {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::InconsistentRevisability,
                    format!("mixing revisable and non-revisable values in {}", context),
                )
                .with_note(
                    "Combined result will be revisable even if some inputs are non-revisable",
                ),
            );
        }
    }

    /// Compute heterogeneity between two statuses
    fn compute_heterogeneity(&self, a: &EpistemicStatus, b: &EpistemicStatus) -> f64 {
        let confidence_diff = (a.confidence.value() - b.confidence.value()).abs();

        let source_diff = if std::mem::discriminant(&a.source) == std::mem::discriminant(&b.source)
        {
            0.0
        } else {
            0.5
        };

        (confidence_diff + source_diff) / 2.0
    }

    /// Detect conflicting sources
    fn detect_conflicting_sources(&self, statuses: &[&EpistemicStatus]) -> bool {
        let mut ontology_sources: Vec<(&str, &str)> = Vec::new();

        for status in statuses {
            if let Source::OntologyAssertion { ontology, term } = &status.source {
                // Check if same term from different ontologies
                for (existing_ontology, existing_term) in &ontology_sources {
                    if existing_term == term && existing_ontology != ontology {
                        return true;
                    }
                }
                ontology_sources.push((ontology, term));
            }
        }

        false
    }

    /// Detect mixed revisability
    fn detect_mixed_revisability(&self, statuses: &[&EpistemicStatus]) -> bool {
        let has_revisable = statuses.iter().any(|s| s.revisability.is_revisable());
        let has_non_revisable = statuses
            .iter()
            .any(|s| matches!(s.revisability, Revisability::NonRevisable));
        has_revisable && has_non_revisable
    }

    /// Check ontology term validity
    pub fn check_ontology_term(
        &mut self,
        ontology: &str,
        term_id: &str,
        deprecated: bool,
        context: &str,
    ) {
        if deprecated && self.config.check_deprecated {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::DeprecatedTerm,
                    format!(
                        "deprecated term {}:{} used in {}",
                        ontology, term_id, context
                    ),
                )
                .with_note("Deprecated terms may be removed in future ontology versions")
                .with_suggestion(EpistemicSuggestion::new(
                    "Check ontology for recommended replacement term",
                )),
            );
        }
    }

    /// Check cross-ontology mapping confidence
    pub fn check_mapping(&mut self, mapping_confidence: f64, from: &str, to: &str, context: &str) {
        if mapping_confidence < self.config.mapping_confidence_threshold {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::UncertainMapping,
                    format!(
                        "uncertain mapping ({:.2}) from {} to {} in {}",
                        mapping_confidence, from, to, context
                    ),
                )
                .with_note(format!(
                    "Mapping confidence {:.2} is below threshold {:.2}",
                    mapping_confidence, self.config.mapping_confidence_threshold
                ))
                .with_suggestion(EpistemicSuggestion::new(
                    "Consider using a more direct ontology term or verify the mapping",
                )),
            );
        }
    }

    /// Check if value assignment violates non-revisability
    pub fn check_non_revisable_modification(
        &mut self,
        original: &EpistemicStatus,
        context: &str,
    ) -> bool {
        if matches!(original.revisability, Revisability::NonRevisable) {
            self.diagnostics.push(
                EpistemicDiagnostic::new(
                    EpistemicCode::NonRevisableModified,
                    format!(
                        "attempted modification of non-revisable value in {}",
                        context
                    ),
                )
                .with_note("Non-revisable values (axioms, definitions) cannot be changed")
                .with_suggestion(EpistemicSuggestion::new(
                    "Create a new derived value instead of modifying the original",
                )),
            );
            true
        } else {
            false
        }
    }

    /// Generate summary report
    pub fn summary(&self) -> IntegritySummary {
        let mut summary = IntegritySummary::default();

        for diag in &self.diagnostics {
            summary.total += 1;
            match diag.severity() {
                EpistemicSeverity::Error => summary.errors += 1,
                EpistemicSeverity::Warning => summary.warnings += 1,
                EpistemicSeverity::Note => summary.notes += 1,
                EpistemicSeverity::Help => summary.help += 1,
            }
        }

        summary
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity() == EpistemicSeverity::Error)
    }

    /// Check if there are any warnings or worse
    pub fn has_warnings(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity() >= EpistemicSeverity::Warning)
    }
}

impl Default for EpistemicIntegrityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of integrity check results
#[derive(Debug, Clone, Default)]
pub struct IntegritySummary {
    pub total: usize,
    pub errors: usize,
    pub warnings: usize,
    pub notes: usize,
    pub help: usize,
}

impl fmt::Display for IntegritySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} total ({} errors, {} warnings, {} notes, {} help)",
            self.total, self.errors, self.warnings, self.notes, self.help
        )
    }
}

/// Quick check function for a single epistemic status
pub fn check_epistemic_integrity(
    status: &EpistemicStatus,
    config: &EpistemicCheckConfig,
) -> Vec<EpistemicDiagnostic> {
    let mut checker = EpistemicIntegrityChecker::with_config(config.clone());
    checker.check_status(status, "value");
    checker.take_diagnostics()
}

/// Quick check for combination
pub fn check_combination_integrity(
    statuses: &[&EpistemicStatus],
    config: &EpistemicCheckConfig,
    strategy: ResolutionStrategy,
) -> Vec<EpistemicDiagnostic> {
    let mut checker = EpistemicIntegrityChecker::with_config(config.clone());
    checker.check_combination(statuses, "combination", strategy);
    checker.take_diagnostics()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{Confidence, Evidence};

    #[test]
    fn test_low_confidence_warning() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.3),
            ..Default::default()
        };

        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_status(&status, "test");

        assert!(!checker.diagnostics().is_empty());
        assert!(
            checker
                .diagnostics()
                .iter()
                .any(|d| d.code == EpistemicCode::LowConfidence)
        );
    }

    #[test]
    fn test_zero_confidence_error() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.0),
            ..Default::default()
        };

        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_status(&status, "test");

        assert!(checker.has_errors());
        assert!(
            checker
                .diagnostics()
                .iter()
                .any(|d| d.code == EpistemicCode::ZeroConfidence)
        );
    }

    #[test]
    fn test_high_confidence_no_warning() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.95),
            ..Default::default()
        };

        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_status(&status, "test");

        // Should have no confidence-related warnings
        assert!(!checker.diagnostics().iter().any(|d| matches!(
            d.code,
            EpistemicCode::LowConfidence
                | EpistemicCode::ZeroConfidence
                | EpistemicCode::ConfidenceDegradation
        )));
    }

    #[test]
    fn test_revision_required_error() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.8),
            revisability: Revisability::MustRevise {
                reason: "provisional data".into(),
            },
            source: Source::Axiom,
            evidence: vec![],
        };

        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_status(&status, "test");

        assert!(checker.has_errors());
        assert!(
            checker
                .diagnostics()
                .iter()
                .any(|d| d.code == EpistemicCode::RevisionRequired)
        );
    }

    #[test]
    fn test_heterogeneity_check() {
        let status1 = EpistemicStatus {
            confidence: Confidence::new(0.9),
            source: Source::Axiom,
            ..Default::default()
        };
        let status2 = EpistemicStatus {
            confidence: Confidence::new(0.3),
            source: Source::Unknown,
            ..Default::default()
        };

        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_combination(&[&status1, &status2], "test", ResolutionStrategy::Bayesian);

        assert!(checker.has_warnings());
        assert!(
            checker
                .diagnostics()
                .iter()
                .any(|d| d.code == EpistemicCode::HighHeterogeneity)
        );
    }

    #[test]
    fn test_deprecated_term_warning() {
        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_ontology_term("GO", "0000001", true, "test");

        assert!(
            checker
                .diagnostics()
                .iter()
                .any(|d| d.code == EpistemicCode::DeprecatedTerm)
        );
    }

    #[test]
    fn test_uncertain_mapping_warning() {
        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_mapping(0.5, "CHEBI:12345", "DRUGBANK:DB00001", "test");

        assert!(
            checker
                .diagnostics()
                .iter()
                .any(|d| d.code == EpistemicCode::UncertainMapping)
        );
    }

    #[test]
    fn test_non_revisable_modification() {
        let status = EpistemicStatus::axiomatic();

        let mut checker = EpistemicIntegrityChecker::new();
        let violated = checker.check_non_revisable_modification(&status, "test");

        assert!(violated);
        assert!(checker.has_errors());
    }

    #[test]
    fn test_provenance_chain_depth() {
        // Create a deep transformation chain
        let mut source = Source::Axiom;
        for i in 0..15 {
            source = Source::Transformation {
                original: Box::new(source),
                via: format!("transform_{}", i),
            };
        }

        let status = EpistemicStatus {
            confidence: Confidence::new(0.8),
            source,
            ..Default::default()
        };

        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_status(&status, "test");

        assert!(
            checker
                .diagnostics()
                .iter()
                .any(|d| d.code == EpistemicCode::ProvenanceChainTooLong)
        );
    }

    #[test]
    fn test_summary() {
        let mut checker = EpistemicIntegrityChecker::new();

        // Add various diagnostics
        let low_conf = EpistemicStatus {
            confidence: Confidence::new(0.3),
            ..Default::default()
        };
        checker.check_status(&low_conf, "test1");

        let zero_conf = EpistemicStatus {
            confidence: Confidence::new(0.0),
            ..Default::default()
        };
        checker.check_status(&zero_conf, "test2");

        let summary = checker.summary();
        assert!(summary.total >= 2);
        assert!(summary.errors >= 1);
        assert!(summary.warnings >= 1);
    }

    #[test]
    fn test_epistemic_code_display() {
        assert_eq!(EpistemicCode::LowConfidence.code(), "E0100");
        assert_eq!(EpistemicCode::HighHeterogeneity.code(), "E0200");
        assert_eq!(EpistemicCode::MissingProvenance.code(), "E0300");
    }

    #[test]
    fn test_quick_check_function() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.2),
            ..Default::default()
        };

        let diagnostics = check_epistemic_integrity(&status, &EpistemicCheckConfig::default());
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn test_wide_confidence_interval() {
        let status = EpistemicStatus {
            confidence: Confidence::with_bounds(0.5, 0.1, 0.9),
            ..Default::default()
        };

        let mut checker = EpistemicIntegrityChecker::new();
        checker.check_status(&status, "test");

        assert!(
            checker
                .diagnostics()
                .iter()
                .any(|d| d.code == EpistemicCode::WideConfidenceInterval)
        );
    }
}
