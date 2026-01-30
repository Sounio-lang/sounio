//! Gradual typing for dependent epistemic types
//!
//! When proof search fails, gradual typing allows deferring checks to runtime.
//! This provides a smooth migration path from dynamically-checked to
//! statically-verified code.
//!
//! # The Unknown Type
//!
//! `?` is a special type compatible with any type:
//! - `Knowledge[τ, ?]` = confidence unknown statically
//! - `?` ~ `ε` for any `ε` (compatible at runtime)
//!
//! # Gradual Guarantees
//!
//! - **Static mode**: All constraints proven at compile time
//! - **Mixed mode**: Some static, some runtime checks
//! - **Dynamic mode**: All checks deferred to runtime
//!
//! # Runtime Checks
//!
//! When proof search fails and gradual is enabled:
//! 1. Insert runtime check at the appropriate point
//! 2. Generate warning for developer
//! 3. At runtime, check fails → panic with clear message

use super::predicates::Predicate;

/// Gradual typing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradualMode {
    /// Strict: all proofs must be static
    #[default]
    Strict,
    /// Permissive: allow runtime checks as fallback
    Permissive,
    /// Dynamic: defer all checks to runtime
    Dynamic,
}

impl GradualMode {
    /// Check if gradual fallback is allowed
    pub fn allows_runtime_checks(&self) -> bool {
        matches!(self, Self::Permissive | Self::Dynamic)
    }

    /// Check if static proofs are required
    pub fn requires_static(&self) -> bool {
        matches!(self, Self::Strict)
    }

    /// Check if in dynamic mode
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic)
    }
}

/// Configuration for gradual typing
#[derive(Debug, Clone)]
pub struct GradualConfig {
    /// The gradual mode
    pub mode: GradualMode,
    /// Whether to emit warnings for runtime checks
    pub warn_on_runtime: bool,
    /// Whether to emit info about deferred checks
    pub verbose: bool,
    /// Maximum number of runtime checks before error
    pub max_runtime_checks: Option<usize>,
    /// Whether to track runtime check locations
    pub track_locations: bool,
}

impl Default for GradualConfig {
    fn default() -> Self {
        Self {
            mode: GradualMode::Strict,
            warn_on_runtime: true,
            verbose: false,
            max_runtime_checks: None,
            track_locations: true,
        }
    }
}

impl GradualConfig {
    /// Create a strict configuration (no runtime checks)
    pub fn strict() -> Self {
        Self {
            mode: GradualMode::Strict,
            ..Default::default()
        }
    }

    /// Create a permissive configuration (allow runtime checks)
    pub fn permissive() -> Self {
        Self {
            mode: GradualMode::Permissive,
            warn_on_runtime: true,
            ..Default::default()
        }
    }

    /// Create a dynamic configuration (defer all checks)
    pub fn dynamic() -> Self {
        Self {
            mode: GradualMode::Dynamic,
            warn_on_runtime: false,
            ..Default::default()
        }
    }
}

/// A runtime check to be inserted
#[derive(Debug, Clone)]
pub struct RuntimeCheck {
    /// The predicate to check
    pub predicate: Predicate,
    /// The kind of check
    pub kind: RuntimeCheckKind,
    /// Location in source (if available)
    pub location: Option<SourceLocation>,
    /// Error message on failure
    pub error_message: String,
}

impl RuntimeCheck {
    /// Create a new runtime check
    pub fn new(predicate: Predicate, kind: RuntimeCheckKind) -> Self {
        let error_message = format!("Runtime check failed: {}", predicate);
        Self {
            predicate,
            kind,
            location: None,
            error_message,
        }
    }

    /// Add source location
    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Set custom error message
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.error_message = message.into();
        self
    }

    /// Generate code for this check (pseudo-code)
    pub fn generate_check_code(&self) -> String {
        match &self.kind {
            RuntimeCheckKind::ConfidenceGeq { value, threshold } => {
                format!(
                    "if {}.confidence() < {} {{ panic!(\"{}\"); }}",
                    value, threshold, self.error_message
                )
            }
            RuntimeCheckKind::OntologyContains { value, required } => {
                format!(
                    "if !{}.ontology().contains({:?}) {{ panic!(\"{}\"); }}",
                    value, required, self.error_message
                )
            }
            RuntimeCheckKind::Identifiable {
                graph,
                treatment,
                outcome,
            } => {
                format!(
                    "if !is_identifiable({}, {:?}, {:?}) {{ panic!(\"{}\"); }}",
                    graph, treatment, outcome, self.error_message
                )
            }
            RuntimeCheckKind::Fresh { value, max_age } => {
                format!(
                    "if {}.age() > {:?} {{ panic!(\"{}\"); }}",
                    value, max_age, self.error_message
                )
            }
            RuntimeCheckKind::Custom { check_expr } => {
                format!(
                    "if !({}) {{ panic!(\"{}\"); }}",
                    check_expr, self.error_message
                )
            }
        }
    }
}

/// Kind of runtime check
#[derive(Debug, Clone)]
pub enum RuntimeCheckKind {
    /// Check confidence ≥ threshold
    ConfidenceGeq {
        /// The expression producing confidence
        value: String,
        /// The threshold to check against
        threshold: f64,
    },

    /// Check ontology containment
    OntologyContains {
        /// The expression producing ontology
        value: String,
        /// The required ontology
        required: String,
    },

    /// Check causal identifiability
    Identifiable {
        /// Graph expression
        graph: String,
        /// Treatment variable
        treatment: String,
        /// Outcome variable
        outcome: String,
    },

    /// Check freshness
    Fresh {
        /// The temporal knowledge
        value: String,
        /// Maximum age
        max_age: std::time::Duration,
    },

    /// Custom check expression
    Custom {
        /// The check expression
        check_expr: String,
    },
}

/// Source location for diagnostics
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File path
    pub file: String,
    /// Line number
    pub line: usize,
    /// Column number
    pub column: usize,
    /// Span length
    pub length: usize,
}

impl SourceLocation {
    /// Create a new source location
    pub fn new(file: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            length: 0,
        }
    }

    /// Add span length
    pub fn with_length(mut self, length: usize) -> Self {
        self.length = length;
        self
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Warning about a runtime check
#[derive(Debug, Clone)]
pub struct GradualWarning {
    /// The predicate that couldn't be proven
    pub predicate: Predicate,
    /// Location in source
    pub location: Option<SourceLocation>,
    /// Reason proof failed
    pub reason: String,
    /// Suggestion for fixing
    pub suggestion: Option<String>,
}

impl GradualWarning {
    /// Create a new warning
    pub fn new(predicate: Predicate, reason: impl Into<String>) -> Self {
        Self {
            predicate,
            location: None,
            reason: reason.into(),
            suggestion: None,
        }
    }

    /// Add location
    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Add suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Format as diagnostic message
    pub fn format(&self) -> String {
        let mut msg = format!(
            "warning: Cannot statically prove predicate\n  predicate: {}\n  reason: {}",
            self.predicate, self.reason
        );

        if let Some(loc) = &self.location {
            msg.push_str(&format!("\n  --> {}", loc));
        }

        if let Some(sug) = &self.suggestion {
            msg.push_str(&format!("\n  help: {}", sug));
        }

        msg
    }
}

/// Collector for gradual typing diagnostics
#[derive(Debug, Default)]
pub struct GradualDiagnostics {
    /// Runtime checks inserted
    pub checks: Vec<RuntimeCheck>,
    /// Warnings emitted
    pub warnings: Vec<GradualWarning>,
    /// Count of checks by kind
    pub check_counts: std::collections::HashMap<String, usize>,
}

impl GradualDiagnostics {
    /// Create new diagnostics collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a runtime check
    pub fn add_check(&mut self, check: RuntimeCheck) {
        let kind_name = match &check.kind {
            RuntimeCheckKind::ConfidenceGeq { .. } => "confidence",
            RuntimeCheckKind::OntologyContains { .. } => "ontology",
            RuntimeCheckKind::Identifiable { .. } => "identifiable",
            RuntimeCheckKind::Fresh { .. } => "fresh",
            RuntimeCheckKind::Custom { .. } => "custom",
        };
        *self.check_counts.entry(kind_name.to_string()).or_insert(0) += 1;
        self.checks.push(check);
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: GradualWarning) {
        self.warnings.push(warning);
    }

    /// Get total number of runtime checks
    pub fn total_checks(&self) -> usize {
        self.checks.len()
    }

    /// Get total number of warnings
    pub fn total_warnings(&self) -> usize {
        self.warnings.len()
    }

    /// Check if any runtime checks were inserted
    pub fn has_runtime_checks(&self) -> bool {
        !self.checks.is_empty()
    }

    /// Format summary
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Gradual typing summary:\n  Runtime checks: {}\n  Warnings: {}\n",
            self.total_checks(),
            self.total_warnings()
        );

        if !self.check_counts.is_empty() {
            summary.push_str("  By kind:\n");
            for (kind, count) in &self.check_counts {
                summary.push_str(&format!("    {}: {}\n", kind, count));
            }
        }

        summary
    }
}

/// Annotation for controlling gradual typing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradualAnnotation {
    /// @static_proof - require static proof
    StaticProof,
    /// @allow_runtime - allow runtime check
    AllowRuntime,
    /// @trusted - assume true (dangerous)
    Trusted,
    /// @debug_proof - print proof derivation
    DebugProof,
}

impl GradualAnnotation {
    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "@static_proof" | "static_proof" => Some(Self::StaticProof),
            "@allow_runtime" | "allow_runtime" => Some(Self::AllowRuntime),
            "@trusted" | "trusted" => Some(Self::Trusted),
            "@debug_proof" | "debug_proof" => Some(Self::DebugProof),
            _ => None,
        }
    }

    /// Check if this annotation allows runtime checks
    pub fn allows_runtime(&self) -> bool {
        matches!(self, Self::AllowRuntime | Self::Trusted)
    }

    /// Check if this annotation is trusted (skip checks entirely)
    pub fn is_trusted(&self) -> bool {
        matches!(self, Self::Trusted)
    }
}

/// Create a confidence runtime check
pub fn confidence_check(
    value_expr: impl Into<String>,
    threshold: f64,
    predicate: Predicate,
) -> RuntimeCheck {
    RuntimeCheck::new(
        predicate,
        RuntimeCheckKind::ConfidenceGeq {
            value: value_expr.into(),
            threshold,
        },
    )
}

/// Create an ontology runtime check
pub fn ontology_check(
    value_expr: impl Into<String>,
    required: impl Into<String>,
    predicate: Predicate,
) -> RuntimeCheck {
    RuntimeCheck::new(
        predicate,
        RuntimeCheckKind::OntologyContains {
            value: value_expr.into(),
            required: required.into(),
        },
    )
}

/// Create an identifiability runtime check
pub fn identifiability_check(
    graph_expr: impl Into<String>,
    treatment: impl Into<String>,
    outcome: impl Into<String>,
    predicate: Predicate,
) -> RuntimeCheck {
    RuntimeCheck::new(
        predicate,
        RuntimeCheckKind::Identifiable {
            graph: graph_expr.into(),
            treatment: treatment.into(),
            outcome: outcome.into(),
        },
    )
}

/// Create a freshness runtime check
pub fn freshness_check(
    value_expr: impl Into<String>,
    max_age: std::time::Duration,
    predicate: Predicate,
) -> RuntimeCheck {
    RuntimeCheck::new(
        predicate,
        RuntimeCheckKind::Fresh {
            value: value_expr.into(),
            max_age,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradual_mode() {
        assert!(!GradualMode::Strict.allows_runtime_checks());
        assert!(GradualMode::Permissive.allows_runtime_checks());
        assert!(GradualMode::Dynamic.allows_runtime_checks());
    }

    #[test]
    fn test_gradual_config() {
        let strict = GradualConfig::strict();
        assert!(strict.mode.requires_static());

        let permissive = GradualConfig::permissive();
        assert!(permissive.warn_on_runtime);

        let dynamic = GradualConfig::dynamic();
        assert!(dynamic.mode.is_dynamic());
    }

    #[test]
    fn test_runtime_check_code_gen() {
        let pred = Predicate::confidence_geq(
            super::super::types::ConfidenceType::var("k"),
            super::super::types::ConfidenceType::literal(0.95),
        );
        let check = confidence_check("k", 0.95, pred);
        let code = check.generate_check_code();
        assert!(code.contains("confidence()"));
        assert!(code.contains("0.95"));
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new("src/main.sio", 42, 10).with_length(5);
        let s = format!("{}", loc);
        assert!(s.contains("src/main.sio"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_gradual_warning() {
        let pred = Predicate::confidence_geq(
            super::super::types::ConfidenceType::var("ε"),
            super::super::types::ConfidenceType::literal(0.95),
        );
        let warning = GradualWarning::new(pred, "Variable unbound")
            .with_location(SourceLocation::new("test.sio", 10, 5))
            .with_suggestion("Add type annotation");

        let formatted = warning.format();
        assert!(formatted.contains("warning"));
        assert!(formatted.contains("unbound"));
        assert!(formatted.contains("test.sio"));
    }

    #[test]
    fn test_gradual_diagnostics() {
        let mut diag = GradualDiagnostics::new();

        let pred = Predicate::true_();
        diag.add_check(confidence_check("k1", 0.90, pred.clone()));
        diag.add_check(confidence_check("k2", 0.95, pred.clone()));
        diag.add_warning(GradualWarning::new(pred, "test"));

        assert_eq!(diag.total_checks(), 2);
        assert_eq!(diag.total_warnings(), 1);
        assert!(diag.has_runtime_checks());

        let summary = diag.summary();
        assert!(summary.contains("Runtime checks: 2"));
    }

    #[test]
    fn test_gradual_annotation_parse() {
        assert_eq!(
            GradualAnnotation::parse("@static_proof"),
            Some(GradualAnnotation::StaticProof)
        );
        assert_eq!(
            GradualAnnotation::parse("allow_runtime"),
            Some(GradualAnnotation::AllowRuntime)
        );
        assert_eq!(
            GradualAnnotation::parse("@trusted"),
            Some(GradualAnnotation::Trusted)
        );
        assert!(GradualAnnotation::parse("unknown").is_none());
    }

    #[test]
    fn test_annotation_properties() {
        assert!(!GradualAnnotation::StaticProof.allows_runtime());
        assert!(GradualAnnotation::AllowRuntime.allows_runtime());
        assert!(GradualAnnotation::Trusted.is_trusted());
    }
}
