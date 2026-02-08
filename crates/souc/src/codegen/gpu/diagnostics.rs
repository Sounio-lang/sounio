//! GPU Diagnostics System
//!
//! Unified diagnostic reporting for GPU compilation and optimization passes.
//! Aggregates errors, warnings, and hints from all GPU-related phases with
//! source location tracking.
//!
//! # Architecture
//!
//! ```text
//! FusionError, OptimizerError, ValidationError, etc.
//!                      │
//!                      ▼
//!              DiagnosticContext
//!                      │
//!                      ▼
//!              DiagnosticReport { errors, warnings, hints }
//! ```

use std::fmt;

use super::fusion::FusionError;
use super::optimizer::OptimizerError;
use super::sourcemap::GpuSourceMapper;
use super::validation::{ValidationError, ValidationIssue};
use crate::common::Span;

// Re-export GpuIrLocation from sourcemap for unified access
pub use super::sourcemap::GpuIrLocation;

// ============================================================================
// Diagnostic Types
// ============================================================================

/// Severity level of a diagnostic
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticSeverity {
    /// Informational message
    Info,
    /// Warning - compilation continues but may indicate problems
    Warning,
    /// Error - compilation failed
    Error,
}

impl fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticSeverity::Info => write!(f, "info"),
            DiagnosticSeverity::Warning => write!(f, "warning"),
            DiagnosticSeverity::Error => write!(f, "error"),
        }
    }
}

/// Kind/category of GPU diagnostic
#[derive(Debug, Clone)]
pub enum GpuDiagnosticKind {
    /// Fusion-related error
    Fusion(FusionError),
    /// Optimizer error
    Optimizer(Box<OptimizerError>),
    /// Validation error
    Validation(ValidationError),
    /// PTX codegen error
    Codegen(String),
    /// Runtime error
    Runtime(String),
    /// Generic GPU error
    Generic(String),
}

impl fmt::Display for GpuDiagnosticKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuDiagnosticKind::Fusion(e) => write!(f, "{}", e),
            GpuDiagnosticKind::Optimizer(e) => write!(f, "{}", e),
            GpuDiagnosticKind::Validation(e) => write!(f, "{}", e),
            GpuDiagnosticKind::Codegen(msg) => write!(f, "{}", msg),
            GpuDiagnosticKind::Runtime(msg) => write!(f, "{}", msg),
            GpuDiagnosticKind::Generic(msg) => write!(f, "{}", msg),
        }
    }
}

/// A unified GPU diagnostic with source locations and recovery hints
#[derive(Debug, Clone)]
pub struct GpuDiagnostic {
    /// The kind/category of diagnostic
    pub kind: GpuDiagnosticKind,
    /// Severity level
    pub severity: DiagnosticSeverity,
    /// Human-readable message
    pub message: String,
    /// Source span in HLIR (if available)
    pub hlir_span: Option<Span>,
    /// Location in GPU IR (if available)
    pub gpu_location: Option<GpuIrLocation>,
    /// PTX line number (if available)
    pub ptx_line: Option<u32>,
    /// Recovery hints
    pub hints: Vec<RecoveryHint>,
    /// Related diagnostics (e.g., "caused by")
    pub related: Vec<GpuDiagnostic>,
}

impl GpuDiagnostic {
    /// Create a new error diagnostic
    pub fn error(kind: GpuDiagnosticKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            severity: DiagnosticSeverity::Error,
            message: message.into(),
            hlir_span: None,
            gpu_location: None,
            ptx_line: None,
            hints: Vec::new(),
            related: Vec::new(),
        }
    }

    /// Create a new warning diagnostic
    pub fn warning(kind: GpuDiagnosticKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            severity: DiagnosticSeverity::Warning,
            message: message.into(),
            hlir_span: None,
            gpu_location: None,
            ptx_line: None,
            hints: Vec::new(),
            related: Vec::new(),
        }
    }

    /// Create a new info diagnostic
    pub fn info(kind: GpuDiagnosticKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            severity: DiagnosticSeverity::Info,
            message: message.into(),
            hlir_span: None,
            gpu_location: None,
            ptx_line: None,
            hints: Vec::new(),
            related: Vec::new(),
        }
    }

    /// Set HLIR source span
    pub fn with_span(mut self, span: Span) -> Self {
        self.hlir_span = Some(span);
        self
    }

    /// Set GPU IR location
    pub fn with_gpu_location(mut self, loc: GpuIrLocation) -> Self {
        self.gpu_location = Some(loc);
        self
    }

    /// Set PTX line number
    pub fn with_ptx_line(mut self, line: u32) -> Self {
        self.ptx_line = Some(line);
        self
    }

    /// Add a recovery hint
    pub fn with_hint(mut self, hint: RecoveryHint) -> Self {
        self.hints.push(hint);
        self
    }

    /// Add multiple recovery hints
    pub fn with_hints(mut self, hints: Vec<RecoveryHint>) -> Self {
        self.hints.extend(hints);
        self
    }

    /// Add a related diagnostic
    pub fn with_related(mut self, related: GpuDiagnostic) -> Self {
        self.related.push(related);
        self
    }

    /// Check if this is an error
    pub fn is_error(&self) -> bool {
        self.severity == DiagnosticSeverity::Error
    }

    /// Check if this is a warning
    pub fn is_warning(&self) -> bool {
        self.severity == DiagnosticSeverity::Warning
    }
}

impl fmt::Display for GpuDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.severity, self.message)?;

        if let Some(span) = &self.hlir_span {
            write!(f, "\n  --> {}", span)?;
        }
        if let Some(loc) = &self.gpu_location {
            write!(f, "\n  at {}", loc)?;
        }
        if let Some(line) = self.ptx_line {
            write!(f, "\n  PTX line {}", line)?;
        }

        for hint in &self.hints {
            write!(f, "\n  help: {}", hint.title)?;
        }

        Ok(())
    }
}

// ============================================================================
// Recovery Hints
// ============================================================================

/// Confidence level for recovery hints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HintConfidence {
    /// Almost certainly correct
    High,
    /// Likely correct
    Medium,
    /// May help
    Low,
}

/// A recovery hint suggesting how to fix an issue
#[derive(Debug, Clone)]
pub struct RecoveryHint {
    /// Short title
    pub title: String,
    /// Detailed explanation
    pub explanation: String,
    /// Confidence level
    pub confidence: HintConfidence,
}

impl RecoveryHint {
    /// Create a new recovery hint
    pub fn new(title: impl Into<String>, explanation: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            explanation: explanation.into(),
            confidence: HintConfidence::Medium,
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: HintConfidence) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Generator for recovery hints based on error patterns
pub struct RecoveryGenerator;

impl RecoveryGenerator {
    /// Generate hints for fusion errors
    pub fn for_fusion_error(err: &FusionError) -> Vec<RecoveryHint> {
        match err {
            FusionError::KernelNotFound(name) => vec![
                RecoveryHint::new(
                    format!("Check kernel name '{}'", name),
                    "Ensure the kernel exists in the module and is spelled correctly",
                )
                .with_confidence(HintConfidence::High),
            ],
            FusionError::BlockNotFound(id) => vec![RecoveryHint::new(
                format!("Block {} not found", id),
                "The block may have been optimized away or the ID is invalid",
            )],
            FusionError::ValueNotFound(id) => vec![RecoveryHint::new(
                format!("Value v{} not found", id.0),
                "The value may have been eliminated during optimization",
            )],
            FusionError::InvalidTransformation(msg) => vec![RecoveryHint::new(
                "Fusion transformation failed",
                format!(
                    "The requested fusion is not valid: {}. Try reducing fusion aggressiveness.",
                    msg
                ),
            )],
        }
    }

    /// Generate hints for validation errors
    pub fn for_validation_error(err: &ValidationError) -> Vec<RecoveryHint> {
        match err {
            ValidationError::SizeMismatch { expected, actual } => vec![RecoveryHint::new(
                "Output size mismatch",
                format!(
                    "Expected {} bytes but got {}. Check kernel output allocation.",
                    expected, actual
                ),
            )],
            ValidationError::ValueMismatch {
                index,
                expected,
                actual,
            } => vec![
                RecoveryHint::new(
                    format!("Value mismatch at index {}", index),
                    format!(
                        "Expected {} but got {}. This may indicate an optimization bug.",
                        expected, actual
                    ),
                )
                .with_confidence(HintConfidence::High),
            ],
            ValidationError::PrecisionLoss {
                max_error,
                threshold,
            } => vec![RecoveryHint::new(
                "Numerical precision degraded",
                format!(
                    "Maximum error {} exceeds threshold {}. Consider using higher precision or adjusting tolerance.",
                    max_error, threshold
                ),
            )],
            ValidationError::InvalidOutput { description } => vec![RecoveryHint::new(
                "Invalid output detected",
                format!("Output contains invalid values: {}", description),
            )],
        }
    }

    /// Generate hints for validation issues
    pub fn for_validation_issue(issue: &ValidationIssue) -> Vec<RecoveryHint> {
        match issue {
            ValidationIssue::ValueMismatch {
                index,
                expected,
                actual,
                ..
            } => vec![RecoveryHint::new(
                format!("Mismatch at element {}", index),
                format!(
                    "Expected {:.6e}, got {:.6e}. Check optimization passes for correctness.",
                    expected, actual
                ),
            )],
            ValidationIssue::NaNDetected { index, .. } => vec![
                RecoveryHint::new(
                    format!("NaN at element {}", index),
                    "Output contains NaN. Check for division by zero or invalid operations.",
                )
                .with_confidence(HintConfidence::High),
            ],
            ValidationIssue::InfDetected { index, .. } => vec![
                RecoveryHint::new(
                    format!("Infinity at element {}", index),
                    "Output contains infinity. Check for overflow or unbounded operations.",
                )
                .with_confidence(HintConfidence::High),
            ],
            ValidationIssue::PrecisionLoss {
                max_error,
                mean_error,
            } => vec![RecoveryHint::new(
                "Accumulated precision loss",
                format!(
                    "Max error: {:.6e}, mean error: {:.6e}. Consider Kahan summation for reductions.",
                    max_error, mean_error
                ),
            )],
        }
    }
}

// ============================================================================
// Diagnostic Context
// ============================================================================

/// Configuration for the diagnostic system
#[derive(Debug, Clone)]
pub struct DiagnosticConfig {
    /// Collect warnings (not just errors)
    pub collect_warnings: bool,
    /// Collect info messages
    pub collect_info: bool,
    /// Maximum diagnostics before stopping
    pub max_diagnostics: usize,
    /// Generate recovery hints
    pub generate_hints: bool,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            collect_warnings: true,
            collect_info: false,
            max_diagnostics: 100,
            generate_hints: true,
        }
    }
}

/// Aggregates diagnostics from all GPU compilation phases
pub struct DiagnosticContext {
    /// Collected diagnostics
    diagnostics: Vec<GpuDiagnostic>,
    /// Source mapper for location resolution
    source_mapper: Option<GpuSourceMapper>,
    /// Configuration
    config: DiagnosticConfig,
}

impl DiagnosticContext {
    /// Create a new diagnostic context
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            source_mapper: None,
            config: DiagnosticConfig::default(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: DiagnosticConfig) -> Self {
        Self {
            diagnostics: Vec::new(),
            source_mapper: None,
            config,
        }
    }

    /// Set the source mapper
    pub fn set_source_mapper(&mut self, mapper: GpuSourceMapper) {
        self.source_mapper = Some(mapper);
    }

    /// Get the source mapper (if available)
    pub fn source_mapper(&self) -> Option<&GpuSourceMapper> {
        self.source_mapper.as_ref()
    }

    /// Report a diagnostic
    pub fn report(&mut self, diagnostic: GpuDiagnostic) {
        if self.diagnostics.len() >= self.config.max_diagnostics {
            return;
        }

        match diagnostic.severity {
            DiagnosticSeverity::Info if !self.config.collect_info => return,
            DiagnosticSeverity::Warning if !self.config.collect_warnings => return,
            _ => {}
        }

        self.diagnostics.push(diagnostic);
    }

    /// Report a fusion error
    pub fn report_fusion_error(&mut self, err: &FusionError) {
        let hints = if self.config.generate_hints {
            RecoveryGenerator::for_fusion_error(err)
        } else {
            Vec::new()
        };

        let diagnostic = GpuDiagnostic::error(
            GpuDiagnosticKind::Fusion(err.clone()),
            format!("Fusion error: {}", err),
        )
        .with_hints(hints);

        self.report(diagnostic);
    }

    /// Report an optimizer error
    pub fn report_optimizer_error(&mut self, err: &OptimizerError) {
        let diagnostic = GpuDiagnostic::error(
            GpuDiagnosticKind::Optimizer(Box::new(err.clone())),
            format!("Optimizer error: {}", err),
        );

        self.report(diagnostic);
    }

    /// Report a validation error
    pub fn report_validation_error(&mut self, err: &ValidationError) {
        let hints = if self.config.generate_hints {
            RecoveryGenerator::for_validation_error(err)
        } else {
            Vec::new()
        };

        let diagnostic = GpuDiagnostic::error(
            GpuDiagnosticKind::Validation(err.clone()),
            format!("Validation error: {}", err),
        )
        .with_hints(hints);

        self.report(diagnostic);
    }

    /// Report a validation issue as a warning
    pub fn report_validation_issue(&mut self, issue: &ValidationIssue) {
        let hints = if self.config.generate_hints {
            RecoveryGenerator::for_validation_issue(issue)
        } else {
            Vec::new()
        };

        let severity = match issue {
            ValidationIssue::NaNDetected { .. } | ValidationIssue::InfDetected { .. } => {
                DiagnosticSeverity::Error
            }
            _ => DiagnosticSeverity::Warning,
        };

        let diagnostic = GpuDiagnostic {
            kind: GpuDiagnosticKind::Validation(ValidationError::InvalidOutput {
                description: format!("{:?}", issue),
            }),
            severity,
            message: format!("{:?}", issue),
            hlir_span: None,
            gpu_location: None,
            ptx_line: None,
            hints,
            related: Vec::new(),
        };

        self.report(diagnostic);
    }

    /// Report a codegen error
    pub fn report_codegen_error(&mut self, msg: impl Into<String>) {
        let diagnostic =
            GpuDiagnostic::error(GpuDiagnosticKind::Codegen(msg.into()), "PTX codegen failed");
        self.report(diagnostic);
    }

    /// Report a warning
    pub fn report_warning(&mut self, msg: impl Into<String>) {
        let diagnostic = GpuDiagnostic::warning(GpuDiagnosticKind::Generic(msg.into()), "");
        self.report(diagnostic);
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_warning())
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_error()).count()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_warning()).count()
    }

    /// Iterate over errors
    pub fn errors(&self) -> impl Iterator<Item = &GpuDiagnostic> {
        self.diagnostics.iter().filter(|d| d.is_error())
    }

    /// Iterate over warnings
    pub fn warnings(&self) -> impl Iterator<Item = &GpuDiagnostic> {
        self.diagnostics.iter().filter(|d| d.is_warning())
    }

    /// Iterate over all diagnostics
    pub fn iter(&self) -> impl Iterator<Item = &GpuDiagnostic> {
        self.diagnostics.iter()
    }

    /// Build the final diagnostic report
    pub fn build_report(&self) -> DiagnosticReport {
        let errors: Vec<_> = self.errors().cloned().collect();
        let warnings: Vec<_> = self.warnings().cloned().collect();

        DiagnosticReport {
            errors,
            warnings,
            summary: DiagnosticSummary {
                error_count: self.error_count(),
                warning_count: self.warning_count(),
            },
        }
    }
}

impl Default for DiagnosticContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Diagnostic Report
// ============================================================================

/// Summary statistics for a diagnostic report
#[derive(Debug, Clone, Default)]
pub struct DiagnosticSummary {
    /// Number of errors
    pub error_count: usize,
    /// Number of warnings
    pub warning_count: usize,
}

/// Final diagnostic report
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// All errors
    pub errors: Vec<GpuDiagnostic>,
    /// All warnings
    pub warnings: Vec<GpuDiagnostic>,
    /// Summary statistics
    pub summary: DiagnosticSummary,
}

impl DiagnosticReport {
    /// Create an empty report
    pub fn empty() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            summary: DiagnosticSummary::default(),
        }
    }

    /// Check if the report has errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if the report has warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        let mut output = String::new();

        for err in &self.errors {
            output.push_str(&format!("{}\n", err));
        }

        for warn in &self.warnings {
            output.push_str(&format!("{}\n", warn));
        }

        output.push_str(&format!(
            "\n{} error(s), {} warning(s)\n",
            self.summary.error_count, self.summary.warning_count
        ));

        output
    }
}

impl fmt::Display for DiagnosticReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl From<FusionError> for GpuDiagnostic {
    fn from(err: FusionError) -> Self {
        let hints = RecoveryGenerator::for_fusion_error(&err);
        GpuDiagnostic::error(GpuDiagnosticKind::Fusion(err.clone()), format!("{}", err))
            .with_hints(hints)
    }
}

impl From<OptimizerError> for GpuDiagnostic {
    fn from(err: OptimizerError) -> Self {
        GpuDiagnostic::error(
            GpuDiagnosticKind::Optimizer(Box::new(err.clone())),
            format!("{}", err),
        )
    }
}

impl From<ValidationError> for GpuDiagnostic {
    fn from(err: ValidationError) -> Self {
        let hints = RecoveryGenerator::for_validation_error(&err);
        GpuDiagnostic::error(
            GpuDiagnosticKind::Validation(err.clone()),
            format!("{}", err),
        )
        .with_hints(hints)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::ir::{BlockId, ValueId};
    use super::*;

    #[test]
    fn test_diagnostic_severity_ordering() {
        assert!(DiagnosticSeverity::Info < DiagnosticSeverity::Warning);
        assert!(DiagnosticSeverity::Warning < DiagnosticSeverity::Error);
    }

    #[test]
    fn test_diagnostic_context_aggregation() {
        let mut ctx = DiagnosticContext::new();

        ctx.report_fusion_error(&FusionError::KernelNotFound("test".into()));
        ctx.report_warning("This is a warning");

        assert_eq!(ctx.error_count(), 1);
        assert_eq!(ctx.warning_count(), 1);
        assert!(ctx.has_errors());
        assert!(ctx.has_warnings());
    }

    #[test]
    fn test_diagnostic_report_format() {
        let mut ctx = DiagnosticContext::new();
        ctx.report_fusion_error(&FusionError::KernelNotFound("missing".into()));

        let report = ctx.build_report();
        assert!(report.has_errors());
        assert_eq!(report.summary.error_count, 1);

        let formatted = report.format();
        assert!(formatted.contains("error"));
        assert!(formatted.contains("1 error(s)"));
    }

    #[test]
    fn test_recovery_hints_generated() {
        let hints =
            RecoveryGenerator::for_fusion_error(&FusionError::KernelNotFound("test".into()));
        assert!(!hints.is_empty());
        assert!(hints[0].title.contains("test"));
    }

    #[test]
    fn test_diagnostic_with_location() {
        let loc = GpuIrLocation {
            kernel: "my_kernel".to_string(),
            block: BlockId(0),
            instruction: 5,
            value: ValueId(10),
        };

        let diag = GpuDiagnostic::error(
            GpuDiagnosticKind::Generic("test error".into()),
            "Something went wrong",
        )
        .with_gpu_location(loc)
        .with_ptx_line(42);

        assert!(diag.gpu_location.is_some());
        assert_eq!(diag.ptx_line, Some(42));

        let formatted = format!("{}", diag);
        assert!(formatted.contains("my_kernel"));
        assert!(formatted.contains("PTX line 42"));
    }

    #[test]
    fn test_max_diagnostics_limit() {
        let config = DiagnosticConfig {
            max_diagnostics: 2,
            ..DiagnosticConfig::default()
        };
        let mut ctx = DiagnosticContext::with_config(config);

        for i in 0..10 {
            ctx.report_fusion_error(&FusionError::KernelNotFound(format!("kernel_{}", i)));
        }

        assert_eq!(ctx.error_count(), 2);
    }

    #[test]
    fn test_diagnostic_conversion_from_fusion_error() {
        let err = FusionError::InvalidTransformation("bad fusion".into());
        let diag: GpuDiagnostic = err.into();

        assert!(diag.is_error());
        assert!(!diag.hints.is_empty());
    }
}
