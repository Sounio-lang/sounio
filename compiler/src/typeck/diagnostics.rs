//! Compatibility Diagnostics
//!
//! Rich error messages showing semantic distance breakdown,
//! threshold source, and actionable suggestions.

use std::fmt;

use crate::common::Span;
use crate::hir::HirType;

use super::suggestions::ScoredSuggestion;
use super::threshold::{ResolvedThreshold, ThresholdContext, ThresholdSource};
use super::unify_distance::{CoercionKind, UnificationError, UnificationErrorKind};

/// Severity of a diagnostic
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticSeverity {
    /// Informational note
    Note,
    /// Warning (compiles but may be unintended)
    Warning,
    /// Error (does not compile)
    Error,
}

/// A compatibility diagnostic with full context
#[derive(Debug, Clone)]
pub struct CompatibilityDiagnostic {
    /// Severity
    pub severity: DiagnosticSeverity,
    /// Primary message
    pub message: String,
    /// Location in source
    pub span: Span,
    /// Expected type
    pub expected: HirType,
    /// Found type
    pub found: HirType,
    /// Semantic distance details (if applicable)
    pub distance_details: Option<DistanceDetails>,
    /// Threshold information
    pub threshold_info: Option<ThresholdInfo>,
    /// Suggestions for fixing
    pub suggestions: Vec<DiagnosticSuggestion>,
    /// Related notes
    pub notes: Vec<DiagnosticNote>,
}

/// Details about semantic distance calculation
#[derive(Debug, Clone)]
pub struct DistanceDetails {
    /// Computed distance value
    pub distance: f32,
    /// Breakdown by component
    pub breakdown: Vec<DistanceComponent>,
    /// Confidence in the distance calculation
    pub confidence: f32,
}

/// A component of the distance calculation
#[derive(Debug, Clone)]
pub struct DistanceComponent {
    /// Name of this component
    pub name: String,
    /// Value (contribution to total distance)
    pub value: f32,
    /// Weight applied
    pub weight: f32,
    /// Human-readable explanation
    pub explanation: String,
}

/// Information about threshold
#[derive(Debug, Clone)]
pub struct ThresholdInfo {
    /// The threshold value
    pub threshold: f32,
    /// Where it came from
    pub source: ThresholdSource,
    /// Human-readable source description
    pub source_description: String,
}

/// A suggested fix
#[derive(Debug, Clone)]
pub struct DiagnosticSuggestion {
    /// Short description
    pub message: String,
    /// Code to insert/replace
    pub replacement: Option<SuggestedCode>,
    /// Priority (higher = better suggestion)
    pub priority: i32,
}

/// Suggested code replacement
#[derive(Debug, Clone)]
pub struct SuggestedCode {
    /// Span to replace
    pub span: Span,
    /// New code
    pub code: String,
}

/// A related note
#[derive(Debug, Clone)]
pub struct DiagnosticNote {
    /// Note message
    pub message: String,
    /// Optional location
    pub span: Option<Span>,
}

impl CompatibilityDiagnostic {
    /// Create from unification error
    pub fn from_unification_error(
        error: &UnificationError,
        suggestions: Vec<ScoredSuggestion>,
        threshold: Option<&ResolvedThreshold>,
    ) -> Self {
        let (message, severity) = match &error.kind {
            UnificationErrorKind::StructuralMismatch => (
                format!(
                    "type mismatch: expected `{}`, found `{}`",
                    format_type(&error.expected),
                    format_type(&error.found)
                ),
                DiagnosticSeverity::Error,
            ),

            UnificationErrorKind::DistanceExceeded { threshold, actual } => (
                format!(
                    "types `{}` and `{}` are semantically incompatible (distance {:.3} exceeds threshold {:.3})",
                    format_type(&error.expected),
                    format_type(&error.found),
                    actual,
                    threshold
                ),
                DiagnosticSeverity::Error,
            ),

            UnificationErrorKind::NoCoercionPath => (
                format!(
                    "no conversion exists from `{}` to `{}`",
                    format_type(&error.found),
                    format_type(&error.expected)
                ),
                DiagnosticSeverity::Error,
            ),

            UnificationErrorKind::AmbiguousCoercion => (
                format!(
                    "ambiguous conversion from `{}` to `{}`",
                    format_type(&error.found),
                    format_type(&error.expected)
                ),
                DiagnosticSeverity::Error,
            ),

            UnificationErrorKind::InfiniteType => (
                "cannot construct infinite type".to_string(),
                DiagnosticSeverity::Error,
            ),
        };

        // Build distance details from error
        let distance_details = error.distance_info.as_ref().map(|info| {
            DistanceDetails {
                distance: info.distance,
                breakdown: Vec::new(), // Breakdown computed separately if needed
                confidence: info.confidence,
            }
        });

        // Build threshold info
        let threshold_info = threshold.map(|t| ThresholdInfo {
            threshold: t.as_f32(),
            source: t.source.clone(),
            source_description: describe_threshold_source(&t.source),
        });

        // Convert suggestions
        let diagnostic_suggestions = suggestions
            .into_iter()
            .map(|s| DiagnosticSuggestion {
                message: s.reason,
                replacement: Some(SuggestedCode {
                    span: error.span,
                    code: format_type(&s.suggested_type),
                }),
                priority: (s.score * 100.0) as i32,
            })
            .collect();

        // Build notes
        let mut notes = Vec::new();

        // Add note about threshold source
        if let Some(info) = &threshold_info {
            notes.push(DiagnosticNote {
                message: format!(
                    "threshold of {:.3} comes from {}",
                    info.threshold, info.source_description
                ),
                span: None,
            });
        }

        // Add note about distance components
        if let Some(details) = &distance_details
            && !details.breakdown.is_empty()
        {
            let components: Vec<String> = details
                .breakdown
                .iter()
                .map(|c| format!("{}: {:.3}", c.name, c.value * c.weight))
                .collect();
            notes.push(DiagnosticNote {
                message: format!("distance breakdown: {}", components.join(", ")),
                span: None,
            });
        }

        Self {
            severity,
            message,
            span: error.span,
            expected: error.expected.clone(),
            found: error.found.clone(),
            distance_details,
            threshold_info,
            suggestions: diagnostic_suggestions,
            notes,
        }
    }

    /// Create a warning for a close-to-threshold coercion
    pub fn coercion_warning(
        span: Span,
        from: &HirType,
        to: &HirType,
        distance: f32,
        threshold: f32,
        kind: CoercionKind,
    ) -> Self {
        let margin = threshold - distance;
        let message = format!(
            "implicit {} coercion from `{}` to `{}` (distance {:.3}, margin {:.3})",
            coercion_kind_name(kind),
            format_type(from),
            format_type(to),
            distance,
            margin
        );

        let mut notes = vec![DiagnosticNote {
            message: "this coercion is close to the threshold; consider using explicit conversion"
                .to_string(),
            span: None,
        }];

        if kind == CoercionKind::CrossOntology {
            notes.push(DiagnosticNote {
                message: "cross-ontology coercions may have subtle semantic differences"
                    .to_string(),
                span: None,
            });
        }

        Self {
            severity: DiagnosticSeverity::Warning,
            message,
            span,
            expected: to.clone(),
            found: from.clone(),
            distance_details: Some(DistanceDetails {
                distance,
                breakdown: Vec::new(),
                confidence: 0.0,
            }),
            threshold_info: None,
            suggestions: vec![DiagnosticSuggestion {
                message: format!(
                    "consider using explicit `as {}` conversion",
                    format_type(to)
                ),
                replacement: None,
                priority: 50,
            }],
            notes,
        }
    }

    /// Add a note
    pub fn with_note(mut self, message: impl Into<String>) -> Self {
        self.notes.push(DiagnosticNote {
            message: message.into(),
            span: None,
        });
        self
    }

    /// Add a note with span
    pub fn with_note_at(mut self, message: impl Into<String>, span: Span) -> Self {
        self.notes.push(DiagnosticNote {
            message: message.into(),
            span: Some(span),
        });
        self
    }

    /// Add a suggestion
    pub fn with_suggestion(mut self, message: impl Into<String>, priority: i32) -> Self {
        self.suggestions.push(DiagnosticSuggestion {
            message: message.into(),
            replacement: None,
            priority,
        });
        self
    }

    /// Format as a complete diagnostic message
    pub fn format(&self) -> String {
        let mut output = String::new();

        // Severity prefix
        let prefix = match self.severity {
            DiagnosticSeverity::Error => "error",
            DiagnosticSeverity::Warning => "warning",
            DiagnosticSeverity::Note => "note",
        };

        // Main message
        output.push_str(&format!("{}: {}\n", prefix, self.message));

        // Location (byte offsets)
        output.push_str(&format!(
            "  --> byte {}..{}\n",
            self.span.start, self.span.end
        ));

        // Distance details
        if let Some(details) = &self.distance_details {
            output.push_str(&format!(
                "   = semantic distance: {:.3} (confidence: {:.1}%)\n",
                details.distance,
                details.confidence * 100.0
            ));

            for component in &details.breakdown {
                output.push_str(&format!(
                    "     - {}: {:.3} (weight {:.2}) - {}\n",
                    component.name, component.value, component.weight, component.explanation
                ));
            }
        }

        // Notes
        for note in &self.notes {
            if let Some(span) = &note.span {
                output.push_str(&format!(
                    "   = note: {} (at byte {}..{})\n",
                    note.message, span.start, span.end
                ));
            } else {
                output.push_str(&format!("   = note: {}\n", note.message));
            }
        }

        // Suggestions
        if !self.suggestions.is_empty() {
            output.push_str("   = help:\n");
            for suggestion in self.suggestions.iter().take(3) {
                output.push_str(&format!("     - {}\n", suggestion.message));
                if let Some(replacement) = &suggestion.replacement {
                    output.push_str(&format!("       replace with: `{}`\n", replacement.code));
                }
            }
        }

        output
    }
}

impl fmt::Display for CompatibilityDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Format a type for display
fn format_type(ty: &HirType) -> String {
    match ty {
        HirType::Unit => "()".to_string(),
        HirType::Bool => "bool".to_string(),
        HirType::I32 => "i32".to_string(),
        HirType::I64 => "i64".to_string(),
        HirType::F32 => "f32".to_string(),
        HirType::F64 => "f64".to_string(),
        HirType::String => "String".to_string(),
        HirType::Var(id) => format!("?T{}", id),
        HirType::Named { name, args } if args.is_empty() => name.clone(),
        HirType::Named { name, args } => {
            let args_str: Vec<_> = args.iter().map(format_type).collect();
            format!("{}<{}>", name, args_str.join(", "))
        }
        HirType::Ontology { namespace, term } => format!("{}:{}", namespace, term),
        HirType::Quantity { numeric, unit } => {
            format!("{}@{}", format_type(numeric), unit.format())
        }
        HirType::Fn {
            params,
            return_type,
        } => {
            let params_str: Vec<_> = params.iter().map(format_type).collect();
            format!(
                "fn({}) -> {}",
                params_str.join(", "),
                format_type(return_type)
            )
        }
        HirType::Tuple(elems) => {
            let elems_str: Vec<_> = elems.iter().map(format_type).collect();
            format!("({})", elems_str.join(", "))
        }
        HirType::Ref { mutable, inner } => {
            if *mutable {
                format!("&mut {}", format_type(inner))
            } else {
                format!("&{}", format_type(inner))
            }
        }
        HirType::RawPointer { mutable, inner } => {
            if *mutable {
                format!("*mut {}", format_type(inner))
            } else {
                format!("*const {}", format_type(inner))
            }
        }
        HirType::Array { element, size } => match size {
            Some(s) => format!("[{}; {}]", format_type(element), s),
            None => format!("[{}]", format_type(element)),
        },
        HirType::Knowledge {
            inner,
            epsilon_bound,
            ..
        } => {
            if let Some(eps) = epsilon_bound {
                format!("Knowledge[{}, ε={}]", format_type(inner), eps)
            } else {
                format!("Knowledge[{}]", format_type(inner))
            }
        }
        HirType::Error => "<error>".to_string(),
        HirType::Never => "!".to_string(),
        HirType::Char => "char".to_string(),
        HirType::I8 => "i8".to_string(),
        HirType::I16 => "i16".to_string(),
        HirType::I128 => "i128".to_string(),
        HirType::Isize => "isize".to_string(),
        HirType::U8 => "u8".to_string(),
        HirType::U16 => "u16".to_string(),
        HirType::U32 => "u32".to_string(),
        HirType::U64 => "u64".to_string(),
        HirType::U128 => "u128".to_string(),
        HirType::Usize => "usize".to_string(),
        HirType::Tensor { element, dims } => {
            let dims_str: Vec<_> = dims
                .iter()
                .map(|d| match d {
                    crate::hir::HirTensorDim::Named(n) => n.clone(),
                    crate::hir::HirTensorDim::Fixed(s) => s.to_string(),
                    crate::hir::HirTensorDim::Dynamic => "?".to_string(),
                })
                .collect();
            format!(
                "Tensor[{}, ({})]",
                format_type(element),
                dims_str.join(", ")
            )
        }
        // Linear algebra primitives
        HirType::Vec2 => "vec2".to_string(),
        HirType::Vec3 => "vec3".to_string(),
        HirType::Vec4 => "vec4".to_string(),
        HirType::Mat2 => "mat2".to_string(),
        HirType::Mat3 => "mat3".to_string(),
        HirType::Mat4 => "mat4".to_string(),
        HirType::Quat => "quat".to_string(),
        // Automatic differentiation
        HirType::Dual => "dual".to_string(),
    }
}

/// Describe threshold source for humans
fn describe_threshold_source(source: &ThresholdSource) -> String {
    match source {
        ThresholdSource::ItemAnnotation => "annotation on this function".to_string(),
        ThresholdSource::ParameterAnnotation => "annotation on this parameter".to_string(),
        ThresholdSource::TypeParameterAnnotation => "annotation on type parameter".to_string(),
        ThresholdSource::ModuleDefault => "module-level default".to_string(),
        ThresholdSource::GlobalDefault => "global default".to_string(),
        ThresholdSource::Inferred(ctx) => match ctx {
            ThresholdContext::FunctionParameter => "parameter position (inferred)".to_string(),
            ThresholdContext::ReturnType => "return type position (inferred)".to_string(),
            ThresholdContext::LocalAssignment => "local variable assignment (inferred)".to_string(),
            ThresholdContext::FieldAccess => "field access (inferred)".to_string(),
            ThresholdContext::MethodReceiver => "method receiver (inferred)".to_string(),
            ThresholdContext::GenericArgument => "generic argument (inferred)".to_string(),
            ThresholdContext::MatchPattern => "match pattern (inferred)".to_string(),
        },
    }
}

/// Get human-readable name for coercion kind
fn coercion_kind_name(kind: CoercionKind) -> &'static str {
    match kind {
        CoercionKind::Subtype => "subtype",
        CoercionKind::SemanticProximity => "semantic",
        CoercionKind::CrossOntology => "cross-ontology",
        CoercionKind::ExplicitCast => "explicit",
    }
}

/// Diagnostic accumulator for type checking
#[derive(Default)]
pub struct DiagnosticAccumulator {
    diagnostics: Vec<CompatibilityDiagnostic>,
}

impl DiagnosticAccumulator {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Add a diagnostic
    pub fn add(&mut self, diagnostic: CompatibilityDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Add error
    pub fn error(&mut self, diagnostic: CompatibilityDiagnostic) {
        debug_assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
        self.diagnostics.push(diagnostic);
    }

    /// Add warning
    pub fn warning(&mut self, diagnostic: CompatibilityDiagnostic) {
        debug_assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
        self.diagnostics.push(diagnostic);
    }

    /// Has errors?
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error)
    }

    /// Get all diagnostics
    pub fn diagnostics(&self) -> &[CompatibilityDiagnostic] {
        &self.diagnostics
    }

    /// Take all diagnostics
    pub fn take_diagnostics(&mut self) -> Vec<CompatibilityDiagnostic> {
        std::mem::take(&mut self.diagnostics)
    }

    /// Count by severity
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Warning)
            .count()
    }

    /// Format all diagnostics
    pub fn format_all(&self) -> String {
        self.diagnostics
            .iter()
            .map(|d| d.format())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_span() -> Span {
        Span::new(0, 10)
    }

    #[test]
    fn test_format_types() {
        assert_eq!(format_type(&HirType::I64), "i64");
        assert_eq!(format_type(&HirType::String), "String");
        assert_eq!(
            format_type(&HirType::Ontology {
                namespace: "pbpk".to_string(),
                term: "Concentration".to_string()
            }),
            "pbpk:Concentration"
        );
    }

    #[test]
    fn test_coercion_warning() {
        let diagnostic = CompatibilityDiagnostic::coercion_warning(
            test_span(),
            &HirType::Ontology {
                namespace: "pbpk".to_string(),
                term: "Concentration".to_string(),
            },
            &HirType::Ontology {
                namespace: "pbpk".to_string(),
                term: "PlasmaConcentration".to_string(),
            },
            0.12,
            0.15,
            CoercionKind::SemanticProximity,
        );

        assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
        assert!(diagnostic.message.contains("semantic"));
        assert!(diagnostic.message.contains("0.12"));
    }

    #[test]
    fn test_diagnostic_accumulator() {
        let mut acc = DiagnosticAccumulator::new();

        acc.add(CompatibilityDiagnostic::coercion_warning(
            test_span(),
            &HirType::I32,
            &HirType::I64,
            0.0,
            0.15,
            CoercionKind::Subtype,
        ));

        assert!(!acc.has_errors());
        assert_eq!(acc.warning_count(), 1);
        assert_eq!(acc.error_count(), 0);
    }
}
