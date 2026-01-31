//! Diagnostics provider
//!
//! Converts compiler errors to LSP diagnostics.
//! Includes specialized diagnostics for epistemic types, effects, and units.

use tower_lsp::lsp_types::*;

use super::query_bridge::{EffectInfo, EpistemicInfo, UnitInfo};

/// Provider for diagnostics
pub struct DiagnosticsProvider;

impl DiagnosticsProvider {
    /// Create a new diagnostics provider
    pub fn new() -> Self {
        Self
    }

    /// Convert a miette error to an LSP diagnostic
    pub fn miette_to_diagnostic(&self, err: &miette::Report, source: &str) -> Diagnostic {
        // Extract error message
        let message = err.to_string();

        // Try to extract span information from the error
        // For now, we'll create a diagnostic at the beginning of the file
        // In a real implementation, we'd parse the miette error for location info
        let range = self.extract_range_from_error(&message, source);

        // Determine severity
        let severity = Some(DiagnosticSeverity::ERROR);

        // Extract error code if available
        let code = self.extract_error_code(&message);

        Diagnostic {
            range,
            severity,
            code: code.map(NumberOrString::String),
            source: Some("sounio".to_string()),
            message: self.clean_message(&message),
            related_information: None,
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Extract range information from error message
    fn extract_range_from_error(&self, message: &str, source: &str) -> Range {
        // Try to parse position from message
        // Common formats:
        // - "at position X"
        // - "line X, column Y"
        // - "X:Y"

        // Look for "at position X" pattern
        if let Some(pos) = message.find("at position ") {
            let start = pos + "at position ".len();
            let end = message[start..]
                .find(|c: char| !c.is_ascii_digit())
                .map(|e| start + e)
                .unwrap_or(message.len());

            if let Ok(offset) = message[start..end].parse::<usize>() {
                let (line, col) = offset_to_line_col(source, offset);
                return Range {
                    start: Position {
                        line: line as u32,
                        character: col as u32,
                    },
                    end: Position {
                        line: line as u32,
                        character: (col + 1) as u32,
                    },
                };
            }
        }

        // Default to start of file
        Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end: Position {
                line: 0,
                character: 1,
            },
        }
    }

    /// Extract error code from message
    fn extract_error_code(&self, message: &str) -> Option<String> {
        // Look for patterns like "E0001" or "[E0001]"
        let msg = message.to_uppercase();

        for pattern in &[
            "[E", "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9",
        ] {
            if let Some(pos) = msg.find(pattern) {
                let start = if msg[pos..].starts_with('[') {
                    pos + 1
                } else {
                    pos
                };
                let end = msg[start..]
                    .find(|c: char| !c.is_ascii_alphanumeric())
                    .map(|e| start + e)
                    .unwrap_or_else(|| (start + 5).min(msg.len()));

                let code = &message[start..end];
                if code.len() >= 2 && code.chars().next() == Some('E') {
                    return Some(code.to_string());
                }
            }
        }

        None
    }

    /// Clean up error message for display
    fn clean_message(&self, message: &str) -> String {
        // Remove ANSI escape codes if present
        let clean = strip_ansi_codes(message);

        // Remove redundant error prefixes
        let clean = clean
            .trim_start_matches("error: ")
            .trim_start_matches("Error: ")
            .trim_start_matches("error[E")
            .split(']')
            .last()
            .unwrap_or(&clean)
            .trim();

        clean.to_string()
    }

    /// Create a diagnostic from components
    pub fn create_diagnostic(
        &self,
        range: Range,
        message: String,
        severity: DiagnosticSeverity,
        code: Option<String>,
    ) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(severity),
            code: code.map(NumberOrString::String),
            source: Some("sounio".to_string()),
            message,
            related_information: None,
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Create an error diagnostic
    pub fn error(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        self.create_diagnostic(range, message.into(), DiagnosticSeverity::ERROR, None)
    }

    /// Create a warning diagnostic
    pub fn warning(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        self.create_diagnostic(range, message.into(), DiagnosticSeverity::WARNING, None)
    }

    /// Create an info diagnostic
    pub fn info(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        self.create_diagnostic(range, message.into(), DiagnosticSeverity::INFORMATION, None)
    }

    /// Create a hint diagnostic
    pub fn hint(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        self.create_diagnostic(range, message.into(), DiagnosticSeverity::HINT, None)
    }

    /// Create a deprecated diagnostic
    pub fn deprecated(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::HINT),
            code: None,
            source: Some("sounio".to_string()),
            message: message.into(),
            related_information: None,
            tags: Some(vec![DiagnosticTag::DEPRECATED]),
            code_description: None,
            data: None,
        }
    }

    /// Create an unused diagnostic
    pub fn unused(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::HINT),
            code: None,
            source: Some("sounio".to_string()),
            message: message.into(),
            related_information: None,
            tags: Some(vec![DiagnosticTag::UNNECESSARY]),
            code_description: None,
            data: None,
        }
    }

    // =========================================================================
    // Epistemic Diagnostics
    // =========================================================================

    /// Default confidence threshold for warnings
    pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.75;

    /// Check epistemic info and generate diagnostics
    pub fn epistemic_diagnostics(
        &self,
        range: Range,
        var_name: &str,
        info: &EpistemicInfo,
        threshold: Option<f64>,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let threshold = threshold.unwrap_or(Self::DEFAULT_CONFIDENCE_THRESHOLD);

        // Check for low confidence
        if let Some(diag) = self.check_confidence_degradation(range, var_name, info, threshold) {
            diagnostics.push(diag);
        }

        // Check for missing provenance on derived values
        if let Some(diag) = self.check_missing_provenance(range, var_name, info) {
            diagnostics.push(diag);
        }

        // Check for missing evidence
        if let Some(diag) = self.check_missing_evidence(range, var_name, info) {
            diagnostics.push(diag);
        }

        diagnostics
    }

    /// Warn when confidence drops below threshold
    fn check_confidence_degradation(
        &self,
        range: Range,
        var_name: &str,
        info: &EpistemicInfo,
        threshold: f64,
    ) -> Option<Diagnostic> {
        if info.confidence < threshold {
            let percent = (info.confidence * 100.0).round() as u32;
            let threshold_percent = (threshold * 100.0).round() as u32;

            let severity = if info.confidence < 0.5 {
                DiagnosticSeverity::WARNING
            } else {
                DiagnosticSeverity::HINT
            };

            let message = format!(
                "Low confidence: '{}' has {}% confidence (threshold: {}%)",
                var_name, percent, threshold_percent
            );

            Some(Diagnostic {
                range,
                severity: Some(severity),
                code: Some(NumberOrString::String("E1001".to_string())),
                source: Some("sounio-epistemic".to_string()),
                message,
                related_information: Some(vec![DiagnosticRelatedInformation {
                    location: Location {
                        uri: Url::parse("file:///").unwrap(),
                        range,
                    },
                    message: format!(
                        "Consider adding evidence to increase confidence, or use 'provisional' to mark as tentative"
                    ),
                }]),
                tags: None,
                code_description: None,
                data: None,
            })
        } else {
            None
        }
    }

    /// Warn when derived values lack provenance information
    fn check_missing_provenance(
        &self,
        range: Range,
        var_name: &str,
        info: &EpistemicInfo,
    ) -> Option<Diagnostic> {
        // Only warn for derived values (not axioms or direct measurements)
        if info.source != "derived" && info.source != "computed" {
            return None;
        }

        if info.provenance.is_none() {
            let message = format!(
                "Missing provenance: derived value '{}' lacks lineage information",
                var_name
            );

            return Some(Diagnostic {
                range,
                severity: Some(DiagnosticSeverity::HINT),
                code: Some(NumberOrString::String("E1002".to_string())),
                source: Some("sounio-epistemic".to_string()),
                message,
                related_information: Some(vec![DiagnosticRelatedInformation {
                    location: Location {
                        uri: Url::parse("file:///").unwrap(),
                        range,
                    },
                    message: "Use 'with_provenance(origin: ...)' to track data lineage".to_string(),
                }]),
                tags: None,
                code_description: None,
                data: None,
            });
        }

        // Check for empty dependencies in provenance
        if let Some(ref prov) = info.provenance {
            if prov.dependencies.is_empty() && prov.transformations.is_empty() {
                let message = format!(
                    "Incomplete provenance: '{}' marked as derived but has no recorded dependencies",
                    var_name
                );

                return Some(Diagnostic {
                    range,
                    severity: Some(DiagnosticSeverity::HINT),
                    code: Some(NumberOrString::String("E1003".to_string())),
                    source: Some("sounio-epistemic".to_string()),
                    message,
                    related_information: None,
                    tags: None,
                    code_description: None,
                    data: None,
                });
            }
        }

        None
    }

    /// Suggest adding evidence when revisable claims lack citations
    fn check_missing_evidence(
        &self,
        range: Range,
        var_name: &str,
        info: &EpistemicInfo,
    ) -> Option<Diagnostic> {
        // Only suggest evidence for revisable claims without any
        if !info.revisable || !info.evidence.is_empty() {
            return None;
        }

        // Skip axioms (they don't need external evidence)
        if info.source == "axiomatic" {
            return None;
        }

        let message = format!(
            "Consider adding evidence: revisable claim '{}' has no supporting citations",
            var_name
        );

        Some(Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::HINT),
            code: Some(NumberOrString::String("E1004".to_string())),
            source: Some("sounio-epistemic".to_string()),
            message,
            related_information: Some(vec![DiagnosticRelatedInformation {
                location: Location {
                    uri: Url::parse("file:///").unwrap(),
                    range,
                },
                message: "Use 'with_evidence(citation: \"...\")' to add supporting evidence"
                    .to_string(),
            }]),
            tags: None,
            code_description: None,
            data: None,
        })
    }

    /// Create diagnostic for confidence interval issues
    pub fn confidence_interval_warning(
        &self,
        range: Range,
        var_name: &str,
        lower: f64,
        upper: f64,
    ) -> Diagnostic {
        let spread = upper - lower;
        let message = if spread > 0.5 {
            format!(
                "Wide confidence interval: '{}' ranges from {:.0}% to {:.0}% (spread: {:.0}%)",
                var_name,
                lower * 100.0,
                upper * 100.0,
                spread * 100.0
            )
        } else {
            format!(
                "Confidence interval for '{}': {:.0}% to {:.0}%",
                var_name,
                lower * 100.0,
                upper * 100.0
            )
        };

        Diagnostic {
            range,
            severity: Some(if spread > 0.5 {
                DiagnosticSeverity::WARNING
            } else {
                DiagnosticSeverity::INFORMATION
            }),
            code: Some(NumberOrString::String("E1005".to_string())),
            source: Some("sounio-epistemic".to_string()),
            message,
            related_information: None,
            tags: None,
            code_description: None,
            data: None,
        }
    }

    // =========================================================================
    // Effect Diagnostics (Phase 4 placeholder)
    // =========================================================================

    /// Check for undeclared effects
    #[allow(dead_code)]
    pub fn undeclared_effect(
        &self,
        range: Range,
        effect_name: &str,
        function_name: &str,
    ) -> Diagnostic {
        let message = format!(
            "Undeclared effect: function '{}' uses '{}' effect but doesn't declare it",
            function_name, effect_name
        );

        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String("E0006".to_string())),
            source: Some("sounio-effects".to_string()),
            message,
            related_information: Some(vec![DiagnosticRelatedInformation {
                location: Location {
                    uri: Url::parse("file:///").unwrap(),
                    range,
                },
                message: format!("Add 'with {}' to the function signature", effect_name),
            }]),
            tags: None,
            code_description: None,
            data: None,
        }
    }

    // =========================================================================
    // Unit Diagnostics (Phase 5)
    // =========================================================================

    /// Warn on dimensional mismatch with helpful suggestions
    pub fn dimension_mismatch(
        &self,
        range: Range,
        expected: &str,
        expected_dim: &str,
        got: &str,
        got_dim: &str,
    ) -> Diagnostic {
        let message = format!(
            "Dimensional mismatch: expected '{}' ({}), got '{}' ({})",
            expected, expected_dim, got, got_dim
        );

        // Generate suggestions based on the dimensional difference
        let suggestion = self.suggest_dimension_fix(expected_dim, got_dim);

        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String("E0007".to_string())),
            source: Some("sounio-units".to_string()),
            message,
            related_information: suggestion.map(|s| {
                vec![DiagnosticRelatedInformation {
                    location: Location {
                        uri: Url::parse("file:///").unwrap(),
                        range,
                    },
                    message: s,
                }]
            }),
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Suggest a fix for dimensional mismatch
    fn suggest_dimension_fix(&self, expected: &str, got: &str) -> Option<String> {
        // Common dimensional fixes
        let fixes = [
            // Velocity = distance / time
            (
                ("M", "L/T"),
                "Did you mean to divide by time? (distance / time = velocity)",
            ),
            (
                ("L/T", "M"),
                "Did you mean to multiply by time? (velocity × time = distance)",
            ),
            // Force = mass × acceleration
            (
                ("M·L/T²", "M"),
                "Did you mean to multiply by acceleration? (mass × acceleration = force)",
            ),
            (
                ("M", "M·L/T²"),
                "Did you mean to divide by acceleration? (force / acceleration = mass)",
            ),
            // Energy = force × distance
            (
                ("M·L²/T²", "M·L/T²"),
                "Did you mean to multiply by distance? (force × distance = energy)",
            ),
            // Power = energy / time
            (
                ("M·L²/T³", "M·L²/T²"),
                "Did you mean to divide by time? (energy / time = power)",
            ),
            // Pressure = force / area
            (
                ("M/L·T²", "M·L/T²"),
                "Did you mean to divide by area? (force / area = pressure)",
            ),
            // Density = mass / volume
            (
                ("M/L³", "M"),
                "Did you mean to divide by volume? (mass / volume = density)",
            ),
        ];

        for ((exp, g), suggestion) in fixes {
            if (expected == exp && got == g) || (expected == g && got == exp) {
                return Some(suggestion.to_string());
            }
        }

        // Generic suggestion
        Some(format!(
            "Check your formula: expected dimension {} but got {}",
            expected, got
        ))
    }

    /// Check unit info and generate diagnostics
    pub fn unit_diagnostics(
        &self,
        range: Range,
        expr_name: &str,
        info: &UnitInfo,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        // Check for dimensionless when dimension expected
        if info.is_dimensionless && !info.unit.is_empty() && info.unit != "1" {
            diagnostics.push(self.dimensionless_warning(range, expr_name));
        }

        diagnostics
    }

    /// Warning for unexpected dimensionless value
    pub fn dimensionless_warning(&self, range: Range, expr_name: &str) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::WARNING),
            code: Some(NumberOrString::String("E0008".to_string())),
            source: Some("sounio-units".to_string()),
            message: format!(
                "Dimensionless result: '{}' produced a dimensionless value, which may indicate a unit cancellation error",
                expr_name
            ),
            related_information: Some(vec![DiagnosticRelatedInformation {
                location: Location {
                    uri: Url::parse("file:///").unwrap(),
                    range,
                },
                message: "Check if this is intentional. Dimensionless values often indicate missing or extra unit terms.".to_string(),
            }]),
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Suggest adding unit annotation
    pub fn missing_unit_annotation(
        &self,
        range: Range,
        var_name: &str,
        inferred_unit: &str,
    ) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::HINT),
            code: Some(NumberOrString::String("E0009".to_string())),
            source: Some("sounio-units".to_string()),
            message: format!(
                "Consider adding unit annotation: '{}' has inferred unit '{}'",
                var_name, inferred_unit
            ),
            related_information: Some(vec![DiagnosticRelatedInformation {
                location: Location {
                    uri: Url::parse("file:///").unwrap(),
                    range,
                },
                message: format!(
                    "Add type annotation: let {}: {} = ...",
                    var_name, inferred_unit
                ),
            }]),
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Warning for implicit unit conversion
    pub fn implicit_unit_conversion(
        &self,
        range: Range,
        from_unit: &str,
        to_unit: &str,
        conversion_factor: f64,
    ) -> Diagnostic {
        let message = format!(
            "Implicit unit conversion: '{}' → '{}' (factor: {})",
            from_unit, to_unit, conversion_factor
        );

        let severity = if (conversion_factor - 1.0).abs() > 100.0 {
            // Large conversion factor might indicate mistake
            DiagnosticSeverity::WARNING
        } else {
            DiagnosticSeverity::INFORMATION
        };

        Diagnostic {
            range,
            severity: Some(severity),
            code: Some(NumberOrString::String("E0010".to_string())),
            source: Some("sounio-units".to_string()),
            message,
            related_information: Some(vec![DiagnosticRelatedInformation {
                location: Location {
                    uri: Url::parse("file:///").unwrap(),
                    range,
                },
                message: format!(
                    "Use explicit conversion: value.to_{}() or value.as_{}()",
                    to_unit, to_unit
                ),
            }]),
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Error for incompatible unit operation
    pub fn incompatible_unit_operation(
        &self,
        range: Range,
        operation: &str,
        left_unit: &str,
        right_unit: &str,
    ) -> Diagnostic {
        let message = format!(
            "Cannot {} values with incompatible units: '{}' and '{}'",
            operation, left_unit, right_unit
        );

        let suggestion = match operation {
            "add" | "subtract" => {
                format!(
                    "Convert one operand to match: use .to_{}() or .to_{}()",
                    left_unit, right_unit
                )
            }
            _ => "Ensure units are compatible for this operation".to_string(),
        };

        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String("E0011".to_string())),
            source: Some("sounio-units".to_string()),
            message,
            related_information: Some(vec![DiagnosticRelatedInformation {
                location: Location {
                    uri: Url::parse("file:///").unwrap(),
                    range,
                },
                message: suggestion,
            }]),
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Warning for potential unit confusion (e.g., mph vs km/h)
    pub fn unit_confusion_warning(
        &self,
        range: Range,
        unit1: &str,
        unit2: &str,
        context: &str,
    ) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::WARNING),
            code: Some(NumberOrString::String("E0012".to_string())),
            source: Some("sounio-units".to_string()),
            message: format!(
                "Potential unit confusion: mixing '{}' and '{}' in {}",
                unit1, unit2, context
            ),
            related_information: Some(vec![DiagnosticRelatedInformation {
                location: Location {
                    uri: Url::parse("file:///").unwrap(),
                    range,
                },
                message: "Consider using consistent units throughout or explicit conversions"
                    .to_string(),
            }]),
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Information about SI unit equivalence
    pub fn si_equivalence_info(
        &self,
        range: Range,
        custom_unit: &str,
        si_equivalent: &str,
    ) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::INFORMATION),
            code: None,
            source: Some("sounio-units".to_string()),
            message: format!(
                "Unit '{}' is equivalent to SI unit '{}'",
                custom_unit, si_equivalent
            ),
            related_information: None,
            tags: None,
            code_description: None,
            data: None,
        }
    }
}

impl Default for DiagnosticsProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert byte offset to line/column
fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let mut line = 0;
    let mut col = 0;

    for (i, c) in source.char_indices() {
        if i >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    (line, col)
}

/// Strip ANSI escape codes from a string
fn strip_ansi_codes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip escape sequence
            if chars.peek() == Some(&'[') {
                chars.next();
                // Skip until 'm' (end of ANSI sequence)
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next == 'm' {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::super::query_bridge::ProvenanceInfo;
    use super::*;

    fn test_range() -> Range {
        Range {
            start: Position::new(0, 0),
            end: Position::new(0, 10),
        }
    }

    #[test]
    fn test_strip_ansi_codes() {
        let input = "\x1b[31mError\x1b[0m: something went wrong";
        let output = strip_ansi_codes(input);
        assert_eq!(output, "Error: something went wrong");
    }

    #[test]
    fn test_offset_to_line_col() {
        let source = "line 1\nline 2\nline 3";
        assert_eq!(offset_to_line_col(source, 0), (0, 0));
        assert_eq!(offset_to_line_col(source, 5), (0, 5));
        assert_eq!(offset_to_line_col(source, 7), (1, 0));
        assert_eq!(offset_to_line_col(source, 14), (2, 0));
    }

    // =========================================================================
    // Epistemic Diagnostic Tests
    // =========================================================================

    #[test]
    fn test_low_confidence_warning() {
        let provider = DiagnosticsProvider::new();
        let info = EpistemicInfo {
            confidence: 0.45,
            source: "empirical".to_string(),
            revisable: true,
            ..Default::default()
        };

        let diagnostics = provider.epistemic_diagnostics(test_range(), "measurement", &info, None);

        assert_eq!(diagnostics.len(), 2); // Low confidence + missing evidence
        let diag = &diagnostics[0];
        assert!(diag.message.contains("45%"));
        assert!(diag.message.contains("confidence"));
        assert_eq!(diag.severity, Some(DiagnosticSeverity::WARNING));
    }

    #[test]
    fn test_high_confidence_no_warning() {
        let provider = DiagnosticsProvider::new();
        let info = EpistemicInfo {
            confidence: 0.95,
            source: "axiomatic".to_string(),
            revisable: false,
            ..Default::default()
        };

        let diagnostics = provider.epistemic_diagnostics(test_range(), "constant", &info, None);

        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_custom_confidence_threshold() {
        let provider = DiagnosticsProvider::new();
        let info = EpistemicInfo {
            confidence: 0.80,
            source: "axiomatic".to_string(),
            revisable: false,
            ..Default::default()
        };

        // With default threshold (0.75), no warning
        let diagnostics = provider.epistemic_diagnostics(test_range(), "value", &info, Some(0.75));
        assert!(diagnostics.is_empty());

        // With higher threshold (0.90), should warn
        let diagnostics = provider.epistemic_diagnostics(test_range(), "value", &info, Some(0.90));
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn test_missing_provenance_on_derived() {
        let provider = DiagnosticsProvider::new();
        let info = EpistemicInfo {
            confidence: 0.90,
            source: "derived".to_string(),
            revisable: false,
            provenance: None, // Missing!
            ..Default::default()
        };

        let diagnostics = provider.epistemic_diagnostics(test_range(), "result", &info, None);

        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("provenance"));
        assert!(diagnostics[0].message.contains("lineage"));
    }

    #[test]
    fn test_no_provenance_warning_for_axiomatic() {
        let provider = DiagnosticsProvider::new();
        let info = EpistemicInfo {
            confidence: 0.99,
            source: "axiomatic".to_string(),
            revisable: false,
            provenance: None,
            ..Default::default()
        };

        let diagnostics = provider.epistemic_diagnostics(test_range(), "pi", &info, None);

        // Axiomatic values don't need provenance
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_incomplete_provenance_warning() {
        let provider = DiagnosticsProvider::new();
        let info = EpistemicInfo {
            confidence: 0.90,
            source: "derived".to_string(),
            revisable: false,
            provenance: Some(ProvenanceInfo {
                origin: "calculation".to_string(),
                transformations: vec![],
                dependencies: vec![], // Empty dependencies
            }),
            ..Default::default()
        };

        let diagnostics = provider.epistemic_diagnostics(test_range(), "computed", &info, None);

        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("Incomplete provenance"));
    }

    #[test]
    fn test_missing_evidence_suggestion() {
        let provider = DiagnosticsProvider::new();
        let info = EpistemicInfo {
            confidence: 0.85,
            source: "empirical".to_string(),
            revisable: true,
            evidence: vec![], // No evidence
            ..Default::default()
        };

        let diagnostics = provider.epistemic_diagnostics(test_range(), "hypothesis", &info, None);

        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("evidence"));
        assert!(diagnostics[0].message.contains("citations"));
    }

    #[test]
    fn test_no_evidence_warning_for_nonrevisable() {
        let provider = DiagnosticsProvider::new();
        let info = EpistemicInfo {
            confidence: 0.85,
            source: "empirical".to_string(),
            revisable: false, // Not revisable
            evidence: vec![],
            ..Default::default()
        };

        let diagnostics = provider.epistemic_diagnostics(test_range(), "fact", &info, None);

        // Non-revisable claims don't get the evidence suggestion
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_confidence_interval_warning() {
        let provider = DiagnosticsProvider::new();

        // Wide interval should warn
        let diag = provider.confidence_interval_warning(test_range(), "estimate", 0.30, 0.90);
        assert_eq!(diag.severity, Some(DiagnosticSeverity::WARNING));
        assert!(diag.message.contains("Wide confidence interval"));
        assert!(diag.message.contains("60%")); // spread

        // Narrow interval should be INFO
        let diag = provider.confidence_interval_warning(test_range(), "precise", 0.85, 0.95);
        assert_eq!(diag.severity, Some(DiagnosticSeverity::INFORMATION));
    }

    #[test]
    fn test_undeclared_effect() {
        let provider = DiagnosticsProvider::new();
        let diag = provider.undeclared_effect(test_range(), "IO", "write_file");

        assert_eq!(diag.severity, Some(DiagnosticSeverity::ERROR));
        assert!(diag.message.contains("IO"));
        assert!(diag.message.contains("write_file"));
        assert_eq!(diag.code, Some(NumberOrString::String("E0006".to_string())));
    }

    // =========================================================================
    // Unit Diagnostic Tests
    // =========================================================================

    #[test]
    fn test_dimension_mismatch() {
        let provider = DiagnosticsProvider::new();
        let diag = provider.dimension_mismatch(test_range(), "m/s", "L/T", "kg", "M");

        assert_eq!(diag.severity, Some(DiagnosticSeverity::ERROR));
        assert!(diag.message.contains("m/s"));
        assert!(diag.message.contains("L/T"));
        assert!(diag.message.contains("kg"));
        assert!(diag.message.contains("M"));
    }

    #[test]
    fn test_dimension_mismatch_with_suggestion() {
        let provider = DiagnosticsProvider::new();
        // Velocity expected but got mass - should suggest dividing by time
        let diag = provider.dimension_mismatch(test_range(), "velocity", "L/T", "mass", "M");

        assert!(diag.related_information.is_some());
        let info = diag.related_information.as_ref().unwrap();
        assert!(!info.is_empty());
        assert!(info[0].message.contains("time") || info[0].message.contains("formula"));
    }

    #[test]
    fn test_dimensionless_warning() {
        let provider = DiagnosticsProvider::new();
        let diag = provider.dimensionless_warning(test_range(), "ratio");

        assert_eq!(diag.severity, Some(DiagnosticSeverity::WARNING));
        assert!(diag.message.contains("ratio"));
        assert!(diag.message.contains("dimensionless"));
    }

    #[test]
    fn test_missing_unit_annotation() {
        let provider = DiagnosticsProvider::new();
        let diag = provider.missing_unit_annotation(test_range(), "distance", "meters");

        assert_eq!(diag.severity, Some(DiagnosticSeverity::HINT));
        assert!(diag.message.contains("distance"));
        assert!(diag.message.contains("meters"));
    }

    #[test]
    fn test_implicit_unit_conversion_small() {
        let provider = DiagnosticsProvider::new();
        // Small conversion factor (close to 1) - should be INFO
        let diag = provider.implicit_unit_conversion(test_range(), "cm", "mm", 10.0);

        assert_eq!(diag.severity, Some(DiagnosticSeverity::INFORMATION));
        assert!(diag.message.contains("cm"));
        assert!(diag.message.contains("10"));
    }

    #[test]
    fn test_implicit_unit_conversion_large() {
        let provider = DiagnosticsProvider::new();
        // Large conversion factor - should be WARNING
        let diag = provider.implicit_unit_conversion(test_range(), "parsec", "m", 3.086e16);

        assert_eq!(diag.severity, Some(DiagnosticSeverity::WARNING));
    }

    #[test]
    fn test_incompatible_unit_operation() {
        let provider = DiagnosticsProvider::new();
        let diag = provider.incompatible_unit_operation(test_range(), "add", "meters", "seconds");

        assert_eq!(diag.severity, Some(DiagnosticSeverity::ERROR));
        assert!(diag.message.contains("add"));
        assert!(diag.message.contains("meters"));
        assert!(diag.message.contains("seconds"));
    }

    #[test]
    fn test_unit_confusion_warning() {
        let provider = DiagnosticsProvider::new();
        let diag =
            provider.unit_confusion_warning(test_range(), "mph", "km/h", "velocity calculations");

        assert_eq!(diag.severity, Some(DiagnosticSeverity::WARNING));
        assert!(diag.message.contains("mph"));
        assert!(diag.message.contains("km/h"));
    }

    #[test]
    fn test_si_equivalence_info() {
        let provider = DiagnosticsProvider::new();
        let diag = provider.si_equivalence_info(test_range(), "Newton", "kg·m/s²");

        assert_eq!(diag.severity, Some(DiagnosticSeverity::INFORMATION));
        assert!(diag.message.contains("Newton"));
        assert!(diag.message.contains("kg·m/s²"));
    }

    #[test]
    fn test_unit_diagnostics_with_info() {
        let provider = DiagnosticsProvider::new();
        let info = UnitInfo {
            unit: "velocity".to_string(),
            dimension: "L/T".to_string(),
            is_dimensionless: true, // Unexpected dimensionless
        };

        let diagnostics = provider.unit_diagnostics(test_range(), "speed", &info);

        // Should warn about unexpected dimensionless
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn test_suggest_dimension_fix_velocity() {
        let provider = DiagnosticsProvider::new();
        let suggestion = provider.suggest_dimension_fix("L/T", "M");

        assert!(suggestion.is_some());
        // Should mention time or formula
        assert!(
            suggestion.as_ref().unwrap().contains("time")
                || suggestion.as_ref().unwrap().contains("formula")
        );
    }
}
