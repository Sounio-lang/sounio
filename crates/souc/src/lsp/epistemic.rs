//! Epistemic LSP Extensions
//!
//! This module provides LSP features for epistemic type awareness:
//!
//! - **Hover**: Show confidence, source, and provenance for values
//! - **Inlay Hints**: Display confidence badges inline
//! - **Code Lenses**: Show epistemic status summaries
//! - **Diagnostics**: Epistemic integrity warnings
//! - **Semantic Tokens**: Color-code by confidence level
//!
//! # Design Philosophy
//!
//! IDEs should surface epistemic information naturally, helping developers
//! understand the certainty and provenance of their data at a glance.

use tower_lsp::lsp_types::*;

use crate::diagnostic::epistemic::{EpistemicDiagnostic, EpistemicSeverity};
use crate::epistemic::{Confidence, EpistemicStatus, Revisability, Source};

/// Epistemic hover provider
pub struct EpistemicHoverProvider;

impl EpistemicHoverProvider {
    /// Generate hover content for an epistemic status
    pub fn format_epistemic_hover(status: &EpistemicStatus) -> Vec<MarkedString> {
        let mut parts = Vec::new();

        // Confidence section
        parts.push(MarkedString::String(Self::format_confidence_header(
            &status.confidence,
        )));

        // Source section
        parts.push(MarkedString::String(Self::format_source(&status.source)));

        // Revisability section
        parts.push(MarkedString::String(Self::format_revisability(
            &status.revisability,
        )));

        // Evidence section (if any)
        if !status.evidence.is_empty() {
            let evidence_text = status
                .evidence
                .iter()
                .map(|e| {
                    format!(
                        "- {}: {} (strength: {:.0}%)",
                        Self::evidence_kind_name(&e.kind),
                        e.reference,
                        e.strength.value() * 100.0
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            parts.push(MarkedString::String(format!(
                "**Evidence:**\n{}",
                evidence_text
            )));
        }

        parts
    }

    /// Format confidence as a visual header
    fn format_confidence_header(confidence: &Confidence) -> String {
        let value = confidence.value();
        let badge = Self::confidence_badge(value);
        let bar = Self::confidence_bar(value);

        let bounds = if let (Some(lower), Some(upper)) =
            (confidence.lower_bound(), confidence.upper_bound())
        {
            format!(" [{:.0}% - {:.0}%]", lower * 100.0, upper * 100.0)
        } else {
            String::new()
        };

        format!(
            "**Confidence:** {} {:.0}%{}\n{}",
            badge,
            value * 100.0,
            bounds,
            bar
        )
    }

    /// Get confidence badge emoji
    fn confidence_badge(value: f64) -> &'static str {
        if value >= 0.95 {
            "🟢" // High confidence
        } else if value >= 0.8 {
            "🟡" // Good confidence
        } else if value >= 0.5 {
            "🟠" // Moderate confidence
        } else if value >= 0.2 {
            "🔴" // Low confidence
        } else {
            "⚫" // Very low confidence
        }
    }

    /// Generate visual confidence bar
    fn confidence_bar(value: f64) -> String {
        let filled = (value * 10.0).round() as usize;
        let empty = 10 - filled;
        format!("`[{}{}]`", "█".repeat(filled), "░".repeat(empty))
    }

    /// Format source information
    fn format_source(source: &Source) -> String {
        match source {
            Source::Axiom => "**Source:** Axiom (by definition)".to_string(),
            Source::Measurement {
                instrument,
                protocol,
                timestamp,
            } => {
                let parts = vec!["**Source:** Measurement"];
                let mut details = Vec::new();
                if let Some(inst) = instrument {
                    details.push(format!("instrument: {}", inst));
                }
                if let Some(proto) = protocol {
                    details.push(format!("protocol: {}", proto));
                }
                if let Some(ts) = timestamp {
                    details.push(format!("at: {}", ts));
                }
                if !details.is_empty() {
                    format!("{}\n- {}", parts.join(""), details.join("\n- "))
                } else {
                    parts.join("")
                }
            }
            Source::Derivation(method) => {
                format!("**Source:** Derived via `{}`", method)
            }
            Source::External { uri, accessed } => {
                let access_info = accessed
                    .as_ref()
                    .map(|a| format!(" (accessed: {})", a))
                    .unwrap_or_default();
                format!("**Source:** External: {}{}", uri, access_info)
            }
            Source::OntologyAssertion { ontology, term } => {
                format!("**Source:** Ontology assertion: `{}:{}`", ontology, term)
            }
            Source::ModelPrediction { model, version } => {
                let ver = version
                    .as_ref()
                    .map(|v| format!(" v{}", v))
                    .unwrap_or_default();
                format!("**Source:** Model prediction: `{}{}`", model, ver)
            }
            Source::Transformation { original, via } => {
                format!(
                    "**Source:** Transformed via `{}` from {}",
                    via,
                    Self::source_brief(original)
                )
            }
            Source::HumanAssertion { asserter } => {
                let who = asserter
                    .as_ref()
                    .map(|a| format!(" by {}", a))
                    .unwrap_or_default();
                format!("**Source:** Human assertion{}", who)
            }
            Source::Unknown => "**Source:** Unknown".to_string(),
        }
    }

    /// Brief source description for nested sources
    fn source_brief(source: &Source) -> String {
        match source {
            Source::Axiom => "axiom".to_string(),
            Source::Measurement { .. } => "measurement".to_string(),
            Source::Derivation(_) => "derivation".to_string(),
            Source::External { .. } => "external".to_string(),
            Source::OntologyAssertion { ontology, term } => format!("{}:{}", ontology, term),
            Source::ModelPrediction { model, .. } => model.clone(),
            Source::Transformation { via, .. } => format!("via {}", via),
            Source::HumanAssertion { .. } => "human assertion".to_string(),
            Source::Unknown => "unknown".to_string(),
        }
    }

    /// Format revisability
    fn format_revisability(revisability: &Revisability) -> String {
        match revisability {
            Revisability::NonRevisable => {
                "**Revisability:** 🔒 Non-revisable (axiom/definition)".to_string()
            }
            Revisability::Revisable { conditions } => {
                if conditions.is_empty() {
                    "**Revisability:** 🔓 Revisable".to_string()
                } else {
                    format!(
                        "**Revisability:** 🔓 Revisable under:\n- {}",
                        conditions.join("\n- ")
                    )
                }
            }
            Revisability::MustRevise { reason } => {
                format!("**Revisability:** ⚠️ Must revise: {}", reason)
            }
        }
    }

    /// Get evidence kind display name
    fn evidence_kind_name(kind: &crate::epistemic::EvidenceKind) -> &'static str {
        match kind {
            crate::epistemic::EvidenceKind::Publication { .. } => "Publication",
            crate::epistemic::EvidenceKind::Dataset { .. } => "Dataset",
            crate::epistemic::EvidenceKind::Experiment { .. } => "Experiment",
            crate::epistemic::EvidenceKind::Computation { .. } => "Computation",
            crate::epistemic::EvidenceKind::ExpertOpinion { .. } => "Expert Opinion",
            crate::epistemic::EvidenceKind::Verified { .. } => "Verification",
            crate::epistemic::EvidenceKind::HumanAssertion { .. } => "Human Assertion",
        }
    }
}

/// Epistemic inlay hint provider
pub struct EpistemicInlayProvider;

impl EpistemicInlayProvider {
    /// Generate inlay hint for epistemic status
    pub fn confidence_hint(confidence: f64, position: Position) -> InlayHint {
        let badge = EpistemicHoverProvider::confidence_badge(confidence);
        let label = format!("{} {:.0}%", badge, confidence * 100.0);

        InlayHint {
            position,
            label: InlayHintLabel::String(label),
            kind: Some(InlayHintKind::TYPE),
            text_edits: None,
            tooltip: Some(InlayHintTooltip::String(format!(
                "Epistemic confidence: {:.1}%",
                confidence * 100.0
            ))),
            padding_left: Some(true),
            padding_right: Some(false),
            data: None,
        }
    }

    /// Generate warning hint for low confidence
    pub fn low_confidence_hint(confidence: f64, position: Position) -> InlayHint {
        InlayHint {
            position,
            label: InlayHintLabel::String(format!("⚠️ low conf ({:.0}%)", confidence * 100.0)),
            kind: Some(InlayHintKind::TYPE),
            text_edits: None,
            tooltip: Some(InlayHintTooltip::String(
                "This value has low epistemic confidence. Consider adding evidence or using a more reliable source.".to_string()
            )),
            padding_left: Some(true),
            padding_right: Some(false),
            data: None,
        }
    }

    /// Generate provenance hint
    pub fn provenance_hint(source: &Source, position: Position) -> InlayHint {
        let label = match source {
            Source::Axiom => "📖 axiom",
            Source::Measurement { .. } => "🔬 measured",
            Source::Derivation(_) => "🧮 derived",
            Source::External { .. } => "🌐 external",
            Source::OntologyAssertion { .. } => "📚 ontology",
            Source::ModelPrediction { .. } => "🤖 predicted",
            Source::Transformation { .. } => "🔄 transformed",
            Source::HumanAssertion { .. } => "👤 asserted",
            Source::Unknown => "❓ unknown",
        };

        InlayHint {
            position,
            label: InlayHintLabel::String(label.to_string()),
            kind: Some(InlayHintKind::TYPE),
            text_edits: None,
            tooltip: Some(InlayHintTooltip::String(
                EpistemicHoverProvider::format_source(source),
            )),
            padding_left: Some(true),
            padding_right: Some(false),
            data: None,
        }
    }
}

/// Epistemic code lens provider
pub struct EpistemicCodeLensProvider;

impl EpistemicCodeLensProvider {
    /// Generate code lens showing epistemic summary for a function
    pub fn function_summary_lens(
        range: Range,
        min_confidence: f64,
        avg_confidence: f64,
        low_confidence_count: usize,
    ) -> CodeLens {
        let badge = EpistemicHoverProvider::confidence_badge(avg_confidence);
        let title = if low_confidence_count > 0 {
            format!(
                "{} avg {:.0}% | {} low-confidence values",
                badge,
                avg_confidence * 100.0,
                low_confidence_count
            )
        } else {
            format!("{} avg {:.0}% confidence", badge, avg_confidence * 100.0)
        };

        CodeLens {
            range,
            command: Some(Command {
                title,
                command: "sounio.showEpistemicDetails".to_string(),
                arguments: None,
            }),
            data: None,
        }
    }

    /// Generate warning lens for functions with epistemic issues
    pub fn warning_lens(range: Range, warning_count: usize, error_count: usize) -> CodeLens {
        let title = if error_count > 0 {
            format!(
                "🔴 {} epistemic errors, {} warnings",
                error_count, warning_count
            )
        } else {
            format!("🟡 {} epistemic warnings", warning_count)
        };

        CodeLens {
            range,
            command: Some(Command {
                title,
                command: "sounio.showEpistemicWarnings".to_string(),
                arguments: None,
            }),
            data: None,
        }
    }
}

/// Convert epistemic diagnostics to LSP diagnostics
pub fn to_lsp_diagnostics(epistemic_diagnostics: &[EpistemicDiagnostic]) -> Vec<Diagnostic> {
    epistemic_diagnostics
        .iter()
        .filter_map(|ed| {
            let span = ed.span?;

            let severity = match ed.severity() {
                EpistemicSeverity::Error => DiagnosticSeverity::ERROR,
                EpistemicSeverity::Warning => DiagnosticSeverity::WARNING,
                EpistemicSeverity::Note => DiagnosticSeverity::INFORMATION,
                EpistemicSeverity::Help => DiagnosticSeverity::HINT,
            };

            // Convert span to LSP range (would need source map in real implementation)
            let range = Range {
                start: Position {
                    line: 0,
                    character: span.start as u32,
                },
                end: Position {
                    line: 0,
                    character: span.end as u32,
                },
            };

            let mut related_info = Vec::new();
            for note in &ed.notes {
                related_info.push(DiagnosticRelatedInformation {
                    location: Location {
                        uri: Url::parse("file:///unknown").unwrap(),
                        range,
                    },
                    message: note.clone(),
                });
            }

            Some(Diagnostic {
                range,
                severity: Some(severity),
                code: Some(NumberOrString::String(ed.code.code())),
                code_description: None,
                source: Some("epistemic".to_string()),
                message: ed.message.clone(),
                related_information: if related_info.is_empty() {
                    None
                } else {
                    Some(related_info)
                },
                tags: None,
                data: None,
            })
        })
        .collect()
}

/// Epistemic semantic token modifiers
pub mod semantic_tokens {
    use tower_lsp::lsp_types::SemanticTokenModifier;

    /// High confidence modifier
    pub const HIGH_CONFIDENCE: SemanticTokenModifier = SemanticTokenModifier::new("highConfidence");

    /// Low confidence modifier
    pub const LOW_CONFIDENCE: SemanticTokenModifier = SemanticTokenModifier::new("lowConfidence");

    /// Uncertain modifier (unknown source)
    pub const UNCERTAIN: SemanticTokenModifier = SemanticTokenModifier::new("uncertain");

    /// Must revise modifier
    pub const MUST_REVISE: SemanticTokenModifier = SemanticTokenModifier::new("mustRevise");

    /// Ontology-backed modifier
    pub const ONTOLOGY_BACKED: SemanticTokenModifier = SemanticTokenModifier::new("ontologyBacked");

    /// Causal identifiable modifier
    pub const CAUSAL_IDENTIFIABLE: SemanticTokenModifier =
        SemanticTokenModifier::new("causalIdentifiable");

    /// Causal non-identifiable modifier
    pub const CAUSAL_NON_IDENTIFIABLE: SemanticTokenModifier =
        SemanticTokenModifier::new("causalNonIdentifiable");

    /// Get all epistemic semantic token modifiers
    pub fn epistemic_modifiers() -> Vec<SemanticTokenModifier> {
        vec![
            HIGH_CONFIDENCE,
            LOW_CONFIDENCE,
            UNCERTAIN,
            MUST_REVISE,
            ONTOLOGY_BACKED,
            CAUSAL_IDENTIFIABLE,
            CAUSAL_NON_IDENTIFIABLE,
        ]
    }

    /// Compute modifier flags for an epistemic status
    pub fn compute_modifiers(status: &super::EpistemicStatus) -> u32 {
        let mut flags = 0u32;

        // Confidence-based modifiers
        let confidence = status.confidence.value();
        if confidence >= 0.9 {
            flags |= 1 << 0; // HIGH_CONFIDENCE
        } else if confidence < 0.5 {
            flags |= 1 << 1; // LOW_CONFIDENCE
        }

        // Source-based modifiers
        if matches!(status.source, super::Source::Unknown) {
            flags |= 1 << 2; // UNCERTAIN
        }

        // Revisability-based modifiers
        if matches!(status.revisability, super::Revisability::MustRevise { .. }) {
            flags |= 1 << 3; // MUST_REVISE
        }

        // Ontology-backed modifier
        if matches!(status.source, super::Source::OntologyAssertion { .. }) {
            flags |= 1 << 4; // ONTOLOGY_BACKED
        }

        flags
    }
}

// ============================================================================
// Causal Hover Provider
// ============================================================================

/// Causal hover provider — shows causal graph structure and identifiability
pub struct CausalHoverProvider;

impl CausalHoverProvider {
    /// Generate hover content for causal information
    pub fn format_causal_hover(
        info: &super::query_bridge::CausalInfo,
    ) -> Vec<MarkedString> {
        let mut parts = Vec::new();

        // Graph structure
        let node_names: Vec<&str> = info.nodes.iter().map(|n| n.name.as_str()).collect();
        parts.push(MarkedString::String(format!(
            "**Causal Graph** ({} nodes, {} edges)",
            info.nodes.len(),
            info.edges.len() + info.bidirected.len()
        )));

        // DAG visualization (text-based)
        let mut dag_lines = Vec::new();
        for edge in &info.edges {
            if let Some(strength) = edge.strength {
                dag_lines.push(format!("  {} --({:.2})--> {}", edge.from, strength, edge.to));
            } else {
                dag_lines.push(format!("  {} --> {}", edge.from, edge.to));
            }
        }
        for (a, b) in &info.bidirected {
            dag_lines.push(format!("  {} <~~> {} (latent confounder)", a, b));
        }
        if !dag_lines.is_empty() {
            parts.push(MarkedString::LanguageString(
                tower_lsp::lsp_types::LanguageString {
                    language: "text".to_string(),
                    value: dag_lines.join("\n"),
                },
            ));
        }

        // Treatment/outcome
        if !info.treatments.is_empty() || !info.outcomes.is_empty() {
            let mut roles = Vec::new();
            if !info.treatments.is_empty() {
                roles.push(format!(
                    "Treatment: `{}`",
                    info.treatments.join("`, `")
                ));
            }
            if !info.outcomes.is_empty() {
                roles.push(format!(
                    "Outcome: `{}`",
                    info.outcomes.join("`, `")
                ));
            }
            parts.push(MarkedString::String(roles.join(" | ")));
        }

        // Identifiability
        parts.push(MarkedString::String(
            Self::format_identifiability(&info.identifiability),
        ));

        parts
    }

    /// Format identifiability status
    fn format_identifiability(
        id_info: &super::query_bridge::CausalIdentifiabilityInfo,
    ) -> String {
        use super::query_bridge::CausalIdMethod;

        let badge = if id_info.identifiable {
            "IDENTIFIABLE"
        } else {
            "NON-IDENTIFIABLE"
        };

        let method_str = match &id_info.method {
            CausalIdMethod::Backdoor => {
                format!(
                    "Backdoor criterion on {{{}}}",
                    id_info.adjustment_set.join(", ")
                )
            }
            CausalIdMethod::Frontdoor => {
                format!(
                    "Frontdoor criterion via {{{}}}",
                    id_info.adjustment_set.join(", ")
                )
            }
            CausalIdMethod::InstrumentalVariable => {
                format!(
                    "Instrumental variable: {{{}}}",
                    id_info.adjustment_set.join(", ")
                )
            }
            CausalIdMethod::Unconfounded => "Direct (no confounders)".to_string(),
            CausalIdMethod::NonIdentifiable => {
                let reason = id_info
                    .reason
                    .as_deref()
                    .unwrap_or("unobserved confounders");
                format!("Not identifiable: {}", reason)
            }
            CausalIdMethod::Unknown => "Not yet analyzed".to_string(),
        };

        format!("**Identifiability:** {} — {}", badge, method_str)
    }

    /// Generate a compact causal badge for inlay hints
    pub fn causal_badge(identifiable: bool) -> &'static str {
        if identifiable {
            "ID"  // Identifiable
        } else {
            "!ID" // Non-identifiable
        }
    }
}

/// Epistemic quick fix provider
pub struct EpistemicQuickFixProvider;

impl EpistemicQuickFixProvider {
    /// Generate code actions for epistemic issues
    pub fn quick_fixes(diagnostic: &EpistemicDiagnostic, uri: &Url) -> Vec<CodeAction> {
        let mut actions = Vec::new();

        for suggestion in &diagnostic.suggestions {
            if let Some(replacement) = &suggestion.replacement {
                // Create a text edit for the suggestion
                if let Some(span) = diagnostic.span {
                    let edit = TextEdit {
                        range: Range {
                            start: Position {
                                line: 0,
                                character: span.start as u32,
                            },
                            end: Position {
                                line: 0,
                                character: span.end as u32,
                            },
                        },
                        new_text: replacement.clone(),
                    };

                    let mut changes = std::collections::HashMap::new();
                    changes.insert(uri.clone(), vec![edit]);

                    actions.push(CodeAction {
                        title: suggestion.message.clone(),
                        kind: Some(CodeActionKind::QUICKFIX),
                        diagnostics: None,
                        edit: Some(WorkspaceEdit {
                            changes: Some(changes),
                            document_changes: None,
                            change_annotations: None,
                        }),
                        command: None,
                        is_preferred: Some(false),
                        disabled: None,
                        data: None,
                    });
                }
            }
        }

        // Add "Learn more" action
        actions.push(CodeAction {
            title: format!("Learn about {}", diagnostic.code.description()),
            kind: Some(CodeActionKind::new("epistemic.learnMore")),
            diagnostics: None,
            edit: None,
            command: Some(Command {
                title: "Open documentation".to_string(),
                command: "sounio.openEpistemicDocs".to_string(),
                arguments: Some(vec![serde_json::json!({
                    "code": diagnostic.code.code()
                })]),
            }),
            is_preferred: Some(false),
            disabled: None,
            data: None,
        });

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{Evidence, EvidenceKind};

    #[test]
    fn test_confidence_badge() {
        assert_eq!(EpistemicHoverProvider::confidence_badge(0.99), "🟢");
        assert_eq!(EpistemicHoverProvider::confidence_badge(0.85), "🟡");
        assert_eq!(EpistemicHoverProvider::confidence_badge(0.6), "🟠");
        assert_eq!(EpistemicHoverProvider::confidence_badge(0.3), "🔴");
        assert_eq!(EpistemicHoverProvider::confidence_badge(0.1), "⚫");
    }

    #[test]
    fn test_confidence_bar() {
        assert_eq!(
            EpistemicHoverProvider::confidence_bar(1.0),
            "`[██████████]`"
        );
        assert_eq!(
            EpistemicHoverProvider::confidence_bar(0.5),
            "`[█████░░░░░]`"
        );
        assert_eq!(
            EpistemicHoverProvider::confidence_bar(0.0),
            "`[░░░░░░░░░░]`"
        );
    }

    #[test]
    fn test_format_epistemic_hover() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.85),
            revisability: Revisability::Revisable {
                conditions: vec!["new_data".into()],
            },
            source: Source::Measurement {
                instrument: Some("mass_spec".into()),
                protocol: None,
                timestamp: None,
            },
            evidence: vec![],
        };

        let hover = EpistemicHoverProvider::format_epistemic_hover(&status);
        assert!(!hover.is_empty());
    }

    #[test]
    fn test_confidence_hint() {
        let hint = EpistemicInlayProvider::confidence_hint(
            0.9,
            Position {
                line: 0,
                character: 10,
            },
        );
        if let InlayHintLabel::String(label) = hint.label {
            assert!(label.contains("90%"));
        } else {
            panic!("Expected string label");
        }
    }

    #[test]
    fn test_semantic_modifiers() {
        let high_conf = EpistemicStatus {
            confidence: Confidence::new(0.95),
            source: Source::OntologyAssertion {
                ontology: "GO".into(),
                term: "0001".into(),
            },
            ..Default::default()
        };

        let flags = semantic_tokens::compute_modifiers(&high_conf);
        assert!(flags & (1 << 0) != 0); // HIGH_CONFIDENCE
        assert!(flags & (1 << 4) != 0); // ONTOLOGY_BACKED
    }

    #[test]
    fn test_low_confidence_modifiers() {
        let low_conf = EpistemicStatus {
            confidence: Confidence::new(0.3),
            source: Source::Unknown,
            revisability: Revisability::MustRevise {
                reason: "provisional".into(),
            },
            evidence: vec![],
        };

        let flags = semantic_tokens::compute_modifiers(&low_conf);
        assert!(flags & (1 << 1) != 0); // LOW_CONFIDENCE
        assert!(flags & (1 << 2) != 0); // UNCERTAIN
        assert!(flags & (1 << 3) != 0); // MUST_REVISE
    }

    #[test]
    fn test_code_lens_summary() {
        let lens = EpistemicCodeLensProvider::function_summary_lens(Range::default(), 0.7, 0.85, 2);

        if let Some(cmd) = lens.command {
            assert!(cmd.title.contains("85%"));
            assert!(cmd.title.contains("2 low-confidence"));
        }
    }

    #[test]
    fn test_format_source_variations() {
        // Test various source formats
        let sources = vec![
            Source::Axiom,
            Source::Derivation("test".into()),
            Source::Unknown,
            Source::OntologyAssertion {
                ontology: "GO".into(),
                term: "0001".into(),
            },
        ];

        for source in sources {
            let formatted = EpistemicHoverProvider::format_source(&source);
            assert!(!formatted.is_empty());
            assert!(formatted.starts_with("**Source:**"));
        }
    }

    #[test]
    fn test_epistemic_modifiers_list() {
        let mods = semantic_tokens::epistemic_modifiers();
        assert_eq!(mods.len(), 7); // 5 original + 2 causal
    }

    #[test]
    fn test_causal_hover_identifiable() {
        use crate::lsp::query_bridge::*;

        let info = CausalInfo {
            nodes: vec![
                CausalNodeInfo { name: "Dose".into(), role: "treatment".into() },
                CausalNodeInfo { name: "Effect".into(), role: "outcome".into() },
                CausalNodeInfo { name: "Genotype".into(), role: "confounder".into() },
            ],
            edges: vec![
                CausalEdgeInfo { from: "Dose".into(), to: "Effect".into(), strength: Some(0.8) },
                CausalEdgeInfo { from: "Genotype".into(), to: "Effect".into(), strength: None },
            ],
            bidirected: vec![],
            treatments: vec!["Dose".into()],
            outcomes: vec!["Effect".into()],
            identifiability: CausalIdentifiabilityInfo {
                identifiable: true,
                method: CausalIdMethod::Backdoor,
                adjustment_set: vec!["Genotype".into()],
                reason: None,
            },
        };

        let hover = CausalHoverProvider::format_causal_hover(&info);
        assert!(!hover.is_empty());
        // Check that identifiability info is present
        let text: String = hover.iter().map(|m| match m {
            MarkedString::String(s) => s.clone(),
            MarkedString::LanguageString(ls) => ls.value.clone(),
        }).collect::<Vec<_>>().join("\n");
        assert!(text.contains("IDENTIFIABLE"));
        assert!(text.contains("Backdoor"));
        assert!(text.contains("Genotype"));
    }

    #[test]
    fn test_causal_hover_non_identifiable() {
        use crate::lsp::query_bridge::*;

        let info = CausalInfo {
            nodes: vec![
                CausalNodeInfo { name: "X".into(), role: "treatment".into() },
                CausalNodeInfo { name: "Y".into(), role: "outcome".into() },
            ],
            edges: vec![
                CausalEdgeInfo { from: "X".into(), to: "Y".into(), strength: None },
            ],
            bidirected: vec![("X".into(), "Y".into())],
            treatments: vec!["X".into()],
            outcomes: vec!["Y".into()],
            identifiability: CausalIdentifiabilityInfo {
                identifiable: false,
                method: CausalIdMethod::NonIdentifiable,
                adjustment_set: vec![],
                reason: Some("latent confounder between X and Y".into()),
            },
        };

        let hover = CausalHoverProvider::format_causal_hover(&info);
        let text: String = hover.iter().map(|m| match m {
            MarkedString::String(s) => s.clone(),
            MarkedString::LanguageString(ls) => ls.value.clone(),
        }).collect::<Vec<_>>().join("\n");
        assert!(text.contains("NON-IDENTIFIABLE"));
        assert!(text.contains("latent confounder"));
    }

    #[test]
    fn test_causal_badge() {
        assert_eq!(CausalHoverProvider::causal_badge(true), "ID");
        assert_eq!(CausalHoverProvider::causal_badge(false), "!ID");
    }
}
