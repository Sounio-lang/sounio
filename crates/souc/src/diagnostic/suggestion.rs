//! Code Suggestions and Fixes
//!
//! This module provides functionality for suggesting fixes to errors.

use super::Span;

/// A suggested code fix
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// The span to replace
    pub span: Span,
    /// The replacement text
    pub replacement: String,
    /// Human-readable description of the fix
    pub message: String,
    /// How confident we are this fix is correct
    pub applicability: SuggestionApplicability,
}

impl Suggestion {
    /// Create a new suggestion
    pub fn new(
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
        applicability: SuggestionApplicability,
    ) -> Self {
        Suggestion {
            span,
            replacement: replacement.into(),
            message: message.into(),
            applicability,
        }
    }

    /// Create a suggestion that can be automatically applied
    pub fn machine_applicable(
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(
            span,
            replacement,
            message,
            SuggestionApplicability::MachineApplicable,
        )
    }

    /// Create a suggestion that might work but needs human review
    pub fn maybe_incorrect(
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(
            span,
            replacement,
            message,
            SuggestionApplicability::MaybeIncorrect,
        )
    }

    /// Create a suggestion for informational purposes only
    pub fn unspecified(
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(
            span,
            replacement,
            message,
            SuggestionApplicability::Unspecified,
        )
    }
}

/// How applicable a suggestion is
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionApplicability {
    /// The suggestion is definitely correct and can be automatically applied
    MachineApplicable,
    /// The suggestion may be correct but should be reviewed by a human
    MaybeIncorrect,
    /// The suggestion has placeholders that need to be filled in
    HasPlaceholders,
    /// Applicability is unknown
    Unspecified,
}

impl SuggestionApplicability {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SuggestionApplicability::MachineApplicable => "machine-applicable",
            SuggestionApplicability::MaybeIncorrect => "maybe-incorrect",
            SuggestionApplicability::HasPlaceholders => "has-placeholders",
            SuggestionApplicability::Unspecified => "unspecified",
        }
    }

    /// Check if this suggestion can be automatically applied
    pub fn is_machine_applicable(&self) -> bool {
        matches!(self, SuggestionApplicability::MachineApplicable)
    }
}

/// Builder for multi-part suggestions
#[derive(Debug)]
pub struct MultiSuggestion {
    /// Message describing the overall fix
    pub message: String,
    /// Individual parts of the suggestion
    pub parts: Vec<SuggestionPart>,
    /// Overall applicability
    pub applicability: SuggestionApplicability,
}

/// A single part of a multi-part suggestion
#[derive(Debug)]
pub struct SuggestionPart {
    /// Span to replace
    pub span: Span,
    /// Replacement text
    pub replacement: String,
}

impl MultiSuggestion {
    /// Create a new multi-part suggestion
    pub fn new(message: impl Into<String>) -> Self {
        MultiSuggestion {
            message: message.into(),
            parts: Vec::new(),
            applicability: SuggestionApplicability::MachineApplicable,
        }
    }

    /// Add a part to the suggestion
    pub fn add_part(mut self, span: Span, replacement: impl Into<String>) -> Self {
        self.parts.push(SuggestionPart {
            span,
            replacement: replacement.into(),
        });
        self
    }

    /// Set applicability
    pub fn applicability(mut self, applicability: SuggestionApplicability) -> Self {
        self.applicability = applicability;
        self
    }
}

/// Common suggestion patterns
pub mod patterns {
    use super::*;

    /// Suggest adding a missing semicolon
    pub fn add_semicolon(span: Span) -> Suggestion {
        Suggestion::machine_applicable(span, ";", "add semicolon")
    }

    /// Suggest adding a missing type annotation
    pub fn add_type_annotation(span: Span, suggested_type: &str) -> Suggestion {
        Suggestion::maybe_incorrect(span, format!(": {}", suggested_type), "add type annotation")
    }

    /// Suggest removing unused code
    pub fn remove_unused(span: Span, what: &str) -> Suggestion {
        Suggestion::machine_applicable(span, "", format!("remove unused {}", what))
    }

    /// Suggest renaming an identifier
    pub fn rename(span: Span, old_name: &str, new_name: &str) -> Suggestion {
        Suggestion::machine_applicable(
            span,
            new_name,
            format!("rename `{}` to `{}`", old_name, new_name),
        )
    }

    /// Suggest adding a missing import
    pub fn add_import(insert_span: Span, module_path: &str) -> Suggestion {
        Suggestion::machine_applicable(
            insert_span,
            format!("use {};\n", module_path),
            format!("import `{}`", module_path),
        )
    }

    /// Suggest making a binding mutable
    pub fn make_mutable(let_span: Span, name: &str) -> Suggestion {
        Suggestion::machine_applicable(
            let_span,
            format!("var {}", name),
            "make binding mutable by changing `let` to `var`",
        )
    }

    /// Suggest adding an effect annotation
    pub fn add_effect(signature_span: Span, effect: &str, current_sig: &str) -> Suggestion {
        let new_sig = if current_sig.contains("with") {
            current_sig.replace("with", &format!("with {}, ", effect))
        } else {
            format!("{} with {}", current_sig, effect)
        };

        Suggestion::machine_applicable(
            signature_span,
            new_sig,
            format!("add `{}` effect to function signature", effect),
        )
    }

    /// Suggest changing a reference type
    pub fn change_reference(span: Span, from: &str, to: &str) -> Suggestion {
        Suggestion::machine_applicable(span, to, format!("change `{}` to `{}`", from, to))
    }

    /// Suggest adding explicit type parameters
    pub fn add_type_params(span: Span, params: &[&str]) -> Suggestion {
        let params_str = params.join(", ");
        Suggestion::maybe_incorrect(
            span,
            format!("<{}>", params_str),
            "add explicit type parameters",
        )
    }

    /// Suggest using a different function/method
    pub fn use_alternative(
        span: Span,
        current: &str,
        alternative: &str,
        reason: &str,
    ) -> Suggestion {
        Suggestion::maybe_incorrect(
            span,
            alternative,
            format!("use `{}` instead of `{}`: {}", alternative, current, reason),
        )
    }

    /// Suggest wrapping in Option/Result
    pub fn wrap_in_some(span: Span, expr: &str) -> Suggestion {
        Suggestion::machine_applicable(span, format!("Some({})", expr), "wrap value in `Some`")
    }

    /// Suggest unwrapping Option/Result
    pub fn add_unwrap(span: Span) -> Suggestion {
        Suggestion::maybe_incorrect(
            span,
            ".unwrap()",
            "add `.unwrap()` (will panic if None/Err)",
        )
    }

    /// Suggest adding question mark operator
    pub fn add_try(span: Span) -> Suggestion {
        Suggestion::machine_applicable(span, "?", "use `?` to propagate the error")
    }

    /// Suggest cloning a value
    pub fn add_clone(span: Span) -> Suggestion {
        Suggestion::maybe_incorrect(span, ".clone()", "add `.clone()` to copy the value")
    }

    /// Suggest dereferencing
    pub fn add_deref(span: Span, expr: &str) -> Suggestion {
        Suggestion::machine_applicable(span, format!("*{}", expr), "dereference the value")
    }

    /// Suggest taking a reference
    pub fn add_ref(span: Span, expr: &str, mutable: bool) -> Suggestion {
        let ref_str = if mutable { "&!" } else { "&" };
        Suggestion::machine_applicable(
            span,
            format!("{}{}", ref_str, expr),
            format!("take a {}reference", if mutable { "mutable " } else { "" }),
        )
    }
}

/// Apply suggestions to source code
pub fn apply_suggestions(source: &str, mut suggestions: Vec<&Suggestion>) -> String {
    // Sort suggestions by span start, in reverse order
    // This allows us to apply them from end to start without invalidating offsets
    suggestions.sort_by(|a, b| b.span.start.cmp(&a.span.start));

    let mut result = source.to_string();

    for suggestion in suggestions {
        if suggestion.applicability.is_machine_applicable() {
            // Check that the span is valid
            if suggestion.span.start <= result.len() && suggestion.span.end <= result.len() {
                result.replace_range(
                    suggestion.span.start..suggestion.span.end,
                    &suggestion.replacement,
                );
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_suggestions() {
        let source = "let x = 1\nlet y = 2";
        let suggestion = Suggestion::machine_applicable(Span::new(9, 9, 1), ";", "add semicolon");
        let suggestions = vec![&suggestion];

        let result = apply_suggestions(source, suggestions);
        assert_eq!(result, "let x = 1;\nlet y = 2");
    }

    #[test]
    fn test_multiple_suggestions() {
        let source = "let x = foo";
        let sug1 = Suggestion::machine_applicable(Span::new(11, 11, 1), "()", "add parentheses");
        let sug2 = Suggestion::machine_applicable(Span::new(4, 5, 1), "result", "rename");
        let suggestions = vec![&sug1, &sug2];

        let result = apply_suggestions(source, suggestions);
        assert_eq!(result, "let result = foo()");
    }

    #[test]
    fn test_suggestion_patterns() {
        let s = patterns::add_semicolon(Span::new(10, 10, 1));
        assert!(s.applicability.is_machine_applicable());
        assert_eq!(s.replacement, ";");

        let s = patterns::make_mutable(Span::new(0, 5, 1), "x");
        assert!(s.message.contains("mutable"));
    }
}
