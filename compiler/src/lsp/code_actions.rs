//! Code actions provider for quick fixes and refactorings
//!
//! Provides code actions for:
//! - Quick fixes from diagnostics (typos, missing imports, etc.)
//! - Refactoring actions (extract function, extract variable, inline, etc.)
//! - Source actions (organize imports, add missing implementations)
//!
//! # Code Action Kinds
//!
//! - `quickfix` - Fixes for diagnostics
//! - `refactor` - Generic refactoring actions
//! - `refactor.extract` - Extract to function/variable/constant
//! - `refactor.inline` - Inline function/variable
//! - `refactor.rewrite` - Rewrite expressions
//! - `source` - Source-level actions
//! - `source.organizeImports` - Organize imports

use tower_lsp::lsp_types::*;

use crate::ast::{Ast, Item};
use crate::common::Span;
use crate::resolve::SymbolTable;

// ============================================================================
// Code Action Provider
// ============================================================================

/// Provider for code actions
pub struct CodeActionProvider {
    /// Enable quick fixes
    pub enable_quick_fixes: bool,
    /// Enable refactoring actions
    pub enable_refactorings: bool,
    /// Enable source actions
    pub enable_source_actions: bool,
}

impl CodeActionProvider {
    /// Create a new code action provider
    pub fn new() -> Self {
        Self {
            enable_quick_fixes: true,
            enable_refactorings: true,
            enable_source_actions: true,
        }
    }

    /// Generate code actions for a range
    pub fn code_actions(
        &self,
        uri: &Url,
        range: Range,
        diagnostics: &[Diagnostic],
        source: &str,
        ast: Option<&Ast>,
        symbols: Option<&SymbolTable>,
    ) -> Vec<CodeActionOrCommand> {
        let mut actions = Vec::new();

        // Quick fixes from diagnostics
        if self.enable_quick_fixes {
            for diag in diagnostics {
                self.add_diagnostic_fixes(uri, diag, source, &mut actions);
            }
        }

        // Refactoring actions based on selection
        if self.enable_refactorings {
            self.add_refactoring_actions(uri, range, source, ast, symbols, &mut actions);
        }

        // Source actions
        if self.enable_source_actions {
            self.add_source_actions(uri, source, ast, &mut actions);
        }

        actions
    }

    /// Add quick fixes for a diagnostic
    fn add_diagnostic_fixes(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Extract error code
        let code = match &diagnostic.code {
            Some(NumberOrString::String(s)) => s.as_str(),
            Some(NumberOrString::Number(n)) => {
                // Convert number to string for matching
                match n {
                    1 => "E0001",
                    2 => "E0002",
                    3 => "E0003",
                    4 => "E0004",
                    5 => "E0005",
                    6 => "E0006",
                    _ => return,
                }
            }
            None => return,
        };

        match code {
            "E0001" => {
                // Type mismatch - suggest type cast or conversion
                self.add_type_mismatch_fix(uri, diagnostic, source, actions);
            }
            "E0002" => {
                // Undefined variable - suggest similar names or add declaration
                self.add_undefined_variable_fix(uri, diagnostic, source, actions);
            }
            "E0003" => {
                // Unused variable - add underscore prefix
                self.add_unused_variable_fix(uri, diagnostic, actions);
            }
            "E0004" => {
                // Missing return type - add return type annotation
                self.add_missing_return_type_fix(uri, diagnostic, actions);
            }
            "E0005" => {
                // Ownership error - suggest clone or borrow
                self.add_ownership_fix(uri, diagnostic, source, actions);
            }
            "E0006" => {
                // Effect not declared - add effect to signature
                self.add_missing_effect_fix(uri, diagnostic, source, actions);
            }
            _ => {}
        }
    }

    /// Add fix for type mismatch
    fn add_type_mismatch_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        _source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Extract expected and found types from message
        let message = &diagnostic.message;

        // Suggest type cast
        if message.contains("expected") && message.contains("found") {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Add type cast".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: None, // Would need to construct proper edit
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add fix for undefined variable
    fn add_undefined_variable_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        _source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        // Extract variable name from message (e.g., "undefined variable `foo`")
        if let Some(var_name) = extract_quoted_name(message) {
            // Suggest adding variable declaration
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: format!("Add declaration for `{}`", var_name),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: Position::new(diagnostic.range.start.line, 0),
                                    end: Position::new(diagnostic.range.start.line, 0),
                                },
                                new_text: format!("let {} = todo!();\n", var_name),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));

            // Suggest similar names if typo detection found them
            if message.contains("did you mean") {
                if let Some(suggestion) = extract_suggestion(message) {
                    actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                        title: format!("Change to `{}`", suggestion),
                        kind: Some(CodeActionKind::QUICKFIX),
                        diagnostics: Some(vec![diagnostic.clone()]),
                        edit: Some(WorkspaceEdit {
                            changes: Some(
                                [(
                                    uri.clone(),
                                    vec![TextEdit {
                                        range: diagnostic.range,
                                        new_text: suggestion,
                                    }],
                                )]
                                .into_iter()
                                .collect(),
                            ),
                            ..Default::default()
                        }),
                        command: None,
                        is_preferred: Some(true),
                        disabled: None,
                        data: None,
                    }));
                }
            }
        }
    }

    /// Add fix for unused variable
    fn add_unused_variable_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        if let Some(var_name) = extract_quoted_name(message) {
            // Add underscore prefix
            if !var_name.starts_with('_') {
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: format!("Rename to `_{}`", var_name),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: diagnostic.range,
                                    new_text: format!("_{}", var_name),
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: None,
                }));
            }

            // Remove variable entirely (if safe)
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: format!("Remove unused variable `{}`", var_name),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: None, // Would need smarter edit to remove entire let statement
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add fix for missing return type
    fn add_missing_return_type_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Suggest adding return type annotation
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Add return type annotation".to_string(),
            kind: Some(CodeActionKind::QUICKFIX),
            diagnostics: Some(vec![diagnostic.clone()]),
            edit: Some(WorkspaceEdit {
                changes: Some(
                    [(
                        uri.clone(),
                        vec![TextEdit {
                            range: Range {
                                start: diagnostic.range.end,
                                end: diagnostic.range.end,
                            },
                            new_text: " -> ()".to_string(),
                        }],
                    )]
                    .into_iter()
                    .collect(),
                ),
                ..Default::default()
            }),
            command: None,
            is_preferred: Some(false),
            disabled: None,
            data: None,
        }));
    }

    /// Add fix for ownership errors
    fn add_ownership_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        _source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        if message.contains("moved") || message.contains("borrow") {
            // Suggest clone
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Add .clone()".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: diagnostic.range.end,
                                    end: diagnostic.range.end,
                                },
                                new_text: ".clone()".to_string(),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));

            // Suggest borrow
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Use reference instead".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: diagnostic.range.start,
                                    end: diagnostic.range.start,
                                },
                                new_text: "&".to_string(),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add fix for missing effect
    fn add_missing_effect_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        _source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        // Extract effect name
        if let Some(effect_name) = extract_quoted_name(message) {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: format!("Add `with {}` to function signature", effect_name),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: None, // Would need to find function signature to edit
                command: None,
                is_preferred: Some(true),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add refactoring actions based on selection
    fn add_refactoring_actions(
        &self,
        uri: &Url,
        range: Range,
        source: &str,
        _ast: Option<&Ast>,
        _symbols: Option<&SymbolTable>,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Check if there's a selection
        if range.start == range.end {
            return;
        }

        // Get selected text
        let selected = self.get_text_in_range(source, range);
        if selected.is_empty() {
            return;
        }

        // Extract to variable
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Extract to variable".to_string(),
            kind: Some(CodeActionKind::REFACTOR_EXTRACT),
            diagnostics: None,
            edit: Some(WorkspaceEdit {
                changes: Some(
                    [(
                        uri.clone(),
                        vec![
                            // Insert variable declaration before current line
                            TextEdit {
                                range: Range {
                                    start: Position::new(range.start.line, 0),
                                    end: Position::new(range.start.line, 0),
                                },
                                new_text: format!("let extracted = {};\n", selected.trim()),
                            },
                            // Replace selection with variable name
                            TextEdit {
                                range,
                                new_text: "extracted".to_string(),
                            },
                        ],
                    )]
                    .into_iter()
                    .collect(),
                ),
                ..Default::default()
            }),
            command: None,
            is_preferred: Some(false),
            disabled: None,
            data: None,
        }));

        // Extract to function (if selection looks like a block)
        if selected.contains(';') || selected.contains('{') {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Extract to function".to_string(),
                kind: Some(CodeActionKind::REFACTOR_EXTRACT),
                diagnostics: None,
                edit: None, // Complex - would need parameter analysis
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }

        // Inline variable (if selection is a variable name)
        if is_identifier(&selected) {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: format!("Inline `{}`", selected.trim()),
                kind: Some(CodeActionKind::REFACTOR_INLINE),
                diagnostics: None,
                edit: None, // Would need to find definition and replace all uses
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }

        // Convert to string interpolation (if selection is string concatenation)
        if selected.contains("+ \"") || selected.contains("\" +") {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Convert to string interpolation".to_string(),
                kind: Some(CodeActionKind::REFACTOR_REWRITE),
                diagnostics: None,
                edit: None, // Would need string parsing
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add source-level actions
    fn add_source_actions(
        &self,
        uri: &Url,
        source: &str,
        ast: Option<&Ast>,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Organize imports
        if source.contains("import ") {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Organize imports".to_string(),
                kind: Some(CodeActionKind::SOURCE_ORGANIZE_IMPORTS),
                diagnostics: None,
                edit: None, // Would need import analysis
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }

        // Add missing implementations (for traits/impls)
        if let Some(ast) = ast {
            for item in &ast.items {
                if let Item::Impl(impl_def) = item {
                    // Check if impl is incomplete
                    if impl_def.items.is_empty() {
                        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                            title: "Generate missing trait methods".to_string(),
                            kind: Some(CodeActionKind::SOURCE),
                            diagnostics: None,
                            edit: None,
                            command: None,
                            is_preferred: Some(false),
                            disabled: None,
                            data: None,
                        }));
                    }
                }
            }
        }

        // Format document
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Format document".to_string(),
            kind: Some(CodeActionKind::SOURCE),
            diagnostics: None,
            edit: None,
            command: Some(Command {
                title: "Format".to_string(),
                command: "sounio.formatDocument".to_string(),
                arguments: Some(vec![serde_json::json!(uri.as_str())]),
            }),
            is_preferred: Some(false),
            disabled: None,
            data: None,
        }));
    }

    /// Get text in a range
    fn get_text_in_range(&self, source: &str, range: Range) -> String {
        let lines: Vec<&str> = source.lines().collect();

        if range.start.line == range.end.line {
            // Single line selection
            if let Some(line) = lines.get(range.start.line as usize) {
                let start = range.start.character as usize;
                let end = range.end.character as usize;
                if start < line.len() && end <= line.len() {
                    return line[start..end].to_string();
                }
            }
        } else {
            // Multi-line selection
            let mut result = String::new();
            for (i, line) in lines.iter().enumerate() {
                let line_num = i as u32;
                if line_num == range.start.line {
                    let start = range.start.character as usize;
                    if start < line.len() {
                        result.push_str(&line[start..]);
                        result.push('\n');
                    }
                } else if line_num > range.start.line && line_num < range.end.line {
                    result.push_str(line);
                    result.push('\n');
                } else if line_num == range.end.line {
                    let end = range.end.character as usize;
                    if end <= line.len() {
                        result.push_str(&line[..end]);
                    }
                }
            }
            return result;
        }

        String::new()
    }
}

impl Default for CodeActionProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract a quoted name from a diagnostic message
fn extract_quoted_name(message: &str) -> Option<String> {
    // Look for `name` or 'name' pattern
    let patterns = [('`', '`'), ('\'', '\''), ('"', '"')];

    for (open, close) in patterns {
        if let Some(start) = message.find(open) {
            let rest = &message[start + 1..];
            if let Some(end) = rest.find(close) {
                return Some(rest[..end].to_string());
            }
        }
    }

    None
}

/// Extract a suggestion from a "did you mean" message
fn extract_suggestion(message: &str) -> Option<String> {
    if let Some(idx) = message.find("did you mean") {
        let rest = &message[idx..];
        return extract_quoted_name(rest);
    }
    None
}

/// Check if a string is a valid identifier
fn is_identifier(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars();
    if let Some(first) = chars.next() {
        if !first.is_alphabetic() && first != '_' {
            return false;
        }
    }

    chars.all(|c| c.is_alphanumeric() || c == '_')
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = CodeActionProvider::new();
        assert!(provider.enable_quick_fixes);
        assert!(provider.enable_refactorings);
        assert!(provider.enable_source_actions);
    }

    #[test]
    fn test_extract_quoted_name() {
        assert_eq!(
            extract_quoted_name("undefined variable `foo`"),
            Some("foo".to_string())
        );
        assert_eq!(
            extract_quoted_name("unknown type 'Bar'"),
            Some("Bar".to_string())
        );
        assert_eq!(extract_quoted_name("no quotes here"), None);
    }

    #[test]
    fn test_extract_suggestion() {
        assert_eq!(
            extract_suggestion("undefined variable `foo`, did you mean `for`?"),
            Some("for".to_string())
        );
        assert_eq!(extract_suggestion("no suggestion"), None);
    }

    #[test]
    fn test_is_identifier() {
        assert!(is_identifier("foo"));
        assert!(is_identifier("_bar"));
        assert!(is_identifier("baz123"));
        assert!(!is_identifier("123abc"));
        assert!(!is_identifier("foo bar"));
        assert!(!is_identifier(""));
    }

    #[test]
    fn test_get_text_in_range_single_line() {
        let provider = CodeActionProvider::new();
        let source = "let x = 42;";
        let range = Range {
            start: Position::new(0, 4),
            end: Position::new(0, 5),
        };
        assert_eq!(provider.get_text_in_range(source, range), "x");
    }

    #[test]
    fn test_get_text_in_range_multi_line() {
        let provider = CodeActionProvider::new();
        let source = "let x = 42;\nlet y = 10;";
        let range = Range {
            start: Position::new(0, 8),
            end: Position::new(1, 3),
        };
        let text = provider.get_text_in_range(source, range);
        assert!(text.contains("42"));
        assert!(text.contains("let"));
    }

    #[test]
    fn test_code_actions_for_undefined_variable() {
        let provider = CodeActionProvider::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let diagnostic = Diagnostic {
            range: Range {
                start: Position::new(0, 0),
                end: Position::new(0, 3),
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String("E0002".to_string())),
            message: "undefined variable `foo`".to_string(),
            ..Default::default()
        };

        let actions =
            provider.code_actions(&uri, diagnostic.range, &[diagnostic], "foo", None, None);
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_refactoring_actions() {
        let provider = CodeActionProvider::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let source = "let x = 1 + 2 + 3;";
        let range = Range {
            start: Position::new(0, 8),
            end: Position::new(0, 17),
        };

        let actions = provider.code_actions(&uri, range, &[], source, None, None);
        // Should have extract to variable action
        let titles: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                CodeActionOrCommand::CodeAction(ca) => Some(ca.title.as_str()),
                _ => None,
            })
            .collect();
        assert!(titles.contains(&"Extract to variable"));
    }
}
