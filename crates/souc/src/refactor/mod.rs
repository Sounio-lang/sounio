//! Refactoring Tools
//!
//! This module provides automated code refactoring capabilities:
//! - Rename: Rename symbols with all references
//! - Extract: Extract expressions/statements to functions
//! - Inline: Inline function calls or variables
//! - Move: Move items between modules

pub mod extract;
// TODO: Fix rename.rs to match current AST structure
// pub mod rename;

pub use extract::{ExtractFunction, ExtractResult, ExtractVariable};
// pub use rename::{RenameError, RenameRefactoring, RenameResult};

use crate::ast::Ast;
use crate::common::Span;
use std::collections::HashMap;

/// A text edit representing a change to source code
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextEdit {
    /// The range to replace (byte offsets)
    pub range: std::ops::Range<usize>,
    /// The new text to insert
    pub new_text: String,
}

impl TextEdit {
    /// Create a new text edit
    pub fn new(range: std::ops::Range<usize>, new_text: impl Into<String>) -> Self {
        Self {
            range,
            new_text: new_text.into(),
        }
    }

    /// Create an insertion edit
    pub fn insert(position: usize, text: impl Into<String>) -> Self {
        Self {
            range: position..position,
            new_text: text.into(),
        }
    }

    /// Create a deletion edit
    pub fn delete(range: std::ops::Range<usize>) -> Self {
        Self {
            range,
            new_text: String::new(),
        }
    }

    /// Create a replacement edit
    pub fn replace(range: std::ops::Range<usize>, text: impl Into<String>) -> Self {
        Self {
            range,
            new_text: text.into(),
        }
    }
}

/// A set of text edits for a single file
#[derive(Debug, Clone, Default)]
pub struct FileEdits {
    /// File path (relative or absolute)
    pub path: String,
    /// Edits to apply (should be applied in reverse order by position)
    pub edits: Vec<TextEdit>,
}

impl FileEdits {
    /// Create a new file edits container
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            edits: Vec::new(),
        }
    }

    /// Add an edit
    pub fn add_edit(&mut self, edit: TextEdit) {
        self.edits.push(edit);
    }

    /// Sort edits in reverse order by position for safe application
    pub fn sort_for_application(&mut self) {
        self.edits.sort_by(|a, b| b.range.start.cmp(&a.range.start));
    }

    /// Apply edits to source text
    pub fn apply(&self, source: &str) -> String {
        let mut result = source.to_string();
        let mut edits = self.edits.clone();
        edits.sort_by(|a, b| b.range.start.cmp(&a.range.start));

        for edit in edits {
            if edit.range.start <= result.len() && edit.range.end <= result.len() {
                result.replace_range(edit.range.clone(), &edit.new_text);
            }
        }

        result
    }
}

/// A workspace edit containing changes to multiple files
#[derive(Debug, Clone, Default)]
pub struct WorkspaceEdit {
    /// Edits grouped by file
    pub file_edits: HashMap<String, FileEdits>,
}

impl WorkspaceEdit {
    /// Create a new workspace edit
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an edit for a file
    pub fn add_edit(&mut self, path: &str, edit: TextEdit) {
        self.file_edits
            .entry(path.to_string())
            .or_insert_with(|| FileEdits::new(path))
            .add_edit(edit);
    }

    /// Get edits for a specific file
    pub fn get_file_edits(&self, path: &str) -> Option<&FileEdits> {
        self.file_edits.get(path)
    }

    /// Check if there are any edits
    pub fn is_empty(&self) -> bool {
        self.file_edits.is_empty() || self.file_edits.values().all(|f| f.edits.is_empty())
    }

    /// Total number of edits across all files
    pub fn edit_count(&self) -> usize {
        self.file_edits.values().map(|f| f.edits.len()).sum()
    }

    /// Number of files affected
    pub fn file_count(&self) -> usize {
        self.file_edits.len()
    }
}

/// Location information for a symbol
#[derive(Debug, Clone)]
pub struct SymbolLocation {
    /// File containing the symbol
    pub file: String,
    /// Span of the symbol
    pub span: Span,
    /// Kind of reference (definition, reference, import)
    pub kind: ReferenceKind,
}

/// Kind of symbol reference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    /// Symbol definition
    Definition,
    /// Symbol reference/usage
    Reference,
    /// Import statement
    Import,
    /// Type annotation
    TypeAnnotation,
    /// Pattern binding
    PatternBinding,
}

/// Result of finding all references to a symbol
#[derive(Debug, Clone)]
pub struct FindReferencesResult {
    /// The symbol name
    pub symbol: String,
    /// Definition location
    pub definition: Option<SymbolLocation>,
    /// All references
    pub references: Vec<SymbolLocation>,
}

impl FindReferencesResult {
    /// Total count of all locations (definition + references)
    pub fn total_count(&self) -> usize {
        let def_count = if self.definition.is_some() { 1 } else { 0 };
        def_count + self.references.len()
    }
}

/// Refactoring context providing access to source and AST
pub struct RefactoringContext<'a> {
    /// Source code by file
    pub sources: HashMap<String, &'a str>,
    /// Parsed ASTs by file
    pub asts: HashMap<String, &'a Ast>,
    /// Current file being refactored
    pub current_file: String,
}

impl<'a> RefactoringContext<'a> {
    /// Create a new refactoring context
    pub fn new(current_file: impl Into<String>) -> Self {
        Self {
            sources: HashMap::new(),
            asts: HashMap::new(),
            current_file: current_file.into(),
        }
    }

    /// Add a source file
    pub fn add_source(&mut self, path: impl Into<String>, source: &'a str) {
        self.sources.insert(path.into(), source);
    }

    /// Add a parsed AST
    pub fn add_ast(&mut self, path: impl Into<String>, ast: &'a Ast) {
        self.asts.insert(path.into(), ast);
    }

    /// Get source for a file
    pub fn get_source(&self, path: &str) -> Option<&str> {
        self.sources.get(path).copied()
    }

    /// Get AST for a file
    pub fn get_ast(&self, path: &str) -> Option<&Ast> {
        self.asts.get(path).copied()
    }

    /// Get current source
    pub fn current_source(&self) -> Option<&str> {
        self.sources.get(&self.current_file).copied()
    }

    /// Get current AST
    pub fn current_ast(&self) -> Option<&Ast> {
        self.asts.get(&self.current_file).copied()
    }
}

/// Common refactoring error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum RefactoringError {
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),

    #[error("Invalid position: {0}")]
    InvalidPosition(usize),

    #[error("Invalid selection: {start}..{end}")]
    InvalidSelection { start: usize, end: usize },

    #[error("Cannot refactor: {0}")]
    CannotRefactor(String),

    #[error("Name conflict: {0} already exists")]
    NameConflict(String),

    #[error("Source not found for file: {0}")]
    SourceNotFound(String),

    #[error("Module not found for file: {0}")]
    ModuleNotFound(String),
}

/// Validate that a name is a valid identifier
pub fn is_valid_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    let mut chars = name.chars();
    let first = chars.next().unwrap();

    // First character must be letter or underscore
    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    // Rest must be alphanumeric or underscore
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// Reserved keywords that cannot be used as identifiers
pub const RESERVED_KEYWORDS: &[&str] = &[
    "let", "var", "const", "fn", "struct", "enum", "trait", "impl", "type", "if", "else", "match",
    "for", "while", "loop", "break", "continue", "return", "true", "false", "self", "Self", "pub",
    "use", "mod", "as", "in", "where", "async", "await", "effect", "handle", "resume", "perform",
    "linear", "affine", "with", "kernel", "device", "host", "shared",
];

/// Check if a name is a reserved keyword
pub fn is_reserved_keyword(name: &str) -> bool {
    RESERVED_KEYWORDS.contains(&name)
}

/// Validate a new name for refactoring
pub fn validate_new_name(name: &str) -> Result<(), RefactoringError> {
    if !is_valid_identifier(name) {
        return Err(RefactoringError::CannotRefactor(format!(
            "'{}' is not a valid identifier",
            name
        )));
    }

    if is_reserved_keyword(name) {
        return Err(RefactoringError::CannotRefactor(format!(
            "'{}' is a reserved keyword",
            name
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_edit_apply() {
        let source = "hello world";
        let edits = FileEdits {
            path: "test.sio".to_string(),
            edits: vec![TextEdit::replace(0..5, "goodbye")],
        };

        assert_eq!(edits.apply(source), "goodbye world");
    }

    #[test]
    fn test_text_edit_insert() {
        let source = "hello world";
        let edits = FileEdits {
            path: "test.sio".to_string(),
            edits: vec![TextEdit::insert(5, " beautiful")],
        };

        assert_eq!(edits.apply(source), "hello beautiful world");
    }

    #[test]
    fn test_text_edit_delete() {
        let source = "hello world";
        let edits = FileEdits {
            path: "test.sio".to_string(),
            edits: vec![TextEdit::delete(5..11)],
        };

        assert_eq!(edits.apply(source), "hello");
    }

    #[test]
    fn test_multiple_edits() {
        let source = "let x = 1 + 2";
        let edits = FileEdits {
            path: "test.sio".to_string(),
            edits: vec![
                TextEdit::replace(4..5, "y"),   // x -> y
                TextEdit::replace(8..9, "3"),   // 1 -> 3
                TextEdit::replace(12..13, "4"), // 2 -> 4
            ],
        };

        assert_eq!(edits.apply(source), "let y = 3 + 4");
    }

    #[test]
    fn test_workspace_edit() {
        let mut ws = WorkspaceEdit::new();
        ws.add_edit("file1.sio", TextEdit::replace(0..5, "test"));
        ws.add_edit("file2.sio", TextEdit::insert(0, "// header\n"));

        assert_eq!(ws.file_count(), 2);
        assert_eq!(ws.edit_count(), 2);
        assert!(!ws.is_empty());
    }

    #[test]
    fn test_is_valid_identifier() {
        assert!(is_valid_identifier("foo"));
        assert!(is_valid_identifier("_bar"));
        assert!(is_valid_identifier("foo123"));
        assert!(is_valid_identifier("_"));

        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("123foo"));
        assert!(!is_valid_identifier("foo-bar"));
        assert!(!is_valid_identifier("foo.bar"));
    }

    #[test]
    fn test_is_reserved_keyword() {
        assert!(is_reserved_keyword("let"));
        assert!(is_reserved_keyword("fn"));
        assert!(is_reserved_keyword("struct"));

        assert!(!is_reserved_keyword("foo"));
        assert!(!is_reserved_keyword("myVar"));
    }

    #[test]
    fn test_validate_new_name() {
        assert!(validate_new_name("valid_name").is_ok());
        assert!(validate_new_name("_private").is_ok());

        assert!(validate_new_name("").is_err());
        assert!(validate_new_name("let").is_err());
        assert!(validate_new_name("123abc").is_err());
    }
}
