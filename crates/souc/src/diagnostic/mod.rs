//! Diagnostic System for the Sounio Compiler
//!
//! This module provides rich error diagnostics with:
//! - Error codes with documentation
//! - Source code highlighting
//! - Suggestions and fixes
//! - Machine-readable output
//! - Semantic annotations for ontological types
//! - Progress reporting for long compilations
//!
//! # Example
//!
//! ```rust,ignore
//! use sounio::diagnostic::{Diagnostic, DiagnosticLevel, DiagnosticBuilder, Span};
//!
//! let expr_span = Span::new(10, 20, 1);
//! let diagnostic = DiagnosticBuilder::error("E0001", "Type mismatch")
//!     .with_span(expr_span)
//!     .with_label(expr_span, "expected `int`, found `bool`")
//!     .with_help("the function signature declares return type `int`")
//!     .build();
//! ```

pub mod codes;
pub mod emitter;
pub mod epistemic;
pub mod progress;
pub mod render;
pub mod semantic;
pub mod suggestion;
pub mod type_diff;
pub mod typo;

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

/// Re-exports
pub use codes::{ErrorCode, ErrorIndex};
pub use emitter::{DiagnosticEmitter, HumanEmitter, JsonEmitter};
pub use progress::{
    CompilationProgress, Progress, ProgressStyle, StatusLine, format_bytes, format_duration,
};
pub use render::{
    DiagnosticStyle, DistanceSuggestion, RichRenderer, TerminalCaps, render_distance_suggestions,
};
pub use semantic::{
    DistanceComponents, SemanticAnnotator, SemanticContext, SemanticSuggestion, TermInfo,
};
pub use suggestion::{Suggestion, SuggestionApplicability};
pub use type_diff::{TypeDiff, TypeErrorBuilder, render_type_diff};
pub use typo::{SuggestionBuilder as TypoSuggestionBuilder, TypoDetector, TypoSuggestion};

/// Source location span
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// Start byte offset
    pub start: usize,
    /// End byte offset
    pub end: usize,
    /// File ID
    pub file_id: u32,
}

impl Span {
    /// Create a new span
    pub const fn new(start: usize, end: usize, file_id: u32) -> Self {
        Span {
            start,
            end,
            file_id,
        }
    }

    /// Create a dummy span
    pub const fn dummy() -> Self {
        Span {
            start: 0,
            end: 0,
            file_id: 0,
        }
    }

    /// Check if span is dummy
    pub fn is_dummy(&self) -> bool {
        self.start == 0 && self.end == 0
    }

    /// Merge two spans (smallest start to largest end)
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            file_id: self.file_id,
        }
    }

    /// Get span length
    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Severity level of a diagnostic
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    /// Internal compiler error
    Bug,
    /// Fatal error (compilation cannot continue)
    Fatal,
    /// Error (compilation can continue but will fail)
    Error,
    /// Warning
    Warning,
    /// Note (additional information)
    Note,
    /// Help suggestion
    Help,
}

impl DiagnosticLevel {
    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            DiagnosticLevel::Bug => "internal compiler error",
            DiagnosticLevel::Fatal => "fatal error",
            DiagnosticLevel::Error => "error",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Note => "note",
            DiagnosticLevel::Help => "help",
        }
    }

    /// Get ANSI color code
    pub fn color(&self) -> &'static str {
        match self {
            DiagnosticLevel::Bug => "\x1b[1;35m",     // Bold magenta
            DiagnosticLevel::Fatal => "\x1b[1;31m",   // Bold red
            DiagnosticLevel::Error => "\x1b[1;31m",   // Bold red
            DiagnosticLevel::Warning => "\x1b[1;33m", // Bold yellow
            DiagnosticLevel::Note => "\x1b[1;36m",    // Bold cyan
            DiagnosticLevel::Help => "\x1b[1;32m",    // Bold green
        }
    }

    /// Check if this is an error level
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            DiagnosticLevel::Bug | DiagnosticLevel::Fatal | DiagnosticLevel::Error
        )
    }
}

impl fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A label attached to a span
#[derive(Debug, Clone)]
pub struct Label {
    /// The span this label covers
    pub span: Span,
    /// Label message
    pub message: String,
    /// Whether this is the primary label
    pub primary: bool,
}

impl Label {
    /// Create a primary label
    pub fn primary(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            primary: true,
        }
    }

    /// Create a secondary label
    pub fn secondary(span: Span, message: impl Into<String>) -> Self {
        Label {
            span,
            message: message.into(),
            primary: false,
        }
    }
}

/// A single diagnostic (error, warning, note, etc.)
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Severity level
    pub level: DiagnosticLevel,
    /// Error code (e.g., "E0001")
    pub code: Option<String>,
    /// Primary message
    pub message: String,
    /// Labels attached to source spans
    pub labels: Vec<Label>,
    /// Notes (additional context)
    pub notes: Vec<String>,
    /// Help messages
    pub help: Vec<String>,
    /// Suggestions for fixes
    pub suggestions: Vec<Suggestion>,
    /// Child diagnostics
    pub children: Vec<Diagnostic>,
}

impl Diagnostic {
    /// Create a new diagnostic
    pub fn new(level: DiagnosticLevel, message: impl Into<String>) -> Self {
        Diagnostic {
            level,
            code: None,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
            help: Vec::new(),
            suggestions: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Create an error diagnostic
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Error, message)
    }

    /// Create a warning diagnostic
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Warning, message)
    }

    /// Create a note diagnostic
    pub fn note(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Note, message)
    }

    /// Create an ICE (internal compiler error)
    pub fn bug(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Bug, message)
    }

    /// Set error code
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Add a primary label
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label::primary(span, message));
        self
    }

    /// Add a secondary label
    pub fn with_secondary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label::secondary(span, message));
        self
    }

    /// Add a note
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add help text
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help.push(help.into());
        self
    }

    /// Add a suggestion
    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Add a child diagnostic
    pub fn with_child(mut self, child: Diagnostic) -> Self {
        self.children.push(child);
        self
    }

    /// Check if this is an error
    pub fn is_error(&self) -> bool {
        self.level.is_error()
    }

    /// Get the primary span (if any)
    pub fn primary_span(&self) -> Option<Span> {
        self.labels.iter().find(|l| l.primary).map(|l| l.span)
    }
}

/// Builder for constructing diagnostics fluently
pub struct DiagnosticBuilder {
    diagnostic: Diagnostic,
}

impl DiagnosticBuilder {
    /// Start building an error
    pub fn error(code: &str, message: impl Into<String>) -> Self {
        DiagnosticBuilder {
            diagnostic: Diagnostic::error(message).with_code(code),
        }
    }

    /// Start building a warning
    pub fn warning(code: &str, message: impl Into<String>) -> Self {
        DiagnosticBuilder {
            diagnostic: Diagnostic::warning(message).with_code(code),
        }
    }

    /// Start building a note
    pub fn note(message: impl Into<String>) -> Self {
        DiagnosticBuilder {
            diagnostic: Diagnostic::note(message),
        }
    }

    /// Add span with label
    pub fn with_span(mut self, span: Span) -> Self {
        if self.diagnostic.labels.is_empty() {
            self.diagnostic.labels.push(Label::primary(span, ""));
        }
        self
    }

    /// Add a label
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostic.labels.push(Label::primary(span, message));
        self
    }

    /// Add a secondary label
    pub fn with_secondary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostic.labels.push(Label::secondary(span, message));
        self
    }

    /// Add a note
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.diagnostic.notes.push(note.into());
        self
    }

    /// Add help text
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.diagnostic.help.push(help.into());
        self
    }

    /// Add a code suggestion
    pub fn with_suggestion(
        mut self,
        span: Span,
        replacement: String,
        message: impl Into<String>,
        applicability: SuggestionApplicability,
    ) -> Self {
        self.diagnostic.suggestions.push(Suggestion {
            span,
            replacement,
            message: message.into(),
            applicability,
        });
        self
    }

    /// Build the diagnostic
    pub fn build(self) -> Diagnostic {
        self.diagnostic
    }

    /// Emit and return the diagnostic
    pub fn emit(self, emitter: &mut dyn DiagnosticEmitter) -> Diagnostic {
        let diagnostic = self.diagnostic;
        emitter.emit(&diagnostic);
        diagnostic
    }
}

/// Source file information
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// File ID
    pub id: u32,
    /// File path
    pub path: PathBuf,
    /// Source content
    pub source: String,
    /// Line start offsets
    line_starts: Vec<usize>,
}

impl SourceFile {
    /// Create a new source file
    pub fn new(id: u32, path: PathBuf, source: String) -> Self {
        let line_starts = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(i, _)| i + 1))
            .collect();

        SourceFile {
            id,
            path,
            source,
            line_starts,
        }
    }

    /// Get line and column from byte offset
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let line = self
            .line_starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        let line_start = self.line_starts.get(line).copied().unwrap_or(0);
        let col = offset - line_start;
        (line + 1, col + 1) // 1-indexed
    }

    /// Get the content of a line (0-indexed)
    pub fn line(&self, line: usize) -> Option<&str> {
        let start = self.line_starts.get(line)?;
        let end = self
            .line_starts
            .get(line + 1)
            .map(|e| e - 1) // Don't include newline
            .unwrap_or(self.source.len());
        Some(&self.source[*start..end])
    }

    /// Get number of lines
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Get snippet around a span
    pub fn snippet(&self, span: Span, context_lines: usize) -> String {
        let (start_line, _) = self.line_col(span.start);
        let (end_line, _) = self.line_col(span.end);

        let first_line = start_line.saturating_sub(context_lines + 1);
        let last_line = (end_line + context_lines).min(self.line_count());

        let mut result = String::new();
        for line_num in first_line..last_line {
            if let Some(line) = self.line(line_num) {
                result.push_str(&format!("{:4} | {}\n", line_num + 1, line));
            }
        }
        result
    }
}

/// Source map for looking up file info from spans
#[derive(Debug, Default)]
pub struct SourceMap {
    files: HashMap<u32, SourceFile>,
    next_id: u32,
}

impl SourceMap {
    /// Create a new source map
    pub fn new() -> Self {
        SourceMap {
            files: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a file to the source map
    pub fn add_file(&mut self, path: PathBuf, source: String) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.files.insert(id, SourceFile::new(id, path, source));
        id
    }

    /// Get a file by ID
    pub fn get_file(&self, id: u32) -> Option<&SourceFile> {
        self.files.get(&id)
    }

    /// Look up location for a span
    pub fn lookup_span(&self, span: Span) -> Option<SpanLocation> {
        let file = self.files.get(&span.file_id)?;
        let (start_line, start_col) = file.line_col(span.start);
        let (end_line, end_col) = file.line_col(span.end);

        Some(SpanLocation {
            file_path: file.path.clone(),
            start_line,
            start_col,
            end_line,
            end_col,
        })
    }
}

/// Resolved location information for a span
#[derive(Debug, Clone)]
pub struct SpanLocation {
    /// File path
    pub file_path: PathBuf,
    /// Start line (1-indexed)
    pub start_line: usize,
    /// Start column (1-indexed)
    pub start_col: usize,
    /// End line (1-indexed)
    pub end_line: usize,
    /// End column (1-indexed)
    pub end_col: usize,
}

impl fmt::Display for SpanLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.file_path.display(),
            self.start_line,
            self.start_col
        )
    }
}

/// Diagnostic handler that collects and reports diagnostics
pub struct DiagnosticHandler {
    /// Source map
    source_map: SourceMap,
    /// Emitter for output
    emitter: Box<dyn DiagnosticEmitter>,
    /// Error count
    error_count: usize,
    /// Warning count
    warning_count: usize,
    /// Whether to treat warnings as errors
    warnings_as_errors: bool,
    /// Maximum errors before stopping
    max_errors: Option<usize>,
}

impl DiagnosticHandler {
    /// Create a new diagnostic handler
    pub fn new(emitter: Box<dyn DiagnosticEmitter>) -> Self {
        DiagnosticHandler {
            source_map: SourceMap::new(),
            emitter,
            error_count: 0,
            warning_count: 0,
            warnings_as_errors: false,
            max_errors: None,
        }
    }

    /// Set the source map
    pub fn with_source_map(mut self, source_map: SourceMap) -> Self {
        self.source_map = source_map;
        self
    }

    /// Treat warnings as errors
    pub fn warnings_as_errors(mut self, enabled: bool) -> Self {
        self.warnings_as_errors = enabled;
        self
    }

    /// Set maximum error count
    pub fn max_errors(mut self, max: usize) -> Self {
        self.max_errors = Some(max);
        self
    }

    /// Get reference to source map
    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    /// Get mutable reference to source map
    pub fn source_map_mut(&mut self) -> &mut SourceMap {
        &mut self.source_map
    }

    /// Emit a diagnostic
    pub fn emit(&mut self, diagnostic: &Diagnostic) {
        // Update counts
        match diagnostic.level {
            DiagnosticLevel::Bug | DiagnosticLevel::Fatal | DiagnosticLevel::Error => {
                self.error_count += 1;
            }
            DiagnosticLevel::Warning => {
                self.warning_count += 1;
                if self.warnings_as_errors {
                    self.error_count += 1;
                }
            }
            _ => {}
        }

        // Emit the diagnostic
        self.emitter
            .emit_with_source_map(diagnostic, &self.source_map);

        // Check error limit
        if let Some(max) = self.max_errors
            && self.error_count >= max
        {
            self.emit_too_many_errors();
        }
    }

    /// Emit an error
    pub fn error(&mut self, message: impl Into<String>) -> DiagnosticBuilder {
        DiagnosticBuilder {
            diagnostic: Diagnostic::error(message),
        }
    }

    /// Emit a warning
    pub fn warning(&mut self, message: impl Into<String>) -> DiagnosticBuilder {
        DiagnosticBuilder {
            diagnostic: Diagnostic::warning(message),
        }
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Check if any errors occurred
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Emit summary and abort if there were errors
    pub fn abort_if_errors(&self) {
        if self.has_errors() {
            self.emitter
                .emit_summary(self.error_count, self.warning_count);
            std::process::exit(1);
        }
    }

    /// Emit "too many errors" and abort
    fn emit_too_many_errors(&self) {
        let msg = format!("aborting due to {} previous errors", self.error_count);
        self.emitter.emit(&Diagnostic::error(msg));
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_operations() {
        let span1 = Span::new(10, 20, 1);
        let span2 = Span::new(15, 30, 1);

        let merged = span1.merge(span2);
        assert_eq!(merged.start, 10);
        assert_eq!(merged.end, 30);
    }

    #[test]
    fn test_source_file_line_col() {
        let source = "line one\nline two\nline three";
        let file = SourceFile::new(1, PathBuf::from("test.sio"), source.to_string());

        assert_eq!(file.line_col(0), (1, 1));
        assert_eq!(file.line_col(5), (1, 6));
        assert_eq!(file.line_col(9), (2, 1));
        assert_eq!(file.line_col(14), (2, 6));
    }

    #[test]
    fn test_diagnostic_builder() {
        let diagnostic = DiagnosticBuilder::error("E0001", "Type mismatch")
            .with_span(Span::new(10, 20, 1))
            .with_label(Span::new(10, 20, 1), "expected `int`, found `bool`")
            .with_help("check the function return type")
            .build();

        assert_eq!(diagnostic.level, DiagnosticLevel::Error);
        assert_eq!(diagnostic.code, Some("E0001".to_string()));
        assert_eq!(diagnostic.message, "Type mismatch");
        assert_eq!(diagnostic.labels.len(), 2);
        assert_eq!(diagnostic.help.len(), 1);
    }

    #[test]
    fn test_diagnostic_level_ordering() {
        // Enum order: Bug (most severe) < Fatal < Error < Warning < Note < Help (least severe)
        // In terms of enum discriminant values, earlier variants have smaller values
        assert!(DiagnosticLevel::Bug < DiagnosticLevel::Error);
        assert!(DiagnosticLevel::Error < DiagnosticLevel::Warning);
        assert!(DiagnosticLevel::Warning < DiagnosticLevel::Note);
    }
}
