//! Common types and utilities used throughout the compiler

use serde::{Deserialize, Serialize};
use std::fmt;
use string_interner::Symbol as SymbolTrait;

/// Source span (byte offsets)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// Unique identifier for AST nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn dummy() -> Self {
        Self(0)
    }
}

/// Counter for generating unique IDs
#[derive(Default)]
pub struct IdGenerator {
    next: u32,
}

impl IdGenerator {
    pub fn new() -> Self {
        Self { next: 1 }
    }

    pub fn with_start(start: u32) -> Self {
        Self { next: start }
    }

    pub fn next(&mut self) -> NodeId {
        let id = NodeId(self.next);
        self.next += 1;
        id
    }

    pub fn next_value(&self) -> u32 {
        self.next
    }
}

/// Interned string
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(pub u32);

impl Symbol {
    pub fn dummy() -> Self {
        Self(0)
    }
}

/// String interner for efficient string storage
pub struct Interner {
    interner: string_interner::StringInterner<string_interner::backend::StringBackend>,
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

impl Interner {
    pub fn new() -> Self {
        Self {
            interner: string_interner::StringInterner::new(),
        }
    }

    pub fn intern(&mut self, s: &str) -> Symbol {
        Symbol(self.interner.get_or_intern(s).to_usize() as u32)
    }

    pub fn resolve(&self, sym: Symbol) -> Option<&str> {
        self.interner
            .resolve(string_interner::symbol::SymbolU32::try_from_usize(
                sym.0 as usize,
            )?)
    }
}

/// Source file information
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// File path (or "<stdin>" for REPL)
    pub path: String,
    /// Source code content
    pub content: String,
    /// Line start byte offsets
    line_starts: Vec<usize>,
}

impl SourceFile {
    pub fn new(path: String, content: String) -> Self {
        let line_starts = std::iter::once(0)
            .chain(content.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        Self {
            path,
            content,
            line_starts,
        }
    }

    pub fn from_str(content: &str) -> Self {
        Self::new("<input>".to_string(), content.to_string())
    }

    /// Get line and column for a byte offset
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let line = self
            .line_starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        let col = offset - self.line_starts.get(line).copied().unwrap_or(0);
        (line + 1, col + 1)
    }

    /// Get the line containing an offset
    pub fn line_at(&self, offset: usize) -> &str {
        let (line, _) = self.line_col(offset);
        self.line(line)
    }

    /// Get a specific line (1-indexed)
    pub fn line(&self, line: usize) -> &str {
        if line == 0 || line > self.line_starts.len() {
            return "";
        }
        let start = self.line_starts[line - 1];
        let end = self
            .line_starts
            .get(line)
            .copied()
            .unwrap_or(self.content.len());
        self.content[start..end].trim_end_matches('\n')
    }
}

/// Diagnostic severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

/// A compiler diagnostic
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Span,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            span,
            notes: Vec::new(),
        }
    }

    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            span,
            notes: Vec::new(),
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_merge() {
        let s1 = Span::new(5, 10);
        let s2 = Span::new(8, 15);
        let merged = s1.merge(s2);
        assert_eq!(merged.start, 5);
        assert_eq!(merged.end, 15);
    }

    #[test]
    fn test_interner() {
        let mut interner = Interner::new();
        let s1 = interner.intern("hello");
        let s2 = interner.intern("world");
        let s3 = interner.intern("hello");

        assert_eq!(s1, s3);
        assert_ne!(s1, s2);
        assert_eq!(interner.resolve(s1), Some("hello"));
        assert_eq!(interner.resolve(s2), Some("world"));
    }

    #[test]
    fn test_source_file_line_col() {
        let src = SourceFile::from_str("line 1\nline 2\nline 3");
        assert_eq!(src.line_col(0), (1, 1));
        assert_eq!(src.line_col(6), (1, 7));
        assert_eq!(src.line_col(7), (2, 1));
        assert_eq!(src.line_col(14), (3, 1));
    }
}
