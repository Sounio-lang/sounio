//! Source locations
//!
//! Provides span and location types for tracking source positions.

use super::files::FileId;
use std::fmt;

/// A span in the source code with file information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    /// Source file
    pub file: FileId,

    /// Start byte offset
    pub start: u32,

    /// End byte offset
    pub end: u32,
}

impl Span {
    /// Create a new span
    pub fn new(file: FileId, start: u32, end: u32) -> Self {
        Self { file, start, end }
    }

    /// Create a span from a basic span (without file info)
    pub fn from_basic(file: FileId, basic: crate::common::Span) -> Self {
        Self {
            file,
            start: basic.start as u32,
            end: basic.end as u32,
        }
    }

    /// Create a dummy span
    pub fn dummy() -> Self {
        Self {
            file: FileId::DUMMY,
            start: 0,
            end: 0,
        }
    }

    /// Check if this is a dummy span
    pub fn is_dummy(&self) -> bool {
        self.file.is_dummy()
    }

    /// Merge two spans (returns span covering both)
    pub fn merge(&self, other: &Span) -> Span {
        if self.is_dummy() {
            return *other;
        }
        if other.is_dummy() {
            return *self;
        }

        debug_assert_eq!(
            self.file, other.file,
            "Cannot merge spans from different files"
        );

        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Get the length of this span
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Get start as usize
    pub fn start_usize(&self) -> usize {
        self.start as usize
    }

    /// Get end as usize
    pub fn end_usize(&self) -> usize {
        self.end as usize
    }

    /// Convert to a basic span (losing file info)
    pub fn to_basic(&self) -> crate::common::Span {
        crate::common::Span {
            start: self.start as usize,
            end: self.end as usize,
        }
    }
}

/// A located value (value with span)
#[derive(Debug, Clone)]
pub struct Located<T> {
    pub value: T,
    pub span: Span,
}

impl<T> Located<T> {
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Located<U> {
        Located {
            value: f(self.value),
            span: self.span,
        }
    }

    pub fn as_ref(&self) -> Located<&T> {
        Located {
            value: &self.value,
            span: self.span,
        }
    }
}

impl<T: Clone> Located<T> {
    pub fn cloned(&self) -> Located<T> {
        Located {
            value: self.value.clone(),
            span: self.span,
        }
    }
}

/// Source location (file, line, column)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    pub file: FileId,
    pub line: usize,
    pub column: usize,
}

impl SourceLocation {
    pub fn new(file: FileId, line: usize, column: usize) -> Self {
        Self { file, line, column }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A range of source locations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceRange {
    pub file: FileId,
    pub start_line: usize,
    pub start_column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

impl SourceRange {
    pub fn new(start: SourceLocation, end: SourceLocation) -> Self {
        debug_assert_eq!(start.file, end.file);
        Self {
            file: start.file,
            start_line: start.line,
            start_column: start.column,
            end_line: end.line,
            end_column: end.column,
        }
    }

    pub fn single_line(file: FileId, line: usize, start_col: usize, end_col: usize) -> Self {
        Self {
            file,
            start_line: line,
            start_column: start_col,
            end_line: line,
            end_column: end_col,
        }
    }
}

impl fmt::Display for SourceRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start_line == self.end_line {
            write!(
                f,
                "{}:{}-{}",
                self.start_line, self.start_column, self.end_column
            )
        } else {
            write!(
                f,
                "{}:{}-{}:{}",
                self.start_line, self.start_column, self.end_line, self.end_column
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_merge() {
        let file = FileId::new(1);
        let s1 = Span::new(file, 5, 10);
        let s2 = Span::new(file, 8, 15);
        let merged = s1.merge(&s2);

        assert_eq!(merged.start, 5);
        assert_eq!(merged.end, 15);
        assert_eq!(merged.file, file);
    }

    #[test]
    fn test_span_dummy() {
        let dummy = Span::dummy();
        assert!(dummy.is_dummy());

        let real = Span::new(FileId::new(1), 0, 10);
        assert!(!real.is_dummy());

        // Merging with dummy should return the real span
        assert_eq!(dummy.merge(&real), real);
        assert_eq!(real.merge(&dummy), real);
    }

    #[test]
    fn test_located() {
        let loc = Located::new(42, Span::new(FileId::new(0), 0, 2));
        assert_eq!(loc.value, 42);

        let mapped = loc.map(|x| x * 2);
        assert_eq!(mapped.value, 84);
    }
}
