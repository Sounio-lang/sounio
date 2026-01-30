//! Source file management
//!
//! Provides a database of source files with efficient line/column lookup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Unique identifier for a source file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

impl FileId {
    pub const DUMMY: FileId = FileId(u32::MAX);

    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn is_dummy(self) -> bool {
        self.0 == u32::MAX
    }
}

impl Default for FileId {
    fn default() -> Self {
        Self::DUMMY
    }
}

/// A source file with line information
#[derive(Debug, Clone)]
pub struct SourceFile {
    /// File ID
    pub id: FileId,

    /// File path
    pub path: PathBuf,

    /// File contents
    pub source: Arc<str>,

    /// Line start byte offsets (0-indexed)
    line_starts: Vec<usize>,
}

impl SourceFile {
    pub fn new(id: FileId, path: PathBuf, source: String) -> Self {
        let line_starts = compute_line_starts(&source);

        Self {
            id,
            path,
            source: source.into(),
            line_starts,
        }
    }

    /// Get line and column for a byte offset (1-indexed)
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let line = self
            .line_starts
            .iter()
            .position(|&start| start > offset)
            .map(|l| l - 1)
            .unwrap_or(self.line_starts.len().saturating_sub(1));

        let line_start = self.line_starts.get(line).copied().unwrap_or(0);
        let col = offset.saturating_sub(line_start);

        (line + 1, col + 1) // 1-indexed
    }

    /// Get the text of a line (1-indexed)
    pub fn line_text(&self, line: usize) -> Option<&str> {
        if line == 0 || line > self.line_starts.len() {
            return None;
        }

        let start = self.line_starts[line - 1];
        let end = self
            .line_starts
            .get(line)
            .copied()
            .unwrap_or(self.source.len());

        Some(
            self.source[start..end]
                .trim_end_matches('\n')
                .trim_end_matches('\r'),
        )
    }

    /// Get the number of lines
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Get file name
    pub fn name(&self) -> &str {
        self.path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unknown>")
    }

    /// Get the source text for a span
    pub fn span_text(&self, start: usize, end: usize) -> &str {
        let start = start.min(self.source.len());
        let end = end.min(self.source.len());
        &self.source[start..end]
    }
}

/// Compute line start offsets
fn compute_line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];

    for (i, c) in source.char_indices() {
        if c == '\n' {
            starts.push(i + 1);
        }
    }

    starts
}

/// Source file database
#[derive(Debug, Default)]
pub struct SourceDb {
    files: HashMap<FileId, SourceFile>,
    paths: HashMap<PathBuf, FileId>,
    next_id: u32,
}

impl SourceDb {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a source file
    pub fn add_file(&mut self, path: PathBuf, source: String) -> FileId {
        if let Some(&id) = self.paths.get(&path) {
            return id;
        }

        let id = FileId::new(self.next_id);
        self.next_id += 1;

        let file = SourceFile::new(id, path.clone(), source);
        self.files.insert(id, file);
        self.paths.insert(path, id);

        id
    }

    /// Add a virtual file (e.g., from REPL)
    pub fn add_virtual(&mut self, name: &str, source: String) -> FileId {
        let path = PathBuf::from(format!("<{}>", name));
        self.add_file(path, source)
    }

    /// Get a file by ID
    pub fn get(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(&id)
    }

    /// Get a file by path
    pub fn get_by_path(&self, path: &Path) -> Option<&SourceFile> {
        self.paths.get(path).and_then(|id| self.files.get(id))
    }

    /// Get line/column for a file and offset
    pub fn line_col(&self, file: FileId, offset: usize) -> Option<(usize, usize)> {
        self.get(file).map(|f| f.line_col(offset))
    }

    /// Get all files
    pub fn files(&self) -> impl Iterator<Item = &SourceFile> {
        self.files.values()
    }

    /// Get file count
    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_col() {
        let file = SourceFile::new(
            FileId::new(0),
            PathBuf::from("test.sio"),
            "line 1\nline 2\nline 3".to_string(),
        );

        assert_eq!(file.line_col(0), (1, 1)); // Start of line 1
        assert_eq!(file.line_col(5), (1, 6)); // "1" in line 1
        assert_eq!(file.line_col(7), (2, 1)); // Start of line 2
        assert_eq!(file.line_col(14), (3, 1)); // Start of line 3
    }

    #[test]
    fn test_line_text() {
        let file = SourceFile::new(
            FileId::new(0),
            PathBuf::from("test.sio"),
            "line 1\nline 2\nline 3".to_string(),
        );

        assert_eq!(file.line_text(1), Some("line 1"));
        assert_eq!(file.line_text(2), Some("line 2"));
        assert_eq!(file.line_text(3), Some("line 3"));
        assert_eq!(file.line_text(4), None);
    }

    #[test]
    fn test_source_db() {
        let mut db = SourceDb::new();

        let id1 = db.add_file(PathBuf::from("a.sio"), "fn main() {}".to_string());
        let id2 = db.add_file(PathBuf::from("b.sio"), "fn foo() {}".to_string());

        assert_ne!(id1, id2);
        assert_eq!(db.file_count(), 2);

        let file = db.get(id1).unwrap();
        assert_eq!(file.name(), "a.sio");
    }
}
