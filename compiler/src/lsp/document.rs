//! Document management with rope data structure
//!
//! Provides efficient text operations for LSP document synchronization.

use ropey::Rope;
use tower_lsp::lsp_types::*;

use crate::common::Span;

/// A line index for efficient byte offset to line/column conversion.
///
/// This is a lightweight structure that can be created from source text
/// and used for span-to-range conversions without needing the full rope.
#[derive(Debug, Clone)]
pub struct LineIndex {
    /// Byte offset of each line start (line 0 starts at offset 0)
    line_starts: Vec<usize>,
    /// Total length in bytes
    len: usize,
}

impl LineIndex {
    /// Create a new line index from source text
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            line_starts,
            len: source.len(),
        }
    }

    /// Convert a byte offset to an LSP Position
    pub fn offset_to_position(&self, offset: usize) -> Position {
        let offset = offset.min(self.len);

        // Binary search to find the line
        let line = match self.line_starts.binary_search(&offset) {
            Ok(line) => line,
            Err(line) => line.saturating_sub(1),
        };

        let line_start = self.line_starts.get(line).copied().unwrap_or(0);
        let character = offset.saturating_sub(line_start);

        Position {
            line: line as u32,
            character: character as u32,
        }
    }

    /// Convert an LSP Position to a byte offset
    pub fn position_to_offset(&self, pos: Position) -> Option<usize> {
        let line = pos.line as usize;
        if line >= self.line_starts.len() {
            return None;
        }

        let line_start = self.line_starts[line];
        let offset = line_start + pos.character as usize;

        if offset <= self.len {
            Some(offset)
        } else {
            None
        }
    }

    /// Convert a Span to an LSP Range
    pub fn span_to_range(&self, span: &Span) -> Range {
        Range {
            start: self.offset_to_position(span.start),
            end: self.offset_to_position(span.end),
        }
    }

    /// Get the number of lines
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Get the byte offset of a line start
    pub fn line_start(&self, line: usize) -> Option<usize> {
        self.line_starts.get(line).copied()
    }
}

/// A managed document with efficient text operations
#[derive(Debug, Clone)]
pub struct Document {
    /// Document content as rope for efficient edits
    rope: Rope,
    /// Document version (increments on each change)
    version: i32,
}

impl Document {
    /// Create a new document from text
    pub fn new(text: String, version: i32) -> Self {
        Self {
            rope: Rope::from_str(&text),
            version,
        }
    }

    /// Get the document version
    pub fn version(&self) -> i32 {
        self.version
    }

    /// Get the full text
    pub fn text(&self) -> String {
        self.rope.to_string()
    }

    /// Get a line by index (0-based)
    pub fn line(&self, idx: usize) -> Option<String> {
        if idx < self.rope.len_lines() {
            Some(self.rope.line(idx).to_string())
        } else {
            None
        }
    }

    /// Get number of lines
    pub fn line_count(&self) -> usize {
        self.rope.len_lines()
    }

    /// Get total byte length
    pub fn len(&self) -> usize {
        self.rope.len_bytes()
    }

    /// Check if document is empty
    pub fn is_empty(&self) -> bool {
        self.rope.len_bytes() == 0
    }

    /// Convert LSP position to byte offset
    pub fn position_to_offset(&self, pos: Position) -> Option<usize> {
        let line_idx = pos.line as usize;

        if line_idx >= self.rope.len_lines() {
            return None;
        }

        let line_start = self.rope.line_to_byte(line_idx);
        let line = self.rope.line(line_idx);

        // Convert character offset to byte offset within line
        let char_idx = pos.character as usize;
        let byte_offset: usize = line.chars().take(char_idx).map(|c| c.len_utf8()).sum();

        Some(line_start + byte_offset)
    }

    /// Convert byte offset to LSP position
    pub fn offset_to_position(&self, offset: usize) -> Position {
        let offset = offset.min(self.rope.len_bytes());
        let line = self.rope.byte_to_line(offset);
        let line_start = self.rope.line_to_byte(line);
        let col_bytes = offset - line_start;

        // Convert byte offset to character offset
        let line_text = self.rope.line(line);
        let mut char_count = 0;
        let mut byte_count = 0;

        for c in line_text.chars() {
            if byte_count >= col_bytes {
                break;
            }
            byte_count += c.len_utf8();
            char_count += 1;
        }

        Position {
            line: line as u32,
            character: char_count as u32,
        }
    }

    /// Apply an incremental change from LSP
    pub fn apply_change(&mut self, change: TextDocumentContentChangeEvent, version: i32) {
        self.version = version;

        if let Some(range) = change.range {
            // Incremental update
            let start = self.position_to_offset(range.start).unwrap_or(0);
            let end = self
                .position_to_offset(range.end)
                .unwrap_or(self.rope.len_bytes());

            // Remove old text
            if start < end && end <= self.rope.len_bytes() {
                self.rope.remove(start..end);
            }

            // Insert new text
            if start <= self.rope.len_bytes() {
                self.rope.insert(start, &change.text);
            }
        } else {
            // Full document replace
            self.rope = Rope::from_str(&change.text);
        }
    }

    /// Get word at position
    pub fn word_at(&self, pos: Position) -> Option<(String, Range)> {
        let offset = self.position_to_offset(pos)?;
        let line_idx = pos.line as usize;

        if line_idx >= self.rope.len_lines() {
            return None;
        }

        let line = self.rope.line(line_idx);
        let line_start = self.rope.line_to_byte(line_idx);
        let col = offset.saturating_sub(line_start);

        let line_str = line.to_string();
        let bytes = line_str.as_bytes();

        if col > bytes.len() {
            return None;
        }

        // Find word boundaries
        let mut start = col;
        while start > 0 && is_word_char(bytes[start - 1]) {
            start -= 1;
        }

        let mut end = col;
        while end < bytes.len() && is_word_char(bytes[end]) {
            end += 1;
        }

        if start == end {
            return None;
        }

        let word = String::from_utf8_lossy(&bytes[start..end]).to_string();

        let range = Range {
            start: Position {
                line: pos.line,
                character: start as u32,
            },
            end: Position {
                line: pos.line,
                character: end as u32,
            },
        };

        Some((word, range))
    }

    /// Get the text in a range
    pub fn text_in_range(&self, range: Range) -> Option<String> {
        let start = self.position_to_offset(range.start)?;
        let end = self.position_to_offset(range.end)?;

        if start <= end && end <= self.rope.len_bytes() {
            Some(self.rope.slice(start..end).to_string())
        } else {
            None
        }
    }

    /// Get context around a position (for completion, etc.)
    pub fn get_context(&self, pos: Position, chars_before: usize) -> Option<String> {
        let offset = self.position_to_offset(pos)?;
        let start = offset.saturating_sub(chars_before);

        if start <= offset && offset <= self.rope.len_bytes() {
            Some(self.rope.slice(start..offset).to_string())
        } else {
            None
        }
    }
}

/// Check if a byte is a valid word character
fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_offset_conversion() {
        let doc = Document::new("hello\nworld\n".to_string(), 1);

        let pos = Position {
            line: 1,
            character: 2,
        };
        let offset = doc.position_to_offset(pos).unwrap();
        assert_eq!(offset, 8); // "hello\nwo" = 8 bytes

        let back = doc.offset_to_position(offset);
        assert_eq!(back, pos);
    }

    #[test]
    fn test_word_at() {
        let doc = Document::new("let foo = bar".to_string(), 1);

        let (word, _) = doc
            .word_at(Position {
                line: 0,
                character: 5,
            })
            .unwrap();
        assert_eq!(word, "foo");

        let (word2, _) = doc
            .word_at(Position {
                line: 0,
                character: 11,
            })
            .unwrap();
        assert_eq!(word2, "bar");
    }

    #[test]
    fn test_apply_incremental_change() {
        let mut doc = Document::new("hello world".to_string(), 1);

        // Replace "world" with "rust"
        doc.apply_change(
            TextDocumentContentChangeEvent {
                range: Some(Range {
                    start: Position {
                        line: 0,
                        character: 6,
                    },
                    end: Position {
                        line: 0,
                        character: 11,
                    },
                }),
                range_length: None,
                text: "rust".to_string(),
            },
            2,
        );

        assert_eq!(doc.text(), "hello rust");
        assert_eq!(doc.version(), 2);
    }

    #[test]
    fn test_full_document_replace() {
        let mut doc = Document::new("old content".to_string(), 1);

        doc.apply_change(
            TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: "new content".to_string(),
            },
            2,
        );

        assert_eq!(doc.text(), "new content");
    }

    #[test]
    fn test_multiline_document() {
        let doc = Document::new("line 1\nline 2\nline 3".to_string(), 1);

        assert_eq!(doc.line_count(), 3);
        assert_eq!(doc.line(0), Some("line 1\n".to_string()));
        assert_eq!(doc.line(1), Some("line 2\n".to_string()));
        assert_eq!(doc.line(2), Some("line 3".to_string()));

        let pos = doc.offset_to_position(7); // Start of "line 2"
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 0);
    }

    #[test]
    fn test_line_index_basic() {
        let source = "hello\nworld\nfoo";
        let idx = LineIndex::new(source);

        assert_eq!(idx.line_count(), 3);
        assert_eq!(idx.line_start(0), Some(0));
        assert_eq!(idx.line_start(1), Some(6)); // After "hello\n"
        assert_eq!(idx.line_start(2), Some(12)); // After "world\n"
    }

    #[test]
    fn test_line_index_offset_to_position() {
        let source = "hello\nworld\nfoo";
        let idx = LineIndex::new(source);

        // Start of file
        assert_eq!(idx.offset_to_position(0), Position { line: 0, character: 0 });

        // Middle of first line
        assert_eq!(idx.offset_to_position(2), Position { line: 0, character: 2 });

        // Start of second line
        assert_eq!(idx.offset_to_position(6), Position { line: 1, character: 0 });

        // Middle of second line
        assert_eq!(idx.offset_to_position(8), Position { line: 1, character: 2 });

        // Start of third line
        assert_eq!(idx.offset_to_position(12), Position { line: 2, character: 0 });

        // End of file
        assert_eq!(idx.offset_to_position(15), Position { line: 2, character: 3 });
    }

    #[test]
    fn test_line_index_position_to_offset() {
        let source = "hello\nworld\nfoo";
        let idx = LineIndex::new(source);

        assert_eq!(idx.position_to_offset(Position { line: 0, character: 0 }), Some(0));
        assert_eq!(idx.position_to_offset(Position { line: 1, character: 0 }), Some(6));
        assert_eq!(idx.position_to_offset(Position { line: 2, character: 3 }), Some(15));

        // Out of bounds
        assert_eq!(idx.position_to_offset(Position { line: 10, character: 0 }), None);
    }

    #[test]
    fn test_line_index_span_to_range() {
        use crate::common::Span;

        let source = "hello\nworld\nfoo";
        let idx = LineIndex::new(source);

        // Span within first line
        let span = Span { start: 0, end: 5 };
        let range = idx.span_to_range(&span);
        assert_eq!(range.start, Position { line: 0, character: 0 });
        assert_eq!(range.end, Position { line: 0, character: 5 });

        // Span crossing lines
        let span = Span { start: 3, end: 9 };
        let range = idx.span_to_range(&span);
        assert_eq!(range.start, Position { line: 0, character: 3 });
        assert_eq!(range.end, Position { line: 1, character: 3 });
    }

    #[test]
    fn test_line_index_empty_source() {
        let idx = LineIndex::new("");
        assert_eq!(idx.line_count(), 1);
        assert_eq!(idx.offset_to_position(0), Position { line: 0, character: 0 });
    }

    #[test]
    fn test_line_index_single_line() {
        let idx = LineIndex::new("hello");
        assert_eq!(idx.line_count(), 1);
        assert_eq!(idx.offset_to_position(3), Position { line: 0, character: 3 });
    }
}
