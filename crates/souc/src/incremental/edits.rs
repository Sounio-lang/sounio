use std::ops::Range;

/// A single text edit operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextEdit {
    /// Byte range in the original text that was replaced
    pub range: Range<usize>,

    /// The new text that replaced the range
    pub new_text: String,

    /// Length of the original text in the range
    pub old_len: usize,
}

impl TextEdit {
    pub fn new(range: Range<usize>, new_text: impl Into<String>) -> Self {
        let old_len = range.end - range.start;
        TextEdit {
            range,
            new_text: new_text.into(),
            old_len,
        }
    }

    /// Insert text at position
    pub fn insert(position: usize, text: impl Into<String>) -> Self {
        TextEdit::new(position..position, text)
    }

    /// Delete text in range
    pub fn delete(range: Range<usize>) -> Self {
        TextEdit::new(range, "")
    }

    /// Replace text in range
    pub fn replace(range: Range<usize>, text: impl Into<String>) -> Self {
        TextEdit::new(range, text)
    }

    /// Length change caused by this edit
    pub fn length_delta(&self) -> isize {
        self.new_text.len() as isize - self.old_len as isize
    }

    /// Does this edit affect the given byte offset?
    pub fn affects(&self, offset: usize) -> bool {
        offset >= self.range.start
    }

    /// Adjust an offset after this edit
    pub fn adjust_offset(&self, offset: usize) -> usize {
        if offset < self.range.start {
            offset
        } else if offset < self.range.end {
            // Offset was in the deleted range
            self.range.start + self.new_text.len()
        } else {
            // Offset was after the edit
            (offset as isize + self.length_delta()) as usize
        }
    }
}

/// A sequence of text edits
#[derive(Debug, Clone, Default)]
pub struct EditSequence {
    /// Edits in order of application
    edits: Vec<TextEdit>,

    /// Original text length
    original_len: usize,
}

impl EditSequence {
    pub fn new(original_len: usize) -> Self {
        EditSequence {
            edits: Vec::new(),
            original_len,
        }
    }

    /// Add an edit to the sequence
    pub fn push(&mut self, edit: TextEdit) {
        // Validate edit is within bounds
        debug_assert!(edit.range.end <= self.current_len());
        self.edits.push(edit);
    }

    /// Current length after all edits
    pub fn current_len(&self) -> usize {
        let delta: isize = self.edits.iter().map(|e| e.length_delta()).sum();
        (self.original_len as isize + delta) as usize
    }

    /// Apply edits to produce new text
    pub fn apply(&self, original: &str) -> String {
        debug_assert_eq!(original.len(), self.original_len);

        let mut result = original.to_string();

        // Apply edits in reverse order to maintain correct offsets
        for edit in self.edits.iter().rev() {
            result.replace_range(edit.range.clone(), &edit.new_text);
        }

        result
    }

    /// Get affected byte range in original text
    pub fn affected_range(&self) -> Option<Range<usize>> {
        if self.edits.is_empty() {
            return None;
        }

        let start = self.edits.iter().map(|e| e.range.start).min().unwrap();
        let end = self.edits.iter().map(|e| e.range.end).max().unwrap();

        Some(start..end)
    }

    /// Convert to a single consolidated edit
    pub fn consolidate(&self) -> Option<TextEdit> {
        self.affected_range().map(|range| {
            // This is approximate - full consolidation would need the text
            TextEdit {
                range: range.clone(),
                new_text: String::new(), // Would be computed from actual text
                old_len: range.end - range.start,
            }
        })
    }

    /// Compose with another edit sequence
    pub fn compose(&self, other: &EditSequence) -> EditSequence {
        let mut composed = self.clone();
        for edit in &other.edits {
            composed.push(edit.clone());
        }
        composed
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn edits(&self) -> &[TextEdit] {
        &self.edits
    }
}

/// Represents a change to source text with before/after snapshots
#[derive(Debug, Clone)]
pub struct TextChange {
    /// Text before the change
    pub old_text: String,

    /// Text after the change
    pub new_text: String,

    /// The edit that produced this change
    pub edit: TextEdit,
}

impl TextChange {
    pub fn new(old_text: String, new_text: String, edit: TextEdit) -> Self {
        TextChange {
            old_text,
            new_text,
            edit,
        }
    }

    /// Compute change from old and new text
    pub fn diff(old_text: &str, new_text: &str) -> Self {
        let edit = compute_minimal_edit(old_text, new_text);
        TextChange {
            old_text: old_text.to_string(),
            new_text: new_text.to_string(),
            edit,
        }
    }
}

/// Compute minimal edit to transform old_text to new_text
pub fn compute_minimal_edit(old_text: &str, new_text: &str) -> TextEdit {
    // Find common prefix
    let common_prefix = old_text
        .chars()
        .zip(new_text.chars())
        .take_while(|(a, b)| a == b)
        .count();

    let common_prefix_bytes = old_text
        .chars()
        .take(common_prefix)
        .map(|c| c.len_utf8())
        .sum::<usize>();

    // Find common suffix (not overlapping with prefix)
    let old_suffix = &old_text[common_prefix_bytes..];
    let new_suffix = &new_text[common_prefix_bytes..];

    let common_suffix = old_suffix
        .chars()
        .rev()
        .zip(new_suffix.chars().rev())
        .take_while(|(a, b)| a == b)
        .count();

    let common_suffix_bytes = old_suffix
        .chars()
        .rev()
        .take(common_suffix)
        .map(|c| c.len_utf8())
        .sum::<usize>();

    let old_len = old_text.len() - common_prefix_bytes - common_suffix_bytes;
    let new_len = new_text.len() - common_prefix_bytes - common_suffix_bytes;

    let replaced_text = &new_text[common_prefix_bytes..new_text.len() - common_suffix_bytes];

    TextEdit {
        range: common_prefix_bytes..(common_prefix_bytes + old_len),
        new_text: replaced_text.to_string(),
        old_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_edit() {
        let old = "hello world";
        let new = "hello rust world";
        let edit = compute_minimal_edit(old, new);

        assert_eq!(edit.range, 6..6);
        assert_eq!(edit.new_text, "rust ");
    }

    #[test]
    fn test_edit_sequence() {
        let mut seq = EditSequence::new(11);
        seq.push(TextEdit::insert(6, "beautiful "));

        let result = seq.apply("hello world");
        assert_eq!(result, "hello beautiful world");
    }
}
