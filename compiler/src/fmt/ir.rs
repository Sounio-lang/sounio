//! Intermediate representation for formatted documents
//!
//! Based on Wadler's "A Prettier Printer" algorithm with extensions
//! for practical code formatting.

/// Document IR for formatting
#[derive(Debug, Clone, Default)]
pub enum Doc {
    /// Empty document
    #[default]
    Empty,

    /// Literal text
    Text(String),

    /// Hard line break (always breaks)
    Hardline,

    /// Soft line break (breaks in multi-line mode, space in flat mode)
    Softline,

    /// Line break (breaks in multi-line mode, empty in flat mode)
    Line,

    /// Concatenation
    Concat(Vec<Doc>),

    /// Indented content
    Indent(Box<Doc>),

    /// Dedented content
    Dedent(Box<Doc>),

    /// Align to current column
    Align(Box<Doc>),

    /// Group (try to fit on one line, break if needed)
    Group(Box<Doc>),

    /// Conditional (different docs for flat vs break mode)
    IfBreak {
        break_doc: Box<Doc>,
        flat_doc: Box<Doc>,
    },

    /// Fill (wrap words)
    Fill(Vec<Doc>),

    /// Line suffix (comments at end of line)
    LineSuffix(Box<Doc>),

    /// Break parent (force parent group to break)
    BreakParent,

    /// Trim trailing whitespace
    Trim,
}

impl Doc {
    /// Create text document
    pub fn text(s: impl Into<String>) -> Doc {
        Doc::Text(s.into())
    }

    /// Create concatenation document
    pub fn concat(docs: Vec<Doc>) -> Doc {
        // Flatten nested concats and remove empties
        let mut result = Vec::new();
        for doc in docs {
            match doc {
                Doc::Empty => {}
                Doc::Concat(inner) => result.extend(inner),
                other => result.push(other),
            }
        }
        if result.is_empty() {
            Doc::Empty
        } else if result.len() == 1 {
            result.pop().unwrap()
        } else {
            Doc::Concat(result)
        }
    }

    /// Create indented document
    pub fn indent(doc: Doc) -> Doc {
        Doc::Indent(Box::new(doc))
    }

    /// Create dedented document
    pub fn dedent(doc: Doc) -> Doc {
        Doc::Dedent(Box::new(doc))
    }

    /// Create aligned document
    pub fn align(doc: Doc) -> Doc {
        Doc::Align(Box::new(doc))
    }

    /// Create a group
    pub fn group(doc: Doc) -> Doc {
        Doc::Group(Box::new(doc))
    }

    /// Create conditional document
    pub fn if_break(break_doc: Doc, flat_doc: Doc) -> Doc {
        Doc::IfBreak {
            break_doc: Box::new(break_doc),
            flat_doc: Box::new(flat_doc),
        }
    }

    /// Create fill document for word wrapping
    pub fn fill(docs: Vec<Doc>) -> Doc {
        Doc::Fill(docs)
    }

    /// Create line suffix (for trailing comments)
    pub fn line_suffix(doc: Doc) -> Doc {
        Doc::LineSuffix(Box::new(doc))
    }

    /// Join documents with separator
    pub fn join(docs: Vec<Doc>, sep: Doc) -> Doc {
        if docs.is_empty() {
            return Doc::Empty;
        }

        let mut result = Vec::with_capacity(docs.len() * 2 - 1);
        for (i, doc) in docs.into_iter().enumerate() {
            if i > 0 {
                result.push(sep.clone());
            }
            result.push(doc);
        }
        Doc::Concat(result)
    }

    /// Join documents with softline separator
    pub fn softline_join(docs: Vec<Doc>) -> Doc {
        Doc::join(docs, Doc::Softline)
    }

    /// Join documents with hardline separator
    pub fn hardline_join(docs: Vec<Doc>) -> Doc {
        Doc::join(docs, Doc::Hardline)
    }

    /// Surround with brackets (with smart breaking)
    pub fn bracket(left: &str, doc: Doc, right: &str) -> Doc {
        Doc::Group(Box::new(Doc::Concat(vec![
            Doc::Text(left.to_string()),
            Doc::Indent(Box::new(Doc::Concat(vec![Doc::Softline, doc]))),
            Doc::Softline,
            Doc::Text(right.to_string()),
        ])))
    }

    /// Surround with brackets (forcing break)
    pub fn bracket_break(left: &str, doc: Doc, right: &str) -> Doc {
        Doc::Concat(vec![
            Doc::Text(left.to_string()),
            Doc::Indent(Box::new(Doc::Concat(vec![Doc::Hardline, doc]))),
            Doc::Hardline,
            Doc::Text(right.to_string()),
        ])
    }

    /// Check if document is empty
    pub fn is_empty(&self) -> bool {
        match self {
            Doc::Empty => true,
            Doc::Text(s) => s.is_empty(),
            Doc::Concat(docs) => docs.iter().all(|d| d.is_empty()),
            _ => false,
        }
    }

    /// Estimate the width if printed flat
    pub fn flat_width(&self) -> Option<usize> {
        match self {
            Doc::Empty => Some(0),
            Doc::Text(s) => Some(s.len()),
            Doc::Hardline => None, // Cannot be flat
            Doc::Softline => Some(1),
            Doc::Line => Some(0),
            Doc::Concat(docs) => {
                let mut total = 0;
                for doc in docs {
                    total += doc.flat_width()?;
                }
                Some(total)
            }
            Doc::Indent(inner) | Doc::Dedent(inner) | Doc::Align(inner) | Doc::Group(inner) => {
                inner.flat_width()
            }
            Doc::IfBreak { flat_doc, .. } => flat_doc.flat_width(),
            Doc::Fill(docs) => {
                let mut total = 0;
                for doc in docs {
                    total += doc.flat_width()?;
                }
                Some(total)
            }
            Doc::LineSuffix(_) => Some(0),
            Doc::BreakParent => None,
            Doc::Trim => Some(0),
        }
    }
}

/// Builder for constructing documents
pub struct DocBuilder {
    parts: Vec<Doc>,
}

impl DocBuilder {
    /// Create new builder
    pub fn new() -> Self {
        DocBuilder { parts: Vec::new() }
    }

    /// Add text
    pub fn text(mut self, s: impl Into<String>) -> Self {
        self.parts.push(Doc::Text(s.into()));
        self
    }

    /// Add hardline
    pub fn hardline(mut self) -> Self {
        self.parts.push(Doc::Hardline);
        self
    }

    /// Add softline
    pub fn softline(mut self) -> Self {
        self.parts.push(Doc::Softline);
        self
    }

    /// Add another document
    pub fn append(mut self, doc: Doc) -> Self {
        self.parts.push(doc);
        self
    }

    /// Indent remaining content
    pub fn indent(mut self) -> Self {
        if !self.parts.is_empty() {
            let last = self.parts.pop().unwrap();
            self.parts.push(Doc::Indent(Box::new(last)));
        }
        self
    }

    /// Build the final document
    pub fn build(self) -> Doc {
        Doc::concat(self.parts)
    }
}

impl Default for DocBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_join() {
        let docs = vec![
            Doc::Text("a".to_string()),
            Doc::Text("b".to_string()),
            Doc::Text("c".to_string()),
        ];
        let joined = Doc::join(docs, Doc::Text(", ".to_string()));

        match joined {
            Doc::Concat(parts) => {
                assert_eq!(parts.len(), 5);
            }
            _ => panic!("Expected Concat"),
        }
    }

    #[test]
    fn test_doc_concat_flatten() {
        let inner = Doc::Concat(vec![Doc::Text("a".to_string()), Doc::Text("b".to_string())]);
        let outer = Doc::concat(vec![inner, Doc::Text("c".to_string())]);

        match outer {
            Doc::Concat(parts) => {
                assert_eq!(parts.len(), 3);
            }
            _ => panic!("Expected Concat"),
        }
    }

    #[test]
    fn test_flat_width() {
        let doc = Doc::Concat(vec![
            Doc::Text("hello".to_string()),
            Doc::Text(" ".to_string()),
            Doc::Text("world".to_string()),
        ]);
        assert_eq!(doc.flat_width(), Some(11));

        let doc_with_hardline = Doc::Concat(vec![
            Doc::Text("hello".to_string()),
            Doc::Hardline,
            Doc::Text("world".to_string()),
        ]);
        assert_eq!(doc_with_hardline.flat_width(), None);
    }

    #[test]
    fn test_doc_builder() {
        let doc = DocBuilder::new().text("fn ").text("foo").text("()").build();

        match doc {
            Doc::Concat(parts) => {
                assert_eq!(parts.len(), 3);
            }
            _ => panic!("Expected Concat"),
        }
    }
}
