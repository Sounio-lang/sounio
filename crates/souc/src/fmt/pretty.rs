//! Pretty-printing algorithm for code formatting
//!
//! This module provides a Wadler-Lindig style pretty-printing implementation
//! with the following components:
//!
//! - `Doc` - The document IR with combinators (text, line, nest, group, etc.)
//! - `Printer` - The actual printing algorithm that renders Doc to string
//!
//! # Wadler-Lindig Algorithm
//!
//! The algorithm works in two phases:
//! 1. Build a `Doc` tree representing the document structure
//! 2. Print the `Doc` tree, making line-breaking decisions based on available width
//!
//! Key features:
//! - Groups try to fit on one line, breaking if they exceed line width
//! - Indentation is handled automatically via `Indent`
//! - Smart line breaks via `Softline` (breaks in break mode, space in flat mode)
//! - Hard line breaks via `Hardline` (always breaks)
//!
//! # Example
//!
//! ```ignore
//! use sounio::fmt::pretty::{Doc, Printer, FormatConfig};
//!
//! let doc = Doc::Group(Box::new(Doc::Concat(vec![
//!     Doc::text("fn foo("),
//!     Doc::Indent(Box::new(Doc::Concat(vec![
//!         Doc::Softline,
//!         Doc::text("x: i32"),
//!     ]))),
//!     Doc::Softline,
//!     Doc::text(")"),
//! ])));
//!
//! let printer = Printer::new(FormatConfig::default());
//! let output = printer.print(&doc);
//! ```

// Re-export the document IR
pub use super::ir::{Doc, DocBuilder};

// Re-export the printer
pub use super::printer::Printer;

// Re-export config for convenience
pub use super::config::FormatConfig;

/// Convenience function to create a text document
pub fn text(s: impl Into<String>) -> Doc {
    Doc::text(s)
}

/// Convenience function to create a concatenation
pub fn concat(docs: Vec<Doc>) -> Doc {
    Doc::concat(docs)
}

/// Convenience function to create an indented document
pub fn indent(doc: Doc) -> Doc {
    Doc::indent(doc)
}

/// Convenience function to create a group
pub fn group(doc: Doc) -> Doc {
    Doc::group(doc)
}

/// Convenience function to join documents with a separator
pub fn join(docs: Vec<Doc>, sep: Doc) -> Doc {
    Doc::join(docs, sep)
}

/// Convenience function to create bracketed content with smart breaking
pub fn bracket(left: &str, doc: Doc, right: &str) -> Doc {
    Doc::bracket(left, doc, right)
}

/// Print a document to string using default configuration
pub fn print_doc(doc: &Doc) -> String {
    let printer = Printer::new(FormatConfig::default());
    printer.print(doc)
}

/// Print a document to string with custom configuration
pub fn print_doc_with_config(doc: &Doc, config: FormatConfig) -> String {
    let printer = Printer::new(config);
    printer.print(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text() {
        let doc = text("hello");
        assert_eq!(print_doc(&doc).trim(), "hello");
    }

    #[test]
    fn test_concat() {
        let doc = concat(vec![text("hello"), text(" "), text("world")]);
        assert_eq!(print_doc(&doc).trim(), "hello world");
    }

    #[test]
    fn test_indent() {
        let doc = concat(vec![
            text("{"),
            indent(concat(vec![Doc::Hardline, text("x")])),
            Doc::Hardline,
            text("}"),
        ]);
        let output = print_doc(&doc);
        assert!(output.contains("    x")); // Default indent is 4 spaces
    }

    #[test]
    fn test_group_fits() {
        // Short content should stay on one line
        let doc = group(concat(vec![
            text("("),
            Doc::Softline,
            text("x"),
            Doc::Softline,
            text(")"),
        ]));
        let output = print_doc(&doc);
        assert!(output.contains("( x )"));
    }

    #[test]
    fn test_join() {
        let items = vec![text("a"), text("b"), text("c")];
        let doc = join(items, text(", "));
        assert_eq!(print_doc(&doc).trim(), "a, b, c");
    }

    #[test]
    fn test_bracket() {
        let inner = join(vec![text("x"), text("y")], text(", "));
        let doc = bracket("[", inner, "]");
        let output = print_doc(&doc);
        // Should be single line for short content
        assert!(output.contains("[") && output.contains("]"));
    }
}
