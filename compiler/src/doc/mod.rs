//! Documentation Generator for Sounio
//!
//! This module provides comprehensive documentation generation capabilities:
//!
//! - **Doc Comments**: Parse `///`, `//!`, and `/** */` style comments
//! - **Markdown**: Full markdown support with syntax highlighting
//! - **HTML Generation**: Responsive, themed HTML output
//! - **mdBook Integration**: Generate guides and tutorials
//! - **Doctests**: Extract and run code examples from documentation
//!
//! # Example
//!
//! ```rust,ignore
//! use sounio::doc::{DocExtractor, HtmlRenderer};
//!
//! let extractor = DocExtractor::new("my_crate", "0.1.0");
//! let crate_doc = extractor.extract(&ast);
//!
//! let renderer = HtmlRenderer::new(crate_doc, output_dir);
//! renderer.generate()?;
//! ```

pub mod book;
pub mod doctest;
pub mod extract;
pub mod html;
pub mod model;
pub mod parser;

pub use book::BookGenerator;
pub use doctest::{DoctestRunner, DoctestSummary};
pub use extract::DocExtractor;
pub use html::HtmlRenderer;
pub use model::{
    ConstantDoc, CrateDoc, DocItem, FunctionDoc, ModuleDoc, SearchEntry, SearchIndex, SearchKind,
    TraitDoc, TypeDoc, Visibility,
};
pub use parser::{DocSections, parse_doc_comment};

/// Documentation attached to an item
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Documentation {
    /// Raw doc comment content (markdown)
    pub content: String,

    /// Parsed sections
    pub sections: DocSections,

    /// Whether this is inner (//!) or outer (///) documentation
    pub is_inner: bool,
}

impl Documentation {
    /// Create new documentation from content
    pub fn new(content: String, is_inner: bool) -> Self {
        let sections = parser::parse_sections(&content);
        Self {
            content,
            sections,
            is_inner,
        }
    }

    /// Create empty documentation
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if documentation is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get the summary (first paragraph)
    pub fn summary(&self) -> Option<&str> {
        self.sections.summary.as_deref()
    }
}
