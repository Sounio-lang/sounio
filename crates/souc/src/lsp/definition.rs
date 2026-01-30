//! Go to definition provider
//!
//! Provides navigation to symbol definitions.

use tower_lsp::lsp_types::*;

use crate::common::Span;
use crate::resolve::SymbolTable;

/// Provider for go-to-definition
pub struct DefinitionProvider;

impl DefinitionProvider {
    /// Create a new definition provider
    pub fn new() -> Self {
        Self
    }

    /// Find the definition of a symbol
    pub fn find_definition(
        &self,
        name: &str,
        symbols: &SymbolTable,
        current_uri: &Url,
    ) -> Option<GotoDefinitionResponse> {
        // Look up in value namespace
        if let Some(def_id) = symbols.lookup(name) {
            if let Some(symbol) = symbols.get(def_id) {
                let location = self.span_to_location(&symbol.span, current_uri);
                return Some(GotoDefinitionResponse::Scalar(location));
            }
        }

        // Look up in type namespace
        if let Some(def_id) = symbols.lookup_type(name) {
            if let Some(symbol) = symbols.get(def_id) {
                let location = self.span_to_location(&symbol.span, current_uri);
                return Some(GotoDefinitionResponse::Scalar(location));
            }
        }

        None
    }

    /// Convert a span to an LSP location
    fn span_to_location(&self, span: &Span, uri: &Url) -> Location {
        // For now, we don't have line/column info in spans
        // We'll use byte offsets converted to position 0:0
        // In a real implementation, we'd track source locations properly
        Location {
            uri: uri.clone(),
            range: Range {
                start: Position {
                    line: 0,
                    character: span.start as u32,
                },
                end: Position {
                    line: 0,
                    character: span.end as u32,
                },
            },
        }
    }
}

impl Default for DefinitionProvider {
    fn default() -> Self {
        Self::new()
    }
}
