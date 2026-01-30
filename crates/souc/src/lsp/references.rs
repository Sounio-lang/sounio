//! Find references provider
//!
//! Provides functionality to find all references to a symbol.

use tower_lsp::lsp_types::*;

use crate::common::Span;
use crate::resolve::SymbolTable;

/// Provider for find-references
pub struct ReferencesProvider;

impl ReferencesProvider {
    /// Create a new references provider
    pub fn new() -> Self {
        Self
    }

    /// Find all references to a symbol
    pub fn find_references(
        &self,
        name: &str,
        symbols: &SymbolTable,
        current_uri: &Url,
    ) -> Vec<Location> {
        let mut locations = Vec::new();

        // Look up the definition first
        let def_id = symbols.lookup(name).or_else(|| symbols.lookup_type(name));

        if let Some(def_id) = def_id {
            if let Some(symbol) = symbols.get(def_id) {
                // Add the definition itself
                locations.push(self.span_to_location(&symbol.span, current_uri));
            }

            // In a full implementation, we would:
            // 1. Walk the AST to find all references
            // 2. Check the node_to_ref mapping in the symbol table
            // 3. Return all locations where this symbol is referenced

            // For now, we just return the definition location
        }

        locations
    }

    /// Convert a span to an LSP location
    fn span_to_location(&self, span: &Span, uri: &Url) -> Location {
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

impl Default for ReferencesProvider {
    fn default() -> Self {
        Self::new()
    }
}
