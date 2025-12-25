use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use super::edits::TextEdit;
use crate::Span;
use crate::ast::{Ast, Item};
use crate::parser::{ParseError, Parser};

/// Node identifier for incremental updates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        NodeId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// A syntax node with identity for incremental updates
#[derive(Debug, Clone)]
pub struct IncrementalNode<T> {
    /// Unique node identifier
    pub id: NodeId,

    /// The actual AST node
    pub node: T,

    /// Byte range in source
    pub range: std::ops::Range<usize>,

    /// Hash of the source text for this node
    pub source_hash: u64,
}

impl<T> IncrementalNode<T> {
    pub fn new(node: T, range: std::ops::Range<usize>, source: &str) -> Self {
        IncrementalNode {
            id: NodeId::new(),
            node,
            range: range.clone(),
            source_hash: hash_range(source, &range),
        }
    }

    /// Check if this node is affected by an edit
    pub fn is_affected(&self, edit: &TextEdit) -> bool {
        // Node is affected if edit range overlaps with node range
        self.range.start < edit.range.end && edit.range.start < self.range.end
    }

    /// Check if node can be reused (source unchanged)
    pub fn can_reuse(&self, new_source: &str, offset_delta: isize) -> bool {
        let new_start = (self.range.start as isize + offset_delta) as usize;
        let new_end = (self.range.end as isize + offset_delta) as usize;

        if new_end > new_source.len() {
            return false;
        }

        let new_hash = hash_range(new_source, &(new_start..new_end));
        new_hash == self.source_hash
    }
}

/// Incremental AST with change tracking
#[derive(Debug, Clone)]
pub struct IncrementalAst {
    /// Root items
    pub items: Vec<IncrementalNode<Item>>,

    /// Source text
    pub source: String,

    /// Version counter
    pub version: u64,
}

impl IncrementalAst {
    /// Create from full parse
    pub fn from_ast(ast: Ast, source: &str) -> Self {
        let items = ast
            .items
            .into_iter()
            .map(|item| {
                let range = item.span.start..item.span.end;
                IncrementalNode::new(item, range, source)
            })
            .collect();

        IncrementalAst {
            items,
            source: source.to_string(),
            version: 1,
        }
    }

    /// Apply an edit incrementally
    pub fn apply_edit(&mut self, edit: &TextEdit, new_source: &str) -> IncrementalParseResult {
        let mut result = IncrementalParseResult::new();

        // Calculate offset delta for items after the edit
        let delta = edit.length_delta();

        // Collect items to keep (unaffected by edit)
        let mut new_items: Vec<IncrementalNode<Item>> = Vec::new();

        for item in &self.items {
            if !item.is_affected(edit) {
                let mut new_item = item.clone();

                // Adjust range for items after the edit
                if item.range.start >= edit.range.end {
                    new_item.range.start = (new_item.range.start as isize + delta) as usize;
                    new_item.range.end = (new_item.range.end as isize + delta) as usize;

                    // Also adjust the span in the node
                    adjust_item_span(&mut new_item.node, delta);
                }

                new_items.push(new_item);
                result.reused_nodes += 1;
            }
        }

        // Re-parse affected region
        let (reparse_start, reparse_end) = self.find_reparse_region(edit, new_source);

        if reparse_start < reparse_end {
            let region_source = &new_source[reparse_start..reparse_end];

            // Parse the affected region
            let mut parser = Parser::new(region_source);
            parser.set_offset(reparse_start);

            match parser.parse_items() {
                Ok(parsed_items) => {
                    for item in parsed_items {
                        let range = item.span.start..item.span.end;
                        let node = IncrementalNode::new(item, range, new_source);

                        // Insert at correct position
                        let insert_pos = new_items
                            .iter()
                            .position(|n| n.range.start > node.range.start)
                            .unwrap_or(new_items.len());

                        new_items.insert(insert_pos, node);
                        result.reparsed_nodes += 1;
                    }
                }
                Err(e) => {
                    result.errors.push(e);
                }
            }
        }

        self.items = new_items;
        self.source = new_source.to_string();
        self.version += 1;

        result
    }

    /// Find the region that needs re-parsing
    fn find_reparse_region(&self, edit: &TextEdit, new_source: &str) -> (usize, usize) {
        // Find item boundaries around the edit
        let mut start = edit.range.start;
        let mut end = edit.range.start + edit.new_text.len();

        // Expand to item boundaries
        for item in &self.items {
            if item.range.start <= start && start < item.range.end {
                start = item.range.start;
            }
            if item.range.start < end && end <= item.range.end {
                let adjusted_end = (item.range.end as isize + edit.length_delta()) as usize;
                end = adjusted_end.min(new_source.len());
            }
        }

        // Expand to line boundaries for safety
        start = new_source[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        end = new_source[end..]
            .find('\n')
            .map(|i| end + i + 1)
            .unwrap_or(new_source.len());

        (start, end.min(new_source.len()))
    }

    /// Convert back to regular AST
    pub fn to_ast(&self) -> Ast {
        Ast {
            items: self.items.iter().map(|n| n.node.clone()).collect(),
        }
    }
}

/// Result of incremental parse
#[derive(Debug, Default)]
pub struct IncrementalParseResult {
    /// Number of nodes reused from previous parse
    pub reused_nodes: usize,

    /// Number of nodes that were re-parsed
    pub reparsed_nodes: usize,

    /// Parse errors encountered
    pub errors: Vec<ParseError>,
}

impl IncrementalParseResult {
    pub fn new() -> Self {
        Self::default()
    }

    /// Percentage of nodes that were reused
    pub fn reuse_percentage(&self) -> f64 {
        let total = self.reused_nodes + self.reparsed_nodes;
        if total == 0 {
            0.0
        } else {
            (self.reused_nodes as f64 / total as f64) * 100.0
        }
    }
}

fn hash_range(source: &str, range: &std::ops::Range<usize>) -> u64 {
    let mut hasher = DefaultHasher::new();
    source[range.clone()].hash(&mut hasher);
    hasher.finish()
}

fn adjust_item_span(item: &mut Item, delta: isize) {
    item.span.start = (item.span.start as isize + delta) as usize;
    item.span.end = (item.span.end as isize + delta) as usize;

    // Recursively adjust nested spans
    // This would need to traverse the entire item tree
}
