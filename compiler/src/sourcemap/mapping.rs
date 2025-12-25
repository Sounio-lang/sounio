//! Source map generation for debugging
//!
//! Maps generated code locations back to source locations.

use super::location::Span;
use crate::hlir::ir::{BlockId, ValueId};
use std::collections::HashMap;

/// Maps HLIR values back to source locations
#[derive(Debug, Default)]
pub struct SourceMap {
    /// Value to span mapping
    value_spans: HashMap<ValueId, Span>,

    /// Block to span mapping
    block_spans: HashMap<BlockId, Span>,

    /// Function name to span
    func_spans: HashMap<String, Span>,

    /// Instruction index to span (for generated code)
    instr_spans: HashMap<(String, usize), Span>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a value's source location
    pub fn record_value(&mut self, value: ValueId, span: Span) {
        self.value_spans.insert(value, span);
    }

    /// Record a block's source location
    pub fn record_block(&mut self, block: BlockId, span: Span) {
        self.block_spans.insert(block, span);
    }

    /// Record a function's source location
    pub fn record_function(&mut self, name: String, span: Span) {
        self.func_spans.insert(name, span);
    }

    /// Record an instruction's source location
    pub fn record_instruction(&mut self, func: &str, index: usize, span: Span) {
        self.instr_spans.insert((func.to_string(), index), span);
    }

    /// Get a value's source location
    pub fn get_value_span(&self, value: ValueId) -> Option<Span> {
        self.value_spans.get(&value).copied()
    }

    /// Get a block's source location
    pub fn get_block_span(&self, block: BlockId) -> Option<Span> {
        self.block_spans.get(&block).copied()
    }

    /// Get a function's source location
    pub fn get_function_span(&self, name: &str) -> Option<Span> {
        self.func_spans.get(name).copied()
    }

    /// Get an instruction's source location
    pub fn get_instruction_span(&self, func: &str, index: usize) -> Option<Span> {
        self.instr_spans.get(&(func.to_string(), index)).copied()
    }

    /// Merge another source map into this one
    pub fn merge(&mut self, other: SourceMap) {
        self.value_spans.extend(other.value_spans);
        self.block_spans.extend(other.block_spans);
        self.func_spans.extend(other.func_spans);
        self.instr_spans.extend(other.instr_spans);
    }
}

/// Debug info for generated code
#[derive(Debug, Default)]
pub struct DebugInfo {
    /// Source map
    pub source_map: SourceMap,

    /// Variable names
    pub variable_names: HashMap<ValueId, String>,

    /// Function local variables
    pub locals: HashMap<String, Vec<LocalVariable>>,

    /// Type information for values
    pub value_types: HashMap<ValueId, String>,
}

/// Local variable debug information
#[derive(Debug, Clone)]
pub struct LocalVariable {
    pub name: String,
    pub value: ValueId,
    pub ty: String,
    pub span: Span,
}

impl DebugInfo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a variable name
    pub fn record_variable(&mut self, value: ValueId, name: String) {
        self.variable_names.insert(value, name);
    }

    /// Get a variable's name
    pub fn get_variable_name(&self, value: ValueId) -> Option<&str> {
        self.variable_names.get(&value).map(|s| s.as_str())
    }

    /// Record function locals
    pub fn record_locals(&mut self, func: String, locals: Vec<LocalVariable>) {
        self.locals.insert(func, locals);
    }

    /// Get function locals
    pub fn get_locals(&self, func: &str) -> Option<&[LocalVariable]> {
        self.locals.get(func).map(|v| v.as_slice())
    }

    /// Record value type
    pub fn record_value_type(&mut self, value: ValueId, ty: String) {
        self.value_types.insert(value, ty);
    }

    /// Get value type
    pub fn get_value_type(&self, value: ValueId) -> Option<&str> {
        self.value_types.get(&value).map(|s| s.as_str())
    }
}

/// Builder for constructing debug info during lowering
#[derive(Debug, Default)]
pub struct DebugInfoBuilder {
    info: DebugInfo,
    current_function: Option<String>,
    current_locals: Vec<LocalVariable>,
}

impl DebugInfoBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new function
    pub fn begin_function(&mut self, name: &str, span: Span) {
        // Finish previous function if any
        self.end_function();

        self.current_function = Some(name.to_string());
        self.current_locals.clear();
        self.info.source_map.record_function(name.to_string(), span);
    }

    /// End the current function
    pub fn end_function(&mut self) {
        if let Some(func) = self.current_function.take() {
            let locals = std::mem::take(&mut self.current_locals);
            self.info.record_locals(func, locals);
        }
    }

    /// Record a local variable in the current function
    pub fn record_local(&mut self, name: &str, value: ValueId, ty: &str, span: Span) {
        self.info.record_variable(value, name.to_string());
        self.info.record_value_type(value, ty.to_string());
        self.info.source_map.record_value(value, span);

        self.current_locals.push(LocalVariable {
            name: name.to_string(),
            value,
            ty: ty.to_string(),
            span,
        });
    }

    /// Record a value's source location
    pub fn record_value(&mut self, value: ValueId, span: Span) {
        self.info.source_map.record_value(value, span);
    }

    /// Record a block's source location
    pub fn record_block(&mut self, block: BlockId, span: Span) {
        self.info.source_map.record_block(block, span);
    }

    /// Finish building and return the debug info
    pub fn finish(mut self) -> DebugInfo {
        self.end_function();
        self.info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sourcemap::files::FileId;

    #[test]
    fn test_source_map() {
        let mut map = SourceMap::new();

        let span = Span::new(FileId::new(0), 10, 20);
        map.record_function("main".to_string(), span);
        map.record_value(ValueId(1), span);
        map.record_block(BlockId(0), span);

        assert_eq!(map.get_function_span("main"), Some(span));
        assert_eq!(map.get_value_span(ValueId(1)), Some(span));
        assert_eq!(map.get_block_span(BlockId(0)), Some(span));
    }

    #[test]
    fn test_debug_info_builder() {
        let mut builder = DebugInfoBuilder::new();

        let file = FileId::new(0);
        builder.begin_function("test", Span::new(file, 0, 100));
        builder.record_local("x", ValueId(1), "i32", Span::new(file, 10, 20));
        builder.record_local("y", ValueId(2), "f64", Span::new(file, 30, 40));
        builder.end_function();

        let info = builder.finish();

        assert_eq!(info.get_variable_name(ValueId(1)), Some("x"));
        assert_eq!(info.get_variable_name(ValueId(2)), Some("y"));

        let locals = info.get_locals("test").unwrap();
        assert_eq!(locals.len(), 2);
        assert_eq!(locals[0].name, "x");
        assert_eq!(locals[1].name, "y");
    }
}
