//! GPU Source Mapping
//!
//! Tracks source locations through the GPU lowering and codegen pipeline:
//! HLIR → GpuIR → PTX
//!
//! This enables error messages and debugging to trace back from PTX to the
//! original source code.

use std::collections::HashMap;
use std::fmt;

use super::ir::{BlockId, ValueId};
use crate::common::Span;

// ============================================================================
// Location Types
// ============================================================================

/// Location in GPU IR (kernel + block + instruction)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuIrLocation {
    /// Kernel name
    pub kernel: String,
    /// Block ID
    pub block: BlockId,
    /// Instruction index within block
    pub instruction: usize,
    /// Value ID (result of instruction)
    pub value: ValueId,
}

impl GpuIrLocation {
    /// Create a new GPU IR location
    pub fn new(
        kernel: impl Into<String>,
        block: BlockId,
        instruction: usize,
        value: ValueId,
    ) -> Self {
        Self {
            kernel: kernel.into(),
            block,
            instruction,
            value,
        }
    }
}

impl fmt::Display for GpuIrLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}::{}[{}] (v{})",
            self.kernel, self.block, self.instruction, self.value.0
        )
    }
}

/// Location in PTX output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtxLocation {
    /// Line number (1-based)
    pub line: u32,
    /// Column number (1-based, optional)
    pub column: u32,
}

impl PtxLocation {
    /// Create a new PTX location
    pub fn new(line: u32, column: u32) -> Self {
        Self { line, column }
    }

    /// Create a PTX location with just line number
    pub fn line(line: u32) -> Self {
        Self { line, column: 0 }
    }
}

/// Full location trace through all IR levels
#[derive(Debug, Clone)]
pub struct LocationTrace {
    /// HLIR source span
    pub hlir_span: Option<Span>,
    /// GPU IR location
    pub gpu_ir: Option<GpuIrLocation>,
    /// PTX location
    pub ptx: Option<PtxLocation>,
}

impl LocationTrace {
    /// Create an empty trace
    pub fn empty() -> Self {
        Self {
            hlir_span: None,
            gpu_ir: None,
            ptx: None,
        }
    }

    /// Check if the trace has any location information
    pub fn has_location(&self) -> bool {
        self.hlir_span.is_some() || self.gpu_ir.is_some() || self.ptx.is_some()
    }
}

// ============================================================================
// Source Mapper
// ============================================================================

/// Maps locations through the GPU lowering pipeline: HLIR → GpuIR → PTX
///
/// This allows tracing errors back to their source locations.
#[derive(Debug, Clone, Default)]
pub struct GpuSourceMapper {
    /// HLIR value ID to HLIR span mapping (from AST)
    hlir_spans: HashMap<u32, Span>,

    /// HLIR value ID to GPU IR location mapping (from lowering)
    hlir_to_gpu: HashMap<u32, GpuIrLocation>,

    /// GPU IR location to PTX location mapping (from codegen)
    gpu_to_ptx: HashMap<GpuIrLocation, PtxLocation>,

    /// Reverse mapping: PTX line to GPU IR location
    ptx_to_gpu: HashMap<u32, GpuIrLocation>,

    /// Reverse mapping: GPU IR location to HLIR value ID
    gpu_to_hlir: HashMap<GpuIrLocation, u32>,
}

impl GpuSourceMapper {
    /// Create a new empty source mapper
    pub fn new() -> Self {
        Self::default()
    }

    // ========================================================================
    // Recording Mappings
    // ========================================================================

    /// Record an HLIR value's source span
    pub fn record_hlir_span(&mut self, hlir_value_id: u32, span: Span) {
        self.hlir_spans.insert(hlir_value_id, span);
    }

    /// Record the lowering from HLIR to GPU IR
    pub fn record_lowering(&mut self, hlir_value_id: u32, gpu_location: GpuIrLocation) {
        self.gpu_to_hlir.insert(gpu_location.clone(), hlir_value_id);
        self.hlir_to_gpu.insert(hlir_value_id, gpu_location);
    }

    /// Record the codegen from GPU IR to PTX
    pub fn record_codegen(&mut self, gpu_location: GpuIrLocation, ptx_location: PtxLocation) {
        self.ptx_to_gpu
            .insert(ptx_location.line, gpu_location.clone());
        self.gpu_to_ptx.insert(gpu_location, ptx_location);
    }

    // ========================================================================
    // Querying Mappings
    // ========================================================================

    /// Get the HLIR span for an HLIR value
    pub fn get_hlir_span(&self, hlir_value_id: u32) -> Option<Span> {
        self.hlir_spans.get(&hlir_value_id).copied()
    }

    /// Get the GPU IR location for an HLIR value
    pub fn get_gpu_location(&self, hlir_value_id: u32) -> Option<&GpuIrLocation> {
        self.hlir_to_gpu.get(&hlir_value_id)
    }

    /// Get the PTX location for a GPU IR location
    pub fn get_ptx_location(&self, gpu_location: &GpuIrLocation) -> Option<PtxLocation> {
        self.gpu_to_ptx.get(gpu_location).copied()
    }

    /// Get the GPU IR location for a PTX line
    pub fn get_gpu_from_ptx(&self, ptx_line: u32) -> Option<&GpuIrLocation> {
        self.ptx_to_gpu.get(&ptx_line)
    }

    /// Get the HLIR value ID for a GPU IR location
    pub fn get_hlir_from_gpu(&self, gpu_location: &GpuIrLocation) -> Option<u32> {
        self.gpu_to_hlir.get(gpu_location).copied()
    }

    // ========================================================================
    // Tracing
    // ========================================================================

    /// Trace from PTX line back to HLIR span
    pub fn trace_ptx_to_hlir(&self, ptx_line: u32) -> Option<Span> {
        let gpu_loc = self.ptx_to_gpu.get(&ptx_line)?;
        let hlir_id = self.gpu_to_hlir.get(gpu_loc)?;
        self.hlir_spans.get(hlir_id).copied()
    }

    /// Trace from GPU IR location back to HLIR span
    pub fn trace_gpu_to_hlir(&self, gpu_location: &GpuIrLocation) -> Option<Span> {
        let hlir_id = self.gpu_to_hlir.get(gpu_location)?;
        self.hlir_spans.get(hlir_id).copied()
    }

    /// Get full location trace for a PTX line
    pub fn full_trace(&self, ptx_line: u32) -> LocationTrace {
        let ptx = Some(PtxLocation::line(ptx_line));
        let gpu_ir = self.ptx_to_gpu.get(&ptx_line).cloned();
        let hlir_span = gpu_ir
            .as_ref()
            .and_then(|gpu| self.gpu_to_hlir.get(gpu))
            .and_then(|hlir_id| self.hlir_spans.get(hlir_id))
            .copied();

        LocationTrace {
            hlir_span,
            gpu_ir,
            ptx,
        }
    }

    /// Get full location trace for a GPU IR location
    pub fn full_trace_from_gpu(&self, gpu_location: &GpuIrLocation) -> LocationTrace {
        let ptx = self.gpu_to_ptx.get(gpu_location).copied();
        let hlir_span = self
            .gpu_to_hlir
            .get(gpu_location)
            .and_then(|hlir_id| self.hlir_spans.get(hlir_id))
            .copied();

        LocationTrace {
            hlir_span,
            gpu_ir: Some(gpu_location.clone()),
            ptx,
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get the number of HLIR to GPU mappings
    pub fn hlir_mapping_count(&self) -> usize {
        self.hlir_to_gpu.len()
    }

    /// Get the number of GPU to PTX mappings
    pub fn ptx_mapping_count(&self) -> usize {
        self.gpu_to_ptx.len()
    }

    /// Check if the mapper is empty
    pub fn is_empty(&self) -> bool {
        self.hlir_to_gpu.is_empty() && self.gpu_to_ptx.is_empty()
    }
}

// ============================================================================
// Span Tracker (for use during lowering)
// ============================================================================

/// Helper for tracking source spans during lowering passes
///
/// Maintains a stack of spans for nested operations.
#[derive(Debug, Default)]
pub struct SpanTracker {
    /// Stack of current spans (for nested operations)
    span_stack: Vec<Span>,
    /// Current kernel name
    current_kernel: Option<String>,
    /// Current block ID
    current_block: Option<BlockId>,
    /// Instruction counter within current block
    instruction_counter: usize,
}

impl SpanTracker {
    /// Create a new span tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the current kernel
    pub fn set_kernel(&mut self, name: impl Into<String>) {
        self.current_kernel = Some(name.into());
        self.current_block = None;
        self.instruction_counter = 0;
    }

    /// Set the current block
    pub fn set_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
        self.instruction_counter = 0;
    }

    /// Increment instruction counter and return current index
    pub fn next_instruction(&mut self) -> usize {
        let idx = self.instruction_counter;
        self.instruction_counter += 1;
        idx
    }

    /// Push a span onto the tracking stack
    pub fn push_span(&mut self, span: Span) {
        self.span_stack.push(span);
    }

    /// Pop and return the current span
    pub fn pop_span(&mut self) -> Option<Span> {
        self.span_stack.pop()
    }

    /// Get the current span (top of stack)
    pub fn current_span(&self) -> Option<Span> {
        self.span_stack.last().copied()
    }

    /// Build a GPU IR location for the current position
    pub fn current_location(&self, value: ValueId) -> Option<GpuIrLocation> {
        let kernel = self.current_kernel.clone()?;
        let block = self.current_block?;
        Some(GpuIrLocation {
            kernel,
            block,
            instruction: self.instruction_counter.saturating_sub(1),
            value,
        })
    }

    /// Record a mapping to the source mapper
    pub fn record_to_mapper(
        &self,
        mapper: &mut GpuSourceMapper,
        hlir_value_id: u32,
        gpu_value: ValueId,
    ) {
        // Record HLIR span if we have one
        if let Some(span) = self.current_span() {
            mapper.record_hlir_span(hlir_value_id, span);
        }

        // Record lowering if we have a location
        if let Some(loc) = self.current_location(gpu_value) {
            mapper.record_lowering(hlir_value_id, loc);
        }
    }
}

// ============================================================================
// PTX Debug Info
// ============================================================================

/// Generates PTX debug directives for source-level debugging
pub struct PtxDebugEmitter {
    /// Whether debug emission is enabled
    enabled: bool,
    /// File table: file ID -> path
    files: Vec<String>,
    /// File path to ID mapping
    file_ids: HashMap<String, u32>,
    /// Current file ID
    current_file: u32,
}

impl PtxDebugEmitter {
    /// Create a new debug emitter
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            files: Vec::new(),
            file_ids: HashMap::new(),
            current_file: 0,
        }
    }

    /// Register a source file and get its ID
    pub fn register_file(&mut self, path: impl Into<String>) -> u32 {
        let path = path.into();
        if let Some(&id) = self.file_ids.get(&path) {
            return id;
        }

        let id = self.files.len() as u32;
        self.file_ids.insert(path.clone(), id);
        self.files.push(path);
        id
    }

    /// Set the current file
    pub fn set_file(&mut self, file_id: u32) {
        self.current_file = file_id;
    }

    /// Emit a .file directive for all registered files
    pub fn emit_file_directives(&self) -> String {
        if !self.enabled || self.files.is_empty() {
            return String::new();
        }

        let mut output = String::new();
        for (id, path) in self.files.iter().enumerate() {
            output.push_str(&format!(".file {} \"{}\"\n", id + 1, path));
        }
        output
    }

    /// Emit a .loc directive for a source location
    pub fn emit_loc(&self, line: u32, column: u32) -> Option<String> {
        if !self.enabled {
            return None;
        }

        Some(format!(
            ".loc {} {} {}",
            self.current_file + 1,
            line,
            column
        ))
    }

    /// Check if debug emission is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for PtxDebugEmitter {
    fn default() -> Self {
        Self::new(false)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_ir_location() {
        let loc = GpuIrLocation::new("my_kernel", BlockId(0), 5, ValueId(10));
        assert_eq!(loc.kernel, "my_kernel");
        assert_eq!(loc.block, BlockId(0));
        assert_eq!(loc.instruction, 5);
        assert_eq!(loc.value, ValueId(10));
    }

    #[test]
    fn test_source_mapper_basic() {
        let mut mapper = GpuSourceMapper::new();

        // Record HLIR span
        let span = Span::new(10, 20);
        mapper.record_hlir_span(1, span);

        // Record lowering
        let gpu_loc = GpuIrLocation::new("test_kernel", BlockId(0), 0, ValueId(100));
        mapper.record_lowering(1, gpu_loc.clone());

        // Record codegen
        let ptx_loc = PtxLocation::line(42);
        mapper.record_codegen(gpu_loc.clone(), ptx_loc);

        // Query
        assert_eq!(mapper.get_hlir_span(1), Some(span));
        assert_eq!(mapper.get_gpu_location(1), Some(&gpu_loc));
        assert_eq!(mapper.get_ptx_location(&gpu_loc), Some(ptx_loc));
    }

    #[test]
    fn test_trace_ptx_to_hlir() {
        let mut mapper = GpuSourceMapper::new();

        let span = Span::new(100, 200);
        let gpu_loc = GpuIrLocation::new("kernel", BlockId(1), 5, ValueId(50));

        mapper.record_hlir_span(42, span);
        mapper.record_lowering(42, gpu_loc.clone());
        mapper.record_codegen(gpu_loc, PtxLocation::line(99));

        // Trace back
        let result = mapper.trace_ptx_to_hlir(99);
        assert_eq!(result, Some(span));
    }

    #[test]
    fn test_full_trace() {
        let mut mapper = GpuSourceMapper::new();

        let span = Span::new(10, 50);
        let gpu_loc = GpuIrLocation::new("test", BlockId(0), 0, ValueId(1));

        mapper.record_hlir_span(1, span);
        mapper.record_lowering(1, gpu_loc.clone());
        mapper.record_codegen(gpu_loc, PtxLocation::new(10, 5));

        let trace = mapper.full_trace(10);
        assert!(trace.has_location());
        assert_eq!(trace.hlir_span, Some(span));
        assert!(trace.gpu_ir.is_some());
        assert!(trace.ptx.is_some());
    }

    #[test]
    fn test_span_tracker() {
        let mut tracker = SpanTracker::new();
        let mut mapper = GpuSourceMapper::new();

        tracker.set_kernel("test_kernel");
        tracker.set_block(BlockId(0));
        tracker.push_span(Span::new(10, 20));

        let idx = tracker.next_instruction();
        assert_eq!(idx, 0);

        tracker.record_to_mapper(&mut mapper, 1, ValueId(100));

        assert_eq!(mapper.get_hlir_span(1), Some(Span::new(10, 20)));
    }

    #[test]
    fn test_ptx_debug_emitter() {
        let mut emitter = PtxDebugEmitter::new(true);

        let id1 = emitter.register_file("src/main.siom");
        let id2 = emitter.register_file("src/utils.dm");
        let id1_again = emitter.register_file("src/main.siom");

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id1_again, 0); // Same file reuses ID

        let directives = emitter.emit_file_directives();
        assert!(directives.contains(".file 1 \"src/main.siom\""));
        assert!(directives.contains(".file 2 \"src/utils.dm\""));
    }

    #[test]
    fn test_location_trace_empty() {
        let trace = LocationTrace::empty();
        assert!(!trace.has_location());
    }
}
