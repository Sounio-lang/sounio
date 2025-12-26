//! Source maps for mapping generated code to source
//!
//! Implements the Source Map v3 specification with D-specific extensions.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Source map for a compilation unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMap {
    /// Version (always 3 for V3 source maps)
    pub version: u8,

    /// Output file name
    pub file: String,

    /// Source root (prepended to source paths)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_root: Option<String>,

    /// List of source files
    pub sources: Vec<String>,

    /// Source content (optional, for embedding)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources_content: Option<Vec<Option<String>>>,

    /// Symbol names
    pub names: Vec<String>,

    /// VLQ encoded mappings
    pub mappings: String,

    /// D-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_sounio: Option<SounioExtensions>,
}

/// D-specific source map extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SounioExtensions {
    /// Effect annotations per function
    pub effects: BTreeMap<String, Vec<String>>,

    /// Type information
    pub types: BTreeMap<String, TypeInfo>,

    /// Inlined function info
    pub inlined: Vec<InlineInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    pub name: String,
    pub kind: String,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineInfo {
    pub callee: String,
    pub caller: String,
    pub line: u32,
    pub column: u32,
}

/// Builder for source maps
pub struct SourceMapBuilder {
    sources: Vec<PathBuf>,
    source_indices: BTreeMap<PathBuf, u32>,
    source_contents: Vec<Option<String>>,
    names: Vec<String>,
    name_indices: BTreeMap<String, u32>,
    mappings: Vec<Mapping>,
    effects: BTreeMap<String, Vec<String>>,
    types: BTreeMap<String, TypeInfo>,
    inlined: Vec<InlineInfo>,
    embed_sources: bool,
}

/// A single mapping entry
#[derive(Debug, Clone)]
pub struct Mapping {
    /// Generated line (0-indexed)
    pub gen_line: u32,

    /// Generated column (0-indexed)
    pub gen_column: u32,

    /// Source file index
    pub source: u32,

    /// Original line (0-indexed)
    pub orig_line: u32,

    /// Original column (0-indexed)
    pub orig_column: u32,

    /// Name index (optional)
    pub name: Option<u32>,

    /// Is this a statement boundary? (for stepping in debuggers)
    pub is_stmt: bool,

    /// Is this the end of function prologue? (first executable statement)
    pub prologue_end: bool,

    /// Is this the beginning of function epilogue?
    pub epilogue_begin: bool,

    /// Is this a basic block start?
    pub basic_block: bool,

    /// Discriminator for same line/column multiple statements
    pub discriminator: u32,
}

/// Statement boundary marker for DWARF line tables
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementKind {
    /// Not a statement (expression result, etc.)
    None,
    /// Regular statement
    Statement,
    /// Block start (opening brace)
    BlockStart,
    /// Block end (closing brace)
    BlockEnd,
    /// Control flow statement (if, while, for, match)
    ControlFlow,
    /// Return statement
    Return,
    /// Function call
    Call,
    /// Assignment
    Assignment,
}

/// Column-accurate source position with enhanced metadata
#[derive(Debug, Clone)]
pub struct EnhancedPosition {
    /// Line number (1-indexed for user display)
    pub line: u32,
    /// Column number (1-indexed for user display)
    pub column: u32,
    /// End column (for range highlighting)
    pub end_column: Option<u32>,
    /// Length in bytes from source
    pub byte_length: Option<u32>,
    /// Statement kind for this position
    pub stmt_kind: StatementKind,
}

impl SourceMapBuilder {
    pub fn new() -> Self {
        SourceMapBuilder {
            sources: Vec::new(),
            source_indices: BTreeMap::new(),
            source_contents: Vec::new(),
            names: Vec::new(),
            name_indices: BTreeMap::new(),
            mappings: Vec::new(),
            effects: BTreeMap::new(),
            types: BTreeMap::new(),
            inlined: Vec::new(),
            embed_sources: false,
        }
    }

    /// Enable source embedding
    pub fn embed_sources(mut self, embed: bool) -> Self {
        self.embed_sources = embed;
        self
    }

    /// Add a source file
    pub fn add_source(&mut self, path: PathBuf) -> u32 {
        if let Some(&idx) = self.source_indices.get(&path) {
            return idx;
        }

        let idx = self.sources.len() as u32;
        self.source_indices.insert(path.clone(), idx);
        self.sources.push(path);
        self.source_contents.push(None);
        idx
    }

    /// Add a source file with content
    pub fn add_source_with_content(&mut self, path: PathBuf, content: String) -> u32 {
        if let Some(&idx) = self.source_indices.get(&path) {
            self.source_contents[idx as usize] = Some(content);
            return idx;
        }

        let idx = self.sources.len() as u32;
        self.source_indices.insert(path.clone(), idx);
        self.sources.push(path);
        self.source_contents.push(Some(content));
        idx
    }

    /// Add a name
    pub fn add_name(&mut self, name: &str) -> u32 {
        if let Some(&idx) = self.name_indices.get(name) {
            return idx;
        }

        let idx = self.names.len() as u32;
        self.name_indices.insert(name.to_string(), idx);
        self.names.push(name.to_string());
        idx
    }

    /// Add a mapping with default flags (is_stmt=true, others=false)
    pub fn add_mapping(
        &mut self,
        gen_line: u32,
        gen_column: u32,
        source: &PathBuf,
        orig_line: u32,
        orig_column: u32,
        name: Option<&str>,
    ) {
        let source_idx = self.add_source(source.clone());
        let name_idx = name.map(|n| self.add_name(n));

        self.mappings.push(Mapping {
            gen_line,
            gen_column,
            source: source_idx,
            orig_line,
            orig_column,
            name: name_idx,
            is_stmt: true,
            prologue_end: false,
            epilogue_begin: false,
            basic_block: false,
            discriminator: 0,
        });
    }

    /// Add a mapping with full control over flags
    pub fn add_mapping_with_flags(
        &mut self,
        gen_line: u32,
        gen_column: u32,
        source: &PathBuf,
        orig_line: u32,
        orig_column: u32,
        name: Option<&str>,
        is_stmt: bool,
        prologue_end: bool,
        epilogue_begin: bool,
    ) {
        let source_idx = self.add_source(source.clone());
        let name_idx = name.map(|n| self.add_name(n));

        self.mappings.push(Mapping {
            gen_line,
            gen_column,
            source: source_idx,
            orig_line,
            orig_column,
            name: name_idx,
            is_stmt,
            prologue_end,
            epilogue_begin,
            basic_block: false,
            discriminator: 0,
        });
    }

    /// Add a prologue end marker (first statement after function entry)
    pub fn add_prologue_end_mapping(
        &mut self,
        gen_line: u32,
        gen_column: u32,
        source: &PathBuf,
        orig_line: u32,
        orig_column: u32,
        name: Option<&str>,
    ) {
        self.add_mapping_with_flags(
            gen_line,
            gen_column,
            source,
            orig_line,
            orig_column,
            name,
            true,  // is_stmt
            true,  // prologue_end
            false, // epilogue_begin
        );
    }

    /// Add an epilogue begin marker (start of function exit)
    pub fn add_epilogue_begin_mapping(
        &mut self,
        gen_line: u32,
        gen_column: u32,
        source: &PathBuf,
        orig_line: u32,
        orig_column: u32,
    ) {
        self.add_mapping_with_flags(
            gen_line,
            gen_column,
            source,
            orig_line,
            orig_column,
            None,
            true,  // is_stmt
            false, // prologue_end
            true,  // epilogue_begin
        );
    }

    /// Add a non-statement mapping (for expressions within statements)
    pub fn add_expression_mapping(
        &mut self,
        gen_line: u32,
        gen_column: u32,
        source: &PathBuf,
        orig_line: u32,
        orig_column: u32,
    ) {
        self.add_mapping_with_flags(
            gen_line,
            gen_column,
            source,
            orig_line,
            orig_column,
            None,
            false, // is_stmt - not a statement boundary
            false,
            false,
        );
    }

    /// Add a mapping with discriminator (for multiple statements on same line)
    pub fn add_discriminated_mapping(
        &mut self,
        gen_line: u32,
        gen_column: u32,
        source: &PathBuf,
        orig_line: u32,
        orig_column: u32,
        discriminator: u32,
    ) {
        let source_idx = self.add_source(source.clone());

        self.mappings.push(Mapping {
            gen_line,
            gen_column,
            source: source_idx,
            orig_line,
            orig_column,
            name: None,
            is_stmt: true,
            prologue_end: false,
            epilogue_begin: false,
            basic_block: false,
            discriminator,
        });
    }

    /// Mark the last mapping as a basic block start
    pub fn mark_basic_block(&mut self) {
        if let Some(mapping) = self.mappings.last_mut() {
            mapping.basic_block = true;
        }
    }

    /// Add a simple mapping (no name)
    pub fn add_simple_mapping(
        &mut self,
        gen_line: u32,
        gen_column: u32,
        source: &PathBuf,
        orig_line: u32,
        orig_column: u32,
    ) {
        self.add_mapping(gen_line, gen_column, source, orig_line, orig_column, None);
    }

    /// Add effect info for a function
    pub fn add_effects(&mut self, function: &str, effects: Vec<String>) {
        self.effects.insert(function.to_string(), effects);
    }

    /// Add type info
    pub fn add_type(&mut self, name: &str, kind: &str, size: usize) {
        self.types.insert(
            name.to_string(),
            TypeInfo {
                name: name.to_string(),
                kind: kind.to_string(),
                size,
            },
        );
    }

    /// Add inline info
    pub fn add_inline(&mut self, callee: &str, caller: &str, line: u32, column: u32) {
        self.inlined.push(InlineInfo {
            callee: callee.to_string(),
            caller: caller.to_string(),
            line,
            column,
        });
    }

    /// Build the source map
    pub fn build(mut self, output_file: &str) -> SourceMap {
        // Sort mappings by generated position
        self.mappings.sort_by(|a, b| {
            a.gen_line
                .cmp(&b.gen_line)
                .then(a.gen_column.cmp(&b.gen_column))
        });

        let mappings = self.encode_mappings();

        let sources_content =
            if self.embed_sources && self.source_contents.iter().any(|c| c.is_some()) {
                Some(self.source_contents)
            } else {
                None
            };

        let x_sounio =
            if !self.effects.is_empty() || !self.types.is_empty() || !self.inlined.is_empty() {
                Some(SounioExtensions {
                    effects: self.effects,
                    types: self.types,
                    inlined: self.inlined,
                })
            } else {
                None
            };

        SourceMap {
            version: 3,
            file: output_file.to_string(),
            source_root: None,
            sources: self
                .sources
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
            sources_content,
            names: self.names,
            mappings,
            x_sounio,
        }
    }

    /// Encode mappings as VLQ
    fn encode_mappings(&self) -> String {
        let mut result = String::new();
        let mut prev_gen_col = 0i64;
        let mut prev_source = 0i64;
        let mut prev_orig_line = 0i64;
        let mut prev_orig_col = 0i64;
        let mut prev_name = 0i64;
        let mut prev_gen_line = 0u32;

        for mapping in &self.mappings {
            // Add semicolons for new lines
            while prev_gen_line < mapping.gen_line {
                result.push(';');
                prev_gen_line += 1;
                prev_gen_col = 0;
            }

            // Add comma separator within same line
            if !result.is_empty() && !result.ends_with(';') {
                result.push(',');
            }

            // Encode segment
            let gen_col_delta = mapping.gen_column as i64 - prev_gen_col;
            result.push_str(&vlq_encode(gen_col_delta));

            let source_delta = mapping.source as i64 - prev_source;
            result.push_str(&vlq_encode(source_delta));

            let orig_line_delta = mapping.orig_line as i64 - prev_orig_line;
            result.push_str(&vlq_encode(orig_line_delta));

            let orig_col_delta = mapping.orig_column as i64 - prev_orig_col;
            result.push_str(&vlq_encode(orig_col_delta));

            if let Some(name_idx) = mapping.name {
                let name_delta = name_idx as i64 - prev_name;
                result.push_str(&vlq_encode(name_delta));
                prev_name = name_idx as i64;
            }

            prev_gen_col = mapping.gen_column as i64;
            prev_source = mapping.source as i64;
            prev_orig_line = mapping.orig_line as i64;
            prev_orig_col = mapping.orig_column as i64;
        }

        result
    }
}

impl Default for SourceMapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// VLQ encode a signed integer
fn vlq_encode(value: i64) -> String {
    const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::new();

    // Convert to unsigned with sign in LSB
    let mut unsigned: u64 = if value < 0 {
        (((-value) << 1) | 1) as u64
    } else {
        (value << 1) as u64
    };

    loop {
        let mut digit = (unsigned & 0x1F) as u8;
        unsigned >>= 5;

        if unsigned > 0 {
            digit |= 0x20; // Continuation bit
        }

        result.push(BASE64_CHARS[digit as usize] as char);

        if unsigned == 0 {
            break;
        }
    }

    result
}

/// VLQ decode a signed integer
fn vlq_decode(chars: &mut std::str::Chars) -> Option<i64> {
    const BASE64_VALUES: [i8; 128] = {
        let mut table = [-1i8; 128];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < chars.len() {
            table[chars[i] as usize] = i as i8;
            i += 1;
        }
        table
    };

    let mut result: u64 = 0;
    let mut shift = 0;

    loop {
        let c = chars.next()?;
        if c as usize >= 128 {
            return None;
        }

        let digit = BASE64_VALUES[c as usize];
        if digit < 0 {
            return None;
        }

        let digit = digit as u64;
        result |= (digit & 0x1F) << shift;
        shift += 5;

        if (digit & 0x20) == 0 {
            break;
        }
    }

    // Convert from unsigned with sign in LSB to signed
    let is_negative = (result & 1) != 0;
    let value = (result >> 1) as i64;

    Some(if is_negative { -value } else { value })
}

impl SourceMap {
    /// Parse from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Convert to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Convert to compact JSON
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Look up original location for generated position
    pub fn lookup(&self, gen_line: u32, gen_column: u32) -> Option<OriginalLocation> {
        let decoded = self.decode_mappings();

        // Binary search for the mapping
        let mut best_match: Option<&DecodedMapping> = None;

        for mapping in &decoded {
            if mapping.gen_line == gen_line {
                if mapping.gen_column <= gen_column {
                    match best_match {
                        None => best_match = Some(mapping),
                        Some(prev) if mapping.gen_column > prev.gen_column => {
                            best_match = Some(mapping)
                        }
                        _ => {}
                    }
                }
            } else if mapping.gen_line > gen_line {
                break;
            }
        }

        best_match.map(|m| OriginalLocation {
            source: self
                .sources
                .get(m.source as usize)
                .cloned()
                .unwrap_or_default(),
            line: m.orig_line,
            column: m.orig_column,
            name: m.name.and_then(|idx| self.names.get(idx as usize).cloned()),
        })
    }

    /// Decode all mappings
    fn decode_mappings(&self) -> Vec<DecodedMapping> {
        let mut result = Vec::new();
        let mut gen_line = 0u32;
        let mut gen_col: i64;
        let mut source = 0i64;
        let mut orig_line = 0i64;
        let mut orig_col = 0i64;
        let mut name = 0i64;

        for segment in self.mappings.split(';') {
            gen_col = 0;

            for mapping in segment.split(',') {
                if mapping.is_empty() {
                    continue;
                }

                let mut chars = mapping.chars();

                // Generated column
                if let Some(delta) = vlq_decode(&mut chars) {
                    gen_col += delta;
                } else {
                    continue;
                }

                // Check if there are more fields
                let mut decoded = DecodedMapping {
                    gen_line,
                    gen_column: gen_col as u32,
                    source: 0,
                    orig_line: 0,
                    orig_column: 0,
                    name: None,
                };

                // Source index
                if let Some(delta) = vlq_decode(&mut chars) {
                    source += delta;
                    decoded.source = source as u32;
                } else {
                    result.push(decoded);
                    continue;
                }

                // Original line
                if let Some(delta) = vlq_decode(&mut chars) {
                    orig_line += delta;
                    decoded.orig_line = orig_line as u32;
                }

                // Original column
                if let Some(delta) = vlq_decode(&mut chars) {
                    orig_col += delta;
                    decoded.orig_column = orig_col as u32;
                }

                // Name index (optional)
                if let Some(delta) = vlq_decode(&mut chars) {
                    name += delta;
                    decoded.name = Some(name as u32);
                }

                result.push(decoded);
            }

            gen_line += 1;
        }

        result
    }

    /// Get all sources
    pub fn sources(&self) -> &[String] {
        &self.sources
    }

    /// Get all names
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Get D-specific extensions
    pub fn extensions(&self) -> Option<&SounioExtensions> {
        self.x_sounio.as_ref()
    }
}

/// Decoded mapping entry
#[derive(Debug, Clone)]
struct DecodedMapping {
    gen_line: u32,
    gen_column: u32,
    source: u32,
    orig_line: u32,
    orig_column: u32,
    name: Option<u32>,
}

/// Original source location
#[derive(Debug, Clone)]
pub struct OriginalLocation {
    pub source: String,
    pub line: u32,
    pub column: u32,
    pub name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vlq_encode_decode() {
        let test_cases = [0i64, 1, -1, 16, -16, 127, -127, 1000, -1000];

        for value in test_cases {
            let encoded = vlq_encode(value);
            let mut chars = encoded.chars();
            let decoded = vlq_decode(&mut chars).unwrap();
            assert_eq!(value, decoded, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_source_map_builder() {
        let mut builder = SourceMapBuilder::new();

        let source = PathBuf::from("test.sio");
        builder.add_mapping(0, 0, &source, 0, 0, Some("main"));
        builder.add_mapping(0, 10, &source, 1, 5, None);
        builder.add_mapping(1, 0, &source, 2, 0, Some("foo"));

        builder.add_effects("main", vec!["IO".to_string(), "Alloc".to_string()]);
        builder.add_type("MyStruct", "struct", 16);

        let source_map = builder.build("output.js");

        assert_eq!(source_map.version, 3);
        assert_eq!(source_map.sources.len(), 1);
        assert_eq!(source_map.names.len(), 2);
        assert!(!source_map.mappings.is_empty());
        assert!(source_map.x_sounio.is_some());
    }

    #[test]
    fn test_source_map_lookup() {
        let mut builder = SourceMapBuilder::new();

        let source = PathBuf::from("test.sio");
        builder.add_mapping(0, 0, &source, 10, 5, Some("start"));
        builder.add_mapping(0, 20, &source, 10, 25, None);
        builder.add_mapping(1, 0, &source, 15, 0, Some("middle"));

        let source_map = builder.build("output.js");

        // Lookup at exact position
        let loc = source_map.lookup(0, 0).unwrap();
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
        assert_eq!(loc.name, Some("start".to_string()));

        // Lookup between mappings
        let loc = source_map.lookup(0, 15).unwrap();
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);

        // Lookup on second line
        let loc = source_map.lookup(1, 0).unwrap();
        assert_eq!(loc.line, 15);
        assert_eq!(loc.column, 0);
    }

    #[test]
    fn test_source_map_json() {
        let mut builder = SourceMapBuilder::new();
        let source = PathBuf::from("test.sio");
        builder.add_mapping(0, 0, &source, 0, 0, None);

        let source_map = builder.build("output.js");
        let json = source_map.to_json().unwrap();

        let parsed = SourceMap::from_json(&json).unwrap();
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.file, "output.js");
    }
}
