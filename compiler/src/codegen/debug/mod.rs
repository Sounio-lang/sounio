//! Debug information generation for DWARF format
//!
//! This module provides infrastructure for generating debug information
//! that enables source-level debugging with GDB, LLDB, and other debuggers.

use std::collections::HashMap;
use std::path::PathBuf;

pub mod source_map;

pub use source_map::{SourceMap, SourceMapBuilder};

/// Debug information builder
pub struct DebugInfoBuilder {
    /// Compilation unit info
    compile_unit: CompileUnit,

    /// Type definitions
    types: HashMap<TypeId, DIType>,

    /// Subprograms (functions)
    subprograms: Vec<DISubprogram>,

    /// Global variables
    globals: Vec<DIGlobalVariable>,

    /// Current scope stack
    scope_stack: Vec<DIScope>,

    /// Source file info
    files: HashMap<PathBuf, DIFile>,

    /// Line table entries
    line_table: Vec<LineTableEntry>,
}

/// Compilation unit debug info
#[derive(Debug, Clone)]
pub struct CompileUnit {
    /// Source file
    pub file: PathBuf,

    /// Compilation directory
    pub directory: PathBuf,

    /// Producer string (compiler version)
    pub producer: String,

    /// Language ID (custom for D)
    pub language: u16,

    /// Optimization level
    pub optimization_level: OptLevel,

    /// Debug info version
    pub dwarf_version: u8,
}

/// Debug info for a type
#[derive(Debug, Clone)]
pub enum DIType {
    /// Basic type (int, float, etc.)
    Basic(DIBasicType),

    /// Pointer type
    Pointer(DIPointerType),

    /// Reference type
    Reference(DIReferenceType),

    /// Array type
    Array(DIArrayType),

    /// Struct type
    Struct(DIStructType),

    /// Enum type
    Enum(DIEnumType),

    /// Function type
    Function(DIFunctionType),

    /// Typedef/alias
    Typedef(DITypedef),
}

#[derive(Debug, Clone)]
pub struct DIBasicType {
    pub name: String,
    pub size_bits: u64,
    pub encoding: DWEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DWEncoding {
    Boolean,
    SignedInt,
    UnsignedInt,
    Float,
    SignedChar,
    UnsignedChar,
    Address,
}

#[derive(Debug, Clone)]
pub struct DIPointerType {
    pub pointee: TypeId,
    pub size_bits: u64,
}

#[derive(Debug, Clone)]
pub struct DIReferenceType {
    pub pointee: TypeId,
    pub is_exclusive: bool, // &! vs &
}

#[derive(Debug, Clone)]
pub struct DIArrayType {
    pub element_type: TypeId,
    pub count: Option<u64>, // None for unsized
}

#[derive(Debug, Clone)]
pub struct DIStructType {
    pub name: String,
    pub size_bits: u64,
    pub align_bits: u64,
    pub members: Vec<DIMember>,
    pub file: PathBuf,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct DIMember {
    pub name: String,
    pub ty: TypeId,
    pub offset_bits: u64,
    pub size_bits: u64,
}

#[derive(Debug, Clone)]
pub struct DIEnumType {
    pub name: String,
    pub underlying_type: TypeId,
    pub variants: Vec<DIEnumVariant>,
}

#[derive(Debug, Clone)]
pub struct DIEnumVariant {
    pub name: String,
    pub value: i64,
}

#[derive(Debug, Clone)]
pub struct DIFunctionType {
    pub return_type: Option<TypeId>,
    pub param_types: Vec<TypeId>,
    pub is_variadic: bool,
}

#[derive(Debug, Clone)]
pub struct DITypedef {
    pub name: String,
    pub underlying: TypeId,
}

/// Debug info for a function
#[derive(Debug, Clone)]
pub struct DISubprogram {
    /// Function name
    pub name: String,

    /// Linkage name (mangled)
    pub linkage_name: String,

    /// Source file
    pub file: PathBuf,

    /// Start line
    pub line: u32,

    /// Function type
    pub ty: TypeId,

    /// Is definition (vs declaration)
    pub is_definition: bool,

    /// Scope
    pub scope: DIScope,

    /// Local variables
    pub variables: Vec<DILocalVariable>,

    /// Parameters
    pub parameters: Vec<DIParameter>,

    /// Inlined at locations
    pub inlined_at: Option<InlinedAt>,

    /// Effects (D-specific)
    pub effects: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DILocalVariable {
    pub name: String,
    pub ty: TypeId,
    pub file: PathBuf,
    pub line: u32,
    pub scope: DIScope,
    pub location: VariableLocation,
}

#[derive(Debug, Clone)]
pub enum VariableLocation {
    /// In a register
    Register(u16),

    /// On stack (frame pointer offset)
    Stack(i32),

    /// Complex location expression
    Expression(Vec<DWOp>),
}

#[derive(Debug, Clone)]
pub enum DWOp {
    /// Push register value
    Reg(u16),

    /// Push frame base
    FrameBase,

    /// Add constant
    PlusConst(i64),

    /// Dereference
    Deref,

    /// Push constant
    Const(i64),
}

#[derive(Debug, Clone)]
pub struct DIParameter {
    pub name: String,
    pub ty: TypeId,
    pub arg_number: u32,
    pub location: VariableLocation,
}

#[derive(Debug, Clone)]
pub struct DIGlobalVariable {
    pub name: String,
    pub linkage_name: String,
    pub ty: TypeId,
    pub file: PathBuf,
    pub line: u32,
    pub is_local: bool,
    pub is_definition: bool,
}

/// Scope for debug info
#[derive(Debug, Clone)]
pub enum DIScope {
    CompileUnit,
    Subprogram(usize), // Index into subprograms
    LexicalBlock(Box<DILexicalBlock>),
    Namespace(String),
    Module(String),
}

#[derive(Debug, Clone)]
pub struct DILexicalBlock {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
    pub parent: DIScope,
}

/// Inlined call site info
#[derive(Debug, Clone)]
pub struct InlinedAt {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
    pub parent: Option<Box<InlinedAt>>,
}

/// Line table entry
#[derive(Debug, Clone)]
pub struct LineTableEntry {
    pub address: u64,
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
    pub is_stmt: bool,
    pub basic_block: bool,
    pub prologue_end: bool,
    pub epilogue_begin: bool,
}

/// Source file info
#[derive(Debug, Clone)]
pub struct DIFile {
    pub path: PathBuf,
    pub directory: PathBuf,
    pub checksum: Option<FileChecksum>,
}

#[derive(Debug, Clone)]
pub struct FileChecksum {
    pub kind: ChecksumKind,
    pub value: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub enum ChecksumKind {
    Md5,
    Sha1,
    Sha256,
}

/// Type ID for cross-referencing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub u64);

/// Optimization level
#[derive(Debug, Clone, Copy)]
pub enum OptLevel {
    None,
    Less,
    Default,
    Aggressive,
}

impl DebugInfoBuilder {
    /// Create a new debug info builder
    pub fn new(file: PathBuf, directory: PathBuf) -> Self {
        DebugInfoBuilder {
            compile_unit: CompileUnit {
                file,
                directory,
                producer: format!("Sounio Compiler {}", env!("CARGO_PKG_VERSION")),
                language: 0x0042, // Custom language ID for D
                optimization_level: OptLevel::None,
                dwarf_version: 5,
            },
            types: HashMap::new(),
            subprograms: Vec::new(),
            globals: Vec::new(),
            scope_stack: vec![DIScope::CompileUnit],
            files: HashMap::new(),
            line_table: Vec::new(),
        }
    }

    /// Set optimization level
    pub fn set_opt_level(&mut self, level: OptLevel) {
        self.compile_unit.optimization_level = level;
    }

    /// Add a basic type
    pub fn add_basic_type(&mut self, name: &str, size_bits: u64, encoding: DWEncoding) -> TypeId {
        let id = TypeId(self.types.len() as u64);
        self.types.insert(
            id,
            DIType::Basic(DIBasicType {
                name: name.to_string(),
                size_bits,
                encoding,
            }),
        );
        id
    }

    /// Add a struct type
    pub fn add_struct_type(
        &mut self,
        name: &str,
        size_bits: u64,
        align_bits: u64,
        members: Vec<DIMember>,
        file: PathBuf,
        line: u32,
    ) -> TypeId {
        let id = TypeId(self.types.len() as u64);
        self.types.insert(
            id,
            DIType::Struct(DIStructType {
                name: name.to_string(),
                size_bits,
                align_bits,
                members,
                file,
                line,
            }),
        );
        id
    }

    /// Add a pointer type
    pub fn add_pointer_type(&mut self, pointee: TypeId) -> TypeId {
        let id = TypeId(self.types.len() as u64);
        self.types.insert(
            id,
            DIType::Pointer(DIPointerType {
                pointee,
                size_bits: 64, // Assume 64-bit pointers
            }),
        );
        id
    }

    /// Add a reference type
    pub fn add_reference_type(&mut self, pointee: TypeId, is_exclusive: bool) -> TypeId {
        let id = TypeId(self.types.len() as u64);
        self.types.insert(
            id,
            DIType::Reference(DIReferenceType {
                pointee,
                is_exclusive,
            }),
        );
        id
    }

    /// Add an array type
    pub fn add_array_type(&mut self, element_type: TypeId, count: Option<u64>) -> TypeId {
        let id = TypeId(self.types.len() as u64);
        self.types.insert(
            id,
            DIType::Array(DIArrayType {
                element_type,
                count,
            }),
        );
        id
    }

    /// Add an enum type
    pub fn add_enum_type(
        &mut self,
        name: &str,
        underlying_type: TypeId,
        variants: Vec<DIEnumVariant>,
    ) -> TypeId {
        let id = TypeId(self.types.len() as u64);
        self.types.insert(
            id,
            DIType::Enum(DIEnumType {
                name: name.to_string(),
                underlying_type,
                variants,
            }),
        );
        id
    }

    /// Add a function type
    pub fn add_function_type(
        &mut self,
        return_type: Option<TypeId>,
        param_types: Vec<TypeId>,
        is_variadic: bool,
    ) -> TypeId {
        let id = TypeId(self.types.len() as u64);
        self.types.insert(
            id,
            DIType::Function(DIFunctionType {
                return_type,
                param_types,
                is_variadic,
            }),
        );
        id
    }

    /// Add a typedef
    pub fn add_typedef(&mut self, name: &str, underlying: TypeId) -> TypeId {
        let id = TypeId(self.types.len() as u64);
        self.types.insert(
            id,
            DIType::Typedef(DITypedef {
                name: name.to_string(),
                underlying,
            }),
        );
        id
    }

    /// Begin a subprogram (function)
    pub fn begin_subprogram(
        &mut self,
        name: &str,
        linkage_name: &str,
        file: PathBuf,
        line: u32,
        ty: TypeId,
    ) -> usize {
        let idx = self.subprograms.len();
        let scope = DIScope::Subprogram(idx);

        self.subprograms.push(DISubprogram {
            name: name.to_string(),
            linkage_name: linkage_name.to_string(),
            file,
            line,
            ty,
            is_definition: true,
            scope: scope.clone(),
            variables: Vec::new(),
            parameters: Vec::new(),
            inlined_at: None,
            effects: Vec::new(),
        });

        self.scope_stack.push(scope);
        idx
    }

    /// End current subprogram
    pub fn end_subprogram(&mut self) {
        self.scope_stack.pop();
    }

    /// Add effects to current function
    pub fn add_effects(&mut self, effects: Vec<String>) {
        if let Some(DIScope::Subprogram(idx)) = self.scope_stack.last() {
            self.subprograms[*idx].effects = effects;
        }
    }

    /// Add a local variable to current function
    pub fn add_local_variable(
        &mut self,
        name: &str,
        ty: TypeId,
        file: PathBuf,
        line: u32,
        location: VariableLocation,
    ) {
        if let Some(DIScope::Subprogram(idx)) = self.scope_stack.last() {
            let scope = self.scope_stack.last().unwrap().clone();
            self.subprograms[*idx].variables.push(DILocalVariable {
                name: name.to_string(),
                ty,
                file,
                line,
                scope,
                location,
            });
        }
    }

    /// Add a parameter to current function
    pub fn add_parameter(
        &mut self,
        name: &str,
        ty: TypeId,
        arg_number: u32,
        location: VariableLocation,
    ) {
        if let Some(DIScope::Subprogram(idx)) = self.scope_stack.last() {
            self.subprograms[*idx].parameters.push(DIParameter {
                name: name.to_string(),
                ty,
                arg_number,
                location,
            });
        }
    }

    /// Add a global variable
    pub fn add_global_variable(
        &mut self,
        name: &str,
        linkage_name: &str,
        ty: TypeId,
        file: PathBuf,
        line: u32,
        is_local: bool,
    ) {
        self.globals.push(DIGlobalVariable {
            name: name.to_string(),
            linkage_name: linkage_name.to_string(),
            ty,
            file,
            line,
            is_local,
            is_definition: true,
        });
    }

    /// Add a line table entry
    pub fn add_line_entry(&mut self, address: u64, file: PathBuf, line: u32, column: u32) {
        self.line_table.push(LineTableEntry {
            address,
            file,
            line,
            column,
            is_stmt: true,
            basic_block: false,
            prologue_end: false,
            epilogue_begin: false,
        });
    }

    /// Mark prologue end
    pub fn mark_prologue_end(&mut self, address: u64) {
        if let Some(entry) = self.line_table.last_mut()
            && entry.address == address
        {
            entry.prologue_end = true;
        }
    }

    /// Mark epilogue begin
    pub fn mark_epilogue_begin(&mut self, address: u64) {
        if let Some(entry) = self.line_table.last_mut()
            && entry.address == address
        {
            entry.epilogue_begin = true;
        }
    }

    /// Enter a lexical block
    pub fn enter_lexical_block(&mut self, file: PathBuf, line: u32, column: u32) {
        let parent = self.scope_stack.last().unwrap().clone();
        let block = DIScope::LexicalBlock(Box::new(DILexicalBlock {
            file,
            line,
            column,
            parent,
        }));
        self.scope_stack.push(block);
    }

    /// Exit lexical block
    pub fn exit_lexical_block(&mut self) {
        if matches!(self.scope_stack.last(), Some(DIScope::LexicalBlock(_))) {
            self.scope_stack.pop();
        }
    }

    /// Enter a namespace
    pub fn enter_namespace(&mut self, name: &str) {
        self.scope_stack.push(DIScope::Namespace(name.to_string()));
    }

    /// Exit namespace
    pub fn exit_namespace(&mut self) {
        if matches!(self.scope_stack.last(), Some(DIScope::Namespace(_))) {
            self.scope_stack.pop();
        }
    }

    /// Register a source file
    pub fn register_file(&mut self, path: PathBuf, directory: PathBuf) {
        self.files.insert(
            path.clone(),
            DIFile {
                path,
                directory,
                checksum: None,
            },
        );
    }

    /// Register a source file with checksum
    pub fn register_file_with_checksum(
        &mut self,
        path: PathBuf,
        directory: PathBuf,
        checksum: FileChecksum,
    ) {
        self.files.insert(
            path.clone(),
            DIFile {
                path,
                directory,
                checksum: Some(checksum),
            },
        );
    }

    /// Get the type by ID
    pub fn get_type(&self, id: TypeId) -> Option<&DIType> {
        self.types.get(&id)
    }

    /// Get all subprograms
    pub fn subprograms(&self) -> &[DISubprogram] {
        &self.subprograms
    }

    /// Get all globals
    pub fn globals(&self) -> &[DIGlobalVariable] {
        &self.globals
    }

    /// Get line table
    pub fn line_table(&self) -> &[LineTableEntry] {
        &self.line_table
    }

    /// Finalize and generate DWARF
    pub fn finalize(self) -> DwarfOutput {
        DwarfOutput::generate(self)
    }
}

/// Output DWARF sections
pub struct DwarfOutput {
    /// .debug_info section
    pub debug_info: Vec<u8>,

    /// .debug_abbrev section
    pub debug_abbrev: Vec<u8>,

    /// .debug_line section
    pub debug_line: Vec<u8>,

    /// .debug_str section
    pub debug_str: Vec<u8>,

    /// .debug_loc section
    pub debug_loc: Vec<u8>,

    /// .debug_ranges section
    pub debug_ranges: Vec<u8>,
}

impl DwarfOutput {
    fn generate(builder: DebugInfoBuilder) -> Self {
        let mut output = DwarfOutput {
            debug_info: Vec::new(),
            debug_abbrev: Vec::new(),
            debug_line: Vec::new(),
            debug_str: Vec::new(),
            debug_loc: Vec::new(),
            debug_ranges: Vec::new(),
        };

        // Generate abbreviations
        output.generate_abbrev();

        // Generate string table
        output.generate_str_table(&builder);

        // Generate debug_info
        output.generate_debug_info(&builder);

        // Generate line table
        output.generate_line_table(&builder);

        output
    }

    fn generate_abbrev(&mut self) {
        // DWARF abbreviation table
        // Each entry defines the structure of a DIE (Debug Information Entry)

        let mut abbrev = Vec::new();

        // Abbreviation 1: Compile Unit
        abbrev.push(1u8); // Abbreviation code
        write_uleb128(&mut abbrev, 0x11); // DW_TAG_compile_unit
        abbrev.push(1); // Has children

        // Attributes for compile unit
        write_uleb128(&mut abbrev, 0x25); // DW_AT_producer
        write_uleb128(&mut abbrev, 0x0e); // DW_FORM_strp
        write_uleb128(&mut abbrev, 0x13); // DW_AT_language
        write_uleb128(&mut abbrev, 0x05); // DW_FORM_data2
        write_uleb128(&mut abbrev, 0x03); // DW_AT_name
        write_uleb128(&mut abbrev, 0x0e); // DW_FORM_strp
        write_uleb128(&mut abbrev, 0x1b); // DW_AT_comp_dir
        write_uleb128(&mut abbrev, 0x0e); // DW_FORM_strp
        write_uleb128(&mut abbrev, 0x00); // End of attributes
        write_uleb128(&mut abbrev, 0x00);

        // Abbreviation 2: Subprogram
        abbrev.push(2u8);
        write_uleb128(&mut abbrev, 0x2e); // DW_TAG_subprogram
        abbrev.push(1); // Has children

        write_uleb128(&mut abbrev, 0x03); // DW_AT_name
        write_uleb128(&mut abbrev, 0x0e); // DW_FORM_strp
        write_uleb128(&mut abbrev, 0x3a); // DW_AT_decl_file
        write_uleb128(&mut abbrev, 0x0b); // DW_FORM_data1
        write_uleb128(&mut abbrev, 0x3b); // DW_AT_decl_line
        write_uleb128(&mut abbrev, 0x0b); // DW_FORM_data1
        write_uleb128(&mut abbrev, 0x00);
        write_uleb128(&mut abbrev, 0x00);

        // Abbreviation 3: Base type
        abbrev.push(3u8);
        write_uleb128(&mut abbrev, 0x24); // DW_TAG_base_type
        abbrev.push(0); // No children

        write_uleb128(&mut abbrev, 0x03); // DW_AT_name
        write_uleb128(&mut abbrev, 0x0e); // DW_FORM_strp
        write_uleb128(&mut abbrev, 0x0b); // DW_AT_byte_size
        write_uleb128(&mut abbrev, 0x0b); // DW_FORM_data1
        write_uleb128(&mut abbrev, 0x3e); // DW_AT_encoding
        write_uleb128(&mut abbrev, 0x0b); // DW_FORM_data1
        write_uleb128(&mut abbrev, 0x00);
        write_uleb128(&mut abbrev, 0x00);

        // End of abbreviations
        abbrev.push(0);

        self.debug_abbrev = abbrev;
    }

    fn generate_str_table(&mut self, builder: &DebugInfoBuilder) {
        // String table - null-terminated strings referenced by offset

        let mut strings = Vec::new();
        strings.push(0u8); // Start with null

        // Add producer
        strings.extend_from_slice(builder.compile_unit.producer.as_bytes());
        strings.push(0);

        // Add file name
        strings.extend_from_slice(builder.compile_unit.file.to_string_lossy().as_bytes());
        strings.push(0);

        // Add directory
        strings.extend_from_slice(builder.compile_unit.directory.to_string_lossy().as_bytes());
        strings.push(0);

        // Add function names
        for subprogram in &builder.subprograms {
            strings.extend_from_slice(subprogram.name.as_bytes());
            strings.push(0);
        }

        self.debug_str = strings;
    }

    fn generate_debug_info(&mut self, builder: &DebugInfoBuilder) {
        let mut info = Vec::new();

        // DWARF header
        let unit_length_offset = info.len();
        info.extend_from_slice(&0u32.to_le_bytes()); // Placeholder for unit length
        info.extend_from_slice(&5u16.to_le_bytes()); // DWARF version 5
        info.push(1); // DW_UT_compile
        info.push(8); // Address size (64-bit)
        info.extend_from_slice(&0u32.to_le_bytes()); // Abbrev offset

        // Compile unit DIE
        write_uleb128(&mut info, 1); // Abbreviation code 1 (compile_unit)
        info.extend_from_slice(&1u32.to_le_bytes()); // Producer string offset
        info.extend_from_slice(&builder.compile_unit.language.to_le_bytes());

        // Calculate and write unit length
        let unit_length = (info.len() - unit_length_offset - 4) as u32;
        info[unit_length_offset..unit_length_offset + 4]
            .copy_from_slice(&unit_length.to_le_bytes());

        self.debug_info = info;
    }

    fn generate_line_table(&mut self, builder: &DebugInfoBuilder) {
        let mut line = Vec::new();

        // Line number program header
        let unit_length_offset = line.len();
        line.extend_from_slice(&0u32.to_le_bytes()); // Placeholder
        line.extend_from_slice(&5u16.to_le_bytes()); // Version 5
        line.push(8); // Address size
        line.push(0); // Segment selector size

        let header_length_offset = line.len();
        line.extend_from_slice(&0u32.to_le_bytes()); // Header length placeholder

        // Standard opcode lengths
        line.push(1); // Minimum instruction length
        line.push(1); // Maximum ops per instruction
        line.push(1); // Default is_stmt
        line.push((-5i8) as u8); // Line base
        line.push(14); // Line range
        line.push(13); // Opcode base

        // Standard opcode lengths array
        for i in 1..13 {
            line.push(match i {
                1 => 0,  // DW_LNS_copy
                2 => 1,  // DW_LNS_advance_pc
                3 => 1,  // DW_LNS_advance_line
                4 => 1,  // DW_LNS_set_file
                5 => 1,  // DW_LNS_set_column
                6 => 0,  // DW_LNS_negate_stmt
                7 => 0,  // DW_LNS_set_basic_block
                8 => 0,  // DW_LNS_const_add_pc
                9 => 1,  // DW_LNS_fixed_advance_pc
                10 => 0, // DW_LNS_set_prologue_end
                11 => 0, // DW_LNS_set_epilogue_begin
                12 => 1, // DW_LNS_set_isa
                _ => 0,
            });
        }

        // Directory table (DWARF 5 format)
        line.push(1); // Directory entry format count
        write_uleb128(&mut line, 0x01); // DW_LNCT_path
        write_uleb128(&mut line, 0x08); // DW_FORM_string
        write_uleb128(&mut line, 1); // Directory count
        line.extend_from_slice(builder.compile_unit.directory.to_string_lossy().as_bytes());
        line.push(0);

        // File name table
        line.push(2); // File name entry format count
        write_uleb128(&mut line, 0x01); // DW_LNCT_path
        write_uleb128(&mut line, 0x08); // DW_FORM_string
        write_uleb128(&mut line, 0x02); // DW_LNCT_directory_index
        write_uleb128(&mut line, 0x0b); // DW_FORM_data1
        write_uleb128(&mut line, 1); // File count
        line.extend_from_slice(builder.compile_unit.file.to_string_lossy().as_bytes());
        line.push(0);
        line.push(0); // Directory index

        // Calculate header length
        let header_length = (line.len() - header_length_offset - 4) as u32;
        line[header_length_offset..header_length_offset + 4]
            .copy_from_slice(&header_length.to_le_bytes());

        // Line number program
        for entry in &builder.line_table {
            if entry.prologue_end {
                line.push(10); // DW_LNS_set_prologue_end
            }
            if entry.epilogue_begin {
                line.push(11); // DW_LNS_set_epilogue_begin
            }

            // Set address (extended opcode)
            line.push(0); // Extended opcode marker
            write_uleb128(&mut line, 9); // Length
            line.push(2); // DW_LNE_set_address
            line.extend_from_slice(&entry.address.to_le_bytes());

            // Advance line
            line.push(3); // DW_LNS_advance_line
            write_sleb128(&mut line, entry.line as i64);

            // Set column
            line.push(5); // DW_LNS_set_column
            write_uleb128(&mut line, entry.column as u64);

            // Copy row
            line.push(1); // DW_LNS_copy
        }

        // End sequence
        line.push(0);
        line.push(1);
        line.push(1); // DW_LNE_end_sequence

        // Calculate unit length
        let unit_length = (line.len() - unit_length_offset - 4) as u32;
        line[unit_length_offset..unit_length_offset + 4]
            .copy_from_slice(&unit_length.to_le_bytes());

        self.debug_line = line;
    }
}

/// Write unsigned LEB128
fn write_uleb128(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Write signed LEB128
fn write_sleb128(buf: &mut Vec<u8>, mut value: i64) {
    let mut more = true;
    while more {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;

        let sign_bit = (byte & 0x40) != 0;
        if (value == 0 && !sign_bit) || (value == -1 && sign_bit) {
            more = false;
        } else {
            byte |= 0x80;
        }
        buf.push(byte);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_info_builder() {
        let mut builder =
            DebugInfoBuilder::new(PathBuf::from("test.sio"), PathBuf::from("/home/user/project"));

        // Add basic types
        let int_type = builder.add_basic_type("int", 32, DWEncoding::SignedInt);
        let bool_type = builder.add_basic_type("bool", 8, DWEncoding::Boolean);

        // Add function type
        let fn_type = builder.add_function_type(Some(int_type), vec![bool_type], false);

        // Begin a function
        let _idx = builder.begin_subprogram("main", "_D4main", PathBuf::from("test.sio"), 1, fn_type);

        // Add a parameter
        builder.add_parameter("flag", bool_type, 0, VariableLocation::Register(0));

        // Add a local variable
        builder.add_local_variable(
            "count",
            int_type,
            PathBuf::from("test.sio"),
            3,
            VariableLocation::Stack(-8),
        );

        // Add line entries
        builder.add_line_entry(0x1000, PathBuf::from("test.sio"), 1, 1);
        builder.add_line_entry(0x1010, PathBuf::from("test.sio"), 2, 5);

        builder.end_subprogram();

        // Finalize
        let output = builder.finalize();

        assert!(!output.debug_info.is_empty());
        assert!(!output.debug_abbrev.is_empty());
        assert!(!output.debug_line.is_empty());
    }

    #[test]
    fn test_uleb128() {
        let mut buf = Vec::new();
        write_uleb128(&mut buf, 127);
        assert_eq!(buf, vec![127]);

        buf.clear();
        write_uleb128(&mut buf, 128);
        assert_eq!(buf, vec![0x80, 0x01]);

        buf.clear();
        write_uleb128(&mut buf, 624485);
        assert_eq!(buf, vec![0xe5, 0x8e, 0x26]);
    }
}
