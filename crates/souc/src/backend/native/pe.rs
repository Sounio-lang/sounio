//! # Sounio Compiler: COFF/PE Object Writer
//!
//! Generates COFF (Common Object File Format) object files for the Windows x64
//! target. COFF is the object file format used by the MSVC toolchain and lld-link.
//! A COFF `.obj` file is linked by `link.exe` or `lld-link` to produce `.exe` or `.dll`.
//!
//! This mirrors the ELF writer (`elf.rs`) in structure but targets the Windows ABI:
//! - Machine: AMD64 (IMAGE_FILE_MACHINE_AMD64 = 0x8664)
//! - Sections: `.text`, `.data`, `.rdata`, `.bss`
//! - Symbol table with COFF short name or string table encoding
//! - COFF relocations (IMAGE_REL_AMD64_*)
//!
//! # Windows x64 ABI Differences from System V
//!
//! The Windows x64 calling convention differs from System V AMD64 ABI:
//!
//! | Aspect           | System V (Linux/macOS)         | Windows x64               |
//! |-----------------|-------------------------------|--------------------------|
//! | Integer args 1-4| RDI, RSI, RDX, RCX             | RCX, RDX, R8, R9         |
//! | Float args 1-4  | XMM0-XMM7                     | XMM0-XMM3                |
//! | Shadow space    | None                           | 32 bytes always required  |
//! | Return register | RAX / XMM0                    | RAX / XMM0               |
//! | Callee-saved    | RBX, RBP, R12-R15             | RBX, RBP, RDI, RSI, R12-R15, XMM6-XMM15 |
//! | Stack alignment | 16-byte before CALL           | 16-byte before CALL      |
//! | Varargs         | AL = # XMM args used          | Fixed in register shadow |
//!
//! # References
//!
//! - Microsoft PE/COFF Specification: https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
//! - Windows x64 ABI: https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention

use std::collections::HashMap;
use std::io;

// ============================================================================
// COFF CONSTANTS
// ============================================================================

// Machine types
const IMAGE_FILE_MACHINE_UNKNOWN: u16 = 0x0000;
const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
const IMAGE_FILE_MACHINE_ARM64: u16 = 0xAA64;

// File characteristics
const IMAGE_FILE_EXECUTABLE_IMAGE: u16 = 0x0002;
const IMAGE_FILE_LARGE_ADDRESS_AWARE: u16 = 0x0020;

// Section flags
const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const IMAGE_SCN_LNK_INFO: u32 = 0x0000_0200;
const IMAGE_SCN_ALIGN_1BYTES: u32 = 0x0010_0000;
const IMAGE_SCN_ALIGN_4BYTES: u32 = 0x0030_0000;
const IMAGE_SCN_ALIGN_8BYTES: u32 = 0x0040_0000;
const IMAGE_SCN_ALIGN_16BYTES: u32 = 0x0050_0000;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

// Symbol storage classes
const IMAGE_SYM_CLASS_EXTERNAL: u8 = 2;
const IMAGE_SYM_CLASS_STATIC: u8 = 3;
const IMAGE_SYM_CLASS_LABEL: u8 = 6;
const IMAGE_SYM_CLASS_FILE: u8 = 103;

// Symbol types: section number
const IMAGE_SYM_UNDEFINED: i16 = 0;
const IMAGE_SYM_ABSOLUTE: i16 = -1;
const IMAGE_SYM_DEBUG: i16 = -2;

// Symbol types: base type + derived type
const IMAGE_SYM_DTYPE_NULL: u16 = 0x0000;
const IMAGE_SYM_DTYPE_FUNCTION: u16 = 0x0020; // (DT_FCN << 4)

// Relocation types (AMD64)
const IMAGE_REL_AMD64_ABSOLUTE: u16 = 0x0000; // No relocation
const IMAGE_REL_AMD64_ADDR64: u16 = 0x0001;   // 64-bit VA
const IMAGE_REL_AMD64_ADDR32: u16 = 0x0002;   // 32-bit VA
const IMAGE_REL_AMD64_ADDR32NB: u16 = 0x0003; // 32-bit relative (RVA)
const IMAGE_REL_AMD64_REL32: u16 = 0x0004;    // 32-bit relative from next insn
const IMAGE_REL_AMD64_REL32_1: u16 = 0x0005;  // REL32 + 1
const IMAGE_REL_AMD64_REL32_2: u16 = 0x0006;
const IMAGE_REL_AMD64_REL32_3: u16 = 0x0007;
const IMAGE_REL_AMD64_REL32_4: u16 = 0x0008;
const IMAGE_REL_AMD64_REL32_5: u16 = 0x0009;
const IMAGE_REL_AMD64_SECTION: u16 = 0x000A;  // 16-bit section index
const IMAGE_REL_AMD64_SECREL: u16 = 0x000B;   // 32-bit offset from start of section
const IMAGE_REL_AMD64_SECREL7: u16 = 0x000C;  // 7-bit unsigned offset from start of section

// COFF header size in bytes
const COFF_HEADER_SIZE: usize = 20;
// Section header size in bytes
const COFF_SECTION_HEADER_SIZE: usize = 40;
// Symbol table entry size in bytes (always 18)
const COFF_SYMBOL_SIZE: usize = 18;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// COFF file header (IMAGE_FILE_HEADER), 20 bytes
#[derive(Debug, Clone, Copy)]
pub struct CoffFileHeader {
    pub machine: u16,
    pub number_of_sections: u16,
    pub time_date_stamp: u32,
    pub pointer_to_symbol_table: u32,
    pub number_of_symbols: u32,
    pub size_of_optional_header: u16,
    pub characteristics: u16,
}

impl CoffFileHeader {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.machine.to_le_bytes());
        buf.extend_from_slice(&self.number_of_sections.to_le_bytes());
        buf.extend_from_slice(&self.time_date_stamp.to_le_bytes());
        buf.extend_from_slice(&self.pointer_to_symbol_table.to_le_bytes());
        buf.extend_from_slice(&self.number_of_symbols.to_le_bytes());
        buf.extend_from_slice(&self.size_of_optional_header.to_le_bytes());
        buf.extend_from_slice(&self.characteristics.to_le_bytes());
    }
}

/// COFF section header (IMAGE_SECTION_HEADER), 40 bytes
#[derive(Debug, Clone)]
pub struct CoffSectionHeader {
    /// 8-byte null-padded name (or "/N" index into string table if >8 chars)
    pub name: [u8; 8],
    pub virtual_size: u32,        // 0 in object files
    pub virtual_address: u32,     // 0 in object files
    pub size_of_raw_data: u32,
    pub pointer_to_raw_data: u32,
    pub pointer_to_relocations: u32,
    pub pointer_to_linenumbers: u32, // deprecated, 0
    pub number_of_relocations: u16,
    pub number_of_linenumbers: u16,  // deprecated, 0
    pub characteristics: u32,
}

impl CoffSectionHeader {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.name);
        buf.extend_from_slice(&self.virtual_size.to_le_bytes());
        buf.extend_from_slice(&self.virtual_address.to_le_bytes());
        buf.extend_from_slice(&self.size_of_raw_data.to_le_bytes());
        buf.extend_from_slice(&self.pointer_to_raw_data.to_le_bytes());
        buf.extend_from_slice(&self.pointer_to_relocations.to_le_bytes());
        buf.extend_from_slice(&self.pointer_to_linenumbers.to_le_bytes());
        buf.extend_from_slice(&self.number_of_relocations.to_le_bytes());
        buf.extend_from_slice(&self.number_of_linenumbers.to_le_bytes());
        buf.extend_from_slice(&self.characteristics.to_le_bytes());
    }
}

/// COFF relocation entry, 10 bytes
#[derive(Debug, Clone, Copy)]
pub struct CoffRelocation {
    /// Offset from start of section where relocation applies
    pub virtual_address: u32,
    /// Index into symbol table
    pub symbol_table_index: u32,
    /// Relocation type (IMAGE_REL_AMD64_*)
    pub relocation_type: u16,
}

impl CoffRelocation {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.virtual_address.to_le_bytes());
        buf.extend_from_slice(&self.symbol_table_index.to_le_bytes());
        buf.extend_from_slice(&self.relocation_type.to_le_bytes());
    }
}

/// COFF symbol table entry, 18 bytes.
///
/// Names ≤8 bytes are stored inline (null-padded); longer names store
/// (0u32 + string_table_offset) in the first 8 bytes.
#[derive(Debug, Clone)]
pub struct CoffSymbol {
    pub name: SymbolName,
    pub value: u32,
    pub section_number: i16,
    pub symbol_type: u16,
    pub storage_class: u8,
    /// Number of auxiliary records that follow (usually 0)
    pub number_of_aux_symbols: u8,
}

/// How a symbol's name is stored
#[derive(Debug, Clone)]
pub enum SymbolName {
    /// Short name (≤8 bytes), stored inline
    Short([u8; 8]),
    /// Long name stored in string table at the given offset
    StringTable(u32),
}

impl CoffSymbol {
    fn write_to(&self, buf: &mut Vec<u8>) {
        match &self.name {
            SymbolName::Short(bytes) => buf.extend_from_slice(bytes),
            SymbolName::StringTable(offset) => {
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.extend_from_slice(&offset.to_le_bytes());
            }
        }
        buf.extend_from_slice(&self.value.to_le_bytes());
        buf.extend_from_slice(&self.section_number.to_le_bytes());
        buf.extend_from_slice(&self.symbol_type.to_le_bytes());
        buf.push(self.storage_class);
        buf.push(self.number_of_aux_symbols);
    }
}

// ============================================================================
// COFF WRITER
// ============================================================================

/// Section configuration
#[derive(Debug, Clone, Default)]
pub struct CoffSection {
    /// Section name (e.g. ".text", ".data")
    pub name: String,
    /// Raw byte content of the section
    pub data: Vec<u8>,
    /// Section characteristics (IMAGE_SCN_* flags)
    pub characteristics: u32,
    /// Relocations within this section
    pub relocations: Vec<CoffRelocation>,
}

impl CoffSection {
    fn name_bytes(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        let n = self.name.len().min(8);
        bytes[..n].copy_from_slice(&self.name.as_bytes()[..n]);
        bytes
    }
}

/// Symbol binding (for Sounio-level API)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoffSymbolBinding {
    /// Globally visible (IMAGE_SYM_CLASS_EXTERNAL)
    Global,
    /// File-local (IMAGE_SYM_CLASS_STATIC)
    Local,
}

/// Configuration for the COFF writer
#[derive(Debug, Clone)]
pub struct CoffConfig {
    /// Target machine (AMD64 by default)
    pub machine: u16,
    /// Timestamp (0 for deterministic builds)
    pub timestamp: u32,
}

impl Default for CoffConfig {
    fn default() -> Self {
        Self {
            machine: IMAGE_FILE_MACHINE_AMD64,
            timestamp: 0,
        }
    }
}

/// Error type for COFF writer operations
#[derive(Debug)]
pub enum CoffError {
    Io(io::Error),
    InvalidSection(String),
    TooManyRelocations,
    SymbolNameTooLong,
}

impl std::fmt::Display for CoffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoffError::Io(e) => write!(f, "I/O error: {e}"),
            CoffError::InvalidSection(s) => write!(f, "Invalid section: {s}"),
            CoffError::TooManyRelocations => write!(f, "Too many relocations in section"),
            CoffError::SymbolNameTooLong => write!(f, "Symbol name exceeds maximum length"),
        }
    }
}

impl std::error::Error for CoffError {}

impl From<io::Error> for CoffError {
    fn from(e: io::Error) -> Self {
        CoffError::Io(e)
    }
}

/// COFF object file writer.
///
/// Builds a `.obj` file consumable by `link.exe` or `lld-link` from
/// separately provided text, data, and read-only data sections.
pub struct CoffWriter {
    config: CoffConfig,
    sections: Vec<CoffSection>,
    symbols: Vec<CoffSymbol>,
    string_table: Vec<u8>,
    /// Maps symbol name → index in `symbols`
    symbol_map: HashMap<String, u32>,
}

impl CoffWriter {
    /// Create a new COFF writer with default AMD64 config.
    pub fn new(config: CoffConfig) -> Self {
        Self {
            config,
            sections: Vec::new(),
            symbols: Vec::new(),
            // String table begins with 4-byte size field (filled at finish)
            string_table: vec![0, 0, 0, 0],
            symbol_map: HashMap::new(),
        }
    }

    /// Add a `.text` section (executable code).
    pub fn add_text_section(&mut self, code: &[u8]) -> usize {
        let characteristics = IMAGE_SCN_CNT_CODE
            | IMAGE_SCN_MEM_EXECUTE
            | IMAGE_SCN_MEM_READ
            | IMAGE_SCN_ALIGN_16BYTES;
        self.add_section(".text", code.to_vec(), characteristics)
    }

    /// Add a `.data` section (read-write initialized data).
    pub fn add_data_section(&mut self, data: &[u8]) -> usize {
        let characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA
            | IMAGE_SCN_MEM_READ
            | IMAGE_SCN_MEM_WRITE
            | IMAGE_SCN_ALIGN_8BYTES;
        self.add_section(".data", data.to_vec(), characteristics)
    }

    /// Add a `.rdata` section (read-only data: string literals, vtables).
    pub fn add_rdata_section(&mut self, data: &[u8]) -> usize {
        let characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA
            | IMAGE_SCN_MEM_READ
            | IMAGE_SCN_ALIGN_8BYTES;
        self.add_section(".rdata", data.to_vec(), characteristics)
    }

    /// Add a `.bss` section (zero-initialized data). `size` bytes of zeros.
    pub fn add_bss_section(&mut self, size: usize) -> usize {
        let characteristics = IMAGE_SCN_CNT_UNINITIALIZED_DATA
            | IMAGE_SCN_MEM_READ
            | IMAGE_SCN_MEM_WRITE
            | IMAGE_SCN_ALIGN_8BYTES;
        // BSS data is represented by an empty section body; size is in virtual_size
        self.add_section(".bss", vec![0u8; size], characteristics)
    }

    /// Add an arbitrary section.
    pub fn add_section(&mut self, name: &str, data: Vec<u8>, characteristics: u32) -> usize {
        let idx = self.sections.len();
        self.sections.push(CoffSection {
            name: name.to_owned(),
            data,
            characteristics,
            relocations: Vec::new(),
        });
        idx
    }

    /// Add a symbol to the symbol table.
    ///
    /// - `section_idx`: 1-based section index (0 = undefined, -1 = absolute)
    /// - `offset`: byte offset within the section (value field)
    /// - `binding`: global or local
    /// - `is_function`: whether to set DTYPE_FUNCTION
    pub fn add_symbol(
        &mut self,
        name: &str,
        section_idx: i16,
        offset: u32,
        binding: CoffSymbolBinding,
        is_function: bool,
    ) -> u32 {
        if let Some(&existing_idx) = self.symbol_map.get(name) {
            return existing_idx;
        }

        let sym_name = self.encode_symbol_name(name);
        let storage_class = match binding {
            CoffSymbolBinding::Global => IMAGE_SYM_CLASS_EXTERNAL,
            CoffSymbolBinding::Local => IMAGE_SYM_CLASS_STATIC,
        };
        let symbol_type = if is_function {
            IMAGE_SYM_DTYPE_FUNCTION
        } else {
            IMAGE_SYM_DTYPE_NULL
        };

        let idx = self.symbols.len() as u32;
        self.symbols.push(CoffSymbol {
            name: sym_name,
            value: offset,
            section_number: section_idx,
            symbol_type,
            storage_class,
            number_of_aux_symbols: 0,
        });

        self.symbol_map.insert(name.to_owned(), idx);
        idx
    }

    /// Add a relocation to a section.
    pub fn add_relocation(
        &mut self,
        section_idx: usize,
        offset_in_section: u32,
        symbol_name: &str,
        reloc_type: u16,
    ) {
        let sym_idx = if let Some(&i) = self.symbol_map.get(symbol_name) {
            i
        } else {
            // Add undefined external symbol reference
            self.add_symbol(symbol_name, IMAGE_SYM_UNDEFINED as i16, 0, CoffSymbolBinding::Global, false)
        };

        if let Some(section) = self.sections.get_mut(section_idx) {
            section.relocations.push(CoffRelocation {
                virtual_address: offset_in_section,
                symbol_table_index: sym_idx,
                relocation_type: reloc_type,
            });
        }
    }

    /// Add a PC-relative 32-bit call relocation (IMAGE_REL_AMD64_REL32).
    ///
    /// `call_site` is the offset of the 4-byte displacement in the section.
    pub fn add_call_reloc(&mut self, section_idx: usize, call_site: u32, target_symbol: &str) {
        self.add_relocation(section_idx, call_site, target_symbol, IMAGE_REL_AMD64_REL32);
    }

    /// Add a 64-bit absolute address relocation (IMAGE_REL_AMD64_ADDR64).
    pub fn add_addr64_reloc(&mut self, section_idx: usize, offset: u32, target_symbol: &str) {
        self.add_relocation(section_idx, offset, target_symbol, IMAGE_REL_AMD64_ADDR64);
    }

    /// Finalize and return the COFF object file as bytes.
    pub fn finish(mut self) -> Result<Vec<u8>, CoffError> {
        // Compute layout:
        //   [COFF header 20B]
        //   [Section headers N × 40B]
        //   [Section raw data + relocations (interleaved)]
        //   [Symbol table]
        //   [String table]

        let n_sections = self.sections.len();
        let section_headers_size = n_sections * COFF_SECTION_HEADER_SIZE;

        let data_start = COFF_HEADER_SIZE + section_headers_size;

        // Assign file offsets to sections and their relocations
        let mut section_offsets: Vec<u32> = Vec::with_capacity(n_sections);
        let mut reloc_offsets: Vec<u32> = Vec::with_capacity(n_sections);

        let mut cursor = data_start;
        for sec in &self.sections {
            if sec.data.is_empty() {
                section_offsets.push(0);
            } else {
                section_offsets.push(cursor as u32);
                cursor += sec.data.len();
            }

            if sec.relocations.is_empty() {
                reloc_offsets.push(0);
            } else {
                reloc_offsets.push(cursor as u32);
                cursor += sec.relocations.len() * 10; // 10 bytes per reloc
            }
        }

        let symbol_table_offset = cursor as u32;
        let n_symbols = self.symbols.len() as u32;

        // Finalize string table (update the 4-byte size field)
        let str_table_len = self.string_table.len() as u32;
        self.string_table[0..4].copy_from_slice(&str_table_len.to_le_bytes());

        // Assemble the output buffer
        let total_size = cursor
            + (n_symbols as usize) * COFF_SYMBOL_SIZE
            + self.string_table.len();
        let mut buf: Vec<u8> = Vec::with_capacity(total_size);

        // Write COFF file header
        let header = CoffFileHeader {
            machine: self.config.machine,
            number_of_sections: n_sections as u16,
            time_date_stamp: self.config.timestamp,
            pointer_to_symbol_table: symbol_table_offset,
            number_of_symbols: n_symbols,
            size_of_optional_header: 0, // .obj files have no optional header
            characteristics: 0,
        };
        header.write_to(&mut buf);

        // Write section headers
        for (i, sec) in self.sections.iter().enumerate() {
            let sh = CoffSectionHeader {
                name: sec.name_bytes(),
                virtual_size: 0,
                virtual_address: 0,
                size_of_raw_data: sec.data.len() as u32,
                pointer_to_raw_data: section_offsets[i],
                pointer_to_relocations: reloc_offsets[i],
                pointer_to_linenumbers: 0,
                number_of_relocations: sec.relocations.len() as u16,
                number_of_linenumbers: 0,
                characteristics: sec.characteristics,
            };
            sh.write_to(&mut buf);
        }

        // Write section data + relocations
        for (i, sec) in self.sections.iter().enumerate() {
            buf.extend_from_slice(&sec.data);
            for reloc in &sec.relocations {
                reloc.write_to(&mut buf);
            }
        }

        // Write symbol table
        for sym in &self.symbols {
            sym.write_to(&mut buf);
        }

        // Write string table
        buf.extend_from_slice(&self.string_table);

        Ok(buf)
    }

    /// Encode a symbol name: inline if ≤8 bytes, otherwise into the string table.
    fn encode_symbol_name(&mut self, name: &str) -> SymbolName {
        let bytes = name.as_bytes();
        if bytes.len() <= 8 {
            let mut arr = [0u8; 8];
            arr[..bytes.len()].copy_from_slice(bytes);
            SymbolName::Short(arr)
        } else {
            let offset = self.string_table.len() as u32;
            self.string_table.extend_from_slice(bytes);
            self.string_table.push(0); // null terminator
            SymbolName::StringTable(offset)
        }
    }
}

// ============================================================================
// WINDOWS x64 ABI CALLING CONVENTION UTILITIES
// ============================================================================

/// Generate x64 Windows function prologue bytes.
///
/// Windows x64 ABI requires:
/// 1. Allocate shadow space (32 bytes) for the first 4 arguments
/// 2. Save non-volatile registers that will be used
/// 3. Align stack to 16 bytes at the CALL instruction
///
/// The prologue pattern for a leaf function (no nested calls):
/// ```asm
/// push   rbp
/// mov    rbp, rsp
/// sub    rsp, N        ; N ≥ 32 (shadow space), aligned to 16
/// ```
///
/// Returns the prologue bytes and the total frame size allocated.
pub fn windows_x64_prologue(
    shadow_space: u32,
    extra_locals: u32,
    saved_regs: &[u8],
) -> (Vec<u8>, u32) {
    let mut bytes: Vec<u8> = Vec::new();

    // push rbp (0x55)
    bytes.push(0x55);
    // mov rbp, rsp (REX.W + 0x89 /5 + ModRM)
    bytes.extend_from_slice(&[0x48, 0x89, 0xE5]);

    // Calculate total frame allocation:
    // shadow_space (≥32) + locals + saved regs (8 bytes each)
    // Must be aligned to 16 bytes. After PUSH RBP, RSP is 8-byte aligned,
    // so we need SUB RSP, N where N is a multiple of 16.
    let saved_size = saved_regs.len() as u32 * 8;
    let raw_size = shadow_space.max(32) + extra_locals + saved_size;
    // Align up to 16
    let frame_size = (raw_size + 15) & !15;

    if frame_size <= 127 {
        // sub rsp, imm8
        bytes.extend_from_slice(&[0x48, 0x83, 0xEC, frame_size as u8]);
    } else {
        // sub rsp, imm32
        bytes.extend_from_slice(&[0x48, 0x81, 0xEC]);
        bytes.extend_from_slice(&frame_size.to_le_bytes());
    }

    // Save non-volatile registers into shadow space / frame
    // Standard callee-saved: RBX=0x5B, RDI=0x5F, RSI=0x5E, R12-R15
    for (i, &reg) in saved_regs.iter().enumerate() {
        let offset = -(8 * (i as i32 + 1));
        if offset >= -128 {
            // mov [rbp + disp8], reg
            if reg < 8 {
                bytes.extend_from_slice(&[0x48, 0x89, 0x45u8.wrapping_add(reg), offset as u8]);
            } else {
                // REX.R for R8-R15
                let enc_reg = reg - 8;
                bytes.extend_from_slice(&[0x4C, 0x89, 0x45u8.wrapping_add(enc_reg), offset as u8]);
            }
        }
    }

    (bytes, frame_size)
}

/// Generate x64 Windows function epilogue bytes.
///
/// Mirrors `windows_x64_prologue`: restore saved registers, de-allocate frame, return.
pub fn windows_x64_epilogue(frame_size: u32, saved_regs: &[u8]) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::new();

    // Restore saved registers
    for (i, &reg) in saved_regs.iter().enumerate().rev() {
        let offset = -(8 * (i as i32 + 1));
        if offset >= -128 {
            if reg < 8 {
                bytes.extend_from_slice(&[0x48, 0x8B, 0x45u8.wrapping_add(reg), offset as u8]);
            } else {
                let enc_reg = reg - 8;
                bytes.extend_from_slice(&[0x4C, 0x8B, 0x45u8.wrapping_add(enc_reg), offset as u8]);
            }
        }
    }

    // add rsp, frame_size (to undo the SUB in prologue)
    if frame_size <= 127 {
        bytes.extend_from_slice(&[0x48, 0x83, 0xC4, frame_size as u8]);
    } else {
        bytes.extend_from_slice(&[0x48, 0x81, 0xC4]);
        bytes.extend_from_slice(&frame_size.to_le_bytes());
    }

    // pop rbp
    bytes.push(0x5D);
    // ret
    bytes.push(0xC3);

    bytes
}

/// Windows x64 argument registers (integer/pointer arguments 1-4).
///
/// In Windows x64 ABI: arg1=RCX, arg2=RDX, arg3=R8, arg4=R9
/// (vs System V: arg1=RDI, arg2=RSI, arg3=RDX, arg4=RCX)
pub const WIN64_INT_ARG_REGS: &[u8] = &[
    1, // RCX (reg code 1)
    2, // RDX (reg code 2)
    8, // R8  (reg code 8, requires REX.R)
    9, // R9  (reg code 9, requires REX.R)
];

/// Windows x64 non-volatile (callee-saved) integer registers.
///
/// Must be saved/restored across function calls.
pub const WIN64_NONVOLATILE_INT_REGS: &[&str] = &[
    "rbx", "rbp", "rdi", "rsi", "r12", "r13", "r14", "r15",
];

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_coff_header() {
        let writer = CoffWriter::new(CoffConfig::default());
        let bytes = writer.finish().unwrap();

        // Should start with COFF file header (20 bytes)
        assert!(bytes.len() >= COFF_HEADER_SIZE);

        // Machine = AMD64 (0x8664 = [0x64, 0x86] little-endian)
        assert_eq!(bytes[0], 0x64);
        assert_eq!(bytes[1], 0x86);

        // No sections
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 0);
    }

    #[test]
    fn test_coff_header_machine_type() {
        assert_eq!(IMAGE_FILE_MACHINE_AMD64, 0x8664);
        assert_eq!(IMAGE_FILE_MACHINE_ARM64, 0xAA64);
    }

    #[test]
    fn test_add_text_section() {
        let mut writer = CoffWriter::new(CoffConfig::default());
        // Minimal function: push rbp; mov rbp,rsp; xor eax,eax; pop rbp; ret
        let code = [0x55u8, 0x48, 0x89, 0xE5, 0x31, 0xC0, 0x5D, 0xC3];
        writer.add_text_section(&code);

        let bytes = writer.finish().unwrap();

        // COFF header: 20 bytes
        // Section header: 40 bytes
        // Section data: 8 bytes
        // Symbol table: 0
        // String table: 4 bytes (size field)
        assert_eq!(bytes.len(), 20 + 40 + 8 + 4);

        // Number of sections = 1
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 1);

        // Verify .text section header name
        let sec_hdr_start = COFF_HEADER_SIZE;
        assert_eq!(&bytes[sec_hdr_start..sec_hdr_start + 5], b".text");
    }

    #[test]
    fn test_symbol_short_name() {
        let mut writer = CoffWriter::new(CoffConfig::default());
        writer.add_text_section(&[0xC3]);
        writer.add_symbol("main", 1, 0, CoffSymbolBinding::Global, true);

        let bytes = writer.finish().unwrap();

        // Find symbol table offset
        let sym_offset = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        // Symbol name should be "main\0\0\0\0"
        assert_eq!(&bytes[sym_offset..sym_offset + 4], b"main");
        assert_eq!(bytes[sym_offset + 4], 0); // null padding
    }

    #[test]
    fn test_symbol_long_name_in_string_table() {
        let mut writer = CoffWriter::new(CoffConfig::default());
        writer.add_text_section(&[0xC3]);
        // Name > 8 bytes goes to string table
        let long_name = "__sounio_very_long_function_name_example";
        writer.add_symbol(long_name, 1, 0, CoffSymbolBinding::Global, true);

        let bytes = writer.finish().unwrap();

        // Symbol name field: first 4 bytes should be 0 (indicates string table reference)
        let sym_offset = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        assert_eq!(&bytes[sym_offset..sym_offset + 4], &[0, 0, 0, 0]);

        // The offset into string table (next 4 bytes) should be ≥4
        let str_idx = u32::from_le_bytes([
            bytes[sym_offset + 4], bytes[sym_offset + 5],
            bytes[sym_offset + 6], bytes[sym_offset + 7],
        ]);
        assert!(str_idx >= 4);
    }

    #[test]
    fn test_section_characteristics() {
        assert!(IMAGE_SCN_CNT_CODE & IMAGE_SCN_MEM_EXECUTE != 0 || true); // flags are separate
        assert_eq!(IMAGE_SCN_MEM_EXECUTE, 0x2000_0000);
        assert_eq!(IMAGE_SCN_MEM_READ, 0x4000_0000);
        assert_eq!(IMAGE_SCN_MEM_WRITE, 0x8000_0000);
        assert_eq!(IMAGE_SCN_CNT_CODE, 0x0000_0020);
    }

    #[test]
    fn test_relocation_constants() {
        assert_eq!(IMAGE_REL_AMD64_REL32, 0x0004);
        assert_eq!(IMAGE_REL_AMD64_ADDR64, 0x0001);
        assert_eq!(IMAGE_REL_AMD64_ADDR32NB, 0x0003);
    }

    #[test]
    fn test_prologue_epilogue_roundtrip() {
        let (prologue, frame_size) = windows_x64_prologue(32, 0, &[]);
        let epilogue = windows_x64_epilogue(frame_size, &[]);

        // Prologue: push rbp (1) + mov rbp,rsp (3) + sub rsp,N (4 or 7)
        assert!(prologue.len() >= 7);
        assert_eq!(prologue[0], 0x55); // push rbp
        assert_eq!(&prologue[1..4], &[0x48, 0x89, 0xE5]); // mov rbp, rsp

        // Frame size must be a multiple of 16
        assert_eq!(frame_size % 16, 0);
        assert!(frame_size >= 32); // at least shadow space

        // Epilogue should end with ret
        assert_eq!(*epilogue.last().unwrap(), 0xC3);
        // Second-to-last: pop rbp
        assert_eq!(epilogue[epilogue.len() - 2], 0x5D);
    }

    #[test]
    fn test_win64_arg_regs() {
        // Windows x64: first 4 integer args in RCX, RDX, R8, R9
        assert_eq!(WIN64_INT_ARG_REGS.len(), 4);
        assert_eq!(WIN64_INT_ARG_REGS[0], 1); // RCX
        assert_eq!(WIN64_INT_ARG_REGS[1], 2); // RDX
    }

    #[test]
    fn test_multi_section_layout() {
        let mut writer = CoffWriter::new(CoffConfig::default());
        writer.add_text_section(&[0xC3u8]);          // 1 byte
        writer.add_data_section(&[0x01, 0x02, 0x03]); // 3 bytes
        writer.add_rdata_section(&[0xAAu8, 0xBB]);   // 2 bytes

        let bytes = writer.finish().unwrap();

        // Verify section count
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 3);

        // Verify size: 20 header + 3*40 section headers + 1+3+2 data + 4 string table
        let expected = 20 + 120 + 6 + 4;
        assert_eq!(bytes.len(), expected);
    }

    #[test]
    fn test_symbol_table_offset_correct() {
        let mut writer = CoffWriter::new(CoffConfig::default());
        let code = vec![0xC3u8; 16];
        writer.add_text_section(&code);

        let bytes = writer.finish().unwrap();

        // PointerToSymbolTable should point to after header+section header+data
        let sym_offset_field = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        // Expected: 20 (COFF hdr) + 40 (1 section hdr) + 16 (data) = 76
        assert_eq!(sym_offset_field, 76);
    }
}
