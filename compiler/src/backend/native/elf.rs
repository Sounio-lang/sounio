//! # Sounio Compiler: ELF Object Writer
//!
//! This module generates ELF (Executable and Linkable Format) object files
//! from machine code segments. Supports ELF64 for x86-64 Linux targets.
//!
//! ## Features
//!
//! - ELF64 object file generation (.o)
//! - Symbol table with visibility control
//! - Relocation entries for external references
//! - Section headers (.text, .data, .bss, .rodata)
//! - Epistemic metadata in custom section (.demetrios.epistemic)
//!
//! ## Usage
//!
//! ```rust,ignore
//! let mut elf = ElfWriter::new(ElfConfig::default());
//! elf.add_text_section(&machine_code);
//! elf.add_symbol("my_func", 0, machine_code.len(), SymbolBinding::Global);
//! let bytes = elf.finish()?;
//! std::fs::write("output.o", bytes)?;
//! ```
//!
//! ## References
//!
//! - System V ABI AMD64 Architecture Processor Supplement
//! - ELF-64 Object File Format, Version 1.5
//! - Linux Standard Base Core Specification
//!
//! ## Author
//!
//! Demetrios Chiuratto Agourakis <demetrios@chiuratto.ai>

use std::collections::HashMap;
use std::io;

// ============================================================================
// ELF CONSTANTS
// ============================================================================

// ELF Magic
const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];

// ELF Class
const ELFCLASS64: u8 = 2;

// ELF Data encoding
const ELFDATA2LSB: u8 = 1;  // Little-endian

// ELF Version
const EV_CURRENT: u8 = 1;

// ELF OS/ABI
const ELFOSABI_SYSV: u8 = 0;
const ELFOSABI_LINUX: u8 = 3;

// ELF Type
const ET_REL: u16 = 1;   // Relocatable file
const ET_EXEC: u16 = 2;  // Executable file
const ET_DYN: u16 = 3;   // Shared object file

// ELF Machine
const EM_X86_64: u16 = 62;

// Section Types
const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHT_REL: u32 = 9;
const SHT_DWARF: u32 = 0x70000001;  // DWARF debug sections

// Section Flags
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_MERGE: u64 = 0x10;
const SHF_STRINGS: u64 = 0x20;
const SHF_INFO_LINK: u64 = 0x40;

// Symbol Binding
const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;

// Symbol Type
const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const STT_SECTION: u8 = 3;
const STT_FILE: u8 = 4;

// Symbol Visibility
const STV_DEFAULT: u8 = 0;
const STV_HIDDEN: u8 = 2;
const STV_PROTECTED: u8 = 3;

// Special Section Indices
const SHN_UNDEF: u16 = 0;
const SHN_ABS: u16 = 0xfff1;

// Relocation Types (x86-64)
const R_X86_64_NONE: u32 = 0;
const R_X86_64_64: u32 = 1;
const R_X86_64_PC32: u32 = 2;
const R_X86_64_GOT32: u32 = 3;
const R_X86_64_PLT32: u32 = 4;
const R_X86_64_GOTPCREL: u32 = 9;
const R_X86_64_32: u32 = 10;
const R_X86_64_32S: u32 = 11;

// ELF64 Header size
const ELF64_EHDR_SIZE: usize = 64;
const ELF64_SHDR_SIZE: usize = 64;
const ELF64_SYM_SIZE: usize = 24;
const ELF64_RELA_SIZE: usize = 24;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// ELF writer configuration
#[derive(Debug, Clone)]
pub struct ElfConfig {
    /// Target OS ABI
    pub os_abi: OsAbi,
    /// Include epistemic metadata section
    pub emit_epistemic_section: bool,
    /// Include debug sections
    pub emit_debug: bool,
    /// Section alignment
    pub section_align: usize,
}

impl Default for ElfConfig {
    fn default() -> Self {
        Self {
            os_abi: OsAbi::Linux,
            emit_epistemic_section: true,
            emit_debug: false,
            section_align: 16,
        }
    }
}

/// Target OS ABI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsAbi {
    SysV,
    Linux,
}

impl OsAbi {
    fn to_elf(&self) -> u8 {
        match self {
            OsAbi::SysV => ELFOSABI_SYSV,
            OsAbi::Linux => ELFOSABI_LINUX,
        }
    }
}

// ============================================================================
// SYMBOL AND RELOCATION TYPES
// ============================================================================

/// Symbol binding type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolBinding {
    Local,
    Global,
    Weak,
}

impl SymbolBinding {
    fn to_elf(&self) -> u8 {
        match self {
            SymbolBinding::Local => STB_LOCAL,
            SymbolBinding::Global => STB_GLOBAL,
            SymbolBinding::Weak => STB_WEAK,
        }
    }
}

/// Symbol type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    NoType,
    Object,
    Function,
    Section,
    File,
}

impl SymbolType {
    fn to_elf(&self) -> u8 {
        match self {
            SymbolType::NoType => STT_NOTYPE,
            SymbolType::Object => STT_OBJECT,
            SymbolType::Function => STT_FUNC,
            SymbolType::Section => STT_SECTION,
            SymbolType::File => STT_FILE,
        }
    }
}

/// Symbol visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolVisibility {
    Default,
    Hidden,
    Protected,
}

impl SymbolVisibility {
    fn to_elf(&self) -> u8 {
        match self {
            SymbolVisibility::Default => STV_DEFAULT,
            SymbolVisibility::Hidden => STV_HIDDEN,
            SymbolVisibility::Protected => STV_PROTECTED,
        }
    }
}

/// Relocation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationType {
    /// 64-bit absolute address
    Abs64,
    /// 32-bit PC-relative
    Pc32,
    /// 32-bit PC-relative PLT entry
    Plt32,
    /// 32-bit GOT-relative
    GotPcRel,
    /// 32-bit absolute
    Abs32,
    /// 32-bit signed absolute
    Abs32S,
}

impl RelocationType {
    fn to_elf(&self) -> u32 {
        match self {
            RelocationType::Abs64 => R_X86_64_64,
            RelocationType::Pc32 => R_X86_64_PC32,
            RelocationType::Plt32 => R_X86_64_PLT32,
            RelocationType::GotPcRel => R_X86_64_GOTPCREL,
            RelocationType::Abs32 => R_X86_64_32,
            RelocationType::Abs32S => R_X86_64_32S,
        }
    }
}

// ============================================================================
// DWARF DEBUG INFO
// ============================================================================

/// DWARF debug information
#[derive(Debug, Clone)]
pub struct DwarfDebugInfo {
    /// Compilation unit information
    pub compilation_unit: String,
    /// Source file paths
    pub source_files: Vec<String>,
    /// Line number information
    pub line_info: Vec<DwarfLineInfo>,
}

impl DwarfDebugInfo {
    pub fn new() -> Self {
        Self {
            compilation_unit: String::new(),
            source_files: Vec::new(),
            line_info: Vec::new(),
        }
    }

    /// Generate .debug_info section
    pub fn generate_debug_info(&self) -> Vec<u8> {
        // TODO: Implement full DWARF .debug_info generation
        // This would include:
        // - Compilation Unit (CU) DIE
        // - Subprogram DIEs for each function
        // - Variable DIEs
        // - Type information
        Vec::new()
    }

    /// Generate .debug_abbrev section
    pub fn generate_debug_abbrev(&self) -> Vec<u8> {
        // TODO: Implement DWARF abbreviation tables
        Vec::new()
    }

    /// Generate .debug_line section
    pub fn generate_debug_line(&self) -> Vec<u8> {
        // TODO: Implement DWARF line number program
        Vec::new()
    }

    /// Generate .debug_str section
    pub fn generate_debug_str(&self) -> Vec<u8> {
        // TODO: Implement DWARF string table
        Vec::new()
    }
}

impl Default for DwarfDebugInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Line number information for DWARF
#[derive(Debug, Clone)]
pub struct DwarfLineInfo {
    pub address: u64,
    pub line: u32,
    pub column: u32,
    pub file_index: u32,
}

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

/// A symbol in the symbol table
#[derive(Debug, Clone)]
struct Symbol {
    name: String,
    name_offset: u32,
    value: u64,
    size: u64,
    binding: SymbolBinding,
    sym_type: SymbolType,
    visibility: SymbolVisibility,
    section_index: u16,
}

/// A relocation entry
#[derive(Debug, Clone)]
struct Relocation {
    offset: u64,
    symbol_index: u32,
    reloc_type: RelocationType,
    addend: i64,
}

/// A section in the ELF file
#[derive(Debug)]
struct Section {
    name: String,
    name_offset: u32,
    section_type: u32,
    flags: u64,
    data: Vec<u8>,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
}

// ============================================================================
// ELF WRITER
// ============================================================================

/// ELF object file writer
pub struct ElfWriter {
    config: ElfConfig,
    sections: Vec<Section>,
    symbols: Vec<Symbol>,
    relocations: HashMap<usize, Vec<Relocation>>,  // section_index -> relocations
    string_table: Vec<u8>,
    section_string_table: Vec<u8>,
    text_section_index: Option<usize>,
    data_section_index: Option<usize>,
    bss_section_index: Option<usize>,
    rodata_section_index: Option<usize>,
    /// DWARF debug info (if enabled)
    dwarf_info: Option<DwarfDebugInfo>,
}

impl ElfWriter {
    /// Create a new ELF writer
    pub fn new(config: ElfConfig) -> Self {
        let mut writer = Self {
            config: config.clone(),
            sections: Vec::new(),
            symbols: Vec::new(),
            relocations: HashMap::new(),
            string_table: vec![0],  // Start with null byte
            section_string_table: vec![0],
            text_section_index: None,
            data_section_index: None,
            bss_section_index: None,
            rodata_section_index: None,
            dwarf_info: if config.emit_debug {
                Some(DwarfDebugInfo::new())
            } else {
                None
            },
        };

        // Add null section (required as first section)
        writer.sections.push(Section {
            name: String::new(),
            name_offset: 0,
            section_type: SHT_NULL,
            flags: 0,
            data: Vec::new(),
            link: 0,
            info: 0,
            addralign: 0,
            entsize: 0,
        });

        writer
    }

    /// Add a .text section with executable code
    pub fn add_text_section(&mut self, code: &[u8]) -> usize {
        let name_offset = self.add_section_string(".text");
        let index = self.sections.len();
        
        self.sections.push(Section {
            name: ".text".to_string(),
            name_offset,
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: code.to_vec(),
            link: 0,
            info: 0,
            addralign: self.config.section_align as u64,
            entsize: 0,
        });

        self.text_section_index = Some(index);
        index
    }

    /// Add a .data section with initialized data
    pub fn add_data_section(&mut self, data: &[u8]) -> usize {
        let name_offset = self.add_section_string(".data");
        let index = self.sections.len();

        self.sections.push(Section {
            name: ".data".to_string(),
            name_offset,
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            data: data.to_vec(),
            link: 0,
            info: 0,
            addralign: self.config.section_align as u64,
            entsize: 0,
        });

        self.data_section_index = Some(index);
        index
    }

    /// Add a .rodata section with read-only data
    pub fn add_rodata_section(&mut self, data: &[u8]) -> usize {
        let name_offset = self.add_section_string(".rodata");
        let index = self.sections.len();

        self.sections.push(Section {
            name: ".rodata".to_string(),
            name_offset,
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC,
            data: data.to_vec(),
            link: 0,
            info: 0,
            addralign: self.config.section_align as u64,
            entsize: 0,
        });

        self.rodata_section_index = Some(index);
        index
    }

    /// Add a .bss section with uninitialized data
    pub fn add_bss_section(&mut self, size: usize) -> usize {
        let name_offset = self.add_section_string(".bss");
        let index = self.sections.len();

        self.sections.push(Section {
            name: ".bss".to_string(),
            name_offset,
            section_type: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            data: vec![0; size],  // Size tracked, not actual data
            link: 0,
            info: 0,
            addralign: self.config.section_align as u64,
            entsize: 0,
        });

        self.bss_section_index = Some(index);
        index
    }

    /// Add epistemic metadata section (custom Demetrios section)
    pub fn add_epistemic_section(&mut self, metadata: &EpistemicSectionData) -> usize {
        if !self.config.emit_epistemic_section {
            return 0;
        }

        let name_offset = self.add_section_string(".demetrios.epistemic");
        let index = self.sections.len();

        // Serialize metadata
        let data = metadata.serialize();

        self.sections.push(Section {
            name: ".demetrios.epistemic".to_string(),
            name_offset,
            section_type: SHT_PROGBITS,
            flags: 0,  // Not loaded into memory
            data,
            link: 0,
            info: 0,
            addralign: 8,
            entsize: 0,
        });

        index
    }

    /// Add a symbol to the symbol table
    pub fn add_symbol(
        &mut self,
        name: &str,
        value: u64,
        size: u64,
        binding: SymbolBinding,
        sym_type: SymbolType,
        section_index: u16,
    ) -> usize {
        let name_offset = self.add_string(name);
        let index = self.symbols.len();

        self.symbols.push(Symbol {
            name: name.to_string(),
            name_offset,
            value,
            size,
            binding,
            sym_type,
            visibility: SymbolVisibility::Default,
            section_index,
        });

        index
    }

    /// Add a function symbol
    pub fn add_function(&mut self, name: &str, offset: u64, size: u64, is_global: bool) -> usize {
        let section_index = self.text_section_index.unwrap_or(1) as u16;
        let binding = if is_global { SymbolBinding::Global } else { SymbolBinding::Local };
        self.add_symbol(name, offset, size, binding, SymbolType::Function, section_index)
    }

    /// Add a data symbol
    pub fn add_data_symbol(&mut self, name: &str, offset: u64, size: u64, is_global: bool) -> usize {
        let section_index = self.data_section_index.unwrap_or(2) as u16;
        let binding = if is_global { SymbolBinding::Global } else { SymbolBinding::Local };
        self.add_symbol(name, offset, size, binding, SymbolType::Object, section_index)
    }

    /// Add an undefined symbol (external reference)
    pub fn add_undefined_symbol(&mut self, name: &str) -> usize {
        self.add_symbol(name, 0, 0, SymbolBinding::Global, SymbolType::NoType, SHN_UNDEF)
    }

    /// Add a relocation entry
    pub fn add_relocation(
        &mut self,
        section_index: usize,
        offset: u64,
        symbol_index: usize,
        reloc_type: RelocationType,
        addend: i64,
    ) {
        let relocs = self.relocations.entry(section_index).or_default();
        relocs.push(Relocation {
            offset,
            symbol_index: symbol_index as u32,
            reloc_type,
            addend,
        });
    }

    /// Add a debug section from DWARF output
    ///
    /// Creates ELF sections for DWARF debug information (.debug_info, .debug_abbrev, etc.)
    pub fn add_debug_section(&mut self, section_name: &str, data: Vec<u8>) -> usize {
        if data.is_empty() {
            return 0; // Don't add empty sections
        }
        
        let name_offset = self.add_section_string(section_name);
        let index = self.sections.len();
        
        self.sections.push(Section {
            name: section_name.to_string(),
            name_offset,
            section_type: SHT_PROGBITS, // Debug sections are PROGBITS
            flags: 0, // Debug sections are not loaded into memory
            data,
            link: 0,
            info: 0,
            addralign: 1,
            entsize: 0,
        });
        
        index
    }
    
    /// Add multiple debug sections from DwarfOutput
    ///
    /// Convenience method to add all DWARF sections at once from a DwarfOutput struct.
    /// This should be called after code/data sections but before finish().
    pub fn add_dwarf_sections(&mut self, dwarf: &crate::codegen::debug::DwarfOutput) {
        if !dwarf.debug_info.is_empty() {
            self.add_debug_section(".debug_info", dwarf.debug_info.clone());
        }
        if !dwarf.debug_abbrev.is_empty() {
            self.add_debug_section(".debug_abbrev", dwarf.debug_abbrev.clone());
        }
        if !dwarf.debug_line.is_empty() {
            self.add_debug_section(".debug_line", dwarf.debug_line.clone());
        }
        if !dwarf.debug_str.is_empty() {
            self.add_debug_section(".debug_str", dwarf.debug_str.clone());
        }
        if !dwarf.debug_loc.is_empty() {
            self.add_debug_section(".debug_loc", dwarf.debug_loc.clone());
        }
        if !dwarf.debug_loclists.is_empty() {
            self.add_debug_section(".debug_loclists", dwarf.debug_loclists.clone());
        }
        if !dwarf.debug_ranges.is_empty() {
            self.add_debug_section(".debug_ranges", dwarf.debug_ranges.clone());
        }
        if !dwarf.debug_rnglists.is_empty() {
            self.add_debug_section(".debug_rnglists", dwarf.debug_rnglists.clone());
        }
        if !dwarf.debug_frame.is_empty() {
            self.add_debug_section(".debug_frame", dwarf.debug_frame.clone());
        }
    }
    /// Finalize and generate the ELF file bytes
    pub fn finish(mut self) -> io::Result<Vec<u8>> {
        // Sort symbols: locals first, then globals. Remap relocations to keep indices valid.
        let mut order: Vec<usize> = (0..self.symbols.len()).collect();
        order.sort_by_key(|&idx| {
            let binding = match self.symbols[idx].binding {
                SymbolBinding::Local => 0usize,
                _ => 1usize,
            };
            (binding, idx)
        });

        if !order.is_empty() {
            let mut remap = vec![0usize; self.symbols.len()];
            let mut new_symbols = Vec::with_capacity(self.symbols.len());
            for (new_idx, &old_idx) in order.iter().enumerate() {
                remap[old_idx] = new_idx;
                new_symbols.push(self.symbols[old_idx].clone());
            }
            self.symbols = new_symbols;

            for relocs in self.relocations.values_mut() {
                for reloc in relocs.iter_mut() {
                    if reloc.symbol_index == 0 {
                        continue;
                    }
                    let old_idx = (reloc.symbol_index - 1) as usize;
                    if old_idx < remap.len() {
                        reloc.symbol_index = (remap[old_idx] + 1) as u32;
                    }
                }
            }
        }

        // Find first global symbol index
        let first_global = self.symbols.iter()
            .position(|s| s.binding != SymbolBinding::Local)
            .unwrap_or(self.symbols.len());

        // Add .note.GNU-stack section (indicates non-executable stack)
        // This suppresses the ld warning about missing section
        let gnu_stack_name_offset = self.add_section_string(".note.GNU-stack");
        self.sections.push(Section {
            name: ".note.GNU-stack".to_string(),
            name_offset: gnu_stack_name_offset,
            section_type: SHT_PROGBITS,
            flags: 0,  // No flags = non-executable stack
            data: Vec::new(),
            link: 0,
            info: 0,
            addralign: 1,
            entsize: 0,
        });

        // Add string table section
        let strtab_name_offset = self.add_section_string(".strtab");
        let strtab_index = self.sections.len();
        self.sections.push(Section {
            name: ".strtab".to_string(),
            name_offset: strtab_name_offset,
            section_type: SHT_STRTAB,
            flags: 0,
            data: self.string_table.clone(),
            link: 0,
            info: 0,
            addralign: 1,
            entsize: 0,
        });

        // Add symbol table section
        let symtab_name_offset = self.add_section_string(".symtab");
        let symtab_index = self.sections.len();
        let symtab_data = self.build_symtab();
        // sh_info is one greater than the index of the last local symbol
        // Symbol table has null symbol at index 0, so add 1 to account for it
        let sh_info = (first_global + 1) as u32;
        self.sections.push(Section {
            name: ".symtab".to_string(),
            name_offset: symtab_name_offset,
            section_type: SHT_SYMTAB,
            flags: 0,
            data: symtab_data,
            link: strtab_index as u32,
            info: sh_info,
            addralign: 8,
            entsize: ELF64_SYM_SIZE as u64,
        });

        // Add relocation sections
        let reloc_sections: Vec<_> = self.relocations.iter()
            .map(|(&sec_idx, relocs)| (sec_idx, relocs.clone()))
            .collect();

        for (section_index, relocs) in reloc_sections {
            let target_name = &self.sections[section_index].name;
            let rela_name = format!(".rela{}", target_name);
            let rela_name_offset = self.add_section_string(&rela_name);
            let rela_data = self.build_rela(&relocs);

            self.sections.push(Section {
                name: rela_name,
                name_offset: rela_name_offset,
                section_type: SHT_RELA,
                flags: SHF_INFO_LINK,
                data: rela_data,
                link: symtab_index as u32,
                info: section_index as u32,
                addralign: 8,
                entsize: ELF64_RELA_SIZE as u64,
            });
        }

        // Add section header string table (must be last)
        let shstrtab_name_offset = self.add_section_string(".shstrtab");
        let shstrtab_index = self.sections.len();
        self.sections.push(Section {
            name: ".shstrtab".to_string(),
            name_offset: shstrtab_name_offset,
            section_type: SHT_STRTAB,
            flags: 0,
            data: self.section_string_table.clone(),
            link: 0,
            info: 0,
            addralign: 1,
            entsize: 0,
        });

        // Calculate offsets
        let mut offset = ELF64_EHDR_SIZE;

        // Section data follows header
        let mut section_offsets = Vec::new();
        for section in &self.sections {
            if section.section_type == SHT_NULL || section.section_type == SHT_NOBITS {
                section_offsets.push(0);
            } else {
                // Align offset
                let align = section.addralign.max(1) as usize;
                offset = (offset + align - 1) & !(align - 1);
                section_offsets.push(offset);
                offset += section.data.len();
            }
        }

        // Section headers follow section data
        let shoff = (offset + 7) & !7;  // 8-byte align

        // Build output
        let mut output = Vec::with_capacity(shoff + self.sections.len() * ELF64_SHDR_SIZE);

        // Write ELF header
        self.write_elf_header(&mut output, shoff, shstrtab_index)?;

        // Write section data
        for (i, section) in self.sections.iter().enumerate() {
            if section.section_type == SHT_NULL || section.section_type == SHT_NOBITS {
                continue;
            }

            // Pad to alignment
            let target_offset = section_offsets[i];
            while output.len() < target_offset {
                output.push(0);
            }

            output.extend_from_slice(&section.data);
        }

        // Pad to section header offset
        while output.len() < shoff {
            output.push(0);
        }

        // Write section headers
        for (i, section) in self.sections.iter().enumerate() {
            self.write_section_header(&mut output, section, section_offsets[i])?;
        }

        Ok(output)
    }

    // ========== Private helpers ==========

    fn add_string(&mut self, s: &str) -> u32 {
        let offset = self.string_table.len() as u32;
        self.string_table.extend_from_slice(s.as_bytes());
        self.string_table.push(0);
        offset
    }

    fn add_section_string(&mut self, s: &str) -> u32 {
        let offset = self.section_string_table.len() as u32;
        self.section_string_table.extend_from_slice(s.as_bytes());
        self.section_string_table.push(0);
        offset
    }

    fn build_symtab(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity((self.symbols.len() + 1) * ELF64_SYM_SIZE);

        // First entry is null
        data.extend_from_slice(&[0u8; ELF64_SYM_SIZE]);

        // Symbol entries
        for sym in &self.symbols {
            // st_name (4 bytes)
            data.extend_from_slice(&sym.name_offset.to_le_bytes());
            // st_info (1 byte): binding << 4 | type
            let info = (sym.binding.to_elf() << 4) | sym.sym_type.to_elf();
            data.push(info);
            // st_other (1 byte): visibility
            data.push(sym.visibility.to_elf());
            // st_shndx (2 bytes)
            data.extend_from_slice(&sym.section_index.to_le_bytes());
            // st_value (8 bytes)
            data.extend_from_slice(&sym.value.to_le_bytes());
            // st_size (8 bytes)
            data.extend_from_slice(&sym.size.to_le_bytes());
        }

        data
    }

    fn build_rela(&self, relocs: &[Relocation]) -> Vec<u8> {
        let mut data = Vec::with_capacity(relocs.len() * ELF64_RELA_SIZE);

        for reloc in relocs {
            // r_offset (8 bytes)
            data.extend_from_slice(&reloc.offset.to_le_bytes());
            // r_info (8 bytes): symbol << 32 | type
            let info = ((reloc.symbol_index as u64) << 32) | (reloc.reloc_type.to_elf() as u64);
            data.extend_from_slice(&info.to_le_bytes());
            // r_addend (8 bytes)
            data.extend_from_slice(&reloc.addend.to_le_bytes());
        }

        data
    }

    fn write_elf_header(&self, w: &mut Vec<u8>, shoff: usize, shstrndx: usize) -> io::Result<()> {
        // e_ident (16 bytes)
        w.extend_from_slice(&ELF_MAGIC);
        w.push(ELFCLASS64);
        w.push(ELFDATA2LSB);
        w.push(EV_CURRENT);
        w.push(self.config.os_abi.to_elf());
        w.extend_from_slice(&[0u8; 8]);  // Padding

        // e_type (2 bytes)
        w.extend_from_slice(&ET_REL.to_le_bytes());
        // e_machine (2 bytes)
        w.extend_from_slice(&EM_X86_64.to_le_bytes());
        // e_version (4 bytes)
        w.extend_from_slice(&(EV_CURRENT as u32).to_le_bytes());
        // e_entry (8 bytes) - 0 for relocatable
        w.extend_from_slice(&0u64.to_le_bytes());
        // e_phoff (8 bytes) - 0 for relocatable
        w.extend_from_slice(&0u64.to_le_bytes());
        // e_shoff (8 bytes)
        w.extend_from_slice(&(shoff as u64).to_le_bytes());
        // e_flags (4 bytes)
        w.extend_from_slice(&0u32.to_le_bytes());
        // e_ehsize (2 bytes)
        w.extend_from_slice(&(ELF64_EHDR_SIZE as u16).to_le_bytes());
        // e_phentsize (2 bytes)
        w.extend_from_slice(&0u16.to_le_bytes());
        // e_phnum (2 bytes)
        w.extend_from_slice(&0u16.to_le_bytes());
        // e_shentsize (2 bytes)
        w.extend_from_slice(&(ELF64_SHDR_SIZE as u16).to_le_bytes());
        // e_shnum (2 bytes)
        w.extend_from_slice(&(self.sections.len() as u16).to_le_bytes());
        // e_shstrndx (2 bytes)
        w.extend_from_slice(&(shstrndx as u16).to_le_bytes());

        Ok(())
    }

    fn write_section_header(&self, w: &mut Vec<u8>, section: &Section, offset: usize) -> io::Result<()> {
        // sh_name (4 bytes)
        w.extend_from_slice(&section.name_offset.to_le_bytes());
        // sh_type (4 bytes)
        w.extend_from_slice(&section.section_type.to_le_bytes());
        // sh_flags (8 bytes)
        w.extend_from_slice(&section.flags.to_le_bytes());
        // sh_addr (8 bytes) - 0 for relocatable
        w.extend_from_slice(&0u64.to_le_bytes());
        // sh_offset (8 bytes)
        let file_offset = if section.section_type == SHT_NOBITS { 0 } else { offset };
        w.extend_from_slice(&(file_offset as u64).to_le_bytes());
        // sh_size (8 bytes)
        w.extend_from_slice(&(section.data.len() as u64).to_le_bytes());
        // sh_link (4 bytes)
        w.extend_from_slice(&section.link.to_le_bytes());
        // sh_info (4 bytes)
        w.extend_from_slice(&section.info.to_le_bytes());
        // sh_addralign (8 bytes)
        w.extend_from_slice(&section.addralign.to_le_bytes());
        // sh_entsize (8 bytes)
        w.extend_from_slice(&section.entsize.to_le_bytes());

        Ok(())
    }
}

// ============================================================================
// EPISTEMIC SECTION DATA
// ============================================================================

/// Data for the .demetrios.epistemic section
#[derive(Debug, Clone, Default)]
pub struct EpistemicSectionData {
    /// Function epistemic metadata
    pub functions: Vec<FunctionEpistemic>,
    /// Value-level epistemic metadata (per-value confidence, uncertainty, provenance)
    pub values: Vec<ValueEpistemic>,
    /// Global confidence floor
    pub confidence_floor: f64,
    /// Thermal degradation applied
    pub thermal_degradation: f64,
    /// Allocation statistics
    pub alloc_stats: Option<AllocStats>,
}

/// Epistemic metadata for a function
#[derive(Debug, Clone)]
pub struct FunctionEpistemic {
    pub name: String,
    pub confidence: f64,
    pub delta: f64,
    pub spill_count: u32,
    pub critical: bool,
}

/// Epistemic metadata for a value (Knowledge<T>)
/// This provides complete epistemic tracking per value for analysis/debugging
#[derive(Debug, Clone)]
pub struct ValueEpistemic {
    /// Value identifier (instruction index or value ID)
    pub value_id: u32,
    /// Function this value belongs to
    pub function_name: String,
    /// Confidence (ε) [0.0, 1.0]
    pub confidence: f64,
    /// Uncertainty (σ) [0.0, 1.0]
    pub uncertainty: f64,
    /// Provenance ID (tracks data lineage)
    pub provenance_id: Option<u32>,
    /// Number of times this value was spilled
    pub spill_count: u32,
    /// Whether this value has tracked provenance
    pub has_provenance: bool,
    /// Whether this value is critical (high confidence, should not be spilled)
    pub critical: bool,
}

/// Allocation statistics summary
#[derive(Debug, Clone)]
pub struct AllocStats {
    pub intervals_allocated: u32,
    pub intervals_spilled: u32,
    pub avg_confidence_allocated: f64,
    pub avg_confidence_spilled: f64,
}

impl EpistemicSectionData {
    /// Serialize to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic: "DMEP" (Demetrios EPistemic)
        data.extend_from_slice(b"DMEP");

        // Version (2 bytes) - increment to 2 for expanded metadata
        data.extend_from_slice(&2u16.to_le_bytes());

        // Number of functions (2 bytes)
        data.extend_from_slice(&(self.functions.len() as u16).to_le_bytes());
        
        // Number of values (4 bytes)
        data.extend_from_slice(&(self.values.len() as u32).to_le_bytes());

        // Confidence floor (8 bytes)
        data.extend_from_slice(&self.confidence_floor.to_le_bytes());

        // Thermal degradation (8 bytes)
        data.extend_from_slice(&self.thermal_degradation.to_le_bytes());

        // Function entries
        for func in &self.functions {
            // Name length + name
            let name_bytes = func.name.as_bytes();
            data.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            data.extend_from_slice(name_bytes);

            // Confidence (8 bytes)
            data.extend_from_slice(&func.confidence.to_le_bytes());
            // Delta (8 bytes)
            data.extend_from_slice(&func.delta.to_le_bytes());
            // Spill count (4 bytes)
            data.extend_from_slice(&func.spill_count.to_le_bytes());
            // Flags (1 byte)
            data.push(if func.critical { 1 } else { 0 });
        }

        // Value-level epistemic metadata
        for value in &self.values {
            // Value ID (4 bytes)
            data.extend_from_slice(&value.value_id.to_le_bytes());
            
            // Function name length + name
            let func_name_bytes = value.function_name.as_bytes();
            data.extend_from_slice(&(func_name_bytes.len() as u16).to_le_bytes());
            data.extend_from_slice(func_name_bytes);
            
            // Confidence (8 bytes)
            data.extend_from_slice(&value.confidence.to_le_bytes());
            // Uncertainty (8 bytes)
            data.extend_from_slice(&value.uncertainty.to_le_bytes());
            
            // Provenance ID (4 bytes, or 0xFFFFFFFF if None)
            let prov_id = value.provenance_id.unwrap_or(0xFFFFFFFF);
            data.extend_from_slice(&prov_id.to_le_bytes());
            
            // Spill count (4 bytes)
            data.extend_from_slice(&value.spill_count.to_le_bytes());
            
            // Flags (1 byte): bit 0 = has_provenance, bit 1 = critical
            let mut flags = 0u8;
            if value.has_provenance {
                flags |= 0x01;
            }
            if value.critical {
                flags |= 0x02;
            }
            data.push(flags);
        }

        // Allocation stats (optional)
        if let Some(stats) = &self.alloc_stats {
            data.push(1);  // Has stats
            data.extend_from_slice(&stats.intervals_allocated.to_le_bytes());
            data.extend_from_slice(&stats.intervals_spilled.to_le_bytes());
            data.extend_from_slice(&stats.avg_confidence_allocated.to_le_bytes());
            data.extend_from_slice(&stats.avg_confidence_spilled.to_le_bytes());
        } else {
            data.push(0);  // No stats
        }

        data
    }
}

// ============================================================================
// CODE SEGMENT CONVERSION
// ============================================================================

/// Convert CodeSegment from sir::emit to ELF object file
pub fn code_segment_to_elf(segment: &crate::sir::emit::CodeSegment) -> Result<Vec<u8>, String> {
    let mut elf = ElfWriter::new(ElfConfig::default());
    
    // Add .text section with code
    let text_section_idx = elf.add_text_section(&segment.code);
    
    // Add symbols and build symbol index map for relocations
    // Note: ELF has a null symbol at index 0, so defined symbols start at index 1
    let mut symbol_indices = HashMap::new();
    for sym in &segment.symbols {
        let size = calculate_symbol_size(sym, &segment.symbols, segment.code.len());
        let elf_idx = elf.add_function(
            &sym.name,
            sym.offset as u64,
            size,
            sym.global,
        );
        // add_function returns 0-based index, but ELF symtab has null at 0, so add 1
        symbol_indices.insert(sym.name.clone(), elf_idx + 1);
    }
    
    // Add undefined symbols for relocations
    for reloc in &segment.relocations {
        if !symbol_indices.contains_key(&reloc.symbol) {
            let idx = elf.add_undefined_symbol(&reloc.symbol);
            // add_undefined_symbol returns 0-based index, add 1 for null symbol
            symbol_indices.insert(reloc.symbol.clone(), idx + 1);
        }
    }
    
    // Add relocations
    for reloc in &segment.relocations {
        let sym_idx = symbol_indices.get(&reloc.symbol)
            .copied()
            .ok_or_else(|| format!("Symbol not found for relocation: {}", reloc.symbol))?;
        let reloc_type = reloc_kind_to_type(&reloc.kind);

        elf.add_relocation(
            text_section_idx,
            reloc.offset as u64,
            sym_idx,
            reloc_type,
            reloc.addend,
        );
    }
    
    elf.finish().map_err(|e| format!("ELF generation failed: {}", e))
}

/// Calculate symbol size from next symbol or end of code
fn calculate_symbol_size(
    sym: &crate::sir::emit::Symbol,
    all_symbols: &[crate::sir::emit::Symbol],
    code_len: usize,
) -> u64 {
    // Find next symbol in same section (sorted by offset)
    let mut next_offset = code_len;
    
    for other in all_symbols {
        if other.offset > sym.offset && other.offset < next_offset {
            next_offset = other.offset;
        }
    }
    
    (next_offset - sym.offset) as u64
}

/// Convert sir::emit::RelocKind to elf::RelocationType
fn reloc_kind_to_type(kind: &crate::sir::emit::RelocKind) -> RelocationType {
    match kind {
        crate::sir::emit::RelocKind::Abs64 => RelocationType::Abs64,
        crate::sir::emit::RelocKind::PCRel32 => RelocationType::Pc32,
        crate::sir::emit::RelocKind::PLT32 => RelocationType::Plt32,
        crate::sir::emit::RelocKind::GOT32 => RelocationType::GotPcRel,
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_elf() {
        let mut elf = ElfWriter::new(ElfConfig::default());

        // Add empty text section
        elf.add_text_section(&[]);

        let bytes = elf.finish().unwrap();

        // Check ELF magic
        assert_eq!(&bytes[0..4], &ELF_MAGIC);
        // Check class (ELF64)
        assert_eq!(bytes[4], ELFCLASS64);
        // Check type (relocatable)
        assert_eq!(u16::from_le_bytes([bytes[16], bytes[17]]), ET_REL);
    }

    #[test]
    fn test_elf_with_code() {
        let mut elf = ElfWriter::new(ElfConfig::default());

        // Simple x86-64 code: ret
        let code = vec![0xc3];  // retq
        elf.add_text_section(&code);
        elf.add_function("test_func", 0, 1, true);

        let bytes = elf.finish().unwrap();
        assert!(bytes.len() > ELF64_EHDR_SIZE);

        // Verify the code is somewhere in the file
        assert!(bytes.windows(1).any(|w| w == [0xc3]));
    }

    #[test]
    fn test_elf_with_symbol() {
        let mut elf = ElfWriter::new(ElfConfig::default());

        let code = vec![0x90, 0x90, 0xc3];  // nop, nop, ret
        elf.add_text_section(&code);
        elf.add_function("my_function", 0, 3, true);

        let bytes = elf.finish().unwrap();

        // Symbol name should be in the file
        let has_name = bytes.windows(11).any(|w| w == b"my_function");
        assert!(has_name);
    }

    #[test]
    fn test_code_segment_conversion() {
        use crate::sir::emit::{CodeSegment, Symbol};
        
        let segment = CodeSegment {
            code: vec![0xc3],  // ret
            symbols: vec![
                Symbol {
                    name: "test".to_string(),
                    offset: 0,
                    global: true,
                },
            ],
            relocations: vec![],
        };
        
        let elf_bytes = code_segment_to_elf(&segment).unwrap();
        assert_eq!(&elf_bytes[0..4], &ELF_MAGIC);
    }
}

// ============================================================================
// MODULE EXPORTS
// ============================================================================

pub mod prelude {
    pub use super::{
        ElfWriter,
        ElfConfig,
        OsAbi,
        SymbolBinding,
        SymbolType,
        SymbolVisibility,
        RelocationType,
        EpistemicSectionData,
        FunctionEpistemic,
        AllocStats,
        code_segment_to_elf,
    };
}
