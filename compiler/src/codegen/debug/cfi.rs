//! Call Frame Information (CFI) generation for DWARF .debug_frame section
//!
//! CFI describes how to restore the call frame at any point during program
//! execution. This is essential for:
//! - Stack unwinding during exceptions
//! - Debugger stack traces
//! - Profiling and crash analysis
//!
//! This module generates DWARF 5 compliant .debug_frame sections.

use std::collections::HashMap;

/// Call Frame Information builder
pub struct CfiBuilder {
    /// Common Information Entries (CIE)
    cies: Vec<Cie>,

    /// Frame Description Entries (FDE)
    fdes: Vec<Fde>,

    /// Current CIE index for new FDEs
    current_cie: usize,
}

/// Common Information Entry
///
/// Contains information shared by multiple FDEs, like the return address
/// register and initial CFI instructions.
#[derive(Debug, Clone)]
pub struct Cie {
    /// CIE identifier (0xFFFFFFFF for DWARF 5 32-bit)
    pub id: u32,

    /// DWARF version
    pub version: u8,

    /// Augmentation string (empty for basic CFI)
    pub augmentation: String,

    /// Address size in bytes
    pub address_size: u8,

    /// Segment selector size
    pub segment_size: u8,

    /// Code alignment factor (usually 1)
    pub code_alignment_factor: u64,

    /// Data alignment factor (usually -8 for x86-64)
    pub data_alignment_factor: i64,

    /// Return address register
    pub return_address_register: u64,

    /// Initial instructions
    pub initial_instructions: Vec<CfaInstruction>,
}

/// Frame Description Entry
///
/// Describes the CFI for a specific address range (typically a function).
#[derive(Debug, Clone)]
pub struct Fde {
    /// Pointer to the CIE this FDE uses
    pub cie_index: usize,

    /// Initial location (function start address)
    pub initial_location: u64,

    /// Address range length
    pub address_range: u64,

    /// CFI instructions for this function
    pub instructions: Vec<CfaInstruction>,
}

/// CFI instruction opcodes
#[derive(Debug, Clone)]
pub enum CfaInstruction {
    // High-frequency instructions (encoded in opcode)
    /// DW_CFA_advance_loc: advance by delta * code_align
    AdvanceLoc { delta: u8 },

    /// DW_CFA_offset: reg at CFA + offset * data_align
    Offset { register: u8, offset: u64 },

    /// DW_CFA_restore: restore reg to initial rule
    Restore { register: u8 },

    // Primary instructions
    /// DW_CFA_nop: no operation
    Nop,

    /// DW_CFA_set_loc: set location to address
    SetLoc { address: u64 },

    /// DW_CFA_advance_loc1: advance by 1-byte delta
    AdvanceLoc1 { delta: u8 },

    /// DW_CFA_advance_loc2: advance by 2-byte delta
    AdvanceLoc2 { delta: u16 },

    /// DW_CFA_advance_loc4: advance by 4-byte delta
    AdvanceLoc4 { delta: u32 },

    /// DW_CFA_offset_extended: extended offset
    OffsetExtended { register: u64, offset: u64 },

    /// DW_CFA_restore_extended: extended restore
    RestoreExtended { register: u64 },

    /// DW_CFA_undefined: register has no value
    Undefined { register: u64 },

    /// DW_CFA_same_value: register unchanged
    SameValue { register: u64 },

    /// DW_CFA_register: register saved in another register
    Register { target: u64, source: u64 },

    /// DW_CFA_remember_state: push state onto stack
    RememberState,

    /// DW_CFA_restore_state: pop state from stack
    RestoreState,

    /// DW_CFA_def_cfa: define CFA as reg + offset
    DefCfa { register: u64, offset: u64 },

    /// DW_CFA_def_cfa_register: set CFA register
    DefCfaRegister { register: u64 },

    /// DW_CFA_def_cfa_offset: set CFA offset
    DefCfaOffset { offset: u64 },

    /// DW_CFA_def_cfa_expression: CFA from expression
    DefCfaExpression { expression: Vec<u8> },

    /// DW_CFA_expression: register from expression
    Expression { register: u64, expression: Vec<u8> },

    /// DW_CFA_offset_extended_sf: signed factored offset
    OffsetExtendedSf { register: u64, offset: i64 },

    /// DW_CFA_def_cfa_sf: signed factored CFA
    DefCfaSf { register: u64, offset: i64 },

    /// DW_CFA_def_cfa_offset_sf: signed factored CFA offset
    DefCfaOffsetSf { offset: i64 },

    /// DW_CFA_val_offset: value at CFA + offset
    ValOffset { register: u64, offset: u64 },

    /// DW_CFA_val_offset_sf: signed value at CFA + offset
    ValOffsetSf { register: u64, offset: i64 },

    /// DW_CFA_val_expression: value from expression
    ValExpression { register: u64, expression: Vec<u8> },
}

/// Complete call frame information for a program
#[derive(Debug, Clone)]
pub struct CallFrameInfo {
    /// All CIEs
    pub cies: Vec<Cie>,

    /// All FDEs
    pub fdes: Vec<Fde>,
}

// x86-64 register numbers for DWARF
pub mod x86_64 {
    pub const RAX: u64 = 0;
    pub const RDX: u64 = 1;
    pub const RCX: u64 = 2;
    pub const RBX: u64 = 3;
    pub const RSI: u64 = 4;
    pub const RDI: u64 = 5;
    pub const RBP: u64 = 6;
    pub const RSP: u64 = 7;
    pub const R8: u64 = 8;
    pub const R9: u64 = 9;
    pub const R10: u64 = 10;
    pub const R11: u64 = 11;
    pub const R12: u64 = 12;
    pub const R13: u64 = 13;
    pub const R14: u64 = 14;
    pub const R15: u64 = 15;
    pub const RA: u64 = 16; // Return address (RIP)
}

// AArch64 register numbers for DWARF
pub mod aarch64 {
    pub const X0: u64 = 0;
    pub const X29: u64 = 29; // Frame pointer
    pub const X30: u64 = 30; // Link register (return address)
    pub const SP: u64 = 31;
}

impl CfiBuilder {
    /// Create a new CFI builder with a default x86-64 CIE
    pub fn new_x86_64() -> Self {
        let default_cie = Cie {
            id: 0xFFFF_FFFF,
            version: 4, // DWARF 4 for wider compatibility
            augmentation: String::new(),
            address_size: 8,
            segment_size: 0,
            code_alignment_factor: 1,
            data_alignment_factor: -8, // Stack grows down
            return_address_register: x86_64::RA,
            initial_instructions: vec![
                // CFA is RSP + 8 (return address pushed)
                CfaInstruction::DefCfa {
                    register: x86_64::RSP,
                    offset: 8,
                },
                // Return address is at CFA - 8
                CfaInstruction::Offset {
                    register: x86_64::RA as u8,
                    offset: 1, // * data_alignment (-8) = -8
                },
            ],
        };

        CfiBuilder {
            cies: vec![default_cie],
            fdes: Vec::new(),
            current_cie: 0,
        }
    }

    /// Create a new CFI builder with a default AArch64 CIE
    pub fn new_aarch64() -> Self {
        let default_cie = Cie {
            id: 0xFFFF_FFFF,
            version: 4,
            augmentation: String::new(),
            address_size: 8,
            segment_size: 0,
            code_alignment_factor: 4, // 4-byte instruction alignment
            data_alignment_factor: -8,
            return_address_register: aarch64::X30,
            initial_instructions: vec![
                CfaInstruction::DefCfa {
                    register: aarch64::SP,
                    offset: 0,
                },
            ],
        };

        CfiBuilder {
            cies: vec![default_cie],
            fdes: Vec::new(),
            current_cie: 0,
        }
    }

    /// Begin a new FDE for a function
    pub fn begin_fde(&mut self, start_address: u64, length: u64) -> usize {
        let idx = self.fdes.len();
        self.fdes.push(Fde {
            cie_index: self.current_cie,
            initial_location: start_address,
            address_range: length,
            instructions: Vec::new(),
        });
        idx
    }

    /// Add instruction to current FDE
    pub fn add_instruction(&mut self, fde_index: usize, instruction: CfaInstruction) {
        if let Some(fde) = self.fdes.get_mut(fde_index) {
            fde.instructions.push(instruction);
        }
    }

    /// Emit standard x86-64 function prologue CFI
    ///
    /// Standard prologue:
    /// ```asm
    /// push rbp        ; save frame pointer
    /// mov rbp, rsp    ; set new frame pointer
    /// sub rsp, N      ; allocate local variables
    /// ```
    pub fn emit_cfi_prologue(&mut self, fde_index: usize, frame_size: u64) {
        // After push rbp: CFA is now RSP + 16, RBP saved at CFA - 16
        self.add_instruction(fde_index, CfaInstruction::AdvanceLoc { delta: 1 });
        self.add_instruction(
            fde_index,
            CfaInstruction::DefCfaOffset { offset: 16 },
        );
        self.add_instruction(
            fde_index,
            CfaInstruction::Offset {
                register: x86_64::RBP as u8,
                offset: 2, // * -8 = -16
            },
        );

        // After mov rbp, rsp: CFA is now RBP + 16
        self.add_instruction(fde_index, CfaInstruction::AdvanceLoc { delta: 3 });
        self.add_instruction(
            fde_index,
            CfaInstruction::DefCfaRegister {
                register: x86_64::RBP,
            },
        );

        // After sub rsp, N: no change to CFA rules (still RBP + 16)
        if frame_size > 0 {
            let advance = ((frame_size / 8) + 1) as u8;
            self.add_instruction(fde_index, CfaInstruction::AdvanceLoc { delta: advance.min(63) });
        }
    }

    /// Emit standard x86-64 function epilogue CFI
    ///
    /// Standard epilogue:
    /// ```asm
    /// mov rsp, rbp    ; deallocate locals
    /// pop rbp         ; restore frame pointer
    /// ret             ; return
    /// ```
    pub fn emit_cfi_epilogue(&mut self, fde_index: usize) {
        // Restore all registers to their entry state
        self.add_instruction(fde_index, CfaInstruction::RestoreState);
    }

    /// Save a callee-saved register
    pub fn emit_save_register(&mut self, fde_index: usize, register: u64, offset: u64) {
        self.add_instruction(
            fde_index,
            CfaInstruction::Offset {
                register: register as u8,
                offset,
            },
        );
    }

    /// Mark a register as restored
    pub fn emit_restore_register(&mut self, fde_index: usize, register: u64) {
        if register < 64 {
            self.add_instruction(
                fde_index,
                CfaInstruction::Restore {
                    register: register as u8,
                },
            );
        } else {
            self.add_instruction(
                fde_index,
                CfaInstruction::RestoreExtended { register },
            );
        }
    }

    /// Remember current CFI state (for branching)
    pub fn emit_remember_state(&mut self, fde_index: usize) {
        self.add_instruction(fde_index, CfaInstruction::RememberState);
    }

    /// Restore CFI state
    pub fn emit_restore_state(&mut self, fde_index: usize) {
        self.add_instruction(fde_index, CfaInstruction::RestoreState);
    }

    /// Build the final CFI
    pub fn build(self) -> CallFrameInfo {
        CallFrameInfo {
            cies: self.cies,
            fdes: self.fdes,
        }
    }
}

impl CallFrameInfo {
    /// Generate the .debug_frame section bytes
    pub fn generate_debug_frame(&self) -> Vec<u8> {
        let mut frame = Vec::new();
        let mut cie_offsets: HashMap<usize, u32> = HashMap::new();

        // Generate all CIEs first
        for (i, cie) in self.cies.iter().enumerate() {
            cie_offsets.insert(i, frame.len() as u32);
            self.emit_cie(&mut frame, cie);
        }

        // Generate all FDEs
        for fde in &self.fdes {
            let cie_offset = cie_offsets.get(&fde.cie_index).copied().unwrap_or(0);
            self.emit_fde(&mut frame, fde, cie_offset);
        }

        frame
    }

    fn emit_cie(&self, buf: &mut Vec<u8>, cie: &Cie) {
        let start = buf.len();

        // Placeholder for length
        buf.extend_from_slice(&0u32.to_le_bytes());

        // CIE ID (0xFFFFFFFF for .debug_frame)
        buf.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());

        // Version
        buf.push(cie.version);

        // Augmentation string (null-terminated)
        buf.extend_from_slice(cie.augmentation.as_bytes());
        buf.push(0);

        if cie.version >= 4 {
            // Address size and segment size (DWARF 4+)
            buf.push(cie.address_size);
            buf.push(cie.segment_size);
        }

        // Code alignment factor (ULEB128)
        write_uleb128(buf, cie.code_alignment_factor);

        // Data alignment factor (SLEB128)
        write_sleb128(buf, cie.data_alignment_factor);

        // Return address register (ULEB128)
        write_uleb128(buf, cie.return_address_register);

        // Initial instructions
        for instr in &cie.initial_instructions {
            emit_cfa_instruction(buf, instr);
        }

        // Align to pointer size
        while (buf.len() - start) % (cie.address_size as usize) != 0 {
            buf.push(0); // DW_CFA_nop padding
        }

        // Fix up length
        let length = (buf.len() - start - 4) as u32;
        buf[start..start + 4].copy_from_slice(&length.to_le_bytes());
    }

    fn emit_fde(&self, buf: &mut Vec<u8>, fde: &Fde, cie_offset: u32) {
        let cie = &self.cies[fde.cie_index];
        let start = buf.len();

        // Placeholder for length
        buf.extend_from_slice(&0u32.to_le_bytes());

        // CIE pointer (offset from start of .debug_frame)
        buf.extend_from_slice(&cie_offset.to_le_bytes());

        // Initial location
        if cie.address_size == 8 {
            buf.extend_from_slice(&fde.initial_location.to_le_bytes());
        } else {
            buf.extend_from_slice(&(fde.initial_location as u32).to_le_bytes());
        }

        // Address range
        if cie.address_size == 8 {
            buf.extend_from_slice(&fde.address_range.to_le_bytes());
        } else {
            buf.extend_from_slice(&(fde.address_range as u32).to_le_bytes());
        }

        // Instructions
        for instr in &fde.instructions {
            emit_cfa_instruction(buf, instr);
        }

        // Align to pointer size
        while (buf.len() - start) % (cie.address_size as usize) != 0 {
            buf.push(0); // DW_CFA_nop padding
        }

        // Fix up length
        let length = (buf.len() - start - 4) as u32;
        buf[start..start + 4].copy_from_slice(&length.to_le_bytes());
    }
}

// CFI opcode constants
mod opcodes {
    pub const DW_CFA_NOP: u8 = 0x00;
    pub const DW_CFA_SET_LOC: u8 = 0x01;
    pub const DW_CFA_ADVANCE_LOC1: u8 = 0x02;
    pub const DW_CFA_ADVANCE_LOC2: u8 = 0x03;
    pub const DW_CFA_ADVANCE_LOC4: u8 = 0x04;
    pub const DW_CFA_OFFSET_EXTENDED: u8 = 0x05;
    pub const DW_CFA_RESTORE_EXTENDED: u8 = 0x06;
    pub const DW_CFA_UNDEFINED: u8 = 0x07;
    pub const DW_CFA_SAME_VALUE: u8 = 0x08;
    pub const DW_CFA_REGISTER: u8 = 0x09;
    pub const DW_CFA_REMEMBER_STATE: u8 = 0x0a;
    pub const DW_CFA_RESTORE_STATE: u8 = 0x0b;
    pub const DW_CFA_DEF_CFA: u8 = 0x0c;
    pub const DW_CFA_DEF_CFA_REGISTER: u8 = 0x0d;
    pub const DW_CFA_DEF_CFA_OFFSET: u8 = 0x0e;
    pub const DW_CFA_DEF_CFA_EXPRESSION: u8 = 0x0f;
    pub const DW_CFA_EXPRESSION: u8 = 0x10;
    pub const DW_CFA_OFFSET_EXTENDED_SF: u8 = 0x11;
    pub const DW_CFA_DEF_CFA_SF: u8 = 0x12;
    pub const DW_CFA_DEF_CFA_OFFSET_SF: u8 = 0x13;
    pub const DW_CFA_VAL_OFFSET: u8 = 0x14;
    pub const DW_CFA_VAL_OFFSET_SF: u8 = 0x15;
    pub const DW_CFA_VAL_EXPRESSION: u8 = 0x16;

    // High 2 bits encode the operation
    pub const DW_CFA_ADVANCE_LOC: u8 = 0x40; // 01xxxxxx
    pub const DW_CFA_OFFSET: u8 = 0x80;      // 10xxxxxx
    pub const DW_CFA_RESTORE: u8 = 0xC0;     // 11xxxxxx
}

fn emit_cfa_instruction(buf: &mut Vec<u8>, instr: &CfaInstruction) {
    use opcodes::*;

    match instr {
        CfaInstruction::Nop => {
            buf.push(DW_CFA_NOP);
        }
        CfaInstruction::AdvanceLoc { delta } => {
            if *delta < 64 {
                buf.push(DW_CFA_ADVANCE_LOC | delta);
            } else {
                buf.push(DW_CFA_ADVANCE_LOC1);
                buf.push(*delta);
            }
        }
        CfaInstruction::AdvanceLoc1 { delta } => {
            buf.push(DW_CFA_ADVANCE_LOC1);
            buf.push(*delta);
        }
        CfaInstruction::AdvanceLoc2 { delta } => {
            buf.push(DW_CFA_ADVANCE_LOC2);
            buf.extend_from_slice(&delta.to_le_bytes());
        }
        CfaInstruction::AdvanceLoc4 { delta } => {
            buf.push(DW_CFA_ADVANCE_LOC4);
            buf.extend_from_slice(&delta.to_le_bytes());
        }
        CfaInstruction::SetLoc { address } => {
            buf.push(DW_CFA_SET_LOC);
            buf.extend_from_slice(&address.to_le_bytes());
        }
        CfaInstruction::Offset { register, offset } => {
            if *register < 64 {
                buf.push(DW_CFA_OFFSET | register);
                write_uleb128(buf, *offset);
            } else {
                buf.push(DW_CFA_OFFSET_EXTENDED);
                write_uleb128(buf, *register as u64);
                write_uleb128(buf, *offset);
            }
        }
        CfaInstruction::OffsetExtended { register, offset } => {
            buf.push(DW_CFA_OFFSET_EXTENDED);
            write_uleb128(buf, *register);
            write_uleb128(buf, *offset);
        }
        CfaInstruction::OffsetExtendedSf { register, offset } => {
            buf.push(DW_CFA_OFFSET_EXTENDED_SF);
            write_uleb128(buf, *register);
            write_sleb128(buf, *offset);
        }
        CfaInstruction::Restore { register } => {
            if *register < 64 {
                buf.push(DW_CFA_RESTORE | register);
            } else {
                buf.push(DW_CFA_RESTORE_EXTENDED);
                write_uleb128(buf, *register as u64);
            }
        }
        CfaInstruction::RestoreExtended { register } => {
            buf.push(DW_CFA_RESTORE_EXTENDED);
            write_uleb128(buf, *register);
        }
        CfaInstruction::Undefined { register } => {
            buf.push(DW_CFA_UNDEFINED);
            write_uleb128(buf, *register);
        }
        CfaInstruction::SameValue { register } => {
            buf.push(DW_CFA_SAME_VALUE);
            write_uleb128(buf, *register);
        }
        CfaInstruction::Register { target, source } => {
            buf.push(DW_CFA_REGISTER);
            write_uleb128(buf, *target);
            write_uleb128(buf, *source);
        }
        CfaInstruction::RememberState => {
            buf.push(DW_CFA_REMEMBER_STATE);
        }
        CfaInstruction::RestoreState => {
            buf.push(DW_CFA_RESTORE_STATE);
        }
        CfaInstruction::DefCfa { register, offset } => {
            buf.push(DW_CFA_DEF_CFA);
            write_uleb128(buf, *register);
            write_uleb128(buf, *offset);
        }
        CfaInstruction::DefCfaRegister { register } => {
            buf.push(DW_CFA_DEF_CFA_REGISTER);
            write_uleb128(buf, *register);
        }
        CfaInstruction::DefCfaOffset { offset } => {
            buf.push(DW_CFA_DEF_CFA_OFFSET);
            write_uleb128(buf, *offset);
        }
        CfaInstruction::DefCfaSf { register, offset } => {
            buf.push(DW_CFA_DEF_CFA_SF);
            write_uleb128(buf, *register);
            write_sleb128(buf, *offset);
        }
        CfaInstruction::DefCfaOffsetSf { offset } => {
            buf.push(DW_CFA_DEF_CFA_OFFSET_SF);
            write_sleb128(buf, *offset);
        }
        CfaInstruction::DefCfaExpression { expression } => {
            buf.push(DW_CFA_DEF_CFA_EXPRESSION);
            write_uleb128(buf, expression.len() as u64);
            buf.extend_from_slice(expression);
        }
        CfaInstruction::Expression { register, expression } => {
            buf.push(DW_CFA_EXPRESSION);
            write_uleb128(buf, *register);
            write_uleb128(buf, expression.len() as u64);
            buf.extend_from_slice(expression);
        }
        CfaInstruction::ValOffset { register, offset } => {
            buf.push(DW_CFA_VAL_OFFSET);
            write_uleb128(buf, *register);
            write_uleb128(buf, *offset);
        }
        CfaInstruction::ValOffsetSf { register, offset } => {
            buf.push(DW_CFA_VAL_OFFSET_SF);
            write_uleb128(buf, *register);
            write_sleb128(buf, *offset);
        }
        CfaInstruction::ValExpression { register, expression } => {
            buf.push(DW_CFA_VAL_EXPRESSION);
            write_uleb128(buf, *register);
            write_uleb128(buf, expression.len() as u64);
            buf.extend_from_slice(expression);
        }
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
    fn test_cfi_builder_x86_64() {
        let mut builder = CfiBuilder::new_x86_64();

        // Create an FDE for a simple function
        let fde_idx = builder.begin_fde(0x1000, 0x100);

        // Emit prologue (push rbp; mov rbp, rsp; sub rsp, 32)
        builder.emit_cfi_prologue(fde_idx, 32);

        // Save some callee-saved registers
        builder.emit_save_register(fde_idx, x86_64::RBX, 3); // at CFA - 24
        builder.emit_save_register(fde_idx, x86_64::R12, 4); // at CFA - 32

        // Remember state before epilogue
        builder.emit_remember_state(fde_idx);

        // Emit epilogue
        builder.emit_cfi_epilogue(fde_idx);

        let cfi = builder.build();

        assert_eq!(cfi.cies.len(), 1);
        assert_eq!(cfi.fdes.len(), 1);
        assert_eq!(cfi.fdes[0].initial_location, 0x1000);
        assert_eq!(cfi.fdes[0].address_range, 0x100);

        // Generate debug_frame section
        let debug_frame = cfi.generate_debug_frame();
        assert!(!debug_frame.is_empty());
    }

    #[test]
    fn test_cfi_builder_aarch64() {
        let mut builder = CfiBuilder::new_aarch64();

        let fde_idx = builder.begin_fde(0x2000, 0x80);

        // AArch64 prologue: stp x29, x30, [sp, #-16]!
        builder.add_instruction(
            fde_idx,
            CfaInstruction::DefCfaOffset { offset: 16 },
        );
        builder.add_instruction(
            fde_idx,
            CfaInstruction::Offset {
                register: aarch64::X29 as u8,
                offset: 2,
            },
        );
        builder.add_instruction(
            fde_idx,
            CfaInstruction::Offset {
                register: aarch64::X30 as u8,
                offset: 1,
            },
        );

        let cfi = builder.build();
        let debug_frame = cfi.generate_debug_frame();

        assert!(!debug_frame.is_empty());
    }

    #[test]
    fn test_leb128() {
        let mut buf = Vec::new();

        write_uleb128(&mut buf, 127);
        assert_eq!(buf, vec![127]);

        buf.clear();
        write_uleb128(&mut buf, 128);
        assert_eq!(buf, vec![0x80, 0x01]);

        buf.clear();
        write_sleb128(&mut buf, -1);
        assert_eq!(buf, vec![0x7f]);

        buf.clear();
        write_sleb128(&mut buf, -128);
        assert_eq!(buf, vec![0x80, 0x7f]);
    }
}
