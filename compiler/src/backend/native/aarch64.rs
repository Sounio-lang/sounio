//! AArch64 (ARM64) Code Emission
//!
//! This module provides AArch64 machine code generation for the native backend.
//! All instructions are 32-bit fixed-width, encoded in little-endian byte order.
//!
//! # Encoding Reference
//!
//! AArch64 uses fixed 32-bit instruction encoding. Key instruction formats:
//! - Data Processing (Register): [sf|opc|S|11010|shift|rm|imm6|rn|rd]
//! - Data Processing (Immediate): [sf|opc|100|shift|imm12|rn|rd]
//! - Branches: [opcode|imm26] or [opcode|imm19|cond]
//! - Load/Store: [size|opc|01|imm12|rn|rt] or [size|opc|00|rm|opt|S|10|rn|rt]
//!
//! # Calling Convention (AAPCS64)
//!
//! - X0-X7: Arguments and return values
//! - X8: Indirect result location register
//! - X9-X15: Temporary registers
//! - X16-X17: Intra-procedure call scratch registers (IP0, IP1)
//! - X18: Platform register (reserved)
//! - X19-X28: Callee-saved registers
//! - X29: Frame pointer (FP)
//! - X30: Link register (LR)
//! - SP: Stack pointer (must be 16-byte aligned)

use crate::sir::module::SirModule;
use crate::sir::blocks::SirFunction;

// ============================================================================
// REGISTERS
// ============================================================================

/// AArch64 registers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AArch64Reg {
    // General-purpose registers (64-bit X form)
    X0 = 0, X1 = 1, X2 = 2, X3 = 3, X4 = 4, X5 = 5, X6 = 6, X7 = 7,
    X8 = 8, X9 = 9, X10 = 10, X11 = 11, X12 = 12, X13 = 13, X14 = 14, X15 = 15,
    X16 = 16, X17 = 17, X18 = 18,
    X19 = 19, X20 = 20, X21 = 21, X22 = 22, X23 = 23, X24 = 24, X25 = 25, X26 = 26, X27 = 27, X28 = 28,
    X29 = 29,  // Frame pointer (FP)
    X30 = 30,  // Link register (LR)
    XZR = 31,  // Zero register / Stack pointer (context-dependent)

    // SIMD/FP registers (128-bit V form, can be accessed as B/H/S/D/Q)
    V0 = 32, V1 = 33, V2 = 34, V3 = 35, V4 = 36, V5 = 37, V6 = 38, V7 = 39,
    V8 = 40, V9 = 41, V10 = 42, V11 = 43, V12 = 44, V13 = 45, V14 = 46, V15 = 47,
    V16 = 48, V17 = 49, V18 = 50, V19 = 51, V20 = 52, V21 = 53, V22 = 54, V23 = 55,
    V24 = 56, V25 = 57, V26 = 58, V27 = 59, V28 = 60, V29 = 61, V30 = 62, V31 = 63,
}

/// Aliases for common registers
pub const FP: AArch64Reg = AArch64Reg::X29;
pub const LR: AArch64Reg = AArch64Reg::X30;
pub const SP: AArch64Reg = AArch64Reg::XZR;  // Context: stack pointer

impl AArch64Reg {
    /// Get the 5-bit register encoding
    #[inline]
    pub fn encoding(self) -> u32 {
        (self as u8 & 0x1F) as u32
    }

    /// Is this a general-purpose register?
    #[inline]
    pub fn is_gpr(self) -> bool {
        (self as u8) < 32
    }

    /// Is this a SIMD/FP register?
    #[inline]
    pub fn is_simd(self) -> bool {
        (self as u8) >= 32
    }

    /// Is this a callee-saved register?
    pub fn is_callee_saved(self) -> bool {
        matches!(
            self,
            AArch64Reg::X19 | AArch64Reg::X20 | AArch64Reg::X21 | AArch64Reg::X22
            | AArch64Reg::X23 | AArch64Reg::X24 | AArch64Reg::X25 | AArch64Reg::X26
            | AArch64Reg::X27 | AArch64Reg::X28
            | AArch64Reg::V8 | AArch64Reg::V9 | AArch64Reg::V10 | AArch64Reg::V11
            | AArch64Reg::V12 | AArch64Reg::V13 | AArch64Reg::V14 | AArch64Reg::V15
        )
    }

    /// Is this a caller-saved (volatile) register?
    pub fn is_caller_saved(self) -> bool {
        matches!(
            self,
            AArch64Reg::X0 | AArch64Reg::X1 | AArch64Reg::X2 | AArch64Reg::X3
            | AArch64Reg::X4 | AArch64Reg::X5 | AArch64Reg::X6 | AArch64Reg::X7
            | AArch64Reg::X8 | AArch64Reg::X9 | AArch64Reg::X10 | AArch64Reg::X11
            | AArch64Reg::X12 | AArch64Reg::X13 | AArch64Reg::X14 | AArch64Reg::X15
            | AArch64Reg::X16 | AArch64Reg::X17
        )
    }
}

// ============================================================================
// CONDITION CODES
// ============================================================================

/// AArch64 condition codes for conditional branches
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Condition {
    EQ = 0b0000,  // Equal (Z=1)
    NE = 0b0001,  // Not equal (Z=0)
    CS = 0b0010,  // Carry set / unsigned higher or same (C=1)
    CC = 0b0011,  // Carry clear / unsigned lower (C=0)
    MI = 0b0100,  // Minus / negative (N=1)
    PL = 0b0101,  // Plus / positive or zero (N=0)
    VS = 0b0110,  // Overflow set (V=1)
    VC = 0b0111,  // Overflow clear (V=0)
    HI = 0b1000,  // Unsigned higher (C=1 && Z=0)
    LS = 0b1001,  // Unsigned lower or same (C=0 || Z=1)
    GE = 0b1010,  // Signed greater than or equal (N=V)
    LT = 0b1011,  // Signed less than (N!=V)
    GT = 0b1100,  // Signed greater than (Z=0 && N=V)
    LE = 0b1101,  // Signed less than or equal (Z=1 || N!=V)
    AL = 0b1110,  // Always (unconditional)
    NV = 0b1111,  // Never (reserved, behaves as AL)
}

/// Aliases for condition codes
pub const HS: Condition = Condition::CS;  // Unsigned higher or same
pub const LO: Condition = Condition::CC;  // Unsigned lower

// ============================================================================
// SHIFT TYPES
// ============================================================================

/// Shift type for data processing instructions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Shift {
    LSL = 0b00,  // Logical shift left
    LSR = 0b01,  // Logical shift right
    ASR = 0b10,  // Arithmetic shift right
    ROR = 0b11,  // Rotate right (or RRX when amount=0)
}

// ============================================================================
// INSTRUCTION ENCODING
// ============================================================================

/// AArch64 code emitter with instruction encoding functions
pub struct AArch64Emitter {
    /// Machine code buffer
    code: Vec<u8>,
    /// Current stack frame size
    frame_size: u32,
    /// Labels and their offsets for branch resolution
    labels: std::collections::HashMap<String, u32>,
    /// Unresolved branch references (offset, label, is_cond)
    unresolved: Vec<(u32, String, bool)>,
}

impl AArch64Emitter {
    /// Create a new AArch64 emitter
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(4096),
            frame_size: 0,
            labels: std::collections::HashMap::new(),
            unresolved: Vec::new(),
        }
    }

    /// Get the current code offset
    #[inline]
    pub fn offset(&self) -> u32 {
        self.code.len() as u32
    }

    /// Emit a 32-bit instruction (little-endian)
    #[inline]
    pub fn emit(&mut self, inst: u32) {
        self.code.extend_from_slice(&inst.to_le_bytes());
    }

    /// Define a label at the current position
    pub fn label(&mut self, name: &str) {
        self.labels.insert(name.to_string(), self.offset());
    }

    /// Get the generated machine code
    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        // Resolve all branch references
        for (offset, label, is_cond) in &self.unresolved {
            let target = self.labels.get(label)
                .ok_or_else(|| format!("Undefined label: {}", label))?;
            let pc = *offset;
            let delta = (*target as i64) - (pc as i64);

            if *is_cond {
                // Conditional branch: imm19 field (bits 5-23), offset in 4-byte units
                let imm19 = (delta / 4) as i32;
                if imm19 < -(1 << 18) || imm19 >= (1 << 18) {
                    return Err(format!("Conditional branch target too far: {}", label));
                }
                let inst = u32::from_le_bytes([
                    self.code[*offset as usize],
                    self.code[*offset as usize + 1],
                    self.code[*offset as usize + 2],
                    self.code[*offset as usize + 3],
                ]);
                let patched = (inst & 0xFF00001F) | (((imm19 as u32) & 0x7FFFF) << 5);
                self.code[*offset as usize..*offset as usize + 4]
                    .copy_from_slice(&patched.to_le_bytes());
            } else {
                // Unconditional branch: imm26 field (bits 0-25), offset in 4-byte units
                let imm26 = (delta / 4) as i32;
                if imm26 < -(1 << 25) || imm26 >= (1 << 25) {
                    return Err(format!("Branch target too far: {}", label));
                }
                let inst = u32::from_le_bytes([
                    self.code[*offset as usize],
                    self.code[*offset as usize + 1],
                    self.code[*offset as usize + 2],
                    self.code[*offset as usize + 3],
                ]);
                let patched = (inst & 0xFC000000) | ((imm26 as u32) & 0x03FFFFFF);
                self.code[*offset as usize..*offset as usize + 4]
                    .copy_from_slice(&patched.to_le_bytes());
            }
        }

        Ok(self.code)
    }

    // ========================================================================
    // DATA PROCESSING - IMMEDIATE
    // ========================================================================

    /// ADD (immediate): Rd = Rn + imm12
    /// sf=1 for 64-bit, sf=0 for 32-bit
    pub fn add_imm(&mut self, rd: AArch64Reg, rn: AArch64Reg, imm12: u16, sf: bool) {
        debug_assert!(imm12 < 4096, "imm12 must be < 4096");
        let inst = ((sf as u32) << 31)
            | (0b00100010 << 23)  // ADD immediate opcode
            | ((imm12 as u32 & 0xFFF) << 10)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// ADDS (immediate): Rd = Rn + imm12, set flags
    pub fn adds_imm(&mut self, rd: AArch64Reg, rn: AArch64Reg, imm12: u16, sf: bool) {
        debug_assert!(imm12 < 4096, "imm12 must be < 4096");
        let inst = ((sf as u32) << 31)
            | (0b01100010 << 23)  // ADDS immediate opcode
            | ((imm12 as u32 & 0xFFF) << 10)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// SUB (immediate): Rd = Rn - imm12
    pub fn sub_imm(&mut self, rd: AArch64Reg, rn: AArch64Reg, imm12: u16, sf: bool) {
        debug_assert!(imm12 < 4096, "imm12 must be < 4096");
        let inst = ((sf as u32) << 31)
            | (0b10100010 << 23)  // SUB immediate opcode
            | ((imm12 as u32 & 0xFFF) << 10)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// SUBS (immediate): Rd = Rn - imm12, set flags
    pub fn subs_imm(&mut self, rd: AArch64Reg, rn: AArch64Reg, imm12: u16, sf: bool) {
        debug_assert!(imm12 < 4096, "imm12 must be < 4096");
        let inst = ((sf as u32) << 31)
            | (0b11100010 << 23)  // SUBS immediate opcode
            | ((imm12 as u32 & 0xFFF) << 10)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// CMP (immediate): compare Rn with imm12 (alias for SUBS XZR, Rn, imm12)
    pub fn cmp_imm(&mut self, rn: AArch64Reg, imm12: u16, sf: bool) {
        self.subs_imm(AArch64Reg::XZR, rn, imm12, sf);
    }

    // ========================================================================
    // DATA PROCESSING - REGISTER
    // ========================================================================

    /// ADD (register): Rd = Rn + Rm
    pub fn add_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0001011 << 24)   // ADD shifted register opcode
            | (rm.encoding() << 16)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// ADD (register, shifted): Rd = Rn + (Rm << shift_amt)
    pub fn add_reg_shifted(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg,
                           shift: Shift, shift_amt: u8, sf: bool) {
        let max_shift = if sf { 63 } else { 31 };
        debug_assert!(shift_amt <= max_shift, "shift amount too large");
        let inst = ((sf as u32) << 31)
            | (0b0001011 << 24)
            | ((shift as u32) << 22)
            | (rm.encoding() << 16)
            | ((shift_amt as u32) << 10)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// SUB (register): Rd = Rn - Rm
    pub fn sub_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b1001011 << 24)   // SUB shifted register opcode
            | (rm.encoding() << 16)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// SUBS (register): Rd = Rn - Rm, set flags
    pub fn subs_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b1101011 << 24)   // SUBS shifted register opcode
            | (rm.encoding() << 16)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// CMP (register): compare Rn with Rm (alias for SUBS XZR, Rn, Rm)
    pub fn cmp_reg(&mut self, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        self.subs_reg(AArch64Reg::XZR, rn, rm, sf);
    }

    /// MUL: Rd = Rn * Rm (alias for MADD Rd, Rn, Rm, XZR)
    pub fn mul(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0011011000 << 21)
            | (rm.encoding() << 16)
            | (AArch64Reg::XZR.encoding() << 10)  // Ra = XZR for MUL
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// SDIV: Rd = Rn / Rm (signed)
    pub fn sdiv(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0011010110 << 21)
            | (rm.encoding() << 16)
            | (0b000011 << 10)   // SDIV opcode
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// UDIV: Rd = Rn / Rm (unsigned)
    pub fn udiv(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0011010110 << 21)
            | (rm.encoding() << 16)
            | (0b000010 << 10)   // UDIV opcode
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    // ========================================================================
    // LOGICAL OPERATIONS
    // ========================================================================

    /// AND (register): Rd = Rn & Rm
    pub fn and_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0001010 << 24)
            | (rm.encoding() << 16)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// ORR (register): Rd = Rn | Rm
    pub fn orr_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0101010 << 24)
            | (rm.encoding() << 16)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// EOR (register): Rd = Rn ^ Rm
    pub fn eor_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b1001010 << 24)
            | (rm.encoding() << 16)
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// MVN (register): Rd = ~Rm (alias for ORN Rd, XZR, Rm)
    pub fn mvn(&mut self, rd: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0101010 << 24)
            | (1 << 21)           // N bit for ORN
            | (rm.encoding() << 16)
            | (AArch64Reg::XZR.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    // ========================================================================
    // SHIFT OPERATIONS
    // ========================================================================

    /// LSL (register): Rd = Rn << (Rm & 63)
    pub fn lsl_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0011010110 << 21)
            | (rm.encoding() << 16)
            | (0b001000 << 10)  // LSL opcode
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// LSR (register): Rd = Rn >> (Rm & 63) (logical)
    pub fn lsr_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0011010110 << 21)
            | (rm.encoding() << 16)
            | (0b001001 << 10)  // LSR opcode
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// ASR (register): Rd = Rn >> (Rm & 63) (arithmetic)
    pub fn asr_reg(&mut self, rd: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, sf: bool) {
        let inst = ((sf as u32) << 31)
            | (0b0011010110 << 21)
            | (rm.encoding() << 16)
            | (0b001010 << 10)  // ASR opcode
            | (rn.encoding() << 5)
            | rd.encoding();
        self.emit(inst);
    }

    // ========================================================================
    // MOVE OPERATIONS
    // ========================================================================

    /// MOV (register): Rd = Rm (alias for ORR Rd, XZR, Rm)
    pub fn mov_reg(&mut self, rd: AArch64Reg, rm: AArch64Reg, sf: bool) {
        self.orr_reg(rd, AArch64Reg::XZR, rm, sf);
    }

    /// MOVZ: Rd = imm16 << (hw * 16), zero other bits
    pub fn movz(&mut self, rd: AArch64Reg, imm16: u16, hw: u8, sf: bool) {
        debug_assert!(hw < if sf { 4 } else { 2 }, "hw out of range");
        let inst = ((sf as u32) << 31)
            | (0b10100101 << 23)
            | ((hw as u32) << 21)
            | ((imm16 as u32) << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// MOVK: Rd = (Rd & ~(0xFFFF << (hw * 16))) | (imm16 << (hw * 16))
    pub fn movk(&mut self, rd: AArch64Reg, imm16: u16, hw: u8, sf: bool) {
        debug_assert!(hw < if sf { 4 } else { 2 }, "hw out of range");
        let inst = ((sf as u32) << 31)
            | (0b11100101 << 23)
            | ((hw as u32) << 21)
            | ((imm16 as u32) << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// MOVN: Rd = ~(imm16 << (hw * 16))
    pub fn movn(&mut self, rd: AArch64Reg, imm16: u16, hw: u8, sf: bool) {
        debug_assert!(hw < if sf { 4 } else { 2 }, "hw out of range");
        let inst = ((sf as u32) << 31)
            | (0b00100101 << 23)
            | ((hw as u32) << 21)
            | ((imm16 as u32) << 5)
            | rd.encoding();
        self.emit(inst);
    }

    /// Load a 64-bit immediate into a register
    pub fn mov_imm64(&mut self, rd: AArch64Reg, imm: u64) {
        // Optimize for common cases
        if imm == 0 {
            self.movz(rd, 0, 0, true);
            return;
        }

        // Check if we can use MOVN (all ones except one chunk)
        let inverted = !imm;
        let mut chunks_inverted = [
            (inverted & 0xFFFF) as u16,
            ((inverted >> 16) & 0xFFFF) as u16,
            ((inverted >> 32) & 0xFFFF) as u16,
            ((inverted >> 48) & 0xFFFF) as u16,
        ];
        let non_zero_inv: Vec<usize> = chunks_inverted.iter().enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(i, _)| i)
            .collect();

        if non_zero_inv.len() == 1 {
            let hw = non_zero_inv[0] as u8;
            self.movn(rd, chunks_inverted[hw as usize], hw, true);
            return;
        }

        // General case: MOVZ + MOVK sequence
        let chunks = [
            (imm & 0xFFFF) as u16,
            ((imm >> 16) & 0xFFFF) as u16,
            ((imm >> 32) & 0xFFFF) as u16,
            ((imm >> 48) & 0xFFFF) as u16,
        ];

        let mut first = true;
        for (hw, &chunk) in chunks.iter().enumerate() {
            if chunk != 0 || (first && hw == 3) {
                if first {
                    self.movz(rd, chunk, hw as u8, true);
                    first = false;
                } else {
                    self.movk(rd, chunk, hw as u8, true);
                }
            }
        }

        // Handle zero case (shouldn't reach here but safety)
        if first {
            self.movz(rd, 0, 0, true);
        }
    }

    // ========================================================================
    // BRANCHES
    // ========================================================================

    /// B: unconditional branch to PC + offset
    pub fn b(&mut self, offset: i32) {
        debug_assert!(offset % 4 == 0, "branch offset must be 4-byte aligned");
        let imm26 = (offset / 4) as u32 & 0x03FFFFFF;
        let inst = (0b000101 << 26) | imm26;
        self.emit(inst);
    }

    /// B.cond: conditional branch to PC + offset
    pub fn b_cond(&mut self, cond: Condition, offset: i32) {
        debug_assert!(offset % 4 == 0, "branch offset must be 4-byte aligned");
        let imm19 = ((offset / 4) as u32) & 0x7FFFF;
        let inst = (0b01010100 << 24) | (imm19 << 5) | (cond as u32);
        self.emit(inst);
    }

    /// B to label (unresolved, will be patched)
    pub fn b_label(&mut self, label: &str) {
        let offset = self.offset();
        self.unresolved.push((offset, label.to_string(), false));
        self.emit(0b000101 << 26);  // Placeholder
    }

    /// B.cond to label (unresolved, will be patched)
    pub fn b_cond_label(&mut self, cond: Condition, label: &str) {
        let offset = self.offset();
        self.unresolved.push((offset, label.to_string(), true));
        self.emit((0b01010100 << 24) | (cond as u32));  // Placeholder
    }

    /// BL: branch with link (call)
    pub fn bl(&mut self, offset: i32) {
        debug_assert!(offset % 4 == 0, "branch offset must be 4-byte aligned");
        let imm26 = (offset / 4) as u32 & 0x03FFFFFF;
        let inst = (0b100101 << 26) | imm26;
        self.emit(inst);
    }

    /// BLR: branch with link to register (indirect call)
    pub fn blr(&mut self, rn: AArch64Reg) {
        let inst = (0b1101011000111111 << 16)
            | (rn.encoding() << 5);
        self.emit(inst);
    }

    /// BR: branch to register (indirect jump)
    pub fn br(&mut self, rn: AArch64Reg) {
        let inst = (0b1101011000011111 << 16)
            | (rn.encoding() << 5);
        self.emit(inst);
    }

    /// RET: return (alias for BR X30)
    pub fn ret(&mut self) {
        let inst = (0b1101011001011111 << 16)
            | (AArch64Reg::X30.encoding() << 5);
        self.emit(inst);
    }

    // ========================================================================
    // LOAD/STORE
    // ========================================================================

    /// LDR (immediate, unsigned offset): Rt = [Rn + imm12 * scale]
    /// scale = 8 for 64-bit, 4 for 32-bit
    pub fn ldr_imm(&mut self, rt: AArch64Reg, rn: AArch64Reg, imm12: u16, size: u8) {
        let scale = match size {
            64 => 3,
            32 => 2,
            16 => 1,
            8 => 0,
            _ => panic!("Invalid load size"),
        };
        debug_assert!(imm12 < 4096, "imm12 must be < 4096");
        let inst = ((size as u32 / 32) << 30)
            | (0b11100101 << 22)
            | ((imm12 as u32) << 10)
            | (rn.encoding() << 5)
            | rt.encoding();
        self.emit(inst);
    }

    /// LDR (register): Rt = [Rn + Rm]
    pub fn ldr_reg(&mut self, rt: AArch64Reg, rn: AArch64Reg, rm: AArch64Reg, size: u8) {
        let opc = match size {
            64 => 0b11,
            32 => 0b10,
            16 => 0b01,
            8 => 0b00,
            _ => panic!("Invalid load size"),
        };
        let inst = ((opc as u32) << 30)
            | (0b111000011 << 21)
            | (rm.encoding() << 16)
            | (0b011010 << 10)  // LSL #0, extend register
            | (rn.encoding() << 5)
            | rt.encoding();
        self.emit(inst);
    }

    /// STR (immediate, unsigned offset): [Rn + imm12 * scale] = Rt
    pub fn str_imm(&mut self, rt: AArch64Reg, rn: AArch64Reg, imm12: u16, size: u8) {
        debug_assert!(imm12 < 4096, "imm12 must be < 4096");
        let opc = match size {
            64 => 0b11,
            32 => 0b10,
            16 => 0b01,
            8 => 0b00,
            _ => panic!("Invalid store size"),
        };
        let inst = ((opc as u32) << 30)
            | (0b11100100 << 22)
            | ((imm12 as u32) << 10)
            | (rn.encoding() << 5)
            | rt.encoding();
        self.emit(inst);
    }

    /// LDP: Load pair of registers
    /// [Rn + imm7 * 8] = (Rt, Rt2) for 64-bit
    pub fn ldp(&mut self, rt: AArch64Reg, rt2: AArch64Reg, rn: AArch64Reg, imm7: i8, sf: bool) {
        debug_assert!(imm7 >= -64 && imm7 < 64, "imm7 out of range");
        let inst = ((sf as u32) << 31)
            | (0b10100101 << 22)
            | (((imm7 as u32) & 0x7F) << 15)
            | (rt2.encoding() << 10)
            | (rn.encoding() << 5)
            | rt.encoding();
        self.emit(inst);
    }

    /// STP: Store pair of registers
    pub fn stp(&mut self, rt: AArch64Reg, rt2: AArch64Reg, rn: AArch64Reg, imm7: i8, sf: bool) {
        debug_assert!(imm7 >= -64 && imm7 < 64, "imm7 out of range");
        let inst = ((sf as u32) << 31)
            | (0b10100100 << 22)
            | (((imm7 as u32) & 0x7F) << 15)
            | (rt2.encoding() << 10)
            | (rn.encoding() << 5)
            | rt.encoding();
        self.emit(inst);
    }

    /// STP (pre-index): [Rn + imm7 * 8]! = (Rt, Rt2)
    pub fn stp_pre(&mut self, rt: AArch64Reg, rt2: AArch64Reg, rn: AArch64Reg, imm7: i8, sf: bool) {
        debug_assert!(imm7 >= -64 && imm7 < 64, "imm7 out of range");
        let inst = ((sf as u32) << 31)
            | (0b10100110 << 22)
            | (((imm7 as u32) & 0x7F) << 15)
            | (rt2.encoding() << 10)
            | (rn.encoding() << 5)
            | rt.encoding();
        self.emit(inst);
    }

    /// LDP (post-index): (Rt, Rt2) = [Rn], Rn += imm7 * 8
    pub fn ldp_post(&mut self, rt: AArch64Reg, rt2: AArch64Reg, rn: AArch64Reg, imm7: i8, sf: bool) {
        debug_assert!(imm7 >= -64 && imm7 < 64, "imm7 out of range");
        let inst = ((sf as u32) << 31)
            | (0b10100011 << 22)
            | (((imm7 as u32) & 0x7F) << 15)
            | (rt2.encoding() << 10)
            | (rn.encoding() << 5)
            | rt.encoding();
        self.emit(inst);
    }

    // ========================================================================
    // FLOATING POINT
    // ========================================================================

    /// FADD (scalar): Vd = Vn + Vm
    pub fn fadd(&mut self, vd: AArch64Reg, vn: AArch64Reg, vm: AArch64Reg, double: bool) {
        let inst = (if double { 0b0001111001 } else { 0b0001111000 } << 22)
            | (vm.encoding() << 16)
            | (0b001010 << 10)
            | (vn.encoding() << 5)
            | vd.encoding();
        self.emit(inst);
    }

    /// FSUB (scalar): Vd = Vn - Vm
    pub fn fsub(&mut self, vd: AArch64Reg, vn: AArch64Reg, vm: AArch64Reg, double: bool) {
        let inst = (if double { 0b0001111001 } else { 0b0001111000 } << 22)
            | (vm.encoding() << 16)
            | (0b001110 << 10)
            | (vn.encoding() << 5)
            | vd.encoding();
        self.emit(inst);
    }

    /// FMUL (scalar): Vd = Vn * Vm
    pub fn fmul(&mut self, vd: AArch64Reg, vn: AArch64Reg, vm: AArch64Reg, double: bool) {
        let inst = (if double { 0b0001111001 } else { 0b0001111000 } << 22)
            | (vm.encoding() << 16)
            | (0b000010 << 10)
            | (vn.encoding() << 5)
            | vd.encoding();
        self.emit(inst);
    }

    /// FDIV (scalar): Vd = Vn / Vm
    pub fn fdiv(&mut self, vd: AArch64Reg, vn: AArch64Reg, vm: AArch64Reg, double: bool) {
        let inst = (if double { 0b0001111001 } else { 0b0001111000 } << 22)
            | (vm.encoding() << 16)
            | (0b000110 << 10)
            | (vn.encoding() << 5)
            | vd.encoding();
        self.emit(inst);
    }

    /// FCMP: compare Vn with Vm, set NZCV
    pub fn fcmp(&mut self, vn: AArch64Reg, vm: AArch64Reg, double: bool) {
        let inst = (if double { 0b0001111001 } else { 0b0001111000 } << 22)
            | (vm.encoding() << 16)
            | (0b001000 << 10)
            | (vn.encoding() << 5);
        self.emit(inst);
    }

    /// FMOV (register): Vd = Vn
    pub fn fmov(&mut self, vd: AArch64Reg, vn: AArch64Reg, double: bool) {
        let inst = (if double { 0b0001111001100000010000 } else { 0b0001111000100000010000 } << 10)
            | (vn.encoding() << 5)
            | vd.encoding();
        self.emit(inst);
    }

    // ========================================================================
    // PROLOGUE/EPILOGUE HELPERS
    // ========================================================================

    /// Emit function prologue
    /// Saves FP/LR and allocates stack frame
    pub fn emit_prologue(&mut self, frame_size: u32) {
        // Align frame size to 16 bytes
        let aligned_size = (frame_size + 15) & !15;
        self.frame_size = aligned_size;

        // STP X29, X30, [SP, #-frame_size]!
        let imm7 = -((aligned_size / 8) as i8);
        self.stp_pre(AArch64Reg::X29, AArch64Reg::X30, SP, imm7, true);

        // MOV X29, SP (set frame pointer)
        self.mov_reg(AArch64Reg::X29, SP, true);
    }

    /// Emit function epilogue
    /// Restores FP/LR and deallocates stack frame
    pub fn emit_epilogue(&mut self) {
        // LDP X29, X30, [SP], #frame_size
        let imm7 = (self.frame_size / 8) as i8;
        self.ldp_post(AArch64Reg::X29, AArch64Reg::X30, SP, imm7, true);

        // RET
        self.ret();
    }

    // ========================================================================
    // SIR COMPILATION (STUBS)
    // ========================================================================

    /// Emit a SIR module to AArch64 machine code
    pub fn emit_module(&mut self, _module: &SirModule) -> Result<Vec<u8>, String> {
        // TODO: Implement full SIR module compilation
        // For now, this is a placeholder
        Ok(self.code.clone())
    }

    /// Emit a function to AArch64 machine code
    pub fn emit_function(&mut self, _func: &SirFunction) -> Result<Vec<u8>, String> {
        // TODO: Implement SIR function emission
        // This will require:
        // 1. Register allocation
        // 2. Instruction selection from SIR ops
        // 3. Stack layout calculation
        // 4. Prologue/epilogue generation

        Ok(self.code.clone())
    }
}

impl Default for AArch64Emitter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_encoding() {
        assert_eq!(AArch64Reg::X0.encoding(), 0);
        assert_eq!(AArch64Reg::X30.encoding(), 30);
        assert_eq!(AArch64Reg::XZR.encoding(), 31);
        assert_eq!(AArch64Reg::V0.encoding(), 0);  // SIMD regs also use 0-31
    }

    #[test]
    fn test_add_imm_encoding() {
        let mut emit = AArch64Emitter::new();
        // ADD X0, X1, #42
        emit.add_imm(AArch64Reg::X0, AArch64Reg::X1, 42, true);
        let code = emit.finish().unwrap();

        // Expected: 0x91010820 = ADD X0, X1, #42
        assert_eq!(code.len(), 4);
        let inst = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        // Verify fields
        assert_eq!((inst >> 31) & 1, 1);      // sf = 1 (64-bit)
        assert_eq!((inst >> 10) & 0xFFF, 42); // imm12 = 42
        assert_eq!((inst >> 5) & 0x1F, 1);    // Rn = X1
        assert_eq!(inst & 0x1F, 0);           // Rd = X0
    }

    #[test]
    fn test_mov_imm64() {
        let mut emit = AArch64Emitter::new();

        // Test zero
        emit.mov_imm64(AArch64Reg::X0, 0);

        // Test small value
        emit.mov_imm64(AArch64Reg::X1, 0x1234);

        // Test large value requiring multiple MOVK
        emit.mov_imm64(AArch64Reg::X2, 0x1234_5678_9ABC_DEF0);

        let code = emit.finish().unwrap();
        assert!(code.len() >= 4);  // At least one instruction
    }

    #[test]
    fn test_branch_encoding() {
        let mut emit = AArch64Emitter::new();

        // Forward branch (will be resolved)
        emit.label("start");
        emit.add_imm(AArch64Reg::X0, AArch64Reg::X0, 1, true);
        emit.b_cond_label(Condition::NE, "end");
        emit.b_label("start");
        emit.label("end");
        emit.ret();

        let code = emit.finish().unwrap();
        assert_eq!(code.len(), 16);  // 4 instructions * 4 bytes
    }

    #[test]
    fn test_prologue_epilogue() {
        let mut emit = AArch64Emitter::new();
        emit.emit_prologue(32);
        emit.mov_imm64(AArch64Reg::X0, 42);
        emit.emit_epilogue();

        let code = emit.finish().unwrap();
        // Should have: STP, MOV, MOVZ (for imm), LDP, RET
        assert!(code.len() >= 16);
    }

    #[test]
    fn test_callee_saved() {
        assert!(AArch64Reg::X19.is_callee_saved());
        assert!(AArch64Reg::X28.is_callee_saved());
        assert!(AArch64Reg::V8.is_callee_saved());
        assert!(!AArch64Reg::X0.is_callee_saved());
        assert!(!AArch64Reg::X16.is_callee_saved());
    }
}
