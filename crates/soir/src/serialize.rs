//! SOIR binary serialization
//!
//! This module provides serialization of IR modules to the SOIR binary format.

use crate::{
    IrFunction, IrInstr, Result, SOIR_MAGIC, SOIR_MAX_SIZE, SOIR_VERSION, SoirError, SoirModule,
};
use std::io::Write;

/// Serialize a SOIR module to bytes
pub fn serialize(module: &SoirModule) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(8192);

    // Write header
    buf.write_all(&SOIR_MAGIC)?;
    buf.write_all(&[SOIR_VERSION, 0, 0, 0])?;

    // Write function count
    write_i64(&mut buf, module.fn_count)?;

    // Write functions
    for i in 0..(module.fn_count as usize) {
        serialize_function(&mut buf, &module.functions[i])?;
    }

    // Write string count
    write_i64(&mut buf, module.string_count)?;

    // Write string table
    for i in 0..(module.string_count as usize) {
        write_name(&mut buf, &module.string_table[i])?;
    }

    if buf.len() > SOIR_MAX_SIZE {
        return Err(SoirError::ModuleTooLarge(buf.len(), SOIR_MAX_SIZE));
    }

    Ok(buf)
}

fn serialize_function(buf: &mut Vec<u8>, func: &IrFunction) -> Result<()> {
    write_name(buf, &func.name)?;
    write_i64(buf, func.instr_count)?;
    write_i64(buf, func.reg_count)?;
    write_i64(buf, func.label_count)?;
    write_i64(buf, func.param_count)?;

    // Write param_regs array
    for reg in &func.param_regs {
        write_i64(buf, *reg)?;
    }

    // Write instructions
    for i in 0..(func.instr_count as usize) {
        serialize_instr(buf, &func.instrs[i])?;
    }

    Ok(())
}

fn serialize_instr(buf: &mut Vec<u8>, instr: &IrInstr) -> Result<()> {
    // Opcode (1 byte + 7 padding)
    buf.write_all(&[instr.op.to_u8()])?;
    buf.write_all(&[0; 7])?;

    // Register fields
    write_i64(buf, instr.dst)?;
    write_i64(buf, instr.src1)?;
    write_i64(buf, instr.src2)?;

    // Immediate values
    write_i64(buf, instr.imm_i64)?;
    write_f64(buf, instr.imm_f64)?;

    // Label and function IDs
    write_i64(buf, instr.label_id)?;
    write_i64(buf, instr.fn_id)?;
    write_i64(buf, instr.field_idx)?;

    // Binary op (1 byte + 7 padding)
    buf.write_all(&[instr.bin_op.to_u8()])?;
    buf.write_all(&[0; 7])?;

    // Unary op (1 byte + 7 padding)
    buf.write_all(&[instr.un_op.to_u8()])?;
    buf.write_all(&[0; 7])?;

    // Name field
    write_name(buf, &instr.name)?;

    // Arg count
    write_i64(buf, instr.arg_count)?;

    Ok(())
}

fn write_i64(buf: &mut Vec<u8>, val: i64) -> Result<()> {
    let bytes = val.to_le_bytes();
    buf.write_all(&bytes)?;
    Ok(())
}

fn write_f64(buf: &mut Vec<u8>, val: f64) -> Result<()> {
    let bytes = val.to_le_bytes();
    buf.write_all(&bytes)?;
    Ok(())
}

fn write_name(buf: &mut Vec<u8>, name: &crate::Name) -> Result<()> {
    // Write length
    write_i64(buf, name.len as i64)?;

    // Write buffer (always 128 bytes)
    buf.write_all(&name.buf)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_serialize_empty_module() {
        let module = SoirModule::empty();
        let bytes = serialize(&module).unwrap();

        // Header (8 bytes) + fn_count (8) + string_count (8) = 24 bytes minimum
        assert!(bytes.len() >= 24);

        // Check magic
        assert_eq!(&bytes[0..4], b"SOIR");

        // Check version
        assert_eq!(bytes[4], 1);
    }

    #[test]
    fn test_serialize_simple_function() {
        let mut module = SoirModule::empty();
        let mut func = IrFunction::empty();
        func.name = Name::from("main");
        func.instr_count = 1;
        func.instrs[0] = IrInstr {
            op: IrOpcode::IrReturn,
            src1: 0,
            ..IrInstr::nop()
        };

        module.functions[0] = func;
        module.fn_count = 1;

        let bytes = serialize(&module).unwrap();
        assert!(bytes.len() > 24);
    }
}
