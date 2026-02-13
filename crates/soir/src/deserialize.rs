//! SOIR binary deserialization
//!
//! This module provides deserialization of the SOIR binary format.

use crate::{
    SoirError, Result, SoirModule, IrFunction, IrInstr, IrOpcode, BinaryOp, UnaryOp, Name,
    SOIR_MAGIC, SOIR_VERSION, SOIR_MAX_SIZE, IR_MAX_FUNCS, IR_MAX_STRINGS, IR_MAX_INSTRS,
    IR_MAX_PARAMS,
};

/// Deserialize a SOIR module from bytes
pub fn deserialize(data: &[u8]) -> Result<SoirModule> {
    if data.len() > SOIR_MAX_SIZE {
        return Err(SoirError::ModuleTooLarge(data.len(), SOIR_MAX_SIZE));
    }

    let mut cursor = Cursor::new(data);

    // Read and validate header
    let magic = cursor.read_bytes(4)?;
    if magic != SOIR_MAGIC {
        let mut bad_magic = [0u8; 4];
        bad_magic.copy_from_slice(magic);
        return Err(SoirError::InvalidMagic(bad_magic));
    }

    let version = cursor.read_u8()?;
    if version != SOIR_VERSION {
        return Err(SoirError::UnsupportedVersion(version, SOIR_VERSION));
    }

    // Skip reserved bytes
    cursor.skip(3)?;

    // Read function count
    let fn_count = cursor.read_i64()?;
    if fn_count < 0 || fn_count as usize > IR_MAX_FUNCS {
        return Err(SoirError::FunctionCountOverflow(fn_count, IR_MAX_FUNCS));
    }

    // Read functions
    let mut functions = vec![IrFunction::empty(); IR_MAX_FUNCS];
    for i in 0..(fn_count as usize) {
        functions[i] = deserialize_function(&mut cursor)?;
    }

    // Read string count
    let string_count = cursor.read_i64()?;
    if string_count < 0 || string_count as usize > IR_MAX_STRINGS {
        return Err(SoirError::StringCountOverflow(string_count, IR_MAX_STRINGS));
    }

    // Read string table
    let mut string_table = vec![Name::new(); IR_MAX_STRINGS];
    for i in 0..(string_count as usize) {
        string_table[i] = read_name(&mut cursor)?;
    }

    Ok(SoirModule {
        functions,
        fn_count,
        string_table,
        string_count,
    })
}

fn deserialize_function(cursor: &mut Cursor) -> Result<IrFunction> {
    let name = read_name(cursor)?;
    let instr_count = cursor.read_i64()?;
    let reg_count = cursor.read_i64()?;
    let label_count = cursor.read_i64()?;
    let param_count = cursor.read_i64()?;

    if instr_count < 0 || instr_count as usize > IR_MAX_INSTRS {
        return Err(SoirError::InstrCountOverflow(instr_count, IR_MAX_INSTRS));
    }

    // Read param_regs
    let mut param_regs = [crate::IR_INVALID_REG; IR_MAX_PARAMS];
    for i in 0..IR_MAX_PARAMS {
        param_regs[i] = cursor.read_i64()?;
    }

    // Read instructions
    let mut instrs = vec![IrInstr::nop(); IR_MAX_INSTRS];
    for i in 0..(instr_count as usize) {
        instrs[i] = deserialize_instr(cursor)?;
    }

    Ok(IrFunction {
        name,
        instrs,
        instr_count,
        reg_count,
        label_count,
        param_count,
        param_regs,
    })
}

fn deserialize_instr(cursor: &mut Cursor) -> Result<IrInstr> {
    // Read opcode
    let op_byte = cursor.read_u8()?;
    let op = IrOpcode::from_u8(op_byte)
        .ok_or(SoirError::InvalidOpcode(op_byte))?;
    cursor.skip(7)?; // padding

    // Read register fields
    let dst = cursor.read_i64()?;
    let src1 = cursor.read_i64()?;
    let src2 = cursor.read_i64()?;

    // Read immediate values
    let imm_i64 = cursor.read_i64()?;
    let imm_f64 = cursor.read_f64()?;

    // Read label and function IDs
    let label_id = cursor.read_i64()?;
    let fn_id = cursor.read_i64()?;
    let field_idx = cursor.read_i64()?;

    // Read binary op
    let bin_op_byte = cursor.read_u8()?;
    let bin_op = BinaryOp::from_u8(bin_op_byte)
        .ok_or(SoirError::InvalidBinaryOp(bin_op_byte))?;
    cursor.skip(7)?; // padding

    // Read unary op
    let un_op_byte = cursor.read_u8()?;
    let un_op = UnaryOp::from_u8(un_op_byte)
        .ok_or(SoirError::InvalidUnaryOp(un_op_byte))?;
    cursor.skip(7)?; // padding

    // Read name
    let name = read_name(cursor)?;

    // Read arg count
    let arg_count = cursor.read_i64()?;

    Ok(IrInstr {
        op,
        dst,
        src1,
        src2,
        imm_i64,
        imm_f64,
        label_id,
        fn_id,
        field_idx,
        bin_op,
        un_op,
        name,
        call_args: None, // Not serialized
        arg_count,
    })
}

fn read_name(cursor: &mut Cursor) -> Result<Name> {
    let len = cursor.read_i64()?;
    let buf_bytes = cursor.read_bytes(128)?;

    let mut buf = [0u8; 128];
    buf.copy_from_slice(buf_bytes);

    Ok(Name {
        buf,
        len: len as usize,
    })
}

// Simple cursor for reading bytes
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(SoirError::UnexpectedEof(self.pos));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0])
    }

    fn read_i64(&mut self) -> Result<i64> {
        let bytes = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(i64::from_le_bytes(arr))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let bytes = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(f64::from_le_bytes(arr))
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.data.len() {
            return Err(SoirError::UnexpectedEof(self.pos));
        }
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::serialize;
    use crate::types::*;

    #[test]
    fn test_roundtrip_empty_module() {
        let module = SoirModule::empty();
        let bytes = serialize(&module).unwrap();
        let deserialized = deserialize(&bytes).unwrap();

        assert_eq!(module.fn_count, deserialized.fn_count);
        assert_eq!(module.string_count, deserialized.string_count);
    }

    #[test]
    fn test_roundtrip_with_function() {
        let mut module = SoirModule::empty();
        let mut func = IrFunction::empty();
        func.name = Name::from("test");
        func.instr_count = 2;
        func.instrs[0] = IrInstr {
            op: IrOpcode::IrLoadImm,
            dst: 0,
            imm_i64: 42,
            ..IrInstr::nop()
        };
        func.instrs[1] = IrInstr {
            op: IrOpcode::IrReturn,
            src1: 0,
            ..IrInstr::nop()
        };

        module.functions[0] = func.clone();
        module.fn_count = 1;

        let bytes = serialize(&module).unwrap();
        let deserialized = deserialize(&bytes).unwrap();

        assert_eq!(deserialized.fn_count, 1);
        let dfunc = &deserialized.functions[0];
        assert_eq!(dfunc.name.as_str().unwrap(), "test");
        assert_eq!(dfunc.instr_count, 2);
        assert_eq!(dfunc.instrs[0].op, IrOpcode::IrLoadImm);
        assert_eq!(dfunc.instrs[0].imm_i64, 42);
    }

    #[test]
    fn test_invalid_magic() {
        let bad_data = b"BAD\x00\x01\x00\x00\x00";
        let result = deserialize(bad_data);
        assert!(matches!(result, Err(SoirError::InvalidMagic(_))));
    }

    #[test]
    fn test_invalid_version() {
        let bad_data = b"SOIR\x99\x00\x00\x00";
        let result = deserialize(bad_data);
        assert!(matches!(result, Err(SoirError::UnsupportedVersion(_, _))));
    }
}
