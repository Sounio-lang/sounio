#!/usr/bin/env python3
"""Generate hand-crafted SOIR v1 test files"""

import struct
import sys

SOIR_MAGIC = 0x52494F53  # "SOIR" in little-endian
SOIR_VERSION = 1

# Opcodes
OP_LOAD_IMM = 0
OP_LOAD_BOOL = 2
OP_COPY = 4
OP_BINOP = 5
OP_UNARYOP = 6
OP_CALL = 7
OP_RETURN = 8
OP_JUMP = 9
OP_BRANCH_TRUE = 10
OP_BRANCH_FALSE = 11
OP_LABEL = 17
OP_NOP = 18

# Binary ops
BINOP_ADD = 0
BINOP_SUB = 1
BINOP_MUL = 2
BINOP_LT = 7

# Unary ops
UNOP_NEG = 0

def write_i64(f, val):
    f.write(struct.pack('<q', val))

def write_i8(f, val):
    f.write(struct.pack('<b', val))

def write_f64(f, val):
    f.write(struct.pack('<d', val))

def write_name(f, s):
    """Write Name structure: i64 len + 128 bytes buf"""
    write_i64(f, len(s))
    buf = s.encode('utf-8')[:128]
    f.write(buf + b'\x00' * (128 - len(buf)))

def write_instr(f, op, dst=-1, src1=-1, src2=-1, imm_i64=0, imm_f64=0.0,
                label_id=-1, fn_id=-1, field_idx=-1, bin_op=0, un_op=0,
                name="", arg_count=0):
    """Write Instruction (fixed 128 bytes in memory, variable in binary)"""
    # op: i8 + 7 padding
    write_i8(f, op)
    f.write(b'\x00' * 7)
    # dst, src1, src2
    write_i64(f, dst)
    write_i64(f, src1)
    write_i64(f, src2)
    # imm_i64, imm_f64
    write_i64(f, imm_i64)
    write_f64(f, imm_f64)
    # label_id, fn_id, field_idx
    write_i64(f, label_id)
    write_i64(f, fn_id)
    write_i64(f, field_idx)
    # bin_op: i8 + 7 padding
    write_i8(f, bin_op)
    f.write(b'\x00' * 7)
    # un_op: i8 + 7 padding
    write_i8(f, un_op)
    f.write(b'\x00' * 7)
    # name: Name structure
    write_name(f, name)
    # arg_count
    write_i64(f, arg_count)

def write_function(f, name, instrs, reg_count, label_count=0, param_count=0, param_regs=None):
    """Write Function"""
    write_name(f, name)
    write_i64(f, len(instrs))
    write_i64(f, reg_count)
    write_i64(f, label_count)
    write_i64(f, param_count)
    # param_regs: [i64; 64]
    if param_regs is None:
        param_regs = []
    for i in range(64):
        write_i64(f, param_regs[i] if i < len(param_regs) else -1)
    # instructions
    for instr in instrs:
        write_instr(f, **instr)

def write_header(f):
    """Write SOIR header"""
    f.write(struct.pack('<I', SOIR_MAGIC))
    f.write(struct.pack('<B', SOIR_VERSION))
    f.write(b'\x00' * 3)  # reserved

def gen_add_soir():
    """T01: fn main() -> i64 { 10 + 32 }"""
    with open('add.soir', 'wb') as f:
        write_header(f)
        write_i64(f, 1)  # fn_count

        instrs = [
            {'op': OP_LOAD_IMM, 'dst': 0, 'imm_i64': 10},
            {'op': OP_LOAD_IMM, 'dst': 1, 'imm_i64': 32},
            {'op': OP_BINOP, 'dst': 2, 'src1': 0, 'src2': 1, 'bin_op': BINOP_ADD},
            {'op': OP_RETURN, 'src1': 2},
        ]
        write_function(f, "main", instrs, reg_count=3)

        write_i64(f, 0)  # string_count

def gen_call_soir():
    """T02: fn add(a,b) { a+b }; fn main() { add(10, 32) }"""
    with open('call.soir', 'wb') as f:
        write_header(f)
        write_i64(f, 2)  # fn_count

        # Function 0: add
        add_instrs = [
            {'op': OP_BINOP, 'dst': 2, 'src1': 0, 'src2': 1, 'bin_op': BINOP_ADD},
            {'op': OP_RETURN, 'src1': 2},
        ]
        write_function(f, "add", add_instrs, reg_count=3, param_count=2, param_regs=[0, 1])

        # Function 1: main
        main_instrs = [
            {'op': OP_LOAD_IMM, 'dst': 0, 'imm_i64': 10},
            {'op': OP_LOAD_IMM, 'dst': 1, 'imm_i64': 32},
            {'op': OP_CALL, 'dst': 2, 'src1': 0, 'src2': 1, 'fn_id': 0, 'name': 'add', 'arg_count': 2},
            {'op': OP_RETURN, 'src1': 2},
        ]
        write_function(f, "main", main_instrs, reg_count=3)

        write_i64(f, 0)  # string_count

def gen_branch_soir():
    """T03: if (42 < 50) { 1 } else { 0 }"""
    with open('branch.soir', 'wb') as f:
        write_header(f)
        write_i64(f, 1)  # fn_count

        instrs = [
            {'op': OP_LOAD_IMM, 'dst': 0, 'imm_i64': 42},
            {'op': OP_LOAD_IMM, 'dst': 1, 'imm_i64': 50},
            {'op': OP_BINOP, 'dst': 2, 'src1': 0, 'src2': 1, 'bin_op': BINOP_LT},
            {'op': OP_BRANCH_TRUE, 'src1': 2, 'label_id': 0},
            # else:
            {'op': OP_LOAD_IMM, 'dst': 3, 'imm_i64': 0},
            {'op': OP_JUMP, 'label_id': 1},
            # then:
            {'op': OP_LABEL, 'label_id': 0},
            {'op': OP_LOAD_IMM, 'dst': 3, 'imm_i64': 1},
            # end:
            {'op': OP_LABEL, 'label_id': 1},
            {'op': OP_RETURN, 'src1': 3},
        ]
        write_function(f, "main", instrs, reg_count=4, label_count=2)

        write_i64(f, 0)  # string_count

def gen_loop_soir():
    """T04: sum = 0; i = 0; while (i < 5) { sum += i; i += 1 }"""
    with open('loop.soir', 'wb') as f:
        write_header(f)
        write_i64(f, 1)  # fn_count

        instrs = [
            {'op': OP_LOAD_IMM, 'dst': 0, 'imm_i64': 0},  # sum = 0
            {'op': OP_LOAD_IMM, 'dst': 1, 'imm_i64': 0},  # i = 0
            # loop start:
            {'op': OP_LABEL, 'label_id': 0},
            {'op': OP_LOAD_IMM, 'dst': 2, 'imm_i64': 5},
            {'op': OP_BINOP, 'dst': 3, 'src1': 1, 'src2': 2, 'bin_op': BINOP_LT},
            {'op': OP_BRANCH_FALSE, 'src1': 3, 'label_id': 1},  # exit if !(i < 5)
            # body:
            {'op': OP_BINOP, 'dst': 4, 'src1': 0, 'src2': 1, 'bin_op': BINOP_ADD},
            {'op': OP_COPY, 'dst': 0, 'src1': 4},  # sum = sum + i
            {'op': OP_LOAD_IMM, 'dst': 5, 'imm_i64': 1},
            {'op': OP_BINOP, 'dst': 6, 'src1': 1, 'src2': 5, 'bin_op': BINOP_ADD},
            {'op': OP_COPY, 'dst': 1, 'src1': 6},  # i = i + 1
            {'op': OP_JUMP, 'label_id': 0},
            # exit:
            {'op': OP_LABEL, 'label_id': 1},
            {'op': OP_RETURN, 'src1': 0},
        ]
        write_function(f, "main", instrs, reg_count=7, label_count=2)

        write_i64(f, 0)  # string_count

def gen_return_soir():
    """T05: fn main() -> i64 { 42 }"""
    with open('return.soir', 'wb') as f:
        write_header(f)
        write_i64(f, 1)  # fn_count

        instrs = [
            {'op': OP_LOAD_IMM, 'dst': 0, 'imm_i64': 42},
            {'op': OP_RETURN, 'src1': 0},
        ]
        write_function(f, "main", instrs, reg_count=1)

        write_i64(f, 0)  # string_count

if __name__ == '__main__':
    gen_add_soir()
    gen_call_soir()
    gen_branch_soir()
    gen_loop_soir()
    gen_return_soir()
    print("Generated 5 test SOIR files")
