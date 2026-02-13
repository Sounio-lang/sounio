/* opcodes.h - SOIR v1 Opcode Definitions */

#ifndef OPCODES_H
#define OPCODES_H

#include <stdint.h>

/* SOIR v1 Opcodes (matching self-hosted IR) */
typedef enum {
    OP_LOAD_IMM = 0,
    OP_LOAD_FLOAT = 1,
    OP_LOAD_BOOL = 2,
    OP_LOAD_STRING = 3,
    OP_COPY = 4,
    OP_BINOP = 5,
    OP_UNARYOP = 6,
    OP_CALL = 7,
    OP_RETURN = 8,
    OP_JUMP = 9,
    OP_BRANCH_TRUE = 10,
    OP_BRANCH_FALSE = 11,
    OP_FIELD_GET = 12,
    OP_FIELD_SET = 13,
    OP_INDEX_GET = 14,
    OP_INDEX_SET = 15,
    OP_ALLOC = 16,
    OP_LABEL = 17,
    OP_NOP = 18,
    OP_PHI = 19
} Opcode;

/* Binary operators */
typedef enum {
    BINOP_ADD = 0,
    BINOP_SUB = 1,
    BINOP_MUL = 2,
    BINOP_DIV = 3,
    BINOP_MOD = 4,
    BINOP_EQ = 5,
    BINOP_NE = 6,
    BINOP_LT = 7,
    BINOP_LE = 8,
    BINOP_GT = 9,
    BINOP_GE = 10,
    BINOP_AND = 11,
    BINOP_OR = 12,
    BINOP_BITAND = 13,
    BINOP_BITOR = 14,
    BINOP_BITXOR = 15,
    BINOP_SHL = 16,
    BINOP_SHR = 17
} BinaryOp;

/* Unary operators */
typedef enum {
    UNOP_NEG = 0,
    UNOP_NOT = 1
} UnaryOp;

#endif /* OPCODES_H */
