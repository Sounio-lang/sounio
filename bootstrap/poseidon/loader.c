/* loader.c - SOIR v1 Bytecode Deserializer */

#include "loader.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOIR_MAGIC 0x52494F53  /* "SOIR" in little-endian */
#define SOIR_VERSION 1
#define MAX_FILE_SIZE (128 * 1024)

/* Read little-endian int64 */
static int64_t read_i64(const uint8_t *buf, size_t *pos) {
    uint64_t val = 0;
    for (int i = 0; i < 8; i++) {
        val |= ((uint64_t)buf[*pos + i]) << (i * 8);
    }
    *pos += 8;
    /* Convert from little-endian to host byte order */
    val = platform_le64_to_host(val);
    return (int64_t)val;
}

/* Read little-endian int8 */
static int8_t read_i8(const uint8_t *buf, size_t *pos) {
    int8_t val = (int8_t)buf[*pos];
    (*pos)++;
    return val;
}

/* Read double (as raw i64 bits, little-endian) */
static double read_f64(const uint8_t *buf, size_t *pos) {
    union {
        uint64_t u;
        double d;
    } conv;
    /* Read as little-endian uint64 */
    conv.u = 0;
    for (int i = 0; i < 8; i++) {
        conv.u |= ((uint64_t)buf[*pos + i]) << (i * 8);
    }
    *pos += 8;
    /* Convert from little-endian to host byte order */
    conv.u = platform_le64_to_host(conv.u);
    return conv.d;
}

/* Read Name structure */
static Name read_name(const uint8_t *buf, size_t *pos) {
    Name n;
    n.len = read_i64(buf, pos);
    memcpy(n.buf, buf + *pos, NAME_BUF_SIZE);
    *pos += NAME_BUF_SIZE;
    return n;
}

/* Read Opcode */
static Opcode read_opcode(const uint8_t *buf, size_t *pos) {
    int8_t tag = read_i8(buf, pos);
    /* 7 bytes padding */
    *pos += 7;
    return (Opcode)tag;
}

/* Read BinaryOp */
static BinaryOp read_binop(const uint8_t *buf, size_t *pos) {
    int8_t tag = read_i8(buf, pos);
    /* 7 bytes padding */
    *pos += 7;
    return (BinaryOp)tag;
}

/* Read UnaryOp */
static UnaryOp read_unop(const uint8_t *buf, size_t *pos) {
    int8_t tag = read_i8(buf, pos);
    /* 7 bytes padding */
    *pos += 7;
    return (UnaryOp)tag;
}

/* Read Instruction */
static Instruction read_instruction(const uint8_t *buf, size_t *pos) {
    Instruction ins;
    ins.op = read_opcode(buf, pos);
    ins.dst = read_i64(buf, pos);
    ins.src1 = read_i64(buf, pos);
    ins.src2 = read_i64(buf, pos);
    ins.imm_i64 = read_i64(buf, pos);
    ins.imm_f64 = read_f64(buf, pos);
    ins.label_id = read_i64(buf, pos);
    ins.fn_id = read_i64(buf, pos);
    ins.field_idx = read_i64(buf, pos);
    ins.bin_op = read_binop(buf, pos);
    ins.un_op = read_unop(buf, pos);
    ins.name = read_name(buf, pos);
    ins.arg_count = read_i64(buf, pos);
    return ins;
}

/* Read Function */
static Function read_function(const uint8_t *buf, size_t *pos) {
    Function func;
    func.name = read_name(buf, pos);
    func.instr_count = read_i64(buf, pos);
    func.reg_count = read_i64(buf, pos);
    func.label_count = read_i64(buf, pos);
    func.param_count = read_i64(buf, pos);

    /* Read param_regs array */
    for (int i = 0; i < MAX_PARAMS; i++) {
        func.param_regs[i] = read_i64(buf, pos);
    }

    /* Allocate and read instructions */
    func.instrs = platform_calloc(func.instr_count, sizeof(Instruction));
    if (!func.instrs) {
        fprintf(stderr, "Failed to allocate instructions\n");
        exit(1);
    }

    for (int64_t i = 0; i < func.instr_count; i++) {
        func.instrs[i] = read_instruction(buf, pos);
    }

    return func;
}

/* Load SOIR from buffer */
Module* load_soir_buffer(const uint8_t *buf, size_t size) {
    size_t pos = 0;

    /* Read header */
    if (size < 8) {
        fprintf(stderr, "File too small\n");
        return NULL;
    }

    uint32_t magic = 0;
    for (int i = 0; i < 4; i++) {
        magic |= ((uint32_t)buf[pos++]) << (i * 8);
    }

    if (magic != SOIR_MAGIC) {
        fprintf(stderr, "Invalid magic: 0x%08x (expected 0x%08x)\n", magic, SOIR_MAGIC);
        return NULL;
    }

    uint8_t version = buf[pos++];
    if (version != SOIR_VERSION) {
        fprintf(stderr, "Unsupported version: %d\n", version);
        return NULL;
    }

    /* Skip 3 reserved bytes */
    pos += 3;

    /* Allocate module */
    Module *module = platform_calloc(1, sizeof(Module));
    if (!module) {
        fprintf(stderr, "Failed to allocate module\n");
        return NULL;
    }

    /* Read function count */
    module->fn_count = read_i64(buf, &pos);

    /* Allocate functions array */
    module->functions = platform_calloc(module->fn_count, sizeof(Function));
    if (!module->functions) {
        fprintf(stderr, "Failed to allocate functions\n");
        platform_free(module);
        return NULL;
    }

    /* Read functions */
    for (int64_t i = 0; i < module->fn_count; i++) {
        module->functions[i] = read_function(buf, &pos);
    }

    /* Read string count */
    module->string_count = read_i64(buf, &pos);

    /* Allocate string table */
    if (module->string_count > 0) {
        module->string_table = platform_calloc(module->string_count, sizeof(Name));
        if (!module->string_table) {
            fprintf(stderr, "Failed to allocate string table\n");
            for (int64_t i = 0; i < module->fn_count; i++) {
                platform_free(module->functions[i].instrs);
            }
            platform_free(module->functions);
            platform_free(module);
            return NULL;
        }

        for (int64_t i = 0; i < module->string_count; i++) {
            module->string_table[i] = read_name(buf, &pos);
        }
    } else {
        module->string_table = NULL;
    }

    return module;
}

/* Load SOIR from file */
Module* load_soir_file(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file");
        return NULL;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize > MAX_FILE_SIZE) {
        fprintf(stderr, "File too large: %ld bytes (max %d)\n", fsize, MAX_FILE_SIZE);
        fclose(f);
        return NULL;
    }

    /* Read file into buffer */
    uint8_t *buf = platform_malloc(fsize);
    if (!buf) {
        fprintf(stderr, "Failed to allocate buffer\n");
        fclose(f);
        return NULL;
    }

    size_t nread = fread(buf, 1, fsize, f);
    fclose(f);

    if (nread != (size_t)fsize) {
        fprintf(stderr, "Read error\n");
        platform_free(buf);
        return NULL;
    }

    /* Deserialize */
    Module *module = load_soir_buffer(buf, fsize);
    platform_free(buf);

    return module;
}

/* Free module */
void free_module(Module *module) {
    if (!module) return;

    for (int64_t i = 0; i < module->fn_count; i++) {
        platform_free(module->functions[i].instrs);
    }
    platform_free(module->functions);
    platform_free(module->string_table);
    platform_free(module);
}
