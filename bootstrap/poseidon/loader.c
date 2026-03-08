/* loader.c - SOIR v1 Bytecode Deserializer */

#include "loader.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOIR_MAGIC 0x52494F53  /* "SOIR" in little-endian */
#define SOIR_VERSION 1
#define MAX_FILE_SIZE (128 * 1024)
#define MAX_STRING_TABLE_ENTRIES 4096

typedef struct {
    const uint8_t *buf;
    size_t size;
    size_t pos;
    LoaderStatus status;
} SoirReader;

static bool reader_take(SoirReader *reader, size_t count, const uint8_t **out) {
    if (reader->status != LOADER_STATUS_OK) {
        return false;
    }
    if (count > reader->size - reader->pos) {
        reader->status = LOADER_STATUS_INVALID_BYTECODE;
        return false;
    }
    *out = reader->buf + reader->pos;
    reader->pos += count;
    return true;
}

/* Read little-endian int64 */
static int64_t read_i64(SoirReader *reader) {
    const uint8_t *data;
    uint64_t val = 0;
    int i;

    if (!reader_take(reader, 8, &data)) {
        return 0;
    }
    for (i = 0; i < 8; i++) {
        val |= ((uint64_t)data[i]) << (i * 8);
    }
    val = platform_le64_to_host(val);
    return (int64_t)val;
}

/* Read little-endian int8 */
static int8_t read_i8(SoirReader *reader) {
    const uint8_t *data;
    if (!reader_take(reader, 1, &data)) {
        return 0;
    }
    return (int8_t)data[0];
}

/* Read double (as raw i64 bits, little-endian) */
static double read_f64(SoirReader *reader) {
    const uint8_t *data;
    union {
        uint64_t u;
        double d;
    } conv;
    int i;

    conv.u = 0;
    if (!reader_take(reader, 8, &data)) {
        return 0.0;
    }
    for (i = 0; i < 8; i++) {
        conv.u |= ((uint64_t)data[i]) << (i * 8);
    }
    conv.u = platform_le64_to_host(conv.u);
    return conv.d;
}

/* Read Name structure */
static void read_name(SoirReader *reader, Name *out) {
    const uint8_t *data;

    memset(out, 0, sizeof(*out));
    out->len = read_i64(reader);
    if (reader->status != LOADER_STATUS_OK) {
        return;
    }
    if (out->len < 0 || out->len > NAME_BUF_SIZE) {
        reader->status = LOADER_STATUS_INVALID_BYTECODE;
        return;
    }
    if (!reader_take(reader, NAME_BUF_SIZE, &data)) {
        return;
    }
    memcpy(out->buf, data, NAME_BUF_SIZE);
}

/* Read Opcode */
static Opcode read_opcode(SoirReader *reader) {
    const uint8_t *padding;
    Opcode tag = (Opcode)read_i8(reader);
    if (!reader_take(reader, 7, &padding)) {
        return OP_NOP;
    }
    return tag;
}

/* Read BinaryOp */
static BinaryOp read_binop(SoirReader *reader) {
    const uint8_t *padding;
    BinaryOp tag = (BinaryOp)read_i8(reader);
    if (!reader_take(reader, 7, &padding)) {
        return BINOP_ADD;
    }
    return tag;
}

/* Read UnaryOp */
static UnaryOp read_unop(SoirReader *reader) {
    const uint8_t *padding;
    UnaryOp tag = (UnaryOp)read_i8(reader);
    if (!reader_take(reader, 7, &padding)) {
        return UNOP_NEG;
    }
    return tag;
}

/* Read Instruction */
static void read_instruction(SoirReader *reader, Instruction *out) {
    memset(out, 0, sizeof(*out));
    out->op = read_opcode(reader);
    out->dst = read_i64(reader);
    out->src1 = read_i64(reader);
    out->src2 = read_i64(reader);
    out->imm_i64 = read_i64(reader);
    out->imm_f64 = read_f64(reader);
    out->label_id = read_i64(reader);
    out->fn_id = read_i64(reader);
    out->field_idx = read_i64(reader);
    out->bin_op = read_binop(reader);
    out->un_op = read_unop(reader);
    read_name(reader, &out->name);
    out->arg_count = read_i64(reader);
    if (reader->status != LOADER_STATUS_OK) {
        return;
    }
    if (out->arg_count < 0 || out->arg_count > MAX_PARAMS) {
        reader->status = LOADER_STATUS_INVALID_BYTECODE;
    }
}

/* Read Function */
static LoaderStatus read_function(SoirReader *reader, Function *func) {
    int i;
    int64_t instr_idx;

    memset(func, 0, sizeof(*func));
    read_name(reader, &func->name);
    func->instr_count = read_i64(reader);
    func->reg_count = read_i64(reader);
    func->label_count = read_i64(reader);
    func->param_count = read_i64(reader);
    if (reader->status != LOADER_STATUS_OK) {
        return reader->status;
    }
    if (func->instr_count < 0 || func->instr_count > MAX_INSTRUCTIONS ||
        func->reg_count < 0 || func->reg_count > MAX_REGISTERS ||
        func->label_count < 0 || func->label_count > MAX_INSTRUCTIONS ||
        func->param_count < 0 || func->param_count > MAX_PARAMS) {
        return LOADER_STATUS_INVALID_BYTECODE;
    }

    for (i = 0; i < MAX_PARAMS; i++) {
        func->param_regs[i] = read_i64(reader);
    }
    if (reader->status != LOADER_STATUS_OK) {
        return reader->status;
    }

    if (func->instr_count > 0) {
        func->instrs = platform_calloc((size_t)func->instr_count, sizeof(Instruction));
        if (!func->instrs) {
            return LOADER_STATUS_OUT_OF_MEMORY;
        }
    }

    for (instr_idx = 0; instr_idx < func->instr_count; instr_idx++) {
        read_instruction(reader, &func->instrs[instr_idx]);
        if (reader->status != LOADER_STATUS_OK) {
            return reader->status;
        }
    }

    return LOADER_STATUS_OK;
}

const char *loader_status_message(LoaderStatus status) {
    switch (status) {
        case LOADER_STATUS_OK:
            return "ok";
        case LOADER_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case LOADER_STATUS_IO_ERROR:
            return "i/o error";
        case LOADER_STATUS_FILE_TOO_LARGE:
            return "file too large";
        case LOADER_STATUS_INVALID_BYTECODE:
            return "invalid bytecode";
        case LOADER_STATUS_UNSUPPORTED_VERSION:
            return "unsupported version";
        case LOADER_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        default:
            return "unknown loader status";
    }
}

LoaderStatus load_soir_buffer_checked(const uint8_t *buf, size_t size, Module **out_module) {
    SoirReader reader;
    Module *module;
    const uint8_t *data;
    uint32_t magic = 0;
    uint8_t version;
    int i;
    int64_t fn_idx;
    int64_t str_idx;

    if (!out_module) {
        return LOADER_STATUS_INVALID_ARGUMENT;
    }
    *out_module = NULL;
    if (!buf || size == 0) {
        return LOADER_STATUS_INVALID_ARGUMENT;
    }

    reader.buf = buf;
    reader.size = size;
    reader.pos = 0;
    reader.status = LOADER_STATUS_OK;

    if (!reader_take(&reader, 4, &data)) {
        return reader.status;
    }
    for (i = 0; i < 4; i++) {
        magic |= ((uint32_t)data[i]) << (i * 8);
    }
    if (magic != SOIR_MAGIC) {
        return LOADER_STATUS_INVALID_BYTECODE;
    }

    if (!reader_take(&reader, 1, &data)) {
        return reader.status;
    }
    version = data[0];
    if (version != SOIR_VERSION) {
        return LOADER_STATUS_UNSUPPORTED_VERSION;
    }

    if (!reader_take(&reader, 3, &data)) {
        return reader.status;
    }

    module = platform_calloc(1, sizeof(Module));
    if (!module) {
        return LOADER_STATUS_OUT_OF_MEMORY;
    }

    module->fn_count = read_i64(&reader);
    if (reader.status != LOADER_STATUS_OK) {
        free_module(module);
        return reader.status;
    }
    if (module->fn_count < 0 || module->fn_count > MAX_FUNCTIONS) {
        free_module(module);
        return LOADER_STATUS_INVALID_BYTECODE;
    }

    if (module->fn_count > 0) {
        module->functions = platform_calloc((size_t)module->fn_count, sizeof(Function));
        if (!module->functions) {
            free_module(module);
            return LOADER_STATUS_OUT_OF_MEMORY;
        }
    }

    for (fn_idx = 0; fn_idx < module->fn_count; fn_idx++) {
        LoaderStatus fn_status = read_function(&reader, &module->functions[fn_idx]);
        if (fn_status != LOADER_STATUS_OK) {
            free_module(module);
            return fn_status;
        }
    }

    module->string_count = read_i64(&reader);
    if (reader.status != LOADER_STATUS_OK) {
        free_module(module);
        return reader.status;
    }
    if (module->string_count < 0 || module->string_count > MAX_STRING_TABLE_ENTRIES) {
        free_module(module);
        return LOADER_STATUS_INVALID_BYTECODE;
    }

    if (module->string_count > 0) {
        module->string_table = platform_calloc((size_t)module->string_count, sizeof(Name));
        if (!module->string_table) {
            free_module(module);
            return LOADER_STATUS_OUT_OF_MEMORY;
        }
        for (str_idx = 0; str_idx < module->string_count; str_idx++) {
            read_name(&reader, &module->string_table[str_idx]);
            if (reader.status != LOADER_STATUS_OK) {
                free_module(module);
                return reader.status;
            }
        }
    }

    *out_module = module;
    return LOADER_STATUS_OK;
}

LoaderStatus load_soir_file_checked(const char *filename, Module **out_module) {
    FILE *file;
    long file_size;
    uint8_t *buf;
    size_t nread;
    LoaderStatus status;

    if (!out_module) {
        return LOADER_STATUS_INVALID_ARGUMENT;
    }
    *out_module = NULL;
    if (!filename) {
        return LOADER_STATUS_INVALID_ARGUMENT;
    }

    file = fopen(filename, "rb");
    if (!file) {
        return LOADER_STATUS_IO_ERROR;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return LOADER_STATUS_IO_ERROR;
    }
    file_size = ftell(file);
    if (file_size < 0) {
        fclose(file);
        return LOADER_STATUS_IO_ERROR;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return LOADER_STATUS_IO_ERROR;
    }
    if (file_size > MAX_FILE_SIZE) {
        fclose(file);
        return LOADER_STATUS_FILE_TOO_LARGE;
    }

    buf = platform_malloc((size_t)file_size);
    if (!buf && file_size > 0) {
        fclose(file);
        return LOADER_STATUS_OUT_OF_MEMORY;
    }

    nread = fread(buf, 1, (size_t)file_size, file);
    fclose(file);
    if (nread != (size_t)file_size) {
        platform_free(buf);
        return LOADER_STATUS_IO_ERROR;
    }

    status = load_soir_buffer_checked(buf, (size_t)file_size, out_module);
    platform_free(buf);
    return status;
}

/* Load SOIR from buffer */
Module* load_soir_buffer(const uint8_t *buf, size_t size) {
    Module *module = NULL;
    if (load_soir_buffer_checked(buf, size, &module) != LOADER_STATUS_OK) {
        return NULL;
    }
    return module;
}

/* Load SOIR from file */
Module* load_soir_file(const char *filename) {
    Module *module = NULL;
    if (load_soir_file_checked(filename, &module) != LOADER_STATUS_OK) {
        return NULL;
    }
    return module;
}

/* Free module */
void free_module(Module *module) {
    int64_t i;

    if (!module) {
        return;
    }

    if (module->functions) {
        for (i = 0; i < module->fn_count; i++) {
            platform_free(module->functions[i].instrs);
        }
    }
    platform_free(module->functions);
    platform_free(module->string_table);
    platform_free(module);
}
