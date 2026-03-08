/* vm.h - Poseidon VM Interface */

#ifndef VM_H
#define VM_H

#include <stdint.h>
#include <stdbool.h>
#include "opcodes.h"

/* VM Configuration */
#define MAX_REGISTERS 1024
#define MAX_CALL_DEPTH 1024
#define MAX_FUNCTIONS 64
#define MAX_INSTRUCTIONS 2048
#define MAX_PARAMS 64
#define NAME_BUF_SIZE 128

/* Name structure (matches Sounio IR) */
typedef struct {
    int8_t buf[NAME_BUF_SIZE];
    int64_t len;
} Name;

/* Instruction structure (fixed 128 bytes in binary format) */
typedef struct {
    Opcode op;
    int64_t dst;
    int64_t src1;
    int64_t src2;
    int64_t imm_i64;
    double imm_f64;
    int64_t label_id;
    int64_t fn_id;
    int64_t field_idx;
    BinaryOp bin_op;
    UnaryOp un_op;
    Name name;
    int64_t arg_count;
} Instruction;

/* Function structure */
typedef struct {
    Name name;
    Instruction *instrs;
    int64_t instr_count;
    int64_t reg_count;
    int64_t label_count;
    int64_t param_count;
    int64_t param_regs[MAX_PARAMS];
} Function;

/* Module structure */
typedef struct {
    Function *functions;
    int64_t fn_count;
    Name *string_table;
    int64_t string_count;
} Module;

/* Call frame */
typedef struct {
    int64_t fn_id;
    int64_t return_pc;
    int64_t return_reg;
} CallFrame;

typedef void (*VmPrintIntCallback)(int64_t value, void *user_data);

/* VM state */
typedef struct {
    int64_t regs[MAX_REGISTERS];
    int64_t pc;
    CallFrame call_stack[MAX_CALL_DEPTH];
    int64_t call_depth;
    int64_t current_fn;
    bool halted;
    bool step_error;
    int64_t exit_code;
    int64_t steps_executed;
    VmPrintIntCallback print_int_cb;
    void *print_int_user_data;
} VMState;

typedef struct {
    int64_t exit_code;
    int64_t steps_executed;
    bool timed_out;
    bool step_error;
} VmRunStats;

/* VM API */
void vm_init(VMState *vm, int64_t entry_fn);
void vm_set_print_int_callback(VMState *vm, VmPrintIntCallback cb, void *user_data);
int vm_step(VMState *vm, const Module *module);
VmRunStats vm_run_with_stats(VMState *vm, const Module *module, int64_t max_steps);
int64_t vm_run(VMState *vm, const Module *module, int64_t max_steps);

/* Utility */
int64_t vm_find_main_fn(const Module *module);
int64_t vm_find_label_pc(const Function *func, int64_t label_id);

#endif /* VM_H */
