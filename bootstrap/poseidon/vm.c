/* vm.c - Poseidon VM Core Execution Engine */

#include "vm.h"
#include "runtime.h"
#include <string.h>
#include <stdio.h>

/* Initialize VM state */
void vm_init(VMState *vm, int64_t entry_fn) {
    memset(vm, 0, sizeof(*vm));
    vm->current_fn = entry_fn;
}

void vm_set_print_int_callback(VMState *vm, VmPrintIntCallback cb, void *user_data) {
    vm->print_int_cb = cb;
    vm->print_int_user_data = user_data;
}

/* Find main function by name */
int64_t vm_find_main_fn(const Module *module) {
    for (int64_t i = 0; i < module->fn_count; i++) {
        const Name *n = &module->functions[i].name;
        if (n->len == 4 &&
            n->buf[0] == 'm' && n->buf[1] == 'a' &&
            n->buf[2] == 'i' && n->buf[3] == 'n') {
            return i;
        }
    }
    return -1;
}

/* Find label PC within a function */
int64_t vm_find_label_pc(const Function *func, int64_t label_id) {
    for (int64_t i = 0; i < func->instr_count; i++) {
        if (func->instrs[i].op == OP_LABEL &&
            func->instrs[i].label_id == label_id) {
            return i;
        }
    }
    return func->instr_count; /* End of function */
}

/* Execute binary operation */
static int64_t eval_binop(BinaryOp op, int64_t lhs, int64_t rhs) {
    switch (op) {
        case BINOP_ADD: return lhs + rhs;
        case BINOP_SUB: return lhs - rhs;
        case BINOP_MUL: return lhs * rhs;
        case BINOP_DIV: return (rhs == 0) ? 0 : lhs / rhs;
        case BINOP_MOD: return (rhs == 0) ? 0 : lhs % rhs;
        case BINOP_EQ: return lhs == rhs ? 1 : 0;
        case BINOP_NE: return lhs != rhs ? 1 : 0;
        case BINOP_LT: return lhs < rhs ? 1 : 0;
        case BINOP_LE: return lhs <= rhs ? 1 : 0;
        case BINOP_GT: return lhs > rhs ? 1 : 0;
        case BINOP_GE: return lhs >= rhs ? 1 : 0;
        case BINOP_AND: return (lhs != 0 && rhs != 0) ? 1 : 0;
        case BINOP_OR: return (lhs != 0 || rhs != 0) ? 1 : 0;
        case BINOP_BITAND: return lhs & rhs;
        case BINOP_BITOR: return lhs | rhs;
        case BINOP_BITXOR: return lhs ^ rhs;
        case BINOP_SHL: return lhs << rhs;
        case BINOP_SHR: return lhs >> rhs;
        default: return 0;
    }
}

/* Execute unary operation */
static int64_t eval_unop(UnaryOp op, int64_t src) {
    switch (op) {
        case UNOP_NEG: return -src;
        case UNOP_NOT: return (src == 0) ? 1 : 0;
        default: return src;
    }
}

/* Check if name is "print_int" */
static bool is_print_int(const Name *n) {
    return n->len == 9 &&
           n->buf[0] == 'p' && n->buf[1] == 'r' && n->buf[2] == 'i' &&
           n->buf[3] == 'n' && n->buf[4] == 't' && n->buf[5] == '_' &&
           n->buf[6] == 'i' && n->buf[7] == 'n' && n->buf[8] == 't';
}

/* Execute return instruction */
static void do_return(VMState *vm, int64_t ret_val) {
    if (vm->call_depth <= 0) {
        vm->halted = true;
        vm->exit_code = ret_val;
        return;
    }

    vm->call_depth--;
    CallFrame *frame = &vm->call_stack[vm->call_depth];
    vm->current_fn = frame->fn_id;
    vm->pc = frame->return_pc;

    if (frame->return_reg >= 0 && frame->return_reg < MAX_REGISTERS) {
        vm->regs[frame->return_reg] = ret_val;
    }
}

/* Execute one instruction */
int vm_step(VMState *vm, const Module *module) {
    if (vm->halted) {
        return 0;
    }

    if (vm->current_fn < 0 || vm->current_fn >= module->fn_count) {
        vm->halted = true;
        vm->step_error = true;
        vm->exit_code = 1;
        return -1;
    }

    const Function *func = &module->functions[vm->current_fn];

    if (vm->pc < 0 || vm->pc >= func->instr_count) {
        do_return(vm, 0);
        return 0;
    }

    const Instruction *ins = &func->instrs[vm->pc];

    switch (ins->op) {
        case OP_LOAD_IMM:
            if (ins->dst >= 0 && ins->dst < MAX_REGISTERS) {
                vm->regs[ins->dst] = ins->imm_i64;
            }
            vm->pc++;
            break;

        case OP_LOAD_BOOL:
            if (ins->dst >= 0 && ins->dst < MAX_REGISTERS) {
                vm->regs[ins->dst] = ins->imm_i64;
            }
            vm->pc++;
            break;

        case OP_COPY:
            if (ins->dst >= 0 && ins->dst < MAX_REGISTERS &&
                ins->src1 >= 0 && ins->src1 < MAX_REGISTERS) {
                vm->regs[ins->dst] = vm->regs[ins->src1];
            }
            vm->pc++;
            break;

        case OP_BINOP:
            if (ins->dst >= 0 && ins->dst < MAX_REGISTERS &&
                ins->src1 >= 0 && ins->src1 < MAX_REGISTERS &&
                ins->src2 >= 0 && ins->src2 < MAX_REGISTERS) {
                vm->regs[ins->dst] = eval_binop(ins->bin_op,
                                                vm->regs[ins->src1],
                                                vm->regs[ins->src2]);
            }
            vm->pc++;
            break;

        case OP_UNARYOP:
            if (ins->dst >= 0 && ins->dst < MAX_REGISTERS &&
                ins->src1 >= 0 && ins->src1 < MAX_REGISTERS) {
                vm->regs[ins->dst] = eval_unop(ins->un_op, vm->regs[ins->src1]);
            }
            vm->pc++;
            break;

        case OP_RETURN: {
            int64_t ret_val = 0;
            if (ins->src1 >= 0 && ins->src1 < MAX_REGISTERS) {
                ret_val = vm->regs[ins->src1];
            }
            do_return(vm, ret_val);
            break;
        }

        case OP_CALL:
            /* Check for builtin print_int */
            if (is_print_int(&ins->name)) {
                if (ins->src1 >= 0 && ins->src1 < MAX_REGISTERS) {
                    if (vm->print_int_cb) {
                        vm->print_int_cb(vm->regs[ins->src1], vm->print_int_user_data);
                    } else {
                        runtime_print_int(vm->regs[ins->src1]);
                    }
                }
                if (ins->dst >= 0 && ins->dst < MAX_REGISTERS) {
                    vm->regs[ins->dst] = 0;
                }
                vm->pc++;
                break;
            }

            /* Regular function call */
            if (ins->fn_id < 0 || ins->fn_id >= module->fn_count) {
                if (ins->dst >= 0 && ins->dst < MAX_REGISTERS) {
                    vm->regs[ins->dst] = 0;
                }
                vm->pc++;
                break;
            }

            if (vm->call_depth >= MAX_CALL_DEPTH) {
                vm->halted = true;
                vm->step_error = true;
                vm->exit_code = 2;
                return -1;
            }

            /* Push call frame */
            CallFrame *frame = &vm->call_stack[vm->call_depth++];
            frame->fn_id = vm->current_fn;
            frame->return_pc = vm->pc + 1;
            frame->return_reg = ins->dst;

            /* Copy arguments to callee parameter registers */
            const Function *callee = &module->functions[ins->fn_id];
            if (ins->arg_count > 0 && callee->param_count > 0) {
                int64_t preg0 = callee->param_regs[0];
                if (preg0 >= 0 && preg0 < MAX_REGISTERS &&
                    ins->src1 >= 0 && ins->src1 < MAX_REGISTERS) {
                    vm->regs[preg0] = vm->regs[ins->src1];
                }
            }
            if (ins->arg_count > 1 && callee->param_count > 1) {
                int64_t preg1 = callee->param_regs[1];
                if (preg1 >= 0 && preg1 < MAX_REGISTERS &&
                    ins->src2 >= 0 && ins->src2 < MAX_REGISTERS) {
                    vm->regs[preg1] = vm->regs[ins->src2];
                }
            }

            vm->current_fn = ins->fn_id;
            vm->pc = 0;
            break;

        case OP_JUMP:
            vm->pc = vm_find_label_pc(func, ins->label_id);
            break;

        case OP_BRANCH_TRUE: {
            int64_t cond = 0;
            if (ins->src1 >= 0 && ins->src1 < MAX_REGISTERS) {
                cond = vm->regs[ins->src1];
            }
            if (cond != 0) {
                vm->pc = vm_find_label_pc(func, ins->label_id);
            } else {
                vm->pc++;
            }
            break;
        }

        case OP_BRANCH_FALSE: {
            int64_t cond = 0;
            if (ins->src1 >= 0 && ins->src1 < MAX_REGISTERS) {
                cond = vm->regs[ins->src1];
            }
            if (cond == 0) {
                vm->pc = vm_find_label_pc(func, ins->label_id);
            } else {
                vm->pc++;
            }
            break;
        }

        case OP_LABEL:
        case OP_NOP:
            vm->pc++;
            break;

        default:
            /* Unknown opcode - skip */
            vm->pc++;
            break;
    }

    return 0;
}

/* Run VM until halt or max steps */
VmRunStats vm_run_with_stats(VMState *vm, const Module *module, int64_t max_steps) {
    VmRunStats stats;
    int64_t steps = 0;

    stats.exit_code = 0;
    stats.steps_executed = 0;
    stats.timed_out = false;
    stats.step_error = false;

    while (!vm->halted && steps < max_steps) {
        if (vm_step(vm, module) < 0) {
            break;
        }
        steps++;
    }

    if (!vm->halted && steps >= max_steps) {
        stats.timed_out = true;
    }

    vm->steps_executed = steps;
    stats.exit_code = vm->exit_code;
    stats.steps_executed = steps;
    stats.step_error = vm->step_error;
    return stats;
}

int64_t vm_run(VMState *vm, const Module *module, int64_t max_steps) {
    VmRunStats stats = vm_run_with_stats(vm, module, max_steps);

    if (stats.timed_out) {
        return -2; /* Timeout */
    }

    return vm->exit_code;
}
