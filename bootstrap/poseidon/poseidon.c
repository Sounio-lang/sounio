/* poseidon.c - Stable Poseidon VM library API */

#include "poseidon.h"

#include "loader.h"
#include "platform.h"
#include "vm.h"

struct PoseidonModule {
    Module *raw;
};

static PoseidonStatus poseidon_status_from_loader(LoaderStatus status) {
    switch (status) {
        case LOADER_STATUS_OK:
            return POSEIDON_STATUS_OK;
        case LOADER_STATUS_INVALID_ARGUMENT:
            return POSEIDON_STATUS_INVALID_ARGUMENT;
        case LOADER_STATUS_IO_ERROR:
            return POSEIDON_STATUS_IO_ERROR;
        case LOADER_STATUS_FILE_TOO_LARGE:
            return POSEIDON_STATUS_FILE_TOO_LARGE;
        case LOADER_STATUS_UNSUPPORTED_VERSION:
            return POSEIDON_STATUS_UNSUPPORTED_VERSION;
        case LOADER_STATUS_OUT_OF_MEMORY:
            return POSEIDON_STATUS_OUT_OF_MEMORY;
        case LOADER_STATUS_INVALID_BYTECODE:
        default:
            return POSEIDON_STATUS_INVALID_BYTECODE;
    }
}

PoseidonRunConfig poseidon_default_config(void) {
    PoseidonRunConfig config;
    config.max_steps = 10000;
    config.print_int_cb = NULL;
    config.user_data = NULL;
    return config;
}

PoseidonStatus poseidon_module_load_file(const char *filename, PoseidonModule **out_module) {
    Module *raw = NULL;
    PoseidonModule *module;
    LoaderStatus status;

    if (!out_module) {
        return POSEIDON_STATUS_INVALID_ARGUMENT;
    }
    *out_module = NULL;

    status = load_soir_file_checked(filename, &raw);
    if (status != LOADER_STATUS_OK) {
        return poseidon_status_from_loader(status);
    }

    module = platform_calloc(1, sizeof(PoseidonModule));
    if (!module) {
        free_module(raw);
        return POSEIDON_STATUS_OUT_OF_MEMORY;
    }
    module->raw = raw;
    *out_module = module;
    return POSEIDON_STATUS_OK;
}

PoseidonStatus poseidon_module_load_bytes(const uint8_t *buf, size_t size, PoseidonModule **out_module) {
    Module *raw = NULL;
    PoseidonModule *module;
    LoaderStatus status;

    if (!out_module) {
        return POSEIDON_STATUS_INVALID_ARGUMENT;
    }
    *out_module = NULL;

    status = load_soir_buffer_checked(buf, size, &raw);
    if (status != LOADER_STATUS_OK) {
        return poseidon_status_from_loader(status);
    }

    module = platform_calloc(1, sizeof(PoseidonModule));
    if (!module) {
        free_module(raw);
        return POSEIDON_STATUS_OUT_OF_MEMORY;
    }
    module->raw = raw;
    *out_module = module;
    return POSEIDON_STATUS_OK;
}

void poseidon_module_free(PoseidonModule *module) {
    if (!module) {
        return;
    }
    free_module(module->raw);
    platform_free(module);
}

PoseidonRunResult poseidon_run_module(const PoseidonModule *module, const PoseidonRunConfig *config) {
    PoseidonRunConfig effective = poseidon_default_config();
    PoseidonRunResult result;
    VMState vm;
    VmRunStats stats;
    int64_t main_fn;

    result.status = POSEIDON_STATUS_OK;
    result.exit_code = 0;
    result.steps_executed = 0;

    if (!module || !module->raw) {
        result.status = POSEIDON_STATUS_INVALID_ARGUMENT;
        return result;
    }
    if (config) {
        effective = *config;
    }
    if (effective.max_steps <= 0) {
        result.status = POSEIDON_STATUS_INVALID_ARGUMENT;
        return result;
    }

    main_fn = vm_find_main_fn(module->raw);
    if (main_fn < 0) {
        result.status = POSEIDON_STATUS_NO_MAIN_FUNCTION;
        return result;
    }

    vm_init(&vm, main_fn);
    vm_set_print_int_callback(&vm, effective.print_int_cb, effective.user_data);
    stats = vm_run_with_stats(&vm, module->raw, effective.max_steps);

    result.exit_code = stats.exit_code;
    result.steps_executed = stats.steps_executed;
    if (stats.timed_out) {
        result.status = POSEIDON_STATUS_EXECUTION_TIMEOUT;
    } else if (stats.step_error) {
        result.status = POSEIDON_STATUS_EXECUTION_ERROR;
        if (result.exit_code == 0) {
            result.exit_code = 1;
        }
    }

    return result;
}

const char *poseidon_status_message(PoseidonStatus status) {
    switch (status) {
        case POSEIDON_STATUS_OK:
            return "ok";
        case POSEIDON_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case POSEIDON_STATUS_IO_ERROR:
            return "i/o error";
        case POSEIDON_STATUS_FILE_TOO_LARGE:
            return "file too large";
        case POSEIDON_STATUS_INVALID_BYTECODE:
            return "invalid bytecode";
        case POSEIDON_STATUS_UNSUPPORTED_VERSION:
            return "unsupported version";
        case POSEIDON_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case POSEIDON_STATUS_NO_MAIN_FUNCTION:
            return "no main function";
        case POSEIDON_STATUS_EXECUTION_ERROR:
            return "execution error";
        case POSEIDON_STATUS_EXECUTION_TIMEOUT:
            return "execution timeout";
        default:
            return "unknown poseidon status";
    }
}
