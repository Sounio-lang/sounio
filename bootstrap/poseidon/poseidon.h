/* poseidon.h - Stable Poseidon VM library API */

#ifndef POSEIDON_H
#define POSEIDON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    POSEIDON_STATUS_OK = 0,
    POSEIDON_STATUS_INVALID_ARGUMENT = 1,
    POSEIDON_STATUS_IO_ERROR = 2,
    POSEIDON_STATUS_FILE_TOO_LARGE = 3,
    POSEIDON_STATUS_INVALID_BYTECODE = 4,
    POSEIDON_STATUS_UNSUPPORTED_VERSION = 5,
    POSEIDON_STATUS_OUT_OF_MEMORY = 6,
    POSEIDON_STATUS_NO_MAIN_FUNCTION = 7,
    POSEIDON_STATUS_EXECUTION_ERROR = 8,
    POSEIDON_STATUS_EXECUTION_TIMEOUT = 9
} PoseidonStatus;

typedef struct PoseidonModule PoseidonModule;

typedef void (*PoseidonPrintIntCallback)(int64_t value, void *user_data);

typedef struct {
    int64_t max_steps;
    PoseidonPrintIntCallback print_int_cb;
    void *user_data;
} PoseidonRunConfig;

typedef struct {
    PoseidonStatus status;
    int64_t exit_code;
    int64_t steps_executed;
} PoseidonRunResult;

PoseidonRunConfig poseidon_default_config(void);
PoseidonStatus poseidon_module_load_file(const char *filename, PoseidonModule **out_module);
PoseidonStatus poseidon_module_load_bytes(const uint8_t *buf, size_t size, PoseidonModule **out_module);
void poseidon_module_free(PoseidonModule *module);
PoseidonRunResult poseidon_run_module(const PoseidonModule *module, const PoseidonRunConfig *config);
const char *poseidon_status_message(PoseidonStatus status);

#ifdef __cplusplus
}
#endif

#endif /* POSEIDON_H */
