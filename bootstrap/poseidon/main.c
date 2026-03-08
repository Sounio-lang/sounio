/* main.c - Poseidon VM Entry Point */

#include <stdio.h>
#include <stdlib.h>
#include "poseidon.h"

int main(int argc, char **argv) {
    PoseidonModule *module = NULL;
    PoseidonRunConfig config;
    PoseidonRunResult run_result;
    PoseidonStatus status;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <soir-file>\n", argv[0]);
        return 1;
    }

    status = poseidon_module_load_file(argv[1], &module);
    if (status != POSEIDON_STATUS_OK) {
        fprintf(stderr, "Failed to load module: %s\n", poseidon_status_message(status));
        return 1;
    }

    config = poseidon_default_config();
    run_result = poseidon_run_module(module, &config);
    poseidon_module_free(module);

    if (run_result.status == POSEIDON_STATUS_NO_MAIN_FUNCTION) {
        fprintf(stderr, "No main function found\n");
        return 1;
    }
    if (run_result.status == POSEIDON_STATUS_EXECUTION_TIMEOUT) {
        fprintf(stderr, "Execution timeout (max 10000 steps)\n");
        return 1;
    }
    if (run_result.status != POSEIDON_STATUS_OK) {
        fprintf(stderr, "Execution failed: %s\n", poseidon_status_message(run_result.status));
        if (run_result.exit_code != 0) {
            return (int)run_result.exit_code;
        }
        return 1;
    }

    return (int)run_result.exit_code;
}
