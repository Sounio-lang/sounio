/* main.c - Poseidon VM Entry Point */

#include <stdio.h>
#include <stdlib.h>
#include "loader.h"
#include "vm.h"
#include "runtime.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <soir-file>\n", argv[0]);
        return 1;
    }

    /* Load SOIR module */
    Module *module = load_soir_file(argv[1]);
    if (!module) {
        fprintf(stderr, "Failed to load module\n");
        return 1;
    }

    /* Find main function */
    int64_t main_fn = vm_find_main_fn(module);
    if (main_fn < 0) {
        fprintf(stderr, "No main function found\n");
        free_module(module);
        return 1;
    }

    /* Initialize VM */
    VMState vm;
    vm_init(&vm, main_fn);

    /* Run with 10000 step limit */
    int64_t result = vm_run(&vm, module, 10000);

    /* Cleanup */
    free_module(module);

    if (result == -2) {
        fprintf(stderr, "Execution timeout (max 10000 steps)\n");
        return 1;
    }

    return (int)result;
}
