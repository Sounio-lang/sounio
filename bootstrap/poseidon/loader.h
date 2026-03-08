/* loader.h - SOIR v1 Bytecode Loader */

#ifndef LOADER_H
#define LOADER_H

#include "vm.h"
#include <stddef.h>

typedef enum {
    LOADER_STATUS_OK = 0,
    LOADER_STATUS_INVALID_ARGUMENT = 1,
    LOADER_STATUS_IO_ERROR = 2,
    LOADER_STATUS_FILE_TOO_LARGE = 3,
    LOADER_STATUS_INVALID_BYTECODE = 4,
    LOADER_STATUS_UNSUPPORTED_VERSION = 5,
    LOADER_STATUS_OUT_OF_MEMORY = 6
} LoaderStatus;

/* Load SOIR bytecode from file with structured status */
LoaderStatus load_soir_file_checked(const char *filename, Module **out_module);

/* Load SOIR bytecode from memory buffer with structured status */
LoaderStatus load_soir_buffer_checked(const uint8_t *buf, size_t size, Module **out_module);

/* Human-readable status text */
const char *loader_status_message(LoaderStatus status);

/* Load SOIR bytecode from file */
Module* load_soir_file(const char *filename);

/* Load SOIR bytecode from memory buffer */
Module* load_soir_buffer(const uint8_t *buf, size_t size);

/* Free loaded module */
void free_module(Module *module);

#endif /* LOADER_H */
