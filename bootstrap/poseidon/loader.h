/* loader.h - SOIR v1 Bytecode Loader */

#ifndef LOADER_H
#define LOADER_H

#include "vm.h"
#include <stddef.h>

/* Load SOIR bytecode from file */
Module* load_soir_file(const char *filename);

/* Load SOIR bytecode from memory buffer */
Module* load_soir_buffer(const uint8_t *buf, size_t size);

/* Free loaded module */
void free_module(Module *module);

#endif /* LOADER_H */
