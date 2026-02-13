/* runtime.h - Minimal Runtime Support */

#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdint.h>

/* Print integer to stdout */
void runtime_print_int(int64_t value);

/* Print character to stdout */
void runtime_print_char(char c);

/* Panic with message */
void runtime_panic(const char *msg);

#endif /* RUNTIME_H */
