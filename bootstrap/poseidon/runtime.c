/* runtime.c - Minimal Runtime Implementation */

#include "runtime.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>

void runtime_print_int(int64_t value) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%lld", (long long)value);
    platform_write_stdout(buf);
}

void runtime_print_char(char c) {
    char buf[2];
    buf[0] = c;
    buf[1] = '\0';
    platform_write_stdout(buf);
}

void runtime_panic(const char *msg) {
    char buf[256];
    snprintf(buf, sizeof(buf), "PANIC: %s\n", msg);
    platform_write_stderr(buf);
    exit(1);
}
