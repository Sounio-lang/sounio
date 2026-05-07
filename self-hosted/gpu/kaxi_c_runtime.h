#ifndef KAXI_C_RUNTIME_H
#define KAXI_C_RUNTIME_H
#include <stdio.h>
#include <string.h>

static void kaxi_run_scalar(void (*kern)(int, long long*, long long*), long long mem[1024]) {
    long long shared[256] = {0};
    kern(0, mem, shared);
}

static void kaxi_run_warp(void (*kern)(int, long long*, long long*), long long mem[1024], int warp_size) {
    long long shared[256] = {0};
    for (int lane = 0; lane < warp_size; lane++) {
        kern(lane, mem, shared);
    }
}

static void kaxi_dump_mem(const char* label, long long mem[1024], int count) {
    printf("%s:", label);
    for (int i = 0; i < count; i++) {
        printf(" %lld", mem[i]);
    }
    printf("\n");
}
#endif
