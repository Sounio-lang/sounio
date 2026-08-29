#ifndef KAXI_C_RUNTIME_H
#define KAXI_C_RUNTIME_H
#include <stdio.h>
#include <string.h>

// Per-lane register and predicate state shared across warp lanes
static long long kaxi_lane_reg[8][64];
static int kaxi_lane_pred[8][8];

static void kaxi_run_scalar(void (*kern)(int, long long*, long long*), long long mem[1024]) {
    long long shared[256] = {0};
    memset(kaxi_lane_reg, 0, sizeof(kaxi_lane_reg));
    memset(kaxi_lane_pred, 0, sizeof(kaxi_lane_pred));
    kern(0, mem, shared);
}

static void kaxi_run_warp(void (*kern)(int, long long*, long long*), long long mem[1024], int warp_size) {
    long long shared[256] = {0};
    memset(kaxi_lane_reg, 0, sizeof(kaxi_lane_reg));
    memset(kaxi_lane_pred, 0, sizeof(kaxi_lane_pred));
    // Two-pass execution: first pass populates global lane state,
    // second pass lets warp ops read fully populated cross-lane state.
    for (int lane = 0; lane < warp_size; lane++) kern(lane, mem, shared);
    for (int lane = 0; lane < warp_size; lane++) kern(lane, mem, shared);
}

static void kaxi_dump_mem(const char* label, long long mem[1024], int count) {
    printf("%s:", label);
    for (int i = 0; i < count; i++) printf(" %lld", mem[i]);
    printf("\n");
}

// Warp-op helpers (read from global lane state)
static long long kaxi_vote_ballot(int tid, int pred_idx) {
    int mask = 0;
    for (int i = 0; i < 8; i++) {
        if (kaxi_lane_pred[i][pred_idx]) mask |= (1 << i);
    }
    return mask;
}

static long long kaxi_shuffle_xor(int tid, int src_reg, int xor_val) {
    int src_lane = tid ^ xor_val;
    if (src_lane >= 0 && src_lane < 8) return kaxi_lane_reg[src_lane][src_reg];
    return 0;
}

static long long kaxi_warp_reduce_add(int tid, int src_reg) {
    long long sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += kaxi_lane_reg[i][src_reg];
    }
    return sum;
}

#endif
