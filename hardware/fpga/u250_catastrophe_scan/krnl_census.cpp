// krnl_census.cpp — Vitis HLS reference outline for the U250 catastrophe-scan
// census kernel (Phase 1 of docs/research/u250_catastrophe_scan_fpga_spec_2026-07-26.md).
//
// STATUS: OUTLINE. Written 2026-07-26 before hardware/toolchain access; it has
// NOT been through vitis_hls. The executable contract is the bit-accurate C
// model scripts/research/fpga_census_kernel_model.c (CI gate
// scripts/ci/fpga_catastrophe_scan_gate.sh); when the U250s arrive this kernel
// must reproduce the model's verdict at every level b = 4..9.
//
// Kernel semantics (identical to the model):
//   sign bits: 0 = +1, 1 = -1.  For candidate pair (i, j), l = i ^ j:
//     d       = stab[i] ^ stab[j]          // sign of S[i,k]*S[j,k]
//     v       = d ^ perm_l(d)              // perm_l: bit k <- d[k ^ l] (rewiring)
//     bad     = N - popcount(v)            // #{k : p(k) = +1}
//     nullity = bad >> 1                   // exact, both signs simultaneously
// One candidate pair per cycle per PE (target II=1 @ 250 MHz).

#include <ap_int.h>
#include <ap_fixed.h>   // (not needed; placeholder to anchor Vitis includes)
#include <stdint.h>

#define BITS        9                 // level-9 scan (512-dim); BITS=10 image also fits
#define N           (1 << BITS)
#define PAIRS       ((N - 1) * (N - 2) / 2)   // 130,305 at BITS=9
#define PE_COUNT    16
#define HIST_BINS   (N / 2 + 1)       // nullity histogram bins
#define LABELS      N                 // fiber counters, one per xor label l

typedef ap_uint<N> row_t;             // one packed sign row (512 bits)

// ---------------------------------------------------------------------------
// perm_l: index-XOR permutation, a BITS-stage conditional-swap mux network.
// Pure rewiring: stage b swaps the two 2^b-halves of every aligned 2^(b+1)
// block iff bit b of l is set.  No arithmetic; ~BITS*N 2:1 muxes.
// ---------------------------------------------------------------------------
static row_t perm_l(row_t d, ap_uint<BITS> l) {
#pragma HLS INLINE
    row_t x = d;
    for (int b = 0; b < BITS; b++) {
#pragma HLS UNROLL
        if (l[b]) {
            row_t y;
            for (int blk = 0; blk < (N >> (b + 1)); blk++) {
#pragma HLS UNROLL
                for (int k = 0; k < (1 << b); k++) {
#pragma HLS UNROLL
                    y[(blk << (b + 1)) + k]           = x[(blk << (b + 1)) + (1 << b) + k];
                    y[(blk << (b + 1)) + (1 << b) + k] = x[(blk << (b + 1)) + k];
                }
            }
            x = y;
        }
    }
    return x;
}

// ---------------------------------------------------------------------------
// popcount512: balanced adder tree over 512 bits -> 10-bit count.
// Carry-save stages in LUT fabric; ~600 LUTs, no DSP.
// ---------------------------------------------------------------------------
static ap_uint<10> popcount_row(row_t v) {
#pragma HLS INLINE
    ap_uint<10> acc = 0;
    // Outline: unrolled tree; Vitis will balance.  Chunked form:
    for (int c = 0; c < N / 32; c++) {
#pragma HLS UNROLL
        ap_uint<6> partial = 0;
        for (int k = 0; k < 32; k++) {
#pragma HLS UNROLL
            partial += v[c * 32 + k];
        }
        acc += partial;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// One census PE: private sign-table replica (dual-port BRAM), private
// histogram.  Streams its share of the (i, j) candidate space at II=1.
// ---------------------------------------------------------------------------
static void census_pe(const row_t stab_init[N],   // loaded once per run
                      uint32_t pair_begin, uint32_t pair_end,
                      uint32_t hist_out[HIST_BINS],
                      uint32_t fiber_out[LABELS],
                      uint32_t *zd_pairs_out) {
    row_t stab[N];
#pragma HLS BIND_STORAGE variable=stab type=ram_2p impl=bram
    for (int r = 0; r < N; r++) stab[r] = stab_init[r];

    uint32_t hist[HIST_BINS];
#pragma HLS BIND_STORAGE variable=hist type=ram_2p impl=bram
    uint32_t fiber[LABELS];
#pragma HLS BIND_STORAGE variable=fiber type=ram_2p impl=bram
    for (int b = 0; b < HIST_BINS; b++) hist[b] = 0;
    for (int b = 0; b < LABELS; b++) fiber[b] = 0;
    uint32_t zd = 0;

    // Pair enumeration: linear index -> (i, j) with 1 <= i < j < N.
    // Outline uses a two-counter walk advanced once per cycle (cheap FSM),
    // avoiding a divider.
    ap_uint<BITS> i = 1, j = 2;
    for (uint32_t t = 0; t < PAIRS; t++) {
#pragma HLS PIPELINE II=1
        bool mine = (t >= pair_begin) && (t < pair_end);
        ap_uint<BITS> l = i ^ j;
        row_t d = stab[i] ^ stab[j];
        row_t v = d ^ perm_l(d, l);
        ap_uint<10> ones = popcount_row(v);
        ap_uint<10> nullity = (N - ones) >> 1;
        if (mine && nullity > 0) {
            hist[nullity]++;
            fiber[l]++;
            zd++;
        }
        // advance (i, j)
        if (j == N - 1) { i++; j = i + 1; } else { j++; }
    }

    for (int b = 0; b < HIST_BINS; b++) hist_out[b] = hist[b];
    for (int b = 0; b < LABELS; b++) fiber_out[b] = fiber[b];
    *zd_pairs_out = zd;
}

// ---------------------------------------------------------------------------
// Top kernel.  Host DMAs the packed sign table (32 KB at BITS=9) once;
// PEs run disjoint pair ranges; partial histograms are summed on the host
// (256 bins x 16 PEs = 16 KB readback).
// ---------------------------------------------------------------------------
extern "C" void krnl_census(const row_t *stab_gmem,      // [N] packed sign rows
                            uint32_t *hist_gmem,         // [PE_COUNT][HIST_BINS]
                            uint32_t *fiber_gmem,        // [PE_COUNT][LABELS]
                            uint32_t *zd_pairs_gmem) {   // [PE_COUNT]
#pragma HLS INTERFACE m_axi port=stab_gmem   bundle=gmem0 depth=N
#pragma HLS INTERFACE m_axi port=hist_gmem   bundle=gmem1 depth=PE_COUNT*HIST_BINS
#pragma HLS INTERFACE m_axi port=fiber_gmem  bundle=gmem1 depth=PE_COUNT*LABELS
#pragma HLS INTERFACE m_axi port=zd_pairs_gmem bundle=gmem1 depth=PE_COUNT
#pragma HLS INTERFACE s_axilite port=return

    // Outline: PE_COUNT instances over contiguous pair ranges.
    // (Vitis dataflow with function replication, or PE_COUNT kernels in the
    // xclbin — final form decided at first synthesis.)
    for (int pe = 0; pe < PE_COUNT; pe++) {
#pragma HLS UNROLL
        uint32_t begin = (uint32_t)(((uint64_t)PAIRS * pe) / PE_COUNT);
        uint32_t end   = (uint32_t)(((uint64_t)PAIRS * (pe + 1)) / PE_COUNT);
        census_pe(stab_gmem, begin, end,
                  &hist_gmem[pe * HIST_BINS],
                  &fiber_gmem[pe * LABELS],
                  &zd_pairs_gmem[pe]);
    }
}
