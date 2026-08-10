// krnl_san_scan.cpp — Vitis HLS kernel: SAN catastrophe scan + FLOP
// metering for the SAN-ImageNet deployment (SAN-ResNet-50-ImageNet,
// SAN-ViT-large-ImageNet) on the AMD Alveo U250.
//
// Status (2026-08-02): SYNTHESIS-READY v2. The SEMANTICS this kernel
// implements are CI-gated: scripts/research/san_imagenet_fpga_dl380.py
// (class U250SanScanModel) is the bit-accurate golden model, enforced by
// scripts/ci/san_imagenet_fpga_dl380_gate.sh (clause I6), and reproduced
// bit-exactly on the DL380 host (SAN_DL380_T3_VERDICT T3_GREEN).
// Spec: docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md
//
// What the kernel does (per validation cohort sweep):
//   * catastrophe scan: for each sample, find the FIRST exit point whose
//     Q0.15 confidence clears the threshold (priority encoder over an
//     S-wide comparator tree). Samples no exit could settle propagate to
//     the final head — the CATASTROPHE events (unsettled at depth); they
//     are counted.
//   * FLOP metering: charge the sample the exact executed-MAC prefix of
//     its exit point from a host-loaded LUT (real ResNet-50 / ViT-L MAC
//     prefixes), accumulated into 64-bit counters. Gated-off stages charge
//     exactly 0 by construction: the LUT holds PREFIX sums.
//
// v2 micro-architecture (measured-bandwidth-driven):
//   * DMA record: one 128-bit word per sample, 7 x 15-bit Q0.15 fields
//     (105 bits used; field k at bits [15k+14:15k]). The host packs at the
//     DMA boundary (it already quantizes there; packing is free).
//   * LANES = 4 samples per 512-bit bus beat, II=1: four independent
//     comparator trees + four private histogram banks (registers), so
//     same-bin exits in one beat never conflict. Sustained rate is
//     bus-limited, not logic-limited: 512 bits/beat / 128 bits/sample.
//   * Lane partials are summed in the epilogue and the kernel writes the
//     FINAL totals (hist[MAX_POINTS], catastrophe count, MAC total) —
//     the host reads three small buffers, no per-lane aggregation.
//
// Deliberately architecture-agnostic: one bitstream serves SAN-ResNet-50
// (5 points), SAN-ViT-large (7 points), or any trunk with <= MAX_POINTS
// exit points — switching architecture is a host-side LUT reload.
//
// No floating point anywhere on the card: confidences arrive pre-quantized
// Q0.15 from the host, decisions and metering are pure integer (spec T2/T3).

#include <ap_int.h>
#include <string.h>

#define MAX_POINTS 8        // exit points per sample (stages + final head)
#define LANES 4             // samples per 512-bit beat (bus-limited width)
#define HIST_BINS MAX_POINTS

typedef ap_uint<15> conf_t;       // Q0.15 confidence, 0..32767
typedef ap_uint<15> thresh_t;     // quantized DELTA
typedef ap_uint<4>  point_t;      // exit point index 0..MAX_POINTS-1
typedef ap_uint<64> flop_t;       // MAC accumulator (bound << 2^63, spec T2)
typedef ap_uint<32> count_t;
typedef ap_uint<128> sample_w_t;  // one packed sample (7 x 15-bit fields)
typedef ap_uint<512> bus_t;       // one DMA beat = LANES packed samples

// Configure-time LUT: lut[k] = cumulative real-architecture MACs charged
// when a sample exits at point k (k = n_points-1 is the final head ==
// dense cost). Passed as a plain m_axi array plus two s_axilite scalars —
// no struct on the interface, so there are no packing/alignment hazards
// between the HLS layout and the host's.

// One lane: extract packed fields, comparator tree + priority encoder,
// private histogram/FLOP update. Fully combinational over MAX_POINTS-1
// compares; registered by the enclosing PIPELINE.
static void san_scan_lane(sample_w_t rec,
                          const thresh_t q_delta,
                          const ap_uint<4> n_points,
                          const flop_t lut[MAX_POINTS],
                          count_t hist[HIST_BINS],
                          count_t &n_catastrophe,
                          flop_t &flop_macs) {
#pragma HLS INLINE
    point_t exit_idx = n_points - 1;  // default: final head (catastrophe)
    ap_uint<1> settled = 0;
    for (int k = 0; k < MAX_POINTS - 1; k++) {
#pragma HLS UNROLL
        conf_t c = rec(15 * k + 14, 15 * k);
        if (k < n_points - 1 && !settled && c >= q_delta) {
            exit_idx = k;
            settled = 1;
        }
    }
    hist[exit_idx]++;
    if (!settled) n_catastrophe++;
    flop_macs += lut[exit_idx];
}

extern "C" void krnl_san_scan(const bus_t *samples,
                              const flop_t *lut,      // [MAX_POINTS]
                              thresh_t q_delta,
                              int n_points,           // <= MAX_POINTS
                              int n_samples,
                              count_t *hist_out,        // [HIST_BINS]
                              count_t *catastrophe_out, // [1]
                              flop_t *flops_out) {      // [1]
#pragma HLS INTERFACE m_axi port = samples bundle = gmem0 depth = 262144 \
    max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = lut bundle = gmem1 depth = 8
#pragma HLS INTERFACE m_axi port = hist_out bundle = gmem1 depth = 8
#pragma HLS INTERFACE m_axi port = catastrophe_out bundle = gmem1 depth = 1
#pragma HLS INTERFACE m_axi port = flops_out bundle = gmem1 depth = 1
#pragma HLS INTERFACE s_axilite port = q_delta
#pragma HLS INTERFACE s_axilite port = n_points
#pragma HLS INTERFACE s_axilite port = n_samples
#pragma HLS INTERFACE s_axilite port = return

    // config is read once into registers (LUT is tiny: 8 x 64 bit)
    thresh_t q_delta_r = q_delta;
    ap_uint<4> n_points_r = n_points;
    flop_t lut_r[MAX_POINTS];
#pragma HLS ARRAY_PARTITION variable = lut_r complete dim = 1
    for (int k = 0; k < MAX_POINTS; k++) {
#pragma HLS UNROLL
        lut_r[k] = lut[k];
    }

    // per-lane private state: no cross-lane conflicts on same-bin exits
    count_t hist_lane[LANES][HIST_BINS];
#pragma HLS ARRAY_PARTITION variable = hist_lane complete dim = 0
    count_t cat_lane[LANES];
#pragma HLS ARRAY_PARTITION variable = cat_lane complete dim = 1
    flop_t flops_lane[LANES];
#pragma HLS ARRAY_PARTITION variable = flops_lane complete dim = 1
    for (int p = 0; p < LANES; p++) {
#pragma HLS UNROLL
        for (int b = 0; b < HIST_BINS; b++) {
#pragma HLS UNROLL
            hist_lane[p][b] = 0;
        }
        cat_lane[p] = 0;
        flops_lane[p] = 0;
    }

    const int n_words = (n_samples + LANES - 1) / LANES;
scan_loop:
    for (int w = 0; w < n_words; w++) {
#pragma HLS PIPELINE II = 1
        bus_t beat = samples[w];
        for (int p = 0; p < LANES; p++) {
#pragma HLS UNROLL
            int s_idx = w * LANES + p;
            if (s_idx < n_samples) {
                sample_w_t rec = beat(128 * p + 127, 128 * p);
                san_scan_lane(rec, q_delta_r, n_points_r, lut_r,
                              hist_lane[p], cat_lane[p], flops_lane[p]);
            }
        }
    }

    // epilogue: reduce lane partials, write final totals
    count_t cat_total = 0;
    flop_t flops_total = 0;
    for (int b = 0; b < HIST_BINS; b++) {
        count_t h = 0;
        for (int p = 0; p < LANES; p++) {
#pragma HLS UNROLL
            h += hist_lane[p][b];
        }
        hist_out[b] = h;
    }
    for (int p = 0; p < LANES; p++) {
#pragma HLS UNROLL
        cat_total += cat_lane[p];
        flops_total += flops_lane[p];
    }
    *catastrophe_out = cat_total;
    *flops_out = flops_total;
}
