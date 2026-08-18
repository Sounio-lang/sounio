// host.cpp — XRT/OpenCL host outline for the U250 catastrophe-scan kernel.
//
// STATUS: OUTLINE (2026-07-26, pre-hardware). Companion to
// docs/research/u250_catastrophe_scan_fpga_spec_2026-07-26.md.
// Reference CPU results come from scripts/research/fpga_census_kernel_model.c
// (CI-gated); the host must assert census equality against it.
//
// Flow:
//   1. Build the Cayley-Dickson sign table on the host (same recursion as
//      scripts/research/routon_zd_contract.py:cds; ~3 ms at L8) and pack it
//      to bits (0 = +1, 1 = -1): 512 x 512 bits = 32 KB at level 9.
//   2. DMA the table to the card, enqueue krnl_census, read back per-PE
//      histograms / fiber counters / ZD pair counts, sum on host.
//   3. Verify: triples == growth law Z(b); histogram == CPU model.

#include <cstdint>
#include <cstdio>
#include <vector>

// #include "xrt/xrt_bo.h"
// #include "xrt/xrt_device.h"
// #include "xrt/xrt_kernel.h"

#define BITS 9
#define N    (1 << BITS)
#define PE_COUNT 16
#define HIST_BINS (N / 2 + 1)

// Same recursion as scripts/research/routon_zd_contract.py:cds.
static int cds(int a, int b, int bits) {
    int s = 1;
    while (bits > 0) {
        if (a == 0 || b == 0) return s;
        if (bits == 1) return -s;
        int h = 1 << (bits - 1);
        bool ah = a >= h, bh = b >= h;
        int al = a & (h - 1), bl = b & (h - 1);
        if (!ah && !bh)      { a = al; b = bl; }
        else if (!ah && bh)  { a = bl; b = al; }
        else if (ah && !bh)  { if (bl == 0) { a = al; b = 0; }
                               else { a = al; b = bl; s = -s; } }
        else                 { if (bl == 0) { a = 0; b = al; s = -s; }
                               else { a = bl; b = al; } }
        bits--;
    }
    return s;
}

// Growth law Z(b) = 4^b - (3b-1)*2^b + 2^(b-1) - 4 (triples, both signs).
static uint64_t census_law(int b) {
    return (1ULL << (2 * b)) - (uint64_t)(3 * b - 1) * (1ULL << b)
           + (1ULL << (b - 1)) - 4;
}

int main(int argc, char **argv) {
    // ---- 1. build + pack sign table (bit = 1 iff sign = -1) ----
    std::vector<uint64_t> stab(N * (N / 64), 0);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            if (cds(i, k, BITS) < 0)
                stab[i * (N / 64) + (k >> 6)] |= 1ULL << (k & 63);

    // ---- 2. XRT skeleton (fill in when hardware arrives) ----
    // auto device = xrt::device(0);
    // auto uuid = device.load_xclbin("krnl_census.xclbin");
    // auto krnl = xrt::kernel(device, uuid, "krnl_census");
    // auto bo_stab  = xrt::bo(device, stab.size() * 8, krnl.group_id(0));
    // auto bo_hist  = xrt::bo(device, PE_COUNT * HIST_BINS * 4, krnl.group_id(1));
    // auto bo_fiber = xrt::bo(device, PE_COUNT * N * 4, krnl.group_id(2));
    // auto bo_zd    = xrt::bo(device, PE_COUNT * 4, krnl.group_id(3));
    // bo_stab.write(stab.data()); bo_stab.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // auto run = krnl(bo_stab, bo_hist, bo_fiber, bo_zd); run.wait();
    // bo_hist.sync(XCL_BO_SYNC_BO_FROM_DEVICE); ...

    // ---- 3. verify against law + CPU model ----
    // uint64_t triples = 2 * sum(bo_zd);
    // if (triples != census_law(BITS)) -> FAIL
    // histogram must equal scripts/research/fpga_census_kernel_model.c output.
    std::printf("host outline: expected triples at L%d = %llu\n",
                BITS, (unsigned long long)census_law(BITS));
    return 0;
}
