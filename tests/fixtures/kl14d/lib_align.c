/*
 * #2537: SysV callee-alignment witness for the Madaros dynlink stub.
 *
 * SysV x86-64: at function entry (%rsp + 8) is a multiple of 16, i.e. the
 * return-address slot sits at an address == 8 (mod 16). The dynlink stub must
 * enter the callee with that alignment: `jmp [rip+GOT]` keeps the caller's
 * `call` frame; the pre-41fae115e `call [rip+GOT]; ret` pushed a second return
 * address and entered with the slot == 0 (mod 16).
 *
 * kl14d_align_probe(seed):
 *   - returns -1000 if the return-address slot is not == 8 (mod 16) -- an
 *     explicit check, so a misaligned entry fails with a value, not a crash;
 *   - otherwise round-trips a 16-byte-aligned stack buffer through
 *     _mm_load_ps/_mm_store_ps (movaps: faults if the frame is misaligned)
 *     and returns (seed+1)+(seed+2)+(seed+3)+(seed+4).
 * Built with -O0 so the frame is rbp-based and the compiler does not realign.
 */
#include <xmmintrin.h>

__attribute__((noinline)) long kl14d_align_probe(long seed) {
    unsigned long ra_slot = (unsigned long)__builtin_frame_address(0) + 8;
    if (ra_slot % 16 != 8) {
        return -1000;
    }
    __attribute__((aligned(16))) float buf[4];
    __attribute__((aligned(16))) float out[4];
    buf[0] = (float)(seed + 1);
    buf[1] = (float)(seed + 2);
    buf[2] = (float)(seed + 3);
    buf[3] = (float)(seed + 4);
    __m128 v = _mm_load_ps(buf);
    _mm_store_ps(out, v);
    long sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += (long)out[i];
    }
    return sum;
}
