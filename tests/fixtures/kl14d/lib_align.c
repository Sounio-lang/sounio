/*
 * #2537: SysV callee alignment witness.
 *
 * Uses _mm_store_ps / _mm_load_ps on a stack buffer. These SSE instructions
 * require 16-byte alignment; a misaligned stack (rsp % 16 != 0 at callee entry)
 * causes SIGSEGV. The old call+ret thunk left rsp % 16 == 8; the new jmp thunk
 * preserves rsp % 16 == 0.
 */
#include <immintrin.h>

long kl14d_align_test(long seed) {
    /* Stack buffer must be 16-byte aligned for movaps (used by _mm_store_ps). */
    __attribute__((aligned(16))) float buf[4];
    __attribute__((aligned(16))) float out[4];

    /* Fill with known values derived from seed. */
    buf[0] = (float)(seed + 1);
    buf[1] = (float)(seed + 2);
    buf[2] = (float)(seed + 3);
    buf[3] = (float)(seed + 4);

    /* SSE store+load — SIGSEGVs if buf is not 16-byte aligned. */
    __m128 v = _mm_load_ps(buf);
    _mm_store_ps(out, v);

    /* Verify round-trip and return checksum. */
    long sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += (long)out[i];
    }
    return sum;
}
