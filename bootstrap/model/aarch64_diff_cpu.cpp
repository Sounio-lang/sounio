// aarch64_diff_cpu: native Apple Silicon CPU differential harness for
// formal/Bootstrap/AArch64Min.lean.
//
// Uses mmap(..., MAP_JIT) + pthread_jit_write_protect_np + sys_icache_invalidate
// and an alternate-stack SIGTRAP handler to execute each generated 32-bit
// AArch64 instruction natively on the local arm64 CPU with arbitrary GPRs, SP,
// NZCV flags, and a 64-byte stack window.
//
// Build: clang++ -std=c++23 -O2 -o aarch64_diff_cpu aarch64_diff_cpu.cpp
// Usage: aarch64_diff_cpu <cases> <seed>
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/ucontext.h>

#include <cerrno>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using s64 = int64_t;

static constexpr u64 WIN_SIZE = 64;
static constexpr u32 CODE_PAGE_SZ = 4096;
static constexpr u32 INS_OFF = 0x800;

struct DiffCtx {
    u64 regs[31];
    u64 sp;
    u64 nzcv;
};

static sigjmp_buf g_jb;
static __darwin_arm_thread_state64 g_cap;
static volatile sig_atomic_t g_got_sig = 0;

static void trap_handler(int sig, siginfo_t*, void* uctx) {
    auto* uc = static_cast<ucontext_t*>(uctx);
    g_cap = uc->uc_mcontext->__ss;
    g_got_sig = sig;
    siglongjmp(g_jb, 1);
}

static u64 rs;
static u64 rnd() {
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return rs;
}

static u64 rval() {
    static const u64 sp[] = {
        0, 1, 2, 0x7f, 0x80, 0xff, 0x100,
        0x7fffffff, 0x80000000, 0xffffffff, 0x100000000ULL,
        0x7fffffffffffffffULL, 0x8000000000000000ULL, ~0ULL, ~0ULL - 1
    };
    const u64 k = rnd() % 4;
    if (k == 0) return sp[rnd() % (sizeof sp / sizeof sp[0])];
    if (k == 1) return rnd() % 64;
    return rnd();
}

static void put_hex64(u64 v) {
    std::printf("%016llx", static_cast<unsigned long long>(v));
}

static void put_bytes(const u8* p, size_t n) {
    static const char hx[] = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i) {
        std::putchar(hx[p[i] >> 4]);
        std::putchar(hx[p[i] & 15]);
    }
}

static void put_state(u64 sp_val, const u64 regs[31], u32 cpsr, const u8* mem) {
    put_hex64(sp_val);
    for (int i = 0; i < 31; ++i) {
        std::putchar(' ');
        put_hex64(regs[i]);
    }
    std::putchar(' ');
    std::putchar('0' + ((cpsr >> 31) & 1)); // N
    std::putchar('0' + ((cpsr >> 30) & 1)); // Z
    std::putchar('0' + ((cpsr >> 29) & 1)); // C
    std::putchar('0' + ((cpsr >> 28) & 1)); // V
    std::putchar(' ');
    put_bytes(mem, WIN_SIZE);
}

static void build_stub(u32* code) {
    for (u32 i = 0; i < CODE_PAGE_SZ / 4; ++i) {
        code[i] = 0xd4200000u; // brk #0
    }
    u32 n = 0;
    auto emit = [&](u32 ins) { code[n++] = ins; };
    // x0 = &DiffCtx
    emit(0xf9408001u); // ldr x1, [x0, #0x100] (nzcv)
    emit(0xd51b4201u); // msr nzcv, x1
    emit(0xf9407c01u); // ldr x1, [x0, #0x0f8] (sp)
    emit(0x9100003fu); // mov sp, x1 (add sp, x1, #0)
    // ldp x(2k), x(2k+1), [x0, #(16*k)] for k = 1..14 (x2..x29)
    for (u32 k = 1; k <= 14; ++k) {
        const u32 imm7 = (k * 2) & 0x7f;
        const u32 rt1 = 2 * k;
        const u32 rt2 = 2 * k + 1;
        emit(0xa9400000u | (imm7 << 15) | (rt2 << 10) | (0 << 5) | rt1);
    }
    emit(0xf940781eu); // ldr x30, [x0, #0x0f0]
    emit(0xf9400401u); // ldr x1, [x0, #0x008]
    emit(0xf9400000u); // ldr x0, [x0, #0x000]
    const u32 slot_idx = INS_OFF / 4;
    const u32 delta = slot_idx - n;
    emit(0x14000000u | (delta & 0x03ffffffu)); // b INS_OFF
}

static int32_t pick_branch_words() {
    int32_t k;
    do {
        k = static_cast<int32_t>(rnd() % 501) - 250;
    } while (k == 0);
    return k;
}

static u32 pick_dst() {
    u32 r = static_cast<u32>(rnd() % 32);
    // On Darwin arm64, x18 is platform-reserved and XNU may zero it on preemption.
    if (r == 18) r = 19;
    return r;
}

static u32 gen_insn(DiffCtx& ctx, u64 win_lo, u64 ins_pc) {
    const u32 form = static_cast<u32>(rnd() % 32);
    const u32 sf = static_cast<u32>(rnd() % 2);
    const u32 rd = pick_dst();
    const u32 rn = static_cast<u32>(rnd() % 32);
    const u32 rm = static_cast<u32>(rnd() % 32);
    const u32 imm16 = static_cast<u32>(rnd() & 0xffff);
    const u32 imm12 = static_cast<u32>(rnd() & 0xfff);
    const u32 sh1 = static_cast<u32>(rnd() % 2);
    const u32 amt = static_cast<u32>(rnd() % (sf ? 64 : 32));

    switch (form) {
    case 0: // movz32
        return (0U << 31) | (2U << 29) | (0x25U << 23) | (static_cast<u32>(rnd() % 2) << 21) | (imm16 << 5) | rd;
    case 1: // movz64
        return (1U << 31) | (2U << 29) | (0x25U << 23) | (static_cast<u32>(rnd() % 4) << 21) | (imm16 << 5) | rd;
    case 2: // movn32
        return (0U << 31) | (0U << 29) | (0x25U << 23) | (static_cast<u32>(rnd() % 2) << 21) | (imm16 << 5) | rd;
    case 3: // movn64
        return (1U << 31) | (0U << 29) | (0x25U << 23) | (static_cast<u32>(rnd() % 4) << 21) | (imm16 << 5) | rd;
    case 4: // movk32
        return (0U << 31) | (3U << 29) | (0x25U << 23) | (static_cast<u32>(rnd() % 2) << 21) | (imm16 << 5) | rd;
    case 5: // movk64
        return (1U << 31) | (3U << 29) | (0x25U << 23) | (static_cast<u32>(rnd() % 4) << 21) | (imm16 << 5) | rd;
    case 6: // addImm32
        return (0U << 31) | (0U << 30) | (0U << 29) | (0x22U << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 7: // addImm64
        return (1U << 31) | (0U << 30) | (0U << 29) | (0x22U << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 8: // addsImm
        return (sf << 31) | (0U << 30) | (1U << 29) | (0x22U << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 9: // subImm32
        return (0U << 31) | (1U << 30) | (0U << 29) | (0x22U << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 10: // subImm64
        return (1U << 31) | (1U << 30) | (0U << 29) | (0x22U << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 11: // subsImm
        return (sf << 31) | (1U << 30) | (1U << 29) | (0x22U << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 12: // addReg
        return (sf << 31) | (0U << 30) | (0U << 29) | (0x0bU << 24) | (static_cast<u32>(rnd() % 3) << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    case 13: // addsReg
        return (sf << 31) | (0U << 30) | (1U << 29) | (0x0bU << 24) | (static_cast<u32>(rnd() % 3) << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    case 14: // subReg
        return (sf << 31) | (1U << 30) | (0U << 29) | (0x0bU << 24) | (static_cast<u32>(rnd() % 3) << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    case 15: // subsReg
        return (sf << 31) | (1U << 30) | (1U << 29) | (0x0bU << 24) | (static_cast<u32>(rnd() % 3) << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    case 16: // andReg
    case 17: // orrReg
    case 18: // eorReg
    case 19: { // andsReg
        const u32 opc = form - 16;
        const u32 inv = static_cast<u32>(rnd() % 2);
        const u32 sh = static_cast<u32>(rnd() % 4);
        return (sf << 31) | (opc << 29) | (0x0aU << 24) | (sh << 22) | (inv << 21) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    }
    case 20: // ldrb
    case 21: // strb
    case 22: // ldr32
    case 23: // str32
    case 24: // ldr64
    case 25: { // str64
        const u32 idx = form - 20;
        const bool is_load = (idx % 2 == 0);
        static const u32 szs[3] = {0, 2, 3};
        static const u32 widths[3] = {1, 4, 8};
        const u32 kind = idx / 2;
        const u32 sz = szs[kind];
        const u32 scale = widths[kind];
        const u32 base = (rnd() % 5 == 0) ? 31U : static_cast<u32>(rnd() % 31);
        const u32 rt = is_load ? pick_dst() : static_cast<u32>(rnd() % 32);
        u32 off_imm12 = 0;
        if (base == 31) {
            const u64 sp_base = win_lo + 16ULL * (rnd() % 3); // 16-byte aligned, >= 32 bytes left
            ctx.sp = sp_base;
            const u64 max_off = (WIN_SIZE - scale - (sp_base - win_lo)) / scale;
            off_imm12 = static_cast<u32>(rnd() % (max_off + 1));
        } else {
            const u32 base_reg = (base == 18) ? 19U : base;
            const u64 ea = win_lo + (rnd() % (WIN_SIZE - scale + 1));
            off_imm12 = (rnd() % 2 == 0) ? 0U : static_cast<u32>(rnd() % 64);
            ctx.regs[base_reg] = ea - static_cast<u64>(off_imm12) * scale;
            const u32 opc = is_load ? 1U : 0U;
            return (sz << 30) | (0x39U << 24) | (opc << 22) | (off_imm12 << 10) | (base_reg << 5) | rt;
        }
        const u32 opc = is_load ? 1U : 0U;
        return (sz << 30) | (0x39U << 24) | (opc << 22) | (off_imm12 << 10) | (base << 5) | rt;
    }
    case 26: { // b
        const int32_t k = pick_branch_words();
        return (0U << 31) | (0x05U << 26) | (static_cast<u32>(k) & 0x03ffffffu);
    }
    case 27: { // bl
        const int32_t k = pick_branch_words();
        return (1U << 31) | (0x05U << 26) | (static_cast<u32>(k) & 0x03ffffffu);
    }
    case 28: { // bcond
        const int32_t k = pick_branch_words();
        const u32 cc = static_cast<u32>(rnd() % 16);
        return (0x54U << 24) | ((static_cast<u32>(k) & 0x7ffffu) << 5) | cc;
    }
    case 29: // cbz
    case 30: { // cbnz
        const u32 nz = (form == 30) ? 1U : 0U;
        const u32 rt = static_cast<u32>(rnd() % 32);
        if (rt < 31 && rnd() % 2 == 0) {
            ctx.regs[rt] = (sf == 0 && rnd() % 2 == 0) ? (rnd() << 32) : 0ULL;
        }
        const int32_t k = pick_branch_words();
        return (sf << 31) | (0x1aU << 25) | (nz << 24) | ((static_cast<u32>(k) & 0x7ffffu) << 5) | rt;
    }
    default: { // ret (form 31)
        u32 r = static_cast<u32>(rnd() % 31);
        if (r == 18) r = 19;
        const int32_t k = pick_branch_words();
        ctx.regs[r] = static_cast<u64>(static_cast<s64>(ins_pc) + static_cast<s64>(k) * 4);
        return 0xd65f0000u | (r << 5);
    }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: aarch64_diff_cpu <cases> <seed>\n");
        return 2;
    }
    const long cases = std::atol(argv[1]);
    rs = 0x9e3779b97f4a7c15ULL ^ static_cast<u64>(std::strtoull(argv[2], nullptr, 10));

    static u8 alt_mem[65536];
    stack_t ss{};
    ss.ss_sp = alt_mem;
    ss.ss_size = sizeof(alt_mem);
    if (sigaltstack(&ss, nullptr) != 0) {
        std::perror("sigaltstack");
        return 2;
    }

    struct sigaction sa{};
    sa.sa_sigaction = trap_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    if (sigaction(SIGTRAP, &sa, nullptr) != 0) {
        std::perror("sigaction");
        return 2;
    }

    void* data = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (data == MAP_FAILED) {
        std::perror("mmap data");
        return 2;
    }
    void* code_raw = mmap(nullptr, CODE_PAGE_SZ, PROT_READ | PROT_WRITE | PROT_EXEC,
                          MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (code_raw == MAP_FAILED) {
        std::fprintf(stderr, "mmap MAP_JIT failed: %d (%s)\n", errno, std::strerror(errno));
        return 2;
    }

    auto* code = static_cast<u32*>(code_raw);
    pthread_jit_write_protect_np(0);
    build_stub(code);
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(code_raw, CODE_PAGE_SZ);

    auto* mem = static_cast<u8*>(data) + 0x400;
    const u64 win_lo = reinterpret_cast<u64>(mem);
    const u64 ins_pc = reinterpret_cast<u64>(code_raw) + INS_OFF;
    auto entry_fn = reinterpret_cast<void (*)(const DiffCtx*)>(code_raw);

    u8 before_mem[WIN_SIZE];
    DiffCtx ctx{};

    for (long k = 0; k < cases; ++k) {
        for (int i = 0; i < 31; ++i) ctx.regs[i] = (i == 18) ? 0ULL : rval();
        ctx.sp = rval();
        const u32 nzcv_nib = static_cast<u32>(rnd() & 15);
        ctx.nzcv = static_cast<u64>(nzcv_nib) << 28;
        for (u64 i = 0; i < WIN_SIZE; ++i) before_mem[i] = mem[i] = static_cast<u8>(rnd());

        const u32 ins = gen_insn(ctx, win_lo, ins_pc);
        const u8 ins_bytes[4] = {
            static_cast<u8>(ins & 0xff),
            static_cast<u8>((ins >> 8) & 0xff),
            static_cast<u8>((ins >> 16) & 0xff),
            static_cast<u8>((ins >> 24) & 0xff),
        };

        put_bytes(ins_bytes, 4);
        std::putchar(' ');
        put_hex64(ins_pc);
        std::putchar(' ');
        put_hex64(win_lo);
        std::putchar(' ');
        put_state(ctx.sp, ctx.regs, static_cast<u32>(ctx.nzcv), before_mem);

        pthread_jit_write_protect_np(0);
        code[INS_OFF / 4] = ins;
        pthread_jit_write_protect_np(1);
        sys_icache_invalidate(code + (INS_OFF / 4), 4);

        g_got_sig = 0;
        if (sigsetjmp(g_jb, 1) == 0) {
            entry_fn(&ctx);
        }
        if (g_got_sig != SIGTRAP) {
            std::fprintf(stderr, "unexpected signal %d on case %ld ins=%08x\n",
                         static_cast<int>(g_got_sig), k, ins);
            return 3;
        }

        u64 out_regs[31];
        for (int i = 0; i < 29; ++i) out_regs[i] = g_cap.__x[i];
        out_regs[29] = g_cap.__fp;
        out_regs[30] = g_cap.__lr;
        const u64 delta = g_cap.__pc - ins_pc;

        std::fputs(" | ", stdout);
        put_hex64(delta);
        std::putchar(' ');
        put_state(g_cap.__sp, out_regs, g_cap.__cpsr, mem);
        std::putchar('\n');
    }
    return 0;
}
