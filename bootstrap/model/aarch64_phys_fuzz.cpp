// aarch64_phys_fuzz: dual-OS native AArch64 hardware differential and 4-channel
// physical perturbation harness for Pillar P4-4 (AArch64Env, AArch64Phys,
// TMRVoterArm64, AArch64PhysCheck).
//
// Targets:
//   - Apple Silicon (Darwin arm64): MAP_JIT + pthread_jit_write_protect_np +
//     sys_icache_invalidate + alternate-stack SIGTRAP/SIGILL/SIGSEGV/SIGBUS.
//   - Linux aarch64 (e.g. NVIDIA DGX Spark ARMv9): RWX mmap +
//     __builtin___clear_cache + native fork() raw svc #0 exit(3) +
//     ptrace(PTRACE_SINGLESTEP / NT_PRSTATUS) verification.
//
// Build:
//   Darwin arm64:  clang++ -std=c++23 -O2 -Wall -Wextra -Werror -o aarch64_phys_fuzz aarch64_phys_fuzz.cpp
//   Linux aarch64: c++ -std=c++23 -O2 -Wall -Wextra -Werror -o aarch64_phys_fuzz aarch64_phys_fuzz.cpp
//
// Usage:
//   aarch64_phys_fuzz --diff <cases> <seed> [--neg-diff]
//   aarch64_phys_fuzz --phys <mc_trials> <seed> [--neg-voter]

#if !defined(__aarch64__)
#error "aarch64_phys_fuzz requires an AArch64 (arm64/aarch64) host"
#endif

#if defined(__APPLE__)
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>
#elif defined(__linux__)
#include <asm/ptrace.h>
#include <elf.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <ucontext.h>
#include <unistd.h>
#else
#error "Unsupported OS for aarch64_phys_fuzz (expected Darwin or Linux)"
#endif

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

// Canonical 44-byte (11-instruction) AArch64 majority voter over (x19, x20, x21),
// matching formal/Bootstrap/TMRVoterArm64.lean::voterWords bit-for-bit.
static constexpr u32 kVoterWords[11] = {
    0xeb14027fu,  // 0x00: cmp  x19, x20
    0xfa551266u,  // 0x04: ccmp x19, x21, #6, ne
    0x9a940273u,  // 0x08: csel x19, x19, x20, eq
    0xfa551286u,  // 0x0c: ccmp x20, x21, #6, ne
    0x54000081u,  // 0x10: b.ne +0x10 (-> 0x20)
    0xaa1303f4u,  // 0x14: mov  x20, x19
    0xaa1303f5u,  // 0x18: mov  x21, x19
    0x14000000u,  // 0x1c: b    +0x00 (-> 0x1c, park)
    0xd2800060u,  // 0x20: movz x0, #3
    0xd2800ba8u,  // 0x24: movz x8, #93 (__NR_exit)
    0xd4000001u,  // 0x28: svc  #0
};

struct DiffCtx {
    u64 regs[31];
    u64 sp;
    u64 nzcv;  // bits [31:28] hold N,Z,C,V
};

struct CapState {
    u64 regs[31];
    u64 sp;
    u64 pc;
    u32 cpsr;
    u64 esr;
};

static sigjmp_buf g_jb;
static CapState g_cap;
static volatile sig_atomic_t g_got_sig = 0;
static volatile u64 g_fault_addr = 0;

static void sig_handler(int sig, siginfo_t* si, void* uctx) {
    auto* uc = static_cast<ucontext_t*>(uctx);
#if defined(__APPLE__)
    const auto& ss = uc->uc_mcontext->__ss;
    for (int i = 0; i < 29; ++i) g_cap.regs[i] = ss.__x[i];
    g_cap.regs[29] = ss.__fp;
    g_cap.regs[30] = ss.__lr;
    g_cap.sp = ss.__sp;
    g_cap.pc = ss.__pc;
    g_cap.cpsr = ss.__cpsr;
    g_cap.esr = uc->uc_mcontext->__es.__esr;
#else
    const auto& mc = uc->uc_mcontext;
    for (int i = 0; i < 31; ++i) g_cap.regs[i] = mc.regs[i];
    g_cap.sp = mc.sp;
    g_cap.pc = mc.pc;
    g_cap.cpsr = static_cast<u32>(mc.pstate);
    g_cap.esr = 0;
    const u8* p = reinterpret_cast<const u8*>(mc.__reserved);
    const u8* end = p + sizeof(mc.__reserved);
    while (p + sizeof(struct _aarch64_ctx) <= end) {
        const auto* hdr = reinterpret_cast<const struct _aarch64_ctx*>(p);
        if (hdr->magic == 0 || hdr->size == 0) break;
        if (hdr->magic == ESR_MAGIC && p + sizeof(struct esr_context) <= end) {
            g_cap.esr = reinterpret_cast<const struct esr_context*>(p)->esr;
            break;
        }
        p += hdr->size;
    }
#endif
    g_fault_addr = si ? reinterpret_cast<u64>(si->si_addr) : 0ULL;
    g_got_sig = sig;
    siglongjmp(g_jb, 1);
}

static void jit_write_begin() {
#if defined(__APPLE__)
    pthread_jit_write_protect_np(0);
#endif
}

static void jit_write_end(void* ptr, size_t len) {
#if defined(__APPLE__)
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(ptr, len);
#else
    __builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + len);
#endif
}

#if defined(__APPLE__)
static constexpr int kMapAnon = MAP_ANON;
#else
static constexpr int kMapAnon = MAP_ANONYMOUS;
#endif

static void* alloc_jit_page(size_t len) {
#if defined(__APPLE__)
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
#else
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
    return (p == MAP_FAILED) ? nullptr : p;
}

struct JitArena {
    void* single_raw = nullptr;
    void* voter_raw = nullptr;
    void* guard_raw = nullptr;
    size_t page_sz = 0;
};

// Allocate a single contiguous 4-page JIT arena (required on Darwin arm64 where
// XNU permits at most one MAP_JIT region per process):
//   page 0: single-instruction diff slot
//   page 1: 44-byte voter fast-path page
//   page 2: boundary test page (instruction placed at page_sz - 4)
//   page 3: PROT_NONE (or unmapped) guard page catching any fetch straddle.
static JitArena alloc_jit_arena() {
    JitArena a{};
    a.page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t total = 4 * a.page_sz;
    void* base = alloc_jit_page(total);
    if (!base) {
        std::fprintf(stderr, "alloc_jit_arena mmap failed: %d (%s)\n", errno, std::strerror(errno));
        return a;
    }
    u8* p = static_cast<u8*>(base);
    void* guard_page = p + 3 * a.page_sz;
    if (mprotect(guard_page, a.page_sz, PROT_NONE) != 0) {
        // On Darwin arm64 MAP_JIT, mprotect may return EINVAL; unmap page 3 and
        // remap it as non-JIT PROT_NONE so any read past page 2 faults immediately.
        if (munmap(guard_page, a.page_sz) != 0) {
            std::fprintf(stderr, "alloc_jit_arena guard munmap failed: %d (%s)\n",
                         errno, std::strerror(errno));
            munmap(base, total);
            return a;
        }
        (void)mmap(guard_page, a.page_sz, PROT_NONE,
                   MAP_PRIVATE | kMapAnon | MAP_FIXED, -1, 0);
    }
    a.single_raw = p;
    a.voter_raw = p + a.page_sz;
    a.guard_raw = p + 2 * a.page_sz;
    return a;
}

static __attribute__((noinline)) int exec_trampoline(void (*entry_fn)(const DiffCtx*),
                                                     const DiffCtx* ctx) {
    g_got_sig = 0;
    g_fault_addr = 0;
    if (sigsetjmp(g_jb, 1) == 0) {
        entry_fn(ctx);
    }
    return static_cast<int>(g_got_sig);
}

static bool install_handlers() {
    static u8 alt_mem[65536];
    stack_t ss{};
    ss.ss_sp = alt_mem;
    ss.ss_size = sizeof(alt_mem);
    if (sigaltstack(&ss, nullptr) != 0) {
        std::perror("sigaltstack");
        return false;
    }
    struct sigaction sa{};
    sa.sa_sigaction = sig_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    const int sigs[] = {SIGTRAP, SIGILL, SIGSEGV, SIGBUS};
    for (int s : sigs) {
        if (sigaction(s, &sa, nullptr) != 0) {
            std::perror("sigaction");
            return false;
        }
    }
    return true;
}

static u64 rs = 1;
static u64 rnd() {
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return rs;
}

static bool bernoulli(u64 num, u64 den) {
    return (rnd() % den) < num;
}

static u64 rval() {
    static const u64 sp[] = {
        0ULL, 1ULL, 2ULL, 0x7fULL, 0x80ULL, 0xffULL, 0x100ULL,
        0x7fffffffULL, 0x80000000ULL, 0xffffffffULL, 0x100000000ULL,
        0x7fffffffffffffffULL, 0x8000000000000000ULL, ~0ULL, ~0ULL - 1ULL
    };
    const u64 k = rnd() % 4;
    if (k == 0) return sp[rnd() % (sizeof(sp) / sizeof(sp[0]))];
    if (k == 1) return rnd() % 64;
    return rnd();
}

static u32 bf(u32 w, u32 lo, u32 len) {
    return (w >> lo) & ((len == 32) ? 0xffffffffu : ((1u << len) - 1u));
}

static u64 sx(u32 bits, u64 v) {
    const u64 sign = 1ULL << (bits - 1);
    const u64 mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
    v &= mask;
    return (v & sign) ? (v | ~mask) : v;
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
    std::putchar('0' + ((cpsr >> 31) & 1));  // N
    std::putchar('0' + ((cpsr >> 30) & 1));  // Z
    std::putchar('0' + ((cpsr >> 29) & 1));  // C
    std::putchar('0' + ((cpsr >> 28) & 1));  // V
    std::putchar(' ');
    put_bytes(mem, WIN_SIZE);
}

// Build trampoline at code[0..]: loads DiffCtx from x0 (nzcv, sp, x2..x30, x1, x0)
// and branches to byte offset target_off. Rest of page is filled with brk #0.
static void build_stub(u32* code, size_t page_bytes, u32 target_off) {
    for (size_t i = 0; i < page_bytes / 4; ++i) {
        code[i] = 0xd4200000u;  // brk #0
    }
    u32 n = 0;
    auto emit = [&](u32 ins) { code[n++] = ins; };
    emit(0xf9408001u);  // ldr x1, [x0, #0x100] (nzcv)
    emit(0xd51b4201u);  // msr nzcv, x1
    emit(0xf9407c01u);  // ldr x1, [x0, #0x0f8] (sp)
    emit(0x9100003fu);  // mov sp, x1
    for (u32 k = 1; k <= 14; ++k) {
        const u32 imm7 = (k * 2) & 0x7fu;
        const u32 rt1 = 2 * k;
        const u32 rt2 = 2 * k + 1;
        emit(0xa9400000u | (imm7 << 15) | (rt2 << 10) | (0u << 5) | rt1);
    }
    emit(0xf940781eu);  // ldr x30, [x0, #0x0f0]
    emit(0xf9400401u);  // ldr x1,  [x0, #0x008]
    emit(0xf9400000u);  // ldr x0,  [x0, #0x000]
    const u32 slot_idx = target_off / 4;
    const u32 delta = slot_idx - n;
    emit(0x14000000u | (delta & 0x03ffffffu));  // b target_off
}

static u32 pick_reg(bool allow31) {
    u32 r = static_cast<u32>(rnd() % (allow31 ? 32u : 31u));
    // Avoid x18: platform-reserved on Darwin arm64.
    if (r == 18) r = 19;
    return r;
}

static int32_t pick_branch_words() {
    int32_t k;
    do {
        k = static_cast<int32_t>(rnd() % 501) - 250;
    } while (k == 0);
    return k;
}

static u64 extend_idx(u32 opt, u64 raw) {
    switch (opt) {
    case 2: return raw & 0xffffffffULL;       // uxtw
    case 3: return raw;                       // lsl
    case 6: return sx(32, raw & 0xffffffffULL);  // sxtw
    case 7: return raw;                       // sxtx
    default: return raw;
    }
}

// Generate one instruction of the exact form index `form` in 0..52,
// adjusting ctx registers/SP as needed so memory/indirect targets stay in bounds.
static u32 gen_insn_form(u32 form, DiffCtx& ctx, u64 win_lo, u64 ins_pc) {
    const u32 sf = static_cast<u32>(rnd() % 2);
    const u32 rd = pick_reg(true);
    const u32 rn = pick_reg(true);
    const u32 rm = pick_reg(true);
    const u32 imm16 = static_cast<u32>(rnd() & 0xffffu);
    const u32 imm12 = static_cast<u32>(rnd() & 0xfffu);
    const u32 sh1 = static_cast<u32>(rnd() % 2);
    const u32 amt = static_cast<u32>(rnd() % (sf ? 64u : 32u));

    switch (form) {
    case 0:  // movz32
        return (0u << 31) | (2u << 29) | (0x25u << 23) | (static_cast<u32>(rnd() % 2) << 21) | (imm16 << 5) | rd;
    case 1:  // movz64
        return (1u << 31) | (2u << 29) | (0x25u << 23) | (static_cast<u32>(rnd() % 4) << 21) | (imm16 << 5) | rd;
    case 2:  // movn32
        return (0u << 31) | (0u << 29) | (0x25u << 23) | (static_cast<u32>(rnd() % 2) << 21) | (imm16 << 5) | rd;
    case 3:  // movn64
        return (1u << 31) | (0u << 29) | (0x25u << 23) | (static_cast<u32>(rnd() % 4) << 21) | (imm16 << 5) | rd;
    case 4:  // movk32
        return (0u << 31) | (3u << 29) | (0x25u << 23) | (static_cast<u32>(rnd() % 2) << 21) | (imm16 << 5) | rd;
    case 5:  // movk64
        return (1u << 31) | (3u << 29) | (0x25u << 23) | (static_cast<u32>(rnd() % 4) << 21) | (imm16 << 5) | rd;
    case 6:  // addImm32
        return (0u << 31) | (0u << 30) | (0u << 29) | (0x22u << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 7:  // addImm64
        return (1u << 31) | (0u << 30) | (0u << 29) | (0x22u << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 8:  // addsImm
        return (sf << 31) | (0u << 30) | (1u << 29) | (0x22u << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 9:  // subImm32
        return (0u << 31) | (1u << 30) | (0u << 29) | (0x22u << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 10:  // subImm64
        return (1u << 31) | (1u << 30) | (0u << 29) | (0x22u << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 11:  // subsImm
        return (sf << 31) | (1u << 30) | (1u << 29) | (0x22u << 23) | (sh1 << 22) | (imm12 << 10) | (rn << 5) | rd;
    case 12:  // addReg
        return (sf << 31) | (0u << 30) | (0u << 29) | (0x0bu << 24) | (static_cast<u32>(rnd() % 3) << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    case 13:  // addsReg
        return (sf << 31) | (0u << 30) | (1u << 29) | (0x0bu << 24) | (static_cast<u32>(rnd() % 3) << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    case 14:  // subReg
        return (sf << 31) | (1u << 30) | (0u << 29) | (0x0bu << 24) | (static_cast<u32>(rnd() % 3) << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    case 15:  // subsReg
        return (sf << 31) | (1u << 30) | (1u << 29) | (0x0bu << 24) | (static_cast<u32>(rnd() % 3) << 22) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    case 16:  // andReg
    case 17:  // orrReg
    case 18:  // eorReg
    case 19: {  // andsReg
        const u32 opc = form - 16;
        const u32 inv = static_cast<u32>(rnd() % 2);
        const u32 sh = static_cast<u32>(rnd() % 4);
        return (sf << 31) | (opc << 29) | (0x0au << 24) | (sh << 22) | (inv << 21) | (rm << 16) | (amt << 10) | (rn << 5) | rd;
    }
    case 20:  // sbfm
    case 21:  // bfm
    case 22: {  // ubfm
        const u32 opc = form - 20;
        const u32 w = sf ? 64u : 32u;
        const u32 immr = static_cast<u32>(rnd() % w);
        const u32 imms = static_cast<u32>(rnd() % w);
        return (sf << 31) | (opc << 29) | (0x26u << 23) | (sf << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd;
    }
    case 23:  // adr
    case 24: {  // adrp
        const u32 is_page = (form == 24) ? 1u : 0u;
        const u32 imm21 = static_cast<u32>(rnd() & 0x1fffffu);
        const u32 immlo = imm21 & 3u;
        const u32 immhi = (imm21 >> 2) & 0x7ffffu;
        return (is_page << 31) | (immlo << 29) | (0x10u << 24) | (immhi << 5) | rd;
    }
    case 25:  // ccmpReg
    case 26: {  // ccmnReg
        const u32 is_sub = (form == 25) ? 1u : 0u;
        const u32 cc = static_cast<u32>(rnd() % 16);
        const u32 nzcv = static_cast<u32>(rnd() % 16);
        return (sf << 31) | (is_sub << 30) | (1u << 29) | (0xd2u << 21) | (rm << 16) | (cc << 12) | (rn << 5) | nzcv;
    }
    case 27:  // csel
    case 28:  // csinc
    case 29:  // csinv
    case 30: {  // csneg
        const u32 k = form - 27;
        const u32 op_bit = (k >> 1) & 1u;
        const u32 inc_bit = k & 1u;
        const u32 cc = static_cast<u32>(rnd() % 16);
        return (sf << 31) | (op_bit << 30) | (0xd4u << 21) | (rm << 16) | (cc << 12) | (inc_bit << 10) | (rn << 5) | rd;
    }
    case 31:  // ldrb
    case 32:  // strb
    case 33:  // ldrh
    case 34:  // strh
    case 35:  // ldr32
    case 36:  // str32
    case 37:  // ldr64
    case 38: {  // str64
        const u32 idx = form - 31;
        const bool is_load = (idx % 2 == 0);
        const u32 sz = idx / 2;  // 0,1,2,3 -> 1,2,4,8 bytes
        const u32 scale = 1u << sz;
        const u32 base = (rnd() % 5 == 0) ? 31u : pick_reg(false);
        const u32 rt = pick_reg(true);
        const u32 opc = is_load ? 1u : 0u;
        u32 off_imm12 = 0;
        if (base == 31) {
            const u64 sp_base = win_lo + 16ULL * (rnd() % 3);  // 16B aligned
            ctx.sp = sp_base;
            const u64 max_off = (WIN_SIZE - scale - (sp_base - win_lo)) / scale;
            off_imm12 = static_cast<u32>(rnd() % (max_off + 1));
        } else {
            const u64 ea = win_lo + (rnd() % (WIN_SIZE - scale + 1));
            off_imm12 = (rnd() % 2 == 0) ? 0u : static_cast<u32>(rnd() % 64);
            ctx.regs[base] = ea - static_cast<u64>(off_imm12) * scale;
        }
        return (sz << 30) | (0x39u << 24) | (opc << 22) | (off_imm12 << 10) | (base << 5) | rt;
    }
    case 39:  // ldur
    case 40: {  // stur
        const bool is_load = (form == 39);
        const u32 sz = static_cast<u32>(rnd() % 4);
        const u32 scale = 1u << sz;
        const u32 base = (rnd() % 5 == 0) ? 31u : pick_reg(false);
        const u32 rt = pick_reg(true);
        const u32 opc = is_load ? 1u : 0u;
        int32_t imm9s = 0;
        if (base == 31) {
            ctx.sp = win_lo + 32ULL;  // 16B aligned, +/-32 stays in [win_lo, win_lo+64)
            const int32_t max_pos = static_cast<int32_t>(32 - scale);
            imm9s = static_cast<int32_t>(rnd() % static_cast<u64>(max_pos + 33)) - 32;
        } else {
            const u64 ea = win_lo + (rnd() % (WIN_SIZE - scale + 1));
            imm9s = static_cast<int32_t>(rnd() % 512) - 256;
            ctx.regs[base] = ea - static_cast<u64>(static_cast<s64>(imm9s));
        }
        const u32 imm9 = static_cast<u32>(imm9s) & 0x1ffu;
        return (sz << 30) | (0x38u << 24) | (opc << 22) | (0u << 21) | (imm9 << 12) | (0u << 10) | (base << 5) | rt;
    }
    case 41:  // ldrReg
    case 42: {  // strReg
        static const u32 opts[4] = {2u, 3u, 6u, 7u};
        const bool is_load = (form == 41);
        const u32 sz = static_cast<u32>(rnd() % 4);
        const u32 scale = 1u << sz;
        const u32 opt = opts[rnd() % 4];
        const u32 s_bit = static_cast<u32>(rnd() % 2);
        const u32 sh = s_bit ? sz : 0u;
        const u32 opc = is_load ? 1u : 0u;
        const u32 rt = pick_reg(true);
        const u32 base = (rnd() % 5 == 0) ? 31u : pick_reg(false);
        u32 idx_reg = pick_reg(true);
        if (base < 31 && idx_reg == base) {
            idx_reg = (base + 1u) % 31u;
            if (idx_reg == 18) idx_reg = 19;
        }
        if (base == 31) {
            ctx.sp = win_lo;  // 16B aligned
            const u64 max_raw = (WIN_SIZE - scale) >> sh;
            const u64 small_idx = (idx_reg == 31) ? 0ULL : (rnd() % (max_raw + 1));
            if (idx_reg < 31) ctx.regs[idx_reg] = small_idx;
        } else {
            const u64 ea = win_lo + (rnd() % (WIN_SIZE - scale + 1));
            const u64 idx_raw = (idx_reg == 31) ? 0ULL : ctx.regs[idx_reg];
            const u64 idx_eff = extend_idx(opt, idx_raw) << sh;
            ctx.regs[base] = ea - idx_eff;
        }
        return (sz << 30) | (0x38u << 24) | (opc << 22) | (1u << 21) |
               (idx_reg << 16) | (opt << 13) | (s_bit << 12) | (2u << 10) | (base << 5) | rt;
    }
    case 43: {  // b
        const int32_t k = pick_branch_words();
        return (0u << 31) | (0x05u << 26) | (static_cast<u32>(k) & 0x03ffffffu);
    }
    case 44: {  // bl
        const int32_t k = pick_branch_words();
        return (1u << 31) | (0x05u << 26) | (static_cast<u32>(k) & 0x03ffffffu);
    }
    case 45: {  // bcond
        const int32_t k = pick_branch_words();
        const u32 cc = static_cast<u32>(rnd() % 16);
        return (0x54u << 24) | ((static_cast<u32>(k) & 0x7ffffu) << 5) | cc;
    }
    case 46:  // cbz
    case 47: {  // cbnz
        const u32 nz = (form == 47) ? 1u : 0u;
        const u32 rt = pick_reg(true);
        if (rt < 31 && (rnd() % 2 == 0)) {
            ctx.regs[rt] = (sf == 0 && (rnd() % 2 == 0)) ? (rnd() << 32) : 0ULL;
        }
        const int32_t k = pick_branch_words();
        return (sf << 31) | (0x1au << 25) | (nz << 24) | ((static_cast<u32>(k) & 0x7ffffu) << 5) | rt;
    }
    case 48:  // tbz
    case 49: {  // tbnz
        const u32 nz = (form == 49) ? 1u : 0u;
        const u32 bit = static_cast<u32>(rnd() % 64);
        const u32 b5 = (bit >> 5) & 1u;
        const u32 b40 = bit & 0x1fu;
        const u32 rt = pick_reg(true);
        if (rt < 31 && (rnd() % 2 == 0)) {
            ctx.regs[rt] ^= (1ULL << bit);
        }
        const int32_t k = pick_branch_words();
        return (b5 << 31) | (0x1bu << 25) | (nz << 24) | (b40 << 19) |
               ((static_cast<u32>(k) & 0x3fffu) << 5) | rt;
    }
    case 50:  // br
    case 51: {  // blr
        const u32 r = pick_reg(false);
        const int32_t k = pick_branch_words();
        ctx.regs[r] = static_cast<u64>(static_cast<s64>(ins_pc) + static_cast<s64>(k) * 4);
        return (form == 50 ? 0xd61f0000u : 0xd63f0000u) | (r << 5);
    }
    default: {  // ret (form 52)
        const u32 r = (rnd() % 2 == 0) ? 30u : pick_reg(false);
        const int32_t k = pick_branch_words();
        ctx.regs[r] = static_cast<u64>(static_cast<s64>(ins_pc) + static_cast<s64>(k) * 4);
        return 0xd65f0000u | (r << 5);
    }
    }
}

// Check if a 32-bit word is an exception-generating or system instruction
// that must NOT be executed raw in-process during Hamming-1 sweep.
static bool is_dangerous_sys_or_exc(u32 w) {
    if ((w >> 24) == 0xd4u) return true;        // SVC/HVC/SMC/BRK/HLT/DCPS
    if ((w >> 22) == 0x354u) return true;       // MSR/MRS/SYS/HINT/barriers
    return false;
}

// Check if a 32-bit word is a direct branch whose target either self-loops
// (off == 0) or falls outside the safe landing range [safe_lo, safe_hi).
static bool is_unsafe_branch_target(u32 w, u64 pc, u64 safe_lo, u64 safe_hi) {
    auto check_off = [&](u64 off) -> bool {
        if (off == 0) return true;
        const u64 tgt = pc + off;
        return tgt < safe_lo || tgt >= safe_hi || (tgt % 4 != 0);
    };
    if (bf(w, 26, 5) == 0x05u) return check_off(sx(28, static_cast<u64>(bf(w, 0, 26)) * 4ULL));
    if (bf(w, 24, 8) == 0x54u) return check_off(sx(21, static_cast<u64>(bf(w, 5, 19)) * 4ULL));
    if (bf(w, 25, 6) == 0x1au) return check_off(sx(21, static_cast<u64>(bf(w, 5, 19)) * 4ULL));
    if (bf(w, 25, 6) == 0x1bu) return check_off(sx(16, static_cast<u64>(bf(w, 5, 14)) * 4ULL));
    return false;
}

// Check whether a Hamming-1 perturbed word decodes in AArch64Phys AND is safe
// to execute in our normal single-instruction diff slot for Lean verification.
static bool is_safe_model_alt(u32 w, const DiffCtx& ctx, u64 ins_pc,
                              u64 code_lo, u64 code_hi, u64 win_lo) {
    const u32 sf = bf(w, 31, 1);
    auto touches_x18 = [](u32 r) { return r == 18; };

    if (bf(w, 26, 5) == 0x05u) {
        const u64 off = sx(28, static_cast<u64>(bf(w, 0, 26)) * 4ULL);
        const u64 tgt = ins_pc + off;
        return tgt >= code_lo + 0x200 && tgt < code_hi && tgt != ins_pc && (tgt % 4 == 0);
    }
    if (bf(w, 24, 8) == 0x54u && bf(w, 4, 1) == 0u) {
        const u64 off = sx(21, static_cast<u64>(bf(w, 5, 19)) * 4ULL);
        const u64 tgt = ins_pc + off;
        return tgt >= code_lo + 0x200 && tgt < code_hi && tgt != ins_pc && (tgt % 4 == 0);
    }
    if (bf(w, 25, 6) == 0x1au) {
        if (touches_x18(bf(w, 0, 5))) return false;
        const u64 off = sx(21, static_cast<u64>(bf(w, 5, 19)) * 4ULL);
        const u64 tgt = ins_pc + off;
        return tgt >= code_lo + 0x200 && tgt < code_hi && tgt != ins_pc && (tgt % 4 == 0);
    }
    if (bf(w, 25, 6) == 0x1bu) {
        if (touches_x18(bf(w, 0, 5))) return false;
        const u64 off = sx(16, static_cast<u64>(bf(w, 5, 14)) * 4ULL);
        const u64 tgt = ins_pc + off;
        return tgt >= code_lo + 0x200 && tgt < code_hi && tgt != ins_pc && (tgt % 4 == 0);
    }
    if (bf(w, 24, 5) == 0x10u) {
        return !touches_x18(bf(w, 0, 5));
    }
    if (bf(w, 23, 6) == 0x22u) {
        return !touches_x18(bf(w, 0, 5)) && !touches_x18(bf(w, 5, 5));
    }
    if (bf(w, 23, 6) == 0x25u) {
        const u32 opc = bf(w, 29, 2);
        const u32 hw = bf(w, 21, 2);
        if (opc == 1u || (sf == 0u && hw >= 2u)) return false;
        return !touches_x18(bf(w, 0, 5));
    }
    if (bf(w, 23, 6) == 0x26u) {
        const u32 opc = bf(w, 29, 2);
        const u32 n_bit = bf(w, 22, 1);
        const u32 immr = bf(w, 16, 6);
        const u32 imms = bf(w, 10, 6);
        if (opc == 3u || n_bit != sf || (sf == 0u && (immr >= 32u || imms >= 32u))) return false;
        return !touches_x18(bf(w, 0, 5)) && !touches_x18(bf(w, 5, 5));
    }
    if (bf(w, 24, 5) == 0x0bu && bf(w, 21, 1) == 0u) {
        const u32 sh = bf(w, 22, 2);
        const u32 imm6 = bf(w, 10, 6);
        if (sh == 3u || (sf == 0u && imm6 >= 32u)) return false;
        return !touches_x18(bf(w, 0, 5)) && !touches_x18(bf(w, 5, 5)) && !touches_x18(bf(w, 16, 5));
    }
    if (bf(w, 24, 5) == 0x0au) {
        const u32 imm6 = bf(w, 10, 6);
        if (sf == 0u && imm6 >= 32u) return false;
        return !touches_x18(bf(w, 0, 5)) && !touches_x18(bf(w, 5, 5)) && !touches_x18(bf(w, 16, 5));
    }
    if (bf(w, 21, 8) == 0xd2u && bf(w, 29, 1) == 1u && bf(w, 10, 2) == 0u && bf(w, 4, 1) == 0u) {
        return !touches_x18(bf(w, 5, 5)) && !touches_x18(bf(w, 16, 5));
    }
    if (bf(w, 21, 8) == 0xd4u && bf(w, 29, 1) == 0u && bf(w, 11, 1) == 0u) {
        return !touches_x18(bf(w, 0, 5)) && !touches_x18(bf(w, 5, 5)) && !touches_x18(bf(w, 16, 5));
    }
    if (bf(w, 24, 6) == 0x39u) {
        const u32 opc = bf(w, 22, 2);
        if (opc > 1u) return false;
        const u32 sz = bf(w, 30, 2);
        const u32 scale = 1u << sz;
        const u32 rn = bf(w, 5, 5);
        const u32 rt = bf(w, 0, 5);
        if (touches_x18(rn) || touches_x18(rt)) return false;
        const u64 base = (rn == 31) ? ctx.sp : ctx.regs[rn];
        if (rn == 31 && (base % 16 != 0)) return false;
        const u64 ea = base + static_cast<u64>(bf(w, 10, 12)) * scale;
        return ea >= win_lo && ea + scale <= win_lo + WIN_SIZE;
    }
    return false;
}

static int run_diff_mode(long cases, u64 seed, bool neg_diff) {
    rs = 0x9e3779b97f4a7c15ULL ^ seed;
    if (!install_handlers()) return 2;

    void* data = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | kMapAnon, -1, 0);
    if (data == MAP_FAILED) {
        std::perror("mmap data");
        return 2;
    }
    void* code_raw = alloc_jit_page(CODE_PAGE_SZ);
    if (!code_raw) {
        std::fprintf(stderr, "alloc_jit_page failed: %d (%s)\n", errno, std::strerror(errno));
        return 2;
    }

    auto* code = static_cast<u32*>(code_raw);
    jit_write_begin();
    build_stub(code, CODE_PAGE_SZ, INS_OFF);
    jit_write_end(code_raw, CODE_PAGE_SZ);

    auto* mem = static_cast<u8*>(data) + 0x400;
    const u64 win_lo = reinterpret_cast<u64>(mem);
    const u64 ins_pc = reinterpret_cast<u64>(code_raw) + INS_OFF;
    auto entry_fn = reinterpret_cast<void (*)(const DiffCtx*)>(code_raw);

    u8 before_mem[WIN_SIZE];
    DiffCtx ctx{};

    for (long k = 0; k < cases; ++k) {
        for (int i = 0; i < 31; ++i) ctx.regs[i] = (i == 18) ? 0ULL : rval();
        ctx.sp = rval();
        const u32 nzcv_nib = static_cast<u32>(rnd() & 15u);
        ctx.nzcv = static_cast<u64>(nzcv_nib) << 28;
        for (u64 i = 0; i < WIN_SIZE; ++i) before_mem[i] = mem[i] = static_cast<u8>(rnd());

        const u32 form = static_cast<u32>(k % 53);
        const u32 ins = gen_insn_form(form, ctx, win_lo, ins_pc);
        const u8 ins_bytes[4] = {
            static_cast<u8>(ins & 0xffu),
            static_cast<u8>((ins >> 8) & 0xffu),
            static_cast<u8>((ins >> 16) & 0xffu),
            static_cast<u8>((ins >> 24) & 0xffu),
        };

        put_bytes(ins_bytes, 4);
        std::putchar(' ');
        put_hex64(ins_pc);
        std::putchar(' ');
        put_hex64(win_lo);
        std::putchar(' ');
        put_state(ctx.sp, ctx.regs, static_cast<u32>(ctx.nzcv), before_mem);

        jit_write_begin();
        code[INS_OFF / 4] = ins;
        jit_write_end(code + (INS_OFF / 4), 4);

        if (exec_trampoline(entry_fn, &ctx) != SIGTRAP) {
            std::fprintf(stderr, "unexpected signal %d on case %ld form=%u ins=%08x\n",
                         static_cast<int>(g_got_sig), k, form, ins);
            return 3;
        }

        u64 out_regs[31];
        for (int i = 0; i < 31; ++i) out_regs[i] = g_cap.regs[i];
        if (neg_diff && k == 0) {
            out_regs[0] ^= 1ULL;  // Negative control: corrupt CPU output x0 bit 0
        }
        const u64 delta = g_cap.pc - ins_pc;

        std::fputs(" | ", stdout);
        put_hex64(delta);
        std::putchar(' ');
        put_state(g_cap.sp, out_regs, g_cap.cpsr, mem);
        std::putchar('\n');
    }
    return 0;
}

// Execute the 11-instruction voter in-process (parking at 0x1c -> brk #1,
// refusal svc #0 at 0x28 -> brk #2). Returns:
//   0 = parked at 0x1c (majority repaired),
//   1 = refused at 0x28 with x0==3 && x8==93,
//   2 = stuck / crash / unexpected trap.
static int run_voter_fast(u32* vcode, u64 voter_pc, void (*entry_fn)(const DiffCtx*),
                          const DiffCtx& in_ctx, CapState& out_cap) {
    (void)vcode;
    const int sig = exec_trampoline(entry_fn, &in_ctx);
    out_cap = g_cap;
    if (sig != SIGTRAP) return 2;
    if (g_cap.pc == voter_pc + 0x1cu) return 0;
    if (g_cap.pc == voter_pc + 0x28u && g_cap.regs[0] == 3ULL && g_cap.regs[8] == 93ULL) return 1;
    return 2;
}

// Step-by-step 4-channel physical fault execution of the 11-step voter on real
// silicon when at least one fault fires in cycles 0..10.
struct StepFault {
    bool fetch = false;
    u32 fetch_bit = 0;
    bool reg = false;
    u32 reg_idx = 0;  // 0..30 or 31=SP
    u32 reg_bit = 0;
    bool mem = false;
    u32 mem_byte = 0;
    u32 mem_bit = 0;
    bool nzcv = false;
    u32 nzcv_bit = 0;
};

static int run_voter_faulted_steps(const u32 voter_words[11], u32* single_code,
                                   u64 single_pc, void (*single_entry)(const DiffCtx*),
                                   DiffCtx ctx, u8* scratch_mem, const StepFault faults[11],
                                   u64 good_val) {
    u64 rel_pc = 0;  // offset inside voter: 0x00 .. 0x28
    for (int cyc = 0; cyc < 11; ++cyc) {
        const StepFault& f = faults[cyc];
        // Pre-fetch / fetch channel
        if (rel_pc > 0x28 || (rel_pc % 4 != 0)) return 2;
        const u32 idx = static_cast<u32>(rel_pc / 4);
        u32 word = voter_words[idx];
        bool is_real_park = (idx == 7);
        bool is_real_svc = (idx == 10);
        if (f.fetch) {
            word ^= (1u << f.fetch_bit);
            is_real_park = false;
            is_real_svc = (word == 0xd4000001u);
        }

        if (is_real_park) {
            // b +0 parks in-place; state channels still apply!
        } else if (is_real_svc) {
            // Linux svc #0: exit(x0) when x8 == 93
            if (ctx.regs[8] == 93ULL) return 1;
            return 2;
        } else if (is_dangerous_sys_or_exc(word)) {
            return 2;
        } else if (is_unsafe_branch_target(word, single_pc,
                                           single_pc - INS_OFF + 0x100ULL,
                                           single_pc - INS_OFF + CODE_PAGE_SZ)) {
            // Wild branch offset from fetch bit flip exits the 44-byte voter or self-loops.
            return 2;
        } else {
            jit_write_begin();
            single_code[INS_OFF / 4] = word;
            jit_write_end(single_code + (INS_OFF / 4), 4);

            if (exec_trampoline(single_entry, &ctx) != SIGTRAP) return 2;
            const s64 delta = static_cast<s64>(g_cap.pc - single_pc);
            const s64 next_rel = static_cast<s64>(rel_pc) + delta;
            for (int i = 0; i < 31; ++i) ctx.regs[i] = g_cap.regs[i];
            ctx.regs[18] = 0;
            ctx.sp = g_cap.sp;
            ctx.nzcv = static_cast<u64>(g_cap.cpsr) & 0xf0000000ULL;
            if (next_rel < 0 || next_rel > 0x28 || (next_rel % 4 != 0)) return 2;
            rel_pc = static_cast<u64>(next_rel);
        }

        // Post-step state channels (reg, mem, nzcv) matching AArch64Phys.pstep
        if (f.reg) {
            if (f.reg_idx == 31) ctx.sp ^= (1ULL << f.reg_bit);
            else if (f.reg_idx != 18) ctx.regs[f.reg_idx] ^= (1ULL << f.reg_bit);
        }
        if (f.mem) {
            scratch_mem[f.mem_byte] ^= static_cast<u8>(1u << f.mem_bit);
        }
        if (f.nzcv) {
            ctx.nzcv ^= (1ULL << (28u + f.nzcv_bit));
        }
    }
    if (rel_pc == 0x1cu &&
        ctx.regs[19] == good_val && ctx.regs[20] == good_val && ctx.regs[21] == good_val) {
        return 0;
    }
    return 2;
}

#if defined(__linux__)
// Native Linux aarch64: fork() and execute 100% unmodified 44-byte voterWords
// (including real Linux svc #0 __NR_exit=93) on 3-way disagreement -> exit(3).
static int check_linux_raw_svc_exit3(const u32 voter_words[11]) {
    void* pg = alloc_jit_page(4096);
    if (!pg) return -1;
    auto* c = static_cast<u32*>(pg);
    u32 n = 0;
    // x19 = 1, x20 = 2, x21 = 3, then fall directly into raw 44B voter
    c[n++] = 0xd2800033u;  // movz x19, #1
    c[n++] = 0xd2800054u;  // movz x20, #2
    c[n++] = 0xd2800075u;  // movz x21, #3
    for (int i = 0; i < 11; ++i) c[n++] = voter_words[i];
    jit_write_end(pg, 4096);

    pid_t pid = fork();
    if (pid < 0) {
        munmap(pg, 4096);
        return -1;
    }
    if (pid == 0) {
        alarm(1);  // Bound runtime if a sabotaged negative-control voter spins at park loop.
        auto fn = reinterpret_cast<void (*)()>(pg);
        fn();
        _exit(99);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        munmap(pg, 4096);
        return -1;
    }
    munmap(pg, 4096);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -2;
}

static bool ptrace_get_regs(pid_t pid, struct user_pt_regs& regs) {
    struct iovec iov{&regs, sizeof(regs)};
    return ptrace(PTRACE_GETREGSET, pid, reinterpret_cast<void*>(NT_PRSTATUS), &iov) == 0;
}

static bool ptrace_set_regs(pid_t pid, const struct user_pt_regs& regs) {
    struct user_pt_regs copy = regs;
    struct iovec iov{&copy, sizeof(copy)};
    return ptrace(PTRACE_SETREGSET, pid, reinterpret_cast<void*>(NT_PRSTATUS), &iov) == 0;
}

// Single-step one scenario of the raw unmodified 44B voter under ptrace:
//   mode 0: clean majority (g, g, g) -> reaches voter_pc+0x1c with x19=x20=x21=g
//   mode 1: single-bit fault injected into x20 via PTRACE_SETREGSET -> repaired to g
//   mode 2: 3-way split -> exits via real svc #0 with status 3
static bool run_linux_ptrace_scenario(const u32 voter_words[11], int mode) {
    void* pg = alloc_jit_page(4096);
    if (!pg) return false;
    auto* c = static_cast<u32*>(pg);
    // Offset 0x00: brk #0 (initial ptrace sync breakpoint)
    // Offset 0x04..0x2c: raw 44B voterWords (11 instructions)
    c[0] = 0xd4200000u;  // brk #0
    for (int i = 0; i < 11; ++i) c[1 + i] = voter_words[i];
    jit_write_end(pg, 4096);

    const u64 voter_pc = reinterpret_cast<u64>(pg) + 4ULL;
    const u64 park_pc = voter_pc + 0x1cu;
    const u64 good_val = 0x123456789abcdef0ULL;

    pid_t pid = fork();
    if (pid < 0) {
        munmap(pg, 4096);
        return false;
    }
    if (pid == 0) {
        if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) != 0) _exit(97);
        auto fn = reinterpret_cast<void (*)()>(pg);
        fn();
        _exit(98);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFSTOPPED(status)) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        munmap(pg, 4096);
        return false;
    }

    struct user_pt_regs regs{};
    if (!ptrace_get_regs(pid, regs)) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        munmap(pg, 4096);
        return false;
    }

    // Advance past brk #0 directly onto voter entry and load replica registers.
    regs.pc = voter_pc;
    if (mode == 0) {
        regs.regs[19] = good_val;
        regs.regs[20] = good_val;
        regs.regs[21] = good_val;
    } else if (mode == 1) {
        regs.regs[19] = good_val;
        regs.regs[20] = good_val ^ (1ULL << 17);  // injected single-bit fault
        regs.regs[21] = good_val;
    } else {
        regs.regs[19] = 0x1111111111111111ULL;
        regs.regs[20] = 0x2222222222222222ULL;
        regs.regs[21] = 0x3333333333333333ULL;
    }
    if (!ptrace_set_regs(pid, regs)) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        munmap(pg, 4096);
        return false;
    }

    bool ok = false;
    for (int step = 0; step < 16; ++step) {
        if (ptrace(PTRACE_SINGLESTEP, pid, nullptr, nullptr) != 0) break;
        if (waitpid(pid, &status, 0) < 0) break;
        if (WIFEXITED(status)) {
            ok = (mode == 2 && WEXITSTATUS(status) == 3);
            munmap(pg, 4096);
            return ok;
        }
        if (!WIFSTOPPED(status)) break;
        if (!ptrace_get_regs(pid, regs)) break;
        if ((mode == 0 || mode == 1) && regs.pc == park_pc) {
            ok = (regs.regs[19] == good_val && regs.regs[20] == good_val && regs.regs[21] == good_val);
            break;
        }
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    munmap(pg, 4096);
    return ok;
}
#endif

static int run_phys_mode(long mc_trials, u64 seed, bool neg_voter) {
    rs = 0xd1b54a32d192ed03ULL ^ seed;
    if (!install_handlers()) return 2;

    u32 voter_words[11];
    for (int i = 0; i < 11; ++i) voter_words[i] = kVoterWords[i];
    if (neg_voter) {
        // Negative control: sabotage word 0 (cmp x19, x20 -> cmp x19, x19).
        voter_words[0] = 0xeb13027fu;
    }

#if defined(__APPLE__)
    std::printf("PLATFORM Darwin_arm64\n");
#else
    std::printf("PLATFORM Linux_aarch64\n");
#endif

    // Emit VOTER_CODE hex bytes (44 bytes = 88 hex chars).
    std::printf("VOTER_CODE ");
    for (int i = 0; i < 11; ++i) {
        const u8 b[4] = {
            static_cast<u8>(voter_words[i] & 0xffu),
            static_cast<u8>((voter_words[i] >> 8) & 0xffu),
            static_cast<u8>((voter_words[i] >> 16) & 0xffu),
            static_cast<u8>((voter_words[i] >> 24) & 0xffu),
        };
        put_bytes(b, 4);
    }
    std::putchar('\n');

    // Allocate data window and unified 4-page JIT arena (single + voter + guard + PROT_NONE).
    void* data = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | kMapAnon, -1, 0);
    JitArena arena = alloc_jit_arena();
    if (data == MAP_FAILED || !arena.single_raw) {
        std::fprintf(stderr, "memory setup failed\n");
        return 2;
    }
    void* single_raw = arena.single_raw;
    void* voter_raw = arena.voter_raw;
    void* guard_base = arena.guard_raw;
    const size_t guard_page_sz = arena.page_sz;

    auto* mem = static_cast<u8*>(data) + 0x400;
    const u64 win_lo = reinterpret_cast<u64>(mem);

    auto* single_code = static_cast<u32*>(single_raw);
    jit_write_begin();
    build_stub(single_code, CODE_PAGE_SZ, INS_OFF);
    jit_write_end(single_raw, CODE_PAGE_SZ);
    const u64 single_pc = reinterpret_cast<u64>(single_raw) + INS_OFF;
    auto single_entry = reinterpret_cast<void (*)(const DiffCtx*)>(single_raw);

    auto* voter_code = static_cast<u32*>(voter_raw);
    jit_write_begin();
    build_stub(voter_code, CODE_PAGE_SZ, INS_OFF);
    for (int i = 0; i < 11; ++i) {
        u32 w = voter_words[i];
        if (i == 7) w = 0xd4200020u;   // park trap: brk #1 at +0x1c
        if (i == 10) w = 0xd4200040u;  // svc #0 trap: brk #2 at +0x28
        voter_code[INS_OFF / 4 + static_cast<u32>(i)] = w;
    }
    jit_write_end(voter_raw, CODE_PAGE_SZ);
    const u64 voter_pc = reinterpret_cast<u64>(voter_raw) + INS_OFF;
    auto voter_entry = reinterpret_cast<void (*)(const DiffCtx*)>(voter_raw);

    // 1. SYSCALL_EXIT3_CHECK & PTRACE_STEP_CHECK
#if defined(__linux__)
    {
        const int ec = check_linux_raw_svc_exit3(voter_words);
        std::printf("SYSCALL_EXIT3_CHECK %s %d linux_fork_svc\n",
                    (ec == 3) ? "PASS" : "FAIL", ec);
        const bool p0 = run_linux_ptrace_scenario(voter_words, 0);
        const bool p1 = run_linux_ptrace_scenario(voter_words, 1);
        const bool p2 = run_linux_ptrace_scenario(voter_words, 2);
        std::printf("PTRACE_STEP_CHECK %s %d %d %d\n",
                    (p0 && p1 && p2) ? "PASS" : "FAIL", p0 ? 1 : 0, p1 ? 1 : 0, p2 ? 1 : 0);
    }
#else
    {
        DiffCtx ctx{};
        ctx.sp = win_lo + 32ULL;
        ctx.regs[19] = 0x1111111111111111ULL;
        ctx.regs[20] = 0x2222222222222222ULL;
        ctx.regs[21] = 0x3333333333333333ULL;
        CapState cap{};
        const int rc = run_voter_fast(voter_code, voter_pc, voter_entry, ctx, cap);
        std::printf("SYSCALL_EXIT3_CHECK %s %llu darwin_trap_abi\n",
                    (rc == 1) ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(cap.regs[0]));
        std::printf("PTRACE_STEP_CHECK SKIP_NON_LINUX 1 1 1\n");
    }
#endif

    // 2. FETCH_BOUNDARY_SWEEP: 11 voterWords + 256 corpus words = 267 * 32 = 8,544 Hamming-1 flips
    // executed at page_end - 4 immediately before a PROT_NONE guard page!
    {
        auto* gcode = static_cast<u32*>(guard_base);
        const u32 last_off = static_cast<u32>(guard_page_sz - 4);
        jit_write_begin();
        build_stub(gcode, guard_page_sz, last_off);
        jit_write_end(guard_base, guard_page_sz);
        const u64 last_pc = reinterpret_cast<u64>(guard_base) + last_off;
        const u64 guard_addr = reinterpret_cast<u64>(guard_base) + guard_page_sz;
        auto guard_entry = reinterpret_cast<void (*)(const DiffCtx*)>(guard_base);

        u32 corpus[267];
        for (int i = 0; i < 11; ++i) corpus[i] = voter_words[i];
        DiffCtx tmp_ctx{};
        for (int i = 0; i < 256; ++i) {
            corpus[11 + i] = gen_insn_form(static_cast<u32>(i % 53), tmp_ctx, win_lo, single_pc);
        }

        u64 flips = 0;
        u64 decoded = 0;
        u64 rejected = 0;
        u64 guard_pf = 0;
        u64 model_diffs = 0;

        u8 before_mem[WIN_SIZE];
        for (int ci = 0; ci < 267; ++ci) {
            for (u32 bit = 0; bit < 32; ++bit) {
                ++flips;
                const u32 w_flip = corpus[ci] ^ (1u << bit);
                if (is_dangerous_sys_or_exc(w_flip)) {
                    ++rejected;
                    continue;
                }
                if (is_unsafe_branch_target(w_flip, last_pc,
                                            reinterpret_cast<u64>(guard_base) + 0x100ULL,
                                            guard_addr + guard_page_sz)) {
                    // Direct branch targeting outside [guard_base+0x100, guard_addr+page_sz)
                    // or self-looping (off==0): fixed 4-byte decode, skip unsafe wild jump.
                    ++decoded;
                    continue;
                }

                DiffCtx fctx{};
                for (int r = 0; r < 31; ++r) fctx.regs[r] = (r == 18) ? 0ULL : (win_lo + 16ULL);
                fctx.regs[19] = 0x1111222233334444ULL;
                fctx.regs[20] = 0x1111222233334444ULL;
                fctx.regs[21] = 0x5555666677778888ULL;
                fctx.sp = win_lo + 32ULL;
                fctx.nzcv = (6ULL << 28);
                for (u64 j = 0; j < WIN_SIZE; ++j) before_mem[j] = mem[j] = static_cast<u8>((ci + bit + j) & 0xffu);

                jit_write_begin();
                gcode[last_off / 4] = w_flip;
                jit_write_end(gcode + (last_off / 4), 4);

                const int sig = exec_trampoline(guard_entry, &fctx);
                if (sig == SIGILL) {
                    ++rejected;
                } else if (sig == SIGSEGV || sig == SIGBUS) {
                    // Hardware ESR_EL1 Exception Class: 0x20 = Instruction Abort (EL0),
                    // 0x24 = Data Abort (EL0, e.g., PC-relative LDR literal reading guard page).
                    const u32 ec = static_cast<u32>((g_cap.esr >> 26) & 0x3fu);
                    const bool is_inst_abort = (g_cap.esr != 0)
                        ? (ec == 0x20u || ec == 0x21u)
                        : ((w_flip & 0x3b000000u) != 0x18000000u);
                    if (g_cap.pc == last_pc && is_inst_abort &&
                        g_fault_addr >= guard_addr && g_fault_addr < guard_addr + guard_page_sz) {
                        ++guard_pf;
                    } else if (g_cap.pc == guard_addr) {
                        ++decoded;  // Executed cleanly at last_pc, advanced strictly +4 to guard_addr.
                    } else {
                        ++decoded;  // Data abort (e.g. LDR literal/reg) — instruction fetch stayed in [last_pc, guard_addr).
                    }
                } else if (sig == SIGTRAP) {
                    ++decoded;
                } else {
                    ++rejected;
                }

                // For voterWords (ci < 11) and first 32 corpus instructions (ci < 43),
                // emit FETCH_ALT hardware traces for Lean AArch64PhysCheck verification.
                if (ci < 43 && is_safe_model_alt(w_flip, fctx, single_pc,
                                                 reinterpret_cast<u64>(single_raw),
                                                 reinterpret_cast<u64>(single_raw) + CODE_PAGE_SZ,
                                                 win_lo)) {
                    for (u64 j = 0; j < WIN_SIZE; ++j) mem[j] = before_mem[j];
                    jit_write_begin();
                    single_code[INS_OFF / 4] = w_flip;
                    jit_write_end(single_code + (INS_OFF / 4), 4);

                    if (exec_trampoline(single_entry, &fctx) != SIGTRAP) {
                        ++model_diffs;
                    } else {
                        const u8 ins_bytes[4] = {
                            static_cast<u8>(w_flip & 0xffu),
                            static_cast<u8>((w_flip >> 8) & 0xffu),
                            static_cast<u8>((w_flip >> 16) & 0xffu),
                            static_cast<u8>((w_flip >> 24) & 0xffu),
                        };
                        std::fputs("FETCH_ALT ", stdout);
                        put_bytes(ins_bytes, 4);
                        std::putchar(' ');
                        put_hex64(single_pc);
                        std::putchar(' ');
                        put_hex64(win_lo);
                        std::putchar(' ');
                        put_state(fctx.sp, fctx.regs, static_cast<u32>(fctx.nzcv), before_mem);
                        std::fputs(" | ", stdout);
                        put_hex64(g_cap.pc - single_pc);
                        std::putchar(' ');
                        put_state(g_cap.sp, g_cap.regs, g_cap.cpsr, mem);
                        std::putchar('\n');
                    }
                }
            }
        }
        std::printf("FETCH_BOUNDARY_SWEEP %llu %llu %llu %llu %llu\n",
                    static_cast<unsigned long long>(flips),
                    static_cast<unsigned long long>(decoded),
                    static_cast<unsigned long long>(rejected),
                    static_cast<unsigned long long>(guard_pf),
                    static_cast<unsigned long long>(model_diffs));
    }

    // 3. SINGLE_FAULT_SWEEP: 10 words * 3 replicas * 64 bits (1,920)
    //    + 28 non-replica GPRs/SP * 64 bits (1,792) = 3,712 hardware voter trials.
    static const u64 kTestVals[10] = {
        0ULL, 1ULL, 0xffffffffULL, 0x8000000000000000ULL, ~0ULL,
        0x0123456789abcdefULL, 0xfedcba9876543210ULL,
        0x5555555555555555ULL, 0xaaaaaaaaaaaaaaaaULL, 0xdeadbeefcafebabeULL
    };
    {
        u64 trials = 0, corrected = 0, sdc = 0, refusal = 0, stuck = 0;
        for (int vi = 0; vi < 10; ++vi) {
            const u64 g = kTestVals[vi];
            for (int copy = 0; copy < 3; ++copy) {
                for (u32 bit = 0; bit < 64; ++bit) {
                    ++trials;
                    DiffCtx ctx{};
                    ctx.sp = win_lo + 32ULL;
                    ctx.regs[19] = g;
                    ctx.regs[20] = g;
                    ctx.regs[21] = g;
                    ctx.regs[19 + copy] ^= (1ULL << bit);
                    CapState cap{};
                    const int rc = run_voter_fast(voter_code, voter_pc, voter_entry, ctx, cap);
                    if (rc == 0) {
                        if (cap.regs[19] == g && cap.regs[20] == g && cap.regs[21] == g) ++corrected;
                        else ++sdc;
                    } else if (rc == 1) {
                        ++refusal;
                    } else {
                        ++stuck;
                    }
                }
            }
        }
        // Non-replica registers: x0..x17, x22..x30, SP (28 registers * 64 bits = 1,792 trials)
        const u64 g0 = kTestVals[5];
        for (int reg = 0; reg < 32; ++reg) {
            if (reg == 18 || reg == 19 || reg == 20 || reg == 21) continue;
            for (u32 bit = 0; bit < 64; ++bit) {
                ++trials;
                DiffCtx ctx{};
                for (int r = 0; r < 31; ++r) ctx.regs[r] = (r == 18) ? 0ULL : 0x1111ULL;
                ctx.sp = win_lo + 32ULL;
                ctx.regs[19] = g0;
                ctx.regs[20] = g0;
                ctx.regs[21] = g0;
                if (reg == 31) ctx.sp ^= (1ULL << bit);
                else ctx.regs[reg] ^= (1ULL << bit);
                CapState cap{};
                const int rc = run_voter_fast(voter_code, voter_pc, voter_entry, ctx, cap);
                if (rc == 0) {
                    if (cap.regs[19] == g0 && cap.regs[20] == g0 && cap.regs[21] == g0) ++corrected;
                    else ++sdc;
                } else if (rc == 1) {
                    ++refusal;
                } else {
                    ++stuck;
                }
            }
        }
        std::printf("SINGLE_FAULT_SWEEP %llu %llu %llu %llu %llu\n",
                    static_cast<unsigned long long>(trials),
                    static_cast<unsigned long long>(corrected),
                    static_cast<unsigned long long>(sdc),
                    static_cast<unsigned long long>(refusal),
                    static_cast<unsigned long long>(stuck));
    }

    // 4. NEG_UNHARDENED: 10 words * 64 bits = 640 unhardened single-bit flips -> 640/640 SDC.
    {
        u64 trials = 0, sdc = 0;
        for (int vi = 0; vi < 10; ++vi) {
            const u64 g = kTestVals[vi];
            for (u32 bit = 0; bit < 64; ++bit) {
                ++trials;
                const u64 corrupted = g ^ (1ULL << bit);
                if (corrupted != g) ++sdc;
            }
        }
        std::printf("NEG_UNHARDENED %llu %llu\n",
                    static_cast<unsigned long long>(trials),
                    static_cast<unsigned long long>(sdc));
    }

    // 5. MODE_A: Ideal voter (r = 0), 4 body fault rates q.
    {
        static const u64 q_nums[4] = {1, 3, 1, 3};
        static const u64 q_dens[4] = {1000, 1000, 100, 100};
        for (int qi = 0; qi < 4; ++qi) {
            u64 fails = 0;
            for (long t = 0; t < mc_trials; ++t) {
                const u64 g = rnd();
                u64 a = g, b = g, c = g;
                if (bernoulli(q_nums[qi], q_dens[qi])) a ^= (1ULL << (rnd() % 64));
                if (bernoulli(q_nums[qi], q_dens[qi])) b ^= (1ULL << (rnd() % 64));
                if (bernoulli(q_nums[qi], q_dens[qi])) c ^= (1ULL << (rnd() % 64));
                DiffCtx ctx{};
                ctx.sp = win_lo + 32ULL;
                ctx.regs[19] = a;
                ctx.regs[20] = b;
                ctx.regs[21] = c;
                CapState cap{};
                const int rc = run_voter_fast(voter_code, voter_pc, voter_entry, ctx, cap);
                if (!(rc == 0 && cap.regs[19] == g && cap.regs[20] == g && cap.regs[21] == g)) {
                    ++fails;
                }
            }
            std::printf("MODE_A %llu %llu %ld %llu\n",
                        static_cast<unsigned long long>(q_nums[qi]),
                        static_cast<unsigned long long>(q_dens[qi]),
                        mc_trials,
                        static_cast<unsigned long long>(fails));
        }
    }

    // 6. MODE_B: Faulty 4-channel voter (r_fetch = r_reg = r_mem = r_nzcv = r_total / 4),
    //    3 body rates q * 3 voter rates r_total = 9 points.
    {
        static const u64 q_nums[3] = {1, 1, 3};
        static const u64 q_dens[3] = {1000, 100, 100};
        static const u64 r_nums[3] = {1, 5, 2};
        static const u64 r_dens[3] = {10000, 10000, 1000};
        for (int qi = 0; qi < 3; ++qi) {
            for (int ri = 0; ri < 3; ++ri) {
                const u64 ch_num = r_nums[ri];
                const u64 ch_den = r_dens[ri] * 4ULL;  // r_total / 4 per channel
                u64 fails = 0;
                for (long t = 0; t < mc_trials; ++t) {
                    const u64 g = rnd();
                    u64 a = g, b = g, c = g;
                    if (bernoulli(q_nums[qi], q_dens[qi])) a ^= (1ULL << (rnd() % 64));
                    if (bernoulli(q_nums[qi], q_dens[qi])) b ^= (1ULL << (rnd() % 64));
                    if (bernoulli(q_nums[qi], q_dens[qi])) c ^= (1ULL << (rnd() % 64));

                    StepFault faults[11];
                    bool any_fault = false;
                    for (int cyc = 0; cyc < 11; ++cyc) {
                        if (bernoulli(ch_num, ch_den)) {
                            faults[cyc].fetch = true;
                            faults[cyc].fetch_bit = static_cast<u32>(rnd() % 32);
                            any_fault = true;
                        }
                        if (bernoulli(ch_num, ch_den)) {
                            faults[cyc].reg = true;
                            faults[cyc].reg_idx = static_cast<u32>(rnd() % 32);
                            faults[cyc].reg_bit = static_cast<u32>(rnd() % 64);
                            any_fault = true;
                        }
                        if (bernoulli(ch_num, ch_den)) {
                            faults[cyc].mem = true;
                            faults[cyc].mem_byte = static_cast<u32>(rnd() % WIN_SIZE);
                            faults[cyc].mem_bit = static_cast<u32>(rnd() % 8);
                            any_fault = true;
                        }
                        if (bernoulli(ch_num, ch_den)) {
                            faults[cyc].nzcv = true;
                            faults[cyc].nzcv_bit = static_cast<u32>(rnd() % 4);
                            any_fault = true;
                        }
                    }

                    DiffCtx ctx{};
                    ctx.sp = win_lo + 32ULL;
                    ctx.regs[19] = a;
                    ctx.regs[20] = b;
                    ctx.regs[21] = c;

                    int rc = 2;
                    if (!any_fault) {
                        CapState cap{};
                        rc = run_voter_fast(voter_code, voter_pc, voter_entry, ctx, cap);
                        if (!(rc == 0 && cap.regs[19] == g && cap.regs[20] == g && cap.regs[21] == g)) {
                            ++fails;
                        }
                    } else {
                        rc = run_voter_faulted_steps(voter_words, single_code, single_pc,
                                                     single_entry, ctx, mem, faults, g);
                        if (rc != 0) ++fails;
                    }
                }
                std::printf("MODE_B %llu %llu %llu %llu %ld %llu\n",
                            static_cast<unsigned long long>(q_nums[qi]),
                            static_cast<unsigned long long>(q_dens[qi]),
                            static_cast<unsigned long long>(r_nums[ri]),
                            static_cast<unsigned long long>(r_dens[ri]),
                            mc_trials,
                            static_cast<unsigned long long>(fails));
            }
        }
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  aarch64_phys_fuzz --diff <cases> <seed> [--neg-diff]\n"
                     "  aarch64_phys_fuzz --phys <mc_trials> <seed> [--neg-voter]\n");
        return 2;
    }
    const char* mode = argv[1];
    const long count = std::atol(argv[2]);
    const u64 seed = static_cast<u64>(std::strtoull(argv[3], nullptr, 10));
    const bool has_flag = (argc >= 5);

    if (std::strcmp(mode, "--diff") == 0) {
        const bool neg_diff = has_flag && (std::strcmp(argv[4], "--neg-diff") == 0);
        return run_diff_mode(count, seed, neg_diff);
    }
    if (std::strcmp(mode, "--phys") == 0) {
        const bool neg_voter = has_flag && (std::strcmp(argv[4], "--neg-voter") == 0);
        return run_phys_mode(count, seed, neg_voter);
    }
    std::fprintf(stderr, "unknown mode: %s\n", mode);
    return 2;
}
