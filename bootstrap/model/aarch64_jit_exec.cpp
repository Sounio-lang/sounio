// aarch64_jit_exec: native Apple Silicon (Darwin arm64) MAP_JIT runner for
// static EM_AARCH64 Linux ELF64 binaries (specifically bootstrap/roots/c/seed-c).
//
// Maps PT_LOAD segments into MAP_JIT memory, hooks Linux AArch64 `svc #0`
// syscalls (63 = read, 64 = write, 93 = exit) via brk #0x517 + SIGTRAP on an
// alternate signal stack, and executes the guest binary natively on Apple Silicon.
//
// Usage:
//   aarch64_jit_exec <elf64>
//   aarch64_jit_exec --batch <elf64> <dir> <count>
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>

#include <cerrno>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

static constexpr u32 INSN_SVC0 = 0xd4000001u;
static constexpr u32 INSN_BRK_SVC = 0xd420a2e0u; // brk #0x517
static constexpr size_t GUEST_STACK_SZ = 65536;

static u16 rd16(const u8* p) {
    return static_cast<u16>(p[0]) | (static_cast<u16>(p[1]) << 8);
}
static u32 rd32(const u8* p) {
    return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
           (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}
static u64 rd64(const u8* p) {
    u64 v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<u64>(p[i]) << (8 * i);
    return v;
}

struct GuestEnv {
    u8* jit_base = nullptr;
    size_t jit_sz = 0;
    u8* stack_base = nullptr;
    size_t stack_sz = 0;
    bool batch_mode = false;
    const std::string* batch_in = nullptr;
    size_t batch_in_pos = 0;
    std::string* batch_out = nullptr;
    int exit_code = 0;
    bool exited = false;
    u64 bad_syscall = 0;
};

static GuestEnv g_env;
static sigjmp_buf g_jb;

static bool valid_guest_ptr(u64 addr, u64 len, bool write_needed) {
    if (addr + len < addr) return false;
    const u64 st_lo = reinterpret_cast<u64>(g_env.stack_base);
    const u64 st_hi = st_lo + g_env.stack_sz;
    if (addr >= st_lo && addr + len <= st_hi) return true;
    if (!write_needed) {
        const u64 j_lo = reinterpret_cast<u64>(g_env.jit_base);
        const u64 j_hi = j_lo + g_env.jit_sz;
        if (addr >= j_lo && addr + len <= j_hi) return true;
    }
    return false;
}

static void trap_handler(int sig, siginfo_t*, void* uctx) {
    if (sig != SIGTRAP) siglongjmp(g_jb, 3);
    auto* uc = static_cast<ucontext_t*>(uctx);
    auto& ss = uc->uc_mcontext->__ss;
    const u64 pc = ss.__pc;
    const u64 j_lo = reinterpret_cast<u64>(g_env.jit_base);
    const u64 j_hi = j_lo + g_env.jit_sz;
    if (pc < j_lo || pc + 4 > j_hi || *reinterpret_cast<const u32*>(pc) != INSN_BRK_SVC) {
        siglongjmp(g_jb, 3);
    }
    const u64 sysno = ss.__x[8];
    if (sysno == 63) { // Linux AArch64 read(fd, buf, count)
        const u64 fd = ss.__x[0];
        const u64 buf = ss.__x[1];
        const u64 cnt = ss.__x[2];
        if (!valid_guest_ptr(buf, cnt, true)) {
            ss.__x[0] = static_cast<u64>(-14LL); // -EFAULT
        } else if (g_env.batch_mode) {
            if (fd != 0) {
                ss.__x[0] = static_cast<u64>(-9LL); // -EBADF
            } else {
                const size_t avail = g_env.batch_in->size() - g_env.batch_in_pos;
                const size_t take = cnt < avail ? static_cast<size_t>(cnt) : avail;
                if (take > 0) {
                    std::memcpy(reinterpret_cast<void*>(buf),
                                g_env.batch_in->data() + g_env.batch_in_pos, take);
                    g_env.batch_in_pos += take;
                }
                ss.__x[0] = static_cast<u64>(take);
            }
        } else {
            ssize_t n;
            do {
                n = ::read(static_cast<int>(fd), reinterpret_cast<void*>(buf),
                           static_cast<size_t>(cnt));
            } while (n < 0 && errno == EINTR);
            ss.__x[0] = (n >= 0) ? static_cast<u64>(n) : static_cast<u64>(-static_cast<int64_t>(errno));
        }
        ss.__pc += 4;
        return;
    }
    if (sysno == 64) { // Linux AArch64 write(fd, buf, count)
        const u64 fd = ss.__x[0];
        const u64 buf = ss.__x[1];
        const u64 cnt = ss.__x[2];
        if (!valid_guest_ptr(buf, cnt, false)) {
            ss.__x[0] = static_cast<u64>(-14LL);
        } else if (g_env.batch_mode) {
            if (fd != 1) {
                ss.__x[0] = static_cast<u64>(-9LL);
            } else {
                g_env.batch_out->append(reinterpret_cast<const char*>(buf), static_cast<size_t>(cnt));
                ss.__x[0] = cnt;
            }
        } else {
            ssize_t n;
            do {
                n = ::write(static_cast<int>(fd), reinterpret_cast<const void*>(buf),
                            static_cast<size_t>(cnt));
            } while (n < 0 && errno == EINTR);
            ss.__x[0] = (n >= 0) ? static_cast<u64>(n) : static_cast<u64>(-static_cast<int64_t>(errno));
        }
        ss.__pc += 4;
        return;
    }
    if (sysno == 93 || sysno == 94) { // exit / exit_group
        g_env.exit_code = static_cast<int>(ss.__x[0] & 0xff);
        g_env.exited = true;
        siglongjmp(g_jb, 1);
    }
    g_env.bad_syscall = sysno;
    siglongjmp(g_jb, 2);
}

static bool load_elf(const char* path, u64& entry_off, void*& tramp_mem, size_t& tramp_sz) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::fprintf(stderr, "aarch64_jit_exec: cannot open %s\n", path);
        return false;
    }
    std::vector<u8> raw((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (raw.size() < 64) {
        std::fprintf(stderr, "aarch64_jit_exec: ELF too small\n");
        return false;
    }
    if (raw[0] != 0x7f || raw[1] != 'E' || raw[2] != 'L' || raw[3] != 'F' ||
        raw[4] != 2 || raw[5] != 1 || raw[6] != 1) {
        std::fprintf(stderr, "aarch64_jit_exec: invalid ELF64 header\n");
        return false;
    }
    const u16 e_type = rd16(&raw[16]);
    const u16 e_machine = rd16(&raw[18]);
    const u64 e_entry = rd64(&raw[24]);
    const u64 e_phoff = rd64(&raw[32]);
    const u16 e_phentsize = rd16(&raw[54]);
    const u16 e_phnum = rd16(&raw[56]);
    if (e_type != 2 || e_machine != 183 || e_phentsize != 56 || e_phnum == 0 ||
        e_phoff + static_cast<u64>(e_phnum) * 56 > raw.size()) {
        std::fprintf(stderr, "aarch64_jit_exec: unsupported ELF64 fields (machine=%u)\n", e_machine);
        return false;
    }

    u64 min_va = ~0ULL, max_va = 0;
    for (u16 i = 0; i < e_phnum; ++i) {
        const u8* ph = &raw[e_phoff + static_cast<size_t>(i) * 56];
        if (rd32(ph) != 1) continue; // PT_LOAD
        const u64 p_off = rd64(ph + 8);
        const u64 p_va = rd64(ph + 16);
        const u64 p_filesz = rd64(ph + 32);
        const u64 p_memsz = rd64(ph + 40);
        if (p_memsz < p_filesz || p_off + p_filesz > raw.size()) {
            std::fprintf(stderr, "aarch64_jit_exec: invalid PT_LOAD\n");
            return false;
        }
        if (p_va < min_va) min_va = p_va;
        if (p_va + p_memsz > max_va) max_va = p_va + p_memsz;
    }
    if (min_va >= max_va || e_entry < min_va || e_entry >= max_va) {
        std::fprintf(stderr, "aarch64_jit_exec: invalid entry or segment range\n");
        return false;
    }

    const size_t page_sz = static_cast<size_t>(getpagesize());
    const size_t span = static_cast<size_t>(max_va - min_va);
    const size_t map_sz = ((span + page_sz - 1) / page_sz) * page_sz;
    void* mem = mmap(nullptr, map_sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (mem == MAP_FAILED) {
        std::perror("mmap MAP_JIT");
        return false;
    }

    const u64 hdr_end = e_phoff + static_cast<u64>(e_phnum) * 56;
    pthread_jit_write_protect_np(0);
    std::memset(mem, 0, map_sz);
    for (u16 i = 0; i < e_phnum; ++i) {
        const u8* ph = &raw[e_phoff + static_cast<size_t>(i) * 56];
        if (rd32(ph) != 1) continue;
        const u32 p_flags = rd32(ph + 4);
        const u64 p_off = rd64(ph + 8);
        const u64 p_va = rd64(ph + 16);
        const u64 p_filesz = rd64(ph + 32);
        u8* dst = static_cast<u8*>(mem) + (p_va - min_va);
        std::memcpy(dst, raw.data() + p_off, static_cast<size_t>(p_filesz));
        if (p_flags & 1U) {
            u64 scan_start = (p_off == 0 && p_filesz > hdr_end) ? hdr_end : 0;
            scan_start = (scan_start + 3ULL) & ~3ULL;
            for (u64 off = scan_start; off + 4 <= p_filesz; off += 4) {
                u32 w = rd32(dst + off);
                if (w == INSN_SVC0) {
                    dst[off + 0] = static_cast<u8>(INSN_BRK_SVC & 0xff);
                    dst[off + 1] = static_cast<u8>((INSN_BRK_SVC >> 8) & 0xff);
                    dst[off + 2] = static_cast<u8>((INSN_BRK_SVC >> 16) & 0xff);
                    dst[off + 3] = static_cast<u8>((INSN_BRK_SVC >> 24) & 0xff);
                }
            }
        }
    }
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(mem, map_sz);

    // Entry trampoline in its own MAP_JIT page: sets sp=x0, poisons x0..x15/x17..x30, br x16.
    tramp_sz = page_sz;
    tramp_mem = mmap(nullptr, tramp_sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (tramp_mem == MAP_FAILED) {
        std::perror("mmap MAP_JIT tramp");
        return false;
    }
    auto* tc = static_cast<u32*>(tramp_mem);
    u32 tn = 0;
    pthread_jit_write_protect_np(0);
    tc[tn++] = 0x9100001fu; // mov sp, x0
    tc[tn++] = 0xaa0103f0u; // mov x16, x1 (entry pc)
    for (u32 r = 0; r <= 30; ++r) {
        if (r == 16 || r == 18) continue;
        // movz xr, #(0xdead), lsl #48; movk xr, #(0xbeef), lsl #32; movk xr, #r
        tc[tn++] = 0xd2e00000u | (0xdeadu << 5) | r;
        tc[tn++] = 0xf2c00000u | (0xbeefu << 5) | r;
        tc[tn++] = 0xf2800000u | (r << 5) | r;
    }
    tc[tn++] = 0xd61f0200u; // br x16
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(tramp_mem, tramp_sz);

    g_env.jit_base = static_cast<u8*>(mem);
    g_env.jit_sz = map_sz;
    entry_off = e_entry - min_va;
    return true;
}

static int run_guest(void* tramp_mem, u64 entry_off) {
    std::memset(g_env.stack_base, 0xaa, g_env.stack_sz);
    const u64 guest_sp = reinterpret_cast<u64>(g_env.stack_base) + (g_env.stack_sz / 2);
    const u64 entry_pc = reinterpret_cast<u64>(g_env.jit_base) + entry_off;
    g_env.exited = false;
    g_env.exit_code = 0;
    g_env.bad_syscall = 0;

    const int jrc = sigsetjmp(g_jb, 1);
    if (jrc == 0) {
        auto fn = reinterpret_cast<void (*)(u64, u64)>(tramp_mem);
        fn(guest_sp, entry_pc);
        std::fprintf(stderr, "aarch64_jit_exec: guest returned without exit syscall\n");
        return 126;
    }
    if (jrc == 1 && g_env.exited) return g_env.exit_code;
    if (jrc == 2) {
        std::fprintf(stderr, "aarch64_jit_exec: unsupported Linux syscall %llu\n",
                     static_cast<unsigned long long>(g_env.bad_syscall));
        return 125;
    }
    std::fprintf(stderr, "aarch64_jit_exec: unexpected trap/signal\n");
    return 124;
}

}  // namespace

int main(int argc, char** argv) {
    bool batch = false;
    const char* elf_path = nullptr;
    std::string dir;
    long count = 0;
    if (argc == 2) {
        elf_path = argv[1];
    } else if (argc == 5 && std::string(argv[1]) == "--batch") {
        batch = true;
        elf_path = argv[2];
        dir = argv[3];
        count = std::atol(argv[4]);
    } else {
        std::fprintf(stderr, "usage: aarch64_jit_exec <elf64> | --batch <elf64> <dir> <count>\n");
        return 2;
    }

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

    void* stk = mmap(nullptr, GUEST_STACK_SZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (stk == MAP_FAILED) {
        std::perror("mmap stack");
        return 2;
    }
    g_env.stack_base = static_cast<u8*>(stk);
    g_env.stack_sz = GUEST_STACK_SZ;

    u64 entry_off = 0;
    void* tramp_mem = nullptr;
    size_t tramp_sz = 0;
    if (!load_elf(elf_path, entry_off, tramp_mem, tramp_sz)) return 2;

    if (!batch) {
        g_env.batch_mode = false;
        return run_guest(tramp_mem, entry_off);
    }

    g_env.batch_mode = true;
    long bad = 0, acc = 0, rej = 0;
    for (long i = 0; i < count; ++i) {
        const std::string base = dir + "/" + std::to_string(i);
        std::ifstream f_in(base + ".in", std::ios::binary);
        std::ifstream f_out(base + ".out", std::ios::binary);
        std::ifstream f_rc(base + ".rc");
        if (!f_in || !f_out || !f_rc) {
            std::fprintf(stderr, "aarch64_jit_exec: missing fuzz file %ld\n", i);
            return 2;
        }
        std::string in_s((std::istreambuf_iterator<char>(f_in)), std::istreambuf_iterator<char>());
        std::string want_out((std::istreambuf_iterator<char>(f_out)), std::istreambuf_iterator<char>());
        int want_rc = -1;
        f_rc >> want_rc;

        std::string got_out;
        g_env.batch_in = &in_s;
        g_env.batch_in_pos = 0;
        g_env.batch_out = &got_out;
        const int rc = run_guest(tramp_mem, entry_off);
        if (rc != want_rc || got_out != want_out) {
            if (bad < 5) {
                std::fprintf(stderr, "FAIL fuzz case %ld: rc=%d want %d, out=%zu want %zu\n",
                             i, rc, want_rc, got_out.size(), want_out.size());
            }
            ++bad;
        }
        if (want_rc == 0) ++acc;
        else ++rej;
    }
    if (bad == 0) {
        std::printf("ok   aarch64_jit_exec differential fuzz: %ld cases (%ld accepted, %ld refused)\n",
                    count, acc, rej);
        return 0;
    }
    std::fprintf(stderr, "FAIL aarch64_jit_exec differential fuzz: %ld of %ld differ\n", bad, count);
    return 1;
}
