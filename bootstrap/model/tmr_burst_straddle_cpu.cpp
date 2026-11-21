// tmr_burst_straddle_cpu: x86-64 hardware verification engine for Pillar #4:
//   1. Page-Straddle #PF vs #UD vs Retirement Differential (>= 50,000 cases +
//      exhaustive k = 1..15 boundary sweep at PAGE0_END - k right against a
//      PROT_NONE guard page at 0x100000000).
//   2. 64B Adversarial Cache-Line Burst & Decision-50 Arena Injection on the
//      canonical 415-byte x86-64 exposure-tracking machine-code kernel
//      (memExposureKernelCode at 0x400000) across 3 page-separated 4096B
//      arenas (0x100000, 0x200000, 0x300000) surrounded by PROT_NONE guard
//      pages.

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <ucontext.h>
#include <unistd.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

static constexpr u64 DATA_PAGE = 0x11000ULL;
static constexpr u64 WIN_LO = 0x117e0ULL;
static constexpr u64 SP_INT = 0x11800ULL;
static constexpr u64 WIN_SIZE = 64;
static constexpr u64 PAGE0 = 0xfffff000ULL;
static constexpr u64 PAGE0_END = 0x100000000ULL;
static constexpr u64 PAGE1 = 0x100000000ULL;
static constexpr u32 POOL_SIZE = 32;

static constexpr u64 ARENA0 = 0x100000ULL;
static constexpr u64 ARENA1 = 0x200000ULL;
static constexpr u64 ARENA2 = 0x300000ULL;
static constexpr u64 PACKED_ARENA1 = 0x100020ULL;
static constexpr u64 ARENA_SIZE = 4096ULL;
static constexpr u64 CACHE_LINE_B = 64ULL;

static constexpr u64 KERNEL_BASE = 0x400000ULL;
static constexpr u64 KERNEL_LEN = 415ULL;
static constexpr u64 KERNEL_PARK = KERNEL_BASE + 0x185ULL;
static constexpr u64 KERNEL_OVERFLOW = KERNEL_BASE + 0x187ULL;
static constexpr u64 KERNEL_OVF_SYS = KERNEL_BASE + 0x191ULL;
static constexpr u64 KERNEL_REFUSE = KERNEL_BASE + 0x193ULL;
static constexpr u64 KERNEL_REF_SYS = KERNEL_BASE + 0x19dULL;

static const int GREG_IDX[16] = {
    REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
    REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};

// Canonical 415-byte Decision-50 machine-code kernel (TMRMemExposure.memExposureKernelCode).
static const u8 MEM_EXPOSURE_KERNEL_CODE[KERNEL_LEN] = {
    // Stage 0: slot bounds check and arena/slot pointer setup (0x000..0x038)
    0x48, 0x81, 0xff, 0xff, 0x00, 0x00, 0x00,  // 0x000: cmp rdi, 255
    0x0f, 0x83, 0x7a, 0x01, 0x00, 0x00,        // 0x007: jae 0x400187
    0x41, 0xb8, 0x00, 0x00, 0x10, 0x00,        // 0x00d: mov r8d, 0x100000
    0x41, 0xb9, 0x00, 0x00, 0x20, 0x00,        // 0x013: mov r9d, 0x200000
    0x41, 0xba, 0x00, 0x00, 0x30, 0x00,        // 0x019: mov r10d, 0x300000
    0x49, 0x89, 0xfb,                          // 0x01f: mov r11, rdi
    0x49, 0xc1, 0xe3, 0x04,                    // 0x022: shl r11, 4
    0x4d, 0x89, 0xc6,                          // 0x026: mov r14, r8
    0x4d, 0x01, 0xde,                          // 0x029: add r14, r11
    0x4d, 0x89, 0xcf,                          // 0x02c: mov r15, r9
    0x4d, 0x01, 0xdf,                          // 0x02f: add r15, r11
    0x4c, 0x89, 0xd2,                          // 0x032: mov rdx, r10
    0x4c, 0x01, 0xda,                          // 0x035: add rdx, r11

    // Stage 1: load, majority-vote, and repair val[w] (0x038..0x067)
    0x49, 0x8b, 0x46, 0x10,                    // 0x038: mov rax, [r14 + 16]
    0x49, 0x8b, 0x5f, 0x10,                    // 0x03c: mov rbx, [r15 + 16]
    0x48, 0x8b, 0x4a, 0x10,                    // 0x040: mov rcx, [rdx + 16]
    0x48, 0x39, 0xd8,                          // 0x044: cmp rax, rbx
    0x74, 0x12,                                // 0x047: je  0x40005b
    0x48, 0x39, 0xc8,                          // 0x049: cmp rax, rcx
    0x74, 0x0d,                                // 0x04c: je  0x40005b
    0x48, 0x39, 0xcb,                          // 0x04e: cmp rbx, rcx
    0x74, 0x05,                                // 0x051: je  0x400058
    0xe9, 0x3b, 0x01, 0x00, 0x00,              // 0x053: jmp 0x400193 (refuse)
    0x48, 0x89, 0xd8,                          // 0x058: mov rax, rbx
    0x49, 0x89, 0x46, 0x10,                    // 0x05b: mov [r14 + 16], rax
    0x49, 0x89, 0x47, 0x10,                    // 0x05f: mov [r15 + 16], rax
    0x48, 0x89, 0x42, 0x10,                    // 0x063: mov [rdx + 16], rax

    // Stage 2: load, majority-vote, and repair last_step[w] into r11 (0x067..0x099)
    0x49, 0x8b, 0x46, 0x18,                    // 0x067: mov rax, [r14 + 24]
    0x49, 0x8b, 0x5f, 0x18,                    // 0x06b: mov rbx, [r15 + 24]
    0x48, 0x8b, 0x4a, 0x18,                    // 0x06f: mov rcx, [rdx + 24]
    0x48, 0x39, 0xd8,                          // 0x073: cmp rax, rbx
    0x74, 0x12,                                // 0x076: je  0x40008a
    0x48, 0x39, 0xc8,                          // 0x078: cmp rax, rcx
    0x74, 0x0d,                                // 0x07b: je  0x40008a
    0x48, 0x39, 0xcb,                          // 0x07d: cmp rbx, rcx
    0x74, 0x05,                                // 0x080: je  0x400087
    0xe9, 0x0c, 0x01, 0x00, 0x00,              // 0x082: jmp 0x400193 (refuse)
    0x48, 0x89, 0xd8,                          // 0x087: mov rax, rbx
    0x49, 0x89, 0x46, 0x18,                    // 0x08a: mov [r14 + 24], rax
    0x49, 0x89, 0x47, 0x18,                    // 0x08e: mov [r15 + 24], rax
    0x48, 0x89, 0x42, 0x18,                    // 0x092: mov [rdx + 24], rax
    0x49, 0x89, 0xc3,                          // 0x096: mov r11, rax

    // Stage 3: load, majority-vote, and repair step into r12 (0x099..0x0c5)
    0x49, 0x8b, 0x00,                          // 0x099: mov rax, [r8]
    0x49, 0x8b, 0x19,                          // 0x09c: mov rbx, [r9]
    0x49, 0x8b, 0x0a,                          // 0x09f: mov rcx, [r10]
    0x48, 0x39, 0xd8,                          // 0x0a2: cmp rax, rbx
    0x74, 0x12,                                // 0x0a5: je  0x4000b9
    0x48, 0x39, 0xc8,                          // 0x0a7: cmp rax, rcx
    0x74, 0x0d,                                // 0x0aa: je  0x4000b9
    0x48, 0x39, 0xcb,                          // 0x0ac: cmp rbx, rcx
    0x74, 0x05,                                // 0x0af: je  0x4000b6
    0xe9, 0xdd, 0x00, 0x00, 0x00,              // 0x0b1: jmp 0x400193 (refuse)
    0x48, 0x89, 0xd8,                          // 0x0b6: mov rax, rbx
    0x49, 0x89, 0x00,                          // 0x0b9: mov [r8], rax
    0x49, 0x89, 0x01,                          // 0x0bc: mov [r9], rax
    0x49, 0x89, 0x02,                          // 0x0bf: mov [r10], rax
    0x49, 0x89, 0xc4,                          // 0x0c2: mov r12, rax

    // Stage 4: load, majority-vote, and repair acc into r13 (0x0c5..0x0f7)
    0x49, 0x8b, 0x40, 0x08,                    // 0x0c5: mov rax, [r8 + 8]
    0x49, 0x8b, 0x59, 0x08,                    // 0x0c9: mov rbx, [r9 + 8]
    0x49, 0x8b, 0x4a, 0x08,                    // 0x0cd: mov rcx, [r10 + 8]
    0x48, 0x39, 0xd8,                          // 0x0d1: cmp rax, rbx
    0x74, 0x12,                                // 0x0d4: je  0x4000e8
    0x48, 0x39, 0xc8,                          // 0x0d6: cmp rax, rcx
    0x74, 0x0d,                                // 0x0d9: je  0x4000e8
    0x48, 0x39, 0xcb,                          // 0x0db: cmp rbx, rcx
    0x74, 0x05,                                // 0x0de: je  0x4000e5
    0xe9, 0xae, 0x00, 0x00, 0x00,              // 0x0e0: jmp 0x400193 (refuse)
    0x48, 0x89, 0xd8,                          // 0x0e5: mov rax, rbx
    0x49, 0x89, 0x40, 0x08,                    // 0x0e8: mov [r8 + 8], rax
    0x49, 0x89, 0x41, 0x08,                    // 0x0ec: mov [r9 + 8], rax
    0x49, 0x89, 0x42, 0x08,                    // 0x0f0: mov [r10 + 8], rax
    0x49, 0x89, 0xc5,                          // 0x0f4: mov r13, rax

    // Stage 5: delta := step - last[w] + 1, step_new := step + 1, overflow checks (0x0f7..0x117)
    0x4c, 0x89, 0xe0,                          // 0x0f7: mov rax, r12
    0x4c, 0x29, 0xd8,                          // 0x0fa: sub rax, r11
    0x0f, 0x82, 0x84, 0x00, 0x00, 0x00,        // 0x0fd: jc  0x400187
    0x48, 0x83, 0xc0, 0x01,                    // 0x103: add rax, 1
    0x0f, 0x82, 0x7a, 0x00, 0x00, 0x00,        // 0x107: jc  0x400187
    0x49, 0x83, 0xc4, 0x01,                    // 0x10d: add r12, 1
    0x0f, 0x82, 0x70, 0x00, 0x00, 0x00,        // 0x111: jc  0x400187

    // Stage 6: shift-and-add loop computing rbx := delta^2 (0x117..0x143)
    0x48, 0x31, 0xdb,                          // 0x117: xor rbx, rbx
    0x48, 0x89, 0xc1,                          // 0x11a: mov rcx, rax
    0x48, 0x89, 0xc5,                          // 0x11d: mov rbp, rax   [sq_loop]
    0x48, 0x83, 0xe5, 0x01,                    // 0x120: and rbp, 1
    0x74, 0x09,                                // 0x124: jz  0x40012f
    0x48, 0x01, 0xcb,                          // 0x126: add rbx, rcx
    0x0f, 0x82, 0x58, 0x00, 0x00, 0x00,        // 0x129: jc  0x400187
    0x48, 0xc1, 0xe8, 0x01,                    // 0x12f: shr rax, 1     [sq_skip_add]
    0x48, 0x85, 0xc0,                          // 0x133: test rax, rax
    0x74, 0x0b,                                // 0x136: jz  0x400143
    0x48, 0x01, 0xc9,                          // 0x138: add rcx, rcx
    0x0f, 0x82, 0x46, 0x00, 0x00, 0x00,        // 0x13b: jc  0x400187
    0xeb, 0xda,                                // 0x141: jmp 0x40011d

    // Stage 7: acc += delta^2, commit to all 3 arenas, normalize scratch, park (0x143..0x187)
    0x49, 0x01, 0xdd,                          // 0x143: add r13, rbx   [sq_done]
    0x0f, 0x82, 0x3b, 0x00, 0x00, 0x00,        // 0x146: jc  0x400187
    0x4d, 0x89, 0x20,                          // 0x14c: mov [r8], r12
    0x4d, 0x89, 0x68, 0x08,                    // 0x14f: mov [r8 + 8], r13
    0x49, 0x89, 0x76, 0x10,                    // 0x153: mov [r14 + 16], rsi
    0x4d, 0x89, 0x66, 0x18,                    // 0x157: mov [r14 + 24], r12
    0x4d, 0x89, 0x21,                          // 0x15b: mov [r9], r12
    0x4d, 0x89, 0x69, 0x08,                    // 0x15e: mov [r9 + 8], r13
    0x49, 0x89, 0x77, 0x10,                    // 0x162: mov [r15 + 16], rsi
    0x4d, 0x89, 0x67, 0x18,                    // 0x166: mov [r15 + 24], r12
    0x4d, 0x89, 0x22,                          // 0x16a: mov [r10], r12
    0x4d, 0x89, 0x6a, 0x08,                    // 0x16d: mov [r10 + 8], r13
    0x48, 0x89, 0x72, 0x10,                    // 0x171: mov [rdx + 16], rsi
    0x4c, 0x89, 0x62, 0x18,                    // 0x175: mov [rdx + 24], r12
    0x4c, 0x89, 0xe8,                          // 0x179: mov rax, r13
    0x48, 0x31, 0xdb,                          // 0x17c: xor rbx, rbx
    0x48, 0x31, 0xc9,                          // 0x17f: xor rcx, rcx
    0x48, 0x31, 0xed,                          // 0x182: xor rbp, rbp
    0xeb, 0xfe,                                // 0x185: jmp 0x400185   [park]

    // Overflow / invalid slot exit(2) (0x187..0x193)
    0xbf, 0x02, 0x00, 0x00, 0x00,              // 0x187: mov edi, 2
    0xb8, 0x3c, 0x00, 0x00, 0x00,              // 0x18c: mov eax, 60
    0x0f, 0x05,                                // 0x191: syscall

    // 3-way split exit(3) (0x193..0x19f)
    0xbf, 0x03, 0x00, 0x00, 0x00,              // 0x193: mov edi, 3
    0xb8, 0x3c, 0x00, 0x00, 0x00,              // 0x198: mov eax, 60
    0x0f, 0x05                                 // 0x19d: syscall
};

static u64 xs_step(u64 x) {
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return x;
}

static u64 rng_state = 0x9e3779b97f4a7c15ULL;

static u64 rnd64() {
  rng_state = xs_step(rng_state);
  return rng_state;
}

static bool bernoulli(u64 num, u64 den) {
  if (num == 0 || den == 0) return false;
  return (rnd64() % den) < num;
}

static u64 gen_safe_word_lo(u64 x) {
  return (x & 0x0000ffffffffffffULL) | 0x5a3c000000000000ULL;
}

static void write_le64(u8* dst, u64 v) {
  for (int i = 0; i < 8; ++i) dst[i] = (u8)(v >> (8 * i));
}

static u64 read_le64(const u8* src) {
  u64 v = 0;
  for (int i = 0; i < 8; ++i) v |= ((u64)src[i]) << (8 * i);
  return v;
}

static int hex_val(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static u64 parse_hex64(const char* s) {
  u64 v = 0;
  while (*s) {
    int d = hex_val(*s++);
    if (d < 0) break;
    v = (v << 4) | (u64)d;
  }
  return v;
}

// ============================================================================
// Part 1: Page-Straddle #PF vs #UD vs Retirement Differential Engine
// ============================================================================

struct BaseState {
  u64 regs[16];
  bool cf;
  bool sf;
  bool zf;
  bool of;
  u8 mem[WIN_SIZE];
};

static BaseState pool[POOL_SIZE];

static void init_pool(u64 seed) {
  for (u32 idx = 0; idx < POOL_SIZE; ++idx) {
    u64 st = xs_step(0x9e3779b97f4a7c15ULL ^ seed ^ (u64)(idx * 0x10001u + 1u));
    for (u32 r = 0; r < 16; ++r) {
      st = xs_step(st);
      u64 v0 = st;
      u64 v = 0;
      if (r == 4) {
        v = SP_INT;
      } else if (r < 4) {
        v = (v0 & 0xffffffffffff00ffULL) | ((u64)(0xa1u + r * 0x11u) << 8);
      } else if (r < 8) {
        v = (v0 & 0xffffffffffffff00ULL) | (u64)(0x10u + r);
      } else {
        v = v0;
      }
      pool[idx].regs[r] = v;
    }
    st = xs_step(st);
    pool[idx].cf = (st & 1) != 0;
    pool[idx].sf = ((st >> 1) & 1) != 0;
    pool[idx].zf = ((st >> 2) & 1) != 0;
    pool[idx].of = ((st >> 3) & 1) != 0;
    for (u32 j = 0; j < WIN_SIZE; ++j) {
      st = xs_step(st);
      pool[idx].mem[j] = (u8)(st & 0xff);
    }
  }
}

enum StraddleKind : u32 {
  KIND_OK = 0,
  KIND_PF = 1,
  KIND_UD = 2
};

struct StraddleCase {
  u32 k;
  u8 bytes[16];
  u32 kind;
  u32 ins_len;
  u32 pool_idx;
  u32 fix_reg_idx;
  u64 fix_reg_val;
  u32 fix_mem_off;
  u64 fix_mem_val;
  u32 is_branch;
  u64 exp_delta;
  u32 mod1_idx;
  u64 mod1_val;
  u32 mod2_idx;
  u64 mod2_val;
  char exp_fl[5];
  u32 mod_mem_off;
  u32 mod_mem_len;
  u64 mod_mem_val;
};

static constexpr u32 BATCH_MAX = 4096;
static StraddleCase batch[BATCH_MAX];
static volatile u32 batch_len = 0;
static volatile u32 batch_pos = 0;
static volatile bool batch_active = false;
static volatile bool suppress_diff_print = false;
static ucontext_t host_ctx;

static volatile u64 straddle_executed = 0;
static volatile u64 straddle_ok_cnt = 0;
static volatile u64 straddle_pf_cnt = 0;
static volatile u64 straddle_ud_cnt = 0;
static volatile u64 straddle_disagreements = 0;

static u64 cur_in_regs[16];
static u8 cur_in_mem[WIN_SIZE];

static void setup_straddle_case(u32 idx, ucontext_t* uc) {
  const StraddleCase& cs = batch[idx];
  const BaseState& bs = pool[cs.pool_idx % POOL_SIZE];
  for (int r = 0; r < 16; ++r) cur_in_regs[r] = bs.regs[r];
  if (cs.fix_reg_idx < 16) cur_in_regs[cs.fix_reg_idx] = cs.fix_reg_val;

  memcpy(cur_in_mem, bs.mem, WIN_SIZE);
  if (cs.fix_mem_off <= 56) write_le64(&cur_in_mem[cs.fix_mem_off], cs.fix_mem_val);
  memcpy((void*)WIN_LO, cur_in_mem, WIN_SIZE);

  // Fill last 32 bytes of PAGE0 with 0x06 (#UD in 64-bit mode) and place
  // ONLY the `cs.k` bytes at `[PAGE0_END - cs.k, PAGE0_END)`.
  memset((void*)(PAGE0_END - 32), 0x06, 32);
  u64 ins_addr = PAGE0_END - cs.k;
  memcpy((void*)ins_addr, cs.bytes, cs.k);

  for (int r = 0; r < 16; ++r) {
    uc->uc_mcontext.gregs[GREG_IDX[r]] = (greg_t)cur_in_regs[r];
  }
  u64 efl = 0x202ULL | 0x100ULL; // IF=1, TF=1
  if (bs.cf) efl |= (1ULL << 0);
  if (bs.zf) efl |= (1ULL << 6);
  if (bs.sf) efl |= (1ULL << 7);
  if (bs.of) efl |= (1ULL << 11);
  uc->uc_mcontext.gregs[REG_EFL] = (greg_t)efl;
  uc->uc_mcontext.gregs[REG_RIP] = (greg_t)ins_addr;
}

static void check_straddle_result(u32 idx, int sig, siginfo_t* si, ucontext_t* uc) {
  const StraddleCase& cs = batch[idx];
  u64 ins_addr = PAGE0_END - cs.k;
  straddle_executed = straddle_executed + 1;

  const u8* cpu_mem = (const u8*)WIN_LO;
  u64 next_rip = (u64)uc->uc_mcontext.gregs[REG_RIP];
  u64 fault_addr = si ? (u64)si->si_addr : 0ULL;

  if (cs.kind == KIND_PF) {
    bool ok = (sig == SIGSEGV) && (fault_addr == PAGE0_END) && (next_rip == ins_addr);
    for (int r = 0; r < 16; ++r) {
      if ((u64)uc->uc_mcontext.gregs[GREG_IDX[r]] != cur_in_regs[r]) ok = false;
    }
    if (memcmp(cpu_mem, cur_in_mem, WIN_SIZE) != 0) ok = false;
    if (ok) {
      straddle_pf_cnt = straddle_pf_cnt + 1;
    } else {
      straddle_disagreements = straddle_disagreements + 1;
      if (!suppress_diff_print && straddle_disagreements <= 20) {
        printf("DIFF k=%u kind=PF sig=%d rip=%016llx addr=%016llx\n",
               cs.k, sig, (unsigned long long)next_rip, (unsigned long long)fault_addr);
      }
    }
    return;
  }

  if (cs.kind == KIND_UD) {
    bool ok = (sig == SIGILL) && (next_rip == ins_addr);
    for (int r = 0; r < 16; ++r) {
      if ((u64)uc->uc_mcontext.gregs[GREG_IDX[r]] != cur_in_regs[r]) ok = false;
    }
    if (memcmp(cpu_mem, cur_in_mem, WIN_SIZE) != 0) ok = false;
    if (ok) {
      straddle_ud_cnt = straddle_ud_cnt + 1;
    } else {
      straddle_disagreements = straddle_disagreements + 1;
      if (!suppress_diff_print && straddle_disagreements <= 20) {
        printf("DIFF k=%u kind=UD sig=%d rip=%016llx addr=%016llx\n",
               cs.k, sig, (unsigned long long)next_rip, (unsigned long long)fault_addr);
      }
    }
    return;
  }

  // KIND_OK: expected clean retirement via SIGTRAP
  u64 exp_regs[16];
  for (int r = 0; r < 16; ++r) exp_regs[r] = cur_in_regs[r];
  if (cs.mod1_idx < 16) exp_regs[cs.mod1_idx] = cs.mod1_val;
  if (cs.mod2_idx < 16) exp_regs[cs.mod2_idx] = cs.mod2_val;

  u8 exp_mem[WIN_SIZE];
  memcpy(exp_mem, cur_in_mem, WIN_SIZE);
  if (cs.mod_mem_len == 1 && cs.mod_mem_off < WIN_SIZE) {
    exp_mem[cs.mod_mem_off] = (u8)(cs.mod_mem_val & 0xff);
  } else if (cs.mod_mem_len == 8 && cs.mod_mem_off <= WIN_SIZE - 8) {
    write_le64(&exp_mem[cs.mod_mem_off], cs.mod_mem_val);
  }

  if (sig != SIGTRAP) {
    straddle_disagreements = straddle_disagreements + 1;
    if (!suppress_diff_print && straddle_disagreements <= 20) {
      printf("DIFF k=%u kind=OK sig=%d rip=%016llx addr=%016llx\n",
             cs.k, sig, (unsigned long long)next_rip, (unsigned long long)fault_addr);
    }
    return;
  }

  u64 cpu_delta = next_rip - ins_addr;
  bool len_bad = (!cs.is_branch && next_rip != ins_addr + cs.ins_len);
  bool bad = len_bad || (cpu_delta != cs.exp_delta);
  for (int r = 0; r < 16; ++r) {
    u64 rv = (u64)uc->uc_mcontext.gregs[GREG_IDX[r]];
    if (rv != exp_regs[r]) bad = true;
  }

  u64 efl = (u64)uc->uc_mcontext.gregs[REG_EFL];
  char cpu_fl[5];
  cpu_fl[0] = (efl & (1ULL << 0)) ? '1' : '0';
  cpu_fl[1] = (efl & (1ULL << 7)) ? '1' : '0';
  cpu_fl[2] = (efl & (1ULL << 6)) ? '1' : '0';
  cpu_fl[3] = (efl & (1ULL << 11)) ? '1' : '0';
  cpu_fl[4] = '\0';
  for (int i = 0; i < 4; ++i) {
    if (cs.exp_fl[i] != '-' && cs.exp_fl[i] != cpu_fl[i]) bad = true;
  }
  if (memcmp(cpu_mem, exp_mem, WIN_SIZE) != 0) bad = true;

  if (!bad) {
    straddle_ok_cnt = straddle_ok_cnt + 1;
  } else {
    straddle_disagreements = straddle_disagreements + 1;
    if (!suppress_diff_print && straddle_disagreements <= 20) {
      printf("DIFF k=%u kind=OK delta=%016llx exp=%016llx\n",
             cs.k, (unsigned long long)cpu_delta, (unsigned long long)cs.exp_delta);
    }
  }
}

static void straddle_sig_handler(int sig, siginfo_t* si, void* uctx) {
  ucontext_t* uc = (ucontext_t*)uctx;
  if (!batch_active) {
    batch_active = true;
    host_ctx = *uc;
    host_ctx.uc_mcontext.gregs[REG_EFL] &= ~0x100ULL;
    batch_pos = 0;
    setup_straddle_case(0, uc);
    return;
  }

  u32 finished_idx = batch_pos;
  check_straddle_result(finished_idx, sig, si, uc);
  u32 next_idx = finished_idx + 1;
  batch_pos = next_idx;
  if (next_idx < batch_len) {
    setup_straddle_case(next_idx, uc);
  } else {
    *uc = host_ctx;
  }
}

[[gnu::noinline]] static void run_straddle_batch() {
  if (batch_len == 0) return;
  batch_active = false;
  batch_pos = 0;
  asm volatile("int3" : : : "memory", "cc");
}

static bool parse_straddle_line(const char* line, StraddleCase& cs) {
  char cand_hex[64];
  char fix_reg_hex[32], fix_mem_hex[32];
  char delta_hex[32], mod1_hex[32], mod2_hex[32], mod_mem_hex[32];
  int matched = sscanf(
      line, "%u %63s %u %u %u %u %31s %u %31s %u | %31s %u %31s %u %31s %4s %u %u %31s",
      &cs.k, cand_hex, &cs.kind, &cs.ins_len, &cs.pool_idx, &cs.fix_reg_idx,
      fix_reg_hex, &cs.fix_mem_off, fix_mem_hex, &cs.is_branch, delta_hex,
      &cs.mod1_idx, mod1_hex, &cs.mod2_idx, mod2_hex, cs.exp_fl,
      &cs.mod_mem_off, &cs.mod_mem_len, mod_mem_hex);
  if (matched != 19 || cs.k == 0 || cs.k > 15) return false;
  for (u32 i = 0; i < cs.k; ++i) {
    int hi = hex_val(cand_hex[2 * i]);
    int lo = hex_val(cand_hex[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    cs.bytes[i] = (u8)((hi << 4) | lo);
  }
  cs.fix_reg_val = parse_hex64(fix_reg_hex);
  cs.fix_mem_val = parse_hex64(fix_mem_hex);
  cs.exp_delta = parse_hex64(delta_hex);
  cs.mod1_val = parse_hex64(mod1_hex);
  cs.mod2_val = parse_hex64(mod2_hex);
  cs.mod_mem_val = parse_hex64(mod_mem_hex);
  return true;
}

// Built-in negative control for Page-Straddle:
//   1) Place a 10-byte `mov rax, imm64` (`48 b8 ...`) at `PAGE0_END - 9` (`k = 9`)
//      with `kind = KIND_OK` (simulating an untruncated byte stream model that
//      ignores the page boundary at `k = 9 < 10`); verify the CPU raises SIGSEGV
//      (`si_addr == PAGE0_END`) and triggers a disagreement.
//   2) Place `[0x48, 0x06]` (`REX.W + #UD`) at `PAGE0_END - 1` (`k = 1`, only `0x48`
//      before `PAGE0_END`) with `kind = KIND_UD`; verify the CPU raises SIGSEGV
//      (`#PF` at `PAGE0_END`), catching the `#PF` vs `#UD` swap.
//   3) Place `[0x48, 0x06]` at `PAGE0_END - 2` (`k = 2`) with `kind = KIND_PF`;
//      verify the CPU raises SIGILL (`#UD`), catching the `#UD` vs `#PF` swap.
static bool run_straddle_negative_control() {
  u64 saved_exec = straddle_executed;
  u64 saved_ok = straddle_ok_cnt;
  u64 saved_pf = straddle_pf_cnt;
  u64 saved_ud = straddle_ud_cnt;
  u64 saved_diffs = straddle_disagreements;

  // Neg 1: 10-byte mov rax, imm64 at k = 9 expecting KIND_OK
  StraddleCase c1{};
  c1.k = 9;
  const u8 mov10[10] = {0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8};
  memcpy(c1.bytes, mov10, 9);
  c1.kind = KIND_OK;
  c1.ins_len = 10;
  c1.pool_idx = 0;
  c1.fix_reg_idx = 16;
  c1.fix_mem_off = 64;
  c1.exp_delta = 10;
  c1.mod1_idx = 16;
  c1.mod2_idx = 16;
  strcpy(c1.exp_fl, "----");
  c1.mod_mem_off = 64;

  // Neg 2: [0x48, 0x06] at k = 1 expecting KIND_UD (actually #PF)
  StraddleCase c2 = c1;
  c2.k = 1;
  c2.bytes[0] = 0x48;
  c2.kind = KIND_UD;
  c2.ins_len = 2;

  // Neg 3: [0x48, 0x06] at k = 2 expecting KIND_PF (actually #UD)
  StraddleCase c3 = c1;
  c3.k = 2;
  c3.bytes[0] = 0x48;
  c3.bytes[1] = 0x06;
  c3.kind = KIND_PF;
  c3.ins_len = 2;

  batch[0] = c1;
  batch[1] = c2;
  batch[2] = c3;
  batch_len = 3;
  suppress_diff_print = true;
  straddle_disagreements = 0;
  run_straddle_batch();
  u64 new_diffs = straddle_disagreements;
  suppress_diff_print = false;
  batch_len = 0;

  straddle_executed = saved_exec;
  straddle_ok_cnt = saved_ok;
  straddle_pf_cnt = saved_pf;
  straddle_ud_cnt = saved_ud;
  straddle_disagreements = saved_diffs;
  return (new_diffs == 3);
}

// ============================================================================
// Part 2: 64B Adversarial Cache-Line Burst & Decision-50 Arena Injection
// ============================================================================

static void set_arena_state(u8* arena, u64 w, u64 step, u64 acc, u64 val, u64 last_step) {
  write_le64(arena + 0, step);
  write_le64(arena + 8, acc);
  write_le64(arena + 16 + 16 * w, val);
  write_le64(arena + 24 + 16 * w, last_step);
}

static void init_three_arenas(u64 w, u64 step, u64 acc, u64 val, u64 last_step) {
  memset((void*)ARENA0, 0, 64);
  memset((void*)ARENA1, 0, 64);
  memset((void*)ARENA2, 0, 64);
  set_arena_state((u8*)ARENA0, w, step, acc, val, last_step);
  set_arena_state((u8*)ARENA1, w, step, acc, val, last_step);
  set_arena_state((u8*)ARENA2, w, step, acc, val, last_step);
}

// Verify in fork() children that touching any of the 6 PROT_NONE guard pages
// around ARENA0 (0x100000), ARENA1 (0x200000), and ARENA2 (0x300000) raises SIGSEGV.
static bool verify_arena_guard_pages() {
  const u64 guards[6] = {
      ARENA0 - 4096, ARENA0 + 4096,
      ARENA1 - 4096, ARENA1 + 4096,
      ARENA2 - 4096, ARENA2 + 4096};
  for (int i = 0; i < 6; ++i) {
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
      signal(SIGSEGV, SIG_DFL);
      signal(SIGILL, SIG_DFL);
      signal(SIGTRAP, SIG_DFL);
      volatile u8* p = (volatile u8*)guards[i];
      *p = 0xff;
      _exit(0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return false;
    if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGSEGV) return false;
  }
  return true;
}

// Verify in fork() children that executing the untouched 415-byte kernel at
// 0x400000 on 3-way split across val[w], last_step[w], step, or acc exits with
// code 3 via real `0f 05` syscall.
static bool verify_fork_syscall_exit3() {
  for (int field = 0; field < 4; ++field) {
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
      alarm(1);
      init_three_arenas(0, 10, 100, 0x1111, 5);
      u64 off = (field == 0) ? 16 : (field == 1) ? 24 : (field == 2) ? 0 : 8;
      write_le64((u8*)ARENA0 + off, 0xaaaaULL);
      write_le64((u8*)ARENA1 + off, 0xbbbbULL);
      write_le64((u8*)ARENA2 + off, 0xccccULL);
      u64 w = 0, v_new = 0x9999ULL, entry = KERNEL_BASE;
      asm volatile(
          "mov %[w], %%rdi\n"
          "mov %[v], %%rsi\n"
          "jmp *%[entry]\n"
          :
          : [w] "r"(w), [v] "r"(v_new), [entry] "r"(entry)
          : "rdi", "rsi");
      _exit(99);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return false;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 3) return false;
  }
  return true;
}

// Verify in fork() children that executing the untouched 415-byte kernel at
// 0x400000 on invalid slot (w = 255), step < last_step, or 64-bit delta^2 /
// acc + delta^2 overflow exits with code 2 via real `0f 05` syscall.
static bool verify_fork_syscall_exit2() {
  for (int mode = 0; mode < 3; ++mode) {
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
      alarm(1);
      u64 w = 0;
      if (mode == 0) {
        init_three_arenas(0, 10, 100, 0x1111, 5);
        w = 255; // invalid slot >= 255
      } else if (mode == 1) {
        init_three_arenas(0, 5, 100, 0x1111, 10); // step < last_step -> underflow
      } else {
        init_three_arenas(0, 0x200000000ULL, 100, 0x1111, 0); // delta^2 >= 2^64
      }
      u64 v_new = 0x9999ULL, entry = KERNEL_BASE;
      asm volatile(
          "mov %[w], %%rdi\n"
          "mov %[v], %%rsi\n"
          "jmp *%[entry]\n"
          :
          : [w] "r"(w), [v] "r"(v_new), [entry] "r"(entry)
          : "rdi", "rsi");
      _exit(99);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return false;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 2) return false;
  }
  return true;
}

enum BurstTrialOut {
  BURST_CLEAN_WIN = 0,
  BURST_CORRECTED = 1,
  BURST_REFUSAL = 2,
  BURST_STUCK = 3,
  BURST_SDC = 4
};

// Hardware single-stepping (RFLAGS.TF = 1) runner on the 100% unmodified 415-byte
// memExposureKernelCode at 0x400000 (stops when RIP reaches KERNEL_PARK 0x400185,
// or intercepts `0f 05` at 0x400191 / 0x40019d).
static volatile bool tf_active = false;
static volatile bool tf_done = false;
static volatile u32 tf_steps = 0;
static volatile u32 tf_exit_code = 0; // 0 = park, 2 = exit2, 3 = exit3, 99 = stuck
static volatile u64 tf_end_rax = 0;
static u64 tf_in_w = 0;
static u64 tf_in_vnew = 0;

static void tf_kernel_sig_handler(int sig, siginfo_t* si, void* uctx) {
  (void)si;
  ucontext_t* uc = (ucontext_t*)uctx;
  if (!tf_active) {
    tf_active = true;
    host_ctx = *uc;
    host_ctx.uc_mcontext.gregs[REG_EFL] &= ~0x100ULL;
    for (int r = 0; r < 16; ++r) {
      uc->uc_mcontext.gregs[GREG_IDX[r]] = 0;
    }
    uc->uc_mcontext.gregs[REG_RSP] = (greg_t)SP_INT;
    uc->uc_mcontext.gregs[REG_RDI] = (greg_t)tf_in_w;
    uc->uc_mcontext.gregs[REG_RSI] = (greg_t)tf_in_vnew;
    uc->uc_mcontext.gregs[REG_RIP] = (greg_t)KERNEL_BASE;
    uc->uc_mcontext.gregs[REG_EFL] = 0x202ULL | 0x100ULL;
    return;
  }

  if (sig != SIGTRAP) {
    tf_exit_code = 99;
    tf_done = true;
    *uc = host_ctx;
    return;
  }

  tf_steps = tf_steps + 1;
  u64 rip = (u64)uc->uc_mcontext.gregs[REG_RIP];
  if (rip == KERNEL_PARK) {
    tf_exit_code = 0;
    tf_end_rax = (u64)uc->uc_mcontext.gregs[REG_RAX];
    tf_done = true;
    *uc = host_ctx;
    return;
  }
  if (rip == KERNEL_OVF_SYS || rip == KERNEL_REF_SYS) {
    u64 nr = (u64)uc->uc_mcontext.gregs[REG_RAX];
    u64 code = (u64)uc->uc_mcontext.gregs[REG_RDI];
    tf_exit_code = (nr == 60) ? (u32)(code & 0xff) : 99;
    tf_done = true;
    *uc = host_ctx;
    return;
  }
  if (tf_steps > 450 || rip < KERNEL_BASE || rip >= KERNEL_BASE + KERNEL_LEN) {
    tf_exit_code = 99;
    tf_done = true;
    *uc = host_ctx;
    return;
  }
  uc->uc_mcontext.gregs[REG_EFL] |= 0x100ULL;
}

[[gnu::noinline]] static u32 run_kernel_unmodified_tf(u64 w, u64 v_new, u64* out_rax) {
  tf_in_w = w;
  tf_in_vnew = v_new;
  tf_active = false;
  tf_done = false;
  tf_steps = 0;
  tf_exit_code = 99;
  tf_end_rax = 0;
  asm volatile("int3" : : : "memory", "cc");
  *out_rax = tf_end_rax;
  return tf_exit_code;
}

// Fast native execution of the 415-byte kernel at 0x400000:
// Stages 0..7 (0x400000..0x400184) execute natively on the CPU without signal
// traps; 0x400185 (park, where rcx == 0 from 0x17f: xor rcx, rcx) returns via
// `ret` (0xc3), 0x40018c (overflow stub) sets ecx = 2 and returns via `ret`,
// and 0x400198 (refuse stub) sets ecx = 3 and returns via `ret`.
static void arm_fast_native_kernel_ret() {
  u8* k = (u8*)KERNEL_BASE;
  k[0x185] = 0xc3; // ret at park (rcx is 0 from 0x17f: xor rcx, rcx)
  // At 0x18c (after 0x187: mov edi, 2): mov ecx, 2; ret
  const u8 ovf_ret[6] = {0xb9, 0x02, 0x00, 0x00, 0x00, 0xc3};
  memcpy(k + 0x18c, ovf_ret, 6);
  // At 0x198 (after 0x193: mov edi, 3): mov ecx, 3; ret
  const u8 ref_ret[6] = {0xb9, 0x03, 0x00, 0x00, 0x00, 0xc3};
  memcpy(k + 0x198, ref_ret, 6);
}

[[gnu::noinline]] static u32 run_kernel_fast_native(u64 w, u64 v_new, u64* out_rax) {
  u64 rax_out = 0;
  u64 rcx_out = 0;
  asm volatile(
      "push %%rbp\n"
      "mov %[w], %%rdi\n"
      "mov %[v], %%rsi\n"
      "mov $0x400000, %%rax\n"
      "call *%%rax\n"
      "pop %%rbp\n"
      : "=a"(rax_out), "=c"(rcx_out)
      : [w] "r"(w), [v] "r"(v_new)
      : "rbx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12", "r13",
        "r14", "r15", "memory", "cc");
  *out_rax = rax_out;
  return (u32)rcx_out; // 0 = park, 2 = exit(2) overflow, 3 = exit(3) refuse
}

// Verify post-execution outcome across all 3 arenas (a0, a1, a2) and check
// whether an underestimated exposure certificate (`SDC_cert`) occurred.
static BurstTrialOut evaluate_burst_outcome(
    u32 exit_code, u64 out_rax, bool any_burst,
    const u8* a0, const u8* a1, const u8* a2,
    u64 w, u64 exp_step, u64 exp_acc, u64 exp_val,
    bool* out_sdc_cert) {
  *out_sdc_cert = false;
  if (exit_code == 3) {
    return BURST_REFUSAL;
  }
  if (exit_code == 2) {
    return BURST_STUCK;
  }
  if (exit_code != 0) {
    return BURST_STUCK;
  }

  const u8* arenas[3] = {a0, a1, a2};
  bool all_match = (out_rax == exp_acc);
  bool cert_under = (out_rax < exp_acc);

  for (int i = 0; i < 3; ++i) {
    u64 st = read_le64(arenas[i] + 0);
    u64 ac = read_le64(arenas[i] + 8);
    u64 vl = read_le64(arenas[i] + 16 + 16 * w);
    u64 ls = read_le64(arenas[i] + 24 + 16 * w);
    if (st != exp_step || ac != exp_acc || vl != exp_val || ls != exp_step) {
      all_match = false;
    }
    if (ac < exp_acc) {
      cert_under = true;
    }
  }

  if (all_match) {
    return any_burst ? BURST_CORRECTED : BURST_CLEAN_WIN;
  }
  *out_sdc_cert = cert_under;
  return BURST_SDC;
}

// Apply a 64-byte adversarial cache-line burst payload of kind `pat_kind` to
// `[dst, dst + 64)` (where `step_cur` is the current step for certificate attacks).
static void inject_64b_burst(u8* dst, int pat_kind, u64 step_cur) {
  switch (pat_kind % 6) {
    case 0:
      // All-0x00 64-byte cache-line wipe (zeros step, acc, val, last_step)
      memset(dst, 0x00, CACHE_LINE_B);
      break;
    case 1:
      // All-0xff 64-byte cache-line saturation
      memset(dst, 0xff, CACHE_LINE_B);
      break;
    case 2:
      // Targeted certificate-underestimation 64B payload:
      // Keeps step = step_cur, zeroes acc = 0, and sets last_step[0..2] = step_cur
      // so delta = 1 (trying to erase accumulated exposure and current delta^2).
      write_le64(dst + 0, step_cur);
      write_le64(dst + 8, 0ULL);
      for (u64 w = 0; w < 3; ++w) {
        write_le64(dst + 16 + 16 * w, 0xdead0000ULL + w);
        write_le64(dst + 24 + 16 * w, step_cur);
      }
      break;
    case 3:
      // Alternating 0xa5 / 0x5a 64-byte cache-line pattern
      for (u64 i = 0; i < CACHE_LINE_B; ++i) {
        dst[i] = (i & 1) ? 0x5a : 0xa5;
      }
      break;
    case 4:
      // 512-bit bitwise inversion across the entire 64-byte cache line
      for (u64 i = 0; i < CACHE_LINE_B; ++i) {
        dst[i] ^= 0xff;
      }
      break;
    default:
      // Random 512-bit (64-byte) adversarial payload
      for (u64 i = 0; i < CACHE_LINE_B; i += 8) {
        write_le64(dst + i, rnd64());
      }
      break;
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: tmr_burst_straddle_cpu <seed> <straddle_cases_file> [mc_trials] [--neg-straddle|--neg-packed]\n");
    return 2;
  }
  u64 seed = strtoull(argv[1], nullptr, 10);
  const char* cases_path = argv[2];
  u64 mc_trials = 20000;
  bool neg_straddle = false;
  bool neg_packed = false;
  for (int i = 3; i < argc; ++i) {
    if (strcmp(argv[i], "--neg-straddle") == 0) {
      neg_straddle = true;
    } else if (strcmp(argv[i], "--neg-packed") == 0) {
      neg_packed = true;
    } else {
      mc_trials = strtoull(argv[i], nullptr, 10);
    }
  }
  rng_state = 0x9e3779b97f4a7c15ULL ^ seed;

  // 1. Map scratch data page (0x11000) and Page-Straddle RX+PROT_NONE pages
  if (mmap((void*)DATA_PAGE, 4096, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != (void*)DATA_PAGE) {
    fprintf(stderr, "mmap DATA_PAGE failed: %s\n", strerror(errno));
    return 2;
  }
  if (mmap((void*)PAGE0, 8192, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != (void*)PAGE0) {
    fprintf(stderr, "mmap PAGE0+PAGE1 failed: %s\n", strerror(errno));
    return 2;
  }
  if (mprotect((void*)PAGE0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
    fprintf(stderr, "mprotect PAGE0 failed: %s\n", strerror(errno));
    return 2;
  }
  memset((void*)PAGE0, 0x06, 4096);

  // 2. Map 3 page-separated 4096B Decision-50 arenas surrounded by PROT_NONE guard pages
  const u64 arena_bases[3] = {ARENA0, ARENA1, ARENA2};
  for (int i = 0; i < 3; ++i) {
    u64 guard_lo = arena_bases[i] - 4096;
    if (mmap((void*)guard_lo, 3 * 4096, PROT_NONE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != (void*)guard_lo) {
      fprintf(stderr, "mmap arena %d + guards failed: %s\n", i, strerror(errno));
      return 2;
    }
    if (mprotect((void*)arena_bases[i], 4096, PROT_READ | PROT_WRITE) != 0) {
      fprintf(stderr, "mprotect arena %d failed: %s\n", i, strerror(errno));
      return 2;
    }
  }

  // 3. Map KERNEL_BASE (0x400000) RWX + guard page (0x401000) PROT_NONE
  if (mmap((void*)KERNEL_BASE, 8192, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != (void*)KERNEL_BASE) {
    fprintf(stderr, "mmap KERNEL_BASE failed: %s\n", strerror(errno));
    return 2;
  }
  if (mprotect((void*)KERNEL_BASE, 4096, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
    fprintf(stderr, "mprotect KERNEL_BASE failed: %s\n", strerror(errno));
    return 2;
  }
  memset((void*)KERNEL_BASE, 0x06, 4096);
  memcpy((void*)KERNEL_BASE, MEM_EXPOSURE_KERNEL_CODE, KERNEL_LEN);

  // Set up alternate signal stack
  stack_t ss{};
  ss.ss_sp = mmap(nullptr, 65536, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ss.ss_size = 65536;
  if (sigaltstack(&ss, nullptr) != 0) {
    fprintf(stderr, "sigaltstack failed\n");
    return 2;
  }

  // ==========================================================================
  // Execute Suite 1: Page-Straddle #PF vs #UD vs Retirement Differential
  // ==========================================================================
  init_pool(seed);

  struct sigaction sa{};
  sa.sa_sigaction = straddle_sig_handler;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGTRAP, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);

  FILE* fp = fopen(cases_path, "r");
  if (!fp) {
    fprintf(stderr, "cannot open %s\n", cases_path);
    return 2;
  }
  char line[512];
  batch_len = 0;
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '\0' || line[0] == '\n') continue;
    if (!parse_straddle_line(line, batch[batch_len])) {
      fprintf(stderr, "unparsed straddle case line: %s", line);
      fclose(fp);
      return 2;
    }
    if (neg_straddle && batch_len == 0 && batch[0].kind == KIND_PF) {
      // Negative control CLI flag: flip expected PF to OK
      batch[0].kind = KIND_OK;
    }
    batch_len = batch_len + 1;
    if (batch_len == BATCH_MAX) {
      run_straddle_batch();
      batch_len = 0;
    }
  }
  if (batch_len > 0) {
    run_straddle_batch();
    batch_len = 0;
  }
  fclose(fp);

  bool straddle_neg_ok = run_straddle_negative_control();
  printf("STRADDLE_SUMMARY executed=%llu ok=%llu pf=%llu ud=%llu disagreements=%llu\n",
         (unsigned long long)straddle_executed,
         (unsigned long long)straddle_ok_cnt,
         (unsigned long long)straddle_pf_cnt,
         (unsigned long long)straddle_ud_cnt,
         (unsigned long long)straddle_disagreements);
  printf("STRADDLE_NEG_CTRL %s\n", straddle_neg_ok ? "PASS" : "FAIL");
  fflush(stdout);

  // ==========================================================================
  // Execute Suite 2: 64B Adversarial Cache-Line Burst & Decision-50 Injection
  // ==========================================================================

  // 2.1 Emit exact 415-byte memExposureKernelCode hex from 0x400000
  printf("MEM_EXPOSURE_CODE ");
  for (u64 i = 0; i < KERNEL_LEN; ++i) {
    printf("%02x", ((const u8*)KERNEL_BASE)[i]);
  }
  printf("\n");

  // 2.2 Verify PROT_NONE guard pages around the 3 arenas and real 0f 05 syscalls
  bool guard_ok = verify_arena_guard_pages();
  printf("ARENA_GUARD_CHECK %s\n", guard_ok ? "PASS" : "FAIL");
  bool exit3_ok = verify_fork_syscall_exit3();
  printf("SYSCALL_EXIT3_SPLIT_CHECK %s\n", exit3_ok ? "PASS" : "FAIL");
  bool exit2_ok = verify_fork_syscall_exit2();
  printf("SYSCALL_EXIT2_OVERFLOW_CHECK %s\n", exit2_ok ? "PASS" : "FAIL");
  fflush(stdout);

  // 2.3 Hardware single-stepping (RFLAGS.TF = 1) on the 100% unmodified 415B kernel
  //     across 252 single 64-byte cache-line burst trials (3 arenas x 14 slots/offsets x 6 payload classes)
  sa.sa_sigaction = tf_kernel_sig_handler;
  sigaction(SIGTRAP, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);

  u64 tf_trials = 0, tf_corr = 0, tf_ref = 0, tf_stuck = 0, tf_sdc = 0, tf_sdc_cert = 0;
  for (int rep = 0; rep < 3; ++rep) {
    u8* target_arena = (u8*)arena_bases[rep];
    for (int pat = 0; pat < 6; ++pat) {
      for (u64 w = 0; w < 14; ++w) {
        u64 step0 = 20 + w * 3;
        u64 last0 = 5 + w;
        u64 acc0 = 500 + w * 100;
        u64 val0 = 0x12340000ULL + w;
        u64 v_new = 0xabcd0000ULL + w;
        u64 delta = step0 - last0 + 1;
        u64 exp_step = step0 + 1;
        u64 exp_acc = acc0 + delta * delta;
        init_three_arenas(w, step0, acc0, val0, last0);
        // Inject 64-byte cache-line burst into target_arena covering [0x00, 0x40) or slot w's cache line
        u64 cl_off = ((16 + 16 * w) / 64) * 64;
        inject_64b_burst(target_arena + cl_off, pat, step0);
        if (cl_off != 0) {
          inject_64b_burst(target_arena, pat, step0);
        }
        u64 out_rax = 0;
        u32 code = run_kernel_unmodified_tf(w, v_new, &out_rax);
        bool sdc_cert = false;
        BurstTrialOut res = evaluate_burst_outcome(
            code, out_rax, true, (const u8*)ARENA0, (const u8*)ARENA1, (const u8*)ARENA2,
            w, exp_step, exp_acc, v_new, &sdc_cert);
        tf_trials++;
        if (res == BURST_CORRECTED) tf_corr++;
        else if (res == BURST_REFUSAL) tf_ref++;
        else if (res == BURST_STUCK) tf_stuck++;
        else {
          tf_sdc++;
          if (sdc_cert) tf_sdc_cert++;
        }
      }
    }
  }
  printf("UNMODIFIED_TF_SWEEP trials=%llu corrected=%llu refusal=%llu stuck=%llu sdc=%llu sdc_cert=%llu\n",
         (unsigned long long)tf_trials, (unsigned long long)tf_corr,
         (unsigned long long)tf_ref, (unsigned long long)tf_stuck,
         (unsigned long long)tf_sdc, (unsigned long long)tf_sdc_cert);
  fflush(stdout);

  // 2.4 Exhaustive Single 64-Byte Cache-Line Burst Campaign (fast native execution):
  //     - 3 arenas x 512 single-bit flips across the 64-byte cache line [0x00, 0x40)
  //       (covering step, acc, val[0..2], last_step[0..2]) = 1,536 trials
  //     - 3 arenas x 6 adversarial 64B payload classes (all-0x00, all-0xff, cert-underestimation,
  //       alternating 0xa5/0x5a, 512b bitwise inversion, random 512b) x 64 cache-line offsets/slots
  //       = 1,152 trials
  //     - 3 arenas x 200 random 512-bit unaligned 64-byte bursts across [0, 4096 - 64] = 600 trials
  //     Total = 3,288 single 64-byte burst trials.
  arm_fast_native_kernel_ret();

  u64 sb_trials = 0, sb_clean = 0, sb_corr = 0, sb_ref = 0, sb_stuck = 0, sb_sdc = 0, sb_sdc_cert = 0;

  if (neg_packed) {
    // When --neg-packed is passed, patch mov r9d immediate at 0x400015 to PACKED_ARENA1 (0x100020)
    // so Arena 0 (0x100000) and Arena 1 (0x100020) share the same 64B cache line [0x100000, 0x100040)!
    u32 packed_addr = (u32)PACKED_ARENA1;
    memcpy((u8*)KERNEL_BASE + 0x015, &packed_addr, 4);
  }

  const u64 active_a1 = neg_packed ? PACKED_ARENA1 : ARENA1;
  const u64 active_arenas[3] = {ARENA0, active_a1, ARENA2};

  // (a) All 512 single-bit flips in the 64-byte cache line [0x00, 0x40) across all 3 arenas
  for (int rep = 0; rep < 3; ++rep) {
    for (u32 bit = 0; bit < 512; ++bit) {
      u64 w = (bit >= 128) ? ((bit - 128) / 128) : 0; // target the slot whose word was flipped
      u64 step0 = 50 + (bit % 17);
      u64 last0 = 10 + (bit % 7);
      u64 acc0 = 1000 + bit * 10;
      u64 val0 = 0x55aa0000ULL + bit;
      u64 v_new = 0x99880000ULL + bit;
      u64 delta = step0 - last0 + 1;
      u64 exp_step = step0 + 1;
      u64 exp_acc = acc0 + delta * delta;

      for (int a = 0; a < 3; ++a) {
        set_arena_state((u8*)active_arenas[a], w, step0, acc0, val0, last0);
      }
      if (neg_packed && rep == 0) {
        // In --neg-packed mode, a single 64B burst at 0x100000 hits both Arena 0 and Arena 1
        set_arena_state((u8*)ARENA0, 0, step0, 0ULL, 0xdeadULL, step0);
        set_arena_state((u8*)PACKED_ARENA1, 0, step0, 0ULL, 0xdeadULL, step0);
        w = 0;
        exp_acc = acc0 + (step0 - last0 + 1) * (step0 - last0 + 1);
      } else {
        ((u8*)active_arenas[rep])[bit / 8] ^= (u8)(1u << (bit % 8));
      }

      u64 out_rax = 0;
      u32 code = run_kernel_fast_native(w, v_new, &out_rax);
      bool sdc_cert = false;
      BurstTrialOut res = evaluate_burst_outcome(
          code, out_rax, true, (const u8*)ARENA0, (const u8*)active_a1, (const u8*)ARENA2,
          w, exp_step, exp_acc, v_new, &sdc_cert);
      sb_trials++;
      if (res == BURST_CLEAN_WIN) sb_clean++;
      else if (res == BURST_CORRECTED) sb_corr++;
      else if (res == BURST_REFUSAL) sb_ref++;
      else if (res == BURST_STUCK) sb_stuck++;
      else {
        sb_sdc++;
        if (sdc_cert) sb_sdc_cert++;
      }
    }
  }

  // (b) 3 arenas x 6 adversarial 64B payload classes x 64 slot/offset combinations
  for (int rep = 0; rep < 3; ++rep) {
    for (int pat = 0; pat < 6; ++pat) {
      for (u64 sidx = 0; sidx < 64; ++sidx) {
        u64 w = sidx % 3;
        u64 step0 = 100 + sidx;
        u64 last0 = 20 + (sidx % 15);
        u64 acc0 = 5000 + sidx * 50;
        u64 val0 = 0x77770000ULL + sidx;
        u64 v_new = 0xeeee0000ULL + sidx;
        u64 delta = step0 - last0 + 1;
        u64 exp_step = step0 + 1;
        u64 exp_acc = acc0 + delta * delta;

        for (int a = 0; a < 3; ++a) {
          set_arena_state((u8*)active_arenas[a], w, step0, acc0, val0, last0);
        }
        inject_64b_burst((u8*)active_arenas[rep], pat, step0);

        u64 out_rax = 0;
        u32 code = run_kernel_fast_native(w, v_new, &out_rax);
        bool sdc_cert = false;
        BurstTrialOut res = evaluate_burst_outcome(
            code, out_rax, true, (const u8*)ARENA0, (const u8*)active_a1, (const u8*)ARENA2,
            w, exp_step, exp_acc, v_new, &sdc_cert);
        sb_trials++;
        if (res == BURST_CLEAN_WIN) sb_clean++;
        else if (res == BURST_CORRECTED) sb_corr++;
        else if (res == BURST_REFUSAL) sb_ref++;
        else if (res == BURST_STUCK) sb_stuck++;
        else {
          sb_sdc++;
          if (sdc_cert) sb_sdc_cert++;
        }
      }
    }
  }

  // (c) 3 arenas x 200 random 512-bit unaligned 64-byte bursts across [0, 4096 - 64]
  for (int rep = 0; rep < 3; ++rep) {
    for (int j = 0; j < 200; ++j) {
      u64 w = (u64)(j % 16);
      u64 step0 = 300 + (u64)j;
      u64 last0 = 40 + (u64)(j % 25);
      u64 acc0 = 10000 + (u64)j * 100;
      u64 val0 = gen_safe_word_lo(rnd64());
      u64 v_new = gen_safe_word_lo(rnd64());
      u64 delta = step0 - last0 + 1;
      u64 exp_step = step0 + 1;
      u64 exp_acc = acc0 + delta * delta;

      for (int a = 0; a < 3; ++a) {
        set_arena_state((u8*)active_arenas[a], w, step0, acc0, val0, last0);
      }
      u64 base_off = (j & 1) ? (16 + 16 * w) : 8;
      u64 shift = rnd64() % 56;
      u64 off = (base_off > shift) ? (base_off - shift) : 0;
      if (off + CACHE_LINE_B > ARENA_SIZE) off = ARENA_SIZE - CACHE_LINE_B;
      inject_64b_burst((u8*)active_arenas[rep] + off, 5, step0);

      u64 out_rax = 0;
      u32 code = run_kernel_fast_native(w, v_new, &out_rax);
      bool sdc_cert = false;
      BurstTrialOut res = evaluate_burst_outcome(
          code, out_rax, true, (const u8*)ARENA0, (const u8*)active_a1, (const u8*)ARENA2,
          w, exp_step, exp_acc, v_new, &sdc_cert);
      sb_trials++;
      if (res == BURST_CLEAN_WIN) sb_clean++;
      else if (res == BURST_CORRECTED) sb_corr++;
      else if (res == BURST_REFUSAL) sb_ref++;
      else if (res == BURST_STUCK) sb_stuck++;
      else {
        sb_sdc++;
        if (sdc_cert) sb_sdc_cert++;
      }
    }
  }

  printf("SINGLE_BURST64_SWEEP trials=%llu clean=%llu corrected=%llu refusal=%llu stuck=%llu sdc=%llu sdc_cert=%llu\n",
         (unsigned long long)sb_trials, (unsigned long long)sb_clean,
         (unsigned long long)sb_corr, (unsigned long long)sb_ref,
         (unsigned long long)sb_stuck, (unsigned long long)sb_sdc,
         (unsigned long long)sb_sdc_cert);
  fflush(stdout);

  // Restore canonical ARENA1 (0x200000) in mov r9d at 0x400015
  {
    u32 a1_canon = (u32)ARENA1;
    memcpy((u8*)KERNEL_BASE + 0x015, &a1_canon, 4);
  }

  // 2.5 Multi-Burst Monte Carlo Sweeps across q_burst in {1/100, 1/50, 1/20, 1/10}
  const u64 q_dens[4] = {100, 50, 20, 10};
  const u64 tol_nums[4] = {1, 2, 3, 5}; // binomial 4-sigma tolerances in /1000
  for (int qi = 0; qi < 4; ++qi) {
    u64 q_den = q_dens[qi];
    u64 c_clean = 0, c_corr = 0, c_ref = 0, c_stuck = 0, c_sdc = 0, c_sdc_cert = 0;
    for (u64 t = 0; t < mc_trials; ++t) {
      u64 w = t % 3;
      u64 step0 = 50 + (t % 40);
      u64 last0 = 10 + (t % 15);
      u64 acc0 = 2000 + (t % 500) * 10;
      u64 val0 = 0x33330000ULL + t;
      u64 v_new = 0xcccc0000ULL + t;
      u64 delta = step0 - last0 + 1;
      u64 exp_step = step0 + 1;
      u64 exp_acc = acc0 + delta * delta;

      init_three_arenas(w, step0, acc0, val0, last0);
      bool any_burst = false;
      for (int rep = 0; rep < 3; ++rep) {
        if (bernoulli(1, q_den)) {
          any_burst = true;
          int pat_kind = (int)(rnd64() % 6);
          inject_64b_burst((u8*)arena_bases[rep], pat_kind, step0);
        }
      }

      u64 out_rax = 0;
      u32 code = run_kernel_fast_native(w, v_new, &out_rax);
      bool sdc_cert = false;
      BurstTrialOut res = evaluate_burst_outcome(
          code, out_rax, any_burst, (const u8*)ARENA0, (const u8*)ARENA1, (const u8*)ARENA2,
          w, exp_step, exp_acc, v_new, &sdc_cert);
      if (res == BURST_CLEAN_WIN) c_clean++;
      else if (res == BURST_CORRECTED) c_corr++;
      else if (res == BURST_REFUSAL) c_ref++;
      else if (res == BURST_STUCK) c_stuck++;
      else {
        c_sdc++;
        if (sdc_cert) c_sdc_cert++;
      }
    }
    u64 c_err = c_ref + c_stuck + c_sdc;
    printf("BURST_MC q=1/%llu trials=%llu clean=%llu corrected=%llu refusal=%llu stuck=%llu sdc=%llu sdc_cert=%llu err=%llu tol=%llu/1000\n",
           (unsigned long long)q_den, (unsigned long long)mc_trials,
           (unsigned long long)c_clean, (unsigned long long)c_corr,
           (unsigned long long)c_ref, (unsigned long long)c_stuck,
           (unsigned long long)c_sdc, (unsigned long long)c_sdc_cert,
           (unsigned long long)c_err, (unsigned long long)tol_nums[qi]);
    fflush(stdout);
  }

  // 2.6 Packed-Arena (< 64B Separation) Negative Control:
  //     Place Arena 0 at 0x100000 and Arena 1 at 0x100020 (32 bytes apart < 64B,
  //     both inside the single 64-byte cache line [0x100000, 0x100040)!).
  //     Apply a SINGLE 64-byte cache-line burst at 0x100000 and execute the native
  //     machine-code kernel: verify that a single 64B burst corrupts BOTH Arena 0
  //     and Arena 1, outvoting Arena 2 and causing SDC > 0 and SDC_cert > 0.
  u64 pk_trials = 0, pk_sdc = 0, pk_sdc_cert = 0;
  {
    u32 packed_a1 = (u32)PACKED_ARENA1;
    memcpy((u8*)KERNEL_BASE + 0x015, &packed_a1, 4);

    for (u64 t = 0; t < 64; ++t) {
      u64 w = 0;
      u64 step0 = 100 + t;
      u64 last0 = 10;
      u64 acc0 = 5000 + t * 100;
      u64 val0 = 0x11112222ULL + t;
      u64 v_new = 0x88889999ULL + t;
      u64 delta = step0 - last0 + 1;
      u64 exp_step = step0 + 1;
      u64 exp_acc = acc0 + delta * delta;

      set_arena_state((u8*)ARENA0, w, step0, acc0, val0, last0);
      set_arena_state((u8*)PACKED_ARENA1, w, step0, acc0, val0, last0);
      set_arena_state((u8*)ARENA2, w, step0, acc0, val0, last0);

      // Single 64-byte burst at 0x100000..0x10003f hitting both Arena 0 and Arena 1:
      // Forges acc = 0 and last_step[0] = step0 across both 32-byte half-lines!
      u8 burst64[64];
      memset(burst64, 0, 64);
      write_le64(burst64 + 0, step0);
      write_le64(burst64 + 8, 0ULL);
      write_le64(burst64 + 16, 0xbad0ULL);
      write_le64(burst64 + 24, step0);
      write_le64(burst64 + 32 + 0, step0);
      write_le64(burst64 + 32 + 8, 0ULL);
      write_le64(burst64 + 32 + 16, 0xbad0ULL);
      write_le64(burst64 + 32 + 24, step0);
      memcpy((void*)ARENA0, burst64, 64);

      u64 out_rax = 0;
      u32 code = run_kernel_fast_native(w, v_new, &out_rax);
      bool sdc_cert = false;
      BurstTrialOut res = evaluate_burst_outcome(
          code, out_rax, true, (const u8*)ARENA0, (const u8*)PACKED_ARENA1, (const u8*)ARENA2,
          w, exp_step, exp_acc, v_new, &sdc_cert);
      pk_trials++;
      if (res == BURST_SDC) {
        pk_sdc++;
        if (sdc_cert) pk_sdc_cert++;
      }
    }

    u32 canon_a1 = (u32)ARENA1;
    memcpy((u8*)KERNEL_BASE + 0x015, &canon_a1, 4);
  }
  bool packed_neg_ok = (pk_trials == 64 && pk_sdc == 64 && pk_sdc_cert == 64);
  printf("PACKED_NEG_CTRL %s trials=%llu sdc=%llu sdc_cert=%llu\n",
         packed_neg_ok ? "PASS" : "FAIL",
         (unsigned long long)pk_trials,
         (unsigned long long)pk_sdc,
         (unsigned long long)pk_sdc_cert);
  fflush(stdout);

  bool all_ok = (straddle_disagreements == 0) && straddle_neg_ok && guard_ok &&
                exit3_ok && exit2_ok && (tf_sdc == 0) && (tf_sdc_cert == 0) &&
                (sb_sdc == 0) && (sb_sdc_cert == 0) && packed_neg_ok;
  return all_ok ? 0 : 1;
}
