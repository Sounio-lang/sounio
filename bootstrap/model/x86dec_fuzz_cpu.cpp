// x86dec_fuzz_cpu: the x86-64 CPU execution engine for the X86Dec.altsAt
// Hamming-1/2 and adversarial byte-boundary differential fuzzer.
//
// Allocates two contiguous pages:
//   - Page 0: [0xfffff000, 0x100000000) mapped PROT_READ | PROT_WRITE | PROT_EXEC
//   - Page 1: [0x100000000, 0x100001000) mapped PROT_NONE (unmapped guard page)
//
// For each instruction sequence of decoded length `len` accepted by X86Dec.decode
// (emitted by formal/Bootstrap/X86DecFuzzModel.lean), copies ONLY the exact `len`
// bytes to `[0x100000000 - len, 0x100000000)` right against the PROT_NONE guard
// page and single-steps the CPU via RFLAGS.TF = 1 (SIGTRAP + ucontext_t on an
// alternate signal stack).
//
// Any over-read beyond `len` into Page 1 triggers SIGSEGV on the guard page;
// any under-read `< len` on a non-branch instruction lands before 0x100000000;
// and all 16 GPRs, CF/SF/ZF/OF flags, rip delta, and 64 memory bytes at
// [0x117e0, 0x11820) are compared against X86Int.step.

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
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

static const int GREG_IDX[16] = {
    REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
    REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};

static u64 xs_step(u64 x) {
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return x;
}

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

static void write_le64(u8* dst, u64 v) {
  for (int i = 0; i < 8; ++i) dst[i] = (u8)(v >> (8 * i));
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

struct CaseSpec {
  u8 bytes[16];
  u32 len;
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
static CaseSpec batch[BATCH_MAX];
static u32 batch_len = 0;
static volatile u32 batch_pos = 0;
static volatile bool batch_active = false;
static ucontext_t host_ctx;

static u64 total_executed = 0;
static u64 total_disagreements = 0;
static u64 total_guard_faults = 0;
static u64 total_len_mismatches = 0;

static u64 cur_in_regs[16];
static u8 cur_in_mem[WIN_SIZE];

static void setup_case(u32 idx, ucontext_t* uc) {
  const CaseSpec& cs = batch[idx];
  const BaseState& bs = pool[cs.pool_idx % POOL_SIZE];
  for (int r = 0; r < 16; ++r) cur_in_regs[r] = bs.regs[r];
  if (cs.fix_reg_idx < 16) cur_in_regs[cs.fix_reg_idx] = cs.fix_reg_val;

  memcpy(cur_in_mem, bs.mem, WIN_SIZE);
  if (cs.fix_mem_off <= 56) write_le64(&cur_in_mem[cs.fix_mem_off], cs.fix_mem_val);
  memcpy((void*)WIN_LO, cur_in_mem, WIN_SIZE);

  // Fill last 32 bytes of PAGE0 with 0x06 (#UD in 64-bit mode) and place
  // ONLY the `cs.len` decoded bytes at `[PAGE0_END - cs.len, PAGE0_END)`.
  memset((void*)(PAGE0_END - 32), 0x06, 32);
  u64 ins_addr = PAGE0_END - cs.len;
  memcpy((void*)ins_addr, cs.bytes, cs.len);

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

static void check_case_result(u32 idx, int sig, siginfo_t* si, ucontext_t* uc) {
  const CaseSpec& cs = batch[idx];
  u64 ins_addr = PAGE0_END - cs.len;
  total_executed++;

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
    if (sig == SIGSEGV) total_guard_faults++;
    total_disagreements++;
    if (total_disagreements <= 20) {
      printf("DIFF ");
      for (u32 i = 0; i < cs.len; ++i) printf("%02x", cs.bytes[i]);
      printf(" (len=%u)\n", cs.len);
      printf("  cpu   SIGNAL_%d rip=%016llx addr=%016llx\n", sig,
             (unsigned long long)uc->uc_mcontext.gregs[REG_RIP],
             (unsigned long long)(si ? (u64)si->si_addr : 0ULL));
      printf("  model delta=%016llx\n", (unsigned long long)cs.exp_delta);
    }
    return;
  }

  u64 next_rip = (u64)uc->uc_mcontext.gregs[REG_RIP];
  u64 cpu_delta = next_rip - ins_addr;
  bool len_bad = (!cs.is_branch && next_rip != PAGE0_END);
  if (len_bad) total_len_mismatches++;

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

  const u8* cpu_mem = (const u8*)WIN_LO;
  if (memcmp(cpu_mem, exp_mem, WIN_SIZE) != 0) bad = true;

  if (bad) {
    total_disagreements++;
    if (total_disagreements <= 20) {
      printf("DIFF ");
      for (u32 i = 0; i < cs.len; ++i) printf("%02x", cs.bytes[i]);
      printf(" (len=%u)\n", cs.len);
      printf("  cpu   %016llx", (unsigned long long)cpu_delta);
      for (int r = 0; r < 16; ++r) {
        printf(" %016llx", (unsigned long long)uc->uc_mcontext.gregs[GREG_IDX[r]]);
      }
      printf(" %s ", cpu_fl);
      for (u32 j = 0; j < WIN_SIZE; ++j) printf("%02x", cpu_mem[j]);
      printf("\n");

      printf("  model %016llx", (unsigned long long)cs.exp_delta);
      for (int r = 0; r < 16; ++r) {
        printf(" %016llx", (unsigned long long)exp_regs[r]);
      }
      printf(" %s ", cs.exp_fl);
      for (u32 j = 0; j < WIN_SIZE; ++j) printf("%02x", exp_mem[j]);
      printf("\n");
    }
  }
}

static void fuzz_sig_handler(int sig, siginfo_t* si, void* uctx) {
  ucontext_t* uc = (ucontext_t*)uctx;
  if (!batch_active) {
    batch_active = true;
    host_ctx = *uc;
    host_ctx.uc_mcontext.gregs[REG_EFL] &= ~0x100ULL;
    batch_pos = 0;
    setup_case(0, uc);
    return;
  }

  u32 finished_idx = batch_pos;
  check_case_result(finished_idx, sig, si, uc);
  u32 next_idx = finished_idx + 1;
  batch_pos = next_idx;
  if (next_idx < batch_len) {
    setup_case(next_idx, uc);
  } else {
    *uc = host_ctx;
  }
}

[[gnu::noinline]] static void run_batch() {
  if (batch_len == 0) return;
  batch_active = false;
  batch_pos = 0;
  asm volatile("int3" : : : "memory", "cc");
}

static bool parse_case_line(const char* line, CaseSpec& cs) {
  char cand_hex[64];
  char fix_reg_hex[32], fix_mem_hex[32];
  char delta_hex[32], mod1_hex[32], mod2_hex[32], mod_mem_hex[32];
  int matched = sscanf(
      line, "%63s %u %u %u %31s %u %31s %u | %31s %u %31s %u %31s %4s %u %u %31s",
      cand_hex, &cs.len, &cs.pool_idx, &cs.fix_reg_idx, fix_reg_hex,
      &cs.fix_mem_off, fix_mem_hex, &cs.is_branch, delta_hex, &cs.mod1_idx,
      mod1_hex, &cs.mod2_idx, mod2_hex, cs.exp_fl, &cs.mod_mem_off,
      &cs.mod_mem_len, mod_mem_hex);
  if (matched != 17 || cs.len == 0 || cs.len > 15) return false;
  for (u32 i = 0; i < cs.len; ++i) {
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

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: x86dec_fuzz_cpu <seed> <cases_file>\n");
    return 2;
  }
  u64 seed = strtoull(argv[1], nullptr, 10);
  const char* cases_path = argv[2];

  if (mmap((void*)DATA_PAGE, 4096, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != (void*)DATA_PAGE) {
    fprintf(stderr, "mmap DATA_PAGE failed: %s\n", strerror(errno));
    return 2;
  }
  // Allocate two contiguous pages: Page 0 (RWX) and Page 1 (PROT_NONE guard page)
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

  init_pool(seed);

  stack_t ss{};
  ss.ss_sp = mmap(nullptr, 65536, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ss.ss_size = 65536;
  if (sigaltstack(&ss, nullptr) != 0) {
    fprintf(stderr, "sigaltstack failed\n");
    return 2;
  }

  struct sigaction sa{};
  sa.sa_sigaction = fuzz_sig_handler;
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
    if (!parse_case_line(line, batch[batch_len])) {
      fprintf(stderr, "unparsed case line: %s", line);
      fclose(fp);
      return 2;
    }
    batch_len++;
    if (batch_len == BATCH_MAX) {
      run_batch();
      batch_len = 0;
    }
  }
  if (batch_len > 0) {
    run_batch();
    batch_len = 0;
  }
  fclose(fp);

  printf("cpu_executed=%llu\n", (unsigned long long)total_executed);
  printf("cpu_guard_faults=%llu\n", (unsigned long long)total_guard_faults);
  printf("cpu_len_mismatches=%llu\n", (unsigned long long)total_len_mismatches);
  printf("cpu_disagreements=%llu\n", (unsigned long long)total_disagreements);
  fflush(stdout);

  return (total_disagreements == 0 && total_guard_faults == 0 && total_len_mismatches == 0) ? 0 : 1;
}
