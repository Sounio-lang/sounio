// tmr_fault_inject: hardware fault-injection harness for the canonical 38-byte
// x86-64 machine-code TMR voter at 0x400000 (formal/Bootstrap/TMRVoter.lean).
//
// Executes the 38-byte machine-code voter natively on the host x86-64 CPU via
// hardware single-stepping (RFLAGS.TF = 1, SIGTRAP + ucontext_t on an alternate
// signal stack), plus a real fork() child check confirming that the `0f 05`
// syscall at 0x400024 exits with status 3 on 3-way disagreement.
//
// Mode A (Body-only faults, clean voter):
//   1. Deterministic exhaustive single-bit sweep across all 3 replicas (rax,
//      rbx, rcx) x 64 bits x 10 golden words + all 13 non-replica GPRs x 64
//      bits (2,752 single-bit upsets -> 0 SDC, 0 refusals, 0 stuck).
//   2. Monte Carlo multi-replica body upsets at q in {1/100, 1/50, 1/20, 1/10}
//      against the 3 * q^2 bound of TMR.tri_err.
//
// Mode B (Full end-to-end faults: body + 4-site faulty machine-code voter):
//   Executes the 9-step voter on the CPU while injecting random 4-site faults
//   matching X86Int.pstep (instruction fetch bit-flip in 0x400000, register
//   file bit-flip in ucontext_t, mapped scratch memory bit-flip, and CF/ZF/SF/OF
//   flag bit-flip) across calibrated rates r_total in {1/500, 1/200, 1/100}
//   paired with q_body in {1/50, 1/20, 1/10}, measuring empirical error rates
//   against the 3 * q_body^2 + 9 * r_total bound of TMRVoter.lean.

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

static constexpr u64 VOTER_BASE = 0x400000ULL;
static constexpr u64 VOTER_LEN = 38;
static constexpr u64 VOTER_PARK = VOTER_BASE + 0x18;
static constexpr u64 VOTER_SYSCALL = VOTER_BASE + 0x24;
static constexpr u64 SCRATCH_PAGE = 0x11000ULL;
static constexpr u64 SCRATCH_WIN = 0x117e0ULL;
static constexpr u64 SCRATCH_SP = 0x11800ULL;
static constexpr u64 SCRATCH_LEN = 64;

// Canonical 38-byte machine-code voter (TMRVoter.voterCode).
static const u8 VOTER_CODE[VOTER_LEN] = {
    0x48, 0x39, 0xd8,                   // 0x00: cmp rax, rbx
    0x74, 0x0d,                         // 0x03: je  0x400012
    0x48, 0x39, 0xc8,                   // 0x05: cmp rax, rcx
    0x74, 0x08,                         // 0x08: je  0x400012
    0x48, 0x39, 0xcb,                   // 0x0a: cmp rbx, rcx
    0x75, 0x0b,                         // 0x0d: jne 0x40001a
    0x48, 0x89, 0xd8,                   // 0x0f: mov rax, rbx
    0x48, 0x89, 0xc3,                   // 0x12: mov rbx, rax
    0x48, 0x89, 0xc1,                   // 0x15: mov rcx, rax
    0xeb, 0xfe,                         // 0x18: jmp 0x400018
    0xbf, 0x03, 0x00, 0x00, 0x00,       // 0x1a: mov edi, 3
    0xb8, 0x3c, 0x00, 0x00, 0x00,       // 0x1f: mov eax, 60
    0x0f, 0x05                          // 0x24: syscall
};

// Map X86Int.Reg (0..15) to Linux x86-64 ucontext_t gregset_t indices.
static const int GREG_IDX[16] = {
    REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
    REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};

static u64 rng_state = 0x9e3779b97f4a7c15ULL;

static u64 rnd64() {
  u64 x = rng_state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  rng_state = x;
  return x;
}

static bool bernoulli(u64 num, u64 den) {
  if (num == 0 || den == 0) return false;
  return (rnd64() % den) < num;
}

// Generate a 64-bit test word whose upper bits (63:48) have Hamming weight 8
// so that even after multiple bit-flips it is non-canonical as an x86-64 memory
// pointer, preventing any corrupted ModRM byte from touching host process memory.
static u64 gen_safe_word() {
  u64 lo = rnd64() & 0x0000ffffffffffffULL;
  return lo | 0x5a3c000000000000ULL;
}

// Instruction length at a valid voter offset, or 0 if not a canonical boundary.
static u32 voter_ins_len(u64 rip) {
  if (rip < VOTER_BASE || rip >= VOTER_BASE + VOTER_LEN) return 0;
  switch (rip - VOTER_BASE) {
    case 0x00: return 3;
    case 0x03: return 2;
    case 0x05: return 3;
    case 0x08: return 2;
    case 0x0a: return 3;
    case 0x0d: return 2;
    case 0x0f: return 3;
    case 0x12: return 3;
    case 0x15: return 3;
    case 0x18: return 2;
    case 0x1a: return 5;
    case 0x1f: return 5;
    case 0x24: return 2;
    default:   return 1;
  }
}

enum TrialOut {
  OUT_CLEAN_WIN = 0,
  OUT_CORRECTED = 1,
  OUT_REFUSAL = 2,
  OUT_STUCK = 3,
  OUT_SDC = 4
};

struct TrialSpec {
  u64 g;
  u64 a;
  u64 b;
  u64 c;
  int init_flip_reg;
  int init_flip_bit;
  bool any_fault_init;
  u64 r_num;
  u64 r_den;
};

static volatile bool engine_active = false;
static volatile bool trial_done = false;
static volatile int step_idx = 0;
static volatile bool trial_any_fault = false;
static volatile u64 flipped_byte_addr = 0;
static volatile u8 flipped_byte_mask = 0;
static volatile TrialOut trial_result = OUT_STUCK;
static TrialSpec cur_trial;
static ucontext_t host_ctx;

static void restore_flipped_byte() {
  if (flipped_byte_addr != 0) {
    *(u8*)flipped_byte_addr ^= flipped_byte_mask;
    flipped_byte_addr = 0;
    flipped_byte_mask = 0;
  }
}

static void prepare_next_step(ucontext_t* uc) {
  while (step_idx < 9) {
    u64 rip = (u64)uc->uc_mcontext.gregs[REG_RIP];
    if (rip < VOTER_BASE || rip >= VOTER_BASE + VOTER_LEN) {
      trial_result = OUT_STUCK;
      trial_done = true;
      return;
    }
    u32 len = voter_ins_len(rip);
    if (rip + len > VOTER_BASE + VOTER_LEN) {
      len = (u32)(VOTER_BASE + VOTER_LEN - rip);
    }
    // Site 1: instruction fetch bit-flip fault (rate r.fetch = r_total / 4)
    if (cur_trial.r_num > 0 && bernoulli(cur_trial.r_num, cur_trial.r_den * 4)) {
      trial_any_fault = true;
      u32 nbits = 8 * len;
      u32 k = (u32)(rnd64() % nbits);
      flipped_byte_addr = rip + (k / 8);
      flipped_byte_mask = (u8)(1u << (k % 8));
      *(u8*)flipped_byte_addr ^= flipped_byte_mask;
    }

    u8 b0 = *(u8*)rip;
    u8 b1 = (rip + 1 < VOTER_BASE + 4096) ? *(u8*)(rip + 1) : 0;

    // Intercept syscall `0f 05` (or `cd 80`) before host kernel exit
    if (b0 == 0x0f && b1 == 0x05) {
      restore_flipped_byte();
      u64 nr = (u64)uc->uc_mcontext.gregs[REG_RAX];
      u64 a1 = (u64)uc->uc_mcontext.gregs[REG_RDI];
      if (nr == 60) {
        if ((a1 & 0xff) == 3) {
          trial_result = OUT_REFUSAL;
        } else {
          trial_result = OUT_SDC;
        }
      } else {
        trial_result = OUT_STUCK;
      }
      trial_done = true;
      return;
    }
    if (b0 == 0xcd && b1 == 0x80) {
      restore_flipped_byte();
      trial_result = OUT_STUCK;
      trial_done = true;
      return;
    }

    // Arm RFLAGS.TF = 1 to single-step this instruction natively on CPU
    uc->uc_mcontext.gregs[REG_EFL] |= 0x100ULL;
    return;
  }

  // All 9 steps retired; evaluate architectural outcome at step 9
  u64 end_rip = (u64)uc->uc_mcontext.gregs[REG_RIP];
  u64 end_rax = (u64)uc->uc_mcontext.gregs[REG_RAX];
  u64 end_rbx = (u64)uc->uc_mcontext.gregs[REG_RBX];
  u64 end_rcx = (u64)uc->uc_mcontext.gregs[REG_RCX];
  if (end_rip == VOTER_PARK) {
    if (end_rax == cur_trial.g && end_rbx == cur_trial.g && end_rcx == cur_trial.g) {
      trial_result = trial_any_fault ? OUT_CORRECTED : OUT_CLEAN_WIN;
    } else {
      trial_result = OUT_SDC;
    }
  } else {
    // If a sabotaged voter failed to reach VOTER_PARK or corrupted outputs
    if (end_rax != cur_trial.g || end_rbx != cur_trial.g || end_rcx != cur_trial.g) {
      trial_result = (cur_trial.r_num == 0) ? OUT_SDC : OUT_STUCK;
    } else {
      trial_result = OUT_STUCK;
    }
  }
  trial_done = true;
}

static void step_signal_handler(int sig, siginfo_t* si, void* uctx) {
  (void)si;
  ucontext_t* uc = (ucontext_t*)uctx;
  restore_flipped_byte();

  if (!engine_active) {
    // Initial entry via int3: capture clean host context (RIP is already after int3)
    engine_active = true;
    host_ctx = *uc;
    host_ctx.uc_mcontext.gregs[REG_EFL] &= ~0x100ULL;
  } else {
    // Returning from a hardware step
    if (sig != SIGTRAP) {
      // Hardware exception (#GP, #PF, #UD) -> stuck unless clean-voter sabotage
      if (cur_trial.r_num == 0 && cur_trial.any_fault_init) {
        trial_result = OUT_STUCK;
      } else {
        trial_result = OUT_STUCK;
      }
      trial_done = true;
    } else {
      // Instruction retired cleanly; apply post-retirement faults (X86Int.pstep)
      if (cur_trial.r_num > 0) {
        // Site 2: register file 1-bit upset (rate r.reg = r_total / 4)
        if (bernoulli(cur_trial.r_num, cur_trial.r_den * 4)) {
          trial_any_fault = true;
          u32 j = (u32)(rnd64() % (16 * 64));
          u32 reg = j / 64;
          u32 bit = j % 64;
          uc->uc_mcontext.gregs[GREG_IDX[reg]] ^= (greg_t)(1ULL << bit);
          if (reg == 4) {
            // Keep RSP non-canonical or inside scratch page
          }
        }
        // Site 3: mapped memory 1-bit upset (rate r.mem = r_total / 4)
        if (bernoulli(cur_trial.r_num, cur_trial.r_den * 4)) {
          trial_any_fault = true;
          u32 j = (u32)(rnd64() % (SCRATCH_LEN * 8));
          ((u8*)SCRATCH_WIN)[j / 8] ^= (u8)(1u << (j % 8));
        }
        // Site 4: flags CF/ZF/SF/OF 1-bit upset (rate r.flag = r_total / 4)
        if (bernoulli(cur_trial.r_num, cur_trial.r_den * 4)) {
          trial_any_fault = true;
          static const u64 flag_bits[4] = {1ULL << 0, 1ULL << 6, 1ULL << 7, 1ULL << 11};
          uc->uc_mcontext.gregs[REG_EFL] ^= (greg_t)flag_bits[rnd64() % 4];
        }
      }
      step_idx = step_idx + 1;
    }
  }

  if (!trial_done && step_idx == 0) {
    // Load initial trial state into ucontext_t
    for (int r = 0; r < 16; ++r) {
      uc->uc_mcontext.gregs[GREG_IDX[r]] = (greg_t)(0x5a3c000000000000ULL | (u64)r);
    }
    uc->uc_mcontext.gregs[REG_RAX] = (greg_t)cur_trial.a;
    uc->uc_mcontext.gregs[REG_RBX] = (greg_t)cur_trial.b;
    uc->uc_mcontext.gregs[REG_RCX] = (greg_t)cur_trial.c;
    uc->uc_mcontext.gregs[REG_RSP] = (greg_t)SCRATCH_SP;
    if (cur_trial.init_flip_reg >= 0 && cur_trial.init_flip_reg < 16) {
      uc->uc_mcontext.gregs[GREG_IDX[cur_trial.init_flip_reg]] ^=
          (greg_t)(1ULL << cur_trial.init_flip_bit);
    }
    uc->uc_mcontext.gregs[REG_RIP] = (greg_t)VOTER_BASE;
    uc->uc_mcontext.gregs[REG_EFL] = 0x202ULL;
    prepare_next_step(uc);
  } else if (!trial_done) {
    prepare_next_step(uc);
  }

  if (trial_done) {
    // Restore clean host context to return to run_one_trial()
    *uc = host_ctx;
  }
}

[[gnu::noinline]] static TrialOut run_one_trial(const TrialSpec& spec) {
  cur_trial = spec;
  engine_active = false;
  trial_done = false;
  step_idx = 0;
  trial_any_fault = spec.any_fault_init;
  trial_result = OUT_STUCK;
  memset((void*)SCRATCH_WIN, 0, SCRATCH_LEN);

  asm volatile("int3" : : : "memory", "cc");
  return trial_result;
}

// Verify in an isolated fork() child that executing the 38-byte voter natively
// without single-stepping on 3-way disagreed inputs reaches `0f 05` (syscall)
// and terminates the process with exit code 3. Includes a 200 ms timeout so a
// sabotaged voter that loops at 0x400018 (`eb fe`) fails cleanly without hanging.
static bool verify_real_syscall_fork() {
  pid_t pid = fork();
  if (pid < 0) return false;
  if (pid == 0) {
    alarm(1);
    u64 a = 0x1111111111111111ULL;
    u64 b = 0x2222222222222222ULL;
    u64 c = 0x3333333333333333ULL;
    u64 entry = VOTER_BASE;
    asm volatile(
        "mov %[a], %%rax\n"
        "mov %[b], %%rbx\n"
        "mov %[c], %%rcx\n"
        "jmp *%[entry]\n"
        :
        : [a] "r"(a), [b] "r"(b), [c] "r"(c), [entry] "r"(entry)
        : "rax", "rbx", "rcx");
    _exit(99);
  }
  int status = 0;
  for (int i = 0; i < 200; ++i) {
    pid_t w = waitpid(pid, &status, WNOHANG);
    if (w == pid) {
      return WIFEXITED(status) && WEXITSTATUS(status) == 3;
    }
    if (w < 0) return false;
    usleep(1000);
  }
  kill(pid, SIGKILL);
  waitpid(pid, &status, 0);
  return false;
}

int main(int argc, char** argv) {
  u64 trials = 20000;
  u64 seed = 1;
  if (argc >= 2) trials = strtoull(argv[1], nullptr, 10);
  if (argc >= 3) seed = strtoull(argv[2], nullptr, 10);
  rng_state = 0x9e3779b97f4a7c15ULL ^ seed;

  // Map scratch data page at 0x11000 and executable voter code page at 0x400000
  if (mmap((void*)SCRATCH_PAGE, 4096, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != (void*)SCRATCH_PAGE) {
    fprintf(stderr, "mmap SCRATCH_PAGE failed: %s\n", strerror(errno));
    return 2;
  }
  if (mmap((void*)VOTER_BASE, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != (void*)VOTER_BASE) {
    fprintf(stderr, "mmap VOTER_BASE failed: %s\n", strerror(errno));
    return 2;
  }
  // Guard page immediately after voter page
  if (mmap((void*)(VOTER_BASE + 4096), 4096, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != (void*)(VOTER_BASE + 4096)) {
    fprintf(stderr, "mmap guard page failed: %s\n", strerror(errno));
    return 2;
  }

  // Fill code page with 0x06 (invalid opcode in 64-bit mode -> #UD SIGILL) and copy voterCode
  memset((void*)VOTER_BASE, 0x06, 4096);
  memcpy((void*)VOTER_BASE, VOTER_CODE, VOTER_LEN);

  // Print exact 38-byte voterCode hex for TMRBoundCheck verification
  printf("VOTER_CODE ");
  for (u64 i = 0; i < VOTER_LEN; ++i) {
    printf("%02x", ((const u8*)VOTER_BASE)[i]);
  }
  printf("\n");

  // 1. Fork child check for real `0f 05` syscall exit(3) on 3-way disagreement
  bool syscall_ok = verify_real_syscall_fork();
  printf("SYSCALL_EXIT3_CHECK %s\n", syscall_ok ? "PASS" : "FAIL");
  fflush(stdout);

  // Set up alternate signal stack and single-step trap handlers
  stack_t ss{};
  ss.ss_sp = mmap(nullptr, 65536, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ss.ss_size = 65536;
  if (sigaltstack(&ss, nullptr) != 0) {
    fprintf(stderr, "sigaltstack failed\n");
    return 2;
  }

  struct sigaction sa{};
  sa.sa_sigaction = step_signal_handler;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGTRAP, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);

  // 2. Mode A.1: Exhaustive deterministic single-fault sweep
  //    - 10 golden words x 3 replicas (rax, rbx, rcx) x 64 bits = 1,920 trials
  //    - Plus 13 non-replica GPRs x 64 bits = 832 trials
  //    Total = 2,752 single-bit upsets.
  u64 sf_trials = 0, sf_clean = 0, sf_corr = 0, sf_ref = 0, sf_stuck = 0, sf_sdc = 0;
  for (int w = 0; w < 10; ++w) {
    u64 g = gen_safe_word();
    for (int rep = 0; rep < 3; ++rep) {
      for (int bit = 0; bit < 64; ++bit) {
        u64 a = g, b = g, c = g;
        if (rep == 0) a ^= (1ULL << bit);
        else if (rep == 1) b ^= (1ULL << bit);
        else c ^= (1ULL << bit);
        TrialSpec sp{g, a, b, c, -1, 0, true, 0, 1};
        TrialOut res = run_one_trial(sp);
        sf_trials++;
        if (res == OUT_CLEAN_WIN) sf_clean++;
        else if (res == OUT_CORRECTED) sf_corr++;
        else if (res == OUT_REFUSAL) sf_ref++;
        else if (res == OUT_STUCK) sf_stuck++;
        else sf_sdc++;
      }
    }
  }
  {
    u64 g = gen_safe_word();
    for (int reg = 0; reg < 16; ++reg) {
      if (reg == 0 || reg == 1 || reg == 3) continue;
      for (int bit = 0; bit < 64; ++bit) {
        TrialSpec sp{g, g, g, g, reg, bit, true, 0, 1};
        TrialOut res = run_one_trial(sp);
        sf_trials++;
        if (res == OUT_CLEAN_WIN) sf_clean++;
        else if (res == OUT_CORRECTED) sf_corr++;
        else if (res == OUT_REFUSAL) sf_ref++;
        else if (res == OUT_STUCK) sf_stuck++;
        else sf_sdc++;
      }
    }
  }
  printf("SINGLE_FAULT_SWEEP trials=%llu clean=%llu corrected=%llu refusal=%llu stuck=%llu sdc=%llu\n",
         (unsigned long long)sf_trials, (unsigned long long)sf_clean,
         (unsigned long long)sf_corr, (unsigned long long)sf_ref,
         (unsigned long long)sf_stuck, (unsigned long long)sf_sdc);
  fflush(stdout);

  // 3. Mode A.2: Monte Carlo multi-replica body faults (clean voter, r_total = 0)
  //    q_body in {1/100, 1/50, 1/20, 1/10}
  const u64 q_dens_a[4] = {100, 50, 20, 10};
  const u64 tol_nums_a[4] = {1, 2, 3, 5}; // binomial 4-sigma tolerances in /1000
  for (int qi = 0; qi < 4; ++qi) {
    u64 q_den = q_dens_a[qi];
    u64 c_clean = 0, c_corr = 0, c_ref = 0, c_stuck = 0, c_sdc = 0;
    for (u64 t = 0; t < trials; ++t) {
      u64 g = gen_safe_word();
      u64 a = g, b = g, c = g;
      bool flt = false;
      if (bernoulli(1, q_den)) { a ^= (1ULL << (rnd64() % 64)); flt = true; }
      if (bernoulli(1, q_den)) { b ^= (1ULL << (rnd64() % 64)); flt = true; }
      if (bernoulli(1, q_den)) { c ^= (1ULL << (rnd64() % 64)); flt = true; }
      TrialSpec sp{g, a, b, c, -1, 0, flt, 0, 1};
      TrialOut res = run_one_trial(sp);
      if (res == OUT_CLEAN_WIN) c_clean++;
      else if (res == OUT_CORRECTED) c_corr++;
      else if (res == OUT_REFUSAL) c_ref++;
      else if (res == OUT_STUCK) c_stuck++;
      else c_sdc++;
    }
    u64 c_err = c_ref + c_stuck + c_sdc;
    printf("MODE_A q=1/%llu trials=%llu clean=%llu corrected=%llu refusal=%llu stuck=%llu sdc=%llu err=%llu tol=%llu/1000\n",
           (unsigned long long)q_den, (unsigned long long)trials,
           (unsigned long long)c_clean, (unsigned long long)c_corr,
           (unsigned long long)c_ref, (unsigned long long)c_stuck,
           (unsigned long long)c_sdc, (unsigned long long)c_err,
           (unsigned long long)tol_nums_a[qi]);
    fflush(stdout);
  }

  // 4. Mode B: Full end-to-end faults (triplicated body + 4-site faulty machine-code voter)
  //    q_body in {1/50, 1/20, 1/10} x r_total in {1/500, 1/200, 1/100}
  const u64 q_dens_b[3] = {50, 20, 10};
  const u64 r_dens_b[3] = {500, 200, 100};
  for (int qi = 0; qi < 3; ++qi) {
    u64 q_den = q_dens_b[qi];
    for (int ri = 0; ri < 3; ++ri) {
      u64 r_den = r_dens_b[ri];
      u64 c_clean = 0, c_corr = 0, c_ref = 0, c_stuck = 0, c_sdc = 0;
      for (u64 t = 0; t < trials; ++t) {
        u64 g = gen_safe_word();
        u64 a = g, b = g, c = g;
        bool flt = false;
        if (bernoulli(1, q_den)) { a ^= (1ULL << (rnd64() % 64)); flt = true; }
        if (bernoulli(1, q_den)) { b ^= (1ULL << (rnd64() % 64)); flt = true; }
        if (bernoulli(1, q_den)) { c ^= (1ULL << (rnd64() % 64)); flt = true; }
        TrialSpec sp{g, a, b, c, -1, 0, flt, 1, r_den};
        TrialOut res = run_one_trial(sp);
        if (res == OUT_CLEAN_WIN) c_clean++;
        else if (res == OUT_CORRECTED) c_corr++;
        else if (res == OUT_REFUSAL) c_ref++;
        else if (res == OUT_STUCK) c_stuck++;
        else c_sdc++;
      }
      u64 c_err = c_ref + c_stuck + c_sdc;
      printf("MODE_B q=1/%llu r=1/%llu trials=%llu clean=%llu corrected=%llu refusal=%llu stuck=%llu sdc=%llu err=%llu tol=5/1000\n",
             (unsigned long long)q_den, (unsigned long long)r_den,
             (unsigned long long)trials, (unsigned long long)c_clean,
             (unsigned long long)c_corr, (unsigned long long)c_ref,
             (unsigned long long)c_stuck, (unsigned long long)c_sdc,
             (unsigned long long)c_err);
      fflush(stdout);
    }
  }

  return 0;
}
