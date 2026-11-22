// Real x86-64 hardware differential oracle and Hamming-1 disturbance runner for
// X86Vm: executes randomized single instructions across all 33 X86Dec/X86Int
// forms and all 31 X86Vm forms (including #DE SIGFPE trap boundary cases on
// div/idiv), as well as `--exec <in.txt> <out.txt>` single-site Hamming-1
// disturbance cases emitted by `formal/Bootstrap/X86VmDiffModel.lean`.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <setjmp.h>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <ucontext.h>
#include <vector>

namespace {

constexpr uint64_t WIN_LO = 0x117e0ULL;
constexpr size_t WIN_SIZE = 64;
constexpr uint64_t SP_INT = 0x11800ULL;
constexpr uint64_t INS_ADDR = 0x20180ULL;
constexpr uint64_t CODE_PAGE = 0x20000ULL;
constexpr uint64_t STUB_PAGE = 0x21000ULL;
constexpr uint64_t STATE_PAGE = 0x22000ULL;

struct StateArea {
  uint64_t regs[16];
  uint64_t rflags;
};

static StateArea *g_state = nullptr;
static sigjmp_buf g_env;
static uint64_t g_post_regs[16];
static uint64_t g_post_rip = 0;
static uint64_t g_post_rflags = 0;
// 0 = none, 1 = SIGTRAP (normal retirement into 0xCC), 2 = SIGFPE (#DE)
static volatile sig_atomic_t g_trap_kind = 0;
static std::vector<uint8_t> g_altstack;

#if defined(__x86_64__) && defined(__linux__)
static void trap_handler(int, siginfo_t *, void *uctx_void) {
  auto *uc = static_cast<ucontext_t *>(uctx_void);
  const auto &gregs = uc->uc_mcontext.gregs;
  g_post_regs[0] = static_cast<uint64_t>(gregs[REG_RAX]);
  g_post_regs[1] = static_cast<uint64_t>(gregs[REG_RCX]);
  g_post_regs[2] = static_cast<uint64_t>(gregs[REG_RDX]);
  g_post_regs[3] = static_cast<uint64_t>(gregs[REG_RBX]);
  g_post_regs[4] = static_cast<uint64_t>(gregs[REG_RSP]);
  g_post_regs[5] = static_cast<uint64_t>(gregs[REG_RBP]);
  g_post_regs[6] = static_cast<uint64_t>(gregs[REG_RSI]);
  g_post_regs[7] = static_cast<uint64_t>(gregs[REG_RDI]);
  g_post_regs[8] = static_cast<uint64_t>(gregs[REG_R8]);
  g_post_regs[9] = static_cast<uint64_t>(gregs[REG_R9]);
  g_post_regs[10] = static_cast<uint64_t>(gregs[REG_R10]);
  g_post_regs[11] = static_cast<uint64_t>(gregs[REG_R11]);
  g_post_regs[12] = static_cast<uint64_t>(gregs[REG_R12]);
  g_post_regs[13] = static_cast<uint64_t>(gregs[REG_R13]);
  g_post_regs[14] = static_cast<uint64_t>(gregs[REG_R14]);
  g_post_regs[15] = static_cast<uint64_t>(gregs[REG_R15]);
  // int3 advances RIP by 1 past the 0xCC trap byte.
  g_post_rip = static_cast<uint64_t>(gregs[REG_RIP]) - 1ULL;
  g_post_rflags = static_cast<uint64_t>(gregs[REG_EFL]);
  g_trap_kind = 1;
  siglongjmp(g_env, 1);
}

static void fpe_handler(int, siginfo_t *, void *uctx_void) {
  auto *uc = static_cast<ucontext_t *>(uctx_void);
  const auto &gregs = uc->uc_mcontext.gregs;
  g_post_rip = static_cast<uint64_t>(gregs[REG_RIP]);
  g_trap_kind = 2;
  siglongjmp(g_env, 1);
}
#else
#error "x86vm_diff_cpu.cpp must be compiled and executed on x86-64 Linux"
#endif

static void emit_mov_reg_abs(std::vector<uint8_t> &buf, int reg, uint32_t addr32) {
  // 48/4c 8b ModRM(00, reg&7, 100) SIB(0x25) disp32
  uint8_t rex = 0x48 | ((reg >= 8) ? 0x04 : 0x00);
  buf.push_back(rex);
  buf.push_back(0x8b);
  buf.push_back(static_cast<uint8_t>(((reg & 7) << 3) | 0x04));
  buf.push_back(0x25);
  buf.push_back(static_cast<uint8_t>(addr32 & 0xff));
  buf.push_back(static_cast<uint8_t>((addr32 >> 8) & 0xff));
  buf.push_back(static_cast<uint8_t>((addr32 >> 16) & 0xff));
  buf.push_back(static_cast<uint8_t>((addr32 >> 24) & 0xff));
}

static void emit_stub() {
  std::vector<uint8_t> code;
  // mov rsp, <addr of g_state->rflags>; popfq
  code.push_back(0x48);
  code.push_back(0xbc);
  uint64_t fl_ptr = STATE_PAGE + offsetof(StateArea, rflags);
  for (int i = 0; i < 8; ++i) {
    code.push_back(static_cast<uint8_t>((fl_ptr >> (8 * i)) & 0xff));
  }
  code.push_back(0x9d); // popfq
  for (int r = 0; r < 16; ++r) {
    if (r == 4) continue; // load rsp last
    emit_mov_reg_abs(code, r, static_cast<uint32_t>(STATE_PAGE + r * 8));
  }
  emit_mov_reg_abs(code, 4, static_cast<uint32_t>(STATE_PAGE + 4 * 8));
  // jmp rel32 to INS_ADDR
  uint64_t next_ip = STUB_PAGE + code.size() + 5;
  int32_t rel = static_cast<int32_t>(static_cast<int64_t>(INS_ADDR) - static_cast<int64_t>(next_ip));
  code.push_back(0xe9);
  for (int i = 0; i < 4; ++i) {
    code.push_back(static_cast<uint8_t>((static_cast<uint32_t>(rel) >> (8 * i)) & 0xff));
  }
  auto *stub = reinterpret_cast<uint8_t *>(STUB_PAGE);
  std::memset(stub, 0xcc, 4096);
  std::memcpy(stub, code.data(), code.size());
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
  uint64_t next() {
    uint64_t x = s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    s = x;
    return x;
  }
  uint64_t range(uint64_t lo, uint64_t hi) { return lo + (next() % (hi - lo)); }
  uint8_t u8() { return static_cast<uint8_t>(next() & 0xff); }
  uint32_t u32() { return static_cast<uint32_t>(next() & 0xffffffffULL); }
  uint64_t rval() {
    switch (next() % 10) {
    case 0: return 0ULL;
    case 1: return 1ULL;
    case 2: return 0xffffffffULL;
    case 3: return 0x80000000ULL;
    case 4: return 0x7fffffffULL;
    case 5: return 0xffffffffffffffffULL;
    case 6: return 0x8000000000000000ULL;
    case 7: return 0x7fffffffffffffffULL;
    case 8: return (next() << 32) | 0ULL;
    default: return next();
    }
  }
  uint64_t shift_count() {
    switch (next() % 10) {
    case 0: return 0ULL;
    case 1: return 1ULL;
    case 2: return 63ULL;
    case 3: return 64ULL;
    case 4: return 65ULL;
    case 5: return 127ULL;
    case 6: return 128ULL;
    case 7: return (next() << 8) | (next() & 0x3f);
    case 8: return (next() << 8) | 0xffULL;
    default: return range(2, 63);
    }
  }
};

struct TestCase {
  std::string tag;
  std::vector<uint8_t> ins;
  uint64_t regs[16];
  uint8_t mem[WIN_SIZE];
  uint8_t cszo; // bits: 0=CF, 1=ZF, 2=SF, 3=OF
};

static uint8_t rex_byte(bool w, int r_reg, bool x, int b_reg, bool force) {
  uint8_t v = 0x40 | (w ? 8 : 0) | ((r_reg >= 8) ? 4 : 0) | (x ? 2 : 0) | ((b_reg >= 8) ? 1 : 0);
  if (v == 0x40 && !force) return 0;
  return v;
}

static void push_rex(std::vector<uint8_t> &ins, bool w, int r_reg, int b_reg, bool force = false) {
  uint8_t r = rex_byte(w, r_reg, false, b_reg, force);
  if (r != 0) ins.push_back(r);
}

static void push_rex_x(std::vector<uint8_t> &ins, bool w, int r_reg, int idx_reg, int b_reg, bool force = false) {
  uint8_t r = rex_byte(w, r_reg, idx_reg >= 8, b_reg, force);
  if (r != 0) ins.push_back(r);
}

static void push_le32(std::vector<uint8_t> &ins, uint32_t v) {
  for (int i = 0; i < 4; ++i) ins.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

static void push_le64(std::vector<uint8_t> &ins, uint64_t v) {
  for (int i = 0; i < 8; ++i) ins.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

static void init_common(Rng &rng, TestCase &tc) {
  tc.tag.clear();
  tc.ins.clear();
  for (int i = 0; i < 16; ++i) tc.regs[i] = rng.rval();
  tc.regs[4] = SP_INT;
  for (size_t i = 0; i < WIN_SIZE; ++i) tc.mem[i] = rng.u8();
  tc.cszo = static_cast<uint8_t>(rng.next() & 0xf);
}

// Non-indexed memory operand [base + disp], including SIB-no-index (base 4/12).
static void emit_memop(Rng &rng, std::vector<uint8_t> &ins, TestCase &tc,
                       bool w, int reg_field, uint8_t opcode, int access_bytes,
                       bool two_byte_0f = false, bool force_rex = false,
                       bool prefix_66 = false) {
  int mod = static_cast<int>(rng.range(0, 3));
  int base;
  do {
    base = static_cast<int>(rng.range(0, 16));
  } while (mod == 0 && (base & 7) == 5);
  if (prefix_66) ins.push_back(0x66);
  push_rex(ins, w, reg_field, base, force_rex);
  if (two_byte_0f) ins.push_back(0x0f);
  ins.push_back(opcode);
  ins.push_back(static_cast<uint8_t>((mod << 6) | ((reg_field & 7) << 3) | (base & 7)));
  if ((base & 7) == 4) {
    ins.push_back(0x24); // SIB: scale=1, no index (4), base=4/12
  }
  uint64_t off = rng.range(0, WIN_SIZE - static_cast<size_t>(access_bytes) + 1);
  uint64_t target = WIN_LO + off;
  if (mod == 0) {
    tc.regs[base] = target;
  } else if (mod == 1) {
    int8_t d8 = static_cast<int8_t>(rng.range(0, 256) - 128);
    ins.push_back(static_cast<uint8_t>(d8));
    tc.regs[base] = target - static_cast<uint64_t>(static_cast<int64_t>(d8));
  } else {
    int32_t d32 = static_cast<int32_t>(rng.u32());
    push_le32(ins, static_cast<uint32_t>(d32));
    tc.regs[base] = target - static_cast<uint64_t>(static_cast<int64_t>(d32));
  }
}

// SIB-indexed memory operand [base + scale * idx + disp], idx != 4, scale in {1,2,4,8}.
static void emit_memop_idx(Rng &rng, std::vector<uint8_t> &ins, TestCase &tc,
                           bool w, int reg_field, uint8_t opcode, int access_bytes,
                           bool two_byte_0f = false, bool force_rex = false,
                           bool prefix_66 = false) {
  int mod = static_cast<int>(rng.range(0, 3));
  int ss = static_cast<int>(rng.range(0, 4));
  uint64_t scale = 1ULL << ss;
  int base;
  do {
    base = static_cast<int>(rng.range(0, 16));
  } while (mod == 0 && (base & 7) == 5);
  int idx;
  do {
    idx = static_cast<int>(rng.range(0, 16));
  } while (idx == 4 || (access_bytes > 0 && idx == base));

  if (prefix_66) ins.push_back(0x66);
  push_rex_x(ins, w, reg_field, idx, base, force_rex);
  if (two_byte_0f) ins.push_back(0x0f);
  ins.push_back(opcode);
  ins.push_back(static_cast<uint8_t>((mod << 6) | ((reg_field & 7) << 3) | 4));
  ins.push_back(static_cast<uint8_t>((ss << 6) | ((idx & 7) << 3) | (base & 7)));

  int64_t disp = 0;
  if (mod == 1) {
    int8_t d8 = static_cast<int8_t>(rng.range(0, 256) - 128);
    ins.push_back(static_cast<uint8_t>(d8));
    disp = d8;
  } else if (mod == 2) {
    int32_t d32 = static_cast<int32_t>(rng.u32());
    push_le32(ins, static_cast<uint32_t>(d32));
    disp = d32;
  }

  if (access_bytes > 0) {
    uint64_t off = rng.range(0, WIN_SIZE - static_cast<size_t>(access_bytes) + 1);
    uint64_t target = WIN_LO + off;
    uint64_t idx_val = (rng.next() % 2 == 0) ? rng.range(0, 64) : rng.rval();
    tc.regs[idx] = idx_val;
    tc.regs[base] = target - idx_val * scale - static_cast<uint64_t>(disp);
  }
}

static uint64_t pick_branch_target(Rng &rng, uint64_t ins_len, bool short_rel8) {
  while (true) {
    uint64_t tgt = short_rel8
        ? (INS_ADDR + ins_len - 100 + rng.range(0, 200))
        : (CODE_PAGE + 0x100 + rng.range(0, 0xd00));
    if (tgt < INS_ADDR || tgt >= INS_ADDR + ins_len) return tgt;
  }
}

static uint8_t pick_cc(Rng &rng) {
  static constexpr uint8_t kCcs[14] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15};
  return kCcs[rng.next() % 14];
}

static uint64_t pick_nonzero_u64(Rng &rng) {
  switch (rng.next() % 8) {
  case 0: return 1ULL;
  case 1: return 2ULL;
  case 2: return 7ULL;
  case 3: return 0xffffffffULL;
  case 4: return 0x8000000000000000ULL;
  case 5: return 0xffffffffffffffffULL;
  default: {
    uint64_t v = rng.next();
    return v ? v : 1ULL;
  }
  }
}

static int64_t pick_nonzero_i64(Rng &rng) {
  switch (rng.next() % 10) {
  case 0: return 1;
  case 1: return -1;
  case 2: return 2;
  case 3: return -2;
  case 4: return 7;
  case 5: return -7;
  case 6: return INT64_MAX;
  case 7: return INT64_MIN;
  default: {
    int64_t v = static_cast<int64_t>(rng.next());
    return v ? v : 1;
  }
  }
}

static void gen_vm(Rng &rng, int kind, TestCase &tc) {
  init_common(rng, tc);
  switch (kind) {
  case 0: { // vm_mul (48/49 f7 /4)
    tc.tag = "vm_mul";
    int src = static_cast<int>(rng.range(0, 16));
    if (rng.next() % 4 == 0) {
      tc.regs[0] = (rng.next() % 2 == 0) ? 0x100000000ULL : 0xffffffffffffffffULL;
      tc.regs[src] = (rng.next() % 2 == 0) ? 0x100000000ULL : 0xffffffffffffffffULL;
    }
    push_rex(tc.ins, true, 0, src);
    tc.ins.push_back(0xf7);
    tc.ins.push_back(static_cast<uint8_t>(0xe0 | (src & 7)));
    break;
  }
  case 1: { // vm_imul (48..4d 0f af /r)
    tc.tag = "vm_imul";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    if (rng.next() % 4 == 0) {
      static constexpr uint64_t kEdges[6] = {
        0x4000000000000000ULL, 0xc000000000000000ULL, 0x8000000000000000ULL,
        0x7fffffffffffffffULL, 0xffffffffffffffffULL, 2ULL
      };
      tc.regs[d] = kEdges[rng.next() % 6];
      tc.regs[s] = kEdges[rng.next() % 6];
    }
    push_rex(tc.ins, true, d, s);
    tc.ins.push_back(0x0f);
    tc.ins.push_back(0xaf);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((d & 7) << 3) | (s & 7)));
    break;
  }
  case 2: { // vm_div (48/49 f7 /6, valid non-trapping)
    tc.tag = "vm_div";
    int src = static_cast<int>(rng.range(0, 16));
    if (src == 2) src = 3;
    if (src == 0) {
      uint64_t den = pick_nonzero_u64(rng);
      uint64_t hi = (rng.next() % 2 == 0) ? 0ULL : (rng.next() % den);
      tc.regs[0] = den;
      tc.regs[2] = hi;
    } else {
      uint64_t den = pick_nonzero_u64(rng);
      uint64_t hi;
      uint64_t mode = rng.next() % 3;
      if (mode == 0) hi = 0ULL;
      else if (mode == 1) hi = den - 1ULL;
      else hi = rng.next() % den;
      tc.regs[0] = rng.rval();
      tc.regs[2] = hi;
      tc.regs[src] = den;
    }
    push_rex(tc.ins, true, 0, src);
    tc.ins.push_back(0xf7);
    tc.ins.push_back(static_cast<uint8_t>(0xf0 | (src & 7)));
    break;
  }
  case 3: { // vm_div_trap (48/49 f7 /6, guaranteed #DE)
    tc.tag = "vm_div_trap";
    int src = static_cast<int>(rng.range(0, 16));
    uint64_t mode = rng.next() % 5;
    if (mode < 2) {
      tc.regs[src] = 0ULL;
    } else if (src == 2) {
      tc.regs[2] = pick_nonzero_u64(rng);
    } else if (src == 0) {
      uint64_t den = pick_nonzero_u64(rng);
      tc.regs[0] = den;
      tc.regs[2] = den;
    } else if (mode < 4) {
      uint64_t den = pick_nonzero_u64(rng);
      tc.regs[src] = den;
      tc.regs[2] = den;
      tc.regs[0] = (rng.next() % 2 == 0) ? 0ULL : rng.rval();
    } else {
      uint64_t den = (rng.next() % 0xfffffffffffffffeULL) + 1ULL;
      tc.regs[src] = den;
      tc.regs[2] = den + 1ULL + (rng.next() % (0xffffffffffffffffULL - den));
      tc.regs[0] = rng.rval();
    }
    push_rex(tc.ins, true, 0, src);
    tc.ins.push_back(0xf7);
    tc.ins.push_back(static_cast<uint8_t>(0xf0 | (src & 7)));
    break;
  }
  case 4: { // vm_idiv (48/49 f7 /7, valid non-trapping)
    tc.tag = "vm_idiv";
    int src = static_cast<int>(rng.range(0, 16));
    if (src == 2) src = 3;
    if (src == 0) {
      int64_t a = pick_nonzero_i64(rng);
      tc.regs[0] = static_cast<uint64_t>(a);
      tc.regs[2] = (a < 0) ? ~0ULL : 0ULL;
    } else if (rng.next() % 2 == 0) {
      int64_t a, den;
      switch (rng.next() % 10) {
      case 0: a = INT64_MIN; den = 1; break;
      case 1: a = INT64_MIN; den = 2; break;
      case 2: a = INT64_MIN; den = -2; break;
      case 3: a = INT64_MIN; den = INT64_MIN; break;
      case 4: a = INT64_MAX; den = -1; break;
      case 5: a = -7; den = 2; break;
      case 6: a = 7; den = -2; break;
      case 7: a = -7; den = -2; break;
      case 8: a = -1; den = INT64_MIN; break;
      default:
        a = static_cast<int64_t>(rng.rval());
        den = pick_nonzero_i64(rng);
        if (a == INT64_MIN && den == -1) den = 1;
        break;
      }
      tc.regs[0] = static_cast<uint64_t>(a);
      tc.regs[2] = (a < 0) ? ~0ULL : 0ULL;
      tc.regs[src] = static_cast<uint64_t>(den);
    } else {
      int64_t den = pick_nonzero_i64(rng);
      int64_t q;
      switch (rng.next() % 8) {
      case 0: q = INT64_MIN; break;
      case 1: q = INT64_MAX; break;
      case 2: q = 0; break;
      case 3: q = 1; break;
      case 4: q = -1; break;
      case 5: q = INT64_MIN + 1; break;
      case 6: q = INT64_MAX - 1; break;
      default: q = static_cast<int64_t>(rng.next()); break;
      }
      uint64_t abs_den = (den == INT64_MIN)
          ? (1ULL << 63)
          : static_cast<uint64_t>(den < 0 ? -den : den);
      uint64_t mag_r = (abs_den <= 1ULL) ? 0ULL : (rng.next() % abs_den);
      __int128 r128;
      if (q == 0) {
        r128 = (rng.next() % 2 == 0) ? static_cast<__int128>(mag_r) : -static_cast<__int128>(mag_r);
      } else if ((q > 0) == (den > 0)) {
        r128 = static_cast<__int128>(mag_r);
      } else {
        r128 = -static_cast<__int128>(mag_r);
      }
      __int128 num = static_cast<__int128>(q) * static_cast<__int128>(den) + r128;
      unsigned __int128 unum = static_cast<unsigned __int128>(num);
      tc.regs[0] = static_cast<uint64_t>(unum);
      tc.regs[2] = static_cast<uint64_t>(unum >> 64);
      tc.regs[src] = static_cast<uint64_t>(den);
    }
    push_rex(tc.ins, true, 0, src);
    tc.ins.push_back(0xf7);
    tc.ins.push_back(static_cast<uint8_t>(0xf8 | (src & 7)));
    break;
  }
  case 5: { // vm_idiv_trap (48/49 f7 /7, guaranteed #DE)
    tc.tag = "vm_idiv_trap";
    int src = static_cast<int>(rng.range(0, 16));
    uint64_t mode = rng.next() % 5;
    if (mode == 0 || src == 0 || src == 2) {
      tc.regs[src] = 0ULL;
    } else if (mode <= 2) {
      tc.regs[0] = 0x8000000000000000ULL;
      tc.regs[2] = 0xffffffffffffffffULL;
      tc.regs[src] = 0xffffffffffffffffULL;
    } else if (mode == 3) {
      int64_t mag = static_cast<int64_t>(rng.range(1, 10000));
      int64_t den = (rng.next() % 2 == 0) ? mag : -mag;
      __int128 num = (static_cast<__int128>(1) << 63) * static_cast<__int128>(den);
      unsigned __int128 unum = static_cast<unsigned __int128>(num);
      tc.regs[0] = static_cast<uint64_t>(unum);
      tc.regs[2] = static_cast<uint64_t>(unum >> 64);
      tc.regs[src] = static_cast<uint64_t>(den);
    } else {
      int64_t mag = static_cast<int64_t>(rng.range(1, 10000));
      int64_t den = (rng.next() % 2 == 0) ? mag : -mag;
      __int128 num = (-(static_cast<__int128>(1) << 63) - 1) * static_cast<__int128>(den);
      unsigned __int128 unum = static_cast<unsigned __int128>(num);
      tc.regs[0] = static_cast<uint64_t>(unum);
      tc.regs[2] = static_cast<uint64_t>(unum >> 64);
      tc.regs[src] = static_cast<uint64_t>(den);
    }
    push_rex(tc.ins, true, 0, src);
    tc.ins.push_back(0xf7);
    tc.ins.push_back(static_cast<uint8_t>(0xf8 | (src & 7)));
    break;
  }
  case 6: { // vm_cqo (48 99)
    tc.tag = "vm_cqo";
    tc.ins = {0x48, 0x99};
    break;
  }
  case 7: { // vm_neg (48/49 f7 /3)
    tc.tag = "vm_neg";
    int d = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(0xf7);
    tc.ins.push_back(static_cast<uint8_t>(0xd8 | (d & 7)));
    break;
  }
  case 8: { // vm_not (48/49 f7 /2)
    tc.tag = "vm_not";
    int d = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(0xf7);
    tc.ins.push_back(static_cast<uint8_t>(0xd0 | (d & 7)));
    break;
  }
  case 9: case 10: case 11: { // vm_shlR / vm_shrR / vm_sarR (48/49 d3 /4 /5 /7)
    static constexpr uint8_t exts[3] = {4, 5, 7};
    static constexpr const char *names[3] = {"vm_shlR", "vm_shrR", "vm_sarR"};
    int idx = kind - 9;
    tc.tag = names[idx];
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[1] = rng.shift_count();
    if (idx == 2 && d != 1 && (rng.next() % 2 == 0)) {
      tc.regs[d] |= 0x8000000000000000ULL;
    }
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(0xd3);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (exts[idx] << 3) | (d & 7)));
    break;
  }
  case 12: { // vm_sarI (48/49 c1 /7 imm8)
    tc.tag = "vm_sarI";
    int d = static_cast<int>(rng.range(0, 16));
    if (rng.next() % 2 == 0) tc.regs[d] |= 0x8000000000000000ULL;
    uint8_t cnt = static_cast<uint8_t>(rng.shift_count() & 0xff);
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(0xc1);
    tc.ins.push_back(static_cast<uint8_t>(0xf8 | (d & 7)));
    tc.ins.push_back(cnt);
    break;
  }
  case 13: { // vm_ld16
    tc.tag = "vm_ld16";
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0xdeadbeefcafe0000ULL;
    bool w = (rng.next() % 2 == 0);
    emit_memop(rng, tc.ins, tc, w, d, 0xb7, 2, true);
    break;
  }
  case 14: { // vm_st16
    tc.tag = "vm_st16";
    int s = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, false, s, 0x89, 2, false, false, true);
    break;
  }
  case 15: { // vm_st32
    tc.tag = "vm_st32";
    int s = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, false, s, 0x89, 4, false);
    break;
  }
  case 16: { // vm_ldIdx_w8
    tc.tag = "vm_ldIdx_w8";
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0xfeedface00000000ULL;
    bool w = (rng.next() % 2 == 0);
    emit_memop_idx(rng, tc.ins, tc, w, d, 0xb6, 1, true);
    break;
  }
  case 17: { // vm_ldIdx_w16
    tc.tag = "vm_ldIdx_w16";
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0xfeedface00000000ULL;
    bool w = (rng.next() % 2 == 0);
    emit_memop_idx(rng, tc.ins, tc, w, d, 0xb7, 2, true);
    break;
  }
  case 18: { // vm_ldIdx_w32
    tc.tag = "vm_ldIdx_w32";
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0xfeedface00000000ULL;
    emit_memop_idx(rng, tc.ins, tc, false, d, 0x8b, 4, false);
    break;
  }
  case 19: { // vm_ldIdx_w64
    tc.tag = "vm_ldIdx_w64";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop_idx(rng, tc.ins, tc, true, d, 0x8b, 8, false);
    break;
  }
  case 20: { // vm_stIdx_w8
    tc.tag = "vm_stIdx_w8";
    bool with_rex = (rng.next() % 2 == 0);
    int s = with_rex ? static_cast<int>(rng.range(0, 16)) : static_cast<int>(rng.range(0, 4));
    emit_memop_idx(rng, tc.ins, tc, false, s, 0x88, 1, false, with_rex);
    break;
  }
  case 21: { // vm_stIdx_w16
    tc.tag = "vm_stIdx_w16";
    int s = static_cast<int>(rng.range(0, 16));
    emit_memop_idx(rng, tc.ins, tc, false, s, 0x89, 2, false, false, true);
    break;
  }
  case 22: { // vm_stIdx_w32
    tc.tag = "vm_stIdx_w32";
    int s = static_cast<int>(rng.range(0, 16));
    emit_memop_idx(rng, tc.ins, tc, false, s, 0x89, 4, false);
    break;
  }
  case 23: { // vm_stIdx_w64
    tc.tag = "vm_stIdx_w64";
    int s = static_cast<int>(rng.range(0, 16));
    emit_memop_idx(rng, tc.ins, tc, true, s, 0x89, 8, false);
    break;
  }
  case 24: { // vm_leaIdx
    tc.tag = "vm_leaIdx";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop_idx(rng, tc.ins, tc, true, d, 0x8d, 0, false);
    break;
  }
  case 25: { // vm_testI
    tc.tag = "vm_testI";
    int a = static_cast<int>(rng.range(0, 16));
    uint32_t imm;
    switch (rng.next() % 6) {
    case 0: imm = 0U; break;
    case 1: imm = 1U; break;
    case 2: imm = 0x7fffffffU; break;
    case 3: imm = 0x80000000U; break;
    case 4: imm = 0xffffffffU; break;
    default: imm = rng.u32(); break;
    }
    push_rex(tc.ins, true, 0, a);
    tc.ins.push_back(0xf7);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (a & 7)));
    push_le32(tc.ins, imm);
    break;
  }
  case 26: { // vm_cmpAlI
    tc.tag = "vm_cmpAlI";
    tc.regs[0] = (rng.next() << 8) | rng.u8();
    uint8_t imm = (rng.next() % 4 == 0)
        ? static_cast<uint8_t>((rng.next() % 4) * 0x40 + (rng.next() % 2 == 0 ? 0 : 0x3f))
        : rng.u8();
    tc.ins = {0x3c, imm};
    break;
  }
  case 27: { // vm_testAlI
    tc.tag = "vm_testAlI";
    tc.regs[0] = (rng.next() << 8) | rng.u8();
    tc.ins = {0xa8, rng.u8()};
    break;
  }
  case 28: { // vm_setcc
    tc.tag = "vm_setcc";
    uint8_t cc = pick_cc(rng);
    bool with_rex = (rng.next() % 2 == 0);
    int d = with_rex ? static_cast<int>(rng.range(0, 16)) : static_cast<int>(rng.range(0, 4));
    tc.regs[d] = (rng.next() << 8) | rng.u8();
    push_rex(tc.ins, false, 0, d, with_rex);
    tc.ins.push_back(0x0f);
    tc.ins.push_back(static_cast<uint8_t>(0x90 + cc));
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (d & 7)));
    break;
  }
  case 29: { // vm_cmovcc
    tc.tag = "vm_cmovcc";
    uint8_t cc = pick_cc(rng);
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, true, d, s);
    tc.ins.push_back(0x0f);
    tc.ins.push_back(static_cast<uint8_t>(0x40 + cc));
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((d & 7) << 3) | (s & 7)));
    break;
  }
  default: { // vm_jmpR
    tc.tag = "vm_jmpR";
    int r = static_cast<int>(rng.range(0, 16));
    tc.regs[r] = pick_branch_target(rng, 3, false);
    push_rex(tc.ins, false, 0, r);
    tc.ins.push_back(0xff);
    tc.ins.push_back(static_cast<uint8_t>(0xe0 | (r & 7)));
    break;
  }
  }
}

static void gen_int(Rng &rng, int kind, TestCase &tc) {
  init_common(rng, tc);
  switch (kind) {
  case 0: {
    tc.tag = "int_movI32";
    int d = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, false, 0, d);
    tc.ins.push_back(static_cast<uint8_t>(0xb8 + (d & 7)));
    push_le32(tc.ins, rng.u32());
    break;
  }
  case 1: {
    tc.tag = "int_movI64";
    int d = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(static_cast<uint8_t>(0xb8 + (d & 7)));
    push_le64(tc.ins, rng.rval());
    break;
  }
  case 2: case 3: case 4: case 5: case 6: {
    static constexpr uint8_t ops[5] = {0x01, 0x29, 0x21, 0x09, 0x31};
    static constexpr const char *names[5] = {"int_alu_add", "int_alu_sub", "int_alu_and", "int_alu_or", "int_alu_xor"};
    int idx = kind - 2;
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    if (idx == 4 && (rng.next() % 4 == 0)) {
      s = d;
      tc.tag = "int_alu_xor";
      push_rex(tc.ins, false, s, d);
    } else {
      tc.tag = names[idx];
      push_rex(tc.ins, true, s, d);
    }
    tc.ins.push_back(ops[idx]);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    break;
  }
  case 7: {
    tc.tag = "int_mov32";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    if (rng.next() % 2 == 0) {
      push_rex(tc.ins, false, s, d);
      tc.ins.push_back(0x89);
      tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    } else {
      push_rex(tc.ins, false, d, s);
      tc.ins.push_back(0x8b);
      tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((d & 7) << 3) | (s & 7)));
    }
    break;
  }
  case 8: {
    tc.tag = "int_mov";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    if (rng.next() % 2 == 0) {
      push_rex(tc.ins, true, s, d);
      tc.ins.push_back(0x89);
      tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    } else {
      push_rex(tc.ins, true, d, s);
      tc.ins.push_back(0x8b);
      tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((d & 7) << 3) | (s & 7)));
    }
    break;
  }
  case 9: {
    tc.tag = "int_ld";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, true, d, 0x8b, 8);
    break;
  }
  case 10: {
    tc.tag = "int_st";
    int s = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, true, s, 0x89, 8);
    break;
  }
  case 11: {
    tc.tag = "int_ld32";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, false, d, 0x8b, 4);
    break;
  }
  case 12: {
    tc.tag = "int_test";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, true, s, d);
    tc.ins.push_back(0x85);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    break;
  }
  case 13: {
    tc.tag = "int_pushI";
    tc.regs[4] = WIN_LO + rng.range(8, WIN_SIZE + 1);
    tc.ins = {0x6a, rng.u8()};
    break;
  }
  case 14: {
    tc.tag = "int_cmp";
    int a = static_cast<int>(rng.range(0, 16));
    int b = static_cast<int>(rng.range(0, 16));
    if (rng.next() % 2 == 0) {
      push_rex(tc.ins, true, b, a);
      tc.ins.push_back(0x39);
      tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((b & 7) << 3) | (a & 7)));
    } else {
      push_rex(tc.ins, true, a, b);
      tc.ins.push_back(0x3b);
      tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((a & 7) << 3) | (b & 7)));
    }
    break;
  }
  case 15: case 16: case 17: case 18: case 19: {
    static constexpr uint8_t subs[5] = {0, 1, 4, 5, 6};
    static constexpr const char *names[5] = {
      "int_aluI_add", "int_aluI_or", "int_aluI_and", "int_aluI_sub", "int_aluI_xor"
    };
    int idx = kind - 15;
    tc.tag = names[idx];
    int d = static_cast<int>(rng.range(0, 16));
    bool imm8_form = (rng.next() % 2 == 0);
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(imm8_form ? 0x83 : 0x81);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (subs[idx] << 3) | (d & 7)));
    if (imm8_form) tc.ins.push_back(rng.u8());
    else push_le32(tc.ins, rng.u32());
    break;
  }
  case 20: {
    tc.tag = "int_cmpI";
    int d = static_cast<int>(rng.range(0, 16));
    bool imm8_form = (rng.next() % 2 == 0);
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(imm8_form ? 0x83 : 0x81);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (7 << 3) | (d & 7)));
    if (imm8_form) tc.ins.push_back(rng.u8());
    else push_le32(tc.ins, rng.u32());
    break;
  }
  case 21: case 22: {
    int idx = kind - 21;
    tc.tag = (idx == 0) ? "int_shlI" : "int_shrI";
    int d = static_cast<int>(rng.range(0, 16));
    uint8_t cnt = static_cast<uint8_t>(rng.shift_count() & 0xff);
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(0xc1);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((idx == 0 ? 4 : 5) << 3) | (d & 7)));
    tc.ins.push_back(cnt);
    break;
  }
  case 23: {
    tc.tag = "int_lea";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, true, d, 0x8d, 1);
    break;
  }
  case 24: {
    tc.tag = "int_st8";
    bool with_rex = (rng.next() % 2 == 0);
    int s = with_rex ? static_cast<int>(rng.range(0, 16)) : static_cast<int>(rng.range(0, 4));
    emit_memop(rng, tc.ins, tc, false, s, 0x88, 1, false, with_rex);
    break;
  }
  case 25: {
    tc.tag = "int_ld8";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, false, d, 0xb6, 1, true);
    break;
  }
  case 26: {
    tc.tag = "int_jcc";
    uint8_t cc = pick_cc(rng);
    if (rng.next() % 2 == 0) {
      uint64_t tgt = pick_branch_target(rng, 2, true);
      int8_t rel8 = static_cast<int8_t>(static_cast<int64_t>(tgt) - static_cast<int64_t>(INS_ADDR + 2));
      tc.ins = {static_cast<uint8_t>(0x70 + cc), static_cast<uint8_t>(rel8)};
    } else {
      uint64_t tgt = pick_branch_target(rng, 6, false);
      int32_t rel32 = static_cast<int32_t>(static_cast<int64_t>(tgt) - static_cast<int64_t>(INS_ADDR + 6));
      tc.ins = {0x0f, static_cast<uint8_t>(0x80 + cc)};
      push_le32(tc.ins, static_cast<uint32_t>(rel32));
    }
    break;
  }
  case 27: {
    tc.tag = "int_jmp";
    if (rng.next() % 2 == 0) {
      uint64_t tgt = pick_branch_target(rng, 2, true);
      int8_t rel8 = static_cast<int8_t>(static_cast<int64_t>(tgt) - static_cast<int64_t>(INS_ADDR + 2));
      tc.ins = {0xeb, static_cast<uint8_t>(rel8)};
    } else {
      uint64_t tgt = pick_branch_target(rng, 5, false);
      int32_t rel32 = static_cast<int32_t>(static_cast<int64_t>(tgt) - static_cast<int64_t>(INS_ADDR + 5));
      tc.ins = {0xe9};
      push_le32(tc.ins, static_cast<uint32_t>(rel32));
    }
    break;
  }
  case 28: {
    tc.tag = "int_call";
    tc.regs[4] = WIN_LO + rng.range(8, WIN_SIZE + 1);
    uint64_t tgt = pick_branch_target(rng, 5, false);
    int32_t rel32 = static_cast<int32_t>(static_cast<int64_t>(tgt) - static_cast<int64_t>(INS_ADDR + 5));
    tc.ins = {0xe8};
    push_le32(tc.ins, static_cast<uint32_t>(rel32));
    break;
  }
  case 29: {
    tc.tag = "int_ret";
    uint64_t sp_off = rng.range(0, WIN_SIZE - 8 + 1);
    tc.regs[4] = WIN_LO + sp_off;
    uint64_t tgt = pick_branch_target(rng, 1, false);
    for (int i = 0; i < 8; ++i) tc.mem[sp_off + i] = static_cast<uint8_t>((tgt >> (8 * i)) & 0xff);
    tc.ins = {0xc3};
    break;
  }
  case 30: {
    tc.tag = "int_push";
    int r = static_cast<int>(rng.range(0, 16));
    tc.regs[4] = WIN_LO + rng.range(8, WIN_SIZE + 1);
    push_rex(tc.ins, false, 0, r);
    tc.ins.push_back(static_cast<uint8_t>(0x50 + (r & 7)));
    break;
  }
  case 31: {
    tc.tag = "int_pop";
    int r = static_cast<int>(rng.range(0, 16));
    tc.regs[4] = WIN_LO + rng.range(0, WIN_SIZE - 8 + 1);
    push_rex(tc.ins, false, 0, r);
    tc.ins.push_back(static_cast<uint8_t>(0x58 + (r & 7)));
    break;
  }
  default: {
    tc.tag = "int_jtab";
    int mod = static_cast<int>(rng.range(0, 3));
    int base = static_cast<int>(rng.range(0, 8));
    if (mod == 0 && base == 5) base = 3;
    int idx = static_cast<int>(rng.range(0, 8));
    if (idx == 4 || idx == base) idx = (base == 0 ? 1 : 0);
    if (idx == 4) idx = 2;
    uint64_t mem_off = rng.range(0, WIN_SIZE - 8 + 1);
    uint64_t ptr_addr = WIN_LO + mem_off;
    uint64_t tgt = pick_branch_target(rng, 7, false);
    for (int i = 0; i < 8; ++i) tc.mem[mem_off + i] = static_cast<uint8_t>((tgt >> (8 * i)) & 0xff);
    uint64_t idx_val = rng.range(0, 32);
    tc.regs[idx] = idx_val;
    int64_t disp = 0;
    tc.ins = {0xff, static_cast<uint8_t>((mod << 6) | (4 << 3) | 4), static_cast<uint8_t>(0xc0 | (idx << 3) | base)};
    if (mod == 1) {
      int8_t d8 = static_cast<int8_t>(rng.range(0, 256) - 128);
      tc.ins.push_back(static_cast<uint8_t>(d8));
      disp = d8;
    } else if (mod == 2) {
      int32_t d32 = static_cast<int32_t>(rng.u32());
      push_le32(tc.ins, static_cast<uint32_t>(d32));
      disp = d32;
    }
    tc.regs[base] = ptr_addr - idx_val * 8ULL - static_cast<uint64_t>(disp);
    break;
  }
  }
}

static void run_case(const TestCase &tc) {
  std::memcpy(reinterpret_cast<void *>(WIN_LO), tc.mem, WIN_SIZE);
  uint8_t *code = reinterpret_cast<uint8_t *>(INS_ADDR);
  std::memcpy(code, tc.ins.data(), tc.ins.size());
  for (int i = 0; i < 16; ++i) g_state->regs[i] = tc.regs[i];
  uint64_t rfl = 0x202ULL; // IF | reserved bit 1
  if (tc.cszo & 1) rfl |= (1ULL << 0);
  if (tc.cszo & 2) rfl |= (1ULL << 6);
  if (tc.cszo & 4) rfl |= (1ULL << 7);
  if (tc.cszo & 8) rfl |= (1ULL << 11);
  g_state->rflags = rfl;
  g_trap_kind = 0;

  if (sigsetjmp(g_env, 0) == 0) {
    auto entry = reinterpret_cast<void (*)()>(STUB_PAGE);
    entry();
    std::fprintf(stderr, "FATAL: instruction returned without trapping!\n");
    std::abort();
  }
  if (g_trap_kind == 0) {
    std::fprintf(stderr, "FATAL: siglongjmp without trap kind!\n");
    std::abort();
  }

  std::memset(code, 0xcc, tc.ins.size());
}

static void print_case(std::FILE *out, const TestCase &tc) {
  std::fprintf(out, "%s ", tc.tag.c_str());
  for (uint8_t b : tc.ins) std::fprintf(out, "%02x", b);
  for (int i = 0; i < 16; ++i) std::fprintf(out, " %016llx", static_cast<unsigned long long>(tc.regs[i]));
  std::fprintf(out, " %d%d%d%d ", (tc.cszo >> 0) & 1, (tc.cszo >> 1) & 1, (tc.cszo >> 2) & 1, (tc.cszo >> 3) & 1);
  for (size_t i = 0; i < WIN_SIZE; ++i) std::fprintf(out, "%02x", tc.mem[i]);
  if (g_trap_kind == 2) {
    std::fprintf(out, " | TRAP_DE\n");
    return;
  }
  uint8_t after_mem[WIN_SIZE];
  std::memcpy(after_mem, reinterpret_cast<const void *>(WIN_LO), WIN_SIZE);
  std::fprintf(out, " | %016llx", static_cast<unsigned long long>(g_post_rip));
  for (int i = 0; i < 16; ++i) std::fprintf(out, " %016llx", static_cast<unsigned long long>(g_post_regs[i]));
  int cf = (g_post_rflags >> 0) & 1;
  int zf = (g_post_rflags >> 6) & 1;
  int sf = (g_post_rflags >> 7) & 1;
  int of = (g_post_rflags >> 11) & 1;
  std::fprintf(out, " %d%d%d%d ", cf, zf, sf, of);
  for (size_t i = 0; i < WIN_SIZE; ++i) std::fprintf(out, "%02x", after_mem[i]);
  std::fputc('\n', out);
}

static int hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static bool parse_hex_bytes(const std::string &s, std::vector<uint8_t> &out) {
  if (s.empty() || (s.size() % 2) != 0) return false;
  out.clear();
  out.reserve(s.size() / 2);
  for (size_t i = 0; i < s.size(); i += 2) {
    int hi = hex_nibble(s[i]);
    int lo = hex_nibble(s[i + 1]);
    if (hi < 0 || lo < 0) return false;
    out.push_back(static_cast<uint8_t>((hi << 4) | lo));
  }
  return true;
}

static bool parse_pre_line(const std::string &line, TestCase &tc) {
  std::istringstream iss(line);
  std::string ins_hex, fl_str, mem_hex;
  if (!(iss >> tc.tag >> ins_hex)) return false;
  if (!parse_hex_bytes(ins_hex, tc.ins) || tc.ins.empty() || tc.ins.size() > 15) return false;
  for (int i = 0; i < 16; ++i) {
    std::string rhex;
    if (!(iss >> rhex)) return false;
    tc.regs[i] = std::strtoull(rhex.c_str(), nullptr, 16);
  }
  if (!(iss >> fl_str >> mem_hex)) return false;
  if (fl_str.size() != 4) return false;
  tc.cszo = 0;
  for (int i = 0; i < 4; ++i) {
    if (fl_str[i] == '1') tc.cszo |= static_cast<uint8_t>(1u << i);
    else if (fl_str[i] != '0') return false;
  }
  std::vector<uint8_t> memv;
  if (!parse_hex_bytes(mem_hex, memv) || memv.size() != WIN_SIZE) return false;
  std::memcpy(tc.mem, memv.data(), WIN_SIZE);
  return true;
}

static bool init_sandbox() {
  void *win_p = mmap(reinterpret_cast<void *>(0x11000ULL), 4096, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  void *code_p = mmap(reinterpret_cast<void *>(CODE_PAGE), 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  void *stub_p = mmap(reinterpret_cast<void *>(STUB_PAGE), 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  void *state_p = mmap(reinterpret_cast<void *>(STATE_PAGE), 4096, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (win_p == MAP_FAILED || code_p == MAP_FAILED || stub_p == MAP_FAILED || state_p == MAP_FAILED) {
    std::perror("mmap fixed sandbox");
    return false;
  }
  std::memset(code_p, 0xcc, 4096);
  g_state = static_cast<StateArea *>(state_p);
  emit_stub();

  g_altstack.assign(65536, 0);
  stack_t ss{};
  ss.ss_sp = g_altstack.data();
  ss.ss_size = g_altstack.size();
  ss.ss_flags = 0;
  if (sigaltstack(&ss, nullptr) != 0) {
    std::perror("sigaltstack");
    return false;
  }
  struct sigaction sa{};
  sa.sa_sigaction = trap_handler;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGTRAP, &sa, nullptr) != 0) {
    std::perror("sigaction SIGTRAP");
    return false;
  }
  struct sigaction sa_fpe{};
  sa_fpe.sa_sigaction = fpe_handler;
  sa_fpe.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
  sigemptyset(&sa_fpe.sa_mask);
  if (sigaction(SIGFPE, &sa_fpe, nullptr) != 0) {
    std::perror("sigaction SIGFPE");
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--exec") == 0) {
    if (argc != 4) {
      std::fprintf(stderr, "usage: %s --exec <in-pre.txt> <out-trace.txt>\n", argv[0]);
      return 2;
    }
    if (!init_sandbox()) return 1;
    std::ifstream in(argv[2]);
    if (!in) {
      std::fprintf(stderr, "cannot open %s\n", argv[2]);
      return 1;
    }
    std::FILE *out = std::fopen(argv[3], "w");
    if (!out) {
      std::perror("fopen out");
      return 1;
    }
    std::string line;
    TestCase tc;
    size_t count = 0;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (!parse_pre_line(line, tc)) {
        std::fprintf(stderr, "parse error at line %zu\n", count + 1);
        std::fclose(out);
        return 1;
      }
      run_case(tc);
      print_case(out, tc);
      ++count;
    }
    std::fclose(out);
    return 0;
  }

  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <out-cases.txt> [total=66000] [seed=0x583836564d3236]\n", argv[0]);
    std::fprintf(stderr, "       %s --exec <in-pre.txt> <out-trace.txt>\n", argv[0]);
    return 2;
  }
  const char *out_path = argv[1];
  int total = (argc >= 3) ? std::atoi(argv[2]) : 66000;
  uint64_t seed = (argc >= 4) ? std::strtoull(argv[3], nullptr, 0) : 0x583836564d3236ULL;

  if (!init_sandbox()) return 1;

  std::FILE *out = std::fopen(out_path, "w");
  if (!out) {
    std::perror("fopen out");
    return 1;
  }

  Rng rng(seed);
  // Schedule: 31 new VM forms + 33 X86Int forms = 64 forms total.
  constexpr int kVmKinds = 31;
  constexpr int kIntKinds = 33;
  int per_vm = (total >= 65000) ? 1050 : 1000;
  int vm_quota = kVmKinds * per_vm;
  if (total < vm_quota + kIntKinds * 50) {
    vm_quota = total / 2;
  }

  TestCase tc;
  for (int i = 0; i < total; ++i) {
    if (i < vm_quota) {
      gen_vm(rng, i % kVmKinds, tc);
    } else {
      gen_int(rng, (i - vm_quota) % kIntKinds, tc);
    }
    run_case(tc);
    print_case(out, tc);
  }
  std::fclose(out);
  return 0;
}
