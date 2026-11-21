// Real x86-64 hardware differential oracle for X86Uni: executes randomized single
// instructions across all X86Dec 64/32-bit forms and all 5 X86Uni narrow forms
// inside an isolated mapped sandbox at INS_ADDR=0x20180 / WIN_LO=0x117e0 and
// emits machine traces verified by formal/Bootstrap/X86UniDiffModel.lean.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
#include <signal.h>
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
static volatile sig_atomic_t g_trapped = 0;
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
  g_trapped = 1;
  siglongjmp(g_env, 1);
}
#else
#error "x86uni_diff_cpu.cpp must be compiled and executed on x86-64 Linux"
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
  // mov rsp, <addr of g_state->rflags> = STATE_PAGE + 128; popfq
  // 48 bc imm64 (mov rsp, STATE_PAGE + 128)
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
    switch (next() % 8) {
    case 0: return 0ULL;
    case 1: return 1ULL;
    case 2: return 0xffffffffULL;
    case 3: return 0x80000000ULL;
    case 4: return 0x7fffffffULL;
    case 5: return 0xffffffffffffffffULL;
    case 6: return (next() << 32) | 0ULL;
    default: return next();
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

static void emit_memop(Rng &rng, std::vector<uint8_t> &ins, TestCase &tc,
                       bool w, int reg_field, uint8_t opcode, int access_bytes,
                       bool two_byte_0f = false, bool force_rex = false) {
  // Choose base != 4, 12 (avoid SIB), != 5, 13 when mod==0 (avoid RIP-rel)
  int mod = static_cast<int>(rng.range(0, 3));
  int base;
  do {
    base = static_cast<int>(rng.range(0, 16));
  } while ((base & 7) == 4 || (mod == 0 && (base & 7) == 5));
  if (two_byte_0f) {
    push_rex(ins, w, reg_field, base, force_rex);
    ins.push_back(0x0f);
    ins.push_back(opcode);
  } else {
    push_rex(ins, w, reg_field, base, force_rex);
    ins.push_back(opcode);
  }
  ins.push_back(static_cast<uint8_t>((mod << 6) | ((reg_field & 7) << 3) | (base & 7)));
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

static void gen_narrow(Rng &rng, int kind, TestCase &tc) {
  init_common(rng, tc);
  switch (kind) {
  case 0: { // alImm8_add
    tc.tag = "narrow_alImm8_add";
    uint8_t imm = rng.u8();
    // Ensure high 56 bits of rax are often nonzero!
    tc.regs[0] = (rng.next() << 8) | rng.u8();
    tc.ins = {0x04, imm};
    break;
  }
  case 1: { // alImm8_sub
    tc.tag = "narrow_alImm8_sub";
    uint8_t imm = rng.u8();
    tc.regs[0] = (rng.next() << 8) | rng.u8();
    tc.ins = {0x2c, imm};
    break;
  }
  case 2: { // alImm8_cmp
    tc.tag = "narrow_alImm8_cmp";
    uint8_t imm = rng.u8();
    tc.regs[0] = (rng.next() << 8) | rng.u8();
    tc.ins = {0x3c, imm};
    break;
  }
  case 3: { // rr32_xor (d != s, upper 32 bits nonzero)
    tc.tag = "narrow_rr32_xor";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    if (d == s) s = (d + 1) % 16;
    tc.regs[d] |= 0xdeadbeef00000000ULL;
    push_rex(tc.ins, false, s, d);
    tc.ins.push_back(0x31);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    break;
  }
  case 4: { // rr32_or
    tc.tag = "narrow_rr32_or";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0xcafebabe00000000ULL;
    push_rex(tc.ins, false, s, d);
    tc.ins.push_back(0x09);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    break;
  }
  case 5: { // rr32_test
    tc.tag = "narrow_rr32_test";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0x1234567800000000ULL;
    tc.regs[s] |= 0x8765432100000000ULL;
    push_rex(tc.ins, false, s, d);
    tc.ins.push_back(0x85);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    break;
  }
  case 6: { // cmpI32_81
    tc.tag = "narrow_cmpI32_81";
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0xabcdef0100000000ULL;
    push_rex(tc.ins, false, 7, d);
    tc.ins.push_back(0x81);
    tc.ins.push_back(static_cast<uint8_t>(0xf8 | (d & 7)));
    push_le32(tc.ins, rng.u32());
    break;
  }
  case 7: { // cmpI32_83
    tc.tag = "narrow_cmpI32_83";
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0xfedcba9800000000ULL;
    push_rex(tc.ins, false, 7, d);
    tc.ins.push_back(0x83);
    tc.ins.push_back(static_cast<uint8_t>(0xf8 | (d & 7)));
    tc.ins.push_back(rng.u8());
    break;
  }
  case 8: { // shl32_cnt0 (both 0 and multiples of 32 up to 224!)
    tc.tag = "narrow_shl32_cnt0";
    int d = static_cast<int>(rng.range(0, 16));
    // Guarantee bits 32..63 are nonzero so zero-extension is caught if broken!
    tc.regs[d] = (rng.next() | 0x0001000000000000ULL);
    uint8_t imm8 = static_cast<uint8_t>((rng.next() % 8) * 32);
    push_rex(tc.ins, false, 4, d);
    tc.ins.push_back(0xc1);
    tc.ins.push_back(static_cast<uint8_t>(0xe0 | (d & 7)));
    tc.ins.push_back(imm8);
    break;
  }
  case 9: { // shl32_cnt1 (1 or 1+32*k)
    tc.tag = "narrow_shl32_cnt1";
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0x1122334400000000ULL;
    uint8_t imm8 = static_cast<uint8_t>(1 + (rng.next() % 8) * 32);
    push_rex(tc.ins, false, 4, d);
    tc.ins.push_back(0xc1);
    tc.ins.push_back(static_cast<uint8_t>(0xe0 | (d & 7)));
    tc.ins.push_back(imm8);
    break;
  }
  case 10: { // shl32_cntgt1 (2..31 mod 32)
    tc.tag = "narrow_shl32_cntgt1";
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0x5566778800000000ULL;
    uint8_t imm8 = static_cast<uint8_t>(rng.range(2, 32) + (rng.next() % 8) * 32);
    push_rex(tc.ins, false, 4, d);
    tc.ins.push_back(0xc1);
    tc.ins.push_back(static_cast<uint8_t>(0xe0 | (d & 7)));
    tc.ins.push_back(imm8);
    break;
  }
  case 11: { // inc32 (preserves CF, zero-extends to 64 bits)
    tc.tag = "narrow_inc32";
    int d = static_cast<int>(rng.range(0, 16));
    if (rng.next() % 4 == 0) tc.regs[d] = (rng.next() << 32) | 0xffffffffULL;
    else if (rng.next() % 4 == 1) tc.regs[d] = (rng.next() << 32) | 0x7fffffffULL;
    else tc.regs[d] |= 0x89abcdef00000000ULL;
    push_rex(tc.ins, false, 0, d);
    tc.ins.push_back(0xff);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (d & 7)));
    break;
  }
  case 13:   // rr32_add (01 /r)
  case 14: { // rr32_sub (29 /r)
    bool add = kind == 13;
    tc.tag = add ? "narrow_rr32_add" : "narrow_rr32_sub";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0x5a5a5a5a00000000ULL;
    tc.regs[s] |= 0xa5a5a5a500000000ULL;
    push_rex(tc.ins, false, s, d);
    tc.ins.push_back(add ? 0x01 : 0x29);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    break;
  }
  case 15: case 16: case 17: case 18: { // aluI32 add/sub: 81 /0, 83 /0, 81 /5, 83 /5
    bool add = kind <= 16;
    bool imm32 = kind == 15 || kind == 17;
    tc.tag = add ? (imm32 ? "narrow_aluI32_add_81" : "narrow_aluI32_add_83")
                 : (imm32 ? "narrow_aluI32_sub_81" : "narrow_aluI32_sub_83");
    int d = static_cast<int>(rng.range(0, 16));
    tc.regs[d] |= 0x0123456700000000ULL;
    push_rex(tc.ins, false, add ? 0 : 5, d);
    tc.ins.push_back(imm32 ? 0x81 : 0x83);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((add ? 0 : 5) << 3) | (d & 7)));
    if (imm32) push_le32(tc.ins, rng.u32()); else tc.ins.push_back(rng.u8());
    break;
  }
  case 19: { // jmp r64 (ff /4, mod 11); REX.W is ignored by the CPU
    tc.tag = "int_jmpR";
    int r = static_cast<int>(rng.range(0, 16));
    if (r == 4) r = 5; // the harness needs rsp
    bool w = rng.next() % 2 == 0;
    push_rex(tc.ins, w, 4, r, w);
    tc.ins.push_back(0xff);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (4 << 3) | (r & 7)));
    tc.regs[r] = pick_branch_target(rng, tc.ins.size(), false);
    break;
  }
  default: { // inc64
    tc.tag = "narrow_inc64";
    int d = static_cast<int>(rng.range(0, 16));
    if (rng.next() % 4 == 0) tc.regs[d] = 0xffffffffffffffffULL;
    else if (rng.next() % 4 == 1) tc.regs[d] = 0x7fffffffffffffffULL;
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(0xff);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (d & 7)));
    break;
  }
  }
}

static void gen_int(Rng &rng, int kind, TestCase &tc) {
  init_common(rng, tc);
  switch (kind) {
  case 0: { // movI32
    tc.tag = "int_movI32";
    int d = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, false, 0, d);
    tc.ins.push_back(static_cast<uint8_t>(0xb8 + (d & 7)));
    push_le32(tc.ins, rng.u32());
    break;
  }
  case 1: { // movI64
    tc.tag = "int_movI64";
    int d = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(static_cast<uint8_t>(0xb8 + (d & 7)));
    push_le64(tc.ins, rng.rval());
    break;
  }
  case 2: case 3: case 4: case 5: case 6: { // alu 64-bit
    static constexpr uint8_t ops[5] = {0x01, 0x29, 0x21, 0x09, 0x31};
    static constexpr const char *names[5] = {"int_alu_add", "int_alu_sub", "int_alu_and", "int_alu_or", "int_alu_xor"};
    int idx = kind - 2;
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    if (idx == 4 && (rng.next() % 4 == 0)) {
      // also exercise 32-bit xor zeroing d==s via X86Dec
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
  case 7: { // mov32 (89/8b)
    tc.tag = "int_mov32";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    bool dir = (rng.next() % 2 == 0);
    if (dir) { // 89: r/m <- reg
      push_rex(tc.ins, false, s, d);
      tc.ins.push_back(0x89);
      tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    } else { // 8b: reg <- r/m
      push_rex(tc.ins, false, d, s);
      tc.ins.push_back(0x8b);
      tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((d & 7) << 3) | (s & 7)));
    }
    break;
  }
  case 8: { // mov64 (89/8b)
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
  case 9: { // ld64
    tc.tag = "int_ld";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, true, d, 0x8b, 8);
    break;
  }
  case 10: { // st64
    tc.tag = "int_st";
    int s = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, true, s, 0x89, 8);
    break;
  }
  case 11: { // ld32
    tc.tag = "int_ld32";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, false, d, 0x8b, 4);
    break;
  }
  case 12: { // test64
    tc.tag = "int_test";
    int d = static_cast<int>(rng.range(0, 16));
    int s = static_cast<int>(rng.range(0, 16));
    push_rex(tc.ins, true, s, d);
    tc.ins.push_back(0x85);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | ((s & 7) << 3) | (d & 7)));
    break;
  }
  case 13: { // pushI
    tc.tag = "int_pushI";
    tc.regs[4] = WIN_LO + rng.range(8, WIN_SIZE + 1);
    tc.ins = {0x6a, rng.u8()};
    break;
  }
  case 14: { // cmp64 (39/3b)
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
  case 15: case 16: case 17: { // aluI64 (add=0, or=1, and=4, sub=5, xor=6) via 81/83
    static constexpr uint8_t subs[5] = {0, 1, 4, 5, 6};
    static constexpr const char *names[5] = {
      "int_aluI_add", "int_aluI_or", "int_aluI_and", "int_aluI_sub", "int_aluI_xor"
    };
    int idx = static_cast<int>((kind - 15 + 3 * (rng.next() % 2)) % 5);
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
  case 18: { // cmpI64 (81/83 /7)
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
  case 19: case 20: case 21: { // shl/shr 64-bit (c1 /4, /5)
    static constexpr uint8_t subs[2] = {4, 5};
    static constexpr const char *names[2] = {"int_shlI", "int_shrI"};
    int idx = (kind == 19) ? 0 : ((kind == 20) ? 1 : static_cast<int>(rng.next() % 2));
    tc.tag = names[idx];
    int d = static_cast<int>(rng.range(0, 16));
    uint8_t cnt;
    uint64_t r = rng.next() % 6;
    if (r == 0) cnt = 0;
    else if (r == 1) cnt = 64;
    else if (r == 2) cnt = 1;
    else if (r == 3) cnt = 65;
    else cnt = static_cast<uint8_t>(rng.range(2, 64));
    push_rex(tc.ins, true, 0, d);
    tc.ins.push_back(0xc1);
    tc.ins.push_back(static_cast<uint8_t>(0xc0 | (subs[idx] << 3) | (d & 7)));
    tc.ins.push_back(cnt);
    break;
  }
  case 22: { // lea64
    tc.tag = "int_lea";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, true, d, 0x8d, 1);
    break;
  }
  case 23: { // st8
    tc.tag = "int_st8";
    bool with_rex = (rng.next() % 2 == 0);
    int s = with_rex ? static_cast<int>(rng.range(0, 16)) : static_cast<int>(rng.range(0, 4));
    emit_memop(rng, tc.ins, tc, false, s, 0x88, 1, false, with_rex);
    break;
  }
  case 24: { // ld8 (0f b6)
    tc.tag = "int_ld8";
    int d = static_cast<int>(rng.range(0, 16));
    emit_memop(rng, tc.ins, tc, false, d, 0xb6, 1, true);
    break;
  }
  case 25: { // jcc (rel8 & rel32)
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
  case 26: { // jmp (eb rel8 & e9 rel32)
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
  case 27: { // call (e8 rel32)
    tc.tag = "int_call";
    tc.regs[4] = WIN_LO + rng.range(8, WIN_SIZE + 1);
    uint64_t tgt = pick_branch_target(rng, 5, false);
    int32_t rel32 = static_cast<int32_t>(static_cast<int64_t>(tgt) - static_cast<int64_t>(INS_ADDR + 5));
    tc.ins = {0xe8};
    push_le32(tc.ins, static_cast<uint32_t>(rel32));
    break;
  }
  case 28: { // ret (c3)
    tc.tag = "int_ret";
    uint64_t sp_off = rng.range(0, WIN_SIZE - 8 + 1);
    tc.regs[4] = WIN_LO + sp_off;
    uint64_t tgt = pick_branch_target(rng, 1, false);
    for (int i = 0; i < 8; ++i) tc.mem[sp_off + i] = static_cast<uint8_t>((tgt >> (8 * i)) & 0xff);
    tc.ins = {0xc3};
    break;
  }
  case 29: { // push (50..57)
    tc.tag = "int_push";
    int r = static_cast<int>(rng.range(0, 16));
    tc.regs[4] = WIN_LO + rng.range(8, WIN_SIZE + 1);
    push_rex(tc.ins, false, 0, r);
    tc.ins.push_back(static_cast<uint8_t>(0x50 + (r & 7)));
    break;
  }
  case 30: { // pop (58..5f)
    tc.tag = "int_pop";
    int r = static_cast<int>(rng.range(0, 16));
    tc.regs[4] = WIN_LO + rng.range(0, WIN_SIZE - 8 + 1);
    push_rex(tc.ins, false, 0, r);
    tc.ins.push_back(static_cast<uint8_t>(0x58 + (r & 7)));
    break;
  }
  default: { // jtab (ff /4 SIB scale=1)
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
  g_trapped = 0;

  if (sigsetjmp(g_env, 0) == 0) {
    auto entry = reinterpret_cast<void (*)()>(STUB_PAGE);
    entry();
    std::fprintf(stderr, "FATAL: instruction returned without trapping!\n");
    std::abort();
  }
  if (!g_trapped) {
    std::fprintf(stderr, "FATAL: siglongjmp without trap flag!\n");
    std::abort();
  }

  std::memset(code, 0xcc, tc.ins.size());
}

static void print_case(std::FILE *out, const TestCase &tc) {
  uint8_t after_mem[WIN_SIZE];
  std::memcpy(after_mem, reinterpret_cast<const void *>(WIN_LO), WIN_SIZE);

  std::fprintf(out, "%s ", tc.tag.c_str());
  for (uint8_t b : tc.ins) std::fprintf(out, "%02x", b);
  for (int i = 0; i < 16; ++i) std::fprintf(out, " %016llx", static_cast<unsigned long long>(tc.regs[i]));
  std::fprintf(out, " %d%d%d%d ", (tc.cszo >> 0) & 1, (tc.cszo >> 1) & 1, (tc.cszo >> 2) & 1, (tc.cszo >> 3) & 1);
  for (size_t i = 0; i < WIN_SIZE; ++i) std::fprintf(out, "%02x", tc.mem[i]);
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

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <out-cases.txt> [total=60000] [seed=0x554e4932303236]\n", argv[0]);
    return 2;
  }
  const char *out_path = argv[1];
  int total = (argc >= 3) ? std::atoi(argv[2]) : 60000;
  uint64_t seed = (argc >= 4) ? std::strtoull(argv[3], nullptr, 0) : 0x554e4932303236ULL;

  // Map fixed scratch pages: stack window, trap-filled code page, stub page, state page.
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
    return 1;
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
    return 1;
  }
  struct sigaction sa{};
  sa.sa_sigaction = trap_handler;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGTRAP, &sa, nullptr) != 0) {
    std::perror("sigaction SIGTRAP");
    return 1;
  }

  std::FILE *out = std::fopen(out_path, "w");
  if (!out) {
    std::perror("fopen out");
    return 1;
  }

  Rng rng(seed);
  // Schedule: guarantee >= 1200 hits per narrow form (20 * 1200 = 24000) + uniform over 32 int forms.
  constexpr int kNarrowKinds = 20;
  constexpr int kIntKinds = 32;
  constexpr int kPerNarrow = 1200;
  int narrow_quota = kNarrowKinds * kPerNarrow;
  if (total < narrow_quota + kIntKinds * 50) {
    narrow_quota = total / 2;
  }

  TestCase tc;
  for (int i = 0; i < total; ++i) {
    if (i < narrow_quota) {
      gen_narrow(rng, i % kNarrowKinds, tc);
    } else {
      gen_int(rng, (i - narrow_quota) % kIntKinds, tc);
    }
    run_case(tc);
    print_case(out, tc);
  }
  std::fclose(out);
  return 0;
}
