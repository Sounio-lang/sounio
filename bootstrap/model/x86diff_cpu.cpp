// x86diff_cpu: the CPU side of the instruction-level differential test of the
// Lean models formal/Bootstrap/X86Min.lean (default) and
// formal/Bootstrap/X86Int.lean (third argument `int`).
//
// For each case it generates one instruction of a form the model decodes, a
// random register file, random CF ZF SF OF and random stack/memory bytes
// around rsp; it then runs the instruction on this CPU and prints the case and
// the state after it. formal/Bootstrap/X86DiffModel.lean (or
// X86IntDiffModel.lean) reads the same lines and prints what the model says;
// bootstrap/model/x86diff_gate.sh (or x86int_diff_gate.sh) compares them.
//
// Freestanding: no libc, so it runs on images without a C library.
// Build: g++ -std=c++23 -O2 -ffreestanding -nostdlib -static -fno-pie -no-pie
//        -fno-stack-protector -fno-exceptions -fno-rtti -o x86diff_cpu x86diff_cpu.cpp
// Usage: x86diff_cpu <cases> <seed> [int]
//
// Line format, fields separated by one space:
//   instr-hex r0 .. r15 cszo mem | delta r0 .. r15 cszo mem
// r* are 16 hex digits, cszo is CF SF ZF OF as 0/1, mem is 32 bytes at
// rsp-16 .. rsp+15 (default) or 64 bytes at rsp-32 .. rsp+31 (`int` mode) in
// hex, delta is rip after the instruction minus rip before (2 hex digits in
// default mode, 16 hex digits in `int` mode).

typedef unsigned long u64;
typedef long s64;
typedef unsigned int u32;
typedef int s32;
typedef unsigned char u8;
typedef signed char s8;

static long sys(long n, long a, long b, long c, long d, long e, long f) {
  long r;
  register long r10 asm("r10") = d;
  register long r8 asm("r8") = e;
  register long r9 asm("r9") = f;
  asm volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
               : "rcx", "r11", "memory");
  return r;
}
[[noreturn]] static void die(int st) { for (;;) sys(60, st, 0, 0, 0, 0, 0); }

// Fixed addresses: data page(s) and code page, mapped with MAP_FIXED.
static constexpr u64 DATA = 0x10000, CODE = 0x20000;
static constexpr u64 IN = DATA + 0x000, OUT = DATA + 0x080, INFL = DATA + 0x100,
                     OUTFL = DATA + 0x108, HOST = DATA + 0x110, TAG = DATA + 0x150;
static constexpr u64 TMPSTACK = DATA + 0x600, SP = DATA + 0x800;
static constexpr u64 SP_INT = DATA + 0x1800, WIN_SIZE = 64, WIN_LO = SP_INT - 32;
static constexpr u64 BWD_ADDR = CODE + 0x150, INS_ADDR = CODE + 0x180;

static u8 obuf[1 << 16];
static u32 olen = 0;
static void flush() {
  u32 o = 0;
  while (o < olen) { long n = sys(1, 1, (long)(obuf + o), olen - o, 0, 0, 0); if (n <= 0) die(3); o += n; }
  olen = 0;
}
static void put(char c) { obuf[olen++] = (u8)c; if (olen > sizeof obuf - 64) flush(); }
static void puts_(const char* s) { while (*s) put(*s++); }
static void hex(u64 v, int digits) {
  for (int i = digits - 1; i >= 0; --i) put("0123456789abcdef"[(v >> (4 * i)) & 15]);
}

static u64 rs;
static u64 rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static u64 rval() {
  static const u64 sp[] = {0, 1, 2, 0x7f, 0x80, 0xff, 0x100, 0x7fffffff, 0x80000000, 0xffffffff,
                           0x100000000, 0x7fffffffffffffff, 0x8000000000000000, ~0ul, ~0ul - 1};
  u64 k = rnd() % 4;
  if (k == 0) return sp[rnd() % (sizeof sp / sizeof sp[0])];
  if (k == 1) return rnd() % 64;
  return rnd();
}

struct Code { u8 b[1024]; u32 n = 0; void e(u8 x) { b[n++] = x; }
  void e32(u32 x) { for (int i = 0; i < 4; ++i) e((u8)(x >> (8 * i))); } };

// mov r64, [abs32] / mov [abs32], r64: REX.W(+R) 8B|89 /r, modrm 04|reg<<3, SIB 25.
static void ldst(Code& c, bool store, int reg, u32 addr) {
  c.e(0x48 | (reg >= 8 ? 4 : 0)); c.e(store ? 0x89 : 0x8b); c.e(0x04 | ((reg & 7) << 3)); c.e(0x25); c.e32(addr);
}
static const int host_regs[7] = {4, 3, 5, 12, 13, 14, 15};

// Fall-through block: mov byte [TAG], 0 (8 bytes) + jmp rel32 over the taken
// block (5 bytes). Jumps under test use rel = 13 to land on the taken block.
static constexpr u32 FALL = 13;

static u32 build(Code& c, const u8* ins, u32 len) {
  c.n = 0;
  for (int i = 0; i < 7; ++i) ldst(c, true, host_regs[i], HOST + 8 * i);
  c.e(0x48); c.e(0xc7); c.e(0xc4); c.e32(TMPSTACK);           // mov rsp, TMPSTACK
  ldst(c, false, 0, INFL); c.e(0x50); c.e(0x9d);             // mov rax,[INFL]; push rax; popfq
  for (int r = 0; r < 16; ++r) if (r != 4) ldst(c, false, r, IN + 8 * r);
  ldst(c, false, 4, IN + 8 * 4);                             // rsp last
  for (u32 i = 0; i < len; ++i) c.e(ins[i]);
  c.e(0xc6); c.e(0x04); c.e(0x25); c.e32(TAG); c.e(0);        // fall through: tag 0
  c.e(0xe9); c.e32(8);                                       // jmp over the taken block
  c.e(0xc6); c.e(0x04); c.e(0x25); c.e32(TAG); c.e(1);        // taken: tag 1
  for (int r = 0; r < 16; ++r) ldst(c, true, r, OUT + 8 * r);
  c.e(0x48); c.e(0xc7); c.e(0xc4); c.e32(TMPSTACK);
  c.e(0x9c); c.e(0x58); ldst(c, true, 0, OUTFL);             // pushfq; pop rax; mov [OUTFL], rax
  for (int i = 6; i >= 0; --i) ldst(c, false, host_regs[i], HOST + 8 * i);
  c.e(0xc3);
  return c.n;
}

static u32 build_int(Code& c, const u8* ins, u32 len) {
  c.n = 0;
  for (int i = 0; i < 7; ++i) ldst(c, true, host_regs[i], HOST + 8 * i);
  c.e(0x48); c.e(0xc7); c.e(0xc4); c.e32(TMPSTACK);           // mov rsp, TMPSTACK
  ldst(c, false, 0, INFL); c.e(0x50); c.e(0x9d);             // mov rax,[INFL]; push rax; popfq
  for (int r = 0; r < 16; ++r) if (r != 4) ldst(c, false, r, IN + 8 * r);
  ldst(c, false, 4, IN + 8 * 4);                             // rsp last
  u32 jmp_ins = (u32)(INS_ADDR - CODE) - (c.n + 5);
  c.e(0xe9); c.e32(jmp_ins);                                 // jmp INS_ADDR
  if (c.n > (u32)(BWD_ADDR - CODE)) die(7);
  while (c.n < (u32)(BWD_ADDR - CODE)) c.e(0xcc);
  c.e(0xc6); c.e(0x04); c.e(0x25); c.e32(TAG); c.e(2);        // backward target: tag 2
  c.e(0xe9); u32 bwd_jmp_off = c.n; c.e32(0);                // jmp epilogue (patched below)
  if (c.n > (u32)(INS_ADDR - CODE)) die(7);
  while (c.n < (u32)(INS_ADDR - CODE)) c.e(0xcc);
  for (u32 i = 0; i < len; ++i) c.e(ins[i]);
  c.e(0xc6); c.e(0x04); c.e(0x25); c.e32(TAG); c.e(0);        // fall through: tag 0
  c.e(0xe9); c.e32(8);                                       // jmp over forward taken block
  c.e(0xc6); c.e(0x04); c.e(0x25); c.e32(TAG); c.e(1);        // forward taken: tag 1
  u32 bwd_rel = c.n - (bwd_jmp_off + 4);
  for (int i = 0; i < 4; ++i) c.b[bwd_jmp_off + i] = (u8)(bwd_rel >> (8 * i));
  for (int r = 0; r < 16; ++r) ldst(c, true, r, OUT + 8 * r);
  c.e(0x48); c.e(0xc7); c.e(0xc4); c.e32(TMPSTACK);
  c.e(0x9c); c.e(0x58); ldst(c, true, 0, OUTFL);             // pushfq; pop rax; mov [OUTFL], rax
  for (int i = 6; i >= 0; --i) ldst(c, false, host_regs[i], HOST + 8 * i);
  c.e(0xc3);
  return c.n;
}

// One random instruction of a form the model decodes. Returns its length.
static u32 gen(u8* p) {
  u32 n = 0;
  auto rex = [&](bool w, int r, int b) { u8 x = 0x40 | (w ? 8 : 0) | (r >= 8 ? 4 : 0) | (b >= 8 ? 1 : 0);
                                         if (x != 0x40 || rnd() % 4 == 0) p[n++] = x; };
  auto i32 = [&](u32 v) { for (int i = 0; i < 4; ++i) p[n++] = (u8)(v >> (8 * i)); };
  int r = rnd() % 16, b = rnd() % 16;
  bool w = rnd() % 2;
  switch (rnd() % 14) {
  case 0: rex(false, 0, b); p[n++] = 0xb8 + (b & 7); i32((u32)rval()); break;
  case 1: { static const u8 ops[4] = {0x31, 0x09, 0x85, 0x89};
            rex(w, r, b); p[n++] = ops[rnd() % 4]; p[n++] = 0xc0 | ((r & 7) << 3) | (b & 7); break; }
  case 2: if (rnd() % 2) { r = rnd() % 4; } else { rex(false, r, 0); if (n == 0) p[n++] = 0x40; }
          p[n++] = 0x88; p[n++] = 0x04 | ((r & 7) << 3); p[n++] = 0x24; break;
  case 3: p[n++] = 0x6a; p[n++] = (u8)rval(); break;
  case 4: rex(false, 0, b); p[n++] = 0x58 + (b & 7); break;
  case 5: rex(false, r, 0); p[n++] = 0x0f; p[n++] = 0xb6; p[n++] = 0x04 | ((r & 7) << 3); p[n++] = 0x24; break;
  case 6: { u8 cc; do cc = rnd() % 16; while (cc == 10 || cc == 11); p[n++] = 0x70 + cc; p[n++] = FALL; break; }
  case 7: p[n++] = 0xeb; p[n++] = FALL; break;
  case 8: p[n++] = 0xe9; i32(FALL); break;
  case 9: { static const u8 ops[3] = {0x04, 0x3c, 0x2c}; p[n++] = ops[rnd() % 3]; p[n++] = (u8)rval(); break; }
  case 10: rex(false, 0, b); p[n++] = 0xc1; p[n++] = 0xe0 | (b & 7); p[n++] = (u8)(rnd() % 3 ? rnd() % 33 : rnd()); break;
  case 11: rex(w, 0, b); p[n++] = 0x81; p[n++] = 0xf8 | (b & 7); i32((u32)rval()); break;
  case 12: rex(w, 0, b); p[n++] = 0x83; p[n++] = 0xf8 | (b & 7); p[n++] = (u8)rval(); break;
  default: rex(w, 0, b); p[n++] = 0xff; p[n++] = 0xc0 | (b & 7); break;
  }
  return n;
}

static void write_le64(u8* dst, u64 v) {
  for (int i = 0; i < 8; ++i) dst[i] = (u8)(v >> (8 * i));
}

static void emit_mem_op(u8* p, u32& n, int reg_field, int base, int width, u64* in, bool fix_base) {
  int md = (int)(rnd() % 3);
  if (md == 0 && (base & 7) == 5) md = (rnd() % 2) ? 1 : 2;
  bool use_sib = ((base & 7) == 4) || (rnd() % 3 == 0);
  s64 disp = 0;
  if (fix_base) {
    if (base == 4) {
      if (md == 0) {
        disp = 0;
      } else {
        u64 ea = WIN_LO + (rnd() % (WIN_SIZE - (u64)width + 1));
        disp = (s64)ea - (s64)SP_INT;
      }
    } else {
      u64 ea = WIN_LO + (rnd() % (WIN_SIZE - (u64)width + 1));
      if (md == 0) {
        disp = 0;
        in[base] = ea;
      } else if (md == 1) {
        s8 d8 = (rnd() % 2) ? (s8)((s64)(rnd() % 64) - 32) : (s8)rval();
        disp = (s64)d8;
        in[base] = ea - (u64)disp;
      } else {
        s32 d32 = (rnd() % 2) ? (s32)((s64)(rnd() % 256) - 128) : (s32)rval();
        disp = (s64)d32;
        in[base] = ea - (u64)disp;
      }
    }
  } else {
    if (md == 1) disp = (s64)(s8)rval();
    else if (md == 2) disp = (s64)(s32)rval();
  }
  p[n++] = (u8)((md << 6) | ((reg_field & 7) << 3) | (use_sib ? 4 : (base & 7)));
  if (use_sib) p[n++] = (u8)(((rnd() % 4) << 6) | (4 << 3) | (base & 7));
  if (md == 1) p[n++] = (u8)(s8)disp;
  else if (md == 2) {
    u32 u = (u32)(s32)disp;
    for (int i = 0; i < 4; ++i) p[n++] = (u8)(u >> (8 * i));
  }
}

// Generate one random instruction covering every non-syscall form accepted by
// formal/Bootstrap/X86Dec.lean, updating memory-base registers / target words as needed.
static u32 gen_int(u8* p, u64* in, u8* stk) {
  u32 n = 0;
  auto rex = [&](bool w, int r, int b, bool force) {
    u8 x = 0x40 | (w ? 8 : 0) | (r >= 8 ? 4 : 0) | (b >= 8 ? 1 : 0);
    if (x != 0x40 || force || rnd() % 4 == 0) p[n++] = x;
  };
  auto emit32 = [&](u32 v) { for (int i = 0; i < 4; ++i) p[n++] = (u8)(v >> (8 * i)); };
  auto emit64 = [&](u64 v) { for (int i = 0; i < 8; ++i) p[n++] = (u8)(v >> (8 * i)); };
  auto pick_cc = [&]() -> u8 {
    static const u8 ccs[14] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15};
    return ccs[rnd() % 14];
  };
  int r = (int)(rnd() % 16), b = (int)(rnd() % 16);
  switch (rnd() % 45) {
  case 0: rex(false, 0, b, false); p[n++] = (u8)(0xb8 + (b & 7)); emit32((u32)rval()); break;
  case 1: rex(true, 0, b, true); p[n++] = (u8)(0xb8 + (b & 7)); emit64(rval()); break;
  case 2: rex(true, r, b, true); p[n++] = 0x01; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 3: rex(true, r, b, true); p[n++] = 0x29; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 4: rex(true, r, b, true); p[n++] = 0x21; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 5: rex(true, r, b, true); p[n++] = 0x09; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 6: rex(true, r, b, true); p[n++] = 0x31; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 7: rex(false, r, r, false); p[n++] = 0x31; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (r & 7)); break;
  case 8: rex(false, r, b, false); p[n++] = 0x89; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 9: rex(false, r, b, false); p[n++] = 0x8b; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 10: rex(false, r, b, false); p[n++] = 0x8b; emit_mem_op(p, n, r, b, 4, in, true); break;
  case 11: rex(true, r, b, true); p[n++] = 0x89; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 12: rex(true, r, b, true); p[n++] = 0x8b; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 13: rex(true, r, b, true); p[n++] = 0x89; emit_mem_op(p, n, r, b, 8, in, true); break;
  case 14: rex(true, r, b, true); p[n++] = 0x8b; emit_mem_op(p, n, r, b, 8, in, true); break;
  case 15: rex(true, r, b, true); p[n++] = 0x85; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 16: p[n++] = 0x6a; p[n++] = (u8)rval(); break;
  case 17: rex(true, r, b, true); p[n++] = 0x39; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 18: rex(true, r, b, true); p[n++] = 0x3b; p[n++] = (u8)(0xc0 | ((r & 7) << 3) | (b & 7)); break;
  case 19: rex(true, 0, b, true); p[n++] = 0x83; p[n++] = (u8)(0xc0 | (0 << 3) | (b & 7)); p[n++] = (u8)rval(); break;
  case 20: rex(true, 0, b, true); p[n++] = 0x83; p[n++] = (u8)(0xc0 | (1 << 3) | (b & 7)); p[n++] = (u8)rval(); break;
  case 21: rex(true, 0, b, true); p[n++] = 0x83; p[n++] = (u8)(0xc0 | (4 << 3) | (b & 7)); p[n++] = (u8)rval(); break;
  case 22: rex(true, 0, b, true); p[n++] = 0x83; p[n++] = (u8)(0xc0 | (5 << 3) | (b & 7)); p[n++] = (u8)rval(); break;
  case 23: rex(true, 0, b, true); p[n++] = 0x83; p[n++] = (u8)(0xc0 | (6 << 3) | (b & 7)); p[n++] = (u8)rval(); break;
  case 24: rex(true, 0, b, true); p[n++] = 0x83; p[n++] = (u8)(0xf8 | (b & 7)); p[n++] = (u8)rval(); break;
  case 25: rex(true, 0, b, true); p[n++] = 0x81; p[n++] = (u8)(0xc0 | (0 << 3) | (b & 7)); emit32((u32)rval()); break;
  case 26: rex(true, 0, b, true); p[n++] = 0x81; p[n++] = (u8)(0xc0 | (1 << 3) | (b & 7)); emit32((u32)rval()); break;
  case 27: rex(true, 0, b, true); p[n++] = 0x81; p[n++] = (u8)(0xc0 | (4 << 3) | (b & 7)); emit32((u32)rval()); break;
  case 28: rex(true, 0, b, true); p[n++] = 0x81; p[n++] = (u8)(0xc0 | (5 << 3) | (b & 7)); emit32((u32)rval()); break;
  case 29: rex(true, 0, b, true); p[n++] = 0x81; p[n++] = (u8)(0xc0 | (6 << 3) | (b & 7)); emit32((u32)rval()); break;
  case 30: rex(true, 0, b, true); p[n++] = 0x81; p[n++] = (u8)(0xf8 | (b & 7)); emit32((u32)rval()); break;
  case 31: rex(true, 0, b, true); p[n++] = 0xc1; p[n++] = (u8)(0xe0 | (b & 7)); p[n++] = (u8)(rnd() % 3 ? rnd() % 65 : rnd()); break;
  case 32: rex(true, 0, b, true); p[n++] = 0xc1; p[n++] = (u8)(0xe8 | (b & 7)); p[n++] = (u8)(rnd() % 3 ? rnd() % 65 : rnd()); break;
  case 33: rex(true, r, b, true); p[n++] = 0x8d; emit_mem_op(p, n, r, b, 8, in, false); break;
  case 34: rex(rnd() % 4 == 0, r, b, r >= 4); p[n++] = 0x88; emit_mem_op(p, n, r, b, 1, in, true); break;
  case 35: rex(rnd() % 4 == 0, r, b, false); p[n++] = 0x0f; p[n++] = 0xb6; emit_mem_op(p, n, r, b, 1, in, true); break;
  case 36: p[n++] = 0x0f; p[n++] = (u8)(0x80 + pick_cc());
           emit32((rnd() % 2) ? FALL : (u32)(s32)((s64)BWD_ADDR - (s64)(INS_ADDR + 6))); break;
  case 37: p[n++] = (u8)(0x70 + pick_cc());
           p[n++] = (rnd() % 2) ? (u8)FALL : (u8)(s8)((s64)BWD_ADDR - (s64)(INS_ADDR + 2)); break;
  case 38: p[n++] = 0xeb;
           p[n++] = (rnd() % 2) ? (u8)FALL : (u8)(s8)((s64)BWD_ADDR - (s64)(INS_ADDR + 2)); break;
  case 39: p[n++] = 0xe9;
           emit32((rnd() % 2) ? FALL : (u32)(s32)((s64)BWD_ADDR - (s64)(INS_ADDR + 5))); break;
  case 40: p[n++] = 0xe8;
           emit32((rnd() % 2) ? FALL : (u32)(s32)((s64)BWD_ADDR - (s64)(INS_ADDR + 5))); break;
  case 41: p[n++] = 0xc3;
           write_le64(stk + (SP_INT - WIN_LO), (rnd() % 2) ? (INS_ADDR + 1 + FALL) : BWD_ADDR); break;
  case 42: rex(false, 0, b, false); p[n++] = (u8)(0x50 + (b & 7)); break;
  case 43: rex(false, 0, b, false); p[n++] = (u8)(0x58 + (b & 7)); break;
  default: {
    int idx; do { idx = (int)(rnd() % 16); } while (idx == 4);
    int md = (int)(rnd() % 3);
    if (md == 0 && (b & 7) == 5) md = (rnd() % 2) ? 1 : 2;
    u8 rx = (u8)((idx >= 8 ? 2 : 0) | (b >= 8 ? 1 : 0));
    if (rx || rnd() % 4 == 0) p[n++] = (u8)(0x40 | rx);
    p[n++] = 0xff;
    p[n++] = (u8)((md << 6) | (4 << 3) | 4);
    p[n++] = (u8)((3 << 6) | ((idx & 7) << 3) | (b & 7));
    u64 sx_disp = 0, ea = 0;
    if (b == 4) {
      if (md == 0) {
        ea = WIN_LO + 8 * (rnd() % 8);
      } else if (md == 1) {
        ea = WIN_LO + (rnd() % (WIN_SIZE - 8 + 1));
        s64 q = (s64)(rnd() % 9) - 4;
        s64 d = (s64)ea - (s64)SP_INT - 8 * q;
        p[n++] = (u8)(s8)d;
        sx_disp = (u64)d;
      } else {
        ea = WIN_LO + (rnd() % (WIN_SIZE - 8 + 1));
        s64 q = (s64)(s32)(rnd() % 0x10000000) - 0x08000000;
        s64 d = (s64)ea - (s64)SP_INT - 8 * q;
        emit32((u32)(s32)d);
        sx_disp = (u64)d;
      }
      in[idx] = (((ea - SP_INT - sx_disp) >> 3) & ((1ul << 61) - 1)) | ((rnd() & 7) << 61);
    } else {
      ea = WIN_LO + (rnd() % (WIN_SIZE - 8 + 1));
      if (md == 1) {
        s8 d8 = (s8)rval();
        p[n++] = (u8)d8;
        sx_disp = (u64)(s64)d8;
      } else if (md == 2) {
        s32 d32 = (s32)rval();
        emit32((u32)d32);
        sx_disp = (u64)(s64)d32;
      }
      if (b == idx) {
        static constexpr u64 inv9 = 0x8e38e38e38e38e39ul;
        in[b] = (ea - sx_disp) * inv9;
      } else {
        in[b] = ea - 8 * in[idx] - sx_disp;
      }
    }
    u64 target = (rnd() % 2) ? (INS_ADDR + n + FALL) : BWD_ADDR;
    write_le64(stk + (ea - WIN_LO), target);
    break;
  }
  }
  return n;
}

static void state(const u64* regs, u64 fl, const u8* stk, u32 stk_len) {
  for (int i = 0; i < 16; ++i) { put(' '); hex(regs[i], 16); }
  put(' '); put('0' + (fl & 1)); put('0' + ((fl >> 7) & 1)); put('0' + ((fl >> 6) & 1)); put('0' + ((fl >> 11) & 1));
  put(' '); for (u32 i = 0; i < stk_len; ++i) hex(stk[i], 2);
}

static long parse(const char* s) { long v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0'); return v; }
static bool streq(const char* a, const char* b) {
  while (*a && *a == *b) { ++a; ++b; }
  return *a == *b;
}

extern "C" [[noreturn]] void cmain(long argc, char** argv) {
  if (argc < 3 || argc > 4) { puts_("usage: x86diff_cpu <cases> <seed> [int]\n"); flush(); die(2); }
  bool int_mode = false;
  if (argc == 4) {
    if (streq(argv[3], "int")) int_mode = true;
    else { puts_("usage: x86diff_cpu <cases> <seed> [int]\n"); flush(); die(2); }
  }
  long cases = parse(argv[1]);
  rs = 0x9e3779b97f4a7c15ul ^ (u64)parse(argv[2]);
  u64 data_sz = int_mode ? 0x2000 : 0x1000;
  if (sys(9, DATA, (long)data_sz, 3, 0x32, -1, 0) != (long)DATA) die(4); // PROT_READ|WRITE, MAP_PRIVATE|ANON|FIXED
  if (sys(9, CODE, 0x1000, 7, 0x32, -1, 0) != (long)CODE) die(5);        // + PROT_EXEC
  u64* in = (u64*)IN; u64* out = (u64*)OUT;
  if (!int_mode) {
    u8* stk = (u8*)(SP - 16);
    for (long k = 0; k < cases; ++k) {
      u8 ins[16]; u32 len = gen(ins);
      for (int i = 0; i < 16; ++i) in[i] = rval();
      in[4] = SP;
      u64 f = rnd();
      u64 fl = 0x202 | (f & 1) | (f & 0x40) | (f & 0x80) | (f & 0x800);
      *(u64*)INFL = fl;
      u8 before[32];
      for (int i = 0; i < 32; ++i) before[i] = stk[i] = (u8)rnd();
      for (int i = 0; i < (int)len; ++i) hex(ins[i], 2);
      state(in, fl, before, 32);
      Code c; u32 n = build(c, ins, len);
      for (u32 i = 0; i < n; ++i) ((u8*)CODE)[i] = c.b[i];
      *(u8*)TAG = 0xee;
      ((void (*)())CODE)();
      u8 tag = *(u8*)TAG;
      if (tag > 1) die(6);
      puts_(" | "); hex(len + (tag ? FALL : 0), 2);
      state(out, *(u64*)OUTFL, stk, 32);
      put('\n');
    }
  } else {
    u8* stk = (u8*)WIN_LO;
    for (long k = 0; k < cases; ++k) {
      for (int i = 0; i < 16; ++i) in[i] = rval();
      in[4] = SP_INT;
      u64 f = rnd();
      u64 fl = 0x202 | (f & 1) | (f & 0x40) | (f & 0x80) | (f & 0x800);
      *(u64*)INFL = fl;
      for (u32 i = 0; i < WIN_SIZE; ++i) stk[i] = (u8)rnd();
      u8 ins[16]; u32 len = gen_int(ins, in, stk);
      u8 before[WIN_SIZE];
      for (u32 i = 0; i < WIN_SIZE; ++i) before[i] = stk[i];
      for (u32 i = 0; i < len; ++i) hex(ins[i], 2);
      state(in, fl, before, WIN_SIZE);
      Code c; u32 n = build_int(c, ins, len);
      for (u32 i = 0; i < n; ++i) ((u8*)CODE)[i] = c.b[i];
      *(u8*)TAG = 0xee;
      ((void (*)())CODE)();
      u8 tag = *(u8*)TAG;
      u64 rip_after = 0;
      if (tag == 0) rip_after = INS_ADDR + len;
      else if (tag == 1) rip_after = INS_ADDR + len + FALL;
      else if (tag == 2) rip_after = BWD_ADDR;
      else die(6);
      puts_(" | "); hex(rip_after - INS_ADDR, 16);
      state(out, *(u64*)OUTFL, stk, WIN_SIZE);
      put('\n');
    }
  }
  flush();
  die(0);
}

asm(".globl _start\n_start:\n  mov (%rsp), %rdi\n  lea 8(%rsp), %rsi\n  and $-16, %rsp\n  call cmain\n");
