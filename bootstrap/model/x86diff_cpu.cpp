// x86diff_cpu: the CPU side of the instruction-level differential test of the
// Lean model formal/Bootstrap/X86Min.lean.
//
// For each case it generates one instruction of a form the model decodes, a
// random register file, random CF ZF SF OF and 32 random stack bytes around
// rsp; it then runs the instruction on this CPU and prints the case and the
// state after it. formal/Bootstrap/X86DiffModel.lean reads the same lines and
// prints what the model says; bootstrap/model/x86diff_gate.sh compares them.
//
// Freestanding: no libc, so it runs on images without a C library.
// Build: g++ -std=c++23 -O2 -ffreestanding -nostdlib -static -fno-pie -no-pie
//        -fno-stack-protector -fno-exceptions -fno-rtti -o x86diff_cpu x86diff_cpu.cpp
// Usage: x86diff_cpu <cases> <seed>
//
// Line format, fields separated by one space:
//   instr-hex r0 .. r15 cszo stack | delta r0 .. r15 cszo stack
// r* are 16 hex digits, cszo is CF SF ZF OF as 0/1, stack is the 32 bytes at
// rsp-16 .. rsp+15 in hex, delta is rip after the instruction minus rip before.

typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned char u8;

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

// Fixed addresses: data page and code page, mapped with MAP_FIXED.
static constexpr u64 DATA = 0x10000, CODE = 0x20000;
static constexpr u64 IN = DATA + 0x000, OUT = DATA + 0x080, INFL = DATA + 0x100,
                     OUTFL = DATA + 0x108, HOST = DATA + 0x110, TAG = DATA + 0x150;
static constexpr u64 TMPSTACK = DATA + 0x600, SP = DATA + 0x800;

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

struct Code { u8 b[512]; u32 n = 0; void e(u8 x) { b[n++] = x; }
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

static void state(const u64* regs, u64 fl, const u8* stk) {
  for (int i = 0; i < 16; ++i) { put(' '); hex(regs[i], 16); }
  put(' '); put('0' + (fl & 1)); put('0' + ((fl >> 7) & 1)); put('0' + ((fl >> 6) & 1)); put('0' + ((fl >> 11) & 1));
  put(' '); for (int i = 0; i < 32; ++i) hex(stk[i], 2);
}

static long parse(const char* s) { long v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0'); return v; }

extern "C" [[noreturn]] void cmain(long argc, char** argv) {
  if (argc < 3) { puts_("usage: x86diff_cpu <cases> <seed>\n"); flush(); die(2); }
  long cases = parse(argv[1]);
  rs = 0x9e3779b97f4a7c15ul ^ (u64)parse(argv[2]);
  if (sys(9, DATA, 0x1000, 3, 0x32, -1, 0) != (long)DATA) die(4);   // PROT_READ|WRITE, MAP_PRIVATE|ANON|FIXED
  if (sys(9, CODE, 0x1000, 7, 0x32, -1, 0) != (long)CODE) die(5);   // + PROT_EXEC
  u64* in = (u64*)IN; u64* out = (u64*)OUT;
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
    state(in, fl, before);
    Code c; u32 n = build(c, ins, len);
    for (u32 i = 0; i < n; ++i) ((u8*)CODE)[i] = c.b[i];
    *(u8*)TAG = 0xee;
    ((void (*)())CODE)();
    u8 tag = *(u8*)TAG;
    if (tag > 1) die(6);
    puts_(" | "); hex(len + (tag ? FALL : 0), 2);
    state(out, *(u64*)OUTFL, stk);
    put('\n');
  }
  flush();
  die(0);
}

asm(".globl _start\n_start:\n  mov (%rsp), %rdi\n  lea 8(%rsp), %rsi\n  and $-16, %rsp\n  call cmain\n");
