// x86env_diff_cpu: the CPU/kernel side of the process-environment differential
// test of formal/Bootstrap/X86Env.lean and formal/Bootstrap/X86Int.lean.
//
// Build (main runner):
//   g++ -std=c++23 -O2 -Wall -Wextra -o x86env_diff_cpu x86env_diff_cpu.cpp
// Build (static-nostdlib execve stack probe):
//   g++ -std=c++23 -O2 -DINIT_STACK_PROBE -ffreestanding -nostdlib -static
//       -fno-pie -no-pie -fno-stack-protector -fno-exceptions -fno-rtti
//       -o x86env_stack_probe x86env_diff_cpu.cpp

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

typedef unsigned long u64;
typedef long s64;
typedef unsigned int u32;
typedef int s32;
typedef unsigned char u8;

#ifdef INIT_STACK_PROBE

static long raw_sys(long n, long a, long b, long c, long d, long e, long f) {
  long r;
  register long r10 asm("r10") = d;
  register long r8 asm("r8") = e;
  register long r9 asm("r9") = f;
  asm volatile("syscall"
               : "=a"(r)
               : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
               : "rcx", "r11", "memory");
  return r;
}

[[noreturn]] static void raw_die(int code) {
  for (;;) raw_sys(60, code, 0, 0, 0, 0, 0);
}

static char pbuf[16384];
static u32 plen = 0;

static void pput(char c) {
  if (plen < sizeof(pbuf)) pbuf[plen++] = c;
}

static void phex(u64 v, int digits) {
  for (int i = digits - 1; i >= 0; --i)
    pput("0123456789abcdef"[(v >> (4 * i)) & 15]);
}

extern "C" [[noreturn]] void probe_main(u64 sp) {
  u64 align_ok = ((sp & 15ULL) == 0) ? 1ULL : 0ULL;
  u64 sp_mod16 = sp & 15ULL;
  const u64* words = reinterpret_cast<const u64*>(sp);
  u64 argc = words[0];
  u64 argv_null = words[argc + 1];
  u64 envp_null = words[argc + 2];

  const u64* auxv = &words[argc + 3];
  u32 aux_limit = 0;
  while (auxv[0] != 0 && aux_limit < 256) {
    auxv += 2;
    ++aux_limit;
  }
  u64 auxv_null = auxv[0] | auxv[1];

  volatile u64* below_top = reinterpret_cast<volatile u64*>(sp - 8);
  volatile u64* below_bot = reinterpret_cast<volatile u64*>(sp - 4096);
  *below_top = 0x1122334455667788ULL;
  *below_bot = 0x8877665544332211ULL;
  bool room_ok = (*below_top == 0x1122334455667788ULL) &&
                 (*below_bot == 0x8877665544332211ULL);
  *below_top = 0;
  *below_bot = 0;

  pput(align_ok ? '1' : '0');
  pput(' ');
  phex(sp_mod16, 2);
  pput(' ');
  phex(argc, 16);
  pput(' ');
  phex(argv_null, 16);
  pput(' ');
  phex(envp_null, 16);
  pput(' ');
  phex(auxv_null, 16);
  pput(' ');
  pput(room_ok ? '1' : '0');
  pput(' ');

  u64 ptr0 = (argc > 0) ? words[1] : 0;
  for (u64 i = 0; i < argc; ++i) {
    if (i > 0) pput(',');
    u64 ptr_i = words[1 + i];
    u64 rel = ptr_i - ptr0;
    phex(rel, 4);
    pput(':');
    const u8* s = reinterpret_cast<const u8*>(ptr_i);
    for (u32 k = 0; k < 4096 && s[k] != 0; ++k) {
      phex(s[k], 2);
    }
  }
  pput('\n');

  u32 off = 0;
  while (off < plen) {
    long w = raw_sys(1, 1, reinterpret_cast<long>(pbuf + off), plen - off, 0, 0, 0);
    if (w <= 0) raw_die(2);
    off += static_cast<u32>(w);
  }
  raw_die(0);
}

asm(".globl _start\n"
    "_start:\n"
    "  mov %rsp, %rdi\n"
    "  and $-16, %rsp\n"
    "  call probe_main\n");

#else

#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifndef O_PATH
#define O_PATH 010000000
#endif

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

static constexpr u64 PAGE_SZ = 4096;
static constexpr u64 TRAMP_DATA = 0x10000;
static constexpr u64 TRAMP_CODE = 0x20000;
static constexpr u64 IN_REGS = TRAMP_DATA + 0x000;
static constexpr u64 OUT_REGS = TRAMP_DATA + 0x080;
static constexpr u64 IN_FL = TRAMP_DATA + 0x100;
static constexpr u64 OUT_FL = TRAMP_DATA + 0x108;
static constexpr u64 HOST_SAVE = TRAMP_DATA + 0x110;
static constexpr u64 TAG_ADDR = TRAMP_DATA + 0x150;
static constexpr u64 TMP_STACK = TRAMP_DATA + 0x800;
static constexpr u64 INS_ADDR = TRAMP_CODE + 0x180;
static constexpr u64 MEM0_BASE = 0x100000;
static constexpr int CTRL_FD = 500;

static u64 rng_state;
static u64 rnd() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

static u64 rval() {
  static const u64 sp[] = {
      0, 1, 2, 0x7f, 0x80, 0xff, 0x100, 0x7fffffff, 0x80000000, 0xffffffff,
      0x100000000ULL, 0x7fffffffffffffffULL, 0x8000000000000000ULL, ~0ULL, ~0ULL - 1};
  u64 k = rnd() % 4;
  if (k == 0) return sp[rnd() % (sizeof(sp) / sizeof(sp[0]))];
  if (k == 1) return rnd() % 64;
  return rnd();
}

static std::string hex_str(u64 v, int digits) {
  std::string s(digits, '0');
  for (int i = digits - 1; i >= 0; --i) {
    s[digits - 1 - i] = "0123456789abcdef"[(v >> (4 * i)) & 15];
  }
  return s;
}

static std::string bytes_hex(const std::vector<u8>& bs) {
  std::string s;
  s.reserve(bs.size() * 2);
  for (u8 b : bs) s += hex_str(b, 2);
  return s;
}

static std::string fl_str(u64 fl) {
  std::string s(4, '0');
  s[0] = '0' + (fl & 1);
  s[1] = '0' + ((fl >> 7) & 1);
  s[2] = '0' + ((fl >> 6) & 1);
  s[3] = '0' + ((fl >> 11) & 1);
  return s;
}

struct CodeBuf {
  u8 b[1024];
  u32 n = 0;
  void e(u8 x) { b[n++] = x; }
  void e32(u32 x) {
    for (int i = 0; i < 4; ++i) e(static_cast<u8>(x >> (8 * i)));
  }
};

static void ldst(CodeBuf& c, bool store, int reg, u32 addr) {
  c.e(0x48 | (reg >= 8 ? 4 : 0));
  c.e(store ? 0x89 : 0x8b);
  c.e(0x04 | ((reg & 7) << 3));
  c.e(0x25);
  c.e32(addr);
}

static const int host_regs[7] = {4, 3, 5, 12, 13, 14, 15};

static void build_syscall_trampoline(CodeBuf& c) {
  c.n = 0;
  for (int i = 0; i < 7; ++i) ldst(c, true, host_regs[i], HOST_SAVE + 8 * i);
  c.e(0x48);
  c.e(0xc7);
  c.e(0xc4);
  c.e32(TMP_STACK);
  ldst(c, false, 0, IN_FL);
  c.e(0x50);
  c.e(0x9d);
  for (int r = 0; r < 16; ++r)
    if (r != 4) ldst(c, false, r, IN_REGS + 8 * r);
  ldst(c, false, 4, IN_REGS + 8 * 4);
  u32 jmp_ins = static_cast<u32>(INS_ADDR - TRAMP_CODE) - (c.n + 5);
  c.e(0xe9);
  c.e32(jmp_ins);
  while (c.n < static_cast<u32>(INS_ADDR - TRAMP_CODE)) c.e(0xcc);
  // 0f 05 at INS_ADDR
  c.e(0x0f);
  c.e(0x05);
  // Immediately after 0f 05 (at INS_ADDR + 2): mark TAG = 0
  c.e(0xc6);
  c.e(0x04);
  c.e(0x25);
  c.e32(TAG_ADDR);
  c.e(0);
  for (int r = 0; r < 16; ++r) ldst(c, true, r, OUT_REGS + 8 * r);
  c.e(0x48);
  c.e(0xc7);
  c.e(0xc4);
  c.e32(TMP_STACK);
  c.e(0x9c);
  c.e(0x58);
  ldst(c, true, 0, OUT_FL);
  for (int i = 6; i >= 0; --i) ldst(c, false, host_regs[i], HOST_SAVE + 8 * i);
  c.e(0xc3);
}

struct StepRecord {
  std::string cat;
  u64 in_regs[16];
  u64 in_fl;
  u64 probe_addr;
  bool has_choice;
  std::string choice_str;
  bool is_exit;
  u64 exit_code;
  u64 delta;
  u64 out_regs[16];
  u64 out_fl;
  u8 probe_bytes[16];
  std::vector<u8> cum_stdout;
};

struct ChildStepMsg {
  char cat[24];
  u64 in_regs[16];
  u64 in_fl;
  u64 probe_addr;
  u32 choice_kind; // 0 = none, 1 = at, 2 = nomem, 3 = count
  u64 choice_val;
  u32 is_exit;
  u64 delta;
  u64 out_regs[16];
  u64 out_fl;
  u8 probe_bytes[16];
};

static bool write_all(int fd, const void* buf, size_t len) {
  const u8* p = static_cast<const u8*>(buf);
  size_t off = 0;
  while (off < len) {
    ssize_t n = write(fd, p + off, len - off);
    if (n <= 0) return false;
    off += static_cast<size_t>(n);
  }
  return true;
}

static bool read_all(int fd, void* buf, size_t len) {
  u8* p = static_cast<u8*>(buf);
  size_t off = 0;
  while (off < len) {
    ssize_t n = read(fd, p + off, len - off);
    if (n == 0) return false;
    if (n < 0) return false;
    off += static_cast<size_t>(n);
  }
  return true;
}

static void fill_rand_regs(u64* regs, u64& fl) {
  for (int i = 0; i < 16; ++i) regs[i] = rval();
  u64 f = rnd();
  fl = 0x202ULL | (f & 1ULL) | (f & 0x40ULL) | (f & 0x80ULL) | (f & 0x800ULL);
}

static void exec_step_in_child(ChildStepMsg& msg, bool force_enomem = false,
                               u64 blocker_addr = 0, u64 unmap_after_addr = 0) {
  u64* in = reinterpret_cast<u64*>(IN_REGS);
  u64* out = reinterpret_cast<u64*>(OUT_REGS);
  for (int i = 0; i < 16; ++i) in[i] = msg.in_regs[i];
  *reinterpret_cast<u64*>(IN_FL) = msg.in_fl;
  *reinterpret_cast<u8*>(TAG_ADDR) = 0xee;

  void* blk = MAP_FAILED;
  if (blocker_addr != 0) {
    blk = mmap(reinterpret_cast<void*>(blocker_addr), PAGE_SZ, PROT_NONE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  }

  struct rlimit old_as {};
  if (force_enomem) {
    getrlimit(RLIMIT_AS, &old_as);
    struct rlimit zero_as = {0, old_as.rlim_max};
    setrlimit(RLIMIT_AS, &zero_as);
  }

  if (msg.is_exit) {
    if (!write_all(CTRL_FD, &msg, sizeof(msg))) _exit(99);
  }

  reinterpret_cast<void (*)()>(TRAMP_CODE)();

  if (force_enomem) {
    setrlimit(RLIMIT_AS, &old_as);
  }
  if (blk != MAP_FAILED && blk == reinterpret_cast<void*>(blocker_addr)) {
    munmap(blk, PAGE_SZ);
  }
  if (unmap_after_addr != 0) {
    munmap(reinterpret_cast<void*>(unmap_after_addr), PAGE_SZ);
  }

  if (*reinterpret_cast<u8*>(TAG_ADDR) != 0) _exit(98);
  msg.delta = 2;
  for (int i = 0; i < 16; ++i) msg.out_regs[i] = out[i];
  msg.out_fl = *reinterpret_cast<u64*>(OUT_FL);
}

static void run_syscall_scenario(long scen_idx, std::ostream& out_stream) {
  int kind = static_cast<int>(scen_idx % 4);

  std::vector<std::pair<std::string, std::vector<u8>>> files;
  std::vector<u8> stdin_chunk1, stdin_chunk2;
  std::vector<u8> m0_prefix(64, 0);

  // Layout inside 0x100000..0x100040:
  // 0x00: "f0\0"
  // 0x08: "f1\0"
  // 0x10: "missing\0"
  // 0x20..0x3f: random initial scratch bytes
  m0_prefix[0] = 'f';
  m0_prefix[1] = '0';
  m0_prefix[2] = 0;
  m0_prefix[8] = 'f';
  m0_prefix[9] = '1';
  m0_prefix[10] = 0;
  const char* miss = "missing";
  for (int i = 0; miss[i]; ++i) m0_prefix[16 + i] = static_cast<u8>(miss[i]);
  for (int i = 32; i < 64; ++i) m0_prefix[i] = static_cast<u8>(1 + (rnd() % 254));

  if (kind == 0) {
    u32 len0 = 12 + (rnd() % 9);
    u32 len1 = 12 + (rnd() % 9);
    std::vector<u8> d0(len0), d1(len1);
    for (u32 i = 0; i < len0; ++i) d0[i] = static_cast<u8>(1 + (rnd() % 254));
    for (u32 i = 0; i < len1; ++i) d1[i] = static_cast<u8>(1 + (rnd() % 254));
    files.push_back({"f0", d0});
    files.push_back({"f1", d1});
  } else if (kind == 1) {
    u32 k1 = 4 + (rnd() % 7);
    u32 k2 = 4 + (rnd() % 7);
    stdin_chunk1.resize(k1);
    stdin_chunk2.resize(k2);
    for (u32 i = 0; i < k1; ++i) stdin_chunk1[i] = static_cast<u8>(1 + (rnd() % 254));
    for (u32 i = 0; i < k2; ++i) stdin_chunk2[i] = static_cast<u8>(1 + (rnd() % 254));
  } else if (kind == 2) {
    std::vector<u8> d0(24);
    for (u32 i = 0; i < 24; ++i) d0[i] = static_cast<u8>(1 + (rnd() % 254));
    files.push_back({"f0", d0});
  } else {
    u32 len0 = 8 + (rnd() % 8);
    std::vector<u8> d0(len0);
    for (u32 i = 0; i < len0; ++i) d0[i] = static_cast<u8>(1 + (rnd() % 254));
    files.push_back({"f0", d0});
  }

  // Write scenario files to current temp working directory
  unlink("f0");
  unlink("f1");
  unlink("missing");
  for (const auto& kv : files) {
    int fd = open(kv.first.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || !write_all(fd, kv.second.data(), kv.second.size())) {
      std::cerr << "failed to create temp file\n";
      std::exit(2);
    }
    close(fd);
  }

  // Pre-generate random register seeds for up to 12 steps
  u64 step_regs[12][16];
  u64 step_fl[12];
  for (int s = 0; s < 12; ++s) fill_rand_regs(step_regs[s], step_fl[s]);
  u32 bad_fd1 = (rnd() % 2 == 0) ? 2 : static_cast<u32>(5 + (rnd() % 5));
  u32 bad_fd2 = (rnd() % 2 == 0) ? 2 : static_cast<u32>(5 + (rnd() % 5));
  u64 mmap_len1 = (rnd() % 4 == 0) ? PAGE_SZ : (1 + (rnd() % PAGE_SZ));
  u64 mmap_len2 = PAGE_SZ + 1 + (rnd() % PAGE_SZ); // 2 pages
  u64 mmap_len3 = 2 * PAGE_SZ + 1 + (rnd() % PAGE_SZ); // 3 pages
  if (scen_idx % 20 == 0) {
    // Periodically exercise larger multi-page mappings up to 32768
    mmap_len3 = 7 * PAGE_SZ + 1 + (rnd() % PAGE_SZ); // 8 pages = 32768
  }
  u64 enomem_len = 1 + (rnd() % 32768);

  int in_pipe[2], out_pipe[2], ctrl_pipe[2], sync_pipe[2];
  if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0 || pipe(ctrl_pipe) != 0 || pipe(sync_pipe) != 0) {
    std::cerr << "pipe failed\n";
    std::exit(2);
  }

  pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "fork failed\n";
    std::exit(2);
  }

  if (pid == 0) {
    // Child process
    close(in_pipe[1]);
    close(out_pipe[0]);
    close(ctrl_pipe[0]);
    close(sync_pipe[1]);

    if (dup2(in_pipe[0], 0) < 0) _exit(90);
    if (dup2(out_pipe[1], 1) < 0) _exit(90);
    if (dup2(ctrl_pipe[1], CTRL_FD) < 0) _exit(90);
    int sync_rfd = 501;
    if (dup2(sync_pipe[0], sync_rfd) < 0) _exit(90);

    // Replace fd 2 with O_PATH so slot 2 stays occupied while read(2)/write(2) return -EBADF
    int path_fd = open("/dev/null", O_PATH);
    if (path_fd < 0 || dup2(path_fd, 2) < 0) _exit(91);
    if (path_fd > 2) close(path_fd);

    for (int fd = 3; fd < 1024; ++fd) {
      if (fd != CTRL_FD && fd != sync_rfd) close(fd);
    }

    if (mmap(reinterpret_cast<void*>(TRAMP_DATA), PAGE_SZ, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) !=
        reinterpret_cast<void*>(TRAMP_DATA))
      _exit(92);
    if (mmap(reinterpret_cast<void*>(TRAMP_CODE), PAGE_SZ,
             PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) !=
        reinterpret_cast<void*>(TRAMP_CODE))
      _exit(93);

    CodeBuf cb;
    build_syscall_trampoline(cb);
    std::memcpy(reinterpret_cast<void*>(TRAMP_CODE), cb.b, cb.n);

    if (mmap(reinterpret_cast<void*>(MEM0_BASE), PAGE_SZ, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) !=
        reinterpret_cast<void*>(MEM0_BASE))
      _exit(94);
    std::memset(reinterpret_cast<void*>(MEM0_BASE), 0, PAGE_SZ);
    std::memcpy(reinterpret_cast<void*>(MEM0_BASE), m0_prefix.data(), m0_prefix.size());

    auto send_step = [&](const char* cat, int s_idx, u64 nr, u64 a1, u64 a2, u64 a3,
                         u64 a4, u64 a5, u64 a6, u64 probe_addr, u32 c_kind,
                         bool is_exit = false, bool force_enomem = false,
                         u64 blocker_addr = 0, u64 unmap_after_addr = 0) -> u64 {
      ChildStepMsg msg{};
      std::strncpy(msg.cat, cat, sizeof(msg.cat) - 1);
      for (int i = 0; i < 16; ++i) msg.in_regs[i] = step_regs[s_idx][i];
      msg.in_regs[0] = nr;
      msg.in_regs[7] = a1;
      msg.in_regs[6] = a2;
      msg.in_regs[2] = a3;
      msg.in_regs[10] = a4;
      msg.in_regs[8] = a5;
      msg.in_regs[9] = a6;
      msg.in_fl = step_fl[s_idx];
      msg.probe_addr = probe_addr;
      msg.choice_kind = c_kind;
      msg.is_exit = is_exit ? 1 : 0;

      exec_step_in_child(msg, force_enomem, blocker_addr, unmap_after_addr);

      u64 rax = msg.out_regs[0];
      if (c_kind == 1 || c_kind == 3) {
        msg.choice_val = rax;
      }
      if (msg.probe_addr == 1) {
        msg.probe_addr = rax;
      }
      if (msg.probe_addr != 0) {
        std::memcpy(msg.probe_bytes, reinterpret_cast<const void*>(msg.probe_addr), 16);
      }
      if (!write_all(CTRL_FD, &msg, sizeof(msg))) _exit(95);
      return rax;
    };

    if (kind == 0) {
      u32 rlen = 12;
      send_step("open_ok", 0, 2, MEM0_BASE + 0, 0, step_regs[0][2], step_regs[0][10],
                step_regs[0][8], step_regs[0][9], MEM0_BASE, 0);
      send_step("open_ok", 1, 2, MEM0_BASE + 8, 0, step_regs[1][2], step_regs[1][10],
                step_regs[1][8], step_regs[1][9], MEM0_BASE, 0);
      send_step("close_ok", 2, 3, 3, step_regs[2][6], step_regs[2][2], step_regs[2][10],
                step_regs[2][8], step_regs[2][9], MEM0_BASE, 0);
      send_step("open_ok", 3, 2, MEM0_BASE + 0, 0, step_regs[3][2], step_regs[3][10],
                step_regs[3][8], step_regs[3][9], MEM0_BASE, 0);
      send_step("read_full", 4, 0, 3, MEM0_BASE + 32, rlen, step_regs[4][10],
                step_regs[4][8], step_regs[4][9], MEM0_BASE + 32, 3);
      send_step("write_ok", 5, 1, 1, MEM0_BASE + 32, rlen, step_regs[5][10],
                step_regs[5][8], step_regs[5][9], MEM0_BASE + 32, 3);
      send_step("exit", 6, 60, step_regs[6][7], step_regs[6][6], step_regs[6][2],
                step_regs[6][10], step_regs[6][8], step_regs[6][9], 0, 0, true);
    } else if (kind == 1) {
      u8 sync_byte = 0;
      u32 k1 = static_cast<u32>(stdin_chunk1.size());
      u32 k2 = static_cast<u32>(stdin_chunk2.size());
      if (!read_all(sync_rfd, &sync_byte, 1)) _exit(96);
      send_step("read_short", 0, 0, 0, MEM0_BASE + 32, k1 + k2, step_regs[0][10],
                step_regs[0][8], step_regs[0][9], MEM0_BASE + 32, 3);
      if (!read_all(sync_rfd, &sync_byte, 1)) _exit(96);
      send_step("read_full", 1, 0, 0, MEM0_BASE + 32 + k1, k2, step_regs[1][10],
                step_regs[1][8], step_regs[1][9], MEM0_BASE + 32 + k1, 3);
      send_step("read_eof", 2, 0, 0, MEM0_BASE + 32, 12, step_regs[2][10],
                step_regs[2][8], step_regs[2][9], MEM0_BASE + 32, 3);
      send_step("read_ebadf", 3, 0, bad_fd1, MEM0_BASE + 32, 8, step_regs[3][10],
                step_regs[3][8], step_regs[3][9], MEM0_BASE + 32, 0);
      send_step("write_ok", 4, 1, 1, MEM0_BASE + 32, k1 + k2, step_regs[4][10],
                step_regs[4][8], step_regs[4][9], MEM0_BASE + 32, 3);
      send_step("write_ebadf", 5, 1, bad_fd2, MEM0_BASE + 32, 8, step_regs[5][10],
                step_regs[5][8], step_regs[5][9], MEM0_BASE + 32, 0);
      send_step("exit", 6, 60, step_regs[6][7], step_regs[6][6], step_regs[6][2],
                step_regs[6][10], step_regs[6][8], step_regs[6][9], 0, 0, true);
    } else if (kind == 2) {
      send_step("open_ok", 0, 2, MEM0_BASE + 0, 0, step_regs[0][2], step_regs[0][10],
                step_regs[0][8], step_regs[0][9], MEM0_BASE, 0);
      // Map a 1-page placeholder at the top-down frontier so Step 1's 1-page mmap
      // lands immediately below it, then unmap the placeholder so Step 3's 2-page
      // mremap expands in-place into that freed page.
      void* blocker0 = mmap(nullptr, PAGE_SZ, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (blocker0 == MAP_FAILED) _exit(97);
      u64 blocker0_addr = reinterpret_cast<u64>(blocker0);

      u64 addr1 = send_step("mmap_ok", 1, 9, 0, mmap_len1, 3, 0x22, ~0ULL, 0,
                            1, 1, false, false, 0, blocker0_addr);
      send_step("read_full", 2, 0, 3, addr1 + PAGE_SZ - 8, 8, step_regs[2][10],
                step_regs[2][8], step_regs[2][9], addr1 + PAGE_SZ - 16, 3);
      u64 addr2 = send_step("mremap_inplace", 3, 25, addr1, mmap_len1, mmap_len2, 1,
                            step_regs[3][8], step_regs[3][9], addr1 + PAGE_SZ - 8, 1);
      // For mremap_moved, we pass probe_addr = 0 in send_step and read the probe after
      // learning the relocated address addr3.
      ChildStepMsg msg{};
      std::strncpy(msg.cat, "mremap_moved", sizeof(msg.cat) - 1);
      for (int i = 0; i < 16; ++i) msg.in_regs[i] = step_regs[4][i];
      msg.in_regs[0] = 25;
      msg.in_regs[7] = addr2;
      msg.in_regs[6] = mmap_len2;
      msg.in_regs[2] = mmap_len3;
      msg.in_regs[10] = 1;
      msg.in_fl = step_fl[4];
      msg.choice_kind = 1;
      msg.is_exit = 0;
      exec_step_in_child(msg, false, addr2 + 2 * PAGE_SZ, 0);
      u64 addr3 = msg.out_regs[0];
      msg.choice_val = addr3;
      msg.probe_addr = addr3 + PAGE_SZ - 8;
      std::memcpy(msg.probe_bytes, reinterpret_cast<const void*>(msg.probe_addr), 16);
      if (!write_all(CTRL_FD, &msg, sizeof(msg))) _exit(95);

      send_step("munmap_ok", 5, 11, addr3, mmap_len3, step_regs[5][2], step_regs[5][10],
                step_regs[5][8], step_regs[5][9], MEM0_BASE, 0);
      send_step("exit", 6, 60, step_regs[6][7], step_regs[6][6], step_regs[6][2],
                step_regs[6][10], step_regs[6][8], step_regs[6][9], 0, 0, true);
    } else {
      u32 flen = static_cast<u32>(files[0].second.size());
      send_step("open_enoent", 0, 2, MEM0_BASE + 16, 0, step_regs[0][2], step_regs[0][10],
                step_regs[0][8], step_regs[0][9], MEM0_BASE + 16, 0);
      send_step("close_ebadf", 1, 3, 4, step_regs[1][6], step_regs[1][2], step_regs[1][10],
                step_regs[1][8], step_regs[1][9], MEM0_BASE, 0);
      send_step("mmap_einval", 2, 9, 0, 0, 3, 0x22, ~0ULL, 0, MEM0_BASE, 0);
      send_step("mmap_enomem", 3, 9, 0, enomem_len, 3, 0x22, ~0ULL, 0, MEM0_BASE, 2,
                false, true);
      send_step("open_ok", 4, 2, MEM0_BASE + 0, 0, step_regs[4][2], step_regs[4][10],
                step_regs[4][8], step_regs[4][9], MEM0_BASE, 0);
      send_step("read_full", 5, 0, 3, MEM0_BASE + 32, flen, step_regs[5][10],
                step_regs[5][8], step_regs[5][9], MEM0_BASE + 32, 3);
      send_step("read_eof", 6, 0, 3, MEM0_BASE + 32, 12, step_regs[6][10],
                step_regs[6][8], step_regs[6][9], MEM0_BASE + 32, 3);
      send_step("close_ok", 7, 3, 3, step_regs[7][6], step_regs[7][2], step_regs[7][10],
                step_regs[7][8], step_regs[7][9], MEM0_BASE, 0);
      send_step("close_ebadf", 8, 3, 3, step_regs[8][6], step_regs[8][2], step_regs[8][10],
                step_regs[8][8], step_regs[8][9], MEM0_BASE, 0);
      send_step("exit", 9, 60, step_regs[9][7], step_regs[9][6], step_regs[9][2],
                step_regs[9][10], step_regs[9][8], step_regs[9][9], 0, 0, true);
    }
    _exit(0);
  }

  // Parent process
  close(in_pipe[0]);
  close(out_pipe[1]);
  close(ctrl_pipe[1]);
  close(sync_pipe[0]);

  if (kind == 1) {
    // Write chunk 1 before Step 0, then wait for Step 0 report before writing chunk 2
    if (!write_all(in_pipe[1], stdin_chunk1.data(), stdin_chunk1.size())) std::exit(3);
    u8 sync_b = 1;
    if (!write_all(sync_pipe[1], &sync_b, 1)) std::exit(3);
  } else {
    close(in_pipe[1]);
    close(sync_pipe[1]);
  }

  std::vector<StepRecord> records;
  std::vector<u8> cum_stdout;
  ChildStepMsg msg{};
  int step_i = 0;
  while (read_all(ctrl_pipe[0], &msg, sizeof(msg))) {
    StepRecord rec{};
    rec.cat = msg.cat;
    for (int i = 0; i < 16; ++i) rec.in_regs[i] = msg.in_regs[i];
    rec.in_fl = msg.in_fl;
    rec.probe_addr = msg.probe_addr;
    rec.has_choice = (msg.choice_kind != 0);
    if (msg.choice_kind == 1) {
      rec.choice_str = "at:" + hex_str(msg.choice_val, 16);
    } else if (msg.choice_kind == 2) {
      rec.choice_str = "nomem";
    } else if (msg.choice_kind == 3) {
      rec.choice_str = "cnt:" + hex_str(msg.choice_val, 16);
    }
    rec.is_exit = (msg.is_exit != 0);
    rec.delta = msg.delta;
    for (int i = 0; i < 16; ++i) rec.out_regs[i] = msg.out_regs[i];
    rec.out_fl = msg.out_fl;
    std::memcpy(rec.probe_bytes, msg.probe_bytes, 16);

    if (rec.cat == "write_ok" && !rec.is_exit) {
      u64 expected_w = rec.out_regs[0];
      std::vector<u8> wbuf(expected_w);
      if (!read_all(out_pipe[0], wbuf.data(), expected_w)) std::exit(4);
      cum_stdout.insert(cum_stdout.end(), wbuf.begin(), wbuf.end());
    }
    rec.cum_stdout = cum_stdout;
    records.push_back(rec);

    if (kind == 1 && step_i == 0) {
      if (!write_all(in_pipe[1], stdin_chunk2.data(), stdin_chunk2.size())) std::exit(3);
      close(in_pipe[1]);
      u8 sync_b = 2;
      if (!write_all(sync_pipe[1], &sync_b, 1)) std::exit(3);
      close(sync_pipe[1]);
    }
    ++step_i;
  }
  close(ctrl_pipe[0]);
  close(out_pipe[0]);

  int wstatus = 0;
  if (waitpid(pid, &wstatus, 0) < 0 || !WIFEXITED(wstatus)) {
    std::cerr << "child did not exit cleanly\n";
    std::exit(5);
  }
  if (records.empty() || !records.back().is_exit) {
    std::cerr << "missing exit step record\n";
    std::exit(6);
  }
  records.back().exit_code = static_cast<u64>(WEXITSTATUS(wstatus));

  // Format SCEN line
  out_stream << "SCEN ";
  if (files.empty()) {
    out_stream << "-";
  } else {
    for (size_t i = 0; i < files.size(); ++i) {
      if (i > 0) out_stream << ",";
      std::vector<u8> nb(files[i].first.begin(), files[i].first.end());
      out_stream << bytes_hex(nb) << ":" << bytes_hex(files[i].second);
    }
  }
  out_stream << " ";

  std::vector<u8> full_stdin = stdin_chunk1;
  full_stdin.insert(full_stdin.end(), stdin_chunk2.begin(), stdin_chunk2.end());
  if (full_stdin.empty()) {
    out_stream << "-";
  } else {
    out_stream << bytes_hex(full_stdin);
  }
  out_stream << " " << bytes_hex(m0_prefix) << " ";

  std::vector<std::string> choices;
  for (const auto& r : records) {
    if (r.has_choice) choices.push_back(r.choice_str);
  }
  if (choices.empty()) {
    out_stream << "-";
  } else {
    for (size_t i = 0; i < choices.size(); ++i) {
      if (i > 0) out_stream << ",";
      out_stream << choices[i];
    }
  }
  out_stream << " ";

  for (size_t i = 0; i < records.size(); ++i) {
    if (i > 0) out_stream << ";";
    const auto& r = records[i];
    out_stream << r.cat << ":0f05:" << hex_str(INS_ADDR, 16);
    for (int reg = 0; reg < 16; ++reg) out_stream << ":" << hex_str(r.in_regs[reg], 16);
    out_stream << ":" << fl_str(r.in_fl) << ":" << hex_str(r.probe_addr, 16);
  }

  out_stream << " | ";

  for (size_t i = 0; i < records.size(); ++i) {
    if (i > 0) out_stream << ";";
    const auto& r = records[i];
    std::string out_s = r.cum_stdout.empty() ? "-" : bytes_hex(r.cum_stdout);
    if (r.is_exit) {
      out_stream << "EXIT_" << hex_str(r.exit_code, 2) << " " << out_s;
    } else {
      out_stream << hex_str(r.delta, 16);
      for (int reg = 0; reg < 16; ++reg) out_stream << " " << hex_str(r.out_regs[reg], 16);
      out_stream << " " << fl_str(r.out_fl) << " ";
      for (int b = 0; b < 16; ++b) out_stream << hex_str(r.probe_bytes[b], 2);
      out_stream << " " << out_s;
    }
  }
  out_stream << "\n";
}

static void run_init_stack_case(const char* probe_path, std::ostream& out_stream) {
  u32 argc = 1 + (rnd() % 8); // 1..8 argv strings
  std::vector<std::string> args(argc);
  for (u32 i = 0; i < argc; ++i) {
    u32 slen = 1 + (rnd() % 16);
    std::string s(slen, 'a');
    for (u32 j = 0; j < slen; ++j) {
      s[j] = static_cast<char>(33 + (rnd() % 90));
    }
    args[i] = s;
  }

  u64 top = 0x7fffffffe000ULL - ((rnd() % 256) * PAGE_SZ);

  int pfd[2];
  if (pipe(pfd) != 0) std::exit(2);

  pid_t pid = fork();
  if (pid < 0) std::exit(2);
  if (pid == 0) {
    close(pfd[0]);
    if (dup2(pfd[1], 1) < 0) _exit(90);
    close(pfd[1]);
    std::vector<char*> c_argv(argc + 1, nullptr);
    for (u32 i = 0; i < argc; ++i) c_argv[i] = const_cast<char*>(args[i].c_str());
    char* const c_envp[] = {nullptr};
    execve(probe_path, c_argv.data(), c_envp);
    _exit(127);
  }
  close(pfd[1]);
  std::string probe_out;
  char buf[512];
  ssize_t n = 0;
  while ((n = read(pfd[0], buf, sizeof(buf))) > 0) {
    probe_out.append(buf, static_cast<size_t>(n));
  }
  close(pfd[0]);

  int wstatus = 0;
  if (waitpid(pid, &wstatus, 0) < 0 || !WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
    std::cerr << "stack probe failed\n";
    std::exit(7);
  }
  while (!probe_out.empty() && (probe_out.back() == '\n' || probe_out.back() == '\r'))
    probe_out.pop_back();

  out_stream << "STACK " << hex_str(top, 16) << " ";
  for (u32 i = 0; i < argc; ++i) {
    if (i > 0) out_stream << ",";
    std::vector<u8> sb(args[i].begin(), args[i].end());
    out_stream << bytes_hex(sb);
  }
  out_stream << " | " << probe_out << "\n";
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: x86env_diff_cpu <scenarios> <seed> <stack_probe_bin>\n";
    return 2;
  }
  long scenarios = std::strtol(argv[1], nullptr, 10);
  u64 seed = std::strtoull(argv[2], nullptr, 10);
  const char* probe_path = argv[3];
  rng_state = 0x9e3779b97f4a7c15ULL ^ seed;

  char tmp_template[] = "/tmp/sounio-env-cpu-XXXXXX";
  char* tmp_dir = mkdtemp(tmp_template);
  if (!tmp_dir) {
    std::cerr << "mkdtemp failed\n";
    return 2;
  }
  char orig_cwd[4096];
  if (!getcwd(orig_cwd, sizeof(orig_cwd)) || chdir(tmp_dir) != 0) {
    std::cerr << "chdir failed\n";
    return 2;
  }

  std::ios::sync_with_stdio(false);
  for (long i = 0; i < scenarios; ++i) {
    run_syscall_scenario(i, std::cout);
  }

  long stack_cases = (scenarios >= 2000) ? 250 : ((scenarios >= 16) ? (scenarios / 4) : 4);
  for (long i = 0; i < stack_cases; ++i) {
    run_init_stack_case(probe_path, std::cout);
  }
  std::cout.flush();

  unlink("f0");
  unlink("f1");
  unlink("missing");
  if (chdir(orig_cwd) == 0) {
    rmdir(tmp_dir);
  }
  return 0;
}

#endif
