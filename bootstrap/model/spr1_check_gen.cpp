// Generator of the spr1_check listing (format A), ELF64 binary, and Lean data.
//
// spr1_check reads a Sporos bytecode file (formal/Sporos/Bytecode.lean) on stdin
// and exits 0 iff it decodes to a program that passes Sporos.checkProg
// (formal/Sporos/Check.lean); 5 on a malformed or ill-typed file, 6 when memory
// is refused or a stack height reaches 2^48, 3 when the three passes disagree.
//
// Note: Root A is MUSE's hand-written checker; spr1_check is an independent
// second check, not Root A or Root A's trust base.
//
// The listing bootstrap/model/spr1_check.hex is the source of record,
// assembled by bootstrap/roots/a/seed-a. Every instruction emitted here is
// re-decoded by a transcription of Bootstrap/X86Dec.decode and compared with
// the instruction that was meant.
//
//   spr1_check_gen [OUTDIR]      writes spr1_check.hex, spr1_check,
//                                SporosCheckBinData.lean, symbols.txt
//                                (default OUTDIR: .)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

[[noreturn]] void fail_msg(std::string_view msg) {
  std::fprintf(stderr, "spr1_check_gen: %.*s\n", static_cast<int>(msg.size()), msg.data());
  std::abort();
}

void require(bool cond, std::string_view msg) {
  if (!cond) {
    fail_msg(msg);
  }
}

constexpr u64 kAccept = 0x5acce970a55a3c96ULL;
constexpr u64 kReject = 0xdeadbeef11110001ULL;
constexpr u64 kOverflow = 0x0bad0f1e33c3c3a5ULL;
constexpr int kHmaxShift = 48;
constexpr int kPad0 = 480;
constexpr int kPad1 = 5300;
constexpr u64 kNegAcc = 0ULL - kAccept;
constexpr u64 kNegOvf = 0ULL - kOverflow;

constexpr u64 kBase = 0x400000ULL;
constexpr u64 kBss = 0x600000ULL;
constexpr std::array<u64, 3> kDesc = {kBss, kBss + 0x1000ULL, kBss + 0x2000ULL};
constexpr u64 kBssSize = 0x3000ULL;
constexpr std::array<int, 3> kRes = {0, 3, 1};  // pass k writes rax, rbx, rcx
constexpr u64 kCap0 = 0x10000ULL;

[[maybe_unused]] constexpr int RAX = 0, RCX = 1, RDX = 2, RBX = 3, RSP = 4, RBP = 5, RSI = 6, RDI = 7;
[[maybe_unused]] constexpr int R8 = 8, R9 = 9, R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15;

constexpr std::array<const char*, 16> kRegName = {
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
};

enum class AluOp : int { Add = 0, Or = 1, And = 2, Sub = 3, Xor = 4 };

const char* alu_name(AluOp op) {
  switch (op) {
    case AluOp::Add: return "add";
    case AluOp::Or: return "or";
    case AluOp::And: return "and";
    case AluOp::Sub: return "sub";
    case AluOp::Xor: return "xor";
  }
  fail_msg("invalid AluOp");
}

u8 alu_rr_opcode(AluOp op) {
  switch (op) {
    case AluOp::Add: return 0x01;
    case AluOp::Sub: return 0x29;
    case AluOp::And: return 0x21;
    case AluOp::Or: return 0x09;
    case AluOp::Xor: return 0x31;
  }
  fail_msg("invalid AluOp");
}

int alu_ext(AluOp op) {
  switch (op) {
    case AluOp::Add: return 0;
    case AluOp::Or: return 1;
    case AluOp::And: return 4;
    case AluOp::Sub: return 5;
    case AluOp::Xor: return 6;
  }
  fail_msg("invalid AluOp");
}

std::optional<AluOp> alu_of_rr(u8 o) {
  switch (o) {
    case 0x01: return AluOp::Add;
    case 0x29: return AluOp::Sub;
    case 0x21: return AluOp::And;
    case 0x09: return AluOp::Or;
    case 0x31: return AluOp::Xor;
    default: return std::nullopt;
  }
}

std::optional<AluOp> alu_of_ext(int e) {
  switch (e) {
    case 0: return AluOp::Add;
    case 1: return AluOp::Or;
    case 4: return AluOp::And;
    case 5: return AluOp::Sub;
    case 6: return AluOp::Xor;
    default: return std::nullopt;
  }
}

enum class Cc : int {
  O = 0, NO = 1, B = 2, AE = 3, E = 4, NE = 5, BE = 6, A = 7,
  S = 8, NS = 9, L = 12, GE = 13, LE = 14, G = 15,
};

const char* cc_name(int cc) {
  switch (cc) {
    case 0: return "o";
    case 1: return "no";
    case 2: return "b";
    case 3: return "ae";
    case 4: return "e";
    case 5: return "ne";
    case 6: return "be";
    case 7: return "a";
    case 8: return "s";
    case 9: return "ns";
    case 12: return "l";
    case 13: return "ge";
    case 14: return "le";
    case 15: return "g";
    default: fail_msg("unknown cc");
  }
}

std::string raw_hex(u64 v, int min_digits = 1) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string s;
  do {
    s.push_back(kHex[v & 0xfULL]);
    v >>= 4;
  } while (v != 0 || static_cast<int>(s.size()) < min_digits);
  std::reverse(s.begin(), s.end());
  return s;
}

std::string hex_0x(u64 v) { return "0x" + raw_hex(v); }

std::string signed_hex(u64 v) {
  i64 s = static_cast<i64>(v);
  if (s < 0) {
    u64 mag = 0ULL - static_cast<u64>(s);
    return "-0x" + raw_hex(mag);
  }
  return "+0x" + raw_hex(static_cast<u64>(s));
}

std::string hex_bytes(const std::vector<u8>& bs) {
  std::string out;
  for (std::size_t i = 0; i < bs.size(); ++i) {
    if (i > 0) out.push_back(' ');
    out += raw_hex(bs[i], 2);
  }
  return out;
}

std::string pad_right(std::string s, std::size_t width) {
  if (s.size() < width) s.append(width - s.size(), ' ');
  return s;
}

void push_le(std::vector<u8>& out, u64 v, int n) {
  for (int i = 0; i < n; ++i) {
    out.push_back(static_cast<u8>((v >> (8 * i)) & 0xffULL));
  }
}

u8 rex(int w, int r, int x, int b) {
  return static_cast<u8>(0x40 | (w << 3) | (r << 2) | (x << 1) | b);
}

void append_memop(std::vector<u8>& out, int reg, int base, i32 disp) {
  int rm = base & 7;
  int md = 0;
  if (disp == 0 && rm != 5) {
    md = 0;
  } else if (disp >= -128 && disp < 128) {
    md = 1;
  } else {
    md = 2;
  }
  out.push_back(static_cast<u8>((md << 6) | ((reg & 7) << 3) | rm));
  if (rm == 4) {
    out.push_back(0x24);
  }
  if (md == 1) {
    push_le(out, static_cast<u64>(static_cast<u8>(disp)), 1);
  } else if (md == 2) {
    push_le(out, static_cast<u64>(static_cast<u32>(disp)), 4);
  }
}

// ---------------------------------------------------------------- X86Int.Ins

struct Ins {
  enum class Kind {
    MovI, Mov, Mov32, Alu, AluI, CmpI, Cmp, Test, ShlI, ShrI,
    Ld, Ld32, Ld8, St, St8, Lea, Syscall, Jmp, Jcc,
    PushI, Push, Pop, Call, Ret, Jtab,
    Pad0, Pad1, Voter,
  };

  Kind kind = Kind::Syscall;
  AluOp alu_op = AluOp::Add;
  int r0 = 0;
  int r1 = 0;
  u64 imm = 0;  // disp / target / imm / cc / count

  bool operator==(const Ins& o) const = default;
};

struct Encoded {
  std::vector<u8> bytes;
  Ins ins;
};

// ---------------------------------------------------------------- encoders

Encoded e_movI(int d, u64 imm) {
  std::vector<u8> b;
  if (imm < (1ULL << 32)) {
    if (d >= 8) b.push_back(rex(0, 0, 0, 1));
    b.push_back(static_cast<u8>(0xb8 + (d & 7)));
    push_le(b, imm, 4);
  } else {
    b.push_back(rex(1, 0, 0, d >> 3));
    b.push_back(static_cast<u8>(0xb8 + (d & 7)));
    push_le(b, imm, 8);
  }
  return {std::move(b), Ins{.kind = Ins::Kind::MovI, .r0 = d, .imm = imm}};
}

Encoded e_movI64(int d, u64 imm) {
  std::vector<u8> b = {rex(1, 0, 0, d >> 3), static_cast<u8>(0xb8 + (d & 7))};
  push_le(b, imm, 8);
  return {std::move(b), Ins{.kind = Ins::Kind::MovI, .r0 = d, .imm = imm}};
}

Encoded e_mov(int d, int s) {
  // 8b /r with mod 11: a flipped mod bit makes it a load, never a store.
  std::vector<u8> b = {
      rex(1, d >> 3, 0, s >> 3),
      0x8b,
      static_cast<u8>(0xc0 | ((d & 7) << 3) | (s & 7)),
  };
  return {std::move(b), Ins{.kind = Ins::Kind::Mov, .r0 = d, .r1 = s}};
}

Encoded e_mov32(int d, int s) {
  std::vector<u8> b;
  if ((d | s) >= 8) b.push_back(rex(0, d >> 3, 0, s >> 3));
  b.push_back(0x8b);
  b.push_back(static_cast<u8>(0xc0 | ((d & 7) << 3) | (s & 7)));
  return {std::move(b), Ins{.kind = Ins::Kind::Mov32, .r0 = d, .r1 = s}};
}

Encoded e_alu(AluOp op, int d, int s) {
  std::vector<u8> b = {
      rex(1, s >> 3, 0, d >> 3),
      alu_rr_opcode(op),
      static_cast<u8>(0xc0 | ((s & 7) << 3) | (d & 7)),
  };
  return {std::move(b), Ins{.kind = Ins::Kind::Alu, .alu_op = op, .r0 = d, .r1 = s}};
}

Encoded e_zero(int d) {
  std::vector<u8> b;
  if (d >= 8) b.push_back(rex(0, d >> 3, 0, d >> 3));
  b.push_back(0x31);
  b.push_back(static_cast<u8>(0xc0 | ((d & 7) << 3) | (d & 7)));
  return {std::move(b), Ins{.kind = Ins::Kind::Alu, .alu_op = AluOp::Xor, .r0 = d, .r1 = d}};
}

std::vector<u8> encode_ext(int ext, int d, u64 imm) {
  i64 s = static_cast<i64>(imm);
  std::vector<u8> b = {rex(1, 0, 0, d >> 3)};
  if (s >= -128 && s < 128) {
    b.push_back(0x83);
    b.push_back(static_cast<u8>(0xc0 | (ext << 3) | (d & 7)));
    push_le(b, static_cast<u64>(static_cast<u8>(s)), 1);
  } else {
    require(s >= -(1LL << 31) && s < (1LL << 31), "aluI/cmpI immediate out of int32 range");
    b.push_back(0x81);
    b.push_back(static_cast<u8>(0xc0 | (ext << 3) | (d & 7)));
    push_le(b, static_cast<u64>(static_cast<u32>(static_cast<i32>(s))), 4);
  }
  return b;
}

Encoded e_aluI(AluOp op, int d, u64 imm) {
  return {encode_ext(alu_ext(op), d, imm),
          Ins{.kind = Ins::Kind::AluI, .alu_op = op, .r0 = d, .imm = imm}};
}

Encoded e_cmpI(int a, u64 imm) {
  return {encode_ext(7, a, imm), Ins{.kind = Ins::Kind::CmpI, .r0 = a, .imm = imm}};
}

Encoded e_cmp(int a, int b) {
  std::vector<u8> bs = {
      rex(1, b >> 3, 0, a >> 3),
      0x39,
      static_cast<u8>(0xc0 | ((b & 7) << 3) | (a & 7)),
  };
  return {std::move(bs), Ins{.kind = Ins::Kind::Cmp, .r0 = a, .r1 = b}};
}

Encoded e_test(int a, int b) {
  std::vector<u8> bs = {
      rex(1, b >> 3, 0, a >> 3),
      0x85,
      static_cast<u8>(0xc0 | ((b & 7) << 3) | (a & 7)),
  };
  return {std::move(bs), Ins{.kind = Ins::Kind::Test, .r0 = a, .r1 = b}};
}

Encoded e_shlI(int d, u64 c) {
  std::vector<u8> bs = {rex(1, 0, 0, d >> 3), 0xc1, static_cast<u8>(0xe0 | (d & 7)),
                        static_cast<u8>(c)};
  return {std::move(bs), Ins{.kind = Ins::Kind::ShlI, .r0 = d, .imm = c}};
}

Encoded e_shrI(int d, u64 c) {
  std::vector<u8> bs = {rex(1, 0, 0, d >> 3), 0xc1, static_cast<u8>(0xe8 | (d & 7)),
                        static_cast<u8>(c)};
  return {std::move(bs), Ins{.kind = Ins::Kind::ShrI, .r0 = d, .imm = c}};
}

Encoded e_ld(int d, int b, i32 disp = 0) {
  std::vector<u8> bs = {rex(1, d >> 3, 0, b >> 3), 0x8b};
  append_memop(bs, d, b, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::Ld, .r0 = d, .r1 = b, .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_ld32(int d, int b, i32 disp = 0) {
  std::vector<u8> bs;
  if ((d | b) >= 8) bs.push_back(rex(0, d >> 3, 0, b >> 3));
  bs.push_back(0x8b);
  append_memop(bs, d, b, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::Ld32, .r0 = d, .r1 = b, .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_ld8(int d, int b, i32 disp = 0) {
  std::vector<u8> bs;
  if ((d | b) >= 8) bs.push_back(rex(0, d >> 3, 0, b >> 3));
  bs.push_back(0x0f);
  bs.push_back(0xb6);
  append_memop(bs, d, b, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::Ld8, .r0 = d, .r1 = b, .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_st(int b, i32 disp, int s) {
  std::vector<u8> bs = {rex(1, s >> 3, 0, b >> 3), 0x89};
  append_memop(bs, s, b, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::St, .r0 = b, .r1 = s, .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_st8(int b, i32 disp, int s) {
  std::vector<u8> bs;
  if (s >= 4 || b >= 8) bs.push_back(rex(0, s >> 3, 0, b >> 3));
  bs.push_back(0x88);
  append_memop(bs, s, b, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::St8, .r0 = b, .r1 = s, .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_lea(int d, int b, i32 disp) {
  std::vector<u8> bs = {rex(1, d >> 3, 0, b >> 3), 0x8d};
  append_memop(bs, d, b, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::Lea, .r0 = d, .r1 = b, .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_syscall() {
  return {{0x0f, 0x05}, Ins{.kind = Ins::Kind::Syscall}};
}

// ------------------------------------------------------- X86Dec transcription

u64 sx8(u8 v) { return static_cast<u64>(static_cast<i64>(static_cast<std::int8_t>(v))); }

u64 sx32(u32 v) { return static_cast<u64>(static_cast<i64>(static_cast<i32>(v))); }

struct ImageSlice {
  const std::vector<u8>& image;
  u64 base_off;

  std::optional<u8> at(std::size_t off) const {
    if (base_off + off >= image.size()) return std::nullopt;
    return image[base_off + off];
  }
};

std::optional<u64> le_bytes(const ImageSlice& b, std::size_t off, int n) {
  u64 r = 0;
  for (int i = 0; i < n; ++i) {
    auto x = b.at(off + static_cast<std::size_t>(i));
    if (!x.has_value()) return std::nullopt;
    r |= static_cast<u64>(*x) << (8 * i);
  }
  return r;
}

struct MemOpDec {
  int base;
  u64 disp;
  std::size_t next_off;
};

std::optional<MemOpDec> mem_op(const ImageSlice& b, std::size_t off, int m, int bb, bool x) {
  int md = m / 64;
  int rm = m % 8;
  if (md == 3) return std::nullopt;
  int base = 0;
  std::size_t off1 = 0;
  if (rm == 4) {
    auto s = b.at(off);
    if (!s.has_value()) return std::nullopt;
    if (((*s / 8) % 8) != 4 || x) return std::nullopt;
    if (md == 0 && (*s % 8) == 5) return std::nullopt;
    base = (*s % 8) + bb;
    off1 = off + 1;
  } else if (md == 0 && rm == 5) {
    return std::nullopt;
  } else {
    base = rm + bb;
    off1 = off;
  }
  if (md == 0) return MemOpDec{base, 0ULL, off1};
  if (md == 1) {
    auto d = b.at(off1);
    if (!d.has_value()) return std::nullopt;
    return MemOpDec{base, sx8(*d), off1 + 1};
  }
  auto d = le_bytes(b, off1, 4);
  if (!d.has_value()) return std::nullopt;
  return MemOpDec{base, sx32(static_cast<u32>(*d)), off1 + 4};
}

struct Decoded {
  Ins ins;
  std::size_t len;
};

std::optional<Decoded> decode(const ImageSlice& b, u64 a) {
  auto b0 = b.at(0);
  if (!b0.has_value()) return std::nullopt;
  bool has_rex = (*b0 >= 0x40 && *b0 <= 0x4f);
  int rx = has_rex ? (*b0 - 0x40) : 0;
  std::size_t p = has_rex ? 1 : 0;
  bool w = ((rx / 8) % 2) == 1;
  int r = ((rx / 4) % 2) * 8;
  bool x = ((rx / 2) % 2) == 1;
  int bb = (rx % 2) * 8;
  auto o_opt = b.at(p);
  if (!o_opt.has_value()) return std::nullopt;
  u8 o = *o_opt;
  auto tgt = [a](std::size_t ln, u64 rel) -> u64 { return a + ln + rel; };

  if (o >= 0xb8 && o <= 0xbf) {
    if (r != 0 || x) return std::nullopt;
    if (w) {
      auto imm = le_bytes(b, p + 1, 8);
      if (!imm.has_value()) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::MovI, .r0 = (o - 0xb8) + bb, .imm = *imm}, p + 9};
    }
    auto imm = le_bytes(b, p + 1, 4);
    if (!imm.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::MovI, .r0 = (o - 0xb8) + bb, .imm = *imm}, p + 5};
  }
  if (o == 0x01 || o == 0x29 || o == 0x21 || o == 0x09 || o == 0x31) {
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value()) return std::nullopt;
    int m = *m_opt;
    if (m / 64 != 3 || x) return std::nullopt;
    int d = (m % 8) + bb;
    int s = ((m / 8) % 8) + r;
    auto op = alu_of_rr(o);
    if (!op.has_value()) return std::nullopt;
    if (w) return Decoded{Ins{.kind = Ins::Kind::Alu, .alu_op = *op, .r0 = d, .r1 = s}, p + 2};
    if (o == 0x31 && d == s) {
      return Decoded{Ins{.kind = Ins::Kind::Alu, .alu_op = AluOp::Xor, .r0 = d, .r1 = d}, p + 2};
    }
    return std::nullopt;
  }
  if (o == 0x89 || o == 0x8b) {
    if (x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value()) return std::nullopt;
    int m = *m_opt;
    if (!w) {
      int rm = (m % 8) + bb;
      int rg = ((m / 8) % 8) + r;
      if (m / 64 == 3) {
        return Decoded{Ins{.kind = Ins::Kind::Mov32,
                           .r0 = (o == 0x89 ? rm : rg),
                           .r1 = (o == 0x89 ? rg : rm)},
                       p + 2};
      }
      if (o == 0x89) return std::nullopt;
      auto mo = mem_op(b, p + 2, m, bb, x);
      if (!mo.has_value()) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::Ld32, .r0 = rg, .r1 = mo->base, .imm = mo->disp},
                     mo->next_off};
    }
    if (m / 64 == 3) {
      int rm = (m % 8) + bb;
      int rg = ((m / 8) % 8) + r;
      return Decoded{Ins{.kind = Ins::Kind::Mov,
                         .r0 = (o == 0x89 ? rm : rg),
                         .r1 = (o == 0x89 ? rg : rm)},
                     p + 2};
    }
    auto mo = mem_op(b, p + 2, m, bb, x);
    if (!mo.has_value()) return std::nullopt;
    int rg = ((m / 8) % 8) + r;
    if (o == 0x89) {
      return Decoded{Ins{.kind = Ins::Kind::St, .r0 = mo->base, .r1 = rg, .imm = mo->disp},
                     mo->next_off};
    }
    return Decoded{Ins{.kind = Ins::Kind::Ld, .r0 = rg, .r1 = mo->base, .imm = mo->disp},
                   mo->next_off};
  }
  if (o == 0x85) {
    if (!w || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || *m_opt / 64 != 3) return std::nullopt;
    int m = *m_opt;
    return Decoded{Ins{.kind = Ins::Kind::Test, .r0 = (m % 8) + bb, .r1 = ((m / 8) % 8) + r},
                   p + 2};
  }
  if (o == 0x6a) {
    if (has_rex) return std::nullopt;
    auto imm = b.at(1);
    if (!imm.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::PushI, .imm = sx8(*imm)}, 2};
  }
  if (o == 0x39 || o == 0x3b) {
    if (!w || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || *m_opt / 64 != 3) return std::nullopt;
    int m = *m_opt;
    int rm = (m % 8) + bb;
    int rg = ((m / 8) % 8) + r;
    return Decoded{Ins{.kind = Ins::Kind::Cmp,
                       .r0 = (o == 0x39 ? rm : rg),
                       .r1 = (o == 0x39 ? rg : rm)},
                   p + 2};
  }
  if (o == 0x83 || o == 0x81) {
    if (!w || r != 0 || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || *m_opt / 64 != 3) return std::nullopt;
    int m = *m_opt;
    int d = (m % 8) + bb;
    int e = (m / 8) % 8;
    u64 imm = 0;
    std::size_t ln = 0;
    if (o == 0x83) {
      auto i_opt = b.at(p + 2);
      if (!i_opt.has_value()) return std::nullopt;
      imm = sx8(*i_opt);
      ln = p + 3;
    } else {
      auto i_opt = le_bytes(b, p + 2, 4);
      if (!i_opt.has_value()) return std::nullopt;
      imm = sx32(static_cast<u32>(*i_opt));
      ln = p + 6;
    }
    if (e == 7) return Decoded{Ins{.kind = Ins::Kind::CmpI, .r0 = d, .imm = imm}, ln};
    auto op = alu_of_ext(e);
    if (!op.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::AluI, .alu_op = *op, .r0 = d, .imm = imm}, ln};
  }
  if (o == 0xc1) {
    if (!w || r != 0 || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    auto c_opt = b.at(p + 2);
    if (!m_opt.has_value() || !c_opt.has_value() || *m_opt / 64 != 3) return std::nullopt;
    int d = (*m_opt % 8) + bb;
    int e = (*m_opt / 8) % 8;
    if (e == 4) return Decoded{Ins{.kind = Ins::Kind::ShlI, .r0 = d, .imm = *c_opt}, p + 3};
    if (e == 5) return Decoded{Ins{.kind = Ins::Kind::ShrI, .r0 = d, .imm = *c_opt}, p + 3};
    return std::nullopt;
  }
  if (o == 0x8d) {
    if (!w || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value()) return std::nullopt;
    auto mo = mem_op(b, p + 2, *m_opt, bb, x);
    if (!mo.has_value()) return std::nullopt;
    return Decoded{
        Ins{.kind = Ins::Kind::Lea, .r0 = ((*m_opt / 8) % 8) + r, .r1 = mo->base, .imm = mo->disp},
        mo->next_off};
  }
  if (o == 0x88) {
    if (x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value()) return std::nullopt;
    auto mo = mem_op(b, p + 2, *m_opt, bb, x);
    if (!mo.has_value()) return std::nullopt;
    int reg = (*m_opt / 8) % 8;
    if (!has_rex && reg >= 4) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::St8, .r0 = mo->base, .r1 = reg + r, .imm = mo->disp},
                   mo->next_off};
  }
  if (o == 0x0f) {
    auto o2_opt = b.at(p + 1);
    if (!o2_opt.has_value()) return std::nullopt;
    u8 o2 = *o2_opt;
    if (o2 == 0x05) {
      if (has_rex) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::Syscall}, 2};
    }
    if (o2 == 0xb6) {
      if (x) return std::nullopt;
      auto m_opt = b.at(p + 2);
      if (!m_opt.has_value()) return std::nullopt;
      auto mo = mem_op(b, p + 3, *m_opt, bb, x);
      if (!mo.has_value()) return std::nullopt;
      return Decoded{
          Ins{.kind = Ins::Kind::Ld8, .r0 = ((*m_opt / 8) % 8) + r, .r1 = mo->base, .imm = mo->disp},
          mo->next_off};
    }
    if (o2 >= 0x80 && o2 <= 0x8f) {
      if (has_rex) return std::nullopt;
      auto d = le_bytes(b, 2, 4);
      if (!d.has_value()) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::Jcc,
                         .r0 = o2 - 0x80,
                         .imm = tgt(6, sx32(static_cast<u32>(*d)))},
                     6};
    }
    return std::nullopt;
  }
  if (o >= 0x70 && o <= 0x7f) {
    if (has_rex) return std::nullopt;
    auto d = b.at(1);
    if (!d.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::Jcc, .r0 = o - 0x70, .imm = tgt(2, sx8(*d))}, 2};
  }
  if (o == 0xeb) {
    if (has_rex) return std::nullopt;
    auto d = b.at(1);
    if (!d.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::Jmp, .imm = tgt(2, sx8(*d))}, 2};
  }
  if (o == 0xe9) {
    if (has_rex) return std::nullopt;
    auto d = le_bytes(b, 1, 4);
    if (!d.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::Jmp, .imm = tgt(5, sx32(static_cast<u32>(*d)))}, 5};
  }
  if (o == 0xe8) {
    if (has_rex) return std::nullopt;
    auto d = le_bytes(b, 1, 4);
    if (!d.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::Call, .imm = tgt(5, sx32(static_cast<u32>(*d)))}, 5};
  }
  if (o == 0xc3) {
    if (has_rex) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::Ret}, 1};
  }
  if (o >= 0x50 && o <= 0x57) {
    if (w || r != 0 || x) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::Push, .r0 = (o - 0x50) + bb}, p + 1};
  }
  if (o >= 0x58 && o <= 0x5f) {
    if (w || r != 0 || x) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::Pop, .r0 = (o - 0x58) + bb}, p + 1};
  }
  if (o == 0xff) {
    if (w || r != 0) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value()) return std::nullopt;
    int m = *m_opt;
    if (((m / 8) % 8) != 4 || (m / 64) == 3 || (m % 8) != 4) return std::nullopt;
    auto s_opt = b.at(p + 2);
    if (!s_opt.has_value() || (*s_opt / 64) != 3) return std::nullopt;
    int s = *s_opt;
    int idx = ((s / 8) % 8) + (x ? 8 : 0);
    if (idx == 4) return std::nullopt;
    int md = m / 64;
    if (md == 0 && (s % 8) == 5) return std::nullopt;
    int base = (s % 8) + bb;
    if (md == 0) return Decoded{Ins{.kind = Ins::Kind::Jtab, .r0 = base, .r1 = idx, .imm = 0}, p + 3};
    if (md == 1) {
      auto d = b.at(p + 3);
      if (!d.has_value()) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::Jtab, .r0 = base, .r1 = idx, .imm = sx8(*d)}, p + 4};
    }
    auto d = le_bytes(b, p + 3, 4);
    if (!d.has_value()) return std::nullopt;
    return Decoded{
        Ins{.kind = Ins::Kind::Jtab, .r0 = base, .r1 = idx, .imm = sx32(static_cast<u32>(*d))},
        p + 7};
  }
  return std::nullopt;
}

// ---------------------------------------------------------------- assembler

struct AsmItem {
  enum class Kind { Label, Inst, Jump, Raw };

  Kind kind = Kind::Label;
  std::string label;
  Encoded enc;
  bool is_jcc = false;
  int cc = 0;
  std::string target_lab;
  std::string comment;
  bool far = false;
};

struct MacroRec {
  std::string name;
  std::map<std::string, int> reg_params;
  std::map<std::string, std::string> addr_params;
  std::size_t start_idx = 0;
  std::size_t end_idx = 0;
};

struct OutEntry {
  u64 addr = 0;
  std::vector<u8> bytes;
  Ins ins;
  std::string comment;
  std::size_t idx = 0;
};

struct Asm {
  std::vector<AsmItem> items;
  int n = 0;
  std::vector<MacroRec> macros;
  std::vector<MacroRec> open;
  std::vector<u64> idx_addr;

  void label(std::string name) {
    AsmItem it;
    it.kind = AsmItem::Kind::Label;
    it.label = std::move(name);
    items.push_back(std::move(it));
  }

  std::string fresh(std::string_view stem) {
    ++n;
    return std::string(stem) + "_" + std::to_string(n);
  }

  void i(Encoded enc, std::string c = "") {
    AsmItem it;
    it.kind = AsmItem::Kind::Inst;
    it.enc = std::move(enc);
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void jmp(std::string lab, std::string c = "", bool far = false) {
    AsmItem it;
    it.kind = AsmItem::Kind::Jump;
    it.is_jcc = false;
    it.target_lab = std::move(lab);
    it.comment = std::move(c);
    it.far = far;
    items.push_back(std::move(it));
  }

  void jcc(Cc cc, std::string lab, std::string c = "", bool far = false) {
    AsmItem it;
    it.kind = AsmItem::Kind::Jump;
    it.is_jcc = true;
    it.cc = static_cast<int>(cc);
    it.target_lab = std::move(lab);
    it.comment = std::move(c);
    it.far = far;
    items.push_back(std::move(it));
  }

  void raw(std::vector<u8> bs, Ins ins, std::string c = "") {
    AsmItem it;
    it.kind = AsmItem::Kind::Raw;
    it.enc = Encoded{std::move(bs), ins};
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void mbegin(std::string name,
              std::map<std::string, int> reg_params,
              std::map<std::string, std::string> addr_params) {
    open.push_back(MacroRec{.name = std::move(name),
                            .reg_params = std::move(reg_params),
                            .addr_params = std::move(addr_params),
                            .start_idx = items.size(),
                            .end_idx = 0});
  }

  void mend() {
    require(!open.empty(), "unmatched mend");
    MacroRec m = std::move(open.back());
    open.pop_back();
    m.end_idx = items.size();
    macros.push_back(std::move(m));
  }

  struct Result {
    std::vector<OutEntry> out;
    std::map<std::string, u64> labels;
    std::vector<std::pair<std::string, u64>> label_order;
  };

  Result assemble(u64 org) {
    std::set<std::size_t> long_jumps;
    for (std::size_t idx = 0; idx < items.size(); ++idx) {
      if (items[idx].kind == AsmItem::Kind::Jump && items[idx].far) {
        long_jumps.insert(idx);
      }
    }
    while (true) {
      u64 addr = org;
      std::map<std::string, u64> labels;
      std::vector<std::pair<std::string, u64>> label_order;
      std::vector<u64> cur_idx_addr(items.size() + 1, 0);

      for (std::size_t idx = 0; idx < items.size(); ++idx) {
        cur_idx_addr[idx] = addr;
        const auto& it = items[idx];
        switch (it.kind) {
          case AsmItem::Kind::Label:
            require(labels.find(it.label) == labels.end(), "duplicate label: " + it.label);
            labels.emplace(it.label, addr);
            label_order.emplace_back(it.label, addr);
            break;
          case AsmItem::Kind::Inst:
          case AsmItem::Kind::Raw:
            addr += it.enc.bytes.size();
            break;
          case AsmItem::Kind::Jump:
            if (long_jumps.contains(idx)) {
              if (!it.is_jcc) {
                addr += 5;
              } else if (it.cc == static_cast<int>(Cc::NE)) {
                addr += 7;
              } else {
                addr += 6;
              }
            } else {
              addr += 2;
            }
            break;
        }
      }
      cur_idx_addr[items.size()] = addr;

      bool grew = false;
      std::vector<OutEntry> out;
      for (std::size_t idx = 0; idx < items.size(); ++idx) {
        u64 a = cur_idx_addr[idx];
        const auto& it = items[idx];
        if (it.kind == AsmItem::Kind::Label) continue;
        if (it.kind == AsmItem::Kind::Inst || it.kind == AsmItem::Kind::Raw) {
          out.push_back(OutEntry{a, it.enc.bytes, it.enc.ins, it.comment, idx});
          continue;
        }
        auto lit = labels.find(it.target_lab);
        require(lit != labels.end(), "missing label: " + it.target_lab);
        u64 t = lit->second;
        if (!long_jumps.contains(idx)) {
          i64 rel = static_cast<i64>(t) - static_cast<i64>(a + 2);
          if (rel >= -128 && rel < 128) {
            u8 rb = static_cast<u8>(rel & 0xff);
            if (!it.is_jcc) {
              out.push_back(OutEntry{a,
                                     {0xeb, rb},
                                     Ins{.kind = Ins::Kind::Jmp, .imm = t},
                                     it.comment + " -> " + it.target_lab,
                                     idx});
            } else {
              out.push_back(OutEntry{a,
                                     {static_cast<u8>(0x70 + it.cc), rb},
                                     Ins{.kind = Ins::Kind::Jcc, .r0 = it.cc, .imm = t},
                                     it.comment + " -> " + it.target_lab,
                                     idx});
            }
            continue;
          }
          long_jumps.insert(idx);
          grew = true;
          break;
        }
        if (!it.is_jcc) {
          i64 rel = static_cast<i64>(t) - static_cast<i64>(a + 5);
          std::vector<u8> bs = {0xe9};
          push_le(bs, static_cast<u64>(static_cast<u32>(static_cast<i32>(rel))), 4);
          out.push_back(OutEntry{a,
                                 std::move(bs),
                                 Ins{.kind = Ins::Kind::Jmp, .imm = t},
                                 it.comment + " -> " + it.target_lab,
                                 idx});
        } else if (it.cc == static_cast<int>(Cc::NE)) {
          // Never emit 0f 85 (one bit from 0f 05): je over a jmp rel32.
          out.push_back(OutEntry{a,
                                 {0x74, 0x05},
                                 Ins{.kind = Ins::Kind::Jcc, .r0 = static_cast<int>(Cc::E), .imm = a + 7},
                                 it.comment + " (jne: je over the jmp)",
                                 idx});
          i64 rel = static_cast<i64>(t) - static_cast<i64>(a + 7);
          std::vector<u8> bs = {0xe9};
          push_le(bs, static_cast<u64>(static_cast<u32>(static_cast<i32>(rel))), 4);
          out.push_back(OutEntry{a + 2,
                                 std::move(bs),
                                 Ins{.kind = Ins::Kind::Jmp, .imm = t},
                                 "-> " + it.target_lab,
                                 idx});
        } else {
          i64 rel = static_cast<i64>(t) - static_cast<i64>(a + 6);
          std::vector<u8> bs = {0x0f, static_cast<u8>(0x80 + it.cc)};
          push_le(bs, static_cast<u64>(static_cast<u32>(static_cast<i32>(rel))), 4);
          out.push_back(OutEntry{a,
                                 std::move(bs),
                                 Ins{.kind = Ins::Kind::Jcc, .r0 = it.cc, .imm = t},
                                 it.comment + " -> " + it.target_lab,
                                 idx});
        }
      }
      if (!grew) {
        idx_addr = std::move(cur_idx_addr);
        return Result{std::move(out), std::move(labels), std::move(label_order)};
      }
    }
  }
};

// ---------------------------------------------------------------- the program
//
// Registers in the passes: rax, rbx, rcx hold the three results and are written
// only by the verdicts; rsp is never touched; rbp is the end of the input copy.
// The macros use only r8-r11, r14, r15 so every instance has identical lengths;
// jumps out of a macro are always rel32.

void skipcode(Asm& A, int p, const std::string& rej, const std::string& c = "") {
  A.mbegin("skipcode", {{"p", p}}, {{"rej", rej}});
  constexpr int cnt = R11, t1 = R14;
  std::string top = A.fresh("sc_top"), done = A.fresh("sc_done");
  std::string l1 = A.fresh("sc_l1"), l5 = A.fresh("sc_l5");
  std::string l9 = A.fresh("sc_l9"), ln = A.fresh("sc_len");
  A.label(top);
  A.i(e_test(cnt, cnt), "skipCode" + c + ": instructions left?");
  A.jcc(Cc::E, done);
  A.i(e_cmp(p, RBP), "opcode byte inside the input?");
  A.jcc(Cc::AE, rej, "", true);
  A.i(e_ld8(t1, p, 0), "opcode");
  A.i(e_cmpI(t1, 57), "prim 0x00..0x38");
  A.jcc(Cc::B, l1);
  A.i(e_cmpI(t1, 0x80));
  A.jcc(Cc::B, rej, "0x39..0x7f: no such opcode", true);
  A.jcc(Cc::E, l9, "0x80 lit u64");
  A.i(e_cmpI(t1, 0x8c));
  A.jcc(Cc::A, rej, "0x8d..0xff: no such opcode", true);
  A.jcc(Cc::E, l1, "0x8c ret");
  A.i(e_cmpI(t1, 0x83));
  A.jcc(Cc::B, l5, "0x81 str, 0x82 pick");
  A.i(e_cmpI(t1, 0x86));
  A.jcc(Cc::B, l1, "0x83 tor, 0x84 fromr, 0x85 rfetch");
  A.label(l5);
  A.i(e_movI(t1, 5), "opcode + u32");
  A.jmp(ln);
  A.label(l9);
  A.i(e_movI(t1, 9), "opcode + u64");
  A.jmp(ln);
  A.label(l1);
  A.i(e_movI(t1, 1), "opcode alone");
  A.label(ln);
  A.i(e_alu(AluOp::Add, p, t1), "past the instruction");
  A.i(e_cmp(p, RBP));
  A.jcc(Cc::A, rej, "truncated instruction", true);
  A.i(e_aluI(AluOp::Sub, cnt, 1));
  A.jmp(top);
  A.label(done);
  A.mend();
}

void skipwords(Asm& A, const std::string& rej, const std::string& c = "") {
  A.mbegin("skipwords", {}, {{"rej", rej}});
  constexpr int p = R10, wc = R15, nc = R11, t1 = R14;
  std::string top = A.fresh("sw_top"), done = A.fresh("sw_done");
  A.label(top);
  A.i(e_test(wc, wc), "skipWords" + c + ": words left?");
  A.jcc(Cc::E, done);
  A.i(e_mov(t1, p));
  A.i(e_aluI(AluOp::Add, t1, 20));
  A.i(e_cmp(t1, RBP));
  A.jcc(Cc::A, rej, "truncated word header", true);
  A.i(e_ld32(t1, p, 16), "na");
  A.i(e_shlI(t1, 32));
  A.i(e_alu(AluOp::Add, wc, t1), "park na in the high half");
  A.i(e_ld32(nc, p, 12), "nc");
  A.i(e_aluI(AluOp::Add, p, 20), "first instruction");
  skipcode(A, p, rej, c);
  A.i(e_mov(t1, wc));
  A.i(e_shrI(t1, 32), "na");
  A.i(e_mov32(wc, wc), "words left, high half cleared");
  A.i(e_mov(nc, t1));
  A.i(e_shlI(nc, 3));
  A.i(e_shlI(t1, 2));
  A.i(e_alu(AluOp::Add, nc, t1), "12 * na");
  A.i(e_alu(AluOp::Add, p, nc), "past the annotations");
  A.i(e_cmp(p, RBP));
  A.jcc(Cc::A, rej, "truncated annotations", true);
  A.i(e_aluI(AluOp::Sub, wc, 1));
  A.jmp(top);
  A.label(done);
  A.mend();
}

void skipdata(Asm& A, const std::string& rej) {
  A.mbegin("skipdata", {}, {{"rej", rej}});
  constexpr int p = R10, cnt = R15, t1 = R11, t2 = R14;
  std::string top = A.fresh("sd_top"), done = A.fresh("sd_done");
  A.label(top);
  A.i(e_test(cnt, cnt), "skipData: items left?");
  A.jcc(Cc::E, done);
  A.i(e_mov(t1, p));
  A.i(e_aluI(AluOp::Add, t1, 4));
  A.i(e_cmp(t1, RBP));
  A.jcc(Cc::A, rej, "truncated length", true);
  A.i(e_ld32(t2, p, 0), "len");
  A.i(e_mov(p, t1));
  A.i(e_mov(t1, RBP));
  A.i(e_alu(AluOp::Sub, t1, p), "bytes left");
  A.i(e_cmp(t1, t2));
  A.jcc(Cc::B, rej, "truncated data", true);
  A.i(e_alu(AluOp::Add, p, t2));
  A.i(e_aluI(AluOp::Sub, cnt, 1));
  A.jmp(top);
  A.label(done);
  A.mend();
}

void chk(Asm& A, int r, const std::string& ovf) {
  A.mbegin("chk", {{"r", r}}, {{"ovf", ovf}});
  A.i(e_mov(R10, r));
  A.i(e_shrI(R10, kHmaxShift));
  A.i(e_test(R10, R10));
  A.jcc(Cc::NE, ovf, "height >= 2^48", true);
  A.mend();
}

void lookup(Asm& A, int t, const std::string& c = "") {
  A.mbegin("lookup", {{"t", t}}, {});
  std::string top = A.fresh("lk_top"), done = A.fresh("lk_done");
  A.i(e_ld32(R14, R12, 16), "lookup" + c + ": na");
  A.i(e_mov(R11, R14));
  A.i(e_shlI(R11, 3));
  A.i(e_shlI(R14, 2));
  A.i(e_alu(AluOp::Add, R11, R14));
  A.i(e_alu(AluOp::Add, R11, R9), "r11 = end of the annotations");
  A.i(e_mov(R10, R9));
  A.label(top);
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::E, done, "none");
  A.i(e_ld32(R14, R10, 0));
  A.i(e_cmp(R14, t));
  A.jcc(Cc::E, done, "found");
  A.i(e_aluI(AluOp::Add, R10, 12));
  A.jmp(top);
  A.label(done);
  A.mend();
}

// Prim.eff by opcode: (operands taken, results pushed). exit (50) is handled apart.
constexpr std::array<std::pair<int, int>, 57> kEff = {{
    {1, 2}, {1, 0}, {2, 2}, {2, 3}, {3, 3}, {2, 1}, {2, 3},
    {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1}, {1, 1},
    {2, 1}, {2, 1}, {2, 1}, {1, 1}, {2, 1}, {2, 1}, {2, 1},
    {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1},
    {1, 1}, {2, 0}, {1, 1}, {2, 0}, {1, 1}, {2, 0}, {1, 1}, {2, 0},
    {3, 0}, {3, 0}, {3, 1},
    {1, 1}, {0, 1}, {1, 0},
    {4, 1}, {2, 1},
    {0, 1}, {3, 0}, {2, 2},
    {0, 1}, {1, 0}, {1, 0}, {1, 0},
    {0, 1}, {0, 1}, {1, 1},
    {1, 2}, {1, 2}, {2, 1},
}};

struct EffRun {
  int lo;
  int hi;
  std::pair<int, int> eff;
};

void effchain_tree(Asm& A, const std::string& have, const std::vector<EffRun>& rs,
                   std::size_t beg, std::size_t end) {
  if (end - beg == 1) {
    int i_cnt = rs[beg].eff.first;
    int o_cnt = rs[beg].eff.second;
    A.i(e_movI(R10, static_cast<u64>(i_cnt)),
        "ops " + std::to_string(rs[beg].lo) + ".." + std::to_string(rs[beg].hi) + ": takes " +
            std::to_string(i_cnt));
    A.i(e_movI(R11, static_cast<u64>(o_cnt)), "pushes " + std::to_string(o_cnt));
    A.jmp(have, "", true);
    return;
  }
  std::size_t mid = beg + (end - beg) / 2;
  std::string right = A.fresh("eff");
  A.i(e_cmpI(R15, static_cast<u64>(rs[mid].lo)));
  A.jcc(Cc::AE, right, "", true);
  effchain_tree(A, have, rs, beg, mid);
  A.label(right);
  effchain_tree(A, have, rs, mid, end);
}

void effchain(Asm& A, const std::string& have) {
  A.mbegin("eff", {}, {{"tgt", have}});
  std::vector<EffRun> runs;
  for (int op = 0; op < 57; ++op) {
    if (!runs.empty() && runs.back().eff == kEff[static_cast<std::size_t>(op)]) {
      runs.back().hi = op;
    } else {
      runs.push_back(EffRun{op, op, kEff[static_cast<std::size_t>(op)]});
    }
  }
  effchain_tree(A, have, runs, 0, runs.size());
  A.mend();
}

void gen_pass(Asm& A, int k) {
  u64 D = kDesc[static_cast<std::size_t>(k)];
  int R = kRes[static_cast<std::size_t>(k)];
  std::string P = "p" + std::to_string(k + 1);
  std::string rej = P + "_rej", ovf = P + "_ovf", acc = P + "_acc";
  std::string nxt = (k < 2) ? ("p" + std::to_string(k + 2)) : "vote";

  A.label(P);
  // ---- P1: parse the whole file
  A.i(e_movI(R15, D), "pass " + std::to_string(k + 1) + ": descriptor " + std::to_string(k + 1));
  A.i(e_ld(RDI, R15, 0), "base of input copy");
  A.i(e_ld(RDX, R15, 8), "N");
  A.i(e_mov(RBP, RDI));
  A.i(e_alu(AluOp::Add, RBP, RDX), "rbp = end of input");
  A.i(e_cmpI(RDX, 16));
  A.jcc(Cc::B, rej, "shorter than the file header", true);
  A.i(e_ld32(RDX, RDI, 0));
  A.i(e_cmpI(RDX, 0x31525053ULL));
  A.jcc(Cc::NE, rej, "not \"SPR1\"", true);
  A.i(e_ld32(R12, RDI, 4), "main");
  A.i(e_ld32(RSI, RDI, 8), "nw");
  A.i(e_ld32(R15, RDI, 12), "nd");
  A.i(e_lea(R10, RDI, 16), "first data item");
  skipdata(A, rej);
  A.i(e_mov(R13, R10), "ws = first word");
  A.i(e_mov(R15, RSI), "nw words");
  skipwords(A, rej, " (parse)");
  A.i(e_cmp(R10, RBP));
  A.jcc(Cc::NE, rej, "trailing bytes", true);
  // ---- P2: main
  A.i(e_cmp(R12, RSI));
  A.jcc(Cc::AE, rej, "main >= nw", true);
  A.i(e_mov(R10, R13));
  A.i(e_mov(R15, R12));
  skipwords(A, rej, " (main)");
  A.i(e_ld32(RDX, R10, 0), "ins of main");
  A.i(e_test(RDX, RDX));
  A.jcc(Cc::NE, rej, "main takes arguments", true);
  // ---- P3: every word
  A.i(e_mov(R12, R13), "r12 = word header");
  std::string wl = P + "_word";
  A.label(wl);
  A.i(e_cmp(R12, RBP));
  A.jcc(Cc::E, acc, "all words checked", true);
  A.i(e_ld32(R11, R12, 12), "nc");
  A.i(e_test(R11, R11));
  A.jcc(Cc::E, rej, "empty code", true);
  A.i(e_lea(R9, R12, 20));
  skipcode(A, R9, rej, " (annotations)");
  A.i(e_zero(R8), "i = 0");
  lookup(A, R8, " (0)");
  std::string init = A.fresh("init");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::E, init);
  A.i(e_ld32(R14, R10, 4));
  A.i(e_ld32(R15, R12, 0));
  A.i(e_cmp(R14, R15));
  A.jcc(Cc::NE, rej, "H[0].d != ins", true);
  A.i(e_ld32(R14, R10, 8));
  A.i(e_test(R14, R14));
  A.jcc(Cc::NE, rej, "H[0].a != 0", true);
  A.label(init);
  A.i(e_ld32(RDX, R12, 0), "curD = ins");
  A.i(e_zero(RDI), "curA = 0");
  A.i(e_lea(RSI, R12, 20), "p = first instruction");
  std::string walk = P + "_walk", wdone = P + "_wdone";
  std::string fall = P + "_fall", nofall = P + "_nofall", adv = P + "_adv";
  A.label(walk);
  A.i(e_ld32(R15, R12, 12));
  A.i(e_cmp(R8, R15));
  A.jcc(Cc::E, wdone, "i == nc", true);
  lookup(A, R8, " (i)");
  std::string hr = A.fresh("hready");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::E, hr);
  A.i(e_ld32(RDX, R10, 4), "h = annotation at i");
  A.i(e_ld32(RDI, R10, 8));
  A.label(hr);
  A.i(e_ld8(R15, RSI, 0), "opcode");
  auto nm = [&](const char* stem) { return P + "_" + stem; };
  A.i(e_cmpI(R15, 57));
  A.jcc(Cc::B, nm("prim"), "", true);
  constexpr std::array<std::pair<u64, const char*>, 13> kOpTargets = {{
      {0x80, "lit"}, {0x81, "str"}, {0x82, "pick"}, {0x83, "tor"}, {0x84, "fromr"},
      {0x85, "rfetch"}, {0x86, "locget"}, {0x87, "locset"}, {0x88, "jmp"},
      {0x89, "jz"}, {0x8a, "jz"}, {0x8b, "call"}, {0x8c, "ret"},
  }};
  for (const auto& [op, target] : kOpTargets) {
    A.i(e_cmpI(R15, op));
    A.jcc(Cc::E, nm(target), "", true);
  }
  A.jmp(rej, "unreachable after parse", true);
  // prim
  A.label(nm("prim"));
  A.i(e_cmpI(R15, 50));
  A.jcc(Cc::E, nm("exit"), "", true);
  effchain(A, nm("have"));
  A.label(nm("have"));
  A.i(e_cmp(RDX, R10));
  A.jcc(Cc::B, rej, "too few operands", true);
  A.i(e_alu(AluOp::Sub, RDX, R10));
  A.i(e_alu(AluOp::Add, RDX, R11));
  chk(A, RDX, ovf);
  A.i(e_movI(R15, 1));
  A.jmp(fall, "", true);
  A.label(nm("exit"));
  A.i(e_test(RDX, RDX));
  A.jcc(Cc::E, rej, "exit needs one operand", true);
  A.i(e_movI(R15, 1));
  A.jmp(nofall, "", true);
  // lit / str
  A.label(nm("lit"));
  A.i(e_aluI(AluOp::Add, RDX, 1));
  chk(A, RDX, ovf);
  A.i(e_movI(R15, 9));
  A.jmp(fall, "", true);
  A.label(nm("str"));
  A.i(e_aluI(AluOp::Add, RDX, 2));
  chk(A, RDX, ovf);
  A.i(e_movI(R15, 5));
  A.jmp(fall, "", true);
  // pick n: n < d
  A.label(nm("pick"));
  A.i(e_ld32(R10, RSI, 1));
  A.i(e_cmp(R10, RDX));
  A.jcc(Cc::AE, rej, "pick beyond the stack", true);
  A.i(e_aluI(AluOp::Add, RDX, 1));
  chk(A, RDX, ovf);
  A.i(e_movI(R15, 5));
  A.jmp(fall, "", true);
  // tor
  A.label(nm("tor"));
  A.i(e_test(RDX, RDX));
  A.jcc(Cc::E, rej, "", true);
  A.i(e_aluI(AluOp::Sub, RDX, 1));
  A.i(e_aluI(AluOp::Add, RDI, 1));
  chk(A, RDI, ovf);
  A.i(e_movI(R15, 1));
  A.jmp(fall, "", true);
  // fromr
  A.label(nm("fromr"));
  A.i(e_test(RDI, RDI));
  A.jcc(Cc::E, rej, "", true);
  A.i(e_aluI(AluOp::Sub, RDI, 1));
  A.i(e_aluI(AluOp::Add, RDX, 1));
  chk(A, RDX, ovf);
  A.i(e_movI(R15, 1));
  A.jmp(fall, "", true);
  // rfetch
  A.label(nm("rfetch"));
  A.i(e_test(RDI, RDI));
  A.jcc(Cc::E, rej, "", true);
  A.i(e_aluI(AluOp::Add, RDX, 1));
  chk(A, RDX, ovf);
  A.i(e_movI(R15, 1));
  A.jmp(fall, "", true);
  // locget i: i < locals
  A.label(nm("locget"));
  A.i(e_ld32(R10, RSI, 1));
  A.i(e_ld32(R11, R12, 8), "locals");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::AE, rej, "", true);
  A.i(e_aluI(AluOp::Add, RDX, 1));
  chk(A, RDX, ovf);
  A.i(e_movI(R15, 5));
  A.jmp(fall, "", true);
  // locset i: 1 <= d and i < locals
  A.label(nm("locset"));
  A.i(e_test(RDX, RDX));
  A.jcc(Cc::E, rej, "", true);
  A.i(e_ld32(R10, RSI, 1));
  A.i(e_ld32(R11, R12, 8), "locals");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::AE, rej, "", true);
  A.i(e_aluI(AluOp::Sub, RDX, 1));
  A.i(e_movI(R15, 5));
  A.jmp(fall, "", true);
  // jmp t: ann(t) = h
  A.label(nm("jmp"));
  A.i(e_ld32(R15, RSI, 1), "t");
  lookup(A, R15, " (jmp t)");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::E, rej, "no annotation at t", true);
  A.i(e_ld32(R14, R10, 4));
  A.i(e_cmp(R14, RDX));
  A.jcc(Cc::NE, rej, "", true);
  A.i(e_ld32(R14, R10, 8));
  A.i(e_cmp(R14, RDI));
  A.jcc(Cc::NE, rej, "", true);
  A.i(e_movI(R15, 5));
  A.jmp(nofall, "", true);
  // jz/jnz t: 1 <= d and ann(t) = (d - 1, a)
  A.label(nm("jz"));
  A.i(e_test(RDX, RDX));
  A.jcc(Cc::E, rej, "", true);
  A.i(e_aluI(AluOp::Sub, RDX, 1));
  A.i(e_ld32(R15, RSI, 1), "t");
  lookup(A, R15, " (jz t)");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::E, rej, "no annotation at t", true);
  A.i(e_ld32(R14, R10, 4));
  A.i(e_cmp(R14, RDX));
  A.jcc(Cc::NE, rej, "", true);
  A.i(e_ld32(R14, R10, 8));
  A.i(e_cmp(R14, RDI));
  A.jcc(Cc::NE, rej, "", true);
  A.i(e_movI(R15, 5));
  A.jmp(fall, "", true);
  // call w: w < nw, c.ins <= d, d - c.ins + c.outs
  A.label(nm("call"));
  A.i(e_ld32(R15, RSI, 1), "w");
  A.i(e_movI(R14, D));
  A.i(e_ld(R14, R14, 0), "base");
  A.i(e_ld32(R14, R14, 8), "nw");
  A.i(e_cmp(R15, R14));
  A.jcc(Cc::AE, rej, "no such word", true);
  A.i(e_mov(R10, R13), "rescan from ws");
  skipwords(A, rej, " (call)");
  A.i(e_ld32(R14, R10, 0), "c.ins");
  A.i(e_cmp(RDX, R14));
  A.jcc(Cc::B, rej, "too few arguments", true);
  A.i(e_alu(AluOp::Sub, RDX, R14));
  A.i(e_ld32(R14, R10, 4), "c.outs");
  A.i(e_alu(AluOp::Add, RDX, R14));
  chk(A, RDX, ovf);
  A.i(e_movI(R15, 5));
  A.jmp(fall, "", true);
  // ret: h = (outs, 0)
  A.label(nm("ret"));
  A.i(e_ld32(R14, R12, 4), "outs");
  A.i(e_cmp(RDX, R14));
  A.jcc(Cc::NE, rej, "", true);
  A.i(e_test(RDI, RDI));
  A.jcc(Cc::NE, rej, "", true);
  A.i(e_movI(R15, 1));
  A.jmp(nofall, "", true);
  // falls through to h' = (rdx, rdi): i + 1 < nc and ann(i + 1) in {none, h'}
  A.label(fall);
  A.i(e_aluI(AluOp::Add, R8, 1));
  A.i(e_ld32(R14, R12, 12));
  A.i(e_cmp(R8, R14));
  A.jcc(Cc::AE, rej, "falls off the end", true);
  lookup(A, R8, " (i + 1)");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::E, adv);
  A.i(e_ld32(R14, R10, 4));
  A.i(e_cmp(R14, RDX));
  A.jcc(Cc::NE, rej, "", true);
  A.i(e_ld32(R14, R10, 8));
  A.i(e_cmp(R14, RDI));
  A.jcc(Cc::NE, rej, "", true);
  A.label(adv);
  A.i(e_alu(AluOp::Add, RSI, R15), "next instruction");
  A.jmp(walk, "", true);
  A.label(nofall);
  A.i(e_aluI(AluOp::Add, R8, 1));
  lookup(A, R8, " (i + 1)");
  std::string nz = A.fresh("nfzero");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::E, nz);
  A.i(e_ld32(RDX, R10, 4));
  A.i(e_ld32(RDI, R10, 8));
  A.jmp(adv, "", true);
  A.label(nz);
  A.i(e_zero(RDX));
  A.i(e_zero(RDI));
  A.jmp(adv, "", true);
  // annotations: every (t, d, a) has t < nc and lookup t = (d, a)
  A.label(wdone);
  A.i(e_ld32(R14, R12, 16), "na");
  A.i(e_mov(RDI, R14));
  A.i(e_shlI(RDI, 3));
  A.i(e_shlI(R14, 2));
  A.i(e_alu(AluOp::Add, RDI, R14));
  A.i(e_alu(AluOp::Add, RDI, R9), "rdi = end of the annotations = next word");
  A.i(e_mov(RSI, R9));
  std::string al = P + "_anns", ad = P + "_annsok";
  A.label(al);
  A.i(e_cmp(RSI, RDI));
  A.jcc(Cc::E, ad);
  A.i(e_ld32(R15, RSI, 0), "t");
  A.i(e_ld32(R14, R12, 12));
  A.i(e_cmp(R15, R14));
  A.jcc(Cc::AE, rej, "annotation past the code", true);
  lookup(A, R15, " (ann t)");
  A.i(e_cmp(R10, R11));
  A.jcc(Cc::E, rej, "", true);
  A.i(e_ld32(R14, R10, 4));
  A.i(e_ld32(RDX, RSI, 4));
  A.i(e_cmp(R14, RDX));
  A.jcc(Cc::NE, rej, "conflicting annotations", true);
  A.i(e_ld32(R14, R10, 8));
  A.i(e_ld32(RDX, RSI, 8));
  A.i(e_cmp(R14, RDX));
  A.jcc(Cc::NE, rej, "conflicting annotations", true);
  A.i(e_aluI(AluOp::Add, RSI, 12));
  A.jmp(al, "", true);
  A.label(ad);
  A.i(e_mov(R12, RDI), "next word");
  A.jmp(wl, "", true);
  // verdicts
  A.label(acc);
  A.i(e_movI64(R, kAccept), "pass " + std::to_string(k + 1) + ": ACCEPT");
  A.jmp(nxt, "", true);
  A.label(ovf);
  A.i(e_movI64(R, kOverflow), "pass " + std::to_string(k + 1) + ": OVERFLOW");
  A.jmp(nxt, "", true);
  A.label(rej);
  A.i(e_movI64(R, kReject), "pass " + std::to_string(k + 1) + ": REJECT");
  A.jmp(nxt, "", true);
  A.label(P + "_end");
}

Asm gen_program() {
  Asm A;
  A.raw(std::vector<u8>(kPad0, 0xf4), Ins{.kind = Ins::Kind::Pad0},
        "barrier between ELF header and tripled region");
  A.label("tripled");
  for (int k = 0; k < 3; ++k) gen_pass(A, k);

  // ---- the voter: TMRVoter.voterCode with the park (eb fe) replaced by eb 0c
  A.label("vote");
  std::vector<u8> voter = {
      0x48, 0x39, 0xd8, 0x74, 0x0d, 0x48, 0x39, 0xc8, 0x74, 0x08, 0x48, 0x39, 0xcb,
      0x75, 0x0b, 0x48, 0x89, 0xd8, 0x48, 0x89, 0xc3, 0x48, 0x89, 0xc1, 0xeb, 0x0c,
      0xbf, 0x03, 0x00, 0x00, 0x00, 0xb8, 0x3c, 0x00, 0x00, 0x00, 0x0f, 0x05,
  };
  A.raw(std::move(voter), Ins{.kind = Ins::Kind::Voter},
        "TMRVoter.voterCode, park eb fe -> eb 0c (to guard)");

  // ---- the dual guard
  A.label("guard");
  A.i(e_movI(RDI, 5), "status 5 unless both guards pass");
  A.i(e_movI64(RDX, kNegAcc), "-ACCEPT");
  A.i(e_alu(AluOp::Add, RDX, RAX), "rdx = rax - ACCEPT");
  A.jcc(Cc::NE, "not_acc", "guard 1");
  A.i(e_movI64(RSI, kNegAcc), "-ACCEPT");
  A.i(e_alu(AluOp::Add, RSI, RBX), "rsi = rbx - ACCEPT");
  A.jcc(Cc::NE, "not_acc", "guard 2");
  A.i(e_movI(RAX, 60), "exit(0): rax first,");
  A.i(e_lea(RDI, RBX, -static_cast<i32>(kAccept & 0xffULL)),
      "then status = (rbx - 0x96) mod 256, zero only for ACCEPT");
  A.i(e_syscall());
  A.jmp("rej_exit");
  A.label("not_acc");
  A.i(e_movI(RDI, 5));
  A.i(e_movI64(RDX, kNegOvf), "-OVERFLOW");
  A.i(e_alu(AluOp::Add, RDX, RAX));
  A.jcc(Cc::NE, "rej_exit", "not OVERFLOW");
  A.label("nomem");
  A.i(e_movI(RDI, 6), "exit(6)");
  A.i(e_movI(RAX, 60));
  A.i(e_movI(RDI, 6));
  A.i(e_syscall());
  A.label("rej_exit");
  A.i(e_movI(RDI, 5), "exit(5)");
  A.i(e_movI(RAX, 60));
  A.i(e_movI(RDI, 5));
  A.i(e_syscall());
  A.label("core_end");

  A.raw(std::vector<u8>(kPad1, 0xf4), Ins{.kind = Ins::Kind::Pad1},
        "barrier between tripled region and acquisition");
  A.label("start");

  // ---- acquisition: mmap a buffer, read all of stdin, growing with mremap
  A.i(e_zero(RDI), "mmap(0, 64 KiB, RW, PRIVATE|ANON, -1, 0)");
  A.i(e_movI(RSI, kCap0));
  A.i(e_movI(RDX, 3));
  A.i(e_movI(R10, 0x22));
  A.i(e_movI64(R8, ~0ULL));
  A.i(e_zero(R9));
  A.i(e_movI(RAX, 9));
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::S, "nomem", "refused");
  A.i(e_mov(R14, RAX), "r14 = buffer");
  A.i(e_movI(R12, kCap0), "r12 = capacity");
  A.i(e_zero(R13), "r13 = length");
  A.label("rd");
  A.i(e_cmp(R13, R12));
  A.jcc(Cc::NE, "rd_go", "room left");
  A.i(e_mov(RDI, R14), "mremap(buf, cap, 2 cap, MAYMOVE)");
  A.i(e_mov(RSI, R12));
  A.i(e_mov(RDX, R12));
  A.i(e_shlI(RDX, 1));
  A.i(e_movI(R10, 1));
  A.i(e_movI(RAX, 25));
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::S, "nomem", "refused");
  A.i(e_mov(R14, RAX));
  A.i(e_shlI(R12, 1));
  A.label("rd_go");
  A.i(e_zero(RDI), "read(0, buf + len, cap - len)");
  A.i(e_mov(RSI, R14));
  A.i(e_alu(AluOp::Add, RSI, R13));
  A.i(e_mov(RDX, R12));
  A.i(e_alu(AluOp::Sub, RDX, R13));
  A.i(e_zero(RAX));
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::S, "nomem", "read failed");
  A.jcc(Cc::E, "eof");
  A.i(e_alu(AluOp::Add, R13, RAX));
  A.jmp("rd");

  // ---- triplication: two more copies, then the three descriptors
  A.label("eof");
  for (const auto& [copy_idx, dst] : std::array<std::pair<int, int>, 2>{{{1, R15}, {2, RBP}}}) {
    A.i(e_zero(RDI), "mmap copy " + std::to_string(copy_idx + 1) + ": len + 1 bytes");
    A.i(e_lea(RSI, R13, 1));
    A.i(e_movI(RDX, 3));
    A.i(e_movI(R10, 0x22));
    A.i(e_zero(R9));
    A.i(e_movI(RAX, 9));
    A.i(e_syscall());
    A.i(e_test(RAX, RAX));
    A.jcc(Cc::S, "nomem");
    A.i(e_mov(dst, RAX));
    std::string top = A.fresh("cp_top"), done = A.fresh("cp_done");
    A.i(e_mov(RSI, R14), "copy len bytes");
    A.i(e_mov(RDI, dst));
    A.i(e_mov(R11, R13));
    A.label(top);
    A.i(e_test(R11, R11));
    A.jcc(Cc::E, done);
    A.i(e_ld8(RDX, RSI, 0));
    A.i(e_st8(RDI, 0, RDX));
    A.i(e_aluI(AluOp::Add, RSI, 1));
    A.i(e_aluI(AluOp::Add, RDI, 1));
    A.i(e_aluI(AluOp::Sub, R11, 1));
    A.jmp(top);
    A.label(done);
  }
  constexpr std::array<int, 3> kCopySrcs = {R14, R15, RBP};
  for (std::size_t k = 0; k < 3; ++k) {
    A.i(e_movI(RDI, kDesc[k]), "descriptor " + std::to_string(k + 1));
    A.i(e_st(RDI, 0, kCopySrcs[k]));
    A.i(e_st(RDI, 8, R13));
  }
  A.i(e_movI64(RAX, kReject), "results start at REJECT");
  A.i(e_movI64(RBX, kReject));
  A.i(e_movI64(RCX, kReject));
  A.jmp("tripled");
  return A;
}

std::vector<u8> elf_header(u64 entry, u64 filesz) {
  std::vector<u8> h = {0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  push_le(h, 2, 2);
  push_le(h, 0x3e, 2);
  push_le(h, 1, 4);
  push_le(h, entry, 8);
  push_le(h, 0x40, 8);
  push_le(h, 0, 8);
  push_le(h, 0, 4);
  push_le(h, 64, 2);
  push_le(h, 56, 2);
  push_le(h, 2, 2);
  push_le(h, 0, 2);
  push_le(h, 0, 2);
  push_le(h, 0, 2);
  // PT_LOAD code R+X
  push_le(h, 1, 4);
  push_le(h, 5, 4);
  push_le(h, 0, 8);
  push_le(h, kBase, 8);
  push_le(h, kBase, 8);
  push_le(h, filesz, 8);
  push_le(h, filesz, 8);
  push_le(h, 0x1000, 8);
  // PT_LOAD bss R+W (descriptors)
  push_le(h, 1, 4);
  push_le(h, 6, 4);
  push_le(h, 0, 8);
  push_le(h, kBss, 8);
  push_le(h, kBss, 8);
  push_le(h, 0, 8);
  push_le(h, kBssSize, 8);
  push_le(h, 0x1000, 8);
  require(h.size() == 0xb0, "unexpected ELF header size");
  return h;
}

struct HdrComment {
  std::size_t off;
  std::size_t len;
  const char* comment;
};

constexpr auto kHdrComments = std::to_array<HdrComment>({
    {0x00, 4, "e_ident: magic 0x7f E L F"},
    {0x04, 1, "ELFCLASS64"},
    {0x05, 1, "little endian"},
    {0x06, 1, "EI_VERSION = 1"},
    {0x07, 1, "System V"},
    {0x08, 8, "padding"},
    {0x10, 2, "e_type = ET_EXEC"},
    {0x12, 2, "e_machine = EM_X86_64"},
    {0x14, 4, "e_version = 1"},
    {0x18, 8, "e_entry"},
    {0x20, 8, "e_phoff = 0x40"},
    {0x28, 8, "e_shoff = 0"},
    {0x30, 4, "e_flags = 0"},
    {0x34, 2, "e_ehsize = 64"},
    {0x36, 2, "e_phentsize = 56"},
    {0x38, 2, "e_phnum = 2"},
    {0x3a, 2, "e_shentsize = 0"},
    {0x3c, 2, "e_shnum = 0"},
    {0x3e, 2, "e_shstrndx = 0"},
    {0x40, 4, "PT_LOAD (code)"},
    {0x44, 4, "PF_R | PF_X"},
    {0x48, 8, "p_offset = 0"},
    {0x50, 8, "p_vaddr = 0x400000"},
    {0x58, 8, "p_paddr"},
    {0x60, 8, "p_filesz = whole file"},
    {0x68, 8, "p_memsz"},
    {0x70, 8, "p_align = 0x1000"},
    {0x78, 4, "PT_LOAD (descriptors)"},
    {0x7c, 4, "PF_R | PF_W"},
    {0x80, 8, "p_offset = 0"},
    {0x88, 8, "p_vaddr = 0x600000"},
    {0x90, 8, "p_paddr"},
    {0x98, 8, "p_filesz = 0"},
    {0xa0, 8, "p_memsz = 0x3000 (three pages)"},
    {0xa8, 8, "p_align = 0x1000"},
});

std::string ins_str(const Ins& ins) {
  switch (ins.kind) {
    case Ins::Kind::MovI:
      return std::string("mov ") + kRegName[ins.r0] + ", " + hex_0x(ins.imm);
    case Ins::Kind::Mov:
      return std::string("mov ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::Mov32:
      return std::string("mov32 ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::Alu:
      return std::string(alu_name(ins.alu_op)) + " " + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::AluI:
      return std::string(alu_name(ins.alu_op)) + " " + kRegName[ins.r0] + ", " + hex_0x(ins.imm);
    case Ins::Kind::CmpI:
      return std::string("cmp ") + kRegName[ins.r0] + ", " + hex_0x(ins.imm);
    case Ins::Kind::Cmp:
      return std::string("cmp ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::Test:
      return std::string("test ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::ShlI:
      return std::string("shl ") + kRegName[ins.r0] + ", " + std::to_string(ins.imm);
    case Ins::Kind::ShrI:
      return std::string("shr ") + kRegName[ins.r0] + ", " + std::to_string(ins.imm);
    case Ins::Kind::Ld:
      return std::string("ld ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] +
             signed_hex(ins.imm) + "]";
    case Ins::Kind::Ld32:
      return std::string("ld32 ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] +
             signed_hex(ins.imm) + "]";
    case Ins::Kind::Ld8:
      return std::string("ld8 ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] +
             signed_hex(ins.imm) + "]";
    case Ins::Kind::Lea:
      return std::string("lea ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] +
             signed_hex(ins.imm) + "]";
    case Ins::Kind::St:
      return std::string("st [") + kRegName[ins.r0] + signed_hex(ins.imm) + "], " +
             kRegName[ins.r1];
    case Ins::Kind::St8:
      return std::string("st8 [") + kRegName[ins.r0] + signed_hex(ins.imm) + "], " +
             kRegName[ins.r1];
    case Ins::Kind::Jmp:
      return "jmp " + hex_0x(ins.imm);
    case Ins::Kind::Jcc:
      return std::string("j") + cc_name(ins.r0) + " " + hex_0x(ins.imm);
    case Ins::Kind::Syscall:
      return "syscall";
    case Ins::Kind::PushI:
      return "pushI";
    case Ins::Kind::Push:
      return "push";
    case Ins::Kind::Pop:
      return "pop";
    case Ins::Kind::Call:
      return "call";
    case Ins::Kind::Ret:
      return "ret";
    case Ins::Kind::Jtab:
      return "jtab";
    case Ins::Kind::Pad0:
      return "pad0";
    case Ins::Kind::Pad1:
      return "pad1";
    case Ins::Kind::Voter:
      return "voter";
  }
  fail_msg("unknown Ins::Kind");
}

std::string lean_ins(const Ins& ins) {
  switch (ins.kind) {
    case Ins::Kind::MovI:
      return ".movI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm);
    case Ins::Kind::Mov:
      return ".mov " + std::to_string(ins.r0) + " " + std::to_string(ins.r1);
    case Ins::Kind::Mov32:
      return ".mov32 " + std::to_string(ins.r0) + " " + std::to_string(ins.r1);
    case Ins::Kind::Alu:
      return std::string(".alu .") + alu_name(ins.alu_op) + " " + std::to_string(ins.r0) + " " +
             std::to_string(ins.r1);
    case Ins::Kind::AluI:
      return std::string(".aluI .") + alu_name(ins.alu_op) + " " + std::to_string(ins.r0) + " " +
             std::to_string(ins.imm);
    case Ins::Kind::CmpI:
      return ".cmpI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm);
    case Ins::Kind::Cmp:
      return ".cmp " + std::to_string(ins.r0) + " " + std::to_string(ins.r1);
    case Ins::Kind::Test:
      return ".test " + std::to_string(ins.r0) + " " + std::to_string(ins.r1);
    case Ins::Kind::ShlI:
      return ".shlI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm);
    case Ins::Kind::ShrI:
      return ".shrI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm);
    case Ins::Kind::Ld:
      return ".ld " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " +
             std::to_string(ins.imm);
    case Ins::Kind::Ld32:
      return ".ld32 " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " +
             std::to_string(ins.imm);
    case Ins::Kind::Ld8:
      return ".ld8 " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " +
             std::to_string(ins.imm);
    case Ins::Kind::St:
      return ".st " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + " " +
             std::to_string(ins.r1);
    case Ins::Kind::St8:
      return ".st8 " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + " " +
             std::to_string(ins.r1);
    case Ins::Kind::Lea:
      return ".lea " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " +
             std::to_string(ins.imm);
    case Ins::Kind::Jmp:
      return ".jmp " + std::to_string(ins.imm);
    case Ins::Kind::Jcc:
      return ".jcc " + std::to_string(ins.r0) + " " + std::to_string(ins.imm);
    case Ins::Kind::Syscall:
      return ".syscall";
    default:
      fail_msg("unsupported Ins::Kind in lean_ins");
  }
}

// ------------------------------------------------------- Lean templates

struct TemplateSpec {
  const char* name;
  std::vector<const char*> reg_params;
  std::vector<const char*> addr_params;
};

const std::vector<TemplateSpec>& template_specs() {
  static const std::vector<TemplateSpec> kSpecs = {
      {"skipcode", {"p"}, {"rej"}},
      {"skipdata", {}, {"rej"}},
      {"skipwords", {}, {"rej"}},
      {"chk", {"r"}, {"ovf"}},
      {"lookup", {"t"}, {}},
      {"eff", {}, {"tgt"}},
  };
  return kSpecs;
}

const TemplateSpec& spec_for(std::string_view name) {
  for (const auto& sp : template_specs()) {
    if (name == sp.name) return sp;
  }
  fail_msg("unknown template name");
}

std::string render_ins(const Ins& ins, u64 L, u64 size,
                       const std::map<u64, std::string>& aparams,
                       const std::map<std::pair<int, int>, std::string>& rparams,
                       const std::map<u64, std::string>& iparams,
                       std::string_view Lname = "L") {
  auto fmt_reg = [&](int j, int v) -> std::string {
    auto it = rparams.find({j, v});
    return it != rparams.end() ? it->second : std::to_string(v);
  };
  auto fmt_imm = [&](u64 v) -> std::string {
    auto it = iparams.find(v);
    return it != iparams.end() ? it->second : std::to_string(v);
  };
  auto fmt_addr = [&](u64 v) -> std::string {
    auto ap = aparams.find(v);
    if (v >= L && v <= L + size && ap == aparams.end()) {
      if (v != L) return "(" + std::string(Lname) + " + " + std::to_string(v - L) + ")";
      return std::string(Lname);
    }
    if (ap != aparams.end()) return ap->second;
    fail_msg("address outside template instance");
  };

  switch (ins.kind) {
    case Ins::Kind::MovI:
      return ".movI " + fmt_reg(0, ins.r0) + " " + fmt_imm(ins.imm);
    case Ins::Kind::Mov:
      return ".mov " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1);
    case Ins::Kind::Mov32:
      return ".mov32 " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1);
    case Ins::Kind::Alu:
      return std::string(".alu .") + alu_name(ins.alu_op) + " " + fmt_reg(1, ins.r0) + " " +
             fmt_reg(2, ins.r1);
    case Ins::Kind::AluI:
      return std::string(".aluI .") + alu_name(ins.alu_op) + " " + fmt_reg(1, ins.r0) + " " +
             fmt_imm(ins.imm);
    case Ins::Kind::CmpI:
      return ".cmpI " + fmt_reg(0, ins.r0) + " " + fmt_imm(ins.imm);
    case Ins::Kind::Cmp:
      return ".cmp " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1);
    case Ins::Kind::Test:
      return ".test " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1);
    case Ins::Kind::ShlI:
      return ".shlI " + fmt_reg(0, ins.r0) + " " + fmt_imm(ins.imm);
    case Ins::Kind::ShrI:
      return ".shrI " + fmt_reg(0, ins.r0) + " " + fmt_imm(ins.imm);
    case Ins::Kind::Ld:
      return ".ld " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " + fmt_imm(ins.imm);
    case Ins::Kind::Ld32:
      return ".ld32 " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " + fmt_imm(ins.imm);
    case Ins::Kind::Ld8:
      return ".ld8 " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " + fmt_imm(ins.imm);
    case Ins::Kind::Lea:
      return ".lea " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " + fmt_imm(ins.imm);
    case Ins::Kind::St:
      return ".st " + fmt_reg(0, ins.r0) + " " + fmt_imm(ins.imm) + " " + fmt_reg(2, ins.r1);
    case Ins::Kind::St8:
      return ".st8 " + fmt_reg(0, ins.r0) + " " + fmt_imm(ins.imm) + " " + fmt_reg(2, ins.r1);
    case Ins::Kind::Jmp:
      return ".jmp " + fmt_addr(ins.imm);
    case Ins::Kind::Jcc:
      return ".jcc " + fmt_imm(static_cast<u64>(ins.r0)) + " " + fmt_addr(ins.imm);
    case Ins::Kind::Syscall:
      return ".syscall";
    default:
      fail_msg("unsupported Ins::Kind in render_ins");
  }
}

std::string join_strings(const std::vector<std::string>& v, std::string_view sep) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i > 0) out += sep;
    out += v[i];
  }
  return out;
}

std::string format_groups(const std::vector<std::pair<bool, std::string>>& items) {
  std::vector<std::string> groups;
  std::vector<std::string> cur;
  for (const auto& [is_ins, text] : items) {
    if (is_ins) {
      cur.push_back(text);
    } else {
      if (!cur.empty()) {
        groups.push_back("[" + join_strings(cur, ", ") + "]");
        cur.clear();
      }
      groups.push_back(text);
    }
  }
  if (!cur.empty()) {
    groups.push_back("[" + join_strings(cur, ", ") + "]");
  }
  return join_strings(groups, " ++\n  ");
}

std::string emit_templates(const Asm& A, const std::vector<OutEntry>& out,
                           const std::map<std::string, u64>& labels) {
  std::map<std::string, std::vector<const MacroRec*>> insts;
  for (const auto& m : A.macros) {
    insts[m.name].push_back(&m);
  }

  std::string result;
  for (const auto& sp : template_specs()) {
    const auto& ilist = insts[sp.name];
    require(!ilist.empty(), std::string("missing macro instances for ") + sp.name);

    std::vector<std::string> sig_parts = {"(L : Nat)"};
    for (const char* p : sp.reg_params) sig_parts.push_back(std::string("(") + p + " : Nat)");
    for (const char* p : sp.addr_params) sig_parts.push_back(std::string("(") + p + " : Nat)");
    std::string sig = join_strings(sig_parts, " ");

    auto body = [&](const MacroRec& inst) -> std::string {
      u64 L = A.idx_addr[inst.start_idx];
      u64 size = A.idx_addr[inst.end_idx] - L;
      std::map<u64, std::string> aparams;
      for (const char* p : sp.addr_params) {
        aparams[labels.at(inst.addr_params.at(p))] = p;
      }

      struct TermItem {
        bool is_ins;
        std::size_t idx;
        std::string text;
      };
      std::vector<TermItem> ts;
      std::set<std::size_t> covered;

      for (const auto& m : A.macros) {
        if (inst.start_idx <= m.start_idx && m.end_idx <= inst.end_idx &&
            (m.start_idx != inst.start_idx || m.end_idx != inst.end_idx)) {
          u64 mL = A.idx_addr[m.start_idx];
          const auto& mspec = spec_for(m.name);
          std::vector<std::string> args;
          for (const char* p : mspec.reg_params) {
            int v = m.reg_params.at(p);
            std::string arg;
            for (const char* q : sp.reg_params) {
              if (inst.reg_params.at(q) == v) {
                arg = q;
                break;
              }
            }
            if (arg.empty()) arg = std::to_string(v);
            args.push_back(std::move(arg));
          }
          for (const char* p : mspec.addr_params) {
            u64 v = labels.at(m.addr_params.at(p));
            auto ap = aparams.find(v);
            if (ap != aparams.end()) {
              args.push_back(ap->second);
            } else {
              args.push_back("(L + " + std::to_string(v - L) + ")");
            }
          }
          for (std::size_t k = m.start_idx; k < m.end_idx; ++k) covered.insert(k);
          ts.push_back(TermItem{
              false,
              m.start_idx,
              m.name + "T (L + " + std::to_string(mL - L) + ") " + join_strings(args, " "),
          });
        }
      }

      for (const auto& e : out) {
        if (e.idx >= inst.start_idx && e.idx < inst.end_idx && !covered.contains(e.idx)) {
          std::map<std::pair<int, int>, std::string> rparams;
          auto map_reg = [&](int j, int reg) {
            for (const char* q : sp.reg_params) {
              if (inst.reg_params.at(q) == reg) rparams[{j, reg}] = q;
            }
          };
          switch (e.ins.kind) {
            case Ins::Kind::MovI:
            case Ins::Kind::CmpI:
            case Ins::Kind::ShlI:
            case Ins::Kind::ShrI:
              map_reg(0, e.ins.r0);
              break;
            case Ins::Kind::Mov:
            case Ins::Kind::Mov32:
            case Ins::Kind::Cmp:
            case Ins::Kind::Test:
            case Ins::Kind::Ld:
            case Ins::Kind::Ld32:
            case Ins::Kind::Ld8:
            case Ins::Kind::Lea:
              map_reg(0, e.ins.r0);
              map_reg(1, e.ins.r1);
              break;
            case Ins::Kind::St:
            case Ins::Kind::St8:
              map_reg(0, e.ins.r0);
              map_reg(2, e.ins.r1);
              break;
            case Ins::Kind::Alu:
              map_reg(1, e.ins.r0);
              map_reg(2, e.ins.r1);
              break;
            case Ins::Kind::AluI:
              map_reg(1, e.ins.r0);
              break;
            default:
              break;
          }
          ts.push_back(TermItem{
              true,
              e.idx,
              "(L + " + std::to_string(e.addr - L) + ", " +
                  render_ins(e.ins, L, size, aparams, rparams, {}) + ", " +
                  std::to_string(e.bytes.size()) + ")",
          });
        }
      }

      std::stable_sort(ts.begin(), ts.end(),
                       [](const TermItem& x, const TermItem& y) { return x.idx < y.idx; });
      std::vector<std::pair<bool, std::string>> items;
      items.reserve(ts.size());
      for (const auto& t : ts) items.emplace_back(t.is_ins, t.text);
      return format_groups(items);
    };

    std::string b0 = body(*ilist[0]);
    for (std::size_t idx = 1; idx < ilist.size(); ++idx) {
      std::string bi = body(*ilist[idx]);
      require(bi == b0, std::string("template mismatch in ") + sp.name);
    }
    result += "/-- The " + std::string(sp.name) + " macro at L (" +
              std::to_string(ilist.size()) + " instances). -/\n";
    result += "def " + std::string(sp.name) + "T " + sig + " : List (Nat × Ins × Nat) :=\n  " +
              b0 + "\n\n";
  }

  // The pass, with its macros as template calls; B its address, D its descriptor,
  // R its result register.
  u64 B = labels.at("p1");
  u64 size = labels.at("p1_end") - B;
  auto find_label_idx = [&](std::string_view name) -> std::size_t {
    for (std::size_t i = 0; i < A.items.size(); ++i) {
      if (A.items[i].kind == AsmItem::Kind::Label && A.items[i].label == name) return i;
    }
    fail_msg("missing label index");
  };
  std::size_t st_idx = find_label_idx("p1");
  std::size_t en_idx = find_label_idx("p1_end");

  std::vector<const MacroRec*> top;
  for (std::size_t i = 0; i < A.macros.size(); ++i) {
    const auto& m = A.macros[i];
    if (st_idx <= m.start_idx && m.end_idx <= en_idx) {
      bool nested = false;
      for (std::size_t j = 0; j < A.macros.size(); ++j) {
        if (i == j) continue;
        const auto& n = A.macros[j];
        if (n.start_idx <= m.start_idx && m.end_idx <= n.end_idx &&
            (n.start_idx != m.start_idx || n.end_idx != m.end_idx)) {
          nested = true;
          break;
        }
      }
      if (!nested) top.push_back(&m);
    }
  }

  struct PassItem {
    bool is_ins;
    std::size_t idx;
    std::string text;
  };
  std::vector<PassItem> ts;
  std::set<std::size_t> covered;

  for (const auto* m : top) {
    u64 mL = A.idx_addr[m->start_idx];
    const auto& mspec = spec_for(m->name);
    std::vector<std::string> args;
    for (const char* p : mspec.reg_params) {
      args.push_back(std::to_string(m->reg_params.at(p)));
    }
    for (const char* p : mspec.addr_params) {
      args.push_back("(B + " + std::to_string(labels.at(m->addr_params.at(p)) - B) + ")");
    }
    for (std::size_t k = m->start_idx; k < m->end_idx; ++k) covered.insert(k);
    ts.push_back(PassItem{
        false,
        m->start_idx,
        m->name + "T (B + " + std::to_string(mL - B) + ") " + join_strings(args, " "),
    });
  }

  for (const auto& e : out) {
    if (st_idx <= e.idx && e.idx < en_idx && !covered.contains(e.idx)) {
      std::map<std::pair<int, int>, std::string> rparams;
      std::map<u64, std::string> iparams = {{kDesc[0], "D"}};
      if (e.ins.kind == Ins::Kind::MovI) {
        if (e.ins.imm == kAccept) {
          rparams[{0, e.ins.r0}] = "R";
          iparams[kAccept] = "ACCEPT";
        } else if (e.ins.imm == kReject) {
          rparams[{0, e.ins.r0}] = "R";
          iparams[kReject] = "REJECT";
        } else if (e.ins.imm == kOverflow) {
          rparams[{0, e.ins.r0}] = "R";
          iparams[kOverflow] = "OVERFLOW";
        }
      }
      ts.push_back(PassItem{
          true,
          e.idx,
          "(B + " + std::to_string(e.addr - B) + ", " +
              render_ins(e.ins, B, size, {}, rparams, iparams, "B") + ", " +
              std::to_string(e.bytes.size()) + ")",
      });
    }
  }

  std::stable_sort(ts.begin(), ts.end(),
                   [](const PassItem& x, const PassItem& y) { return x.idx < y.idx; });
  std::vector<std::pair<bool, std::string>> items;
  items.reserve(ts.size());
  for (const auto& t : ts) items.emplace_back(t.is_ins, t.text);
  result += "/-- Pass code at B (" + std::to_string(size) +
            " bytes), descriptor D, result register R. -/\n";
  result += "def passCode (B D R : Nat) : List (Nat × Ins × Nat) :=\n  " +
            format_groups(items) + "\n\n";
  result += "def passSize : Nat := " + std::to_string(size) + "\n\n";
  return result;
}

constexpr const char* kLeanHead =
    "/-\n"
    "Generated by bootstrap/model/spr1_check_gen.cpp from the listing\n"
    "bootstrap/model/spr1_check.hex; do not edit. `spr1Bytes` is the file\n"
    "bootstrap/model/spr1_check, byte for byte, and `spr1Table` the instruction at\n"
    "every address of its code (checked against `X86Dec.decode` in\n"
    "Bootstrap/SporosCheckBin.lean).\n"
    "-/\n"
    "import Bootstrap.X86Int\n\n"
    "namespace SporosCheckBin\n\n"
    "open X86Int\n\n";

// Verbatim frozen header of the committed b887f1b51e listing
// (bootstrap/model/spr1_check.hex) required for byte-for-byte cmp -s parity;
// spr1_check itself is an independent second check, not Root A or Root A's
// trust base.
std::string format_listing_head(u64 size, u64 entry, u64 tripled, u64 vote, u64 guard,
                                u64 overflow) {
  std::string s;
  s += "# spr1_check: the checker of Sporos bytecode, stage 1 of bootstrap root A.\n";
  s += "#\n";
  s += "# What it does: reads a Sporos bytecode file (formal/Sporos/Bytecode.lean) from\n";
  s += "# standard input and checks it as Sporos.decode and Sporos.checkProg\n";
  s += "# (formal/Sporos/Check.lean) do, three times, on three copies of the input.\n";
  s += "#\n";
  s += "# Exit status: 0 when the file decodes and passes the checker; 5 when it is\n";
  s += "# malformed or ill-typed; 6 when memory is refused or a stack height reaches\n";
  s += "# 2^64; 3 when the three passes all disagree. Every non-zero status is at\n";
  s += "# Hamming distance 2 or more from 0.\n";
  s += "#\n";
  s += "# Structure (addresses are 0x400000 + file offset):\n";
  s += "#   start   (" + hex_0x(entry) +
       ")  mmap a 64 KiB buffer, read all of stdin, doubling it\n";
  s += "#                        with mremap; triplicate the input into two more\n";
  s += "#                        mappings; descriptors (base, N) at 0x600000,\n";
  s += "#                        0x601000, 0x602000; rax = rbx = rcx = REJECT.\n";
  s += "#   tripled (" + hex_0x(tripled) +
       ")  pass 1 (result in rax), pass 2 (rbx), pass 3 (rcx).\n";
  s += "#                        Each pass reads only its own copy and descriptor, makes\n";
  s += "#                        no store, call, push or syscall, and writes only its\n";
  s += "#                        result register: ACCEPT 0x5acce970a55a3c96, REJECT\n";
  s += "#                        0xdeadbeef11110001 or OVERFLOW " + hex_0x(overflow) + ".\n";
  s += "#   vote    (" + hex_0x(vote) +
       ")  the 38-byte voter of Bootstrap/TMRVoter.lean, with its\n";
  s += "#                        park (eb fe) replaced by eb 0c, a jump to the guard.\n";
  s += "#   guard   (" + hex_0x(guard) +
       ")  exit(0) only when rax - ACCEPT and rbx - ACCEPT are both\n";
  s += "#                        zero, tested by two conditional branches; the exit\n";
  s += "#                        status is computed from the two differences.\n";
  s += "#\n";
  s += "# The bytes 0f 85 (jne rel32, one bit from syscall 0f 05) never occur, and\n";
  s += "# 0f 05 occurs only at the syscalls.\n";
  s += "#\n";
  s += "# Total " + std::to_string(size) + " bytes.\n";
  return s;
}

void write_bytes_file(const std::filesystem::path& path, const std::vector<u8>& data) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  require(ofs.good(), "failed to open for write: " + path.string());
  ofs.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
  require(ofs.good(), "failed to write: " + path.string());
}

void write_text_file(const std::filesystem::path& path, const std::string& data) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  require(ofs.good(), "failed to open for write: " + path.string());
  ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
  require(ofs.good(), "failed to write: " + path.string());
}

bool keep_lean_label(const std::string& name) {
  if (name.find('_') == std::string::npos) return true;
  if (name.rfind("p1_", 0) == 0 || name.rfind("p2_", 0) == 0 || name.rfind("p3_", 0) == 0) {
    return true;
  }
  return name == "not_acc" || name == "rej_exit" || name == "rd_go" || name == "core_end";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path outdir = (argc > 1) ? argv[1] : ".";
  Asm A = gen_program();
  constexpr u64 kHdrLen = 0xb0;
  auto res = A.assemble(kBase + kHdrLen);
  const auto& out = res.out;
  const auto& labels = res.labels;
  const auto& label_order = res.label_order;

  std::vector<u8> code;
  for (const auto& e : out) {
    require(e.addr == kBase + kHdrLen + code.size(), "non-contiguous assembled code");
    code.insert(code.end(), e.bytes.begin(), e.bytes.end());
  }
  u64 filesz = kHdrLen + code.size();
  std::vector<u8> image = elf_header(labels.at("start"), filesz);
  image.insert(image.end(), code.begin(), code.end());

  // Every instruction decodes to what was meant.
  struct TableEntry {
    u64 addr;
    Ins ins;
    std::size_t len;
  };
  std::vector<TableEntry> table;
  for (const auto& e : out) {
    if (e.ins.kind == Ins::Kind::Pad0 || e.ins.kind == Ins::Kind::Pad1) continue;
    if (e.ins.kind == Ins::Kind::Voter) {
      std::size_t off = 0;
      while (off < e.bytes.size()) {
        ImageSlice sl{image, (e.addr - kBase) + off};
        auto d = decode(sl, e.addr + off);
        require(d.has_value(), "failed to decode voter instruction");
        table.push_back(TableEntry{e.addr + off, d->ins, d->len});
        off += d->len;
      }
      continue;
    }
    ImageSlice sl{image, e.addr - kBase};
    auto d = decode(sl, e.addr);
    require(d.has_value() && d->ins == e.ins && d->len == e.bytes.size(),
            "decoded instruction mismatch at " + hex_0x(e.addr));
    table.push_back(TableEntry{e.addr, e.ins, e.bytes.size()});
  }

  // No 0f 85 anywhere, and 0f 05 only at the syscalls.
  std::set<u64> sys_at;
  for (const auto& te : table) {
    if (te.ins.kind == Ins::Kind::Syscall) sys_at.insert(te.addr);
  }
  for (std::size_t j = 0; j + 1 < image.size(); ++j) {
    if (image[j] == 0x0f && image[j + 1] == 0x05) {
      require(sys_at.contains(kBase + j), "unexpected 0f 05 at " + hex_0x(kBase + j));
    }
    require(!(image[j] == 0x0f && image[j + 1] == 0x85),
            "forbidden 0f 85 at " + hex_0x(kBase + j));
  }

  std::filesystem::create_directories(outdir);
  write_bytes_file(outdir / "spr1_check", image);

  std::map<u64, std::vector<std::string>> lab_at;
  for (const auto& [k, v] : label_order) {
    lab_at[v].push_back(k);
  }

  // Write spr1_check.hex
  {
    std::string hex_out;
    hex_out += format_listing_head(filesz, labels.at("start"), labels.at("tripled"),
                                   labels.at("vote"), labels.at("guard"), kOverflow);
    hex_out += "\n# ---------------------------------------------------------------- ELF header\n";
    auto emit_hdr_line = [&](const HdrComment& hc) {
      if (hc.off == 0x40) {
        hex_out +=
            "\n# ----------------------------------------------------------- program headers\n";
      }
      std::vector<u8> slice(image.begin() + static_cast<std::ptrdiff_t>(hc.off),
                            image.begin() + static_cast<std::ptrdiff_t>(hc.off + hc.len));
      hex_out += pad_right(hex_bytes(slice), 32) + "# " + raw_hex(hc.off, 3) + "  " +
                 hc.comment + "\n";
    };
    for (const auto& hc : kHdrComments) emit_hdr_line(hc);

    hex_out += "\n# --------------------------------------------------------------------- code\n";
    hex_out += "# Offsets on the left are file offsets; address = 0x400000 + offset.\n";

    for (const auto& e : out) {
      auto lit = lab_at.find(e.addr);
      if (lit != lab_at.end()) {
        for (const auto& l : lit->second) {
          if (l != "core_end") hex_out += "\n# " + l + ":\n";
        }
      }
      if (e.ins.kind == Ins::Kind::Pad0 || e.ins.kind == Ins::Kind::Pad1) {
        hex_out += "# " + ins_str(e.ins) + " (" + std::to_string(e.bytes.size()) +
                   " bytes of 0xf4)\n";
        for (std::size_t j = 0; j < e.bytes.size(); j += 16) {
          std::size_t end = std::min(j + 16, e.bytes.size());
          std::vector<u8> seg(e.bytes.begin() + static_cast<std::ptrdiff_t>(j),
                              e.bytes.begin() + static_cast<std::ptrdiff_t>(end));
          hex_out += hex_bytes(seg) + "\n";
        }
        continue;
      }
      if (e.ins.kind == Ins::Kind::Voter) {
        hex_out += "# " + e.comment + "\n";
        std::size_t off = 0;
        while (off < e.bytes.size()) {
          ImageSlice sl{image, (e.addr - kBase) + off};
          auto d = decode(sl, e.addr + off);
          require(d.has_value(), "voter decode failed");
          std::vector<u8> seg(e.bytes.begin() + static_cast<std::ptrdiff_t>(off),
                              e.bytes.begin() + static_cast<std::ptrdiff_t>(off + d->len));
          hex_out += pad_right(hex_bytes(seg), 32) + "# " +
                     raw_hex((e.addr - kBase) + off, 3) + "  " + ins_str(d->ins) + "\n";
          off += d->len;
        }
        continue;
      }
      std::string txt = ins_str(e.ins);
      hex_out += pad_right(hex_bytes(e.bytes), 32) + "# " + raw_hex(e.addr - kBase, 3) + "  " +
                 txt + (e.comment.empty() ? "" : ("   " + e.comment)) + "\n";
    }
    write_text_file(outdir / "spr1_check.hex", hex_out);
  }

  // Write symbols.txt (stable sort by address, preserving declaration order on ties)
  std::vector<std::pair<std::string, u64>> sorted_labels = label_order;
  std::stable_sort(sorted_labels.begin(), sorted_labels.end(),
                   [](const auto& a, const auto& b) { return a.second < b.second; });
  {
    std::string sym_out;
    for (const auto& [k, v] : sorted_labels) {
      sym_out += hex_0x(v) + " " + k + "\n";
    }
    write_text_file(outdir / "symbols.txt", sym_out);
  }

  // Write SporosCheckBinData.lean
  {
    std::size_t core_start = static_cast<std::size_t>(labels.at("tripled") - kBase);
    std::size_t core_end_off = static_cast<std::size_t>(labels.at("core_end") - kBase);
    std::size_t init_start = static_cast<std::size_t>(labels.at("start") - kBase);
    std::vector<u8> hdr_bytes(image.begin(), image.begin() + static_cast<std::ptrdiff_t>(kHdrLen));
    std::vector<u8> core_bytes(image.begin() + static_cast<std::ptrdiff_t>(core_start),
                               image.begin() + static_cast<std::ptrdiff_t>(core_end_off));
    std::vector<u8> init_bytes(image.begin() + static_cast<std::ptrdiff_t>(init_start),
                               image.end());

    std::vector<TableEntry> core_table, init_table;
    for (const auto& te : table) {
      if (te.addr < labels.at("core_end")) core_table.push_back(te);
      if (te.addr >= labels.at("start")) init_table.push_back(te);
    }

    auto fmt_byte_list = [](const std::vector<u8>& bs, std::size_t beg, std::size_t end) {
      std::string out_s = "[";
      for (std::size_t i = beg; i < end; ++i) {
        if (i > beg) out_s += ", ";
        out_s += std::to_string(bs[i]);
      }
      out_s += "]";
      return out_s;
    };

    std::string lean_out = kLeanHead;
    lean_out += "def hdrBytes : List Nat := " + fmt_byte_list(hdr_bytes, 0, hdr_bytes.size()) + "\n";

    auto emit_byte_chunks = [&](std::string_view prefix, std::string_view total_name,
                                const std::vector<u8>& bs) {
      std::vector<std::string> names;
      for (std::size_t i = 0, j = 0; i < bs.size(); i += 256, ++j) {
        std::size_t end = std::min(i + 256, bs.size());
        std::string nm = std::string(prefix) + "_" + std::to_string(j);
        lean_out += "def " + nm + " : List Nat := " + fmt_byte_list(bs, i, end) + "\n";
        names.push_back(std::move(nm));
      }
      lean_out += "def " + std::string(total_name) + " : List Nat :=\n  " +
                  join_strings(names, " ++\n  ") + "\n";
    };
    emit_byte_chunks("core", "coreBytes", core_bytes);
    emit_byte_chunks("init", "initBytes", init_bytes);
    lean_out += "\n/-- bootstrap/model/spr1_check -/\ndef spr1Bytes : List Nat :=\n  hdrBytes ++ "
                "List.replicate " +
                std::to_string(kPad0) + " 0xf4 ++ coreBytes ++ List.replicate " +
                std::to_string(kPad1) + " 0xf4 ++ initBytes\n\n";

    auto emit_tbl = [&](std::string_view name, const std::vector<TableEntry>& tbl) {
      std::vector<std::string> names;
      for (std::size_t i = 0, j = 0; i < tbl.size(); i += 64, ++j) {
        std::size_t end = std::min(i + 64, tbl.size());
        std::string nm = std::string(name) + "_" + std::to_string(j);
        lean_out += "def " + nm + " : List (Nat × Ins × Nat) := [\n";
        for (std::size_t k = i; k < end; ++k) {
          if (k > i) lean_out += ",\n";
          lean_out += "  (" + std::to_string(tbl[k].addr) + ", " + lean_ins(tbl[k].ins) + ", " +
                      std::to_string(tbl[k].len) + ")";
        }
        lean_out += "]\n";
        names.push_back(std::move(nm));
      }
      lean_out += "def " + std::string(name) + " : List (Nat × Ins × Nat) :=\n  " +
                  join_strings(names, " ++\n  ") + "\n\n";
    };
    emit_tbl("coreTable", core_table);
    emit_tbl("initTable", init_table);

    lean_out += "/-- Every instruction of the code, at its address, with its length. -/\n";
    lean_out += "def spr1Table : List (Nat × Ins × Nat) := coreTable ++ initTable\n\n";
    lean_out += "def ACCEPT : Nat := " + hex_0x(kAccept) + "\n";
    lean_out += "def REJECT : Nat := " + hex_0x(kReject) + "\n";
    lean_out += "def OVERFLOW : Nat := " + hex_0x(kOverflow) + "\n\n";
    lean_out += emit_templates(A, out, labels);
    lean_out += "/-- Addresses named in the listing. -/\n";
    for (const auto& [k, v] : sorted_labels) {
      if (keep_lean_label(k)) {
        lean_out += "def L_" + k + " : Nat := " + hex_0x(v) + "\n";
      }
    }
    lean_out += "\nend SporosCheckBin\n";
    write_text_file(outdir / "SporosCheckBinData.lean", lean_out);
  }

  std::printf("size %llu bytes, %zu instructions, entry %s, tripled %s, vote %s, guard %s\n",
              static_cast<unsigned long long>(filesz), table.size(),
              hex_0x(labels.at("start")).c_str(), hex_0x(labels.at("tripled")).c_str(),
              hex_0x(labels.at("vote")).c_str(), hex_0x(labels.at("guard")).c_str());
  return 0;
}
