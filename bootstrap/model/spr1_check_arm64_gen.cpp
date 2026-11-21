// Generator of the spr1_check_arm64 Format C (.bin) and Format A (.hex)
// listings, static EM_AARCH64 ELF64 binary, and Lean 4 data module.
//
// spr1_check_arm64 reads a Sporos bytecode file (formal/Sporos/Bytecode.lean)
// on stdin and exits 0 iff it decodes to a program that passes Sporos.checkProg
// (formal/Sporos/Check.lean); 5 on a malformed or ill-typed file; 6 when memory
// is refused or a stack height reaches 2^48; 3 when the three redundant passes
// all disagree.
//
// Note: Root A is MUSE's hand-written checker
// (bootstrap/roots/sporos/sporos-stage1.hex). spr1_check_arm64 is an
// independent second check of stage 1, not Root A or Root A's trust base.
//
// Both listings assemble byte-for-byte to bootstrap/model/spr1_check_arm64:
//   - bootstrap/roots/c/seed-c < spr1_check_arm64.bin == spr1_check_arm64
//   - bootstrap/roots/a/seed-a < spr1_check_arm64.hex == spr1_check_arm64
// Every instruction emitted here is re-decoded by an embedded transcription of
// Bootstrap/AArch64Phys.decodeWord and compared with the intended instruction.
//
//   spr1_check_arm64_gen [MODEL_OUTDIR [LEAN_OUTDIR]]
//       writes spr1_check_arm64, spr1_check_arm64.bin, spr1_check_arm64.hex into
//       MODEL_OUTDIR (default "."), and SporosCheckArm64BinData.lean into
//       LEAN_OUTDIR (default MODEL_OUTDIR).

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
  std::fprintf(stderr, "spr1_check_arm64_gen: %.*s\n", static_cast<int>(msg.size()), msg.data());
  std::abort();
}

void require(bool cond, std::string_view msg) {
  if (!cond) fail_msg(msg);
}

constexpr u64 kAccept = 0x5acce970a55a3c96ULL;
constexpr u64 kReject = 0xdeadbeef11110001ULL;
constexpr u64 kOverflow = 0x0bad0f1e33c3c3a5ULL;
constexpr int kHmaxShift = 48;
constexpr u64 kPad0Bytes = 4176;   // 1044 UDF words isolating ELF header from tripled passes
constexpr u64 kPad1Bytes = 17872;  // 4468 UDF words isolating tripled passes from voter+guard tail
constexpr u64 kPad2Bytes = 16640;  // 4160 UDF words isolating voter+guard tail from acquisition _start
constexpr u32 kUdfWord = 0x000007f0u;  // permanent architectural A64 UDF #2032

constexpr u64 kBase = 0x400000ULL;
constexpr u64 kBss = 0x600000ULL;
constexpr std::array<u64, 3> kDesc = {kBss, kBss + 0x1000ULL, kBss + 0x2000ULL};
constexpr u64 kBssSize = 0x3000ULL;
constexpr std::array<int, 3> kRes = {19, 20, 21};  // TMRVoterArm64: x19, x20, x21
constexpr u64 kCap0 = 0x10000ULL;
constexpr u64 kHdrLen = 0xb0ULL;

// Canonical 44-byte TMRVoterArm64.voterWords, with word 7 (park `b +0`)
// replaced by `b +0x10` (0x14000004) jumping over exit(3) to `guard`.
constexpr std::array<u32, 11> kVoterWords = {
    0xeb14027fu,  // 0x00: cmp  x19, x20
    0xfa551266u,  // 0x04: ccmp x19, x21, #6, ne
    0x9a940273u,  // 0x08: csel x19, x19, x20, eq
    0xfa551286u,  // 0x0c: ccmp x20, x21, #6, ne
    0x54000081u,  // 0x10: b.ne +0x10 (-> 0x20)
    0xaa1303f4u,  // 0x14: mov  x20, x19
    0xaa1303f5u,  // 0x18: mov  x21, x19
    0x14000004u,  // 0x1c: b    +0x10 (-> 0x2c guard; original voterWords[7] = 0x14000000)
    0xd2800060u,  // 0x20: movz x0, #3
    0xd2800ba8u,  // 0x24: movz x8, #93
    0xd4000001u,  // 0x28: svc  #0
};

// Registers chosen inside `gen_pass` (every scratch/live register is in
// {0,1,6,7,8,9,10,11,12,13,14,15,24,25}, all Hamming distance >= 2 from
// {x18, x19, x20, x21} and != sp/xzr=31).
constexpr int R_BASE = 0;   // x0:  input copy base
constexpr int R_NW = 1;     // x1:  nw
constexpr int R_INS = 6;    // x6:  current instruction pointer / ann_scan
constexpr int R_CURA = 7;   // x7:  curA / ann_end in wdone
constexpr int R_IDX = 8;    // x8:  instruction index i
constexpr int R_ANN = 9;    // x9:  ann_base of current word
constexpr int R_P = 10;     // x10: cursor / arg / lookup ptr
constexpr int R_NC = 11;    // x11: scratch / nc / ann_end from lookup
constexpr int R_HDR = 12;   // x12: main (P1-P2) / word_hdr (P3)
constexpr int R_CURD = 13;  // x13: curD
constexpr int R_T1 = 14;    // x14: scratch 1
constexpr int R_T2 = 15;    // x15: scratch 2 / wc / opcode
constexpr int R_END = 24;   // x24: p_end = base + N
constexpr int R_WS = 25;    // x25: ws (first word)

enum class ShiftKind : int { Lsl = 0, Lsr = 1, Asr = 2, Ror = 3 };
enum class LogicOp : int { And = 0, Orr = 1, Eor = 2, Ands = 3 };
enum class BitfieldKind : int { Sbfm = 0, Bfm = 1, Ubfm = 2 };
enum class MemSize : int { B1 = 0, H2 = 1, W4 = 2, X8 = 3 };
enum class ExtendKind : int { Uxtw = 2, Lsl = 3, Sxtw = 6, Sxtx = 7 };
enum class CSelOp : int { Csel = 0, Csinc = 1, Csinv = 2, Csneg = 3 };

enum Cc : int {
  CC_EQ = 0,
  CC_NE = 1,
  CC_HS = 2,
  CC_LO = 3,
  CC_MI = 4,
  CC_PL = 5,
  CC_HI = 8,
  CC_LS = 9,
};

u32 mem_bytes(MemSize sz) {
  switch (sz) {
    case MemSize::B1: return 1;
    case MemSize::H2: return 2;
    case MemSize::W4: return 4;
    case MemSize::X8: return 8;
  }
  fail_msg("bad MemSize");
}

const char* shift_name(ShiftKind s) {
  switch (s) {
    case ShiftKind::Lsl: return "lsl";
    case ShiftKind::Lsr: return "lsr";
    case ShiftKind::Asr: return "asr";
    case ShiftKind::Ror: return "ror";
  }
  fail_msg("bad ShiftKind");
}

const char* logic_name(LogicOp o) {
  switch (o) {
    case LogicOp::And: return "and";
    case LogicOp::Orr: return "orr";
    case LogicOp::Eor: return "eor";
    case LogicOp::Ands: return "ands";
  }
  fail_msg("bad LogicOp");
}

const char* memsz_name(MemSize sz) {
  switch (sz) {
    case MemSize::B1: return "b1";
    case MemSize::H2: return "h2";
    case MemSize::W4: return "w4";
    case MemSize::X8: return "x8";
  }
  fail_msg("bad MemSize");
}

const char* cc_name(int cc) {
  switch (cc) {
    case 0: return "eq";
    case 1: return "ne";
    case 2: return "hs";
    case 3: return "lo";
    case 4: return "mi";
    case 5: return "pl";
    case 8: return "hi";
    case 9: return "ls";
    default: return "cc";
  }
}

std::string reg_str(int w, int r, bool sp_allowed = false) {
  if (r == 31) return sp_allowed ? (w == 64 ? "sp" : "wsp") : (w == 64 ? "xzr" : "wzr");
  return (w == 64 ? "x" : "w") + std::to_string(r);
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

std::string bin_byte(u8 b) {
  std::string s(8, '0');
  for (int i = 0; i < 8; ++i) {
    if ((b >> (7 - i)) & 1U) s[static_cast<std::size_t>(i)] = '1';
  }
  return s;
}

std::string bin_bytes(const std::vector<u8>& bs) {
  std::string out;
  for (std::size_t i = 0; i < bs.size(); ++i) {
    if (i > 0) out.push_back(' ');
    out += bin_byte(bs[i]);
  }
  return out;
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

// ------------------------------------------------------- AArch64Phys.Ins model

struct Ins {
  enum class Kind {
    MovWide,
    AddSubImm,
    AddSubReg,
    LogicReg,
    Bitfield,
    Adr,
    CcmpReg,
    Csel,
    Ldr,
    Str,
    Ldur,
    Stur,
    LdrReg,
    StrReg,
    B,
    Bcond,
    Cbz,
    Tbz,
    Br,
    Ret,
    Svc,
    // Generator structural markers (never decoded instructions in table)
    Pad0,
    Pad1,
    Pad2,
    Voter,
  };

  Kind kind = Kind::Svc;
  bool b0 = false;  // isSub / isPage / inv / nz / link / scaled
  bool b1 = false;  // setFlags
  int w = 64;       // 32 or 64
  int r0 = 0;       // rd / rt / kind / cc / bit
  int r1 = 0;       // rn
  int r2 = 0;       // rm / nzcv
  int aux = 0;      // hw / amt / cc / immr
  int aux2 = 0;     // imms
  LogicOp lop = LogicOp::Orr;
  ShiftKind sh = ShiftKind::Lsl;
  BitfieldKind bfk = BitfieldKind::Ubfm;
  MemSize sz = MemSize::X8;
  ExtendKind ext = ExtendKind::Uxtw;
  CSelOp csop = CSelOp::Csel;
  u64 imm = 0;

  bool operator==(const Ins& o) const = default;
};

// ------------------------------------------- embedded AArch64Phys.decodeWord

u32 bf(u32 w, int lo, int len) {
  u32 mask = (len == 32) ? 0xffffffffu : ((1u << len) - 1u);
  return (w >> lo) & mask;
}

u64 sx(int bits, u64 v) {
  u64 sign = 1ULL << (bits - 1);
  if (v < sign) return v;
  return v + (0ULL - (1ULL << bits));
}

std::optional<MemSize> mem_size_of(u32 sz_bits) {
  switch (sz_bits) {
    case 0: return MemSize::B1;
    case 1: return MemSize::H2;
    case 2: return MemSize::W4;
    case 3: return MemSize::X8;
    default: return std::nullopt;
  }
}

std::optional<ExtendKind> extend_of(u32 opt_bits) {
  switch (opt_bits) {
    case 2: return ExtendKind::Uxtw;
    case 3: return ExtendKind::Lsl;
    case 6: return ExtendKind::Sxtw;
    case 7: return ExtendKind::Sxtx;
    default: return std::nullopt;
  }
}

std::optional<Ins> decode_word(u32 w) {
  u32 sf = bf(w, 31, 1);
  int wd = (sf == 1) ? 64 : 32;
  if (bf(w, 26, 5) == 0x05) {
    u64 imm26 = bf(w, 0, 26);
    return Ins{.kind = Ins::Kind::B, .b0 = (sf == 1), .imm = sx(28, imm26 * 4ULL)};
  }
  if (bf(w, 24, 8) == 0x54 && bf(w, 4, 1) == 0) {
    u64 imm19 = bf(w, 5, 19);
    int cc = static_cast<int>(bf(w, 0, 4));
    return Ins{.kind = Ins::Kind::Bcond, .r0 = cc, .imm = sx(21, imm19 * 4ULL)};
  }
  if (bf(w, 25, 6) == 0x1a) {
    bool nz = (bf(w, 24, 1) == 1);
    u64 imm19 = bf(w, 5, 19);
    int rt = static_cast<int>(bf(w, 0, 5));
    return Ins{.kind = Ins::Kind::Cbz, .b0 = nz, .w = wd, .r0 = rt, .imm = sx(21, imm19 * 4ULL)};
  }
  if (bf(w, 25, 6) == 0x1b) {
    bool nz = (bf(w, 24, 1) == 1);
    int bit = static_cast<int>(sf * 32 + bf(w, 19, 5));
    u64 imm14 = bf(w, 5, 14);
    int rt = static_cast<int>(bf(w, 0, 5));
    return Ins{.kind = Ins::Kind::Tbz, .b0 = nz, .r0 = bit, .r1 = rt, .imm = sx(16, imm14 * 4ULL)};
  }
  if (w == 0xd4000001u) {
    return Ins{.kind = Ins::Kind::Svc, .imm = 0};
  }
  if (bf(w, 10, 22) == 0x3587c0 && bf(w, 0, 5) == 0) {
    return Ins{.kind = Ins::Kind::Br, .b0 = false, .r0 = static_cast<int>(bf(w, 5, 5))};
  }
  if (bf(w, 10, 22) == 0x358fc0 && bf(w, 0, 5) == 0) {
    return Ins{.kind = Ins::Kind::Br, .b0 = true, .r0 = static_cast<int>(bf(w, 5, 5))};
  }
  if (bf(w, 10, 22) == 0x3597c0 && bf(w, 0, 5) == 0) {
    return Ins{.kind = Ins::Kind::Ret, .r0 = static_cast<int>(bf(w, 5, 5))};
  }
  if (bf(w, 24, 5) == 0x10) {
    bool is_page = (sf == 1);
    u64 immlo = bf(w, 29, 2);
    u64 immhi = bf(w, 5, 19);
    int rd = static_cast<int>(bf(w, 0, 5));
    u64 imm21 = immhi * 4ULL + immlo;
    u64 imm = is_page ? sx(33, imm21 * 4096ULL) : sx(21, imm21);
    return Ins{.kind = Ins::Kind::Adr, .b0 = is_page, .r0 = rd, .imm = imm};
  }
  if (bf(w, 23, 6) == 0x22) {
    bool is_sub = (bf(w, 30, 1) == 1);
    bool set_flags = (bf(w, 29, 1) == 1);
    u32 sh = bf(w, 22, 1);
    u64 imm12 = bf(w, 10, 12);
    int rn = static_cast<int>(bf(w, 5, 5));
    int rd = static_cast<int>(bf(w, 0, 5));
    u64 imm = (sh == 1) ? imm12 * 4096ULL : imm12;
    return Ins{.kind = Ins::Kind::AddSubImm,
               .b0 = is_sub,
               .b1 = set_flags,
               .w = wd,
               .r0 = rd,
               .r1 = rn,
               .imm = imm};
  }
  if (bf(w, 23, 6) == 0x25) {
    int opc = static_cast<int>(bf(w, 29, 2));
    int hw = static_cast<int>(bf(w, 21, 2));
    u64 imm16 = bf(w, 5, 16);
    int rd = static_cast<int>(bf(w, 0, 5));
    if (opc == 1 || (sf == 0 && hw >= 2)) return std::nullopt;
    return Ins{.kind = Ins::Kind::MovWide, .w = wd, .r0 = rd, .r1 = opc, .aux = hw, .imm = imm16};
  }
  if (bf(w, 23, 6) == 0x26) {
    int opc = static_cast<int>(bf(w, 29, 2));
    u32 n_bit = bf(w, 22, 1);
    int immr = static_cast<int>(bf(w, 16, 6));
    int imms = static_cast<int>(bf(w, 10, 6));
    int rn = static_cast<int>(bf(w, 5, 5));
    int rd = static_cast<int>(bf(w, 0, 5));
    if (opc == 3 || n_bit != sf || (sf == 0 && (immr >= 32 || imms >= 32))) return std::nullopt;
    return Ins{.kind = Ins::Kind::Bitfield,
               .w = wd,
               .r0 = rd,
               .r1 = rn,
               .aux = immr,
               .aux2 = imms,
               .bfk = static_cast<BitfieldKind>(opc)};
  }
  if (bf(w, 24, 5) == 0x0b && bf(w, 21, 1) == 0) {
    bool is_sub = (bf(w, 30, 1) == 1);
    bool set_flags = (bf(w, 29, 1) == 1);
    u32 shift_bits = bf(w, 22, 2);
    int rm = static_cast<int>(bf(w, 16, 5));
    int imm6 = static_cast<int>(bf(w, 10, 6));
    int rn = static_cast<int>(bf(w, 5, 5));
    int rd = static_cast<int>(bf(w, 0, 5));
    if (shift_bits == 3 || (sf == 0 && imm6 >= 32)) return std::nullopt;
    return Ins{.kind = Ins::Kind::AddSubReg,
               .b0 = is_sub,
               .b1 = set_flags,
               .w = wd,
               .r0 = rd,
               .r1 = rn,
               .r2 = rm,
               .aux = imm6,
               .sh = static_cast<ShiftKind>(shift_bits)};
  }
  if (bf(w, 24, 5) == 0x0a) {
    int opc = static_cast<int>(bf(w, 29, 2));
    u32 shift_bits = bf(w, 22, 2);
    bool inv = (bf(w, 21, 1) == 1);
    int rm = static_cast<int>(bf(w, 16, 5));
    int imm6 = static_cast<int>(bf(w, 10, 6));
    int rn = static_cast<int>(bf(w, 5, 5));
    int rd = static_cast<int>(bf(w, 0, 5));
    if (sf == 0 && imm6 >= 32) return std::nullopt;
    return Ins{.kind = Ins::Kind::LogicReg,
               .b0 = inv,
               .w = wd,
               .r0 = rd,
               .r1 = rn,
               .r2 = rm,
               .aux = imm6,
               .lop = static_cast<LogicOp>(opc),
               .sh = static_cast<ShiftKind>(shift_bits)};
  }
  if (bf(w, 21, 8) == 0xd2 && bf(w, 29, 1) == 1 && bf(w, 10, 2) == 0 && bf(w, 4, 1) == 0) {
    bool is_sub = (bf(w, 30, 1) == 1);
    int rm = static_cast<int>(bf(w, 16, 5));
    int cc = static_cast<int>(bf(w, 12, 4));
    int rn = static_cast<int>(bf(w, 5, 5));
    int nzcv = static_cast<int>(bf(w, 0, 4));
    return Ins{.kind = Ins::Kind::CcmpReg, .b0 = is_sub, .w = wd, .r0 = rn, .r1 = rm, .r2 = nzcv, .aux = cc};
  }
  if (bf(w, 21, 8) == 0xd4 && bf(w, 29, 1) == 0 && bf(w, 11, 1) == 0) {
    bool op_bit = (bf(w, 30, 1) == 1);
    bool inc_bit = (bf(w, 10, 1) == 1);
    int rm = static_cast<int>(bf(w, 16, 5));
    int cc = static_cast<int>(bf(w, 12, 4));
    int rn = static_cast<int>(bf(w, 5, 5));
    int rd = static_cast<int>(bf(w, 0, 5));
    CSelOp op = (!op_bit && !inc_bit) ? CSelOp::Csel
                : (!op_bit && inc_bit) ? CSelOp::Csinc
                : (op_bit && !inc_bit) ? CSelOp::Csinv
                                       : CSelOp::Csneg;
    return Ins{.kind = Ins::Kind::Csel, .w = wd, .r0 = rd, .r1 = rn, .r2 = rm, .aux = cc, .csop = op};
  }
  if (bf(w, 24, 6) == 0x39) {
    auto sz = mem_size_of(bf(w, 30, 2));
    u32 opc = bf(w, 22, 2);
    u64 imm12 = bf(w, 10, 12);
    int rn = static_cast<int>(bf(w, 5, 5));
    int rt = static_cast<int>(bf(w, 0, 5));
    if (!sz) return std::nullopt;
    u64 off = imm12 * mem_bytes(*sz);
    if (opc == 0) return Ins{.kind = Ins::Kind::Str, .r0 = rt, .r1 = rn, .sz = *sz, .imm = off};
    if (opc == 1) return Ins{.kind = Ins::Kind::Ldr, .r0 = rt, .r1 = rn, .sz = *sz, .imm = off};
    return std::nullopt;
  }
  if (bf(w, 24, 6) == 0x38 && bf(w, 21, 1) == 0 && bf(w, 10, 2) == 0) {
    auto sz = mem_size_of(bf(w, 30, 2));
    u32 opc = bf(w, 22, 2);
    u64 imm9 = bf(w, 12, 9);
    int rn = static_cast<int>(bf(w, 5, 5));
    int rt = static_cast<int>(bf(w, 0, 5));
    if (!sz) return std::nullopt;
    u64 off = sx(9, imm9);
    if (opc == 0) return Ins{.kind = Ins::Kind::Stur, .r0 = rt, .r1 = rn, .sz = *sz, .imm = off};
    if (opc == 1) return Ins{.kind = Ins::Kind::Ldur, .r0 = rt, .r1 = rn, .sz = *sz, .imm = off};
    return std::nullopt;
  }
  if (bf(w, 24, 6) == 0x38 && bf(w, 21, 1) == 1 && bf(w, 10, 2) == 2) {
    auto sz = mem_size_of(bf(w, 30, 2));
    auto ext = extend_of(bf(w, 13, 3));
    u32 opc = bf(w, 22, 2);
    int rm = static_cast<int>(bf(w, 16, 5));
    bool sbit = (bf(w, 12, 1) == 1);
    int rn = static_cast<int>(bf(w, 5, 5));
    int rt = static_cast<int>(bf(w, 0, 5));
    if (!sz || !ext) return std::nullopt;
    if (opc == 0) return Ins{.kind = Ins::Kind::StrReg, .b0 = sbit, .r0 = rt, .r1 = rn, .r2 = rm, .sz = *sz, .ext = *ext};
    if (opc == 1) return Ins{.kind = Ins::Kind::LdrReg, .b0 = sbit, .r0 = rt, .r1 = rn, .r2 = rm, .sz = *sz, .ext = *ext};
    return std::nullopt;
  }
  return std::nullopt;
}

// ------------------------------------------------------------------ encoders

struct Encoded {
  u32 word = 0;
  std::vector<u8> bytes;
  Ins ins;
};

Encoded mk_word(u32 w, Ins ins) {
  std::vector<u8> bs;
  push_le(bs, w, 4);
  return {w, std::move(bs), ins};
}

void check_no_x18(int r) { require(r != 18, "x18 is forbidden (Darwin reserved)"); }

Encoded e_movWide(int kind, int w, int rd, u32 imm16, int hw = 0) {
  check_no_x18(rd);
  require(rd >= 0 && rd < 31, "movWide rd must be GPR 0..30");
  require(w == 32 || w == 64, "bad width");
  require(kind == 0 || kind == 2 || kind == 3, "bad movWide kind");
  require(imm16 < 65536u, "imm16 out of range");
  require(hw >= 0 && hw < (w == 64 ? 4 : 2), "hw out of range");
  u32 sf = (w == 64) ? 1u : 0u;
  u32 word = (sf << 31) | (static_cast<u32>(kind) << 29) | (0x25u << 23) |
             (static_cast<u32>(hw) << 21) | (imm16 << 5) | static_cast<u32>(rd);
  return mk_word(word, Ins{.kind = Ins::Kind::MovWide, .w = w, .r0 = rd, .r1 = kind, .aux = hw, .imm = imm16});
}

Encoded e_movz(int w, int rd, u32 imm16, int hw = 0) { return e_movWide(2, w, rd, imm16, hw); }
Encoded e_movn(int w, int rd, u32 imm16, int hw = 0) { return e_movWide(0, w, rd, imm16, hw); }
Encoded e_movk(int w, int rd, u32 imm16, int hw = 0) { return e_movWide(3, w, rd, imm16, hw); }

Encoded e_addSubImm(bool is_sub, bool set_flags, int w, int rd, int rn, u32 imm12) {
  check_no_x18(rd);
  check_no_x18(rn);
  require(set_flags || (rd != 31 && rn != 31), "sp not used in addSubImm");
  require(imm12 < 4096u, "imm12 out of range");
  u32 sf = (w == 64) ? 1u : 0u;
  u32 word = (sf << 31) | ((is_sub ? 1u : 0u) << 30) | ((set_flags ? 1u : 0u) << 29) |
             (0x11u << 24) | (imm12 << 10) | (static_cast<u32>(rn) << 5) | static_cast<u32>(rd);
  return mk_word(word, Ins{.kind = Ins::Kind::AddSubImm,
                           .b0 = is_sub,
                           .b1 = set_flags,
                           .w = w,
                           .r0 = rd,
                           .r1 = rn,
                           .imm = imm12});
}

Encoded e_addI(int rd, int rn, u32 imm12) { return e_addSubImm(false, false, 64, rd, rn, imm12); }
Encoded e_subI(int rd, int rn, u32 imm12) { return e_addSubImm(true, false, 64, rd, rn, imm12); }
Encoded e_cmpI(int rn, u32 imm12) { return e_addSubImm(true, true, 64, 31, rn, imm12); }

Encoded e_addSubReg(bool is_sub, bool set_flags, int w, int rd, int rn, int rm, ShiftKind sh = ShiftKind::Lsl,
                    int amt = 0) {
  check_no_x18(rd);
  check_no_x18(rn);
  check_no_x18(rm);
  require(sh != ShiftKind::Ror, "addSubReg does not support ror");
  require(amt >= 0 && amt < w, "shift amount out of range");
  u32 sf = (w == 64) ? 1u : 0u;
  u32 word = (sf << 31) | ((is_sub ? 1u : 0u) << 30) | ((set_flags ? 1u : 0u) << 29) |
             (0x0bu << 24) | (static_cast<u32>(sh) << 22) | (static_cast<u32>(rm) << 16) |
             (static_cast<u32>(amt) << 10) | (static_cast<u32>(rn) << 5) | static_cast<u32>(rd);
  return mk_word(word, Ins{.kind = Ins::Kind::AddSubReg,
                           .b0 = is_sub,
                           .b1 = set_flags,
                           .w = w,
                           .r0 = rd,
                           .r1 = rn,
                           .r2 = rm,
                           .aux = amt,
                           .sh = sh});
}

Encoded e_addR(int rd, int rn, int rm, ShiftKind sh = ShiftKind::Lsl, int amt = 0) {
  return e_addSubReg(false, false, 64, rd, rn, rm, sh, amt);
}
Encoded e_subR(int rd, int rn, int rm, ShiftKind sh = ShiftKind::Lsl, int amt = 0) {
  return e_addSubReg(true, false, 64, rd, rn, rm, sh, amt);
}
Encoded e_cmpR(int rn, int rm) { return e_addSubReg(true, true, 64, 31, rn, rm); }

Encoded e_mov(int rd, int rm, int w = 64) {
  check_no_x18(rd);
  check_no_x18(rm);
  require(rd < 31 && rm < 31, "mov GPR only");
  u32 sf = (w == 64) ? 1u : 0u;
  u32 word = (sf << 31) | (1u << 29) | (0x0au << 24) | (static_cast<u32>(rm) << 16) | (31u << 5) |
             static_cast<u32>(rd);
  return mk_word(word, Ins{.kind = Ins::Kind::LogicReg,
                           .b0 = false,
                           .w = w,
                           .r0 = rd,
                           .r1 = 31,
                           .r2 = rm,
                           .aux = 0,
                           .lop = LogicOp::Orr,
                           .sh = ShiftKind::Lsl});
}

Encoded e_orrR(int rd, int rn, int rm, int w = 64) {
  check_no_x18(rd);
  check_no_x18(rn);
  check_no_x18(rm);
  require(rd < 31 && rn < 31 && rm < 31, "orrR GPR only");
  u32 sf = (w == 64) ? 1u : 0u;
  u32 word = (sf << 31) | (1u << 29) | (0x0au << 24) | (static_cast<u32>(rm) << 16) |
             (static_cast<u32>(rn) << 5) | static_cast<u32>(rd);
  return mk_word(word, Ins{.kind = Ins::Kind::LogicReg,
                           .b0 = false,
                           .w = w,
                           .r0 = rd,
                           .r1 = rn,
                           .r2 = rm,
                           .aux = 0,
                           .lop = LogicOp::Orr,
                           .sh = ShiftKind::Lsl});
}

Encoded e_ldr(MemSize sz, int rt, int rn, u32 off = 0) {
  check_no_x18(rt);
  check_no_x18(rn);
  require(rt < 31 && rn < 31, "ldr base/dest GPR only");
  u32 b = mem_bytes(sz);
  require(off % b == 0 && off / b < 4096u, "ldr off out of range");
  u32 imm12 = off / b;
  u32 word = (static_cast<u32>(sz) << 30) | (0x39u << 24) | (1u << 22) | (imm12 << 10) |
             (static_cast<u32>(rn) << 5) | static_cast<u32>(rt);
  return mk_word(word, Ins{.kind = Ins::Kind::Ldr, .r0 = rt, .r1 = rn, .sz = sz, .imm = off});
}

Encoded e_str(MemSize sz, int rt, int rn, u32 off = 0) {
  check_no_x18(rt);
  check_no_x18(rn);
  require(rt < 31 && rn < 31, "str base/src GPR only");
  u32 b = mem_bytes(sz);
  require(off % b == 0 && off / b < 4096u, "str off out of range");
  u32 imm12 = off / b;
  u32 word = (static_cast<u32>(sz) << 30) | (0x39u << 24) | (imm12 << 10) |
             (static_cast<u32>(rn) << 5) | static_cast<u32>(rt);
  return mk_word(word, Ins{.kind = Ins::Kind::Str, .r0 = rt, .r1 = rn, .sz = sz, .imm = off});
}

Encoded e_ldur(MemSize sz, int rt, int rn, i32 off) {
  check_no_x18(rt);
  check_no_x18(rn);
  require(rt < 31 && rn < 31, "ldur base/dest GPR only");
  require(off >= -256 && off < 256, "ldur off out of range");
  u32 imm9 = static_cast<u32>(off) & 0x1ffu;
  u32 word = (static_cast<u32>(sz) << 30) | (0x38u << 24) | (1u << 22) | (imm9 << 12) |
             (static_cast<u32>(rn) << 5) | static_cast<u32>(rt);
  return mk_word(word,
                 Ins{.kind = Ins::Kind::Ldur, .r0 = rt, .r1 = rn, .sz = sz, .imm = static_cast<u64>(static_cast<i64>(off))});
}

Encoded e_svc() {
  return mk_word(0xd4000001u, Ins{.kind = Ins::Kind::Svc, .imm = 0});
}

Encoded e_adrp_at(int rd, u64 pc, u64 target_page) {
  check_no_x18(rd);
  require(rd < 31, "adrp GPR only");
  require((target_page & 0xfffULL) == 0, "adrp target must be 4KiB page aligned");
  i64 diff = static_cast<i64>(target_page) - static_cast<i64>(pc & ~0xfffULL);
  require((diff & 0xfffLL) == 0, "unaligned adrp diff");
  i64 pages = diff >> 12;
  require(pages >= -(1LL << 20) && pages < (1LL << 20), "adrp range overflow");
  u32 imm21 = static_cast<u32>(pages) & 0x1fffffu;
  u32 immlo = imm21 & 3u;
  u32 immhi = (imm21 >> 2) & 0x7ffffu;
  u32 word = (1u << 31) | (immlo << 29) | (0x10u << 24) | (immhi << 5) | static_cast<u32>(rd);
  return mk_word(word, Ins{.kind = Ins::Kind::Adr,
                           .b0 = true,
                           .r0 = rd,
                           .imm = sx(33, static_cast<u64>(imm21) * 4096ULL)});
}

Encoded e_b_at(u64 pc, u64 target) {
  i64 diff = static_cast<i64>(target) - static_cast<i64>(pc);
  require((diff & 3LL) == 0 && diff >= -(1LL << 27) && diff < (1LL << 27), "b out of range");
  u32 imm26 = static_cast<u32>(diff >> 2) & 0x03ffffffu;
  u32 word = (0x05u << 26) | imm26;
  return mk_word(word, Ins{.kind = Ins::Kind::B, .b0 = false, .imm = sx(28, static_cast<u64>(imm26) * 4ULL)});
}

Encoded e_bcond_at(int cc, u64 pc, u64 target) {
  require(cc >= 0 && cc < 16, "bad cc");
  i64 diff = static_cast<i64>(target) - static_cast<i64>(pc);
  require((diff & 3LL) == 0 && diff >= -(1LL << 20) && diff < (1LL << 20), "b.cond out of range");
  u32 imm19 = static_cast<u32>(diff >> 2) & 0x7ffffu;
  u32 word = (0x54u << 24) | (imm19 << 5) | static_cast<u32>(cc);
  return mk_word(word, Ins{.kind = Ins::Kind::Bcond, .r0 = cc, .imm = sx(21, static_cast<u64>(imm19) * 4ULL)});
}

Encoded e_cbz_at(bool nz, int w, int rt, u64 pc, u64 target) {
  check_no_x18(rt);
  require(rt < 31 && (w == 32 || w == 64), "bad cbz rt/w");
  i64 diff = static_cast<i64>(target) - static_cast<i64>(pc);
  require((diff & 3LL) == 0 && diff >= -(1LL << 20) && diff < (1LL << 20), "cbz out of range");
  u32 sf = (w == 64) ? 1u : 0u;
  u32 imm19 = static_cast<u32>(diff >> 2) & 0x7ffffu;
  u32 word = (sf << 31) | (0x1au << 25) | ((nz ? 1u : 0u) << 24) | (imm19 << 5) | static_cast<u32>(rt);
  return mk_word(word,
                 Ins{.kind = Ins::Kind::Cbz, .b0 = nz, .w = w, .r0 = rt, .imm = sx(21, static_cast<u64>(imm19) * 4ULL)});
}

// ------------------------------------------------------------------ assembler

struct AsmItem {
  enum class Kind { Label, Word, B, Bcond, Cbz, Adrp, Raw };

  Kind kind = Kind::Label;
  std::string label;
  Encoded enc;
  bool nz = false;
  int w = 64;
  int r0 = 0;
  std::string target_lab;
  u64 target_page = 0;
  std::string comment;
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
    it.kind = AsmItem::Kind::Word;
    it.enc = std::move(enc);
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void mov64(int rd, u64 imm, std::string c = "") {
    i(e_movz(64, rd, static_cast<u32>(imm & 0xffffu), 0), std::move(c));
    i(e_movk(64, rd, static_cast<u32>((imm >> 16) & 0xffffu), 1));
    i(e_movk(64, rd, static_cast<u32>((imm >> 32) & 0xffffu), 2));
    i(e_movk(64, rd, static_cast<u32>((imm >> 48) & 0xffffu), 3));
  }

  void adrp(int rd, u64 target_page, std::string c = "") {
    AsmItem it;
    it.kind = AsmItem::Kind::Adrp;
    it.r0 = rd;
    it.target_page = target_page;
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void b(std::string lab, std::string c = "") {
    AsmItem it;
    it.kind = AsmItem::Kind::B;
    it.target_lab = std::move(lab);
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void bcond(int cc, std::string lab, std::string c = "") {
    AsmItem it;
    it.kind = AsmItem::Kind::Bcond;
    it.r0 = cc;
    it.target_lab = std::move(lab);
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void cbz(int w, int rt, std::string lab, std::string c = "") {
    AsmItem it;
    it.kind = AsmItem::Kind::Cbz;
    it.nz = false;
    it.w = w;
    it.r0 = rt;
    it.target_lab = std::move(lab);
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void cbnz(int w, int rt, std::string lab, std::string c = "") {
    AsmItem it;
    it.kind = AsmItem::Kind::Cbz;
    it.nz = true;
    it.w = w;
    it.r0 = rt;
    it.target_lab = std::move(lab);
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void raw(std::vector<u8> bs, Ins ins, std::string c = "") {
    AsmItem it;
    it.kind = AsmItem::Kind::Raw;
    it.enc = Encoded{0u, std::move(bs), ins};
    it.comment = std::move(c);
    items.push_back(std::move(it));
  }

  void mbegin(std::string name, std::map<std::string, int> reg_params,
              std::map<std::string, std::string> addr_params) {
    open.push_back(MacroRec{.name = std::move(name),
                            .reg_params = std::move(reg_params),
                            .addr_params = std::move(addr_params),
                            .start_idx = items.size(),
                            .end_idx = 0});
  }

  void mend() {
    require(!open.empty(), "mend without mbegin");
    MacroRec rec = std::move(open.back());
    open.pop_back();
    rec.end_idx = items.size();
    macros.push_back(std::move(rec));
  }

  struct Result {
    std::vector<OutEntry> out;
    std::map<std::string, u64> labels;
    std::vector<std::pair<std::string, u64>> label_order;
  };

  Result assemble(u64 base_addr) {
    // Fixed-width 4-byte AArch64 instructions: single exact pass computes all addresses!
    std::map<std::string, u64> labels;
    std::vector<std::pair<std::string, u64>> label_order;
    std::vector<u64> cur_idx_addr(items.size() + 1, 0);
    u64 addr = base_addr;
    for (std::size_t idx = 0; idx < items.size(); ++idx) {
      cur_idx_addr[idx] = addr;
      const auto& it = items[idx];
      switch (it.kind) {
        case AsmItem::Kind::Label:
          require(!labels.contains(it.label), "duplicate label: " + it.label);
          labels[it.label] = addr;
          label_order.emplace_back(it.label, addr);
          break;
        case AsmItem::Kind::Word:
        case AsmItem::Kind::B:
        case AsmItem::Kind::Bcond:
        case AsmItem::Kind::Cbz:
        case AsmItem::Kind::Adrp:
          require(addr % 4 == 0, "unaligned code address");
          addr += 4;
          break;
        case AsmItem::Kind::Raw:
          require(it.enc.bytes.size() % 4 == 0, "raw size must be 4-byte multiple");
          addr += it.enc.bytes.size();
          break;
      }
    }
    cur_idx_addr[items.size()] = addr;

    std::vector<OutEntry> out;
    for (std::size_t idx = 0; idx < items.size(); ++idx) {
      u64 a = cur_idx_addr[idx];
      const auto& it = items[idx];
      if (it.kind == AsmItem::Kind::Label) continue;
      if (it.kind == AsmItem::Kind::Word || it.kind == AsmItem::Kind::Raw) {
        out.push_back(OutEntry{a, it.enc.bytes, it.enc.ins, it.comment, idx});
        continue;
      }
      if (it.kind == AsmItem::Kind::Adrp) {
        Encoded enc = e_adrp_at(it.r0, a, it.target_page);
        std::string cmt = it.comment.empty() ? hex_0x(it.target_page)
                                             : (hex_0x(it.target_page) + "   " + it.comment);
        out.push_back(OutEntry{a, std::move(enc.bytes), enc.ins, cmt, idx});
        continue;
      }
      auto lit = labels.find(it.target_lab);
      require(lit != labels.end(), "missing label: " + it.target_lab);
      u64 t = lit->second;
      Encoded enc;
      if (it.kind == AsmItem::Kind::B) {
        enc = e_b_at(a, t);
      } else if (it.kind == AsmItem::Kind::Bcond) {
        enc = e_bcond_at(it.r0, a, t);
      } else {
        enc = e_cbz_at(it.nz, it.w, it.r0, a, t);
      }
      std::string cmt = (it.comment.empty() ? "" : (it.comment + " ")) + "-> " + it.target_lab;
      out.push_back(OutEntry{a, std::move(enc.bytes), enc.ins, std::move(cmt), idx});
    }
    idx_addr = std::move(cur_idx_addr);
    return Result{std::move(out), std::move(labels), std::move(label_order)};
  }
};

// ---------------------------------------------------------------- the program

void skipcode(Asm& A, int p, const std::string& rej, const std::string& c = "") {
  A.mbegin("skipcode", {{"p", p}}, {{"rej", rej}});
  constexpr int cnt = R_NC, t1 = R_T1;
  std::string top = A.fresh("sc_top"), done = A.fresh("sc_done");
  std::string l1 = A.fresh("sc_l1"), l5 = A.fresh("sc_l5");
  std::string l9 = A.fresh("sc_l9"), ln = A.fresh("sc_len");
  A.label(top);
  A.cbz(64, cnt, done, "skipCode" + c + ": instructions left?");
  A.i(e_cmpR(p, R_END), "opcode byte inside input?");
  A.bcond(CC_HS, rej);
  A.i(e_ldr(MemSize::B1, t1, p, 0), "opcode");
  A.i(e_cmpI(t1, 57), "prim 0x00..0x38");
  A.bcond(CC_LO, l1);
  A.i(e_cmpI(t1, 0x80));
  A.bcond(CC_LO, rej, "0x39..0x7f: no such opcode");
  A.bcond(CC_EQ, l9, "0x80 lit u64");
  A.i(e_cmpI(t1, 0x8c));
  A.bcond(CC_HI, rej, "0x8d..0xff: no such opcode");
  A.bcond(CC_EQ, l1, "0x8c ret");
  A.i(e_cmpI(t1, 0x83));
  A.bcond(CC_LO, l5, "0x81 str, 0x82 pick");
  A.i(e_cmpI(t1, 0x86));
  A.bcond(CC_LO, l1, "0x83 tor, 0x84 fromr, 0x85 rfetch");
  A.label(l5);
  A.i(e_movz(64, t1, 5), "opcode + u32");
  A.b(ln);
  A.label(l9);
  A.i(e_movz(64, t1, 9), "opcode + u64");
  A.b(ln);
  A.label(l1);
  A.i(e_movz(64, t1, 1), "opcode alone");
  A.label(ln);
  A.i(e_addR(p, p, t1), "past instruction");
  A.i(e_cmpR(p, R_END));
  A.bcond(CC_HI, rej, "truncated instruction");
  A.i(e_subI(cnt, cnt, 1));
  A.b(top);
  A.label(done);
  A.mend();
}

void skipwords(Asm& A, const std::string& rej, const std::string& c = "") {
  A.mbegin("skipwords", {}, {{"rej", rej}});
  constexpr int p = R_P, wc = R_T2, nc = R_NC, t1 = R_T1;
  std::string top = A.fresh("sw_top"), done = A.fresh("sw_done");
  A.label(top);
  A.cbz(64, wc, done, "skipWords" + c + ": words left?");
  A.i(e_addI(t1, p, 20));
  A.i(e_cmpR(t1, R_END));
  A.bcond(CC_HI, rej, "truncated word header");
  A.i(e_ldr(MemSize::W4, t1, p, 16), "na");
  A.i(e_addR(wc, wc, t1, ShiftKind::Lsl, 32), "park na in high half");
  A.i(e_ldr(MemSize::W4, nc, p, 12), "nc");
  A.i(e_addI(p, p, 20), "first instruction");
  skipcode(A, p, rej, c);
  A.i(e_addR(t1, 31, wc, ShiftKind::Lsr, 32), "na");
  A.i(e_mov(wc, wc, 32), "words left, high half cleared");
  A.i(e_addR(nc, t1, t1, ShiftKind::Lsl, 1), "3 * na");
  A.i(e_addR(p, p, nc, ShiftKind::Lsl, 2), "past annotations (p += 12 * na)");
  A.i(e_cmpR(p, R_END));
  A.bcond(CC_HI, rej, "truncated annotations");
  A.i(e_subI(wc, wc, 1));
  A.b(top);
  A.label(done);
  A.mend();
}

void skipdata(Asm& A, const std::string& rej) {
  A.mbegin("skipdata", {}, {{"rej", rej}});
  constexpr int p = R_P, cnt = R_T2, t1 = R_NC, t2 = R_T1;
  std::string top = A.fresh("sd_top"), done = A.fresh("sd_done");
  A.label(top);
  A.cbz(64, cnt, done, "skipData: items left?");
  A.i(e_addI(t1, p, 4));
  A.i(e_cmpR(t1, R_END));
  A.bcond(CC_HI, rej, "truncated length");
  A.i(e_ldr(MemSize::W4, t2, p, 0), "len");
  A.i(e_mov(p, t1));
  A.i(e_subR(t1, R_END, p), "bytes left");
  A.i(e_cmpR(t1, t2));
  A.bcond(CC_LO, rej, "truncated data");
  A.i(e_addR(p, p, t2));
  A.i(e_subI(cnt, cnt, 1));
  A.b(top);
  A.label(done);
  A.mend();
}

void chk(Asm& A, int r, const std::string& ovf) {
  A.mbegin("chk", {{"r", r}}, {{"ovf", ovf}});
  A.i(e_movz(64, R_P, 1, kHmaxShift / 16), "2^48");
  A.i(e_cmpR(r, R_P));
  A.bcond(CC_HS, ovf, "height >= 2^48");
  A.mend();
}

void lookup(Asm& A, int t, const std::string& c = "") {
  A.mbegin("lookup", {{"t", t}}, {});
  std::string top = A.fresh("lk_top"), done = A.fresh("lk_done");
  A.i(e_ldr(MemSize::W4, R_T1, R_HDR, 16), "lookup" + c + ": na");
  A.i(e_addR(R_NC, R_T1, R_T1, ShiftKind::Lsl, 1), "3 * na");
  A.i(e_addR(R_NC, R_ANN, R_NC, ShiftKind::Lsl, 2), "x11 = end of annotations");
  A.i(e_mov(R_P, R_ANN));
  A.label(top);
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_EQ, done, "none");
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 0));
  A.i(e_cmpR(R_T1, t));
  A.bcond(CC_EQ, done, "found");
  A.i(e_addI(R_P, R_P, 12));
  A.b(top);
  A.label(done);
  A.mend();
}

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
    A.i(e_movz(64, R_P, static_cast<u32>(i_cnt)),
        "ops " + std::to_string(rs[beg].lo) + ".." + std::to_string(rs[beg].hi) + ": takes " +
            std::to_string(i_cnt));
    A.i(e_movz(64, R_NC, static_cast<u32>(o_cnt)), "pushes " + std::to_string(o_cnt));
    A.b(have);
    return;
  }
  std::size_t mid = beg + (end - beg) / 2;
  std::string right = A.fresh("eff");
  A.i(e_cmpI(R_T2, static_cast<u32>(rs[mid].lo)));
  A.bcond(CC_HS, right);
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
  A.adrp(R_T2, D, "pass " + std::to_string(k + 1) + ": descriptor " + std::to_string(k + 1));
  A.i(e_ldr(MemSize::X8, R_BASE, R_T2, 0), "base of input copy");
  A.i(e_ldr(MemSize::X8, R_T1, R_T2, 8), "N");
  A.i(e_addR(R_END, R_BASE, R_T1), "p_end = base + N");
  A.i(e_cmpI(R_T1, 16));
  A.bcond(CC_LO, rej, "shorter than file header");
  A.i(e_ldr(MemSize::W4, R_T1, R_BASE, 0));
  A.i(e_movz(64, R_T2, 0x5053u, 0), "magic \"SPR1\"");
  A.i(e_movk(64, R_T2, 0x3152u, 1));
  A.i(e_cmpR(R_T1, R_T2));
  A.bcond(CC_NE, rej, "not \"SPR1\"");
  A.i(e_ldr(MemSize::W4, R_HDR, R_BASE, 4), "main");
  A.i(e_ldr(MemSize::W4, R_NW, R_BASE, 8), "nw");
  A.i(e_ldr(MemSize::W4, R_T2, R_BASE, 12), "nd");
  A.i(e_addI(R_P, R_BASE, 16), "first data item");
  skipdata(A, rej);
  A.i(e_mov(R_WS, R_P), "ws = first word");
  A.i(e_mov(R_T2, R_NW), "nw words");
  skipwords(A, rej, " (parse)");
  A.i(e_cmpR(R_P, R_END));
  A.bcond(CC_NE, rej, "trailing bytes");
  // ---- P2: main
  A.i(e_cmpR(R_HDR, R_NW));
  A.bcond(CC_HS, rej, "main >= nw");
  A.i(e_mov(R_P, R_WS));
  A.i(e_mov(R_T2, R_HDR));
  skipwords(A, rej, " (main)");
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 0), "ins of main");
  A.cbnz(64, R_T1, rej, "main takes arguments");
  // ---- P3: every word
  A.i(e_mov(R_HDR, R_WS), "word_hdr");
  std::string wl = P + "_word";
  A.label(wl);
  A.i(e_cmpR(R_HDR, R_END));
  A.bcond(CC_EQ, acc, "all words checked");
  A.i(e_ldr(MemSize::W4, R_NC, R_HDR, 12), "nc");
  A.cbz(64, R_NC, rej, "empty code");
  A.i(e_addI(R_ANN, R_HDR, 20));
  skipcode(A, R_ANN, rej, " (annotations)");
  A.i(e_movz(64, R_IDX, 0), "i = 0");
  lookup(A, R_IDX, " (0)");
  std::string init = A.fresh("init");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_EQ, init);
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 4));
  A.i(e_ldr(MemSize::W4, R_T2, R_HDR, 0));
  A.i(e_cmpR(R_T1, R_T2));
  A.bcond(CC_NE, rej, "H[0].d != ins");
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 8));
  A.cbnz(64, R_T1, rej, "H[0].a != 0");
  A.label(init);
  A.i(e_ldr(MemSize::W4, R_CURD, R_HDR, 0), "curD = ins");
  A.i(e_movz(64, R_CURA, 0), "curA = 0");
  A.i(e_addI(R_INS, R_HDR, 20), "p = first instruction");
  std::string walk = P + "_walk", wdone = P + "_wdone";
  std::string fall = P + "_fall", nofall = P + "_nofall", adv = P + "_adv";
  A.label(walk);
  A.i(e_ldr(MemSize::W4, R_T2, R_HDR, 12));
  A.i(e_cmpR(R_IDX, R_T2));
  A.bcond(CC_EQ, wdone, "i == nc");
  lookup(A, R_IDX, " (i)");
  std::string hr = A.fresh("hready");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_EQ, hr);
  A.i(e_ldr(MemSize::W4, R_CURD, R_P, 4), "h = annotation at i");
  A.i(e_ldr(MemSize::W4, R_CURA, R_P, 8));
  A.label(hr);
  A.i(e_ldr(MemSize::B1, R_T2, R_INS, 0), "opcode");
  auto nm = [&](const char* stem) { return P + "_" + stem; };
  A.i(e_cmpI(R_T2, 57));
  A.bcond(CC_LO, nm("prim"));
  constexpr std::array<std::pair<u32, const char*>, 13> kOpTargets = {{
      {0x80u, "lit"}, {0x81u, "str"}, {0x82u, "pick"}, {0x83u, "tor"}, {0x84u, "fromr"},
      {0x85u, "rfetch"}, {0x86u, "locget"}, {0x87u, "locset"}, {0x88u, "jmp"},
      {0x89u, "jz"}, {0x8au, "jz"}, {0x8bu, "call"}, {0x8cu, "ret"},
  }};
  for (const auto& [op, target] : kOpTargets) {
    A.i(e_cmpI(R_T2, op));
    A.bcond(CC_EQ, nm(target));
  }
  A.b(rej, "unreachable after parse");
  // prim
  A.label(nm("prim"));
  A.i(e_cmpI(R_T2, 50));
  A.bcond(CC_EQ, nm("exit"));
  effchain(A, nm("have"));
  A.label(nm("have"));
  A.i(e_cmpR(R_CURD, R_P));
  A.bcond(CC_LO, rej, "too few operands");
  A.i(e_subR(R_CURD, R_CURD, R_P));
  A.i(e_addR(R_CURD, R_CURD, R_NC));
  chk(A, R_CURD, ovf);
  A.i(e_movz(64, R_T2, 1));
  A.b(fall);
  A.label(nm("exit"));
  A.cbz(64, R_CURD, rej, "exit needs one operand");
  A.i(e_movz(64, R_T2, 1));
  A.b(nofall);
  // lit / str
  A.label(nm("lit"));
  A.i(e_addI(R_CURD, R_CURD, 1));
  chk(A, R_CURD, ovf);
  A.i(e_movz(64, R_T2, 9));
  A.b(fall);
  A.label(nm("str"));
  A.i(e_addI(R_CURD, R_CURD, 2));
  chk(A, R_CURD, ovf);
  A.i(e_movz(64, R_T2, 5));
  A.b(fall);
  // pick n: n < d
  A.label(nm("pick"));
  A.i(e_ldur(MemSize::W4, R_P, R_INS, 1));
  A.i(e_cmpR(R_P, R_CURD));
  A.bcond(CC_HS, rej, "pick beyond stack");
  A.i(e_addI(R_CURD, R_CURD, 1));
  chk(A, R_CURD, ovf);
  A.i(e_movz(64, R_T2, 5));
  A.b(fall);
  // tor
  A.label(nm("tor"));
  A.cbz(64, R_CURD, rej);
  A.i(e_subI(R_CURD, R_CURD, 1));
  A.i(e_addI(R_CURA, R_CURA, 1));
  chk(A, R_CURA, ovf);
  A.i(e_movz(64, R_T2, 1));
  A.b(fall);
  // fromr
  A.label(nm("fromr"));
  A.cbz(64, R_CURA, rej);
  A.i(e_subI(R_CURA, R_CURA, 1));
  A.i(e_addI(R_CURD, R_CURD, 1));
  chk(A, R_CURD, ovf);
  A.i(e_movz(64, R_T2, 1));
  A.b(fall);
  // rfetch
  A.label(nm("rfetch"));
  A.cbz(64, R_CURA, rej);
  A.i(e_addI(R_CURD, R_CURD, 1));
  chk(A, R_CURD, ovf);
  A.i(e_movz(64, R_T2, 1));
  A.b(fall);
  // locget i: i < locals
  A.label(nm("locget"));
  A.i(e_ldur(MemSize::W4, R_P, R_INS, 1));
  A.i(e_ldr(MemSize::W4, R_NC, R_HDR, 8), "locals");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_HS, rej);
  A.i(e_addI(R_CURD, R_CURD, 1));
  chk(A, R_CURD, ovf);
  A.i(e_movz(64, R_T2, 5));
  A.b(fall);
  // locset i: 1 <= d and i < locals
  A.label(nm("locset"));
  A.cbz(64, R_CURD, rej);
  A.i(e_ldur(MemSize::W4, R_P, R_INS, 1));
  A.i(e_ldr(MemSize::W4, R_NC, R_HDR, 8), "locals");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_HS, rej);
  A.i(e_subI(R_CURD, R_CURD, 1));
  A.i(e_movz(64, R_T2, 5));
  A.b(fall);
  // jmp t: ann(t) = h
  A.label(nm("jmp"));
  A.i(e_ldur(MemSize::W4, R_T2, R_INS, 1), "t");
  lookup(A, R_T2, " (jmp t)");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_EQ, rej, "no annotation at t");
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 4));
  A.i(e_cmpR(R_T1, R_CURD));
  A.bcond(CC_NE, rej);
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 8));
  A.i(e_cmpR(R_T1, R_CURA));
  A.bcond(CC_NE, rej);
  A.i(e_movz(64, R_T2, 5));
  A.b(nofall);
  // jz/jnz t: 1 <= d and ann(t) = (d - 1, a)
  A.label(nm("jz"));
  A.cbz(64, R_CURD, rej);
  A.i(e_subI(R_CURD, R_CURD, 1));
  A.i(e_ldur(MemSize::W4, R_T2, R_INS, 1), "t");
  lookup(A, R_T2, " (jz t)");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_EQ, rej, "no annotation at t");
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 4));
  A.i(e_cmpR(R_T1, R_CURD));
  A.bcond(CC_NE, rej);
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 8));
  A.i(e_cmpR(R_T1, R_CURA));
  A.bcond(CC_NE, rej);
  A.i(e_movz(64, R_T2, 5));
  A.b(fall);
  // call w: w < nw, c.ins <= d, d - c.ins + c.outs
  A.label(nm("call"));
  A.i(e_ldur(MemSize::W4, R_T2, R_INS, 1), "w");
  A.i(e_cmpR(R_T2, R_NW));
  A.bcond(CC_HS, rej, "no such word");
  A.i(e_mov(R_P, R_WS), "rescan from ws");
  skipwords(A, rej, " (call)");
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 0), "c.ins");
  A.i(e_cmpR(R_CURD, R_T1));
  A.bcond(CC_LO, rej, "too few arguments");
  A.i(e_subR(R_CURD, R_CURD, R_T1));
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 4), "c.outs");
  A.i(e_addR(R_CURD, R_CURD, R_T1));
  chk(A, R_CURD, ovf);
  A.i(e_movz(64, R_T2, 5));
  A.b(fall);
  // ret: h = (outs, 0)
  A.label(nm("ret"));
  A.i(e_ldr(MemSize::W4, R_T1, R_HDR, 4), "outs");
  A.i(e_cmpR(R_CURD, R_T1));
  A.bcond(CC_NE, rej);
  A.cbnz(64, R_CURA, rej);
  A.i(e_movz(64, R_T2, 1));
  A.b(nofall);
  // falls through to h' = (curD, curA): i + 1 < nc and ann(i + 1) in {none, h'}
  A.label(fall);
  A.i(e_addI(R_IDX, R_IDX, 1));
  A.i(e_ldr(MemSize::W4, R_T1, R_HDR, 12));
  A.i(e_cmpR(R_IDX, R_T1));
  A.bcond(CC_HS, rej, "falls off end");
  lookup(A, R_IDX, " (i + 1)");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_EQ, adv);
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 4));
  A.i(e_cmpR(R_T1, R_CURD));
  A.bcond(CC_NE, rej);
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 8));
  A.i(e_cmpR(R_T1, R_CURA));
  A.bcond(CC_NE, rej);
  A.label(adv);
  A.i(e_addR(R_INS, R_INS, R_T2), "next instruction");
  A.b(walk);
  A.label(nofall);
  A.i(e_addI(R_IDX, R_IDX, 1));
  lookup(A, R_IDX, " (i + 1)");
  std::string nz = A.fresh("nfzero");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_EQ, nz);
  A.i(e_ldr(MemSize::W4, R_CURD, R_P, 4));
  A.i(e_ldr(MemSize::W4, R_CURA, R_P, 8));
  A.b(adv);
  A.label(nz);
  A.i(e_movz(64, R_CURD, 0));
  A.i(e_movz(64, R_CURA, 0));
  A.b(adv);
  // annotations: every (t, d, a) has t < nc and lookup t = (d, a)
  A.label(wdone);
  A.i(e_ldr(MemSize::W4, R_T1, R_HDR, 16), "na");
  A.i(e_addR(R_CURA, R_T1, R_T1, ShiftKind::Lsl, 1), "3 * na");
  A.i(e_addR(R_CURA, R_ANN, R_CURA, ShiftKind::Lsl, 2), "x7 = end of annotations = next word");
  A.i(e_mov(R_INS, R_ANN));
  std::string al = P + "_anns", ad = P + "_annsok";
  A.label(al);
  A.i(e_cmpR(R_INS, R_CURA));
  A.bcond(CC_EQ, ad);
  A.i(e_ldr(MemSize::W4, R_T2, R_INS, 0), "t");
  A.i(e_ldr(MemSize::W4, R_T1, R_HDR, 12));
  A.i(e_cmpR(R_T2, R_T1));
  A.bcond(CC_HS, rej, "annotation past code");
  lookup(A, R_T2, " (ann t)");
  A.i(e_cmpR(R_P, R_NC));
  A.bcond(CC_EQ, rej);
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 4));
  A.i(e_ldr(MemSize::W4, R_T2, R_INS, 4));
  A.i(e_cmpR(R_T1, R_T2));
  A.bcond(CC_NE, rej, "conflicting annotations");
  A.i(e_ldr(MemSize::W4, R_T1, R_P, 8));
  A.i(e_ldr(MemSize::W4, R_T2, R_INS, 8));
  A.i(e_cmpR(R_T1, R_T2));
  A.bcond(CC_NE, rej, "conflicting annotations");
  A.i(e_addI(R_INS, R_INS, 12));
  A.b(al);
  A.label(ad);
  A.i(e_mov(R_HDR, R_CURA), "next word");
  A.b(wl);
  // verdicts
  A.label(acc);
  A.mov64(R, kAccept, "pass " + std::to_string(k + 1) + ": ACCEPT");
  A.b(nxt);
  A.label(ovf);
  A.mov64(R, kOverflow, "pass " + std::to_string(k + 1) + ": OVERFLOW");
  A.b(nxt);
  A.label(rej);
  A.mov64(R, kReject, "pass " + std::to_string(k + 1) + ": REJECT");
  A.b(nxt);
  A.label(P + "_end");
}

std::vector<u8> make_pad_bytes(u64 bytes) {
  require(bytes % 4 == 0, "pad bytes must be 4-byte multiple");
  std::vector<u8> out;
  out.reserve(bytes);
  for (u64 i = 0; i < bytes / 4; ++i) push_le(out, kUdfWord, 4);
  return out;
}

Asm gen_program() {
  Asm A;
  A.raw(make_pad_bytes(kPad0Bytes), Ins{.kind = Ins::Kind::Pad0, .imm = kPad0Bytes},
        "barrier between ELF header and tripled region");
  A.label("tripled");
  for (int k = 0; k < 3; ++k) gen_pass(A, k);

  A.raw(make_pad_bytes(kPad1Bytes), Ins{.kind = Ins::Kind::Pad1, .imm = kPad1Bytes},
        "barrier between tripled passes and voter/guard tail");

  // ---- the voter: TMRVoterArm64.voterWords, park (0x14000000) -> b +0x10 (0x14000004)
  A.label("vote");
  std::vector<u8> voter;
  for (u32 w : kVoterWords) push_le(voter, w, 4);
  A.raw(std::move(voter), Ins{.kind = Ins::Kind::Voter},
        "TMRVoterArm64.voterCode, park b +0 -> b +0x10 (to guard)");

  // ---- the dual guard
  A.label("guard");
  A.i(e_movz(64, 0, 5), "status 5 unless both guards pass");
  A.i(e_movz(64, 8, 0), "clear x8 until accept is authorized");
  A.i(e_movz(64, 2, 0), "pad & scrub x2");
  A.i(e_movz(64, 7, 0), "pad & scrub x7");
  A.mov64(1, kAccept, "guard 1: load ACCEPT");
  A.i(e_cmpR(19, 1));
  A.bcond(CC_NE, "not_acc", "guard 1");
  A.i(e_movz(64, 2, 0), "align guard 1 -> not_acc Hamming offset");
  A.mov64(6, kAccept, "guard 2: load ACCEPT independently");
  A.i(e_cmpR(20, 6));
  A.bcond(CC_NE, "not_acc", "guard 2");
  A.i(e_movz(64, 8, 93), "exit(0): x8 first,");
  A.i(e_subI(0, 20, static_cast<u32>(kAccept & 0xffULL)),
      "then status = (x20 - 0x96) mod 256, zero only when (x20 & 0xff) == 0x96");
  A.label("accept_svc");
  A.i(e_svc());
  A.b("rej_exit");

  A.label("not_acc");
  A.mov64(1, kOverflow, "check OVERFLOW");
  A.i(e_cmpR(19, 1));
  A.bcond(CC_NE, "rej_exit", "not OVERFLOW");

  A.label("nomem");
  A.i(e_movz(64, 0, 6), "exit(6) via dual-reg OR (popcount(6)=2; immune to 1-bit hw/adrp/reg)");
  A.i(e_movz(64, 2, 6));
  A.i(e_orrR(0, 0, 2));
  A.i(e_movz(64, 8, 93));
  A.i(e_svc());

  A.label("rej_exit");
  A.i(e_movz(64, 0, 5), "exit(5) via dual-reg OR (popcount(5)=2; immune to 1-bit hw/adrp/reg)");
  A.i(e_movz(64, 2, 5));
  A.i(e_orrR(0, 0, 2));
  A.i(e_movz(64, 8, 93));
  A.i(e_svc());
  A.label("core_end");

  A.raw(make_pad_bytes(kPad2Bytes), Ins{.kind = Ins::Kind::Pad2, .imm = kPad2Bytes},
        "barrier between voter/guard tail and acquisition");
  A.label("start");

  // ---- acquisition: mmap 64 KiB, read stdin, grow with mremap
  A.i(e_movz(64, 0, 0), "mmap(0, 64 KiB, PROT_RW=3, MAP_PRIVATE|ANON=0x22, -1, 0)");
  A.i(e_movz(64, 1, static_cast<u32>(kCap0 >> 16), 1));  // 0x10000
  A.i(e_movz(64, 2, 3));
  A.i(e_movz(64, 3, 0x22));
  A.i(e_movn(64, 4, 0));     // -1
  A.i(e_movz(64, 5, 0));
  A.i(e_movz(64, 8, 222));   // __NR_mmap
  A.i(e_svc());
  A.i(e_cmpI(0, 0));
  A.bcond(CC_MI, "nomem", "refused");
  A.i(e_mov(14, 0), "x14 = buffer");
  A.i(e_movz(64, 12, static_cast<u32>(kCap0 >> 16), 1), "x12 = capacity");
  A.i(e_movz(64, 13, 0), "x13 = length");
  A.label("rd");
  A.i(e_cmpR(13, 12));
  A.bcond(CC_NE, "rd_go", "room left");
  A.i(e_mov(0, 14), "mremap(buf, cap, 2*cap, MREMAP_MAYMOVE=1)");
  A.i(e_mov(1, 12));
  A.i(e_addR(2, 31, 12, ShiftKind::Lsl, 1));
  A.i(e_movz(64, 3, 1));
  A.i(e_movz(64, 8, 216));   // __NR_mremap
  A.i(e_svc());
  A.i(e_cmpI(0, 0));
  A.bcond(CC_MI, "nomem", "refused");
  A.i(e_mov(14, 0));
  A.i(e_addR(12, 31, 12, ShiftKind::Lsl, 1));
  A.label("rd_go");
  A.i(e_movz(64, 0, 0), "read(0, buf + len, cap - len)");
  A.i(e_addR(1, 14, 13));
  A.i(e_subR(2, 12, 13));
  A.i(e_movz(64, 8, 63));    // __NR_read
  A.i(e_svc());
  A.i(e_cmpI(0, 0));
  A.bcond(CC_MI, "nomem", "read failed");
  A.bcond(CC_EQ, "eof");
  A.i(e_addR(13, 13, 0));
  A.b("rd");

  // ---- triplication: two more copies, then three descriptors
  A.label("eof");
  for (const auto& [copy_idx, dst] : std::array<std::pair<int, int>, 2>{{{1, 15}, {2, 24}}}) {
    A.i(e_movz(64, 0, 0), "mmap copy " + std::to_string(copy_idx + 1) + ": len + 1 bytes");
    A.i(e_addI(1, 13, 1));
    A.i(e_movz(64, 2, 3));
    A.i(e_movz(64, 3, 0x22));
    A.i(e_movn(64, 4, 0));
    A.i(e_movz(64, 5, 0));
    A.i(e_movz(64, 8, 222));
    A.i(e_svc());
    A.i(e_cmpI(0, 0));
    A.bcond(CC_MI, "nomem");
    A.i(e_mov(dst, 0));
    std::string top = A.fresh("cp_top"), done = A.fresh("cp_done");
    A.i(e_mov(6, 14), "copy len bytes");
    A.i(e_mov(7, dst));
    A.i(e_mov(11, 13));
    A.label(top);
    A.cbz(64, 11, done);
    A.i(e_ldr(MemSize::B1, 10, 6, 0));
    A.i(e_str(MemSize::B1, 10, 7, 0));
    A.i(e_addI(6, 6, 1));
    A.i(e_addI(7, 7, 1));
    A.i(e_subI(11, 11, 1));
    A.b(top);
    A.label(done);
  }
  constexpr std::array<int, 3> kCopySrcs = {14, 15, 24};
  for (std::size_t k = 0; k < 3; ++k) {
    A.adrp(10, kDesc[k], "descriptor " + std::to_string(k + 1));
    A.i(e_str(MemSize::X8, kCopySrcs[k], 10, 0));
    A.i(e_str(MemSize::X8, 13, 10, 8));
  }
  A.mov64(19, kReject, "results start at REJECT");
  A.mov64(20, kReject);
  A.mov64(21, kReject);
  A.b("tripled");
  return A;
}

std::vector<u8> elf_header(u64 entry, u64 filesz) {
  std::vector<u8> h = {0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  push_le(h, 2, 2);        // e_type = ET_EXEC
  push_le(h, 183, 2);      // e_machine = EM_AARCH64 (183 = 0xb7)
  push_le(h, 1, 4);        // e_version = 1
  push_le(h, entry, 8);    // e_entry
  push_le(h, 0x40, 8);     // e_phoff
  push_le(h, 0, 8);        // e_shoff
  push_le(h, 0, 4);        // e_flags
  push_le(h, 64, 2);       // e_ehsize
  push_le(h, 56, 2);       // e_phentsize
  push_le(h, 2, 2);        // e_phnum
  push_le(h, 0, 2);        // e_shentsize
  push_le(h, 0, 2);        // e_shnum
  push_le(h, 0, 2);        // e_shstrndx
  // PT_LOAD 0: code R+X
  push_le(h, 1, 4);
  push_le(h, 5, 4);
  push_le(h, 0, 8);
  push_le(h, kBase, 8);
  push_le(h, kBase, 8);
  push_le(h, filesz, 8);
  push_le(h, filesz, 8);
  push_le(h, 0x1000, 8);
  // PT_LOAD 1: bss R+W (descriptors)
  push_le(h, 1, 4);
  push_le(h, 6, 4);
  push_le(h, 0, 8);
  push_le(h, kBss, 8);
  push_le(h, kBss, 8);
  push_le(h, 0, 8);
  push_le(h, kBssSize, 8);
  push_le(h, 0x1000, 8);
  require(h.size() == kHdrLen, "unexpected ELF header size");
  return h;
}

struct HdrComment {
  std::size_t off;
  std::size_t len;
  const char* comment;
};

constexpr auto kHdrComments = std::to_array<HdrComment>({
    {0x00, 4, "e_ident: magic 0x7f 'E' 'L' 'F'"},
    {0x04, 1, "EI_CLASS   = ELFCLASS64"},
    {0x05, 1, "EI_DATA    = little endian"},
    {0x06, 1, "EI_VERSION = 1"},
    {0x07, 1, "EI_OSABI   = System V"},
    {0x08, 8, "EI_ABIVERSION and padding"},
    {0x10, 2, "e_type      = ET_EXEC"},
    {0x12, 2, "e_machine   = EM_AARCH64 (183 = 0xb7)"},
    {0x14, 4, "e_version   = 1"},
    {0x18, 8, "e_entry"},
    {0x20, 8, "e_phoff     = 0x40"},
    {0x28, 8, "e_shoff     = 0"},
    {0x30, 4, "e_flags     = 0"},
    {0x34, 2, "e_ehsize    = 64"},
    {0x36, 2, "e_phentsize = 56"},
    {0x38, 2, "e_phnum     = 2"},
    {0x3a, 2, "e_shentsize = 0"},
    {0x3c, 2, "e_shnum     = 0"},
    {0x3e, 2, "e_shstrndx  = 0"},
    {0x40, 4, "p_type   = PT_LOAD (code)"},
    {0x44, 4, "p_flags  = PF_R | PF_X"},
    {0x48, 8, "p_offset = 0"},
    {0x50, 8, "p_vaddr  = 0x400000"},
    {0x58, 8, "p_paddr  = 0x400000"},
    {0x60, 8, "p_filesz = whole file"},
    {0x68, 8, "p_memsz  = whole file"},
    {0x70, 8, "p_align  = 0x1000"},
    {0x78, 4, "p_type   = PT_LOAD (descriptors)"},
    {0x7c, 4, "p_flags  = PF_R | PF_W"},
    {0x80, 8, "p_offset = 0"},
    {0x88, 8, "p_vaddr  = 0x600000"},
    {0x90, 8, "p_paddr  = 0x600000"},
    {0x98, 8, "p_filesz = 0"},
    {0xa0, 8, "p_memsz  = 0x3000 (three pages)"},
    {0xa8, 8, "p_align  = 0x1000"},
});

std::string ins_str(const Ins& ins, u64 pc) {
  switch (ins.kind) {
    case Ins::Kind::MovWide: {
      const char* mn = (ins.r1 == 0) ? "movn " : (ins.r1 == 2) ? "movz " : "movk ";
      std::string s = std::string(mn) + reg_str(ins.w, ins.r0) + ", " + hex_0x(ins.imm);
      if (ins.aux > 0) s += ", lsl #" + std::to_string(ins.aux * 16);
      return s;
    }
    case Ins::Kind::AddSubImm: {
      if (ins.b0 && ins.b1 && ins.r0 == 31) {
        return "cmp " + reg_str(ins.w, ins.r1, true) + ", #" + std::to_string(ins.imm);
      }
      if (!ins.b0 && ins.b1 && ins.r0 == 31) {
        return "cmn " + reg_str(ins.w, ins.r1, true) + ", #" + std::to_string(ins.imm);
      }
      const char* mn = ins.b0 ? (ins.b1 ? "subs " : "sub ") : (ins.b1 ? "adds " : "add ");
      return std::string(mn) + reg_str(ins.w, ins.r0, !ins.b1) + ", " +
             reg_str(ins.w, ins.r1, true) + ", #" + std::to_string(ins.imm);
    }
    case Ins::Kind::AddSubReg: {
      if (ins.b0 && ins.b1 && ins.r0 == 31) {
        return "cmp " + reg_str(ins.w, ins.r1) + ", " + reg_str(ins.w, ins.r2);
      }
      const char* mn = ins.b0 ? (ins.b1 ? "subs " : "sub ") : (ins.b1 ? "adds " : "add ");
      std::string s = std::string(mn) + reg_str(ins.w, ins.r0) + ", " + reg_str(ins.w, ins.r1) +
                      ", " + reg_str(ins.w, ins.r2);
      if (ins.aux != 0 || ins.sh != ShiftKind::Lsl) {
        s += std::string(", ") + shift_name(ins.sh) + " #" + std::to_string(ins.aux);
      }
      return s;
    }
    case Ins::Kind::LogicReg: {
      if (ins.lop == LogicOp::Orr && !ins.b0 && ins.r1 == 31 && ins.aux == 0 &&
          ins.sh == ShiftKind::Lsl) {
        return "mov " + reg_str(ins.w, ins.r0) + ", " + reg_str(ins.w, ins.r2);
      }
      return std::string(logic_name(ins.lop)) + " " + reg_str(ins.w, ins.r0) + ", " +
             reg_str(ins.w, ins.r1) + ", " + reg_str(ins.w, ins.r2);
    }
    case Ins::Kind::Adr: {
      u64 base = ins.b0 ? (pc & ~0xfffULL) : pc;
      u64 tgt = base + ins.imm;
      return std::string(ins.b0 ? "adrp " : "adr ") + reg_str(64, ins.r0) + ", " + hex_0x(tgt);
    }
    case Ins::Kind::CcmpReg:
      return std::string("ccmp ") + reg_str(ins.w, ins.r0) + ", " + reg_str(ins.w, ins.r1) +
             ", #" + std::to_string(ins.r2) + ", " + cc_name(ins.aux);
    case Ins::Kind::Csel:
      return std::string("csel ") + reg_str(ins.w, ins.r0) + ", " + reg_str(ins.w, ins.r1) +
             ", " + reg_str(ins.w, ins.r2) + ", " + cc_name(ins.aux);
    case Ins::Kind::Ldr:
    case Ins::Kind::Str:
    case Ins::Kind::Ldur:
    case Ins::Kind::Stur: {
      bool is_ld = (ins.kind == Ins::Kind::Ldr || ins.kind == Ins::Kind::Ldur);
      const char* base_mn =
          (ins.kind == Ins::Kind::Ldur)   ? "ldur"
          : (ins.kind == Ins::Kind::Stur) ? "stur"
          : is_ld                         ? "ldr"
                                          : "str";
      std::string sfx = (ins.sz == MemSize::B1) ? "b " : (ins.sz == MemSize::H2) ? "h " : " ";
      int rw = (ins.sz == MemSize::X8) ? 64 : 32;
      i64 disp = static_cast<i64>(ins.imm);
      return std::string(base_mn) + sfx + reg_str(rw, ins.r0) + ", [" + reg_str(64, ins.r1, true) +
             ", #" + std::to_string(disp) + "]";
    }
    case Ins::Kind::B:
      return std::string(ins.b0 ? "bl " : "b ") + hex_0x(pc + ins.imm);
    case Ins::Kind::Bcond:
      return std::string("b.") + cc_name(ins.r0) + " " + hex_0x(pc + ins.imm);
    case Ins::Kind::Cbz:
      return std::string(ins.b0 ? "cbnz " : "cbz ") + reg_str(ins.w, ins.r0) + ", " +
             hex_0x(pc + ins.imm);
    case Ins::Kind::Svc:
      return "svc #0";
    case Ins::Kind::Pad0:
      return "pad0";
    case Ins::Kind::Pad1:
      return "pad1";
    case Ins::Kind::Pad2:
      return "pad2";
    case Ins::Kind::Voter:
      return "voter";
    default:
      fail_msg("unsupported Ins::Kind in ins_str");
  }
}

std::string lean_shift(ShiftKind sh) {
  switch (sh) {
    case ShiftKind::Lsl: return ".lsl";
    case ShiftKind::Lsr: return ".lsr";
    case ShiftKind::Asr: return ".asr";
    case ShiftKind::Ror: return ".ror";
  }
  fail_msg("bad shift");
}

std::string lean_logic(LogicOp op) {
  switch (op) {
    case LogicOp::And: return ".and";
    case LogicOp::Orr: return ".orr";
    case LogicOp::Eor: return ".eor";
    case LogicOp::Ands: return ".ands";
  }
  fail_msg("bad logic");
}

std::string lean_csel(CSelOp op) {
  switch (op) {
    case CSelOp::Csel: return ".csel";
    case CSelOp::Csinc: return ".csinc";
    case CSelOp::Csinv: return ".csinv";
    case CSelOp::Csneg: return ".csneg";
  }
  fail_msg("bad csel");
}

std::string lean_bbool(bool b) { return b ? "true" : "false"; }

std::string render_ins(const Ins& ins, u64 L, u64 size, u64 cur_pc,
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
  auto fmt_branch_off = [&](u64 rel_off) -> std::string {
    if (aparams.empty() && L == 0 && size == 0) {
      return std::to_string(rel_off);
    }
    u64 tgt = cur_pc + rel_off;
    auto ap = aparams.find(tgt);
    if (ap != aparams.end()) {
      // Branch immediate is (target - cur_pc) mod 2^64.
      u64 off_in_tmpl = cur_pc - L;
      std::string pc_expr = (off_in_tmpl == 0) ? std::string(Lname)
                                               : ("(" + std::string(Lname) + " + " + std::to_string(off_in_tmpl) + ")");
      return "relOff " + pc_expr + " " + ap->second;
    }
    if (tgt >= L && tgt <= L + size) {
      return std::to_string(rel_off);
    }
    fail_msg("branch target outside template instance");
  };

  switch (ins.kind) {
    case Ins::Kind::MovWide:
      return ".movWide " + std::to_string(ins.r1) + " " + std::to_string(ins.w) + " " +
             fmt_reg(0, ins.r0) + " " + fmt_imm(ins.imm) + " " + std::to_string(ins.aux);
    case Ins::Kind::AddSubImm:
      return ".addSubImm " + lean_bbool(ins.b0) + " " + lean_bbool(ins.b1) + " " +
             std::to_string(ins.w) + " " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " +
             fmt_imm(ins.imm);
    case Ins::Kind::AddSubReg:
      return ".addSubReg " + lean_bbool(ins.b0) + " " + lean_bbool(ins.b1) + " " +
             std::to_string(ins.w) + " " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " +
             fmt_reg(2, ins.r2) + " " + lean_shift(ins.sh) + " " + std::to_string(ins.aux);
    case Ins::Kind::LogicReg:
      return ".logicReg " + lean_logic(ins.lop) + " " + lean_bbool(ins.b0) + " " +
             std::to_string(ins.w) + " " + fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " +
             fmt_reg(2, ins.r2) + " " + lean_shift(ins.sh) + " " + std::to_string(ins.aux);
    case Ins::Kind::Adr:
      return ".adr " + lean_bbool(ins.b0) + " " + fmt_reg(0, ins.r0) + " " + std::to_string(ins.imm);
    case Ins::Kind::CcmpReg:
      return ".ccmpReg " + lean_bbool(ins.b0) + " " + std::to_string(ins.w) + " " +
             fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " + std::to_string(ins.r2) + " " +
             std::to_string(ins.aux);
    case Ins::Kind::Csel:
      return ".csel " + lean_csel(ins.csop) + " " + std::to_string(ins.w) + " " +
             fmt_reg(0, ins.r0) + " " + fmt_reg(1, ins.r1) + " " + fmt_reg(2, ins.r2) + " " +
             std::to_string(ins.aux);
    case Ins::Kind::Ldr:
      return ".ldr ." + std::string(memsz_name(ins.sz)) + " " + fmt_reg(0, ins.r0) + " " +
             fmt_reg(1, ins.r1) + " " + std::to_string(ins.imm);
    case Ins::Kind::Str:
      return ".str ." + std::string(memsz_name(ins.sz)) + " " + fmt_reg(0, ins.r0) + " " +
             fmt_reg(1, ins.r1) + " " + std::to_string(ins.imm);
    case Ins::Kind::Ldur:
      return ".ldur ." + std::string(memsz_name(ins.sz)) + " " + fmt_reg(0, ins.r0) + " " +
             fmt_reg(1, ins.r1) + " " + std::to_string(ins.imm);
    case Ins::Kind::B:
      return ".b " + lean_bbool(ins.b0) + " (" + fmt_branch_off(ins.imm) + ")";
    case Ins::Kind::Bcond:
      return ".bcond " + std::to_string(ins.r0) + " (" + fmt_branch_off(ins.imm) + ")";
    case Ins::Kind::Cbz:
      return ".cbz " + lean_bbool(ins.b0) + " " + std::to_string(ins.w) + " " +
             fmt_reg(0, ins.r0) + " (" + fmt_branch_off(ins.imm) + ")";
    case Ins::Kind::Svc:
      return ".svc " + std::to_string(ins.imm);
    default:
      fail_msg("unsupported Ins::Kind in render_ins");
  }
}

std::string lean_ins(const Ins& ins) { return render_ins(ins, 0, 0, 0, {}, {}, {}); }

std::string join_strings(const std::vector<std::string>& v, std::string_view sep) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i > 0) out += sep;
    out += v[i];
  }
  return out;
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

struct InstInfo {
  const MacroRec* rec;
  u64 L;
  u64 size;
  std::vector<std::string> reg_args;
  std::vector<u64> addr_args;
};

struct ExprChunk {
  u64 addr = 0;
  std::string expr;
};

std::string format_template_fn(const TemplateSpec& sp, const Asm& A,
                               const std::vector<OutEntry>& out,
                               const std::map<std::string, u64>& labels,
                               const std::vector<InstInfo>& insts, const InstInfo& rep) {
  std::map<std::size_t, OutEntry> by_idx;
  for (const auto& e : out) by_idx[e.idx] = e;

  std::map<u64, std::string> aparams;
  for (std::size_t k = 0; k < sp.addr_params.size(); ++k) {
    aparams[rep.addr_args[k]] = sp.addr_params[k];
  }
  std::map<std::pair<int, int>, std::string> rparams;
  std::map<u64, std::string> iparams;
  for (std::size_t j = 0; j < sp.reg_params.size(); ++j) {
    int v = rep.rec->reg_params.at(sp.reg_params[j]);
    rparams[{0, v}] = sp.reg_params[j];
    rparams[{1, v}] = sp.reg_params[j];
    rparams[{2, v}] = sp.reg_params[j];
  }

  std::vector<std::pair<u64, const MacroRec*>> children;
  for (const auto& m : A.macros) {
    if (m.start_idx >= rep.rec->start_idx && m.end_idx <= rep.rec->end_idx &&
        !(m.start_idx == rep.rec->start_idx && m.end_idx == rep.rec->end_idx)) {
      bool nested = false;
      for (const auto& p : A.macros) {
        if (p.start_idx >= rep.rec->start_idx && p.end_idx <= rep.rec->end_idx &&
            !(p.start_idx == rep.rec->start_idx && p.end_idx == rep.rec->end_idx) &&
            p.start_idx <= m.start_idx && m.end_idx <= p.end_idx &&
            !(p.start_idx == m.start_idx && p.end_idx == m.end_idx)) {
          nested = true;
          break;
        }
      }
      if (!nested) children.emplace_back(A.idx_addr[m.start_idx], &m);
    }
  }
  std::sort(children.begin(), children.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::map<u64, const MacroRec*> child_at(children.begin(), children.end());

  std::vector<std::string> sig_params = {"L : Nat"};
  for (const char* rp : sp.reg_params) sig_params.push_back(std::string(rp) + " : Nat");
  for (const char* ap : sp.addr_params) sig_params.push_back(std::string(ap) + " : Nat");
  std::string hdr = "def " + std::string(sp.name) + "T (" + join_strings(sig_params, ") (") +
                    ") : List (Nat × Ins × Nat) :=";

  if (std::string_view(sp.name) == "eff") {
    std::vector<std::pair<std::string, std::string>> items;
    std::size_t idx = rep.rec->start_idx;
    while (idx < rep.rec->end_idx) {
      u64 a = A.idx_addr[idx];
      auto it = by_idx.find(idx);
      if (it == by_idx.end()) {
        ++idx;
        continue;
      }
      u64 off = a - rep.L;
      std::string addr_expr = (off == 0) ? "L" : ("L + " + std::to_string(off));
      std::string ins_expr = render_ins(it->second.ins, rep.L, rep.size, a, aparams, rparams, iparams);
      items.emplace_back(addr_expr, ins_expr);
      ++idx;
    }
    std::string body = "  [\n";
    for (std::size_t k = 0; k < items.size(); ++k) {
      body += "    (" + items[k].first + ", " + items[k].second + ", 4)";
      body += (k + 1 < items.size()) ? ",\n" : "\n  ]\n";
    }
    return hdr + "\n" + body;
  }

  std::vector<std::pair<std::string, std::string>> direct;
  std::vector<ExprChunk> sub_chunks;
  std::size_t idx = rep.rec->start_idx;
  while (idx < rep.rec->end_idx) {
    u64 a = A.idx_addr[idx];
    auto cit = child_at.find(a);
    if (cit != child_at.end()) {
      const MacroRec* cm = cit->second;
      const InstInfo* ci = nullptr;
      for (const auto& cand : insts) {
        if (cand.rec == cm) {
          ci = &cand;
          break;
        }
      }
      require(ci != nullptr, "missing child macro instance");
      const TemplateSpec& csp = spec_for(cm->name);
      std::vector<std::string> args = {"(L + " + std::to_string(a - rep.L) + ")"};
      for (std::size_t k = 0; k < csp.reg_params.size(); ++k) {
        int val = cm->reg_params.at(csp.reg_params[k]);
        args.push_back(rparams.contains({0, val}) ? rparams.at({0, val}) : std::to_string(val));
      }
      for (std::size_t k = 0; k < csp.addr_params.size(); ++k) {
        u64 av = labels.at(cm->addr_params.at(csp.addr_params[k]));
        args.push_back(aparams.contains(av) ? aparams.at(av) : std::to_string(av));
      }
      sub_chunks.push_back(ExprChunk{a, std::string(csp.name) + "T " + join_strings(args, " ")});
      idx = cm->end_idx;
      continue;
    }
    auto it = by_idx.find(idx);
    if (it != by_idx.end()) {
      u64 off = a - rep.L;
      std::string addr_expr = (off == 0) ? "L" : ("L + " + std::to_string(off));
      std::string ins_expr = render_ins(it->second.ins, rep.L, rep.size, a, aparams, rparams, iparams);
      direct.emplace_back(addr_expr, ins_expr);
    }
    ++idx;
  }

  auto emit_pair_list = [](const std::vector<std::pair<std::string, std::string>>& vec,
                           std::size_t b, std::size_t e) {
    std::string s = "[";
    for (std::size_t k = b; k < e; ++k) {
      if (k > b) s += ", ";
      s += "(" + vec[k].first + ", " + vec[k].second + ", 4)";
    }
    s += "]";
    return s;
  };

  if (sub_chunks.empty()) {
    return hdr + "\n  " + emit_pair_list(direct, 0, direct.size()) + "\n";
  }

  std::vector<std::string> parts;
  std::size_t d = 0;
  std::size_t s = 0;
  while (s < sub_chunks.size()) {
    u64 sub_addr = sub_chunks[s].addr;
    std::vector<std::pair<std::string, std::string>> pre;
    while (d < direct.size()) {
      const std::string& ax = direct[d].first;
      u64 off = (ax == "L") ? 0 : std::stoull(ax.substr(4));
      if (rep.L + off >= sub_addr) break;
      pre.push_back(direct[d++]);
    }
    if (!pre.empty()) parts.push_back(emit_pair_list(pre, 0, pre.size()));
    parts.push_back(sub_chunks[s++].expr);
  }
  if (d < direct.size()) parts.push_back(emit_pair_list(direct, d, direct.size()));
  return hdr + "\n  " + join_strings(parts, " ++\n  ") + "\n";
}

std::string emit_templates(const Asm& A, const std::vector<OutEntry>& out,
                           const std::map<std::string, u64>& labels) {
  std::vector<InstInfo> insts;
  for (const auto& rec : A.macros) {
    const TemplateSpec& sp = spec_for(rec.name);
    u64 L = A.idx_addr[rec.start_idx];
    u64 size = A.idx_addr[rec.end_idx] - L;
    std::vector<std::string> rargs;
    for (const char* rp : sp.reg_params) rargs.push_back(std::to_string(rec.reg_params.at(rp)));
    std::vector<u64> aargs;
    for (const char* ap : sp.addr_params) aargs.push_back(labels.at(rec.addr_params.at(ap)));
    insts.push_back(InstInfo{&rec, L, size, std::move(rargs), std::move(aargs)});
  }
  std::string out_s;
  for (const auto& sp : template_specs()) {
    const InstInfo* rep = nullptr;
    for (const auto& ii : insts) {
      if (ii.rec->name == sp.name) {
        rep = &ii;
        break;
      }
    }
    require(rep != nullptr, std::string("missing template instance: ") + sp.name);
    out_s += format_template_fn(sp, A, out, labels, insts, *rep) + "\n";
  }
  return out_s;
}

std::string format_listing_head(char cmt, u64 size, u64 entry, u64 tripled, u64 vote, u64 guard) {
  auto ln = [&](const std::string& body) -> std::string {
    return body.empty() ? (std::string(1, cmt) + "\n") : (std::string(1, cmt) + " " + body + "\n");
  };
  std::string s;
  s += ln("spr1_check_arm64: AArch64 checker of Sporos bytecode \xe2\x80\x94 independent second check of stage 1 (Root A is bootstrap/roots/sporos/sporos-stage1.hex).");
  s += ln("");
  s += ln("Exit status: 0 when the file decodes and passes Sporos.checkProg; 5 when");
  s += ln("malformed or ill-typed; 6 when memory is refused or a stack height reaches");
  s += ln("2^48; 3 when the three redundant passes all disagree. Every non-zero status");
  s += ln("is at Hamming distance >= 2 from 0.");
  s += ln("");
  s += ln("Structure (ELF load address 0x400000 + file offset, PC-relative ADRP BSS):");
  s += ln("  start   (" + hex_0x(entry) + ")  mmap 64 KiB, read stdin growing with mremap,");
  s += ln("                       triplicate into two more mappings; descriptors");
  s += ln("                       (base, N) at 0x600000, 0x601000, 0x602000; init");
  s += ln("                       x19 = x20 = x21 = REJECT.");
  s += ln("  tripled (" + hex_0x(tripled) + ")  pass 1 (x19), pass 2 (x20), pass 3 (x21).");
  s += ln("                       Zero stack/x18 writes, zero stores or syscalls;");
  s += ln("                       writes only own result register (ACCEPT/REJECT/OVERFLOW).");
  s += ln("  vote    (" + hex_0x(vote) + ")  canonical 44-byte TMRVoterArm64 (x19,x20,x21),");
  s += ln("                       park b +0 (0x14000000) -> b +0x10 (0x14000004 to guard).");
  s += ln("  guard   (" + hex_0x(guard) + ")  dual independent checks of x19==ACCEPT and");
  s += ln("                       x20==ACCEPT before exit((x20 - 0x96) mod 256).");
  s += ln("");
  s += ln("Every instruction decodes via AArch64Phys.decodeWord (4-byte fixed width),");
  s += ln("no svc #0 or Hamming-1 neighbour of svc #0 occurs outside real syscalls.");
  s += ln("");
  s += ln("Total " + std::to_string(size) + " bytes.");
  return s;
}

void write_bytes_file(const std::filesystem::path& path, const std::vector<u8>& data) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  require(ofs.good(), "failed to open for write: " + path.string());
  ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
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

constexpr const char* kLeanHead = R"(-- Generated by bootstrap/model/spr1_check_arm64_gen.cpp.
-- Static EM_AARCH64 binary image, instruction table, and templates for Lean verification.
import Bootstrap.AArch64Env
import Bootstrap.AArch64Phys

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys

/-- Signed 64-bit PC-relative offset from `pc` to `tgt` (`(tgt - pc) mod 2^64`). -/
def relOff (pc tgt : Nat) : Nat := (tgt + W - pc % W) % W

def HMAX : Nat := 2 ^ 48

)";

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model_outdir = (argc > 1) ? argv[1] : ".";
  std::filesystem::path lean_outdir = (argc > 2) ? argv[2] : model_outdir;

  Asm A = gen_program();
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

  struct TableEntry {
    u64 addr;
    Ins ins;
  };
  std::vector<TableEntry> table;
  for (const auto& e : out) {
    if (e.ins.kind == Ins::Kind::Pad0 || e.ins.kind == Ins::Kind::Pad1 ||
        e.ins.kind == Ins::Kind::Pad2) {
      continue;
    }
    if (e.ins.kind == Ins::Kind::Voter) {
      require(e.bytes.size() == 44, "unexpected voter byte size");
      for (std::size_t off = 0; off < 44; off += 4) {
        std::size_t foff = (e.addr - kBase) + off;
        u32 w = static_cast<u32>(image[foff]) | (static_cast<u32>(image[foff + 1]) << 8) |
                (static_cast<u32>(image[foff + 2]) << 16) | (static_cast<u32>(image[foff + 3]) << 24);
        auto d = decode_word(w);
        require(d.has_value(), "failed to decode voter instruction");
        table.push_back(TableEntry{e.addr + off, *d});
      }
      continue;
    }
    require(e.bytes.size() == 4, "expected 4-byte instruction");
    std::size_t foff = static_cast<std::size_t>(e.addr - kBase);
    u32 w = static_cast<u32>(image[foff]) | (static_cast<u32>(image[foff + 1]) << 8) |
            (static_cast<u32>(image[foff + 2]) << 16) | (static_cast<u32>(image[foff + 3]) << 24);
    auto d = decode_word(w);
    require(d.has_value() && *d == e.ins, "decoded instruction mismatch at " + hex_0x(e.addr));
    table.push_back(TableEntry{e.addr, e.ins});
  }

  // Self-check: svc #0 (0xd4000001) only at intended syscalls, and no emitted
  // non-svc word is at Hamming distance <= 1 from svc #0.
  std::set<u64> sys_at;
  for (const auto& te : table) {
    if (te.ins.kind == Ins::Kind::Svc) sys_at.insert(te.addr);
  }
  for (std::size_t foff = kHdrLen; foff + 4 <= image.size(); foff += 4) {
    u32 w = static_cast<u32>(image[foff]) | (static_cast<u32>(image[foff + 1]) << 8) |
            (static_cast<u32>(image[foff + 2]) << 16) | (static_cast<u32>(image[foff + 3]) << 24);
    u64 addr = kBase + foff;
    if (w == 0xd4000001u) {
      require(sys_at.contains(addr), "unexpected svc #0 at " + hex_0x(addr));
    } else {
      u32 diff = w ^ 0xd4000001u;
      int bits = 0;
      while (diff) {
        bits += static_cast<int>(diff & 1u);
        diff >>= 1;
      }
      require(bits >= 2, "word within Hamming-1 of svc #0 at " + hex_0x(addr));
    }
  }

  // Exhaustive 32-bit x all instructions self-audit across [L_p1, L_p3_end) and [L_vote, L_core_end).
  u64 p1_addr = labels.at("p1");
  u64 p2_addr = labels.at("p2");
  u64 p3_addr = labels.at("p3");
  u64 p3_end_addr = labels.at("p3_end");
  u64 vote_addr = labels.at("vote");
  u64 core_end_addr = labels.at("core_end");
  u64 accept_svc_addr = labels.at("accept_svc");
  u64 start_addr = labels.at("start");
  u64 end_addr = kBase + image.size();
  auto read_word = [&](u64 a) -> u32 {
    std::size_t foff = static_cast<std::size_t>(a - kBase);
    return static_cast<u32>(image[foff]) | (static_cast<u32>(image[foff + 1]) << 8) |
           (static_cast<u32>(image[foff + 2]) << 16) | (static_cast<u32>(image[foff + 3]) << 24);
  };
  auto is_branch_kind = [](Ins::Kind k) {
    return k == Ins::Kind::B || k == Ins::Kind::Bcond || k == Ins::Kind::Cbz || k == Ins::Kind::Tbz;
  };
  u64 reloc_02cc_tgt = 0;
  u64 reloc_12f0_tgt = 0;
  u64 reloc_12e0_tgt = 0;
  u64 reloc_21d4_tgt = 0;
  for (u64 addr = p1_addr; addr < p3_end_addr; addr += 4) {
    u32 w = read_word(addr);
    auto orig = decode_word(w);
    require(orig.has_value(), "failed to decode pass instruction at " + hex_0x(addr));
    bool is_p3 = (addr >= p3_addr);
    bool is_p3_bvote = (addr == p3_addr + 2684 || addr == p3_addr + 2704 || addr == p3_addr + 2724);
    for (int bit = 0; bit < 32; ++bit) {
      u32 fw = w ^ (1u << bit);
      auto fd = decode_word(fw);
      if (!fd) continue;
      require(fd->kind != Ins::Kind::Svc, "1-bit flip in passes decodes to svc at " + hex_0x(addr));
      require(fd->kind != Ins::Kind::Br && fd->kind != Ins::Kind::Ret,
              "1-bit flip in passes decodes to indirect branch at " + hex_0x(addr));
      if (orig->kind == Ins::Kind::Adr) {
        require(fd->kind != Ins::Kind::Str && fd->kind != Ins::Kind::Stur &&
                    fd->kind != Ins::Kind::StrReg,
                "1-bit flip turns adrp into store at " + hex_0x(addr));
      }
      if (is_branch_kind(fd->kind)) {
        u64 tgt = addr + fd->imm;
        if (addr == p1_addr + 60 && bit == 16) reloc_02cc_tgt = tgt;
        if (addr == p2_addr + 1464 && bit == 15) reloc_12f0_tgt = tgt;
        if (addr == p2_addr + 1448 && bit == 15) reloc_12e0_tgt = tgt;
        if (addr == p3_addr + 2548 && bit == 11) reloc_21d4_tgt = tgt;
        require(tgt < kBase || tgt >= kBase + kHdrLen,
                "1-bit flip in passes reaches hdrBytes from " + hex_0x(addr));
        require(tgt < start_addr || tgt >= end_addr,
                "1-bit flip in passes reaches _start from " + hex_0x(addr));
        if (tgt >= vote_addr && tgt < core_end_addr) {
          require(is_p3 && is_p3_bvote,
                  "non-b-vote 1-bit flip in passes reaches TAIL from " + hex_0x(addr));
          u64 k = (tgt - vote_addr) / 4;
          require((tgt - vote_addr) % 4 == 0 &&
                      (k == 0 || k == 1 || k == 2 || k == 4 || k == 8),
                  "P3 b-vote 1-bit flip reaches unsafe TAIL index from " + hex_0x(addr));
        }
      }
    }
  }
  require(reloc_02cc_tgt >= p3_end_addr && reloc_02cc_tgt < vote_addr,
          "former 0x4002cc bit 16 did not land in PAD1_UDF");
  require(reloc_12f0_tgt >= p3_end_addr && reloc_12f0_tgt < vote_addr,
          "former 0x4012f0 bit 15 did not land in PAD1_UDF");
  require(reloc_12e0_tgt >= p3_end_addr && reloc_12e0_tgt < vote_addr,
          "former 0x4012e0 bit 15 did not land in PAD1_UDF");
  require(reloc_21d4_tgt >= p3_end_addr && reloc_21d4_tgt < vote_addr,
          "former 0x4021d4 bit 11 did not land in PAD1_UDF");

  for (u64 addr = vote_addr; addr < core_end_addr; addr += 4) {
    u32 w = read_word(addr);
    for (int bit = 0; bit < 32; ++bit) {
      u32 fw = w ^ (1u << bit);
      auto fd = decode_word(fw);
      if (!fd) continue;
      if (fd->kind == Ins::Kind::Svc) {
        // Flipping into svc can only happen from real svc -> no other word becomes svc.
        require(w == 0xd4000001u, "non-svc word flips to svc in tail");
      }
      require(fd->kind != Ins::Kind::Br && fd->kind != Ins::Kind::Ret,
              "1-bit flip in tail decodes to indirect branch at " + hex_0x(addr));
      if (is_branch_kind(fd->kind)) {
        u64 tgt = addr + fd->imm;
        if (tgt != addr) {
          require(tgt != accept_svc_addr && tgt != accept_svc_addr - 4,
                  "1-bit flip in tail branches directly to accept_svc/sub at " + hex_0x(addr));
        }
        require(tgt < kBase || tgt >= kBase + kHdrLen,
                "1-bit flip in tail reaches hdrBytes from " + hex_0x(addr));
        require(tgt < p1_addr || tgt >= p3_end_addr,
                "1-bit flip in tail reaches passes from " + hex_0x(addr));
        require(tgt < start_addr || tgt >= end_addr,
                "1-bit flip in tail jumps into start from " + hex_0x(addr));
      }
    }
  }

  std::filesystem::create_directories(model_outdir);
  std::filesystem::create_directories(lean_outdir);
  std::filesystem::path bin_path = model_outdir / "spr1_check_arm64";
  write_bytes_file(bin_path, image);
  std::filesystem::permissions(
      bin_path,
      std::filesystem::perms::owner_all | std::filesystem::perms::group_read |
          std::filesystem::perms::group_exec | std::filesystem::perms::others_read |
          std::filesystem::perms::others_exec,
      std::filesystem::perm_options::replace);

  std::map<u64, std::vector<std::string>> lab_at;
  for (const auto& [k, v] : label_order) lab_at[v].push_back(k);

  auto emit_listing = [&](bool binary_fmt) {
    char cmt = binary_fmt ? ':' : '#';
    std::size_t col = binary_fmt ? 38 : 20;
    std::string s = format_listing_head(cmt, filesz, labels.at("start"), labels.at("tripled"),
                                        labels.at("vote"), labels.at("guard"));
    s += "\n" + std::string(1, cmt) + " ---------------------------------------------------------------- ELF header\n";
    for (const auto& hc : kHdrComments) {
      if (hc.off == 0x40) {
        s += "\n" + std::string(1, cmt) + " ----------------------------------------------------------- program headers\n";
      }
      std::vector<u8> slice(image.begin() + static_cast<std::ptrdiff_t>(hc.off),
                            image.begin() + static_cast<std::ptrdiff_t>(hc.off + hc.len));
      std::string data = binary_fmt ? bin_bytes(slice) : hex_bytes(slice);
      std::size_t hdr_col = binary_fmt ? 76 : 32;
      s += pad_right(data, hdr_col) + std::string(1, cmt) + " " + raw_hex(hc.off, 3) + "  " +
           hc.comment + "\n";
    }
    s += "\n" + std::string(1, cmt) + " --------------------------------------------------------------------- code\n";
    s += std::string(1, cmt) + " Offsets on the left are file offsets; address = 0x400000 + offset.\n";

    for (const auto& e : out) {
      auto lit = lab_at.find(e.addr);
      if (lit != lab_at.end()) {
        for (const auto& l : lit->second) {
          if (l != "core_end" && l != "p3_end") s += "\n" + std::string(1, cmt) + " " + l + ":\n";
        }
      }
      if (e.ins.kind == Ins::Kind::Pad0 || e.ins.kind == Ins::Kind::Pad1 ||
          e.ins.kind == Ins::Kind::Pad2) {
        s += std::string(1, cmt) + " " + ins_str(e.ins, e.addr) + " (" +
             std::to_string(e.bytes.size()) + " bytes of udf #2032)\n";
        std::size_t per_line = 16;
        for (std::size_t j = 0; j < e.bytes.size(); j += per_line) {
          std::size_t end = std::min(j + per_line, e.bytes.size());
          std::vector<u8> seg(e.bytes.begin() + static_cast<std::ptrdiff_t>(j),
                              e.bytes.begin() + static_cast<std::ptrdiff_t>(end));
          s += (binary_fmt ? bin_bytes(seg) : hex_bytes(seg)) + "\n";
        }
        continue;
      }
      if (e.ins.kind == Ins::Kind::Voter) {
        s += std::string(1, cmt) + " " + e.comment + "\n";
        for (std::size_t off = 0; off < e.bytes.size(); off += 4) {
          std::size_t foff = (e.addr - kBase) + off;
          u32 w = static_cast<u32>(image[foff]) | (static_cast<u32>(image[foff + 1]) << 8) |
                  (static_cast<u32>(image[foff + 2]) << 16) | (static_cast<u32>(image[foff + 3]) << 24);
          auto d = decode_word(w);
          require(d.has_value(), "voter decode failed");
          std::vector<u8> seg(e.bytes.begin() + static_cast<std::ptrdiff_t>(off),
                              e.bytes.begin() + static_cast<std::ptrdiff_t>(off + 4));
          std::string data = binary_fmt ? bin_bytes(seg) : hex_bytes(seg);
          s += pad_right(data, col) + std::string(1, cmt) + " " +
               raw_hex((e.addr - kBase) + off, 3) + "  " + ins_str(*d, e.addr + off) + "\n";
        }
        continue;
      }
      std::string data = binary_fmt ? bin_bytes(e.bytes) : hex_bytes(e.bytes);
      s += pad_right(data, col) + std::string(1, cmt) + " " + raw_hex(e.addr - kBase, 3) + "  " +
           ins_str(e.ins, e.addr) + (e.comment.empty() ? "" : ("   " + e.comment)) + "\n";
    }
    return s;
  };

  write_text_file(model_outdir / "spr1_check_arm64.bin", emit_listing(true));
  write_text_file(model_outdir / "spr1_check_arm64.hex", emit_listing(false));

  // Write SporosCheckArm64BinData.lean
  {
    std::size_t pass_start = static_cast<std::size_t>(labels.at("tripled") - kBase);
    std::size_t pass_end = static_cast<std::size_t>(labels.at("p3_end") - kBase);
    std::size_t tail_start = static_cast<std::size_t>(labels.at("vote") - kBase);
    std::size_t tail_end = static_cast<std::size_t>(labels.at("core_end") - kBase);
    std::size_t init_start = static_cast<std::size_t>(labels.at("start") - kBase);
    std::vector<u8> hdr_bytes(image.begin(), image.begin() + static_cast<std::ptrdiff_t>(kHdrLen));
    std::vector<u8> core_pass_bytes(image.begin() + static_cast<std::ptrdiff_t>(pass_start),
                                    image.begin() + static_cast<std::ptrdiff_t>(pass_end));
    std::vector<u8> tail_bytes(image.begin() + static_cast<std::ptrdiff_t>(tail_start),
                               image.begin() + static_cast<std::ptrdiff_t>(tail_end));
    std::vector<u8> init_bytes(image.begin() + static_cast<std::ptrdiff_t>(init_start), image.end());

    std::vector<TableEntry> core_pass_table, tail_table, init_table;
    for (const auto& te : table) {
      if (te.addr >= labels.at("tripled") && te.addr < labels.at("p3_end")) {
        core_pass_table.push_back(te);
      } else if (te.addr >= labels.at("vote") && te.addr < labels.at("core_end")) {
        tail_table.push_back(te);
      } else if (te.addr >= labels.at("start")) {
        init_table.push_back(te);
      }
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
    emit_byte_chunks("corePass", "corePassBytes", core_pass_bytes);
    emit_byte_chunks("tail", "tailBytes", tail_bytes);
    emit_byte_chunks("init", "initBytes", init_bytes);
    lean_out += "def padWordBytes : List Nat := wordBytes " + hex_0x(kUdfWord) + "\n";
    lean_out += "def pad0Bytes : List Nat := (List.replicate " + std::to_string(kPad0Bytes / 4) +
                " padWordBytes).flatten\n";
    lean_out += "def pad1Bytes : List Nat := (List.replicate " + std::to_string(kPad1Bytes / 4) +
                " padWordBytes).flatten\n";
    lean_out += "def pad2Bytes : List Nat := (List.replicate " + std::to_string(kPad2Bytes / 4) +
                " padWordBytes).flatten\n";
    lean_out += "\n/-- bootstrap/model/spr1_check_arm64 -/\ndef spr1Arm64Bytes : List Nat :=\n"
                "  hdrBytes ++ pad0Bytes ++ corePassBytes ++ pad1Bytes ++ tailBytes ++ pad2Bytes ++ initBytes\n\n";

    auto emit_tbl = [&](std::string_view name, const std::vector<TableEntry>& tbl) {
      std::vector<std::string> names;
      for (std::size_t i = 0, j = 0; i < tbl.size(); i += 64, ++j) {
        std::size_t end = std::min(i + 64, tbl.size());
        std::string nm = std::string(name) + "_" + std::to_string(j);
        lean_out += "def " + nm + " : List (Nat × Ins × Nat) := [\n";
        for (std::size_t k = i; k < end; ++k) {
          if (k > i) lean_out += ",\n";
          lean_out += "  (" + std::to_string(tbl[k].addr) + ", " + lean_ins(tbl[k].ins) + ", 4)";
        }
        lean_out += "]\n";
        names.push_back(std::move(nm));
      }
      lean_out += "def " + std::string(name) + " : List (Nat × Ins × Nat) :=\n  " +
                  join_strings(names, " ++\n  ") + "\n\n";
    };
    emit_tbl("corePassTable", core_pass_table);
    emit_tbl("tailTable", tail_table);
    emit_tbl("initTable", init_table);

    lean_out += "/-- Every instruction of the code, at its address, with fixed length 4. -/\n";
    lean_out += "def spr1Arm64Table : List (Nat × Ins × Nat) := corePassTable ++ tailTable ++ initTable\n\n";
    lean_out += "def ACCEPT : Nat := " + hex_0x(kAccept) + "\n";
    lean_out += "def REJECT : Nat := " + hex_0x(kReject) + "\n";
    lean_out += "def OVERFLOW : Nat := " + hex_0x(kOverflow) + "\n\n";
    lean_out += emit_templates(A, out, labels);

    std::vector<std::pair<std::string, u64>> sorted_labels = label_order;
    std::stable_sort(sorted_labels.begin(), sorted_labels.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });
    lean_out += "/-- Addresses named in the listing. -/\n";
    for (const auto& [k, v] : sorted_labels) {
      if (keep_lean_label(k)) {
        lean_out += "def L_" + k + " : Nat := " + hex_0x(v) + "\n";
      }
    }
    lean_out += "\nend SporosCheckArm64Bin\n";
    write_text_file(lean_outdir / "SporosCheckArm64BinData.lean", lean_out);
  }

  std::printf("size %llu bytes, %zu instructions, entry %s, tripled %s, p3_end %s, vote %s, guard %s\n",
              static_cast<unsigned long long>(filesz), table.size(),
              hex_0x(labels.at("start")).c_str(), hex_0x(labels.at("tripled")).c_str(),
              hex_0x(labels.at("p3_end")).c_str(), hex_0x(labels.at("vote")).c_str(),
              hex_0x(labels.at("guard")).c_str());
  std::printf("fetch_audit_ok: pad0=%llu pad1=%llu pad2=%llu reloc_0x4002cc_b16=%s@%s reloc_0x4012f0_b15=%s@%s reloc_0x4012e0_b15=%s@%s reloc_0x4021d4_b11=%s@%s\n",
              static_cast<unsigned long long>(kPad0Bytes),
              static_cast<unsigned long long>(kPad1Bytes),
              static_cast<unsigned long long>(kPad2Bytes),
              hex_0x(reloc_02cc_tgt).c_str(), hex_0x(p1_addr + 60).c_str(),
              hex_0x(reloc_12f0_tgt).c_str(), hex_0x(p2_addr + 1464).c_str(),
              hex_0x(reloc_12e0_tgt).c_str(), hex_0x(p2_addr + 1448).c_str(),
              hex_0x(reloc_21d4_tgt).c_str(), hex_0x(p3_addr + 2548).c_str());
  return 0;
}
