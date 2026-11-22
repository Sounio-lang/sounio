// Generator of sporos_vm (static zero-libc Linux x86-64 ELF64 binary),
// bootstrap/model/sporos_vm.hex (Format A listing for seed-a), and
// formal/Bootstrap/SporosVmBinData.lean.
//
// Root A framing: Root A is MUSE's hand-written checker
// (`bootstrap/roots/sporos/sporos-stage1.hex`). `spr1_check` is an independent
// second check (not Root A or Root A's trust base). `sporos_vm` is the
// post-admission native x86-64 Sporos bytecode VM paired with `Sporos.decode`
// and `Sporos.checkProg` (`formal/Sporos/Bytecode.lean`, `Check.lean`),
// executing `SporosProvLen.lstep` / `Sporos.step _ noPhys` with TMR protection
// over 3 read-only copies (`code1`, `code2`, `code3`) of the flat
// word/data-meta/instruction table and the canonical 38-byte TMR voter at
// `L_vm_vote`.
//
// Every emitted instruction is self-checked against a C++23 transcription of
// `Bootstrap.X86Vm.decode` (`X86Dec.decode` + `X86Vm.decodeVm`), ensuring
// 0f 85 never occurs and 0f 05 occurs solely at syscalls.
//
//   sporos_vm_gen [OUTDIR]
//     writes sporos_vm, sporos_vm.hex, SporosVmBinData.lean (default OUTDIR: .)

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
  std::fprintf(stderr, "sporos_vm_gen: %.*s\n", static_cast<int>(msg.size()), msg.data());
  std::abort();
}

void require(bool cond, std::string_view msg) {
  if (!cond) fail_msg(msg);
}

// ---------------------------------------------------------------------------
// Constants & VM Context (`VmCtx` at 0x600000)
// ---------------------------------------------------------------------------

constexpr u64 kBase = 0x400000ULL;
constexpr u64 kCtxBase = 0x600000ULL;
constexpr u64 kCtxSize = 0x1000ULL;
constexpr u64 kHdrLen = 0xb0ULL;
constexpr u64 kPage = 4096ULL;
constexpr u64 kDefaultMsize = 1ULL << 24;
constexpr u64 kMaxMsize = 1ULL << 44;

constexpr int RAX = 0, RCX = 1, RDX = 2, RBX = 3, RSP = 4;
[[maybe_unused]] constexpr int RBP = 5;
constexpr int RSI = 6, RDI = 7;
constexpr int R8 = 8, R9 = 9, R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15;

constexpr std::array<const char*, 16> kRegName = {
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
};

constexpr i32 OFF_DS_BASE = 0x00;
constexpr i32 OFF_DS_LEN = 0x08;
constexpr i32 OFF_DS_CAP = 0x10;
constexpr i32 OFF_AUX_BASE = 0x18;
constexpr i32 OFF_AUX_LEN = 0x20;
constexpr i32 OFF_AUX_CAP = 0x28;
constexpr i32 OFF_RS_BASE = 0x30;
constexpr i32 OFF_RS_LEN = 0x38;
constexpr i32 OFF_RS_CAP = 0x40;
constexpr i32 OFF_LOCS_BASE = 0x48;
constexpr i32 OFF_LOCS_LEN = 0x50;
constexpr i32 OFF_LOCS_CAP = 0x58;
constexpr i32 OFF_MEMW_BASE = 0x60;
constexpr i32 OFF_MEMP_BASE = 0x68;
constexpr i32 OFF_MSIZE = 0x70;
constexpr i32 OFF_HP = 0x78;
constexpr i32 OFF_TAB_HEADS = 0x80;
constexpr i32 OFF_TABS_LEN = 0x88;
constexpr i32 OFF_TABS_CAP = 0x90;
constexpr i32 OFF_TAB_NODES = 0x98;
constexpr i32 OFF_TAB_NODES_LEN = 0xa0;
constexpr i32 OFF_TAB_NODES_CAP = 0xa8;
constexpr i32 OFF_INP_BASE = 0xb0;
constexpr i32 OFF_INP_LEN = 0xb8;
constexpr i32 OFF_IPOS = 0xc0;
constexpr i32 OFF_OUT_BASE = 0xc8;
constexpr i32 OFF_OUT_LEN = 0xd0;
constexpr i32 OFF_OUT_CAP = 0xd8;
constexpr i32 OFF_PC_PROV = 0xe0;
constexpr i32 OFF_CODE1 = 0xe8;
constexpr i32 OFF_CODE2 = 0xf0;
constexpr i32 OFF_CODE3 = 0xf8;
constexpr i32 OFF_CUR_OP = 0x100;
constexpr i32 OFF_CUR_IMM = 0x108;
constexpr i32 OFF_CUR_Q = 0x110;
constexpr i32 OFF_CUR_EIN = 0x118;
constexpr i32 OFF_CUR_EOUT = 0x120;
constexpr i32 OFF_A0_W = 0x128;
constexpr i32 OFF_A0_P = 0x130;
constexpr i32 OFF_A1_W = 0x138;
constexpr i32 OFF_A1_P = 0x140;
constexpr i32 OFF_A2_W = 0x148;
constexpr i32 OFF_A2_P = 0x150;
constexpr i32 OFF_A3_W = 0x158;
constexpr i32 OFF_A3_P = 0x160;
constexpr i32 OFF_RES0_W = 0x168;
constexpr i32 OFF_RES0_P = 0x170;
constexpr i32 OFF_RES1_W = 0x178;
constexpr i32 OFF_RES1_P = 0x180;
constexpr i32 OFF_RES2_W = 0x188;
constexpr i32 OFF_RES2_P = 0x190;
constexpr i32 OFF_ARGV1 = 0x198;
constexpr i32 OFF_FILE_BASE = 0x1a0;
constexpr i32 OFF_FILE_LEN = 0x1a8;
constexpr i32 OFF_FILE_CAP = 0x1b0;
constexpr i32 OFF_FILE_FD = 0x1b8;
constexpr i32 OFF_HALT_CODE = 0x1c0;
constexpr i32 OFF_ERR_ID = 0x1c8;
constexpr i32 OFF_MAIN = 0x1d0;
constexpr i32 OFF_NW = 0x1d8;
constexpr i32 OFF_ND = 0x1e0;
constexpr i32 OFF_TOT_INSTRS = 0x1e8;
constexpr i32 OFF_TOT_DATA = 0x1f0;
constexpr i32 OFF_WORDS_FILE_OFF = 0x1f8;
constexpr i32 OFF_TABLE_BYTES = 0x200;
constexpr i32 OFF_HBUF_BASE = 0x208;
constexpr i32 OFF_HBUF_CAP = 0x210;
constexpr i32 OFF_CHK_W = 0x218;
constexpr i32 OFF_CHK_NC = 0x220;
constexpr i32 OFF_CHK_NA = 0x228;
constexpr i32 OFF_CHK_ANN = 0x230;
constexpr i32 OFF_CHK_CURD = 0x238;
constexpr i32 OFF_CHK_CURA = 0x240;
constexpr i32 OFF_CHK_HD = 0x248;
constexpr i32 OFF_CHK_HA = 0x250;
constexpr i32 OFF_CHK_WBASE = 0x258;
constexpr i32 OFF_CHK_INP_CAP = 0x260;
constexpr i32 OFF_TMP0 = 0x268;
constexpr i32 OFF_TMP1 = 0x270;

constexpr int kNPrims = 57;
constexpr int kNOps = 70;

constexpr std::array<u8, 57> kEffIn = {
    1, 1, 2, 2, 3, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 1,
    2, 2, 2, 1, 2, 2, 2,
    2, 2, 2, 2, 2, 2,
    1, 2, 1, 2, 1, 2, 1, 2,
    3, 3, 3,
    1, 0, 1,
    4, 2,
    0, 3, 2,
    0, 1, 1, 1,
    0, 0, 1,
    1, 1, 2,
};

constexpr std::array<u8, 57> kEffOut = {
    2, 0, 2, 3, 3, 1, 3,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 0, 1, 0, 1, 0, 1, 0,
    0, 0, 1,
    1, 1, 0,
    1, 1,
    1, 0, 2,
    1, 0, 0, 0,
    1, 1, 1,
    2, 2, 1,
};

constexpr int OP_LIT = 57;
constexpr int OP_STR = 58;
constexpr int OP_PICK = 59;
constexpr int OP_TOR = 60;
constexpr int OP_FROMR = 61;
constexpr int OP_RFETCH = 62;
constexpr int OP_LOCGET = 63;
constexpr int OP_LOCSET = 64;
constexpr int OP_JMP = 65;
constexpr int OP_JZ = 66;
constexpr int OP_JNZ = 67;
constexpr int OP_CALL = 68;
constexpr int OP_RET = 69;

enum ErrId : u64 {
  ERR_UNDERFLOW = 0,
  ERR_AUX_UNDERFLOW = 1,
  ERR_BAD_LOCAL = 2,
  ERR_BAD_JUMP = 3,
  ERR_BAD_WORD = 4,
  ERR_NO_FRAME = 5,
  ERR_BOUNDS = 6,
  ERR_DIV_ZERO = 7,
  ERR_BAD_TABLE = 8,
  ERR_DEMAND = 9,
};

// ---------------------------------------------------------------------------
// Formatting utilities
// ---------------------------------------------------------------------------

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
  if (s < 0) return "-0x" + raw_hex(0ULL - static_cast<u64>(s));
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
  for (int i = 0; i < n; ++i) out.push_back(static_cast<u8>((v >> (8 * i)) & 0xffULL));
}

std::string join_strings(const std::vector<std::string>& v, std::string_view sep) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i > 0) out += sep;
    out += v[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// X86Vm.Ins representation & encoders
// ---------------------------------------------------------------------------

enum class AluOp : int { Add = 0, Or = 1, And = 2, Sub = 3, Xor = 4 };
enum class MemW : int { W8 = 0, W16 = 1, W32 = 2, W64 = 3 };
enum class Cc : int {
  O = 0, NO = 1, B = 2, AE = 3, E = 4, NE = 5, BE = 6, A = 7,
  S = 8, NS = 9, L = 12, GE = 13, LE = 14, G = 15,
};

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

const char* memw_name(MemW w) {
  switch (w) {
    case MemW::W8: return "w8";
    case MemW::W16: return "w16";
    case MemW::W32: return "w32";
    case MemW::W64: return "w64";
  }
  fail_msg("invalid MemW");
}

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

struct Ins {
  enum class Kind {
    MovI, Mov, Mov32, Alu, AluI, CmpI, Cmp, Test, ShlI, ShrI,
    Ld, Ld32, Ld8, St, St8, Lea, Syscall, Jmp, Jcc,
    PushI, Push, Pop, Call, Ret, Jtab, JmpR,
    Mul, Imul, Div, Idiv, Cqo, Neg, Not,
    ShlR, ShrR, SarR, SarI,
    Ld16, St16, St32,
    LdIdx, StIdx, LeaIdx,
    TestI, CmpAlI, TestAlI, Setcc, Cmovcc,
    Voter,
  };

  Kind kind = Kind::Syscall;
  AluOp alu_op = AluOp::Add;
  MemW mw = MemW::W64;
  int r0 = 0;
  int r1 = 0;
  int r2 = 0;
  int scale = 1;
  u64 imm = 0;

  bool operator==(const Ins& o) const = default;
};

struct Encoded {
  std::vector<u8> bytes = {};
  Ins ins = {};
};

u8 rex(int w, int r, int x, int b) {
  return static_cast<u8>(0x40 | (w << 3) | (r << 2) | (x << 1) | b);
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

void append_memop(std::vector<u8>& out, int reg, int base, i32 disp) {
  int rm = base & 7;
  int md = 0;
  if (disp == 0 && rm != 5) md = 0;
  else if (disp >= -128 && disp < 128) md = 1;
  else md = 2;
  out.push_back(static_cast<u8>((md << 6) | ((reg & 7) << 3) | rm));
  if (rm == 4) out.push_back(0x24);
  if (md == 1) push_le(out, static_cast<u64>(static_cast<u8>(disp)), 1);
  else if (md == 2) push_le(out, static_cast<u64>(static_cast<u32>(disp)), 4);
}

void append_memop_idx(std::vector<u8>& out, int reg, int base, int idx, int scale, i32 disp) {
  require(idx != 4, "SIB index cannot be RSP (4)");
  int sc = (scale == 1) ? 0 : (scale == 2) ? 1 : (scale == 4) ? 2 : (scale == 8) ? 3 : -1;
  require(sc >= 0, "SIB scale must be 1,2,4,8");
  int b3 = base & 7;
  int md = 0;
  if (disp == 0 && b3 != 5) md = 0;
  else if (disp >= -128 && disp < 128) md = 1;
  else md = 2;
  out.push_back(static_cast<u8>((md << 6) | ((reg & 7) << 3) | 4));
  out.push_back(static_cast<u8>((sc << 6) | ((idx & 7) << 3) | b3));
  if (md == 1) push_le(out, static_cast<u64>(static_cast<u8>(disp)), 1);
  else if (md == 2) push_le(out, static_cast<u64>(static_cast<u32>(disp)), 4);
}

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

Encoded e_mov(int d, int s) {
  std::vector<u8> b = {rex(1, d >> 3, 0, s >> 3), 0x8b,
                       static_cast<u8>(0xc0 | ((d & 7) << 3) | (s & 7))};
  return {std::move(b), Ins{.kind = Ins::Kind::Mov, .r0 = d, .r1 = s}};
}

Encoded e_alu(AluOp op, int d, int s) {
  std::vector<u8> b = {rex(1, s >> 3, 0, d >> 3), alu_rr_opcode(op),
                       static_cast<u8>(0xc0 | ((s & 7) << 3) | (d & 7))};
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
    require(s >= -(1LL << 31) && s < (1LL << 31), "aluI/cmpI imm32 overflow");
    b.push_back(0x81);
    b.push_back(static_cast<u8>(0xc0 | (ext << 3) | (d & 7)));
    push_le(b, static_cast<u64>(static_cast<u32>(static_cast<i32>(s))), 4);
  }
  return b;
}

Encoded e_aluI(AluOp op, int d, i64 imm) {
  u64 uimm = static_cast<u64>(imm);
  return {encode_ext(alu_ext(op), d, uimm),
          Ins{.kind = Ins::Kind::AluI, .alu_op = op, .r0 = d, .imm = uimm}};
}

Encoded e_cmpI(int a, i64 imm) {
  u64 uimm = static_cast<u64>(imm);
  return {encode_ext(7, a, uimm), Ins{.kind = Ins::Kind::CmpI, .r0 = a, .imm = uimm}};
}

Encoded e_cmp(int a, int b) {
  std::vector<u8> bs = {rex(1, b >> 3, 0, a >> 3), 0x39,
                        static_cast<u8>(0xc0 | ((b & 7) << 3) | (a & 7))};
  return {std::move(bs), Ins{.kind = Ins::Kind::Cmp, .r0 = a, .r1 = b}};
}

Encoded e_test(int a, int b) {
  std::vector<u8> bs = {rex(1, b >> 3, 0, a >> 3), 0x85,
                        static_cast<u8>(0xc0 | ((b & 7) << 3) | (a & 7))};
  return {std::move(bs), Ins{.kind = Ins::Kind::Test, .r0 = a, .r1 = b}};
}

Encoded e_shlI(int d, u8 c) {
  return {{rex(1, 0, 0, d >> 3), 0xc1, static_cast<u8>(0xe0 | (d & 7)), c},
          Ins{.kind = Ins::Kind::ShlI, .r0 = d, .imm = c}};
}

Encoded e_shrI(int d, u8 c) {
  return {{rex(1, 0, 0, d >> 3), 0xc1, static_cast<u8>(0xe8 | (d & 7)), c},
          Ins{.kind = Ins::Kind::ShrI, .r0 = d, .imm = c}};
}

Encoded e_shlR(int d) {
  return {{rex(1, 0, 0, d >> 3), 0xd3, static_cast<u8>(0xe0 | (d & 7))},
          Ins{.kind = Ins::Kind::ShlR, .r0 = d}};
}

Encoded e_shrR(int d) {
  return {{rex(1, 0, 0, d >> 3), 0xd3, static_cast<u8>(0xe8 | (d & 7))},
          Ins{.kind = Ins::Kind::ShrR, .r0 = d}};
}

Encoded e_sarR(int d) {
  return {{rex(1, 0, 0, d >> 3), 0xd3, static_cast<u8>(0xf8 | (d & 7))},
          Ins{.kind = Ins::Kind::SarR, .r0 = d}};
}

Encoded e_neg(int d) {
  return {{rex(1, 0, 0, d >> 3), 0xf7, static_cast<u8>(0xd8 | (d & 7))},
          Ins{.kind = Ins::Kind::Neg, .r0 = d}};
}

Encoded e_not(int d) {
  return {{rex(1, 0, 0, d >> 3), 0xf7, static_cast<u8>(0xd0 | (d & 7))},
          Ins{.kind = Ins::Kind::Not, .r0 = d}};
}

Encoded e_imul(int d, int s) {
  return {{rex(1, d >> 3, 0, s >> 3), 0x0f, 0xaf,
           static_cast<u8>(0xc0 | ((d & 7) << 3) | (s & 7))},
          Ins{.kind = Ins::Kind::Imul, .r0 = d, .r1 = s}};
}

Encoded e_div(int s) {
  return {{rex(1, 0, 0, s >> 3), 0xf7, static_cast<u8>(0xf0 | (s & 7))},
          Ins{.kind = Ins::Kind::Div, .r0 = s}};
}

Encoded e_idiv(int s) {
  return {{rex(1, 0, 0, s >> 3), 0xf7, static_cast<u8>(0xf8 | (s & 7))},
          Ins{.kind = Ins::Kind::Idiv, .r0 = s}};
}

Encoded e_cqo() { return {{0x48, 0x99}, Ins{.kind = Ins::Kind::Cqo}}; }

Encoded e_setcc(Cc cc, int d) {
  std::vector<u8> bs;
  if (d >= 4) bs.push_back(rex(0, 0, 0, d >> 3));
  bs.push_back(0x0f);
  bs.push_back(static_cast<u8>(0x90 + static_cast<int>(cc)));
  bs.push_back(static_cast<u8>(0xc0 | (d & 7)));
  return {std::move(bs), Ins{.kind = Ins::Kind::Setcc, .r0 = d, .imm = static_cast<u64>(cc)}};
}

Encoded e_cmovcc(Cc cc, int d, int s) {
  return {{rex(1, d >> 3, 0, s >> 3), 0x0f, static_cast<u8>(0x40 + static_cast<int>(cc)),
           static_cast<u8>(0xc0 | ((d & 7) << 3) | (s & 7))},
          Ins{.kind = Ins::Kind::Cmovcc, .r0 = d, .r1 = s, .imm = static_cast<u64>(cc)}};
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

Encoded e_st(int b, i32 disp, int s) {
  std::vector<u8> bs = {rex(1, s >> 3, 0, b >> 3), 0x89};
  append_memop(bs, s, b, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::St, .r0 = b, .r1 = s, .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_lea(int d, int b, i32 disp) {
  std::vector<u8> bs = {rex(1, d >> 3, 0, b >> 3), 0x8d};
  append_memop(bs, d, b, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::Lea, .r0 = d, .r1 = b, .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_ld_idx(MemW mw, int d, int b, int idx, int scale, i32 disp = 0) {
  std::vector<u8> bs;
  int wbit = (mw == MemW::W64) ? 1 : 0;
  if (wbit || d >= 8 || idx >= 8 || b >= 8) bs.push_back(rex(wbit, d >> 3, idx >> 3, b >> 3));
  if (mw == MemW::W8) {
    bs.push_back(0x0f);
    bs.push_back(0xb6);
  } else if (mw == MemW::W16) {
    bs.push_back(0x0f);
    bs.push_back(0xb7);
  } else {
    bs.push_back(0x8b);
  }
  append_memop_idx(bs, d, b, idx, scale, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::LdIdx,
              .mw = mw,
              .r0 = d,
              .r1 = b,
              .r2 = idx,
              .scale = scale,
              .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_st_idx(MemW mw, int b, int idx, int scale, i32 disp, int s) {
  std::vector<u8> bs;
  if (mw == MemW::W16) bs.push_back(0x66);
  int wbit = (mw == MemW::W64) ? 1 : 0;
  bool need_rex = (wbit != 0) || (s >= 8) || (idx >= 8) || (b >= 8) || (mw == MemW::W8 && s >= 4);
  if (need_rex) bs.push_back(rex(wbit, s >> 3, idx >> 3, b >> 3));
  bs.push_back(mw == MemW::W8 ? 0x88 : 0x89);
  append_memop_idx(bs, s, b, idx, scale, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::StIdx,
              .mw = mw,
              .r0 = b,
              .r1 = idx,
              .r2 = s,
              .scale = scale,
              .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_lea_idx(int d, int b, int idx, int scale, i32 disp = 0) {
  std::vector<u8> bs = {rex(1, d >> 3, idx >> 3, b >> 3), 0x8d};
  append_memop_idx(bs, d, b, idx, scale, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::LeaIdx,
              .r0 = d,
              .r1 = b,
              .r2 = idx,
              .scale = scale,
              .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_jtab(int b, int idx, i32 disp = 0) {
  require(idx != 4, "jtab idx cannot be RSP");
  std::vector<u8> bs;
  if (idx >= 8 || b >= 8) bs.push_back(rex(0, 0, idx >> 3, b >> 3));
  bs.push_back(0xff);
  append_memop_idx(bs, 4, b, idx, 8, disp);
  return {std::move(bs),
          Ins{.kind = Ins::Kind::Jtab,
              .r0 = b,
              .r1 = idx,
              .imm = static_cast<u64>(static_cast<i64>(disp))}};
}

Encoded e_ret() { return {{0xc3}, Ins{.kind = Ins::Kind::Ret}}; }

Encoded e_syscall() { return {{0x0f, 0x05}, Ins{.kind = Ins::Kind::Syscall}}; }

// ---------------------------------------------------------------------------
// Exact C++ transcription of X86Vm.decode (`X86Dec.decode` + `X86Vm.decodeVm`)
// ---------------------------------------------------------------------------

u64 sx8(u8 v) { return static_cast<u64>(static_cast<i64>(static_cast<std::int8_t>(v))); }
u64 sx32(u32 v) { return static_cast<u64>(static_cast<i64>(static_cast<i32>(v))); }

struct ImageSlice {
  const std::vector<u8>& image;
  std::size_t base_off;

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

struct MemOpIdxDec {
  int base;
  int idx;
  int scale;
  u64 disp;
  std::size_t next_off;
};

std::optional<MemOpIdxDec> mem_op_idx(const ImageSlice& b, std::size_t off, int m, int bb, bool x) {
  int md = m / 64;
  int rm = m % 8;
  if (md == 3 || rm != 4) return std::nullopt;
  auto s_opt = b.at(off);
  if (!s_opt.has_value()) return std::nullopt;
  int s = *s_opt;
  int idx = (s / 8) % 8 + (x ? 8 : 0);
  if (idx == 4) return std::nullopt;
  if (md == 0 && (s % 8) == 5) return std::nullopt;
  int scale = 1 << (s / 64);
  int base = (s % 8) + bb;
  if (md == 0) return MemOpIdxDec{base, idx, scale, 0ULL, off + 1};
  if (md == 1) {
    auto d = b.at(off + 1);
    if (!d.has_value()) return std::nullopt;
    return MemOpIdxDec{base, idx, scale, sx8(*d), off + 2};
  }
  auto d = le_bytes(b, off + 1, 4);
  if (!d.has_value()) return std::nullopt;
  return MemOpIdxDec{base, idx, scale, sx32(static_cast<u32>(*d)), off + 5};
}

struct Decoded {
  Ins ins;
  std::size_t len;
};

std::optional<Decoded> decode_int(const ImageSlice& b, u64 a) {
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

std::optional<Decoded> decode_vm(const ImageSlice& b) {
  auto b0_opt = b.at(0);
  if (!b0_opt.has_value()) return std::nullopt;
  u8 b0 = *b0_opt;
  if (b0 == 0x66) {
    auto b1_opt = b.at(1);
    if (!b1_opt.has_value()) return std::nullopt;
    u8 b1 = *b1_opt;
    bool has_rex = (b1 >= 0x40 && b1 <= 0x4f);
    int rx = has_rex ? (b1 - 0x40) : 0;
    std::size_t p = has_rex ? 2 : 1;
    bool w = ((rx / 8) % 2) == 1;
    int r = ((rx / 4) % 2) * 8;
    bool x = ((rx / 2) % 2) == 1;
    int bb = (rx % 2) * 8;
    if (w) return std::nullopt;
    auto o_opt = b.at(p);
    if (!o_opt.has_value() || *o_opt != 0x89) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || (*m_opt / 64) == 3) return std::nullopt;
    int m = *m_opt;
    int rg = (m / 8) % 8 + r;
    if (!x) {
      auto mo = mem_op(b, p + 2, m, bb, x);
      if (mo.has_value()) {
        return Decoded{Ins{.kind = Ins::Kind::St16, .r0 = mo->base, .r1 = rg, .imm = mo->disp},
                       mo->next_off};
      }
    }
    auto moi = mem_op_idx(b, p + 2, m, bb, x);
    if (!moi.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::StIdx,
                       .mw = MemW::W16,
                       .r0 = moi->base,
                       .r1 = moi->idx,
                       .r2 = rg,
                       .scale = moi->scale,
                       .imm = moi->disp},
                   moi->next_off};
  }
  bool has_rex = (b0 >= 0x40 && b0 <= 0x4f);
  int rx = has_rex ? (b0 - 0x40) : 0;
  std::size_t p = has_rex ? 1 : 0;
  bool w = ((rx / 8) % 2) == 1;
  int r = ((rx / 4) % 2) * 8;
  bool x = ((rx / 2) % 2) == 1;
  int bb = (rx % 2) * 8;
  auto o_opt = b.at(p);
  if (!o_opt.has_value()) return std::nullopt;
  u8 o = *o_opt;

  if (o == 0xf7) {
    if (!w || r != 0 || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || (*m_opt / 64) != 3) return std::nullopt;
    int m = *m_opt;
    int d = (m % 8) + bb;
    int ext = (m / 8) % 8;
    if (ext == 0) {
      auto imm = le_bytes(b, p + 2, 4);
      if (!imm.has_value()) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::TestI, .r0 = d, .imm = sx32(static_cast<u32>(*imm))},
                     p + 6};
    }
    if (ext == 2) return Decoded{Ins{.kind = Ins::Kind::Not, .r0 = d}, p + 2};
    if (ext == 3) return Decoded{Ins{.kind = Ins::Kind::Neg, .r0 = d}, p + 2};
    if (ext == 4) return Decoded{Ins{.kind = Ins::Kind::Mul, .r0 = d}, p + 2};
    if (ext == 6) return Decoded{Ins{.kind = Ins::Kind::Div, .r0 = d}, p + 2};
    if (ext == 7) return Decoded{Ins{.kind = Ins::Kind::Idiv, .r0 = d}, p + 2};
    return std::nullopt;
  }
  if (o == 0x99) {
    if (has_rex && w && r == 0 && !x && bb == 0) return Decoded{Ins{.kind = Ins::Kind::Cqo}, 2};
    return std::nullopt;
  }
  if (o == 0xd3) {
    if (!w || r != 0 || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || (*m_opt / 64) != 3) return std::nullopt;
    int m = *m_opt;
    int d = (m % 8) + bb;
    int ext = (m / 8) % 8;
    if (ext == 4) return Decoded{Ins{.kind = Ins::Kind::ShlR, .r0 = d}, p + 2};
    if (ext == 5) return Decoded{Ins{.kind = Ins::Kind::ShrR, .r0 = d}, p + 2};
    if (ext == 7) return Decoded{Ins{.kind = Ins::Kind::SarR, .r0 = d}, p + 2};
    return std::nullopt;
  }
  if (o == 0xc1) {
    if (!w || r != 0 || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    auto c_opt = b.at(p + 2);
    if (!m_opt.has_value() || !c_opt.has_value()) return std::nullopt;
    if ((*m_opt / 64) != 3 || ((*m_opt / 8) % 8) != 7) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::SarI, .r0 = (*m_opt % 8) + bb, .imm = *c_opt}, p + 3};
  }
  if (o == 0x3c) {
    if (has_rex) return std::nullopt;
    auto imm = b.at(1);
    if (!imm.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::CmpAlI, .imm = *imm}, 2};
  }
  if (o == 0xa8) {
    if (has_rex) return std::nullopt;
    auto imm = b.at(1);
    if (!imm.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::TestAlI, .imm = *imm}, 2};
  }
  if (o == 0x89) {
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || (*m_opt / 64) == 3) return std::nullopt;
    int m = *m_opt;
    int rg = ((m / 8) % 8) + r;
    if (!w) {
      if (!x) {
        auto mo = mem_op(b, p + 2, m, bb, x);
        if (mo.has_value()) {
          return Decoded{Ins{.kind = Ins::Kind::St32, .r0 = mo->base, .r1 = rg, .imm = mo->disp},
                         mo->next_off};
        }
      }
      auto moi = mem_op_idx(b, p + 2, m, bb, x);
      if (!moi.has_value()) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::StIdx,
                         .mw = MemW::W32,
                         .r0 = moi->base,
                         .r1 = moi->idx,
                         .r2 = rg,
                         .scale = moi->scale,
                         .imm = moi->disp},
                     moi->next_off};
    }
    auto moi = mem_op_idx(b, p + 2, m, bb, x);
    if (!moi.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::StIdx,
                       .mw = MemW::W64,
                       .r0 = moi->base,
                       .r1 = moi->idx,
                       .r2 = rg,
                       .scale = moi->scale,
                       .imm = moi->disp},
                   moi->next_off};
  }
  if (o == 0x8b) {
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || (*m_opt / 64) == 3) return std::nullopt;
    int m = *m_opt;
    int rg = ((m / 8) % 8) + r;
    auto moi = mem_op_idx(b, p + 2, m, bb, x);
    if (!moi.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::LdIdx,
                       .mw = w ? MemW::W64 : MemW::W32,
                       .r0 = rg,
                       .r1 = moi->base,
                       .r2 = moi->idx,
                       .scale = moi->scale,
                       .imm = moi->disp},
                   moi->next_off};
  }
  if (o == 0x88) {
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || (*m_opt / 64) == 3) return std::nullopt;
    int m = *m_opt;
    int reg = (m / 8) % 8;
    if (!has_rex && reg >= 4) return std::nullopt;
    auto moi = mem_op_idx(b, p + 2, m, bb, x);
    if (!moi.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::StIdx,
                       .mw = MemW::W8,
                       .r0 = moi->base,
                       .r1 = moi->idx,
                       .r2 = reg + r,
                       .scale = moi->scale,
                       .imm = moi->disp},
                   moi->next_off};
  }
  if (o == 0x8d) {
    if (!w) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value()) return std::nullopt;
    auto moi = mem_op_idx(b, p + 2, *m_opt, bb, x);
    if (!moi.has_value()) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::LeaIdx,
                       .r0 = ((*m_opt / 8) % 8) + r,
                       .r1 = moi->base,
                       .r2 = moi->idx,
                       .scale = moi->scale,
                       .imm = moi->disp},
                   moi->next_off};
  }
  if (o == 0x0f) {
    auto o2_opt = b.at(p + 1);
    if (!o2_opt.has_value()) return std::nullopt;
    u8 o2 = *o2_opt;
    if (o2 == 0xb7) {
      auto m_opt = b.at(p + 2);
      if (!m_opt.has_value() || (*m_opt / 64) == 3) return std::nullopt;
      int m = *m_opt;
      int rg = ((m / 8) % 8) + r;
      if (!x) {
        auto mo = mem_op(b, p + 3, m, bb, x);
        if (mo.has_value()) {
          return Decoded{Ins{.kind = Ins::Kind::Ld16, .r0 = rg, .r1 = mo->base, .imm = mo->disp},
                         mo->next_off};
        }
      }
      auto moi = mem_op_idx(b, p + 3, m, bb, x);
      if (!moi.has_value()) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::LdIdx,
                         .mw = MemW::W16,
                         .r0 = rg,
                         .r1 = moi->base,
                         .r2 = moi->idx,
                         .scale = moi->scale,
                         .imm = moi->disp},
                     moi->next_off};
    }
    if (o2 == 0xb6) {
      auto m_opt = b.at(p + 2);
      if (!m_opt.has_value() || (*m_opt / 64) == 3) return std::nullopt;
      int m = *m_opt;
      int rg = ((m / 8) % 8) + r;
      auto moi = mem_op_idx(b, p + 3, m, bb, x);
      if (!moi.has_value()) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::LdIdx,
                         .mw = MemW::W8,
                         .r0 = rg,
                         .r1 = moi->base,
                         .r2 = moi->idx,
                         .scale = moi->scale,
                         .imm = moi->disp},
                     moi->next_off};
    }
    if (o2 == 0xaf) {
      if (!w || x) return std::nullopt;
      auto m_opt = b.at(p + 2);
      if (!m_opt.has_value() || (*m_opt / 64) != 3) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::Imul,
                         .r0 = ((*m_opt / 8) % 8) + r,
                         .r1 = (*m_opt % 8) + bb},
                     p + 3};
    }
    if (o2 >= 0x90 && o2 <= 0x9f) {
      if (w || r != 0 || x) return std::nullopt;
      auto m_opt = b.at(p + 2);
      if (!m_opt.has_value() || (*m_opt / 64) != 3 || ((*m_opt / 8) % 8) != 0) return std::nullopt;
      int rm = *m_opt % 8;
      if (!has_rex && rm >= 4) return std::nullopt;
      return Decoded{
          Ins{.kind = Ins::Kind::Setcc, .r0 = rm + bb, .imm = static_cast<u64>(o2 - 0x90)},
          p + 3};
    }
    if (o2 >= 0x40 && o2 <= 0x4f) {
      if (!w || x) return std::nullopt;
      auto m_opt = b.at(p + 2);
      if (!m_opt.has_value() || (*m_opt / 64) != 3) return std::nullopt;
      return Decoded{Ins{.kind = Ins::Kind::Cmovcc,
                         .r0 = ((*m_opt / 8) % 8) + r,
                         .r1 = (*m_opt % 8) + bb,
                         .imm = static_cast<u64>(o2 - 0x40)},
                     p + 3};
    }
    return std::nullopt;
  }
  if (o == 0xff) {
    if (w || r != 0 || x) return std::nullopt;
    auto m_opt = b.at(p + 1);
    if (!m_opt.has_value() || (*m_opt / 64) != 3 || ((*m_opt / 8) % 8) != 4) return std::nullopt;
    return Decoded{Ins{.kind = Ins::Kind::JmpR, .r0 = (*m_opt % 8) + bb}, p + 2};
  }
  return std::nullopt;
}

std::optional<Decoded> decode_vm_unified(const ImageSlice& b, u64 a) {
  auto d = decode_int(b, a);
  if (d.has_value()) return d;
  return decode_vm(b);
}

// ---------------------------------------------------------------------------
// Printers (listing comments + Lean `X86Vm.Ins`)
// ---------------------------------------------------------------------------

std::string ins_str(const Ins& ins) {
  switch (ins.kind) {
    case Ins::Kind::MovI: return std::string("mov ") + kRegName[ins.r0] + ", " + hex_0x(ins.imm);
    case Ins::Kind::Mov: return std::string("mov ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::Mov32: return std::string("mov32 ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::Alu: return std::string(alu_name(ins.alu_op)) + " " + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::AluI: return std::string(alu_name(ins.alu_op)) + " " + kRegName[ins.r0] + ", " + hex_0x(ins.imm);
    case Ins::Kind::CmpI: return std::string("cmp ") + kRegName[ins.r0] + ", " + hex_0x(ins.imm);
    case Ins::Kind::Cmp: return std::string("cmp ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::Test: return std::string("test ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::TestI: return std::string("test ") + kRegName[ins.r0] + ", " + hex_0x(ins.imm);
    case Ins::Kind::ShlI: return std::string("shl ") + kRegName[ins.r0] + ", " + std::to_string(ins.imm);
    case Ins::Kind::ShrI: return std::string("shr ") + kRegName[ins.r0] + ", " + std::to_string(ins.imm);
    case Ins::Kind::SarI: return std::string("sar ") + kRegName[ins.r0] + ", " + std::to_string(ins.imm);
    case Ins::Kind::ShlR: return std::string("shl ") + kRegName[ins.r0] + ", cl";
    case Ins::Kind::ShrR: return std::string("shr ") + kRegName[ins.r0] + ", cl";
    case Ins::Kind::SarR: return std::string("sar ") + kRegName[ins.r0] + ", cl";
    case Ins::Kind::Mul: return std::string("mul ") + kRegName[ins.r0];
    case Ins::Kind::Imul: return std::string("imul ") + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::Div: return std::string("div ") + kRegName[ins.r0];
    case Ins::Kind::Idiv: return std::string("idiv ") + kRegName[ins.r0];
    case Ins::Kind::Cqo: return "cqo";
    case Ins::Kind::Neg: return std::string("neg ") + kRegName[ins.r0];
    case Ins::Kind::Not: return std::string("not ") + kRegName[ins.r0];
    case Ins::Kind::Ld: return std::string("ld ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] + signed_hex(ins.imm) + "]";
    case Ins::Kind::Ld32: return std::string("ld32 ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] + signed_hex(ins.imm) + "]";
    case Ins::Kind::Ld16: return std::string("ld16 ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] + signed_hex(ins.imm) + "]";
    case Ins::Kind::Ld8: return std::string("ld8 ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] + signed_hex(ins.imm) + "]";
    case Ins::Kind::St: return std::string("st [") + kRegName[ins.r0] + signed_hex(ins.imm) + "], " + kRegName[ins.r1];
    case Ins::Kind::St32: return std::string("st32 [") + kRegName[ins.r0] + signed_hex(ins.imm) + "], " + kRegName[ins.r1];
    case Ins::Kind::St16: return std::string("st16 [") + kRegName[ins.r0] + signed_hex(ins.imm) + "], " + kRegName[ins.r1];
    case Ins::Kind::St8: return std::string("st8 [") + kRegName[ins.r0] + signed_hex(ins.imm) + "], " + kRegName[ins.r1];
    case Ins::Kind::Lea: return std::string("lea ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] + signed_hex(ins.imm) + "]";
    case Ins::Kind::LdIdx:
      return std::string("ld_") + memw_name(ins.mw) + " " + kRegName[ins.r0] + ", [" + kRegName[ins.r1] + "+" +
             kRegName[ins.r2] + "*" + std::to_string(ins.scale) + signed_hex(ins.imm) + "]";
    case Ins::Kind::StIdx:
      return std::string("st_") + memw_name(ins.mw) + " [" + kRegName[ins.r0] + "+" + kRegName[ins.r1] + "*" +
             std::to_string(ins.scale) + signed_hex(ins.imm) + "], " + kRegName[ins.r2];
    case Ins::Kind::LeaIdx:
      return std::string("lea ") + kRegName[ins.r0] + ", [" + kRegName[ins.r1] + "+" + kRegName[ins.r2] + "*" +
             std::to_string(ins.scale) + signed_hex(ins.imm) + "]";
    case Ins::Kind::CmpAlI: return "cmp al, " + hex_0x(ins.imm);
    case Ins::Kind::TestAlI: return "test al, " + hex_0x(ins.imm);
    case Ins::Kind::Setcc: return std::string("set") + cc_name(static_cast<int>(ins.imm)) + " " + kRegName[ins.r0];
    case Ins::Kind::Cmovcc: return std::string("cmov") + cc_name(static_cast<int>(ins.imm)) + " " + kRegName[ins.r0] + ", " + kRegName[ins.r1];
    case Ins::Kind::Jmp: return "jmp " + hex_0x(ins.imm);
    case Ins::Kind::Jcc: return std::string("j") + cc_name(ins.r0) + " " + hex_0x(ins.imm);
    case Ins::Kind::Call: return "call " + hex_0x(ins.imm);
    case Ins::Kind::Ret: return "ret";
    case Ins::Kind::Push: return std::string("push ") + kRegName[ins.r0];
    case Ins::Kind::Pop: return std::string("pop ") + kRegName[ins.r0];
    case Ins::Kind::PushI: return "push " + hex_0x(ins.imm);
    case Ins::Kind::Jtab:
      return std::string("jmp [") + kRegName[ins.r0] + "+" + kRegName[ins.r1] + "*8" + signed_hex(ins.imm) + "]";
    case Ins::Kind::JmpR: return std::string("jmp ") + kRegName[ins.r0];
    case Ins::Kind::Syscall: return "syscall";
    case Ins::Kind::Voter: return "voter";
  }
  fail_msg("unknown Ins::Kind");
}

std::string lean_ins(const Ins& ins) {
  switch (ins.kind) {
    case Ins::Kind::MovI: return "(.ofInt (.movI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Mov: return "(.ofInt (.mov " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + "))";
    case Ins::Kind::Mov32: return "(.ofInt (.mov32 " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + "))";
    case Ins::Kind::Alu:
      return std::string("(.ofInt (.alu .") + alu_name(ins.alu_op) + " " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + "))";
    case Ins::Kind::AluI:
      return std::string("(.ofInt (.aluI .") + alu_name(ins.alu_op) + " " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::CmpI: return "(.ofInt (.cmpI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Cmp: return "(.ofInt (.cmp " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + "))";
    case Ins::Kind::Test: return "(.ofInt (.test " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + "))";
    case Ins::Kind::ShlI: return "(.ofInt (.shlI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::ShrI: return "(.ofInt (.shrI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Ld: return "(.ofInt (.ld " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Ld32: return "(.ofInt (.ld32 " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Ld8: return "(.ofInt (.ld8 " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::St: return "(.ofInt (.st " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + " " + std::to_string(ins.r1) + "))";
    case Ins::Kind::St8: return "(.ofInt (.st8 " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + " " + std::to_string(ins.r1) + "))";
    case Ins::Kind::Lea: return "(.ofInt (.lea " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Jmp: return "(.ofInt (.jmp " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Jcc: return "(.ofInt (.jcc " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Call: return "(.ofInt (.call " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Ret: return "(.ofInt .ret)";
    case Ins::Kind::Push: return "(.ofInt (.push " + std::to_string(ins.r0) + "))";
    case Ins::Kind::Pop: return "(.ofInt (.pop " + std::to_string(ins.r0) + "))";
    case Ins::Kind::PushI: return "(.ofInt (.pushI " + std::to_string(ins.imm) + "))";
    case Ins::Kind::Jtab: return "(.ofInt (.jtab " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " + std::to_string(ins.imm) + "))";
    case Ins::Kind::JmpR: return "(.ofInt (.jmpR " + std::to_string(ins.r0) + "))";
    case Ins::Kind::Syscall: return "(.ofInt .syscall)";
    case Ins::Kind::Mul: return "(.mul " + std::to_string(ins.r0) + ")";
    case Ins::Kind::Imul: return "(.imul " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + ")";
    case Ins::Kind::Div: return "(.div " + std::to_string(ins.r0) + ")";
    case Ins::Kind::Idiv: return "(.idiv " + std::to_string(ins.r0) + ")";
    case Ins::Kind::Cqo: return ".cqo";
    case Ins::Kind::Neg: return "(.neg " + std::to_string(ins.r0) + ")";
    case Ins::Kind::Not: return "(.not " + std::to_string(ins.r0) + ")";
    case Ins::Kind::ShlR: return "(.shlR " + std::to_string(ins.r0) + ")";
    case Ins::Kind::ShrR: return "(.shrR " + std::to_string(ins.r0) + ")";
    case Ins::Kind::SarR: return "(.sarR " + std::to_string(ins.r0) + ")";
    case Ins::Kind::SarI: return "(.sarI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + ")";
    case Ins::Kind::Ld16: return "(.ld16 " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " + std::to_string(ins.imm) + ")";
    case Ins::Kind::St16: return "(.st16 " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + " " + std::to_string(ins.r1) + ")";
    case Ins::Kind::St32: return "(.st32 " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + " " + std::to_string(ins.r1) + ")";
    case Ins::Kind::LdIdx:
      return std::string("(.ldIdx .") + memw_name(ins.mw) + " " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) +
             " " + std::to_string(ins.r2) + " " + std::to_string(ins.scale) + " " + std::to_string(ins.imm) + ")";
    case Ins::Kind::StIdx:
      return std::string("(.stIdx .") + memw_name(ins.mw) + " " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) +
             " " + std::to_string(ins.scale) + " " + std::to_string(ins.imm) + " " + std::to_string(ins.r2) + ")";
    case Ins::Kind::LeaIdx:
      return "(.leaIdx " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + " " + std::to_string(ins.r2) +
             " " + std::to_string(ins.scale) + " " + std::to_string(ins.imm) + ")";
    case Ins::Kind::TestI: return "(.testI " + std::to_string(ins.r0) + " " + std::to_string(ins.imm) + ")";
    case Ins::Kind::CmpAlI: return "(.cmpAlI " + std::to_string(ins.imm) + ")";
    case Ins::Kind::TestAlI: return "(.testAlI " + std::to_string(ins.imm) + ")";
    case Ins::Kind::Setcc: return "(.setcc " + std::to_string(ins.imm) + " " + std::to_string(ins.r0) + ")";
    case Ins::Kind::Cmovcc: return "(.cmovcc " + std::to_string(ins.imm) + " " + std::to_string(ins.r0) + " " + std::to_string(ins.r1) + ")";
    case Ins::Kind::Voter: fail_msg("Voter must be split into decoded entries before lean_ins");
  }
  fail_msg("unknown Ins::Kind in lean_ins");
}

// ---------------------------------------------------------------------------
// Assembler
// ---------------------------------------------------------------------------

struct AsmItem {
  enum class Kind { Label, Inst, Jump, Call, MovLabel, Raw };
  Kind kind = Kind::Label;
  std::string label = {};
  Encoded enc = {};
  bool is_jcc = false;
  int cc = 0;
  int reg = 0;
  std::string target_lab = {};
  std::string comment = {};
  bool far = false;
};

struct RoItem {
  enum class Kind { Label, Bytes, QwordLabel };
  Kind kind = Kind::Label;
  std::string label = {};
  std::vector<u8> bytes = {};
  std::string comment = {};
};

struct OutEntry {
  u64 addr = 0;
  std::vector<u8> bytes = {};
  Ins ins = {};
  std::string comment = {};
};

struct Asm {
  std::vector<AsmItem> items;
  std::vector<RoItem> ro_items;
  int n = 0;

  void label(std::string name) {
    items.push_back(AsmItem{.kind = AsmItem::Kind::Label, .label = std::move(name)});
  }

  std::string fresh(std::string_view stem) {
    ++n;
    return std::string(stem) + "_" + std::to_string(n);
  }

  void i(Encoded enc, std::string c = "") {
    items.push_back(AsmItem{.kind = AsmItem::Kind::Inst, .enc = std::move(enc), .comment = std::move(c)});
  }

  void jmp(std::string lab, std::string c = "", bool far = false) {
    items.push_back(AsmItem{.kind = AsmItem::Kind::Jump, .is_jcc = false, .target_lab = std::move(lab), .comment = std::move(c), .far = far});
  }

  void jcc(Cc cc, std::string lab, std::string c = "", bool far = false) {
    items.push_back(AsmItem{.kind = AsmItem::Kind::Jump, .is_jcc = true, .cc = static_cast<int>(cc), .target_lab = std::move(lab), .comment = std::move(c), .far = far});
  }

  void call(std::string lab, std::string c = "") {
    items.push_back(AsmItem{.kind = AsmItem::Kind::Call, .target_lab = std::move(lab), .comment = std::move(c)});
  }

  void mov_label(int d, std::string lab, std::string c = "") {
    items.push_back(AsmItem{.kind = AsmItem::Kind::MovLabel, .reg = d, .target_lab = std::move(lab), .comment = std::move(c)});
  }

  void raw_voter(std::vector<u8> bs, std::string c = "") {
    items.push_back(AsmItem{.kind = AsmItem::Kind::Raw, .enc = Encoded{std::move(bs), Ins{.kind = Ins::Kind::Voter}}, .comment = std::move(c)});
  }

  void ro_label(std::string name) {
    ro_items.push_back(RoItem{.kind = RoItem::Kind::Label, .label = std::move(name)});
  }

  void ro_bytes(std::vector<u8> bs, std::string c = "") {
    ro_items.push_back(RoItem{.kind = RoItem::Kind::Bytes, .bytes = std::move(bs), .comment = std::move(c)});
  }

  void ro_cstr(std::string_view s, std::string c = "") {
    std::vector<u8> bs(s.begin(), s.end());
    bs.push_back(0);
    ro_bytes(std::move(bs), std::move(c));
  }

  void ro_str(std::string_view s, std::string c = "") {
    std::vector<u8> bs(s.begin(), s.end());
    ro_bytes(std::move(bs), std::move(c));
  }

  void ro_qword_label(std::string target, std::string c = "") {
    ro_items.push_back(RoItem{.kind = RoItem::Kind::QwordLabel, .label = std::move(target), .comment = std::move(c)});
  }

  struct Result {
    std::vector<OutEntry> out;
    std::vector<u8> rodata;
    std::vector<std::pair<u64, std::vector<u8>>> ro_chunks;
    std::map<u64, std::vector<std::string>> ro_labels;
    std::map<std::string, u64> labels;
    std::vector<std::pair<std::string, u64>> label_order;
    u64 code_end = 0;
  };

  Result assemble(u64 org) {
    std::set<std::size_t> long_jumps;
    for (std::size_t idx = 0; idx < items.size(); ++idx) {
      if (items[idx].kind == AsmItem::Kind::Jump && items[idx].far) long_jumps.insert(idx);
    }
    while (true) {
      u64 addr = org;
      std::map<std::string, u64> labels;
      std::vector<std::pair<std::string, u64>> label_order;
      std::vector<u64> cur_addr(items.size() + 1, 0);

      for (std::size_t idx = 0; idx < items.size(); ++idx) {
        cur_addr[idx] = addr;
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
          case AsmItem::Kind::Call:
            addr += 5;
            break;
          case AsmItem::Kind::MovLabel:
            addr += (it.reg >= 8) ? 6 : 5;
            break;
          case AsmItem::Kind::Jump:
            if (long_jumps.contains(idx)) {
              if (!it.is_jcc) addr += 5;
              else if (it.cc == static_cast<int>(Cc::NE)) addr += 7;
              else addr += 6;
            } else {
              addr += 2;
            }
            break;
        }
      }
      cur_addr[items.size()] = addr;
      u64 code_end = addr;
      u64 ro_addr = code_end;
      for (const auto& ri : ro_items) {
        switch (ri.kind) {
          case RoItem::Kind::Label:
            require(labels.find(ri.label) == labels.end(), "duplicate ro label: " + ri.label);
            labels.emplace(ri.label, ro_addr);
            label_order.emplace_back(ri.label, ro_addr);
            break;
          case RoItem::Kind::Bytes:
            ro_addr += ri.bytes.size();
            break;
          case RoItem::Kind::QwordLabel:
            ro_addr += 8;
            break;
        }
      }

      bool grew = false;
      std::vector<OutEntry> out;
      for (std::size_t idx = 0; idx < items.size(); ++idx) {
        u64 a = cur_addr[idx];
        const auto& it = items[idx];
        if (it.kind == AsmItem::Kind::Label) continue;
        if (it.kind == AsmItem::Kind::Inst || it.kind == AsmItem::Kind::Raw) {
          out.push_back(OutEntry{a, it.enc.bytes, it.enc.ins, it.comment});
          continue;
        }
        auto lit = labels.find(it.target_lab);
        require(lit != labels.end(), "missing label: " + it.target_lab);
        u64 t = lit->second;
        if (it.kind == AsmItem::Kind::MovLabel) {
          require(t < (1ULL << 32), "label address >= 2^32: " + it.target_lab);
          Encoded enc = e_movI(it.reg, t);
          out.push_back(OutEntry{a, std::move(enc.bytes), enc.ins,
                                 (it.comment.empty() ? "" : (it.comment + " ")) + "(" + it.target_lab + ")"});
          continue;
        }
        if (it.kind == AsmItem::Kind::Call) {
          i64 rel = static_cast<i64>(t) - static_cast<i64>(a + 5);
          std::vector<u8> bs = {0xe8};
          push_le(bs, static_cast<u64>(static_cast<u32>(static_cast<i32>(rel))), 4);
          out.push_back(OutEntry{a, std::move(bs), Ins{.kind = Ins::Kind::Call, .imm = t},
                                 it.comment + " -> " + it.target_lab});
          continue;
        }
        if (!long_jumps.contains(idx)) {
          i64 rel = static_cast<i64>(t) - static_cast<i64>(a + 2);
          if (rel >= -128 && rel < 128) {
            u8 rb = static_cast<u8>(rel & 0xff);
            if (!it.is_jcc) {
              out.push_back(OutEntry{a, {0xeb, rb}, Ins{.kind = Ins::Kind::Jmp, .imm = t},
                                     it.comment + " -> " + it.target_lab});
            } else {
              out.push_back(OutEntry{a, {static_cast<u8>(0x70 + it.cc), rb},
                                     Ins{.kind = Ins::Kind::Jcc, .r0 = it.cc, .imm = t},
                                     it.comment + " -> " + it.target_lab});
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
          out.push_back(OutEntry{a, std::move(bs), Ins{.kind = Ins::Kind::Jmp, .imm = t},
                                 it.comment + " -> " + it.target_lab});
        } else if (it.cc == static_cast<int>(Cc::NE)) {
          out.push_back(OutEntry{a, {0x74, 0x05},
                                 Ins{.kind = Ins::Kind::Jcc, .r0 = static_cast<int>(Cc::E), .imm = a + 7},
                                 it.comment + " (jne: je over jmp)"});
          i64 rel = static_cast<i64>(t) - static_cast<i64>(a + 7);
          std::vector<u8> bs = {0xe9};
          push_le(bs, static_cast<u64>(static_cast<u32>(static_cast<i32>(rel))), 4);
          out.push_back(OutEntry{a + 2, std::move(bs), Ins{.kind = Ins::Kind::Jmp, .imm = t},
                                 "-> " + it.target_lab});
        } else {
          i64 rel = static_cast<i64>(t) - static_cast<i64>(a + 6);
          std::vector<u8> bs = {0x0f, static_cast<u8>(0x80 + it.cc)};
          push_le(bs, static_cast<u64>(static_cast<u32>(static_cast<i32>(rel))), 4);
          out.push_back(OutEntry{a, std::move(bs), Ins{.kind = Ins::Kind::Jcc, .r0 = it.cc, .imm = t},
                                 it.comment + " -> " + it.target_lab});
        }
      }
      if (!grew) {
        std::vector<u8> rodata;
        std::vector<std::pair<u64, std::vector<u8>>> ro_chunks;
        std::map<u64, std::vector<std::string>> ro_labels;
        u64 ra = code_end;
        for (const auto& ri : ro_items) {
          if (ri.kind == RoItem::Kind::Label) {
            ro_labels[ra].push_back(ri.label);
          } else if (ri.kind == RoItem::Kind::Bytes) {
            rodata.insert(rodata.end(), ri.bytes.begin(), ri.bytes.end());
            ro_chunks.emplace_back(ra, ri.bytes);
            ra += ri.bytes.size();
          } else {
            u64 t = labels.at(ri.label);
            std::vector<u8> q;
            push_le(q, t, 8);
            rodata.insert(rodata.end(), q.begin(), q.end());
            ro_chunks.emplace_back(ra, std::move(q));
            ra += 8;
          }
        }
        return Result{std::move(out), std::move(rodata), std::move(ro_chunks),
                      std::move(ro_labels), std::move(labels), std::move(label_order), code_end};
      }
    }
  }
};

// ---------------------------------------------------------------------------
// VM Instruction Emitter Helpers
// ---------------------------------------------------------------------------

void emit_vote3_inline(Asm& A, std::string_view tag) {
  std::string ok = A.fresh(std::string("v3_") + std::string(tag));
  A.i(e_cmp(RAX, RBX), "vote3: a == b?");
  A.jcc(Cc::E, ok);
  A.i(e_cmp(RAX, RCX), "vote3: a == c?");
  A.jcc(Cc::E, ok);
  A.i(e_cmp(RBX, RCX), "vote3: b == c?");
  A.jcc(Cc::NE, "L_exit3", "3-way disagreement -> exit 3", true);
  A.i(e_mov(RAX, RBX), "repair rax = b");
  A.label(ok);
}

void emit_vote_code_off(Asm& A, int off_reg, i32 disp, std::string_view tag) {
  A.i(e_ld(RDX, R15, OFF_CODE1), "code1");
  A.i(e_ld_idx(MemW::W64, RAX, RDX, off_reg, 1, disp), "replica 1");
  A.i(e_ld(RDX, R15, OFF_CODE2), "code2");
  A.i(e_ld_idx(MemW::W64, RBX, RDX, off_reg, 1, disp), "replica 2");
  A.i(e_ld(RDX, R15, OFF_CODE3), "code3");
  A.i(e_ld_idx(MemW::W64, RCX, RDX, off_reg, 1, disp), "replica 3");
  emit_vote3_inline(A, tag);
}

void emit_vote_code_const(Asm& A, i32 disp, std::string_view tag) {
  A.i(e_ld(RDX, R15, OFF_CODE1));
  A.i(e_ld(RAX, RDX, disp));
  A.i(e_ld(RDX, R15, OFF_CODE2));
  A.i(e_ld(RBX, RDX, disp));
  A.i(e_ld(RDX, R15, OFF_CODE3));
  A.i(e_ld(RCX, RDX, disp));
  emit_vote3_inline(A, tag);
}

// Bounds check: require addr <= msize && len <= msize - addr, else jump L_fail_bounds.
// Clobbers r10, r11. addr_reg and len_reg must not be r10 or r11.
void emit_check_inb(Asm& A, int addr_reg, int len_reg) {
  A.i(e_ld(R10, R15, OFF_MSIZE), "msize");
  A.i(e_cmp(addr_reg, R10), "addr <= msize?");
  A.jcc(Cc::A, "L_fail_bounds", "addr > msize", true);
  A.i(e_mov(R11, R10));
  A.i(e_alu(AluOp::Sub, R11, addr_reg), "msize - addr");
  A.i(e_cmp(len_reg, R11), "len <= msize - addr?");
  A.jcc(Cc::A, "L_fail_bounds", "len > msize - addr", true);
}

// Advance top frame pc by 1 (clobbers rax, rcx).
void emit_adv_pc(Asm& A) {
  A.i(e_ld(RAX, R15, OFF_RS_BASE));
  A.i(e_ld(RCX, R15, OFF_RS_LEN));
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.i(e_shlI(RCX, 5), "* 32");
  A.i(e_alu(AluOp::Add, RAX, RCX));
  A.i(e_ld(RCX, RAX, 8), "f.pc");
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.i(e_st(RAX, 8, RCX), "f.pc + 1");
}

// Emit fail jump with ErrId.
void emit_fail_jump(Asm& A, ErrId e) {
  A.i(e_movI(RAX, static_cast<u64>(e)));
  A.i(e_st(R15, OFF_ERR_ID, RAX));
  A.jmp("L_vm_fail", "", true);
}

// Emit common runtime syscall helpers.
void emit_helpers(Asm& A) {
  // fn_mmap: rdi = size_bytes (> 0) -> rax = addr (or exits 2 on -ENOMEM)
  A.label("fn_mmap");
  A.i(e_mov(RSI, RDI), "len");
  A.i(e_zero(RDI), "addr = 0");
  A.i(e_movI(RDX, 3), "PROT_READ|PROT_WRITE");
  A.i(e_movI(R10, 0x22), "MAP_PRIVATE|MAP_ANONYMOUS");
  A.i(e_movI(R8, ~0ULL), "fd = -1");
  A.i(e_zero(R9), "off = 0");
  A.i(e_movI(RAX, 9), "sys_mmap");
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::S, "L_exit_nomem", "-ENOMEM", true);
  A.i(e_ret());

  // fn_mremap: rdi = old_addr, rsi = old_bytes, rdx = new_bytes -> rax = new_addr
  A.label("fn_mremap");
  A.i(e_movI(R10, 1), "MREMAP_MAYMOVE");
  A.i(e_movI(R8, 0));
  A.i(e_zero(R9));
  A.i(e_movI(RAX, 25), "sys_mremap");
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::S, "L_exit_nomem", "-ENOMEM", true);
  A.i(e_ret());

  // fn_write_stderr: rsi = buf, rdx = len (ignores return value)
  A.label("fn_write_stderr");
  A.i(e_movI(RDI, 2), "fd 2 = stderr");
  A.i(e_movI(RAX, 1), "sys_write");
  A.i(e_syscall());
  A.i(e_ret());

  // fn_write_stderr_cstr: rdi = NUL-terminated string
  A.label("fn_write_stderr_cstr");
  A.i(e_mov(RSI, RDI));
  A.i(e_zero(RDX));
  A.label("wsc_loop");
  A.i(e_ld_idx(MemW::W8, RAX, RSI, RDX, 1, 0));
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::E, "wsc_done");
  A.i(e_aluI(AluOp::Add, RDX, 1));
  A.jmp("wsc_loop");
  A.label("wsc_done");
  A.i(e_test(RDX, RDX));
  A.jcc(Cc::E, "wsc_ret");
  A.i(e_movI(RDI, 2));
  A.i(e_movI(RAX, 1));
  A.i(e_syscall());
  A.label("wsc_ret");
  A.i(e_ret());

  // fn_ds_push: rdi = w, rsi = plen. Clobbers rax..r11, r12, r13.
  A.label("fn_ds_push");
  A.i(e_mov(R12, RDI), "save w");
  A.i(e_mov(R13, RSI), "save plen");
  A.i(e_ld(RAX, R15, OFF_DS_LEN));
  A.i(e_ld(RCX, R15, OFF_DS_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::B, "dsp_room");
  A.i(e_ld(RDI, R15, OFF_DS_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_shlI(RSI, 4), "old_bytes = cap * 16");
  A.i(e_mov(RDX, RSI));
  A.i(e_shlI(RDX, 1), "new_bytes = 2 * old_bytes");
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_DS_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_DS_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_DS_CAP, RCX));
  A.label("dsp_room");
  A.i(e_ld(RAX, R15, OFF_DS_BASE));
  A.i(e_ld(RCX, R15, OFF_DS_LEN));
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 4), "len * 16");
  A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 0, R12), "ds[len].w");
  A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 8, R13), "ds[len].plen");
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.i(e_st(R15, OFF_DS_LEN, RCX));
  A.i(e_ret());

  // fn_aux_push: rdi = w, rsi = plen. Clobbers rax..r11, r12, r13.
  A.label("fn_aux_push");
  A.i(e_mov(R12, RDI));
  A.i(e_mov(R13, RSI));
  A.i(e_ld(RAX, R15, OFF_AUX_LEN));
  A.i(e_ld(RCX, R15, OFF_AUX_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::B, "axp_room");
  A.i(e_ld(RDI, R15, OFF_AUX_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_shlI(RSI, 4));
  A.i(e_mov(RDX, RSI));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_AUX_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_AUX_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_AUX_CAP, RCX));
  A.label("axp_room");
  A.i(e_ld(RAX, R15, OFF_AUX_BASE));
  A.i(e_ld(RCX, R15, OFF_AUX_LEN));
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 4));
  A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 0, R12));
  A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 8, R13));
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.i(e_st(R15, OFF_AUX_LEN, RCX));
  A.i(e_ret());

  // fn_out_push_byte: rdi = byte. Clobbers rax..r11, r12.
  A.label("fn_out_push_byte");
  A.i(e_mov(R12, RDI));
  A.i(e_ld(RAX, R15, OFF_OUT_LEN));
  A.i(e_ld(RCX, R15, OFF_OUT_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::B, "ob_room");
  A.i(e_ld(RDI, R15, OFF_OUT_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_OUT_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_OUT_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_OUT_CAP, RCX));
  A.label("ob_room");
  A.i(e_ld(RAX, R15, OFF_OUT_BASE));
  A.i(e_ld(RCX, R15, OFF_OUT_LEN));
  A.i(e_st_idx(MemW::W8, RAX, RCX, 1, 0, R12));
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.i(e_st(R15, OFF_OUT_LEN, RCX));
  A.i(e_ret());

  // fn_out_push_u64: rdi = u64. Clobbers rax..r11, r12.
  A.label("fn_out_push_u64");
  A.i(e_mov(R12, RDI));
  A.label("ou_chk");
  A.i(e_ld(RAX, R15, OFF_OUT_LEN));
  A.i(e_aluI(AluOp::Add, RAX, 8));
  A.i(e_ld(RCX, R15, OFF_OUT_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::BE, "ou_room");
  A.i(e_ld(RDI, R15, OFF_OUT_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_OUT_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_OUT_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_OUT_CAP, RCX));
  A.jmp("ou_chk");
  A.label("ou_room");
  A.i(e_ld(RAX, R15, OFF_OUT_BASE));
  A.i(e_ld(RCX, R15, OFF_OUT_LEN));
  A.i(e_st_idx(MemW::W64, RAX, RCX, 1, 0, R12));
  A.i(e_aluI(AluOp::Add, RCX, 8));
  A.i(e_st(R15, OFF_OUT_LEN, RCX));
  A.i(e_ret());
}

// ---------------------------------------------------------------------------
// VM Execution Loop, Voter, Dispatch, Primitives, and Ops
// ---------------------------------------------------------------------------

constexpr std::string_view kUsageStr = "usage: sporos_vm PROGRAM.spb [MEMORY-BYTES] < input > output\n";
constexpr std::string_view kProgPreStr = "sporos_vm: ";
constexpr std::string_view kOpenSufStr = ": cannot open\n";
constexpr std::string_view kDecSufStr = ": not a Sporos bytecode file\n";
constexpr std::string_view kChkSufStr = ": refused by the checker\n";
constexpr std::string_view kNomemStr = "sporos_vm: out of memory\n";
constexpr std::string_view kRunFailPreStr = "sporos: run failed: ";
constexpr std::string_view kNlStr = "\n";

void emit_vm_core(Asm& A) {
  A.label("L_vm_ready");
  A.label("L_vm_loop");

  // 1. Check rs_len > 0 else noFrame
  A.i(e_ld(RAX, R15, OFF_RS_LEN), "rs_len");
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::E, "L_fail_noFrame", "", true);

  // 2. Load top frame f = rs[rs_len - 1]
  A.i(e_aluI(AluOp::Sub, RAX, 1));
  A.i(e_shlI(RAX, 5), "(rs_len - 1) * 32");
  A.i(e_ld(RDX, R15, OFF_RS_BASE));
  A.i(e_alu(AluOp::Add, RAX, RDX), "top frame ptr");
  A.i(e_ld(R12, RAX, 0), "r12 = f.word");
  A.i(e_ld(R13, RAX, 8), "r13 = f.pc");

  // 3. Vote nw from code1/2/3[+8]; check f.word < nw
  emit_vote_code_const(A, 8, "nw");
  A.i(e_cmp(R12, RAX), "f.word < nw?");
  A.jcc(Cc::AE, "L_fail_badJump", "", true);

  // 4. Vote nd from code1/2/3[+16]; compute w_off = 32 + 16*nd + 64*f.word
  emit_vote_code_const(A, 16, "nd");
  A.i(e_shlI(RAX, 4), "16 * nd");
  A.i(e_mov(RSI, R12));
  A.i(e_shlI(RSI, 6), "64 * f.word");
  A.i(e_alu(AluOp::Add, RSI, RAX));
  A.i(e_aluI(AluOp::Add, RSI, 32), "rsi = w_off");

  // 5. Vote ncode from words_meta[w] (+24); check f.pc < ncode
  emit_vote_code_off(A, RSI, 24, "ncode");
  A.i(e_cmp(R13, RAX), "f.pc < ncode?");
  A.jcc(Cc::AE, "L_fail_badJump", "", true);

  // 6. Vote code_off from words_meta[w] (+32); compute inst_off = code_off + 16 * f.pc
  emit_vote_code_off(A, RSI, 32, "code_off");
  A.i(e_mov(RSI, R13));
  A.i(e_shlI(RSI, 4), "16 * f.pc");
  A.i(e_alu(AluOp::Add, RSI, RAX), "rsi = inst_off");

  // 7. Vote imm from instr[+8]; store in OFF_CUR_IMM
  emit_vote_code_off(A, RSI, 8, "imm");
  A.i(e_st(R15, OFF_CUR_IMM, RAX), "cur_imm");

  // 8. Load op1, op2, op3 into (rax, rbx, rcx) for canonical 38-byte voter
  A.i(e_ld(RDX, R15, OFF_CODE1));
  A.i(e_ld_idx(MemW::W64, RAX, RDX, RSI, 1, 0), "op replica 1");
  A.i(e_ld(RDX, R15, OFF_CODE2));
  A.i(e_ld_idx(MemW::W64, RBX, RDX, RSI, 1, 0), "op replica 2");
  A.i(e_ld(RDX, R15, OFF_CODE3));
  A.i(e_ld_idx(MemW::W64, RCX, RDX, RSI, 1, 0), "op replica 3");

  // 9. Canonical 38-byte TMR voter (`Bootstrap/TMRVoter.lean` pattern, park -> jmp +0x0c to L_vm_dispatch)
  A.label("L_vm_vote");
  A.raw_voter({
                  0x48, 0x39, 0xd8,  // cmp rax, rbx
                  0x74, 0x0d,        // je  +0x0d
                  0x48, 0x39, 0xc8,  // cmp rax, rcx
                  0x74, 0x08,        // je  +0x08
                  0x48, 0x39, 0xcb,  // cmp rbx, rcx
                  0x75, 0x0b,        // jne +0x0b (L_exit3)
                  0x48, 0x89, 0xd8,  // mov rax, rbx
                  0x48, 0x89, 0xc3,  // mov rbx, rax
                  0x48, 0x89, 0xc1,  // mov rcx, rax
                  0xeb, 0x0c,        // jmp +0x0c (L_vm_dispatch)
              },
              "TMRVoter compare/repair + jmp L_vm_dispatch");
  A.label("L_exit3");
  A.raw_voter({
                  0xbf, 0x03, 0x00, 0x00, 0x00,  // mov edi, 3
                  0xb8, 0x3c, 0x00, 0x00, 0x00,  // mov eax, 60
                  0x0f, 0x05,                    // syscall
              },
              "TMRVoter 3-way disagreement -> exit(3)");

  // 10. Dispatch (at exact L_vm_vote + 0x26)
  A.label("L_vm_dispatch");
  A.i(e_st(R15, OFF_CUR_OP, RAX), "cur_op");
  A.i(e_cmpI(RAX, kNOps));
  A.jcc(Cc::AE, "L_fail_badJump", "op out of range", true);
  A.mov_label(RDX, "L_jtab", "dispatch table");
  A.i(e_jtab(RDX, RAX, 0), "jmp [L_jtab + op*8]");

  // -------------------------------------------------------------------------
  // Shared Op.prim prologue & epilogue
  // -------------------------------------------------------------------------
  A.label("L_prim_prologue");
  A.mov_label(RDX, "L_prim_eff_in");
  A.i(e_ld_idx(MemW::W8, R8, RDX, RAX, 1, 0), "r8 = e_in");
  A.mov_label(RDX, "L_prim_eff_out");
  A.i(e_ld_idx(MemW::W8, R9, RDX, RAX, 1, 0), "r9 = e_out");
  A.i(e_st(R15, OFF_CUR_EIN, R8));
  A.i(e_st(R15, OFF_CUR_EOUT, R9));
  A.i(e_ld(RCX, R15, OFF_DS_LEN), "ds_len");
  A.i(e_cmp(RCX, R8), "ds_len < e_in?");
  A.jcc(Cc::B, "L_fail_underflow", "", true);

  // Load up to 4 operands (a0..a3) and accumulate qa
  A.i(e_zero(R10), "qa = 0");
  A.i(e_ld(RSI, R15, OFF_DS_BASE), "ds_base");
  A.i(e_test(R8, R8));
  A.jcc(Cc::E, "pp_q");

  auto load_arg = [&](int k, i32 off_w, i32 off_p, const std::string& next_lab) {
    A.i(e_mov(RDX, RCX));
    A.i(e_aluI(AluOp::Sub, RDX, k + 1), "ds_len - (k+1)");
    A.i(e_shlI(RDX, 4), "* 16");
    A.i(e_ld_idx(MemW::W64, R11, RSI, RDX, 1, 0), "ak.w");
    A.i(e_st(R15, off_w, R11));
    A.i(e_ld_idx(MemW::W64, R11, RSI, RDX, 1, 8), "ak.p");
    A.i(e_st(R15, off_p, R11));
    A.i(e_alu(AluOp::Add, R10, R11), "qa += ak.p");
    if (!next_lab.empty()) {
      A.i(e_cmpI(R8, k + 1));
      A.jcc(Cc::E, next_lab);
    }
  };
  load_arg(0, OFF_A0_W, OFF_A0_P, "pp_q");
  load_arg(1, OFF_A1_W, OFF_A1_P, "pp_q");
  load_arg(2, OFF_A2_W, OFF_A2_P, "pp_q");
  load_arg(3, OFF_A3_W, OFF_A3_P, "");

  A.label("pp_q");
  A.i(e_ld(R11, R15, OFF_PC_PROV));
  A.i(e_alu(AluOp::Add, R10, R11), "qa + pc");
  A.i(e_aluI(AluOp::Add, R10, 1), "q = qa + pc + 1");
  A.i(e_st(R15, OFF_CUR_Q, R10), "cur_q");
  A.mov_label(RDX, "L_prim_jtab");
  A.i(e_jtab(RDX, RAX, 0), "jmp [L_prim_jtab + prim*8]");

  // Epilogue variants
  A.label("L_prim_epilogue_mk1");
  A.i(e_st(R15, OFF_RES0_W, RAX));
  A.i(e_ld(RDX, R15, OFF_CUR_Q));
  A.i(e_st(R15, OFF_RES0_P, RDX));
  A.jmp("L_prim_epilogue");

  A.label("L_prim_epilogue_0");
  A.label("L_prim_epilogue");
  A.i(e_ld(RAX, R15, OFF_DS_LEN));
  A.i(e_ld(RCX, R15, OFF_CUR_EIN));
  A.i(e_alu(AluOp::Sub, RAX, RCX), "pop e_in");
  A.i(e_st(R15, OFF_DS_LEN, RAX));
  A.i(e_ld(RBX, R15, OFF_CUR_EOUT), "rbx = e_out (0..3)");
  A.i(e_test(RBX, RBX));
  A.jcc(Cc::E, "pe_adv");
  // Ensure ds_len + e_out <= ds_cap
  A.i(e_mov(R12, RAX));
  A.i(e_alu(AluOp::Add, R12, RBX));
  A.i(e_ld(RCX, R15, OFF_DS_CAP));
  A.i(e_cmp(R12, RCX));
  A.jcc(Cc::BE, "pe_push");
  A.i(e_ld(RDI, R15, OFF_DS_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_shlI(RSI, 4));
  A.i(e_mov(RDX, RSI));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_DS_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_DS_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_DS_CAP, RCX));
  A.label("pe_push");
  // Push res[e_out - 1 .. 0]
  A.i(e_ld(R12, R15, OFF_DS_BASE));
  A.i(e_ld(R13, R15, OFF_DS_LEN));
  A.i(e_ld(RBX, R15, OFF_CUR_EOUT));
  A.label("pe_loop");
  A.i(e_aluI(AluOp::Sub, RBX, 1));
  A.i(e_mov(RDX, RBX));
  A.i(e_shlI(RDX, 4), "res_k offset");
  A.i(e_ld_idx(MemW::W64, RAX, R15, RDX, 1, OFF_RES0_W), "res_k.w");
  A.i(e_ld_idx(MemW::W64, RCX, R15, RDX, 1, OFF_RES0_P), "res_k.p");
  A.i(e_mov(RSI, R13));
  A.i(e_shlI(RSI, 4));
  A.i(e_st_idx(MemW::W64, R12, RSI, 1, 0, RAX));
  A.i(e_st_idx(MemW::W64, R12, RSI, 1, 8, RCX));
  A.i(e_aluI(AluOp::Add, R13, 1));
  A.i(e_test(RBX, RBX));
  A.jcc(Cc::NE, "pe_loop");
  A.i(e_st(R15, OFF_DS_LEN, R13));
  A.label("pe_adv");
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // -------------------------------------------------------------------------
  // 57 Primitive Handlers
  // -------------------------------------------------------------------------
  auto tag_val = [&](i32 off_w, i32 off_p, i32 res_w, i32 res_p) {
    A.i(e_ld(RAX, R15, off_w));
    A.i(e_st(R15, res_w, RAX));
    A.i(e_ld(RAX, R15, off_p));
    A.i(e_ld(RCX, R15, OFF_PC_PROV));
    A.i(e_alu(AluOp::Add, RAX, RCX));
    A.i(e_aluI(AluOp::Add, RAX, 1));
    A.i(e_st(R15, res_p, RAX));
  };

  // 0: dup ( a -- a a )
  A.label("P_dup");
  tag_val(OFF_A0_W, OFF_A0_P, OFF_RES0_W, OFF_RES0_P);
  A.i(e_ld(RAX, R15, OFF_RES0_W));
  A.i(e_st(R15, OFF_RES1_W, RAX));
  A.i(e_ld(RAX, R15, OFF_RES0_P));
  A.i(e_st(R15, OFF_RES1_P, RAX));
  A.jmp("L_prim_epilogue", "", true);

  // 1: drop
  A.label("P_drop");
  A.jmp("L_prim_epilogue_0", "", true);

  // 2: swap ( a b -- b a ) => res = [t(a1), t(a0)]
  A.label("P_swap");
  tag_val(OFF_A1_W, OFF_A1_P, OFF_RES0_W, OFF_RES0_P);
  tag_val(OFF_A0_W, OFF_A0_P, OFF_RES1_W, OFF_RES1_P);
  A.jmp("L_prim_epilogue", "", true);

  // 3: over ( a b -- a b a ) => res = [t(a1), t(a0), t(a1)]
  A.label("P_over");
  tag_val(OFF_A1_W, OFF_A1_P, OFF_RES0_W, OFF_RES0_P);
  tag_val(OFF_A0_W, OFF_A0_P, OFF_RES1_W, OFF_RES1_P);
  A.i(e_ld(RAX, R15, OFF_RES0_W));
  A.i(e_st(R15, OFF_RES2_W, RAX));
  A.i(e_ld(RAX, R15, OFF_RES0_P));
  A.i(e_st(R15, OFF_RES2_P, RAX));
  A.jmp("L_prim_epilogue", "", true);

  // 4: rot ( a b c -- b c a ) => a0=c,a1=b,a2=a -> res = [t(a2), t(a0), t(a1)]
  A.label("P_rot");
  tag_val(OFF_A2_W, OFF_A2_P, OFF_RES0_W, OFF_RES0_P);
  tag_val(OFF_A0_W, OFF_A0_P, OFF_RES1_W, OFF_RES1_P);
  tag_val(OFF_A1_W, OFF_A1_P, OFF_RES2_W, OFF_RES2_P);
  A.jmp("L_prim_epilogue", "", true);

  // 5: nip ( a b -- b ) => res = [t(a0)]
  A.label("P_nip");
  tag_val(OFF_A0_W, OFF_A0_P, OFF_RES0_W, OFF_RES0_P);
  A.jmp("L_prim_epilogue", "", true);

  // 6: tuck ( a b -- b a b ) => res = [t(a0), t(a1), t(a0)]
  A.label("P_tuck");
  tag_val(OFF_A0_W, OFF_A0_P, OFF_RES0_W, OFF_RES0_P);
  tag_val(OFF_A1_W, OFF_A1_P, OFF_RES1_W, OFF_RES1_P);
  A.i(e_ld(RAX, R15, OFF_RES0_W));
  A.i(e_st(R15, OFF_RES2_W, RAX));
  A.i(e_ld(RAX, R15, OFF_RES0_P));
  A.i(e_st(R15, OFF_RES2_P, RAX));
  A.jmp("L_prim_epilogue", "", true);

  // 7..9: add, sub, mul
  A.label("P_add");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_alu(AluOp::Add, RAX, RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_sub");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_alu(AluOp::Sub, RAX, RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_mul");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_imul(RAX, RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  // 10..11: divu, modu
  A.label("P_divu");
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "L_fail_divZero", "", true);
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_zero(RDX));
  A.i(e_div(RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_modu");
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "L_fail_divZero", "", true);
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_zero(RDX));
  A.i(e_div(RCX));
  A.i(e_mov(RAX, RDX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  // 12..13: divs, mods (with explicit INT64_MIN / -1 and % -1 guards against #DE)
  A.label("P_divs");
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "L_fail_divZero", "", true);
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_movI(RDX, 0x8000000000000000ULL));
  A.i(e_cmp(RAX, RDX));
  A.jcc(Cc::NE, "divs_go");
  A.i(e_cmpI(RCX, -1));
  A.jcc(Cc::E, "L_prim_epilogue_mk1", "INT64_MIN / -1 wraps to INT64_MIN", true);
  A.label("divs_go");
  A.i(e_cqo());
  A.i(e_idiv(RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_mods");
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "L_fail_divZero", "", true);
  A.i(e_cmpI(RCX, -1));
  A.jcc(Cc::NE, "mods_go");
  A.i(e_zero(RAX), "x % -1 == 0 without #DE");
  A.jmp("L_prim_epilogue_mk1", "", true);
  A.label("mods_go");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_cqo());
  A.i(e_idiv(RCX));
  A.i(e_mov(RAX, RDX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  // 14: neg
  A.label("P_neg");
  A.i(e_ld(RAX, R15, OFF_A0_W));
  A.i(e_neg(RAX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  // 15..18: band, bor, bxor, bnot
  A.label("P_band");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_alu(AluOp::And, RAX, RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_bor");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_alu(AluOp::Or, RAX, RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_bxor");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_alu(AluOp::Xor, RAX, RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_bnot");
  A.i(e_ld(RAX, R15, OFF_A0_W));
  A.i(e_not(RAX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  // 19..21: shl, shr, sar (hardware masks cl % 64)
  A.label("P_shl");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_shlR(RAX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_shr");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_shrR(RAX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_sar");
  A.i(e_ld(RAX, R15, OFF_A1_W));
  A.i(e_ld(RCX, R15, OFF_A0_W));
  A.i(e_sarR(RAX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  // 22..27: eq, ne, ltu, lts, leu, les
  auto emit_cmp_prim = [&](const char* name, Cc cc) {
    A.label(name);
    A.i(e_zero(RDX));
    A.i(e_ld(RAX, R15, OFF_A1_W));
    A.i(e_ld(RCX, R15, OFF_A0_W));
    A.i(e_cmp(RAX, RCX));
    A.i(e_setcc(cc, RDX));
    A.i(e_mov(RAX, RDX));
    A.jmp("L_prim_epilogue_mk1", "", true);
  };
  emit_cmp_prim("P_eq", Cc::E);
  emit_cmp_prim("P_ne", Cc::NE);
  emit_cmp_prim("P_ltu", Cc::B);
  emit_cmp_prim("P_lts", Cc::L);
  emit_cmp_prim("P_leu", Cc::BE);
  emit_cmp_prim("P_les", Cc::LE);

  // 28,30,32,34: ld8, ld16, ld32, ld64
  auto emit_ld_prim = [&](const char* name, MemW mw, int k) {
    A.label(name);
    A.i(e_ld(RAX, R15, OFF_A0_W), "addr");
    A.i(e_movI(RCX, static_cast<u64>(k)));
    emit_check_inb(A, RAX, RCX);
    A.i(e_ld(R8, R15, OFF_MEMW_BASE));
    A.i(e_ld_idx(mw, RDX, R8, RAX, 1, 0), "val");
    A.i(e_st(R15, OFF_RES0_W, RDX));
    A.i(e_ld(R9, R15, OFF_MEMP_BASE));
    A.i(e_ld(R10, R15, OFF_CUR_Q), "pm = cur_q");
    for (int i = 0; i < k; ++i) {
      A.i(e_ld_idx(MemW::W64, R11, R9, RAX, 8, i * 8));
      A.i(e_alu(AluOp::Add, R10, R11));
    }
    A.i(e_st(R15, OFF_RES0_P, R10));
    A.jmp("L_prim_epilogue", "", true);
  };
  emit_ld_prim("P_ld8", MemW::W8, 1);
  emit_ld_prim("P_ld16", MemW::W16, 2);
  emit_ld_prim("P_ld32", MemW::W32, 4);
  emit_ld_prim("P_ld64", MemW::W64, 8);

  // 29,31,33,35: st8, st16, st32, st64 ( a0=addr, a1=val )
  auto emit_st_prim = [&](const char* name, MemW mw, int k) {
    A.label(name);
    A.i(e_ld(RAX, R15, OFF_A0_W), "addr");
    A.i(e_movI(RCX, static_cast<u64>(k)));
    emit_check_inb(A, RAX, RCX);
    A.i(e_ld(R8, R15, OFF_MEMW_BASE));
    A.i(e_ld(RDX, R15, OFF_A1_W), "val");
    A.i(e_st_idx(mw, R8, RAX, 1, 0, RDX));
    A.i(e_ld(R9, R15, OFF_MEMP_BASE));
    A.i(e_ld(R10, R15, OFF_CUR_Q));
    for (int i = 0; i < k; ++i) {
      A.i(e_st_idx(MemW::W64, R9, RAX, 8, i * 8, R10));
    }
    A.jmp("L_prim_epilogue_0", "", true);
  };
  emit_st_prim("P_st8", MemW::W8, 1);
  emit_st_prim("P_st16", MemW::W16, 2);
  emit_st_prim("P_st32", MemW::W32, 4);
  emit_st_prim("P_st64", MemW::W64, 8);

  // 36: copy ( a0=n, a1=dst, a2=src ) overlap-safe
  A.label("P_copy");
  A.i(e_ld(RCX, R15, OFF_A0_W), "n");
  A.i(e_ld(RDX, R15, OFF_A1_W), "dst");
  A.i(e_ld(RAX, R15, OFF_A2_W), "src");
  emit_check_inb(A, RAX, RCX);
  emit_check_inb(A, RDX, RCX);
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "L_prim_epilogue_0", "", true);
  A.i(e_ld(R8, R15, OFF_MEMW_BASE));
  A.i(e_lea_idx(RSI, R8, RAX, 1, 0), "memw + src");
  A.i(e_lea_idx(RDI, R8, RDX, 1, 0), "memw + dst");
  A.i(e_ld(R9, R15, OFF_MEMP_BASE));
  A.i(e_lea_idx(R12, R9, RAX, 8, 0), "memp + src*8");
  A.i(e_lea_idx(R13, R9, RDX, 8, 0), "memp + dst*8");
  A.i(e_ld(R10, R15, OFF_CUR_Q));
  A.i(e_cmp(RDX, RAX), "dst <= src?");
  A.jcc(Cc::A, "cp_back");
  A.i(e_zero(R11));
  A.label("cp_fwd");
  A.i(e_cmp(R11, RCX));
  A.jcc(Cc::AE, "L_prim_epilogue_0", "", true);
  A.i(e_ld_idx(MemW::W8, RAX, RSI, R11, 1, 0));
  A.i(e_st_idx(MemW::W8, RDI, R11, 1, 0, RAX));
  A.i(e_ld_idx(MemW::W64, RAX, R12, R11, 8, 0));
  A.i(e_alu(AluOp::Add, RAX, R10));
  A.i(e_st_idx(MemW::W64, R13, R11, 8, 0, RAX));
  A.i(e_aluI(AluOp::Add, R11, 1));
  A.jmp("cp_fwd");
  A.label("cp_back");
  A.i(e_mov(R11, RCX));
  A.label("cp_bkw");
  A.i(e_test(R11, R11));
  A.jcc(Cc::E, "L_prim_epilogue_0", "", true);
  A.i(e_aluI(AluOp::Sub, R11, 1));
  A.i(e_ld_idx(MemW::W8, RAX, RSI, R11, 1, 0));
  A.i(e_st_idx(MemW::W8, RDI, R11, 1, 0, RAX));
  A.i(e_ld_idx(MemW::W64, RAX, R12, R11, 8, 0));
  A.i(e_alu(AluOp::Add, RAX, R10));
  A.i(e_st_idx(MemW::W64, R13, R11, 8, 0, RAX));
  A.jmp("cp_bkw");

  // 37: fill ( a0=b, a1=n, a2=dst )
  A.label("P_fill");
  A.i(e_ld(RDX, R15, OFF_A0_W), "b");
  A.i(e_ld(RCX, R15, OFF_A1_W), "n");
  A.i(e_ld(RAX, R15, OFF_A2_W), "dst");
  emit_check_inb(A, RAX, RCX);
  A.i(e_ld(R8, R15, OFF_MEMW_BASE));
  A.i(e_lea_idx(RDI, R8, RAX, 1, 0));
  A.i(e_ld(R9, R15, OFF_MEMP_BASE));
  A.i(e_lea_idx(RSI, R9, RAX, 8, 0));
  A.i(e_ld(R10, R15, OFF_CUR_Q));
  A.i(e_zero(R11));
  A.label("fill_loop");
  A.i(e_cmp(R11, RCX));
  A.jcc(Cc::AE, "L_prim_epilogue_0", "", true);
  A.i(e_st_idx(MemW::W8, RDI, R11, 1, 0, RDX));
  A.i(e_st_idx(MemW::W64, RSI, R11, 8, 0, R10));
  A.i(e_aluI(AluOp::Add, R11, 1));
  A.jmp("fill_loop");

  // 38: mcmp ( a0=n, a1=y, a2=x )
  A.label("P_mcmp");
  A.i(e_ld(RCX, R15, OFF_A0_W), "n");
  A.i(e_ld(RDX, R15, OFF_A1_W), "y");
  A.i(e_ld(RAX, R15, OFF_A2_W), "x");
  emit_check_inb(A, RAX, RCX);
  emit_check_inb(A, RDX, RCX);
  A.i(e_ld(R8, R15, OFF_MEMW_BASE));
  A.i(e_lea_idx(RSI, R8, RAX, 1, 0), "memw + x");
  A.i(e_lea_idx(RDI, R8, RDX, 1, 0), "memw + y");
  A.i(e_ld(R9, R15, OFF_MEMP_BASE));
  A.i(e_lea_idx(R12, R9, RAX, 8, 0), "memp + x*8");
  A.i(e_lea_idx(R13, R9, RDX, 8, 0), "memp + y*8");
  A.i(e_movI(RBX, 1), "eq = 1");
  A.i(e_ld(R10, R15, OFF_CUR_Q), "pm = cur_q");
  A.i(e_zero(R11), "i = 0");
  A.i(e_zero(R14), "0 constant for cmovne");
  A.label("mcmp_loop");
  A.i(e_cmp(R11, RCX));
  A.jcc(Cc::AE, "mcmp_done");
  A.i(e_ld_idx(MemW::W8, RAX, RSI, R11, 1, 0));
  A.i(e_ld_idx(MemW::W8, RDX, RDI, R11, 1, 0));
  A.i(e_cmp(RAX, RDX));
  A.i(e_cmovcc(Cc::NE, RBX, R14), "if mismatch eq = 0 (no short-circuit)");
  A.i(e_ld_idx(MemW::W64, RAX, R12, R11, 8, 0));
  A.i(e_alu(AluOp::Add, R10, RAX));
  A.i(e_ld_idx(MemW::W64, RAX, R13, R11, 8, 0));
  A.i(e_alu(AluOp::Add, R10, RAX));
  A.i(e_aluI(AluOp::Add, R11, 1));
  A.jmp("mcmp_loop");
  A.label("mcmp_done");
  A.i(e_st(R15, OFF_RES0_W, RBX));
  A.i(e_st(R15, OFF_RES0_P, R10));
  A.jmp("L_prim_epilogue", "", true);

  // 39..41: alloc, mark, release
  A.label("P_alloc");
  A.i(e_ld(RAX, R15, OFF_HP));
  A.i(e_ld(RCX, R15, OFF_A0_W), "n");
  emit_check_inb(A, RAX, RCX);
  A.i(e_mov(RDX, RAX));
  A.i(e_alu(AluOp::Add, RDX, RCX));
  A.i(e_st(R15, OFF_HP, RDX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_mark");
  A.i(e_ld(RAX, R15, OFF_HP));
  A.jmp("L_prim_epilogue_mk1", "", true);

  A.label("P_release");
  A.i(e_ld(RAX, R15, OFF_A0_W));
  A.i(e_ld(RCX, R15, OFF_HP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::A, "L_fail_bounds", "", true);
  A.i(e_st(R15, OFF_HP, RAX));
  A.jmp("L_prim_epilogue_0", "", true);

  // 42: streq ( a0=l2, a1=a2, a2=l1, a3=a1 )
  A.label("P_streq");
  A.i(e_ld(RAX, R15, OFF_A3_W), "a1");
  A.i(e_ld(RCX, R15, OFF_A2_W), "l1");
  emit_check_inb(A, RAX, RCX);
  A.i(e_ld(RDX, R15, OFF_A1_W), "a2");
  A.i(e_ld(R8, R15, OFF_A0_W), "l2");
  emit_check_inb(A, RDX, R8);
  A.i(e_cmp(RCX, R8), "l1 == l2?");
  A.jcc(Cc::E, "streq_eq_len");
  // Unequal length: return (0, cur_q), do NOT read memp!
  A.i(e_zero(RAX));
  A.jmp("L_prim_epilogue_mk1", "", true);
  A.label("streq_eq_len");
  A.i(e_ld(R8, R15, OFF_MEMW_BASE));
  A.i(e_lea_idx(RSI, R8, RAX, 1, 0), "memw + a1");
  A.i(e_lea_idx(RDI, R8, RDX, 1, 0), "memw + a2");
  A.i(e_ld(R9, R15, OFF_MEMP_BASE));
  A.i(e_lea_idx(R12, R9, RAX, 8, 0), "memp + a1*8");
  A.i(e_lea_idx(R13, R9, RDX, 8, 0), "memp + a2*8");
  A.i(e_movI(RBX, 1), "eq = 1");
  A.i(e_ld(R10, R15, OFF_CUR_Q), "pm = cur_q");
  A.i(e_zero(R11));
  A.i(e_zero(R14));
  A.label("streq_loop");
  A.i(e_cmp(R11, RCX));
  A.jcc(Cc::AE, "streq_done");
  A.i(e_ld_idx(MemW::W8, RAX, RSI, R11, 1, 0));
  A.i(e_ld_idx(MemW::W8, RDX, RDI, R11, 1, 0));
  A.i(e_cmp(RAX, RDX));
  A.i(e_cmovcc(Cc::NE, RBX, R14));
  A.i(e_ld_idx(MemW::W64, RAX, R12, R11, 8, 0));
  A.i(e_alu(AluOp::Add, R10, RAX));
  A.i(e_ld_idx(MemW::W64, RAX, R13, R11, 8, 0));
  A.i(e_alu(AluOp::Add, R10, RAX));
  A.i(e_aluI(AluOp::Add, R11, 1));
  A.jmp("streq_loop");
  A.label("streq_done");
  A.i(e_st(R15, OFF_RES0_W, RBX));
  A.i(e_st(R15, OFF_RES0_P, R10));
  A.jmp("L_prim_epilogue", "", true);

  // 43: strhash ( a0=l, a1=a ) FNV-1a
  A.label("P_strhash");
  A.i(e_ld(RCX, R15, OFF_A0_W), "l");
  A.i(e_ld(RAX, R15, OFF_A1_W), "a");
  emit_check_inb(A, RAX, RCX);
  A.i(e_ld(R8, R15, OFF_MEMW_BASE));
  A.i(e_lea_idx(RSI, R8, RAX, 1, 0));
  A.i(e_ld(R9, R15, OFF_MEMP_BASE));
  A.i(e_lea_idx(RDI, R9, RAX, 8, 0));
  A.i(e_movI(RAX, 14695981039346656037ULL), "FNV offset basis");
  A.i(e_movI(RDX, 1099511628211ULL), "FNV prime");
  A.i(e_ld(R10, R15, OFF_CUR_Q), "pm = cur_q");
  A.i(e_zero(R11), "i = 0");
  A.label("sh_loop");
  A.i(e_cmp(R11, RCX));
  A.jcc(Cc::AE, "sh_done");
  A.i(e_ld_idx(MemW::W8, RBX, RSI, R11, 1, 0));
  A.i(e_alu(AluOp::Xor, RAX, RBX));
  A.i(e_imul(RAX, RDX));
  A.i(e_ld_idx(MemW::W64, RBX, RDI, R11, 8, 0));
  A.i(e_alu(AluOp::Add, R10, RBX));
  A.i(e_aluI(AluOp::Add, R11, 1));
  A.jmp("sh_loop");
  A.label("sh_done");
  A.i(e_st(R15, OFF_RES0_W, RAX));
  A.i(e_st(R15, OFF_RES0_P, R10));
  A.jmp("L_prim_epilogue", "", true);

  // 44: tnew
  A.label("P_tnew");
  A.i(e_ld(RAX, R15, OFF_TABS_LEN));
  A.i(e_ld(RCX, R15, OFF_TABS_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::B, "tn_room");
  A.i(e_ld(RDI, R15, OFF_TAB_HEADS));
  A.i(e_mov(RSI, RCX));
  A.i(e_shlI(RSI, 3));
  A.i(e_mov(RDX, RSI));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_TAB_HEADS, RAX));
  A.i(e_ld(RCX, R15, OFF_TABS_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_TABS_CAP, RCX));
  A.label("tn_room");
  A.i(e_ld(RAX, R15, OFF_TABS_LEN));
  A.i(e_ld(RDX, R15, OFF_TAB_HEADS));
  A.i(e_movI(RCX, ~0ULL));
  A.i(e_st_idx(MemW::W64, RDX, RAX, 8, 0, RCX), "tab_heads[id] = ~0");
  A.i(e_lea(RCX, RAX, 1));
  A.i(e_st(R15, OFF_TABS_LEN, RCX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  // 45: tput ( a0=v, a1=k, a2=tb )
  A.label("P_tput");
  A.i(e_ld(R12, R15, OFF_A2_W), "tb");
  A.i(e_ld(RCX, R15, OFF_TABS_LEN));
  A.i(e_cmp(R12, RCX));
  A.jcc(Cc::AE, "L_fail_badTable", "", true);
  A.i(e_ld(RAX, R15, OFF_TAB_NODES_LEN));
  A.i(e_ld(RCX, R15, OFF_TAB_NODES_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::B, "tp_room");
  A.i(e_ld(RDI, R15, OFF_TAB_NODES));
  A.i(e_mov(RSI, RCX));
  A.i(e_shlI(RSI, 5), "* 32");
  A.i(e_mov(RDX, RSI));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_TAB_NODES, RAX));
  A.i(e_ld(RCX, R15, OFF_TAB_NODES_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_TAB_NODES_CAP, RCX));
  A.label("tp_room");
  A.i(e_ld(RAX, R15, OFF_TAB_NODES_LEN), "idx");
  A.i(e_mov(RDX, RAX));
  A.i(e_shlI(RDX, 5), "idx * 32");
  A.i(e_ld(R8, R15, OFF_TAB_NODES));
  A.i(e_alu(AluOp::Add, R8, RDX), "node ptr");
  A.i(e_ld(RCX, R15, OFF_A1_W), "k.w");
  A.i(e_st(R8, 0, RCX), "node.key");
  A.i(e_ld(RCX, R15, OFF_A0_W), "v.w");
  A.i(e_st(R8, 8, RCX), "node.val_w");
  A.i(e_ld(RCX, R15, OFF_A0_P), "v.p");
  A.i(e_ld(RDX, R15, OFF_CUR_Q));
  A.i(e_alu(AluOp::Add, RCX, RDX));
  A.i(e_st(R8, 16, RCX), "node.val_p = v.p + q");
  A.i(e_ld(R9, R15, OFF_TAB_HEADS));
  A.i(e_ld_idx(MemW::W64, RCX, R9, R12, 8, 0), "old head");
  A.i(e_st(R8, 24, RCX), "node.prev = old head");
  A.i(e_st_idx(MemW::W64, R9, R12, 8, 0, RAX), "tab_heads[tb] = idx");
  A.i(e_aluI(AluOp::Add, RAX, 1));
  A.i(e_st(R15, OFF_TAB_NODES_LEN, RAX));
  A.jmp("L_prim_epilogue_0", "", true);

  // 46: tget ( a0=k, a1=tb ) => [mk(1), {v.w, v.p + q}] or [mk(0), mk(0)]
  A.label("P_tget");
  A.i(e_ld(R12, R15, OFF_A1_W), "tb");
  A.i(e_ld(RCX, R15, OFF_TABS_LEN));
  A.i(e_cmp(R12, RCX));
  A.jcc(Cc::AE, "L_fail_badTable", "", true);
  A.i(e_ld(R9, R15, OFF_TAB_HEADS));
  A.i(e_ld_idx(MemW::W64, RAX, R9, R12, 8, 0), "idx = tab_heads[tb]");
  A.i(e_ld(R8, R15, OFF_TAB_NODES));
  A.i(e_ld(RDX, R15, OFF_A0_W), "target key");
  A.i(e_movI(RSI, ~0ULL));
  A.label("tg_loop");
  A.i(e_cmp(RAX, RSI));
  A.jcc(Cc::E, "tg_miss");
  A.i(e_mov(RCX, RAX));
  A.i(e_shlI(RCX, 5));
  A.i(e_ld_idx(MemW::W64, R10, R8, RCX, 1, 0), "node.key");
  A.i(e_cmp(R10, RDX));
  A.jcc(Cc::E, "tg_hit");
  A.i(e_ld_idx(MemW::W64, RAX, R8, RCX, 1, 24), "node.prev");
  A.jmp("tg_loop");
  A.label("tg_hit");
  A.i(e_movI(R10, 1));
  A.i(e_st(R15, OFF_RES0_W, R10));
  A.i(e_ld(R11, R15, OFF_CUR_Q));
  A.i(e_st(R15, OFF_RES0_P, R11));
  A.i(e_ld_idx(MemW::W64, RAX, R8, RCX, 1, 8), "node.val_w");
  A.i(e_st(R15, OFF_RES1_W, RAX));
  A.i(e_ld_idx(MemW::W64, RAX, R8, RCX, 1, 16), "node.val_p");
  A.i(e_alu(AluOp::Add, RAX, R11), "node.val_p + cur_q");
  A.i(e_st(R15, OFF_RES1_P, RAX));
  A.jmp("L_prim_epilogue", "", true);
  A.label("tg_miss");
  A.i(e_zero(RAX));
  A.i(e_st(R15, OFF_RES0_W, RAX));
  A.i(e_st(R15, OFF_RES1_W, RAX));
  A.i(e_ld(R11, R15, OFF_CUR_Q));
  A.i(e_st(R15, OFF_RES0_P, R11));
  A.i(e_st(R15, OFF_RES1_P, R11));
  A.jmp("L_prim_epilogue", "", true);

  // 47: getb
  A.label("P_getb");
  A.i(e_ld(RCX, R15, OFF_IPOS));
  A.i(e_ld(RDX, R15, OFF_INP_LEN));
  A.i(e_cmp(RCX, RDX));
  A.jcc(Cc::B, "gb_have");
  A.i(e_movI(RAX, ~0ULL));
  A.jmp("L_prim_epilogue_mk1", "", true);
  A.label("gb_have");
  A.i(e_ld(R8, R15, OFF_INP_BASE));
  A.i(e_ld_idx(MemW::W8, RAX, R8, RCX, 1, 0), "inp[ipos]");
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.i(e_st(R15, OFF_IPOS, RCX));
  A.i(e_st(R15, OFF_RES0_W, RAX));
  A.i(e_ld(RDX, R15, OFF_PC_PROV));
  A.i(e_aluI(AluOp::Add, RDX, 1));
  A.i(e_st(R15, OFF_RES0_P, RDX));
  A.jmp("L_prim_epilogue", "", true);

  // 48: putb
  A.label("P_putb");
  A.i(e_ld(RDI, R15, OFF_A0_W));
  A.call("fn_out_push_byte");
  A.jmp("L_prim_epilogue_0", "", true);

  // 49: putn
  A.label("P_putn");
  A.i(e_ld(RDI, R15, OFF_A0_W));
  A.call("fn_out_push_u64");
  A.jmp("L_prim_epilogue_0", "", true);

  // 50: exit
  A.label("P_exit");
  A.i(e_ld(RAX, R15, OFF_A0_W));
  A.i(e_aluI(AluOp::And, RAX, 0xff));
  A.i(e_st(R15, OFF_HALT_CODE, RAX));
  A.jmp("L_vm_halt", "", true);

  // 51..53: clock, core, sensor -> mk(0)
  A.label("P_zero_mk1");
  A.i(e_zero(RAX));
  A.jmp("L_prim_epilogue_mk1", "", true);

  // 54: provget ( a -- a len ) => res0 = mk(a.plen), res1 = t(a)
  A.label("P_provget");
  A.i(e_ld(RAX, R15, OFF_A0_P));
  A.i(e_st(R15, OFF_RES0_W, RAX));
  A.i(e_ld(RDX, R15, OFF_CUR_Q));
  A.i(e_st(R15, OFF_RES0_P, RDX));
  tag_val(OFF_A0_W, OFF_A0_P, OFF_RES1_W, OFF_RES1_P);
  A.jmp("L_prim_epilogue", "", true);

  // 55: confget ( a -- a 0 ) => res0 = mk(0), res1 = t(a)
  A.label("P_confget");
  A.i(e_zero(RAX));
  A.i(e_st(R15, OFF_RES0_W, RAX));
  A.i(e_ld(RDX, R15, OFF_CUR_Q));
  A.i(e_st(R15, OFF_RES0_P, RDX));
  tag_val(OFF_A0_W, OFF_A0_P, OFF_RES1_W, OFF_RES1_P);
  A.jmp("L_prim_epilogue", "", true);

  // 56: demand ( a th -- a ) => a0=th, a1=a. Under noPhys fail iff 0 < th.w; result is t(a1)!
  A.label("P_demand");
  A.i(e_ld(RAX, R15, OFF_A0_W));
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::NE, "L_fail_demand", "", true);
  tag_val(OFF_A1_W, OFF_A1_P, OFF_RES0_W, OFF_RES0_P);
  A.jmp("L_prim_epilogue", "", true);

  // -------------------------------------------------------------------------
  // 13 Non-Prim Op Handlers (57..69)
  // -------------------------------------------------------------------------

  // 57: lit n
  A.label("O_lit");
  A.i(e_ld(RDI, R15, OFF_CUR_IMM));
  A.i(e_ld(RSI, R15, OFF_PC_PROV));
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.call("fn_ds_push");
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // 58: str i
  A.label("O_str");
  emit_vote_code_const(A, 16, "str_nd");
  A.i(e_ld(R8, R15, OFF_CUR_IMM), "i");
  A.i(e_cmp(R8, RAX));
  A.jcc(Cc::AE, "L_fail_bounds", "", true);
  A.i(e_shlI(R8, 4), "16 * i");
  A.i(e_aluI(AluOp::Add, R8, 32), "off = 32 + 16*i");
  emit_vote_code_off(A, R8, 0, "str_addr");
  A.i(e_st(R15, OFF_TMP0, RAX));
  emit_vote_code_off(A, R8, 8, "str_len");
  A.i(e_st(R15, OFF_TMP1, RAX));
  A.i(e_ld(RDI, R15, OFF_TMP0), "addr");
  A.i(e_ld(RSI, R15, OFF_PC_PROV));
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.call("fn_ds_push");
  A.i(e_ld(RDI, R15, OFF_TMP1), "len");
  A.i(e_ld(RSI, R15, OFF_PC_PROV));
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.call("fn_ds_push");
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // 59: pick n
  A.label("O_pick");
  A.i(e_ld(RAX, R15, OFF_CUR_IMM), "n");
  A.i(e_ld(RCX, R15, OFF_DS_LEN));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::AE, "L_fail_underflow", "", true);
  A.i(e_alu(AluOp::Sub, RCX, RAX));
  A.i(e_aluI(AluOp::Sub, RCX, 1), "idx = ds_len - 1 - n");
  A.i(e_shlI(RCX, 4));
  A.i(e_ld(RDX, R15, OFF_DS_BASE));
  A.i(e_ld_idx(MemW::W64, RDI, RDX, RCX, 1, 0));
  A.i(e_ld_idx(MemW::W64, RSI, RDX, RCX, 1, 8));
  A.i(e_ld(R8, R15, OFF_PC_PROV));
  A.i(e_alu(AluOp::Add, RSI, R8));
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.call("fn_ds_push");
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // 60: tor
  A.label("O_tor");
  A.i(e_ld(RCX, R15, OFF_DS_LEN));
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "L_fail_underflow", "", true);
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.i(e_st(R15, OFF_DS_LEN, RCX));
  A.i(e_shlI(RCX, 4));
  A.i(e_ld(RDX, R15, OFF_DS_BASE));
  A.i(e_ld_idx(MemW::W64, RDI, RDX, RCX, 1, 0));
  A.i(e_ld_idx(MemW::W64, RSI, RDX, RCX, 1, 8));
  A.i(e_ld(R8, R15, OFF_PC_PROV));
  A.i(e_alu(AluOp::Add, RSI, R8));
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.call("fn_aux_push");
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // 61: fromr
  A.label("O_fromr");
  A.i(e_ld(RCX, R15, OFF_AUX_LEN));
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "L_fail_auxUnderflow", "", true);
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.i(e_st(R15, OFF_AUX_LEN, RCX));
  A.i(e_shlI(RCX, 4));
  A.i(e_ld(RDX, R15, OFF_AUX_BASE));
  A.i(e_ld_idx(MemW::W64, RDI, RDX, RCX, 1, 0));
  A.i(e_ld_idx(MemW::W64, RSI, RDX, RCX, 1, 8));
  A.i(e_ld(R8, R15, OFF_PC_PROV));
  A.i(e_alu(AluOp::Add, RSI, R8));
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.call("fn_ds_push");
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // 62: rfetch
  A.label("O_rfetch");
  A.i(e_ld(RCX, R15, OFF_AUX_LEN));
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "L_fail_auxUnderflow", "", true);
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.i(e_shlI(RCX, 4));
  A.i(e_ld(RDX, R15, OFF_AUX_BASE));
  A.i(e_ld_idx(MemW::W64, RDI, RDX, RCX, 1, 0));
  A.i(e_ld_idx(MemW::W64, RSI, RDX, RCX, 1, 8));
  A.i(e_ld(R8, R15, OFF_PC_PROV));
  A.i(e_alu(AluOp::Add, RSI, R8));
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.call("fn_ds_push");
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // 63: locget i
  A.label("O_locget");
  A.i(e_ld(RAX, R15, OFF_RS_BASE));
  A.i(e_ld(RCX, R15, OFF_RS_LEN));
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.i(e_shlI(RCX, 5));
  A.i(e_alu(AluOp::Add, RAX, RCX), "top frame");
  A.i(e_ld(RDX, R15, OFF_CUR_IMM), "i");
  A.i(e_ld(R8, RAX, 24), "nlocs");
  A.i(e_cmp(RDX, R8));
  A.jcc(Cc::AE, "L_fail_badLocal", "", true);
  A.i(e_ld(R9, RAX, 16), "locs_off");
  A.i(e_alu(AluOp::Add, R9, RDX));
  A.i(e_shlI(R9, 4));
  A.i(e_ld(R10, R15, OFF_LOCS_BASE));
  A.i(e_ld_idx(MemW::W64, RDI, R10, R9, 1, 0));
  A.i(e_ld_idx(MemW::W64, RSI, R10, R9, 1, 8));
  A.i(e_ld(R11, R15, OFF_PC_PROV));
  A.i(e_alu(AluOp::Add, RSI, R11));
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.call("fn_ds_push");
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // 64: locset i (check ds not empty FIRST, then i < nlocs — matches SporosProvLen.lstep:278-283)
  A.label("O_locset");
  A.i(e_ld(R12, R15, OFF_DS_LEN));
  A.i(e_test(R12, R12));
  A.jcc(Cc::E, "L_fail_underflow", "", true);
  A.i(e_ld(RAX, R15, OFF_RS_BASE));
  A.i(e_ld(RCX, R15, OFF_RS_LEN));
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.i(e_shlI(RCX, 5));
  A.i(e_alu(AluOp::Add, RAX, RCX), "top frame");
  A.i(e_ld(RDX, R15, OFF_CUR_IMM), "i");
  A.i(e_ld(R8, RAX, 24), "nlocs");
  A.i(e_cmp(RDX, R8));
  A.jcc(Cc::AE, "L_fail_badLocal", "", true);
  A.i(e_aluI(AluOp::Sub, R12, 1));
  A.i(e_st(R15, OFF_DS_LEN, R12));
  A.i(e_shlI(R12, 4));
  A.i(e_ld(RSI, R15, OFF_DS_BASE));
  A.i(e_ld_idx(MemW::W64, RDI, RSI, R12, 1, 0), "v.w");
  A.i(e_ld_idx(MemW::W64, R13, RSI, R12, 1, 8), "v.p");
  A.i(e_ld(R11, R15, OFF_PC_PROV));
  A.i(e_alu(AluOp::Add, R13, R11));
  A.i(e_aluI(AluOp::Add, R13, 1));
  A.i(e_ld(R9, RAX, 16), "locs_off");
  A.i(e_alu(AluOp::Add, R9, RDX));
  A.i(e_shlI(R9, 4));
  A.i(e_ld(R10, R15, OFF_LOCS_BASE));
  A.i(e_st_idx(MemW::W64, R10, R9, 1, 0, RDI));
  A.i(e_st_idx(MemW::W64, R10, R9, 1, 8, R13));
  emit_adv_pc(A);
  A.jmp("L_vm_loop", "", true);

  // 65: jmp t
  A.label("O_jmp");
  A.i(e_ld(RAX, R15, OFF_RS_BASE));
  A.i(e_ld(RCX, R15, OFF_RS_LEN));
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.i(e_shlI(RCX, 5));
  A.i(e_alu(AluOp::Add, RAX, RCX));
  A.i(e_ld(RDX, R15, OFF_CUR_IMM));
  A.i(e_st(RAX, 8, RDX));
  A.jmp("L_vm_loop", "", true);

  // 66, 67: jz t, jnz t
  auto emit_jcc_op = [&](const char* name, bool is_jz) {
    A.label(name);
    A.i(e_ld(RCX, R15, OFF_DS_LEN));
    A.i(e_test(RCX, RCX));
    A.jcc(Cc::E, "L_fail_underflow", "", true);
    A.i(e_aluI(AluOp::Sub, RCX, 1));
    A.i(e_st(R15, OFF_DS_LEN, RCX));
    A.i(e_shlI(RCX, 4));
    A.i(e_ld(RDX, R15, OFF_DS_BASE));
    A.i(e_ld_idx(MemW::W64, R8, RDX, RCX, 1, 0), "c.w");
    A.i(e_ld_idx(MemW::W64, R9, RDX, RCX, 1, 8), "c.plen");
    A.i(e_ld(R10, R15, OFF_PC_PROV));
    A.i(e_alu(AluOp::Add, R10, R9));
    A.i(e_aluI(AluOp::Add, R10, 1));
    A.i(e_st(R15, OFF_PC_PROV, R10), "pc = pc + c.plen + 1");
    A.i(e_ld(RAX, R15, OFF_RS_BASE));
    A.i(e_ld(RCX, R15, OFF_RS_LEN));
    A.i(e_aluI(AluOp::Sub, RCX, 1));
    A.i(e_shlI(RCX, 5));
    A.i(e_alu(AluOp::Add, RAX, RCX), "top frame");
    A.i(e_ld(RDX, RAX, 8), "f.pc");
    A.i(e_aluI(AluOp::Add, RDX, 1), "fallthrough = f.pc + 1");
    A.i(e_ld(RSI, R15, OFF_CUR_IMM), "target = t");
    A.i(e_test(R8, R8));
    A.i(e_cmovcc(is_jz ? Cc::E : Cc::NE, RDX, RSI));
    A.i(e_st(RAX, 8, RDX));
    A.jmp("L_vm_loop", "", true);
  };
  emit_jcc_op("O_jz", true);
  emit_jcc_op("O_jnz", false);

  // 68: call w
  A.label("O_call");
  emit_vote_code_const(A, 8, "call_nw");
  A.i(e_ld(R12, R15, OFF_CUR_IMM), "w");
  A.i(e_cmp(R12, RAX));
  A.jcc(Cc::AE, "L_fail_badWord", "", true);
  emit_vote_code_const(A, 16, "call_nd");
  A.i(e_shlI(RAX, 4), "16 * nd");
  A.i(e_mov(R8, R12));
  A.i(e_shlI(R8, 6), "64 * w");
  A.i(e_alu(AluOp::Add, R8, RAX));
  A.i(e_aluI(AluOp::Add, R8, 32), "w_off = 32 + 16*nd + 64*w");
  emit_vote_code_off(A, R8, 0, "call_ins");
  A.i(e_ld(RCX, R15, OFF_DS_LEN));
  A.i(e_cmp(RCX, RAX), "ds_len < c.ins?");
  A.jcc(Cc::B, "L_fail_underflow", "", true);
  emit_vote_code_off(A, R8, 16, "call_locs");
  A.i(e_mov(R13, RAX), "r13 = c.locals");
  // Ensure rs_len < rs_cap
  A.i(e_ld(RAX, R15, OFF_RS_LEN));
  A.i(e_ld(RCX, R15, OFF_RS_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::B, "call_rs_ok");
  A.i(e_ld(RDI, R15, OFF_RS_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_shlI(RSI, 5), "cap * 32");
  A.i(e_mov(RDX, RSI));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_RS_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_RS_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_RS_CAP, RCX));
  A.label("call_rs_ok");
  // Ensure locs_len + c.locals <= locs_cap
  A.label("call_locs_chk");
  A.i(e_ld(RAX, R15, OFF_LOCS_LEN));
  A.i(e_alu(AluOp::Add, RAX, R13), "needed_locs");
  A.i(e_ld(RCX, R15, OFF_LOCS_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::BE, "call_locs_ok");
  A.i(e_ld(RDI, R15, OFF_LOCS_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_shlI(RSI, 4));
  A.i(e_mov(RDX, RSI));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_LOCS_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_LOCS_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_LOCS_CAP, RCX));
  A.jmp("call_locs_chk");
  A.label("call_locs_ok");
  // Advance caller f.pc += 1
  emit_adv_pc(A);
  // Init c.locals slots at locs_base + locs_len*16 to (0, pc_prov + 1)
  A.i(e_ld(R8, R15, OFF_LOCS_BASE));
  A.i(e_ld(R9, R15, OFF_LOCS_LEN), "old locs_len");
  A.i(e_ld(R10, R15, OFF_PC_PROV));
  A.i(e_aluI(AluOp::Add, R10, 1), "pc + 1");
  A.i(e_zero(RBX), "i = 0");
  A.i(e_mov(R11, RBX), "0 (avoids reading syscall-clobbered R11)");
  A.label("call_init_locs");
  A.i(e_cmp(RBX, R13));
  A.jcc(Cc::AE, "call_locs_done");
  A.i(e_mov(RDX, R9));
  A.i(e_alu(AluOp::Add, RDX, RBX));
  A.i(e_shlI(RDX, 4));
  A.i(e_st_idx(MemW::W64, R8, RDX, 1, 0, R11));
  A.i(e_st_idx(MemW::W64, R8, RDX, 1, 8, R10));
  A.i(e_aluI(AluOp::Add, RBX, 1));
  A.jmp("call_init_locs");
  A.label("call_locs_done");
  A.i(e_mov(RAX, R9));
  A.i(e_alu(AluOp::Add, RAX, R13));
  A.i(e_st(R15, OFF_LOCS_LEN, RAX));
  // Push frame (w=r12, pc=0, locs_off=r9, nlocs=r13)
  A.i(e_ld(RAX, R15, OFF_RS_BASE));
  A.i(e_ld(RCX, R15, OFF_RS_LEN));
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 5));
  A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 0, R12));
  A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 8, R11));
  A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 16, R9));
  A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 24, R13));
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.i(e_st(R15, OFF_RS_LEN, RCX));
  A.jmp("L_vm_loop", "", true);

  // 69: ret
  A.label("O_ret");
  A.i(e_ld(RCX, R15, OFF_RS_LEN));
  A.i(e_cmpI(RCX, 1));
  A.jcc(Cc::A, "ret_pop");
  A.i(e_zero(RAX));
  A.i(e_st(R15, OFF_HALT_CODE, RAX));
  A.jmp("L_vm_halt", "", true);
  A.label("ret_pop");
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.i(e_st(R15, OFF_RS_LEN, RCX));
  A.i(e_shlI(RCX, 5));
  A.i(e_ld(RAX, R15, OFF_RS_BASE));
  A.i(e_ld_idx(MemW::W64, RDX, RAX, RCX, 1, 16), "popped.locs_off");
  A.i(e_st(R15, OFF_LOCS_LEN, RDX));
  A.jmp("L_vm_loop", "", true);

  // -------------------------------------------------------------------------
  // Halt, Fail, and Out-of-memory exits
  // -------------------------------------------------------------------------
  A.label("L_vm_halt");
  A.i(e_ld(R12, R15, OFF_OUT_LEN));
  A.i(e_zero(R13), "written = 0");
  A.label("halt_flush");
  A.i(e_cmp(R13, R12));
  A.jcc(Cc::AE, "halt_exit");
  A.i(e_ld(RSI, R15, OFF_OUT_BASE));
  A.i(e_alu(AluOp::Add, RSI, R13));
  A.i(e_mov(RDX, R12));
  A.i(e_alu(AluOp::Sub, RDX, R13));
  A.i(e_movI(RDI, 1), "fd 1");
  A.i(e_movI(RAX, 1), "sys_write");
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::LE, "halt_werr");
  A.i(e_alu(AluOp::Add, R13, RAX));
  A.jmp("halt_flush");
  A.label("halt_werr");
  A.i(e_movI(RDI, 1));
  A.i(e_movI(RAX, 60));
  A.i(e_syscall());
  A.label("halt_exit");
  A.i(e_ld(RDI, R15, OFF_HALT_CODE));
  A.i(e_aluI(AluOp::And, RDI, 0xff));
  A.i(e_movI(RAX, 60));
  A.i(e_syscall());

  auto emit_fail_stub = [&](const char* name, ErrId id) {
    A.label(name);
    emit_fail_jump(A, id);
  };
  emit_fail_stub("L_fail_underflow", ERR_UNDERFLOW);
  emit_fail_stub("L_fail_auxUnderflow", ERR_AUX_UNDERFLOW);
  emit_fail_stub("L_fail_badLocal", ERR_BAD_LOCAL);
  emit_fail_stub("L_fail_badJump", ERR_BAD_JUMP);
  emit_fail_stub("L_fail_badWord", ERR_BAD_WORD);
  emit_fail_stub("L_fail_noFrame", ERR_NO_FRAME);
  emit_fail_stub("L_fail_bounds", ERR_BOUNDS);
  emit_fail_stub("L_fail_divZero", ERR_DIV_ZERO);
  emit_fail_stub("L_fail_badTable", ERR_BAD_TABLE);
  emit_fail_stub("L_fail_demand", ERR_DEMAND);

  A.label("L_vm_fail");
  A.mov_label(RSI, "L_str_run_failed");
  A.i(e_movI(RDX, static_cast<u64>(kRunFailPreStr.size())));
  A.call("fn_write_stderr");
  A.i(e_ld(RAX, R15, OFF_ERR_ID));
  A.mov_label(RCX, "L_err_ptrs");
  A.i(e_ld_idx(MemW::W64, RSI, RCX, RAX, 8, 0));
  A.mov_label(RCX, "L_err_lens");
  A.i(e_ld_idx(MemW::W64, RDX, RCX, RAX, 8, 0));
  A.call("fn_write_stderr");
  A.mov_label(RSI, "L_str_nl");
  A.i(e_movI(RDX, static_cast<u64>(kNlStr.size())));
  A.call("fn_write_stderr");
  A.i(e_movI(RDI, 1));
  A.i(e_movI(RAX, 60));
  A.i(e_syscall());

  A.label("L_exit_nomem");
  A.mov_label(RSI, "L_str_nomem");
  A.i(e_movI(RDX, static_cast<u64>(kNomemStr.size())));
  A.call("fn_write_stderr");
  A.i(e_movI(RDI, 2));
  A.i(e_movI(RAX, 60));
  A.i(e_syscall());
}

// ---------------------------------------------------------------------------
// Startup: CLI, open/read/close, SPR1 decode, checkProg, arena setup, triplication
// ---------------------------------------------------------------------------

void emit_checker_helpers(Asm& A) {
  // fn_ann_lookup: rdi = ann_ptr, rsi = na, rdx = target_i
  // returns rax = 1 (found) or 0, rcx = d, r8 = a. Clobbers r9, r10, r11.
  A.label("fn_ann_lookup");
  A.i(e_zero(R9), "j = 0");
  A.label("al_loop");
  A.i(e_cmp(R9, RSI));
  A.jcc(Cc::AE, "al_miss");
  A.i(e_mov(R10, R9));
  A.i(e_shlI(R10, 3));
  A.i(e_mov(R11, R9));
  A.i(e_shlI(R11, 2));
  A.i(e_alu(AluOp::Add, R10, R11), "12 * j");
  A.i(e_ld_idx(MemW::W32, R11, RDI, R10, 1, 0), "ann[j].t");
  A.i(e_cmp(R11, RDX));
  A.jcc(Cc::E, "al_hit");
  A.i(e_aluI(AluOp::Add, R9, 1));
  A.jmp("al_loop");
  A.label("al_hit");
  A.i(e_ld_idx(MemW::W32, RCX, RDI, R10, 1, 4), "d");
  A.i(e_ld_idx(MemW::W32, R8, RDI, R10, 1, 8), "a");
  A.i(e_movI(RAX, 1));
  A.i(e_ret());
  A.label("al_miss");
  A.i(e_zero(RAX));
  A.i(e_ret());

  // fn_hstep: uses OFF_CHK_WBASE, OFF_NW, OFF_CHK_HD, OFF_CHK_HA;
  // takes rdi = op, rsi = imm.
  // returns rax = 1 (ok) or 0; rdx = has_fall (0/1), rcx = d', r8 = a'.
  A.label("fn_hstep");
  A.i(e_ld(R9, R15, OFF_CHK_HD), "h.d");
  A.i(e_ld(R10, R15, OFF_CHK_HA), "h.a");
  A.i(e_cmpI(RDI, kNPrims));
  A.jcc(Cc::AE, "hs_nonprim");
  A.mov_label(R11, "L_prim_eff_in");
  A.i(e_ld_idx(MemW::W8, RCX, R11, RDI, 1, 0), "e_in");
  A.i(e_cmp(R9, RCX));
  A.jcc(Cc::B, "hs_bad", "", true);
  A.i(e_cmpI(RDI, 50), "exit?");
  A.jcc(Cc::E, "hs_nofall", "", true);
  A.mov_label(R11, "L_prim_eff_out");
  A.i(e_ld_idx(MemW::W8, R8, R11, RDI, 1, 0), "e_out");
  A.i(e_alu(AluOp::Sub, R9, RCX));
  A.i(e_alu(AluOp::Add, R9, R8));
  A.i(e_mov(RCX, R9));
  A.i(e_mov(R8, R10));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_nonprim");
  A.i(e_cmpI(RDI, OP_LIT));
  A.jcc(Cc::NE, "hs_str");
  A.i(e_lea(RCX, R9, 1));
  A.i(e_mov(R8, R10));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_str");
  A.i(e_cmpI(RDI, OP_STR));
  A.jcc(Cc::NE, "hs_pick");
  A.i(e_lea(RCX, R9, 2));
  A.i(e_mov(R8, R10));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_pick");
  A.i(e_cmpI(RDI, OP_PICK));
  A.jcc(Cc::NE, "hs_tor");
  A.i(e_cmp(RSI, R9));
  A.jcc(Cc::AE, "hs_bad", "", true);
  A.i(e_lea(RCX, R9, 1));
  A.i(e_mov(R8, R10));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_tor");
  A.i(e_cmpI(RDI, OP_TOR));
  A.jcc(Cc::NE, "hs_fromr");
  A.i(e_cmpI(R9, 1));
  A.jcc(Cc::B, "hs_bad", "", true);
  A.i(e_lea(RCX, R9, -1));
  A.i(e_lea(R8, R10, 1));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_fromr");
  A.i(e_cmpI(RDI, OP_FROMR));
  A.jcc(Cc::NE, "hs_rfetch");
  A.i(e_cmpI(R10, 1));
  A.jcc(Cc::B, "hs_bad", "", true);
  A.i(e_lea(RCX, R9, 1));
  A.i(e_lea(R8, R10, -1));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_rfetch");
  A.i(e_cmpI(RDI, OP_RFETCH));
  A.jcc(Cc::NE, "hs_locget");
  A.i(e_cmpI(R10, 1));
  A.jcc(Cc::B, "hs_bad", "", true);
  A.i(e_lea(RCX, R9, 1));
  A.i(e_mov(R8, R10));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_locget");
  A.i(e_cmpI(RDI, OP_LOCGET));
  A.jcc(Cc::NE, "hs_locset");
  A.i(e_ld(R11, R15, OFF_CHK_WBASE));
  A.i(e_ld(RCX, R11, 16), "wd.locals");
  A.i(e_cmp(RSI, RCX));
  A.jcc(Cc::AE, "hs_bad", "", true);
  A.i(e_lea(RCX, R9, 1));
  A.i(e_mov(R8, R10));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_locset");
  A.i(e_cmpI(RDI, OP_LOCSET));
  A.jcc(Cc::NE, "hs_jmp");
  A.i(e_cmpI(R9, 1));
  A.jcc(Cc::B, "hs_bad", "", true);
  A.i(e_ld(R11, R15, OFF_CHK_WBASE));
  A.i(e_ld(RCX, R11, 16), "wd.locals");
  A.i(e_cmp(RSI, RCX));
  A.jcc(Cc::AE, "hs_bad", "", true);
  A.i(e_lea(RCX, R9, -1));
  A.i(e_mov(R8, R10));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_jmp");
  A.i(e_cmpI(RDI, OP_JMP));
  A.jcc(Cc::NE, "hs_jz");
  A.i(e_ld(RDI, R15, OFF_CHK_ANN));
  A.i(e_mov(RDX, RSI));
  A.i(e_ld(RSI, R15, OFF_CHK_NA));
  A.call("fn_ann_lookup");
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::E, "hs_bad", "", true);
  A.i(e_ld(R9, R15, OFF_CHK_HD));
  A.i(e_ld(R10, R15, OFF_CHK_HA));
  A.i(e_cmp(RCX, R9));
  A.jcc(Cc::NE, "hs_bad", "", true);
  A.i(e_cmp(R8, R10));
  A.jcc(Cc::NE, "hs_bad", "", true);
  A.jmp("hs_nofall", "", true);

  A.label("hs_jz");
  A.i(e_cmpI(RDI, OP_JZ));
  A.jcc(Cc::E, "hs_jcc");
  A.i(e_cmpI(RDI, OP_JNZ));
  A.jcc(Cc::NE, "hs_call");
  A.label("hs_jcc");
  A.i(e_cmpI(R9, 1));
  A.jcc(Cc::B, "hs_bad", "", true);
  A.i(e_ld(RDI, R15, OFF_CHK_ANN));
  A.i(e_mov(RDX, RSI));
  A.i(e_ld(RSI, R15, OFF_CHK_NA));
  A.call("fn_ann_lookup");
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::E, "hs_bad", "", true);
  A.i(e_ld(R9, R15, OFF_CHK_HD));
  A.i(e_aluI(AluOp::Sub, R9, 1));
  A.i(e_ld(R10, R15, OFF_CHK_HA));
  A.i(e_cmp(RCX, R9));
  A.jcc(Cc::NE, "hs_bad", "", true);
  A.i(e_cmp(R8, R10));
  A.jcc(Cc::NE, "hs_bad", "", true);
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_call");
  A.i(e_cmpI(RDI, OP_CALL));
  A.jcc(Cc::NE, "hs_ret");
  A.i(e_ld(R11, R15, OFF_NW));
  A.i(e_cmp(RSI, R11));
  A.jcc(Cc::AE, "hs_bad", "", true);
  A.i(e_ld(RAX, R15, OFF_CODE1));
  A.i(e_ld(RDX, R15, OFF_ND));
  A.i(e_shlI(RDX, 4));
  A.i(e_aluI(AluOp::Add, RDX, 32));
  A.i(e_shlI(RSI, 6));
  A.i(e_alu(AluOp::Add, RDX, RSI));
  A.i(e_alu(AluOp::Add, RAX, RDX), "words[w]");
  A.i(e_ld(RCX, RAX, 0), "c.ins");
  A.i(e_ld(R8, RAX, 8), "c.outs");
  A.i(e_cmp(R9, RCX));
  A.jcc(Cc::B, "hs_bad", "", true);
  A.i(e_alu(AluOp::Sub, R9, RCX));
  A.i(e_alu(AluOp::Add, R9, R8));
  A.i(e_mov(RCX, R9));
  A.i(e_mov(R8, R10));
  A.i(e_movI(RDX, 1));
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_ret");
  A.i(e_cmpI(RDI, OP_RET));
  A.jcc(Cc::NE, "hs_bad");
  A.i(e_ld(R11, R15, OFF_CHK_WBASE));
  A.i(e_ld(RCX, R11, 8), "wd.outs");
  A.i(e_cmp(R9, RCX));
  A.jcc(Cc::NE, "hs_bad");
  A.i(e_test(R10, R10));
  A.jcc(Cc::NE, "hs_bad");
  A.label("hs_nofall");
  A.i(e_zero(RDX), "has_fall = 0");
  A.i(e_movI(RAX, 1));
  A.i(e_ret());

  A.label("hs_bad");
  A.i(e_zero(RAX));
  A.i(e_ret());
}

void emit_boot(Asm& A) {
  A.label("L_start");
  // Initialize all GPRs (except RSP) to 0 so X86Vm states never hold undefined registers.
  for (int r = 0; r < 16; ++r) {
    if (r != RSP && r != R15) A.i(e_zero(r));
  }
  A.i(e_movI(R15, kCtxBase), "r15 = VmCtx");

  // Parse CLI: argc = [rsp], argv[1] = [rsp+16], optional argv[2] = [rsp+24]
  A.i(e_ld(RAX, RSP, 0), "argc");
  A.i(e_cmpI(RAX, 2));
  A.jcc(Cc::AE, "boot_argc_ok");
  A.mov_label(RSI, "L_str_usage");
  A.i(e_movI(RDX, static_cast<u64>(kUsageStr.size())));
  A.call("fn_write_stderr");
  A.i(e_movI(RDI, 1));
  A.i(e_movI(RAX, 60));
  A.i(e_syscall());

  A.label("boot_argc_ok");
  A.i(e_ld(RDI, RSP, 16), "argv[1]");
  A.i(e_st(R15, OFF_ARGV1, RDI));
  A.i(e_movI(RAX, kDefaultMsize));
  A.i(e_st(R15, OFF_MSIZE, RAX));
  A.i(e_ld(RAX, RSP, 0), "argc");
  A.i(e_cmpI(RAX, 3));
  A.jcc(Cc::B, "boot_msize_ok");
  // Parse decimal argv[2]
  A.i(e_ld(RSI, RSP, 24), "argv[2]");
  A.i(e_zero(RAX));
  A.i(e_zero(RCX));
  A.label("parse_ms_loop");
  A.i(e_ld_idx(MemW::W8, RDX, RSI, RCX, 1, 0));
  A.i(e_cmpI(RDX, '0'));
  A.jcc(Cc::B, "parse_ms_done");
  A.i(e_cmpI(RDX, '9'));
  A.jcc(Cc::A, "parse_ms_done");
  A.i(e_movI(R8, 10));
  A.i(e_imul(RAX, R8));
  A.i(e_aluI(AluOp::Sub, RDX, '0'));
  A.i(e_alu(AluOp::Add, RAX, RDX));
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.jmp("parse_ms_loop");
  A.label("parse_ms_done");
  A.i(e_st(R15, OFF_MSIZE, RAX));

  A.label("boot_msize_ok");
  A.i(e_ld(RAX, R15, OFF_MSIZE));
  A.i(e_movI(RCX, kMaxMsize));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::AE, "L_exit_nomem", "msize >= 2^44", true);

  // Open argv[1]
  A.i(e_ld(RDI, R15, OFF_ARGV1));
  A.i(e_zero(RSI), "O_RDONLY");
  A.i(e_zero(RDX), "mode = 0");
  A.i(e_movI(RAX, 2), "sys_open");
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::S, "L_err_open", "", true);
  A.i(e_st(R15, OFF_FILE_FD, RAX));

  // Read file into mmap/mremap buffer (initial 4096 B)
  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_FILE_BASE, RAX));
  A.i(e_movI(RCX, kPage));
  A.i(e_st(R15, OFF_FILE_CAP, RCX));
  A.i(e_zero(RCX));
  A.i(e_st(R15, OFF_FILE_LEN, RCX));

  A.label("rd_file_loop");
  A.i(e_ld(RAX, R15, OFF_FILE_LEN));
  A.i(e_ld(RCX, R15, OFF_FILE_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::B, "rd_file_go");
  A.i(e_ld(RDI, R15, OFF_FILE_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_FILE_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_FILE_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_FILE_CAP, RCX));
  A.label("rd_file_go");
  A.i(e_ld(RDI, R15, OFF_FILE_FD));
  A.i(e_ld(RSI, R15, OFF_FILE_BASE));
  A.i(e_ld(RAX, R15, OFF_FILE_LEN));
  A.i(e_alu(AluOp::Add, RSI, RAX));
  A.i(e_ld(RDX, R15, OFF_FILE_CAP));
  A.i(e_alu(AluOp::Sub, RDX, RAX));
  A.i(e_zero(RAX), "sys_read");
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::S, "L_err_open", "", true);
  A.jcc(Cc::E, "rd_file_done");
  A.i(e_ld(RCX, R15, OFF_FILE_LEN));
  A.i(e_alu(AluOp::Add, RCX, RAX));
  A.i(e_st(R15, OFF_FILE_LEN, RCX));
  A.jmp("rd_file_loop");

  A.label("rd_file_done");
  A.i(e_ld(RDI, R15, OFF_FILE_FD));
  A.i(e_movI(RAX, 3), "sys_close");
  A.i(e_syscall());

  // -------------------------------------------------------------------------
  // Pass 1: Validate SPR1 format & count total_instrs, total_data
  // -------------------------------------------------------------------------
  A.i(e_ld(R12, R15, OFF_FILE_BASE), "file_base");
  A.i(e_ld(R13, R15, OFF_FILE_LEN), "file_len");
  A.i(e_cmpI(R13, 16));
  A.jcc(Cc::B, "L_err_decode", "", true);
  A.i(e_ld32(RAX, R12, 0));
  A.i(e_cmpI(RAX, 0x31525053));
  A.jcc(Cc::NE, "L_err_decode", "", true);
  A.i(e_ld32(RAX, R12, 4));
  A.i(e_st(R15, OFF_MAIN, RAX));
  A.i(e_ld32(RAX, R12, 8));
  A.i(e_st(R15, OFF_NW, RAX));
  A.i(e_ld32(RAX, R12, 12));
  A.i(e_st(R15, OFF_ND, RAX));

  // Skip & sum nd data items
  A.i(e_movI(RSI, 16), "pos = 16");
  A.i(e_zero(R8), "total_data = 0");
  A.i(e_ld(RCX, R15, OFF_ND), "k = nd");
  A.label("p1_data_loop");
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "p1_data_done");
  A.i(e_lea(RAX, RSI, 4));
  A.i(e_cmp(RAX, R13));
  A.jcc(Cc::A, "L_err_decode", "", true);
  A.i(e_ld_idx(MemW::W32, RDX, R12, RSI, 1, 0), "dlen");
  A.i(e_mov(RSI, RAX));
  A.i(e_mov(RAX, R13));
  A.i(e_alu(AluOp::Sub, RAX, RSI), "rem");
  A.i(e_cmp(RDX, RAX));
  A.jcc(Cc::A, "L_err_decode", "", true);
  A.i(e_alu(AluOp::Add, RSI, RDX));
  A.i(e_alu(AluOp::Add, R8, RDX));
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.jmp("p1_data_loop");

  A.label("p1_data_done");
  A.i(e_st(R15, OFF_TOT_DATA, R8));
  A.i(e_st(R15, OFF_WORDS_FILE_OFF, RSI));

  // Skip & validate nw words, summing total_instrs
  A.i(e_zero(R8), "total_instrs = 0");
  A.i(e_ld(RCX, R15, OFF_NW), "w_rem = nw");
  A.label("p1_word_loop");
  A.i(e_test(RCX, RCX));
  A.jcc(Cc::E, "p1_word_done");
  A.i(e_lea(RAX, RSI, 20));
  A.i(e_cmp(RAX, R13));
  A.jcc(Cc::A, "L_err_decode", "", true);
  A.i(e_ld_idx(MemW::W32, R9, R12, RSI, 1, 12), "nc");
  A.i(e_ld_idx(MemW::W32, R10, R12, RSI, 1, 16), "na");
  A.i(e_alu(AluOp::Add, R8, R9));
  A.i(e_mov(RSI, RAX));
  // Scan nc instructions
  A.label("p1_code_loop");
  A.i(e_test(R9, R9));
  A.jcc(Cc::E, "p1_code_done");
  A.i(e_cmp(RSI, R13));
  A.jcc(Cc::AE, "L_err_decode", "", true);
  A.i(e_ld_idx(MemW::W8, RAX, R12, RSI, 1, 0), "op");
  A.i(e_cmpI(RAX, kNPrims));
  A.jcc(Cc::B, "p1_ilen1");
  A.i(e_cmpI(RAX, 0x80));
  A.jcc(Cc::B, "L_err_decode", "", true);
  A.jcc(Cc::E, "p1_ilen9");
  A.i(e_cmpI(RAX, 0x8c));
  A.jcc(Cc::A, "L_err_decode", "", true);
  A.jcc(Cc::E, "p1_ilen1");
  A.i(e_cmpI(RAX, 0x83));
  A.jcc(Cc::B, "p1_ilen5");
  A.i(e_cmpI(RAX, 0x86));
  A.jcc(Cc::B, "p1_ilen1");
  A.label("p1_ilen5");
  A.i(e_movI(RDX, 5));
  A.jmp("p1_ilen_chk");
  A.label("p1_ilen9");
  A.i(e_movI(RDX, 9));
  A.jmp("p1_ilen_chk");
  A.label("p1_ilen1");
  A.i(e_movI(RDX, 1));
  A.label("p1_ilen_chk");
  A.i(e_mov(R11, R13));
  A.i(e_alu(AluOp::Sub, R11, RSI));
  A.i(e_cmp(RDX, R11));
  A.jcc(Cc::A, "L_err_decode", "", true);
  A.i(e_alu(AluOp::Add, RSI, RDX));
  A.i(e_aluI(AluOp::Sub, R9, 1));
  A.jmp("p1_code_loop");

  A.label("p1_code_done");
  A.i(e_mov(RDX, R10));
  A.i(e_shlI(RDX, 3));
  A.i(e_shlI(R10, 2));
  A.i(e_alu(AluOp::Add, RDX, R10), "12 * na");
  A.i(e_mov(R11, R13));
  A.i(e_alu(AluOp::Sub, R11, RSI));
  A.i(e_cmp(RDX, R11));
  A.jcc(Cc::A, "L_err_decode", "", true);
  A.i(e_alu(AluOp::Add, RSI, RDX));
  A.i(e_aluI(AluOp::Sub, RCX, 1));
  A.jmp("p1_word_loop");

  A.label("p1_word_done");
  A.i(e_cmp(RSI, R13), "no trailing bytes");
  A.jcc(Cc::NE, "L_err_decode", "", true);
  A.i(e_st(R15, OFF_TOT_INSTRS, R8));

  // -------------------------------------------------------------------------
  // Pass 2: Allocate & populate code1 flat table
  // -------------------------------------------------------------------------
  // table_bytes = pages(32 + 16*nd + 64*nw + 16*tot_instrs)
  A.i(e_ld(RAX, R15, OFF_ND));
  A.i(e_shlI(RAX, 4));
  A.i(e_aluI(AluOp::Add, RAX, 32));
  A.i(e_ld(RCX, R15, OFF_NW));
  A.i(e_shlI(RCX, 6));
  A.i(e_alu(AluOp::Add, RAX, RCX));
  A.i(e_ld(RDX, R15, OFF_TOT_INSTRS));
  A.i(e_shlI(RDX, 4));
  A.i(e_alu(AluOp::Add, RAX, RDX));
  A.i(e_aluI(AluOp::Add, RAX, 4095));
  A.i(e_shrI(RAX, 12));
  A.i(e_shlI(RAX, 12));
  A.i(e_st(R15, OFF_TABLE_BYTES, RAX));
  A.i(e_mov(RDI, RAX));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_CODE1, RAX));
  A.i(e_mov(R14, RAX), "r14 = code1");

  // Header
  A.i(e_ld(RAX, R15, OFF_MAIN));
  A.i(e_st(R14, 0, RAX));
  A.i(e_ld(RAX, R15, OFF_NW));
  A.i(e_st(R14, 8, RAX));
  A.i(e_ld(RAX, R15, OFF_ND));
  A.i(e_st(R14, 16, RAX));
  A.i(e_ld(RAX, R15, OFF_TOT_INSTRS));
  A.i(e_st(R14, 24, RAX));

  // Data metadata
  A.i(e_movI(RSI, 16), "file pos");
  A.i(e_zero(R8), "addr = 0");
  A.i(e_zero(RCX), "k = 0");
  A.i(e_ld(R9, R15, OFF_ND));
  A.label("p2_data_loop");
  A.i(e_cmp(RCX, R9));
  A.jcc(Cc::AE, "p2_data_done");
  A.i(e_ld_idx(MemW::W32, RAX, R12, RSI, 1, 0), "dlen");
  A.i(e_aluI(AluOp::Add, RSI, 4));
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 4));
  A.i(e_aluI(AluOp::Add, RDX, 32));
  A.i(e_st_idx(MemW::W64, R14, RDX, 1, 0, R8), "data[k].addr");
  A.i(e_st_idx(MemW::W64, R14, RDX, 1, 8, RAX), "data[k].len");
  A.i(e_alu(AluOp::Add, R8, RAX));
  A.i(e_alu(AluOp::Add, RSI, RAX));
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.jmp("p2_data_loop");

  A.label("p2_data_done");
  // Words & instructions: r13 = cur_code_byte_off in code1
  A.i(e_ld(R13, R15, OFF_ND));
  A.i(e_shlI(R13, 4));
  A.i(e_aluI(AluOp::Add, R13, 32));
  A.i(e_ld(RAX, R15, OFF_NW));
  A.i(e_shlI(RAX, 6));
  A.i(e_alu(AluOp::Add, R13, RAX), "start of instrs in code1");
  A.i(e_mov(R8, R13));
  A.i(e_alu(AluOp::Sub, R8, RAX));
  A.i(e_alu(AluOp::Add, R8, R14), "start of words_meta in code1");
  A.i(e_zero(RCX), "w = 0");

  A.label("p2_word_loop");
  A.i(e_ld(RAX, R15, OFF_NW));
  A.i(e_cmp(RCX, RAX));
  A.jcc(Cc::AE, "p2_word_done", "", true);
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 6));
  A.i(e_alu(AluOp::Add, RDX, R8), "w_ptr = words_meta[w]");
  A.i(e_ld_idx(MemW::W32, RAX, R12, RSI, 1, 0));
  A.i(e_st(RDX, 0, RAX), "ins");
  A.i(e_ld_idx(MemW::W32, RAX, R12, RSI, 1, 4));
  A.i(e_st(RDX, 8, RAX), "outs");
  A.i(e_ld_idx(MemW::W32, RAX, R12, RSI, 1, 8));
  A.i(e_st(RDX, 16, RAX), "locals");
  A.i(e_ld_idx(MemW::W32, R9, R12, RSI, 1, 12), "nc");
  A.i(e_st(RDX, 24, R9), "ncode");
  A.i(e_st(RDX, 32, R13), "code_byte_off");
  A.i(e_ld_idx(MemW::W32, R10, R12, RSI, 1, 16), "na");
  A.i(e_st(RDX, 40, R10), "tmp na");
  A.i(e_aluI(AluOp::Add, RSI, 20));
  // Decode nc instructions into code1 + r13
  A.i(e_movI(R11, 0), "j = 0 (avoids reading syscall-clobbered R11)");
  A.label("p2_code_loop");
  A.i(e_cmp(R11, R9));
  A.jcc(Cc::AE, "p2_code_done", "", true);
  A.i(e_ld_idx(MemW::W8, RAX, R12, RSI, 1, 0), "byte op");
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.i(e_cmpI(RAX, kNPrims));
  A.jcc(Cc::AE, "p2_nonprim");
  A.i(e_zero(RBX), "imm = 0");
  A.jmp("p2_store_ins");

  A.label("p2_nonprim");
  A.i(e_cmpI(RAX, 0x80));
  A.jcc(Cc::NE, "p2_np_other");
  A.i(e_movI(RAX, OP_LIT));
  A.i(e_ld_idx(MemW::W64, RBX, R12, RSI, 1, 0), "u64 imm");
  A.i(e_aluI(AluOp::Add, RSI, 8));
  A.jmp("p2_store_ins");

  A.label("p2_np_other");
  A.i(e_aluI(AluOp::Sub, RAX, 0x80));
  A.i(e_aluI(AluOp::Add, RAX, OP_LIT), "op_id = 57 + (byte - 0x80)");
  A.i(e_cmpI(RAX, OP_TOR));
  A.jcc(Cc::E, "p2_imm0");
  A.i(e_cmpI(RAX, OP_FROMR));
  A.jcc(Cc::E, "p2_imm0");
  A.i(e_cmpI(RAX, OP_RFETCH));
  A.jcc(Cc::E, "p2_imm0");
  A.i(e_cmpI(RAX, OP_RET));
  A.jcc(Cc::E, "p2_imm0");
  A.i(e_ld_idx(MemW::W32, RBX, R12, RSI, 1, 0), "u32 imm");
  A.i(e_aluI(AluOp::Add, RSI, 4));
  A.jmp("p2_store_ins");
  A.label("p2_imm0");
  A.i(e_zero(RBX));

  A.label("p2_store_ins");
  A.i(e_st_idx(MemW::W64, R14, R13, 1, 0, RAX));
  A.i(e_st_idx(MemW::W64, R14, R13, 1, 8, RBX));
  A.i(e_aluI(AluOp::Add, R13, 16));
  A.i(e_aluI(AluOp::Add, R11, 1));
  A.jmp("p2_code_loop");

  A.label("p2_code_done");
  A.i(e_st(RDX, 48, RSI), "tmp ann_file_off");
  A.i(e_mov(RAX, R10));
  A.i(e_shlI(RAX, 3));
  A.i(e_shlI(R10, 2));
  A.i(e_alu(AluOp::Add, RAX, R10));
  A.i(e_alu(AluOp::Add, RSI, RAX), "skip na * 12");
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.jmp("p2_word_loop");

  A.label("p2_word_done");

  // -------------------------------------------------------------------------
  // checkProg
  // -------------------------------------------------------------------------
  A.i(e_ld(RAX, R15, OFF_MAIN));
  A.i(e_ld(RCX, R15, OFF_NW));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::AE, "L_err_check", "main >= nw", true);
  A.i(e_ld(RDX, R15, OFF_ND));
  A.i(e_shlI(RDX, 4));
  A.i(e_aluI(AluOp::Add, RDX, 32));
  A.i(e_shlI(RAX, 6));
  A.i(e_alu(AluOp::Add, RDX, RAX));
  A.i(e_ld_idx(MemW::W64, RAX, R14, RDX, 1, 0), "words[main].ins");
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::NE, "L_err_check", "main.ins != 0", true);

  // Allocate temporary H_buf (initial 256 entries = 4096 B)
  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_HBUF_BASE, RAX));
  A.i(e_movI(RAX, 256));
  A.i(e_st(R15, OFF_HBUF_CAP, RAX));

  A.i(e_zero(RAX));
  A.i(e_st(R15, OFF_CHK_W, RAX));

  A.label("chk_word_loop");
  A.i(e_ld(RCX, R15, OFF_CHK_W));
  A.i(e_ld(RAX, R15, OFF_NW));
  A.i(e_cmp(RCX, RAX));
  A.jcc(Cc::AE, "chk_all_ok", "", true);

  A.i(e_ld(RDX, R15, OFF_ND));
  A.i(e_shlI(RDX, 4));
  A.i(e_aluI(AluOp::Add, RDX, 32));
  A.i(e_shlI(RCX, 6));
  A.i(e_alu(AluOp::Add, RDX, RCX));
  A.i(e_ld(R14, R15, OFF_CODE1));
  A.i(e_alu(AluOp::Add, R14, RDX), "r14 = wd ptr");
  A.i(e_st(R15, OFF_CHK_WBASE, R14));
  A.i(e_ld(R8, R14, 24), "nc");
  A.i(e_test(R8, R8));
  A.jcc(Cc::E, "chk_fail_munmap", "empty word code", true);
  A.i(e_st(R15, OFF_CHK_NC, R8));
  A.i(e_ld(RAX, R14, 40), "na");
  A.i(e_st(R15, OFF_CHK_NA, RAX));
  A.i(e_ld(RAX, R14, 48), "ann_file_off");
  A.i(e_ld(R12, R15, OFF_FILE_BASE));
  A.i(e_alu(AluOp::Add, RAX, R12));
  A.i(e_st(R15, OFF_CHK_ANN, RAX));

  // Ensure H_buf capacity >= nc
  A.label("chk_hbuf_grow");
  A.i(e_ld(RAX, R15, OFF_CHK_NC));
  A.i(e_ld(RCX, R15, OFF_HBUF_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::BE, "chk_hbuf_ok");
  A.i(e_ld(RDI, R15, OFF_HBUF_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_shlI(RSI, 4));
  A.i(e_mov(RDX, RSI));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_HBUF_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_HBUF_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_HBUF_CAP, RCX));
  A.jmp("chk_hbuf_grow");

  A.label("chk_hbuf_ok");
  // Phase A: infer H[0..nc-1]
  A.i(e_ld(R14, R15, OFF_CHK_WBASE));
  A.i(e_ld(RAX, R14, 0), "wd.ins");
  A.i(e_st(R15, OFF_CHK_CURD, RAX));
  A.i(e_zero(RAX));
  A.i(e_st(R15, OFF_CHK_CURA, RAX));
  A.i(e_zero(R13), "i = 0");

  A.label("chk_infer_loop");
  A.i(e_ld(RAX, R15, OFF_CHK_NC));
  A.i(e_cmp(R13, RAX));
  A.jcc(Cc::AE, "chk_infer_done", "", true);
  A.i(e_ld(RDI, R15, OFF_CHK_ANN));
  A.i(e_ld(RSI, R15, OFF_CHK_NA));
  A.i(e_mov(RDX, R13));
  A.call("fn_ann_lookup");
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::NE, "ci_has_ann");
  A.i(e_ld(RCX, R15, OFF_CHK_CURD));
  A.i(e_ld(R8, R15, OFF_CHK_CURA));
  A.label("ci_has_ann");
  A.i(e_st(R15, OFF_CHK_HD, RCX));
  A.i(e_st(R15, OFF_CHK_HA, R8));
  A.i(e_ld(R9, R15, OFF_HBUF_BASE));
  A.i(e_mov(R10, R13));
  A.i(e_shlI(R10, 4));
  A.i(e_st_idx(MemW::W64, R9, R10, 1, 0, RCX));
  A.i(e_st_idx(MemW::W64, R9, R10, 1, 8, R8));
  // hstep on code[i]
  A.i(e_ld(R14, R15, OFF_CHK_WBASE));
  A.i(e_ld(RAX, R14, 32), "code_byte_off");
  A.i(e_alu(AluOp::Add, RAX, R10), "code_byte_off + i*16");
  A.i(e_ld(R11, R15, OFF_CODE1));
  A.i(e_ld_idx(MemW::W64, RDI, R11, RAX, 1, 0), "op");
  A.i(e_ld_idx(MemW::W64, RSI, R11, RAX, 1, 8), "imm");
  A.call("fn_hstep");
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::E, "ci_fallback");
  A.i(e_test(RDX, RDX));
  A.jcc(Cc::E, "ci_fallback");
  A.i(e_st(R15, OFF_CHK_CURD, RCX));
  A.i(e_st(R15, OFF_CHK_CURA, R8));
  A.jmp("ci_next");
  A.label("ci_fallback");
  A.i(e_ld(RDI, R15, OFF_CHK_ANN));
  A.i(e_ld(RSI, R15, OFF_CHK_NA));
  A.i(e_lea(RDX, R13, 1));
  A.call("fn_ann_lookup");
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::NE, "ci_fb_set");
  A.i(e_zero(RCX));
  A.i(e_zero(R8));
  A.label("ci_fb_set");
  A.i(e_st(R15, OFF_CHK_CURD, RCX));
  A.i(e_st(R15, OFF_CHK_CURA, R8));
  A.label("ci_next");
  A.i(e_aluI(AluOp::Add, R13, 1));
  A.jmp("chk_infer_loop");

  A.label("chk_infer_done");
  // Phase B1: H[0] == (wd.ins, 0)
  A.i(e_ld(R9, R15, OFF_HBUF_BASE));
  A.i(e_ld(R14, R15, OFF_CHK_WBASE));
  A.i(e_ld(RAX, R9, 0));
  A.i(e_ld(RCX, R14, 0));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::NE, "chk_fail_munmap", "", true);
  A.i(e_ld(RAX, R9, 8));
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::NE, "chk_fail_munmap", "", true);

  // Phase B2: every instruction i in 0..nc-1
  A.i(e_zero(R13), "i = 0");
  A.label("chk_verify_loop");
  A.i(e_ld(RAX, R15, OFF_CHK_NC));
  A.i(e_cmp(R13, RAX));
  A.jcc(Cc::AE, "chk_verify_anns", "", true);
  A.i(e_ld(R9, R15, OFF_HBUF_BASE));
  A.i(e_mov(R10, R13));
  A.i(e_shlI(R10, 4));
  A.i(e_ld_idx(MemW::W64, RAX, R9, R10, 1, 0));
  A.i(e_st(R15, OFF_CHK_HD, RAX));
  A.i(e_ld_idx(MemW::W64, RAX, R9, R10, 1, 8));
  A.i(e_st(R15, OFF_CHK_HA, RAX));
  A.i(e_ld(R14, R15, OFF_CHK_WBASE));
  A.i(e_ld(RAX, R14, 32));
  A.i(e_alu(AluOp::Add, RAX, R10));
  A.i(e_ld(R11, R15, OFF_CODE1));
  A.i(e_ld_idx(MemW::W64, RDI, R11, RAX, 1, 0));
  A.i(e_ld_idx(MemW::W64, RSI, R11, RAX, 1, 8));
  A.call("fn_hstep");
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::E, "chk_fail_munmap", "", true);
  A.i(e_test(RDX, RDX));
  A.jcc(Cc::E, "cv_next");
  A.i(e_lea(RAX, R13, 1));
  A.i(e_ld(RDX, R15, OFF_CHK_NC));
  A.i(e_cmp(RAX, RDX));
  A.jcc(Cc::AE, "chk_fail_munmap", "falls off end", true);
  A.i(e_ld(R9, R15, OFF_HBUF_BASE));
  A.i(e_shlI(RAX, 4));
  A.i(e_ld_idx(MemW::W64, R10, R9, RAX, 1, 0));
  A.i(e_cmp(R10, RCX));
  A.jcc(Cc::NE, "chk_fail_munmap", "", true);
  A.i(e_ld_idx(MemW::W64, R10, R9, RAX, 1, 8));
  A.i(e_cmp(R10, R8));
  A.jcc(Cc::NE, "chk_fail_munmap", "", true);
  A.label("cv_next");
  A.i(e_aluI(AluOp::Add, R13, 1));
  A.jmp("chk_verify_loop");

  // Phase B3: all annotations (t, d, a) have t < nc && H[t] == (d, a)
  A.label("chk_verify_anns");
  A.i(e_zero(R13), "j = 0");
  A.i(e_ld(R8, R15, OFF_CHK_NA));
  A.i(e_ld(R9, R15, OFF_HBUF_BASE));
  A.i(e_ld(R10, R15, OFF_CHK_NC));
  A.i(e_ld(R11, R15, OFF_CHK_ANN));
  A.label("cva_loop");
  A.i(e_cmp(R13, R8));
  A.jcc(Cc::AE, "chk_word_next");
  A.i(e_mov(RAX, R13));
  A.i(e_shlI(RAX, 3));
  A.i(e_mov(RCX, R13));
  A.i(e_shlI(RCX, 2));
  A.i(e_alu(AluOp::Add, RAX, RCX));
  A.i(e_ld_idx(MemW::W32, RDX, R11, RAX, 1, 0), "t");
  A.i(e_cmp(RDX, R10));
  A.jcc(Cc::AE, "chk_fail_munmap", "", true);
  A.i(e_shlI(RDX, 4));
  A.i(e_ld_idx(MemW::W32, RCX, R11, RAX, 1, 4), "ann.d");
  A.i(e_ld_idx(MemW::W64, RSI, R9, RDX, 1, 0), "H[t].d");
  A.i(e_cmp(RCX, RSI));
  A.jcc(Cc::NE, "chk_fail_munmap", "", true);
  A.i(e_ld_idx(MemW::W32, RCX, R11, RAX, 1, 8), "ann.a");
  A.i(e_ld_idx(MemW::W64, RSI, R9, RDX, 1, 8), "H[t].a");
  A.i(e_cmp(RCX, RSI));
  A.jcc(Cc::NE, "chk_fail_munmap", "", true);
  A.i(e_aluI(AluOp::Add, R13, 1));
  A.jmp("cva_loop");

  A.label("chk_word_next");
  A.i(e_ld(RAX, R15, OFF_CHK_W));
  A.i(e_aluI(AluOp::Add, RAX, 1));
  A.i(e_st(R15, OFF_CHK_W, RAX));
  A.jmp("chk_word_loop", "", true);

  A.label("chk_fail_munmap");
  A.i(e_ld(RDI, R15, OFF_HBUF_BASE));
  A.i(e_ld(RSI, R15, OFF_HBUF_CAP));
  A.i(e_shlI(RSI, 4));
  A.i(e_movI(RAX, 11), "sys_munmap");
  A.i(e_syscall());
  A.jmp("L_err_check", "", true);

  A.label("chk_all_ok");
  A.i(e_ld(RDI, R15, OFF_HBUF_BASE));
  A.i(e_ld(RSI, R15, OFF_HBUF_CAP));
  A.i(e_shlI(RSI, 4));
  A.i(e_movI(RAX, 11), "sys_munmap H_buf");
  A.i(e_syscall());

  // Clear temporary +40/+48 fields in words_meta before triplication so code1..3 match encodeFlatProg
  A.i(e_ld(R14, R15, OFF_CODE1));
  A.i(e_ld(R8, R15, OFF_ND));
  A.i(e_shlI(R8, 4));
  A.i(e_aluI(AluOp::Add, R8, 32));
  A.i(e_ld(R9, R15, OFF_NW));
  A.i(e_zero(RCX));
  A.i(e_zero(RAX));
  A.label("clr_meta_loop");
  A.i(e_cmp(RCX, R9));
  A.jcc(Cc::AE, "clr_meta_done");
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 6));
  A.i(e_alu(AluOp::Add, RDX, R8));
  A.i(e_st_idx(MemW::W64, R14, RDX, 1, 40, RAX));
  A.i(e_st_idx(MemW::W64, R14, RDX, 1, 48, RAX));
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.jmp("clr_meta_loop");
  A.label("clr_meta_done");

  // -------------------------------------------------------------------------
  // Triplicate code1 -> code2, code3
  // -------------------------------------------------------------------------
  auto mk_copy = [&](i32 off_dst) {
    A.i(e_ld(RDI, R15, OFF_TABLE_BYTES));
    A.call("fn_mmap");
    A.i(e_st(R15, off_dst, RAX));
    A.i(e_ld(RSI, R15, OFF_CODE1));
    A.i(e_ld(RCX, R15, OFF_TABLE_BYTES));
    A.i(e_zero(RDX));
    std::string lp = A.fresh("trip_lp"), dn = A.fresh("trip_dn");
    A.label(lp);
    A.i(e_cmp(RDX, RCX));
    A.jcc(Cc::AE, dn);
    A.i(e_ld_idx(MemW::W64, R8, RSI, RDX, 1, 0));
    A.i(e_st_idx(MemW::W64, RAX, RDX, 1, 0, R8));
    A.i(e_aluI(AluOp::Add, RDX, 8));
    A.jmp(lp);
    A.label(dn);
  };
  mk_copy(OFF_CODE2);
  mk_copy(OFF_CODE3);

  // -------------------------------------------------------------------------
  // Read all stdin into inp_buf
  // -------------------------------------------------------------------------
  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_INP_BASE, RAX));
  A.i(e_movI(RCX, kPage));
  A.i(e_st(R15, OFF_CHK_INP_CAP, RCX));
  A.i(e_zero(RCX));
  A.i(e_st(R15, OFF_INP_LEN, RCX));
  A.i(e_st(R15, OFF_IPOS, RCX));

  A.label("rd_inp_loop");
  A.i(e_ld(RAX, R15, OFF_INP_LEN));
  A.i(e_ld(RCX, R15, OFF_CHK_INP_CAP));
  A.i(e_cmp(RAX, RCX));
  A.jcc(Cc::B, "rd_inp_go");
  A.i(e_ld(RDI, R15, OFF_INP_BASE));
  A.i(e_mov(RSI, RCX));
  A.i(e_mov(RDX, RCX));
  A.i(e_shlI(RDX, 1));
  A.call("fn_mremap");
  A.i(e_st(R15, OFF_INP_BASE, RAX));
  A.i(e_ld(RCX, R15, OFF_CHK_INP_CAP));
  A.i(e_shlI(RCX, 1));
  A.i(e_st(R15, OFF_CHK_INP_CAP, RCX));
  A.label("rd_inp_go");
  A.i(e_zero(RDI), "fd 0 = stdin");
  A.i(e_ld(RSI, R15, OFF_INP_BASE));
  A.i(e_ld(RAX, R15, OFF_INP_LEN));
  A.i(e_alu(AluOp::Add, RSI, RAX));
  A.i(e_ld(RDX, R15, OFF_CHK_INP_CAP));
  A.i(e_alu(AluOp::Sub, RDX, RAX));
  A.i(e_zero(RAX), "sys_read");
  A.i(e_syscall());
  A.i(e_test(RAX, RAX));
  A.jcc(Cc::S, "L_exit_nomem", "", true);
  A.jcc(Cc::E, "rd_inp_done");
  A.i(e_ld(RCX, R15, OFF_INP_LEN));
  A.i(e_alu(AluOp::Add, RCX, RAX));
  A.i(e_st(R15, OFF_INP_LEN, RCX));
  A.jmp("rd_inp_loop");
  A.label("rd_inp_done");

  // -------------------------------------------------------------------------
  // Allocate runtime arenas: ds, aux, rs, locs, memw, memp, tabs, out
  // -------------------------------------------------------------------------
  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_DS_BASE, RAX));
  A.i(e_movI(RAX, 256));
  A.i(e_st(R15, OFF_DS_CAP, RAX));

  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_AUX_BASE, RAX));
  A.i(e_movI(RAX, 256));
  A.i(e_st(R15, OFF_AUX_CAP, RAX));

  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_RS_BASE, RAX));
  A.i(e_movI(RAX, 128));
  A.i(e_st(R15, OFF_RS_CAP, RAX));

  // Initial locs arena: ensure >= main.locals (at least 256)
  A.i(e_ld(R8, R15, OFF_MAIN));
  A.i(e_ld(R9, R15, OFF_ND));
  A.i(e_shlI(R9, 4));
  A.i(e_aluI(AluOp::Add, R9, 32));
  A.i(e_shlI(R8, 6));
  A.i(e_alu(AluOp::Add, R8, R9));
  A.i(e_ld(R14, R15, OFF_CODE1));
  A.i(e_ld_idx(MemW::W64, R12, R14, R8, 1, 16), "main.locals");
  A.i(e_movI(RCX, 256));
  A.label("init_locs_cap");
  A.i(e_cmp(R12, RCX));
  A.jcc(Cc::BE, "init_locs_alloc");
  A.i(e_shlI(RCX, 1));
  A.jmp("init_locs_cap");
  A.label("init_locs_alloc");
  A.i(e_st(R15, OFF_LOCS_CAP, RCX));
  A.i(e_mov(RDI, RCX));
  A.i(e_shlI(RDI, 4));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_LOCS_BASE, RAX));
  A.i(e_st(R15, OFF_LOCS_LEN, R12));

  // Push initial frame: {word=main, pc=0, locs_off=0, nlocs=main.locals}
  A.i(e_ld(RAX, R15, OFF_RS_BASE));
  A.i(e_ld(RCX, R15, OFF_MAIN));
  A.i(e_st(RAX, 0, RCX));
  A.i(e_zero(RDX));
  A.i(e_st(RAX, 8, RDX));
  A.i(e_st(RAX, 16, RDX));
  A.i(e_st(RAX, 24, R12));
  A.i(e_movI(RCX, 1));
  A.i(e_st(R15, OFF_RS_LEN, RCX));

  // memw & memp: max(msize, 1)
  A.i(e_ld(RDI, R15, OFF_MSIZE));
  A.i(e_movI(RCX, 1));
  A.i(e_test(RDI, RDI));
  A.i(e_cmovcc(Cc::E, RDI, RCX));
  A.i(e_mov(R12, RDI), "bytes");
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_MEMW_BASE, RAX));
  A.i(e_mov(RDI, R12));
  A.i(e_shlI(RDI, 3));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_MEMP_BASE, RAX));

  // Populate memw[0 .. min(tot_data, msize)) and set hp = tot_data
  A.i(e_ld(R12, R15, OFF_FILE_BASE));
  A.i(e_ld(R8, R15, OFF_MEMW_BASE));
  A.i(e_ld(R9, R15, OFF_MSIZE));
  A.i(e_movI(RSI, 16), "file pos");
  A.i(e_zero(R10), "a = 0");
  A.i(e_zero(RCX), "k = 0");
  A.i(e_ld(R11, R15, OFF_ND));
  A.label("init_data_loop");
  A.i(e_cmp(RCX, R11));
  A.jcc(Cc::AE, "init_data_done");
  A.i(e_ld_idx(MemW::W32, R13, R12, RSI, 1, 0), "dlen");
  A.i(e_aluI(AluOp::Add, RSI, 4));
  A.i(e_zero(RDX), "i = 0");
  A.label("init_dbyte_loop");
  A.i(e_cmp(RDX, R13));
  A.jcc(Cc::AE, "init_dbyte_done");
  A.i(e_cmp(R10, R9));
  A.jcc(Cc::AE, "init_dbyte_skip");
  A.i(e_ld_idx(MemW::W8, RAX, R12, RSI, 1, 0));
  A.i(e_st_idx(MemW::W8, R8, R10, 1, 0, RAX));
  A.label("init_dbyte_skip");
  A.i(e_aluI(AluOp::Add, RSI, 1));
  A.i(e_aluI(AluOp::Add, R10, 1));
  A.i(e_aluI(AluOp::Add, RDX, 1));
  A.jmp("init_dbyte_loop");
  A.label("init_dbyte_done");
  A.i(e_aluI(AluOp::Add, RCX, 1));
  A.jmp("init_data_loop");
  A.label("init_data_done");
  A.i(e_st(R15, OFF_HP, R10));

  // Unmap file_buf now that decoding, checking, triplicating, and data load are complete
  A.i(e_ld(RDI, R15, OFF_FILE_BASE));
  A.i(e_ld(RSI, R15, OFF_FILE_CAP));
  A.i(e_movI(RAX, 11), "sys_munmap file_buf");
  A.i(e_syscall());

  // Tabs & out
  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_TAB_HEADS, RAX));
  A.i(e_movI(RAX, 512));
  A.i(e_st(R15, OFF_TABS_CAP, RAX));

  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_TAB_NODES, RAX));
  A.i(e_movI(RAX, 128));
  A.i(e_st(R15, OFF_TAB_NODES_CAP, RAX));

  A.i(e_movI(RDI, kPage));
  A.call("fn_mmap");
  A.i(e_st(R15, OFF_OUT_BASE, RAX));
  A.i(e_movI(RAX, kPage));
  A.i(e_st(R15, OFF_OUT_CAP, RAX));

  A.jmp("L_vm_ready", "", true);

  // Startup error exits
  A.label("L_err_open");
  A.mov_label(RSI, "L_str_prog_pre");
  A.i(e_movI(RDX, static_cast<u64>(kProgPreStr.size())));
  A.call("fn_write_stderr");
  A.i(e_ld(RDI, R15, OFF_ARGV1));
  A.call("fn_write_stderr_cstr");
  A.mov_label(RSI, "L_str_open_suf");
  A.i(e_movI(RDX, static_cast<u64>(kOpenSufStr.size())));
  A.call("fn_write_stderr");
  A.i(e_movI(RDI, 1));
  A.i(e_movI(RAX, 60));
  A.i(e_syscall());

  A.label("L_err_decode");
  A.mov_label(RSI, "L_str_prog_pre");
  A.i(e_movI(RDX, static_cast<u64>(kProgPreStr.size())));
  A.call("fn_write_stderr");
  A.i(e_ld(RDI, R15, OFF_ARGV1));
  A.call("fn_write_stderr_cstr");
  A.mov_label(RSI, "L_str_dec_suf");
  A.i(e_movI(RDX, static_cast<u64>(kDecSufStr.size())));
  A.call("fn_write_stderr");
  A.i(e_movI(RDI, 1));
  A.i(e_movI(RAX, 60));
  A.i(e_syscall());

  A.label("L_err_check");
  A.mov_label(RSI, "L_str_prog_pre");
  A.i(e_movI(RDX, static_cast<u64>(kProgPreStr.size())));
  A.call("fn_write_stderr");
  A.i(e_ld(RDI, R15, OFF_ARGV1));
  A.call("fn_write_stderr_cstr");
  A.mov_label(RSI, "L_str_chk_suf");
  A.i(e_movI(RDX, static_cast<u64>(kChkSufStr.size())));
  A.call("fn_write_stderr");
  A.i(e_movI(RDI, 1));
  A.i(e_movI(RAX, 60));
  A.i(e_syscall());
}

void emit_rodata(Asm& A) {
  A.ro_label("L_rodata");
  A.ro_label("L_prim_eff_in");
  A.ro_bytes(std::vector<u8>(kEffIn.begin(), kEffIn.end()), "Prim.eff.1");
  A.ro_label("L_prim_eff_out");
  A.ro_bytes(std::vector<u8>(kEffOut.begin(), kEffOut.end()), "Prim.eff.2");

  // Main dispatch table (70 entries)
  A.ro_label("L_jtab");
  for (int i = 0; i < kNPrims; ++i) A.ro_qword_label("L_prim_prologue");
  const std::array<const char*, 13> op_labs = {
      "O_lit", "O_str", "O_pick", "O_tor", "O_fromr", "O_rfetch", "O_locget",
      "O_locset", "O_jmp", "O_jz", "O_jnz", "O_call", "O_ret",
  };
  for (const char* l : op_labs) A.ro_qword_label(l);

  // Primitive jump table (57 entries)
  A.ro_label("L_prim_jtab");
  const std::array<const char*, 57> prim_labs = {
      "P_dup", "P_drop", "P_swap", "P_over", "P_rot", "P_nip", "P_tuck",
      "P_add", "P_sub", "P_mul", "P_divu", "P_modu", "P_divs", "P_mods", "P_neg",
      "P_band", "P_bor", "P_bxor", "P_bnot", "P_shl", "P_shr", "P_sar",
      "P_eq", "P_ne", "P_ltu", "P_lts", "P_leu", "P_les",
      "P_ld8", "P_st8", "P_ld16", "P_st16", "P_ld32", "P_st32", "P_ld64", "P_st64",
      "P_copy", "P_fill", "P_mcmp",
      "P_alloc", "P_mark", "P_release",
      "P_streq", "P_strhash",
      "P_tnew", "P_tput", "P_tget",
      "P_getb", "P_putb", "P_putn", "P_exit",
      "P_zero_mk1", "P_zero_mk1", "P_zero_mk1",
      "P_provget", "P_confget", "P_demand",
  };
  for (const char* l : prim_labs) A.ro_qword_label(l);

  // Error strings
  A.ro_label("L_str_usage");
  A.ro_str(kUsageStr);
  A.ro_label("L_str_prog_pre");
  A.ro_str(kProgPreStr);
  A.ro_label("L_str_open_suf");
  A.ro_str(kOpenSufStr);
  A.ro_label("L_str_dec_suf");
  A.ro_str(kDecSufStr);
  A.ro_label("L_str_chk_suf");
  A.ro_str(kChkSufStr);
  A.ro_label("L_str_nomem");
  A.ro_str(kNomemStr);
  A.ro_label("L_str_run_failed");
  A.ro_str(kRunFailPreStr);
  A.ro_label("L_str_nl");
  A.ro_str(kNlStr);

  constexpr std::array<const char*, 10> err_names = {
      "underflow", "aux-underflow", "bad-local", "bad-jump", "bad-word",
      "no-frame",  "bounds",        "div-zero",  "bad-table", "demand",
  };
  for (int i = 0; i < 10; ++i) {
    A.ro_label("L_err_s_" + std::to_string(i));
    A.ro_str(err_names[static_cast<std::size_t>(i)]);
  }
  A.ro_label("L_err_ptrs");
  for (int i = 0; i < 10; ++i) A.ro_qword_label("L_err_s_" + std::to_string(i));
  A.ro_label("L_err_lens");
  for (int i = 0; i < 10; ++i) {
    std::vector<u8> le;
    push_le(le, std::string_view(err_names[static_cast<std::size_t>(i)]).size(), 8);
    A.ro_bytes(std::move(le));
  }
}

Asm gen_program() {
  Asm A;
  A.label("L_code_start");
  emit_vm_core(A);
  emit_helpers(A);
  emit_checker_helpers(A);
  emit_boot(A);
  A.label("L_code_end");
  emit_rodata(A);
  return A;
}

// ---------------------------------------------------------------------------
// ELF64 Header & Output Generation
// ---------------------------------------------------------------------------

std::vector<u8> elf_header(u64 entry, u64 filesz) {
  std::vector<u8> h = {0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  push_le(h, 2, 2);      // ET_EXEC
  push_le(h, 0x3e, 2);   // EM_X86_64
  push_le(h, 1, 4);      // EV_CURRENT
  push_le(h, entry, 8);  // e_entry
  push_le(h, 0x40, 8);   // e_phoff
  push_le(h, 0, 8);
  push_le(h, 0, 4);
  push_le(h, 64, 2);
  push_le(h, 56, 2);
  push_le(h, 2, 2);      // e_phnum = 2
  push_le(h, 0, 2);
  push_le(h, 0, 2);
  push_le(h, 0, 2);
  // PT_LOAD 0: R+X code + rodata at 0x400000
  push_le(h, 1, 4);
  push_le(h, 5, 4);
  push_le(h, 0, 8);
  push_le(h, kBase, 8);
  push_le(h, kBase, 8);
  push_le(h, filesz, 8);
  push_le(h, filesz, 8);
  push_le(h, 0x1000, 8);
  // PT_LOAD 1: R+W VmCtx BSS at 0x600000
  push_le(h, 1, 4);
  push_le(h, 6, 4);
  push_le(h, 0, 8);
  push_le(h, kCtxBase, 8);
  push_le(h, kCtxBase, 8);
  push_le(h, 0, 8);
  push_le(h, kCtxSize, 8);
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
    {0x00, 4, "e_ident: magic 0x7f E L F"},
    {0x04, 1, "ELFCLASS64"},
    {0x05, 1, "little endian"},
    {0x06, 1, "EI_VERSION = 1"},
    {0x07, 1, "System V"},
    {0x08, 8, "padding"},
    {0x10, 2, "e_type = ET_EXEC"},
    {0x12, 2, "e_machine = EM_X86_64"},
    {0x14, 4, "e_version = 1"},
    {0x18, 8, "e_entry = L_start"},
    {0x20, 8, "e_phoff = 0x40"},
    {0x28, 8, "e_shoff = 0"},
    {0x30, 4, "e_flags = 0"},
    {0x34, 2, "e_ehsize = 64"},
    {0x36, 2, "e_phentsize = 56"},
    {0x38, 2, "e_phnum = 2"},
    {0x3a, 2, "e_shentsize = 0"},
    {0x3c, 2, "e_shnum = 0"},
    {0x3e, 2, "e_shstrndx = 0"},
    {0x40, 4, "PT_LOAD 0 (code + rodata)"},
    {0x44, 4, "PF_R | PF_X"},
    {0x48, 8, "p_offset = 0"},
    {0x50, 8, "p_vaddr = 0x400000"},
    {0x58, 8, "p_paddr = 0x400000"},
    {0x60, 8, "p_filesz = whole file"},
    {0x68, 8, "p_memsz = whole file"},
    {0x70, 8, "p_align = 0x1000"},
    {0x78, 4, "PT_LOAD 1 (VmCtx BSS)"},
    {0x7c, 4, "PF_R | PF_W"},
    {0x80, 8, "p_offset = 0"},
    {0x88, 8, "p_vaddr = 0x600000"},
    {0x90, 8, "p_paddr = 0x600000"},
    {0x98, 8, "p_filesz = 0"},
    {0xa0, 8, "p_memsz = 0x1000"},
    {0xa8, 8, "p_align = 0x1000"},
});

int verify_format_a(std::string_view hex_text, const std::vector<u8>& expected) {
  std::vector<u8> decoded;
  unsigned acc = 1;
  bool comment = false;
  for (unsigned char c : hex_text) {
    if (comment) {
      if (c == '\n') comment = false;
      continue;
    }
    if (c == '#') {
      comment = true;
      continue;
    }
    if (c == ' ' || c == '\n' || c == '\t') continue;
    unsigned v = 0;
    if (c >= '0' && c <= '9') v = c - '0';
    else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
    else return 1;
    acc = acc * 16 + v;
    if (acc >= 0x100) {
      decoded.push_back(static_cast<u8>(acc & 0xff));
      acc = 1;
    }
  }
  if (acc != 1 || decoded != expected) return 1;
  return 0;
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
  return name.rfind("L_", 0) == 0 || name.rfind("P_", 0) == 0 || name.rfind("O_", 0) == 0 ||
         name.rfind("fn_", 0) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path outdir = (argc > 1) ? argv[1] : ".";
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
  require(kBase + kHdrLen + code.size() == res.code_end, "code_end mismatch");

  u64 filesz = kHdrLen + code.size() + res.rodata.size();
  std::vector<u8> image = elf_header(labels.at("L_start"), filesz);
  image.insert(image.end(), code.begin(), code.end());
  image.insert(image.end(), res.rodata.begin(), res.rodata.end());

  // Self-check every emitted instruction against our C++ transcription of X86Vm.decode!
  struct TableEntry {
    u64 addr;
    Ins ins;
    std::size_t len;
  };
  std::vector<TableEntry> table;
  for (const auto& e : out) {
    if (e.ins.kind == Ins::Kind::Voter) {
      std::size_t off = 0;
      while (off < e.bytes.size()) {
        ImageSlice sl{image, static_cast<std::size_t>(e.addr - kBase) + off};
        auto d = decode_vm_unified(sl, e.addr + off);
        require(d.has_value(), "failed to decode voter instruction");
        table.push_back(TableEntry{e.addr + off, d->ins, d->len});
        off += d->len;
      }
      continue;
    }
    ImageSlice sl{image, static_cast<std::size_t>(e.addr - kBase)};
    auto d = decode_vm_unified(sl, e.addr);
    require(d.has_value() && d->ins == e.ins && d->len == e.bytes.size(),
            "X86Vm.decode mismatch at " + hex_0x(e.addr) + " (" + ins_str(e.ins) + ")");
    table.push_back(TableEntry{e.addr, e.ins, e.bytes.size()});
  }

  // Verify L_exit3 == L_vm_vote + 0x1a and L_vm_dispatch == L_vm_vote + 0x26
  require(labels.at("L_exit3") == labels.at("L_vm_vote") + 0x1a, "L_exit3 offset mismatch");
  require(labels.at("L_vm_dispatch") == labels.at("L_vm_vote") + 0x26, "L_vm_dispatch offset mismatch");

  // Physical hygiene: 0f 85 never appears anywhere in the binary, and 0f 05 only at Syscalls
  std::set<u64> sys_at;
  for (const auto& te : table) {
    if (te.ins.kind == Ins::Kind::Syscall) sys_at.insert(te.addr);
  }
  for (std::size_t j = 0; j + 1 < image.size(); ++j) {
    if (image[j] == 0x0f && image[j + 1] == 0x05) {
      require(sys_at.contains(kBase + j), "unexpected 0f 05 at " + hex_0x(kBase + j));
    }
    require(!(image[j] == 0x0f && image[j + 1] == 0x85), "forbidden 0f 85 at " + hex_0x(kBase + j));
  }

  std::filesystem::create_directories(outdir);
  write_bytes_file(outdir / "sporos_vm", image);
  std::filesystem::permissions(
      outdir / "sporos_vm",
      std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
          std::filesystem::perms::owner_exec | std::filesystem::perms::group_read |
          std::filesystem::perms::group_exec | std::filesystem::perms::others_read |
          std::filesystem::perms::others_exec,
      std::filesystem::perm_options::replace);

  std::map<u64, std::vector<std::string>> lab_at;
  for (const auto& [k, v] : label_order) lab_at[v].push_back(k);

  // Format A listing (`sporos_vm.hex`)
  {
    std::string hex_out;
    hex_out += "# sporos_vm: post-admission native x86-64 Sporos bytecode VM (`Sporos.decode` + `Sporos.checkProg` + `SporosProvLen.lstep`).\n";
    hex_out += "# Root A framing: Root A is MUSE's hand-written checker (`bootstrap/roots/sporos/sporos-stage1.hex`);\n";
    hex_out += "# `spr1_check` is an independent second check (not Root A or Root A's trust base).\n";
    hex_out += "# Canonical 38-byte TMR voter at L_vm_vote (" + hex_0x(labels.at("L_vm_vote")) +
               ") -> L_vm_dispatch (" + hex_0x(labels.at("L_vm_dispatch")) + "), exit 3 on 3-way split.\n";
    hex_out += "# Total " + std::to_string(filesz) + " bytes, entry " + hex_0x(labels.at("L_start")) + ".\n";
    hex_out += "\n# ---------------------------------------------------------------- ELF header\n";
    for (const auto& hc : kHdrComments) {
      if (hc.off == 0x40) {
        hex_out += "\n# ----------------------------------------------------------- program headers\n";
      }
      std::vector<u8> slice(image.begin() + static_cast<std::ptrdiff_t>(hc.off),
                            image.begin() + static_cast<std::ptrdiff_t>(hc.off + hc.len));
      hex_out += pad_right(hex_bytes(slice), 32) + "# " + raw_hex(hc.off, 3) + "  " + hc.comment + "\n";
    }
    hex_out += "\n# --------------------------------------------------------------------- code\n";
    for (const auto& e : out) {
      auto lit = lab_at.find(e.addr);
      if (lit != lab_at.end()) {
        for (const auto& l : lit->second) hex_out += "\n# " + l + ":\n";
      }
      if (e.ins.kind == Ins::Kind::Voter) {
        hex_out += "# " + e.comment + "\n";
        std::size_t off = 0;
        while (off < e.bytes.size()) {
          ImageSlice sl{image, static_cast<std::size_t>(e.addr - kBase) + off};
          auto d = decode_vm_unified(sl, e.addr + off);
          require(d.has_value(), "voter decode failed");
          std::vector<u8> seg(e.bytes.begin() + static_cast<std::ptrdiff_t>(off),
                              e.bytes.begin() + static_cast<std::ptrdiff_t>(off + d->len));
          hex_out += pad_right(hex_bytes(seg), 32) + "# " +
                     raw_hex((e.addr - kBase) + off, 3) + "  " + ins_str(d->ins) + "\n";
          off += d->len;
        }
        continue;
      }
      hex_out += pad_right(hex_bytes(e.bytes), 32) + "# " + raw_hex(e.addr - kBase, 3) + "  " +
                 ins_str(e.ins) + (e.comment.empty() ? "" : ("   " + e.comment)) + "\n";
    }
    hex_out += "\n# ------------------------------------------------------------------- rodata\n";
    for (const auto& [ra, chunk] : res.ro_chunks) {
      auto lit = res.ro_labels.find(ra);
      if (lit != res.ro_labels.end()) {
        for (const auto& l : lit->second) hex_out += "\n# " + l + ":\n";
      }
      for (std::size_t j = 0; j < chunk.size(); j += 16) {
        std::size_t end = std::min(j + 16, chunk.size());
        std::vector<u8> seg(chunk.begin() + static_cast<std::ptrdiff_t>(j),
                            chunk.begin() + static_cast<std::ptrdiff_t>(end));
        hex_out += pad_right(hex_bytes(seg), 52) + "# " + raw_hex((ra - kBase) + j, 4) + "\n";
      }
    }
    require(verify_format_a(hex_out, image) == 0, "Format A listing failed internal seed-a verification");
    write_text_file(outdir / "sporos_vm.hex", hex_out);
  }

  // Lean data module (`SporosVmBinData.lean`)
  {
    auto fmt_byte_list = [](const std::vector<u8>& bs, std::size_t beg, std::size_t end) {
      std::string out_s = "[";
      for (std::size_t i = beg; i < end; ++i) {
        if (i > beg) out_s += ", ";
        out_s += std::to_string(bs[i]);
      }
      out_s += "]";
      return out_s;
    };

    std::string lean_out;
    lean_out += "/-\n";
    lean_out += "Generated by bootstrap/model/sporos_vm_gen.cpp; do not edit.\n";
    lean_out += "Root A is MUSE's hand-written checker (`bootstrap/roots/sporos/sporos-stage1.hex`);\n";
    lean_out += "`spr1_check` is an independent second check (not Root A or Root A's trust base).\n";
    lean_out += "`vmBytes` matches `bootstrap/model/sporos_vm` byte for byte, and `vmTable` lists\n";
    lean_out += "every decoded instruction `(addr, X86Vm.Ins, len)` across `[L_code_start .. L_code_end)`.\n";
    lean_out += "-/\n";
    lean_out += "import Bootstrap.X86Vm\n\n";
    lean_out += "namespace SporosVmBin\n\n";
    lean_out += "open X86Vm\n\n";

    std::vector<u8> hdr_bytes(image.begin(), image.begin() + static_cast<std::ptrdiff_t>(kHdrLen));
    lean_out += "def hdrBytes : List Nat := " + fmt_byte_list(hdr_bytes, 0, hdr_bytes.size()) + "\n";

    auto emit_chunks = [&](std::string_view prefix, std::string_view total_name,
                           const std::vector<u8>& bs) {
      std::vector<std::string> names;
      for (std::size_t i = 0, j = 0; i < bs.size(); i += 256, ++j) {
        std::size_t end = std::min(i + 256, bs.size());
        std::string nm = std::string(prefix) + "_" + std::to_string(j);
        lean_out += "def " + nm + " : List Nat := " + fmt_byte_list(bs, i, end) + "\n";
        names.push_back(std::move(nm));
      }
      lean_out += "def " + std::string(total_name) + " : List Nat :=\n  " +
                  join_strings(names, " ++\n  ") + "\n\n";
    };
    emit_chunks("code", "codeBytes", code);
    emit_chunks("ro", "rodataBytes", res.rodata);
    lean_out += "def vmBytes : List Nat := hdrBytes ++ codeBytes ++ rodataBytes\n\n";

    std::vector<std::string> tbl_chunks;
    std::vector<u64> tbl_chunk_starts;
    for (std::size_t i = 0, j = 0; i < table.size(); i += 64, ++j) {
      std::size_t end = std::min(i + 64, table.size());
      std::string nm = "vmTable_" + std::to_string(j);
      lean_out += "def " + nm + " : List (Nat × Ins × Nat) := [\n";
      for (std::size_t k = i; k < end; ++k) {
        if (k > i) lean_out += ",\n";
        lean_out += "  (" + std::to_string(table[k].addr) + ", " + lean_ins(table[k].ins) + ", " +
                    std::to_string(table[k].len) + ")";
      }
      lean_out += "]\n";
      tbl_chunk_starts.push_back(table[i].addr);
      tbl_chunks.push_back(std::move(nm));
    }
    lean_out += "def vmTableChunks : List (Nat × List (Nat × Ins × Nat)) := [\n";
    for (std::size_t j = 0; j < tbl_chunks.size(); ++j) {
      if (j > 0) lean_out += ",\n";
      lean_out += "  (" + std::to_string(tbl_chunk_starts[j]) + ", " + tbl_chunks[j] + ")";
    }
    lean_out += "]\n\n";
    lean_out += "def vmTable : List (Nat × Ins × Nat) :=\n  " + join_strings(tbl_chunks, " ++\n  ") + "\n\n";

    std::vector<std::pair<std::string, u64>> sorted_labels = label_order;
    std::stable_sort(sorted_labels.begin(), sorted_labels.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });
    lean_out += "/-- Named addresses in `sporos_vm`. -/\n";
    for (const auto& [k, v] : sorted_labels) {
      if (keep_lean_label(k)) {
        lean_out += "def " + k + " : Nat := " + hex_0x(v) + "\n";
      }
    }
    lean_out += "\nend SporosVmBin\n";
    write_text_file(outdir / "SporosVmBinData.lean", lean_out);
  }

  std::printf("sporos_vm_gen: size=%llu bytes, instrs=%zu, code=%zu, rodata=%zu, L_start=%s, L_vm_vote=%s, L_vm_dispatch=%s\n",
              static_cast<unsigned long long>(filesz), table.size(), code.size(), res.rodata.size(),
              hex_0x(labels.at("L_start")).c_str(), hex_0x(labels.at("L_vm_vote")).c_str(),
              hex_0x(labels.at("L_vm_dispatch")).c_str());
  return 0;
}
